#!/usr/bin/env python3
"""Canary for the clippy determinism gates (G1 disallowed transcendentals and
G10 wall clock / OS entropy): the lists in clippy.toml only protect anything if
clippy actually flags a call to each entry.

A throwaway crate is built in a temporary directory with an exact copy of the
repository's clippy.toml, rust-toolchain.toml and `[lints.*]` tables. It holds
one library function per planted row of scripts/clippy_canary_calls.tsv, each
planting one banned call, plus one clean control function. `cargo clippy --lib`
runs with fixed flags (`-W clippy::pedantic -D warnings`), whatever flags the
repository's own clippy job uses (a disallowed-method lint fires under any of
them).

The data file is generated from clippy.toml by
scripts/gen_clippy_canary_calls.py. A row is either planted (3 columns: key,
path, Rust statement) or skipped (4 columns: key, path, `skip`, reason); a
skipped row must give a non-empty reason and is not planted. A planted row whose
path starts with an external crate makes the throwaway crate depend on that
crate at the exact version locked in the repository's Cargo.lock (the highest
one if several are locked).

The script fails unless

  (i)   the rows and the clippy.toml entries are equal: every entry of
        disallowed-methods / disallowed-types has exactly one row (planted or
        skipped) under the same key, and every row names such an entry,
  (ii)  every planted row with an external crate finds that crate in Cargo.lock,
  (iii) clippy exits non-zero,
  (iv)  every planted row's function gets exactly one `disallowed_*` diagnostic
        naming that row's path, and no other `disallowed_*` diagnostic appears,
        with one allowance: calling an inherent method names its type, so a row
        whose path starts with a banned type (`RandomState::new` with
        `RandomState` in disallowed-types) also gets that type's diagnostic. At
        most one such extra is accepted per function, and only as
        `clippy::disallowed_types` naming a disallowed-types entry of
        clippy.toml that is a `::`-segment prefix of the row's path,
  (v)   something was compared: the data file has at least one planted row and
        clippy.toml has at least one entry.

The control function must get no `disallowed_*` diagnostic.

This script carries no repository-specific logic beyond the file names below.

usage: scripts/clippy_canary.py [--calls FILE] [--root DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALLS = ROOT / "scripts" / "clippy_canary_calls.tsv"
KEYS = ("disallowed-methods", "disallowed-types")
LINTS = {"clippy::disallowed_methods", "clippy::disallowed_types"}
# fixed flags, independent of the repository's own clippy job (see the module doc)
CLIPPY_FLAGS = ["-W", "clippy::pedantic", "-D", "warnings"]
CONTROL = "canary_control"
SKIP = "skip"
# first path segments that need no dependency in the throwaway crate
BUILTIN_ROOTS = frozenset(
    ["std", "core", "alloc", "f32", "f64", "bool", "char", "str", "isize", "usize"]
    + [f"{s}{n}" for s in "iu" for n in (8, 16, 32, 64, 128)])


@dataclass(frozen=True)
class Row:
    line: int
    key: str
    path: str
    stmt: str | None  # None for a skipped row
    reason: str | None = None

    @property
    def planted(self) -> bool:
        return self.stmt is not None


def norm(crate: str) -> str:
    """crate names compare with '-' and '_' equivalent"""
    return crate.replace("-", "_")


def first_segment(path: str) -> str:
    return path.lstrip(":").split("::", 1)[0]


def parse_rows(text: str) -> tuple[list[Row], list[str]]:
    """Rows of the data file; `#` lines and blank lines are skipped."""
    rows: list[Row] = []
    problems: list[str] = []
    for n, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cols = [c.strip() for c in raw.split("\t")]
        if len(cols) == 4 and cols[2] == SKIP:
            key, path, _, reason = cols
            if not key or not path:
                problems.append(f"data line {n}: a skip row needs a key and a path, got {raw!r}")
                continue
            if not reason:
                problems.append(f"data line {n}: skip row for `{path}` has an empty reason")
                continue
            row = Row(n, key, path, None, reason)
        elif len(cols) == 3 and all(cols):
            if cols[2] == SKIP:
                problems.append(f"data line {n}: skip row for `{cols[1]}` has no reason column")
                continue
            row = Row(n, *cols)
        else:
            problems.append(f"data line {n}: expected 3 tab-separated columns (planted) or "
                            f"4 with `skip` and a reason, got {raw!r}")
            continue
        if row.key not in KEYS:
            problems.append(f"data line {n}: unknown key {row.key!r} (expected one of {', '.join(KEYS)})")
            continue
        rows.append(row)
    return rows, problems


def clippy_toml_entries(text: str) -> dict[str, list[str]]:
    """Paths under each disallowed-methods / disallowed-types array of clippy.toml,
    in file order. An entry is a table with `path` or a bare string."""
    data = tomllib.loads(text)
    found: dict[str, list[str]] = {}
    for key in KEYS:
        found[key] = []
        for entry in data.get(key, []):
            path = entry.get("path") if isinstance(entry, dict) else entry
            if isinstance(path, str):
                found[key].append(path)
    return found


def _version_key(v: str) -> tuple:
    core, _, pre = v.partition("-")
    nums = tuple(int(x) if x.isdigit() else 0 for x in core.split("+", 1)[0].split("."))
    return nums, pre == "", pre


def lock_packages(text: str) -> dict[str, list[tuple[str, str, str | None]]]:
    """Cargo.lock packages by normalised name: (name, version, source) each."""
    out: dict[str, list[tuple[str, str, str | None]]] = {}
    for pkg in tomllib.loads(text).get("package", []):
        out.setdefault(norm(pkg["name"]), []).append((pkg["name"], pkg["version"], pkg.get("source")))
    return out


def registry_version(lock: dict, crate: str) -> tuple[str, str, list[str]] | None:
    """(name, highest locked registry version, all locked versions) or None."""
    found = [(n, v) for n, v, s in lock.get(norm(crate), []) if s and s.startswith("registry+")]
    if not found:
        return None
    found.sort(key=lambda nv: _version_key(nv[1]))
    return found[-1][0], found[-1][1], [v for _, v in found]


def dependencies(rows: list[Row], lock: dict) -> tuple[str, list[str]]:
    """The `[dependencies]` table for planted rows that name an external crate."""
    lines: list[str] = []
    problems: list[str] = []
    seen: set[str] = set()
    for row in rows:
        crate = first_segment(row.path)
        if not row.planted or crate in BUILTIN_ROOTS or norm(crate) in seen:
            continue
        hit = registry_version(lock, crate)
        if hit is None:
            problems.append(f"data line {row.line}: `{row.path}` is planted but `{crate}` is not a "
                            "registry crate in Cargo.lock, so the canary crate cannot depend on it")
            continue
        seen.add(norm(crate))
        name, version, versions = hit
        if len(versions) > 1:
            lines.append(f"# {name}: Cargo.lock has {', '.join(versions)}; the highest is used")
        lines.append(f'{name} = "={version}"')
    return ("[dependencies]\n" + "\n".join(lines) + "\n\n") if lines else "", problems


def lints_tables(cargo_toml: str) -> str:
    """The `[lints]` / `[lints.*]` tables of Cargo.toml, verbatim."""
    out: list[str] = []
    keep = False
    for raw in cargo_toml.splitlines():
        s = raw.strip()
        if s.startswith("["):
            keep = s == "[lints]" or s.startswith("[lints.")
        if keep:
            out.append(raw)
    return "\n".join(out) + ("\n" if out else "")


def manifest(edition: str, deps: str, lints: str) -> str:
    return ("[package]\nname = \"clippy-canary\"\nversion = \"0.0.0\"\n"
            f"edition = \"{edition}\"\npublish = false\n\n"
            "[lib]\npath = \"src/lib.rs\"\n\n[workspace]\n\n" + deps + lints)


def render_lib(rows: list[Row]) -> tuple[str, dict[str, tuple[int, int]]]:
    """lib.rs text and the 1-based line range of each function by name; one
    function per planted row (`canary_<index into rows>`)."""
    lines = ["//! clippy canary (generated, throwaway)"]
    ranges: dict[str, tuple[int, int]] = {}

    def add(name: str, body: str) -> None:
        start = len(lines) + 1
        lines.extend([f"pub fn {name}() {{", f"    {body}", "}"])
        ranges[name] = (start, len(lines))

    for i, row in enumerate(rows):
        if row.planted:
            add(f"canary_{i}", row.stmt)
    add(CONTROL, "let _ = core::hint::black_box(0.5_f64).sqrt();")
    return "\n".join(lines) + "\n", ranges


def disallowed_diagnostics(stdout: str) -> list[tuple[str, int, str]]:
    """(lint, line, message) of every disallowed_* diagnostic in cargo's JSON."""
    out = []
    for raw in stdout.splitlines():
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") != "compiler-message":
            continue
        d = msg.get("message") or {}
        code = (d.get("code") or {}).get("code")
        if code not in LINTS:
            continue
        spans = [s for s in d.get("spans", []) if s.get("is_primary")] or d.get("spans", [])
        line = spans[0]["line_start"] if spans else 0
        out.append((code, line, d.get("message", "")))
    return out


def named_path(message: str) -> str | None:
    """The path a disallowed_* message names (the first backticked span)."""
    m = re.search(r"`([^`]+)`", message)
    return m.group(1) if m else None


def owning_type_allowed(row: Row, diag: tuple[str, int, str], entries: dict[str, list[str]]) -> bool:
    """Whether `diag` is the one extra diagnostic a row may carry: calling an
    inherent method names its type, so a banned method of a banned type
    (`RandomState::new` with `RandomState` banned) also gets the type's
    diagnostic. Allowed only as clippy::disallowed_types naming a
    disallowed-types entry of clippy.toml that is a `::`-segment prefix of the
    row's path."""
    lint, _, message = diag
    if lint != "clippy::disallowed_types":
        return False
    named = named_path(message)
    if named is None or named not in entries.get("disallowed-types", []):
        return False
    prefix, segs = named.split("::"), row.path.split("::")
    return len(prefix) < len(segs) and segs[:len(prefix)] == prefix


def equality(rows: list[Row], entries: dict[str, list[str]]) -> list[str]:
    """Every clippy.toml entry has exactly one row, and every row is an entry."""
    problems: list[str] = []
    counts = Counter((r.key, r.path) for r in rows)
    for key in KEYS:
        for path in dict.fromkeys(entries.get(key, [])):
            n = counts[(key, path)]
            if n == 0:
                problems.append(f"clippy.toml {key} entry `{path}` has no row in the data file")
            elif n > 1:
                lines = ", ".join(str(r.line) for r in rows if (r.key, r.path) == (key, path))
                problems.append(f"clippy.toml {key} entry `{path}` has {n} rows (data lines {lines}); "
                                "expected exactly 1")
    for row in rows:
        if row.path not in entries.get(row.key, []):
            problems.append(f"data line {row.line}: `{row.path}` is not listed under {row.key} in clippy.toml")
    return problems


def compared_nothing(rows: list[Row], entries: dict[str, list[str]]) -> list[str]:
    problems: list[str] = []
    if not any(r.planted for r in rows):
        problems.append("compared nothing: the data file has no planted rows")
    if not any(entries.values()):
        problems.append("compared nothing: clippy.toml has no disallowed-methods / disallowed-types entries")
    return problems


def static_problems(rows: list[Row], entries: dict[str, list[str]]) -> list[str]:
    """What can be decided without running clippy."""
    return compared_nothing(rows, entries) or equality(rows, entries)


def check(rows: list[Row], entries: dict[str, list[str]], exit_code: int,
          diags: list[tuple[str, int, str]], ranges: dict[str, tuple[int, int]]) -> list[str]:
    problems = compared_nothing(rows, entries)
    if problems:
        return problems
    problems += equality(rows, entries)
    if exit_code == 0:
        problems.append("clippy exited 0 although every canary function calls a banned item")
    used: set[int] = set()
    for i, row in enumerate(rows):
        if not row.planted:
            continue
        lo, hi = ranges[f"canary_{i}"]
        hits = [j for j, (_, ln, _) in enumerate(diags) if lo <= ln <= hi]
        used.update(hits)
        named = [j for j in hits if named_path(diags[j][2]) == row.path]
        extras = [j for j in hits if j not in named]
        if len(named) != 1:
            problems.append(f"data line {row.line}: `{row.path}` got {len(hits)} disallowed diagnostic(s), "
                            f"{len(named)} naming it (expected exactly 1)")
        if len(extras) > 1 or (extras and not owning_type_allowed(row, diags[extras[0]], entries)):
            problems.append(f"data line {row.line}: `{row.path}` got extra disallowed diagnostic(s) "
                            f"{[diags[j] for j in extras]} (at most one is allowed, a disallowed_types "
                            "diagnostic naming a disallowed-types entry that is a path prefix of the row)")
    lo, hi = ranges[CONTROL]
    control = [j for j, (_, ln, _) in enumerate(diags) if lo <= ln <= hi]
    if control:
        problems.append(f"the clean control function got {len(control)} disallowed diagnostic(s)")
    stray = [diags[j] for j in range(len(diags)) if j not in used and j not in control]
    if stray:
        problems.append(f"{len(stray)} disallowed diagnostic(s) outside any canary function: {stray}")
    return problems


def run_clippy(root: Path, lib: str, deps: str) -> tuple[int, str, str]:
    tmp = Path(tempfile.mkdtemp(prefix="clippy-canary-"))
    try:
        cargo = (root / "Cargo.toml").read_text(encoding="utf-8")
        edition = re.search(r'^edition\s*=\s*"([^"]+)"', cargo, re.M)
        (tmp / "src").mkdir()
        (tmp / "Cargo.toml").write_text(
            manifest(edition.group(1) if edition else "2021", deps, lints_tables(cargo)), encoding="utf-8")
        (tmp / "src" / "lib.rs").write_text(lib, encoding="utf-8")
        shutil.copyfile(root / "clippy.toml", tmp / "clippy.toml")
        for name in ("rust-toolchain.toml", "rust-toolchain"):
            if (root / name).exists():
                shutil.copyfile(root / name, tmp / name)
        p = subprocess.run(["cargo", "clippy", "--lib", "--quiet", "--message-format=json", "--", *CLIPPY_FLAGS],
                           cwd=tmp, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return p.returncode, p.stdout, p.stderr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calls", type=Path, default=CALLS)
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args(argv)

    rows, problems = parse_rows(args.calls.read_text(encoding="utf-8"))
    entries = clippy_toml_entries((args.root / "clippy.toml").read_text(encoding="utf-8"))
    lock_file = args.root / "Cargo.lock"
    lock = lock_packages(lock_file.read_text(encoding="utf-8")) if lock_file.exists() else {}
    deps, dep_problems = dependencies(rows, lock)
    problems += dep_problems
    problems += static_problems(rows, entries)
    diags: list[tuple[str, int, str]] = []
    if not problems:
        lib, ranges = render_lib(rows)
        code, stdout, stderr = run_clippy(args.root, lib, deps)
        diags = disallowed_diagnostics(stdout)
        problems = check(rows, entries, code, diags, ranges)
        if problems and not diags:
            sys.stderr.write(stderr[-4000:])
    if problems:
        for p in problems:
            print(f"clippy_canary: {p}", file=sys.stderr)
        return 1
    planted = sum(r.planted for r in rows)
    entries_n = sum(len(v) for v in entries.values())
    print(f"clippy_canary: {planted} planted, {len(rows) - planted} skipped, {entries_n} clippy.toml entries "
          f"(disallowed-methods / disallowed-types), {len(diags)} disallowed diagnostic(s), control clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
