#!/usr/bin/env python3
"""WGSL gate: no `return`/`discard` before a compute entry point's first barrier.

A `@compute` entry point that textually contains a `return;` or `discard;`
before its first `workgroupBarrier()`/`storageBarrier()` call was measured to
fault on D3D12 WARP (0xC0000005) even when the guard controlling that return
is workgroup-uniform (depends only on `@builtin(workgroup_id)`, never on
`@builtin(local_invocation_id)` or a subgroup builtin) — WGSL-spec-legal
uniform control flow is not sufficient evidence of safety on every backend.
The only safe pattern for an out-of-range workgroup is: every invocation
reaches every barrier regardless of any guard, and only the *final write* is
conditioned on the guard (`if (in_bounds) { buf[idx] = value; }`), never an
early `return`/`discard` ahead of a barrier.

This is a static, textual check (not a dataflow analysis): it does not try to
determine whether a given `return`/`discard` is actually reachable, or
whether it is upstream or downstream of a *particular* barrier in a
branch — only whether the function body contains the forbidden token
*anywhere* before the *last* barrier call textually. That is deliberately
conservative (a `return` that only looks early could still be downstream of
some other barrier in a sibling branch) but matches how the original 16-site
audit was done, and false positives are cheap to fix by moving the guard to
only control the final write.

Scans both `src/shaders/*.wgsl` and any Rust raw string literal under `src/`
whose content contains `@compute` (a kernel embedded as a Rust string, now
or in the future, is not invisible to this gate just because it is not a
`.wgsl` file).

Usage: `python3 no_return_before_barrier.py [--root DIR]`
`--root DIR` to check a different tree (used by the test)
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(HERE)
DEFAULT_SRC_DIRS = ("src/shaders",)
# `.rs` files under this directory are also scanned for WGSL embedded in a
# raw string literal (`r"..."` / `r#"..."#` / ... up to 8 `#`) that contains
# `@compute` — a future kernel inlined as a Rust string (or a `const
# X_SHADER: &str = r#"..."#;` the way an early draft of this very rewrite's
# oracle test embedded the old kernels) must not be invisible to this gate.
DEFAULT_RS_DIRS = ("src",)

BARRIER_RE = re.compile(r"\b(?:workgroupBarrier|storageBarrier)\s*\(")
RETURN_OR_DISCARD_RE = re.compile(r"\breturn\b|\bdiscard\b")
# `@compute` may have other attributes (`@workgroup_size(...)`) interleaved
# before the `fn name(` it applies to; attributes don't nest arbitrarily in
# WGSL so a simple "the next `fn` after `@compute`" scan is sufficient.
COMPUTE_FN_RE = re.compile(r"@compute\b.*?\bfn\s+(\w+)\s*\(", re.S)
# Rust raw string: `r` + 0-8 `#` + `"` + content + `"` + the same `#` run
# (the backreference enforces the closing delimiter matches the opening
# exactly, so a `"#` inside a `r#"..."#`-delimited string that doesn't
# repeat the full run is correctly treated as content, not a terminator).
RUST_RAW_STRING_RE = re.compile(r'r(#{0,8})"(.*?)"\1', re.S)


class LintError(Exception):
    """a WGSL tree or shader this check cannot read"""


def strip_comments(text: str) -> str:
    """Blank out `//` line comments and `/* */` block comments (WGSL block
    comments nest, per the WGSL spec) with spaces, preserving every
    original newline and the overall length so reported positions still
    map 1:1 onto the original text."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        two = text[i : i + 2]
        if two == "//":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if two == "/*":
            depth = 1
            out.append("  ")
            i += 2
            while i < n and depth > 0:
                pair = text[i : i + 2]
                if pair == "/*":
                    depth += 1
                    out.append("  ")
                    i += 2
                elif pair == "*/":
                    depth -= 1
                    out.append("  ")
                    i += 2
                else:
                    out.append(text[i] if text[i] == "\n" else " ")
                    i += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def find_entry_points(stripped: str) -> list[tuple[str, int, int]]:
    """(name, body_start, body_end) for every `@compute fn name(...) { ... }`,
    body_end exclusive, matched by brace depth from the first `{` after the
    parameter list's closing `)`."""
    out: list[tuple[str, int, int]] = []
    for m in COMPUTE_FN_RE.finditer(stripped):
        name = m.group(1)
        # Walk from the matched `(` to its matching `)`, then to the body's `{`.
        paren_start = m.end() - 1
        depth = 1
        j = paren_start + 1
        while j < len(stripped) and depth > 0:
            if stripped[j] == "(":
                depth += 1
            elif stripped[j] == ")":
                depth -= 1
            j += 1
        brace_start = stripped.find("{", j)
        if brace_start == -1:
            raise LintError(f"{name}: parameter list closed but no body `{{` found")
        depth = 1
        k = brace_start + 1
        while k < len(stripped) and depth > 0:
            if stripped[k] == "{":
                depth += 1
            elif stripped[k] == "}":
                depth -= 1
            k += 1
        if depth != 0:
            raise LintError(f"{name}: unbalanced braces (body never closed)")
        out.append((name, brace_start + 1, k - 1))
    return out


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def check_file(path: str, text: str, line_offset: int = 0) -> tuple[list[str], int]:
    """`line_offset` shifts every reported line number — used for WGSL
    embedded inside a Rust raw string literal, so errors point at the
    `.rs` file's real line, not a line count starting over at 1 inside the
    literal."""
    errors: list[str] = []
    stripped = strip_comments(text)
    entry_points = find_entry_points(stripped)
    for name, start, end in entry_points:
        body = stripped[start:end]
        barriers = list(BARRIER_RE.finditer(body))
        # No barrier at all in this entry point: nothing to be "before".
        if not barriers:
            continue
        # Checked against the LAST barrier, not just the first: a return
        # between two barriers has the same risk profile as one before the
        # first (some invocations could stop there while others go on to a
        # later barrier the return skipped).
        last_barrier = barriers[-1]
        before = body[: last_barrier.start()]
        rm = RETURN_OR_DISCARD_RE.search(before)
        if rm is not None:
            pos = start + rm.start()
            errors.append(
                f"{path}:{line_offset + line_of(text, pos)}: entry point `{name}` has a "
                f"`{rm.group(0)}` before a barrier "
                f"({path}:{line_offset + line_of(text, start + last_barrier.start())}) — "
                f"move the guard to condition only the final write, see "
                f"this file's module doc"
            )
    return errors, len(entry_points)


def iter_wgsl_files(root: str, src_dirs: tuple[str, ...]):
    for rel_dir in src_dirs:
        base = os.path.join(root, rel_dir)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirnames, filenames in os.walk(base):
            for fname in filenames:
                if fname.endswith(".wgsl"):
                    yield os.path.join(dirpath, fname)


def iter_rs_files(root: str, rs_dirs: tuple[str, ...]):
    for rel_dir in rs_dirs:
        base = os.path.join(root, rel_dir)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirnames, filenames in os.walk(base):
            for fname in filenames:
                if fname.endswith(".rs"):
                    yield os.path.join(dirpath, fname)


def embedded_wgsl_literals(text: str) -> list[tuple[int, str]]:
    """(absolute start offset of the literal's content, content) for every
    Rust raw string literal that looks like it embeds a WGSL compute
    shader (contains `@compute`) — plain `"..."` string literals are not
    checked, since nobody writes a multi-line shader through `\\n` escapes."""
    out: list[tuple[int, str]] = []
    for m in RUST_RAW_STRING_RE.finditer(text):
        content = m.group(2)
        if "@compute" in content:
            out.append((m.start(2), content))
    return out


def check(
    root: str,
    src_dirs: tuple[str, ...] = DEFAULT_SRC_DIRS,
    rs_dirs: tuple[str, ...] = DEFAULT_RS_DIRS,
) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    files = 0
    entry_points_total = 0
    for path in sorted(iter_wgsl_files(root, src_dirs)):
        files += 1
        with open(path, encoding="utf-8") as f:
            text = f.read()
        file_errors, n_entry_points = check_file(os.path.relpath(path, root), text)
        errors.extend(file_errors)
        entry_points_total += n_entry_points
    for path in sorted(iter_rs_files(root, rs_dirs)):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        rel = os.path.relpath(path, root)
        for content_start, content in embedded_wgsl_literals(text):
            files += 1
            line_offset = text.count("\n", 0, content_start)
            file_errors, n_entry_points = check_file(
                f"{rel} (embedded raw string)", content, line_offset
            )
            errors.extend(file_errors)
            entry_points_total += n_entry_points
    if files == 0:
        raise LintError(
            f"0 件の .wgsl file / embedded WGSL raw string "
            f"(src_dirs={src_dirs}, rs_dirs={rs_dirs} が存在しない、または空、比較できない)"
        )
    return errors, {"files": files, "entry_points": entry_points_total}


def main() -> int:
    # Windows runners default stdout/stderr to the system ANSI codepage (not
    # UTF-8), which cannot encode this script's Japanese error text and
    # crashes with UnicodeEncodeError before the actual check result ever
    # prints — measured on windows-latest CI.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()

    try:
        errors, counts = check(args.root)
    except LintError as e:
        print(f"error: {e}")
        return 2

    if counts["entry_points"] == 0:
        print("error: .wgsl file はあったが @compute entry point が 0 件 (比較できない)")
        return 2

    if errors:
        for e in errors:
            print(f"error: {e}")
        print(
            f"compared: files {counts['files']}, entry points {counts['entry_points']}, "
            f"violations {len(errors)}"
        )
        return 1
    print(
        f"compared: files {counts['files']}, entry points {counts['entry_points']}, "
        "violations 0"
    )
    print("no-return-before-barrier: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
