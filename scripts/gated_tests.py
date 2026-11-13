#!/usr/bin/env python3
"""Check that every feature-gated test file is run (and, in counted mode, counted) by every cover.

usage: python3 scripts/gated_tests.py [--root DIR]

The script is shared byte for byte between repositories; what differs lives in
scripts/gated_tests.toml:

    mode = "counted"          # or "run"
    ignored = "run"           # ignored tests must run (--include-ignored); "skip" when they are
                              # ignored on purpose (a known defect kept as an oracle)
    [[cover]]                 # one per list of commands that must run every gated file
    name = "ci"
    globs = [".github/workflows/*.yml"]
    [[cover]]
    name = "preflight"
    globs = ["scripts/preflight.sh"]
    [unread]                  # gated files whose gate this script cannot read, with the reason
    some_file = "any(): run by the full-features job"

A test file is gated when it holds `cfg(...)` naming a feature (the file, or an item in it). Its
tests run only in a `cargo test` command whose features enable the gate, so a file that no
command enables is silently never run. Each item gate (`#[cfg]`) needs its own features and the
file's (`#![cfg]`). In counted mode one command must meet every gate of the file (it counts all
its tests); in run mode each gate must be met by some command. The command must
  - enable the gate: the features of `--features` / `-F`, `--all-features`, and the default
    features unless `--no-default-features`, closed over the [features] table of Cargo.toml
    (`all(...)` needs every feature it names),
  - run test files at all (no target filter, `--tests`, `--all-targets`, or `--test <file>`;
    `--lib` / `--doc` / `--bins` alone run none),
  - pass `--include-ignored` (or `--ignored`) after `--` when the file has an ignored test and
    `ignored = "run"`,
  - and, in counted mode, pipe into `scripts/min_passed.py` naming the file with a floor equal
    to its `#[test]` and `#[wasm_bindgen_test]` count (a lower floor lets a new test go uncounted)
    and `--no-ignored` when the file has an ignored test.

A dev-dependency on the crate itself (`path = "."`) enables its features in every command, and
`[[test]] required-features` in Cargo.toml is a gate of the whole file. A YAML folded block (`run: >-`) is one command. A `$NAME` / `${NAME}` in a command is replaced by the value set in the same file (a YAML `env:`
entry `NAME: value`, or a shell line of assignments only: `NAME=value ...`, also after `export`,
`local`, `readonly` or `declare [-flags]`, values bare or quoted; the last one wins); one left unset
enables nothing. A `NAME=value` word inside a command (`echo A=b`, `A=b cargo test`) sets nothing.

Fails when: there is no configuration, the mode or the ignored policy is unknown or there is no
cover; a cover's globs match no file or the files hold no `cargo test` command; no gated file is
compared; a gate holds `any(` or `not(` and the file is not listed in [unread]; an [unread] entry
names a file that does not exist or whose gate is readable (a stale exemption).

Known limits (not read, so not checked): `cargo hack` and `cargo nextest` (they run no `cargo test`
command here); `-p` / `--package` and workspace members (the root Cargo.toml's features are used);
features of dependencies (`dep:x`, `x/f`); variables set in another file, by a matrix or in a
loop; a variable's scope (the last assignment in the file wins); `cfg` gates on
anything but features (target, test) are ignored.
"""

import re
import sys
from fnmatch import fnmatch
from pathlib import Path

if sys.version_info < (3, 11):
    print("gated_tests: needs Python >= 3.11 (tomllib)", file=sys.stderr)
    sys.exit(1)

import tomllib  # noqa: E402

CFG = re.compile(r"#!?\[cfg\((.*?)\)\]\s*$", re.M)
FEATURE = re.compile(r'feature\s*=\s*"([A-Za-z0-9_-]+)"')
CARGO_TEST = re.compile(r"\bcargo\s+(?:\+\S+\s+)?test\b(?P<rest>[^\n]*)")
# a YAML `env:` entry `NAME: value`
YAML_VARIABLE = re.compile(r'^\s*(?:-\s*)?([A-Z_][A-Z0-9_]*):\s+"?([^"\s]*)"?\s*$')
# a shell line of assignments only, optionally after export / local / readonly / declare [-flags]
SHELL_WORD = r"""[A-Z_][A-Z0-9_]*=(?:"[^"]*"|'[^']*'|[^\s"';]*)"""
SHELL_LINE = re.compile(rf"^\s*(?:-\s*)?(?:(?:export|local|readonly|declare(?:\s+-[A-Za-z]+)*)\s+)?((?:{SHELL_WORD}\s*)+)$")
SHELL_ASSIGN = re.compile(r"""([A-Z_][A-Z0-9_]*)=(?:"([^"]*)"|'([^']*)'|([^\s"';]*))""")
FLOOR = re.compile(r"\b([A-Za-z0-9_]+)=(\d+)\b")
TARGET_FLAGS = {"--lib", "--doc", "--bins", "--bin", "--examples", "--example", "--benches", "--bench"}


def features_table(root: Path):
    """(feature -> the features it enables, default features, features every `cargo test` has,
    test file -> its required-features)"""
    manifest = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    table = manifest.get("features", {})
    graph = {k: [x for x in v if "/" not in x and not x.startswith("dep:")] for k, v in table.items()}
    default = graph.get("default", [])
    # a dev-dependency on the crate itself enables its features in every test build
    always = []
    for dep in manifest.get("dev-dependencies", {}).values():
        if isinstance(dep, dict) and str(dep.get("path", "")).rstrip("/") in (".", "./."):
            always += dep.get("features", []) + ([] if dep.get("default-features", True) is False else default)
    required = {}
    for t in manifest.get("test", []):
        stem = Path(t["path"]).stem if "path" in t else t.get("name")
        if stem and t.get("required-features"):
            required[stem] = set(t["required-features"])
    return graph, default, always, required


def closure(graph, start):
    seen, stack = set(), list(start)
    while stack:
        f = stack.pop()
        if f not in seen:
            seen.add(f)
            stack.extend(graph.get(f, []))
    return seen


def gate(text: str):
    """None when the file is not gated, "unread" when a gate holds any()/not(), else the feature
    sets its tests need: the file's gate (`#![cfg]`) with each item's gate (`#[cfg]`)"""
    gates = [(m.group(0).startswith("#!"), m.group(1)) for m in CFG.finditer(text) if FEATURE.search(m.group(1))]
    if not gates:
        return None
    if any(re.search(r"\b(?:any|not)\s*\(", e) for _, e in gates):
        return "unread"
    whole = {f for inner, e in gates if inner for f in FEATURE.findall(e)}
    items = [set(FEATURE.findall(e)) for inner, e in gates if not inner]
    return [whole | i for i in items] if items else [whole]


def assignments(text: str):
    """(name, value) of each variable set in text, in order: YAML `env:` entries and shell lines of
    assignments (several on one line, after export / local / readonly / declare)"""
    out = []
    for line in text.split("\n"):
        m = YAML_VARIABLE.match(line)
        if m:
            out.append((m.group(1), m.group(2)))
            continue
        m = SHELL_LINE.match(line)
        if m:
            out += [(a.group(1), next(g for g in a.groups()[1:] if g is not None)) for a in SHELL_ASSIGN.finditer(m.group(1))]
    return out


def fold(text: str):
    """a YAML folded block (`key: >` / `key: >-`) as one line: its lines joined with spaces"""
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        if re.search(r":\s*>[-+]?\s*$", line):
            indent = len(line) - len(line.lstrip())
            block = []
            while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent):
                block.append(lines[i].strip())
                i += 1
            out[-1] = re.sub(r">[-+]?\s*$", "", line) + " ".join(b for b in block if b)
    return "\n".join(out)


def commands(text: str):
    """(cargo arguments, min_passed arguments or None) of each `cargo test` command"""
    text = re.sub(r"\\\n\s*", " ", text)  # a command continued with a trailing backslash
    text = fold(text)
    # variables set in the same file (a YAML `env:` entry or a shell assignment), last one wins
    values = dict(assignments(text))
    text = re.sub(r"\$\{?([A-Z_][A-Z0-9_]*)\}?", lambda m: values.get(m.group(1), m.group(0)), text)
    out = []
    for m in CARGO_TEST.finditer(text):
        cmd = re.split(r"\s*(?:&&|\|\||;)\s*", m.group("rest"))[0]
        parts = cmd.split("|")
        mp = None
        if len(parts) > 1 and "scripts/min_passed.py" in parts[1]:
            mp = parts[1].split("scripts/min_passed.py", 1)[1]
        out.append((re.sub(r"\d?>&?\d?\S*", " ", parts[0]), mp))
    return out


def parse_cargo(args: str, graph, default, always):
    """(enabled features, runs test files, all test files, the --test filters, test-binary arguments)"""
    before, _, after = args.partition(" -- ")
    toks = before.split()
    feats, all_feats, no_default, tests, targets = [], False, False, [], set()
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in ("--features", "-F") and i + 1 < len(toks):
            feats += re.split(r"[,\s]+", toks[i + 1].strip("'\""))
            i += 1
        elif t.startswith("--features="):
            feats += t.split("=", 1)[1].strip("'\"").split(",")
        elif t == "--all-features":
            all_feats = True
        elif t == "--no-default-features":
            no_default = True
        elif t == "--test" and i + 1 < len(toks) and not toks[i + 1].startswith("-"):
            tests.append(toks[i + 1])
            i += 1
        elif t in ("--tests", "--all-targets"):
            targets.add("tests")
        elif t in TARGET_FLAGS:
            targets.add(t)
        i += 1
    enabled = set(graph) if all_feats else closure(graph, feats + ([] if no_default else default) + always)
    runs_tests = not targets and not tests or "tests" in targets or bool(tests)
    return enabled, runs_tests, "tests" in targets, tests, after


def covers(cmd, name, need, count, ignored, mode, graph, default, always):
    # ignored: the file has an ignored test that must run
    """None when the command runs (and counts) the file as required, else why not"""
    cargo, mp = cmd
    enabled, runs_tests, all_tests, tests, after = parse_cargo(cargo, graph, default, always)
    if mode == "counted":
        if mp is None:
            return "not piped into min_passed"
        floors = dict((n, int(v)) for n, v in FLOOR.findall(mp))
        if name not in floors:
            return "not named to min_passed"
    if not need <= enabled:
        return f"features {sorted(need - enabled)} not enabled"
    if not runs_tests:
        return "runs no test file (--lib / --doc / ... only)"
    if tests and not all_tests and name not in tests:
        return "not among the --test filters"
    if ignored and "--include-ignored" not in after and "--ignored" not in after:
        return "has ignored tests but the command does not pass --include-ignored"
    if mode == "counted":
        if floors[name] != count:
            return f"floor {floors[name]}, the file has {count} tests"
        if ignored and "--no-ignored" not in mp:
            return "has ignored tests but min_passed is not given --no-ignored"
    return None


def check(root: Path):
    """the problems found, and how many (file, cover) pairs were compared"""
    conf_path = root / "scripts" / "gated_tests.toml"
    if not conf_path.exists():
        return [f"no configuration {conf_path.relative_to(root)}"], 0
    conf = tomllib.loads(conf_path.read_text(encoding="utf-8"))
    mode = conf.get("mode")
    if mode not in ("counted", "run"):
        return [f"mode {mode!r}: must be \"counted\" or \"run\""], 0
    cover_list = conf.get("cover", [])
    if not cover_list:
        return ["no [[cover]] in the configuration"], 0
    policy = conf.get("ignored", "run")
    if policy not in ("run", "skip"):
        return [f"ignored {policy!r}: must be \"run\" (ignored tests must run) or \"skip\""], 0
    unread = conf.get("unread", {})
    graph, default, always, required = features_table(root)

    problems, files = [], {}
    gates = {}
    for f in sorted((root / "tests").glob("*.rs")):
        text = f.read_text(encoding="utf-8")
        g = gate(text)
        if f.stem in required and g != "unread":
            # [[test]] required-features is a gate of the whole file
            g = [set(required[f.stem]) | x for x in (g or [set()])]
        gates[f.stem] = g
        if g == "unread":
            if f.stem not in unread:
                problems.append(f"tests/{f.stem}.rs: gate with any()/not() this script cannot read; list it in [unread] with the reason")
        elif g is not None:
            files[f.stem] = (g, text.count("#[test]") + text.count("#[wasm_bindgen_test]"), policy == "run" and "#[ignore" in text)
    for name in unread:
        if name not in gates:
            problems.append(f"[unread] {name}: no such file tests/{name}.rs (stale)")
        elif gates.get(name) != "unread":
            problems.append(f"[unread] {name}: its gate is readable now (stale), remove the entry")
    for name, (sets, _, _) in files.items():
        unknown = set().union(*sets) - set(graph)
        if unknown:
            problems.append(f"tests/{name}.rs: features {sorted(unknown)} are not in Cargo.toml [features]")
    if not files:
        problems.append("no readable feature-gated test file found under tests/")

    compared = 0
    for cover in cover_list:
        label = cover.get("name", "?")
        paths = sorted({p for g in cover.get("globs", []) for p in root.glob(g) if p.is_file()})
        if not paths:
            problems.append(f"{label}: globs {cover.get('globs')} match no file")
            continue
        cmds = [c for p in paths for c in commands(p.read_text(encoding="utf-8"))]
        if not cmds:
            problems.append(f"{label}: no `cargo test` command found")
            continue
        for name, (sets, count, ignored) in files.items():
            compared += 1
            # counted: one command runs every test of the file; run: each gate is met by some command
            for need in [set().union(*sets)] if mode == "counted" else sets:
                why = [covers(c, name, need, count, ignored, mode, graph, default, always) for c in cmds]
                if all(w is not None for w in why):
                    # the reason from a command that names the file, if one does
                    named = [w for w in why if w not in ("not named to min_passed", "not piped into min_passed")]
                    problems.append(
                        f"{label}: tests/{name}.rs (features {sorted(need)}, {count} tests): "
                        + (named[0] if named else "no min_passed command names it")
                    )
    return problems, compared


def main() -> int:
    args = sys.argv[1:]
    root = Path(args[args.index("--root") + 1]) if "--root" in args else Path(__file__).resolve().parent.parent
    problems, compared = check(root)
    print(f"gated_tests: {compared} (file, cover) pairs compared", file=sys.stderr)
    for p in problems:
        print(f"gated_tests: {p}", file=sys.stderr)
    return 1 if problems or compared == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
