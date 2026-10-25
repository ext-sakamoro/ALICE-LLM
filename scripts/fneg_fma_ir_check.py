#!/usr/bin/env python3
"""ALICE-Law G3, IR half: no negation of a whole fused multiply-add result in the
optimised LLVM IR of the library.

The hazard G3 is about is a code-generation rewrite: `fneg(fma(x, y, z))` becomes
`fma(x, -y, -z)` (an fnmadd / fnmsub instruction) in the backend's instruction
selection, which is only value-preserving under a "no signed zero" assumption. The
release IR still holds the `fneg` (the fold happens after it), so this check reads
the IR the compiler optimised, after inlining, where a negation reached through a
wrapper function, a closure, an alias or a cast has collapsed to the same shape.
The syntax-tree checker (tools/no-outer-fneg-of-fma) is the fast pre-check with
source lines; this is the gate.

What it does, per invocation:

1. Builds the library in the release profile with
   `cargo rustc --lib --release --crate-type rlib -- --emit=llvm-ir
   -C debuginfo=line-tables-only` (a rlib needs no linker, so a cross `--target`
   works with only the target's standard library installed). The package's own
   artefacts are cleaned first, so a cached target directory cannot hand back an
   old `.ll`.
2. Reads the crate's `.ll` (missing or empty fails).
3. In every function, follows the operand of each negation back through
   `fpext` / `fptrunc` / `bitcast` / `freeze` / `phi` / `select` /
   `insertelement` / `extractelement` / `shufflevector` to a call of
   `@llvm.fma.*`, `@llvm.fmuladd.*`, `@fma` or `@fmaf`. A negation is `fneg`,
   `fsub` from negative zero, `fmul` by negative one, and an `xor` of the sign
   bit on a value bitcast from a float.
4. Counts the fma-family calls in the module and compares them with the floor
   recorded for the target in scripts/no-outer-fneg-of-fma-allowlist.toml
   (`[[ir]]` entries): fewer than the floor fails, and so does 0 calls when the
   library's sources contain a fma call (`expected-src-fma-calls` >= 1). 0 is
   allowed only where both the floor and `expected-src-fma-calls` are 0.
   `expected-src-fma-calls` counts the library's own calls (the syntax-tree check
   leaves out items under `#[test]` / `#[cfg(test)]`, which are not in the release
   build). When it is 0, the IR must hold no fma call either (one the source does
   not show is a finding), and the 0 findings must not be vacuous: the shared
   probes in tools/fneg-fma-ir-probes/ are compiled for the same target and each
   must get its expected verdict (`// expect-ir:`), with at least one probe run.
   Measured before this rule (2026-10-18, the G3 IR jobs of each carrier's main):
   every carrier whose library holds no fma call (ALICE-ML, -Crypto, -Kinematics,
   -DetMath, -Sync, -Zip) has 0 fma calls in its release IR on both targets.

What the IR half can and cannot see: it refuses a negation that reaches the IR. A
negation whose only use the optimiser folds into the following arithmetic leaves no
negation to find: `-(a.mul_add(b, c)) * C` becomes `fmul (fma), -C`, the same IR as
the conforming `a.mul_add(b, c) * -C` (G3 bans negating the whole result, not
multiplying it by a negative constant), and `x + -(fma) * C` becomes `fsub x,
(fma * C)`. That is ALICE-Edge's `-(lat_deg.mul_add(100.0, -lat_raw))` used only as
`lat_deg + lat_min * INV_60` (2026-10-18: the syntax-tree check refused it, the IR
check found 0 negated fma calls on aarch64). G3 is defined on the source, so those
forms are the syntax-tree checker's to refuse; the IR half claims only what the IR
can distinguish. Probes 48 and 49 pin both folds as `expect-ir: clean` with
`expect-syn: flag`, and probe 50 pins the other side: when the negated result is
also used on its own, the negation survives as an fneg of the fma and is refused.

A finding is reported with the source location from the `!dbg` attachment of the
negation (for inlined code, the location in the callee, which is the line to fix).
There is no allowlist for findings: the IR covers the library only, and G3 allows
no exception in library sources.

Usage:
    python3 scripts/fneg_fma_ir_check.py [--target TRIPLE] [--package NAME]
        [--features LIST | --all-features] [--no-default-features]
        [--write-floor]
    python3 scripts/fneg_fma_ir_check.py --all-floor-targets [--package NAME] [features]
    python3 scripts/fneg_fma_ir_check.py --ll FILE   (analyse one .ll, no build)

`--all-floor-targets` checks every target that has an `[[ir]]` floor for the package
and whose standard library is installed, and fails when none of them could be run.
`--findings-only` checks the host build for negated fma results and prints the counts
without comparing them with a floor (scripts/preflight.sh on a machine whose host is
not a CI target: a crate with a C build script cannot be cross-built without a cross
C toolchain, so the floors are checked in CI, on runners native to each target).

`--write-floor` records the measured count as this target's floor (run it on the
CI host class, not on a developer machine). Exit 1 on a finding or a count
problem, 2 on a usage or build error.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ALLOWLIST = Path("scripts/no-outer-fneg-of-fma-allowlist.toml")

FMA_CALL = re.compile(
    r"=\s*(?:tail |musttail |notail )?call\b[^@]*@"
    r"(llvm\.fma\.[\w.]+|llvm\.fmuladd\.[\w.]+|fmaf?)\("
)
DEF = re.compile(r"^\s*(%[-\w.$]+)\s*=\s*(.*)$")
OPERAND = re.compile(r"%[-\w.$]+")
DBG = re.compile(r"!dbg (!\d+)")
NEG_ZERO = re.compile(r"-0\.0+e\+00|0x8000000000000000|splat \((?:float|double) -0\.0+e\+00\)")
NEG_ONE = re.compile(r"-1\.0+e\+00|0xBFF0000000000000|splat \((?:float|double) -1\.0+e\+00\)")
XOR = re.compile(r"xor (?:<\d+ x )?i(32|64)>? (.*)")
INT_LITERAL = re.compile(r"(?<![%\w.])-?\d+(?![\w.])")


def is_sign_mask(rhs: str) -> bool:
    """`rhs` (an `xor`) has a constant operand made only of float sign bits: per
    element 0x80000000 for i32, and for i64 either the f64 sign bit or the two f32
    sign bits a bitcast of a float vector to i64 lanes packs into one element."""
    m = XOR.match(rhs)
    if not m:
        return False
    width = int(m.group(1))
    operands = m.group(2).split(", !")[0]
    values = [int(v) & ((1 << width) - 1) for v in INT_LITERAL.findall(operands.replace(f"i{width} ", " "))]
    allowed = {0x8000_0000} if width == 32 else {0x8000_0000_0000_0000, 0x8000_0000_8000_0000}
    return bool(values) and all(v in allowed for v in values)


# rustc flags for the IR: `link-dead-code` makes rustc emit every function of the
# crate, including the small `pub` ones it would otherwise leave to be generated in
# the caller's crate (cross-crate inlining); line tables give the `!dbg` locations
IR_FLAGS = ("--emit=llvm-ir", "-C", "link-dead-code", "-C", "debuginfo=line-tables-only")
PASS_THROUGH = ("fpext", "fptrunc", "bitcast", "freeze", "phi", "select",
                "insertelement", "extractelement", "shufflevector")


@dataclass
class Finding:
    function: str
    location: str
    instruction: str


@dataclass
class Analysis:
    fma_calls: int = 0
    functions: int = 0
    findings: list[Finding] = field(default_factory=list)


def _operands(text: str) -> list[str]:
    # drop metadata and attribute groups so `!dbg !12` / `#3` never read as values
    text = text.split(", !")[0]
    return OPERAND.findall(text)


def _locations(lines: list[str]) -> dict[str, str]:
    """`!N` of a DILocation -> "file:line:column". An inlined location is reported
    at the first place in the inlining chain that is in the crate's own files, so a
    negation inlined from the standard library (Neg::neg, a SIMD intrinsic's
    wrapper) points at the line of the crate that wrote it. A file is the crate's
    own when its DIFile directory is the compile unit's (the directory rustc ran
    in) and its name is relative: the standard library's files carry a remapped
    or absolute path, and with the rust-src component missing they are relative
    names under another directory, so the name alone does not tell them apart."""
    files: dict[str, tuple[str, str]] = {}
    unit_file = ""
    scopes: dict[str, str] = {}
    locs: dict[str, tuple[str, str, str | None]] = {}
    file_re = re.compile(r'^(!\d+) = !DIFile\(filename: "([^"]*)", directory: "([^"]*)"')
    unit_re = re.compile(r"^!\d+ = distinct !DICompileUnit\(.*?file: (!\d+)")
    scope_re = re.compile(r"^(!\d+) = (?:distinct )?!DI(?:Subprogram|LexicalBlock(?:File)?)\(.*?file: (!\d+)")
    loc_re = re.compile(r"^(!\d+) = (?:distinct )?!DILocation\(line: (\d+)(?:, column: (\d+))?, scope: (!\d+)(?:, inlinedAt: (!\d+))?")
    for line in lines:
        if m := file_re.match(line):
            files[m.group(1)] = (m.group(2), m.group(3))
        elif m := unit_re.match(line):
            unit_file = m.group(1)
        elif m := scope_re.match(line):
            scopes[m.group(1)] = m.group(2)
        elif m := loc_re.match(line):
            locs[m.group(1)] = (f"{m.group(2)}:{m.group(3) or 0}", m.group(4), m.group(5))

    unit_dir = files.get(unit_file, ("", ""))[1]

    def render(key: str) -> tuple[str, bool]:
        pos, scope, _ = locs[key]
        name, directory = files.get(scopes.get(scope, ""), ("?", ""))
        own = bool(unit_dir) and directory == unit_dir and not name.startswith("/") and name != "?"
        return f"{name}:{pos}", own

    out = {}
    for key in locs:
        chain, k, seen = [], key, set()
        while k in locs and k not in seen:
            seen.add(k)
            chain.append(render(k))
            k = locs[k][2]
        own = [text for text, local in chain if local]
        out[key] = own[0] if own else chain[-1][0]
    return out


def analyse(text: str) -> Analysis:
    """Find negated fma results in one LLVM IR module."""
    lines = text.splitlines()
    locs = _locations(lines)
    result = Analysis()
    i = 0
    while i < len(lines):
        if not lines[i].startswith("define "):
            i += 1
            continue
        header = lines[i]
        name = re.search(r"@([-\w.$\"]+)\(", header)
        fname = name.group(1) if name else "?"
        result.functions += 1
        body: list[str] = []
        i += 1
        while i < len(lines) and lines[i] != "}":
            body.append(lines[i])
            i += 1
        _analyse_function(fname, body, locs, result)
    return result


def _analyse_function(fname: str, body: list[str], locs: dict[str, str], result: Analysis) -> None:
    defs: dict[str, str] = {}
    for line in body:
        if m := DEF.match(line):
            defs[m.group(1)] = m.group(2)
    fma_defs = {name for name, rhs in defs.items() if FMA_CALL.search("= " + rhs)}
    result.fma_calls += len(fma_defs)
    if not fma_defs:
        return

    def reaches_fma(value: str, seen: set[str]) -> bool:
        if value in fma_defs:
            return True
        if value in seen or value not in defs:
            return False
        seen.add(value)
        rhs = defs[value]
        op = rhs.split()[0] if rhs.split() else ""
        if op not in PASS_THROUGH:
            return False
        if op == "phi":
            return any(reaches_fma(v, seen) for v in re.findall(r"\[\s*(%[-\w.$]+)", rhs))
        if op == "select":
            ops = _operands(rhs)
            return any(reaches_fma(v, seen) for v in ops[1:])
        return any(reaches_fma(v, seen) for v in _operands(rhs))

    for line in body:
        m = DEF.match(line)
        if not m:
            continue
        rhs = m.group(2)
        words = rhs.split()
        if not words:
            continue
        op = words[0]
        operand: list[str] = []
        if op == "fneg":
            operand = _operands(rhs)[:1]
        elif op == "fsub" and NEG_ZERO.search(rhs.split(",")[0]):
            operand = _operands(rhs)[:1]
        elif op == "fmul" and NEG_ONE.search(rhs):
            operand = _operands(rhs)
        elif op == "xor" and is_sign_mask(rhs):
            operand = _operands(rhs)
        if any(reaches_fma(v, set()) for v in operand):
            d = DBG.search(line)
            loc = locs.get(d.group(1), "?") if d else "?"
            result.findings.append(Finding(fname, loc, line.strip()))


def _cargo_metadata(package: str | None) -> tuple[str, str]:
    meta = json.loads(subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        check=True, capture_output=True, text=True).stdout)
    members = [p for p in meta["packages"] if p["id"] in meta["workspace_members"]]
    if package:
        members = [p for p in members if p["name"] == package]
    elif len(members) > 1:
        raise SystemExit("fneg_fma_ir_check: a workspace has several packages, pass --package")
    if len(members) != 1:
        raise SystemExit(f"fneg_fma_ir_check: package {package!r} not found")
    pkg = members[0]
    libs = [t for t in pkg["targets"] if set(t["kind"]) & {"lib", "rlib", "cdylib", "staticlib"}]
    if not libs:
        raise SystemExit(f"fneg_fma_ir_check: {pkg['name']} has no library target")
    return pkg["name"], libs[0]["name"].replace("-", "_")


def build_ll(args: argparse.Namespace) -> Path:
    package, crate = _cargo_metadata(args.package)
    shared = os.environ.get("CARGO_TARGET_DIR")
    target_dir = Path(shared or tempfile.mkdtemp(prefix="g3-ir-"))
    profile_dir = target_dir / (args.target or "") / "release" / "deps"
    sel = ["-p", package]
    tgt = ["--target", args.target] if args.target else []
    # a shared target directory may hold an older build of the package: clean it so the IR
    # is emitted again; a fresh temporary directory holds nothing, and cargo refuses to
    # clean a directory that has no CACHEDIR.TAG yet
    if shared:
        subprocess.run(["cargo", "clean", "--release", *sel, *tgt, "--target-dir", str(target_dir)],
                       check=True, capture_output=True)
    for old in profile_dir.glob(f"{crate}-*.ll"):
        old.unlink()
    feat: list[str] = []
    if args.all_features:
        feat.append("--all-features")
    if args.no_default_features:
        feat.append("--no-default-features")
    if args.features:
        feat += ["--features", args.features]
    cmd = ["cargo", "rustc", "--lib", "--release", "--crate-type", "rlib", *sel, *tgt, *feat,
           "--target-dir", str(target_dir), "--",
           *IR_FLAGS]
    print("fneg_fma_ir_check: running " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(2)
    found = sorted(profile_dir.glob(f"{crate}-*.ll"))
    if len(found) != 1:
        print(f"fneg_fma_ir_check: expected one {crate}-*.ll in {profile_dir}, found {len(found)}")
        raise SystemExit(1)
    return found[0]


def host_triple() -> str:
    out = subprocess.run(["rustc", "-vV"], check=True, capture_output=True, text=True).stdout
    return re.search(r"^host: (\S+)", out, re.M).group(1)


def read_manifest(root: Path) -> dict:
    path = root / ALLOWLIST
    if not path.is_file():
        raise SystemExit(f"fneg_fma_ir_check: {ALLOWLIST} is missing")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def floor_for(manifest: dict, target: str, package: str) -> int | None:
    for entry in manifest.get("ir", []):
        if entry.get("target") == target and entry.get("package") == package:
            return int(entry["min-fma-calls"])
    return None


def write_floor(root: Path, target: str, package: str, count: int) -> None:
    path = root / ALLOWLIST
    text = path.read_text(encoding="utf-8")
    block = re.compile(
        r'\[\[ir\]\]\ntarget = "' + re.escape(target) + r'"\npackage = "' + re.escape(package)
        + r'"\nmin-fma-calls = \d+\n')
    new = f'[[ir]]\ntarget = "{target}"\npackage = "{package}"\nmin-fma-calls = {count}\n'
    if block.search(text):
        text = block.sub(new, text)
    else:
        text = text.rstrip("\n") + "\n\n" + new
    path.write_text(text, encoding="utf-8")


def check_counts(analysis: Analysis, floor: int | None, src_calls: int | None) -> list[str]:
    problems = []
    if analysis.functions == 0:
        problems.append("the IR holds no function definition (wrong or empty .ll)")
    if src_calls is None:
        problems.append(f"{ALLOWLIST} has no expected-src-fma-calls")
    if floor is None:
        problems.append(f"{ALLOWLIST} has no [[ir]] floor for this target and package")
        return problems
    if analysis.fma_calls < floor:
        problems.append(f"fma calls in IR {analysis.fma_calls} < recorded floor {floor}")
    if analysis.fma_calls == 0 and not (floor == 0 and src_calls == 0):
        problems.append("0 fma calls in IR, but the floor or expected-src-fma-calls is not 0")
    if src_calls and analysis.fma_calls == 0:
        problems.append(f"the sources hold {src_calls} fma call(s) but the IR holds none")
    if src_calls == 0 and analysis.fma_calls > 0:
        problems.append(f"the library's sources hold no fma call but the IR holds {analysis.fma_calls}"
                        " (a fma the source does not show)")
    return problems


PROBES_REL = Path("tools/fneg-fma-ir-probes")


def probe_proof(root: Path, target: str, compile_ll=None) -> tuple[list[str], int]:
    """(problems, probes run): every shared probe that applies to `target`'s architecture, compiled for it,
    gets the verdict its `// expect-ir:` line names; none run, or a wrong verdict, is a problem. This is what
    makes a run with 0 fma calls and 0 findings non-vacuous: the detector flags what it must on this target."""
    arch = target.split("-", 1)[0]
    probes = sorted((root / PROBES_REL).glob("*.rs"))
    if compile_ll is None:
        def compile_ll(probe: Path, out: Path) -> str:
            subprocess.run(["rustc", "--edition", "2021", "--crate-type", "lib", "-C", "opt-level=3",
                            *IR_FLAGS[1:], "--emit=llvm-ir", "--target", target, "-o", str(out), str(probe)],
                           check=True, capture_output=True, text=True, cwd=root)
            return out.read_text(encoding="utf-8")
    problems: list[str] = []
    ran = 0
    with tempfile.TemporaryDirectory() as d:
        for probe in probes:
            text = probe.read_text(encoding="utf-8")
            where = re.search(r"^// targets: (\S+)", text, re.M)
            want = re.search(r"^// expect-ir: (\S+)", text, re.M)
            if not where or not want:
                problems.append(f"{probe.name}: no `// targets:` / `// expect-ir:` line")
                continue
            if where.group(1) not in ("any", arch):
                continue
            try:
                a = analyse(compile_ll(probe, Path(d) / f"{probe.stem}.ll"))
            except subprocess.CalledProcessError as e:
                problems.append(f"{probe.name}: does not compile for {target}: {e.stderr.strip()[-300:]}")
                continue
            ran += 1
            got = "flag" if a.findings else "clean"
            if got != want.group(1):
                problems.append(f"{probe.name} on {target}: expect {want.group(1)}, got {got}")
    if ran == 0:
        problems.append(f"no probe of {root / PROBES_REL} ran for {target} (the 0 findings would be vacuous)")
    return problems, ran


def report(analysis: Analysis, label: str) -> None:
    for f in analysis.findings:
        print(f"error: {f.location}: fneg of a fma result in {f.function}: {f.instruction}")
    print(f"fneg_fma_ir_check: {label}: {analysis.functions} functions, "
          f"{analysis.fma_calls} fma calls, {len(analysis.findings)} negated")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target")
    ap.add_argument("--package")
    ap.add_argument("--features")
    ap.add_argument("--all-features", action="store_true")
    ap.add_argument("--no-default-features", action="store_true")
    ap.add_argument("--write-floor", action="store_true")
    ap.add_argument("--all-floor-targets", action="store_true")
    ap.add_argument("--findings-only", action="store_true")
    ap.add_argument("--ll", type=Path, help="analyse this .ll only (no build, no count check)")
    args = ap.parse_args(argv)

    if args.ll:
        text = args.ll.read_text(encoding="utf-8") if args.ll.is_file() else ""
        if not text.strip():
            print(f"fneg_fma_ir_check: {args.ll} is missing or empty")
            return 1
        analysis = analyse(text)
        report(analysis, str(args.ll))
        return 1 if analysis.findings or analysis.functions == 0 else 0

    root = Path.cwd()
    manifest = read_manifest(root)
    package, _ = _cargo_metadata(args.package)
    if args.all_floor_targets:
        sysroot = subprocess.run(["rustc", "--print", "sysroot"], check=True,
                                 capture_output=True, text=True).stdout.strip()
        rustlib = Path(sysroot) / "lib" / "rustlib"
        targets = [e["target"] for e in manifest.get("ir", []) if e.get("package") == package]
        runnable = [t for t in targets if (rustlib / t / "lib").is_dir()]
        if not runnable:
            print(f"fneg_fma_ir_check: none of the floor targets {targets} is installed "
                  f"(rustup target add {' '.join(targets) or '<target>'})")
            return 1
        rc = 0
        for t in runnable:
            args.target = t
            rc |= run_one(args, root, manifest, package)
        return rc
    return run_one(args, root, manifest, package)


def run_one(args: argparse.Namespace, root: Path, manifest: dict, package: str) -> int:
    target = args.target or host_triple()
    ll = build_ll(args)
    text = ll.read_text(encoding="utf-8")
    if not text.strip():
        print(f"fneg_fma_ir_check: {ll} is empty")
        return 1
    analysis = analyse(text)
    report(analysis, f"{package} {target}")
    if args.findings_only:
        problems = [] if analysis.functions else ["the IR holds no function definition (wrong or empty .ll)"]
        for p in problems:
            print(f"error: {p}")
        return 1 if analysis.findings or problems else 0
    if args.write_floor:
        write_floor(root, target, package, analysis.fma_calls)
        print(f"fneg_fma_ir_check: recorded floor {analysis.fma_calls} for {package} {target}")
        return 1 if analysis.findings else 0
    src_calls = manifest.get("expected-src-fma-calls")
    problems = check_counts(analysis, floor_for(manifest, target, package), src_calls)
    if src_calls == 0:
        probe_problems, ran = probe_proof(root, target)
        problems += probe_problems
        print(f"fneg_fma_ir_check: probes for {target}: {ran} ran, {len(probe_problems)} problem(s)")
    for p in problems:
        print(f"error: {p}")
    return 1 if analysis.findings or problems else 0


if __name__ == "__main__":
    sys.exit(main())
