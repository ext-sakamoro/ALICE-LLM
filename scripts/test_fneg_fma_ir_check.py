#!/usr/bin/env python3
"""Tests of scripts/fneg_fma_ir_check.py (ALICE-Law G3, IR half).

The probes in tools/fneg-fma-ir-probes/ (shared with the syntax-tree checker's
tests) are compiled with the pinned rustc at test time, in the release profile
and with the same flags the check passes, for an installed aarch64 target and an
installed x86_64 target (both are required: the SIMD probes only build on their
own architecture). No IR is checked in, so nothing is hand-edited. Each probe
states its verdict on an `expect-ir:` line.

Usage: python3 scripts/test_fneg_fma_ir_check.py
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import fneg_fma_ir_check as g  # noqa: E402

PROBES = ROOT / "tools" / "fneg-fma-ir-probes"


def installed_targets() -> list[str]:
    sysroot = subprocess.run(["rustc", "--print", "sysroot"], check=True,
                             capture_output=True, text=True).stdout.strip()
    base = Path(sysroot) / "lib" / "rustlib"
    return sorted(p.name for p in base.iterdir() if (p / "lib").is_dir())


def pick(arch: str, installed: list[str]) -> str | None:
    host = g.host_triple()
    if host.startswith(arch):
        return host
    linux = f"{arch}-unknown-linux-gnu"
    if linux in installed:
        return linux
    return next((t for t in installed if t.startswith(arch + "-") and "none" not in t), None)


def compile_probe(probe: Path, target: str, out: Path) -> str:
    subprocess.run(["rustc", "--edition", "2021", "--crate-type", "lib", "-C", "opt-level=3",
                    *g.IR_FLAGS[1:], "--emit=llvm-ir", "--target", target,
                    "-o", str(out), str(probe)], check=True, capture_output=True, text=True,
                   cwd=ROOT)
    return out.read_text(encoding="utf-8")


def header(probe: Path, key: str) -> str:
    m = re.search(rf"^// {key}: (\S+)", probe.read_text(encoding="utf-8"), re.M)
    if not m:
        raise AssertionError(f"{probe.name}: no `{key}:` line")
    return m.group(1)


class Probes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        installed = installed_targets()
        cls.targets = {arch: pick(arch, installed) for arch in ("aarch64", "x86_64")}
        missing = [a for a, t in cls.targets.items() if t is None]
        if missing:
            raise AssertionError(
                f"no installed rustc target for {missing}: rustup target add "
                + " ".join(f"{a}-unknown-linux-gnu" for a in missing))
        cls.tmp = tempfile.TemporaryDirectory()
        cls.results: dict[tuple[str, str], g.Analysis] = {}
        for probe in sorted(PROBES.glob("*.rs")):
            where = header(probe, "targets")
            for arch, target in cls.targets.items():
                if where not in ("any", arch):
                    continue
                ll = Path(cls.tmp.name) / f"{target}-{probe.stem}.ll"
                cls.results[(probe.stem, arch)] = g.analyse(compile_probe(probe, target, ll))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_there_are_enough_probes_and_each_ran_somewhere(self):
        probes = sorted(PROBES.glob("*.rs"))
        self.assertGreaterEqual(len(probes), 40)
        ran = {stem for stem, _ in self.results}
        self.assertEqual(sorted(p.stem for p in probes), sorted(ran))

    def test_every_probe_has_its_expected_verdict_on_every_target(self):
        wrong = []
        for (stem, arch), a in sorted(self.results.items()):
            want = header(PROBES / f"{stem}.rs", "expect-ir")
            got = "flag" if a.findings else "clean"
            if want != got:
                wrong.append(f"{stem} on {arch}: expect {want}, got {got} ({a.fma_calls} fma calls)")
        self.assertEqual(wrong, [])

    def test_every_probe_but_the_control_holds_a_fma_call(self):
        # a probe whose fma was optimised away would pass as "clean" for the wrong reason
        empty = [f"{s} on {a}" for (s, a), r in self.results.items()
                 if r.fma_calls == 0 and "no_fma" not in s]
        self.assertEqual(empty, [])

    def test_a_finding_points_at_the_probe_line_that_negates(self):
        for (stem, arch), a in self.results.items():
            for f in a.findings:
                self.assertTrue(f.location.startswith(f"tools/fneg-fma-ir-probes/{stem}.rs:"),
                                f"{stem} on {arch}: {f.location}")
        loc = self.results[("01_bare_unary", "aarch64")].findings[0].location
        self.assertTrue(loc.startswith("tools/fneg-fma-ir-probes/01_bare_unary.rs:5:"), loc)


    def test_library_files_without_rust_src_are_not_taken_for_the_crate(self):
        # without the rust-src component the standard library's DIFile entries are
        # relative names under the remapped /rustc/<hash> directory (seen on CI);
        # derive that form from a real build and check the location still resolves
        # to the probe, not to library/core/src/ops/arith.rs
        target = self.targets["aarch64"]
        probe = PROBES / "03_neg_method.rs"
        ll = Path(self.tmp.name) / "remapped-03.ll"
        text = compile_probe(probe, target, ll)
        # (a toolchain without rust-src already writes this form; one with it writes
        # absolute paths, which are rewritten into it)
        library = r'!DIFile\(filename: "(?:[^"]*?/)?(library/[^"]+)", directory: "[^"]*"'
        remapped = re.sub(
            library,
            r'!DIFile(filename: "\1", directory: "/rustc/0000000000000000000000000000000000000000"',
            text)
        self.assertTrue(re.search(r'!DIFile\(filename: "library/', remapped),
                        "no standard-library DIFile entry in the IR")
        found = g.analyse(remapped).findings
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0].location.startswith("tools/fneg-fma-ir-probes/03_neg_method.rs:"),
                        found[0].location)


class ProbeProof(unittest.TestCase):
    """with no fma call in the library, the 0 findings are made non-vacuous by the shared probes"""

    def probes(self, files: dict[str, str]) -> Path:
        d = Path(self.tmp.name) / "repo"
        (d / "tools/fneg-fma-ir-probes").mkdir(parents=True)
        for name, text in files.items():
            (d / "tools/fneg-fma-ir-probes" / name).write_text(text, encoding="utf-8")
        return d

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    FLAG = "// expect-ir: flag\n// targets: any\npub fn p(a: f32) -> f32 { -a.mul_add(a, a) }\n"
    CLEAN = "// expect-ir: clean\n// targets: any\npub fn p(a: f32) -> f32 { a.mul_add(a, a) }\n"

    def fake(self, verdicts: dict[str, bool]):
        # flagged probes analyse to one finding, clean ones to none
        def compile_ll(probe, out):
            if verdicts[probe.stem]:
                return ('define float @p(float %a) {\n  %f = call float @llvm.fma.f32(float %a, float %a, float %a)\n'
                        '  %n = fneg float %f\n  ret float %n\n}\n')
            return ('define float @p(float %a) {\n  %f = call float @llvm.fma.f32(float %a, float %a, float %a)\n'
                    '  ret float %f\n}\n')
        return compile_ll

    def test_the_expected_verdicts_pass(self):
        d = self.probes({"a.rs": self.FLAG, "b.rs": self.CLEAN})
        problems, ran = g.probe_proof(d, "x86_64-unknown-linux-gnu", self.fake({"a": True, "b": False}))
        self.assertEqual((problems, ran), ([], 2))

    def test_no_probe_run_fails(self):
        problems, ran = g.probe_proof(self.probes({}), "x86_64-unknown-linux-gnu", self.fake({}))
        self.assertEqual(ran, 0)
        self.assertTrue(any("vacuous" in p for p in problems), problems)

    def test_a_probe_for_another_architecture_does_not_count(self):
        d = self.probes({"a.rs": self.FLAG.replace("targets: any", "targets: aarch64")})
        problems, ran = g.probe_proof(d, "x86_64-unknown-linux-gnu", self.fake({"a": True}))
        self.assertEqual(ran, 0)
        self.assertTrue(problems)

    def test_a_wrong_verdict_fails(self):
        d = self.probes({"a.rs": self.FLAG})
        problems, ran = g.probe_proof(d, "x86_64-unknown-linux-gnu", self.fake({"a": False}))
        self.assertEqual(ran, 1)
        self.assertTrue(any("expect flag, got clean" in p for p in problems), problems)

    def test_the_real_probes_hold_on_the_host(self):
        problems, ran = g.probe_proof(ROOT, g.host_triple())
        self.assertGreaterEqual(ran, 40)
        self.assertEqual(problems, [])


class Counts(unittest.TestCase):
    def analysis(self, fma: int, functions: int = 3) -> g.Analysis:
        return g.Analysis(fma_calls=fma, functions=functions)

    def test_a_count_at_or_above_the_floor_passes(self):
        self.assertEqual(g.check_counts(self.analysis(10), 10, 4), [])
        self.assertEqual(g.check_counts(self.analysis(12), 10, 4), [])

    def test_a_count_below_the_floor_fails(self):
        self.assertTrue(any("< recorded floor" in p for p in g.check_counts(self.analysis(9), 10, 4)))

    def test_a_missing_floor_or_header_fails(self):
        self.assertTrue(g.check_counts(self.analysis(10), None, 4))
        self.assertTrue(g.check_counts(self.analysis(10), 10, None))

    def test_zero_passes_only_where_floor_and_sources_are_zero(self):
        self.assertEqual(g.check_counts(self.analysis(0), 0, 0), [])
        self.assertTrue(g.check_counts(self.analysis(0), 0, 3))
        self.assertTrue(g.check_counts(self.analysis(0), 5, 0))

    def test_an_ir_fma_the_library_does_not_show_fails(self):
        # the library's own sources hold no fma call (test-only items are not counted) but the IR has one
        problems = g.check_counts(self.analysis(2), 0, 0)
        self.assertTrue(any("a fma the source does not show" in p for p in problems), problems)
        self.assertEqual(g.check_counts(self.analysis(0), 0, 0), [])

    def test_a_module_without_functions_fails(self):
        self.assertTrue(g.check_counts(self.analysis(0, functions=0), 0, 0))

    def test_an_empty_module_analyses_to_nothing(self):
        a = g.analyse("")
        self.assertEqual((a.functions, a.fma_calls, a.findings), (0, 0, []))


class Floor(unittest.TestCase):
    def test_write_floor_adds_then_replaces_the_entry(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / g.ALLOWLIST
            path.parent.mkdir(parents=True)
            path.write_text("expected-fma-calls = 2\nexpected-src-fma-calls = 1\n", encoding="utf-8")
            g.write_floor(root, "x86_64-unknown-linux-gnu", "p", 7)
            g.write_floor(root, "aarch64-unknown-linux-gnu", "p", 9)
            g.write_floor(root, "x86_64-unknown-linux-gnu", "p", 8)
            m = g.read_manifest(root)
            self.assertEqual(g.floor_for(m, "x86_64-unknown-linux-gnu", "p"), 8)
            self.assertEqual(g.floor_for(m, "aarch64-unknown-linux-gnu", "p"), 9)
            self.assertIsNone(g.floor_for(m, "x86_64-unknown-linux-gnu", "q"))
            self.assertEqual(len(m["ir"]), 2)


class SignMask(unittest.TestCase):
    def test_float_sign_bits_are_a_sign_mask(self):
        for rhs in ("xor i32 %1, -2147483648",
                    "xor i64 %1, -9223372036854775808",
                    "xor <2 x i64> %1, <i64 -9223372034707292160, i64 poison>",
                    "xor <4 x i32> %1, splat (i32 -2147483648)"):
            self.assertTrue(g.is_sign_mask(rhs), rhs)

    def test_other_constants_are_not(self):
        for rhs in ("xor i32 %1, 1", "xor i32 %1, 2147483647", "xor i64 %1, -1", "xor i32 %1, %2"):
            self.assertFalse(g.is_sign_mask(rhs), rhs)


class Cli(unittest.TestCase):
    def test_a_build_without_cargo_target_dir_runs(self):
        # without CARGO_TARGET_DIR the check builds into a fresh temporary directory; it
        # must not ask cargo to clean that directory first (cargo refuses one without a
        # CACHEDIR.TAG), so a local run without the variable completes
        with tempfile.TemporaryDirectory() as d:
            crate = Path(d) / "g3_fixture"
            (crate / "src").mkdir(parents=True)
            (crate / "Cargo.toml").write_text(
                '[package]\nname = "g3_fixture"\nversion = "0.0.0"\nedition = "2021"\n\n[workspace]\n',
                encoding="utf-8")
            (crate / "src" / "lib.rs").write_text(
                "pub fn f(a: f32, b: f32, c: f32) -> f32 { a * b + c }\n", encoding="utf-8")
            (crate / "scripts").mkdir()
            (crate / "scripts" / "no-outer-fneg-of-fma-allowlist.toml").write_text(
                "expected-fma-calls = 0\nexpected-src-fma-calls = 0\n", encoding="utf-8")
            env = {k: v for k, v in __import__("os").environ.items() if k != "CARGO_TARGET_DIR"}
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "fneg_fma_ir_check.py"),
                                "--findings-only"], cwd=crate, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("g3_fixture", r.stdout + r.stderr)
            self.assertIn("0 negated", r.stdout + r.stderr)

    def test_a_shared_cargo_target_dir_rebuilds_on_every_run(self):
        # with CARGO_TARGET_DIR the check cleans the package first: a second run on the same
        # target directory has nothing left to compile and would find no .ll, so it must
        # rebuild and report the same negated fma as the first
        with tempfile.TemporaryDirectory() as d:
            crate = Path(d) / "g3_fixture"
            (crate / "src").mkdir(parents=True)
            (crate / "Cargo.toml").write_text(
                '[package]\nname = "g3_fixture"\nversion = "0.0.0"\nedition = "2021"\n\n[workspace]\n',
                encoding="utf-8")
            (crate / "src" / "lib.rs").write_text(
                "pub fn f(a: f32, b: f32, c: f32) -> f32 { -(a.mul_add(b, c)) }\n", encoding="utf-8")
            (crate / "scripts").mkdir()
            (crate / "scripts" / "no-outer-fneg-of-fma-allowlist.toml").write_text(
                "expected-fma-calls = 1\nexpected-src-fma-calls = 1\n", encoding="utf-8")
            env = dict(__import__("os").environ, CARGO_TARGET_DIR=str(Path(d) / "target"))
            outs = []
            for _ in range(2):
                r = subprocess.run([sys.executable, str(ROOT / "scripts" / "fneg_fma_ir_check.py"),
                                    "--findings-only"], cwd=crate, env=env, capture_output=True,
                                   text=True)
                outs.append(r.stdout + r.stderr)
                self.assertIn("1 negated", outs[-1], outs[-1])
                self.assertNotIn("found 0", outs[-1], outs[-1])

    def test_a_missing_or_empty_ll_fails(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(g.main(["--ll", str(Path(d) / "absent.ll")]), 1)
            empty = Path(d) / "empty.ll"
            empty.write_text("\n", encoding="utf-8")
            self.assertEqual(g.main(["--ll", str(empty)]), 1)


if __name__ == "__main__":
    unittest.main()
