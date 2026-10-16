#!/usr/bin/env python3
"""Tests for scripts/g1_debt_pin.py.

The fixture test builds a throwaway crate with the repository's clippy.toml and
rust-toolchain.toml and lints it with the pin's own command: every way source
can silence the lint (#[allow], an #[expect] on a lint group, cfg_attr, an
inner #![allow], #[allow(warnings)]) must still be reported."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_debt_pin as p  # noqa: E402

FIXTURE = '''#[allow(clippy::disallowed_methods, reason = "fixture")]
pub fn via_allow(x: f32) -> f32 { x.sin() }
#[expect(clippy::style, reason = "fixture")]
pub fn via_group_expect(x: f32) -> f32 { x.cos() }
#[cfg_attr(all(), expect(clippy::disallowed_methods, reason = "fixture"))]
pub fn via_cfg_attr(x: f32) -> f32 { x.tan() }
pub mod inner {
    #![allow(clippy::disallowed_methods)]
    pub fn via_inner_allow(x: f32) -> f32 { x.atan() }
}
#[allow(warnings)]
pub fn via_allow_warnings(x: f32) -> f32 { x.asin() }
#[expect(clippy::disallowed_methods, reason = "fixture")]
pub fn via_expect(x: f32) -> f32 { x.exp() }
pub fn plain(x: f32) -> f32 { x.ln() }
'''
# (file, line, method) of each call; one per silencing form plus a plain call
EXPECTED = {("src/lib.rs", 2, "f32::sin"), ("src/lib.rs", 4, "f32::cos"),
            ("src/lib.rs", 6, "f32::tan"), ("src/lib.rs", 9, "f32::atan"),
            ("src/lib.rs", 12, "f32::asin"), ("src/lib.rs", 14, "f32::exp"),
            ("src/lib.rs", 15, "f32::ln")}


class ForceWarnFixture(unittest.TestCase):
    def test_every_silencing_form_is_still_reported(self):
        with tempfile.TemporaryDirectory(prefix="g1-pin-fixture-") as tmp:
            crate = Path(tmp)
            (crate / "src").mkdir()
            (crate / "src" / "lib.rs").write_text(FIXTURE, encoding="utf-8")
            (crate / "Cargo.toml").write_text(
                '[package]\nname = "g1_pin_fixture"\nversion = "0.0.0"\nedition = "2021"\n\n[workspace]\n',
                encoding="utf-8")
            for name in ("clippy.toml", "rust-toolchain.toml"):
                shutil.copy2(p.ROOT / name, crate / name)
            env = dict(os.environ)
            env["RUSTFLAGS"] = ""
            env["CARGO_TARGET_DIR"] = str(crate / "target")
            cmd = p.clippy_command(p.TARGETS[0], [], ["--lib"])
            proc = subprocess.run(cmd, cwd=crate, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, encoding="utf-8", check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            found = p.parse_sites(proc.stdout.splitlines())
            self.assertEqual({(f, ln, m) for f, ln, _col, m in found}, EXPECTED)


ACCEPTED_FORM = '''#[expect(
    clippy::disallowed_methods,
    reason = "G1 debt: a \\"quoted\\" reason, with [brackets] and (parens)"
)]
fn ok(x: f32) -> f32 { x.exp() }
'''


class Marking(unittest.TestCase):
    def refused(self, src: str) -> list:
        return p.marking_violations(src)[0]

    def test_the_expected_form_is_accepted(self):
        self.assertEqual(p.marking_violations(ACCEPTED_FORM), ([], 1))

    def test_every_other_form_is_refused(self):
        forms = [
            '#[allow(clippy::disallowed_methods, reason = "x")]\nfn f() {}\n',
            '#[allow(clippy::disallowed_methods)]\nfn f() {}\n',
            '#[expect(clippy::style, reason = "x")]\nfn f() {}\n',
            '#[allow(clippy::all)]\nfn f() {}\n',
            '#[allow(warnings)]\nfn f() {}\n',
            '#[cfg_attr(all(), expect(clippy::disallowed_methods, reason = "x"))]\nfn f() {}\n',
            '#[cfg_attr(test, allow(clippy::disallowed_methods))]\nfn f() {}\n',
            'mod m {\n    #![allow(clippy::disallowed_methods)]\n}\n',
            '#![expect(clippy::disallowed_methods, reason = "x")]\n',
            '#[expect(clippy::disallowed_methods, clippy::float_cmp, reason = "x")]\nfn f() {}\n',
        ]
        for src in forms:
            with self.subTest(src=src):
                self.assertEqual(len(self.refused(src)), 1)

    def test_level_raising_attributes_are_ignored(self):
        src = '#![warn(clippy::all, clippy::pedantic)]\n#[deny(clippy::disallowed_methods)]\nfn f() {}\n'
        self.assertEqual(p.marking_violations(src), ([], 0))

    def test_strings_and_comments_are_not_attributes(self):
        src = ('// #[allow(clippy::disallowed_methods)]\n'
               '/// #[allow(clippy::disallowed_methods)]\n'
               '/* #[allow(clippy::disallowed_methods)] */\n'
               'const S: &str = "#[allow(clippy::disallowed_methods)]";\n'
               'const R: &str = r#"#[allow(clippy::disallowed_methods)]"#;\n')
        self.assertEqual(p.marking_violations(src), ([], 0))

    def test_line_of_a_refused_attribute(self):
        self.assertEqual(self.refused('fn a() {}\n\n#[allow(warnings)]\nfn f() {}\n')[0][0], 3)


class Command(unittest.TestCase):
    def test_force_warn_is_passed_to_rustc(self):
        cmd = p.clippy_command("x86_64-unknown-linux-gnu", ["--features", "a,b"], ["--all-targets"])
        self.assertEqual(cmd[cmd.index("--"):], ["--", "--force-warn", p.LINT])
        self.assertIn("--message-format=json", cmd)
        self.assertEqual(cmd[cmd.index("--target") + 1], "x86_64-unknown-linux-gnu")


class Features(unittest.TestCase):
    def test_reads_every_clippy_feature_set(self):
        yml = '''
      - run: cargo clippy --all-targets -- -D warnings
      - run: cargo clippy --all-targets --features simd -- -D warnings
        run: cargo clippy --all-targets --features "a,b,c" -- -D warnings
      - run: cargo test --features "not,clippy"
      - run: cargo clippy --all-targets --features simd -- -D warnings
'''
        self.assertEqual(p.feature_sets(yml), [[], ["--features", "simd"], ["--features", "a,b,c"]])

    def test_no_clippy_command_gives_the_default_set_only(self):
        self.assertEqual(p.feature_sets("jobs: {}\n"), [[]])


def diag(file: str, line: int, col: int, method: str, code: str = p.LINT) -> str:
    return json.dumps({"reason": "compiler-message", "message": {
        "code": {"code": code}, "message": f"use of a disallowed method `{method}`",
        "spans": [{"is_primary": True, "file_name": file, "line_start": line, "column_start": col}]}})


class Sites(unittest.TestCase):
    def test_parse_keeps_only_the_lint(self):
        lines = [diag("src/a.rs", 3, 5, "f32::exp"), diag("src/a.rs", 3, 5, "f32::exp"),
                 diag("src/a.rs", 4, 1, "f32::exp", code="clippy::float_cmp"), "not json"]
        self.assertEqual(p.parse_sites(lines), {("src/a.rs", 3, 5, "f32::exp")})

    def test_list_round_trip(self):
        rows = {("src/b.rs", 10, 2, "f32::tanh"), ("src/a.rs", 3, 5, "f32::exp")}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sites.tsv"
            p.write_list(path, rows, "test")
            self.assertEqual(p.read_list(path), rows)
            self.assertTrue(path.read_text(encoding="utf-8").startswith("# Generated"))


if __name__ == "__main__":
    unittest.main()
