#!/usr/bin/env python3
"""Tests of scripts/id_move_check.py: the rules on fixed values, then end to end on a
throwaway crate in a throwaway git repository (built with cargo)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import id_move_check as imc  # noqa: E402

A, B, C = "a1" * 32, "b2" * 32, "c3" * 32
CFG = imc.Config("probe-crate", "Cargo.toml", (), True, ("v{version}",),
                 (imc.IdSpec("probe_crate::SEMANTICS_ID", "probe_crate::PREVIOUS_SEMANTICS_IDS"),))


def ids(value, previous=()):
    return {"probe_crate::SEMANTICS_ID": None if value is None else {"id": value, "previous": list(previous)}}


class Rules(unittest.TestCase):
    def test_unmoved_passes(self):
        self.assertEqual(imc.judge("0.3.0", "0.3.1", ids(A), ids(A), "", CFG)[0], [])

    def test_moved_without_a_breaking_version_fails(self):
        problems = imc.judge("0.3.0", "0.3.1", ids(A), ids(B, [A]), f"{A[:8]} -> {B[:8]}", CFG)[0]
        self.assertTrue(any("not a breaking step" in p for p in problems), problems)
        problems = imc.judge("1.2.0", "1.3.0", ids(A), ids(B, [A]), f"{A[:8]} -> {B[:8]}", CFG)[0]
        self.assertTrue(any("not a breaking step" in p for p in problems), problems)

    def test_moved_with_a_breaking_version_previous_and_changelog_passes(self):
        log = f"SEMANTICS_ID moves from {A[:8]}… to {B[:8]}…"
        self.assertEqual(imc.judge("0.3.0", "0.4.0", ids(A), ids(B, [A]), log, CFG)[0], [])
        self.assertEqual(imc.judge("1.9.0", "2.0.0", ids(A), ids(B, [A]), log, CFG)[0], [])

    def test_moved_without_the_base_value_in_previous_fails(self):
        problems = imc.judge("0.3.0", "0.4.0", ids(A), ids(B, [C]), f"{A[:8]} {B[:8]}", CFG)[0]
        self.assertTrue(any("does not hold the base value" in p for p in problems), problems)

    def test_moved_without_the_changelog_naming_both_fails(self):
        problems = imc.judge("0.3.0", "0.4.0", ids(A), ids(B, [A]), f"new id {B[:8]}", CFG)[0]
        self.assertTrue(any("does not name " + A[:8] in p for p in problems), problems)

    def test_new_identifier_passes_and_a_vanished_one_fails(self):
        self.assertEqual(imc.judge("0.3.0", "0.3.1", ids(None), ids(A), "", CFG)[0], [])
        problems = imc.judge("0.3.0", "0.3.1", ids(A), ids(None), "", CFG)[0]
        self.assertTrue(any("not found in the working tree" in p for p in problems), problems)

    def test_a_move_at_the_released_version_says_to_bump_it_first(self):
        problems = imc.judge("0.6.0", "0.6.0", ids(A), ids(B, [A]), f"{A[:8]} {B[:8]}", CFG)[0]
        self.assertTrue(any('bump the version first (set version = "0.7.0"' in p for p in problems), problems)
        problems = imc.judge("3.1.0", "3.1.2", ids(A), ids(B, [A]), f"{A[:8]} {B[:8]}", CFG)[0]
        self.assertTrue(any('set version = "4.0.0"' in p for p in problems), problems)

    def test_a_version_bump_without_a_move_passes(self):
        self.assertEqual(imc.judge("0.6.0", "0.7.0", ids(A), ids(A), "", CFG)[0], [])

    def test_an_unreadable_registry_fails_on_main_and_tags_and_warns_on_ci_branches(self):
        p = imc.unreadable_base_policy
        self.assertEqual(p({"GITHUB_REF": "refs/heads/main", "GITHUB_EVENT_NAME": "push"}), "fail")
        self.assertEqual(p({"GITHUB_REF": "refs/tags/v1.0.0", "GITHUB_EVENT_NAME": "push"}), "fail")
        self.assertEqual(p({}), "fail")
        self.assertEqual(p({"GITHUB_REF": "refs/heads/ci/x", "GITHUB_EVENT_NAME": "push"}), "warn")
        self.assertEqual(p({"GITHUB_REF": "refs/pull/3/merge", "GITHUB_EVENT_NAME": "pull_request"}), "warn")
        self.assertEqual(p({"GITHUB_REF": "refs/heads/main", "GITHUB_EVENT_NAME": "schedule"}), "fail")

    def test_every_32_byte_constant_is_an_identifier_or_exempt(self):
        found = {"SEMANTICS_ID": ["src/lib.rs:1"], "FORMAT_ID": ["src/container.rs:9"]}
        cfg = imc.Config("p", "Cargo.toml", (), True, ("v{version}",), (imc.IdSpec("p::SEMANTICS_ID", None),))
        problems = imc.completeness(cfg, found)
        self.assertTrue(any(p.startswith("FORMAT_ID (src/container.rs:9)") for p in problems), problems)
        ok = imc.Config("p", "Cargo.toml", (), True, ("v{version}",), cfg.ids,
                        (("FORMAT_ID", "a key, not a stamp of what was computed"),))
        self.assertEqual(imc.completeness(ok, found), [])
        stale = imc.Config("p", "Cargo.toml", (), True, ("v{version}",), cfg.ids,
                           (("FORMAT_ID", "a key, not a stamp of what was computed"), ("GONE", "removed long ago, stale")))
        self.assertTrue(any("GONE" in p and "stale" in p for p in imc.completeness(stale, found)))
        self.assertEqual(imc.completeness(cfg, {}), ["no const / static item of type [u8; 32] found in the crate (compared nothing)"])

    def test_breaking_steps(self):
        self.assertTrue(imc.is_breaking("0.10.0", "0.11.0"))
        self.assertTrue(imc.is_breaking("0.10.3", "1.0.0"))
        self.assertTrue(imc.is_breaking("3.1.0", "4.0.0"))
        self.assertFalse(imc.is_breaking("0.10.0", "0.10.1"))
        self.assertFalse(imc.is_breaking("3.1.0", "3.2.0"))
        self.assertFalse(imc.is_breaking("0.11.0", "0.11.0"))

    def test_a_config_without_identifiers_fails_as_comparing_nothing(self):
        d = Path(tempfile.mkdtemp())
        (d / "scripts").mkdir()
        (d / "scripts" / "id_move.toml").write_text('package = "x"\nmanifest = "Cargo.toml"\ntags = ["v{version}"]\n')
        with self.assertRaisesRegex(imc.CheckError, "compared nothing"):
            imc.load_config(d)


LIB = """pub const SEMANTICS_ID: [u8; 32] = [{b}; 32];
pub const PREVIOUS_SEMANTICS_IDS: &[[u8; 32]] = &[{prev}];
"""


@unittest.skipUnless(shutil.which("cargo") and shutil.which("git"), "needs cargo and git")
class EndToEnd(unittest.TestCase):
    """A crate released as v0.3.0 with SEMANTICS_ID = [0x11; 32], then changed in the working tree."""

    @classmethod
    def setUpClass(cls):
        cls.target = Path(tempfile.mkdtemp(prefix="id-move-test-target-"))
        os.environ["ID_MOVE_TARGET_DIR"] = str(cls.target)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("ID_MOVE_TARGET_DIR", None)
        shutil.rmtree(cls.target, ignore_errors=True)

    def repo(self) -> Path:
        d = Path(tempfile.mkdtemp(prefix="id-move-test-"))
        (d / "src").mkdir()
        (d / "scripts").mkdir()
        (d / "scripts" / "id_move.toml").write_text(
            'package = "probe-crate"\nmanifest = "Cargo.toml"\ntags = ["v{version}"]\n'
            '[[id]]\npath = "probe_crate::SEMANTICS_ID"\nprevious = "probe_crate::PREVIOUS_SEMANTICS_IDS"\n')
        self.write(d, "0.3.0", "0x11", "")
        (d / "CHANGELOG.md").write_text("# Changelog\n")
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@t")
        for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "release"], ["tag", "v0.3.0"]):
            subprocess.run(["git", "-C", str(d), *args], check=True, env=env, capture_output=True)
        return d

    def write(self, d: Path, version: str, byte: str, prev: str):
        (d / "Cargo.toml").write_text(f'[package]\nname = "probe-crate"\nversion = "{version}"\nedition = "2021"\n\n[workspace]\n')
        (d / "src" / "lib.rs").write_text(LIB.format(b=byte, prev=prev))

    def test_rustdoc_finds_every_form_of_a_32_byte_item(self):
        d = Path(tempfile.mkdtemp(prefix="id-move-forms-"))
        (d / "src").mkdir()
        (d / "Cargo.toml").write_text('[package]\nname = "forms"\nversion = "0.1.0"\nedition = "2021"\n\n[workspace]\n')
        (d / "src" / "lib.rs").write_text(
            "pub const ZZ_A: [ u8 ; 32 ] = [0; 32];\n"
            "pub const ZZ_B:\n    [u8; 32] = [0; 32];\n"
            "pub(crate) const ZZ_C: [u8; 32] = [0; 32];\n"
            "pub const ZZ_D: &[u8; 32] = &[0; 32];\n"
            "pub static ZZ_E: [u8; 0x20] = [0; 32];\n"
            "pub const ZZ_F: [u8; 32usize] = [0; 32];\n"
            "const N: usize = 32;\npub const ZZ_G: [u8; N] = [0; 32];\n"
            "pub type Digest = [u8; 32];\npub const ZZ_H: Digest = [0; 32];\n"
            "mod inner { #[allow(dead_code)] pub const ZZ_I: [u8; 32] = [0; 32]; }\n"
            "const ZZ_J: [u8; 32] = [0; 32];\n"
            "pub const NOT_16: [u8; 16] = [0; 16];\npub const NOT_I8: [i8; 32] = [0; 32];\n"
            "#[cfg(feature = \"extra\")]\npub const ZZ_K: [u8; 32] = [0; 32];\n"
            "pub struct ZzH;\nimpl ZzH { pub const ID: [u8; 32] = [0; 32]; }\n"
            "pub trait Stamp { const TID: [u8; 32]; }\nimpl Stamp for ZzH { const TID: [u8; 32] = [2; 32]; }\n")
        (d / "Cargo.toml").write_text('[package]\nname = "forms"\nversion = "0.1.0"\nedition = "2021"\n\n'
                                      '[features]\nextra = []\n\n[workspace]\n')
        cfg = imc.Config("forms", "Cargo.toml", (), True, ("v{version}",), (imc.IdSpec("forms::ZZ_A", None),))
        # a re-export of another crate's constant: here `core::f64::consts::PI` stands in for an upstream id
        with open(d / "src" / "lib.rs", "a") as f:
            f.write("pub use core::f64::consts::PI as UPSTREAM_ID;\nuse core::f64::consts::E;\n#[allow(dead_code)] const _E: f64 = E;\n"
                    "pub mod zz_glob { pub use core::f64::consts::*; }\n")
        found = imc.id_constants(d, cfg)
        self.assertEqual(sorted(found), sorted([f"ZZ_{c}" for c in "ABCDEFGHIJK"] + ["ZzH::ID", "ZzH::TID", "Stamp::TID", "UPSTREAM_ID",
                                               "core::f64::consts::*"]))
        self.assertIn("re-export of core::f64::consts::PI", found["UPSTREAM_ID"][0])
        self.assertEqual(found["ZZ_B"], ["src/lib.rs:2"])
        problems = imc.completeness(cfg, found)
        self.assertEqual(len(problems), 15, problems)
        # an associated const is listed by Type::NAME
        listed = imc.Config("forms", "Cargo.toml", (), True, ("v{version}",),
                            (imc.IdSpec("forms::ZZ_A", None), imc.IdSpec("forms::ZzH::ID", None)))
        self.assertFalse(any(p.startswith("ZzH::ID ") for p in imc.completeness(listed, found)))
        # without the feature the gated item is not seen: rustdoc runs with every feature
        narrow = imc.Config("forms", "Cargo.toml", (), True, ("v{version}",), cfg.ids, (), ())
        self.assertNotIn("ZZ_K", imc.id_constants(d, narrow))

    def test_unmoved_passes(self):
        d = self.repo()
        self.write(d, "0.3.1", "0x11", "")
        problems, report = imc.check(d, None, "tag")
        self.assertEqual(problems, [])
        self.assertIn("unchanged (11111111)", report[1])

    def test_moved_without_a_bump_fails(self):
        d = self.repo()
        self.write(d, "0.3.1", "0x22", "[0x11; 32]")
        (d / "CHANGELOG.md").write_text("11111111 -> 22222222\n")
        problems = imc.check(d, None, "tag")[0]
        self.assertTrue(any("not a breaking step" in p for p in problems), problems)

    def test_moved_with_a_bump_previous_and_changelog_passes(self):
        d = self.repo()
        self.write(d, "0.4.0", "0x22", "[0x11; 32]")
        (d / "CHANGELOG.md").write_text("SEMANTICS_ID 11111111… -> 22222222…\n")
        problems, report = imc.check(d, None, "tag")
        self.assertEqual(problems, [])
        self.assertIn("moved 11111111 -> 22222222", report[1])

    def test_moved_with_a_bump_but_no_previous_fails(self):
        d = self.repo()
        self.write(d, "0.4.0", "0x22", "")
        (d / "CHANGELOG.md").write_text("11111111 -> 22222222\n")
        problems = imc.check(d, None, "tag")[0]
        self.assertTrue(any("does not hold the base value" in p for p in problems), problems)

    def test_an_identifier_the_release_did_not_have_is_new(self):
        d = self.repo()
        # rewrite history: the release had no SEMANTICS_ID, and the base reads no previous list
        (d / "src" / "lib.rs").write_text("pub const OTHER: u8 = 1;\n")
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@t")
        for args in (["commit", "-qam", "release without the id"], ["tag", "-f", "v0.3.0"]):
            subprocess.run(["git", "-C", str(d), *args], check=True, env=env, capture_output=True)
        self.write(d, "0.3.1", "0x11", "")
        problems, report = imc.check(d, None, "tag")
        self.assertEqual(problems, [])
        self.assertIn("new since the base", report[1])

    def test_a_base_that_does_not_build_is_an_error_not_a_new_identifier(self):
        d = self.repo()
        (d / "src" / "lib.rs").write_text(LIB.format(b="0x11", prev="") + "pub fn broken() -> u8 { \"x\" }\n")
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@t")
        for args in (["commit", "-qam", "broken release"], ["tag", "-f", "v0.3.0"]):
            subprocess.run(["git", "-C", str(d), *args], check=True, env=env, capture_output=True)
        self.write(d, "0.3.1", "0x22", "")
        with self.assertRaisesRegex(imc.CheckError, "did not run"):
            imc.check(d, None, "tag")

    def test_the_tag_tree_keeps_export_ignored_files_and_leaves_the_index_alone(self):
        d = self.repo()
        # a module the crate needs, marked export-ignore at the release: `git archive` would drop it
        (d / "src" / "lib.rs").write_text("mod extra;\npub use extra::*;\n")
        (d / "src" / "extra.rs").write_text(LIB.format(b="0x11", prev=""))
        (d / ".gitattributes").write_text("src/extra.rs export-ignore\n")
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@t")
        for args in (["add", "-A"], ["commit", "-qm", "release with an export-ignored module"], ["tag", "-f", "v0.3.0"]):
            subprocess.run(["git", "-C", str(d), *args], check=True, env=env, capture_output=True)
        (d / "src" / "extra.rs").write_text(LIB.format(b="0x11", prev=""))
        self.write(d, "0.3.1", "0x11", "")
        before = subprocess.run(["git", "-C", str(d), "status", "--porcelain"], capture_output=True, text=True).stdout
        problems, report = imc.check(d, None, "tag")
        self.assertEqual(problems, [])
        self.assertIn("unchanged (11111111)", report[1])
        after = subprocess.run(["git", "-C", str(d), "status", "--porcelain"], capture_output=True, text=True).stdout
        # the build of the working tree may write its Cargo.lock; nothing else of the repository moves
        self.assertEqual(before, "".join(ln for ln in after.splitlines(True) if "Cargo.lock" not in ln))
        self.assertEqual([p.name for p in d.parent.glob(f".{d.name}.id-move-base-*")], [])

    def test_no_release_tag_fails(self):
        d = self.repo()
        subprocess.run(["git", "-C", str(d), "tag", "-d", "v0.3.0"], check=True, capture_output=True)
        with self.assertRaisesRegex(imc.CheckError, "no release tag"):
            imc.check(d, None, "tag")


if __name__ == "__main__":
    unittest.main()
