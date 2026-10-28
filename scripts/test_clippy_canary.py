#!/usr/bin/env python3
"""Tests for scripts/clippy_canary.py: the data file and clippy.toml parsers,
skip rows, the rows-equal-entries check, the Cargo.lock dependency injection,
the per-row diagnostic accounting (with the owning-type allowance), the control
function and the compared-nothing failures. All but one test run without cargo;
Main.test_the_real_canary_passes runs the canary on the repository end to end."""
import contextlib
import io
import os
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clippy_canary as cc  # noqa: E402

TOML = """# comment ]
disallowed-methods = [
    { path = "f64::sin", reason = "r" },
    # a commented entry { path = "f64::cos" }
    { path = "std::collections::hash_map::RandomState::new", reason = "r" },
    { path = "other_crate::clock::now", reason = "r" },
]
disallowed-types = [
    { path = "std::collections::hash_map::RandomState", reason = "r" },
]
disallowed-macros = [
    { path = "std::println", reason = "not read" },
]
"""
DATA = ("# header\n\n"
        "disallowed-methods\tf64::sin\tlet _ = f64::sin;\n"
        "disallowed-methods\tstd::collections::hash_map::RandomState::new\t"
        "let _ = std::collections::hash_map::RandomState::new;\n"
        "disallowed-methods\tother_crate::clock::now\tskip\tother_crate is not in Cargo.lock\n"
        "disallowed-types\tstd::collections::hash_map::RandomState\t"
        "#[allow(unused_imports)] use std::collections::hash_map::RandomState as CanaryType;\n")
RS = "std::collections::hash_map::RandomState"
LOCK = """version = 4

[[package]]
name = "my-crate"
version = "0.1.0"

[[package]]
name = "dep-a"
version = "1.2.3"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "dep-a"
version = "1.10.0"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "local-dep"
version = "0.3.0"
"""


def diag(lint, line, path):
    kind = "method" if lint.endswith("methods") else "type"
    return (lint, line, f"use of a disallowed {kind} `{path}`")


class Parsers(unittest.TestCase):
    def test_rows_skip_comments_and_blanks(self):
        rows, problems = cc.parse_rows(DATA)
        self.assertEqual(problems, [])
        self.assertEqual([(r.key, r.path, r.planted) for r in rows],
                         [("disallowed-methods", "f64::sin", True),
                          ("disallowed-methods", RS + "::new", True),
                          ("disallowed-methods", "other_crate::clock::now", False),
                          ("disallowed-types", RS, True)])
        self.assertEqual(rows[0].line, 3)
        self.assertEqual(rows[2].reason, "other_crate is not in Cargo.lock")

    def test_malformed_rows_are_reported(self):
        rows, problems = cc.parse_rows("disallowed-methods\tf64::sin\n"
                                       "disallowed-things\tx\tlet _ = 1;\n")
        self.assertEqual(rows, [])
        self.assertEqual(len(problems), 2, problems)
        self.assertIn("3 tab-separated", problems[0])
        self.assertIn("unknown key", problems[1])

    def test_a_skip_row_with_an_empty_reason_fails(self):
        for text in ("disallowed-methods\tf64::sin\tskip\t\n", "disallowed-methods\tf64::sin\tskip\t  \n",
                     "disallowed-methods\tf64::sin\tskip\n"):
            rows, problems = cc.parse_rows(text)
            self.assertEqual(rows, [], text)
            self.assertEqual(len(problems), 1, problems)
            self.assertIn("reason", problems[0])

    def test_skip_rows_are_not_planted(self):
        rows, _ = cc.parse_rows(DATA)
        lib, ranges = cc.render_lib(rows)
        self.assertNotIn("other_crate", lib)
        self.assertEqual(sorted(ranges), ["canary_0", "canary_1", "canary_3", cc.CONTROL])

    def test_clippy_toml_entries_by_key_in_order(self):
        found = cc.clippy_toml_entries(TOML)
        self.assertEqual(found["disallowed-methods"], ["f64::sin", RS + "::new", "other_crate::clock::now"])
        self.assertEqual(found["disallowed-types"], [RS])
        self.assertNotIn("disallowed-macros", found)

    def test_lints_tables_are_copied(self):
        cargo = '[package]\nname = "x"\n\n[lints.clippy]\na = "allow"\n\n[dev-dependencies]\nb = "1"\n'
        self.assertEqual(cc.lints_tables(cargo), '[lints.clippy]\na = "allow"\n\n')
        self.assertEqual(cc.lints_tables('[package]\nname = "x"\n'), "")

    def test_render_gives_each_row_its_own_lines(self):
        rows, _ = cc.parse_rows(DATA)
        lib, ranges = cc.render_lib(rows)
        lines = lib.splitlines()
        for i, row in enumerate(rows):
            if row.planted:
                lo, hi = ranges[f"canary_{i}"]
                self.assertIn(row.stmt, "\n".join(lines[lo - 1:hi]))
        self.assertIn(cc.CONTROL, ranges)

    def test_type_rows_render_as_a_use_inside_the_function(self):
        rows, _ = cc.parse_rows(DATA)
        lib, ranges = cc.render_lib(rows)
        lo, hi = ranges["canary_3"]
        body = lib.splitlines()[lo - 1:hi]
        self.assertEqual(body[0], "pub fn canary_3() {")
        self.assertEqual(body[1].strip(), f"#[allow(unused_imports)] use {RS} as CanaryType;")

    def test_json_diagnostics_are_filtered_to_disallowed(self):
        out = "\n".join([
            "not json",
            '{"reason":"compiler-artifact"}',
            '{"reason":"compiler-message","message":{"code":{"code":"clippy::disallowed_methods"},'
            '"message":"use of a disallowed method `f64::sin`","spans":[{"is_primary":true,"line_start":3}]}}',
            '{"reason":"compiler-message","message":{"code":{"code":"unused_variables"},'
            '"message":"x","spans":[{"is_primary":true,"line_start":9}]}}',
        ])
        self.assertEqual(cc.disallowed_diagnostics(out),
                         [("clippy::disallowed_methods", 3, "use of a disallowed method `f64::sin`")])


class Dependencies(unittest.TestCase):
    def row(self, path, stmt="let _ = x;"):
        return cc.Row(7, "disallowed-methods", path, stmt)

    def test_builtin_rows_need_no_dependency(self):
        rows, _ = cc.parse_rows(DATA)
        self.assertEqual(cc.dependencies(rows, cc.lock_packages(LOCK)), ("", []))

    def test_an_external_row_pins_the_highest_locked_version(self):
        deps, problems = cc.dependencies([self.row("dep_a::clock::now"), self.row("dep-a::other")],
                                         cc.lock_packages(LOCK))
        self.assertEqual(problems, [])
        self.assertIn('dep-a = "=1.10.0"', deps)
        self.assertIn("Cargo.lock has 1.2.3, 1.10.0; the highest is used", deps)
        self.assertEqual(deps.count("dep-a ="), 1)
        text = cc.manifest("2021", deps, "")
        self.assertIn('[dependencies]\n# dep-a: Cargo.lock has 1.2.3, 1.10.0; the highest is used\n'
                      'dep-a = "=1.10.0"\n', text)
        tomllib.loads(text)

    def test_an_external_row_missing_from_the_lock_fails(self):
        deps, problems = cc.dependencies([self.row("nowhere::f")], cc.lock_packages(LOCK))
        self.assertEqual(deps, "")
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("`nowhere` is not a registry crate in Cargo.lock", problems[0])

    def test_a_local_crate_in_the_lock_cannot_be_pinned(self):
        _, problems = cc.dependencies([self.row("local_dep::f")], cc.lock_packages(LOCK))
        self.assertEqual(len(problems), 1, problems)

    def test_skip_rows_add_no_dependency(self):
        rows = [cc.Row(1, "disallowed-methods", "nowhere::f", None, "reason")]
        self.assertEqual(cc.dependencies(rows, {}), ("", []))

    GIT = "a" * 40
    GIT_LOCK = LOCK + (f'\n[[package]]\nname = "dep-g"\nversion = "0.6.0"\n'
                       f'source = "git+https://example.invalid/dep-g?rev=aaaaaaa#{"a" * 40}"\n')

    def test_a_git_source_is_taken_at_its_locked_commit(self):
        deps, problems = cc.dependencies([self.row("dep_g::Key::generate")], cc.lock_packages(self.GIT_LOCK))
        self.assertEqual(problems, [])
        self.assertIn(f'dep-g = {{ git = "https://example.invalid/dep-g", rev = "{self.GIT}" }}', deps)
        dep = tomllib.loads(cc.manifest("2021", deps, ""))["dependencies"]["dep-g"]
        self.assertEqual(dep, {"git": "https://example.invalid/dep-g", "rev": self.GIT})

    def test_a_registry_source_still_wins_over_a_git_one(self):
        lock = self.GIT_LOCK + ('\n[[package]]\nname = "dep-g"\nversion = "0.5.0"\n'
                                'source = "registry+https://github.com/rust-lang/crates.io-index"\n')
        deps, problems = cc.dependencies([self.row("dep_g::f")], cc.lock_packages(lock))
        self.assertEqual(problems, [])
        self.assertIn('dep-g = "=0.5.0"', deps)

    def test_a_path_source_declared_by_the_root_manifest_is_taken_by_path(self):
        deps, problems = cc.dependencies([self.row("local_dep::f")], cc.lock_packages(LOCK),
                                         {"local_dep": "/r/local-dep"})
        self.assertEqual(problems, [])
        self.assertEqual(tomllib.loads(cc.manifest("2021", deps, ""))["dependencies"]["local-dep"],
                         {"path": "/r/local-dep"})

    def test_the_highest_locked_git_version_is_taken(self):
        lock = self.GIT_LOCK + ('\n[[package]]\nname = "dep-g"\nversion = "0.10.0"\n'
                                f'source = "git+https://example.invalid/dep-g#{"b" * 40}"\n')
        deps, problems = cc.dependencies([self.row("dep_g::f")], cc.lock_packages(lock))
        self.assertEqual(problems, [])
        self.assertIn(f'rev = "{"b" * 40}"', deps)

    def test_a_declared_path_that_cargo_lock_does_not_hold_fails(self):
        deps, problems = cc.dependencies([self.row("dep_r::f")], cc.lock_packages(LOCK), {"dep_r": "/r/dep-r"})
        self.assertEqual(deps, "")
        self.assertEqual(len(problems), 1, problems)

    def test_the_workspace_dependency_table_is_read(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "Cargo.toml").write_text('[workspace]\nmembers = []\n\n[workspace.dependencies]\n'
                                             'local-dep = { path = "vendor/local-dep" }\n', encoding="utf-8")
            got = cc.path_dependencies(root)
        self.assertEqual(got, {"local_dep": str((root / "vendor/local-dep").resolve())})

    def test_main_gives_the_canary_crate_the_declared_path(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "Cargo.toml").write_text('[package]\nname = "my-crate"\nversion = "0.1.0"\n\n'
                                             '[dependencies]\nlocal-dep = { path = "../local-dep" }\n',
                                             encoding="utf-8")
            (root / "Cargo.lock").write_text(LOCK, encoding="utf-8")
            (root / "clippy.toml").write_text('disallowed-methods = ["local_dep::f"]\n', encoding="utf-8")
            calls = root / "calls.tsv"
            calls.write_text("disallowed-methods\tlocal_dep::f\tlet _ = local_dep::f;\n", encoding="utf-8")
            seen = []
            real = cc.run_clippy
            cc.run_clippy = lambda _root, _lib, deps: (seen.append(deps), (1, "", ""))[1]
            try:
                with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                    cc.main(["--calls", str(calls), "--root", str(root)])
            finally:
                cc.run_clippy = real
        self.assertEqual(len(seen), 1, "the dependency check stopped before clippy")
        self.assertEqual(tomllib.loads(seen[0])["dependencies"]["local-dep"],
                         {"path": str((root / "../local-dep").resolve())})

    def test_the_root_manifest_names_its_path_dependencies(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "Cargo.toml").write_text(
                '[package]\nname = "x"\n\n[dependencies]\nlocal-dep = { path = "../local-dep" }\n'
                'reg = "1"\n\n[dev-dependencies]\nother = { path = "crates/other", package = "other-dep" }\n'
                "\n[target.'cfg(unix)'.dependencies]\nunixy = { path = \"../unixy\" }\n", encoding="utf-8")
            got = cc.path_dependencies(root)
        self.assertEqual(got, {"local_dep": str((root / "../local-dep").resolve()),
                               "other_dep": str((root / "crates/other").resolve()),
                               "unixy": str((root / "../unixy").resolve())})


class Equality(unittest.TestCase):
    def setUp(self):
        self.rows, _ = cc.parse_rows(DATA)
        self.entries = cc.clippy_toml_entries(TOML)

    def test_equal_rows_and_entries_pass(self):
        self.assertEqual(cc.equality(self.rows, self.entries), [])

    def test_an_entry_without_a_row_fails_naming_it(self):
        p = cc.equality([r for r in self.rows if r.path != "f64::sin"], self.entries)
        self.assertEqual(p, ["clippy.toml disallowed-methods entry `f64::sin` has no row in the data file"])

    def test_a_skipped_entry_still_counts_as_its_row(self):
        p = cc.equality([r for r in self.rows if r.planted], self.entries)
        self.assertEqual(len(p), 1, p)
        self.assertIn("other_crate::clock::now` has no row", p[0])

    def test_a_duplicated_row_fails_naming_it(self):
        p = cc.equality(self.rows + [cc.Row(99, "disallowed-methods", "f64::sin", "let _ = f64::sin;")],
                        self.entries)
        self.assertEqual(len(p), 1, p)
        self.assertIn("`f64::sin` has 2 rows (data lines 3, 99)", p[0])

    def test_an_extra_row_fails_naming_it(self):
        p = cc.equality(self.rows + [cc.Row(99, "disallowed-methods", "f64::cos", "let _ = f64::cos;")],
                        self.entries)
        self.assertEqual(p, ["data line 99: `f64::cos` is not listed under disallowed-methods in clippy.toml"])

    def test_a_row_under_the_wrong_key_fails(self):
        rows = [cc.Row(r.line, "disallowed-types", r.path, r.stmt) if r.path == "f64::sin" else r
                for r in self.rows]
        p = cc.equality(rows, self.entries)
        self.assertEqual(len(p), 2, p)


class Check(unittest.TestCase):
    def setUp(self):
        self.rows, _ = cc.parse_rows(DATA)
        self.toml = cc.clippy_toml_entries(TOML)
        _, self.ranges = cc.render_lib(self.rows)
        self.new_line = self.ranges["canary_1"][0] + 1
        self.good = [diag("clippy::disallowed_methods", self.ranges["canary_0"][0] + 1, "f64::sin"),
                     diag("clippy::disallowed_methods", self.new_line, RS + "::new"),
                     diag("clippy::disallowed_types", self.new_line, RS),
                     diag("clippy::disallowed_types", self.ranges["canary_3"][0] + 1, RS)]

    def run_check(self, diags=None, code=101, rows=None, toml=None):
        return cc.check(self.rows if rows is None else rows, self.toml if toml is None else toml,
                        code, self.good if diags is None else diags, self.ranges)

    def test_one_diagnostic_per_row_passes(self):
        self.assertEqual(self.run_check(), [])

    def test_a_row_without_its_diagnostic_fails(self):
        p = self.run_check(diags=self.good[:3])
        self.assertEqual(len(p), 1, p)
        self.assertIn("RandomState` got 0", p[0])

    def test_a_diagnostic_naming_another_path_does_not_count(self):
        wrong = self.good[:3] + [diag("clippy::disallowed_types", self.good[3][1], "other::Type")]
        p = self.run_check(diags=wrong)
        self.assertTrue(any("1 disallowed diagnostic(s), 0 naming it" in x for x in p), p)

    def test_two_diagnostics_naming_the_row_fail(self):
        p = self.run_check(diags=self.good + [self.good[0]])
        self.assertTrue(any("f64::sin` got 2" in x for x in p), p)

    def test_clippy_exiting_zero_fails(self):
        p = self.run_check(code=0)
        self.assertTrue(any("exited 0" in x for x in p), p)

    def test_a_diagnostic_in_the_control_fails(self):
        ctl = diag("clippy::disallowed_methods", self.ranges[cc.CONTROL][0] + 1, "f64::sqrt")
        p = self.run_check(diags=self.good + [ctl])
        self.assertTrue(any("control" in x for x in p), p)

    def test_a_stray_diagnostic_fails(self):
        p = self.run_check(diags=self.good + [diag("clippy::disallowed_methods", 1, "f64::cos")])
        self.assertTrue(any("outside any canary" in x for x in p), p)

    def test_a_row_missing_from_clippy_toml_fails_naming_it(self):
        toml = cc.clippy_toml_entries(TOML.replace('{ path = "f64::sin", reason = "r" },', ""))
        p = self.run_check(toml=toml)
        self.assertEqual(len(p), 1, p)
        self.assertIn("`f64::sin` is not listed under disallowed-methods", p[0])

    def test_no_planted_rows_compares_nothing(self):
        p = cc.check([], self.toml, 1, [], {})
        self.assertEqual(p, ["compared nothing: the data file has no planted rows"])
        only_skips = [r for r in self.rows if not r.planted]
        self.assertEqual(cc.check(only_skips, self.toml, 1, [], {}), p)

    def test_empty_clippy_toml_compares_nothing(self):
        p = cc.check(self.rows, cc.clippy_toml_entries(""), 1, [], {})
        self.assertEqual(p, ["compared nothing: clippy.toml has no disallowed-methods / disallowed-types entries"])


class OwningTypeAllowance(unittest.TestCase):
    """A banned method of a banned type also gets the type's diagnostic, because
    calling an inherent method names its type. At most one such extra, and only
    a disallowed_types diagnostic naming a clippy.toml disallowed-types entry
    that is a `::`-segment prefix of the row's path."""

    def scene(self, row_path, type_entries, extras):
        toml = {"disallowed-methods": [row_path], "disallowed-types": type_entries}
        rows = [cc.Row(1, "disallowed-methods", row_path, f"let _ = {row_path};")]
        rows += [cc.Row(2 + i, "disallowed-types", t, f"use {t} as CanaryType;") for i, t in enumerate(type_entries)]
        _, ranges = cc.render_lib(rows)
        line = ranges["canary_0"][0] + 1
        diags = [diag("clippy::disallowed_methods", line, row_path)]
        diags += [(lint, line, msg) for lint, msg in extras]
        diags += [diag("clippy::disallowed_types", ranges[f"canary_{1 + i}"][0] + 1, t)
                  for i, t in enumerate(type_entries)]
        return cc.check(rows, toml, 101, diags, ranges)

    @staticmethod
    def type_msg(path):
        return ("clippy::disallowed_types", f"use of a disallowed type `{path}`")

    @staticmethod
    def method_msg(path):
        return ("clippy::disallowed_methods", f"use of a disallowed method `{path}`")

    def extra_problem(self, p):
        self.assertTrue(any("extra disallowed diagnostic" in x for x in p), p)

    def test_the_owning_type_extra_passes(self):
        self.assertEqual(self.scene(RS + "::new", [RS], [self.type_msg(RS)]), [])

    def test_no_extra_passes_too(self):
        self.assertEqual(self.scene(RS + "::new", [RS], []), [])

    def test_an_unrelated_type_extra_fails(self):
        other = "std::collections::BTreeMap"
        self.extra_problem(self.scene(RS + "::new", [RS, other], [self.type_msg(other)]))

    def test_an_extra_disallowed_methods_naming_another_path_fails(self):
        self.extra_problem(self.scene(RS + "::new", [RS], [self.method_msg("f64::sin")]))

    def test_an_extra_disallowed_methods_naming_the_prefix_fails(self):
        self.extra_problem(self.scene(RS + "::new", [RS], [self.method_msg(RS)]))

    def test_a_type_in_another_module_is_not_a_prefix(self):
        row = "std::collections::hash_set::X::new"
        self.extra_problem(self.scene(row, [RS], [self.type_msg(RS)]))

    def test_a_string_prefix_that_is_not_a_segment_prefix_fails(self):
        row = RS + "X::new"
        self.extra_problem(self.scene(row, [RS], [self.type_msg(RS)]))

    def test_a_prefix_type_that_is_not_in_clippy_toml_fails(self):
        self.extra_problem(self.scene(RS + "::new", [], [self.type_msg(RS)]))

    def test_two_extras_fail(self):
        self.extra_problem(self.scene(RS + "::new", [RS], [self.type_msg(RS), self.type_msg(RS)]))

    def test_the_extra_does_not_replace_the_row_diagnostic(self):
        toml = {"disallowed-methods": [RS + "::new"], "disallowed-types": [RS]}
        rows = [cc.Row(1, "disallowed-methods", RS + "::new", "let _ = x;"),
                cc.Row(2, "disallowed-types", RS, "use x;")]
        _, ranges = cc.render_lib(rows)
        diags = [diag("clippy::disallowed_types", ranges["canary_0"][0] + 1, RS),
                 diag("clippy::disallowed_types", ranges["canary_1"][0] + 1, RS)]
        p = cc.check(rows, toml, 101, diags, ranges)
        self.assertTrue(any("0 naming it" in x for x in p), p)


class Main(unittest.TestCase):
    def run_main(self, args):
        err, out = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            rc = cc.main(args)
        return rc, out.getvalue(), err.getvalue()

    def test_an_empty_data_file_fails_before_cargo(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "calls.tsv"
            f.write_text("# only a comment\n", encoding="utf-8")
            rc, _, err = self.run_main(["--calls", str(f)])
        self.assertEqual(rc, 1)
        self.assertIn("compared nothing: the data file has no planted rows", err)

    def test_the_repository_rows_equal_clippy_toml(self):
        rows, problems = cc.parse_rows(cc.CALLS.read_text(encoding="utf-8"))
        self.assertEqual(problems, [])
        self.assertGreater(sum(r.planted for r in rows), 0)
        toml = cc.clippy_toml_entries((cc.ROOT / "clippy.toml").read_text(encoding="utf-8"))
        self.assertEqual(cc.equality(rows, toml), [])

    def test_the_real_canary_passes(self):
        rc, out, err = self.run_main([])
        self.assertEqual(rc, 0, err)
        self.assertRegex(out, r"^clippy_canary: \d+ planted, \d+ skipped, \d+ clippy\.toml entries")


if __name__ == "__main__":
    unittest.main()
