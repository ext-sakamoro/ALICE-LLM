#!/usr/bin/env python3
"""Tests for scripts/entropy_guard.py: the repository passes, each entropy or
clock source in production code is reported, test modules are exempt, and the
allowed counts are exact."""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entropy_guard as eg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


_TREES: list[Path] = []


def tree(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp(prefix="entropy-guard-"))
    _TREES.append(root)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


def allow(*entries: tuple[str, str, int, str]) -> str:
    """scripts/entropy_guard.toml with one [[allow]] per (path, pattern, count, reason)"""
    return "".join(f'[[allow]]\npath = "{p}"\npattern = "{n}"\ncount = {c}\nreason = "{r}"\n\n' for p, n, c, r in entries)


TOML = "scripts/entropy_guard.toml"
CALL = "pub fn f() { let _ = std::time::SystemTime::now(); }\n"


class EntropyGuard(unittest.TestCase):
    def tearDown(self):
        while _TREES:
            shutil.rmtree(_TREES.pop(), ignore_errors=True)

    def problems(self, files):
        return eg.check(tree(files))[0]

    def test_clippy_toml_bans_elapsed(self):
        # `.elapsed()` reads the wall clock exactly like `::now()`, but a
        # lexical guard pattern for it would be too broad (any value named
        # `elapsed` anywhere); clippy's disallowed-methods is type-resolved
        # and carries this one instead -- this canary only confirms the ban
        # is actually present, not absent-by-typo.
        clippy_toml = (ROOT / "clippy.toml").read_text(encoding="utf-8")
        for path in ("std::time::Instant::elapsed", "std::time::SystemTime::elapsed"):
            self.assertIn(
                f'path = "{path}"',
                clippy_toml,
                f"clippy.toml must disallow {path}",
            )

    def test_the_repository_passes(self):
        problems, files, patterns = eg.check(ROOT)
        self.assertEqual(problems, [])
        self.assertGreater(files, 5)
        self.assertEqual(patterns, len(eg.PATTERNS))

    def test_every_source_in_production_code_is_reported(self):
        for call in ["getrandom::getrandom(&mut b)", "rand::rngs::OsRng", "rand::thread_rng()",
                     "rand::random::<u64>()", "XorShift64::from_entropy()", "Rng::try_from_entropy(1.0, 1.0)",
                     "std::time::SystemTime::now()", "std::time::Instant::now()",
                     "std::collections::hash_map::RandomState::new()", "uuid::Uuid::new_v4()"]:
            p = self.problems({"src/a.rs": f"pub fn f() {{ let _ = {call}; }}\n"})
            self.assertEqual(len(p), 1, (call, p))

    def test_a_test_module_is_exempt(self):
        src = "pub fn f() {}\n\n#[cfg(test)]\nmod tests {\n    fn t() {\n        let _ = std::time::Instant::now();\n    }\n}\n"
        self.assertEqual(self.problems({"src/a.rs": src}), [])

    def test_a_cfg_all_test_module_is_exempt(self):
        src = ('pub fn f() {}\n\n#[cfg(all(test, feature = "std"))]\nmod tests {\n'
               '    fn t() {\n        let _ = std::time::Instant::now();\n    }\n}\n')
        self.assertEqual(self.problems({"src/a.rs": src}), [])

    def test_cfg_not_test_is_not_exempt(self):
        # `not(test)` contains the word "test" but means the opposite of it; a
        # regex keyed on the substring "test" would wrongly treat this as a
        # test module and hide a real production call.
        src = ('pub fn f() {}\n\n#[cfg(not(test))]\nmod prod_only {\n'
               '    fn t() {\n        let _ = std::time::Instant::now();\n    }\n}\n')
        p = self.problems({"src/a.rs": src})
        self.assertTrue(any("Instant::now" in x for x in p), p)

    def test_code_after_a_test_module_is_checked_again(self):
        src = "#[cfg(test)]\nmod tests {\n    fn t() {}\n}\n\npub fn f() { let _ = rand::thread_rng(); }\n"
        p = self.problems({"src/a.rs": src})
        self.assertEqual(len(p), 1, p)
        self.assertIn("src/a.rs:6", p[0])

    def test_a_definition_strings_and_comments_are_not_calls(self):
        src = ('pub fn from_entropy() -> u64 { 0 }\n'
               'pub const S: &str = "thread_rng()";\n'
               '// SystemTime::now() is not used here\n')
        self.assertEqual(self.problems({"src/a.rs": src}), [])

    # The allowances are this repository's own data, read from scripts/entropy_guard.toml (the guard
    # itself is shared byte-for-byte, so it cannot carry them). oracle: each expected message is written
    # out here from the rule it states.
    def test_an_allowed_site_needs_its_exact_count(self):
        self.assertEqual(self.problems({"src/a.rs": CALL, TOML: allow(("src/a.rs", "SystemTime::now", 1, "why"))}), [])

    def test_a_call_beyond_the_allowed_count_fails(self):
        two = CALL + "pub fn g() { let _ = std::time::SystemTime::now(); }\n"
        p = self.problems({"src/a.rs": two, TOML: allow(("src/a.rs", "SystemTime::now", 1, "why"))})
        self.assertTrue(any("2 call sites, scripts/entropy_guard.toml allows 1" in x for x in p), p)

    def test_a_count_above_the_calls_fails(self):
        p = self.problems({"src/a.rs": CALL, TOML: allow(("src/a.rs", "SystemTime::now", 2, "why"))})
        self.assertTrue(any("1 call sites, scripts/entropy_guard.toml allows 2" in x for x in p), p)

    def test_an_allowed_entry_needs_a_reason(self):
        p = self.problems({"src/a.rs": CALL, TOML: allow(("src/a.rs", "SystemTime::now", 1, "  "))})
        self.assertTrue(any("has no reason" in x for x in p), p)

    def test_a_stale_allowed_entry_fails(self):
        p = self.problems({"src/a.rs": "pub fn f() {}\n", TOML: allow(("src/a.rs", "OsRng", 1, "why"))})
        self.assertTrue(any("matches nothing" in x for x in p), p)

    def test_a_missing_toml_allows_nothing(self):
        """an absent file is no allowance, not a skip: the call is a finding"""
        p = self.problems({"src/a.rs": CALL})
        self.assertTrue(any("SystemTime::now in production code: src/a.rs:1" in x for x in p), p)

    def test_a_toml_that_does_not_parse_fails(self):
        p = self.problems({"src/a.rs": "pub fn f() {}\n", TOML: "[[allow]\npath = \n"})
        self.assertTrue(any("scripts/entropy_guard.toml is not TOML" in x for x in p), p)

    def test_an_entry_with_an_unknown_pattern_or_key_or_a_bad_count_fails(self):
        for text, want in (
            (allow(("src/a.rs", "rand::thread", 1, "why")), "unknown pattern"),
            (allow(("src/a.rs", "SystemTime::now", 0, "why")), "count must be a positive integer"),
            ('[[allow]]\npath = "src/a.rs"\npattern = "SystemTime::now"\ncount = 1\nreason = "w"\nwhy = "x"\n', "unknown key"),
            ('[[allow]]\npath = "src/a.rs"\npattern = "SystemTime::now"\nreason = "w"\n', "count must be a positive integer"),
            ('[other]\nx = 1\n', "only [[allow]] entries"),
        ):
            p = self.problems({"src/a.rs": CALL, TOML: text})
            self.assertTrue(any(want in x for x in p), (want, p))

    def test_the_same_site_listed_twice_fails(self):
        p = self.problems({"src/a.rs": CALL, TOML: allow(("src/a.rs", "SystemTime::now", 1, "a"),
                                                          ("src/a.rs", "SystemTime::now", 1, "b"))})
        self.assertTrue(any("listed twice" in x for x in p), p)

    def test_an_aliased_clock_is_found(self):
        src = "use std::time::Instant as Clock;\npub fn f() { let _ = Clock::now(); }\n"
        p = self.problems({"src/a.rs": src})
        self.assertTrue(any("Instant::now" in x and "src/a.rs:2" in x for x in p), p)

    def test_a_call_split_over_lines_is_found(self):
        src = "pub fn f() {\n    let _ = std::time::Instant\n        ::now();\n}\n"
        p = self.problems({"src/a.rs": src})
        self.assertTrue(any("Instant::now" in x and "src/a.rs:2" in x for x in p), p)

    def test_random_state_new_is_found(self):
        src = "pub fn f() { let _ = std::collections::hash_map::RandomState::new(); }\n"
        self.assertTrue(any(x.startswith("RandomState") for x in self.problems({"src/a.rs": src})))

    def test_random_state_default_is_found(self):
        # `RandomState::default()` seeds the same OS-randomness hasher as
        # `::new()` but clippy's `disallowed-methods` cannot ban a trait
        # method (`Default::default`) by the implementing type alone, so this
        # guard is the only layer that catches it; a pattern keyed on `::new`
        # specifically lets it straight through.
        src = "pub fn f() { let _ = std::collections::hash_map::RandomState::default(); }\n"
        self.assertTrue(any(x.startswith("RandomState") for x in self.problems({"src/a.rs": src})))

    def test_a_char_literal_brace_does_not_break_the_test_module_skip(self):
        # A char literal that happens to spell a brace (`'{'`) must not be
        # counted as an opening brace: if it were, the `#[cfg(test)] mod t {
        # .. }` skip region's depth tracking would close early (or never),
        # and a production call right after would be silently missed.
        src = (
            "#[cfg(test)]\nmod t {\n    const C: char = '{';\n    fn t() {}\n}\n\n"
            "pub fn f() { let _ = std::time::Instant::now(); }\n"
        )
        p = self.problems({"src/a.rs": src})
        self.assertTrue(any("Instant::now" in x for x in p), p)

    def test_a_qualified_self_clock_call_is_found(self):
        # `<std::time::Instant>::now()` has a `>` between the type name and
        # `::now`, which the plain `Instant\s*::\s*now` pattern does not
        # tolerate; clippy's `disallowed-methods` catches this form, but this
        # guard is meant to be a readable second layer, not a narrower one.
        src = "pub fn f() { let _ = <std::time::Instant>::now(); }\n"
        p = self.problems({"src/a.rs": src})
        self.assertTrue(any("Instant::now" in x for x in p), p)

    def test_a_qualified_self_with_as_trait_is_found(self):
        # `<Type as Trait>::method()`'s `as Trait` sits between the type name
        # and the closing `>`; this source has no plain-form occurrence, so a
        # pattern that only tolerates the bare `<Type>::` shape (no `as
        # Trait`) would miss it entirely rather than falling back to a
        # redundant match elsewhere in the line.
        src = "pub fn f() { let _ = <std::time::Instant as SomeTrait>::now(); }\n"
        p = self.problems({"src/a.rs": src})
        self.assertTrue(any("Instant::now" in x for x in p), p)

    def test_a_raw_string_brace_does_not_break_the_test_module_skip(self):
        # A `#`-delimited raw string with one embedded plain quote before its
        # brace: a naive "pair up quote characters" approach consumes the
        # first `"..."`-looking span (`"say "`) and, finding no second quote
        # to close a span around the rest, leaves `hi{` -- including the
        # brace -- as unstripped code (confirmed against the pre-fix script:
        # `r#"say "hi{"#;` reduces to `r#""hi{"#;`, brace intact).
        src = (
            '#[cfg(test)]\nmod t {\n    const S: &str = r#"say "hi{"#;\n    fn t() {}\n}\n\n'
            "pub fn f() { let _ = std::time::Instant::now(); }\n"
        )
        p = self.problems({"src/a.rs": src})
        self.assertTrue(any("Instant::now" in x for x in p), p)

    def test_a_block_comment_brace_does_not_break_the_test_module_skip(self):
        src = (
            "#[cfg(test)]\nmod t {\n    /* { */\n    fn t() {}\n}\n\n"
            "pub fn f() { let _ = std::time::Instant::now(); }\n"
        )
        p = self.problems({"src/a.rs": src})
        self.assertTrue(any("Instant::now" in x for x in p), p)

    def test_uuid_new_v4_is_found(self):
        src = "pub fn f() { let _ = uuid::Uuid::new_v4(); }\n"
        self.assertTrue(any(x.startswith("Uuid::new_v4") for x in self.problems({"src/a.rs": src})))

    def test_unix_epoch_elapsed_is_found(self):
        src = "pub fn f() { let _ = std::time::UNIX_EPOCH.elapsed(); }\n"
        self.assertTrue(any(x.startswith("UNIX_EPOCH") for x in self.problems({"src/a.rs": src})))

    def test_attributes_between_cfg_test_and_the_module_keep_it_exempt(self):
        src = ("pub fn f() {}\n\n#[cfg(test)]\n// timing only\n#[allow(clippy::disallowed_methods)]\n"
               "mod tests {\n    fn t() { let _ = std::time::Instant::now(); }\n}\n")
        self.assertEqual(self.problems({"src/a.rs": src}), [])

    def test_main_exits_1_when_it_scans_no_file(self):
        root = tree({"README.md": "x\n"})
        saved = eg.ROOT
        try:
            eg.ROOT = root
            with contextlib.redirect_stderr(io.StringIO()) as err, contextlib.redirect_stdout(io.StringIO()):
                code = eg.main()
        finally:
            eg.ROOT = saved
        self.assertEqual(code, 1)
        self.assertIn("compared nothing", err.getvalue())

    def test_no_source_file_compares_nothing(self):
        _, files, _ = eg.check(tree({"README.md": "x\n"}))
        self.assertEqual(files, 0)


if __name__ == "__main__":
    unittest.main()
