#!/usr/bin/env python3
"""Tests for scripts/run_tests.sh: a cargo-shaped log is replayed through the
script with `cat`, and the exit code and messages are checked for a passing
run, a run in which no test ran, a named file that ran none (or was not built
at all), a failing command, Windows output (backslash paths, CRLF, ANSI
colour), and a per-binary minimum (`name=N`: met, missed because a test is
ignored, a missing binary, and the refused forms `=0`, `=`, a non-number and
an empty name). The script itself is shared byte for byte with ALICE-DB and
is not modified here."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_tests.sh"


def find_bash():
    """Git's bash on Windows (CreateProcess and often PATH find System32's WSL
    launcher first), the first `bash` on PATH elsewhere."""
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            for cand in (Path(git).parents[1] / "bin" / "bash.exe", Path(git).parents[1] / "usr" / "bin" / "bash.exe"):
                if cand.exists():
                    return str(cand)
        found = shutil.which("bash")
        return None if found is None or "system32" in found.lower() else found
    return shutil.which("bash")


BASH = find_bash()


def running(name, sep="/"):
    return f"     Running tests{sep}{name}.rs (target/debug/deps/{name}-0123abcd)"


def result(n, ignored=0):
    return (f"test result: ok. {n} passed; 0 failed; {ignored} ignored; 0 measured; 0 filtered out; "
            "finished in 0.01s")


UNIT = ["     Running unittests src/lib.rs (target/debug/deps/alice_analytics-0123abcd)", result(7)]


class RunTests(unittest.TestCase):
    def setUp(self):
        # not a skip: a run without bash would otherwise report success
        self.assertIsNotNone(BASH, "bash not found: run_tests.sh cannot be exercised")

    def run_log(self, required, lines, *, newline="\n", status=0):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "cargo.log"
            log.write_bytes(newline.join(lines + [""]).encode("utf-8"))
            cmd = f'cat "{log.as_posix()}"; exit {status}'
            p = subprocess.run([BASH, SCRIPT.as_posix(), required, "--", "bash", "-c", cmd],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
        return p.returncode, p.stdout, p.stderr

    def test_a_passing_run_with_every_named_file(self):
        rc, out, err = self.run_log("alpha,beta", UNIT + [running("alpha"), result(3), running("beta"), result(2)])
        self.assertEqual(rc, 0, err)
        self.assertIn("run_tests: 12 tests passed in total, 3 in alpha, 2 in beta", out)
        self.assertEqual(err, "")

    def test_no_test_ran_fails(self):
        rc, out, err = self.run_log("alpha", ["     Running unittests src/lib.rs (x)", result(0)])
        self.assertEqual(rc, 1)
        self.assertIn("run_tests: 0 tests passed in total", out)
        self.assertIn("run_tests: no test ran", err)

    def test_an_empty_log_fails(self):
        rc, _, err = self.run_log("alpha", [])
        self.assertEqual(rc, 1)
        self.assertIn("no test ran", err)

    def test_a_named_file_that_ran_zero_fails(self):
        rc, out, err = self.run_log("alpha,beta", UNIT + [running("alpha"), result(3), running("beta"), result(0)])
        self.assertEqual(rc, 1)
        self.assertIn("0 in beta", out)
        self.assertIn("run_tests: beta did not run any test", err)
        self.assertNotIn("alpha did not", err)

    def test_a_named_file_that_was_not_built_fails(self):
        rc, _, err = self.run_log("alpha,beta", UNIT + [running("alpha"), result(3)])
        self.assertEqual(rc, 1)
        self.assertIn("run_tests: beta did not run any test", err)

    def test_a_similar_name_does_not_count(self):
        # `tests/alpha_extra.rs` is not `tests/alpha.rs`
        rc, _, err = self.run_log("alpha", UNIT + [running("alpha_extra"), result(4)])
        self.assertEqual(rc, 1)
        self.assertIn("alpha did not run any test", err)

    def test_a_failing_command_fails(self):
        rc, _, _ = self.run_log("alpha", UNIT + [running("alpha"), result(3)], status=101)
        self.assertNotEqual(rc, 0)

    def test_windows_output_is_counted(self):
        esc = "\x1b"
        lines = UNIT + [f"     {esc}[1m{esc}[32mRunning{esc}[0m tests\\alpha.rs (target\\debug\\deps\\alpha-0.exe)",
                        result(5)]
        rc, out, err = self.run_log("alpha", lines, newline="\r\n")
        self.assertEqual(rc, 0, err)
        self.assertIn("12 tests passed in total, 5 in alpha", out)


    def test_a_minimum_that_is_met_passes(self):
        rc, out, err = self.run_log("alpha=2", UNIT + [running("alpha"), result(2)])
        self.assertEqual(rc, 0, err)
        self.assertIn("2 in alpha", out)

    def test_a_minimum_missed_because_a_test_is_ignored_fails(self):
        rc, out, err = self.run_log("alpha=2", UNIT + [running("alpha"), result(1, ignored=1)])
        self.assertEqual(rc, 1)
        self.assertIn("1 in alpha", out)
        self.assertIn("run_tests: alpha passed 1 test(s), needs at least 2", err)

    def test_a_missing_binary_with_a_minimum_fails(self):
        rc, _, err = self.run_log("alpha=2", UNIT)
        self.assertEqual(rc, 1)
        self.assertIn("run_tests: alpha did not run any test", err)

    def test_a_mixed_list_checks_each_minimum(self):
        lines = UNIT + [running("alpha"), result(1), running("beta"), result(2)]
        rc, _, err = self.run_log("alpha,beta=3", lines)
        self.assertEqual(rc, 1)
        self.assertIn("beta passed 2 test(s), needs at least 3", err)
        self.assertNotIn("alpha", err)
        rc, _, err = self.run_log("alpha,beta=2", lines)
        self.assertEqual(rc, 0, err)

    def test_refused_minimums_exit_2_before_the_command_runs(self):
        for spec in ("alpha=0", "alpha=", "alpha=x", "alpha=-1", "alpha=01", "=2", "beta,alpha=0"):
            with self.subTest(spec=spec):
                rc, out, err = self.run_log(spec, UNIT + [running("alpha"), result(3)])
                self.assertEqual(rc, 2, err)
                self.assertNotIn("tests passed in total", out)
                self.assertIn("run_tests: ", err)

if __name__ == "__main__":
    unittest.main()
