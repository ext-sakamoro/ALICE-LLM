#!/usr/bin/env python3
"""Oracle for scripts/workflow_timeouts.py: a job without the key fails, a job with it
or with a job-level `uses:` passes, a step-level timeout does not count, an empty tree fails."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import workflow_timeouts as wt  # noqa: E402

HEAD = "name: x\non: push\njobs:\n"


def run(body: str) -> int:
    with tempfile.TemporaryDirectory() as d:
        Path(d, "a.yml").write_text(HEAD + body, encoding="utf-8")
        return wt.main([d])


class Timeouts(unittest.TestCase):
    def test_a_job_without_the_key_fails(self):
        self.assertEqual(run("  a:\n    runs-on: ubuntu-latest\n    steps:\n      - run: true\n"), 1)

    def test_a_step_level_timeout_does_not_count(self):
        self.assertEqual(run("  a:\n    runs-on: ubuntu-latest\n    steps:\n      - run: true\n"
                             "        timeout-minutes: 5\n"), 1)

    def test_a_job_with_the_key_passes(self):
        self.assertEqual(run("  a:\n    runs-on: ubuntu-latest\n    timeout-minutes: 5\n"), 0)

    def test_a_reusable_workflow_job_is_skipped(self):
        self.assertEqual(run("  a:\n    uses: ./.github/workflows/b.yml\n"), 0)

    def test_one_bad_job_among_good_ones_fails(self):
        self.assertEqual(run("  a:\n    runs-on: x\n    timeout-minutes: 5\n  b:\n    runs-on: x\n"), 1)

    def test_no_job_fails(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(wt.main([d]), 1)

    def test_the_repository_passes(self):
        self.assertEqual(wt.main([]), 0)


if __name__ == "__main__":
    unittest.main()
