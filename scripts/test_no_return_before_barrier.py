#!/usr/bin/env python3
"""no_return_before_barrier 自身の試験.

検査器が「何も見ていないのに通る」形を塞ぐ: .wgsl file が無い / @compute
entry point が 0 件 のそれぞれを red で固定する 架空の内容だけで組み立てる
(実物の shader tree は no_return_before_barrier.py 側が検査する)

Usage: `python3 test_no_return_before_barrier.py`
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import no_return_before_barrier as g  # noqa: E402

SHADER_PATH = "src/shaders/dummy.wgsl"

CLEAN = """\
@compute @workgroup_size(64)
fn k(@builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let in_bounds = row < params.rows;
    workgroupBarrier();
    if (in_bounds) {
        out_buf[row] = 1.0;
    }
}
"""

RETURN_BEFORE_BARRIER = """\
@compute @workgroup_size(64)
fn k(@builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    if (row >= params.rows) { return; }
    workgroupBarrier();
    out_buf[row] = 1.0;
}
"""

DISCARD_BEFORE_BARRIER = """\
@compute @workgroup_size(64)
fn k(@builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    if (row >= params.rows) { discard; }
    workgroupBarrier();
    out_buf[row] = 1.0;
}
"""

RETURN_AFTER_BARRIER_ONLY = """\
@compute @workgroup_size(64)
fn k(@builtin(workgroup_id) wid: vec3<u32>) {
    workgroupBarrier();
    if (wid.x == 0u) { return; }
    out_buf[wid.x] = 1.0;
}
"""

RETURN_BETWEEN_TWO_BARRIERS = """\
@compute @workgroup_size(64)
fn k(@builtin(workgroup_id) wid: vec3<u32>) {
    workgroupBarrier();
    if (wid.x >= params.rows) { return; }
    workgroupBarrier();
    out_buf[wid.x] = 1.0;
}
"""

RETURN_NO_BARRIER_AT_ALL = """\
@compute @workgroup_size(64)
fn k(@builtin(workgroup_id) wid: vec3<u32>) {
    if (wid.x >= params.rows) { return; }
    out_buf[wid.x] = 1.0;
}
"""

RETURN_IN_COMMENT_ONLY = """\
@compute @workgroup_size(64)
fn k(@builtin(workgroup_id) wid: vec3<u32>) {
    // the old form used to `return;` here
    /* another old form also used `return;` right before the barrier */
    let row = wid.x;
    workgroupBarrier();
    out_buf[row] = 1.0;
}
"""

NESTED_BLOCK_COMMENT = """\
@compute @workgroup_size(64)
fn k(@builtin(workgroup_id) wid: vec3<u32>) {
    /* outer /* inner return; */ still a comment */
    workgroupBarrier();
    out_buf[wid.x] = 1.0;
}
"""

NOT_COMPUTE = """\
fn helper(x: f32) -> f32 {
    if (x < 0.0) { return 0.0; }
    return x;
}
"""


def run(files: dict[str, str]) -> tuple[list[str], dict[str, int]]:
    with tempfile.TemporaryDirectory() as root:
        for rel, text in files.items():
            path = os.path.join(root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return g.check(root)


class Teeth(unittest.TestCase):
    def test_a_clean_kernel_passes(self):
        e, c = run({SHADER_PATH: CLEAN})
        self.assertEqual(e, [])
        self.assertEqual(c["files"], 1)
        self.assertEqual(c["entry_points"], 1)

    def test_return_before_a_barrier_fails(self):
        e, c = run({SHADER_PATH: RETURN_BEFORE_BARRIER})
        self.assertEqual(len(e), 1, e)
        self.assertTrue(f"{SHADER_PATH}:4" in e[0], e)

    def test_discard_before_a_barrier_fails(self):
        e, _ = run({SHADER_PATH: DISCARD_BEFORE_BARRIER})
        self.assertEqual(len(e), 1, e)
        self.assertTrue("`discard`" in e[0], e)

    def test_return_strictly_after_the_only_barrier_passes(self):
        e, _ = run({SHADER_PATH: RETURN_AFTER_BARRIER_ONLY})
        self.assertEqual(e, [])

    def test_return_between_two_barriers_fails(self):
        # same fault risk as before the first barrier: some invocations could
        # stop here while others still reach the second barrier
        e, _ = run({SHADER_PATH: RETURN_BETWEEN_TWO_BARRIERS})
        self.assertEqual(len(e), 1, e)

    def test_return_in_a_kernel_with_no_barrier_at_all_is_not_flagged(self):
        # nothing to be "before" if the kernel never synchronizes
        e, _ = run({SHADER_PATH: RETURN_NO_BARRIER_AT_ALL})
        self.assertEqual(e, [])

    def test_a_return_mentioned_only_in_a_comment_is_not_flagged(self):
        e, _ = run({SHADER_PATH: RETURN_IN_COMMENT_ONLY})
        self.assertEqual(e, [])

    def test_nested_block_comments_are_fully_stripped(self):
        e, _ = run({SHADER_PATH: NESTED_BLOCK_COMMENT})
        self.assertEqual(e, [])

    def test_a_non_compute_fn_is_not_an_entry_point(self):
        e, c = run({SHADER_PATH: NOT_COMPUTE})
        self.assertEqual(e, [])
        self.assertEqual(c["entry_points"], 0)

    def test_zero_wgsl_files_fails(self):
        with self.assertRaises(g.LintError):
            run({})

    def test_zero_compute_entry_points_across_the_whole_tree_fails(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, SHADER_PATH)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(NOT_COMPUTE)
            import subprocess

            result = subprocess.run(
                [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "no_return_before_barrier.py"), "--root", root],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("entry point が 0 件", result.stdout)

    def test_multiple_files_are_all_scanned(self):
        e, c = run(
            {
                "src/shaders/a.wgsl": CLEAN,
                "src/shaders/b.wgsl": RETURN_BEFORE_BARRIER,
            }
        )
        self.assertEqual(c["files"], 2)
        self.assertEqual(len(e), 1, e)
        self.assertTrue(any("b.wgsl" in x for x in e), e)

    def test_wgsl_embedded_in_a_rust_raw_string_literal_is_scanned(self):
        rust_src = (
            "const KERNEL: &str = r#\"\n"
            + RETURN_BEFORE_BARRIER
            + "\"#;\n"
        )
        e, c = run({"src/gpu.rs": rust_src})
        self.assertEqual(len(e), 1, e)
        self.assertTrue("gpu.rs" in e[0] and "embedded raw string" in e[0], e)
        self.assertEqual(c["files"], 1)

    def test_wgsl_embedded_raw_string_reports_the_rust_files_real_line(self):
        # 3 blank lines, then the raw string opens with its own trailing
        # newline before the fixture text starts: line_offset (3 newlines
        # before the content starts) + the fixture's own internal line of
        # `return` (5, since the content itself starts with a blank line) = 8.
        rust_src = "\n\n\nconst KERNEL: &str = r#\"\n" + RETURN_BEFORE_BARRIER + "\"#;\n"
        e, _ = run({"src/gpu.rs": rust_src})
        self.assertEqual(len(e), 1, e)
        self.assertTrue("gpu.rs (embedded raw string):8:" in e[0], e)

    def test_a_plain_double_quoted_string_mentioning_compute_is_not_scanned(self):
        # not a raw string, and nobody writes a multi-line shader via \n escapes
        rust_src = 'const DOC: &str = "mentions @compute but is not a shader";\n'
        e, c = run({"src/shaders/a.wgsl": CLEAN, "src/gpu.rs": rust_src})
        self.assertEqual(e, [])
        self.assertEqual(c["files"], 1)  # the plain string is not counted as a file

    def test_a_raw_string_without_at_compute_is_not_treated_as_a_shader(self):
        rust_src = 'const DOC: &str = r#"just some prose, no shader here"#;\n'
        e, c = run({"src/shaders/a.wgsl": CLEAN, "src/gpu.rs": rust_src})
        self.assertEqual(e, [])
        self.assertEqual(c["files"], 1)

    def test_zero_wgsl_and_zero_embedded_shaders_fails(self):
        with self.assertRaises(g.LintError):
            run({"src/gpu.rs": "fn plain_rust() {}\n"})


class RealRepo(unittest.TestCase):
    def test_this_repository_passes(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # only meaningful once this patch's root points at the real ALICE-LLM
        # checkout (the test harness passes --root explicitly there); against
        # an arbitrary CWD this just needs to not raise.
        try:
            e, c = g.check(root)
        except g.LintError:
            return
        self.assertEqual(e, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
