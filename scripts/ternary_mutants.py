#!/usr/bin/env python3
"""Breaks the ternary refusals and the MoE draft one way at a time; a test must go red for each.

usage: python3 scripts/ternary_mutants.py
exit 0 if every mutant applied, compiled and made a test report FAILED; exit 1 otherwise.

Run by hand when the ternary constructors of src/gguf.rs or `forward_draft` change. A mutant that does
not compile or no longer applies counts against the result. A mutant past the time limit is killed
with its process group and counts as red. Each file is restored after its mutant.
"""
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GGUF = ROOT / "src" / "gguf.rs"
LLAMA3 = ROOT / "src" / "llama3.rs"
TRAINING = ROOT / "src" / "training.rs"
COMMANDS = [
    ["cargo", "test", "--test", "ternary_refuses"],
    ["cargo", "test", "--lib", "a_model_that_produces_nan"],
]
FAILED = re.compile(r"^test .* \.\.\. .*FAILED|signal: \d+", re.M)
TIMEOUT_SECS = 300

CHECK_ROW = """        check_finite(weights)?;
        if !threshold_ratio.is_finite()"""
# (what breaks, file, old text, new text)
MUTANTS = [
    ("a row takes a weight that is not finite", GGUF, CHECK_ROW, "        if !threshold_ratio.is_finite()"),
    ("the position of the bad weight is off", GGUF, "Some(index) => Err(TernaryError::NonFiniteWeight { index }),", "Some(index) => Err(TernaryError::NonFiniteWeight { index: index + 1 }),"),
    ("only NaN counts as not finite", GGUF, "weights.iter().position(|w| !w.is_finite())", "weights.iter().position(|w| w.is_nan())"),
    ("a threshold of NaN allowed", GGUF, "if !threshold_ratio.is_finite() || threshold_ratio < 0.0 {", "if threshold_ratio == f32::INFINITY || threshold_ratio == f32::NEG_INFINITY || threshold_ratio < 0.0 {"),
    ("a negative threshold allowed", GGUF, "if !threshold_ratio.is_finite() || threshold_ratio < 0.0 {", "if !threshold_ratio.is_finite() {"),
    ("a threshold of zero refused", GGUF, "if !threshold_ratio.is_finite() || threshold_ratio < 0.0 {", "if !threshold_ratio.is_finite() || threshold_ratio <= 0.0 {"),
    ("an unknown type reads as zeros", GGUF, "other => return Err(TernaryError::UnsupportedType(other)),", "_ => row_f32.fill(0.0),"),
    ("quantized data length not checked", GGUF, """        if data.len() < expected {
            return Err(TernaryError::LengthMismatch {
                expected,
                got: data.len(),
            });
        }
""", ""),
    ("original weights not checked", GGUF, "        check_finite(original_weights)?;\n", ""),
    ("original weights length not checked", GGUF, """        if original_weights.len() != num_cols {
            return Err(TernaryError::LengthMismatch {
                expected: num_cols,
                got: original_weights.len(),
            });
        }
""", ""),
    ("a block keeps its smallest", GGUF, "candidates.sort_by(|a, b| b.1.total_cmp(&a.1));", "candidates.sort_by(|a, b| a.1.total_cmp(&b.1));"),
    ("row count of original weights not checked", GGUF, """        if original_weights_per_row.len() != dense.rows.len() {
            return Err(TernaryError::LengthMismatch {
                expected: dense.rows.len(),
                got: original_weights_per_row.len(),
            });
        }
""", ""),
    ("matrix weights length not checked", GGUF, """        if weights.len() != expected {
            return Err(TernaryError::LengthMismatch {
                expected,
                got: weights.len(),
            });
        }
""", ""),
    ("the shape product wraps", GGUF, """        let expected = num_rows.saturating_mul(num_cols);
        if weights.len()""", """        let expected = num_rows.wrapping_mul(num_cols);
        if weights.len()"""),
    ("a threshold past f32 is left infinite", TRAINING, """    if threshold.is_finite() {
        return threshold;
    }
""", "    if true {\n        return threshold;\n    }\n"),
    ("the wide threshold has no ratio", TRAINING, "(f64::from(ratio) * (sum / weights.len() as f64)) as f32", "(sum / weights.len() as f64) as f32"),
    ("a scale past f32 is left infinite", GGUF, "            if abs_sum.is_finite() {\n", "            if true {\n"),
    ("a MoE layer is a panic in the ternary form", LLAMA3, """gate_proj: ternarize_weight(
                    layer
                        .gate_proj
                        .as_ref()
                        .ok_or(TernaryError::NoTernaryForm { layer: i })?,""", """gate_proj: ternarize_weight(
                    layer.gate_proj.as_ref().expect("gate_proj"),"""),
    ("a MoE layer is a panic in the sparse form", LLAMA3, """gate_proj: sparsify_weight(
                    layer
                        .gate_proj
                        .as_ref()
                        .ok_or(TernaryError::NoTernaryForm { layer: i })?,""", """gate_proj: sparsify_weight(
                    layer.gate_proj.as_ref().expect("gate_proj"),"""),
    ("the draft path has no MoE", LLAMA3, """            // the same dispatch as `forward`: a MoE layer has no gate_proj
            if layer.ffn_gate_inp().is_some() {""", """            if false {"""),
]


def run(command: list[str]) -> tuple[int | None, str]:
    p = subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    )
    try:
        out, _ = p.communicate(timeout=TIMEOUT_SECS)
        return p.returncode, out
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        return None, out


def main() -> int:
    if sys.argv[1:]:
        print(__doc__, file=sys.stderr)
        return 2
    bad = []
    for name, path, old, new in MUTANTS:
        original = path.read_text(encoding="utf-8")
        if original.count(old) != 1:
            bad.append(name)
            print(f"NOT APPLIED  {name}", flush=True)
            continue
        verdict = None
        try:
            path.write_text(original.replace(old, new), encoding="utf-8")
            for command in COMMANDS:
                rc, output = run(command)
                if "could not compile" in output:
                    verdict = "BROKEN      "
                elif rc is None:
                    verdict = "red (killed)"
                elif rc != 0 and FAILED.search(output):
                    verdict = "red         "
                if verdict:
                    break
        finally:
            path.write_text(original, encoding="utf-8")
        verdict = verdict or "SURVIVED    "
        if not verdict.startswith("red"):
            bad.append(name)
        print(f"{verdict} {name}", flush=True)
    print(f"{len(MUTANTS) - len(bad)} of {len(MUTANTS)} red")
    return 1 if bad or not MUTANTS else 0


if __name__ == "__main__":
    sys.exit(main())
