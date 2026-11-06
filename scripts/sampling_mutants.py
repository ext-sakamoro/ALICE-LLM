#!/usr/bin/env python3
"""Breaks the refusals of `sampling` one way at a time; tests/sampling_refuses.rs must go red.

usage: python3 scripts/sampling_mutants.py
exit 0 if every mutant applied, compiled and made a test report FAILED; exit 1 otherwise.

Run by hand when src/sampling.rs changes: each mutant rebuilds the crate. A mutant that does not
compile or no longer applies counts against the result, since it tested nothing. The file is restored
after each mutant, and at the end even after an interrupt.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src" / "sampling.rs"
COMMAND = ["cargo", "test", "--test", "sampling_refuses"]
FAILED = re.compile(r"^test .* \.\.\. .*FAILED", re.M)

CHECK = "    check_logits(logits)?;\n"
# (what breaks, old text, new text)
MUTANTS = [
    ("top_k does not check", "pub fn top_k_filter(logits: &mut [f32], k: usize) -> Result<(), SamplingError> {\n" + CHECK, "pub fn top_k_filter(logits: &mut [f32], k: usize) -> Result<(), SamplingError> {\n"),
    ("top_k checks after its early return", CHECK + "    if k == 0 || k >= logits.len() {\n        return Ok(());\n    }\n", "    if k == 0 || k >= logits.len() {\n        return Ok(());\n    }\n" + CHECK),
    ("top_p does not check", "pub fn top_p_filter(logits: &mut [f32], p: f32) -> Result<(), SamplingError> {\n" + CHECK, "pub fn top_p_filter(logits: &mut [f32], p: f32) -> Result<(), SamplingError> {\n"),
    ("top_p checks after p >= 1", CHECK + "    if p.is_nan() {\n        return Err(SamplingError::InvalidParameter);\n    }\n    if p >= 1.0 {\n        return Ok(());\n    }\n", "    if p >= 1.0 {\n        return Ok(());\n    }\n" + CHECK),
    ("p NaN allowed", "    if p.is_nan() {\n        return Err(SamplingError::InvalidParameter);\n    }\n", ""),
    ("softmax_inplace does not check", "    check_logits(v)?;\n", ""),
    ("argmax does not check", "pub fn sample_argmax(logits: &[f32]) -> Result<usize, SamplingError> {\n" + CHECK, "pub fn sample_argmax(logits: &[f32]) -> Result<usize, SamplingError> {\n"),
    ("NaN allowed", "l.is_nan() || *l == f32::INFINITY", "*l == f32::INFINITY"),
    ("+inf allowed", "l.is_nan() || *l == f32::INFINITY", "l.is_nan()"),
    ("-inf refused", "l.is_nan() || *l == f32::INFINITY", "!l.is_finite()"),
    ("all masked allowed", "    if !logits.is_empty() && logits.iter().all(|l| *l == f32::NEG_INFINITY) {\n        return Err(SamplingError::AllMasked);\n    }\n", ""),
    ("no tokens is all masked", "!logits.is_empty() && logits.iter().all(|l| *l == f32::NEG_INFINITY)", "logits.iter().all(|l| *l == f32::NEG_INFINITY)"),
    ("probability check gone", "    if let Some(index) = probs.iter().position(|p| !p.is_finite() || *p < 0.0) {\n        return Err(SamplingError::InvalidProbability { index });\n    }\n", ""),
    ("a negative probability allowed", "!p.is_finite() || *p < 0.0", "!p.is_finite()"),
    ("zero probabilities allowed", "    if !probs.is_empty() && probs.iter().all(|p| *p == 0.0) {\n        return Err(SamplingError::AllMasked);\n    }\n", ""),
    ("random NaN allowed", "    if rand_val.is_nan() {\n        return Err(SamplingError::InvalidParameter);\n    }\n", ""),
    ("the fallback is the last token", "Ok(probs.iter().rposition(|&p| p > 0.0).unwrap_or(0))", "Ok(probs.len().saturating_sub(1))"),
    ("NaN first in the order", "(true, false) => std::cmp::Ordering::Greater,\n        (false, true) => std::cmp::Ordering::Less,", "(true, false) => std::cmp::Ordering::Less,\n        (false, true) => std::cmp::Ordering::Greater,"),
    ("the order parts the zeros", "b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal),\n    }", "b.total_cmp(&a),\n    }"),
    ("the order has no NaN case", "(true, true) => std::cmp::Ordering::Equal,", "(true, true) => std::cmp::Ordering::Less,"),
    ("top_k keeps one more", "let kth = k - 1;", "let kth = k.min(logits.len() - 1);"),
    ("top_p keeps one fewer", "cutoff_idx = rank + 1;", "cutoff_idx = rank.max(1);"),
    ("argmax takes the first of a tie", ".max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))", ".max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal).then(std::cmp::Ordering::Greater))"),
]


def main() -> int:
    if sys.argv[1:]:
        print(__doc__, file=sys.stderr)
        return 2
    original = SOURCE.read_text(encoding="utf-8")
    bad = []
    try:
        for name, old, new in MUTANTS:
            if original.count(old) != 1:
                bad.append(name)
                print(f"NOT APPLIED  {name}", flush=True)
                continue
            SOURCE.write_text(original.replace(old, new), encoding="utf-8")
            done = subprocess.run(COMMAND, cwd=ROOT, capture_output=True, text=True, check=False)
            output = done.stdout + done.stderr
            if "could not compile" in output:
                bad.append(name)
                print(f"BROKEN       {name}", flush=True)
            elif done.returncode != 0 and FAILED.search(output):
                print(f"red          {name}", flush=True)
            else:
                bad.append(name)
                print(f"SURVIVED     {name} (rc {done.returncode})", flush=True)
    finally:
        SOURCE.write_text(original, encoding="utf-8")
    print(f"{len(MUTANTS) - len(bad)} of {len(MUTANTS)} red")
    return 1 if bad or not MUTANTS else 0


if __name__ == "__main__":
    sys.exit(main())
