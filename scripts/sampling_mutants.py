#!/usr/bin/env python3
"""Breaks the refusals of `sampling` and of its callers in `llama3`, one way at a time; a test must go
red for each.

usage: python3 scripts/sampling_mutants.py
exit 0 if every mutant applied, compiled and made a test report FAILED; exit 1 otherwise.

Run by hand when src/sampling.rs or the sampling of src/llama3.rs changes: each mutant rebuilds the crate. A mutant that does not
compile or no longer applies counts against the result, since it tested nothing. The file is restored
after each mutant, and at the end even after an interrupt.
"""
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src" / "sampling.rs"
LLAMA3 = ROOT / "src" / "llama3.rs"
COMMANDS = [
    ["cargo", "test", "--test", "sampling_refuses"],
    ["cargo", "test", "--lib", "llama3::"],
]
# a failed test or the test process dying
FAILED = re.compile(r"^test .* \.\.\. .*FAILED|signal: \d+", re.M)
# a mutant can make a test run forever: past this it is killed, and that counts as red
TIMEOUT_SECS = 300

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

# (what breaks, old text, new text): in src/llama3.rs
LLAMA3_MUTANTS = [
    ("the replay ignores top-p", "        top_p_filter(&mut scaled, p)?;\n", "        let _ = p;\n"),
    ("the replay does not check its temperature", "    if !(temperature > 0.0 && temperature.is_finite()) {\n        return Err(SamplingError::InvalidParameter);\n    }\n    let mut scaled", "    let mut scaled"),
    ("the token ignores the temperature", "    apply_temperature(&mut logits, temperature);\n    top_k_filter(&mut logits, top_k)?;\n    softmax_inplace", "    top_k_filter(&mut logits, top_k)?;\n    softmax_inplace"),
    ("the token ignores top-k", "    apply_temperature(&mut logits, temperature);\n    top_k_filter(&mut logits, top_k)?;\n    softmax_inplace", "    apply_temperature(&mut logits, temperature);\n    softmax_inplace"),
    ("the token is greedy", "    Ok(sample_with_random(&logits, rng.next_f32())? as u32)\n}", "    let _ = rng.next_f32();\n    Ok(sample_argmax(&logits)? as u32)\n}"),
    ("the seed changes each time", "    Ok(sample_with_random(&logits, rng.next_f32())? as u32)\n}", "    let mut r = Rng64::new(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().subsec_nanos().into());\n    let _ = rng.next_f32();\n    Ok(sample_with_random(&logits, r.next_f32())? as u32)\n}"),
    ("a router sorts with the old comparator", "descending_nan_last(a.1, b.1));\n    indexed.truncate(top_k);\n\n    // Weights use the raw scores", "b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));\n    indexed.truncate(top_k);\n\n    // Weights use the raw scores"),
    ("the other router sorts with the old comparator", "idx_biased.sort_by(|a, b| descending_nan_last(a.1, b.1));", "idx_biased.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));"),
]


def run_tests() -> tuple[int | None, str]:
    """Exit status of the first command that fails, or 0, and the output; None when out of time (the
    whole process group is killed: killing cargo alone would leave the test binary running)."""
    output = ""
    for command in COMMANDS:
        p = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
        )
        try:
            out, _ = p.communicate(timeout=TIMEOUT_SECS)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, _ = p.communicate()
            return None, output + out
        output += out
        if p.returncode != 0:
            return p.returncode, output
    return 0, output


def main() -> int:
    if sys.argv[1:]:
        print(__doc__, file=sys.stderr)
        return 2
    bad = []
    everything = [(SOURCE, m) for m in MUTANTS] + [(LLAMA3, m) for m in LLAMA3_MUTANTS]
    for path, (name, old, new) in everything:
        original = path.read_text(encoding="utf-8")
        if original.count(old) != 1:
            bad.append(name)
            print(f"NOT APPLIED  {name}", flush=True)
            continue
        try:
            path.write_text(original.replace(old, new), encoding="utf-8")
            rc, output = run_tests()
        finally:
            path.write_text(original, encoding="utf-8")
        if "could not compile" in output:
            bad.append(name)
            print(f"BROKEN       {name}", flush=True)
        elif rc is None:
            print(f"red (killed) {name}", flush=True)
        elif rc != 0 and FAILED.search(output):
            print(f"red          {name}", flush=True)
        else:
            bad.append(name)
            print(f"SURVIVED     {name} (rc {rc})", flush=True)
    total = len(everything)
    print(f"{total - len(bad)} of {total} red")
    return 1 if bad or not everything else 0


if __name__ == "__main__":
    sys.exit(main())
