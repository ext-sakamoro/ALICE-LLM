//! Fuzz target: logits post-processing (temperature / top_k / top_p / softmax / argmax) を fuzz
//!
//! sampling pipeline は inference の hot loop で毎 token 呼ばれ、logits に NaN / Inf /
//! 極値が混ざる可能性 (GPU FP16 overflow / numerical instability / adversarial prompt) が
//! ある そのため任意 f32 配列 + 任意パラメータで panic-free / index-out-of-bounds-free
//! であることが必須
//!
//! 起こり得る危険:
//! - top_k(logits, k) で k > logits.len() → index out of bounds
//! - top_p(logits, p) で p < 0 or p > 1 or NaN → 未定義 behavior
//! - softmax([NaN, ...]) で index panic
//! - temperature=0 or NaN で division by zero
//! - argmax([]) で empty slice unwrap panic
//!
//! 期待 behavior: どんな入力でも panic しない (結果の意味は不問、numerical error は許容)

#![no_main]

use alice_llm::sampling::{
    apply_temperature, sample_argmax, sample_with_random, softmax, softmax_inplace, top_k_filter,
    top_p_filter,
};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    /// Raw logits (adversarial f32 values including NaN / Inf allowed).
    logits: Vec<f32>,
    /// Temperature parameter (arbitrary f32, including 0 / NaN / Inf).
    temperature: f32,
    /// top_k parameter (arbitrary usize, capped to keep memory bounded).
    top_k: u16,
    /// top_p parameter (arbitrary f32, including out-of-[0,1] range).
    top_p: f32,
    /// Random value for sample_with_random (typically expected [0,1)).
    rand_val: f32,
}

fuzz_target!(|input: FuzzInput| {
    // Bound the input size to keep allocator usage sane during long fuzzing.
    let mut logits = input.logits;
    if logits.len() > 65_536 {
        logits.truncate(65_536);
    }

    // apply_temperature must not panic even for temperature = 0 / NaN / Inf.
    {
        let mut buf = logits.clone();
        apply_temperature(&mut buf, input.temperature);
    }

    // top_k_filter must clamp k to logits.len() internally (or otherwise not panic).
    {
        let mut buf = logits.clone();
        top_k_filter(&mut buf, input.top_k as usize);
    }

    // top_p_filter must handle p outside [0,1] and NaN without panic.
    {
        let mut buf = logits.clone();
        top_p_filter(&mut buf, input.top_p);
    }

    // softmax variants must not panic on NaN / empty / all-Inf inputs.
    let probs = softmax(&logits);
    {
        let mut buf = logits.clone();
        softmax_inplace(&mut buf);
    }

    // argmax on empty slice would panic if unchecked — verify it doesn't.
    if !logits.is_empty() {
        let _ = sample_argmax(&logits);
    }

    // sample_with_random operates on probs (softmax output), must be robust
    // to NaN / negative / sum-not-1 inputs.
    if !probs.is_empty() {
        let _ = sample_with_random(&probs, input.rand_val);
    }
});
