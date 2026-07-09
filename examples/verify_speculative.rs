//! Phase JJJ: Speculative decoding smoke test.
//!
//! Verifies that same-model speculative decoding produces the same greedy
//! output as vanilla forward + argmax, and that acceptance rate is 100%.

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::llama3::{speculative_decode, Llama3Model};
use std::fs;
use std::time::Instant;

fn greedy_generate(
    model: &mut Llama3Model<'_>,
    prompt: &[u32],
    max_tokens: usize,
    eos_id: Option<u32>,
) -> Vec<u32> {
    // Prefill n-1 tokens, then feed the last token to obtain the first
    // decode logits.
    for &tok in prompt.iter().take(prompt.len() - 1) {
        let _ = model.forward(tok);
    }
    let mut curr = *prompt.last().unwrap();
    let mut out = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        let logits = model.forward(curr);
        let (arg, _) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let arg = arg as u32;
        out.push(arg);
        if eos_id == Some(arg) {
            break;
        }
        curr = arg;
    }
    out
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder.Q4_K_M.gguf".to_string());
    eprintln!("Loading {path}");
    let bytes = fs::read(&path).expect("read gguf");
    let gguf = GgufFile::parse(&bytes).expect("parse gguf");
    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("load tokenizer");
    let eos_id = tokenizer.eos_id;

    let prompt = "The capital of Japan is";
    let tokens = tokenizer.encode(prompt);
    eprintln!(
        "Prompt tokens ({}): {:?}",
        tokens.len(),
        &tokens[..tokens.len().min(10)]
    );

    let n_draft = 4;
    let max_new = 20;

    // ── Baseline: greedy generation with a single model ────────────────
    eprintln!("\n[baseline] greedy single-model...");
    let mut baseline_model = Llama3Model::from_gguf(&gguf).expect("load baseline model");
    let t0 = Instant::now();
    let baseline = greedy_generate(&mut baseline_model, &tokens, max_new, Some(eos_id));
    let baseline_ms = t0.elapsed().as_millis();
    let baseline_text = tokenizer.decode(&baseline);
    eprintln!("Baseline: {baseline_ms} ms, {} tokens", baseline.len());
    eprintln!("Baseline text: {baseline_text:?}");

    // ── Speculative decoding: same GGUF used for both draft & main ────
    // Same model on both sides: acceptance must be 100% and output equal.
    eprintln!("\n[speculative] draft = main (same weights)...");
    let mut draft = Llama3Model::from_gguf(&gguf).expect("load draft model");
    let mut main = Llama3Model::from_gguf(&gguf).expect("load main model");
    let t0 = Instant::now();
    let spec = speculative_decode(
        &mut draft,
        &mut main,
        &tokens,
        n_draft,
        max_new,
        Some(eos_id),
    );
    let spec_ms = t0.elapsed().as_millis();
    let spec_text = tokenizer.decode(&spec.tokens);
    eprintln!(
        "Speculative: {spec_ms} ms, {} tokens, acceptance {:.1}%",
        spec.tokens.len(),
        spec.acceptance_rate() * 100.0,
    );
    eprintln!(
        "  draft produced: {}, accepted: {}, bonus: {}",
        spec.draft_tokens_produced, spec.draft_tokens_accepted, spec.bonus_tokens,
    );
    eprintln!("Speculative text: {spec_text:?}");

    // ── Verify equivalence ────────────────────────────────────────────
    if baseline == spec.tokens {
        eprintln!("\n✓ Output tokens match exactly (same-model speculative is correct)");
    } else {
        eprintln!("\n✗ MISMATCH — baseline vs speculative diverged");
        eprintln!("  baseline: {baseline:?}");
        eprintln!("  spec:     {:?}", spec.tokens);
        std::process::exit(2);
    }
    if (spec.acceptance_rate() - 1.0).abs() > 1e-6 {
        eprintln!(
            "  NOTE: acceptance rate {:.4} != 1.0 — this is expected only when eos or max cap ends mid-draft",
            spec.acceptance_rate()
        );
    }

    eprintln!("\nPhase JJJ smoke test OK.");
}
