//! Phase JJJ v0.2: Rejection-sampling speculative decoding smoke test.
//!
//! Runs `speculative_decode_v2` with temperature / top-p sampling, and
//! verifies that the same seed produces deterministic output.
//!
//! Usage:
//!   verify_speculative_v2 <gguf_path> [temperature] [top_p] [seed]
//!
//! Defaults: temperature=0.7, top_p=0.9, seed=42.

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::llama3::{speculative_decode_v2, Llama3Model, SpeculativeConfig};
use std::fs;
use std::time::Instant;

fn main() {
    let mut args = std::env::args();
    let _bin = args.next();
    let path = args
        .next()
        .unwrap_or_else(|| "models/Llama-3.2-1B-Instruct-IQ4_XS.gguf".to_string());
    let temperature: f32 = args
        .next()
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.7);
    let top_p: f32 = args
        .next()
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.9);
    let seed: u64 = args
        .next()
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    eprintln!("Loading {path}");
    eprintln!("temperature={temperature}, top_p={top_p}, seed={seed}");
    let bytes = fs::read(&path).expect("read gguf");
    let gguf = GgufFile::parse(&bytes).expect("parse gguf");
    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("load tokenizer");
    let eos_id = tokenizer.eos_id;

    let prompt = "The capital of Japan is";
    let tokens = tokenizer.encode(prompt);
    eprintln!(
        "\nPrompt tokens ({}): {:?}",
        tokens.len(),
        &tokens[..tokens.len().min(10)]
    );

    let cfg = SpeculativeConfig {
        n_draft: 4,
        max_new_tokens: 20,
        eos_id: Some(eos_id),
        temperature: Some(temperature),
        top_p: Some(top_p),
        sample_seed: Some(seed),
    };

    // ── Run 1 ─────────────────────────────────────────────────────────────
    eprintln!("\n[run 1] speculative_decode_v2 (draft=main same weights)");
    let mut draft = Llama3Model::from_gguf(&gguf).expect("draft");
    let mut main1 = Llama3Model::from_gguf(&gguf).expect("main");
    let t0 = Instant::now();
    let r1 = speculative_decode_v2(&mut draft, &mut main1, &tokens, &cfg);
    let t1 = t0.elapsed().as_millis();
    let text1 = tokenizer.decode(&r1.tokens);
    eprintln!(
        "Run 1: {t1} ms, {} tokens, acceptance {:.1}% ({}/{}, bonus {})",
        r1.tokens.len(),
        r1.acceptance_rate() * 100.0,
        r1.draft_tokens_accepted,
        r1.draft_tokens_produced,
        r1.bonus_tokens,
    );
    eprintln!("Run 1 text: {text1:?}");

    // ── Run 2 with same seed → must be identical ─────────────────────────
    eprintln!("\n[run 2] same seed — verify determinism");
    let mut draft2 = Llama3Model::from_gguf(&gguf).expect("draft2");
    let mut main2 = Llama3Model::from_gguf(&gguf).expect("main2");
    let r2 = speculative_decode_v2(&mut draft2, &mut main2, &tokens, &cfg);
    let text2 = tokenizer.decode(&r2.tokens);
    eprintln!(
        "Run 2: {} tokens, acceptance {:.1}%",
        r2.tokens.len(),
        r2.acceptance_rate() * 100.0,
    );
    eprintln!("Run 2 text: {text2:?}");

    if r1.tokens == r2.tokens {
        eprintln!("\n[determinism] same seed produces same output");
    } else {
        eprintln!("\n[determinism] MISMATCH — same seed gave different output");
        std::process::exit(2);
    }

    // ── Run 3 with different seed → likely diverges ──────────────────────
    let cfg2 = SpeculativeConfig {
        sample_seed: Some(seed.wrapping_add(1)),
        ..cfg.clone()
    };
    eprintln!("\n[run 3] different seed — expect divergence (informational)");
    let mut draft3 = Llama3Model::from_gguf(&gguf).expect("draft3");
    let mut main3 = Llama3Model::from_gguf(&gguf).expect("main3");
    let r3 = speculative_decode_v2(&mut draft3, &mut main3, &tokens, &cfg2);
    let text3 = tokenizer.decode(&r3.tokens);
    eprintln!(
        "Run 3: {} tokens, acceptance {:.1}%",
        r3.tokens.len(),
        r3.acceptance_rate() * 100.0,
    );
    eprintln!("Run 3 text: {text3:?}");

    if r1.tokens != r3.tokens {
        eprintln!("\n[diversity] different seed yields different output (expected)");
    } else {
        eprintln!(
            "\n[diversity] same output at both seeds — either low-entropy prompt or short cap"
        );
    }

    eprintln!("\nPhase JJJ v0.2 smoke test OK.");
}
