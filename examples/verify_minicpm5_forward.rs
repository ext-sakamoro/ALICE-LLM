//! Smoke test — run forward on MiniCPM5-2B GGUF (OpenBMB, Apache-2.0).
//!
//! MiniCPM5 は LlamaForCausalLM base + ChatML tokenizer template を採用しており、
//! GGUF metadata の `general.architecture` は `llama` prefix、
//! `tokenizer.chat_template` に `<|im_start|>` を含むため、
//! ALICE-LLM 側は **既存 Llama arch + Qwen2 chat template auto-detect** で
//! そのまま動作する MiniCPM 専用の arch enum 追加は不要
//!
//! 参考 metadata (Q4_K_M 実測、2026-09-12):
//! - general.architecture: llama
//! - llama.block_count: 42
//! - llama.embedding_length: 2048
//! - llama.feed_forward_length: 6144
//! - llama.attention.head_count: 16
//! - llama.attention.head_count_kv: 2 (GQA)
//! - llama.context_length: 131072
//! - llama.rope.freq_base: 5_000_000
//! - tokenizer.ggml.model: gpt2 (BPE)
//! - tokenizer.ggml.pre: minicpm5
//! - vocab_size: 130560
//!
//! Usage:
//! ```
//! cargo run --release --example verify_minicpm5_forward --features gguf,parallel -- \
//!     models/MiniCPM5-2B-Q4_K_M.gguf
//! ```

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::llama3::{Llama3Model, ModelArch};
use std::fs;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/MiniCPM5-2B-Q4_K_M.gguf".to_string());
    eprintln!("Loading {path}");
    let bytes = fs::read(&path).expect("read gguf");
    let gguf = GgufFile::parse(&bytes).expect("parse gguf");
    let mut model = Llama3Model::from_gguf(&gguf).expect("load model");
    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("load tokenizer");
    eprintln!(
        "Model loaded arch={:?}, vocab={}, layers={}, hidden={}, ffn={}, heads={}/{}kv, ctx={}",
        model.config.arch,
        model.config.vocab_size,
        model.config.num_layers,
        model.config.hidden_dim,
        model.config.intermediate_dim,
        model.config.num_heads,
        model.config.num_kv_heads,
        model.config.max_seq_len,
    );

    // Sanity: MiniCPM5-2B canonical spec (OpenBMB 公開値)
    assert_eq!(
        model.config.arch,
        ModelArch::Llama,
        "MiniCPM5 は llama arch fallback で動作する想定"
    );
    assert_eq!(model.config.num_layers, 42, "MiniCPM5-2B layers=42");
    assert_eq!(model.config.hidden_dim, 2048, "MiniCPM5-2B hidden=2048");
    assert_eq!(model.config.num_heads, 16, "MiniCPM5-2B Q heads=16");
    assert_eq!(
        model.config.num_kv_heads, 2,
        "MiniCPM5-2B GQA KV heads=2 (8x compression)"
    );
    assert_eq!(model.config.vocab_size, 130_560, "MiniCPM5-2B vocab=130560");

    // ChatML prompt (MiniCPM5 tokenizer.chat_template と一致)
    let prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
                  <|im_start|>user\nWhat is the capital of Japan? Answer in one word.<|im_end|>\n\
                  <|im_start|>assistant\n";
    let tokens = tokenizer.encode(prompt);
    eprintln!(
        "Prompt tokens ({}): {:?}",
        tokens.len(),
        &tokens[..tokens.len().min(20)]
    );
    assert!(!tokens.is_empty(), "encode returned empty");

    eprintln!("\nPrefill (n-1 tokens)...");
    let start = Instant::now();
    for &tok in &tokens[..tokens.len() - 1] {
        let _ = model.forward(tok);
    }
    let prefill_ms = start.elapsed().as_millis();
    eprintln!("Prefill: {} tokens in {} ms", tokens.len() - 1, prefill_ms);

    eprintln!("\nDecoding 1 token (feeding last prompt token)...");
    let start = Instant::now();
    let logits = model.forward(*tokens.last().unwrap());
    let decode_ms = start.elapsed().as_millis();
    eprintln!("First decode: {decode_ms} ms");

    let finite_count = logits.iter().filter(|v| v.is_finite()).count();
    eprintln!("Finite logits: {finite_count} / {}", logits.len());
    assert_eq!(
        finite_count,
        logits.len(),
        "全 logits が finite であること (NaN/Inf ゼロ)"
    );
    assert_eq!(
        logits.len(),
        model.config.vocab_size,
        "logits 長が vocab_size と一致"
    );

    // Top 5
    let mut idx_val: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    idx_val.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("Top 5:");
    for &(i, v) in idx_val.iter().take(5) {
        eprintln!(
            "  id={i}, logit={v:.4}, text={:?}",
            tokenizer.decode(&[i as u32])
        );
    }
    let (arg, val) = idx_val[0];
    let text = tokenizer.decode(&[arg as u32]);
    eprintln!("Greedy next token: id={arg}, logit={val:.4}, text={text:?}");

    eprintln!("\nGenerating 15 tokens greedily:");
    let mut generated = vec![arg as u32];
    for _ in 0..15 {
        let last = *generated.last().unwrap();
        let logits = model.forward(last);
        let (next, _) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        generated.push(next as u32);
    }
    let full_text = tokenizer.decode(&generated);
    eprintln!("Generated: {full_text:?}");
    eprintln!("\nMiniCPM5-2B smoke test OK.");
}
