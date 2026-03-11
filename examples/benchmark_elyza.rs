//! ELYZA-JP-8B benchmark: Q4_K_M GGUF inference quality & speed.
//!
//! Usage:
//!   cargo run --release --example benchmark_elyza --features gguf -- \
//!     --model models/elyza-gguf/Llama-3-ELYZA-JP-8B-q4_k_m.gguf

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::llama3::Llama3Model;
use std::env;
use std::fs;
use std::time::Instant;

const PROMPTS: &[&str] = &[
    "日本の首都はどこですか？",
    "Rustプログラミング言語の特徴を3つ挙げてください。",
    "量子コンピュータとは何ですか？簡潔に説明してください。",
    "東京タワーとスカイツリーの高さの違いは？",
    "「吾輩は猫である」の作者は誰ですか？",
    "機械学習と深層学習の違いを説明してください。",
    "日本の四季について英語で説明してください。",
    "SDFとは何ですか？3Dグラフィックスの文脈で説明してください。",
    "1から100までの素数を列挙してください。",
    "HTTPとHTTPSの違いを説明してください。",
];

fn main() {
    let args: Vec<String> = env::args().collect();

    let model_path = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .expect("Usage: --model <path.gguf>");

    let max_tokens: usize = args
        .iter()
        .position(|a| a == "--max-tokens")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  ALICE-LLM Benchmark: ELYZA-JP-8B Q4_K_M               ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Load model
    println!("Loading model: {model_path}");
    let load_start = Instant::now();
    let data = fs::read(model_path).expect("Failed to read GGUF file");
    let file_size_mb = data.len() as f64 / 1e6;
    println!("  File size: {file_size_mb:.1} MB");

    let gguf = GgufFile::parse(&data).expect("Failed to parse GGUF");
    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");
    let mut model = Llama3Model::from_gguf(&gguf).expect("Failed to load model");
    let load_ms = load_start.elapsed().as_millis();
    println!("  Loaded in {load_ms}ms");
    println!("  Config: hidden={}, heads={}, kv_heads={}, layers={}",
        model.config.hidden_dim, model.config.num_heads,
        model.config.num_kv_heads, model.config.num_layers);
    println!("  Vocab: {}", tokenizer.vocab_size());
    println!();

    // Run benchmark
    println!("Running {}/{} prompts, max_tokens={max_tokens}", PROMPTS.len(), PROMPTS.len());
    println!("{}", "=".repeat(60));

    let mut total_tokens = 0usize;
    let mut total_decode_ms = 0u64;
    let mut total_prefill_ms = 0u64;
    let mut results = Vec::new();

    for (i, prompt) in PROMPTS.iter().enumerate() {
        let formatted = format!(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );

        let result = model.generate(&tokenizer, &formatted, max_tokens, 0.0, 1);

        println!("\n[{}/{}] {prompt}", i + 1, PROMPTS.len());
        println!("  → {}", result.text.chars().take(200).collect::<String>());
        println!(
            "  {} tok, {:.1} tok/s, prefill={}ms, decode={}ms",
            result.tokens_generated, result.tokens_per_sec,
            result.prefill_ms, result.decode_ms
        );

        total_tokens += result.tokens_generated;
        total_decode_ms += result.decode_ms;
        total_prefill_ms += result.prefill_ms;
        results.push(result);
    }

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("SUMMARY");
    println!("{}", "=".repeat(60));
    println!("  Model: ELYZA-JP-8B Q4_K_M ({file_size_mb:.0} MB)");
    println!("  Quantization: Q4_K_M (4-bit block)");
    println!("  Engine: ALICE-LLM (pure Rust, fused dequant+matvec)");
    println!("  Prompts: {}", PROMPTS.len());
    println!("  Total tokens generated: {total_tokens}");
    println!("  Total prefill: {total_prefill_ms}ms");
    println!("  Total decode: {total_decode_ms}ms");

    let avg_tok_per_sec = if total_decode_ms > 0 {
        total_tokens as f64 / (total_decode_ms as f64 / 1000.0)
    } else {
        0.0
    };
    println!("  Average speed: {avg_tok_per_sec:.2} tok/s");

    let per_prompt_stats: Vec<String> = results
        .iter()
        .map(|r| format!("{:.1}", r.tokens_per_sec))
        .collect();
    println!("  Per-prompt tok/s: [{}]", per_prompt_stats.join(", "));
}
