//! True speculative decoding with separate draft model.
//!
//! Uses a small 1B model as draft and 8B model as verifier.
//!
//! Usage:
//!   cargo run --release --example speculative_dual --features "gguf,parallel" -- \
//!     --model models/Llama-3-ELYZA-JP-8B-q4_k_m.gguf \
//!     --draft-model models/llama-3.2-1b-gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
//!     --prompt "日本の首都は" \
//!     --speculative-k 4

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::llama3::Llama3Model;
use std::env;
use std::fs;
use std::time::Instant;

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let model_path = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .expect("Usage: --model <main_model.gguf> --draft-model <draft_model.gguf>");

    let draft_path = args
        .iter()
        .position(|a| a == "--draft-model")
        .and_then(|i| args.get(i + 1))
        .expect("Usage: --draft-model <draft_model.gguf>");

    let prompt = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("日本の首都はどこですか？");

    let max_tokens: usize = parse_arg(&args, "--max-tokens").unwrap_or(100);
    let temperature: f32 = parse_arg(&args, "--temperature").unwrap_or(0.0);
    let _spec_k: usize = parse_arg(&args, "--speculative-k").unwrap_or(4);

    // Load main model
    println!("=== True Speculative Decoding (Dual Model) ===");
    println!();
    println!("Loading main model: {model_path}");
    let load_start = Instant::now();
    let main_data = fs::read(model_path).expect("Failed to read main model");
    let main_gguf = GgufFile::parse(&main_data).expect("Failed to parse main model");
    let tokenizer = GgufTokenizer::from_gguf(&main_gguf).expect("Failed to load tokenizer");
    let mut main_model = Llama3Model::from_gguf(&main_gguf).expect("Failed to load main model");
    let main_ms = load_start.elapsed().as_millis();
    println!(
        "  Main: {} layers, hidden={}, vocab={} ({main_ms}ms)",
        main_model.config.num_layers, main_model.config.hidden_dim, main_model.config.vocab_size
    );

    // Load draft model
    println!("Loading draft model: {draft_path}");
    let draft_start = Instant::now();
    let draft_data = fs::read(draft_path).expect("Failed to read draft model");
    let draft_gguf = GgufFile::parse(&draft_data).expect("Failed to parse draft model");
    let mut draft_model = Llama3Model::from_gguf(&draft_gguf).expect("Failed to load draft model");
    let draft_ms = draft_start.elapsed().as_millis();
    println!(
        "  Draft: {} layers, hidden={}, vocab={} ({draft_ms}ms)",
        draft_model.config.num_layers, draft_model.config.hidden_dim, draft_model.config.vocab_size
    );

    let size_ratio = main_model.config.num_layers as f64 * main_model.config.hidden_dim as f64
        / (draft_model.config.num_layers as f64 * draft_model.config.hidden_dim as f64);
    println!("  Size ratio: {size_ratio:.1}x");
    println!();

    // Format prompt
    let formatted = format!(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    );

    // --- Baseline: main model only ---
    println!("--- Baseline (main model only) ---");
    let baseline = main_model.generate(&tokenizer, &formatted, max_tokens, temperature, 40);
    println!("{}", baseline.text);
    println!(
        "  {} tokens, {:.1} tok/s ({}ms prefill + {}ms decode)",
        baseline.tokens_generated, baseline.tokens_per_sec, baseline.prefill_ms, baseline.decode_ms
    );
    println!();

    // --- Speculative: dual model ---
    for k in [3, 4, 5] {
        println!("--- Speculative K={k} (1B draft → 8B verify) ---");
        let result = main_model.generate_speculative_dual(
            &mut draft_model,
            &tokenizer,
            &formatted,
            max_tokens,
            temperature,
            k,
        );
        println!("{}", result.text);
        println!(
            "  {} tokens, {:.1} tok/s ({}ms prefill + {}ms decode)",
            result.tokens_generated, result.tokens_per_sec, result.prefill_ms, result.decode_ms
        );
        if let Some(stats) = &result.spec_stats {
            let accept_rate = if stats.draft_tokens > 0 {
                stats.accepted_tokens as f64 / stats.draft_tokens as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "  Speculation: {}/{} accepted ({:.0}%), K={}",
                stats.accepted_tokens, stats.draft_tokens, accept_rate, stats.spec_k
            );
            let speedup = result.tokens_per_sec / baseline.tokens_per_sec;
            println!("  Speedup: {speedup:.2}x vs baseline");
        }
        println!();
    }
}
