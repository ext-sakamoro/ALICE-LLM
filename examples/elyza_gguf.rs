//! ELYZA-JP-8B GGUF inference example.
//!
//! Usage:
//!   cargo run --example elyza_gguf --features gguf -- \
//!     --model path/to/Llama-3-ELYZA-JP-8B-q4_k_m.gguf \
//!     --prompt "日本の首都はどこですか？"
//!
//! Download model:
//!   huggingface-cli download elyza/Llama-3-ELYZA-JP-8B-GGUF \
//!     Llama-3-ELYZA-JP-8B-q4_k_m.gguf --local-dir models/

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

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let model_path = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .expect("Usage: --model <path.gguf>");

    let prompt = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("日本の首都はどこですか？");

    let max_tokens: usize = parse_arg(&args, "--max-tokens").unwrap_or(256);
    let temperature: f32 = parse_arg(&args, "--temperature").unwrap_or(0.7);
    let spec_k: usize = parse_arg(&args, "--speculative-k").unwrap_or(0);
    let draft_layers: usize = parse_arg(&args, "--draft-layers").unwrap_or(8);
    let ternary = has_flag(&args, "--ternary");
    let ternary_threshold: f32 = parse_arg(&args, "--ternary-threshold").unwrap_or(0.7);

    println!("Loading GGUF model: {model_path}");
    let load_start = Instant::now();

    let data = fs::read(model_path).expect("Failed to read GGUF file");
    let gguf = GgufFile::parse(&data).expect("Failed to parse GGUF");

    println!(
        "  Tensors: {}, Metadata keys: {}",
        gguf.tensors.len(),
        gguf.metadata.len()
    );

    if let Some(arch) = gguf.meta_str("general.architecture") {
        println!("  Architecture: {arch}");
    }
    if let Some(name) = gguf.meta_str("general.name") {
        println!("  Model name: {name}");
    }

    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");
    println!("  Vocab size: {}", tokenizer.vocab_size());

    let mut model = Llama3Model::from_gguf(&gguf).expect("Failed to load model");
    let load_ms = load_start.elapsed().as_millis();
    println!("  Model loaded in {load_ms}ms");
    println!("  Config: {:?}", model.config);

    if ternary {
        println!("  Ternarizing weights (threshold={ternary_threshold})...");
        let t_start = Instant::now();
        model.load_ternary(ternary_threshold);
        println!("  Ternarized in {}ms", t_start.elapsed().as_millis());
    }
    println!();

    // Format as Llama-3 instruct
    let formatted = format!(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    );

    println!("Prompt: {prompt}");
    if ternary {
        println!("Generating [TERNARY] (max {max_tokens} tokens, temp={temperature})...");
    } else if spec_k > 0 {
        println!(
            "Generating (max {max_tokens} tokens, temp={temperature}, speculative K={spec_k}, draft_layers={draft_layers})..."
        );
    } else {
        println!("Generating (max {max_tokens} tokens, temp={temperature})...");
    }
    println!("---");

    let result = if ternary {
        model.generate_ternary(&tokenizer, &formatted, max_tokens, temperature, 40)
    } else if spec_k > 0 {
        model.generate_speculative(&tokenizer, &formatted, max_tokens, temperature, 40, spec_k, draft_layers)
    } else {
        model.generate(&tokenizer, &formatted, max_tokens, temperature, 40)
    };

    println!("{}", result.text);
    println!("---");
    println!(
        "Tokens: {} generated, {} prompt",
        result.tokens_generated, result.prompt_tokens
    );
    println!(
        "Speed: {:.1} tok/s ({} prefill + {} decode = {} total ms)",
        result.tokens_per_sec, result.prefill_ms, result.decode_ms, result.total_ms
    );

    if let Some(stats) = &result.spec_stats {
        let accept_rate = if stats.draft_tokens > 0 {
            stats.accepted_tokens as f64 / stats.draft_tokens as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "Speculation: {}/{} drafts accepted ({:.0}%), K={}, draft_layers={}",
            stats.accepted_tokens, stats.draft_tokens, accept_rate, stats.spec_k, stats.draft_layers
        );
    }
}
