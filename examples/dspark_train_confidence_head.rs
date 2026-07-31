//! DSpark Phase 7: PositionConfidenceHead 学習 example
//!
//! vanilla speculative dual pipeline を走らせて各 draft position の
//! `(hidden_state, was_accepted)` を collect し、`PositionConfidenceHead::train_step`
//! で SGD BCE 学習する 学習済 head は bincode で save し、
//! `speculative_dspark_dual --confidence-head <path>` で confidence-gated mode に load できる
//!
//! Usage:
//!   cargo run --release --example dspark_train_confidence_head \
//!     --features "dspark,dspark-serde,gguf,parallel" -- \
//!     --model models/main.gguf \
//!     --draft-model models/draft.gguf \
//!     --prompt "日本の首都は" \
//!     --max-tokens 200 \
//!     --speculative-k 4 \
//!     --epochs 20 \
//!     --lr 0.05 \
//!     --output trained_head.bincode

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::llama3::Llama3Model;
use alice_llm::speculative_dspark::PositionConfidenceHead;
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
        .expect("Usage: --model <main.gguf> --draft-model <draft.gguf>");

    let draft_path = args
        .iter()
        .position(|a| a == "--draft-model")
        .and_then(|i| args.get(i + 1))
        .expect("Usage: --draft-model <draft.gguf>");

    let prompt = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("日本の首都はどこですか？");

    let max_tokens: usize = parse_arg(&args, "--max-tokens").unwrap_or(200);
    let temperature: f32 = parse_arg(&args, "--temperature").unwrap_or(0.0);
    let spec_k: usize = parse_arg(&args, "--speculative-k").unwrap_or(4);
    let epochs: usize = parse_arg(&args, "--epochs").unwrap_or(20);
    let lr: f32 = parse_arg(&args, "--lr").unwrap_or(0.05);
    let output_path = args
        .iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("trained_head.bincode");
    let hidden_layer: Option<usize> = parse_arg(&args, "--hidden-layer");

    println!("=== DSpark Phase 7: PositionConfidenceHead Training ===");
    println!();

    println!("Loading main model: {model_path}");
    let load_start = Instant::now();
    let main_data = fs::read(model_path).expect("Failed to read main model");
    let main_gguf = GgufFile::parse(&main_data).expect("Failed to parse main model");
    let tokenizer = GgufTokenizer::from_gguf(&main_gguf).expect("Failed to load tokenizer");
    let mut main_model = Llama3Model::from_gguf(&main_gguf).expect("Failed to load main model");
    println!(
        "  Main: {} layers, hidden={}, vocab={} ({}ms)",
        main_model.config.num_layers,
        main_model.config.hidden_dim,
        main_model.config.vocab_size,
        load_start.elapsed().as_millis()
    );

    let draft_start = Instant::now();
    let draft_data = fs::read(draft_path).expect("Failed to read draft model");
    let draft_gguf = GgufFile::parse(&draft_data).expect("Failed to parse draft model");
    let mut draft_model = Llama3Model::from_gguf(&draft_gguf).expect("Failed to load draft model");
    println!(
        "  Draft: {} layers, hidden={}, vocab={} ({}ms)",
        draft_model.config.num_layers,
        draft_model.config.hidden_dim,
        draft_model.config.vocab_size,
        draft_start.elapsed().as_millis()
    );
    println!();

    let formatted = format!(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    );

    // --- Label collection (vanilla dual + hidden capture) ---
    println!("--- Label collection: spec_k={spec_k}, max_tokens={max_tokens} ---");
    let collect_start = Instant::now();
    let (gen_result, samples) = main_model
        .generate_speculative_dual_collect_labels(
            &mut draft_model,
            &tokenizer,
            &formatted,
            max_tokens,
            temperature,
            spec_k,
            hidden_layer,
        )
        .expect("collect labels");
    let collect_ms = collect_start.elapsed().as_millis();
    println!("{}", gen_result.text);
    println!(
        "  {} tokens generated, {} labels collected in {}ms",
        gen_result.tokens_generated,
        samples.len(),
        collect_ms
    );
    if let Some(stats) = &gen_result.spec_stats {
        let accept_rate = if stats.draft_tokens > 0 {
            stats.accepted_tokens as f64 / stats.draft_tokens as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "  Baseline speculation: {}/{} accepted ({:.1}%)",
            stats.accepted_tokens, stats.draft_tokens, accept_rate
        );
    }
    println!();

    // --- Position 別 accept rate 統計 ---
    println!("--- Position-wise accept rate ---");
    let mut per_pos_total = vec![0_u32; spec_k];
    let mut per_pos_accepted = vec![0_u32; spec_k];
    for s in &samples {
        let idx = s.position as usize;
        if idx < spec_k {
            per_pos_total[idx] += 1;
            if s.was_accepted {
                per_pos_accepted[idx] += 1;
            }
        }
    }
    for pos in 0..spec_k {
        let total = per_pos_total[pos];
        let acc = per_pos_accepted[pos];
        let rate = if total > 0 {
            acc as f64 / total as f64 * 100.0
        } else {
            0.0
        };
        println!("  pos {pos}: {acc}/{total} accepted ({rate:.1}%)");
    }
    println!();

    // --- Training ---
    let hidden_dim = draft_model.config.hidden_dim as u32;
    let block_size = spec_k as u32;
    println!(
        "--- Training PositionConfidenceHead (block={block_size}, hidden={hidden_dim}, epochs={epochs}, lr={lr}) ---"
    );
    let mut head = PositionConfidenceHead::new(block_size, hidden_dim).expect("valid");

    let train_start = Instant::now();
    for epoch in 0..epochs {
        let mut sum_loss = 0.0_f64;
        let mut count = 0_usize;
        for s in &samples {
            if s.position >= block_size {
                continue;
            }
            let loss = head
                .train_step(s.position, &s.hidden, s.was_accepted, lr)
                .expect("train_step");
            sum_loss += loss as f64;
            count += 1;
        }
        let mean = if count > 0 {
            sum_loss / count as f64
        } else {
            0.0
        };
        println!("  epoch {epoch:3}: mean loss = {mean:.4} (n={count})");
    }
    println!(
        "  Training completed in {}ms",
        train_start.elapsed().as_millis()
    );
    println!();

    // --- Save trained head ---
    let encoded = bincode::serialize(&head).expect("bincode serialize");
    fs::write(output_path, &encoded).expect("write output");
    println!(
        "--- Saved trained head to {output_path} ({} bytes) ---",
        encoded.len()
    );

    // --- Sanity check: predict per position on last sample ---
    if let Some(last) = samples.last() {
        let pred = head.predict(last.position, &last.hidden).expect("predict");
        println!(
            "  Sanity: last sample position={} was_accepted={} → predicted confidence={:.4}",
            last.position, last.was_accepted, pred
        );
    }
}
