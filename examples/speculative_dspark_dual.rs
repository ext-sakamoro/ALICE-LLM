//! DSpark Phase 5: llama3 speculative dual decoding with MarkovBigramBias
//!
//! Baseline (vanilla speculative_dual) と DSpark 3 パターン (bigram_strength 0.1/0.5/1.0)
//! を A/B 比較し、accept rate + tok/s + speedup を出力する
//!
//! prompt tokens を `MarkovBigramBias::from_sequence` で bootstrap した簡易 warm-up
//! (実運用では別途学習した bigram を load する想定)
//!
//! Usage:
//!   cargo run --release --example speculative_dspark_dual --features "dspark,gguf,parallel" -- \
//!     --model models/Llama-3-ELYZA-JP-8B-q4_k_m.gguf \
//!     --draft-model models/llama-3.2-1b-gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
//!     --prompt "日本の首都は" \
//!     --speculative-k 4 \
//!     --max-tokens 100 \
//!     --bigram-rank 256

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::llama3::Llama3Model;
use alice_llm::speculative_dspark::MarkovBigramBias;
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

    let max_tokens: usize = parse_arg(&args, "--max-tokens").unwrap_or(100);
    let temperature: f32 = parse_arg(&args, "--temperature").unwrap_or(0.0);
    let spec_k: usize = parse_arg(&args, "--speculative-k").unwrap_or(4);
    let bigram_rank: u32 = parse_arg(&args, "--bigram-rank").unwrap_or(256);
    let confidence_head_path: Option<&str> = args
        .iter()
        .position(|a| a == "--confidence-head")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str);

    println!("=== DSpark Phase 5: Speculative Dual with MarkovBigramBias ===");
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
    println!();

    let formatted = format!(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    );

    // --- Bigram bootstrap ---
    // prompt tokens から隣接 pair を observe した簡易 warm-up
    // 実運用では別途 corpus 学習した bigram を load する
    let vocab = draft_model.config.vocab_size as u32;
    let prompt_tokens = tokenizer.encode(&formatted);
    println!(
        "Bootstrapping MarkovBigramBias (rank={bigram_rank}) from prompt tokens (n={})",
        prompt_tokens.len()
    );
    let bigram = MarkovBigramBias::from_sequence(vocab, bigram_rank, &prompt_tokens)
        .expect("bigram bootstrap");
    println!(
        "  observed_prev_count = {}, empty = {}",
        bigram.observed_prev_count(),
        bigram.is_empty()
    );
    println!();

    // --- Baseline: vanilla generate_speculative_dual ---
    println!("--- Baseline (vanilla speculative_dual, K={spec_k}) ---");
    let baseline = main_model.generate_speculative_dual(
        &mut draft_model,
        &tokenizer,
        &formatted,
        max_tokens,
        temperature,
        spec_k,
    );
    println!("{}", baseline.text);
    println!(
        "  {} tokens, {:.2} tok/s ({}ms prefill + {}ms decode)",
        baseline.tokens_generated, baseline.tokens_per_sec, baseline.prefill_ms, baseline.decode_ms
    );
    if let Some(stats) = &baseline.spec_stats {
        let accept = if stats.draft_tokens > 0 {
            stats.accepted_tokens as f64 / stats.draft_tokens as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "  Baseline: {}/{} accepted ({:.1}%)",
            stats.accepted_tokens, stats.draft_tokens, accept
        );
    }
    let baseline_tps = baseline.tokens_per_sec;
    println!();

    // --- DSpark variants ---
    // Phase 6 拡張: 第 9 引数 `advanced` に None を渡すと Phase 5 と bit-exact 同一動作
    // trained PositionConfidenceHead を使った confidence-gated 早期打切りは Phase 7+ で
    // accept/reject label collection example を追加後に demonstrable になる
    for strength in [0.1_f32, 0.5, 1.0] {
        println!("--- DSpark (bigram_strength={strength:.1}, K={spec_k}) ---");
        let result = main_model
            .generate_speculative_dual_dspark(
                &mut draft_model,
                &tokenizer,
                &formatted,
                max_tokens,
                temperature,
                spec_k,
                Some(&bigram),
                strength,
                None,
            )
            .expect("dspark generate");
        println!("{}", result.text);
        println!(
            "  {} tokens, {:.2} tok/s ({}ms prefill + {}ms decode)",
            result.tokens_generated, result.tokens_per_sec, result.prefill_ms, result.decode_ms
        );
        if let Some(stats) = &result.spec_stats {
            let accept = if stats.draft_tokens > 0 {
                stats.accepted_tokens as f64 / stats.draft_tokens as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "  DSpark: {}/{} accepted ({:.1}%)",
                stats.accepted_tokens, stats.draft_tokens, accept
            );
        }
        if baseline_tps > 0.0 {
            let ratio = result.tokens_per_sec / baseline_tps;
            println!("  Speedup vs baseline: {ratio:.2}x");
        }
        println!();
    }

    // --- DSpark + confidence-gated variants (Phase 7、trained head 指定時のみ) ---
    // `dspark-serde` feature が有効なら trained PositionConfidenceHead を bincode で load して
    // confidence_threshold 0.3/0.5/0.7 で追加 A/B/C 比較する trained head は
    // `examples/dspark_train_confidence_head.rs` で事前学習する
    #[cfg(feature = "dspark-serde")]
    if let Some(head_path) = confidence_head_path {
        use alice_llm::speculative_dspark::{DsparkAdvancedConfig, PositionConfidenceHead};
        println!("=== Confidence-Gated variants (trained head loaded from {head_path}) ===");
        let head_bytes = fs::read(head_path).expect("Failed to read confidence head");
        let head: PositionConfidenceHead =
            bincode::deserialize(&head_bytes).expect("Failed to deserialize confidence head");
        println!(
            "  loaded head: block_size={}, hidden_dim={}",
            head.block_size(),
            head.hidden_dim()
        );
        println!();

        for threshold in [0.3_f32, 0.5, 0.7] {
            println!(
                "--- DSpark + confidence-gated (strength=0.5, threshold={threshold:.1}, K={spec_k}) ---"
            );
            let cfg = DsparkAdvancedConfig {
                confidence_head: &head,
                confidence_threshold: threshold,
                hidden_capture_layer: None,
            };
            let result = main_model
                .generate_speculative_dual_dspark(
                    &mut draft_model,
                    &tokenizer,
                    &formatted,
                    max_tokens,
                    temperature,
                    spec_k,
                    Some(&bigram),
                    0.5,
                    Some(&cfg),
                )
                .expect("dspark generate (advanced)");
            println!("{}", result.text);
            println!(
                "  {} tokens, {:.2} tok/s",
                result.tokens_generated, result.tokens_per_sec
            );
            if let Some(stats) = &result.spec_stats {
                let accept = if stats.draft_tokens > 0 {
                    stats.accepted_tokens as f64 / stats.draft_tokens as f64 * 100.0
                } else {
                    0.0
                };
                println!(
                    "  DSpark: {}/{} accepted ({:.1}%)",
                    stats.accepted_tokens, stats.draft_tokens, accept
                );
            }
            if baseline_tps > 0.0 {
                let ratio = result.tokens_per_sec / baseline_tps;
                println!("  Speedup vs baseline: {ratio:.2}x");
            }
            println!();
        }
    }

    #[cfg(not(feature = "dspark-serde"))]
    if confidence_head_path.is_some() {
        eprintln!(
            "warning: --confidence-head requires --features dspark-serde to load trained head; ignoring"
        );
    }
}
