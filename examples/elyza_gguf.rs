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
use alice_llm::sample_argmax;
use std::env;
use std::fs;
use std::io::Write;
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

/// Emit one JSONL line of top-5 logits to stderr for CPU/GPU divergence diagnostics
/// (Issue #40). Format is intentionally identical between `qwen_gpu.rs` (GPU) and
/// `elyza_gguf.rs` (CPU) so the two streams can be diffed line-by-line.
fn dump_logits_jsonl(backend: &str, pos: i64, logits: &[f32], tokenizer: &GgufTokenizer) {
    let mut top: [(u32, f32); 5] = [(u32::MAX, f32::NEG_INFINITY); 5];
    for (i, &v) in logits.iter().enumerate() {
        let mut min_idx = 0usize;
        for k in 1..5 {
            if top[k].1 < top[min_idx].1 {
                min_idx = k;
            }
        }
        if v > top[min_idx].1 {
            top[min_idx] = (i as u32, v);
        }
    }
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut line = String::with_capacity(256);
    line.push_str(&format!(
        "{{\"backend\":\"{backend}\",\"pos\":{pos},\"top5\":["
    ));
    for (k, (tid, logit)) in top.iter().enumerate() {
        if k > 0 {
            line.push(',');
        }
        let decoded = tokenizer.decode(&[*tid]);
        let mut esc = String::with_capacity(decoded.len() + 2);
        for ch in decoded.chars() {
            match ch {
                '"' => esc.push_str("\\\""),
                '\\' => esc.push_str("\\\\"),
                '\n' => esc.push_str("\\n"),
                '\r' => esc.push_str("\\r"),
                '\t' => esc.push_str("\\t"),
                c if (c as u32) < 0x20 => esc.push_str(&format!("\\u{:04x}", c as u32)),
                c => esc.push(c),
            }
        }
        line.push_str(&format!(
            "{{\"id\":{tid},\"logit\":{logit:.6},\"tok\":\"{esc}\"}}"
        ));
    }
    line.push_str("]}");
    eprintln!("{line}");
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
    // Issue #40 diagnostic: dump top-5 logits per position (JSONL to stderr) for
    // the prompt-end position and first N decoded tokens. When set, forces a
    // plain-argmax manual forward loop (no speculative, no ternary generation).
    let logits_dump: usize = parse_arg(&args, "--logits-dump").unwrap_or(0);
    // Issue #40 diagnostic: dump post-final-RMSNorm hidden state (pre-output-
    // projection) to stderr as JSONL. Combined with the same flag on
    // `qwen_gpu`, allows offline cos-sim / L2 diff to isolate whether the
    // logit gap comes from the layer stack or from output_proj. Sets the
    // ALICE_LLM_DUMP_HIDDEN env var so the library's forward() path emits
    // one JSONL line per forward call.
    let dump_final_hidden = has_flag(&args, "--dump-final-hidden");
    if dump_final_hidden {
        std::env::set_var("ALICE_LLM_DUMP_HIDDEN", "1");
    }

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

    // Detect chat template from GGUF metadata (model architecture).
    let arch_str = gguf.meta_str("general.architecture").unwrap_or("llama");
    let formatted = match arch_str {
        // Qwen 3 has "thinking mode" default-on which loops <think>...</think>
        // in greedy sampling. Pre-fill empty <think></think> to disable it.
        "qwen3" | "qwen3moe" => format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ),
        "qwen2" => format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        ),
        "gemma2" | "gemma3" | "gemma3n" => format!(
            "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        ),
        _ => format!(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
    };

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

    // Issue #40 diagnostic path: manual argmax loop with per-position top-5 logits
    // dump to stderr. Bypasses speculative / ternary generate() paths to keep the
    // CPU vs GPU comparison apples-to-apples (temperature=0 argmax on both sides).
    if logits_dump > 0 {
        let prompt_tokens = tokenizer.encode(&formatted);
        let n_prompt = prompt_tokens.len();
        println!("  Prompt tokens: {n_prompt}");
        println!("  logits-dump: N={logits_dump} (stderr JSONL, backend=\"cpu\")");

        let t_prefill = Instant::now();
        // Prefill all prompt tokens except the last.
        for &tok in &prompt_tokens[..n_prompt - 1] {
            let _ = model.forward(tok);
        }
        // Last prompt token: capture logits for pos=-1 (prompt end).
        let last_prompt = *prompt_tokens.last().unwrap();
        let mut logits = model.forward(last_prompt);
        let prefill_ms = t_prefill.elapsed().as_millis();
        dump_logits_jsonl("cpu", -1, &logits, &tokenizer);

        let t_decode = Instant::now();
        let mut generated: Vec<u32> = Vec::new();
        let eos_id = tokenizer.eos_id;
        for step in 0..max_tokens {
            let next_token = sample_argmax(&logits) as u32;
            if next_token == eos_id {
                break;
            }
            generated.push(next_token);
            let text = tokenizer.decode(&[next_token]);
            print!("{text}");
            std::io::stdout().flush().ok();

            logits = model.forward(next_token);
            if step < logits_dump {
                dump_logits_jsonl("cpu", step as i64, &logits, &tokenizer);
            }
        }
        let decode_ms = t_decode.elapsed().as_millis();
        let n_gen = generated.len();
        let tok_per_sec = if decode_ms > 0 {
            n_gen as f64 / (decode_ms as f64 / 1000.0)
        } else {
            0.0
        };
        println!();
        println!("---");
        println!(
            "Tokens: {n_gen} generated, {n_prompt} prompt (logits-dump manual argmax)"
        );
        println!(
            "Speed: {tok_per_sec:.1} tok/s ({prefill_ms} prefill + {decode_ms} decode = {} total ms)",
            prefill_ms + decode_ms
        );
        return;
    }

    let result = if ternary {
        model.generate_ternary(&tokenizer, &formatted, max_tokens, temperature, 40)
    } else if spec_k > 0 {
        model.generate_speculative(
            &tokenizer,
            &formatted,
            max_tokens,
            temperature,
            40,
            spec_k,
            draft_layers,
        )
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
            stats.accepted_tokens,
            stats.draft_tokens,
            accept_rate,
            stats.spec_k,
            stats.draft_layers
        );
    }
}
