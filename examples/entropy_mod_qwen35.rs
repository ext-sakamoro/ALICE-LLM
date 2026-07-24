//! Entropy-driven Mixture-of-Depths (MoD) observation tool.
//!
//! Emits per-(token, layer) hidden-state statistics as JSONL for post-hoc
//! analysis of natural thresholds for depth routing / early-exit strategies.
//!
//! Uses `Llama3Model::forward_with_layer_hook` (see `llama3.rs`) to observe
//! the hidden state at the input of each layer without modifying the forward
//! path. The observation closure always returns `false` (read-only, never
//! skips a layer). Works for any GGUF architecture that flows through the
//! standard hook-enabled forward path (Qwen 3.5, Llama, Mistral, etc.).
//!
//! Emitted per-layer metrics:
//! - `l2_norm`: hidden-state L2 magnitude (signal strength proxy)
//! - `delta_l2`: L2 distance from previous layer's hidden state (per-layer
//!   "surprise" proxy in the CALM sense — stable / small delta typically
//!   indicates the model has converged on a prediction)
//! - `mean`, `variance`, `max_abs`, `sparsity_1e_3`: basic activation
//!   statistics for downstream analysis
//!
//! References:
//! - Schuster et al. 2022, "Confident Adaptive Language Modeling" (CALM)
//! - Raposo et al. 2024, "Mixture-of-Depths" (Google DeepMind)
//!
//! Usage:
//!   cargo run --release --features gguf --example entropy_mod_qwen35 -- \
//!     --model models/Qwen3.5-4B-Q4_K_M.gguf \
//!     --dataset data/wikitext-2/wiki.test.raw \
//!     --n-samples 500 \
//!     --output /tmp/qwen35_entropy.jsonl

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::llama3::{Llama3Config, Llama3Model};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    let idx = args.iter().position(|a| a == flag)?;
    args.get(idx + 1)?.parse().ok()
}

fn parse_arg_str<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let idx = args.iter().position(|a| a == flag)?;
    args.get(idx + 1).map(String::as_str)
}

#[derive(Debug, Clone, Copy)]
struct HiddenStats {
    l2_norm: f32,
    mean: f32,
    variance: f32,
    max_abs: f32,
    sparsity_1e_3: f32,
}

fn compute_hidden_stats(hidden: &[f32]) -> HiddenStats {
    let n = hidden.len() as f32;
    let sum: f32 = hidden.iter().sum();
    let mean = sum / n;
    let mut sum_sq_dev = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut max_abs = 0.0f32;
    let mut sparse_count = 0usize;
    for &x in hidden {
        let d = x - mean;
        sum_sq_dev += d * d;
        sum_sq += x * x;
        let ax = x.abs();
        if ax > max_abs {
            max_abs = ax;
        }
        if ax < 1e-3 {
            sparse_count += 1;
        }
    }
    HiddenStats {
        l2_norm: sum_sq.sqrt(),
        mean,
        variance: sum_sq_dev / n,
        max_abs,
        sparsity_1e_3: sparse_count as f32 / n,
    }
}

fn delta_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum_sq = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        sum_sq += d * d;
    }
    sum_sq.sqrt()
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "Usage: entropy_mod_qwen35 --model <path.gguf> --dataset <path.txt> \\
                                 [--n-samples 500] [--output <path.jsonl>] \\
                                 [--progress-every 100]

  --model            Path to GGUF model file
  --dataset          Path to raw text dataset (UTF-8)
  --n-samples        Number of tokens to observe (default: 500)
  --output           JSONL output path (default: stderr summary only)
  --progress-every   Emit stderr progress every N tokens (default: 100)

Observes hidden-state statistics per (token, layer) for post-hoc analysis
of depth-routing / early-exit thresholds. Read-only: the observation
closure never modifies hidden state or skips layers.

References:
  Schuster et al. 2022, Confident Adaptive Language Modeling (CALM)
  Raposo et al. 2024, Mixture-of-Depths (Google DeepMind)"
        );
        std::process::exit(1);
    }

    let model_path = parse_arg_str(&args, "--model").expect("--model required");
    let dataset_path = parse_arg_str(&args, "--dataset").expect("--dataset required");
    let n_samples: usize = parse_arg(&args, "--n-samples").unwrap_or(500);
    let output_path: Option<&str> = parse_arg_str(&args, "--output");
    let progress_every: usize = parse_arg(&args, "--progress-every").unwrap_or(100);

    eprintln!("=== entropy_mod observation ===");
    eprintln!("  model:      {model_path}");
    eprintln!("  dataset:    {dataset_path}");
    eprintln!("  n_samples:  {n_samples}");
    eprintln!(
        "  output:     {}",
        output_path.unwrap_or("(stderr summary only)")
    );

    let t_load = Instant::now();
    eprintln!("Loading GGUF...");
    let data = std::fs::read(model_path).expect("failed to read GGUF file");
    let gguf = GgufFile::parse(&data).expect("failed to parse GGUF");
    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("failed to load tokenizer");
    let config = Llama3Config::from_gguf(&gguf).expect("failed to load Llama3Config from GGUF");
    eprintln!(
        "  GGUF parsed: {}ms (vocab={} layers={} hidden={})",
        t_load.elapsed().as_millis(),
        tokenizer.vocab_size(),
        config.num_layers,
        config.hidden_dim
    );

    let t_model = Instant::now();
    let mut model = Llama3Model::from_gguf(&gguf).expect("failed to load Llama3Model");
    eprintln!(
        "  Model loaded: {}ms (CPU forward path)",
        t_model.elapsed().as_millis()
    );

    let t_tok = Instant::now();
    eprintln!("Loading + tokenizing dataset...");
    let raw = std::fs::read_to_string(dataset_path).expect("failed to read dataset file");
    // Pre-cap raw text so encode() is bounded (tokenizer is O(chars)).
    // Rough upper bound: ~10 chars/token × 2 safety = 20 chars/token.
    let cap = n_samples.saturating_mul(20);
    let raw_slice: &str = if raw.len() > cap {
        let mut end = cap.min(raw.len());
        while end < raw.len() && !raw.as_bytes()[end].is_ascii_whitespace() {
            end += 1;
        }
        &raw[..end.min(raw.len())]
    } else {
        &raw
    };
    let mut tokens = tokenizer.encode(raw_slice);
    if tokens.len() > n_samples {
        tokens.truncate(n_samples);
    }
    if tokens.is_empty() {
        eprintln!("no tokens after tokenization — dataset empty or too small");
        std::process::exit(2);
    }
    eprintln!(
        "  Tokenized: {} tokens ({}ms)",
        tokens.len(),
        t_tok.elapsed().as_millis()
    );

    let num_layers = config.num_layers;
    let hidden_dim = config.hidden_dim;

    let mut writer: Option<BufWriter<File>> =
        output_path.map(|p| BufWriter::new(File::create(p).expect("failed to open output file")));

    // Header line: JSON object with `_meta=true` so downstream parsers can
    // skip it (or use it for schema info). Keeps output pure JSONL.
    if let Some(w) = writer.as_mut() {
        writeln!(
            w,
            "{{\"_meta\":true,\"model\":\"{}\",\"num_layers\":{},\"hidden_dim\":{},\"n_tokens\":{}}}",
            json_escape(model_path),
            num_layers,
            hidden_dim,
            tokens.len()
        )
        .expect("failed to write JSONL header");
    }

    let t_forward = Instant::now();
    eprintln!("Forward pass with per-layer observation...");

    // Per-layer running sums for the stderr summary at the end.
    let mut sum_l2_by_layer = vec![0.0f64; num_layers];
    let mut sum_delta_by_layer = vec![0.0f64; num_layers];
    let mut count_by_layer = vec![0u64; num_layers];

    for (token_idx, &token_id) in tokens.iter().enumerate() {
        // Buffer holding the previous layer's hidden state, for delta_l2.
        let mut prev_hidden: Option<Vec<f32>> = None;
        // Per-token buffer of (stats, delta_l2) to emit as JSONL after the
        // forward call (keeps the closure hot loop free of I/O).
        let mut layer_stats: Vec<(HiddenStats, f32)> = Vec::with_capacity(num_layers);

        let _logits = model.forward_with_layer_hook(token_id, |layer_idx, hidden| {
            let stats = compute_hidden_stats(hidden);
            let d = match &prev_hidden {
                None => 0.0f32,
                Some(prev) => delta_l2(hidden, prev),
            };
            // Clone the current hidden state so the next iteration can
            // compute delta_l2 against it. This is O(hidden_dim) per layer
            // per token; acceptable for observation-only work.
            prev_hidden = Some(hidden.clone());
            layer_stats.push((stats, d));

            sum_l2_by_layer[layer_idx] += f64::from(stats.l2_norm);
            sum_delta_by_layer[layer_idx] += f64::from(d);
            count_by_layer[layer_idx] += 1;

            // Never skip a layer — this pass is pure observation.
            false
        });

        if let Some(w) = writer.as_mut() {
            for (layer_idx, (stats, d)) in layer_stats.iter().enumerate() {
                writeln!(
                    w,
                    "{{\"token_idx\":{},\"token_id\":{},\"layer_idx\":{},\"l2_norm\":{:.6},\"delta_l2\":{:.6},\"mean\":{:.6},\"variance\":{:.6},\"max_abs\":{:.6},\"sparsity_1e_3\":{:.6}}}",
                    token_idx,
                    token_id,
                    layer_idx,
                    stats.l2_norm,
                    d,
                    stats.mean,
                    stats.variance,
                    stats.max_abs,
                    stats.sparsity_1e_3
                )
                .expect("failed to write JSONL row");
            }
        }

        if (token_idx + 1) % progress_every == 0 {
            let elapsed = t_forward.elapsed().as_secs_f32();
            let tok_per_sec = (token_idx + 1) as f32 / elapsed.max(1e-6);
            eprintln!(
                "  {}/{} tokens ({:.2} tok/s, {:.1}s elapsed)",
                token_idx + 1,
                tokens.len(),
                tok_per_sec,
                elapsed
            );
        }
    }

    if let Some(mut w) = writer {
        w.flush().expect("failed to flush output");
    }

    let total_elapsed = t_forward.elapsed();
    eprintln!(
        "Forward complete: {} tokens in {:.1}s ({:.2} tok/s)",
        tokens.len(),
        total_elapsed.as_secs_f32(),
        tokens.len() as f32 / total_elapsed.as_secs_f32().max(1e-6)
    );

    // Per-layer averages (stderr, human-readable).
    eprintln!();
    eprintln!(
        "=== per-layer averages (across {} tokens) ===",
        tokens.len()
    );
    eprintln!("  layer  |    mean L2    |   mean delta L2");
    eprintln!("  -------+---------------+-----------------");
    for layer_idx in 0..num_layers {
        let count = count_by_layer[layer_idx].max(1) as f64;
        eprintln!(
            "  {:>5}  |  {:>12.4}  |  {:>15.4}",
            layer_idx,
            sum_l2_by_layer[layer_idx] / count,
            sum_delta_by_layer[layer_idx] / count
        );
    }
}
