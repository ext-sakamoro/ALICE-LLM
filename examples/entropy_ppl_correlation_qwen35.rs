//! Correlation validation between hidden-state variance and per-token log-loss.
//!
//! Companion tool to `entropy_mod_qwen35.rs` (observation) and `perplexity.rs`
//! (per-token log-loss). Emits paired (variance @ specified layer, log-loss)
//! per token as JSONL so downstream analysis can decide whether the observed
//! bimodal variance signal actually correlates with token difficulty.
//!
//! Rationale: a bimodal signal by itself does not prove it discriminates
//! "easy vs hard" tokens — it may reflect token frequency, sentence position,
//! POS category, etc. Depth-routing based on such a signal is only valid if
//! it correlates with actual per-token loss.
//!
//! Uses `Llama3Model::forward_with_layer_hook` (read-only, closure always
//! returns false). No modification to forward path.
//!
//! References:
//! - Schuster et al. 2022, "Confident Adaptive Language Modeling" (CALM)
//! - Raposo et al. 2024, "Mixture-of-Depths" (Google DeepMind)
//!
//! Usage:
//!   cargo run --release --features gguf --example entropy_ppl_correlation_qwen35 -- \
//!     --model models/Qwen3.5-4B-Q4_K_M.gguf \
//!     --dataset data/wikitext-2/wiki.test.raw \
//!     --n-samples 500 \
//!     --observe-layer 7 \
//!     --output /tmp/qwen35_ppl_corr.jsonl

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

/// Variance of a slice, dividing by n (matches `entropy_mod_qwen35.rs` so the
/// numbers align across tools).
fn variance(hidden: &[f32]) -> f32 {
    let n = hidden.len() as f32;
    let mean: f32 = hidden.iter().sum::<f32>() / n;
    let mut sum_sq_dev = 0.0f32;
    for &x in hidden {
        let d = x - mean;
        sum_sq_dev += d * d;
    }
    sum_sq_dev / n
}

/// Numerically stable log P(target) from raw logits. Mirrors
/// `perplexity.rs::log_prob_at`, minus the eog-filter machinery (plain text
/// scoring on wikitext does not need it).
fn log_prob_at(logits: &[f32], target: u32) -> f32 {
    let target_us = target as usize;
    debug_assert!(
        target_us < logits.len(),
        "target token id out of vocab range"
    );
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max {
            max = v;
        }
    }
    let mut sum_exp = 0.0_f64;
    for &v in logits {
        sum_exp += (f64::from(v - max)).exp();
    }
    let logsumexp = max + (sum_exp.ln() as f32);
    logits[target_us] - logsumexp
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "Usage: entropy_ppl_correlation_qwen35 --model <path.gguf> --dataset <path.txt> \\
                                            [--n-samples 500] [--observe-layer 7] \\
                                            [--output <path.jsonl>] [--progress-every 100]

  --model            Path to GGUF model file
  --dataset          Path to raw text dataset (UTF-8)
  --n-samples        Number of tokens to score (default: 500)
  --observe-layer    Layer index at which to record variance (default: 7)
  --output           JSONL output path (default: stderr summary only)
  --progress-every   Emit stderr progress every N tokens (default: 100)

Emits per-token (variance @ observe-layer, log_loss for next token) pairs
for downstream correlation analysis of a candidate depth-routing signal.

References:
  Schuster et al. 2022, Confident Adaptive Language Modeling (CALM)
  Raposo et al. 2024, Mixture-of-Depths (Google DeepMind)"
        );
        std::process::exit(1);
    }

    let model_path = parse_arg_str(&args, "--model").expect("--model required");
    let dataset_path = parse_arg_str(&args, "--dataset").expect("--dataset required");
    let n_samples: usize = parse_arg(&args, "--n-samples").unwrap_or(500);
    let observe_layer: usize = parse_arg(&args, "--observe-layer").unwrap_or(7);
    let output_path: Option<&str> = parse_arg_str(&args, "--output");
    let progress_every: usize = parse_arg(&args, "--progress-every").unwrap_or(100);

    eprintln!("=== entropy_ppl_correlation ===");
    eprintln!("  model:           {model_path}");
    eprintln!("  dataset:         {dataset_path}");
    eprintln!("  n_samples:       {n_samples}");
    eprintln!("  observe_layer:   {observe_layer}");
    eprintln!(
        "  output:          {}",
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
    if observe_layer >= config.num_layers {
        eprintln!(
            "--observe-layer {} out of range for model with {} layers",
            observe_layer, config.num_layers
        );
        std::process::exit(2);
    }

    let t_model = Instant::now();
    let mut model = Llama3Model::from_gguf(&gguf).expect("failed to load Llama3Model");
    eprintln!(
        "  Model loaded: {}ms (CPU forward path)",
        t_model.elapsed().as_millis()
    );

    let t_tok = Instant::now();
    eprintln!("Loading + tokenizing dataset...");
    let raw = std::fs::read_to_string(dataset_path).expect("failed to read dataset file");
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
    // Need n_samples + 1 tokens so we can score the transition into the last one.
    let target_len = n_samples.saturating_add(1);
    if tokens.len() > target_len {
        tokens.truncate(target_len);
    }
    if tokens.len() < 2 {
        eprintln!(
            "dataset too short ({} tokens after tokenization)",
            tokens.len()
        );
        std::process::exit(3);
    }
    eprintln!(
        "  Tokenized: {} tokens ({}ms)",
        tokens.len(),
        t_tok.elapsed().as_millis()
    );

    let mut writer: Option<BufWriter<File>> =
        output_path.map(|p| BufWriter::new(File::create(p).expect("failed to open output file")));

    if let Some(w) = writer.as_mut() {
        writeln!(
            w,
            "{{\"_meta\":true,\"model\":\"{}\",\"num_layers\":{},\"hidden_dim\":{},\"observe_layer\":{},\"n_scored\":{}}}",
            model_path.replace('\\', "\\\\").replace('"', "\\\""),
            config.num_layers,
            config.hidden_dim,
            observe_layer,
            tokens.len() - 1
        )
        .expect("failed to write JSONL header");
    }

    model.reset();

    let t_run = Instant::now();
    eprintln!("Forward + variance capture + log-loss scoring...");

    // Per-scored-transition running sums.
    let mut sum_nll = 0.0f64;
    let mut scored: usize = 0;

    // Correlation accumulator over (variance, nll) pairs.
    let mut n_pairs: usize = 0;
    let mut sum_v = 0.0f64;
    let mut sum_n = 0.0f64;
    let mut sum_vv = 0.0f64;
    let mut sum_nn = 0.0f64;
    let mut sum_vn = 0.0f64;

    // Lag-1 state: the variance we captured on iteration T is what produced
    // the logits scored on iteration T+1.
    let mut prev_logits: Option<Vec<f32>> = None;
    let mut prev_variance: Option<f32> = None;

    for (t_idx, &token_id) in tokens.iter().enumerate() {
        // Score the transition INTO this token, using logits + variance from
        // the previous iteration.
        if let Some(logits) = prev_logits.as_ref() {
            if (token_id as usize) < logits.len() {
                let log_p = log_prob_at(logits, token_id);
                let nll = -f64::from(log_p);
                let v = prev_variance.unwrap_or(f32::NAN);

                if let Some(w) = writer.as_mut() {
                    writeln!(
                        w,
                        "{{\"token_idx\":{},\"variance\":{:.6},\"log_loss\":{:.6},\"target_token_id\":{}}}",
                        t_idx - 1,
                        v,
                        nll,
                        token_id
                    )
                    .expect("failed to write JSONL row");
                }

                sum_nll += nll;
                scored += 1;

                if v.is_finite() && nll.is_finite() {
                    let vf = f64::from(v);
                    n_pairs += 1;
                    sum_v += vf;
                    sum_n += nll;
                    sum_vv += vf * vf;
                    sum_nn += nll * nll;
                    sum_vn += vf * nll;
                }
            }
        }

        // Forward for this token, capturing variance at the requested layer.
        let cap_layer = observe_layer;
        let mut captured_var: f32 = f32::NAN;
        let logits = model.forward_with_layer_hook(token_id, |layer_idx, hidden| {
            if layer_idx == cap_layer {
                captured_var = variance(hidden);
            }
            false // observation only, never skip
        });
        prev_logits = Some(logits);
        prev_variance = Some(captured_var);

        if scored > 0 && scored % progress_every == 0 {
            let elapsed = t_run.elapsed().as_secs_f32();
            let tok_per_sec = scored as f32 / elapsed.max(1e-6);
            let ppl_running = (sum_nll / scored as f64).exp();
            eprintln!(
                "  scored={} ppl={:.4} ({:.2} tok/s, {:.1}s elapsed)",
                scored, ppl_running, tok_per_sec, elapsed
            );
        }
    }

    if let Some(mut w) = writer {
        w.flush().expect("failed to flush output");
    }

    let ppl = if scored > 0 {
        (sum_nll / scored as f64).exp()
    } else {
        f64::NAN
    };
    let wall = t_run.elapsed().as_secs_f64();
    eprintln!();
    eprintln!("=== summary ===");
    eprintln!(
        "  scored={} ppl={:.4} nll/tok={:.4} wall={:.1}s ({:.2} tok/s)",
        scored,
        ppl,
        sum_nll / (scored.max(1) as f64),
        wall,
        scored as f64 / wall.max(1e-6)
    );

    if n_pairs >= 2 {
        let n = n_pairs as f64;
        let mean_v = sum_v / n;
        let mean_n = sum_n / n;
        let var_v = (sum_vv / n) - mean_v * mean_v;
        let var_n = (sum_nn / n) - mean_n * mean_n;
        let cov = (sum_vn / n) - mean_v * mean_n;
        let denom = (var_v.max(0.0) * var_n.max(0.0)).sqrt();
        let pearson = if denom > 1e-12 { cov / denom } else { f64::NAN };
        eprintln!(
            "  Pearson correlation (variance@L{} vs log_loss): r = {:.4}  (n = {})",
            observe_layer, pearson, n_pairs
        );
        eprintln!("  mean_var = {:.6}  var_var = {:.6}", mean_v, var_v);
        eprintln!("  mean_nll = {:.6}  var_nll = {:.6}", mean_n, var_n);
        if pearson.abs() > 0.5 {
            eprintln!("  → strong correlation, signal is a difficulty predictor");
        } else if pearson.abs() > 0.3 {
            eprintln!("  → moderate correlation, usable signal");
        } else if pearson.abs() > 0.1 {
            eprintln!("  → weak correlation, marginal signal");
        } else {
            eprintln!("  → no meaningful correlation — signal does not predict difficulty");
        }
    }
}
