//! Alternate depth-routing signal validation: shannon entropy of the
//! layer-N *projected* logits vs. next-token log-loss.
//!
//! Follow-up to `entropy_ppl_correlation_qwen35.rs` (which validated raw
//! hidden-state variance as a candidate signal at the same layer boundary
//! and found Pearson r=0.15 with a step-function bucket structure). This
//! tool projects the intermediate hidden state through `output_norm` +
//! `output_proj` to produce vocab logits at layer N, computes the shannon
//! entropy of the softmax over those logits, and correlates that entropy
//! with the full-forward log-loss of the next token.
//!
//! Rationale: variance is a low-order statistic of the raw residual stream
//! and might not reflect predictive uncertainty. The softmax entropy of
//! the projected logits is the information-theoretically correct measure
//! of "how uncertain the model would be about the next token if it stopped
//! at this depth" — the metric used by CALM confidence-based early exit.
//! If entropy is a stronger signal (Pearson r > 0.3) than variance
//! (r ~ 0.15), the gate-signal choice — not the depth-routing idea — is
//! the bottleneck. If entropy also gives weak correlation, per-token
//! naive depth routing is unlikely to work with any low-order layer-N
//! statistic.
//!
//! Data captured per (token, target) pair (same lag-1 protocol as
//! `entropy_ppl_correlation_qwen35.rs`):
//!   - `l7_entropy` — shannon entropy in nats of
//!     `softmax(project_hidden_to_logits(hidden_at_input_of_observe_layer))`.
//!   - `log_loss` — `-log P(target | full forward)` in nats.
//!
//! Uses `Llama3Model::forward_with_layer_hook` (read-only, closure always
//! returns false) + `Llama3Model::project_hidden_to_logits`. No modification
//! to the forward path.
//!
//! References:
//! - Schuster et al. 2022, "Confident Adaptive Language Modeling" (CALM).
//! - Raposo et al. 2024, "Mixture-of-Depths" (Google DeepMind).
//!
//! Usage:
//!   cargo run --release --features gguf --example l7_entropy_signal_qwen35 -- \
//!     --model models/Qwen3.5-4B-Q4_K_M.gguf \
//!     --dataset data/wikitext-2/wiki.test.raw \
//!     --n-samples 500 \
//!     --observe-layer 7 \
//!     --output /tmp/qwen35_l7_entropy.jsonl

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

/// Shannon entropy of `softmax(logits)` in nats. Numerically stable via the
/// max-subtraction trick; f64 accumulator for the sum-of-exp.
fn softmax_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        if v > max {
            max = v;
        }
    }
    let mut z = 0.0f64;
    for &v in logits {
        z += f64::from(v - max).exp();
    }
    let log_z = max + (z.ln() as f32);
    // H = -Σ p_i * log(p_i)  where  log(p_i) = (l_i - max) - log(z / exp(0))
    //                            = l_i - log_z
    // So H = -Σ p_i * (l_i - log_z) = log_z - Σ p_i * l_i.
    let inv_z = 1.0f64 / z;
    let mut weighted = 0.0f64;
    for &v in logits {
        let p = f64::from(v - max).exp() * inv_z;
        weighted += p * f64::from(v);
    }
    let h = f64::from(log_z) - weighted;
    h.max(0.0) as f32
}

/// Numerically stable log P(target) from raw logits. Mirrors
/// `perplexity.rs::log_prob_at`.
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

/// Fractional ranks (average-of-ties) of a slice, in stable order.
fn ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut r = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && values[idx[j]] == values[idx[i]] {
            j += 1;
        }
        // Average rank for ties, 1-based.
        let avg_rank = ((i + 1) as f64 + j as f64) / 2.0;
        for k in i..j {
            r[idx[k]] = avg_rank;
        }
        i = j;
    }
    r
}

/// Pearson correlation of two equal-length slices. Returns NaN for n<2 or
/// zero variance.
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    assert_eq!(n, y.len());
    if n < 2 {
        return f64::NAN;
    }
    let nn = n as f64;
    let mean_x = x.iter().sum::<f64>() / nn;
    let mean_y = y.iter().sum::<f64>() / nn;
    let mut cov = 0.0f64;
    let mut var_x = 0.0f64;
    let mut var_y = 0.0f64;
    for (&a, &b) in x.iter().zip(y.iter()) {
        let dx = a - mean_x;
        let dy = b - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    let denom = (var_x.max(0.0) * var_y.max(0.0)).sqrt();
    if denom > 1e-12 {
        cov / denom
    } else {
        f64::NAN
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "Usage: l7_entropy_signal_qwen35 --model <path.gguf> --dataset <path.txt> \\
                                       [--n-samples 500] [--observe-layer 7] \\
                                       [--output <path.jsonl>] [--progress-every 100]

  --model            Path to GGUF model file
  --dataset          Path to raw text dataset (UTF-8)
  --n-samples        Number of tokens to score (default: 500)
  --observe-layer    Layer index at which to project the hidden state and
                     compute softmax entropy (default: 7). The hook fires
                     before the layer body, so this observes the residual
                     stream after layers 0..observe_layer completed.
  --output           JSONL output path (default: stderr summary only).
  --progress-every   Emit stderr progress every N tokens (default: 100).

Emits per-token (l7_entropy, log_loss for next token) pairs for downstream
correlation analysis of a candidate depth-routing signal, plus a summary
with Pearson r, Spearman rho, and Q1-Q4 bucket means (matching the layout
of entropy_ppl_correlation_qwen35.rs so results are directly comparable).

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

    eprintln!("=== l7_entropy_signal ===");
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
            "{{\"_meta\":true,\"model\":\"{}\",\"num_layers\":{},\"hidden_dim\":{},\
             \"observe_layer\":{},\"n_scored\":{}}}",
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
    eprintln!("Forward + L7 entropy capture + log-loss scoring...");

    let mut sum_nll = 0.0f64;
    let mut scored: usize = 0;

    // Paired series for correlation analysis.
    let mut entropies: Vec<f64> = Vec::with_capacity(tokens.len());
    let mut losses: Vec<f64> = Vec::with_capacity(tokens.len());

    // Lag-1 state: the entropy captured on iteration T is what produced the
    // logits scored on iteration T+1.
    let mut prev_logits: Option<Vec<f32>> = None;
    let mut prev_entropy: Option<f32> = None;

    for (t_idx, &token_id) in tokens.iter().enumerate() {
        if let Some(logits) = prev_logits.as_ref() {
            if (token_id as usize) < logits.len() {
                let log_p = log_prob_at(logits, token_id);
                let nll = -f64::from(log_p);
                let e = prev_entropy.unwrap_or(f32::NAN);

                if let Some(w) = writer.as_mut() {
                    writeln!(
                        w,
                        "{{\"token_idx\":{},\"l7_entropy\":{:.6},\
                         \"log_loss\":{:.6},\"target_token_id\":{}}}",
                        t_idx - 1,
                        e,
                        nll,
                        token_id
                    )
                    .expect("failed to write JSONL row");
                }

                sum_nll += nll;
                scored += 1;
                if e.is_finite() && nll.is_finite() {
                    entropies.push(f64::from(e));
                    losses.push(nll);
                }
            }
        }

        // Forward, capturing the observe-layer hidden state and immediately
        // projecting it to L_observe logits for a softmax-entropy read.
        let cap_layer = observe_layer;
        let mut captured_hidden: Option<Vec<f32>> = None;
        let logits = model.forward_with_layer_hook(token_id, |layer_idx, hidden| {
            if layer_idx == cap_layer && captured_hidden.is_none() {
                captured_hidden = Some(hidden.clone());
            }
            false
        });
        let entropy = match captured_hidden {
            Some(h) => {
                let l_logits = model.project_hidden_to_logits(&h);
                softmax_entropy(&l_logits)
            }
            None => f32::NAN,
        };
        prev_logits = Some(logits);
        prev_entropy = Some(entropy);

        if scored > 0 && scored.is_multiple_of(progress_every) {
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

    if entropies.len() >= 2 {
        let r = pearson(&entropies, &losses);
        let re = ranks(&entropies);
        let rl = ranks(&losses);
        let rho = pearson(&re, &rl);
        let mean_e: f64 = entropies.iter().sum::<f64>() / entropies.len() as f64;
        let mean_l: f64 = losses.iter().sum::<f64>() / losses.len() as f64;
        let var_e: f64 =
            entropies.iter().map(|&x| (x - mean_e).powi(2)).sum::<f64>() / entropies.len() as f64;
        let var_l: f64 =
            losses.iter().map(|&x| (x - mean_l).powi(2)).sum::<f64>() / losses.len() as f64;
        eprintln!(
            "  Pearson r  (l7_entropy vs log_loss): {:.4}  (n = {})",
            r,
            entropies.len()
        );
        eprintln!("  Spearman rho                       : {:.4}", rho);
        eprintln!("  mean_entropy = {:.6}  var_entropy = {:.6}", mean_e, var_e);
        eprintln!("  mean_loss    = {:.6}  var_loss    = {:.6}", mean_l, var_l);
        if r.abs() > 0.5 {
            eprintln!("  → strong linear correlation, entropy is a difficulty predictor");
        } else if r.abs() > 0.3 {
            eprintln!("  → moderate linear correlation, usable signal");
        } else if r.abs() > 0.1 {
            eprintln!("  → weak linear correlation, marginal signal");
        } else {
            eprintln!("  → no meaningful linear correlation");
        }

        // Q1-Q4 bucket means of loss, keyed on entropy quartile. Mirrors
        // the entropy_ppl_correlation_qwen35 analysis so the step-function
        // structure (if any) is immediately visible.
        let mut idx: Vec<usize> = (0..entropies.len()).collect();
        idx.sort_by(|&a, &b| {
            entropies[a]
                .partial_cmp(&entropies[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let n = idx.len();
        let q = n / 4;
        for (q_idx, chunk) in [&idx[..q], &idx[q..2 * q], &idx[2 * q..3 * q], &idx[3 * q..]]
            .iter()
            .enumerate()
        {
            if chunk.is_empty() {
                continue;
            }
            let e_min = entropies[chunk[0]];
            let e_max = entropies[*chunk.last().unwrap()];
            let mean_loss: f64 = chunk.iter().map(|&i| losses[i]).sum::<f64>() / chunk.len() as f64;
            let ppl_bucket = mean_loss.exp();
            eprintln!(
                "  Q{}  entropy=[{:.4},{:.4}]  n={}  mean_loss={:.4}  ppl={:.2}",
                q_idx + 1,
                e_min,
                e_max,
                chunk.len(),
                mean_loss,
                ppl_bucket
            );
        }
    }

    // Single-line JSON on stdout for scripted collection.
    let pearson_r = if entropies.len() >= 2 {
        pearson(&entropies, &losses)
    } else {
        f64::NAN
    };
    let spearman_rho = if entropies.len() >= 2 {
        let re = ranks(&entropies);
        let rl = ranks(&losses);
        pearson(&re, &rl)
    } else {
        f64::NAN
    };
    println!(
        "{{\"model\":\"{}\",\"dataset\":\"{}\",\"observe_layer\":{},\
         \"n_scored\":{},\"n_pairs\":{},\"ppl\":{:.6},\
         \"pearson_r\":{:.6},\"spearman_rho\":{:.6},\
         \"wall_sec\":{:.3},\"tok_per_sec\":{:.4}}}",
        model_path.replace('\\', "\\\\").replace('"', "\\\""),
        dataset_path.replace('\\', "\\\\").replace('"', "\\\""),
        observe_layer,
        scored,
        entropies.len(),
        ppl,
        pearson_r,
        spearman_rho,
        wall,
        scored as f64 / wall.max(1e-6)
    );
}
