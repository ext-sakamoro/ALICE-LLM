//! Early-exit ablation for entropy-driven depth routing on Qwen 3.5.
//!
//! Companion to `entropy_mod_qwen35.rs` (per-layer statistic observation) and
//! `entropy_ppl_correlation_qwen35.rs` (correlation validation). Given that
//! the Qwen 3.5-4B hidden-state variance measured at the input to layer 7 is
//! a bimodal signal weakly correlated with per-token difficulty
//! (Pearson r=0.15) but sharply separating the lowest quartile from Q2-Q4
//! (mean_loss 2.05 vs 3.00), this tool measures whether skipping layers
//! 7..num_layers for tokens in the low-variance Q1 preserves usable
//! perplexity. This is the classical Confident Adaptive Language Modeling
//! (CALM) "early-exit" question specialized to a single variance-gated depth.
//!
//! Two modes evaluated on the same (model, dataset, chunking) configuration:
//!
//!   - `baseline` — full 32-layer forward (reference PPL).
//!   - `early` — gate at the configured layer; when
//!     `variance(hidden_at_gate_input) < threshold` the layer hook returns
//!     `true` for every layer from the gate onward so CPU skips their
//!     compute. The residual stream at the gate boundary is then fed to the
//!     standard `output_norm` + `output_proj` unchanged.
//!
//! Both PPLs are computed in identical non-overlapping chunks (KV cache is
//! `model.reset()`-cleared between chunks) so any within-chunk pollution
//! from skipped layers stays local to a single chunk.
//!
//! Note on gate semantics: the layer hook fires *before* the corresponding
//! layer body. Setting `--gate-layer 7` therefore observes the state
//! entering layer 7 (equivalently, the residual after layers 0..=6
//! completed) and, on a Q1 hit, causes layers 7..num_layers to be skipped.
//! Per Fast-Path token, `num_layers - gate_layer` layers are skipped, so
//! for Qwen 3.5-4B (32 layers) with `--gate-layer 7` the maximum per-hit
//! saving is 25 / 32 = 78.1% of the layer stack.
//!
//! References:
//! - Schuster et al. 2022, "Confident Adaptive Language Modeling" (CALM).
//! - Raposo et al. 2024, "Mixture-of-Depths" (Google DeepMind).
//!
//! Usage:
//!   cargo run --release --features gguf --example early_exit_qwen35 -- \
//!     --model models/Qwen3.5-4B-Q4_K_M.gguf \
//!     --dataset data/wikitext-2/wiki.test.raw \
//!     --n-samples 500 --ctx 512 \
//!     --gate-layer 7 --variance-threshold 0.0156 \
//!     --mode both --output /tmp/early_exit_qwen35.jsonl
//!
//! Emits one JSON line per mode on stdout for scripted comparison.

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

/// Variance of a slice, dividing by n. Matches the divisor used in
/// `entropy_mod_qwen35.rs` / `entropy_ppl_correlation_qwen35.rs` so the
/// threshold learned there transfers to this tool without renormalisation.
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
/// `perplexity.rs::log_prob_at` minus the EOG-inf-filter path (plain-text
/// wikitext scoring does not need it).
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

/// Per-mode aggregated stats collected over all chunks of one pass.
struct RunSummary {
    mode: &'static str,
    n_scored: usize,
    n_fast_path: usize,
    sum_nll: f64,
    wall_sec: f64,
}

impl RunSummary {
    fn ppl(&self) -> f64 {
        if self.n_scored == 0 {
            f64::NAN
        } else {
            (self.sum_nll / self.n_scored as f64).exp()
        }
    }

    fn nll_per_token(&self) -> f64 {
        if self.n_scored == 0 {
            f64::NAN
        } else {
            self.sum_nll / self.n_scored as f64
        }
    }

    fn fast_path_rate(&self) -> f64 {
        if self.n_scored == 0 {
            0.0
        } else {
            self.n_fast_path as f64 / self.n_scored as f64
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn to_json(
        &self,
        model: &str,
        dataset: &str,
        ctx: usize,
        gate_layer: usize,
        threshold: f32,
        num_layers: usize,
    ) -> String {
        let skipped = num_layers.saturating_sub(gate_layer);
        let per_hit_skip_ratio = if num_layers == 0 {
            0.0
        } else {
            skipped as f64 / num_layers as f64
        };
        let effective_skip = self.fast_path_rate() * per_hit_skip_ratio;
        format!(
            "{{\"mode\":\"{}\",\"model\":\"{}\",\"dataset\":\"{}\",\
             \"ctx\":{},\"gate_layer\":{},\"variance_threshold\":{:.6},\
             \"num_layers\":{},\"per_hit_skip_ratio\":{:.4},\
             \"n_scored\":{},\"n_fast_path\":{},\"fast_path_rate\":{:.4},\
             \"effective_skip\":{:.4},\"ppl\":{:.6},\"nll_per_token\":{:.6},\
             \"wall_sec\":{:.3},\"tok_per_sec\":{:.4}}}",
            self.mode,
            model.replace('\\', "\\\\").replace('"', "\\\""),
            dataset.replace('\\', "\\\\").replace('"', "\\\""),
            ctx,
            gate_layer,
            threshold,
            num_layers,
            per_hit_skip_ratio,
            self.n_scored,
            self.n_fast_path,
            self.fast_path_rate(),
            effective_skip,
            self.ppl(),
            self.nll_per_token(),
            self.wall_sec,
            self.n_scored as f64 / self.wall_sec.max(1e-6)
        )
    }
}

/// Run one full pass over all chunks under the given policy.
///
/// `gate = None` disables the variance gate and matches `model.forward(tok)`
/// exactly. `gate = Some((layer, threshold))` observes hidden at the input
/// of `layer` and, when the variance is below `threshold`, causes CPU to
/// skip every layer from `layer` onward for that token.
#[allow(clippy::too_many_arguments)]
fn run_pass(
    model: &mut Llama3Model,
    tokens: &[u32],
    ctx: usize,
    gate: Option<(usize, f32)>,
    mut per_token_writer: Option<&mut BufWriter<File>>,
    progress_every: usize,
    mode_label: &'static str,
) -> RunSummary {
    let t_run = Instant::now();
    let mut sum_nll = 0.0_f64;
    let mut n_scored: usize = 0;
    let mut n_fast_path: usize = 0;
    let mut last_report_at: usize = 0;

    for chunk in tokens.chunks(ctx) {
        if chunk.len() < 2 {
            break;
        }
        model.reset();
        let mut prev_logits: Option<Vec<f32>> = None;
        let mut prev_variance: f32 = f32::NAN;
        let mut prev_was_fast: bool = false;

        for &tok in chunk {
            // Score the transition INTO `tok` using logits produced by the
            // previous position (same lag-1 protocol as
            // `entropy_ppl_correlation_qwen35.rs`).
            if let Some(logits) = prev_logits.as_ref() {
                if (tok as usize) < logits.len() {
                    let log_p = log_prob_at(logits, tok);
                    let nll = -f64::from(log_p);
                    sum_nll += nll;
                    n_scored += 1;
                    if let Some(w) = per_token_writer.as_deref_mut() {
                        writeln!(
                            w,
                            "{{\"mode\":\"{}\",\"target_token_id\":{},\
                             \"variance\":{:.6},\"was_fast_path\":{},\
                             \"log_loss\":{:.6}}}",
                            mode_label, tok, prev_variance, prev_was_fast, nll
                        )
                        .expect("failed to write per-token JSONL row");
                    }
                }
            }

            // Forward with (optionally) a variance-gated early exit.
            let mut engaged = false;
            let mut captured_var: f32 = f32::NAN;
            let logits = model.forward_with_layer_hook(tok, |layer_idx, hidden| {
                if let Some((gate_layer, threshold)) = gate {
                    if layer_idx == gate_layer {
                        let v = variance(hidden);
                        captured_var = v;
                        if v < threshold {
                            engaged = true;
                            return true;
                        }
                    } else if layer_idx > gate_layer && engaged {
                        return true;
                    }
                }
                false
            });
            if engaged {
                n_fast_path += 1;
            }
            prev_logits = Some(logits);
            prev_variance = captured_var;
            prev_was_fast = engaged;

            if n_scored > 0 && n_scored - last_report_at >= progress_every {
                let elapsed = t_run.elapsed().as_secs_f32();
                let tok_per_sec = n_scored as f32 / elapsed.max(1e-6);
                let ppl_running = (sum_nll / n_scored as f64).exp();
                let hit_rate = n_fast_path as f32 / n_scored as f32;
                eprintln!(
                    "  [{mode_label}] scored={} fast_path={} ({:.1}%) \
                     ppl={:.4} ({:.2} tok/s, {:.1}s elapsed)",
                    n_scored,
                    n_fast_path,
                    hit_rate * 100.0,
                    ppl_running,
                    tok_per_sec,
                    elapsed
                );
                last_report_at = n_scored;
                let _ = std::io::stderr().flush();
            }
        }
    }

    let wall_sec = t_run.elapsed().as_secs_f64();
    RunSummary {
        mode: mode_label,
        n_scored,
        n_fast_path,
        sum_nll,
        wall_sec,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "Usage: early_exit_qwen35 --model <path.gguf> --dataset <path.txt> \\
                                [--n-samples 500] [--ctx 512] \\
                                [--gate-layer 7] [--variance-threshold 0.0156] \\
                                [--mode baseline|early|both] \\
                                [--output <path.jsonl>] [--progress-every 100]

  --model                Path to GGUF model file
  --dataset              Path to raw text dataset (UTF-8)
  --n-samples            Number of tokens to score (default: 500)
  --ctx                  Context length per chunk (default: 512)
  --gate-layer           Layer index at which to evaluate the variance gate
                         (default: 7). The hook fires at the input boundary
                         of this layer, so on a hit the layers gate..num_layers
                         are skipped.
  --variance-threshold   Fast-Path variance ceiling (default: 0.0156, from
                         WikiText-2 500-token Q1/Q2 boundary of the L7-input
                         variance distribution).
  --mode                 baseline | early | both (default: both). `both`
                         runs baseline first, then early — each mode starts
                         from a fresh KV cache.
  --output               Optional per-token JSONL path (mode, variance,
                         was_fast_path, log_loss, target_token_id).
  --progress-every       Emit stderr progress every N scored tokens (default: 100).

Emits one JSON line per mode on stdout for scripted comparison.

References:
  Schuster et al. 2022, Confident Adaptive Language Modeling (CALM)
  Raposo et al. 2024, Mixture-of-Depths (Google DeepMind)"
        );
        std::process::exit(1);
    }

    let model_path = parse_arg_str(&args, "--model").expect("--model required");
    let dataset_path = parse_arg_str(&args, "--dataset").expect("--dataset required");
    let n_samples: usize = parse_arg(&args, "--n-samples").unwrap_or(500);
    let ctx: usize = parse_arg(&args, "--ctx").unwrap_or(512);
    let gate_layer: usize = parse_arg(&args, "--gate-layer").unwrap_or(7);
    let variance_threshold: f32 = parse_arg(&args, "--variance-threshold").unwrap_or(0.0156);
    let mode: String = parse_arg::<String>(&args, "--mode").unwrap_or_else(|| "both".to_string());
    let output_path: Option<&str> = parse_arg_str(&args, "--output");
    let progress_every: usize = parse_arg(&args, "--progress-every").unwrap_or(100);

    if !matches!(mode.as_str(), "baseline" | "early" | "both") {
        eprintln!("--mode must be one of: baseline | early | both (got '{mode}')");
        std::process::exit(2);
    }

    eprintln!("=== early_exit_qwen35 ===");
    eprintln!("  model:               {model_path}");
    eprintln!("  dataset:             {dataset_path}");
    eprintln!("  n_samples:           {n_samples}");
    eprintln!("  ctx:                 {ctx}");
    eprintln!("  gate_layer:          {gate_layer}");
    eprintln!("  variance_threshold:  {variance_threshold:.6}");
    eprintln!("  mode:                {mode}");
    eprintln!(
        "  output:              {}",
        output_path.unwrap_or("(stdout summaries only)")
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
    if gate_layer >= config.num_layers {
        eprintln!(
            "--gate-layer {} out of range for model with {} layers",
            gate_layer, config.num_layers
        );
        std::process::exit(3);
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
    if tokens.len() > n_samples {
        tokens.truncate(n_samples);
    }
    if tokens.len() < 2 {
        eprintln!(
            "dataset too short ({} tokens after tokenization)",
            tokens.len()
        );
        std::process::exit(4);
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
             \"gate_layer\":{},\"variance_threshold\":{:.6},\"ctx\":{},\"n_tokens\":{}}}",
            model_path.replace('\\', "\\\\").replace('"', "\\\""),
            config.num_layers,
            config.hidden_dim,
            gate_layer,
            variance_threshold,
            ctx,
            tokens.len()
        )
        .expect("failed to write JSONL header");
    }

    let mut baseline: Option<RunSummary> = None;
    let mut early: Option<RunSummary> = None;

    if mode == "baseline" || mode == "both" {
        eprintln!();
        eprintln!(
            "--- pass 1: baseline (full {} layers) ---",
            config.num_layers
        );
        baseline = Some(run_pass(
            &mut model,
            &tokens,
            ctx,
            None,
            writer.as_mut(),
            progress_every,
            "baseline",
        ));
    }
    if mode == "early" || mode == "both" {
        eprintln!();
        eprintln!(
            "--- pass 2: early (gate=L{}, variance<{:.6} → skip L{}..L{}) ---",
            gate_layer,
            variance_threshold,
            gate_layer,
            config.num_layers - 1
        );
        early = Some(run_pass(
            &mut model,
            &tokens,
            ctx,
            Some((gate_layer, variance_threshold)),
            writer.as_mut(),
            progress_every,
            "early",
        ));
    }

    if let Some(mut w) = writer {
        w.flush().expect("failed to flush output");
    }

    eprintln!();
    eprintln!("=== summary ===");

    if let Some(b) = baseline.as_ref() {
        eprintln!(
            "  baseline: ppl={:.4} nll/tok={:.4} scored={} wall={:.1}s ({:.2} tok/s)",
            b.ppl(),
            b.nll_per_token(),
            b.n_scored,
            b.wall_sec,
            b.n_scored as f64 / b.wall_sec.max(1e-6)
        );
        println!(
            "{}",
            b.to_json(
                model_path,
                dataset_path,
                ctx,
                gate_layer,
                variance_threshold,
                config.num_layers,
            )
        );
    }
    if let Some(e) = early.as_ref() {
        eprintln!(
            "  early:    ppl={:.4} nll/tok={:.4} scored={} fast_path={} ({:.1}%) \
             wall={:.1}s ({:.2} tok/s)",
            e.ppl(),
            e.nll_per_token(),
            e.n_scored,
            e.n_fast_path,
            e.fast_path_rate() * 100.0,
            e.wall_sec,
            e.n_scored as f64 / e.wall_sec.max(1e-6)
        );
        println!(
            "{}",
            e.to_json(
                model_path,
                dataset_path,
                ctx,
                gate_layer,
                variance_threshold,
                config.num_layers,
            )
        );
    }

    if let (Some(b), Some(e)) = (baseline.as_ref(), early.as_ref()) {
        let delta_ppl_pct = (e.ppl() / b.ppl() - 1.0) * 100.0;
        let delta_nll = e.nll_per_token() - b.nll_per_token();
        let per_hit_skip_ratio = if config.num_layers == 0 {
            0.0
        } else {
            (config.num_layers - gate_layer) as f64 / config.num_layers as f64
        };
        let effective_skip = e.fast_path_rate() * per_hit_skip_ratio;
        eprintln!();
        eprintln!(
            "  compare:  delta_ppl={:+.4} ({:+.2}%)  delta_nll={:+.4}  \
             fast_path_rate={:.2}%  effective_skip={:.2}%",
            e.ppl() - b.ppl(),
            delta_ppl_pct,
            delta_nll,
            e.fast_path_rate() * 100.0,
            effective_skip * 100.0
        );
        eprintln!(
            "  interpretation:  {}",
            if delta_ppl_pct.abs() < 5.0 {
                "delta<5% — early-exit preserves PPL, viable Fast-Path"
            } else if delta_ppl_pct.abs() < 20.0 {
                "5%<=delta<20% — partial degradation, tune threshold or narrow Fast-Path"
            } else {
                "delta>=20% — early-exit breaks PPL, gate signal insufficient"
            }
        );
    }
}
