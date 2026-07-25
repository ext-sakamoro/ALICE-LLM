//! External-signal-driven per-layer routing via `Llama3Model::forward_with_surprise`.
//!
//! Structural variant of `early_exit_qwen35.rs`: instead of computing the
//! routing signal (variance) inside the hook from the layer's own hidden
//! state, this example feeds an *external* per-token signal vector through
//! `forward_with_surprise` and lets the caller-supplied `gate` closure make
//! the depth-skip decision.
//!
//! The signal contract (`SurpriseVec<'_> = &[f32]`) is intentionally
//! unopinionated. In practice a lightweight upstream model — an auxiliary
//! world-model prediction error, a saliency map, a difficulty scorer, or a
//! signal produced outside the LLM entirely — populates the slice, and the
//! wrapper standardises how it reaches `forward_with_layer_hook`.
//!
//! Two modes evaluated on the same (model, dataset, chunking) configuration:
//!
//!   - `baseline` — full forward via `forward(token_id)` (reference PPL).
//!   - `incarnated` — `forward_with_surprise(token_id, Some(&signal), gate)`
//!     where `signal[i]` is a pre-computed per-token score in `[0, 1]` and
//!     `gate(layer_idx, surprise)` returns `true` for `layer_idx >= gate_layer`
//!     when the signal's mean falls below `--signal-threshold`. This is the
//!     external-signal analogue of `early_exit_qwen35.rs`'s Q1 gate.
//!
//! Both PPLs are computed in identical non-overlapping chunks; the KV cache
//! is `model.reset()`-cleared between chunks so per-chunk pollution stays
//! local to a single chunk.
//!
//! Also verifies the wrapper's backward-compatibility guarantee:
//! `forward_with_surprise(id, None, |_, _| false)` produces bit-exact
//! identical logits to `forward(id)` on a real loaded model. Any drift here
//! would indicate a regression in the wrapper implementation.
//!
//! Usage:
//!   cargo run --release --features gguf --example incarnation_forward -- \
//!     --model models/Qwen3.5-4B-Q4_K_M.gguf \
//!     --dataset data/wikitext-2/wiki.test.raw \
//!     --n-samples 500 --ctx 512 \
//!     --gate-layer 7 --signal-threshold 0.5 \
//!     --signal-seed 42 \
//!     --mode both --output /tmp/incarnation_forward.jsonl
//!
//! Emits one JSON line per mode on stdout for scripted comparison.

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::llama3::{Llama3Config, Llama3Model, SurpriseVec};
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

/// Numerically stable log P(target) from raw logits.
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

/// Deterministic pseudo-random signal generator. Fills `out` with values in
/// `[0.0, 1.0]` driven by `(token_id, seed)` — a stand-in for whatever
/// upstream module actually produces the routing signal in production.
fn make_signal(token_id: u32, seed: u64, len: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), len);
    // splitmix64-ish mixer, fully deterministic for a given (token_id, seed).
    let mut state = seed
        .wrapping_add(u64::from(token_id))
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for slot in out.iter_mut().take(len) {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Map to [0.0, 1.0].
        *slot = (z >> 40) as f32 / (1u64 << 24) as f32;
    }
}

/// Per-mode aggregated stats collected over all chunks of one pass.
struct RunSummary {
    mode: &'static str,
    n_scored: usize,
    n_gated: usize,
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

    fn to_json_line(&self, extra: &str) -> String {
        format!(
            "{{\"mode\":\"{}\",\"n_scored\":{},\"n_gated\":{},\
             \"ppl\":{:.6},\"wall_sec\":{:.3}{}}}",
            self.mode,
            self.n_scored,
            self.n_gated,
            self.ppl(),
            self.wall_sec,
            extra
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    model: &mut Llama3Model,
    tokens: &[u32],
    ctx: usize,
    mode_label: &'static str,
    gate: Option<(usize, f32, u64, usize)>, // (gate_layer, signal_threshold, seed, signal_len)
    mut per_token_writer: Option<&mut BufWriter<File>>,
    progress_every: usize,
) -> RunSummary {
    let t_run = Instant::now();
    let mut sum_nll = 0.0_f64;
    let mut n_scored = 0usize;
    let mut n_gated = 0usize;
    let mut last_report_at = 0usize;

    // Scratch buffer for the per-token signal (populated deterministically).
    let signal_len = gate.map_or(0, |(_, _, _, len)| len);
    let mut signal_buf: Vec<f32> = vec![0.0; signal_len];

    for chunk in tokens.chunks(ctx) {
        if chunk.len() < 2 {
            break;
        }
        model.reset();
        let mut prev_logits: Option<Vec<f32>> = None;
        let mut prev_was_gated: bool = false;

        for &tok in chunk {
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
                             \"was_gated\":{},\"log_loss\":{:.6}}}",
                            mode_label, tok, prev_was_gated, nll
                        )
                        .expect("failed to write per-token JSONL row");
                    }
                }
            }

            let (logits, was_gated) = match gate {
                None => (model.forward(tok), false),
                Some((gate_layer, threshold, seed, len)) => {
                    // Populate the per-token signal deterministically.
                    make_signal(tok, seed, len, &mut signal_buf);
                    let signal_mean =
                        signal_buf.iter().copied().sum::<f32>() / signal_buf.len() as f32;
                    let will_gate = signal_mean < threshold;
                    let logits = model.forward_with_surprise(
                        tok,
                        Some(&signal_buf as SurpriseVec<'_>),
                        |layer_idx, surprise| {
                            // The gate closure inspects the (immutable)
                            // external signal and returns `true` for layers
                            // beyond `gate_layer` when mean drops below
                            // `threshold` — a signal-driven early exit.
                            let s = surprise.expect("surprise present in incarnated mode");
                            let mean = s.iter().copied().sum::<f32>() / s.len() as f32;
                            layer_idx >= gate_layer && mean < threshold
                        },
                    );
                    (logits, will_gate)
                }
            };
            if was_gated {
                n_gated += 1;
            }
            prev_logits = Some(logits);
            prev_was_gated = was_gated;

            if n_scored > 0 && n_scored - last_report_at >= progress_every {
                let elapsed = t_run.elapsed().as_secs_f32();
                let tok_per_sec = n_scored as f32 / elapsed.max(1e-6);
                let ppl_running = (sum_nll / n_scored as f64).exp();
                let gate_rate = n_gated as f32 / n_scored.max(1) as f32;
                eprintln!(
                    "  [{mode_label}] scored={} gated={} ({:.1}%) \
                     ppl={:.4} ({:.2} tok/s, {:.1}s elapsed)",
                    n_scored,
                    n_gated,
                    gate_rate * 100.0,
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
        n_gated,
        sum_nll,
        wall_sec,
    }
}

/// Verify `forward_with_surprise(id, None, |_,_| false) == forward(id)` bit-exact
/// on a real loaded model for a small batch of tokens. This is the real-world
/// backward-compatibility check the wrapper's docstring promises.
fn verify_backward_compat(model: &mut Llama3Model, tokens: &[u32], n: usize) {
    let n = n.min(tokens.len());
    eprintln!("Verifying backward compatibility on first {n} tokens...");
    for &tok in tokens.iter().take(n) {
        model.reset();
        let baseline = model.forward(tok);
        model.reset();
        let via_wrapper = model.forward_with_surprise(tok, None, |_layer_idx, _surprise| false);
        assert_eq!(
            baseline.len(),
            via_wrapper.len(),
            "vocab dimension mismatch at token {tok}"
        );
        for (i, (a, b)) in baseline.iter().zip(via_wrapper.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "logit drift at token {tok} vocab {i}: forward={a} wrapper={b}"
            );
        }
    }
    eprintln!("  Backward compat verified: bit-exact for {n} tokens");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "Usage: incarnation_forward --model <path.gguf> --dataset <path.txt> \\
                                       [--n-samples 500] [--ctx 512] \\
                                       [--gate-layer 7] [--signal-threshold 0.5] \\
                                       [--signal-seed 42] [--signal-len 4] \\
                                       [--mode baseline|incarnated|both] \\
                                       [--output <path.jsonl>] [--progress-every 100] \\
                                       [--skip-backward-compat]

  --model                Path to GGUF model file
  --dataset              Path to raw text dataset (UTF-8)
  --n-samples            Number of tokens to score (default: 500)
  --ctx                  Context length per chunk (default: 512)
  --gate-layer           Layer index at which the external signal decides
                         whether to skip (default: 7). Layers >= gate_layer
                         are skipped on a gate hit.
  --signal-threshold     Mean-signal ceiling below which the gate fires
                         (default: 0.5).
  --signal-seed          RNG seed for the deterministic external signal
                         (default: 42).
  --signal-len           Signal vector length (default: 4).
  --mode                 baseline | incarnated | both (default: both).
                         `both` runs baseline first, then incarnated —
                         each mode starts from a fresh KV cache.
  --output               Optional per-token JSONL path.
  --progress-every       Emit stderr progress every N scored tokens
                         (default: 100).
  --skip-backward-compat Skip the bit-exact verify pass (faster startup).

Emits one JSON line per mode on stdout for scripted comparison."
        );
        std::process::exit(1);
    }

    let model_path = parse_arg_str(&args, "--model").expect("--model required");
    let dataset_path = parse_arg_str(&args, "--dataset").expect("--dataset required");
    let n_samples: usize = parse_arg(&args, "--n-samples").unwrap_or(500);
    let ctx: usize = parse_arg(&args, "--ctx").unwrap_or(512);
    let gate_layer: usize = parse_arg(&args, "--gate-layer").unwrap_or(7);
    let signal_threshold: f32 = parse_arg(&args, "--signal-threshold").unwrap_or(0.5);
    let signal_seed: u64 = parse_arg(&args, "--signal-seed").unwrap_or(42);
    let signal_len: usize = parse_arg(&args, "--signal-len").unwrap_or(4);
    let mode: String = parse_arg::<String>(&args, "--mode").unwrap_or_else(|| "both".to_string());
    let output_path: Option<&str> = parse_arg_str(&args, "--output");
    let progress_every: usize = parse_arg(&args, "--progress-every").unwrap_or(100);
    let skip_backward_compat = args.iter().any(|a| a == "--skip-backward-compat");

    if !matches!(mode.as_str(), "baseline" | "incarnated" | "both") {
        eprintln!("--mode must be one of: baseline | incarnated | both (got '{mode}')");
        std::process::exit(2);
    }

    eprintln!("=== incarnation_forward ===");
    eprintln!("  model:               {model_path}");
    eprintln!("  dataset:             {dataset_path}");
    eprintln!("  n_samples:           {n_samples}");
    eprintln!("  ctx:                 {ctx}");
    eprintln!("  gate_layer:          {gate_layer}");
    eprintln!("  signal_threshold:    {signal_threshold:.6}");
    eprintln!("  signal_seed:         {signal_seed}");
    eprintln!("  signal_len:          {signal_len}");
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

    let mut model = Llama3Model::from_gguf(&gguf).expect("failed to load Llama3Model");
    eprintln!("  Model loaded.");

    // Tokenise the dataset (first `n_samples` tokens).
    let raw = std::fs::read_to_string(dataset_path).expect("failed to read dataset");
    let mut tokens = tokenizer.encode(&raw);
    if tokens.len() > n_samples {
        tokens.truncate(n_samples);
    }
    eprintln!("  Tokenised {} tokens.", tokens.len());

    if !skip_backward_compat {
        verify_backward_compat(&mut model, &tokens, 8);
    }

    let mut per_token_writer = output_path.map(|p| {
        BufWriter::new(File::create(p).expect("failed to open --output file for writing"))
    });

    let run_baseline = matches!(mode.as_str(), "baseline" | "both");
    let run_incarnated = matches!(mode.as_str(), "incarnated" | "both");

    if run_baseline {
        let summary = run(
            &mut model,
            &tokens,
            ctx,
            "baseline",
            None,
            per_token_writer.as_mut(),
            progress_every,
        );
        println!("{}", summary.to_json_line(""));
    }

    if run_incarnated {
        let summary = run(
            &mut model,
            &tokens,
            ctx,
            "incarnated",
            Some((gate_layer, signal_threshold, signal_seed, signal_len)),
            per_token_writer.as_mut(),
            progress_every,
        );
        let extra = format!(
            ",\"gate_layer\":{},\"signal_threshold\":{:.6},\
             \"signal_seed\":{},\"signal_len\":{}",
            gate_layer, signal_threshold, signal_seed, signal_len
        );
        println!("{}", summary.to_json_line(&extra));
    }
}
