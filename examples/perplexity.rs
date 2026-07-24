//! Perplexity measurement for GGUF models via ALICE-LLM's CPU forward path.
//!
//! Loads a GGUF model, tokenizes a raw text dataset (e.g. WikiText-2), and
//! computes perplexity (PPL) as `exp(mean(-log P(t_i | t_0..t_{i-1})))`.
//!
//! Two modes:
//!   - **chunked** (default, faster): non-overlapping chunks of `ctx` tokens.
//!     Each chunk resets the KV cache and scores positions `1..ctx`.
//!     Underestimates context length for early tokens in each chunk but is
//!     ~stride/ctx times faster than sliding.
//!   - **sliding** (llama.cpp `perplexity` default): overlapping chunks
//!     stepped by `stride`, scores only the last `stride` tokens per chunk
//!     with full `ctx-1` prior context. Higher-quality PPL, `ctx/stride`
//!     times slower.
//!
//! Uses `Llama3Model::from_gguf` (CPU only, no GPU allocation) so it works
//! for any GGUF architecture already supported by ALICE-LLM's `forward()`
//! (Llama 3 / Qwen 2 / 2.5 / 3 / 3.5 / Bonsai 27B via hybrid attention).
//!
//! # ⚠ Numerical divergence vs llama.cpp (2026-07-24 finding)
//!
//! On Mac M3 Metal, WikiText-2 test first 500 tokens, ctx=512, chunked mode:
//!   - ALICE-LLM  Qwen 3.5-4B Q4_K_M PPL = 16.38
//!   - llama.cpp  Qwen 3.5-4B Q4_K_M PPL =  6.09 ± 1.05  (--chunks 1)
//!   - divergence: 2.68× (ALICE-LLM inflated)
//!
//! Cause is NOT BOS handling — Qwen 3.5 GGUF defines no `bos_token_id` and
//! neither implementation prepends BOS in this configuration. The gap is
//! attributable to Qwen 3.5 forward path numerical drift in ALICE-LLM
//! (tracked as Phase X.3.e.3.30-35 in `memory/alice_llm_future_work.md`)
//! and possibly to Q4_K dequant precision differences.
//!
//! Bonsai 27B Q1_0 cannot be validated against llama.cpp: mainline
//! llama.cpp rejects Bonsai's custom Q1_0 quant type (ggml type 41).
//!
//! **Use this example for internal diagnostics only** until the ALICE-LLM
//! Qwen forward pass matches llama.cpp within ~5%. See
//! `memory/perplexity_alice_llm_vs_llamacpp_divergence_2026_07_24.md`
//! for the full validation snapshot + roadmap.
//!
//! Usage:
//!   cargo run --release --features gguf --example perplexity -- \
//!     --model models/Qwen3.5-4B-Q4_K_M.gguf \
//!     --dataset data/wikitext-2/wiki.test.raw \
//!     --ctx 2048 --stride 512 --n-samples 5000 --mode chunked
//!
//! Output: one JSON line on stdout with model / dataset / ctx / mode /
//! n_tokens_scored / ppl / nll_per_token / wall_sec / tok_per_sec.

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::llama3::{Llama3Config, Llama3Model};
use std::io::Write;
use std::time::Instant;

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn parse_arg_str<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

/// Compute log P(target | context) = logits[target] - logsumexp(logits),
/// numerically stable via the max-subtraction trick.
///
/// If `eog_inf_ids` is non-empty, those token indices have their logit
/// treated as -inf (excluded from the softmax denominator). This mirrors
/// llama.cpp's `common_init_result: added <tok> logit bias = -inf` for EOG
/// tokens, which shifts the denominator down and slightly reduces PPL.
fn log_prob_at(logits: &[f32], target: u32, eog_inf_ids: &[u32]) -> f32 {
    let target_us = target as usize;
    debug_assert!(
        target_us < logits.len(),
        "target token id out of vocab range"
    );
    let mut max = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if eog_inf_ids.contains(&(i as u32)) {
            continue;
        }
        if v > max {
            max = v;
        }
    }
    let mut sum_exp = 0.0_f64;
    for (i, &v) in logits.iter().enumerate() {
        if eog_inf_ids.contains(&(i as u32)) {
            continue;
        }
        sum_exp += ((v - max) as f64).exp();
    }
    let logsumexp = max + (sum_exp.ln() as f32);
    // Target is expected to not be in eog_inf_ids for real text scoring;
    // if it is, log_prob is -inf (which the caller may or may not want).
    if eog_inf_ids.contains(&target) {
        f32::NEG_INFINITY
    } else {
        logits[target_us] - logsumexp
    }
}

/// Escape a JSON string value (minimal — only `\`, `"`, control chars).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
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
            "Usage: perplexity --model <path.gguf> --dataset <path.txt> \\
                          [--ctx 2048] [--stride 512] [--n-samples 0] \\
                          [--mode chunked|sliding] [--progress-every 200]

  --model         Path to GGUF model file
  --dataset       Path to raw text dataset (UTF-8, plain text)
  --ctx           Context length per chunk (default: 2048)
  --stride        Sliding window stride (default: ctx/4, only used in sliding mode)
  --n-samples     Cap on tokens read from dataset (0 = all, default: 0)
  --mode          chunked (default, non-overlapping) or sliding
  --progress-every  Emit stderr progress every N scored tokens (default: 200)"
        );
        std::process::exit(1);
    }

    let model_path = parse_arg_str(&args, "--model").expect("--model required");
    let dataset_path = parse_arg_str(&args, "--dataset").expect("--dataset required");
    let ctx: usize = parse_arg(&args, "--ctx").unwrap_or(2048);
    let mode: String =
        parse_arg::<String>(&args, "--mode").unwrap_or_else(|| "chunked".to_string());
    let stride: usize = parse_arg(&args, "--stride").unwrap_or(ctx / 4);
    let n_samples: usize = parse_arg(&args, "--n-samples").unwrap_or(0);
    let progress_every: usize = parse_arg(&args, "--progress-every").unwrap_or(200);
    let prepend_bos: bool = args.iter().any(|a| a == "--prepend-bos");
    let dump_tokens: usize = parse_arg(&args, "--dump-tokens").unwrap_or(0);
    // --dump-layer L: after forwarding the first N tokens (--dump-layer-tokens),
    // print the hidden state observed at the START of layer L (before its
    // pre-attention RMSNorm). Format: sum + first 3 + last 3 values,
    // matching llama-eval-callback's common_debug_cb_eval output format so
    // we can visually compare. Uses forward_with_layer_hook so it works for
    // any GGUF architecture ALICE-LLM supports.
    let dump_layer: i32 = parse_arg(&args, "--dump-layer").unwrap_or(-1);
    let dump_layer_tokens: usize = parse_arg(&args, "--dump-layer-tokens").unwrap_or(2);
    let eog_inf_csv: String = parse_arg::<String>(&args, "--eog-inf").unwrap_or_default();
    let eog_inf_ids: Vec<u32> = if eog_inf_csv.is_empty() {
        Vec::new()
    } else {
        eog_inf_csv
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect()
    };

    if mode != "chunked" && mode != "sliding" {
        eprintln!("--mode must be 'chunked' or 'sliding', got '{mode}'");
        std::process::exit(2);
    }
    if mode == "sliding" && stride >= ctx {
        eprintln!("sliding mode requires stride < ctx ({stride} >= {ctx})");
        std::process::exit(2);
    }

    eprintln!("=== perplexity ===");
    eprintln!("  model:   {model_path}");
    eprintln!("  dataset: {dataset_path}");
    eprintln!("  ctx:     {ctx}");
    eprintln!(
        "  stride:  {stride} ({} tokens scored per {ctx}-token window)",
        if mode == "sliding" {
            stride
        } else {
            ctx.saturating_sub(1)
        }
    );
    eprintln!("  mode:    {mode}");
    eprintln!(
        "  n_samples: {}",
        if n_samples == 0 {
            "all".to_string()
        } else {
            n_samples.to_string()
        }
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
    let raw_full = std::fs::read_to_string(dataset_path).expect("failed to read dataset file");
    // Perf: pre-cap raw text when n_samples is set, since tokenize is O(chars).
    // Upper bound of ~10 chars/token × 2 safety margin = 20 chars/token cap.
    // Cut at nearest ASCII whitespace after cap to avoid partial UTF-8 / word split.
    let raw: &str = if n_samples > 0 && raw_full.len() > n_samples * 20 {
        let cap = n_samples * 20;
        let mut end = cap.min(raw_full.len());
        while end < raw_full.len() && !raw_full.as_bytes()[end].is_ascii_whitespace() {
            end += 1;
        }
        &raw_full[..end.min(raw_full.len())]
    } else {
        &raw_full
    };
    let mut tokens = tokenizer.encode(raw);
    if n_samples > 0 && tokens.len() > n_samples {
        tokens.truncate(n_samples);
    }
    if prepend_bos {
        let decoded = tokenizer.decode(&[tokenizer.bos_id]);
        tokens.insert(0, tokenizer.bos_id);
        eprintln!(
            "  Prepended BOS token id {} (decoded: {:?}) at position 0",
            tokenizer.bos_id, decoded
        );
        if tokenizer.bos_id <= 2 {
            eprintln!(
                "  WARNING: bos_id {} looks like the default fallback \
                 (`gguf.meta_u32(\"tokenizer.ggml.bos_token_id\").unwrap_or(1)`). \
                 This model's GGUF likely does not define a BOS token \
                 (Qwen 2/3/3.5 do not use BOS). Prepending a random low-id \
                 token will inflate PPL, not reduce it. Consider running \
                 without --prepend-bos for such models.",
                tokenizer.bos_id
            );
        }
    }
    let n_dataset_tokens = tokens.len();
    eprintln!(
        "  Dataset: raw={} chars, cap-scanned={} chars, tokens={} ({}ms)",
        raw_full.len(),
        raw.len(),
        n_dataset_tokens,
        t_tok.elapsed().as_millis()
    );

    if n_dataset_tokens < 2 {
        eprintln!("dataset too short ({} tokens)", n_dataset_tokens);
        std::process::exit(3);
    }

    if dump_tokens > 0 {
        for t in tokens.iter().take(dump_tokens) {
            println!("{}", t);
        }
        return;
    }

    if dump_layer >= 0 {
        let target_layer = dump_layer as usize;
        model.reset();
        for (pos, &tok) in tokens.iter().take(dump_layer_tokens).enumerate() {
            let mut captured: Option<Vec<f32>> = None;
            let _ = model.forward_with_layer_hook(tok, |lyr, hidden| {
                if lyr == target_layer && captured.is_none() {
                    captured = Some(hidden.clone());
                }
                false // do not skip CPU layer body
            });
            if let Some(h) = captured {
                let sum: f64 = h.iter().map(|&x| x as f64).sum();
                let n = h.len();
                let first3 = if n >= 3 {
                    format!("{:>10.4}, {:>10.4}, {:>10.4}", h[0], h[1], h[2])
                } else {
                    "n<3".to_string()
                };
                let last3 = if n >= 3 {
                    format!("{:>10.4}, {:>10.4}, {:>10.4}", h[n - 3], h[n - 2], h[n - 1])
                } else {
                    "n<3".to_string()
                };
                println!(
                    "layer={} pos={} tok={} shape={} sum={:.6} first3=[{}] last3=[{}]",
                    target_layer, pos, tok, n, sum, first3, last3
                );
            } else {
                println!(
                    "layer={} pos={} tok={} NOT CAPTURED (hook fired but layer index mismatch)",
                    target_layer, pos, tok
                );
            }
        }
        return;
    }

    let t_ppl = Instant::now();
    let mut sum_nll = 0.0_f64;
    let mut count: usize = 0;
    let mut last_report_count: usize = 0;
    let mut last_report_time = Instant::now();

    let vocab_size = tokenizer.vocab_size() as u32;

    let report = |scored: usize, sum_nll: f64, elapsed_sec: f64, is_final: bool| {
        let ppl = (sum_nll / scored as f64).exp();
        let nll_per_tok = sum_nll / scored as f64;
        let tok_per_sec = scored as f64 / elapsed_sec;
        let tag = if is_final { "FINAL" } else { "progress" };
        eprintln!(
            "  [{tag}] scored={scored} ppl={ppl:.4} nll/tok={nll_per_tok:.4} \
             elapsed={elapsed_sec:.1}s tok/s={tok_per_sec:.2}"
        );
    };

    match mode.as_str() {
        "chunked" => {
            eprintln!(
                "Running chunked mode: {} chunks of {ctx} tokens (last chunk may be shorter)",
                n_dataset_tokens.div_ceil(ctx)
            );
            for (chunk_idx, chunk) in tokens.chunks(ctx).enumerate() {
                if chunk.len() < 2 {
                    break;
                }
                model.reset();
                let mut prev_logits: Option<Vec<f32>> = None;
                for &tok in chunk {
                    if let Some(logits) = prev_logits.as_ref() {
                        if (tok as usize) < logits.len() {
                            let log_p = log_prob_at(logits, tok, &eog_inf_ids);
                            sum_nll += (-log_p) as f64;
                            count += 1;
                        }
                    }
                    prev_logits = Some(model.forward(tok));

                    if count > 0 && count.saturating_sub(last_report_count) >= progress_every {
                        let elapsed = t_ppl.elapsed().as_secs_f64();
                        report(count, sum_nll, elapsed, false);
                        last_report_count = count;
                        last_report_time = Instant::now();
                        let _ = std::io::stderr().flush();
                    }
                }
                if chunk_idx == 0 {
                    eprintln!("  (chunk 0 done, {} tokens scored so far)", count);
                }
                let _ = last_report_time; // silence unused if progress never fires
            }
        }
        "sliding" => {
            let context_size = ctx - stride;
            let n_chunks = n_dataset_tokens
                .saturating_sub(context_size)
                .div_ceil(stride);
            eprintln!(
                "Running sliding mode: ~{n_chunks} chunks of {ctx} tokens, \
                 stride={stride}, score-from={context_size} in each chunk (chunk 0 scores from 1)"
            );
            let mut chunk_start: usize = 0;
            let mut chunk_idx: usize = 0;
            while chunk_start + 2 <= n_dataset_tokens {
                let chunk_end = (chunk_start + ctx).min(n_dataset_tokens);
                let chunk = &tokens[chunk_start..chunk_end];
                if chunk.len() < 2 {
                    break;
                }
                model.reset();
                let score_from = if chunk_idx == 0 { 1 } else { context_size };
                let mut prev_logits: Option<Vec<f32>> = None;
                for (pos, &tok) in chunk.iter().enumerate() {
                    if pos >= score_from {
                        if let Some(logits) = prev_logits.as_ref() {
                            if (tok as usize) < logits.len() {
                                let log_p = log_prob_at(logits, tok, &eog_inf_ids);
                                sum_nll += (-log_p) as f64;
                                count += 1;
                            }
                        }
                    }
                    prev_logits = Some(model.forward(tok));

                    if count > 0 && count.saturating_sub(last_report_count) >= progress_every {
                        let elapsed = t_ppl.elapsed().as_secs_f64();
                        report(count, sum_nll, elapsed, false);
                        last_report_count = count;
                        let _ = std::io::stderr().flush();
                    }
                }
                chunk_idx += 1;
                if chunk_end == n_dataset_tokens {
                    break;
                }
                chunk_start += stride;
            }
        }
        _ => unreachable!(),
    }

    let wall_sec = t_ppl.elapsed().as_secs_f64();
    if count == 0 {
        eprintln!("no tokens scored — check --n-samples / --ctx / dataset length");
        std::process::exit(4);
    }
    report(count, sum_nll, wall_sec, true);

    let ppl = (sum_nll / count as f64).exp();
    let nll_per_tok = sum_nll / count as f64;
    let tok_per_sec = count as f64 / wall_sec;

    // Single-line JSON output on stdout for scripted collection.
    let json = format!(
        "{{\"model\":\"{}\",\"dataset\":\"{}\",\"mode\":\"{}\",\
         \"ctx\":{},\"stride\":{},\"n_tokens_dataset\":{},\"n_tokens_scored\":{},\
         \"vocab_size\":{},\"ppl\":{:.6},\"nll_per_token\":{:.6},\
         \"wall_sec\":{:.3},\"tok_per_sec\":{:.4}}}",
        json_escape(model_path),
        json_escape(dataset_path),
        mode,
        ctx,
        stride,
        n_dataset_tokens,
        count,
        vocab_size,
        ppl,
        nll_per_tok,
        wall_sec,
        tok_per_sec,
    );
    println!("{json}");
}
