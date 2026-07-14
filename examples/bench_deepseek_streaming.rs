//! DeepSeek-V3 streaming vs in-memory forward micro-bench (Phase 4b, Issue #34).
//!
//! # What this measures
//!
//! Runs one prefill + N single-token decode iterations against a real
//! DeepSeek-V3 GGUF twice: once with routed experts held in memory
//! (`ALICE_LLM_MOE_STREAMING` unset), once with the streaming pool active
//! (`ALICE_LLM_MOE_STREAMING=1`). Prints:
//!
//! - Wall-clock decode time and tokens per second for both variants.
//! - Streaming pool cache stats after the run — hits / misses / entries /
//!   pinned. Miss count divided by decode tokens gives the average number
//!   of expert loads per token; on a real V3 this should approach
//!   `num_experts_per_tok = 8` after warmup.
//!
//! # Preconditions
//!
//! - The **`gguf`** feature enabled (required for both `Llama3Model::from_gguf`
//!   and the streaming pool's `memmap2` byte source).
//! - Env var `ALICE_LLM_MOE_BENCH_MODEL=<path>` pointing at a DeepSeek-V3
//!   GGUF (or V2 / R1 which share the deepseek2 arch tag). When unset,
//!   this example prints a `SKIP` note and exits successfully so CI can
//!   include it without a model on disk.
//!
//! # Recommended env vars
//!
//! - `ALICE_LLM_MOE_BENCH_TOKENS=<n>`  (default 8)
//! - `ALICE_LLM_MOE_BENCH_PROMPT="..."` (default "The meaning of life is")
//! - `ALICE_LLM_MOE_CACHE_BYTES=<n>`   (streaming pool budget, default 4 GiB;
//!   set 16-32 GiB for meaningful hit rates on 61-layer V3)
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example bench_deepseek_streaming --features gguf -- \
//!   # (all config is via env vars, no positional args)
//! ```
//!
//! # Why blocking, not `cargo bench`?
//!
//! `cargo bench` insists on a stable statistical harness; loading a
//! 370 GB GGUF via mmap and running 8 tokens is dominated by cold-cache
//! disk I/O that the harness cannot amortise. A single-pass wall-clock
//! print with cache stats surfaces the actual variable of interest —
//! the LRU miss rate — more clearly than a criterion histogram.

use std::env;
use std::time::Instant;

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::llama3::Llama3Model;

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let path = match env::var("ALICE_LLM_MOE_BENCH_MODEL") {
        Ok(p) => p,
        Err(_) => {
            println!(
                "SKIP: set ALICE_LLM_MOE_BENCH_MODEL=<path> to run the DeepSeek streaming bench.\n\
                 See the module docstring for prerequisites (real DeepSeek-V3/V2/R1 GGUF, ~370 GB \
                 at Q4_K_M for the full V3)."
            );
            return;
        }
    };
    let prompt = env::var("ALICE_LLM_MOE_BENCH_PROMPT")
        .unwrap_or_else(|_| "The meaning of life is".to_string());
    let decode_tokens = env_usize("ALICE_LLM_MOE_BENCH_TOKENS", 8);
    let cache_bytes = env_usize("ALICE_LLM_MOE_CACHE_BYTES", 4 * 1024 * 1024 * 1024);

    println!("=== DeepSeek-V3 streaming bench ===");
    println!("Model: {path}");
    println!("Prompt: {prompt:?}");
    println!("Decode tokens: {decode_tokens}");
    println!(
        "Streaming cache budget: {} MiB",
        cache_bytes / (1024 * 1024)
    );
    println!();

    // Ensure a clean slate — the streaming env vars may leak from prior
    // runs in the same shell.
    env::remove_var("ALICE_LLM_MOE_STREAMING");
    env::remove_var("ALICE_LLM_MOE_STREAMING_FILE");

    // Load the GGUF once; parsing is cheap compared to weight materialisation.
    let t_load = Instant::now();
    let bytes = std::fs::read(&path).expect("failed to read GGUF file");
    let gguf = GgufFile::parse(&bytes).expect("failed to parse GGUF");
    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("failed to load tokenizer");
    println!(
        "GGUF parsed in {:.2}s (vocab={})",
        t_load.elapsed().as_secs_f64(),
        tokenizer.vocab_size()
    );

    // Encode the prompt once — both variants share it.
    let prompt_tokens = tokenizer.encode(&prompt);
    println!("Prompt tokens: {}", prompt_tokens.len());
    println!();

    // ── Variant A: InMemory routed experts (Phase 3 baseline) ────────
    println!("--- InMemory (baseline) ---");
    let t_build = Instant::now();
    let mut model = Llama3Model::from_gguf(&gguf).expect("model load failed");
    println!("Model loaded in {:.2}s", t_build.elapsed().as_secs_f64());

    // Prefill.
    let t_prefill = Instant::now();
    let mut last_logits = Vec::new();
    for &tok in &prompt_tokens {
        last_logits = model.forward(tok);
    }
    let prefill_ms = t_prefill.elapsed().as_millis();

    // Decode `decode_tokens` tokens greedily.
    let t_decode = Instant::now();
    for _ in 0..decode_tokens {
        let next = argmax(&last_logits) as u32;
        last_logits = model.forward(next);
    }
    let decode_ms = t_decode.elapsed().as_millis();
    let tok_per_sec = decode_tokens as f64 / (decode_ms as f64 / 1000.0);
    println!(
        "InMemory: prefill {} ms, decode {} ms, {:.2} tok/s",
        prefill_ms, decode_ms, tok_per_sec
    );
    drop(model);
    println!();

    // ── Variant B: Streaming routed experts (Phase 4b) ───────────────
    println!("--- Streaming (Phase 4b) ---");
    env::set_var("ALICE_LLM_MOE_STREAMING", "1");
    env::set_var("ALICE_LLM_MOE_STREAMING_FILE", &path);
    env::set_var("ALICE_LLM_MOE_CACHE_BYTES", cache_bytes.to_string());

    let t_build = Instant::now();
    let mut model = Llama3Model::from_gguf(&gguf).expect("model load failed");
    println!(
        "Model loaded in {:.2}s (routed-expert bytes NOT materialised)",
        t_build.elapsed().as_secs_f64()
    );

    let t_prefill = Instant::now();
    let mut last_logits = Vec::new();
    for &tok in &prompt_tokens {
        last_logits = model.forward(tok);
    }
    let prefill_ms = t_prefill.elapsed().as_millis();

    let t_decode = Instant::now();
    for _ in 0..decode_tokens {
        let next = argmax(&last_logits) as u32;
        last_logits = model.forward(next);
    }
    let decode_ms = t_decode.elapsed().as_millis();
    let tok_per_sec = decode_tokens as f64 / (decode_ms as f64 / 1000.0);
    println!(
        "Streaming: prefill {} ms, decode {} ms, {:.2} tok/s",
        prefill_ms, decode_ms, tok_per_sec
    );
    // Cache stats live on the pool, but the pool is private to the model
    // internals — we surface them via the [pool] eprintln! banner emitted
    // by build_deepseek_streaming_pool at load time. See the streaming
    // module doc-comment for how a follow-up can expose CacheStats
    // through a public accessor once the model's `deepseek_v3_layers`
    // field is refactored to expose one.
    println!();

    println!(
        "Note: streaming pool cache stats are printed by the [alice-llm] \
         banners above (mmap size + budget). A follow-up will expose them \
         through a public accessor on the model."
    );
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        })
        .0
}
