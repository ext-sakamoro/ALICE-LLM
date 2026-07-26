//! Metal (or Vulkan / DX12) GPU determinism ε-bound bench.
//!
//! Runs `GpuModel::forward_and_read` twice with the same seed + token +
//! initial KV state, computes the max absolute difference on the logits
//! vectors, and reports it. Also runs the new
//! `GpuModel::forward_with_early_exit_and_read` at the full depth
//! (`early_exit_layer = num_layers`) to assert parity with the existing
//! full-forward path.
//!
//! Purpose: the surprise-driven early-exit GPU forward
//! (`forward_with_early_exit_and_read` +
//! `forward_with_surprise_and_read`, added in v1.5.0) preserves the
//! deterministic contract of the CPU path only if the underlying GPU
//! runtime itself is deterministic across runs. This bench provides the
//! empirical ε bound downstream users can quote when reasoning about
//! whether a bit-exact same-seed → same-logits invariant can be
//! promised on the GPU path.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example metal_determinism_check \
//!     --features "gguf,gpu" -- \
//!     --model ~/ALICE-LLM/models/Qwen3.5-4B-Q4_K_M.gguf \
//!     --token 42 --iterations 3
//! ```

use std::path::PathBuf;
use std::time::Instant;

use alice_llm::gguf::GgufFile;
use alice_llm::llama3::Llama3Config;

#[cfg(feature = "gpu")]
use alice_llm::gpu::{GpuEngine, GpuModel, GpuModelConfig};

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
        .map(String::as_str)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "logits length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "metal_determinism_check requires the `gpu` feature — rerun with \
         `--features gguf,gpu`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "Usage: metal_determinism_check --model <path.gguf> [--token N] [--iterations N]

Reports the max |Δlogits| between paired same-seed forwards on the GPU
path, plus a parity check between forward_and_read and
forward_with_early_exit_and_read(num_layers)."
        );
        std::process::exit(0);
    }

    let model_path: PathBuf = parse_arg_str(&args, "--model")
        .map(PathBuf::from)
        .expect("--model <path.gguf> is required");
    let token: u32 = parse_arg(&args, "--token").unwrap_or(42);
    let iterations: usize = parse_arg(&args, "--iterations").unwrap_or(3);

    eprintln!("=== metal_determinism_check (v1.5.0) ===");
    eprintln!("  model:        {}", model_path.display());
    eprintln!("  token:        {token}");
    eprintln!("  iterations:   {iterations}");
    eprintln!();

    let t_load = Instant::now();
    let bytes = std::fs::read(&model_path).expect("read GGUF");
    let gguf = GgufFile::parse(&bytes).expect("parse GGUF");
    let config = Llama3Config::from_gguf(&gguf).expect("Llama3Config");
    eprintln!(
        "  GGUF parsed: {}ms (layers={} vocab={} hidden={})",
        t_load.elapsed().as_millis(),
        config.num_layers,
        config.vocab_size,
        config.hidden_dim,
    );

    let engine = GpuEngine::new();
    // Mirror the qwen_gpu.rs field-by-field construction — GpuModelConfig
    // does not expose an `from_llama3_config` constructor as of v1.5.0.
    let gpu_cfg = GpuModelConfig {
        num_layers: config.num_layers,
        hidden_dim: config.hidden_dim,
        intermediate_dim: config.intermediate_dim,
        num_heads: config.num_heads as u32,
        num_kv_heads: config.num_kv_heads as u32,
        head_dim: config.head_dim as u32,
        rope_theta: config.rope_theta,
        eps: config.norm_eps,
        max_seq_len: config.max_seq_len,
        full_attention_interval: config.full_attention_interval(),
        linear_num_kv_heads: config.linear_num_kv_heads().map(|v| v as u32),
        linear_qk_head_dim: config.linear_qk_head_dim().map(|v| v as u32),
        linear_kv_head_dim: config.linear_kv_head_dim().map(|v| v as u32),
        linear_num_v_heads: config.linear_num_v_heads().map(|v| v as u32),
        linear_conv_kernel_dim: config.linear_conv_kernel_dim().map(|v| v as u32),
        neox_rope: config.arch.use_neox_rope(),
        attention_only_load: false,
    };

    eprintln!("  Loading GPU model (Metal shader compile is one-time ~10-30 s)...");
    let t_gpu_load = Instant::now();
    let mut gpu_model = GpuModel::load(engine, &gguf, gpu_cfg);
    eprintln!("  GPU model loaded: {}ms", t_gpu_load.elapsed().as_millis());
    eprintln!();

    let mut max_pair_diff: f32 = 0.0;
    for i in 0..iterations {
        gpu_model.reset();
        let a = gpu_model.forward_and_read(token);
        gpu_model.reset();
        let b = gpu_model.forward_and_read(token);
        let diff = max_abs_diff(&a, &b);
        eprintln!("  pair {i:>2}: forward_and_read × 2, max |Δlogits| = {diff:.6e}");
        if diff > max_pair_diff {
            max_pair_diff = diff;
        }
    }

    eprintln!();
    eprintln!("=== ε bound summary ===");
    eprintln!("  max across {iterations} paired runs: {max_pair_diff:.6e}");
    if max_pair_diff == 0.0 {
        eprintln!("  → GPU forward is bit-exact deterministic on this host");
    } else {
        eprintln!("  → GPU forward has non-zero ε bound on this host");
    }
    eprintln!();

    gpu_model.reset();
    let c = gpu_model.forward_and_read(token);
    gpu_model.reset();
    let d = gpu_model.forward_with_early_exit_and_read(token, config.num_layers);
    let parity_diff = max_abs_diff(&c, &d);
    eprintln!("=== parity check (early_exit = num_layers vs full forward) ===");
    eprintln!(
        "  max |Δlogits| = {parity_diff:.6e}  \
         (must be ≤ ε bound {max_pair_diff:.6e})"
    );
    if parity_diff <= max_pair_diff + f32::EPSILON {
        eprintln!("  → PASS — early_exit at full depth matches plain forward within ε");
    } else {
        eprintln!(
            "  → FAIL — early_exit at full depth exceeds ε bound, \
             new dispatch path may have introduced drift"
        );
        std::process::exit(1);
    }

    eprintln!();
    eprintln!("metal_determinism_check: complete");
}
