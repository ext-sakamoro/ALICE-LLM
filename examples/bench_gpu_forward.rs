//! Full 16-layer GPU forward pass benchmark using GpuModel.
//! Zero per-token allocation — all bind groups pre-cached at load time.
//!
//! Usage:
//!   cargo run --example bench_gpu_forward --features gpu,gguf --release
//!   cargo run --example bench_gpu_forward --features gpu,gguf --release -- --profile
//!   cargo run --example bench_gpu_forward --features gpu,gguf --release -- --batch

use alice_llm::gguf::GgufFile;
use std::fs;
use std::time::Instant;

#[cfg(feature = "gpu")]
use alice_llm::gpu::{GpuEngine, GpuModel, GpuModelConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let profile_mode = args.iter().any(|a| a == "--profile");
    let batch_mode = args.iter().any(|a| a == "--batch");
    let model_path = args.iter().find(|a| !a.starts_with('-') && **a != args[0])
        .cloned()
        .unwrap_or_else(|| {
            "/Users/ys/models/llama-3.2-1b-gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_string()
        });

    println!("Loading model: {model_path}");
    let data = fs::read(&model_path).unwrap();
    let gguf = GgufFile::parse(&data).unwrap();

    let config = GpuModelConfig {
        num_layers: 16,
        hidden_dim: 2048,
        intermediate_dim: 8192,
        num_heads: 32,
        num_kv_heads: 8,
        head_dim: 64,
        rope_theta: 500000.0,
        eps: 1e-5,
        max_seq_len: 2048,
    };

    println!(
        "Llama-3.2-1B: {} layers, hidden={}, inter={}, heads={}/{}, head_dim={}",
        config.num_layers, config.hidden_dim, config.intermediate_dim,
        config.num_heads, config.num_kv_heads, config.head_dim,
    );

    #[cfg(feature = "gpu")]
    {
        println!("\nInitializing GpuModel (static bind groups)...");
        let t_load = Instant::now();
        let engine = GpuEngine::new();
        let engine_period = engine.timestamp_period;
        let mut model = GpuModel::load(engine, &gguf, config);
        println!("GpuModel ready: {:.0}ms", t_load.elapsed().as_millis());

        let test_token: u32 = 1; // BOS token

        if batch_mode {
            // --- Batch-4 benchmark: K=1 (scalar) vs K=4 (unrolled) ---
            println!("\n--- Batch-4 Benchmark (K=1 scalar vs K=4 unrolled) ---\n");

            let n_warmup = 5;
            let n_iter = 30;

            // K=1: original scalar fast path
            for _ in 0..n_warmup { model.forward(test_token); }
            model.sync();
            model.reset();

            let t0 = Instant::now();
            for _ in 0..n_iter { model.forward(test_token); }
            model.sync();
            let k1_us = t0.elapsed().as_micros() as f64 / n_iter as f64;
            model.reset();

            // K=4: batch4 unrolled scalar path
            let batch_tokens = [1u32, 1, 1, 1];
            for _ in 0..n_warmup { model.forward_batch(&batch_tokens); }
            model.sync();
            model.reset();

            let t0 = Instant::now();
            for _ in 0..n_iter { model.forward_batch(&batch_tokens); }
            model.sync();
            let k4_us = t0.elapsed().as_micros() as f64 / n_iter as f64;
            let k4_per_token = k4_us / 4.0;
            model.reset();

            println!("K=1 (scalar):    {k1_us:.0} µs/token  ({:.1} tok/s)", 1e6 / k1_us);
            println!("K=4 (unrolled):  {k4_us:.0} µs/batch  ({:.1} ms)", k4_us / 1000.0);
            println!("  per token:     {k4_per_token:.0} µs/token  ({:.1} tok/s)", 1e6 / k4_per_token);
            println!("  speedup:       {:.2}×", k1_us / k4_per_token);

            // Correctness: compare single-token vs batch logits
            println!("\nCorrectness check: single vs batch(K=4) logits...");
            model.reset();
            let single_logits = model.forward_and_read(1);
            model.reset();
            let batch_logits = model.forward_batch_and_read(&[1, 1, 1, 1]);

            let first_batch_logits = &batch_logits[..model.vocab_size()];
            let max_diff = single_logits.iter()
                .zip(first_batch_logits.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let valid_single = single_logits.iter().filter(|v| !v.is_nan()).count();
            let valid_batch = first_batch_logits.iter().filter(|v| !v.is_nan()).count();
            println!(
                "Token 0 max|diff|: {max_diff:.6} (single: {valid_single} valid, batch: {valid_batch} valid)"
            );
            if max_diff < 0.01 {
                println!("PASS: batch-4 logits match single-token logits");
            } else {
                println!("WARN: logits differ (max_diff={max_diff:.4})");
            }
        } else if profile_mode {
            // --- Profile mode: per-operation timing breakdown ---
            println!("\n--- Profile Mode ---");

            // Warmup
            for _ in 0..3 {
                model.forward(test_token);
            }
            model.sync();
            model.reset();

            // Run GPU-timestamp profiled forward pass (averaged over N iterations)
            let n_profile = 5;
            let mut totals = [0.0f64; 8];
            let mut counts = [0u32; 8];

            println!("timestamp_period: {} ns/tick", engine_period);

            for _ in 0..n_profile {
                let r = model.forward_gpu_profiled(test_token);
                totals[0] += r.rmsnorm_us;
                totals[1] += r.matvec_q4k_us;
                totals[2] += r.matvec_q6k_us;
                totals[3] += r.swiglu_us;
                totals[4] += r.rope_us;
                totals[5] += r.kv_append_us;
                totals[6] += r.attention_us;
                totals[7] += r.residual_us;
                counts[0] = r.rmsnorm_n;
                counts[1] = r.matvec_q4k_n;
                counts[2] = r.matvec_q6k_n;
                counts[3] = r.swiglu_n;
                counts[4] = r.rope_n;
                counts[5] = r.kv_append_n;
                counts[6] = r.attention_n;
                counts[7] = r.residual_n;
            }

            let labels = [
                "rmsnorm", "matvec_q4k", "matvec_q6k", "swiglu",
                "rope", "kv_append", "attention", "residual",
            ];
            let grand_total: f64 = totals.iter().sum::<f64>() / n_profile as f64;

            println!("\n{:<14} {:>6} {:>10} {:>10} {:>7}",
                "Operation", "Count", "Total(µs)", "Avg(µs)", "Share");
            println!("{}", "-".repeat(55));
            for i in 0..8 {
                let avg_total = totals[i] / n_profile as f64;
                let avg_per = if counts[i] > 0 { avg_total / counts[i] as f64 } else { 0.0 };
                let share = avg_total / grand_total * 100.0;
                println!("{:<14} {:>6} {:>10.0} {:>10.1} {:>6.1}%",
                    labels[i], counts[i], avg_total, avg_per, share);
            }
            println!("{}", "-".repeat(55));
            println!("{:<14} {:>6} {:>10.0} {:>10} {:>6}",
                "TOTAL", counts.iter().sum::<u32>(), grand_total, "", "100%");
            println!("\n{:.1} ms/token (GPU timestamps, zero sync overhead)", grand_total / 1000.0);
        } else {
            // --- Normal benchmark mode ---
            let n_warmup = 5;
            let n_iter = 50;

            println!("\nWarmup ({n_warmup} iterations)...");
            for _ in 0..n_warmup {
                model.forward(test_token);
            }
            model.sync();
            model.reset();

            println!("\n--- GpuModel Forward Benchmark ({n_iter} tokens, pipelined) ---");
            let t0 = Instant::now();
            for _ in 0..n_iter {
                model.forward(test_token);
            }
            model.sync();
            let total_us = t0.elapsed().as_micros() as f64;
            let avg_us = total_us / n_iter as f64;
            let final_pos = model.position();
            println!(
                "{avg_us:.0} µs/token avg ({n_iter} tokens, {:.1}ms total, final pos={final_pos})",
                total_us / 1000.0,
            );

            // Correctness check
            println!("\nCorrectness check (logits readback at pos=0)...");
            model.reset();
            let logits = model.forward_and_read(test_token);

            let mut indexed: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            println!("Top-5 logits (token_id: logit):");
            for (id, logit) in indexed.iter().take(5) {
                println!("  {id:6}: {logit:+.4}");
            }

            let valid = logits.iter().filter(|v| !v.is_nan() && !v.is_infinite()).count();
            let nan_count = logits.iter().filter(|v| v.is_nan()).count();
            println!(
                "Logits: {valid} valid, {nan_count} NaN, {} total (vocab={})",
                logits.len(),
                model.vocab_size(),
            );
        }
    }

    #[cfg(not(feature = "gpu"))]
    println!("GPU feature not enabled. Run with: cargo run --example bench_gpu_forward --features gpu,gguf --release");
}
