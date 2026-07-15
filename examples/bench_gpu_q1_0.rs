//! Standalone Q1_0 (PrismML Bonsai binary g128) GPU matvec bench + parity test.
//!
//! Usage:
//!   cargo run --release --example bench_gpu_q1_0 --features gguf,gpu -- \
//!     [path/to/Bonsai-27B-Q1_0.gguf]
//!
//! Picks a Q1_0 tensor from the model (`blk.0.attn_qkv.weight` if present,
//! otherwise the first Q1_0 tensor found), runs the CPU NEON matvec kernel
//! and the wgpu compute-shader matvec kernel over 100 iterations each, and
//! prints:
//!
//! - The CPU / GPU per-matvec latency.
//! - The relative L2 error between the two outputs (a sanity check that the
//!   WGSL shader matches the CPU reference within FP summation noise).
//!
//! On a memory-bound decode workload, GPU matvec should approach the device
//! memory bandwidth × row byte cost lower bound. On Jetson Orin Nano 8GB
//! (Vulkan, Ampere), Bonsai 27B's `attn_qkv` is ~7 MB / row-family × 40 blocks
//! → the theoretical ceiling is on the order of 100+ µs / matvec, orders of
//! magnitude faster than the CPU NEON `q1_0_dot_row_pos_only` path at ~10 s /
//! full-layer stack.
//!
//! This bench isolates the single-tensor cost so the wgpu shader can be
//! optimised without dragging the full 64-layer inference stack behind it.

use alice_llm::gguf::{GgmlType, GgufFile};
use std::fs;
use std::time::Instant;

#[cfg(feature = "gpu")]
use alice_llm::gpu::GpuEngine;

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/Bonsai-27B-Q1_0.gguf".to_string());

    println!("Loading GGUF: {model_path}");
    let data = fs::read(&model_path).expect("failed to read GGUF file");
    let gguf = GgufFile::parse(&data).expect("failed to parse GGUF");

    // Pick a Q1_0 tensor. Prefer `attn_qkv` from layer 0 (Bonsai DeltaNet
    // fused Q+K+V, the largest per-forward matvec) if present.
    let (tensor_name, info) = ["blk.0.attn_qkv.weight", "blk.0.ffn_gate.weight"]
        .iter()
        .find_map(|name| gguf.tensor_info(name).map(|info| (name.to_string(), info)))
        .or_else(|| {
            // Fall back to whichever Q1_0 tensor comes first.
            gguf.tensors
                .iter()
                .find(|(_, info)| info.qtype == GgmlType::Q1_0)
                .map(|(name, info)| (name.clone(), info))
        })
        .expect("no Q1_0 tensor found in this GGUF");

    assert_eq!(
        info.qtype,
        GgmlType::Q1_0,
        "expected Q1_0 tensor, got {:?}",
        info.qtype
    );

    let cols = info.dims[0] as usize;
    let rows = info.dims[1] as usize;
    let tensor_data = gguf.tensor_data(&tensor_name).expect("tensor data missing");

    println!(
        "Tensor: {tensor_name} ({rows} × {cols}, Q1_0, {:.2} MB)",
        tensor_data.len() as f64 / 1_048_576.0
    );

    // Deterministic input vector.
    let input: Vec<f32> = (0..cols)
        .map(|i| ((i as f32) * 0.017).sin() * 0.1)
        .collect();

    // ---- CPU (NEON on aarch64, scalar fallback elsewhere) ----
    let mut cpu_output = vec![0.0f32; rows];
    alice_llm::gguf::quantized_matvec(
        &input,
        tensor_data,
        GgmlType::Q1_0,
        rows,
        cols,
        &mut cpu_output,
    );

    let n_iter = 20;
    let t0 = Instant::now();
    for _ in 0..n_iter {
        alice_llm::gguf::quantized_matvec(
            &input,
            tensor_data,
            GgmlType::Q1_0,
            rows,
            cols,
            &mut cpu_output,
        );
    }
    let cpu_us = t0.elapsed().as_micros() as f64 / n_iter as f64;
    println!("CPU matvec: {cpu_us:>10.1} µs/iter ({n_iter} iters)");

    // ---- GPU ----
    #[cfg(feature = "gpu")]
    {
        println!("\nInitialising GPU...");
        let engine = GpuEngine::new();
        println!("  device_type: {:?}", engine.device_type);

        let weights = engine.upload_weights_q1_0(tensor_data, rows, cols);
        let input_buf = engine.upload_f32(&input);
        let output_buf = engine.alloc_f32(rows);

        // Warmup (compile + first dispatch cost)
        for _ in 0..5 {
            let mut pass = engine.begin_pass();
            pass.matvec_q1_0(&weights, &input_buf, &output_buf);
            pass.execute();
        }

        let t0 = Instant::now();
        for _ in 0..n_iter {
            let mut pass = engine.begin_pass();
            pass.matvec_q1_0(&weights, &input_buf, &output_buf);
            pass.execute();
        }
        engine.sync();
        let gpu_us = t0.elapsed().as_micros() as f64 / n_iter as f64;

        let gpu_output = engine.read_f32(&output_buf);
        println!("GPU matvec: {gpu_us:>10.1} µs/iter ({n_iter} iters)");
        println!("\nSpeedup CPU / GPU: {:.2}×", cpu_us / gpu_us);

        // Parity check
        let mut l2_num = 0.0f64;
        let mut l2_den = 0.0f64;
        let mut max_abs_err = 0.0f32;
        for (c, g) in cpu_output.iter().zip(gpu_output.iter()) {
            let diff = (c - g).abs();
            max_abs_err = max_abs_err.max(diff);
            l2_num += (diff as f64) * (diff as f64);
            l2_den += (*c as f64) * (*c as f64);
        }
        let l2_rel = (l2_num / l2_den.max(1e-30)).sqrt();
        println!(
            "\nParity: L2_rel = {:.3e}, max_abs_err = {:.3e}",
            l2_rel, max_abs_err
        );
        println!(
            "  CPU[0..4]  = {:?}",
            &cpu_output[..4.min(cpu_output.len())]
        );
        println!(
            "  GPU[0..4]  = {:?}",
            &gpu_output[..4.min(gpu_output.len())]
        );
    }
}
