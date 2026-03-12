//! FFN chain benchmark: GPU vs CPU for a complete SwiGLU FFN block.
//! RMSNorm → gate matvec → up matvec → SiLU(gate)*up → down matvec → residual add

use alice_llm::gguf::{quantized_matvec, GgmlType, GgufFile};
use std::fs;
use std::time::Instant;

#[cfg(feature = "gpu")]
use alice_llm::gpu::GpuEngine;

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mut ss = 0.0f64;
    for &v in x {
        ss += (v as f64) * (v as f64);
    }
    let scale = 1.0f32 / ((ss / n as f64) as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
}

fn main() {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/Users/ys/models/llama-3.2-1b-gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_string()
    });

    println!("Loading model: {model_path}");
    let data = fs::read(&model_path).unwrap();
    let gguf = GgufFile::parse(&data).unwrap();

    // Extract layer 0 FFN weights
    let gate_info = gguf.tensor_info("blk.0.ffn_gate.weight").unwrap();
    let _up_info = gguf.tensor_info("blk.0.ffn_up.weight").unwrap();
    let _down_info = gguf.tensor_info("blk.0.ffn_down.weight").unwrap();
    let norm_data = gguf.tensor_to_f32("blk.0.ffn_norm.weight").unwrap();

    let hidden_dim = gate_info.dims[0] as usize; // 2048
    let inter_dim = gate_info.dims[1] as usize; // 8192
    let eps = 1e-5f32;

    println!(
        "FFN: hidden_dim={hidden_dim}, intermediate_dim={inter_dim}, eps={eps}"
    );

    let gate_data = gguf.tensor_data("blk.0.ffn_gate.weight").unwrap();
    let up_data = gguf.tensor_data("blk.0.ffn_up.weight").unwrap();
    let down_data = gguf.tensor_data("blk.0.ffn_down.weight").unwrap();

    // Deterministic input
    let hidden: Vec<f32> = (0..hidden_dim)
        .map(|i| (i as f32 * 0.001).sin() * 0.01)
        .collect();

    // --- CPU FFN chain ---
    let n_iter = 50;
    let mut cpu_hidden = hidden.clone();
    // Warmup
    {
        let mut norm_buf = vec![0.0f32; hidden_dim];
        let mut gate_buf = vec![0.0f32; inter_dim];
        let mut up_buf = vec![0.0f32; inter_dim];
        let mut down_buf = vec![0.0f32; hidden_dim];

        rms_norm(&cpu_hidden, &norm_data, eps, &mut norm_buf);
        quantized_matvec(&norm_buf, gate_data, GgmlType::Q4_K, inter_dim, hidden_dim, &mut gate_buf);
        quantized_matvec(&norm_buf, up_data, GgmlType::Q4_K, inter_dim, hidden_dim, &mut up_buf);
        for i in 0..inter_dim {
            gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
        }
        quantized_matvec(&gate_buf, down_data, GgmlType::Q4_K, hidden_dim, inter_dim, &mut down_buf);
        for i in 0..hidden_dim {
            cpu_hidden[i] += down_buf[i];
        }
    }

    cpu_hidden = hidden.clone();
    let t0 = Instant::now();
    for _ in 0..n_iter {
        let mut norm_buf = vec![0.0f32; hidden_dim];
        let mut gate_buf = vec![0.0f32; inter_dim];
        let mut up_buf = vec![0.0f32; inter_dim];
        let mut down_buf = vec![0.0f32; hidden_dim];

        rms_norm(&cpu_hidden, &norm_data, eps, &mut norm_buf);
        quantized_matvec(&norm_buf, gate_data, GgmlType::Q4_K, inter_dim, hidden_dim, &mut gate_buf);
        quantized_matvec(&norm_buf, up_data, GgmlType::Q4_K, inter_dim, hidden_dim, &mut up_buf);
        for i in 0..inter_dim {
            gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
        }
        quantized_matvec(&gate_buf, down_data, GgmlType::Q4_K, hidden_dim, inter_dim, &mut down_buf);
        for i in 0..hidden_dim {
            cpu_hidden[i] += down_buf[i];
        }
        // Reset for next iteration
        cpu_hidden = hidden.clone();
    }
    // Run one final time to get the result
    {
        let mut norm_buf = vec![0.0f32; hidden_dim];
        let mut gate_buf = vec![0.0f32; inter_dim];
        let mut up_buf = vec![0.0f32; inter_dim];
        let mut down_buf = vec![0.0f32; hidden_dim];
        rms_norm(&cpu_hidden, &norm_data, eps, &mut norm_buf);
        quantized_matvec(&norm_buf, gate_data, GgmlType::Q4_K, inter_dim, hidden_dim, &mut gate_buf);
        quantized_matvec(&norm_buf, up_data, GgmlType::Q4_K, inter_dim, hidden_dim, &mut up_buf);
        for i in 0..inter_dim {
            gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
        }
        quantized_matvec(&gate_buf, down_data, GgmlType::Q4_K, hidden_dim, inter_dim, &mut down_buf);
        for i in 0..hidden_dim {
            cpu_hidden[i] += down_buf[i];
        }
    }
    let cpu_us = t0.elapsed().as_micros() as f64 / n_iter as f64;
    println!("\nCPU FFN chain: {cpu_us:.0} µs/iter ({n_iter} iters)");

    // --- GPU FFN chain ---
    #[cfg(feature = "gpu")]
    {
        println!("\nInitializing GPU...");
        let gpu = GpuEngine::new();

        // Upload weights
        let g_gate = gpu.upload_weights(gate_data, inter_dim, hidden_dim);
        let g_up = gpu.upload_weights(up_data, inter_dim, hidden_dim);
        let g_down = gpu.upload_weights(down_data, hidden_dim, inter_dim);
        let g_norm_w = gpu.upload_f32(&norm_data);

        // Allocate intermediate buffers (persistent on GPU)
        let g_hidden = gpu.upload_f32(&hidden);
        let g_norm_buf = gpu.alloc_f32(hidden_dim);
        let g_gate_buf = gpu.alloc_f32(inter_dim);
        let g_up_buf = gpu.alloc_f32(inter_dim);
        let g_down_buf = gpu.alloc_f32(hidden_dim);

        // Warmup
        for _ in 0..3 {
            gpu.write_f32(&g_hidden, &hidden);
            let mut pass = gpu.begin_pass();
            pass.rmsnorm(&g_hidden, &g_norm_w, &g_norm_buf, eps);
            pass.matvec_q4k(&g_gate, &g_norm_buf, &g_gate_buf);
            pass.matvec_q4k(&g_up, &g_norm_buf, &g_up_buf);
            pass.silu_mul(&g_gate_buf, &g_up_buf);
            pass.matvec_q4k(&g_down, &g_gate_buf, &g_down_buf);
            pass.residual_add(&g_hidden, &g_down_buf);
            pass.execute();
        }

        // Benchmark
        let t0 = Instant::now();
        for _ in 0..n_iter {
            gpu.write_f32(&g_hidden, &hidden);
            let mut pass = gpu.begin_pass();
            pass.rmsnorm(&g_hidden, &g_norm_w, &g_norm_buf, eps);
            pass.matvec_q4k(&g_gate, &g_norm_buf, &g_gate_buf);
            pass.matvec_q4k(&g_up, &g_norm_buf, &g_up_buf);
            pass.silu_mul(&g_gate_buf, &g_up_buf);
            pass.matvec_q4k(&g_down, &g_gate_buf, &g_down_buf);
            pass.residual_add(&g_hidden, &g_down_buf);
            pass.execute();
        }
        let gpu_us = t0.elapsed().as_micros() as f64 / n_iter as f64;

        // Read final result
        gpu.write_f32(&g_hidden, &hidden);
        let mut pass = gpu.begin_pass();
        pass.rmsnorm(&g_hidden, &g_norm_w, &g_norm_buf, eps);
        pass.matvec_q4k(&g_gate, &g_norm_buf, &g_gate_buf);
        pass.matvec_q4k(&g_up, &g_norm_buf, &g_up_buf);
        pass.silu_mul(&g_gate_buf, &g_up_buf);
        pass.matvec_q4k(&g_down, &g_gate_buf, &g_down_buf);
        pass.residual_add(&g_hidden, &g_down_buf);
        let gpu_hidden = pass.execute_and_read(&g_hidden);

        println!("GPU FFN chain: {gpu_us:.0} µs/iter ({n_iter} iters)");
        println!("Speedup: {:.2}x", cpu_us / gpu_us);

        // Correctness (relative error for large values)
        println!("\nCorrectness (first 8):");
        let mut max_rel: f32 = 0.0;
        let mut nan_match = 0u32;
        let mut total_checked = 0u32;
        for i in 0..hidden_dim.min(8) {
            let c = cpu_hidden[i];
            let g = gpu_hidden[i];
            if c.is_nan() && g.is_nan() {
                println!("  [{i:4}] cpu=NaN  gpu=NaN  (both NaN — OK)");
                continue;
            }
            let rel = if c.abs() > 1e-6 {
                (c - g).abs() / c.abs()
            } else {
                (c - g).abs()
            };
            println!("  [{i:4}] cpu={c:+.4}  gpu={g:+.4}  rel={rel:.4}");
        }
        for i in 0..hidden_dim {
            let c = cpu_hidden[i];
            let g = gpu_hidden[i];
            if c.is_nan() && g.is_nan() {
                nan_match += 1;
                continue;
            }
            if c.is_nan() != g.is_nan() {
                println!("  MISMATCH at [{i}]: cpu_nan={} gpu_nan={}", c.is_nan(), g.is_nan());
            }
            total_checked += 1;
            if c.abs() > 1e-6 {
                max_rel = max_rel.max((c - g).abs() / c.abs());
            }
        }
        println!(
            "Max relative error: {max_rel:.4} ({total_checked} values, {nan_match} NaN matches) — {}",
            if max_rel < 0.05 { "PASS" } else { "WARN" }
        );
    }

    #[cfg(not(feature = "gpu"))]
    println!("\nGPU feature not enabled. Run with: cargo run --features gpu,gguf");
}
