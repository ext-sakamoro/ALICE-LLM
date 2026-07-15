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

        // ---- GPU batch4 (Step 2) ----
        // Layout: input buffer packs 4 input vectors of length `cols`
        // back-to-back; output buffer packs 4 output vectors of length `rows`
        // back-to-back. Same weight tensor, one dispatch produces all 4
        // outputs, so the weight decode work is shared across the batch.
        println!("\n--- Step 2: batch-4 kernel ---");
        let mut input_batch4: Vec<f32> = Vec::with_capacity(cols * 4);
        for k in 0..4 {
            let shift = (k as f32) * 0.13;
            input_batch4.extend((0..cols).map(|i| ((i as f32) * 0.017 + shift).sin() * 0.1));
        }
        let input_buf_b4 = engine.upload_f32(&input_batch4);
        let output_buf_b4 = engine.alloc_f32(rows * 4);

        // Warmup
        for _ in 0..5 {
            let mut pass = engine.begin_pass();
            pass.matvec_q1_0_batch4(&weights, &input_buf_b4, &output_buf_b4);
            pass.execute();
        }

        let t0 = Instant::now();
        for _ in 0..n_iter {
            let mut pass = engine.begin_pass();
            pass.matvec_q1_0_batch4(&weights, &input_buf_b4, &output_buf_b4);
            pass.execute();
        }
        engine.sync();
        let gpu_b4_us = t0.elapsed().as_micros() as f64 / n_iter as f64;
        let gpu_b4_per_matvec = gpu_b4_us / 4.0;

        println!(
            "GPU batch4:  {:>10.1} µs/dispatch (4 matvecs / dispatch), {:.1} µs/matvec",
            gpu_b4_us, gpu_b4_per_matvec
        );
        println!(
            "Speedup batch4/single: {:.2}× per matvec (dispatch overhead 分散)",
            gpu_us / gpu_b4_per_matvec
        );
        println!(
            "Speedup CPU / GPU-batch4: {:.2}× per matvec",
            cpu_us / gpu_b4_per_matvec
        );

        // Parity check on batch element 0 (should match single-dispatch output)
        let gpu_b4_output = engine.read_f32(&output_buf_b4);
        // The batch4 shader stores output as [rows] × 4 batches:
        //   output_vec[batch * rows + row]
        // So batch 0's output is `gpu_b4_output[0..rows]`.
        let batch0 = &gpu_b4_output[..rows];
        let mut l2_num_b4 = 0.0f64;
        let mut l2_den_b4 = 0.0f64;
        let mut max_abs_err_b4 = 0.0f32;
        for (c, g) in cpu_output.iter().zip(batch0.iter()) {
            let diff = (c - g).abs();
            max_abs_err_b4 = max_abs_err_b4.max(diff);
            l2_num_b4 += (diff as f64) * (diff as f64);
            l2_den_b4 += (*c as f64) * (*c as f64);
        }
        let l2_rel_b4 = (l2_num_b4 / l2_den_b4.max(1e-30)).sqrt();
        println!(
            "Parity (batch4[0] vs CPU): L2_rel = {:.3e}, max_abs_err = {:.3e}",
            l2_rel_b4, max_abs_err_b4
        );

        // ---- Step 3: dispatch batching (single-tensor microbench) ----
        //
        // Hypothesis: at 1013 µs / matvec (batch4) vs the 103 µs
        // memory-bandwidth ceiling, the residual 10× gap is dominated by
        // wgpu / Vulkan submit + fence-wait overhead, not shader execute
        // time. If that holds, submitting N matvec dispatches inside a
        // single ComputePass + single execute() should collapse that
        // overhead across N iterations.
        //
        // We reuse the batch4 pipeline (same bind group / buffers) and
        // record 20 dispatches into one command buffer, then execute + sync
        // once. Per-matvec time = total / (N * 4) since each dispatch still
        // produces 4 batch outputs.
        println!("\n--- Step 3: dispatch batching (1 command buffer, {n_iter} dispatches) ---");

        // Warmup
        {
            let mut pass = engine.begin_pass();
            for _ in 0..5 {
                pass.matvec_q1_0_batch4(&weights, &input_buf_b4, &output_buf_b4);
            }
            pass.execute();
            engine.sync();
        }

        let t0 = Instant::now();
        {
            let mut pass = engine.begin_pass();
            for _ in 0..n_iter {
                pass.matvec_q1_0_batch4(&weights, &input_buf_b4, &output_buf_b4);
            }
            pass.execute();
            engine.sync();
        }
        let total_us = t0.elapsed().as_micros() as f64;
        let batched_per_dispatch = total_us / n_iter as f64;
        let batched_per_matvec = batched_per_dispatch / 4.0;

        println!(
            "GPU batch4×{n_iter} in 1 cmd buffer:  {:>10.1} µs total = {:.1} µs/dispatch = {:.1} µs/matvec",
            total_us, batched_per_dispatch, batched_per_matvec
        );
        println!(
            "Speedup vs Step 2 (per matvec): {:.2}× (submit overhead 分散)",
            gpu_b4_per_matvec / batched_per_matvec
        );
        println!(
            "Speedup CPU / GPU-batched: {:.2}× per matvec",
            cpu_us / batched_per_matvec
        );

        // Bandwidth headroom check
        let bytes_per_matvec = tensor_data.len() as f64;
        let bandwidth_gb_s = bytes_per_matvec / (batched_per_matvec * 1000.0);
        // ~68 GB/s LPDDR5 on Jetson Orin Nano, ~200 GB/s LPDDR5X on M3 Max
        println!(
            "Effective per-matvec bandwidth: {:.2} GB/s (weight bytes / matvec time)",
            bandwidth_gb_s
        );

        // ---- Step 4: row4×batch4 kernel (4 rows × 4 batches / workgroup) ----
        //
        // Compounds Step 2's batch4 win with a 4× reduction in workgroup
        // count (10240 rows → 2560 row-groups on Bonsai attn_qkv) and 4×
        // fewer total input reads (input is now shared across 4 output rows
        // instead of being re-fetched by every row's workgroup).
        //
        // Step 3 showed submit overhead is not the bottleneck (~1× speedup)
        // → 4000 µs / dispatch is dominated by in-shader work. Reducing the
        // shader's per-dispatch memory traffic is the remaining lever.
        println!("\n--- Step 4: row4×batch4 kernel (4 rows × 4 batches / workgroup) ---");

        // Warmup
        for _ in 0..5 {
            let mut pass = engine.begin_pass();
            pass.matvec_q1_0_row4_batch4(&weights, &input_buf_b4, &output_buf_b4);
            pass.execute();
            engine.sync();
        }

        let t0 = Instant::now();
        for _ in 0..n_iter {
            let mut pass = engine.begin_pass();
            pass.matvec_q1_0_row4_batch4(&weights, &input_buf_b4, &output_buf_b4);
            pass.execute();
            engine.sync();
        }
        let gpu_r4b4_per_dispatch = t0.elapsed().as_micros() as f64 / n_iter as f64;
        // Each dispatch produces 4 rows × 4 batches = 16 outputs, but the
        // per-matvec unit for comparison with prior steps is one full
        // (rows × batch=1) matvec → 16 outputs / dispatch is equivalent to
        // 4 matvecs (each single-batch matvec produces `rows` outputs, and
        // each dispatch produces `rows × 4 batches` outputs).
        let gpu_r4b4_per_matvec = gpu_r4b4_per_dispatch / 4.0;

        println!(
            "GPU row4×batch4:  {:.1} µs/dispatch (16 outputs / dispatch), {:.1} µs/matvec",
            gpu_r4b4_per_dispatch, gpu_r4b4_per_matvec
        );
        println!(
            "Speedup vs Step 2 batch4 (per matvec): {:.2}×",
            gpu_b4_per_matvec / gpu_r4b4_per_matvec
        );
        println!(
            "Speedup CPU / GPU-row4×batch4: {:.2}× per matvec",
            cpu_us / gpu_r4b4_per_matvec
        );

        let r4b4_bandwidth_gb_s = bytes_per_matvec / (gpu_r4b4_per_matvec * 1000.0);
        println!(
            "Effective per-matvec bandwidth: {:.2} GB/s",
            r4b4_bandwidth_gb_s
        );

        // Parity: row4×batch4 batch element 0 should match single-dispatch CPU.
        let gpu_r4b4_output = engine.read_f32(&output_buf_b4);
        let r4b4_batch0 = &gpu_r4b4_output[..rows];
        let mut l2_num_r4 = 0.0f64;
        let mut l2_den_r4 = 0.0f64;
        let mut max_abs_err_r4 = 0.0f32;
        for (c, g) in cpu_output.iter().zip(r4b4_batch0.iter()) {
            let diff = (c - g).abs();
            max_abs_err_r4 = max_abs_err_r4.max(diff);
            l2_num_r4 += (diff as f64) * (diff as f64);
            l2_den_r4 += (*c as f64) * (*c as f64);
        }
        let l2_rel_r4 = (l2_num_r4 / l2_den_r4.max(1e-30)).sqrt();
        println!(
            "Parity (row4×batch4[0] vs CPU): L2_rel = {:.3e}, max_abs_err = {:.3e}",
            l2_rel_r4, max_abs_err_r4
        );

        // ---- Step 5: row8×batch4 kernel (8 rows × 4 batches / workgroup) ----
        //
        // Doubles the row batching factor over Step 4: workgroup count
        // drops another 2× (2560 → 1280 on Bonsai attn_qkv), total
        // input-read traffic drops another 2×, and per-thread register
        // pressure grows to 32 f32 accumulators.
        //
        // The main risk is register spilling — if the compiler can't fit
        // all 32 accumulators in physical registers, they land in local
        // memory (VRAM) and every FMA becomes a load/store round trip. We
        // measure to check whether the extra amortization outweighs any
        // spill cost.
        println!("\n--- Step 5: row8×batch4 kernel (8 rows × 4 batches / workgroup) ---");

        for _ in 0..5 {
            let mut pass = engine.begin_pass();
            pass.matvec_q1_0_row8_batch4(&weights, &input_buf_b4, &output_buf_b4);
            pass.execute();
            engine.sync();
        }

        let t0 = Instant::now();
        for _ in 0..n_iter {
            let mut pass = engine.begin_pass();
            pass.matvec_q1_0_row8_batch4(&weights, &input_buf_b4, &output_buf_b4);
            pass.execute();
            engine.sync();
        }
        let gpu_r8b4_per_dispatch = t0.elapsed().as_micros() as f64 / n_iter as f64;
        // Each dispatch produces 8 rows × 4 batches = 32 outputs; comparing
        // "per matvec" (rows × batch=1 = rows outputs), one dispatch is
        // worth 4 matvecs (same as row4×batch4, since we still amortize
        // across 4 batches — the row batching contributes to per-dispatch
        // throughput, not per-matvec cost normalization).
        let gpu_r8b4_per_matvec = gpu_r8b4_per_dispatch / 4.0;

        println!(
            "GPU row8×batch4:  {:.1} µs/dispatch (32 outputs / dispatch), {:.1} µs/matvec",
            gpu_r8b4_per_dispatch, gpu_r8b4_per_matvec
        );
        println!(
            "Speedup vs Step 4 row4×batch4 (per matvec): {:.2}×",
            gpu_r4b4_per_matvec / gpu_r8b4_per_matvec
        );
        println!(
            "Speedup vs Step 2 batch4 (per matvec): {:.2}×",
            gpu_b4_per_matvec / gpu_r8b4_per_matvec
        );
        println!(
            "Speedup CPU / GPU-row8×batch4: {:.2}× per matvec",
            cpu_us / gpu_r8b4_per_matvec
        );

        let r8b4_bandwidth_gb_s = bytes_per_matvec / (gpu_r8b4_per_matvec * 1000.0);
        println!(
            "Effective per-matvec bandwidth: {:.2} GB/s",
            r8b4_bandwidth_gb_s
        );

        // Parity: row8×batch4 batch element 0 should match single-dispatch CPU.
        let gpu_r8b4_output = engine.read_f32(&output_buf_b4);
        let r8b4_batch0 = &gpu_r8b4_output[..rows];
        let mut l2_num_r8 = 0.0f64;
        let mut l2_den_r8 = 0.0f64;
        let mut max_abs_err_r8 = 0.0f32;
        for (c, g) in cpu_output.iter().zip(r8b4_batch0.iter()) {
            let diff = (c - g).abs();
            max_abs_err_r8 = max_abs_err_r8.max(diff);
            l2_num_r8 += (diff as f64) * (diff as f64);
            l2_den_r8 += (*c as f64) * (*c as f64);
        }
        let l2_rel_r8 = (l2_num_r8 / l2_den_r8.max(1e-30)).sqrt();
        println!(
            "Parity (row8×batch4[0] vs CPU): L2_rel = {:.3e}, max_abs_err = {:.3e}",
            l2_rel_r8, max_abs_err_r8
        );
    }
}
