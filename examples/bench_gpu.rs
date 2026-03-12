use alice_llm::gguf::{GgufFile, GgmlType};
use std::fs;
use std::time::Instant;

#[cfg(feature = "gpu")]
use alice_llm::gpu::GpuEngine;

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/Users/ys/models/llama-3.2-1b-gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_string());

    println!("Loading model: {model_path}");
    let data = fs::read(&model_path).unwrap();
    let gguf = GgufFile::parse(&data).unwrap();

    // Find a Q4_K tensor for benchmarking
    let tensor_name = "blk.0.ffn_gate.weight";
    let info = gguf
        .tensor_info(tensor_name)
        .unwrap_or_else(|| panic!("tensor {tensor_name} not found"));

    assert!(
        info.qtype == GgmlType::Q4_K,
        "expected Q4_K tensor, got {:?}",
        info.qtype
    );

    let rows = info.dims[1] as usize;
    let cols = info.dims[0] as usize;
    let tensor_data = gguf.tensor_data(tensor_name).unwrap();

    println!(
        "Tensor: {tensor_name} ({rows}×{cols}, {:?}, {} KB)",
        info.qtype,
        tensor_data.len() / 1024
    );

    let input: Vec<f32> = (0..cols)
        .map(|i| (i as f32 * 0.0001).sin() * 0.1)
        .collect();

    // --- CPU benchmark ---
    let mut cpu_output = vec![0.0f32; rows];
    alice_llm::gguf::quantized_matvec(&input, tensor_data, GgmlType::Q4_K, rows, cols, &mut cpu_output);

    let n_iter = 100;
    let t0 = Instant::now();
    for _ in 0..n_iter {
        alice_llm::gguf::quantized_matvec(&input, tensor_data, GgmlType::Q4_K, rows, cols, &mut cpu_output);
    }
    let cpu_us = t0.elapsed().as_micros() as f64 / n_iter as f64;
    println!("\nCPU matvec: {cpu_us:.0} µs/iter ({n_iter} iters)");

    #[cfg(feature = "gpu")]
    {
        println!("\nInitializing GPU...");
        let gpu = GpuEngine::new();

        let gpu_weights = gpu.upload_weights(tensor_data, rows, cols);
        let ctx = gpu.create_matvec_context(&gpu_weights);

        // Warmup
        for _ in 0..5 {
            let _ = gpu.matvec_fast(&ctx, &input);
        }

        // --- GPU with readback ---
        let t0 = Instant::now();
        let mut gpu_output = vec![0.0f32; rows];
        for _ in 0..n_iter {
            gpu_output = gpu.matvec_fast(&ctx, &input);
        }
        let gpu_readback_us = t0.elapsed().as_micros() as f64 / n_iter as f64;

        // --- GPU dispatch only (no readback) ---
        // Submit N dispatches in one command buffer, measure total time
        let t0 = Instant::now();
        for _ in 0..n_iter {
            gpu.dispatch_matvec_only(&ctx, &input);
        }
        // Single sync at end
        gpu.sync();
        let gpu_dispatch_us = t0.elapsed().as_micros() as f64 / n_iter as f64;

        // --- GPU kernel only (batched dispatches, single command buffer) ---
        // Warmup
        gpu.bench_dispatch_batch(&ctx, &input, 10);
        let batch_n = 200u32;
        let t0 = Instant::now();
        gpu.bench_dispatch_batch(&ctx, &input, batch_n);
        let gpu_kernel_us = t0.elapsed().as_micros() as f64 / batch_n as f64;

        let readback_overhead = gpu_readback_us - gpu_dispatch_us;

        println!("\nGPU kernel (batch):  {gpu_kernel_us:.0} µs/iter  ({:.2}x vs CPU)", cpu_us / gpu_kernel_us);
        println!("GPU dispatch+submit: {gpu_dispatch_us:.0} µs/iter  ({:.2}x vs CPU)", cpu_us / gpu_dispatch_us);
        println!("GPU with readback:   {gpu_readback_us:.0} µs/iter  ({:.2}x vs CPU)", cpu_us / gpu_readback_us);
        println!("Readback overhead:   {readback_overhead:.0} µs/iter");
        println!("\n=> Full-pipeline GPU (batch) would run at {:.2}x CPU speed", cpu_us / gpu_kernel_us);

        // --- Correctness ---
        println!("\nCorrectness (first 5):");
        let mut max_diff: f32 = 0.0;
        for i in 0..rows.min(5) {
            let diff = (cpu_output[i] - gpu_output[i]).abs();
            max_diff = max_diff.max(diff);
            println!(
                "  [{i:4}] cpu={:+.6}  gpu={:+.6}  diff={:.6}",
                cpu_output[i], gpu_output[i], diff
            );
        }
        for i in 0..rows {
            max_diff = max_diff.max((cpu_output[i] - gpu_output[i]).abs());
        }
        println!("Max diff: {max_diff:.6} — {}", if max_diff < 0.1 { "PASS" } else { "FAIL" });
    }

    #[cfg(not(feature = "gpu"))]
    println!("\nGPU feature not enabled. Run with: cargo run --features gpu,gguf");
}
