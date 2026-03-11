//! Verify fused Q4_K matvec against unfused dequantize + matmul.

use alice_llm::gguf::{self, GgufFile};
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_path = args.get(1).expect("Usage: matvec_verify <model.gguf>");
    let data = fs::read(model_path).expect("Failed to read");
    let gguf = GgufFile::parse(&data).expect("Failed to parse");

    // Test with blk.0.attn_q.weight (Q4_K, 4096x4096)
    let name = "blk.0.attn_q.weight";
    let info = gguf.tensor_info(name).expect("tensor not found");
    let tensor_data = gguf.tensor_data(name).expect("tensor data");
    let rows = info.dims[1] as usize;  // ne[1] = rows
    let cols = info.dims[0] as usize;  // ne[0] = cols
    println!("Testing {name}: rows={rows}, cols={cols}, qtype={:?}", info.qtype);

    // Create test input
    let mut input = vec![0.0f32; cols];
    for i in 0..cols {
        input[i] = ((i as f32) * 0.001).sin();  // non-trivial input
    }

    // Method 1: Fused matvec
    let mut output_fused = vec![0.0f32; rows];
    gguf::q4k_matvec(&input, tensor_data, rows, cols, &mut output_fused);

    // Method 2: Dequantize then matmul
    println!("Dequantizing {}x{} ({:.1} MB)...", rows, cols, (rows * cols * 4) as f64 / 1e6);
    let weights = gguf.tensor_to_f32(name).expect("dequantize");
    let mut output_unfused = vec![0.0f32; rows];
    for r in 0..rows {
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += weights[r * cols + c] * input[c];
        }
        output_unfused[r] = acc;
    }

    // Compare
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut max_diff_idx = 0;
    for i in 0..rows {
        let diff = (output_fused[i] - output_unfused[i]).abs();
        sum_diff += diff as f64;
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }
    let avg_diff = sum_diff / rows as f64;

    println!("\n=== Q4_K Matvec Comparison ===");
    println!("  Max diff: {max_diff:.6e} at index {max_diff_idx}");
    println!("  Avg diff: {avg_diff:.6e}");
    println!("  Fused[0..5]: {:?}", &output_fused[0..5]);
    println!("  Unfused[0..5]: {:?}", &output_unfused[0..5]);

    if max_diff < 1e-4 {
        println!("  ✓ Q4_K fused matvec matches unfused (max diff < 1e-4)");
    } else {
        println!("  ✗ MISMATCH! Max diff = {max_diff:.6e}");
    }

    // Also test Q6_K (blk.0.attn_v.weight)
    let name6 = "blk.0.attn_v.weight";
    let info6 = gguf.tensor_info(name6).expect("tensor not found");
    let data6 = gguf.tensor_data(name6).expect("tensor data");
    let rows6 = info6.dims[1] as usize;
    let cols6 = info6.dims[0] as usize;
    println!("\nTesting {name6}: rows={rows6}, cols={cols6}, qtype={:?}", info6.qtype);

    let input6 = &input[..cols6];
    let mut out_fused6 = vec![0.0f32; rows6];
    gguf::q6k_matvec(input6, data6, rows6, cols6, &mut out_fused6);

    let weights6 = gguf.tensor_to_f32(name6).expect("dequantize");
    let mut out_unfused6 = vec![0.0f32; rows6];
    for r in 0..rows6 {
        let mut acc = 0.0f32;
        for c in 0..cols6 {
            acc += weights6[r * cols6 + c] * input6[c];
        }
        out_unfused6[r] = acc;
    }

    let mut max_diff6 = 0.0f32;
    for i in 0..rows6 {
        let diff = (out_fused6[i] - out_unfused6[i]).abs();
        if diff > max_diff6 { max_diff6 = diff; }
    }

    println!("\n=== Q6_K Matvec Comparison ===");
    println!("  Max diff: {max_diff6:.6e}");
    println!("  Fused[0..5]: {:?}", &out_fused6[0..5]);
    println!("  Unfused[0..5]: {:?}", &out_unfused6[0..5]);
    if max_diff6 < 1e-4 {
        println!("  ✓ Q6_K fused matvec matches unfused");
    } else {
        println!("  ✗ MISMATCH! Max diff = {max_diff6:.6e}");
    }

    // Check: is the unfused matmul using correct row-major order?
    // Verify by checking if output_unfused makes sense
    println!("\n=== Weight matrix sanity ===");
    println!("  weights[0..5] (row 0, first 5 cols): {:?}", &weights[0..5]);
    let row_sum: f32 = weights[0..cols].iter().sum();
    println!("  Row 0 sum: {row_sum:.6}");
    println!("  Row 0 L2 norm: {:.6}", weights[0..cols].iter().map(|x| x*x).sum::<f32>().sqrt());
}
