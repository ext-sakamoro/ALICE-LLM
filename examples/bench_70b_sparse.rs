//! Simulated 70B Sparse Ternary Benchmark.
//!
//! Generates one layer's worth of SparseTernaryMatrix weights (random),
//! runs matvec in a loop, and extrapolates to 80-layer 70B throughput.
//!
//! Usage:
//!   cargo run --release --example bench_70b_sparse --features "gguf,parallel"

use alice_llm::gguf::{
    SparseTernaryMatrix, SparseTernaryRow, SPARSE_BLOCK,
    sparse_ternary_matvec,
};
use std::time::Instant;

/// Llama-3 70B config
const HIDDEN_DIM: usize = 8192;
const NUM_HEADS: usize = 64;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const INTERMEDIATE_DIM: usize = 28672;
const NUM_LAYERS: usize = 80;
const VOCAB_SIZE: usize = 128_256;

/// Structured sparsity: keep N out of 16 per block
const N_KEEP: usize = 8; // 8:16 = 50% sparsity

fn make_sparse_matrix(rows: usize, cols: usize, n_keep: usize) -> SparseTernaryMatrix {
    let num_blocks = (cols + SPARSE_BLOCK - 1) / SPARSE_BLOCK;
    let mut mat_rows = Vec::with_capacity(rows);

    for r in 0..rows {
        let mut active_masks = vec![0u16; num_blocks];
        let mut sign_masks = vec![0u16; num_blocks];

        for blk in 0..num_blocks {
            let mut mask = 0u16;
            let block_end = ((blk + 1) * SPARSE_BLOCK).min(cols) - blk * SPARSE_BLOCK;
            let keep = n_keep.min(block_end);

            let seed = r.wrapping_mul(17) ^ blk.wrapping_mul(31);
            for i in 0..block_end {
                if ((seed.wrapping_mul(i + 1).wrapping_add(7)) % block_end) < keep {
                    mask |= 1u16 << i;
                }
            }
            while mask.count_ones() as usize > keep {
                let bit = mask.trailing_zeros();
                mask &= !(1u16 << bit);
            }
            while (mask.count_ones() as usize) < keep && block_end > 0 {
                for i in 0..block_end {
                    if mask & (1u16 << i) == 0 {
                        mask |= 1u16 << i;
                        break;
                    }
                }
            }

            active_masks[blk] = mask;
            sign_masks[blk] = mask & 0xAAAA;
        }

        mat_rows.push(SparseTernaryRow {
            active_masks,
            sign_masks,
            scale: 0.02,
            num_cols: cols,
            num_blocks,
        });
    }

    SparseTernaryMatrix::from_rows(mat_rows, rows, cols, 1.0 - (n_keep as f32 / SPARSE_BLOCK as f32))
}

fn bench_matvec(name: &str, matrix: &SparseTernaryMatrix, input: &[f32], iters: usize) -> f64 {
    let mut output = vec![0.0f32; matrix.num_rows];

    // Warmup
    for _ in 0..3 {
        sparse_ternary_matvec(matrix, input, &mut output);
    }

    let start = Instant::now();
    for _ in 0..iters {
        sparse_ternary_matvec(matrix, input, &mut output);
    }
    let elapsed = start.elapsed();
    let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iters as f64;

    let nnz = matrix.total_nnz();
    let mem_bytes = matrix.memory_bytes();
    println!(
        "  {name}: {:.3} ms/iter ({rows}×{cols}, nnz={nnz}, mem={mem_kb:.1} KB)",
        ms_per_iter,
        rows = matrix.num_rows,
        cols = matrix.num_cols,
        mem_kb = mem_bytes as f64 / 1024.0,
    );

    ms_per_iter
}

fn main() {
    println!("=== 70B Sparse Ternary Benchmark (Simulated) ===");
    println!();
    println!("Config: Llama-3 70B architecture");
    println!("  Hidden: {HIDDEN_DIM}, Heads: {NUM_HEADS}, KV Heads: {NUM_KV_HEADS}");
    println!("  FFN intermediate: {INTERMEDIATE_DIM}");
    println!("  Layers: {NUM_LAYERS}");
    println!("  Sparsity: {N_KEEP}:16 ({:.0}% density)", N_KEEP as f32 / 16.0 * 100.0);
    println!();

    let kv_dim = NUM_KV_HEADS * HEAD_DIM; // 1024

    // Generate layer weights
    println!("Generating sparse ternary weights for 1 layer...");
    let gen_start = Instant::now();

    let q_proj = make_sparse_matrix(HIDDEN_DIM, HIDDEN_DIM, N_KEEP);    // 8192×8192
    let k_proj = make_sparse_matrix(kv_dim, HIDDEN_DIM, N_KEEP);        // 1024×8192
    let v_proj = make_sparse_matrix(kv_dim, HIDDEN_DIM, N_KEEP);        // 1024×8192
    let o_proj = make_sparse_matrix(HIDDEN_DIM, HIDDEN_DIM, N_KEEP);    // 8192×8192
    let gate_proj = make_sparse_matrix(INTERMEDIATE_DIM, HIDDEN_DIM, N_KEEP); // 28672×8192
    let up_proj = make_sparse_matrix(INTERMEDIATE_DIM, HIDDEN_DIM, N_KEEP);   // 28672×8192
    let down_proj = make_sparse_matrix(HIDDEN_DIM, INTERMEDIATE_DIM, N_KEEP); // 8192×28672

    let gen_ms = gen_start.elapsed().as_millis();
    println!("  Generated in {gen_ms}ms");

    // Memory per layer
    let layer_bytes = q_proj.memory_bytes()
        + k_proj.memory_bytes()
        + v_proj.memory_bytes()
        + o_proj.memory_bytes()
        + gate_proj.memory_bytes()
        + up_proj.memory_bytes()
        + down_proj.memory_bytes();

    let layer_mb = layer_bytes as f64 / (1024.0 * 1024.0);
    let model_gb = (layer_bytes as f64 * NUM_LAYERS as f64
        + VOCAB_SIZE as f64 * HIDDEN_DIM as f64 * 4.0 * 2.0) // embedding + output (f32)
        / (1024.0 * 1024.0 * 1024.0);

    println!();
    println!("=== Memory ===");
    println!("  1 layer (sparse ternary): {layer_mb:.1} MB");
    println!("  {NUM_LAYERS} layers: {:.1} GB", layer_mb * NUM_LAYERS as f64 / 1024.0);
    println!("  Full model estimate (+ embedding): {model_gb:.1} GB");

    // Benchmark
    println!();
    println!("=== Matvec Benchmark (1 layer) ===");

    let hidden_input: Vec<f32> = (0..HIDDEN_DIM).map(|i| (i as f32 * 0.001).sin()).collect();
    let ffn_input: Vec<f32> = (0..INTERMEDIATE_DIM).map(|i| (i as f32 * 0.001).cos()).collect();

    let iters = 5; // Few iters since each is expensive

    let ms_q = bench_matvec("Q proj ", &q_proj, &hidden_input, iters);
    let ms_k = bench_matvec("K proj ", &k_proj, &hidden_input, iters);
    let ms_v = bench_matvec("V proj ", &v_proj, &hidden_input, iters);
    let ms_o = bench_matvec("O proj ", &o_proj, &hidden_input, iters);
    let ms_gate = bench_matvec("Gate   ", &gate_proj, &hidden_input, iters);
    let ms_up = bench_matvec("Up     ", &up_proj, &hidden_input, iters);
    let ms_down = bench_matvec("Down   ", &down_proj, &ffn_input, iters);

    let layer_ms = ms_q + ms_k + ms_v + ms_o + ms_gate + ms_up + ms_down;
    let token_ms = layer_ms * NUM_LAYERS as f64;
    let tok_per_sec = 1000.0 / token_ms;

    println!();
    println!("=== Throughput ===");
    println!("  1 layer forward:  {layer_ms:.1} ms");
    println!("  1 token ({NUM_LAYERS} layers): {token_ms:.0} ms → {tok_per_sec:.2} tok/s");
    println!();

    // Bandwidth analysis
    // Each matvec reads: active weights (masks) + input vector
    // Sparse ternary: 4 bytes per block (active_mask + sign_mask) + scale per row
    let bytes_per_token = layer_bytes as f64 * NUM_LAYERS as f64
        + HIDDEN_DIM as f64 * 4.0 * NUM_LAYERS as f64 * 7.0; // input vectors
    let bandwidth_gb_s = bytes_per_token / (token_ms / 1000.0) / 1e9;

    println!("=== Bandwidth Analysis ===");
    println!("  Data read per token: {:.1} GB", bytes_per_token / 1e9);
    println!("  Effective bandwidth: {bandwidth_gb_s:.1} GB/s");
    println!("  M1 Pro theoretical:  200 GB/s");
    println!("  Utilization:         {:.0}%", bandwidth_gb_s / 200.0 * 100.0);
    println!();

    // Comparison table
    println!("=== Device Projections (bandwidth-limited) ===");
    let data_per_token_gb = bytes_per_token / 1e9;
    let devices = [
        ("Raspberry Pi 5 (LPDDR4X)", 34.1),
        ("Mac Mini M4 (LPDDR5)", 120.0),
        ("M1 Pro (measured)", bandwidth_gb_s.min(200.0)),
        ("Mac Mini M4 Pro", 273.0),
        ("Mac Studio M4 Ultra", 800.0),
    ];
    println!("  {:30} {:>10} {:>10}", "Device", "Bandwidth", "Est. tok/s");
    println!("  {:30} {:>10} {:>10}", "------", "---------", "----------");
    for (name, bw) in &devices {
        let est_tps = bw / data_per_token_gb;
        println!("  {:30} {:>7.0} GB/s {:>7.1}", name, bw, est_tps);
    }

    // Speculative decoding projections
    println!();
    println!("=== Speculative Decoding Projections ===");
    println!("  (Layer-skip draft: first N layers, verify: all {NUM_LAYERS} layers)");
    println!();

    // Simulate different draft configurations
    // Draft cost = (draft_layers / NUM_LAYERS) * token_ms per draft token
    // Verify cost = token_ms per verification step (all layers)
    // With acceptance rate α: effective tokens per cycle = 1 + α*k
    // Cycle cost = token_ms (verify the accepted token) + k * draft_ms + token_ms (verify k drafts)
    // But accepted drafts skip verify → amortized:
    //   Per cycle: 1 verify + k drafts + verify-rejections
    //   Expected accepted = k*α, total output = 1 + k*α
    //   Cost = token_ms + k * draft_ms + token_ms (for verification)
    //   → effective_tok_s = (1 + k*α) / (token_ms + k*draft_ms + token_ms) * 1000

    let configs = [
        (4, 8, 0.6),   // spec_k=4, draft_layers=8 (10%), acceptance=60%
        (4, 16, 0.5),  // spec_k=4, draft_layers=16 (20%), acceptance=50%
        (3, 8, 0.7),   // spec_k=3, draft_layers=8 (10%), acceptance=70%
        (4, 8, 0.8),   // spec_k=4, draft_layers=8 (10%), acceptance=80% (optimistic)
    ];

    println!("  {:>6} {:>12} {:>10} {:>12} {:>10}", "spec_k", "draft_layers", "accept_α", "eff. tok/s", "speedup");
    println!("  {:>6} {:>12} {:>10} {:>12} {:>10}", "------", "------------", "--------", "----------", "-------");
    for (k, draft_layers, alpha) in &configs {
        let draft_ms = token_ms * (*draft_layers as f64) / (NUM_LAYERS as f64);
        // Per speculation cycle:
        // 1. Process current token (verify, full forward): token_ms
        // 2. Draft k tokens: k * draft_ms
        // 3. Verify: re-run full forward for accepted token, then next reject → ~(1 + α*k) verifies
        //    Simplified: cost = token_ms (initial) + k * draft_ms + token_ms (for verification batch)
        // Output tokens = 1 + k * α (on average)
        let cycle_cost_ms = token_ms + (*k as f64) * draft_ms + token_ms;
        let output_tokens = 1.0 + (*k as f64) * alpha;
        let eff_tps = output_tokens / cycle_cost_ms * 1000.0;
        let speedup = eff_tps / tok_per_sec;
        println!("  {:>6} {:>12} {:>10.0}% {:>10.2} {:>9.1}x", k, draft_layers, alpha * 100.0, eff_tps, speedup);
    }

    println!();
    println!("  Baseline (no speculation): {tok_per_sec:.2} tok/s");
}
