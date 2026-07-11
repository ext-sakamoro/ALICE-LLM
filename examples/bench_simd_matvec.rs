//! Quantized matvec micro-benchmark for the SIMD dispatch paths
//! introduced in PRs #21 / #22 / #23 / #26 / #28.
//!
//! Runs each quant type's row-major matvec against a fixed synthetic
//! weight matrix and f32 input, then prints wall-clock timings. The
//! numbers are indicative — the same runner will report the AVX2 or
//! AVX-512BW path on x86_64 (whichever `is_x86_feature_detected!`
//! reports), and the NEON path on aarch64 (compile-time). Scalar
//! numbers are captured by running the same input under a target where
//! neither SIMD path applies (i.e. a wasm build or an x86_64 box with
//! `RUSTFLAGS='-C target-feature=-avx2,-avx512bw'`); this binary
//! prints the currently-dispatched runtime path so the user can
//! label the row accordingly.
//!
//! Real benchmark harness (criterion + comparison table) is tracked as
//! Issue #25. This example is the zero-dependency stopgap so the
//! numbers are available today on Mac aarch64 / Jetson without waiting
//! for an x86_64 machine allocation.
//!
//! Usage:
//! ```
//! cargo run --release --example bench_simd_matvec --features gguf
//! ```

use alice_llm::gguf::{
    q4k_matvec, q5k_matvec, q6k_matvec, q8_0_matvec, ternary_matvec, TernaryMatrix, TernaryRow,
};
use std::time::Instant;

/// Rows × cols of the synthetic weight matrix. Chosen to be a multiple
/// of every quant type's block size (32 for Q8_0, 256 for K-quants).
const ROWS: usize = 128;
const COLS: usize = 4096;
const ITERS: usize = 200;

fn detect_x86_path() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512bw") {
            return "x86_64 AVX-512BW";
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return "x86_64 AVX2";
        }
        "x86_64 scalar (no AVX2)"
    }
    #[cfg(target_arch = "aarch64")]
    {
        "aarch64 NEON"
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "scalar"
    }
}

fn build_q4k_data() -> Vec<u8> {
    // Block layout: 144 bytes per K-quant block, 256 elements per block.
    let blocks_per_row = COLS / 256;
    let mut data = Vec::with_capacity(ROWS * blocks_per_row * 144);
    for r in 0..ROWS {
        for bi in 0..blocks_per_row {
            let seed = (r * blocks_per_row + bi) as u8;
            // f16 d/dmin, 12-byte scales/mins, 128-byte packed nibbles.
            data.extend_from_slice(&0x3800u16.to_le_bytes());
            data.extend_from_slice(&0x3400u16.to_le_bytes());
            for i in 0..12 {
                data.push(seed.wrapping_add(i as u8));
            }
            for i in 0..128 {
                data.push(seed.wrapping_mul(3).wrapping_add(i as u8));
            }
        }
    }
    data
}

fn build_q5k_data() -> Vec<u8> {
    // Block layout: 176 bytes per K-quant block, 256 elements per block.
    let blocks_per_row = COLS / 256;
    let mut data = Vec::with_capacity(ROWS * blocks_per_row * 176);
    for r in 0..ROWS {
        for bi in 0..blocks_per_row {
            let seed = (r * blocks_per_row + bi) as u8;
            data.extend_from_slice(&0x3800u16.to_le_bytes());
            data.extend_from_slice(&0x3400u16.to_le_bytes());
            for i in 0..12 {
                data.push(seed.wrapping_add(i as u8));
            }
            for i in 0..32 {
                data.push(seed.wrapping_mul(7).wrapping_add(i as u8));
            }
            for i in 0..128 {
                data.push(seed.wrapping_mul(3).wrapping_add(i as u8));
            }
        }
    }
    data
}

fn build_q6k_data() -> Vec<u8> {
    // Block layout: 210 bytes per K-quant block, 256 elements per block.
    let blocks_per_row = COLS / 256;
    let mut data = Vec::with_capacity(ROWS * blocks_per_row * 210);
    for r in 0..ROWS {
        for bi in 0..blocks_per_row {
            let seed = (r * blocks_per_row + bi) as u8;
            for i in 0..128 {
                data.push(seed.wrapping_add(i as u8));
            }
            for i in 0..64 {
                data.push(seed.wrapping_mul(5).wrapping_add(i as u8));
            }
            for i in 0..16 {
                data.push((i as u8).wrapping_sub(4));
            }
            data.extend_from_slice(&0x3800u16.to_le_bytes());
        }
    }
    data
}

fn build_q8_0_data() -> Vec<u8> {
    // Block layout: 34 bytes per Q8_0 block, 32 elements per block.
    let blocks_per_row = COLS / 32;
    let mut data = Vec::with_capacity(ROWS * blocks_per_row * 34);
    for r in 0..ROWS {
        for bi in 0..blocks_per_row {
            let seed = (r * blocks_per_row + bi) as u8;
            data.extend_from_slice(&0x3800u16.to_le_bytes());
            for i in 0..32 {
                let raw = seed.wrapping_add(i as u8);
                data.push((raw as i8).wrapping_sub(32) as u8);
            }
        }
    }
    data
}

fn build_ternary_matrix() -> TernaryMatrix {
    let mut rows = Vec::with_capacity(ROWS);
    for r in 0..ROWS {
        let seed = r as u8;
        let num_bytes = COLS.div_ceil(8);
        let mut pos_mask = vec![0u8; num_bytes];
        let mut neg_mask = vec![0u8; num_bytes];
        for i in 0..COLS {
            let raw = seed.wrapping_mul(3).wrapping_add(i as u8) as i8;
            let byte_idx = i / 8;
            let bit = 1u8 << (i % 8);
            if raw > 20 {
                pos_mask[byte_idx] |= bit;
            } else if raw < -20 {
                neg_mask[byte_idx] |= bit;
            }
        }
        rows.push(TernaryRow {
            pos_mask,
            neg_mask,
            scale: 0.125,
            num_cols: COLS,
        });
    }
    TernaryMatrix {
        rows,
        num_rows: ROWS,
        num_cols: COLS,
    }
}

fn build_input() -> Vec<f32> {
    (0..COLS).map(|i| f32::from(i as u8 as i8) * 0.01).collect()
}

/// Measure `iters` invocations of `run` and return the median per-invocation
/// nanosecond count. Median is more robust than mean against runner noise
/// (Ubuntu VMs are especially jittery).
fn measure_ns(iters: usize, mut run: impl FnMut()) -> u128 {
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        run();
        samples.push(t0.elapsed().as_nanos());
    }
    samples.sort_unstable();
    samples[iters / 2]
}

fn main() {
    println!("ALICE-LLM SIMD matvec micro-benchmark (Issue #25)");
    println!("Runtime dispatch path: {}", detect_x86_path());
    println!("Matrix: {ROWS} × {COLS} f32 output = {ROWS} rows");
    println!("Iterations: {ITERS} (median reported)");
    println!();

    let input = build_input();
    let mut output = vec![0.0f32; ROWS];

    // Warm-up + measurement per quant type. The input `input` and output
    // buffer are reused so allocator noise stays out of the numbers.
    let q4k_data = build_q4k_data();
    let ns = measure_ns(ITERS, || {
        q4k_matvec(&input, &q4k_data, ROWS, COLS, &mut output);
    });
    println!(
        "  Q4_K  : {:>8} ns/matvec ({:>6.3} µs)",
        ns,
        ns as f64 / 1_000.0
    );

    let q5k_data = build_q5k_data();
    let ns = measure_ns(ITERS, || {
        q5k_matvec(&input, &q5k_data, ROWS, COLS, &mut output);
    });
    println!(
        "  Q5_K  : {:>8} ns/matvec ({:>6.3} µs)",
        ns,
        ns as f64 / 1_000.0
    );

    let q6k_data = build_q6k_data();
    let ns = measure_ns(ITERS, || {
        q6k_matvec(&input, &q6k_data, ROWS, COLS, &mut output);
    });
    println!(
        "  Q6_K  : {:>8} ns/matvec ({:>6.3} µs)",
        ns,
        ns as f64 / 1_000.0
    );

    let q8_0_data = build_q8_0_data();
    let ns = measure_ns(ITERS, || {
        q8_0_matvec(&input, &q8_0_data, ROWS, COLS, &mut output);
    });
    println!(
        "  Q8_0  : {:>8} ns/matvec ({:>6.3} µs)",
        ns,
        ns as f64 / 1_000.0
    );

    let ternary = build_ternary_matrix();
    let ns = measure_ns(ITERS, || {
        ternary_matvec(&ternary, &input, &mut output);
    });
    println!(
        "  Ternary: {:>7} ns/matvec ({:>6.3} µs)",
        ns,
        ns as f64 / 1_000.0
    );

    println!();
    println!("Notes:");
    println!("  • Run under different builds to compare paths:");
    println!("    • Default (aarch64 → NEON, x86_64 → AVX2/AVX-512 per CPU)");
    println!("    • Scalar fallback: on x86_64, `RUSTFLAGS=\"-C target-feature=-avx2,-avx512bw\"`");
    println!("  • Real benchmark harness with statistical rigor: Issue #25.");
}
