//! Closed-form oracles for the numeric primitives of alice-llm
//!
//! Every assertion compares against a value known independently of the code:
//! an algebraic identity, a hand-computable formula, or a constructive input
//! whose answer is exact No golden values The vector length is swept across
//! the 8-lane SIMD boundary so the remainder paths are covered

use alice_llm::attention::{causal_mask, scaled_dot_product_attention};
use alice_llm::gguf::quantize_row_q8_k;
use alice_llm::linalg::{layer_norm, rms_norm, silu};
use alice_llm::matrix::dot_flat as dot;
use alice_llm::rope::apply_rope;
use alice_llm::sampling::{
    apply_temperature, sample_argmax, sample_with_random, softmax, top_k_filter, top_p_filter,
};

/// Q8_K block length (llama.cpp `QK_K`)
const QK_K: usize = 256;

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32).mul_add(0.37, -2.0)).collect()
}

fn ints(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 7) as f32 - 3.0).collect()
}

#[test]
fn dot_flat_is_exact_on_integer_inputs_for_every_length() {
    for n in 0..=70usize {
        let a = ints(n);
        let b: Vec<f32> = (0..n).map(|i| (i % 5) as f32 - 2.0).collect();
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert_eq!(dot(&a, &b), expected, "dot n={n}");
    }
}

// ----------------------------------------------------------------------------
// Normalisation / activation identities
// ----------------------------------------------------------------------------

#[test]
fn rms_norm_has_unit_rms_and_is_scale_invariant() {
    for n in [1usize, 2, 7, 8, 9, 64, 65] {
        let v = ramp(n);
        let out = rms_norm(&v, 0.0);
        let rms = (out.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
        assert!((rms - 1.0).abs() < 1e-4, "n={n} rms {rms}");
        let scaled: Vec<f32> = v.iter().map(|x| x * 8.0).collect();
        for (a, b) in rms_norm(&scaled, 0.0).iter().zip(&out) {
            assert!((a - b).abs() < 1e-5, "n={n}: {a} vs {b}");
        }
        // eps ≪ mean square must not move the result
        for eps in [1e-12f32, 1e-8, 1e-6] {
            for (a, b) in rms_norm(&v, eps).iter().zip(&out) {
                assert!((a - b).abs() < 1e-4, "n={n} eps={eps}: {a} vs {b}");
            }
        }
    }
}

#[test]
fn layer_norm_output_has_zero_mean_unit_variance_and_is_affine_invariant() {
    for n in [2usize, 3, 8, 9, 33, 64, 65] {
        let v = ramp(n);
        let out = layer_norm(&v, 0.0);
        let mean = out.iter().sum::<f32>() / n as f32;
        let var = out.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 1e-5, "n={n} mean {mean}");
        assert!((var - 1.0).abs() < 1e-4, "n={n} var {var}");
        let affine: Vec<f32> = v.iter().map(|x| 3.0 * x - 7.0).collect();
        for (a, b) in layer_norm(&affine, 0.0).iter().zip(&out) {
            assert!((a - b).abs() < 1e-4, "n={n}: {a} vs {b}");
        }
    }
}

#[test]
fn silu_closed_form_points() {
    assert_eq!(silu(0.0), 0.0);
    // silu(x) = x·σ(x); σ(ln 3) = 3/4
    let x = 3.0f32.ln();
    assert!((silu(x) - 0.75 * x).abs() < 1e-6);
    // odd-symmetric part: silu(x) − silu(−x) = x
    for x in [0.5f32, 1.0, 2.5, 10.0] {
        assert!((silu(x) - silu(-x) - x).abs() < 1e-5, "{x}");
    }
    // saturation
    assert!((silu(20.0) - 20.0).abs() < 1e-4);
    assert!(silu(-20.0).abs() < 1e-6);
}

// ----------------------------------------------------------------------------
// RoPE: a rotation (norm preserved), exact angle on the first pair, relative
// position property q(m)·k(n) depends only on m − n
// ----------------------------------------------------------------------------

#[test]
fn rope_preserves_norm_rotates_first_pair_by_position_and_is_relative() {
    let base = 10_000.0f32;
    for dim in [2usize, 4, 8, 64, 128] {
        let v = ramp(dim);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for pos in [0usize, 1, 7, 100, 4095] {
            let r = apply_rope(&v, pos, base);
            let rn = r.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((rn - norm).abs() < 1e-3 * norm, "dim={dim} pos={pos}");
            // pair (0, half) has frequency 1 → rotated by exactly `pos` radians
            let (c, s) = ((pos as f32).cos(), (pos as f32).sin());
            let half = dim / 2;
            assert!(
                (r[0] - (v[0] * c - v[half] * s)).abs() < 1e-3,
                "dim={dim} pos={pos}"
            );
            assert!(
                (r[half] - (v[0] * s + v[half] * c)).abs() < 1e-3,
                "dim={dim} pos={pos}"
            );
        }
        // position 0 is the identity
        assert_eq!(apply_rope(&v, 0, base), v);
    }
    // relative property: <rope(q, m), rope(k, n)> = <rope(q, m + d), rope(k, n + d)>
    let dim = 16;
    let q = ramp(dim);
    let k: Vec<f32> = (0..dim).map(|i| ((i * 3) % 5) as f32 - 1.5).collect();
    let inner =
        |m: usize, n: usize| dot(&apply_rope(&q, m, 10_000.0), &apply_rope(&k, n, 10_000.0));
    for (m, n) in [(3usize, 1usize), (10, 4), (20, 20)] {
        let a = inner(m, n);
        for d in [1usize, 5, 50] {
            let b = inner(m + d, n + d);
            assert!(
                (a - b).abs() < 1e-3 * a.abs().max(1.0),
                "({m},{n})+{d}: {a} vs {b}"
            );
        }
    }
}

// ----------------------------------------------------------------------------
// Sampling laws
// ----------------------------------------------------------------------------

#[test]
fn softmax_sums_to_one_is_shift_invariant_and_matches_two_element_formula() {
    for n in [1usize, 2, 3, 8, 9, 65] {
        let l = ramp(n);
        let p = softmax(&l);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5, "n={n}");
        let shifted: Vec<f32> = l.iter().map(|x| x + 9.5).collect();
        for (a, b) in softmax(&shifted).iter().zip(&p) {
            assert!((a - b).abs() < 1e-6, "n={n}: {a} vs {b}");
        }
        assert_eq!(sample_argmax(&l), n - 1);
    }
    for d in [-3.0f32, -0.5, 0.0, 0.5, 3.0] {
        let p = softmax(&[0.0, d]);
        let p1 = d.exp() / (1.0 + d.exp());
        assert!(
            (p[1] - p1).abs() < 1e-6 && (p[0] - (1.0 - p1)).abs() < 1e-6,
            "d={d}: {p:?}"
        );
    }
}

#[test]
fn temperature_scales_logits_and_the_limits_are_argmax_and_uniform() {
    let l = [1.0f32, 2.0, 4.0, 0.5];
    let mut t = l;
    apply_temperature(&mut t, 0.5);
    assert_eq!(t, [2.0, 4.0, 8.0, 1.0]);
    // T → 0: probability mass collapses on the argmax
    let mut cold = l;
    apply_temperature(&mut cold, 1e-3);
    let p = softmax(&cold);
    assert!(p[2] > 0.999, "{p:?}");
    // T → ∞: uniform
    let mut hot = l;
    apply_temperature(&mut hot, 1e6);
    for q in softmax(&hot) {
        assert!((q - 0.25).abs() < 1e-4, "{q}");
    }
    // temperature 0 is a no-op (documented guard), not a division by zero
    let mut zero = l;
    apply_temperature(&mut zero, 0.0);
    assert_eq!(zero, l);
}

#[test]
fn top_k_keeps_exactly_k_finite_logits_and_top_p_the_smallest_nucleus() {
    let l: Vec<f32> = (0..20).map(|i| ((i * 7) % 20) as f32 * 0.3).collect();
    for k in 1..=20usize {
        let mut f = l.clone();
        top_k_filter(&mut f, k);
        let kept: Vec<usize> = (0..20).filter(|&i| f[i].is_finite()).collect();
        assert_eq!(kept.len(), k, "k={k}");
        // every kept logit ≥ every dropped one
        let min_kept = kept.iter().map(|&i| l[i]).fold(f32::INFINITY, f32::min);
        for i in 0..20 {
            if !f[i].is_finite() {
                assert!(l[i] < min_kept, "k={k} dropped {i}");
            }
        }
    }
    // k = 0 and k ≥ n are documented no-ops
    let mut f = l.clone();
    top_k_filter(&mut f, 0);
    assert_eq!(f, l);
    let mut f = l.clone();
    top_k_filter(&mut f, 20);
    assert_eq!(f, l);

    // top-p: probabilities 0.5, 0.3, 0.2 (logits ln p) → p = 0.6 keeps the first two
    let logits = [0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()];
    let mut f = logits;
    top_p_filter(&mut f, 0.6);
    assert!(
        f[0].is_finite() && f[1].is_finite() && f[2] == f32::NEG_INFINITY,
        "{f:?}"
    );
    let mut f = logits;
    top_p_filter(&mut f, 0.4);
    assert!(
        f[0].is_finite() && f[1] == f32::NEG_INFINITY && f[2] == f32::NEG_INFINITY,
        "{f:?}"
    );
    let mut f = logits;
    top_p_filter(&mut f, 1.0);
    assert_eq!(f, logits);
}

#[test]
fn sample_with_random_inverts_the_cdf() {
    let probs = [0.1f32, 0.2, 0.3, 0.4];
    // cdf = 0.1, 0.3, 0.6, 1.0
    assert_eq!(sample_with_random(&probs, 0.0), 0);
    assert_eq!(sample_with_random(&probs, 0.0999), 0);
    assert_eq!(sample_with_random(&probs, 0.1001), 1);
    assert_eq!(sample_with_random(&probs, 0.2999), 1);
    assert_eq!(sample_with_random(&probs, 0.3001), 2);
    assert_eq!(sample_with_random(&probs, 0.5999), 2);
    assert_eq!(sample_with_random(&probs, 0.6001), 3);
    assert_eq!(sample_with_random(&probs, 0.9999), 3);
    // r ≥ Σp (rounding) → last index, never out of range
    assert_eq!(sample_with_random(&probs, 1.0), 3);
    assert_eq!(sample_with_random(&[], 0.5), 0);
}

// ----------------------------------------------------------------------------
// Attention: one-hot key → the matching value; causal mask shape
// ----------------------------------------------------------------------------

#[test]
fn attention_with_a_dominant_key_returns_that_value_and_uniform_keys_average() {
    let d = 4usize;
    // keys: e_i scaled so the softmax saturates on the matching query
    let key: Vec<Vec<f32>> = (0..3)
        .map(|i| (0..d).map(|j| if i == j { 40.0 } else { 0.0 }).collect())
        .collect();
    let value: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 2.0, 0.0, 0.0],
        vec![0.0, 0.0, 3.0, 0.0],
    ];
    for i in 0..3 {
        let query = vec![(0..d)
            .map(|j| if i == j { 1.0 } else { 0.0 })
            .collect::<Vec<f32>>()];
        let out = scaled_dot_product_attention(&query, &key, &value, None);
        for (j, o) in out[0].iter().enumerate() {
            let expect = value[i][j];
            assert!((o - expect).abs() < 1e-4, "query {i}: {out:?}");
        }
    }
    // identical keys → mean of values
    let same_key: Vec<Vec<f32>> = vec![vec![1.0; d]; 3];
    let out = scaled_dot_product_attention(&[vec![0.5; d]], &same_key, &value, None);
    let mean: Vec<f32> = (0..d)
        .map(|j| value.iter().map(|v| v[j]).sum::<f32>() / 3.0)
        .collect();
    for (o, m) in out[0].iter().zip(&mean) {
        assert!((o - m).abs() < 1e-5, "{out:?}");
    }
    // causal mask: 0 on and below the diagonal, −inf above
    let m = causal_mask(4);
    for (i, row) in m.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if j <= i {
                assert_eq!(v, 0.0, "({i},{j})");
            } else {
                assert_eq!(v, f32::NEG_INFINITY, "({i},{j})");
            }
        }
    }
    // with the mask, the first query can only see the first key
    let out =
        scaled_dot_product_attention(&vec![vec![0.0; d]; 3], &key, &value, Some(&causal_mask(3)));
    for (o, v) in out[0].iter().zip(&value[0]) {
        assert!((o - v).abs() < 1e-5, "{out:?}");
    }
}

// ----------------------------------------------------------------------------
// Q8_K quantisation law (llama.cpp reference): d = max / 128, q = round(x / d),
// |x − q·d| ≤ d / 2, block sums are the sums of the stored bytes
// ----------------------------------------------------------------------------

#[test]
fn q8_k_reconstruction_error_is_bounded_by_half_a_step_and_sums_match() {
    let n = 3 * QK_K;
    let x: Vec<f32> = (0..n)
        .map(|i| ((i * 37) % 101) as f32 / 25.0 - 2.0)
        .collect();
    let blocks = quantize_row_q8_k(&x);
    assert_eq!(blocks.len(), 3);
    for (b, block) in blocks.iter().enumerate() {
        let xs = &x[b * QK_K..(b + 1) * QK_K];
        let (max_signed, amax) = xs.iter().fold((0.0f32, 0.0f32), |(m, am), &v| {
            if v.abs() > am {
                (v, v.abs())
            } else {
                (m, am)
            }
        });
        // d = max_signed / 128 (llama.cpp uses the signed extreme so it maps to ±128)
        assert!(
            (block.d - max_signed / 128.0).abs() <= 1e-7 * amax,
            "block {b}: d {}",
            block.d
        );
        for (j, &v) in xs.iter().enumerate() {
            let rec = f32::from(block.qs[j]) * block.d;
            // the signed extreme maps to ±128 and is clamped to the i8 range, so
            // values beyond 127.5 steps carry up to one full step of error
            // (llama.cpp reference behaviour); everything else half a step
            let steps = (v / block.d).abs();
            let bound = if steps > 127.5 {
                block.d.abs()
            } else {
                block.d.abs() / 2.0
            };
            assert!(
                (v - rec).abs() <= bound + 1e-6,
                "block {b} elem {j}: {v} vs {rec}"
            );
        }
        for g in 0..16 {
            let s: i32 = block.qs[g * 16..(g + 1) * 16]
                .iter()
                .map(|&q| i32::from(q))
                .sum();
            assert_eq!(i32::from(block.bsums[g]), s, "block {b} group {g}");
        }
    }
    // all-zero block: d = 0, every q = 0
    let z = quantize_row_q8_k(&vec![0.0; QK_K]);
    assert_eq!(z[0].d, 0.0);
    assert!(z[0].qs.iter().all(|&q| q == 0));
}
