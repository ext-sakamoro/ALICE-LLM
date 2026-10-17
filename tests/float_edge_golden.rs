//! ALICE-Law G4: signed zeros and exact cancellations through the public
//! kernels, against fixed bit patterns
//!
//! CI runs this file in the debug and the release profile on x86_64 (ubuntu)
//! and aarch64 (macOS); every pattern must hold on all four.

#![cfg(not(feature = "platform-math"))]

use alice_llm::linalg::{gelu, layer_norm, rms_norm, silu};
use alice_llm::quantization::QuantizedInt8;
use alice_llm::rope::apply_rope;
use alice_llm::sampling::softmax;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

const POS0: u32 = 0x0000_0000;
const NEG0: u32 = 0x8000_0000;

#[test]
fn activations_keep_the_sign_of_zero() {
    assert_eq!(silu(0.0).to_bits(), POS0);
    assert_eq!(silu(-0.0).to_bits(), NEG0);
    assert_eq!(gelu(0.0).to_bits(), POS0);
    assert_eq!(gelu(-0.0).to_bits(), NEG0);
}

/// A constant vector cancels exactly against its mean: every output is +0
/// (x - mean = +0 for equal operands under round-to-nearest-even)
#[test]
fn layer_norm_of_a_constant_vector_is_positive_zero() {
    for c in [3.0_f32, -7.25, 0.0, -0.0] {
        let out = layer_norm(&[c; 8], 1e-5);
        assert_eq!(bits(&out), vec![POS0; 8], "constant {c}");
    }
}

#[test]
fn rms_norm_of_zeros_keeps_their_signs() {
    let out = rms_norm(&[0.0, -0.0, 0.0, -0.0], 1e-6);
    assert_eq!(bits(&out), vec![POS0, NEG0, POS0, NEG0]);
}

/// Equal logits: x - max is exactly +0, exp(+0) = 1, the sum is exactly 4
#[test]
fn softmax_of_equal_logits_is_exactly_uniform() {
    for c in [0.0_f32, -0.0, 1.5, -88.0] {
        assert_eq!(
            bits(&softmax(&[c; 4])),
            vec![0.25_f32.to_bits(); 4],
            "logit {c}"
        );
    }
}

/// Position 0: cos 0 = 1 and sin 0 = 0 exactly. The pairs are (i, i + 4);
/// out[i] = fma(v[i], 1, -(v[i + 4] * 0)) and out[i + 4] = fma(v[i], 0,
/// v[i + 4] * 1), so the values come back unchanged and only the signs of
/// zeros follow round-to-nearest-even (+0 + -0 = +0)
#[test]
fn rope_at_position_zero_returns_the_values() {
    let v = [1.0_f32, -0.0, 0.0, -2.5, 3.25, -0.0, 7.0, 0.0];
    let want = [
        1.0_f32.to_bits(),
        // fma(-0, 1, -(-0 * 0)) = -0 + +0 = +0
        POS0,
        POS0,
        (-2.5_f32).to_bits(),
        // fma(1, 0, 3.25) = 3.25
        3.25_f32.to_bits(),
        // fma(-0, 0, -0 * 1) = -0 + -0 = -0
        NEG0,
        7.0_f32.to_bits(),
        // fma(-2.5, 0, 0 * 1) = -0 + +0 = +0
        POS0,
    ];
    assert_eq!(bits(&apply_rope(&v, 0, 10_000.0)), want);
}

/// A slice of one repeated value has range 0: scale 1, the zero point is the
/// value, and every code is 0
#[test]
fn quantize_of_a_constant_slice_is_exact() {
    let q = QuantizedInt8::quantize(&[2.5; 6]);
    assert_eq!(q.scale.to_bits(), 1.0_f32.to_bits());
    assert_eq!(q.zero_point.to_bits(), 2.5_f32.to_bits());
    assert_eq!(q.data, vec![0_i8; 6]);
    assert_eq!(bits(&q.dequantize()), vec![2.5_f32.to_bits(); 6]);
}
