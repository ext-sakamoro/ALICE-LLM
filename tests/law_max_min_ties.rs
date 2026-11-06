//! ALICE-Law G2 tie tests: max / min / clamp sites through the public API,
//! at the inputs where `lawmath` (alice-det-math) and other ways of writing
//! max / min disagree (an equal pair of zeros, a NaN operand)
//!
//! The clippy ban stops `f32::max` / `min` / `clamp` at a call site, not a
//! comparison written out by hand. These tests compare bits at the public API
//! so that such a rewrite of a site class changes a result here.
//!
//! Site classes with a test:
//! - min / max folds over a slice: `QuantizedInt8::quantize`
//!   (src/quantization.rs; the same fold opens `softmax_inplace` and the
//!   logit softmax of the decoder)
//! - max fold before a softmax: `sampling::softmax_of_checked`, in `src/sampling.rs` (the public
//!   `softmax_inplace` refuses a NaN first; attention scores reach the fold unchecked)
//! - clamp of a gradient: `StraightThroughEstimator::backward`
//!   (src/training.rs)
//!
//! The classes no tie input reaches from the public API, with the reason for
//! each, are listed in alice-law-conformance.toml (G2).

use alice_llm::quantization::{QuantizedInt4, QuantizedInt8};
use alice_llm::sampling::softmax_inplace;
use alice_llm::training::StraightThroughEstimator;

/// `f32::NAN` as `alice-det-math` returns it
const CANONICAL_NAN: u32 = 0x7fc0_0000;

fn payload_nan() -> f32 {
    f32::from_bits(0x7fc0_dead)
}

/// Both orders of an equal pair of zeros: the min is -0 and the max +0, so
/// the zero point (their midpoint) is +0. A comparison fold that keeps the
/// first of an equal pair gives -0 for one of the orders.
#[test]
fn quantize_of_a_pair_of_zeros_has_a_positive_zero_point() {
    for values in [[-0.0_f32, 0.0], [0.0, -0.0]] {
        let q = QuantizedInt8::quantize(&values);
        assert_eq!(q.zero_point.to_bits(), 0.0_f32.to_bits(), "{values:?}");
        assert_eq!(q.scale.to_bits(), 1.0_f32.to_bits());
    }
}

/// A NaN among the values is skipped by the min and max (a NaN operand gives
/// the other operand), so the range is that of the other values. A fold that
/// passes a NaN on gives a NaN zero point.
#[test]
fn quantize_skips_a_nan_in_the_range() {
    for values in [
        [1.0_f32, payload_nan(), 3.0],
        [1.0, 3.0, payload_nan()],
        [payload_nan(), 1.0, 3.0],
    ] {
        let q = QuantizedInt8::quantize(&values);
        assert_eq!(q.zero_point.to_bits(), 2.0_f32.to_bits(), "{values:?}");
        assert_eq!(q.scale.to_bits(), (1.0_f32 / 127.0).to_bits(), "{values:?}");
    }
}

/// The INT4 zero point is the min itself: -0 for an equal pair of zeros in
/// either order (a fold that keeps the first of a tie gives +0 for one
/// order), and a NaN among the values is skipped
#[test]
fn int4_zero_point_is_the_signed_min() {
    for values in [[-0.0_f32, 0.0], [0.0, -0.0]] {
        let q = QuantizedInt4::quantize(&values);
        assert_eq!(q.zero_point.to_bits(), (-0.0_f32).to_bits(), "{values:?}");
    }
    for values in [
        [1.0_f32, payload_nan(), 3.0],
        [1.0, 3.0, payload_nan()],
        [payload_nan(), 1.0, 3.0],
    ] {
        let q = QuantizedInt4::quantize(&values);
        assert_eq!(q.zero_point.to_bits(), 1.0_f32.to_bits(), "{values:?}");
        assert_eq!(q.scale.to_bits(), (2.0_f32 / 15.0).to_bits(), "{values:?}");
    }
}

/// `softmax_inplace` refuses a NaN before this fold. The fold itself keeps its tie test in
/// `src/sampling.rs` (`the_softmax_fold_skips_a_nan_in_the_maximum`), since attention scores reach it
/// unchecked.
#[test]
fn softmax_refuses_a_nan_and_leaves_it() {
    let mut v = [1.0_f32, 2.0, payload_nan()];
    assert_eq!(
        softmax_inplace(&mut v),
        Err(alice_llm::sampling::SamplingError::NonFinite { index: 2 })
    );
    assert_eq!(v[2].to_bits(), payload_nan().to_bits());
}

#[test]
fn gradient_clip_of_nan_and_negative_zero() {
    let ste = StraightThroughEstimator::new(0.5);
    let g = ste.backward(&[payload_nan(), -0.0, 0.0, 2.0, -3.0], true);
    let bits: Vec<u32> = g.iter().map(|x| x.to_bits()).collect();
    assert_eq!(
        bits,
        vec![
            CANONICAL_NAN,
            (-0.0_f32).to_bits(),
            0.0_f32.to_bits(),
            1.0_f32.to_bits(),
            (-1.0_f32).to_bits()
        ]
    );
}
