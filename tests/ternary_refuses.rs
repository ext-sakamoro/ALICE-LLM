//! Weights that have no ternary form are refused, and the rest keep their form.
//!
//! oracle: the threshold of a row is `ratio * mean(|w|)`, by the definition in `TernaryRow::from_f32`.
//! One weight that is not finite makes that mean not finite, and then no weight is above it: the row
//! would come out all zero, with no sign that anything was wrong. Rows here are small enough to read
//! the expected masks from their values by hand.

use alice_llm::gguf::{
    GgmlType, SparseTernaryMatrix, SparseTernaryRow, TernaryError, TernaryMatrix, TernaryRow,
};
use TernaryError::{InvalidThreshold, LengthMismatch, NonFiniteWeight, UnsupportedType};

const BAD: [f32; 3] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];

/// 1, 2, .. 32: the mean is 16.5, so at a ratio of 0.3 the threshold is 4.95 and 5.. pass it.
fn weights() -> Vec<f32> {
    (1..=32u8).map(f32::from).collect()
}

fn with_bad(bad: f32, at: usize) -> Vec<f32> {
    let mut w = weights();
    w[at] = bad;
    w
}

#[test]
fn a_row_with_a_weight_that_is_not_finite_is_refused() {
    for bad in BAD {
        let w = with_bad(bad, 3);
        let index = NonFiniteWeight { index: 3 };
        assert_eq!(TernaryRow::from_f32(&w, 0.3).err(), Some(index), "{bad}");
        assert_eq!(
            SparseTernaryRow::from_f32(&w, 0.3, 8).err(),
            Some(index),
            "{bad}"
        );
    }
}

#[test]
fn a_threshold_ratio_is_finite_and_at_least_zero() {
    for ratio in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1] {
        assert_eq!(
            TernaryRow::from_f32(&weights(), ratio).err(),
            Some(InvalidThreshold),
            "{ratio}"
        );
    }
    assert!(TernaryRow::from_f32(&weights(), 0.0).is_ok());
}

#[test]
fn a_finite_row_keeps_its_form() {
    // above 4.95: 5..=32, the first 4 stay out. Their mean is 18.5
    let row = TernaryRow::from_f32(&weights(), 0.3).unwrap();
    assert_eq!(row.pos_mask, [0xF0, 0xFF, 0xFF, 0xFF]);
    assert_eq!(row.neg_mask, [0, 0, 0, 0]);
    assert_eq!(row.scale.to_bits(), 18.5_f32.to_bits());
}

#[test]
fn a_block_keeps_its_largest_weights() {
    // one block of 16: 5..=16 are above the threshold, and the 2 largest are 15 and 16
    let w: Vec<f32> = (1..=16u8).map(f32::from).collect();
    let row = TernaryRow::from_f32(&w, 0.5).unwrap();
    let sparse = SparseTernaryRow::from_ternary_row(&row, &w, 2).unwrap();
    assert_eq!(sparse.active_masks, [0xC000]);
    assert_eq!(sparse.scale.to_bits(), 15.5_f32.to_bits());
}

#[test]
fn original_weights_are_one_finite_weight_per_column() {
    let row = TernaryRow::from_f32(&weights(), 0.3).unwrap();
    for bad in BAD {
        // column 15 is above the threshold, so it is one of the weights the block sorts
        assert_eq!(
            SparseTernaryRow::from_ternary_row(&row, &with_bad(bad, 15), 8).err(),
            Some(NonFiniteWeight { index: 15 }),
            "{bad}"
        );
    }
    assert_eq!(
        SparseTernaryRow::from_ternary_row(&row, &weights()[..31], 8).err(),
        Some(LengthMismatch {
            expected: 32,
            got: 31
        })
    );
}

fn f32_bytes(w: &[f32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn a_matrix_refuses_what_a_row_refuses() {
    for bad in BAD {
        let w = with_bad(bad, 3);
        assert_eq!(
            TernaryMatrix::from_quantized(&f32_bytes(&w), GgmlType::F32, 1, 32, 0.3).err(),
            Some(NonFiniteWeight { index: 3 }),
            "{bad}"
        );
        assert_eq!(
            SparseTernaryMatrix::from_f32_weights(&w, 1, 32, 0.3, 8).err(),
            Some(NonFiniteWeight { index: 3 }),
            "{bad}"
        );
    }
    assert!(
        TernaryMatrix::from_quantized(&f32_bytes(&weights()), GgmlType::F32, 1, 32, 0.3).is_ok()
    );
}

#[test]
fn a_matrix_refuses_what_does_not_match_its_shape() {
    assert_eq!(
        SparseTernaryMatrix::from_f32_weights(&weights(), 2, 32, 0.3, 8).err(),
        Some(LengthMismatch {
            expected: 64,
            got: 32
        })
    );
    assert_eq!(
        SparseTernaryMatrix::from_f32_weights(&weights(), usize::MAX, 2, 0.3, 8).err(),
        Some(LengthMismatch {
            expected: usize::MAX,
            got: 32
        })
    );
    assert_eq!(
        TernaryMatrix::from_quantized(&[0; 4], GgmlType::F32, 2, 32, 0.3).err(),
        Some(LengthMismatch {
            expected: 256,
            got: 4
        })
    );
    let dense =
        TernaryMatrix::from_quantized(&f32_bytes(&weights()), GgmlType::F32, 1, 32, 0.3).unwrap();
    assert_eq!(
        SparseTernaryMatrix::from_ternary(&dense, &[], 8).err(),
        Some(LengthMismatch {
            expected: 1,
            got: 0
        })
    );
    assert!(SparseTernaryMatrix::from_ternary(&dense, &[weights()], 8).is_ok());
}

#[test]
fn a_tensor_type_with_no_dequantizer_is_refused() {
    // it was read as a row of zeros
    assert_eq!(
        TernaryMatrix::from_quantized(&[0; 2048], GgmlType::Q4_0, 1, 32, 0.3).err(),
        Some(UnsupportedType(GgmlType::Q4_0))
    );
}
