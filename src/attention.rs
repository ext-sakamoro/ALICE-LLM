//! attention.

use crate::sampling::softmax_inplace;

// Attention
// ---------------------------------------------------------------------------

/// Scaled dot-product attention.
///
/// `query`, `key`, `value` are 2D: `[seq_len, head_dim]`.
/// Returns the attention output `[query_len, head_dim]`.
#[must_use]
pub fn scaled_dot_product_attention(
    query: &[Vec<f32>],
    key: &[Vec<f32>],
    value: &[Vec<f32>],
    mask: Option<&[Vec<f32>]>,
) -> Vec<Vec<f32>> {
    if query.is_empty() || key.is_empty() || value.is_empty() {
        return Vec::new();
    }
    let head_dim = query[0].len();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_len = query.len();
    let k_len = key.len();

    // Compute attention scores: Q * K^T / sqrt(d)
    let mut scores = vec![vec![0.0f32; k_len]; q_len];
    for i in 0..q_len {
        for j in 0..k_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += query[i][d] * key[j][d];
            }
            scores[i][j] = dot * scale;
        }
    }

    // Apply mask
    if let Some(m) = mask {
        for i in 0..q_len {
            for j in 0..k_len {
                if j < m[i].len() {
                    scores[i][j] += m[i][j];
                }
            }
        }
    }

    // Softmax per row
    for row in &mut scores {
        softmax_inplace(row);
    }

    // Weighted sum of values
    let v_dim = value[0].len();
    let mut output = vec![vec![0.0f32; v_dim]; q_len];
    for i in 0..q_len {
        for j in 0..k_len {
            let w = scores[i][j];
            for d in 0..v_dim {
                output[i][d] += w * value[j][d];
            }
        }
    }

    output
}

/// Causal (lower-triangular) attention mask.
#[must_use]
pub fn causal_mask(seq_len: usize) -> Vec<Vec<f32>> {
    let mut mask = vec![vec![0.0f32; seq_len]; seq_len];
    for (i, row) in mask.iter_mut().enumerate() {
        for val in row.iter_mut().skip(i + 1) {
            *val = f32::NEG_INFINITY;
        }
    }
    mask
}
