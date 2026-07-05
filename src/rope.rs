//! rope.

// RoPE (Rotary Position Embedding)
// ---------------------------------------------------------------------------

/// Apply Rotary Position Embedding to a vector at a given position.
///
/// `vec` has length `head_dim` (must be even).
/// `position` is the token position in the sequence.
/// `base` is the `RoPE` base frequency (typically 10000).
#[must_use]
pub fn apply_rope(vec: &[f32], position: usize, base: f32) -> Vec<f32> {
    let dim = vec.len();
    let mut out = vec![0.0f32; dim];
    let half = dim / 2;

    for i in 0..half {
        let freq = 1.0 / base.powf(2.0 * i as f32 / dim as f32);
        let angle = position as f32 * freq;
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        out[i] = vec[i].mul_add(cos_a, -(vec[i + half] * sin_a));
        out[i + half] = vec[i].mul_add(sin_a, vec[i + half] * cos_a);
    }
    out
}

/// Apply `RoPE` to all positions in a batch of vectors.
///
/// `vectors[pos][dim]` — returns same shape with `RoPE` applied.
#[must_use]
pub fn apply_rope_batch(vectors: &[Vec<f32>], base: f32) -> Vec<Vec<f32>> {
    vectors
        .iter()
        .enumerate()
        .map(|(pos, v)| apply_rope(v, pos, base))
        .collect()
}
