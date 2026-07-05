//! linalg.

// Linear algebra helpers
// ---------------------------------------------------------------------------

/// Dot product of two slices.
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix-vector multiply: `mat[rows][cols] * vec[cols] -> out[rows]`.
#[must_use]
pub fn matvec(mat: &[Vec<f32>], vec: &[f32]) -> Vec<f32> {
    mat.iter().map(|row| dot(row, vec)).collect()
}

/// Matrix-matrix multiply: `a[m][k] * b[k][n] -> out[m][n]`.
#[must_use]
pub fn matmul(lhs: &[Vec<f32>], rhs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if lhs.is_empty() || rhs.is_empty() {
        return Vec::new();
    }
    let rows = lhs.len();
    let cols = rhs[0].len();
    let inner = rhs.len();
    let mut out = vec![vec![0.0f32; cols]; rows];
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0f32;
            for idx in 0..inner {
                sum += lhs[row][idx] * rhs[idx][col];
            }
            out[row][col] = sum;
        }
    }
    out
}

/// Transpose a 2D matrix.
#[must_use]
pub fn transpose(mat: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if mat.is_empty() {
        return Vec::new();
    }
    let rows = mat.len();
    let cols = mat[0].len();
    let mut out = vec![vec![0.0f32; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j][i] = mat[i][j];
        }
    }
    out
}

/// RMS normalization.
#[must_use]
pub fn rms_norm(v: &[f32], eps: f32) -> Vec<f32> {
    let sq_mean: f32 = v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32;
    let scale = 1.0 / (sq_mean + eps).sqrt();
    v.iter().map(|x| x * scale).collect()
}

/// Layer normalization.
#[must_use]
pub fn layer_norm(v: &[f32], eps: f32) -> Vec<f32> {
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    v.iter().map(|x| (x - mean) * inv_std).collect()
}

/// `SiLU` (Swish) activation.
#[must_use]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `GeLU` activation (approximate).
#[must_use]
pub fn gelu(x: f32) -> f32 {
    let c = (2.0 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (0.044_715 * x * x).mul_add(x, x)).tanh())
}
