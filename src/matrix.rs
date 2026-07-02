//! Flat cache-aligned matrix (SoA row-major layout).
//!
//! Replaces `Vec<Vec<T>>` (= per-row heap allocation, fragmented) with a
//! single contiguous `Vec<T>` so that rows sit next to each other in
//! memory. Cache lines coalesce across the inner loop of matmul / dot,
//! which is the largest win before SIMD.
//!
//! FMA is used on the hot inner loop via `f32::mul_add`, and an
//! optional `wide::f32x8` SIMD path is exposed under the `simd`
//! feature (`--features simd`).

use std::ops::Range;

/// Row-major flat 2D matrix.
///
/// `data.len() == rows * cols`, laid out as
/// `[row0_col0, row0_col1, ..., row0_col{cols-1}, row1_col0, ...]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Clone + Default> Matrix<T> {
    /// Creates a matrix filled with `T::default()`.
    #[must_use]
    pub fn new_zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![T::default(); rows * cols],
            rows,
            cols,
        }
    }

    /// Builds a matrix from row-vectors (interop with legacy `Vec<Vec<T>>`).
    ///
    /// # Panics
    /// Panics when the rows have non-uniform length.
    #[must_use]
    pub fn from_rows(rows_vec: Vec<Vec<T>>) -> Self {
        let rows = rows_vec.len();
        let cols = rows_vec.first().map_or(0, Vec::len);
        assert!(
            rows_vec.iter().all(|r| r.len() == cols),
            "Matrix::from_rows: rows must have uniform length"
        );
        let mut data = Vec::with_capacity(rows * cols);
        for row in rows_vec {
            data.extend(row);
        }
        Self { data, rows, cols }
    }

    /// Returns row-vectors (interop with legacy `Vec<Vec<T>>`).
    #[must_use]
    pub fn to_rows(&self) -> Vec<Vec<T>> {
        (0..self.rows).map(|i| self.row(i).to_vec()).collect()
    }
}

impl<T> Matrix<T> {
    #[must_use]
    pub const fn rows_count(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn cols_count(&self) -> usize {
        self.cols
    }

    #[must_use]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns row `i` as a slice.
    #[must_use]
    pub fn row(&self, i: usize) -> &[T] {
        let r = self.row_range(i);
        &self.data[r]
    }

    /// Returns row `i` as a mutable slice.
    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        let r = self.row_range(i);
        &mut self.data[r]
    }

    fn row_range(&self, i: usize) -> Range<usize> {
        assert!(i < self.rows, "row index out of bounds");
        (i * self.cols)..((i + 1) * self.cols)
    }
}

impl<T: Copy + Default> Matrix<T> {
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> T {
        assert!(i < self.rows && j < self.cols, "index out of bounds");
        self.data[i * self.cols + j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: T) {
        assert!(i < self.rows && j < self.cols, "index out of bounds");
        self.data[i * self.cols + j] = v;
    }
}

// ---------------------------------------------------------------------------
// f32-specific numeric ops
// ---------------------------------------------------------------------------

impl Matrix<f32> {
    /// Naive but cache-friendly matrix multiplication.
    ///
    /// Hot inner loop is `j`-contiguous with FMA (`mul_add`). Under the
    /// `simd` feature the row-times-row inner product uses `f32x8`.
    ///
    /// # Panics
    /// Panics when `self.cols != rhs.rows`.
    #[must_use]
    pub fn matmul_flat(&self, rhs: &Self) -> Self {
        assert_eq!(
            self.cols, rhs.rows,
            "matmul dim mismatch: {} vs {}",
            self.cols, rhs.rows
        );
        let mut out = Self::new_zeros(self.rows, rhs.cols);
        for i in 0..self.rows {
            let a_row = &self.data[i * self.cols..(i + 1) * self.cols];
            for (k, &a_ik) in a_row.iter().enumerate() {
                let b_row = &rhs.data[k * rhs.cols..(k + 1) * rhs.cols];
                let out_row = &mut out.data[i * rhs.cols..(i + 1) * rhs.cols];
                for j in 0..rhs.cols {
                    out_row[j] = a_ik.mul_add(b_row[j], out_row[j]);
                }
            }
        }
        out
    }

    /// Returns a transposed copy.
    #[must_use]
    pub fn transpose_flat(&self) -> Self {
        let mut out = Self::new_zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        out
    }

    /// `y = self · x`.
    ///
    /// # Panics
    /// Panics when `self.cols != x.len()`.
    #[must_use]
    pub fn matvec_flat(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(self.cols, x.len(), "matvec dim mismatch");
        (0..self.rows)
            .map(|i| dot_flat(&self.data[i * self.cols..(i + 1) * self.cols], x))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Dot product (scalar + SIMD)
// ---------------------------------------------------------------------------

/// Flat FMA-friendly dot product. Uses `f32x8` when compiled with
/// `--features simd`.
///
/// # Panics
/// Panics when `a.len() != b.len()`.
#[must_use]
pub fn dot_flat(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot dim mismatch");
    #[cfg(feature = "simd")]
    {
        dot_flat_simd(a, b)
    }
    #[cfg(not(feature = "simd"))]
    {
        let mut acc = 0.0_f32;
        for (&ai, &bi) in a.iter().zip(b.iter()) {
            acc = ai.mul_add(bi, acc);
        }
        acc
    }
}

#[cfg(feature = "simd")]
#[must_use]
fn dot_flat_simd(a: &[f32], b: &[f32]) -> f32 {
    use wide::f32x8;
    let n = a.len();
    let chunks = n / 8;
    let mut acc = f32x8::ZERO;
    for c in 0..chunks {
        let base = c * 8;
        let av = f32x8::new([
            a[base],
            a[base + 1],
            a[base + 2],
            a[base + 3],
            a[base + 4],
            a[base + 5],
            a[base + 6],
            a[base + 7],
        ]);
        let bv = f32x8::new([
            b[base],
            b[base + 1],
            b[base + 2],
            b[base + 3],
            b[base + 4],
            b[base + 5],
            b[base + 6],
            b[base + 7],
        ]);
        acc = av.mul_add(bv, acc);
    }
    let arr = acc.to_array();
    let mut sum = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
    for k in (chunks * 8)..n {
        sum = a[k].mul_add(b[k], sum);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_and_to_rows_round_trip() {
        let rows = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0]];
        let m = Matrix::from_rows(rows.clone());
        assert_eq!(m.rows_count(), 2);
        assert_eq!(m.cols_count(), 2);
        assert_eq!(m.to_rows(), rows);
    }

    #[test]
    fn zeros_has_default_values() {
        let m: Matrix<f32> = Matrix::new_zeros(3, 4);
        assert_eq!(m.rows_count(), 3);
        assert_eq!(m.cols_count(), 4);
        assert!(m.data().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn get_set_round_trip() {
        let mut m: Matrix<f32> = Matrix::new_zeros(2, 3);
        m.set(1, 2, 7.5);
        assert_eq!(m.get(1, 2), 7.5);
        assert_eq!(m.get(0, 0), 0.0);
    }

    #[test]
    fn row_and_row_mut_slice() {
        let mut m = Matrix::from_rows(vec![vec![1.0_f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(m.row(1), &[4.0, 5.0, 6.0]);
        m.row_mut(0)[0] = 9.0;
        assert_eq!(m.row(0), &[9.0, 2.0, 3.0]);
    }

    #[test]
    fn matmul_flat_matches_reference() {
        let a = Matrix::from_rows(vec![vec![1.0_f32, 2.0], vec![3.0, 4.0]]);
        let b = Matrix::from_rows(vec![vec![5.0_f32, 6.0], vec![7.0, 8.0]]);
        let c = a.matmul_flat(&b);
        assert_eq!(c.to_rows(), vec![vec![19.0, 22.0], vec![43.0, 50.0]]);
    }

    #[test]
    fn matmul_flat_non_square() {
        // (2x3) · (3x1) = (2x1)
        let a = Matrix::from_rows(vec![vec![1.0_f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let b = Matrix::from_rows(vec![vec![1.0_f32], vec![0.0], vec![-1.0]]);
        let c = a.matmul_flat(&b);
        assert_eq!(c.rows_count(), 2);
        assert_eq!(c.cols_count(), 1);
        // row0: 1*1 + 2*0 + 3*(-1) = -2
        // row1: 4*1 + 5*0 + 6*(-1) = -2
        assert_eq!(c.to_rows(), vec![vec![-2.0], vec![-2.0]]);
    }

    #[test]
    fn transpose_flat_swaps_axes() {
        let a = Matrix::from_rows(vec![vec![1.0_f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let t = a.transpose_flat();
        assert_eq!(t.rows_count(), 3);
        assert_eq!(t.cols_count(), 2);
        assert_eq!(
            t.to_rows(),
            vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
        );
    }

    #[test]
    fn transpose_flat_involutive() {
        let a = Matrix::from_rows(vec![vec![1.0_f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(a.transpose_flat().transpose_flat(), a);
    }

    #[test]
    fn matvec_flat_correct() {
        let a = Matrix::from_rows(vec![vec![1.0_f32, 2.0], vec![3.0, 4.0]]);
        let v = vec![5.0_f32, 6.0];
        // row0: 1*5 + 2*6 = 17; row1: 3*5 + 4*6 = 39
        assert_eq!(a.matvec_flat(&v), vec![17.0, 39.0]);
    }

    #[test]
    fn dot_flat_basic() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![4.0_f32, 5.0, 6.0];
        // 4 + 10 + 18 = 32
        assert_eq!(dot_flat(&a, &b), 32.0);
    }

    #[test]
    fn dot_flat_handles_multiples_of_eight() {
        // exercises the f32x8 fast path when the simd feature is on
        let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
        // sum_{i=0..15} i * (i+1) = sum(i^2) + sum(i)
        //                         = 15*16*31/6 + 15*16/2 = 1240 + 120 = 1360
        assert!((dot_flat(&a, &b) - 1360.0).abs() < 1e-3);
    }

    #[test]
    fn dot_flat_tail_after_eight_chunks() {
        // 11 elements = one f32x8 chunk + 3-element tail
        let a: Vec<f32> = (0..11).map(|i| i as f32).collect();
        let b = vec![1.0_f32; 11];
        // 0 + 1 + 2 + ... + 10 = 55
        assert!((dot_flat(&a, &b) - 55.0).abs() < 1e-3);
    }
}
