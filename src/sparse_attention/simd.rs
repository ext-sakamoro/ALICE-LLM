//! SIMD helpers for the sparse-attention hot loops.
//!
//! Each helper has two paths:
//!
//! * `#[cfg(feature = "simd")]` — `wide::f32x8` vectorized fast path over
//!   `chunks_exact(8)`, with a scalar tail for the remainder.
//! * default — plain scalar loop, ordering matches the SIMD path bit-for-bit
//!   for the aligned prefix (both use `+=` in element order), so numerical
//!   results agree modulo the usual FP re-association tolerance.
//!
//! All helpers assume `dst.len() == src.len()` and panic in debug builds
//! otherwise. Call sites must respect that.

#[cfg(feature = "simd")]
use wide::f32x8;

// Dot product
// ---------------------------------------------------------------------------

/// `Σ a[i] * b[i]`.
#[inline]
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(feature = "simd")]
    {
        let mut acc = f32x8::splat(0.0);
        let (a_chunks, a_rem) = a.as_chunks::<8>();
        let (b_chunks, b_rem) = b.as_chunks::<8>();
        for (av, bv) in a_chunks.iter().zip(b_chunks) {
            acc += f32x8::from(*av) * f32x8::from(*bv);
        }
        let mut s = acc.reduce_add();
        for (a, b) in a_rem.iter().zip(b_rem) {
            s += a * b;
        }
        s
    }
    #[cfg(not(feature = "simd"))]
    {
        let mut s = 0.0f32;
        for (a, b) in a.iter().zip(b) {
            s += a * b;
        }
        s
    }
}

// AXPY (dst += scale * src)
// ---------------------------------------------------------------------------

/// `dst[i] += scale * src[i]`.
#[inline]
pub fn axpy(dst: &mut [f32], scale: f32, src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(feature = "simd")]
    {
        let sc = f32x8::splat(scale);
        let (dst_chunks, dst_rem) = dst.as_chunks_mut::<8>();
        let (src_chunks, src_rem) = src.as_chunks::<8>();
        for (dc, sv) in dst_chunks.iter_mut().zip(src_chunks) {
            *dc = (f32x8::from(*dc) + f32x8::from(*sv) * sc).to_array();
        }
        for (d, s) in dst_rem.iter_mut().zip(src_rem) {
            *d += scale * *s;
        }
    }
    #[cfg(not(feature = "simd"))]
    {
        for (d, s) in dst.iter_mut().zip(src) {
            *d += scale * *s;
        }
    }
}

// In-place scalar multiply
// ---------------------------------------------------------------------------

/// `dst[i] *= scale`.
#[inline]
pub fn scale_in_place(dst: &mut [f32], scale: f32) {
    #[cfg(feature = "simd")]
    {
        let sc = f32x8::splat(scale);
        let (chunks, rem) = dst.as_chunks_mut::<8>();
        for chunk in chunks.iter_mut() {
            *chunk = (f32x8::from(*chunk) * sc).to_array();
        }
        for v in rem {
            *v *= scale;
        }
    }
    #[cfg(not(feature = "simd"))]
    {
        for v in dst.iter_mut() {
            *v *= scale;
        }
    }
}

// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
        let mut s = 0.0f32;
        for (a, b) in a.iter().zip(b) {
            s += a * b;
        }
        s
    }

    #[test]
    fn dot_matches_scalar_for_various_lengths() {
        for &n in &[0, 1, 4, 7, 8, 9, 16, 17, 31, 32, 33, 64, 128] {
            let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.13 - 0.5).collect();
            let b: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07).sin()).collect();
            let d_simd = dot(&a, &b);
            let d_ref = scalar_dot(&a, &b);
            let tol = 1e-5 * (n as f32 + 1.0);
            assert!(
                (d_simd - d_ref).abs() < tol,
                "n={n}: simd={d_simd}, scalar={d_ref}"
            );
        }
    }

    #[test]
    fn axpy_matches_scalar_for_various_lengths() {
        for &n in &[0, 3, 8, 15, 16, 63, 128] {
            let src: Vec<f32> = (0..n).map(|i| (i as f32) * 0.11).collect();
            let mut dst_simd: Vec<f32> = (0..n).map(|i| (i as f32) * 0.2).collect();
            let mut dst_ref = dst_simd.clone();
            let scale = 0.37f32;
            axpy(&mut dst_simd, scale, &src);
            for (d, s) in dst_ref.iter_mut().zip(&src) {
                *d += scale * *s;
            }
            for (i, (a, b)) in dst_simd.iter().zip(&dst_ref).enumerate() {
                assert!((a - b).abs() < 1e-5, "n={n} idx={i}: simd={a}, scalar={b}");
            }
        }
    }

    #[test]
    fn scale_in_place_matches_scalar_for_various_lengths() {
        for &n in &[0, 1, 8, 12, 16, 17, 64] {
            let mut dst_simd: Vec<f32> = (0..n).map(|i| (i as f32) - 3.0).collect();
            let mut dst_ref = dst_simd.clone();
            let scale = 0.42f32;
            scale_in_place(&mut dst_simd, scale);
            for v in &mut dst_ref {
                *v *= scale;
            }
            for (i, (a, b)) in dst_simd.iter().zip(&dst_ref).enumerate() {
                assert!((a - b).abs() < 1e-6, "n={n} idx={i}: simd={a}, scalar={b}");
            }
        }
    }
}
