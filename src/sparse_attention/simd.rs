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
        let mut a_chunks = a.chunks_exact(8);
        let mut b_chunks = b.chunks_exact(8);
        for (av, bv) in (&mut a_chunks).zip(&mut b_chunks) {
            let a_arr: [f32; 8] = av.try_into().expect("chunks_exact(8) yields 8 elems");
            let b_arr: [f32; 8] = bv.try_into().expect("chunks_exact(8) yields 8 elems");
            let av = f32x8::from(a_arr);
            let bv = f32x8::from(b_arr);
            acc += av * bv;
        }
        let mut s = acc.reduce_add();
        for (a, b) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
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
        let mut dst_chunks = dst.chunks_exact_mut(8);
        let mut src_chunks = src.chunks_exact(8);
        for (dc, sc_chunk) in (&mut dst_chunks).zip(&mut src_chunks) {
            let d_arr: [f32; 8] = (&*dc)
                .try_into()
                .expect("chunks_exact_mut(8) yields 8 elems");
            let s_arr: [f32; 8] = sc_chunk.try_into().expect("chunks_exact(8) yields 8 elems");
            let dv = f32x8::from(d_arr);
            let sv = f32x8::from(s_arr);
            let out = dv + sv * sc;
            let arr = out.to_array();
            dc.copy_from_slice(&arr);
        }
        for (d, s) in dst_chunks
            .into_remainder()
            .iter_mut()
            .zip(src_chunks.remainder())
        {
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
        let mut chunks = dst.chunks_exact_mut(8);
        for chunk in &mut chunks {
            let arr_in: [f32; 8] = (&*chunk)
                .try_into()
                .expect("chunks_exact_mut(8) yields 8 elems");
            let out = f32x8::from(arr_in) * sc;
            chunk.copy_from_slice(&out.to_array());
        }
        for v in chunks.into_remainder() {
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
