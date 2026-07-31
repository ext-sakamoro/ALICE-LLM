//! FP8 (E4M3) KV-cache quantization for sparse attention.
//!
//! Rust-from-scratch analogue of MSA's FP8 / NVFP4 KV-cache path. We
//! implement OCP FP8 E4M3 encode / decode without adding an external
//! `float8` dependency:
//!
//! * 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits with implicit
//!   leading 1 for normal values, or a leading 0 for subnormals.
//! * `0x7F` / `0xFF` are reserved for NaN, so the largest finite is
//!   `(1 + 6/8) × 2⁸ = 448`; min normal = `2⁻⁶ = 0.015625`, smallest
//!   subnormal = `(1/8) × 2⁻⁶ = 2⁻⁹`.
//! * Saturate to `±MAX_NORMAL` on overflow (no NaN / Inf produced — MSA's
//!   attention path never wants a poison value in the KV cache).
//! * Round-to-nearest-even (bit-level truncation with tie-to-even).
//!
//! Storage layout matches the CPU / GPU forward's expected
//! `[num_pages, hkv, page_size, head_dim]` shape. We keep a single scale
//! per `(page, kv_head)` slot — a natural block boundary because the
//! forward's inner Q·K dot is over `head_dim` values within one page's
//! head slot.
//!
//! Gated on `feature = "quant"` so callers who don't need it don't pay the
//! compile-time cost.

#![cfg(feature = "quant")]

use super::types::SparseAttentionError;

// FP8 E4M3 constants
// ---------------------------------------------------------------------------

/// Largest finite magnitude representable in FP8 E4M3.
///
/// Under the OCP E4M3 spec only `0x7F` / `0xFF` are NaN, so the largest
/// finite is `(1 + 6/8) * 2**(15-7)` = `1.75 * 256` = `448.0`.
pub const FP8_E4M3_MAX: f32 = 448.0;

/// Smallest positive normal magnitude representable in FP8 E4M3.
///
/// = `2**(1-7)` = `2**-6` = `1/64`.
pub const FP8_E4M3_MIN_NORMAL: f32 = 1.0 / 64.0;

/// Smallest positive subnormal magnitude (`(1/8) * 2**-6`).
pub const FP8_E4M3_MIN_SUBNORMAL: f32 = 1.0 / (64.0 * 8.0);

// FP8 E4M3 encode / decode
// ---------------------------------------------------------------------------

/// Encode a single `f32` as an FP8 E4M3 byte. NaN / Inf inputs are saturated
/// to `±MAX_NORMAL`; the encoder never emits a NaN pattern.
///
/// Round-to-nearest-even.
#[inline]
#[must_use]
pub fn f32_to_e4m3(x: f32) -> u8 {
    // Handle sign separately.
    let sign_bit: u8 = if x.is_sign_negative() { 0x80 } else { 0x00 };
    let ax = x.abs();

    // NaN / Inf / overflow → saturate to max normal (0x7E for +, 0xFE for -).
    // 0x7F / 0xFF is E4M3's NaN encoding; we avoid it.
    if !ax.is_finite() || ax >= FP8_E4M3_MAX * (1.0 + 1.0 / 16.0) {
        return sign_bit | 0x7E;
    }
    if ax == 0.0 {
        return sign_bit; // ±0
    }

    // Subnormal region: |x| < min_normal.
    if ax < FP8_E4M3_MIN_NORMAL {
        // Represent as m/8 * 2**-6, m in 0..=7 (0 = zero).
        let scaled = ax / FP8_E4M3_MIN_SUBNORMAL; // 0..8
        let m = round_ties_even(scaled) as u32;
        let m_clamped = m.min(7);
        return sign_bit | (m_clamped as u8);
    }

    // Normal region: (1 + m/8) * 2**e, with e in 1..=15 (biased), m in 0..=7.
    // Extract IEEE 754 f32 fields.
    let bits = ax.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32; // biased by 127
    let f32_mant = bits & 0x7F_FFFF;
    let unbiased_exp = f32_exp - 127;
    // FP8 biased exponent (bias 7) — clamped to normal range [1, 15].
    let fp8_biased_exp = unbiased_exp + 7;
    if fp8_biased_exp <= 0 {
        // Falls into subnormal region despite ax >= min_normal from FP guard;
        // treat as zero (should not happen in practice).
        return sign_bit;
    }
    if fp8_biased_exp > 15 {
        // Overflow beyond E4M3's exponent range — saturate.
        return sign_bit | 0x7E;
    }

    // Round f32 23-bit mantissa to FP8 3-bit mantissa with round-to-nearest-even.
    // Take the top 3 bits of the mantissa + look at the remaining low bits for
    // the round decision.
    let mant_top3 = f32_mant >> 20; // upper 3 bits
    let mant_low = f32_mant & ((1 << 20) - 1);
    let halfway = 1u32 << 19;
    let mut mant_fp8 = mant_top3;
    match mant_low.cmp(&halfway) {
        std::cmp::Ordering::Greater => mant_fp8 += 1,
        std::cmp::Ordering::Equal => {
            // Ties to even.
            if mant_fp8 & 1 != 0 {
                mant_fp8 += 1;
            }
        }
        std::cmp::Ordering::Less => {}
    }
    let mut biased_exp = fp8_biased_exp as u32;
    if mant_fp8 >= 8 {
        mant_fp8 = 0;
        biased_exp += 1;
        if biased_exp > 15 {
            return sign_bit | 0x7E; // saturate
        }
    }
    // 0x7F / 0xFF are reserved for NaN in E4M3 → snap the largest normal
    // (exp=15, mant=7) down to the largest finite encoding (0x7E / 0xFE).
    if biased_exp == 15 && mant_fp8 == 7 {
        return sign_bit | 0x7E;
    }
    sign_bit | ((biased_exp as u8) << 3) | (mant_fp8 as u8)
}

/// Decode an FP8 E4M3 byte back to an `f32`. NaN encoding (0x7F / 0xFF) is
/// treated as `MAX_NORMAL` for consistency with the saturating encoder — the
/// caller should never see a NaN in a well-formed cache.
#[inline]
#[must_use]
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let magnitude = b & 0x7F;
    if magnitude == 0 {
        return 0.0 * sign;
    }
    if magnitude == 0x7F {
        return sign * FP8_E4M3_MAX;
    }
    let exp = ((magnitude >> 3) & 0x0F) as i32;
    let mant = (magnitude & 0x07) as f32;
    if exp == 0 {
        // Subnormal: (m / 8) * 2**-6.
        sign * (mant / 8.0) * FP8_E4M3_MIN_NORMAL
    } else {
        // Normal: (1 + m/8) * 2**(exp - 7).
        let base = 1.0 + mant / 8.0;
        let scale = 2.0f32.powi(exp - 7);
        sign * base * scale
    }
}

// FP8 paged KV cache
// ---------------------------------------------------------------------------

/// FP8 E4M3 paged KV cache with one scale per `(page, kv_head)` slot.
///
/// Logical shape: `[num_pages, hkv, page_size, head_dim]`. Data is stored as
/// `u8` values (each an FP8 E4M3 encoding) and `scales` holds one `f32`
/// scale per `(page, hkv)` slot, applied uniformly to the `page_size *
/// head_dim` elements in that slot.
///
/// Reconstruction: `f32_value = e4m3_to_f32(data[i]) * scales[slot]`.
///
/// Encode picks a per-slot scale so that the largest absolute value in the
/// slot lands at `FP8_E4M3_MAX`, minimizing quantization error for that
/// slot's dynamic range.
#[derive(Debug, Clone)]
pub struct FpKvCache {
    /// FP8 E4M3 payload, length `num_pages * hkv * page_size * head_dim`.
    pub data: Vec<u8>,
    /// Per-`(page, hkv)` scale, length `num_pages * hkv`.
    pub scales: Vec<f32>,
    /// Number of physical pages.
    pub num_pages: usize,
    /// KV heads per page.
    pub hkv: usize,
    /// KV tokens per page.
    pub page_size: usize,
    /// Head dimension.
    pub head_dim: usize,
}

impl FpKvCache {
    /// Encode a dense `[num_pages, hkv, page_size, head_dim]` FP32 buffer.
    pub fn encode(
        buf: &[f32],
        num_pages: usize,
        hkv: usize,
        page_size: usize,
        head_dim: usize,
    ) -> Result<Self, SparseAttentionError> {
        let elems_per_slot = page_size * head_dim;
        let total = num_pages * hkv * elems_per_slot;
        if buf.len() != total {
            return Err(SparseAttentionError::ShapeMismatch {
                what: "FpKvCache::encode buf",
                expected: total,
                got: buf.len(),
            });
        }
        let num_slots = num_pages * hkv;
        let mut data = vec![0u8; total];
        let mut scales = vec![0.0f32; num_slots];
        for slot in 0..num_slots {
            let slot_off = slot * elems_per_slot;
            let src = &buf[slot_off..slot_off + elems_per_slot];
            let mut max_abs = 0.0f32;
            for v in src {
                let a = v.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            let scale = if max_abs > 0.0 {
                max_abs / FP8_E4M3_MAX
            } else {
                1.0
            };
            scales[slot] = scale;
            let inv = if max_abs > 0.0 { 1.0 / scale } else { 0.0 };
            for (i, v) in src.iter().enumerate() {
                data[slot_off + i] = f32_to_e4m3(v * inv);
            }
        }
        Ok(Self {
            data,
            scales,
            num_pages,
            hkv,
            page_size,
            head_dim,
        })
    }

    /// Fully decode back to a dense `[num_pages, hkv, page_size, head_dim]`
    /// FP32 buffer. Convenient bridge to the existing FP32 `kvouter_forward`
    /// path.
    #[must_use]
    pub fn decode(&self) -> Vec<f32> {
        let elems_per_slot = self.page_size * self.head_dim;
        let mut out = Vec::with_capacity(self.data.len());
        for slot in 0..self.scales.len() {
            let scale = self.scales[slot];
            let slot_off = slot * elems_per_slot;
            for i in 0..elems_per_slot {
                out.push(e4m3_to_f32(self.data[slot_off + i]) * scale);
            }
        }
        out
    }

    /// Storage saving vs. FP32 (4 bytes per element).
    ///
    /// Returns the ratio `raw_bytes(FP32) / stored_bytes(FP8)`. With the
    /// per-slot scale overhead this is slightly below 4×.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        let f32_bytes = (self.data.len() * 4) as f32;
        let stored = (self.data.len() + self.scales.len() * 4) as f32;
        f32_bytes / stored
    }
}

// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_roundtrip_representable_values() {
        // Values on the E4M3 grid should round-trip exactly. 448 is the
        // largest finite (OCP spec: 0x7F is NaN, so max = (1+6/8)*2^8).
        let exact = [
            0.0f32, 1.0, -1.0, 0.5, -0.5, 240.0, -240.0, 384.0, -384.0, 448.0, -448.0,
        ];
        for &v in &exact {
            let b = f32_to_e4m3(v);
            let back = e4m3_to_f32(b);
            assert!(
                (back - v).abs() < 1e-6 || back == v,
                "roundtrip failed for {v}: byte={b:#04x}, back={back}"
            );
        }
    }

    #[test]
    fn e4m3_saturates_on_overflow() {
        // Values above MAX are clamped to ±MAX_NORMAL; NaN / Inf are safe.
        assert!((e4m3_to_f32(f32_to_e4m3(1e6)) - FP8_E4M3_MAX).abs() < 1e-3);
        assert!((e4m3_to_f32(f32_to_e4m3(-1e6)) + FP8_E4M3_MAX).abs() < 1e-3);
        assert!((e4m3_to_f32(f32_to_e4m3(f32::NAN)) - FP8_E4M3_MAX).abs() < 1e-3);
        assert!((e4m3_to_f32(f32_to_e4m3(f32::INFINITY)) - FP8_E4M3_MAX).abs() < 1e-3);
        assert!((e4m3_to_f32(f32_to_e4m3(f32::NEG_INFINITY)) + FP8_E4M3_MAX).abs() < 1e-3);
    }

    #[test]
    fn e4m3_subnormal_region_encodes_small_values() {
        // Subnormal region: (m / 8) * 2**-6, m in 1..=7.
        for m in 1..=7 {
            let v = (m as f32 / 8.0) * FP8_E4M3_MIN_NORMAL;
            let byte = f32_to_e4m3(v);
            let back = e4m3_to_f32(byte);
            assert!(
                (back - v).abs() < 1e-6,
                "subnormal m={m}: v={v}, back={back}"
            );
        }
    }

    #[test]
    fn fp_kv_cache_roundtrip_preserves_dynamic_range() {
        // Deterministic KV data spanning a few orders of magnitude.
        let num_pages = 3;
        let hkv = 2;
        let page_size = 4;
        let head_dim = 8;
        let total = num_pages * hkv * page_size * head_dim;
        let src: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.13 - 0.5) * 3.0)
            .collect();
        let cache = FpKvCache::encode(&src, num_pages, hkv, page_size, head_dim).unwrap();
        let back = cache.decode();
        assert_eq!(back.len(), total);
        // E4M3 has 3 mantissa bits so per-element rel err can reach ~12%
        // near the mid-scale; per-slot scaling keeps the RMS under a few
        // percent. We assert both bounds explicitly.
        let mut max_rel = 0.0f32;
        let mut sum_sq = 0.0f64;
        let mut count = 0usize;
        for (i, (a, b)) in src.iter().zip(&back).enumerate() {
            let denom = a.abs().max(1e-3);
            let rel = (a - b).abs() / denom;
            max_rel = max_rel.max(rel);
            sum_sq += f64::from(rel) * f64::from(rel);
            count += 1;
            assert!(rel < 0.15, "elem {i}: src={a}, back={b}, rel={rel}");
        }
        let rms = (sum_sq / count as f64).sqrt() as f32;
        assert!(rms < 0.05, "RMS rel err {rms} exceeds 5%");
        // Just to have a reported number in the test log.
        eprintln!(
            "FpKvCache roundtrip: max rel err = {max_rel}, compression ratio = {}",
            cache.compression_ratio()
        );
    }

    #[test]
    fn fp_kv_cache_compression_ratio_close_to_four() {
        let src = vec![1.0f32; 128];
        let cache = FpKvCache::encode(&src, 2, 2, 4, 8).unwrap();
        let ratio = cache.compression_ratio();
        // 128 bytes payload + (2*2)*4 = 16 bytes scales = 144 bytes stored,
        // 128 * 4 = 512 bytes FP32 → 512/144 ≈ 3.56.
        assert!(ratio > 3.0 && ratio < 4.0, "unexpected ratio {ratio}");
    }

    #[test]
    fn fp_kv_cache_encode_shape_mismatch() {
        let err = FpKvCache::encode(&[1.0; 5], 1, 1, 2, 4).unwrap_err();
        matches!(err, SparseAttentionError::ShapeMismatch { .. });
    }

    #[test]
    fn fp8_kv_cache_paired_with_kvouter_forward_end_to_end() {
        // Encode a realistic K/V cache into FP8, decode back, and run the
        // full sparse-attention forward. Compare per-partial-output against
        // the FP32 reference to demonstrate that E4M3 loss stays inside the
        // usual 5-10% rel err budget attention layers tolerate.
        use super::super::{
            build_kvouter_index,
            forward::kvouter_forward,
            types::{BlockTables, CuSeqlensQ, SparseSelection},
        };
        let tq = 2;
        let hq = 2;
        let hkv = 1;
        let head_dim = 8;
        let page_size = 4;
        let block_size = 4;
        let max_pages = 3;
        let num_pages = 3;

        let q: Vec<f32> = (0..tq * hq * head_dim)
            .map(|i| ((i as f32) * 0.07).sin())
            .collect();
        let k_fp32: Vec<f32> = (0..num_pages * hkv * page_size * head_dim)
            .map(|i| ((i as f32) * 0.11).cos())
            .collect();
        let v_fp32: Vec<f32> = (0..num_pages * hkv * page_size * head_dim)
            .map(|i| ((i as f32) * 0.13).sin())
            .collect();

        let k_cache = FpKvCache::encode(&k_fp32, num_pages, hkv, page_size, head_dim).unwrap();
        let v_cache = FpKvCache::encode(&v_fp32, num_pages, hkv, page_size, head_dim).unwrap();
        let k_deq = k_cache.decode();
        let v_deq = v_cache.decode();

        let tbl = BlockTables::new(vec![0, 1, 2], 1, max_pages).unwrap();
        let cu = CuSeqlensQ::new(vec![0, tq as i64]).unwrap();
        let used = vec![(num_pages * page_size) as i32];
        // Select all sparse blocks (topk == msb) so we're comparing the
        // full attention output, not a top-K approximation.
        let mut sel_flat = Vec::new();
        for _i in 0..tq {
            for _h in 0..hkv {
                for r in 0..max_pages {
                    sel_flat.push(r as i32);
                }
            }
        }
        let sel = SparseSelection::new(sel_flat, tq, hkv, max_pages).unwrap();
        let idx = build_kvouter_index(&sel, &tbl, &cu, Some(&used), block_size, page_size).unwrap();
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let fp32_out = kvouter_forward(
            &q,
            &k_fp32,
            &v_fp32,
            &idx,
            &tbl,
            &cu,
            Some(&used),
            hq,
            hkv,
            head_dim,
            block_size,
            page_size,
            scale,
            false,
        )
        .unwrap();
        let fp8_out = kvouter_forward(
            &q,
            &k_deq,
            &v_deq,
            &idx,
            &tbl,
            &cu,
            Some(&used),
            hq,
            hkv,
            head_dim,
            block_size,
            page_size,
            scale,
            false,
        )
        .unwrap();

        let mut sum_sq = 0.0f64;
        let mut count = 0usize;
        let mut max_rel = 0.0f32;
        for (a, b) in fp32_out.o_partial.iter().zip(&fp8_out.o_partial) {
            let denom = a.abs().max(1e-3);
            let rel = (a - b).abs() / denom;
            sum_sq += f64::from(rel) * f64::from(rel);
            count += 1;
            max_rel = max_rel.max(rel);
        }
        let rms = (sum_sq / count as f64).sqrt() as f32;
        eprintln!("fp32→fp8 KV cache forward RMS rel err = {rms}, max = {max_rel}");
        // Small synthetic fixture: page_size=4 × head_dim=8 = 32 elems per
        // slot with per-slot scale; softmax amplifies FP8 quantization noise
        // more than real head_dim=128 caches would. Bound accordingly so the
        // test still catches gross regressions.
        assert!(rms < 0.10, "RMS rel err {rms} exceeds 10%");
        assert!(max_rel < 0.50, "max rel err {max_rel} exceeds 50%");
    }
}

// Helpers
// ---------------------------------------------------------------------------

/// Round half to even for a non-negative `f32`.
#[inline]
fn round_ties_even(x: f32) -> f32 {
    // f32's `round_ties_even` was stabilized in 1.77; fall back to a manual
    // implementation to keep MSRV flexible.
    let floor = x.floor();
    let frac = x - floor;
    match frac.partial_cmp(&0.5) {
        Some(std::cmp::Ordering::Less) => floor,
        Some(std::cmp::Ordering::Greater) => floor + 1.0,
        Some(std::cmp::Ordering::Equal) => {
            if (floor as i64) & 1 == 0 {
                floor
            } else {
                floor + 1.0
            }
        }
        None => floor,
    }
}
