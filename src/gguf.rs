//! GGUF v3 format parser with fused quantized matvec.
//!
//! Reads GGUF files (llama.cpp format) and performs inference directly
//! on quantized weights without materializing full FP32 tensors.
//! Supports Q4_K, Q8_0, F16, and F32.

use std::collections::HashMap;

// ─── Constants ──────────────────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x4655_4747;
const GGUF_DEFAULT_ALIGNMENT: usize = 32;
const QK_K: usize = 256;
const QK8_0: usize = 32;

// ─── Half-precision conversion ──────────────────────────────────────────────

#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exponent = ((h >> 10) & 0x1f) as u32;
    let mantissa = (h & 0x3ff) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut e = 0u32;
            let mut m = mantissa;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            let exp = 127 - 15 - e + 1;
            let man = (m & 0x3ff) << 13;
            f32::from_bits((sign << 31) | (exp << 23) | man)
        }
    } else if exponent == 31 {
        f32::from_bits((sign << 31) | (0xff << 23) | (mantissa << 13))
    } else {
        let exp = exponent + 127 - 15;
        let man = mantissa << 13;
        f32::from_bits((sign << 31) | (exp << 23) | man)
    }
}

// ─── Quantization types ────────────────────────────────────────────────────

/// GGML tensor quantization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q8_0,
    Q4_K,
    Q5_K,
    Q6_K,
    Other(u32),
}

impl GgmlType {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            8 => Self::Q8_0,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            other => Self::Other(other),
        }
    }

    /// Byte size per block of quantized data.
    #[must_use]
    pub fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q8_0 => 34,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Other(_) => 0,
        }
    }

    /// Number of elements per block.
    #[must_use]
    pub fn elements_per_block(&self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q8_0 => QK8_0,
            Self::Q4_K | Self::Q5_K | Self::Q6_K => QK_K,
            Self::Other(_) => 1,
        }
    }
}

// ─── GGUF metadata values ──────────────────────────────────────────────────

/// A GGUF metadata value.
#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    /// Get as u32.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::I32(v) => Some(*v as u32),
            Self::U64(v) => Some(*v as u32),
            _ => None,
        }
    }

    /// Get as f32.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Get as string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Get as string array.
    pub fn as_str_array(&self) -> Option<Vec<&str>> {
        match self {
            Self::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr {
                    out.push(v.as_str()?);
                }
                Some(out)
            }
            _ => None,
        }
    }
}

// ─── Tensor info ────────────────────────────────────────────────────────────

/// Information about a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub qtype: GgmlType,
    pub offset: u64,
}

impl TensorInfo {
    /// Total number of elements in this tensor.
    #[must_use]
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }

    /// Size in bytes of the raw quantized data.
    #[must_use]
    pub fn data_size(&self) -> usize {
        let n = self.n_elements();
        let epb = self.qtype.elements_per_block();
        let n_blocks = (n + epb - 1) / epb;
        n_blocks * self.qtype.block_bytes()
    }
}

// ─── Binary reader ──────────────────────────────────────────────────────────

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return None;
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Some(slice)
    }

    fn read_u8(&mut self) -> Option<u8> {
        let b = self.read_bytes(1)?;
        Some(b[0])
    }

    fn read_i8(&mut self) -> Option<i8> {
        Some(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Option<u16> {
        let b = self.read_bytes(2)?;
        Some(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_i16(&mut self) -> Option<i16> {
        let b = self.read_bytes(2)?;
        Some(i16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self) -> Option<u32> {
        let b = self.read_bytes(4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self) -> Option<i32> {
        let b = self.read_bytes(4)?;
        Some(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Option<u64> {
        let b = self.read_bytes(8)?;
        Some(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i64(&mut self) -> Option<i64> {
        let b = self.read_bytes(8)?;
        Some(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_f32(&mut self) -> Option<f32> {
        let b = self.read_bytes(4)?;
        Some(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f64(&mut self) -> Option<f64> {
        let b = self.read_bytes(8)?;
        Some(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_string(&mut self) -> Option<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        Some(String::from_utf8_lossy(bytes).into_owned())
    }

    fn read_bool(&mut self) -> Option<bool> {
        Some(self.read_u8()? != 0)
    }

    fn read_meta_value(&mut self, vtype: u32) -> Option<MetaValue> {
        match vtype {
            0 => Some(MetaValue::U8(self.read_u8()?)),
            1 => Some(MetaValue::I8(self.read_i8()?)),
            2 => Some(MetaValue::U16(self.read_u16()?)),
            3 => Some(MetaValue::I16(self.read_i16()?)),
            4 => Some(MetaValue::U32(self.read_u32()?)),
            5 => Some(MetaValue::I32(self.read_i32()?)),
            6 => Some(MetaValue::F32(self.read_f32()?)),
            7 => Some(MetaValue::Bool(self.read_bool()?)),
            8 => Some(MetaValue::Str(self.read_string()?)),
            9 => {
                let arr_type = self.read_u32()?;
                let arr_len = self.read_u64()? as usize;
                let mut items = Vec::with_capacity(arr_len.min(1_000_000));
                for _ in 0..arr_len {
                    items.push(self.read_meta_value(arr_type)?);
                }
                Some(MetaValue::Array(items))
            }
            10 => Some(MetaValue::U64(self.read_u64()?)),
            11 => Some(MetaValue::I64(self.read_i64()?)),
            12 => Some(MetaValue::F64(self.read_f64()?)),
            _ => None,
        }
    }

    fn align(&mut self, alignment: usize) {
        let rem = self.pos % alignment;
        if rem != 0 {
            self.pos += alignment - rem;
        }
    }
}

// ─── GGUF file ──────────────────────────────────────────────────────────────

/// A parsed GGUF file. Holds metadata and raw data references for
/// zero-copy tensor access.
pub struct GgufFile<'a> {
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: HashMap<String, TensorInfo>,
    tensor_data_start: usize,
    data: &'a [u8],
    _alignment: usize,
}

impl<'a> GgufFile<'a> {
    /// Parse a GGUF file from a byte slice (typically mmap'd).
    pub fn parse(data: &'a [u8]) -> Option<Self> {
        let mut r = Reader::new(data);

        let magic = r.read_u32()?;
        if magic != GGUF_MAGIC {
            return None;
        }

        let version = r.read_u32()?;
        if version < 2 || version > 3 {
            return None;
        }

        let tensor_count = r.read_u64()? as usize;
        let metadata_count = r.read_u64()? as usize;

        // Parse metadata
        let mut metadata = HashMap::with_capacity(metadata_count);
        for _ in 0..metadata_count {
            let key = r.read_string()?;
            let vtype = r.read_u32()?;
            let value = r.read_meta_value(vtype)?;
            metadata.insert(key, value);
        }

        // Parse tensor info
        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.read_string()?;
            let n_dims = r.read_u32()?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(r.read_u64()?);
            }
            let qtype = GgmlType::from_u32(r.read_u32()?);
            let offset = r.read_u64()?;
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    n_dims,
                    dims,
                    qtype,
                    offset,
                },
            );
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT as u32) as usize;

        // Align to data section
        r.align(alignment);
        let tensor_data_start = r.pos;

        Some(Self {
            metadata,
            tensors,
            tensor_data_start,
            data,
            _alignment: alignment,
        })
    }

    /// Get metadata value by key.
    #[must_use]
    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    /// Get metadata string.
    #[must_use]
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.meta(key)?.as_str()
    }

    /// Get metadata u32.
    #[must_use]
    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.meta(key)?.as_u32()
    }

    /// Get metadata f32.
    #[must_use]
    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.meta(key)?.as_f32()
    }

    /// Get raw quantized data for a tensor.
    #[must_use]
    pub fn tensor_data(&self, name: &str) -> Option<&'a [u8]> {
        let info = self.tensors.get(name)?;
        let start = self.tensor_data_start + info.offset as usize;
        let size = info.data_size();
        if start + size > self.data.len() {
            return None;
        }
        Some(&self.data[start..start + size])
    }

    /// Get tensor info by name.
    #[must_use]
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Dequantize a tensor to f32 (allocates). Use fused matvec for inference.
    pub fn tensor_to_f32(&self, name: &str) -> Option<Vec<f32>> {
        let info = self.tensors.get(name)?;
        let data = self.tensor_data(name)?;
        let n_elements = info.n_elements();
        let mut out = vec![0.0f32; n_elements];

        match info.qtype {
            GgmlType::F32 => {
                for i in 0..n_elements {
                    let off = i * 4;
                    out[i] = f32::from_le_bytes([
                        data[off],
                        data[off + 1],
                        data[off + 2],
                        data[off + 3],
                    ]);
                }
            }
            GgmlType::F16 => {
                for i in 0..n_elements {
                    let off = i * 2;
                    out[i] = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
                }
            }
            GgmlType::Q8_0 => dequantize_q8_0(data, &mut out),
            GgmlType::Q4_K => dequantize_q4_k(data, &mut out),
            GgmlType::Q6_K => dequantize_q6_k(data, &mut out),
            _ => return None,
        }

        Some(out)
    }
}

// ─── Q4_K dequantization ────────────────────────────────────────────────────

#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

fn dequantize_q4_k(data: &[u8], out: &mut [f32]) {
    let block_bytes = 144;
    let n_blocks = data.len() / block_bytes;
    let mut out_idx = 0;

    for i in 0..n_blocks {
        let block = &data[i * block_bytes..(i + 1) * block_bytes];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qs = &block[16..144];

        let mut is = 0usize;
        let mut q_offset = 0usize;

        for _ in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * f32::from(sc1);
            let m1f = dmin * f32::from(m1);
            let d2 = d * f32::from(sc2);
            let m2f = dmin * f32::from(m2);

            for l in 0..32 {
                out[out_idx] = d1 * f32::from(qs[q_offset + l] & 0xF) - m1f;
                out_idx += 1;
            }
            for l in 0..32 {
                out[out_idx] = d2 * f32::from(qs[q_offset + l] >> 4) - m2f;
                out_idx += 1;
            }

            q_offset += 32;
            is += 2;
        }
    }
}

fn dequantize_q8_0(data: &[u8], out: &mut [f32]) {
    let block_bytes = 34;
    let n_blocks = data.len() / block_bytes;
    let mut out_idx = 0;

    for i in 0..n_blocks {
        let off = i * block_bytes;
        let d = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));

        for l in 0..QK8_0 {
            out[out_idx] = d * f32::from(data[off + 2 + l] as i8);
            out_idx += 1;
        }
    }
}

// ─── Q6_K dequantization ────────────────────────────────────────────────────

fn dequantize_q6_k(data: &[u8], out: &mut [f32]) {
    let block_bytes = 210;
    let n_blocks = data.len() / block_bytes;
    let mut out_idx = 0;

    for i in 0..n_blocks {
        let block = &data[i * block_bytes..(i + 1) * block_bytes];
        // Layout: ql[128] + qh[64] + scales[16] + d[2]
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

        let mut ql_off = 0usize;
        let mut qh_off = 0usize;

        for n in (0..QK_K).step_by(128) {
            let is = n / 16;
            for l in 0..32 {
                let q1 = ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i8 - 32;
                let q2 =
                    ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8
                        - 32;
                let q3 =
                    ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 =
                    ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 - 32;

                out[out_idx + l] = d * f32::from(scales[is] as i8) * f32::from(q1);
                out[out_idx + l + 32] = d * f32::from(scales[is + 2] as i8) * f32::from(q2);
                out[out_idx + l + 64] = d * f32::from(scales[is + 4] as i8) * f32::from(q3);
                out[out_idx + l + 96] = d * f32::from(scales[is + 6] as i8) * f32::from(q4);
            }
            out_idx += 128;
            ql_off += 64;
            qh_off += 32;
        }
    }
}

// ─── Q8_K quantization (matches llama.cpp) ──────────────────────────────────

/// Q8_K block: intermediate quantization format for input vectors.
/// Matches llama.cpp's `block_q8_K` exactly.
pub struct BlockQ8K {
    pub d: f32,              // scale factor (f32, not f16)
    pub qs: [i8; QK_K],     // 256 quantized values
    pub bsums: [i16; 16],   // pre-computed sums of groups of 16
}

/// Round to nearest integer using the same algorithm as llama.cpp's `nearest_int`.
/// Uses IEEE 754 magic number trick for round-to-nearest-even.
#[inline]
fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12_582_912.0f32; // 0x4B400000
    let i = val.to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}

/// Quantize a row of f32 values to Q8_K blocks.
/// Matches llama.cpp's `quantize_row_q8_K_ref`.
pub fn quantize_row_q8_k(input: &[f32]) -> Vec<BlockQ8K> {
    let nb = input.len() / QK_K;
    let mut blocks = Vec::with_capacity(nb);

    for i in 0..nb {
        let x = &input[i * QK_K..(i + 1) * QK_K];

        // Find max absolute value
        let mut amax = 0.0f32;
        let mut max_val = 0.0f32;
        for &v in x {
            let av = v.abs();
            if av > amax {
                amax = av;
                max_val = v;
            }
        }

        let mut block = BlockQ8K {
            d: 0.0,
            qs: [0i8; QK_K],
            bsums: [0i16; 16],
        };

        if amax == 0.0 {
            blocks.push(block);
            continue;
        }

        let iscale = -127.0f32 / max_val;

        for j in 0..QK_K {
            let v = nearest_int(iscale * x[j]);
            block.qs[j] = v.min(127) as i8;
        }

        // Pre-compute group-of-16 sums
        for j in 0..16 {
            let mut sum = 0i32;
            for ii in 0..16 {
                sum += block.qs[j * 16 + ii] as i32;
            }
            block.bsums[j] = sum as i16;
        }

        block.d = 1.0 / iscale;
        blocks.push(block);
    }

    blocks
}

// ─── Fused quantized matvec (llama.cpp compatible) ──────────────────────────

// ─── NEON SIMD dot products (aarch64 / Apple Silicon) ───────────────────────

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
mod neon_dot {
    use super::*;
    use std::arch::aarch64::*;

    /// Horizontal sum of int32x4_t → i32 (stable intrinsics only).
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn hsum_s32(v: int32x4_t) -> i32 {
        let low = vget_low_s32(v);
        let high = vget_high_s32(v);
        let sum2 = vadd_s32(low, high);
        let pair = vpadd_s32(sum2, sum2);
        vget_lane_s32::<0>(pair)
    }

    /// NEON dot product of 32 i8 values using vmull_s8 (stable).
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn dot_32(a: *const i8, b: *const i8) -> i32 {
        let a0 = vld1q_s8(a);
        let a1 = vld1q_s8(a.add(16));
        let b0 = vld1q_s8(b);
        let b1 = vld1q_s8(b.add(16));

        // 4 widening multiplies: 8×i8×i8 → 8×i16 each
        let p0 = vmull_s8(vget_low_s8(a0), vget_low_s8(b0));
        let p1 = vmull_s8(vget_high_s8(a0), vget_high_s8(b0));
        let p2 = vmull_s8(vget_low_s8(a1), vget_low_s8(b1));
        let p3 = vmull_s8(vget_high_s8(a1), vget_high_s8(b1));

        // Pairwise add i16 → i32
        let s01 = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
        let s23 = vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3));
        hsum_s32(vaddq_s32(s01, s23))
    }

    /// NEON dot product of 16 i8 values.
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn dot_16(a: *const i8, b: *const i8) -> i32 {
        let a0 = vld1q_s8(a);
        let b0 = vld1q_s8(b);

        let p0 = vmull_s8(vget_low_s8(a0), vget_low_s8(b0));
        let p1 = vmull_s8(vget_high_s8(a0), vget_high_s8(b0));

        hsum_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)))
    }

    /// Q4_K × Q8_K dot product using NEON.
    #[target_feature(enable = "neon")]
    pub unsafe fn q4k_q8k_dot(q4k_block: &[u8], q8k: &BlockQ8K) -> f32 {
        let d = f16_to_f32(u16::from_le_bytes([q4k_block[0], q4k_block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([q4k_block[2], q4k_block[3]]));
        let q4 = &q4k_block[16..144];

        let (scales, mins) = unpack_scales_mins_q4k(&q4k_block[4..16]);

        // Mins correction using bsums
        let mut sumi = 0i32;
        for j in 0..16 {
            sumi += q8k.bsums[j] as i32 * mins[j / 2] as i32;
        }

        // Unpack nibbles and dot product with NEON
        let mask_lo = vdupq_n_u8(0x0F);
        let mut total = 0i32;
        let q8 = &q8k.qs;

        // Process 4 groups of 32 q4 bytes → 8 sub-blocks of 32 elements
        // But for NEON dot, we unpack to aux8 first, then use dot_32
        let mut aux8 = [0i8; QK_K];
        let mut a_off = 0usize;
        let mut q4_off = 0usize;

        // Unpack using NEON: extract low/high nibbles
        for _ in 0..4 {
            let q4_0 = vld1q_u8(q4.as_ptr().add(q4_off));
            let q4_1 = vld1q_u8(q4.as_ptr().add(q4_off + 16));

            let lo_0 = vreinterpretq_s8_u8(vandq_u8(q4_0, mask_lo));
            let lo_1 = vreinterpretq_s8_u8(vandq_u8(q4_1, mask_lo));
            let hi_0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q4_0));
            let hi_1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q4_1));

            vst1q_s8(aux8.as_mut_ptr().add(a_off), lo_0);
            vst1q_s8(aux8.as_mut_ptr().add(a_off + 16), lo_1);
            vst1q_s8(aux8.as_mut_ptr().add(a_off + 32), hi_0);
            vst1q_s8(aux8.as_mut_ptr().add(a_off + 48), hi_1);

            a_off += 64;
            q4_off += 32;
        }

        // 8 sub-blocks of 32 elements each
        for is in 0..8 {
            let off = is * 32;
            let dot = dot_32(aux8.as_ptr().add(off), q8.as_ptr().add(off));
            total += scales[is] as i32 * dot;
        }

        d * q8k.d * total as f32 - dmin * q8k.d * sumi as f32
    }

    /// Q6_K × Q8_K dot product using NEON.
    #[target_feature(enable = "neon")]
    pub unsafe fn q6k_q8k_dot(q6k_block: &[u8], q8k: &BlockQ8K) -> f32 {
        let ql = &q6k_block[0..128];
        let qh = &q6k_block[128..192];
        let scales = &q6k_block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([q6k_block[208], q6k_block[209]]));

        // Reconstruct 6-bit signed quants (scalar — complex bit packing)
        let mut aux8 = [0i8; QK_K];
        let mut a_off = 0usize;
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        for _ in 0..2 {
            for l in 0..32 {
                aux8[a_off + l] =
                    ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i8 - 32;
                aux8[a_off + l + 32] =
                    ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8 - 32;
                aux8[a_off + l + 64] =
                    ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 - 32;
                aux8[a_off + l + 96] =
                    ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 - 32;
            }
            a_off += 128;
            ql_off += 64;
            qh_off += 32;
        }

        // NEON dot product: 16 sub-blocks of 16 elements each
        let q8 = &q8k.qs;
        let mut total = 0i32;
        for is in 0..16 {
            let scale = scales[is] as i8 as i32;
            let off = is * 16;
            let dot = dot_16(aux8.as_ptr().add(off), q8.as_ptr().add(off));
            total += scale * dot;
        }

        d * q8k.d * total as f32
    }

    #[inline]
    fn unpack_scales_mins_q4k(scale_bytes: &[u8]) -> ([u8; 8], [u8; 8]) {
        const KMASK1: u32 = 0x3f3f_3f3f;
        const KMASK2: u32 = 0x0f0f_0f0f;
        const KMASK3: u32 = 0x0303_0303;

        let mut utmp = [0u32; 4];
        utmp[0] = u32::from_le_bytes([scale_bytes[0], scale_bytes[1], scale_bytes[2], scale_bytes[3]]);
        utmp[1] = u32::from_le_bytes([scale_bytes[4], scale_bytes[5], scale_bytes[6], scale_bytes[7]]);
        utmp[2] = u32::from_le_bytes([scale_bytes[8], scale_bytes[9], scale_bytes[10], scale_bytes[11]]);

        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        let s0 = utmp[0].to_le_bytes();
        let s1 = utmp[1].to_le_bytes();
        let m0 = utmp[2].to_le_bytes();
        let m1 = utmp[3].to_le_bytes();
        (
            [s0[0], s0[1], s0[2], s0[3], s1[0], s1[1], s1[2], s1[3]],
            [m0[0], m0[1], m0[2], m0[3], m1[0], m1[1], m1[2], m1[3]],
        )
    }
}

// ─── Scalar dot product fallback ────────────────────────────────────────────

/// Scalar Q4_K × Q8_K dot product (fallback for non-NEON platforms).
#[inline]
#[allow(dead_code)]
fn q4k_q8k_dot_scalar(q4k_block: &[u8], q8k: &BlockQ8K) -> f32 {
    const KMASK1: u32 = 0x3f3f_3f3f;
    const KMASK2: u32 = 0x0f0f_0f0f;
    const KMASK3: u32 = 0x0303_0303;

    let d = f16_to_f32(u16::from_le_bytes([q4k_block[0], q4k_block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([q4k_block[2], q4k_block[3]]));
    let q4 = &q4k_block[16..144];

    let mut aux8 = [0i8; QK_K];
    let mut a_off = 0usize;
    let mut q4_off = 0usize;
    for _ in 0..4 {
        for l in 0..32 { aux8[a_off + l] = (q4[q4_off + l] & 0xF) as i8; }
        a_off += 32;
        for l in 0..32 { aux8[a_off + l] = (q4[q4_off + l] >> 4) as i8; }
        a_off += 32;
        q4_off += 32;
    }

    let mut utmp = [0u32; 4];
    let sb = &q4k_block[4..16];
    utmp[0] = u32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]]);
    utmp[1] = u32::from_le_bytes([sb[4], sb[5], sb[6], sb[7]]);
    utmp[2] = u32::from_le_bytes([sb[8], sb[9], sb[10], sb[11]]);
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;

    let s0 = utmp[0].to_le_bytes(); let s1 = utmp[1].to_le_bytes();
    let m0 = utmp[2].to_le_bytes(); let m1 = utmp[3].to_le_bytes();
    let scales = [s0[0],s0[1],s0[2],s0[3],s1[0],s1[1],s1[2],s1[3]];
    let mins = [m0[0],m0[1],m0[2],m0[3],m1[0],m1[1],m1[2],m1[3]];

    let mut sumi = 0i32;
    for j in 0..16 { sumi += q8k.bsums[j] as i32 * mins[j / 2] as i32; }

    let mut aux32 = [0i32; 8];
    let mut a_idx = 0usize;
    let mut q8_idx = 0usize;
    for is in 0..8 {
        let scale = scales[is] as i32;
        for _ in 0..4 {
            for l in 0..8 {
                aux32[l] += scale * (q8k.qs[q8_idx + l] as i32 * aux8[a_idx + l] as i32);
            }
            q8_idx += 8; a_idx += 8;
        }
    }

    let d_all = d * q8k.d;
    let dmin_all = dmin * q8k.d;
    let mut sumf = 0.0f32;
    for l in 0..8 { sumf += d_all * aux32[l] as f32; }
    sumf -= dmin_all * sumi as f32;
    sumf
}

/// Scalar Q6_K × Q8_K dot product (fallback for non-NEON platforms).
#[inline]
#[allow(dead_code)]
fn q6k_q8k_dot_scalar(q6k_block: &[u8], q8k: &BlockQ8K) -> f32 {
    let ql = &q6k_block[0..128];
    let qh = &q6k_block[128..192];
    let scales = &q6k_block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([q6k_block[208], q6k_block[209]]));

    let mut aux8 = [0i8; QK_K];
    let mut a_off = 0usize;
    let mut ql_off = 0usize;
    let mut qh_off = 0usize;
    for _ in 0..2 {
        for l in 0..32 {
            aux8[a_off + l] =
                ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i8 - 32;
            aux8[a_off + l + 32] =
                ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8 - 32;
            aux8[a_off + l + 64] =
                ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 - 32;
            aux8[a_off + l + 96] =
                ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 - 32;
        }
        a_off += 128; ql_off += 64; qh_off += 32;
    }

    let mut aux32 = [0i32; 8];
    let mut a_idx = 0usize;
    let mut q8_idx = 0usize;
    for is in 0..16 {
        let scale = scales[is] as i8 as i32;
        for _ in 0..2 {
            for l in 0..8 {
                aux32[l] += scale * (q8k.qs[q8_idx + l] as i32 * aux8[a_idx + l] as i32);
            }
            q8_idx += 8; a_idx += 8;
        }
    }

    let d_all = d * q8k.d;
    let mut sumf = 0.0f32;
    for l in 0..8 { sumf += d_all * aux32[l] as f32; }
    sumf
}

// ─── Dispatch: NEON or scalar ───────────────────────────────────────────────

/// Q4_K × Q8_K dot product. Auto-vectorizes well with -C target-cpu=native.
#[inline]
fn q4k_q8k_dot(q4k_block: &[u8], q8k: &BlockQ8K) -> f32 {
    const KMASK1: u32 = 0x3f3f_3f3f;
    const KMASK2: u32 = 0x0f0f_0f0f;
    const KMASK3: u32 = 0x0303_0303;

    let d = f16_to_f32(u16::from_le_bytes([q4k_block[0], q4k_block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([q4k_block[2], q4k_block[3]]));
    let q4 = &q4k_block[16..144];

    // Unpack nibbles into flat array for vectorizable loops
    let mut aux8 = [0u8; QK_K];
    for g in 0..4 {
        for l in 0..32 {
            aux8[g * 64 + l] = q4[g * 32 + l] & 0xF;
            aux8[g * 64 + 32 + l] = q4[g * 32 + l] >> 4;
        }
    }

    // Unpack scales and mins
    let mut utmp = [0u32; 4];
    let sb = &q4k_block[4..16];
    utmp[0] = u32::from_le_bytes([sb[0], sb[1], sb[2], sb[3]]);
    utmp[1] = u32::from_le_bytes([sb[4], sb[5], sb[6], sb[7]]);
    utmp[2] = u32::from_le_bytes([sb[8], sb[9], sb[10], sb[11]]);
    utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
    let uaux = utmp[1] & KMASK1;
    utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
    utmp[2] = uaux;
    utmp[0] &= KMASK1;
    let s0 = utmp[0].to_le_bytes(); let s1 = utmp[1].to_le_bytes();
    let m0 = utmp[2].to_le_bytes(); let m1 = utmp[3].to_le_bytes();
    let scales = [s0[0],s0[1],s0[2],s0[3],s1[0],s1[1],s1[2],s1[3]];
    let mins = [m0[0],m0[1],m0[2],m0[3],m1[0],m1[1],m1[2],m1[3]];

    // Mins correction
    let mut sumi = 0i32;
    for j in 0..16 { sumi += q8k.bsums[j] as i32 * mins[j / 2] as i32; }

    // 8 sub-blocks of 32 elements — simple reduction loop for auto-vectorization
    let mut total = 0i32;
    for is in 0..8 {
        let off = is * 32;
        let mut dot = 0i32;
        for l in 0..32 {
            dot += aux8[off + l] as i32 * q8k.qs[off + l] as i32;
        }
        total += scales[is] as i32 * dot;
    }

    d * q8k.d * total as f32 - dmin * q8k.d * sumi as f32
}

/// Q6_K × Q8_K dot product. Auto-vectorizes well with -C target-cpu=native.
#[inline]
fn q6k_q8k_dot(q6k_block: &[u8], q8k: &BlockQ8K) -> f32 {
    let ql = &q6k_block[0..128];
    let qh = &q6k_block[128..192];
    let scales = &q6k_block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([q6k_block[208], q6k_block[209]]));

    // Reconstruct 6-bit signed quants
    let mut aux8 = [0i8; QK_K];
    let mut a_off = 0usize;
    let mut ql_off = 0usize;
    let mut qh_off = 0usize;
    for _ in 0..2 {
        for l in 0..32 {
            aux8[a_off + l] =
                ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i8 - 32;
            aux8[a_off + l + 32] =
                ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8 - 32;
            aux8[a_off + l + 64] =
                ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 - 32;
            aux8[a_off + l + 96] =
                ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 - 32;
        }
        a_off += 128; ql_off += 64; qh_off += 32;
    }

    // 16 sub-blocks of 16 — simple reduction for auto-vectorization
    let mut total = 0i32;
    for is in 0..16 {
        let scale = scales[is] as i8 as i32;
        let off = is * 16;
        let mut dot = 0i32;
        for l in 0..16 {
            dot += aux8[off + l] as i32 * q8k.qs[off + l] as i32;
        }
        total += scale * dot;
    }

    d * q8k.d * total as f32
}

// ─── Fused quantized matvec ─────────────────────────────────────────────────

/// Q4_K matvec: quantizes input to Q8_K, then uses integer dot product.
pub fn q4k_matvec(input: &[f32], data: &[u8], rows: usize, cols: usize, output: &mut [f32]) {
    let q8_blocks = quantize_row_q8_k(input);
    q4k_matvec_preq(data, rows, cols, &q8_blocks, output);
}

/// Q4_K matvec with pre-quantized Q8_K input (avoids redundant quantization).
pub fn q4k_matvec_preq(data: &[u8], _rows: usize, cols: usize, q8_blocks: &[BlockQ8K], output: &mut [f32]) {
    let blocks_per_row = cols / QK_K;
    let block_bytes = 144;
    let row_bytes = blocks_per_row * block_bytes;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        output.par_iter_mut().enumerate().for_each(|(row, out)| {
            let row_data = &data[row * row_bytes..(row + 1) * row_bytes];
            let mut sumf = 0.0f32;
            for bi in 0..blocks_per_row {
                sumf += q4k_q8k_dot(&row_data[bi * block_bytes..(bi + 1) * block_bytes], &q8_blocks[bi]);
            }
            *out = sumf;
        });
        return;
    }

    #[cfg(not(feature = "parallel"))]
    for row in 0..rows {
        let row_data = &data[row * row_bytes..(row + 1) * row_bytes];
        let mut sumf = 0.0f32;
        for bi in 0..blocks_per_row {
            sumf += q4k_q8k_dot(&row_data[bi * block_bytes..(bi + 1) * block_bytes], &q8_blocks[bi]);
        }
        output[row] = sumf;
    }
}

/// Fused Q8_0 dequantize + matrix-vector multiply.
pub fn q8_0_matvec(input: &[f32], data: &[u8], rows: usize, cols: usize, output: &mut [f32]) {
    let blocks_per_row = cols / QK8_0;
    let block_bytes = 34;
    let row_bytes = blocks_per_row * block_bytes;

    for row in 0..rows {
        let mut acc = 0.0f32;
        let row_data = &data[row * row_bytes..(row + 1) * row_bytes];

        for bi in 0..blocks_per_row {
            let off = bi * block_bytes;
            let d = f16_to_f32(u16::from_le_bytes([row_data[off], row_data[off + 1]]));
            let col_base = bi * QK8_0;

            for l in 0..QK8_0 {
                let w = d * f32::from(row_data[off + 2 + l] as i8);
                acc += w * input[col_base + l];
            }
        }

        output[row] = acc;
    }
}

/// Fused F16 matvec.
pub fn f16_matvec(input: &[f32], data: &[u8], rows: usize, cols: usize, output: &mut [f32]) {
    for row in 0..rows {
        let mut acc = 0.0f32;
        let row_start = row * cols * 2;
        for col in 0..cols {
            let off = row_start + col * 2;
            let w = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
            acc += w * input[col];
        }
        output[row] = acc;
    }
}

/// Fused F32 matvec.
pub fn f32_matvec(input: &[f32], data: &[u8], rows: usize, cols: usize, output: &mut [f32]) {
    for row in 0..rows {
        let mut acc = 0.0f32;
        let row_start = row * cols * 4;
        for col in 0..cols {
            let off = row_start + col * 4;
            let w = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            acc += w * input[col];
        }
        output[row] = acc;
    }
}

/// Q6_K matvec: quantizes input to Q8_K, then uses integer dot product.
pub fn q6k_matvec(input: &[f32], data: &[u8], rows: usize, cols: usize, output: &mut [f32]) {
    let q8_blocks = quantize_row_q8_k(input);
    q6k_matvec_preq(data, rows, cols, &q8_blocks, output);
}

/// Q6_K matvec with pre-quantized Q8_K input.
pub fn q6k_matvec_preq(data: &[u8], _rows: usize, cols: usize, q8_blocks: &[BlockQ8K], output: &mut [f32]) {
    let blocks_per_row = cols / QK_K;
    let block_bytes = 210;
    let row_bytes = blocks_per_row * block_bytes;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        output.par_iter_mut().enumerate().for_each(|(row, out)| {
            let row_data = &data[row * row_bytes..(row + 1) * row_bytes];
            let mut sumf = 0.0f32;
            for bi in 0..blocks_per_row {
                sumf += q6k_q8k_dot(&row_data[bi * block_bytes..(bi + 1) * block_bytes], &q8_blocks[bi]);
            }
            *out = sumf;
        });
        return;
    }

    #[cfg(not(feature = "parallel"))]
    for row in 0..rows {
        let row_data = &data[row * row_bytes..(row + 1) * row_bytes];
        let mut sumf = 0.0f32;
        for bi in 0..blocks_per_row {
            sumf += q6k_q8k_dot(&row_data[bi * block_bytes..(bi + 1) * block_bytes], &q8_blocks[bi]);
        }
        output[row] = sumf;
    }
}

/// Dispatch matvec based on quantization type.
pub fn quantized_matvec(
    input: &[f32],
    data: &[u8],
    qtype: GgmlType,
    rows: usize,
    cols: usize,
    output: &mut [f32],
) {
    match qtype {
        GgmlType::Q4_K => q4k_matvec(input, data, rows, cols, output),
        GgmlType::Q6_K => q6k_matvec(input, data, rows, cols, output),
        GgmlType::Q8_0 => q8_0_matvec(input, data, rows, cols, output),
        GgmlType::F16 => f16_matvec(input, data, rows, cols, output),
        GgmlType::F32 => f32_matvec(input, data, rows, cols, output),
        _ => panic!("unsupported quantization type: {qtype:?}"),
    }
}

/// Matvec with pre-quantized Q8_K input. For Q4_K/Q6_K weight types,
/// avoids redundant quantization when the same input is used multiple times.
pub fn quantized_matvec_preq(
    data: &[u8],
    qtype: GgmlType,
    rows: usize,
    cols: usize,
    q8_blocks: &[BlockQ8K],
    output: &mut [f32],
) {
    match qtype {
        GgmlType::Q4_K => q4k_matvec_preq(data, rows, cols, q8_blocks, output),
        GgmlType::Q6_K => q6k_matvec_preq(data, rows, cols, q8_blocks, output),
        _ => panic!("quantized_matvec_preq only supports Q4_K/Q6_K, got {qtype:?}"),
    }
}

// ─── Tokenizer from GGUF metadata ──────────────────────────────────────────

// ─── GPT-2 byte encoding ────────────────────────────────────────────────────

/// Build GPT-2 byte-to-unicode table.
/// Maps each byte 0-255 to a printable Unicode character.
/// Printable bytes (33-126, 161-172, 174-255) map to themselves.
/// Non-printable bytes map to Unicode 256+ in order.
fn gpt2_byte_to_char() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 256u32;
    for b in 0u32..256 {
        let is_printable = matches!(b, 33..=126 | 161..=172 | 174..=255);
        let ch = if is_printable {
            b
        } else {
            let c = n;
            n += 1;
            c
        };
        table[b as usize] = char::from_u32(ch).unwrap();
    }
    table
}

/// Build reverse mapping: GPT-2 unicode char → original byte value.
fn gpt2_char_to_byte() -> HashMap<char, u8> {
    let table = gpt2_byte_to_char();
    let mut map = HashMap::with_capacity(256);
    for (b, &ch) in table.iter().enumerate() {
        map.insert(ch, b as u8);
    }
    map
}

// ─── Tokenizer ──────────────────────────────────────────────────────────────

/// BPE tokenizer loaded from GGUF metadata.
/// Uses GPT-2 byte encoding for Llama-3 compatibility.
pub struct GgufTokenizer {
    tokens: Vec<Vec<u8>>,
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    token_to_id: HashMap<Vec<u8>, u32>,
    /// Special tokens (e.g. `<|begin_of_text|>`) sorted by length desc.
    special_tokens: Vec<(String, u32)>,
    /// GPT-2 byte→char mapping (for encoding input text).
    byte_encoder: [char; 256],
    /// GPT-2 char→byte mapping (for decoding tokens to text).
    byte_decoder: HashMap<char, u8>,
    pub bos_id: u32,
    pub eos_id: u32,
}

impl GgufTokenizer {
    /// Load tokenizer from GGUF metadata.
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Option<Self> {
        let tokens_meta = gguf.meta("tokenizer.ggml.tokens")?;
        let token_strs = tokens_meta.as_str_array()?;

        let mut tokens = Vec::with_capacity(token_strs.len());
        let mut token_to_id = HashMap::with_capacity(token_strs.len());
        let mut special_tokens = Vec::new();

        for (i, t) in token_strs.iter().enumerate() {
            let bytes = t.as_bytes().to_vec();
            token_to_id.insert(bytes.clone(), i as u32);
            tokens.push(bytes);

            // Detect special tokens: <|...|> pattern
            if t.starts_with("<|") && t.ends_with("|>") {
                special_tokens.push((t.to_string(), i as u32));
            }
        }

        // Sort special tokens by length descending for greedy matching
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let merges = if let Some(merges_meta) = gguf.meta("tokenizer.ggml.merges") {
            if let Some(merge_strs) = merges_meta.as_str_array() {
                merge_strs
                    .iter()
                    .filter_map(|s| {
                        let parts: Vec<&str> = s.splitn(2, ' ').collect();
                        if parts.len() == 2 {
                            Some((parts[0].as_bytes().to_vec(), parts[1].as_bytes().to_vec()))
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let bos_id = gguf.meta_u32("tokenizer.ggml.bos_token_id").unwrap_or(1);
        let eos_id = gguf.meta_u32("tokenizer.ggml.eos_token_id").unwrap_or(2);

        Some(Self {
            tokens,
            merges,
            token_to_id,
            special_tokens,
            byte_encoder: gpt2_byte_to_char(),
            byte_decoder: gpt2_char_to_byte(),
            bos_id,
            eos_id,
        })
    }

    /// Encode text to token IDs.
    /// Handles special tokens as atomic units, then applies GPT-2
    /// byte-level BPE to remaining text segments.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            // Try to match a special token (greedy, longest first)
            let mut matched = false;
            for (tok_str, tok_id) in &self.special_tokens {
                if remaining.starts_with(tok_str.as_str()) {
                    result.push(*tok_id);
                    remaining = &remaining[tok_str.len()..];
                    matched = true;
                    break;
                }
            }
            if matched {
                continue;
            }

            // Find next special token boundary
            let mut next_boundary = remaining.len();
            for (tok_str, _) in &self.special_tokens {
                if let Some(pos) = remaining.find(tok_str.as_str()) {
                    if pos > 0 && pos < next_boundary {
                        next_boundary = pos;
                    }
                }
            }

            // BPE encode the text chunk
            let chunk = &remaining[..next_boundary];
            result.extend(self.bpe_encode_chunk(chunk));
            remaining = &remaining[next_boundary..];
        }

        result
    }

    /// BPE encode a text chunk using GPT-2 byte encoding.
    /// Input bytes are mapped to GPT-2 unicode characters before BPE.
    fn bpe_encode_chunk(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Convert each input byte to its GPT-2 unicode character (as UTF-8 bytes)
        let mut pieces: Vec<Vec<u8>> = text
            .bytes()
            .map(|b| {
                let ch = self.byte_encoder[b as usize];
                let mut buf = [0u8; 4];
                ch.encode_utf8(&mut buf).as_bytes().to_vec()
            })
            .collect();

        // Apply BPE merges in priority order (GPT-2 encoded pieces)
        for (left, right) in &self.merges {
            let mut i = 0;
            while i + 1 < pieces.len() {
                if pieces[i] == *left && pieces[i + 1] == *right {
                    let mut merged = pieces[i].clone();
                    merged.extend_from_slice(&pieces[i + 1]);
                    pieces[i] = merged;
                    pieces.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        // Map pieces to token IDs with fallback
        let mut ids = Vec::new();
        for p in &pieces {
            if let Some(&id) = self.token_to_id.get(p) {
                ids.push(id);
            } else if let Ok(s) = std::str::from_utf8(p) {
                // Fallback: try each GPT-2 char as individual token
                for ch in s.chars() {
                    let mut buf = [0u8; 4];
                    let ch_bytes = ch.encode_utf8(&mut buf).as_bytes().to_vec();
                    if let Some(&id) = self.token_to_id.get(&ch_bytes) {
                        ids.push(id);
                    }
                }
            }
        }
        ids
    }

    /// Decode token IDs to text.
    /// Converts GPT-2 unicode characters back to raw bytes.
    /// Skips control tokens and handles `<0xNN>` byte tokens.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(token) = self.tokens.get(id as usize) {
                if let Ok(s) = std::str::from_utf8(token) {
                    // Skip control/special tokens in output
                    if s.starts_with("<|") && s.ends_with("|>") {
                        continue;
                    }
                    // Handle byte tokens <0xNN>
                    if s.starts_with("<0x") && s.ends_with('>') && s.len() == 6 {
                        if let Ok(byte_val) = u8::from_str_radix(&s[3..5], 16) {
                            bytes.push(byte_val);
                            continue;
                        }
                    }
                    // Decode GPT-2 unicode chars → raw bytes
                    for ch in s.chars() {
                        if let Some(&b) = self.byte_decoder.get(&ch) {
                            bytes.push(b);
                        } else {
                            // Unknown char, keep as UTF-8
                            let mut buf = [0u8; 4];
                            bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                    }
                } else {
                    bytes.extend_from_slice(token);
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
}

// ─── Ternary quantization ────────────────────────────────────────────────────

/// Ternary-quantized weight row: {-1, 0, +1} with bitmask packing.
/// 2 bits per weight: pos_mask (positive) + neg_mask (negative).
/// Zero weights have neither bit set.
pub struct TernaryRow {
    pub pos_mask: Vec<u8>,
    pub neg_mask: Vec<u8>,
    pub scale: f32,
    pub num_cols: usize,
}

/// Ternary-quantized weight matrix.
pub struct TernaryMatrix {
    pub rows: Vec<TernaryRow>,
    pub num_rows: usize,
    pub num_cols: usize,
}

impl TernaryRow {
    /// Ternarize a f32 weight row. threshold_ratio controls sparsity
    /// (0.7 = weights below 0.7 × mean(|w|) become zero).
    pub fn from_f32(weights: &[f32], threshold_ratio: f32) -> Self {
        let num_cols = weights.len();
        let num_bytes = (num_cols + 7) / 8;

        let mean_abs: f32 = weights.iter().map(|w| w.abs()).sum::<f32>() / num_cols as f32;
        let threshold = threshold_ratio * mean_abs;

        let mut pos_mask = vec![0u8; num_bytes];
        let mut neg_mask = vec![0u8; num_bytes];
        let mut abs_sum = 0.0f32;
        let mut nonzero_count = 0usize;

        for (i, &w) in weights.iter().enumerate() {
            if w.abs() > threshold {
                let byte_idx = i / 8;
                let bit = 1u8 << (i % 8);
                if w > 0.0 {
                    pos_mask[byte_idx] |= bit;
                } else {
                    neg_mask[byte_idx] |= bit;
                }
                abs_sum += w.abs();
                nonzero_count += 1;
            }
        }

        let scale = if nonzero_count > 0 {
            abs_sum / nonzero_count as f32
        } else {
            0.0
        };

        Self { pos_mask, neg_mask, scale, num_cols }
    }
}

impl TernaryMatrix {
    /// Ternarize from quantized GGUF data (dequantizes row-by-row, then ternarizes).
    pub fn from_quantized(
        data: &[u8],
        qtype: GgmlType,
        num_rows: usize,
        num_cols: usize,
        threshold_ratio: f32,
    ) -> Self {
        let elems_per_block = qtype.elements_per_block();
        let bytes_per_block = qtype.block_bytes();
        let blocks_per_row = num_cols / elems_per_block;
        let row_bytes = blocks_per_row * bytes_per_block;

        let mut rows = Vec::with_capacity(num_rows);
        let mut row_f32 = vec![0.0f32; num_cols];

        for r in 0..num_rows {
            let row_data = &data[r * row_bytes..(r + 1) * row_bytes];
            // Dequantize row to f32
            match qtype {
                GgmlType::Q4_K => dequantize_row_q4k(row_data, &mut row_f32),
                GgmlType::Q6_K => dequantize_row_q6k(row_data, &mut row_f32),
                GgmlType::Q8_0 => dequantize_row_q8_0(row_data, &mut row_f32),
                GgmlType::F16 => dequantize_row_f16(row_data, &mut row_f32),
                GgmlType::F32 => {
                    for i in 0..num_cols {
                        row_f32[i] = f32::from_le_bytes([
                            row_data[i * 4],
                            row_data[i * 4 + 1],
                            row_data[i * 4 + 2],
                            row_data[i * 4 + 3],
                        ]);
                    }
                }
                _ => {
                    row_f32.fill(0.0);
                }
            }
            rows.push(TernaryRow::from_f32(&row_f32, threshold_ratio));
        }

        Self { rows, num_rows, num_cols }
    }
}

/// Ternary matvec: output = TernaryMatrix × input.
/// No multiplications in the inner loop — only additions and subtractions.
pub fn ternary_matvec(matrix: &TernaryMatrix, input: &[f32], output: &mut [f32]) {
    ternary_matvec_impl(&matrix.rows, input, output);
}

fn ternary_matvec_impl(rows: &[TernaryRow], input: &[f32], output: &mut [f32]) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(r, out)| {
                *out = ternary_dot_row(&rows[r], input);
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for (r, out) in output.iter_mut().enumerate() {
            *out = ternary_dot_row(&rows[r], input);
        }
    }
}

#[inline]
fn ternary_dot_row(row: &TernaryRow, input: &[f32]) -> f32 {
    let mut pos_sum = 0.0f32;
    let mut neg_sum = 0.0f32;
    let num_bytes = row.pos_mask.len();

    for b in 0..num_bytes {
        let pm = row.pos_mask[b];
        let nm = row.neg_mask[b];
        if (pm | nm) == 0 {
            continue;
        }
        let base = b * 8;
        // Unrolled 8-element loop for auto-vectorization
        for bit in 0..8 {
            let idx = base + bit;
            if idx >= row.num_cols {
                break;
            }
            let p = ((pm >> bit) & 1) as f32;
            let n = ((nm >> bit) & 1) as f32;
            pos_sum += p * input[idx];
            neg_sum += n * input[idx];
        }
    }

    row.scale * (pos_sum - neg_sum)
}

// ─── Sparse Ternary (N:M Structured Sparsity) ──────────────────────────────

/// Block size for N:M structured sparsity. Within each block of SPARSE_BLOCK elements,
/// at most SPARSE_N are non-zero. This avoids per-element metadata overhead.
pub const SPARSE_BLOCK: usize = 16;

/// A sparse ternary row using N:M structured sparsity.
/// Within each block of SPARSE_BLOCK elements, non-zero positions are stored
/// as a compact bitmask (16 bits = 2 bytes per block), and signs are packed
/// in another bitmask (only for non-zero positions).
pub struct SparseTernaryRow {
    /// Per-block: which positions are non-zero (bit i = 1 means position i is active).
    pub active_masks: Vec<u16>,
    /// Per-block: sign of active positions (bit i = 1 means negative).
    pub sign_masks: Vec<u16>,
    /// Scale factor (mean of |non-zero weights|).
    pub scale: f32,
    pub num_cols: usize,
    pub num_blocks: usize,
}

/// Flat-buffer sparse ternary matrix for cache-friendly SIMD matvec.
/// All row masks are stored contiguously in a single allocation.
pub struct SparseTernaryMatrix {
    /// Flat buffer: [row0_active, row0_sign, row1_active, row1_sign, ...]
    /// Each row has `blocks_per_row` u16 active masks followed by `blocks_per_row` u16 sign masks.
    mask_buf: Vec<u16>,
    /// Per-row scale factors (contiguous).
    scales: Vec<f32>,
    pub num_rows: usize,
    pub num_cols: usize,
    pub blocks_per_row: usize,
    /// Target sparsity: fraction of zeros per block (e.g. 0.5 = 8:16).
    pub target_sparsity: f32,
}

impl SparseTernaryRow {
    /// Create from a dense TernaryRow, enforcing N:M structured sparsity.
    /// Within each block of SPARSE_BLOCK, keeps only the top-N by magnitude
    /// (from the original f32 weights) and zeros the rest.
    pub fn from_ternary_row(row: &TernaryRow, original_weights: &[f32], n_keep: usize) -> Self {
        let num_cols = row.num_cols;
        let num_blocks = (num_cols + SPARSE_BLOCK - 1) / SPARSE_BLOCK;
        let mut active_masks = vec![0u16; num_blocks];
        let mut sign_masks = vec![0u16; num_blocks];
        let mut abs_sum = 0.0f64;
        let mut nonzero_count = 0usize;

        for blk in 0..num_blocks {
            let base = blk * SPARSE_BLOCK;
            let end = (base + SPARSE_BLOCK).min(num_cols);
            let block_len = end - base;

            let mut candidates: Vec<(usize, f32)> = Vec::new();
            for i in 0..block_len {
                let global = base + i;
                let byte_idx = global / 8;
                let bit = 1u8 << (global % 8);
                let is_nonzero = (row.pos_mask[byte_idx] & bit) != 0
                    || (row.neg_mask[byte_idx] & bit) != 0;
                if is_nonzero {
                    candidates.push((i, original_weights[global].abs()));
                }
            }

            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            candidates.truncate(n_keep.min(block_len));

            for &(i, mag) in &candidates {
                let global = base + i;
                active_masks[blk] |= 1u16 << i;

                let byte_idx = global / 8;
                let bit = 1u8 << (global % 8);
                if (row.neg_mask[byte_idx] & bit) != 0 {
                    sign_masks[blk] |= 1u16 << i;
                }

                abs_sum += mag as f64;
                nonzero_count += 1;
            }
        }

        let scale = if nonzero_count > 0 {
            (abs_sum / nonzero_count as f64) as f32
        } else {
            row.scale
        };

        Self { active_masks, sign_masks, scale, num_cols, num_blocks }
    }

    /// Create directly from f32 weights with N:M structured sparsity.
    pub fn from_f32(weights: &[f32], threshold_ratio: f32, n_keep: usize) -> Self {
        let row = TernaryRow::from_f32(weights, threshold_ratio);
        Self::from_ternary_row(&row, weights, n_keep)
    }

    /// Count of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.active_masks.iter().map(|m| m.count_ones() as usize).sum()
    }

    /// Effective bits per parameter (including the 2-byte masks per block overhead).
    pub fn effective_bits_per_param(&self) -> f32 {
        if self.num_cols == 0 {
            return 0.0;
        }
        let mask_bits = self.num_blocks as f32 * 32.0;
        let total_bits = mask_bits + 32.0;
        total_bits / self.num_cols as f32
    }
}

impl SparseTernaryMatrix {
    /// Build from Vec of SparseTernaryRows, packing into flat contiguous buffer.
    pub fn from_rows(rows: Vec<SparseTernaryRow>, num_rows: usize, num_cols: usize, target_sparsity: f32) -> Self {
        let blocks_per_row = (num_cols + SPARSE_BLOCK - 1) / SPARSE_BLOCK;
        // Layout: for each row, blocks_per_row active masks then blocks_per_row sign masks
        let stride = blocks_per_row * 2; // active + sign
        let mut mask_buf = vec![0u16; num_rows * stride];
        let mut scales = Vec::with_capacity(num_rows);

        for (r, row) in rows.iter().enumerate() {
            let base = r * stride;
            mask_buf[base..base + blocks_per_row].copy_from_slice(&row.active_masks);
            mask_buf[base + blocks_per_row..base + stride].copy_from_slice(&row.sign_masks);
            scales.push(row.scale);
        }

        Self { mask_buf, scales, num_rows, num_cols, blocks_per_row, target_sparsity }
    }

    /// Convert a dense TernaryMatrix to sparse, enforcing N:M structured sparsity.
    pub fn from_ternary(
        dense: &TernaryMatrix,
        original_weights_per_row: &[Vec<f32>],
        n_keep: usize,
    ) -> Self {
        let target_sparsity = 1.0 - (n_keep as f32 / SPARSE_BLOCK as f32);
        let rows: Vec<SparseTernaryRow> = dense.rows.iter()
            .zip(original_weights_per_row.iter())
            .map(|(row, orig)| SparseTernaryRow::from_ternary_row(row, orig, n_keep))
            .collect();
        Self::from_rows(rows, dense.num_rows, dense.num_cols, target_sparsity)
    }

    /// Create directly from f32 weight data (row-major).
    pub fn from_f32_weights(
        weights: &[f32],
        num_rows: usize,
        num_cols: usize,
        threshold_ratio: f32,
        n_keep: usize,
    ) -> Self {
        let target_sparsity = 1.0 - (n_keep as f32 / SPARSE_BLOCK as f32);
        let mut rows = Vec::with_capacity(num_rows);
        for r in 0..num_rows {
            let row_data = &weights[r * num_cols..(r + 1) * num_cols];
            rows.push(SparseTernaryRow::from_f32(row_data, threshold_ratio, n_keep));
        }
        Self::from_rows(rows, num_rows, num_cols, target_sparsity)
    }

    /// Get active_masks slice for a row.
    #[inline]
    fn active_masks(&self, row: usize) -> &[u16] {
        let base = row * self.blocks_per_row * 2;
        &self.mask_buf[base..base + self.blocks_per_row]
    }

    /// Get sign_masks slice for a row.
    #[inline]
    fn sign_masks(&self, row: usize) -> &[u16] {
        let base = row * self.blocks_per_row * 2 + self.blocks_per_row;
        &self.mask_buf[base..base + self.blocks_per_row]
    }

    /// Total non-zero elements across all rows.
    pub fn total_nnz(&self) -> usize {
        let stride = self.blocks_per_row * 2;
        let mut total = 0usize;
        for r in 0..self.num_rows {
            let base = r * stride;
            for b in 0..self.blocks_per_row {
                total += self.mask_buf[base + b].count_ones() as usize;
            }
        }
        total
    }

    /// Estimated memory in bytes (flat buffer + scales).
    pub fn memory_bytes(&self) -> usize {
        self.mask_buf.len() * 2 + self.scales.len() * 4
    }
}

/// Quantize f32 activation vector to i8 with a single global scale.
/// Returns (quantized_i8, scale) where original ≈ quantized * scale.
/// Padding to SPARSE_BLOCK alignment is applied for safe NEON loads.
fn quantize_activation_i8(input: &[f32]) -> (Vec<i8>, f32) {
    let abs_max = input.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    if abs_max == 0.0 {
        let padded = ((input.len() + SPARSE_BLOCK - 1) / SPARSE_BLOCK) * SPARSE_BLOCK;
        return (vec![0i8; padded], 1.0);
    }
    let inv_scale = 127.0 / abs_max;
    let scale = abs_max / 127.0;
    let padded_len = ((input.len() + SPARSE_BLOCK - 1) / SPARSE_BLOCK) * SPARSE_BLOCK;
    let mut quant = vec![0i8; padded_len];
    for (i, &val) in input.iter().enumerate() {
        quant[i] = (val * inv_scale).round().max(-127.0).min(127.0) as i8;
    }
    (quant, scale)
}

/// Sparse ternary matvec: output = SparseTernaryMatrix × input.
/// On aarch64 with dotprod: i8 dynamic quantization + vdotq_s32 (16 elements/instruction).
/// Branchless 4-row micro-kernel with shared activation loads.
pub fn sparse_ternary_matvec(matrix: &SparseTernaryMatrix, input: &[f32], output: &mut [f32]) {
    // Quantize activation to i8 once (shared across all rows)
    #[cfg(target_arch = "aarch64")]
    let (act_i8, act_scale) = quantize_activation_i8(input);

    #[cfg(all(target_arch = "aarch64", feature = "parallel"))]
    {
        use rayon::prelude::*;
        output
            .par_chunks_mut(4)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let row_start = chunk_idx * 4;
                if chunk.len() == 4 {
                    unsafe {
                        sparse_ternary_microkernel_4row_sdot(
                            matrix, &act_i8, act_scale, chunk, row_start,
                        );
                    }
                } else {
                    for (i, out) in chunk.iter_mut().enumerate() {
                        unsafe {
                            *out = sparse_ternary_dot_sdot(
                                matrix, &act_i8, act_scale, row_start + i,
                            );
                        }
                    }
                }
            });
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "parallel")))]
    {
        let rows = matrix.num_rows;
        let mut r = 0;
        while r + 4 <= rows {
            unsafe {
                sparse_ternary_microkernel_4row_sdot(
                    matrix, &act_i8, act_scale, &mut output[r..r + 4], r,
                );
            }
            r += 4;
        }
        while r < rows {
            unsafe {
                output[r] = sparse_ternary_dot_sdot(matrix, &act_i8, act_scale, r);
            }
            r += 1;
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            output
                .par_iter_mut()
                .enumerate()
                .for_each(|(r, out)| {
                    *out = sparse_ternary_dot_flat(matrix, r, input);
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for (r, out) in output.iter_mut().enumerate() {
                *out = sparse_ternary_dot_flat(matrix, r, input);
            }
        }
    }
}

/// Scalar fallback dispatch for single-row dot product (non-aarch64).
#[inline]
fn sparse_ternary_dot_flat(matrix: &SparseTernaryMatrix, row: usize, input: &[f32]) -> f32 {
    sparse_ternary_dot_scalar(matrix, row, input)
}

/// Single-row sdot-based dot product for tail rows.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn sparse_ternary_dot_sdot(
    matrix: &SparseTernaryMatrix,
    act_i8: &[i8],
    act_scale: f32,
    row: usize,
) -> f32 {
    use std::arch::aarch64::*;

    let active = matrix.active_masks(row);
    let signs = matrix.sign_masks(row);
    let row_scale = matrix.scales[row];
    let blocks = matrix.blocks_per_row;

    // i32 accumulator — single register, sdot adds 16 products at once
    let mut acc = vdupq_n_s32(0);

    // Mask expansion constants
    let bit_pattern = vld1q_u8(
        [1u8, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128].as_ptr(),
    );
    let zero_u8 = vdupq_n_u8(0);
    let pos_one_i8 = vdupq_n_s8(1);
    let neg_one_i8 = vdupq_n_s8(-1);
    let zero_i8 = vdupq_n_s8(0);

    // Branchless loop
    for blk in 0..blocks {
        // Load i8 quantized activation (1 instruction for 16 elements!)
        let aq = vld1q_s8(act_i8.as_ptr().add(blk * SPARSE_BLOCK));

        // Expand u16 masks → i8 weight vector
        let w = expand_masks_to_i8(
            active[blk], signs[blk],
            bit_pattern, zero_u8, pos_one_i8, neg_one_i8, zero_i8,
        );

        // sdot: accumulate 16 × (weight_i8 × activation_i8) → 4 × i32
        acc = sdot_s32(acc, w, aq);
    }

    // Horizontal sum of 4 i32 lanes → single i32 → f32
    let sum_i32 = vaddvq_s32(acc);
    row_scale * act_scale * (sum_i32 as f32)
}

/// Branchless 4-row sdot micro-kernel (Register Blocking + Integer Dot Product).
/// - i8 quantized activation loaded ONCE per block, shared across 4 rows
/// - u16 masks expanded to i8 weights via byte-level NEON (vandq_u8 + vcgtq_u8 + vbslq_s8)
/// - vdotq_s32: 16 × i8×i8 multiplied and accumulated in 1 instruction
/// - ZERO branches in inner loop
/// - 4 i32 accumulators + 1 shared activation + 5 constants = ~10 registers
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sparse_ternary_microkernel_4row_sdot(
    matrix: &SparseTernaryMatrix,
    act_i8: &[i8],
    act_scale: f32,
    output: &mut [f32],
    row_start: usize,
) {
    use std::arch::aarch64::*;

    let blocks = matrix.blocks_per_row;

    // 4 × i32 accumulators (one int32x4_t per row)
    let mut acc0 = vdupq_n_s32(0);
    let mut acc1 = vdupq_n_s32(0);
    let mut acc2 = vdupq_n_s32(0);
    let mut acc3 = vdupq_n_s32(0);

    // Constants for mask expansion — live in registers
    let bit_pattern = vld1q_u8(
        [1u8, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128].as_ptr(),
    );
    let zero_u8 = vdupq_n_u8(0);
    let pos_one_i8 = vdupq_n_s8(1);
    let neg_one_i8 = vdupq_n_s8(-1);
    let zero_i8 = vdupq_n_s8(0);

    // Mask slices for 4 rows
    let act_r0 = matrix.active_masks(row_start);
    let act_r1 = matrix.active_masks(row_start + 1);
    let act_r2 = matrix.active_masks(row_start + 2);
    let act_r3 = matrix.active_masks(row_start + 3);
    let sgn_r0 = matrix.sign_masks(row_start);
    let sgn_r1 = matrix.sign_masks(row_start + 1);
    let sgn_r2 = matrix.sign_masks(row_start + 2);
    let sgn_r3 = matrix.sign_masks(row_start + 3);

    for blk in 0..blocks {
        // 1. Load i8 activation ONCE — shared across 4 rows
        let aq = vld1q_s8(act_i8.as_ptr().add(blk * SPARSE_BLOCK));

        // 2. Row 0: expand masks → i8 weights → sdot
        let w0 = expand_masks_to_i8(
            act_r0[blk], sgn_r0[blk],
            bit_pattern, zero_u8, pos_one_i8, neg_one_i8, zero_i8,
        );
        acc0 = sdot_s32(acc0, w0, aq);

        // 3. Row 1
        let w1 = expand_masks_to_i8(
            act_r1[blk], sgn_r1[blk],
            bit_pattern, zero_u8, pos_one_i8, neg_one_i8, zero_i8,
        );
        acc1 = sdot_s32(acc1, w1, aq);

        // 4. Row 2
        let w2 = expand_masks_to_i8(
            act_r2[blk], sgn_r2[blk],
            bit_pattern, zero_u8, pos_one_i8, neg_one_i8, zero_i8,
        );
        acc2 = sdot_s32(acc2, w2, aq);

        // 5. Row 3
        let w3 = expand_masks_to_i8(
            act_r3[blk], sgn_r3[blk],
            bit_pattern, zero_u8, pos_one_i8, neg_one_i8, zero_i8,
        );
        acc3 = sdot_s32(acc3, w3, aq);
    }

    // Horizontal reduction: i32 → f32, apply scales
    let s0 = matrix.scales[row_start] * act_scale;
    let s1 = matrix.scales[row_start + 1] * act_scale;
    let s2 = matrix.scales[row_start + 2] * act_scale;
    let s3 = matrix.scales[row_start + 3] * act_scale;

    output[0] = s0 * (vaddvq_s32(acc0) as f32);
    output[1] = s1 * (vaddvq_s32(acc1) as f32);
    output[2] = s2 * (vaddvq_s32(acc2) as f32);
    output[3] = s3 * (vaddvq_s32(acc3) as f32);
}

/// SDOT via inline assembly: acc.4s += a.16b · b.16b
/// Processes 16 × i8×i8 → 4 × i32 accumulation in 1 instruction.
/// Uses inline asm because vdotq_s32 is unstable in Rust's std::arch.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sdot_s32(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let mut result = acc;
    core::arch::asm!(
        "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) result,
        a = in(vreg) a,
        b = in(vreg) b,
        options(nostack, preserves_flags),
    );
    result
}

/// Expand u16 active + sign masks into a single int8x16_t weight vector.
/// Each element: active ? (sign ? -1 : +1) : 0.
/// Uses byte-level NEON: vandq_u8 + vcgtq_u8 for bit expansion, vbslq_s8 for select.
/// All constants passed as parameters to keep them in registers.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn expand_masks_to_i8(
    active: u16,
    sign: u16,
    bit_pattern: std::arch::aarch64::uint8x16_t,
    zero_u8: std::arch::aarch64::uint8x16_t,
    pos_one_i8: std::arch::aarch64::int8x16_t,
    neg_one_i8: std::arch::aarch64::int8x16_t,
    zero_i8: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int8x16_t {
    use std::arch::aarch64::*;

    // Expand active mask: broadcast lo/hi bytes, AND with bit pattern, compare > 0
    let act_lo = vdup_n_u8((active & 0xFF) as u8);
    let act_hi = vdup_n_u8((active >> 8) as u8);
    let act_vec = vcombine_u8(act_lo, act_hi);
    let act_mask = vcgtq_u8(vandq_u8(act_vec, bit_pattern), zero_u8);

    // Expand sign mask: same approach
    let sgn_lo = vdup_n_u8((sign & 0xFF) as u8);
    let sgn_hi = vdup_n_u8((sign >> 8) as u8);
    let sgn_vec = vcombine_u8(sgn_lo, sgn_hi);
    let sgn_mask = vcgtq_u8(vandq_u8(sgn_vec, bit_pattern), zero_u8);

    // weight = active ? (sign ? -1 : +1) : 0
    // vbslq_s8(mask: uint8x16_t, a: int8x16_t, b: int8x16_t)
    // sgn_mask and act_mask are already uint8x16_t from vcgtq_u8
    let sign_weight = vbslq_s8(sgn_mask, neg_one_i8, pos_one_i8);
    vbslq_s8(act_mask, sign_weight, zero_i8)
}

/// Scalar fallback (used on non-aarch64 and as reference implementation).
#[inline]
fn sparse_ternary_dot_scalar(matrix: &SparseTernaryMatrix, row: usize, input: &[f32]) -> f32 {
    let active = matrix.active_masks(row);
    let signs = matrix.sign_masks(row);
    let scale = matrix.scales[row];

    let mut sum = 0.0f32;

    for blk in 0..matrix.blocks_per_row {
        let a = active[blk];
        if a == 0 {
            continue;
        }
        let s = signs[blk];
        let base = blk * SPARSE_BLOCK;

        let pos_mask = a & !s;
        let neg_mask = a & s;

        let mut m = pos_mask;
        while m != 0 {
            let bit = m.trailing_zeros() as usize;
            sum += input[base + bit];
            m &= m - 1;
        }

        m = neg_mask;
        while m != 0 {
            let bit = m.trailing_zeros() as usize;
            sum -= input[base + bit];
            m &= m - 1;
        }
    }

    scale * sum
}

/// Dequantize a single Q4_K row to f32.
fn dequantize_row_q4k(data: &[u8], out: &mut [f32]) {
    let blocks = out.len() / QK_K;
    let block_size = 144; // Q4_K block size

    for blk in 0..blocks {
        let b = &data[blk * block_size..];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
        let scales_raw = &b[4..16];
        let qs = &b[16..144];

        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        for i in 0..4 {
            scales[i] = scales_raw[i] & 0x3F;
            mins[i] = scales_raw[i + 4] & 0x3F;
        }
        scales[4] = (scales_raw[0 + 8] & 0xF) | ((scales_raw[0] >> 6) << 4);
        scales[5] = (scales_raw[1 + 8] & 0xF) | ((scales_raw[1] >> 6) << 4);
        scales[6] = (scales_raw[2 + 8] & 0xF) | ((scales_raw[2] >> 6) << 4);
        scales[7] = (scales_raw[3 + 8] & 0xF) | ((scales_raw[3] >> 6) << 4);
        mins[4] = (scales_raw[0 + 8] >> 4) | ((scales_raw[4] >> 6) << 4);
        mins[5] = (scales_raw[1 + 8] >> 4) | ((scales_raw[5] >> 6) << 4);
        mins[6] = (scales_raw[2 + 8] >> 4) | ((scales_raw[6] >> 6) << 4);
        mins[7] = (scales_raw[3 + 8] >> 4) | ((scales_raw[7] >> 6) << 4);

        for is in 0..8 {
            let sc = d * scales[is] as f32;
            let mn = dmin * mins[is] as f32;
            let off = is * 32;
            for l in 0..32 {
                let qi = if l < 16 {
                    let byte_idx = (is / 2) * 32 + l;
                    if is % 2 == 0 { qs[byte_idx] & 0xF } else { qs[byte_idx] >> 4 }
                } else {
                    let byte_idx = (is / 2) * 32 + 16 + (l - 16);
                    if is % 2 == 0 { qs[byte_idx] & 0xF } else { qs[byte_idx] >> 4 }
                };
                out[blk * QK_K + off + l] = sc * qi as f32 - mn;
            }
        }
    }
}

/// Dequantize a single Q6_K row to f32 (matches llama.cpp bit-packing layout).
fn dequantize_row_q6k(data: &[u8], out: &mut [f32]) {
    let blocks = out.len() / QK_K;
    let block_bytes = 210;

    for blk in 0..blocks {
        let b = &data[blk * block_bytes..];
        let d = f16_to_f32(u16::from_le_bytes([b[208], b[209]]));
        let ql = &b[0..128];
        let qh = &b[128..192];
        let scales = &b[192..208];

        // Reconstruct 6-bit signed quants (same as q6k_q8k_dot)
        let mut aux8 = [0i8; QK_K];
        let mut a_off = 0usize;
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        for _ in 0..2 {
            for l in 0..32 {
                aux8[a_off + l] =
                    ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i8 - 32;
                aux8[a_off + l + 32] =
                    ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8 - 32;
                aux8[a_off + l + 64] =
                    ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 - 32;
                aux8[a_off + l + 96] =
                    ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 - 32;
            }
            a_off += 128;
            ql_off += 64;
            qh_off += 32;
        }

        // Apply scales and write output
        let base = blk * QK_K;
        for is in 0..16 {
            let scale = d * (scales[is] as i8) as f32;
            let off = is * 16;
            for l in 0..16 {
                out[base + off + l] = scale * aux8[off + l] as f32;
            }
        }
    }
}

/// Dequantize a single Q8_0 row to f32.
fn dequantize_row_q8_0(data: &[u8], out: &mut [f32]) {
    let blocks = out.len() / QK8_0;
    let block_size = 34; // 2 bytes f16 scale + 32 i8 values

    for blk in 0..blocks {
        let b = &data[blk * block_size..];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        for i in 0..QK8_0 {
            out[blk * QK8_0 + i] = d * (b[2 + i] as i8) as f32;
        }
    }
}

/// Dequantize a single F16 row to f32.
fn dequantize_row_f16(data: &[u8], out: &mut [f32]) {
    for i in 0..out.len() {
        out[i] = f16_to_f32(u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]));
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_to_f32_basic() {
        // 0x0000 = 0.0
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // 0x3C00 = 1.0
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // 0xBC00 = -1.0
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);
        // 0x4000 = 2.0
        assert!((f16_to_f32(0x4000) - 2.0).abs() < 1e-6);
        // 0x3800 = 0.5
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_f16_inf_nan() {
        // +Inf = 0x7C00
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7C00) > 0.0);
        // -Inf = 0xFC00
        assert!(f16_to_f32(0xFC00).is_infinite());
        assert!(f16_to_f32(0xFC00) < 0.0);
        // NaN = 0x7C01
        assert!(f16_to_f32(0x7C01).is_nan());
    }

    #[test]
    fn test_ggml_type_block_sizes() {
        assert_eq!(GgmlType::F32.block_bytes(), 4);
        assert_eq!(GgmlType::F16.block_bytes(), 2);
        assert_eq!(GgmlType::Q8_0.block_bytes(), 34);
        assert_eq!(GgmlType::Q4_K.block_bytes(), 144);
        assert_eq!(GgmlType::Q4_K.elements_per_block(), 256);
        assert_eq!(GgmlType::Q8_0.elements_per_block(), 32);
    }

    #[test]
    fn test_q8_0_dequantize_roundtrip() {
        // Create a Q8_0 block: scale=0.1, values=[0,1,2,...,31]
        let mut block = vec![0u8; 34];
        // scale = 0.1 in f16 ≈ 0x2E66
        let scale_f16: u16 = 0x2E66;
        block[0] = scale_f16 as u8;
        block[1] = (scale_f16 >> 8) as u8;
        for i in 0..32 {
            block[2 + i] = i as u8;
        }

        let mut out = vec![0.0f32; 32];
        dequantize_q8_0(&block, &mut out);

        let scale = f16_to_f32(scale_f16);
        for i in 0..32 {
            let expected = scale * i as f32;
            assert!(
                (out[i] - expected).abs() < 1e-4,
                "q8_0[{i}]: got {}, expected {expected}",
                out[i]
            );
        }
    }

    #[test]
    fn test_q8_0_fused_matvec() {
        // 2x32 identity-like matrix: row0=[1,0,0,...], row1=[0,1,0,...]
        // Scale = 1.0 (f16 = 0x3C00)
        let mut data = vec![0u8; 2 * 34];
        let scale: u16 = 0x3C00;

        // Row 0
        data[0] = scale as u8;
        data[1] = (scale >> 8) as u8;
        data[2] = 1; // qs[0] = 1

        // Row 1
        data[34] = scale as u8;
        data[35] = (scale >> 8) as u8;
        data[36] = 0; // qs[0] = 0
        data[37] = 1; // qs[1] = 1

        let mut input = vec![0.0f32; 32];
        input[0] = 3.0;
        input[1] = 7.0;

        let mut output = vec![0.0f32; 2];
        q8_0_matvec(&input, &data, 2, 32, &mut output);

        assert!((output[0] - 3.0).abs() < 1e-4);
        assert!((output[1] - 7.0).abs() < 1e-4);
    }

    #[test]
    fn test_get_scale_min_k4() {
        let scales = [10, 20, 30, 40, 50, 60, 70, 80, 0xAB, 0xCD, 0xEF, 0x12];

        // j < 4: sc = scales[j] & 63, m = scales[j+4] & 63
        let (sc, m) = get_scale_min_k4(0, &scales);
        assert_eq!(sc, 10 & 63);
        assert_eq!(m, 50 & 63);

        let (sc, m) = get_scale_min_k4(3, &scales);
        assert_eq!(sc, 40 & 63);
        assert_eq!(m, 80 & 63);
    }

    #[test]
    fn test_gguf_parse_invalid_magic() {
        let data = [0u8; 64];
        assert!(GgufFile::parse(&data).is_none());
    }

    #[test]
    fn test_tensor_info_n_elements() {
        let info = TensorInfo {
            name: "test".into(),
            n_dims: 2,
            dims: vec![4096, 4096],
            qtype: GgmlType::Q4_K,
            offset: 0,
        };
        assert_eq!(info.n_elements(), 4096 * 4096);
    }

    #[test]
    fn test_tensor_info_data_size() {
        let info = TensorInfo {
            name: "test".into(),
            n_dims: 2,
            dims: vec![4096, 4096],
            qtype: GgmlType::Q4_K,
            offset: 0,
        };
        let n = 4096 * 4096;
        let n_blocks = n / 256;
        assert_eq!(info.data_size(), n_blocks * 144);
    }

    #[test]
    fn test_meta_value_conversions() {
        let u = MetaValue::U32(42);
        assert_eq!(u.as_u32(), Some(42));
        assert_eq!(u.as_f32(), None);

        let f = MetaValue::F32(3.14);
        assert!((f.as_f32().unwrap() - 3.14).abs() < 1e-6);

        let s = MetaValue::Str("hello".into());
        assert_eq!(s.as_str(), Some("hello"));
    }

    #[test]
    fn test_sparse_ternary_row_from_f32() {
        // 32 weights, keep 8 per block of 16 (50% sparsity)
        let weights: Vec<f32> = (0..32).map(|i| (i as f32 - 15.5) * 0.1).collect();
        let row = SparseTernaryRow::from_f32(&weights, 0.3, 8);

        assert_eq!(row.num_cols, 32);
        assert_eq!(row.num_blocks, 2);
        // At most 8 active per block
        assert!(row.active_masks[0].count_ones() <= 8);
        assert!(row.active_masks[1].count_ones() <= 8);
        assert!(row.scale > 0.0);
    }

    #[test]
    fn test_sparse_ternary_enforces_nm_sparsity() {
        // 16 weights, keep 4 per block (75% sparsity = 4:16)
        let weights: Vec<f32> = vec![
            1.0, -2.0, 0.5, -0.1, 3.0, -0.3, 0.8, -0.05,
            0.02, -1.5, 0.7, -0.9, 0.01, -0.4, 2.5, -0.6,
        ];
        let row = SparseTernaryRow::from_f32(&weights, 0.1, 4);
        assert_eq!(row.active_masks[0].count_ones(), 4);
        assert_eq!(row.nnz(), 4);
    }

    #[test]
    fn test_sparse_ternary_matvec_matches_dense() {
        // Create a small matrix, compare sparse vs dense ternary matvec
        let weights = vec![
            1.0, -2.0, 0.5, -0.1, 3.0, -0.3, 0.8, -0.05,
            0.02, -1.5, 0.7, -0.9, 0.01, -0.4, 2.5, -0.6,
            // Row 2
            -1.0, 2.0, -0.5, 0.1, -3.0, 0.3, -0.8, 0.05,
            -0.02, 1.5, -0.7, 0.9, -0.01, 0.4, -2.5, 0.6,
        ];
        let num_rows = 2;
        let num_cols = 16;

        // Dense ternary (keep all)
        let dense = TernaryMatrix {
            rows: (0..num_rows).map(|r| {
                TernaryRow::from_f32(&weights[r * num_cols..(r + 1) * num_cols], 0.05)
            }).collect(),
            num_rows,
            num_cols,
        };

        // Sparse ternary (keep 16 = all, so should match dense)
        let sparse = SparseTernaryMatrix::from_f32_weights(&weights, num_rows, num_cols, 0.05, 16);

        let input = vec![1.0, 0.5, -1.0, 2.0, 0.3, -0.7, 1.5, -0.2,
                         0.8, -1.2, 0.4, 0.6, -0.9, 1.1, -0.3, 0.7];
        let mut dense_out = vec![0.0f32; num_rows];
        let mut sparse_out = vec![0.0f32; num_rows];

        ternary_matvec(&dense, &input, &mut dense_out);
        sparse_ternary_matvec(&sparse, &input, &mut sparse_out);

        // With n_keep=16 (all), sparse should match dense
        // Tolerance widened for i8 activation quantization (~2% relative error)
        for i in 0..num_rows {
            let tol = dense_out[i].abs().max(0.01) * 0.05;
            assert!(
                (dense_out[i] - sparse_out[i]).abs() < tol,
                "row {i}: dense={} sparse={}", dense_out[i], sparse_out[i]
            );
        }
    }

    #[test]
    fn test_sparse_ternary_memory_reduction() {
        // 1024-col row, dense ternary = 2 bitmasks of 128 bytes each = 256 bytes + scale
        // sparse ternary = 64 blocks × (2+2) bytes = 256 bytes + scale
        // But with 50% sparsity, fewer ops in matvec (not less storage for masks)
        let weights: Vec<f32> = (0..1024).map(|i| ((i as f32 * 0.7).sin()) * 2.0).collect();
        let sparse = SparseTernaryMatrix::from_f32_weights(&weights, 1, 1024, 0.5, 8);

        // Verify nnz is roughly 50% of 1024
        let nnz = sparse.total_nnz();
        assert!(nnz <= 1024 / 2 + 64, "nnz={nnz} should be ~512");
        assert!(nnz > 0);

        // Verify memory is reasonable
        let mem = sparse.memory_bytes();
        assert!(mem > 0 && mem < 1024 * 1024, "memory={mem} bytes");
    }

    #[test]
    fn test_sparse_ternary_dot_correctness() {
        // Manual test: 16 weights, keep top-2, verify dot product
        let weights = vec![0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let row = SparseTernaryRow::from_f32(&weights, 0.01, 2);

        // Only positions 3 (+5.0→+1) and 10 (-3.0→-1) should be active
        assert_eq!(row.nnz(), 2);
        let scale = row.scale;

        // Build 1-row matrix for matvec test
        let matrix = SparseTernaryMatrix::from_rows(vec![row], 1, 16, 0.5);

        let mut input_test = vec![0.0f32; 16];
        input_test[3] = 10.0;
        input_test[10] = 7.0;

        let mut output = vec![0.0f32; 1];
        sparse_ternary_matvec(&matrix, &input_test, &mut output);

        // Expected: scale * (10.0 - 7.0) = scale * 3.0
        // Tolerance widened for i8 activation quantization (~0.5% error)
        let expected = scale * 3.0;
        assert!(
            (output[0] - expected).abs() < expected.abs() * 0.05,
            "result={}, expected={expected}", output[0]
        );
    }
}
