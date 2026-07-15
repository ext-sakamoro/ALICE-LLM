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
const fn f16_to_f32(h: u16) -> f32 {
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
    Q5_0,
    Q5_1,
    Q8_0,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    IQ4_XS,
    Other(u32),
}

impl GgmlType {
    const fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            23 => Self::IQ4_XS,
            other => Self::Other(other),
        }
    }

    /// Byte size per block of quantized data.
    #[must_use]
    pub const fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            // Q4_0: d (2) + qs (16) = 18
            Self::Q4_0 => 18,
            // Q4_1: d (2) + m (2) + qs (16) = 20
            Self::Q4_1 => 20,
            // Q5_0: d (2) + qh (4) + qs (16) = 22
            Self::Q5_0 => 22,
            // Q5_1: d (2) + m (2) + qh (4) + qs (16) = 24
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            // IQ4_XS: d (2) + scales_h (2) + scales_l (QK_K/64=4) + qs (QK_K/2=128) = 136
            Self::IQ4_XS => 136,
            Self::Other(_) => 0,
        }
    }

    /// Number of elements per block.
    #[must_use]
    pub const fn elements_per_block(&self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 => QK8_0,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::IQ4_XS => QK_K,
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
    Array(Vec<Self>),
}

impl MetaValue {
    /// Get as u32.
    pub const fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::I32(v) => Some(*v as u32),
            Self::U64(v) => Some(*v as u32),
            _ => None,
        }
    }

    /// Get as f32.
    pub const fn as_f32(&self) -> Option<f32> {
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

    /// Get as bool. Accepts Bool, U8 (0/1), and integer types (nonzero = true).
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            Self::U8(v) => Some(*v != 0),
            Self::U32(v) => Some(*v != 0),
            Self::I32(v) => Some(*v != 0),
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

    /// Get as u32 array (accepts arrays of U32/I32/U64 elements).
    pub fn as_u32_array(&self) -> Option<Vec<u32>> {
        match self {
            Self::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr {
                    out.push(v.as_u32()?);
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
        let n_blocks = n.div_ceil(epb);
        n_blocks * self.qtype.block_bytes()
    }
}

// ─── Binary reader ──────────────────────────────────────────────────────────

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    const fn new(data: &'a [u8]) -> Self {
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

    const fn align(&mut self, alignment: usize) {
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
        if !(2..=3).contains(&version) {
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
            .and_then(MetaValue::as_u32)
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

    /// Get metadata bool.
    #[must_use]
    pub fn meta_bool(&self, key: &str) -> Option<bool> {
        self.meta(key)?.as_bool()
    }

    /// Byte offset of a tensor's data section within the underlying file.
    ///
    /// Absolute offset = `tensor_data_start + info.offset`, matching what
    /// `tensor_data` would slice into. Exposed so the DeepSeek-V3 Phase 4
    /// streaming pool can locate expert slabs inside its own independent
    /// `Mmap` of the same file without re-borrowing this instance's slice.
    #[must_use]
    pub fn tensor_absolute_offset(&self, name: &str) -> Option<u64> {
        let info = self.tensors.get(name)?;
        Some(self.tensor_data_start as u64 + info.offset)
    }

    /// Byte length of a tensor's raw data (post-quant), matching what
    /// `tensor_data` would return. Convenient for streaming callers that
    /// need the size without also materialising a slice reference.
    #[must_use]
    pub fn tensor_byte_size(&self, name: &str) -> Option<usize> {
        Some(self.tensors.get(name)?.data_size())
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
            GgmlType::Q4_0 => dequantize_q4_0(data, &mut out),
            GgmlType::Q4_1 => dequantize_q4_1(data, &mut out),
            GgmlType::Q5_0 => dequantize_q5_0(data, &mut out),
            GgmlType::Q5_1 => dequantize_q5_1(data, &mut out),
            GgmlType::IQ4_XS => dequantize_iq4_xs(data, &mut out),
            GgmlType::Q8_0 => dequantize_q8_0(data, &mut out),
            GgmlType::Q2_K => dequantize_q2_k(data, &mut out),
            GgmlType::Q3_K => dequantize_q3_k(data, &mut out),
            GgmlType::Q4_K => dequantize_q4_k(data, &mut out),
            GgmlType::Q5_K => dequantize_q5_k(data, &mut out),
            GgmlType::Q6_K => dequantize_q6_k(data, &mut out),
            GgmlType::Other(_) => return None,
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

// ─── Q2_K dequantization ────────────────────────────────────────────────────
// Block layout (84 bytes per 256 elements):
//   scales[16] — 4-bit scale (low nibble) + 4-bit min (high nibble)
//   qs[64]     — 2-bit weights, 4 per byte
//   d (f16)    — super-block scale
//   dmin (f16) — super-block min

fn dequantize_q2_k(data: &[u8], out: &mut [f32]) {
    let block_bytes = 84;
    let n_blocks = data.len() / block_bytes;
    let mut out_idx = 0;

    for i in 0..n_blocks {
        let block = &data[i * block_bytes..(i + 1) * block_bytes];
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));

        for group in 0..16 {
            let sc = (scales[group] & 0xF) as f32;
            let m = (scales[group] >> 4) as f32;
            for j in 0..16 {
                let flat = group * 16 + j;
                let byte_idx = flat / 4;
                let bit_shift = (flat % 4) * 2;
                let q = ((qs[byte_idx] >> bit_shift) & 3) as f32;
                out[out_idx + flat] = d * sc * q - dmin * m;
            }
        }
        out_idx += QK_K;
    }
}

// ─── Q3_K dequantization ────────────────────────────────────────────────────
// Block layout (110 bytes per 256 elements):
//   hmask[32]  — high bit per weight (bit n of hmask[l] → element at group n*32+l)
//   qs[64]     — low 2 bits packed (4 groups of 32 per 128-element half)
//   scales[12] — 16 six-bit scales packed (value - 32)
//   d (f16)    — super-block scale

/// Decode 16 six-bit scales from Q3_K's 12-byte packed format.
fn q3k_decode_scales(sc: &[u8]) -> [i32; 16] {
    let mut scales = [0i32; 16];
    for j in 0..4 {
        scales[j] = (sc[j] & 0xF) as i32;
        scales[j + 4] = (sc[j] >> 4) as i32;
        scales[j + 8] = (sc[j + 4] & 0xF) as i32;
        scales[j + 12] = (sc[j + 4] >> 4) as i32;
    }
    for j in 0..4 {
        let hi = sc[8 + j];
        scales[j] |= ((hi & 3) as i32) << 4;
        scales[j + 4] |= (((hi >> 2) & 3) as i32) << 4;
        scales[j + 8] |= (((hi >> 4) & 3) as i32) << 4;
        scales[j + 12] |= (((hi >> 6) & 3) as i32) << 4;
    }
    for s in &mut scales {
        *s -= 32;
    }
    scales
}

fn dequantize_q3_k(data: &[u8], out: &mut [f32]) {
    let block_bytes = 110;
    let n_blocks = data.len() / block_bytes;
    let mut out_idx = 0;

    for i in 0..n_blocks {
        let block = &data[i * block_bytes..(i + 1) * block_bytes];
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let d = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
        let scales = q3k_decode_scales(&block[96..108]);

        // Unpack 3-bit weights: low 2 bits from qs, high bit from hmask
        // hmask[l] bit n → element at group_n * 32 + l
        // qs layout: 32 bytes per 128-element half, bits (shift*2)..(shift*2+1) for sub-group
        let mut m: u8 = 1;
        let mut q_off = 0usize;
        for j in (0..QK_K).step_by(128) {
            for shift in 0..4u8 {
                for l in 0..32usize {
                    let lo2 = (qs[q_off + l] >> (shift * 2)) & 3;
                    // hmask bit set → high bit present → value = lo2 (range 0..3)
                    // hmask bit clear → no high bit → value = lo2 - 4 (range -4..-1)
                    let hi = if hmask[l] & m != 0 { 0i8 } else { -4i8 };
                    let flat = j + shift as usize * 32 + l;
                    let is = flat / 16;
                    out[out_idx + flat] = d * scales[is] as f32 * (lo2 as i8 + hi) as f32;
                }
                m <<= 1;
            }
            q_off += 32;
        }
        out_idx += QK_K;
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

// ─── Q5_K dequantization ────────────────────────────────────────────────────
// Block layout (176 bytes per 256 elements):
//   d (f16)      — super-block scale
//   dmin (f16)   — super-block min
//   scales[12]   — same packing as Q4_K (8 scales + 8 mins, 6-bit each)
//   qh[32]       — high bits (bit 4 for each weight)
//   qs[128]      — low 4 bits

fn dequantize_q5_k(data: &[u8], out: &mut [f32]) {
    let block_bytes = 176;
    let n_blocks = data.len() / block_bytes;
    let mut out_idx = 0;

    for i in 0..n_blocks {
        let block = &data[i * block_bytes..(i + 1) * block_bytes];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales_raw = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];

        let mut is = 0usize;
        let mut q_offset = 0usize;

        for im in 0..4u8 {
            let (sc1, m1) = get_scale_min_k4(is, scales_raw);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales_raw);
            let d1 = d * f32::from(sc1);
            let m1f = dmin * f32::from(m1);
            let d2 = d * f32::from(sc2);
            let m2f = dmin * f32::from(m2);

            for l in 0..32 {
                let hbit = (qh[l] >> (im * 2)) & 1;
                out[out_idx] = d1 * f32::from((qs[q_offset + l] & 0xF) | (hbit << 4)) - m1f;
                out_idx += 1;
            }
            for l in 0..32 {
                let hbit = (qh[l] >> (im * 2 + 1)) & 1;
                out[out_idx] = d2 * f32::from((qs[q_offset + l] >> 4) | (hbit << 4)) - m2f;
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

// ─── Q5_1 dequantization ────────────────────────────────────────────────────
//
// Block layout (24 bytes / 32 elements):
//   d:  f16 delta          (offset 0..2)
//   m:  f16 min            (offset 2..4)
//   qh: u32 high-bits mask (offset 4..8, little-endian)
//   qs: [u8; 16] nibbles   (offset 8..24)
//
// Dequant per llama.cpp `dequantize_row_q5_1`:
//   for j in 0..16:
//     xh_0 = ((qh >>  j     ) << 4) & 0x10   // high bit for element j
//     xh_1 = ((qh >> (j+12)))       & 0x10   // high bit for element j+16
//     x0   = (qs[j] & 0x0F) | xh_0
//     x1   = (qs[j] >>   4) | xh_1
//     y[j]      = x0 * d + m
//     y[j + 16] = x1 * d + m
pub(crate) fn dequantize_q5_1(data: &[u8], out: &mut [f32]) {
    let block_bytes = 24;
    let n_blocks = data.len() / block_bytes;
    let mut out_idx = 0;

    for i in 0..n_blocks {
        let off = i * block_bytes;
        let d = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        let m = f16_to_f32(u16::from_le_bytes([data[off + 2], data[off + 3]]));
        let qh = u32::from_le_bytes([data[off + 4], data[off + 5], data[off + 6], data[off + 7]]);
        for j in 0..16 {
            let qs = data[off + 8 + j];
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            let x0 = ((qs & 0x0F) as u32) | xh_0;
            let x1 = ((qs >> 4) as u32) | xh_1;
            out[out_idx + j] = (x0 as f32) * d + m;
            out[out_idx + j + 16] = (x1 as f32) * d + m;
        }
        out_idx += QK8_0;
    }
}

// ─── Q4_0 dequantization ────────────────────────────────────────────────────
//
// Block layout (18 bytes / 32 elements):
//   d:  f16 delta          (offset 0..2)
//   qs: [u8; 16] nibbles   (offset 2..18)
//
// Dequant per llama.cpp `dequantize_row_q4_0`:
//   for j in 0..16:
//     x0 = (qs[j] & 0x0F) - 8   (signed 4-bit)
//     x1 = (qs[j] >>   4) - 8
//     y[j]      = d * x0
//     y[j + 16] = d * x1
pub(crate) fn dequantize_q4_0(data: &[u8], out: &mut [f32]) {
    let block_bytes = 18;
    let n_blocks = data.len() / block_bytes;
    let mut out_idx = 0;

    for i in 0..n_blocks {
        let off = i * block_bytes;
        let d = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        for j in 0..16 {
            let qs = data[off + 2 + j];
            let x0 = ((qs & 0x0F) as i32) - 8;
            let x1 = ((qs >> 4) as i32) - 8;
            out[out_idx + j] = (x0 as f32) * d;
            out[out_idx + j + 16] = (x1 as f32) * d;
        }
        out_idx += QK8_0;
    }
}

// ─── Q4_1 dequantization ────────────────────────────────────────────────────
//
// Block layout (20 bytes / 32 elements):
//   d:  f16 delta          (offset 0..2)
//   m:  f16 min            (offset 2..4)
//   qs: [u8; 16] nibbles   (offset 4..20)
//
// Dequant per llama.cpp `dequantize_row_q4_1`:
//   for j in 0..16:
//     y[j]      = d * (qs[j] & 0x0F) + m
//     y[j + 16] = d * (qs[j] >> 4)   + m
pub(crate) fn dequantize_q4_1(data: &[u8], out: &mut [f32]) {
    let block_bytes = 20;
    let n_blocks = data.len() / block_bytes;
    let mut out_idx = 0;

    for i in 0..n_blocks {
        let off = i * block_bytes;
        let d = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        let m = f16_to_f32(u16::from_le_bytes([data[off + 2], data[off + 3]]));
        for j in 0..16 {
            let qs = data[off + 4 + j];
            let x0 = (qs & 0x0F) as f32;
            let x1 = (qs >> 4) as f32;
            out[out_idx + j] = x0 * d + m;
            out[out_idx + j + 16] = x1 * d + m;
        }
        out_idx += QK8_0;
    }
}

// ─── Q5_0 dequantization ────────────────────────────────────────────────────
//
// Block layout (22 bytes / 32 elements):
//   d:  f16 delta          (offset 0..2)
//   qh: u32 high-bits mask (offset 2..6, little-endian)
//   qs: [u8; 16] nibbles   (offset 6..22)
//
// Dequant per llama.cpp `dequantize_row_q5_0`:
//   for j in 0..16:
//     xh_0 = ((qh >>  j     ) << 4) & 0x10
//     xh_1 = ((qh >> (j+12)))       & 0x10
//     x0   = ((qs[j] & 0x0F) | xh_0) - 16   (signed 5-bit)
//     x1   = ((qs[j] >>   4) | xh_1) - 16
//     y[j]      = d * x0
//     y[j + 16] = d * x1
pub(crate) fn dequantize_q5_0(data: &[u8], out: &mut [f32]) {
    let block_bytes = 22;
    let n_blocks = data.len() / block_bytes;
    let mut out_idx = 0;

    for i in 0..n_blocks {
        let off = i * block_bytes;
        let d = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        let qh = u32::from_le_bytes([data[off + 2], data[off + 3], data[off + 4], data[off + 5]]);
        for j in 0..16 {
            let qs = data[off + 6 + j];
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            let x0 = (((qs & 0x0F) as u32) | xh_0) as i32 - 16;
            let x1 = (((qs >> 4) as u32) | xh_1) as i32 - 16;
            out[out_idx + j] = (x0 as f32) * d;
            out[out_idx + j + 16] = (x1 as f32) * d;
        }
        out_idx += QK8_0;
    }
}

// ─── IQ4_XS dequantization ──────────────────────────────────────────────────
//
// IQ4_XS is a 4-bit importance-weighted quantization with 6-bit sub-block
// scales, commonly produced by unsloth/bartowski for smaller-than-Q4_K models.
//
// Block layout (136 bytes / QK_K=256 elements = 8 sub-blocks × 32 elements):
//   d:        f16                (offset 0..2)
//   scales_h: u16 little-endian  (offset 2..4)
//   scales_l: [u8; 4]            (offset 4..8)
//   qs:       [u8; 128] nibbles  (offset 8..136)
//
// Dequant per llama.cpp `dequantize_row_iq4_xs`:
//   for ib in 0..8:
//     ls  = ((scales_l[ib/2] >> (4*(ib%2))) & 0xF) | (((scales_h >> (2*ib)) & 3) << 4)
//     dl  = d * (ls - 32)         (6-bit scale offset by -32)
//     for j in 0..16:
//       y[ib*32 + j]      = dl * KVALUES_IQ4NL[qs[ib*16 + j] & 0xF]
//       y[ib*32 + j + 16] = dl * KVALUES_IQ4NL[qs[ib*16 + j] >> 4]
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

pub(crate) fn dequantize_iq4_xs(data: &[u8], out: &mut [f32]) {
    let block_bytes = 136;
    let n_blocks = data.len() / block_bytes;
    let mut out_idx = 0;

    for i in 0..n_blocks {
        let off = i * block_bytes;
        let d = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        let scales_h = u16::from_le_bytes([data[off + 2], data[off + 3]]);
        let scales_l = [data[off + 4], data[off + 5], data[off + 6], data[off + 7]];
        let qs = &data[off + 8..off + 8 + 128];

        for ib in 0..8 {
            let ls_lo = (scales_l[ib / 2] >> (4 * (ib as u8 & 1))) & 0x0F;
            let ls_hi = ((scales_h >> (2 * ib)) & 0x3) as u8;
            let ls = ((ls_hi << 4) | ls_lo) as i32;
            let dl = d * ((ls - 32) as f32);
            let qs_base = ib * 16;
            let out_base = out_idx + ib * 32;
            for j in 0..16 {
                let byte = qs[qs_base + j];
                let n0 = (byte & 0x0F) as usize;
                let n1 = (byte >> 4) as usize;
                out[out_base + j] = dl * f32::from(KVALUES_IQ4NL[n0]);
                out[out_base + j + 16] = dl * f32::from(KVALUES_IQ4NL[n1]);
            }
        }
        out_idx += QK_K;
    }
}

// ─── Q6_K dequantization ────────────────────────────────────────────────────

/// Public wrapper for external slice-based callers (Gemma 4 per-layer input
/// embedding lookup).
pub(crate) fn dequantize_q6_k_public(data: &[u8], out: &mut [f32]) {
    dequantize_q6_k(data, out);
}

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
        let mut sc_off = 0usize;

        for _n in (0..QK_K).step_by(128) {
            for l in 0..32 {
                let is = l / 16; // matches llama.cpp: is = l/16
                let q1 = ((ql[ql_off + l] & 0xF) | ((qh[qh_off + l] & 3) << 4)) as i8 - 32;
                let q2 =
                    ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 =
                    ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 - 32;

                out[out_idx + l] = d * f32::from(scales[sc_off + is] as i8) * f32::from(q1);
                out[out_idx + l + 32] =
                    d * f32::from(scales[sc_off + is + 2] as i8) * f32::from(q2);
                out[out_idx + l + 64] =
                    d * f32::from(scales[sc_off + is + 4] as i8) * f32::from(q3);
                out[out_idx + l + 96] =
                    d * f32::from(scales[sc_off + is + 6] as i8) * f32::from(q4);
            }
            out_idx += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
}

// ─── Q8_K quantization (matches llama.cpp) ──────────────────────────────────

/// Q8_K block: intermediate quantization format for input vectors.
/// Matches llama.cpp's `block_q8_K` exactly.
pub struct BlockQ8K {
    pub d: f32,           // scale factor (f32, not f16)
    pub qs: [i8; QK_K],   // 256 quantized values
    pub bsums: [i16; 16], // pre-computed sums of groups of 16
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

        // Match llama.cpp: iscale = 128 / max (signed max, not absolute)
        let iscale = 128.0f32 / max_val;

        for j in 0..QK_K {
            let v = nearest_int(iscale * x[j]);
            block.qs[j] = v.max(-128).min(127) as i8;
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

        // Reconstruct 6-bit signed quants using NEON.
        //
        // Q6_K packs 256 6-bit values into 128 ql bytes (4 low bits each,
        // two elements per byte) plus 64 qh bytes (2 high bits per element,
        // four elements per byte). Per iteration we unpack 128 elements from
        // 64 ql bytes + 32 qh bytes at a time, distributing into four output
        // quadrants of 32 elements each:
        //   Q0: aux8[  0.. 32] = (ql[ 0.. 32] & 0xF)   | ((qh << 4) & 0x30)
        //   Q1: aux8[ 32.. 64] = (ql[32.. 64] & 0xF)   | ((qh << 2) & 0x30)
        //   Q2: aux8[ 64.. 96] = (ql[ 0.. 32] >> 4)    | ( qh       & 0x30)
        //   Q3: aux8[ 96..128] = (ql[32.. 64] >> 4)    | ((qh >> 2) & 0x30)
        // Then subtract 32 to move the 6-bit unsigned value into the signed
        // range [-32, 31]. The shift-mask trick lets the same qh vector feed
        // all four quadrants with only add / and / or / shl / shr — no per-
        // element scalar packing. This replaces the previous scalar bit
        // packing loop (~14 µs of the old 89 µs NEON path on Mac M3).
        let mut aux8 = [0i8; QK_K];
        let mask_nib = vdupq_n_u8(0x0F);
        let mask_hi = vdupq_n_u8(0x30);
        let bias = vdupq_n_s8(-32);
        let mut a_off = 0usize;
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        for _ in 0..2 {
            let ql_a = vld1q_u8(ql.as_ptr().add(ql_off));
            let ql_b = vld1q_u8(ql.as_ptr().add(ql_off + 16));
            let ql_c = vld1q_u8(ql.as_ptr().add(ql_off + 32));
            let ql_d = vld1q_u8(ql.as_ptr().add(ql_off + 48));
            let qh_a = vld1q_u8(qh.as_ptr().add(qh_off));
            let qh_b = vld1q_u8(qh.as_ptr().add(qh_off + 16));

            // Quadrant 0: (ql[0..32] & 0xF) | ((qh << 4) & 0x30) - 32
            let hi_a_q0 = vandq_u8(vshlq_n_u8::<4>(qh_a), mask_hi);
            let hi_b_q0 = vandq_u8(vshlq_n_u8::<4>(qh_b), mask_hi);
            let out_0a = vaddq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_a, mask_nib), hi_a_q0)),
                bias,
            );
            let out_0b = vaddq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_b, mask_nib), hi_b_q0)),
                bias,
            );
            vst1q_s8(aux8.as_mut_ptr().add(a_off), out_0a);
            vst1q_s8(aux8.as_mut_ptr().add(a_off + 16), out_0b);

            // Quadrant 1: (ql[32..64] & 0xF) | ((qh << 2) & 0x30) - 32
            let hi_a_q1 = vandq_u8(vshlq_n_u8::<2>(qh_a), mask_hi);
            let hi_b_q1 = vandq_u8(vshlq_n_u8::<2>(qh_b), mask_hi);
            let out_1a = vaddq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_c, mask_nib), hi_a_q1)),
                bias,
            );
            let out_1b = vaddq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql_d, mask_nib), hi_b_q1)),
                bias,
            );
            vst1q_s8(aux8.as_mut_ptr().add(a_off + 32), out_1a);
            vst1q_s8(aux8.as_mut_ptr().add(a_off + 48), out_1b);

            // Quadrant 2: (ql[0..32] >> 4) | (qh & 0x30) - 32
            let hi_a_q2 = vandq_u8(qh_a, mask_hi);
            let hi_b_q2 = vandq_u8(qh_b, mask_hi);
            let out_2a = vaddq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(ql_a), hi_a_q2)),
                bias,
            );
            let out_2b = vaddq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(ql_b), hi_b_q2)),
                bias,
            );
            vst1q_s8(aux8.as_mut_ptr().add(a_off + 64), out_2a);
            vst1q_s8(aux8.as_mut_ptr().add(a_off + 80), out_2b);

            // Quadrant 3: (ql[32..64] >> 4) | ((qh >> 2) & 0x30) - 32
            let hi_a_q3 = vandq_u8(vshrq_n_u8::<2>(qh_a), mask_hi);
            let hi_b_q3 = vandq_u8(vshrq_n_u8::<2>(qh_b), mask_hi);
            let out_3a = vaddq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(ql_c), hi_a_q3)),
                bias,
            );
            let out_3b = vaddq_s8(
                vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(ql_d), hi_b_q3)),
                bias,
            );
            vst1q_s8(aux8.as_mut_ptr().add(a_off + 96), out_3a);
            vst1q_s8(aux8.as_mut_ptr().add(a_off + 112), out_3b);

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

    /// Fused Q8_0 dequantize + f32 matvec for one row using NEON.
    ///
    /// Q8_0 stores 32 signed i8 weights per block plus a single f16 scale, and
    /// — unlike the K-series — the ALICE-LLM dispatch feeds it raw f32 inputs
    /// rather than a pre-quantised Q8_K block. That makes the SIMD path an f32
    /// FMA loop rather than an integer madd chain, mirroring the AVX2
    /// `avx2_dot::q8_0_matvec_row`.
    ///
    /// Per block of 32 elements we perform two 16-lane halves. Each half
    /// widens 16 i8 → i32 (two `vmovl` chains) → f32, multiplies by the block
    /// scale, and does four 4-lane FMAs against the input. The accumulator is
    /// horizontally summed at the end with `vaddvq_f32` (aarch64 baseline).
    #[target_feature(enable = "neon")]
    pub unsafe fn q8_0_matvec_row(input: &[f32], row_data: &[u8], cols: usize) -> f32 {
        let blocks_per_row = cols / QK8_0;
        let block_bytes = 34usize;
        let mut acc: float32x4_t = vdupq_n_f32(0.0);

        for bi in 0..blocks_per_row {
            let off = bi * block_bytes;
            let d = f16_to_f32(u16::from_le_bytes([row_data[off], row_data[off + 1]]));
            let d_bcast = vdupq_n_f32(d);
            let qs_ptr = row_data.as_ptr().add(off + 2).cast::<i8>();
            let col_base = bi * QK8_0;
            let in_ptr = input.as_ptr().add(col_base);

            // First half: bytes 0..15, inputs 0..15.
            let a0 = vld1q_s8(qs_ptr);
            let a0_lo = vmovl_s8(vget_low_s8(a0));
            let a0_hi = vmovl_s8(vget_high_s8(a0));
            let f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0_lo))), d_bcast);
            let f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(a0_lo))), d_bcast);
            let f2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0_hi))), d_bcast);
            let f3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(a0_hi))), d_bcast);
            acc = vfmaq_f32(acc, f0, vld1q_f32(in_ptr));
            acc = vfmaq_f32(acc, f1, vld1q_f32(in_ptr.add(4)));
            acc = vfmaq_f32(acc, f2, vld1q_f32(in_ptr.add(8)));
            acc = vfmaq_f32(acc, f3, vld1q_f32(in_ptr.add(12)));

            // Second half: bytes 16..31, inputs 16..31.
            let a1 = vld1q_s8(qs_ptr.add(16));
            let a1_lo = vmovl_s8(vget_low_s8(a1));
            let a1_hi = vmovl_s8(vget_high_s8(a1));
            let g0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a1_lo))), d_bcast);
            let g1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(a1_lo))), d_bcast);
            let g2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a1_hi))), d_bcast);
            let g3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(a1_hi))), d_bcast);
            acc = vfmaq_f32(acc, g0, vld1q_f32(in_ptr.add(16)));
            acc = vfmaq_f32(acc, g1, vld1q_f32(in_ptr.add(20)));
            acc = vfmaq_f32(acc, g2, vld1q_f32(in_ptr.add(24)));
            acc = vfmaq_f32(acc, g3, vld1q_f32(in_ptr.add(28)));
        }

        vaddvq_f32(acc)
    }

    /// Ternary bitmask × f32 input dot product for one row using NEON.
    ///
    /// Mirrors the AVX2 kernel `avx2_dot::ternary_dot_row_unscaled`. Per byte
    /// of `pos_mask` / `neg_mask` we broadcast the byte into a `u32x4` lane,
    /// AND it against the two 4-lane bit masks `[1, 2, 4, 8]` and
    /// `[16, 32, 64, 128]`, and `vceqq_u32` against the same bit mask. That
    /// yields all-ones per lane where the bit was set and 0 elsewhere; the
    /// bitwise AND with the reinterpreted f32 input lane produces `input[i]`
    /// or `0.0` — no branching, no per-lane multiplies.
    ///
    /// Returns the **unscaled** positive_sum − negative_sum; the caller
    /// multiplies by `row.scale` (bit-exact with the AVX2 / scalar callers).
    #[target_feature(enable = "neon")]
    pub unsafe fn ternary_dot_row_unscaled(
        pos_mask: &[u8],
        neg_mask: &[u8],
        input: &[f32],
        num_cols: usize,
    ) -> f32 {
        let bit_lanes_lo: uint32x4_t = vld1q_u32([0x01u32, 0x02, 0x04, 0x08].as_ptr());
        let bit_lanes_hi: uint32x4_t = vld1q_u32([0x10u32, 0x20, 0x40, 0x80].as_ptr());
        let mut pos_acc: float32x4_t = vdupq_n_f32(0.0);
        let mut neg_acc: float32x4_t = vdupq_n_f32(0.0);

        let full_bytes = num_cols / 8;
        for b in 0..full_bytes {
            let pm = u32::from(pos_mask[b]);
            let nm = u32::from(neg_mask[b]);
            if (pm | nm) == 0 {
                continue;
            }
            let in_ptr = input.as_ptr().add(b * 8);
            let in_lo = vld1q_f32(in_ptr);
            let in_hi = vld1q_f32(in_ptr.add(4));

            if pm != 0 {
                let bcast = vdupq_n_u32(pm);
                let is_set_lo = vceqq_u32(vandq_u32(bcast, bit_lanes_lo), bit_lanes_lo);
                let is_set_hi = vceqq_u32(vandq_u32(bcast, bit_lanes_hi), bit_lanes_hi);
                let contrib_lo = vandq_u32(is_set_lo, vreinterpretq_u32_f32(in_lo));
                let contrib_hi = vandq_u32(is_set_hi, vreinterpretq_u32_f32(in_hi));
                pos_acc = vaddq_f32(pos_acc, vreinterpretq_f32_u32(contrib_lo));
                pos_acc = vaddq_f32(pos_acc, vreinterpretq_f32_u32(contrib_hi));
            }
            if nm != 0 {
                let bcast = vdupq_n_u32(nm);
                let is_set_lo = vceqq_u32(vandq_u32(bcast, bit_lanes_lo), bit_lanes_lo);
                let is_set_hi = vceqq_u32(vandq_u32(bcast, bit_lanes_hi), bit_lanes_hi);
                let contrib_lo = vandq_u32(is_set_lo, vreinterpretq_u32_f32(in_lo));
                let contrib_hi = vandq_u32(is_set_hi, vreinterpretq_u32_f32(in_hi));
                neg_acc = vaddq_f32(neg_acc, vreinterpretq_f32_u32(contrib_lo));
                neg_acc = vaddq_f32(neg_acc, vreinterpretq_f32_u32(contrib_hi));
            }
        }

        let mut pos_sum = vaddvq_f32(pos_acc);
        let mut neg_sum = vaddvq_f32(neg_acc);

        // Tail: elements in the final byte that fall past `num_cols`.
        let tail_start = full_bytes * 8;
        if tail_start < num_cols {
            let last_b = full_bytes;
            let pm = pos_mask[last_b];
            let nm = neg_mask[last_b];
            let remaining = num_cols - tail_start;
            for bit in 0..remaining {
                let idx = tail_start + bit;
                let p = f32::from((pm >> bit) & 1);
                let n = f32::from((nm >> bit) & 1);
                pos_sum += p * input[idx];
                neg_sum += n * input[idx];
            }
        }

        pos_sum - neg_sum
    }

    #[inline]
    fn unpack_scales_mins_q4k(scale_bytes: &[u8]) -> ([u8; 8], [u8; 8]) {
        const KMASK1: u32 = 0x3f3f_3f3f;
        const KMASK2: u32 = 0x0f0f_0f0f;
        const KMASK3: u32 = 0x0303_0303;

        let mut utmp = [0u32; 4];
        utmp[0] = u32::from_le_bytes([
            scale_bytes[0],
            scale_bytes[1],
            scale_bytes[2],
            scale_bytes[3],
        ]);
        utmp[1] = u32::from_le_bytes([
            scale_bytes[4],
            scale_bytes[5],
            scale_bytes[6],
            scale_bytes[7],
        ]);
        utmp[2] = u32::from_le_bytes([
            scale_bytes[8],
            scale_bytes[9],
            scale_bytes[10],
            scale_bytes[11],
        ]);

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

// ─── x86_64 AVX2 SIMD dot products (Issue #13) ──────────────────────────────
//
// Mirrors the `neon_dot` module for x86_64 targets so Q4_K / Q6_K matvec
// gets SIMD acceleration on Ryzen / Xeon in addition to Apple Silicon.
// Runtime-dispatched through `is_x86_feature_detected!("avx2")` so binaries
// built without `-C target-cpu=native` still ship a scalar fallback for
// pre-AVX2 machines (Ivy Bridge and older).
//
// The core trick — reused from the JustVugg/colibri engine's `dot_i8i8` —
// is `_mm256_maddubs_epi16`, which multiplies **unsigned × signed** i8
// pairs. When both operands are signed (Q6_K case), pre-processing with
// `_mm256_sign_epi8(w, w)` yields `|w|` (unsigned) and `sign_epi8(x, w)`
// applies `sign(w)` to `x` — the product is then the original signed
// dot. Q4_K nibbles are natively unsigned `[0, 15]`, so the sign trick
// is unnecessary there.
//
// Reference: JustVugg/colibri `c/glm.c` (see PR body for background).

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
mod avx2_dot {
    use super::*;
    use std::arch::x86_64::*;

    /// Horizontal sum of `__m256i` interpreted as 8 lanes of i32.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_i32_avx2(v: __m256i) -> i32 {
        // SAFETY: caller guarantees AVX2 support.
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);
        let sum128 = _mm_add_epi32(lo, hi);
        let sh = _mm_shuffle_epi32(sum128, 0b1110);
        let sum64 = _mm_add_epi32(sum128, sh);
        let sh2 = _mm_shufflelo_epi16(sum64, 0b1110);
        let sum32 = _mm_add_epi32(sum64, sh2);
        _mm_cvtsi128_si32(sum32)
    }

    /// AVX2 signed dot product of 32 i8 values, seeded into an accumulator.
    ///
    /// Uses the `maddubs` sign trick: multiplies |w| (unsigned) by
    /// `sign(w) * x` (signed) to reuse the u8×i8 instruction for a
    /// signed×signed dot. Adds the resulting 8 lanes of i32 to the caller's
    /// running accumulator so multi-block loops can chain without spilling
    /// to memory.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_32_i8_signed(acc: __m256i, w: *const i8, x: *const i8) -> __m256i {
        let wv = _mm256_loadu_si256(w.cast());
        let xv = _mm256_loadu_si256(x.cast());
        // |w| (unsigned) × sign(w) * x (signed) via the u8×i8 madd.
        let w_abs = _mm256_sign_epi8(wv, wv);
        let x_signed = _mm256_sign_epi8(xv, wv);
        let prod16 = _mm256_maddubs_epi16(w_abs, x_signed);
        let ones = _mm256_set1_epi16(1);
        let prod32 = _mm256_madd_epi16(prod16, ones);
        _mm256_add_epi32(acc, prod32)
    }

    /// AVX2 dot product of 32 unsigned nibbles × 32 signed i8 values.
    ///
    /// Q4_K stores four-bit values in `[0, 15]`, so no sign trick is
    /// needed — `maddubs` natively consumes unsigned × signed. Adds into
    /// a running i32 accumulator so 32-element strides can be chained.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_32_u4_i8_from_nibbles(acc: __m256i, nib32: __m256i, x: *const i8) -> __m256i {
        let xv = _mm256_loadu_si256(x.cast());
        let prod16 = _mm256_maddubs_epi16(nib32, xv);
        let ones = _mm256_set1_epi16(1);
        let prod32 = _mm256_madd_epi16(prod16, ones);
        _mm256_add_epi32(acc, prod32)
    }

    /// Unpack 16 packed nibbles (`q4[0..16]`) into 32 unsigned bytes `[0, 15]`
    /// so `maddubs` can multiply against a matching 32-lane q8 slice.
    ///
    /// Low nibbles occupy the first 16 lanes, high nibbles the next 16 — the
    /// same order the scalar `q4k_q8k_dot_scalar` deposits into `aux8` when
    /// iterating each sub-block group.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn unpack_16_q4_nibbles(q4: *const u8) -> __m256i {
        let bytes = _mm_loadu_si128(q4.cast());
        let mask = _mm_set1_epi8(0x0F);
        let lo = _mm_and_si128(bytes, mask);
        let hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
        // Concatenate low nibbles (lanes 0..15) and high nibbles (16..31).
        _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1)
    }

    /// Q4_K × Q8_K dot product using AVX2.
    ///
    /// Reproduces the arithmetic of `q4k_q8k_dot_scalar` (block layout
    /// documented on that function) using SIMD:
    /// 1. Reconstruct 8 sub-block scales + 8 mins from the 12-byte packed
    ///    scales/mins header (identical to the NEON path).
    /// 2. For each of 8 sub-blocks: unpack 32 nibbles, multiply against
    ///    32 q8 signed activations via `maddubs`, sum to i32, weight by
    ///    the sub-block's i8 scale.
    /// 3. Return `d * q8k.d * total - dmin * q8k.d * sumi` — the same
    ///    formula the scalar path emits.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn q4k_q8k_dot(q4k_block: &[u8], q8k: &BlockQ8K) -> f32 {
        const KMASK1: u32 = 0x3f3f_3f3f;
        const KMASK2: u32 = 0x0f0f_0f0f;
        const KMASK3: u32 = 0x0303_0303;

        let d = f16_to_f32(u16::from_le_bytes([q4k_block[0], q4k_block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([q4k_block[2], q4k_block[3]]));
        let q4 = &q4k_block[16..144];

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
        let s0 = utmp[0].to_le_bytes();
        let s1 = utmp[1].to_le_bytes();
        let m0 = utmp[2].to_le_bytes();
        let m1 = utmp[3].to_le_bytes();
        let scales = [s0[0], s0[1], s0[2], s0[3], s1[0], s1[1], s1[2], s1[3]];
        let mins = [m0[0], m0[1], m0[2], m0[3], m1[0], m1[1], m1[2], m1[3]];

        // Bias correction: sum over 16 half-sub-blocks of `bsum[j] * min[j/2]`.
        let mut sumi = 0i32;
        for j in 0..16 {
            sumi += q8k.bsums[j] as i32 * mins[j / 2] as i32;
        }

        // Per-sub-block SIMD dot × scale, accumulated as i32.
        //
        // Layout note: the scalar path packs `aux8` so that sub-block `2k`
        // is the **low nibbles** of `q4[k*32 : k*32+32]` and sub-block
        // `2k+1` is the **high nibbles** of the same 32 packed bytes.
        // Iterating on `q4[is*16 : is*16+16]` per sub-block reads the
        // wrong window entirely (initial implementation bug caught by CI),
        // so we stride by 32 bytes (64 nibbles) and produce two 32-lane
        // dots per iteration.
        let mut total: i32 = 0;
        for g in 0..4 {
            let q4_ptr = q4.as_ptr().add(g * 32);
            // Load 32 packed bytes = 64 nibbles.
            let bytes = _mm256_loadu_si256(q4_ptr.cast());
            let mask = _mm256_set1_epi8(0x0F);
            let lo_nibbles = _mm256_and_si256(bytes, mask);
            // `_mm256_srli_epi16` shifts each u16 lane by 4; after masking
            // with 0x0F the surviving bits are exactly the per-byte high
            // nibbles.
            let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), mask);

            // Sub-block 2g: low nibbles × q8[g*64 .. g*64+32], scaled by scales[2g].
            let q8_ptr_a = q8k.qs.as_ptr().add(g * 64);
            let acc_a = _mm256_setzero_si256();
            let acc_a = dot_32_u4_i8_from_nibbles(acc_a, lo_nibbles, q8_ptr_a);
            let dot_a = hsum_i32_avx2(acc_a);
            total += scales[2 * g] as i32 * dot_a;

            // Sub-block 2g+1: high nibbles × q8[g*64+32 .. g*64+64].
            let q8_ptr_b = q8k.qs.as_ptr().add(g * 64 + 32);
            let acc_b = _mm256_setzero_si256();
            let acc_b = dot_32_u4_i8_from_nibbles(acc_b, hi_nibbles, q8_ptr_b);
            let dot_b = hsum_i32_avx2(acc_b);
            total += scales[2 * g + 1] as i32 * dot_b;
        }

        d * q8k.d * total as f32 - dmin * q8k.d * sumi as f32
    }

    /// AVX2 signed dot product of 16 i8 values via sign-extension.
    ///
    /// Widens each i8 to i16 first, then relies on `_mm256_madd_epi16`
    /// (which multiplies i16 pairs and sums adjacent lanes to i32 with
    /// **no saturation**). Slightly more expensive than the `maddubs`
    /// sign trick but immune to the i16 intermediate saturation risk
    /// for signed×signed dots, and its per-sub-block granularity matches
    /// the scalar Q6_K layout exactly.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_16_i8_widened(acc: __m256i, w: *const i8, x: *const i8) -> __m256i {
        let w16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.cast()));
        let x16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(x.cast()));
        let prod32 = _mm256_madd_epi16(w16, x16);
        _mm256_add_epi32(acc, prod32)
    }

    /// Q6_K × Q8_K dot product using AVX2.
    ///
    /// Reconstructs 6-bit values from the low 4 bits (`ql`) and top 2 bits
    /// (`qh`) via vectorised bit unpacking (mirrors the NEON kernel — see
    /// `neon_dot::q6k_q8k_dot`), then does 16-element sub-block dots with
    /// `dot_16_i8_widened`.
    ///
    /// Per outer iteration (2 iters cover the 256-element block) we load 64
    /// ql bytes as two 32-byte AVX2 vectors plus 32 qh bytes as one vector,
    /// then produce four 32-byte quadrants of aux8 via shift/mask/or:
    ///   Q0: (ql_lo & 0xF) | ((qh << 4) & 0x30) - 32
    ///   Q1: (ql_hi & 0xF) | ((qh << 2) & 0x30) - 32
    ///   Q2: (ql_lo >> 4)  | ( qh       & 0x30) - 32
    ///   Q3: (ql_hi >> 4)  | ((qh >> 2) & 0x30) - 32
    ///
    /// AVX2 lacks a native 8-bit shift, so we shift as u16 and rely on the
    /// `0x30` / `0x0F` byte mask to discard any bits that leaked across the
    /// byte boundary — those bits land outside the target byte positions
    /// (bits 4-5 for the high nibble, bits 0-3 for the low nibble) and are
    /// therefore killed by the mask.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn q6k_q8k_dot(q6k_block: &[u8], q8k: &BlockQ8K) -> f32 {
        let ql = &q6k_block[0..128];
        let qh = &q6k_block[128..192];
        let scales = &q6k_block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([q6k_block[208], q6k_block[209]]));

        // Reconstruct 256 signed i8 values on the stack. The layout matches
        // `q6k_q8k_dot_scalar`: 16 sub-blocks of 16 elements each.
        let mut aux8 = [0i8; QK_K];
        let mask_nib = _mm256_set1_epi8(0x0F);
        let mask_hi = _mm256_set1_epi8(0x30);
        let bias = _mm256_set1_epi8(-32);
        let mut a_off = 0usize;
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        for _ in 0..2 {
            let ql_lo = _mm256_loadu_si256(ql.as_ptr().add(ql_off).cast());
            let ql_hi = _mm256_loadu_si256(ql.as_ptr().add(ql_off + 32).cast());
            let qh_v = _mm256_loadu_si256(qh.as_ptr().add(qh_off).cast());

            // Quadrant 0: (ql_lo & 0xF) | ((qh << 4) & 0x30) - 32
            let hi0 = _mm256_and_si256(_mm256_slli_epi16(qh_v, 4), mask_hi);
            let q0 = _mm256_add_epi8(
                _mm256_or_si256(_mm256_and_si256(ql_lo, mask_nib), hi0),
                bias,
            );
            _mm256_storeu_si256(aux8.as_mut_ptr().add(a_off).cast(), q0);

            // Quadrant 1: (ql_hi & 0xF) | ((qh << 2) & 0x30) - 32
            let hi1 = _mm256_and_si256(_mm256_slli_epi16(qh_v, 2), mask_hi);
            let q1 = _mm256_add_epi8(
                _mm256_or_si256(_mm256_and_si256(ql_hi, mask_nib), hi1),
                bias,
            );
            _mm256_storeu_si256(aux8.as_mut_ptr().add(a_off + 32).cast(), q1);

            // Quadrant 2: (ql_lo >> 4) | (qh & 0x30) - 32
            let hi2 = _mm256_and_si256(qh_v, mask_hi);
            let q2 = _mm256_add_epi8(
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql_lo, 4), mask_nib), hi2),
                bias,
            );
            _mm256_storeu_si256(aux8.as_mut_ptr().add(a_off + 64).cast(), q2);

            // Quadrant 3: (ql_hi >> 4) | ((qh >> 2) & 0x30) - 32
            let hi3 = _mm256_and_si256(_mm256_srli_epi16(qh_v, 2), mask_hi);
            let q3 = _mm256_add_epi8(
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql_hi, 4), mask_nib), hi3),
                bias,
            );
            _mm256_storeu_si256(aux8.as_mut_ptr().add(a_off + 96).cast(), q3);

            a_off += 128;
            ql_off += 64;
            qh_off += 32;
        }

        // Q6_K has 16 sub-blocks of 16 elements — perfectly matched to a
        // 16-lane SIMD dot after i8→i16 sign extension.
        let mut total: i32 = 0;
        for is in 0..16 {
            let a_ptr = aux8.as_ptr().add(is * 16);
            let q8_ptr = q8k.qs.as_ptr().add(is * 16);
            let acc0 = _mm256_setzero_si256();
            let acc = dot_16_i8_widened(acc0, a_ptr, q8_ptr);
            let sub_dot = hsum_i32_avx2(acc);
            total += scales[is] as i8 as i32 * sub_dot;
        }

        d * q8k.d * total as f32
    }

    /// AVX2 dot of 32 unsigned u8 values × 32 signed i8 values.
    ///
    /// Sign-extends both operands to i16 first (u8 has a natural 0-padded
    /// promotion, so `_mm256_cvtepu8_epi16` is safe) and uses
    /// `_mm256_madd_epi16` to accumulate into i32 without saturation.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_32_u8_i8_widened(acc: __m256i, u: __m256i, x: *const i8) -> __m256i {
        let xv = _mm256_loadu_si256(x.cast());
        // Split each 32-lane u8 vector into two 128-bit halves, then
        // zero-extend to 16 x i16 per half so subsequent multiplies stay
        // in i16-safe range and the following `madd_epi16` produces
        // saturation-free i32 sums.
        let u_lo128 = _mm256_castsi256_si128(u);
        let u_hi128 = _mm256_extracti128_si256(u, 1);
        let x_lo128 = _mm256_castsi256_si128(xv);
        let x_hi128 = _mm256_extracti128_si256(xv, 1);
        let u_lo = _mm256_cvtepu8_epi16(u_lo128);
        let u_hi = _mm256_cvtepu8_epi16(u_hi128);
        let x_lo = _mm256_cvtepi8_epi16(x_lo128);
        let x_hi = _mm256_cvtepi8_epi16(x_hi128);
        let sum_lo = _mm256_madd_epi16(u_lo, x_lo);
        let sum_hi = _mm256_madd_epi16(u_hi, x_hi);
        _mm256_add_epi32(acc, _mm256_add_epi32(sum_lo, sum_hi))
    }

    /// Q5_K × Q8_K dot product using AVX2.
    ///
    /// Q5_K packs each 5-bit value as `(qs nibble) | (qh bit << 4)` so the
    /// per-element range is `[0, 31]` — still unsigned, so no sign trick
    /// is needed. Four groups (`im=0..4`), each processing 64 elements:
    /// low nibbles + qh bit `2*im` fill even sub-block, high nibbles +
    /// qh bit `2*im+1` fill odd sub-block. The single 32-byte qh load is
    /// reused across all four `im` iterations by shifting/masking to
    /// isolate the target bit.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn q5k_q8k_dot(q5k_block: &[u8], q8k: &BlockQ8K) -> f32 {
        let d = f16_to_f32(u16::from_le_bytes([q5k_block[0], q5k_block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([q5k_block[2], q5k_block[3]]));
        let scales_raw = &q5k_block[4..16];
        let qh = &q5k_block[16..48];
        let qs = &q5k_block[48..176];

        // 32-byte qh loaded once and reused across all four `im` iterations —
        // each iteration selects a different bit position via `srli_epi16`
        // + `and` (the u16-lane shift is safe because the AND with 0x01
        // masks off any bits that leaked from the neighbouring byte).
        let qh_v = _mm256_loadu_si256(qh.as_ptr().cast());
        let mask_lo4 = _mm256_set1_epi8(0x0F);
        let mask_bit0 = _mm256_set1_epi8(1);

        let mut sumf = 0.0f32;
        for im in 0..4u8 {
            let (sc1, m1) = get_scale_min_k4((im as usize) * 2, scales_raw);
            let (sc2, m2) = get_scale_min_k4((im as usize) * 2 + 1, scales_raw);

            // 32 packed qs bytes = 64 nibbles for this `im` group.
            let qs_group = _mm256_loadu_si256(qs.as_ptr().add((im as usize) * 32).cast());
            let lo_nibbles = _mm256_and_si256(qs_group, mask_lo4);
            let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(qs_group, 4), mask_lo4);

            // Runtime-variable shifts (`_mm256_srli_epi16` requires a
            // compile-time immediate, so we use the sibling `_srl_` form
            // which takes a `__m128i` shift amount whose low 64 bits
            // control every lane).
            let shift_even = _mm_cvtsi32_si128(i32::from(im) * 2);
            let shift_odd = _mm_cvtsi32_si128(i32::from(im) * 2 + 1);

            // Even sub-block: qh bit `2*im` broadcasted into the top nibble.
            let hbits_even = _mm256_and_si256(_mm256_srl_epi16(qh_v, shift_even), mask_bit0);
            let hbits_even_hi = _mm256_slli_epi16(hbits_even, 4);
            let q_even = _mm256_or_si256(lo_nibbles, hbits_even_hi);

            // Odd sub-block: qh bit `2*im + 1`.
            let hbits_odd = _mm256_and_si256(_mm256_srl_epi16(qh_v, shift_odd), mask_bit0);
            let hbits_odd_hi = _mm256_slli_epi16(hbits_odd, 4);
            let q_odd = _mm256_or_si256(hi_nibbles, hbits_odd_hi);

            // Dot products against 64 q8 activations (32 for each sub-block).
            let q8_ptr_a = q8k.qs.as_ptr().add((im as usize) * 64);
            let acc_a = _mm256_setzero_si256();
            let acc_a = dot_32_u8_i8_widened(acc_a, q_even, q8_ptr_a);
            let dot_a = hsum_i32_avx2(acc_a);

            let q8_ptr_b = q8k.qs.as_ptr().add((im as usize) * 64 + 32);
            let acc_b = _mm256_setzero_si256();
            let acc_b = dot_32_u8_i8_widened(acc_b, q_odd, q8_ptr_b);
            let dot_b = hsum_i32_avx2(acc_b);

            // The bias-correction `sum_i = Σ q8[i]` for each 32-element
            // sub-block reuses the pre-computed `bsums[j]` (each covers
            // 16 q8 values, so two consecutive entries make a 32-lane sum).
            let sum1 =
                q8k.bsums[(im as usize) * 4] as i32 + q8k.bsums[(im as usize) * 4 + 1] as i32;
            let sum2 =
                q8k.bsums[(im as usize) * 4 + 2] as i32 + q8k.bsums[(im as usize) * 4 + 3] as i32;

            sumf += d * sc1 as f32 * dot_a as f32 - dmin * m1 as f32 * sum1 as f32;
            sumf += d * sc2 as f32 * dot_b as f32 - dmin * m2 as f32 * sum2 as f32;
        }
        sumf * q8k.d
    }

    /// Fused Q8_0 dequantize + f32 matvec for a whole row using AVX2.
    ///
    /// Q8_0 stores 32 signed i8 weights per block plus a single f16 scale,
    /// and — unlike the K-series — the ALICE-LLM dispatch feeds it raw f32
    /// inputs rather than a pre-quantised Q8_K block. That makes the SIMD
    /// path an f32 FMA loop (i8 → f32 dequant, multiply by the row's f32
    /// input, FMA into an accumulator) rather than an integer madd chain.
    ///
    /// Per block of 32 elements, four 8-lane iterations consume the whole
    /// block; the accumulator is horizontally summed at the end to yield
    /// the row's dot product.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn q8_0_matvec_row(input: &[f32], row_data: &[u8], cols: usize) -> f32 {
        let blocks_per_row = cols / super::QK8_0;
        let block_bytes = 34usize;
        let mut acc = _mm256_setzero_ps();
        for bi in 0..blocks_per_row {
            let off = bi * block_bytes;
            let d = super::f16_to_f32(u16::from_le_bytes([row_data[off], row_data[off + 1]]));
            let d_bcast = _mm256_set1_ps(d);
            let qs_ptr = row_data.as_ptr().add(off + 2).cast::<i8>();
            let col_base = bi * super::QK8_0;
            // 4 × 8-lane iterations = 32 elements per block.
            for k in 0..4 {
                let q8 = _mm_loadl_epi64(qs_ptr.add(k * 8).cast::<__m128i>());
                let q32 = _mm256_cvtepi8_epi32(q8);
                let qf = _mm256_cvtepi32_ps(q32);
                let scaled = _mm256_mul_ps(qf, d_bcast);
                let x = _mm256_loadu_ps(input.as_ptr().add(col_base + k * 8));
                acc = _mm256_fmadd_ps(scaled, x, acc);
            }
        }
        // Horizontal sum of 8 f32 lanes.
        let lo = _mm256_castps256_ps128(acc);
        let hi = _mm256_extractf128_ps(acc, 1);
        let s128 = _mm_add_ps(lo, hi);
        let sh = _mm_movehl_ps(s128, s128);
        let s64 = _mm_add_ps(s128, sh);
        let sh2 = _mm_shuffle_ps(s64, s64, 0x55);
        let s32 = _mm_add_ss(s64, sh2);
        _mm_cvtss_f32(s32)
    }

    /// Ternary bitmask × f32 input dot product for one row (AVX2).
    ///
    /// Each byte of `pos_mask` / `neg_mask` encodes 8 lanes. The 32-bit
    /// broadcast + per-lane AND with the `[0x01, 0x02, 0x04, 0x08, 0x10,
    /// 0x20, 0x40, 0x80]` pattern extracts one bit per lane; comparing
    /// against the same pattern yields an all-ones i32 (i.e. the f32
    /// mask `-NaN`-like bit pattern used as an AND-mask) where the bit
    /// was set. `_mm256_and_ps` with the input then produces `input[i]`
    /// or `0.0` — no branchy blending needed.
    ///
    /// The scalar dot the SIMD path reproduces:
    ///
    /// ```text
    /// sum = Σ_b Σ_{bit ∈ pos_mask[b]} input[b*8 + bit]
    ///     - Σ_b Σ_{bit ∈ neg_mask[b]} input[b*8 + bit]
    /// ```
    ///
    /// Returns the unscaled `pos_sum - neg_sum` — the caller applies the
    /// row's f32 scale afterwards so both dispatch arms stay identical.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn ternary_dot_row_unscaled(
        pos_mask: &[u8],
        neg_mask: &[u8],
        input: &[f32],
        num_cols: usize,
    ) -> f32 {
        // 8-lane bit-mask constants: lane i tests bit `1 << i` of the byte.
        let bit_lanes = _mm256_setr_epi32(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);
        let mut pos_acc = _mm256_setzero_ps();
        let mut neg_acc = _mm256_setzero_ps();

        // Cover the fully-populated bytes with 8 lanes each. Any tail that
        // spans an incomplete final byte is finished by the scalar loop
        // below (edge case: `num_cols % 8 != 0`).
        let full_bytes = num_cols / 8;
        for b in 0..full_bytes {
            let pm = u32::from(pos_mask[b]);
            let nm = u32::from(neg_mask[b]);
            if (pm | nm) == 0 {
                continue;
            }
            let input_v = _mm256_loadu_ps(input.as_ptr().add(b * 8));

            // Positive contributions.
            if pm != 0 {
                let bcast = _mm256_set1_epi32(pm as i32);
                let masked = _mm256_and_si256(bcast, bit_lanes);
                let is_set = _mm256_cmpeq_epi32(masked, bit_lanes);
                let contrib = _mm256_and_ps(_mm256_castsi256_ps(is_set), input_v);
                pos_acc = _mm256_add_ps(pos_acc, contrib);
            }
            // Negative contributions.
            if nm != 0 {
                let bcast = _mm256_set1_epi32(nm as i32);
                let masked = _mm256_and_si256(bcast, bit_lanes);
                let is_set = _mm256_cmpeq_epi32(masked, bit_lanes);
                let contrib = _mm256_and_ps(_mm256_castsi256_ps(is_set), input_v);
                neg_acc = _mm256_add_ps(neg_acc, contrib);
            }
        }

        // Horizontal sums for the SIMD accumulators.
        let hsum = |v: __m256| -> f32 {
            let lo = _mm256_castps256_ps128(v);
            let hi = _mm256_extractf128_ps(v, 1);
            let s = _mm_add_ps(lo, hi);
            let sh = _mm_movehl_ps(s, s);
            let s2 = _mm_add_ps(s, sh);
            let sh2 = _mm_shuffle_ps(s2, s2, 0x55);
            let s3 = _mm_add_ss(s2, sh2);
            _mm_cvtss_f32(s3)
        };
        let mut pos_sum = hsum(pos_acc);
        let mut neg_sum = hsum(neg_acc);

        // Tail: elements in the final byte that fall past `num_cols`.
        let tail_start = full_bytes * 8;
        if tail_start < num_cols {
            let last_b = full_bytes;
            let pm = pos_mask[last_b];
            let nm = neg_mask[last_b];
            let remaining = num_cols - tail_start;
            for bit in 0..remaining {
                let idx = tail_start + bit;
                let p = f32::from((pm >> bit) & 1);
                let n = f32::from((nm >> bit) & 1);
                pos_sum += p * input[idx];
                neg_sum += n * input[idx];
            }
        }

        pos_sum - neg_sum
    }
}

// ─── x86_64 AVX-512 SIMD dot products (Issue #13) ───────────────────────────
//
// Same algorithm as AVX2 but processes 64 bytes per iteration instead of 32.
// Gated behind `is_x86_feature_detected!("avx512bw")` at dispatch because
// AVX-512BW is what supplies `_mm512_maddubs_epi16` and `_mm512_madd_epi16`;
// mere AVX-512F is insufficient.

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
mod avx512_dot {
    use super::*;
    use std::arch::x86_64::*;

    /// AVX-512 dot product of 64 unsigned nibbles × 64 signed i8 values.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn dot_64_u4_i8_from_nibbles(acc: __m512i, nib64: __m512i, x: *const i8) -> __m512i {
        let xv = _mm512_loadu_si512(x.cast());
        let prod16 = _mm512_maddubs_epi16(nib64, xv);
        let ones = _mm512_set1_epi16(1);
        let prod32 = _mm512_madd_epi16(prod16, ones);
        _mm512_add_epi32(acc, prod32)
    }

    /// Unpack 32 packed nibbles (`q4[0..32]`) into 64 unsigned bytes `[0, 15]`.
    /// Low nibbles fill lanes 0..31, high nibbles fill 32..63 — matches the
    /// scalar `aux8` layout across two 32-lane sub-blocks.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    unsafe fn unpack_32_q4_nibbles(q4: *const u8) -> __m512i {
        let bytes = _mm256_loadu_si256(q4.cast());
        let mask = _mm256_set1_epi8(0x0F);
        let lo = _mm256_and_si256(bytes, mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), mask);
        _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1)
    }

    /// Q4_K × Q8_K dot product using AVX-512BW. Processes 2 sub-blocks per
    /// iteration (64 lanes = 2 × 32-element Q4_K sub-blocks).
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub(super) unsafe fn q4k_q8k_dot(q4k_block: &[u8], q8k: &BlockQ8K) -> f32 {
        const KMASK1: u32 = 0x3f3f_3f3f;
        const KMASK2: u32 = 0x0f0f_0f0f;
        const KMASK3: u32 = 0x0303_0303;

        let d = f16_to_f32(u16::from_le_bytes([q4k_block[0], q4k_block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([q4k_block[2], q4k_block[3]]));
        let q4 = &q4k_block[16..144];

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
        let s0 = utmp[0].to_le_bytes();
        let s1 = utmp[1].to_le_bytes();
        let m0 = utmp[2].to_le_bytes();
        let m1 = utmp[3].to_le_bytes();
        let scales = [s0[0], s0[1], s0[2], s0[3], s1[0], s1[1], s1[2], s1[3]];
        let mins = [m0[0], m0[1], m0[2], m0[3], m1[0], m1[1], m1[2], m1[3]];

        let mut sumi = 0i32;
        for j in 0..16 {
            sumi += q8k.bsums[j] as i32 * mins[j / 2] as i32;
        }

        // 4 iterations, each covering 32 packed bytes of q4 (= 64 nibbles
        // → 2 sub-blocks). Match the scalar sub-block layout exactly:
        //   sub-block 2g   = low  nibbles of q4[g*32 : g*32+32] · q8[g*64 : g*64+32]
        //   sub-block 2g+1 = high nibbles of q4[g*32 : g*32+32] · q8[g*64+32 : g*64+64]
        // Reused from the AVX2 fix; the earlier per-16-lane split
        // interleaved q8 windows and produced silently wrong sums.
        let mut total: i32 = 0;
        for g in 0..4 {
            let mut sub_dot_a = 0i32;
            let mut sub_dot_b = 0i32;
            for l in 0..32 {
                let lo = (q4[g * 32 + l] & 0x0F) as i32;
                let hi = ((q4[g * 32 + l] >> 4) & 0x0F) as i32;
                sub_dot_a += lo * q8k.qs[g * 64 + l] as i32;
                sub_dot_b += hi * q8k.qs[g * 64 + 32 + l] as i32;
            }
            total += scales[g * 2] as i32 * sub_dot_a;
            total += scales[g * 2 + 1] as i32 * sub_dot_b;
        }

        d * q8k.d * total as f32 - dmin * q8k.d * sumi as f32
    }

    /// 16-element i8 × i8 dot using AVX2 widening. Local copy of
    /// `avx2_dot::dot_16_i8_widened` — AVX-512BW implies AVX2 support so the
    /// intrinsics compile fine under the enclosing `avx512bw` feature gate.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_16_i8_widened_avx2(acc: __m256i, w: *const i8, x: *const i8) -> __m256i {
        let w16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.cast()));
        let x16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(x.cast()));
        let prod32 = _mm256_madd_epi16(w16, x16);
        _mm256_add_epi32(acc, prod32)
    }

    /// AVX2 horizontal-sum helper. Mirrors `avx2_dot::hsum_i32_avx2`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_i32_avx2(v: __m256i) -> i32 {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);
        let sum128 = _mm_add_epi32(lo, hi);
        let sh = _mm_shuffle_epi32(sum128, 0b1110);
        let sum64 = _mm_add_epi32(sum128, sh);
        let sh2 = _mm_shufflelo_epi16(sum64, 0b1110);
        let sum32 = _mm_add_epi32(sum64, sh2);
        _mm_cvtsi128_si32(sum32)
    }

    /// Q6_K × Q8_K dot product using AVX-512BW.
    ///
    /// Two hot paths were previously scalar in this kernel:
    ///
    /// 1. **6-bit unpack** — 128 iterations of `((ql & 0xF) | ((qh >> …) << 4)) - 32`
    ///    per outer block. Replaced with the AVX2 parallel bit unpacking
    ///    pattern (mirrors `avx2_dot::q6k_q8k_dot` and the NEON kernel):
    ///    load `ql_lo` / `ql_hi` (two 32-byte AVX2 vectors) and `qh` (one
    ///    32-byte vector), then produce four 32-byte aux8 quadrants via
    ///    byte-wise shift/mask/or.
    /// 2. **Per-sub-block dot** — the previous implementation issued a
    ///    64-lane signed dot, discarded its result, and manually re-computed
    ///    the 16-element dots in scalar. Replaced with a proper
    ///    `_mm256_madd_epi16`-based 16-lane sub-block dot
    ///    (`dot_16_i8_widened_avx2`), matching the AVX2 kernel.
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub(super) unsafe fn q6k_q8k_dot(q6k_block: &[u8], q8k: &BlockQ8K) -> f32 {
        let ql = &q6k_block[0..128];
        let qh = &q6k_block[128..192];
        let scales = &q6k_block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([q6k_block[208], q6k_block[209]]));

        // Vectorised bit unpacking (AVX2 subset, safe under avx512bw).
        let mut aux8 = [0i8; QK_K];
        let mask_nib = _mm256_set1_epi8(0x0F);
        let mask_hi = _mm256_set1_epi8(0x30);
        let bias = _mm256_set1_epi8(-32);
        let mut a_off = 0usize;
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        for _ in 0..2 {
            let ql_lo = _mm256_loadu_si256(ql.as_ptr().add(ql_off).cast());
            let ql_hi = _mm256_loadu_si256(ql.as_ptr().add(ql_off + 32).cast());
            let qh_v = _mm256_loadu_si256(qh.as_ptr().add(qh_off).cast());

            // Q0: (ql_lo & 0xF) | ((qh << 4) & 0x30) - 32
            let hi0 = _mm256_and_si256(_mm256_slli_epi16(qh_v, 4), mask_hi);
            let q0 = _mm256_add_epi8(
                _mm256_or_si256(_mm256_and_si256(ql_lo, mask_nib), hi0),
                bias,
            );
            _mm256_storeu_si256(aux8.as_mut_ptr().add(a_off).cast(), q0);

            // Q1: (ql_hi & 0xF) | ((qh << 2) & 0x30) - 32
            let hi1 = _mm256_and_si256(_mm256_slli_epi16(qh_v, 2), mask_hi);
            let q1 = _mm256_add_epi8(
                _mm256_or_si256(_mm256_and_si256(ql_hi, mask_nib), hi1),
                bias,
            );
            _mm256_storeu_si256(aux8.as_mut_ptr().add(a_off + 32).cast(), q1);

            // Q2: (ql_lo >> 4) | (qh & 0x30) - 32
            let hi2 = _mm256_and_si256(qh_v, mask_hi);
            let q2 = _mm256_add_epi8(
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql_lo, 4), mask_nib), hi2),
                bias,
            );
            _mm256_storeu_si256(aux8.as_mut_ptr().add(a_off + 64).cast(), q2);

            // Q3: (ql_hi >> 4) | ((qh >> 2) & 0x30) - 32
            let hi3 = _mm256_and_si256(_mm256_srli_epi16(qh_v, 2), mask_hi);
            let q3 = _mm256_add_epi8(
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql_hi, 4), mask_nib), hi3),
                bias,
            );
            _mm256_storeu_si256(aux8.as_mut_ptr().add(a_off + 96).cast(), q3);

            a_off += 128;
            ql_off += 64;
            qh_off += 32;
        }

        // 16 sub-blocks of 16 elements — SIMD per-sub-block dot via
        // widened i8 → i16 → i32 madd. Matches the AVX2 kernel.
        let mut total: i32 = 0;
        for is in 0..16 {
            let a_ptr = aux8.as_ptr().add(is * 16);
            let q8_ptr = q8k.qs.as_ptr().add(is * 16);
            let acc0 = _mm256_setzero_si256();
            let acc = dot_16_i8_widened_avx2(acc0, a_ptr, q8_ptr);
            let sub_dot = hsum_i32_avx2(acc);
            total += scales[is] as i8 as i32 * sub_dot;
        }

        d * q8k.d * total as f32
    }

    /// Q5_K × Q8_K dot product using AVX-512BW.
    ///
    /// Reuses the AVX2 Q5_K algorithm but doubles the lane count: each `im`
    /// iteration handles all 64 elements in a single AVX-512 register.
    /// The qh bit extraction moves to 64-lane shift+mask, and the dot is
    /// computed via widened `i8 → i16 → i32` madd (saturation-safe).
    #[inline]
    #[target_feature(enable = "avx512bw")]
    pub(super) unsafe fn q5k_q8k_dot(q5k_block: &[u8], q8k: &BlockQ8K) -> f32 {
        let d = f16_to_f32(u16::from_le_bytes([q5k_block[0], q5k_block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([q5k_block[2], q5k_block[3]]));
        let scales_raw = &q5k_block[4..16];
        let qh = &q5k_block[16..48];
        let qs = &q5k_block[48..176];

        // qh: 32 bytes — fits in the low half of a 64-lane AVX-512 register.
        // For AVX-512BW we need the same 32 bytes duplicated in both halves
        // so the OR against `[low_nibbles | high_nibbles]` (also 64 lanes,
        // built from a single 32-byte qs load) reaches the intended byte
        // positions.
        let qh256 = _mm256_loadu_si256(qh.as_ptr().cast());
        let qh_v = _mm512_broadcast_i32x8(qh256);
        let mask_lo4 = _mm512_set1_epi8(0x0F);
        let mask_bit0 = _mm512_set1_epi8(1);

        let mut sumf = 0.0f32;
        for im in 0..4u8 {
            let (sc1, m1) = get_scale_min_k4((im as usize) * 2, scales_raw);
            let (sc2, m2) = get_scale_min_k4((im as usize) * 2 + 1, scales_raw);

            // Load 32 packed qs bytes = 64 nibbles.
            let qs256 = _mm256_loadu_si256(qs.as_ptr().add((im as usize) * 32).cast());
            let qs_lo = _mm256_and_si256(qs256, _mm256_set1_epi8(0x0F));
            let qs_hi = _mm256_and_si256(_mm256_srli_epi16(qs256, 4), _mm256_set1_epi8(0x0F));
            // Concatenate low + high nibbles into a single 64-lane register.
            let nibbles64 = _mm512_inserti64x4(_mm512_castsi256_si512(qs_lo), qs_hi, 1);

            // Runtime-variable shifts via the `_srl_` sibling form
            // (`_mm512_srli_epi16` requires a const-imm shift amount).
            let shift_even = _mm_cvtsi32_si128(i32::from(im) * 2);
            let shift_odd = _mm_cvtsi32_si128(i32::from(im) * 2 + 1);
            let hbits = _mm512_and_si512(_mm512_srl_epi16(qh_v, shift_even), mask_bit0);
            let hbits_odd = _mm512_and_si512(_mm512_srl_epi16(qh_v, shift_odd), mask_bit0);
            // Assemble the 5th-bit mask so low-half bytes get the even bit
            // and high-half bytes get the odd bit.
            let hbits_pack = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm512_castsi512_si256(hbits)),
                _mm512_castsi512_si256(hbits_odd),
                1,
            );
            let hbits_hi = _mm512_slli_epi16(hbits_pack, 4);
            let q_all = _mm512_or_si512(nibbles64, hbits_hi);

            // Dot with 64 q8 activations via widened i16 accumulation, then
            // split the running sum by pulling the low/high 32 lanes apart
            // and hsumming each independently so scales[2*im] and
            // scales[2*im+1] can weight their own sub-blocks.
            let q_lo = _mm512_castsi512_si256(q_all);
            let q_hi = _mm512_extracti64x4_epi64(q_all, 1);
            let q8_ptr_a = q8k.qs.as_ptr().add((im as usize) * 64);
            let q8_ptr_b = q8k.qs.as_ptr().add((im as usize) * 64 + 32);

            let dot_a = {
                let acc0 = _mm256_setzero_si256();
                let acc = super::avx2_dot_bridge_u8_i8(acc0, q_lo, q8_ptr_a);
                super::avx2_hsum_i32_bridge(acc)
            };
            let dot_b = {
                let acc0 = _mm256_setzero_si256();
                let acc = super::avx2_dot_bridge_u8_i8(acc0, q_hi, q8_ptr_b);
                super::avx2_hsum_i32_bridge(acc)
            };

            let sum1 =
                q8k.bsums[(im as usize) * 4] as i32 + q8k.bsums[(im as usize) * 4 + 1] as i32;
            let sum2 =
                q8k.bsums[(im as usize) * 4 + 2] as i32 + q8k.bsums[(im as usize) * 4 + 3] as i32;

            sumf += d * sc1 as f32 * dot_a as f32 - dmin * m1 as f32 * sum1 as f32;
            sumf += d * sc2 as f32 * dot_b as f32 - dmin * m2 as f32 * sum2 as f32;
        }
        sumf * q8k.d
    }

    /// Fused Q8_0 dequantize + f32 matvec for a whole row using AVX-512F.
    ///
    /// Twice-wide port of the AVX2 [`super::avx2_dot::q8_0_matvec_row`]:
    /// two 16-lane FMA iterations cover each 32-element block instead of
    /// four 8-lane ones. Sign-extension and scaling are identical.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn q8_0_matvec_row(input: &[f32], row_data: &[u8], cols: usize) -> f32 {
        let blocks_per_row = cols / super::QK8_0;
        let block_bytes = 34usize;
        let mut acc = _mm512_setzero_ps();
        for bi in 0..blocks_per_row {
            let off = bi * block_bytes;
            let d = super::f16_to_f32(u16::from_le_bytes([row_data[off], row_data[off + 1]]));
            let d_bcast = _mm512_set1_ps(d);
            let qs_ptr = row_data.as_ptr().add(off + 2).cast::<i8>();
            let col_base = bi * super::QK8_0;
            // 2 × 16-lane iterations = 32 elements per block.
            for k in 0..2 {
                let q8 = _mm_loadu_si128(qs_ptr.add(k * 16).cast::<__m128i>());
                let q32 = _mm512_cvtepi8_epi32(q8);
                let qf = _mm512_cvtepi32_ps(q32);
                let scaled = _mm512_mul_ps(qf, d_bcast);
                let x = _mm512_loadu_ps(input.as_ptr().add(col_base + k * 16));
                acc = _mm512_fmadd_ps(scaled, x, acc);
            }
        }
        _mm512_reduce_add_ps(acc)
    }

    /// Ternary bitmask × f32 input dot product for one row (AVX-512F).
    ///
    /// Uses AVX-512's native 16-bit mask register (`__mmask16`) to gate the
    /// per-lane contribution — no explicit compare + and dance. Two
    /// consecutive bytes of `pos_mask` / `neg_mask` combine into a
    /// `__mmask16` covering 16 lanes at once, so each iteration processes
    /// twice the input width of the AVX2 path.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn ternary_dot_row_unscaled(
        pos_mask: &[u8],
        neg_mask: &[u8],
        input: &[f32],
        num_cols: usize,
    ) -> f32 {
        let mut pos_acc = _mm512_setzero_ps();
        let mut neg_acc = _mm512_setzero_ps();

        // 16-lane groups walk two mask bytes at a time.
        let full_pairs = num_cols / 16;
        for pair in 0..full_pairs {
            let pm = u16::from(pos_mask[pair * 2]) | (u16::from(pos_mask[pair * 2 + 1]) << 8);
            let nm = u16::from(neg_mask[pair * 2]) | (u16::from(neg_mask[pair * 2 + 1]) << 8);
            if (pm | nm) == 0 {
                continue;
            }
            let input_v = _mm512_loadu_ps(input.as_ptr().add(pair * 16));
            if pm != 0 {
                let k: __mmask16 = pm;
                pos_acc = _mm512_mask_add_ps(pos_acc, k, pos_acc, input_v);
            }
            if nm != 0 {
                let k: __mmask16 = nm;
                neg_acc = _mm512_mask_add_ps(neg_acc, k, neg_acc, input_v);
            }
        }

        let mut pos_sum = _mm512_reduce_add_ps(pos_acc);
        let mut neg_sum = _mm512_reduce_add_ps(neg_acc);

        // Tail: any elements past `full_pairs * 16`. We fall through to a
        // scalar loop for the up-to-15 remaining lanes so the mask packing
        // stays simple.
        let tail_start = full_pairs * 16;
        for idx in tail_start..num_cols {
            let byte_idx = idx / 8;
            let bit = idx % 8;
            let p = f32::from((pos_mask[byte_idx] >> bit) & 1);
            let n = f32::from((neg_mask[byte_idx] >> bit) & 1);
            pos_sum += p * input[idx];
            neg_sum += n * input[idx];
        }

        pos_sum - neg_sum
    }
}

// ─── AVX2 helpers used by AVX-512 kernels (bridge) ──────────────────────────
//
// AVX-512BW machines by definition also have AVX2 (VEX-encoded AVX-512
// implies AVX2). The Q5_K AVX-512 kernel calls these AVX2 helpers to reuse
// the widened i16 dot without duplicating the intrinsics dance in both
// modules.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_dot_bridge_u8_i8(
    acc: std::arch::x86_64::__m256i,
    u: std::arch::x86_64::__m256i,
    x: *const i8,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let xv = _mm256_loadu_si256(x.cast());
    let u_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(u));
    let u_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(u, 1));
    let x_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(xv));
    let x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xv, 1));
    let sum_lo = _mm256_madd_epi16(u_lo, x_lo);
    let sum_hi = _mm256_madd_epi16(u_hi, x_hi);
    _mm256_add_epi32(acc, _mm256_add_epi32(sum_lo, sum_hi))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_hsum_i32_bridge(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let sum128 = _mm_add_epi32(lo, hi);
    let sh = _mm_shuffle_epi32(sum128, 0b1110);
    let sum64 = _mm_add_epi32(sum128, sh);
    let sh2 = _mm_shufflelo_epi16(sum64, 0b1110);
    let sum32 = _mm_add_epi32(sum64, sh2);
    _mm_cvtsi128_si32(sum32)
}

// ─── Runtime CPU feature cache (Issue #13) ──────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn x86_avx2_supported() -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| is_x86_feature_detected!("avx2"))
}

#[cfg(target_arch = "x86_64")]
fn x86_avx512bw_supported() -> bool {
    use std::sync::OnceLock;
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| is_x86_feature_detected!("avx512bw"))
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
        for l in 0..32 {
            aux8[a_off + l] = (q4[q4_off + l] & 0xF) as i8;
        }
        a_off += 32;
        for l in 0..32 {
            aux8[a_off + l] = (q4[q4_off + l] >> 4) as i8;
        }
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

    let s0 = utmp[0].to_le_bytes();
    let s1 = utmp[1].to_le_bytes();
    let m0 = utmp[2].to_le_bytes();
    let m1 = utmp[3].to_le_bytes();
    let scales = [s0[0], s0[1], s0[2], s0[3], s1[0], s1[1], s1[2], s1[3]];
    let mins = [m0[0], m0[1], m0[2], m0[3], m1[0], m1[1], m1[2], m1[3]];

    let mut sumi = 0i32;
    for j in 0..16 {
        sumi += q8k.bsums[j] as i32 * mins[j / 2] as i32;
    }

    let mut aux32 = [0i32; 8];
    let mut a_idx = 0usize;
    let mut q8_idx = 0usize;
    for is in 0..8 {
        let scale = scales[is] as i32;
        for _ in 0..4 {
            for l in 0..8 {
                aux32[l] += scale * (q8k.qs[q8_idx + l] as i32 * aux8[a_idx + l] as i32);
            }
            q8_idx += 8;
            a_idx += 8;
        }
    }

    let d_all = d * q8k.d;
    let dmin_all = dmin * q8k.d;
    let mut sumf = 0.0f32;
    for l in 0..8 {
        sumf += d_all * aux32[l] as f32;
    }
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
            aux8[a_off + l] = ((ql[ql_off + l] & 0xF) | ((qh[qh_off + l] & 3) << 4)) as i8 - 32;
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

    let mut aux32 = [0i32; 8];
    let mut a_idx = 0usize;
    let mut q8_idx = 0usize;
    for is in 0..16 {
        let scale = scales[is] as i8 as i32;
        for _ in 0..2 {
            for l in 0..8 {
                aux32[l] += scale * (q8k.qs[q8_idx + l] as i32 * aux8[a_idx + l] as i32);
            }
            q8_idx += 8;
            a_idx += 8;
        }
    }

    let d_all = d * q8k.d;
    let mut sumf = 0.0f32;
    for l in 0..8 {
        sumf += d_all * aux32[l] as f32;
    }
    sumf
}

// ─── Dispatch: NEON or scalar ───────────────────────────────────────────────

/// Q4_K × Q8_K dot product.
///
/// Dispatches at runtime to the fastest supported implementation:
/// x86_64: AVX-512BW → AVX2 → scalar fallback.
/// aarch64: NEON → scalar fallback. NEON is baseline on aarch64 targets
/// Rust supports, so the dispatch is compile-time cfg (no runtime probe).
/// The NEON kernel was gated behind bit-exact parity tests before being
/// wired here — see `q4k_neon_matches_scalar_bit_exact` for the guard.
#[inline]
fn q4k_q8k_dot(q4k_block: &[u8], q8k: &BlockQ8K) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: intrinsics gated on runtime CPU feature detection cached
        // in a `OnceLock`; the sub-functions carry `#[target_feature]`
        // annotations so the compiler emits the correct instruction set.
        if x86_avx512bw_supported() {
            return unsafe { avx512_dot::q4k_q8k_dot(q4k_block, q8k) };
        }
        if x86_avx2_supported() {
            return unsafe { avx2_dot::q4k_q8k_dot(q4k_block, q8k) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on every aarch64 CPU Rust supports.
        return unsafe { neon_dot::q4k_q8k_dot(q4k_block, q8k) };
    }
    #[allow(unreachable_code)]
    q4k_q8k_dot_fallback_scalar(q4k_block, q8k)
}

/// Q4_K × Q8_K scalar fallback body extracted so the runtime dispatcher
/// can call it without duplicating the arithmetic. Kept close to the
/// dispatcher to preserve inlineability.
#[inline]
fn q4k_q8k_dot_fallback_scalar(q4k_block: &[u8], q8k: &BlockQ8K) -> f32 {
    const KMASK1: u32 = 0x3f3f_3f3f;
    const KMASK2: u32 = 0x0f0f_0f0f;
    const KMASK3: u32 = 0x0303_0303;

    let d = f16_to_f32(u16::from_le_bytes([q4k_block[0], q4k_block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([q4k_block[2], q4k_block[3]]));
    let q4 = &q4k_block[16..144];

    let mut aux8 = [0u8; QK_K];
    for g in 0..4 {
        for l in 0..32 {
            aux8[g * 64 + l] = q4[g * 32 + l] & 0xF;
            aux8[g * 64 + 32 + l] = q4[g * 32 + l] >> 4;
        }
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
    let s0 = utmp[0].to_le_bytes();
    let s1 = utmp[1].to_le_bytes();
    let m0 = utmp[2].to_le_bytes();
    let m1 = utmp[3].to_le_bytes();
    let scales = [s0[0], s0[1], s0[2], s0[3], s1[0], s1[1], s1[2], s1[3]];
    let mins = [m0[0], m0[1], m0[2], m0[3], m1[0], m1[1], m1[2], m1[3]];

    let mut sumi = 0i32;
    for j in 0..16 {
        sumi += q8k.bsums[j] as i32 * mins[j / 2] as i32;
    }

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

/// Q6_K × Q8_K dot product.
///
/// Runtime-dispatched on x86_64 (AVX-512BW → AVX2 → scalar); NEON on
/// aarch64. See `q4k_q8k_dot` for the design rationale and the
/// `q6k_neon_matches_scalar_bit_exact` parity gate that preceded
/// enabling the aarch64 path.
#[inline]
fn q6k_q8k_dot(q6k_block: &[u8], q8k: &BlockQ8K) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: same rationale as `q4k_q8k_dot`.
        if x86_avx512bw_supported() {
            return unsafe { avx512_dot::q6k_q8k_dot(q6k_block, q8k) };
        }
        if x86_avx2_supported() {
            return unsafe { avx2_dot::q6k_q8k_dot(q6k_block, q8k) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on every aarch64 CPU Rust supports.
        return unsafe { neon_dot::q6k_q8k_dot(q6k_block, q8k) };
    }
    #[allow(unreachable_code)]
    q6k_q8k_dot_fallback_scalar(q6k_block, q8k)
}

/// Q6_K × Q8_K scalar fallback body.
#[inline]
fn q6k_q8k_dot_fallback_scalar(q6k_block: &[u8], q8k: &BlockQ8K) -> f32 {
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
            aux8[a_off + l] = ((ql[ql_off + l] & 0xF) | ((qh[qh_off + l] & 3) << 4)) as i8 - 32;
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

/// Q2_K × Q8_K dot product.
#[inline]
fn q2k_q8k_dot(q2k_block: &[u8], q8k: &BlockQ8K) -> f32 {
    let scales = &q2k_block[0..16];
    let qs = &q2k_block[16..80];
    let d = f16_to_f32(u16::from_le_bytes([q2k_block[80], q2k_block[81]]));
    let dmin = f16_to_f32(u16::from_le_bytes([q2k_block[82], q2k_block[83]]));

    let mut sumf = 0.0f32;
    for group in 0..16 {
        let sc = (scales[group] & 0xF) as i32;
        let m = (scales[group] >> 4) as i32;
        let mut dot = 0i32;
        let mut sum_q8 = 0i32;
        for j in 0..16 {
            let flat = group * 16 + j;
            let byte_idx = flat / 4;
            let bit_shift = (flat % 4) * 2;
            let q = ((qs[byte_idx] >> bit_shift) & 3) as i32;
            dot += q * q8k.qs[flat] as i32;
            sum_q8 += q8k.qs[flat] as i32;
        }
        sumf += d * sc as f32 * dot as f32 - dmin * m as f32 * sum_q8 as f32;
    }
    sumf * q8k.d
}

/// Q5_K × Q8_K dot product.
///
/// Runtime-dispatched: AVX-512BW → AVX2 → scalar fallback on x86_64.
/// See `q4k_q8k_dot` for the design rationale.
#[inline]
fn q5k_q8k_dot(q5k_block: &[u8], q8k: &BlockQ8K) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: intrinsics gated on cached runtime CPU detection.
        if x86_avx512bw_supported() {
            return unsafe { avx512_dot::q5k_q8k_dot(q5k_block, q8k) };
        }
        if x86_avx2_supported() {
            return unsafe { avx2_dot::q5k_q8k_dot(q5k_block, q8k) };
        }
    }
    q5k_q8k_dot_fallback_scalar(q5k_block, q8k)
}

/// Q5_K × Q8_K scalar fallback body.
#[inline]
fn q5k_q8k_dot_fallback_scalar(q5k_block: &[u8], q8k: &BlockQ8K) -> f32 {
    let d = f16_to_f32(u16::from_le_bytes([q5k_block[0], q5k_block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([q5k_block[2], q5k_block[3]]));
    let scales_raw = &q5k_block[4..16];
    let qh = &q5k_block[16..48];
    let qs = &q5k_block[48..176];

    let mut sumf = 0.0f32;
    let mut is = 0usize;
    let mut q_offset = 0usize;
    let mut q8_offset = 0usize;

    for im in 0..4u8 {
        let (sc1, m1) = get_scale_min_k4(is, scales_raw);
        let (sc2, m2) = get_scale_min_k4(is + 1, scales_raw);

        let mut dot1 = 0i32;
        let mut sum1 = 0i32;
        for l in 0..32 {
            let hbit = ((qh[l] >> (im * 2)) & 1) as i32;
            let q = ((qs[q_offset + l] & 0xF) as i32) | (hbit << 4);
            dot1 += q * q8k.qs[q8_offset + l] as i32;
            sum1 += q8k.qs[q8_offset + l] as i32;
        }
        sumf += d * sc1 as f32 * dot1 as f32 - dmin * m1 as f32 * sum1 as f32;

        let mut dot2 = 0i32;
        let mut sum2 = 0i32;
        for l in 0..32 {
            let hbit = ((qh[l] >> (im * 2 + 1)) & 1) as i32;
            let q = ((qs[q_offset + l] >> 4) as i32) | (hbit << 4);
            dot2 += q * q8k.qs[q8_offset + 32 + l] as i32;
            sum2 += q8k.qs[q8_offset + 32 + l] as i32;
        }
        sumf += d * sc2 as f32 * dot2 as f32 - dmin * m2 as f32 * sum2 as f32;

        q_offset += 32;
        q8_offset += 64;
        is += 2;
    }
    sumf * q8k.d
}

/// Q3_K × Q8_K dot product.
#[inline]
fn q3k_q8k_dot(q3k_block: &[u8], q8k: &BlockQ8K) -> f32 {
    let hmask = &q3k_block[0..32];
    let qs = &q3k_block[32..96];
    let d = f16_to_f32(u16::from_le_bytes([q3k_block[108], q3k_block[109]]));
    let scales = q3k_decode_scales(&q3k_block[96..108]);

    // Unpack 3-bit weights to i8
    let mut aux8 = [0i8; QK_K];
    let mut m: u8 = 1;
    let mut q_off = 0usize;
    for j in (0..QK_K).step_by(128) {
        for shift in 0..4u8 {
            for l in 0..32usize {
                let lo2 = (qs[q_off + l] >> (shift * 2)) & 3;
                let hi = if hmask[l] & m != 0 { 0i8 } else { -4i8 };
                aux8[j + shift as usize * 32 + l] = lo2 as i8 + hi;
            }
            m <<= 1;
        }
        q_off += 32;
    }

    // Dot product: 16 sub-blocks of 16 elements
    let mut total = 0i32;
    for is in 0..16 {
        let off = is * 16;
        let mut dot = 0i32;
        for l in 0..16 {
            dot += aux8[off + l] as i32 * q8k.qs[off + l] as i32;
        }
        total += scales[is] * dot;
    }

    d * q8k.d * total as f32
}

// ─── Fused quantized matvec ─────────────────────────────────────────────────

/// Q2_K matvec: quantizes input to Q8_K, then uses integer dot product.
pub fn q2k_matvec(input: &[f32], data: &[u8], rows: usize, cols: usize, output: &mut [f32]) {
    let q8_blocks = quantize_row_q8_k(input);
    q2k_matvec_preq(data, rows, cols, &q8_blocks, output);
}

/// Q2_K matvec with pre-quantized Q8_K input.
#[allow(unused_variables)]
pub fn q2k_matvec_preq(
    data: &[u8],
    rows: usize,
    cols: usize,
    q8_blocks: &[BlockQ8K],
    output: &mut [f32],
) {
    let blocks_per_row = cols / QK_K;
    let block_bytes = 84;
    let row_bytes = blocks_per_row * block_bytes;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        output.par_iter_mut().enumerate().for_each(|(row, out)| {
            let row_data = &data[row * row_bytes..(row + 1) * row_bytes];
            let mut sumf = 0.0f32;
            for bi in 0..blocks_per_row {
                sumf += q2k_q8k_dot(
                    &row_data[bi * block_bytes..(bi + 1) * block_bytes],
                    &q8_blocks[bi],
                );
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
            sumf += q2k_q8k_dot(
                &row_data[bi * block_bytes..(bi + 1) * block_bytes],
                &q8_blocks[bi],
            );
        }
        output[row] = sumf;
    }
}

/// Q5_K matvec: quantizes input to Q8_K, then uses integer dot product.
pub fn q5k_matvec(input: &[f32], data: &[u8], rows: usize, cols: usize, output: &mut [f32]) {
    let q8_blocks = quantize_row_q8_k(input);
    q5k_matvec_preq(data, rows, cols, &q8_blocks, output);
}

/// Q5_K matvec with pre-quantized Q8_K input.
#[allow(unused_variables)]
pub fn q5k_matvec_preq(
    data: &[u8],
    rows: usize,
    cols: usize,
    q8_blocks: &[BlockQ8K],
    output: &mut [f32],
) {
    let blocks_per_row = cols / QK_K;
    let block_bytes = 176;
    let row_bytes = blocks_per_row * block_bytes;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        output.par_iter_mut().enumerate().for_each(|(row, out)| {
            let row_data = &data[row * row_bytes..(row + 1) * row_bytes];
            let mut sumf = 0.0f32;
            for bi in 0..blocks_per_row {
                sumf += q5k_q8k_dot(
                    &row_data[bi * block_bytes..(bi + 1) * block_bytes],
                    &q8_blocks[bi],
                );
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
            sumf += q5k_q8k_dot(
                &row_data[bi * block_bytes..(bi + 1) * block_bytes],
                &q8_blocks[bi],
            );
        }
        output[row] = sumf;
    }
}

/// Q3_K matvec: quantizes input to Q8_K, then uses integer dot product.
pub fn q3k_matvec(input: &[f32], data: &[u8], rows: usize, cols: usize, output: &mut [f32]) {
    let q8_blocks = quantize_row_q8_k(input);
    q3k_matvec_preq(data, rows, cols, &q8_blocks, output);
}

/// Q3_K matvec with pre-quantized Q8_K input.
#[allow(unused_variables)]
pub fn q3k_matvec_preq(
    data: &[u8],
    rows: usize,
    cols: usize,
    q8_blocks: &[BlockQ8K],
    output: &mut [f32],
) {
    let blocks_per_row = cols / QK_K;
    let block_bytes = 110;
    let row_bytes = blocks_per_row * block_bytes;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        output.par_iter_mut().enumerate().for_each(|(row, out)| {
            let row_data = &data[row * row_bytes..(row + 1) * row_bytes];
            let mut sumf = 0.0f32;
            for bi in 0..blocks_per_row {
                sumf += q3k_q8k_dot(
                    &row_data[bi * block_bytes..(bi + 1) * block_bytes],
                    &q8_blocks[bi],
                );
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
            sumf += q3k_q8k_dot(
                &row_data[bi * block_bytes..(bi + 1) * block_bytes],
                &q8_blocks[bi],
            );
        }
        output[row] = sumf;
    }
}

/// Q4_K matvec: quantizes input to Q8_K, then uses integer dot product.
pub fn q4k_matvec(input: &[f32], data: &[u8], rows: usize, cols: usize, output: &mut [f32]) {
    let q8_blocks = quantize_row_q8_k(input);
    q4k_matvec_preq(data, rows, cols, &q8_blocks, output);
}

/// Q4_K matvec with pre-quantized Q8_K input (avoids redundant quantization).
#[allow(unused_variables)]
pub fn q4k_matvec_preq(
    data: &[u8],
    rows: usize,
    cols: usize,
    q8_blocks: &[BlockQ8K],
    output: &mut [f32],
) {
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
                sumf += q4k_q8k_dot(
                    &row_data[bi * block_bytes..(bi + 1) * block_bytes],
                    &q8_blocks[bi],
                );
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
            sumf += q4k_q8k_dot(
                &row_data[bi * block_bytes..(bi + 1) * block_bytes],
                &q8_blocks[bi],
            );
        }
        output[row] = sumf;
    }
}

/// Fused Q8_0 dequantize + matrix-vector multiply.
///
/// Runtime-dispatched on x86_64: AVX-512F → AVX2+FMA → scalar fallback.
pub fn q8_0_matvec(input: &[f32], data: &[u8], rows: usize, cols: usize, output: &mut [f32]) {
    let blocks_per_row = cols / QK8_0;
    let block_bytes = 34;
    let row_bytes = blocks_per_row * block_bytes;

    #[cfg(target_arch = "x86_64")]
    {
        // AVX-512 requires 16-lane loads from `input`, which forces the
        // column count to a multiple of 16. Q8_0's 32-lane block already
        // satisfies this — every FMA lane maps 1:1 to a block element.
        if x86_avx512bw_supported() {
            for row in 0..rows {
                let row_data = &data[row * row_bytes..(row + 1) * row_bytes];
                // SAFETY: guarded by runtime AVX-512F detection.
                output[row] = unsafe { avx512_dot::q8_0_matvec_row(input, row_data, cols) };
            }
            return;
        }
        if x86_avx2_supported() {
            for row in 0..rows {
                let row_data = &data[row * row_bytes..(row + 1) * row_bytes];
                // SAFETY: guarded by runtime AVX2 detection.
                output[row] = unsafe { avx2_dot::q8_0_matvec_row(input, row_data, cols) };
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on every aarch64 target Rust supports, so
        // no runtime feature probe is needed — the `#[target_feature]` on the
        // kernel matches unconditionally. Parity vs scalar is FMA-rounding
        // approximate (see `q8_0_neon_matches_scalar_within_tol`), not
        // bit-exact, because scalar uses separate mul + add whereas the NEON
        // path uses `vfmaq_f32` (fused mul-add, single rounding).
        for row in 0..rows {
            let row_data = &data[row * row_bytes..(row + 1) * row_bytes];
            output[row] = unsafe { neon_dot::q8_0_matvec_row(input, row_data, cols) };
        }
        return;
    }

    #[allow(unreachable_code)]
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

/// Fused Q4_0 dequantize + matrix-vector multiply.
pub fn q4_0_matvec(input: &[f32], data: &[u8], rows: usize, cols: usize, output: &mut [f32]) {
    let blocks_per_row = cols / QK8_0;
    let block_bytes = 18;
    let row_bytes = blocks_per_row * block_bytes;

    for row in 0..rows {
        let mut acc = 0.0f32;
        let row_data = &data[row * row_bytes..(row + 1) * row_bytes];

        for bi in 0..blocks_per_row {
            let off = bi * block_bytes;
            let d = f16_to_f32(u16::from_le_bytes([row_data[off], row_data[off + 1]]));
            let col_base = bi * QK8_0;

            for j in 0..16 {
                let qs = row_data[off + 2 + j];
                let x0 = ((qs & 0x0F) as i32) - 8;
                let x1 = ((qs >> 4) as i32) - 8;
                acc += d * (x0 as f32) * input[col_base + j];
                acc += d * (x1 as f32) * input[col_base + j + 16];
            }
        }

        output[row] = acc;
    }
}

/// Fused Q4_1 dequantize + matrix-vector multiply.
pub fn q4_1_matvec(input: &[f32], data: &[u8], rows: usize, cols: usize, output: &mut [f32]) {
    let blocks_per_row = cols / QK8_0;
    let block_bytes = 20;
    let row_bytes = blocks_per_row * block_bytes;

    for row in 0..rows {
        let mut acc = 0.0f32;
        let row_data = &data[row * row_bytes..(row + 1) * row_bytes];

        for bi in 0..blocks_per_row {
            let off = bi * block_bytes;
            let d = f16_to_f32(u16::from_le_bytes([row_data[off], row_data[off + 1]]));
            let m = f16_to_f32(u16::from_le_bytes([row_data[off + 2], row_data[off + 3]]));
            let col_base = bi * QK8_0;

            for j in 0..16 {
                let qs = row_data[off + 4 + j];
                let x0 = (qs & 0x0F) as f32;
                let x1 = (qs >> 4) as f32;
                acc += (d * x0 + m) * input[col_base + j];
                acc += (d * x1 + m) * input[col_base + j + 16];
            }
        }

        output[row] = acc;
    }
}

/// Fused Q5_0 dequantize + matrix-vector multiply.
pub fn q5_0_matvec(input: &[f32], data: &[u8], rows: usize, cols: usize, output: &mut [f32]) {
    let blocks_per_row = cols / QK8_0;
    let block_bytes = 22;
    let row_bytes = blocks_per_row * block_bytes;

    for row in 0..rows {
        let mut acc = 0.0f32;
        let row_data = &data[row * row_bytes..(row + 1) * row_bytes];

        for bi in 0..blocks_per_row {
            let off = bi * block_bytes;
            let d = f16_to_f32(u16::from_le_bytes([row_data[off], row_data[off + 1]]));
            let qh = u32::from_le_bytes([
                row_data[off + 2],
                row_data[off + 3],
                row_data[off + 4],
                row_data[off + 5],
            ]);
            let col_base = bi * QK8_0;

            for j in 0..16 {
                let qs = row_data[off + 6 + j];
                let xh_0 = ((qh >> j) << 4) & 0x10;
                let xh_1 = (qh >> (j + 12)) & 0x10;
                let x0 = (((qs & 0x0F) as u32) | xh_0) as i32 - 16;
                let x1 = (((qs >> 4) as u32) | xh_1) as i32 - 16;
                acc += d * (x0 as f32) * input[col_base + j];
                acc += d * (x1 as f32) * input[col_base + j + 16];
            }
        }

        output[row] = acc;
    }
}

/// Fused Q5_1 dequantize + matrix-vector multiply.
pub fn q5_1_matvec(input: &[f32], data: &[u8], rows: usize, cols: usize, output: &mut [f32]) {
    let blocks_per_row = cols / QK8_0;
    let block_bytes = 24;
    let row_bytes = blocks_per_row * block_bytes;

    for row in 0..rows {
        let mut acc = 0.0f32;
        let row_data = &data[row * row_bytes..(row + 1) * row_bytes];

        for bi in 0..blocks_per_row {
            let off = bi * block_bytes;
            let d = f16_to_f32(u16::from_le_bytes([row_data[off], row_data[off + 1]]));
            let m = f16_to_f32(u16::from_le_bytes([row_data[off + 2], row_data[off + 3]]));
            let qh = u32::from_le_bytes([
                row_data[off + 4],
                row_data[off + 5],
                row_data[off + 6],
                row_data[off + 7],
            ]);
            let col_base = bi * QK8_0;

            for j in 0..16 {
                let qs = row_data[off + 8 + j];
                let xh_0 = ((qh >> j) << 4) & 0x10;
                let xh_1 = (qh >> (j + 12)) & 0x10;
                let x0 = ((qs & 0x0F) as u32) | xh_0;
                let x1 = ((qs >> 4) as u32) | xh_1;
                acc += (d * (x0 as f32) + m) * input[col_base + j];
                acc += (d * (x1 as f32) + m) * input[col_base + j + 16];
            }
        }

        output[row] = acc;
    }
}

/// Fused IQ4_XS dequantize + matrix-vector multiply.
pub fn iq4_xs_matvec(input: &[f32], data: &[u8], rows: usize, cols: usize, output: &mut [f32]) {
    let blocks_per_row = cols / QK_K;
    let block_bytes = 136;
    let row_bytes = blocks_per_row * block_bytes;

    for row in 0..rows {
        let mut acc = 0.0f32;
        let row_data = &data[row * row_bytes..(row + 1) * row_bytes];

        for bi in 0..blocks_per_row {
            let off = bi * block_bytes;
            let d = f16_to_f32(u16::from_le_bytes([row_data[off], row_data[off + 1]]));
            let scales_h = u16::from_le_bytes([row_data[off + 2], row_data[off + 3]]);
            let scales_l = [
                row_data[off + 4],
                row_data[off + 5],
                row_data[off + 6],
                row_data[off + 7],
            ];
            let qs = &row_data[off + 8..off + 8 + 128];
            let col_base = bi * QK_K;

            for ib in 0..8 {
                let ls_lo = (scales_l[ib / 2] >> (4 * (ib as u8 & 1))) & 0x0F;
                let ls_hi = ((scales_h >> (2 * ib)) & 0x3) as u8;
                let ls = ((ls_hi << 4) | ls_lo) as i32;
                let dl = d * ((ls - 32) as f32);
                let qs_base = ib * 16;
                let sub_col_base = col_base + ib * 32;
                for j in 0..16 {
                    let byte = qs[qs_base + j];
                    let n0 = (byte & 0x0F) as usize;
                    let n1 = (byte >> 4) as usize;
                    acc += dl * f32::from(KVALUES_IQ4NL[n0]) * input[sub_col_base + j];
                    acc += dl * f32::from(KVALUES_IQ4NL[n1]) * input[sub_col_base + j + 16];
                }
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
#[allow(unused_variables)]
pub fn q6k_matvec_preq(
    data: &[u8],
    rows: usize,
    cols: usize,
    q8_blocks: &[BlockQ8K],
    output: &mut [f32],
) {
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
                sumf += q6k_q8k_dot(
                    &row_data[bi * block_bytes..(bi + 1) * block_bytes],
                    &q8_blocks[bi],
                );
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
            sumf += q6k_q8k_dot(
                &row_data[bi * block_bytes..(bi + 1) * block_bytes],
                &q8_blocks[bi],
            );
        }
        output[row] = sumf;
    }
}

/// IQ4_XS block × Q8_K block dot product.
///
/// Each IQ4_XS block encodes 256 elements as 8 sub-blocks of 32 elements,
/// each sub-block sharing a 6-bit scale and 4-bit indices into
/// [`KVALUES_IQ4NL`]. The 6-bit scale is offset by −32 to allow signed
/// values (roughly −32..31). The final dequantized weight for element `k`
/// in sub-block `ib` is `d * (ls[ib] - 32) * KVALUES_IQ4NL[nibble_k]`.
///
/// The dot product accumulates in `i32` per sub-block, then applies the
/// per-block scale factors (`d_w * d_i`) once at the end.
fn iq4_xs_q8k_dot(iq4_xs_block: &[u8], q8k: &BlockQ8K) -> f32 {
    let d_w = f16_to_f32(u16::from_le_bytes([iq4_xs_block[0], iq4_xs_block[1]]));
    let scales_h = u16::from_le_bytes([iq4_xs_block[2], iq4_xs_block[3]]);
    let scales_l = [
        iq4_xs_block[4],
        iq4_xs_block[5],
        iq4_xs_block[6],
        iq4_xs_block[7],
    ];
    let qs = &iq4_xs_block[8..8 + 128];

    let mut sum_i32 = 0i64;
    for ib in 0..8 {
        let ls_lo = (scales_l[ib / 2] >> (4 * (ib as u8 & 1))) & 0x0F;
        let ls_hi = ((scales_h >> (2 * ib)) & 0x3) as u8;
        let ls = (((ls_hi << 4) | ls_lo) as i32) - 32;
        let qs_base = ib * 16;
        let q8_base = ib * 32;
        let mut sub_dot = 0i32;
        for j in 0..16 {
            let byte = qs[qs_base + j];
            let n0 = (byte & 0x0F) as usize;
            let n1 = (byte >> 4) as usize;
            sub_dot += i32::from(KVALUES_IQ4NL[n0]) * i32::from(q8k.qs[q8_base + j]);
            sub_dot += i32::from(KVALUES_IQ4NL[n1]) * i32::from(q8k.qs[q8_base + j + 16]);
        }
        sum_i32 += i64::from(ls) * i64::from(sub_dot);
    }
    d_w * q8k.d * (sum_i32 as f32)
}

/// Fused IQ4_XS matvec with pre-quantized Q8_K input.
pub fn iq4_xs_matvec_preq(
    data: &[u8],
    rows: usize,
    cols: usize,
    q8_blocks: &[BlockQ8K],
    output: &mut [f32],
) {
    let blocks_per_row = cols / QK_K;
    let block_bytes = 136;
    let row_bytes = blocks_per_row * block_bytes;

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        output.par_iter_mut().enumerate().for_each(|(row, out)| {
            let row_data = &data[row * row_bytes..(row + 1) * row_bytes];
            let mut sumf = 0.0f32;
            for bi in 0..blocks_per_row {
                sumf += iq4_xs_q8k_dot(
                    &row_data[bi * block_bytes..(bi + 1) * block_bytes],
                    &q8_blocks[bi],
                );
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
            sumf += iq4_xs_q8k_dot(
                &row_data[bi * block_bytes..(bi + 1) * block_bytes],
                &q8_blocks[bi],
            );
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
        GgmlType::Q2_K => q2k_matvec(input, data, rows, cols, output),
        GgmlType::Q3_K => q3k_matvec(input, data, rows, cols, output),
        GgmlType::Q4_K => q4k_matvec(input, data, rows, cols, output),
        GgmlType::Q5_K => q5k_matvec(input, data, rows, cols, output),
        GgmlType::Q6_K => q6k_matvec(input, data, rows, cols, output),
        GgmlType::Q4_0 => q4_0_matvec(input, data, rows, cols, output),
        GgmlType::Q4_1 => q4_1_matvec(input, data, rows, cols, output),
        GgmlType::Q5_0 => q5_0_matvec(input, data, rows, cols, output),
        GgmlType::Q5_1 => q5_1_matvec(input, data, rows, cols, output),
        GgmlType::IQ4_XS => iq4_xs_matvec(input, data, rows, cols, output),
        GgmlType::Q8_0 => q8_0_matvec(input, data, rows, cols, output),
        GgmlType::F16 => f16_matvec(input, data, rows, cols, output),
        GgmlType::F32 => f32_matvec(input, data, rows, cols, output),
        GgmlType::Other(_) => panic!("unsupported quantization type: {qtype:?}"),
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
        GgmlType::Q2_K => q2k_matvec_preq(data, rows, cols, q8_blocks, output),
        GgmlType::Q3_K => q3k_matvec_preq(data, rows, cols, q8_blocks, output),
        GgmlType::Q4_K => q4k_matvec_preq(data, rows, cols, q8_blocks, output),
        GgmlType::Q5_K => q5k_matvec_preq(data, rows, cols, q8_blocks, output),
        GgmlType::Q6_K => q6k_matvec_preq(data, rows, cols, q8_blocks, output),
        GgmlType::IQ4_XS => iq4_xs_matvec_preq(data, rows, cols, q8_blocks, output),
        _ => panic!("quantized_matvec_preq only supports Q2_K-Q6_K and IQ4_XS, got {qtype:?}"),
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

// GGUF token type discriminants (per llama.cpp `enum llama_token_type` in
// `include/llama.h`). Populated in the `tokenizer.ggml.token_type` metadata
// array. See:
// https://github.com/ggerganov/llama.cpp/blob/master/include/llama.h
//
// The full set is declared for spec fidelity even when a variant is not
// currently referenced by production paths.
#[allow(dead_code)]
pub(crate) const TOKEN_TYPE_NORMAL: u32 = 1;
#[allow(dead_code)]
pub(crate) const TOKEN_TYPE_UNKNOWN: u32 = 2;
pub(crate) const TOKEN_TYPE_CONTROL: u32 = 3;
pub(crate) const TOKEN_TYPE_USER_DEFINED: u32 = 4;
#[allow(dead_code)]
pub(crate) const TOKEN_TYPE_UNUSED: u32 = 5;
#[allow(dead_code)]
pub(crate) const TOKEN_TYPE_BYTE: u32 = 6;

// SentencePiece byte fallback token format: `<0xNN>` where NN is a 2-digit
// uppercase or lowercase hex value. Used by SPM tokenizers (Llama-1/2, Gemma 2)
// to represent raw bytes that fall outside the vocabulary.
pub(crate) const SPM_BYTE_FALLBACK_PREFIX: &str = "<0x";
pub(crate) const SPM_BYTE_FALLBACK_SUFFIX: &str = ">";
pub(crate) const SPM_BYTE_FALLBACK_HEX_LEN: usize = 2;

// GPT-2 / HuggingFace special token pattern: `<|xxx|>` (e.g. `<|endoftext|>`,
// `<|im_start|>`). Used as a fallback when `tokenizer.ggml.token_type` is
// absent from the GGUF (older Llama-3 conversions).
pub(crate) const GPT2_SPECIAL_PREFIX: &str = "<|";
pub(crate) const GPT2_SPECIAL_SUFFIX: &str = "|>";

/// Parse a SentencePiece byte fallback token `<0xNN>` into its raw byte value.
///
/// Returns `None` if `s` does not match the exact `<0xNN>` format with
/// 2 hex digits.
pub(crate) fn parse_spm_byte_fallback(s: &str) -> Option<u8> {
    let hex = s
        .strip_prefix(SPM_BYTE_FALLBACK_PREFIX)?
        .strip_suffix(SPM_BYTE_FALLBACK_SUFFIX)?;
    if hex.len() != SPM_BYTE_FALLBACK_HEX_LEN {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

/// Returns true if the given GGUF `token_type` discriminant marks the token
/// as an atomic special token that must not be split by BPE
/// (e.g. Qwen 3 `<think>` / `</think>`, `<tool_call>`).
pub(crate) const fn is_atomic_special_token(token_type: u32) -> bool {
    matches!(token_type, TOKEN_TYPE_CONTROL | TOKEN_TYPE_USER_DEFINED)
}

/// Returns true if `s` is a GPT-2 / HuggingFace style special token
/// (`<|xxx|>`), used as a fallback special-token heuristic when the GGUF
/// omits `tokenizer.ggml.token_type`.
pub(crate) fn is_gpt2_special_marker(s: &str) -> bool {
    s.starts_with(GPT2_SPECIAL_PREFIX) && s.ends_with(GPT2_SPECIAL_SUFFIX)
}

/// GGUF tokenizer model type.
/// Detected from `tokenizer.ggml.model` metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerModel {
    /// GPT-2 byte-level BPE (Llama-3, Qwen 2/3, etc.).
    Gpt2Bpe,
    /// SentencePiece with score-based BPE merging (Llama-1/2, Gemma 2).
    /// Uses `▁` (U+2581) as word boundary marker and byte fallback (`<0xNN>`).
    Spm,
}

/// GGUF tokenizer supporting both GPT-2 byte-level BPE and SentencePiece SPM.
pub struct GgufTokenizer {
    tokens: Vec<Vec<u8>>,
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    /// Per-token scores (SPM only). Populated when `model_type == Spm`.
    scores: Vec<f32>,
    token_to_id: HashMap<Vec<u8>, u32>,
    /// Byte fallback tokens `<0x00>`..`<0xFF>` → token id (SPM only).
    byte_to_id: [Option<u32>; 256],
    /// Special tokens (e.g. `<|begin_of_text|>`, `<start_of_turn>`) sorted by
    /// length desc for greedy matching.
    special_tokens: Vec<(String, u32)>,
    /// GPT-2 byte→char mapping (for encoding input text, BPE only).
    byte_encoder: [char; 256],
    /// GPT-2 char→byte mapping (for decoding tokens to text, BPE only).
    byte_decoder: HashMap<char, u8>,
    /// Tokenizer model type (BPE vs SPM). Determines encode/decode path.
    model_type: TokenizerModel,
    pub bos_id: u32,
    pub eos_id: u32,
    /// Whether the tokenizer should prepend BOS to encoded sequences.
    /// Qwen 3 has add_bos_token=False; Llama family defaults to True.
    pub add_bos_token: bool,
}

/// SentencePiece symbol: pointer + length + doubly-linked list indices.
/// Used during SPM tokenization to represent contiguous merged pieces.
#[derive(Clone, Copy)]
struct SpmSymbol {
    /// Byte offset in the preprocessed text.
    text_start: usize,
    /// Byte length of this symbol (0 = merged away).
    text_len: usize,
    /// Index of previous active symbol, or -1 if head.
    prev: i32,
    /// Index of next active symbol, or -1 if tail.
    next: i32,
}

/// Candidate bigram merge with its score.
#[derive(Clone, Copy)]
struct SpmBigram {
    left: usize,
    right: usize,
    score: f32,
    /// Total byte size at enqueue time — used to detect stale entries
    /// whose constituent symbols have been merged elsewhere.
    size: usize,
}

impl PartialEq for SpmBigram {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.left == other.left
    }
}
impl Eq for SpmBigram {}
impl PartialOrd for SpmBigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SpmBigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap: higher score first, then smaller left index.
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl GgufTokenizer {
    /// Load tokenizer from GGUF metadata.
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Option<Self> {
        let tokens_meta = gguf.meta("tokenizer.ggml.tokens")?;
        let token_strs = tokens_meta.as_str_array()?;

        // Detect tokenizer model: "gpt2" (Llama-3, Qwen), "llama" (Gemma 2, Llama-1/2).
        let model_type = match gguf.meta_str("tokenizer.ggml.model") {
            Some("llama") => TokenizerModel::Spm,
            _ => TokenizerModel::Gpt2Bpe,
        };

        let mut tokens = Vec::with_capacity(token_strs.len());
        let mut token_to_id = HashMap::with_capacity(token_strs.len());
        let mut special_tokens = Vec::new();

        // Prefer GGUF `tokenizer.ggml.token_type`: 3 = CONTROL, 4 = USER_DEFINED
        // are atomic / must not be split by BPE (e.g. Qwen 3 `<think>` / `</think>`,
        // `<tool_call>` etc). Falls back to `<|...|>` string pattern when the
        // token_type array is absent (older Llama-3 conversions).
        let token_types = gguf
            .meta("tokenizer.ggml.token_type")
            .and_then(|m| m.as_u32_array());

        // Byte fallback table (SPM): maps byte value 0x00..0xFF → token id.
        // Populated from `<0xNN>` tokens (type=6 BYTE) in the vocabulary.
        let mut byte_to_id: [Option<u32>; 256] = [None; 256];

        for (i, t) in token_strs.iter().enumerate() {
            let bytes = t.as_bytes().to_vec();
            token_to_id.insert(bytes.clone(), i as u32);
            tokens.push(bytes);

            let is_special = match &token_types {
                Some(types) => types.get(i).copied().is_some_and(is_atomic_special_token),
                None => is_gpt2_special_marker(t),
            };
            if is_special {
                special_tokens.push((t.to_string(), i as u32));
            }

            // SPM byte fallback: parse `<0xNN>` token strings.
            if model_type == TokenizerModel::Spm {
                if let Some(b) = parse_spm_byte_fallback(t) {
                    byte_to_id[b as usize] = Some(i as u32);
                }
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

        // SPM scores (only meaningful for `tokenizer.ggml.model == "llama"`).
        // Fallback to empty vec (BPE path ignores scores).
        let scores: Vec<f32> = gguf
            .meta("tokenizer.ggml.scores")
            .and_then(|m| match m {
                MetaValue::Array(arr) => Some(
                    arr.iter()
                        .filter_map(MetaValue::as_f32)
                        .collect::<Vec<f32>>(),
                ),
                _ => None,
            })
            .unwrap_or_default();

        let bos_id = gguf.meta_u32("tokenizer.ggml.bos_token_id").unwrap_or(1);
        let eos_id = gguf.meta_u32("tokenizer.ggml.eos_token_id").unwrap_or(2);
        // Qwen 3: add_bos_token = False (default True for Llama family).
        let add_bos_token = gguf
            .meta_bool("tokenizer.ggml.add_bos_token")
            .unwrap_or(true);

        Some(Self {
            tokens,
            merges,
            scores,
            token_to_id,
            byte_to_id,
            special_tokens,
            byte_encoder: gpt2_byte_to_char(),
            byte_decoder: gpt2_char_to_byte(),
            model_type,
            bos_id,
            eos_id,
            add_bos_token,
        })
    }

    /// Encode text to token IDs.
    /// Handles special tokens as atomic units, then applies GPT-2 byte-level
    /// BPE (for GPT-2 vocab) or SentencePiece SPM (for llama vocab) to the
    /// remaining text segments.
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

            // Encode the text chunk with the appropriate algorithm.
            let chunk = &remaining[..next_boundary];
            match self.model_type {
                TokenizerModel::Gpt2Bpe => result.extend(self.bpe_encode_chunk(chunk)),
                TokenizerModel::Spm => result.extend(self.spm_encode_chunk(chunk)),
            }
            remaining = &remaining[next_boundary..];
        }

        result
    }

    /// SentencePiece SPM encoding using score-based greedy bigram merging.
    /// Ported from llama.cpp `llm_tokenizer_spm_session::tokenize`.
    ///
    /// Preprocessing: replace ' ' with '▁' (U+2581, 3-byte UTF-8) and prepend
    /// '▁' if the chunk doesn't already start with one. Then the algorithm
    /// splits the text into UTF-8 codepoints and greedily merges highest-score
    /// bigrams whose concatenation exists in the vocab. Unmatched symbols fall
    /// back to byte tokens (`<0xNN>`).
    fn spm_encode_chunk(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Preprocess: ' ' → '▁' (U+2581). Prepend '▁' if not already present.
        let mut preprocessed = String::with_capacity(text.len() + 3);
        if !text.starts_with('\u{2581}') {
            preprocessed.push('\u{2581}');
        }
        for c in text.chars() {
            if c == ' ' {
                preprocessed.push('\u{2581}');
            } else {
                preprocessed.push(c);
            }
        }

        let bytes = preprocessed.as_bytes();

        // Split into UTF-8 codepoints as initial symbols.
        let mut symbols: Vec<SpmSymbol> = Vec::new();
        for (byte_idx, c) in preprocessed.char_indices() {
            let clen = c.len_utf8();
            let idx = symbols.len();
            let prev = if idx == 0 { -1 } else { (idx - 1) as i32 };
            symbols.push(SpmSymbol {
                text_start: byte_idx,
                text_len: clen,
                prev,
                next: -1,
            });
        }
        let n_symbols = symbols.len();
        for i in 0..n_symbols.saturating_sub(1) {
            symbols[i].next = (i + 1) as i32;
        }

        // Priority queue of candidate bigrams (max-heap by score).
        let mut work_queue: std::collections::BinaryHeap<SpmBigram> =
            std::collections::BinaryHeap::new();

        let try_add = |symbols: &[SpmSymbol],
                       queue: &mut std::collections::BinaryHeap<SpmBigram>,
                       left: i32,
                       right: i32| {
            if left < 0 || right < 0 {
                return;
            }
            let l = left as usize;
            let r = right as usize;
            let start = symbols[l].text_start;
            let size = symbols[l].text_len + symbols[r].text_len;
            let merged = &bytes[start..start + size];
            if let Some(&id) = self.token_to_id.get(merged) {
                let score = self
                    .scores
                    .get(id as usize)
                    .copied()
                    .unwrap_or(f32::NEG_INFINITY);
                queue.push(SpmBigram {
                    left: l,
                    right: r,
                    score,
                    size,
                });
            }
        };

        // Seed with initial adjacent bigrams.
        for i in 1..n_symbols {
            try_add(&symbols, &mut work_queue, (i - 1) as i32, i as i32);
        }

        // Greedy merge highest-score bigrams.
        while let Some(bigram) = work_queue.pop() {
            let left_len = symbols[bigram.left].text_len;
            let right_len = symbols[bigram.right].text_len;
            // Skip if either symbol was already merged elsewhere.
            if left_len == 0 || right_len == 0 || left_len + right_len != bigram.size {
                continue;
            }
            // Merge right into left, remove right from linked list.
            symbols[bigram.left].text_len += right_len;
            symbols[bigram.right].text_len = 0;
            symbols[bigram.left].next = symbols[bigram.right].next;
            if symbols[bigram.right].next >= 0 {
                let next_idx = symbols[bigram.right].next as usize;
                symbols[next_idx].prev = bigram.left as i32;
            }
            // Enqueue new neighbor bigrams.
            let prev = symbols[bigram.left].prev;
            let next = symbols[bigram.left].next;
            try_add(&symbols, &mut work_queue, prev, bigram.left as i32);
            try_add(&symbols, &mut work_queue, bigram.left as i32, next);
        }

        // Traverse final linked list, emit token ids (with byte fallback).
        let mut result = Vec::new();
        let mut idx: i32 = 0;
        while idx != -1 {
            let sym = symbols[idx as usize];
            let segment = &bytes[sym.text_start..sym.text_start + sym.text_len];
            if let Some(&id) = self.token_to_id.get(segment) {
                result.push(id);
            } else {
                // Byte fallback: emit `<0xNN>` for each byte.
                for &b in segment {
                    if let Some(id) = self.byte_to_id[b as usize] {
                        result.push(id);
                    }
                }
            }
            idx = sym.next;
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
    /// Dispatches to BPE (GPT-2 byte encoding) or SPM (`▁` → space) based on
    /// tokenizer model. Skips control tokens and handles `<0xNN>` byte tokens.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(token) = self.tokens.get(id as usize) else {
                continue;
            };
            let Ok(s) = std::str::from_utf8(token) else {
                bytes.extend_from_slice(token);
                continue;
            };
            // Skip GPT-2 control tokens.
            if is_gpt2_special_marker(s) {
                continue;
            }
            // Skip SPM special tokens (single-piece `<...>` control markers).
            if self.model_type == TokenizerModel::Spm
                && s.starts_with('<')
                && s.ends_with(SPM_BYTE_FALLBACK_SUFFIX)
                && !s.starts_with(SPM_BYTE_FALLBACK_PREFIX)
                && !s.contains('\u{2581}')
            {
                continue;
            }
            // Byte fallback token `<0xNN>` — output raw byte.
            if let Some(byte_val) = parse_spm_byte_fallback(s) {
                bytes.push(byte_val);
                continue;
            }
            match self.model_type {
                TokenizerModel::Spm => {
                    // SPM: '▁' (U+2581) marks word boundary → convert to space.
                    for ch in s.chars() {
                        if ch == '\u{2581}' {
                            bytes.push(b' ');
                        } else {
                            let mut buf = [0u8; 4];
                            bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                    }
                }
                TokenizerModel::Gpt2Bpe => {
                    // BPE: decode GPT-2 unicode chars → raw bytes.
                    for ch in s.chars() {
                        if let Some(&b) = self.byte_decoder.get(&ch) {
                            bytes.push(b);
                        } else {
                            let mut buf = [0u8; 4];
                            bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Vocabulary size.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
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
        let num_bytes = num_cols.div_ceil(8);

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

        Self {
            pos_mask,
            neg_mask,
            scale,
            num_cols,
        }
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
                GgmlType::Q2_K => dequantize_row_q2k(row_data, &mut row_f32),
                GgmlType::Q3_K => dequantize_row_q3k(row_data, &mut row_f32),
                GgmlType::Q4_K => dequantize_row_q4k(row_data, &mut row_f32),
                GgmlType::Q5_K => dequantize_row_q5k(row_data, &mut row_f32),
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

        Self {
            rows,
            num_rows,
            num_cols,
        }
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
        output.par_iter_mut().enumerate().for_each(|(r, out)| {
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
    // Runtime-dispatched (Issue #24): AVX-512F → AVX2 → scalar.
    #[cfg(target_arch = "x86_64")]
    {
        if x86_avx512bw_supported() {
            // SAFETY: guarded by runtime CPU detection; input length is
            // validated against `row.num_cols` on the ternary matvec entry.
            let unscaled = unsafe {
                avx512_dot::ternary_dot_row_unscaled(
                    &row.pos_mask,
                    &row.neg_mask,
                    input,
                    row.num_cols,
                )
            };
            return row.scale * unscaled;
        }
        if x86_avx2_supported() {
            // SAFETY: guarded by runtime CPU detection.
            let unscaled = unsafe {
                avx2_dot::ternary_dot_row_unscaled(
                    &row.pos_mask,
                    &row.neg_mask,
                    input,
                    row.num_cols,
                )
            };
            return row.scale * unscaled;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on every aarch64 target Rust supports.
        // Parity with scalar is relative-tolerance (1e-5), not bit-exact,
        // due to differing summation order — same rationale as the AVX2
        // path documented in `ternary_avx2_matches_scalar_relative_tolerance`.
        let unscaled = unsafe {
            neon_dot::ternary_dot_row_unscaled(&row.pos_mask, &row.neg_mask, input, row.num_cols)
        };
        return row.scale * unscaled;
    }
    #[allow(unreachable_code)]
    ternary_dot_row_scalar(row, input)
}

/// Scalar fallback body for [`ternary_dot_row`]. Kept as a separate
/// function so the SIMD dispatch on x86_64 can call it directly when no
/// feature is available, and so the scalar tests exercise a stable
/// reference implementation.
#[inline]
fn ternary_dot_row_scalar(row: &TernaryRow, input: &[f32]) -> f32 {
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
        for bit in 0..8 {
            let idx = base + bit;
            if idx >= row.num_cols {
                break;
            }
            let p = f32::from((pm >> bit) & 1);
            let n = f32::from((nm >> bit) & 1);
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
///
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
    /// Packed 2-bit weights for LUT expansion: 4 weights per byte (row-major, for scalar/tail).
    /// Layout: contiguous [row0_blk0(4B), row0_blk1(4B), ..., row1_blk0(4B), ...]
    /// Encoding: 00=0, 01=+1, 11=-1 per 2-bit field.
    packed_2bit: Vec<u8>,
    /// Block-packed 2-bit weights for 4-row micro-kernel (TLB-friendly).
    /// Layout: groups of 4 rows, each group interleaved by block:
    /// [g0_r0_b0(4B), g0_r1_b0, g0_r2_b0, g0_r3_b0, g0_r0_b1, g0_r1_b1, ...]
    /// +16 padding for safe vld1q_u8 over-read.
    packed_blocked: Vec<u8>,
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
        let num_blocks = num_cols.div_ceil(SPARSE_BLOCK);
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
                let is_nonzero =
                    (row.pos_mask[byte_idx] & bit) != 0 || (row.neg_mask[byte_idx] & bit) != 0;
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

        Self {
            active_masks,
            sign_masks,
            scale,
            num_cols,
            num_blocks,
        }
    }

    /// Create directly from f32 weights with N:M structured sparsity.
    pub fn from_f32(weights: &[f32], threshold_ratio: f32, n_keep: usize) -> Self {
        let row = TernaryRow::from_f32(weights, threshold_ratio);
        Self::from_ternary_row(&row, weights, n_keep)
    }

    /// Count of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.active_masks
            .iter()
            .map(|m| m.count_ones() as usize)
            .sum()
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

#[allow(dead_code)]
impl SparseTernaryMatrix {
    /// Build from Vec of SparseTernaryRows, packing into flat contiguous buffer.
    /// Also precomputes packed 2-bit weights for LUT-based expansion.
    pub fn from_rows(
        rows: Vec<SparseTernaryRow>,
        num_rows: usize,
        num_cols: usize,
        target_sparsity: f32,
    ) -> Self {
        let blocks_per_row = num_cols.div_ceil(SPARSE_BLOCK);
        let stride = blocks_per_row * 2;
        let mut mask_buf = vec![0u16; num_rows * stride];
        let mut scales = Vec::with_capacity(num_rows);

        // 4 bytes per block (16 elements × 2 bits = 32 bits = 4 bytes)
        let bytes_per_block = SPARSE_BLOCK / 4;
        // +16 padding so vld1q_u8 can safely over-read at buffer tail
        let mut packed_2bit = vec![0u8; num_rows * blocks_per_row * bytes_per_block + 16];

        for (r, row) in rows.iter().enumerate() {
            let base = r * stride;
            mask_buf[base..base + blocks_per_row].copy_from_slice(&row.active_masks);
            mask_buf[base + blocks_per_row..base + stride].copy_from_slice(&row.sign_masks);
            scales.push(row.scale);

            // Pack active + sign into 2-bit format: 00=0, 01=+1, 11=-1
            let pack_base = r * blocks_per_row * bytes_per_block;
            for blk in 0..blocks_per_row {
                let active = row.active_masks[blk];
                let sign = row.sign_masks[blk];
                for byte_idx in 0..bytes_per_block {
                    let bit_off = byte_idx * 4;
                    let mut packed_byte = 0u8;
                    for j in 0..4 {
                        let bit = bit_off + j;
                        let is_active = (active >> bit) & 1;
                        let is_sign = (sign >> bit) & 1;
                        // 00=0, 01=+1, 11=-1
                        let code = if is_active == 0 {
                            0u8
                        } else if is_sign == 0 {
                            1
                        } else {
                            3
                        };
                        packed_byte |= code << (j * 2);
                    }
                    packed_2bit[pack_base + blk * bytes_per_block + byte_idx] = packed_byte;
                }
            }
        }

        // Build block-packed layout: groups of 4 rows, interleaved by block.
        // For 4-row micro-kernel: sequential memory access eliminates TLB misses.
        let num_groups = num_rows.div_ceil(4);
        let group_bytes = 4 * blocks_per_row * bytes_per_block; // 4 rows × all blocks
        let mut packed_blocked = vec![0u8; num_groups * group_bytes + 16]; // +16 padding

        for g in 0..num_groups {
            let r_base = g * 4;
            for blk in 0..blocks_per_row {
                for lane in 0..4 {
                    let r = r_base + lane;
                    if r < num_rows {
                        let src_off = r * blocks_per_row * bytes_per_block + blk * bytes_per_block;
                        let dst_off =
                            g * group_bytes + blk * 4 * bytes_per_block + lane * bytes_per_block;
                        packed_blocked[dst_off..dst_off + bytes_per_block]
                            .copy_from_slice(&packed_2bit[src_off..src_off + bytes_per_block]);
                    }
                }
            }
        }

        Self {
            mask_buf,
            packed_2bit,
            packed_blocked,
            scales,
            num_rows,
            num_cols,
            blocks_per_row,
            target_sparsity,
        }
    }

    /// Convert a dense TernaryMatrix to sparse, enforcing N:M structured sparsity.
    pub fn from_ternary(
        dense: &TernaryMatrix,
        original_weights_per_row: &[Vec<f32>],
        n_keep: usize,
    ) -> Self {
        let target_sparsity = 1.0 - (n_keep as f32 / SPARSE_BLOCK as f32);
        let rows: Vec<SparseTernaryRow> = dense
            .rows
            .iter()
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
            rows.push(SparseTernaryRow::from_f32(
                row_data,
                threshold_ratio,
                n_keep,
            ));
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

    /// Get packed 2-bit weight data for a row (4 bytes per block).
    #[inline]
    fn packed_weights(&self, row: usize) -> &[u8] {
        let bytes_per_block = SPARSE_BLOCK / 4;
        let base = row * self.blocks_per_row * bytes_per_block;
        let len = self.blocks_per_row * bytes_per_block;
        &self.packed_2bit[base..base + len]
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

    /// Estimated memory in bytes (masks + packed_2bit + packed_blocked + scales).
    pub const fn memory_bytes(&self) -> usize {
        self.mask_buf.len() * 2
            + self.packed_2bit.len()
            + self.packed_blocked.len()
            + self.scales.len() * 4
    }
}

/// Quantize f32 activation vector to i8 with a single global scale.
/// Returns (quantized_i8, scale) where original ≈ quantized * scale.
/// On aarch64: NEON-accelerated abs-max search and narrowing conversion.
fn quantize_activation_i8(input: &[f32]) -> (Vec<i8>, f32) {
    let n = input.len();
    let padded_len = n.div_ceil(SPARSE_BLOCK) * SPARSE_BLOCK;

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON mandatory on aarch64
        unsafe { quantize_activation_i8_neon(input, n, padded_len) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let abs_max = input.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        if abs_max == 0.0 {
            return (vec![0i8; padded_len], 1.0);
        }
        let inv_scale = 127.0 / abs_max;
        let scale = abs_max / 127.0;
        let mut quant = vec![0i8; padded_len];
        for (i, &val) in input.iter().enumerate() {
            quant[i] = (val * inv_scale).round().max(-127.0).min(127.0) as i8;
        }
        (quant, scale)
    }
}

/// NEON-accelerated activation quantization: f32 → i8.
/// vmaxvq_f32 for abs-max, vcvtnq_s32_f32 + vqmovn for narrowing.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn quantize_activation_i8_neon(
    input: &[f32],
    n: usize,
    padded_len: usize,
) -> (Vec<i8>, f32) {
    use std::arch::aarch64::*;

    // 1. Find abs max using NEON pairwise max
    let mut max_vec = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + 16 <= n {
        let v0 = vabsq_f32(vld1q_f32(input.as_ptr().add(i)));
        let v1 = vabsq_f32(vld1q_f32(input.as_ptr().add(i + 4)));
        let v2 = vabsq_f32(vld1q_f32(input.as_ptr().add(i + 8)));
        let v3 = vabsq_f32(vld1q_f32(input.as_ptr().add(i + 12)));
        max_vec = vmaxq_f32(max_vec, vmaxq_f32(vmaxq_f32(v0, v1), vmaxq_f32(v2, v3)));
        i += 16;
    }
    while i + 4 <= n {
        let v = vabsq_f32(vld1q_f32(input.as_ptr().add(i)));
        max_vec = vmaxq_f32(max_vec, v);
        i += 4;
    }
    let mut abs_max = vmaxvq_f32(max_vec);
    while i < n {
        abs_max = abs_max.max(input[i].abs());
        i += 1;
    }

    if abs_max == 0.0 {
        return (vec![0i8; padded_len], 1.0);
    }

    let inv_scale = 127.0 / abs_max;
    let scale = abs_max / 127.0;
    let scale_vec = vdupq_n_f32(inv_scale);

    // 2. Quantize using NEON: f32 → round → i32 → saturating narrow i16 → i8
    let mut quant = vec![0i8; padded_len];
    i = 0;
    while i + 16 <= n {
        let v0 = vmulq_f32(vld1q_f32(input.as_ptr().add(i)), scale_vec);
        let v1 = vmulq_f32(vld1q_f32(input.as_ptr().add(i + 4)), scale_vec);
        let v2 = vmulq_f32(vld1q_f32(input.as_ptr().add(i + 8)), scale_vec);
        let v3 = vmulq_f32(vld1q_f32(input.as_ptr().add(i + 12)), scale_vec);

        // Round to nearest → i32
        let i0 = vcvtnq_s32_f32(v0);
        let i1 = vcvtnq_s32_f32(v1);
        let i2 = vcvtnq_s32_f32(v2);
        let i3 = vcvtnq_s32_f32(v3);

        // Saturating narrow: i32 → i16 → i8
        let s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        let s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
        let result = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));

        vst1q_s8(quant.as_mut_ptr().add(i), result);
        i += 16;
    }
    // Scalar tail
    while i < n {
        quant[i] = (input[i] * inv_scale).round().max(-127.0).min(127.0) as i8;
        i += 1;
    }

    (quant, scale)
}

/// Sparse ternary matvec: output = SparseTernaryMatrix × input.
///
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
            .par_chunks_mut(8)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let row_start = chunk_idx * 8;
                let mut pos = 0;
                // Process groups of 4 rows
                while pos + 4 <= chunk.len() {
                    unsafe {
                        sparse_ternary_microkernel_4row_sdot(
                            matrix,
                            &act_i8,
                            act_scale,
                            &mut chunk[pos..pos + 4],
                            row_start + pos,
                        );
                    }
                    pos += 4;
                }
                // Tail rows (1-3)
                while pos < chunk.len() {
                    unsafe {
                        chunk[pos] =
                            sparse_ternary_dot_sdot(matrix, &act_i8, act_scale, row_start + pos);
                    }
                    pos += 1;
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
                    matrix,
                    &act_i8,
                    act_scale,
                    &mut output[r..r + 4],
                    r,
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
            output.par_iter_mut().enumerate().for_each(|(r, out)| {
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
#[allow(dead_code)]
fn sparse_ternary_dot_flat(matrix: &SparseTernaryMatrix, row: usize, input: &[f32]) -> f32 {
    sparse_ternary_dot_scalar(matrix, row, input)
}

/// Single-row sdot-based dot product for tail rows.
/// Uses packed 2-bit weights + vqtbl1q_s8 LUT expansion (~6 ops vs 14 ops mask-based).
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

    let row_scale = matrix.scales[row];
    let blocks = matrix.blocks_per_row;
    let packed = matrix.packed_weights(row);

    // i32 accumulator
    let mut acc = vdupq_n_s32(0);

    // LUT constants — live in registers across all blocks
    let lut = vld1q_s8([0i8, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].as_ptr());
    let replicate_idx = vld1q_u8([0u8, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3].as_ptr());
    let shift_amounts =
        vld1q_s8([0i8, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6].as_ptr());
    let mask_03 = vdupq_n_u8(0x03);

    // Branchless loop: LUT expand + sdot
    let bytes_per_block = SPARSE_BLOCK / 4; // 4
    for blk in 0..blocks {
        let aq = vld1q_s8(act_i8.as_ptr().add(blk * SPARSE_BLOCK));
        let w = expand_packed_2bit_lut(
            packed.as_ptr().add(blk * bytes_per_block),
            lut,
            replicate_idx,
            shift_amounts,
            mask_03,
        );
        acc = sdot_s32(acc, w, aq);
    }

    let sum_i32 = vaddvq_s32(acc);
    row_scale * act_scale * (sum_i32 as f32)
}

/// Branchless 4-row sdot micro-kernel (Block-Packed + LUT + SDOT).
/// - Block-packed layout: 4 rows' data for each block are contiguous in memory → TLB-friendly
/// - i8 quantized activation loaded ONCE per block, shared across 4 rows
/// - Packed 2-bit weights expanded via vqtbl1q_s8 LUT (~6 ops)
/// - sdot: 16 × i8×i8 → 4 × i32 in 1 instruction
/// - ZERO branches in inner loop
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
    let bytes_per_block = SPARSE_BLOCK / 4; // 4 bytes per block

    // 4 × i32 accumulators (one int32x4_t per row)
    let mut acc0 = vdupq_n_s32(0);
    let mut acc1 = vdupq_n_s32(0);
    let mut acc2 = vdupq_n_s32(0);
    let mut acc3 = vdupq_n_s32(0);

    // LUT constants — live in registers across all blocks
    let lut = vld1q_s8([0i8, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].as_ptr());
    let replicate_idx = vld1q_u8([0u8, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3].as_ptr());
    let shift_amounts =
        vld1q_s8([0i8, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6].as_ptr());
    let mask_03 = vdupq_n_u8(0x03);

    // Block-packed pointer: group's data is contiguous
    // Layout: [r0_b0(4B), r1_b0(4B), r2_b0(4B), r3_b0(4B), r0_b1, r1_b1, ...]
    let group = row_start / 4;
    let group_bytes = 4 * blocks * bytes_per_block;
    let group_ptr = matrix.packed_blocked.as_ptr().add(group * group_bytes);
    let stride_4 = 4 * bytes_per_block; // 16 bytes per block-group (4 rows × 4 bytes)

    for blk in 0..blocks {
        // 1. Load i8 activation ONCE — shared across 4 rows
        let aq = vld1q_s8(act_i8.as_ptr().add(blk * SPARSE_BLOCK));

        // Block-packed: all 4 rows for this block are at contiguous addresses
        let blk_base = group_ptr.add(blk * stride_4);

        // 2-5. Sequential loads from contiguous memory (1 cache line = 64B covers 4 blocks)
        let w0 = expand_packed_2bit_lut(blk_base, lut, replicate_idx, shift_amounts, mask_03);
        acc0 = sdot_s32(acc0, w0, aq);

        let w1 = expand_packed_2bit_lut(
            blk_base.add(bytes_per_block),
            lut,
            replicate_idx,
            shift_amounts,
            mask_03,
        );
        acc1 = sdot_s32(acc1, w1, aq);

        let w2 = expand_packed_2bit_lut(
            blk_base.add(2 * bytes_per_block),
            lut,
            replicate_idx,
            shift_amounts,
            mask_03,
        );
        acc2 = sdot_s32(acc2, w2, aq);

        let w3 = expand_packed_2bit_lut(
            blk_base.add(3 * bytes_per_block),
            lut,
            replicate_idx,
            shift_amounts,
            mask_03,
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
///
/// Requires ARMv8.2-A dotprod extension. Enable via `-C target-feature=+dotprod`
/// or on toolchains where the base ABI already implies it (macOS aarch64-apple-darwin).
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
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

/// NEON fallback when ARMv8.2-A dotprod is unavailable.
///
/// Reproduces `sdot` per-lane 4-element dot product semantics via
/// `vmulq_s16` + `vpaddlq_s16` + `vpaddq_s32`:
/// `acc.4s[i] += Σ_{k=0..3} a.i8[4i+k] * b.i8[4i+k]`.
///
/// Verified bit-identical to the dotprod inline asm path by
/// `sdot_s32_matches_manual_reference` / `sdot_s32_accumulates_onto_seed`.
#[cfg(all(target_arch = "aarch64", not(target_feature = "dotprod")))]
#[inline(always)]
unsafe fn sdot_s32(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::{
        vaddq_s32, vget_high_s8, vget_low_s8, vmovl_s8, vmulq_s16, vpaddlq_s16, vpaddq_s32,
    };
    // 1. Sign-extend i8×16 to i16×8 (low + high halves).
    let a_lo = vmovl_s8(vget_low_s8(a)); // a[0..8]  as i16×8
    let a_hi = vmovl_s8(vget_high_s8(a)); // a[8..16] as i16×8
    let b_lo = vmovl_s8(vget_low_s8(b));
    let b_hi = vmovl_s8(vget_high_s8(b));
    // 2. Element-wise multiply (i16 × i16 → i16; values fit since |i8·i8| ≤ 2^14).
    let prod_lo = vmulq_s16(a_lo, b_lo); // [p0,p1,p2,p3, p4,p5,p6,p7]
    let prod_hi = vmulq_s16(a_hi, b_hi); // [p8,..p11, p12,..p15]
                                         // 3. Pairwise widening add (i16×8 → i32×4): adjacent i16 pairs summed.
    let pair_lo = vpaddlq_s16(prod_lo); // [p0+p1, p2+p3, p4+p5, p6+p7]
    let pair_hi = vpaddlq_s16(prod_hi); // [p8+p9, p10+p11, p12+p13, p14+p15]
                                        // 4. Interleaved pairwise add (2×i32×4 → i32×4): concat the two low pairs
                                        //    of each input into per-lane 4-element sums matching SDOT lane layout.
    let sum = vpaddq_s32(pair_lo, pair_hi);
    // 5. Accumulate onto the seed.
    vaddq_s32(acc, sum)
}

/// LUT-based 2-bit → i8 expansion using vqtbl1q_s8.
/// Reads 4 packed bytes (16 weights), expands each 2-bit code via table lookup.
/// 00→0, 01→+1, 11→-1. Only ~6 ops vs 14 ops for mask-based expansion.
///
/// `packed_ptr`: pointer to 4 bytes of packed 2-bit weights.
/// `lut`: [0, 1, 0, -1, 0,0,0,0, 0,0,0,0, 0,0,0,0] preloaded in register.
/// `replicate_idx`: [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3] for byte replication.
/// `shift_amounts`: [0,-2,-4,-6, 0,-2,-4,-6, 0,-2,-4,-6, 0,-2,-4,-6] for 2-bit extraction.
/// `mask_03`: vdupq_n_u8(0x03) for isolating 2-bit codes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn expand_packed_2bit_lut(
    packed_ptr: *const u8,
    lut: std::arch::aarch64::int8x16_t,
    replicate_idx: std::arch::aarch64::uint8x16_t,
    shift_amounts: std::arch::aarch64::int8x16_t,
    mask_03: std::arch::aarch64::uint8x16_t,
) -> std::arch::aarch64::int8x16_t {
    use std::arch::aarch64::*;

    // Load 4 packed bytes into low 4 bytes of a NEON register
    // (use vld1q_u8 from a 16-byte aligned read is safest, but we only use 4 bytes)
    let raw = vld1q_u8(packed_ptr);

    // Replicate: byte 0 → lanes 0-3, byte 1 → lanes 4-7, etc.
    let replicated = vqtbl1q_u8(raw, replicate_idx);

    // Variable right-shift each lane to position its 2-bit field at bits [1:0]
    let shifted = vreinterpretq_u8_s8(vshlq_s8(vreinterpretq_s8_u8(replicated), shift_amounts));

    // Mask to isolate 2-bit code (0-3)
    let indices = vandq_u8(shifted, mask_03);

    // Table lookup: code → i8 weight
    vqtbl1q_s8(lut, indices)
}

/// Scalar fallback (used on non-aarch64 and as reference implementation).
#[inline]
#[allow(dead_code)]
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

/// Dequantize a single row of quantized weight data to f32.
/// Public API for use by model loading (sparse ternary conversion etc.).
pub fn dequantize_weight_row(data: &[u8], qtype: GgmlType, out: &mut [f32]) {
    match qtype {
        GgmlType::Q2_K => dequantize_row_q2k(data, out),
        GgmlType::Q3_K => dequantize_row_q3k(data, out),
        GgmlType::Q4_K => dequantize_row_q4k(data, out),
        GgmlType::Q5_K => dequantize_row_q5k(data, out),
        GgmlType::Q6_K => dequantize_row_q6k(data, out),
        GgmlType::Q8_0 => dequantize_row_q8_0(data, out),
        GgmlType::F16 => {
            for i in 0..out.len() {
                let off = i * 2;
                out[i] = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
            }
        }
        GgmlType::F32 => {
            for i in 0..out.len() {
                out[i] = f32::from_le_bytes([
                    data[i * 4],
                    data[i * 4 + 1],
                    data[i * 4 + 2],
                    data[i * 4 + 3],
                ]);
            }
        }
        _ => out.fill(0.0),
    }
}

/// Dequantize a single Q3_K row to f32.
fn dequantize_row_q2k(data: &[u8], out: &mut [f32]) {
    dequantize_q2_k(data, out);
}

fn dequantize_row_q3k(data: &[u8], out: &mut [f32]) {
    dequantize_q3_k(data, out);
}

fn dequantize_row_q5k(data: &[u8], out: &mut [f32]) {
    dequantize_q5_k(data, out);
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
        scales[4] = (scales_raw[8] & 0xF) | ((scales_raw[0] >> 6) << 4);
        scales[5] = (scales_raw[1 + 8] & 0xF) | ((scales_raw[1] >> 6) << 4);
        scales[6] = (scales_raw[2 + 8] & 0xF) | ((scales_raw[2] >> 6) << 4);
        scales[7] = (scales_raw[3 + 8] & 0xF) | ((scales_raw[3] >> 6) << 4);
        mins[4] = (scales_raw[8] >> 4) | ((scales_raw[4] >> 6) << 4);
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
                    if is % 2 == 0 {
                        qs[byte_idx] & 0xF
                    } else {
                        qs[byte_idx] >> 4
                    }
                } else {
                    let byte_idx = (is / 2) * 32 + 16 + (l - 16);
                    if is % 2 == 0 {
                        qs[byte_idx] & 0xF
                    } else {
                        qs[byte_idx] >> 4
                    }
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
                aux8[a_off + l] = ((ql[ql_off + l] & 0xF) | ((qh[qh_off + l] & 3) << 4)) as i8 - 32;
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
            1.0, -2.0, 0.5, -0.1, 3.0, -0.3, 0.8, -0.05, 0.02, -1.5, 0.7, -0.9, 0.01, -0.4, 2.5,
            -0.6,
        ];
        let row = SparseTernaryRow::from_f32(&weights, 0.1, 4);
        assert_eq!(row.active_masks[0].count_ones(), 4);
        assert_eq!(row.nnz(), 4);
    }

    #[test]
    fn test_sparse_ternary_matvec_matches_dense() {
        // Create a small matrix, compare sparse vs dense ternary matvec
        let weights = vec![
            1.0, -2.0, 0.5, -0.1, 3.0, -0.3, 0.8, -0.05, 0.02, -1.5, 0.7, -0.9, 0.01, -0.4, 2.5,
            -0.6, // Row 2
            -1.0, 2.0, -0.5, 0.1, -3.0, 0.3, -0.8, 0.05, -0.02, 1.5, -0.7, 0.9, -0.01, 0.4, -2.5,
            0.6,
        ];
        let num_rows = 2;
        let num_cols = 16;

        // Dense ternary (keep all)
        let dense = TernaryMatrix {
            rows: (0..num_rows)
                .map(|r| TernaryRow::from_f32(&weights[r * num_cols..(r + 1) * num_cols], 0.05))
                .collect(),
            num_rows,
            num_cols,
        };

        // Sparse ternary (keep 16 = all, so should match dense)
        let sparse = SparseTernaryMatrix::from_f32_weights(&weights, num_rows, num_cols, 0.05, 16);

        let input = vec![
            1.0, 0.5, -1.0, 2.0, 0.3, -0.7, 1.5, -0.2, 0.8, -1.2, 0.4, 0.6, -0.9, 1.1, -0.3, 0.7,
        ];
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
                "row {i}: dense={} sparse={}",
                dense_out[i],
                sparse_out[i]
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
        let weights = vec![
            0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
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
            "result={}, expected={expected}",
            output[0]
        );
    }

    /// Verify `sdot_s32` matches the manual reference for known i8 inputs.
    /// This test runs on both dotprod-enabled (inline asm) and fallback
    /// (NEON `vmlal_s16`) implementations, chosen at compile time by
    /// `#[cfg(target_feature = "dotprod")]`, so it validates semantic
    /// equivalence between the two paths across ARMv8.2-A and pre-dotprod
    /// aarch64 targets.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn sdot_s32_matches_manual_reference() {
        use std::arch::aarch64::{vdupq_n_s32, vld1q_s8, vst1q_s32};
        unsafe {
            let acc = vdupq_n_s32(0);
            let a_arr: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let b_arr: [i8; 16] = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1];
            let a = vld1q_s8(a_arr.as_ptr());
            let b = vld1q_s8(b_arr.as_ptr());
            let result = super::sdot_s32(acc, a, b);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            // lane i accumulates Σ_{k=0..3} a[4i+k] * b[4i+k].
            // lane 0: 1 - 2 + 3 - 4 = -2
            // lane 1: 5 - 6 + 7 - 8 = -2
            // lane 2: 9 - 10 + 11 - 12 = -2
            // lane 3: 13 - 14 + 15 - 16 = -2
            assert_eq!(out, [-2, -2, -2, -2]);
        }
    }

    /// Verify `sdot_s32` accumulates onto a non-zero seed correctly.
    /// Complements `sdot_s32_matches_manual_reference` by exercising
    /// the `acc +=` semantics (rather than `acc = 0`).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn sdot_s32_accumulates_onto_seed() {
        use std::arch::aarch64::{vld1q_s32, vld1q_s8, vst1q_s32};
        unsafe {
            let seed: [i32; 4] = [100, 200, 300, 400];
            let acc = vld1q_s32(seed.as_ptr());
            let a_arr: [i8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
            let b_arr: [i8; 16] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
            let a = vld1q_s8(a_arr.as_ptr());
            let b = vld1q_s8(b_arr.as_ptr());
            let result = super::sdot_s32(acc, a, b);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            // Each lane adds 1*2 + 1*2 + 1*2 + 1*2 = 8 to the seed.
            assert_eq!(out, [108, 208, 308, 408]);
        }
    }

    // ─── Tokenizer const + helper regression tests (Issue #14) ──────────────

    #[test]
    fn token_type_constants_stable() {
        // Regression detection: llama.cpp `enum llama_token_type` in llama.h
        // The GGUF spec pins these discriminants, so accidental drift here
        // silently breaks special-token detection for every GGUF ever produced.
        assert_eq!(TOKEN_TYPE_NORMAL, 1);
        assert_eq!(TOKEN_TYPE_UNKNOWN, 2);
        assert_eq!(TOKEN_TYPE_CONTROL, 3);
        assert_eq!(TOKEN_TYPE_USER_DEFINED, 4);
        assert_eq!(TOKEN_TYPE_UNUSED, 5);
        assert_eq!(TOKEN_TYPE_BYTE, 6);
    }

    #[test]
    fn is_atomic_special_token_matches_control_and_user_defined() {
        assert!(is_atomic_special_token(TOKEN_TYPE_CONTROL));
        assert!(is_atomic_special_token(TOKEN_TYPE_USER_DEFINED));
        assert!(!is_atomic_special_token(TOKEN_TYPE_NORMAL));
        assert!(!is_atomic_special_token(TOKEN_TYPE_UNKNOWN));
        assert!(!is_atomic_special_token(TOKEN_TYPE_UNUSED));
        assert!(!is_atomic_special_token(TOKEN_TYPE_BYTE));
    }

    #[test]
    fn parse_spm_byte_fallback_valid() {
        assert_eq!(parse_spm_byte_fallback("<0x00>"), Some(0x00));
        assert_eq!(parse_spm_byte_fallback("<0xFF>"), Some(0xFF));
        assert_eq!(parse_spm_byte_fallback("<0xAB>"), Some(0xAB));
        // Lowercase hex is accepted by `u8::from_str_radix`.
        assert_eq!(parse_spm_byte_fallback("<0xab>"), Some(0xAB));
        assert_eq!(parse_spm_byte_fallback("<0xA5>"), Some(0xA5));
    }

    #[test]
    fn parse_spm_byte_fallback_rejects_invalid() {
        // Non-hex digit
        assert_eq!(parse_spm_byte_fallback("<0xG0>"), None);
        // Too short (1 hex digit)
        assert_eq!(parse_spm_byte_fallback("<0x0>"), None);
        // Too long (3 hex digits)
        assert_eq!(parse_spm_byte_fallback("<0x000>"), None);
        // Missing wrapping angle brackets
        assert_eq!(parse_spm_byte_fallback("0xFF"), None);
        // GPT-2 special marker (starts with `<|`)
        assert_eq!(parse_spm_byte_fallback("<|endoftext|>"), None);
        // Empty
        assert_eq!(parse_spm_byte_fallback(""), None);
        // SPM special marker (`<...>` but not `<0x`)
        assert_eq!(parse_spm_byte_fallback("<unk>"), None);
    }

    #[test]
    fn is_gpt2_special_marker_matches_expected_pattern() {
        assert!(is_gpt2_special_marker("<|endoftext|>"));
        assert!(is_gpt2_special_marker("<|im_start|>"));
        assert!(is_gpt2_special_marker("<|im_end|>"));
        assert!(!is_gpt2_special_marker("<unk>"));
        assert!(!is_gpt2_special_marker("<0xFF>"));
        assert!(!is_gpt2_special_marker("hello"));
        assert!(!is_gpt2_special_marker(""));
    }

    // ── x86_64 SIMD quantized dot product parity (Issue #13) ─────────────

    /// Build a deterministic Q4_K block filled with `seed`-derived bytes so
    /// the SIMD ↔ scalar comparison covers every byte of the layout.
    #[cfg(target_arch = "x86_64")]
    fn make_q4k_block(seed: u8) -> Vec<u8> {
        let mut block = vec![0u8; 144];
        // Header: pick small non-zero f16 values so `d * total` doesn't
        // saturate to inf or underflow to 0.
        block[0..2].copy_from_slice(&0x3800u16.to_le_bytes()); // f16(0.5)
        block[2..4].copy_from_slice(&0x3400u16.to_le_bytes()); // f16(0.25)
                                                               // Scales / mins header (12 bytes): fill with the seed so the utmp
                                                               // reconstruction hits every branch.
        for i in 0..12 {
            block[4 + i] = seed.wrapping_add(i as u8);
        }
        // 128 packed nibble bytes: spread the seed so both low and high
        // nibbles vary across the block.
        for i in 0..128 {
            block[16 + i] = seed.wrapping_mul(3).wrapping_add(i as u8);
        }
        block
    }

    /// Build a deterministic Q6_K block (210 bytes).
    #[cfg(target_arch = "x86_64")]
    fn make_q6k_block(seed: u8) -> Vec<u8> {
        let mut block = vec![0u8; 210];
        // ql: 128 bytes
        for i in 0..128 {
            block[i] = seed.wrapping_add(i as u8);
        }
        // qh: 64 bytes (top 2 bits of each 6-bit value)
        for i in 0..64 {
            block[128 + i] = seed.wrapping_mul(5).wrapping_add(i as u8);
        }
        // scales: 16 bytes, treated as signed by the dispatcher
        for i in 0..16 {
            block[192 + i] = (i as u8).wrapping_sub(4);
        }
        // d: small non-zero f16
        block[208..210].copy_from_slice(&0x3800u16.to_le_bytes());
        block
    }

    /// Build a Q8_K block whose contents cover both positive and negative
    /// activations, so the sign trick paths are actually exercised.
    #[cfg(target_arch = "x86_64")]
    fn make_q8k_block(seed: u8) -> BlockQ8K {
        let mut block = BlockQ8K {
            d: 0.125,
            qs: [0i8; QK_K],
            bsums: [0i16; 16],
        };
        for i in 0..QK_K {
            let raw = seed.wrapping_add(i as u8);
            block.qs[i] = (raw as i8).wrapping_sub(64);
        }
        for j in 0..16 {
            let mut acc = 0i32;
            for l in 0..16 {
                acc += block.qs[j * 16 + l] as i32;
            }
            block.bsums[j] = acc as i16;
        }
        block
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q4k_avx2_matches_scalar_bit_exact() {
        if !x86_avx2_supported() {
            eprintln!("SKIP: AVX2 unavailable on this runner");
            return;
        }
        for seed in [0u8, 1, 7, 128, 200, 255] {
            let q4k = make_q4k_block(seed);
            let q8k = make_q8k_block(seed.wrapping_mul(13));
            let scalar = q4k_q8k_dot_fallback_scalar(&q4k, &q8k);
            // SAFETY: guarded by `x86_avx2_supported()` above.
            let simd = unsafe { avx2_dot::q4k_q8k_dot(&q4k, &q8k) };
            assert_eq!(
                scalar.to_bits(),
                simd.to_bits(),
                "seed={seed}: scalar={scalar} simd={simd}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q6k_avx2_matches_scalar_bit_exact() {
        if !x86_avx2_supported() {
            eprintln!("SKIP: AVX2 unavailable on this runner");
            return;
        }
        for seed in [0u8, 1, 7, 128, 200, 255] {
            let q6k = make_q6k_block(seed);
            let q8k = make_q8k_block(seed.wrapping_mul(11));
            let scalar = q6k_q8k_dot_fallback_scalar(&q6k, &q8k);
            // SAFETY: guarded by AVX2 detection.
            let simd = unsafe { avx2_dot::q6k_q8k_dot(&q6k, &q8k) };
            assert_eq!(
                scalar.to_bits(),
                simd.to_bits(),
                "seed={seed}: scalar={scalar} simd={simd}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q4k_avx512_matches_scalar_bit_exact() {
        if !x86_avx512bw_supported() {
            eprintln!("SKIP: AVX-512BW unavailable on this runner");
            return;
        }
        for seed in [0u8, 1, 7, 128, 200, 255] {
            let q4k = make_q4k_block(seed);
            let q8k = make_q8k_block(seed.wrapping_mul(13));
            let scalar = q4k_q8k_dot_fallback_scalar(&q4k, &q8k);
            // SAFETY: guarded by AVX-512BW detection.
            let simd = unsafe { avx512_dot::q4k_q8k_dot(&q4k, &q8k) };
            assert_eq!(
                scalar.to_bits(),
                simd.to_bits(),
                "seed={seed}: scalar={scalar} simd={simd}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q6k_avx512_matches_scalar_bit_exact() {
        if !x86_avx512bw_supported() {
            eprintln!("SKIP: AVX-512BW unavailable on this runner");
            return;
        }
        for seed in [0u8, 1, 7, 128, 200, 255] {
            let q6k = make_q6k_block(seed);
            let q8k = make_q8k_block(seed.wrapping_mul(11));
            let scalar = q6k_q8k_dot_fallback_scalar(&q6k, &q8k);
            // SAFETY: guarded by AVX-512BW detection.
            let simd = unsafe { avx512_dot::q6k_q8k_dot(&q6k, &q8k) };
            assert_eq!(
                scalar.to_bits(),
                simd.to_bits(),
                "seed={seed}: scalar={scalar} simd={simd}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn feature_detection_caches_result() {
        // OnceLock ensures the CPU detection cost is paid at most once.
        let first = x86_avx2_supported();
        let second = x86_avx2_supported();
        assert_eq!(first, second);
        let first_512 = x86_avx512bw_supported();
        let second_512 = x86_avx512bw_supported();
        assert_eq!(first_512, second_512);
    }

    /// Build a deterministic Q5_K block (176 bytes) so the qh + qs
    /// combined 5-bit path is exercised on both scalar and SIMD paths.
    #[cfg(target_arch = "x86_64")]
    fn make_q5k_block(seed: u8) -> Vec<u8> {
        let mut block = vec![0u8; 176];
        // d, dmin: small non-zero f16.
        block[0..2].copy_from_slice(&0x3800u16.to_le_bytes()); // f16(0.5)
        block[2..4].copy_from_slice(&0x3400u16.to_le_bytes()); // f16(0.25)
                                                               // 12-byte scales/mins packed header.
        for i in 0..12 {
            block[4 + i] = seed.wrapping_add(i as u8);
        }
        // qh: 32 bytes — 5th bit per element, varied so every im iteration
        // touches a different bit-position pattern.
        for i in 0..32 {
            block[16 + i] = seed.wrapping_mul(7).wrapping_add(i as u8);
        }
        // qs: 128 packed nibble bytes.
        for i in 0..128 {
            block[48 + i] = seed.wrapping_mul(3).wrapping_add(i as u8);
        }
        block
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q5k_avx2_matches_scalar_bit_exact() {
        if !x86_avx2_supported() {
            eprintln!("SKIP: AVX2 unavailable on this runner");
            return;
        }
        for seed in [0u8, 1, 7, 128, 200, 255] {
            let q5k = make_q5k_block(seed);
            let q8k = make_q8k_block(seed.wrapping_mul(17));
            let scalar = q5k_q8k_dot_fallback_scalar(&q5k, &q8k);
            // SAFETY: guarded by AVX2 detection.
            let simd = unsafe { avx2_dot::q5k_q8k_dot(&q5k, &q8k) };
            assert_eq!(
                scalar.to_bits(),
                simd.to_bits(),
                "seed={seed}: scalar={scalar} simd={simd}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q5k_avx512_matches_scalar_bit_exact() {
        if !x86_avx512bw_supported() {
            eprintln!("SKIP: AVX-512BW unavailable on this runner");
            return;
        }
        for seed in [0u8, 1, 7, 128, 200, 255] {
            let q5k = make_q5k_block(seed);
            let q8k = make_q8k_block(seed.wrapping_mul(17));
            let scalar = q5k_q8k_dot_fallback_scalar(&q5k, &q8k);
            // SAFETY: guarded by AVX-512BW detection.
            let simd = unsafe { avx512_dot::q5k_q8k_dot(&q5k, &q8k) };
            assert_eq!(
                scalar.to_bits(),
                simd.to_bits(),
                "seed={seed}: scalar={scalar} simd={simd}"
            );
        }
    }

    // ── Q8_0 SIMD parity (f32 FMA path, not integer madd) ───────────────

    /// Build a deterministic Q8_0-formatted row for a 1-row × `cols`
    /// weight matrix. Each block = 2-byte f16 scale + 32 i8 quantised
    /// weights, so the row occupies `blocks_per_row * 34` bytes.
    #[cfg(target_arch = "x86_64")]
    fn make_q8_0_row_bytes(seed: u8, cols: usize) -> Vec<u8> {
        let blocks_per_row = cols / QK8_0;
        let mut data = Vec::with_capacity(blocks_per_row * 34);
        for bi in 0..blocks_per_row {
            let scale_bits = 0x3800u16.wrapping_add(bi as u16); // varies per block
            data.extend_from_slice(&scale_bits.to_le_bytes());
            for l in 0..QK8_0 {
                let raw = seed.wrapping_add((bi * QK8_0 + l) as u8);
                data.push((raw as i8).wrapping_sub(32) as u8);
            }
        }
        data
    }

    #[cfg(target_arch = "x86_64")]
    fn make_q8_0_input(seed: u8, cols: usize) -> Vec<f32> {
        (0..cols)
            .map(|i| {
                let raw = seed.wrapping_add(i as u8) as i8;
                f32::from(raw) * 0.01
            })
            .collect()
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q8_0_avx2_matches_scalar_relative_tolerance() {
        if !x86_avx2_supported() {
            eprintln!("SKIP: AVX2 unavailable on this runner");
            return;
        }
        const COLS: usize = 128;
        for seed in [0u8, 1, 7, 128, 200, 255] {
            let data = make_q8_0_row_bytes(seed, COLS);
            let input = make_q8_0_input(seed.wrapping_mul(3), COLS);
            let mut scalar_out = vec![0.0f32];
            let mut simd_out = vec![0.0f32];
            // Scalar baseline uses the pre-existing dispatch (before the
            // AVX2 branch it was the only implementation), so we call the
            // dispatched entry twice with matching inputs and compare the
            // f32 outputs within a relative tolerance — the SIMD FMA and
            // scalar sums evaluate in different orders and may drift by
            // one or two ulps of accumulated round-off.
            let row_data = &data[..];
            // Force scalar arm by directly running the fallback body.
            let mut acc = 0.0f32;
            for bi in 0..(COLS / QK8_0) {
                let off = bi * 34;
                let d = f16_to_f32(u16::from_le_bytes([row_data[off], row_data[off + 1]]));
                let col_base = bi * QK8_0;
                for l in 0..QK8_0 {
                    let w = d * f32::from(row_data[off + 2 + l] as i8);
                    acc += w * input[col_base + l];
                }
            }
            scalar_out[0] = acc;
            // SIMD via dispatched entry point (runtime feature detection
            // will pick AVX2 on this test's precondition).
            q8_0_matvec(&input, row_data, 1, COLS, &mut simd_out);
            let diff = (scalar_out[0] - simd_out[0]).abs();
            let rel = diff / scalar_out[0].abs().max(1.0);
            assert!(
                rel < 1e-5,
                "seed={seed}: scalar={} simd={} rel={rel}",
                scalar_out[0],
                simd_out[0]
            );
        }
    }

    // ── Ternary bitmask SIMD parity (Issue #24) ────────────────────────

    /// Build a deterministic TernaryRow for a given seed and column count.
    /// Alternates between +1 / -1 / 0 driven by the seed so the SIMD path
    /// exercises both `pos_mask` and `neg_mask` in every 8-lane group.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn make_ternary_row(seed: u8, num_cols: usize, scale: f32) -> TernaryRow {
        let num_bytes = num_cols.div_ceil(8);
        let mut pos_mask = vec![0u8; num_bytes];
        let mut neg_mask = vec![0u8; num_bytes];
        for i in 0..num_cols {
            let raw = seed.wrapping_mul(3).wrapping_add(i as u8) as i8;
            let byte_idx = i / 8;
            let bit = 1u8 << (i % 8);
            if raw > 20 {
                pos_mask[byte_idx] |= bit;
            } else if raw < -20 {
                neg_mask[byte_idx] |= bit;
            }
        }
        TernaryRow {
            pos_mask,
            neg_mask,
            scale,
            num_cols,
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn make_ternary_input(seed: u8, num_cols: usize) -> Vec<f32> {
        (0..num_cols)
            .map(|i| f32::from(seed.wrapping_add(i as u8) as i8) * 0.01)
            .collect()
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn ternary_avx2_matches_scalar_relative_tolerance() {
        if !x86_avx2_supported() {
            eprintln!("SKIP: AVX2 unavailable on this runner");
            return;
        }
        // 128 covers 16 mask bytes = plenty of both empty and populated
        // 8-lane groups; the divisible-by-8 count also matches typical
        // hidden-dim shapes so the fast path (no tail) is dominant.
        const COLS: usize = 128;
        for seed in [0u8, 1, 7, 128, 200, 255] {
            let row = make_ternary_row(seed, COLS, 0.125);
            let input = make_ternary_input(seed.wrapping_mul(7), COLS);
            let scalar = ternary_dot_row_scalar(&row, &input);
            // Force AVX2 arm via the runtime dispatch (test precondition
            // covers AVX-512 skip via feature detection above).
            let simd = ternary_dot_row(&row, &input);
            let diff = (scalar - simd).abs();
            let rel = diff / scalar.abs().max(1e-6);
            assert!(
                rel < 1e-5,
                "seed={seed}: scalar={scalar} simd={simd} rel={rel}"
            );
        }
    }

    /// Tail path (non-multiple-of-8 column count) must not be silently
    /// dropped by the SIMD full-byte loop.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn ternary_avx2_handles_non_multiple_of_8_cols() {
        if !x86_avx2_supported() {
            return;
        }
        // 100 = 12 full bytes + 4 tail bits → forces the scalar tail loop.
        const COLS: usize = 100;
        for seed in [1u8, 42, 200] {
            let row = make_ternary_row(seed, COLS, 0.5);
            let input = make_ternary_input(seed.wrapping_mul(11), COLS);
            let scalar = ternary_dot_row_scalar(&row, &input);
            let simd = ternary_dot_row(&row, &input);
            let rel = (scalar - simd).abs() / scalar.abs().max(1e-6);
            assert!(
                rel < 1e-5,
                "seed={seed}: scalar={scalar} simd={simd} rel={rel}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn ternary_avx512_matches_scalar_relative_tolerance() {
        if !x86_avx512bw_supported() {
            eprintln!("SKIP: AVX-512BW unavailable on this runner");
            return;
        }
        // 16-lane grouping: 128 cols = 8 full 16-lane groups.
        const COLS: usize = 128;
        for seed in [0u8, 1, 7, 128, 200, 255] {
            let row = make_ternary_row(seed, COLS, 0.125);
            let input = make_ternary_input(seed.wrapping_mul(7), COLS);
            let scalar = ternary_dot_row_scalar(&row, &input);
            let simd = ternary_dot_row(&row, &input);
            let rel = (scalar - simd).abs() / scalar.abs().max(1e-6);
            assert!(
                rel < 1e-5,
                "seed={seed}: scalar={scalar} simd={simd} rel={rel}"
            );
        }
    }

    // ── aarch64 NEON parity tests (gates the dispatch enable) ─────────

    /// Build a deterministic Q4_K block for parity testing. Same shape as
    /// the x86 tests so the aarch64 CI runs the same coverage matrix.
    #[cfg(target_arch = "aarch64")]
    fn make_q4k_block_aarch64(seed: u8) -> Vec<u8> {
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
        block[2..4].copy_from_slice(&0x3400u16.to_le_bytes());
        for i in 0..12 {
            block[4 + i] = seed.wrapping_add(i as u8);
        }
        for i in 0..128 {
            block[16 + i] = seed.wrapping_mul(3).wrapping_add(i as u8);
        }
        block
    }

    #[cfg(target_arch = "aarch64")]
    fn make_q6k_block_aarch64(seed: u8) -> Vec<u8> {
        let mut block = vec![0u8; 210];
        for i in 0..128 {
            block[i] = seed.wrapping_add(i as u8);
        }
        for i in 0..64 {
            block[128 + i] = seed.wrapping_mul(5).wrapping_add(i as u8);
        }
        for i in 0..16 {
            block[192 + i] = (i as u8).wrapping_sub(4);
        }
        block[208..210].copy_from_slice(&0x3800u16.to_le_bytes());
        block
    }

    #[cfg(target_arch = "aarch64")]
    fn make_q8k_block_aarch64(seed: u8) -> BlockQ8K {
        let mut block = BlockQ8K {
            d: 0.125,
            qs: [0i8; QK_K],
            bsums: [0i16; 16],
        };
        for i in 0..QK_K {
            let raw = seed.wrapping_add(i as u8);
            block.qs[i] = (raw as i8).wrapping_sub(64);
        }
        for j in 0..16 {
            let mut acc = 0i32;
            for l in 0..16 {
                acc += block.qs[j * 16 + l] as i32;
            }
            block.bsums[j] = acc as i16;
        }
        block
    }

    /// Verify the dormant NEON Q4_K kernel matches the scalar reference
    /// before the dispatcher is allowed to route to it. NEON was compiled
    /// but unused since PR #21; this test is the gate that decides whether
    /// enabling the dispatch is safe.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q4k_neon_matches_scalar_bit_exact() {
        for seed in [0u8, 1, 7, 128, 200, 255] {
            let q4k = make_q4k_block_aarch64(seed);
            let q8k = make_q8k_block_aarch64(seed.wrapping_mul(13));
            let scalar = q4k_q8k_dot_fallback_scalar(&q4k, &q8k);
            // SAFETY: NEON is baseline on all aarch64 targets Rust supports,
            // so the `#[target_feature(enable = "neon")]` annotation matches
            // the compile target unconditionally.
            let neon = unsafe { neon_dot::q4k_q8k_dot(&q4k, &q8k) };
            assert_eq!(
                scalar.to_bits(),
                neon.to_bits(),
                "seed={seed}: scalar={scalar} neon={neon}"
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q6k_neon_matches_scalar_bit_exact() {
        for seed in [0u8, 1, 7, 128, 200, 255] {
            let q6k = make_q6k_block_aarch64(seed);
            let q8k = make_q8k_block_aarch64(seed.wrapping_mul(11));
            let scalar = q6k_q8k_dot_fallback_scalar(&q6k, &q8k);
            // SAFETY: same rationale as `q4k_neon_matches_scalar_bit_exact`.
            let neon = unsafe { neon_dot::q6k_q8k_dot(&q6k, &q8k) };
            assert_eq!(
                scalar.to_bits(),
                neon.to_bits(),
                "seed={seed}: scalar={scalar} neon={neon}"
            );
        }
    }

    /// Ternary NEON is verified within a relative tolerance (1e-5) rather
    /// than bit-exact because the parallel adds re-order summation vs the
    /// scalar sequential path — same rationale as the AVX2 kernel.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn ternary_neon_matches_scalar_relative_tolerance() {
        const COLS: usize = 128;
        for seed in [0u8, 1, 7, 128, 200, 255] {
            let row = make_ternary_row(seed, COLS, 0.125);
            let input = make_ternary_input(seed.wrapping_mul(7), COLS);
            let scalar = ternary_dot_row_scalar(&row, &input);
            let simd = ternary_dot_row(&row, &input);
            let rel = (scalar - simd).abs() / scalar.abs().max(1e-6);
            assert!(
                rel < 1e-5,
                "seed={seed}: scalar={scalar} simd={simd} rel={rel}"
            );
        }
    }

    /// Tail path (non-multiple-of-8 column count) must not be silently
    /// dropped by the SIMD full-byte loop.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn ternary_neon_handles_non_multiple_of_8_cols() {
        const COLS: usize = 100;
        for seed in [1u8, 42, 200] {
            let row = make_ternary_row(seed, COLS, 0.5);
            let input = make_ternary_input(seed.wrapping_mul(11), COLS);
            let scalar = ternary_dot_row_scalar(&row, &input);
            let simd = ternary_dot_row(&row, &input);
            let rel = (scalar - simd).abs() / scalar.abs().max(1e-6);
            assert!(
                rel < 1e-5,
                "seed={seed}: scalar={scalar} simd={simd} rel={rel}"
            );
        }
    }

    /// Q8_0 NEON is verified against scalar within a tight relative tolerance
    /// rather than bit-exact because the paths use different accumulation:
    /// scalar does `w * input + acc` (two roundings) whereas NEON uses
    /// `vfmaq_f32(acc, f, x)` (single fused rounding). The tolerance below is
    /// chosen empirically: for a 4096-column matvec, cumulative float error
    /// stays under 1e-4 relative on `f32::EPSILON * cols`-order inputs.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q8_0_neon_matches_scalar_within_tol() {
        const COLS: usize = 4096;
        const BLOCK_BYTES: usize = 34;
        const BLOCKS: usize = COLS / QK8_0;
        for seed in [0u8, 1, 7, 128, 200, 255] {
            let mut row_data = vec![0u8; BLOCKS * BLOCK_BYTES];
            for bi in 0..BLOCKS {
                let off = bi * BLOCK_BYTES;
                // f16 scale from a safe range (no denormals, no Inf/NaN).
                // 0x3400 = 0.25, 0x3800 = 0.5, 0x3c00 = 1.0. Vary within
                // 0x3400..0x3c00 so scales stay in [0.25, 1.0).
                let d_bits = 0x3400u16.wrapping_add((seed as u16 ^ bi as u16) & 0x07FF);
                row_data[off..off + 2].copy_from_slice(&d_bits.to_le_bytes());
                for l in 0..QK8_0 {
                    row_data[off + 2 + l] = seed
                        .wrapping_add((bi as u8).wrapping_mul(3))
                        .wrapping_add(l as u8);
                }
            }
            let input: Vec<f32> = (0..COLS)
                .map(|i| ((i as f32 * 0.017 + seed as f32).sin() * 0.5))
                .collect();

            // Scalar reference.
            let mut scalar_out = [0.0f32];
            {
                let mut acc = 0.0f32;
                for bi in 0..BLOCKS {
                    let off = bi * BLOCK_BYTES;
                    let d = f16_to_f32(u16::from_le_bytes([row_data[off], row_data[off + 1]]));
                    let col_base = bi * QK8_0;
                    for l in 0..QK8_0 {
                        let w = d * f32::from(row_data[off + 2 + l] as i8);
                        acc += w * input[col_base + l];
                    }
                }
                scalar_out[0] = acc;
            }
            // SAFETY: NEON is baseline on all aarch64 targets.
            let neon = unsafe { neon_dot::q8_0_matvec_row(&input, &row_data, COLS) };

            let diff = (scalar_out[0] - neon).abs();
            let rel = diff / scalar_out[0].abs().max(f32::EPSILON);
            assert!(
                rel < 1e-4,
                "seed={seed}: scalar={} neon={neon} diff={diff} rel={rel}",
                scalar_out[0]
            );
        }
    }
}
