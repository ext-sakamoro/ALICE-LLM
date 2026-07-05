//! quantization.

// Quantization (INT8 / INT4)
// ---------------------------------------------------------------------------

/// INT8 quantized tensor with scale factor.
#[derive(Debug, Clone)]
pub struct QuantizedInt8 {
    pub data: Vec<i8>,
    pub scale: f32,
    pub zero_point: f32,
}

impl QuantizedInt8 {
    /// Quantize a float slice to INT8.
    #[must_use]
    pub fn quantize(values: &[f32]) -> Self {
        if values.is_empty() {
            return Self {
                data: Vec::new(),
                scale: 1.0,
                zero_point: 0.0,
            };
        }
        let min_val = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mid = (min_val + max_val) * 0.5;
        let half_range = (max_val - min_val) * 0.5;
        let scale = if half_range < f32::EPSILON {
            1.0
        } else {
            half_range / 127.0
        };

        let data: Vec<i8> = values
            .iter()
            .map(|&v| {
                let q = ((v - mid) / scale).round();
                q.clamp(-127.0, 127.0) as i8
            })
            .collect();

        Self {
            data,
            scale,
            zero_point: mid,
        }
    }

    /// Dequantize back to floats.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&q| f32::from(q).mul_add(self.scale, self.zero_point))
            .collect()
    }

    /// Number of elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// INT4 quantized tensor (packed, 2 values per byte).
#[derive(Debug, Clone)]
pub struct QuantizedInt4 {
    pub data: Vec<u8>,
    pub scale: f32,
    pub zero_point: f32,
    pub len: usize,
}

impl QuantizedInt4 {
    /// Quantize a float slice to INT4 (range 0..15 packed).
    #[must_use]
    pub fn quantize(values: &[f32]) -> Self {
        if values.is_empty() {
            return Self {
                data: Vec::new(),
                scale: 1.0,
                zero_point: 0.0,
                len: 0,
            };
        }
        let min_val = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range < f32::EPSILON {
            1.0
        } else {
            range / 15.0
        };
        let zero_point = min_val;

        let quantized: Vec<u8> = values
            .iter()
            .map(|&v| {
                let q = ((v - zero_point) / scale).round();
                q.clamp(0.0, 15.0) as u8
            })
            .collect();

        let packed_len = quantized.len().div_ceil(2);
        let mut packed = vec![0u8; packed_len];
        for (i, &q) in quantized.iter().enumerate() {
            if i % 2 == 0 {
                packed[i / 2] |= q & 0x0F;
            } else {
                packed[i / 2] |= (q & 0x0F) << 4;
            }
        }

        Self {
            data: packed,
            scale,
            zero_point,
            len: values.len(),
        }
    }

    /// Dequantize back to floats.
    #[must_use]
    pub fn dequantize(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.len);
        for i in 0..self.len {
            let byte = self.data[i / 2];
            let nibble = if i % 2 == 0 {
                byte & 0x0F
            } else {
                (byte >> 4) & 0x0F
            };
            result.push(f32::from(nibble) * self.scale + self.zero_point);
        }
        result
    }

    /// Number of elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the tensor is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}
