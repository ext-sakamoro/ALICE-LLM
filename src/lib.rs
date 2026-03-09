#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]

//! ALICE-LLM: Pure Rust LLM inference engine.
//!
//! Provides tokenizer (BPE), KV cache, scaled dot-product attention,
//! rotary position embeddings (`RoPE`), quantization (INT8/INT4),
//! sampling (top-k, top-p, temperature), and batch inference.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// BPE Tokenizer
// ---------------------------------------------------------------------------

/// A byte-pair encoding merge rule.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MergeRule {
    pub left: String,
    pub right: String,
    pub merged: String,
}

/// BPE tokenizer that encodes text into token IDs and decodes back.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    merges: Vec<MergeRule>,
    vocab: HashMap<String, u32>,
    inverse_vocab: HashMap<u32, String>,
}

impl BpeTokenizer {
    /// Create a new tokenizer from merge rules and a vocabulary map.
    #[must_use]
    pub fn new(merges: Vec<MergeRule>, vocab: HashMap<String, u32>) -> Self {
        let inverse_vocab: HashMap<u32, String> =
            vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        Self {
            merges,
            vocab,
            inverse_vocab,
        }
    }

    /// Build a simple character-level tokenizer with optional merges.
    #[must_use]
    pub fn from_chars(text: &str, merges: Vec<MergeRule>) -> Self {
        let mut vocab = HashMap::new();
        let mut id = 0u32;
        for ch in text.chars() {
            let s = ch.to_string();
            if let std::collections::hash_map::Entry::Vacant(e) = vocab.entry(s) {
                e.insert(id);
                id += 1;
            }
        }
        for rule in &merges {
            if !vocab.contains_key(&rule.merged) {
                vocab.insert(rule.merged.clone(), id);
                id += 1;
            }
        }
        Self::new(merges, vocab)
    }

    /// Encode text into a sequence of token IDs.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        for rule in &self.merges {
            let mut i = 0;
            while i + 1 < tokens.len() {
                if tokens[i] == rule.left && tokens[i + 1] == rule.right {
                    tokens[i].clone_from(&rule.merged);
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        tokens
            .iter()
            .map(|t| self.vocab.get(t).copied().unwrap_or(0))
            .collect()
    }

    /// Decode a sequence of token IDs back to text.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|id| self.inverse_vocab.get(id))
            .cloned()
            .collect()
    }

    /// Vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ---------------------------------------------------------------------------
// KV Cache
// ---------------------------------------------------------------------------

/// Key-value cache for transformer attention layers.
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Cached keys: `[layer][seq_pos][head_dim]`
    pub keys: Vec<Vec<Vec<f32>>>,
    /// Cached values: `[layer][seq_pos][head_dim]`
    pub values: Vec<Vec<Vec<f32>>>,
    num_layers: usize,
    head_dim: usize,
    max_seq_len: usize,
}

impl KvCache {
    /// Create a new empty KV cache.
    #[must_use]
    pub fn new(num_layers: usize, head_dim: usize, max_seq_len: usize) -> Self {
        Self {
            keys: vec![Vec::new(); num_layers],
            values: vec![Vec::new(); num_layers],
            num_layers,
            head_dim,
            max_seq_len,
        }
    }

    /// Append a key-value pair for a given layer.
    /// Returns `false` if the cache is full.
    pub fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) -> bool {
        if layer >= self.num_layers {
            return false;
        }
        if self.keys[layer].len() >= self.max_seq_len {
            return false;
        }
        self.keys[layer].push(key);
        self.values[layer].push(value);
        true
    }

    /// Current sequence length for a given layer.
    #[must_use]
    pub fn seq_len(&self, layer: usize) -> usize {
        if layer >= self.num_layers {
            return 0;
        }
        self.keys[layer].len()
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        for layer in 0..self.num_layers {
            self.keys[layer].clear();
            self.values[layer].clear();
        }
    }

    /// Head dimension.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Maximum sequence length.
    #[must_use]
    pub const fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Number of layers.
    #[must_use]
    pub const fn num_layers(&self) -> usize {
        self.num_layers
    }
}

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/// Apply temperature scaling to logits (in-place).
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature < f32::EPSILON {
        return;
    }
    let inv_t = 1.0 / temperature;
    for l in logits.iter_mut() {
        *l *= inv_t;
    }
}

/// Top-k filtering: keep only the top k logits, set the rest to -inf.
pub fn top_k_filter(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let threshold = indexed[k - 1].1;
    for l in logits.iter_mut() {
        if *l < threshold {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Top-p (nucleus) filtering: keep smallest set of tokens whose
/// cumulative probability exceeds p.
pub fn top_p_filter(logits: &mut [f32], p: f32) {
    if p >= 1.0 {
        return;
    }

    let mut probs = logits.to_vec();
    softmax_inplace(&mut probs);

    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0f32;
    let mut cutoff_idx = indexed.len();
    for (rank, &(_, prob)) in indexed.iter().enumerate() {
        cumsum += prob;
        if cumsum > p {
            cutoff_idx = rank + 1;
            break;
        }
    }

    let keep: std::collections::HashSet<usize> =
        indexed[..cutoff_idx].iter().map(|&(i, _)| i).collect();

    for (i, l) in logits.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Softmax in-place.
pub fn softmax_inplace(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let max_val = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// Softmax returning a new vector.
#[must_use]
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut v = logits.to_vec();
    softmax_inplace(&mut v);
    v
}

/// Argmax sampling (greedy).
#[must_use]
pub fn sample_argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i)
}

/// Deterministic weighted sampling using a provided random value in `[0, 1)`.
#[must_use]
pub fn sample_with_random(probs: &[f32], rand_val: f32) -> usize {
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if rand_val < cumsum {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Batch Inference
// ---------------------------------------------------------------------------

/// Configuration for a minimal transformer model.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub rope_base: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256,
            hidden_dim: 64,
            num_heads: 4,
            num_layers: 2,
            max_seq_len: 128,
            rope_base: 10000.0,
        }
    }
}

/// A minimal transformer model for inference (weights are random/zero for testing).
#[derive(Debug, Clone)]
pub struct TransformerModel {
    pub config: ModelConfig,
    pub embedding: Vec<Vec<f32>>,
    pub layers: Vec<TransformerLayer>,
    pub output_proj: Vec<Vec<f32>>,
}

/// A single transformer layer.
#[derive(Debug, Clone)]
pub struct TransformerLayer {
    pub q_proj: Vec<Vec<f32>>,
    pub k_proj: Vec<Vec<f32>>,
    pub v_proj: Vec<Vec<f32>>,
    pub o_proj: Vec<Vec<f32>>,
    pub ff_up: Vec<Vec<f32>>,
    pub ff_down: Vec<Vec<f32>>,
}

impl TransformerModel {
    /// Create a model with identity-like initialization for testing.
    #[must_use]
    pub fn new_test(config: ModelConfig) -> Self {
        let d = config.hidden_dim;
        let v = config.vocab_size;

        let embedding = (0..v)
            .map(|i| {
                let mut row = vec![0.0f32; d];
                row[i % d] = 1.0;
                row
            })
            .collect();

        let make_identity = || -> Vec<Vec<f32>> {
            (0..d)
                .map(|i| {
                    let mut row = vec![0.0f32; d];
                    row[i] = 0.1;
                    row
                })
                .collect()
        };

        let layers = (0..config.num_layers)
            .map(|_| TransformerLayer {
                q_proj: make_identity(),
                k_proj: make_identity(),
                v_proj: make_identity(),
                o_proj: make_identity(),
                ff_up: make_identity(),
                ff_down: make_identity(),
            })
            .collect();

        let output_proj = (0..v)
            .map(|i| {
                let mut row = vec![0.0f32; d];
                row[i % d] = 1.0;
                row
            })
            .collect();

        Self {
            config,
            embedding,
            layers,
            output_proj,
        }
    }

    /// Run a forward pass on a single token sequence, returning logits.
    ///
    /// # Panics
    ///
    /// Panics if `token_ids` is empty.
    #[must_use]
    pub fn forward(&self, token_ids: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let d = self.config.hidden_dim;
        let head_dim = d / self.config.num_heads;

        // Embedding lookup
        let mut hidden: Vec<Vec<f32>> = token_ids
            .iter()
            .map(|&id| {
                let idx = id as usize;
                if idx < self.embedding.len() {
                    self.embedding[idx].clone()
                } else {
                    vec![0.0f32; d]
                }
            })
            .collect();

        // Process each layer
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let seq_len = hidden.len();
            let cache_offset = cache.seq_len(layer_idx);

            // Project Q, K, V
            let queries: Vec<Vec<f32>> = hidden.iter().map(|h| matvec(&layer.q_proj, h)).collect();
            let keys: Vec<Vec<f32>> = hidden.iter().map(|h| matvec(&layer.k_proj, h)).collect();
            let values: Vec<Vec<f32>> = hidden.iter().map(|h| matvec(&layer.v_proj, h)).collect();

            // Apply RoPE to Q and K (simplified: first head only)
            let queries_rope: Vec<Vec<f32>> = queries
                .iter()
                .enumerate()
                .map(|(i, q)| {
                    let mut roped = q.clone();
                    let pos = cache_offset + i;
                    let chunk: Vec<f32> = roped[..head_dim].to_vec();
                    let rotated = apply_rope(&chunk, pos, self.config.rope_base);
                    roped[..head_dim].copy_from_slice(&rotated);
                    roped
                })
                .collect();

            let keys_rope: Vec<Vec<f32>> = keys
                .iter()
                .enumerate()
                .map(|(i, k)| {
                    let mut roped = k.clone();
                    let pos = cache_offset + i;
                    let chunk: Vec<f32> = roped[..head_dim].to_vec();
                    let rotated = apply_rope(&chunk, pos, self.config.rope_base);
                    roped[..head_dim].copy_from_slice(&rotated);
                    roped
                })
                .collect();

            // Append to KV cache
            for i in 0..seq_len {
                cache.append(layer_idx, keys_rope[i].clone(), values[i].clone());
            }

            // Attention with full cached K, V
            let all_keys = &cache.keys[layer_idx];
            let all_values = &cache.values[layer_idx];
            let total_len = all_keys.len();

            // Build causal mask for the query positions
            let mut mask = vec![vec![0.0f32; total_len]; seq_len];
            for (i, mask_row) in mask.iter_mut().enumerate() {
                let query_pos = cache_offset + i;
                for (j, val) in mask_row.iter_mut().enumerate() {
                    if j > query_pos {
                        *val = f32::NEG_INFINITY;
                    }
                }
            }

            let attn_out =
                scaled_dot_product_attention(&queries_rope, all_keys, all_values, Some(&mask));

            // Output projection + residual
            for i in 0..seq_len {
                let projected = matvec(&layer.o_proj, &attn_out[i]);
                for j in 0..d {
                    hidden[i][j] += projected[j];
                }
            }

            // Feedforward + residual
            for h in hidden.iter_mut().take(seq_len) {
                let normed = rms_norm(h, 1e-6);
                let up = matvec(&layer.ff_up, &normed);
                let activated: Vec<f32> = up.iter().map(|&x| silu(x)).collect();
                let down = matvec(&layer.ff_down, &activated);
                for (hj, &dj) in h.iter_mut().zip(down.iter()) {
                    *hj += dj;
                }
            }
        }

        // Output projection: take last token
        let last = hidden.last().unwrap();
        matvec(&self.output_proj, last)
    }

    /// Batch inference: run forward pass on multiple sequences.
    #[must_use]
    pub fn forward_batch(&self, batch: &[Vec<u32>]) -> Vec<Vec<f32>> {
        batch
            .iter()
            .map(|seq| {
                let mut cache = KvCache::new(
                    self.config.num_layers,
                    self.config.hidden_dim / self.config.num_heads,
                    self.config.max_seq_len,
                );
                self.forward(seq, &mut cache)
            })
            .collect()
    }

    /// Generate tokens autoregressively.
    #[must_use]
    pub fn generate(
        &self,
        prompt: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> Vec<u32> {
        let mut cache = KvCache::new(
            self.config.num_layers,
            self.config.hidden_dim / self.config.num_heads,
            self.config.max_seq_len,
        );

        let mut tokens = prompt.to_vec();

        // Prefill
        let mut logits = self.forward(&tokens, &mut cache);

        for _ in 0..max_new_tokens {
            apply_temperature(&mut logits, temperature);
            top_k_filter(&mut logits, top_k);
            let next_token = sample_argmax(&logits) as u32;
            tokens.push(next_token);

            // Decode step (single token)
            logits = self.forward(&[next_token], &mut cache);
        }

        tokens
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- BPE Tokenizer tests ----

    #[test]
    fn test_tokenizer_encode_chars() {
        let tok = BpeTokenizer::from_chars("abc", vec![]);
        let ids = tok.encode("abc");
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_tokenizer_decode_roundtrip() {
        let tok = BpeTokenizer::from_chars("hello", vec![]);
        let ids = tok.encode("hello");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_tokenizer_with_merge() {
        let merges = vec![MergeRule {
            left: "h".into(),
            right: "e".into(),
            merged: "he".into(),
        }];
        let tok = BpeTokenizer::from_chars("hello", merges);
        let ids = tok.encode("hello");
        assert_eq!(ids.len(), 4); // "he", "l", "l", "o"
    }

    #[test]
    fn test_tokenizer_empty() {
        let tok = BpeTokenizer::from_chars("a", vec![]);
        let ids = tok.encode("");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_tokenizer_vocab_size() {
        let tok = BpeTokenizer::from_chars("abcdef", vec![]);
        assert_eq!(tok.vocab_size(), 6);
    }

    #[test]
    fn test_tokenizer_unknown_char() {
        let tok = BpeTokenizer::from_chars("ab", vec![]);
        let ids = tok.encode("abc");
        assert_eq!(ids.len(), 3);
        // 'c' is unknown, maps to 0
    }

    #[test]
    fn test_tokenizer_multiple_merges() {
        let merges = vec![
            MergeRule {
                left: "a".into(),
                right: "b".into(),
                merged: "ab".into(),
            },
            MergeRule {
                left: "ab".into(),
                right: "c".into(),
                merged: "abc".into(),
            },
        ];
        let tok = BpeTokenizer::from_chars("abcabc", merges);
        let ids = tok.encode("abcabc");
        assert_eq!(ids.len(), 2); // "abc", "abc"
    }

    #[test]
    fn test_tokenizer_decode_unknown_id() {
        let tok = BpeTokenizer::from_chars("ab", vec![]);
        let decoded = tok.decode(&[999]);
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_tokenizer_single_char() {
        let tok = BpeTokenizer::from_chars("x", vec![]);
        let ids = tok.encode("x");
        assert_eq!(ids.len(), 1);
        assert_eq!(tok.decode(&ids), "x");
    }

    #[test]
    fn test_tokenizer_repeated_chars() {
        let tok = BpeTokenizer::from_chars("aaa", vec![]);
        let ids = tok.encode("aaa");
        assert_eq!(ids.len(), 3);
    }

    // ---- KV Cache tests ----

    #[test]
    fn test_kv_cache_new() {
        let cache = KvCache::new(2, 16, 64);
        assert_eq!(cache.num_layers(), 2);
        assert_eq!(cache.head_dim(), 16);
        assert_eq!(cache.max_seq_len(), 64);
    }

    #[test]
    fn test_kv_cache_append() {
        let mut cache = KvCache::new(1, 4, 10);
        assert!(cache.append(0, vec![1.0; 4], vec![2.0; 4]));
        assert_eq!(cache.seq_len(0), 1);
    }

    #[test]
    fn test_kv_cache_full() {
        let mut cache = KvCache::new(1, 2, 2);
        assert!(cache.append(0, vec![1.0; 2], vec![1.0; 2]));
        assert!(cache.append(0, vec![1.0; 2], vec![1.0; 2]));
        assert!(!cache.append(0, vec![1.0; 2], vec![1.0; 2]));
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = KvCache::new(2, 4, 10);
        cache.append(0, vec![1.0; 4], vec![1.0; 4]);
        cache.append(1, vec![1.0; 4], vec![1.0; 4]);
        cache.clear();
        assert_eq!(cache.seq_len(0), 0);
        assert_eq!(cache.seq_len(1), 0);
    }

    #[test]
    fn test_kv_cache_invalid_layer() {
        let mut cache = KvCache::new(1, 4, 10);
        assert!(!cache.append(5, vec![1.0; 4], vec![1.0; 4]));
        assert_eq!(cache.seq_len(5), 0);
    }

    #[test]
    fn test_kv_cache_multiple_layers() {
        let mut cache = KvCache::new(3, 4, 10);
        cache.append(0, vec![1.0; 4], vec![1.0; 4]);
        cache.append(1, vec![2.0; 4], vec![2.0; 4]);
        cache.append(1, vec![3.0; 4], vec![3.0; 4]);
        assert_eq!(cache.seq_len(0), 1);
        assert_eq!(cache.seq_len(1), 2);
        assert_eq!(cache.seq_len(2), 0);
    }

    // ---- Attention tests ----

    #[test]
    fn test_attention_basic() {
        let q = vec![vec![1.0, 0.0]];
        let k = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let v = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 2);
    }

    #[test]
    fn test_attention_identity() {
        let q = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let k = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let v = vec![vec![0.5, 0.5, 0.0, 0.0]];
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        assert!((out[0][0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_attention_empty() {
        let out = scaled_dot_product_attention(&[], &[], &[], None);
        assert!(out.is_empty());
    }

    #[test]
    fn test_attention_with_causal_mask() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let k = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let v = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mask = causal_mask(2);
        let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask));
        assert_eq!(out.len(), 2);
        // First token can only attend to itself
        assert!((out[0][0] - 1.0).abs() < 1e-5);
        assert!(out[0][1].abs() < 1e-5);
    }

    #[test]
    fn test_causal_mask_shape() {
        let mask = causal_mask(4);
        assert_eq!(mask.len(), 4);
        assert_eq!(mask[0].len(), 4);
        assert_eq!(mask[0][0], 0.0);
        assert!(mask[0][1].is_infinite());
    }

    #[test]
    fn test_attention_preserves_sum() {
        let q = vec![vec![0.5, 0.5]];
        let k = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let v = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let out = scaled_dot_product_attention(&q, &k, &v, None);
        let sum: f32 = out[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    // ---- RoPE tests ----

    #[test]
    fn test_rope_position_zero() {
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let r = apply_rope(&v, 0, 10000.0);
        // At position 0, angles are 0, so cos=1, sin=0
        assert!((r[0] - 1.0).abs() < 1e-5);
        assert!((r[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_rope_preserves_norm() {
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let r = apply_rope(&v, 5, 10000.0);
        let norm_orig: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_roped: f32 = r.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_orig - norm_roped).abs() < 1e-4);
    }

    #[test]
    fn test_rope_different_positions() {
        let v = vec![1.0, 0.0, 1.0, 0.0];
        let r1 = apply_rope(&v, 1, 10000.0);
        let r2 = apply_rope(&v, 2, 10000.0);
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_rope_batch() {
        let vecs = vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0]];
        let result = apply_rope_batch(&vecs, 10000.0);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 4);
    }

    #[test]
    fn test_rope_even_dim() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r = apply_rope(&v, 3, 10000.0);
        assert_eq!(r.len(), 6);
    }

    // ---- Quantization INT8 tests ----

    #[test]
    fn test_int8_quantize_roundtrip() {
        let values = vec![0.0, 0.5, 1.0, -1.0, -0.5];
        let q = QuantizedInt8::quantize(&values);
        let dq = q.dequantize();
        for (orig, deq) in values.iter().zip(dq.iter()) {
            assert!((orig - deq).abs() < 0.05);
        }
    }

    #[test]
    fn test_int8_len() {
        let q = QuantizedInt8::quantize(&[1.0, 2.0, 3.0]);
        assert_eq!(q.len(), 3);
        assert!(!q.is_empty());
    }

    #[test]
    fn test_int8_empty() {
        let q = QuantizedInt8::quantize(&[]);
        assert!(q.is_empty());
    }

    #[test]
    fn test_int8_constant() {
        let q = QuantizedInt8::quantize(&[5.0, 5.0, 5.0]);
        let dq = q.dequantize();
        for v in &dq {
            assert!((v - 5.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_int8_negative_values() {
        let values = vec![-10.0, -5.0, 0.0, 5.0, 10.0];
        let q = QuantizedInt8::quantize(&values);
        let dq = q.dequantize();
        for (orig, deq) in values.iter().zip(dq.iter()) {
            assert!((orig - deq).abs() < 0.2);
        }
    }

    // ---- Quantization INT4 tests ----

    #[test]
    fn test_int4_quantize_roundtrip() {
        let values = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let q = QuantizedInt4::quantize(&values);
        let dq = q.dequantize();
        for (orig, deq) in values.iter().zip(dq.iter()) {
            assert!((orig - deq).abs() < 0.1);
        }
    }

    #[test]
    fn test_int4_len() {
        let q = QuantizedInt4::quantize(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(q.len(), 5);
        assert!(!q.is_empty());
    }

    #[test]
    fn test_int4_empty() {
        let q = QuantizedInt4::quantize(&[]);
        assert!(q.is_empty());
    }

    #[test]
    fn test_int4_packing() {
        let q = QuantizedInt4::quantize(&[0.0, 1.0, 2.0]);
        // 3 values packed into 2 bytes
        assert_eq!(q.data.len(), 2);
    }

    #[test]
    fn test_int4_odd_count() {
        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let q = QuantizedInt4::quantize(&values);
        let dq = q.dequantize();
        assert_eq!(dq.len(), 7);
    }

    #[test]
    fn test_int4_single_value() {
        let q = QuantizedInt4::quantize(&[42.0]);
        let dq = q.dequantize();
        assert!((dq[0] - 42.0).abs() < 0.1);
    }

    // ---- Sampling tests ----

    #[test]
    fn test_temperature_scaling() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_temperature(&mut logits, 2.0);
        assert!((logits[0] - 0.5).abs() < 1e-6);
        assert!((logits[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_temperature_zero() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_temperature(&mut logits, 0.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_k_filter() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        top_k_filter(&mut logits, 2);
        let finite_count = logits.iter().filter(|x| x.is_finite()).count();
        assert!(finite_count <= 3); // top-2 plus ties
    }

    #[test]
    fn test_top_k_zero() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        top_k_filter(&mut logits, 0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_k_larger_than_len() {
        let mut logits = vec![1.0, 2.0];
        let original = logits.clone();
        top_k_filter(&mut logits, 10);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_p_filter() {
        let mut logits = vec![1.0, 2.0, 10.0, 0.5];
        top_p_filter(&mut logits, 0.9);
        // The highest logit should remain
        assert!(logits[2].is_finite());
    }

    #[test]
    fn test_top_p_one() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        top_p_filter(&mut logits, 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_softmax_basic() {
        let probs = softmax(&[0.0, 0.0, 0.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!((probs[0] - probs[1]).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_large_values() {
        let probs = softmax(&[1000.0, 1000.0, 0.0]);
        assert!((probs[0] - 0.5).abs() < 1e-3);
        assert!((probs[1] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_softmax_empty() {
        let probs = softmax(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_softmax_single() {
        let probs = softmax(&[5.0]);
        assert!((probs[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sample_argmax() {
        assert_eq!(sample_argmax(&[1.0, 5.0, 3.0]), 1);
        assert_eq!(sample_argmax(&[10.0, 2.0, 3.0]), 0);
    }

    #[test]
    fn test_sample_argmax_single() {
        assert_eq!(sample_argmax(&[42.0]), 0);
    }

    #[test]
    fn test_sample_with_random() {
        let probs = vec![0.1, 0.2, 0.7];
        assert_eq!(sample_with_random(&probs, 0.05), 0);
        assert_eq!(sample_with_random(&probs, 0.25), 1);
        assert_eq!(sample_with_random(&probs, 0.95), 2);
    }

    #[test]
    fn test_sample_with_random_boundary() {
        let probs = vec![0.5, 0.5];
        assert_eq!(sample_with_random(&probs, 0.0), 0);
        assert_eq!(sample_with_random(&probs, 0.49), 0);
        assert_eq!(sample_with_random(&probs, 0.51), 1);
    }

    // ---- Linear algebra tests ----

    #[test]
    fn test_dot_product() {
        assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot_product_zero() {
        assert!((dot(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-5);
    }

    #[test]
    fn test_matvec() {
        let mat = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let v = vec![3.0, 4.0];
        let out = matvec(&mat, &v);
        assert!((out[0] - 3.0).abs() < 1e-5);
        assert!((out[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![vec![3.0, 4.0], vec![5.0, 6.0]];
        let c = matmul(&a, &b);
        assert!((c[0][0] - 3.0).abs() < 1e-5);
        assert!((c[1][1] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_empty() {
        let out = matmul(&[], &[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_transpose() {
        let m = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let t = transpose(&m);
        assert_eq!(t.len(), 3);
        assert_eq!(t[0].len(), 2);
        assert!((t[0][0] - 1.0).abs() < 1e-5);
        assert!((t[0][1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_transpose_empty() {
        let t = transpose(&[]);
        assert!(t.is_empty());
    }

    #[test]
    fn test_rms_norm() {
        let v = vec![1.0, 1.0, 1.0, 1.0];
        let n = rms_norm(&v, 1e-6);
        // RMS of [1,1,1,1] is 1, so output ~ [1,1,1,1]
        for x in &n {
            assert!((x - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_rms_norm_scaling() {
        let v = vec![2.0, 2.0];
        let n = rms_norm(&v, 1e-6);
        // RMS = 2, so output ~ [1, 1]
        for x in &n {
            assert!((x - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_layer_norm() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let n = layer_norm(&v, 1e-6);
        let mean: f32 = n.iter().sum::<f32>() / n.len() as f32;
        assert!(mean.abs() < 1e-4);
    }

    #[test]
    fn test_layer_norm_constant() {
        let v = vec![5.0, 5.0, 5.0];
        let n = layer_norm(&v, 1e-6);
        for x in &n {
            assert!(x.abs() < 1e-3);
        }
    }

    #[test]
    fn test_silu() {
        assert!((silu(0.0)).abs() < 1e-5);
        assert!(silu(1.0) > 0.5);
        assert!(silu(-10.0).abs() < 0.01);
    }

    #[test]
    fn test_gelu() {
        assert!((gelu(0.0)).abs() < 1e-5);
        assert!(gelu(1.0) > 0.5);
        assert!(gelu(-3.0).abs() < 0.01);
    }

    #[test]
    fn test_silu_monotonic() {
        let a = silu(1.0);
        let b = silu(2.0);
        let c = silu(3.0);
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn test_gelu_monotonic() {
        let a = gelu(1.0);
        let b = gelu(2.0);
        let c = gelu(3.0);
        assert!(a < b);
        assert!(b < c);
    }

    // ---- Model tests ----

    #[test]
    fn test_model_creation() {
        let config = ModelConfig::default();
        let model = TransformerModel::new_test(config);
        assert_eq!(model.config.vocab_size, 256);
        assert_eq!(model.config.hidden_dim, 64);
    }

    #[test]
    fn test_model_forward() {
        let config = ModelConfig {
            vocab_size: 32,
            hidden_dim: 16,
            num_heads: 2,
            num_layers: 1,
            max_seq_len: 32,
            rope_base: 10000.0,
        };
        let model = TransformerModel::new_test(config);
        let mut cache = KvCache::new(1, 8, 32);
        let logits = model.forward(&[0, 1, 2], &mut cache);
        assert_eq!(logits.len(), 32);
    }

    #[test]
    fn test_model_forward_single_token() {
        let config = ModelConfig {
            vocab_size: 16,
            hidden_dim: 8,
            num_heads: 2,
            num_layers: 1,
            max_seq_len: 16,
            rope_base: 10000.0,
        };
        let model = TransformerModel::new_test(config);
        let mut cache = KvCache::new(1, 4, 16);
        let logits = model.forward(&[0], &mut cache);
        assert_eq!(logits.len(), 16);
    }

    #[test]
    fn test_model_forward_cache_grows() {
        let config = ModelConfig {
            vocab_size: 16,
            hidden_dim: 8,
            num_heads: 2,
            num_layers: 1,
            max_seq_len: 32,
            rope_base: 10000.0,
        };
        let model = TransformerModel::new_test(config);
        let mut cache = KvCache::new(1, 4, 32);
        let _ = model.forward(&[0, 1], &mut cache);
        assert_eq!(cache.seq_len(0), 2);
        let _ = model.forward(&[2], &mut cache);
        assert_eq!(cache.seq_len(0), 3);
    }

    #[test]
    fn test_batch_inference() {
        let config = ModelConfig {
            vocab_size: 16,
            hidden_dim: 8,
            num_heads: 2,
            num_layers: 1,
            max_seq_len: 16,
            rope_base: 10000.0,
        };
        let model = TransformerModel::new_test(config);
        let batch = vec![vec![0, 1, 2], vec![3, 4], vec![5]];
        let results = model.forward_batch(&batch);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert_eq!(r.len(), 16);
        }
    }

    #[test]
    fn test_generate() {
        let config = ModelConfig {
            vocab_size: 16,
            hidden_dim: 8,
            num_heads: 2,
            num_layers: 1,
            max_seq_len: 32,
            rope_base: 10000.0,
        };
        let model = TransformerModel::new_test(config);
        let output = model.generate(&[0, 1], 3, 1.0, 5);
        assert_eq!(output.len(), 5); // 2 prompt + 3 generated
    }

    #[test]
    fn test_generate_deterministic() {
        let config = ModelConfig {
            vocab_size: 16,
            hidden_dim: 8,
            num_heads: 2,
            num_layers: 1,
            max_seq_len: 32,
            rope_base: 10000.0,
        };
        let model = TransformerModel::new_test(config.clone());
        let out1 = model.generate(&[0], 5, 1.0, 1);
        let model2 = TransformerModel::new_test(config);
        let out2 = model2.generate(&[0], 5, 1.0, 1);
        assert_eq!(out1, out2);
    }

    #[test]
    fn test_model_multi_layer() {
        let config = ModelConfig {
            vocab_size: 16,
            hidden_dim: 8,
            num_heads: 2,
            num_layers: 3,
            max_seq_len: 16,
            rope_base: 10000.0,
        };
        let model = TransformerModel::new_test(config);
        let mut cache = KvCache::new(3, 4, 16);
        let logits = model.forward(&[0, 1, 2], &mut cache);
        assert_eq!(logits.len(), 16);
        assert_eq!(cache.seq_len(0), 3);
        assert_eq!(cache.seq_len(1), 3);
        assert_eq!(cache.seq_len(2), 3);
    }

    #[test]
    fn test_model_oov_token() {
        let config = ModelConfig {
            vocab_size: 8,
            hidden_dim: 8,
            num_heads: 2,
            num_layers: 1,
            max_seq_len: 16,
            rope_base: 10000.0,
        };
        let model = TransformerModel::new_test(config);
        let mut cache = KvCache::new(1, 4, 16);
        // Token ID 999 is out of vocab
        let logits = model.forward(&[999], &mut cache);
        assert_eq!(logits.len(), 8);
    }

    // ---- Integration tests ----

    #[test]
    fn test_tokenize_and_infer() {
        let tok = BpeTokenizer::from_chars("abcd", vec![]);
        let ids = tok.encode("abcd");
        let config = ModelConfig {
            vocab_size: 8,
            hidden_dim: 8,
            num_heads: 2,
            num_layers: 1,
            max_seq_len: 16,
            rope_base: 10000.0,
        };
        let model = TransformerModel::new_test(config);
        let mut cache = KvCache::new(1, 4, 16);
        let logits = model.forward(&ids, &mut cache);
        assert_eq!(logits.len(), 8);
        let next_token = sample_argmax(&logits);
        assert!(next_token < 8);
    }

    #[test]
    fn test_quantize_and_use() {
        let weights = vec![0.1, -0.5, 0.3, 0.8, -0.2];
        let q8 = QuantizedInt8::quantize(&weights);
        let dq8 = q8.dequantize();
        let q4 = QuantizedInt4::quantize(&weights);
        let dq4 = q4.dequantize();
        // Both should roughly match
        for i in 0..weights.len() {
            assert!((dq8[i] - weights[i]).abs() < 0.1);
            assert!((dq4[i] - weights[i]).abs() < 0.2);
        }
    }

    #[test]
    fn test_full_pipeline() {
        let merges = vec![MergeRule {
            left: "a".into(),
            right: "b".into(),
            merged: "ab".into(),
        }];
        let tok = BpeTokenizer::from_chars("abcde", merges);
        let ids = tok.encode("abcde");

        let config = ModelConfig {
            vocab_size: 8,
            hidden_dim: 8,
            num_heads: 2,
            num_layers: 1,
            max_seq_len: 16,
            rope_base: 10000.0,
        };
        let model = TransformerModel::new_test(config);
        let results = model.forward_batch(&[ids.clone(), ids]);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_softmax_then_sample() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        let idx = sample_with_random(&probs, 0.5);
        assert!(idx < probs.len());
    }

    #[test]
    fn test_temperature_affects_distribution() {
        let mut logits_hot = vec![1.0, 2.0, 10.0];
        let mut logits_cold = logits_hot.clone();
        apply_temperature(&mut logits_hot, 100.0);
        apply_temperature(&mut logits_cold, 0.01);
        let probs_hot = softmax(&logits_hot);
        let probs_cold = softmax(&logits_cold);
        // Hot temperature -> more uniform
        assert!((probs_hot[0] - probs_hot[2]).abs() < 0.1);
        // Cold temperature -> more peaked
        assert!(probs_cold[2] > 0.99);
    }

    #[test]
    fn test_rope_orthogonality() {
        // Two orthogonal vectors should remain orthogonal after RoPE
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 1.0, 0.0];
        let ra = apply_rope(&a, 3, 10000.0);
        let rb = apply_rope(&b, 3, 10000.0);
        let d = dot(&ra, &rb);
        assert!(d.abs() < 1e-4);
    }

    #[test]
    fn test_matmul_associative() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let c = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let ab_c = matmul(&matmul(&a, &b), &c);
        let a_bc = matmul(&a, &matmul(&b, &c));
        for i in 0..2 {
            for j in 0..2 {
                assert!((ab_c[i][j] - a_bc[i][j]).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_transpose_involution() {
        let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let tt = transpose(&transpose(&m));
        assert_eq!(m, tt);
    }

    #[test]
    fn test_kv_cache_with_model() {
        let config = ModelConfig {
            vocab_size: 8,
            hidden_dim: 8,
            num_heads: 2,
            num_layers: 2,
            max_seq_len: 32,
            rope_base: 10000.0,
        };
        let model = TransformerModel::new_test(config);
        let mut cache = KvCache::new(2, 4, 32);
        let l1 = model.forward(&[0, 1], &mut cache);
        let l2 = model.forward(&[2], &mut cache);
        assert_eq!(l1.len(), 8);
        assert_eq!(l2.len(), 8);
        assert_eq!(cache.seq_len(0), 3);
        assert_eq!(cache.seq_len(1), 3);
    }

    #[test]
    fn test_int8_large_range() {
        let values: Vec<f32> = (-100..=100).map(|x| x as f32).collect();
        let q = QuantizedInt8::quantize(&values);
        let dq = q.dequantize();
        assert_eq!(dq.len(), 201);
        for (orig, deq) in values.iter().zip(dq.iter()) {
            assert!((orig - deq).abs() < 2.0);
        }
    }

    #[test]
    fn test_int4_large_range() {
        let values: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let q = QuantizedInt4::quantize(&values);
        let dq = q.dequantize();
        assert_eq!(dq.len(), 16);
    }

    #[test]
    fn test_default_model_config() {
        let config = ModelConfig::default();
        assert_eq!(config.vocab_size, 256);
        assert_eq!(config.hidden_dim, 64);
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.max_seq_len, 128);
        assert!((config.rope_base - 10000.0).abs() < 1e-5);
    }

    // ---- Additional tests to reach 100+ ----

    #[test]
    fn test_softmax_inplace_uniform() {
        let mut v = vec![0.0; 4];
        softmax_inplace(&mut v);
        for x in &v {
            assert!((x - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_inplace_peaked() {
        let mut v = vec![0.0, 0.0, 100.0];
        softmax_inplace(&mut v);
        assert!(v[2] > 0.99);
    }

    #[test]
    fn test_dot_empty() {
        assert!((dot(&[], &[])).abs() < 1e-9);
    }

    #[test]
    fn test_matvec_single_row() {
        let mat = vec![vec![2.0, 3.0]];
        let result = matvec(&mat, &[4.0, 5.0]);
        assert!((result[0] - 23.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_non_square() {
        let a = vec![vec![1.0, 2.0, 3.0]];
        let b = vec![vec![4.0], vec![5.0], vec![6.0]];
        let c = matmul(&a, &b);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].len(), 1);
        assert!((c[0][0] - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_rope_large_position() {
        let v = vec![1.0, 0.5, 0.5, 1.0];
        let r = apply_rope(&v, 1000, 10000.0);
        let norm: f32 = r.iter().map(|x| x * x).sum::<f32>().sqrt();
        let orig_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - orig_norm).abs() < 1e-3);
    }

    #[test]
    fn test_rope_batch_empty() {
        let result = apply_rope_batch(&[], 10000.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_int8_single_value() {
        let q = QuantizedInt8::quantize(&[3.14]);
        let dq = q.dequantize();
        assert!((dq[0] - 3.14).abs() < 0.1);
    }

    #[test]
    fn test_int4_constant() {
        let q = QuantizedInt4::quantize(&[7.0, 7.0, 7.0]);
        let dq = q.dequantize();
        for v in &dq {
            assert!((v - 7.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_top_k_keeps_highest() {
        let mut logits = vec![1.0, 10.0, 2.0, 3.0, 4.0];
        top_k_filter(&mut logits, 1);
        assert!(logits[1].is_finite());
        assert_eq!(sample_argmax(&logits), 1);
    }

    #[test]
    fn test_generate_prompt_preserved() {
        let config = ModelConfig {
            vocab_size: 16,
            hidden_dim: 8,
            num_heads: 2,
            num_layers: 1,
            max_seq_len: 32,
            rope_base: 10000.0,
        };
        let model = TransformerModel::new_test(config);
        let prompt = vec![3, 7, 11];
        let output = model.generate(&prompt, 2, 1.0, 5);
        assert_eq!(&output[..3], &prompt);
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_layer_norm_two_elements() {
        let v = vec![0.0, 2.0];
        let n = layer_norm(&v, 1e-6);
        assert!((n[0] + n[1]).abs() < 1e-4);
    }

    #[test]
    fn test_causal_mask_single() {
        let mask = causal_mask(1);
        assert_eq!(mask.len(), 1);
        assert_eq!(mask[0][0], 0.0);
    }

    #[test]
    fn test_kv_cache_seq_len_after_clear() {
        let mut cache = KvCache::new(2, 8, 16);
        cache.append(0, vec![0.0; 8], vec![0.0; 8]);
        cache.append(0, vec![0.0; 8], vec![0.0; 8]);
        assert_eq!(cache.seq_len(0), 2);
        cache.clear();
        assert_eq!(cache.seq_len(0), 0);
        assert!(cache.append(0, vec![0.0; 8], vec![0.0; 8]));
        assert_eq!(cache.seq_len(0), 1);
    }
}
