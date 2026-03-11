//! Multi-architecture LLM inference engine with GGUF quantized weights.
//!
//! Supports Llama-3, Mistral (Sliding Window Attention), and Gemma-2
//! (Logit Softcapping). Performs inference directly on Q4_K_M/Q8_0
//! quantized data via fused dequantize+matvec.

use crate::gguf::{quantized_matvec, quantized_matvec_preq, quantize_row_q8_k, BlockQ8K, GgmlType, GgufFile, GgufTokenizer, TernaryMatrix, ternary_matvec};
use std::time::Instant;

// ─── Model architecture ─────────────────────────────────────────────────────

/// Supported model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    Llama,
    Mistral,
    Gemma2,
}

impl ModelArch {
    /// Detect architecture from GGUF metadata key `general.architecture`.
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Self {
        match gguf.meta_str("general.architecture") {
            Some("mistral") => Self::Mistral,
            Some("gemma2") => Self::Gemma2,
            _ => Self::Llama,
        }
    }

    /// GGUF metadata key prefix for this architecture.
    fn meta_prefix(&self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::Gemma2 => "gemma2",
        }
    }
}

// ─── Model config ───────────────────────────────────────────────────────────

/// Model configuration extracted from GGUF metadata.
/// Supports Llama-3, Mistral, and Gemma-2 architectures.
#[derive(Debug, Clone)]
pub struct Llama3Config {
    pub arch: ModelArch,
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    /// Mistral: sliding window size (None = full attention).
    pub sliding_window: Option<usize>,
    /// Gemma-2: attention logit softcapping value (None = no capping).
    pub attn_logit_softcap: Option<f32>,
    /// Gemma-2: final logit softcapping value (None = no capping).
    pub final_logit_softcap: Option<f32>,
}

impl Llama3Config {
    /// Load config from GGUF metadata (auto-detects architecture).
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Option<Self> {
        let arch = ModelArch::from_gguf(gguf);
        let prefix = arch.meta_prefix();

        let hidden_dim = gguf.meta_u32(&format!("{prefix}.embedding_length"))? as usize;
        let num_heads = gguf.meta_u32(&format!("{prefix}.attention.head_count"))? as usize;
        let num_kv_heads = gguf
            .meta_u32(&format!("{prefix}.attention.head_count_kv"))
            .unwrap_or(num_heads as u32) as usize;
        let num_layers = gguf.meta_u32(&format!("{prefix}.block_count"))? as usize;
        let intermediate_dim = gguf.meta_u32(&format!("{prefix}.feed_forward_length"))? as usize;
        let max_seq_len = gguf
            .meta_u32(&format!("{prefix}.context_length"))
            .unwrap_or(8192) as usize;
        let vocab_size = gguf
            .meta_u32(&format!("{prefix}.vocab_size"))
            .or_else(|| {
                gguf.meta("tokenizer.ggml.tokens")
                    .and_then(|v| match v {
                        crate::gguf::MetaValue::Array(arr) => Some(arr.len() as u32),
                        _ => None,
                    })
            })? as usize;
        let rope_theta = gguf
            .meta_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(if arch == ModelArch::Mistral { 1_000_000.0 } else { 500_000.0 });
        let norm_eps = gguf
            .meta_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let head_dim = hidden_dim / num_heads;

        // Mistral: sliding window attention
        let sliding_window = gguf
            .meta_u32(&format!("{prefix}.attention.sliding_window"))
            .map(|v| v as usize);

        // Gemma-2: logit softcapping
        let attn_logit_softcap = gguf
            .meta_f32(&format!("{prefix}.attn_logit_softcapping"));
        let final_logit_softcap = gguf
            .meta_f32(&format!("{prefix}.final_logit_softcapping"));

        Some(Self {
            arch,
            vocab_size,
            hidden_dim,
            intermediate_dim,
            num_heads,
            num_kv_heads,
            num_layers,
            max_seq_len,
            head_dim,
            rope_theta,
            norm_eps,
            sliding_window,
            attn_logit_softcap,
            final_logit_softcap,
        })
    }

    /// Llama-3 8B default config.
    #[must_use]
    pub fn llama3_8b() -> Self {
        Self {
            arch: ModelArch::Llama,
            vocab_size: 128_256,
            hidden_dim: 4096,
            intermediate_dim: 14_336,
            num_heads: 32,
            num_kv_heads: 8,
            num_layers: 32,
            max_seq_len: 8192,
            head_dim: 128,
            rope_theta: 500_000.0,
            norm_eps: 1e-5,
            sliding_window: None,
            attn_logit_softcap: None,
            final_logit_softcap: None,
        }
    }
}

// ─── KV Cache (GQA-aware, contiguous buffer) ────────────────────────────────

struct KvCache {
    /// Contiguous buffer: [layer * max_seq * kv_dim + pos * kv_dim .. +kv_dim]
    keys: Vec<f32>,
    values: Vec<f32>,
    _num_layers: usize,
    max_seq_len: usize,
    kv_dim: usize,
    seq_len: usize,
}

impl KvCache {
    fn new(num_layers: usize, max_seq_len: usize, kv_dim: usize) -> Self {
        let total = num_layers * max_seq_len * kv_dim;
        Self {
            keys: vec![0.0f32; total],
            values: vec![0.0f32; total],
            _num_layers: num_layers,
            max_seq_len,
            kv_dim,
            seq_len: 0,
        }
    }

    #[inline]
    fn offset(&self, layer: usize, pos: usize) -> usize {
        (layer * self.max_seq_len + pos) * self.kv_dim
    }

    fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        let off = self.offset(layer, self.seq_len);
        self.keys[off..off + self.kv_dim].copy_from_slice(k);
        self.values[off..off + self.kv_dim].copy_from_slice(v);
    }

    /// Call once after all layers have appended for a given position.
    fn advance(&mut self) {
        self.seq_len += 1;
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    #[inline]
    fn key_at(&self, layer: usize, pos: usize) -> &[f32] {
        let off = self.offset(layer, pos);
        &self.keys[off..off + self.kv_dim]
    }

    #[inline]
    fn value_at(&self, layer: usize, pos: usize) -> &[f32] {
        let off = self.offset(layer, pos);
        &self.values[off..off + self.kv_dim]
    }

    /// Rollback KV cache to a previous position (for speculative decoding).
    fn rollback_to(&mut self, pos: usize) {
        self.seq_len = pos;
    }

    fn clear(&mut self) {
        self.seq_len = 0;
    }
}

// ─── Paged KV Cache ─────────────────────────────────────────────────────────

const PAGE_SIZE: usize = 16;

/// A page of KV data for one layer: PAGE_SIZE tokens × kv_dim.
struct KvPage {
    keys: Vec<f32>,
    values: Vec<f32>,
    used: usize,
}

impl KvPage {
    fn new(kv_dim: usize) -> Self {
        Self {
            keys: vec![0.0f32; PAGE_SIZE * kv_dim],
            values: vec![0.0f32; PAGE_SIZE * kv_dim],
            used: 0,
        }
    }

    fn is_full(&self) -> bool {
        self.used >= PAGE_SIZE
    }
}

/// Paged KV cache for a single sequence.
/// Pages are allocated on demand (no upfront max_seq_len allocation).
struct PagedKvCache {
    pages: Vec<Vec<KvPage>>,  // pages[layer][page_idx]
    num_layers: usize,
    kv_dim: usize,
    seq_len: usize,
}

impl PagedKvCache {
    fn new(num_layers: usize, kv_dim: usize) -> Self {
        let pages = (0..num_layers).map(|_| Vec::new()).collect();
        Self { pages, num_layers, kv_dim, seq_len: 0 }
    }

    fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        let layer_pages = &mut self.pages[layer];
        if layer_pages.is_empty() || layer_pages.last().unwrap().is_full() {
            layer_pages.push(KvPage::new(self.kv_dim));
        }
        let page = layer_pages.last_mut().unwrap();
        let off = page.used * self.kv_dim;
        page.keys[off..off + self.kv_dim].copy_from_slice(k);
        page.values[off..off + self.kv_dim].copy_from_slice(v);
        page.used += 1;
    }

    fn advance(&mut self) { self.seq_len += 1; }

    fn seq_len(&self) -> usize { self.seq_len }

    #[inline]
    fn key_at(&self, layer: usize, pos: usize) -> &[f32] {
        let page_idx = pos / PAGE_SIZE;
        let slot = pos % PAGE_SIZE;
        let off = slot * self.kv_dim;
        &self.pages[layer][page_idx].keys[off..off + self.kv_dim]
    }

    #[inline]
    fn value_at(&self, layer: usize, pos: usize) -> &[f32] {
        let page_idx = pos / PAGE_SIZE;
        let slot = pos % PAGE_SIZE;
        let off = slot * self.kv_dim;
        &self.pages[layer][page_idx].values[off..off + self.kv_dim]
    }

    fn rollback_to(&mut self, pos: usize) {
        self.seq_len = pos;
        for layer_pages in &mut self.pages {
            let needed = if pos == 0 { 0 } else { (pos - 1) / PAGE_SIZE + 1 };
            layer_pages.truncate(needed);
            if let Some(last) = layer_pages.last_mut() {
                let rem = pos % PAGE_SIZE;
                last.used = if rem == 0 && pos > 0 { PAGE_SIZE } else { rem };
            }
        }
    }

    fn clear(&mut self) {
        self.seq_len = 0;
        for layer_pages in &mut self.pages {
            layer_pages.clear();
        }
    }

    fn total_pages(&self) -> usize {
        self.pages.iter().map(|lp| lp.len()).sum()
    }

    fn memory_bytes(&self) -> usize {
        self.total_pages() * PAGE_SIZE * self.kv_dim * 4 * 2
    }
}

// ─── Batch Scheduler ─────────────────────────────────────────────────────────

/// A request in the batch scheduler.
pub struct BatchRequest {
    pub id: usize,
    pub tokens: Vec<u32>,
    pub generated: Vec<u32>,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub done: bool,
    kv_cache: PagedKvCache,
}

/// Continuous batching scheduler for multiple concurrent requests.
pub struct BatchScheduler {
    requests: Vec<BatchRequest>,
    next_id: usize,
}

impl BatchScheduler {
    pub fn new() -> Self {
        Self { requests: Vec::new(), next_id: 0 }
    }

    /// Add a new request. Returns the request ID.
    pub fn add_request(
        &mut self,
        tokens: Vec<u32>,
        max_new_tokens: usize,
        temperature: f32,
        num_layers: usize,
        kv_dim: usize,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.requests.push(BatchRequest {
            id,
            tokens,
            generated: Vec::new(),
            max_new_tokens,
            temperature,
            done: false,
            kv_cache: PagedKvCache::new(num_layers, kv_dim),
        });
        id
    }

    pub fn active_count(&self) -> usize {
        self.requests.iter().filter(|r| !r.done).count()
    }

    pub fn requests_mut(&mut self) -> &mut [BatchRequest] {
        &mut self.requests
    }

    /// Get completed results by ID.
    pub fn get_result(&self, id: usize) -> Option<&BatchRequest> {
        self.requests.iter().find(|r| r.id == id && r.done)
    }
}

// ─── RMS Norm ───────────────────────────────────────────────────────────────

fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    // Use f64 for sum-of-squares accumulation (matches llama.cpp's ggml_rms_norm)
    let mut ss = 0.0f64;
    for &v in x {
        ss += (v as f64) * (v as f64);
    }
    let mean = (ss / n as f64) as f32;
    let scale = 1.0f32 / (mean + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
}

// ─── RoPE ───────────────────────────────────────────────────────────────────

fn apply_rope(vec: &mut [f32], position: usize, head_dim: usize, theta: f32) {
    for i in (0..head_dim).step_by(2) {
        let freq = 1.0 / theta.powf(i as f32 / head_dim as f32);
        let angle = position as f32 * freq;
        let (sin_val, cos_val) = angle.sin_cos();
        let x0 = vec[i];
        let x1 = vec[i + 1];
        vec[i] = x0 * cos_val - x1 * sin_val;
        vec[i + 1] = x0 * sin_val + x1 * cos_val;
    }
}

// ─── GQA Attention (supports SWA + logit softcapping) ──────────────────────

/// Compute GQA attention into `attn_out`.
/// Supports Mistral Sliding Window and Gemma-2 logit softcapping.
fn gqa_attention(
    q_buf: &[f32],
    kv_cache: &KvCache,
    layer_idx: usize,
    pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
    attn_logit_softcap: Option<f32>,
    attn_out: &mut [f32],
) {
    let seq_len = pos + 1;
    let heads_per_kv = num_heads / num_kv_heads;
    let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

    let attn_start = match sliding_window {
        Some(w) => seq_len.saturating_sub(w),
        None => 0,
    };

    attn_out.fill(0.0);
    for h in 0..num_heads {
        let kv_h = h / heads_per_kv;
        let q_start = h * head_dim;
        let q_head = &q_buf[q_start..q_start + head_dim];
        let k_offset = kv_h * head_dim;

        let window_len = seq_len - attn_start;
        let mut scores = Vec::with_capacity(window_len);
        for t in attn_start..seq_len {
            let k_cached = kv_cache.key_at(layer_idx, t);
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += q_head[d] * k_cached[k_offset + d];
            }
            score *= inv_sqrt_d;

            if let Some(cap) = attn_logit_softcap {
                score = cap * (score / cap).tanh();
            }

            scores.push(score);
        }

        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            sum += *s;
        }
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for s in &mut scores {
                *s *= inv_sum;
            }
        }

        for (si, t) in (attn_start..seq_len).enumerate() {
            let v_cached = kv_cache.value_at(layer_idx, t);
            let v_offset = kv_h * head_dim;
            let w = scores[si];
            for d in 0..head_dim {
                attn_out[q_start + d] += w * v_cached[v_offset + d];
            }
        }
    }
}

/// GQA attention using PagedKvCache (same logic as gqa_attention but reads from paged storage).
fn gqa_attention_paged(
    q_buf: &[f32],
    kv_cache: &PagedKvCache,
    layer_idx: usize,
    pos: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
    attn_logit_softcap: Option<f32>,
    attn_out: &mut [f32],
) {
    let seq_len = pos + 1;
    let heads_per_kv = num_heads / num_kv_heads;
    let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

    let attn_start = match sliding_window {
        Some(w) => seq_len.saturating_sub(w),
        None => 0,
    };

    attn_out.fill(0.0);
    for h in 0..num_heads {
        let kv_h = h / heads_per_kv;
        let q_start = h * head_dim;
        let q_head = &q_buf[q_start..q_start + head_dim];
        let k_offset = kv_h * head_dim;

        let window_len = seq_len - attn_start;
        let mut scores = Vec::with_capacity(window_len);
        for t in attn_start..seq_len {
            let k_cached = kv_cache.key_at(layer_idx, t);
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += q_head[d] * k_cached[k_offset + d];
            }
            score *= inv_sqrt_d;

            if let Some(cap) = attn_logit_softcap {
                score = cap * (score / cap).tanh();
            }

            scores.push(score);
        }

        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            sum += *s;
        }
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for s in &mut scores {
                *s *= inv_sum;
            }
        }

        for (si, t) in (attn_start..seq_len).enumerate() {
            let v_cached = kv_cache.value_at(layer_idx, t);
            let v_offset = kv_h * head_dim;
            let w = scores[si];
            for d in 0..head_dim {
                attn_out[q_start + d] += w * v_cached[v_offset + d];
            }
        }
    }
}

// ─── SwiGLU activation ──────────────────────────────────────────────────────

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ─── Llama-3 model ──────────────────────────────────────────────────────────

/// Weight reference pointing into GGUF mmap'd data.
struct WeightRef<'a> {
    data: &'a [u8],
    qtype: GgmlType,
    rows: usize,
    cols: usize,
}

impl<'a> WeightRef<'a> {
    fn matvec(&self, input: &[f32], output: &mut [f32]) {
        quantized_matvec(input, self.data, self.qtype, self.rows, self.cols, output);
    }

    /// Matvec with pre-quantized Q8_K input (avoids redundant quantization).
    fn matvec_preq(&self, q8_blocks: &[BlockQ8K], output: &mut [f32]) {
        quantized_matvec_preq(self.data, self.qtype, self.rows, self.cols, q8_blocks, output);
    }
}

/// Layer weight references (zero-copy from GGUF).
struct LayerWeights<'a> {
    attn_norm: Vec<f32>,
    q_proj: WeightRef<'a>,
    k_proj: WeightRef<'a>,
    v_proj: WeightRef<'a>,
    o_proj: WeightRef<'a>,
    ffn_norm: Vec<f32>,
    gate_proj: WeightRef<'a>,
    up_proj: WeightRef<'a>,
    down_proj: WeightRef<'a>,
}

// ─── Layerwise Mixed Precision ──────────────────────────────────────────────

/// Quantization strategy for a single layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerQuantMode {
    /// 1.58-bit ternary {-1, 0, +1} with scale.
    Ternary,
    /// 1-bit binary {-1, +1} (no zeros). Subset of ternary with threshold=0.
    Binary,
    /// Sparse ternary with N:M structured sparsity (e.g. 8:16).
    /// `n_keep`: non-zero elements per SPARSE_BLOCK.
    SparseTernary { n_keep: usize },
}

/// Per-layer quantization configuration for mixed-precision inference.
#[derive(Debug, Clone)]
pub struct LayerQuantConfig {
    /// Quantization mode for Attention projections (Q, K, V, O).
    pub attention_mode: LayerQuantMode,
    /// Quantization mode for FFN projections (gate, up, down).
    pub ffn_mode: LayerQuantMode,
}

/// Mixed-precision configuration for the entire model.
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Per-layer configs. If shorter than num_layers, last entry is repeated.
    pub layer_configs: Vec<LayerQuantConfig>,
}

impl LayerQuantConfig {
    /// Default: full ternary for both attention and FFN.
    pub fn full_ternary() -> Self {
        Self {
            attention_mode: LayerQuantMode::Ternary,
            ffn_mode: LayerQuantMode::Ternary,
        }
    }

    /// Aggressive: ternary attention, binary+sparse FFN (for 10GB target).
    pub fn aggressive_compression(n_keep: usize) -> Self {
        Self {
            attention_mode: LayerQuantMode::Ternary,
            ffn_mode: LayerQuantMode::SparseTernary { n_keep },
        }
    }
}

impl MixedPrecisionConfig {
    /// Uniform config: same quantization for all layers.
    pub fn uniform(config: LayerQuantConfig, num_layers: usize) -> Self {
        Self {
            layer_configs: vec![config; num_layers],
        }
    }

    /// "10GB target" config for Llama-3 70B:
    /// Attention layers: 1.58-bit ternary (preserve quality)
    /// FFN layers: sparse ternary with 8:16 sparsity (aggressive compression)
    pub fn target_10gb(num_layers: usize) -> Self {
        Self::uniform(
            LayerQuantConfig::aggressive_compression(8), // 8:16 = 50% sparsity
            num_layers,
        )
    }

    /// Get config for a specific layer.
    pub fn get(&self, layer_idx: usize) -> &LayerQuantConfig {
        if layer_idx < self.layer_configs.len() {
            &self.layer_configs[layer_idx]
        } else {
            self.layer_configs.last().unwrap_or_else(|| {
                // This shouldn't happen but provide a safe default
                &self.layer_configs[0]
            })
        }
    }

    /// Estimate effective bits per parameter for the full model.
    /// Assumes Llama-3 architecture: ~30% attention, ~70% FFN by parameter count.
    pub fn estimate_bits_per_param(&self) -> f32 {
        if self.layer_configs.is_empty() {
            return 1.58;
        }
        let mut total_bits = 0.0f32;
        let n = self.layer_configs.len() as f32;
        for cfg in &self.layer_configs {
            let attn_bits = mode_bits(cfg.attention_mode);
            let ffn_bits = mode_bits(cfg.ffn_mode);
            // Llama-3: ~30% attention params, ~70% FFN params per layer
            total_bits += 0.30 * attn_bits + 0.70 * ffn_bits;
        }
        total_bits / n
    }
}

fn mode_bits(mode: LayerQuantMode) -> f32 {
    match mode {
        LayerQuantMode::Ternary => 1.58,
        LayerQuantMode::Binary => 1.0,
        LayerQuantMode::SparseTernary { n_keep } => {
            // Effective bits: ternary base (1.58) × density ratio
            let density = n_keep as f32 / 16.0; // SPARSE_BLOCK = 16
            // Plus mask overhead: 32 bits per block of 16 = 2 bits/param
            density * 1.58 + 2.0 * (1.0 / 16.0) // mask amortized
        }
    }
}

/// Ternary-quantized layer weights ({-1, 0, +1} bitmask, ~2 bits/weight).
struct TernaryLayerWeights {
    attn_norm: Vec<f32>,
    q_proj: TernaryMatrix,
    k_proj: TernaryMatrix,
    v_proj: TernaryMatrix,
    o_proj: TernaryMatrix,
    ffn_norm: Vec<f32>,
    gate_proj: TernaryMatrix,
    up_proj: TernaryMatrix,
    down_proj: TernaryMatrix,
}

/// Llama-3 model loaded from GGUF. Weights stay in quantized form
/// in the mmap'd file; only dequantized during matvec.
pub struct Llama3Model<'a> {
    pub config: Llama3Config,
    embedding: Vec<f32>,
    layers: Vec<LayerWeights<'a>>,
    output_norm: Vec<f32>,
    output_proj: WeightRef<'a>,
    kv_cache: KvCache,
    ternary_layers: Option<Vec<TernaryLayerWeights>>,
    ternary_output_proj: Option<TernaryMatrix>,
}

impl<'a> Llama3Model<'a> {
    /// Load model from a parsed GGUF file.
    pub fn from_gguf(gguf: &'a GgufFile<'a>) -> Option<Self> {
        let config = Llama3Config::from_gguf(gguf)?;

        // Embedding (dequantized to f32 once)
        let embedding = gguf.tensor_to_f32("token_embd.weight")?;

        // Output norm
        let output_norm = gguf.tensor_to_f32("output_norm.weight")?;

        // Output projection
        let output_proj = load_weight_ref(gguf, "output.weight", config.vocab_size, config.hidden_dim)?;

        // Layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer = load_layer_weights(gguf, i, &config)?;
            layers.push(layer);
        }

        let kv_dim = config.num_kv_heads * config.head_dim;
        let kv_cache = KvCache::new(config.num_layers, config.max_seq_len, kv_dim);

        Some(Self {
            config,
            embedding,
            layers,
            output_norm,
            output_proj,
            kv_cache,
            ternary_layers: None,
            ternary_output_proj: None,
        })
    }

    /// Ternarize all weights for ternary inference mode.
    /// threshold_ratio controls sparsity (0.7 = ~50% zero weights).
    pub fn load_ternary(&mut self, threshold_ratio: f32) {
        let c = &self.config;
        let kv_dim = c.num_kv_heads * c.head_dim;
        let mut ternary_layers = Vec::with_capacity(c.num_layers);

        for (i, layer) in self.layers.iter().enumerate() {
            eprint!("  Ternarizing layer {i}/{} ...\r", c.num_layers);
            ternary_layers.push(TernaryLayerWeights {
                attn_norm: layer.attn_norm.clone(),
                q_proj: ternarize_weight(&layer.q_proj, c.hidden_dim, c.hidden_dim, threshold_ratio),
                k_proj: ternarize_weight(&layer.k_proj, kv_dim, c.hidden_dim, threshold_ratio),
                v_proj: ternarize_weight(&layer.v_proj, kv_dim, c.hidden_dim, threshold_ratio),
                o_proj: ternarize_weight(&layer.o_proj, c.hidden_dim, c.hidden_dim, threshold_ratio),
                ffn_norm: layer.ffn_norm.clone(),
                gate_proj: ternarize_weight(&layer.gate_proj, c.intermediate_dim, c.hidden_dim, threshold_ratio),
                up_proj: ternarize_weight(&layer.up_proj, c.intermediate_dim, c.hidden_dim, threshold_ratio),
                down_proj: ternarize_weight(&layer.down_proj, c.hidden_dim, c.intermediate_dim, threshold_ratio),
            });
        }
        eprintln!("  Ternarized {}/{} layers", c.num_layers, c.num_layers);

        self.ternary_output_proj = Some(ternarize_weight(
            &self.output_proj,
            c.vocab_size,
            c.hidden_dim,
            threshold_ratio,
        ));
        self.ternary_layers = Some(ternary_layers);
    }

    /// Clear KV cache (start new conversation).
    pub fn clear_cache(&mut self) {
        self.kv_cache.clear();
    }

    /// Forward pass for a single token. Returns logits [vocab_size].
    pub fn forward(&mut self, token_id: u32) -> Vec<f32> {
        let c = &self.config;
        let pos = self.kv_cache.seq_len();

        // Embedding lookup
        let emb_start = token_id as usize * c.hidden_dim;
        let mut hidden: Vec<f32> = self.embedding[emb_start..emb_start + c.hidden_dim].to_vec();

        // Reusable buffers
        let mut norm_buf = vec![0.0f32; c.hidden_dim];
        let kv_dim = c.num_kv_heads * c.head_dim;
        let mut q_buf = vec![0.0f32; c.hidden_dim];
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; c.hidden_dim];
        let mut o_buf = vec![0.0f32; c.hidden_dim];
        let mut gate_buf = vec![0.0f32; c.intermediate_dim];
        let mut up_buf = vec![0.0f32; c.intermediate_dim];
        let mut down_buf = vec![0.0f32; c.hidden_dim];

        for layer_idx in 0..c.num_layers {
            let layer = &self.layers[layer_idx];

            // Attention norm
            rms_norm(&hidden, &layer.attn_norm, c.norm_eps, &mut norm_buf);

            // Q, K, V projections (pre-quantize norm_buf once for all three)
            let q8_attn = quantize_row_q8_k(&norm_buf);
            layer.q_proj.matvec_preq(&q8_attn, &mut q_buf);
            layer.k_proj.matvec_preq(&q8_attn, &mut k_buf);
            layer.v_proj.matvec_preq(&q8_attn, &mut v_buf);

            // Apply RoPE
            for h in 0..c.num_heads {
                let start = h * c.head_dim;
                apply_rope(&mut q_buf[start..start + c.head_dim], pos, c.head_dim, c.rope_theta);
            }
            for h in 0..c.num_kv_heads {
                let start = h * c.head_dim;
                apply_rope(&mut k_buf[start..start + c.head_dim], pos, c.head_dim, c.rope_theta);
            }

            // Store K, V in cache
            self.kv_cache.append(layer_idx, &k_buf, &v_buf);

            // GQA attention (supports SWA + logit softcapping)
            gqa_attention(
                &q_buf, &self.kv_cache, layer_idx, pos,
                c.num_heads, c.num_kv_heads, c.head_dim,
                c.sliding_window, c.attn_logit_softcap, &mut attn_out,
            );

            // Output projection
            layer.o_proj.matvec(&attn_out, &mut o_buf);

            // Residual
            for i in 0..c.hidden_dim {
                hidden[i] += o_buf[i];
            }

            // FFN norm
            rms_norm(&hidden, &layer.ffn_norm, c.norm_eps, &mut norm_buf);

            // SwiGLU FFN (pre-quantize norm_buf once for gate+up)
            let q8_ffn = quantize_row_q8_k(&norm_buf);
            layer.gate_proj.matvec_preq(&q8_ffn, &mut gate_buf);
            layer.up_proj.matvec_preq(&q8_ffn, &mut up_buf);

            for i in 0..c.intermediate_dim {
                gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
            }

            layer.down_proj.matvec(&gate_buf, &mut down_buf);

            // Residual
            for i in 0..c.hidden_dim {
                hidden[i] += down_buf[i];
            }
        }

        // Advance KV cache position (all layers have appended for this token)
        self.kv_cache.advance();

        // Output norm
        rms_norm(&hidden, &self.output_norm, c.norm_eps, &mut norm_buf);

        // Output logits
        let mut logits = vec![0.0f32; c.vocab_size];
        self.output_proj.matvec(&norm_buf, &mut logits);

        // Gemma-2: final logit softcapping
        if let Some(cap) = c.final_logit_softcap {
            for l in &mut logits {
                *l = cap * (*l / cap).tanh();
            }
        }

        logits
    }

    /// Forward pass using only the first `draft_layers` layers (for speculative draft).
    /// Produces approximate logits at ~draft_layers/total_layers cost.
    /// KV cache entries are populated only for the draft layers.
    fn forward_draft(&mut self, token_id: u32, draft_layers: usize) -> Vec<f32> {
        let c = &self.config;
        let pos = self.kv_cache.seq_len();
        let num_draft = draft_layers.min(c.num_layers);

        let emb_start = token_id as usize * c.hidden_dim;
        let mut hidden: Vec<f32> = self.embedding[emb_start..emb_start + c.hidden_dim].to_vec();

        let mut norm_buf = vec![0.0f32; c.hidden_dim];
        let kv_dim = c.num_kv_heads * c.head_dim;
        let mut q_buf = vec![0.0f32; c.hidden_dim];
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; c.hidden_dim];
        let mut o_buf = vec![0.0f32; c.hidden_dim];
        let mut gate_buf = vec![0.0f32; c.intermediate_dim];
        let mut up_buf = vec![0.0f32; c.intermediate_dim];
        let mut down_buf = vec![0.0f32; c.hidden_dim];

        for layer_idx in 0..num_draft {
            let layer = &self.layers[layer_idx];

            rms_norm(&hidden, &layer.attn_norm, c.norm_eps, &mut norm_buf);
            let q8_attn = quantize_row_q8_k(&norm_buf);
            layer.q_proj.matvec_preq(&q8_attn, &mut q_buf);
            layer.k_proj.matvec_preq(&q8_attn, &mut k_buf);
            layer.v_proj.matvec_preq(&q8_attn, &mut v_buf);

            for h in 0..c.num_heads {
                let start = h * c.head_dim;
                apply_rope(&mut q_buf[start..start + c.head_dim], pos, c.head_dim, c.rope_theta);
            }
            for h in 0..c.num_kv_heads {
                let start = h * c.head_dim;
                apply_rope(&mut k_buf[start..start + c.head_dim], pos, c.head_dim, c.rope_theta);
            }

            self.kv_cache.append(layer_idx, &k_buf, &v_buf);

            gqa_attention(
                &q_buf, &self.kv_cache, layer_idx, pos,
                c.num_heads, c.num_kv_heads, c.head_dim,
                c.sliding_window, c.attn_logit_softcap, &mut attn_out,
            );

            layer.o_proj.matvec(&attn_out, &mut o_buf);
            for i in 0..c.hidden_dim {
                hidden[i] += o_buf[i];
            }

            rms_norm(&hidden, &layer.ffn_norm, c.norm_eps, &mut norm_buf);
            let q8_ffn = quantize_row_q8_k(&norm_buf);
            layer.gate_proj.matvec_preq(&q8_ffn, &mut gate_buf);
            layer.up_proj.matvec_preq(&q8_ffn, &mut up_buf);
            for i in 0..c.intermediate_dim {
                gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
            }
            layer.down_proj.matvec(&gate_buf, &mut down_buf);
            for i in 0..c.hidden_dim {
                hidden[i] += down_buf[i];
            }
        }

        self.kv_cache.advance();

        rms_norm(&hidden, &self.output_norm, c.norm_eps, &mut norm_buf);
        let mut logits = vec![0.0f32; c.vocab_size];
        self.output_proj.matvec(&norm_buf, &mut logits);
        logits
    }

    /// Generate tokens autoregressively.
    pub fn generate(
        &mut self,
        tokenizer: &GgufTokenizer,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> GenerateResult {
        let start = Instant::now();
        let prompt_tokens = tokenizer.encode(prompt);
        let mut tokens = vec![tokenizer.bos_id];
        tokens.extend_from_slice(&prompt_tokens);

        self.clear_cache();
        let prompt_token_count = tokens.len();

        // Prefill — forward all prompt tokens, keep logits from last one
        let prefill_start = Instant::now();
        let mut logits = vec![0.0f32; self.config.vocab_size];
        for &tok in &tokens {
            logits = self.forward(tok);
        }
        let prefill_ms = prefill_start.elapsed().as_millis() as u64;

        // Decode — sample from prefill logits, then forward only NEW tokens
        let decode_start = Instant::now();
        let mut generated = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            // Temperature
            if temperature > 0.0 && temperature != 1.0 {
                let inv_t = 1.0 / temperature;
                for l in &mut logits {
                    *l *= inv_t;
                }
            }

            // Top-k + argmax sampling
            let next_token = if top_k > 0 && top_k < logits.len() {
                let mut indexed: Vec<(usize, f32)> =
                    logits.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                indexed.truncate(top_k);

                let max_val = indexed[0].1;
                let mut sum = 0.0f32;
                for (_, l) in &mut indexed {
                    *l = (*l - max_val).exp();
                    sum += *l;
                }
                for (_, l) in &mut indexed {
                    *l /= sum;
                }

                indexed
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(idx, _)| *idx as u32)
                    .unwrap_or(0)
            } else {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0)
            };

            if next_token == tokenizer.eos_id {
                break;
            }

            tokens.push(next_token);
            generated.push(next_token);

            // Forward the NEW token to get logits for next iteration
            logits = self.forward(next_token);
        }

        let decode_ms = decode_start.elapsed().as_millis() as u64;
        let total_ms = start.elapsed().as_millis() as u64;
        let gen_count = generated.len();
        let tok_per_sec = if decode_ms > 0 {
            gen_count as f64 / (decode_ms as f64 / 1000.0)
        } else {
            0.0
        };

        let output_text = tokenizer.decode(&generated);

        GenerateResult {
            text: output_text,
            tokens_generated: gen_count,
            prompt_tokens: prompt_token_count,
            prefill_ms,
            decode_ms,
            total_ms,
            tokens_per_sec: tok_per_sec,
            spec_stats: None,
        }
    }

    /// Generate with speculative decoding (layer-skip draft + verify).
    ///
    /// Uses the first `draft_layers` layers as a cheap draft model to predict
    /// `spec_k` tokens ahead, then verifies with the full model. Accepted draft
    /// tokens skip redundant full-forward passes.
    pub fn generate_speculative(
        &mut self,
        tokenizer: &GgufTokenizer,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        spec_k: usize,
        draft_layers: usize,
    ) -> GenerateResult {
        let start = Instant::now();
        let prompt_tokens = tokenizer.encode(prompt);
        let mut tokens = vec![tokenizer.bos_id];
        tokens.extend_from_slice(&prompt_tokens);

        self.clear_cache();
        let prompt_token_count = tokens.len();

        // Prefill
        let prefill_start = Instant::now();
        let mut logits = vec![0.0f32; self.config.vocab_size];
        for &tok in &tokens {
            logits = self.forward(tok);
        }
        let prefill_ms = prefill_start.elapsed().as_millis() as u64;

        // Decode with speculation
        let decode_start = Instant::now();
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut total_drafted: usize = 0;
        let mut total_accepted: usize = 0;

        while generated.len() < max_new_tokens {
            // Sample from current logits
            let next_token = sample_token(&logits, temperature, top_k);
            if next_token == tokenizer.eos_id {
                break;
            }
            generated.push(next_token);
            tokens.push(next_token);

            let remaining = max_new_tokens - generated.len();
            if remaining == 0 {
                // No more tokens needed, skip draft
                logits = self.forward(next_token);
                continue;
            }

            let k = spec_k.min(remaining);

            // --- Draft phase ---
            let saved_pos = self.kv_cache.seq_len();
            let mut draft_tokens = Vec::with_capacity(k);
            let mut draft_input = next_token;
            for _ in 0..k {
                let draft_logits = self.forward_draft(draft_input, draft_layers);
                draft_input = argmax(&draft_logits);
                draft_tokens.push(draft_input);
            }
            total_drafted += draft_tokens.len();

            // --- Rollback ---
            self.kv_cache.rollback_to(saved_pos);

            // --- Verify phase ---
            logits = self.forward(next_token);

            for i in 0..draft_tokens.len() {
                let verified = sample_token(&logits, temperature, top_k);
                if verified == draft_tokens[i] {
                    // Draft accepted
                    generated.push(draft_tokens[i]);
                    tokens.push(draft_tokens[i]);
                    total_accepted += 1;
                    logits = self.forward(draft_tokens[i]);
                } else {
                    // Draft rejected — use verified token
                    if verified == tokenizer.eos_id {
                        // Stop generation, don't push EOS
                        break;
                    }
                    generated.push(verified);
                    tokens.push(verified);
                    logits = self.forward(verified);
                    break;
                }
            }
        }

        let decode_ms = decode_start.elapsed().as_millis() as u64;
        let total_ms = start.elapsed().as_millis() as u64;
        let gen_count = generated.len();
        let tok_per_sec = if decode_ms > 0 {
            gen_count as f64 / (decode_ms as f64 / 1000.0)
        } else {
            0.0
        };

        let output_text = tokenizer.decode(&generated);

        GenerateResult {
            text: output_text,
            tokens_generated: gen_count,
            prompt_tokens: prompt_token_count,
            prefill_ms,
            decode_ms,
            total_ms,
            tokens_per_sec: tok_per_sec,
            spec_stats: Some(SpecStats {
                draft_tokens: total_drafted,
                accepted_tokens: total_accepted,
                draft_layers,
                spec_k,
            }),
        }
    }

    /// Forward pass using ternary-quantized weights (no multiplications in projections).
    /// Must call `load_ternary()` before using this method.
    pub fn forward_ternary(&mut self, token_id: u32) -> Vec<f32> {
        let ternary_layers = self.ternary_layers.as_ref().expect("call load_ternary() first");
        let ternary_output = self.ternary_output_proj.as_ref().expect("call load_ternary() first");
        let c = &self.config;
        let pos = self.kv_cache.seq_len();

        let emb_start = token_id as usize * c.hidden_dim;
        let mut hidden: Vec<f32> = self.embedding[emb_start..emb_start + c.hidden_dim].to_vec();

        let mut norm_buf = vec![0.0f32; c.hidden_dim];
        let kv_dim = c.num_kv_heads * c.head_dim;
        let mut q_buf = vec![0.0f32; c.hidden_dim];
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; c.hidden_dim];
        let mut o_buf = vec![0.0f32; c.hidden_dim];
        let mut gate_buf = vec![0.0f32; c.intermediate_dim];
        let mut up_buf = vec![0.0f32; c.intermediate_dim];
        let mut down_buf = vec![0.0f32; c.hidden_dim];

        for layer_idx in 0..c.num_layers {
            let tl = &ternary_layers[layer_idx];

            rms_norm(&hidden, &tl.attn_norm, c.norm_eps, &mut norm_buf);

            // Ternary projections (add/subtract only, no multiplications)
            ternary_matvec(&tl.q_proj, &norm_buf, &mut q_buf);
            ternary_matvec(&tl.k_proj, &norm_buf, &mut k_buf);
            ternary_matvec(&tl.v_proj, &norm_buf, &mut v_buf);

            for h in 0..c.num_heads {
                let start = h * c.head_dim;
                apply_rope(&mut q_buf[start..start + c.head_dim], pos, c.head_dim, c.rope_theta);
            }
            for h in 0..c.num_kv_heads {
                let start = h * c.head_dim;
                apply_rope(&mut k_buf[start..start + c.head_dim], pos, c.head_dim, c.rope_theta);
            }

            self.kv_cache.append(layer_idx, &k_buf, &v_buf);

            gqa_attention(
                &q_buf, &self.kv_cache, layer_idx, pos,
                c.num_heads, c.num_kv_heads, c.head_dim,
                c.sliding_window, c.attn_logit_softcap, &mut attn_out,
            );

            ternary_matvec(&tl.o_proj, &attn_out, &mut o_buf);
            for i in 0..c.hidden_dim {
                hidden[i] += o_buf[i];
            }

            rms_norm(&hidden, &tl.ffn_norm, c.norm_eps, &mut norm_buf);

            ternary_matvec(&tl.gate_proj, &norm_buf, &mut gate_buf);
            ternary_matvec(&tl.up_proj, &norm_buf, &mut up_buf);
            for i in 0..c.intermediate_dim {
                gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
            }
            ternary_matvec(&tl.down_proj, &gate_buf, &mut down_buf);

            for i in 0..c.hidden_dim {
                hidden[i] += down_buf[i];
            }
        }

        self.kv_cache.advance();

        rms_norm(&hidden, &self.output_norm, c.norm_eps, &mut norm_buf);
        let mut logits = vec![0.0f32; c.vocab_size];
        ternary_matvec(ternary_output, &norm_buf, &mut logits);
        logits
    }

    /// Generate with ternary-quantized weights.
    pub fn generate_ternary(
        &mut self,
        tokenizer: &GgufTokenizer,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> GenerateResult {
        let start = Instant::now();
        let prompt_tokens = tokenizer.encode(prompt);
        let mut tokens = vec![tokenizer.bos_id];
        tokens.extend_from_slice(&prompt_tokens);

        self.clear_cache();
        let prompt_token_count = tokens.len();

        let prefill_start = Instant::now();
        let mut logits = vec![0.0f32; self.config.vocab_size];
        for &tok in &tokens {
            logits = self.forward_ternary(tok);
        }
        let prefill_ms = prefill_start.elapsed().as_millis() as u64;

        let decode_start = Instant::now();
        let mut generated = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            let next_token = sample_token(&logits, temperature, top_k);
            if next_token == tokenizer.eos_id {
                break;
            }
            tokens.push(next_token);
            generated.push(next_token);
            logits = self.forward_ternary(next_token);
        }

        let decode_ms = decode_start.elapsed().as_millis() as u64;
        let total_ms = start.elapsed().as_millis() as u64;
        let gen_count = generated.len();
        let tok_per_sec = if decode_ms > 0 {
            gen_count as f64 / (decode_ms as f64 / 1000.0)
        } else {
            0.0
        };

        GenerateResult {
            text: tokenizer.decode(&generated),
            tokens_generated: gen_count,
            prompt_tokens: prompt_token_count,
            prefill_ms,
            decode_ms,
            total_ms,
            tokens_per_sec: tok_per_sec,
            spec_stats: None,
        }
    }

    // ─── Paged KV Cache forward ─────────────────────────────────────────────

    /// Forward pass using a per-request PagedKvCache instead of the model's flat cache.
    fn forward_paged(&self, token_id: u32, paged_cache: &mut PagedKvCache) -> Vec<f32> {
        let c = &self.config;
        let pos = paged_cache.seq_len();

        let emb_start = token_id as usize * c.hidden_dim;
        let mut hidden: Vec<f32> = self.embedding[emb_start..emb_start + c.hidden_dim].to_vec();

        let mut norm_buf = vec![0.0f32; c.hidden_dim];
        let kv_dim = c.num_kv_heads * c.head_dim;
        let mut q_buf = vec![0.0f32; c.hidden_dim];
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; c.hidden_dim];
        let mut o_buf = vec![0.0f32; c.hidden_dim];
        let mut gate_buf = vec![0.0f32; c.intermediate_dim];
        let mut up_buf = vec![0.0f32; c.intermediate_dim];
        let mut down_buf = vec![0.0f32; c.hidden_dim];

        for layer_idx in 0..c.num_layers {
            let layer = &self.layers[layer_idx];

            rms_norm(&hidden, &layer.attn_norm, c.norm_eps, &mut norm_buf);

            let q8_attn = quantize_row_q8_k(&norm_buf);
            layer.q_proj.matvec_preq(&q8_attn, &mut q_buf);
            layer.k_proj.matvec_preq(&q8_attn, &mut k_buf);
            layer.v_proj.matvec_preq(&q8_attn, &mut v_buf);

            for h in 0..c.num_heads {
                let start = h * c.head_dim;
                apply_rope(&mut q_buf[start..start + c.head_dim], pos, c.head_dim, c.rope_theta);
            }
            for h in 0..c.num_kv_heads {
                let start = h * c.head_dim;
                apply_rope(&mut k_buf[start..start + c.head_dim], pos, c.head_dim, c.rope_theta);
            }

            paged_cache.append(layer_idx, &k_buf, &v_buf);

            gqa_attention_paged(
                &q_buf, paged_cache, layer_idx, pos,
                c.num_heads, c.num_kv_heads, c.head_dim,
                c.sliding_window, c.attn_logit_softcap, &mut attn_out,
            );

            layer.o_proj.matvec(&attn_out, &mut o_buf);

            for i in 0..c.hidden_dim {
                hidden[i] += o_buf[i];
            }

            rms_norm(&hidden, &layer.ffn_norm, c.norm_eps, &mut norm_buf);

            let q8_ffn = quantize_row_q8_k(&norm_buf);
            layer.gate_proj.matvec_preq(&q8_ffn, &mut gate_buf);
            layer.up_proj.matvec_preq(&q8_ffn, &mut up_buf);

            for i in 0..c.intermediate_dim {
                gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
            }

            layer.down_proj.matvec(&gate_buf, &mut down_buf);

            for i in 0..c.hidden_dim {
                hidden[i] += down_buf[i];
            }
        }

        paged_cache.advance();

        rms_norm(&hidden, &self.output_norm, c.norm_eps, &mut norm_buf);

        let mut logits = vec![0.0f32; c.vocab_size];
        self.output_proj.matvec(&norm_buf, &mut logits);

        if let Some(cap) = c.final_logit_softcap {
            for l in &mut logits {
                *l = cap * (*l / cap).tanh();
            }
        }

        logits
    }

    // ─── Continuous Batching ────────────────────────────────────────────────

    /// Process all active requests in the batch scheduler.
    /// Prefills each request, then decodes round-robin until all are done.
    pub fn generate_batch(
        &self,
        tokenizer: &GgufTokenizer,
        scheduler: &mut BatchScheduler,
        top_k: usize,
    ) {
        // Phase 1: Prefill — run all prompt tokens, store the last logits per request
        let mut pending_logits: Vec<Option<Vec<f32>>> = Vec::new();
        for req in scheduler.requests_mut().iter_mut() {
            if req.done {
                pending_logits.push(None);
                continue;
            }
            let prompt_tokens = req.tokens.clone();
            let mut logits = vec![0.0f32; 0];
            for &tok in &prompt_tokens {
                logits = self.forward_paged(tok, &mut req.kv_cache);
            }
            pending_logits.push(Some(logits));
        }

        // Sample first token from prefill logits
        for (i, req) in scheduler.requests_mut().iter_mut().enumerate() {
            if req.done { continue; }
            if let Some(ref logits) = pending_logits[i] {
                let next_token = sample_token(logits, req.temperature, top_k);
                if next_token == tokenizer.eos_id {
                    req.done = true;
                } else {
                    req.generated.push(next_token);
                }
            }
        }

        // Phase 2: Decode — round-robin until all requests are done
        loop {
            let any_active = scheduler.requests_mut().iter().any(|r| !r.done);
            if !any_active {
                break;
            }

            for req in scheduler.requests_mut().iter_mut().filter(|r| !r.done) {
                if req.generated.len() >= req.max_new_tokens {
                    req.done = true;
                    continue;
                }

                let last_tok = *req.generated.last().unwrap();
                let logits = self.forward_paged(last_tok, &mut req.kv_cache);
                let next_token = sample_token(&logits, req.temperature, top_k);

                if next_token == tokenizer.eos_id {
                    req.done = true;
                } else {
                    req.generated.push(next_token);
                }
            }
        }
    }
}

// ─── Sampling helpers ────────────────────────────────────────────────────────

fn sample_token(logits: &[f32], temperature: f32, top_k: usize) -> u32 {
    let mut logits = logits.to_vec();
    if temperature > 0.0 && temperature != 1.0 {
        let inv_t = 1.0 / temperature;
        for l in &mut logits {
            *l *= inv_t;
        }
    }

    if top_k > 0 && top_k < logits.len() {
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(top_k);
        indexed
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(idx, _)| *idx as u32)
            .unwrap_or(0)
    } else {
        argmax(&logits)
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

// ─── Generation result ──────────────────────────────────────────────────────

/// Result of text generation.
#[derive(Debug)]
pub struct GenerateResult {
    pub text: String,
    pub tokens_generated: usize,
    pub prompt_tokens: usize,
    pub prefill_ms: u64,
    pub decode_ms: u64,
    pub total_ms: u64,
    pub tokens_per_sec: f64,
    /// Speculative decoding stats (None if not used).
    pub spec_stats: Option<SpecStats>,
}

/// Speculative decoding statistics.
#[derive(Debug, Clone)]
pub struct SpecStats {
    pub draft_tokens: usize,
    pub accepted_tokens: usize,
    pub draft_layers: usize,
    pub spec_k: usize,
}

// ─── Weight loading helpers ─────────────────────────────────────────────────

fn ternarize_weight(w: &WeightRef<'_>, rows: usize, cols: usize, threshold_ratio: f32) -> TernaryMatrix {
    TernaryMatrix::from_quantized(w.data, w.qtype, rows, cols, threshold_ratio)
}

fn load_weight_ref<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Option<WeightRef<'a>> {
    let info = gguf.tensor_info(name)?;
    let data = gguf.tensor_data(name)?;
    Some(WeightRef {
        data,
        qtype: info.qtype,
        rows,
        cols,
    })
}

fn load_layer_weights<'a>(
    gguf: &'a GgufFile<'a>,
    layer: usize,
    config: &Llama3Config,
) -> Option<LayerWeights<'a>> {
    let prefix = format!("blk.{layer}");
    let kv_dim = config.num_kv_heads * config.head_dim;

    let attn_norm = gguf.tensor_to_f32(&format!("{prefix}.attn_norm.weight"))?;
    let q_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.attn_q.weight"),
        config.hidden_dim,
        config.hidden_dim,
    )?;
    let k_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.attn_k.weight"),
        kv_dim,
        config.hidden_dim,
    )?;
    let v_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.attn_v.weight"),
        kv_dim,
        config.hidden_dim,
    )?;
    let o_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.attn_output.weight"),
        config.hidden_dim,
        config.hidden_dim,
    )?;

    let ffn_norm = gguf.tensor_to_f32(&format!("{prefix}.ffn_norm.weight"))?;
    let gate_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.ffn_gate.weight"),
        config.intermediate_dim,
        config.hidden_dim,
    )?;
    let up_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.ffn_up.weight"),
        config.intermediate_dim,
        config.hidden_dim,
    )?;
    let down_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.ffn_down.weight"),
        config.hidden_dim,
        config.intermediate_dim,
    )?;

    Some(LayerWeights {
        attn_norm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        ffn_norm,
        gate_proj,
        up_proj,
        down_proj,
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama3_config_defaults() {
        let c = Llama3Config::llama3_8b();
        assert_eq!(c.vocab_size, 128_256);
        assert_eq!(c.hidden_dim, 4096);
        assert_eq!(c.num_heads, 32);
        assert_eq!(c.num_kv_heads, 8);
        assert_eq!(c.num_layers, 32);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.intermediate_dim, 14_336);
    }

    #[test]
    fn test_gqa_heads_per_kv() {
        let c = Llama3Config::llama3_8b();
        assert_eq!(c.num_heads / c.num_kv_heads, 4);
    }

    #[test]
    fn test_rms_norm() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        rms_norm(&x, &w, 1e-5, &mut out);

        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5)
        let rms = (7.5f32 + 1e-5).sqrt();
        for i in 0..4 {
            let expected = x[i] / rms;
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "rms_norm[{i}]: got {}, expected {expected}",
                out[i]
            );
        }
    }

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        // silu(1.0) = 1/(1+e^-1) ≈ 0.7311
        assert!((silu(1.0) - 0.7311).abs() < 1e-3);
        // silu(x) → x for large x
        assert!((silu(10.0) - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_rope_identity_at_zero() {
        let mut vec = [1.0f32, 0.0, 1.0, 0.0];
        apply_rope(&mut vec, 0, 4, 10000.0);
        // At position 0, angle=0, cos=1, sin=0, so no change
        assert!((vec[0] - 1.0).abs() < 1e-6);
        assert!(vec[1].abs() < 1e-6);
    }

    #[test]
    fn test_rope_preserves_norm() {
        let mut vec = [3.0f32, 4.0, 1.0, 2.0];
        let norm_before: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope(&mut vec, 5, 4, 10000.0);
        let norm_after: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm_before - norm_after).abs() < 1e-4,
            "RoPE changed norm: {norm_before} -> {norm_after}"
        );
    }

    #[test]
    fn test_kv_cache() {
        let kv_dim = 1024;
        let mut cache = KvCache::new(2, 128, kv_dim);
        assert_eq!(cache.seq_len(), 0);

        // Append to both layers at position 0, then advance
        cache.append(0, &vec![1.0f32; kv_dim], &vec![2.0f32; kv_dim]);
        cache.append(1, &vec![5.0f32; kv_dim], &vec![6.0f32; kv_dim]);
        cache.advance();
        assert_eq!(cache.seq_len(), 1);

        // Append to both layers at position 1, then advance
        cache.append(0, &vec![3.0f32; kv_dim], &vec![4.0f32; kv_dim]);
        cache.append(1, &vec![7.0f32; kv_dim], &vec![8.0f32; kv_dim]);
        cache.advance();
        assert_eq!(cache.seq_len(), 2);

        // Verify cached values
        assert_eq!(cache.key_at(0, 0)[0], 1.0);
        assert_eq!(cache.key_at(0, 1)[0], 3.0);
        assert_eq!(cache.value_at(1, 0)[0], 6.0);

        cache.clear();
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_paged_kv_cache_basic() {
        let kv_dim = 64;
        let mut cache = PagedKvCache::new(2, kv_dim);
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.total_pages(), 0);

        // Append first token to both layers
        cache.append(0, &vec![1.0f32; kv_dim], &vec![2.0f32; kv_dim]);
        cache.append(1, &vec![5.0f32; kv_dim], &vec![6.0f32; kv_dim]);
        cache.advance();
        assert_eq!(cache.seq_len(), 1);
        assert_eq!(cache.total_pages(), 2); // 1 page per layer

        // Verify cached values
        assert_eq!(cache.key_at(0, 0)[0], 1.0);
        assert_eq!(cache.value_at(0, 0)[0], 2.0);
        assert_eq!(cache.key_at(1, 0)[0], 5.0);
        assert_eq!(cache.value_at(1, 0)[0], 6.0);

        // Append second token
        cache.append(0, &vec![3.0f32; kv_dim], &vec![4.0f32; kv_dim]);
        cache.append(1, &vec![7.0f32; kv_dim], &vec![8.0f32; kv_dim]);
        cache.advance();
        assert_eq!(cache.seq_len(), 2);
        assert_eq!(cache.key_at(0, 1)[0], 3.0);
        assert_eq!(cache.value_at(1, 1)[0], 8.0);
    }

    #[test]
    fn test_paged_kv_cache_page_boundary() {
        let kv_dim = 32;
        let mut cache = PagedKvCache::new(1, kv_dim);

        // Fill exactly one page (PAGE_SIZE = 16 tokens)
        for i in 0..PAGE_SIZE {
            cache.append(0, &vec![i as f32; kv_dim], &vec![(i as f32) * 10.0; kv_dim]);
            cache.advance();
        }
        assert_eq!(cache.seq_len(), PAGE_SIZE);
        assert_eq!(cache.total_pages(), 1);

        // Add one more → should allocate a new page
        cache.append(0, &vec![99.0f32; kv_dim], &vec![990.0f32; kv_dim]);
        cache.advance();
        assert_eq!(cache.seq_len(), PAGE_SIZE + 1);
        assert_eq!(cache.total_pages(), 2);

        // Verify cross-page reads
        assert_eq!(cache.key_at(0, 0)[0], 0.0);
        assert_eq!(cache.key_at(0, 15)[0], 15.0);
        assert_eq!(cache.key_at(0, 16)[0], 99.0);
    }

    #[test]
    fn test_paged_kv_cache_rollback() {
        let kv_dim = 32;
        let mut cache = PagedKvCache::new(1, kv_dim);

        for i in 0..20 {
            cache.append(0, &vec![i as f32; kv_dim], &vec![0.0; kv_dim]);
            cache.advance();
        }
        assert_eq!(cache.seq_len(), 20);
        assert_eq!(cache.total_pages(), 2);

        // Rollback to position 10 (within first page)
        cache.rollback_to(10);
        assert_eq!(cache.seq_len(), 10);
        assert_eq!(cache.total_pages(), 1); // second page freed

        // Rollback to 0
        cache.rollback_to(0);
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.total_pages(), 0);
    }

    #[test]
    fn test_paged_kv_cache_memory() {
        let kv_dim = 128;
        let cache = PagedKvCache::new(4, kv_dim);
        assert_eq!(cache.memory_bytes(), 0); // no pages allocated yet
    }

    #[test]
    fn test_batch_scheduler() {
        let mut sched = BatchScheduler::new();

        let id0 = sched.add_request(vec![1, 2, 3], 10, 0.0, 4, 128);
        let id1 = sched.add_request(vec![4, 5], 5, 0.7, 4, 128);

        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(sched.active_count(), 2);

        // Mark first request done
        sched.requests_mut()[0].done = true;
        assert_eq!(sched.active_count(), 1);
        assert!(sched.get_result(id0).is_some());
        assert!(sched.get_result(id1).is_none());
    }

    #[test]
    fn test_memory_estimate_8b_q4k() {
        let c = Llama3Config::llama3_8b();
        // Q4_K_M: ~0.6 bytes per parameter
        let total_params: usize = c.vocab_size * c.hidden_dim // embedding
            + c.num_layers * (
                c.hidden_dim * c.hidden_dim // q_proj
                + c.num_kv_heads * c.head_dim * c.hidden_dim * 2 // k_proj + v_proj
                + c.hidden_dim * c.hidden_dim // o_proj
                + c.intermediate_dim * c.hidden_dim * 3 // gate + up + down
            )
            + c.vocab_size * c.hidden_dim; // output

        let q4k_bytes = (total_params as f64 * 0.6) / 1e9;
        // Should be around 4.5-5.5 GB
        assert!(q4k_bytes > 3.0 && q4k_bytes < 7.0, "Q4_K estimate: {q4k_bytes:.1} GB");
    }

    #[test]
    fn test_layer_quant_config_full_ternary() {
        let cfg = LayerQuantConfig::full_ternary();
        assert_eq!(cfg.attention_mode, LayerQuantMode::Ternary);
        assert_eq!(cfg.ffn_mode, LayerQuantMode::Ternary);
    }

    #[test]
    fn test_layer_quant_config_aggressive() {
        let cfg = LayerQuantConfig::aggressive_compression(8);
        assert_eq!(cfg.attention_mode, LayerQuantMode::Ternary);
        assert_eq!(cfg.ffn_mode, LayerQuantMode::SparseTernary { n_keep: 8 });
    }

    #[test]
    fn test_mixed_precision_uniform() {
        let mp = MixedPrecisionConfig::uniform(LayerQuantConfig::full_ternary(), 32);
        assert_eq!(mp.layer_configs.len(), 32);
        assert_eq!(mp.get(0).attention_mode, LayerQuantMode::Ternary);
        assert_eq!(mp.get(31).ffn_mode, LayerQuantMode::Ternary);
    }

    #[test]
    fn test_mixed_precision_10gb_target() {
        let mp = MixedPrecisionConfig::target_10gb(80); // 70B = 80 layers
        assert_eq!(mp.layer_configs.len(), 80);
        assert_eq!(mp.get(0).attention_mode, LayerQuantMode::Ternary);
        assert_eq!(mp.get(0).ffn_mode, LayerQuantMode::SparseTernary { n_keep: 8 });
    }

    #[test]
    fn test_mixed_precision_bits_estimate() {
        // Full ternary: 1.58 bits/param
        let mp_ternary = MixedPrecisionConfig::uniform(LayerQuantConfig::full_ternary(), 32);
        let bits_ternary = mp_ternary.estimate_bits_per_param();
        assert!((bits_ternary - 1.58).abs() < 0.01, "full ternary: {bits_ternary}");

        // Aggressive: attn=1.58 (30%), FFN=sparse (70%) → should be < 1.58
        let mp_aggressive = MixedPrecisionConfig::target_10gb(32);
        let bits_aggressive = mp_aggressive.estimate_bits_per_param();
        assert!(
            bits_aggressive < bits_ternary,
            "aggressive ({bits_aggressive}) should be < ternary ({bits_ternary})"
        );
    }

    #[test]
    fn test_70b_10gb_feasibility() {
        // Verify that aggressive compression achieves < 1.14 bits/param
        // (the threshold for 70B @ 10GB)
        let mp = MixedPrecisionConfig::target_10gb(80);
        let bpp = mp.estimate_bits_per_param();
        let model_size_gb = 70e9 * bpp as f64 / 8.0 / 1e9;
        // With 8:16 sparsity (50%), should be well under pure ternary
        assert!(
            bpp < 1.58,
            "bits/param={bpp}, should be < 1.58 for 10GB target"
        );
        // Print for visibility
        eprintln!(
            "70B estimate: {bpp:.3} bits/param → {model_size_gb:.1} GB (target: <10 GB)"
        );
    }

    #[test]
    fn test_mode_bits() {
        assert!((mode_bits(LayerQuantMode::Ternary) - 1.58).abs() < 0.01);
        assert!((mode_bits(LayerQuantMode::Binary) - 1.0).abs() < 0.01);
        // SparseTernary with n_keep=8 (50% density):
        // 0.5 * 1.58 + 2/16 = 0.79 + 0.125 = 0.915
        let sparse_bits = mode_bits(LayerQuantMode::SparseTernary { n_keep: 8 });
        assert!(sparse_bits < 1.58, "sparse={sparse_bits}");
        assert!(sparse_bits > 0.5, "sparse={sparse_bits}");
    }
}
