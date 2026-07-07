//! Multi-architecture LLM inference engine with GGUF quantized weights.
//!
//! Supports Llama-3, Mistral (Sliding Window Attention), and Gemma-2
//! (Logit Softcapping). Performs inference directly on Q4_K_M/Q8_0
//! quantized data via fused dequantize+matvec.

use crate::gguf::{
    quantize_row_q8_k, quantized_matvec, quantized_matvec_preq, sparse_ternary_matvec,
    ternary_matvec, BlockQ8K, GgmlType, GgufFile, GgufTokenizer, SparseTernaryMatrix,
    TernaryMatrix,
};
use std::time::Instant;

// ─── Model architecture ─────────────────────────────────────────────────────

/// Supported model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    Llama,
    Mistral,
    Gemma2,
    /// Gemma 3n (E2B/E4B): AltUp + Laurel + per-layer input embedding +
    /// shared KV cache + activation sparsity for first N layers.
    Gemma3n,
    Qwen2,
    Qwen3,
    Qwen3_5,
}

impl ModelArch {
    /// Detect architecture from GGUF metadata key `general.architecture`.
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Self {
        match gguf.meta_str("general.architecture") {
            Some("mistral") => Self::Mistral,
            Some("gemma2") => Self::Gemma2,
            Some("gemma3n") => Self::Gemma3n,
            Some("qwen3moe" | "qwen3") => {
                if gguf.meta_u32("qwen3.full_attention_interval").is_some()
                    || gguf.meta_u32("qwen3moe.full_attention_interval").is_some()
                {
                    Self::Qwen3_5
                } else {
                    Self::Qwen3
                }
            }
            Some(s) if s.starts_with("qwen") => Self::Qwen2,
            _ => Self::Llama,
        }
    }

    /// GGUF metadata key prefix for this architecture.
    const fn meta_prefix(&self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::Gemma2 => "gemma2",
            Self::Gemma3n => "gemma3n",
            Self::Qwen2 => "qwen2",
            Self::Qwen3 | Self::Qwen3_5 => "qwen3",
        }
    }

    /// Returns true if this architecture uses NEOX RoPE (half rotation:
    /// pair (i, i+d/2) rotated together) as opposed to NORM RoPE (pair
    /// (i, i+1)). llama.cpp source `llama_model_rope_type` reference.
    /// - NORM (Llama family, Mistral): q/k weights are permuted in GGUF
    ///   conversion so paired rotation is equivalent to HF half rotation.
    /// - NEOX (Qwen 2/3, Gemma 2): weights stored as-is in HF layout,
    ///   forward pass applies half rotation directly.
    pub const fn use_neox_rope(&self) -> bool {
        matches!(
            self,
            Self::Qwen2 | Self::Qwen3 | Self::Qwen3_5 | Self::Gemma2 | Self::Gemma3n
        )
    }

    /// Resolve the actual GGUF metadata prefix (some models use versioned keys).
    fn resolve_prefix(&self, gguf: &GgufFile<'_>) -> String {
        let raw = gguf
            .meta_str("general.architecture")
            .unwrap_or(self.meta_prefix());
        if gguf.meta_u32(&format!("{raw}.embedding_length")).is_some() {
            return raw.to_string();
        }
        self.meta_prefix().to_string()
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
    /// Qwen3.5: full attention interval (e.g. 4 = every 4th layer is full attention).
    /// None for pure attention models.
    pub full_attention_interval: Option<usize>,
    /// Qwen3.5 DeltaNet: number of QK heads for linear attention.
    pub linear_num_kv_heads: Option<usize>,
    /// Qwen3.5 DeltaNet: QK head dimension.
    pub linear_qk_head_dim: Option<usize>,
    /// Qwen3.5 DeltaNet: V head dimension.
    pub linear_kv_head_dim: Option<usize>,
    /// Qwen3.5 DeltaNet: number of V heads.
    pub linear_num_v_heads: Option<usize>,
    /// Qwen3.5 DeltaNet: causal conv1d kernel size (typically 4).
    pub linear_conv_kernel_dim: Option<usize>,
    /// Gemma 3n: per-layer sliding window boolean pattern. When Some,
    /// entry `i = true` means layer `i` uses SWA, `false` = full attention.
    /// Supersedes Gemma 2 even/odd alternation.
    pub sliding_window_pattern: Option<Vec<bool>>,
    /// Gemma 3n: per-layer FFN activation sparsity scale. First N entries
    /// are finite (GELU + sparsity threshold `scale * std`), rest are -inf
    /// (dense, no sparsity). Absent → all layers dense (SiLU for non-Gemma).
    pub activation_sparsity_scale: Option<Vec<f32>>,
    /// Gemma 3n: number of layers with unique KV cache. Later layers reuse
    /// KV cache from earlier layers (layer_i for i >= shared_kv_layers
    /// shares cache with layer_(i - shared_kv_layers) or similar mapping,
    /// exact scheme confirmed in Phase 4).
    pub shared_kv_layers: Option<usize>,
    /// Gemma 3n: per-layer input embedding dimension (256 for E2B).
    /// Additional embedding table `per_layer_token_embd.weight` projected
    /// and added to hidden state at each layer.
    pub per_layer_input_embedding_dim: Option<usize>,
    /// Gemma 3n: number of AltUp residual streams (4 for E2B). Enables the
    /// AltUp mechanism (alternative up-projection with router + correct/predict).
    pub altup_num_inputs: Option<usize>,
    /// Gemma 3n: AltUp active input index (0 for E2B).
    pub altup_active_idx: Option<usize>,
}

impl Llama3Config {
    /// Load config from GGUF metadata (auto-detects architecture).
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Option<Self> {
        let arch = ModelArch::from_gguf(gguf);
        let prefix = arch.resolve_prefix(gguf);

        let hidden_dim = gguf.meta_u32(&format!("{prefix}.embedding_length"))? as usize;
        let num_heads = gguf.meta_u32(&format!("{prefix}.attention.head_count"))? as usize;
        let num_kv_heads = gguf
            .meta_u32(&format!("{prefix}.attention.head_count_kv"))
            .unwrap_or(num_heads as u32) as usize;
        let num_layers = gguf.meta_u32(&format!("{prefix}.block_count"))? as usize;
        // Gemma 3n stores `feed_forward_length` as a per-layer array (all same
        // value for E2B). Fall back to reading first element when scalar u32
        // read fails.
        let intermediate_dim = gguf
            .meta_u32(&format!("{prefix}.feed_forward_length"))
            .map(|v| v as usize)
            .or_else(|| {
                gguf.meta(&format!("{prefix}.feed_forward_length"))
                    .and_then(|v| match v {
                        crate::gguf::MetaValue::Array(arr) => {
                            arr.first().and_then(|item| item.as_u32().map(|v| v as usize))
                        }
                        _ => None,
                    })
            })?;
        let max_seq_len = gguf
            .meta_u32(&format!("{prefix}.context_length"))
            .unwrap_or(8192) as usize;
        let max_seq_len = max_seq_len.min(8192);
        let vocab_size = gguf.meta_u32(&format!("{prefix}.vocab_size")).or_else(|| {
            gguf.meta("tokenizer.ggml.tokens").and_then(|v| match v {
                crate::gguf::MetaValue::Array(arr) => Some(arr.len() as u32),
                _ => None,
            })
        })? as usize;
        let rope_theta = gguf
            .meta_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(if arch == ModelArch::Mistral {
                1_000_000.0
            } else {
                500_000.0
            });
        let norm_eps = gguf
            .meta_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        // Gemma-2 has explicit head_dim (256 for 2B, != hidden_dim/num_heads).
        // Qwen 3 also stores explicit key_length.
        // Fall back to hidden_dim/num_heads for models without this metadata (Llama, Mistral).
        let head_dim = gguf
            .meta_u32(&format!("{prefix}.attention.key_length"))
            .map_or(hidden_dim / num_heads, |v| v as usize);

        // Mistral: sliding window attention
        let sliding_window = gguf
            .meta_u32(&format!("{prefix}.attention.sliding_window"))
            .map(|v| v as usize);

        // Gemma-2: logit softcapping
        let attn_logit_softcap = gguf.meta_f32(&format!("{prefix}.attn_logit_softcapping"));
        let final_logit_softcap = gguf.meta_f32(&format!("{prefix}.final_logit_softcapping"));

        // Qwen3.5 DeltaNet hybrid fields
        let full_attention_interval = gguf
            .meta_u32(&format!("{prefix}.full_attention_interval"))
            .map(|v| v as usize);
        let linear_num_kv_heads = gguf
            .meta_u32(&format!("{prefix}.linear_num_key_heads"))
            .map(|v| v as usize);
        let linear_qk_head_dim = gguf
            .meta_u32(&format!("{prefix}.linear_qk_head_dim"))
            .map(|v| v as usize);
        let linear_kv_head_dim = gguf
            .meta_u32(&format!("{prefix}.linear_key_value_head_dim"))
            .map(|v| v as usize);
        let linear_num_v_heads = gguf
            .meta_u32(&format!("{prefix}.linear_num_value_heads"))
            .map(|v| v as usize);
        let linear_conv_kernel_dim = gguf
            .meta_u32(&format!("{prefix}.linear_conv_kernel_dim"))
            .map(|v| v as usize);

        // Gemma 3n: per-layer sliding window boolean pattern
        let sliding_window_pattern =
            gguf.meta(&format!("{prefix}.attention.sliding_window_pattern"))
                .and_then(|v| match v {
                    crate::gguf::MetaValue::Array(arr) => {
                        let mut out = Vec::with_capacity(arr.len());
                        for item in arr {
                            match item {
                                crate::gguf::MetaValue::Bool(b) => out.push(*b),
                                _ => return None,
                            }
                        }
                        Some(out)
                    }
                    _ => None,
                });

        // Gemma 3n: per-layer activation sparsity scale (f32 array)
        let activation_sparsity_scale = gguf
            .meta(&format!("{prefix}.activation_sparsity_scale"))
            .and_then(|v| match v {
                crate::gguf::MetaValue::Array(arr) => {
                    let mut out = Vec::with_capacity(arr.len());
                    for item in arr {
                        match item {
                            crate::gguf::MetaValue::F32(f) => out.push(*f),
                            _ => return None,
                        }
                    }
                    Some(out)
                }
                _ => None,
            });

        // Gemma 3n: shared KV cache layer count
        let shared_kv_layers = gguf
            .meta_u32(&format!("{prefix}.attention.shared_kv_layers"))
            .map(|v| v as usize);

        // Gemma 3n: per-layer input embedding dimension
        let per_layer_input_embedding_dim = gguf
            .meta_u32(&format!("{prefix}.embedding_length_per_layer_input"))
            .map(|v| v as usize);

        // Gemma 3n: AltUp mechanism
        let altup_num_inputs = gguf
            .meta_u32(&format!("{prefix}.altup.num_inputs"))
            .map(|v| v as usize);
        let altup_active_idx = gguf
            .meta_u32(&format!("{prefix}.altup.active_idx"))
            .map(|v| v as usize);

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
            full_attention_interval,
            linear_num_kv_heads,
            linear_qk_head_dim,
            linear_kv_head_dim,
            linear_num_v_heads,
            linear_conv_kernel_dim,
            sliding_window_pattern,
            activation_sparsity_scale,
            shared_kv_layers,
            per_layer_input_embedding_dim,
            altup_num_inputs,
            altup_active_idx,
        })
    }

    /// Returns true if this is a hybrid DeltaNet model (Qwen3.5).
    pub const fn is_hybrid(&self) -> bool {
        self.full_attention_interval.is_some()
    }

    /// Returns true if layer `i` is a DeltaNet (linear attention) layer.
    /// Full attention layers are at indices where `(i + 1) % interval == 0`.
    pub const fn is_deltanet_layer(&self, i: usize) -> bool {
        match self.full_attention_interval {
            Some(interval) => !(i + 1).is_multiple_of(interval),
            None => false,
        }
    }

    /// Returns true if the model uses NEOX (half rotation) RoPE convention.
    /// Delegates to `ModelArch::use_neox_rope`.
    pub const fn use_neox_rope(&self) -> bool {
        self.arch.use_neox_rope()
    }

    /// Apply the FFN gate activation for this architecture at a given layer.
    /// - Gemma 2: GELU (tanh approximation) uniformly.
    /// - Gemma 3n: GELU uniformly (llama.cpp `ggml_gelu` matches HF exact GELU
    ///   for Gemma 3n). Per-layer sparsity masking is applied **before** this
    ///   step via `apply_ffn_sparsity`.
    /// - All others: SiLU (SwiGLU).
    #[inline]
    pub fn apply_ffn_act(&self, _layer_idx: usize, x: f32) -> f32 {
        match self.arch {
            ModelArch::Gemma2 | ModelArch::Gemma3n => gelu_approx(x),
            _ => silu(x),
        }
    }

    /// Apply Gemma 3n activation sparsity (gaussian_topk) in-place to
    /// `gate_buf` for sparse layers. No-op for non-Gemma-3n or layers where
    /// `activation_sparsity_scale[layer_idx]` is not finite (dense).
    ///
    /// Semantics (llama.cpp `gaussian_topk`):
    ///   mean = mean(gate_buf)
    ///   std  = sqrt(sum((x - mean)^2) / (n - 1))   (unbiased std)
    ///   cutoff = mean + scale * std
    ///   gate_buf[i] = max(0, gate_buf[i] - cutoff)  (ReLU shift)
    ///
    /// For scale = 1.6448 (~= Φ⁻¹(0.95)), retains the top ~5% of values.
    pub fn apply_ffn_sparsity(&self, layer_idx: usize, gate_buf: &mut [f32]) {
        if self.arch != ModelArch::Gemma3n {
            return;
        }
        let scale = match self
            .activation_sparsity_scale
            .as_ref()
            .and_then(|arr| arr.get(layer_idx))
        {
            Some(s) if s.is_finite() => *s,
            _ => return,
        };
        let n = gate_buf.len();
        if n < 2 {
            return;
        }
        // Compute mean
        let mut sum = 0.0f64;
        for &x in gate_buf.iter() {
            sum += x as f64;
        }
        let mean = (sum / n as f64) as f32;
        // Compute unbiased variance: sum((x - mean)^2) / (n - 1)
        let mut sq_sum = 0.0f64;
        for &x in gate_buf.iter() {
            let d = (x - mean) as f64;
            sq_sum += d * d;
        }
        let std = (sq_sum / (n - 1) as f64).sqrt() as f32;
        let cutoff = mean + scale * std;
        // ReLU shift in-place
        for x in gate_buf.iter_mut() {
            *x = (*x - cutoff).max(0.0);
        }
    }

    /// Number of "root" layers with unique KV cache. Later layers reuse KV
    /// from these roots (Gemma 3n shared-KV mechanism).
    ///
    /// For non-Gemma3n architectures, returns `num_layers` (every layer has
    /// its own KV cache). For Gemma 3n, returns `num_layers - shared_kv_layers`
    /// per the metadata (E2B: 30 - 10 = 20, E4B: 35 - 15 = 20). Matches
    /// llama.cpp `hparams.n_layer_kv_from_start`.
    pub fn kv_from_start_layers(&self) -> usize {
        match (self.arch, self.shared_kv_layers) {
            (ModelArch::Gemma3n, Some(shared)) if shared < self.num_layers => {
                self.num_layers - shared
            }
            _ => self.num_layers,
        }
    }

    /// Return the KV-cache "source layer" whose K/V should be read/written for
    /// layer `i`. For most architectures this is `i` itself. For Gemma 3n
    /// layers at index `i >= kv_from_start_layers()`:
    ///   - SWA layer → source = `kv_from_start_layers() - 2`
    ///   - Full attention → source = `kv_from_start_layers() - 1`
    ///
    /// This matches the llama.cpp `layer_reuse_cb` for `LLM_ARCH_GEMMA3N`.
    pub fn kv_source_layer(&self, i: usize) -> usize {
        if self.arch != ModelArch::Gemma3n {
            return i;
        }
        let root = self.kv_from_start_layers();
        if i < root {
            return i;
        }
        // For layers >= root, redirect to the last "own KV" layer of matching type.
        let is_swa = self.sliding_window_for_layer(i).is_some();
        let offset = if is_swa { 2 } else { 1 };
        root.saturating_sub(offset)
    }

    /// Build the layer→KV-source mapping for the whole model. Length equals
    /// `num_layers`. `map[i] == i` means layer i owns its KV cache; `map[i] != i`
    /// means layer i redirects reads and skips writes.
    pub fn build_kv_layer_map(&self) -> Vec<usize> {
        (0..self.num_layers).map(|i| self.kv_source_layer(i)).collect()
    }

    /// Effective sliding window for layer `i`.
    /// - Gemma-2: even layers use sliding_window, odd layers use full attention.
    /// - Gemma 3n: per-layer boolean pattern from
    ///   `attention.sliding_window_pattern` (true = SWA, false = full).
    /// - Others: uniform sliding_window across all layers.
    pub fn sliding_window_for_layer(&self, i: usize) -> Option<usize> {
        match self.arch {
            ModelArch::Gemma2 => {
                if i.is_multiple_of(2) {
                    self.sliding_window
                } else {
                    None
                }
            }
            ModelArch::Gemma3n => {
                if let Some(pattern) = self.sliding_window_pattern.as_ref() {
                    if pattern.get(i).copied().unwrap_or(false) {
                        self.sliding_window
                    } else {
                        None
                    }
                } else {
                    self.sliding_window
                }
            }
            _ => self.sliding_window,
        }
    }

    /// Llama-3 8B default config.
    #[must_use]
    pub const fn llama3_8b() -> Self {
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
            full_attention_interval: None,
            linear_num_kv_heads: None,
            linear_qk_head_dim: None,
            linear_kv_head_dim: None,
            linear_num_v_heads: None,
            linear_conv_kernel_dim: None,
            sliding_window_pattern: None,
            activation_sparsity_scale: None,
            shared_kv_layers: None,
            per_layer_input_embedding_dim: None,
            altup_num_inputs: None,
            altup_active_idx: None,
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
    /// Layer → KV-source layer mapping (Gemma 3n shared-KV support). For most
    /// architectures this is the identity `[0, 1, ..., num_layers-1]`. For
    /// Gemma 3n, later layers redirect reads and skip writes to earlier layers.
    /// `map[i] == i` means layer i owns its KV cache; `map[i] != i` means
    /// layer i is a shared read from `map[i]`.
    kv_layer_map: Vec<usize>,
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
            kv_layer_map: (0..num_layers).collect(),
        }
    }

    /// Install a custom layer→KV-source mapping. Must be called before any
    /// `append`. `map.len()` must equal `num_layers`.
    fn set_layer_map(&mut self, map: Vec<usize>) {
        assert_eq!(map.len(), self._num_layers, "kv_layer_map length mismatch");
        self.kv_layer_map = map;
    }

    #[inline]
    const fn offset(&self, layer: usize, pos: usize) -> usize {
        (layer * self.max_seq_len + pos) * self.kv_dim
    }

    fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        // Skip writes for shared layers (Gemma 3n): the target KV cache was
        // already populated by the "source" layer earlier in the forward pass.
        if self.kv_layer_map[layer] != layer {
            return;
        }
        let off = self.offset(layer, self.seq_len);
        self.keys[off..off + self.kv_dim].copy_from_slice(k);
        self.values[off..off + self.kv_dim].copy_from_slice(v);
    }

    /// Call once after all layers have appended for a given position.
    const fn advance(&mut self) {
        self.seq_len += 1;
    }

    const fn seq_len(&self) -> usize {
        self.seq_len
    }

    #[inline]
    fn key_at(&self, layer: usize, pos: usize) -> &[f32] {
        // Redirect via layer map for shared-KV layers (Gemma 3n).
        let src_layer = self.kv_layer_map[layer];
        let off = self.offset(src_layer, pos);
        &self.keys[off..off + self.kv_dim]
    }

    #[inline]
    fn value_at(&self, layer: usize, pos: usize) -> &[f32] {
        let src_layer = self.kv_layer_map[layer];
        let off = self.offset(src_layer, pos);
        &self.values[off..off + self.kv_dim]
    }

    /// Rollback KV cache to a previous position (for speculative decoding).
    const fn rollback_to(&mut self, pos: usize) {
        self.seq_len = pos;
    }

    const fn clear(&mut self) {
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

    const fn is_full(&self) -> bool {
        self.used >= PAGE_SIZE
    }
}

/// Paged KV cache for a single sequence.
/// Pages are allocated on demand (no upfront max_seq_len allocation).
#[allow(dead_code)]
struct PagedKvCache {
    pages: Vec<Vec<KvPage>>, // pages[layer][page_idx]
    num_layers: usize,
    kv_dim: usize,
    seq_len: usize,
}

#[allow(dead_code)]
impl PagedKvCache {
    fn new(num_layers: usize, kv_dim: usize) -> Self {
        let pages = (0..num_layers).map(|_| Vec::new()).collect();
        Self {
            pages,
            num_layers,
            kv_dim,
            seq_len: 0,
        }
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

    const fn advance(&mut self) {
        self.seq_len += 1;
    }

    const fn seq_len(&self) -> usize {
        self.seq_len
    }

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
            let needed = if pos == 0 {
                0
            } else {
                (pos - 1) / PAGE_SIZE + 1
            };
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
        self.pages.iter().map(std::vec::Vec::len).sum()
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

impl Default for BatchScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchScheduler {
    pub const fn new() -> Self {
        Self {
            requests: Vec::new(),
            next_id: 0,
        }
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

/// Per-head RMSNorm without weight (in place). Same as `apply_qk_norm` but
/// with an implicit identity weight vector. Used for V normalization in
/// Gemma 3n.
fn apply_head_rms_norm_identity(buf: &mut [f32], head_dim: usize, eps: f32) {
    let num_heads = buf.len() / head_dim;
    for h in 0..num_heads {
        let start = h * head_dim;
        let slice = &mut buf[start..start + head_dim];
        let mut ss = 0.0f64;
        for &v in slice.iter() {
            ss += (v as f64) * (v as f64);
        }
        let mean = (ss / head_dim as f64) as f32;
        let scale = 1.0f32 / (mean + eps).sqrt();
        for v in slice.iter_mut() {
            *v *= scale;
        }
    }
}

/// F32 dense matrix-vector product: `out[i] = sum_j w[i * cols + j] * x[j]`.
/// Row-major storage; `w.len()` must equal `rows * cols`.
fn mat_vec_f32(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    for i in 0..rows {
        let row = &w[i * cols..(i + 1) * cols];
        let mut sum = 0.0f32;
        for j in 0..cols {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}

/// L2 magnitude (Frobenius norm): sqrt(sum(x^2)). Used by Gemma 3n AltUp
/// magnitude-preserving projection.
fn l2_magnitude(x: &[f32]) -> f32 {
    let mut ss = 0.0f64;
    for &v in x {
        ss += (v as f64) * (v as f64);
    }
    (ss as f32).sqrt()
}

/// Qwen 3 QK-Norm: apply per-head RMSNorm to Q or K buffer in-place.
/// `buf` shape: [num_heads * head_dim] (Q) or [num_kv_heads * head_dim] (K).
/// `weight` shape: [head_dim] (broadcast across heads).
fn apply_qk_norm(buf: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) {
    let num_heads = buf.len() / head_dim;
    for h in 0..num_heads {
        let start = h * head_dim;
        let slice = &mut buf[start..start + head_dim];
        let mut ss = 0.0f64;
        for &v in slice.iter() {
            ss += (v as f64) * (v as f64);
        }
        let mean = (ss / head_dim as f64) as f32;
        let scale = 1.0f32 / (mean + eps).sqrt();
        for (i, w) in weight.iter().enumerate() {
            slice[i] = slice[i] * scale * w;
        }
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

/// Apply RoPE with per-dimension frequency factors (Llama-3.1/3.2 NTK-aware context extension).
/// `freq_factors` has `head_dim / 2` entries — base frequency is divided by each factor.
/// freq[i] = (1/theta^(2i/d)) / freq_factors[i]
/// (llama.cpp convention: higher factor = slower rotation = longer effective context)
fn apply_rope_scaled(
    vec: &mut [f32],
    position: usize,
    head_dim: usize,
    theta: f32,
    freq_factors: &[f32],
) {
    for i in (0..head_dim).step_by(2) {
        let freq_idx = i / 2;
        let base_freq = 1.0 / theta.powf(i as f32 / head_dim as f32);
        let factor = if freq_idx < freq_factors.len() {
            freq_factors[freq_idx]
        } else {
            1.0
        };
        let freq = base_freq / factor;
        let angle = position as f32 * freq;
        let (sin_val, cos_val) = angle.sin_cos();
        let x0 = vec[i];
        let x1 = vec[i + 1];
        vec[i] = x0 * cos_val - x1 * sin_val;
        vec[i + 1] = x0 * sin_val + x1 * cos_val;
    }
}

/// NEOX RoPE (GPT-NeoX / HF convention): rotate pairs (i, i + head_dim/2).
/// Used by Qwen 2/3 and Gemma 2. Q/K weights in GGUF are stored in HF layout
/// (not permuted like Llama family), so we apply the half rotation directly.
fn apply_rope_neox(vec: &mut [f32], position: usize, head_dim: usize, theta: f32) {
    let half = head_dim / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
        let angle = position as f32 * freq;
        let (sin_val, cos_val) = angle.sin_cos();
        let x0 = vec[i];
        let x1 = vec[i + half];
        vec[i] = x0 * cos_val - x1 * sin_val;
        vec[i + half] = x0 * sin_val + x1 * cos_val;
    }
}

/// NEOX RoPE with per-dimension frequency factors (for NTK-aware scaling).
fn apply_rope_scaled_neox(
    vec: &mut [f32],
    position: usize,
    head_dim: usize,
    theta: f32,
    freq_factors: &[f32],
) {
    let half = head_dim / 2;
    for i in 0..half {
        let base_freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
        let factor = if i < freq_factors.len() {
            freq_factors[i]
        } else {
            1.0
        };
        let freq = base_freq / factor;
        let angle = position as f32 * freq;
        let (sin_val, cos_val) = angle.sin_cos();
        let x0 = vec[i];
        let x1 = vec[i + half];
        vec[i] = x0 * cos_val - x1 * sin_val;
        vec[i + half] = x0 * sin_val + x1 * cos_val;
    }
}

/// Apply RoPE: dispatches to NORM (paired) vs NEOX (half rotation) based on `neox`,
/// then to scaled vs scalar based on `freq_scales`.
#[inline]
fn apply_rope_auto(
    vec: &mut [f32],
    position: usize,
    head_dim: usize,
    theta: f32,
    freq_scales: Option<&[f32]>,
    neox: bool,
) {
    match (neox, freq_scales) {
        (false, Some(s)) => apply_rope_scaled(vec, position, head_dim, theta, s),
        (false, None) => apply_rope(vec, position, head_dim, theta),
        (true, Some(s)) => apply_rope_scaled_neox(vec, position, head_dim, theta, s),
        (true, None) => apply_rope_neox(vec, position, head_dim, theta),
    }
}

// ─── GQA Attention (supports SWA + logit softcapping) ──────────────────────

/// Compute GQA attention into `attn_out`.
/// Supports Mistral Sliding Window and Gemma-2 logit softcapping.
/// When `attention_scale` is Some(x), uses x as the score-scaling factor
/// instead of the default 1/sqrt(head_dim). Gemma 3n uses 1.0.
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
    attention_scale: Option<f32>,
    attn_out: &mut [f32],
) {
    let seq_len = pos + 1;
    let heads_per_kv = num_heads / num_kv_heads;
    let inv_sqrt_d = attention_scale.unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());

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

// ─── FFN activations ────────────────────────────────────────────────────────

/// SiLU (Swish) activation: `x / (1 + exp(-x))`. Used by Llama, Mistral,
/// Qwen 2/3 for SwiGLU-style FFN.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU tanh approximation: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
/// Used by Gemma 2 (HF `gelu_pytorch_tanh` / llama.cpp `LLM_FFN_GELU`).
#[inline]
fn gelu_approx(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
}

// ─── Llama-3 model ──────────────────────────────────────────────────────────

/// Weight reference pointing into GGUF mmap'd data.
struct WeightRef<'a> {
    data: &'a [u8],
    qtype: GgmlType,
    rows: usize,
    cols: usize,
}

impl WeightRef<'_> {
    fn matvec(&self, input: &[f32], output: &mut [f32]) {
        quantized_matvec(input, self.data, self.qtype, self.rows, self.cols, output);
    }

    /// Matvec with pre-quantized Q8_K input (avoids redundant quantization).
    fn matvec_preq(&self, q8_blocks: &[BlockQ8K], output: &mut [f32]) {
        quantized_matvec_preq(
            self.data, self.qtype, self.rows, self.cols, q8_blocks, output,
        );
    }

    /// Dequantize all weights to f32 (row-major, rows × cols).
    fn dequantize_all(&self, rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        let mut row_buf = vec![0.0f32; cols];
        let elems_per_block = self.qtype.elements_per_block();
        let bytes_per_block = self.qtype.block_bytes();
        let blocks_per_row = cols / elems_per_block;
        let row_bytes = blocks_per_row * bytes_per_block;
        for r in 0..rows {
            let row_data = &self.data[r * row_bytes..(r + 1) * row_bytes];
            crate::gguf::dequantize_weight_row(row_data, self.qtype, &mut row_buf);
            out[r * cols..(r + 1) * cols].copy_from_slice(&row_buf);
        }
        out
    }
}

/// Layer weight references (zero-copy from GGUF).
struct LayerWeights<'a> {
    attn_norm: Vec<f32>,
    q_proj: WeightRef<'a>,
    k_proj: WeightRef<'a>,
    v_proj: WeightRef<'a>,
    o_proj: WeightRef<'a>,
    /// Qwen 2/2.5 attention Q projection bias (None for Llama/Mistral/Gemma/Qwen 3).
    q_bias: Option<Vec<f32>>,
    /// Qwen 2/2.5 attention K projection bias.
    k_bias: Option<Vec<f32>>,
    /// Qwen 2/2.5 attention V projection bias.
    v_bias: Option<Vec<f32>>,
    /// Qwen 3 per-head RMSNorm applied to Q before RoPE (None for other arch).
    q_norm: Option<Vec<f32>>,
    /// Qwen 3 per-head RMSNorm applied to K before RoPE.
    k_norm: Option<Vec<f32>>,
    /// Gemma-2 post-attention RMSNorm (before residual add).
    post_attn_norm: Option<Vec<f32>>,
    /// Gemma-2 post-FFN RMSNorm (before residual add).
    post_ffn_norm: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
    gate_proj: WeightRef<'a>,
    up_proj: WeightRef<'a>,
    down_proj: WeightRef<'a>,
    // ── Gemma 3n per-layer tensors (None for other architectures) ──────────
    /// Gemma 3n: extra RMSNorm applied at the end of layer (post_norm).
    post_norm: Option<Vec<f32>>,
    /// Gemma 3n: per-layer input embedding gate projection.
    /// Shape [hidden_dim, per_layer_input_embedding_dim].
    inp_gate: Option<WeightRef<'a>>,
    /// Gemma 3n: extra projection connecting the per-layer embedding branch.
    /// Shape [per_layer_input_embedding_dim, hidden_dim].
    proj: Option<WeightRef<'a>>,
    /// Gemma 3n Laurel left projection [hidden_dim, laurel_rank=64].
    laurel_l: Option<Vec<f32>>,
    /// Gemma 3n Laurel right projection [laurel_rank=64, hidden_dim].
    laurel_r: Option<Vec<f32>>,
    /// Gemma 3n Laurel post-branch RMSNorm [hidden_dim].
    laurel_post_norm: Option<Vec<f32>>,
    /// Gemma 3n AltUp router [hidden_dim, altup_num_inputs].
    altup_router: Option<Vec<f32>>,
    /// Gemma 3n AltUp router pre-RMSNorm [hidden_dim].
    altup_router_norm: Option<Vec<f32>>,
    /// Gemma 3n AltUp predict coefficient matrix [altup_num_inputs, altup_num_inputs^2].
    altup_predict_coef: Option<Vec<f32>>,
    /// Gemma 3n AltUp correct coefficient matrix [altup_num_inputs, altup_num_inputs].
    altup_correct_coef: Option<Vec<f32>>,
    /// Gemma 3n AltUp correction scale [hidden_dim].
    altup_correct_scale: Option<Vec<f32>>,
}

// ─── Layerwise Mixed Precision ──────────────────────────────────────────────

/// Quantization strategy for a single layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    pub const fn full_ternary() -> Self {
        Self {
            attention_mode: LayerQuantMode::Ternary,
            ffn_mode: LayerQuantMode::Ternary,
        }
    }

    /// Aggressive: ternary attention, binary+sparse FFN (for 10GB target).
    pub const fn aggressive_compression(n_keep: usize) -> Self {
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

/// Sparse ternary layer weights (N:M structured sparsity + block-packed).
struct SparseTernaryLayerWeights {
    attn_norm: Vec<f32>,
    q_proj: SparseTernaryMatrix,
    k_proj: SparseTernaryMatrix,
    v_proj: SparseTernaryMatrix,
    o_proj: SparseTernaryMatrix,
    ffn_norm: Vec<f32>,
    gate_proj: SparseTernaryMatrix,
    up_proj: SparseTernaryMatrix,
    down_proj: SparseTernaryMatrix,
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
    /// Per-dimension RoPE frequencies (Llama-3.1/3.2). None = use rope_theta scalar.
    pub rope_freqs: Option<Vec<f32>>,
    ternary_layers: Option<Vec<TernaryLayerWeights>>,
    ternary_output_proj: Option<TernaryMatrix>,
    sparse_ternary_layers: Option<Vec<SparseTernaryLayerWeights>>,
    sparse_ternary_output: Option<SparseTernaryMatrix>,
    // ── Gemma 3n global weights (None for other architectures) ─────────────
    /// Gemma 3n per-layer input embedding table
    /// [num_layers * per_layer_input_embedding_dim, vocab_size] Q5_1.
    /// Kept as raw bytes for memory efficiency; slice-dequantize per token
    /// via [`Self::per_layer_embedding_for_token`].
    per_layer_token_embd_raw: Option<&'a [u8]>,
    /// Gemma 3n per-layer embedding projection
    /// [hidden_dim, num_layers * per_layer_input_embedding_dim].
    per_layer_model_proj: Option<WeightRef<'a>>,
    /// Gemma 3n per-layer projection RMSNorm weight
    /// [per_layer_input_embedding_dim].
    per_layer_proj_norm: Option<Vec<f32>>,
    /// Gemma 3n AltUp projection [hidden_dim, hidden_dim, altup_num_inputs - 1]
    /// (dequantized F16 → f32).
    altup_proj: Option<Vec<f32>>,
    /// Gemma 3n AltUp un-embed projection (mirror shape of altup_proj).
    altup_unembd_proj: Option<Vec<f32>>,
}

impl<'a> Llama3Model<'a> {
    /// Load model from a parsed GGUF file.
    pub fn from_gguf(gguf: &'a GgufFile<'a>) -> Option<Self> {
        let config = Llama3Config::from_gguf(gguf)?;

        // Embedding (dequantized to f32 once)
        let embedding = gguf.tensor_to_f32("token_embd.weight")?;

        // Output norm
        let output_norm = gguf.tensor_to_f32("output_norm.weight")?;

        // Output projection (fallback to tied embedding if output.weight absent)
        let output_proj =
            load_weight_ref(gguf, "output.weight", config.vocab_size, config.hidden_dim).or_else(
                || {
                    load_weight_ref(
                        gguf,
                        "token_embd.weight",
                        config.vocab_size,
                        config.hidden_dim,
                    )
                },
            )?;

        // Layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer = load_layer_weights(gguf, i, &config)?;
            layers.push(layer);
        }

        let kv_dim = config.num_kv_heads * config.head_dim;
        let mut kv_cache = KvCache::new(config.num_layers, config.max_seq_len, kv_dim);
        // Gemma 3n shared-KV layer mapping (no-op for other architectures).
        kv_cache.set_layer_map(config.build_kv_layer_map());

        // RoPE frequency scaling tensor (Llama-3.1/3.2 NTK-aware context extension)
        // Values are scaling factors: actual_freq[i] = base_freq[i] * scale[i]
        // scale=1.0 means no change, scale>1 means faster rotation (extended context)
        let rope_freqs: Option<Vec<f32>> =
            gguf.tensor_to_f32("rope_freqs.weight").and_then(|scales| {
                let half_dim = config.head_dim / 2;
                if scales.len() != half_dim {
                    return None;
                }
                // Only use if any scale differs from 1.0 (i.e., non-trivial scaling)
                let needs_scaling = scales.iter().any(|&s| (s - 1.0).abs() > 0.01);
                if needs_scaling {
                    Some(scales)
                } else {
                    None
                }
            });

        // Gemma 3n global weights (per-layer input embedding + AltUp projections).
        // All fields are None for non-Gemma3n architectures.
        let (
            per_layer_token_embd_raw,
            per_layer_model_proj,
            per_layer_proj_norm,
            altup_proj,
            altup_unembd_proj,
        ) = if config.arch == ModelArch::Gemma3n {
            let per_layer_token_embd_raw = gguf.tensor_data("per_layer_token_embd.weight");
            // per_layer_model_proj.weight ggml shape {n_embd, n_layer * n_embd_altup}
            // (ne0=n_embd=in, ne1=n_layer*n_embd_altup=out)
            // → rows=n_layer*n_embd_altup, cols=n_embd.
            let per_layer_model_proj = config.per_layer_input_embedding_dim.and_then(|dim| {
                load_weight_ref(
                    gguf,
                    "per_layer_model_proj.weight",
                    config.num_layers * dim,  // rows = out_dim
                    config.hidden_dim,        // cols = in_dim
                )
            });
            let per_layer_proj_norm = gguf.tensor_to_f32("per_layer_proj_norm.weight");
            let altup_proj = gguf.tensor_to_f32("altup_proj.weight");
            let altup_unembd_proj = gguf.tensor_to_f32("altup_unembd_proj.weight");
            (
                per_layer_token_embd_raw,
                per_layer_model_proj,
                per_layer_proj_norm,
                altup_proj,
                altup_unembd_proj,
            )
        } else {
            (None, None, None, None, None)
        };

        Some(Self {
            config,
            embedding,
            layers,
            output_norm,
            output_proj,
            kv_cache,
            rope_freqs,
            ternary_layers: None,
            ternary_output_proj: None,
            sparse_ternary_layers: None,
            sparse_ternary_output: None,
            per_layer_token_embd_raw,
            per_layer_model_proj,
            per_layer_proj_norm,
            altup_proj,
            altup_unembd_proj,
        })
    }

    /// Gemma 3n: dequantize the per-layer input embedding slice for a single
    /// token. Returns a `Vec<f32>` of length `num_layers * per_layer_dim`.
    /// Returns `None` for non-Gemma3n models or when the per-layer token
    /// embedding table is absent.
    ///
    /// Memory-efficient: dequantizes only the requested token's slice (~30 KB)
    /// rather than materializing the full ~8 GB f32 table upfront.
    pub fn per_layer_embedding_for_token(&self, token_id: u32) -> Option<Vec<f32>> {
        let raw = self.per_layer_token_embd_raw?;
        let per_layer_dim = self.config.per_layer_input_embedding_dim?;
        let elements_per_token = self.config.num_layers * per_layer_dim;
        // Q5_1: block size = 32 elements, block bytes = 24.
        let qk = crate::gguf::GgmlType::Q5_1.elements_per_block();
        let block_bytes = crate::gguf::GgmlType::Q5_1.block_bytes();
        if !elements_per_token.is_multiple_of(qk) {
            return None;
        }
        let blocks_per_token = elements_per_token / qk;
        let start = (token_id as usize) * blocks_per_token * block_bytes;
        let end = start + blocks_per_token * block_bytes;
        if end > raw.len() {
            return None;
        }
        let mut out = vec![0.0f32; elements_per_token];
        crate::gguf::dequantize_q5_1(&raw[start..end], &mut out);
        Some(out)
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
                q_proj: ternarize_weight(
                    &layer.q_proj,
                    c.hidden_dim,
                    c.hidden_dim,
                    threshold_ratio,
                ),
                k_proj: ternarize_weight(&layer.k_proj, kv_dim, c.hidden_dim, threshold_ratio),
                v_proj: ternarize_weight(&layer.v_proj, kv_dim, c.hidden_dim, threshold_ratio),
                o_proj: ternarize_weight(
                    &layer.o_proj,
                    c.hidden_dim,
                    c.hidden_dim,
                    threshold_ratio,
                ),
                ffn_norm: layer.ffn_norm.clone(),
                gate_proj: ternarize_weight(
                    &layer.gate_proj,
                    c.intermediate_dim,
                    c.hidden_dim,
                    threshold_ratio,
                ),
                up_proj: ternarize_weight(
                    &layer.up_proj,
                    c.intermediate_dim,
                    c.hidden_dim,
                    threshold_ratio,
                ),
                down_proj: ternarize_weight(
                    &layer.down_proj,
                    c.hidden_dim,
                    c.intermediate_dim,
                    threshold_ratio,
                ),
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
        // Gemma 3n: use dedicated forward path (AltUp + Laurel + per-layer
        // input embedding + shared KV are fundamentally different from the
        // standard single-stream flow).
        if self.config.arch == ModelArch::Gemma3n {
            return self.forward_gemma3n(token_id);
        }
        let c = &self.config;
        let pos = self.kv_cache.seq_len();
        let rope_freqs_ref = self.rope_freqs.as_deref();

        // Embedding lookup
        let emb_start = token_id as usize * c.hidden_dim;
        let mut hidden: Vec<f32> = self.embedding[emb_start..emb_start + c.hidden_dim].to_vec();
        // Gemma-2: scale embeddings by sqrt(hidden_dim) (no-op for others).
        if c.arch == ModelArch::Gemma2 {
            let scale = (c.hidden_dim as f32).sqrt();
            for h in &mut hidden {
                *h *= scale;
            }
        }

        // Reusable buffers
        let mut norm_buf = vec![0.0f32; c.hidden_dim];
        let kv_dim = c.num_kv_heads * c.head_dim;
        let mut q_buf = vec![0.0f32; c.num_heads * c.head_dim];
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
            // Qwen 2/2.5 bias (no-op for Llama/Mistral/Gemma/Qwen 3)
            if let Some(b) = &layer.q_bias {
                for (q, bi) in q_buf.iter_mut().zip(b.iter()) {
                    *q += bi;
                }
            }
            if let Some(b) = &layer.k_bias {
                for (k, bi) in k_buf.iter_mut().zip(b.iter()) {
                    *k += bi;
                }
            }
            if let Some(b) = &layer.v_bias {
                for (v, bi) in v_buf.iter_mut().zip(b.iter()) {
                    *v += bi;
                }
            }
            // Qwen 3 QK-Norm (per-head RMSNorm on Q, K before RoPE; no-op for others)
            if let Some(w) = &layer.q_norm {
                apply_qk_norm(&mut q_buf, w, c.head_dim, c.norm_eps);
            }
            if let Some(w) = &layer.k_norm {
                apply_qk_norm(&mut k_buf, w, c.head_dim, c.norm_eps);
            }

            // Apply RoPE
            for h in 0..c.num_heads {
                let start = h * c.head_dim;
                apply_rope_auto(
                    &mut q_buf[start..start + c.head_dim],
                    pos,
                    c.head_dim,
                    c.rope_theta,
                    rope_freqs_ref,
                    c.use_neox_rope(),
                );
            }
            for h in 0..c.num_kv_heads {
                let start = h * c.head_dim;
                apply_rope_auto(
                    &mut k_buf[start..start + c.head_dim],
                    pos,
                    c.head_dim,
                    c.rope_theta,
                    rope_freqs_ref,
                    c.use_neox_rope(),
                );
            }

            // Store K, V in cache
            self.kv_cache.append(layer_idx, &k_buf, &v_buf);

            // GQA attention (supports SWA + logit softcapping)
            gqa_attention(
                &q_buf,
                &self.kv_cache,
                layer_idx,
                pos,
                c.num_heads,
                c.num_kv_heads,
                c.head_dim,
                c.sliding_window_for_layer(layer_idx),
                c.attn_logit_softcap,
                if c.arch == ModelArch::Gemma3n { Some(1.0) } else { None },
                &mut attn_out,
            );

            // Output projection
            layer.o_proj.matvec(&attn_out, &mut o_buf);

            // Gemma-2 post-attention RMSNorm (before residual add; no-op for others)
            if let Some(w) = &layer.post_attn_norm {
                let mut tmp = vec![0.0f32; c.hidden_dim];
                rms_norm(&o_buf, w, c.norm_eps, &mut tmp);
                o_buf.copy_from_slice(&tmp);
            }

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

            c.apply_ffn_sparsity(layer_idx, &mut gate_buf);
            for i in 0..c.intermediate_dim {
                gate_buf[i] = c.apply_ffn_act(layer_idx, gate_buf[i]) * up_buf[i];
            }

            layer.down_proj.matvec(&gate_buf, &mut down_buf);

            // Gemma-2 post-FFN RMSNorm (before residual add; no-op for others)
            if let Some(w) = &layer.post_ffn_norm {
                let mut tmp = vec![0.0f32; c.hidden_dim];
                rms_norm(&down_buf, w, c.norm_eps, &mut tmp);
                down_buf.copy_from_slice(&tmp);
            }

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
        // output_proj points to output.weight or token_embd.weight (tied)
        self.output_proj.matvec(&norm_buf, &mut logits);

        // Gemma-2: final logit softcapping
        if let Some(cap) = c.final_logit_softcap {
            for l in &mut logits {
                *l = cap * (*l / cap).tanh();
            }
        }

        logits
    }

    /// Gemma 3n forward pass. Mirrors llama.cpp
    /// `llama_model_gemma3n::graph::build_arch_graph`.
    ///
    /// Data shape convention (single-token autoregressive):
    ///   * altup streams:      `Vec<Vec<f32>>` of shape `[n_altup][hidden_dim]`
    ///   * inp_per_layer:      `Vec<Vec<f32>>` of shape `[n_layer][n_embd_altup]`
    ///
    /// # Panics
    /// Panics if the model was not loaded with Gemma 3n architecture (missing
    /// required per-layer / global tensors).
    fn forward_gemma3n(&mut self, token_id: u32) -> Vec<f32> {
        let c = self.config.clone();
        let hidden_dim = c.hidden_dim;
        let n_altup = c.altup_num_inputs.expect("Gemma3n: altup_num_inputs");
        let n_embd_altup = c
            .per_layer_input_embedding_dim
            .expect("Gemma3n: per_layer_input_embedding_dim");
        let i_altup_act = c.altup_active_idx.unwrap_or(0);
        let n_layer_sparsity = c
            .activation_sparsity_scale
            .as_ref()
            .map_or(0, |arr| arr.iter().take_while(|s| s.is_finite()).count());
        let pos = self.kv_cache.seq_len();
        let rope_freqs_ref = self.rope_freqs.as_deref();

        // ── Embedding lookup + Gemma-style scale ────────────────────────────
        let emb_start = token_id as usize * hidden_dim;
        let mut inpl: Vec<f32> = self.embedding[emb_start..emb_start + hidden_dim].to_vec();
        let scale = (hidden_dim as f32).sqrt();
        for v in &mut inpl {
            *v *= scale;
        }

        // ── Per-layer input embedding lookup + projection ───────────────────
        // Shape: [n_layer][n_embd_altup]
        let inp_per_layer =
            self.gemma3n_per_layer_inputs(token_id, &inpl, n_embd_altup, c.num_layers);

        // ── Initialize AltUp streams: [n_altup][hidden_dim] ────────────────
        let mut streams: Vec<Vec<f32>> = vec![Vec::new(); n_altup];
        streams[i_altup_act] = inpl.clone();
        // Compute other streams via altup_proj (magnitude-preserving)
        let target_magnitude = l2_magnitude(&inpl);
        let altup_proj = self
            .altup_proj
            .as_ref()
            .expect("Gemma3n: altup_proj missing");
        // altup_proj shape: [n_embd, n_embd, n_altup - 1] in ggml row-major.
        // ggml stores innermost dim first, so the layout is:
        //   for i_altup in 0..n_altup-1:
        //     for row in 0..n_embd:  (== output dim)
        //       for col in 0..n_embd:  (== input dim)
        // We index as: altup_proj[i_altup * n_embd * n_embd + row * n_embd + col]
        let n_embd = hidden_dim;
        for i_altup in 0..(n_altup - 1) {
            let slab_start = i_altup * n_embd * n_embd;
            let slab = &altup_proj[slab_start..slab_start + n_embd * n_embd];
            let mut added = vec![0.0f32; n_embd];
            mat_vec_f32(slab, n_embd, n_embd, &inpl, &mut added);
            // Magnitude-preserve normalization
            let new_mag = l2_magnitude(&added);
            if new_mag > 0.0 {
                let factor = target_magnitude / new_mag;
                for v in &mut added {
                    *v *= factor;
                }
            }
            // Fill the stream after i_altup_act (skip active).
            let dest_idx = if i_altup < i_altup_act {
                i_altup
            } else {
                i_altup + 1
            };
            streams[dest_idx] = added;
        }

        // Reusable buffers
        let mut norm_buf = vec![0.0f32; hidden_dim];
        let kv_dim = c.num_kv_heads * c.head_dim;
        let mut q_buf = vec![0.0f32; c.num_heads * c.head_dim];
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; hidden_dim];
        let mut o_buf = vec![0.0f32; hidden_dim];
        let mut gate_buf = vec![0.0f32; c.intermediate_dim];
        let mut up_buf = vec![0.0f32; c.intermediate_dim];
        let mut down_buf = vec![0.0f32; hidden_dim];

        // ── Per-layer loop ─────────────────────────────────────────────────
        for layer_idx in 0..c.num_layers {
            let layer = &self.layers[layer_idx];

            // ── Altup predict ──────────────────────────────────────────────
            let predictions = self.altup_predict(&streams, layer_idx, n_altup);
            // active_prediction is predictions[i_altup_act]
            let active_prediction = predictions[i_altup_act].clone();
            let mut cur = active_prediction.clone();

            // ── attn_norm ──────────────────────────────────────────────────
            rms_norm(&cur, &layer.attn_norm, c.norm_eps, &mut norm_buf);
            cur.copy_from_slice(&norm_buf);

            // ── Laurel branch (parallel to attention) ──────────────────────
            let laurel_out = self.laurel(&cur, layer_idx);

            // ── Attention (Q, K, V projections + norms + RoPE) ─────────────
            let q8_attn = quantize_row_q8_k(&cur);
            layer.q_proj.matvec_preq(&q8_attn, &mut q_buf);
            // Only compute K, V for "own KV" layers (Gemma 3n shared KV: shared
            // layers reuse a previous layer's cache).
            let owns_kv = self.kv_cache.kv_layer_map[layer_idx] == layer_idx;
            if owns_kv {
                layer.k_proj.matvec_preq(&q8_attn, &mut k_buf);
                layer.v_proj.matvec_preq(&q8_attn, &mut v_buf);
            }

            // Q, K per-head RMSNorm (Gemma 3n uses them like Qwen 3)
            if let Some(w) = &layer.q_norm {
                apply_qk_norm(&mut q_buf, w, c.head_dim, c.norm_eps);
            }
            if owns_kv {
                if let Some(w) = &layer.k_norm {
                    apply_qk_norm(&mut k_buf, w, c.head_dim, c.norm_eps);
                }
                // V RMSNorm without weight (identity gain)
                apply_head_rms_norm_identity(&mut v_buf, c.head_dim, c.norm_eps);
            }

            // Apply RoPE to Q and K
            for h in 0..c.num_heads {
                let start = h * c.head_dim;
                apply_rope_auto(
                    &mut q_buf[start..start + c.head_dim],
                    pos,
                    c.head_dim,
                    c.rope_theta,
                    rope_freqs_ref,
                    c.use_neox_rope(),
                );
            }
            if owns_kv {
                for h in 0..c.num_kv_heads {
                    let start = h * c.head_dim;
                    apply_rope_auto(
                        &mut k_buf[start..start + c.head_dim],
                        pos,
                        c.head_dim,
                        c.rope_theta,
                        rope_freqs_ref,
                        c.use_neox_rope(),
                    );
                }
                // Store K, V in cache (skips write for shared layers internally).
                self.kv_cache.append(layer_idx, &k_buf, &v_buf);
            }

            // GQA attention (uses layer_idx which the cache remaps to source layer)
            gqa_attention(
                &q_buf,
                &self.kv_cache,
                layer_idx,
                pos,
                c.num_heads,
                c.num_kv_heads,
                c.head_dim,
                c.sliding_window_for_layer(layer_idx),
                c.attn_logit_softcap,
                if c.arch == ModelArch::Gemma3n { Some(1.0) } else { None },
                &mut attn_out,
            );

            // Output projection
            let layer = &self.layers[layer_idx];
            layer.o_proj.matvec(&attn_out, &mut o_buf);

            // Post-attention RMSNorm (Gemma-2/3n sandwich)
            if let Some(w) = &layer.post_attn_norm {
                rms_norm(&o_buf, w, c.norm_eps, &mut norm_buf);
                o_buf.copy_from_slice(&norm_buf);
            }

            // cur = attn_output + active_prediction  (gated residual)
            for i in 0..hidden_dim {
                cur[i] = o_buf[i] + active_prediction[i];
            }

            // attn_laurel = (cur + laurel_out) / sqrt(2)
            let inv_sqrt2 = 1.0f32 / 2.0f32.sqrt();
            let mut attn_laurel = vec![0.0f32; hidden_dim];
            for i in 0..hidden_dim {
                attn_laurel[i] = (cur[i] + laurel_out[i]) * inv_sqrt2;
            }

            // ── FFN ────────────────────────────────────────────────────────
            rms_norm(&attn_laurel, &layer.ffn_norm, c.norm_eps, &mut norm_buf);
            let q8_ffn = quantize_row_q8_k(&norm_buf);
            layer.gate_proj.matvec_preq(&q8_ffn, &mut gate_buf);
            layer.up_proj.matvec_preq(&q8_ffn, &mut up_buf);

            // Sparsity: gaussian_topk for first n_layer_sparsity layers
            if layer_idx < n_layer_sparsity {
                c.apply_ffn_sparsity(layer_idx, &mut gate_buf);
            }
            // GELU + gate * up
            for i in 0..c.intermediate_dim {
                gate_buf[i] = gelu_approx(gate_buf[i]) * up_buf[i];
            }
            layer.down_proj.matvec(&gate_buf, &mut down_buf);

            // Post-FFN RMSNorm
            if let Some(w) = &layer.post_ffn_norm {
                rms_norm(&down_buf, w, c.norm_eps, &mut norm_buf);
                down_buf.copy_from_slice(&norm_buf);
            }

            // attn_ffw_laurel_gated = down_buf + attn_laurel
            let mut ffn_out = attn_laurel.clone();
            for i in 0..hidden_dim {
                ffn_out[i] += down_buf[i];
            }

            // ── AltUp correct ──────────────────────────────────────────────
            let mut corrected = self.altup_correct(&predictions, &ffn_out, layer_idx, n_altup);

            // ── Per-layer first_prediction (bottom of layer) ──────────────
            let first_prediction = self.gemma3n_first_prediction(
                &corrected[i_altup_act],
                layer_idx,
                &inp_per_layer[layer_idx],
                n_embd_altup,
            );
            // corrected[1..] += first_prediction  (skip active stream)
            for a in 0..n_altup {
                if a == i_altup_act {
                    continue;
                }
                for i in 0..hidden_dim {
                    corrected[a][i] += first_prediction[i];
                }
            }

            streams = corrected;
        }

        // Advance KV cache position after all layers appended
        self.kv_cache.advance();
        let _ = inp_per_layer; // keep alive across loop

        // ── Merge altup streams back to single via altup_unembd_proj ──────
        let mut cur = streams[i_altup_act].clone();
        let target_magnitude = l2_magnitude(&cur);
        let altup_unembd_proj = self
            .altup_unembd_proj
            .as_ref()
            .expect("Gemma3n: altup_unembd_proj missing");
        // For each non-active stream, project + magnitude-preserve, then add.
        for i_altup in 0..(n_altup - 1) {
            let src_stream_idx = if i_altup < i_altup_act {
                i_altup
            } else {
                i_altup + 1
            };
            let slab_start = i_altup * n_embd * n_embd;
            let slab = &altup_unembd_proj[slab_start..slab_start + n_embd * n_embd];
            let mut unembd = vec![0.0f32; n_embd];
            mat_vec_f32(slab, n_embd, n_embd, &streams[src_stream_idx], &mut unembd);
            let new_mag = l2_magnitude(&unembd);
            if new_mag > 0.0 {
                let factor = target_magnitude / new_mag;
                for v in &mut unembd {
                    *v *= factor;
                }
            }
            for i in 0..hidden_dim {
                cur[i] += unembd[i];
            }
        }
        // Average (divide by n_altup)
        let inv_n_altup = 1.0f32 / n_altup as f32;
        for v in &mut cur {
            *v *= inv_n_altup;
        }

        // Output norm
        rms_norm(&cur, &self.output_norm, c.norm_eps, &mut norm_buf);

        // Output logits
        let mut logits = vec![0.0f32; c.vocab_size];
        self.output_proj.matvec(&norm_buf, &mut logits);

        // Final logit softcap (Gemma family)
        if let Some(cap) = c.final_logit_softcap {
            for l in &mut logits {
                *l = cap * (*l / cap).tanh();
            }
        }
        logits
    }

    /// Compute per-layer input embeddings for a single token (Gemma 3n).
    ///
    /// Returns `Vec<Vec<f32>>` of shape `[n_layer][n_embd_altup]`.
    fn gemma3n_per_layer_inputs(
        &self,
        token_id: u32,
        inpl_scaled: &[f32],
        n_embd_altup: usize,
        n_layer: usize,
    ) -> Vec<Vec<f32>> {
        let c = &self.config;
        // Step 1: Look up per_layer_token_embd row, scale by sqrt(n_embd_altup).
        let raw = self
            .per_layer_embedding_for_token(token_id)
            .expect("Gemma3n: per_layer_token_embd");
        assert_eq!(raw.len(), n_layer * n_embd_altup);
        let tok_scale = (n_embd_altup as f32).sqrt();
        let mut inp_per_layer_lookup: Vec<Vec<f32>> = (0..n_layer)
            .map(|l| {
                raw[l * n_embd_altup..(l + 1) * n_embd_altup]
                    .iter()
                    .map(|v| v * tok_scale)
                    .collect()
            })
            .collect();

        // Step 2: Project inpl_scaled through per_layer_model_proj to get
        // per-layer contribution, then RMSNorm(per_layer_proj_norm), then add.
        let proj = self
            .per_layer_model_proj
            .as_ref()
            .expect("Gemma3n: per_layer_model_proj");
        // proj shape: [hidden_dim, n_layer * n_embd_altup]. matvec computes
        // `out[i] = sum_j W[i, j] * inpl[j]` where W is [rows=n_layer * n_embd_altup, cols=hidden_dim].
        // WeightRef stores rows/cols as (out_dim, in_dim). We want out_dim = n_layer*n_embd_altup.
        let mut per_layer_proj_flat = vec![0.0f32; n_layer * n_embd_altup];
        proj.matvec(inpl_scaled, &mut per_layer_proj_flat);
        // Scale by 1 / sqrt(n_embd) = 1 / sqrt(hidden_dim)
        let proj_scale = 1.0f32 / (c.hidden_dim as f32).sqrt();
        for v in &mut per_layer_proj_flat {
            *v *= proj_scale;
        }

        // Step 3: RMSNorm per (n_embd_altup) slice with per_layer_proj_norm weight
        let proj_norm_w = self
            .per_layer_proj_norm
            .as_ref()
            .expect("Gemma3n: per_layer_proj_norm");
        let mut per_layer_proj_normed = vec![0.0f32; n_layer * n_embd_altup];
        for l in 0..n_layer {
            let start = l * n_embd_altup;
            rms_norm(
                &per_layer_proj_flat[start..start + n_embd_altup],
                proj_norm_w,
                c.norm_eps,
                &mut per_layer_proj_normed[start..start + n_embd_altup],
            );
        }

        // Step 4: Add and scale by 1/sqrt(2)
        let inv_sqrt2 = 1.0f32 / 2.0f32.sqrt();
        for l in 0..n_layer {
            for i in 0..n_embd_altup {
                inp_per_layer_lookup[l][i] =
                    (inp_per_layer_lookup[l][i] + per_layer_proj_normed[l * n_embd_altup + i])
                        * inv_sqrt2;
            }
        }
        inp_per_layer_lookup
    }

    /// AltUp router modalities: compute a per-altup-input activation vector
    /// from the active stream (per llama.cpp `altup_compute_router_modalities`).
    fn altup_router_modalities(&self, active: &[f32], layer_idx: usize, n_altup: usize) -> Vec<f32> {
        let c = &self.config;
        let layer = &self.layers[layer_idx];
        let router_norm_w = layer
            .altup_router_norm
            .as_ref()
            .expect("Gemma3n: altup_router_norm");
        let router_w = layer
            .altup_router
            .as_ref()
            .expect("Gemma3n: altup_router");
        // router_inputs = RMSNorm(active, router_norm_w) / n_embd
        let mut router_inputs = vec![0.0f32; c.hidden_dim];
        rms_norm(active, router_norm_w, c.norm_eps, &mut router_inputs);
        let scale = 1.0f32 / c.hidden_dim as f32;
        for v in &mut router_inputs {
            *v *= scale;
        }
        // router_w shape: [hidden_dim, n_altup]. But mat_vec_f32 expects row-major
        // [rows=n_altup, cols=hidden_dim]. In ggml/f16 storage it's [ne0=hidden_dim, ne1=n_altup]
        // == row-major with rows=n_altup, cols=hidden_dim. So matmul(w, x) →
        // out[i] = sum_j w[i * hidden_dim + j] * x[j].
        let mut modalities = vec![0.0f32; n_altup];
        mat_vec_f32(router_w, n_altup, c.hidden_dim, &router_inputs, &mut modalities);
        for v in &mut modalities {
            *v = v.tanh();
        }
        modalities
    }

    /// AltUp predict step.
    ///
    /// Input:  `streams` — `[n_altup][hidden_dim]`, the current AltUp states.
    /// Output: `predictions` — `[n_altup][hidden_dim]`, streams after prediction.
    fn altup_predict(
        &self,
        streams: &[Vec<f32>],
        layer_idx: usize,
        n_altup: usize,
    ) -> Vec<Vec<f32>> {
        let c = &self.config;
        let layer = &self.layers[layer_idx];
        let i_altup_act = c.altup_active_idx.unwrap_or(0);
        let hidden_dim = c.hidden_dim;
        let modalities = self.altup_router_modalities(&streams[i_altup_act], layer_idx, n_altup);
        // predict_coef shape: [n_altup, n_altup * n_altup]. matmul with modalities
        // gives a vector of length n_altup*n_altup. Reshape to [n_altup, n_altup]
        // coefficient matrix (coef[i][j] = weight for using stream j to predict stream i).
        let predict_coef = layer
            .altup_predict_coef
            .as_ref()
            .expect("Gemma3n: altup_predict_coef");
        let mut coef_flat = vec![0.0f32; n_altup * n_altup];
        mat_vec_f32(predict_coef, n_altup * n_altup, n_altup, &modalities, &mut coef_flat);
        // Coefficient matrix: `coef[out][in]`
        // For each output stream: predictions[out] = sum_in coef[out][in] * streams[in] + streams[out]
        let mut predictions: Vec<Vec<f32>> = vec![vec![0.0f32; hidden_dim]; n_altup];
        for out_i in 0..n_altup {
            for in_i in 0..n_altup {
                let c_val = coef_flat[out_i * n_altup + in_i];
                if c_val == 0.0 {
                    continue;
                }
                for j in 0..hidden_dim {
                    predictions[out_i][j] += c_val * streams[in_i][j];
                }
            }
            // Add residual (streams[out_i])
            for j in 0..hidden_dim {
                predictions[out_i][j] += streams[out_i][j];
            }
        }
        predictions
    }

    /// AltUp correct step.
    ///
    /// - `predictions`: `[n_altup][hidden_dim]` from `altup_predict`.
    /// - `activated`: `[hidden_dim]` — the FFN output for the active stream.
    ///
    /// Output: `[n_altup][hidden_dim]` corrected streams.
    fn altup_correct(
        &self,
        predictions: &[Vec<f32>],
        activated: &[f32],
        layer_idx: usize,
        n_altup: usize,
    ) -> Vec<Vec<f32>> {
        let c = &self.config;
        let layer = &self.layers[layer_idx];
        let i_altup_act = c.altup_active_idx.unwrap_or(0);
        let hidden_dim = c.hidden_dim;
        let modalities = self.altup_router_modalities(activated, layer_idx, n_altup);
        let correct_coef = layer
            .altup_correct_coef
            .as_ref()
            .expect("Gemma3n: altup_correct_coef");
        let mut all_coefs = vec![0.0f32; n_altup];
        mat_vec_f32(correct_coef, n_altup, n_altup, &modalities, &mut all_coefs);
        // + 1.0 offset
        for v in &mut all_coefs {
            *v += 1.0;
        }
        // innovation = activated - predictions[i_altup_act]
        let mut innovation = vec![0.0f32; hidden_dim];
        for i in 0..hidden_dim {
            innovation[i] = activated[i] - predictions[i_altup_act][i];
        }
        // corrected[a] = predictions[a] + all_coefs[a] * innovation
        let mut corrected: Vec<Vec<f32>> = (0..n_altup).map(|a| predictions[a].clone()).collect();
        for a in 0..n_altup {
            let coef = all_coefs[a];
            for j in 0..hidden_dim {
                corrected[a][j] += coef * innovation[j];
            }
        }
        corrected
    }

    /// Laurel branch: low-rank projection `laurel_r @ laurel_l @ cur` +
    /// RMSNorm + residual add.
    fn laurel(&self, cur: &[f32], layer_idx: usize) -> Vec<f32> {
        let c = &self.config;
        let layer = &self.layers[layer_idx];
        let hidden_dim = c.hidden_dim;
        let laurel_l = layer.laurel_l.as_ref().expect("Gemma3n: laurel_l");
        let laurel_r = layer.laurel_r.as_ref().expect("Gemma3n: laurel_r");
        let laurel_post_norm = layer
            .laurel_post_norm
            .as_ref()
            .expect("Gemma3n: laurel_post_norm");
        // laurel_l shape: [hidden_dim, laurel_rank] in ggml means
        //   ne[0] = hidden_dim (fastest, = in-dim, cols)
        //   ne[1] = laurel_rank (out-dim, rows)
        // → matmul: out[i] = sum_j laurel_l[i * hidden_dim + j] * cur[j], out_dim = laurel_rank.
        let laurel_rank = laurel_l.len() / hidden_dim;
        let mut mid = vec![0.0f32; laurel_rank];
        mat_vec_f32(laurel_l, laurel_rank, hidden_dim, cur, &mut mid);
        // laurel_r shape: [laurel_rank, hidden_dim] → out_dim = hidden_dim.
        let mut tmp = vec![0.0f32; hidden_dim];
        mat_vec_f32(laurel_r, hidden_dim, laurel_rank, &mid, &mut tmp);
        // RMSNorm with weight
        let mut normed = vec![0.0f32; hidden_dim];
        rms_norm(&tmp, laurel_post_norm, c.norm_eps, &mut normed);
        // Residual add: laurel_out = normed + cur
        for i in 0..hidden_dim {
            normed[i] += cur[i];
        }
        normed
    }

    /// Per-layer first_prediction step (Gemma 3n bottom of layer):
    ///
    ///   fp = active_altup_stream * altup_correct_scale         (elementwise)
    ///   fp = per_layer_inp_gate @ fp                           (hidden→altup_dim)
    ///   fp = GELU(fp)
    ///   fp = fp * inp_this_layer                               (elementwise, altup_dim)
    ///   fp = per_layer_proj @ fp                               (altup_dim→hidden)
    ///   fp = RMSNorm(fp, per_layer_post_norm)                  (aka post_norm)
    fn gemma3n_first_prediction(
        &self,
        active_corrected: &[f32],
        layer_idx: usize,
        inp_this_layer: &[f32],
        n_embd_altup: usize,
    ) -> Vec<f32> {
        let c = &self.config;
        let layer = &self.layers[layer_idx];
        let hidden_dim = c.hidden_dim;

        let correct_scale = layer
            .altup_correct_scale
            .as_ref()
            .expect("Gemma3n: altup_correct_scale");
        // Scale
        let mut scaled = vec![0.0f32; hidden_dim];
        for i in 0..hidden_dim {
            scaled[i] = active_corrected[i] * correct_scale[i];
        }
        // Gate matmul: inp_gate shape [hidden_dim, n_embd_altup] → out_dim=n_embd_altup
        let inp_gate = layer.inp_gate.as_ref().expect("Gemma3n: inp_gate");
        let mut gated = vec![0.0f32; n_embd_altup];
        inp_gate.matvec(&scaled, &mut gated);
        // GELU
        for v in &mut gated {
            *v = gelu_approx(*v);
        }
        // elementwise mul with per-layer input for this layer
        for i in 0..n_embd_altup {
            gated[i] *= inp_this_layer[i];
        }
        // Project up to hidden_dim via per_layer_proj (shape [n_embd_altup, hidden_dim])
        let proj = layer.proj.as_ref().expect("Gemma3n: proj");
        let mut projected = vec![0.0f32; hidden_dim];
        proj.matvec(&gated, &mut projected);
        // post_norm RMSNorm
        let post_norm = layer.post_norm.as_ref().expect("Gemma3n: post_norm");
        let mut normed = vec![0.0f32; hidden_dim];
        rms_norm(&projected, post_norm, c.norm_eps, &mut normed);
        normed
    }

    /// Forward multiple tokens sequentially, returning logits for each.
    /// More efficient than calling forward() in a loop because buffers are reused.
    pub fn forward_batch(&mut self, token_ids: &[u32]) -> Vec<Vec<f32>> {
        let mut all_logits = Vec::with_capacity(token_ids.len());
        for &tok in token_ids {
            all_logits.push(self.forward(tok));
        }
        all_logits
    }

    /// Forward pass using only the first `draft_layers` layers (for speculative draft).
    /// Produces approximate logits at ~draft_layers/total_layers cost.
    /// KV cache entries are populated only for the draft layers.
    fn forward_draft(&mut self, token_id: u32, draft_layers: usize) -> Vec<f32> {
        let c = &self.config;
        let pos = self.kv_cache.seq_len();
        let rope_freqs_ref = self.rope_freqs.as_deref();
        let num_draft = draft_layers.min(c.num_layers);

        let emb_start = token_id as usize * c.hidden_dim;
        let mut hidden: Vec<f32> = self.embedding[emb_start..emb_start + c.hidden_dim].to_vec();
        // Gemma-2: scale embeddings by sqrt(hidden_dim) (no-op for others).
        if c.arch == ModelArch::Gemma2 {
            let scale = (c.hidden_dim as f32).sqrt();
            for h in &mut hidden {
                *h *= scale;
            }
        }

        let mut norm_buf = vec![0.0f32; c.hidden_dim];
        let kv_dim = c.num_kv_heads * c.head_dim;
        let mut q_buf = vec![0.0f32; c.num_heads * c.head_dim];
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
            // Qwen 2/2.5 bias (no-op for Llama/Mistral/Gemma/Qwen 3)
            if let Some(b) = &layer.q_bias {
                for (q, bi) in q_buf.iter_mut().zip(b.iter()) {
                    *q += bi;
                }
            }
            if let Some(b) = &layer.k_bias {
                for (k, bi) in k_buf.iter_mut().zip(b.iter()) {
                    *k += bi;
                }
            }
            if let Some(b) = &layer.v_bias {
                for (v, bi) in v_buf.iter_mut().zip(b.iter()) {
                    *v += bi;
                }
            }
            // Qwen 3 QK-Norm (per-head RMSNorm on Q, K before RoPE; no-op for others)
            if let Some(w) = &layer.q_norm {
                apply_qk_norm(&mut q_buf, w, c.head_dim, c.norm_eps);
            }
            if let Some(w) = &layer.k_norm {
                apply_qk_norm(&mut k_buf, w, c.head_dim, c.norm_eps);
            }

            for h in 0..c.num_heads {
                let start = h * c.head_dim;
                apply_rope_auto(
                    &mut q_buf[start..start + c.head_dim],
                    pos,
                    c.head_dim,
                    c.rope_theta,
                    rope_freqs_ref,
                    c.use_neox_rope(),
                );
            }
            for h in 0..c.num_kv_heads {
                let start = h * c.head_dim;
                apply_rope_auto(
                    &mut k_buf[start..start + c.head_dim],
                    pos,
                    c.head_dim,
                    c.rope_theta,
                    rope_freqs_ref,
                    c.use_neox_rope(),
                );
            }

            self.kv_cache.append(layer_idx, &k_buf, &v_buf);

            gqa_attention(
                &q_buf,
                &self.kv_cache,
                layer_idx,
                pos,
                c.num_heads,
                c.num_kv_heads,
                c.head_dim,
                c.sliding_window_for_layer(layer_idx),
                c.attn_logit_softcap,
                if c.arch == ModelArch::Gemma3n { Some(1.0) } else { None },
                &mut attn_out,
            );

            layer.o_proj.matvec(&attn_out, &mut o_buf);
            for i in 0..c.hidden_dim {
                hidden[i] += o_buf[i];
            }

            rms_norm(&hidden, &layer.ffn_norm, c.norm_eps, &mut norm_buf);
            let q8_ffn = quantize_row_q8_k(&norm_buf);
            layer.gate_proj.matvec_preq(&q8_ffn, &mut gate_buf);
            layer.up_proj.matvec_preq(&q8_ffn, &mut up_buf);
            c.apply_ffn_sparsity(layer_idx, &mut gate_buf);
            for i in 0..c.intermediate_dim {
                gate_buf[i] = c.apply_ffn_act(layer_idx, gate_buf[i]) * up_buf[i];
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
        let mut tokens = tokenizer.encode(prompt);
        // Prepend BOS if not already present
        // Only prepend BOS if the tokenizer's add_bos_token is True (Qwen 3: False).
        if tokenizer.add_bos_token && (tokens.is_empty() || tokens[0] != tokenizer.bos_id) {
            tokens.insert(0, tokenizer.bos_id);
        }

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

        // Repetition penalty (env-configurable, default 1.0 = disabled).
        // Applied to recently generated tokens' logits before sampling.
        // ALICE_LLM_REP_PENALTY=1.1 is typical for anti-repetition in Qwen 3.
        let rep_penalty: f32 = std::env::var("ALICE_LLM_REP_PENALTY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0);
        let rep_window: usize = std::env::var("ALICE_LLM_REP_WINDOW")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64);

        for _ in 0..max_new_tokens {
            // Repetition penalty on recently generated tokens (if enabled).
            if (rep_penalty - 1.0).abs() > f32::EPSILON {
                let start = generated.len().saturating_sub(rep_window);
                for &tok in &generated[start..] {
                    let idx = tok as usize;
                    if idx < logits.len() {
                        if logits[idx] > 0.0 {
                            logits[idx] /= rep_penalty;
                        } else {
                            logits[idx] *= rep_penalty;
                        }
                    }
                }
            }

            // Temperature
            if temperature > 0.0 && temperature != 1.0 {
                let inv_t = 1.0 / temperature;
                for l in &mut logits {
                    *l *= inv_t;
                }
            }

            // Top-k + argmax sampling
            let next_token = if top_k > 0 && top_k < logits.len() {
                let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
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
                    .map_or(0, |(idx, _)| *idx as u32)
            } else {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map_or(0, |(idx, _)| idx as u32)
            };

            if next_token == tokenizer.eos_id {
                break;
            }

            tokens.push(next_token);
            generated.push(next_token);

            // DEBUG: print token id + individual decode (temporary for arch verify)
            if std::env::var("ALICE_LLM_DEBUG_TOKENS").is_ok() {
                let tok_text = tokenizer.decode(&[next_token]);
                eprintln!("[TOK] id={next_token} text={tok_text:?}");
            }

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
        _top_k: usize,
        spec_k: usize,
        draft_layers: usize,
    ) -> GenerateResult {
        let start = Instant::now();
        let mut tokens = tokenizer.encode(prompt);
        // Only prepend BOS if the tokenizer's add_bos_token is True (Qwen 3: False).
        if tokenizer.add_bos_token && (tokens.is_empty() || tokens[0] != tokenizer.bos_id) {
            tokens.insert(0, tokenizer.bos_id);
        }

        self.clear_cache();
        let prompt_token_count = tokens.len();

        // Prefill
        let prefill_start = Instant::now();
        let mut logits = vec![0.0f32; self.config.vocab_size];
        for &tok in &tokens {
            logits = self.forward(tok);
        }
        let prefill_ms = prefill_start.elapsed().as_millis() as u64;

        // Decode with probabilistic speculative sampling
        let decode_start = Instant::now();
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut total_drafted: usize = 0;
        let mut total_accepted: usize = 0;
        let mut rng = Rng64::new(42);

        while generated.len() < max_new_tokens {
            // Sample from current logits
            let next_token = if temperature > 0.0 {
                sample_from_probs(&softmax(&logits), rng.next_f32())
            } else {
                argmax(&logits)
            };
            if next_token == tokenizer.eos_id {
                break;
            }
            generated.push(next_token);
            tokens.push(next_token);

            let remaining = max_new_tokens - generated.len();
            if remaining == 0 {
                logits = self.forward(next_token);
                continue;
            }

            let k = spec_k.min(remaining);

            // --- Draft phase: store logits for probabilistic acceptance ---
            let saved_pos = self.kv_cache.seq_len();
            let mut draft_tokens = Vec::with_capacity(k);
            let mut draft_logits_all = Vec::with_capacity(k);
            let mut draft_input = next_token;
            for _ in 0..k {
                let dl = self.forward_draft(draft_input, draft_layers);
                draft_input = argmax(&dl);
                draft_tokens.push(draft_input);
                draft_logits_all.push(dl);
            }
            total_drafted += draft_tokens.len();

            // --- Rollback ---
            self.kv_cache.rollback_to(saved_pos);

            // --- Verify phase: probabilistic speculative sampling ---
            logits = self.forward(next_token);

            let mut all_accepted = true;
            for i in 0..draft_tokens.len() {
                let p = softmax(&logits);
                let q = softmax(&draft_logits_all[i]);
                let x = draft_tokens[i] as usize;

                let p_x = if x < p.len() { p[x] } else { 0.0 };
                let q_x = if x < q.len() { q[x] } else { 1e-10 };
                let accept_prob = (p_x / q_x.max(1e-10)).min(1.0);

                let r = rng.next_f32();
                if r < accept_prob {
                    // Accepted by probabilistic criterion
                    generated.push(draft_tokens[i]);
                    tokens.push(draft_tokens[i]);
                    total_accepted += 1;
                    logits = self.forward(draft_tokens[i]);
                } else {
                    // Rejected: resample from max(0, p(x) - q(x))
                    let mut adjusted = vec![0.0f32; p.len()];
                    let mut adj_sum = 0.0f32;
                    for j in 0..p.len() {
                        adjusted[j] = (p[j] - q[j]).max(0.0);
                        adj_sum += adjusted[j];
                    }
                    let resampled = if adj_sum > 0.0 {
                        let inv = 1.0 / adj_sum;
                        for a in &mut adjusted {
                            *a *= inv;
                        }
                        sample_from_probs(&adjusted, rng.next_f32())
                    } else {
                        sample_from_probs(&p, rng.next_f32())
                    };
                    if resampled == tokenizer.eos_id {
                        all_accepted = false;
                        break;
                    }
                    generated.push(resampled);
                    tokens.push(resampled);
                    logits = self.forward(resampled);
                    all_accepted = false;
                    break;
                }
            }

            // If all K drafts accepted, sample one more from the final verify logits
            if all_accepted && generated.len() < max_new_tokens {
                let bonus = sample_from_probs(&softmax(&logits), rng.next_f32());
                if bonus != tokenizer.eos_id {
                    generated.push(bonus);
                    tokens.push(bonus);
                    logits = self.forward(bonus);
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

    /// True speculative decoding with a separate draft model.
    /// The draft model generates K candidate tokens, the main model verifies them
    /// using probabilistic speculative sampling (Leviathan et al.).
    pub fn generate_speculative_dual(
        &mut self,
        draft_model: &mut Llama3Model,
        tokenizer: &GgufTokenizer,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        spec_k: usize,
    ) -> GenerateResult {
        let start = Instant::now();
        let mut tokens = tokenizer.encode(prompt);
        // Only prepend BOS if the tokenizer's add_bos_token is True (Qwen 3: False).
        if tokenizer.add_bos_token && (tokens.is_empty() || tokens[0] != tokenizer.bos_id) {
            tokens.insert(0, tokenizer.bos_id);
        }

        self.clear_cache();
        draft_model.clear_cache();
        let prompt_token_count = tokens.len();

        // Prefill both models
        let prefill_start = Instant::now();
        let mut logits = vec![0.0f32; self.config.vocab_size];
        for &tok in &tokens {
            logits = self.forward(tok);
            draft_model.forward(tok);
        }
        let prefill_ms = prefill_start.elapsed().as_millis() as u64;

        // Decode with dual-model speculation
        let decode_start = Instant::now();
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut total_drafted: usize = 0;
        let mut total_accepted: usize = 0;
        let mut rng = Rng64::new(42);

        while generated.len() < max_new_tokens {
            // Sample from current main model logits
            let next_token = if temperature > 0.0 {
                sample_from_probs(&softmax(&logits), rng.next_f32())
            } else {
                argmax(&logits)
            };
            if next_token == tokenizer.eos_id {
                break;
            }
            generated.push(next_token);
            tokens.push(next_token);

            let remaining = max_new_tokens - generated.len();
            if remaining == 0 {
                logits = self.forward(next_token);
                draft_model.forward(next_token);
                continue;
            }

            let k = spec_k.min(remaining);

            // --- Draft phase: generate K tokens with draft model ---
            let saved_draft_pos = draft_model.kv_cache.seq_len();
            let _saved_main_pos = self.kv_cache.seq_len();
            let mut draft_tokens = Vec::with_capacity(k);
            let mut draft_logits_all = Vec::with_capacity(k);
            let mut draft_input = next_token;

            // Feed the accepted token to draft model first
            let dl = draft_model.forward(draft_input);
            draft_input = argmax(&dl);
            draft_tokens.push(draft_input);
            draft_logits_all.push(dl);

            for _ in 1..k {
                let dl = draft_model.forward(draft_input);
                draft_input = argmax(&dl);
                draft_tokens.push(draft_input);
                draft_logits_all.push(dl);
            }
            total_drafted += draft_tokens.len();

            // --- Verify phase: forward incrementally, stop early on rejection ---
            // Forward accepted token through main model
            logits = self.forward(next_token);

            let mut num_accepted = 0;
            let mut rejected = false;
            for i in 0..draft_tokens.len() {
                let p = softmax(&logits);
                let q = softmax(&draft_logits_all[i]);
                let x = draft_tokens[i] as usize;

                let p_x = if x < p.len() { p[x] } else { 0.0 };
                let q_x = if x < q.len() { q[x] } else { 1e-10 };
                let accept_prob = (p_x / q_x.max(1e-10)).min(1.0);

                let r = rng.next_f32();
                if r < accept_prob {
                    // Accepted — forward this draft token to get logits for next check
                    generated.push(draft_tokens[i]);
                    tokens.push(draft_tokens[i]);
                    total_accepted += 1;
                    num_accepted += 1;
                    logits = self.forward(draft_tokens[i]);
                } else {
                    // Rejected: resample from max(0, p - q)
                    let mut adjusted = vec![0.0f32; p.len()];
                    let mut adj_sum = 0.0f32;
                    for j in 0..p.len() {
                        adjusted[j] = (p[j] - q[j]).max(0.0);
                        adj_sum += adjusted[j];
                    }
                    let resampled = if adj_sum > 0.0 {
                        let inv = 1.0 / adj_sum;
                        for a in &mut adjusted {
                            *a *= inv;
                        }
                        sample_from_probs(&adjusted, rng.next_f32())
                    } else {
                        sample_from_probs(&p, rng.next_f32())
                    };
                    if resampled != tokenizer.eos_id {
                        generated.push(resampled);
                        tokens.push(resampled);
                        logits = self.forward(resampled);
                    }
                    rejected = true;
                    break;
                }
            }

            // If all K drafts accepted, bonus token from final verify logits
            if !rejected && num_accepted == draft_tokens.len() && generated.len() < max_new_tokens {
                let bonus = sample_from_probs(&softmax(&logits), rng.next_f32());
                if bonus != tokenizer.eos_id {
                    generated.push(bonus);
                    tokens.push(bonus);
                    logits = self.forward(bonus);
                }
            }

            // Sync draft model KV cache:
            // Draft has entries for: next_token + draft_tokens[0..k]
            // We accepted num_accepted of those.
            let draft_keep = saved_draft_pos + 1 + num_accepted;
            draft_model.kv_cache.rollback_to(draft_keep);
            // Feed the resampled/bonus token that draft hasn't seen
            if let Some(&last) = tokens.last() {
                draft_model.forward(last);
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

        GenerateResult {
            text: tokenizer.decode(&generated),
            tokens_generated: gen_count,
            prompt_tokens: prompt_token_count,
            prefill_ms,
            decode_ms,
            total_ms,
            tokens_per_sec: tok_per_sec,
            spec_stats: Some(SpecStats {
                draft_tokens: total_drafted,
                accepted_tokens: total_accepted,
                draft_layers: draft_model.config.num_layers,
                spec_k,
            }),
        }
    }

    /// Forward pass using ternary-quantized weights (no multiplications in projections).
    /// Must call `load_ternary()` before using this method.
    pub fn forward_ternary(&mut self, token_id: u32) -> Vec<f32> {
        let ternary_layers = self
            .ternary_layers
            .as_ref()
            .expect("call load_ternary() first");
        let ternary_output = self
            .ternary_output_proj
            .as_ref()
            .expect("call load_ternary() first");
        let c = &self.config;
        let pos = self.kv_cache.seq_len();
        let rope_freqs_ref = self.rope_freqs.as_deref();

        let emb_start = token_id as usize * c.hidden_dim;
        let mut hidden: Vec<f32> = self.embedding[emb_start..emb_start + c.hidden_dim].to_vec();
        // Gemma-2: scale embeddings by sqrt(hidden_dim) (no-op for others).
        if c.arch == ModelArch::Gemma2 {
            let scale = (c.hidden_dim as f32).sqrt();
            for h in &mut hidden {
                *h *= scale;
            }
        }

        let mut norm_buf = vec![0.0f32; c.hidden_dim];
        let kv_dim = c.num_kv_heads * c.head_dim;
        let mut q_buf = vec![0.0f32; c.num_heads * c.head_dim];
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
                apply_rope_auto(
                    &mut q_buf[start..start + c.head_dim],
                    pos,
                    c.head_dim,
                    c.rope_theta,
                    rope_freqs_ref,
                    c.use_neox_rope(),
                );
            }
            for h in 0..c.num_kv_heads {
                let start = h * c.head_dim;
                apply_rope_auto(
                    &mut k_buf[start..start + c.head_dim],
                    pos,
                    c.head_dim,
                    c.rope_theta,
                    rope_freqs_ref,
                    c.use_neox_rope(),
                );
            }

            self.kv_cache.append(layer_idx, &k_buf, &v_buf);

            gqa_attention(
                &q_buf,
                &self.kv_cache,
                layer_idx,
                pos,
                c.num_heads,
                c.num_kv_heads,
                c.head_dim,
                c.sliding_window_for_layer(layer_idx),
                c.attn_logit_softcap,
                if c.arch == ModelArch::Gemma3n { Some(1.0) } else { None },
                &mut attn_out,
            );

            ternary_matvec(&tl.o_proj, &attn_out, &mut o_buf);
            for i in 0..c.hidden_dim {
                hidden[i] += o_buf[i];
            }

            rms_norm(&hidden, &tl.ffn_norm, c.norm_eps, &mut norm_buf);

            ternary_matvec(&tl.gate_proj, &norm_buf, &mut gate_buf);
            ternary_matvec(&tl.up_proj, &norm_buf, &mut up_buf);
            c.apply_ffn_sparsity(layer_idx, &mut gate_buf);
            for i in 0..c.intermediate_dim {
                gate_buf[i] = c.apply_ffn_act(layer_idx, gate_buf[i]) * up_buf[i];
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
        let mut tokens = tokenizer.encode(prompt);
        // Only prepend BOS if the tokenizer's add_bos_token is True (Qwen 3: False).
        if tokenizer.add_bos_token && (tokens.is_empty() || tokens[0] != tokenizer.bos_id) {
            tokens.insert(0, tokenizer.bos_id);
        }

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

    // ─── Sparse Ternary (N:M structured sparsity) ─────────────────────────

    /// Convert all weights to sparse ternary format with N:M structured sparsity.
    /// n_keep = number of non-zero weights per 16-element block (e.g. 8 = 8:16 = 50% density).
    pub fn load_sparse_ternary(&mut self, threshold_ratio: f32, n_keep: usize) {
        let c = &self.config;
        let kv_dim = c.num_kv_heads * c.head_dim;
        let mut layers = Vec::with_capacity(c.num_layers);

        for (i, layer) in self.layers.iter().enumerate() {
            eprint!("  Sparse-ternarizing layer {i}/{} ...\r", c.num_layers);
            layers.push(SparseTernaryLayerWeights {
                attn_norm: layer.attn_norm.clone(),
                q_proj: sparsify_weight(
                    &layer.q_proj,
                    c.hidden_dim,
                    c.hidden_dim,
                    threshold_ratio,
                    n_keep,
                ),
                k_proj: sparsify_weight(
                    &layer.k_proj,
                    kv_dim,
                    c.hidden_dim,
                    threshold_ratio,
                    n_keep,
                ),
                v_proj: sparsify_weight(
                    &layer.v_proj,
                    kv_dim,
                    c.hidden_dim,
                    threshold_ratio,
                    n_keep,
                ),
                o_proj: sparsify_weight(
                    &layer.o_proj,
                    c.hidden_dim,
                    c.hidden_dim,
                    threshold_ratio,
                    n_keep,
                ),
                ffn_norm: layer.ffn_norm.clone(),
                gate_proj: sparsify_weight(
                    &layer.gate_proj,
                    c.intermediate_dim,
                    c.hidden_dim,
                    threshold_ratio,
                    n_keep,
                ),
                up_proj: sparsify_weight(
                    &layer.up_proj,
                    c.intermediate_dim,
                    c.hidden_dim,
                    threshold_ratio,
                    n_keep,
                ),
                down_proj: sparsify_weight(
                    &layer.down_proj,
                    c.hidden_dim,
                    c.intermediate_dim,
                    threshold_ratio,
                    n_keep,
                ),
            });
        }
        eprintln!(
            "  Sparse-ternarized {}/{} layers ({}:16)",
            c.num_layers, c.num_layers, n_keep
        );

        self.sparse_ternary_output = Some(sparsify_weight(
            &self.output_proj,
            c.vocab_size,
            c.hidden_dim,
            threshold_ratio,
            n_keep,
        ));
        self.sparse_ternary_layers = Some(layers);
    }

    /// Forward pass using sparse ternary weights (block-packed, SDOT+LUT optimized).
    pub fn forward_sparse_ternary(&mut self, token_id: u32) -> Vec<f32> {
        let st_layers = self
            .sparse_ternary_layers
            .as_ref()
            .expect("call load_sparse_ternary() first");
        let st_output = self
            .sparse_ternary_output
            .as_ref()
            .expect("call load_sparse_ternary() first");
        let c = &self.config;
        let pos = self.kv_cache.seq_len();
        let rope_freqs_ref = self.rope_freqs.as_deref();

        let emb_start = token_id as usize * c.hidden_dim;
        let mut hidden: Vec<f32> = self.embedding[emb_start..emb_start + c.hidden_dim].to_vec();
        // Gemma-2: scale embeddings by sqrt(hidden_dim) (no-op for others).
        if c.arch == ModelArch::Gemma2 {
            let scale = (c.hidden_dim as f32).sqrt();
            for h in &mut hidden {
                *h *= scale;
            }
        }

        let mut norm_buf = vec![0.0f32; c.hidden_dim];
        let kv_dim = c.num_kv_heads * c.head_dim;
        let mut q_buf = vec![0.0f32; c.num_heads * c.head_dim];
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; c.hidden_dim];
        let mut o_buf = vec![0.0f32; c.hidden_dim];
        let mut gate_buf = vec![0.0f32; c.intermediate_dim];
        let mut up_buf = vec![0.0f32; c.intermediate_dim];
        let mut down_buf = vec![0.0f32; c.hidden_dim];

        for layer_idx in 0..c.num_layers {
            let sl = &st_layers[layer_idx];

            rms_norm(&hidden, &sl.attn_norm, c.norm_eps, &mut norm_buf);

            sparse_ternary_matvec(&sl.q_proj, &norm_buf, &mut q_buf);
            sparse_ternary_matvec(&sl.k_proj, &norm_buf, &mut k_buf);
            sparse_ternary_matvec(&sl.v_proj, &norm_buf, &mut v_buf);

            for h in 0..c.num_heads {
                let start = h * c.head_dim;
                apply_rope_auto(
                    &mut q_buf[start..start + c.head_dim],
                    pos,
                    c.head_dim,
                    c.rope_theta,
                    rope_freqs_ref,
                    c.use_neox_rope(),
                );
            }
            for h in 0..c.num_kv_heads {
                let start = h * c.head_dim;
                apply_rope_auto(
                    &mut k_buf[start..start + c.head_dim],
                    pos,
                    c.head_dim,
                    c.rope_theta,
                    rope_freqs_ref,
                    c.use_neox_rope(),
                );
            }

            self.kv_cache.append(layer_idx, &k_buf, &v_buf);

            gqa_attention(
                &q_buf,
                &self.kv_cache,
                layer_idx,
                pos,
                c.num_heads,
                c.num_kv_heads,
                c.head_dim,
                c.sliding_window_for_layer(layer_idx),
                c.attn_logit_softcap,
                if c.arch == ModelArch::Gemma3n { Some(1.0) } else { None },
                &mut attn_out,
            );

            sparse_ternary_matvec(&sl.o_proj, &attn_out, &mut o_buf);
            for i in 0..c.hidden_dim {
                hidden[i] += o_buf[i];
            }

            rms_norm(&hidden, &sl.ffn_norm, c.norm_eps, &mut norm_buf);

            sparse_ternary_matvec(&sl.gate_proj, &norm_buf, &mut gate_buf);
            sparse_ternary_matvec(&sl.up_proj, &norm_buf, &mut up_buf);
            c.apply_ffn_sparsity(layer_idx, &mut gate_buf);
            for i in 0..c.intermediate_dim {
                gate_buf[i] = c.apply_ffn_act(layer_idx, gate_buf[i]) * up_buf[i];
            }
            sparse_ternary_matvec(&sl.down_proj, &gate_buf, &mut down_buf);

            for i in 0..c.hidden_dim {
                hidden[i] += down_buf[i];
            }
        }

        self.kv_cache.advance();

        rms_norm(&hidden, &self.output_norm, c.norm_eps, &mut norm_buf);
        let mut logits = vec![0.0f32; c.vocab_size];
        sparse_ternary_matvec(st_output, &norm_buf, &mut logits);
        logits
    }

    /// Draft forward pass using sparse ternary weights (first N layers only).
    fn forward_sparse_ternary_draft(&mut self, token_id: u32, draft_layers: usize) -> Vec<f32> {
        let st_layers = self
            .sparse_ternary_layers
            .as_ref()
            .expect("call load_sparse_ternary() first");
        let st_output = self
            .sparse_ternary_output
            .as_ref()
            .expect("call load_sparse_ternary() first");
        let c = &self.config;
        let pos = self.kv_cache.seq_len();
        let rope_freqs_ref = self.rope_freqs.as_deref();
        let num_draft = draft_layers.min(c.num_layers);

        let emb_start = token_id as usize * c.hidden_dim;
        let mut hidden: Vec<f32> = self.embedding[emb_start..emb_start + c.hidden_dim].to_vec();
        // Gemma-2: scale embeddings by sqrt(hidden_dim) (no-op for others).
        if c.arch == ModelArch::Gemma2 {
            let scale = (c.hidden_dim as f32).sqrt();
            for h in &mut hidden {
                *h *= scale;
            }
        }

        let mut norm_buf = vec![0.0f32; c.hidden_dim];
        let kv_dim = c.num_kv_heads * c.head_dim;
        let mut q_buf = vec![0.0f32; c.num_heads * c.head_dim];
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; c.hidden_dim];
        let mut o_buf = vec![0.0f32; c.hidden_dim];
        let mut gate_buf = vec![0.0f32; c.intermediate_dim];
        let mut up_buf = vec![0.0f32; c.intermediate_dim];
        let mut down_buf = vec![0.0f32; c.hidden_dim];

        for layer_idx in 0..num_draft {
            let sl = &st_layers[layer_idx];

            rms_norm(&hidden, &sl.attn_norm, c.norm_eps, &mut norm_buf);

            sparse_ternary_matvec(&sl.q_proj, &norm_buf, &mut q_buf);
            sparse_ternary_matvec(&sl.k_proj, &norm_buf, &mut k_buf);
            sparse_ternary_matvec(&sl.v_proj, &norm_buf, &mut v_buf);

            for h in 0..c.num_heads {
                let start = h * c.head_dim;
                apply_rope_auto(
                    &mut q_buf[start..start + c.head_dim],
                    pos,
                    c.head_dim,
                    c.rope_theta,
                    rope_freqs_ref,
                    c.use_neox_rope(),
                );
            }
            for h in 0..c.num_kv_heads {
                let start = h * c.head_dim;
                apply_rope_auto(
                    &mut k_buf[start..start + c.head_dim],
                    pos,
                    c.head_dim,
                    c.rope_theta,
                    rope_freqs_ref,
                    c.use_neox_rope(),
                );
            }

            self.kv_cache.append(layer_idx, &k_buf, &v_buf);

            gqa_attention(
                &q_buf,
                &self.kv_cache,
                layer_idx,
                pos,
                c.num_heads,
                c.num_kv_heads,
                c.head_dim,
                c.sliding_window_for_layer(layer_idx),
                c.attn_logit_softcap,
                if c.arch == ModelArch::Gemma3n { Some(1.0) } else { None },
                &mut attn_out,
            );

            sparse_ternary_matvec(&sl.o_proj, &attn_out, &mut o_buf);
            for i in 0..c.hidden_dim {
                hidden[i] += o_buf[i];
            }

            rms_norm(&hidden, &sl.ffn_norm, c.norm_eps, &mut norm_buf);

            sparse_ternary_matvec(&sl.gate_proj, &norm_buf, &mut gate_buf);
            sparse_ternary_matvec(&sl.up_proj, &norm_buf, &mut up_buf);
            c.apply_ffn_sparsity(layer_idx, &mut gate_buf);
            for i in 0..c.intermediate_dim {
                gate_buf[i] = c.apply_ffn_act(layer_idx, gate_buf[i]) * up_buf[i];
            }
            sparse_ternary_matvec(&sl.down_proj, &gate_buf, &mut down_buf);
            for i in 0..c.hidden_dim {
                hidden[i] += down_buf[i];
            }
        }

        self.kv_cache.advance();

        rms_norm(&hidden, &self.output_norm, c.norm_eps, &mut norm_buf);
        let mut logits = vec![0.0f32; c.vocab_size];
        sparse_ternary_matvec(st_output, &norm_buf, &mut logits);
        logits
    }

    /// Generate with sparse ternary speculative decoding.
    /// Draft model = first `draft_layers` layers (layer-skip).
    /// Verify model = all layers.
    pub fn generate_sparse_ternary_speculative(
        &mut self,
        tokenizer: &GgufTokenizer,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        _top_k: usize,
        spec_k: usize,
        draft_layers: usize,
    ) -> GenerateResult {
        let start = Instant::now();
        let mut tokens = tokenizer.encode(prompt);
        // Only prepend BOS if the tokenizer's add_bos_token is True (Qwen 3: False).
        if tokenizer.add_bos_token && (tokens.is_empty() || tokens[0] != tokenizer.bos_id) {
            tokens.insert(0, tokenizer.bos_id);
        }

        self.clear_cache();
        let prompt_token_count = tokens.len();

        // Prefill
        let prefill_start = Instant::now();
        let mut logits = vec![0.0f32; self.config.vocab_size];
        for &tok in &tokens {
            logits = self.forward_sparse_ternary(tok);
        }
        let prefill_ms = prefill_start.elapsed().as_millis() as u64;

        // Decode with probabilistic speculative sampling
        let decode_start = Instant::now();
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut total_drafted: usize = 0;
        let mut total_accepted: usize = 0;
        let mut rng = Rng64::new(42);

        while generated.len() < max_new_tokens {
            let next_token = if temperature > 0.0 {
                sample_from_probs(&softmax(&logits), rng.next_f32())
            } else {
                argmax(&logits)
            };
            if next_token == tokenizer.eos_id {
                break;
            }
            generated.push(next_token);
            tokens.push(next_token);

            let remaining = max_new_tokens - generated.len();
            if remaining == 0 {
                logits = self.forward_sparse_ternary(next_token);
                continue;
            }

            let k = spec_k.min(remaining);

            // --- Draft phase: store logits for probabilistic acceptance ---
            let saved_pos = self.kv_cache.seq_len();
            let mut draft_tokens = Vec::with_capacity(k);
            let mut draft_logits_all = Vec::with_capacity(k);
            let mut draft_input = next_token;
            for _ in 0..k {
                let dl = self.forward_sparse_ternary_draft(draft_input, draft_layers);
                draft_input = argmax(&dl);
                draft_tokens.push(draft_input);
                draft_logits_all.push(dl);
            }
            total_drafted += draft_tokens.len();

            // --- Rollback ---
            self.kv_cache.rollback_to(saved_pos);

            // --- Verify phase: probabilistic speculative sampling ---
            logits = self.forward_sparse_ternary(next_token);

            let mut all_accepted = true;
            for i in 0..draft_tokens.len() {
                let p = softmax(&logits);
                let q = softmax(&draft_logits_all[i]);
                let x = draft_tokens[i] as usize;

                let p_x = if x < p.len() { p[x] } else { 0.0 };
                let q_x = if x < q.len() { q[x] } else { 1e-10 };
                let accept_prob = (p_x / q_x.max(1e-10)).min(1.0);

                let r = rng.next_f32();
                if r < accept_prob {
                    generated.push(draft_tokens[i]);
                    tokens.push(draft_tokens[i]);
                    total_accepted += 1;
                    logits = self.forward_sparse_ternary(draft_tokens[i]);
                } else {
                    // Rejected: resample from max(0, p(x) - q(x))
                    let mut adjusted = vec![0.0f32; p.len()];
                    let mut adj_sum = 0.0f32;
                    for j in 0..p.len() {
                        adjusted[j] = (p[j] - q[j]).max(0.0);
                        adj_sum += adjusted[j];
                    }
                    let resampled = if adj_sum > 0.0 {
                        let inv = 1.0 / adj_sum;
                        for a in &mut adjusted {
                            *a *= inv;
                        }
                        sample_from_probs(&adjusted, rng.next_f32())
                    } else {
                        sample_from_probs(&p, rng.next_f32())
                    };
                    if resampled == tokenizer.eos_id {
                        all_accepted = false;
                        break;
                    }
                    generated.push(resampled);
                    tokens.push(resampled);
                    logits = self.forward_sparse_ternary(resampled);
                    all_accepted = false;
                    break;
                }
            }

            // If all K drafts accepted, sample one more from final verify logits
            if all_accepted && generated.len() < max_new_tokens {
                let bonus = sample_from_probs(&softmax(&logits), rng.next_f32());
                if bonus != tokenizer.eos_id {
                    generated.push(bonus);
                    tokens.push(bonus);
                    logits = self.forward_sparse_ternary(bonus);
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

        GenerateResult {
            text: tokenizer.decode(&generated),
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

    // ─── Paged KV Cache forward ─────────────────────────────────────────────

    /// Forward pass using a per-request PagedKvCache instead of the model's flat cache.
    fn forward_paged(&self, token_id: u32, paged_cache: &mut PagedKvCache) -> Vec<f32> {
        let c = &self.config;
        let pos = paged_cache.seq_len();
        let rope_freqs_ref = self.rope_freqs.as_deref();

        let emb_start = token_id as usize * c.hidden_dim;
        let mut hidden: Vec<f32> = self.embedding[emb_start..emb_start + c.hidden_dim].to_vec();
        // Gemma-2: scale embeddings by sqrt(hidden_dim) (no-op for others).
        if c.arch == ModelArch::Gemma2 {
            let scale = (c.hidden_dim as f32).sqrt();
            for h in &mut hidden {
                *h *= scale;
            }
        }

        let mut norm_buf = vec![0.0f32; c.hidden_dim];
        let kv_dim = c.num_kv_heads * c.head_dim;
        let mut q_buf = vec![0.0f32; c.num_heads * c.head_dim];
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
            // Qwen 2/2.5 bias (no-op for Llama/Mistral/Gemma/Qwen 3)
            if let Some(b) = &layer.q_bias {
                for (q, bi) in q_buf.iter_mut().zip(b.iter()) {
                    *q += bi;
                }
            }
            if let Some(b) = &layer.k_bias {
                for (k, bi) in k_buf.iter_mut().zip(b.iter()) {
                    *k += bi;
                }
            }
            if let Some(b) = &layer.v_bias {
                for (v, bi) in v_buf.iter_mut().zip(b.iter()) {
                    *v += bi;
                }
            }
            // Qwen 3 QK-Norm (per-head RMSNorm on Q, K before RoPE; no-op for others)
            if let Some(w) = &layer.q_norm {
                apply_qk_norm(&mut q_buf, w, c.head_dim, c.norm_eps);
            }
            if let Some(w) = &layer.k_norm {
                apply_qk_norm(&mut k_buf, w, c.head_dim, c.norm_eps);
            }

            for h in 0..c.num_heads {
                let start = h * c.head_dim;
                apply_rope_auto(
                    &mut q_buf[start..start + c.head_dim],
                    pos,
                    c.head_dim,
                    c.rope_theta,
                    rope_freqs_ref,
                    c.use_neox_rope(),
                );
            }
            for h in 0..c.num_kv_heads {
                let start = h * c.head_dim;
                apply_rope_auto(
                    &mut k_buf[start..start + c.head_dim],
                    pos,
                    c.head_dim,
                    c.rope_theta,
                    rope_freqs_ref,
                    c.use_neox_rope(),
                );
            }

            paged_cache.append(layer_idx, &k_buf, &v_buf);

            gqa_attention_paged(
                &q_buf,
                paged_cache,
                layer_idx,
                pos,
                c.num_heads,
                c.num_kv_heads,
                c.head_dim,
                c.sliding_window_for_layer(layer_idx),
                c.attn_logit_softcap,
                &mut attn_out,
            );

            layer.o_proj.matvec(&attn_out, &mut o_buf);

            for i in 0..c.hidden_dim {
                hidden[i] += o_buf[i];
            }

            rms_norm(&hidden, &layer.ffn_norm, c.norm_eps, &mut norm_buf);

            let q8_ffn = quantize_row_q8_k(&norm_buf);
            layer.gate_proj.matvec_preq(&q8_ffn, &mut gate_buf);
            layer.up_proj.matvec_preq(&q8_ffn, &mut up_buf);

            c.apply_ffn_sparsity(layer_idx, &mut gate_buf);
            for i in 0..c.intermediate_dim {
                gate_buf[i] = c.apply_ffn_act(layer_idx, gate_buf[i]) * up_buf[i];
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
            if req.done {
                continue;
            }
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
            .map_or(0, |(idx, _)| *idx as u32)
    } else {
        argmax(&logits)
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map_or(0, |(idx, _)| idx as u32)
}

/// Convert logits to probability distribution via softmax.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for p in &mut probs {
            *p *= inv;
        }
    }
    probs
}

/// Sample a token index from a probability distribution using the provided RNG value.
/// `rand_val` should be uniform in [0, 1).
fn sample_from_probs(probs: &[f32], rand_val: f32) -> u32 {
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if rand_val < cumsum {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Simple xorshift64 PRNG for speculative sampling.
struct Rng64 {
    state: u64,
}

impl Rng64 {
    const fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_BABEu64
            } else {
                seed
            },
        }
    }

    /// Returns a uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        // Use upper 24 bits for mantissa
        (x >> 40) as f32 / (1u64 << 24) as f32
    }
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

fn ternarize_weight(
    w: &WeightRef<'_>,
    rows: usize,
    cols: usize,
    threshold_ratio: f32,
) -> TernaryMatrix {
    TernaryMatrix::from_quantized(w.data, w.qtype, rows, cols, threshold_ratio)
}

fn sparsify_weight(
    w: &WeightRef<'_>,
    rows: usize,
    cols: usize,
    threshold_ratio: f32,
    n_keep: usize,
) -> SparseTernaryMatrix {
    // Dequantize to f32 then convert to sparse ternary with N:M sparsity
    let weights_f32 = w.dequantize_all(rows, cols);
    SparseTernaryMatrix::from_f32_weights(&weights_f32, rows, cols, threshold_ratio, n_keep)
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
    // Gemma-2: num_heads * head_dim (= 2048) != hidden_dim (= 2304).
    // Other models: q_dim == hidden_dim (identity).
    let q_dim = config.num_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;

    let attn_norm = gguf.tensor_to_f32(&format!("{prefix}.attn_norm.weight"))?;
    let q_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.attn_q.weight"),
        q_dim,
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
        q_dim,
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

    // Qwen 2/2.5: attention projection biases (absent for Llama/Mistral/Gemma/Qwen 3).
    let q_bias = gguf.tensor_to_f32(&format!("{prefix}.attn_q.bias"));
    let k_bias = gguf.tensor_to_f32(&format!("{prefix}.attn_k.bias"));
    let v_bias = gguf.tensor_to_f32(&format!("{prefix}.attn_v.bias"));

    // Qwen 3: per-head RMSNorm on Q/K before RoPE (absent for other arch).
    let q_norm = gguf.tensor_to_f32(&format!("{prefix}.attn_q_norm.weight"));
    let k_norm = gguf.tensor_to_f32(&format!("{prefix}.attn_k_norm.weight"));

    // Gemma-2: sandwich norm (post-attention + post-FFN, before residual add).
    let post_attn_norm = gguf.tensor_to_f32(&format!("{prefix}.post_attention_norm.weight"));
    let post_ffn_norm = gguf.tensor_to_f32(&format!("{prefix}.post_ffw_norm.weight"));

    // Gemma 3n per-layer tensors (None for other architectures).
    let post_norm = gguf.tensor_to_f32(&format!("{prefix}.post_norm.weight"));
    let laurel_l = gguf.tensor_to_f32(&format!("{prefix}.laurel_l.weight"));
    let laurel_r = gguf.tensor_to_f32(&format!("{prefix}.laurel_r.weight"));
    let laurel_post_norm = gguf.tensor_to_f32(&format!("{prefix}.laurel_post_norm.weight"));
    let altup_router = gguf.tensor_to_f32(&format!("{prefix}.altup_router.weight"));
    let altup_router_norm = gguf.tensor_to_f32(&format!("{prefix}.altup_router_norm.weight"));
    let altup_predict_coef = gguf.tensor_to_f32(&format!("{prefix}.altup_predict_coef.weight"));
    let altup_correct_coef = gguf.tensor_to_f32(&format!("{prefix}.altup_correct_coef.weight"));
    let altup_correct_scale = gguf.tensor_to_f32(&format!("{prefix}.altup_correct_scale.weight"));

    // Gemma 3n per-layer input embedding gate/proj (WeightRef, quantized).
    //
    // Convention (matches WeightRef.matvec): `rows = out_dim, cols = in_dim`.
    //   - inp_gate.weight [n_embd, n_embd_altup] in ggml (ne0=in, ne1=out)
    //     → in=hidden_dim, out=per_layer_dim → rows=per_layer_dim, cols=hidden_dim
    //   - proj.weight     [n_embd_altup, n_embd] → in=per_layer_dim, out=hidden_dim
    //     → rows=hidden_dim, cols=per_layer_dim
    let (inp_gate, proj) = if let Some(per_layer_dim) = config.per_layer_input_embedding_dim {
        let inp_gate = load_weight_ref(
            gguf,
            &format!("{prefix}.inp_gate.weight"),
            per_layer_dim,      // rows = out_dim
            config.hidden_dim,  // cols = in_dim
        );
        let proj = load_weight_ref(
            gguf,
            &format!("{prefix}.proj.weight"),
            config.hidden_dim,  // rows = out_dim
            per_layer_dim,      // cols = in_dim
        );
        (inp_gate, proj)
    } else {
        (None, None)
    };

    Some(LayerWeights {
        attn_norm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        q_bias,
        k_bias,
        v_bias,
        q_norm,
        k_norm,
        post_attn_norm,
        post_ffn_norm,
        ffn_norm,
        gate_proj,
        up_proj,
        down_proj,
        post_norm,
        inp_gate,
        proj,
        laurel_l,
        laurel_r,
        laurel_post_norm,
        altup_router,
        altup_router_norm,
        altup_predict_coef,
        altup_correct_coef,
        altup_correct_scale,
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
        assert!(
            q4k_bytes > 3.0 && q4k_bytes < 7.0,
            "Q4_K estimate: {q4k_bytes:.1} GB"
        );
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
        assert_eq!(
            mp.get(0).ffn_mode,
            LayerQuantMode::SparseTernary { n_keep: 8 }
        );
    }

    #[test]
    fn test_mixed_precision_bits_estimate() {
        // Full ternary: 1.58 bits/param
        let mp_ternary = MixedPrecisionConfig::uniform(LayerQuantConfig::full_ternary(), 32);
        let bits_ternary = mp_ternary.estimate_bits_per_param();
        assert!(
            (bits_ternary - 1.58).abs() < 0.01,
            "full ternary: {bits_ternary}"
        );

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
        eprintln!("70B estimate: {bpp:.3} bits/param → {model_size_gb:.1} GB (target: <10 GB)");
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

    // ── Gemma 3n activation sparsity (gaussian_topk) ─────────────────────────

    fn gemma3n_config_with_sparsity(scales: Vec<f32>) -> Llama3Config {
        Llama3Config {
            arch: ModelArch::Gemma3n,
            activation_sparsity_scale: Some(scales),
            ..Llama3Config::llama3_8b()
        }
    }

    #[test]
    fn test_apply_ffn_sparsity_noop_non_gemma3n() {
        // Non-Gemma3n architectures must not touch the buffer.
        let c = Llama3Config::llama3_8b(); // arch = Llama
        let mut buf = vec![1.0, 2.0, 3.0, 4.0];
        c.apply_ffn_sparsity(0, &mut buf);
        assert_eq!(buf, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_apply_ffn_sparsity_noop_dense_layer() {
        // -inf scale means the layer is dense (no sparsity).
        let c = gemma3n_config_with_sparsity(vec![f32::NEG_INFINITY; 4]);
        let mut buf = vec![1.0, 2.0, 3.0, 4.0];
        c.apply_ffn_sparsity(0, &mut buf);
        assert_eq!(buf, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_apply_ffn_sparsity_gaussian_topk_math() {
        // scale = 1.6448 with buf = [1,2,3,4,5]:
        //   mean = 3.0
        //   unbiased var = ((-2)^2 + (-1)^2 + 0 + 1 + 4) / 4 = 10/4 = 2.5
        //   std = sqrt(2.5) ≈ 1.5811
        //   cutoff = 3.0 + 1.6448 * 1.5811 ≈ 5.601
        //   result = ReLU(x - 5.601) = [0, 0, 0, 0, 0]
        // Only the extreme tail survives — with scale ~1.645 (Φ⁻¹(0.95)) on a
        // uniform-ish buffer this should be all zeros.
        let c = gemma3n_config_with_sparsity(vec![1.6448536; 4]);
        let mut buf = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        c.apply_ffn_sparsity(0, &mut buf);
        for &v in &buf {
            assert!(v.abs() < 1e-5, "expected all zeros, got {buf:?}");
        }
    }

    #[test]
    fn test_apply_ffn_sparsity_preserves_extreme_tail() {
        // With an outlier, only the outlier should survive.
        //   buf = [0,0,0,0,0,0,0,0,0,10]
        //   mean = 1.0
        //   unbiased var = (9 * 1 + 81) / 9 = 90/9 = 10
        //   std = sqrt(10) ≈ 3.162
        //   cutoff = 1.0 + 1.6448 * 3.162 ≈ 6.201
        //   result = [0, 0, ..., 0, ReLU(10 - 6.201) = 3.799]
        let c = gemma3n_config_with_sparsity(vec![1.6448536; 4]);
        let mut buf = vec![0.0; 10];
        buf[9] = 10.0;
        c.apply_ffn_sparsity(0, &mut buf);
        for &v in &buf[..9] {
            assert!(v.abs() < 1e-5, "prefix must be zero, got {buf:?}");
        }
        assert!(
            (buf[9] - 3.799).abs() < 0.01,
            "outlier survives with expected value, got {}",
            buf[9]
        );
    }

    #[test]
    fn test_apply_ffn_sparsity_short_buffer_noop() {
        // n < 2 → cannot compute unbiased variance, must be a no-op.
        let c = gemma3n_config_with_sparsity(vec![1.6448; 4]);
        let mut buf = vec![42.0];
        c.apply_ffn_sparsity(0, &mut buf);
        assert_eq!(buf, vec![42.0]);
    }

    #[test]
    fn test_apply_ffn_act_gemma3n_always_gelu() {
        // Gemma 3n uses GELU for all layers (sparse or dense) — never SiLU.
        let c = gemma3n_config_with_sparsity(vec![1.6448, f32::NEG_INFINITY]);
        let sparse_result = c.apply_ffn_act(0, 1.0);
        let dense_result = c.apply_ffn_act(1, 1.0);
        let expected_gelu = gelu_approx(1.0);
        assert!((sparse_result - expected_gelu).abs() < 1e-6);
        assert!((dense_result - expected_gelu).abs() < 1e-6);
    }

    // ── Gemma 3n shared KV cache ────────────────────────────────────────────

    fn gemma3n_e2b_config() -> Llama3Config {
        // Approximate Gemma 3n E2B: 30 layers, shared_kv_layers=10 (→ 20 unique),
        // sliding_window_pattern with every 5th layer as full attention.
        let pattern: Vec<bool> = (0..30).map(|i| i % 5 != 4).collect();
        Llama3Config {
            arch: ModelArch::Gemma3n,
            num_layers: 30,
            sliding_window: Some(512),
            shared_kv_layers: Some(10),
            sliding_window_pattern: Some(pattern),
            ..Llama3Config::llama3_8b()
        }
    }

    #[test]
    fn test_kv_from_start_layers_non_gemma3n() {
        // Non-Gemma3n: no shared KV, kv_from_start = num_layers.
        let c = Llama3Config::llama3_8b();
        assert_eq!(c.kv_from_start_layers(), c.num_layers);
    }

    #[test]
    fn test_kv_from_start_layers_gemma3n_e2b() {
        // Gemma 3n E2B: 30 layers, shared 10 → 20 unique roots.
        let c = gemma3n_e2b_config();
        assert_eq!(c.kv_from_start_layers(), 20);
    }

    #[test]
    fn test_kv_source_layer_identity_for_roots() {
        // Layers 0..20 map to themselves (own KV).
        let c = gemma3n_e2b_config();
        for i in 0..20 {
            assert_eq!(c.kv_source_layer(i), i, "layer {i} should own its KV");
        }
    }

    #[test]
    fn test_kv_source_layer_shared_layers_gemma3n() {
        // Layer 20+ redirects: SWA → 18, full attention → 19.
        let c = gemma3n_e2b_config();
        // Layer 20: pattern[20] = true (SWA) → 18
        assert_eq!(c.kv_source_layer(20), 18, "layer 20 (SWA) should map to 18");
        // Layer 24: pattern[24] = false (full attention) → 19
        assert_eq!(c.kv_source_layer(24), 19, "layer 24 (full) should map to 19");
        // Layer 29: pattern[29] = false (full attention) → 19
        assert_eq!(c.kv_source_layer(29), 19, "layer 29 (full) should map to 19");
        // Layer 21-23: SWA → 18
        for i in 21..24 {
            assert_eq!(c.kv_source_layer(i), 18, "layer {i} (SWA) should map to 18");
        }
    }

    #[test]
    fn test_kv_source_layer_non_gemma3n_identity() {
        // Non-Gemma3n architectures return the layer unchanged.
        let c = Llama3Config::llama3_8b();
        for i in 0..c.num_layers {
            assert_eq!(c.kv_source_layer(i), i);
        }
    }

    #[test]
    fn test_build_kv_layer_map_gemma3n() {
        let c = gemma3n_e2b_config();
        let map = c.build_kv_layer_map();
        assert_eq!(map.len(), 30);
        // Identity for 0..20
        for (i, &m) in map.iter().enumerate().take(20) {
            assert_eq!(m, i);
        }
        // Remapped for 20..30
        assert_eq!(map[20], 18); // SWA
        assert_eq!(map[24], 19); // full
        assert_eq!(map[29], 19); // full
    }

    #[test]
    fn test_kv_cache_shared_read_uses_source_layer() {
        // Verify that reads from a shared layer return data written to its source.
        let mut cache = KvCache::new(30, 8, 4);
        // Install the Gemma3n E2B mapping.
        let config = gemma3n_e2b_config();
        cache.set_layer_map(config.build_kv_layer_map());

        // Write unique data at layer 18 (source for shared SWA layers).
        let k18 = vec![1.0, 2.0, 3.0, 4.0];
        let v18 = vec![10.0, 20.0, 30.0, 40.0];
        cache.append(18, &k18, &v18);
        // Layer 19 (source for shared full-attention layers)
        let k19 = vec![5.0, 6.0, 7.0, 8.0];
        let v19 = vec![50.0, 60.0, 70.0, 80.0];
        cache.append(19, &k19, &v19);
        // "Write" to layer 20 (SWA shared) — should be no-op.
        let sink_k = vec![99.0; 4];
        let sink_v = vec![99.0; 4];
        cache.append(20, &sink_k, &sink_v);
        cache.advance();

        // Reads from layer 20 (SWA shared) should return layer 18 data.
        assert_eq!(cache.key_at(20, 0), k18.as_slice());
        assert_eq!(cache.value_at(20, 0), v18.as_slice());
        // Reads from layer 24 (full attention shared) should return layer 19 data.
        assert_eq!(cache.key_at(24, 0), k19.as_slice());
        assert_eq!(cache.value_at(24, 0), v19.as_slice());
        // Layer 18 reads still return its own data (not overwritten).
        assert_eq!(cache.key_at(18, 0), k18.as_slice());
    }

    #[test]
    fn test_kv_cache_default_identity_map() {
        // Without set_layer_map, KvCache should use identity (backward compat).
        let mut cache = KvCache::new(3, 4, 2);
        let k0 = vec![1.0, 2.0];
        let k1 = vec![3.0, 4.0];
        let k2 = vec![5.0, 6.0];
        cache.append(0, &k0, &k0);
        cache.append(1, &k1, &k1);
        cache.append(2, &k2, &k2);
        cache.advance();
        assert_eq!(cache.key_at(0, 0), k0.as_slice());
        assert_eq!(cache.key_at(1, 0), k1.as_slice());
        assert_eq!(cache.key_at(2, 0), k2.as_slice());
    }

    // ── Q5_1 dequantization (Gemma 3n per_layer_token_embd) ─────────────────

    #[test]
    fn test_q5_1_dequant_zero_block() {
        // A single all-zero block should dequantize to all zeros.
        // Block layout (24 bytes): d f16, m f16, qh u32, qs [u8; 16]
        // With d=0 and m=0, every element becomes 0*(0..0) + 0 = 0.
        let data = vec![0u8; 24];
        let mut out = vec![0.0f32; 32];
        crate::gguf::dequantize_q5_1(&data, &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q5_1_dequant_constant_block() {
        // Set m = 1.0, d = 0, qs = [0; 16], qh = 0. Every element should be
        // 0 * 0 + 1.0 = 1.0.
        let mut data = vec![0u8; 24];
        // d = 0 (2 bytes), m = 1.0 as f16 (2 bytes)
        // f16 1.0 = 0x3C00
        data[2] = 0x00;
        data[3] = 0x3C;
        let mut out = vec![0.0f32; 32];
        crate::gguf::dequantize_q5_1(&data, &mut out);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-4, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_ggml_type_q5_1_block_layout() {
        use crate::gguf::GgmlType;
        assert_eq!(GgmlType::Q5_1.block_bytes(), 24);
        assert_eq!(GgmlType::Q5_1.elements_per_block(), 32);
    }
}
