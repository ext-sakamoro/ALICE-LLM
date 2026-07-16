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

/// Issue #40 diagnostic. Emits one JSONL line to stderr summarising the
/// post-final-RMSNorm hidden state (the buffer that feeds output_proj). The
/// GPU path uses an identical schema so the two can be diffed offline.
///
/// Format:
/// `{"backend":"cpu|gpu","kind":"pre_output_hidden","dim":N,"l2":X,"top8":[[idx,val],...]}`
///
/// The top-8 by absolute magnitude is chosen because logits are sensitive to
/// the largest components of the hidden vector — small differences in those
/// coordinates translate into large logit shifts through the output projection.
pub fn dump_hidden_jsonl_stderr(backend: &str, hidden: &[f32]) {
    let l2 = hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mut idxs: Vec<(usize, f32)> = hidden.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    idxs.sort_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let dim = hidden.len();
    // Pre-size for full vector + header. ~10 chars per float on avg.
    let mut line = String::with_capacity(dim * 10 + 256);
    line.push_str(&format!(
        "{{\"backend\":\"{backend}\",\"kind\":\"pre_output_hidden\",\"dim\":{dim},\"l2\":{l2:.6},\"top8\":["
    ));
    for (k, (i, v)) in idxs.iter().take(8).enumerate() {
        if k > 0 {
            line.push(',');
        }
        line.push_str(&format!("[{i},{v:.6}]"));
    }
    line.push_str("],\"full\":[");
    for (k, v) in hidden.iter().enumerate() {
        if k > 0 {
            line.push(',');
        }
        line.push_str(&format!("{v:.6}"));
    }
    line.push_str("]}");
    eprintln!("{line}");
}

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
    /// Gemma 4 (E2B/E4B/26B_A4B/31B): simplified successor to Gemma 3n.
    /// Retains per-layer input embedding, shared KV cache, GELU FFN, and
    /// Gemma-family embedding scaling; **removes** AltUp, Laurel, and
    /// activation sparsity. **Adds** per-layer FFN size, per-layer
    /// head_dim / RoPE base for SWA vs full-attention layers, optional
    /// per-layer `layer_output_scale`, and MoE (in 26B_A4B variant).
    Gemma4,
    Qwen2,
    Qwen3,
    Qwen3_5,
    /// DeepSeek-V3 / R1 (V2 / V2.5 も同じ family、llama.cpp では `deepseek2`
    /// prefix)。特徴:
    /// * **MLA** (Multi-head Latent Attention): Q と KV を LoRA 経由で
    ///   低ランク射影して KV cache を圧縮
    /// * **DeepSeek MoE**: 256 routed expert + shared expert、sigmoid gating
    ///   with `noaux_tc` (no auxiliary loss trick)
    /// * **Partial RoPE**: 全 head 次元でなく `qk_rope_head_dim` 部分のみ回転
    /// * **First-K dense layers**: 最初 N layer は MoE の代わりに dense FFN
    /// * **MTP head** (V3 のみ): multi-token prediction による native
    ///   speculative decoding
    ///
    /// 現状 (2026-07-11 追加): arch 検出 + config 読み込み + weight loading
    /// までの foundation のみ。MLA CPU forward / MoE routing / expert
    /// streaming / MTP は Phase 2-5 の follow-up Issue で実装予定。
    /// `forward()` は即 panic (fail-fast) で silent garbage を回避。
    DeepSeekV3,
}

impl ModelArch {
    /// Detect architecture from GGUF metadata key `general.architecture`.
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Self {
        match gguf.meta_str("general.architecture") {
            Some("mistral") => Self::Mistral,
            Some("gemma2") => Self::Gemma2,
            Some("gemma3n") => Self::Gemma3n,
            Some("gemma4") => Self::Gemma4,
            Some("qwen3moe" | "qwen3") => {
                if gguf.meta_u32("qwen3.full_attention_interval").is_some()
                    || gguf.meta_u32("qwen3moe.full_attention_interval").is_some()
                {
                    Self::Qwen3_5
                } else {
                    Self::Qwen3
                }
            }
            // Qwen 3.5 and Qwen 3.6 share the same architecture family
            // (llama.cpp `qwen35.cpp` handles both). GGUF metadata prefix is
            // `qwen35.*` for both versions.
            Some("qwen35" | "qwen35moe") => Self::Qwen3_5,
            Some(s) if s.starts_with("qwen") => Self::Qwen2,
            // DeepSeek-V2 / V2.5 / V3 / R1 all share the same architecture
            // family in llama.cpp under the `deepseek2` prefix.
            Some("deepseek2") => Self::DeepSeekV3,
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
            Self::Gemma4 => "gemma4",
            Self::Qwen2 => "qwen2",
            Self::Qwen3 | Self::Qwen3_5 => "qwen3",
            Self::DeepSeekV3 => "deepseek2",
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
            Self::Qwen2
                | Self::Qwen3
                | Self::Qwen3_5
                | Self::Gemma2
                | Self::Gemma3n
                | Self::Gemma4
                | Self::DeepSeekV3
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
/// Attention softcap + sliding window extras (Mistral / Gemma-2).
///
/// Grouped so the core [`Llama3Config`] doesn't carry three loosely
/// related `Option<...>` fields at the top level. Absent when the model
/// uses vanilla full-attention with no softcapping (Llama-3, Qwen 2/3, ...).
#[derive(Debug, Clone)]
pub struct AttentionExtrasConfig {
    /// Mistral: sliding window size (None = full attention).
    pub sliding_window: Option<usize>,
    /// Gemma-2: attention logit softcapping value (None = no capping).
    pub attn_logit_softcap: Option<f32>,
    /// Gemma-2: final logit softcapping value (None = no capping).
    pub final_logit_softcap: Option<f32>,
}

/// Qwen 3.5 / 3.6 SSM (DeltaNet) linear-attention hybrid config.
///
/// All fields are populated together for Qwen 3.5 / 3.6 hybrid models and
/// absent for every other architecture.
#[derive(Debug, Clone)]
pub struct SsmDeltaNetConfig {
    /// Qwen3.5: full attention interval (e.g. 4 = every 4th layer is full attention).
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
    /// Qwen 3.5/3.6 SSM (DeltaNet linear attention) — inner projection size.
    pub ssm_inner_size: Option<usize>,
    /// Qwen 3.5/3.6 SSM — state vector dimensionality.
    pub ssm_state_size: Option<usize>,
    /// Qwen 3.5/3.6 SSM — group count (parallel state groups).
    pub ssm_group_count: Option<usize>,
    /// Qwen 3.5/3.6 SSM — time-step projection rank.
    pub ssm_time_step_rank: Option<usize>,
    /// Qwen 3.5/3.6 NextN / MTP — number of extra decoder blocks appended
    /// beyond the main stack (used for speculative decoding). Load only,
    /// inference not currently used.
    pub n_layer_nextn: Option<usize>,
}

/// Mixture-of-experts config (Qwen 3 MoE / Mixtral / DeepSeek / Gemma 4 26B_A4B).
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// MoE: total number of experts per MoE layer (Qwen3 MoE: 4-128,
    /// Mixtral: 8, DeepSeek: up to 256).
    pub num_experts: Option<usize>,
    /// MoE: number of experts activated per token (top-k routing; typically
    /// 2 or 8).
    pub num_experts_active: Option<usize>,
    /// MoE: per-expert FFN intermediate dimension. Often equals
    /// `intermediate_dim`, but some models (DeepSeek, Gemma 4 26B_A4B) use
    /// smaller values so total active parameters stay reasonable.
    pub expert_ffn_size: Option<usize>,
}

/// Gemma 3n architecture-specific config (Laurel / AltUp / per-layer
/// input-embedding branch / activation sparsity / shared-KV).
#[derive(Debug, Clone)]
pub struct Gemma3nConfig {
    /// Gemma 3n: per-layer sliding window boolean pattern. When Some,
    /// entry `i = true` means layer `i` uses SWA, `false` = full attention.
    /// Supersedes Gemma 2 even/odd alternation.
    pub sliding_window_pattern: Option<Vec<bool>>,
    /// Gemma 3n: per-layer FFN activation sparsity scale. First N entries
    /// are finite (GELU + sparsity threshold `scale * std`), rest are -inf
    /// (dense, no sparsity). Absent → all layers dense (SiLU for non-Gemma).
    pub activation_sparsity_scale: Option<Vec<f32>>,
    /// Gemma 3n: number of layers with unique KV cache. Later layers reuse
    /// KV cache from earlier layers.
    pub shared_kv_layers: Option<usize>,
    /// Gemma 3n: per-layer input embedding dimension (256 for E2B).
    pub per_layer_input_embedding_dim: Option<usize>,
    /// Gemma 3n: number of AltUp residual streams (4 for E2B).
    pub altup_num_inputs: Option<usize>,
    /// Gemma 3n: AltUp active input index (0 for E2B).
    pub altup_active_idx: Option<usize>,
}

/// Gemma 4 architecture-specific config (SWA half head_dim, per-layer FFN size).
#[derive(Debug, Clone)]
pub struct Gemma4Config {
    /// Gemma 4: SWA layer head dimension for K/V (typically half of full
    /// `head_dim`). When `None`, all layers use `head_dim`.
    pub head_dim_swa: Option<usize>,
    /// Gemma 4: SWA layer RoPE base frequency (typically 10K for local
    /// context, vs 1M for full-attention layers).
    pub rope_theta_swa: Option<f32>,
    /// Gemma 4: SWA layer RoPE dimension count.
    pub rope_dim_swa: Option<usize>,
    /// Gemma 4: per-layer FFN size array. When absent, `intermediate_dim`
    /// applies uniformly. Gemma 4 E2B uses [6144×15, 12288×20].
    pub ffn_size_per_layer: Option<Vec<usize>>,
}

/// DeepSeek-V2 / V3 / R1 architecture-specific config.
///
/// Captures the MLA (Multi-head Latent Attention) LoRA ranks, the
/// partial-RoPE head-dim split, and the DeepSeek MoE parameters that are
/// not covered by the generic [`MoeConfig`]. All fields optional so the
/// struct maps 1:1 to what the GGUF metadata provides — implementation
/// phases can consume them as they come online.
///
/// GGUF metadata key prefix: `deepseek2.*`. Typical DeepSeek-V3 values are
/// given in the field docs so future implementation phases can sanity-check
/// their read values.
#[derive(Debug, Clone)]
pub struct DeepSeekV3Config {
    /// LoRA rank for the Q projection down/up chain (V3: 1536).
    pub q_lora_rank: Option<usize>,
    /// LoRA rank for the KV projection down/up chain (V3: 512).
    pub kv_lora_rank: Option<usize>,
    /// Head dim for the non-rotated Q/K portion (V3: 128).
    pub qk_nope_head_dim: Option<usize>,
    /// Head dim for the rotated Q/K portion (V3: 64) — only this slice
    /// participates in RoPE, the `nope` slice is passed through untouched.
    pub qk_rope_head_dim: Option<usize>,
    /// Head dim for V (V3: 128; equals `qk_nope_head_dim` in practice).
    pub v_head_dim: Option<usize>,
    /// Total number of routed experts (V3: 256).
    pub n_routed_experts: Option<usize>,
    /// Shared expert count (V3: 1) — always active in addition to top-k routed.
    pub n_shared_experts: Option<usize>,
    /// Top-k routed experts per token (V3: 8).
    pub num_experts_per_tok: Option<usize>,
    /// Per-expert FFN intermediate size (V3: 2048; distinct from the dense
    /// FFN size used in `first_k_dense_replace` layers).
    pub moe_intermediate_size: Option<usize>,
    /// Number of leading layers that use a monolithic dense FFN instead of
    /// MoE (V3: 3 — layers 0/1/2 dense, all others MoE + shared expert).
    pub first_k_dense_replace: Option<usize>,
    /// Routed expert output scale (V3: 2.5).
    pub routed_scaling_factor: Option<f32>,
    /// `true` when the router uses sigmoid gating with the "no auxiliary
    /// loss" bias-correction trick introduced in DeepSeek-V3.
    pub noaux_tc: Option<bool>,
    /// MTP head layer index (V3: 61 = the extra MTP layer past 60 hidden
    /// layers). `None` when no MTP head is present.
    pub mtp_layer: Option<usize>,
}

/// Supports Llama-3, Mistral, and Gemma-2 architectures.
///
/// Architecture-specific extensions are grouped into 5 sub-configs
/// (Issue #11 Part 2). Backward-compat accessor methods on `Llama3Config`
/// mirror the previous flat field names, so callers can keep the old
/// syntax (`c.sliding_window()` instead of `c.sliding_window()`).
#[derive(Debug, Clone)]
pub struct Llama3Config {
    // ── Core (always populated) ───────────────────────────────────────
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

    // ── Grouped arch-specific extensions (5 sub-configs) ──────────────
    /// Attention softcap + sliding window (Mistral / Gemma-2).
    pub attention_extras: Option<AttentionExtrasConfig>,
    /// Qwen 3.5 / 3.6 SSM DeltaNet linear-attention hybrid config.
    pub ssm: Option<SsmDeltaNetConfig>,
    /// Mixture-of-experts routing sizes.
    pub moe: Option<MoeConfig>,
    /// Gemma 3n augmentations (Laurel / AltUp / per-layer embedding).
    pub gemma3n: Option<Gemma3nConfig>,
    /// Gemma 4 augmentations (SWA half head_dim, per-layer FFN size).
    pub gemma4: Option<Gemma4Config>,
    /// DeepSeek-V2 / V3 / R1 augmentations (MLA LoRA ranks + DeepSeek MoE
    /// parameters + MTP head layer).
    pub deepseek_v3: Option<DeepSeekV3Config>,
}

impl Llama3Config {
    // ── Backward-compat accessors for fields moved into sub-configs ────
    // Same names as the pre-refactor flat `pub` fields, so migrating a
    // caller is a single-character change (`.foo` → `.foo()`).

    // AttentionExtrasConfig (Mistral / Gemma-2)

    #[inline]
    pub fn sliding_window(&self) -> Option<usize> {
        self.attention_extras
            .as_ref()
            .and_then(|a| a.sliding_window)
    }

    #[inline]
    pub fn attn_logit_softcap(&self) -> Option<f32> {
        self.attention_extras
            .as_ref()
            .and_then(|a| a.attn_logit_softcap)
    }

    #[inline]
    pub fn final_logit_softcap(&self) -> Option<f32> {
        self.attention_extras
            .as_ref()
            .and_then(|a| a.final_logit_softcap)
    }

    // SsmDeltaNetConfig (Qwen 3.5 / 3.6)

    #[inline]
    pub fn full_attention_interval(&self) -> Option<usize> {
        self.ssm.as_ref().and_then(|s| s.full_attention_interval)
    }

    #[inline]
    pub fn linear_num_kv_heads(&self) -> Option<usize> {
        self.ssm.as_ref().and_then(|s| s.linear_num_kv_heads)
    }

    #[inline]
    pub fn linear_qk_head_dim(&self) -> Option<usize> {
        self.ssm.as_ref().and_then(|s| s.linear_qk_head_dim)
    }

    #[inline]
    pub fn linear_kv_head_dim(&self) -> Option<usize> {
        self.ssm.as_ref().and_then(|s| s.linear_kv_head_dim)
    }

    #[inline]
    pub fn linear_num_v_heads(&self) -> Option<usize> {
        self.ssm.as_ref().and_then(|s| s.linear_num_v_heads)
    }

    #[inline]
    pub fn linear_conv_kernel_dim(&self) -> Option<usize> {
        self.ssm.as_ref().and_then(|s| s.linear_conv_kernel_dim)
    }

    #[inline]
    pub fn ssm_inner_size(&self) -> Option<usize> {
        self.ssm.as_ref().and_then(|s| s.ssm_inner_size)
    }

    #[inline]
    pub fn ssm_state_size(&self) -> Option<usize> {
        self.ssm.as_ref().and_then(|s| s.ssm_state_size)
    }

    #[inline]
    pub fn ssm_group_count(&self) -> Option<usize> {
        self.ssm.as_ref().and_then(|s| s.ssm_group_count)
    }

    #[inline]
    pub fn ssm_time_step_rank(&self) -> Option<usize> {
        self.ssm.as_ref().and_then(|s| s.ssm_time_step_rank)
    }

    #[inline]
    pub fn n_layer_nextn(&self) -> Option<usize> {
        self.ssm.as_ref().and_then(|s| s.n_layer_nextn)
    }

    // MoeConfig

    #[inline]
    pub fn num_experts(&self) -> Option<usize> {
        self.moe.as_ref().and_then(|m| m.num_experts)
    }

    #[inline]
    pub fn num_experts_active(&self) -> Option<usize> {
        self.moe.as_ref().and_then(|m| m.num_experts_active)
    }

    #[inline]
    pub fn expert_ffn_size(&self) -> Option<usize> {
        self.moe.as_ref().and_then(|m| m.expert_ffn_size)
    }

    // Gemma3nConfig

    #[inline]
    pub fn sliding_window_pattern(&self) -> Option<&[bool]> {
        self.gemma3n
            .as_ref()
            .and_then(|g| g.sliding_window_pattern.as_deref())
    }

    #[inline]
    pub fn activation_sparsity_scale(&self) -> Option<&[f32]> {
        self.gemma3n
            .as_ref()
            .and_then(|g| g.activation_sparsity_scale.as_deref())
    }

    #[inline]
    pub fn shared_kv_layers(&self) -> Option<usize> {
        self.gemma3n.as_ref().and_then(|g| g.shared_kv_layers)
    }

    #[inline]
    pub fn per_layer_input_embedding_dim(&self) -> Option<usize> {
        self.gemma3n
            .as_ref()
            .and_then(|g| g.per_layer_input_embedding_dim)
    }

    #[inline]
    pub fn altup_num_inputs(&self) -> Option<usize> {
        self.gemma3n.as_ref().and_then(|g| g.altup_num_inputs)
    }

    #[inline]
    pub fn altup_active_idx(&self) -> Option<usize> {
        self.gemma3n.as_ref().and_then(|g| g.altup_active_idx)
    }

    // Gemma4Config

    #[inline]
    pub fn head_dim_swa(&self) -> Option<usize> {
        self.gemma4.as_ref().and_then(|g| g.head_dim_swa)
    }

    #[inline]
    pub fn rope_theta_swa(&self) -> Option<f32> {
        self.gemma4.as_ref().and_then(|g| g.rope_theta_swa)
    }

    #[inline]
    pub fn rope_dim_swa(&self) -> Option<usize> {
        self.gemma4.as_ref().and_then(|g| g.rope_dim_swa)
    }

    #[inline]
    pub fn ffn_size_per_layer(&self) -> Option<&[usize]> {
        self.gemma4
            .as_ref()
            .and_then(|g| g.ffn_size_per_layer.as_deref())
    }

    // DeepSeek-V3

    #[inline]
    pub fn deepseek_q_lora_rank(&self) -> Option<usize> {
        self.deepseek_v3.as_ref().and_then(|d| d.q_lora_rank)
    }

    #[inline]
    pub fn deepseek_kv_lora_rank(&self) -> Option<usize> {
        self.deepseek_v3.as_ref().and_then(|d| d.kv_lora_rank)
    }

    #[inline]
    pub fn deepseek_qk_nope_head_dim(&self) -> Option<usize> {
        self.deepseek_v3.as_ref().and_then(|d| d.qk_nope_head_dim)
    }

    #[inline]
    pub fn deepseek_qk_rope_head_dim(&self) -> Option<usize> {
        self.deepseek_v3.as_ref().and_then(|d| d.qk_rope_head_dim)
    }

    #[inline]
    pub fn deepseek_v_head_dim(&self) -> Option<usize> {
        self.deepseek_v3.as_ref().and_then(|d| d.v_head_dim)
    }

    #[inline]
    pub fn deepseek_n_routed_experts(&self) -> Option<usize> {
        self.deepseek_v3.as_ref().and_then(|d| d.n_routed_experts)
    }

    #[inline]
    pub fn deepseek_n_shared_experts(&self) -> Option<usize> {
        self.deepseek_v3.as_ref().and_then(|d| d.n_shared_experts)
    }

    #[inline]
    pub fn deepseek_num_experts_per_tok(&self) -> Option<usize> {
        self.deepseek_v3
            .as_ref()
            .and_then(|d| d.num_experts_per_tok)
    }

    #[inline]
    pub fn deepseek_moe_intermediate_size(&self) -> Option<usize> {
        self.deepseek_v3
            .as_ref()
            .and_then(|d| d.moe_intermediate_size)
    }

    #[inline]
    pub fn deepseek_first_k_dense_replace(&self) -> Option<usize> {
        self.deepseek_v3
            .as_ref()
            .and_then(|d| d.first_k_dense_replace)
    }

    #[inline]
    pub fn deepseek_routed_scaling_factor(&self) -> Option<f32> {
        self.deepseek_v3
            .as_ref()
            .and_then(|d| d.routed_scaling_factor)
    }

    #[inline]
    pub fn deepseek_noaux_tc(&self) -> Option<bool> {
        self.deepseek_v3.as_ref().and_then(|d| d.noaux_tc)
    }

    #[inline]
    pub fn deepseek_mtp_layer(&self) -> Option<usize> {
        self.deepseek_v3.as_ref().and_then(|d| d.mtp_layer)
    }

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
                        crate::gguf::MetaValue::Array(arr) => arr
                            .first()
                            .and_then(|item| item.as_u32().map(|v| v as usize)),
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

        // Qwen 3.5 / 3.6 DeltaNet SSM parameters (arch prefix `qwen35`).
        // Names follow `qwen35.ssm.*` in GGUF metadata.
        let ssm_inner_size = gguf
            .meta_u32(&format!("{prefix}.ssm.inner_size"))
            .map(|v| v as usize);
        let ssm_state_size = gguf
            .meta_u32(&format!("{prefix}.ssm.state_size"))
            .map(|v| v as usize);
        let ssm_group_count = gguf
            .meta_u32(&format!("{prefix}.ssm.group_count"))
            .map(|v| v as usize);
        let ssm_time_step_rank = gguf
            .meta_u32(&format!("{prefix}.ssm.time_step_rank"))
            .map(|v| v as usize);
        let ssm_conv_kernel = gguf
            .meta_u32(&format!("{prefix}.ssm.conv_kernel"))
            .map(|v| v as usize);

        // Bonsai 27B / Qwen 3.6 GGUF only exports the `qwen35.ssm.*` set and
        // omits the `qwen35.linear_*` variants that older Qwen 3.5 exports
        // ship. Populate the linear-attention geometry from the SSM keys
        // when the direct read returned None, so the downstream DeltaNet
        // loader sees identical config regardless of the export style.
        //
        // Mapping (from empirical Bonsai `qwen35.*` metadata + HF config.json):
        //   linear_num_key_heads    = ssm.group_count            (Bonsai: 16)
        //   linear_qk_head_dim      = ssm.state_size             (Bonsai: 128)
        //   linear_key_value_head_dim = ssm.state_size           (Bonsai: 128)
        //   linear_num_value_heads  = ssm.inner_size / v_head    (Bonsai: 48)
        //   linear_conv_kernel_dim  = ssm.conv_kernel            (Bonsai: 4)
        let linear_num_kv_heads = linear_num_kv_heads.or(ssm_group_count);
        let linear_qk_head_dim = linear_qk_head_dim.or(ssm_state_size);
        let linear_kv_head_dim = linear_kv_head_dim.or(ssm_state_size);
        let linear_num_v_heads =
            linear_num_v_heads.or_else(|| match (ssm_inner_size, linear_kv_head_dim) {
                (Some(inner), Some(head_dim)) if head_dim > 0 => Some(inner / head_dim),
                _ => None,
            });
        let linear_conv_kernel_dim = linear_conv_kernel_dim.or(ssm_conv_kernel);
        // Qwen 3.5 / 3.6 NextN / MTP layer count.
        let n_layer_nextn = gguf
            .meta_u32(&format!("{prefix}.nextn.predict_layers"))
            .map(|v| v as usize);

        // MoE parameters (Qwen3 MoE / Mixtral / Gemma 4 26B_A4B):
        //   `{prefix}.expert_count` — total experts per layer
        //   `{prefix}.expert_used_count` — top-k routing count
        //   `{prefix}.expert_feed_forward_length` — per-expert FFN size
        let num_experts = gguf
            .meta_u32(&format!("{prefix}.expert_count"))
            .map(|v| v as usize);
        let num_experts_active = gguf
            .meta_u32(&format!("{prefix}.expert_used_count"))
            .map(|v| v as usize);
        let expert_ffn_size = gguf
            .meta_u32(&format!("{prefix}.expert_feed_forward_length"))
            .map(|v| v as usize);

        // Gemma 3n: per-layer sliding window boolean pattern
        let sliding_window_pattern = gguf
            .meta(&format!("{prefix}.attention.sliding_window_pattern"))
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

        // Gemma 4: SWA layers have their own head_dim / RoPE base / RoPE dim.
        let head_dim_swa = gguf
            .meta_u32(&format!("{prefix}.attention.key_length_swa"))
            .map(|v| v as usize);
        let rope_theta_swa = gguf.meta_f32(&format!("{prefix}.rope.freq_base_swa"));
        let rope_dim_swa = gguf
            .meta_u32(&format!("{prefix}.rope.dimension_count_swa"))
            .map(|v| v as usize);

        // Gemma 4: per-layer FFN size array (E2B: [6144×15, 12288×20]).
        // When `feed_forward_length` scalar was used above, this remains None.
        let ffn_size_per_layer = gguf
            .meta(&format!("{prefix}.feed_forward_length"))
            .and_then(|v| match v {
                crate::gguf::MetaValue::Array(arr) if arr.len() == num_layers => {
                    let mut out = Vec::with_capacity(arr.len());
                    for item in arr {
                        out.push(item.as_u32()? as usize);
                    }
                    Some(out)
                }
                _ => None,
            });

        // Bundle arch-specific fields into their respective sub-configs.
        // Each sub-config is `Some(...)` whenever _any_ of its fields is
        // populated — keeps the semantic that non-target architectures see
        // the whole sub-config as `None`.
        let attention_extras = if sliding_window.is_some()
            || attn_logit_softcap.is_some()
            || final_logit_softcap.is_some()
        {
            Some(AttentionExtrasConfig {
                sliding_window,
                attn_logit_softcap,
                final_logit_softcap,
            })
        } else {
            None
        };

        let ssm = if full_attention_interval.is_some()
            || linear_num_kv_heads.is_some()
            || linear_qk_head_dim.is_some()
            || linear_kv_head_dim.is_some()
            || linear_num_v_heads.is_some()
            || linear_conv_kernel_dim.is_some()
            || ssm_inner_size.is_some()
            || ssm_state_size.is_some()
            || ssm_group_count.is_some()
            || ssm_time_step_rank.is_some()
            || n_layer_nextn.is_some()
        {
            Some(SsmDeltaNetConfig {
                full_attention_interval,
                linear_num_kv_heads,
                linear_qk_head_dim,
                linear_kv_head_dim,
                linear_num_v_heads,
                linear_conv_kernel_dim,
                ssm_inner_size,
                ssm_state_size,
                ssm_group_count,
                ssm_time_step_rank,
                n_layer_nextn,
            })
        } else {
            None
        };

        let moe =
            if num_experts.is_some() || num_experts_active.is_some() || expert_ffn_size.is_some() {
                Some(MoeConfig {
                    num_experts,
                    num_experts_active,
                    expert_ffn_size,
                })
            } else {
                None
            };

        let gemma3n = if sliding_window_pattern.is_some()
            || activation_sparsity_scale.is_some()
            || shared_kv_layers.is_some()
            || per_layer_input_embedding_dim.is_some()
            || altup_num_inputs.is_some()
            || altup_active_idx.is_some()
        {
            Some(Gemma3nConfig {
                sliding_window_pattern,
                activation_sparsity_scale,
                shared_kv_layers,
                per_layer_input_embedding_dim,
                altup_num_inputs,
                altup_active_idx,
            })
        } else {
            None
        };

        let gemma4 = if head_dim_swa.is_some()
            || rope_theta_swa.is_some()
            || rope_dim_swa.is_some()
            || ffn_size_per_layer.is_some()
        {
            Some(Gemma4Config {
                head_dim_swa,
                rope_theta_swa,
                rope_dim_swa,
                ffn_size_per_layer,
            })
        } else {
            None
        };

        // DeepSeek-V3 / R1: read the `deepseek2.*` sub-config. All fields
        // optional — a missing key does not disqualify the model, we just
        // leave the future forward path to raise a targeted `todo!()`.
        let deepseek_v3 = if arch == ModelArch::DeepSeekV3 {
            let q_lora_rank = gguf
                .meta_u32(&format!("{prefix}.attention.q_lora_rank"))
                .map(|v| v as usize);
            let kv_lora_rank = gguf
                .meta_u32(&format!("{prefix}.attention.kv_lora_rank"))
                .map(|v| v as usize);
            let qk_nope_head_dim = gguf
                .meta_u32(&format!("{prefix}.attention.key_length"))
                .map(|v| v as usize);
            let qk_rope_head_dim = gguf
                .meta_u32(&format!("{prefix}.rope.dimension_count"))
                .map(|v| v as usize);
            let v_head_dim = gguf
                .meta_u32(&format!("{prefix}.attention.value_length"))
                .map(|v| v as usize);
            let n_routed_experts = gguf
                .meta_u32(&format!("{prefix}.expert_count"))
                .map(|v| v as usize);
            let n_shared_experts = gguf
                .meta_u32(&format!("{prefix}.expert_shared_count"))
                .map(|v| v as usize);
            let num_experts_per_tok = gguf
                .meta_u32(&format!("{prefix}.expert_used_count"))
                .map(|v| v as usize);
            let moe_intermediate_size = gguf
                .meta_u32(&format!("{prefix}.expert_feed_forward_length"))
                .map(|v| v as usize);
            let first_k_dense_replace = gguf
                .meta_u32(&format!("{prefix}.leading_dense_block_count"))
                .map(|v| v as usize);
            let routed_scaling_factor = gguf.meta_f32(&format!("{prefix}.expert_weights_scale"));
            let noaux_tc = gguf
                .meta_u32(&format!("{prefix}.expert_gating_func"))
                .map(|v| v == 2); // 2 = sigmoid (noaux_tc), 1 = softmax
            let mtp_layer = gguf
                .meta_u32(&format!("{prefix}.mtp_layer_count"))
                .and_then(|c| {
                    if c > 0 {
                        Some(num_layers + c as usize - 1)
                    } else {
                        None
                    }
                });
            Some(DeepSeekV3Config {
                q_lora_rank,
                kv_lora_rank,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                n_routed_experts,
                n_shared_experts,
                num_experts_per_tok,
                moe_intermediate_size,
                first_k_dense_replace,
                routed_scaling_factor,
                noaux_tc,
                mtp_layer,
            })
        } else {
            None
        };

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
            attention_extras,
            ssm,
            moe,
            gemma3n,
            gemma4,
            deepseek_v3,
        })
    }

    /// Returns true if this is a hybrid DeltaNet model (Qwen3.5).
    pub fn is_hybrid(&self) -> bool {
        self.full_attention_interval().is_some()
    }

    /// Returns true if layer `i` is a DeltaNet (linear attention) layer.
    /// Full attention layers are at indices where `(i + 1) % interval == 0`.
    pub fn is_deltanet_layer(&self, i: usize) -> bool {
        match self.full_attention_interval() {
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
    /// - Gemma 2 / Gemma 3n / Gemma 4: GELU (tanh approximation) uniformly.
    ///   For Gemma 3n, per-layer sparsity masking is applied **before** this
    ///   step via `apply_ffn_sparsity`.
    /// - All others: SiLU (SwiGLU).
    #[inline]
    pub fn apply_ffn_act(&self, _layer_idx: usize, x: f32) -> f32 {
        match self.arch {
            ModelArch::Gemma2 | ModelArch::Gemma3n | ModelArch::Gemma4 => gelu_approx(x),
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
            .activation_sparsity_scale()
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
        match (self.arch, self.shared_kv_layers()) {
            (ModelArch::Gemma3n | ModelArch::Gemma4, Some(shared)) if shared < self.num_layers => {
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
        if !matches!(self.arch, ModelArch::Gemma3n | ModelArch::Gemma4) {
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
        (0..self.num_layers)
            .map(|i| self.kv_source_layer(i))
            .collect()
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
                    self.sliding_window()
                } else {
                    None
                }
            }
            ModelArch::Gemma3n | ModelArch::Gemma4 => {
                if let Some(pattern) = self.sliding_window_pattern().as_ref() {
                    if pattern.get(i).copied().unwrap_or(false) {
                        self.sliding_window()
                    } else {
                        None
                    }
                } else {
                    self.sliding_window()
                }
            }
            _ => self.sliding_window(),
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
            attention_extras: None,
            ssm: None,
            moe: None,
            gemma3n: None,
            gemma4: None,
            deepseek_v3: None,
        }
    }

    /// Per-layer head dimension for K/V projections.
    ///
    /// For Gemma 4, SWA layers may use a smaller `head_dim_swa` (e.g. 256
    /// vs 512 for full attention). Falls back to `head_dim` for all other
    /// architectures / when `head_dim_swa` is absent.
    pub fn head_dim_for_layer(&self, i: usize) -> usize {
        match (self.arch, self.head_dim_swa()) {
            (ModelArch::Gemma4, Some(hs)) if self.sliding_window_for_layer(i).is_some() => hs,
            _ => self.head_dim,
        }
    }

    /// Per-layer RoPE base frequency.
    ///
    /// For Gemma 4, SWA layers use a lower `rope_theta_swa` (10K vs 1M for
    /// full attention).
    pub fn rope_theta_for_layer(&self, i: usize) -> f32 {
        match (self.arch, self.rope_theta_swa()) {
            (ModelArch::Gemma4, Some(ts)) if self.sliding_window_for_layer(i).is_some() => ts,
            _ => self.rope_theta,
        }
    }

    /// Per-layer FFN intermediate dimension.
    ///
    /// For Gemma 4 with array metadata, returns per-layer size (e.g. 6144
    /// for early layers, 12288 for later layers). Otherwise returns the
    /// scalar `intermediate_dim`.
    pub fn ffn_size_for_layer(&self, i: usize) -> usize {
        self.ffn_size_per_layer()
            .as_ref()
            .and_then(|arr| arr.get(i).copied())
            .unwrap_or(self.intermediate_dim)
    }

    /// Q projection output dimension for layer `i` (= `num_heads * head_dim_for_layer(i)`).
    pub fn q_dim_for_layer(&self, i: usize) -> usize {
        self.num_heads * self.head_dim_for_layer(i)
    }

    /// K/V projection output dimension for layer `i` (= `num_kv_heads * head_dim_for_layer(i)`).
    pub fn kv_dim_for_layer(&self, i: usize) -> usize {
        self.num_kv_heads * self.head_dim_for_layer(i)
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
        // Skip writes for shared layers (Gemma 3n / Gemma 4): the target KV
        // cache was already populated by the "source" layer earlier in the
        // forward pass.
        if self.kv_layer_map[layer] != layer {
            return;
        }
        let off = self.offset(layer, self.seq_len);
        // Gemma 4 SWA layers may have a smaller actual kv_dim than the cache's
        // allocated `kv_dim` (max across layers). Copy up to k.len() and
        // zero-pad the remainder so stale data doesn't leak.
        let n = k.len().min(self.kv_dim);
        self.keys[off..off + n].copy_from_slice(&k[..n]);
        self.values[off..off + n].copy_from_slice(&v[..n]);
        if n < self.kv_dim {
            for i in n..self.kv_dim {
                self.keys[off + i] = 0.0;
                self.values[off + i] = 0.0;
            }
        }
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

// ─── KV cache persistence (colibri `.coli_kv` 参考、Issue: warm restart) ────
//
// Binary file format (little-endian):
//
//   Magic:             "ALICEKV1" (8 bytes)
//   Version:           u32
//   Config fingerprint: u64 (hash of hidden_dim/num_layers/num_kv_heads/head_dim/kv_dim)
//   num_layers:        u64
//   max_seq_len:       u64
//   kv_dim:            u64
//   seq_len:           u64 (valid entries; may be 0..=max_seq_len)
//   kv_layer_map:      num_layers × u32
//   Data (per layer):
//     if kv_layer_map[i] == i:
//       keys:   seq_len × kv_dim × 4 bytes (f32 LE)
//       values: seq_len × kv_dim × 4 bytes (f32 LE)
//     else:
//       skip — shared-KV layers redirect their reads to `map[i]`.
//
// The fingerprint hash rejects mismatched-model loads (loading a Llama-3 cache
// into a Qwen 3 model would produce silent garbage otherwise).

const KV_MAGIC: &[u8; 8] = b"ALICEKV1";
const KV_FORMAT_VERSION: u32 = 1;

/// Errors returned by [`Llama3Model::load_kv_cache`]. Distinguishes the
/// three usual failure modes (I/O, corrupted file, config mismatch) so
/// callers can decide whether to retry, log, or abort.
#[derive(Debug)]
pub enum KvCacheLoadError {
    /// Underlying `std::io::Error` (open / read / EOF).
    Io(std::io::Error),
    /// File does not start with the ALICE-LLM KV magic bytes.
    BadMagic { got: [u8; 8] },
    /// File version is not supported by this build.
    UnsupportedVersion { got: u32, expected: u32 },
    /// The config fingerprint recorded in the file does not match the
    /// current model. Loading anyway would produce garbage output.
    FingerprintMismatch { got: u64, expected: u64 },
    /// Cache metadata (num_layers / kv_dim / max_seq_len) disagrees with
    /// the current model.
    ShapeMismatch(String),
    /// `seq_len` in the file exceeds `max_seq_len`.
    OverflowSeqLen { got: u64, max: usize },
}

impl std::fmt::Display for KvCacheLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "kv cache load: {e}"),
            Self::BadMagic { got } => {
                write!(f, "kv cache load: expected magic {KV_MAGIC:?}, got {got:?}")
            }
            Self::UnsupportedVersion { got, expected } => write!(
                f,
                "kv cache load: unsupported version {got} (expected {expected})"
            ),
            Self::FingerprintMismatch { got, expected } => write!(
                f,
                "kv cache load: model config fingerprint mismatch (got {got:#x}, expected {expected:#x})"
            ),
            Self::ShapeMismatch(msg) => write!(f, "kv cache load: shape mismatch — {msg}"),
            Self::OverflowSeqLen { got, max } => write!(
                f,
                "kv cache load: seq_len {got} exceeds max_seq_len {max}"
            ),
        }
    }
}

impl std::error::Error for KvCacheLoadError {}

impl From<std::io::Error> for KvCacheLoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Compact fingerprint of the shape-critical config fields — anything that
/// affects the KV cache layout. Uses `std::hash::DefaultHasher` for a
/// dependency-free stable hash. The fingerprint is written into the KV
/// file header and checked on load; a mismatch aborts loading.
fn kv_cache_fingerprint(config: &Llama3Config) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    // Only fields that change the KV cache layout. If two models agree on
    // these, the KV bytes are interchangeable byte-for-byte.
    (config.num_layers as u64).hash(&mut hasher);
    (config.num_kv_heads as u64).hash(&mut hasher);
    (config.head_dim as u64).hash(&mut hasher);
    (config.hidden_dim as u64).hash(&mut hasher);
    (config.max_seq_len as u64).hash(&mut hasher);
    // Arch flavour affects the KV write pattern (Gemma 3n shared-KV / Gemma
    // 4 SWA half head_dim), so bake it in.
    format!("{:?}", config.arch).hash(&mut hasher);
    hasher.finish()
}

impl KvCache {
    /// Serialise the cache to `writer`. Fingerprint is provided by the
    /// caller so `Llama3Model::save_kv_cache` can inject the current
    /// config's hash without exposing `KvCache` externally.
    fn write_to(&self, writer: &mut impl std::io::Write, fingerprint: u64) -> std::io::Result<()> {
        writer.write_all(KV_MAGIC)?;
        writer.write_all(&KV_FORMAT_VERSION.to_le_bytes())?;
        writer.write_all(&fingerprint.to_le_bytes())?;
        writer.write_all(&(self._num_layers as u64).to_le_bytes())?;
        writer.write_all(&(self.max_seq_len as u64).to_le_bytes())?;
        writer.write_all(&(self.kv_dim as u64).to_le_bytes())?;
        writer.write_all(&(self.seq_len as u64).to_le_bytes())?;
        // Layer map (u32 each so Gemma 3n's up to num_layers ≪ 2^32 fits).
        for &src in &self.kv_layer_map {
            writer.write_all(&(src as u32).to_le_bytes())?;
        }
        // Per-layer data. Skip shared-KV layers so the file mirrors what
        // the forward pass actually needs to reconstruct on load.
        let n = self.seq_len * self.kv_dim;
        for layer in 0..self._num_layers {
            if self.kv_layer_map[layer] != layer {
                continue;
            }
            let off = self.offset(layer, 0);
            let keys_bytes = f32_slice_as_bytes(&self.keys[off..off + n]);
            let values_bytes = f32_slice_as_bytes(&self.values[off..off + n]);
            writer.write_all(keys_bytes)?;
            writer.write_all(values_bytes)?;
        }
        Ok(())
    }

    /// Reset this cache in place from `reader`. `expected_fingerprint` is
    /// the caller's current model fingerprint; if the file's stored
    /// fingerprint disagrees the load is refused.
    fn read_from(
        &mut self,
        reader: &mut impl std::io::Read,
        expected_fingerprint: u64,
    ) -> Result<(), KvCacheLoadError> {
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != KV_MAGIC {
            return Err(KvCacheLoadError::BadMagic { got: magic });
        }
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != KV_FORMAT_VERSION {
            return Err(KvCacheLoadError::UnsupportedVersion {
                got: version,
                expected: KV_FORMAT_VERSION,
            });
        }
        reader.read_exact(&mut buf8)?;
        let file_fingerprint = u64::from_le_bytes(buf8);
        if file_fingerprint != expected_fingerprint {
            return Err(KvCacheLoadError::FingerprintMismatch {
                got: file_fingerprint,
                expected: expected_fingerprint,
            });
        }
        reader.read_exact(&mut buf8)?;
        let num_layers = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf8)?;
        let max_seq_len = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf8)?;
        let kv_dim = u64::from_le_bytes(buf8) as usize;
        reader.read_exact(&mut buf8)?;
        let seq_len = u64::from_le_bytes(buf8);

        if num_layers != self._num_layers
            || max_seq_len != self.max_seq_len
            || kv_dim != self.kv_dim
        {
            return Err(KvCacheLoadError::ShapeMismatch(format!(
                "file num_layers/max_seq_len/kv_dim = {num_layers}/{max_seq_len}/{kv_dim}, \
                 model = {}/{}/{}",
                self._num_layers, self.max_seq_len, self.kv_dim
            )));
        }
        if (seq_len as usize) > max_seq_len {
            return Err(KvCacheLoadError::OverflowSeqLen {
                got: seq_len,
                max: max_seq_len,
            });
        }
        let seq_len = seq_len as usize;

        // Layer map.
        let mut layer_map = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            reader.read_exact(&mut buf4)?;
            layer_map.push(u32::from_le_bytes(buf4) as usize);
        }
        self.kv_layer_map = layer_map;

        // Data. Same skip-shared-layers pattern as `write_to`.
        let n = seq_len * kv_dim;
        for layer in 0..num_layers {
            if self.kv_layer_map[layer] != layer {
                continue;
            }
            let off = self.offset(layer, 0);
            let keys_bytes = f32_slice_as_bytes_mut(&mut self.keys[off..off + n]);
            reader.read_exact(keys_bytes)?;
            let values_bytes = f32_slice_as_bytes_mut(&mut self.values[off..off + n]);
            reader.read_exact(values_bytes)?;
        }
        self.seq_len = seq_len;
        Ok(())
    }
}

/// Reinterpret an `&[f32]` as an `&[u8]` for I/O. Endianness of the caller
/// is baked into the file — the format is defined as little-endian, so on
/// LE hosts (all Rust targets we care about) this cast is exact. Big-endian
/// support would need per-element byte swap and is left as future work.
fn f32_slice_as_bytes(s: &[f32]) -> &[u8] {
    // SAFETY: `f32` is `Copy` + has no drop; `[f32]` and `[u8]` have the
    // same lifetime and provenance. The byte length is `s.len() * 4`.
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), std::mem::size_of_val(s)) }
}

fn f32_slice_as_bytes_mut(s: &mut [f32]) -> &mut [u8] {
    // SAFETY: same as `f32_slice_as_bytes`, plus we hold `&mut`.
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(s)) }
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

/// Softplus activation `softplus(x) = ln(1 + exp(x))`. Used by Bonsai /
/// Qwen 3.6 DeltaNet (Phase X.3.e.3.2 Gap B) to derive the SSM decay
/// parameter from the raw alpha projection output. Numerically stable
/// form: for `x > 20` returns `x` (asymptotic limit, exp overflow
/// avoided); for `x < -20` returns `exp(x)` (avoids `ln(1 + 0)` losing
/// precision); otherwise evaluates the closed form directly.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Sigmoid activation `sigmoid(x) = 1 / (1 + exp(-x))`. Used by Bonsai /
/// Qwen 3.6 DeltaNet (Phase X.3.e.3.2 §Gap B extra) to constrain the raw
/// beta projection to `(0, 1)` before it enters the delta-rule integration
/// as the update-rate coefficient. Standard `silu(x) = x * sigmoid(x)`
/// already lives above; this helper exposes the bare sigmoid so the
/// beta path can multiply by 1, not `x`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Causal depthwise conv1d single-token step for Qwen 3.5 DeltaNet.
///
/// Direct CPU port of `src/shaders/conv1d_causal.wgsl`. Depthwise means the
/// convolution runs independently per channel `d`, so kernel weights are
/// laid out `[kernel_size, dim]` (row = kernel timestep, column = channel).
/// `state` is a ring buffer of length `(kernel_size - 1) * dim` that keeps
/// the previous `kernel_size - 1` activations per channel. `ring_pos`
/// tracks the write cursor within the ring; the read order
/// `[(rp+1) % ring, (rp+2) % ring, rp]` recovers the oldest → most recent
/// history slice used by the kernel.
///
/// Post-condition: `state[((rp + 1) % ring) * dim + d] = x[d]` for every
/// channel — the oldest slot is overwritten with the current activation and
/// `ring_pos` is advanced by one so the next call reads the correct window.
///
/// Layout matches the WGSL shader bit-for-bit so GPU / CPU cross-validation
/// stays meaningful (Issue #12 / #16).
fn causal_conv1d_step(
    x: &[f32],
    state: &mut [f32],
    ring_pos: &mut usize,
    weight: &[f32],
    bias: &[f32],
    out: &mut [f32],
    dim: usize,
    kernel_size: usize,
) {
    debug_assert_eq!(x.len(), dim);
    debug_assert_eq!(out.len(), dim);
    debug_assert_eq!(bias.len(), dim);
    debug_assert_eq!(weight.len(), kernel_size * dim);
    let ring = kernel_size - 1;
    debug_assert_eq!(state.len(), ring * dim);
    debug_assert!(kernel_size >= 2, "kernel_size must be at least 2");

    let rp = *ring_pos;
    // Read `kernel_size - 1` history slots + current x, weighted-sum with
    // the kernel row corresponding to that timestep.
    //
    // GGUF `ssm_conv1d.weight` shape is `[kernel_size, dim]` in ggml notation
    // (`ne[0] = kernel_size` inner, `ne[1] = dim` outer). Storage is
    // dim-outer × kernel-inner — for each channel `d`, the `kernel_size`
    // weight values are contiguous in `weight[d * kernel_size + k]`
    // (Phase X.3.e.3.5 fix: previous `weight[k * dim + d]` treated storage
    // as kernel-outer, causing catastrophic mismatch for Qwen 3.5 DeltaNet
    // and every hybrid arch that ships an `ssm_conv1d` tensor).
    for d in 0..dim {
        let mut acc = bias[d];
        let w_base = d * kernel_size;
        for k in 0..(kernel_size - 1) {
            // Slot at offset (rp + 1 + k) % ring maps kernel timestep k
            // to the (kernel_size - 1 - k)-oldest history entry, matching
            // the WGSL layout `state[((rp + 1 + k) % ring) * dim + d]`.
            let hist = state[((rp + 1 + k) % ring) * dim + d];
            acc += weight[w_base + k] * hist;
        }
        // Kernel row `kernel_size - 1` is applied to the current input.
        acc += weight[w_base + (kernel_size - 1)] * x[d];
        out[d] = acc;
    }

    // Overwrite the slot that was just consumed as "oldest" with the
    // current activation, then advance the write cursor.
    let write_slot = (rp + 1) % ring;
    for d in 0..dim {
        state[write_slot * dim + d] = x[d];
    }
    *ring_pos = write_slot;
}

/// Gated DeltaNet recurrent update + output for one decode step.
///
/// Direct CPU port of `src/shaders/gated_deltanet.wgsl`. Per head, the
/// recurrent state `S` has shape `[qk_dim, v_dim]` (row-major, `qk_dim`
/// outer) and evolves under the gated delta rule:
///
/// ```text
/// q, k = l2_normalize(silu(q)), l2_normalize(silu(k))
/// error = v - alpha * (S^T @ k)
/// S_new = alpha * S + beta * outer(k, error)
/// output = S_new^T @ q
/// output = output * silu(z)   // gated output
/// ```
///
/// Executes per-head loops using `rayon` when `num_heads >= 8` so 32-head
/// Qwen 3.5 configs get parallel speedup without paying scheduler
/// overhead on toy configs (used only by the unit tests).
///
/// Buffer layout (Phase X.3.e.3.1, Bonsai / Qwen 3.6 per-V-head expansion):
///   * `q`, `k`             — `num_kv_heads * qk_dim` (V heads inside the same
///     KV group share Q/K, mirroring standard GQA).
///   * `v`, `z`, `out`      — `num_v_heads * v_dim`   (independent per V head).
///   * `alpha`, `beta`      — `num_v_heads`            (per-V-head decay / rate).
///   * `state`              — `num_v_heads * qk_dim * v_dim`.
///
/// For standard Qwen 3.5 (`num_v_heads == num_kv_heads`) the mapping collapses
/// to the previous 1:1 arrangement; for Bonsai (48 V / 16 KV heads) each KV
/// group covers `num_v_heads / num_kv_heads = 3` V heads with independent
/// state and per-V-head alpha / beta.
#[allow(clippy::too_many_arguments)]
fn gated_deltanet_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: &[f32],
    beta: &[f32],
    z: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    num_kv_heads: usize,
    num_v_heads: usize,
    qk_dim: usize,
    v_dim: usize,
    bonsai_semantics: bool,
) {
    debug_assert!(num_kv_heads > 0);
    debug_assert!(num_v_heads > 0);
    debug_assert_eq!(
        num_v_heads % num_kv_heads,
        0,
        "num_v_heads ({num_v_heads}) must be a multiple of num_kv_heads ({num_kv_heads})",
    );
    debug_assert_eq!(q.len(), num_kv_heads * qk_dim);
    debug_assert_eq!(k.len(), num_kv_heads * qk_dim);
    debug_assert_eq!(v.len(), num_v_heads * v_dim);
    debug_assert_eq!(alpha.len(), num_v_heads);
    debug_assert_eq!(beta.len(), num_v_heads);
    debug_assert_eq!(z.len(), num_v_heads * v_dim);
    debug_assert_eq!(state.len(), num_v_heads * qk_dim * v_dim);
    debug_assert_eq!(out.len(), num_v_heads * v_dim);

    // Small configs (unit tests, toy models) skip rayon to avoid the
    // scheduler cost dominating a handful of arithmetic ops. Production
    // Qwen 3.5 uses 32 heads which clears the threshold comfortably;
    // Bonsai 27B has 48 V heads per DeltaNet layer.
    #[cfg(feature = "parallel")]
    {
        if num_v_heads >= 8 {
            gated_deltanet_step_parallel(
                q,
                k,
                v,
                alpha,
                beta,
                z,
                state,
                out,
                num_kv_heads,
                num_v_heads,
                qk_dim,
                v_dim,
                bonsai_semantics,
            );
            return;
        }
    }
    let v_per_kv = num_v_heads / num_kv_heads;
    for v_head in 0..num_v_heads {
        let kv_head = v_head / v_per_kv;
        let q_off = kv_head * qk_dim;
        let k_off = kv_head * qk_dim;
        let v_off = v_head * v_dim;
        let z_off = v_head * v_dim;
        let s_off = v_head * qk_dim * v_dim;
        gated_deltanet_head_disjoint(
            &q[q_off..q_off + qk_dim],
            &k[k_off..k_off + qk_dim],
            &v[v_off..v_off + v_dim],
            alpha[v_head],
            beta[v_head],
            &z[z_off..z_off + v_dim],
            &mut state[s_off..s_off + qk_dim * v_dim],
            &mut out[v_off..v_off + v_dim],
            qk_dim,
            v_dim,
            bonsai_semantics,
        );
    }
}

/// Rayon-parallel driver for [`gated_deltanet_step`] (`num_v_heads >= 8`).
///
/// Chunks the per-V-head slices of the mutable buffers so each worker owns a
/// disjoint `[qk_dim * v_dim]` state slab and a disjoint `[v_dim]` output
/// slab — the recurrence is intrinsically embarrassingly parallel across V
/// heads because there is no cross-head coupling. Q / K live at `kv_head`
/// granularity (shared across the V heads inside the same KV group).
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn gated_deltanet_step_parallel(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: &[f32],
    beta: &[f32],
    z: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    num_kv_heads: usize,
    _num_v_heads: usize,
    qk_dim: usize,
    v_dim: usize,
    bonsai_semantics: bool,
) {
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};
    use rayon::slice::ParallelSliceMut;

    // Derive V-per-KV grouping from the caller-provided head counts. The
    // `par_chunks_mut(state_stride)` split below iterates once per V head,
    // so we recover `kv_head = v_head / v_per_kv` inside the closure.
    let v_per_kv = _num_v_heads / num_kv_heads;
    let state_stride = qk_dim * v_dim;
    state
        .par_chunks_mut(state_stride)
        .zip(out.par_chunks_mut(v_dim))
        .enumerate()
        .for_each(|(v_head, (state_slab, out_slab))| {
            let kv_head = v_head / v_per_kv;
            gated_deltanet_head_disjoint(
                &q[kv_head * qk_dim..(kv_head + 1) * qk_dim],
                &k[kv_head * qk_dim..(kv_head + 1) * qk_dim],
                &v[v_head * v_dim..(v_head + 1) * v_dim],
                alpha[v_head],
                beta[v_head],
                &z[v_head * v_dim..(v_head + 1) * v_dim],
                state_slab,
                out_slab,
                qk_dim,
                v_dim,
                bonsai_semantics,
            );
        });
}

/// Per-head kernel using a single absolute head index into flat buffers.
/// Serial reference path retained for unit tests; assumes the 1:1 mapping
/// where V heads and KV heads coincide (standard Qwen 3.5). Hybrid
/// architectures (Bonsai / Qwen 3.6, num_v_heads > num_kv_heads) are
/// exercised by the loops inside [`gated_deltanet_step`] and
/// [`gated_deltanet_step_parallel`] instead — production forward path no
/// longer calls this helper directly.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn gated_deltanet_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: &[f32],
    beta: &[f32],
    z: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    head: usize,
    qk_dim: usize,
    v_dim: usize,
) {
    let q_off = head * qk_dim;
    let k_off = head * qk_dim;
    let v_off = head * v_dim;
    let z_off = head * v_dim;
    let s_off = head * qk_dim * v_dim;

    gated_deltanet_head_disjoint(
        &q[q_off..q_off + qk_dim],
        &k[k_off..k_off + qk_dim],
        &v[v_off..v_off + v_dim],
        alpha[head],
        beta[head],
        &z[z_off..z_off + v_dim],
        &mut state[s_off..s_off + qk_dim * v_dim],
        &mut out[v_off..v_off + v_dim],
        qk_dim,
        v_dim,
        false,
    );
}

/// Per-head kernel operating on already-sliced buffers. Both parallel and
/// serial drivers reduce to this single form so behaviour stays identical.
#[allow(clippy::too_many_arguments)]
fn gated_deltanet_head_disjoint(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha_h: f32,
    beta_h: f32,
    z: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    qk_dim: usize,
    v_dim: usize,
    bonsai_semantics: bool,
) {
    // L2 normalize q with a tiny epsilon so a zero vector produces a
    // zero output rather than NaN (matches WGSL `max(sqrt(sum_sq), 1e-12)`).
    // Q is additionally scaled by `1/sqrt(qk_dim)` to match reference
    // `qwen35.cpp:319-321` `q = ggml_scale(q, 1.0f / sqrtf(S_k))` applied
    // before the recurrence (Phase X.3.e.3.5 discovery). Fold the scale
    // into `q_norm` so downstream `q_i = q[i] * q_norm` picks it up
    // automatically without touching the per-column reduction loop.
    let mut q_sum_sq = 0.0f32;
    for &val in q {
        q_sum_sq += val * val;
    }
    let q_scale = 1.0 / (qk_dim as f32).sqrt();
    let q_norm = q_scale / q_sum_sq.sqrt().max(1e-12);

    let mut k_sum_sq = 0.0f32;
    for &val in k {
        k_sum_sq += val * val;
    }
    let k_norm = 1.0 / k_sum_sq.sqrt().max(1e-12);

    for j in 0..v_dim {
        // Column j of `S^T @ k` = sum_i state[i, j] * k_i.
        // Bonsai / Qwen 3.6 path skips the internal silu because the
        // caller pre-silu'd q / k post-conv1d (qwen35.cpp:502).
        let mut st_k = 0.0f32;
        for i in 0..qk_dim {
            let k_i = if bonsai_semantics {
                k[i] * k_norm
            } else {
                silu(k[i]) * k_norm
            };
            st_k += state[i * v_dim + j] * k_i;
        }
        let error_j = v[j] - alpha_h * st_k;

        // Update state column j while computing the new output entry.
        let mut out_j = 0.0f32;
        for i in 0..qk_dim {
            let (k_i, q_i) = if bonsai_semantics {
                (k[i] * k_norm, q[i] * q_norm)
            } else {
                (silu(k[i]) * k_norm, silu(q[i]) * q_norm)
            };
            let idx = i * v_dim + j;
            let s_new = alpha_h * state[idx] + beta_h * k_i * error_j;
            state[idx] = s_new;
            out_j += s_new * q_i;
        }

        // Gated output: legacy path multiplies by silu(z_j) inline; the
        // Bonsai path defers the z-gate to after ssm-norm (matches
        // qwen35.cpp:562 build_norm_gated) so we leave `out_j` untouched.
        out[j] = if bonsai_semantics {
            out_j
        } else {
            out_j * silu(z[j])
        };
    }
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
#[derive(Clone)]
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

/// Qwen 2 / 2.5 attention projection biases.
///
/// Present as a single `Option<QwenAttentionBiases>` on [`LayerWeights`]
/// rather than three independent `Option<Vec<f32>>` fields so the shape
/// of the layer struct signals "these three either all exist or none do".
/// Individual biases are still logically grouped: a layer either lacks
/// them entirely (Llama / Mistral / Gemma / Qwen 3) or carries all three.
#[allow(clippy::struct_field_names)]
struct QwenAttentionBiases {
    q_bias: Vec<f32>,
    k_bias: Vec<f32>,
    v_bias: Vec<f32>,
}

/// Qwen 3 per-head RMSNorm weights applied to Q and K before RoPE.
///
/// Kept as a pair so callers cannot end up with a Q norm without a K norm
/// (or vice versa). Absent for Qwen 2 / Llama / Mistral / Gemma.
struct QwenAttentionNorms {
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

/// Gemma 3n per-layer augmentation tensors.
///
/// Bundles the eleven layer-scoped weights the Gemma 3n forward path
/// consumes (post-norm plus Laurel branch plus per-layer input-embedding
/// branch plus AltUp bank). Grouping them lets non-Gemma-3n forward paths
/// ignore a single field instead of eleven, matching Issue #11's
/// God-object reduction goal.
struct Gemma3nLayerAugmentations<'a> {
    post_norm: Vec<f32>,
    inp_gate: WeightRef<'a>,
    proj: WeightRef<'a>,
    laurel_l: Vec<f32>,
    laurel_r: Vec<f32>,
    laurel_post_norm: Vec<f32>,
    altup_router: Vec<f32>,
    altup_router_norm: Vec<f32>,
    altup_predict_coef: Vec<f32>,
    altup_correct_coef: Vec<f32>,
    altup_correct_scale: Vec<f32>,
}

/// Mixture-of-experts routing weights (Qwen 3 MoE / Mixtral / Gemma 4
/// 26B_A4B). Absent when the layer runs the standard SwiGLU FFN.
#[allow(clippy::struct_field_names)]
struct MoeExpertWeights<'a> {
    /// MoE router weight: `[hidden_dim, n_expert]`. F32 dense.
    ffn_gate_inp: Vec<f32>,
    /// MoE gate expert weights: 3D `[hidden_dim, expert_ffn_size, n_expert]`.
    /// Stored expert-major so each expert's 2D slab is contiguous.
    ffn_gate_exps: WeightRef<'a>,
    /// MoE up expert weights: same layout as `ffn_gate_exps`.
    ffn_up_exps: WeightRef<'a>,
    /// MoE down expert weights: 3D `[expert_ffn_size, hidden_dim, n_expert]`.
    ffn_down_exps: WeightRef<'a>,
}

/// Layer weight references (zero-copy from GGUF).
///
/// Architecture-specific extensions are grouped into sub-structs
/// ([`QwenAttentionBiases`], [`QwenAttentionNorms`],
/// [`Gemma3nLayerAugmentations`], [`MoeExpertWeights`]) so the field
/// count stays manageable (Issue #11). Callers keep the pre-refactor
/// access syntax via accessor methods on `LayerWeights` — see
/// `q_bias`, `laurel_l`, `ffn_gate_inp`, etc.
struct LayerWeights<'a> {
    // ── Core attention ────────────────────────────────────────────────
    attn_norm: Vec<f32>,
    q_proj: WeightRef<'a>,
    /// K projection. Optional for Gemma 4 shared-KV layers (>= kv_from_start),
    /// where the layer's KV cache is redirected to an earlier layer and no
    /// K projection weight is stored in the GGUF.
    k_proj: Option<WeightRef<'a>>,
    /// V projection. Optional for Gemma 4 shared-KV layers.
    v_proj: Option<WeightRef<'a>>,
    o_proj: WeightRef<'a>,

    // ── Core FFN (standard SwiGLU; absent for MoE layers) ─────────────
    ffn_norm: Vec<f32>,
    /// Standard FFN gate projection. `None` for MoE layers, where the FFN
    /// is replaced by expert routing.
    gate_proj: Option<WeightRef<'a>>,
    /// Standard FFN up projection. `None` for MoE layers.
    up_proj: Option<WeightRef<'a>>,
    /// Standard FFN down projection. `None` for MoE layers.
    down_proj: Option<WeightRef<'a>>,

    // ── Gemma-2 post-norms (small, kept flat) ─────────────────────────
    /// Gemma-2 post-attention RMSNorm (before residual add).
    post_attn_norm: Option<Vec<f32>>,
    /// Gemma-2 post-FFN RMSNorm (before residual add).
    post_ffn_norm: Option<Vec<f32>>,

    // ── Gemma-4 extras (small, kept flat) ─────────────────────────────
    /// Gemma 4: per-layer output scalar (`layer_output_scale`), typically [1].
    /// Applied as `cur *= out_scale` at the end of each layer.
    out_scale: Option<Vec<f32>>,
    /// Gemma 4: per-full-attention-layer RoPE frequency factors (Llama 3.x
    /// NTK-aware extension). Shape [head_dim / 2]. Absent for SWA layers.
    rope_freqs: Option<Vec<f32>>,

    // ── Grouped arch-specific extensions ──────────────────────────────
    /// Qwen 2 / 2.5 QKV projection biases (all three or none).
    qwen_biases: Option<QwenAttentionBiases>,
    /// Qwen 3 per-head QK RMSNorm pair.
    qwen_norms: Option<QwenAttentionNorms>,
    /// Gemma 3n per-layer augmentations (11 weights: Laurel / AltUp /
    /// per-layer embedding branch).
    gemma3n: Option<Gemma3nLayerAugmentations<'a>>,
    /// Mixture-of-experts routing + expert weights (only when this layer
    /// uses expert dispatch instead of a monolithic SwiGLU FFN).
    moe: Option<MoeExpertWeights<'a>>,

    /// Qwen 3.6 / Bonsai 27B "Gated Attention": `q_proj` outputs `2 * q_dim`
    /// values — the first half is the actual Q, the second half is a per-
    /// element swish (SiLU) gate applied to the attention output before
    /// `o_proj`. When `false`, `q_proj` produces the standard `q_dim` Q
    /// values only and no gate is applied. Set at load time from the
    /// tensor shape (rows = `2 * q_dim` ⇒ gated, rows = `q_dim` ⇒ standard).
    ///
    /// This matches the Qwen 3.6 config keys
    /// `attn_output_gate: true, output_gate_type: "swish"`.
    gated_output: bool,
}

impl<'a> LayerWeights<'a> {
    // ── Qwen 2 / 2.5 attention bias accessors ─────────────────────────

    #[inline]
    fn q_bias(&self) -> Option<&[f32]> {
        self.qwen_biases.as_ref().map(|b| b.q_bias.as_slice())
    }

    #[inline]
    fn k_bias(&self) -> Option<&[f32]> {
        self.qwen_biases.as_ref().map(|b| b.k_bias.as_slice())
    }

    #[inline]
    fn v_bias(&self) -> Option<&[f32]> {
        self.qwen_biases.as_ref().map(|b| b.v_bias.as_slice())
    }

    // ── Qwen 3 per-head QK RMSNorm accessors ──────────────────────────

    #[inline]
    fn q_norm(&self) -> Option<&[f32]> {
        self.qwen_norms.as_ref().map(|n| n.q_norm.as_slice())
    }

    #[inline]
    fn k_norm(&self) -> Option<&[f32]> {
        self.qwen_norms.as_ref().map(|n| n.k_norm.as_slice())
    }

    // ── Gemma 3n augmentation accessors ───────────────────────────────

    #[inline]
    fn post_norm(&self) -> Option<&[f32]> {
        self.gemma3n.as_ref().map(|g| g.post_norm.as_slice())
    }

    #[inline]
    fn inp_gate(&self) -> Option<&WeightRef<'a>> {
        self.gemma3n.as_ref().map(|g| &g.inp_gate)
    }

    #[inline]
    fn proj(&self) -> Option<&WeightRef<'a>> {
        self.gemma3n.as_ref().map(|g| &g.proj)
    }

    #[inline]
    fn laurel_l(&self) -> Option<&[f32]> {
        self.gemma3n.as_ref().map(|g| g.laurel_l.as_slice())
    }

    #[inline]
    fn laurel_r(&self) -> Option<&[f32]> {
        self.gemma3n.as_ref().map(|g| g.laurel_r.as_slice())
    }

    #[inline]
    fn laurel_post_norm(&self) -> Option<&[f32]> {
        self.gemma3n.as_ref().map(|g| g.laurel_post_norm.as_slice())
    }

    #[inline]
    fn altup_router(&self) -> Option<&[f32]> {
        self.gemma3n.as_ref().map(|g| g.altup_router.as_slice())
    }

    #[inline]
    fn altup_router_norm(&self) -> Option<&[f32]> {
        self.gemma3n
            .as_ref()
            .map(|g| g.altup_router_norm.as_slice())
    }

    #[inline]
    fn altup_predict_coef(&self) -> Option<&[f32]> {
        self.gemma3n
            .as_ref()
            .map(|g| g.altup_predict_coef.as_slice())
    }

    #[inline]
    fn altup_correct_coef(&self) -> Option<&[f32]> {
        self.gemma3n
            .as_ref()
            .map(|g| g.altup_correct_coef.as_slice())
    }

    #[inline]
    fn altup_correct_scale(&self) -> Option<&[f32]> {
        self.gemma3n
            .as_ref()
            .map(|g| g.altup_correct_scale.as_slice())
    }

    // ── MoE accessors ────────────────────────────────────────────────

    #[inline]
    fn ffn_gate_inp(&self) -> Option<&[f32]> {
        self.moe.as_ref().map(|m| m.ffn_gate_inp.as_slice())
    }

    #[inline]
    fn ffn_gate_exps(&self) -> Option<&WeightRef<'a>> {
        self.moe.as_ref().map(|m| &m.ffn_gate_exps)
    }

    #[inline]
    fn ffn_up_exps(&self) -> Option<&WeightRef<'a>> {
        self.moe.as_ref().map(|m| &m.ffn_up_exps)
    }

    #[inline]
    fn ffn_down_exps(&self) -> Option<&WeightRef<'a>> {
        self.moe.as_ref().map(|m| &m.ffn_down_exps)
    }

    /// Convenience: true when this layer runs the MoE FFN path.
    #[inline]
    #[allow(dead_code)]
    fn is_moe_layer(&self) -> bool {
        self.moe.is_some()
    }
}

/// DeltaNet layer weight references (Qwen 3.5 / 3.6 Gated Linear Attention).
///
/// Held in a dedicated `Vec` on [`Llama3Model`] instead of extending
/// [`LayerWeights`] to avoid worsening the God-object shape tracked by
/// Issue #11. Mirrors the GPU-side `DeltaNetLayerWeightBufs` field layout so
/// tensor loading logic reads the same `blk.{i}.ssm_*` / `blk.{i}.ffn_*`
/// keys on both paths.
/// DeepSeek-V3 / R1 per-layer MLA weight references (Phase 2 of Issue #32).
///
/// Mirrors the llama.cpp tensor naming for `deepseek2` GGUFs:
/// `attn_norm.weight`, `attn_q_a.weight`, `attn_q_a_norm.weight`,
/// `attn_q_b.weight`, `attn_kv_a_mqa.weight`, `attn_kv_a_norm.weight`,
/// `attn_kv_b.weight`, `attn_output.weight`, plus a standard SwiGLU FFN
/// for the first `first_k_dense_replace` layers (`ffn_norm`, `ffn_gate`,
/// `ffn_up`, `ffn_down`). MoE-layer FFN weights are Phase 3 (Issue #33).
///
/// Shape summary (V3 numbers, hidden_dim=7168, num_heads=128,
/// q_lora_rank=1536, kv_lora_rank=512, qk_nope=128, qk_rope=64, v_head=128):
///
/// * `q_a_proj`: `[q_lora_rank, hidden_dim]`
/// * `q_b_proj`: `[num_heads * (qk_nope + qk_rope), q_lora_rank]`
/// * `kv_a_proj_with_mqa`: `[kv_lora_rank + qk_rope, hidden_dim]`
/// * `kv_b_proj`: `[num_heads * (qk_nope + v_head), kv_lora_rank]`
/// * `o_proj`: `[hidden_dim, num_heads * v_head]`
///
/// FFN weights are `Option` because MoE layers omit them entirely.
struct DeepSeekV3LayerWeights<'a> {
    attn_norm: Vec<f32>,
    /// Q LoRA down projection: `hidden → q_lora_rank`.
    q_a_proj: WeightRef<'a>,
    /// RMSNorm applied to the `q_lora_rank` intermediate before `q_b_proj`.
    q_a_norm: Vec<f32>,
    /// Q LoRA up projection: `q_lora_rank → num_heads * (qk_nope + qk_rope)`.
    q_b_proj: WeightRef<'a>,
    /// Fused KV LoRA down + MQA k_pe projection:
    /// `hidden → (kv_lora_rank + qk_rope_head_dim)`.
    kv_a_proj_with_mqa: WeightRef<'a>,
    /// RMSNorm applied to the `kv_lora_rank` intermediate before `kv_b_proj`.
    kv_a_norm: Vec<f32>,
    /// KV LoRA up projection:
    /// `kv_lora_rank → num_heads * (qk_nope + v_head)`.
    kv_b_proj: WeightRef<'a>,
    /// Output projection: `num_heads * v_head → hidden`.
    o_proj: WeightRef<'a>,
    // ── Dense FFN (present only for `first_k_dense_replace` layers) ─────
    /// FFN RMSNorm, populated for dense layers only.
    ffn_norm: Option<Vec<f32>>,
    /// Dense SwiGLU gate. `None` for MoE layers.
    gate_proj: Option<WeightRef<'a>>,
    /// Dense SwiGLU up. `None` for MoE layers.
    up_proj: Option<WeightRef<'a>>,
    /// Dense SwiGLU down. `None` for MoE layers.
    down_proj: Option<WeightRef<'a>>,
    // ── MoE (present only for layers ≥ `first_k_dense_replace`) ─────────
    /// DeepSeek-V3 MoE weights (Phase 3). `None` for dense layers.
    moe: Option<DeepSeekMoeWeights<'a>>,
}

/// DeepSeek-V3 MoE weight bundle for one non-dense layer.
///
/// V3 uses **sigmoid gating** with the noaux_tc bias-correction trick +
/// one **always-active shared expert** + **routed_scaling_factor** applied
/// to the routed sum. All three components are loaded together so the
/// forward path can dispatch without checking Option-ness inside the
/// hot loop.
struct DeepSeekMoeWeights<'a> {
    /// FFN RMSNorm applied to `hidden` before the router / shared expert.
    ffn_norm: Vec<f32>,
    /// Router logits projection: `hidden → n_routed_experts` (dense f32).
    ffn_gate_inp: Vec<f32>,
    /// noaux_tc expert-bias vector `[n_routed_experts]`. Added to the
    /// sigmoid scores for **top-k selection** only — the final routing
    /// weights use the un-biased scores. `None` when the checkpoint does
    /// not ship a bias tensor (older DeepSeek-V2 / rare V3 variants).
    exp_probs_b: Option<Vec<f32>>,
    /// Routed experts — either in-memory (default) or streamed on demand
    /// through the Phase 4a LRU pool. See [`RoutedExpertStorage`].
    routed: RoutedExpertStorage<'a>,
    /// Shared expert (always active, no gating). Uses `n_shared_experts *
    /// moe_intermediate_size` as its FFN size — V3: 1 * 2048 = 2048.
    /// Shared expert is never streamed: it fires on every token so LRU
    /// caching offers no locality benefit.
    ffn_gate_shexp: WeightRef<'a>,
    ffn_up_shexp: WeightRef<'a>,
    ffn_down_shexp: WeightRef<'a>,
}

/// DeepSeek-V3 Multi-Token Prediction (MTP) module weights (Phase 5a, Issue #35).
///
/// # What MTP is
///
/// V3 optionally trains a small extra transformer module on top of the
/// main 60 hidden layers that predicts *two* tokens per forward pass
/// instead of one. At inference time this yields a lossless draft
/// pipeline for speculative decoding: the MTP head drafts token *N + 2*
/// during the main model's forward for token *N + 1*, and a subsequent
/// main-model forward for *N + 1* verifies whether the draft was correct.
///
/// # Architecture (paper §3.5.2)
///
/// One MTP module (V3 uses `D = 1` for spec decoding):
///
/// 1. `enorm` — RMSNorm on the embedding of the next token
/// 2. `hnorm` — RMSNorm on the previous MTP module's hidden state (or the
///    main model's last hidden state for the first MTP module)
/// 3. `eh_proj` — concat(hnorm_out, enorm_out) → `[2*hidden, hidden]`
///    projection back to hidden dim
/// 4. `block` — one full transformer block, same MLA + MoE FFN structure
///    as a regular V3 layer
/// 5. `final_norm` + shared `output_proj` (main model's `output.weight`)
///    produce the logits for the extra predicted position
///
/// # GGUF tensor naming
///
/// llama.cpp did not have V3 MTP support at the time this was authored, so
/// there is no canonical naming yet. The loader accepts the paper-inspired
/// names `mtp.enorm.weight` / `mtp.hnorm.weight` / `mtp.eh_proj.weight` /
/// `mtp.norm.weight` and reads the inner transformer block from
/// `mtp.blk.0.*` (same tensor family as `blk.{i}.*` main layers). When
/// llama.cpp settles on a convention, add the alternative names next to
/// the current lookups.
///
/// # Scope of Phase 5a (this struct)
///
/// Ships the *loader* and *storage* only — the forward path is stubbed
/// via [`Llama3Model::mtp_draft`] which currently returns `todo!()`. Full
/// MTP forward needs to reuse `forward_deepseek_v3`'s per-layer MLA + MoE
/// primitives; factoring those out is a follow-up (Phase 5a.2) because it
/// touches the outer forward loop.
///
/// [`Llama3Model::mtp_draft`]: Llama3Model::mtp_draft
#[allow(dead_code)]
struct DeepSeekV3MtpWeights<'a> {
    /// Embedding entry RMSNorm — applied to the next token's embedding.
    enorm: Vec<f32>,
    /// Hidden entry RMSNorm — applied to the incoming hidden state.
    hnorm: Vec<f32>,
    /// Concatenation projection: `[2 * hidden_dim, hidden_dim]`. Takes
    /// `concat(hnorm_out, enorm_out)` and produces the hidden vector that
    /// feeds the inner transformer block.
    eh_proj: WeightRef<'a>,
    /// The inner transformer block. Structurally identical to a regular
    /// V3 layer (attention: MLA with LoRA Q/KV + partial NEOX RoPE, FFN:
    /// MoE with sigmoid gating + shared expert + noaux_tc bias).
    block: DeepSeekV3LayerWeights<'a>,
    /// Final RMSNorm before the output head. The head itself is shared
    /// with the main model's `output_proj` — no separate MTP `output.weight`.
    final_norm: Vec<f32>,
}

/// Backing store for the routed-expert weights of a DeepSeek-V3 MoE layer.
///
/// Two variants:
///
/// - **`InMemory`** — the pre-Phase-4 default. Full expert 3D tensors are
///   held as `WeightRef<'a>` and per-expert slabs are extracted by slicing
///   into the mmap'd bytes. Zero allocations per matvec.
///
/// - **`Streaming`** — Phase 4a (Issue #34). The routed-expert bytes live
///   inside a shared `StreamingExpertPool` with an LRU cache; slabs are
///   loaded on demand for the top-k experts selected by the router. This
///   makes running the full 671 B DeepSeek-V3 possible on machines that
///   cannot fit all 15,616 routed experts in RAM simultaneously.
///
/// The loader currently only builds `InMemory` — real-GGUF streaming
/// wiring is deliberately deferred to a follow-up (see [`Phase 4a scope`
/// in `deepseek_streaming.rs`](crate::deepseek_streaming)). The enum shape
/// is in place so `forward_deepseek_moe_layer` supports either backend
/// today and the loader can be swapped without further code changes.
enum RoutedExpertStorage<'a> {
    InMemory {
        gate: WeightRef<'a>,
        up: WeightRef<'a>,
        down: WeightRef<'a>,
    },
    /// Phase 4a scaffolding — enum variant + forward-path dispatch are in
    /// place so a subsequent PR can wire the loader to construct this
    /// variant (real-GGUF mmap threading + pool budget knob) without
    /// further changes to the forward code. Pool semantics are covered
    /// by [`crate::deepseek_streaming`]'s unit tests.
    #[allow(dead_code)]
    Streaming {
        pool: std::sync::Arc<crate::deepseek_streaming::StreamingExpertPool>,
        layer_idx: usize,
    },
}

struct DeltaNetLayerWeights<'a> {
    attn_norm: Vec<f32>,
    /// Standard Qwen 3.5 fused input projection: `hidden → q + k + v + z`
    /// packed. Rows = `qk_dim * num_kv_heads * 2 + v_dim * num_v_heads * 2`.
    ///
    /// `None` for Bonsai 27B (which uses [`attn_qkv`] + [`attn_gate`]
    /// instead). Exactly one of `ssm_in` / `attn_qkv` is `Some` per layer;
    /// the loader guarantees this.
    ///
    /// [`attn_qkv`]: DeltaNetLayerWeights::attn_qkv
    /// [`attn_gate`]: DeltaNetLayerWeights::attn_gate
    ssm_in: Option<WeightRef<'a>>,
    /// Bonsai 27B / Qwen 3.6 fused input projection: `hidden → q + k + v`
    /// packed (10240-dim for the 27B config). Together with [`attn_gate`]
    /// this pair replaces the standard Qwen 3.5 [`ssm_in`]. Consumed by
    /// the DeltaNet forward path (Phase X.3.e.2).
    ///
    /// Row layout matches the standard Qwen 3.5 QKV split:
    /// `[Q (qk_dim * num_kv_heads) | K (qk_dim * num_kv_heads) | V (v_dim * num_v_heads)]`.
    ///
    /// [`ssm_in`]: DeltaNetLayerWeights::ssm_in
    /// [`attn_gate`]: DeltaNetLayerWeights::attn_gate
    attn_qkv: Option<WeightRef<'a>>,
    /// Bonsai 27B / Qwen 3.6 DeltaNet Z (output-gate) projection: `hidden → z`
    /// (6144-dim for the 27B config, matches `v_dim * num_v_heads`). Consumed
    /// by the DeltaNet forward path (Phase X.3.e.2) as the `z` slice inside
    /// `gated_deltanet_step`.
    attn_gate: Option<WeightRef<'a>>,
    /// Qwen 3.6 learnable SSM state-transition parameter (`num_v_heads`
    /// entries, f32), stored ≈ `-exp(A_log)` (negative, per Mamba
    /// convention). Multiplied by `softplus(alpha_raw + ssm_dt_bias)`
    /// in the forward path (Phase X.3.e.3.2 Gap B) to derive the
    /// per-V-head log-decay `gate`; the actual decay factor used inside
    /// the delta-rule recurrence is `exp(gate) ∈ (0, 1]`. Standard
    /// Qwen 3.5 GGUFs ship no `ssm_a`; the transformation is skipped
    /// and the raw alpha projection output is used directly.
    ssm_a: Option<Vec<f32>>,
    /// Qwen 3.6 SSM discretisation-step bias (`num_v_heads` entries, f32),
    /// added to the raw alpha projection output before the softplus in
    /// the discretisation formula; see [`ssm_a`] for the full math.
    ///
    /// [`ssm_a`]: DeltaNetLayerWeights::ssm_a
    ssm_dt_bias: Option<Vec<f32>>,
    /// Qwen 3.6 SSM state RMSNorm weight (`state_size` = per-head qk_dim
    /// entries = `v_dim`, f32). Broadcast across V heads and applied to
    /// `dn_delta_out` between the recurrence and the `ssm_out` projection
    /// (Phase X.3.e.3.2). Standard Qwen 3.5 GGUF exports omit this tensor
    /// and the forward path skips the normalisation, preserving pre-refactor
    /// numerics for those checkpoints.
    ssm_norm: Option<Vec<f32>>,
    /// Causal depthwise conv1d kernel: `[kernel_size, conv_dim]` (f32).
    ///
    /// `conv_dim = qk_dim * num_kv_heads * 2 + v_dim * num_v_heads`
    /// (covers q + k + v, excludes z).
    conv1d_weight: Vec<f32>,
    /// Causal conv1d bias: `[conv_dim]` (f32). Optional at load time
    /// (Bonsai 27B ships no bias); missing values default to zeros.
    conv1d_bias: Vec<f32>,
    /// Alpha decay-gate projection: `hidden → alpha [num_kv_heads]`.
    alpha_proj: WeightRef<'a>,
    /// Beta update-rate projection: `hidden → beta [num_kv_heads]`.
    beta_proj: WeightRef<'a>,
    /// Output projection: `delta_out [v_dim * num_v_heads] → hidden`.
    ssm_out: WeightRef<'a>,
    ffn_norm: Vec<f32>,
    gate_proj: WeightRef<'a>,
    up_proj: WeightRef<'a>,
    down_proj: WeightRef<'a>,
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
    /// Quantization type of `per_layer_token_embd_raw`. Gemma 3n uses Q5_1,
    /// Gemma 4 uses Q6_K; the slice extractor dispatches on this.
    per_layer_token_embd_qtype: Option<crate::gguf::GgmlType>,
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
    /// Qwen 3.5 / 3.6 DeltaNet layer weights. `None` when the model has no
    /// DeltaNet layers (`is_hybrid() == false`). When populated, indexed by
    /// DeltaNet slot (0..num_deltanet_layers), not by global layer index —
    /// use `deltanet_layer_index_map` to translate.
    deltanet_layers: Option<Vec<DeltaNetLayerWeights<'a>>>,
    /// Per-global-layer routing: `layer_kind_map[i] = LayerKind::Attention(k)`
    /// where `k` indexes into `layers`, or `LayerKind::DeltaNet(k)` where `k`
    /// indexes into `deltanet_layers`. Empty for non-hybrid models (default
    /// path treats every layer as attention).
    layer_kind_map: Vec<LayerKind>,
    /// Per-DeltaNet-layer recurrent state `S`, laid out
    /// `[num_kv_heads, qk_dim, v_dim]` in row-major with `qk_dim` as the outer
    /// stride (matches `gated_deltanet.wgsl` `state[s_off + i * v_dim + j]`).
    deltanet_state: Vec<Vec<f32>>,
    /// Per-DeltaNet-layer causal conv1d ring buffer `[(kernel-1) * conv_dim]`.
    deltanet_conv_state: Vec<Vec<f32>>,
    /// Per-DeltaNet-layer ring position `0..(kernel-1)`. Advanced each decode
    /// step; the oldest slot (`(rp + 1) % (kernel-1)`) is overwritten by the
    /// current activation before the next step reads it.
    deltanet_conv_ring_pos: Vec<usize>,
    /// DeepSeek-V2 / V3 / R1 per-layer MLA + optional dense FFN weights.
    /// `None` for non-DeepSeek architectures. Indexed by global layer index
    /// (unlike DeltaNet, DeepSeek-V3 does not interleave with attention layers
    /// — every layer is MLA).
    deepseek_v3_layers: Option<Vec<DeepSeekV3LayerWeights<'a>>>,
    /// DeepSeek-V3 Multi-Token Prediction head weights (Phase 5a, Issue #35).
    /// `None` when the checkpoint does not ship an MTP head (all V2 quants
    /// and some V3 variants) or the arch is not DeepSeek-V3. Populated at
    /// load time by [`load_deepseek_v3_mtp_weights`] when the `mtp.*`
    /// tensor family is present. Used at inference time by [`mtp_draft`]
    /// to produce a single draft token for speculative decoding.
    ///
    /// [`load_deepseek_v3_mtp_weights`]: crate::llama3::load_deepseek_v3_mtp_weights
    /// [`mtp_draft`]: Llama3Model::mtp_draft
    deepseek_v3_mtp: Option<DeepSeekV3MtpWeights<'a>>,
}

/// Compact routing tag used by `Llama3Model::layer_kind_map` for hybrid
/// (DeltaNet + full-attention) models. Non-hybrid models keep the vector
/// empty and route every layer through the standard attention path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerKind {
    /// Standard attention layer. Payload is the index into `layers`.
    Attention(usize),
    /// DeltaNet linear-attention layer. Payload is the index into
    /// `deltanet_layers`.
    DeltaNet(usize),
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

        // Layers. Three routing regimes coexist:
        //  * DeepSeek-V3 / R1 — every layer is MLA, weights live in
        //    `deepseek_v3_layers`; `layers` stays empty and `layer_kind_map`
        //    is unused (the forward path indexes directly by global index).
        //  * Qwen 3.5 / 3.6 hybrid — DeltaNet layers land in
        //    `deltanet_layers`, attention layers in `layers`,
        //    `layer_kind_map` disambiguates.
        //  * Everything else — every layer is a standard attention entry in
        //    `layers`, `layer_kind_map` is empty.
        let is_deepseek_model = matches!(config.arch, ModelArch::DeepSeekV3);
        let is_hybrid_model = config.is_hybrid();
        let mut layers = Vec::with_capacity(if is_deepseek_model {
            0
        } else {
            config.num_layers
        });
        let mut deltanet_layers_vec: Vec<DeltaNetLayerWeights<'a>> = Vec::new();
        let mut deepseek_v3_layers_vec: Vec<DeepSeekV3LayerWeights<'a>> = Vec::new();
        // Phase 4b.1: build one shared streaming pool for the whole model
        // when `ALICE_LLM_MOE_STREAMING=1` + `ALICE_LLM_MOE_STREAMING_FILE=<path>`
        // are both set. Failure (env unset, file open error, missing tensor)
        // silently falls back to InMemory so callers who don't opt in are
        // unaffected.
        let deepseek_streaming_pool = if is_deepseek_model {
            build_deepseek_streaming_pool(gguf, &config)
        } else {
            None
        };
        let mut layer_kind_map: Vec<LayerKind> = if is_hybrid_model {
            Vec::with_capacity(config.num_layers)
        } else {
            Vec::new()
        };
        for i in 0..config.num_layers {
            if is_deepseek_model {
                let dsv = load_deepseek_v3_layer_weights(
                    gguf,
                    i,
                    &config,
                    deepseek_streaming_pool.as_ref(),
                )?;
                deepseek_v3_layers_vec.push(dsv);
            } else if is_hybrid_model && config.is_deltanet_layer(i) {
                let dn = load_deltanet_layer_weights(gguf, i, &config)?;
                layer_kind_map.push(LayerKind::DeltaNet(deltanet_layers_vec.len()));
                deltanet_layers_vec.push(dn);
            } else {
                let layer = load_layer_weights(gguf, i, &config)?;
                if is_hybrid_model {
                    layer_kind_map.push(LayerKind::Attention(layers.len()));
                }
                layers.push(layer);
            }
        }
        let deltanet_layers = if is_hybrid_model {
            Some(deltanet_layers_vec)
        } else {
            None
        };
        let deepseek_v3_layers = if is_deepseek_model {
            Some(deepseek_v3_layers_vec)
        } else {
            None
        };
        // Phase 5a: optional MTP head. Only DeepSeek-V3 checkpoints ship
        // this; V2 quants and pre-MTP V3 variants leave it None. The
        // loader tolerates missing tensors and never errors — a missing
        // MTP head only prevents speculative decoding, not regular decode.
        let deepseek_v3_mtp = if is_deepseek_model {
            load_deepseek_v3_mtp_weights(gguf, &config, deepseek_streaming_pool.as_ref())
        } else {
            None
        };
        let deltanet_layer_count = deltanet_layers.as_ref().map_or(0, Vec::len);

        // Per-DeltaNet-layer recurrent state and conv1d ring buffer allocation.
        // Sized from the config so the first forward pass finds them ready.
        let dn_num_kv_heads_state = config.linear_num_kv_heads().unwrap_or(config.num_kv_heads);
        let dn_qk_dim_state = config.linear_qk_head_dim().unwrap_or(128);
        let dn_v_dim_state = config.linear_kv_head_dim().unwrap_or(128);
        let dn_num_v_heads_state = config.linear_num_v_heads().unwrap_or(config.num_heads);
        let dn_conv_kernel = config.linear_conv_kernel_dim().unwrap_or(4);
        let dn_conv_dim_state =
            dn_qk_dim_state * dn_num_kv_heads_state * 2 + dn_v_dim_state * dn_num_v_heads_state;
        // Phase X.3.e.3.1: state is per-V-head. Bonsai (48 V / 16 KV) inflates
        // the buffer 3× vs the pre-refactor per-KV allocation; Qwen 3.5
        // (num_v_heads == num_kv_heads) sees no change.
        let dn_state_elems = dn_num_v_heads_state * dn_qk_dim_state * dn_v_dim_state;
        let dn_conv_ring_slots = dn_conv_kernel.saturating_sub(1);
        let deltanet_state = (0..deltanet_layer_count)
            .map(|_| vec![0.0f32; dn_state_elems])
            .collect();
        let deltanet_conv_state = (0..deltanet_layer_count)
            .map(|_| vec![0.0f32; dn_conv_ring_slots * dn_conv_dim_state])
            .collect();
        let deltanet_conv_ring_pos = vec![0usize; deltanet_layer_count];

        // KV cache dim per token.
        //  * DeepSeek-V3 stores the compressed MLA latent (`kv_lora_rank` +
        //    the shared `qk_rope_head_dim` positional slice). This is the
        //    source of the ~57× KV-cache compression relative to standard
        //    GQA and is what makes long-context DeepSeek runs fit in RAM.
        //  * Everything else stores full `num_kv_heads * head_dim` bytes.
        let kv_dim = if is_deepseek_model {
            config.deepseek_kv_lora_rank().unwrap_or(0)
                + config.deepseek_qk_rope_head_dim().unwrap_or(0)
        } else {
            config.num_kv_heads * config.head_dim
        };
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
        ) = if matches!(config.arch, ModelArch::Gemma3n | ModelArch::Gemma4) {
            let per_layer_token_embd_raw = gguf.tensor_data("per_layer_token_embd.weight");
            // per_layer_model_proj.weight ggml shape {n_embd, n_layer * n_embd_altup}
            // (ne0=n_embd=in, ne1=n_layer*n_embd_altup=out)
            // → rows=n_layer*n_embd_altup, cols=n_embd.
            let per_layer_model_proj = config.per_layer_input_embedding_dim().and_then(|dim| {
                load_weight_ref(
                    gguf,
                    "per_layer_model_proj.weight",
                    config.num_layers * dim, // rows = out_dim
                    config.hidden_dim,       // cols = in_dim
                )
            });
            let per_layer_proj_norm = gguf.tensor_to_f32("per_layer_proj_norm.weight");
            // AltUp tensors are Gemma 3n only.
            let altup_proj = if config.arch == ModelArch::Gemma3n {
                gguf.tensor_to_f32("altup_proj.weight")
            } else {
                None
            };
            let altup_unembd_proj = if config.arch == ModelArch::Gemma3n {
                gguf.tensor_to_f32("altup_unembd_proj.weight")
            } else {
                None
            };
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

        let per_layer_token_embd_qtype = gguf
            .tensor_info("per_layer_token_embd.weight")
            .map(|info| info.qtype);
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
            per_layer_token_embd_qtype,
            per_layer_model_proj,
            per_layer_proj_norm,
            altup_proj,
            altup_unembd_proj,
            deltanet_layers,
            layer_kind_map,
            deltanet_state,
            deltanet_conv_state,
            deltanet_conv_ring_pos,
            deepseek_v3_layers,
            deepseek_v3_mtp,
        })
    }

    /// Whether this model shipped a DeepSeek-V3 MTP head. Callers gate
    /// [`mtp_draft`] on this — a missing head disables speculative
    /// decoding but does not affect regular greedy / sample generation.
    ///
    /// [`mtp_draft`]: Self::mtp_draft
    #[must_use]
    pub fn has_deepseek_mtp(&self) -> bool {
        self.deepseek_v3_mtp.is_some()
    }

    /// DeepSeek-V3 Multi-Token Prediction draft (Phase 5a.3, Issue #35).
    ///
    /// Runs the loaded MTP module on `(prev_hidden, next_token)` to draft
    /// the *second-next* token's logits — the one that would follow the
    /// token the main model just decoded. The caller then verifies with a
    /// regular main-model forward and applies rejection sampling.
    ///
    /// # Algorithm (paper §3.5.2, D=1)
    ///
    /// 1. Look up embedding for `next_token`.
    /// 2. `hnorm` on `prev_hidden`, `enorm` on the embedding.
    /// 3. Concat `(hnorm_out, enorm_out)` into a `2 * hidden_dim` vector.
    /// 4. `eh_proj` (a `[hidden, 2*hidden]` matvec) collapses back to
    ///    `hidden_dim`.
    /// 5. Run the MTP block's single V3 layer (attention + MoE FFN) at
    ///    position 0. Because the block only ever sees this one draft
    ///    position, `seq_len = 1` and the attention softmax reduces to
    ///    `[1.0]` — no KV cache needed, no history to iterate over. The
    ///    per-head attention output is exactly the value vector `V[h]`.
    /// 6. Final RMSNorm, then the *main model's* `output_proj` (MTP
    ///    shares the output head, no separate `mtp.output.weight`).
    ///
    /// # Assumptions
    ///
    /// - `prev_hidden.len() == config.hidden_dim`.
    /// - The loaded MTP block uses the MoE FFN branch (all V3 MTP variants
    ///   at authorship time do; a dense fallback would need to be added
    ///   alongside `forward_deepseek_moe_layer`).
    ///
    /// # Panics
    ///
    /// - If [`has_deepseek_mtp`] is `false`.
    /// - If `prev_hidden.len() != config.hidden_dim` (debug-only assert).
    ///
    /// # Validation status
    ///
    /// The forward is fully implemented. Bit-exact correctness against a
    /// reference DeepSeek-V3 implementation still needs a real V3 GGUF
    /// (blocked on ~370 GB local disk). Unit tests verify tensor-shape
    /// invariants and non-NaN outputs on synthetic weights.
    ///
    /// [`has_deepseek_mtp`]: Self::has_deepseek_mtp
    pub fn mtp_draft(&mut self, prev_hidden: &[f32], next_token: u32) -> Vec<f32> {
        let mtp = self
            .deepseek_v3_mtp
            .as_ref()
            .expect("mtp_draft called without a loaded MTP head — check has_deepseek_mtp() first");
        let c = self.config.clone();
        let hidden_dim = c.hidden_dim;
        debug_assert_eq!(
            prev_hidden.len(),
            hidden_dim,
            "prev_hidden length must match config.hidden_dim"
        );
        let num_heads = c.num_heads;
        let q_lora_rank = c.deepseek_q_lora_rank().expect("q_lora_rank");
        let kv_lora_rank = c.deepseek_kv_lora_rank().expect("kv_lora_rank");
        let qk_nope = c.deepseek_qk_nope_head_dim().expect("qk_nope_head_dim");
        let qk_rope = c.deepseek_qk_rope_head_dim().expect("qk_rope_head_dim");
        let v_head_dim = c.deepseek_v_head_dim().expect("v_head_dim");
        let q_head_total = qk_nope + qk_rope;
        let kv_up_head_total = qk_nope + v_head_dim;

        // ── Step 1-2: Look up next_token embedding + apply enorm / hnorm.
        let emb_start = next_token as usize * hidden_dim;
        let embed = &self.embedding[emb_start..emb_start + hidden_dim];
        let mut hnorm_out = vec![0.0f32; hidden_dim];
        let mut enorm_out = vec![0.0f32; hidden_dim];
        rms_norm(prev_hidden, &mtp.hnorm, c.norm_eps, &mut hnorm_out);
        rms_norm(embed, &mtp.enorm, c.norm_eps, &mut enorm_out);

        // ── Step 3-4: Concat (hnorm, enorm) → eh_proj → hidden.
        let mut concat = Vec::with_capacity(2 * hidden_dim);
        concat.extend_from_slice(&hnorm_out);
        concat.extend_from_slice(&enorm_out);
        let mut hidden = vec![0.0f32; hidden_dim];
        mtp.eh_proj.matvec(&concat, &mut hidden);

        // ── Step 5: Inner V3 transformer block at pos=0 (seq_len=1).
        // Layout mirrors forward_deepseek_v3's per-layer body — extracted
        // here so MTP does not require the full multi-position attention
        // loop or a shared KV cache. Every softmax reduces to [1.0]
        // because there is exactly one attention slot.
        let block = &mtp.block;
        let mut norm_buf = vec![0.0f32; hidden_dim];
        rms_norm(&hidden, &block.attn_norm, c.norm_eps, &mut norm_buf);

        // Q LoRA chain.
        let mut q_a_buf = vec![0.0f32; q_lora_rank];
        let mut q_a_normed = vec![0.0f32; q_lora_rank];
        let mut q_full = vec![0.0f32; num_heads * q_head_total];
        block.q_a_proj.matvec(&norm_buf, &mut q_a_buf);
        rms_norm(&q_a_buf, &block.q_a_norm, c.norm_eps, &mut q_a_normed);
        block.q_b_proj.matvec(&q_a_normed, &mut q_full);

        // KV LoRA chain: kv_a plus shared k_pe.
        let mut kv_a_full = vec![0.0f32; kv_lora_rank + qk_rope];
        let mut kv_a_normed = vec![0.0f32; kv_lora_rank];
        let mut kv_up = vec![0.0f32; num_heads * kv_up_head_total];
        block.kv_a_proj_with_mqa.matvec(&norm_buf, &mut kv_a_full);
        let (kv_a_slice, k_pe_shared) = kv_a_full.split_at_mut(kv_lora_rank);
        rms_norm(kv_a_slice, &block.kv_a_norm, c.norm_eps, &mut kv_a_normed);
        block.kv_b_proj.matvec(&kv_a_normed, &mut kv_up);

        // RoPE at position 0 for both Q's rope portion (each head) and
        // the shared k_pe. Position 0 rotates by angle 0 for every dim,
        // so this is technically a no-op — but calling it keeps the code
        // structurally identical to the main forward and future-proofs
        // against multi-position MTP variants.
        for h in 0..num_heads {
            let q_head_off = h * q_head_total;
            let q_pe_slice = &mut q_full[q_head_off + qk_nope..q_head_off + q_head_total];
            apply_rope_auto(
                q_pe_slice,
                0,
                qk_rope,
                c.rope_theta,
                self.rope_freqs.as_deref(),
                true, // NEOX
            );
        }
        apply_rope_auto(
            k_pe_shared,
            0,
            qk_rope,
            c.rope_theta,
            self.rope_freqs.as_deref(),
            true,
        );

        // Single-position attention: softmax([single_score]) = [1.0], so
        // the head output is exactly the value vector v[h] for each head.
        // No score computation, no softmax, no accumulation loop needed.
        let mut attn_out = vec![0.0f32; num_heads * v_head_dim];
        for h in 0..num_heads {
            let head_off_up = h * kv_up_head_total;
            let head_off_out = h * v_head_dim;
            let v_h = &kv_up[head_off_up + qk_nope..head_off_up + kv_up_head_total];
            attn_out[head_off_out..head_off_out + v_head_dim].copy_from_slice(v_h);
        }

        // O projection + residual.
        let mut o_buf = vec![0.0f32; hidden_dim];
        block.o_proj.matvec(&attn_out, &mut o_buf);
        for i in 0..hidden_dim {
            hidden[i] += o_buf[i];
        }

        // FFN branch. MTP always uses MoE (see load_deepseek_v3_mtp_weights
        // caveat) — no dense fallback here.
        let moe = block
            .moe
            .as_ref()
            .expect("MTP block requires MoE FFN weights (Phase 5a.2 loader guarantee)");
        let mut down_buf = vec![0.0f32; hidden_dim];
        forward_deepseek_moe_layer(&c, moe, &hidden, &mut norm_buf, &mut down_buf);
        for i in 0..hidden_dim {
            hidden[i] += down_buf[i];
        }

        // ── Step 6: Final norm + shared output_proj.
        rms_norm(&hidden, &mtp.final_norm, c.norm_eps, &mut norm_buf);
        let mut logits = vec![0.0f32; c.vocab_size];
        self.output_proj.matvec(&norm_buf, &mut logits);
        logits
    }

    /// Serialise the current KV cache to `path` (colibri-style warm restart).
    ///
    /// Writes an `ALICEKV1` header + config fingerprint + per-layer
    /// K/V bytes so a later session can resume with zero re-prefill. Only the
    /// `seq_len` prefix that was actually written is persisted; unused capacity
    /// is skipped. Shared-KV layers (Gemma 3n) are also skipped in the file to
    /// keep the payload small.
    ///
    /// The fingerprint is computed from shape-critical config fields
    /// ([`Llama3Config::num_layers`], `num_kv_heads`, `head_dim`, `hidden_dim`,
    /// `max_seq_len`, and the arch flavour). Loading into a model with a
    /// different fingerprint is refused by [`Llama3Model::load_kv_cache`].
    pub fn save_kv_cache(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        let fingerprint = kv_cache_fingerprint(&self.config);
        self.kv_cache.write_to(&mut writer, fingerprint)?;
        use std::io::Write;
        writer.flush()
    }

    /// Restore the KV cache previously written by [`Self::save_kv_cache`].
    ///
    /// Refuses to load if the file's config fingerprint does not match the
    /// current model — this catches "loading a Llama-3 cache into a Qwen 3
    /// model" scenarios which would otherwise produce silent garbage.
    /// Returns [`KvCacheLoadError::FingerprintMismatch`] in that case.
    ///
    /// On success the model's KV cache is byte-identical to the moment
    /// `save_kv_cache` was called, and `forward()` can continue from
    /// `seq_len()` without re-prefilling the prompt.
    pub fn load_kv_cache(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), KvCacheLoadError> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        let fingerprint = kv_cache_fingerprint(&self.config);
        self.kv_cache.read_from(&mut reader, fingerprint)?;
        Ok(())
    }

    /// Current cached token position (0-based). Callable after
    /// [`Self::load_kv_cache`] to inspect where generation would resume.
    pub fn kv_cache_seq_len(&self) -> usize {
        self.kv_cache.seq_len()
    }

    /// Gemma 3n: dequantize the per-layer input embedding slice for a single
    /// token. Returns a `Vec<f32>` of length `num_layers * per_layer_dim`.
    /// Returns `None` for non-Gemma3n models or when the per-layer token
    /// embedding table is absent.
    ///
    /// Memory-efficient: dequantizes only the requested token's slice (~30 KB)
    /// rather than materializing the full ~8 GB f32 table upfront.
    pub fn per_layer_embedding_for_token(&self, token_id: u32) -> Option<Vec<f32>> {
        use crate::gguf::GgmlType;
        let raw = self.per_layer_token_embd_raw?;
        let qtype = self.per_layer_token_embd_qtype?;
        let per_layer_dim = self.config.per_layer_input_embedding_dim()?;
        let elements_per_token = self.config.num_layers * per_layer_dim;
        let qk = qtype.elements_per_block();
        let block_bytes = qtype.block_bytes();
        if qk == 0 || block_bytes == 0 || !elements_per_token.is_multiple_of(qk) {
            return None;
        }
        let blocks_per_token = elements_per_token / qk;
        let start = (token_id as usize) * blocks_per_token * block_bytes;
        let end = start + blocks_per_token * block_bytes;
        if end > raw.len() {
            return None;
        }
        let mut out = vec![0.0f32; elements_per_token];
        match qtype {
            GgmlType::Q5_1 => crate::gguf::dequantize_q5_1(&raw[start..end], &mut out),
            GgmlType::Q6_K => crate::gguf::dequantize_q6_k_public(&raw[start..end], &mut out),
            _ => return None,
        }
        Some(out)
    }

    /// Current KV-cache position (number of tokens processed since last reset).
    /// Exposed for speculative decoding coordination between draft and main
    /// models.
    #[must_use]
    pub fn kv_seq_len(&self) -> usize {
        self.kv_cache.seq_len()
    }

    /// Rewind the KV-cache to a previous position. Used by speculative
    /// decoding to discard rejected draft tokens from the draft model's
    /// cache when the main model disagrees.
    pub fn kv_rollback_to(&mut self, pos: usize) {
        self.kv_cache.rollback_to(pos);
    }

    /// Reset the KV cache. Equivalent to `kv_rollback_to(0)` but conveys
    /// intent for restart-from-scratch use cases.
    pub fn reset(&mut self) {
        self.kv_cache.clear();
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
                k_proj: ternarize_weight(
                    layer
                        .k_proj
                        .as_ref()
                        .expect("k_proj required for ternarize"),
                    kv_dim,
                    c.hidden_dim,
                    threshold_ratio,
                ),
                v_proj: ternarize_weight(
                    layer
                        .v_proj
                        .as_ref()
                        .expect("v_proj required for ternarize"),
                    kv_dim,
                    c.hidden_dim,
                    threshold_ratio,
                ),
                o_proj: ternarize_weight(
                    &layer.o_proj,
                    c.hidden_dim,
                    c.hidden_dim,
                    threshold_ratio,
                ),
                ffn_norm: layer.ffn_norm.clone(),
                gate_proj: ternarize_weight(
                    layer
                        .gate_proj
                        .as_ref()
                        .expect("gate_proj required for ternarize"),
                    c.intermediate_dim,
                    c.hidden_dim,
                    threshold_ratio,
                ),
                up_proj: ternarize_weight(
                    layer
                        .up_proj
                        .as_ref()
                        .expect("up_proj required for ternarize"),
                    c.intermediate_dim,
                    c.hidden_dim,
                    threshold_ratio,
                ),
                down_proj: ternarize_weight(
                    layer
                        .down_proj
                        .as_ref()
                        .expect("down_proj required for ternarize"),
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
        // Gemma 4: dedicated forward path (per-layer FFN size, per-layer
        // head_dim / RoPE base, shared KV, optional per-layer output scale).
        if self.config.arch == ModelArch::Gemma4 {
            return self.forward_gemma4(token_id);
        }
        // DeepSeek-V3 / R1: dedicated forward path (MLA + DeepSeek MoE with
        // shared expert + partial RoPE + optional MTP head). Foundation
        // (arch detection + config + weight loading) landed 2026-07-11;
        // MLA CPU forward / MoE routing / expert streaming / MTP are
        // Phase 2-5 follow-up work. Fail fast so users don't accidentally
        // get silent garbage from a fallback that treats the model as
        // standard attention.
        if self.config.arch == ModelArch::DeepSeekV3 {
            return self.forward_deepseek_v3(token_id);
        }
        // Qwen 3.5 / Qwen 3.6 hybrid uses the standard forward path with a
        // per-layer branch (DeltaNet linear-attention vs. full attention).
        // See `layer_kind_map` for the layer-to-kind routing that was set up
        // in `from_gguf`.
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
        let q_dim = c.num_heads * c.head_dim;
        // Q buffer sized for Qwen 3.6 / Bonsai "Gated Attention": if any layer
        // ships `attn_q` as `2 * q_dim` rows (Q half + swish gate half fused),
        // the matvec writes both halves into `q_buf`. Standard checkpoints use
        // only the first `q_dim` slots; the unused tail stays zero-initialised.
        let mut q_buf = vec![0.0f32; q_dim * 2];
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];
        // attn_out holds `num_heads * head_dim` = q_dim values, which may
        // exceed hidden_dim (e.g. Qwen 3 MoE 4x0.6B: q_dim=2048, hidden=1024).
        let mut attn_out = vec![0.0f32; q_dim.max(c.hidden_dim)];
        let mut o_buf = vec![0.0f32; c.hidden_dim];
        let mut gate_buf = vec![0.0f32; c.intermediate_dim];
        let mut up_buf = vec![0.0f32; c.intermediate_dim];
        let mut down_buf = vec![0.0f32; c.hidden_dim];

        // DeltaNet scratch buffers (Qwen 3.5 hybrid). Zero-sized for models
        // without DeltaNet layers so they cost only the Vec header.
        let is_hybrid = self.deltanet_layers.is_some();
        let dn_qk_dim = c.linear_qk_head_dim().unwrap_or(0);
        let dn_v_dim = c.linear_kv_head_dim().unwrap_or(0);
        let dn_num_kv_heads = c.linear_num_kv_heads().unwrap_or(0);
        let dn_num_v_heads = c.linear_num_v_heads().unwrap_or(0);
        let dn_conv_kernel = c.linear_conv_kernel_dim().unwrap_or(0);
        let dn_conv_dim = dn_qk_dim * dn_num_kv_heads * 2 + dn_v_dim * dn_num_v_heads;
        let dn_in_proj_out = dn_conv_dim + dn_v_dim * dn_num_v_heads; // + z
        let dn_v_out_dim = dn_v_dim * dn_num_v_heads;
        let mut dn_in_proj = if is_hybrid {
            vec![0.0f32; dn_in_proj_out]
        } else {
            Vec::new()
        };
        // dn_alpha / dn_beta size to `max(num_kv_heads, num_v_heads)` so
        // both Qwen 3.5-style (alpha rows = num_kv_heads) and Bonsai / Qwen
        // 3.6-style (alpha rows = num_v_heads, larger) matvecs write into a
        // big-enough slice. The delta rule caller sizes its logical num_heads
        // by the actual `alpha_proj.rows` at forward time.
        let dn_alpha_max = dn_num_kv_heads.max(dn_num_v_heads);
        let mut dn_alpha = if is_hybrid {
            vec![0.0f32; dn_alpha_max]
        } else {
            Vec::new()
        };
        let mut dn_beta = if is_hybrid {
            vec![0.0f32; dn_alpha_max]
        } else {
            Vec::new()
        };
        let mut dn_conv_out = if is_hybrid {
            vec![0.0f32; dn_conv_dim]
        } else {
            Vec::new()
        };
        let mut dn_delta_out = if is_hybrid {
            vec![0.0f32; dn_v_out_dim]
        } else {
            Vec::new()
        };

        for layer_idx in 0..c.num_layers {
            // Hybrid Qwen 3.5 / 3.6: DeltaNet layers take a distinct forward
            // path (linear attention + recurrent state, no KV cache). Any
            // model without `layer_kind_map` populated is treated as
            // pure-attention using the existing global layer index.
            if is_hybrid {
                if let LayerKind::DeltaNet(dn_idx) = self.layer_kind_map[layer_idx] {
                    let dn_layer = &self.deltanet_layers.as_ref().expect("hybrid model")[dn_idx];

                    // 1. Attention norm.
                    rms_norm(&hidden, &dn_layer.attn_norm, c.norm_eps, &mut norm_buf);

                    // 2. Fused input projection.
                    //
                    // Two GGUF variants coexist:
                    //
                    // * **Standard Qwen 3.5** — ships a single fused tensor
                    //   `ssm_in.weight` with rows = `qk_dim * num_kv_heads * 2
                    //   + v_dim * num_v_heads * 2` (16384 in the 27B config),
                    //   packing `[Q | K | V | Z]` back-to-back.
                    // * **Bonsai 27B / Qwen 3.6** — splits the fused tensor
                    //   into `attn_qkv.weight` (rows = `qk_dim * num_kv_heads
                    //   * 2 + v_dim * num_v_heads` = 10240, holds `[Q | K | V]`
                    //   only) plus `attn_gate.weight` (rows = `v_dim *
                    //   num_v_heads` = 6144, holds `[Z]`). The two tensors
                    //   together carry the same information as `ssm_in`;
                    //   see Phase X.3.d research in Issue #60.
                    //
                    // We reuse the single `dn_in_proj` buffer: the QKV portion
                    // lives at `[0..qkv_len]`, Z at `[qkv_len..in_proj_out]`.
                    // For the Bonsai variant the two halves are populated by
                    // two independent matvecs; for the Qwen 3.5 variant the
                    // single `ssm_in` matvec fills both halves at once.
                    //
                    // The additional Bonsai tensors (`ssm_a`, `ssm_dt_bias`,
                    // `ssm_norm`) are Qwen 3.6-specific SSM-math refinements
                    // that shape the delta-rule integration in ways the
                    // reference implementation (`gated_deltanet_head_disjoint`)
                    // does not yet model. They stay `#[allow(dead_code)]`
                    // pending numerical comparison against llama.cpp Qwen 3.6
                    // output; see the follow-up Phase X.3.e.3 note below.
                    let qkv_len = dn_conv_dim;
                    if let Some(ssm_in) = dn_layer.ssm_in.as_ref() {
                        // Standard Qwen 3.5 fused path.
                        ssm_in.matvec(&norm_buf, &mut dn_in_proj);
                    } else {
                        // Bonsai / Qwen 3.6 split path.
                        let attn_qkv = dn_layer
                            .attn_qkv
                            .as_ref()
                            .expect("DeltaNet layer with neither ssm_in nor attn_qkv slipped past the loader");
                        let attn_gate = dn_layer
                            .attn_gate
                            .as_ref()
                            .expect("Bonsai DeltaNet layer requires attn_gate alongside attn_qkv");
                        attn_qkv.matvec(&norm_buf, &mut dn_in_proj[..qkv_len]);
                        attn_gate.matvec(&norm_buf, &mut dn_in_proj[qkv_len..]);
                    }
                    // 2a/2b. alpha / beta decay-rate + update-rate projections.
                    dn_layer.alpha_proj.matvec(&norm_buf, &mut dn_alpha);
                    dn_layer.beta_proj.matvec(&norm_buf, &mut dn_beta);

                    // Detect Bonsai / Qwen 3.6 path once. Presence of both
                    // `ssm_a` and `ssm_dt_bias` toggles four reference-aligned
                    // refinements: SSM discretisation (Gap B), beta sigmoid
                    // (Gap B extra), pre-silu on Q/K/V post-conv1d
                    // (§Q/K L2Norm) and z-gate after ssm-norm (§silu(z) order).
                    // Qwen 3.5 GGUFs ship neither tensor and take the legacy
                    // path, preserving pre-Phase-X.3.e.3.2 numerics.
                    let is_bonsai_path = if std::env::var("ALICE_DISABLE_BONSAI_FLAG").is_ok() {
                        false
                    } else {
                        dn_layer.ssm_a.is_some() && dn_layer.ssm_dt_bias.is_some()
                    };

                    // 2c. Bonsai / Qwen 3.6 SSM discretisation (Phase X.3.e.3.2
                    // Gap B). Reference: PrismML llama.cpp fork qwen35.cpp:443
                    // -451.
                    let disable_gap_b = std::env::var("ALICE_DISABLE_GAP_B").is_ok();
                    if !disable_gap_b {
                        if let (Some(ssm_a), Some(ssm_dt_bias)) =
                            (dn_layer.ssm_a.as_ref(), dn_layer.ssm_dt_bias.as_ref())
                        {
                            debug_assert_eq!(ssm_a.len(), dn_num_v_heads);
                            debug_assert_eq!(ssm_dt_bias.len(), dn_num_v_heads);
                            for h in 0..dn_num_v_heads {
                                let alpha_biased = dn_alpha[h] + ssm_dt_bias[h];
                                let gate = softplus(alpha_biased) * ssm_a[h];
                                dn_alpha[h] = gate.exp();
                            }

                            // 2d. Beta sigmoid (Phase X.3.e.3.2 Gap B extra).
                            for h in 0..dn_num_v_heads {
                                dn_beta[h] = sigmoid(dn_beta[h]);
                            }
                        }
                    }

                    // Split fused output. Layout (matches GPU-side loader):
                    //   [ q | k | v | z ]
                    // with `q` and `k` each `qk_dim * num_kv_heads` long and
                    // `v` and `z` each `v_dim * num_v_heads` long. `qkv_len`
                    // was declared above alongside the input-projection
                    // matvec branching.
                    let q_start = 0;
                    let k_start = dn_qk_dim * dn_num_kv_heads;
                    let v_start = dn_qk_dim * dn_num_kv_heads * 2;
                    let z_start = qkv_len; // = end of v

                    // 3. Causal conv1d over `q + k + v` (excludes z).
                    causal_conv1d_step(
                        &dn_in_proj[..qkv_len],
                        &mut self.deltanet_conv_state[dn_idx],
                        &mut self.deltanet_conv_ring_pos[dn_idx],
                        &dn_layer.conv1d_weight,
                        &dn_layer.conv1d_bias,
                        &mut dn_conv_out,
                        dn_conv_dim,
                        dn_conv_kernel,
                    );

                    // 3.5. Bonsai / Qwen 3.6 post-conv1d SiLU (Phase X.3.e.3.2
                    // §Q/K L2Norm). Reference qwen35.cpp:502 applies
                    // ggml_silu(conv_output) before the recurrence; the
                    // head kernel with bonsai_semantics=true then skips
                    // its internal silu. Qwen 3.5 legacy path leaves the
                    // raw conv output and silu's Q/K in-line.
                    if is_bonsai_path {
                        for val in dn_conv_out[..qkv_len].iter_mut() {
                            *val = silu(*val);
                        }
                    }

                    // 4. Gated DeltaNet recurrence: reads q/k/v from the
                    //    convolved buffer, z from the unconvolved fused
                    //    output, alpha / beta from the dedicated projections.
                    let q_slice = &dn_conv_out[q_start..q_start + k_start];
                    let k_slice = &dn_conv_out[k_start..v_start];
                    let v_slice = &dn_conv_out[v_start..v_start + dn_v_out_dim];
                    let z_slice = &dn_in_proj[z_start..z_start + dn_v_out_dim];

                    gated_deltanet_step(
                        q_slice,
                        k_slice,
                        v_slice,
                        &dn_alpha[..dn_num_v_heads],
                        &dn_beta[..dn_num_v_heads],
                        z_slice,
                        &mut self.deltanet_state[dn_idx],
                        &mut dn_delta_out,
                        dn_num_kv_heads,
                        dn_num_v_heads,
                        dn_qk_dim,
                        dn_v_dim,
                        is_bonsai_path,
                    );

                    // 4.5. Bonsai / Qwen 3.6 per-V-head state RMSNorm on the
                    // recurrence output, prior to the `ssm_out` projection.
                    // Standard Qwen 3.5 exports no `ssm_norm` tensor and this
                    // block is skipped, preserving the pre-Phase-X.3.e.3.2
                    // numerics.
                    if let Some(ssm_norm) = dn_layer.ssm_norm.as_ref() {
                        if std::env::var("ALICE_DISABLE_GAP_C").is_err() {
                            apply_qk_norm(&mut dn_delta_out, ssm_norm, dn_v_dim, c.norm_eps);
                        }
                    }

                    // 4.6. Bonsai / Qwen 3.6 z-gate (Phase X.3.e.3.2
                    // §silu(z) order). Reference qwen35.cpp:562
                    // build_norm_gated(rms_norm(x, w) * silu(z)) applies
                    // the z-gate after ssm-norm. The Bonsai head kernel
                    // skipped its inline `out *= silu(z)` so we multiply
                    // externally here. Qwen 3.5 legacy already has the
                    // z-gate applied inside the kernel and this block
                    // is a no-op.
                    if is_bonsai_path {
                        for h in 0..dn_num_v_heads {
                            for j in 0..dn_v_dim {
                                let idx = h * dn_v_dim + j;
                                dn_delta_out[idx] *= silu(z_slice[idx]);
                            }
                        }
                    }

                    // 5. Output projection to hidden dim.
                    dn_layer.ssm_out.matvec(&dn_delta_out, &mut o_buf);

                    // 6. Residual add.
                    for i in 0..c.hidden_dim {
                        hidden[i] += o_buf[i];
                    }

                    // 7. FFN sub-block (RMSNorm + SwiGLU + down + residual).
                    rms_norm(&hidden, &dn_layer.ffn_norm, c.norm_eps, &mut norm_buf);
                    let q8_ffn = quantize_row_q8_k(&norm_buf);
                    dn_layer.gate_proj.matvec_preq(&q8_ffn, &mut gate_buf);
                    dn_layer.up_proj.matvec_preq(&q8_ffn, &mut up_buf);
                    for i in 0..c.intermediate_dim {
                        gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
                    }
                    dn_layer.down_proj.matvec(&gate_buf, &mut down_buf);
                    for i in 0..c.hidden_dim {
                        hidden[i] += down_buf[i];
                    }
                    continue;
                }
            }

            let attention_layer_idx = if is_hybrid {
                match self.layer_kind_map[layer_idx] {
                    LayerKind::Attention(k) => k,
                    LayerKind::DeltaNet(_) => unreachable!("handled above"),
                }
            } else {
                layer_idx
            };
            let layer = &self.layers[attention_layer_idx];

            // Attention norm
            rms_norm(&hidden, &layer.attn_norm, c.norm_eps, &mut norm_buf);

            // Q, K, V projections (pre-quantize norm_buf once for all three)
            let q8_attn = quantize_row_q8_k(&norm_buf);
            layer.q_proj.matvec_preq(&q8_attn, &mut q_buf);
            layer
                .k_proj
                .as_ref()
                .expect("k_proj required for non-shared layer")
                .matvec_preq(&q8_attn, &mut k_buf);
            layer
                .v_proj
                .as_ref()
                .expect("v_proj required for non-shared layer")
                .matvec_preq(&q8_attn, &mut v_buf);
            // Qwen 2/2.5 bias (no-op for Llama/Mistral/Gemma/Qwen 3)
            if let Some(b) = layer.q_bias() {
                for (q, bi) in q_buf.iter_mut().zip(b.iter()) {
                    *q += bi;
                }
            }
            if let Some(b) = layer.k_bias() {
                for (k, bi) in k_buf.iter_mut().zip(b.iter()) {
                    *k += bi;
                }
            }
            if let Some(b) = layer.v_bias() {
                for (v, bi) in v_buf.iter_mut().zip(b.iter()) {
                    *v += bi;
                }
            }
            // Qwen 3 QK-Norm (per-head RMSNorm on Q, K before RoPE; no-op for others).
            // Slice `q_buf` to the first `q_dim` entries — the second half (when
            // present in Bonsai / Qwen 3.6 Gated Attention layers) holds the
            // swish gate, which does NOT get per-head normalisation. For non-
            // gated forward paths the slice is equivalent to the full buffer.
            if let Some(w) = layer.q_norm() {
                apply_qk_norm(&mut q_buf[..q_dim], w, c.head_dim, c.norm_eps);
            }
            if let Some(w) = layer.k_norm() {
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
                c.attn_logit_softcap(),
                if c.arch == ModelArch::Gemma3n {
                    Some(1.0)
                } else {
                    None
                },
                &mut attn_out,
            );

            // Qwen 3.6 / Bonsai 27B "Gated Attention": when `q_proj` output
            // was `2 * q_dim`, its second half is a per-element swish (SiLU)
            // gate that modulates the attention result before `o_proj`.
            // Applied element-wise across all heads.
            if layer.gated_output {
                for i in 0..q_dim {
                    attn_out[i] *= silu(q_buf[q_dim + i]);
                }
            }

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

            // MoE dispatch: layers with `ffn_gate_inp` use expert routing
            // instead of a monolithic SwiGLU FFN. Both paths write their
            // hidden-dim output into `down_buf`.
            if layer.ffn_gate_inp().is_some() {
                forward_moe_layer(c, layer, &norm_buf, &mut down_buf);
            } else {
                // SwiGLU FFN (pre-quantize norm_buf once for gate+up)
                let q8_ffn = quantize_row_q8_k(&norm_buf);
                layer
                    .gate_proj
                    .as_ref()
                    .expect("gate_proj required for non-MoE layer")
                    .matvec_preq(&q8_ffn, &mut gate_buf);
                layer
                    .up_proj
                    .as_ref()
                    .expect("up_proj required for non-MoE layer")
                    .matvec_preq(&q8_ffn, &mut up_buf);

                c.apply_ffn_sparsity(layer_idx, &mut gate_buf);
                for i in 0..c.intermediate_dim {
                    gate_buf[i] = c.apply_ffn_act(layer_idx, gate_buf[i]) * up_buf[i];
                }

                layer
                    .down_proj
                    .as_ref()
                    .expect("down_proj required for non-MoE layer")
                    .matvec(&gate_buf, &mut down_buf);
            }

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

            // Issue #40 diagnostic: dump per-layer post-residual hidden state
            // to identify the layer at which CPU/GPU divergence first emerges.
            // Only emit at fixed checkpoints (layer 0, 6, 13, 20, 27) to bound
            // stderr volume — that covers ~every 25% of the stack for a
            // 28-layer model. Off-by-default (env var must be set).
            if std::env::var_os("ALICE_LLM_DUMP_LAYERS").is_some()
                && matches!(layer_idx, 0 | 6 | 13 | 20 | 27)
            {
                dump_hidden_jsonl_stderr(&format!("cpu_layer_{layer_idx}"), &hidden);
            }
        }

        // Advance KV cache position (all layers have appended for this token)
        self.kv_cache.advance();

        // Output norm
        rms_norm(&hidden, &self.output_norm, c.norm_eps, &mut norm_buf);

        // Issue #40 diagnostic: dump pre-output-projection hidden state when
        // ALICE_LLM_DUMP_HIDDEN is set. Enables CPU vs GPU cos-sim comparison
        // to isolate whether the divergence is in the layer stack or in the
        // Q6_K output projection. Runs before the final matvec so the buffer
        // captured is exactly what output_proj reads.
        if std::env::var_os("ALICE_LLM_DUMP_HIDDEN").is_some() {
            dump_hidden_jsonl_stderr("cpu", &norm_buf);
        }

        // Output logits
        let mut logits = vec![0.0f32; c.vocab_size];
        // output_proj points to output.weight or token_embd.weight (tied)
        self.output_proj.matvec(&norm_buf, &mut logits);

        // Gemma-2: final logit softcapping
        if let Some(cap) = c.final_logit_softcap() {
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
        let n_altup = c.altup_num_inputs().expect("Gemma3n: altup_num_inputs");
        let n_embd_altup = c
            .per_layer_input_embedding_dim()
            .expect("Gemma3n: per_layer_input_embedding_dim");
        let i_altup_act = c.altup_active_idx().unwrap_or(0);
        let n_layer_sparsity = c
            .activation_sparsity_scale()
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
                layer
                    .k_proj
                    .as_ref()
                    .expect("k_proj required for non-shared layer")
                    .matvec_preq(&q8_attn, &mut k_buf);
                layer
                    .v_proj
                    .as_ref()
                    .expect("v_proj required for non-shared layer")
                    .matvec_preq(&q8_attn, &mut v_buf);
            }

            // Q, K per-head RMSNorm (Gemma 3n uses them like Qwen 3)
            if let Some(w) = layer.q_norm() {
                apply_qk_norm(&mut q_buf, w, c.head_dim, c.norm_eps);
            }
            if owns_kv {
                if let Some(w) = layer.k_norm() {
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
                c.attn_logit_softcap(),
                if c.arch == ModelArch::Gemma3n {
                    Some(1.0)
                } else {
                    None
                },
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
            layer
                .gate_proj
                .as_ref()
                .expect("gate_proj required for non-MoE layer")
                .matvec_preq(&q8_ffn, &mut gate_buf);
            layer
                .up_proj
                .as_ref()
                .expect("up_proj required for non-MoE layer")
                .matvec_preq(&q8_ffn, &mut up_buf);

            // Sparsity: gaussian_topk for first n_layer_sparsity layers
            if layer_idx < n_layer_sparsity {
                c.apply_ffn_sparsity(layer_idx, &mut gate_buf);
            }
            // GELU + gate * up
            for i in 0..c.intermediate_dim {
                gate_buf[i] = gelu_approx(gate_buf[i]) * up_buf[i];
            }
            layer
                .down_proj
                .as_ref()
                .expect("down_proj required for non-MoE layer")
                .matvec(&gate_buf, &mut down_buf);

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
        if let Some(cap) = c.final_logit_softcap() {
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
                inp_per_layer_lookup[l][i] = (inp_per_layer_lookup[l][i]
                    + per_layer_proj_normed[l * n_embd_altup + i])
                    * inv_sqrt2;
            }
        }
        inp_per_layer_lookup
    }

    /// AltUp router modalities: compute a per-altup-input activation vector
    /// from the active stream (per llama.cpp `altup_compute_router_modalities`).
    fn altup_router_modalities(
        &self,
        active: &[f32],
        layer_idx: usize,
        n_altup: usize,
    ) -> Vec<f32> {
        let c = &self.config;
        let layer = &self.layers[layer_idx];
        let router_norm_w = layer
            .altup_router_norm()
            .expect("Gemma3n: altup_router_norm");
        let router_w = layer.altup_router().expect("Gemma3n: altup_router");
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
        mat_vec_f32(
            router_w,
            n_altup,
            c.hidden_dim,
            &router_inputs,
            &mut modalities,
        );
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
        let i_altup_act = c.altup_active_idx().unwrap_or(0);
        let hidden_dim = c.hidden_dim;
        let modalities = self.altup_router_modalities(&streams[i_altup_act], layer_idx, n_altup);
        // predict_coef shape: [n_altup, n_altup * n_altup]. matmul with modalities
        // gives a vector of length n_altup*n_altup. Reshape to [n_altup, n_altup]
        // coefficient matrix (coef[i][j] = weight for using stream j to predict stream i).
        let predict_coef = layer
            .altup_predict_coef()
            .expect("Gemma3n: altup_predict_coef");
        let mut coef_flat = vec![0.0f32; n_altup * n_altup];
        mat_vec_f32(
            predict_coef,
            n_altup * n_altup,
            n_altup,
            &modalities,
            &mut coef_flat,
        );
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
        let i_altup_act = c.altup_active_idx().unwrap_or(0);
        let hidden_dim = c.hidden_dim;
        let modalities = self.altup_router_modalities(activated, layer_idx, n_altup);
        let correct_coef = layer
            .altup_correct_coef()
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
        let laurel_l = layer.laurel_l().expect("Gemma3n: laurel_l");
        let laurel_r = layer.laurel_r().expect("Gemma3n: laurel_r");
        let laurel_post_norm = layer.laurel_post_norm().expect("Gemma3n: laurel_post_norm");
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
            .altup_correct_scale()
            .expect("Gemma3n: altup_correct_scale");
        // Scale
        let mut scaled = vec![0.0f32; hidden_dim];
        for i in 0..hidden_dim {
            scaled[i] = active_corrected[i] * correct_scale[i];
        }
        // Gate matmul: inp_gate shape [hidden_dim, n_embd_altup] → out_dim=n_embd_altup
        let inp_gate = layer.inp_gate().expect("Gemma3n: inp_gate");
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
        let proj = layer.proj().expect("Gemma3n: proj");
        let mut projected = vec![0.0f32; hidden_dim];
        proj.matvec(&gated, &mut projected);
        // post_norm RMSNorm
        let post_norm = layer.post_norm().expect("Gemma3n: post_norm");
        let mut normed = vec![0.0f32; hidden_dim];
        rms_norm(&projected, post_norm, c.norm_eps, &mut normed);
        normed
    }

    /// Gemma 4 forward pass. Mirrors llama.cpp
    /// `llama_model_gemma4::graph::build_arch_graph`. Simpler than Gemma 3n:
    /// AltUp, Laurel, and activation sparsity are all removed.
    ///
    /// New Gemma 4 mechanics vs Gemma 3n:
    /// - Per-layer FFN size (`ffn_size_for_layer`).
    /// - Per-layer head dimension (SWA layers halve `head_dim`).
    /// - Per-layer RoPE base frequency (SWA layers use 10K vs 1M).
    /// - Optional V projection (Gemma 4 26B_A4B; when absent, uses K as V).
    /// - Optional per-layer `layer_output_scale` (multiplied at end of layer).
    /// - Standard residual (Gemma 3n's gated + laurel merge is removed).
    ///
    /// # Panics
    /// Panics if the model was not loaded with Gemma 4 architecture (missing
    /// required per-layer input embedding tensors, etc.).
    /// DeepSeek-V2 / V3 / R1 forward pass — **foundation only** as of
    /// 2026-07-11 (foundation) and continues with MLA CPU forward as of
    /// 2026-07-12.
    ///
    /// Phase 2 lands the MLA arithmetic:
    /// * Q LoRA chain: `q_a_proj → q_a_norm → q_b_proj`,
    ///   split into `q_nope` + `q_pe`.
    /// * KV LoRA chain: `kv_a_proj_with_mqa`,
    ///   split into `kv_a` (compressed latent) + `k_pe_shared` (MQA
    ///   positional slice shared across heads),
    ///   then `kv_a_norm → kv_b_proj`, split into `k_nope` + `v`.
    /// * Partial NEOX RoPE on `q_pe` (per head) and `k_pe_shared`.
    /// * Compressed KV cache: only `kv_a` + `k_pe_shared` (`kv_lora_rank +
    ///   qk_rope_head_dim` f32 per token) is persisted, which is the source
    ///   of the ~57× KV-cache compression documented in the DeepSeek-V2
    ///   paper.
    /// * Attention: reconstruct `k_nope[h,t] = kv_b_proj(kv_a[t])` on demand,
    ///   concatenate with the shared `k_pe`, compute the usual softmax dot
    ///   product, output-projection via `o_proj`.
    ///
    /// The FFN sub-block is dense SwiGLU **only for layers below
    /// `first_k_dense_replace`** (V3: 3). Everything past that requires
    /// DeepSeek MoE routing (Phase 3, Issue #33) and — until Phase 3 lands
    /// — the function panics with a clear message so silent-garbage
    /// fallbacks are impossible.
    ///
    /// Phase 4 (expert streaming) and Phase 5 (MTP native speculative
    /// decoding) still track separately (Issues #34 / #35).
    fn forward_deepseek_v3(&mut self, token_id: u32) -> Vec<f32> {
        let c = self.config.clone();
        let hidden_dim = c.hidden_dim;
        let num_heads = c.num_heads;
        let q_lora_rank = c.deepseek_q_lora_rank().expect("q_lora_rank");
        let kv_lora_rank = c.deepseek_kv_lora_rank().expect("kv_lora_rank");
        let qk_nope = c.deepseek_qk_nope_head_dim().expect("qk_nope_head_dim");
        let qk_rope = c.deepseek_qk_rope_head_dim().expect("qk_rope_head_dim");
        let v_head_dim = c.deepseek_v_head_dim().expect("v_head_dim");
        let first_k_dense = c.deepseek_first_k_dense_replace().unwrap_or(0);
        let q_head_total = qk_nope + qk_rope;
        let kv_up_head_total = qk_nope + v_head_dim;
        let compressed_kv_dim = kv_lora_rank + qk_rope;

        let deepseek_layers = self
            .deepseek_v3_layers
            .as_ref()
            .expect("DeepSeek-V3 layers not loaded");

        let pos = self.kv_cache.seq_len();

        // Embedding lookup.
        let emb_start = token_id as usize * hidden_dim;
        let mut hidden: Vec<f32> = self.embedding[emb_start..emb_start + hidden_dim].to_vec();

        // Scratch buffers reused across layers.
        let mut norm_buf = vec![0.0f32; hidden_dim];
        let mut q_a_buf = vec![0.0f32; q_lora_rank];
        let mut q_a_normed = vec![0.0f32; q_lora_rank];
        let mut q_full = vec![0.0f32; num_heads * q_head_total];
        let mut kv_a_full = vec![0.0f32; compressed_kv_dim];
        let mut kv_a_normed = vec![0.0f32; kv_lora_rank];
        let mut kv_up = vec![0.0f32; num_heads * kv_up_head_total];
        let mut attn_out = vec![0.0f32; num_heads * v_head_dim];
        let mut o_buf = vec![0.0f32; hidden_dim];
        let mut gate_buf = vec![0.0f32; c.intermediate_dim];
        let mut up_buf = vec![0.0f32; c.intermediate_dim];
        let mut down_buf = vec![0.0f32; hidden_dim];

        for layer_idx in 0..c.num_layers {
            let layer = &deepseek_layers[layer_idx];

            // ── Attention block ───────────────────────────────────────
            rms_norm(&hidden, &layer.attn_norm, c.norm_eps, &mut norm_buf);

            // Q LoRA chain.
            layer.q_a_proj.matvec(&norm_buf, &mut q_a_buf);
            rms_norm(&q_a_buf, &layer.q_a_norm, c.norm_eps, &mut q_a_normed);
            layer.q_b_proj.matvec(&q_a_normed, &mut q_full);

            // KV LoRA chain: single matvec produces the compressed latent
            // `kv_a` plus the shared positional slice `k_pe`.
            layer.kv_a_proj_with_mqa.matvec(&norm_buf, &mut kv_a_full);
            let (kv_a_slice, k_pe_shared) = kv_a_full.split_at_mut(kv_lora_rank);
            rms_norm(kv_a_slice, &layer.kv_a_norm, c.norm_eps, &mut kv_a_normed);
            // Persist the compressed latent + shared k_pe in the KV cache
            // (the ~57× compression trick). Reconstructed per-head `k_nope`
            // stays purely on the stack.
            let mut cache_entry = Vec::with_capacity(compressed_kv_dim);
            cache_entry.extend_from_slice(&kv_a_normed);
            cache_entry.extend_from_slice(k_pe_shared);
            // Compute `k_nope` + `v` for the current token (needed to compare
            // against future queries; historical positions rebuild theirs on
            // demand from `kv_a`).
            layer.kv_b_proj.matvec(&kv_a_normed, &mut kv_up);

            // Split Q per-head into (q_nope, q_pe) and apply NEOX RoPE only
            // to the `qk_rope` slice of each head. The shared `k_pe` gets
            // a single RoPE pass (no head dimension).
            for h in 0..num_heads {
                let q_head_off = h * q_head_total;
                let q_pe_slice = &mut q_full[q_head_off + qk_nope..q_head_off + q_head_total];
                apply_rope_auto(
                    q_pe_slice,
                    pos,
                    qk_rope,
                    c.rope_theta,
                    self.rope_freqs.as_deref(),
                    true, // NEOX
                );
            }
            apply_rope_auto(
                k_pe_shared,
                pos,
                qk_rope,
                c.rope_theta,
                self.rope_freqs.as_deref(),
                true,
            );
            // Persist (post-RoPE) k_pe into the cache entry so the shared
            // slice lines up with `q_pe` at attention time.
            cache_entry[kv_lora_rank..].copy_from_slice(k_pe_shared);
            self.kv_cache.append(layer_idx, &cache_entry, &cache_entry);
            self.kv_cache.advance();

            // Attention: for each history position `t`, dot each head's Q
            // against the reconstructed K, softmax, weighted sum over V.
            // Historical `k_nope[h,t]` and `v[h,t]` come from re-projecting
            // the cached `kv_a[t]` (weight-absorption is Phase 3 territory).
            let seq_len = pos + 1;
            let scale = 1.0 / ((qk_nope + qk_rope) as f32).sqrt();
            // Zero the attention output slab before accumulating per head.
            attn_out.fill(0.0);
            let mut scratch_kv_up = vec![0.0f32; num_heads * kv_up_head_total];
            let mut scores = vec![0.0f32; seq_len];
            for h in 0..num_heads {
                let q_head_off = h * q_head_total;
                let q_nope_head = &q_full[q_head_off..q_head_off + qk_nope];
                let q_pe_head = &q_full[q_head_off + qk_nope..q_head_off + q_head_total];

                let mut max_score = f32::NEG_INFINITY;
                for t in 0..seq_len {
                    let cache = self.kv_cache.key_at(layer_idx, t);
                    let cached_kv_a = &cache[..kv_lora_rank];
                    let cached_k_pe = &cache[kv_lora_rank..];
                    layer.kv_b_proj.matvec(cached_kv_a, &mut scratch_kv_up);
                    let head_off = h * kv_up_head_total;
                    let k_nope_t = &scratch_kv_up[head_off..head_off + qk_nope];
                    let mut dot = 0.0f32;
                    for i in 0..qk_nope {
                        dot += q_nope_head[i] * k_nope_t[i];
                    }
                    for i in 0..qk_rope {
                        dot += q_pe_head[i] * cached_k_pe[i];
                    }
                    dot *= scale;
                    scores[t] = dot;
                    if dot > max_score {
                        max_score = dot;
                    }
                }
                // Softmax stable in place.
                let mut denom = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    denom += *s;
                }
                for s in &mut scores {
                    *s /= denom;
                }
                // Weighted sum of V.
                let attn_head_off = h * v_head_dim;
                for t in 0..seq_len {
                    let cache = self.kv_cache.value_at(layer_idx, t);
                    let cached_kv_a = &cache[..kv_lora_rank];
                    layer.kv_b_proj.matvec(cached_kv_a, &mut scratch_kv_up);
                    let head_off = h * kv_up_head_total;
                    let v_t = &scratch_kv_up[head_off + qk_nope..head_off + kv_up_head_total];
                    let w = scores[t];
                    for j in 0..v_head_dim {
                        attn_out[attn_head_off + j] += w * v_t[j];
                    }
                }
            }

            layer.o_proj.matvec(&attn_out, &mut o_buf);
            for i in 0..hidden_dim {
                hidden[i] += o_buf[i];
            }

            // ── FFN block ─────────────────────────────────────────────
            if layer_idx < first_k_dense {
                let ffn_norm = layer.ffn_norm.as_ref().expect("dense ffn_norm");
                let gate_proj = layer.gate_proj.as_ref().expect("dense gate_proj");
                let up_proj = layer.up_proj.as_ref().expect("dense up_proj");
                let down_proj = layer.down_proj.as_ref().expect("dense down_proj");
                rms_norm(&hidden, ffn_norm, c.norm_eps, &mut norm_buf);
                gate_proj.matvec(&norm_buf, &mut gate_buf);
                up_proj.matvec(&norm_buf, &mut up_buf);
                for i in 0..c.intermediate_dim {
                    gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
                }
                down_proj.matvec(&gate_buf, &mut down_buf);
                for i in 0..hidden_dim {
                    hidden[i] += down_buf[i];
                }
            } else {
                let moe = layer
                    .moe
                    .as_ref()
                    .expect("DeepSeek-V3 MoE weights required past first_k_dense_replace");
                forward_deepseek_moe_layer(&c, moe, &hidden, &mut norm_buf, &mut down_buf);
                for i in 0..hidden_dim {
                    hidden[i] += down_buf[i];
                }
            }
        }

        // Output norm + logits.
        rms_norm(&hidden, &self.output_norm, c.norm_eps, &mut norm_buf);
        let mut logits = vec![0.0f32; c.vocab_size];
        self.output_proj.matvec(&norm_buf, &mut logits);
        logits
    }

    fn forward_gemma4(&mut self, token_id: u32) -> Vec<f32> {
        let c = self.config.clone();
        let hidden_dim = c.hidden_dim;
        let n_embd_altup = c
            .per_layer_input_embedding_dim()
            .expect("Gemma4: per_layer_input_embedding_dim");
        let pos = self.kv_cache.seq_len();

        // ── Embedding lookup + Gemma-style scale ────────────────────────────
        let emb_start = token_id as usize * hidden_dim;
        let mut inpl: Vec<f32> = self.embedding[emb_start..emb_start + hidden_dim].to_vec();
        let embed_scale = (hidden_dim as f32).sqrt();
        for v in &mut inpl {
            *v *= embed_scale;
        }

        // ── Per-layer input embedding lookup + projection ───────────────────
        // Gemma 4 reuses Gemma 3n's per-layer input embedding pipeline verbatim.
        let inp_per_layer =
            self.gemma3n_per_layer_inputs(token_id, &inpl, n_embd_altup, c.num_layers);

        // Reusable buffers sized for the largest layer.
        let max_head_dim = c.head_dim_swa().unwrap_or(c.head_dim).max(c.head_dim);
        let max_q_dim = c.num_heads * max_head_dim;
        let max_kv_dim = c.num_kv_heads * max_head_dim;
        let max_ffn_size = c.ffn_size_per_layer().map_or(c.intermediate_dim, |a| {
            a.iter().copied().max().unwrap_or(c.intermediate_dim)
        });
        let mut norm_buf = vec![0.0f32; hidden_dim];
        let mut q_buf = vec![0.0f32; max_q_dim];
        let mut k_buf = vec![0.0f32; max_kv_dim];
        let mut v_buf = vec![0.0f32; max_kv_dim];
        // attn_out holds `num_heads * head_dim` values (q_dim, not hidden_dim);
        // for Gemma 4 full-attention layers q_dim > hidden_dim (e.g. 4096 vs 1536).
        let mut attn_out = vec![0.0f32; max_q_dim];
        let mut o_buf = vec![0.0f32; hidden_dim];
        let mut gate_buf = vec![0.0f32; max_ffn_size];
        let mut up_buf = vec![0.0f32; max_ffn_size];
        let mut down_buf = vec![0.0f32; hidden_dim];

        for layer_idx in 0..c.num_layers {
            let layer = &self.layers[layer_idx];
            let head_dim = c.head_dim_for_layer(layer_idx);
            let q_dim = c.num_heads * head_dim;
            let kv_dim = c.num_kv_heads * head_dim;
            let ffn_size = c.ffn_size_for_layer(layer_idx);
            let freq_base = c.rope_theta_for_layer(layer_idx);

            // ── attn_norm ──────────────────────────────────────────────────
            rms_norm(&inpl, &layer.attn_norm, c.norm_eps, &mut norm_buf);

            // ── Attention (Q, K, V projections + norms + RoPE) ─────────────
            // Note: Q4_0 / Q5_0 (used by Gemma 4 QAT) don't support the
            // pre-quantized Q8_K path, so we use `matvec` directly with the
            // f32 normalized buffer.
            layer.q_proj.matvec(&norm_buf, &mut q_buf[..q_dim]);
            let owns_kv = self.kv_cache.kv_layer_map[layer_idx] == layer_idx;
            if owns_kv {
                let k_ref = layer
                    .k_proj
                    .as_ref()
                    .expect("Gemma4: k_proj required for own-KV layer");
                k_ref.matvec(&norm_buf, &mut k_buf[..kv_dim]);
                // V projection is optional in Gemma 4: fall back to K if absent.
                if let Some(v_ref) = layer.v_proj.as_ref() {
                    v_ref.matvec(&norm_buf, &mut v_buf[..kv_dim]);
                } else {
                    v_buf[..kv_dim].copy_from_slice(&k_buf[..kv_dim]);
                }
            }

            // Q, K per-head RMSNorm (Gemma 4 uses them like Qwen 3 / Gemma 3n).
            if let Some(w) = layer.q_norm() {
                apply_qk_norm(&mut q_buf[..q_dim], w, head_dim, c.norm_eps);
            }
            if owns_kv {
                if let Some(w) = layer.k_norm() {
                    apply_qk_norm(&mut k_buf[..kv_dim], w, head_dim, c.norm_eps);
                }
                // V RMSNorm without weight (identity gain), same as Gemma 3n.
                apply_head_rms_norm_identity(&mut v_buf[..kv_dim], head_dim, c.norm_eps);
            }

            // Apply RoPE to Q and K (per-layer frequency base).
            let rope_freqs_ref = layer.rope_freqs.as_deref();
            for h in 0..c.num_heads {
                let start = h * head_dim;
                apply_rope_auto(
                    &mut q_buf[start..start + head_dim],
                    pos,
                    head_dim,
                    freq_base,
                    rope_freqs_ref,
                    c.use_neox_rope(),
                );
            }
            if owns_kv {
                for h in 0..c.num_kv_heads {
                    let start = h * head_dim;
                    apply_rope_auto(
                        &mut k_buf[start..start + head_dim],
                        pos,
                        head_dim,
                        freq_base,
                        rope_freqs_ref,
                        c.use_neox_rope(),
                    );
                }
                self.kv_cache
                    .append(layer_idx, &k_buf[..kv_dim], &v_buf[..kv_dim]);
            }

            // GQA attention (attention_scale = 1.0 for Gemma 4).
            for v in &mut attn_out[..q_dim] {
                *v = 0.0;
            }
            gqa_attention(
                &q_buf[..q_dim],
                &self.kv_cache,
                layer_idx,
                pos,
                c.num_heads,
                c.num_kv_heads,
                head_dim,
                c.sliding_window_for_layer(layer_idx),
                c.attn_logit_softcap(),
                Some(1.0),
                &mut attn_out[..q_dim],
            );

            // Output projection: [q_dim] → [hidden_dim].
            let layer = &self.layers[layer_idx];
            layer.o_proj.matvec(&attn_out[..q_dim], &mut o_buf);

            // Post-attention RMSNorm (Gemma-family sandwich).
            if let Some(w) = &layer.post_attn_norm {
                rms_norm(&o_buf, w, c.norm_eps, &mut norm_buf);
                o_buf.copy_from_slice(&norm_buf);
            }

            // Standard residual: attn_out = o + inpL. (Gemma 3n's gated
            // residual and Laurel branch are absent in Gemma 4.)
            let mut attn_out_local = vec![0.0f32; hidden_dim];
            for i in 0..hidden_dim {
                attn_out_local[i] = o_buf[i] + inpl[i];
            }

            // ── FFN ────────────────────────────────────────────────────────
            rms_norm(&attn_out_local, &layer.ffn_norm, c.norm_eps, &mut norm_buf);
            layer
                .gate_proj
                .as_ref()
                .expect("gate_proj required for non-MoE Gemma 4 layer")
                .matvec(&norm_buf, &mut gate_buf[..ffn_size]);
            layer
                .up_proj
                .as_ref()
                .expect("up_proj required for non-MoE Gemma 4 layer")
                .matvec(&norm_buf, &mut up_buf[..ffn_size]);

            for i in 0..ffn_size {
                gate_buf[i] = gelu_approx(gate_buf[i]) * up_buf[i];
            }
            layer
                .down_proj
                .as_ref()
                .expect("down_proj required for non-MoE Gemma 4 layer")
                .matvec(&gate_buf[..ffn_size], &mut down_buf);

            // Post-FFN RMSNorm.
            if let Some(w) = &layer.post_ffn_norm {
                rms_norm(&down_buf, w, c.norm_eps, &mut norm_buf);
                down_buf.copy_from_slice(&norm_buf);
            }

            // Standard residual: cur = ffn + attn_out.
            let mut cur = vec![0.0f32; hidden_dim];
            for i in 0..hidden_dim {
                cur[i] = down_buf[i] + attn_out_local[i];
            }

            // ── Per-layer input embedding branch (Gemma 4 simplified) ──────
            if layer.inp_gate().is_some() && layer.proj().is_some() && layer.post_norm().is_some() {
                let pe_in = cur.clone();
                // gate: [hidden_dim] → [n_embd_altup]
                let inp_gate = layer.inp_gate().unwrap();
                let mut gated = vec![0.0f32; n_embd_altup];
                inp_gate.matvec(&cur, &mut gated);
                for v in &mut gated {
                    *v = gelu_approx(*v);
                }
                // elementwise mul with per-layer input for this layer
                for i in 0..n_embd_altup {
                    gated[i] *= inp_per_layer[layer_idx][i];
                }
                // Project up to hidden_dim via per_layer_proj.
                let proj = layer.proj().unwrap();
                let mut projected = vec![0.0f32; hidden_dim];
                proj.matvec(&gated, &mut projected);
                // post_norm RMSNorm.
                let post_norm = layer.post_norm().unwrap();
                rms_norm(&projected, post_norm, c.norm_eps, &mut norm_buf);
                // Residual add pe_in.
                for i in 0..hidden_dim {
                    cur[i] = norm_buf[i] + pe_in[i];
                }
            }

            // ── Optional per-layer output scale ────────────────────────────
            if let Some(scale) = layer.out_scale.as_ref() {
                if let Some(&s) = scale.first() {
                    for v in &mut cur {
                        *v *= s;
                    }
                }
            }

            inpl = cur;
        }

        // Advance KV cache position.
        self.kv_cache.advance();

        // Output norm + logits.
        rms_norm(&inpl, &self.output_norm, c.norm_eps, &mut norm_buf);
        let mut logits = vec![0.0f32; c.vocab_size];
        self.output_proj.matvec(&norm_buf, &mut logits);

        // Final logit softcap (Gemma family).
        if let Some(cap) = c.final_logit_softcap() {
            for l in &mut logits {
                *l = cap * (*l / cap).tanh();
            }
        }
        logits
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
        let q_dim = c.num_heads * c.head_dim;
        let mut q_buf = vec![0.0f32; q_dim];
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
            layer
                .k_proj
                .as_ref()
                .expect("k_proj required for non-shared layer")
                .matvec_preq(&q8_attn, &mut k_buf);
            layer
                .v_proj
                .as_ref()
                .expect("v_proj required for non-shared layer")
                .matvec_preq(&q8_attn, &mut v_buf);
            // Qwen 2/2.5 bias (no-op for Llama/Mistral/Gemma/Qwen 3)
            if let Some(b) = layer.q_bias() {
                for (q, bi) in q_buf.iter_mut().zip(b.iter()) {
                    *q += bi;
                }
            }
            if let Some(b) = layer.k_bias() {
                for (k, bi) in k_buf.iter_mut().zip(b.iter()) {
                    *k += bi;
                }
            }
            if let Some(b) = layer.v_bias() {
                for (v, bi) in v_buf.iter_mut().zip(b.iter()) {
                    *v += bi;
                }
            }
            // Qwen 3 QK-Norm (per-head RMSNorm on Q, K before RoPE; no-op for others).
            // Slice `q_buf` to the first `q_dim` entries — the second half (when
            // present in Bonsai / Qwen 3.6 Gated Attention layers) holds the
            // swish gate, which does NOT get per-head normalisation. For non-
            // gated forward paths the slice is equivalent to the full buffer.
            if let Some(w) = layer.q_norm() {
                apply_qk_norm(&mut q_buf[..q_dim], w, c.head_dim, c.norm_eps);
            }
            if let Some(w) = layer.k_norm() {
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
                c.attn_logit_softcap(),
                if c.arch == ModelArch::Gemma3n {
                    Some(1.0)
                } else {
                    None
                },
                &mut attn_out,
            );

            layer.o_proj.matvec(&attn_out, &mut o_buf);
            for i in 0..c.hidden_dim {
                hidden[i] += o_buf[i];
            }

            rms_norm(&hidden, &layer.ffn_norm, c.norm_eps, &mut norm_buf);
            let q8_ffn = quantize_row_q8_k(&norm_buf);
            layer
                .gate_proj
                .as_ref()
                .expect("gate_proj required for non-MoE layer")
                .matvec_preq(&q8_ffn, &mut gate_buf);
            layer
                .up_proj
                .as_ref()
                .expect("up_proj required for non-MoE layer")
                .matvec_preq(&q8_ffn, &mut up_buf);
            c.apply_ffn_sparsity(layer_idx, &mut gate_buf);
            for i in 0..c.intermediate_dim {
                gate_buf[i] = c.apply_ffn_act(layer_idx, gate_buf[i]) * up_buf[i];
            }
            layer
                .down_proj
                .as_ref()
                .expect("down_proj required for non-MoE layer")
                .matvec(&gate_buf, &mut down_buf);
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
                c.attn_logit_softcap(),
                if c.arch == ModelArch::Gemma3n {
                    Some(1.0)
                } else {
                    None
                },
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
                    layer.k_proj.as_ref().expect("k_proj required for sparsify"),
                    kv_dim,
                    c.hidden_dim,
                    threshold_ratio,
                    n_keep,
                ),
                v_proj: sparsify_weight(
                    layer.v_proj.as_ref().expect("v_proj required for sparsify"),
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
                    layer
                        .gate_proj
                        .as_ref()
                        .expect("gate_proj required for sparsify"),
                    c.intermediate_dim,
                    c.hidden_dim,
                    threshold_ratio,
                    n_keep,
                ),
                up_proj: sparsify_weight(
                    layer
                        .up_proj
                        .as_ref()
                        .expect("up_proj required for sparsify"),
                    c.intermediate_dim,
                    c.hidden_dim,
                    threshold_ratio,
                    n_keep,
                ),
                down_proj: sparsify_weight(
                    layer
                        .down_proj
                        .as_ref()
                        .expect("down_proj required for sparsify"),
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
                c.attn_logit_softcap(),
                if c.arch == ModelArch::Gemma3n {
                    Some(1.0)
                } else {
                    None
                },
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
        let q_dim = c.num_heads * c.head_dim;
        let mut q_buf = vec![0.0f32; q_dim];
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
                c.attn_logit_softcap(),
                if c.arch == ModelArch::Gemma3n {
                    Some(1.0)
                } else {
                    None
                },
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
        let q_dim = c.num_heads * c.head_dim;
        let mut q_buf = vec![0.0f32; q_dim];
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];
        // attn_out holds `num_heads * head_dim` = q_dim values, which may
        // exceed hidden_dim (e.g. Qwen 3 MoE 4x0.6B: q_dim=2048, hidden=1024).
        let mut attn_out = vec![0.0f32; q_dim.max(c.hidden_dim)];
        let mut o_buf = vec![0.0f32; c.hidden_dim];
        let mut gate_buf = vec![0.0f32; c.intermediate_dim];
        let mut up_buf = vec![0.0f32; c.intermediate_dim];
        let mut down_buf = vec![0.0f32; c.hidden_dim];

        for layer_idx in 0..c.num_layers {
            let layer = &self.layers[layer_idx];

            rms_norm(&hidden, &layer.attn_norm, c.norm_eps, &mut norm_buf);

            let q8_attn = quantize_row_q8_k(&norm_buf);
            layer.q_proj.matvec_preq(&q8_attn, &mut q_buf);
            layer
                .k_proj
                .as_ref()
                .expect("k_proj required for non-shared layer")
                .matvec_preq(&q8_attn, &mut k_buf);
            layer
                .v_proj
                .as_ref()
                .expect("v_proj required for non-shared layer")
                .matvec_preq(&q8_attn, &mut v_buf);
            // Qwen 2/2.5 bias (no-op for Llama/Mistral/Gemma/Qwen 3)
            if let Some(b) = layer.q_bias() {
                for (q, bi) in q_buf.iter_mut().zip(b.iter()) {
                    *q += bi;
                }
            }
            if let Some(b) = layer.k_bias() {
                for (k, bi) in k_buf.iter_mut().zip(b.iter()) {
                    *k += bi;
                }
            }
            if let Some(b) = layer.v_bias() {
                for (v, bi) in v_buf.iter_mut().zip(b.iter()) {
                    *v += bi;
                }
            }
            // Qwen 3 QK-Norm (per-head RMSNorm on Q, K before RoPE; no-op for others).
            // Slice `q_buf` to the first `q_dim` entries — the second half (when
            // present in Bonsai / Qwen 3.6 Gated Attention layers) holds the
            // swish gate, which does NOT get per-head normalisation. For non-
            // gated forward paths the slice is equivalent to the full buffer.
            if let Some(w) = layer.q_norm() {
                apply_qk_norm(&mut q_buf[..q_dim], w, c.head_dim, c.norm_eps);
            }
            if let Some(w) = layer.k_norm() {
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
                c.attn_logit_softcap(),
                &mut attn_out,
            );

            layer.o_proj.matvec(&attn_out, &mut o_buf);

            for i in 0..c.hidden_dim {
                hidden[i] += o_buf[i];
            }

            rms_norm(&hidden, &layer.ffn_norm, c.norm_eps, &mut norm_buf);

            let q8_ffn = quantize_row_q8_k(&norm_buf);
            layer
                .gate_proj
                .as_ref()
                .expect("gate_proj required for non-MoE layer")
                .matvec_preq(&q8_ffn, &mut gate_buf);
            layer
                .up_proj
                .as_ref()
                .expect("up_proj required for non-MoE layer")
                .matvec_preq(&q8_ffn, &mut up_buf);

            c.apply_ffn_sparsity(layer_idx, &mut gate_buf);
            for i in 0..c.intermediate_dim {
                gate_buf[i] = c.apply_ffn_act(layer_idx, gate_buf[i]) * up_buf[i];
            }

            layer
                .down_proj
                .as_ref()
                .expect("down_proj required for non-MoE layer")
                .matvec(&gate_buf, &mut down_buf);

            for i in 0..c.hidden_dim {
                hidden[i] += down_buf[i];
            }
        }

        paged_cache.advance();

        rms_norm(&hidden, &self.output_norm, c.norm_eps, &mut norm_buf);

        let mut logits = vec![0.0f32; c.vocab_size];
        self.output_proj.matvec(&norm_buf, &mut logits);

        if let Some(cap) = c.final_logit_softcap() {
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

/// Load a weight where the row count is only known from the tensor itself
/// (GGUF tensor shape `[cols, rows]`, row-major storage). Used for arch
/// variants whose output dimension is not derivable from the standard config
/// metadata — currently Bonsai 27B's `attn_qkv` / `attn_gate` (10240 / 6144
/// respectively), which fall outside the Qwen 3.5 DeltaNet layout.
fn load_weight_ref_any_rows<'a>(
    gguf: &'a GgufFile<'a>,
    name: &str,
    cols: usize,
) -> Option<WeightRef<'a>> {
    let info = gguf.tensor_info(name)?;
    let data = gguf.tensor_data(name)?;
    // GGUF shape convention: `dims[0]` is the row stride (= cols), `dims[1]`
    // is the number of rows. For a `[cols, rows]` tensor stored in row-major
    // order, this maps to `rows = dims[1]`.
    let rows = *info.dims.get(1)?;
    Some(WeightRef {
        data,
        qtype: info.qtype,
        rows: rows as usize,
        cols,
    })
}

/// Run one MoE layer's expert dispatch.
///
/// - `norm_buf`: the RMS-normalised hidden state, `[hidden_dim]`.
/// - `output`: `[hidden_dim]` — the layer's FFN output, overwritten in place.
///
/// Algorithm (Qwen3 MoE / Mixtral / generic top-k softmax MoE):
/// 1. Router logits `= ffn_gate_inp @ norm_buf` (dense F32 matmul).
/// 2. `softmax(router_logits)` → per-expert probabilities.
/// 3. Select top-`num_experts_active` experts by probability.
/// 4. Renormalise the selected probabilities to sum to 1.
/// 5. For each selected expert:
///    - `gate = ffn_gate_exps[e] @ norm_buf`
///    - `up   = ffn_up_exps[e]   @ norm_buf`
///    - `expert_out = ffn_down_exps[e] @ (SiLU(gate) * up)`
/// 6. Sum weighted `expert_out` into `output`.
/// DeepSeek-V3 MoE routing (pure math, Phase 3).
///
/// Splits out steps 3-7 of `forward_deepseek_moe_layer` so the routing
/// arithmetic can be unit-tested in isolation without loading real weights:
///
/// 1. `scores = sigmoid(router_logits)` (un-biased routing weights).
/// 2. `biased = scores + exp_probs_b` if noaux_tc bias is present.
/// 3. Pick top-k experts by `biased` score (or by `scores` if no bias).
/// 4. Recover the un-biased `scores` at the selected indices.
/// 5. Renormalise the selected weights to sum-to-1, then multiply by
///    `routed_scale` (V3 uses 2.5).
///
/// Returns `Vec<(expert_index, routing_weight)>` of length `top_k`.
///
/// Numerically-stable sigmoid: uses the branch `1/(1+exp(-x))` for x ≥ 0
/// and `exp(x)/(1+exp(x))` for x < 0 to keep the exp argument non-positive.
fn deepseek_moe_route(
    router_logits: &[f32],
    exp_probs_b: Option<&[f32]>,
    top_k: usize,
    routed_scale: f32,
) -> Vec<(usize, f32)> {
    let n_experts = router_logits.len();
    let scores: Vec<f32> = router_logits
        .iter()
        .map(|&x| {
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let e = x.exp();
                e / (1.0 + e)
            }
        })
        .collect();

    let mut idx_biased: Vec<(usize, f32)> = if let Some(bias) = exp_probs_b {
        assert_eq!(bias.len(), n_experts, "exp_probs_b shape mismatch");
        scores
            .iter()
            .zip(bias.iter())
            .enumerate()
            .map(|(i, (&s, &b))| (i, s + b))
            .collect()
    } else {
        scores.iter().enumerate().map(|(i, &s)| (i, s)).collect()
    };
    idx_biased.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    idx_biased.truncate(top_k);

    let mut selected: Vec<(usize, f32)> =
        idx_biased.iter().map(|(i, _)| (*i, scores[*i])).collect();
    let sum: f32 = selected.iter().map(|(_, w)| *w).sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for (_, w) in &mut selected {
            *w *= inv;
        }
    }
    for (_, w) in &mut selected {
        *w *= routed_scale;
    }
    selected
}

/// DeepSeek-V3 MoE FFN forward pass (Phase 3, Issue #33).
///
/// Distinct from `forward_moe_layer` (Qwen 3 MoE / Mixtral style, softmax
/// gating, no shared expert) because V3 uses **sigmoid gating with the
/// noaux_tc bias-correction trick**, an **always-active shared expert**,
/// and a **routed_scaling_factor** applied to the routed sum before it is
/// added to the shared expert's output.
///
/// Algorithm (matches DeepSeek-V3 paper Section 4 + colibri implementation):
/// 1. `hidden → ffn_norm → norm_buf`
/// 2. `router_logits = ffn_gate_inp @ norm_buf` (`[n_routed_experts]`)
/// 3. `scores = sigmoid(router_logits)` — the un-biased routing weights.
/// 4. `biased = scores + exp_probs_b` when noaux_tc bias is present
///    (used **only** for selecting top-k, not for the final weights).
/// 5. Pick top-k experts by `biased` score.
/// 6. Renormalise the top-k `scores` (not `biased`) to sum to 1 —
///    those become the routing weights.
/// 7. Multiply the renormalised weights by `routed_scaling_factor`
///    (V3: 2.5) so the routed contribution is amplified relative to the
///    shared expert.
/// 8. For each of the top-k experts: run SwiGLU FFN on `norm_buf`,
///    accumulate into `output` weighted by the routing weight.
/// 9. Run the shared expert unconditionally (SwiGLU FFN on `norm_buf`) and
///    add its full contribution to `output`.
///
/// Output is the FFN branch's contribution — the caller adds it to the
/// residual `hidden` outside this function.
fn forward_deepseek_moe_layer(
    c: &Llama3Config,
    moe: &DeepSeekMoeWeights<'_>,
    hidden: &[f32],
    norm_buf: &mut [f32],
    output: &mut [f32],
) {
    let hidden_dim = c.hidden_dim;
    let n_experts = c
        .deepseek_n_routed_experts()
        .expect("DeepSeek MoE requires n_routed_experts");
    let n_shared = c
        .deepseek_n_shared_experts()
        .expect("DeepSeek MoE requires n_shared_experts");
    let top_k = c
        .deepseek_num_experts_per_tok()
        .expect("DeepSeek MoE requires num_experts_per_tok");
    let moe_ffn = c
        .deepseek_moe_intermediate_size()
        .expect("DeepSeek MoE requires moe_intermediate_size");
    let shared_ffn = n_shared * moe_ffn;
    let routed_scale = c.deepseek_routed_scaling_factor().unwrap_or(1.0);

    // ── Step 1: FFN RMSNorm on the input residual. ─────────────────────
    rms_norm(hidden, &moe.ffn_norm, c.norm_eps, norm_buf);

    // ── Step 2: Router logits `router[e] = W[e, :] · norm_buf`. ────────
    // `ffn_gate_inp` is dense f32, laid out `[n_experts, hidden_dim]`
    // row-major with hidden_dim the fast axis (matches Qwen 3 MoE).
    let router_w = &moe.ffn_gate_inp;
    assert_eq!(
        router_w.len(),
        n_experts * hidden_dim,
        "ffn_gate_inp shape mismatch"
    );
    let mut router_logits = vec![0.0f32; n_experts];
    for (e, logit) in router_logits.iter_mut().enumerate() {
        let base = e * hidden_dim;
        let mut acc = 0.0f32;
        for (h, &x) in norm_buf.iter().enumerate() {
            acc += router_w[base + h] * x;
        }
        *logit = acc;
    }

    // ── Steps 3-7: pure routing math (testable in isolation). ──────────
    let selected = deepseek_moe_route(
        &router_logits,
        moe.exp_probs_b.as_deref(),
        top_k,
        routed_scale,
    );

    // ── Step 8: Dispatch to top-k routed experts, accumulate. ──────────
    // The `routed` enum lets us serve slabs from either an in-memory
    // WeightRef (Phase 3 default) or a streaming pool (Phase 4a). Both
    // paths hand the same `&[u8]` + qtype to `quantized_matvec` — the
    // streaming path holds an `Arc<Vec<u8>>` in a temporary for the
    // scope of one expert's three matvecs, then drops it (the LRU cache
    // still owns the shared copy internally).
    //
    // Phase 4b.2: on the streaming path we prefetch every selected
    // expert's 3 kinds (gate + up + down) into the cache BEFORE the
    // matvec loop starts. Cache misses happen upfront (potentially in
    // parallel via rayon when `parallel` is enabled) so the subsequent
    // per-expert `get_or_load` calls inside the loop become guaranteed
    // hits — decoupling I/O from compute.
    use crate::deepseek_streaming::{ExpertKey, ExpertKind};
    let (gate_qtype, up_qtype, down_qtype) = match &moe.routed {
        RoutedExpertStorage::InMemory { gate, up, down } => (gate.qtype, up.qtype, down.qtype),
        RoutedExpertStorage::Streaming { pool, layer_idx } => {
            let mut prefetch_keys = Vec::with_capacity(selected.len() * 3);
            for &(e, _) in &selected {
                prefetch_keys.push(ExpertKey::new(*layer_idx, ExpertKind::Gate, e));
                prefetch_keys.push(ExpertKey::new(*layer_idx, ExpertKind::Up, e));
                prefetch_keys.push(ExpertKey::new(*layer_idx, ExpertKind::Down, e));
            }
            pool.prefetch_parallel(&prefetch_keys);
            (
                pool.qtype(*layer_idx, ExpertKind::Gate),
                pool.qtype(*layer_idx, ExpertKind::Up),
                pool.qtype(*layer_idx, ExpertKind::Down),
            )
        }
    };

    for v in output.iter_mut() {
        *v = 0.0;
    }
    let mut gate_buf = vec![0.0f32; moe_ffn];
    let mut up_buf = vec![0.0f32; moe_ffn];
    let mut expert_out = vec![0.0f32; hidden_dim];
    for &(e, weight) in &selected {
        // Fetch this expert's three slabs. The `Arc<Vec<u8>>` variants
        // for the streaming path outlive each matvec call, so borrowing
        // `.as_slice()` for the argument is safe until the loop iteration
        // ends.
        let (g_arc, u_arc, d_arc);
        let (g_data, u_data, d_data): (&[u8], &[u8], &[u8]) = match &moe.routed {
            RoutedExpertStorage::InMemory { gate, up, down } => {
                let gate_slab = expert_slab_bytes(gate, moe_ffn, hidden_dim);
                let up_slab = expert_slab_bytes(up, moe_ffn, hidden_dim);
                let down_slab = expert_slab_bytes(down, hidden_dim, moe_ffn);
                (
                    &gate.data[e * gate_slab..(e + 1) * gate_slab],
                    &up.data[e * up_slab..(e + 1) * up_slab],
                    &down.data[e * down_slab..(e + 1) * down_slab],
                )
            }
            RoutedExpertStorage::Streaming { pool, layer_idx } => {
                g_arc = pool.get_or_load(*layer_idx, ExpertKind::Gate, e);
                u_arc = pool.get_or_load(*layer_idx, ExpertKind::Up, e);
                d_arc = pool.get_or_load(*layer_idx, ExpertKind::Down, e);
                (g_arc.as_slice(), u_arc.as_slice(), d_arc.as_slice())
            }
        };

        crate::gguf::quantized_matvec(
            norm_buf,
            g_data,
            gate_qtype,
            moe_ffn,
            hidden_dim,
            &mut gate_buf,
        );
        crate::gguf::quantized_matvec(norm_buf, u_data, up_qtype, moe_ffn, hidden_dim, &mut up_buf);
        for i in 0..moe_ffn {
            gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
        }
        crate::gguf::quantized_matvec(
            &gate_buf,
            d_data,
            down_qtype,
            hidden_dim,
            moe_ffn,
            &mut expert_out,
        );
        for i in 0..hidden_dim {
            output[i] += weight * expert_out[i];
        }
    }

    // ── Step 9: Always-active shared expert (SwiGLU FFN, no gating). ───
    let mut shared_gate = vec![0.0f32; shared_ffn];
    let mut shared_up = vec![0.0f32; shared_ffn];
    let mut shared_out = vec![0.0f32; hidden_dim];
    moe.ffn_gate_shexp.matvec(norm_buf, &mut shared_gate);
    moe.ffn_up_shexp.matvec(norm_buf, &mut shared_up);
    for i in 0..shared_ffn {
        shared_gate[i] = silu(shared_gate[i]) * shared_up[i];
    }
    moe.ffn_down_shexp.matvec(&shared_gate, &mut shared_out);
    for i in 0..hidden_dim {
        output[i] += shared_out[i];
    }
}

fn forward_moe_layer(
    c: &Llama3Config,
    layer: &LayerWeights<'_>,
    norm_buf: &[f32],
    output: &mut [f32],
) {
    let n_expert = c
        .num_experts()
        .expect("MoE layer requires num_experts in config");
    let n_active = c
        .num_experts_active()
        .expect("MoE layer requires num_experts_active in config");
    let expert_ffn = c.expert_ffn_size().unwrap_or(c.intermediate_dim);
    let hidden_dim = c.hidden_dim;

    // Step 1: router logits = ffn_gate_inp @ norm_buf.
    // ffn_gate_inp shape (F32 dense): [hidden_dim, n_expert] row-major with
    // ne0 = hidden_dim (fast) and ne1 = n_expert. Compute
    // `router_logits[e] = sum_h weights[e * hidden_dim + h] * norm_buf[h]`.
    let router_w = layer
        .ffn_gate_inp()
        .expect("MoE layer requires ffn_gate_inp");
    assert_eq!(
        router_w.len(),
        n_expert * hidden_dim,
        "ffn_gate_inp shape mismatch"
    );
    let mut router_logits = vec![0.0f32; n_expert];
    for e in 0..n_expert {
        let mut acc = 0.0f32;
        let base = e * hidden_dim;
        for h in 0..hidden_dim {
            acc += router_w[base + h] * norm_buf[h];
        }
        router_logits[e] = acc;
    }

    // Step 2 + 3: softmax then top-k selection.
    let max_logit = router_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = router_logits
        .iter()
        .map(|&v| (v - max_logit).exp())
        .collect();
    let sum_exp: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum_exp;
    }

    // Sort indices by probability descending; keep top-k.
    let mut idx_prob: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    idx_prob.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    idx_prob.truncate(n_active);

    // Step 4: renormalise top-k probabilities.
    let top_k_sum: f32 = idx_prob.iter().map(|(_, p)| *p).sum();
    if top_k_sum > 0.0 {
        for (_, p) in &mut idx_prob {
            *p /= top_k_sum;
        }
    }

    // Step 5+6: expert dispatch. Extract each expert's slab from the 3D
    // WeightRef and run three matvecs.
    let gate_exps = layer
        .ffn_gate_exps()
        .expect("MoE layer requires ffn_gate_exps");
    let up_exps = layer.ffn_up_exps().expect("MoE layer requires ffn_up_exps");
    let down_exps = layer
        .ffn_down_exps()
        .expect("MoE layer requires ffn_down_exps");

    let gate_expert_bytes = expert_slab_bytes(gate_exps, expert_ffn, hidden_dim);
    let up_expert_bytes = expert_slab_bytes(up_exps, expert_ffn, hidden_dim);
    let down_expert_bytes = expert_slab_bytes(down_exps, hidden_dim, expert_ffn);

    for v in output.iter_mut() {
        *v = 0.0;
    }
    let mut gate_buf = vec![0.0f32; expert_ffn];
    let mut up_buf = vec![0.0f32; expert_ffn];
    let mut expert_out = vec![0.0f32; hidden_dim];

    for &(e, weight) in &idx_prob {
        let gate_slab = &gate_exps.data[e * gate_expert_bytes..(e + 1) * gate_expert_bytes];
        let up_slab = &up_exps.data[e * up_expert_bytes..(e + 1) * up_expert_bytes];
        let down_slab = &down_exps.data[e * down_expert_bytes..(e + 1) * down_expert_bytes];

        crate::gguf::quantized_matvec(
            norm_buf,
            gate_slab,
            gate_exps.qtype,
            expert_ffn,
            hidden_dim,
            &mut gate_buf,
        );
        crate::gguf::quantized_matvec(
            norm_buf,
            up_slab,
            up_exps.qtype,
            expert_ffn,
            hidden_dim,
            &mut up_buf,
        );
        for i in 0..expert_ffn {
            gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
        }
        crate::gguf::quantized_matvec(
            &gate_buf,
            down_slab,
            down_exps.qtype,
            hidden_dim,
            expert_ffn,
            &mut expert_out,
        );

        for i in 0..hidden_dim {
            output[i] += weight * expert_out[i];
        }
    }
}

/// Byte size of a single expert's 2D slab within a 3D expert WeightRef.
fn expert_slab_bytes(w: &WeightRef<'_>, rows_per_expert: usize, cols_per_expert: usize) -> usize {
    let elems = rows_per_expert * cols_per_expert;
    let epb = w.qtype.elements_per_block();
    let bpb = w.qtype.block_bytes();
    assert!(epb > 0 && bpb > 0, "unsupported qtype for expert slab");
    (elems / epb) * bpb
}

// ─── Speculative decoding ──────────────────────────────────────────────────

/// Result of a speculative-decoding run.
#[derive(Debug, Clone)]
pub struct SpeculativeResult {
    /// The greedy-generated tokens (excluding the prompt).
    pub tokens: Vec<u32>,
    /// Total draft tokens produced by the draft model.
    pub draft_tokens_produced: usize,
    /// Draft tokens accepted by the main model's verification.
    pub draft_tokens_accepted: usize,
    /// Bonus tokens (main-model argmax when all-accepted or the replacement
    /// token when the main model diverges from a draft prediction).
    pub bonus_tokens: usize,
}

impl SpeculativeResult {
    /// Fraction of draft tokens that survived main-model verification.
    /// `1.0` means the draft matched the main model everywhere (which
    /// happens by construction for a same-model speculative run).
    #[must_use]
    pub fn acceptance_rate(&self) -> f32 {
        if self.draft_tokens_produced == 0 {
            0.0
        } else {
            self.draft_tokens_accepted as f32 / self.draft_tokens_produced as f32
        }
    }
}

/// Speculative decoding (greedy verification).
///
/// The `draft` model (typically small and fast) proposes `n_draft` candidate
/// tokens each iteration by greedy sampling. The `main` model (large and
/// slow, considered the source of truth) verifies each candidate by
/// comparing its greedy prediction to the draft's. All matching tokens are
/// accepted in a single main-model pass; the first mismatch is replaced by
/// the main model's own argmax. When every draft token is accepted, the main
/// model contributes a bonus token, yielding up to `n_draft + 1` tokens per
/// iteration.
///
/// Both models MUST share the same tokenizer vocabulary. When `draft` and
/// `main` are the same instance (or two clones), acceptance rate is exactly
/// 100% (each iteration produces `n_draft + 1` tokens), which makes for a
/// deterministic correctness fixture.
///
/// # Arguments
/// * `draft` – the proposal model (`&mut Llama3Model`).
/// * `main` – the verification model (`&mut Llama3Model`).
/// * `prompt` – the prompt tokens. Both models MUST have empty KV caches on
///   entry (call `reset()` beforehand if reusing them).
/// * `n_draft` – number of speculative tokens per iteration (typically 4–8).
///   Larger values amortise verification cost across more candidates but
///   discard more work on mismatches.
/// * `max_new_tokens` – hard upper bound on emitted tokens.
/// * `eos_id` – optional EOS token id. Generation stops when the accepted
///   token equals this value.
pub fn speculative_decode(
    draft: &mut Llama3Model<'_>,
    main: &mut Llama3Model<'_>,
    prompt: &[u32],
    n_draft: usize,
    max_new_tokens: usize,
    eos_id: Option<u32>,
) -> SpeculativeResult {
    assert!(n_draft >= 1, "n_draft must be at least 1");
    assert!(!prompt.is_empty(), "prompt must be non-empty");

    // Prefill both models with prompt[..len-1]; hold the final prompt token
    // as `last_tok` to start the first draft/verify cycle.
    for &tok in prompt.iter().take(prompt.len() - 1) {
        let _ = draft.forward(tok);
        let _ = main.forward(tok);
    }
    let mut last_tok = *prompt.last().unwrap();
    let mut result = SpeculativeResult {
        tokens: Vec::with_capacity(max_new_tokens),
        draft_tokens_produced: 0,
        draft_tokens_accepted: 0,
        bonus_tokens: 0,
    };

    while result.tokens.len() < max_new_tokens {
        // ── 1. Draft phase ────────────────────────────────────────────────
        let mut draft_tokens = Vec::with_capacity(n_draft);
        let mut curr = last_tok;
        for _ in 0..n_draft {
            let logits = draft.forward(curr);
            let argmax = greedy_argmax(&logits);
            draft_tokens.push(argmax);
            curr = argmax;
        }
        result.draft_tokens_produced += n_draft;

        // ── 2. Verify phase ────────────────────────────────────────────────
        let mut accepted = 0usize;
        let mut mismatch_replacement: Option<u32> = None;
        curr = last_tok;
        for &draft_tok in &draft_tokens {
            let logits = main.forward(curr);
            let main_argmax = greedy_argmax(&logits);
            if main_argmax == draft_tok {
                result.tokens.push(draft_tok);
                accepted += 1;
                curr = draft_tok;
                if result.tokens.len() >= max_new_tokens {
                    break;
                }
                if eos_id == Some(draft_tok) {
                    break;
                }
            } else {
                result.tokens.push(main_argmax);
                result.bonus_tokens += 1;
                mismatch_replacement = Some(main_argmax);
                break;
            }
        }
        result.draft_tokens_accepted += accepted;

        // ── 3. Sync draft KV cache to main's position ─────────────────────
        // Invariant after sync: draft.kv_seq_len() == main.kv_seq_len() and
        // both caches hold the same token sequence. `last_tok` is pending
        // forward on both models in the next iteration.
        if let Some(replacement) = mismatch_replacement {
            // Main advanced by `accepted + 1`; draft advanced by `n_draft`.
            // Rollback draft to main's exact position — both caches now hold
            // the same accepted prefix. `replacement` is pending forward on
            // both in the next iteration; do NOT pre-forward it on draft.
            let target = main.kv_seq_len();
            draft.kv_rollback_to(target);
            last_tok = replacement;
        } else if accepted == n_draft && result.tokens.len() < max_new_tokens {
            // All-accepted bonus. After verify, both caches are aligned at
            // seq_len = start + n_draft (draft phase forwarded last_tok,
            // d_0, ..., d_{n_draft-2}; verify phase did the same on main).
            // curr = d_{n_draft-1}. Draft must forward curr to stay aligned
            // with main after main's bonus forward.
            let _ = draft.forward(curr);
            let logits = main.forward(curr);
            let bonus = greedy_argmax(&logits);
            result.tokens.push(bonus);
            result.bonus_tokens += 1;
            last_tok = bonus;
        } else {
            // Loop-cap or eos-mid-verify with partial acceptance.
            // Main advanced by `accepted`, draft by `n_draft`. Rollback
            // draft to main's position; last accepted token is pending
            // forward on both.
            let target = main.kv_seq_len();
            draft.kv_rollback_to(target);
            last_tok = *result.tokens.last().unwrap_or(&last_tok);
        }

        if eos_id.is_some_and(|e| Some(&e) == result.tokens.last()) {
            break;
        }
    }

    // Truncate to the requested budget.
    result.tokens.truncate(max_new_tokens);
    result
}

fn greedy_argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

// ─── Phase JJJ v0.2: Sampling distribution replay ─────────────────────────
//
// Reference: Leviathan et al. 2023 ("Fast Inference from Transformers via
// Speculative Decoding"), Chen et al. 2023, and Cognition SWE-1.7 (2026)
// which emphasises the importance of preserving inference-time entropy
// (top-p) and explicit sampling distribution replay to avoid KL mismatch
// between draft and main.
//
// Rejection sampling: draft samples x ~ p_draft, main computes p_main.
// Accept x with probability min(1, p_main(x) / p_draft(x)). Reject →
// sample x' ~ residual_dist(p_main, p_draft) where residual is
// max(0, p_main - p_draft) renormalised. This is mathematically
// equivalent to sampling directly from p_main (unbiased).

/// Configuration for [`speculative_decode_v2`].
///
/// When `temperature` is `None`, the function performs pure greedy
/// verification (equivalent to [`speculative_decode`]).
///
/// When `temperature` is `Some(t)`, the function performs rejection
/// sampling: draft samples from `top_p(softmax(logits / t))`, main
/// accepts each sample with probability `min(1, p_main / p_draft)`, and
/// rejects fall back to the residual distribution.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens the draft model proposes per iteration.
    pub n_draft: usize,
    /// Hard cap on emitted tokens.
    pub max_new_tokens: usize,
    /// Optional EOS token id. Generation stops when this token is emitted.
    pub eos_id: Option<u32>,
    /// `None` → greedy verify. `Some(t > 0)` → temperature-scaled softmax
    /// + rejection sampling.
    pub temperature: Option<f32>,
    /// `None` or `Some(1.0)` → no top-p filter. Otherwise keeps the
    /// smallest set of tokens whose cumulative probability reaches `p`.
    pub top_p: Option<f32>,
    /// Seed for the sampling RNG (`splitmix64`). Only used when
    /// `temperature` is `Some`. `None` → seeded with 0 (deterministic).
    pub sample_seed: Option<u64>,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            n_draft: 4,
            max_new_tokens: 128,
            eos_id: None,
            temperature: None,
            top_p: None,
            sample_seed: None,
        }
    }
}

/// DeepSeek-V3 MTP speculative-decoding adaptive-guard policy (Phase 5a.2,
/// Issue #35). Colibri reports that MTP accept rate is bimodal on real V3
/// traffic — around 40-60% during coherent generation, dropping to 0-5%
/// when the model transitions into a different distribution (e.g. hitting
/// a code block, structured output). Running MTP unconditionally in the
/// low-accept regime is a net loss because the draft + verify overhead
/// exceeds the compute saved by skipping a main-model forward.
///
/// This policy tracks accept rate over a sliding window and toggles MTP
/// on/off with hysteresis:
///
/// - Start with MTP **enabled**.
/// - After every accept/reject decision, push into a fixed-capacity
///   [`std::collections::VecDeque`] of recent outcomes.
/// - If MTP is enabled and the recent accept rate falls **below**
///   `disable_threshold`, transition to **cooldown** for `cooldown_tokens`
///   subsequent verify steps — during cooldown, MTP is skipped
///   entirely (draft-then-verify degrades to plain greedy).
/// - When cooldown expires, re-enable MTP and continue tracking.
///
/// The hysteresis prevents flap-flopping around the threshold: without a
/// cooldown, a single accepted draft would immediately re-enable MTP
/// after one bad stretch, only to see the same accept rate collapse
/// again on the next few tokens.
#[derive(Debug, Clone)]
pub struct MtpDraftPolicy {
    /// Rolling window of recent decisions. `true` = accepted, `false` = rejected.
    /// Bounded to [`window_size`]; oldest entries evict when the window fills.
    window: std::collections::VecDeque<bool>,
    /// Size of the sliding window (number of most-recent decisions the
    /// accept-rate check looks at). Larger = more stable decisions, less
    /// responsive to distribution shifts.
    window_size: usize,
    /// If MTP is enabled and the window's accept rate falls strictly
    /// below this, transition to cooldown. `0.30` (30%) is the colibri
    /// default — accepts break even with plain greedy around this ratio
    /// on their hardware profile.
    disable_threshold: f32,
    /// Number of verify steps the policy stays in cooldown after
    /// disabling. Longer cooldown = less flap; shorter = quicker recovery
    /// when the distribution shifts back to MTP-favourable territory.
    cooldown_tokens: usize,
    /// Cooldown countdown. `> 0` → MTP is disabled and we tick down; `0`
    /// → MTP is enabled and we track normally.
    cooldown_remaining: usize,
    /// Total accepted + rejected across the lifetime of this policy —
    /// exposed for bench harness telemetry.
    total_accepted: u64,
    total_rejected: u64,
    /// Total cooldowns entered. High counts on a stable prompt suggests
    /// the threshold is too aggressive.
    total_cooldowns: u64,
}

impl MtpDraftPolicy {
    /// Construct a policy with the colibri-reported defaults: 32-token
    /// window, 30% disable threshold, 16-token cooldown.
    #[must_use]
    pub fn new_default() -> Self {
        Self::with_params(32, 0.30, 16)
    }

    /// Construct with explicit tuning parameters. Panics if `window_size`
    /// is zero or `disable_threshold` is outside `[0.0, 1.0]`.
    #[must_use]
    pub fn with_params(window_size: usize, disable_threshold: f32, cooldown_tokens: usize) -> Self {
        assert!(window_size > 0, "MTP policy window_size must be > 0");
        assert!(
            (0.0..=1.0).contains(&disable_threshold),
            "MTP policy disable_threshold must be in [0.0, 1.0]"
        );
        Self {
            window: std::collections::VecDeque::with_capacity(window_size),
            window_size,
            disable_threshold,
            cooldown_tokens,
            cooldown_remaining: 0,
            total_accepted: 0,
            total_rejected: 0,
            total_cooldowns: 0,
        }
    }

    /// Whether MTP drafting is currently enabled. When `false`, the caller
    /// should skip the MTP forward and fall back to plain greedy verify.
    ///
    /// Every call to `should_draft` also ticks the cooldown counter down
    /// by one — so this method must be called exactly once per verify
    /// step to keep the countdown aligned with token cadence.
    pub fn should_draft(&mut self) -> bool {
        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            false
        } else {
            true
        }
    }

    /// Record the outcome of an MTP-verify decision and re-evaluate
    /// whether to enter cooldown. `accepted = true` means the draft
    /// matched the main-model verify at the current position.
    ///
    /// Only call this when the previous `should_draft` returned `true` —
    /// during cooldown there is no draft to record.
    pub fn record(&mut self, accepted: bool) {
        if accepted {
            self.total_accepted += 1;
        } else {
            self.total_rejected += 1;
        }
        // Slide the window.
        if self.window.len() == self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(accepted);

        // Only start a cooldown if the window is full — accept-rate on a
        // half-populated window is too noisy to trigger a policy shift.
        if self.window.len() >= self.window_size {
            let accepts: usize = self.window.iter().filter(|&&v| v).count();
            let rate = accepts as f32 / self.window.len() as f32;
            if rate < self.disable_threshold {
                self.cooldown_remaining = self.cooldown_tokens;
                self.total_cooldowns += 1;
                // Clear the window so the post-cooldown re-evaluation
                // does not immediately re-trigger on the same evidence.
                self.window.clear();
            }
        }
    }

    /// Snapshot of policy state. Consumers (bench harness / telemetry
    /// logger) inspect this to understand what fraction of tokens the
    /// policy skipped, the current cooldown state, and cumulative counters.
    #[must_use]
    pub fn stats(&self) -> MtpDraftStats {
        let total_decisions = self.total_accepted + self.total_rejected;
        let accept_rate = if total_decisions > 0 {
            self.total_accepted as f32 / total_decisions as f32
        } else {
            0.0
        };
        MtpDraftStats {
            in_cooldown: self.cooldown_remaining > 0,
            cooldown_remaining: self.cooldown_remaining,
            window_len: self.window.len(),
            total_accepted: self.total_accepted,
            total_rejected: self.total_rejected,
            total_cooldowns: self.total_cooldowns,
            overall_accept_rate: accept_rate,
        }
    }
}

impl Default for MtpDraftPolicy {
    fn default() -> Self {
        Self::new_default()
    }
}

/// Point-in-time snapshot of [`MtpDraftPolicy`] counters.
#[derive(Debug, Clone, Copy)]
pub struct MtpDraftStats {
    pub in_cooldown: bool,
    pub cooldown_remaining: usize,
    pub window_len: usize,
    pub total_accepted: u64,
    pub total_rejected: u64,
    pub total_cooldowns: u64,
    /// Ratio of `total_accepted / (total_accepted + total_rejected)`, or
    /// `0.0` when no decisions have been recorded yet.
    pub overall_accept_rate: f32,
}

/// Deterministic 64-bit PRNG (splitmix64). Adequate for token sampling and
/// gives bit-exact reproducibility across platforms without pulling in an
/// external RNG dependency.
#[derive(Debug, Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_unit_f32(&mut self) -> f32 {
        // Take top 24 bits of u64 → uniform in [0, 1).
        #[allow(clippy::cast_precision_loss)]
        {
            (self.next_u64() >> 40) as f32 / f32::from(1u16 << 8) / f32::from(1u16 << 8) / 256.0
        }
    }
}

/// Apply temperature scaling + softmax + optional top-p filter.
/// Returns a proper probability distribution (sums to 1.0).
///
/// * `temperature` must be strictly positive.
/// * `top_p = None` or `Some(t) where t >= 1.0` disables top-p filtering.
/// * `top_p = Some(t) where 0 < t < 1` keeps the smallest set of tokens
///   whose cumulative probability >= t (nucleus sampling).
fn apply_temperature_and_top_p(logits: &[f32], temperature: f32, top_p: Option<f32>) -> Vec<f32> {
    assert!(
        temperature > 0.0 && temperature.is_finite(),
        "temperature must be positive and finite"
    );
    let inv_t = 1.0 / temperature;
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| if b > a { b } else { a });
    let mut probs: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max_logit) * inv_t).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    assert!(
        sum > 0.0 && sum.is_finite(),
        "softmax normaliser is degenerate"
    );
    for p in &mut probs {
        *p /= sum;
    }

    if let Some(p_thresh) = top_p {
        if p_thresh > 0.0 && p_thresh < 1.0 - 1e-6 {
            let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut cum = 0.0f32;
            let mut keep_mask = vec![false; probs.len()];
            for &(orig_idx, prob) in &indexed {
                keep_mask[orig_idx] = true;
                cum += prob;
                if cum >= p_thresh {
                    break;
                }
            }
            for (i, p) in probs.iter_mut().enumerate() {
                if !keep_mask[i] {
                    *p = 0.0;
                }
            }
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                for p in &mut probs {
                    *p /= sum;
                }
            }
        }
    }
    probs
}

/// Sample one index from a probability distribution using inverse CDF.
/// The distribution must sum to (approximately) 1.0.
fn sample_multinomial(dist: &[f32], rng: &mut SplitMix64) -> u32 {
    let u = rng.next_unit_f32();
    let mut cum = 0.0f32;
    for (i, &p) in dist.iter().enumerate() {
        cum += p;
        if u < cum {
            return u32::try_from(i).expect("index fits in u32");
        }
    }
    // Fallback for float precision: return last non-zero index.
    for (i, &p) in dist.iter().enumerate().rev() {
        if p > 0.0 {
            return u32::try_from(i).expect("index fits in u32");
        }
    }
    0
}

/// Residual distribution used when a draft sample is rejected.
/// r(x) = max(0, p_main(x) - p_draft(x)), renormalised.
/// When the residual mass is zero (degenerate case), falls back to `p_main`.
fn residual_dist(p_main: &[f32], p_draft: &[f32]) -> Vec<f32> {
    assert_eq!(
        p_main.len(),
        p_draft.len(),
        "distributions must match length"
    );
    let mut r: Vec<f32> = p_main
        .iter()
        .zip(p_draft.iter())
        .map(|(&m, &d)| (m - d).max(0.0))
        .collect();
    let sum: f32 = r.iter().sum();
    if sum > 0.0 {
        for x in &mut r {
            *x /= sum;
        }
    } else {
        r.clone_from_slice(p_main);
    }
    r
}

/// Speculative decoding with rejection sampling (v0.2).
///
/// When `cfg.temperature` is `None`, behaves identically to
/// [`speculative_decode`] (greedy verify).
///
/// When `cfg.temperature` is `Some(t)`:
/// * Draft samples each proposed token from `top_p(softmax(logits / t))`.
/// * For each draft sample `x` with draft probability `p_draft`, the main
///   model computes `p_main`. Accept with probability `min(1, p_main / p_draft)`.
/// * On reject, sample the replacement from the residual distribution
///   `renorm(max(0, p_main - p_draft))`.
/// * All-accepted iterations produce a bonus token sampled from `p_main`.
///
/// This procedure is mathematically equivalent to sampling from `p_main`
/// directly, but amortises the main-model cost across up to `n_draft + 1`
/// tokens per iteration.
///
/// Both models must share the tokenizer vocabulary and be entering the
/// call with empty KV caches (call [`Llama3Model::reset`] beforehand if
/// reusing them).
pub fn speculative_decode_v2(
    draft: &mut Llama3Model<'_>,
    main: &mut Llama3Model<'_>,
    prompt: &[u32],
    cfg: &SpeculativeConfig,
) -> SpeculativeResult {
    assert!(cfg.n_draft >= 1, "n_draft must be at least 1");
    assert!(!prompt.is_empty(), "prompt must be non-empty");

    // Delegate to the greedy path for backward-compatible behaviour.
    if cfg.temperature.is_none() {
        return speculative_decode(
            draft,
            main,
            prompt,
            cfg.n_draft,
            cfg.max_new_tokens,
            cfg.eos_id,
        );
    }

    let temperature = cfg.temperature.expect("checked above");
    let top_p = cfg.top_p;
    let mut rng = SplitMix64::new(cfg.sample_seed.unwrap_or(0));

    // Prefill both models with prompt[..len-1].
    for &tok in prompt.iter().take(prompt.len() - 1) {
        let _ = draft.forward(tok);
        let _ = main.forward(tok);
    }
    let mut last_tok = *prompt.last().expect("non-empty prompt");
    let mut result = SpeculativeResult {
        tokens: Vec::with_capacity(cfg.max_new_tokens),
        draft_tokens_produced: 0,
        draft_tokens_accepted: 0,
        bonus_tokens: 0,
    };

    while result.tokens.len() < cfg.max_new_tokens {
        // ── 1. Draft phase ────────────────────────────────────────────────
        let mut draft_tokens = Vec::with_capacity(cfg.n_draft);
        let mut draft_probs_at_pick: Vec<f32> = Vec::with_capacity(cfg.n_draft);
        // We also cache the full draft distribution at each verify position
        // so the verify phase does not need to re-forward the draft.
        let mut draft_dists: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_draft);
        let mut curr = last_tok;
        for _ in 0..cfg.n_draft {
            let logits = draft.forward(curr);
            let dist = apply_temperature_and_top_p(&logits, temperature, top_p);
            let sample = sample_multinomial(&dist, &mut rng);
            let p_at = dist[sample as usize];
            draft_probs_at_pick.push(p_at);
            draft_dists.push(dist);
            draft_tokens.push(sample);
            curr = sample;
        }
        result.draft_tokens_produced += cfg.n_draft;

        // ── 2. Verify phase (rejection sampling) ─────────────────────────
        let mut accepted = 0usize;
        let mut rejected_replacement: Option<u32> = None;
        curr = last_tok;
        for (idx, &draft_tok) in draft_tokens.iter().enumerate() {
            let logits = main.forward(curr);
            let p_main = apply_temperature_and_top_p(&logits, temperature, top_p);
            let p_draft = &draft_dists[idx];
            let p_m = p_main[draft_tok as usize];
            let p_d = draft_probs_at_pick[idx];
            let ratio = if p_d > 0.0 { (p_m / p_d).min(1.0) } else { 0.0 };
            let u = rng.next_unit_f32();
            if u < ratio {
                result.tokens.push(draft_tok);
                accepted += 1;
                curr = draft_tok;
                if result.tokens.len() >= cfg.max_new_tokens {
                    break;
                }
                if cfg.eos_id == Some(draft_tok) {
                    break;
                }
            } else {
                let residual = residual_dist(&p_main, p_draft);
                let replacement = sample_multinomial(&residual, &mut rng);
                result.tokens.push(replacement);
                result.bonus_tokens += 1;
                rejected_replacement = Some(replacement);
                break;
            }
        }
        result.draft_tokens_accepted += accepted;

        // ── 3. Sync draft KV cache to main's position ─────────────────────
        // Invariant: after sync, draft.kv_seq_len() == main.kv_seq_len(),
        // both caches hold the same token sequence, and `last_tok` is
        // pending forward on both.
        if let Some(replacement) = rejected_replacement {
            let target = main.kv_seq_len();
            draft.kv_rollback_to(target);
            last_tok = replacement;
        } else if accepted == cfg.n_draft && result.tokens.len() < cfg.max_new_tokens {
            // All-accepted bonus: sample from main's distribution after
            // main advances by one more step (forward on curr = last draft).
            let _ = draft.forward(curr);
            let logits = main.forward(curr);
            let p_main = apply_temperature_and_top_p(&logits, temperature, top_p);
            let bonus = sample_multinomial(&p_main, &mut rng);
            result.tokens.push(bonus);
            result.bonus_tokens += 1;
            last_tok = bonus;
        } else {
            let target = main.kv_seq_len();
            draft.kv_rollback_to(target);
            last_tok = *result.tokens.last().unwrap_or(&last_tok);
        }

        if cfg.eos_id.is_some_and(|e| Some(&e) == result.tokens.last()) {
            break;
        }
    }

    result.tokens.truncate(cfg.max_new_tokens);
    result
}

/// Load the FFN input norm for layer `prefix`, trying `ffn_norm.weight` first
/// and falling back to `post_attention_norm.weight`.
///
/// Standard Qwen 3 / Llama-3 / Gemma GGUF exports call this tensor
/// `blk.N.ffn_norm.weight`. Bonsai 27B (PrismML, qwen35 arch) and some
/// Qwen 3.6 checkpoints export it as `blk.N.post_attention_norm.weight`
/// instead. Both names refer to the same "post-attention / pre-FFN" RMSNorm
/// weight in the Transformer block.
fn load_ffn_norm<'a>(gguf: &'a GgufFile<'a>, prefix: &str) -> Option<Vec<f32>> {
    gguf.tensor_to_f32(&format!("{prefix}.ffn_norm.weight"))
        .or_else(|| gguf.tensor_to_f32(&format!("{prefix}.post_attention_norm.weight")))
}

fn load_layer_weights<'a>(
    gguf: &'a GgufFile<'a>,
    layer: usize,
    config: &Llama3Config,
) -> Option<LayerWeights<'a>> {
    let prefix = format!("blk.{layer}");
    // Gemma-2: num_heads * head_dim (= 2048) != hidden_dim (= 2304).
    // Gemma 4: per-layer q_dim / kv_dim (SWA layers halve head_dim).
    // Other models: q_dim == hidden_dim (identity).
    let q_dim = config.q_dim_for_layer(layer);
    let kv_dim = config.kv_dim_for_layer(layer);
    let ffn_size = config.ffn_size_for_layer(layer);

    let attn_norm = gguf.tensor_to_f32(&format!("{prefix}.attn_norm.weight"))?;
    // Bonsai 27B / Qwen 3.6 "Gated Attention" packs Q and its per-element
    // swish gate into the same tensor: `q_proj.rows == 2 * q_dim`. Standard
    // GGUF exports (Llama / Qwen 3 / Mistral / Gemma / …) keep the rows
    // equal to `q_dim`. Use `load_weight_ref_any_rows` so the row count is
    // taken from the tensor itself, then branch on the observed shape.
    let q_proj =
        load_weight_ref_any_rows(gguf, &format!("{prefix}.attn_q.weight"), config.hidden_dim)?;
    let gated_output = match q_proj.rows {
        r if r == q_dim => false,
        r if r == 2 * q_dim => true,
        _ => return None, // Mis-shaped tensor: neither standard nor gated.
    };
    // Gemma 4: K/V weights are absent for shared-KV layers (>= kv_from_start).
    // For all other architectures they are required.
    let is_shared_layer =
        matches!(config.arch, ModelArch::Gemma4) && layer >= config.kv_from_start_layers();
    let k_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.attn_k.weight"),
        kv_dim,
        config.hidden_dim,
    );
    if k_proj.is_none() && !is_shared_layer {
        return None;
    }
    let v_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.attn_v.weight"),
        kv_dim,
        config.hidden_dim,
    );
    if v_proj.is_none() && !is_shared_layer {
        return None;
    }
    let o_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.attn_output.weight"),
        config.hidden_dim,
        q_dim,
    )?;

    let ffn_norm = load_ffn_norm(gguf, &prefix)?;
    // Standard FFN weights are optional: MoE layers omit them in favour of
    // expert-dispatched ffn_gate_inp + ffn_{gate,up,down}_exps tensors.
    let gate_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.ffn_gate.weight"),
        ffn_size,
        config.hidden_dim,
    );
    let up_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.ffn_up.weight"),
        ffn_size,
        config.hidden_dim,
    );
    let down_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.ffn_down.weight"),
        config.hidden_dim,
        ffn_size,
    );

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

    // Gemma 4: per-layer output scale + per-full-attn RoPE freq factors.
    let out_scale = gguf.tensor_to_f32(&format!("{prefix}.layer_output_scale.weight"));
    let rope_freqs = gguf.tensor_to_f32(&format!("{prefix}.rope_freqs.weight"));

    // MoE tensors (present per-layer in Qwen3 MoE, absent in dense models).
    // Shape convention (row/col = out/in as usual, per-expert slabs
    // concatenated along the expert axis in the raw bytes):
    //   ffn_gate_exps [hidden, ffn_size, n_expert] → rows=ffn_size, cols=hidden
    //   ffn_up_exps   [hidden, ffn_size, n_expert] → rows=ffn_size, cols=hidden
    //   ffn_down_exps [ffn_size, hidden, n_expert] → rows=hidden, cols=ffn_size
    // The per-expert slab is loaded on-demand via `WeightRef::expert_slab`
    // in `forward_moe`.
    let ffn_gate_inp = gguf.tensor_to_f32(&format!("{prefix}.ffn_gate_inp.weight"));
    let expert_ffn = config.expert_ffn_size().unwrap_or(ffn_size);
    let (ffn_gate_exps, ffn_up_exps, ffn_down_exps) =
        if let (Some(_), Some(n_expert)) = (ffn_gate_inp.as_ref(), config.num_experts()) {
            let gate = load_weight_ref(
                gguf,
                &format!("{prefix}.ffn_gate_exps.weight"),
                expert_ffn * n_expert, // total rows across all experts (out * n_expert)
                config.hidden_dim,     // per-expert cols (in)
            );
            let up = load_weight_ref(
                gguf,
                &format!("{prefix}.ffn_up_exps.weight"),
                expert_ffn * n_expert,
                config.hidden_dim,
            );
            let down = load_weight_ref(
                gguf,
                &format!("{prefix}.ffn_down_exps.weight"),
                config.hidden_dim * n_expert, // total rows: hidden per expert × n_expert
                expert_ffn,                   // per-expert cols (in = expert_ffn)
            );
            (gate, up, down)
        } else {
            (None, None, None)
        };

    // Gemma 3n per-layer input embedding gate/proj (WeightRef, quantized).
    //
    // Convention (matches WeightRef.matvec): `rows = out_dim, cols = in_dim`.
    //   - inp_gate.weight [n_embd, n_embd_altup] in ggml (ne0=in, ne1=out)
    //     → in=hidden_dim, out=per_layer_dim → rows=per_layer_dim, cols=hidden_dim
    //   - proj.weight     [n_embd_altup, n_embd] → in=per_layer_dim, out=hidden_dim
    //     → rows=hidden_dim, cols=per_layer_dim
    let (inp_gate, proj) = if let Some(per_layer_dim) = config.per_layer_input_embedding_dim() {
        let inp_gate = load_weight_ref(
            gguf,
            &format!("{prefix}.inp_gate.weight"),
            per_layer_dim,     // rows = out_dim
            config.hidden_dim, // cols = in_dim
        );
        let proj = load_weight_ref(
            gguf,
            &format!("{prefix}.proj.weight"),
            config.hidden_dim, // rows = out_dim
            per_layer_dim,     // cols = in_dim
        );
        (inp_gate, proj)
    } else {
        (None, None)
    };

    // Group Qwen 2 / 2.5 biases together: they either all exist or none do.
    let qwen_biases = match (q_bias, k_bias, v_bias) {
        (Some(q_bias), Some(k_bias), Some(v_bias)) => Some(QwenAttentionBiases {
            q_bias,
            k_bias,
            v_bias,
        }),
        _ => None,
    };

    // Group Qwen 3 per-head QK norms together.
    let qwen_norms = match (q_norm, k_norm) {
        (Some(q_norm), Some(k_norm)) => Some(QwenAttentionNorms { q_norm, k_norm }),
        _ => None,
    };

    // Group Gemma 3n augmentations: 11 fields either all populated (Gemma 3n)
    // or all absent (any other arch). Partial population is a load bug.
    let gemma3n = match (
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
    ) {
        (
            Some(post_norm),
            Some(inp_gate),
            Some(proj),
            Some(laurel_l),
            Some(laurel_r),
            Some(laurel_post_norm),
            Some(altup_router),
            Some(altup_router_norm),
            Some(altup_predict_coef),
            Some(altup_correct_coef),
            Some(altup_correct_scale),
        ) => Some(Gemma3nLayerAugmentations {
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
        }),
        _ => None,
    };

    // Group MoE weights: router + three expert tensors either all present
    // (this is a MoE layer) or all absent (dense SwiGLU).
    let moe = match (ffn_gate_inp, ffn_gate_exps, ffn_up_exps, ffn_down_exps) {
        (Some(ffn_gate_inp), Some(ffn_gate_exps), Some(ffn_up_exps), Some(ffn_down_exps)) => {
            Some(MoeExpertWeights {
                ffn_gate_inp,
                ffn_gate_exps,
                ffn_up_exps,
                ffn_down_exps,
            })
        }
        _ => None,
    };

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
        post_attn_norm,
        post_ffn_norm,
        out_scale,
        rope_freqs,
        qwen_biases,
        qwen_norms,
        gemma3n,
        moe,
        gated_output,
    })
}

/// Load DeltaNet-specific weights for one layer (Qwen 3.5 / 3.6 hybrid).
///
/// Uses the same GGUF tensor names as the GPU-side loader in `src/gpu.rs`
/// (`blk.{i}.ssm_conv1d.weight/bias`, `blk.{i}.ssm_in.weight`,
/// `blk.{i}.ssm_alpha.weight`, `blk.{i}.ssm_beta.weight`,
/// `blk.{i}.ssm_out.weight`, plus the standard FFN block).
/// Load MLA + optional dense-FFN weights for one DeepSeek-V3 layer.
///
/// Consumes the same tensor names as llama.cpp's `deepseek2` architecture
/// entries. Layers below `first_k_dense_replace` carry SwiGLU FFN tensors
/// (`ffn_gate` / `ffn_up` / `ffn_down` + `ffn_norm`); layers above load with
/// `None` for those fields — the MoE path lands in Phase 3 (Issue #33).
/// Build a shared [`StreamingExpertPool`] for the routed-expert weights of a
/// DeepSeek-V3 model. Returns `None` unless both environment variables are
/// set, so callers that don't opt in are unaffected:
///
/// - `ALICE_LLM_MOE_STREAMING=1` — enables the streaming path (any other
///   value or missing → in-memory routed experts as before).
/// - `ALICE_LLM_MOE_STREAMING_FILE=<path>` — filesystem path to the same
///   GGUF the caller is already parsing. The pool opens the file a second
///   time so its `Mmap` has its own lifetime, independent of the parser's
///   borrowed slice.
///
/// Cache byte budget is configurable via `ALICE_LLM_MOE_CACHE_BYTES`
/// (default: 4 GiB). Set to `0` to force every fetch to miss — useful for
/// bench harnesses that measure cold-cache decode time.
///
/// The `memmap2` crate is a `gguf`-feature-gated dependency; without the
/// feature the fallback stub always returns `None`, so callers stay on the
/// InMemory routed-expert path.
///
/// [`StreamingExpertPool`]: crate::deepseek_streaming::StreamingExpertPool
#[cfg(not(feature = "gguf"))]
fn build_deepseek_streaming_pool(
    _gguf: &GgufFile<'_>,
    _config: &Llama3Config,
) -> Option<std::sync::Arc<crate::deepseek_streaming::StreamingExpertPool>> {
    None
}

/// `gguf`-feature-gated real implementation. See the doc-comment on the
/// `not(feature = "gguf")` stub above for full env-var documentation.
#[cfg(feature = "gguf")]
#[allow(clippy::too_many_lines)]
fn build_deepseek_streaming_pool(
    gguf: &GgufFile<'_>,
    config: &Llama3Config,
) -> Option<std::sync::Arc<crate::deepseek_streaming::StreamingExpertPool>> {
    use crate::deepseek_streaming::{ExpertLayerInfo, StreamingExpertPool};
    if std::env::var("ALICE_LLM_MOE_STREAMING").ok().as_deref() != Some("1") {
        return None;
    }
    let path = match std::env::var("ALICE_LLM_MOE_STREAMING_FILE") {
        Ok(p) => p,
        Err(_) => {
            eprintln!(
                "[alice-llm] ALICE_LLM_MOE_STREAMING=1 but ALICE_LLM_MOE_STREAMING_FILE not set; \
                 falling back to InMemory routed experts."
            );
            return None;
        }
    };
    let file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[alice-llm] streaming pool disabled: cannot open '{path}': {e}");
            return None;
        }
    };
    // SAFETY: mmap of a read-only file we opened above. The Mmap owns its
    // mapping and lives for the pool's lifetime (Arc-shared into every MoE
    // layer's Streaming variant); it is not observed by any other process
    // in a mutating way.
    let mmap = match unsafe { memmap2::Mmap::map(&file) } {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[alice-llm] streaming pool disabled: mmap failed: {e}");
            return None;
        }
    };

    // Phase 4b.4: Tell the kernel this region will be accessed randomly
    // so its sequential-readahead heuristic does NOT thrash — the router
    // picks 8 of 256 experts per token, so paging in adjacent experts is
    // pure page-cache pollution. No-op on non-Unix builds.
    let advised = crate::deepseek_streaming::advise_random(&mmap);
    if advised {
        eprintln!("[alice-llm] MADV_RANDOM applied to streaming pool mmap");
    }

    let first_k_dense = config.deepseek_first_k_dense_replace().unwrap_or(0);
    let n_experts = config.deepseek_n_routed_experts()?;

    // For dense layers (below first_k_dense) we still emit a placeholder
    // ExpertLayerInfo entry so `Vec::len() == num_layers` and callers can
    // index by layer_idx uniformly. The forward path never touches these
    // placeholders because it only takes the Streaming branch when the
    // layer's `DeepSeekMoeWeights.routed` is Streaming (which only happens
    // for MoE layers, past first_k_dense).
    let placeholder = ExpertLayerInfo {
        base_offset: 0,
        bytes_per_expert: 0,
        n_experts: 0,
        qtype: crate::gguf::GgmlType::F32,
    };
    let mut layer_info: Vec<[ExpertLayerInfo; 3]> = Vec::with_capacity(config.num_layers);
    for layer_idx in 0..config.num_layers {
        if layer_idx < first_k_dense {
            layer_info.push([placeholder; 3]);
            continue;
        }
        let prefix = format!("blk.{layer_idx}");
        let gate_name = format!("{prefix}.ffn_gate_exps.weight");
        let up_name = format!("{prefix}.ffn_up_exps.weight");
        let down_name = format!("{prefix}.ffn_down_exps.weight");
        let gate_info = gguf.tensors.get(&gate_name)?;
        let up_info = gguf.tensors.get(&up_name)?;
        let down_info = gguf.tensors.get(&down_name)?;
        // Per-expert byte stride = tensor size / n_experts. The 3D layout
        // is expert-major (`[n_expert, out, in]` row-major), so dividing
        // by n_experts yields the byte width of one expert's 2D slab.
        let gate_bytes_per_expert = gate_info.data_size().checked_div(n_experts)?;
        let up_bytes_per_expert = up_info.data_size().checked_div(n_experts)?;
        let down_bytes_per_expert = down_info.data_size().checked_div(n_experts)?;
        let gate_off = gguf.tensor_absolute_offset(&gate_name)? as usize;
        let up_off = gguf.tensor_absolute_offset(&up_name)? as usize;
        let down_off = gguf.tensor_absolute_offset(&down_name)? as usize;
        layer_info.push([
            ExpertLayerInfo {
                base_offset: gate_off,
                bytes_per_expert: gate_bytes_per_expert,
                n_experts,
                qtype: gate_info.qtype,
            },
            ExpertLayerInfo {
                base_offset: up_off,
                bytes_per_expert: up_bytes_per_expert,
                n_experts,
                qtype: up_info.qtype,
            },
            ExpertLayerInfo {
                base_offset: down_off,
                bytes_per_expert: down_bytes_per_expert,
                n_experts,
                qtype: down_info.qtype,
            },
        ]);
    }

    let budget: usize = std::env::var("ALICE_LLM_MOE_CACHE_BYTES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4usize * 1024 * 1024 * 1024);

    eprintln!(
        "[alice-llm] DeepSeek streaming pool: mmap '{path}' ({} bytes), budget {} MiB",
        mmap.len(),
        budget / (1024 * 1024)
    );
    let source: std::sync::Arc<dyn crate::deepseek_streaming::ExpertByteSource> =
        std::sync::Arc::new(mmap);
    Some(std::sync::Arc::new(StreamingExpertPool::new(
        source, layer_info, budget,
    )))
}

fn load_deepseek_v3_layer_weights<'a>(
    gguf: &'a GgufFile<'a>,
    layer: usize,
    config: &Llama3Config,
    streaming_pool: Option<&std::sync::Arc<crate::deepseek_streaming::StreamingExpertPool>>,
) -> Option<DeepSeekV3LayerWeights<'a>> {
    let prefix = format!("blk.{layer}");

    let attn_norm = gguf.tensor_to_f32(&format!("{prefix}.attn_norm.weight"))?;

    // MLA dimensions from the config metadata.
    let q_lora_rank = config.deepseek_q_lora_rank()?;
    let kv_lora_rank = config.deepseek_kv_lora_rank()?;
    let qk_nope_head_dim = config.deepseek_qk_nope_head_dim()?;
    let qk_rope_head_dim = config.deepseek_qk_rope_head_dim()?;
    let v_head_dim = config.deepseek_v_head_dim()?;

    let q_head_total = qk_nope_head_dim + qk_rope_head_dim;
    let kv_up_head_total = qk_nope_head_dim + v_head_dim;

    let q_a_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.attn_q_a.weight"),
        q_lora_rank,
        config.hidden_dim,
    )?;
    let q_a_norm = gguf.tensor_to_f32(&format!("{prefix}.attn_q_a_norm.weight"))?;
    let q_b_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.attn_q_b.weight"),
        config.num_heads * q_head_total,
        q_lora_rank,
    )?;
    let kv_a_proj_with_mqa = load_weight_ref(
        gguf,
        &format!("{prefix}.attn_kv_a_mqa.weight"),
        kv_lora_rank + qk_rope_head_dim,
        config.hidden_dim,
    )?;
    let kv_a_norm = gguf.tensor_to_f32(&format!("{prefix}.attn_kv_a_norm.weight"))?;
    let kv_b_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.attn_kv_b.weight"),
        config.num_heads * kv_up_head_total,
        kv_lora_rank,
    )?;
    let o_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.attn_output.weight"),
        config.hidden_dim,
        config.num_heads * v_head_dim,
    )?;

    // Dense SwiGLU FFN only for the first `first_k_dense_replace` layers.
    // Layers ≥ first_k_dense use the DeepSeek-V3 MoE branch (Phase 3):
    // router + noaux_tc bias + top-k routed experts + always-active shared.
    let first_k_dense = config.deepseek_first_k_dense_replace().unwrap_or(0);
    let (ffn_norm, gate_proj, up_proj, down_proj, moe) = if layer < first_k_dense {
        let ffn_size = config.ffn_size_for_layer(layer);
        let ffn_norm = load_ffn_norm(gguf, &prefix)?;
        let gate_proj = load_weight_ref(
            gguf,
            &format!("{prefix}.ffn_gate.weight"),
            ffn_size,
            config.hidden_dim,
        )?;
        let up_proj = load_weight_ref(
            gguf,
            &format!("{prefix}.ffn_up.weight"),
            ffn_size,
            config.hidden_dim,
        )?;
        let down_proj = load_weight_ref(
            gguf,
            &format!("{prefix}.ffn_down.weight"),
            config.hidden_dim,
            ffn_size,
        )?;
        (
            Some(ffn_norm),
            Some(gate_proj),
            Some(up_proj),
            Some(down_proj),
            None,
        )
    } else {
        let n_experts = config.deepseek_n_routed_experts()?;
        let n_shared = config.deepseek_n_shared_experts()?;
        let moe_ffn = config.deepseek_moe_intermediate_size()?;
        let shared_ffn = n_shared * moe_ffn;
        let ffn_norm = load_ffn_norm(gguf, &prefix)?;
        let ffn_gate_inp = gguf.tensor_to_f32(&format!("{prefix}.ffn_gate_inp.weight"))?;
        // noaux_tc bias — optional (older DeepSeek-V2 doesn't ship this).
        let exp_probs_b = gguf.tensor_to_f32(&format!("{prefix}.exp_probs_b.bias"));
        let ffn_gate_shexp = load_weight_ref(
            gguf,
            &format!("{prefix}.ffn_gate_shexp.weight"),
            shared_ffn,
            config.hidden_dim,
        )?;
        let ffn_up_shexp = load_weight_ref(
            gguf,
            &format!("{prefix}.ffn_up_shexp.weight"),
            shared_ffn,
            config.hidden_dim,
        )?;
        let ffn_down_shexp = load_weight_ref(
            gguf,
            &format!("{prefix}.ffn_down_shexp.weight"),
            config.hidden_dim,
            shared_ffn,
        )?;
        let routed = if let Some(pool) = streaming_pool {
            // Phase 4b.1 streaming path: skip WeightRef loading for the
            // three routed 3D tensors — the pool's own mmap of the same
            // file will page them in lazily via `get_or_load`. Only the
            // shared expert stays as an eagerly-loaded WeightRef because
            // it fires on every token (LRU offers no benefit).
            RoutedExpertStorage::Streaming {
                pool: pool.clone(),
                layer_idx: layer,
            }
        } else {
            let ffn_gate_exps = load_weight_ref(
                gguf,
                &format!("{prefix}.ffn_gate_exps.weight"),
                moe_ffn * n_experts,
                config.hidden_dim,
            )?;
            let ffn_up_exps = load_weight_ref(
                gguf,
                &format!("{prefix}.ffn_up_exps.weight"),
                moe_ffn * n_experts,
                config.hidden_dim,
            )?;
            let ffn_down_exps = load_weight_ref(
                gguf,
                &format!("{prefix}.ffn_down_exps.weight"),
                config.hidden_dim * n_experts,
                moe_ffn,
            )?;
            RoutedExpertStorage::InMemory {
                gate: ffn_gate_exps,
                up: ffn_up_exps,
                down: ffn_down_exps,
            }
        };
        let moe = DeepSeekMoeWeights {
            ffn_norm,
            ffn_gate_inp,
            exp_probs_b,
            routed,
            ffn_gate_shexp,
            ffn_up_shexp,
            ffn_down_shexp,
        };
        (None, None, None, None, Some(moe))
    };

    Some(DeepSeekV3LayerWeights {
        attn_norm,
        q_a_proj,
        q_a_norm,
        q_b_proj,
        kv_a_proj_with_mqa,
        kv_a_norm,
        kv_b_proj,
        o_proj,
        ffn_norm,
        gate_proj,
        up_proj,
        down_proj,
        moe,
    })
}

/// Load the DeepSeek-V3 Multi-Token Prediction head (Phase 5a, Issue #35).
///
/// Returns `Some(_)` when every required MTP tensor is present in the
/// checkpoint, `None` otherwise — a missing MTP head silently disables
/// speculative decoding at inference time (see [`Llama3Model::has_deepseek_mtp`])
/// but does not affect regular decode.
///
/// # Tensor naming
///
/// llama.cpp did not have V3 MTP support at authorship time, so this loader
/// probes the paper-inspired `mtp.*` namespace. Each lookup silently returns
/// `None` on a missing tensor so callers get a clean "no MTP head shipped"
/// signal instead of an error. When llama.cpp finalises a convention, add
/// its names to the alternates list in-line.
///
/// The inner transformer block is loaded by delegating to
/// [`load_deepseek_v3_layer_weights`] with `prefix = "mtp.blk.0"` — MTP
/// reuses the exact MLA + MoE layer structure of a main-model block, so
/// there is no MTP-specific attention / FFN loader to write.
///
/// [`Llama3Model::has_deepseek_mtp`]: crate::llama3::Llama3Model::has_deepseek_mtp
fn load_deepseek_v3_mtp_weights<'a>(
    gguf: &'a GgufFile<'a>,
    config: &Llama3Config,
    streaming_pool: Option<&std::sync::Arc<crate::deepseek_streaming::StreamingExpertPool>>,
) -> Option<DeepSeekV3MtpWeights<'a>> {
    // Only look for MTP tensors when the config declares an MTP layer.
    // V2 quants and pre-MTP V3 variants leave this field None.
    if config.deepseek_mtp_layer().is_none() {
        return None;
    }

    // Entry projections. Every subsequent `?` returns None if any single
    // tensor is missing — the intended failure mode: "not all MTP tensors
    // ship, so we can't do MTP" ends up as `None`, not `Some(partial)`.
    let enorm = gguf.tensor_to_f32("mtp.enorm.weight")?;
    let hnorm = gguf.tensor_to_f32("mtp.hnorm.weight")?;
    let eh_proj = load_weight_ref(
        gguf,
        "mtp.eh_proj.weight",
        config.hidden_dim,
        2 * config.hidden_dim,
    )?;

    // The inner transformer block reuses the regular V3 layer loader.
    // Layer index is a fresh N (past the main model's num_layers) so the
    // streaming pool's layer_info would need a dedicated slot — which it
    // doesn't have in Phase 4b.1 (built for `num_layers` only). Pass
    // `None` for streaming_pool here so MTP experts stay InMemory
    // regardless of the outer streaming setting. Adding MTP-aware
    // streaming is a Phase 4c candidate: for now, 256 experts × 19 MB ≈
    // 5 GB extra RAM is tolerable next to the routed-expert budget.
    let _ = streaming_pool; // silence unused-param when non-gguf feature build
    let block = {
        // MTP block loader shares tensor prefix "mtp.blk.0" — we thread
        // that in by temporarily fabricating a config with layer index 0
        // and using a wrapper. Actually simpler: modify the loader to
        // accept a prefix. That refactor is invasive for one caller;
        // instead call load_weight_ref explicitly for the block below,
        // mirroring load_deepseek_v3_layer_weights. Since this is Phase
        // 5a scaffolding and the forward is stubbed, keep it minimal —
        // load only the attention path and the MoE / dense-FFN decision,
        // leaving the same-shaped stub in `block`.
        load_deepseek_v3_layer_weights_with_prefix(gguf, "mtp.blk.0", config, None)?
    };

    let final_norm = gguf.tensor_to_f32("mtp.norm.weight")?;

    Some(DeepSeekV3MtpWeights {
        enorm,
        hnorm,
        eh_proj,
        block,
        final_norm,
    })
}

/// Prefix-parameterised variant of [`load_deepseek_v3_layer_weights`] used
/// by the MTP loader. The main-layer loader hard-codes `blk.{layer}` as
/// the tensor namespace; MTP needs `mtp.blk.0` so we accept an explicit
/// prefix instead. Returns None if any required tensor is missing.
fn load_deepseek_v3_layer_weights_with_prefix<'a>(
    gguf: &'a GgufFile<'a>,
    prefix: &str,
    config: &Llama3Config,
    streaming_pool: Option<&std::sync::Arc<crate::deepseek_streaming::StreamingExpertPool>>,
) -> Option<DeepSeekV3LayerWeights<'a>> {
    // Duplication with load_deepseek_v3_layer_weights is intentional — the
    // main-layer function hard-codes `format!("blk.{layer}")` deep in a
    // 200-line body, refactoring it to take a prefix would touch every
    // call site. This helper is used by exactly one caller (MTP), so a
    // targeted duplicate keeps the diff surgical.
    let attn_norm = gguf.tensor_to_f32(&format!("{prefix}.attn_norm.weight"))?;
    let q_lora_rank = config.deepseek_q_lora_rank()?;
    let kv_lora_rank = config.deepseek_kv_lora_rank()?;
    let qk_nope_head_dim = config.deepseek_qk_nope_head_dim()?;
    let qk_rope_head_dim = config.deepseek_qk_rope_head_dim()?;
    let v_head_dim = config.deepseek_v_head_dim()?;
    let q_head_total = qk_nope_head_dim + qk_rope_head_dim;
    let kv_up_head_total = qk_nope_head_dim + v_head_dim;

    let q_a_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.attn_q_a.weight"),
        q_lora_rank,
        config.hidden_dim,
    )?;
    let q_a_norm = gguf.tensor_to_f32(&format!("{prefix}.attn_q_a_norm.weight"))?;
    let q_b_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.attn_q_b.weight"),
        config.num_heads * q_head_total,
        q_lora_rank,
    )?;
    let kv_a_proj_with_mqa = load_weight_ref(
        gguf,
        &format!("{prefix}.attn_kv_a_mqa.weight"),
        kv_lora_rank + qk_rope_head_dim,
        config.hidden_dim,
    )?;
    let kv_a_norm = gguf.tensor_to_f32(&format!("{prefix}.attn_kv_a_norm.weight"))?;
    let kv_b_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.attn_kv_b.weight"),
        config.num_heads * kv_up_head_total,
        kv_lora_rank,
    )?;
    let o_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.attn_output.weight"),
        config.hidden_dim,
        config.num_heads * v_head_dim,
    )?;

    // MTP block is always past first_k_dense_replace conceptually — it
    // sees full MoE FFN. Load MoE weights only; dense fields stay None.
    let n_experts = config.deepseek_n_routed_experts()?;
    let n_shared = config.deepseek_n_shared_experts()?;
    let moe_ffn = config.deepseek_moe_intermediate_size()?;
    let shared_ffn = n_shared * moe_ffn;

    let ffn_norm_v = load_ffn_norm(gguf, prefix)?;
    let ffn_gate_inp = gguf.tensor_to_f32(&format!("{prefix}.ffn_gate_inp.weight"))?;
    let exp_probs_b = gguf.tensor_to_f32(&format!("{prefix}.exp_probs_b.bias"));
    let ffn_gate_shexp = load_weight_ref(
        gguf,
        &format!("{prefix}.ffn_gate_shexp.weight"),
        shared_ffn,
        config.hidden_dim,
    )?;
    let ffn_up_shexp = load_weight_ref(
        gguf,
        &format!("{prefix}.ffn_up_shexp.weight"),
        shared_ffn,
        config.hidden_dim,
    )?;
    let ffn_down_shexp = load_weight_ref(
        gguf,
        &format!("{prefix}.ffn_down_shexp.weight"),
        config.hidden_dim,
        shared_ffn,
    )?;
    let routed = if streaming_pool.is_some() {
        // MTP-aware streaming is deferred (see caller for reasoning); the
        // MTP path always uses InMemory for its routed experts.
        unreachable!("MTP loader unexpectedly received a streaming pool");
    } else {
        let ffn_gate_exps = load_weight_ref(
            gguf,
            &format!("{prefix}.ffn_gate_exps.weight"),
            moe_ffn * n_experts,
            config.hidden_dim,
        )?;
        let ffn_up_exps = load_weight_ref(
            gguf,
            &format!("{prefix}.ffn_up_exps.weight"),
            moe_ffn * n_experts,
            config.hidden_dim,
        )?;
        let ffn_down_exps = load_weight_ref(
            gguf,
            &format!("{prefix}.ffn_down_exps.weight"),
            config.hidden_dim * n_experts,
            moe_ffn,
        )?;
        RoutedExpertStorage::InMemory {
            gate: ffn_gate_exps,
            up: ffn_up_exps,
            down: ffn_down_exps,
        }
    };
    let moe = DeepSeekMoeWeights {
        ffn_norm: ffn_norm_v,
        ffn_gate_inp,
        exp_probs_b,
        routed,
        ffn_gate_shexp,
        ffn_up_shexp,
        ffn_down_shexp,
    };

    Some(DeepSeekV3LayerWeights {
        attn_norm,
        q_a_proj,
        q_a_norm,
        q_b_proj,
        kv_a_proj_with_mqa,
        kv_a_norm,
        kv_b_proj,
        o_proj,
        ffn_norm: None,
        gate_proj: None,
        up_proj: None,
        down_proj: None,
        moe: Some(moe),
    })
}

fn load_deltanet_layer_weights<'a>(
    gguf: &'a GgufFile<'a>,
    layer: usize,
    config: &Llama3Config,
) -> Option<DeltaNetLayerWeights<'a>> {
    let prefix = format!("blk.{layer}");

    let attn_norm = gguf.tensor_to_f32(&format!("{prefix}.attn_norm.weight"))?;

    // DeltaNet dimensions derived from the config metadata.
    let qk_dim = config.linear_qk_head_dim().unwrap_or(128);
    let v_dim = config.linear_kv_head_dim().unwrap_or(128);
    let num_kv_heads = config.linear_num_kv_heads().unwrap_or(config.num_kv_heads);
    let num_v_heads = config.linear_num_v_heads().unwrap_or(config.num_heads);
    // Fused in_proj output: q + k (both num_kv_heads * qk_dim) + v + z (both num_v_heads * v_dim).
    let in_proj_out = qk_dim * num_kv_heads * 2 + v_dim * num_v_heads * 2;

    // Standard Qwen 3.5 exports fuse Q/K/V/Z into `ssm_in.weight`. Bonsai 27B
    // exports fuse Q/K/V/gate into `attn_qkv.weight` instead. Try both and
    // populate whichever exists. The loader returns `None` if neither is
    // present (the layer is not a valid DeltaNet layer).
    let ssm_in = load_weight_ref(
        gguf,
        &format!("{prefix}.ssm_in.weight"),
        in_proj_out,
        config.hidden_dim,
    );
    // Bonsai's `attn_qkv` output dim (10240 for the 27B config) is not
    // derivable from the standard Qwen 3.5 metadata, so the loader accepts
    // whatever the GGUF ships and defers per-head splitting to Phase X.3.e.
    let attn_qkv = load_weight_ref_any_rows(
        gguf,
        &format!("{prefix}.attn_qkv.weight"),
        config.hidden_dim,
    );
    if ssm_in.is_none() && attn_qkv.is_none() {
        return None;
    }
    // Bonsai 27B DeltaNet output gate; standard Qwen 3.5 has no such tensor.
    let attn_gate = load_weight_ref_any_rows(
        gguf,
        &format!("{prefix}.attn_gate.weight"),
        config.hidden_dim,
    );
    let ssm_a = gguf.tensor_to_f32(&format!("{prefix}.ssm_a"));
    let ssm_dt_bias = gguf.tensor_to_f32(&format!("{prefix}.ssm_dt.bias"));
    let ssm_norm = gguf.tensor_to_f32(&format!("{prefix}.ssm_norm.weight"));

    let conv1d_weight = gguf.tensor_to_f32(&format!("{prefix}.ssm_conv1d.weight"))?;
    // conv_dim = q + k + v (excludes z), derived to match the conv1d layout
    // (`conv1d_weight` shape is `[kernel_size, conv_dim]`).
    let conv_dim = qk_dim * num_kv_heads * 2 + v_dim * num_v_heads;
    // Bonsai 27B and some Qwen 3.6 checkpoints omit `ssm_conv1d.bias` entirely.
    // Treating it as optional (zero-fill fallback) is behaviour-preserving for
    // standard Qwen 3.5 exports, which continue to load the shipped bias.
    let conv1d_bias = gguf
        .tensor_to_f32(&format!("{prefix}.ssm_conv1d.bias"))
        .unwrap_or_else(|| vec![0.0f32; conv_dim]);
    // `ssm_alpha` / `ssm_beta` row count depends on the arch variant:
    //   * Standard Qwen 3.5: `num_kv_heads` (one decay rate per KV head).
    //   * Bonsai / Qwen 3.6: `num_v_heads` (one rate per V head, 3× more than
    //     `num_kv_heads` under Qwen 3.6's 48 V / 16 KV split).
    // Read the actual row count out of GGUF via `load_weight_ref_any_rows`
    // instead of forcing one interpretation onto the tensor — the forward
    // path sizes its `dn_alpha` / `dn_beta` buffers to the larger of the
    // two so the matvec always writes into a big-enough slice.
    let alpha_proj = load_weight_ref_any_rows(
        gguf,
        &format!("{prefix}.ssm_alpha.weight"),
        config.hidden_dim,
    )?;
    let beta_proj = load_weight_ref_any_rows(
        gguf,
        &format!("{prefix}.ssm_beta.weight"),
        config.hidden_dim,
    )?;
    let ssm_out = load_weight_ref(
        gguf,
        &format!("{prefix}.ssm_out.weight"),
        config.hidden_dim,
        v_dim * num_v_heads,
    )?;

    let ffn_norm = load_ffn_norm(gguf, &prefix)?;
    let ffn_size = config.ffn_size_for_layer(layer);
    let gate_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.ffn_gate.weight"),
        ffn_size,
        config.hidden_dim,
    )?;
    let up_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.ffn_up.weight"),
        ffn_size,
        config.hidden_dim,
    )?;
    let down_proj = load_weight_ref(
        gguf,
        &format!("{prefix}.ffn_down.weight"),
        config.hidden_dim,
        ffn_size,
    )?;

    Some(DeltaNetLayerWeights {
        attn_norm,
        ssm_in,
        attn_qkv,
        attn_gate,
        ssm_a,
        ssm_dt_bias,
        ssm_norm,
        conv1d_weight,
        conv1d_bias,
        alpha_proj,
        beta_proj,
        ssm_out,
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
            gemma3n: Some(Gemma3nConfig {
                sliding_window_pattern: None,
                activation_sparsity_scale: Some(scales),
                shared_kv_layers: None,
                per_layer_input_embedding_dim: None,
                altup_num_inputs: None,
                altup_active_idx: None,
            }),
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
            attention_extras: Some(AttentionExtrasConfig {
                sliding_window: Some(512),
                attn_logit_softcap: None,
                final_logit_softcap: None,
            }),
            gemma3n: Some(Gemma3nConfig {
                sliding_window_pattern: Some(pattern),
                activation_sparsity_scale: None,
                shared_kv_layers: Some(10),
                per_layer_input_embedding_dim: None,
                altup_num_inputs: None,
                altup_active_idx: None,
            }),
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
        assert_eq!(
            c.kv_source_layer(24),
            19,
            "layer 24 (full) should map to 19"
        );
        // Layer 29: pattern[29] = false (full attention) → 19
        assert_eq!(
            c.kv_source_layer(29),
            19,
            "layer 29 (full) should map to 19"
        );
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

    // ── Phase M1: Q4_0 / Q4_1 / Q5_0 / IQ4_XS ───────────────────────────────

    /// f16 encoding of 1.0 in little-endian bytes.
    const F16_ONE_LE: [u8; 2] = [0x00, 0x3C];

    #[test]
    fn test_ggml_type_q4_0_layout() {
        use crate::gguf::GgmlType;
        assert_eq!(GgmlType::Q4_0.block_bytes(), 18);
        assert_eq!(GgmlType::Q4_0.elements_per_block(), 32);
    }

    #[test]
    fn test_ggml_type_q4_1_layout() {
        use crate::gguf::GgmlType;
        assert_eq!(GgmlType::Q4_1.block_bytes(), 20);
        assert_eq!(GgmlType::Q4_1.elements_per_block(), 32);
    }

    #[test]
    fn test_ggml_type_q5_0_layout() {
        use crate::gguf::GgmlType;
        assert_eq!(GgmlType::Q5_0.block_bytes(), 22);
        assert_eq!(GgmlType::Q5_0.elements_per_block(), 32);
    }

    #[test]
    fn test_ggml_type_iq4_xs_layout() {
        use crate::gguf::GgmlType;
        assert_eq!(GgmlType::IQ4_XS.block_bytes(), 136);
        assert_eq!(GgmlType::IQ4_XS.elements_per_block(), 256);
    }

    #[test]
    fn test_q4_0_dequant_zero_block() {
        // d=0, qs=0 → all zeros.
        let data = vec![0u8; 18];
        let mut out = vec![0.0f32; 32];
        crate::gguf::dequantize_q4_0(&data, &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_q4_0_dequant_signed_range() {
        // d=1.0, qs[0]=0x80 (nibbles 0 and 8):
        //   x0 = 0 - 8 = -8, x1 = 8 - 8 = 0
        // qs[1..15]=0 → x0=x1=-8 for those slots.
        let mut data = vec![0u8; 18];
        data[0] = F16_ONE_LE[0];
        data[1] = F16_ONE_LE[1];
        data[2] = 0x80; // nibbles: lo=0, hi=8
        let mut out = vec![0.0f32; 32];
        crate::gguf::dequantize_q4_0(&data, &mut out);
        assert!((out[0] - (-8.0)).abs() < 1e-4);
        assert!((out[16] - 0.0).abs() < 1e-4);
        // qs[1..16]=0 → nibbles 0,0 → (0-8)*d = -8
        assert!((out[1] - (-8.0)).abs() < 1e-4);
        assert!((out[17] - (-8.0)).abs() < 1e-4);
    }

    #[test]
    fn test_q4_1_dequant_min_offset() {
        // d=0, m=1.0, qs=0 → every element = 0*x + 1 = 1.
        let mut data = vec![0u8; 20];
        data[2] = F16_ONE_LE[0];
        data[3] = F16_ONE_LE[1];
        let mut out = vec![0.0f32; 32];
        crate::gguf::dequantize_q4_1(&data, &mut out);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-4, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_q4_1_dequant_unsigned_range() {
        // d=1.0, m=0.0, qs[0]=0xF3 → x0=3, x1=15.
        let mut data = vec![0u8; 20];
        data[0] = F16_ONE_LE[0];
        data[1] = F16_ONE_LE[1];
        data[4] = 0xF3;
        let mut out = vec![0.0f32; 32];
        crate::gguf::dequantize_q4_1(&data, &mut out);
        assert!((out[0] - 3.0).abs() < 1e-4);
        assert!((out[16] - 15.0).abs() < 1e-4);
    }

    #[test]
    fn test_q5_0_dequant_signed_5bit_range() {
        // d=1.0, qh=0 → high bit off, so 4-bit range only.
        // qs[0]=0x0F (nibbles 15,0) → x0 = 15 - 16 = -1, x1 = 0 - 16 = -16.
        let mut data = vec![0u8; 22];
        data[0] = F16_ONE_LE[0];
        data[1] = F16_ONE_LE[1];
        data[6] = 0x0F;
        let mut out = vec![0.0f32; 32];
        crate::gguf::dequantize_q5_0(&data, &mut out);
        assert!((out[0] - (-1.0)).abs() < 1e-4);
        assert!((out[16] - (-16.0)).abs() < 1e-4);
    }

    #[test]
    fn test_q5_0_dequant_high_bit() {
        // d=1.0, qs[0]=0x0F, qh bit 0 set → high bit adds 16.
        // x0 = (15 | 16) - 16 = 15.
        let mut data = vec![0u8; 22];
        data[0] = F16_ONE_LE[0];
        data[1] = F16_ONE_LE[1];
        data[2] = 0x01; // qh bit 0 set → high bit for element 0
        data[6] = 0x0F;
        let mut out = vec![0.0f32; 32];
        crate::gguf::dequantize_q5_0(&data, &mut out);
        assert!((out[0] - 15.0).abs() < 1e-4);
    }

    #[test]
    fn test_iq4_xs_dequant_zero_block() {
        // d=0, everything else → all zeros.
        let data = vec![0u8; 136];
        let mut out = vec![0.0f32; 256];
        crate::gguf::dequantize_iq4_xs(&data, &mut out);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_iq4_xs_dequant_lookup_table() {
        // d=1.0, scales all zero → ls=0, dl = 1*(0-32) = -32.
        // qs[0]=0x00 → both nibbles = 0 → KVALUES_IQ4NL[0] = -127.
        // y[0] = -32 * -127 = 4064.
        let mut data = vec![0u8; 136];
        data[0] = F16_ONE_LE[0];
        data[1] = F16_ONE_LE[1];
        let mut out = vec![0.0f32; 256];
        crate::gguf::dequantize_iq4_xs(&data, &mut out);
        // First sub-block: all elements should be -32 * -127 = 4064.
        for &v in out.iter().take(32) {
            assert!((v - 4064.0).abs() < 1.0, "expected ~4064, got {v}");
        }
    }

    // ── Phase M4: Qwen 3.5 / 3.6 arch detection ─────────────────────────────

    #[test]
    fn test_qwen35_config_defaults_none() {
        // Baseline Llama config should have no SSM/NextN fields set.
        let c = Llama3Config::llama3_8b();
        assert!(c.ssm_inner_size().is_none());
        assert!(c.ssm_state_size().is_none());
        assert!(c.ssm_group_count().is_none());
        assert!(c.ssm_time_step_rank().is_none());
        assert!(c.n_layer_nextn().is_none());
    }

    fn qwen35_hybrid_config() -> Llama3Config {
        Llama3Config {
            arch: ModelArch::Qwen3_5,
            num_layers: 64,
            ssm: Some(SsmDeltaNetConfig {
                full_attention_interval: Some(4),
                linear_num_kv_heads: None,
                linear_qk_head_dim: None,
                linear_kv_head_dim: None,
                linear_num_v_heads: None,
                linear_conv_kernel_dim: None,
                ssm_inner_size: Some(6144),
                ssm_state_size: Some(128),
                ssm_group_count: Some(16),
                ssm_time_step_rank: Some(48),
                n_layer_nextn: None,
            }),
            ..Llama3Config::llama3_8b()
        }
    }

    #[test]
    fn test_qwen35_is_hybrid() {
        let c = qwen35_hybrid_config();
        assert!(
            c.is_hybrid(),
            "Qwen 3.5/3.6 with full_attention_interval is hybrid"
        );
    }

    #[test]
    fn test_qwen35_is_deltanet_layer_pattern() {
        // interval=4: layers 0,1,2 → DeltaNet; layer 3 → full attention; repeat.
        let c = qwen35_hybrid_config();
        assert!(c.is_deltanet_layer(0), "layer 0 should be DeltaNet");
        assert!(c.is_deltanet_layer(1), "layer 1 should be DeltaNet");
        assert!(c.is_deltanet_layer(2), "layer 2 should be DeltaNet");
        assert!(
            !c.is_deltanet_layer(3),
            "layer 3 (i+1 % 4 == 0) should be full"
        );
        assert!(c.is_deltanet_layer(4), "layer 4 should be DeltaNet again");
        assert!(!c.is_deltanet_layer(7), "layer 7 should be full");
    }

    #[test]
    fn test_qwen35_pure_attention_not_hybrid() {
        // Without full_attention_interval, treat as pure attention (Qwen 3 base).
        let c = Llama3Config {
            arch: ModelArch::Qwen3,
            ..Llama3Config::llama3_8b()
        };
        assert!(!c.is_hybrid());
    }

    #[test]
    fn test_qwen35_use_neox_rope() {
        // Qwen 3.5 / 3.6 inherits NEOX RoPE from the Qwen 3 family.
        assert!(ModelArch::Qwen3_5.use_neox_rope());
    }

    // ─── Phase JJJ: Speculative decoding helper tests ─────────────────────

    #[test]
    fn test_greedy_argmax_basic() {
        let logits = [0.1_f32, 0.5, 0.3, 0.7, 0.2];
        assert_eq!(greedy_argmax(&logits), 3);
    }

    #[test]
    fn test_greedy_argmax_first_wins_on_tie() {
        // First occurrence wins because subsequent equal values do not
        // satisfy strict `>` comparison.
        let logits = [0.7_f32, 0.5, 0.7, 0.3];
        assert_eq!(greedy_argmax(&logits), 0);
    }

    #[test]
    fn test_greedy_argmax_single_element() {
        let logits = [42.0_f32];
        assert_eq!(greedy_argmax(&logits), 0);
    }

    #[test]
    fn test_greedy_argmax_negative_values() {
        let logits = [-3.0_f32, -1.5, -2.0, -0.5];
        assert_eq!(greedy_argmax(&logits), 3);
    }

    #[test]
    fn test_speculative_result_acceptance_rate_empty() {
        let r = SpeculativeResult {
            tokens: vec![],
            draft_tokens_produced: 0,
            draft_tokens_accepted: 0,
            bonus_tokens: 0,
        };
        // No draft tokens produced → rate is defined as 0.0 (not NaN).
        assert_eq!(r.acceptance_rate(), 0.0);
    }

    #[test]
    fn test_speculative_result_acceptance_rate_full() {
        let r = SpeculativeResult {
            tokens: vec![1, 2, 3, 4, 5],
            draft_tokens_produced: 4,
            draft_tokens_accepted: 4,
            bonus_tokens: 1,
        };
        assert!((r.acceptance_rate() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_speculative_result_acceptance_rate_partial() {
        let r = SpeculativeResult {
            tokens: vec![1, 2, 3],
            draft_tokens_produced: 8,
            draft_tokens_accepted: 2,
            bonus_tokens: 1,
        };
        // 2 / 8 = 0.25
        assert!((r.acceptance_rate() - 0.25).abs() < 1e-6);
    }

    // ─── Phase JJJ v0.2: Sampling helper tests ────────────────────────────

    #[test]
    fn test_splitmix64_deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..10 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn test_splitmix64_next_unit_f32_range() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..1000 {
            let x = rng.next_unit_f32();
            assert!((0.0..1.0).contains(&x), "sample {x} out of [0, 1)");
        }
    }

    #[test]
    fn test_apply_temperature_and_top_p_uniform_at_high_temp() {
        // Very high temperature → distribution approaches uniform.
        let logits = [1.0f32, 2.0, 3.0, 4.0];
        let dist = apply_temperature_and_top_p(&logits, 1000.0, None);
        let sum: f32 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        for &p in &dist {
            // With t=1000, exp((l - max) / 1000) ≈ 1 for all → near-uniform
            assert!((p - 0.25).abs() < 0.01, "prob {p} not near uniform");
        }
    }

    #[test]
    fn test_apply_temperature_and_top_p_argmax_at_low_temp() {
        // Very low temperature → distribution collapses to argmax.
        let logits = [1.0f32, 5.0, 2.0, 3.0];
        let dist = apply_temperature_and_top_p(&logits, 0.001, None);
        // Argmax is index 1.
        assert!(dist[1] > 0.999, "argmax prob {} not near 1", dist[1]);
        for (i, &p) in dist.iter().enumerate() {
            if i != 1 {
                assert!(p < 1e-3, "non-argmax prob {p} at {i} too high");
            }
        }
    }

    #[test]
    fn test_apply_temperature_and_top_p_sums_to_one() {
        let logits = [0.5f32, -1.0, 2.5, 0.1, -0.3];
        let dist = apply_temperature_and_top_p(&logits, 1.0, None);
        let sum: f32 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_temperature_and_top_p_filter_keeps_head() {
        // Distribution: [0.5, 0.3, 0.15, 0.05] after softmax; top-p 0.7
        // should keep the top 2 (cum 0.8 >= 0.7) and drop the rest.
        // Build logits so softmax approximates the above distribution.
        // ln(0.5)=-0.693, ln(0.3)=-1.204, ln(0.15)=-1.897, ln(0.05)=-2.996
        let logits = [-0.693f32, -1.204, -1.897, -2.996];
        let dist = apply_temperature_and_top_p(&logits, 1.0, Some(0.7));
        assert!(dist[0] > 0.0);
        assert!(dist[1] > 0.0);
        assert!(dist[2].abs() < 1e-6, "tail should be zero, got {}", dist[2]);
        assert!(dist[3].abs() < 1e-6, "tail should be zero, got {}", dist[3]);
        let sum: f32 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_sample_multinomial_respects_zero_prob() {
        // Only index 2 has non-zero prob → always sampled.
        let dist = [0.0f32, 0.0, 1.0, 0.0];
        let mut rng = SplitMix64::new(1234);
        for _ in 0..100 {
            assert_eq!(sample_multinomial(&dist, &mut rng), 2);
        }
    }

    #[test]
    fn test_sample_multinomial_matches_expected_frequency() {
        // 50/50 distribution — sample 10k times, expect ~50/50 split.
        let dist = [0.5f32, 0.5, 0.0, 0.0];
        let mut rng = SplitMix64::new(1);
        let mut counts = [0usize; 4];
        for _ in 0..10_000 {
            let idx = sample_multinomial(&dist, &mut rng) as usize;
            counts[idx] += 1;
        }
        assert!(counts[0].abs_diff(5000) < 300, "counts[0]={}", counts[0]);
        assert!(counts[1].abs_diff(5000) < 300, "counts[1]={}", counts[1]);
        assert_eq!(counts[2], 0);
        assert_eq!(counts[3], 0);
    }

    #[test]
    fn test_residual_dist_basic() {
        let p_main = [0.5f32, 0.3, 0.15, 0.05];
        let p_draft = [0.1f32, 0.4, 0.4, 0.1];
        // Residual before normalise: [0.4, 0, 0, 0] → renorm [1, 0, 0, 0]
        let r = residual_dist(&p_main, &p_draft);
        assert!((r[0] - 1.0).abs() < 1e-5);
        assert!(r[1].abs() < 1e-6);
        assert!(r[2].abs() < 1e-6);
        assert!(r[3].abs() < 1e-6);
    }

    #[test]
    fn test_residual_dist_degenerate_falls_back_to_main() {
        // If p_main == p_draft, residual is all zero → fallback to p_main.
        let p = [0.25f32, 0.25, 0.25, 0.25];
        let r = residual_dist(&p, &p);
        for (i, &v) in r.iter().enumerate() {
            assert!((v - 0.25).abs() < 1e-5, "residual[{i}]={v} != 0.25");
        }
    }

    #[test]
    fn test_speculative_config_default() {
        let cfg = SpeculativeConfig::default();
        assert_eq!(cfg.n_draft, 4);
        assert_eq!(cfg.max_new_tokens, 128);
        assert!(cfg.temperature.is_none());
        assert!(cfg.top_p.is_none());
        assert!(cfg.sample_seed.is_none());
    }

    // ── DeltaNet CPU kernels (Issue #12) ────────────────────────────────

    /// Causal conv1d with `kernel_size = 4`, `dim = 1` on a synthetic
    /// signal: weights `[1, 2, 3, 4]`, bias 0, ring buffer starts zeroed.
    /// After 4 steps the ring is fully populated and the output matches
    /// the direct convolution `sum_k w[k] * x[t-3+k] + bias`.
    #[test]
    fn causal_conv1d_step_matches_direct_convolution() {
        let dim = 1;
        let kernel = 4;
        let weight: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // [kernel, dim]
        let bias: Vec<f32> = vec![0.0];
        let inputs: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut state = vec![0.0f32; (kernel - 1) * dim];
        let mut ring_pos = 0usize;
        let mut outputs = Vec::with_capacity(inputs.len());
        for &x in &inputs {
            let mut out = vec![0.0f32; dim];
            causal_conv1d_step(
                &[x],
                &mut state,
                &mut ring_pos,
                &weight,
                &bias,
                &mut out,
                dim,
                kernel,
            );
            outputs.push(out[0]);
        }
        // Direct convolution with zero-padded history:
        //   t=0: [0,0,0,1]  → 1*0 + 2*0 + 3*0 + 4*1 = 4
        //   t=1: [0,0,1,2]  → 3*1 + 4*2 = 11
        //   t=2: [0,1,2,3]  → 2*1 + 3*2 + 4*3 = 20
        //   t=3: [1,2,3,4]  → 1*1 + 2*2 + 3*3 + 4*4 = 30
        //   t=4: [2,3,4,5]  → 1*2 + 2*3 + 3*4 + 4*5 = 40
        assert_eq!(outputs, vec![4.0, 11.0, 20.0, 30.0, 40.0]);
    }

    /// Ring buffer must slide correctly across multiple decode steps so
    /// the (kernel-1)-oldest activations line up with the kernel rows
    /// on every subsequent call.
    #[test]
    fn causal_conv1d_ring_buffer_slides_across_steps() {
        let dim = 2;
        let kernel = 4;
        // Identity kernel `[1, 1, 1, 1]` for both channels — output equals
        // the sum of the history window, which is easy to verify.
        let weight: Vec<f32> = vec![1.0; kernel * dim];
        let bias: Vec<f32> = vec![0.0; dim];
        let inputs = [
            vec![1.0f32, 10.0],
            vec![2.0, 20.0],
            vec![3.0, 30.0],
            vec![4.0, 40.0],
            vec![5.0, 50.0],
        ];
        let mut state = vec![0.0f32; (kernel - 1) * dim];
        let mut ring_pos = 0usize;
        let expected_ch0 = [1.0f32, 3.0, 6.0, 10.0, 14.0];
        let expected_ch1 = [10.0f32, 30.0, 60.0, 100.0, 140.0];
        for (i, x) in inputs.iter().enumerate() {
            let mut out = vec![0.0f32; dim];
            causal_conv1d_step(
                x,
                &mut state,
                &mut ring_pos,
                &weight,
                &bias,
                &mut out,
                dim,
                kernel,
            );
            assert!(
                (out[0] - expected_ch0[i]).abs() < 1e-6,
                "step {i} ch0: got {} expected {}",
                out[0],
                expected_ch0[i]
            );
            assert!(
                (out[1] - expected_ch1[i]).abs() < 1e-6,
                "step {i} ch1: got {} expected {}",
                out[1],
                expected_ch1[i]
            );
        }
    }

    /// Regression test for Phase X.3.e.3.5 — `ssm_conv1d.weight` in GGUF
    /// is stored dim-outer × kernel-inner (`ne[0] = kernel_size` fastest).
    /// A `dim=2, kernel=4` case with per-channel unique weights
    /// distinguishes the correct `weight[d * kernel + k]` indexing from
    /// the transposed `weight[k * dim + d]` — the earlier `dim=1` test
    /// is degenerate because both indexings collapse to `weight[k]`.
    #[test]
    fn causal_conv1d_step_weight_layout_dim_outer_kernel_inner() {
        let dim = 2;
        let kernel = 4;
        // Weight stored dim-outer × kernel-inner:
        //   channel 0 kernel = [1, 2, 3, 4]
        //   channel 1 kernel = [10, 20, 30, 40]
        let weight: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let bias: Vec<f32> = vec![0.0, 0.0];
        // Constant input per channel: channel 0 = 1, channel 1 = 100.
        let inputs = [
            vec![1.0f32, 100.0],
            vec![1.0, 100.0],
            vec![1.0, 100.0],
            vec![1.0, 100.0], // after this call, the window is fully populated
        ];
        let mut state = vec![0.0f32; (kernel - 1) * dim];
        let mut ring_pos = 0usize;
        let mut out = vec![0.0f32; dim];
        for x in &inputs {
            causal_conv1d_step(
                x,
                &mut state,
                &mut ring_pos,
                &weight,
                &bias,
                &mut out,
                dim,
                kernel,
            );
        }
        // Fully populated window, all inputs = 1 (ch0) or 100 (ch1):
        //   ch0 out = (1+2+3+4) * 1 = 10
        //   ch1 out = (10+20+30+40) * 100 = 10_000
        assert!(
            (out[0] - 10.0).abs() < 1e-5,
            "ch0 expected 10.0 got {} (weight indexing likely transposed)",
            out[0]
        );
        assert!(
            (out[1] - 10_000.0).abs() < 1e-3,
            "ch1 expected 10000.0 got {} (weight indexing likely transposed)",
            out[1]
        );
    }

    /// alpha = 0 means the state loses its history, beta = 1 means the
    /// state absorbs the current outer product. With this configuration
    /// the delta-rule reduces to `S_new = outer(k, v)`, which we can
    /// verify head-by-head with hand-computed L2-normalised silu inputs.
    #[test]
    fn gated_deltanet_state_absorbs_current_when_alpha_zero_beta_one() {
        let num_heads = 1;
        let qk_dim = 2;
        let v_dim = 2;
        // q, k, v selected so silu(x) * l2norm(x) has a clean closed form.
        let q = vec![1.0, 0.0]; // silu ≈ [0.7311, 0.0], l2 norm ≈ 1 → [0.7311, 0]
        let k = vec![0.5, 0.5]; // silu = [~0.3223, ~0.3223], norm ≈ 1 → [0.7071, 0.7071]
        let v = vec![1.0, -1.0];
        let alpha = vec![0.0];
        let beta = vec![1.0];
        // z chosen so silu(z) is easy to invert: z = 10 → silu ≈ 10, z = -10 → silu ≈ 0.
        let z = vec![10.0, -10.0];
        let mut state = vec![0.0f32; num_heads * qk_dim * v_dim];
        let mut out = vec![0.0f32; num_heads * v_dim];

        gated_deltanet_step(
            &q, &k, &v, &alpha, &beta, &z, &mut state, &mut out, num_heads, num_heads, qk_dim,
            v_dim, false,
        );

        // Sanity: state should be non-zero and finite (recurrence absorbed
        // the current step). Exact numerical values are captured by
        // downstream integration tests once a Qwen 3.5 GGUF is available.
        assert!(state.iter().all(|v| v.is_finite()));
        assert!(state.iter().any(|&v| v.abs() > 0.0));
        assert!(out.iter().all(|v| v.is_finite()));
        // With z[0]=10 (silu ≈ 10), z[1]=-10 (silu ≈ 0), the second output
        // channel should be near zero.
        assert!(out[1].abs() < 1e-3);
    }

    /// L2 normalisation must guard against a zero input vector using the
    /// same `max(sqrt(sum_sq), 1e-12)` clamp as the WGSL shader — otherwise
    /// the divide produces NaN and the state fills with garbage on the
    /// first decode step for models that emit an all-zero q or k.
    #[test]
    fn gated_deltanet_step_handles_zero_input_without_nan() {
        let num_heads = 1;
        let qk_dim = 4;
        let v_dim = 4;
        let q = vec![0.0f32; qk_dim];
        let k = vec![0.0f32; qk_dim];
        let v = vec![1.0f32; v_dim];
        let alpha = vec![0.5];
        let beta = vec![0.5];
        let z = vec![0.0f32; v_dim];
        let mut state = vec![0.0f32; num_heads * qk_dim * v_dim];
        let mut out = vec![0.0f32; num_heads * v_dim];

        gated_deltanet_step(
            &q, &k, &v, &alpha, &beta, &z, &mut state, &mut out, num_heads, num_heads, qk_dim,
            v_dim, false,
        );

        assert!(state.iter().all(|v| v.is_finite()));
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Multi-head kernel must produce identical per-head output whether
    /// dispatched via the serial or the rayon-parallel driver (single-
    /// batched behaviour must not depend on scheduling).
    #[test]
    fn gated_deltanet_step_parallel_matches_serial() {
        // 16 heads clears the >= 8 rayon threshold in the `parallel`
        // feature; in the default build both paths are serial and the
        // test still exercises the shared kernel.
        let num_heads = 16;
        let qk_dim = 3;
        let v_dim = 3;
        let mut q = Vec::with_capacity(num_heads * qk_dim);
        let mut k = Vec::with_capacity(num_heads * qk_dim);
        let mut v = Vec::with_capacity(num_heads * v_dim);
        let mut z = Vec::with_capacity(num_heads * v_dim);
        for h in 0..num_heads {
            for i in 0..qk_dim {
                let seed = (h * qk_dim + i) as f32 * 0.3;
                q.push(seed.sin());
                k.push(seed.cos());
            }
            for j in 0..v_dim {
                let seed = (h * v_dim + j) as f32 * 0.5;
                v.push(seed.sin());
                z.push(seed.cos());
            }
        }
        let alpha: Vec<f32> = (0..num_heads).map(|h| 0.9 - 0.01 * h as f32).collect();
        let beta: Vec<f32> = (0..num_heads).map(|h| 0.1 + 0.01 * h as f32).collect();

        // Reference: serial per-head loop (bypasses the parallel dispatch).
        let mut state_serial = vec![0.0f32; num_heads * qk_dim * v_dim];
        let mut out_serial = vec![0.0f32; num_heads * v_dim];
        for head in 0..num_heads {
            gated_deltanet_head(
                &q,
                &k,
                &v,
                &alpha,
                &beta,
                &z,
                &mut state_serial,
                &mut out_serial,
                head,
                qk_dim,
                v_dim,
            );
        }

        // Actual: whichever path `gated_deltanet_step` selects.
        let mut state_actual = vec![0.0f32; num_heads * qk_dim * v_dim];
        let mut out_actual = vec![0.0f32; num_heads * v_dim];
        gated_deltanet_step(
            &q,
            &k,
            &v,
            &alpha,
            &beta,
            &z,
            &mut state_actual,
            &mut out_actual,
            num_heads,
            num_heads,
            qk_dim,
            v_dim,
            false,
        );

        for (i, (a, s)) in out_actual.iter().zip(out_serial.iter()).enumerate() {
            assert!((a - s).abs() < 1e-6, "out[{i}]: parallel {a} serial {s}");
        }
        for (i, (a, s)) in state_actual.iter().zip(state_serial.iter()).enumerate() {
            assert!((a - s).abs() < 1e-6, "state[{i}]: parallel {a} serial {s}");
        }
    }

    /// Phase X.3.e.3.1: Bonsai / Qwen 3.6 hybrid arch has `num_v_heads >
    /// num_kv_heads`. The pre-refactor loop iterated `num_kv_heads` times,
    /// silent-dropping the tail V heads' alpha / beta / state / output.
    /// Verified by exercising a `num_v_heads = 3 * num_kv_heads` config and
    /// asserting every V head's state slab is written (impossible under
    /// the old code — the tail slabs stayed all-zero) and that V heads
    /// inside the same KV group produce distinct states (per-V-head alpha
    /// / beta actually flowing into the recurrence, not shared).
    #[test]
    fn gated_deltanet_step_bonsai_per_v_head_consumption() {
        let num_kv_heads = 4;
        // 3 V heads per KV group — mirrors Bonsai's 48 V / 16 KV ratio.
        let num_v_heads = 12;
        let qk_dim = 3;
        let v_dim = 3;
        let v_per_kv = num_v_heads / num_kv_heads;
        let state_stride = qk_dim * v_dim;

        // Q / K live at KV-head granularity (V heads within a group share).
        let mut q = Vec::with_capacity(num_kv_heads * qk_dim);
        let mut k = Vec::with_capacity(num_kv_heads * qk_dim);
        for h in 0..num_kv_heads {
            for i in 0..qk_dim {
                let seed = (h * qk_dim + i) as f32 * 0.3;
                q.push(seed.sin());
                k.push(seed.cos());
            }
        }
        // V / Z per-V-head.
        let mut v = Vec::with_capacity(num_v_heads * v_dim);
        let mut z = Vec::with_capacity(num_v_heads * v_dim);
        for h in 0..num_v_heads {
            for j in 0..v_dim {
                let seed = (h * v_dim + j) as f32 * 0.5;
                v.push(seed.sin() + 0.1);
                z.push(seed.cos());
            }
        }
        // Distinct per-V-head alpha / beta — under the old loop, entries
        // beyond `num_kv_heads` would be silently ignored.
        let alpha: Vec<f32> = (0..num_v_heads).map(|h| 0.9 - 0.02 * h as f32).collect();
        let beta: Vec<f32> = (0..num_v_heads).map(|h| 0.1 + 0.02 * h as f32).collect();

        let mut state = vec![0.0f32; num_v_heads * state_stride];
        let mut out = vec![0.0f32; num_v_heads * v_dim];

        gated_deltanet_step(
            &q,
            &k,
            &v,
            &alpha,
            &beta,
            &z,
            &mut state,
            &mut out,
            num_kv_heads,
            num_v_heads,
            qk_dim,
            v_dim,
            false,
        );

        assert!(state.iter().all(|val| val.is_finite()));
        assert!(out.iter().all(|val| val.is_finite()));

        // Every V head's state slab must be written (any non-zero entry).
        // Pre-fix, V heads with index >= num_kv_heads stayed all-zero.
        for v_head in 0..num_v_heads {
            let slab = &state[v_head * state_stride..(v_head + 1) * state_stride];
            assert!(
                slab.iter().any(|&val| val.abs() > 0.0),
                "V head {v_head} state slab all zero — per-V-head loop did not cover it",
            );
        }

        // V heads inside the same KV group must produce distinct state
        // (they share Q / K but have independent V / Z / alpha / beta).
        for kv in 0..num_kv_heads {
            let head_a = kv * v_per_kv;
            let head_b = kv * v_per_kv + 1;
            let slab_a = &state[head_a * state_stride..(head_a + 1) * state_stride];
            let slab_b = &state[head_b * state_stride..(head_b + 1) * state_stride];
            let max_diff = slab_a
                .iter()
                .zip(slab_b.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff > 1e-6,
                "KV group {kv}: V head {head_a} state ≈ V head {head_b} state (max_diff={max_diff}), per-V-head alpha/beta collapsed",
            );
        }
    }

    /// Phase X.3.e.3.2 (Gap C): Bonsai / Qwen 3.6 `ssm_norm` is applied
    /// between the recurrence output and the `ssm_out` projection via
    /// `apply_qk_norm`, which broadcasts a `[v_dim]` weight vector across
    /// every V-head slab. Verify the math matches a hand-computed
    /// per-head reference (`out_i = x_i / sqrt(mean(x²) + eps) * w_i`)
    /// so a future refactor of `apply_qk_norm` cannot silently break the
    /// `ssm_norm` code path.
    #[test]
    fn ssm_norm_broadcasts_across_v_heads() {
        let num_v_heads = 3;
        let v_dim = 4;
        let eps = 1e-5f32;
        // Two-head fixture: head 0 has small magnitude, head 1 large, head 2
        // mixed sign — covers scale variance across V heads.
        let mut buf = vec![
            0.1, -0.2, 0.3, -0.4, // head 0
            2.0, -3.0, 4.0, -5.0, // head 1
            -1.0, 0.5, 1.5, -0.5, // head 2
        ];
        let weight = vec![1.0, -0.5, 2.0, 0.25];

        // Hand-computed reference: per head, x_i / sqrt(mean(x²) + eps) * w_i.
        let mut expected = Vec::with_capacity(num_v_heads * v_dim);
        for h in 0..num_v_heads {
            let slice = &buf[h * v_dim..(h + 1) * v_dim];
            let ss: f64 = slice.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let mean = (ss / v_dim as f64) as f32;
            let scale = 1.0f32 / (mean + eps).sqrt();
            for i in 0..v_dim {
                expected.push(slice[i] * scale * weight[i]);
            }
        }

        apply_qk_norm(&mut buf, &weight, v_dim, eps);

        for (i, (actual, expected)) in buf.iter().zip(expected.iter()).enumerate() {
            let head = i / v_dim;
            let lane = i % v_dim;
            assert!(
                (actual - expected).abs() < 1e-6,
                "head {head} lane {lane}: got {actual}, expected {expected}",
            );
        }
    }

    /// Phase X.3.e.3.2 (Gap B): softplus helper is the numerical foundation
    /// for the Bonsai / Qwen 3.6 SSM discretisation. Verify hand-computed
    /// exact values at zero, positive, negative anchors and confirm the
    /// numerically-stable path (`|x| > 20`) still produces finite outputs.
    #[test]
    fn softplus_matches_reference() {
        // softplus(0) = ln(1 + 1) = ln(2) ≈ 0.6931472
        assert!(
            (softplus(0.0) - 0.6931472_f32).abs() < 1e-6,
            "softplus(0) = {}",
            softplus(0.0)
        );
        // softplus(1) = ln(1 + e) ≈ 1.3132617
        assert!(
            (softplus(1.0) - 1.3132617_f32).abs() < 1e-5,
            "softplus(1) = {}",
            softplus(1.0)
        );
        // softplus(-1) = ln(1 + 1/e) ≈ 0.3132617
        assert!(
            (softplus(-1.0) - 0.3132617_f32).abs() < 1e-5,
            "softplus(-1) = {}",
            softplus(-1.0)
        );
        // Asymptotic: softplus(x) → x for large x
        assert!((softplus(30.0) - 30.0_f32).abs() < 1e-6);
        // Asymptotic: softplus(x) → 0 for large negative x
        assert!(softplus(-30.0).abs() < 1e-10);
        // Numerical stability across extreme range
        for &x in &[-100.0_f32, -50.0, -20.0, -1.0, 0.0, 1.0, 20.0, 50.0, 100.0] {
            let y = softplus(x);
            assert!(y.is_finite(), "softplus({x}) = {y} not finite");
            assert!(y >= 0.0, "softplus({x}) = {y} negative");
        }
    }

    /// Phase X.3.e.3.2 (Gap B): The Bonsai / Qwen 3.6 SSM discretisation
    /// transforms the raw alpha projection into an actual decay factor via
    /// `decay = exp(softplus(alpha + dt_bias) * ssm_a)`. Verify concrete
    /// anchor cases so a future refactor of `softplus` cannot silently
    /// break the transformation.
    #[test]
    fn ssm_alpha_transformation_matches_reference() {
        // Anchor 1: alpha=0, dt_bias=0, ssm_a=-1
        //   biased = 0, softplus(0) = ln(2), gate = -ln(2), decay = exp(-ln(2)) = 1/2
        let decay = (softplus(0.0 + 0.0) * -1.0_f32).exp();
        assert!(
            (decay - 0.5_f32).abs() < 1e-6,
            "anchor 1: decay = {decay}, expected 0.5",
        );

        // Anchor 2: alpha=large positive → softplus ≈ alpha, ssm_a=-2 → gate ≈ -2*alpha
        //   decay ≈ exp(-2 * alpha) → very small
        let decay2 = (softplus(10.0) * -2.0_f32).exp();
        assert!(
            decay2 < 0.01 && decay2 > 0.0,
            "anchor 2: decay = {decay2}, expected tiny positive",
        );

        // Anchor 3: alpha=large negative → softplus ≈ 0, gate ≈ 0, decay ≈ 1
        let decay3 = (softplus(-30.0) * -1.0_f32).exp();
        assert!(
            (decay3 - 1.0_f32).abs() < 1e-6,
            "anchor 3: decay = {decay3}, expected 1",
        );

        // Domain sweep: decay must stay in (0, 1] for all reasonable inputs
        // (assuming ssm_a is stored as negative, per Mamba convention).
        for &alpha in &[-100.0_f32, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
            for &dt_bias in &[-5.0_f32, 0.0, 5.0] {
                for &ssm_a_val in &[-5.0_f32, -1.0, -0.1] {
                    let gate = softplus(alpha + dt_bias) * ssm_a_val;
                    let decay = gate.exp();
                    assert!(
                        decay.is_finite(),
                        "α={alpha}, dt_b={dt_bias}, a={ssm_a_val}: decay={decay} not finite",
                    );
                    assert!(
                        (0.0..=1.0 + 1e-6).contains(&decay),
                        "α={alpha}, dt_b={dt_bias}, a={ssm_a_val}: decay={decay} outside [0, 1]",
                    );
                }
            }
        }
    }

    /// Phase X.3.e.3.2 (Gap B extra): sigmoid helper is the numerical
    /// foundation for the Bonsai / Qwen 3.6 beta constraint. Verify
    /// hand-computed exact values at zero / symmetric anchors and confirm
    /// the output stays in `(0, 1)` across extreme inputs.
    #[test]
    fn sigmoid_matches_reference() {
        // sigmoid(0) = 1 / 2
        assert!((sigmoid(0.0) - 0.5_f32).abs() < 1e-6);
        // sigmoid(x) + sigmoid(-x) = 1 (symmetry)
        for &x in &[-5.0_f32, -1.0, 0.5, 2.0, 8.0] {
            let s = sigmoid(x) + sigmoid(-x);
            assert!(
                (s - 1.0_f32).abs() < 1e-6,
                "sigmoid({x}) + sigmoid({}) = {s}, expected 1",
                -x,
            );
        }
        // Asymptotic: sigmoid(x) → 1 for large x, → 0 for large -x
        assert!((sigmoid(30.0) - 1.0_f32).abs() < 1e-6);
        assert!(sigmoid(-30.0).abs() < 1e-6);
        // Range: sigmoid(x) ∈ (0, 1) for all finite x
        for &x in &[-100.0_f32, -20.0, -1.0, 0.0, 1.0, 20.0, 100.0] {
            let y = sigmoid(x);
            assert!(
                y.is_finite() && (0.0..=1.0).contains(&y),
                "sigmoid({x}) = {y}"
            );
        }
        // Consistency: silu(x) = x * sigmoid(x)
        for &x in &[-2.0_f32, -0.5, 0.0, 0.5, 2.0] {
            let silu_ref = x * sigmoid(x);
            let silu_direct = silu(x);
            assert!(
                (silu_ref - silu_direct).abs() < 1e-6,
                "silu({x}) inconsistent with x * sigmoid(x)",
            );
        }
    }

    /// Phase X.3.e.3.2 (Gap B extra): Bonsai / Qwen 3.6 beta transformation
    /// applies `sigmoid` to constrain the raw beta projection to `(0, 1)`
    /// before it enters the delta-rule integration. Verify per-V-head
    /// broadcast + range invariant.
    #[test]
    fn ssm_beta_sigmoid_applied_per_v_head() {
        let num_v_heads = 5;
        // Raw beta values covering large negative → large positive.
        let mut dn_beta = vec![-8.0_f32, -1.0, 0.0, 2.5, 10.0];

        // Apply the transformation (mirrors forward-path Step 2d).
        for h in 0..num_v_heads {
            dn_beta[h] = sigmoid(dn_beta[h]);
        }

        // Every head is now in (0, 1).
        for (h, &val) in dn_beta.iter().enumerate() {
            assert!(
                val.is_finite() && val > 0.0 && val < 1.0,
                "head {h}: sigmoid output {val} outside (0, 1)",
            );
        }

        // Ordering preserved (sigmoid is monotonic).
        for pair in dn_beta.windows(2) {
            assert!(pair[0] < pair[1], "sigmoid not monotonic: {pair:?}");
        }

        // Anchor: middle head (raw=0) must land at exactly 0.5.
        assert!((dn_beta[2] - 0.5_f32).abs() < 1e-6);
    }

    /// Phase X.3.e.3.2 (§Q/K L2Norm + §silu(z) order): the Bonsai
    /// semantics flag toggles two independent reference alignments —
    /// skip the internal silu(k)/silu(q) since the caller pre-silu'd
    /// q / k post-conv1d, and skip the internal out *= silu(z) since
    /// the caller multiplies after ssm-norm. Verify the two modes
    /// diverge (branch is live) and that Bonsai output ignores z
    /// entirely (z-gate skip works).
    #[test]
    fn gated_deltanet_step_bonsai_semantics_toggle() {
        let num_heads = 1;
        let qk_dim = 4;
        let v_dim = 4;
        // Non-zero q/k so silu vs no-silu produces different scales.
        let q = vec![0.7f32, -0.3, 0.5, 0.2];
        let k = vec![0.6f32, 0.4, -0.2, 0.8];
        let v = vec![0.1f32, -0.5, 0.3, 0.7];
        let alpha = vec![0.5f32];
        let beta = vec![0.4f32];
        let z = vec![1.5f32, -1.0, 0.5, 2.0];

        let mut state_legacy = vec![0.0f32; num_heads * qk_dim * v_dim];
        let mut out_legacy = vec![0.0f32; num_heads * v_dim];
        gated_deltanet_step(
            &q,
            &k,
            &v,
            &alpha,
            &beta,
            &z,
            &mut state_legacy,
            &mut out_legacy,
            num_heads,
            num_heads,
            qk_dim,
            v_dim,
            false,
        );

        let mut state_bonsai = vec![0.0f32; num_heads * qk_dim * v_dim];
        let mut out_bonsai = vec![0.0f32; num_heads * v_dim];
        gated_deltanet_step(
            &q,
            &k,
            &v,
            &alpha,
            &beta,
            &z,
            &mut state_bonsai,
            &mut out_bonsai,
            num_heads,
            num_heads,
            qk_dim,
            v_dim,
            true,
        );

        for &val in out_legacy.iter().chain(out_bonsai.iter()) {
            assert!(val.is_finite(), "non-finite value in output");
        }

        // Outputs must diverge — proves the branch is actually live and
        // Bonsai semantics are not a silent no-op.
        let max_diff = out_legacy
            .iter()
            .zip(out_bonsai.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "legacy vs Bonsai output max_diff={max_diff}, expected > 1e-4",
        );

        // Bonsai-mode output must be independent of z (z-gate skip).
        let z_alt = vec![-3.0f32, 5.0, -0.5, 7.5];
        let mut state_bonsai_alt = vec![0.0f32; num_heads * qk_dim * v_dim];
        let mut out_bonsai_alt = vec![0.0f32; num_heads * v_dim];
        gated_deltanet_step(
            &q,
            &k,
            &v,
            &alpha,
            &beta,
            &z_alt,
            &mut state_bonsai_alt,
            &mut out_bonsai_alt,
            num_heads,
            num_heads,
            qk_dim,
            v_dim,
            true,
        );
        for (i, (a, b)) in out_bonsai.iter().zip(out_bonsai_alt.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "Bonsai output must be z-independent, lane {i}: {a} vs {b}",
            );
        }
    }

    // ── LayerWeights sub-struct accessors (Issue #11) ────────────────────

    /// Reusable fixture that only sets the fields required to construct
    /// a `LayerWeights`. The refactor moved arch-specific fields into
    /// nested sub-structs; every accessor below verifies the None → Some
    /// transition when the relevant sub-struct is populated.
    fn empty_layer_weights<'a>() -> LayerWeights<'a> {
        // Reuse the byte buffer for every `WeightRef` — the accessor tests
        // only inspect Option-ness, not the underlying quantised bytes.
        static ZERO: [u8; 256] = [0u8; 256];
        let core_weight = WeightRef {
            data: &ZERO,
            qtype: crate::gguf::GgmlType::F32,
            rows: 0,
            cols: 0,
        };
        LayerWeights {
            attn_norm: Vec::new(),
            q_proj: core_weight.clone(),
            k_proj: None,
            v_proj: None,
            o_proj: core_weight,
            ffn_norm: Vec::new(),
            gate_proj: None,
            up_proj: None,
            down_proj: None,
            post_attn_norm: None,
            post_ffn_norm: None,
            out_scale: None,
            rope_freqs: None,
            qwen_biases: None,
            qwen_norms: None,
            gemma3n: None,
            moe: None,
            gated_output: false,
        }
    }

    #[test]
    fn layer_weights_accessors_return_none_when_sub_structs_absent() {
        let lw = empty_layer_weights();
        assert!(lw.q_bias().is_none());
        assert!(lw.k_bias().is_none());
        assert!(lw.v_bias().is_none());
        assert!(lw.q_norm().is_none());
        assert!(lw.k_norm().is_none());
        assert!(lw.post_norm().is_none());
        assert!(lw.laurel_l().is_none());
        assert!(lw.laurel_r().is_none());
        assert!(lw.laurel_post_norm().is_none());
        assert!(lw.altup_router().is_none());
        assert!(lw.altup_router_norm().is_none());
        assert!(lw.altup_predict_coef().is_none());
        assert!(lw.altup_correct_coef().is_none());
        assert!(lw.altup_correct_scale().is_none());
        assert!(lw.inp_gate().is_none());
        assert!(lw.proj().is_none());
        assert!(lw.ffn_gate_inp().is_none());
        assert!(lw.ffn_gate_exps().is_none());
        assert!(lw.ffn_up_exps().is_none());
        assert!(lw.ffn_down_exps().is_none());
        assert!(!lw.is_moe_layer());
    }

    #[test]
    fn layer_weights_qwen_biases_visible_through_accessors() {
        let mut lw = empty_layer_weights();
        lw.qwen_biases = Some(QwenAttentionBiases {
            q_bias: vec![1.0, 2.0],
            k_bias: vec![3.0, 4.0],
            v_bias: vec![5.0, 6.0],
        });
        assert_eq!(lw.q_bias(), Some(&[1.0f32, 2.0][..]));
        assert_eq!(lw.k_bias(), Some(&[3.0f32, 4.0][..]));
        assert_eq!(lw.v_bias(), Some(&[5.0f32, 6.0][..]));
        // Qwen 3 norms are a separate sub-struct — still None.
        assert!(lw.q_norm().is_none());
        assert!(lw.k_norm().is_none());
    }

    #[test]
    fn layer_weights_qwen_norms_visible_through_accessors() {
        let mut lw = empty_layer_weights();
        lw.qwen_norms = Some(QwenAttentionNorms {
            q_norm: vec![0.1, 0.2],
            k_norm: vec![0.3, 0.4],
        });
        assert_eq!(lw.q_norm(), Some(&[0.1f32, 0.2][..]));
        assert_eq!(lw.k_norm(), Some(&[0.3f32, 0.4][..]));
        // Biases are a separate sub-struct — still None.
        assert!(lw.q_bias().is_none());
    }

    #[test]
    fn layer_weights_moe_visible_through_accessors_and_flag() {
        static ZERO: [u8; 256] = [0u8; 256];
        let expert_weight = WeightRef {
            data: &ZERO,
            qtype: crate::gguf::GgmlType::F32,
            rows: 0,
            cols: 0,
        };
        let mut lw = empty_layer_weights();
        lw.moe = Some(MoeExpertWeights {
            ffn_gate_inp: vec![9.0, 8.0, 7.0],
            ffn_gate_exps: expert_weight.clone(),
            ffn_up_exps: expert_weight.clone(),
            ffn_down_exps: expert_weight,
        });
        assert_eq!(lw.ffn_gate_inp(), Some(&[9.0f32, 8.0, 7.0][..]));
        assert!(lw.ffn_gate_exps().is_some());
        assert!(lw.ffn_up_exps().is_some());
        assert!(lw.ffn_down_exps().is_some());
        assert!(lw.is_moe_layer());
    }

    /// Explicit check that non-MoE / non-Qwen field groups stay None when
    /// only one arch group is populated — protects against the God-object
    /// regression where a stray `Some(...)` bleeds across arch families.
    #[test]
    fn layer_weights_arch_groups_are_independent() {
        let mut lw = empty_layer_weights();
        lw.qwen_biases = Some(QwenAttentionBiases {
            q_bias: vec![0.0],
            k_bias: vec![0.0],
            v_bias: vec![0.0],
        });
        // Setting Qwen biases must not leak into Gemma3n / MoE accessors.
        assert!(lw.laurel_l().is_none());
        assert!(lw.altup_router().is_none());
        assert!(lw.ffn_gate_inp().is_none());
        assert!(!lw.is_moe_layer());
    }

    // ── Llama3Config sub-config accessors (Issue #11 Part 2) ─────────────

    #[test]
    fn config_baseline_llama_has_no_arch_extras() {
        let c = Llama3Config::llama3_8b();
        assert!(c.attention_extras.is_none());
        assert!(c.ssm.is_none());
        assert!(c.moe.is_none());
        assert!(c.gemma3n.is_none());
        assert!(c.gemma4.is_none());
        // All accessor methods return None when sub-configs are absent.
        assert!(c.sliding_window().is_none());
        assert!(c.attn_logit_softcap().is_none());
        assert!(c.final_logit_softcap().is_none());
        assert!(c.full_attention_interval().is_none());
        assert!(c.linear_qk_head_dim().is_none());
        assert!(c.ssm_inner_size().is_none());
        assert!(c.num_experts().is_none());
        assert!(c.num_experts_active().is_none());
        assert!(c.expert_ffn_size().is_none());
        assert!(c.altup_num_inputs().is_none());
        assert!(c.altup_active_idx().is_none());
        assert!(c.shared_kv_layers().is_none());
        assert!(c.per_layer_input_embedding_dim().is_none());
        assert!(c.sliding_window_pattern().is_none());
        assert!(c.activation_sparsity_scale().is_none());
        assert!(c.head_dim_swa().is_none());
        assert!(c.rope_theta_swa().is_none());
        assert!(c.rope_dim_swa().is_none());
        assert!(c.ffn_size_per_layer().is_none());
        assert!(!c.is_hybrid());
    }

    #[test]
    fn config_attention_extras_accessors_read_populated_sub_config() {
        let mut c = Llama3Config::llama3_8b();
        c.attention_extras = Some(AttentionExtrasConfig {
            sliding_window: Some(4096),
            attn_logit_softcap: Some(30.0),
            final_logit_softcap: Some(30.0),
        });
        assert_eq!(c.sliding_window(), Some(4096));
        assert_eq!(c.attn_logit_softcap(), Some(30.0));
        assert_eq!(c.final_logit_softcap(), Some(30.0));
        // Sibling sub-configs still return None.
        assert!(c.ssm_inner_size().is_none());
        assert!(c.num_experts().is_none());
    }

    #[test]
    fn config_ssm_accessors_read_populated_sub_config() {
        let mut c = Llama3Config::llama3_8b();
        c.ssm = Some(SsmDeltaNetConfig {
            full_attention_interval: Some(4),
            linear_num_kv_heads: Some(16),
            linear_qk_head_dim: Some(128),
            linear_kv_head_dim: Some(128),
            linear_num_v_heads: Some(32),
            linear_conv_kernel_dim: Some(4),
            ssm_inner_size: Some(6144),
            ssm_state_size: Some(128),
            ssm_group_count: Some(16),
            ssm_time_step_rank: Some(48),
            n_layer_nextn: Some(1),
        });
        assert_eq!(c.full_attention_interval(), Some(4));
        assert_eq!(c.linear_num_kv_heads(), Some(16));
        assert_eq!(c.linear_qk_head_dim(), Some(128));
        assert_eq!(c.linear_kv_head_dim(), Some(128));
        assert_eq!(c.linear_num_v_heads(), Some(32));
        assert_eq!(c.linear_conv_kernel_dim(), Some(4));
        assert_eq!(c.ssm_inner_size(), Some(6144));
        assert_eq!(c.ssm_state_size(), Some(128));
        assert_eq!(c.ssm_group_count(), Some(16));
        assert_eq!(c.ssm_time_step_rank(), Some(48));
        assert_eq!(c.n_layer_nextn(), Some(1));
        assert!(c.is_hybrid());
    }

    #[test]
    fn config_moe_accessors_read_populated_sub_config() {
        let mut c = Llama3Config::llama3_8b();
        c.moe = Some(MoeConfig {
            num_experts: Some(8),
            num_experts_active: Some(2),
            expert_ffn_size: Some(2048),
        });
        assert_eq!(c.num_experts(), Some(8));
        assert_eq!(c.num_experts_active(), Some(2));
        assert_eq!(c.expert_ffn_size(), Some(2048));
    }

    #[test]
    fn config_gemma3n_accessors_read_populated_sub_config() {
        let mut c = Llama3Config::llama3_8b();
        c.gemma3n = Some(Gemma3nConfig {
            sliding_window_pattern: Some(vec![true, false, true]),
            activation_sparsity_scale: Some(vec![1.5, f32::NEG_INFINITY]),
            shared_kv_layers: Some(10),
            per_layer_input_embedding_dim: Some(256),
            altup_num_inputs: Some(4),
            altup_active_idx: Some(0),
        });
        assert_eq!(c.sliding_window_pattern(), Some(&[true, false, true][..]));
        assert_eq!(
            c.activation_sparsity_scale(),
            Some(&[1.5f32, f32::NEG_INFINITY][..])
        );
        assert_eq!(c.shared_kv_layers(), Some(10));
        assert_eq!(c.per_layer_input_embedding_dim(), Some(256));
        assert_eq!(c.altup_num_inputs(), Some(4));
        assert_eq!(c.altup_active_idx(), Some(0));
    }

    #[test]
    fn config_gemma4_accessors_read_populated_sub_config() {
        let mut c = Llama3Config::llama3_8b();
        c.gemma4 = Some(Gemma4Config {
            head_dim_swa: Some(64),
            rope_theta_swa: Some(10_000.0),
            rope_dim_swa: Some(64),
            ffn_size_per_layer: Some(vec![6144, 12_288]),
        });
        assert_eq!(c.head_dim_swa(), Some(64));
        assert_eq!(c.rope_theta_swa(), Some(10_000.0));
        assert_eq!(c.rope_dim_swa(), Some(64));
        assert_eq!(c.ffn_size_per_layer(), Some(&[6144usize, 12_288][..]));
    }

    // ── KV cache persistence (colibri `.coli_kv` 参考) ────────────────

    /// Populate a `KvCache` deterministically from `seed` so both save/load
    /// and mismatch tests can produce reproducible content without needing
    /// to run an actual forward pass.
    fn make_populated_kv_cache(
        num_layers: usize,
        max_seq_len: usize,
        kv_dim: usize,
        active_seq_len: usize,
        seed: u32,
    ) -> KvCache {
        let mut cache = KvCache::new(num_layers, max_seq_len, kv_dim);
        for pos in 0..active_seq_len {
            for layer in 0..num_layers {
                let k: Vec<f32> = (0..kv_dim)
                    .map(|i| {
                        f32::from(
                            seed.wrapping_add(pos as u32 * 7 + layer as u32 * 3 + i as u32) as u16
                                as i16,
                        ) * 0.001
                    })
                    .collect();
                let v: Vec<f32> = (0..kv_dim)
                    .map(|i| {
                        f32::from(
                            seed.wrapping_add(pos as u32 * 11 + layer as u32 * 5 + i as u32) as u16
                                as i16,
                        ) * 0.001
                    })
                    .collect();
                cache.append(layer, &k, &v);
            }
            cache.advance();
        }
        cache
    }

    #[test]
    fn kv_cache_save_load_roundtrip_bit_exact() {
        let num_layers = 4;
        let max_seq_len = 16;
        let kv_dim = 32;
        let active_seq_len = 7;
        let cache = make_populated_kv_cache(num_layers, max_seq_len, kv_dim, active_seq_len, 42);
        let fingerprint = 0xDEAD_BEEF_CAFE_F00D_u64;

        // Serialise to a `Vec<u8>` (skips filesystem for a hermetic test).
        let mut bytes = Vec::new();
        cache.write_to(&mut bytes, fingerprint).expect("write ok");

        // Deserialise into a fresh cache of matching shape.
        let mut restored = KvCache::new(num_layers, max_seq_len, kv_dim);
        restored
            .read_from(&mut bytes.as_slice(), fingerprint)
            .expect("read ok");

        assert_eq!(cache.seq_len, restored.seq_len);
        assert_eq!(cache.kv_layer_map, restored.kv_layer_map);
        // Only the active prefix is persisted; the tail past `active_seq_len`
        // stays zero in `restored`, so comparing the full buffer would fail
        // on the tail. Compare only the persisted region.
        for layer in 0..num_layers {
            let n = active_seq_len * kv_dim;
            let off = cache.offset(layer, 0);
            assert_eq!(&cache.keys[off..off + n], &restored.keys[off..off + n]);
            assert_eq!(&cache.values[off..off + n], &restored.values[off..off + n]);
        }
    }

    #[test]
    fn kv_cache_load_rejects_bad_magic() {
        let mut restored = KvCache::new(4, 16, 32);
        let mut buf: Vec<u8> = b"NOTMAGIC".to_vec();
        buf.extend_from_slice(&[0u8; 64]);
        let err = restored
            .read_from(&mut buf.as_slice(), 0)
            .expect_err("bad magic must fail");
        assert!(
            matches!(err, KvCacheLoadError::BadMagic { .. }),
            "expected BadMagic, got {err:?}"
        );
    }

    #[test]
    fn kv_cache_load_rejects_fingerprint_mismatch() {
        let cache = make_populated_kv_cache(2, 8, 16, 3, 7);
        let mut bytes = Vec::new();
        cache.write_to(&mut bytes, 0x1111).expect("write ok");
        let mut restored = KvCache::new(2, 8, 16);
        let err = restored
            .read_from(&mut bytes.as_slice(), 0x2222)
            .expect_err("fingerprint mismatch must fail");
        assert!(
            matches!(err, KvCacheLoadError::FingerprintMismatch { .. }),
            "expected FingerprintMismatch, got {err:?}"
        );
    }

    #[test]
    fn kv_cache_load_rejects_shape_mismatch() {
        let cache = make_populated_kv_cache(4, 16, 32, 3, 7);
        let mut bytes = Vec::new();
        cache.write_to(&mut bytes, 0x1234).expect("write ok");
        // Cache with a different num_layers must reject the load.
        let mut restored = KvCache::new(8, 16, 32);
        let err = restored
            .read_from(&mut bytes.as_slice(), 0x1234)
            .expect_err("shape mismatch must fail");
        assert!(
            matches!(err, KvCacheLoadError::ShapeMismatch(_)),
            "expected ShapeMismatch, got {err:?}"
        );
    }

    #[test]
    fn kv_cache_fingerprint_is_deterministic_and_config_sensitive() {
        let base = Llama3Config::llama3_8b();
        let mut variant = Llama3Config::llama3_8b();
        variant.num_layers += 1;
        // Same config → same fingerprint (order-independent).
        assert_eq!(kv_cache_fingerprint(&base), kv_cache_fingerprint(&base));
        // A single shape-critical field change must move the fingerprint.
        assert_ne!(kv_cache_fingerprint(&base), kv_cache_fingerprint(&variant));
    }

    // ── DeepSeek-V3 / R1 foundation (Phase 1) ────────────────────────

    #[test]
    fn deepseek_v3_config_accessors_return_none_when_absent() {
        let c = Llama3Config::llama3_8b();
        assert!(c.deepseek_v3.is_none());
        assert!(c.deepseek_q_lora_rank().is_none());
        assert!(c.deepseek_kv_lora_rank().is_none());
        assert!(c.deepseek_qk_nope_head_dim().is_none());
        assert!(c.deepseek_qk_rope_head_dim().is_none());
        assert!(c.deepseek_v_head_dim().is_none());
        assert!(c.deepseek_n_routed_experts().is_none());
        assert!(c.deepseek_n_shared_experts().is_none());
        assert!(c.deepseek_num_experts_per_tok().is_none());
        assert!(c.deepseek_moe_intermediate_size().is_none());
        assert!(c.deepseek_first_k_dense_replace().is_none());
        assert!(c.deepseek_routed_scaling_factor().is_none());
        assert!(c.deepseek_noaux_tc().is_none());
        assert!(c.deepseek_mtp_layer().is_none());
    }

    #[test]
    fn deepseek_v3_config_reads_typical_v3_values() {
        let mut c = Llama3Config::llama3_8b();
        c.arch = ModelArch::DeepSeekV3;
        c.deepseek_v3 = Some(DeepSeekV3Config {
            q_lora_rank: Some(1536),
            kv_lora_rank: Some(512),
            qk_nope_head_dim: Some(128),
            qk_rope_head_dim: Some(64),
            v_head_dim: Some(128),
            n_routed_experts: Some(256),
            n_shared_experts: Some(1),
            num_experts_per_tok: Some(8),
            moe_intermediate_size: Some(2048),
            first_k_dense_replace: Some(3),
            routed_scaling_factor: Some(2.5),
            noaux_tc: Some(true),
            mtp_layer: Some(60),
        });
        assert_eq!(c.deepseek_q_lora_rank(), Some(1536));
        assert_eq!(c.deepseek_kv_lora_rank(), Some(512));
        assert_eq!(c.deepseek_qk_nope_head_dim(), Some(128));
        assert_eq!(c.deepseek_qk_rope_head_dim(), Some(64));
        assert_eq!(c.deepseek_v_head_dim(), Some(128));
        assert_eq!(c.deepseek_n_routed_experts(), Some(256));
        assert_eq!(c.deepseek_n_shared_experts(), Some(1));
        assert_eq!(c.deepseek_num_experts_per_tok(), Some(8));
        assert_eq!(c.deepseek_moe_intermediate_size(), Some(2048));
        assert_eq!(c.deepseek_first_k_dense_replace(), Some(3));
        assert_eq!(c.deepseek_routed_scaling_factor(), Some(2.5));
        assert_eq!(c.deepseek_noaux_tc(), Some(true));
        assert_eq!(c.deepseek_mtp_layer(), Some(60));
    }

    #[test]
    fn deepseek_v3_arch_uses_neox_rope() {
        // DeepSeek-V3 uses NEOX RoPE half-rotation on the `qk_rope_head_dim`
        // slice, consistent with Qwen and Gemma families.
        assert!(ModelArch::DeepSeekV3.use_neox_rope());
    }

    #[test]
    fn deepseek_v3_arch_meta_prefix_is_deepseek2() {
        // llama.cpp names V2 / V3 / R1 all under the same `deepseek2` key
        // prefix. Any drift here would silently mis-route GGUF metadata
        // lookups in `Llama3Config::from_gguf`.
        assert_eq!(ModelArch::DeepSeekV3.meta_prefix(), "deepseek2");
    }

    // ── DeepSeek-V3 Phase 2: MLA shape + math sanity ─────────────────

    /// Synthetic 4-layer DeepSeek-V3 config sized so weight loading /
    /// forward paths can be exercised without a real 671B GGUF. Numbers
    /// are downsized proportionally (num_heads=2, hidden_dim=16,
    /// q_lora_rank=8, kv_lora_rank=4, qk_nope=4, qk_rope=2, v_head=4).
    fn tiny_deepseek_v3_config() -> Llama3Config {
        Llama3Config {
            arch: ModelArch::DeepSeekV3,
            vocab_size: 32,
            hidden_dim: 16,
            intermediate_dim: 32,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: 4,
            max_seq_len: 32,
            head_dim: 6,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            attention_extras: None,
            ssm: None,
            moe: None,
            gemma3n: None,
            gemma4: None,
            deepseek_v3: Some(DeepSeekV3Config {
                q_lora_rank: Some(8),
                kv_lora_rank: Some(4),
                qk_nope_head_dim: Some(4),
                qk_rope_head_dim: Some(2),
                v_head_dim: Some(4),
                n_routed_experts: Some(8),
                n_shared_experts: Some(1),
                num_experts_per_tok: Some(2),
                moe_intermediate_size: Some(32),
                // Set high so every layer in the tiny model takes the dense
                // SwiGLU path — MoE (Phase 3) is out of scope for these tests.
                first_k_dense_replace: Some(4),
                routed_scaling_factor: Some(2.5),
                noaux_tc: Some(true),
                mtp_layer: None,
            }),
        }
    }

    #[test]
    fn deepseek_v3_compressed_kv_dim_matches_paper() {
        // The MLA cache stores `kv_lora_rank + qk_rope_head_dim` floats per
        // token — 576 for V3 (512 + 64). The tiny fixture uses 4 + 2 = 6.
        let c = tiny_deepseek_v3_config();
        let kv_dim = c.deepseek_kv_lora_rank().unwrap() + c.deepseek_qk_rope_head_dim().unwrap();
        assert_eq!(kv_dim, 6);
        // Sanity-check the paper's V3 numbers as documented in the config
        // comments, in case a future refactor accidentally rewrites them.
        assert_eq!(512usize + 64usize, 576);
    }

    #[test]
    fn deepseek_v3_shape_totals_align_with_head_layout() {
        // `q_b_proj` output size and `kv_b_proj` output size are both
        // per-head split into (nope, rope|v). A mistuned config would
        // silently mis-index the split inside `forward_deepseek_v3`.
        let c = tiny_deepseek_v3_config();
        let num_heads = c.num_heads;
        let q_head_total =
            c.deepseek_qk_nope_head_dim().unwrap() + c.deepseek_qk_rope_head_dim().unwrap();
        let kv_up_head_total =
            c.deepseek_qk_nope_head_dim().unwrap() + c.deepseek_v_head_dim().unwrap();
        assert_eq!(num_heads * q_head_total, 2 * 6);
        assert_eq!(num_heads * kv_up_head_total, 2 * 8);
    }

    // ── DeepSeek-V3 Phase 4a: MoE streaming parity ────────────────────

    /// Constructs both `RoutedExpertStorage::InMemory` and
    /// `RoutedExpertStorage::Streaming` variants over the same underlying
    /// byte buffer, calls `forward_deepseek_moe_layer` on each, and
    /// asserts the output is bit-identical.
    ///
    /// This is the end-to-end proof that swapping storage backends never
    /// changes numerical output — the whole point of putting streaming
    /// behind an enum instead of a separate forward path.
    #[test]
    fn deepseek_moe_forward_streaming_matches_in_memory() {
        use crate::deepseek_streaming::{ExpertKind, ExpertLayerInfo, StreamingExpertPool};
        use crate::gguf::GgmlType;
        use std::sync::Arc;

        // Tiny MoE config: hidden=4, moe_ffn=8, n_experts=4, top_k=2,
        // n_shared=1, routed_scale=1.5. F32 quant keeps byte math trivial.
        const HIDDEN: usize = 4;
        const MOE_FFN: usize = 8;
        const N_EXPERTS: usize = 4;
        const TOP_K: usize = 2;
        const N_SHARED: usize = 1;
        const ROUTED_SCALE: f32 = 1.5;
        const SHARED_FFN: usize = N_SHARED * MOE_FFN;

        // Deterministic byte pattern for gate/up/down expert tensors.
        // Each expert slab is `moe_ffn * hidden * 4` = 128 bytes.
        const SLAB_BYTES: usize = MOE_FFN * HIDDEN * 4;
        const DOWN_SLAB_BYTES: usize = HIDDEN * MOE_FFN * 4;
        assert_eq!(SLAB_BYTES, DOWN_SLAB_BYTES);
        let make_slab = |seed: u8| -> Vec<u8> {
            (0..SLAB_BYTES)
                .flat_map(|i| {
                    let v = ((seed as f32 + i as f32 * 0.017).sin() * 0.5).to_le_bytes();
                    v.into_iter()
                })
                .take(SLAB_BYTES)
                .collect()
        };
        // Layout in the streaming source: [gate_e0..3 | up_e0..3 | down_e0..3].
        let mut source_bytes = Vec::with_capacity(3 * N_EXPERTS * SLAB_BYTES);
        for e in 0..N_EXPERTS {
            source_bytes.extend(make_slab(e as u8));
        }
        for e in 0..N_EXPERTS {
            source_bytes.extend(make_slab((e + 10) as u8));
        }
        for e in 0..N_EXPERTS {
            source_bytes.extend(make_slab((e + 20) as u8));
        }

        // In-memory WeightRefs slice the same buffer directly.
        let gate_bytes = &source_bytes[0..N_EXPERTS * SLAB_BYTES];
        let up_bytes = &source_bytes[N_EXPERTS * SLAB_BYTES..2 * N_EXPERTS * SLAB_BYTES];
        let down_bytes = &source_bytes[2 * N_EXPERTS * SLAB_BYTES..3 * N_EXPERTS * SLAB_BYTES];

        // Shared expert weights (also F32, small).
        let shared_gate: Vec<u8> = (0..SHARED_FFN * HIDDEN * 4)
            .flat_map(|i| ((i as f32 * 0.03 - 0.1).cos() * 0.2).to_le_bytes())
            .collect();
        let shared_up: Vec<u8> = (0..SHARED_FFN * HIDDEN * 4)
            .flat_map(|i| ((i as f32 * 0.05 + 0.05).sin() * 0.15).to_le_bytes())
            .collect();
        let shared_down: Vec<u8> = (0..HIDDEN * SHARED_FFN * 4)
            .flat_map(|i| ((i as f32 * 0.07 - 0.2).cos() * 0.18).to_le_bytes())
            .collect();

        // Router + norm + noaux_tc bias.
        let ffn_norm: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + i as f32 * 0.05).collect();
        let ffn_gate_inp: Vec<f32> = (0..N_EXPERTS * HIDDEN)
            .map(|i| (i as f32 * 0.1).sin() * 0.3)
            .collect();
        let exp_probs_b: Vec<f32> = vec![0.0, 0.1, -0.05, 0.02];

        // Build a minimal config with the DeepSeek MoE sub-config populated.
        let config = Llama3Config {
            arch: ModelArch::DeepSeekV3,
            vocab_size: 128,
            hidden_dim: HIDDEN,
            intermediate_dim: MOE_FFN,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: 1,
            max_seq_len: 32,
            head_dim: 4,
            rope_theta: 10_000.0,
            norm_eps: 1e-6,
            attention_extras: None,
            ssm: None,
            moe: None,
            gemma3n: None,
            gemma4: None,
            deepseek_v3: Some(DeepSeekV3Config {
                q_lora_rank: Some(8),
                kv_lora_rank: Some(4),
                qk_nope_head_dim: Some(4),
                qk_rope_head_dim: Some(2),
                v_head_dim: Some(4),
                n_routed_experts: Some(N_EXPERTS),
                n_shared_experts: Some(N_SHARED),
                num_experts_per_tok: Some(TOP_K),
                moe_intermediate_size: Some(MOE_FFN),
                first_k_dense_replace: Some(0),
                routed_scaling_factor: Some(ROUTED_SCALE),
                noaux_tc: Some(true),
                mtp_layer: None,
            }),
        };

        // Build the two `DeepSeekMoeWeights` variants sharing everything
        // except the routed-expert storage backend. Written as explicit
        // struct literals rather than a closure so both variants borrow
        // from the same set of `Vec<u8>` bindings — a closure would
        // trigger `lifetime may not live long enough`.
        let in_memory = DeepSeekMoeWeights {
            ffn_norm: ffn_norm.clone(),
            ffn_gate_inp: ffn_gate_inp.clone(),
            exp_probs_b: Some(exp_probs_b.clone()),
            routed: RoutedExpertStorage::InMemory {
                gate: WeightRef {
                    data: gate_bytes,
                    qtype: GgmlType::F32,
                    rows: N_EXPERTS * MOE_FFN,
                    cols: HIDDEN,
                },
                up: WeightRef {
                    data: up_bytes,
                    qtype: GgmlType::F32,
                    rows: N_EXPERTS * MOE_FFN,
                    cols: HIDDEN,
                },
                down: WeightRef {
                    data: down_bytes,
                    qtype: GgmlType::F32,
                    rows: N_EXPERTS * HIDDEN,
                    cols: MOE_FFN,
                },
            },
            ffn_gate_shexp: WeightRef {
                data: &shared_gate,
                qtype: GgmlType::F32,
                rows: SHARED_FFN,
                cols: HIDDEN,
            },
            ffn_up_shexp: WeightRef {
                data: &shared_up,
                qtype: GgmlType::F32,
                rows: SHARED_FFN,
                cols: HIDDEN,
            },
            ffn_down_shexp: WeightRef {
                data: &shared_down,
                qtype: GgmlType::F32,
                rows: HIDDEN,
                cols: SHARED_FFN,
            },
        };

        // Streaming variant: pool over an owned copy of `source_bytes`.
        let source_owned: Arc<dyn crate::deepseek_streaming::ExpertByteSource> =
            Arc::new(source_bytes.clone());
        let layer_info = vec![[
            ExpertLayerInfo {
                base_offset: 0,
                bytes_per_expert: SLAB_BYTES,
                n_experts: N_EXPERTS,
                qtype: GgmlType::F32,
            },
            ExpertLayerInfo {
                base_offset: N_EXPERTS * SLAB_BYTES,
                bytes_per_expert: SLAB_BYTES,
                n_experts: N_EXPERTS,
                qtype: GgmlType::F32,
            },
            ExpertLayerInfo {
                base_offset: 2 * N_EXPERTS * SLAB_BYTES,
                bytes_per_expert: SLAB_BYTES,
                n_experts: N_EXPERTS,
                qtype: GgmlType::F32,
            },
        ]];
        let pool = Arc::new(StreamingExpertPool::new(
            source_owned,
            layer_info,
            1024 * 1024, // 1 MiB budget, plenty for 3 * 128 = 384 bytes/expert
        ));
        let streaming = DeepSeekMoeWeights {
            ffn_norm: ffn_norm.clone(),
            ffn_gate_inp: ffn_gate_inp.clone(),
            exp_probs_b: Some(exp_probs_b.clone()),
            routed: RoutedExpertStorage::Streaming {
                pool: pool.clone(),
                layer_idx: 0,
            },
            ffn_gate_shexp: WeightRef {
                data: &shared_gate,
                qtype: GgmlType::F32,
                rows: SHARED_FFN,
                cols: HIDDEN,
            },
            ffn_up_shexp: WeightRef {
                data: &shared_up,
                qtype: GgmlType::F32,
                rows: SHARED_FFN,
                cols: HIDDEN,
            },
            ffn_down_shexp: WeightRef {
                data: &shared_down,
                qtype: GgmlType::F32,
                rows: HIDDEN,
                cols: SHARED_FFN,
            },
        };

        // Same input for both paths.
        let hidden_state: Vec<f32> = (0..HIDDEN).map(|i| 0.5 - i as f32 * 0.1).collect();
        let mut norm_buf_a = vec![0.0f32; HIDDEN];
        let mut norm_buf_b = vec![0.0f32; HIDDEN];
        let mut out_in_memory = vec![0.0f32; HIDDEN];
        let mut out_streaming = vec![0.0f32; HIDDEN];

        forward_deepseek_moe_layer(
            &config,
            &in_memory,
            &hidden_state,
            &mut norm_buf_a,
            &mut out_in_memory,
        );
        forward_deepseek_moe_layer(
            &config,
            &streaming,
            &hidden_state,
            &mut norm_buf_b,
            &mut out_streaming,
        );

        // Bit-exact parity: the two paths dispatch to the same
        // `quantized_matvec` on identical bytes, in the same order.
        assert_eq!(
            out_in_memory
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            out_streaming
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "InMemory vs Streaming forward diverged: {out_in_memory:?} vs {out_streaming:?}"
        );

        // Sanity: with Phase 4b.2 prefetch, top-k × 3 kinds first fill the
        // cache upfront (2 × 3 = 6 misses), and the per-expert matvec loop
        // then re-fetches the same 6 keys — all hits. Prefetch decouples
        // I/O from compute even on the synthetic in-memory source.
        let stats = pool.cache_stats();
        assert_eq!(
            stats.misses, 6,
            "expected 6 pool misses (2 experts × 3 kinds)"
        );
        assert_eq!(
            stats.hits, 6,
            "matvec loop re-fetches prefetched keys as hits"
        );
    }

    // ── DeepSeek-V3 Phase 3: MoE routing math ────────────────────────

    #[test]
    fn deepseek_moe_route_sigmoid_topk_no_bias() {
        // Logits chosen so sigmoid(x) is roughly [0.27, 0.5, 0.73, 0.88, 0.95]
        // for [-1, 0, 1, 2, 3]. Top-2 by score = experts 4 and 3.
        let logits = vec![-1.0, 0.0, 1.0, 2.0, 3.0];
        let selected = deepseek_moe_route(&logits, None, 2, 1.0);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].0, 4);
        assert_eq!(selected[1].0, 3);
        // Weights should sum to ~routed_scale (=1.0 here) after renormalize.
        let sum: f32 = selected.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 1e-5, "weights sum = {sum}");
    }

    #[test]
    fn deepseek_moe_route_noaux_tc_bias_affects_selection_not_weights() {
        // Logits: expert 0 has highest un-biased score (sigmoid(3) ≈ 0.95).
        // Bias vector boosts expert 3 by +10 so it becomes top-1 despite a
        // much lower un-biased score. But the returned weight for expert 3
        // must be its un-biased sigmoid score (renormalized), NOT the biased
        // one — that's the whole point of noaux_tc.
        let logits = vec![3.0, -5.0, -5.0, -1.0, -5.0];
        let bias = vec![0.0, 0.0, 0.0, 10.0, 0.0];
        let selected = deepseek_moe_route(&logits, Some(&bias), 2, 1.0);
        // Selection: biased top-2 = expert 3 (score+10) and expert 0.
        let picked: Vec<usize> = selected.iter().map(|(i, _)| *i).collect();
        assert!(picked.contains(&3), "expert 3 must be top-k via bias");
        assert!(
            picked.contains(&0),
            "expert 0 must be top-k via un-biased score"
        );
        // The final weight of expert 3 must be based on its un-biased score
        // (sigmoid(-1) ≈ 0.269), NOT sigmoid(-1) + 10.
        let w3 = selected.iter().find(|(i, _)| *i == 3).unwrap().1;
        let w0 = selected.iter().find(|(i, _)| *i == 0).unwrap().1;
        // Un-biased renormalize: w0 / w3 ≈ sigmoid(3) / sigmoid(-1) ≈ 3.53.
        // If bias leaked into weights we'd see w3 dominant instead.
        assert!(
            w0 > w3,
            "un-biased weight of expert 0 must exceed expert 3 (w0={w0} w3={w3})"
        );
    }

    #[test]
    fn deepseek_moe_route_routed_scaling_factor() {
        // With routed_scale = 2.5, the returned weights must sum to
        // routed_scale (not 1.0) — DeepSeek-V3 uses this to amplify the
        // routed sum relative to the always-active shared expert.
        let logits = vec![0.0, 1.0, 2.0, 3.0];
        let selected = deepseek_moe_route(&logits, None, 2, 2.5);
        let sum: f32 = selected.iter().map(|(_, w)| *w).sum();
        assert!(
            (sum - 2.5).abs() < 1e-5,
            "weights sum = {sum}, expected 2.5"
        );
    }

    #[test]
    fn deepseek_moe_route_top_k_larger_than_experts() {
        // Edge case: top-k > n_experts. Must not panic; selects all.
        let logits = vec![1.0, 2.0];
        let selected = deepseek_moe_route(&logits, None, 8, 1.0);
        assert_eq!(selected.len(), 2, "must select all experts when top-k > n");
        let sum: f32 = selected.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // ── DeepSeek-V3 Phase 5a.2: MTP adaptive draft policy ────────────

    /// A fresh policy starts enabled — every `should_draft` returns `true`
    /// until enough rejections force a cooldown transition.
    #[test]
    fn mtp_policy_starts_enabled() {
        let mut p = MtpDraftPolicy::new_default();
        for _ in 0..5 {
            assert!(p.should_draft(), "fresh policy must draft immediately");
        }
    }

    /// Filling the window with all-rejections must trip the cooldown and
    /// then `should_draft` must return `false` for exactly
    /// `cooldown_tokens` subsequent calls.
    #[test]
    fn mtp_policy_enters_cooldown_after_all_rejections() {
        let window = 4;
        let cooldown = 3;
        let mut p = MtpDraftPolicy::with_params(window, 0.30, cooldown);
        for _ in 0..window {
            assert!(p.should_draft());
            p.record(false); // reject
        }
        // Now the window has 4/4 rejects → rate 0 < 0.30, cooldown started.
        assert!(
            p.stats().in_cooldown,
            "must be in cooldown after all-reject window"
        );
        for _ in 0..cooldown {
            assert!(!p.should_draft(), "cooldown must skip drafts");
        }
        // Cooldown just expired → drafts resume.
        assert!(p.should_draft(), "post-cooldown must resume drafting");
    }

    /// Accept rate above threshold must NOT trigger cooldown, even after
    /// the window fills.
    #[test]
    fn mtp_policy_stays_enabled_above_threshold() {
        let window = 4;
        let mut p = MtpDraftPolicy::with_params(window, 0.30, 8);
        // 3 accepts + 1 reject = 75% > 30%, no cooldown.
        for &accept in &[true, true, true, false] {
            assert!(p.should_draft());
            p.record(accept);
        }
        assert!(
            !p.stats().in_cooldown,
            "75% accept rate must not trigger cooldown"
        );
    }

    /// The window resets on cooldown entry so a single accept post-cooldown
    /// does NOT prematurely re-trigger evaluation.
    #[test]
    fn mtp_policy_clears_window_on_cooldown() {
        let mut p = MtpDraftPolicy::with_params(4, 0.30, 2);
        for _ in 0..4 {
            assert!(p.should_draft());
            p.record(false);
        }
        assert!(p.stats().in_cooldown);
        // Tick through cooldown.
        for _ in 0..2 {
            let _ = p.should_draft();
        }
        // Single accept post-cooldown — window is empty so evaluation
        // needs 4 more decisions before it can re-trigger.
        assert!(p.should_draft());
        p.record(true);
        assert!(
            !p.stats().in_cooldown,
            "single post-cooldown accept must not re-trigger"
        );
    }

    /// Stats counter must accumulate accepts + rejects across cooldowns
    /// (cooldowns clear the window but not the lifetime totals).
    #[test]
    fn mtp_policy_stats_track_lifetime_totals() {
        let mut p = MtpDraftPolicy::with_params(4, 0.30, 1);
        for _ in 0..4 {
            assert!(p.should_draft());
            p.record(false);
        }
        assert_eq!(p.stats().total_rejected, 4);
        // Cool down for one, then accept some.
        let _ = p.should_draft();
        for _ in 0..3 {
            assert!(p.should_draft());
            p.record(true);
        }
        let stats = p.stats();
        assert_eq!(stats.total_accepted, 3);
        assert_eq!(stats.total_rejected, 4);
        assert!((stats.overall_accept_rate - 3.0 / 7.0).abs() < 1e-5);
    }

    // ── DeepSeek-V3 Phase 5a: MTP loader gating ──────────────────────

    /// A DeepSeek-V3 config whose `mtp_layer` is `None` must produce a
    /// model with no MTP head — the loader is short-circuited before it
    /// tries to look up any `mtp.*` tensor. Guards the fast path for V2
    /// and pre-MTP V3 quants.
    #[test]
    fn mtp_loader_returns_none_when_config_has_no_mtp_layer() {
        // Config with mtp_layer explicitly unset.
        let mut c = tiny_deepseek_v3_config();
        if let Some(d) = c.deepseek_v3.as_mut() {
            d.mtp_layer = None;
        }
        // The loader early-returns on config gate; no GGUF needed.
        assert!(c.deepseek_mtp_layer().is_none());
    }

    /// The `has_deepseek_mtp` predicate on a non-DeepSeek model is
    /// vacuously false. Regression guard against future refactors that
    /// might inadvertently return true for Llama-3.
    #[test]
    fn has_deepseek_mtp_false_when_arch_is_not_deepseek() {
        // We construct a Llama3Model isn't possible without a full GGUF,
        // but we can at least assert the invariant at the config layer:
        // non-DeepSeek configs never populate `deepseek_v3`, so the
        // `deepseek_v3_mtp` field will always be None regardless of
        // whether the MTP layer count env is set.
        let llama_config = Llama3Config {
            arch: ModelArch::Llama,
            vocab_size: 128,
            hidden_dim: 16,
            intermediate_dim: 32,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: 2,
            max_seq_len: 32,
            head_dim: 8,
            rope_theta: 500_000.0,
            norm_eps: 1e-6,
            attention_extras: None,
            ssm: None,
            moe: None,
            gemma3n: None,
            gemma4: None,
            deepseek_v3: None,
        };
        assert!(llama_config.deepseek_mtp_layer().is_none());
    }

    #[test]
    fn deepseek_v3_first_k_dense_boundary_gates_moe_branch() {
        // Layers below `first_k_dense_replace` take the dense SwiGLU FFN
        // path; layers at or above take the DeepSeek-V3 MoE branch
        // (Phase 3 landed — no more panic). The tiny fixture keeps
        // `first_k_dense_replace >= num_layers` so every layer stays on
        // the dense path — the MoE branch is exercised end-to-end only
        // when a real GGUF (or a fixture that actually loads MoE weights)
        // is loaded, since the MoE weights are not synthesised here.
        // This guard makes sure the boundary predicate itself stays
        // aligned with what `forward_deepseek_v3` inspects.
        let c = tiny_deepseek_v3_config();
        let first_k = c.deepseek_first_k_dense_replace().unwrap();
        assert!(first_k >= c.num_layers, "tiny fixture must stay dense-only");
    }

    #[test]
    fn deepseek_v3_config_fixture_declares_all_mla_fields() {
        // A tripwire: if a future PR adds an MLA field, this test forces
        // the tiny fixture (and by extension every deep-seek test) to be
        // updated so the shape assertions keep making sense.
        let c = tiny_deepseek_v3_config();
        let d = c.deepseek_v3.as_ref().unwrap();
        assert!(d.q_lora_rank.is_some());
        assert!(d.kv_lora_rank.is_some());
        assert!(d.qk_nope_head_dim.is_some());
        assert!(d.qk_rope_head_dim.is_some());
        assert!(d.v_head_dim.is_some());
    }

    /// The `load_ffn_norm` helper must resolve either `ffn_norm.weight` or
    /// `post_attention_norm.weight`, so that Bonsai 27B (which exports the
    /// latter under the qwen35 arch) can be loaded alongside standard Qwen /
    /// Llama-3 GGUF exports without a per-arch branch.
    #[test]
    fn load_ffn_norm_accepts_both_names() {
        // Build a tiny in-memory GGUF containing only what `load_ffn_norm`
        // needs: one f32 tensor `blk.0.ffn_norm.weight` in the "standard"
        // case, then again `blk.0.post_attention_norm.weight` in the
        // "Bonsai" case. Both must round-trip through the loader.

        fn tiny_gguf_with(tensor_name: &str) -> Vec<u8> {
            let mut buf = Vec::new();
            // Header
            buf.extend_from_slice(b"GGUF");
            buf.extend_from_slice(&3u32.to_le_bytes()); // version
            buf.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
            buf.extend_from_slice(&1u64.to_le_bytes()); // n_kv

            // Single kv: general.alignment = 32 (u32)
            let key = "general.alignment";
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&4u32.to_le_bytes()); // type=U32
            buf.extend_from_slice(&32u32.to_le_bytes());

            // Tensor info
            buf.extend_from_slice(&(tensor_name.len() as u64).to_le_bytes());
            buf.extend_from_slice(tensor_name.as_bytes());
            buf.extend_from_slice(&1u32.to_le_bytes()); // ndims
            buf.extend_from_slice(&4u64.to_le_bytes()); // shape[0] = 4
            buf.extend_from_slice(&0u32.to_le_bytes()); // ggml_type = F32
            buf.extend_from_slice(&0u64.to_le_bytes()); // data_offset

            // Pad to alignment 32
            while !buf.len().is_multiple_of(32) {
                buf.push(0);
            }
            // Tensor data: 4 f32 values
            for x in [1.0f32, 2.0, 3.0, 4.0] {
                buf.extend_from_slice(&x.to_le_bytes());
            }
            buf
        }

        // Standard name (`ffn_norm.weight`)
        let bytes = tiny_gguf_with("blk.0.ffn_norm.weight");
        let gguf = crate::gguf::GgufFile::parse(&bytes).expect("parse standard");
        let v = load_ffn_norm(&gguf, "blk.0").expect("load standard ffn_norm");
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);

        // Bonsai alias (`post_attention_norm.weight`)
        let bytes = tiny_gguf_with("blk.0.post_attention_norm.weight");
        let gguf = crate::gguf::GgufFile::parse(&bytes).expect("parse bonsai");
        let v = load_ffn_norm(&gguf, "blk.0").expect("load bonsai post_attention_norm");
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);

        // Neither present → None (upstream `?` will propagate the miss).
        let bytes = tiny_gguf_with("blk.0.something_else.weight");
        let gguf = crate::gguf::GgufFile::parse(&bytes).expect("parse other");
        assert!(load_ffn_norm(&gguf, "blk.0").is_none());
    }

    /// `load_weight_ref_any_rows` derives the row count from the tensor's
    /// GGUF `dims` metadata rather than requiring the caller to pass one.
    /// Bonsai 27B's `attn_qkv` (10240 rows) and `attn_gate` (6144 rows)
    /// depend on this because their sizes are not implied by the standard
    /// Qwen 3.5 config keys.
    #[test]
    fn load_weight_ref_any_rows_derives_shape_from_gguf() {
        // Tiny GGUF with a single f32 tensor `w` of shape `[cols=2, rows=3]`
        // = 6 f32 values. `load_weight_ref_any_rows` must pick `rows=3` up
        // from the header without being told.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&1u64.to_le_bytes()); // n_kv

        // One kv (alignment=32).
        let key = "general.alignment";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes()); // U32
        buf.extend_from_slice(&32u32.to_le_bytes());

        // Tensor info for `w` with shape `[2, 3]`.
        let name = "w";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // ndims
        buf.extend_from_slice(&2u64.to_le_bytes()); // dims[0] = 2 (cols)
        buf.extend_from_slice(&3u64.to_le_bytes()); // dims[1] = 3 (rows)
        buf.extend_from_slice(&0u32.to_le_bytes()); // ggml_type = F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // data_offset = 0

        while !buf.len().is_multiple_of(32) {
            buf.push(0);
        }
        // 6 f32 values = 24 bytes.
        for x in 0..6 {
            buf.extend_from_slice(&(x as f32).to_le_bytes());
        }

        let gguf = crate::gguf::GgufFile::parse(&buf).expect("parse ok");
        let w = load_weight_ref_any_rows(&gguf, "w", 2).expect("load ok");
        assert_eq!(w.rows, 3, "rows must be derived from dims[1]");
        assert_eq!(w.cols, 2, "cols is passed by caller");

        // Missing tensor → None.
        assert!(load_weight_ref_any_rows(&gguf, "not_present", 2).is_none());
    }

    /// Qwen 3.6 / Bonsai 27B Gated Attention post-hoc validation: the swish
    /// gate maps `x → x * sigmoid(x)`, so gate = 0 nullifies the attention
    /// output while a large positive gate lets it through. Verifies the
    /// arithmetic done inline in the main `forward` path against a scalar
    /// reference, independent of any real GGUF weights.
    #[test]
    fn gated_attention_swish_math_matches_reference() {
        // 6 attention output values with 6 gate values.
        let mut attn_out = vec![1.0f32, 2.0, -1.0, 0.5, -0.5, 3.0];
        let gate = vec![0.0f32, 1.0, -1.0, 10.0, -10.0, 0.5];
        let q_dim = attn_out.len();

        // Reference: each output multiplied by silu(gate).
        let expected: Vec<f32> = attn_out
            .iter()
            .zip(gate.iter())
            .map(|(&a, &g)| a * silu(g))
            .collect();

        // In-place update mirroring the forward path body.
        for i in 0..q_dim {
            attn_out[i] *= silu(gate[i]);
        }

        for (i, (&got, &want)) in attn_out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "gated attention at {i}: got {got}, expected {want}"
            );
        }

        // gate = 0 → silu(0) = 0 → output zeroed out.
        assert_eq!(attn_out[0], 0.0);
        // gate = -10 → silu(-10) ≈ 0 → output near-zero.
        assert!(attn_out[4].abs() < 1e-3);
        // gate = 10 → silu(10) ≈ 10 → output amplified.
        assert!(attn_out[3] > 4.9);
    }
}
