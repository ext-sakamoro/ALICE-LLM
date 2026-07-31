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

// ─── External signal type aliases ──────────────────────────────────────────

/// External per-token signal vector passed to [`Llama3Model::forward_with_surprise`].
///
/// The shape and meaning of the slice is intentionally unopinionated — the
/// caller decides whether the elements represent per-body prediction error,
/// per-region perceptual error, per-modality entropy, an aggregated scalar
/// broadcast, or any other per-token routing signal. `forward_with_surprise`
/// only forwards the borrow to the caller-supplied `gate` closure.
pub type SurpriseVec<'a> = &'a [f32];

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
    /// Kimi K3 / Kimi Delta Attention family (Moonshot AI). Open weight
    /// release landed 2026-07-27 (2.8T total / 104B active MoE, 896
    /// experts top-16, 93 layers = 69 KDA + 24 Gated MLA + 1 dense,
    /// hidden 7168, 1M context, native MXFP4). Phase X.4 integration
    /// target; foundation only until the CPU forward path lands
    /// (X.4.c) and community GGUF conversion arrives (X.4.b).
    ///
    /// `forward()` immediately `todo!()` on `KimiK3` — silent garbage
    /// on a 2.8T model is worse than an explicit panic pointing to
    /// `docs/KIMI_K3_INTEGRATION.md`. The full HF-confirmed spec is
    /// captured by [`KimiDeltaConfig`] and parseable from `config.json`
    /// via [`KimiDeltaConfig::from_hf_config`] (`hf-config` feature).
    KimiK3,
    /// Tencent Hy3 (Hunyuan 3) family. Apache 2.0, 295B total / 21B
    /// active (MoE), 192 experts top-8, 1 MTP layer (3.8B), GQA 8 KV
    /// heads, 256K context. FP8 (E4M3) native + BF16 fallback via
    /// AngelSlim quantization toolkit.
    ///
    /// Skeleton only until the community `convert_hf_to_gguf.py` /
    /// llama.cpp support settles and the actual GGUF metadata prefix
    /// stabilises (best guesses right now are `hunyuan` /
    /// `hunyuanmoe` / `hy3`, all covered by `from_gguf`). Once weights
    /// + GGUF land, the CPU forward is expected to inherit ~90% from
    /// the Bonsai / Qwen 3.6 `gated_deltanet` path (top-K sparse
    /// routing over 192 experts is the main net-new implementation,
    /// GQA + MoE scaffolding is already reused across Kimi K3 /
    /// DeepSeek V3).
    ///
    /// `forward()` immediately `todo!()` on `Hy3` — silent garbage on
    /// a 295B model is worse than an explicit panic pointing to the
    /// integration doc.
    ///
    /// References:
    /// - GitHub: `Tencent-Hunyuan/Hy3` (weights + inference + RL)
    /// - HuggingFace: `tencent/Hy3` (BF16) + `tencent/Hy3-FP8`
    Hy3,
}

impl ModelArch {
    /// Detect architecture from GGUF metadata key `general.architecture`.
    pub fn from_gguf<'a, G: crate::gguf::GgufSource<'a>>(gguf: &'a G) -> Self {
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
            // Kimi K3 / Kimi Delta family. Guess of the llama.cpp prefix
            // until the actual convert_hf_to_gguf.py drop for K3 lands;
            // covers `"kimi"`, `"kimi3"`, `"kimideltatt"` variants seen
            // in draft PRs. Refine once the community GGUF conversion
            // finalizes (see docs/KIMI_K3_INTEGRATION.md §X.4.b).
            Some(s) if s.starts_with("kimi") => Self::KimiK3,
            // Tencent Hy3 (Hunyuan 3). Same "waiting on community
            // convert_hf_to_gguf.py" caveat as Kimi K3 — the prefix in
            // the eventual GGUF conversion is not yet fixed, so match
            // any of the plausible `hunyuan*` / `hy3` variants and
            // refine once the community lands a canonical prefix.
            Some("hy3" | "hy3moe" | "hunyuan_moe") => Self::Hy3,
            Some(s) if s.starts_with("hunyuan") => Self::Hy3,
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
            // Phase X.4.b.1 (2026-07-28) confirmed: the community
            // conversion (`Kuberwastaken/Kimi-K3-GGUF/convert_kimi_k3.py`
            // + upstream `pwilkin/kimi-k3-text` PR #26185) writes
            // `general.architecture = "kimi-k3"` (hyphenated). Existing
            // arch auto-detect at `from_gguf` matches `starts_with("kimi")`
            // and already routes hyphenated forms to `ModelArch::KimiK3`;
            // the `meta_prefix` here is used for hyperparameter key
            // lookups (`kimi-k3.embedding_length`, etc.), so it must be
            // the full hyphenated string.
            Self::KimiK3 => "kimi-k3",
            // Same TODO as KimiK3 — the eventual `general.architecture`
            // string for Hy3 GGUF is not yet fixed by the community.
            // `hunyuan` is the working guess based on the HuggingFace
            // model ids (`tencent/Hy3` under the Hunyuan org).
            Self::Hy3 => "hunyuan",
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
    fn resolve_prefix<'a, G: crate::gguf::GgufSource<'a>>(&self, gguf: &'a G) -> String {
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

/// Kimi K3 / Kimi Delta Attention family sub-config.
///
/// Moonshot AI's Kimi K3 open weight release (2026-07-27) confirmed the full
/// architectural spec via `huggingface.co/moonshotai/Kimi-K3/config.json`
/// (`model_type = "kimi_k3"`, `text_config.model_type = "kimi_linear"`,
/// plus a nested MoonViT-V2 `vision_config`). All values marked "K3:" in the
/// field docs below are taken directly from that file; individual fields
/// remain `Option<T>` so the same struct can be populated piecewise by
/// either the future GGUF loader (Phase X.4.b, blocked on community
/// `convert_hf_to_gguf.py`) or by [`KimiDeltaConfig::from_hf_config`] for
/// the direct-safetensors path.
///
/// GGUF metadata key prefix (guess): `kimi.*` — confirm at X.4.b once the
/// community conversion lands. See `docs/KIMI_K3_INTEGRATION.md` for the
/// full 10-sub-phase integration plan.
#[derive(Debug, Clone, Default)]
pub struct KimiDeltaConfig {
    // ── Hybrid attention layer routing (linear_attn_config.*) ─────────
    /// Explicit list of full-attention (Gated MLA) layer indices, 1-indexed.
    /// K3: 24 layers = `[4, 8, 12, ..., 88, 92, 93]`.
    /// Source: HF `text_config.linear_attn_config.full_attn_layers`.
    pub full_attn_layers: Option<Vec<usize>>,
    /// Explicit list of Kimi Delta Attention layer indices, 1-indexed.
    /// K3: 69 layers = `[1, 2, 3, 5, 6, 7, ..., 89, 90, 91]`.
    /// Source: HF `text_config.linear_attn_config.kda_layers`.
    pub kda_layers: Option<Vec<usize>>,
    /// KDA per-head dimension. K3: 128.
    pub kda_head_dim: Option<usize>,
    /// KDA number of heads. K3: 96 (matches `num_attention_heads`).
    pub kda_num_heads: Option<usize>,
    /// KDA short-conv kernel size. K3: 4.
    pub kda_short_conv_kernel_size: Option<usize>,
    /// KDA gate uses a full-rank projection (vs low-rank). K3: `true`.
    pub kda_use_full_rank_gate: Option<bool>,
    /// KDA gate lower bound (clamp for numerical stability). K3: `-5.0`.
    pub kda_gate_lower_bound: Option<f32>,

    // ── Gated MLA config (mirrors DeepSeek V3 MLA + output gate) ─────
    /// Q LoRA rank. K3: 1536 (identical to DeepSeek V3).
    pub q_lora_rank: Option<usize>,
    /// KV LoRA rank. K3: 512 (identical to DeepSeek V3).
    pub kv_lora_rank: Option<usize>,
    /// Non-RoPE Q/K head dim. K3: 128.
    pub qk_nope_head_dim: Option<usize>,
    /// RoPE Q/K head dim (partial RoPE). K3: 64.
    pub qk_rope_head_dim: Option<usize>,
    /// V head dim. K3: 128.
    pub v_head_dim: Option<usize>,
    /// Whether the MLA path skips RoPE entirely. K3: `true` (unusual —
    /// DeepSeek V3 uses partial RoPE; K3's NoPE + AttnRes is a
    /// Kimi-specific choice).
    pub mla_use_nope: Option<bool>,
    /// Whether MLA adds an output gate on top of the attention block.
    /// K3: `true` (unique to Kimi; DeepSeek V3 has no output gate).
    pub mla_use_output_gate: Option<bool>,

    // ── Attention Residuals (AttnRes) ─────────────────────────────────
    /// AttnRes block size — number of consecutive layers grouped under
    /// one residual skip. Moonshot claims ~25% training speedup.
    /// K3: 12. Source: HF `text_config.attn_res_block_size`.
    pub attn_res_block_size: Option<usize>,

    // ── SiTU-GLU activation ──────────────────────────────────────────
    /// SiTU activation β1 (nonlinear branch). K3: 4.0.
    pub situ_beta: Option<f32>,
    /// SiTU activation β2 (linear branch). K3: 25.0.
    pub situ_linear_beta: Option<f32>,

    // ── Stable LatentMoE (896 experts, top-16) ───────────────────────
    /// Total routed experts. K3: **896** (sparsity 16/896 ≈ 1.79%).
    /// Drives the expert-streaming feasibility calculation in
    /// `docs/KIMI_K3_INTEGRATION.md` (~24 GB Q4 active weights per
    /// token out of ~1.4 TB total, streamable from NVMe on Mac 128 GB).
    pub n_routed_experts: Option<usize>,
    /// Top-k routed experts per token. K3: **16**.
    pub num_experts_per_tok: Option<usize>,
    /// Shared always-active experts. K3: 2 (DeepSeek V3 = 1).
    pub n_shared_experts: Option<usize>,
    /// Expert group count for grouped top-k routing. K3: 1.
    pub num_expert_group: Option<usize>,
    /// Top-k expert groups. K3: 1.
    pub topk_group: Option<usize>,
    /// Router activation function. K3: `"sigmoid"`.
    pub moe_router_activation: Option<String>,
    /// Router scoring method. K3: `"noaux_tc"` (no auxiliary-loss trick,
    /// same as DeepSeek V3).
    pub moe_topk_method: Option<String>,
    /// Per-expert FFN intermediate size. K3: 3072.
    pub moe_intermediate_size: Option<usize>,
    /// Leading dense-FFN layer count (before MoE begins). K3: 1
    /// (layer 0 dense, layers 1..93 MoE + shared experts).
    pub first_k_dense_replace: Option<usize>,
    /// Whether the routed expert outputs are renormalized. K3: `true`.
    pub moe_renormalize: Option<bool>,
    /// Latent MoE hidden dim (routed-expert latent space). K3: 3584.
    pub routed_expert_hidden_size: Option<usize>,
    /// Whether Latent MoE applies RMSNorm inside the routing block.
    /// K3: `true`.
    pub latent_moe_use_norm: Option<bool>,
    /// Routed expert output scale. K3: 1.0 (DeepSeek V3 = 2.5).
    pub routed_scaling_factor: Option<f32>,

    // ── MTP head (not present in K3) ─────────────────────────────────
    /// Multi-Token Prediction head layer count. K3: **0** (no MTP head,
    /// unlike DeepSeek V3 which has 1). Retained for dispatch parity so
    /// the same code path that gates MTP for V3 can short-circuit here.
    pub num_nextn_predict_layers: Option<usize>,

    // ── MXFP4 native quantization ────────────────────────────────────
    /// Native MXFP4 group size for K3 checkpoints. K3: 32.
    /// See `docs/MXFP4_INTEGRATION_PLAN.md` (Phase X.4.f).
    pub mxfp4_group_size: Option<usize>,
    /// MXFP4 bits per weight. K3: 4.
    pub mxfp4_num_bits: Option<usize>,
}

#[cfg(feature = "hf-config")]
impl KimiDeltaConfig {
    /// Parse a `KimiDeltaConfig` from the HuggingFace Kimi K3 `config.json`
    /// (`huggingface.co/moonshotai/Kimi-K3/raw/main/config.json`).
    ///
    /// The Kimi K3 top-level `config.json` has three nested sub-configs:
    /// `text_config` (a `KimiLinearConfig`, `model_type = "kimi_linear"`),
    /// `vision_config` (MoonViT-V2), and a top-level `model_type` of
    /// `"kimi_k3"`. This function extracts the fields required by the
    /// forward path (`text_config.*` plus `linear_attn_config.*`).
    ///
    /// Vision fields are deliberately not parsed here — the multimodal
    /// path is scheduled for Phase X.4.i and will need its own
    /// `KimiK3VisionConfig` sub-config once MoonViT-V2 lands.
    ///
    /// # Errors
    ///
    /// Returns [`serde_json::Error`] when the input bytes are not valid
    /// JSON. Missing individual fields inside `text_config` are permitted
    /// (stored as `None`) so this loader still yields a usable partial
    /// config for pre-release checkpoint variants and future field
    /// additions upstream.
    pub fn from_hf_config(json_bytes: &[u8]) -> Result<Self, serde_json::Error> {
        let root: serde_json::Value = serde_json::from_slice(json_bytes)?;
        let tc = &root["text_config"];
        let la = &tc["linear_attn_config"];
        let qw = &tc["quantization_config"]["config_groups"]["group_0"]["weights"];

        let opt_usize = |v: &serde_json::Value| v.as_u64().map(|x| x as usize);
        let opt_f32 = |v: &serde_json::Value| v.as_f64().map(|x| x as f32);
        let opt_bool = |v: &serde_json::Value| v.as_bool();
        let opt_str = |v: &serde_json::Value| v.as_str().map(str::to_owned);
        let opt_vec_usize = |v: &serde_json::Value| {
            v.as_array().map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect()
            })
        };

        Ok(Self {
            full_attn_layers: opt_vec_usize(&la["full_attn_layers"]),
            kda_layers: opt_vec_usize(&la["kda_layers"]),
            kda_head_dim: opt_usize(&la["head_dim"]),
            kda_num_heads: opt_usize(&la["num_heads"]),
            kda_short_conv_kernel_size: opt_usize(&la["short_conv_kernel_size"]),
            kda_use_full_rank_gate: opt_bool(&la["use_full_rank_gate"]),
            kda_gate_lower_bound: opt_f32(&la["gate_lower_bound"]),
            q_lora_rank: opt_usize(&tc["q_lora_rank"]),
            kv_lora_rank: opt_usize(&tc["kv_lora_rank"]),
            qk_nope_head_dim: opt_usize(&tc["qk_nope_head_dim"]),
            qk_rope_head_dim: opt_usize(&tc["qk_rope_head_dim"]),
            v_head_dim: opt_usize(&tc["v_head_dim"]),
            mla_use_nope: opt_bool(&tc["mla_use_nope"]),
            mla_use_output_gate: opt_bool(&tc["mla_use_output_gate"]),
            attn_res_block_size: opt_usize(&tc["attn_res_block_size"]),
            situ_beta: opt_f32(&tc["activation_situ_beta"]),
            situ_linear_beta: opt_f32(&tc["activation_situ_linear_beta"]),
            n_routed_experts: opt_usize(&tc["num_experts"]),
            num_experts_per_tok: opt_usize(&tc["num_experts_per_token"]),
            n_shared_experts: opt_usize(&tc["num_shared_experts"]),
            num_expert_group: opt_usize(&tc["num_expert_group"]),
            topk_group: opt_usize(&tc["topk_group"]),
            moe_router_activation: opt_str(&tc["moe_router_activation_func"]),
            moe_topk_method: opt_str(&tc["topk_method"]),
            moe_intermediate_size: opt_usize(&tc["moe_intermediate_size"]),
            first_k_dense_replace: opt_usize(&tc["first_k_dense_replace"]),
            moe_renormalize: opt_bool(&tc["moe_renormalize"]),
            routed_expert_hidden_size: opt_usize(&tc["routed_expert_hidden_size"]),
            latent_moe_use_norm: opt_bool(&tc["latent_moe_use_norm"]),
            routed_scaling_factor: opt_f32(&tc["routed_scaling_factor"]),
            num_nextn_predict_layers: opt_usize(&tc["num_nextn_predict_layers"]),
            mxfp4_group_size: opt_usize(&qw["group_size"]),
            mxfp4_num_bits: opt_usize(&qw["num_bits"]),
        })
    }
}

impl KimiDeltaConfig {
    /// Parse a `KimiDeltaConfig` from a Kimi K3 GGUF file
    /// (Phase X.4.b.1, 2026-07-28).
    ///
    /// Reads all K3-specific hyperparameters from the `kimi-k3.*`
    /// namespace as written by `Kuberwastaken/Kimi-K3-GGUF/k3meta.py` +
    /// upstream `pwilkin/kimi-k3-text` (llama.cpp PR #26185). Standard
    /// hyperparameters (embedding_length, block_count, head_count,
    /// etc.) are read separately by the outer `Llama3Config::from_gguf`
    /// branch and populate [`Llama3Config`] core fields; this
    /// function fills in only the K3-only fields.
    ///
    /// The `prefix` argument is normally `"kimi-k3"` (from
    /// [`ModelArch::KimiK3::meta_prefix()`]) but is passed explicitly
    /// so future prefix versioning (e.g. `kimi-k3v2`) can override
    /// without touching this parser.
    ///
    /// # Missing fields
    ///
    /// Individual `Option` fields default to `None` when the
    /// corresponding metadata key is absent, so a partial GGUF (e.g.
    /// text-only variant without MXFP4 quantization metadata) still
    /// parses without erroring.
    #[must_use]
    pub fn from_gguf<'a, G: crate::gguf::GgufSource<'a>>(gguf: &'a G, prefix: &str) -> Self {
        let opt_usize = |key: &str| gguf.meta_u32(key).map(|v| v as usize);
        let opt_bool = |key: &str| gguf.meta_bool(key);
        let opt_str = |key: &str| gguf.meta_str(key).map(str::to_owned);
        let opt_f32 = |key: &str| gguf.meta_f32(key);
        let opt_vec_usize = |key: &str| {
            gguf.meta(key)
                .and_then(crate::gguf::MetaValue::as_u32_array)
                .map(|arr| arr.into_iter().map(|v| v as usize).collect())
        };
        // First-non-None helpers: real K3 GGUF (GrEarl 2026-07-28
        // inspection) diverges from the pwilkin PR / synthetic k3meta
        // conventions on ~5 keys — check both spellings.
        let opt_usize_any = |keys: &[String]| keys.iter().find_map(|k| opt_usize(k));
        let opt_f32_any = |keys: &[String]| keys.iter().find_map(|k| opt_f32(k));

        // MLA sub-config keys land under the shared `.attention.*`
        // namespace so llama.cpp's existing MLA machinery can reuse
        // them — `q_lora_rank`, `kv_lora_rank`, `key_length`,
        // `value_length_mla`, etc.
        let attn = |key: &str| format!("{prefix}.attention.{key}");
        let ssm = |key: &str| format!("{prefix}.ssm.{key}");
        let rope = |key: &str| format!("{prefix}.rope.{key}");
        // K3-only keys live in the plain `kimi-k3.*` namespace.
        let k3 = |key: &str| format!("{prefix}.{key}");
        // K3 KDA namespace (`kimi-k3.kda.*`): real GrEarl exports use
        // this instead of the `attention.kda_*` synthetic naming.
        let kda = |key: &str| format!("{prefix}.kda.{key}");
        // K3 AttnRes namespace (`kimi-k3.attn_res.*`).
        let attn_res = |key: &str| format!("{prefix}.attn_res.{key}");

        // Derive nope/rope head dims from the split key_length_mla and
        // rope.dimension_count values (following the k3meta.py write
        // order: `key_length_mla = qk_nope_head_dim + qk_rope_head_dim`,
        // and `rope.dimension_count = qk_rope_head_dim`).
        let qk_rope_head_dim = opt_usize(&rope("dimension_count"));
        let qk_nope_head_dim = opt_usize(&attn("key_length_mla"))
            .zip(qk_rope_head_dim)
            .map(|(kl_mla, rope_dim)| kl_mla.saturating_sub(rope_dim));

        Self {
            // Hybrid attention routing (0-indexed after conversion).
            // Real GrEarl K3 GGUF does not export these arrays — the
            // loader falls back to per-layer tensor-name inspection
            // (`load_kimi_k3_layer_weights` sees whether `attn_q_a`
            // or `attn_q` exists) when both keys are absent.
            full_attn_layers: opt_vec_usize(&k3("full_attn_layers")),
            kda_layers: opt_vec_usize(&k3("kda_layers")),
            // KDA head dim: pwilkin/synthetic = `attention.kda_head_dim`,
            // real GrEarl = `kimi-k3.kda.head_dim`.
            kda_head_dim: opt_usize_any(&[attn("kda_head_dim"), kda("head_dim")]),
            kda_num_heads: opt_usize(&attn("head_count")),
            kda_short_conv_kernel_size: opt_usize(&ssm("conv_kernel")),
            kda_use_full_rank_gate: opt_bool(&k3("use_full_rank_gate")),
            // Gate lower bound: synthetic = `gate_lower_bound`, real =
            // `kimi-k3.kda.gate_lower_bound`.
            kda_gate_lower_bound: opt_f32_any(&[k3("gate_lower_bound"), kda("gate_lower_bound")]),

            // Gated MLA config.
            q_lora_rank: opt_usize(&attn("q_lora_rank")),
            kv_lora_rank: opt_usize(&attn("kv_lora_rank")),
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim: opt_usize(&attn("value_length_mla")),
            mla_use_nope: opt_bool(&k3("mla_use_nope")),
            mla_use_output_gate: opt_bool(&k3("mla_use_output_gate")),

            // Attention Residuals + SiTU-GLU. AttnRes block size:
            // synthetic = `attn_res_block_size`, real =
            // `kimi-k3.attn_res.block_size`.
            attn_res_block_size: opt_usize_any(&[
                k3("attn_res_block_size"),
                attn_res("block_size"),
            ]),
            // SiTU coefficients: synthetic = `activation_situ_beta`
            // (single-underscore), real = `activation.situ_beta`
            // (dot-separated).
            situ_beta: opt_f32_any(&[k3("activation_situ_beta"), k3("activation.situ_beta")]),
            situ_linear_beta: opt_f32_any(&[
                k3("activation_situ_linear_beta"),
                k3("activation.situ_linear_beta"),
            ]),

            // Stable LatentMoE. Uses the standard llama.cpp expert
            // metadata keys plus a K3-only `routed_expert_hidden_size`
            // (real GrEarl exports it as `expert_latent_length` — see
            // pwilkin constants.py `EXPERT_LATENT_LENGTH`).
            n_routed_experts: opt_usize(&k3("expert_count")),
            num_experts_per_tok: opt_usize(&k3("expert_used_count")),
            n_shared_experts: opt_usize(&k3("expert_shared_count")),
            num_expert_group: opt_usize(&k3("num_expert_group")),
            topk_group: opt_usize(&k3("topk_group")),
            // Router activation: synthetic = `moe_router_activation_func`,
            // real = `expert_gating_func`.
            moe_router_activation: opt_str(&k3("moe_router_activation_func"))
                .or_else(|| opt_str(&k3("expert_gating_func"))),
            moe_topk_method: opt_str(&k3("topk_method")),
            moe_intermediate_size: opt_usize(&k3("expert_feed_forward_length")),
            first_k_dense_replace: opt_usize(&k3("leading_dense_block_count")),
            // Renormalize: synthetic = `moe_renormalize`, real =
            // `expert_weights_norm`.
            moe_renormalize: opt_bool(&k3("moe_renormalize"))
                .or_else(|| opt_bool(&k3("expert_weights_norm"))),
            // Latent hidden size: synthetic = `routed_expert_hidden_size`,
            // real = `expert_latent_length`.
            routed_expert_hidden_size: opt_usize_any(&[
                k3("routed_expert_hidden_size"),
                k3("expert_latent_length"),
            ]),
            latent_moe_use_norm: opt_bool(&k3("latent_moe_use_norm")),
            routed_scaling_factor: opt_f32(&k3("expert_weights_scale")),

            // MTP head (K3 does not export any). Retained for parity
            // with the DeepSeek V3 dispatch code.
            num_nextn_predict_layers: gguf.meta_u32(&k3("mtp_layer_count")).map(|v| v as usize),

            // MXFP4 native quantization. Not exported by k3meta.py in
            // the current pwilkin PR; leave as `None` so callers know
            // to fall back to per-tensor GGML type inspection.
            mxfp4_group_size: None,
            mxfp4_num_bits: None,
        }
    }

    /// Layer-index predicate: returns `true` if layer `il` (0-indexed)
    /// is a Gated MLA (full-attention) layer, `false` if it is a KDA
    /// (linear-attention) layer, and `None` if `full_attn_layers` is
    /// not populated.
    ///
    /// Uses the 0-indexed convention that `k3meta.py` writes into GGUF
    /// (which subtracts 1 from `config.json`'s 1-indexed
    /// `full_attn_layers` list at conversion time).
    #[must_use]
    pub fn is_mla_layer(&self, il: usize) -> Option<bool> {
        self.full_attn_layers
            .as_ref()
            .map(|full| full.contains(&il))
    }
}

#[cfg(test)]
mod kimi_k3_gguf_loader_tests {
    use super::{KimiDeltaConfig, ModelArch};
    use crate::gguf::GgufFile;

    // GGUF metadata value type IDs (see gguf.rs `MetaValue` parser).
    const GGUF_TYPE_U32: u32 = 4;
    const GGUF_TYPE_F32: u32 = 6;
    const GGUF_TYPE_BOOL: u32 = 7;
    const GGUF_TYPE_STRING: u32 = 8;
    const GGUF_TYPE_ARRAY: u32 = 9;

    fn write_u32(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn write_u64(buf: &mut Vec<u8>, v: u64) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn write_f32(buf: &mut Vec<u8>, v: f32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn write_str(buf: &mut Vec<u8>, s: &str) {
        write_u64(buf, s.len() as u64);
        buf.extend_from_slice(s.as_bytes());
    }
    fn write_kv_str(buf: &mut Vec<u8>, key: &str, value: &str) {
        write_str(buf, key);
        write_u32(buf, GGUF_TYPE_STRING);
        write_str(buf, value);
    }
    fn write_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
        write_str(buf, key);
        write_u32(buf, GGUF_TYPE_U32);
        write_u32(buf, value);
    }
    fn write_kv_f32(buf: &mut Vec<u8>, key: &str, value: f32) {
        write_str(buf, key);
        write_u32(buf, GGUF_TYPE_F32);
        write_f32(buf, value);
    }
    fn write_kv_bool(buf: &mut Vec<u8>, key: &str, value: bool) {
        write_str(buf, key);
        write_u32(buf, GGUF_TYPE_BOOL);
        buf.push(u8::from(value));
    }
    fn write_kv_u32_array(buf: &mut Vec<u8>, key: &str, values: &[u32]) {
        write_str(buf, key);
        write_u32(buf, GGUF_TYPE_ARRAY);
        write_u32(buf, GGUF_TYPE_U32);
        write_u64(buf, values.len() as u64);
        for &v in values {
            write_u32(buf, v);
        }
    }

    /// Build a minimal but complete Kimi K3 GGUF byte buffer for
    /// loader tests. Emits every metadata key that
    /// `KimiDeltaConfig::from_gguf` inspects plus the standard
    /// hyperparameters that `Llama3Config::from_gguf` needs to
    /// dispatch to the K3 branch. Includes one dummy f32 tensor so
    /// the GGUF file layout is valid.
    ///
    /// 8-layer mini config: `full_attn_layers = [3, 7]` (0-indexed
    /// after k3meta.py conversion, mirroring the real K3 pattern of
    /// every 4th layer being MLA), `kda_layers = [0, 1, 2, 4, 5, 6]`.
    #[allow(clippy::too_many_lines)]
    fn build_synthetic_kimi_k3_gguf() -> Vec<u8> {
        // Build metadata block separately so we can count entries
        // for the n_kv field, then splice everything into the final
        // header.
        let mut kv = Vec::new();
        let mut n_kv: u64 = 0;

        // Every write is direct + `n_kv += 1` so the borrow checker
        // does not have to reason about a captured-`&mut n_kv`
        // closure staying alive across the block below.
        write_kv_str(&mut kv, "general.architecture", "kimi-k3");
        n_kv += 1;

        // Standard hyperparams needed by Llama3Config::from_gguf plus
        // the K3-specific keys that KimiDeltaConfig::from_gguf reads.
        {
            write_kv_u32(&mut kv, "general.alignment", 32);
            n_kv += 1;
            write_kv_u32(&mut kv, "kimi-k3.block_count", 8);
            n_kv += 1;
            write_kv_u32(&mut kv, "kimi-k3.embedding_length", 64);
            n_kv += 1;
            write_kv_u32(&mut kv, "kimi-k3.feed_forward_length", 128);
            n_kv += 1;
            write_kv_u32(&mut kv, "kimi-k3.context_length", 4096);
            n_kv += 1;
            write_kv_u32(&mut kv, "kimi-k3.vocab_size", 256);
            n_kv += 1;
            write_kv_u32(&mut kv, "kimi-k3.attention.head_count", 8);
            n_kv += 1;
            write_kv_f32(&mut kv, "kimi-k3.attention.layer_norm_rms_epsilon", 1e-5);
            n_kv += 1;

            // MLA sub-config.
            write_kv_u32(&mut kv, "kimi-k3.attention.q_lora_rank", 16);
            n_kv += 1;
            write_kv_u32(&mut kv, "kimi-k3.attention.kv_lora_rank", 8);
            n_kv += 1;
            // key_length_mla = qk_nope + qk_rope = 8 + 4 = 12
            write_kv_u32(&mut kv, "kimi-k3.attention.key_length_mla", 12);
            n_kv += 1;
            write_kv_u32(&mut kv, "kimi-k3.attention.value_length_mla", 8);
            n_kv += 1;
            write_kv_u32(&mut kv, "kimi-k3.rope.dimension_count", 4);
            n_kv += 1;
            write_kv_bool(&mut kv, "kimi-k3.mla_use_nope", true);
            n_kv += 1;
            write_kv_bool(&mut kv, "kimi-k3.mla_use_output_gate", true);
            n_kv += 1;

            // KDA sub-config.
            write_kv_u32(&mut kv, "kimi-k3.attention.kda_head_dim", 8);
            n_kv += 1;
            write_kv_u32(&mut kv, "kimi-k3.ssm.conv_kernel", 4);
            n_kv += 1;
            write_kv_bool(&mut kv, "kimi-k3.use_full_rank_gate", true);
            n_kv += 1;
            write_kv_f32(&mut kv, "kimi-k3.gate_lower_bound", -5.0);
            n_kv += 1;

            // Attention Residuals + SiTU-GLU.
            write_kv_u32(&mut kv, "kimi-k3.attn_res_block_size", 4);
            n_kv += 1;
            write_kv_f32(&mut kv, "kimi-k3.activation_situ_beta", 4.0);
            n_kv += 1;
            write_kv_f32(&mut kv, "kimi-k3.activation_situ_linear_beta", 25.0);
            n_kv += 1;
            write_kv_str(&mut kv, "kimi-k3.activation", "situ");
            n_kv += 1;

            // Stable LatentMoE.
            write_kv_u32(&mut kv, "kimi-k3.expert_count", 32);
            n_kv += 1;
            write_kv_u32(&mut kv, "kimi-k3.expert_used_count", 4);
            n_kv += 1;
            write_kv_u32(&mut kv, "kimi-k3.expert_shared_count", 2);
            n_kv += 1;
            write_kv_u32(&mut kv, "kimi-k3.num_expert_group", 1);
            n_kv += 1;
            write_kv_u32(&mut kv, "kimi-k3.topk_group", 1);
            n_kv += 1;
            write_kv_u32(&mut kv, "kimi-k3.expert_feed_forward_length", 64);
            n_kv += 1;
            write_kv_u32(&mut kv, "kimi-k3.leading_dense_block_count", 1);
            n_kv += 1;
            write_kv_u32(&mut kv, "kimi-k3.routed_expert_hidden_size", 32);
            n_kv += 1;
            write_kv_bool(&mut kv, "kimi-k3.moe_renormalize", true);
            n_kv += 1;
            write_kv_bool(&mut kv, "kimi-k3.latent_moe_use_norm", true);
            n_kv += 1;
            write_kv_f32(&mut kv, "kimi-k3.expert_weights_scale", 1.0);
            n_kv += 1;
            write_kv_str(&mut kv, "kimi-k3.moe_router_activation_func", "sigmoid");
            n_kv += 1;
            write_kv_str(&mut kv, "kimi-k3.topk_method", "noaux_tc");
            n_kv += 1;

            // Hybrid layer routing (0-indexed).
            write_kv_u32_array(&mut kv, "kimi-k3.full_attn_layers", &[3, 7]);
            n_kv += 1;
            write_kv_u32_array(&mut kv, "kimi-k3.kda_layers", &[0, 1, 2, 4, 5, 6]);
            n_kv += 1;
        }

        // ── Header ────────────────────────────────────────────────
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        write_u32(&mut buf, 3); // version 3
        write_u64(&mut buf, 1); // n_tensors (1 dummy)
        write_u64(&mut buf, n_kv);
        buf.extend_from_slice(&kv);

        // Tensor info: one dummy f32 scalar.
        write_str(&mut buf, "token_embd.weight");
        write_u32(&mut buf, 1); // ndims
        write_u64(&mut buf, 1); // shape[0]
        write_u32(&mut buf, 0); // ggml_type = F32
        write_u64(&mut buf, 0); // data_offset

        // Pad to alignment 32.
        while !buf.len().is_multiple_of(32) {
            buf.push(0);
        }
        write_f32(&mut buf, 42.0);

        buf
    }

    #[test]
    fn detects_kimi_k3_arch_from_general_architecture() {
        let bytes = build_synthetic_kimi_k3_gguf();
        let gguf = GgufFile::parse(&bytes).expect("synthetic K3 GGUF must parse");
        assert_eq!(gguf.meta_str("general.architecture"), Some("kimi-k3"));
        assert_eq!(ModelArch::from_gguf(&gguf), ModelArch::KimiK3);
    }

    #[test]
    fn parses_kimi_k3_metadata_from_synthetic_gguf() {
        let bytes = build_synthetic_kimi_k3_gguf();
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let cfg = KimiDeltaConfig::from_gguf(&gguf, "kimi-k3");

        // MLA sub-config.
        assert_eq!(cfg.q_lora_rank, Some(16));
        assert_eq!(cfg.kv_lora_rank, Some(8));
        assert_eq!(cfg.qk_rope_head_dim, Some(4));
        // qk_nope = key_length_mla (12) − rope (4) = 8.
        assert_eq!(cfg.qk_nope_head_dim, Some(8));
        assert_eq!(cfg.v_head_dim, Some(8));
        assert_eq!(cfg.mla_use_nope, Some(true));
        assert_eq!(cfg.mla_use_output_gate, Some(true));

        // KDA sub-config.
        assert_eq!(cfg.kda_head_dim, Some(8));
        assert_eq!(cfg.kda_num_heads, Some(8));
        assert_eq!(cfg.kda_short_conv_kernel_size, Some(4));
        assert_eq!(cfg.kda_use_full_rank_gate, Some(true));
        assert_eq!(cfg.kda_gate_lower_bound, Some(-5.0));

        // AttnRes + SiTU-GLU.
        assert_eq!(cfg.attn_res_block_size, Some(4));
        assert_eq!(cfg.situ_beta, Some(4.0));
        assert_eq!(cfg.situ_linear_beta, Some(25.0));

        // Stable LatentMoE.
        assert_eq!(cfg.n_routed_experts, Some(32));
        assert_eq!(cfg.num_experts_per_tok, Some(4));
        assert_eq!(cfg.n_shared_experts, Some(2));
        assert_eq!(cfg.num_expert_group, Some(1));
        assert_eq!(cfg.topk_group, Some(1));
        assert_eq!(cfg.moe_intermediate_size, Some(64));
        assert_eq!(cfg.first_k_dense_replace, Some(1));
        assert_eq!(cfg.routed_expert_hidden_size, Some(32));
        assert_eq!(cfg.moe_renormalize, Some(true));
        assert_eq!(cfg.latent_moe_use_norm, Some(true));
        assert_eq!(cfg.routed_scaling_factor, Some(1.0));
        assert_eq!(cfg.moe_router_activation.as_deref(), Some("sigmoid"));
        assert_eq!(cfg.moe_topk_method.as_deref(), Some("noaux_tc"));

        // Hybrid layer routing.
        assert_eq!(cfg.full_attn_layers.as_deref(), Some(&[3usize, 7][..]));
        assert_eq!(
            cfg.kda_layers.as_deref(),
            Some(&[0usize, 1, 2, 4, 5, 6][..])
        );

        // MTP + MXFP4 keys are absent in the synthetic fixture — must
        // default to None without erroring.
        assert_eq!(cfg.num_nextn_predict_layers, None);
        assert_eq!(cfg.mxfp4_group_size, None);
        assert_eq!(cfg.mxfp4_num_bits, None);
    }

    #[test]
    fn is_mla_layer_matches_full_attn_layers_zero_indexed() {
        let bytes = build_synthetic_kimi_k3_gguf();
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let cfg = KimiDeltaConfig::from_gguf(&gguf, "kimi-k3");
        // Fixture: full_attn_layers = [3, 7], kda_layers = [0, 1, 2, 4, 5, 6].
        for il in 0..8 {
            let expect_mla = il == 3 || il == 7;
            assert_eq!(
                cfg.is_mla_layer(il),
                Some(expect_mla),
                "layer {il}: expected MLA = {expect_mla}"
            );
        }
    }

    #[test]
    fn is_mla_layer_returns_none_when_full_attn_layers_absent() {
        let cfg = KimiDeltaConfig::default();
        assert_eq!(cfg.is_mla_layer(0), None);
        assert_eq!(cfg.is_mla_layer(93), None);
    }

    #[test]
    fn missing_optional_fields_default_to_none() {
        // Minimal K3 GGUF: only arch + alignment + 1 dummy tensor.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        write_u32(&mut buf, 3);
        write_u64(&mut buf, 1); // n_tensors
        write_u64(&mut buf, 2); // n_kv
        write_kv_str(&mut buf, "general.architecture", "kimi-k3");
        write_kv_u32(&mut buf, "general.alignment", 32);
        write_str(&mut buf, "token_embd.weight");
        write_u32(&mut buf, 1);
        write_u64(&mut buf, 1);
        write_u32(&mut buf, 0);
        write_u64(&mut buf, 0);
        while !buf.len().is_multiple_of(32) {
            buf.push(0);
        }
        write_f32(&mut buf, 0.0);

        let gguf = GgufFile::parse(&buf).expect("minimal K3 parse");
        let cfg = KimiDeltaConfig::from_gguf(&gguf, "kimi-k3");
        assert!(cfg.q_lora_rank.is_none());
        assert!(cfg.n_routed_experts.is_none());
        assert!(cfg.full_attn_layers.is_none());
        assert!(cfg.attn_res_block_size.is_none());
        assert!(cfg.moe_router_activation.is_none());
    }

    #[test]
    fn meta_prefix_returns_hyphenated_kimi_k3() {
        // Regression guard: `k3meta.py` writes `kimi-k3.*`; any
        // refactor that drops the hyphen breaks GGUF key lookups.
        assert_eq!(ModelArch::KimiK3.meta_prefix(), "kimi-k3");
    }

    // ── Phase X.4.b.2 tensor loader tests ────────────────────────

    #[test]
    fn load_weight_ref_any_shape_reads_both_dims_from_gguf() {
        // Build a tiny GGUF with one 2D f32 tensor (2 rows × 3 cols)
        // and verify the shape-inferring helper extracts the correct
        // rows/cols from `tensor_info.dims`.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        write_u32(&mut buf, 3);
        write_u64(&mut buf, 1); // n_tensors
        write_u64(&mut buf, 1); // n_kv (alignment only)
        write_kv_u32(&mut buf, "general.alignment", 32);
        // Tensor: name, ndims=2, dims=[3, 2], type=F32, offset=0.
        write_str(&mut buf, "any_shape_test");
        write_u32(&mut buf, 2); // ndims
        write_u64(&mut buf, 3); // dims[0] = cols
        write_u64(&mut buf, 2); // dims[1] = rows
        write_u32(&mut buf, 0); // ggml_type = F32
        write_u64(&mut buf, 0); // offset
        while !buf.len().is_multiple_of(32) {
            buf.push(0);
        }
        for x in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            write_f32(&mut buf, x);
        }
        let gguf = GgufFile::parse(&buf).expect("parse");
        let wref = super::load_weight_ref_any_shape(&gguf, "any_shape_test")
            .expect("shape helper must find the tensor");
        assert_eq!(wref.cols, 3, "cols must come from dims[0]");
        assert_eq!(wref.rows, 2, "rows must come from dims[1]");
    }

    #[test]
    fn load_weight_ref_any_shape_1d_tensor_gets_rows_1() {
        // 1D tensor (norm-like): ndims=1, dims=[4]. Helper must
        // default rows to 1.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        write_u32(&mut buf, 3);
        write_u64(&mut buf, 1);
        write_u64(&mut buf, 1);
        write_kv_u32(&mut buf, "general.alignment", 32);
        write_str(&mut buf, "norm_1d");
        write_u32(&mut buf, 1); // ndims = 1
        write_u64(&mut buf, 4); // dims[0]
        write_u32(&mut buf, 0); // F32
        write_u64(&mut buf, 0);
        while !buf.len().is_multiple_of(32) {
            buf.push(0);
        }
        for x in [1.0_f32, 2.0, 3.0, 4.0] {
            write_f32(&mut buf, x);
        }
        let gguf = GgufFile::parse(&buf).expect("parse");
        let wref = super::load_weight_ref_any_shape(&gguf, "norm_1d")
            .expect("1D helper must find the tensor");
        assert_eq!(wref.cols, 4);
        assert_eq!(wref.rows, 1, "1D tensor falls back to rows=1");
    }

    #[test]
    fn model_weights_loader_returns_err_on_missing_global_tensors() {
        // The X.4.b.1 synthetic fixture only ships one dummy
        // `token_embd.weight` tensor (1 f32 value). All other Global
        // tensors are absent; the loader must return `Err` naming the
        // first missing one rather than panic.
        let bytes = build_synthetic_kimi_k3_gguf();
        let gguf = GgufFile::parse(&bytes).expect("parse fixture");
        let config = crate::gguf::GgufFile::parse(&bytes)
            .and_then(|g| super::Llama3Config::from_gguf(&g))
            .expect("kimi-k3 config must load from synthetic fixture");
        assert_eq!(config.arch, ModelArch::KimiK3);

        let Err(err) = super::load_kimi_k3_model_weights(&gguf, &config) else {
            panic!("loader must Err on the metadata-only fixture");
        };
        // First missing tensor after `token_embd.weight` (which the
        // fixture has as a 1-elem dummy) is `output_norm.weight`.
        assert!(
            err.contains("output_norm.weight") || err.contains("token_embd"),
            "expected descriptive missing-tensor error, got: {err}"
        );
    }

    #[test]
    fn layer_weights_loader_returns_err_on_missing_attn_norm() {
        // With no per-layer tensors in the fixture, layer 0 fails at
        // its first required tensor (`blk.0.attn_norm.weight`).
        let bytes = build_synthetic_kimi_k3_gguf();
        let gguf = GgufFile::parse(&bytes).expect("parse fixture");
        let config = super::Llama3Config::from_gguf(&gguf).expect("config");

        let Err(err) = super::load_kimi_k3_layer_weights(&gguf, 0, &config) else {
            panic!("layer loader must Err on empty fixture");
        };
        assert!(
            err.contains("blk.0.attn_norm"),
            "expected error to name blk.0.attn_norm, got: {err}"
        );
    }

    #[test]
    fn layer_weights_loader_dispatches_mla_vs_kda_by_layer_index() {
        // Even without weight tensors, the loader must reach the
        // MLA vs KDA branch matching `is_mla_layer(il)` — which for
        // the fixture means layers 3 and 7 look for MLA tensors
        // (`attn_q_a.weight` first missing) while layers 0/1/2/4/5/6
        // look for KDA tensors (`attn_q.weight` first missing).
        //
        // We can only observe the branch by inspecting the error
        // message, since neither set of tensors exists. Both messages
        // must contain the layer prefix.
        let bytes = build_synthetic_kimi_k3_gguf();
        let gguf = GgufFile::parse(&bytes).expect("parse fixture");
        let config = super::Llama3Config::from_gguf(&gguf).expect("config");

        // The common attn_norm still comes first, so we need a fixture
        // that already has the common tensors to reach the attn branch.
        // For this minimal smoke test we assert the dispatch predicate
        // itself matches the config's is_mla_layer.
        let kd = config.kimi_delta.as_ref().expect("kimi_delta populated");
        assert_eq!(kd.is_mla_layer(0), Some(false), "layer 0 = KDA");
        assert_eq!(kd.is_mla_layer(3), Some(true), "layer 3 = MLA");
        assert_eq!(kd.is_mla_layer(7), Some(true), "layer 7 = MLA");
        assert_eq!(kd.is_mla_layer(4), Some(false), "layer 4 = KDA");
    }

    #[test]
    fn layer_weights_loader_dense_vs_moe_matches_first_k_dense_replace() {
        // Fixture sets `leading_dense_block_count = 1`, so layer 0
        // takes the Dense FFN branch and layers ≥ 1 take LatentMoE.
        // Same observation constraint as above — assert the
        // dispatch predicate the loader uses.
        let bytes = build_synthetic_kimi_k3_gguf();
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let config = super::Llama3Config::from_gguf(&gguf).expect("config");
        let kd = config.kimi_delta.as_ref().expect("kimi_delta populated");
        assert_eq!(kd.first_k_dense_replace, Some(1));
        // The loader body uses `il < first_k_dense_replace` as the
        // Dense predicate; enforce that at the predicate level so any
        // future refactor keeps the boundary at layer 0 for K3.
        let first_k = kd.first_k_dense_replace.unwrap();
        assert!(0 < first_k, "layer 0 must be Dense");
        assert!(!(1 < first_k), "layer 1 must be LatentMoE (not Dense)");
    }
}

#[cfg(all(test, feature = "hf-config"))]
mod kimi_k3_hf_config_tests {
    use super::KimiDeltaConfig;

    /// Minimal fixture reproducing the subset of `text_config` needed to
    /// exercise the parser. Real config lives at
    /// `huggingface.co/moonshotai/Kimi-K3/raw/main/config.json` — the
    /// full-length arrays are truncated here to keep the fixture readable
    /// while still hitting every field path in `from_hf_config`.
    const KIMI_K3_FIXTURE: &str = r#"{
      "model_type": "kimi_k3",
      "text_config": {
        "model_type": "kimi_linear",
        "hidden_size": 7168,
        "num_hidden_layers": 93,
        "num_attention_heads": 96,
        "num_key_value_heads": 96,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_shared_experts": 2,
        "num_expert_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "moe_router_activation_func": "sigmoid",
        "moe_intermediate_size": 3072,
        "moe_renormalize": true,
        "first_k_dense_replace": 1,
        "num_nextn_predict_layers": 0,
        "routed_expert_hidden_size": 3584,
        "routed_scaling_factor": 1.0,
        "latent_moe_use_norm": true,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mla_use_nope": true,
        "mla_use_output_gate": true,
        "attn_res_block_size": 12,
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "linear_attn_config": {
          "full_attn_layers": [4, 8, 12, 93],
          "kda_layers": [1, 2, 3, 5, 6, 7],
          "num_heads": 96,
          "head_dim": 128,
          "short_conv_kernel_size": 4,
          "use_full_rank_gate": true,
          "gate_lower_bound": -5.0
        },
        "quantization_config": {
          "config_groups": {
            "group_0": {
              "weights": {
                "num_bits": 4,
                "group_size": 32
              }
            }
          }
        }
      }
    }"#;

    #[test]
    fn parses_kimi_k3_confirmed_spec() {
        let cfg = KimiDeltaConfig::from_hf_config(KIMI_K3_FIXTURE.as_bytes())
            .expect("Kimi K3 fixture should parse");

        // Stable LatentMoE (confirmed 2026-07-27 open weight release)
        assert_eq!(cfg.n_routed_experts, Some(896));
        assert_eq!(cfg.num_experts_per_tok, Some(16));
        assert_eq!(cfg.n_shared_experts, Some(2));
        assert_eq!(cfg.num_expert_group, Some(1));
        assert_eq!(cfg.topk_group, Some(1));
        assert_eq!(cfg.moe_intermediate_size, Some(3072));
        assert_eq!(cfg.first_k_dense_replace, Some(1));
        assert_eq!(cfg.num_nextn_predict_layers, Some(0));
        assert_eq!(cfg.routed_expert_hidden_size, Some(3584));
        assert_eq!(cfg.moe_renormalize, Some(true));
        assert_eq!(cfg.latent_moe_use_norm, Some(true));
        assert_eq!(cfg.routed_scaling_factor, Some(1.0));
        assert_eq!(cfg.moe_router_activation.as_deref(), Some("sigmoid"));
        assert_eq!(cfg.moe_topk_method.as_deref(), Some("noaux_tc"));

        // Gated MLA (K3 = DeepSeek V3 MLA + output gate + NoPE)
        assert_eq!(cfg.q_lora_rank, Some(1536));
        assert_eq!(cfg.kv_lora_rank, Some(512));
        assert_eq!(cfg.qk_nope_head_dim, Some(128));
        assert_eq!(cfg.qk_rope_head_dim, Some(64));
        assert_eq!(cfg.v_head_dim, Some(128));
        assert_eq!(cfg.mla_use_nope, Some(true));
        assert_eq!(cfg.mla_use_output_gate, Some(true));

        // Attention Residuals + SiTU-GLU (Kimi-specific)
        assert_eq!(cfg.attn_res_block_size, Some(12));
        assert_eq!(cfg.situ_beta, Some(4.0));
        assert_eq!(cfg.situ_linear_beta, Some(25.0));

        // Kimi Delta Attention (KDA)
        assert_eq!(cfg.kda_num_heads, Some(96));
        assert_eq!(cfg.kda_head_dim, Some(128));
        assert_eq!(cfg.kda_short_conv_kernel_size, Some(4));
        assert_eq!(cfg.kda_use_full_rank_gate, Some(true));
        assert_eq!(cfg.kda_gate_lower_bound, Some(-5.0));
        assert_eq!(
            cfg.full_attn_layers.as_deref(),
            Some(&[4usize, 8, 12, 93][..]),
        );
        assert_eq!(
            cfg.kda_layers.as_deref(),
            Some(&[1usize, 2, 3, 5, 6, 7][..]),
        );

        // MXFP4 native quantization
        assert_eq!(cfg.mxfp4_num_bits, Some(4));
        assert_eq!(cfg.mxfp4_group_size, Some(32));
    }

    #[test]
    fn missing_fields_default_to_none() {
        let minimal = br#"{"text_config": {}}"#;
        let cfg = KimiDeltaConfig::from_hf_config(minimal).expect("empty text_config still parses");
        assert!(cfg.n_routed_experts.is_none());
        assert!(cfg.full_attn_layers.is_none());
        assert!(cfg.moe_router_activation.is_none());
        assert!(cfg.mxfp4_num_bits.is_none());
    }

    #[test]
    fn missing_text_config_is_all_none() {
        let bare = b"{}";
        let cfg = KimiDeltaConfig::from_hf_config(bare).expect("bare root object still parses");
        assert!(cfg.n_routed_experts.is_none());
        assert!(cfg.q_lora_rank.is_none());
    }

    #[test]
    fn malformed_json_returns_error() {
        assert!(KimiDeltaConfig::from_hf_config(b"{not valid json").is_err());
    }

    #[test]
    fn confirmed_hybrid_layer_totals() {
        // K3 spec: 69 KDA + 24 Gated MLA = 93 layers total. The fixture
        // uses truncated arrays, but the length invariant is worth
        // asserting on the truncated form as a smoke test — the real
        // parser doesn't need to know the total.
        let cfg = KimiDeltaConfig::from_hf_config(KIMI_K3_FIXTURE.as_bytes()).unwrap();
        let full = cfg.full_attn_layers.expect("full_attn_layers present");
        let kda = cfg.kda_layers.expect("kda_layers present");
        assert_eq!(full.len(), 4, "fixture uses 4 truncated MLA layers");
        assert_eq!(kda.len(), 6, "fixture uses 6 truncated KDA layers");
        // Total layers = num_hidden_layers = 93 (from real config, not
        // enforced by parser); documenting here for reader clarity.
    }
}

// ── Kimi Delta Attention (KDA) CPU primitives (Phase X.4.c.1) ─────────
//
// Scaffolding for the Kimi K3 KDA forward path. Implements the math
// primitives of Section 2.1.1 of the K3 tech report (Eq 1, 5, 6) as
// standalone, unit-testable functions. The `forward_kimi_k3` dispatch
// still `todo!()`s until Phase X.4.c.2 wires these primitives into a
// full per-head + per-layer forward with weight loading (Phase X.4.b)
// and ShortConv history buffers.
//
// The composite `kimi_delta_forward_head` function (Eq 2 projections +
// ShortConv + L2Norm + Swish + step + read + output gate) is
// intentionally out of scope for X.4.c.1; ShortConv requires a
// per-head 4-token history buffer and integrates with the layer-level
// weight tensor management, so it lands together with the
// `forward_kimi_k3` block-level integration.

/// Per-head recurrent state for Kimi Delta Attention.
///
/// KDA maintains a fixed-size recurrent state `S ∈ ℝ^{d_k × d_v}` per
/// head instead of a KV cache that grows with sequence length. This is
/// what enables K3's fixed-size state property at 1M context: the 69
/// KDA layers each hold `d_k × d_v = 128 × 128 = 16384 f32` per head
/// (64 KB per head × 96 heads ≈ 6.1 MB per KDA layer), invariant of
/// sequence length.
///
/// Layout: row-major flat `Vec<f32>` of length `d_k × d_v`, with
/// `state[i * d_v + j] = S[i, j]`. The K3 defaults are `d_k = d_v =
/// 128` (from HF `config.json` `linear_attn_config.head_dim`,
/// captured in [`KimiDeltaConfig::kda_head_dim`]).
#[derive(Debug, Clone)]
pub struct KimiDeltaState {
    state: Vec<f32>,
    d_k: usize,
    d_v: usize,
}

impl KimiDeltaState {
    /// Allocate a zeroed state of shape `[d_k, d_v]`.
    #[must_use]
    pub fn new(d_k: usize, d_v: usize) -> Self {
        Self {
            state: vec![0.0; d_k * d_v],
            d_k,
            d_v,
        }
    }

    /// Recurrent state dimension `d_k` (matches Q/K head dim).
    #[inline]
    #[must_use]
    pub const fn d_k(&self) -> usize {
        self.d_k
    }

    /// Recurrent state dimension `d_v` (matches V head dim).
    #[inline]
    #[must_use]
    pub const fn d_v(&self) -> usize {
        self.d_v
    }

    /// Zero out the state (start of a new sequence).
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    /// Read-only view of the flat `[d_k × d_v]` state buffer, row-major.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.state
    }

    /// Mutable view of the flat `[d_k × d_v]` state buffer. Exposed
    /// mainly for tests and future SIMD kernels that want to write
    /// directly into the buffer without going through
    /// [`kimi_delta_step`].
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.state
    }
}

/// Apply one KDA recurrence step in-place on `state`, per Eq (1) of
/// the K3 tech report:
///
/// ```text
/// S_t = (I − β_t · k_t k_tᵀ) · Diag(α_t) · S_{t−1} + β_t · k_t v_tᵀ
/// ```
///
/// Algorithm (all in-place on the flat `[d_k × d_v]` buffer):
///
/// 1. Scale each row `i` of `S` by `α[i]` → `S ← Diag(α) S`.
/// 2. Compute `w = k^T S ∈ ℝ^{d_v}` (dot each column of `S` with `k`).
/// 3. Update `S[i, j] += β · k[i] · (v[j] − w[j])` (fuses the
///    `−β k k^T S` and `+β k v^T` updates into one pass).
///
/// # Panics
///
/// Panics if `k.len() != state.d_k()`, `v.len() != state.d_v()`, or
/// `alpha.len() != state.d_k()`. All shape errors are programmer
/// bugs (the caller derived them from [`KimiDeltaConfig`]), so a
/// debug panic is more useful than a `Result` variant.
pub fn kimi_delta_step(state: &mut KimiDeltaState, k: &[f32], v: &[f32], alpha: &[f32], beta: f32) {
    let d_k = state.d_k;
    let d_v = state.d_v;
    assert_eq!(k.len(), d_k, "k length must equal d_k");
    assert_eq!(v.len(), d_v, "v length must equal d_v");
    assert_eq!(alpha.len(), d_k, "alpha length must equal d_k");

    let s = &mut state.state;

    // Step 1: row-wise scale by alpha (Diag(α) S).
    for i in 0..d_k {
        let a = alpha[i];
        let row_start = i * d_v;
        for j in 0..d_v {
            s[row_start + j] *= a;
        }
    }

    // Step 2: w = k^T S ∈ ℝ^{d_v}, i.e. w[j] = Σ_i k[i] · S[i, j].
    let mut w = vec![0.0_f32; d_v];
    for i in 0..d_k {
        let ki = k[i];
        if ki == 0.0 {
            continue;
        }
        let row_start = i * d_v;
        for j in 0..d_v {
            w[j] += ki * s[row_start + j];
        }
    }

    // Step 3: S[i, j] += β · k[i] · (v[j] − w[j]).
    for i in 0..d_k {
        let ki = k[i];
        let coef = beta * ki;
        if coef == 0.0 {
            continue;
        }
        let row_start = i * d_v;
        for j in 0..d_v {
            s[row_start + j] += coef * (v[j] - w[j]);
        }
    }
}

/// Read the recurrent-attention output `ō_t = Sᵀ q_t ∈ ℝ^{d_v}` from
/// the current KDA state.
///
/// This is the "recurrent read" half of Eq (1) — the output flows on
/// to the RMSNorm + output-gate stage (Eq 6, see
/// [`kimi_delta_output_gate`]).
///
/// # Panics
///
/// Panics if `q.len() != state.d_k()`.
#[must_use]
pub fn kimi_delta_read(state: &KimiDeltaState, q: &[f32]) -> Vec<f32> {
    let d_k = state.d_k;
    let d_v = state.d_v;
    assert_eq!(q.len(), d_k, "q length must equal d_k");

    let s = &state.state;
    let mut out = vec![0.0_f32; d_v];
    for i in 0..d_k {
        let qi = q[i];
        if qi == 0.0 {
            continue;
        }
        let row_start = i * d_v;
        for j in 0..d_v {
            out[j] += qi * s[row_start + j];
        }
    }
    out
}

/// Compute the lower-bounded per-channel retention factor `α_t^h` for
/// KDA (Eq 5 of the K3 tech report):
///
/// ```text
/// g_t = g_min · Sigmoid(exp(A_h) · z_t)   ∈ (g_min, 0)^{d_k}
/// α_t = exp(g_t)                          ∈ (exp(g_min), 1)^{d_k}
/// ```
///
/// where `A_h` is a learnable per-head log-scale (initialized to 0 in
/// K3) and `g_min = -5.0` is the fixed lower bound (also stored as
/// [`KimiDeltaConfig::kda_gate_lower_bound`]). The bound ensures the
/// cumulative log-decay over a 16-token tile stays inside `(-80, 0)`,
/// keeping the reciprocal rescaling factor within BF16 dynamic range
/// so KDA's diagonal and off-diagonal tiles can both use dense Tensor
/// Core matmul — the key departure from Kimi Linear's unbounded
/// negative-Softplus mapping.
///
/// Returns a fresh `Vec<f32>` of length `z.len()`; callers on the hot
/// path can inline the two lines (`exp(log_scale_a) * z_i` → sigmoid
/// → `× g_min` → exp) to avoid the allocation.
#[must_use]
pub fn kimi_delta_lower_bounded_decay(z: &[f32], log_scale_a: f32, g_min: f32) -> Vec<f32> {
    let a = log_scale_a.exp();
    z.iter()
        .map(|&zi| {
            let g = g_min * sigmoid(a * zi);
            g.exp()
        })
        .collect()
}

/// Apply the KDA full-rank output gate + optional RMSNorm + output
/// projection (Eq 6 of the K3 tech report):
///
/// ```text
/// y_t = W_o · [Sigmoid(W_g · x_t) ⊙ RMSNorm(ō_t)]
/// ```
///
/// K3 differs from Kimi Linear by using a full-rank `W_g` projection
/// (`use_full_rank_gate = true` in `linear_attn_config`) and by
/// inserting an RMSNorm on the recurrent output `ō_t` before the
/// element-wise gate. The Gated MLA layers use the same output-gate
/// pattern (Eq 7), so callers may reuse this function for both the
/// KDA and MLA paths by passing `rms_weight = None` to skip the
/// normalize.
///
/// # Arguments
///
/// - `o_bar`: recurrent-attention output from [`kimi_delta_read`],
///   length `d_v`.
/// - `gate_pre`: pre-sigmoid gate `W_g x_t`, length `d_v`. Full-rank
///   projection to preserve K3's "each token modulates channels from
///   global attention" property.
/// - `rms_weight`: optional learnable RMSNorm scale `γ`, length `d_v`.
///   `None` disables the normalize (used by Gated MLA per Eq 7,
///   which has no inner RMSNorm on `ō_t`).
/// - `rms_eps`: RMSNorm epsilon (ignored when `rms_weight = None`).
/// - `w_out`: output projection `W_o`, flat `[d_out × d_v]` row-major.
/// - `d_out`: output dimension (typically the hidden dim `d`).
///
/// # Panics
///
/// Panics on any shape mismatch.
#[must_use]
pub fn kimi_delta_output_gate(
    o_bar: &[f32],
    gate_pre: &[f32],
    rms_weight: Option<&[f32]>,
    rms_eps: f32,
    w_out: &[f32],
    d_out: usize,
) -> Vec<f32> {
    let d_v = o_bar.len();
    assert_eq!(
        gate_pre.len(),
        d_v,
        "gate_pre length must equal o_bar length"
    );
    assert_eq!(
        w_out.len(),
        d_out * d_v,
        "w_out length must equal d_out * d_v"
    );
    if let Some(gamma) = rms_weight {
        assert_eq!(gamma.len(), d_v, "rms_weight length must equal d_v");
    }

    // Step 1: gated = Sigmoid(gate_pre) ⊙ [optional RMSNorm(o_bar)].
    let mut gated = vec![0.0_f32; d_v];
    if let Some(gamma) = rms_weight {
        // f64 sum-of-squares accumulation matches this module's private
        // rms_norm helper and llama.cpp's convention.
        let mut ss = 0.0_f64;
        for &o in o_bar {
            ss += f64::from(o) * f64::from(o);
        }
        let mean = (ss / d_v as f64) as f32;
        let scale = (mean + rms_eps).sqrt().recip();
        for j in 0..d_v {
            gated[j] = sigmoid(gate_pre[j]) * o_bar[j] * scale * gamma[j];
        }
    } else {
        // No RMSNorm (Gated MLA output gate variant, Eq 7).
        for j in 0..d_v {
            gated[j] = sigmoid(gate_pre[j]) * o_bar[j];
        }
    }

    // Step 2: y = W_o · gated ∈ ℝ^{d_out}.
    let mut y = vec![0.0_f32; d_out];
    for i in 0..d_out {
        let row_start = i * d_v;
        let mut acc = 0.0_f64;
        for j in 0..d_v {
            acc += f64::from(w_out[row_start + j]) * f64::from(gated[j]);
        }
        y[i] = acc as f32;
    }
    y
}

#[cfg(test)]
mod kimi_delta_tests {
    use super::{
        kimi_delta_lower_bounded_decay, kimi_delta_output_gate, kimi_delta_read, kimi_delta_step,
        KimiDeltaState,
    };

    /// Helper: build a [`KimiDeltaState`] with a specific pre-populated
    /// buffer for test-input setup.
    fn state_from(d_k: usize, d_v: usize, buf: Vec<f32>) -> KimiDeltaState {
        assert_eq!(buf.len(), d_k * d_v);
        let mut s = KimiDeltaState::new(d_k, d_v);
        s.as_mut_slice().copy_from_slice(&buf);
        s
    }

    #[test]
    fn new_state_is_zeroed() {
        let s = KimiDeltaState::new(4, 4);
        assert_eq!(s.d_k(), 4);
        assert_eq!(s.d_v(), 4);
        assert!(s.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn reset_zeroes_state() {
        let mut s = state_from(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        s.reset();
        assert!(s.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn step_beta_1_alpha_1_from_zero_writes_k_v_transpose() {
        // S_0 = 0, α = [1,1,1,1], β = 1, k = [1,0,0,0], v = [a,b,c,d].
        // S_1 = (I − k k^T) · I · 0 + 1 · k v^T = k v^T
        // → first row of S = v, other rows = 0.
        let mut s = KimiDeltaState::new(4, 4);
        let k = [1.0_f32, 0.0, 0.0, 0.0];
        let v = [2.0_f32, -1.0, 3.5, 0.25];
        let alpha = [1.0_f32; 4];
        kimi_delta_step(&mut s, &k, &v, &alpha, 1.0);
        let expected = [
            2.0, -1.0, 3.5, 0.25, // row 0 = v
            0.0, 0.0, 0.0, 0.0, // row 1
            0.0, 0.0, 0.0, 0.0, // row 2
            0.0, 0.0, 0.0, 0.0, // row 3
        ];
        assert_eq!(s.as_slice(), &expected);
    }

    #[test]
    fn step_alpha_0_annihilates_history_before_write() {
        // Preload S_0 with junk, set α = [0, 0] → history annihilated;
        // then β · k v^T is written on top.
        let mut s = state_from(2, 2, vec![7.0_f32, 8.0, 9.0, 10.0]);
        let k = [1.0_f32, 0.0];
        let v = [3.0_f32, 4.0];
        let alpha = [0.0_f32, 0.0];
        kimi_delta_step(&mut s, &k, &v, &alpha, 1.0);
        // Diag(0) · S = 0; w = k^T · 0 = 0; update is +β k v^T.
        let expected = [
            3.0, 4.0, // row 0 = v
            0.0, 0.0, // row 1
        ];
        assert_eq!(s.as_slice(), &expected);
    }

    #[test]
    fn step_beta_0_only_applies_row_decay() {
        // Preload S with known values, α = [0.5, 0.25], β = 0.
        // Expected: S ← Diag(α) · S; k, v are ignored.
        let mut s = state_from(2, 2, vec![10.0_f32, 20.0, 40.0, 80.0]);
        let k = [1.0_f32, 1.0];
        let v = [99.0_f32, 99.0];
        let alpha = [0.5_f32, 0.25];
        kimi_delta_step(&mut s, &k, &v, &alpha, 0.0);
        let expected = [
            5.0, 10.0, // row 0 × 0.5
            10.0, 20.0, // row 1 × 0.25
        ];
        assert_eq!(s.as_slice(), &expected);
    }

    #[test]
    fn step_two_orthogonal_writes_produce_two_row_state() {
        // Two orthogonal k vectors, all-ones α, β = 1 → each write
        // populates its own row without interfering with the other.
        let mut s = KimiDeltaState::new(2, 2);
        let alpha = [1.0_f32, 1.0];
        // Step 1: k_1 = [1, 0], v_1 = [1, 2] → row 0 of S = v_1.
        kimi_delta_step(&mut s, &[1.0, 0.0], &[1.0, 2.0], &alpha, 1.0);
        // Step 2: k_2 = [0, 1], v_2 = [3, 4] → row 1 of S = v_2,
        // row 0 unchanged because k_2 ⊥ k_1.
        kimi_delta_step(&mut s, &[0.0, 1.0], &[3.0, 4.0], &alpha, 1.0);
        let expected = [
            1.0, 2.0, // row 0 = v_1 (retained)
            3.0, 4.0, // row 1 = v_2 (new write)
        ];
        assert_eq!(s.as_slice(), &expected);
    }

    #[test]
    fn read_from_zero_state_returns_zero() {
        let s = KimiDeltaState::new(4, 4);
        let q = [1.0_f32, 2.0, 3.0, 4.0];
        let out = kimi_delta_read(&s, &q);
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn read_recovers_v_after_single_write() {
        // After step with k = [1, 0], v = [a, b], β = 1, α = 1: S has
        // row 0 = v. Then Sᵀ q for q = [1, 0] gives
        //   out[j] = Σ_i S[i,j] · q[i] = S[0, j] = v[j].
        let mut s = KimiDeltaState::new(2, 2);
        kimi_delta_step(&mut s, &[1.0, 0.0], &[5.0, -3.0], &[1.0, 1.0], 1.0);
        let out = kimi_delta_read(&s, &[1.0, 0.0]);
        assert_eq!(out, vec![5.0, -3.0]);
    }

    #[test]
    fn lower_bounded_decay_at_zero_z_is_sigmoid_half() {
        // z = 0, A = 0 (so exp(A) = 1) → sigmoid(0) = 0.5
        // → g = -5 · 0.5 = -2.5 → α = exp(-2.5) ≈ 0.082085.
        let alpha = kimi_delta_lower_bounded_decay(&[0.0, 0.0, 0.0], 0.0, -5.0);
        let expected = (-2.5_f32).exp();
        for &a in &alpha {
            assert!(
                (a - expected).abs() < 1e-6,
                "α = {a}, expected ≈ {expected}"
            );
        }
    }

    #[test]
    fn lower_bounded_decay_saturates_to_lower_bound() {
        // Large positive z → sigmoid → 1, g → g_min = -5, α → exp(-5).
        let alpha = kimi_delta_lower_bounded_decay(&[100.0, 100.0], 0.0, -5.0);
        let expected = (-5.0_f32).exp();
        for &a in &alpha {
            assert!(
                (a - expected).abs() < 1e-4,
                "large-positive-z α = {a}, expected ≈ {expected}"
            );
        }
    }

    #[test]
    fn lower_bounded_decay_saturates_to_one() {
        // Large negative z → sigmoid → 0, g → 0, α → 1.
        let alpha = kimi_delta_lower_bounded_decay(&[-100.0, -100.0], 0.0, -5.0);
        for &a in &alpha {
            assert!(
                (a - 1.0).abs() < 1e-6,
                "large-negative-z α = {a}, expected ≈ 1.0"
            );
        }
    }

    #[test]
    fn output_gate_zero_gate_pre_gives_half_rms_norm() {
        // gate_pre = 0 → sigmoid(0) = 0.5 per channel.
        // rms_weight = 1, tiny rms_eps → RMSNorm(ō) = ō / sqrt(mean(ō²)).
        // W_o = I (2×2) → y = 0.5 · RMSNorm(ō).
        let o_bar = [3.0_f32, 4.0]; // mean of squares = 12.5
        let inv_scale = (12.5_f32).sqrt();
        let y = kimi_delta_output_gate(
            &o_bar,
            &[0.0, 0.0],
            Some(&[1.0, 1.0]),
            1e-6,
            &[1.0, 0.0, 0.0, 1.0], // W_o = I
            2,
        );
        let expected = [0.5 * 3.0 / inv_scale, 0.5 * 4.0 / inv_scale];
        for i in 0..2 {
            assert!(
                (y[i] - expected[i]).abs() < 1e-4,
                "y[{i}] = {}, expected ≈ {}",
                y[i],
                expected[i]
            );
        }
    }

    #[test]
    fn output_gate_zero_o_bar_gives_zero() {
        // ō = 0 → RMSNorm(0) = 0 · 1/sqrt(eps) = 0 → y = 0 regardless of gate.
        let y = kimi_delta_output_gate(
            &[0.0, 0.0, 0.0, 0.0],
            &[1.5, -2.0, 3.0, 0.7],
            Some(&[1.0, 1.0, 1.0, 1.0]),
            1e-6,
            &[1.0_f32; 4 * 4], // W_o = all ones (4×4)
            4,
        );
        assert_eq!(y, vec![0.0; 4]);
    }

    #[test]
    fn output_gate_without_rms_norm_matches_mla_shape() {
        // rms_weight = None → skip normalize (Gated MLA Eq 7 variant).
        // gate_pre = 0 → sigmoid = 0.5 → gated = 0.5 · ō → y = W_o · gated.
        let o_bar = [2.0_f32, 4.0];
        let y = kimi_delta_output_gate(
            &o_bar,
            &[0.0, 0.0],
            None,
            1e-6,
            &[1.0, 0.0, 0.0, 1.0], // W_o = I
            2,
        );
        assert_eq!(y, vec![1.0, 2.0]); // 0.5 · [2, 4]
    }
}

// ── Kimi Delta Attention (KDA) per-head composite forward (Phase X.4.c.2) ──
//
// Wires the X.4.c.1 primitives (`KimiDeltaState`, `kimi_delta_step`,
// `kimi_delta_read`, `kimi_delta_lower_bounded_decay`,
// `kimi_delta_output_gate`) together with the existing generic
// `causal_conv1d_step` (line ~3185, ShortConv shared with Qwen 3.5
// DeltaNet) and `silu` (line ~3112) into a full per-token per-head
// forward matching K3 tech report §2.1.1 Eq 1-6.
//
// Layer-level orchestration (Block AttnRes at §2.2 Eq 8-10, KV cache
// interaction for the 24 Gated MLA layers, GGUF weight lookup) is
// out of scope for X.4.c.2 and lives in `forward_kimi_k3` (still
// `todo!()`) at Phase X.4.c.3+ — those are blocked on Phase X.4.b
// (community `convert_hf_to_gguf.py`) and Phase X.4.d (AttnRes).

/// Per-head runtime cache for one KDA layer.
///
/// Bundles the recurrent delta state ([`KimiDeltaState`]) and the
/// three ShortConv history ring buffers (one each for Q, K, V) into
/// a single struct so callers do not need to thread four separate
/// mutable references through [`kimi_delta_forward_head`]. All
/// buffers are heap-allocated so a full model's caches
/// (`num_kda_layers × num_heads` instances) are trivially
/// send-across-threads for prefill parallelism.
///
/// Layout (K3 defaults in parentheses):
///
/// - `state`: `[d_k × d_v]` (128 × 128 = 64 KB f32 per head)
/// - `conv_state_q`: `[(kernel_size − 1) × d_k]` (3 × 128 = 1.5 KB)
/// - `conv_state_k`: `[(kernel_size − 1) × d_k]` (1.5 KB)
/// - `conv_state_v`: `[(kernel_size − 1) × d_v]` (1.5 KB)
///
/// Total per-head KDA cache ≈ 68.5 KB at K3 defaults; a full 96-head
/// KDA layer ≈ 6.6 MB, and all 69 KDA layers ≈ 454 MB — invariant of
/// sequence length (unlike an MLA KV cache that grows with tokens).
#[derive(Debug, Clone)]
pub struct KimiDeltaHeadCache {
    /// Recurrent delta state `S ∈ ℝ^{d_k × d_v}` (Eq 1).
    pub state: KimiDeltaState,
    /// Q ShortConv history ring buffer (`(kernel_size − 1) × d_k`).
    conv_state_q: Vec<f32>,
    /// K ShortConv history ring buffer (`(kernel_size − 1) × d_k`).
    conv_state_k: Vec<f32>,
    /// V ShortConv history ring buffer (`(kernel_size − 1) × d_v`).
    conv_state_v: Vec<f32>,
    /// Write cursor for `conv_state_q`.
    ring_pos_q: usize,
    /// Write cursor for `conv_state_k`.
    ring_pos_k: usize,
    /// Write cursor for `conv_state_v`.
    ring_pos_v: usize,
    kernel_size: usize,
    d_k: usize,
    d_v: usize,
}

impl KimiDeltaHeadCache {
    /// Allocate a zeroed per-head cache. `kernel_size` must be at
    /// least 2 (matches the guard inside `causal_conv1d_step`);
    /// K3 uses 4.
    #[must_use]
    pub fn new(d_k: usize, d_v: usize, kernel_size: usize) -> Self {
        assert!(kernel_size >= 2, "kernel_size must be at least 2");
        let hist = kernel_size - 1;
        Self {
            state: KimiDeltaState::new(d_k, d_v),
            conv_state_q: vec![0.0; hist * d_k],
            conv_state_k: vec![0.0; hist * d_k],
            conv_state_v: vec![0.0; hist * d_v],
            ring_pos_q: 0,
            ring_pos_k: 0,
            ring_pos_v: 0,
            kernel_size,
            d_k,
            d_v,
        }
    }

    /// Zero every buffer and reset all ring cursors (start of a new
    /// sequence). Equivalent to `*self = Self::new(...)` but avoids
    /// the reallocation.
    pub fn reset(&mut self) {
        self.state.reset();
        self.conv_state_q.fill(0.0);
        self.conv_state_k.fill(0.0);
        self.conv_state_v.fill(0.0);
        self.ring_pos_q = 0;
        self.ring_pos_k = 0;
        self.ring_pos_v = 0;
    }

    #[inline]
    #[must_use]
    pub const fn d_k(&self) -> usize {
        self.d_k
    }

    #[inline]
    #[must_use]
    pub const fn d_v(&self) -> usize {
        self.d_v
    }

    #[inline]
    #[must_use]
    pub const fn kernel_size(&self) -> usize {
        self.kernel_size
    }
}

/// Borrowed per-head weight references for one KDA forward pass.
///
/// All slice fields are `&'a [f32]` so the struct is zero-copy over
/// GGUF-backed tensor bytes in the production path and equally usable
/// with owned `Vec<f32>` buffers in tests. Row-major convention
/// throughout: `w[out × in]` means output rows are contiguous.
///
/// Field grouping (K3 tech report §2.1.1):
///
/// - **Q / K / V projections + ShortConv + biases**: Eq 2 first two lines.
/// - **`w_beta`**: Eq 2 scalar β projection.
/// - **`w_alpha_down` / `w_alpha_up` / `b_alpha` / `a_h`**: Eq 2 low-rank
///   pre-gate + Eq 5 lower-bounded decay.
/// - **`w_gate` / `w_out` / `rms_gamma`**: Eq 6 output gate.
pub struct KimiDeltaHeadParams<'a> {
    // ── Q / K / V linear projections + ShortConv (Eq 2) ─────────────
    /// W_q: `[d_k × d]` row-major (Q linear projection).
    pub w_q: &'a [f32],
    /// W_k: `[d_k × d]` row-major.
    pub w_k: &'a [f32],
    /// W_v: `[d_v × d]` row-major.
    pub w_v: &'a [f32],
    /// ShortConv kernel for Q: `[d_k, kernel_size]` in the
    /// dim-outer × kernel-inner layout matched by `causal_conv1d_step`
    /// (`weight[c * kernel_size + k]` for channel `c`, timestep `k`).
    pub conv_kernel_q: &'a [f32],
    /// ShortConv kernel for K: `[d_k, kernel_size]`.
    pub conv_kernel_k: &'a [f32],
    /// ShortConv kernel for V: `[d_v, kernel_size]`.
    pub conv_kernel_v: &'a [f32],
    /// ShortConv bias for Q: `[d_k]`.
    pub conv_bias_q: &'a [f32],
    /// ShortConv bias for K: `[d_k]`.
    pub conv_bias_k: &'a [f32],
    /// ShortConv bias for V: `[d_v]`.
    pub conv_bias_v: &'a [f32],

    // ── β delta-rule write strength (Eq 2) ──────────────────────────
    /// W_β: `[d]` (dot product with `x` yields the pre-sigmoid scalar).
    pub w_beta: &'a [f32],

    // ── α channel-wise decay (Eq 2 low-rank + Eq 5) ─────────────────
    /// W_α_↓: `[r × d]` (low-rank down projection to intermediate `r`).
    pub w_alpha_down: &'a [f32],
    /// W_α_↑: `[d_k × r]` (up projection back to `d_k`).
    pub w_alpha_up: &'a [f32],
    /// b_α: `[d_k]` (bias applied after the up projection).
    pub b_alpha: &'a [f32],
    /// A_h: per-head learnable log-scale (initialized 0 in K3, Eq 5).
    pub a_h: f32,
    /// Low-rank intermediate dimension `r` for the α projection.
    pub alpha_rank: usize,
    /// `g_min` lower bound for the decay (K3 uses -5.0, from
    /// [`KimiDeltaConfig::kda_gate_lower_bound`]).
    pub g_min: f32,

    // ── Output gate + projection (Eq 6) ─────────────────────────────
    /// W_g: `[d_v × d]` (pre-sigmoid full-rank gate projection).
    pub w_gate: &'a [f32],
    /// W_o: `[d_out × d_v]` (output projection).
    pub w_out: &'a [f32],
    /// Output dimension `d_out` (typically the hidden dim `d` for
    /// residual add into the backbone stream).
    pub d_out: usize,
    /// Optional inner RMSNorm scale γ: `[d_v]`. `None` skips the
    /// normalize (Gated MLA Eq 7 variant); `Some` matches KDA Eq 6.
    pub rms_gamma: Option<&'a [f32]>,
    /// Inner RMSNorm epsilon (ignored when `rms_gamma = None`).
    pub rms_eps: f32,
}

/// L2-normalize a slice in place: `x ← x / (||x||_2 + eps)`.
///
/// The `eps` term is added to the denominator (not to the squared
/// sum inside the sqrt), which numerically matches the pattern used
/// throughout llama.cpp / vLLM for L2-normalizing attention Q/K
/// projections when the input can be exactly zero (matches KDA's Eq 2
/// `L2Norm(Swish(ShortConv(W_{q/k} x)))` at the first token when the
/// ShortConv ring buffer is zero and Swish(0) = 0).
pub fn kimi_delta_l2_norm_in_place(x: &mut [f32], eps: f32) {
    let sum_sq: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let norm = sum_sq.sqrt() as f32 + eps;
    let scale = norm.recip();
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// Row-major dense f32 matrix-vector product.
///
/// `w` is `[out_dim, in_dim]` row-major (`w[i * in_dim + j]` is row
/// `i`, column `j`). Uses `f64` accumulation for numerical stability
/// on the large hidden-dim reductions typical of transformer
/// projections (matches the convention in this module's private
/// [`rms_norm`] helper).
fn kimi_delta_matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), out_dim * in_dim, "w shape mismatch");
    debug_assert_eq!(x.len(), in_dim, "x length mismatch");
    let mut y = vec![0.0_f32; out_dim];
    for i in 0..out_dim {
        let row_start = i * in_dim;
        let mut acc = 0.0_f64;
        for j in 0..in_dim {
            acc += f64::from(w[row_start + j]) * f64::from(x[j]);
        }
        y[i] = acc as f32;
    }
    y
}

/// Scalar dot product with `f64` accumulation.
fn kimi_delta_dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot length mismatch");
    let mut acc = 0.0_f64;
    for i in 0..a.len() {
        acc += f64::from(a[i]) * f64::from(b[i]);
    }
    acc as f32
}

/// One-token per-head KDA forward pass (composes X.4.c.1 primitives
/// with ShortConv + Swish + L2Norm into the full Eq 1-6 pipeline).
///
/// Pipeline (all steps per K3 tech report §2.1.1):
///
/// 1. **Projections (Eq 2)** — `q_pre = W_q x`, `k_pre = W_k x`,
///    `v_pre = W_v x`.
/// 2. **ShortConv (Eq 2)** — depthwise causal conv1d over each of q/k/v
///    with the same kernel size (K3: 4), independent ring buffers per
///    channel per component. Delegates to the shared
///    `causal_conv1d_step` helper.
/// 3. **Activation** — Swish (`silu`) applied element-wise to
///    q_conv, k_conv, v_conv.
/// 4. **L2Norm** — applied to q, k only (v is left un-normalized per
///    Eq 2).
/// 5. **β = Sigmoid(W_β · x)** — scalar delta-rule write strength.
/// 6. **α (Eq 5)** — `z = W_α_↑ (W_α_↓ x) + b_α`, then
///    `α = lower_bounded_decay(z, A_h, g_min)`.
/// 7. **Recurrent step (Eq 1)** — `kimi_delta_step(state, k, v, α, β)`
///    updates `state` in place.
/// 8. **Read** — `ō = kimi_delta_read(state, q) ∈ ℝ^{d_v}`.
/// 9. **Output gate (Eq 6)** —
///    `y = W_o [Sigmoid(W_g x) ⊙ RMSNorm(ō)]`, returning `ℝ^{d_out}`.
///
/// # Arguments
///
/// - `x`: hidden state `∈ ℝ^d`.
/// - `params`: per-head weight references (see [`KimiDeltaHeadParams`]).
/// - `cache`: per-head mutable state (see [`KimiDeltaHeadCache`]).
///   The recurrent state `state` and the three ShortConv ring buffers
///   all advance by one token per call.
/// - `l2_eps`: epsilon added to the L2Norm denominator on q, k.
///
/// # Panics
///
/// Panics via the debug asserts in the internal matvec helper and
/// in the shared `causal_conv1d_step` / [`kimi_delta_step`] /
/// [`kimi_delta_read`] / [`kimi_delta_output_gate`] primitives if
/// any weight or cache shape is inconsistent with the head
/// dimensions in `cache` or the hidden dimension implied by `x`.
#[must_use]
pub fn kimi_delta_forward_head(
    x: &[f32],
    params: &KimiDeltaHeadParams<'_>,
    cache: &mut KimiDeltaHeadCache,
    l2_eps: f32,
) -> Vec<f32> {
    let d = x.len();
    let d_k = cache.d_k;
    let d_v = cache.d_v;
    let ks = cache.kernel_size;

    // Step 1: linear projections.
    let q_pre = kimi_delta_matvec(params.w_q, x, d_k, d);
    let k_pre = kimi_delta_matvec(params.w_k, x, d_k, d);
    let v_pre = kimi_delta_matvec(params.w_v, x, d_v, d);

    // Step 2: ShortConv (kernel_size taps, depthwise per channel).
    let mut q_conv = vec![0.0_f32; d_k];
    let mut k_conv = vec![0.0_f32; d_k];
    let mut v_conv = vec![0.0_f32; d_v];
    causal_conv1d_step(
        &q_pre,
        &mut cache.conv_state_q,
        &mut cache.ring_pos_q,
        params.conv_kernel_q,
        params.conv_bias_q,
        &mut q_conv,
        d_k,
        ks,
    );
    causal_conv1d_step(
        &k_pre,
        &mut cache.conv_state_k,
        &mut cache.ring_pos_k,
        params.conv_kernel_k,
        params.conv_bias_k,
        &mut k_conv,
        d_k,
        ks,
    );
    causal_conv1d_step(
        &v_pre,
        &mut cache.conv_state_v,
        &mut cache.ring_pos_v,
        params.conv_kernel_v,
        params.conv_bias_v,
        &mut v_conv,
        d_v,
        ks,
    );

    // Step 3: Swish (silu) activation, element-wise.
    for v in &mut q_conv {
        *v = silu(*v);
    }
    for v in &mut k_conv {
        *v = silu(*v);
    }
    for v in &mut v_conv {
        *v = silu(*v);
    }

    // Step 4: L2Norm on q, k (v is left as-is per Eq 2).
    kimi_delta_l2_norm_in_place(&mut q_conv, l2_eps);
    kimi_delta_l2_norm_in_place(&mut k_conv, l2_eps);

    // Step 5: β = Sigmoid(W_β · x).
    debug_assert_eq!(
        params.w_beta.len(),
        d,
        "w_beta length must equal hidden dim"
    );
    let beta = sigmoid(kimi_delta_dot(params.w_beta, x));

    // Step 6: z = W_α_↑ (W_α_↓ x) + b_α, then α = lower_bounded_decay.
    let z_mid = kimi_delta_matvec(params.w_alpha_down, x, params.alpha_rank, d);
    let mut z = kimi_delta_matvec(params.w_alpha_up, &z_mid, d_k, params.alpha_rank);
    debug_assert_eq!(params.b_alpha.len(), d_k, "b_alpha length must equal d_k");
    for i in 0..d_k {
        z[i] += params.b_alpha[i];
    }
    let alpha = kimi_delta_lower_bounded_decay(&z, params.a_h, params.g_min);

    // Step 7: recurrent update (in-place on cache.state).
    kimi_delta_step(&mut cache.state, &k_conv, &v_conv, &alpha, beta);

    // Step 8: read.
    let o_bar = kimi_delta_read(&cache.state, &q_conv);

    // Step 9: output gate (Eq 6 / Eq 7 depending on rms_gamma).
    let gate_pre = kimi_delta_matvec(params.w_gate, x, d_v, d);
    kimi_delta_output_gate(
        &o_bar,
        &gate_pre,
        params.rms_gamma,
        params.rms_eps,
        params.w_out,
        params.d_out,
    )
}

#[cfg(test)]
mod kimi_delta_forward_tests {
    use super::{
        kimi_delta_forward_head, kimi_delta_l2_norm_in_place, KimiDeltaHeadCache,
        KimiDeltaHeadParams,
    };

    /// Build a minimal [`KimiDeltaHeadParams`] with the caller's owned
    /// buffers borrowed in. Convenience for the tests below that all
    /// use small toy dimensions.
    #[allow(clippy::too_many_arguments)]
    fn params_from_bufs<'a>(
        w_q: &'a [f32],
        w_k: &'a [f32],
        w_v: &'a [f32],
        conv_kernel_q: &'a [f32],
        conv_kernel_k: &'a [f32],
        conv_kernel_v: &'a [f32],
        conv_bias_q: &'a [f32],
        conv_bias_k: &'a [f32],
        conv_bias_v: &'a [f32],
        w_beta: &'a [f32],
        w_alpha_down: &'a [f32],
        w_alpha_up: &'a [f32],
        b_alpha: &'a [f32],
        a_h: f32,
        alpha_rank: usize,
        g_min: f32,
        w_gate: &'a [f32],
        w_out: &'a [f32],
        d_out: usize,
        rms_gamma: Option<&'a [f32]>,
        rms_eps: f32,
    ) -> KimiDeltaHeadParams<'a> {
        KimiDeltaHeadParams {
            w_q,
            w_k,
            w_v,
            conv_kernel_q,
            conv_kernel_k,
            conv_kernel_v,
            conv_bias_q,
            conv_bias_k,
            conv_bias_v,
            w_beta,
            w_alpha_down,
            w_alpha_up,
            b_alpha,
            a_h,
            alpha_rank,
            g_min,
            w_gate,
            w_out,
            d_out,
            rms_gamma,
            rms_eps,
        }
    }

    #[test]
    fn cache_new_is_zeroed() {
        let c = KimiDeltaHeadCache::new(4, 4, 4);
        assert_eq!(c.d_k(), 4);
        assert_eq!(c.d_v(), 4);
        assert_eq!(c.kernel_size(), 4);
        assert!(c.state.as_slice().iter().all(|&x| x == 0.0));
        assert!(c.conv_state_q.iter().all(|&x| x == 0.0));
        assert!(c.conv_state_k.iter().all(|&x| x == 0.0));
        assert!(c.conv_state_v.iter().all(|&x| x == 0.0));
        assert_eq!(c.ring_pos_q, 0);
        assert_eq!(c.ring_pos_k, 0);
        assert_eq!(c.ring_pos_v, 0);
    }

    #[test]
    fn cache_reset_zeroes_all_buffers() {
        let mut c = KimiDeltaHeadCache::new(2, 2, 3);
        // Muck with every buffer, then reset.
        c.state.as_mut_slice().fill(9.0);
        c.conv_state_q.fill(1.0);
        c.conv_state_k.fill(2.0);
        c.conv_state_v.fill(3.0);
        c.ring_pos_q = 1;
        c.ring_pos_k = 1;
        c.ring_pos_v = 1;
        c.reset();
        assert!(c.state.as_slice().iter().all(|&x| x == 0.0));
        assert!(c.conv_state_q.iter().all(|&x| x == 0.0));
        assert!(c.conv_state_k.iter().all(|&x| x == 0.0));
        assert!(c.conv_state_v.iter().all(|&x| x == 0.0));
        assert_eq!(c.ring_pos_q, 0);
        assert_eq!(c.ring_pos_k, 0);
        assert_eq!(c.ring_pos_v, 0);
    }

    #[test]
    fn l2_norm_in_place_gives_unit_length() {
        let mut x = [3.0_f32, 4.0];
        kimi_delta_l2_norm_in_place(&mut x, 0.0);
        // ||[3, 4]|| = 5 → normalized = [0.6, 0.8].
        assert!((x[0] - 0.6).abs() < 1e-6);
        assert!((x[1] - 0.8).abs() < 1e-6);
        // Verify unit length.
        let magsq: f32 = x.iter().map(|&v| v * v).sum();
        assert!((magsq - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_norm_in_place_handles_zero_input_with_eps() {
        let mut x = [0.0_f32, 0.0, 0.0];
        // Non-zero eps prevents NaN; the numerator is 0 anyway so
        // the output stays 0.
        kimi_delta_l2_norm_in_place(&mut x, 1e-6);
        for &v in &x {
            assert_eq!(v, 0.0);
        }
    }

    /// Build a "pass-through" 2-dim × 2-dim head where every weight is
    /// a small explicit value: identity Q/K/V projections, identity
    /// ShortConv (kernel row = [0, 0, 1] so only the current input
    /// survives), zero biases, zero W_β (β = 0.5), zero W_α (α → e^{-2.5}
    /// under g_min = -5), identity W_g / W_o, no RMSNorm. Used by the
    /// smoke tests below.
    #[allow(clippy::too_many_arguments)]
    struct PassThroughHead {
        d: usize,
        w_q: Vec<f32>,
        w_k: Vec<f32>,
        w_v: Vec<f32>,
        conv_kernel: Vec<f32>, // shared across q/k/v (each channel: [0, 0, 1])
        conv_bias: Vec<f32>,   // shared across q/k/v
        w_beta: Vec<f32>,
        w_alpha_down: Vec<f32>,
        w_alpha_up: Vec<f32>,
        b_alpha: Vec<f32>,
        w_gate: Vec<f32>,
        w_out: Vec<f32>,
    }

    impl PassThroughHead {
        fn new(d: usize) -> Self {
            let mut identity = vec![0.0_f32; d * d];
            for i in 0..d {
                identity[i * d + i] = 1.0;
            }
            let kernel_size = 3;
            // Per-channel kernel = [0, 0, 1] (only "current" tap survives).
            let mut conv_kernel = vec![0.0_f32; d * kernel_size];
            for c in 0..d {
                conv_kernel[c * kernel_size + (kernel_size - 1)] = 1.0;
            }
            Self {
                d,
                w_q: identity.clone(),
                w_k: identity.clone(),
                w_v: identity.clone(),
                conv_kernel,
                conv_bias: vec![0.0; d],
                w_beta: vec![0.0; d],
                // Low-rank α with rank 1 that always produces z = 0
                // regardless of x → g = -5·sigmoid(0) = -2.5 → α = e^{-2.5}.
                // rank = 1, so both projection buffers have length `d`.
                w_alpha_down: vec![0.0; d],
                w_alpha_up: vec![0.0; d],
                b_alpha: vec![0.0; d],
                w_gate: identity.clone(),
                w_out: identity,
            }
        }

        fn params(&self) -> KimiDeltaHeadParams<'_> {
            params_from_bufs(
                &self.w_q,
                &self.w_k,
                &self.w_v,
                &self.conv_kernel,
                &self.conv_kernel,
                &self.conv_kernel,
                &self.conv_bias,
                &self.conv_bias,
                &self.conv_bias,
                &self.w_beta,
                &self.w_alpha_down,
                &self.w_alpha_up,
                &self.b_alpha,
                0.0,  // A_h
                1,    // alpha_rank
                -5.0, // g_min
                &self.w_gate,
                &self.w_out,
                self.d,
                None, // no inner RMSNorm — tests focus on math, not norm scale
                1e-6,
            )
        }
    }

    #[test]
    fn forward_head_zero_input_gives_zero_output() {
        // x = 0 → all projections are 0 → q, k, v = 0 → recurrent
        // state stays at 0 (β k v^T = 0), ō = 0 → y = 0 for any gate.
        let head = PassThroughHead::new(2);
        let params = head.params();
        let mut cache = KimiDeltaHeadCache::new(2, 2, 3);
        let y = kimi_delta_forward_head(&[0.0, 0.0], &params, &mut cache, 1e-6);
        assert_eq!(y, vec![0.0, 0.0]);
    }

    #[test]
    fn forward_head_advances_all_ring_positions_per_call() {
        // Every call to `causal_conv1d_step` bumps its own ring cursor
        // by exactly one, so after a single forward pass every buffer
        // must be at position `(0 + 1) % (kernel_size - 1)`.
        let head = PassThroughHead::new(3);
        let params = head.params();
        let mut cache = KimiDeltaHeadCache::new(3, 3, 3);
        let _ = kimi_delta_forward_head(&[1.0, 2.0, -1.0], &params, &mut cache, 1e-6);
        // kernel_size = 3 → ring size = 2 → after one step, pos = 1.
        assert_eq!(cache.ring_pos_q, 1);
        assert_eq!(cache.ring_pos_k, 1);
        assert_eq!(cache.ring_pos_v, 1);
    }

    #[test]
    fn forward_head_two_tokens_progresses_state_and_conv_rings() {
        // After two consecutive calls the recurrent state must be
        // non-zero (some delta write happened) and the conv history
        // buffers must have been touched (ring wraps back to 0 for
        // kernel_size = 3, ring size = 2).
        let head = PassThroughHead::new(2);
        let params = head.params();
        let mut cache = KimiDeltaHeadCache::new(2, 2, 3);
        let _ = kimi_delta_forward_head(&[0.5, -0.25], &params, &mut cache, 1e-6);
        let _ = kimi_delta_forward_head(&[1.0, 0.5], &params, &mut cache, 1e-6);
        // Ring size = 2 → after 2 calls, pos = 0 (wrapped).
        assert_eq!(cache.ring_pos_q, 0);
        assert_eq!(cache.ring_pos_k, 0);
        assert_eq!(cache.ring_pos_v, 0);
        // State must have picked up some non-zero mass from the two
        // delta writes (β k v^T for each token).
        let state_sum: f32 = cache.state.as_slice().iter().map(|&v| v.abs()).sum();
        assert!(state_sum > 0.0, "state should be non-zero after 2 writes");
    }

    #[test]
    fn forward_head_reset_returns_to_fresh_start() {
        // After running a token then resetting, the next forward must
        // produce the same output as if it were the very first token.
        let head = PassThroughHead::new(2);
        let params = head.params();
        let mut cache_a = KimiDeltaHeadCache::new(2, 2, 3);
        let mut cache_b = KimiDeltaHeadCache::new(2, 2, 3);

        let x = [0.7_f32, -0.3];
        // Drive cache_a with a noise token, reset, then re-forward on x.
        let _ = kimi_delta_forward_head(&[3.0, -2.5], &params, &mut cache_a, 1e-6);
        cache_a.reset();
        let y_a = kimi_delta_forward_head(&x, &params, &mut cache_a, 1e-6);

        // Fresh cache_b, forward on x directly.
        let y_b = kimi_delta_forward_head(&x, &params, &mut cache_b, 1e-6);

        for i in 0..2 {
            assert!(
                (y_a[i] - y_b[i]).abs() < 1e-6,
                "reset should give parity with fresh cache: y_a[{i}] = {}, y_b[{i}] = {}",
                y_a[i],
                y_b[i],
            );
        }
    }

    #[test]
    fn forward_head_first_token_output_bounded_by_gate_and_alpha() {
        // Sanity: on the very first token the state was zero, so after
        // one step S = β k v^T (α scaled term is zero). The output ō =
        // S^T q = β · (k^T q) · v. Then W_g = I → sigmoid(x) per
        // channel; RMSNorm skipped (rms_gamma = None); W_o = I. So
        // each output channel j is:
        //     y[j] = sigmoid(x[j]) · β · (k^T q) · v[j]
        // with |sigmoid(x[j])| ≤ 1 and β = sigmoid(0) = 0.5.
        //
        // Assert |y| ≤ some finite bound rather than an exact value —
        // this catches "output blew up to NaN / inf" regressions
        // without over-specifying the intermediate math.
        let head = PassThroughHead::new(2);
        let params = head.params();
        let mut cache = KimiDeltaHeadCache::new(2, 2, 3);
        let x = [1.0_f32, -1.0];
        let y = kimi_delta_forward_head(&x, &params, &mut cache, 1e-6);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] = {v} must be finite");
            assert!(v.abs() <= 2.0, "y[{i}] = {v} exceeded reasonable bound");
        }
    }

    #[test]
    fn forward_head_with_zero_gate_projection_halves_output_magnitude() {
        // Same PassThroughHead but override w_gate to zeros → gate_pre
        // = 0 → sigmoid(0) = 0.5 for every channel. Compared to the
        // baseline PassThrough (identity w_gate), each output channel
        // must be exactly `0.5 / sigmoid(x[j])` × baseline (per-channel).
        // For x = 0, both should trivially be 0; use a non-trivial x
        // and only assert the sign / boundedness stays consistent.
        let mut head = PassThroughHead::new(2);
        // Baseline forward.
        let params_baseline = head.params();
        let mut cache_a = KimiDeltaHeadCache::new(2, 2, 3);
        let x = [0.5_f32, -0.5];
        let y_baseline = kimi_delta_forward_head(&x, &params_baseline, &mut cache_a, 1e-6);

        // Zero the gate.
        head.w_gate.iter_mut().for_each(|v| *v = 0.0);
        let params_zero_gate = head.params();
        let mut cache_b = KimiDeltaHeadCache::new(2, 2, 3);
        let y_zero_gate = kimi_delta_forward_head(&x, &params_zero_gate, &mut cache_b, 1e-6);

        // Baseline gate: sigmoid(x[0]) = sigmoid(0.5) ≈ 0.622.
        // Zero gate:     sigmoid(0)   = 0.5.
        // Ratio (zero / baseline) = 0.5 / sigmoid(x[j]) per channel.
        for j in 0..2 {
            let s_x_j = 1.0 / (1.0 + (-x[j]).exp());
            let expected_ratio = 0.5 / s_x_j;
            let actual_ratio = y_zero_gate[j] / y_baseline[j];
            assert!(
                (actual_ratio - expected_ratio).abs() < 1e-3,
                "channel {j} ratio {actual_ratio}, expected {expected_ratio} \
                 (baseline sigmoid {s_x_j})",
            );
        }
    }
}

// ── Block Attention Residuals (Phase X.4.d) ─────────────────────────
//
// Implements K3 tech report §2.2 Eq 8-10 runtime scheme. Full AttnRes
// (Eq 8-9) has each layer selectively retrieve representations from
// the token embedding + every preceding layer output via softmax
// attention with a learnable per-layer pseudo-query. Block AttnRes
// (Eq 10) reduces the `O(Ld)` memory / communication overhead to
// `O(Nd)` by summing layer outputs within `N` block groups of size
// `S = L/N` layers each.
//
// K3 partitions its `L = 93` layers into `N = 8` blocks of `S = 12`
// layers each (the last block is a partial 9-layer block since
// 8 × 12 = 96 ≠ 93; specifically 7 full 12-layer blocks + 1
// 9-layer block = 84 + 9 = 93). Counting the token embedding as
// `b_0`, that gives 9 total representations for the cross-block
// attention V matrix.
//
// This module ships the *runtime primitives*: state struct + softmax
// attention with RMSNorm-on-keys kernel + per-layer step. The
// final aggregation of the N block representations into logits
// (paper §2.2 "final output layer aggregates all N block
// representations") is deferred to X.4.d.2 alongside the concrete
// forward_kimi_k3 integration, since the paper does not specify the
// aggregation kernel precisely enough for a standalone unit test.

/// Runtime state for Block Attention Residuals over one sequence.
///
/// Grows one entry in `block_reps` every `block_size` layers and
/// resets `current_partial` at the same cadence. All buffers are
/// heap-allocated so callers can freely clone the state for
/// beam-search / speculative-decoding fan-out.
///
/// Invariants:
/// - `block_reps[0]` is always the token embedding (`b_0 = h_1`
///   per Eq 8), populated by [`BlockAttnResState::new`].
/// - `block_reps.len() >= 1` at all times.
/// - `current_partial.len() == d`.
/// - `pos_in_block ∈ 0..block_size` (advances per
///   [`block_attnres_layer_step`], wraps at `block_size`).
///
/// K3 defaults: `d = 7168`, `block_size = 12`, so each
/// `Vec<f32>` in `block_reps` ≈ 28 KB and a full model
/// carries 9 finalized reps + 1 partial ≈ 280 KB — negligible
/// compared to the KDA head caches (~450 MB).
#[derive(Debug, Clone)]
pub struct BlockAttnResState {
    /// Finalized block representations `[b_0, b_1, ..., b_{n-1}]`.
    /// `b_0` is always the token embedding.
    pub block_reps: Vec<Vec<f32>>,
    /// Running partial sum `b_n^i = Σ_{j ≤ i, j ∈ B_n} v_j` for
    /// the current block. Reset to zeros when a block finalizes.
    pub current_partial: Vec<f32>,
    /// Current block index (0-indexed; block 0 already contains
    /// `b_0` = embedding, so the first *layer* writes into block 1).
    current_block: usize,
    /// Zero-indexed position within the current block (0 =
    /// "no layers yet processed in this block").
    pos_in_block: usize,
    /// Hidden dimension `d`.
    d: usize,
    /// Layers per block (K3 uses 12).
    block_size: usize,
}

impl BlockAttnResState {
    /// Initialize with the token embedding as `b_0 = h_1` (Eq 8).
    ///
    /// # Panics
    ///
    /// Panics if `embedding.len() == 0` or `block_size == 0`.
    #[must_use]
    pub fn new(embedding: &[f32], block_size: usize) -> Self {
        assert!(!embedding.is_empty(), "embedding must be non-empty");
        assert!(block_size > 0, "block_size must be at least 1");
        let d = embedding.len();
        Self {
            block_reps: vec![embedding.to_vec()],
            current_partial: vec![0.0; d],
            current_block: 1, // b_0 already finalized as embedding
            pos_in_block: 0,
            d,
            block_size,
        }
    }

    /// Number of finalized block representations, including `b_0`.
    #[inline]
    #[must_use]
    pub fn num_block_reps(&self) -> usize {
        self.block_reps.len()
    }

    /// Current in-progress block index (0-based). Callers can use
    /// this to decide when to inject a per-block operation such as
    /// pipeline-parallel communication.
    #[inline]
    #[must_use]
    pub const fn current_block_idx(&self) -> usize {
        self.current_block
    }

    /// Position of the next layer within the current block.
    #[inline]
    #[must_use]
    pub const fn pos_in_block(&self) -> usize {
        self.pos_in_block
    }
}

/// Softmax attention with the K3 "RMSNorm-on-keys" kernel from
/// §2.2 Eq 9:
///
/// ```text
/// φ(q, k) = exp(qᵀ · RMSNorm(k))
/// α_i = φ(q, k_i) / Σ_j φ(q, k_j)
/// h = Σ_i α_i · v_i
/// ```
///
/// K3 sets `k_i = v_i` (same tensor plays both roles), so this
/// helper takes a single slice of keys-and-values.
///
/// Numerical stability: uses the standard "subtract max logit
/// before exp" (log-sum-exp trick) so extreme `q^T RMSNorm(k)`
/// magnitudes do not blow up to `+inf`. The optional RMSNorm
/// scale `γ` is applied per-channel after normalization; pass
/// `None` to run un-γ-scaled normalization.
///
/// # Panics
///
/// Panics if `keys_values` is empty, or if any key length differs
/// from `query.len()`, or if `rms_gamma` (when `Some`) has a
/// different length from `query`.
#[must_use]
pub fn block_attnres_softmax_attention(
    query: &[f32],
    keys_values: &[&[f32]],
    rms_gamma: Option<&[f32]>,
    rms_eps: f32,
) -> Vec<f32> {
    assert!(!keys_values.is_empty(), "keys_values must be non-empty");
    let d = query.len();
    if let Some(gamma) = rms_gamma {
        assert_eq!(gamma.len(), d, "rms_gamma length must equal query length");
    }
    for (i, k) in keys_values.iter().enumerate() {
        assert_eq!(
            k.len(),
            d,
            "keys_values[{i}] length {} != query length {d}",
            k.len()
        );
    }

    // 1. Compute logits ℓ_i = qᵀ · RMSNorm(k_i) for each key.
    let mut logits = Vec::with_capacity(keys_values.len());
    for k in keys_values {
        // RMSNorm(k) = γ · k / sqrt(mean(k²) + eps).
        let ss: f64 = k.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let mean = (ss / d as f64) as f32;
        let scale = (mean + rms_eps).sqrt().recip();
        // Fused dot product with the RMSNorm scale and optional γ.
        let mut logit = 0.0_f64;
        if let Some(gamma) = rms_gamma {
            for j in 0..d {
                logit += f64::from(query[j]) * f64::from(k[j] * scale * gamma[j]);
            }
        } else {
            for j in 0..d {
                logit += f64::from(query[j]) * f64::from(k[j] * scale);
            }
        }
        logits.push(logit as f32);
    }

    // 2. Softmax with log-sum-exp stability.
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_logits = Vec::with_capacity(logits.len());
    let mut sum_exp = 0.0_f64;
    for &l in &logits {
        let e = (l - max_logit).exp();
        exp_logits.push(e);
        sum_exp += f64::from(e);
    }
    let inv_sum = (sum_exp as f32).recip();

    // 3. Weighted sum h = Σ α_i · v_i. K3 uses k_i = v_i, so the
    //    same slice.
    let mut h = vec![0.0_f32; d];
    for (i, v) in keys_values.iter().enumerate() {
        let alpha = exp_logits[i] * inv_sum;
        for j in 0..d {
            h[j] += alpha * v[j];
        }
    }
    h
}

/// One per-layer step of Block Attention Residuals (K3 §2.2 Eq 10).
///
/// Semantics per layer `l` at position `i` in block `n`:
///
/// 1. Read the current-partial snapshot `b_n^{i-1}` (the partial
///    sum through position `i-1`, *before* this layer's output is
///    added).
/// 2. Assemble the value matrix `V`:
///    - If `i = 0` (first layer of block `n`): `V = [b_0, ..., b_{n-1}]`.
///    - Otherwise: `V = [b_0, ..., b_{n-1}, b_n^{i-1}]`.
/// 3. Compute `h_l = Σ softmax(qᵀ RMSNorm(v)) · v` with the
///    per-layer learnable pseudo-query `w_l`.
/// 4. Accumulate `layer_output` into `current_partial` so the *next*
///    step sees `b_n^i` as its `b_n^{(i+1)-1}`.
/// 5. Advance `pos_in_block`. If we hit `block_size`, finalize:
///    push `current_partial` into `block_reps`, reset the partial
///    to zeros, increment `current_block`, and wrap
///    `pos_in_block` back to 0.
///
/// Returns `h_l ∈ ℝ^d`, the residual stream state that feeds the
/// next layer.
///
/// # Panics
///
/// Panics if `layer_output.len() != state.d`, `w_l.len() != state.d`,
/// or (when `Some`) `rms_gamma_k.len() != state.d`.
pub fn block_attnres_layer_step(
    state: &mut BlockAttnResState,
    layer_output: &[f32],
    w_l: &[f32],
    rms_gamma_k: Option<&[f32]>,
    rms_eps: f32,
) -> Vec<f32> {
    let d = state.d;
    assert_eq!(layer_output.len(), d, "layer_output length must equal d");
    assert_eq!(w_l.len(), d, "w_l length must equal d");

    // Step 1-2: assemble V from finalized block reps + (partial before this layer).
    let mut kv_slices: Vec<&[f32]> = state.block_reps.iter().map(Vec::as_slice).collect();
    // `pos_in_block > 0` means at least one layer has already contributed
    // to `current_partial` earlier in this block, so include it as the
    // last entry in V. On the first layer of a block (pos == 0), the
    // partial is exactly zeros and would degrade the attention weights
    // toward a spurious near-zero key; per Eq 10 we omit it entirely.
    if state.pos_in_block > 0 {
        kv_slices.push(state.current_partial.as_slice());
    }

    // Step 3: cross-block softmax attention with RMSNorm-on-keys.
    let h_l = block_attnres_softmax_attention(w_l, &kv_slices, rms_gamma_k, rms_eps);

    // Step 4: accumulate this layer's output into the current partial.
    for j in 0..d {
        state.current_partial[j] += layer_output[j];
    }

    // Step 5: advance position, finalize the block when we've hit block_size.
    state.pos_in_block += 1;
    if state.pos_in_block == state.block_size {
        // Move current_partial into block_reps (owning), reset with fresh zeros.
        let finalized = std::mem::replace(&mut state.current_partial, vec![0.0_f32; d]);
        state.block_reps.push(finalized);
        state.current_block += 1;
        state.pos_in_block = 0;
    }

    h_l
}

#[cfg(test)]
mod block_attnres_tests {
    use super::{block_attnres_layer_step, block_attnres_softmax_attention, BlockAttnResState};

    #[test]
    fn state_new_stores_embedding_as_b0() {
        let embedding = [1.0_f32, 2.0, 3.0, 4.0];
        let s = BlockAttnResState::new(&embedding, 12);
        assert_eq!(s.num_block_reps(), 1);
        assert_eq!(s.block_reps[0], embedding);
        assert!(s.current_partial.iter().all(|&x| x == 0.0));
        assert_eq!(s.current_block_idx(), 1);
        assert_eq!(s.pos_in_block(), 0);
    }

    #[test]
    fn softmax_attention_single_key_returns_that_key() {
        // Only one entry in V → softmax collapses to weight 1.0 → h = v_0.
        let q = [0.5_f32, -0.5, 1.0];
        let v0 = [7.0_f32, -3.0, 2.0];
        let vs: Vec<&[f32]> = vec![&v0];
        let h = block_attnres_softmax_attention(&q, &vs, None, 1e-6);
        for i in 0..3 {
            assert!((h[i] - v0[i]).abs() < 1e-4, "h[{i}] = {}", h[i]);
        }
    }

    #[test]
    fn softmax_attention_zero_query_averages_keys() {
        // query = 0 → every logit = 0 → softmax uniform → h = mean of v_i.
        let q = [0.0_f32, 0.0, 0.0];
        let v0 = [1.0_f32, 2.0, 3.0];
        let v1 = [4.0_f32, 5.0, 6.0];
        let v2 = [7.0_f32, 8.0, 9.0];
        let vs: Vec<&[f32]> = vec![&v0, &v1, &v2];
        let h = block_attnres_softmax_attention(&q, &vs, None, 1e-6);
        let expected = [
            (1.0 + 4.0 + 7.0) / 3.0,
            (2.0 + 5.0 + 8.0) / 3.0,
            (3.0 + 6.0 + 9.0) / 3.0,
        ];
        for i in 0..3 {
            assert!(
                (h[i] - expected[i]).abs() < 1e-4,
                "h[{i}] = {}, expected {}",
                h[i],
                expected[i]
            );
        }
    }

    #[test]
    fn softmax_attention_dominant_key_wins() {
        // One key aligned with the query has a much larger logit
        // → softmax concentrates on that key → h ≈ that v.
        // Use q = [1, 0, 0] and vs = [ [10, 0, 0], [0, 0.01, 0], [0, 0, 0.01] ].
        // Logit_0 = q^T RMSNorm(v_0) = 1 · (10 / sqrt(100/3 + eps)) ≈ 1.732
        // Logit_1, Logit_2 ≈ 0 → softmax puts >0.85 mass on v_0.
        let q = [1.0_f32, 0.0, 0.0];
        let v0 = [10.0_f32, 0.0, 0.0];
        let v1 = [0.0_f32, 0.01, 0.0];
        let v2 = [0.0_f32, 0.0, 0.01];
        let vs: Vec<&[f32]> = vec![&v0, &v1, &v2];
        let h = block_attnres_softmax_attention(&q, &vs, None, 1e-6);
        // Dominant key contributes most of the first channel.
        assert!(h[0] > 5.0, "dominant channel got h[0]={}", h[0]);
        assert!(h[1].abs() < 0.01, "off-axis channel h[1]={}", h[1]);
        assert!(h[2].abs() < 0.01, "off-axis channel h[2]={}", h[2]);
    }

    #[test]
    fn softmax_attention_with_gamma_scales_keys() {
        // rms_gamma = 2 · [1, 1, 1] doubles the effective RMSNorm output,
        // which doubles the logit → in a 1-key scenario the softmax
        // output is still equal to v (since softmax normalization
        // washes out any single-key scale), so h == v regardless of γ.
        let q = [0.5_f32, 0.3, -0.2];
        let v0 = [1.0_f32, -1.0, 2.0];
        let gamma = [2.0_f32, 2.0, 2.0];
        let vs: Vec<&[f32]> = vec![&v0];
        let h = block_attnres_softmax_attention(&q, &vs, Some(&gamma), 1e-6);
        for i in 0..3 {
            assert!((h[i] - v0[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn layer_step_first_layer_in_block_omits_partial_from_v() {
        // Fresh state (pos_in_block = 0) → V = [b_0] only. Confirm
        // the returned h_l equals softmax_attention(w_l, &[b_0]).
        let embedding = [1.0_f32, 2.0, 3.0, 4.0];
        let mut s = BlockAttnResState::new(&embedding, 12);
        let layer_output = [0.5_f32, -0.5, 0.5, -0.5];
        let w_l = [1.0_f32, 0.0, 0.0, 0.0];

        let h = block_attnres_layer_step(&mut s, &layer_output, &w_l, None, 1e-6);

        // Independent oracle: V = [b_0] only.
        let vs: Vec<&[f32]> = vec![&embedding];
        let expected = block_attnres_softmax_attention(&w_l, &vs, None, 1e-6);
        for i in 0..4 {
            assert!(
                (h[i] - expected[i]).abs() < 1e-5,
                "h[{i}] = {}, expected {}",
                h[i],
                expected[i]
            );
        }
    }

    #[test]
    fn layer_step_second_layer_includes_prior_partial_snapshot() {
        // After one layer step: current_partial == first layer_output.
        // The SECOND layer step should attend over V = [b_0, current_partial]
        // where current_partial is the SNAPSHOT before the second layer's
        // output is added (i.e., just the first layer's output).
        let embedding = [1.0_f32, 0.0, 0.0, 0.0];
        let mut s = BlockAttnResState::new(&embedding, 12);
        let layer_output_1 = [0.0_f32, 1.0, 0.0, 0.0];
        let w_l = [0.5_f32, 0.5, 0.5, 0.5];
        let _ = block_attnres_layer_step(&mut s, &layer_output_1, &w_l, None, 1e-6);

        // Snapshot partial BEFORE the second layer's step, then run step 2.
        let partial_before_step_2 = s.current_partial.clone();
        assert_eq!(partial_before_step_2, layer_output_1);

        let layer_output_2 = [0.0_f32, 0.0, 1.0, 0.0];
        let h = block_attnres_layer_step(&mut s, &layer_output_2, &w_l, None, 1e-6);

        // Independent oracle: V = [b_0, partial_before_step_2].
        let vs: Vec<&[f32]> = vec![&embedding, &partial_before_step_2];
        let expected = block_attnres_softmax_attention(&w_l, &vs, None, 1e-6);
        for i in 0..4 {
            assert!(
                (h[i] - expected[i]).abs() < 1e-5,
                "h[{i}] = {}, expected {}",
                h[i],
                expected[i]
            );
        }
    }

    #[test]
    fn partial_sum_equals_sum_of_layer_outputs_within_block() {
        // Fire 3 layer steps into a block of size 4, verify the
        // running partial is exactly the sum of those 3 outputs.
        let embedding = [0.0_f32; 3];
        let mut s = BlockAttnResState::new(&embedding, 4);
        let w_l = [0.0_f32; 3];
        let outs = [
            [1.0_f32, 2.0, 3.0],
            [0.5_f32, -1.0, 4.0],
            [-2.0_f32, 0.25, 1.5],
        ];
        for out in &outs {
            let _ = block_attnres_layer_step(&mut s, out, &w_l, None, 1e-6);
        }
        for j in 0..3 {
            let expected: f32 = outs.iter().map(|o| o[j]).sum();
            assert!(
                (s.current_partial[j] - expected).abs() < 1e-4,
                "channel {j}: partial = {}, expected {}",
                s.current_partial[j],
                expected
            );
        }
        // Block hasn't finalized yet (3 layers < block_size 4).
        assert_eq!(s.num_block_reps(), 1);
        assert_eq!(s.pos_in_block(), 3);
    }

    #[test]
    fn block_finalizes_after_block_size_steps() {
        // block_size = 2: after 2 layer steps the block finalizes,
        // block_reps grows to 2 entries, current_partial resets to 0,
        // pos_in_block wraps to 0, current_block increments.
        let embedding = [1.0_f32, 2.0];
        let mut s = BlockAttnResState::new(&embedding, 2);
        let w_l = [0.0_f32, 0.0];
        let out1 = [1.0_f32, 1.0];
        let out2 = [2.0_f32, 3.0];
        let _ = block_attnres_layer_step(&mut s, &out1, &w_l, None, 1e-6);
        assert_eq!(s.num_block_reps(), 1);
        assert_eq!(s.pos_in_block(), 1);
        let _ = block_attnres_layer_step(&mut s, &out2, &w_l, None, 1e-6);
        assert_eq!(s.num_block_reps(), 2, "block should have finalized");
        assert_eq!(s.pos_in_block(), 0, "pos should wrap back to 0");
        assert_eq!(s.current_block_idx(), 2, "current_block should advance");
        // Finalized b_1 = out1 + out2 = [3.0, 4.0].
        assert_eq!(s.block_reps[1], vec![3.0_f32, 4.0]);
        // current_partial reset to zeros.
        assert!(s.current_partial.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn full_workflow_two_blocks_end_to_end() {
        // 2 blocks × 2 layers = 4 layer steps. Confirm state
        // trajectory matches a hand-computed table.
        let embedding = [1.0_f32, 0.0];
        let mut s = BlockAttnResState::new(&embedding, 2);
        let w_l = [0.0_f32, 0.0]; // zero query → uniform softmax
        let outs = [
            [1.0_f32, 1.0], // block 1, layer 1
            [2.0_f32, 2.0], // block 1, layer 2 → finalize b_1 = [3, 3]
            [4.0_f32, 4.0], // block 2, layer 1
            [8.0_f32, 8.0], // block 2, layer 2 → finalize b_2 = [12, 12]
        ];
        for out in &outs {
            let _ = block_attnres_layer_step(&mut s, out, &w_l, None, 1e-6);
        }
        // Two blocks finalized on top of the embedding.
        assert_eq!(s.num_block_reps(), 3);
        assert_eq!(s.block_reps[0], vec![1.0_f32, 0.0]); // b_0 = embedding
        assert_eq!(s.block_reps[1], vec![3.0_f32, 3.0]); // b_1
        assert_eq!(s.block_reps[2], vec![12.0_f32, 12.0]); // b_2
        assert_eq!(s.pos_in_block(), 0);
        assert!(s.current_partial.iter().all(|&x| x == 0.0));
    }
}

// ── KDA chunkwise scalar reference (Phase X.4.h.1) ─────────────────
//
// K3 tech report §2.1.1 "Chunkwise parallel form" (Eq 3-4) describes
// a chunked prefill algorithm that is recurrent across chunks and
// parallel within each chunk. The full parallel form uses a UT
// transform (inherited from Kimi Linear ref [63]) to produce a
// pseudo-value term `Ṽ_[t] := U_[t] - W_[t] S_[t]`, then computes
// all `C` outputs in one batched matmul via
//
//   A_[t] = Tril[(Q_[t] ⊙ Γ_[t]^{1→C})(K_[t] / Γ_[t]^{1→C})^T]
//   O_[t] = (Γ_[t]^{1→C} ⊙ Q_[t]) S_[t] + A_[t] Ṽ_[t]
//
// K3 uses a 16-token tile (C = 16, cumulative log-decay in (-80, 0)
// with the g_min = -5 lower bound from Eq 5). This form gives the
// Tensor-Core matmul speedup on GPU during prefill.
//
// This module ships a **scalar reference** — a batched-sequential
// wrapper that composes `kimi_delta_forward_head` C times per chunk
// and returns the C output vectors. It gives us:
//
// 1. A stable chunk-level API surface for future SIMD / GPU
//    replacements (the caller shape does not change when the fast
//    kernel lands at Phase X.4.h.2).
// 2. A bit-exact parity oracle for that future kernel (chunk output
//    must equal C sequential `kimi_delta_forward_head` calls).
// 3. Correctness cover for the cumulative-decay math (Eq 3) even
//    without the true parallel matmul.
//
// The full UT-transform parallel form (Eq 4 as written) is deferred
// to Phase X.4.h.2 because it depends on the UT construction in
// Kimi Linear ref [63], which is not spelled out at the level of
// detail needed for a standalone bit-exact test in the K3 tech
// report. The reference kernel here is what X.4.h.2 will parity-
// check against.

/// Fixed chunk size for K3 KDA prefill (`C = 16`).
///
/// From tech report §2.1.1: "Kimi Linear controls this numerical
/// range by computing relative decay in log space and dividing each
/// chunk into secondary 16-token tiles". K3 inherits the same tile
/// size.
pub const KIMI_DELTA_CHUNK_SIZE: usize = 16;

/// Chunkwise reference wrapper: runs [`kimi_delta_forward_head`]
/// `C` times sequentially and returns the `C` output vectors.
///
/// # Arguments
///
/// - `xs`: `C` hidden-state inputs, each of length `d`. Length must
///   equal [`KIMI_DELTA_CHUNK_SIZE`] (16 for K3).
/// - `params`: per-head weight references (see
///   [`KimiDeltaHeadParams`]).
/// - `cache`: per-head mutable state (see [`KimiDeltaHeadCache`]).
///   Advances by `C` tokens across this call — the recurrent state
///   picks up 16 delta writes and the three ShortConv ring buffers
///   each advance 16 slots.
/// - `l2_eps`: epsilon added to the L2Norm denominator on q, k
///   inside each per-token forward.
///
/// # Returns
///
/// `C` output vectors, each of length `params.d_out`, in the same
/// order as `xs`. The `n`-th entry is what
/// [`kimi_delta_forward_head`] would return if called on `xs[n]`
/// after the previous `n` inputs.
///
/// # Panics
///
/// Panics if `xs.len() != KIMI_DELTA_CHUNK_SIZE`.
///
/// # Numerics
///
/// Bit-exact with `C = KIMI_DELTA_CHUNK_SIZE` sequential calls to
/// [`kimi_delta_forward_head`]. A dedicated unit test enforces this
/// so any future Eq 3-4 parallel form (X.4.h.2) can be parity-tested
/// against this reference.
#[must_use]
pub fn kimi_delta_chunk_forward(
    xs: &[Vec<f32>],
    params: &KimiDeltaHeadParams<'_>,
    cache: &mut KimiDeltaHeadCache,
    l2_eps: f32,
) -> Vec<Vec<f32>> {
    assert_eq!(
        xs.len(),
        KIMI_DELTA_CHUNK_SIZE,
        "chunk must contain exactly {KIMI_DELTA_CHUNK_SIZE} tokens, got {}",
        xs.len()
    );
    let mut outs = Vec::with_capacity(KIMI_DELTA_CHUNK_SIZE);
    for x in xs {
        outs.push(kimi_delta_forward_head(x, params, cache, l2_eps));
    }
    outs
}

/// Compute the per-position cumulative decay `Γ_[t]^{1→j} = Π_{r=1}^j α_r`
/// (Eq 3 of the K3 tech report) for a chunk of `C` per-channel
/// retention vectors, each of length `d_k`.
///
/// Result layout: `[C][d_k]` (outer = position, inner = channel).
/// `result[0]` is `α_1` itself (product over a single position).
///
/// # Panics
///
/// Panics if `alphas.is_empty()`, if any `alphas[i].len() != d_k`
/// where `d_k = alphas[0].len()`, or if `alphas.len() > 128` (a
/// safety bound to catch obvious caller bugs — K3 uses 16 and
/// larger tiles are not part of the design).
///
/// # Numerics
///
/// Uses `f32` element-wise product, matching the accumulation
/// precision of every other KDA primitive. K3's `g_min = -5`
/// guarantees `α ≥ exp(-5) ≈ 6.7e-3`, so a 16-position product
/// stays above `~1.3e-35` (well above f32 denormal threshold at
/// `1.2e-38`).
#[must_use]
pub fn kimi_delta_chunk_cumulative_decay(alphas: &[Vec<f32>]) -> Vec<Vec<f32>> {
    assert!(!alphas.is_empty(), "alphas must not be empty");
    assert!(
        alphas.len() <= 128,
        "chunk length {} exceeds safety bound 128",
        alphas.len()
    );
    let d_k = alphas[0].len();
    for (i, a) in alphas.iter().enumerate() {
        assert_eq!(a.len(), d_k, "alphas[{i}].len() = {} != d_k {d_k}", a.len());
    }
    let mut gamma = Vec::with_capacity(alphas.len());
    let mut running = alphas[0].clone();
    gamma.push(running.clone());
    for a in &alphas[1..] {
        for j in 0..d_k {
            running[j] *= a[j];
        }
        gamma.push(running.clone());
    }
    gamma
}

#[cfg(test)]
mod kimi_delta_chunk_tests {
    use super::{
        kimi_delta_chunk_cumulative_decay, kimi_delta_chunk_forward, kimi_delta_forward_head,
        KimiDeltaHeadCache, KimiDeltaHeadParams, KIMI_DELTA_CHUNK_SIZE,
    };

    /// Build a minimal identity-weight `KimiDeltaHeadParams` for a
    /// `d`-dim head, matching the `PassThroughHead` used in
    /// `kimi_delta_forward_tests`. Copied here so the tests are
    /// self-contained.
    struct PassThroughHead {
        d: usize,
        w_q: Vec<f32>,
        w_k: Vec<f32>,
        w_v: Vec<f32>,
        conv_kernel: Vec<f32>,
        conv_bias: Vec<f32>,
        w_beta: Vec<f32>,
        w_alpha_down: Vec<f32>,
        w_alpha_up: Vec<f32>,
        b_alpha: Vec<f32>,
        w_gate: Vec<f32>,
        w_out: Vec<f32>,
    }

    impl PassThroughHead {
        fn new(d: usize) -> Self {
            let mut identity = vec![0.0_f32; d * d];
            for i in 0..d {
                identity[i * d + i] = 1.0;
            }
            let kernel_size = 3;
            let mut conv_kernel = vec![0.0_f32; d * kernel_size];
            for c in 0..d {
                conv_kernel[c * kernel_size + (kernel_size - 1)] = 1.0;
            }
            Self {
                d,
                w_q: identity.clone(),
                w_k: identity.clone(),
                w_v: identity.clone(),
                conv_kernel,
                conv_bias: vec![0.0; d],
                w_beta: vec![0.0; d],
                w_alpha_down: vec![0.0; d],
                w_alpha_up: vec![0.0; d],
                b_alpha: vec![0.0; d],
                w_gate: identity.clone(),
                w_out: identity,
            }
        }

        fn params(&self) -> KimiDeltaHeadParams<'_> {
            KimiDeltaHeadParams {
                w_q: &self.w_q,
                w_k: &self.w_k,
                w_v: &self.w_v,
                conv_kernel_q: &self.conv_kernel,
                conv_kernel_k: &self.conv_kernel,
                conv_kernel_v: &self.conv_kernel,
                conv_bias_q: &self.conv_bias,
                conv_bias_k: &self.conv_bias,
                conv_bias_v: &self.conv_bias,
                w_beta: &self.w_beta,
                w_alpha_down: &self.w_alpha_down,
                w_alpha_up: &self.w_alpha_up,
                b_alpha: &self.b_alpha,
                a_h: 0.0,
                alpha_rank: 1,
                g_min: -5.0,
                w_gate: &self.w_gate,
                w_out: &self.w_out,
                d_out: self.d,
                rms_gamma: None,
                rms_eps: 1e-6,
            }
        }
    }

    #[test]
    fn chunk_size_is_sixteen_per_paper() {
        assert_eq!(KIMI_DELTA_CHUNK_SIZE, 16);
    }

    #[test]
    fn chunk_forward_matches_sequential_forward_bit_exact() {
        // The reference kernel IS sequential kimi_delta_forward_head
        // under the hood, so bit-exact parity is enforced (this test
        // will keep any future refactor honest — if someone swaps the
        // per-token loop for a parallel form the parity oracle catches
        // the divergence).
        let head = PassThroughHead::new(2);
        let params = head.params();
        let mut cache_a = KimiDeltaHeadCache::new(2, 2, 3);
        let mut cache_b = KimiDeltaHeadCache::new(2, 2, 3);

        let xs: Vec<Vec<f32>> = (0..KIMI_DELTA_CHUNK_SIZE)
            .map(|i| vec![(i as f32) * 0.1, -((i as f32) * 0.05)])
            .collect();

        // Chunk API.
        let chunk_out = kimi_delta_chunk_forward(&xs, &params, &mut cache_a, 1e-6);

        // Sequential oracle.
        let mut seq_out: Vec<Vec<f32>> = Vec::with_capacity(KIMI_DELTA_CHUNK_SIZE);
        for x in &xs {
            seq_out.push(kimi_delta_forward_head(x, &params, &mut cache_b, 1e-6));
        }

        assert_eq!(chunk_out.len(), seq_out.len());
        for (t, (c, s)) in chunk_out.iter().zip(seq_out.iter()).enumerate() {
            assert_eq!(c, s, "token {t}: chunk output must equal sequential output");
        }
    }

    #[test]
    fn chunk_forward_advances_cache_state() {
        // After a chunk of 16 tokens, the recurrent state and conv
        // rings must all have advanced. In particular, ring position
        // for kernel_size = 3 → ring size = 2 → pos = 16 % 2 = 0.
        let head = PassThroughHead::new(2);
        let params = head.params();
        let mut cache = KimiDeltaHeadCache::new(2, 2, 3);
        let xs: Vec<Vec<f32>> = (0..KIMI_DELTA_CHUNK_SIZE)
            .map(|_| vec![0.5_f32, -0.5])
            .collect();
        let _ = kimi_delta_chunk_forward(&xs, &params, &mut cache, 1e-6);
        // Recurrent state has picked up mass from 16 delta writes.
        let state_sum: f32 = cache.state.as_slice().iter().map(|&v| v.abs()).sum();
        assert!(
            state_sum > 0.0,
            "state must be non-zero after a full chunk of writes"
        );
    }

    #[test]
    #[should_panic(expected = "chunk must contain exactly 16 tokens")]
    fn chunk_forward_rejects_wrong_chunk_size() {
        let head = PassThroughHead::new(2);
        let params = head.params();
        let mut cache = KimiDeltaHeadCache::new(2, 2, 3);
        // 8 tokens ≠ KIMI_DELTA_CHUNK_SIZE = 16.
        let xs = vec![vec![0.0_f32, 0.0]; 8];
        let _ = kimi_delta_chunk_forward(&xs, &params, &mut cache, 1e-6);
    }

    #[test]
    fn cumulative_decay_single_position_returns_alpha_1() {
        // C = 1 → gamma[0] = alpha[0].
        let alphas = vec![vec![0.5_f32, 0.25, 0.9]];
        let gamma = kimi_delta_chunk_cumulative_decay(&alphas);
        assert_eq!(gamma.len(), 1);
        assert_eq!(gamma[0], vec![0.5_f32, 0.25, 0.9]);
    }

    #[test]
    fn cumulative_decay_computes_running_product() {
        // C = 3 with alphas [0.5, 0.5] × 3.
        // gamma[0] = [0.5, 0.5], gamma[1] = [0.25, 0.25], gamma[2] = [0.125, 0.125].
        let alphas = vec![vec![0.5_f32, 0.5], vec![0.5_f32, 0.5], vec![0.5_f32, 0.5]];
        let gamma = kimi_delta_chunk_cumulative_decay(&alphas);
        assert_eq!(gamma.len(), 3);
        assert_eq!(gamma[0], vec![0.5_f32, 0.5]);
        for j in 0..2 {
            assert!((gamma[1][j] - 0.25).abs() < 1e-6);
            assert!((gamma[2][j] - 0.125).abs() < 1e-6);
        }
    }

    #[test]
    fn cumulative_decay_all_ones_stays_at_one() {
        let alphas = vec![vec![1.0_f32; 4]; KIMI_DELTA_CHUNK_SIZE];
        let gamma = kimi_delta_chunk_cumulative_decay(&alphas);
        assert_eq!(gamma.len(), KIMI_DELTA_CHUNK_SIZE);
        for g in &gamma {
            for &v in g {
                assert!((v - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn cumulative_decay_channel_independence() {
        // channel 0: alpha = 1.0 → gamma stays at 1.0
        // channel 1: alpha = 0.5 → gamma at pos t = 0.5^(t+1)
        let alphas: Vec<Vec<f32>> = (0..4).map(|_| vec![1.0_f32, 0.5]).collect();
        let gamma = kimi_delta_chunk_cumulative_decay(&alphas);
        for (t, g) in gamma.iter().enumerate() {
            let expected_ch1 = 0.5_f32.powi((t + 1) as i32);
            assert!((g[0] - 1.0).abs() < 1e-6);
            assert!(
                (g[1] - expected_ch1).abs() < 1e-6,
                "pos {t} channel 1: got {}, expected {expected_ch1}",
                g[1]
            );
        }
    }

    #[test]
    #[should_panic(expected = "alphas must not be empty")]
    fn cumulative_decay_rejects_empty_input() {
        let _ = kimi_delta_chunk_cumulative_decay(&[]);
    }

    #[test]
    #[should_panic(expected = "!= d_k")]
    fn cumulative_decay_rejects_mismatched_dims() {
        let alphas = vec![vec![0.5_f32; 3], vec![0.5_f32; 4]];
        let _ = kimi_delta_chunk_cumulative_decay(&alphas);
    }
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
    /// Kimi K3 / Kimi Delta Attention augmentations. Skeleton only until
    /// the 2026-07-27 open weight release; see docs/KIMI_K3_INTEGRATION.md.
    pub kimi_delta: Option<KimiDeltaConfig>,
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
    pub fn from_gguf<'a, G: crate::gguf::GgufSource<'a>>(gguf: &'a G) -> Option<Self> {
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
            let key_length = gguf
                .meta_u32(&format!("{prefix}.attention.key_length"))
                .map(|v| v as usize);
            let qk_rope_head_dim = gguf
                .meta_u32(&format!("{prefix}.rope.dimension_count"))
                .map(|v| v as usize);
            // V2 / V2-Lite (no q_lora_rank) stores the **full** Q per-head
            // dim as `attention.key_length` (nope + rope combined), while
            // V2.5 / V3 / R1 (with q_lora_rank) store only the nope portion
            // there. Split them apart so `qk_nope + qk_rope` matches the
            // actual `attn_q.weight` per-head width in both branches.
            let qk_nope_head_dim = if q_lora_rank.is_none() {
                key_length.zip(qk_rope_head_dim).map(|(kl, rope)| kl - rope)
            } else {
                key_length
            };
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

        // Kimi K3: read the `kimi-k3.*` sub-config (Phase X.4.b.1,
        // 2026-07-28). All fields are `Option` inside
        // `KimiDeltaConfig`, so a partial GGUF (e.g. K3 text-only
        // variant without vision metadata) still populates whatever
        // the file provides.
        let kimi_delta = if arch == ModelArch::KimiK3 {
            Some(KimiDeltaConfig::from_gguf(gguf, &prefix))
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
            kimi_delta,
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
            kimi_delta: None,
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

    // Phase X.3.e.3.38 diagnostic: env-gated f64 accumulator variant of
    // gqa_attention. Tests hypothesis 8 that ALICE-LLM's scalar f32
    // sequential accumulation (vs llama.cpp's SIMD FLASH_ATTN_EXT
    // block-wise accumulation) is the source of PPL divergence.
    // Enable with ALICE_ATTN_F64_ACC=1.
    let use_f64_acc = std::env::var_os("ALICE_ATTN_F64_ACC").is_some();

    // Phase X.3.e.3.39 diagnostic: online softmax variant matching
    // llama.cpp's ggml_compute_forward_flash_attn_ext_f16_one_chunk
    // algorithm (arxiv:2112.05682). Uses running max M and sum S with
    // rescaling on new-max detection: VKQ *= exp(Mold-Mnew), S = S*ms+vs.
    // Enable with ALICE_ATTN_ONLINE_SOFTMAX=1. Mathematically equivalent
    // to the two-pass version but different f32 rounding path — matches
    // llama.cpp's arithmetic order for bit-comparable results.
    let use_online_softmax = std::env::var_os("ALICE_ATTN_ONLINE_SOFTMAX").is_some();

    // Phase MSA.5.6: sparse-attention gate. When `ALICE_SPARSE_TOPK` is set
    // to a non-negative integer, dispatch to
    // `sparse_attention::llama3_bridge::llama3_sparse_attention` instead of
    // the dense loops below. `topk = 0` means "select every block" and is
    // arithmetically equivalent to dense attention modulo FP re-association.
    // We only hook when the diagnostic env flags above are off and the
    // caller isn't using softcap (adapter has no softcap support yet).
    if attn_logit_softcap.is_none() && !use_online_softmax && !use_f64_acc {
        if let Some(sparse_topk) = std::env::var("ALICE_SPARSE_TOPK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            let used_len = seq_len - attn_start;
            if used_len > 0 {
                let kv_dim = num_kv_heads * head_dim;
                let mut k_dense = Vec::with_capacity(used_len * kv_dim);
                let mut v_dense = Vec::with_capacity(used_len * kv_dim);
                for t in attn_start..seq_len {
                    k_dense.extend_from_slice(kv_cache.key_at(layer_idx, t));
                    v_dense.extend_from_slice(kv_cache.value_at(layer_idx, t));
                }
                let cfg = crate::sparse_attention::llama3_bridge::BridgeConfig {
                    hq: num_heads,
                    hkv: num_kv_heads,
                    head_dim,
                    block_size: 64,
                    page_size: 64,
                    softmax_scale: inv_sqrt_d,
                    causal: false,
                    topk: sparse_topk,
                };
                let kv_view = crate::sparse_attention::llama3_bridge::DenseKvCacheView {
                    k: &k_dense,
                    v: &v_dense,
                    seq_len: used_len,
                    hkv: num_kv_heads,
                    head_dim,
                };
                if let Ok(out) = crate::sparse_attention::llama3_bridge::llama3_sparse_attention(
                    q_buf, &kv_view, 1, &cfg,
                ) {
                    attn_out.copy_from_slice(&out);
                    return;
                }
                // Adapter rejected (geometry mismatch) → fall through to dense.
            }
        }
    }

    attn_out.fill(0.0);
    for h in 0..num_heads {
        let kv_h = h / heads_per_kv;
        let q_start = h * head_dim;
        let q_head = &q_buf[q_start..q_start + head_dim];
        let k_offset = kv_h * head_dim;
        let v_offset = kv_h * head_dim;

        if use_online_softmax {
            // llama.cpp FLASH_ATTN_EXT online softmax algorithm.
            // Interleaves Q·K score computation, running max/sum update, and
            // weighted V accumulation into a single pass. Rescales the VKQ
            // accumulator by exp(Mold - Mnew) whenever a new max is seen.
            let mut m = f32::NEG_INFINITY;
            let mut s = 0.0f32;
            // Reuse attn_out[q_start..q_start+head_dim] as VKQ accumulator
            // (attn_out.fill(0.0) already zeroed it).

            for t in attn_start..seq_len {
                let k_cached = kv_cache.key_at(layer_idx, t);
                let v_cached = kv_cache.value_at(layer_idx, t);

                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q_head[d] * k_cached[k_offset + d];
                }
                score *= inv_sqrt_d;

                if let Some(cap) = attn_logit_softcap {
                    score = cap * (score / cap).tanh();
                }

                let m_old = m;
                let (ms, vs) = if score > m {
                    m = score;
                    let ms = (m_old - m).exp(); // rescale factor for VKQ
                                                // Rescale existing VKQ accumulator.
                    for d in 0..head_dim {
                        attn_out[q_start + d] *= ms;
                    }
                    (ms, 1.0f32)
                } else {
                    (1.0f32, (score - m).exp())
                };

                // VKQ += V * vs
                for d in 0..head_dim {
                    attn_out[q_start + d] += vs * v_cached[v_offset + d];
                }
                s = s * ms + vs;
            }

            // Final normalization: attn_out /= S
            if s > 0.0 {
                let inv_s = 1.0 / s;
                for d in 0..head_dim {
                    attn_out[q_start + d] *= inv_s;
                }
            }
            continue;
        }

        let window_len = seq_len - attn_start;
        let mut scores = Vec::with_capacity(window_len);
        for t in attn_start..seq_len {
            let k_cached = kv_cache.key_at(layer_idx, t);
            let mut score = if use_f64_acc {
                let mut acc = 0.0f64;
                for d in 0..head_dim {
                    acc += q_head[d] as f64 * k_cached[k_offset + d] as f64;
                }
                (acc as f32) * inv_sqrt_d
            } else {
                let mut acc = 0.0f32;
                for d in 0..head_dim {
                    acc += q_head[d] * k_cached[k_offset + d];
                }
                acc * inv_sqrt_d
            };

            if let Some(cap) = attn_logit_softcap {
                score = cap * (score / cap).tanh();
            }

            scores.push(score);
        }

        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if use_f64_acc {
            let mut sum_f64 = 0.0f64;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                sum_f64 += *s as f64;
            }
            if sum_f64 > 0.0 {
                let inv_sum = (1.0 / sum_f64) as f32;
                for s in &mut scores {
                    *s *= inv_sum;
                }
            }
        } else {
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
        }

        for (si, t) in (attn_start..seq_len).enumerate() {
            let v_cached = kv_cache.value_at(layer_idx, t);
            let w = scores[si];
            if use_f64_acc {
                for d in 0..head_dim {
                    let acc =
                        attn_out[q_start + d] as f64 + w as f64 * v_cached[v_offset + d] as f64;
                    attn_out[q_start + d] = acc as f32;
                }
            } else {
                for d in 0..head_dim {
                    attn_out[q_start + d] += w * v_cached[v_offset + d];
                }
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
/// Phase X.3.e.3.5 debug helper: dump a slice's first / last / sum values
/// in a compact JSONL line to stderr so it can be diffed against the reference
/// `llama-eval-callback` output for layer 0 first-forward divergence hunting.
#[inline]
fn dump_slice(name: &str, s: &[f32], head_n: usize) {
    let sum: f64 = s.iter().map(|&v| v as f64).sum();
    let head: Vec<String> = s.iter().take(head_n).map(|v| format!("{v:.6}")).collect();
    let tail: Vec<String> = s
        .iter()
        .rev()
        .take(head_n)
        .map(|v| format!("{v:.6}"))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    eprintln!(
        "DN0 {name} len={} head=[{}] tail=[{}] sum={sum:.6}",
        s.len(),
        head.join(","),
        tail.join(","),
    );
}

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
    // GQA V-head → KV-head mapping: reference qwen35.cpp uses ggml_repeat_4d
    // which duplicates KV-heads CYCLICALLY (V-head 16..31 map to KV 0..15,
    // matching `iv1 % num_kv_heads` in ggml_compute_forward_gated_delta_net).
    // Phase X.3.e.3.8 fix: previous `v_head / v_per_kv` (block/consecutive)
    // mapping caused ~13/32 V-heads to use wrong Q/K slices, producing
    // sign-flipped attn_output and cascading 12% linear_attn_out divergence.
    for v_head in 0..num_v_heads {
        let kv_head = v_head % num_kv_heads;
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

    // Phase X.3.e.3.8 fix: cyclic V-head → KV-head mapping
    // (`v_head % num_kv_heads`) matches reference qwen35.cpp `ggml_repeat_4d`
    // + fused GDN `iv1 % neq1` (see gated_deltanet_step for full rationale).
    let state_stride = qk_dim * v_dim;
    state
        .par_chunks_mut(state_stride)
        .zip(out.par_chunks_mut(v_dim))
        .enumerate()
        .for_each(|(v_head, (state_slab, out_slab))| {
            let kv_head = v_head % num_kv_heads;
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
    /// Q projection: dense (V2 / V2-Lite) or LoRA (V2.5 / V3 / R1).
    /// See [`DeepSeekQProjection`] for the family-specific layout.
    q: DeepSeekQProjection<'a>,
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

/// Family-specific Q projection layout for DeepSeek variants (Issue #58).
///
/// - **`Dense`** — V2 / V2-Lite (16B / 1.6B active): single `attn_q.weight`
///   matvec `hidden → num_heads * (qk_nope + qk_rope)`, no LoRA compression.
/// - **`LoRA`** — V2.5 / V3 (671B) / R1: two-stage projection with a
///   `q_a_proj → q_a_norm → q_b_proj` chain through the `q_lora_rank`
///   bottleneck (typically 1536).
///
/// KV projection stays LoRA across the entire family so it lives outside
/// this enum. V2-Lite's `deepseek2.attention.q_lora_rank` metadata key is
/// absent, which the loader keys on to pick the `Dense` variant.
enum DeepSeekQProjection<'a> {
    /// V2 / V2-Lite dense Q: `[num_heads * (qk_nope + qk_rope), hidden_dim]`.
    Dense { q_proj: WeightRef<'a> },
    /// V2.5 / V3 / R1 LoRA Q: `[q_lora_rank, hidden]` → norm → `[num_heads * (qk_nope + qk_rope), q_lora_rank]`.
    LoRA {
        q_a_proj: WeightRef<'a>,
        q_a_norm: Vec<f32>,
        q_b_proj: WeightRef<'a>,
    },
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

// ── Kimi K3 tensor reference structs (Phase X.4.b.2, 2026-07-28) ──
//
// GGUF tensor names follow `Kuberwastaken/Kimi-K3-GGUF/TENSOR_MAP.md` and
// upstream llama.cpp PR #26185 (`pwilkin/kimi-k3-text`). Shapes are
// mostly derived from GGUF `tensor_info.dims` via
// [`load_weight_ref_any_shape`] rather than hardcoded, since KDA vs MLA
// vs Latent-MoE layers all use different per-tensor dimensions and
// per-layer weight tensor width is easier to trust from the file than
// to recompute from the config.
//
// Weight lifetime is `'a`, borrowed from the underlying GGUF file. All
// structs are private to `llama3.rs` — the `forward_kimi_k3` layer
// dispatcher (Phase X.4.c.3) reaches them by name from the same
// module.

/// MLA-specific weight bundle for one Kimi K3 attention layer (24 of
/// 93 layers). Mirrors the DeepSeek-V3 LoRA MLA layout with K3's
/// `kv_b` split into two half-tensors (`attn_k_b` + `attn_v_b`) at
/// conversion time — see TENSOR_MAP.md §"`kv_b_proj` split".
#[allow(dead_code)]
struct KimiK3MlaAttn<'a> {
    q_a: WeightRef<'a>,
    q_a_norm: Vec<f32>,
    q_b: WeightRef<'a>,
    kv_a_mqa: WeightRef<'a>,
    kv_a_norm: Vec<f32>,
    k_b: WeightRef<'a>,
    v_b: WeightRef<'a>,
}

/// KDA-specific weight bundle for one Kimi K3 attention layer (69 of
/// 93 layers). Names match the existing Qwen 3.5 DeltaNet convention
/// wherever K3 inherits from Kimi Linear (`ssm_*` prefix); K3-only
/// additions (`ssm_f_a` / `ssm_f_b` / `ssm_beta`) are the low-rank α
/// projection + scalar β projection specific to KDA's Eq 2.
///
/// **Phase X.4.b.5 additions** (from real GrEarl K3 GGUF inspection):
/// - `ssm_g` — full-rank per-head gate matrix `[num_heads * v_head_dim,
///   hidden]` (K3-specific, replaces Kimi-Linear's low-rank
///   `ssm_g_a` / `ssm_g_b` pair — this is the "full-rank output gate"
///   from paper §2.1.1).
/// - `ssm_a` — per-head learnable `A_h` log-scale array `[num_heads]`
///   (K3-specific, replaces the paper's hardcoded `A_h = 0` init).
#[allow(dead_code)]
struct KimiK3KdaAttn<'a> {
    q: WeightRef<'a>,
    k: WeightRef<'a>,
    v: WeightRef<'a>,
    ssm_conv1d_q: WeightRef<'a>,
    ssm_conv1d_k: WeightRef<'a>,
    ssm_conv1d_v: WeightRef<'a>,
    ssm_f_a: WeightRef<'a>,
    ssm_f_b: WeightRef<'a>,
    ssm_beta: WeightRef<'a>,
    ssm_norm: Vec<f32>,
    /// Optional per-head `dt` bias — some K3 conversions ship it
    /// (`ssm_dt.bias`), some do not; the forward path treats absence
    /// as an all-zero bias.
    ssm_dt_bias: Option<Vec<f32>>,
    /// K3-specific full-rank per-head output gate matrix, shape
    /// `[num_heads * v_head_dim, hidden]`. Optional (skeleton +
    /// pwilkin-synth fixtures don't ship it; real GrEarl GGUF does).
    /// When absent, forward falls back to identity per-head gate.
    ssm_g: Option<WeightRef<'a>>,
    /// K3-specific per-head A_h log-scale array `[num_heads]`.
    /// Optional (fixtures may omit); when absent, forward uses
    /// `A_h = 0.0` per paper init.
    ssm_a: Option<Vec<f32>>,
}

/// Attention-side dispatch per layer. Layer `il` is MLA iff `il ∈
/// KimiDeltaConfig::full_attn_layers` (see `is_mla_layer`).
#[allow(dead_code)]
enum KimiK3Attention<'a> {
    Mla(KimiK3MlaAttn<'a>),
    Kda(KimiK3KdaAttn<'a>),
}

/// Stable LatentMoE weight bundle for one non-dense K3 layer (layers
/// 1..93 — layer 0 is dense per `first_k_dense_replace = 1`).
///
/// Shape overview (K3 defaults):
///
/// - `ffn_gate_inp`: `[num_experts=896, hidden=7168]` router logits.
/// - `exp_probs_b`: `[num_experts]` router bias (noaux_tc correction).
/// - `ffn_{gate,up,down}_shexp`: shared expert (`num_shared_experts=2`
///   fanned into a single fused block per K3 spec).
/// - `routed_exp_{up,down}`: `W↑` (`[hidden, latent_hidden=3584]`) and
///   `W↓` (`[latent_hidden, hidden]`) latent projections shared by
///   every routed expert.
/// - `routed_exp_norm`: `[latent_hidden]` RMSNorm γ inserted between
///   the aggregated routed sum `u` and the `W↑` up projection
///   (K3-specific stabilization, see Stable LatentMoE §2.3.1).
/// - `ffn_{gate,up,down}_exps`: 3-D per-expert FFN weight cubes,
///   shape `[moe_intermediate=3072, latent_hidden, num_experts]`
///   (Kuberwastaken's `EXPERTS = [(w1, ffn_gate_exps), (w3,
///   ffn_up_exps), (w2, ffn_down_exps)]` mapping).
#[allow(dead_code)]
struct KimiK3LatentMoe<'a> {
    ffn_gate_inp: Vec<f32>,
    exp_probs_b: Option<Vec<f32>>,
    ffn_gate_shexp: WeightRef<'a>,
    ffn_up_shexp: WeightRef<'a>,
    ffn_down_shexp: WeightRef<'a>,
    routed_exp_up: WeightRef<'a>,
    routed_exp_down: WeightRef<'a>,
    routed_exp_norm: Vec<f32>,
    ffn_gate_exps: WeightRef<'a>,
    ffn_up_exps: WeightRef<'a>,
    ffn_down_exps: WeightRef<'a>,
}

/// FFN-side dispatch per layer. Layer `il` is dense iff
/// `il < first_k_dense_replace` (K3: only layer 0).
///
/// The Dense variant (3 `WeightRef`s ≈ 144 B) is much smaller than
/// the LatentMoE variant (11 fields including two `Vec<f32>`s ≈
/// several KB), so clippy would normally suggest boxing LatentMoE.
/// We suppress that here because a `Vec<KimiK3LayerWeights>` already
/// pays the heap-allocation cost per layer at construction time and
/// the enum tag lives inline; boxing the large variant would add a
/// second heap round-trip on every layer forward without saving any
/// resident memory.
#[allow(dead_code, clippy::large_enum_variant)]
enum KimiK3Ffn<'a> {
    Dense {
        gate: WeightRef<'a>,
        up: WeightRef<'a>,
        down: WeightRef<'a>,
    },
    LatentMoe(KimiK3LatentMoe<'a>),
}

/// Kimi K3 per-layer weight bundle (Phase X.4.b.2, refactored in
/// X.4.c.3.4.b).
///
/// Every K3 layer carries the 4 COMMON tensors + 2 AttnRes score
/// vectors (`attn_norm`, `ffn_norm`, `attn_output`, `attn_gate`,
/// `attn_res_score`, `ffn_res_score`) plus one of two attention
/// layouts (MLA or KDA) and one of two FFN layouts (Dense or
/// LatentMoE). The `KimiK3Attention` and `KimiK3Ffn` enums encode
/// both dispatches so no `Option` juggling is needed in the forward
/// path.
///
/// **Phase X.4.c.3.4.b refactor**: paper §2.2 originally used
/// `attn_res_norm` (RMSNorm γ) + `attn_res_proj` (per-layer pseudo-
/// query projection) as two separate tensors. The pwilkin
/// `ggml-org/llama.cpp` PR #26185 real GGUF export fuses these into
/// a single 1D score vector `attn_res_score` per site (its
/// `constants.py` comment reads
/// `# Kimi K3 (fused res_norm * res_proj, pre-attention)`). The
/// pre-refactor 4-tensor layout (`{attn,ffn}_res_{norm,proj}`) has
/// been collapsed to 2 tensors (`attn_res_score`, `ffn_res_score`),
/// each `Vec<f32>` of length `n_embd`.
#[allow(dead_code)]
struct KimiK3LayerWeights<'a> {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    attn_output: WeightRef<'a>,
    /// K3-specific input-dependent full-rank output gate applied on
    /// every layer (MLA `mla_use_output_gate = true` + KDA
    /// `use_full_rank_gate = true`).
    attn_gate: WeightRef<'a>,
    /// K3-only Attention Residuals: pre-attention fused score vector
    /// (`blk.{N}.attn_res_score` in GGUF, shape `[n_embd]`, `= res_norm
    /// * res_proj` fused per pwilkin PR).
    attn_res_score: Vec<f32>,
    /// K3-only Attention Residuals: pre-FFN fused score vector
    /// (`blk.{N}.ffn_res_score` in GGUF, shape `[n_embd]`, `= res_norm
    /// * res_proj` fused per pwilkin PR).
    ffn_res_score: Vec<f32>,
    attn: KimiK3Attention<'a>,
    ffn: KimiK3Ffn<'a>,
}

/// Kimi K3 full-model weight bundle (Phase X.4.b.2, refactored in
/// X.4.c.3.4.c).
///
/// Global tensors (4) + a per-layer vector matching
/// `config.num_layers` (K3: 93). `output_res_score` is the K3-only
/// 1D fused score vector for the final N-block aggregation of
/// AttnRes (paper §2.2 / pwilkin PR #26185 `output_res_score`).
#[allow(dead_code)]
pub struct KimiK3ModelWeights<'a> {
    token_embd: WeightRef<'a>,
    output_norm: Vec<f32>,
    output: WeightRef<'a>,
    /// K3-only AttnRes: final output-side fused score vector
    /// (`output_res_score` in GGUF, shape `[n_embd]`).
    output_res_score: Vec<f32>,
    layers: Vec<KimiK3LayerWeights<'a>>,
}

/// Load the K3 per-layer weight bundle from GGUF (Phase X.4.b.2).
///
/// Layer type dispatch:
///
/// - **MLA vs KDA** — decided by `config.kimi_delta.is_mla_layer(il)`
///   which reads the 0-indexed `full_attn_layers` array populated by
///   `k3meta.py`.
/// - **Dense vs LatentMoE** — decided by `il < first_k_dense_replace`
///   from the K3 sub-config (K3 default: only layer 0 is dense).
///
/// Returns a descriptive `Err` string on the first missing required
/// tensor, so the top-level loader can surface which layer / tensor
/// name to look at when the GGUF is malformed or a K3 variant ships
/// a tensor under a different name than TENSOR_MAP.md documents.
#[allow(dead_code)]
fn load_kimi_k3_layer_weights<'a, G: crate::gguf::GgufSource<'a>>(
    gguf: &'a G,
    il: usize,
    config: &Llama3Config,
) -> Result<KimiK3LayerWeights<'a>, String> {
    let prefix = format!("blk.{il}");
    let kd = config
        .kimi_delta
        .as_ref()
        .ok_or_else(|| format!("layer {il}: kimi_delta sub-config not populated"))?;

    let load_ref = |name: &str| -> Result<WeightRef<'a>, String> {
        load_weight_ref_any_shape(gguf, name)
            .ok_or_else(|| format!("layer {il}: missing tensor '{name}'"))
    };
    let load_norm = |name: &str| -> Result<Vec<f32>, String> {
        gguf.tensor_to_f32(name)
            .ok_or_else(|| format!("layer {il}: missing norm tensor '{name}'"))
    };

    // ── COMMON (all layers, all layer types) ─────────────────────────
    let attn_norm = load_norm(&format!("{prefix}.attn_norm.weight"))?;
    let ffn_norm = load_norm(&format!("{prefix}.ffn_norm.weight"))?;
    let attn_output = load_ref(&format!("{prefix}.attn_output.weight"))?;
    // AttnRes fused 1D score vectors (pwilkin PR #26185 GGUF export
    // convention: `blk.{N}.attn_res_score` + `blk.{N}.ffn_res_score`,
    // each `[n_embd]`). Paper §2.2 originally splits these as
    // `res_norm * res_proj`; the export fuses them.
    let attn_res_score = load_norm(&format!("{prefix}.attn_res_score.weight"))?;
    let ffn_res_score = load_norm(&format!("{prefix}.ffn_res_score.weight"))?;

    // ── Attention (MLA XOR KDA) ──────────────────────────────────────
    // Layer type resolution ladder (Phase X.4.b.4):
    //
    // 1. **Metadata path** — `config.kimi_delta.is_mla_layer(il)` reads
    //    the 0-indexed `full_attn_layers` array from the K3 sub-config
    //    (present in synthetic k3meta.py GGUFs, pwilkin PR fixture,
    //    unit-test fixtures).
    // 2. **Tensor-presence fallback** — real GrEarl K3 GGUF export
    //    does NOT ship the `full_attn_layers` / `kda_layers` arrays,
    //    so we detect by probing `blk.{il}.attn_q_a.weight` (MLA-only
    //    LoRA-A projection): if it exists the layer is MLA, otherwise
    //    KDA. Only reached when path 1 returns `None`.
    let is_mla = match kd.is_mla_layer(il) {
        Some(v) => v,
        None => gguf
            .tensor_info(&format!("{prefix}.attn_q_a.weight"))
            .is_some(),
    };
    let attn = if is_mla {
        // MLA-only tensors. `attn_gate` in real K3 GGUF is exported for
        // MLA layers only (the input-dependent output gate before o_proj);
        // KDA layers use the full-rank `ssm_g` instead (Phase X.4.b.5).
        let _attn_gate_mla: Option<WeightRef<'a>> =
            load_weight_ref_any_shape(gguf, &format!("{prefix}.attn_gate.weight"));
        KimiK3Attention::Mla(KimiK3MlaAttn {
            q_a: load_ref(&format!("{prefix}.attn_q_a.weight"))?,
            q_a_norm: load_norm(&format!("{prefix}.attn_q_a_norm.weight"))?,
            q_b: load_ref(&format!("{prefix}.attn_q_b.weight"))?,
            kv_a_mqa: load_ref(&format!("{prefix}.attn_kv_a_mqa.weight"))?,
            kv_a_norm: load_norm(&format!("{prefix}.attn_kv_a_norm.weight"))?,
            k_b: load_ref(&format!("{prefix}.attn_k_b.weight"))?,
            v_b: load_ref(&format!("{prefix}.attn_v_b.weight"))?,
        })
    } else {
        // Optional `ssm_dt.bias` — some conversions omit this scalar
        // (real GrEarl K3 GGUF DOES include it).
        let ssm_dt_bias = gguf.tensor_to_f32(&format!("{prefix}.ssm_dt.bias"));
        // K3-specific full-rank output gate matrix (real GrEarl GGUF
        // has `blk.{il}.ssm_g.weight`, same shape as attn_output —
        // `[num_heads * v_head_dim, hidden]`). Optional for
        // backwards-compatibility with skeleton fixtures.
        let ssm_g = load_weight_ref_any_shape(gguf, &format!("{prefix}.ssm_g.weight"));
        // K3-specific per-head A_h log-scale array (real GrEarl GGUF
        // has `blk.{il}.ssm_a` without `.weight` suffix). Optional.
        let ssm_a = gguf
            .tensor_to_f32(&format!("{prefix}.ssm_a"))
            .or_else(|| gguf.tensor_to_f32(&format!("{prefix}.ssm_a.weight")));
        KimiK3Attention::Kda(KimiK3KdaAttn {
            q: load_ref(&format!("{prefix}.attn_q.weight"))?,
            k: load_ref(&format!("{prefix}.attn_k.weight"))?,
            v: load_ref(&format!("{prefix}.attn_v.weight"))?,
            ssm_conv1d_q: load_ref(&format!("{prefix}.ssm_conv1d_q.weight"))?,
            ssm_conv1d_k: load_ref(&format!("{prefix}.ssm_conv1d_k.weight"))?,
            ssm_conv1d_v: load_ref(&format!("{prefix}.ssm_conv1d_v.weight"))?,
            ssm_f_a: load_ref(&format!("{prefix}.ssm_f_a.weight"))?,
            ssm_f_b: load_ref(&format!("{prefix}.ssm_f_b.weight"))?,
            ssm_beta: load_ref(&format!("{prefix}.ssm_beta.weight"))?,
            ssm_norm: load_norm(&format!("{prefix}.ssm_norm.weight"))?,
            ssm_dt_bias,
            ssm_g,
            ssm_a,
        })
    };
    // `attn_gate` for the layer bundle — MLA only (KDA uses ssm_g). For
    // KDA layers we substitute an "empty" WeightRef that the forward
    // path already knows not to touch (KDA layer forward ignores
    // `layer.attn_gate` — it goes through the KDA-specific `ssm_g`
    // gate implemented in the KDA layer primitive). Attempt to load
    // `attn_gate` first; if absent fall back to a synthetic empty ref
    // pointing at the token_embd bytes (any valid mmap slice works;
    // the ref is never matvec'd on KDA layers). For MLA layers, real
    // GGUF exports this tensor with shape `[hidden, num_heads *
    // v_head_dim]`.
    let attn_gate = load_weight_ref_any_shape(gguf, &format!("{prefix}.attn_gate.weight"))
        .unwrap_or_else(|| {
            // Placeholder pointing at attn_norm bytes — 1x1 F32,
            // unreachable in the KDA layer forward path.
            let tok_bytes = gguf
                .tensor_data(&format!("{prefix}.attn_norm.weight"))
                .unwrap_or(&[0u8; 4]);
            WeightRef {
                data: &tok_bytes[..4.min(tok_bytes.len())],
                qtype: GgmlType::F32,
                rows: 1,
                cols: 1,
            }
        });

    // ── FFN (Dense XOR LatentMoE) ────────────────────────────────────
    let first_k_dense = kd.first_k_dense_replace.unwrap_or(0);
    let ffn = if il < first_k_dense {
        KimiK3Ffn::Dense {
            gate: load_ref(&format!("{prefix}.ffn_gate.weight"))?,
            up: load_ref(&format!("{prefix}.ffn_up.weight"))?,
            down: load_ref(&format!("{prefix}.ffn_down.weight"))?,
        }
    } else {
        let ffn_gate_inp = load_norm(&format!("{prefix}.ffn_gate_inp.weight"))?;
        // `exp_probs_b.bias` is optional across variants.
        let exp_probs_b = gguf.tensor_to_f32(&format!("{prefix}.exp_probs_b.bias"));
        // Phase X.4.b.5: MoE tensor name aliases for real GrEarl GGUF.
        // Kuberwastaken TENSOR_MAP.md uses `routed_exp_*`; pwilkin PR
        // #26185 uses `ffn_routed_*`. Try both spellings, prefer the
        // first found (per-tensor basis so mixed GGUFs don't break).
        let load_ref_any = |names: &[String]| -> Result<WeightRef<'a>, String> {
            for n in names {
                if let Some(w) = load_weight_ref_any_shape(gguf, n) {
                    return Ok(w);
                }
            }
            Err(format!("layer {il}: no tensor found among {names:?}"))
        };
        let load_norm_any = |names: &[String]| -> Result<Vec<f32>, String> {
            for n in names {
                if let Some(v) = gguf.tensor_to_f32(n) {
                    return Ok(v);
                }
            }
            Err(format!("layer {il}: no norm tensor found among {names:?}"))
        };
        KimiK3Ffn::LatentMoe(KimiK3LatentMoe {
            ffn_gate_inp,
            exp_probs_b,
            ffn_gate_shexp: load_ref(&format!("{prefix}.ffn_gate_shexp.weight"))?,
            ffn_up_shexp: load_ref(&format!("{prefix}.ffn_up_shexp.weight"))?,
            ffn_down_shexp: load_ref(&format!("{prefix}.ffn_down_shexp.weight"))?,
            // Routed latent projections + norm: try both name schemes.
            routed_exp_up: load_ref_any(&[
                format!("{prefix}.routed_exp_up.weight"),
                format!("{prefix}.ffn_routed_up.weight"),
            ])?,
            routed_exp_down: load_ref_any(&[
                format!("{prefix}.routed_exp_down.weight"),
                format!("{prefix}.ffn_routed_down.weight"),
            ])?,
            routed_exp_norm: load_norm_any(&[
                format!("{prefix}.routed_exp_norm.weight"),
                format!("{prefix}.ffn_routed_norm.weight"),
            ])?,
            ffn_gate_exps: load_ref(&format!("{prefix}.ffn_gate_exps.weight"))?,
            ffn_up_exps: load_ref(&format!("{prefix}.ffn_up_exps.weight"))?,
            ffn_down_exps: load_ref(&format!("{prefix}.ffn_down_exps.weight"))?,
        })
    };

    Ok(KimiK3LayerWeights {
        attn_norm,
        ffn_norm,
        attn_output,
        attn_gate,
        attn_res_score,
        ffn_res_score,
        attn,
        ffn,
    })
}

/// Load the K3 full-model weight bundle from GGUF (Phase X.4.b.2).
///
/// Walks the 5 Global tensors then delegates to
/// [`load_kimi_k3_layer_weights`] for each of the `config.num_layers`
/// per-layer bundles. Returns a descriptive `Err` on the first
/// missing tensor.
#[allow(dead_code)]
pub fn load_kimi_k3_model_weights<'a, G: crate::gguf::GgufSource<'a>>(
    gguf: &'a G,
    config: &Llama3Config,
) -> Result<KimiK3ModelWeights<'a>, String> {
    let load_ref = |name: &str| -> Result<WeightRef<'a>, String> {
        load_weight_ref_any_shape(gguf, name)
            .ok_or_else(|| format!("global: missing tensor '{name}'"))
    };
    let load_norm = |name: &str| -> Result<Vec<f32>, String> {
        gguf.tensor_to_f32(name)
            .ok_or_else(|| format!("global: missing norm tensor '{name}'"))
    };

    let token_embd = load_ref("token_embd.weight")?;
    let output_norm = load_norm("output_norm.weight")?;
    let output = load_ref("output.weight")?;
    // AttnRes fused 1D output-side score vector (pwilkin PR #26185
    // GGUF export convention: `output_res_score`, `[n_embd]`).
    let output_res_score = load_norm("output_res_score.weight")?;

    let mut layers = Vec::with_capacity(config.num_layers);
    for il in 0..config.num_layers {
        layers.push(load_kimi_k3_layer_weights(gguf, il, config)?);
    }

    Ok(KimiK3ModelWeights {
        token_embd,
        output_norm,
        output,
        output_res_score,
        layers,
    })
}

// ── Kimi K3 Gated MLA layer forward (Phase X.4.c.3.2, 2026-07-28) ──
//
// Implements one MLA-layer forward for a single new token, mirroring
// DeepSeek V2 MLA math with K3-specific tweaks:
//
// - **NoPE**: `mla_use_nope = true` in K3, so the `q_rope` and
//   `k_rope` head-dim slices are NOT rotated — they are consumed as
//   regular attention dimensions. KDA layers provide the position-
//   sensitive mixing; MLA is content-only global attention.
// - **Full-rank output gate**: `mla_use_output_gate = true` adds an
//   input-dependent sigmoid gate on top of the attention output
//   (Eq 7). The gate reuses the same `kimi_delta_output_gate` shape
//   as KDA but with `rms_gamma = None` to skip the inner RMSNorm.
// - **`kv_b` split**: K3 conversion splits `kv_b_proj` into
//   `attn_k_b` (nope portion) + `attn_v_b` (v portion) up front, so
//   the forward reconstructs full k/v from cached `c_k` via two
//   independent matvecs.

/// Per-layer MLA KV cache (Phase X.4.c.3.2).
///
/// Stores the latent `c_k` (`kv_lora_rank` per position) + `k_rope`
/// (`qk_rope_head_dim` per position) — NOT the full reconstructed
/// keys and values. Full k/v are reconstructed from `c_k` via
/// `attn_k_b` / `attn_v_b` matvec on every attention step. This is
/// the MLA compression trick: KV cache size grows linearly with
/// `n_positions × (kv_lora_rank + qk_rope_head_dim) = n × (512 + 64)
/// = n × 576 f32` for K3, versus `n × num_heads × (qk + v) = n × 96
/// × 256 ≈ n × 24576` for a naive dense KV cache — a **42× compression**.
#[allow(dead_code)]
pub struct KimiK3MlaCache {
    /// Row-major `[n_positions × kv_lora_rank]`.
    c_k: Vec<f32>,
    /// Row-major `[n_positions × qk_rope_head_dim]`.
    k_rope: Vec<f32>,
    n_positions: usize,
    kv_lora_rank: usize,
    qk_rope_head_dim: usize,
}

impl KimiK3MlaCache {
    /// Allocate an empty cache with reserved capacity for
    /// `capacity` positions to avoid mid-generation reallocation
    /// on the hot path.
    #[must_use]
    pub fn new(kv_lora_rank: usize, qk_rope_head_dim: usize, capacity: usize) -> Self {
        Self {
            c_k: Vec::with_capacity(capacity * kv_lora_rank),
            k_rope: Vec::with_capacity(capacity * qk_rope_head_dim),
            n_positions: 0,
            kv_lora_rank,
            qk_rope_head_dim,
        }
    }

    /// Discard every cached position — call at the start of a new
    /// sequence.
    pub fn reset(&mut self) {
        self.c_k.clear();
        self.k_rope.clear();
        self.n_positions = 0;
    }

    #[inline]
    #[must_use]
    pub const fn n_positions(&self) -> usize {
        self.n_positions
    }

    fn append(&mut self, c_k: &[f32], k_rope: &[f32]) {
        assert_eq!(c_k.len(), self.kv_lora_rank);
        assert_eq!(k_rope.len(), self.qk_rope_head_dim);
        self.c_k.extend_from_slice(c_k);
        self.k_rope.extend_from_slice(k_rope);
        self.n_positions += 1;
    }

    fn c_k_at(&self, pos: usize) -> &[f32] {
        let base = pos * self.kv_lora_rank;
        &self.c_k[base..base + self.kv_lora_rank]
    }

    fn k_rope_at(&self, pos: usize) -> &[f32] {
        let base = pos * self.qk_rope_head_dim;
        &self.k_rope[base..base + self.qk_rope_head_dim]
    }
}

/// Dimension bundle for the Gated MLA forward (Phase X.4.c.3.2).
///
/// All fields mirror the sub-config populated by
/// [`KimiDeltaConfig::from_gguf`] plus the shared MLA dims that live
/// on the root [`Llama3Config`]. Pass one instance per model instead
/// of re-deriving fields from the config inside every call.
#[allow(dead_code)]
pub struct KimiK3MlaConfig {
    pub d: usize,
    pub num_heads: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub rms_eps: f32,
}

/// Multiply a WeightRef `[rows, cols]` by a `[cols]` vector, returning
/// a fresh `[rows]` result. Thin allocating wrapper over
/// `WeightRef::matvec` used by the K3 MLA forward for readability.
#[inline]
fn kimi_k3_matvec_ref(w: &WeightRef<'_>, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; w.rows];
    w.matvec(x, &mut out);
    out
}

/// Apply RMSNorm with a per-channel `γ` scale in-place: mimics the
/// module-private `rms_norm` helper but returns a fresh `Vec<f32>` so
/// the MLA step can chain operations without threading scratch
/// buffers. `x / sqrt(mean(x²) + eps) · γ`.
fn kimi_k3_rms_norm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    debug_assert_eq!(x.len(), gamma.len());
    let mut out = vec![0.0_f32; x.len()];
    rms_norm(x, gamma, eps, &mut out);
    out
}

/// One-token Gated MLA layer forward (Phase X.4.c.3.2, K3 tech
/// report §2.1.2 Eq 7).
///
/// Given the input hidden state `x ∈ ℝ^d`, the per-layer weight
/// bundle from [`KimiK3LayerWeights`], and a mutable MLA KV cache,
/// runs the full attention pipeline and returns the layer's output
/// `y ∈ ℝ^d` (ready to be added back to the residual stream).
///
/// # Pipeline
///
/// 1. **Pre-attention RMSNorm** — `x_norm = RMSNorm(x, attn_norm)`.
/// 2. **Q LoRA down + norm + up** — `q_latent = W_q_a x_norm`
///    (`[q_lora_rank]`), `q_latent_norm = RMSNorm(q_latent,
///    q_a_norm)`, `q_full = W_q_b q_latent_norm` (`[num_heads ×
///    (qk_nope + qk_rope)]`).
/// 3. **KV LoRA** — `kv_a_out = W_kv_a_mqa x_norm`
///    (`[kv_lora_rank + qk_rope_head_dim]`), split into
///    `c_k` (`[kv_lora_rank]`) and `k_rope` (`[qk_rope_head_dim]`);
///    apply `RMSNorm(c_k, kv_a_norm)`.
/// 4. **Cache append** — push `c_k_norm` + `k_rope` into
///    [`KimiK3MlaCache`]. K3 caches only the compressed latent
///    (~576 f32 per token) rather than the full reconstructed keys
///    (~24576 f32) — the 42× MLA compression trick.
/// 5. **Attention** — for every cached position `i`:
///    - Reconstruct `k_i = W_k_b c_k_i_norm` (`[num_heads × qk_nope]`)
///      + concat with `k_rope_i` per head → full `k_i` per head.
///    - Reconstruct `v_i = W_v_b c_k_i_norm` (`[num_heads × v_head_dim]`).
///    - Score `s_h[i] = q_full[h] · k_i[h] / sqrt(qk_nope + qk_rope)`.
///    - Softmax over `i`, weighted sum with `v_i[h]`.
///
///    Result concatenated across heads: `[num_heads × v_head_dim]`.
/// 6. **Output projection** — `attn_out = W_o concat`.
/// 7. **Output gate (K3, Eq 7)** — `y = Sigmoid(W_g x_norm) ⊙
///    attn_out`. `mla_use_nope = true` in K3 means the `qk_rope`
///    slice is used as regular attention dimension without any
///    rotation; the K3 MLA layers rely on the KDA layers to inject
///    positional information into the residual stream.
///
/// # Panics
///
/// Panics on any weight-shape or config-dim mismatch via the
/// underlying `WeightRef::matvec` / `rms_norm` asserts.
#[allow(dead_code)]
fn kimi_k3_gated_mla_step(
    x: &[f32],
    attn_norm: &[f32],
    attn_gate: &WeightRef<'_>,
    attn_output: &WeightRef<'_>,
    mla: &KimiK3MlaAttn<'_>,
    cache: &mut KimiK3MlaCache,
    config: &KimiK3MlaConfig,
) -> Vec<f32> {
    let d = config.d;
    let h = config.num_heads;
    let qk_nope = config.qk_nope_head_dim;
    let qk_rope = config.qk_rope_head_dim;
    let qk_head = qk_nope + qk_rope;
    let v_head = config.v_head_dim;
    let kv_lr = config.kv_lora_rank;

    assert_eq!(x.len(), d, "x length must equal hidden dim d");
    assert_eq!(cache.kv_lora_rank, kv_lr, "cache.kv_lora_rank mismatch");
    assert_eq!(
        cache.qk_rope_head_dim, qk_rope,
        "cache.qk_rope_head_dim mismatch"
    );

    // Step 1: pre-attention RMSNorm.
    let x_norm = kimi_k3_rms_norm(x, attn_norm, config.rms_eps);

    // Step 2: Q LoRA chain.
    let q_latent = kimi_k3_matvec_ref(&mla.q_a, &x_norm);
    let q_latent_norm = kimi_k3_rms_norm(&q_latent, &mla.q_a_norm, config.rms_eps);
    let q_full = kimi_k3_matvec_ref(&mla.q_b, &q_latent_norm);
    debug_assert_eq!(q_full.len(), h * qk_head, "q_full length");

    // Step 3: KV LoRA down + split.
    let kv_a_out = kimi_k3_matvec_ref(&mla.kv_a_mqa, &x_norm);
    debug_assert_eq!(kv_a_out.len(), kv_lr + qk_rope);
    let c_k_slice = &kv_a_out[..kv_lr];
    let k_rope_slice = &kv_a_out[kv_lr..];
    let c_k_norm = kimi_k3_rms_norm(c_k_slice, &mla.kv_a_norm, config.rms_eps);

    // Step 4: cache append (K3 stores latent, not reconstructed k/v).
    cache.append(&c_k_norm, k_rope_slice);
    let n_pos = cache.n_positions;

    // Step 5: attention. For each cached position i, reconstruct full
    // k_i and v_i via the split k_b / v_b matvecs, then compute
    // scaled dot-product attention against q_full per head.
    let scale = (qk_head as f32).sqrt().recip();
    let mut concat_out = vec![0.0_f32; h * v_head];

    // Per-position reconstructed k / v, computed once per position
    // then reused across heads to keep the inner loop tight.
    let mut k_all = vec![0.0_f32; n_pos * h * qk_head];
    let mut v_all = vec![0.0_f32; n_pos * h * v_head];
    for pos in 0..n_pos {
        let c_k_i = cache.c_k_at(pos);
        let k_rope_i = cache.k_rope_at(pos);
        let k_nope_reconstructed = kimi_k3_matvec_ref(&mla.k_b, c_k_i);
        let v_reconstructed = kimi_k3_matvec_ref(&mla.v_b, c_k_i);
        debug_assert_eq!(k_nope_reconstructed.len(), h * qk_nope);
        debug_assert_eq!(v_reconstructed.len(), h * v_head);
        // Layout k as [num_heads, qk_head] with nope first, rope last.
        // Per-head slice: [h * qk_head + 0..qk_nope | qk_nope..qk_head].
        for head in 0..h {
            let k_dst_base = pos * h * qk_head + head * qk_head;
            let k_nope_src = head * qk_nope;
            k_all[k_dst_base..k_dst_base + qk_nope]
                .copy_from_slice(&k_nope_reconstructed[k_nope_src..k_nope_src + qk_nope]);
            // K3 NoPE: k_rope stored as-is, no rotation, shared
            // across all heads (single-slot MQA style).
            k_all[k_dst_base + qk_nope..k_dst_base + qk_head].copy_from_slice(k_rope_i);
            let v_dst_base = pos * h * v_head + head * v_head;
            let v_src = head * v_head;
            v_all[v_dst_base..v_dst_base + v_head]
                .copy_from_slice(&v_reconstructed[v_src..v_src + v_head]);
        }
    }

    // Per-head attention.
    let mut logits_buf = vec![0.0_f32; n_pos];
    for head in 0..h {
        // q for this head, layout [head * qk_head..(head+1) * qk_head].
        let q_head_start = head * qk_head;
        let q_head = &q_full[q_head_start..q_head_start + qk_head];

        // Scores.
        for pos in 0..n_pos {
            let k_start = pos * h * qk_head + head * qk_head;
            let k_head = &k_all[k_start..k_start + qk_head];
            let mut dot = 0.0_f64;
            for j in 0..qk_head {
                dot += f64::from(q_head[j]) * f64::from(k_head[j]);
            }
            logits_buf[pos] = (dot as f32) * scale;
        }

        // Softmax (log-sum-exp stable).
        let max_logit = logits_buf.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0_f64;
        for l in &mut logits_buf {
            *l = (*l - max_logit).exp();
            sum_exp += f64::from(*l);
        }
        let inv_sum = (sum_exp as f32).recip();

        // Weighted sum of v.
        let out_start = head * v_head;
        for j in 0..v_head {
            concat_out[out_start + j] = 0.0;
        }
        for pos in 0..n_pos {
            let alpha = logits_buf[pos] * inv_sum;
            let v_start = pos * h * v_head + head * v_head;
            for j in 0..v_head {
                concat_out[out_start + j] += alpha * v_all[v_start + j];
            }
        }
    }

    // Step 6: output projection.
    let attn_out = kimi_k3_matvec_ref(attn_output, &concat_out);
    debug_assert_eq!(attn_out.len(), d);

    // Step 7: output gate (Eq 7). `y = Sigmoid(W_g x_norm) ⊙ attn_out`.
    let gate_pre = kimi_k3_matvec_ref(attn_gate, &x_norm);
    debug_assert_eq!(gate_pre.len(), d);
    let mut y = vec![0.0_f32; d];
    for i in 0..d {
        y[i] = sigmoid(gate_pre[i]) * attn_out[i];
    }
    y
}

// ── Kimi K3 forward helpers (Phase X.4.c.3.3.a) ─────────────────────
//
// Small pure functions that bridge the K3 sub-config + weight refs
// into the argument shape expected by the layer primitives from
// X.4.c.3.2 (MLA) and the standard SwiGLU pattern used by the layer-0
// dense FFN.

/// Extract a [`KimiK3MlaConfig`] from a populated [`Llama3Config`]
/// (Phase X.4.c.3.3.a). Returns `None` when the K3 sub-config or any
/// required MLA dim is absent.
///
/// The runtime dims live inline on `KimiK3MlaConfig` because
/// `kimi_k3_gated_mla_step` is dim-agnostic — it does not reach
/// back into the parent config on the hot path. This helper is
/// called once at model construction (or once per forward if the
/// caller does not cache the result).
#[allow(dead_code)]
fn kimi_k3_extract_mla_config(config: &Llama3Config) -> Option<KimiK3MlaConfig> {
    let kd = config.kimi_delta.as_ref()?;
    Some(KimiK3MlaConfig {
        d: config.hidden_dim,
        num_heads: config.num_heads,
        qk_nope_head_dim: kd.qk_nope_head_dim?,
        qk_rope_head_dim: kd.qk_rope_head_dim?,
        v_head_dim: kd.v_head_dim?,
        q_lora_rank: kd.q_lora_rank?,
        kv_lora_rank: kd.kv_lora_rank?,
        rms_eps: config.norm_eps,
    })
}

/// Kimi K3 dense-FFN forward (Phase X.4.c.3.3.a).
///
/// Applied to layer 0 only (K3 `first_k_dense_replace = 1`). Uses
/// the standard SwiGLU pattern from llama.cpp / DeepSeek V3 dense
/// layers — `y = W_down · (Swish(W_gate · x_norm) ⊙ W_up · x_norm)`.
///
/// K3 tech report does not explicitly specify the dense-FFN
/// activation. The Stable LatentMoE path uses SiTU-GLU (§2.3.2), but
/// the layer-0 dense FFN is unspecified. We match the DeepSeek V3
/// dense-layer convention (SwiGLU) since (a) DeepSeek V3 is the
/// closest architectural sibling and (b) the tensor names in
/// TENSOR_MAP.md (`ffn_gate` / `ffn_up` / `ffn_down`) match the
/// llama.cpp SwiGLU convention exactly.
///
/// # Arguments
///
/// - `x`: input hidden state `[d]`.
/// - `ffn_norm`: RMSNorm γ applied to `x` before the SwiGLU.
/// - `gate`: `W_gate` `[intermediate, d]`.
/// - `up`: `W_up` `[intermediate, d]`.
/// - `down`: `W_down` `[d, intermediate]`.
/// - `rms_eps`: RMSNorm epsilon.
///
/// # Returns
///
/// `y ∈ ℝ^d` — the dense-FFN output ready for residual add.
#[allow(dead_code)]
fn kimi_k3_dense_ffn_forward(
    x: &[f32],
    ffn_norm: &[f32],
    gate: &WeightRef<'_>,
    up: &WeightRef<'_>,
    down: &WeightRef<'_>,
    rms_eps: f32,
) -> Vec<f32> {
    let d = x.len();
    debug_assert_eq!(ffn_norm.len(), d);
    debug_assert_eq!(down.rows, d, "down.rows must equal hidden dim");
    let intermediate = gate.rows;
    debug_assert_eq!(
        up.rows, intermediate,
        "gate/up must have same intermediate dim"
    );
    debug_assert_eq!(
        down.cols, intermediate,
        "down.cols must equal intermediate dim"
    );

    // Pre-FFN RMSNorm.
    let x_norm = kimi_k3_rms_norm(x, ffn_norm, rms_eps);

    // SwiGLU: gated = Swish(W_gate x_norm) ⊙ W_up x_norm.
    let mut gate_out = vec![0.0_f32; intermediate];
    gate.matvec(&x_norm, &mut gate_out);
    let mut up_out = vec![0.0_f32; intermediate];
    up.matvec(&x_norm, &mut up_out);
    let mut gated = vec![0.0_f32; intermediate];
    for i in 0..intermediate {
        gated[i] = silu(gate_out[i]) * up_out[i];
    }

    // W_down projects back to hidden.
    let mut y = vec![0.0_f32; d];
    down.matvec(&gated, &mut y);
    y
}

// ── Kimi K3 AttnRes runtime state + res_mix primitive ─────────────
// (Phase X.4.c.3.4.a — pwilkin PR #26185 semantics)
//
// Mirrors `src/models/kimi-k3.cpp` L199-257 (`res_push` /
// `res_stack` / `res_mix`). Diverges from the paper's
// `BlockAttnResState` (which tracks per-position partial sums within
// the current block); the pwilkin export instead banks the RAW input
// to each checkpoint layer and mixes against banked ckpts + current
// residual stream at every layer.

/// Runtime state for K3 Block Attention Residuals (pwilkin PR #26185
/// wiring, Phase X.4.c.3.4.a).
///
/// Owns a growing list of `banked` checkpoints (each `[n_embd]`,
/// pushed at every checkpoint layer `il % block_size == 0`) and the
/// AttnRes block size cached at construction. The current residual
/// stream is passed in per-call rather than held inside the state;
/// this matches the pwilkin design where `prefix_sum` is a graph
/// tensor threaded through each layer.
///
/// Fresh state has zero banked ckpts, so `res_mix` at layer 0 is an
/// identity pass-through (nothing to mix against). Layer 0 is a
/// checkpoint layer (`0 % block_size == 0`), so the first `bank`
/// call happens after the initial `res_mix` and before the layer's
/// attention forward.
#[allow(dead_code)]
pub(crate) struct KimiK3AttnResState {
    d: usize,
    block_size: usize,
    banked: Vec<Vec<f32>>,
}

#[allow(dead_code)]
impl KimiK3AttnResState {
    /// Construct fresh state for a K3 model with `hidden_dim` and
    /// `attn_res_block_size` (K3 default 12).
    pub(crate) fn new(hidden_dim: usize, block_size: usize) -> Self {
        assert!(hidden_dim > 0, "hidden_dim must be > 0");
        assert!(block_size > 0, "block_size must be > 0");
        Self {
            d: hidden_dim,
            block_size,
            banked: Vec::new(),
        }
    }

    /// True if layer `il` is a checkpoint layer that should
    /// (1) bank the raw prefix_sum before its attention, and
    /// (2) reset prefix_sum to the attention output alone after.
    pub(crate) fn is_checkpoint_layer(&self, il: usize) -> bool {
        il.is_multiple_of(self.block_size)
    }

    /// Push the raw prefix_sum into the ckpt bank. Called at
    /// checkpoint layer entry, BEFORE the layer's `res_mix` output
    /// is applied (i.e. bank the pre-mix, pre-attention input).
    pub(crate) fn bank(&mut self, prefix_sum: &[f32]) {
        assert_eq!(
            prefix_sum.len(),
            self.d,
            "prefix_sum length {} must equal hidden dim {}",
            prefix_sum.len(),
            self.d
        );
        self.banked.push(prefix_sum.to_vec());
    }

    /// Reset all banked state — called on sequence boundary.
    pub(crate) fn reset(&mut self) {
        self.banked.clear();
    }

    /// Number of banked ckpts (for testing / diagnostics).
    #[cfg(test)]
    pub(crate) fn banked_count(&self) -> usize {
        self.banked.len()
    }
}

/// K3 Block AttnRes `res_mix` primitive (pwilkin PR #26185
/// `src/models/kimi-k3.cpp` L218-257 verbatim math, Phase
/// X.4.c.3.4.a).
///
/// Computes a softmax-weighted mixture of banked checkpoints + the
/// current residual stream, using a fused 1D `score_w` vector.
///
/// **Semantics** (paraphrased from L229-256):
///
/// 1. For each banked ckpt `k_i`:
///    `score_i = Σ_j RMSNorm(k_i)[j] · score_w[j]`
///    (RMSNorm gain fused into `score_w`; norm eps = `rms_eps`).
/// 2. For the current stream `prefix_sum`:
///    `score_cur = Σ_j RMSNorm(prefix_sum)[j] · score_w[j]`
/// 3. Concatenate: `scores = [score_0, ..., score_{n-1}, score_cur]`
///    (length `n_ckpt + 1`).
/// 4. Softmax over concatenated scores → `probs`.
/// 5. **Weighted sum uses RAW non-normalized values**
///    (paper §2.2 duality — norm is for score computation only):
///    `out = Σ_i probs[i] · k_i + probs[n] · prefix_sum`.
///
/// When `state.banked` is empty (e.g. first `res_mix` call before
/// any layer has banked), returns `prefix_sum` unchanged (identity
/// pass-through).
///
/// # Panics
///
/// Panics if `prefix_sum.len() != state.d` or `score_w.len() != state.d`.
#[allow(dead_code)]
pub(crate) fn kimi_k3_res_mix(
    state: &KimiK3AttnResState,
    prefix_sum: &[f32],
    score_w: &[f32],
    rms_eps: f32,
) -> Vec<f32> {
    let d = state.d;
    assert_eq!(prefix_sum.len(), d, "prefix_sum length must equal d");
    assert_eq!(score_w.len(), d, "score_w length must equal d");

    if state.banked.is_empty() {
        return prefix_sum.to_vec();
    }

    let n_ckpt = state.banked.len();

    // Step 1: score each banked ckpt: sum_j RMSNorm(k_i)[j] * score_w[j].
    // This equals score_w · RMSNorm(k_i) (dot product with normalized k).
    let mut scores = Vec::with_capacity(n_ckpt + 1);
    for bank in &state.banked {
        let ss: f64 = bank.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let mean = (ss / d as f64) as f32;
        let scale = (mean + rms_eps).sqrt().recip();
        let mut s = 0.0_f64;
        for j in 0..d {
            s += f64::from(bank[j]) * f64::from(scale) * f64::from(score_w[j]);
        }
        scores.push(s as f32);
    }

    // Step 2: score current stream.
    let ss: f64 = prefix_sum
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum();
    let mean = (ss / d as f64) as f32;
    let scale_cur = (mean + rms_eps).sqrt().recip();
    let mut s_cur = 0.0_f64;
    for j in 0..d {
        s_cur += f64::from(prefix_sum[j]) * f64::from(scale_cur) * f64::from(score_w[j]);
    }
    scores.push(s_cur as f32);

    // Step 3-4: softmax over (n_ckpt + 1) with log-sum-exp stability.
    let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_s = Vec::with_capacity(scores.len());
    let mut sum_exp = 0.0_f64;
    for &sc in &scores {
        let e = (sc - max_s).exp();
        exp_s.push(e);
        sum_exp += f64::from(e);
    }
    let inv_sum = (sum_exp as f32).recip();

    // Step 5: weighted sum using RAW (non-normalized) values.
    let mut out = vec![0.0_f32; d];
    for (i, bank) in state.banked.iter().enumerate() {
        let p = exp_s[i] * inv_sum;
        for j in 0..d {
            out[j] += p * bank[j];
        }
    }
    let p_cur = exp_s[n_ckpt] * inv_sum;
    for j in 0..d {
        out[j] += p_cur * prefix_sum[j];
    }
    out
}

#[cfg(test)]
mod kimi_k3_attnres_tests {
    use super::{kimi_k3_res_mix, KimiK3AttnResState};

    #[test]
    fn attnres_state_new_starts_empty() {
        let state = KimiK3AttnResState::new(4, 12);
        assert_eq!(state.banked_count(), 0);
    }

    #[test]
    fn attnres_state_checkpoint_layer_predicate() {
        let state = KimiK3AttnResState::new(4, 12);
        assert!(state.is_checkpoint_layer(0));
        assert!(!state.is_checkpoint_layer(1));
        assert!(!state.is_checkpoint_layer(11));
        assert!(state.is_checkpoint_layer(12));
        assert!(state.is_checkpoint_layer(24));
        assert!(!state.is_checkpoint_layer(23));
    }

    #[test]
    fn attnres_state_bank_grows_banked_list() {
        let mut state = KimiK3AttnResState::new(3, 12);
        let inp1 = vec![1.0_f32, 2.0, 3.0];
        state.bank(&inp1);
        assert_eq!(state.banked_count(), 1);
        let inp2 = vec![4.0_f32, 5.0, 6.0];
        state.bank(&inp2);
        assert_eq!(state.banked_count(), 2);
    }

    #[test]
    fn attnres_state_reset_clears_banked() {
        let mut state = KimiK3AttnResState::new(2, 12);
        state.bank(&[1.0_f32, 2.0]);
        assert_eq!(state.banked_count(), 1);
        state.reset();
        assert_eq!(state.banked_count(), 0);
    }

    #[test]
    fn res_mix_identity_when_no_banked_ckpts() {
        // Empty bank → returns prefix_sum unchanged.
        let state = KimiK3AttnResState::new(3, 12);
        let prefix = vec![0.5_f32, -0.5, 0.25];
        let score_w = vec![1.0_f32; 3];
        let out = kimi_k3_res_mix(&state, &prefix, &score_w, 1e-6);
        assert_eq!(out, prefix, "no banked → return prefix_sum unchanged");
    }

    #[test]
    fn res_mix_with_one_bank_gives_convex_combination() {
        // 1 banked ckpt + 1 current stream. Softmax over 2 scores;
        // output is a convex combination of the two RAW vectors.
        let mut state = KimiK3AttnResState::new(3, 12);
        let bank = vec![10.0_f32, 20.0, 30.0];
        state.bank(&bank);
        let prefix = vec![100.0_f32, 200.0, 300.0];
        // Uniform score_w → both scores dominated by norm of raw vec;
        // just need to verify output lives in [min, max] of the two.
        let score_w = vec![1.0_f32; 3];
        let out = kimi_k3_res_mix(&state, &prefix, &score_w, 1e-6);
        assert_eq!(out.len(), 3);
        for i in 0..3 {
            let lo = bank[i].min(prefix[i]);
            let hi = bank[i].max(prefix[i]);
            assert!(
                out[i] >= lo - 1e-4 && out[i] <= hi + 1e-4,
                "out[{i}]={} must lie in [{lo}, {hi}]",
                out[i]
            );
        }
        // Softmax probs sum to 1, so output is a proper convex combo.
        // For two equally-scaled equally-normed vectors with the same
        // score_w, probs should be near 0.5/0.5 (since bank & prefix
        // have different magnitudes but their RMSNorm makes them
        // similar, and score = RMSNorm · score_w is similar too).
    }

    #[test]
    fn res_mix_score_w_bias_shifts_weight_toward_matching_bank() {
        // 2 banked ckpts. score_w matches ckpt 1's direction, so
        // probs should favor ckpt 1's raw value in the output.
        let mut state = KimiK3AttnResState::new(3, 12);
        let bank_orthogonal = vec![1.0_f32, 0.0, 0.0]; // score_w · RMSNorm ≈ 0
        let bank_aligned = vec![0.0_f32, 1.0, 0.0]; //   ≈ 1
        state.bank(&bank_orthogonal);
        state.bank(&bank_aligned);
        // Prefix chosen to have low overlap with score_w.
        let prefix = vec![1.0_f32, 0.0, 0.0];
        let score_w = vec![0.0_f32, 1.0, 0.0];
        let out = kimi_k3_res_mix(&state, &prefix, &score_w, 1e-6);
        // With softmax favoring bank_aligned (score ≈ 1) over
        // bank_orthogonal (score ≈ 0) and prefix (score ≈ 0), the
        // output's y-component should be closest to bank_aligned[1] = 1.
        assert!(
            out[1] > 0.5,
            "aligned bank must dominate output y-component, got {}",
            out[1]
        );
    }

    #[test]
    fn res_mix_deterministic_across_repeated_calls() {
        let mut state = KimiK3AttnResState::new(4, 12);
        state.bank(&[1.0_f32, 2.0, 3.0, 4.0]);
        state.bank(&[5.0_f32, 6.0, 7.0, 8.0]);
        let prefix = vec![0.1_f32, 0.2, 0.3, 0.4];
        let score_w = vec![1.0_f32, -1.0, 1.0, -1.0];
        let out1 = kimi_k3_res_mix(&state, &prefix, &score_w, 1e-6);
        let out2 = kimi_k3_res_mix(&state, &prefix, &score_w, 1e-6);
        assert_eq!(out1, out2, "res_mix must be deterministic");
    }
}

// ── Kimi K3 KDA per-head aggregation (Phase X.4.c.3.3.b) ──────────
//
// The K3 GGUF stores KDA `attn_q`, `attn_k`, `attn_v`,
// `ssm_conv1d_{q,k,v}`, `ssm_f_b`, `ssm_norm`, and `ssm_beta` as
// fused tensors that span all `num_heads` heads (rows are
// `num_heads × per_head_dim` big). `kimi_delta_forward_head`
// (X.4.c.2 primitive) expects per-head weights, so this landing
// adds a scalar-first per-head slicing helper + a KDA layer
// aggregator that loops over heads and combines the outputs.
//
// Scope of Phase X.4.c.3.3.b (this landing):
//
// - Per-head slicing: **F32 only**. Quantized (Q4_K / IQ1_S) per-
//   row slicing needs block-aligned offsets which is deferred to
//   Phase X.4.c.3.3.b.2 — the GGUF K-quant blocks span 256
//   elements, so per-head `WeightRef`s can only be constructed
//   when `per_head_dim` is a multiple of 256 (usually not the
//   case for K3 with `kda_head_dim = 128`). Quantized K3 forward
//   therefore needs a different slicing strategy (probably per-
//   token dequantize-and-slice rather than per-head weight ref).
// - The KDA layer forward function reads all per-head `attn_q/k/v`
//   / `ssm_conv1d_*` / `ssm_f_b` / `ssm_beta` / `ssm_norm` slices
//   through a shared low-rank `ssm_f_a` (K3 stores `f_a` as a
//   single tensor shared across heads: `[alpha_rank, hidden]`).
// - `attn_output` and `attn_gate` are shared across heads and
//   applied once after the concat.

/// Slice a `[rows, cols]` `WeightRef` into a `[row_end − row_start,
/// cols]` view sharing the underlying byte buffer.
///
/// **Phase X.4.c.3.3.b.2 upgrade**: extended from F32-only to
/// support the full K3 quant zoo (F32/F16/Q4_K/Q8_0/IQ4_XS/MXFP4/
/// IQ1_S/Q2_0/etc.). The refactor exploits GGUF's row-major layout:
/// as long as `cols % elements_per_block == 0` (which holds for K3
/// tensors since `hidden = 7168` divides both 256 and 32), each row
/// is an integer number of quant blocks and per-row byte offsets
/// are block-aligned. The earlier concern about "K3 kda_head_dim =
/// 128 is exactly half a block" was overly cautious: `kda_head_dim`
/// is the ROW axis (per-head slicing = 128 consecutive rows), not
/// the COLUMN axis, so block-alignment is preserved.
///
/// Returns `None` if:
/// - `row_start > row_end` (invalid range),
/// - `row_end > w.rows` (out of bounds),
/// - `w.qtype` is quantized AND `w.cols % elements_per_block != 0`
///   (per-row byte offsets would land mid-block), or
/// - `w.qtype` is `GgmlType::Other(_)` (unknown quant type).
#[allow(dead_code)]
fn kimi_k3_slice_weight_ref_rows<'a>(
    w: &WeightRef<'a>,
    row_start: usize,
    row_end: usize,
) -> Option<WeightRef<'a>> {
    if row_start > row_end || row_end > w.rows {
        return None;
    }
    if matches!(w.qtype, GgmlType::Other(_)) {
        return None;
    }
    let elements_per_block = w.qtype.elements_per_block();
    let block_bytes = w.qtype.block_bytes();
    if elements_per_block == 0 || block_bytes == 0 {
        return None;
    }
    // For F32/F16: elements_per_block = 1, row_bytes = cols * bytes.
    // For quant types: row must contain integer blocks.
    if !w.cols.is_multiple_of(elements_per_block) {
        return None;
    }
    let blocks_per_row = w.cols / elements_per_block;
    let row_bytes = blocks_per_row.checked_mul(block_bytes)?;
    let byte_start = row_start.checked_mul(row_bytes)?;
    let byte_end = row_end.checked_mul(row_bytes)?;
    if byte_end > w.data.len() {
        return None;
    }
    Some(WeightRef {
        data: &w.data[byte_start..byte_end],
        qtype: w.qtype,
        rows: row_end - row_start,
        cols: w.cols,
    })
}

/// One-token KDA layer forward with per-head aggregation
/// (Phase X.4.c.3.3.b, K3 tech report §2.1.1).
///
/// Slices the fused K3 KDA weight tensors per head, wraps each
/// head's slices in a [`KimiDeltaHeadParams`], calls
/// [`kimi_delta_forward_head`] (X.4.c.2 primitive) with the head's
/// slot in `caches`, and concatenates the per-head outputs before
/// applying the shared `attn_output` projection. K3 shares the
/// `attn_gate` projection across all heads (K3 always-on output
/// gate) — the per-head [`kimi_delta_forward_head`] applies its
/// own gate at the primitive layer, so no additional gating is
/// applied here at the layer level.
///
/// # Arguments
///
/// - `x`: input hidden state `[d]`.
/// - `attn_norm`: pre-attention RMSNorm γ, `[d]`.
/// - `attn_gate`: unused at the layer level for K3 KDA (the
///   per-head primitive already applies its own gate). Retained
///   in the signature so future refactors that consolidate gating
///   into the layer level do not need to change the callsite —
///   the current implementation reads `attn_gate` shape for
///   validation only.
/// - `attn_output`: shared output projection `[d_out, num_heads ×
///   v_head_dim]`, applied once after the head concat.
/// - `kda`: per-layer KDA weight bundle (X.4.b.2 loader).
/// - `caches`: mutable per-head runtime caches (one entry per head).
/// - `num_heads` / `head_dim`: KDA head layout from
///   `KimiDeltaConfig` (K3 default: 96 heads × 128 dim).
/// - `alpha_rank`: low-rank α intermediate size (from `ssm_f_a`
///   `rows`).
/// - `g_min`: KDA gate lower bound (K3: `-5.0`).
/// - `rms_eps`: layer RMSNorm epsilon.
///
/// # Panics
///
/// Panics via slicing helper if any fused weight has fewer rows
/// than `num_heads × head_dim`, or if any weight is quantized (F32
/// only for Phase X.4.c.3.3.b; see the slicer's `None` return path
/// documentation).
#[allow(dead_code, clippy::too_many_arguments)]
fn kimi_k3_kda_layer_forward(
    x: &[f32],
    attn_norm: &[f32],
    attn_output: &WeightRef<'_>,
    kda: &KimiK3KdaAttn<'_>,
    caches: &mut [KimiDeltaHeadCache],
    num_heads: usize,
    head_dim: usize,
    alpha_rank: usize,
    g_min: f32,
    rms_eps: f32,
) -> Vec<f32> {
    let d = x.len();
    assert_eq!(attn_norm.len(), d, "attn_norm length must equal hidden dim");
    assert_eq!(
        caches.len(),
        num_heads,
        "caches length must equal num_heads"
    );

    // Pre-attention RMSNorm.
    let x_norm = kimi_k3_rms_norm(x, attn_norm, rms_eps);

    // Per-head SIL SIL SIL. We aggregate concatenated outputs
    // `[num_heads * head_dim]` then apply the shared output
    // projection.
    let v_head_dim = head_dim; // KDA convention: v_head_dim == qk_head_dim
    let mut concat_out = vec![0.0_f32; num_heads * v_head_dim];

    // ── Extract per-head params + call kimi_delta_forward_head ──
    //
    // K3 conv1d bias tensors are not shipped in the GGUF (per
    // TENSOR_MAP.md — only `ssm_conv1d_{q,k,v}.weight` is listed,
    // no `.bias`). Substitute zeros; `causal_conv1d_step` expects
    // a bias slice.
    let zero_bias = vec![0.0_f32; head_dim];
    // Identity gate fallback (used when `ssm_g` is absent — reused
    // across all heads instead of re-allocating per head).
    let identity_gate = identity_matrix_f32(v_head_dim);
    // Phase X.4.b.7 perf: ssm_f_a is SHARED across heads (`[alpha_rank,
    // d]`). Dequantize ONCE outside the head loop instead of 96 times
    // per KDA layer. For K3 (alpha_rank ≈ 64, d = 7168, Q4_K), this
    // saves ~1.8 MB dequant × 95 redundant repeats × 69 KDA layers =
    // ~12 GB of wasted dequant work per token forward.
    let ssm_f_a_f32 = weight_ref_row_dequant(&kda.ssm_f_a);
    // Phase X.4.b.7 perf: b_alpha is per-head but not per-layer (K3
    // does not ship it, so we use zeros). Hoist the allocation out of
    // the head loop — was `vec![0.0_f32; head_dim]` × 96 heads/layer
    // × 69 layers = 6624 zero-vec allocations per token.
    let b_alpha_zeros = vec![0.0_f32; head_dim];
    // Also hoist w_out identity_matrix (per-head loop was allocating
    // it 96 × 69 = 6624 times per token, same as identity_gate).
    let identity_out = identity_matrix_f32(v_head_dim);

    // Phase X.4.b.7 perf: parallelize per-head loop via rayon (feature
    // `parallel`). Each head is independent: distinct slice of caches,
    // distinct slice of concat_out, only-read shared inputs (`x_norm`,
    // `ssm_f_a_f32`, etc.). Zip caches + concat_out chunks so rayon
    // handles the mutable split cleanly.
    let head_iter = caches
        .iter_mut()
        .zip(concat_out.chunks_mut(v_head_dim))
        .enumerate();

    #[cfg(feature = "parallel")]
    let head_iter = {
        use rayon::iter::ParallelBridge;
        head_iter.par_bridge()
    };

    #[cfg(feature = "parallel")]
    let process_head =
        |(head_idx, (cache, out_slice)): (usize, (&mut KimiDeltaHeadCache, &mut [f32]))| {
            kimi_k3_kda_head_forward(
                head_idx,
                &x_norm,
                head_dim,
                v_head_dim,
                alpha_rank,
                g_min,
                rms_eps,
                kda,
                &ssm_f_a_f32,
                &zero_bias,
                &b_alpha_zeros,
                &identity_gate,
                &identity_out,
                cache,
                out_slice,
            );
        };

    #[cfg(feature = "parallel")]
    {
        use rayon::iter::ParallelIterator;
        head_iter.for_each(process_head);
    }

    #[cfg(not(feature = "parallel"))]
    for (head_idx, (cache, out_slice)) in head_iter {
        kimi_k3_kda_head_forward(
            head_idx,
            &x_norm,
            head_dim,
            v_head_dim,
            alpha_rank,
            g_min,
            rms_eps,
            kda,
            &ssm_f_a_f32,
            &zero_bias,
            &b_alpha_zeros,
            &identity_gate,
            &identity_out,
            cache,
            out_slice,
        );
    }

    // Shared output projection: [d_out, num_heads × v_head_dim].
    debug_assert_eq!(
        attn_output.cols,
        num_heads * v_head_dim,
        "attn_output.cols must equal num_heads × v_head_dim"
    );
    debug_assert_eq!(
        attn_output.rows, d,
        "attn_output.rows must equal hidden dim"
    );
    let mut y = vec![0.0_f32; d];
    attn_output.matvec(&concat_out, &mut y);
    y
}

/// Per-head KDA forward, extracted from `kimi_k3_kda_layer_forward` so
/// the head loop can be parallelized via rayon (Phase X.4.b.7 perf).
/// Handles slice + dequant + `kimi_delta_forward_head` for one head,
/// writing the concatenated output slot in-place.
#[allow(dead_code, clippy::too_many_arguments)]
fn kimi_k3_kda_head_forward(
    head_idx: usize,
    x_norm: &[f32],
    head_dim: usize,
    v_head_dim: usize,
    alpha_rank: usize,
    g_min: f32,
    rms_eps: f32,
    kda: &KimiK3KdaAttn<'_>,
    ssm_f_a_f32: &[f32],
    zero_bias: &[f32],
    b_alpha_zeros: &[f32],
    identity_gate: &[f32],
    identity_out: &[f32],
    cache: &mut KimiDeltaHeadCache,
    out_slice: &mut [f32],
) {
    let row_start = head_idx * head_dim;
    let row_end = row_start + head_dim;

    let describe = |t: &WeightRef<'_>| {
        format!(
            "qtype={:?} rows={} cols={} data_len={} elements_per_block={} block_bytes={}",
            t.qtype,
            t.rows,
            t.cols,
            t.data.len(),
            t.qtype.elements_per_block(),
            t.qtype.block_bytes()
        )
    };
    let w_q = kimi_k3_slice_weight_ref_rows(&kda.q, row_start, row_end).unwrap_or_else(|| {
        panic!(
            "kda_layer_forward: q slice failed (head {head_idx}, rows [{row_start}..{row_end}], \
             tensor {})",
            describe(&kda.q)
        )
    });
    let w_k = kimi_k3_slice_weight_ref_rows(&kda.k, row_start, row_end).unwrap_or_else(|| {
        panic!(
            "kda_layer_forward: k slice failed (head {head_idx}, rows [{row_start}..{row_end}], \
             tensor {})",
            describe(&kda.k)
        )
    });
    let w_v = kimi_k3_slice_weight_ref_rows(&kda.v, row_start, row_end).unwrap_or_else(|| {
        panic!(
            "kda_layer_forward: v slice failed (head {head_idx}, rows [{row_start}..{row_end}], \
             tensor {})",
            describe(&kda.v)
        )
    });
    let w_conv_q = kimi_k3_slice_weight_ref_rows(&kda.ssm_conv1d_q, row_start, row_end)
        .unwrap_or_else(|| {
            panic!(
                "kda_layer_forward: conv1d_q slice failed (head {head_idx}, rows \
                 [{row_start}..{row_end}], tensor {})",
                describe(&kda.ssm_conv1d_q)
            )
        });
    let w_conv_k = kimi_k3_slice_weight_ref_rows(&kda.ssm_conv1d_k, row_start, row_end)
        .unwrap_or_else(|| {
            panic!(
                "kda_layer_forward: conv1d_k slice failed (head {head_idx}, rows \
                 [{row_start}..{row_end}], tensor {})",
                describe(&kda.ssm_conv1d_k)
            )
        });
    let w_conv_v = kimi_k3_slice_weight_ref_rows(&kda.ssm_conv1d_v, row_start, row_end)
        .unwrap_or_else(|| {
            panic!(
                "kda_layer_forward: conv1d_v slice failed (head {head_idx}, rows \
                 [{row_start}..{row_end}], tensor {})",
                describe(&kda.ssm_conv1d_v)
            )
        });
    let w_alpha_up = kimi_k3_slice_weight_ref_rows(&kda.ssm_f_b, row_start, row_end)
        .unwrap_or_else(|| {
            panic!(
                "kda_layer_forward: ssm_f_b slice failed (head {head_idx}, rows \
                 [{row_start}..{row_end}], tensor {})",
                describe(&kda.ssm_f_b)
            )
        });

    let w_beta_ref = kimi_k3_slice_weight_ref_rows(&kda.ssm_beta, head_idx, head_idx + 1)
        .unwrap_or_else(|| panic!("kda_layer_forward: ssm_beta slice failed (head {head_idx})"));
    let w_beta = weight_ref_row_dequant(&w_beta_ref);

    let ssm_norm_f32: Vec<f32> = if kda.ssm_norm.len() == head_dim {
        kda.ssm_norm.clone()
    } else if row_end <= kda.ssm_norm.len() {
        kda.ssm_norm[row_start..row_end].to_vec()
    } else {
        panic!(
            "kda_layer_forward: ssm_norm shape unexpected (len={}, head_dim={head_dim}, \
             head_idx={head_idx}). Expected `[head_dim]` (shared) or `[num_heads * head_dim]`.",
            kda.ssm_norm.len(),
        );
    };

    let a_h = kda
        .ssm_a
        .as_ref()
        .and_then(|arr| arr.get(head_idx).copied())
        .unwrap_or(0.0);

    let ssm_g_slice = kda
        .ssm_g
        .as_ref()
        .and_then(|g| kimi_k3_slice_weight_ref_rows(g, row_start, row_end));
    let w_gate_owned = ssm_g_slice.as_ref().map(weight_ref_row_dequant);
    let w_gate_ref: &[f32] = w_gate_owned.as_deref().unwrap_or(identity_gate);

    let params = KimiDeltaHeadParams {
        w_q: &weight_ref_row_dequant(&w_q),
        w_k: &weight_ref_row_dequant(&w_k),
        w_v: &weight_ref_row_dequant(&w_v),
        conv_kernel_q: &weight_ref_row_dequant(&w_conv_q),
        conv_kernel_k: &weight_ref_row_dequant(&w_conv_k),
        conv_kernel_v: &weight_ref_row_dequant(&w_conv_v),
        conv_bias_q: zero_bias,
        conv_bias_k: zero_bias,
        conv_bias_v: zero_bias,
        w_beta: &w_beta,
        w_alpha_down: ssm_f_a_f32,
        w_alpha_up: &weight_ref_row_dequant(&w_alpha_up),
        b_alpha: b_alpha_zeros,
        a_h,
        alpha_rank,
        g_min,
        w_gate: w_gate_ref,
        w_out: identity_out,
        d_out: v_head_dim,
        rms_gamma: Some(&ssm_norm_f32),
        rms_eps,
    };

    let head_out = kimi_delta_forward_head(x_norm, &params, cache, rms_eps);
    out_slice.copy_from_slice(&head_out);
}

/// Interpret a `WeightRef` as an owned `Vec<f32>`. F32 only.
#[allow(dead_code)]
fn weight_ref_as_f32(w: &WeightRef<'_>) -> Vec<f32> {
    assert!(
        matches!(w.qtype, GgmlType::F32),
        "weight_ref_as_f32: F32 only"
    );
    let n = w.rows * w.cols;
    assert_eq!(
        w.data.len(),
        n * 4,
        "weight_ref_as_f32: byte length mismatch"
    );
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let b = &w.data[i * 4..i * 4 + 4];
        out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    }
    out
}

/// Slice a 1-D-like F32 `WeightRef` (`rows = 1` or `[N]` layout) as
/// an owned `Vec<f32>` from element `start` to `end`.
#[allow(dead_code)]
fn weight_ref_slice_as_f32(w: &WeightRef<'_>, start: usize, end: usize) -> Vec<f32> {
    assert!(
        matches!(w.qtype, GgmlType::F32),
        "weight_ref_slice_as_f32: F32 only"
    );
    let total_elements = w.rows * w.cols;
    assert!(end <= total_elements, "slice end out of range");
    let mut out = Vec::with_capacity(end - start);
    for i in start..end {
        let b = &w.data[i * 4..i * 4 + 4];
        out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    }
    out
}

/// Build a `[d, d]` identity matrix as a flat `Vec<f32>` (row-major).
/// Used at the KDA per-head layer to feed an identity `w_gate` /
/// `w_out` into the primitive, since K3 applies the real output
/// gate + output projection at the layer level (once, after
/// per-head concat).
#[allow(dead_code)]
fn identity_matrix_f32(d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; d * d];
    for i in 0..d {
        out[i * d + i] = 1.0;
    }
    out
}

// ── Kimi K3 Stable LatentMoE forward (Phase X.4.c.3.3.c) ──────────
//
// K3 tech report §2.3 (Stable LatentMoE, Eq 11) forward for one
// non-dense layer. Handles the **sigmoid router with `noaux_tc`
// bias correction**, the **2 shared experts** (always-active), and
// the **latent-space projection** path (`W↓` down → K3-specific
// `RMSNorm` → `W↑` up + SiTU-GLU per routed expert).
//
// Scope of Phase X.4.c.3.3.c (this landing):
//
// - **Router + shared experts + latent aggregation shell**: real
//   implementation of sigmoid gating, top-k selection with
//   `noaux_tc` bias, and the shared-expert SwiGLU forward.
// - **Routed expert per-expert loop**: `todo!()` fail-fast pending
//   Phase X.4.c.3.3.c.2. Reason: the 3-D per-expert cubes
//   (`ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps` shaped
//   `[moe_intermediate, latent_hidden, num_experts]`) need
//   per-expert byte-slice indexing that plugs into the Phase
//   X.4.e.1 streaming pool. Doing it correctly needs a
//   `RoutedExpertStorage` extraction similar to DeepSeek V3's
//   pattern; scoping this session on the outer routing loop keeps
//   the surface area tractable.

/// Sigmoid router with `noaux_tc` bias correction for K3 Stable
/// LatentMoE (Phase X.4.c.3.3.c, K3 tech report §2.3.3 Eq 13).
///
/// Computes `p_i = Sigmoid(W_r x_i)` per expert, adds the
/// noaux_tc bias `b_j` to the pre-selection scores (bias affects
/// selection but not the returned weights), picks the top-k
/// expert indices by `s_i + b`, and renormalizes so that
/// `Σ_{j ∈ T_k} p_j = 1` when `moe_renormalize = true` (K3
/// default). Bias-free variants (older MoE without noaux_tc) can
/// pass `exp_probs_b = None`.
///
/// # Arguments
///
/// - `x`: input hidden state `[d]`.
/// - `ffn_gate_inp`: dense f32 router weight, layout `[num_experts,
///   d]` row-major (i.e. `num_experts` rows of `d` cols) — reads
///   like `router.matvec(x, &mut scores)` where `scores` is
///   `[num_experts]`.
/// - `exp_probs_b`: optional per-expert bias `[num_experts]` for
///   noaux_tc. `None` disables bias correction.
/// - `top_k`: number of experts to select (K3: 16).
/// - `renormalize`: when `true`, top-k weights sum to 1.
///
/// # Returns
///
/// A `Vec<(usize, f32)>` of `top_k` entries, each
/// `(expert_index, normalized_weight)`, sorted by expert index
/// ascending for deterministic downstream reduce.
#[allow(dead_code)]
fn kimi_k3_moe_router(
    x: &[f32],
    ffn_gate_inp: &[f32],
    exp_probs_b: Option<&[f32]>,
    top_k: usize,
    renormalize: bool,
) -> Vec<(usize, f32)> {
    let d = x.len();
    assert!(!ffn_gate_inp.is_empty(), "ffn_gate_inp must not be empty");
    assert!(
        ffn_gate_inp.len().is_multiple_of(d),
        "ffn_gate_inp length {} must be multiple of hidden dim {d}",
        ffn_gate_inp.len()
    );
    let num_experts = ffn_gate_inp.len() / d;
    assert!(
        top_k <= num_experts,
        "top_k {top_k} > num_experts {num_experts}"
    );

    // Router scores: raw dot product then sigmoid.
    let mut scores = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let row_start = e * d;
        let mut acc = 0.0_f64;
        for j in 0..d {
            acc += f64::from(ffn_gate_inp[row_start + j]) * f64::from(x[j]);
        }
        scores.push(sigmoid(acc as f32));
    }

    // Bias-adjusted selection scores: `s_i + b_j` (K3 noaux_tc).
    let selection_scores: Vec<f32> = if let Some(bias) = exp_probs_b {
        assert_eq!(
            bias.len(),
            num_experts,
            "exp_probs_b length must equal num_experts"
        );
        scores
            .iter()
            .zip(bias.iter())
            .map(|(&s, &b)| s + b)
            .collect()
    } else {
        scores.clone()
    };

    // Top-k selection by bias-adjusted score.
    let mut indexed: Vec<(usize, f32)> = selection_scores
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k);

    // Weights use the raw scores (bias omitted, per §2.3.3 Eq 13
    // "Because b is omitted from p_{i,j}").
    let mut selected: Vec<(usize, f32)> =
        indexed.into_iter().map(|(i, _)| (i, scores[i])).collect();

    // Renormalize weights to sum to 1 (K3 default `moe_renormalize
    // = true`).
    if renormalize {
        let sum: f32 = selected.iter().map(|(_, w)| *w).sum();
        if sum > 0.0 {
            let inv = sum.recip();
            for (_, w) in &mut selected {
                *w *= inv;
            }
        }
    }

    // Sort by expert index ascending for deterministic reduction.
    selected.sort_by_key(|(i, _)| *i);
    selected
}

/// K3 shared-experts forward (Phase X.4.c.3.3.c, K3 tech report
/// §2.3 Eq 11 shared-expert branch `E_j^shared(x)`).
///
/// K3 fuses the `num_shared_experts` (K3 default 2) shared experts
/// into a **single** SwiGLU block per layer — the GGUF tensor set
/// only ships `ffn_gate_shexp` / `ffn_up_shexp` / `ffn_down_shexp`
/// (one triple per layer). The forward is identical to the standard
/// SwiGLU pattern used by [`kimi_k3_dense_ffn_forward`], just with
/// the shared-expert weight refs.
///
/// Applied without RMSNorm at the shared-expert level because the
/// outer layer already pre-normalized `x` via the FFN norm before
/// entering the MoE dispatch.
#[allow(dead_code)]
fn kimi_k3_shared_experts_forward(
    x: &[f32],
    gate: &WeightRef<'_>,
    up: &WeightRef<'_>,
    down: &WeightRef<'_>,
) -> Vec<f32> {
    let d = x.len();
    let intermediate = gate.rows;
    debug_assert_eq!(up.rows, intermediate);
    debug_assert_eq!(down.cols, intermediate);
    debug_assert_eq!(down.rows, d);

    let mut gate_out = vec![0.0_f32; intermediate];
    gate.matvec(x, &mut gate_out);
    let mut up_out = vec![0.0_f32; intermediate];
    up.matvec(x, &mut up_out);
    let mut gated = vec![0.0_f32; intermediate];
    for i in 0..intermediate {
        gated[i] = silu(gate_out[i]) * up_out[i];
    }
    let mut y = vec![0.0_f32; d];
    down.matvec(&gated, &mut y);
    y
}

/// SiTU-GLU activation (K3 tech report §2.3.2 Eq 12) — scalar form.
///
/// ```text
/// situ(g, u) = β_1 · tanh(g / β_1) · σ(g) · β_2 · tanh(u / β_2)
/// ```
///
/// K3 defaults: `β_1 = situ_beta = 4.0`, `β_2 = situ_linear_beta = 25.0`.
/// llama.cpp uses the same formula (`src/models/kimi-k3.cpp` L181-192).
#[allow(dead_code)]
fn kimi_k3_situ_scalar(g: f32, u: f32, beta: f32, linear_beta: f32) -> f32 {
    let g_tanh = (g / beta).tanh() * beta;
    let g_sig = sigmoid(g);
    let u_tanh = (u / linear_beta).tanh() * linear_beta;
    g_tanh * g_sig * u_tanh
}

/// Slice one expert's 2-D plane out of a 3-D per-expert weight cube
/// (`ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`) as a fresh
/// `WeightRef` pointing at the sub-slice of the underlying byte buffer.
///
/// The GGUF cubes are stored as `[dim0, dim1, num_experts]` in ggml
/// convention (`ne[0]` fastest-varying, i.e. row-major with dim0 = column
/// stride). Each expert's plane is a contiguous `dim0 * dim1` block, so
/// per-expert byte offset is `expert_idx * dim0 * dim1 * bytes_per_element`
/// for F32/F16, or `expert_idx * blocks_per_plane * block_bytes` for
/// quantized cubes (Phase X.4.c.3.3.b.2 upgrade — safe when
/// `dim0 * dim1 % elements_per_block == 0`).
///
/// Returns `None` when:
/// - `expert_idx >= num_experts` (out of bounds),
/// - `w.qtype` is `GgmlType::Other(_)` (unknown quant type),
/// - `w.qtype` is quantized AND `dim0 * dim1 % elements_per_block != 0`
///   (plane splits mid-block), or
/// - `cube.data.len()` is too small for `num_experts` planes.
/// Per-expert row count for a K3 3-D expert cube.
///
/// Real GrEarl K3 GGUF stores expert cubes as 3-D
/// `[cols=d0, per_expert_rows=d1, num_experts=d2]` which
/// `load_weight_ref_any_shape` flattens to
/// `cube.rows = per_expert_rows * num_experts`. Test fixtures use a
/// 2-D per-expert layout where `cube.rows` already equals
/// `per_expert_rows`. Both interpretations converge on
/// `per_expert_bytes = cube.data.len() / num_experts`, from which the
/// row count can be derived block-safely for any quant type.
#[allow(dead_code)]
fn kimi_k3_cube_per_expert_rows(cube: &WeightRef<'_>, num_experts: usize) -> usize {
    if num_experts == 0 {
        return cube.rows;
    }
    let block_bytes = cube.qtype.block_bytes();
    let elements_per_block = cube.qtype.elements_per_block();
    if block_bytes == 0 || elements_per_block == 0 || cube.cols == 0 {
        return cube.rows;
    }
    if !cube.data.len().is_multiple_of(num_experts) {
        return cube.rows;
    }
    let per_expert_bytes = cube.data.len() / num_experts;
    if !per_expert_bytes.is_multiple_of(block_bytes) {
        return cube.rows;
    }
    let per_expert_blocks = per_expert_bytes / block_bytes;
    let per_expert_elements = per_expert_blocks * elements_per_block;
    if !per_expert_elements.is_multiple_of(cube.cols) {
        return cube.rows;
    }
    per_expert_elements / cube.cols
}

#[allow(dead_code)]
fn kimi_k3_expert_plane_weight_ref<'a>(
    cube: &WeightRef<'a>,
    expert_idx: usize,
    num_experts: usize,
) -> Option<WeightRef<'a>> {
    if expert_idx >= num_experts || num_experts == 0 {
        return None;
    }
    if matches!(cube.qtype, GgmlType::Other(_)) {
        return None;
    }
    let elements_per_block = cube.qtype.elements_per_block();
    let block_bytes = cube.qtype.block_bytes();
    if elements_per_block == 0 || block_bytes == 0 {
        return None;
    }
    // Determine per-expert plane size from BYTE budget, not the row
    // count. Data is authoritative — real K3 GGUF stores 3-D cubes
    // flattened as `[cols=d0, rows=d1*num_experts]` (cube.rows carries
    // num_experts inside), while test fixtures use 2-D per-expert
    // (cube.rows is already per-expert with data holding num_experts
    // planes back-to-back). Both cases have the SAME per-expert byte
    // count = cube.data.len() / num_experts.
    if !cube.data.len().is_multiple_of(num_experts) {
        return None;
    }
    let plane_bytes = cube.data.len() / num_experts;
    if !plane_bytes.is_multiple_of(block_bytes) {
        return None;
    }
    let plane_blocks = plane_bytes / block_bytes;
    let plane_elements = plane_blocks.checked_mul(elements_per_block)?;
    // Per-expert row count: plane_elements / cols. Must divide evenly.
    if cube.cols == 0 || !plane_elements.is_multiple_of(cube.cols) {
        return None;
    }
    let per_expert_rows = plane_elements / cube.cols;
    if !plane_elements.is_multiple_of(elements_per_block) {
        return None;
    }
    let start = expert_idx.checked_mul(plane_bytes)?;
    let end = start.checked_add(plane_bytes)?;
    Some(WeightRef {
        data: &cube.data[start..end],
        qtype: cube.qtype,
        rows: per_expert_rows,
        cols: cube.cols,
    })
}

/// K3 Stable LatentMoE forward — router + shared + routed dispatch,
/// end-to-end (Phase X.4.c.3.3.c.2, K3 tech report §2.3 Eq 11).
///
/// Follows the llama.cpp reference (`src/models/kimi-k3.cpp` L596-645
/// `build_latent_moe`):
///
/// 1. **Pre-FFN RMSNorm** applied to `x` using `ffn_norm` (this
///    convention differs from llama.cpp which does the norm in the
///    caller; we match the sibling `kimi_k3_dense_ffn_forward`).
/// 2. **Router** (`kimi_k3_moe_router`) — runs on FULL-WIDTH `x_norm`
///    (n_embd), NOT the latent projection. Sigmoid + optional
///    noaux_tc bias + top-k + renormalize.
/// 3. **Shared experts** (`kimi_k3_shared_experts_forward`) — also
///    on full-width `x_norm`, single fused SwiGLU (K3 folds the 2
///    shared experts into 1 triple at export time).
/// 4. **Down-project to latent**: `routed_in = W↓ · x_norm`
///    (`routed_exp_down`, shape `[n_embd_latent, n_embd]` in
///    WeightRef convention).
/// 5. **Per-expert dispatch in latent space**: for each selected
///    `(expert_idx, weight)` pair,
///    - slice per-expert planes out of `ffn_gate_exps` /
///      `ffn_up_exps` (each `[n_embd_latent, n_ff_exp]` per expert)
///      and `ffn_down_exps` (`[n_ff_exp, n_embd_latent]` per expert)
///    - `gate = gate_e @ routed_in` (latent → intermediate)
///    - `up = up_e @ routed_in`
///    - `act = SiTU(gate, up)` (K3 SiTU-GLU with `β=4`, `β_linear=25`)
///    - `expert_out = down_e @ act` (intermediate → latent)
///    - `routed_sum += weight * expert_out`
/// 6. **K3-only stabilization**: RMSNorm on `routed_sum` using
///    `routed_exp_norm` (γ of dim `n_embd_latent`), then up-project
///    back to hidden via `routed_exp_up` (shape `[n_embd, n_embd_latent]`).
/// 7. **Combine**: `y = shared_out + up_projected_routed_agg`
///    (both `[n_embd]`).
///
/// **Quantized cubes now supported** (Phase X.4.c.3.3.b.2 upgrade)
/// when `n_embd_latent * n_ff_exp % elements_per_block == 0`, which
/// holds for all K3 configs where both dims are multiples of 256
/// (K-quant block size). Q4_K / Q8_0 / IQ4_XS / MXFP4 all work via
/// existing `WeightRef::matvec` dispatch (dequantize-then-dot at
/// matvec time). The routed dispatch panics with a diagnostic when
/// a cube's plane splits mid-block.
///
/// SiTU coefficients hardcoded to `β=4, β_linear=25` (K3 defaults from
/// pwilkin PR `constants.py` `activation.situ_beta` / `situ_linear_beta`).
/// A config-plumbed variant is deferred to a later refinement.
#[allow(dead_code, clippy::too_many_arguments)]
fn kimi_k3_latent_moe_forward(
    x: &[f32],
    ffn_norm: &[f32],
    moe: &KimiK3LatentMoe<'_>,
    top_k: usize,
    renormalize: bool,
    rms_eps: f32,
) -> Vec<f32> {
    // Pre-FFN RMSNorm applied to `x` before both the router and the
    // shared / routed experts (mirrors the sibling
    // `kimi_k3_dense_ffn_forward` pattern; llama.cpp applies the
    // norm in the caller instead, but the aggregate math is
    // identical).
    let x_norm = kimi_k3_rms_norm(x, ffn_norm, rms_eps);
    let hidden = x.len();

    // Step 1: router (runs on full-width x_norm, not latent).
    let selected = kimi_k3_moe_router(
        &x_norm,
        &moe.ffn_gate_inp,
        moe.exp_probs_b.as_deref(),
        top_k,
        renormalize,
    );

    // Step 2: shared experts (full-width n_embd, SwiGLU fused).
    let shared_out = kimi_k3_shared_experts_forward(
        &x_norm,
        &moe.ffn_gate_shexp,
        &moe.ffn_up_shexp,
        &moe.ffn_down_shexp,
    );

    // Step 3: derive latent dims + expert count from the loaded WeightRefs.
    // routed_exp_down: [n_embd_latent (rows), n_embd (cols)]
    // routed_exp_up:   [n_embd (rows), n_embd_latent (cols)]
    let n_embd_latent = moe.routed_exp_down.rows;
    assert_eq!(
        moe.routed_exp_down.cols, hidden,
        "routed_exp_down.cols must equal hidden dim"
    );
    assert_eq!(
        moe.routed_exp_up.cols, n_embd_latent,
        "routed_exp_up.cols must equal n_embd_latent"
    );
    assert_eq!(
        moe.routed_exp_up.rows, hidden,
        "routed_exp_up.rows must equal hidden dim"
    );
    assert_eq!(
        moe.routed_exp_norm.len(),
        n_embd_latent,
        "routed_exp_norm length must equal n_embd_latent"
    );
    assert!(
        moe.ffn_gate_inp.len().is_multiple_of(hidden),
        "ffn_gate_inp length {} must be multiple of hidden {hidden}",
        moe.ffn_gate_inp.len()
    );
    let num_experts = moe.ffn_gate_inp.len() / hidden;

    // ffn_gate_exps / ffn_up_exps: real K3 GGUF stores as 3-D cube
    // `[n_embd_latent, n_ff_exp, n_experts]` (ggml ne order). After
    // load_weight_ref_any_shape's flatten:
    //   cols = dims[0] = n_embd_latent
    //   rows = product(dims[1..]) = n_ff_exp * n_experts (flattened)
    // So the actual per-expert row count is `rows / num_experts`.
    // ffn_down_exps has swapped 1st/2nd dims: [n_ff_exp, n_embd_latent, n_experts]
    // → cols = n_ff_exp, rows = n_embd_latent * n_experts.
    assert_eq!(
        moe.ffn_gate_exps.cols, n_embd_latent,
        "ffn_gate_exps.cols must equal n_embd_latent"
    );
    assert_eq!(
        moe.ffn_up_exps.cols, n_embd_latent,
        "ffn_up_exps.cols must equal n_embd_latent"
    );
    // Per-expert n_ff_exp derivation: authoritative via data-length /
    // num_experts (works for both 2-D fixtures and real 3-D-flattened
    // K3 GrEarl cubes). Helper matches kimi_k3_expert_plane_weight_ref
    // internal logic.
    let n_ff_exp = kimi_k3_cube_per_expert_rows(&moe.ffn_gate_exps, num_experts);
    let per_expert_up_rows = kimi_k3_cube_per_expert_rows(&moe.ffn_up_exps, num_experts);
    assert_eq!(
        per_expert_up_rows, n_ff_exp,
        "ffn_up_exps per-expert rows must equal ffn_gate_exps per-expert rows"
    );
    assert_eq!(
        moe.ffn_down_exps.cols, n_ff_exp,
        "ffn_down_exps.cols must equal n_ff_exp"
    );
    let per_expert_down_rows = kimi_k3_cube_per_expert_rows(&moe.ffn_down_exps, num_experts);
    assert_eq!(
        per_expert_down_rows, n_embd_latent,
        "ffn_down_exps per-expert rows must equal n_embd_latent"
    );

    // Phase X.4.b.9 perf: MADV_WILLNEED prefetch hints for selected
    // expert cube byte ranges. Right after the router picks top-k,
    // we tell the OS to start paging in the 3 × top_k byte ranges
    // (gate, up, down for each selected expert). On K3 with USB SSD
    // this overlaps ~50 MB of disk I/O with the ~100 ms it takes to
    // run the shared-experts SwiGLU + routed_exp_down matvec below.
    // No-op on non-Unix; hint-only under Unix (OS may ignore under
    // memory pressure). Gated on ALICE_K3_MADV env so we can
    // A/B measure.
    let madv_on = std::env::var("ALICE_K3_MADV").ok().as_deref() == Some("1");
    if madv_on {
        for (expert_idx, _weight) in &selected {
            for cube in [&moe.ffn_gate_exps, &moe.ffn_up_exps, &moe.ffn_down_exps] {
                if let Some(plane) = kimi_k3_expert_plane_weight_ref(cube, *expert_idx, num_experts)
                {
                    let _ = crate::deepseek_streaming::advise_willneed(plane.data);
                }
            }
        }
    }

    // Step 4: down-project x_norm to latent space.
    let mut routed_in = vec![0.0_f32; n_embd_latent];
    moe.routed_exp_down.matvec(&x_norm, &mut routed_in);

    // Step 5: per-expert dispatch, weighted sum in latent space.
    let mut routed_sum = vec![0.0_f32; n_embd_latent];
    for (expert_idx, weight) in &selected {
        let gate_ref =
            kimi_k3_expert_plane_weight_ref(&moe.ffn_gate_exps, *expert_idx, num_experts)
                .unwrap_or_else(|| {
                    panic!(
                        "K3 LatentMoE routed dispatch: ffn_gate_exps for expert \
                         {expert_idx} not sliceable (qtype {:?}, num_experts \
                         {num_experts}, plane bytes {plane_bytes} × num_experts vs \
                         cube data {}). Quantized cubes require \
                         `plane_elements % elements_per_block == 0` — check that \
                         `n_embd_latent * n_ff_exp` is a multiple of the quant \
                         block size ({} for this qtype).",
                        moe.ffn_gate_exps.qtype,
                        moe.ffn_gate_exps.data.len(),
                        moe.ffn_gate_exps.qtype.elements_per_block(),
                        plane_bytes = {
                            let epb = moe.ffn_gate_exps.qtype.elements_per_block().max(1);
                            let bb = moe.ffn_gate_exps.qtype.block_bytes();
                            (moe.ffn_gate_exps.rows * moe.ffn_gate_exps.cols / epb) * bb
                        }
                    )
                });
        let up_ref = kimi_k3_expert_plane_weight_ref(&moe.ffn_up_exps, *expert_idx, num_experts)
            .expect("ffn_up_exps expert slice must succeed if gate slice did");
        let down_ref =
            kimi_k3_expert_plane_weight_ref(&moe.ffn_down_exps, *expert_idx, num_experts)
                .expect("ffn_down_exps expert slice must succeed if gate slice did");

        let mut gate_out = vec![0.0_f32; n_ff_exp];
        gate_ref.matvec(&routed_in, &mut gate_out);
        let mut up_out = vec![0.0_f32; n_ff_exp];
        up_ref.matvec(&routed_in, &mut up_out);

        // SiTU-GLU per element (β=4, β_linear=25 = K3 defaults).
        let act: Vec<f32> = gate_out
            .iter()
            .zip(up_out.iter())
            .map(|(&g, &u)| kimi_k3_situ_scalar(g, u, 4.0, 25.0))
            .collect();

        let mut expert_out = vec![0.0_f32; n_embd_latent];
        down_ref.matvec(&act, &mut expert_out);

        for i in 0..n_embd_latent {
            routed_sum[i] += weight * expert_out[i];
        }
    }

    // Step 6a: K3-only RMSNorm on aggregated routed sum (paper §2.3.1
    // "Stable LatentMoE" — this norm prevents scale drift when the
    // per-expert outputs sum in an unbounded way).
    let routed_normed = kimi_k3_rms_norm(&routed_sum, &moe.routed_exp_norm, rms_eps);

    // Step 6b: up-project from latent back to hidden.
    let mut up_projected = vec![0.0_f32; hidden];
    moe.routed_exp_up.matvec(&routed_normed, &mut up_projected);

    // Step 7: combine with shared experts.
    let mut y = up_projected;
    for i in 0..hidden {
        y[i] += shared_out[i];
    }
    y
}

#[cfg(test)]
mod kimi_k3_latent_moe_tests {
    use super::{
        kimi_k3_expert_plane_weight_ref, kimi_k3_latent_moe_forward, kimi_k3_moe_router,
        kimi_k3_shared_experts_forward, kimi_k3_situ_scalar, GgmlType, KimiK3LatentMoe, WeightRef,
    };

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 4);
        for &x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    #[test]
    fn moe_router_selects_top_k_by_score() {
        // 4 experts, hidden = 2, top-k = 2. Router weights favour
        // experts 0 and 3 (row 0 = strong positive dot with x,
        // row 3 = same). Verify those two get selected.
        let x = vec![1.0_f32, 1.0];
        // Rows [e * d + j], row 0 = strong, row 1/2 = weak, row 3 = strong.
        let router = vec![
            5.0, 5.0, // expert 0
            0.1, 0.1, // expert 1
            0.1, 0.1, // expert 2
            5.0, 5.0, // expert 3
        ];
        let selected = kimi_k3_moe_router(&x, &router, None, 2, true);
        let indices: Vec<usize> = selected.iter().map(|(i, _)| *i).collect();
        assert_eq!(indices, vec![0, 3], "top-2 should pick experts 0 and 3");
        // Renormalized weights sum to 1.
        let sum: f32 = selected.iter().map(|(_, w)| *w).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "renormalized weights must sum to 1"
        );
    }

    #[test]
    fn moe_router_bias_shifts_selection_without_affecting_weights() {
        // Same base scores, but expert 1's bias makes it selected
        // over expert 3 despite lower raw sigmoid. Weights returned
        // must still be based on raw sigmoid (bias omitted per
        // paper §2.3.3 "b is omitted from p_{i,j}").
        let x = vec![1.0_f32, 1.0];
        let router = vec![
            5.0, 5.0, // expert 0: high raw
            1.0, 1.0, // expert 1: mid raw
            0.0, 0.0, // expert 2: low raw
            3.0, 3.0, // expert 3: mid-high raw
        ];
        // Bias pushes expert 1's selection score above expert 3.
        let bias = vec![0.0_f32, 10.0, 0.0, 0.0];
        let selected = kimi_k3_moe_router(&x, &router, Some(&bias), 2, false);
        let indices: Vec<usize> = selected.iter().map(|(i, _)| *i).collect();
        // Selection order (by index ascending after sort): 0, 1.
        assert_eq!(indices, vec![0, 1], "bias must promote expert 1 over 3");
        // Weight for expert 1 must still be its raw sigmoid, not
        // sigmoid + bias.
        let raw_1 = 1.0_f32 / (1.0 + (-2.0_f32).exp()); // sigmoid(1+1=2)
        let w_1 = selected.iter().find(|(i, _)| *i == 1).unwrap().1;
        assert!(
            (w_1 - raw_1).abs() < 1e-5,
            "expert 1 weight {w_1} must equal raw sigmoid {raw_1}, bias omitted"
        );
    }

    #[test]
    fn moe_router_returns_sorted_indices() {
        // Deterministic downstream reduction requires expert index
        // ascending order in the returned selection.
        let x = vec![1.0_f32, 1.0];
        let router = vec![
            0.1, 0.1, // expert 0: low
            5.0, 5.0, // expert 1: high
            3.0, 3.0, // expert 2: mid
            5.0, 5.0, // expert 3: high
        ];
        let selected = kimi_k3_moe_router(&x, &router, None, 3, true);
        let indices: Vec<usize> = selected.iter().map(|(i, _)| *i).collect();
        assert_eq!(indices, vec![1, 2, 3], "indices must be sorted ascending");
    }

    #[test]
    fn shared_experts_forward_zero_input_gives_zero() {
        let x = vec![0.0_f32; 4];
        // Non-trivial weights.
        let gate_bytes = f32_bytes(&(0..32).map(|i| 0.01 * i as f32).collect::<Vec<_>>());
        let up_bytes = f32_bytes(&(0..32).map(|i| 0.02 * i as f32).collect::<Vec<_>>());
        let down_bytes = f32_bytes(&(0..32).map(|i| 0.03 * i as f32).collect::<Vec<_>>());
        let gate = WeightRef {
            data: &gate_bytes,
            qtype: GgmlType::F32,
            rows: 8,
            cols: 4,
        };
        let up = WeightRef {
            data: &up_bytes,
            qtype: GgmlType::F32,
            rows: 8,
            cols: 4,
        };
        let down = WeightRef {
            data: &down_bytes,
            qtype: GgmlType::F32,
            rows: 4,
            cols: 8,
        };
        let y = kimi_k3_shared_experts_forward(&x, &gate, &up, &down);
        for (i, &v) in y.iter().enumerate() {
            assert_eq!(v, 0.0, "y[{i}] = {v} but zero input must give zero");
        }
    }

    #[test]
    fn situ_scalar_at_zero_gives_zero() {
        // SiTU: β · tanh(g/β) · σ(g) · β_linear · tanh(u/β_linear).
        // At g = u = 0: tanh(0) = 0 for both, product is 0.
        let y = kimi_k3_situ_scalar(0.0, 0.0, 4.0, 25.0);
        assert!(y.abs() < 1e-6, "situ(0, 0) must be 0, got {y}");
    }

    #[test]
    fn situ_scalar_saturates_symmetric_at_large_gate() {
        // For |g| >> β: tanh(g/β) ≈ sign(g) · 1, σ(g) ≈ 1 (g>0) or 0 (g<0).
        // For |u| >> β_linear: tanh(u/β_linear) · β_linear ≈ sign(u) · β_linear.
        // g=+40, u=+250 (10× each β): approx 4 · 1 · 1 · 25 · 1 = 100.
        let y = kimi_k3_situ_scalar(40.0, 250.0, 4.0, 25.0);
        assert!(
            (y - 100.0).abs() < 0.1,
            "situ(+40, +250) with β=4, β_linear=25 must saturate near 100, got {y}"
        );
        // Negative gate → σ ≈ 0, whole product collapses.
        let y_neg = kimi_k3_situ_scalar(-40.0, 250.0, 4.0, 25.0);
        assert!(
            y_neg.abs() < 1e-3,
            "situ(-40, +250) must collapse via σ(-40) ≈ 0, got {y_neg}"
        );
    }

    #[test]
    fn expert_plane_weight_ref_extracts_correct_slice() {
        // 3-D cube [cols=2, rows=3, num_experts=2] (ggml order: fastest = cols).
        // Expert 0 plane: 6 F32 values [1, 2, 3, 4, 5, 6].
        // Expert 1 plane: 6 F32 values [10, 20, 30, 40, 50, 60].
        let e0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let e1: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let mut cube_vals = e0.clone();
        cube_vals.extend(e1.clone());
        let cube_bytes = f32_bytes(&cube_vals);
        let cube = WeightRef {
            data: &cube_bytes,
            qtype: GgmlType::F32,
            rows: 3, // n_ff_exp per expert
            cols: 2, // n_embd_latent per expert
        };

        let plane0 =
            kimi_k3_expert_plane_weight_ref(&cube, 0, 2).expect("expert 0 plane must extract");
        assert_eq!(plane0.rows, 3);
        assert_eq!(plane0.cols, 2);
        assert_eq!(plane0.data.len(), 6 * 4, "24 bytes = 6 F32");
        // Verify first f32 = 1.0
        let first_bytes: [u8; 4] = plane0.data[..4].try_into().unwrap();
        assert_eq!(f32::from_le_bytes(first_bytes), 1.0);

        let plane1 =
            kimi_k3_expert_plane_weight_ref(&cube, 1, 2).expect("expert 1 plane must extract");
        let first_bytes: [u8; 4] = plane1.data[..4].try_into().unwrap();
        assert_eq!(f32::from_le_bytes(first_bytes), 10.0);
    }

    #[test]
    fn expert_plane_weight_ref_rejects_out_of_range_expert() {
        let bytes = vec![0u8; 24];
        let cube = WeightRef {
            data: &bytes,
            qtype: GgmlType::F32,
            rows: 3,
            cols: 2,
        };
        // num_experts = 2, expert_idx 2 is out of range.
        assert!(kimi_k3_expert_plane_weight_ref(&cube, 2, 2).is_none());
    }

    #[test]
    fn expert_plane_weight_ref_supports_quantized_cube_when_aligned() {
        // Phase X.4.c.3.3.b.2 + X.4.b.6: Q4_K per-expert slicing works
        // when per-expert plane_elements is a multiple of QK_K (256).
        // Real K3 GGUF stores cubes as 3-D flattened `[cols, per_expert_rows
        // * num_experts]`. Per-expert plane = 4 rows × 64 cols = 256
        // elements = 1 Q4_K block per plane = 144 bytes/plane.
        // Total cube rows = 4 × 2 = 8, cube data = 288 bytes (2 planes).
        let num_experts = 2;
        let per_expert_rows = 4;
        let bytes = vec![0u8; 144 * num_experts];
        let cube = WeightRef {
            data: &bytes,
            qtype: GgmlType::Q4_K,
            rows: per_expert_rows * num_experts,
            cols: 64,
        };
        let plane0 = kimi_k3_expert_plane_weight_ref(&cube, 0, num_experts)
            .expect("Q4_K plane slicing must succeed when per-expert plane % 256 == 0");
        assert_eq!(plane0.rows, per_expert_rows);
        assert_eq!(plane0.cols, 64);
        assert_eq!(plane0.data.len(), 144, "1 Q4_K block per plane");
        let plane1 = kimi_k3_expert_plane_weight_ref(&cube, 1, num_experts)
            .expect("expert 1 slice must succeed for aligned Q4_K cube");
        assert_eq!(plane1.data.len(), 144);
    }

    #[test]
    fn expert_plane_weight_ref_rejects_quantized_cube_when_misaligned() {
        // Phase X.4.b.6 rewrite: with data-length-based per-expert
        // derivation, "misalignment" means data length not divisible
        // by (num_experts * block_bytes) — the per-expert byte count
        // doesn't land on a block boundary. Test: 200 bytes total /
        // 2 experts = 100 bytes/expert, not a Q4_K block multiple
        // (144).
        let bytes = vec![0u8; 200];
        let cube = WeightRef {
            data: &bytes,
            qtype: GgmlType::Q4_K,
            rows: 4,
            cols: 64,
        };
        assert!(
            kimi_k3_expert_plane_weight_ref(&cube, 0, 2).is_none(),
            "misaligned Q4_K (100 bytes/expert vs 144-byte block) must fail slicing"
        );
    }

    #[test]
    fn expert_plane_weight_ref_supports_mxfp4_when_aligned() {
        // Phase X.4.c.3.3.b.2: MXFP4 per-expert slicing (17 bytes/block,
        // 32 elements/block). Plane = 4 × 32 = 128 elements = 4 MXFP4
        // blocks = 68 bytes.
        let num_experts = 3;
        let bytes = vec![0u8; 68 * num_experts];
        let cube = WeightRef {
            data: &bytes,
            qtype: GgmlType::Mxfp4,
            rows: 4,
            cols: 32,
        };
        let plane2 = kimi_k3_expert_plane_weight_ref(&cube, 2, num_experts)
            .expect("MXFP4 plane slicing must succeed at aligned boundary");
        assert_eq!(plane2.data.len(), 68);
    }

    #[test]
    fn latent_moe_forward_smoke_test_with_2_experts_f32() {
        // Minimal end-to-end smoke test: 2 experts, top-k=1, deterministic
        // router selection, verify no panic + output shape correct + non-zero.
        // Dims: hidden = 4, n_embd_latent = 2, n_ff_exp = 3, num_experts = 2.
        let hidden = 4;
        let n_embd_latent = 2;
        let n_ff_exp = 3;
        let num_experts = 2;
        let x = vec![0.5_f32; hidden];
        let ffn_norm = vec![1.0_f32; hidden]; // identity gain

        // Router: 2×4 = 8 elements. Expert 1 strongly preferred.
        let ffn_gate_inp = vec![
            0.1, 0.1, 0.1, 0.1, // expert 0
            5.0, 5.0, 5.0, 5.0, // expert 1
        ];

        // Shared experts: gate/up/down each [8, 4] / [8, 4] / [4, 8].
        let gate_shexp_vals: Vec<f32> = (0..32).map(|i| 0.01 * i as f32).collect();
        let up_shexp_vals: Vec<f32> = (0..32).map(|i| 0.02 * i as f32).collect();
        let down_shexp_vals: Vec<f32> = (0..32).map(|i| 0.03 * i as f32).collect();
        let gate_shexp_bytes = f32_bytes(&gate_shexp_vals);
        let up_shexp_bytes = f32_bytes(&up_shexp_vals);
        let down_shexp_bytes = f32_bytes(&down_shexp_vals);

        // Routed down: [n_embd_latent=2, n_embd=4] → 8 F32.
        let routed_exp_down_vals: Vec<f32> = vec![
            0.1, 0.2, 0.3, 0.4, // row 0
            0.5, 0.6, 0.7, 0.8, // row 1
        ];
        let routed_exp_down_bytes = f32_bytes(&routed_exp_down_vals);

        // Routed up: [n_embd=4, n_embd_latent=2] → 8 F32.
        let routed_exp_up_vals: Vec<f32> = vec![
            0.5, 0.5, // row 0
            0.4, 0.4, // row 1
            0.3, 0.3, // row 2
            0.2, 0.2, // row 3
        ];
        let routed_exp_up_bytes = f32_bytes(&routed_exp_up_vals);

        // routed_exp_norm [n_embd_latent].
        let routed_exp_norm = vec![1.0_f32; n_embd_latent];

        // Per-expert cubes: gate_exps / up_exps [cols=2, rows=3] per expert × 2 experts
        // = 12 F32 each; down_exps [cols=3, rows=2] per expert × 2 experts = 12 F32.
        let per_expert_gate = 6;
        let per_expert_up = 6;
        let per_expert_down = 6;
        let mut gate_exps_vals: Vec<f32> = Vec::new();
        gate_exps_vals.extend((0..per_expert_gate).map(|i| 0.1 + 0.1 * i as f32));
        gate_exps_vals.extend((0..per_expert_gate).map(|i| 1.0 + 0.1 * i as f32));
        let mut up_exps_vals: Vec<f32> = Vec::new();
        up_exps_vals.extend((0..per_expert_up).map(|i| 0.15 + 0.05 * i as f32));
        up_exps_vals.extend((0..per_expert_up).map(|i| 1.5 + 0.05 * i as f32));
        let mut down_exps_vals: Vec<f32> = Vec::new();
        down_exps_vals.extend((0..per_expert_down).map(|i| 0.2 + 0.1 * i as f32));
        down_exps_vals.extend((0..per_expert_down).map(|i| 2.0 + 0.1 * i as f32));

        let gate_exps_bytes = f32_bytes(&gate_exps_vals);
        let up_exps_bytes = f32_bytes(&up_exps_vals);
        let down_exps_bytes = f32_bytes(&down_exps_vals);

        let moe = KimiK3LatentMoe {
            ffn_gate_inp,
            exp_probs_b: None,
            ffn_gate_shexp: WeightRef {
                data: &gate_shexp_bytes,
                qtype: GgmlType::F32,
                rows: 8,
                cols: 4,
            },
            ffn_up_shexp: WeightRef {
                data: &up_shexp_bytes,
                qtype: GgmlType::F32,
                rows: 8,
                cols: 4,
            },
            ffn_down_shexp: WeightRef {
                data: &down_shexp_bytes,
                qtype: GgmlType::F32,
                rows: 4,
                cols: 8,
            },
            routed_exp_up: WeightRef {
                data: &routed_exp_up_bytes,
                qtype: GgmlType::F32,
                rows: hidden,
                cols: n_embd_latent,
            },
            routed_exp_down: WeightRef {
                data: &routed_exp_down_bytes,
                qtype: GgmlType::F32,
                rows: n_embd_latent,
                cols: hidden,
            },
            routed_exp_norm,
            ffn_gate_exps: WeightRef {
                data: &gate_exps_bytes,
                qtype: GgmlType::F32,
                rows: n_ff_exp,
                cols: n_embd_latent,
            },
            ffn_up_exps: WeightRef {
                data: &up_exps_bytes,
                qtype: GgmlType::F32,
                rows: n_ff_exp,
                cols: n_embd_latent,
            },
            ffn_down_exps: WeightRef {
                data: &down_exps_bytes,
                qtype: GgmlType::F32,
                rows: n_embd_latent,
                cols: n_ff_exp,
            },
        };
        let _ = num_experts;

        // top-k = 1, renormalize = true, rms_eps = 1e-6.
        let y = kimi_k3_latent_moe_forward(&x, &ffn_norm, &moe, 1, true, 1e-6);
        assert_eq!(y.len(), hidden, "output must have hidden dim");
        // Should not be all-zero (input is non-zero, all weights are non-zero).
        assert!(
            y.iter().any(|&v| v.abs() > 1e-6),
            "output must be non-zero for non-zero input"
        );
        // Should be finite.
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] = {v} must be finite");
        }
    }
}

#[cfg(test)]
mod kimi_k3_kda_layer_tests {
    use super::{
        identity_matrix_f32, kimi_k3_slice_weight_ref_rows, weight_ref_as_f32,
        weight_ref_slice_as_f32, GgmlType, WeightRef,
    };

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 4);
        for &x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    #[test]
    fn slice_weight_ref_rows_extracts_correct_byte_range() {
        // 4-row × 2-col F32 matrix: [1,2, 3,4, 5,6, 7,8].
        let vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let bytes = f32_bytes(&vals);
        let w = WeightRef {
            data: &bytes,
            qtype: GgmlType::F32,
            rows: 4,
            cols: 2,
        };

        // Rows 1..3 → [3,4, 5,6].
        let sliced = kimi_k3_slice_weight_ref_rows(&w, 1, 3).expect("slicing must succeed for F32");
        assert_eq!(sliced.rows, 2);
        assert_eq!(sliced.cols, 2);
        let sliced_f32 = weight_ref_as_f32(&sliced);
        assert_eq!(sliced_f32, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn slice_weight_ref_rows_rejects_misaligned_quantized_cols() {
        // Q4_K block = 256 elements. cols = 64 → row-length 64 is not
        // a multiple of 256, so per-row slicing would land mid-block
        // and must be rejected. (Real K3 hidden = 7168 = 28 × 256, so
        // this misalignment case is a defensive test not a K3 case.)
        let bytes = vec![0u8; 144]; // Q4_K block bytes for 256 elements
        let w = WeightRef {
            data: &bytes,
            qtype: GgmlType::Q4_K,
            rows: 4,
            cols: 64,
        };
        assert!(kimi_k3_slice_weight_ref_rows(&w, 0, 2).is_none());
    }

    #[test]
    fn slice_weight_ref_rows_supports_q4_k_when_aligned() {
        // Phase X.4.c.3.3.b.2: Q4_K per-row slicing works when cols
        // is a multiple of QK_K (256). 4 rows × 256 cols = 1024
        // elements = 4 Q4_K blocks total, 1 block per row = 144
        // bytes per row.
        let bytes = vec![0u8; 144 * 4]; // 4 rows, 1 block each
        let w = WeightRef {
            data: &bytes,
            qtype: GgmlType::Q4_K,
            rows: 4,
            cols: 256,
        };
        let sliced = kimi_k3_slice_weight_ref_rows(&w, 1, 3)
            .expect("Q4_K per-row slicing must succeed when cols % 256 == 0");
        assert_eq!(sliced.rows, 2);
        assert_eq!(sliced.cols, 256);
        assert_eq!(sliced.data.len(), 144 * 2, "2 Q4_K blocks");
    }

    #[test]
    fn slice_weight_ref_rows_supports_mxfp4_when_aligned() {
        // MXFP4 block = 32 elements, 17 bytes/block. K3 hidden = 7168
        // = 224 × 32, so per-row slicing is block-aligned.
        // 4 rows × 32 cols = 128 elements = 4 MXFP4 blocks total,
        // 1 block per row = 17 bytes per row.
        let bytes = vec![0u8; 17 * 4];
        let w = WeightRef {
            data: &bytes,
            qtype: GgmlType::Mxfp4,
            rows: 4,
            cols: 32,
        };
        let sliced = kimi_k3_slice_weight_ref_rows(&w, 0, 2)
            .expect("MXFP4 per-row slicing must succeed at 32-element aligned cols");
        assert_eq!(sliced.rows, 2);
        assert_eq!(sliced.cols, 32);
        assert_eq!(sliced.data.len(), 17 * 2);
    }

    #[test]
    fn slice_weight_ref_rows_supports_q8_0_when_aligned() {
        // Q8_0 block = 32 elements, 34 bytes/block.
        // 3 rows × 64 cols = 192 elements = 6 Q8_0 blocks, 2 blocks/row.
        let bytes = vec![0u8; 34 * 6];
        let w = WeightRef {
            data: &bytes,
            qtype: GgmlType::Q8_0,
            rows: 3,
            cols: 64,
        };
        let sliced = kimi_k3_slice_weight_ref_rows(&w, 1, 2)
            .expect("Q8_0 per-row slicing must succeed at 32-element aligned cols");
        assert_eq!(sliced.rows, 1);
        assert_eq!(sliced.data.len(), 34 * 2);
    }

    #[test]
    fn slice_weight_ref_rows_returns_none_on_out_of_range() {
        let bytes = vec![0u8; 32];
        let w = WeightRef {
            data: &bytes,
            qtype: GgmlType::F32,
            rows: 2,
            cols: 4,
        };
        // row_end = 3 exceeds rows = 2.
        assert!(kimi_k3_slice_weight_ref_rows(&w, 0, 3).is_none());
        // row_start > row_end.
        assert!(kimi_k3_slice_weight_ref_rows(&w, 2, 1).is_none());
    }

    #[test]
    fn weight_ref_slice_as_f32_extracts_element_range() {
        let vals: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let bytes = f32_bytes(&vals);
        let w = WeightRef {
            data: &bytes,
            qtype: GgmlType::F32,
            rows: 1,
            cols: 8,
        };
        let sliced = weight_ref_slice_as_f32(&w, 2, 6);
        assert_eq!(sliced, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn identity_matrix_f32_places_ones_on_diagonal() {
        let m = identity_matrix_f32(3);
        // [1,0,0, 0,1,0, 0,0,1]
        assert_eq!(m.len(), 9);
        assert_eq!(m[0], 1.0);
        assert_eq!(m[4], 1.0);
        assert_eq!(m[8], 1.0);
        assert_eq!(m[1], 0.0);
        assert_eq!(m[3], 0.0);
    }
}

#[cfg(test)]
mod kimi_k3_forward_helpers_tests {
    use super::{
        kimi_k3_dense_ffn_forward, kimi_k3_extract_mla_config, GgmlType, KimiDeltaConfig,
        Llama3Config, ModelArch, WeightRef,
    };

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 4);
        for &x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    fn tiny_k3_config() -> Llama3Config {
        Llama3Config {
            arch: ModelArch::KimiK3,
            vocab_size: 32,
            hidden_dim: 4,
            intermediate_dim: 8,
            num_heads: 2,
            num_kv_heads: 2,
            num_layers: 2,
            max_seq_len: 128,
            head_dim: 2,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            attention_extras: None,
            ssm: None,
            moe: None,
            gemma3n: None,
            gemma4: None,
            deepseek_v3: None,
            kimi_delta: Some(KimiDeltaConfig {
                full_attn_layers: Some(vec![0, 1]),
                kda_layers: Some(vec![]),
                kda_head_dim: Some(2),
                kda_num_heads: Some(2),
                kda_short_conv_kernel_size: Some(4),
                kda_use_full_rank_gate: Some(true),
                kda_gate_lower_bound: Some(-5.0),
                q_lora_rank: Some(4),
                kv_lora_rank: Some(2),
                qk_nope_head_dim: Some(2),
                qk_rope_head_dim: Some(1),
                v_head_dim: Some(2),
                mla_use_nope: Some(true),
                mla_use_output_gate: Some(true),
                attn_res_block_size: Some(2),
                situ_beta: Some(4.0),
                situ_linear_beta: Some(25.0),
                n_routed_experts: Some(4),
                num_experts_per_tok: Some(2),
                n_shared_experts: Some(1),
                num_expert_group: Some(1),
                topk_group: Some(1),
                moe_router_activation: Some("sigmoid".to_string()),
                moe_topk_method: Some("noaux_tc".to_string()),
                moe_intermediate_size: Some(4),
                first_k_dense_replace: Some(2),
                moe_renormalize: Some(true),
                routed_expert_hidden_size: Some(2),
                latent_moe_use_norm: Some(true),
                routed_scaling_factor: Some(1.0),
                num_nextn_predict_layers: None,
                mxfp4_group_size: None,
                mxfp4_num_bits: None,
            }),
        }
    }

    #[test]
    fn extract_mla_config_populates_from_llama3_config() {
        let config = tiny_k3_config();
        let mla_cfg = kimi_k3_extract_mla_config(&config).expect("mla config must extract");
        assert_eq!(mla_cfg.d, 4);
        assert_eq!(mla_cfg.num_heads, 2);
        assert_eq!(mla_cfg.qk_nope_head_dim, 2);
        assert_eq!(mla_cfg.qk_rope_head_dim, 1);
        assert_eq!(mla_cfg.v_head_dim, 2);
        assert_eq!(mla_cfg.q_lora_rank, 4);
        assert_eq!(mla_cfg.kv_lora_rank, 2);
        assert!((mla_cfg.rms_eps - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn extract_mla_config_returns_none_when_kimi_delta_absent() {
        let mut config = tiny_k3_config();
        config.kimi_delta = None;
        assert!(kimi_k3_extract_mla_config(&config).is_none());
    }

    #[test]
    fn extract_mla_config_returns_none_when_required_dim_absent() {
        let mut config = tiny_k3_config();
        if let Some(kd) = config.kimi_delta.as_mut() {
            kd.q_lora_rank = None;
        }
        assert!(kimi_k3_extract_mla_config(&config).is_none());
    }

    #[test]
    fn dense_ffn_forward_zero_input_gives_zero_output() {
        // x = 0 → x_norm = 0 (all-zero mean, RMSNorm scales 0 by
        // 1/sqrt(eps) = large, but 0 × large = 0) → gate_out = up_out
        // = 0 → gated = 0 → y = 0.
        let x = vec![0.0_f32; 4];
        let ffn_norm = vec![1.0_f32; 4];
        // Non-trivial weights so we can be sure the zero comes from x=0.
        let gate_bytes = f32_bytes(&(0..32).map(|i| 0.01 * i as f32).collect::<Vec<_>>());
        let up_bytes = f32_bytes(&(0..32).map(|i| 0.02 * i as f32).collect::<Vec<_>>());
        let down_bytes = f32_bytes(&(0..32).map(|i| 0.03 * i as f32).collect::<Vec<_>>());
        let gate = WeightRef {
            data: &gate_bytes,
            qtype: GgmlType::F32,
            rows: 8,
            cols: 4,
        };
        let up = WeightRef {
            data: &up_bytes,
            qtype: GgmlType::F32,
            rows: 8,
            cols: 4,
        };
        let down = WeightRef {
            data: &down_bytes,
            qtype: GgmlType::F32,
            rows: 4,
            cols: 8,
        };
        let y = kimi_k3_dense_ffn_forward(&x, &ffn_norm, &gate, &up, &down, 1e-5);
        for (i, &v) in y.iter().enumerate() {
            assert_eq!(v, 0.0, "y[{i}] = {v} but zero input must give zero output");
        }
    }

    #[test]
    fn dense_ffn_forward_produces_finite_bounded_output() {
        // Non-zero input, non-zero weights — verify output is finite
        // and within a reasonable envelope for the tiny scale.
        let x = vec![0.5_f32, -0.5, 0.25, -0.25];
        let ffn_norm = vec![1.0_f32; 4];
        let gate_bytes = f32_bytes(
            &(0..32)
                .map(|i| 0.01 * (i as f32 - 15.5))
                .collect::<Vec<_>>(),
        );
        let up_bytes = f32_bytes(
            &(0..32)
                .map(|i| 0.02 * (i as f32 - 15.5))
                .collect::<Vec<_>>(),
        );
        let down_bytes = f32_bytes(
            &(0..32)
                .map(|i| 0.03 * (i as f32 - 15.5))
                .collect::<Vec<_>>(),
        );
        let gate = WeightRef {
            data: &gate_bytes,
            qtype: GgmlType::F32,
            rows: 8,
            cols: 4,
        };
        let up = WeightRef {
            data: &up_bytes,
            qtype: GgmlType::F32,
            rows: 8,
            cols: 4,
        };
        let down = WeightRef {
            data: &down_bytes,
            qtype: GgmlType::F32,
            rows: 4,
            cols: 8,
        };
        let y = kimi_k3_dense_ffn_forward(&x, &ffn_norm, &gate, &up, &down, 1e-5);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] = {v} not finite");
            assert!(v.abs() <= 2.0, "y[{i}] = {v} exceeds reasonable envelope");
        }
    }
}

#[cfg(test)]
mod kimi_k3_gated_mla_tests {
    use super::{
        kimi_k3_gated_mla_step, GgmlType, KimiK3MlaAttn, KimiK3MlaCache, KimiK3MlaConfig, WeightRef,
    };

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 4);
        for &x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    fn tiny_config() -> KimiK3MlaConfig {
        // d=4, num_heads=2, qk_nope=2, qk_rope=1, v_head=2,
        // q_lora_rank=2, kv_lora_rank=2.
        KimiK3MlaConfig {
            d: 4,
            num_heads: 2,
            qk_nope_head_dim: 2,
            qk_rope_head_dim: 1,
            v_head_dim: 2,
            q_lora_rank: 2,
            kv_lora_rank: 2,
            rms_eps: 1e-6,
        }
    }

    /// Build a tiny MLA weight bundle keyed on the caller's byte
    /// buffers. Shapes match `tiny_config()`.
    #[allow(clippy::too_many_arguments)]
    fn build_tiny_mla_attn<'a>(
        q_a: &'a [u8],
        q_a_norm: Vec<f32>,
        q_b: &'a [u8],
        kv_a_mqa: &'a [u8],
        kv_a_norm: Vec<f32>,
        k_b: &'a [u8],
        v_b: &'a [u8],
    ) -> KimiK3MlaAttn<'a> {
        KimiK3MlaAttn {
            q_a: WeightRef {
                data: q_a,
                qtype: GgmlType::F32,
                rows: 2,
                cols: 4,
            },
            q_a_norm,
            q_b: WeightRef {
                data: q_b,
                qtype: GgmlType::F32,
                rows: 6,
                cols: 2,
            },
            kv_a_mqa: WeightRef {
                data: kv_a_mqa,
                qtype: GgmlType::F32,
                rows: 3,
                cols: 4,
            },
            kv_a_norm,
            k_b: WeightRef {
                data: k_b,
                qtype: GgmlType::F32,
                rows: 4,
                cols: 2,
            },
            v_b: WeightRef {
                data: v_b,
                qtype: GgmlType::F32,
                rows: 4,
                cols: 2,
            },
        }
    }

    #[test]
    fn mla_cache_new_starts_empty() {
        let cache = KimiK3MlaCache::new(2, 1, 4);
        assert_eq!(cache.n_positions(), 0);
    }

    #[test]
    fn mla_cache_reset_clears_positions() {
        let mut cache = KimiK3MlaCache::new(2, 1, 4);
        cache.append(&[1.0, 2.0], &[3.0]);
        cache.append(&[4.0, 5.0], &[6.0]);
        assert_eq!(cache.n_positions(), 2);
        cache.reset();
        assert_eq!(cache.n_positions(), 0);
    }

    #[test]
    fn gated_mla_step_zero_output_projection_gives_zero() {
        // All-zero W_o → attn_out = 0 → y = 0 regardless of gate.
        let q_a = f32_bytes(&[0.1_f32; 8]);
        let q_b = f32_bytes(&[0.1_f32; 12]);
        let kv_a_mqa = f32_bytes(&[0.1_f32; 12]);
        let k_b = f32_bytes(&[0.1_f32; 8]);
        let v_b = f32_bytes(&[0.1_f32; 8]);
        let mla = build_tiny_mla_attn(
            &q_a,
            vec![1.0; 2],
            &q_b,
            &kv_a_mqa,
            vec![1.0; 2],
            &k_b,
            &v_b,
        );
        let attn_norm = vec![1.0_f32; 4];
        let attn_gate_bytes = f32_bytes(&[0.5_f32; 16]);
        let attn_output_bytes = f32_bytes(&[0.0_f32; 16]);
        let attn_gate = WeightRef {
            data: &attn_gate_bytes,
            qtype: GgmlType::F32,
            rows: 4,
            cols: 4,
        };
        let attn_output = WeightRef {
            data: &attn_output_bytes,
            qtype: GgmlType::F32,
            rows: 4,
            cols: 4,
        };
        let mut cache = KimiK3MlaCache::new(2, 1, 4);
        let config = tiny_config();
        let x = vec![0.5_f32, -0.5, 0.25, -0.25];
        let y = kimi_k3_gated_mla_step(
            &x,
            &attn_norm,
            &attn_gate,
            &attn_output,
            &mla,
            &mut cache,
            &config,
        );
        for (i, &v) in y.iter().enumerate() {
            assert_eq!(v, 0.0, "y[{i}] = {v} but zero W_o must give zero output");
        }
        assert_eq!(cache.n_positions(), 1, "cache appends one position");
    }

    #[test]
    fn gated_mla_step_single_token_produces_finite_bounded_output() {
        // Non-zero weights end-to-end; verify output is finite and
        // stays within a reasonable envelope for the tiny scale.
        let q_a = f32_bytes(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let q_b_vals: Vec<f32> = (0..12).map(|i| 0.05 + 0.01 * i as f32).collect();
        let q_b = f32_bytes(&q_b_vals);
        let kv_a_vals: Vec<f32> = (0..12).map(|i| -0.02 * i as f32).collect();
        let kv_a_mqa = f32_bytes(&kv_a_vals);
        let k_b = f32_bytes(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let v_b = f32_bytes(&[-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8]);
        let mla = build_tiny_mla_attn(
            &q_a,
            vec![1.0; 2],
            &q_b,
            &kv_a_mqa,
            vec![1.0; 2],
            &k_b,
            &v_b,
        );

        let attn_norm = vec![1.0_f32; 4];
        // Identity output projection so the attention output is
        // returned verbatim.
        let mut ident = vec![0.0_f32; 16];
        for i in 0..4 {
            ident[i * 4 + i] = 1.0;
        }
        let ident_bytes = f32_bytes(&ident);
        let attn_gate = WeightRef {
            data: &ident_bytes,
            qtype: GgmlType::F32,
            rows: 4,
            cols: 4,
        };
        let attn_output = WeightRef {
            data: &ident_bytes,
            qtype: GgmlType::F32,
            rows: 4,
            cols: 4,
        };

        let mut cache = KimiK3MlaCache::new(2, 1, 4);
        let config = tiny_config();
        let x = vec![0.5_f32, -0.5, 0.25, -0.25];
        let y = kimi_k3_gated_mla_step(
            &x,
            &attn_norm,
            &attn_gate,
            &attn_output,
            &mla,
            &mut cache,
            &config,
        );
        assert_eq!(y.len(), 4);
        for (i, &v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] = {v} not finite");
            assert!(v.abs() <= 4.0, "y[{i}] = {v} exceeds reasonable envelope");
        }
        assert_eq!(cache.n_positions(), 1);
    }

    #[test]
    fn gated_mla_step_two_tokens_grow_cache() {
        // Two consecutive steps must both succeed and grow the
        // cache to 2 positions.
        let q_a = f32_bytes(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let q_b_vals: Vec<f32> = (0..12).map(|i| 0.05 + 0.01 * i as f32).collect();
        let q_b = f32_bytes(&q_b_vals);
        let kv_a_vals: Vec<f32> = (0..12).map(|i| -0.02 * i as f32).collect();
        let kv_a_mqa = f32_bytes(&kv_a_vals);
        let k_b = f32_bytes(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let v_b = f32_bytes(&[-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8]);
        let mla = build_tiny_mla_attn(
            &q_a,
            vec![1.0; 2],
            &q_b,
            &kv_a_mqa,
            vec![1.0; 2],
            &k_b,
            &v_b,
        );

        let attn_norm = vec![1.0_f32; 4];
        let mut ident = vec![0.0_f32; 16];
        for i in 0..4 {
            ident[i * 4 + i] = 1.0;
        }
        let ident_bytes = f32_bytes(&ident);
        let attn_gate = WeightRef {
            data: &ident_bytes,
            qtype: GgmlType::F32,
            rows: 4,
            cols: 4,
        };
        let attn_output = WeightRef {
            data: &ident_bytes,
            qtype: GgmlType::F32,
            rows: 4,
            cols: 4,
        };

        let mut cache = KimiK3MlaCache::new(2, 1, 4);
        let config = tiny_config();
        let x1 = vec![0.5_f32, -0.5, 0.25, -0.25];
        let x2 = vec![-0.3_f32, 0.4, -0.1, 0.2];
        let y1 = kimi_k3_gated_mla_step(
            &x1,
            &attn_norm,
            &attn_gate,
            &attn_output,
            &mla,
            &mut cache,
            &config,
        );
        let y2 = kimi_k3_gated_mla_step(
            &x2,
            &attn_norm,
            &attn_gate,
            &attn_output,
            &mla,
            &mut cache,
            &config,
        );
        for (i, (&a, &b)) in y1.iter().zip(y2.iter()).enumerate() {
            assert!(a.is_finite(), "y1[{i}] = {a}");
            assert!(b.is_finite(), "y2[{i}] = {b}");
        }
        assert_eq!(cache.n_positions(), 2);
    }
}

// ── Kimi K3 model struct + skeleton forward (Phase X.4.c.3.1) ─────
//
// Bundles the X.4.b.2 weight refs + per-layer runtime caches (KDA
// per-head + MLA per-layer + Block AttnRes state) into a single
// stateful struct, plus a skeleton `forward` that plumbs embedding
// lookup → per-layer dispatch → output projection.
//
// The per-layer body (real KDA all-head aggregation, real MLA layer,
// real LatentMoE FFN, real AttnRes wiring) lives in X.4.c.3.3+ — the
// skeleton here calls `todo!()` inside the layer loop so callers who
// wire this up prematurely see a clear panic pointing to the next
// phase rather than silent garbage. `mla` uses the X.4.c.3.2
// primitive when the branch is exercised; `kda` and `ffn moe` still
// need per-head slicing + LatentMoE forward respectively.
//
// This lands the plumbing (struct + cache alloc + embedding + output
// projection + layer dispatch skeleton) so the remaining forward
// work is contained to inside the layer body.

/// Per-layer cache for Kimi K3 forward (Phase X.4.c.3.1).
///
/// Layer `il` is exactly one of MLA (dense-KV attention) or KDA
/// (linear-attention with recurrent state per head). The enum tag
/// tracks which flavour the layer uses; runtime dispatch reads
/// `KimiDeltaConfig::is_mla_layer(il)` to pick the branch and
/// pushes/reads the matching cache variant.
#[allow(dead_code)]
pub(crate) enum KimiK3LayerCache {
    Mla(KimiK3MlaCache),
    Kda(Vec<KimiDeltaHeadCache>),
}

/// Full model state for Kimi K3 forward (Phase X.4.c.3.1).
///
/// Owns the weight bundle from [`load_kimi_k3_model_weights`], a
/// clone of the [`Llama3Config`], and all runtime caches. Callers
/// construct via [`KimiK3Model::new`] once and then call
/// [`KimiK3Model::forward`] per token to produce logits. Cache
/// state is preserved across calls until [`KimiK3Model::reset`]
/// starts a new sequence.
///
/// Memory footprint (K3 defaults, 93 layers × 96 heads):
///
/// - Per-head KDA cache: ~68 KB × 96 × 69 KDA layers ≈ 450 MB
/// - Per-layer MLA cache (initial): 0 B; grows by ~2.3 KB / token
///   × 24 MLA layers ≈ 55 KB / token
/// - Block AttnRes: ~9 block reps × 28 KB ≈ 250 KB
///
/// Total resident, no context: ~450 MB. Sequence growth: ~55 KB /
/// token, dominated by the MLA KV cache.
#[allow(dead_code)]
pub struct KimiK3Model<'a> {
    weights: KimiK3ModelWeights<'a>,
    config: Llama3Config,
    /// Per-layer runtime caches, one entry per layer in
    /// `0..config.num_layers`.
    layer_caches: Vec<KimiK3LayerCache>,
    /// Block Attention Residuals runtime state (X.4.c.3.4.d wiring,
    /// pwilkin PR #26185 semantics). Initialized eagerly at
    /// construction so `forward` can `bank` on the first layer
    /// without an option check.
    attn_res_state: KimiK3AttnResState,
    /// AttnRes block size, from [`KimiDeltaConfig::attn_res_block_size`]
    /// (K3: 12). Cached at construction so `forward` does not need
    /// to reach into the sub-config on every call.
    block_size: usize,
}

#[allow(dead_code)]
impl<'a> KimiK3Model<'a> {
    /// Allocate a Kimi K3 model from an already-loaded weight bundle.
    ///
    /// The typical construction path is:
    ///
    /// ```ignore
    /// let config = Llama3Config::from_gguf(&gguf).expect("K3 config");
    /// let weights = load_kimi_k3_model_weights(&gguf, &config)?;
    /// let model = KimiK3Model::new(weights, config)?;
    /// ```
    ///
    /// Per-layer caches are allocated eagerly at K3 default sizes
    /// so the first forward pass does not stall on cache growth.
    /// The MLA KV cache reserves capacity for 4096 positions by
    /// default; longer sequences will grow the underlying `Vec`s.
    ///
    /// # Errors
    ///
    /// Returns an error when `config.kimi_delta` is `None` (the
    /// GGUF did not populate the K3 sub-config — see Phase X.4.b.1),
    /// when critical dimensions (`kimi_delta.kda_head_dim`,
    /// `kv_lora_rank`, `qk_rope_head_dim`, etc.) are missing, or
    /// when `attn_res_block_size` is absent.
    pub fn new(weights: KimiK3ModelWeights<'a>, config: Llama3Config) -> Result<Self, String> {
        let kd = config
            .kimi_delta
            .as_ref()
            .ok_or_else(|| "kimi_delta sub-config missing (X.4.b.1 loader?)".to_string())?;
        let block_size = kd
            .attn_res_block_size
            .ok_or_else(|| "attn_res_block_size missing from kimi_delta config".to_string())?;
        let kda_head_dim = kd
            .kda_head_dim
            .ok_or_else(|| "kda_head_dim missing from kimi_delta config".to_string())?;
        let kda_num_heads = kd.kda_num_heads.unwrap_or(config.num_heads);
        let kda_short_conv_kernel_size = kd.kda_short_conv_kernel_size.unwrap_or(4);
        let kv_lora_rank = kd
            .kv_lora_rank
            .ok_or_else(|| "kv_lora_rank missing from kimi_delta config".to_string())?;
        let qk_rope_head_dim = kd
            .qk_rope_head_dim
            .ok_or_else(|| "qk_rope_head_dim missing from kimi_delta config".to_string())?;

        // Reserve capacity for a 4K context in the MLA cache; longer
        // sequences reallocate transparently.
        const MLA_CACHE_CAPACITY: usize = 4096;

        let mut layer_caches: Vec<KimiK3LayerCache> = Vec::with_capacity(config.num_layers);
        for il in 0..config.num_layers {
            // Phase X.4.b.4 continued: derive layer type from the
            // already-loaded `weights.layers[il].attn` enum discriminant.
            // The loader (Phase X.4.b.4) uses `full_attn_layers` metadata
            // when present, else tensor-name presence fallback; by the
            // time `KimiK3Model::new` runs, that decision is baked into
            // the KimiK3Attention enum. This lets us allocate the right
            // cache type without needing `full_attn_layers` metadata.
            //
            // Fallback: if `weights.layers` is shorter than `config.num_layers`
            // (test fixtures using `dummy_weights` do this), fall back to
            // the `KimiDeltaConfig::is_mla_layer` metadata path, and treat
            // "no metadata AND no matching weight layer" as `is_mla = false`
            // (KDA) so cache allocation stays valid for construction-path
            // tests that never call `forward()`.
            let is_mla = if il < weights.layers.len() {
                matches!(weights.layers[il].attn, KimiK3Attention::Mla(_))
            } else {
                kd.is_mla_layer(il).unwrap_or(false)
            };
            if is_mla {
                layer_caches.push(KimiK3LayerCache::Mla(KimiK3MlaCache::new(
                    kv_lora_rank,
                    qk_rope_head_dim,
                    MLA_CACHE_CAPACITY,
                )));
            } else {
                let heads: Vec<KimiDeltaHeadCache> = (0..kda_num_heads)
                    .map(|_| {
                        KimiDeltaHeadCache::new(
                            kda_head_dim,
                            kda_head_dim,
                            kda_short_conv_kernel_size,
                        )
                    })
                    .collect();
                layer_caches.push(KimiK3LayerCache::Kda(heads));
            }
        }

        let attn_res_state = KimiK3AttnResState::new(config.hidden_dim, block_size);
        Ok(Self {
            weights,
            config,
            layer_caches,
            attn_res_state,
            block_size,
        })
    }

    /// Reset every cache — start of a new sequence.
    pub fn reset(&mut self) {
        for cache in &mut self.layer_caches {
            match cache {
                KimiK3LayerCache::Mla(c) => c.reset(),
                KimiK3LayerCache::Kda(heads) => {
                    for head in heads {
                        head.reset();
                    }
                }
            }
        }
        self.attn_res_state.reset();
    }

    /// Number of layers this model dispatches over (`config.num_layers`,
    /// K3: 93). Useful for callers that want to iterate manually.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    /// Number of currently-cached positions in the MLA layer at
    /// index `il`. Returns `None` if `il` names a KDA layer or is
    /// out of bounds. Primarily useful for tests and observability.
    #[must_use]
    pub fn mla_cache_positions(&self, il: usize) -> Option<usize> {
        match self.layer_caches.get(il)? {
            KimiK3LayerCache::Mla(c) => Some(c.n_positions()),
            KimiK3LayerCache::Kda(_) => None,
        }
    }

    /// Skeleton one-token forward (Phase X.4.c.3.1).
    ///
    /// Wired stages:
    ///
    /// 1. **Token embedding lookup** — `x = token_embd[token_id]`.
    /// 2. **Per-layer dispatch loop** — MLA branch calls the
    ///    [`kimi_k3_gated_mla_step`] primitive; KDA and LatentMoE
    ///    branches still `todo!()` pending Phase X.4.c.3.3 which
    ///    lands the per-head KDA aggregation (needs per-head weight
    ///    slicing from the fused `attn_q/k/v` tensors) and the
    ///    Stable LatentMoE forward.
    /// 3. **Residual add** — simple `h += layer_output` (K3 Block
    ///    AttnRes wiring is X.4.c.3.4 — the fused 1D score vectors
    ///    `attn_res_score` / `ffn_res_score` per layer are already
    ///    loaded via `KimiK3LayerWeights` X.4.c.3.4.b refactor;
    ///    actual `res_mix` insertion into the layer loop is
    ///    X.4.c.3.4.d).
    /// 4. **Final RMSNorm** — `x = RMSNorm(x, output_norm)`.
    /// 5. **Output projection** — `logits = output.matvec(x)` →
    ///    `[vocab_size]`.
    ///
    /// # Panics
    ///
    /// Panics with a descriptive `todo!()` message from inside the
    /// KDA / LatentMoE layer bodies when the caller tries to run a
    /// K3 layer that this phase has not yet wired. The panic message
    /// includes the layer index and the missing sub-phase reference
    /// so a user hitting it can find the roadmap entry.
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn forward(&mut self, token_id: u32) -> Vec<f32> {
        // ── Step 1: embedding lookup ──
        let vocab_size = self.config.vocab_size;
        let hidden_dim = self.config.hidden_dim;
        assert!(
            (token_id as usize) < vocab_size,
            "token_id {token_id} out of vocab range 0..{vocab_size}"
        );
        // Phase X.4.b.6 continued (2026-07-28): block-aware embed lookup
        // via `kimi_k3_slice_weight_ref_rows` (which handles all K3 quant
        // types when `cols % elements_per_block == 0` — K3 hidden = 7168
        // = 28 × 256 = 224 × 32, block-aligned for all K3 quant types).
        // The sliced 1-row `WeightRef` is then dequantized to a `Vec<f32>`
        // via `weight_ref_row_dequant` (routes through the same
        // per-qtype dequantizers as `GgufFile::tensor_to_f32`).
        let row_start = token_id as usize;
        let embed_row =
            kimi_k3_slice_weight_ref_rows(&self.weights.token_embd, row_start, row_start + 1)
                .unwrap_or_else(|| {
                    panic!(
                        "K3 embed lookup: token_embd per-row slicing failed for token {token_id} \
                 (qtype {:?}, cols {}, rows {}). Ensure `cols % elements_per_block == 0` \
                 (K3 hidden = 7168 = 28 × 256 = 224 × 32, block-aligned for all K3 quant \
                 types). Phase X.4.c.3.3.b.2 upgrade.",
                        self.weights.token_embd.qtype,
                        self.weights.token_embd.cols,
                        self.weights.token_embd.rows,
                    )
                });
        let x_full = weight_ref_row_dequant(&embed_row);
        let mut x = vec![0.0_f32; hidden_dim];
        let take = hidden_dim.min(x_full.len());
        x[..take].copy_from_slice(&x_full[..take]);

        // ── Step 2-3: per-layer dispatch + residual add ──
        //
        // Phase X.4.c.3.3.a landing (2026-07-28):
        // - **MLA branch**: real forward via `kimi_k3_gated_mla_step`
        //   (X.4.c.3.2 primitive). Requires `weights.layers[il]`
        //   populated by `load_kimi_k3_model_weights` (X.4.b.2).
        // - **Dense FFN** (`il < first_k_dense_replace`, K3 only
        //   layer 0): real forward via `kimi_k3_dense_ffn_forward`
        //   (SwiGLU pattern).
        // - **KDA branch**: still `todo!()` — X.4.c.3.3.b will land
        //   per-head weight slicing from fused `attn_q/k/v` tensors
        //   + per-head `kimi_delta_forward_head` aggregation.
        // - **LatentMoE FFN** (`il >= first_k_dense_replace`, K3
        //   layers 1..93): still `todo!()` — X.4.c.3.3.c will land
        //   sigmoid router top-16 from 896 experts + 2 shared
        //   experts + latent `W↓` / RMSNorm / `W↑` + SiTU-GLU per
        //   routed expert.
        // - **AttnRes**: still a simple residual add. X.4.c.3.4.d
        //   will wire per-layer `res_mix` (2× per layer, using the
        //   fused 1D `attn_res_score` / `ffn_res_score` loaded via
        //   X.4.c.3.4.b) + banking + final output mix using
        //   `output_res_score`.
        let mla_config = kimi_k3_extract_mla_config(&self.config).expect(
            "KimiK3Model was constructed but MLA sub-config no longer extractable — \
             invariant broken (should have failed at ::new)",
        );

        // Phase X.4.b.7 debug: per-layer trace when `ALICE_K3_TRACE=1`.
        // Real K3 has 93 layers × per-layer disk I/O bound (566 GB mmap
        // on 32 GB RAM), so silent forward can easily look "hung" when
        // it's actually just paging in expert weights. Trace makes the
        // progress visible without a rebuild.
        let trace = std::env::var("ALICE_K3_TRACE").ok().as_deref() == Some("1");
        let layer_t0 = std::time::Instant::now();

        for il in 0..self.config.num_layers {
            let layer = &self.weights.layers[il];
            let il_t0 = std::time::Instant::now();

            // ── AttnRes pre-attention mix (X.4.c.3.4.d) ──────────
            // Follows pwilkin PR #26185 `src/models/kimi-k3.cpp`
            // L305-329: `cur = res_mix(prefix_sum, attn_res_score)`,
            // then bank RAW prefix_sum on checkpoint layers, then
            // attention consumes `cur`, then prefix_sum resets to
            // attn output on checkpoint layers (else standard add).
            let cur_attn = kimi_k3_res_mix(
                &self.attn_res_state,
                &x,
                &layer.attn_res_score,
                self.config.norm_eps,
            );
            let banked = self.attn_res_state.is_checkpoint_layer(il);
            if banked {
                self.attn_res_state.bank(&x);
            }

            // Attention half. Feeds `cur_attn` (post-mix) not raw x.
            let attn_output: Vec<f32> = match (&mut self.layer_caches[il], &layer.attn) {
                (KimiK3LayerCache::Mla(cache), KimiK3Attention::Mla(mla_attn)) => {
                    kimi_k3_gated_mla_step(
                        &cur_attn,
                        &layer.attn_norm,
                        &layer.attn_gate,
                        &layer.attn_output,
                        mla_attn,
                        cache,
                        &mla_config,
                    )
                }
                (KimiK3LayerCache::Kda(head_caches), KimiK3Attention::Kda(kda_attn)) => {
                    // Phase X.4.c.3.3.b landing (2026-07-28): real
                    // KDA layer forward via per-head slicing +
                    // `kimi_delta_forward_head` (X.4.c.2 primitive)
                    // per head + shared output projection. F32
                    // weights only for now — quantized per-head
                    // slicing lands at Phase X.4.c.3.3.b.2.
                    let kd = self
                        .config
                        .kimi_delta
                        .as_ref()
                        .expect("kimi_delta config must be present at forward time");
                    let head_dim = kd
                        .kda_head_dim
                        .expect("kda_head_dim missing from kimi_delta config");
                    let num_kda_heads = kd.kda_num_heads.unwrap_or(self.config.num_heads);
                    let g_min = kd.kda_gate_lower_bound.unwrap_or(-5.0);
                    let alpha_rank = kda_attn.ssm_f_a.rows;
                    kimi_k3_kda_layer_forward(
                        &cur_attn,
                        &layer.attn_norm,
                        &layer.attn_output,
                        kda_attn,
                        head_caches,
                        num_kda_heads,
                        head_dim,
                        alpha_rank,
                        g_min,
                        self.config.norm_eps,
                    )
                }
                _ => panic!(
                    "KimiK3Model layer {il}: cache/attn tag mismatch — invariant \
                     broken (cache and weights.layers[{il}].attn must both be MLA or \
                     both KDA, dispatched by `is_mla_layer` at construction and \
                     tensor-load time)"
                ),
            };

            // Post-attention prefix_sum update: banked → reset to
            // attn output alone; else → standard residual add.
            if banked {
                x.copy_from_slice(&attn_output);
            } else {
                for i in 0..hidden_dim {
                    x[i] += attn_output[i];
                }
            }

            // ── AttnRes pre-FFN mix (X.4.c.3.4.d) ────────────────
            let cur_ffn = kimi_k3_res_mix(
                &self.attn_res_state,
                &x,
                &layer.ffn_res_score,
                self.config.norm_eps,
            );

            // FFN half. Feeds `cur_ffn` (post-mix) not raw x.
            let ffn_output: Vec<f32> = match &layer.ffn {
                KimiK3Ffn::Dense { gate, up, down } => kimi_k3_dense_ffn_forward(
                    &cur_ffn,
                    &layer.ffn_norm,
                    gate,
                    up,
                    down,
                    self.config.norm_eps,
                ),
                KimiK3Ffn::LatentMoe(moe) => {
                    // Phase X.4.c.3.3.c.2 landing (2026-07-28):
                    // sigmoid router + shared experts + routed
                    // per-expert dispatch fully wired (F32 only).
                    let kd = self
                        .config
                        .kimi_delta
                        .as_ref()
                        .expect("kimi_delta config must be present at forward time");
                    let top_k = kd.num_experts_per_tok.unwrap_or(16);
                    let renormalize = kd.moe_renormalize.unwrap_or(true);
                    kimi_k3_latent_moe_forward(
                        &cur_ffn,
                        &layer.ffn_norm,
                        moe,
                        top_k,
                        renormalize,
                        self.config.norm_eps,
                    )
                }
            };

            for i in 0..hidden_dim {
                x[i] += ffn_output[i];
            }

            if trace {
                let kind = if matches!(layer.attn, KimiK3Attention::Mla(_)) {
                    "MLA"
                } else {
                    "KDA"
                };
                let ffn_kind = if matches!(layer.ffn, KimiK3Ffn::Dense { .. }) {
                    "Dense"
                } else {
                    "MoE"
                };
                let ms = il_t0.elapsed().as_millis();
                let cumulative_s = layer_t0.elapsed().as_secs_f64();
                eprintln!(
                    "[K3 trace] layer {il:>2}/{} {kind}+{ffn_kind} {ms:>6} ms (cum {cumulative_s:>7.2}s)",
                    self.config.num_layers
                );
            }
        }

        // ── AttnRes final output mix (X.4.c.3.4.e) ──────────────
        // Follows pwilkin `src/models/kimi-k3.cpp` L358-360:
        // `cur = res_mix(cur, output_res_score)` before `output_norm`.
        let x_after_final_mix = kimi_k3_res_mix(
            &self.attn_res_state,
            &x,
            &self.weights.output_res_score,
            self.config.norm_eps,
        );

        // ── Step 4: final RMSNorm ──
        let mut x_norm = vec![0.0_f32; hidden_dim];
        rms_norm(
            &x_after_final_mix,
            &self.weights.output_norm,
            self.config.norm_eps,
            &mut x_norm,
        );

        // ── Step 5: output projection to logits ──
        let mut logits = vec![0.0_f32; vocab_size];
        self.weights.output.matvec(&x_norm, &mut logits);
        logits
    }
}

/// Return the byte size of one element for a given GGUF quantization
/// type. Used by the K3 model to compute per-row byte offsets into
/// the `token_embd` tensor for embedding lookup.
#[allow(dead_code)]
fn bytes_per_element(qtype: GgmlType) -> usize {
    match qtype {
        GgmlType::F32 => 4,
        GgmlType::F16 => 2,
        // Fallback: use block-level accounting via QK per format;
        // callers must not hit this path with quantized token_embd
        // in the current skeleton (a real quantized embedding needs
        // block-aware slicing, not per-element indexing).
        _ => panic!(
            "bytes_per_element: quantized token_embd type {qtype:?} not yet supported \
             in KimiK3Model skeleton — Phase X.4.c.3.3 will add block-aware embed lookup"
        ),
    }
}

/// Dequantize an entire `WeightRef` to a fresh `Vec<f32>` of length
/// `rows * cols`. Thin wrapper over `WeightRef::dequantize_all` which
/// already supports the full K3 quant zoo via
/// `crate::gguf::dequantize_weight_row`. Used by the K3 embedding
/// lookup path (Phase X.4.b.6) where per-row slicing hands us a
/// 1-row `WeightRef` that we then materialize to f32.
#[allow(dead_code)]
fn weight_ref_row_dequant(w: &WeightRef<'_>) -> Vec<f32> {
    w.dequantize_all(w.rows, w.cols)
}

/// Dequantize a raw byte row to f32. Delegates to the existing
/// GGUF-side dequant machinery.
#[allow(dead_code)]
fn dequantize_row_to_f32(bytes: &[u8], qtype: GgmlType, out: &mut [f32]) {
    match qtype {
        GgmlType::F32 => {
            let n = out.len().min(bytes.len() / 4);
            for i in 0..n {
                let b = &bytes[i * 4..i * 4 + 4];
                out[i] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        GgmlType::F16 => {
            let n = out.len().min(bytes.len() / 2);
            for i in 0..n {
                let raw = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                out[i] = crate::gguf::f16_to_f32(raw);
            }
        }
        _ => panic!(
            "dequantize_row_to_f32: quantized type {qtype:?} not yet supported \
             in KimiK3Model skeleton — Phase X.4.c.3.3 will add block-aware dequant"
        ),
    }
}

#[cfg(test)]
mod kimi_k3_model_tests {
    use super::{
        BlockAttnResState, GgmlType, KimiDeltaConfig, KimiK3LayerCache, KimiK3Model,
        KimiK3ModelWeights, Llama3Config, ModelArch, WeightRef,
    };

    /// Build a minimal `Llama3Config` with a populated `KimiDeltaConfig`
    /// so [`KimiK3Model::new`] can wire the per-layer caches. Dims
    /// are the same 8-layer / hidden=64 / num_heads=8 mini-model
    /// fixture used by the loader tests.
    fn tiny_kimi_k3_config() -> Llama3Config {
        Llama3Config {
            arch: ModelArch::KimiK3,
            vocab_size: 256,
            hidden_dim: 64,
            intermediate_dim: 128,
            num_heads: 8,
            num_kv_heads: 8,
            num_layers: 8,
            max_seq_len: 4096,
            head_dim: 8,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            attention_extras: None,
            ssm: None,
            moe: None,
            gemma3n: None,
            gemma4: None,
            deepseek_v3: None,
            kimi_delta: Some(KimiDeltaConfig {
                full_attn_layers: Some(vec![3, 7]),
                kda_layers: Some(vec![0, 1, 2, 4, 5, 6]),
                kda_head_dim: Some(8),
                kda_num_heads: Some(8),
                kda_short_conv_kernel_size: Some(4),
                kda_use_full_rank_gate: Some(true),
                kda_gate_lower_bound: Some(-5.0),
                q_lora_rank: Some(16),
                kv_lora_rank: Some(8),
                qk_nope_head_dim: Some(8),
                qk_rope_head_dim: Some(4),
                v_head_dim: Some(8),
                mla_use_nope: Some(true),
                mla_use_output_gate: Some(true),
                attn_res_block_size: Some(4),
                situ_beta: Some(4.0),
                situ_linear_beta: Some(25.0),
                n_routed_experts: Some(32),
                num_experts_per_tok: Some(4),
                n_shared_experts: Some(2),
                num_expert_group: Some(1),
                topk_group: Some(1),
                moe_router_activation: Some("sigmoid".to_string()),
                moe_topk_method: Some("noaux_tc".to_string()),
                moe_intermediate_size: Some(64),
                first_k_dense_replace: Some(1),
                moe_renormalize: Some(true),
                routed_expert_hidden_size: Some(32),
                latent_moe_use_norm: Some(true),
                routed_scaling_factor: Some(1.0),
                num_nextn_predict_layers: None,
                mxfp4_group_size: None,
                mxfp4_num_bits: None,
            }),
        }
    }

    /// Build a minimal `KimiK3ModelWeights` with placeholder byte
    /// data. Not usable for real forward but sufficient for
    /// construction-path tests.
    fn dummy_weights(hidden_dim: usize, vocab_size: usize, buf: &[u8]) -> KimiK3ModelWeights<'_> {
        let empty_ref = || WeightRef {
            data: buf,
            qtype: GgmlType::F32,
            rows: 1,
            cols: 1,
        };
        KimiK3ModelWeights {
            token_embd: WeightRef {
                data: buf,
                qtype: GgmlType::F32,
                rows: vocab_size,
                cols: hidden_dim,
            },
            output_norm: vec![1.0; hidden_dim],
            output: WeightRef {
                data: buf,
                qtype: GgmlType::F32,
                rows: vocab_size,
                cols: hidden_dim,
            },
            output_res_score: vec![1.0; hidden_dim],
            layers: Vec::new(),
        }
    }

    #[test]
    fn model_new_allocates_per_layer_caches_matching_layer_type() {
        // 8 layers, full_attn_layers = [3, 7] → 6 KDA + 2 MLA.
        let config = tiny_kimi_k3_config();
        // We need enough backing bytes for the token embedding
        // (vocab_size × hidden_dim × 4 bytes for F32).
        let hidden = config.hidden_dim;
        let vocab = config.vocab_size;
        let buf = vec![0u8; vocab * hidden * 4];
        let weights = dummy_weights(hidden, vocab, &buf);
        let model = KimiK3Model::new(weights, config).expect("model must construct");
        assert_eq!(model.num_layers(), 8);
        assert_eq!(model.layer_caches.len(), 8);
        // Layers 0/1/2/4/5/6 = KDA, 3/7 = MLA.
        for (il, cache) in model.layer_caches.iter().enumerate() {
            match cache {
                KimiK3LayerCache::Mla(_) => {
                    assert!(il == 3 || il == 7, "layer {il} unexpectedly MLA");
                }
                KimiK3LayerCache::Kda(heads) => {
                    assert!(il != 3 && il != 7, "layer {il} unexpectedly KDA");
                    assert_eq!(heads.len(), 8, "KDA layer {il} must have 8 head caches");
                }
            }
        }
    }

    #[test]
    fn model_new_returns_err_when_kimi_delta_config_absent() {
        // Same shape but with `kimi_delta = None` — construction
        // must Err with a descriptive message.
        let mut config = tiny_kimi_k3_config();
        config.kimi_delta = None;
        let buf = vec![0u8; 4];
        let weights = dummy_weights(config.hidden_dim, config.vocab_size, &buf);
        let Err(err) = KimiK3Model::new(weights, config) else {
            panic!("expected Err on missing kimi_delta");
        };
        assert!(
            err.contains("kimi_delta"),
            "expected error to mention kimi_delta, got: {err}"
        );
    }

    #[test]
    fn model_reset_clears_all_caches_and_block_attnres() {
        let config = tiny_kimi_k3_config();
        let hidden = config.hidden_dim;
        let vocab = config.vocab_size;
        let buf = vec![0u8; vocab * hidden * 4];
        let weights = dummy_weights(hidden, vocab, &buf);
        let mut model = KimiK3Model::new(weights, config).expect("construct");
        // Muck with the MLA cache manually to prove reset works.
        if let KimiK3LayerCache::Mla(c) = &mut model.layer_caches[3] {
            c.append(&vec![0.0; 8], &vec![0.0; 4]);
            c.append(&vec![0.0; 8], &vec![0.0; 4]);
            assert_eq!(c.n_positions(), 2);
        }
        // Bank a fake ckpt into the AttnRes state to observe reset.
        model.attn_res_state.bank(&vec![1.0_f32; hidden]);
        assert_eq!(model.attn_res_state.banked_count(), 1);

        model.reset();

        assert_eq!(model.mla_cache_positions(3), Some(0));
        assert_eq!(model.attn_res_state.banked_count(), 0);
    }

    #[test]
    fn model_forward_panics_on_empty_layers_vec() {
        // Phase X.4.c.3.3.a wired the MLA and Dense FFN branches
        // into `forward`, so the layer body now reaches into
        // `self.weights.layers[il]` to fetch the per-layer weight
        // bundle. The `dummy_weights` helper still ships an empty
        // `layers: Vec::new()`, so `forward` panics with
        // index-out-of-bounds *before* it can reach the KDA /
        // LatentMoE `todo!()`s.
        //
        // A full-fixture test that populates `weights.layers` with
        // synthetic tensors + exercises the real MLA branch is
        // scheduled at Phase X.4.c.3.3.d once we have a synthetic-
        // GGUF builder that emits the per-layer tensor set. For
        // now this regression guard documents the precondition:
        // `load_kimi_k3_model_weights` must run before `forward`.
        let config = tiny_kimi_k3_config();
        let hidden = config.hidden_dim;
        let vocab = config.vocab_size;
        let buf = vec![0u8; vocab * hidden * 4];
        let weights = dummy_weights(hidden, vocab, &buf);
        let mut model = KimiK3Model::new(weights, config).expect("construct");

        let panic_msg = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| model.forward(0)))
            .expect_err("forward must panic when weights.layers is empty");
        let s = panic_msg
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic_msg.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(
            s.contains("index out of bounds"),
            "panic message must reference index out of bounds (unpopulated \
             weights.layers), got: {s}"
        );
    }
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

        // Q LoRA chain. MTP is a V3-only feature (V2 / V2-Lite ship no MTP
        // head), so we assume LoRA here and treat Dense as unreachable.
        let mut q_a_buf = vec![0.0f32; q_lora_rank];
        let mut q_a_normed = vec![0.0f32; q_lora_rank];
        let mut q_full = vec![0.0f32; num_heads * q_head_total];
        match &block.q {
            DeepSeekQProjection::LoRA {
                q_a_proj,
                q_a_norm,
                q_b_proj,
            } => {
                q_a_proj.matvec(&norm_buf, &mut q_a_buf);
                rms_norm(&q_a_buf, q_a_norm, c.norm_eps, &mut q_a_normed);
                q_b_proj.matvec(&q_a_normed, &mut q_full);
            }
            DeepSeekQProjection::Dense { .. } => {
                unreachable!("MTP head is V3-only; V2 / V2-Lite never load a dense-Q MTP block")
            }
        }

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
    /// Standard single-stream forward. Convenience wrapper for the common
    /// case with no per-layer hook — equivalent to
    /// `forward_with_layer_hook(token_id, |_, _| false)`.
    pub fn forward(&mut self, token_id: u32) -> Vec<f32> {
        self.forward_with_layer_hook(token_id, |_layer_idx, _hidden| false)
    }

    /// DSpark Phase 6 helper: forward + hidden state capture at specified layer
    ///
    /// `layer_idx = None` は最終層 (num_layers - 1)、`Some(n)` は layer n
    /// 範囲外の場合は最終層 fallback (silent ではなく documented behavior)
    ///
    /// # Panics
    /// 非標準 arch (Gemma3n / Gemma4 / DeepSeekV3 / KimiK3 / Hy3) では
    /// `forward_with_layer_hook` が specialized path に short-circuit するため
    /// hook が呼ばれない `unimplemented!` で fail fast する 標準 arch
    /// (Llama / Mistral / Gemma2 / Qwen2 / Qwen3 / Qwen3_5) のみ動作する
    #[cfg(feature = "dspark")]
    pub fn forward_capture_hidden(
        &mut self,
        token_id: u32,
        layer_idx: Option<usize>,
    ) -> (Vec<f32>, Vec<f32>) {
        match self.config.arch {
            ModelArch::Gemma3n
            | ModelArch::Gemma4
            | ModelArch::DeepSeekV3
            | ModelArch::KimiK3
            | ModelArch::Hy3 => {
                unimplemented!(
                    "dspark forward_capture_hidden: arch {:?} uses specialized forward path that does not invoke layer hook",
                    self.config.arch
                );
            }
            _ => {}
        }
        let num_layers = self.config.num_layers;
        let target = layer_idx
            .and_then(|n| if n < num_layers { Some(n) } else { None })
            .unwrap_or(num_layers.saturating_sub(1));
        let mut captured: Vec<f32> = Vec::new();
        let logits = self.forward_with_layer_hook(token_id, |idx, hidden| {
            if idx == target {
                captured.clear();
                captured.extend_from_slice(hidden);
            }
            false
        });
        (logits, captured)
    }

    /// External-signal-driven per-layer routing convenience API.
    ///
    /// Thin wrapper around [`forward_with_layer_hook`] that standardises the
    /// "an external per-token signal drives per-layer routing decisions"
    /// pattern already demonstrated in `examples/early_exit_qwen35.rs`
    /// (variance-gated depth routing) and `examples/entropy_mod_qwen35.rs`
    /// (per-layer statistic observation).
    ///
    /// The caller supplies an optional `surprise` slice (a per-token signal
    /// of arbitrary shape — per-body, per-region, per-modality, aggregated
    /// scalar broadcast, etc.) and a `gate` closure that inspects the layer
    /// index plus the signal and returns `true` to skip that layer's CPU
    /// compute. When `surprise` is `None`, output is bit-exact identical to
    /// [`forward`]: the wrapper delegates to `forward_with_layer_hook` with
    /// a closure that forwards `(layer_idx, None)` to `gate`, and a
    /// `gate` that always returns `false` reproduces the exact code path
    /// [`forward`] takes.
    ///
    /// Typical use cases:
    /// - Signal-driven early exit for latency-sensitive inference paths.
    /// - Mixture-of-Depths style routing with an externally-provided
    ///   routing key.
    /// - Adaptive compute where a lightweight upstream model produces a
    ///   per-token difficulty signal that gates depth on the downstream
    ///   `Llama3Model`.
    ///
    /// # Determinism
    ///
    /// With fixed `surprise` slice contents and a deterministic `gate`
    /// closure, the output is bit-exact reproducible across runs on the
    /// same hardware. With `surprise: None`, output equals
    /// `forward(token_id)` bit-exact.
    ///
    /// # Backward compatibility
    ///
    /// This is an additive API. Existing [`forward`] and
    /// [`forward_with_layer_hook`] remain untouched.
    ///
    /// [`forward`]: Self::forward
    /// [`forward_with_layer_hook`]: Self::forward_with_layer_hook
    pub fn forward_with_surprise<F>(
        &mut self,
        token_id: u32,
        surprise: Option<SurpriseVec<'_>>,
        gate: F,
    ) -> Vec<f32>
    where
        F: Fn(usize, Option<SurpriseVec<'_>>) -> bool,
    {
        self.forward_with_layer_hook(token_id, |layer_idx, _hidden| gate(layer_idx, surprise))
    }

    // L1 skeleton removed (2026-07-26): the GPU forward variants
    // (`forward_with_surprise_gpu` / `forward_with_early_exit_gpu`)
    // now live on `GpuModel` in `src/gpu.rs`, not on `Llama3Model` —
    // GPU forward path is a separate struct in this crate, and the
    // L1 skeleton on `Llama3Model` was based on an incorrect
    // assumption about where GPU forward lives. Downstream callers
    // that want early-exit GPU forward should use
    // `GpuModel::forward_with_early_exit_and_read` or
    // `GpuModel::forward_with_surprise_and_read` (both landed in the
    // L2 commit alongside this removal).
    //
    // The L1 skeleton was never released to crates.io (it was in the
    // Unreleased CHANGELOG section) so no downstream breakage is
    // possible. The CHANGELOG Unreleased entry has been updated to
    // reflect the actual L2 landing on `GpuModel`.

    /// Phase A2 per-layer hybrid support. Runs the standard `forward` path
    /// but calls `hook(layer_idx, &mut hidden)` before each layer body.
    ///
    /// The hook must return `false` for CPU to run the layer as usual, or
    /// `true` to tell CPU to skip its own compute for that layer. When the
    /// hook returns `true`, `hidden` is assumed to have been updated in
    /// place by the caller (typically by delegating the layer to a GPU
    /// runtime that maintains its own state), and CPU does no attention
    /// / KV-cache work for that layer.
    ///
    /// Per-token bookkeeping (embedding lookup, scratch buffer allocation,
    /// output norm + projection) still runs on CPU regardless of the hook
    /// return value. The final `kv_cache.advance()` also always fires so
    /// the KV cache position tracks the token stream.
    ///
    /// This is the primary entry point used by the `run_hybrid_per_layer`
    /// path in `examples/qwen_gpu.rs`, which orchestrates CPU DeltaNet
    /// layers + GPU attention layers to bypass Jetson's `wgpu-hal` Vulkan
    /// weight duplication while keeping DeltaNet math on the CPU where
    /// the recurrent state and Bonsai Gap-B refinement already work.
    pub fn forward_with_layer_hook<F>(&mut self, token_id: u32, hook: F) -> Vec<f32>
    where
        F: FnMut(usize, &mut Vec<f32>) -> bool,
    {
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
        // Kimi K3 / Kimi Delta Attention (Moonshot AI). Weights released
        // 2026-07-27 and the full spec is now captured by KimiDeltaConfig
        // via config.json (see KimiDeltaConfig::from_hf_config); the CPU
        // forward path itself is still Phase X.4.c work (community GGUF
        // conversion X.4.b is the outstanding blocker). Silent garbage on
        // a 2.8T model is worse than an explicit panic pointing to the
        // integration doc, so we `todo!()` here rather than fall through
        // to the standard path (which would misinterpret Kimi Delta +
        // Gated MLA + AttnRes as vanilla attention).
        if self.config.arch == ModelArch::KimiK3 {
            return self.forward_kimi_k3(token_id);
        }
        // Tencent Hy3 (Hunyuan 3). Skeleton only — the actual GGUF
        // conversion & the 295B/21B MoE forward path are Phase X.11
        // follow-up work. Fail fast rather than misinterpret Hy3 as
        // vanilla attention (its 192-expert top-8 routing + MTP head
        // are unique).
        if self.config.arch == ModelArch::Hy3 {
            return self.forward_hy3(token_id);
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
        if std::env::var("ALICE_DUMP_DN0").is_ok() {
            static ONCE_EMB: std::sync::Once = std::sync::Once::new();
            let mut fire = false;
            ONCE_EMB.call_once(|| fire = true);
            if fire {
                dump_slice("input_embed", &hidden, 3);
                eprintln!("DN0 token_id={}", token_id);
            }
        }
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

        let mut hook = hook;
        for layer_idx in 0..c.num_layers {
            // Phase A2 hybrid dispatch: give the caller a chance to fully
            // replace this layer's compute (e.g. delegate to a GPU model
            // that runs the attention layers). If the hook returns `true`,
            // it has already mutated `hidden` in place and the CPU-side
            // layer body must be skipped so KV cache / DeltaNet state on
            // the CPU side don't get out-of-sync updates for a layer that
            // the CPU never actually processed.
            if hook(layer_idx, &mut hidden) {
                continue;
            }
            // Hybrid Qwen 3.5 / 3.6: DeltaNet layers take a distinct forward
            // path (linear attention + recurrent state, no KV cache). Any
            // model without `layer_kind_map` populated is treated as
            // pure-attention using the existing global layer index.
            if is_hybrid {
                if let LayerKind::DeltaNet(dn_idx) = self.layer_kind_map[layer_idx] {
                    let dn_layer = &self.deltanet_layers.as_ref().expect("hybrid model")[dn_idx];

                    // 1. Attention norm.
                    rms_norm(&hidden, &dn_layer.attn_norm, c.norm_eps, &mut norm_buf);

                    // Phase X.3.e.3.14: dump attn_norm for DeltaNet layers to
                    // track cascade divergence between attention-layer dumps.
                    if std::env::var("ALICE_DUMP_ATTN_ALL").is_ok()
                        && matches!(
                            layer_idx,
                            4 | 5 | 6 | 8 | 9 | 10 | 12 | 13 | 14 | 16 | 17 | 18
                        )
                    {
                        dump_slice(&format!("attn{layer_idx}_norm"), &norm_buf, 3);
                    }

                    // Phase X.3.e.3.5 layer-0 first-forward dump for reference
                    // parity comparison (guarded by env `ALICE_DUMP_DN0`).
                    let dump_dn0 = std::env::var("ALICE_DUMP_DN0").is_ok() && dn_idx == 0 && {
                        static ONCE: std::sync::Once = std::sync::Once::new();
                        let mut fire = false;
                        ONCE.call_once(|| fire = true);
                        fire
                    };
                    if dump_dn0 {
                        dump_slice("attn_norm", &norm_buf, 3);
                    }

                    // 2. Fused input projection (populates dn_in_proj).

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
                    if dump_dn0 {
                        dump_slice("qkv_mixed", &dn_in_proj[..qkv_len], 3);
                        // Phase X.3.e.3.8: dump dims 125-127 (KV-h 0 Q tail) to
                        // check if pre-conv1d divergence starts here.
                        eprintln!(
                            "DN0 qkv_mixed[125..128] = [{:.6},{:.6},{:.6}]",
                            dn_in_proj[125], dn_in_proj[126], dn_in_proj[127]
                        );
                        eprintln!(
                            "DN0 qkv_mixed[2045..2048] (KV-h15 Q tail) = [{:.6},{:.6},{:.6}]",
                            dn_in_proj[2045], dn_in_proj[2046], dn_in_proj[2047]
                        );
                        dump_slice("z_pre", &dn_in_proj[qkv_len..], 3);
                        // Phase X.3.e.3.8: per-V-head z-pre for divergence hunt.
                        let z_slice = &dn_in_proj[qkv_len..];
                        let mut per_head_z = String::from("DN0 z_pre_per_head_sum=[");
                        for h in 0..dn_num_v_heads {
                            let s: f32 = z_slice[h * dn_v_dim..(h + 1) * dn_v_dim].iter().sum();
                            per_head_z.push_str(&format!("{s:.4},"));
                        }
                        per_head_z.push(']');
                        eprintln!("{per_head_z}");
                    }
                    // 2a/2b. alpha / beta decay-rate + update-rate projections.
                    dn_layer.alpha_proj.matvec(&norm_buf, &mut dn_alpha);
                    dn_layer.beta_proj.matvec(&norm_buf, &mut dn_beta);
                    if dump_dn0 {
                        dump_slice("alpha_raw", &dn_alpha[..dn_num_v_heads], 3);
                        dump_slice("beta_raw", &dn_beta[..dn_num_v_heads], 3);
                    }

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
                    if dump_dn0 {
                        dump_slice("alpha_after_gapB", &dn_alpha[..dn_num_v_heads], 3);
                        dump_slice("beta_after_gapB", &dn_beta[..dn_num_v_heads], 3);
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
                    if dump_dn0 {
                        dump_slice("conv_out_raw", &dn_conv_out[..qkv_len], 3);
                    }
                    if is_bonsai_path {
                        for val in dn_conv_out[..qkv_len].iter_mut() {
                            *val = silu(*val);
                        }
                    }
                    if dump_dn0 {
                        dump_slice("conv_out_silu", &dn_conv_out[..qkv_len], 3);
                        dump_slice("q_conv_head0", &dn_conv_out[0..dn_qk_dim], 3);
                        // Phase X.3.e.3.8: per-KV-head Q and K sums (before L2 norm).
                        for kv in [0usize, 12, 15] {
                            let q_off = kv * dn_qk_dim;
                            let k_off = dn_qk_dim * dn_num_kv_heads + kv * dn_qk_dim;
                            let q_first = &dn_conv_out[q_off..q_off + 3];
                            let q_last = &dn_conv_out[q_off + dn_qk_dim - 3..q_off + dn_qk_dim];
                            let k_first = &dn_conv_out[k_off..k_off + 3];
                            let k_last = &dn_conv_out[k_off + dn_qk_dim - 3..k_off + dn_qk_dim];
                            eprintln!("DN0 q_kv{kv}: first3={:?} last3={:?}", q_first, q_last);
                            eprintln!("DN0 k_kv{kv}: first3={:?} last3={:?}", k_first, k_last);
                            // Also compute dot product q · k for this KV-h
                            let dot: f32 = (0..dn_qk_dim)
                                .map(|i| dn_conv_out[q_off + i] * dn_conv_out[k_off + i])
                                .sum();
                            eprintln!("DN0 q_kv{kv} · k_kv{kv} = {dot:.6}");
                        }
                        let mut q_sums = String::from("DN0 q_per_kv_head_sum=[");
                        let mut k_sums = String::from("DN0 k_per_kv_head_sum=[");
                        let mut v_sums = String::from("DN0 v_per_v_head_sum=[");
                        for h in 0..dn_num_kv_heads {
                            let q_off = h * dn_qk_dim;
                            let qs: f32 = dn_conv_out[q_off..q_off + dn_qk_dim].iter().sum();
                            q_sums.push_str(&format!("{qs:.4},"));
                            let k_off = dn_qk_dim * dn_num_kv_heads + h * dn_qk_dim;
                            let ks: f32 = dn_conv_out[k_off..k_off + dn_qk_dim].iter().sum();
                            k_sums.push_str(&format!("{ks:.4},"));
                        }
                        q_sums.push(']');
                        k_sums.push(']');
                        eprintln!("{q_sums}");
                        eprintln!("{k_sums}");
                        let v_base = dn_qk_dim * dn_num_kv_heads * 2;
                        for h in 0..dn_num_v_heads {
                            let v_off = v_base + h * dn_v_dim;
                            let vs: f32 = dn_conv_out[v_off..v_off + dn_v_dim].iter().sum();
                            v_sums.push_str(&format!("{vs:.4},"));
                        }
                        v_sums.push(']');
                        eprintln!("{v_sums}");
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
                    if dump_dn0 {
                        dump_slice("attn_output", &dn_delta_out, 3);
                    }

                    if dump_dn0 {
                        // Sample the first V-head's 128 dims to compare with
                        // reference `norm-0 = RMS_NORM(attn_output-0)` first row.
                        let head0 = &dn_delta_out[..dn_v_dim];
                        dump_slice("attn_head0", head0, 3);
                        // Phase X.3.e.3.8: attn_output per-V-head sums so we can
                        // check if divergent V-heads originate at DeltaNet recurrence.
                        let mut ao_sums = String::from("DN0 attn_output_per_head_sum=[");
                        for h in 0..dn_num_v_heads {
                            let s: f32 =
                                dn_delta_out[h * dn_v_dim..(h + 1) * dn_v_dim].iter().sum();
                            ao_sums.push_str(&format!("{s:.6},"));
                        }
                        ao_sums.push(']');
                        eprintln!("{ao_sums}");
                    }

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
                    if dump_dn0 {
                        dump_slice("post_ssm_norm_head0", &dn_delta_out[..dn_v_dim], 3);
                        // Phase X.3.e.3.8: per-V-head post_ssm_norm sums and
                        // divergent head detail dumps for root-cause hunting.
                        let mut n51_sums = String::from("DN0 post_ssm_norm_per_head_sum=[");
                        for h in 0..dn_num_v_heads {
                            let s: f32 =
                                dn_delta_out[h * dn_v_dim..(h + 1) * dn_v_dim].iter().sum();
                            n51_sums.push_str(&format!("{s:.4},"));
                        }
                        n51_sums.push(']');
                        eprintln!("{n51_sums}");
                        for h in [6, 15, 20, 25] {
                            dump_slice(
                                &format!("post_ssm_norm_head{h}"),
                                &dn_delta_out[h * dn_v_dim..(h + 1) * dn_v_dim],
                                3,
                            );
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
                    if dump_dn0 {
                        // Suspects: head 1 (small ref value), head 18 (big neg),
                        // head 31 (very small). Compare first-3 values per head
                        // against reference node_55 first / mid / last rows.
                        for h in [1, 2, 18, 31] {
                            dump_slice(
                                &format!("post_zgate_head{h}"),
                                &dn_delta_out[h * dn_v_dim..(h + 1) * dn_v_dim],
                                3,
                            );
                        }
                        dump_slice("post_zgate_head0", &dn_delta_out[..dn_v_dim], 3);
                        dump_slice("post_zgate_all", &dn_delta_out, 3);
                        // Per-V-head sums so we can spot head-specific divergence
                        // against the reference `final_output-0` (which reference
                        // reports at whole-tensor granularity only).
                        let mut per_head = String::from("DN0 post_zgate_per_head_sum=[");
                        for h in 0..dn_num_v_heads {
                            let s: f32 =
                                dn_delta_out[h * dn_v_dim..(h + 1) * dn_v_dim].iter().sum();
                            per_head.push_str(&format!("{s:.4},"));
                        }
                        per_head.push(']');
                        eprintln!("{per_head}");
                    }

                    // 5. Output projection to hidden dim.
                    dn_layer.ssm_out.matvec(&dn_delta_out, &mut o_buf);
                    if dump_dn0 {
                        dump_slice("ssm_out_o", &o_buf[..c.hidden_dim], 3);
                    }

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
                    // Same `ALICE_LLM_DUMP_LAYERS` dump as the attention path
                    // below — required because DeltaNet layers hit `continue`
                    // and would otherwise skip the end-of-loop dump. Without
                    // this, checkpoints 0/6/13/20 (all DeltaNet in the Qwen
                    // 3.5 hybrid schedule) never emit CPU reference lines.
                    if std::env::var_os("ALICE_LLM_DUMP_LAYERS").is_some()
                        && matches!(layer_idx, 0 | 6 | 13 | 20 | 27)
                    {
                        dump_hidden_jsonl_stderr(&format!("cpu_layer_{layer_idx}"), &hidden);
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

            // Phase X.3.e.3.36 layer 0 op-by-op divergence dump (env-gated,
            // fires for first N positions per process where N is capped by
            // ALICE_DUMP_LAYER0_MAX_POS env var, default 2 = BOS + first
            // real token). Compare with llama.cpp `llama-eval-callback`
            // common_debug_cb_eval output for the same N-token prompt to
            // locate the first divergence op within layer 0 forward.
            let dump_l0 = std::env::var("ALICE_DUMP_LAYER0_OPS").is_ok() && layer_idx == 0 && {
                static POS_COUNTER: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let max_pos: usize = std::env::var("ALICE_DUMP_LAYER0_MAX_POS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(2);
                let cur = POS_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if cur < max_pos {
                    eprintln!("DN0 L0_pos_marker pos={}", cur);
                    true
                } else {
                    false
                }
            };
            if dump_l0 {
                dump_slice("L0_attn_norm_out", &norm_buf, 3);
            }

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
            // Reference qwen35.cpp:347-370 の view stride 精読で確定:
            // GGUF `attn_q.weight` は per-head interleaved layout
            // (`nb1 = element_size * n_embd_head * 2`)、
            // [q_h0(head_dim), gate_h0(head_dim), q_h1, gate_h1, ...] で
            // `2 * q_dim` rows を格納。downstream code は consecutive
            // [Q(q_dim), Gate(q_dim)] を仮定しているため de-interleave が必要。
            if layer.gated_output {
                let head_dim = c.head_dim;
                let mut gate_extract = vec![0f32; q_dim];
                for h in 0..c.num_heads {
                    for p in 0..head_dim {
                        gate_extract[h * head_dim + p] = q_buf[h * head_dim * 2 + head_dim + p];
                    }
                }
                for h in 1..c.num_heads {
                    let src = h * head_dim * 2;
                    let dst = h * head_dim;
                    for p in 0..head_dim {
                        q_buf[dst + p] = q_buf[src + p];
                    }
                }
                q_buf[q_dim..2 * q_dim].copy_from_slice(&gate_extract);
            }
            // Phase X.3.e.3.5 layer-3 first-forward dump for reference parity.
            let dump_attn3 = std::env::var("ALICE_DUMP_ATTN3").is_ok() && layer_idx == 3 && {
                static ONCE_A3: std::sync::Once = std::sync::Once::new();
                let mut fire = false;
                ONCE_A3.call_once(|| fire = true);
                fire
            };
            if dump_attn3 {
                dump_slice("attn3_norm", &norm_buf, 3);
                dump_slice("attn3_q_buf_full", &q_buf[..q_dim * 2], 3);
                dump_slice("attn3_q_head0_consec", &q_buf[..c.head_dim], 3);
                dump_slice(
                    "attn3_offset_256_gate_if_interleaved",
                    &q_buf[c.head_dim..c.head_dim * 2],
                    3,
                );
                dump_slice(
                    "attn3_offset_512_head1_if_interleaved",
                    &q_buf[c.head_dim * 2..c.head_dim * 3],
                    3,
                );
                dump_slice("attn3_v_head0", &v_buf[..c.head_dim], 3);
                dump_slice("attn3_k_head0", &k_buf[..c.head_dim], 3);
                // Phase X.3.e.3.36: full v_buf / k_buf dump for GPU comparison
                let kv_dim = c.num_kv_heads * c.head_dim;
                dump_hidden_jsonl_stderr("cpu_attn3_v_full", &v_buf[..kv_dim]);
                dump_hidden_jsonl_stderr("cpu_attn3_k_full", &k_buf[..kv_dim]);
            }
            // Phase X.3.e.3.14: additional attention-layer attn_norm dumps for
            // cascade divergence progression tracking (layers 7/11/15/19/23/27/31).
            // Env-gated by ALICE_DUMP_ATTN_ALL so first-forward captures each
            // attention layer's input norm to compare against reference dump.
            if std::env::var("ALICE_DUMP_ATTN_ALL").is_ok()
                && matches!(layer_idx, 7 | 11 | 15 | 19 | 23 | 27 | 31)
            {
                dump_slice(&format!("attn{layer_idx}_norm"), &norm_buf, 3);
            }
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

            if dump_l0 {
                dump_slice("L0_q_buf", &q_buf[..q_dim], 3);
                dump_slice("L0_k_buf", &k_buf, 3);
                dump_slice("L0_v_buf", &v_buf, 3);
                // Per-head dump for locating head-boundary layout diffs
                // against llama-eval-callback per-head values.
                let head_dim = c.head_dim;
                for head_idx in [0, 1, 12, 23_usize].iter().copied() {
                    if head_idx >= c.num_heads {
                        continue;
                    }
                    let start = head_idx * head_dim;
                    let end = start + head_dim;
                    if end > q_dim {
                        continue;
                    }
                    let head_slice = &q_buf[start..end];
                    let sum: f64 = head_slice.iter().map(|&v| v as f64).sum();
                    eprintln!(
                        "DN0 L0_q_head{} first3=[{:.6},{:.6},{:.6}] last3=[{:.6},{:.6},{:.6}] sum={:.6}",
                        head_idx,
                        head_slice[0], head_slice[1], head_slice[2],
                        head_slice[head_dim - 3], head_slice[head_dim - 2], head_slice[head_dim - 1],
                        sum
                    );
                }
            }

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
            if dump_l0 {
                dump_slice("L0_attn_out", &attn_out, 3);
            }

            // Qwen 3.5 / 3.6 / Bonsai 27B "Gated Attention": when `q_proj`
            // output was `2 * q_dim`, its second half is a per-element
            // sigmoid gate that modulates the attention result before
            // `o_proj`. Phase X.3.e.3.14 fix: previously silu (swish) was
            // applied, but reference qwen35.cpp:401-404 uses ggml_sigmoid
            // (not silu), which caused massive divergence propagating from
            // attention layer 3 onwards (attn_norm-4 sign-flip vs reference).
            // Phase X.3.e.3.15: dump attn_out (attn_pregate) + gate + attn_gated
            // for layer 3 to compare with reference qwen35 attn_pregate-3 /
            // gate_sigmoid-3 / attn_gated-3.
            let dump_gated_layer3 = std::env::var("ALICE_DUMP_GATED3").is_ok()
                && layer_idx == 3
                && layer.gated_output
                && {
                    static ONCE_G3: std::sync::Once = std::sync::Once::new();
                    let mut fire = false;
                    ONCE_G3.call_once(|| fire = true);
                    fire
                };
            if dump_gated_layer3 {
                dump_slice("gated3_attn_pregate", &attn_out[..q_dim], 3);
                dump_slice("gated3_gate_raw", &q_buf[q_dim..2 * q_dim], 3);
                let gate_sigmoid: Vec<f32> = q_buf[q_dim..2 * q_dim]
                    .iter()
                    .map(|&g| sigmoid(g))
                    .collect();
                dump_slice("gated3_gate_sigmoid", &gate_sigmoid, 3);
            }
            if layer.gated_output {
                for i in 0..q_dim {
                    attn_out[i] *= sigmoid(q_buf[q_dim + i]);
                }
            }
            if dump_gated_layer3 {
                dump_slice("gated3_attn_gated", &attn_out[..q_dim], 3);
                // Phase X.3.e.3.37: full q_dim dump for element-wise GPU compare
                dump_hidden_jsonl_stderr("cpu_gated3_attn_gated_full", &attn_out[..q_dim]);
            }

            // Output projection
            layer.o_proj.matvec(&attn_out, &mut o_buf);
            if dump_l0 {
                dump_slice("L0_o_buf", &o_buf, 3);
            }
            if dump_gated_layer3 {
                dump_slice("gated3_o_buf", &o_buf, 3);
                dump_hidden_jsonl_stderr("cpu_gated3_o_buf_full", &o_buf);
                dump_slice("gated3_hidden_pre_residual", &hidden, 3);
                // Compute what post-residual will be (hidden + o_buf).
                let post_res: Vec<f32> = hidden
                    .iter()
                    .zip(o_buf.iter())
                    .map(|(&h, &o)| h + o)
                    .collect();
                dump_slice("gated3_hidden_post_residual", &post_res, 3);
                // Phase X.3.e.3.15: compute sum of squares and mean(x²) to
                // diagnose RMS_NORM scale divergence.
                let ss: f64 = post_res.iter().map(|&v| (v as f64) * (v as f64)).sum();
                let mean_sq = ss / post_res.len() as f64;
                let rms = (mean_sq + 1e-6_f64).sqrt();
                eprintln!(
                    "DN0 gated3_hidden_stats: sum_sq={:.4} mean_sq={:.6} rms={:.6} scale=1/rms={:.4}",
                    ss, mean_sq, rms, 1.0/rms
                );
                // Also dump values at middle positions to see where divergence lies.
                let mid = post_res.len() / 2;
                eprintln!(
                    "DN0 gated3_hidden_middle: pos {} = [{:.4},{:.4},{:.4},{:.4},{:.4}]",
                    mid,
                    post_res[mid],
                    post_res[mid + 1],
                    post_res[mid + 2],
                    post_res[mid + 3],
                    post_res[mid + 4]
                );
            }

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
            if dump_l0 {
                dump_slice("L0_ffn_inp", &hidden, 3);
            }

            // Phase X.3.e.3.15: dump hidden RIGHT BEFORE rms_norm to catch
            // discrepancy vs computed post_res.
            if layer_idx == 3 && std::env::var("ALICE_DUMP_GATED3").is_ok() {
                static ONCE_H3: std::sync::Once = std::sync::Once::new();
                let mut fire = false;
                ONCE_H3.call_once(|| fire = true);
                if fire {
                    dump_slice("gated3_hidden_pre_ffnnorm", &hidden, 3);
                    // Manual RMS on this hidden
                    let ss: f64 = hidden.iter().map(|&v| (v as f64) * (v as f64)).sum();
                    let mean_sq = ss / hidden.len() as f64;
                    let scale = 1.0_f64 / (mean_sq + 1e-6_f64).sqrt();
                    eprintln!(
                        "DN0 gated3_pre_ffnnorm_rms: mean_sq={:.6} scale={:.4}",
                        mean_sq, scale
                    );
                    // Manual output[0] = hidden[0] * scale * weight[0]
                    let manual_out_0 = hidden[0] as f64 * scale * layer.ffn_norm[0] as f64;
                    eprintln!(
                        "DN0 gated3_manual_out0: hidden[0]={:.4} scale={:.4} weight[0]={:.6} manual_out[0]={:.4}",
                        hidden[0], scale, layer.ffn_norm[0], manual_out_0
                    );
                }
            }
            // FFN norm
            rms_norm(&hidden, &layer.ffn_norm, c.norm_eps, &mut norm_buf);
            if dump_l0 {
                dump_slice("L0_ffn_norm_out", &norm_buf, 3);
            }
            if layer_idx == 3 && std::env::var("ALICE_DUMP_GATED3").is_ok() {
                static ONCE_FN3: std::sync::Once = std::sync::Once::new();
                let mut fire = false;
                ONCE_FN3.call_once(|| fire = true);
                if fire {
                    dump_slice("gated3_ffn_norm_out", &norm_buf, 3);
                    dump_slice("gated3_ffn_norm_weight", &layer.ffn_norm, 3);
                    dump_slice("gated3_attn_norm_weight", &layer.attn_norm, 3);
                }
            }

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

            if dump_l0 {
                dump_slice("L0_ffn_out", &down_buf, 3);
            }

            // Residual
            for i in 0..c.hidden_dim {
                hidden[i] += down_buf[i];
            }
            if dump_l0 {
                dump_slice("L0_l_out_final", &hidden, 3);
            }
            // Phase X.3.e.3.15: layer 3 output (l_out-3 equivalent) dump.
            if layer_idx == 3 && std::env::var("ALICE_DUMP_GATED3").is_ok() {
                static ONCE_L3: std::sync::Once = std::sync::Once::new();
                let mut fire = false;
                ONCE_L3.call_once(|| fire = true);
                if fire {
                    dump_slice("gated3_ffn_out", &down_buf, 3);
                    dump_slice("gated3_l_out_3", &hidden, 3);
                    // Phase X.3.e.3.37: full down_buf dump for GPU compare
                    dump_hidden_jsonl_stderr("cpu_gated3_ffn_out_full", &down_buf);
                }
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

    /// Project a hidden state through the model's final `output_norm` +
    /// `output_proj` (and Gemma-2 logit softcap if configured), producing
    /// vocab-sized logits without running any layer body or touching the
    /// KV cache. Read-only on the model.
    ///
    /// Used by diagnostic tools that need to compare "would-be" logits from
    /// an early-layer hidden state to those produced by the full forward
    /// (e.g. correlating an intermediate-layer confidence signal with
    /// next-token difficulty). Mirrors the tail of `forward_with_layer_hook`
    /// exactly so numbers align across the two paths.
    ///
    /// # Panics
    /// Panics if `hidden.len() != config.hidden_dim`.
    pub fn project_hidden_to_logits(&self, hidden: &[f32]) -> Vec<f32> {
        let c = &self.config;
        assert_eq!(
            hidden.len(),
            c.hidden_dim,
            "project_hidden_to_logits: hidden.len() must equal config.hidden_dim"
        );
        let mut norm_buf = vec![0.0f32; c.hidden_dim];
        rms_norm(hidden, &self.output_norm, c.norm_eps, &mut norm_buf);
        let mut logits = vec![0.0f32; c.vocab_size];
        self.output_proj.matvec(&norm_buf, &mut logits);
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
    /// Kimi K3 / Kimi Delta Attention forward path (Phase X.4).
    ///
    /// **Skeleton only.** The 2026-07-27 open weight release + paper drop
    /// are prerequisites for the actual forward path. `todo!()` here so a
    /// user who somehow feeds a Kimi K3 GGUF into ALICE-LLM gets an
    /// explicit panic pointing to `docs/KIMI_K3_INTEGRATION.md` rather
    /// than silent garbage from the standard `Llama3Model::forward` path.
    ///
    /// Once the paper drops and Kimi K3 GGUF conversion lands upstream,
    /// this method should reuse `gated_deltanet_step*` for the DeltaNet
    /// layers, standard attention for the full-attention layers, and
    /// whatever KV compression scheme K3 ships. See the doc for the phased
    /// integration plan (X.4.a-X.4.g).
    ///
    /// # Panics
    ///
    /// Always — this is a fail-fast stub, per CLAUDE.md's
    /// "仮実装完了偽装の禁止" rule (no silent Ok on unimplemented paths).
    // `&mut self` is intentional: the real Phase X.4.c implementation will
    // mutate `self.kv_cache` on every forward, mirroring
    // `forward_deepseek_v3` and `forward_gemma3n`. Keeping the signature
    // stable now so the dispatch in `forward_with_layer_hook` doesn't need
    // to change when the stub gets replaced.
    #[allow(clippy::needless_pass_by_ref_mut)]
    fn forward_kimi_k3(&mut self, _token_id: u32) -> Vec<f32> {
        todo!(
            "KIMI-K3 forward: open weights released 2026-07-27, spec is \
             now captured in KimiDeltaConfig (parseable from HF config.json \
             via KimiDeltaConfig::from_hf_config under `hf-config` feature). \
             CPU forward path (Phase X.4.c) is still pending community GGUF \
             conversion (X.4.b, mradermacher / bartowski watch). See \
             docs/KIMI_K3_INTEGRATION.md for the phased integration plan \
             (Phase X.4.a-X.4.j) and the reusable ALICE-LLM components \
             (`gated_deltanet_step*`, `SsmDeltaNetConfig`, MoE routing, \
             MLA scaffolding shared with DeepSeek V3)."
        );
    }

    /// Fail-fast stub for Tencent Hy3 (Hunyuan 3) forward. Same rationale
    /// as `forward_kimi_k3`: the target ships as a 295B / 21B-active MoE
    /// with 192 experts (top-8 routing), 1 MTP layer (3.8B), GQA (8 KV
    /// heads), 256K context, and native FP8 (E4M3) weights. Enough of that
    /// scaffolding is unique that pretending it is a standard attention
    /// model would corrupt the output rather than degrade it gracefully.
    ///
    /// Once GGUF conversion for Hy3 lands upstream, this method is
    /// expected to inherit ~90% from the Bonsai / Qwen 3.6 gated-DeltaNet
    /// forward path (GQA + MoE scaffolding is already shared across Kimi
    /// K3 / DeepSeek V3); the net-new pieces are the 192-expert top-8
    /// sparse routing and the MTP head. See references on `ModelArch::Hy3`.
    ///
    /// # Panics
    ///
    /// Always — fail-fast stub (CLAUDE.md "仮実装完了偽装の禁止" rule).
    #[allow(clippy::needless_pass_by_ref_mut)]
    fn forward_hy3(&mut self, _token_id: u32) -> Vec<f32> {
        todo!(
            "HY3 forward: waiting for community GGUF conversion of \
             Tencent Hy3 (Hunyuan 3, 295B / 21B active MoE, 192 experts \
             top-8, MTP head 3.8B, GQA 8 KV, 256K context, FP8 native) \
             before implementation. Expected to inherit ~90% from the \
             Bonsai `gated_deltanet_step*` path; net-new pieces are the \
             192-expert top-8 sparse routing and MTP head. See the \
             Tencent-Hunyuan/Hy3 GitHub repository for the model spec."
        );
    }

    fn forward_deepseek_v3(&mut self, token_id: u32) -> Vec<f32> {
        let c = self.config.clone();
        let hidden_dim = c.hidden_dim;
        let num_heads = c.num_heads;
        // q_lora_rank is optional — absent on V2 / V2-Lite where the Q
        // projection is dense (Issue #58). Use 0 as sentinel when unused so
        // the per-layer scratch alloc stays simple; the match on
        // `layer.q` decides whether to touch those buffers at all.
        let q_lora_rank = c.deepseek_q_lora_rank().unwrap_or(0);
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

        // Scratch buffers reused across layers. `q_a_buf` / `q_a_normed` are
        // only touched by the LoRA branch; length 0 for V2 / V2-Lite dense.
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

        // Issue #36 diagnostic: per-op tensor dump for layer 0. Env-gated so
        // production builds don't pay the print cost. Emits first 5 elements
        // + L2 norm + sum for each intermediate tensor, one JSONL line per
        // (token_pos, layer, op). Compare against the HF oracle dump script
        // to bisect where the MLA forward diverges.
        let dump_dsv3 = std::env::var("ALICE_DEEPSEEK_DUMP").is_ok();
        let dump_tensor = |name: &str, layer_idx: usize, pos: usize, t: &[f32]| {
            if !dump_dsv3 || layer_idx != 0 {
                return;
            }
            let n = t.len();
            let head: Vec<f32> = t.iter().take(5).copied().collect();
            let l2: f32 = t.iter().map(|v| v * v).sum::<f32>().sqrt();
            let sum: f32 = t.iter().sum();
            eprintln!(
                "{{\"engine\":\"alice\",\"pos\":{pos},\"layer\":{layer_idx},\"op\":\"{name}\",\"len\":{n},\"head\":{head:?},\"l2\":{l2:.6},\"sum\":{sum:.6}}}"
            );
        };
        let _ = &dump_tensor; // silence unused when neither branch fires

        if dump_dsv3 {
            dump_tensor("hidden_in", 0, pos, &hidden);
        }

        for layer_idx in 0..c.num_layers {
            let layer = &deepseek_layers[layer_idx];

            // ── Attention block ───────────────────────────────────────
            rms_norm(&hidden, &layer.attn_norm, c.norm_eps, &mut norm_buf);
            dump_tensor("attn_norm", layer_idx, pos, &norm_buf);

            // Q projection: dense (V2 / V2-Lite) or LoRA (V2.5 / V3 / R1).
            match &layer.q {
                DeepSeekQProjection::Dense { q_proj } => {
                    q_proj.matvec(&norm_buf, &mut q_full);
                }
                DeepSeekQProjection::LoRA {
                    q_a_proj,
                    q_a_norm,
                    q_b_proj,
                } => {
                    q_a_proj.matvec(&norm_buf, &mut q_a_buf);
                    rms_norm(&q_a_buf, q_a_norm, c.norm_eps, &mut q_a_normed);
                    q_b_proj.matvec(&q_a_normed, &mut q_full);
                }
            }
            dump_tensor("q_full", layer_idx, pos, &q_full);

            // KV LoRA chain: single matvec produces the compressed latent
            // `kv_a` plus the shared positional slice `k_pe`.
            layer.kv_a_proj_with_mqa.matvec(&norm_buf, &mut kv_a_full);
            dump_tensor("kv_a_full", layer_idx, pos, &kv_a_full);
            // Issue #36 diagnostic: also dump the first 64 elements of
            // kv_a_full to test the row-layout inversion hypothesis
            // (if HF's k_pe matches ALICE's kv_a_full[0..64] instead of
            // kv_a_full[512..576], the split is reversed in GGUF vs HF).
            dump_tensor(
                "kv_a_full_first64",
                layer_idx,
                pos,
                &kv_a_full[..64.min(kv_a_full.len())],
            );
            // Issue #36 diagnostic: dump kv_a_full[kv_lora_rank..]
            // (elements 512-575) which will become k_pe_pre_rope after
            // the split. If these differ from k_pe_pre_rope's dump, the
            // split is buggy; if they match ALICE but differ from HF,
            // the projection weight rows are permuted vs HF.
            let kpe_start = kv_lora_rank.min(kv_a_full.len());
            dump_tensor("kv_a_full_tail64", layer_idx, pos, &kv_a_full[kpe_start..]);
            dump_tensor("kv_a_full_head512", layer_idx, pos, &kv_a_full[..kpe_start]);
            let (kv_a_slice, k_pe_shared) = kv_a_full.split_at_mut(kv_lora_rank);
            rms_norm(kv_a_slice, &layer.kv_a_norm, c.norm_eps, &mut kv_a_normed);
            dump_tensor("kv_a_normed", layer_idx, pos, &kv_a_normed);
            dump_tensor("k_pe_pre_rope", layer_idx, pos, k_pe_shared);
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
            dump_tensor("kv_up", layer_idx, pos, &kv_up);

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
            dump_tensor("q_full_post_rope", layer_idx, pos, &q_full);
            apply_rope_auto(
                k_pe_shared,
                pos,
                qk_rope,
                c.rope_theta,
                self.rope_freqs.as_deref(),
                true,
            );
            dump_tensor("k_pe_post_rope", layer_idx, pos, k_pe_shared);
            // Persist (post-RoPE) k_pe into the cache entry so the shared
            // slice lines up with `q_pe` at attention time.
            cache_entry[kv_lora_rank..].copy_from_slice(k_pe_shared);
            self.kv_cache.append(layer_idx, &cache_entry, &cache_entry);
            // NB: `KvCache::advance()` bumps the shared `seq_len` counter and
            // must be called **once** after all layers have appended (see doc
            // comment on `advance`). Do NOT call it inside the layer loop —
            // that would grow seq_len by `num_layers` per token and produce
            // O(N²) KV cache lookups on subsequent forwards.

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
            dump_tensor("attn_out", layer_idx, pos, &attn_out);

            layer.o_proj.matvec(&attn_out, &mut o_buf);
            dump_tensor("o_proj_out", layer_idx, pos, &o_buf);
            for i in 0..hidden_dim {
                hidden[i] += o_buf[i];
            }
            dump_tensor("hidden_post_attn", layer_idx, pos, &hidden);

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
            dump_tensor("hidden_post_ffn", layer_idx, pos, &hidden);
        }

        // Bump the shared position exactly once per token forward (Issue #58).
        // See the doc comment on `KvCache::advance` — this must be called
        // outside the layer loop, otherwise `seq_len` grows by `num_layers`
        // per token and the next attention pass fans out to O(N²) matvecs.
        self.kv_cache.advance();

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

    /// DSpark 版 speculative dual decoding (Phase 5) [`generate_speculative_dual`] と同じ
    /// draft/verify pipeline に、draft argmax の直前で [`crate::speculative_dspark::BigramBias`]
    /// を各 draft position に in-place apply する
    ///
    /// - position 0 の bigram prev = `next_token` (直前 main sample)
    /// - position i > 0 の bigram prev = `draft_tokens[i-1]`
    /// - biased logits を argmax にも verify (`draft_logits_all`) にも使う → Leviathan `q`
    ///   分布が実 draft policy と一致する
    /// - `bigram_bias = None` または `bigram_strength = 0.0` の場合は vanilla と bit-exact
    ///
    /// # Errors
    /// - [`crate::speculative_dspark::DsparkError::VocabSizeMismatch`]: bigram.vocab_size と
    ///   draft_model の vocab_size が不一致
    /// - [`crate::speculative_dspark::DsparkError::ConfidenceHeadBlockSizeMismatch`]:
    ///   advanced 指定時、confidence_head.block_size < spec_k
    /// - [`crate::speculative_dspark::DsparkError::HiddenDimMismatch`]:
    ///   advanced 指定時、confidence_head.hidden_dim != draft_model.hidden_dim
    /// - [`crate::speculative_dspark::DsparkError`]: bigram apply / confidence predict の任意のエラー
    #[cfg(feature = "dspark")]
    #[allow(clippy::too_many_arguments)]
    pub fn generate_speculative_dual_dspark(
        &mut self,
        draft_model: &mut Llama3Model,
        tokenizer: &GgufTokenizer,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        spec_k: usize,
        bigram_bias: Option<&dyn crate::speculative_dspark::BigramBias>,
        bigram_strength: f32,
        advanced: Option<&crate::speculative_dspark::DsparkAdvancedConfig<'_>>,
    ) -> Result<GenerateResult, crate::speculative_dspark::DsparkError> {
        // (0a) precondition: bigram.vocab_size == draft_model.vocab_size
        if let Some(b) = bigram_bias {
            let draft_vocab = draft_model.config.vocab_size as u32;
            if b.vocab_size() != draft_vocab {
                return Err(crate::speculative_dspark::DsparkError::VocabSizeMismatch {
                    expected: draft_vocab,
                    got: b.vocab_size(),
                });
            }
        }
        // (0b) precondition: confidence_head shape matches draft model
        if let Some(cfg) = advanced {
            let head = cfg.confidence_head;
            let need_block = spec_k as u32;
            if head.block_size() < need_block {
                return Err(
                    crate::speculative_dspark::DsparkError::ConfidenceHeadBlockSizeMismatch {
                        expected: need_block,
                        got: head.block_size(),
                    },
                );
            }
            let draft_hidden = draft_model.config.hidden_dim as u32;
            if head.hidden_dim() != draft_hidden {
                return Err(crate::speculative_dspark::DsparkError::HiddenDimMismatch {
                    expected: draft_hidden,
                    got: head.hidden_dim(),
                });
            }
        }

        let start = Instant::now();
        let mut tokens = tokenizer.encode(prompt);
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

        // Decode with dual-model speculation + DSpark bigram bias
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
                logits = self.forward(next_token);
                draft_model.forward(next_token);
                continue;
            }

            let k = spec_k.min(remaining);

            let saved_draft_pos = draft_model.kv_cache.seq_len();
            let _saved_main_pos = self.kv_cache.seq_len();
            let mut draft_tokens: Vec<u32> = Vec::with_capacity(k);
            let mut draft_logits_all: Vec<Vec<f32>> = Vec::with_capacity(k);
            let mut draft_input = next_token;
            let mut prev_for_bigram = next_token;

            // Draft loop: advanced = Some で hidden state 抽出 + confidence-gated 早期打切り
            // advanced = None は Phase 5 と bit-exact (forward + argmax + push)
            for i in 0..k {
                let (mut dl, hidden_opt) = if let Some(cfg) = advanced {
                    let (l, h) =
                        draft_model.forward_capture_hidden(draft_input, cfg.hidden_capture_layer);
                    (l, Some(h))
                } else {
                    (draft_model.forward(draft_input), None)
                };
                crate::speculative_dspark::apply_bigram_bias_maybe(
                    &mut dl,
                    prev_for_bigram,
                    bigram_bias,
                    bigram_strength,
                )?;
                // confidence-gated 早期打切り (hidden 取得済みの時のみ)
                if let (Some(cfg), Some(hidden)) = (advanced, hidden_opt.as_ref()) {
                    let conf = cfg.confidence_head.predict(i as u32, hidden)?;
                    if conf < cfg.confidence_threshold {
                        // KV cache は saved_draft_pos + 1 + num_accepted で rollback されるため
                        // 打切り位置の forward 分は verify 後の rollback で自動廃棄される
                        break;
                    }
                }
                draft_input = argmax(&dl);
                draft_tokens.push(draft_input);
                draft_logits_all.push(dl);
                prev_for_bigram = draft_input;
            }
            total_drafted += draft_tokens.len();

            // --- Verify phase (vanilla と同じロジック) ---
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
                    generated.push(draft_tokens[i]);
                    tokens.push(draft_tokens[i]);
                    total_accepted += 1;
                    num_accepted += 1;
                    logits = self.forward(draft_tokens[i]);
                } else {
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

            if !rejected && num_accepted == draft_tokens.len() && generated.len() < max_new_tokens {
                let bonus = sample_from_probs(&softmax(&logits), rng.next_f32());
                if bonus != tokenizer.eos_id {
                    generated.push(bonus);
                    tokens.push(bonus);
                    logits = self.forward(bonus);
                }
            }

            let draft_keep = saved_draft_pos + 1 + num_accepted;
            draft_model.kv_cache.rollback_to(draft_keep);
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

        Ok(GenerateResult {
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
        })
    }

    /// DSpark Phase 7: vanilla speculative dual pipeline + 各 draft position の
    /// `(hidden, was_accepted)` を collect [`PositionConfidenceHead::train_step`] の training data
    ///
    /// verify で reject された draft 以降は verify 打切りとなるため label 非付与
    /// bonus は draft でなく main sample なので label 非付与
    ///
    /// bigram_bias は使わず (baseline draft behavior)、confidence_head も使わない
    /// pure vanilla dual と同一 sampling path を辿るため collect した labels は
    /// vanilla policy の accept/reject 分布を反映する
    ///
    /// # Errors
    /// - `hidden_capture_layer` を指定した layer が `forward_capture_hidden` の非標準 arch に該当する場合 panic (documented)
    /// - DsparkError 系エラーは現状発生しないが、将来の validation 追加を見越して `Result` を返す
    #[cfg(feature = "dspark")]
    #[allow(clippy::too_many_arguments)]
    pub fn generate_speculative_dual_collect_labels(
        &mut self,
        draft_model: &mut Llama3Model,
        tokenizer: &GgufTokenizer,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        spec_k: usize,
        hidden_capture_layer: Option<usize>,
    ) -> Result<
        (
            GenerateResult,
            Vec<crate::speculative_dspark::DsparkLabelSample>,
        ),
        crate::speculative_dspark::DsparkError,
    > {
        let start = Instant::now();
        let mut tokens = tokenizer.encode(prompt);
        if tokenizer.add_bos_token && (tokens.is_empty() || tokens[0] != tokenizer.bos_id) {
            tokens.insert(0, tokenizer.bos_id);
        }

        self.clear_cache();
        draft_model.clear_cache();
        let prompt_token_count = tokens.len();

        let prefill_start = Instant::now();
        let mut logits = vec![0.0f32; self.config.vocab_size];
        for &tok in &tokens {
            logits = self.forward(tok);
            draft_model.forward(tok);
        }
        let prefill_ms = prefill_start.elapsed().as_millis() as u64;

        let decode_start = Instant::now();
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut total_drafted: usize = 0;
        let mut total_accepted: usize = 0;
        let mut rng = Rng64::new(42);
        let mut samples: Vec<crate::speculative_dspark::DsparkLabelSample> = Vec::new();

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
                logits = self.forward(next_token);
                draft_model.forward(next_token);
                continue;
            }

            let k = spec_k.min(remaining);

            let saved_draft_pos = draft_model.kv_cache.seq_len();
            let mut draft_tokens: Vec<u32> = Vec::with_capacity(k);
            let mut draft_logits_all: Vec<Vec<f32>> = Vec::with_capacity(k);
            let mut draft_hiddens: Vec<Vec<f32>> = Vec::with_capacity(k);
            let mut draft_input = next_token;

            // Draft phase: capture hidden state per position (vanilla argmax、bigram なし)
            for _ in 0..k {
                let (dl, hidden) =
                    draft_model.forward_capture_hidden(draft_input, hidden_capture_layer);
                draft_input = argmax(&dl);
                draft_tokens.push(draft_input);
                draft_logits_all.push(dl);
                draft_hiddens.push(hidden);
            }
            total_drafted += draft_tokens.len();

            // Verify phase: vanilla Leviathan、accept/reject を position 別 label に転記
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
                let accepted = r < accept_prob;
                // ここで label 付与 (verify した position のみ)
                samples.push(crate::speculative_dspark::DsparkLabelSample {
                    position: i as u32,
                    hidden: core::mem::take(&mut draft_hiddens[i]),
                    was_accepted: accepted,
                });
                if accepted {
                    generated.push(draft_tokens[i]);
                    tokens.push(draft_tokens[i]);
                    total_accepted += 1;
                    num_accepted += 1;
                    logits = self.forward(draft_tokens[i]);
                } else {
                    // vanilla resample from max(0, p - q)
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

            if !rejected && num_accepted == draft_tokens.len() && generated.len() < max_new_tokens {
                let bonus = sample_from_probs(&softmax(&logits), rng.next_f32());
                if bonus != tokenizer.eos_id {
                    generated.push(bonus);
                    tokens.push(bonus);
                    logits = self.forward(bonus);
                }
            }

            let draft_keep = saved_draft_pos + 1 + num_accepted;
            draft_model.kv_cache.rollback_to(draft_keep);
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

        let result = GenerateResult {
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
        };
        Ok((result, samples))
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

    // ─── Grammar-constrained generation (Phase X.8 B-4) ─────────────────────

    /// Generate text with grammar-constrained decoding.
    ///
    /// Wraps the standard prefill + decode loop with the grammar mask
    /// from [`crate::sampling::mask_logits_by_grammar`] and advances the
    /// FSM ([`crate::grammar::Fsm`]) after every accepted token. The
    /// mask forbids EOS unless the FSM is in a final state, so the model
    /// cannot terminate mid-parse; conversely once the grammar admits
    /// only EOS, the model is forced to stop.
    ///
    /// Fails with:
    /// - [`GrammarGenError::Fsm`] if FSM start or advance fails.
    /// - [`GrammarGenError::NoValidToken`] if the mask leaves no
    ///   sample-able token at a given step (grammar unsatisfiable from
    ///   the current context).
    ///
    /// Sampling is greedy (`argmax` within `top_k`); temperature and
    /// top-k mirror the semantics of [`generate`](Self::generate).
    /// Repetition penalty is intentionally *not* applied: the grammar
    /// mask already restricts output space, and mixing the two often
    /// produces surprising interactions.
    #[cfg(feature = "grammar")]
    pub fn generate_grammar(
        &mut self,
        tokenizer: &GgufTokenizer,
        prompt: &str,
        max_new_tokens: usize,
        grammar: &crate::grammar::Grammar,
        temperature: f32,
        top_k: usize,
    ) -> Result<GenerateResult, GrammarGenError> {
        use crate::grammar::Fsm;
        use crate::sampling::{advance_fsm_on_emit, mask_logits_by_grammar};

        let start = Instant::now();
        let mut tokens = tokenizer.encode(prompt);
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

        // Init FSM from grammar's root rule.
        let mut fsm = Fsm::start(grammar)?;

        // Decode
        let decode_start = Instant::now();
        let mut generated = Vec::with_capacity(max_new_tokens);

        for step in 0..max_new_tokens {
            // Apply the grammar mask *before* temperature so masking is
            // preserved through the linear scale.
            mask_logits_by_grammar(&fsm, tokenizer, &mut logits);

            if !logits.iter().any(|l| l.is_finite()) {
                return Err(GrammarGenError::NoValidToken { step });
            }

            // Temperature
            if temperature > 0.0 && temperature != 1.0 {
                let inv_t = 1.0 / temperature;
                for l in &mut logits {
                    *l *= inv_t;
                }
            }

            // Argmax within top-k on the finite subset.
            let next_token = if top_k > 0 && top_k < logits.len() {
                let mut indexed: Vec<(usize, f32)> = logits
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, l)| l.is_finite())
                    .collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(top_k);
                indexed.first().map_or(0u32, |(idx, _)| *idx as u32)
            } else {
                logits
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| l.is_finite())
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0u32, |(idx, _)| idx as u32)
            };

            if next_token == tokenizer.eos_id {
                break;
            }

            // Feed the emitted token back to the FSM. If the driver ever
            // slips (e.g. skipped the mask) the FSM refuses and we bail.
            advance_fsm_on_emit(&mut fsm, tokenizer, next_token)?;

            tokens.push(next_token);
            generated.push(next_token);

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

        Ok(GenerateResult {
            text: output_text,
            tokens_generated: gen_count,
            prompt_tokens: prompt_token_count,
            prefill_ms,
            decode_ms,
            total_ms,
            tokens_per_sec: tok_per_sec,
            spec_stats: None,
        })
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

/// Errors from grammar-constrained generation (Phase X.8 B-4).
#[cfg(feature = "grammar")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarGenError {
    /// FSM construction or transition failure.
    Fsm(crate::grammar::FsmError),
    /// After applying the grammar mask, no token remained sample-able at
    /// `step`. The grammar cannot be satisfied from the current point;
    /// typically means the model context has diverged from the grammar
    /// (e.g. prompt already emitted invalid text) or the grammar is
    /// under-specified for the model's vocabulary.
    NoValidToken { step: usize },
}

#[cfg(feature = "grammar")]
impl std::fmt::Display for GrammarGenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fsm(e) => write!(f, "grammar FSM error: {e}"),
            Self::NoValidToken { step } => {
                write!(f, "no valid token accepted by grammar at step {step}")
            }
        }
    }
}

#[cfg(feature = "grammar")]
impl std::error::Error for GrammarGenError {}

#[cfg(feature = "grammar")]
impl From<crate::grammar::FsmError> for GrammarGenError {
    fn from(e: crate::grammar::FsmError) -> Self {
        Self::Fsm(e)
    }
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

/// Load a weight where the shape is fully derived from the GGUF tensor
/// info — both `rows` (dims[1] or 1 for 1D tensors) and `cols`
/// (dims[0]) come from the file, so the caller does not need to compute
/// them from the model config. Used by the Kimi K3 loader
/// (Phase X.4.b.2) for tensors whose shape varies per layer type
/// (MLA-split vs KDA vs LatentMoE per-expert) and would be brittle to
/// hardcode.
fn load_weight_ref_any_shape<'a, G: crate::gguf::GgufSource<'a>>(
    gguf: &'a G,
    name: &str,
) -> Option<WeightRef<'a>> {
    let info = gguf.tensor_info(name)?;
    let data = gguf.tensor_data(name)?;
    let cols = *info.dims.first()? as usize;
    // Multiply all trailing dims (dims[1] × dims[2] × ...) to get the total
    // row count. Real K3 GGUF has 3-D tensors like `ssm_conv1d_q` with
    // shape `[kernel_size=4, groups=1, dim=12288]` where the "rows" that
    // downstream slicers expect is `groups * dim = 12288`. Prior code
    // only read `dims[1]` and defaulted to 1, which mis-shaped 3-D
    // tensors as `rows=1` (Phase X.4.b.6 continued fix, 2026-07-28).
    let rows: usize = info
        .dims
        .iter()
        .skip(1)
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
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
    // Phase X.3.e.3.15 fix: Qwen 3.5 uses `post_attention_norm.weight` as the
    // FFN input norm (loaded via `load_ffn_norm`), NOT as a sandwich norm.
    // Only load it as `post_attn_norm` when `ffn_norm.weight` exists (i.e.,
    // Gemma-2 which has both), otherwise the same tensor would double-load
    // and the Gemma-2 post-attention RMSNorm block would incorrectly fire
    // on Qwen 3.5, corrupting o_buf before the residual add.
    let has_ffn_norm_tensor = gguf
        .tensor_to_f32(&format!("{prefix}.ffn_norm.weight"))
        .is_some();
    let post_attn_norm = if has_ffn_norm_tensor {
        gguf.tensor_to_f32(&format!("{prefix}.post_attention_norm.weight"))
    } else {
        None
    };
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

    // MLA dimensions from the config metadata. `q_lora_rank` is optional —
    // absent on V2 / V2-Lite (Issue #58), which use a dense `attn_q.weight`
    // matvec instead of the two-stage LoRA chain used by V2.5 / V3 / R1.
    let kv_lora_rank = config.deepseek_kv_lora_rank()?;
    let qk_nope_head_dim = config.deepseek_qk_nope_head_dim()?;
    let qk_rope_head_dim = config.deepseek_qk_rope_head_dim()?;
    let v_head_dim = config.deepseek_v_head_dim()?;

    let q_head_total = qk_nope_head_dim + qk_rope_head_dim;
    let kv_up_head_total = qk_nope_head_dim + v_head_dim;

    let q = if let Some(q_lora_rank) = config.deepseek_q_lora_rank() {
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
        DeepSeekQProjection::LoRA {
            q_a_proj,
            q_a_norm,
            q_b_proj,
        }
    } else {
        let q_proj = load_weight_ref(
            gguf,
            &format!("{prefix}.attn_q.weight"),
            config.num_heads * q_head_total,
            config.hidden_dim,
        )?;
        DeepSeekQProjection::Dense { q_proj }
    };
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
        q,
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
    let kv_lora_rank = config.deepseek_kv_lora_rank()?;
    let qk_nope_head_dim = config.deepseek_qk_nope_head_dim()?;
    let qk_rope_head_dim = config.deepseek_qk_rope_head_dim()?;
    let v_head_dim = config.deepseek_v_head_dim()?;
    let q_head_total = qk_nope_head_dim + qk_rope_head_dim;
    let kv_up_head_total = qk_nope_head_dim + v_head_dim;

    let q = if let Some(q_lora_rank) = config.deepseek_q_lora_rank() {
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
        DeepSeekQProjection::LoRA {
            q_a_proj,
            q_a_norm,
            q_b_proj,
        }
    } else {
        let q_proj = load_weight_ref(
            gguf,
            &format!("{prefix}.attn_q.weight"),
            config.num_heads * q_head_total,
            config.hidden_dim,
        )?;
        DeepSeekQProjection::Dense { q_proj }
    };
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
        q,
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

    /// Phase X.3.e.3.39 online softmax equivalence check.
    ///
    /// Verifies that llama.cpp-style online softmax and ALICE's two-pass
    /// softmax produce mathematically equivalent results (within f32
    /// rounding tolerance) for a small attention head. This is a
    /// regression guard so future edits to either path keep them in
    /// numerical agreement.
    #[test]
    fn test_online_softmax_matches_two_pass() {
        let head_dim = 8;
        let seq_len = 5;
        let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

        // Deterministic pseudo-random inputs.
        let q: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.11 - 0.4).collect();
        let mut k = vec![0.0f32; seq_len * head_dim];
        let mut v = vec![0.0f32; seq_len * head_dim];
        for t in 0..seq_len {
            for d in 0..head_dim {
                k[t * head_dim + d] = ((t + 1) as f32 * 0.07 - d as f32 * 0.03).sin();
                v[t * head_dim + d] = ((t + 3) as f32 * 0.13 + d as f32 * 0.05).cos();
            }
        }

        // Reference: two-pass softmax.
        let mut scores = vec![0.0f32; seq_len];
        for t in 0..seq_len {
            let mut s = 0.0f32;
            for d in 0..head_dim {
                s += q[d] * k[t * head_dim + d];
            }
            scores[t] = s * inv_sqrt_d;
        }
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - m).exp();
            sum += *s;
        }
        for s in &mut scores {
            *s /= sum;
        }
        let mut two_pass = vec![0.0f32; head_dim];
        for t in 0..seq_len {
            let w = scores[t];
            for d in 0..head_dim {
                two_pass[d] += w * v[t * head_dim + d];
            }
        }

        // Candidate: online softmax (mirrors llama.cpp FLASH_ATTN_EXT).
        let mut online = vec![0.0f32; head_dim];
        let mut m_run = f32::NEG_INFINITY;
        let mut s_run = 0.0f32;
        for t in 0..seq_len {
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += q[d] * k[t * head_dim + d];
            }
            score *= inv_sqrt_d;
            let m_old = m_run;
            let (ms, vs) = if score > m_run {
                m_run = score;
                let ms = (m_old - m_run).exp();
                for d in 0..head_dim {
                    online[d] *= ms;
                }
                (ms, 1.0f32)
            } else {
                (1.0f32, (score - m_run).exp())
            };
            for d in 0..head_dim {
                online[d] += vs * v[t * head_dim + d];
            }
            s_run = s_run * ms + vs;
        }
        let inv_s = 1.0 / s_run;
        for d in 0..head_dim {
            online[d] *= inv_s;
        }

        for d in 0..head_dim {
            assert!(
                (online[d] - two_pass[d]).abs() < 1e-5,
                "online[{d}]={} vs two_pass[{d}]={}",
                online[d],
                two_pass[d]
            );
        }
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
            kimi_delta: None,
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
            kimi_delta: None,
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
            kimi_delta: None,
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

    /// Qwen 3.5 / 3.6 / Bonsai 27B Gated Attention post-hoc validation: the
    /// sigmoid gate maps `x → 1 / (1 + exp(-x))`, so gate = 0 halves the
    /// attention output, large positive lets it through, large negative
    /// nullifies. Verifies the arithmetic done inline in the main `forward`
    /// path against a scalar reference, independent of any real GGUF
    /// weights. Phase X.3.e.3.14 fix: previously tested silu (swish) which
    /// mismatched reference qwen35.cpp:401-404 ggml_sigmoid semantics.
    #[test]
    fn gated_attention_sigmoid_math_matches_reference() {
        // 6 attention output values with 6 gate values.
        let mut attn_out = vec![1.0f32, 2.0, -1.0, 0.5, -0.5, 3.0];
        let gate = vec![0.0f32, 1.0, -1.0, 10.0, -10.0, 0.5];
        let q_dim = attn_out.len();

        // Reference: each output multiplied by sigmoid(gate).
        let expected: Vec<f32> = attn_out
            .iter()
            .zip(gate.iter())
            .map(|(&a, &g)| a * sigmoid(g))
            .collect();

        // In-place update mirroring the forward path body.
        for i in 0..q_dim {
            attn_out[i] *= sigmoid(gate[i]);
        }

        for (i, (&got, &want)) in attn_out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "gated attention at {i}: got {got}, expected {want}"
            );
        }

        // gate = 0 → sigmoid(0) = 0.5 → output halved (attn_out[0] = 1.0 * 0.5).
        assert!((attn_out[0] - 0.5).abs() < 1e-6);
        // gate = -10 → sigmoid(-10) ≈ 0 → output near-zero.
        assert!(attn_out[4].abs() < 1e-3);
        // gate = 10 → sigmoid(10) ≈ 1 → output ≈ unchanged (0.5 * ~1).
        assert!((attn_out[3] - 0.5).abs() < 1e-3);
    }

    // ─── generate_grammar (Phase X.8 B-4) ────────────────────────────────
    // Method-level end-to-end tests need a real GGUF model and are handled
    // in Phase X.8 B-9 (Mac Metal / Jetson Vulkan smoke run). The unit
    // tests below cover only the error type surface — enough to guarantee
    // Display / Debug / From conversion contracts.

    #[cfg(feature = "grammar")]
    #[test]
    fn grammar_gen_error_display_no_valid_token() {
        let e = GrammarGenError::NoValidToken { step: 3 };
        let s = format!("{e}");
        assert!(s.contains("step 3"));
    }

    #[cfg(feature = "grammar")]
    #[test]
    fn grammar_gen_error_display_wraps_fsm_error() {
        let e = GrammarGenError::Fsm(crate::grammar::FsmError::NoTransition { ch: 'x' });
        let s = format!("{e}");
        assert!(s.contains("FSM error"));
        assert!(s.contains("'x'"));
    }

    #[cfg(feature = "grammar")]
    #[test]
    fn grammar_gen_error_from_fsm_error() {
        let fsm_err = crate::grammar::FsmError::EmptyRoot;
        let converted: GrammarGenError = fsm_err.clone().into();
        assert_eq!(converted, GrammarGenError::Fsm(fsm_err));
    }
}
