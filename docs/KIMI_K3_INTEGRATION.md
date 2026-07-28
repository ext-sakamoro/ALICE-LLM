# Kimi K3 / Kimi Delta Attention Integration Plan (Phase X.4)

**Status**: Skeleton + full HF-confirmed spec landed
(Phase X.4.a.1, 2026-07-28). `todo!()` fail-fast on `forward_kimi_k3`
remains, now blocking on community GGUF conversion (Phase X.4.b,
mradermacher / bartowski watch) rather than the initial weight release.
Open weights + `config.json` dropped on schedule 2026-07-27; the entire
`text_config` structure is now captured by [`KimiDeltaConfig`] and
parseable via `KimiDeltaConfig::from_hf_config` under the `hf-config`
Cargo feature. Confirmed spec values are marked ✅ in the tables below;
paper-only unknowns (KDA gate formula, AttnRes runtime scheme) remain.

**Strategic context**: This integration is not merely "supporting a new
model". It is the flagship test case of the ALICE-LLM
**Transformer Hybrid Hegemony Thesis** (see `docs/HEGEMONY_THESIS.md`).
Kimi K3 is Bonsai 27B's strategy scaled 10×: hybrid linear attention
(KDA) + sparse MoE (896/16) + quantization-aware training (MXFP4). If
ALICE-LLM runs Bonsai 27B on Jetson 8GB, running Kimi K3 on Mac M3 Max
consumer hardware is a logical necessity. See also
`~/.claude/projects/-Users-ys/memory/alice_llm_moe_phase_x4_kimi_k3_roadmap.md`
for the memory-side roadmap with sub-phase breakdown and user decision
points (A: scope, B: edge/cloud, C: MXFP4 GPU shader, D: start timing).

## What we know (public, 2026-07-17 + 2026-07-24 update)

| Item | Value | Source |
|---|---|---|
| Total params | ~2.8 T ✅ | Moonshot AI announcement |
| Context length | 1 M tokens ✅ | Announcement |
| Attention family | "Kimi Delta Attention" (Gated DeltaNet variant) ✅ | Announcement |
| Additional trick | "Attention Residuals" (~25% training speedup) ✅ | Announcement |
| Long-context decode | 6.3× faster vs baseline @ 1M ctx ✅ | Announcement |
| Modality | Native multimodal (text + vision, audio unconfirmed) ✅ | wan27.org 2026-07-24 |
| **Weight format (native)** | **MXFP4 weights + MXFP8 activations** ✅ | marktechpost / goml.io 2026-07 |
| **Weight file size (MXFP4 native)** | **~594 GB** ✅ | wan27.org 2026-07 |
| Weight file size (推定 Q4 GGUF community conversion) | ~1.4 TB | Derived from param count |
| Weight license | Open weights, TBD license (待ち) | Announcement |
| **Open weight release** | **2026-07-27 (3 days from 2026-07-24)** ✅ | Confirmed by Moonshot |
| **HuggingFace org** | **`huggingface.co/moonshotai`** (weight は 2026-07-27 appear 予定) ✅ | wan27.org 2026-07-24 |
| Training quant | MXFP4 quantization-aware training from SFT stage onward ✅ | goml.io |
| API input price | $3 / 1M tokens ✅ | Investing.com 2026-07-17 |
| API output price | $15 / 1M tokens ✅ | Investing.com |
| API cache-hit input | $0.30 / 1M tokens ✅ | Investing.com |
| Pricing vs Claude Opus 4.8 | ~60% | Investing.com |
| Pricing vs GPT-5.6 Sol | ~50% | Investing.com |
| **MoE topology** | **896 total experts, top-16 active per token** ✅ | Multiple sources confirmed |
| Active params per token | ~48-50 B (2.8T × 16/896) ✅ | Derived |
| **Active weights per token (Q4)** | **~24 GB** ✅ | 48B × 0.5 bytes/param, derived |
| Intelligence Index | 57 (Artificial Analysis) — matches Claude 3.5 Sonnet + o1 | Investing.com |
| Benchmark (Moonshot) | Beats GPT-5.6 Sol / Claude Fable 5 / Claude Opus 4.8 (Artificial Analysis) | Announcement |
| Frontend Code Arena | 1679 pt で首位 (Claude Fable 5 抜き) ✅ | Moonshot 公式 |
| 独立 benchmark 順位 | 4 位 (Claude Fable 5 + GPT-5.6 Sol の下、Claude Opus 4.8 の上) ✅ | Tom's Hardware |
| Market reaction | Tech + semiconductor stocks dropped on release day | Investing.com |
| **GPU 逼迫** | **公開 48h で新規 subscription 停止** ✅ | ITmedia 2026-07-21 |

## Confirmed via HF config.json (2026-07-28 update)

The Kimi K3 `config.json` at
`huggingface.co/moonshotai/Kimi-K3/raw/main/config.json` resolves all
of the previously-blocking numeric unknowns. Values below are captured
in [`KimiDeltaConfig`] fields with matching names and parseable via
`KimiDeltaConfig::from_hf_config`.

| Item | Value (config.json) | HF field path |
|---|---|---|
| `hidden_size` | 7168 ✅ | `text_config.hidden_size` |
| `num_hidden_layers` | 93 ✅ | `text_config.num_hidden_layers` |
| `num_attention_heads` | 96 ✅ | `text_config.num_attention_heads` |
| `num_key_value_heads` | 96 (no GQA) ✅ | `text_config.num_key_value_heads` |
| Dense-FFN `intermediate_size` | 33792 ✅ | `text_config.intermediate_size` |
| **Hybrid layer routing** | 24 MLA (`[4,8,...,92,93]`) + 69 KDA + 1 dense ✅ | `text_config.linear_attn_config.{full_attn_layers,kda_layers}` |
| KDA head_dim / num_heads / conv_k | 128 / 96 / 4 ✅ | `text_config.linear_attn_config.*` |
| KDA gate `use_full_rank_gate` | `true` ✅ | `linear_attn_config.use_full_rank_gate` |
| KDA gate `gate_lower_bound` | `-5.0` ✅ | `linear_attn_config.gate_lower_bound` |
| Gated MLA `q_lora_rank` | 1536 ✅ | `text_config.q_lora_rank` |
| Gated MLA `kv_lora_rank` | 512 ✅ | `text_config.kv_lora_rank` |
| MLA `qk_nope_head_dim` | 128 ✅ | `text_config.qk_nope_head_dim` |
| MLA `qk_rope_head_dim` | 64 (partial RoPE) ✅ | `text_config.qk_rope_head_dim` |
| MLA `v_head_dim` | 128 ✅ | `text_config.v_head_dim` |
| MLA `mla_use_nope` | `true` ✅ | `text_config.mla_use_nope` |
| MLA `mla_use_output_gate` | `true` (Kimi-unique) ✅ | `text_config.mla_use_output_gate` |
| AttnRes `attn_res_block_size` | 12 ✅ | `text_config.attn_res_block_size` |
| Activation | SiTU-GLU (β1=4.0, β2=25.0) ✅ | `text_config.hidden_act`, `.activation_situ_*` |
| MoE `num_experts` | **896** ✅ | `text_config.num_experts` |
| MoE `num_experts_per_token` | **16** ✅ | `text_config.num_experts_per_token` |
| MoE `num_shared_experts` | **2** ✅ | `text_config.num_shared_experts` |
| MoE `moe_intermediate_size` | 3072 ✅ | `text_config.moe_intermediate_size` |
| MoE `first_k_dense_replace` | 1 (layer 0 dense) ✅ | `text_config.first_k_dense_replace` |
| MoE `routed_expert_hidden_size` | 3584 (Latent MoE) ✅ | `text_config.routed_expert_hidden_size` |
| MoE `routed_scaling_factor` | 1.0 (V3 = 2.5) ✅ | `text_config.routed_scaling_factor` |
| MoE router activation / topk | sigmoid / `noaux_tc` ✅ | `text_config.moe_router_activation_func`, `.topk_method` |
| MoE `num_nextn_predict_layers` | **0 (no MTP)** ✅ | `text_config.num_nextn_predict_layers` |
| MXFP4 group size / bits | 32 / 4 ✅ | `text_config.quantization_config.config_groups.group_0.weights.*` |
| Max position embeddings | 1,048,576 ✅ | `text_config.max_position_embeddings` |
| Vocab size | 163,840 (bos 163584, eos 163586, pad 163839) ✅ | `text_config.vocab_size` |
| Vision encoder | MoonViT-V2, 27 layers, hidden 1024, 12 heads, patch 14 ✅ | `vision_config.*` |
| Tech report | `github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf` | — |

## Confirmed via tech report (paper 実読 2026-07-28)

The Kimi K3 tech report (`github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf`,
47 pages) resolves the remaining paper-drop dependencies. Sections
2.1.1 (KDA), 2.1.2 (Gated MLA), 2.2 (AttnRes), 2.3 (Stable LatentMoE),
4.1.4 (Deployment-Aware Post-Training), and 5.4 (Inference & Online
Serving) were the load-bearing sections for the ALICE-LLM integration.

### KDA gate function (§2.1.1, Eq 1-6)

**Recurrence (Eq 1):**

```text
S_t = (I − β_t · k_t k_tᵀ) · Diag(α_t) · S_{t−1} + β_t · k_t v_tᵀ
ō_t = Sᵀ_t · q_t                                        ∈ ℝ^{d_v}
```

- `S_t ∈ ℝ^{d_k × d_v}` fixed-size recurrent state per head
  (K3: d_k = d_v = 128, so 64 KB / head; sequence-length invariant)
- `α_t ∈ (0,1)^{d_k}` channel-wise retention (per Eq 5 below)
- `β_t ∈ (0,1)` scalar delta-rule write strength

**Per-head parameterization (Eq 2):**

```text
q_t, k_t = L2Norm(Swish(ShortConv(W_{q/k} x_t)))       ∈ ℝ^{d_k}
v_t      = Swish(ShortConv(W_v x_t))                   ∈ ℝ^{d_v}
β_t      = Sigmoid(W_β x_t)                            ∈ (0,1)
z_t      = W_α^↑ W_α^↓ x_t + b_α                      ∈ ℝ^{d_k}   (low-rank pre-gate)
```

- ShortConv kernel size = 4 (config `short_conv_kernel_size`)
- L2Norm on q, k (Kimi Linear inheritance)

**Lower-bounded decay (Eq 5) — key departure from Kimi Linear:**

```text
g_t = g_min · Sigmoid(exp(A_h) · z_t)                  ∈ (g_min, 0)^{d_k}
α_t = exp(g_t)                                         ∈ (e^{g_min}, 1)^{d_k}
```

- `A_h` learnable per-head log-scale (init 0)
- `g_min = -5.0` fixed (config `gate_lower_bound`)
- Cumulative log-decay over a 16-token tile ∈ (-80, 0) → fits BF16
  dynamic range → both diagonal and off-diagonal tiles use dense
  Tensor Core matmul (Kimi Linear's unbounded negative-Softplus
  mapping required explicit position-pair computations on the
  diagonal tile)

**Full-rank output gate (Eq 6) — key departure from Kimi Linear:**

```text
y_t = W_o · [Sigmoid(W_g x_t) ⊙ RMSNorm(ō_t)]
```

- `W_g` is full-rank (Kimi Linear used low-rank)
- Head-wise RMSNorm on recurrent output before gating

Landed as `KimiDeltaState` + `kimi_delta_step` + `kimi_delta_read` +
`kimi_delta_lower_bounded_decay` + `kimi_delta_output_gate` in
`src/llama3.rs` at Phase X.4.c.1 (2026-07-28, commit 55df1c6), with
14 unit tests covering the recurrence math on hand-computed inputs.

### Gated MLA (§2.1.2, Eq 7)

- DeepSeek V2 MLA base (low-rank latent `c_t = W_c x_t` compresses KV
  cache)
- **NoPE (No Positional Encoding) on all MLA layers** — no RoPE on
  queries or keys. KDA layers provide the position-sensitive mixing,
  MLA layers provide content-only global attention. Consequence: no
  RoPE base retune / YARN needed at 1M context extension.
- **Input-dependent full-rank output gate (Eq 7)**:

  ```text
  y_t = W_o · [Sigmoid(W_g x_t) ⊙ ō_t]
  ```

  Same gate pattern as KDA (Eq 6) but WITHOUT the inner RMSNorm on
  `ō_t`. `kimi_delta_output_gate(rms_weight=None, ...)` covers this
  Gated MLA variant.
- Attention output kept in FP32 during training (corrects flash
  attention's biased rounding error).

### Block AttnRes (§2.2, Eq 8-10)

Layers `L = 93` partitioned into `N ≈ 8` blocks of `S = L/N ≈ 12`
layers each (`attn_res_block_size = 12`; the final block is a partial
9-layer block, giving 9 total representations when counting the
embedding layer as `b_0`).

**Within-block reduction:**

```text
b_n^i = Σ_{j ∈ B_n, j ≤ i} f_j(h_j)                    (partial sum over first i layers of block n)
b_0    = h_1                                            (token embedding always a source)
```

**Across-block attention:**

```text
V = { [b_0, b_1, ..., b_{n-1}]                        if i = 1 (first layer of block n)
    { [b_0, b_1, ..., b_{n-1}, b_n^{i-1}]             if i ≥ 2 (subsequent layers)
q_l = w_l ∈ ℝ^d                                        (learnable per-layer pseudo-query)
k_i = v_i (same tensor for both roles)
φ(q, k) = exp(qᵀ RMSNorm(k))                          (softmax kernel with RMSNorm on keys)
α_{i→l} = φ(q_l, k_i) / Σ φ(q_l, k_j)
h_l = Σ α_{i→l} · v_i
```

Memory / communication overhead: `O(Ld) → O(Nd)` (reduces by ~12× at
K3 scale). RMSNorm on keys prevents large-magnitude layer outputs
from dominating the attention weights.

#### llama.cpp reference wiring (pwilkin PR #26185, 2026-07-28 実読)

**File**: `src/models/kimi-k3.cpp` (645 行、SHA `2043a6a8...`)
**Key files**: `constants.py` (§675-677, §1271-1273), `llama-hparams.h`,
`llama-graph.cpp` (`ggml_dsv4_hc_pre` reuse from DeepSeek V4)

**tensor names (GGUF)**:

- `blk.{N}.attn_res_score` — **shape `[n_embd]` (1D vector)**、per-layer
- `blk.{N}.ffn_res_score`  — **shape `[n_embd]` (1D vector)**、per-layer
- `output_res_score`       — **shape `[n_embd]` (1D vector)**、model-level

いずれも `# Kimi K3 (fused res_norm * res_proj, ...)` の comment 付き:
paper §2.2 の `RMSNorm(k)` (norm) + learnable score projection の 2
tensor を **単一 1D vector に fusion** した実装 (paper では別々の
matrix + norm gain だが GGUF export ではまとめて "score" weight として
export される)

**per-layer wiring (`src/models/kimi-k3.cpp` L300-353 verbatim)**:

```cpp
for (int il = 0; il < n_layer; ++il) {
    const auto & layer = model.layers[il];

    // `prefix_sum` is the residual stream. On checkpoint layers it is banked
    // into res_stack and restarts from the attention output alone.
    ggml_tensor * prefix_sum = inpL;

    cur = use_attn_res ? res_mix(prefix_sum, layer.attn_res_score, n_embd, n_tokens, il)
                       : prefix_sum;

    bool banked = false;
    if (use_attn_res && (uint32_t) il % res_bs == 0) {
        res_push(prefix_sum, n_embd, n_tokens);  // banks the RAW layer input, not `cur`
        banked = true;
    }

    cur = build_norm(cur, layer.attn_norm, NULL, LLM_NORM_RMS, il);
    cb(cur, "attn_norm", il);

    if (hparams.is_recr(il)) {
        cur = build_kda_layer(cur, layer, inp_rs, ...);
    } else {
        cur = build_mla_layer(cur, layer, inp_attn_k, inp_attn_kv, ...);
    }

    prefix_sum = banked ? cur : ggml_add(ctx0, prefix_sum, cur);

    cur = use_attn_res ? res_mix(prefix_sum, layer.ffn_res_score, n_embd, n_tokens, il)
                       : prefix_sum;

    cur = build_norm(cur, layer.ffn_norm, NULL, LLM_NORM_RMS, il);

    if ((uint32_t) il < hparams.n_layer_dense_lead) {
        // dense SiTU-GLU FFN
        ggml_tensor * g = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        ggml_tensor * u = ggml_mul_mat(ctx0, layer.ffn_up,   cur);
        cur = kimi_k3_situ(ctx0, g, u, hparams.situ_beta, hparams.situ_linear_beta);
        cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);
    } else {
        cur = build_latent_moe(cur, layer, n_embd_latent, il);
    }

    prefix_sum = ggml_add(ctx0, prefix_sum, cur);
    inpL = prefix_sum;
}

cur = inpL;

// final mix, then narrow to the output tokens
if (use_attn_res) {
    cur = res_mix(cur, model.output_res_score, n_embd, n_tokens, -1);
}
```

**`res_mix` (L218-257 verbatim, weighted convex combination)**:

```cpp
ggml_tensor * llama_model_kimi_k3::graph::res_mix(
    ggml_tensor * cur, ggml_tensor * score_w,
    int64_t n_embd, int64_t n_tokens, int il)
{
    const int n_ckpt = (int) ckpts.size();
    if (n_ckpt == 0) {
        return cur; // layer 0: nothing banked yet
    }
    const float eps = hparams.f_norm_rms_eps;
    ggml_tensor * src = res_stack(n_embd, n_tokens);   // [n_embd, n_ckpt, n_tokens]

    // Scores for banked ckpts (rms_norm * score_w, then sum_rows over n_embd)
    ggml_tensor * sc_src = ggml_rms_norm(ctx0, src, eps);
    sc_src = ggml_mul(ctx0, sc_src, score_w);
    sc_src = ggml_sum_rows(ctx0, sc_src);              // [1, n_ckpt, n_tokens]
    sc_src = ggml_reshape_2d(ctx0, sc_src, n_ckpt, n_tokens);

    // Current residual stream scored separately (kept out of stack)
    ggml_tensor * sc_cur = ggml_rms_norm(ctx0, cur, eps);
    sc_cur = ggml_mul(ctx0, sc_cur, score_w);
    sc_cur = ggml_sum_rows(ctx0, sc_cur);              // [1, n_tokens]

    ggml_tensor * scores = ggml_concat(ctx0, sc_src, sc_cur, 0);  // [n_ckpt+1, n_tokens]
    ggml_tensor * probs  = ggml_soft_max(ctx0, scores);           // softmax over ne0

    // Split convex combination: hc_pre reduces over ne1 for the stacked part,
    // plain broadcast-multiply handles the current stream
    ggml_tensor * p_src = ggml_cont(ctx0, ggml_view_2d(ctx0, probs, n_ckpt, n_tokens, ...));
    ggml_tensor * p_cur = ggml_cont(ctx0, ggml_view_2d(ctx0, probs, 1, n_tokens, ...));

    ggml_tensor * out = ggml_dsv4_hc_pre(ctx0, src, p_src);       // weighted sum over ne1
    out = ggml_add(ctx0, out, ggml_mul(ctx0, cur, p_cur));

    return out;
}
```

**重要な semantic 要点** (paper §2.2 との差分):

1. **AttnRes は per-layer で 2 回、model 末尾で 1 回、計 `2L+1` 回呼ぶ**
   - 各 layer で `attn_norm` の前 (`attn_res_score` 使用) と `ffn_norm`
     の前 (`ffn_res_score` 使用) に `res_mix` を挟む
   - `output_norm` の前に `output_res_score` で最終 mix
2. **checkpoint layer (`il % res_bs == 0`) で banking + prefix_sum リセット**
   - 銀行に入れるのは **RAW `prefix_sum` (= 前 layer 末の inpL)**、
     `res_mix` 適用後の `cur` ではない
   - Banked layer では attention 後の `prefix_sum` が **attn 出力単独**に
     reset (直前 residual を足さない、bank に既に保存されているため)
3. **`res_mix` は「stacked ckpts + current stream」の softmax 加重和**
   - scores = `sum_rows(RMSNorm(x) * score_w)` per bank + per current
     = paper Eq 9 の `φ(q_l, k_i) = exp(q_l^T RMSNorm(k_i))` を 1D 化
     (learnable pseudo-query が score_w vector 1 本に fusion)
   - **weighted sum は non-normalized (raw) 値**を使う (paper §2.2 と同じ、
     norm は score 計算用のみ)
4. **`ggml_dsv4_hc_pre`** = DeepSeek V4 由来の融合 kernel
   - `out[d, t] = Σ_c p_src[c, t] * src[d, c, t]` を 1 pass fused (SIMD)
   - Rust 版で相当なもの: 明示的 loop `for c in 0..n_ckpt { for d in 0..n_embd { out[d] += p_src[c] * src[d, c] } }`
5. **`hparams.attn_res_block_size = 0` で AttnRes 全体 disable**
   - K3 config は `attn_res_block_size = 12`、`hparams.n_layer = 93`
     → checkpoint layer は `il = 0, 12, 24, 36, 48, 60, 72, 84` (計 8 個)
     + embedding = 9 total sources (最初の checkpoint は raw embedding)

**GGUF metadata keys** (PR #26185 で追加):

| key | type | comment |
|---|---|---|
| `kimi-k3.attn_res_block_size` | u32 | `S = 12` (K3 default) |
| `kimi-k3.expert_latent_length` | u32 | `moe_intermediate_size / 2` (latent MoE hidden) |
| `kimi-k3.activation.situ_beta` | f32 | SiTU-GLU β_1 (default 4.0) |
| `kimi-k3.activation.situ_linear_beta` | f32 | SiTU-GLU β_2 (default 25.0) |

**ALICE-LLM 実装 (X.4.c.3.4) への reflection**:

- 現状 `KimiK3LayerWeights::attn_res_norm/attn_res_proj/ffn_res_norm/ffn_res_proj`
  の 4 tensor を想定していたが、pwilkin PR reference では **`attn_res_score`
  + `ffn_res_score` の 2 tensor だけ、両方とも 1D vector [n_embd]** に fusion
- 実装 map の更新: `KimiK3LayerWeights` の 4 field → 2 field に集約
  (`attn_res_score` + `ffn_res_score`、いずれも `Vec<f32>` or `WeightRef` [n_embd])
- `KimiK3ModelWeights::output_res_score` (1D [n_embd]) を追加
- `BlockAttnResState` は `ckpts: Vec<Vec<f32>>` (banked streams) + `stack_cache`
  の維持責務、`res_mix(cur, score_w, ...) -> Vec<f32>` primitive を提供
- 現在の `block_attnres_softmax_attention` primitive は概念的に paper §2.2 に忠実
  だが、pwilkin PR の 1D-fused 実装に合わせて `score_w: &[f32]` を受ける
  simplified 版に refactor 予定 (X.4.c.3.4.a)
- 既存 primitive の per-layer pseudo-query `w_l ∈ ℝ^d` は、GGUF の
  `attn_res_score / ffn_res_score` として import する (import 側で 1D として
  読む、fusion 前提)

**まだ未実装 (X.4.c.3.4 sub-task)**:

- X.4.c.3.4.a: `BlockAttnResState::res_mix` primitive を pwilkin 版に合わせて
  refactor (1D score_w + non-normalized raw weighted sum)
- X.4.c.3.4.b: `KimiK3LayerWeights` を 4 tensor → 2 tensor に集約 +
  `load_kimi_k3_layer_weights` の tensor 名 update
- X.4.c.3.4.c: `KimiK3ModelWeights` に `output_res_score` 追加 +
  `load_kimi_k3_model_weights` update
- X.4.c.3.4.d: `KimiK3Model::forward` の layer loop に per-layer `res_mix`
  を 2 回挿入 (attn_norm 前 + ffn_norm 前) + checkpoint banking logic
- X.4.c.3.4.e: 最終 output mix (`output_norm` 前) 実装
- X.4.c.3.4.f: unit test — `res_mix` primitive の scoring / weighted-sum
  correctness (RMSNorm on keys で normalize、weighted sum で raw を使う 2 段の
  duality 検証)

### Stable LatentMoE (§2.3, Eq 11)

**Forward:**

```text
u = Σ_{i ∈ T_k(x)} p_i · E_i^routed(W^↓ x)             ∈ ℝ^ℓ          (routed in latent space)
y = Σ_{j=1}^{N_s} E_j^shared(x) + W^↑ RMSNorm(u)       ∈ ℝ^d          (shared in full width)
```

- `d = 7168` hidden dim, `ℓ = 3584` latent dim (`routed_expert_hidden_size`)
- `E_i^routed: ℝ^ℓ → ℝ^ℓ` per-expert FFN (`moe_intermediate_size = 3072`)
- `E_j^shared: ℝ^d → ℝ^d` shared experts (`N_s = 2`)
- **RMSNorm inserted between routed aggregation `u` and up-projection
  `W^↑`** (K3-specific stabilization; DeepSeek MoE lacks this norm)

### SiTU-GLU activation (§2.3.2, Eq 12)

```text
SiTU-GLU(x) = [β_1 tanh(W_g x / β_1) ⊙ Sigmoid(W_g x)] ⊙ [β_2 tanh(W_u x / β_2)]
```

- `β_1 = 4` (gate softcap), `β_2 = 25` (up softcap)
- Bounded: `|f(x)| ≤ β_1 · β_2 = 100` — avoids activation outliers
  that SwiGLU (unbounded) can produce, matters especially in MXFP8
  activation-quantization scale.
- Near-origin: `softcap(x, β) = β · tanh(x / β)` approximates the
  Swish (SiLU) response of SwiGLU.

### Quantile Balancing (§2.3.3, Eq 13-14)

Auxiliary-loss-free routing (`topk_method = "noaux_tc"`, same family
as DeepSeek V3). Per-expert bias `b_j` added to router scores for
Top-k selection but omitted from `p_{i,j}` normalization. Bias update
uses histogram estimation to scale to 896 experts × millions of
margins per training batch.

### Deployment-aware MXFP4 scope (§4.1.4)

- **MoE expert weights**: MXFP4 (group 32, 4-bit, per-tensor + E8M0
  scale)
- **MoE activations**: MXFP8
- **Non-expert modules**: BF16 / FP32 (attention projections, latent
  MoE `W^↓`/`W^↑`, shared experts, MoE router, `lm_head`, vision
  tower, MM projector — the config.json `quantization_config.ignore`
  regex list captures the exclusion set)
- QAT applied throughout post-training (SFT + RL + rollout), so
  train/rollout/inference share the exact same quantization scheme
  and there is no train-inference mismatch.

### Pretrain-time MTP head (§4.1.4)

- `num_nextn_predict_layers = 0` in `config.json` reflects the
  **inference-time** config only.
- Pretrain includes 1 MTP layer that mirrors a backbone block, later
  fine-tuned into an EAGLE-3 style draft model (7-step unroll, LK
  loss `L_LK = -log Σ min(p(x), q(x))`).
- Not required for the initial ALICE-LLM Phase X.4.c CPU forward path
  but relevant for a future X.4.k speculative-decode integration.

### KDA-aware prefix cache (§5.4.1)

- 6144-token physical block = 12 × 512-token hash blocks
- KDA recurrent state is fixed size (128 × 128 f32 per head) but its
  serial dependence forces checkpointing at coarse boundaries
  (1024-6144 tokens per KDA layer)
- MLA per-token entries + KDA per-block checkpoints packed into the
  same unified paged pool. Same page byte size for both types, KDA
  state stored contiguously per head so byte streams are
  self-contained.
- Prefix caching reusable at any 512-token boundary regardless of
  request length / chunking / scheduling interleaving.

Relevant to Phase X.4.h (1M context validation) and X.4.e (streaming
pool + KV cache interaction).

## What we still DON'T know (post-paper-read, 2026-07-28)

The tech report resolves every load-bearing unknown for Phase X.4.c
(CPU forward). Two smaller items remain, none blocking the CPU
forward path itself:

- **Multimodal fusion timing** — MoonViT-V2 encoder is described in
  §2.4 (27-layer transformer, RMSNorm, no bias) and pixel-shuffle
  down-sampling is captured, but the exact runtime insertion contract
  for how image tokens are spliced into the text stream (before the
  first backbone block? per-image position ID assignment? re-entry
  during agent execution?) still needs the model card or an early
  reference implementation. Not blocking Phase X.4.c/d/e (text-only
  path). Scheduled at Phase X.4.i.
- **GGUF metadata prefix + tensor naming** — depends entirely on
  which prefix mradermacher / bartowski / llama.cpp settle on when
  `convert_hf_to_gguf.py` lands for `model_type = "kimi_k3"`. Guess
  `"kimi"` still stands. This is the sole external-dependency
  blocker for Phase X.4.b.

## Existing ALICE-LLM code that can be reused (~80-95%)

Kimi Delta is a Gated DeltaNet family, which ALICE-LLM already ships:

| Reusable component | Location | Reuse level |
|---|---|---|
| Gated DeltaNet CPU forward | `src/llama3.rs:gated_deltanet_step*` (llama3.rs:3400+) | ~90% (Kimi Delta likely swaps gating fn) |
| Bonsai hybrid layer routing | `src/llama3.rs:layer_kind_map` (Qwen 3.5 pattern) | ~95% (only ratio changes) |
| SSM DeltaNet config extractor | `SsmDeltaNetConfig` (llama3.rs:192) | ~70% (add Kimi-specific fields) |
| MoE routing | `src/llama3.rs:moe_forward*` | ~80% (Kimi K3 MoE topology TBD) |
| Q1_0-Q8_0 quant + SwiGLU shader | `src/shaders/*`, `src/gguf.rs` | 100% (quant-agnostic) |
| 4-bit KV cache path | `src/kv_cache.rs` (in-progress) | 100% |
| Hybrid CPU+GPU per-layer | `--hybrid` flag on `qwen_gpu` / Phase A2 | 100% (works for any hybrid family) |
| Prompt template dispatcher | `elyza_gguf.rs:make_chat_template` | +1 case (`"kimi"` template TBD) |

## Integration Phases (2026-07-24 update — 10 sub-phases + Phase X.11 追加)

| Phase | Scope | 工数 | Blocker |
|---|---|---|---|
| **X.4.a** ✅ (2026-07-17) | Architecture enum variant + KimiDeltaConfig stub + fail-fast forward + docs | 完了 | — |
| **X.4.a.1** ✅ (2026-07-28) | Post-release spec refinement: KimiDeltaConfig expanded from 10 → 32 fields with all HF-confirmed values + `KimiDeltaConfig::from_hf_config` JSON loader (`hf-config` feature) + 5 unit tests + doc sync | 完了 | — |
| X.4.b | GGUF metadata detection + weight tensor mapping + config parity | 1-2 日 | 🚧 (weight loader 残)。metadata 側は X.4.b.1 で完了 |
| **X.4.b.1** ✅ (2026-07-28) | Kimi K3 GGUF metadata loader (`ModelArch::KimiK3::meta_prefix() = "kimi-k3"` + `KimiDeltaConfig::from_gguf(gguf, prefix)` で 30+ field 読取 + `Llama3Config::from_gguf` KimiK3 branch + `is_mla_layer(il)` predicate + 6 test with synthetic mini K3 GGUF fixture)。community conversion (`Kuberwastaken/Kimi-K3-GGUF/convert_kimi_k3.py` + upstream `pwilkin/kimi-k3-text` PR #26185) の layout に準拠。GrEarl/Kimi-K3-GGUF (Q2_K 94-part) + GrEarl/Kimi-K3-GGUF-IQ1_S (527 GB 94-part) が実 download target | 完了 | — |
| X.4.b.2 | 実 GGUF tensor loader (per-layer tensor lookup + shape validation for ~2573 tensor、`TENSOR_MAP.md` に準拠) | 1-2 日 | forward path 実装と同時期 |
| **X.4.b.2** ✅ (2026-07-28) | Kimi K3 tensor reference structs + walker: `KimiK3ModelWeights` + `KimiK3LayerWeights` + `enum KimiK3Attention {Mla, Kda}` + `enum KimiK3Ffn {Dense, LatentMoe}` + `KimiK3MlaAttn` / `KimiK3KdaAttn` / `KimiK3LatentMoe` sub-structs (35 field 総計) + `load_kimi_k3_layer_weights` + `load_kimi_k3_model_weights` + `load_weight_ref_any_shape` helper + 6 loader test | 完了 | — |
| X.4.c | CPU forward path (reuse Bonsai gated_deltanet ~90%, swap gating if paper differs) | 3-5 日 | X.4.b + paper drop |
| **X.4.c.1** ✅ (2026-07-28) | KDA CPU forward primitives scaffold: `KimiDeltaState` + `kimi_delta_step` (Eq 1) + `kimi_delta_read` + `kimi_delta_lower_bounded_decay` (Eq 5) + `kimi_delta_output_gate` (Eq 6/7) + 14 unit tests | 完了 | — |
| **X.4.c.2** ✅ (2026-07-28) | KDA per-head composite forward: `KimiDeltaHeadCache` + `KimiDeltaHeadParams` + `kimi_delta_l2_norm_in_place` + `kimi_delta_forward_head` (Eq 2 + ShortConv + Swish + L2Norm + step + read + output gate、reuse 既存 `causal_conv1d_step` / `silu`) + 10 unit tests | 完了 | — |
| X.4.c.3 | Block-level integration into `forward_kimi_k3`: per-layer weight lookup + Block AttnRes wiring + Gated MLA layer forward + KV cache | 2-3 日 | X.4.b + X.4.d + Bonsai ShortConv weight loader lift |
| **X.4.c.3.1** ✅ (2026-07-28) | `KimiK3Model` struct + `forward` skeleton (embedding + output projection real、per-layer dispatch loop with `todo!()` inside) + 4 test | 完了 | — |
| **X.4.c.3.2** ✅ (2026-07-28) | Gated MLA layer forward primitive (`kimi_k3_gated_mla_step` + `KimiK3MlaCache` MLA 42× 圧縮 + NoPE + Eq 7 output gate) + 5 test | 完了 | — |
| **X.4.c.3.3.a** ✅ (2026-07-28) | MLA + Dense FFN wiring into `forward_kimi_k3` (`kimi_k3_extract_mla_config` helper + `kimi_k3_dense_ffn_forward` SwiGLU + forward MLA branch real delegate + Dense FFN branch real、KDA/LatentMoE 依然 todo) + 5 test | 完了 | — |
| X.4.c.3.3.b | KDA per-head aggregation into forward (fused `attn_q/k/v` per-head slicing + `kimi_delta_forward_head` × num_heads + aggregate、F32 first、Q4_K per-head slicing 別途) | 1-2 日 | — |
| X.4.c.3.3.c | Stable LatentMoE forward into forward (router top-16 + shared expert + latent W↓/W↑ + SiTU-GLU) | 2-3 日 | — |
| X.4.c.3.4 | **Block AttnRes wiring (pwilkin PR #26185 精読済 2026-07-28、上 §Block AttnRes llama.cpp reference wiring 節に semantics 集約)**: X.4.c.3.4.a `res_mix` primitive 1D fused refactor + X.4.c.3.4.b `KimiK3LayerWeights` 4 tensor → 2 tensor (`attn_res_score` + `ffn_res_score`、いずれも 1D [n_embd]) + X.4.c.3.4.c `KimiK3ModelWeights::output_res_score` 追加 + X.4.c.3.4.d layer loop 2 回挿入 + banking + X.4.c.3.4.e 最終 output mix + X.4.c.3.4.f unit test | 2-3 日 | — |
| X.4.d | Attention Residuals (AttnRes) 実装 (skip connection の runtime scheme) | 3-5 日 | Kimi 論文 or reference impl |
| **X.4.d.1** ✅ (2026-07-28) | Block AttnRes runtime primitives: `BlockAttnResState` + `block_attnres_softmax_attention` (Eq 9 RMSNorm-on-keys softmax + log-sum-exp stable) + `block_attnres_layer_step` (Eq 10 per-layer update with pre-partial snapshot semantics) + 10 unit tests. 最終 N-block aggregation into logits は paper 詳細不足のため X.4.d.2 (forward_kimi_k3 integration 時) に defer | 完了 | — |
| **X.4.e.1** ✅ (2026-07-28) | Pool infrastructure K3 対応 — `deepseek_streaming.rs` の module doc / `StreamingExpertPool` doc / `PersistenceHeuristic` doc を K3 (896 experts, top-16, Stable LatentMoE) 対応化 + `kimi_k3_active_bytes` / `recommended_budget_bytes` sizing helper 追加 + 4 unit test (K3 24GB paper estimate 検証 / 896-index top-16 dispatch / out-of-range boundary at 896 / persistence heuristic scaling) 追加。pool の infrastructure 側は既に n_experts agnostic だったため実質的な struct 変更なしで K3 topology を受け付ける | 完了 (半日) | — |
| **X.4.e ⭐ 最高 ROI** | **Expert streaming from NVMe + LRU cache 実運用** (X.4.e.1 の pool infra を実 K3 GGUF に接続、`forward_deepseek_moe_layer` を K3 の LatentMoE + RMSNorm 挿入 + SiTU-GLU に拡張) — enables Mac/Linux consumer targets at 0.5-2 tok/s given the 896/16 sparsity (1.79%) | 5-7 日 | X.4.c CPU baseline (KDA forward), X.4.b GGUF |
| **X.4.f (skeleton ✅ 2026-07-24)** | **MXFP4 CPU skeleton landed** (E2M1 table + E8M0 scale + block dequant + `MxfP4Row/Matrix` + correctness-first `mxfp4_matvec_fallback` routing + 11 unit tests、詳細は `docs/MXFP4_INTEGRATION_PLAN.md`) 残: 融合 scalar/SIMD matvec + PyTorch oracle 検証 | ✅ 1 日 (skeleton) / 🚧 2-3 日 (fused kernel + SIMD 残) | Weight release 2026-07-27 (fused kernel validation) |
| **X.4.f.1** ✅ (2026-07-28) | MXFP4 fused scalar matvec kernel: `mxfp4_matvec_fused_scalar` (stack-resident `[f32; 32]` dequant + f32 dot 融合、per-matvec scratch なし、fallback と bit-exact parity) + `mxfp4_matvec()` free fn 実装 (todo! 置換) + `quantized_matvec` routing 切替 + 5 新 parity test | 完了 | — |
| X.4.f.2 | MXFP4 SIMD variants: NEON (aarch64) + AVX2 / AVX-512 (x86_64)、Q1_0/Q2_0 と同 pattern (per-block sum precompute) で `mxfp4_matvec_fused_scalar` に対する bit-exact parity 検証 | 3-5 日 | — |
| X.4.f.3 | MXFP4 PyTorch `microxcaling` oracle validation: 実 K3 GGUF block の byte-exact dequant + full-tensor stats 比較 | 2-3 日 | X.4.b 実 GGUF |
| **X.4.g** | **MXFP4/MXFP8 GPU shader** (Metal + wgpu WGSL、新規 quant format の GPU 実装) | 7-10 日 | X.4.f CPU parity |
| X.4.h | 1M context validation (RoPE YARN, hybrid attn windowing, KV compression) | 5-7 日 | X.4.d GPU throughput baseline |
| X.4.i | Multimodal input path (text-first release と分離ならば) | 3-5 日 | Kimi model card + modality spec |
| X.4.j | E2E integration test + benchmark parity (MMLU/HumanEval/GSM8K subset vs Moonshot API) | 3-5 日 | X.4.h 完了 |

**合計工数**: ~35-53 日 (単独作業、blocker なし前提 = 4-6 週)、edge target only なら X.4.d + X.4.g + X.4.i を skip 可能で ~20-30 日 (3-4 週)

### Phase X.11 (MoE 汎化、+4-5 週)

Kimi K3 実装で得た知見を横展開し 7 系統 MoE を共通 loader + forward で扱えるようにする ([[alice_llm_moe_phase_x4_kimi_k3_roadmap]] §Phase X.11 参照)

| Phase | Scope | 工数 |
|---|---|---|
| X.11.1 | MoE loader 一般化 (7 系統共通 config extractor + tensor naming dispatcher) | 3-5 日 |
| X.11.2 | `expert_gating_func` variants (sigmoid / softmax / noaux_tc) 統合 | 2-3 日 |
| X.11.3 | Shared expert pattern (DeepSeek V2/V3 / Gemma 4 26B_A4B) 完成 | 3-5 日 |
| X.11.4 | Hy3 対応 (FP8 quant + MTP head 3.8B + 192/top-8 routing) | 5-7 日 |
| X.11.5 | LongCat-2 対応 (LSA 3 改善実装、cloud backend 側) | 7-10 日 |
| X.11.6 | Mixtral / Gemma 4 26B_A4B forward test + numerical parity | 3-5 日 |
| X.11.7 | E2E benchmark 7 系統横断 + roadmap ドキュメント化 | 3-5 日 |

**Phase X.4 + X.11 合計 8-10 週で 7 系統 MoE 統合完成**、ALICE-LLM が MoE 系 open weight model の実質参照実装地位確立

## Weight-load hardware feasibility (updated 2026-07-17)

Key insight from the 896-expert / top-16-active topology (2026-07-17
Investing.com confirmation + 2026-07-24 MXFP4 native size update):
**total weights are 594 GB in MXFP4 native (or 1.4 TB in Q4 GGUF community
conversion), but per-token active weights are only ~24 GB Q4**
(16 experts × ~48 B active / 896 = 48B active × 0.5 bytes/param).
That's Mac M3 Max tier — the constraint shifts from "does it fit in RAM"
to "can we stream the top-16 experts from disk fast enough per token".

| Target hardware | Total weight | Active/token (Q4) | Feasibility |
|---|---|---|---|
| **Mac M3 Ultra 512 GB + 2 TB NVMe** | 594 GB MXFP4 native | ~24 GB hot | ✅ **Full in-memory 可能** (NVMe streaming 不要、0.5-2 tok/s 目標) |
| **Mac M3 Max 128 GB + 2 TB NVMe** | 594 GB on disk | ~24 GB hot | ✅ **Viable if expert streaming works** — target 0.5-2 tok/s bounded by NVMe I/O |
| Mac M3 Max 128 GB (in-memory only) | — | — | ❌ (128 GB < 594 GB) |
| Jetson USB Orin 8 GB | — | — | ❌ Not viable (even active weights don't fit, 24 GB > 8 GB unified) |
| **RunPod H100 80 GB × 8 (640 GB)** | ~594 GB MXFP4 | — | ✅ **Full in-memory 可能** (MXFP4 native なら 640 > 594) |
| RunPod H200 141 GB × 8 (1.13 TB) | ~594 GB MXFP4 | — | ✅ 余裕 |
| Paperspace A6000 48 GB | — | — | ❌ Not viable single-card |
| CPU-only w/ 2 TB NVMe (Linux 128 GB RAM) | 594 GB stream | ~24 GB hot | ✅ Viable at 0.5-2 tok/s (Mac path と同等、RAM headroom あり) |

**Strategic implication**: MXFP4 native support (Phase X.4.f/g) is not
just a nice-to-have — it is the enabling factor for the H100 8× cluster
target. Without MXFP4 support, ALICE-LLM must rely on community Q4 GGUF
conversion (~1.4 TB) which fits only in H200 8× (marginal on H100 8×).
With MXFP4 support, H100 8× becomes a comfortable target and consumer
Mac hardware runs Kimi K3 with expert streaming = the "consumer hardware
runs 2.8T MoE" reversal that validates the [[HEGEMONY_THESIS]].

Practical near-term target on consumer hardware: **expert-streaming mode**
adapting the DeepSeek V3 Phase 4 (Issue #34) design. Each forward:

1. Router selects top-16 experts (from 896)
2. Fetch those 16 expert weight blocks from NVMe (~24 GB Q4, streamed
   in ~2-4 s on Gen4 NVMe at ~7 GB/s)
3. Attention (Kimi Delta hybrid) uses always-hot weights in RAM
4. FFN dispatches to the 16 streamed experts, accumulate

The 896/16 sparsity ratio (1.79%) is much sparser than DeepSeek V3
(256/8 = 3.13%) or Bonsai (64/6 = 9.38%), which means expert
locality/reuse across the prompt is high — a well-designed LRU cache
of hot experts in RAM (say 64 experts × 1.5 GB = 96 GB) should cover
>90% of forward passes without disk hits.

**Bonsai `--hybrid` CPU+GPU per-layer path (Phase A2) generalizes
directly**: DeltaNet layers on CPU (recurrent state fits in RAM),
full-attention layers on GPU, MoE FFN routed to whichever tier holds
the selected experts.

## Test plan (post-release)

1. **Config parity**: dump `KimiDeltaConfig` from GGUF, compare to public
   `config.json` (should exist in Moonshot release)
2. **Weight load**: verify all `blk.N.*` tensors resolved, no orphan tensors,
   dequant Q8_0 first-N rows byte-match HF safetensors (mirror Phase X.3.e.3.30
   V2-Lite methodology)
3. **CPU forward**: run 1-token forward on `"The capital of Japan is"`,
   dump per-op tensor l2 (attn_norm, q, kv_a, k_pe, attn_out, ffn_out) at
   layer 0, compare vs HF Mac mainlined `transformers >= 5.5` oracle
4. **Argmax match**: full-27-layer top-1 must match HF Mac oracle
5. **Long-context probe**: 32K, 128K, 512K, 1M context stress test with
   fixed hash prompt (validates RoPE + KV compression + hybrid attn window)
6. **Benchmark parity**: subset of MMLU / HumanEval / GSM8K comparing
   ALICE Q4 tok-argmax vs Moonshot API

## References (add when available)

- Paper: TBD (expected 2026-07-27 or after)
- HuggingFace: `moonshot-ai/Kimi-K3-Base` / `Kimi-K3-Instruct` (TBD)
- GGUF community: `mradermacher/Kimi-K3-*-gguf` / `bartowski/Kimi-K3-*-gguf` (TBD)
- llama.cpp support: PR TBD
- Reference implementation: TBD

## Related ALICE-LLM work

- [[docs/DEEPSEEK_V2_LITE_VALIDATION.md]] — validation methodology template
- [[docs/BONSAI_GPU_SUPPORT.md]] — hybrid DeltaNet reference
- [[docs/PHASE_X_3_E_3_3_VALIDATION.md]] — GPU numerical parity approach
- `src/llama3.rs:gated_deltanet_step*` — DeltaNet CPU forward
- `SsmDeltaNetConfig` (`src/llama3.rs:192`) — extend for Kimi-specific fields
- `alice-tracker.toml` — new project entry `kimi-k3-integration` needed

## Fail-fast checkpoint

Until 2026-07-27:
- `ModelArch::from_gguf` returns `KimiK3` if `general.architecture` starts with
  `"kimi"` (guessed prefix)
- `Llama3Model::forward` dispatches `KimiK3` → `forward_kimi_k3(...)` →
  `todo!("KIMI-K3 forward: waiting for open weight release 2026-07-27 — see docs/KIMI_K3_INTEGRATION.md")`
- No silent Ok, no default values, no placeholder logits

**Reason**: "仮実装完了偽装の禁止" (CLAUDE.md) — never let a stub silently
succeed. If a user tries to load a Kimi K3 GGUF before X.4.b/c lands, they
get an explicit panic pointing to this doc, not garbage output.
