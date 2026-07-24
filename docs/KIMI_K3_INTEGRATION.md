# Kimi K3 / Kimi Delta Attention Integration Plan (Phase X.4)

**Status**: Skeleton only. `todo!()` fail-fast on `forward_kimi_k3` until
2026-07-27 open weight release + paper drop. Confirmed spec values from
2026-07-24 investigation (public sources) are marked ✅; remaining `TODO`
placeholders are derived from public gigazine coverage (2026-07-17).

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

## What we DON'T know yet (blocks implementation)

- Exact hidden dim, num_layers, num_heads (attn), num_kv_heads
- Kimi Delta specific: DeltaNet chunk size, recurrent state dim per head
- Hybrid attn : deltanet ratio (Bonsai 27B uses 16:48, unknown for K3)
- Shared always-active expert count (`n_shared_experts`)
- Per-expert FFN intermediate size (`moe_intermediate_size`)
- MoE gating scheme (sigmoid vs softmax, aux-loss vs `noaux_tc`)
- RoPE style (NEOX vs interleaved), theta, YARN scaling
- KV cache compression scheme (MLA-style compressed KV? per-position?)
- GGUF metadata prefix (guess: `"kimi"` / `"kimi3"` / `"kimideltatt"`)
- GGUF tensor naming (guess: `blk.N.attn_delta_*` or `blk.N.attn_kimi_*`)
- Multimodal input path (unified with text embed? separate encoder?)
- Attention Residuals mechanism (skip connection variant? paper drop needed)

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
| X.4.b | GGUF metadata detection + weight tensor mapping + config parity | 1-2 日 | 🚧 2026-07-27 open weight release + community GGUF conversion (mradermacher/bartowski) |
| X.4.c | CPU forward path (reuse Bonsai gated_deltanet ~90%, swap gating if paper differs) | 3-5 日 | X.4.b + paper drop |
| X.4.d | Attention Residuals (AttnRes) 実装 (skip connection の runtime scheme) | 3-5 日 | Kimi 論文 or reference impl |
| **X.4.e ⭐ 最高 ROI** | **Expert streaming from NVMe + LRU cache** (deepseek_streaming.rs を 896 experts 対応化) — enables Mac/Linux consumer targets at 0.5-2 tok/s given the 896/16 sparsity (1.79%) | 5-7 日 | X.4.c CPU baseline, DeepSeek V3 Issue #34 groundwork |
| **X.4.f (skeleton ✅ 2026-07-24)** | **MXFP4 CPU skeleton landed** (E2M1 table + E8M0 scale + block dequant + `MxfP4Row/Matrix` + correctness-first `mxfp4_matvec_fallback` routing + 11 unit tests、詳細は `docs/MXFP4_INTEGRATION_PLAN.md`) 残: 融合 scalar/SIMD matvec + PyTorch oracle 検証 | ✅ 1 日 (skeleton) / 🚧 2-3 日 (fused kernel + SIMD 残) | Weight release 2026-07-27 (fused kernel validation) |
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
