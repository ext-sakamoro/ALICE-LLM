# Kimi K3 / Kimi Delta Attention Integration Plan (Phase X.4)

**Status**: Skeleton only. `todo!()` fail-fast on `forward_kimi_k3` until
2026-07-27 open weight release + paper drop. All values below are `TODO`
placeholders derived from public gigazine coverage (2026-07-17).

## What we know (public, 2026-07-17)

| Item | Value | Source |
|---|---|---|
| Total params | ~2.8 T | Moonshot AI announcement |
| Context length | 1 M tokens | Announcement |
| Attention family | "Kimi Delta Attention" (Gated DeltaNet variant) | Announcement |
| Additional trick | "Attention Residuals" (~25% training speedup) | Announcement |
| Long-context decode | 6.3× faster vs baseline @ 1M ctx | Announcement |
| Modality | Native multimodal (text + vision + audio unspecified) | Announcement |
| Weight license | Open weights, TBD license | Announcement |
| Open weight release | 2026-07-27 (10 days from now) | Announcement |
| API input price | $3 / 1M tokens | Investing.com 2026-07-17 |
| API output price | $15 / 1M tokens | Investing.com 2026-07-17 |
| Pricing vs Claude Opus 4.8 | ~60% | Investing.com |
| Pricing vs GPT-5.6 Sol | ~50% | Investing.com |
| **MoE topology** | **896 total experts, top-16 active per token** | Investing.com |
| Active params per token | ~48 B (2.8T × 16/896) | Derived |
| Intelligence Index | 57 (Artificial Analysis) — matches Claude 3.5 Sonnet + o1 | Investing.com |
| Benchmark (Moonshot) | Beats GPT-5.6 Sol / Claude Fable 5 / Claude Opus 4.8 (Artificial Analysis) | Announcement |
| Market reaction | Tech + semiconductor stocks dropped on release day | Investing.com |

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

## Integration Phases

| Phase | Scope | Blocker |
|---|---|---|
| **X.4.a** (this PR, 2026-07-17) | Architecture enum variant + KimiDeltaConfig stub + fail-fast forward + docs | — |
| X.4.b | GGUF metadata detection + weight tensor mapping | 2026-07-27 open weight release + community GGUF conversion |
| X.4.c | CPU forward path (reuse Bonsai gated_deltanet, swap gating if paper differs) | X.4.b + paper drop |
| X.4.d | GPU forward path (WGSL shader adapt from Bonsai) | X.4.c numerical parity vs HF |
| **X.4.e (elevated)** | **Expert streaming from NVMe + LRU cache** — highest ROI given the 896/16 sparsity; enables Mac/Linux consumer targets at 0.5-2 tok/s | X.4.c CPU baseline, DeepSeek V3 Issue #34 groundwork |
| X.4.f | Quant policy (Q4 default, Q1_0 target for RunPod H100 8× at 350 GB, Q3 for H200 8×) | Reference HF forward for oracle |
| X.4.g | 1M context validation (RoPE YARN, hybrid attn windowing, KV compression) | X.4.d GPU throughput baseline |
| X.4.h | Multimodal input path (if text-first release is separate) | Kimi model card + modality spec |

## Weight-load hardware feasibility (updated 2026-07-17)

Key insight from the 896-expert / top-16-active topology (2026-07-17
Investing.com confirmation): **total weights are 1.4 TB Q4, but per-token
active weights are only ~24 GB Q4** (16 experts × ~48 B active / 896 = 48B
active × 0.5 bytes/param). That's Mac M3 Max tier — the constraint shifts
from "does it fit in RAM" to "can we stream the top-16 experts from disk
fast enough per token".

| Target hardware | Total weight (Q4) | Active/token (Q4) | Feasibility |
|---|---|---|---|
| **Mac M3 Max 128 GB + 2 TB NVMe** | 1.4 TB on disk | ~24 GB hot | **Viable if expert streaming works** — target 0.5-2 tok/s bounded by NVMe I/O |
| Mac M3 Max 128 GB (in-memory only) | — | — | Not viable (128 GB < 1.4 TB) |
| Jetson USB Orin 8 GB | — | — | Not viable (even active weights don't fit) |
| RunPod H100 80 GB × 8 (640 GB) | ~1.4 TB Q4 | — | Marginal: needs Q1_0 (~350 GB) or Q2_K (~700 GB) |
| RunPod H200 141 GB × 8 (1.13 TB) | ~1.4 TB Q4 | — | Viable at Q3, comfortable at Q4 with light offload |
| Paperspace A6000 48 GB | — | — | Not viable single-card |
| CPU-only w/ 2 TB NVMe (Linux 128 GB RAM) | ~1.4 TB stream | ~24 GB hot | Viable at 0.5-2 tok/s (same as Mac path, with more RAM headroom) |

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
