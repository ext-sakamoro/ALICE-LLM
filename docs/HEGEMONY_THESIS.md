# ALICE-LLM Hegemony Thesis: The Transformer Hybrid Position

**Status**: Ratified 2026-07-24 as the north star for Phase X.4 (Kimi K3)
+ Phase X.11 (MoE generalization) and all subsequent architecture
decisions.

**Rationale author**: user (2026-07-24 conversation)
**Editorial capture**: Claude Code + ALICE-LLM maintainer
**Related memory**: `~/.claude/projects/-Users-ys/memory/alice_llm_transformer_hybrid_hegemony_thesis.md`
**Related docs**: `docs/KIMI_K3_INTEGRATION.md`, `docs/MXFP4_INTEGRATION_PLAN.md`

## Context

On 2026-07-24, Kimi K3 (2.8 trillion parameters, 896-expert MoE, MXFP4
weights, 1M context, Kimi Delta Attention) went live and, within 48
hours, its inference cluster hit compute saturation. Moonshot AI
suspended new subscriptions. Similar patterns had already played out
for GPT-5.6 Sol and Claude Fable 5.

The mainstream reading is: "these models are too big; more GPUs would
fix it." The load-bearing framing behind this ADR is different:

> **Kimi K3 didn't fall over because it was too big. It fell over
> because the Transformer architecture it inherited from GPT-4 has
> three structural defects that put it on a direct collision course
> with physics.**

## The three structural defects of vanilla Transformers

### Defect 1 — Self-attention's O(N²) curse

`softmax(QK^T / √d)` is an N×N dense matrix. Doubling context length
quadruples compute and memory. Reaching 1M tokens is an exponential
tax on hardware.

Industry palliatives (sliding-window / sparse attention / FlashAttention
/ RoPE scaling) reduce constants but not the exponent. Pure-Transformer
lineage cannot economically serve 1M context on consumer hardware.

### Defect 2 — KV cache bloat and bandwidth starvation

Autoregressive generation re-reads the entire KV cache on every token.
Compute units (ALUs) idle while memory bandwidth saturates. This is
the Memory Wall (Wulf & McKee 1995) hitting inference directly.

Concrete evidence from ALICE-Train: on M1 Pro the theoretical bandwidth
is 200 GB/s, but Ternary QAT could only extract 45% of it. Reaching a
usable throughput required a hand-crafted TLB miss elimination + huge
page + cache-line-aware layout hack (see
`memory/alice-train-tuning-playbook.md`). If ALICE-Train's dense compute
kernel already lives at the bandwidth ceiling, Transformer inference —
whose only workload *is* KV sweeps — is pinned to it.

Palliatives (GQA / MLA / 4-bit KV / PagedAttention) shift the constant,
not the wall.

### Defect 3 — No state compression

RNNs and humans compress the past into a fixed-size state vector.
Transformers require random access to every past token, spreading "the
entire transcript on the table and re-reading it from scratch every
generation step". This is expensive and, in a real sense, unintelligent.

Palliatives (RWKV / Mamba / RetNet) restore fixed state but historically
lag Transformers on local reasoning benchmarks.

## Why hybrid DeltaNet + Attention is the antidote

Defect 1 and Defect 3 are the same problem — "keeping the entire past
dense" — expressed at different layers of the stack. Solving them jointly
requires compressing the past into a fixed state (which kills O(N²) and
restores state compression simultaneously) while preserving enough dense
attention to keep local reasoning intact.

This is exactly what Bonsai 27B (16 full-attention + 48 gated-DeltaNet
layers) and Kimi K3's KDA family do:

| Transformer defect | Hybrid solution |
|---|---|
| O(N²) self-attention | 75% of layers become O(N) linear-attn DeltaNet; overall complexity flattens |
| KV memory wall | DeltaNet layers hold no KV cache — footprint drops to 25% before further 4-bit compression |
| No state compression | DeltaNet's fixed-size recurrent state *is* the compressed past |

Combined with 4-bit KV cache, effective KV footprint drops to ~12.5% of
a pure Transformer. That is the mechanical reason **Bonsai 27B fits and
runs on a Jetson USB Orin 8 GB** — an outcome that pure-Transformer
lineage cannot reach on any consumer device today.

## ALICE-LLM's position

ALICE-LLM has, over Phase X.3.e.3 (Qwen 3.5-4B / Bonsai 27B GPU debug)
and Phase A2 (`--hybrid` per-layer CPU+GPU), assembled the concrete
infrastructure needed to exploit this architectural gap:

- Gated DeltaNet CPU forward (`llama3.rs:gated_deltanet_step*`) + Metal
  shader (`shaders/gated_deltanet.wgsl`)
- Hybrid layer routing (`layer_kind_map` — 16:48 for Bonsai, 8:24 for
  Qwen 3.5, extensible to KDA once weights land)
- `--hybrid` CPU+GPU dispatch (`qwen_gpu --hybrid`, commit `b5d08d8`),
  giving Jetson USB Orin 8 GB a 3.3× speedup on Bonsai 27B vs pure CPU
- 4-bit KV cache scaffolding (`src/kv_cache.rs`)
- Quantization matrix (Q1_0 – Q8_0 + BitNet Ternary + IQ4_XS),
  quant-agnostic across shaders
- DeepSeek V3 routed-expert LRU streaming (`src/deepseek_streaming.rs`,
  971 lines), directly reusable as the 896-expert extension for Kimi K3
- MoE inference base (`src/llama3.rs:forward_moe_layer`, ~130 lines) —
  Qwen3 MoE 4×0.6B Q4_K_M at 32.9 tok/s on Mac M2 (2026-07-09
  `e395eb4`)

The measured "physical reversal" (small hardware running large models
that industry-consensus says cannot fit) is the empirical validation:

| Model | Hardware | Baseline expectation | ALICE-LLM measured |
|---|---|---|---|
| Bonsai 27B Q1_0 | Jetson USB Orin 8 GB | OOM | **0.3 tok/s** (via `--hybrid`) |
| Bonsai 27B Q1_0 | Mac M3 CPU | 0.05–0.1 tok/s | **0.2 tok/s** |
| Bonsai 27B Q1_0 | Mac M3 Metal | 0.3–0.5 tok/s | **1.1 tok/s** |
| Qwen 3.5-4B Q4_K_M | Mac M3 Metal | — | 2.9 tok/s |
| Ornith-1.0-9B (Qwen 3.5 f-tune) | Mac / Jetson zero-config | — | 1.8 / 2.1 / 2.3 tok/s |

These are already the "reversal" outcomes the thesis claims. Kimi K3
extends them by an order of magnitude.

## Why Kimi K3 is the flagship test case

Kimi K3's architectural choices are exactly the Bonsai hybrid strategy
scaled 10× (2.8T total / 896 experts / top-16 / KDA / 1M context /
MXFP4). If ALICE-LLM can run Bonsai 27B on Jetson 8 GB via its hybrid
+ streaming stack, running Kimi K3 on Mac M3 Max consumer hardware is
a logical necessity — not a wish. The 896/16 sparsity (1.79%) means
per-token active weights are ~24 GB Q4, which is Mac M3 Max tier
memory territory with NVMe expert streaming picking up the tail.

Successfully landing Phase X.4 shifts the industry framing:

| Before Phase X.4 | After Phase X.4 completion |
|---|---|
| Kimi K3 only accessible via Moonshot API ($3/$15 per M tokens) | Kimi K3 runs on Mac M3 Max 128 GB with expert streaming (free, private) |
| Kimi K3 requires 64+ accelerator supernodes | Full MXFP4 fits on H100 8× (640 GB > 594 GB native) |
| "2.8T MoE is only for hyperscalers" | Individual developers can inspect, benchmark, and integrate it |
| MoE support fragmented across Qwen / Mixtral / Gemma / DeepSeek / Kimi / Hy3 / LongCat forks | ALICE-LLM provides one loader + one forward for all seven |

## Decision principles derived from this thesis

Every design decision in Phase X.4 and beyond must be tested against
these four criteria before it is committed.

### A. Feature-fit test
"Which of the three Transformer defects does this feature close?" If
the answer is none, the feature is deprioritized regardless of demand.

### B. Consumer-hardware invariant
"Does Jetson 8 GB / Mac M3 Max consumer hardware still work after this
change?" If not, the change is rolled back. The invariant that
'consumer hardware runs frontier models' is not an accident — it is
the thesis.

### C. Upstream-wait vs self-build
"Would self-building this create an ALICE-LLM specific reversal, or is
it pure catch-up to llama.cpp / vLLM?" Reversals justify self-building;
pure catch-up waits for upstream.

### D. MoE integration order
Kimi K3 first (highest-difficulty test case: 2.8T / 896 experts /
MXFP4), then Hy3 (295B / 192 experts / FP8), then LongCat-2 (1.6T /
LSA / 1M context). Reason: Kimi K3 is the hardest case; if it lands,
the remaining six MoE families become trivial derivatives.

## Scope of this ADR

- **In scope**: Strategic positioning of Phase X.4, X.11, and downstream
  MoE work. Design principles A–D above.
- **Out of scope**: Concrete API design, numerical validation
  methodology (see `docs/KIMI_K3_INTEGRATION.md` and
  `docs/PHASE_X_3_E_3_3_VALIDATION.md`), MXFP4 format details (see
  `docs/MXFP4_INTEGRATION_PLAN.md`).

## Revisiting

This thesis is durable but not immortal. It should be revisited when:

- A pure Transformer architecture demonstrates linear-time attention +
  fixed-state compression at scale without a hybrid (currently unknown)
- Consumer hardware bandwidth exceeds current DRAM by 10× (would
  reduce the pressure to close Defect 2)
- The regulatory or economic landscape makes cloud inference
  categorically cheaper than edge (unlikely in the current geopolitical
  climate)

Absent one of those triggers, ALICE-LLM will continue investing in
hybrid + sparse-MoE + quantization-aware inference as its structural
bet.
