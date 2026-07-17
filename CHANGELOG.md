# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Aggregated work since `Cargo.toml` version was set to `1.0.0`. Grouped
by Phase; the "Phase X.Y.Z" references map back to the roadmap in
`memory/alice_llm_future_work.md` and the journey entries in
`memory/alice_llm_phase_x3e3_journey.md`.

### Added

- **Phase X.4.a — Kimi K3 / Kimi Delta Attention skeleton** (e9f8586).
  `ModelArch::KimiK3` variant, `KimiDeltaConfig` struct (9-field
  `Option<>` skeleton reflecting the confirmed 896-expert / top-16-active
  MoE topology and $3/$15 API pricing from the 2026-07-17 Moonshot AI
  announcement), `Llama3Model::forward_kimi_k3` fail-fast stub with
  `todo!()`, GGUF `"kimi"` prefix detection, dispatch wiring. The
  actual forward path waits on the 2026-07-27 open-weight release.
- **Phase X.3.e.3.29 — `attention_only_load` flag** (d479e6a).
  Enables real Phase A2 hybrid on Jetson (Qwen 3.5-4B) by skipping
  DeltaNet weight upload to GPU when the CPU handles those layers.
- **Phase X.3.e.3.28 — `--hybrid-per-layer`** (9d644fb).
  CPU DeltaNet + GPU Attention concurrent execution.
- **Phase X.3.e.3.27 — Q1_0 fused SwiGLU shader** (966abac).
  Bonsai 27B first coherent GPU generation.
- **Phase X.3.e.3.22-3.26 — Bonsai / Qwen 3.5 GPU coherent generation
  fixes**: DeltaNet scratch `k_buf/v_buf` sizing (d728552),
  `attn_q` Bonsai gated attention per-head interleaved layout
  (ddc6603), DeltaNet `conv1d` `ring_pos` per-token update + `reset()`
  zero-init (12ab0ae), BOS token prepend + `attn_out_normed` field
  + prompt template sync (29f6db4). Cumulative: 6 CPU + 6 GPU fixes
  landing Bonsai 27B on Mac Metal (1.1 tok/s) and A6000 (6.9 tok/s)
  and Qwen 3.5-4B on Jetson USB Orin 8GB via hybrid mode (0.3 tok/s
  = 3.3× speedup).
- **V2-Lite Q8_0 validation methodology** (e9f8586). Prompt token IDs
  dump in `examples/elyza_gguf.rs` and `kv_a_full` split diagnostic
  dumps (`first64` / `tail64` / `head512`) in `forward_deepseek_v3`,
  env-gated via `ALICE_DEEPSEEK_DUMP=1`.
- **Issue #58 — V2 / V2-Lite dense Q path** (4ababd9). Supports models
  where the Q projection is dense rather than the V3 LoRA
  `q_a_proj` → `q_a_layernorm` → `q_b_proj` chain.
- **`docs/KIMI_K3_INTEGRATION.md`** (e9f8586). Phase X.4.a-h roadmap,
  hardware feasibility (Mac 128 GB + 2 TB NVMe expert-streaming at
  0.5-2 tok/s marked viable given the 896/16 topology), existing
  Bonsai/DeltaNet reuse map, post-release test plan.
- **`docs/DEEPSEEK_V2_LITE_VALIDATION.md`** (d041a5e + successors).
  Full HF-vs-ALICE per-op diff methodology, per-position table,
  root-cause investigation trail.

### Fixed

- **Phase X.3.e.3.29 layer_bgs regression** (d57f566). Removed the
  double `push` in the DeltaNet arm that shifted indexing for
  Attention layers and caused a runtime panic on Bonsai 27B loading.
- **Issue #58 (part 2) — `kv_cache.advance()` location** (8b10f26).
  Moved outside the per-layer loop; the previous placement grew
  `seq_len` by 27× per token and made V2-Lite generation appear hung.
- **Issue #58 (part 3) — `deepseek2` chat template** (8b10f26).
  Added `"User: {prompt}\n\nAssistant:"`; the Llama-3 fallback
  template produced ~30 junk tokens on DeepSeek's tokenizer.
- **`cargo fmt` CI failure** (6877ce0). Long inline `dump_tensor()`
  call exceeded `max_width=100`.

### Documentation

- **Phase X.3.e.3.30 § Root cause 決着** (e9f8586, in
  `docs/DEEPSEEK_V2_LITE_VALIDATION.md`). Q4_K dequant bug hypothesis
  refuted: Q8_0 GGUF weight byte-matches HF `safetensors` (row-by-row
  mean diff ~0.0001 = Q8 noise, no permutation). Mac mainlined
  `transformers` 5.3.0 + ALICE's real token IDs forced input forward
  yields `k_pe` L2 = 16.63 vs ALICE Q8 16.61 (Q8 noise), and full
  27-layer top-1 = `' The'` (id 429) for both engines — argmax match.
  Paperspace HF (transformers 4.42 + `trust_remote_code`) exposed as
  buggy oracle. Canonical oracle switched to Mac mainlined. V2-Lite
  forward path numerically validated.
- **README EN/JP updates** (0887e97, 0fea2b3, 1275bde, f7c421f).
  Positioning as research + engineering project, Q1_0 = 1.125 bpw
  binary correction (was "Ternary"), 5 missing features added to
  JP (DeltaNet CPU forward, x86_64 SIMD, per-layer hybrid,
  God-object-free config, multi-arch model list), Phase X.3.e.3.22-3.29
  achievements reflected.
- **Issue #36 partial closure** (d041a5e, 9ef2152, d604418, 7ae53bb).
  V2-Lite oracle validation methodology, per-op layer-0 dump, Q5_K_M
  cross-check, `kv_a_mqa` element-wise divergence investigation
  chain (later resolved by Phase X.3.e.3.30 root-cause work above).

### Notes

- `Cargo.toml` version stays at `1.0.0`. No git tag has been cut for
  this window; releases will start being tagged from the next
  semantic version bump.
- The `todo!()` in `forward_kimi_k3` is intentional per CLAUDE.md's
  "仮実装完了偽装の禁止" rule — no silent Ok on unimplemented paths,
  and users get an explicit panic pointing to the integration doc if
  they somehow feed a Kimi K3 GGUF before Phase X.4.b lands.
- Two open stubs are tracked in ALICE-CodeTracker (ID `019f6f7f`):
  `src/llama3.rs:149` (Kimi K3 TodoComment) and `src/llama3.rs:5547`
  (Kimi K3 Todo Warning).

## [1.0.0]

Starting point for changelog tracking. The pre-1.0 history is
recoverable via `git log` (~50 commits leading up to this marker) and
covers the Phase X foundation work: hybrid attention infrastructure,
DeltaNet CPU/GPU implementations, multi-arch (Llama / Mistral /
Gemma-2 / Gemma-3n / Gemma-4 / Qwen-2 / Qwen-3 / Qwen-3.5 / DeepSeek-V3)
support scaffolding, GGUF loading, Q1_0-Q8_0 quantization, wgpu-based
GPU backend, PEFT / LoRA adapters, integration with ALICE-CodeTracker,
and the initial round of Phase X.3.e.3 (Phase 5-20) numerical parity
work.

[Unreleased]: https://github.com/ext-sakamoro/ALICE-LLM/compare/main...HEAD
