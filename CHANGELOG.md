# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Grammar-constrained decoding (Phase X.8, B-1 → B-4, B-8, B-9-C)**
  behind the new `grammar` feature. Downstream can constrain sampling
  to any GBNF (LOL DSL, JSON schema, tool-call payloads) with a
  single feature flag; no new dependencies are pulled in.
  - `alice_llm::grammar` module:
    - `parse_gbnf(&str) -> Result<Grammar, GbnfError>` — hand-written
      llama.cpp-compatible GBNF subset parser (terminal + char class
      including negation + rule ref + group + `* + ?` quantifiers +
      `#` comments; unsupported syntax `. / {n,m} / lookahead` fails
      loud).
    - `Fsm { start, advance, accepts, accepts_str, is_final, is_dead,
      allowed_chars, with_max_depth }` — NFA over parse positions,
      forks eagerly through rule refs / groups / quantifiers,
      recursion capped by `DEFAULT_MAX_DEPTH = 256` so left-recursion
      without a base case surfaces as `FsmError::RecursionOverflow`.
    - `CharSet` for allowed-char introspection.
  - `alice_llm::sampling` module:
    - `GrammarTokenizer` trait (blanket impl for `GgufTokenizer`).
    - `mask_logits_by_grammar(&fsm, tokenizer, &mut logits)` — sets
      `-inf` on tokens whose decoded text the FSM refuses; skips
      already-masked logits, forbids EOS unless final.
    - `advance_fsm_on_emit(&mut fsm, tokenizer, token_id)` — feeds
      the emitted token's text back to the FSM; drift surfaces as
      `FsmError::NoTransition`.
  - `Llama3Model::generate_grammar(tokenizer, prompt, max_new_tokens,
    grammar, temperature, top_k) -> Result<GenerateResult,
    GrammarGenError>` — grammar-constrained variant of `generate`.
    `GrammarGenError { Fsm(FsmError), NoValidToken { step } }` with
    `Display`, `std::error::Error`, and `From<FsmError>` impls.
  - `examples/lol_gen.rs` — DSL-agnostic reference: point `--grammar`
    at any GBNF file and `--model` at any GGUF and dump guaranteed-
    valid output. `cargo run --example lol_gen --features "grammar
    gguf" -- --model <path> --grammar <path> --prompt "..."`.
  - Server (`alice-llm-server`, `--features server`):
    - `CompletionRequest` / `ChatCompletionRequest` gained
      `grammar: Option<String>`. When present, the sampler is
      constrained by the parsed GBNF; when absent, behavior is
      byte-identical to prior versions.
    - `POST /v1/completions` and non-streaming
      `POST /v1/chat/completions` honor the field.
    - Chat streaming (`stream = true`) with `grammar` returns
      HTTP 400 explicitly — SSE + mask is future work; a loud error
      beats silently unconstrained output.
    - `server` feature now implies `grammar`, so the mask code
      always ships with the binary.
- CI: new `Examples & server compile` job compiles the new example
  and the server binary against the enabling features, protecting
  the grammar surface from silent bit-rot.

### Fixed

- `impl GrammarTokenizer for GgufTokenizer` was gated on
  `all(grammar, gguf)`, but `GgufTokenizer` itself is always
  compiled. Downstream crates enabling only `grammar` (e.g.
  `alice-lol/tests/lol_gbnf_test.rs`) hit an unresolved bound at
  `advance_fsm_on_emit`. Loosened to `grammar`-only. (commit
  `d048270`, follow-up to B-3.)

### Notes

- Phase X.8 B-9-A validated the end-to-end grammar → SdfNode path on
  Mac Metal with Qwen 3.5-4B Q4_K_M (CPU hybrid, ~1 tok/s): prompt
  `"generate lol: sphere(1.5)"` produced `SdfNode::Sphere { radius:
  1.5 }` in ~477s. The grammar mask + `BridgeError::Parse` two-stage
  safety net fired as designed when `max_new_tokens` capped a
  partial parse. Fine-tuned LOL emission is future work — the mask
  guarantees syntactic validity but semantic quality tracks the
  underlying model.
- Phase X.8 B-9-B (Jetson Vulkan smoke run) and a version bump to
  1.3.0 (additive `grammar` feature is SemVer-minor) are follow-up
  work.

## [1.2.1] - 2026-07-22

### Added

- **crates.io metadata** (Cargo.toml). Added `repository`, `homepage`,
  `readme`, `documentation`, `keywords = ["llm", "inference", "gguf",
  "gpu", "quantization"]`, and `categories = ["science", "algorithms",
  "wasm"]` for `alice-llm` crate publish readiness. Package verified
  via `cargo publish --dry-run`: 98 files, 1.8 MiB (429.9 KiB
  compressed). Description expanded to cover the actual feature set
  (GGUF v3, K-quants, hybrid CPU/GPU DeltaNet+Attention, wgpu
  compute shaders, speculative decoding, OpenAI-compatible HTTP
  server).
- **`.github/workflows/release.yml`** — multi-platform binary release
  workflow. Triggers on tag push `v*.*.*` (or `workflow_dispatch` for
  manual replay). Matrix build over 5 targets:
  `aarch64-apple-darwin` (macos-14), `x86_64-apple-darwin`
  (macos-15-intel), `x86_64-unknown-linux-gnu`,
  `aarch64-unknown-linux-gnu` (cross-compile with `gcc-aarch64-linux-gnu`),
  and `x86_64-pc-windows-msvc`. Builds `alice-llm-server --features
  server`, packages as `.tar.gz` (Unix) / `.zip` (Windows) with
  README + LICENSE + CHANGELOG + SHA256 checksum, uploads to GitHub
  Release via `softprops/action-gh-release@v2`. Enables single-binary
  distribution across Mac / Linux / Windows / aarch64 without
  requiring end-users to install Rust toolchain or ML framework
  dependencies.

## [1.2.0] - 2026-07-22

### Fixed

- **Phase X.3.e.3.37 — `o_proj` weight upload `cols` dimension bug** (0bd5d8e).
  For Qwen 3.5+ hybrid architectures where `q_dim = num_heads × head_dim`
  is not equal to `hidden_dim` (e.g. Qwen 3.5-4B has `hidden_dim=2560` and
  `q_dim=4096`), the GPU attention-layer `o_proj` upload was hardcoded as
  `upload_w(name, hidden_dim, hidden_dim)`. Only 62.5% of the Q4_K weight
  bytes were loaded, so 37.5% of the projection matrix was truncated,
  producing near-orthogonal `o_proj` output (cos 0.118 vs the CPU
  reference). Every downstream layer then compounded the drift, and the
  hybrid-per-layer path produced text like "I'm not sure what the user is
  asking about" instead of the correct Tokyo answer. One-line fix: pass
  `q_dim_attn = num_heads * head_dim` for `cols` (neutral for standard
  models where `q_dim == hidden_dim` — Llama, Qwen 2 / 2.5 / 3, Mistral).
  Result on both Mac Metal and Jetson Vulkan:
  Qwen 3.5-4B L3 pos 17 hidden cosine `0.7057 → 0.9970` across all
  positions; end-to-end generation "The capital of Japan is **Tokyo**.
  It is the country's capital, largest city, ..." Diagnostic journey:
  Phase X.3.e.3.30-37, 6 hypothesis revisions (DeltaNet layer 6/13 →
  attention gate → KV cache accumulation → f64/f32 accumulator precision
  → Metal `pow()` precision → Q4_K dequant → attention-tail projection)
  all rejected via bit-exact zero-delta ablations (Kahan summation and
  RoPE precomputed frequencies had zero effect), until direct per-op
  dumps revealed that V projection was correct (cos 0.9992) but `o_buf`
  was orthogonal (cos 0.118) — the shape parameter was the root cause.
  All 326 tests continue to pass on both Apple M3 Metal and Jetson Orin
  Nano 8GB (Vulkan).
- **`src/bin/server.rs` stale config API** (c1cfabe). Followed the
  `Llama3Config` God-object-free refactor: six field accesses
  (`full_attention_interval`, `linear_num_kv_heads`,
  `linear_qk_head_dim`, `linear_kv_head_dim`, `linear_num_v_heads`,
  `linear_conv_kernel_dim`) converted to method calls; `GpuModelConfig`
  gained two required fields (`neox_rope`,
  `attention_only_load`) sourced from `llm_config.use_neox_rope()` and
  `false` respectively. Restored `cargo build --release --features
  server --bin alice-llm-server` on both Apple Silicon and aarch64
  Vulkan. Verified end-to-end on Extoria-Jetson (Yahboom Orin Nano
  8GB): Llama-3.2-1B-Instruct-Q4_K_M loaded via `alice-llm-server
  --model … --port 8000`, `/v1/chat/completions` returns "Tokyo."
  at 9.68 tok/s over Tailscale MagicDNS. `attention_only_load: false`
  means the server bin still requires the full model to fit in unified
  memory (Qwen 3.5-4B needs 7.82 GB projected peak against 3.28 GB
  available on Jetson and OOM-kills); routing hybrid architectures
  through `GpuModel::run_attention_layer_only` inside the server is
  future work.

### Added

- **Phase X.3.e.3.36-37 — GPU per-op diagnostic infrastructure**
  (4bb2dfa, 0bd5d8e). Five new public `GpuModel` methods reading a
  single intermediate buffer after `stop_after` layer via
  `copy_buffer_to_buffer` + `map_staging`:
  `forward_stop_after_layer_and_read_v_buf` /
  `forward_stop_after_layer_and_read_k_buf` /
  `forward_stop_after_layer_and_read_attn_out` /
  `forward_stop_after_layer_and_read_o_buf` /
  `forward_stop_after_layer_and_read_down_buf`. `DiagBuf` enum +
  private `diag_read_buffer` helper deduplicate the staging-copy
  pattern. New `GpuModel::config()` public accessor for downstream
  diagnostic examples. `qwen_gpu` example gains `--dump-l3` and
  `--dump-l3-ext` flags emitting the JSONL dumps for direct
  element-wise comparison against CPU reference. On the CPU side,
  three new full-buffer JSONL dumps (`cpu_attn3_v_full`,
  `cpu_attn3_k_full`, `cpu_gated3_attn_gated_full`,
  `cpu_gated3_o_buf_full`, `cpu_gated3_ffn_out_full`) are emitted
  under the existing `ALICE_DUMP_ATTN3` / `ALICE_DUMP_GATED3`
  env-gated blocks in `llama3.rs`. These entry points are what
  narrowed Phase X.3.e.3.37 to a single-op mismatch after six
  hypothesis revisions had exhausted precision-oriented fixes.

### Documentation

- **`docs/ALICE_ROUTER_SPEC.md`** — design specification for the
  `alice-router` sibling crate (orchestration + verification layer
  above ALICE-LLM engines and external HTTP APIs). Spec only, no
  crate yet. 21 sections following `comprehensive-spec-templates`
  skill: vision / scope / positioning (vs Sakana Fugu / LangChain /
  LiteLLM / OpenRouter) / architecture / data model / Rust trait API
  surface / routing strategies / verification / backend integration
  (Kimi K3 API + AliceLLMBackend) / caching / observability / config
  TOML / error taxonomy / security / performance targets / testing
  plan / roadmap (R.0-R.7) / open questions / related ALICE work /
  glossary.
- **`README.md` + `README_JP.md`** (fc78c12, 5bc4ec8). Added
  Phase X.3.e.3.37 fix highlight, updated the Jetson Qwen 3.5-4B
  hybrid-per-layer line from `0.3 tok/s` (pre-fix, incoherent) to
  `0.4 tok/s` returning the correct "The capital of Japan is Tokyo.
  It is the country's capital, largest city," output, and added a
  Jetson multi-model support statement covering the four models
  verified on Extoria-Jetson (Yahboom Orin Nano 8GB) on 2026-07-21:
  Qwen 3.5-4B Q4_K_M `--hybrid-per-layer` at 0.4 tok/s, Ornith 9B
  Q4_K_M `--hybrid` at 0.2 tok/s, Bonsai 27B Q1_0 `--hybrid` at
  0.1 tok/s, and DeepSeek V2-Lite Q4_K_M (deepseek2 arch, MoE 64
  experts / 6 active per token) CPU at 0.1 tok/s.

## [1.1.0] - 2026-07-18

Aggregated work since `1.0.0`. Grouped by Phase; the "Phase X.Y.Z"
references map back to the roadmap in `memory/alice_llm_future_work.md`
and the journey entries in `memory/alice_llm_phase_x3e3_journey.md`.

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

[Unreleased]: https://github.com/ext-sakamoro/ALICE-LLM/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/ext-sakamoro/ALICE-LLM/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ext-sakamoro/ALICE-LLM/releases/tag/v1.0.0
