# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **DSpark Phase 7 — `DsparkLabelSample` + label collection method +
  `dspark_train_confidence_head` example + `--confidence-head` load path**
  (2026-07-31). RadixArk/Kimi-K3-DSpark 吸収の Phase 7 として Phase 6 の
  `DsparkAdvancedConfig` を実運用可能にする training pipeline を追加
  (1) `speculative_dspark::DsparkLabelSample { position, hidden, was_accepted }`
  を新規追加、`dspark-serde` feature で serde derive (2) `Llama3Model::generate_speculative_dual_collect_labels`
  を `#[cfg(feature = "dspark")]` gate で追加、vanilla speculative dual pipeline
  と同じ sampling path で各 draft position の `(hidden, was_accepted)` を
  collect (verify で reject された draft 以降と bonus は label 非付与)
  戻り値 `Result<(GenerateResult, Vec<DsparkLabelSample>), DsparkError>`
  (3) `examples/dspark_train_confidence_head.rs` 新規追加、label collect →
  position 別 accept rate 統計 → `PositionConfidenceHead::train_step` で
  epochs × samples SGD BCE 学習 → bincode で save (required-features =
  `[dspark, dspark-serde, gguf]`) (4) `examples/speculative_dspark_dual.rs`
  に `--confidence-head <path>` optional CLI arg 追加、指定時は bincode
  load + threshold 0.3/0.5/0.7 の A/B/C confidence-gated variant 追加、
  `dspark-serde` feature 無しなら warning + skip 3 追加 unit test (計 79
  default / 84 with dspark-serde) 全 pass、default lib test 558 pass、
  clippy pedantic+nursery 0 warn 新規範囲、fmt check pass Phase 8
  (KimiK3Model::forward_capture_hidden or DFlashParallelDraft の llama3
  統合検討) は次 session

- **Sparse attention env hook in `gqa_attention` — Phase MSA.5.6**
  (2026-07-31). `llama3.rs::gqa_attention` (Qwen 3.5 / Llama 3 / Bonsai /
  Elyza / Gemma / every standard GQA arch) now checks the
  `ALICE_SPARSE_TOPK` environment variable at call time. When set to a
  non-negative integer (and no other diagnostic env is active, and
  `attn_logit_softcap` is `None`) the function gathers the current layer's
  dense KV slice, wraps it as `DenseKvCacheView`, and dispatches to
  `sparse_attention::llama3_bridge::llama3_sparse_attention` (renamed from
  the earlier `k3_sparse_attention` — the K3-specific naming was
  misleading; the adapter targets standard dense-KV Llama-style attention,
  not K3's MLA path). `TOPK=0` selects every sparse block and is
  arithmetically equivalent to dense attention modulo FP re-association;
  larger values pick only the top-K KV blocks per query, cutting attention
  compute at the cost of some accuracy. When the adapter rejects the input
  (e.g. exotic geometry) the function transparently falls back to the
  existing dense path. Smoke: the full 558-test lib suite passes with
  `ALICE_SPARSE_TOPK=0` and `ALICE_SPARSE_TOPK=4` set. Does **not** hook
  the Kimi K3 MLA path (`kimi_k3_gated_mla_step`): K3 uses Multi-head
  Latent Attention with LoRA-compressed KV, so sparsifying it requires a
  separate MLA-aware bridge (documented in
  `src/sparse_attention/llama3_bridge.rs` module header).

- **Sparse attention llama3 bridge adapter — Phase MSA.5.5** (2026-07-31).
  New `src/sparse_attention/llama3_bridge.rs` module ships
  `DenseKvCacheView`, `BridgeConfig`, and `llama3_sparse_attention` as a
  one-shot high-level adapter that wraps
  `sparse_attention::kvouter_attention` for callers that keep K / V as
  dense `[seq_len, hkv, head_dim]` tensors (the shape `gqa_attention`
  consumes). Internally repacks the dense KV into the paged layout the
  sparse pipeline expects, builds the block table + `SparseSelection` (dense
  fallback when `topk == 0`, otherwise runs the dense proxy pass through
  `sparse_topk_select_batch` to pick top-K blocks), and finally invokes
  `kvouter_attention`. 5 tests cover 3-way parity (naive
  `scaled_dot_product_attention` ≡ `sparse_attention::kvouter_attention`
  with all blocks selected ≡ `llama3_sparse_attention` with `topk == 0`,
  rel err < 1e-4), explicit `topk == num_blocks`, partial-`topk` bounded
  output, and shape / head-dim mismatch rejection.

- **Sparse attention (KV-outer) module — Phase MSA.1 – MSA.4** (2026-07-31).
  Rust-from-scratch port of the algorithm described in MiniMax Sparse
  Attention (`MiniMax-AI/MSA`, MIT) and Fireworks AI's M3 KV-outer sparse
  attention (`fw-ai/minimax-kernels`, Apache-2.0); no upstream CUDA /
  CuTe-DSL kernel is vendored, only mathematical formulation and tensor
  contracts. The new module `src/sparse_attention/` ships (1) tensor types
  (`SparseSelection`, `BlockTables`, `CuSeqlensQ`, `KvOuterIndex`,
  `SparseAttentionError`); (2) CSR inverse-index builder
  `build_kvouter_index` that collapses the upstream 5-kernel index-build
  pipeline (`InitSlotsAndCounts` / `CountEdges` / `ReduceReplicas` /
  `CountToOffsets` / `ScatterRanks`) into two linear passes; (3) top-K
  KV-block selector `sparse_topk_select` (histogram + insertion-sort
  translated from MSA's `sparse_topk_select.cuh`, itself derived from
  TensorRT-LLM's `indexerTopK.cu`); (4) dense proxy pass
  `compute_proxy_block_max_scores` for cheap-Q-slice per-block max score;
  (5) load-balance scheduler (`WorkSplit`, `enumerate_work_units`,
  `build_fixed_schedule`); (6) KV-outer forward `kvouter_forward` that
  loads each selected KV block once and emits per-`(edge, qhead_lane)`
  unnormalized partials plus online-softmax `(m, l)` stats with GQA row
  packing and right-aligned causal masking; (7) LSE combine `lse_combine`
  implementing the standard FlashAttention `M = max mᵢ`,
  `l_out = Σ exp(mᵢ - M)·lᵢ`, `o_out = Σ exp(mᵢ - M)·oᵢ / l_out` identity;
  and (8) one-shot `kvouter_attention` public entry point that chains the
  three stages. Verified against a naïve dense reference in three
  configurations (no-causal, GQA `Hq=4`/`Hkv=2`, causal) with relative
  error < 1e-4 when every sparse block is selected. Adds
  `examples/sparse_attention_demo.rs` (end-to-end pipeline runner) and a
  new `NOTICE` file for upstream attribution. Everything is pure CPU with
  zero new dependencies; parallel / SIMD / GPU (wgpu) / FP8 paths are
  deferred to a follow-up Phase MSA.5. 33 new tests (11 index / 8 topk /
  4 proxy / 4 scheduler / 4 combine / 1 API + 1 demo), 547 lib tests
  total, clippy pedantic + nursery 0 warnings, `cargo fmt` clean.

- **DSpark Phase 6 — `DsparkAdvancedConfig` + `forward_capture_hidden` +
  `PositionConfidenceHead` 統合 with confidence-gated 早期打切り**
  (2026-07-31). RadixArk/Kimi-K3-DSpark 吸収の Phase 6 として (1)
  `Llama3Model::forward_capture_hidden(token_id, layer_idx) -> (Vec<f32>, Vec<f32>)`
  を `#[cfg(feature = "dspark")]` gate で新規追加 既存
  `forward_with_layer_hook` を経由して target layer の hidden state を
  clone、`layer_idx = None` は最終層 (num_layers - 1)、非標準 arch
  (Gemma3n/Gemma4/DeepSeekV3/KimiK3/Hy3) は specialized forward path が
  hook を bypass するため `unimplemented!` で fail fast (2)
  `speculative_dspark::DsparkAdvancedConfig<'a>` 新規追加、`confidence_head`
  + `confidence_threshold` + `hidden_capture_layer` の 3 field、reference
  field を持つため serde derive は付けない (weight 単体で serialize する
  ことは可能) (3) `Llama3Model::generate_speculative_dual_dspark` に第 9
  引数 `advanced: Option<&DsparkAdvancedConfig>` 追加、`None` は Phase 5
  と bit-exact 同一動作、`Some(cfg)` で draft position ごとに hidden state
  抽出 + `PositionConfidenceHead::predict` で confidence 算出 +
  `confidence < threshold` で draft 早期打切り (KV cache は既存 rollback
  で自動整合) (4) method entry で cfg.confidence_head.block_size >=
  spec_k + hidden_dim 一致を検証、失敗時は `ConfidenceHeadBlockSizeMismatch`
  / `HiddenDimMismatch` を返す (5) `examples/speculative_dspark_dual.rs`
  を Phase 5 の 3 呼出 sites に `None` を追加して bit-exact 動作維持、
  Phase 7+ の trained head 追加を promise API: `DsparkAdvancedConfig::{new,
  confidence_head, confidence_threshold, hidden_capture_layer}` +
  `Llama3Model::forward_capture_hidden` 3 追加 unit test (計 77 default /
  81 with dspark-serde) 全 pass、clippy pedantic+nursery 0 warn 新規範囲、
  fmt check pass、default lib test 515 pass (+3) Phase 7 (trained
  PositionConfidenceHead の accept/reject label collection example +
  DFlashParallelDraft の llama3 統合検討) は次 session

- **DSpark Phase 5 — llama3 `generate_speculative_dual_dspark` +
  `dspark` feature + `apply_bigram_bias_maybe` helper +
  `examples/speculative_dspark_dual.rs`** (2026-07-31). RadixArk/Kimi-K3-DSpark
  吸収の Phase 5 として llama3.rs の speculative dual decoding pipeline
  に DSpark bigram bias を統合 (1) `Llama3Model::generate_speculative_dual_dspark`
  を `#[cfg(feature = "dspark")]` gate で新規追加 (既存 `generate_speculative_dual`
  完全無改変)、signature は vanilla + `bigram_bias: Option<&dyn BigramBias>` +
  `bigram_strength: f32` の 2 引数追加、戻り値は `Result<GenerateResult, DsparkError>`
  で bigram apply エラーを silent 化せず propagate (2) draft argmax 前に
  bigram bias を in-place 加算、biased logits を argmax にも
  `draft_logits_all` (verify の `q`) にも使うため Leviathan formula の
  q 分布が実 draft policy と一致 (3) `speculative_dspark::apply_bigram_bias_maybe`
  helper を追加 (None or strength=0 で no-op、それ以外は `?` 伝播) 4 追加
  unit test で helper 挙動を実測 (4) `Cargo.toml` に `dspark = []` feature
  追加 + `[[example]] name = "speculative_dspark_dual"` (baseline vs
  bigram_strength 0.1/0.5/1.0 の accept rate + tok/s + speedup 比較、実 K3
  models で Track 5-4 続報記事の accept length 実測に直結) API:
  `generate_speculative_dual_dspark(&mut self, draft_model, tokenizer, prompt,
  max_new_tokens, temperature, spec_k, bigram_bias, bigram_strength)
  -> Result<GenerateResult, DsparkError>` 全 test: default 512 pass /
  `--features dspark-serde` で 78 speculative_dspark test / `cargo build
  --features dspark --example speculative_dspark_dual` compile pass /
  clippy pedantic+nursery 0 warn 新規範囲 / fmt check pass Phase 6
  (PositionConfidenceHead / DFlashParallelDraft の llama3 統合) は hidden
  state exposure が必要で次 session

- **DSpark Phase 4 — `BigramBias` trait + `FullCountBigramBias` +
  `dspark-serde` feature** (2026-07-31). RadixArk/Kimi-K3-DSpark 吸収の
  Phase 4 として (1) `BigramBias` trait を追加 (`vocab_size` / `apply` の
  2 method) + `MarkovBigramBias` / `FullCountBigramBias` の両方が実装、
  (2) `FullCountBigramBias` は `HashMap<TokenId, HashMap<TokenId, u32>>`
  で全観測 count を sparse 保持し apply 時に count 降順 → token id 昇順
  で top-K を選択、Phase 1 の eager truncate 制約 (rank 到達後の tied
  arrival が drop される罠) を解消、(3) `DFlashParallelDraft::draft` の
  signature を `Option<&MarkovBigramBias>` → `Option<&dyn BigramBias>` に
  変更 (Rust 型推論で既存呼出は無改修で coerce)、(4) optional feature
  `dspark-serde = ["dep:serde"]` を追加、`MarkovBigramBias` /
  `FullCountBigramBias` / `PositionConfidenceHead` / `DFlashParallelDraft`
  / `DraftPosition` / `DraftBlock` / `DsparkError` に serde derive を条件付
  追加、`[dev-dependencies] bincode = "1"` で roundtrip test API:
  `FullCountBigramBias::{new, from_sequence, vocab_size, rank, is_empty,
  observed_prev_count, unique_next_count, count, observe, observe_sequence,
  apply}` 18 追加 unit test (計 70 test、+ serde feature で 4 追加、計 74
  test) 全 pass、clippy pedantic+nursery 0 warn (両 feature)、fmt check
  pass Phase 5 (llama3.rs `generate_speculative_dual` optional feature gate
  配線) は次 session

- **DSpark Phase 3 — `speculative_dspark::DFlashParallelDraft`** (2026-07-29).
  RadixArk/Kimi-K3-DSpark 3 要素の 3 番目 DFlash 並列 draft を実装 外部
  draft model は `FnOnce(&[TokenId], u32) -> Result<Vec<DraftPosition>, DsparkError>`
  closure 経由で受け取り (llama3.rs 非依存)、`DraftPosition { hidden, logits }` /
  `DraftBlock { tokens, confidences, hidden_states }` I/O 型で交換する
  各 position i について prev = `if i == 0 { prefix.last() } else { tokens[i-1] }`
  を bigram prev として使い、strength != 0 かつ bigram_bias 提供時のみ
  `bigram.apply(prev, &mut logits, strength)` を適用、`argmax_finite` (NaN skip、
  全 NaN で `DraftLogitsAllNonFinite`) で token 確定、`confidence_head.predict(i, hidden)`
  で位置別 confidence を算出 API: `DFlashParallelDraft::{new, block_size,
  vocab_size, hidden_dim, bigram_strength, set_bigram_strength, draft}`
  `DsparkError` に 8 variant 追加 (EmptyPrefix / DraftModelBlockSizeMismatch /
  VocabSizeMismatch / HiddenDimMismatch / ConfidenceHeadBlockSizeMismatch /
  BigramVocabMismatch / DraftLogitsAllNonFinite / DraftModelFailed(String)) 計
  17 variant 19 追加 unit test 全 pass (計 52 test)、clippy pedantic+nursery
  0 warn、fmt check pass Phase 4 (llama3.rs `generate_speculative_dual` optional
  feature 配線 + full-count sketch) は次 session

- **DSpark Phase 2 — `speculative_dspark::PositionConfidenceHead`** (2026-07-29).
  RadixArk/Kimi-K3-DSpark 3 要素の 2 番目 位置別 confidence head を実装
  各 draft position i ごとに per-position 重み `w_i ∈ R^H` + bias `b_i ∈ R`
  を持ち、`confidence_i = sigmoid(w_i · hidden_i + b_i) ∈ [0, 1]` を返す
  SGD BCE 学習は canonical form `dL/dz = p - y` で数値安定 sigmoid + eps
  clamp BCE 使用、log(0) 回避 API: `PositionConfidenceHead::{new, zeros,
  block_size, hidden_dim, predict, predict_block, train_step, accept_mask}`
  17 追加 unit test 全 pass (計 33 test)、clippy pedantic+nursery 0 warn、
  `DsparkError` に 5 variant 追加 (ZeroBlockSize / ZeroHiddenDim /
  HiddenLenMismatch / PositionOutOfRange / BlockStatesCountMismatch) Phase 3
  (DFlashParallelDraft + `generate_speculative_dual` 配線) は次 session

- **DSpark Phase 1 — `speculative_dspark::MarkovBigramBias`** (2026-07-29).
  RadixArk/Kimi-K3-DSpark 吸収 3 要素 (DFlash 並列 draft + Markov logit-bias +
  位置別 confidence head) の 1 要素目 Markov bigram bias を実装 各 prev
  token に対して top-K next の観測頻度を eager truncate で保持し、apply 時に
  `logits[next] += strength * ln_1p(count)` を加算する vanilla DSpark rank=256
  設定に対応 API: `MarkovBigramBias::{new, from_sequence, observe,
  observe_sequence, apply, vocab_size, rank, is_empty, observed_prev_count,
  bucket_len}` + `DsparkError` (4 variant) 16 unit test 全 pass、clippy
  pedantic+nursery 0 warn (`imprecise_flops` 修正で `ln_1p` 使用)

- **🎉 Real Kimi K3 (moonshotai/Kimi-K3 2.8T MoE) 1 token forward
  完走達成** (2026-07-28 22:47 JST). Full pipeline demonstrated on
  real GrEarl/Kimi-K3-GGUF-IQ1_S (566GB across 94 shards) via
  ALICE-LLM pure Rust implementation. Only Rust K3 implementation
  known to actually run real K3 weights end-to-end (llama.cpp
  pwilkin PR #26185 is still draft stage; the reference GGUF
  converter has not landed upstream). Environment: Mac mini M2 Pro
  (10-core, 32 GB unified) + external USB SSD 960 GB ExFAT for
  the split GGUF, macOS 26.5.1, Rust 1.94.1, alice-llm 1.6.0
  features `gguf,hf-config,parallel`, invoked via
  `cargo run --release --example kimi_k3_forward -- shard1.gguf 1`
  with the split shard 1 path (sibling shards auto-discovered).
  Metrics:
  - **GGUF parse** (94 shards, 2573 tensors): 7 ms mmap + parse
  - **Config extraction** (`Llama3Config::from_gguf`): 150 µs
  - **Weight loading** (all K3 tensors across shards): 73.5 sec
    (uncached cold; ~1 sec on second run with OS page cache)
  - **Model construction** (`KimiK3Model::new`): 16 ms
  - **Forward pass** (token_id=1, 93 layers): **1636 sec = 27.3 min**
    - KDA + MoE layer: ~18.8 sec each (69 layers)
    - MLA + MoE layer: ~12.3 sec each (23 layers)
    - Dense layer 0 (KDA + Dense FFN Q4_K): 25 sec
  - **Output**: 163840 logits (vocab_size), **163840/163840 finite**
    (NaN/Inf = 0), argmax token_id=**220** with logit 7.8632, top-5
    `[220, 269, 62, 12, 25]` — all math paths (KDA aggregation +
    MLA + Dense + AttnRes + LatentMoE + IQ1_S expert dispatch)
    produce sane numeric output.
  - Bottleneck: **I/O bound** on external USB SSD (25-30 MB/s random
    reads to page in mmapped expert cubes). Rayon parallelism helps
    on the compute side (up to 6 cores active during rayon phases)
    but the SSD I/O is the wall. Improvements roadmap: (1) NVMe
    upgrade for 5-10× faster random reads, (2) per-token expert
    cache to avoid re-paging the same experts across gate/up/down
    matvec calls, (3) NEON-optimized fused IQ1_S × F32 matvec
    (llama.cpp reference).

- **Phase X.4.b.7 — IQ1_S dequant + fused-block matvec + rayon parallel**
  (2026-07-28). llama.cpp k_quants `iq1_s` reference port. New
  `src/iq1_s.rs` module (~625 LOC) with `IQ1S_GRID: [u64; 2048]`
  codebook (ternary values packed 8-at-a-time), `IQ1S_DELTA = 0.125`,
  and `dequantize_iq1_s(data, out)`. Wired into
  `crate::gguf::tensor_to_f32` for the IQ1_S arm and into
  `quantized_matvec` via new `iq1_s_matvec_fallback` that:
  1. Row-parallelizes via rayon (`par_iter_mut` on output rows) —
     each row = fully independent dequant-and-dot workload, up to
     linear speedup with core count.
  2. Uses `iq1_s_row_fused_dot`: per-block (256 elements) dequant
     into a stack-allocated `[f32; 256]` buffer, dot with input,
     accumulate. Zero per-row heap alloc (was 14 KB × 3072 rows
     = 43 MB churn per matvec in the correctness-first version).

- **Phase X.4.b.6 — GgufMultiFile split GGUF loader + GgufSource trait**
  (2026-07-28). Real K3 distributions are 94-file splits (each
  `<prefix>-NNNNN-of-TTTTT.gguf`), too large to merge (94 × ~6 GB
  = 566 GB, and `llama-gguf-split --merge` would need 566 GB more
  headroom that doesn't exist on the K3 test rig). Split loader
  virtualises across shards: each shard is parsed as its own
  `GgufFile`, then a global `tensor_name → shard_index` map
  routes `tensor_data` / `tensor_info` calls to the right shard.
  Model-level metadata is delegated to shard 0 (llama.cpp split
  convention: shard 0 has full metadata, shards 1..N carry only
  `split.*` keys). New `GgufSource<'a>` trait exposes the subset
  of methods the K3 loader needs; blanket-implemented for both
  `GgufFile<'a>` and `GgufMultiFile<'a>`, so downstream loaders
  (`load_kimi_k3_model_weights`, `KimiDeltaConfig::from_gguf`,
  `ModelArch::from_gguf`, `Llama3Config::from_gguf`,
  `load_weight_ref_any_shape`) accept either transparently.

- **Phase X.4.b.5 — real K3 tensor adaptations (ssm_g / ssm_a / MoE
  aliases)** (2026-07-28). Real GrEarl K3 GGUF shard-1 inspection
  revealed KDA layer structure additions:
  - `blk.{N}.ssm_g.weight`: K3 full-rank per-head output gate
    matrix, shape `[num_heads * v_head_dim, hidden]`, replaces
    Kimi-Linear's low-rank `ssm_g_a` / `ssm_g_b` pair. Wired into
    `KimiK3KdaAttn::ssm_g: Option<WeightRef>`; per-head slice fed
    to `kimi_delta_forward_head` as `w_gate` (identity fallback
    when absent for skeleton fixtures).
  - `blk.{N}.ssm_a`: K3 per-head `A_h` log-scale array of length
    `num_heads`. Replaces the paper's hardcoded `A_h = 0` init.
    Wired into `KimiK3KdaAttn::ssm_a: Option<Vec<f32>>`; per-head
    scalar plumbed to `KimiDeltaHeadParams::a_h`.
  - MoE tensor name aliases: `routed_exp_up/down/norm` (Kuberwastaken
    TENSOR_MAP.md) vs `ffn_routed_up/down/norm` (pwilkin PR #26185)
    — loader tries both spellings, uses whichever exists per-tensor.
  - `attn_gate` now treated as MLA-only (real K3 KDA layer has no
    `attn_gate` — uses `ssm_g` instead). Placeholder `WeightRef`
    substituted for KDA layers so struct layout stays uniform.

- **Phase X.4.b.4 — layer type detection via tensor presence** (2026-07-28).
  Real GrEarl K3 GGUF does not export `kimi-k3.full_attn_layers` /
  `kimi-k3.kda_layers` metadata arrays that the skeleton loader
  depended on. Two-tier resolution:
  1. **Metadata path** — `KimiDeltaConfig::is_mla_layer(il)` returns
     `Some(true/false)` when arrays are present (synthetic fixtures,
     pwilkin PR export).
  2. **Tensor-presence fallback** — when metadata returns `None`,
     probe `blk.{il}.attn_q_a.weight` (MLA-only LoRA A projection)
     via `gguf.tensor_info()`: exists → MLA, absent → KDA. Applied
     in `load_kimi_k3_layer_weights` (loader-time dispatch) and
     mirrored in `KimiK3Model::new` cache allocation (reads
     `weights.layers[il].attn` enum discriminant, which was set
     by the loader's fallback, so no re-probe needed).

- **Phase X.4.b.3 — config key aliases for real GrEarl GGUF** (2026-07-28).
  Six K3 metadata key spellings diverge between the pwilkin PR
  synthetic export and real GrEarl output; loader now accepts both:
  - `kimi-k3.attention.kda_head_dim` (synth) / `kimi-k3.kda.head_dim` (real)
  - `kimi-k3.attn_res_block_size` / `kimi-k3.attn_res.block_size`
  - `kimi-k3.gate_lower_bound` / `kimi-k3.kda.gate_lower_bound`
  - `kimi-k3.activation_situ_beta` (underscore) / `kimi-k3.activation.situ_beta` (dot)
  - `kimi-k3.moe_router_activation_func` / `kimi-k3.expert_gating_func`
  - `kimi-k3.routed_expert_hidden_size` / `kimi-k3.expert_latent_length`

- **Phase X.4.c.3.3.b.2 — quantized per-row + per-expert slicing
  (Q4_K/Q8_0/MXFP4/etc.)** (2026-07-28). Extends
  `kimi_k3_slice_weight_ref_rows` and `kimi_k3_expert_plane_weight_ref`
  from F32-only to full quant zoo (F32/F16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/
  Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_XS/Q1_0/Q2_0/MXFP4). Key insight:
  GGUF row-major storage has quant blocks along the COLUMN axis, not
  spanning rows. As long as `cols % elements_per_block == 0` (which
  holds for K3 tensors since `hidden = 7168` divides both 256 and
  32), per-row byte offsets ARE block-aligned. The earlier
  "kda_head_dim = 128 is half a block" concern applied only to per-
  column splits, not per-row (= per-head) splits. This closes the
  last "F32-only scoping" limitation in K3's forward path: real K3
  GGUF (Q2_K / IQ1_S / MXFP4 native) can now be loaded and forward
  passes through KDA per-head aggregation + LatentMoE per-expert
  dispatch without hitting the previous `panic!("F32 fixtures only")`
  paths.
  - **`kimi_k3_slice_weight_ref_rows` extension** — replaces the
    `F32-only` gate with `cols % elements_per_block == 0` check + per-
    row byte offset math via `blocks_per_row × block_bytes`.
    Row_start / row_end must land on row boundaries (always true when
    the caller passes head indices).
  - **`kimi_k3_expert_plane_weight_ref` extension** — replaces the
    `F32-only` gate with `(rows × cols) % elements_per_block == 0`
    check + per-expert byte offset via `plane_blocks × block_bytes`.
    Real K3's `ffn_gate_exps` (shape `[n_embd_latent, moe_intermediate,
    num_experts]` = `[1024, 2048, 896]` typically) has per-expert
    plane = 2M elements = block-aligned for all K3 quant types.
  - **5 new unit tests** — `slice_weight_ref_rows_supports_q4_k_when_aligned`,
    `slice_weight_ref_rows_supports_mxfp4_when_aligned`,
    `slice_weight_ref_rows_supports_q8_0_when_aligned`,
    `expert_plane_weight_ref_supports_quantized_cube_when_aligned`,
    `expert_plane_weight_ref_supports_mxfp4_when_aligned`.
  - **2 existing misalignment tests preserved** as defensive checks
    (`_rejects_misaligned_quantized_cols`,
    `_rejects_quantized_cube_when_misaligned`).
  - **LatentMoE routed dispatch panic diagnostic upgrade** — replaces
    the "F32 fixtures only" scoping message with a plane-alignment-
    focused diagnostic that reports qtype, elements_per_block, plane
    bytes, and cube data length. Users hitting the panic get an
    actionable "check `n_embd_latent * n_ff_exp` divisibility"
    hint.
  - **439 tests pass (hf-config, +5)** / **434 tests pass (default)**
    / fmt clean / doc clean / clippy K3 region warning 0.

- **Phase X.4.c.3.4.a-f — AttnRes tensor refactor + res_mix wiring
  into forward + tests** (2026-07-28). Implements the actual K3
  Block Attention Residuals wiring following the pwilkin PR #26185
  spec (which was documented in the Phase X.4.c.3.4 research
  landing earlier the same day). Six sub-tasks:
  - **X.4.c.3.4.a: `kimi_k3_res_mix` primitive** (pwilkin
    `src/models/kimi-k3.cpp` L218-257 verbatim math) — softmax-
    weighted mixture of banked checkpoints + current residual
    stream using a fused 1D score vector. `KimiK3AttnResState`
    struct manages the banked-checkpoint list + block_size +
    hidden_dim, with `bank()` / `reset()` / `is_checkpoint_layer()`
    methods.
  - **X.4.c.3.4.b: `KimiK3LayerWeights` refactor** — collapses the
    paper-derived 4 tensor layout (`attn_res_norm`/`_proj` +
    `ffn_res_norm`/`_proj`) to the actual GGUF export 2 tensor
    layout (`attn_res_score`, `ffn_res_score`, both 1D
    `[n_embd]`). Loader updated to `blk.{N}.attn_res_score.weight`
    + `blk.{N}.ffn_res_score.weight`.
  - **X.4.c.3.4.c: `KimiK3ModelWeights::output_res_score`** —
    replaces `output_attn_res_norm/proj` with a single 1D
    `output_res_score` field. Loader updated to
    `output_res_score.weight`.
  - **X.4.c.3.4.d: `KimiK3Model::forward` per-layer wiring** —
    inserts `res_mix` twice per layer (before `attn_norm` +
    before `ffn_norm`) with checkpoint layer banking (bank raw
    `prefix_sum` on `il % block_size == 0`) + prefix_sum reset
    (banked layer's post-attn `prefix_sum = attn_output` alone,
    non-banked layer's post-attn `prefix_sum += attn_output`).
    The attention and FFN forwards now receive `cur = res_mix(...)`
    instead of raw `x`.
  - **X.4.c.3.4.e: final output mix** — inserts `res_mix(x,
    output_res_score)` before `output_norm` (pwilkin L358-360).
  - **X.4.c.3.4.f: 8 new unit tests** — state creation,
    checkpoint predicate, bank/reset, `res_mix` identity when
    no banked, convex combination correctness, score bias effect
    on prob distribution, deterministic across repeated calls.
    Plus updated `model_reset_clears_all_caches_and_block_attnres`
    to use the new `attn_res_state` field.
  - **Model field change** — `block_attnres: Option<BlockAttnResState>`
    replaced by `attn_res_state: KimiK3AttnResState` (eager init
    from `hidden_dim` + `block_size` at construction). `reset()`
    now calls `attn_res_state.reset()`. The paper-conformant
    `BlockAttnResState` primitive is preserved for reference but
    no longer referenced by `KimiK3Model`.

- **Phase X.4.c.3.4 (research) — pwilkin PR #26185 精読 + AttnRes wiring
  semantics docs update** (2026-07-28). Downloaded `src/models/kimi-k3.cpp`
  (645 行, SHA `2043a6a8...`) + `gguf-py/gguf/constants.py` from the pending
  llama.cpp PR (`ggml-org/llama.cpp#26185` "model: add Kimi-K3 text model")
  and reverse-engineered the actual GGUF export shape for the AttnRes
  mechanism. Key finding vs. our previous assumption based on Kuberwastaken's
  TENSOR_MAP.md: the paper's separate `_norm` + `_proj` per-layer AttnRes
  tensors are **fused into a single 1D `_score` vector per site** in the
  llama.cpp GGUF export. Extracted verbatim `res_mix` / layer-loop / banking
  logic and captured in `docs/KIMI_K3_INTEGRATION.md` under a new
  "llama.cpp reference wiring (pwilkin PR #26185)" subsection of the
  Block AttnRes section (~180 lines with C++ excerpts, semantic call-outs,
  and derived 6-step ALICE-LLM `X.4.c.3.4.a-f` sub-task list). No production
  code changed in this phase — documentation only, unblocking the real
  wiring work planned for Phase X.4.c.3.4.
  - **Tensor names** (GGUF, per-layer + model): `blk.{N}.attn_res_score`,
    `blk.{N}.ffn_res_score`, `output_res_score`, all 1D `[n_embd]` with
    llama.cpp comment `# Kimi K3 (fused res_norm * res_proj, ...)`.
  - **`res_mix` semantic**: (1) stack banked ckpts → `[n_embd, n_ckpt, T]`;
    (2) score each via `sum_rows(RMSNorm(x) * score_w)`; (3) softmax over
    `n_ckpt+1` (banked + current); (4) weighted sum uses **raw
    non-normalized** `src` + `cur` (RMSNorm only for score computation,
    mirroring paper §2.2 Eq 8-10 duality).
  - **Layer wiring**: `res_mix` called 2× per layer (before `attn_norm`
    via `attn_res_score`, before `ffn_norm` via `ffn_res_score`), plus
    1× at model output before `output_norm` via `output_res_score`
    (`2L+1 = 187` calls at K3's `L=93`). Checkpoint layer (`il % res_bs
    == 0`, `res_bs = 12`) banks the **raw `prefix_sum`** (not the
    `res_mix`-transformed value) and resets `prefix_sum` to the attention
    output alone (`prefix_sum = banked ? cur : prefix_sum + cur`).
  - **Fused kernel**: `ggml_dsv4_hc_pre` (from DeepSeek V4 lineage) does
    `out[d, t] = Σ_c p_src[c, t] * src[d, c, t]` in one SIMD pass. The
    corresponding Rust primitive will be an explicit `for c in 0..n_ckpt`
    weighted-accumulate.
  - **GGUF metadata keys added by PR #26185** (all in `kimi-k3.*`
    namespace): `attn_res_block_size` (u32, K3 default 12),
    `expert_latent_length` (u32, `moe_intermediate_size / 2`),
    `activation.situ_beta` (f32, default 4.0),
    `activation.situ_linear_beta` (f32, default 25.0).
  - **Impact on Phase X.4.c.3.4 (real wiring)**: existing 4-tensor
    per-layer skeleton (`attn_res_norm/proj` + `ffn_res_norm/proj`) will
    be collapsed to 2 tensors (`attn_res_score` + `ffn_res_score`, both
    1D `[n_embd]`) + a new `output_res_score` field on
    `KimiK3ModelWeights`. `BlockAttnResState::res_mix` primitive will be
    refactored to accept `score_w: &[f32]` instead of the fuller
    `q_proj` matrix path currently in `block_attnres_softmax_attention`.
    Phase table row updated with derived 6-step sub-task list
    (X.4.c.3.4.a-f).

- **Phase X.4.c.3.3.c — Stable LatentMoE forward SCOPED (router + shared
  real, routed `todo!()`)** (2026-07-28). Wires the K3 LatentMoE MoE-layer
  primitive into `KimiK3Model::forward`, replacing the third of the four
  original `todo!()` fail-fasts. The router (sigmoid gating with `noaux_tc`
  bias correction + top-k selection + renormalization) and the fused
  shared-experts SwiGLU branch are real; the per-expert routed dispatch
  (3-D cube slicing → SiTU-GLU → weight-sum → K3-only `routed_exp_norm`
  RMSNorm → `routed_exp_up` projection) remains `todo!()` for Phase
  X.4.c.3.3.c.2. Real K3's 92 MoE layers now execute router + shared
  experts end-to-end when hit; the panic surfaces cleanly on the routed
  aggregation with a phase-named message.
  - **`kimi_k3_moe_router(x, ffn_gate_inp, exp_probs_b, top_k,
    renormalize)`** — sigmoid gating + top-k selection with `noaux_tc`
    bias correction. Selection scores use `sigmoid(router @ x) + bias`,
    but the emitted weights are the **raw** sigmoid scores (bias is
    omitted from the weight — mirrors paper §2.3.3 Eq 13 "b is omitted
    from p_{i,j}"). Returns selected `(expert_idx, weight)` pairs sorted
    by `expert_idx` for deterministic downstream reduce; weights
    renormalized to sum to 1 when `renormalize = true` (K3 default).
  - **`kimi_k3_shared_experts_forward(x, gate, up, down)`** — 2 shared
    experts fused SwiGLU. K3 GGUF exports `num_shared_experts = 2` as a
    single weight triple (`ffn_gate_shexp / ffn_up_shexp /
    ffn_down_shexp`) with the two shared experts pre-summed into the
    intermediate dim, so the forward is a single SwiGLU application.
  - **`kimi_k3_latent_moe_forward(x, ffn_norm, moe, top_k, renormalize,
    eps)`** — end-to-end skeleton with router + shared wired; per-expert
    routed dispatch stops on `todo!()` with a docstring listing the
    complete 5-step routed-path plan.
  - **`KimiK3Model::forward` update** — LatentMoE branch delegates to
    `kimi_k3_latent_moe_forward`, extracting `top_k` and `renormalize`
    from `Llama3Config` (`num_experts_per_tok`, `norm_topk_prob`).
  - **4 new unit tests** — `moe_router_selects_top_k_by_score`,
    `moe_router_bias_shifts_selection_without_affecting_weights` (paper
    §2.3.3 spec check), `moe_router_returns_sorted_indices` (deterministic
    reduce), `shared_experts_forward_zero_input_gives_zero`.

- **Phase X.4.c.3.3.a — MLA + Dense FFN wiring into `forward_kimi_k3`**
  (2026-07-28). Wires the X.4.c.3.2 Gated MLA primitive + a new
  Dense-FFN SwiGLU primitive into `KimiK3Model::forward`, replacing
  two of the four remaining `todo!()` fail-fasts. K3 forward now
  really runs the MLA attention path + the layer-0 dense FFN path
  end-to-end when handed a populated `KimiK3ModelWeights` bundle;
  KDA per-head aggregation (X.4.c.3.3.b) and Stable LatentMoE
  (X.4.c.3.3.c) remain the two panics keeping the full 24-MLA +
  69-KDA + 92-MoE forward from running.
  - **`kimi_k3_extract_mla_config(config)`** — pulls the runtime
    dims for `kimi_k3_gated_mla_step` (`d`, `num_heads`,
    `qk_nope/rope/v_head_dim`, `q_lora_rank`, `kv_lora_rank`,
    `rms_eps`) off a populated `Llama3Config` in one step.
    Returns `None` when the K3 sub-config or any required MLA dim
    is absent.
  - **`kimi_k3_dense_ffn_forward(x, ffn_norm, gate, up, down, eps)`**
    — SwiGLU forward for the K3 layer-0 dense FFN
    (`first_k_dense_replace = 1`). Follows the DeepSeek V3 dense-
    layer convention (Swish gate ⊙ up branch → down projection)
    since the K3 tech report leaves the dense-FFN activation
    unspecified but TENSOR_MAP.md ships the standard `ffn_gate` /
    `ffn_up` / `ffn_down` triple.
  - **`KimiK3Model::forward` update** — MLA branch delegates to
    `kimi_k3_gated_mla_step` with per-layer weight refs from the
    X.4.b.2 loader; Dense branch delegates to
    `kimi_k3_dense_ffn_forward`. Cache/attn tag mismatch (i.e. a
    KDA layer wrapping an MLA cache or vice-versa, which
    `KimiK3Model::new` should have made impossible) hits an
    explicit `panic!` naming the invariant instead of silently
    misinterpreting the weights.
- **5 new unit tests** covering the helper surfaces:
  `extract_mla_config` positive path + two failure modes
  (missing sub-config, missing individual dim), and
  `dense_ffn_forward` zero-input-gives-zero + bounded-output
  smoke tests.

### Changed

- **`model_forward_panics_at_first_kda_layer_with_todo_message`**
  → **`model_forward_panics_on_empty_layers_vec`**. The KDA
  `todo!()` message was previously the first panic
  `KimiK3Model::forward` raised because the layer loop never
  touched `weights.layers`. With X.4.c.3.3.a wiring the MLA and
  Dense branches into real per-layer weight lookups, the
  metadata-only `dummy_weights` fixture (which ships
  `layers: Vec::new()`) now panics with index-out-of-bounds
  first. The updated test asserts that behaviour and documents
  the precondition: `load_kimi_k3_model_weights` must run before
  `forward`. A full-fixture end-to-end forward test lands at
  Phase X.4.c.3.3.d when a synthetic-GGUF-with-tensors builder
  is available.

### Deferred (X.4.c.3.3.b/c + X.4.c.3.4, next sessions — still needed for real K3)

- **X.4.c.3.3.b KDA per-head aggregation**: per-head slicing from
  the fused `attn_q` / `attn_k` / `attn_v` tensors + per-head
  `kimi_delta_forward_head` (X.4.c.2 primitive) call + output
  aggregation. Quantized-per-head slicing (Q4_K / IQ1_S block
  alignment) is the tricky bit; F32 first, then quantized helper.
- **X.4.c.3.3.c Stable LatentMoE forward**: sigmoid router (with
  `noaux_tc` bias correction) top-16 from 896 experts + 2 shared
  experts fused + latent `W↓` / RMSNorm / `W↑` + SiTU-GLU per
  routed expert. Wires into layers 1..93 (K3
  `first_k_dense_replace = 1`).
- **X.4.c.3.4 Block AttnRes wiring**: replace the current simple
  residual add (`x += layer_output`) with the X.4.d.1 Block
  AttnRes primitive, plus the X.4.d.2 final aggregation via
  `output_attn_res_norm` + `output_attn_res_proj`. `attn_res_norm`
  / `attn_res_proj` / `ffn_res_norm` / `ffn_res_proj` semantics
  need pwilkin PR #26185 precise reading before wiring — the K3
  tech report §2.2 leaves the exact per-layer AttnRes projection
  underspecified relative to the GGUF tensor set.
- **Real K3 GGUF forward on Mac mini (via Tailscale)**: user
  flagged that Mac mini has enough disk to hold the ~527 GB
  GrEarl IQ1_S 94-part upload. Scheduled after X.4.c.3.3.b/c/4
  land the real forward path.

- **Phase X.4.b.2 — Kimi K3 GGUF tensor loader** (2026-07-28).
  Follows X.4.b.1 (metadata + config) with the tensor reference
  data structures + walker that turns a K3 GGUF into a fully
  categorized `KimiK3ModelWeights<'a>` bundle, one step short of
  the actual forward pass (Phase X.4.c.3). Structs:
  - **`KimiK3ModelWeights<'a>`** — Global 5-tensor bundle
    (`token_embd`, `output_norm`, `output`,
    `output_attn_res_norm`, `output_attn_res_proj`) + a per-layer
    `Vec<KimiK3LayerWeights>`.
  - **`KimiK3LayerWeights<'a>`** — 8 COMMON tensors
    (`attn_norm`, `ffn_norm`, `attn_output`, `attn_gate` +
    K3-only AttnRes `{attn,ffn}_res_{norm,proj}`) plus dispatched
    `attn: KimiK3Attention` and `ffn: KimiK3Ffn`.
  - **`enum KimiK3Attention<'a> { Mla(KimiK3MlaAttn), Kda(KimiK3KdaAttn) }`**
    — dispatched by `config.kimi_delta.is_mla_layer(il)` (K3: 24
    MLA layers `[3, 7, 11, ..., 91, 92]`, 69 KDA otherwise).
  - **`enum KimiK3Ffn<'a> { Dense { … }, LatentMoe(KimiK3LatentMoe) }`**
    — dispatched by `il < first_k_dense_replace` (K3: only
    layer 0 is Dense).
  - **`KimiK3MlaAttn<'a>`** — 7 tensors: `q_a` + `q_a_norm` +
    `q_b` (LoRA Q), `kv_a_mqa` + `kv_a_norm` (LoRA KV), split
    `k_b` + `v_b` (K3's `kv_b` is written as two half-tensors
    at conversion per TENSOR_MAP.md §"`kv_b_proj` split").
  - **`KimiK3KdaAttn<'a>`** — 11 tensors: `q` + `k` + `v` (dense
    projections), `ssm_conv1d_{q,k,v}` (ShortConv per Q/K/V),
    `ssm_f_a` + `ssm_f_b` (low-rank α path Eq 2), `ssm_beta`
    (β delta-rule strength), `ssm_norm` (RMSNorm on `ō`), and
    an optional `ssm_dt.bias` (present in some conversions).
  - **`KimiK3LatentMoe<'a>`** — 11 fields covering router
    (`ffn_gate_inp` + optional `exp_probs_b`), shared experts
    (`ffn_{gate,up,down}_shexp`), latent projections
    (`routed_exp_{up,down,norm}` — K3-only RMSNorm before `W↑`),
    and the 3-D per-expert cubes
    (`ffn_{gate,up,down}_exps`).
- **`load_kimi_k3_layer_weights(gguf, il, config)`** + wrapper
  **`load_kimi_k3_model_weights(gguf, config)`** — walk GGUF
  tensors following `Kuberwastaken/Kimi-K3-GGUF/TENSOR_MAP.md`,
  dispatching per layer type (MLA/KDA + Dense/LatentMoE),
  returning descriptive `Err(String)` on the first missing
  tensor. Uses the new **`load_weight_ref_any_shape`** helper
  which reads both `rows` and `cols` from `tensor_info.dims`,
  avoiding the fragile per-tensor shape-computation-from-config
  that MLA/KDA/LatentMoE variance would otherwise require.
- **6 new loader tests** on top of the 6 metadata tests from
  X.4.b.1: shape-helper 2D + 1D coverage, model-loader Err on
  missing globals, layer-loader Err on missing attn_norm with
  layer-prefix in the error message, MLA-vs-KDA dispatch
  predicate correctness at layers 0/3/4/7, and Dense-vs-MoE
  boundary at `first_k_dense_replace = 1`.

### Deferred (X.4.c.3, next session — makes K3 actually run)

- **`forward_kimi_k3` real implementation**: wire X.4.b.2 tensor
  refs into a 93-layer forward using the X.4.c.1 KDA primitives,
  a new Gated MLA layer forward (`q_a → q_a_norm → q_b` LoRA Q +
  NoPE + full-rank output gate), the X.4.d.1 Block AttnRes state,
  and the X.4.d.2 final aggregation (`output_attn_res_norm` +
  `output_attn_res_proj`). Landing this + real GGUF (the GrEarl
  IQ1_S upload weighs 527 GB) is what turns "K3 loads" into "K3
  runs".
- **Stable LatentMoE forward** (part of X.4.c.3): router top-16
  from 896 + shared experts + latent `W↓` / RMSNorm / `W↑` +
  SiTU-GLU per routed expert. The 3-D per-expert cube tensors
  need per-expert byte-slice indexing that plugs into the Phase
  X.4.e.1 streaming pool.

- **Phase X.4.b.1 — Kimi K3 GGUF metadata loader** (2026-07-28).
  The community `Kuberwastaken/Kimi-K3-GGUF/convert_kimi_k3.py`
  + upstream llama.cpp PR #26185 (`pwilkin/kimi-k3-text`) settled
  the GGUF metadata layout the same day as the 2026-07-27 open
  weight release; this landing wires ALICE-LLM to consume that
  layout directly. `GrEarl/Kimi-K3-GGUF` (Q2_K, 94-part) and
  `GrEarl/Kimi-K3-GGUF-IQ1_S` (527 GB, 94-part) are the live
  download targets — the loader here is what parses their
  metadata. Changes:
  - **`ModelArch::KimiK3::meta_prefix()`** — updated from the
    guessed `"kimi"` to the confirmed `"kimi-k3"` (with hyphen).
  - **`KimiDeltaConfig::from_gguf(gguf, prefix)`** — reads all
    K3 hyperparameters from the `kimi-k3.*` namespace: MLA
    sub-config (`q_lora_rank`, `kv_lora_rank`, split
    `qk_nope`/`qk_rope`/`v_head_dim`, `mla_use_nope`,
    `mla_use_output_gate`), KDA sub-config (`kda_head_dim`,
    `ssm.conv_kernel`, `use_full_rank_gate`,
    `gate_lower_bound`), Attention Residuals + SiTU-GLU
    (`attn_res_block_size`, `activation_situ_beta`,
    `activation_situ_linear_beta`), Stable LatentMoE (standard
    llama.cpp `expert_*` keys + K3-only
    `routed_expert_hidden_size` / `latent_moe_use_norm` /
    `moe_router_activation_func` / `topk_method`), and the
    hybrid-layer-routing arrays (`full_attn_layers` +
    `kda_layers`, both 0-indexed after `k3meta.py` subtracts 1
    from `config.json`'s 1-indexed lists).
  - **`Llama3Config::from_gguf` K3 branch** — dispatches to
    `KimiDeltaConfig::from_gguf` when `arch == ModelArch::KimiK3`,
    populating the previously-`None` `kimi_delta` field.
  - **`KimiDeltaConfig::is_mla_layer(il)`** — layer-index
    predicate: `Some(true)` when layer `il` is a Gated MLA
    layer, `Some(false)` when KDA, `None` when
    `full_attn_layers` is absent. Layer `il` is MLA iff
    `il ∈ full_attn_layers` (0-indexed to match the GGUF-side
    convention).
- **6 integration tests** driven off a synthetic mini K3 GGUF
  built entirely in-test (no download required): arch detection
  from `general.architecture`, full 30-field metadata parse
  parity, `is_mla_layer` behavior on both populated and empty
  configs, graceful handling of missing optional keys, and a
  regression guard on the hyphenated `meta_prefix`.

### Deferred (X.4.b.2 + X.4.c.3, next session)

- **Weight tensor loader**: per-layer tensor lookup + shape
  validation for the ~2573 tensors K3 emits (F32 + Q4_K + F16 +
  IQ1_S mix in the GrEarl IQ1_S build), following
  `Kuberwastaken/Kimi-K3-GGUF/TENSOR_MAP.md`.
- **`forward_kimi_k3` block-level integration**: wire the
  primitives shipped in X.4.c.1 + X.4.c.2 + X.4.d.1 + X.4.h.1
  together, add a Gated MLA layer forward, apply
  `output_attn_res_norm` + `output_attn_res_proj` for the final
  N-block aggregation (X.4.d.2), and produce logits.

- **Phase X.4.f.1 — MXFP4 fused scalar matvec kernel** (2026-07-28).
  Lands the fused scalar matvec that was left as a `todo!()`
  fail-fast in the 2026-07-24 MXFP4 skeleton, closing the CPU-side
  Phase X.4.f milestone one step short of the SIMD variants
  (X.4.f.2+). Changes:
  - **`mxfp4_matvec_fused_scalar()`** (private) — iterates rows and
    32-element blocks, dequantizes each block into a stack-resident
    `[f32; QK_MXFP4]` buffer via `dequantize_mxfp4_block`, and
    multiply-accumulates against the input in `f32` in the same
    element order as the correctness-first fallback. Avoids the
    per-matvec `Vec<f32>` scratch allocation.
  - **`mxfp4_matvec()`** (public free function) — `todo!()`
    replaced with a real implementation that iterates a
    `MxfP4Matrix` and dispatches per-row to the fused kernel.
  - **`quantized_matvec` routing** — `GgmlType::Mxfp4` now goes
    to `mxfp4_matvec_fused_scalar` instead of the fallback; the
    fallback becomes `#[cfg(test)]`-only, retained as the parity
    reference.
- **5 new MXFP4 unit tests** replacing the removed
  `test_mxfp4_matvec_fail_fast` (`#[should_panic]`) since the free
  function is no longer a fail-fast:
  - `mxfp4_matvec_fused_scalar_matches_fallback` — random 4×128
    matrix, bit-exact parity between fused and fallback.
  - `mxfp4_matvec_free_fn_matches_kernel` — `MxfP4Matrix` wrapper
    dispatches identically to the kernel.
  - `mxfp4_matvec_zero_input_returns_zero` — zero input → zero
    output, output pre-poisoned with NaN to catch write skips.
  - `mxfp4_matvec_unit_input_recovers_row_sum` — input of all
    ones, output equals row sum (independent oracle via
    `dequantize_row_mxfp4`).
  - `mxfp4_matvec_single_block_hand_computed` — 1 block × 1 row
    with E8M0 scale = 2.0 and all nibbles = 0x2 (E2M1 → 1.0),
    input = [1, 0, ..., 0] → output = 2.0.
- **`docs/KIMI_K3_INTEGRATION.md` phase table** — X.4.f row split
  into X.4.f.1 (完了) and X.4.f.2 (SIMD NEON / AVX2 / AVX-512
  variants, deferred).

### Deferred

- Phase X.4.f.2 (SIMD variants): NEON kernel for aarch64
  (Jetson / Mac M-series), AVX2 / AVX-512 for x86_64. Follows the
  Q1_0 / Q2_0 pattern (see `neon_dot::q1_0_dot_row_pos_only` +
  per-block sum precompute), validated against
  `mxfp4_matvec_fused_scalar` for bit-exact parity.
- Phase X.4.f.3 (PyTorch `microxcaling` oracle): blocked on
  actual K3 GGUF availability (Phase X.4.b, community
  conversion).

- **Phase X.4.d — Block Attention Residuals runtime scheme**
  (2026-07-28). Ships the paper §2.2 Eq 8-10 runtime primitives as
  a standalone module ahead of the eventual
  `forward_kimi_k3`-level integration. Block AttnRes reduces the
  `O(Ld)` memory/communication cost of Full AttnRes to `O(Nd)` by
  summing layer outputs within `N` block groups; K3 partitions its
  93 layers into 8 blocks of 12 layers each (with the last block a
  partial 9-layer block). New API:
  - **`BlockAttnResState`** — carries finalized `block_reps`
    (starting with `b_0 = h_1` = token embedding), the running
    `current_partial` for the in-progress block, and the current
    block index + within-block position. Sized in the K3 default
    at ~28 KB per finalized rep (`d = 7168 · f32`), so all 9
    reps + 1 partial ≈ 280 KB per sequence — negligible relative
    to the ~450 MB KDA head caches.
  - **`block_attnres_softmax_attention`** — the K3
    "RMSNorm-on-keys" softmax kernel from Eq 9
    (`φ(q, k) = exp(qᵀ · RMSNorm(k))`), with the standard
    subtract-max-logit trick for numerical stability, optional
    `γ` scale on the normalized keys, and `k_i = v_i` (same
    tensor for both roles, as K3 specifies).
  - **`block_attnres_layer_step`** — one-per-layer step function
    matching Eq 10. Reads the pre-update partial as the last
    entry in `V` (skipped on the first layer of a block since
    the partial would be zeros), computes the residual stream
    `h_l = softmax(Wₗᵀ · RMSNorm(v)) · v`, then adds
    `layer_output` into `current_partial` so the next step sees
    `b_n^i` as its `b_n^{(i+1)-1}`. Finalizes the block
    (pushes `current_partial` into `block_reps`, resets the
    partial to zeros, wraps `pos_in_block` back to 0) every
    `block_size` calls.
- **10 unit tests** covering (a) state init with embedding as
  `b_0`, (b) softmax with 1 key returning that key, (c)
  zero-query averaging keys uniformly, (d) dominant-key mass
  concentration, (e) γ-scaled keys giving the same 1-key
  answer (softmax normalization washes out uniform scale), (f)
  first-layer-in-block omitting the partial from V, (g)
  second-layer-in-block using the pre-update partial snapshot
  as the last V entry, (h) partial sum equalling the sum of
  layer outputs within a block, (i) block finalization at
  `block_size`, and (j) a full 2-block × 2-layer end-to-end
  walk with hand-computed finalized `block_reps`.
- **`docs/KIMI_K3_INTEGRATION.md`** phase table now has an
  X.4.d row (完了 2026-07-28) with a note that final N-block
  aggregation into logits is deferred to the integration phase
  since the paper leaves that kernel's parameterization
  underspecified.

### Deferred

- Final N-block aggregation layer (paper §2.2 "the final output
  layer aggregates all N block representations"): the exact
  aggregation formula is not spelled out in the tech report at
  the level of detail needed for a standalone unit test. Will
  land alongside `forward_kimi_k3` integration in a follow-up
  X.4.d.2 once we see the reference implementation.
- Block AttnRes integration into `forward_kimi_k3` (still
  `todo!()`): blocked on Phase X.4.b (community GGUF weight
  loader for the per-layer pseudo-queries `w_l` and the optional
  key-side RMSNorm `γ`).

- **Phase X.4.c.2 — KDA per-head composite forward** (2026-07-28).
  Wires the Phase X.4.c.1 primitives together with the existing
  shared `causal_conv1d_step` (Qwen 3.5 DeltaNet's ShortConv,
  reused unchanged) and `silu` (Swish) into a full per-token
  per-head KDA forward matching K3 tech report §2.1.1 Eq 1-6:
  - **`KimiDeltaHeadCache`** — bundles the recurrent
    [`KimiDeltaState`] and three ShortConv history ring buffers
    (Q, K, V, each `(kernel_size − 1) × dim`) into one struct so
    callers thread one mutable reference through the forward
    call. Total per-head KDA cache ≈ 68.5 KB at K3 defaults
    (state 64 KB + three conv rings 1.5 KB each); a full 96-head
    KDA layer ≈ 6.6 MB; all 69 KDA layers ≈ 454 MB —
    sequence-length invariant.
  - **`KimiDeltaHeadParams<'a>`** — borrowed per-head weight
    reference struct with 20 fields grouped by K3 tech report
    subsection (Q/K/V projections + biases + ShortConv kernels,
    β projection, α low-rank decay path, A_h, g_min, output
    gate, output projection, optional inner RMSNorm γ + eps).
    Zero-copy over GGUF-backed tensors in the production path,
    equally usable with owned `Vec<f32>` in tests.
  - **`kimi_delta_l2_norm_in_place`** — L2 normalize a slice
    in-place with `x ← x / (||x||_2 + eps)`, matching the
    llama.cpp / vLLM style that adds eps to the denominator so
    zero inputs stay zero without NaN.
  - **`kimi_delta_forward_head`** — one-token composite fn.
    Nine-step pipeline: (1) linear projections, (2) ShortConv,
    (3) Swish, (4) L2Norm on q/k, (5) β = Sigmoid(W_β · x),
    (6) α via `kimi_delta_lower_bounded_decay`, (7) recurrent
    step, (8) read `ō = Sᵀ q`, (9) output gate (Eq 6 / Eq 7).
- **10 unit tests** covering cache init/reset (2), L2Norm
  unit-length + zero-input (2), forward zero-input → zero-output,
  ring cursor advance per call, two-token state + conv ring
  progression, reset parity with fresh cache, first-token output
  boundedness (`|y| ≤ 2, finite`), and zero-gate output
  proportional to the sigmoid ratio vs baseline. All tolerances
  are 1e-3 to 1e-6.

### Deferred

- Phase X.4.c.3 (block-level integration): wiring
  `kimi_delta_forward_head` into the eventual `forward_kimi_k3`
  layer dispatcher, which requires (a) per-layer weight lookup
  from a GGUF tensor table (blocked on Phase X.4.b, community
  `convert_hf_to_gguf.py`), (b) the Block Attention Residuals
  Eq 8-10 wiring (Phase X.4.d), and (c) an MLA-specific KV
  cache + NoPE + output-gate variant for the 24 Gated MLA
  layers.
- Chunkwise parallel form for prefill (Eq 3-4, 16-token tiles):
  decode-only path is what X.4.c.2 provides; prefill throughput
  is a post-integration optimization.
- SIMD / GPU kernel for `kimi_delta_forward_head`: the CPU
  reference is what the WGSL / Metal shader will be
  bit-for-bit validated against once the numerics settle.

- **Phase X.4.c.1 — KDA CPU forward primitives scaffold**
  (2026-07-28). Ships the Kimi K3 Kimi Delta Attention math
  primitives from the tech report §2.1.1 as standalone,
  unit-testable Rust functions ahead of the block-level integration
  (X.4.c.2). New public API:
  - **`KimiDeltaState`** — per-head recurrent state
    `S ∈ ℝ^{d_k × d_v}` as a row-major flat `Vec<f32>`. Sequence-
    length invariant (K3 default 128 × 128 = 64 KB / head; 96 heads
    × 69 KDA layers ≈ 421 MB total across a full model regardless
    of context length).
  - **`kimi_delta_step`** (Eq 1) — in-place recurrence
    `S_t = (I − β k kᵀ) Diag(α) S_{t−1} + β k vᵀ`. Fused 3-pass
    implementation (row-scale by α → aggregate `w = kᵀ S` →
    `S += β k (v − w)ᵀ`) with minor short-circuits when `k[i] == 0`
    or `coef == 0`.
  - **`kimi_delta_read`** — `ō_t = Sᵀ q_t ∈ ℝ^{d_v}` read-out.
  - **`kimi_delta_lower_bounded_decay`** (Eq 5) —
    `g = g_min · Sigmoid(exp(A_h) · z), α = exp(g) ∈
    (exp(g_min), 1)^{d_k}`. K3 fixes `g_min = -5` to keep the
    cumulative log-decay over a 16-token tile inside `(-80, 0)`, so
    KDA's diagonal and off-diagonal tiles can both use dense
    Tensor Core matmul — the key departure from Kimi Linear's
    unbounded negative-Softplus mapping.
  - **`kimi_delta_output_gate`** (Eq 6/7) —
    `y = W_o [Sigmoid(W_g x) ⊙ RMSNorm(ō)]`. Covers both the KDA
    variant (`rms_weight = Some(γ)`) and the Gated MLA variant
    (`rms_weight = None`, Eq 7 which omits the inner RMSNorm on
    `ō_t`).
- **14 unit tests** covering state init/reset (2), recurrence with
  β=0 / α=0 / α=β=1 from zero / two orthogonal writes (4), read
  from zero + read after single write (2), decay saturation at
  z=0 / z→+∞ / z→−∞ (3), and output gate for zero-gate / zero-ō /
  no-RMSNorm variant (3). All tolerances are 1e-4 or exact bit
  equality; the recurrence tests hand-compute the expected 2×2 and
  4×4 state matrices.
- **`docs/KIMI_K3_INTEGRATION.md`** gains a "Confirmed via tech
  report" section that transcribes every equation the ALICE-LLM
  Phase X.4.c/d/e/h implementation needs — KDA (Eq 1-6), Gated MLA
  (Eq 7), Block AttnRes (Eq 8-10), Stable LatentMoE (Eq 11), SiTU-
  GLU (Eq 12), Quantile Balancing (Eq 13-14), the MXFP4 QAT
  scope, the pretrain-time MTP head, and the KDA-aware unified
  prefix cache from §5.4.1. "What we still DON'T know" is rewritten
  to reflect that the only remaining blockers post-paper-read are
  (a) the multimodal fusion timing (scheduled at X.4.i) and (b)
  the community GGUF metadata prefix (external dependency, X.4.b).

### Deferred

- Phase X.4.c.2 (block-level integration): wiring the primitives
  above into `forward_kimi_k3`, including per-head projection with
  ShortConv (kernel 4) history buffers, chunkwise parallel form
  (Eq 3-4, 16-token tiles) for prefill, and interaction with the
  KV cache for the 24 Gated MLA layers. Blocked on Phase X.4.b
  (GGUF metadata + weight tensor mapping, community-conversion
  dependent) and the ShortConv primitive lift from the existing
  Bonsai `gated_deltanet_step*` codepath.

- **Phase X.4.e.1 — Kimi K3 streaming pool infrastructure**
  (2026-07-28). `deepseek_streaming.rs` is generalised from
  "DeepSeek-V3 only" to "DeepSeek-V3 / Kimi K3-family sparse-MoE"
  documentation without touching any struct definitions — the LRU
  cache, `ExpertLayerInfo`, `StreamingExpertPool`, and
  `PersistenceHeuristic` were already expert-count and top-k
  agnostic, so the K3 topology (896 routed experts, top-16, 2
  shared, Stable LatentMoE) drops in by construction. New sizing
  helpers landed with the paper-anchored formula:
  - **`kimi_k3_active_bytes(...)`** — `const fn` returning the
    per-token active byte footprint under a K3-family config,
    parameterised by `num_moe_layers × num_experts_per_tok × 3
    slabs × latent_hidden × moe_intermediate ×
    bytes_per_weight`. Encodes the paper's ≈ 24 GB estimate (92
    layers × 16 experts × 3 slabs × 3584 × 3072 × 0.5 byte/Q4).
    The `bytes_per_weight` argument is fixed-point (`×100`) so a
    quantization table can be encoded without `f32` in a public
    API.
  - **`recommended_budget_bytes(active_bytes,
    safety_multiplier_x10)`** — companion helper that scales
    `active_bytes` by a fixed-point multiplier for LRU cache
    sizing (default 1.2× gives 24 → 30 GB budget for K3 on Mac M3
    Max 128 GB unified memory).
- **Four Kimi K3 unit tests** covering (a) sizing-helper
  correctness against the paper's ≈ 24 GB estimate, (b)
  recommended-budget arithmetic at three safety multipliers, (c)
  end-to-end pool dispatch with 896 experts + top-16 fetch + LRU
  hit accounting, and (d) `PersistenceHeuristic::predict` scaling
  to a 896-length logits vector with top-16 selection. Also adds
  a boundary-guard test that expert index 896 (equal to
  `n_experts`) panics rather than serving garbage from an
  adjacent slab.
- **Doc sync** — `docs/KIMI_K3_INTEGRATION.md` gains an
  **X.4.e.1** row (this session, complete) ahead of the original
  **X.4.e** (now scoped to "connect the pool infra to real K3
  GGUF + implement the K3 LatentMoE + RMSNorm + SiTU-GLU forward
  path in `forward_deepseek_moe_layer`", still blocked on X.4.b
  and X.4.c).

- **Phase X.4.a.1 — Kimi K3 post-release spec refinement**
  (2026-07-28). Moonshot's Kimi K3 open weight release landed on
  schedule 2026-07-27; the full `text_config` structure is now
  captured by [`KimiDeltaConfig`], expanded from 10 placeholder
  `Option` fields to 32 concretely-typed fields grouped into 6
  sub-sections (hybrid attention routing, Gated MLA, Attention
  Residuals, SiTU-GLU, Stable LatentMoE 896/16, MXFP4 native
  quantization). All values are cross-checked against the published
  `config.json` at
  `huggingface.co/moonshotai/Kimi-K3/raw/main/config.json`.
- **`KimiDeltaConfig::from_hf_config`** — direct HuggingFace
  `config.json` loader gated behind the new `hf-config` Cargo feature
  (`hf-config = ["dep:serde", "dep:serde_json"]`). Parses
  `text_config.*` including the nested `linear_attn_config` (24 MLA
  layers `[4, 8, ..., 92, 93]` + 69 KDA layers) and the
  `quantization_config.config_groups.group_0.weights` (MXFP4 group
  size 32, 4 bits/weight). Individual missing fields degrade to
  `None` rather than erroring, so the loader still yields a usable
  partial config for pre-release checkpoint variants. Five unit
  tests cover: full-fixture parity with the 2026-07-27 spec, missing
  fields, missing `text_config`, malformed JSON, and truncated
  hybrid-layer arrays. This unblocks the direct-safetensors path
  ahead of community GGUF conversion (Phase X.4.b).
- **Doc sync** — `docs/KIMI_K3_INTEGRATION.md` gains a
  "Confirmed via HF config.json" table listing all newly-resolved
  numeric spec unknowns; "What we DON'T know" is rewritten to only
  list paper-drop dependencies (KDA gate formula, AttnRes runtime
  scheme, GGUF metadata prefix, multimodal fusion path, 1M context
  KV compression for the 24 MLA layers). Integration Phase table
  now lists X.4.a.1 as complete.

### Changed

- **`ModelArch::KimiK3` doc comment** — updated to reflect the
  released spec (2.8T total / 104B active, 896 experts top-16, 93
  layers = 69 KDA + 24 Gated MLA + 1 dense, hidden 7168, 1M context,
  native MXFP4). The `todo!()` fail-fast in `forward_kimi_k3` and
  its dispatch site now point to Phase X.4.c (CPU forward) with
  Phase X.4.b (community GGUF conversion) as the outstanding
  blocker, rather than the initial weight release.

### Notes

- The `todo!()` in `forward_kimi_k3` remains intentional per
  CLAUDE.md's "仮実装完了偽装の禁止" rule. Downstream users who
  feed a Kimi K3 GGUF (once the community conversion lands) will
  hit an explicit panic pointing to `docs/KIMI_K3_INTEGRATION.md`
  rather than silent garbage from the vanilla-attention path.
- The `hf-config` feature only pulls in `serde` + `serde_json`
  (both already declared optional at the workspace level). The
  default build surface is unchanged; enable with
  `cargo build --features hf-config`.

## [1.6.0] - 2026-07-26

### Added

- **`GpuModel::forward_with_early_exit_no_read`** — no-read companion
  to `forward_with_early_exit_and_read` (added in v1.5.0). Runs the
  first `early_exit_layer` transformer layers on GPU and advances the
  KV cache identically to the `_and_read` variant, but skips the
  output-head compute (RMSNorm + output projection) AND the CPU↔GPU
  logits readback. Intended for prefill workloads where only the last
  token's logits matter, or for external-signal-driven decode where a
  gate has committed to not emitting this token. Downstream callers
  can now mix `_and_read` (when logits are needed) and `_no_read`
  (when only KV state advance is needed) to avoid paying the dominant
  per-token readback cost on latency-bound workloads.
- **`GpuModel::forward_with_surprise_no_read`** — signature-parity
  monotonic-gate adapter over `forward_with_early_exit_no_read`,
  matching the CPU-side `Llama3Model::forward_with_surprise` gate
  closure signature. Same semantic contract as
  `forward_with_surprise_and_read` (monotonic gate in `layer_idx`,
  collapses per-layer gate decisions into a single early-exit depth
  per token). Delegates to `forward_with_early_exit_no_read` after
  the CPU-side gate scan.

### Rationale

The v1.5.0 `_and_read` variants always pay the full output-head
compute + logits readback per invocation. On workloads where the
caller processes many tokens but only reads logits for a subset —
prefill of an N-token prompt (only token N's logits drive sampling)
or a surprise-driven consumer that emits selectively — the readback
dominates end-to-end latency. The `_no_read` variants let callers
preserve KV cache correctness without paying the readback cost, and
compose freely with the `_and_read` variants on a per-token basis.

Additive API only — v1.5.0 `_and_read` variants unchanged.
Downstream crates can adopt `_no_read` incrementally.

## [1.5.0] - 2026-07-26

### Added

- **`GpuModel::forward_with_early_exit_and_read`** — GPU forward
  primitive that runs the first `early_exit_layer` transformer
  layers on GPU, then dispatches the output head against the
  hidden state at that depth. Reads logits back and returns them
  as `Vec<f32>` (same shape as `forward_and_read`). One GPU
  command submission per token, no per-layer CPU↔GPU round trip.
  Enables single-early-exit patterns without paying the full
  N-layer forward cost when the gate closure would have skipped
  the remaining layers on the CPU path.
- **`GpuModel::forward_with_surprise_and_read`** — signature-parity
  adapter over `forward_with_early_exit_and_read` that matches
  the CPU-side `Llama3Model::forward_with_surprise` signature
  (accepts a monotonic gate closure and `Option<SurpriseVec<'_>>`).
  Internally CPU-scans the gate to compute the early-exit depth,
  then delegates to the primary API.
- Both APIs enable a GPU forward path that preserves the
  external-signal-driven per-layer routing semantics of
  `Llama3Model::forward_with_surprise` (v1.4.0) at the "how deep
  to go" granularity, without needing a general per-layer hook
  GPU API that would pay round-trip cost per layer.

### Removed

- **`Llama3Model::forward_with_surprise_gpu`** +
  **`Llama3Model::forward_with_early_exit_gpu`** signature-only
  skeletons (added in the same Unreleased window). The GPU
  forward path lives on `GpuModel` in this crate, not on
  `Llama3Model`; the earlier skeleton placement on `Llama3Model`
  was based on an incorrect assumption about where GPU forward
  lives. Skeletons never released to crates.io — the actual API
  landed on `GpuModel` in the same Unreleased window (see
  Added). Downstream callers should use
  `GpuModel::forward_with_early_exit_and_read` /
  `GpuModel::forward_with_surprise_and_read` from the outset.

### Changed

- **`examples/external_signal_routing.rs`** — CLI `--mode` now
  accepts `baseline | routed | both` (previous mode label updated
  for vocabulary alignment with the example's documented
  external-signal-driven routing intent). Public API unchanged
  (`forward_with_surprise` / `SurpriseVec` untouched); no new
  crates.io release is required.

## [1.4.0] - 2026-07-25

### Added

- **`Llama3Model::forward_with_surprise`** — external-signal-driven
  per-layer routing convenience API. Thin wrapper around
  `forward_with_layer_hook` that standardises the pattern
  "an external per-token signal drives per-layer routing decisions"
  already demonstrated in `examples/early_exit_qwen35.rs` (variance
  gate) and `examples/entropy_mod_qwen35.rs` (per-layer statistic
  observation). The caller supplies an optional `SurpriseVec<'_>`
  slice (per-body, per-region, per-modality, aggregated scalar
  broadcast, etc.) and a `gate: Fn(usize, Option<SurpriseVec<'_>>) -> bool`
  closure that returns `true` to skip a layer's CPU compute.
  - `pub type SurpriseVec<'a> = &'a [f32];` type alias — intentionally
    unopinionated; the caller owns the slice's shape and meaning.
  - Backward-compatibility guarantee: `forward_with_surprise(id, None,
    |_, _| false)` is bit-exact identical to `forward(id)` (both
    reduce to `forward_with_layer_hook(id, |_, _| false)`).
  - Determinism: with fixed `surprise` contents and a deterministic
    `gate` closure, output is bit-exact reproducible across runs on
    the same hardware.
  - Additive API only — existing `forward` and
    `forward_with_layer_hook` are untouched.
  - Fits the same adaptive-compute-gate pattern shown in
    `examples/early_exit_qwen35.rs` (in-hook variance gate) and
    `examples/entropy_mod_qwen35.rs` (per-layer statistic observation),
    generalised so the routing signal can originate outside the LLM.
- **`examples/external_signal_routing.rs`** — new example demonstrating
  the external-signal pattern. Structural variant of
  `early_exit_qwen35.rs` where the routing signal is a per-token
  vector produced outside the LLM rather than an internally-computed
  layer statistic. Includes a startup pass that verifies
  `forward_with_surprise(id, None, ..)` matches `forward(id)`
  bit-exact on a real loaded model.

## [1.3.0] - 2026-07-23

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
