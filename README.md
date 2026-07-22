# ALICE-LLM

**English** | [日本語](README_JP.md)

Pure Rust LLM inference engine focused on understanding and optimizing every layer of inference—from GGUF parsing to SIMD, GPU kernels, speculative decoding, and hybrid architectures. Built as a research and engineering project rather than a wrapper around existing ML frameworks.

GGUF quantized models, zero external ML dependencies, 326 tests.

**GPU (wgpu/Metal): 125ms → 71ms/token (1B), batch-4 speculative: 1B draft + 8B verify = 5.89× speedup, 90% accept rate.**

**Bonsai 27B Q1_0 (1.125 bpw binary, 3.6 GB GGUF) on Apple M3 Metal: coherent generation at 1.1 tok/s (Phase X.3.e.3.27 Q1_0 fused SwiGLU + attn_q per-head interleaved layout fix).**

**Qwen 3.5-4B hybrid (DeltaNet + Full Attention) on Apple M3 Metal: coherent generation at 2.9 tok/s (Q4_K_M mixed quant with Q5_K/Q8_0 shader coverage).**

**Phase X.3.e.3.37 (2026-07-21): o_proj cols dimension fix for Qwen 3.5+ hybrid architectures where q_dim = num_heads × head_dim ≠ hidden_dim. Previous hardcoded hidden_dim in `upload_w` produced near-orthogonal o_proj output (cos 0.118) for Qwen 3.5-4B. One-line fix restored cross-arch coherence: L3 pos 17 hidden cos 0.7057 → 0.9970 across all positions.**

**CPU: 0.16 → 1.76 tok/s (11x) on 70B sparse ternary — hitting 45% memory bandwidth on M1 Pro.**

**x86_64 SIMD (2026-07): Q4_K / Q5_K / Q6_K / Q8_0 / Ternary all get AVX2 and AVX-512BW/F kernels with runtime dispatch, matching the existing NEON parity on Apple Silicon.**

**Per-layer hybrid (`--hybrid-per-layer`): CPU processes DeltaNet layers + GPU processes Attention layers with per-token hidden-state shuttle. Intermediate speed between pure GPU and pure CPU while bypassing full-model GPU allocation.**

**Jetson Orin Nano 8GB (Vulkan iGPU): Qwen 3.5-4B hybrid at 0.4 tok/s answering "The capital of Japan is Tokyo. It is the country's capital, largest city," — Phase X.3.e.3.37 o_proj weight upload cols dim fix restored the qwen35 hybrid arch (previously loaded o_proj as [hidden_dim, hidden_dim] instead of [hidden_dim, q_dim], truncating 37.5% of Q4_K weight bytes for models where q_dim ≠ hidden_dim). `attention_only_load` skips DeltaNet weight upload so the whole hybrid fits under the Vulkan 2×-duplication budget (Phase X.3.e.3.29).**

**Ornith-1.0-9B (DeepReinforce, MIT, Qwen 3.5 fine-tune for agentic coding): verified load-and-run across Apple M3 CPU (1.8 tok/s), Apple M3 Metal iGPU (2.1 tok/s), and Jetson Orin Nano 8GB CPU (2.3 tok/s) with zero config — arch auto-detected from `general.architecture = qwen35`, all Phase X.3.e.3.14-29 CPU/GPU fixes cascade cleanly to the fine-tune.**

**Jetson multi-model support (2026-07-21 verified on Yahboom Orin Nano 8GB)**: Qwen 3.5-4B Q4_K_M `--hybrid-per-layer` (GPU+CPU) 0.4 tok/s, Ornith 9B Q4_K_M `--hybrid` (pure CPU) 0.2 tok/s, Bonsai 27B Q1_0 `--hybrid` 0.1 tok/s, DeepSeek V2-Lite Q4_K_M (deepseek2 arch, MoE 64 experts / 6 active per token) CPU 0.1 tok/s — 4B–27B model class runs on 8GB unified memory via CPU delegate path when full GPU allocation exceeds the wgpu-hal Vulkan 2×-duplication budget.**

## Quick Start

```bash
# Download model
huggingface-cli download elyza/Llama-3-ELYZA-JP-8B-GGUF \
  Llama-3-ELYZA-JP-8B-q4_k_m.gguf --local-dir models/

# Run inference
cargo run --release --example elyza_gguf --features "gguf,parallel" -- \
  --model models/Llama-3-ELYZA-JP-8B-q4_k_m.gguf \
  --prompt "日本の首都は" \
  --max-tokens 100 \
  --temperature 0.0
```

```
日本の首都は東京です。
Tokens: 8 generated, 16 prompt
Speed: 5.9 tok/s (4434 prefill + 1432 decode = 5883 total ms)
```

## Desktop App — ALICE-LLM Studio

[ALICE-LLM Studio](https://github.com/ext-sakamoro/ALICE-LLM-Studio) is a companion desktop app that wraps this engine in a Tauri GUI: an embedded `alice-llm-server` sidecar, a HuggingFace GGUF browser with per-file streaming download and hardware-fit hints (🟢 comfortable / 🟡 tight / 🟠 hybrid / 🔴 oversized), and an Ollama-style chat pane with `max_tokens` / `temperature` controls. Local models live under `~/.alice-llm-studio/models/`.

**Latest release: [v0.1.0-alpha](https://github.com/ext-sakamoro/ALICE-LLM-Studio/releases/tag/v0.1.0-alpha)** (2026-07-22, embeds ALICE-LLM v1.2.1). Signed installers are not shipped yet, so the first launch may need a Gatekeeper / SmartScreen bypass.

| Platform | Download |
|---|---|
| macOS (Apple Silicon) | [`.dmg`](https://github.com/ext-sakamoro/ALICE-LLM-Studio/releases/download/v0.1.0-alpha/alice-llm-studio-v0.1.0-alpha-aarch64-apple-darwin.dmg) |
| macOS (Intel) | [`.dmg`](https://github.com/ext-sakamoro/ALICE-LLM-Studio/releases/download/v0.1.0-alpha/alice-llm-studio-v0.1.0-alpha-x86_64-apple-darwin.dmg) |
| Linux x86_64 | [`.AppImage`](https://github.com/ext-sakamoro/ALICE-LLM-Studio/releases/download/v0.1.0-alpha/alice-llm-studio-v0.1.0-alpha-x86_64-unknown-linux-gnu.AppImage) |
| Windows x86_64 | [`.msi`](https://github.com/ext-sakamoro/ALICE-LLM-Studio/releases/download/v0.1.0-alpha/alice-llm-studio-v0.1.0-alpha-x86_64-pc-windows-msvc.msi) |

Chat is non-streaming in this alpha because the embedded sidecar predates the OpenAI SSE surface — the hardware-fit hints are advisory since the shipped `alice-llm-server` doesn't yet accept `--hybrid` / `--hybrid-per-layer` from the GUI. Both gaps are tracked upstream; the CLI examples below expose the full feature surface today.

## Features

- **GGUF v3 parser** — zero-copy mmap weight loading
- **Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_0 / F16 / F32** quantization (all GGML K-quant types)
- **llama.cpp-compatible** — Q2_K–Q6_K×Q8_K integer dot product (matches `ggml_vec_dot_q*_K_q8_K`, ±0.03 logits)
- **Multi-architecture** — Llama-3/3.1/3.2, Mistral (sliding window), Gemma-2 (softcapping), **Qwen 2 / 2.5 (QKV bias), Qwen 3 (per-head QK RMSNorm), Qwen 3.5 hybrid (DeltaNet linear attention + full attention; downstream fine-tunes like Ornith-1.0 verified), Gemma 3n (Laurel / AltUp / per-layer embedding), Gemma 4 (SWA half head_dim), MoE (Qwen 3 MoE / Mixtral / Gemma 4 26B_A4B)** — all auto-detected
- **DeltaNet CPU forward** — Qwen 3.5 Gated Linear Attention (causal conv1d + gated delta rule + recurrent state), matches the WGSL GPU shader bit-for-bit
- **Tied embeddings** — Llama-3.2-1B/3B output projection via quantized `token_embd.weight` (Q6_K matvec)
- **BPE tokenizer** — GPT-2 byte encoding from GGUF metadata, named constants for GGUF `token_type` and SentencePiece byte-fallback
- **Contiguous + Paged KV cache** — flat buffer with rollback, or 16-tok/page on-demand allocation
- **Continuous batching** — round-robin scheduler, per-request PagedKvCache
- **Speculative decoding** — layer-skip draft + dual-model (1B→8B) + probabilistic sampling (Leviathan et al.)
- **RoPE frequency scaling** — NTK-aware context extension via `rope_freqs.weight` tensor (Llama-3.1/3.2)
- **LLVM auto-vectorization** — `target-cpu=native` generates ARM SDOT instructions (37 sdot in Q4_K dot product)
- **x86_64 SIMD (AVX2 + AVX-512BW/F)** — Q4_K / Q5_K / Q6_K / Q8_0 / Ternary dot products with runtime CPU-feature dispatch (`is_x86_feature_detected!` cached in `OnceLock`); AVX-512 uses `__mmask16` for the Ternary bitmask path
- **Sparse ternary** — N:M structured sparsity, packed 2-bit, LUT+SDOT optimized, block-packed layout
- **GPU inference (wgpu)** — Metal/Vulkan/DX12 compute shaders, Q4_K / **Q5_K** / Q6_K / **Q8_0** / **Q1_0** dequant-fused matvec, **fused SwiGLU (Q4_K and Q1_0 variants)**, batch-4 speculative decoding, zero per-token allocation, subgroup SIMD reduction, **DeltaNet SSM path (alpha / beta / conv1d / gated delta rule with Bonsai Gap-B refinement) for Qwen 3.5 / Qwen 3.6 / Bonsai 27B hybrid**
- **Per-layer hybrid execution** — `--hybrid-per-layer` orchestrator + `Llama3Model::forward_with_layer_hook` (CPU) + `GpuModel::run_attention_layer_only` (GPU) split DeltaNet layers on CPU and Attention layers on GPU, exchanging hidden state per token via `write_f32` / `read_f32`. Bypasses the wgpu-hal Vulkan weight-duplication penalty on Jetson-class unified-memory targets when combined with a future attention-only load path.
- **Ternary QAT** — STE, L1 regularization, AdamW, layerwise mixed precision
- **God-object–free config** — `Llama3Config` (38 → 16 fields) and `LayerWeights` (33 → 17 fields) split into cohesive sub-structs (`AttentionExtrasConfig`, `SsmDeltaNetConfig`, `MoeConfig`, `Gemma3nConfig`, `Gemma4Config`, `QwenAttentionBiases`, `QwenAttentionNorms`, `Gemma3nLayerAugmentations`, `MoeExpertWeights`) with backward-compat accessor methods
- **Flat cache-aligned `Matrix<T>`** — single contiguous `Vec<T>` row-major layout replaces `Vec<Vec<T>>`, `matmul_flat` uses FMA `mul_add` on a `j`-contiguous inner loop for cache-line coalescing
- **Partial-sort sampling** — `top_k_filter` uses `select_nth_unstable_by` for O(N) K-th selection, keeps large vocabularies (100k+) affordable
- **Optional SIMD (`--features simd`)** — `wide::f32x8` chunked dot product with FMA `mul_add` and scalar tail handling; ARM NEON and x86_64 AVX2 both routed through `wide`

## Architecture

```
src/
├── lib.rs       — BPE tokenizer, KV cache, attention, RoPE, sampling
├── gguf.rs      — GGUF v3 parser, Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K quantization, fused matvec,
│                  ternary/sparse-ternary matrices, LLVM auto-vectorized SDOT kernels
├── gpu.rs       — wgpu compute engine: GpuEngine, GpuModel, batch-4 speculative decoding,
│                  zero per-token allocation, subgroup SIMD, Rc-shared engine for dual-model
├── shaders/     — WGSL compute shaders (Q4_K/Q6_K matvec, fused SwiGLU, RMSNorm, RoPE,
│                  attention, KV cache, residual — each with K=1 scalar + K=4 batch variant)
├── llama3.rs    — Multi-arch forward pass, model loading, speculative decoding (layer-skip + dual-model),
│                  paged KV cache, continuous batching, tied embeddings, RoPE freq scaling
└── training.rs  — Ternary QAT: STE, QatLinear, L1 regularization, AdamW, MSE loss

.cargo/
└── config.toml  — rustflags: target-cpu=native (enables LLVM SDOT auto-vectorization)
```

---

## GPU Inference (wgpu / Metal)

Pure Rust GPU inference via wgpu compute shaders. Zero external ML frameworks — all kernels hand-written in WGSL.

### Quick Start

```bash
# Single-token autoregressive generation
cargo run --example generate_gpu --features gpu,gguf --release -- \
  --prompt "The meaning of life is" --max-tokens 64

# Dual-model speculative decoding (1B draft + 8B verify)
cargo run --example speculative_dual_gpu --features gpu,gguf --release -- \
  --prompt "What is the capital of Japan?" --max-tokens 64

# Qwen 3.5-4B or Bonsai 27B hybrid (DeltaNet + Attention) with full GPU forward
cargo run --example qwen_gpu --features "gpu,gguf" --release -- \
  --model models/Qwen3.5-4B-Q4_K_M.gguf \
  --prompt "The capital of Japan is" --max-tokens 40

# Per-layer hybrid (CPU DeltaNet + GPU Attention) — Phase A2
cargo run --example qwen_gpu --features "gpu,gguf" --release -- \
  --model models/Bonsai-27B-Q1_0.gguf \
  --prompt "The capital of Japan is" --max-tokens 40 \
  --hybrid-per-layer --max-seq-len 512

# CPU delegate hybrid (skips GPU entirely, mmap zero-copy) — Jetson-friendly
cargo run --example qwen_gpu --features "gpu,gguf" --release -- \
  --model models/Bonsai-27B-Q1_0.gguf \
  --prompt "The capital of Japan is" --max-tokens 40 --hybrid
```

### The GPU Optimization Journey (Apple M3 Metal, Llama-3.2-1B Q4_K_M)

```
                        ┌─────────────────────────────────────────┐
  71ms ────────────────▶│████████████████████████████  Phase 6    │ 14.0 tok/s
                        │                                         │
  84ms ────────────────▶│████████████████████████      Phase 5    │ 11.9 tok/s
                        │                                         │
  94ms ────────────────▶│██████████████████████        Phase 4    │ 10.6 tok/s
  97ms ────────────────▶│█████████████████████         Phase 3    │ 10.3 tok/s
 104ms ────────────────▶│███████████████████           Phase 2    │  9.6 tok/s
 108ms ────────────────▶│██████████████████            Phase 1.5  │  9.3 tok/s
 125ms ────────────────▶│████████████████              Phase 1    │  8.0 tok/s
                        └─────────────────────────────────────────┘
                         125    100     80      60   ms/token
```

| Phase | Technique | ms/token | Speedup | Key Insight |
|---|---|---|---|---|
| 1 | Naive GPU matvec (Q4_K per-element) | 125 | 1.0x | GPU compute bound on dequant overhead |
| 1.5 | Fused dequant-matvec (single kernel) | 108 | 1.16x | Eliminate intermediate buffer: weight decode + FMA in one pass |
| 2 | Subgroup SIMD reduction (`subgroupAdd`) | 104 | 1.20x | Register-to-register shuffle replaces shared memory reduction |
| 3 | RMSNorm + RoPE batch-aware | 97 | 1.29x | Zero K=1 regression with batch-capable shaders |
| 4 | Fused SwiGLU (gate+up+silu×mul) | 94 | 1.33x | Single kernel replaces 3 dispatches, weight read shared |
| 5 | 2D dispatch (row > 65535 fix) | 84 | 1.49x | `wid.y * grid_x + wid.x` unlocks large output projections |
| **6** | **Zero per-token allocation** | **71** | **1.76x** | **All 167 bind groups pre-cached at load time** |

### Batch-4 Speculative Decoding

K=4 unrolled scalar accumulators — weight decoded once, 4 FMAs in registers:

```
K=1 (scalar):    71 ms/token   (14.0 tok/s)
K=4 (batch):    101 ms/4 tokens (25.3 ms/token, 39.5 tok/s)
Speedup:        2.82×
Correctness:    max|diff| = 0.000000 (PASS)
```

**Dual pipeline architecture**: Original scalar shaders (K=1 fast path) + K=4 unrolled batch shaders. No K=1 regression — `var acc0, acc1, acc2, acc3` guarantees physical GPU registers (vs `var acc: array<f32, 4>` which spills to thread memory).

### Dual-Model GPU Speculative Decoding (1B → 8B)

```
8B Baseline (K=1):     0.4 tok/s  (2,773 ms/token)
1B+8B Speculative:     2.1 tok/s  (471 ms/token)
Speedup:               5.89×
Accept rate:           90% (19/21)
```

Both models share a single `GpuEngine` (`Rc<GpuEngine>`) — same wgpu Device/Queue, independent KV caches.

### Bonsai 27B Q1_0 on Jetson Orin Nano 8GB (Vulkan iGPU)

**End-to-end status (Phase X.3.e.3.22-3.29, 2026-07-17)**

- **Mac M3 Metal**: full GPU forward runs Bonsai 27B Q1_0 end-to-end at 1.1 tok/s, generating coherent English + LaTeX. Fixed by shipping Q1_0 fused SwiGLU (`swiglu_fused_q1_0.wgsl`), the attn_q per-head interleaved layout de-interleave in `upload_w_bonsai_split`, and Q5_K/Q8_0 dequant kernels for Qwen 3.5-4B's mixed-quant weights.
- **Jetson Orin Nano 8GB (Tegra iGPU, Vulkan) — Qwen 3.5-4B**: **coherent generation at 0.3 tok/s via `--hybrid-per-layer` + `attention_only_load`** (Phase X.3.e.3.29). CPU processes 24 DeltaNet layers, GPU processes 8 Attention layers, DeltaNet weights are skipped from the GPU upload so the total footprint fits under Vulkan's 2×-duplication budget. `"I'm not sure what the user is asking about, and I'm not"` in 50 s / 15 tokens.
- **Jetson Orin Nano 8GB (Tegra iGPU, Vulkan) — Bonsai 27B**: `--hybrid` (Phase X.3.e.3.17 CPU-delegate MVP) generates `"The capital of Japan is **Tokyo**."` at ~0.09 tok/s. `--hybrid-per-layer` still exceeds the memory budget because 3.6 GB CPU model + 3.8 GB attention-only GPU × 2 duplication (7.6 GB) leaves nothing for headroom; llama.cpp Vulkan with unified-memory zero-copy is the recommended path for Bonsai 27B on Jetson today, or wait for `wgpu-hal` Vulkan zero-copy upstream.
- **Per-layer hybrid on Mac M3 Metal**: coherent at 1.2 tok/s. Weight upload cut from 2338 ms to 1059 ms once DeltaNet weights are skipped (Phase X.3.e.3.29).

Micro-benchmark: ALICE-LLM's Q1_0 wgpu path takes the largest layer matvec (`blk.0.attn_qkv.weight`, 10240 × 5120, 7.03 MB) from CPU NEON 20 ms down to GPU Vulkan sub-ms:

| Step | Kernel | µs/matvec | vs CPU | vs prev | Bandwidth util |
|:---:|:---|---:|---:|---:|---:|
| — | CPU NEON pos-only baseline | 20,120 | 1.0× | — | — |
| 1 | wgpu single-matvec | 4,369 | 4.61× | — | 1.61 GB/s (2.4%) |
| **6** | **wgpu row4 (batch=1)** | **2,407** | **8.36×** | **1.82×** | **3.06 GB/s (4.5%)** |
| 2 | wgpu batch4 (speculative) | 1,014 | 19.84× | — | 6.94 GB/s (10.2%) |
| 4 | wgpu row4×batch4 (speculative) | 764 | 26.35× | 1.33× | 9.67 GB/s (14.2%) |
| 5 | wgpu row8×batch4 (speculative) | 741 | 27.15× | 1.03× | 9.95 GB/s (14.6%) |

Theoretical ceiling: 103 µs / matvec at 68 GB/s LPDDR5 peak. Parity vs CPU is bit-exact (L2_rel = 6e-7) across all six kernels.

The row-batching approach: 1 workgroup produces N output rows sharing one input read, cutting workgroup count and input-read traffic by N×. Diminishing returns confirmed at row8 (1.03× over row4), so 4 rows / workgroup is the sweet spot for Q1_0 on Ampere iGPU.

### WGSL Shader Architecture

| Shader | K=1 (scalar) | K=4 (batch) | Bindings |
|---|---|---|---|
| `dequant_matvec_q4k` | 1 acc, 1 subgroupAdd | 4 acc, 4 subgroupAdd | weights, input, output, params |
| `dequant_matvec_q6k` | 1 acc, 1 subgroupAdd | 4 acc, 4 subgroupAdd | weights, input, output, params |
| `dequant_matvec_q1_0` | 1 acc (batch=1) | 4 acc (batch4) | weights, input, output, params |
| `dequant_matvec_q1_0_row4` | 4 acc (4 rows / workgroup, batch=1) | — | weights, input, output, params |
| `dequant_matvec_q1_0_row4_batch4` | — | 16 acc (4 rows × 4 batches) | weights, input, output, params |
| `dequant_matvec_q1_0_row8_batch4` | — | 32 acc (8 rows × 4 batches) | weights, input, output, params |
| `swiglu_fused_q4k` | 2 acc (gate+up) | 8 acc (gate×4+up×4) | gate_w, up_w, input, output, params |
| `rmsnorm` | batch via `wid.x` | — | input, weights, output, params |
| `rope` | batch via `wid.y` | — | data, params |
| `attention` | batch via `wg.y` (causal) | — | Q, K_cache, V_cache, output, params |
| `kv_cache_append` | batch via `wid.y` | — | input, K/V_cache, params |
| `residual_add` | element-wise | — | a, b |

---

## The CPU Optimization Journey (70B Sparse Ternary)

Complete record of optimizing 70B sparse ternary matvec from scalar implementation to the physical memory bandwidth limit of M1 Pro.

### The Wall

```
                        ┌─────────────────────────────────────┐
  45% BW ──────────────▶│████████████████████████  Phase 6    │ 1.76 tok/s
                        │                                     │
  27% BW ──────────────▶│███████████████          Phase 5     │ 1.59 tok/s
                        │                                     │
                        │                                     │
   6% BW ──────────────▶│████                     Phase 4     │ 0.70 tok/s
   3% BW ──────────────▶│██                       Phase 3     │ 0.37 tok/s
   2% BW ──────────────▶│█                        Phase 2     │ 0.26 tok/s
   1% BW ──────────────▶│▌                        Phase 1     │ 0.16 tok/s
                        └─────────────────────────────────────┘
                         0        0.5       1.0       1.5    tok/s
```

### Phase 1 → Scalar Baseline (0.16 tok/s, 1% BW)

Naive implementation. Extracting bits one-by-one from 16-bit masks, a branch prediction nightmare with `if bit { sum += x }`.

```rust
// 70B: 76.5 ms/layer × 80 layers = 6,120 ms/token
for bit in 0..16 {
    if active_mask & (1 << bit) != 0 {
        let sign = if sign_mask & (1 << bit) != 0 { -1.0 } else { 1.0 };
        sum += sign * input[col + bit];  // Branch per weight
    }
}
```

**Bottleneck**: Branch prediction misses. Nearly random branch patterns at 50% sparsity.

### Phase 2 → NEON Mask Expansion (0.26 tok/s, 2% BW, 1.6x)

Bulk-expanding 16-bit masks with NEON `vtstq_u16`. From 8 calls to `expand_mask_4` down to 1 call to `expand_mask_16`.

```rust
// 16 weights expanded in 1 NEON op instead of 16 branches
let mask_vec = vdupq_n_u16(active_mask);
let expanded = vtstq_u16(mask_vec, bit_positions);  // 16 lanes simultaneously
```

**Finding**: Branch elimination alone is not enough. Data type remains f32 — wasting bandwidth.

### Phase 3 → Branchless 4-Row Micro-kernel (0.37 tok/s, 3% BW, 2.3x)

Register blocking with 4 rows processed simultaneously. Activation vector loads shared across 4 rows. **Zero branches**.

```rust
// 4 rows share the same activation load — 4x register reuse
let act = vld1q_f32(input.as_ptr().add(col));
acc0 = vmlaq_f32(acc0, w0, act);  // Row 0
acc1 = vmlaq_f32(acc1, w1, act);  // Row 1
acc2 = vmlaq_f32(acc2, w2, act);  // Row 2
acc3 = vmlaq_f32(acc3, w3, act);  // Row 3
```

**Finding**: f32 arithmetic is fast, but **f32 bandwidth consumption is the real enemy**. The input vector is read at 32-bit precision per block, consuming most of the bandwidth.

### Phase 4 → i8 Dynamic Quantization + SDOT (0.70 tok/s, 6% BW, 4.4x)

Dynamic quantization of activations from f32→i8. SDOT instruction executes 16×i8×i8 → 4×i32 in **a single instruction**.

```rust
// NEON-accelerated f32 → i8 quantization
let amax = vmaxvq_f32(vabsq_f32(chunk));  // Global max in 1 instruction
let scale = 127.0 / amax;
let quantized = vqmovn_s16(vcombine_s16(
    vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(v0, vscale))),
    vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(v1, vscale))),
));

// SDOT: 16 multiply-accumulates in 1 instruction (inline asm)
// sdot v_acc.4s, v_weights.16b, v_activations.16b
core::arch::asm!(
    "sdot {acc:v}.4s, {w:v}.16b, {a:v}.16b",
    acc = inout(vreg) acc, w = in(vreg) weights, a = in(vreg) activations,
);
```

Bandwidth consumption 1/4 (f32→i8). Compute throughput 4x (FMA→SDOT). **But only 6% BW — still slow.**

**Finding**: Weight expansion cost became dominant. 14 instructions to expand 16-bit mask → i8 weight vector.

### Phase 5 → vqtbl1q LUT Expansion (1.59 tok/s, 27% BW, 9.9x)

**Turning point**. Expanding 2-bit packed weights (00=0, 01=+1, 11=−1) via NEON table lookup in just 6 instructions.

```rust
// Packed 2-bit format: 4 weights per byte (00=0, 01=+1, 11=-1)
// 16 weights = 4 bytes (vs 32 bytes in f32)

// LUT: 2-bit index → i8 weight
let lut = [0i8, 1, 0, -1, 0, 0, 0, 0, ...];  // vqtbl1q lookup table

unsafe fn expand_packed_2bit_lut(packed_ptr: *const u8, ...) -> int8x16_t {
    let raw = vld1q_u8(packed_ptr);              // Load 4 bytes (16 weights)
    let replicated = vqtbl1q_u8(raw, rep_idx);   // Replicate bytes to lanes
    let shifted = vshlq_s8(replicated, shifts);   // Shift to isolate 2-bit pairs
    let indices = vandq_u8(shifted, mask_03);     // Mask to 2-bit index
    vqtbl1q_s8(lut, indices)                      // LUT: index → {0, +1, -1}
}
// 6 ops total. Previous mask-based: 14 ops.
```

**27% — but Phase 6 reveals the true cause.**

After reaching 9.9x at Phase 5, the question arose: **where is the remaining 73% going?**

Software prefetch (`prfm`), E-core exclusion, chunk size changes — all made things **worse** (see below). These failures proved by elimination that the true bottleneck was **neither compute nor prefetching**.

### Phase 6 → Block-Packed Weight Layout (1.76 tok/s, 45% BW, 11.0x)

**Discovery and elimination of TLB misses**.

Previous memory layout (Row-major):
```
Row 0: [blk0][blk1][blk2]...[blk511]    ← 2 KB per row
Row 1: [blk0][blk1][blk2]...[blk511]    ← 2 KB, different page
Row 2: [blk0][blk1][blk2]...[blk511]    ← 2 KB, different page
Row 3: [blk0][blk1][blk2]...[blk511]    ← 2 KB, different page
       ↑ 4-row kernel reads blk0 from 4 different pages → 4 TLB misses
```

Block-packed layout:
```
Group 0: [R0-blk0][R1-blk0][R2-blk0][R3-blk0] [R0-blk1][R1-blk1]...
         ↑ 4 rows' blk0 are contiguous → 0 TLB misses, sequential read
```

```rust
// Block-packed layout construction: 4 rows interleaved by block
let num_groups = (num_rows + 3) / 4;
for g in 0..num_groups {
    for blk in 0..blocks_per_row {
        for lane in 0..4 {  // 4 rows contiguous per block
            packed_blocked[dst] = packed_2bit[src];  // Sequential write
        }
    }
}

// Micro-kernel reads sequentially — no stride, no TLB miss
let blk_base = group_ptr.add(blk * stride_4);
let w0 = expand_packed_2bit_lut(blk_base);                          // Row 0
let w1 = expand_packed_2bit_lut(blk_base.add(bytes_per_block));     // Row 1 (adjacent!)
let w2 = expand_packed_2bit_lut(blk_base.add(2 * bytes_per_block)); // Row 2
let w3 = expand_packed_2bit_lut(blk_base.add(3 * bytes_per_block)); // Row 3
```

**27% → 45%. TLB miss elimination alone improved bandwidth utilization by 1.67x.**

### The Ceiling: 45% = M1 Pro CPU Physical Limit

The M1 Pro SoC bandwidth of 200 GB/s is shared with GPU/NPU/Media Engine. The bandwidth available exclusively to the CPU is approximately 90 GB/s (45%). This is a hard constraint imposed by the OS kernel and memory controller arbitration, and cannot be broken through software.

### Summary Table

| Phase | Technique | tok/s | ms/layer | BW | Speedup | Key Insight |
|---|---|---|---|---|---|---|
| 1 | Scalar baseline | 0.16 | 76.5 | 1% | 1.0x | Branch prediction disaster on random sparsity |
| 2 | NEON mask expansion | 0.26 | 47.6 | 2% | 1.6x | Branchless, but f32 bandwidth waste |
| 3 | 4-row micro-kernel | 0.37 | 33.9 | 3% | 2.3x | Register blocking — activation reuse |
| 4 | i8 quant + SDOT | 0.70 | 17.9 | 6% | 4.4x | 4x bandwidth reduction + 4x compute |
| 5 | vqtbl1q LUT | 1.59 | 7.9 | 27% | 9.9x | 6 ops vs 14 — weight expansion dominance |
| **6** | **Block Packing** | **1.76** | **7.1** | **45%** | **11.0x** | **TLB miss elimination → bandwidth ceiling** |

### What Didn't Work (Equally Valuable Data)

| Attempt | Result | Root Cause |
|---|---|---|
| Software prefetch (`prfm`) | **−10%** | M1 Pro HW prefetcher already optimal; `prfm` pollutes L1 cache |
| E-core exclusion (4 P-core threads) | **−14%** | Apple UMA: E-cores contribute bandwidth even if individually slower |
| Chunk size 8→16 rows | **−16%** | Rayon work-stealing imbalance; 8 = optimal for M1 Pro L1 geometry |
| 2x block unrolling | **−3%** | acc0–acc3 already independent; extra loop overhead for no gain |

**Lessons from failure**: The M1 Pro has an exceptionally capable HW prefetcher, and software intervention causes cache pollution. In Apple UMA, excluding E-cores means discarding part of the available bandwidth. These experiments served as **proof by elimination** that TLB misses were the true cause of the "27% wall".

---

## aarch64 NEON Quantized Matvec Bench (Issue #30)

Per-quantization NEON matvec kernels on Apple M3 (Mac) and NVIDIA Tegra Orin Nano (Jetson). 128 × 4096 f32-output matvec, 200-iter median (µs/matvec):

| Quant | Mac M3 scalar | Mac M3 NEON | Speedup | Jetson scalar | Jetson NEON | Speedup |
|:---|---:|---:|---:|---:|---:|---:|
| Q4_K | (already NEON) | **32.3** | — | (already NEON) | **172.7** | — |
| Q5_K | (scalar + SDOT auto-vec) | **42.9** | — | (scalar + SDOT auto-vec) | **282.7** | — |
| Q6_K | 89.6 (old NEON) | **35.3** | 2.54× | 344.9 (old NEON) | **165.2** | 2.09× |
| Q8_0 | 398.5 | **130.1** | 3.06× | 951.2 | **250.8** | 3.79× |
| Ternary | 344.7 | **68.9** | 5.00× | 967.2 | **190.4** | 5.08× |

**Design notes** (all kernels bit-exact or `rel_err < 1e-5` vs scalar reference — see `q*_neon_matches_scalar_*` unit tests):

- **Q4_K / Q6_K**: `vmull_s8` widening dot with `vpaddlq_s16` reduction. Q6_K also unpacks its 6-bit ql + qh layout in parallel using the `((qh << 4) & 0x30)` / `((qh << 2) & 0x30)` / `(qh & 0x30)` / `((qh >> 2) & 0x30)` shift-mask trick that feeds all four quadrants without per-element scalar packing.
- **Q8_0**: i8 → i16 → i32 → f32 widening chain (`vmovl` × 2, `vcvtq_f32_s32`) followed by `vfmaq_f32` FMA against the row's f32 input. Memory-bandwidth bound at 1.06 bytes/element.
- **Ternary**: bitmask expansion via `vdupq_n_u32(byte)` broadcast → `vandq_u32` against `[1,2,4,8]` / `[16,32,64,128]` → `vceqq_u32` → `vandq_u32` with reinterpreted f32 input. Zero per-lane multiplies — pure branchless select.

**Observations**:

- On Jetson, **Q6_K NEON (165.2 µs) is now faster than Q4_K NEON (172.7 µs)** since the qh packing overhead is gone. The two K-quant kernels are essentially at the memory-bandwidth floor.
- Q8_0 sits ~4× above Q4_K on both platforms because Q8_0 packs only 1 byte per element vs Q4_K's 0.56 bytes/element — nearly 2× more bytes read per output.
- Mac M3 speedups over Jetson (2.4–6.6×) correlate with LPDDR5X vs LPDDR5 bandwidth (~5.8× ratio); compute-heavy quants get closer to that ceiling.

Bench harness: `cargo run --release --example bench_simd_matvec --features gguf`

---

## Speculative Decoding

Layer-skip self-speculative decoding + probabilistic sampling:

1. **Draft**: Rapidly predict K tokens using only the first N layers (all logits saved)
2. **Rollback**: Rewind KV cache to pre-draft position
3. **Probabilistic acceptance**: Accept/reject via `min(1, p(x)/q(x))` — higher acceptance rate than greedy matching
4. **Rejection resampling**: Resample from `max(0, p(x) - q(x))` — strictly preserves output distribution
5. **Bonus token**: When all K drafts are accepted, generate one more token from verify logits (K+1 output/cycle)

```bash
cargo run --release --example elyza_gguf --features "gguf,parallel" -- \
  --model models/Llama-3-ELYZA-JP-8B-q4_k_m.gguf \
  --prompt "日本の首都は" --temperature 0.0 \
  --speculative-k 4 --draft-layers 30
```

### Probabilistic Speculative Sampling (Leviathan et al.)

```
for each draft token x with draft_logits q, verify_logits p:
    p_dist = softmax(p), q_dist = softmax(q)
    if rand() < min(1, p_dist[x] / q_dist[x]):
        ACCEPT x                                          ← high acceptance when distributions are close
    else:
        REJECT → resample from normalized max(0, p_dist - q_dist)  ← corrective resample from deficit
if all K drafts accepted:
    bonus token from final verify logits                   ← K+1 tokens per cycle
```

### 8B Empirical Results: Greedy vs Probabilistic

| draft_layers | Layers% | Greedy α | **Probabilistic α** | Speed |
|---|---|---|---|---|
| Baseline | 100% | — | — | **5.9 tok/s** |
| 24 | 75% | 4% | 17% | 2.1 tok/s |
| 28 | 87.5% | 13% | 17% | 2.4 tok/s |
| 30 | 93.8% | 37% | **25–75%** | 2.5 tok/s |
| 31 | 96.9% | 58% | **59–62%** | 2.9–3.1 tok/s |

**8B limitation**: In a 32-layer model, each layer's representational contribution is too large, making layer-skip acceptance rates insufficient to justify draft costs. 70B (80 layers) has significantly higher skip tolerance.

### Dual-Model Speculative Decoding (1B → 8B)

True speculative decoding using a separate draft model. Llama-3.2-1B (16 layers, 2048dim) → Llama-3.1-8B (32 layers, 4096dim):

```bash
cargo run --release --example speculative_dual --features "gguf,parallel" -- \
  --model models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --draft-model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --prompt "What is the capital of Japan?" --max-tokens 60
```

| Mode | tok/s | Accept Rate | Speedup |
|---|---|---|---|
| Baseline (8B only) | 5.7 | — | 1.0x |
| Speculative K=3 | 3.3 | 54% | 0.58x |
| Speculative K=4 | 5.6 | 63% | 0.99x |
| Speculative K=5 | 6.5 | 100% | 1.15x |

**CPU constraint**: Batch parallel verification is not possible, so the draft model cost becomes pure overhead. 1.15x speedup confirmed on short responses (all K drafts accepted). GPU batch verification is expected to provide significant acceleration.

### 70B Speculative Decoding Projections

| spec_k | draft_layers | Accept α | Effective tok/s | Speedup |
|---|---|---|---|---|
| 4 | 8 (10%) | 60% | 2.51 | 1.4x |
| 3 | 8 (10%) | 70% | 2.39 | 1.3x |
| 4 | 8 (10%) | 80% | **3.10** | **1.7x** |

---

## Computation Path (Q4_K)

llama.cpp-compatible integer dot product path:

1. Quantize input vector to **Q8_K** (256 elements/block, f32 scale, i8 values, i16 bsums)
2. Compute weight × input via **Q4_K×Q8_K** / **Q6_K×Q8_K** integer dot product (multiplication only at scale application)
3. 8 sub-blocks × 32 element integer accumulation, 6-bit packed scales/mins
4. Final result: `d × d_q8 × Σ(scale × q4 × q8) − dmin × d_q8 × Σ(mins × bsums)`

Exact match with llama.cpp's `ggml_vec_dot_q4_K_q8_K` (logit difference within ±0.03).

## Multi-Architecture Support

Auto-detected from GGUF `general.architecture`:

| Architecture | GQA | RoPE θ | Sliding Window | Logit Softcapping |
|---|---|---|---|---|
| Llama-3 | 4:1 | 500,000 | — | — |
| Mistral | 4:1 | 10,000 | 4096 | — |
| Gemma-2 | configurable | configurable | optional | attention + final |

## Sparse Ternary Quantization

Extreme compression via N:M structured sparsity:

- **N:M structured sparsity**: Only N non-zero out of 16 elements (8:16 = 50% density)
- **Packed 2-bit encoding**: `00=0, 01=+1, 11=-1` — 4 weights/byte
- **Block-packed layout**: 4-row interleave → TLB miss elimination
- **vqtbl1q_s8 LUT**: 2-bit → i8 on-the-fly expansion in 6 instructions
- **SDOT**: 16×i8×i8 → 4×i32 in 1 instruction (inline asm)
- **i8 dynamic quantization**: NEON-accelerated f32→i8 activation quantization

## Ternary QAT (Quantization-Aware Training)

BitNet b1.58-style quantization-aware training:

- **Straight-Through Estimator**: Ternarize in forward pass, pass gradients through in backward
- **L1 regularization**: `λ × Σ|w|` converges 30%+ of weights to 0
- **AdamW**: bias correction + weight decay
- **Layerwise Mixed Precision**: Attention=1.58bit / FFN=1bit+sparse

```
70B effective bits/parameter estimate:
  Attention (30%): 1.58 bit (ternary)
  FFN (70%):       0.92 bit (sparse ternary 8:16)
  → Overall avg:   ~1.1 bit/param → 70B ≈ 9.6 GB
```

## Performance

### 1B Model (Llama-3.2-1B-Instruct Q4_K_M, M1 Pro)

| Configuration | Decode Speed |
|---|---|
| Full 16-layer inference | **20.2 tok/s** |
| Logit accuracy vs llama.cpp | ±0.09 (top-1 token match) |

### 8B Model (ELYZA-JP Q4_K_M, M1 Pro)

| Phase | Configuration | Decode Speed | Prefill (16 tok) |
|---|---|---|---|
| Phase 1 | Bug fix (dequant→integer dot product) | 1.2 tok/s | 14.2s |
| Phase 2 | + Q8_K reuse, auto-vec, Rayon | 5.1 tok/s | 4.1s |
| Phase 3 | + Contiguous KV cache | **5.9 tok/s** | **3.3s** |

### 70B Sparse Ternary (Simulated, M1 Pro)

```bash
cargo run --release --example bench_70b_sparse --features "gguf,parallel"
```

| Projection | Size | Time/iter |
|---|---|---|
| Q proj | 8192×8192 | 0.54 ms |
| K proj | 1024×8192 | 0.10 ms |
| V proj | 1024×8192 | 0.10 ms |
| O proj | 8192×8192 | 0.53 ms |
| Gate (FFN) | 28672×8192 | 1.70 ms |
| Up (FFN) | 28672×8192 | 2.00 ms |
| Down (FFN) | 8192×28672 | 2.17 ms |
| **1 layer** | | **7.1 ms** |
| **1 token (80 layers)** | | **569 ms → 1.76 tok/s** |

### Device Projections (bandwidth-limited)

| Device | Memory BW | Est. tok/s |
|---|---|---|
| Raspberry Pi 5 | 34 GB/s | 0.7 |
| Mac Mini M4 | 120 GB/s | 2.3 |
| M1 Pro (measured) | 91 GB/s | 1.8 |
| Mac Mini M4 Pro | 273 GB/s | 5.3 |
| Mac Studio M4 Ultra | 800 GB/s | 15.6 |

### Quantization Block Specs

| Type | Block Size | Bytes/Block | bpw | Layout |
|---|---|---|---|---|
| Q2_K | 256 | 84 | 2.625 | scales[16] + qs[64] + d(f16) + dmin(f16) |
| Q3_K | 256 | 110 | 3.4375 | hmask[32] + qs[64] + scales[12] + d(f16) |
| Q4_K | 256 | 144 | 4.5 | d(f16) + dmin(f16) + scales[12] + qs[128] |
| Q5_K | 256 | 176 | 5.5 | d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128] |
| Q6_K | 256 | 210 | 6.5625 | ql[128] + qh[64] + scales[16] + d(f16) |
| Q8_0 | 32 | 34 | 8.5 | d(f16) + qs[32] |

Mixed quantization variants (`_S`, `_M`, `_L`) use different types per layer — e.g. Q3_K for attention, Q4_K/Q5_K for FFN, Q6_K for embeddings, F32 for norms — so effective bpw is higher than the base type.

### 30B / 70B Model Size Estimates

| Quant | bpw (eff.) | 30B | 70B |
|---|---|---|---|
| F16 | 16.0 | 60.0 GB | 140.0 GB |
| Q8_0 | 8.5 | 31.9 GB | 74.4 GB |
| Q6_K | 6.56 | 24.6 GB | 57.4 GB |
| Q5_K_M | ~5.7 | ~21.4 GB | ~49.9 GB |
| Q4_K_M | ~4.8 | ~18.0 GB | ~42.0 GB |
| Q3_K_L | ~4.0 | ~15.0 GB | ~35.0 GB |
| Q3_K_M | ~3.9 | ~14.6 GB | ~34.1 GB |
| Q3_K_S | ~3.5 | ~13.1 GB | ~30.6 GB |
| Q2_K | ~3.2 | ~12.0 GB | ~28.0 GB |

### Recommended Quantization by Device Memory

| Memory | 30B Model | 70B Model |
|---|---|---|
| **16 GB** (MBA M3) | Q2_K (12GB) ○ / Q3_K_S (13GB) △ | × |
| **24 GB** (M3 Pro) | Q4_K_M (18GB) ○ | × |
| **32 GB** (M3 Max) | Q6_K (25GB) ○ | Q2_K (28GB) △ |
| **36 GB** (M3 Pro) | Q6_K (25GB) ○ | Q3_K_S (31GB) △ |
| **48 GB** (M4 Max) | Q8_0 (32GB) ○ | Q3_K_M (34GB) ○ |
| **64 GB** (M2 Ultra) | F16 (60GB) △ | Q4_K_M (42GB) ○ |
| **128 GB** (M4 Ultra) | F16 (60GB) ○ | Q8_0 (74GB) ○ |

○ = comfortable (model + OS + KV cache fit in RAM), △ = runs with swap pressure

## Inference Server (OpenAI-compatible API)

```bash
cargo run --bin alice-llm-server --features server --release -- \
  --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --port 8090
```

```bash
# Text completion
curl http://localhost:8090/v1/completions -H "Content-Type: application/json" \
  -d '{"prompt":"What is the capital of Japan?","max_tokens":32,"temperature":0}'

# Response:
# {"choices":[{"text":"The capital of Japan is Tokyo.","finish_reason":"stop"}],
#  "usage":{"tokens_per_second":14.8,"decode_ms":473}}

# Health check
curl http://localhost:8090/health

# List models
curl http://localhost:8090/v1/models
```

## CLI Options

```
--model <path>            Path to GGUF model file
--prompt <text>           Input prompt
--max-tokens <n>          Maximum tokens to generate (default: 256)
--temperature <f>         Sampling temperature (default: 0.7, 0.0 = greedy)
--speculative-k <n>       Speculative decoding: draft K tokens (default: 0 = off)
--draft-layers <n>        Layers for draft model (default: 8)
--ternary                 Use ternary quantized weights
--ternary-threshold <f>   Ternary sparsity threshold (default: 0.7)
```

## Cargo Features

| Feature | Description |
|---|---|
| `gguf` | GGUF file loading and multi-arch inference |
| `gpu` | wgpu GPU compute (Metal/Vulkan/DX12), requires `gguf` |
| `server` | HTTP inference server (axum), includes `gpu` + `gguf` |
| `parallel` | Rayon-based multi-threaded CPU matvec |

## License

**Dual-Licensed: AGPL-3.0+ OR Commercial**

ALICE-LLM is open-source and free to use under the **GNU Affero General Public License v3.0 or later (AGPL-3.0+)**.
This ensures that any modifications or network-based SaaS deployments using this engine contribute back to the open-source community.

**Commercial License**
If you intend to use ALICE-LLM in a proprietary SaaS environment, cloud infrastructure, or closed-source commercial product without disclosing your backend source code, you must obtain a **Commercial License**.

For commercial licensing inquiries and enterprise support, contact: licensing@alicelaw.net
