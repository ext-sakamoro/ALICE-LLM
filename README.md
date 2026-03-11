# ALICE-LLM

Pure Rust LLM inference engine. Runs GGUF quantized models with zero external ML dependencies. 156 tests.

## Features

- **GGUF v3 parser** — zero-copy weight loading
- **Q4_K / Q6_K / Q8_0 / F16 / F32** quantization support
- **llama.cpp-compatible computation** — Q8_K quantized input × Q4_K/Q6_K integer dot product (matches `ggml_vec_dot_q4_K_q8_K`)
- **Multi-architecture** — Llama-3, Mistral (Sliding Window Attention), Gemma-2 (logit softcapping), auto-detected from GGUF
- **BPE tokenizer** — GPT-2 byte encoding, loaded from GGUF metadata
- **Contiguous KV cache** — pre-allocated flat buffer, GQA-aware, rollback support
- **Paged KV cache** — on-demand page allocation (16 tokens/page), per-request isolation
- **Continuous batching** — round-robin multi-request scheduler with per-request PagedKvCache
- **Rayon parallelization** — multi-threaded matvec (optional `parallel` feature)
- **NEON SIMD intrinsics** — hand-tuned aarch64 kernels: vqtbl1q LUT expansion, SDOT integer dot product, branchless 4-row micro-kernel
- **Block-packed weight layout** — 4-row interleaved memory layout for TLB-friendly micro-kernel access
- **i8 dynamic quantization** — NEON-accelerated f32→i8 activation quantization with global scale
- **Speculative decoding** — layer-skip draft model with KV cache rollback (Q4_K and sparse ternary)
- **Ternary quantization** — {-1, 0, +1} bitmask weights (~2 bits/param)
- **Sparse ternary** — N:M structured sparsity (8:16), packed 2-bit weights, LUT+SDOT optimized matvec
- **Ternary QAT** — Straight-Through Estimator, L1 sparsity regularization, AdamW optimizer
- **Layerwise mixed precision** — Attention=1.58bit / FFN=1bit+sparse configurable per layer

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

Output:
```
日本の首都は東京です。
Tokens: 8 generated, 16 prompt
Speed: 5.9 tok/s (4434 prefill + 1432 decode = 5883 total ms)
```

## Architecture

```
src/
├── lib.rs       — BPE tokenizer, KV cache, attention, RoPE, sampling
├── gguf.rs      — GGUF v3 parser, Q4_K/Q6_K/Q8_K quantization, fused matvec,
│                  ternary/sparse-ternary matrices, NEON SDOT+LUT kernels
├── llama3.rs    — Multi-arch forward pass, model loading, speculative decoding,
│                  paged KV cache, continuous batching, sparse ternary inference
└── training.rs  — Ternary QAT: STE, QatLinear, L1 regularization, AdamW, MSE loss
```

### Computation Path (Q4_K)

llama.cpp 互換の整数内積パスを使用:

1. 入力ベクトルを **Q8_K** に量子化 (256要素/ブロック, f32 scale, i8 values, i16 bsums)
2. 重み × 入力を **Q4_K×Q8_K** / **Q6_K×Q8_K** 整数内積で計算（乗算はスケール適用時のみ）
3. 8サブブロック × 32要素の整数累積、6-bit packed scales/mins
4. 最終結果: `d × d_q8 × Σ(scale × q4 × q8) − dmin × d_q8 × Σ(mins × bsums)`

llama.cpp の `ggml_vec_dot_q4_K_q8_K` と完全一致（logits差 ±0.03 以内）。

### Multi-Architecture Support

GGUF の `general.architecture` から自動検出:

| Architecture | GQA | RoPE θ | Sliding Window | Logit Softcapping |
|---|---|---|---|---|
| Llama-3 | 4:1 | 500,000 | — | — |
| Mistral | 4:1 | 10,000 | 4096 | — |
| Gemma-2 | configurable | configurable | optional | attention + final |

### Sparse Ternary Matvec — Optimization Journey

70Bモデルのスパースターナリ行列ベクトル積を、10フェーズにわたって最適化:

```
Phase 1  (Scalar)           → 0.16 tok/s  76.5 ms/layer   1% BW   ──  baseline
Phase 2  (NEON expand_mask) → 0.26 tok/s  47.6 ms/layer   2% BW   ──  1.6x
Phase 3  (4-row branchless) → 0.37 tok/s  33.9 ms/layer   3% BW   ──  2.3x
Phase 4  (i8 quant + SDOT)  → 0.70 tok/s  17.9 ms/layer   6% BW   ──  4.4x
Phase 5  (vqtbl1q LUT)      → 1.59 tok/s   7.9 ms/layer  27% BW   ──  9.9x
Phase 6  (Block Packing)    → 1.76 tok/s   7.1 ms/layer  45% BW   ── 11.0x
```

各フェーズの技術:

| Phase | Technique | Key Insight |
|---|---|---|
| 1→2 | NEON `vtstq_u16` mask expansion | 8× expand_mask_4 → 1× expand_mask_16 |
| 2→3 | Branchless 4-row micro-kernel | Register blocking, shared activation loads, ZERO branches |
| 3→4 | i8 dynamic quantization + `sdot` asm | 16×i8×i8→4×i32 in 1 instruction |
| 4→5 | Packed 2-bit + `vqtbl1q_s8` LUT | 6 ops vs 14 ops for mask→weight expansion |
| 5→6 | Block-packed weight layout | 4 rows interleaved → TLB miss elimination |

**帯域利用率 45% = M1 Pro CPU推論の物理限界** — SoC帯域の残り55%はGPU/NPU/Media Engine用。

#### What Didn't Work (Equally Valuable Data)

| Attempt | Result | Root Cause |
|---|---|---|
| Software prefetch (`prfm`) | −10% regression | M1 Pro HW prefetcher already optimal; prfm causes cache pollution |
| E-core exclusion (Rayon 4 threads) | −14% regression | E-cores contribute bandwidth even if slower; Apple UMA design |
| Chunk size 8→16 | −16% regression | Load imbalance; 8 rows = golden ratio for M1 Pro L1 cache |
| 2x block unrolling | −3% regression | acc split didn't help (acc0-3 already independent); extra branches |

### Speculative Decoding

レイヤースキップ方式の自己投機的デコード:

1. **ドラフト**: 先頭 N 層のみでK個のトークンを高速推測
2. **ロールバック**: KVキャッシュをドラフト前の位置に巻き戻し
3. **検証**: 全層でドラフトトークンを逐次検証、一致すれば受理

```bash
cargo run --release --example elyza_gguf --features "gguf,parallel" -- \
  --model models/Llama-3-ELYZA-JP-8B-q4_k_m.gguf \
  --prompt "日本の首都は" --temperature 0.0 \
  --speculative-k 4 --draft-layers 24
```

#### 8B Layer-Skip Acceptance Rate (Empirical, Greedy)

| draft_layers | Layers% | Accept α | Effective Speed |
|---|---|---|---|
| Baseline | 100% | — | **5.9 tok/s** |
| 8 | 25% | 0% | 4.2 tok/s |
| 16 | 50% | 0% | 2.9 tok/s |
| 24 | 75% | 4% | 2.5 tok/s |
| 28 | 87.5% | 13% | 2.4 tok/s |
| 30 | 93.8% | 37% | 2.5 tok/s |
| 31 | 96.9% | 58% | 2.8 tok/s |

**Key insight**: 8Bモデル(32層)ではレイヤースキップの受理率がドラフトコストに見合わない。各層の表現寄与が大きすぎるため。70B(80層)ではスキップ耐性が大幅に向上する。

### Continuous Batching + PagedAttention

複数リクエストを同時処理:

- **PagedKvCache**: 16トークン/ページ、オンデマンド確保（最大シーケンス長の事前確保不要）
- **BatchScheduler**: リクエスト追加→prefill→round-robin decode→完了判定
- 各リクエストが独立した PagedKvCache を持ち、メモリ効率的に並行推論

### Sparse Ternary Quantization

N:M 構造化スパース性による極限圧縮:

- **N:M structured sparsity**: 16要素ブロックごとにN個だけ非ゼロ (例: 8:16 = 50%密度)
- **Packed 2-bit encoding**: 00=0, 01=+1, 11=-1 — 4 weights per byte
- **Block-packed layout**: 4行×全ブロックをインターリーブ配置 → TLBミス消滅
- **vqtbl1q_s8 LUT expansion**: 2-bit → i8 を6命令でオンザフライ展開（RAM上は2-bitのまま）
- **SDOT integer dot product**: 16×i8×i8 → 4×i32 in 1 instruction (inline asm)
- **i8 dynamic quantization**: NEON加速 f32→i8 活性化量子化

### Ternary QAT (Quantization-Aware Training)

BitNet b1.58 スタイルの学習時量子化:

- **Straight-Through Estimator**: forward で ternarize、backward で勾配素通し
- **L1 正則化**: `λ × Σ|w|` ペナルティで学習中に重みの30%+を0に収束
- **AdamW**: bias correction + weight decay
- **Layerwise Mixed Precision**: Attention=1.58bit維持、FFN=1bit+sparsity で攻める

```
70B モデルの実効ビット/パラメータ推定:
  Attention (30%): 1.58 bit (ternary)
  FFN (70%):       0.92 bit (sparse ternary 8:16)
  → 全体平均:      ~1.1 bit/param → 70B ≈ 9.6 GB
```

## Performance

### 8B Model (ELYZA-JP Q4_K_M, M1 Pro)

| Phase | Configuration | Decode Speed | Prefill (16 tok) |
|---|---|---|---|
| Phase 1 | Bug fix (dequant→integer dot product) | 1.2 tok/s | 14.2s |
| Phase 2 | + Q8_K reuse, auto-vec, Rayon | 5.1 tok/s | 4.1s |
| Phase 3 | + Contiguous KV cache | **5.9 tok/s** | **3.3s** |

**4.9x speedup** from Phase 1 to Phase 3.

### 70B Sparse Ternary Benchmark (Simulated, M1 Pro)

8:16 構造化スパースでのmatvecスループット:

```bash
cargo run --release --example bench_70b_sparse --features "gguf,parallel"
```

| Projection | Size | Time/iter (Phase 6) |
|---|---|---|
| Q proj | 8192×8192 | 0.54 ms |
| K proj | 1024×8192 | 0.10 ms |
| V proj | 1024×8192 | 0.10 ms |
| O proj | 8192×8192 | 0.53 ms |
| Gate (FFN) | 28672×8192 | 1.70 ms |
| Up (FFN) | 28672×8192 | 2.00 ms |
| Down (FFN) | 8192×28672 | 2.17 ms |
| **1 layer total** | | **7.1 ms** |
| **1 token (80 layers)** | | **569 ms → 1.76 tok/s** |

#### Optimization Progression (70B, 1 token)

| Phase | tok/s | ms/layer | BW Utilization | Cumulative Speedup |
|---|---|---|---|---|
| Scalar baseline | 0.16 | 76.5 | 1% | 1.0x |
| NEON mask expand | 0.26 | 47.6 | 2.3% | 1.6x |
| 4-row micro-kernel | 0.37 | 33.9 | 3.2% | 2.3x |
| i8 quant + SDOT | 0.70 | 17.9 | 6% | 4.4x |
| vqtbl1q LUT | 1.59 | 7.9 | 27% | 9.9x |
| **Block Packing** | **1.76** | **7.1** | **45%** | **11.0x** |

#### Speculative Decoding Projections (70B)

| spec_k | draft_layers | Accept α | Effective tok/s | Speedup |
|---|---|---|---|---|
| 4 | 8 (10%) | 60% | 2.51 | 1.4x |
| 3 | 8 (10%) | 70% | 2.39 | 1.3x |
| 4 | 8 (10%) | 80% | **3.10** | **1.7x** |

#### Device Projections (bandwidth-limited)

| Device | Memory BW | Est. tok/s |
|---|---|---|
| Raspberry Pi 5 | 34 GB/s | 0.7 |
| Mac Mini M4 | 120 GB/s | 2.3 |
| M1 Pro (measured) | 91 GB/s | 1.8 |
| Mac Mini M4 Pro | 273 GB/s | 5.3 |
| Mac Studio M4 Ultra | 800 GB/s | 15.6 |

## Cargo Features

| Feature | Description |
|---|---|
| `gguf` | GGUF file loading and multi-arch inference |
| `parallel` | Rayon-based multi-threaded matvec |

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

## License

MIT OR Apache-2.0
