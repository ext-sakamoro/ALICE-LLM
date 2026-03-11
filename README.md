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
- **Auto-vectorized SIMD** — loop structure optimized for LLVM auto-vectorization with `target-cpu=native`
- **Speculative decoding** — layer-skip draft model with KV cache rollback and acceptance stats
- **Ternary quantization** — {-1, 0, +1} bitmask weights (~2 bits/param)
- **Sparse ternary** — N:M structured sparsity (8:16), flat-buffer zero-multiply matvec
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
Speed: 6.2 tok/s (3345 prefill + 1281 decode = 4640 total ms)
```

## Architecture

```
src/
├── lib.rs       — BPE tokenizer, KV cache, attention, RoPE, sampling
├── gguf.rs      — GGUF v3 parser, Q4_K/Q6_K/Q8_K quantization, fused matvec,
│                  ternary/sparse-ternary matrices
├── llama3.rs    — Multi-arch forward pass, model loading, speculative decoding,
│                  paged KV cache, continuous batching, mixed precision config
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

### Continuous Batching + PagedAttention

複数リクエストを同時処理:

- **PagedKvCache**: 16トークン/ページ、オンデマンド確保（最大シーケンス長の事前確保不要）
- **BatchScheduler**: リクエスト追加→prefill→round-robin decode→完了判定
- 各リクエストが独立した PagedKvCache を持ち、メモリ効率的に並行推論

### Sparse Ternary Quantization

N:M 構造化スパース性による極限圧縮:

- **N:M structured sparsity**: 16要素ブロックごとにN個だけ非ゼロ (例: 8:16 = 50%密度)
- **フラットバッファ**: 全行のマスクを連続メモリに配置（キャッシュフレンドリー）
- **Zero-multiply matvec**: pos/neg マスク分離、加算/減算のみ（内部ループに乗算なし）
- **メタデータ効率**: 構造化により per-block 32bit (active_mask + sign_mask) のみ

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
| Phase 3 | + Contiguous KV cache | **6.2 tok/s** | **3.3s** |

**5.2x speedup** from Phase 1 to Phase 3.

### 70B Sparse Ternary Benchmark (Simulated)

8:16 構造化スパースでのmatvecスループット（M1 Pro実測）:

```bash
cargo run --release --example bench_70b_sparse --features "gguf,parallel"
```

| Projection | Size | Time/iter |
|---|---|---|
| Q proj | 8192×8192 | 6.2 ms |
| K proj | 1024×8192 | 1.1 ms |
| Gate (FFN) | 28672×8192 | 21.0 ms |
| **1 layer total** | | **76.5 ms** |
| **1 token (80 layers)** | | **6.1 s → 0.16 tok/s** |

帯域律速時の理論スループット:

| Device | Memory BW | Est. tok/s |
|---|---|---|
| Raspberry Pi 5 | 34 GB/s | 2.0 |
| Mac Mini M4 | 120 GB/s | 7.0 |
| Mac Mini M4 Pro | 273 GB/s | 15.9 |
| Mac Studio M4 Ultra | 800 GB/s | 46.6 |

現在の実測は帯域利用率~1%（compute-bound）。NEON intrinsics によるマスク処理最適化で理論値に接近予定。

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
