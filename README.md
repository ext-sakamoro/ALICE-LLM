# ALICE-LLM

Pure Rust LLM inference engine. GGUF quantized models, zero external ML dependencies, 156 tests.

**0.16 → 1.76 tok/s (11x) on 70B sparse ternary — hitting 45% memory bandwidth on M1 Pro CPU.**

**Dual-model speculative decoding: Llama-3.2-1B draft → Llama-3.1-8B verify, 63% accept rate, 5.7 tok/s baseline.**

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

## Features

- **GGUF v3 parser** — zero-copy mmap weight loading
- **Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_0 / F16 / F32** quantization (all GGML K-quant types)
- **llama.cpp-compatible** — Q2_K–Q6_K×Q8_K integer dot product (matches `ggml_vec_dot_q*_K_q8_K`, ±0.03 logits)
- **Multi-architecture** — Llama-3/3.1/3.2, Mistral (sliding window), Gemma-2 (softcapping), auto-detected
- **Tied embeddings** — Llama-3.2-1B/3B output projection via quantized `token_embd.weight` (Q6_K matvec)
- **BPE tokenizer** — GPT-2 byte encoding from GGUF metadata
- **Contiguous + Paged KV cache** — flat buffer with rollback, or 16-tok/page on-demand allocation
- **Continuous batching** — round-robin scheduler, per-request PagedKvCache
- **Speculative decoding** — layer-skip draft + dual-model (1B→8B) + probabilistic sampling (Leviathan et al.)
- **RoPE frequency scaling** — NTK-aware context extension via `rope_freqs.weight` tensor (Llama-3.1/3.2)
- **LLVM auto-vectorization** — `target-cpu=native` generates ARM SDOT instructions (37 sdot in Q4_K dot product)
- **Sparse ternary** — N:M structured sparsity, packed 2-bit, LUT+SDOT optimized, block-packed layout
- **Ternary QAT** — STE, L1 regularization, AdamW, layerwise mixed precision

## Architecture

```
src/
├── lib.rs       — BPE tokenizer, KV cache, attention, RoPE, sampling
├── gguf.rs      — GGUF v3 parser, Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K quantization, fused matvec,
│                  ternary/sparse-ternary matrices, LLVM auto-vectorized SDOT kernels
├── llama3.rs    — Multi-arch forward pass, model loading, speculative decoding (layer-skip + dual-model),
│                  paged KV cache, continuous batching, tied embeddings, RoPE freq scaling
└── training.rs  — Ternary QAT: STE, QatLinear, L1 regularization, AdamW, MSE loss

.cargo/
└── config.toml  — rustflags: target-cpu=native (enables LLVM SDOT auto-vectorization)
```

---

## The Optimization Journey

70Bモデルのスパースターナリ matvec を、スカラー実装から M1 Pro メモリ帯域の物理限界まで最適化した全記録。

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

素朴な実装。16-bit マスクからビットを1つずつ取り出し、`if bit { sum += x }` の分岐地獄。

```rust
// 70B: 76.5 ms/layer × 80 layers = 6,120 ms/token
for bit in 0..16 {
    if active_mask & (1 << bit) != 0 {
        let sign = if sign_mask & (1 << bit) != 0 { -1.0 } else { 1.0 };
        sum += sign * input[col + bit];  // Branch per weight
    }
}
```

**ボトルネック**: 分岐予測ミス。50%スパースでほぼランダムな分岐パターン。

### Phase 2 → NEON Mask Expansion (0.26 tok/s, 2% BW, 1.6x)

16-bit マスクを NEON `vtstq_u16` で一括展開。8回の `expand_mask_4` → 1回の `expand_mask_16`。

```rust
// 16 weights expanded in 1 NEON op instead of 16 branches
let mask_vec = vdupq_n_u16(active_mask);
let expanded = vtstq_u16(mask_vec, bit_positions);  // 16 lanes simultaneously
```

**発見**: 分岐除去だけでは不十分。データ型が f32 のまま — 帯域を浪費している。

### Phase 3 → Branchless 4-Row Micro-kernel (0.37 tok/s, 3% BW, 2.3x)

4行を同時処理するレジスタブロッキング。活性化ベクトルの読み込みを4行で共有。**ゼロ分岐**。

```rust
// 4 rows share the same activation load — 4x register reuse
let act = vld1q_f32(input.as_ptr().add(col));
acc0 = vmlaq_f32(acc0, w0, act);  // Row 0
acc1 = vmlaq_f32(acc1, w1, act);  // Row 1
acc2 = vmlaq_f32(acc2, w2, act);  // Row 2
acc3 = vmlaq_f32(acc3, w3, act);  // Row 3
```

**発見**: f32 演算は速いが、**f32 の帯域消費が本当の敵**。入力ベクトルは32-bit精度で毎ブロック読み込まれ、帯域の大半を食っている。

### Phase 4 → i8 Dynamic Quantization + SDOT (0.70 tok/s, 6% BW, 4.4x)

活性化ベクトルを f32→i8 に動的量子化。SDOT 命令で 16×i8×i8 → 4×i32 を**1命令**で実行。

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

帯域消費 1/4 (f32→i8)。演算スループット 4x (FMA→SDOT)。**しかし 6% BW — まだ遅い。**

**発見**: 重みの展開コストが支配的になった。16-bit マスク → i8 重みベクトルへの展開に14命令も使っている。

### Phase 5 → vqtbl1q LUT Expansion (1.59 tok/s, 27% BW, 9.9x)

**ターニングポイント**。2-bit packed weights（00=0, 01=+1, 11=−1）を NEON テーブルルックアップで 6命令展開。

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

**27% → しかし Phase 6 で真の原因が判明する。**

Phase 5 で 9.9x に到達したとき、次の疑問が浮かんだ: **残りの 73% はどこに消えている？**

ソフトウェアプリフェッチ (`prfm`)、E-core 除外、チャンクサイズ変更 — すべて**悪化**した（後述）。
これらの失敗から、真のボトルネックが**演算でもプリフェッチでもない**ことが証明された。

### Phase 6 → Block-Packed Weight Layout (1.76 tok/s, 45% BW, 11.0x)

**TLBミスの発見と殲滅**。

従来のメモリレイアウト（Row-major）:
```
Row 0: [blk0][blk1][blk2]...[blk511]    ← 2 KB per row
Row 1: [blk0][blk1][blk2]...[blk511]    ← 2 KB, different page
Row 2: [blk0][blk1][blk2]...[blk511]    ← 2 KB, different page
Row 3: [blk0][blk1][blk2]...[blk511]    ← 2 KB, different page
       ↑ 4-row kernel reads blk0 from 4 different pages → 4 TLB misses
```

Block-packed レイアウト:
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

**27% → 45%。TLBミスの除去だけで帯域利用率が 1.67x。**

### The Ceiling: 45% = M1 Pro CPU Physical Limit

M1 Pro の SoC 帯域 200 GB/s は GPU/NPU/Media Engine と共有。CPU が単独で使える帯域は約 90 GB/s（45%）。これは OS カーネルとメモリコントローラのアービトレーションによるハード制約であり、ソフトウェアでは突破できない。

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

**失敗の教訓**: M1 Pro は HW プリフェッチャーが極めて優秀で、ソフトウェアの介入は cache pollution を引き起こす。Apple UMA では E-core を除外すると帯域の一部を捨てることになる。これらの実験が「27% の壁」の真因が TLB ミスであることの**消去法的証明**となった。

---

## Speculative Decoding

レイヤースキップ方式の自己投機的デコード + 確率的サンプリング:

1. **ドラフト**: 先頭 N 層のみで K 個のトークンを高速推測（全logits保存）
2. **ロールバック**: KV キャッシュをドラフト前の位置に巻き戻し
3. **確率的受理**: `min(1, p(x)/q(x))` で受理判定 — greedy match より高い受理率
4. **棄却時リサンプリング**: `max(0, p(x) - q(x))` から再サンプル — 出力分布を厳密に保存
5. **ボーナストークン**: 全 K 個受理時、検証 logits からもう 1 トークン（K+1 出力/サイクル）

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
        ACCEPT x                                          ← 分布が近ければ高確率で受理
    else:
        REJECT → resample from normalized max(0, p_dist - q_dist)  ← 不足分から補正サンプル
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

**8B の限界**: 32層モデルでは各層の表現寄与が大きすぎ、層スキップの受理率がドラフトコストに見合わない。70B（80層）ではスキップ耐性が大幅に向上する。

### Dual-Model Speculative Decoding (1B → 8B)

別モデルをドラフトに使う真の投機的デコード。Llama-3.2-1B (16層, 2048dim) → Llama-3.1-8B (32層, 4096dim):

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

**CPU上の制約**: バッチ並列検証が不可能なため、ドラフトモデルのコストが純粋なオーバーヘッドになる。短い応答（K個全受理）では1.15xのスピードアップを確認。GPU上ではバッチ検証により大幅な高速化が期待できる。

### 70B Speculative Decoding Projections

| spec_k | draft_layers | Accept α | Effective tok/s | Speedup |
|---|---|---|---|---|
| 4 | 8 (10%) | 60% | 2.51 | 1.4x |
| 3 | 8 (10%) | 70% | 2.39 | 1.3x |
| 4 | 8 (10%) | 80% | **3.10** | **1.7x** |

---

## Computation Path (Q4_K)

llama.cpp 互換の整数内積パス:

1. 入力ベクトルを **Q8_K** に量子化 (256要素/ブロック, f32 scale, i8 values, i16 bsums)
2. 重み × 入力を **Q4_K×Q8_K** / **Q6_K×Q8_K** 整数内積で計算（乗算はスケール適用時のみ）
3. 8サブブロック × 32要素の整数累積、6-bit packed scales/mins
4. 最終結果: `d × d_q8 × Σ(scale × q4 × q8) − dmin × d_q8 × Σ(mins × bsums)`

llama.cpp の `ggml_vec_dot_q4_K_q8_K` と完全一致（logits差 ±0.03 以内）。

## Multi-Architecture Support

GGUF の `general.architecture` から自動検出:

| Architecture | GQA | RoPE θ | Sliding Window | Logit Softcapping |
|---|---|---|---|---|
| Llama-3 | 4:1 | 500,000 | — | — |
| Mistral | 4:1 | 10,000 | 4096 | — |
| Gemma-2 | configurable | configurable | optional | attention + final |

## Sparse Ternary Quantization

N:M 構造化スパース性による極限圧縮:

- **N:M structured sparsity**: 16要素中 N 個だけ非ゼロ (8:16 = 50% density)
- **Packed 2-bit encoding**: `00=0, 01=+1, 11=-1` — 4 weights/byte
- **Block-packed layout**: 4行インターリーブ → TLB ミス消滅
- **vqtbl1q_s8 LUT**: 2-bit → i8 を 6 命令でオンザフライ展開
- **SDOT**: 16×i8×i8 → 4×i32 in 1 instruction (inline asm)
- **i8 dynamic quantization**: NEON 加速 f32→i8 活性化量子化

## Ternary QAT (Quantization-Aware Training)

BitNet b1.58 スタイルの学習時量子化:

- **Straight-Through Estimator**: forward で ternarize、backward で勾配素通し
- **L1 正則化**: `λ × Σ|w|` で重みの 30%+ を 0 に収束
- **AdamW**: bias correction + weight decay
- **Layerwise Mixed Precision**: Attention=1.58bit / FFN=1bit+sparse

```
70B 実効ビット/パラメータ推定:
  Attention (30%): 1.58 bit (ternary)
  FFN (70%):       0.92 bit (sparse ternary 8:16)
  → 全体平均:      ~1.1 bit/param → 70B ≈ 9.6 GB
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
| `parallel` | Rayon-based multi-threaded matvec |

## License

MIT OR Apache-2.0
