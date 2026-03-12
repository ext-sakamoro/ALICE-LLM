# ALICE-LLM

[English](README.md) | **日本語**

Pure Rust LLM推論エンジン。GGUF量子化モデル、外部MLライブラリ依存ゼロ、156テスト。

**GPU (wgpu/Metal): 125ms → 71ms/トークン (1B)、バッチ4投機的デコード: 1Bドラフト + 8B検証 = 5.89倍高速化、受理率90%。**

**CPU: 0.16 → 1.76 tok/s (11倍) 70Bスパースターナリ — M1 Proメモリ帯域の45%に到達。**

## クイックスタート

```bash
# モデルダウンロード
huggingface-cli download elyza/Llama-3-ELYZA-JP-8B-GGUF \
  Llama-3-ELYZA-JP-8B-q4_k_m.gguf --local-dir models/

# 推論実行
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

## 機能一覧

- **GGUF v3 パーサー** — ゼロコピー mmap 重み読み込み
- **Q2_K / Q3_K / Q4_K / Q5_K / Q6_K / Q8_0 / F16 / F32** 量子化（全GGML K-quant型対応）
- **llama.cpp互換** — Q2_K–Q6_K×Q8_K 整数内積（`ggml_vec_dot_q*_K_q8_K` と一致、±0.03 logits）
- **マルチアーキテクチャ** — Llama-3/3.1/3.2、Mistral（スライディングウィンドウ）、Gemma-2（ソフトキャッピング）、自動検出
- **Tied embeddings** — Llama-3.2-1B/3B の出力射影を量子化 `token_embd.weight` で実行（Q6_K matvec）
- **BPE トークナイザ** — GGUFメタデータからGPT-2バイトエンコーディング
- **Contiguous + Paged KVキャッシュ** — ロールバック付きフラットバッファ、または16トークン/ページのオンデマンド確保
- **Continuous batching** — ラウンドロビンスケジューラ、リクエスト単位のPagedKvCache
- **投機的デコード** — レイヤースキップドラフト + デュアルモデル（1B→8B） + 確率的サンプリング（Leviathan et al.）
- **RoPE周波数スケーリング** — `rope_freqs.weight` テンソルによるNTK-awareコンテキスト拡張（Llama-3.1/3.2）
- **LLVM自動ベクトル化** — `target-cpu=native` でARM SDOT命令生成（Q4_K内積で37個のsdot）
- **スパースターナリ** — N:M構造化スパース性、2-bitパック、LUT+SDOT最適化、ブロックパックレイアウト
- **GPU推論 (wgpu)** — Metal/Vulkan/DX12コンピュートシェーダー、Q4_K/Q6_K脱量子化融合matvec、融合SwiGLU、バッチ4投機的デコード、トークン単位メモリ確保ゼロ、サブグループSIMDリダクション
- **Ternary QAT** — STE、L1正則化、AdamW、レイヤー単位混合精度

## アーキテクチャ

```
src/
├── lib.rs       — BPEトークナイザ、KVキャッシュ、アテンション、RoPE、サンプリング
├── gguf.rs      — GGUF v3パーサー、Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K量子化、融合matvec、
│                  ターナリ/スパースターナリ行列、LLVM自動ベクトル化SDOTカーネル
├── gpu.rs       — wgpuコンピュートエンジン: GpuEngine, GpuModel, バッチ4投機的デコード、
│                  トークン単位メモリ確保ゼロ、サブグループSIMD、Rc共有エンジン（デュアルモデル用）
├── shaders/     — WGSLコンピュートシェーダー（Q4_K/Q6_K matvec、融合SwiGLU、RMSNorm、RoPE、
│                  アテンション、KVキャッシュ、残差 — 各K=1スカラー + K=4バッチ版）
├── llama3.rs    — マルチアーキテクチャフォワードパス、モデル読み込み、投機的デコード（レイヤースキップ + デュアルモデル）、
│                  ページドKVキャッシュ、continuous batching、tied embeddings、RoPE周波数スケーリング
└── training.rs  — Ternary QAT: STE、QatLinear、L1正則化、AdamW、MSE損失

.cargo/
└── config.toml  — rustflags: target-cpu=native（LLVM SDOT自動ベクトル化有効化）
```

---

## GPU推論 (wgpu / Metal)

wgpuコンピュートシェーダーによるPure Rust GPU推論。外部MLフレームワーク不要 — 全カーネルをWGSLで手書き。

### クイックスタート

```bash
# 単一トークン自己回帰生成
cargo run --example generate_gpu --features gpu,gguf --release -- \
  --prompt "The meaning of life is" --max-tokens 64

# デュアルモデル投機的デコード（1Bドラフト + 8B検証）
cargo run --example speculative_dual_gpu --features gpu,gguf --release -- \
  --prompt "What is the capital of Japan?" --max-tokens 64
```

### GPU最適化の軌跡 (Apple M3 Metal, Llama-3.2-1B Q4_K_M)

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
                         125    100     80      60   ms/トークン
```

| フェーズ | 手法 | ms/トークン | 高速化 | 重要な知見 |
|---|---|---|---|---|
| 1 | 素朴なGPU matvec（Q4_K要素単位） | 125 | 1.0x | 脱量子化オーバーヘッドでGPU演算律速 |
| 1.5 | 融合脱量子化-matvec（単一カーネル） | 108 | 1.16x | 中間バッファ除去: 重みデコード + FMAを1パスで |
| 2 | サブグループSIMDリダクション (`subgroupAdd`) | 104 | 1.20x | レジスタ間シャッフルで共有メモリリダクションを置換 |
| 3 | RMSNorm + RoPE バッチ対応 | 97 | 1.29x | バッチ対応シェーダーでK=1の性能低下なし |
| 4 | 融合SwiGLU (gate+up+silu×mul) | 94 | 1.33x | 3ディスパッチを単一カーネルに統合、重み読み込み共有 |
| 5 | 2Dディスパッチ (row > 65535対応) | 84 | 1.49x | `wid.y * grid_x + wid.x` で大規模出力射影に対応 |
| **6** | **トークン単位メモリ確保ゼロ** | **71** | **1.76x** | **167個のバインドグループをロード時に全キャッシュ** |

### バッチ4投機的デコード

K=4展開スカラーアキュムレータ — 重みを1回デコード、4つのFMAをレジスタ内で実行:

```
K=1 (スカラー):  71 ms/トークン  (14.0 tok/s)
K=4 (バッチ):   101 ms/4トークン (25.3 ms/トークン, 39.5 tok/s)
高速化:         2.82×
正確性:         max|diff| = 0.000000 (PASS)
```

**デュアルパイプラインアーキテクチャ**: オリジナルのスカラーシェーダー（K=1高速パス） + K=4展開バッチシェーダー。K=1の性能低下なし — `var acc0, acc1, acc2, acc3` が物理GPUレジスタを保証（`var acc: array<f32, 4>` はスレッドメモリにスピルする）。

### デュアルモデルGPU投機的デコード (1B → 8B)

```
8B ベースライン (K=1):   0.4 tok/s  (2,773 ms/トークン)
1B+8B 投機的:            2.1 tok/s  (471 ms/トークン)
高速化:                  5.89×
受理率:                  90% (19/21)
```

両モデルは単一の `GpuEngine`（`Rc<GpuEngine>`）を共有 — 同一のwgpu Device/Queue、独立したKVキャッシュ。

### WGSLシェーダーアーキテクチャ

| シェーダー | K=1 (スカラー) | K=4 (バッチ) | バインディング |
|---|---|---|---|
| `dequant_matvec_q4k` | acc 1個, subgroupAdd 1回 | acc 4個, subgroupAdd 4回 | weights, input, output, params |
| `dequant_matvec_q6k` | acc 1個, subgroupAdd 1回 | acc 4個, subgroupAdd 4回 | weights, input, output, params |
| `swiglu_fused_q4k` | acc 2個 (gate+up) | acc 8個 (gate×4+up×4) | gate_w, up_w, input, output, params |
| `rmsnorm` | `wid.x` でバッチ | — | input, weights, output, params |
| `rope` | `wid.y` でバッチ | — | data, params |
| `attention` | `wg.y` でバッチ (因果) | — | Q, K_cache, V_cache, output, params |
| `kv_cache_append` | `wid.y` でバッチ | — | input, K/V_cache, params |
| `residual_add` | 要素単位 | — | a, b |

---

## CPU最適化の軌跡 (70Bスパースターナリ)

70Bモデルのスパースターナリ matvec を、スカラー実装から M1 Pro メモリ帯域の物理限界まで最適化した全記録。

### 壁

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

### Phase 1 → スカラーベースライン (0.16 tok/s, 1% BW)

素朴な実装。16-bit マスクからビットを1つずつ取り出し、`if bit { sum += x }` の分岐地獄。

```rust
// 70B: 76.5 ms/layer × 80 layers = 6,120 ms/token
for bit in 0..16 {
    if active_mask & (1 << bit) != 0 {
        let sign = if sign_mask & (1 << bit) != 0 { -1.0 } else { 1.0 };
        sum += sign * input[col + bit];  // 重みごとに分岐
    }
}
```

**ボトルネック**: 分岐予測ミス。50%スパースでほぼランダムな分岐パターン。

### Phase 2 → NEON マスク展開 (0.26 tok/s, 2% BW, 1.6x)

16-bit マスクを NEON `vtstq_u16` で一括展開。8回の `expand_mask_4` → 1回の `expand_mask_16`。

```rust
// 16分岐の代わりに1回のNEON命令で16重みを展開
let mask_vec = vdupq_n_u16(active_mask);
let expanded = vtstq_u16(mask_vec, bit_positions);  // 16レーン同時処理
```

**発見**: 分岐除去だけでは不十分。データ型が f32 のまま — 帯域を浪費している。

### Phase 3 → ブランチレス4行マイクロカーネル (0.37 tok/s, 3% BW, 2.3x)

4行を同時処理するレジスタブロッキング。活性化ベクトルの読み込みを4行で共有。**ゼロ分岐**。

```rust
// 4行が同じ活性化ベクトルを共有 — 4倍のレジスタ再利用
let act = vld1q_f32(input.as_ptr().add(col));
acc0 = vmlaq_f32(acc0, w0, act);  // 行0
acc1 = vmlaq_f32(acc1, w1, act);  // 行1
acc2 = vmlaq_f32(acc2, w2, act);  // 行2
acc3 = vmlaq_f32(acc3, w3, act);  // 行3
```

**発見**: f32 演算は速いが、**f32 の帯域消費が本当の敵**。入力ベクトルは32-bit精度で毎ブロック読み込まれ、帯域の大半を食っている。

### Phase 4 → i8 動的量子化 + SDOT (0.70 tok/s, 6% BW, 4.4x)

活性化ベクトルを f32→i8 に動的量子化。SDOT 命令で 16×i8×i8 → 4×i32 を**1命令**で実行。

```rust
// NEON加速 f32 → i8 量子化
let amax = vmaxvq_f32(vabsq_f32(chunk));  // 1命令でグローバル最大値
let scale = 127.0 / amax;
let quantized = vqmovn_s16(vcombine_s16(
    vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(v0, vscale))),
    vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(v1, vscale))),
));

// SDOT: 1命令で16個の積和演算（インラインアセンブリ）
// sdot v_acc.4s, v_weights.16b, v_activations.16b
core::arch::asm!(
    "sdot {acc:v}.4s, {w:v}.16b, {a:v}.16b",
    acc = inout(vreg) acc, w = in(vreg) weights, a = in(vreg) activations,
);
```

帯域消費 1/4 (f32→i8)。演算スループット 4x (FMA→SDOT)。**しかし 6% BW — まだ遅い。**

**発見**: 重みの展開コストが支配的になった。16-bit マスク → i8 重みベクトルへの展開に14命令も使っている。

### Phase 5 → vqtbl1q LUT展開 (1.59 tok/s, 27% BW, 9.9x)

**ターニングポイント**。2-bit packed weights（00=0, 01=+1, 11=−1）を NEON テーブルルックアップで 6命令展開。

```rust
// 2-bitパック形式: 1バイトに4重み (00=0, 01=+1, 11=-1)
// 16重み = 4バイト (f32なら32バイト)

// LUT: 2-bitインデックス → i8重み
let lut = [0i8, 1, 0, -1, 0, 0, 0, 0, ...];  // vqtbl1qルックアップテーブル

unsafe fn expand_packed_2bit_lut(packed_ptr: *const u8, ...) -> int8x16_t {
    let raw = vld1q_u8(packed_ptr);              // 4バイト読み込み（16重み）
    let replicated = vqtbl1q_u8(raw, rep_idx);   // レーンにバイト複製
    let shifted = vshlq_s8(replicated, shifts);   // シフトで2-bitペア分離
    let indices = vandq_u8(shifted, mask_03);     // 2-bitインデックスにマスク
    vqtbl1q_s8(lut, indices)                      // LUT: インデックス → {0, +1, -1}
}
// 合計6命令。以前のマスク方式: 14命令。
```

**27% → しかし Phase 6 で真の原因が判明する。**

Phase 5 で 9.9x に到達したとき、次の疑問が浮かんだ: **残りの 73% はどこに消えている？**

ソフトウェアプリフェッチ (`prfm`)、E-core 除外、チャンクサイズ変更 — すべて**悪化**した（後述）。これらの失敗から、真のボトルネックが**演算でもプリフェッチでもない**ことが証明された。

### Phase 6 → ブロックパック重みレイアウト (1.76 tok/s, 45% BW, 11.0x)

**TLBミスの発見と殲滅**。

従来のメモリレイアウト（Row-major）:
```
Row 0: [blk0][blk1][blk2]...[blk511]    ← 1行2KB
Row 1: [blk0][blk1][blk2]...[blk511]    ← 2KB、別ページ
Row 2: [blk0][blk1][blk2]...[blk511]    ← 2KB、別ページ
Row 3: [blk0][blk1][blk2]...[blk511]    ← 2KB、別ページ
       ↑ 4行カーネルは4つの異なるページからblk0を読む → TLBミス4回
```

ブロックパックレイアウト:
```
Group 0: [R0-blk0][R1-blk0][R2-blk0][R3-blk0] [R0-blk1][R1-blk1]...
         ↑ 4行のblk0が連続 → TLBミス0回、シーケンシャルリード
```

```rust
// ブロックパックレイアウト構築: ブロック単位で4行インターリーブ
let num_groups = (num_rows + 3) / 4;
for g in 0..num_groups {
    for blk in 0..blocks_per_row {
        for lane in 0..4 {  // ブロックごとに4行を連続配置
            packed_blocked[dst] = packed_2bit[src];  // シーケンシャル書き込み
        }
    }
}

// マイクロカーネルはシーケンシャルに読み込み — ストライドなし、TLBミスなし
let blk_base = group_ptr.add(blk * stride_4);
let w0 = expand_packed_2bit_lut(blk_base);                          // 行0
let w1 = expand_packed_2bit_lut(blk_base.add(bytes_per_block));     // 行1（隣接！）
let w2 = expand_packed_2bit_lut(blk_base.add(2 * bytes_per_block)); // 行2
let w3 = expand_packed_2bit_lut(blk_base.add(3 * bytes_per_block)); // 行3
```

**27% → 45%。TLBミスの除去だけで帯域利用率が 1.67x。**

### 天井: 45% = M1 Pro CPUの物理限界

M1 Pro の SoC 帯域 200 GB/s は GPU/NPU/Media Engine と共有。CPU が単独で使える帯域は約 90 GB/s（45%）。これは OS カーネルとメモリコントローラのアービトレーションによるハード制約であり、ソフトウェアでは突破できない。

### まとめ

| フェーズ | 手法 | tok/s | ms/レイヤー | BW | 高速化 | 重要な知見 |
|---|---|---|---|---|---|---|
| 1 | スカラーベースライン | 0.16 | 76.5 | 1% | 1.0x | ランダムスパース性で分岐予測が崩壊 |
| 2 | NEON マスク展開 | 0.26 | 47.6 | 2% | 1.6x | ブランチレスだがf32帯域浪費 |
| 3 | 4行マイクロカーネル | 0.37 | 33.9 | 3% | 2.3x | レジスタブロッキング — 活性化再利用 |
| 4 | i8量子化 + SDOT | 0.70 | 17.9 | 6% | 4.4x | 帯域4倍削減 + 演算4倍 |
| 5 | vqtbl1q LUT | 1.59 | 7.9 | 27% | 9.9x | 6命令 vs 14 — 重み展開が支配的 |
| **6** | **ブロックパッキング** | **1.76** | **7.1** | **45%** | **11.0x** | **TLBミス除去 → 帯域天井** |

### 効果がなかった試み（同等に価値あるデータ）

| 試み | 結果 | 根本原因 |
|---|---|---|
| ソフトウェアプリフェッチ (`prfm`) | **−10%** | M1 Pro HWプリフェッチャーが既に最適。`prfm` はL1キャッシュを汚染 |
| E-core除外（P-core 4スレッドのみ） | **−14%** | Apple UMA: E-coreは個別には遅くても帯域に貢献 |
| チャンクサイズ 8→16行 | **−16%** | Rayonワークスティーリングの不均衡。8 = M1 Pro L1ジオメトリに最適 |
| 2倍ブロックアンローリング | **−3%** | acc0–acc3は既に独立。追加ループオーバーヘッドのみ |

**失敗の教訓**: M1 Pro は HW プリフェッチャーが極めて優秀で、ソフトウェアの介入は cache pollution を引き起こす。Apple UMA では E-core を除外すると帯域の一部を捨てることになる。これらの実験が「27% の壁」の真因が TLB ミスであることの**消去法的証明**となった。

---

## 投機的デコード

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

### 確率的投機サンプリング (Leviathan et al.)

```
各ドラフトトークン x（ドラフトlogits q、検証logits p）について:
    p_dist = softmax(p), q_dist = softmax(q)
    if rand() < min(1, p_dist[x] / q_dist[x]):
        受理 x                                             ← 分布が近ければ高確率で受理
    else:
        棄却 → normalized max(0, p_dist - q_dist) から再サンプル  ← 不足分から補正サンプル
全K個のドラフトが受理された場合:
    最終検証logitsからボーナストークン                        ← サイクルあたりK+1トークン
```

### 8B 実測結果: Greedy vs 確率的

| ドラフト層数 | 層% | Greedy α | **確率的 α** | 速度 |
|---|---|---|---|---|
| ベースライン | 100% | — | — | **5.9 tok/s** |
| 24 | 75% | 4% | 17% | 2.1 tok/s |
| 28 | 87.5% | 13% | 17% | 2.4 tok/s |
| 30 | 93.8% | 37% | **25–75%** | 2.5 tok/s |
| 31 | 96.9% | 58% | **59–62%** | 2.9–3.1 tok/s |

**8Bの限界**: 32層モデルでは各層の表現寄与が大きすぎ、層スキップの受理率がドラフトコストに見合わない。70B（80層）ではスキップ耐性が大幅に向上する。

### デュアルモデル投機的デコード (1B → 8B)

別モデルをドラフトに使う真の投機的デコード。Llama-3.2-1B (16層, 2048dim) → Llama-3.1-8B (32層, 4096dim):

```bash
cargo run --release --example speculative_dual --features "gguf,parallel" -- \
  --model models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --draft-model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --prompt "What is the capital of Japan?" --max-tokens 60
```

| モード | tok/s | 受理率 | 高速化 |
|---|---|---|---|
| ベースライン (8Bのみ) | 5.7 | — | 1.0x |
| 投機的 K=3 | 3.3 | 54% | 0.58x |
| 投機的 K=4 | 5.6 | 63% | 0.99x |
| 投機的 K=5 | 6.5 | 100% | 1.15x |

**CPU上の制約**: バッチ並列検証が不可能なため、ドラフトモデルのコストが純粋なオーバーヘッドになる。短い応答（K個全受理）では1.15xのスピードアップを確認。GPU上ではバッチ検証により大幅な高速化が期待できる。

### 70B 投機的デコード予測

| spec_k | ドラフト層数 | 受理率 α | 実効 tok/s | 高速化 |
|---|---|---|---|---|
| 4 | 8 (10%) | 60% | 2.51 | 1.4x |
| 3 | 8 (10%) | 70% | 2.39 | 1.3x |
| 4 | 8 (10%) | 80% | **3.10** | **1.7x** |

---

## 計算パス (Q4_K)

llama.cpp 互換の整数内積パス:

1. 入力ベクトルを **Q8_K** に量子化 (256要素/ブロック, f32 scale, i8 values, i16 bsums)
2. 重み × 入力を **Q4_K×Q8_K** / **Q6_K×Q8_K** 整数内積で計算（乗算はスケール適用時のみ）
3. 8サブブロック × 32要素の整数累積、6-bit packed scales/mins
4. 最終結果: `d × d_q8 × Σ(scale × q4 × q8) − dmin × d_q8 × Σ(mins × bsums)`

llama.cpp の `ggml_vec_dot_q4_K_q8_K` と完全一致（logits差 ±0.03 以内）。

## マルチアーキテクチャサポート

GGUF の `general.architecture` から自動検出:

| アーキテクチャ | GQA | RoPE θ | スライディングウィンドウ | Logitソフトキャッピング |
|---|---|---|---|---|
| Llama-3 | 4:1 | 500,000 | — | — |
| Mistral | 4:1 | 10,000 | 4096 | — |
| Gemma-2 | 設定可能 | 設定可能 | オプション | アテンション + 最終 |

## スパースターナリ量子化

N:M 構造化スパース性による極限圧縮:

- **N:M 構造化スパース性**: 16要素中 N 個だけ非ゼロ (8:16 = 50% density)
- **2-bitパックエンコーディング**: `00=0, 01=+1, 11=-1` — 1バイトに4重み
- **ブロックパックレイアウト**: 4行インターリーブ → TLB ミス消滅
- **vqtbl1q_s8 LUT**: 2-bit → i8 を 6 命令でオンザフライ展開
- **SDOT**: 16×i8×i8 → 4×i32（1命令、インラインアセンブリ）
- **i8 動的量子化**: NEON 加速 f32→i8 活性化量子化

## Ternary QAT（量子化再学習）

BitNet b1.58 スタイルの学習時量子化:

- **Straight-Through Estimator**: forward で ternarize、backward で勾配素通し
- **L1 正則化**: `λ × Σ|w|` で重みの 30%+ を 0 に収束
- **AdamW**: bias correction + weight decay
- **レイヤー単位混合精度**: Attention=1.58bit / FFN=1bit+sparse

```
70B 実効ビット/パラメータ推定:
  Attention (30%): 1.58 bit (ternary)
  FFN (70%):       0.92 bit (sparse ternary 8:16)
  → 全体平均:      ~1.1 bit/param → 70B ≈ 9.6 GB
```

## 性能

### 1B モデル (Llama-3.2-1B-Instruct Q4_K_M, M1 Pro)

| 構成 | デコード速度 |
|---|---|
| 全16層推論 | **20.2 tok/s** |
| llama.cppとのlogit精度 | ±0.09 (top-1トークン一致) |

### 8B モデル (ELYZA-JP Q4_K_M, M1 Pro)

| フェーズ | 構成 | デコード速度 | プリフィル (16トークン) |
|---|---|---|---|
| Phase 1 | バグ修正（脱量子化→整数内積） | 1.2 tok/s | 14.2s |
| Phase 2 | + Q8_K再利用、自動ベクトル化、Rayon | 5.1 tok/s | 4.1s |
| Phase 3 | + Contiguous KVキャッシュ | **5.9 tok/s** | **3.3s** |

### 70B スパースターナリ（シミュレーション、M1 Pro）

```bash
cargo run --release --example bench_70b_sparse --features "gguf,parallel"
```

| 射影 | サイズ | 時間/反復 |
|---|---|---|
| Q proj | 8192×8192 | 0.54 ms |
| K proj | 1024×8192 | 0.10 ms |
| V proj | 1024×8192 | 0.10 ms |
| O proj | 8192×8192 | 0.53 ms |
| Gate (FFN) | 28672×8192 | 1.70 ms |
| Up (FFN) | 28672×8192 | 2.00 ms |
| Down (FFN) | 8192×28672 | 2.17 ms |
| **1レイヤー** | | **7.1 ms** |
| **1トークン (80レイヤー)** | | **569 ms → 1.76 tok/s** |

### デバイス別推定性能（帯域律速）

| デバイス | メモリ帯域 | 推定 tok/s |
|---|---|---|
| Raspberry Pi 5 | 34 GB/s | 0.7 |
| Mac Mini M4 | 120 GB/s | 2.3 |
| M1 Pro (実測) | 91 GB/s | 1.8 |
| Mac Mini M4 Pro | 273 GB/s | 5.3 |
| Mac Studio M4 Ultra | 800 GB/s | 15.6 |

### 量子化ブロック仕様

| 型 | ブロックサイズ | バイト/ブロック | bpw | レイアウト |
|---|---|---|---|---|
| Q2_K | 256 | 84 | 2.625 | scales[16] + qs[64] + d(f16) + dmin(f16) |
| Q3_K | 256 | 110 | 3.4375 | hmask[32] + qs[64] + scales[12] + d(f16) |
| Q4_K | 256 | 144 | 4.5 | d(f16) + dmin(f16) + scales[12] + qs[128] |
| Q5_K | 256 | 176 | 5.5 | d(f16) + dmin(f16) + scales[12] + qh[32] + qs[128] |
| Q6_K | 256 | 210 | 6.5625 | ql[128] + qh[64] + scales[16] + d(f16) |
| Q8_0 | 32 | 34 | 8.5 | d(f16) + qs[32] |

混合量子化バリアント（`_S`, `_M`, `_L`）はレイヤーごとに異なる型を使用 — 例: アテンションにQ3_K、FFNにQ4_K/Q5_K、embeddingsにQ6_K、normにF32 — そのため実効bpwはベース型より高くなる。

### 30B / 70B モデルサイズ推定

| 量子化 | bpw (実効) | 30B | 70B |
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

### デバイスメモリ別推奨量子化

| メモリ | 30B モデル | 70B モデル |
|---|---|---|
| **16 GB** (MBA M3) | Q2_K (12GB) ○ / Q3_K_S (13GB) △ | × |
| **24 GB** (M3 Pro) | Q4_K_M (18GB) ○ | × |
| **32 GB** (M3 Max) | Q6_K (25GB) ○ | Q2_K (28GB) △ |
| **36 GB** (M3 Pro) | Q6_K (25GB) ○ | Q3_K_S (31GB) △ |
| **48 GB** (M4 Max) | Q8_0 (32GB) ○ | Q3_K_M (34GB) ○ |
| **64 GB** (M2 Ultra) | F16 (60GB) △ | Q4_K_M (42GB) ○ |
| **128 GB** (M4 Ultra) | F16 (60GB) ○ | Q8_0 (74GB) ○ |

○ = 快適（モデル + OS + KVキャッシュがRAMに収まる）、△ = スワップ圧力あり

## 推論サーバー（OpenAI互換API）

```bash
cargo run --bin alice-llm-server --features server --release -- \
  --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --port 8090
```

```bash
# テキスト補完
curl http://localhost:8090/v1/completions -H "Content-Type: application/json" \
  -d '{"prompt":"What is the capital of Japan?","max_tokens":32,"temperature":0}'

# レスポンス:
# {"choices":[{"text":"The capital of Japan is Tokyo.","finish_reason":"stop"}],
#  "usage":{"tokens_per_second":14.8,"decode_ms":473}}

# ヘルスチェック
curl http://localhost:8090/health

# モデル一覧
curl http://localhost:8090/v1/models
```

## CLIオプション

```
--model <path>            GGUFモデルファイルのパス
--prompt <text>           入力プロンプト
--max-tokens <n>          最大生成トークン数（デフォルト: 256）
--temperature <f>         サンプリング温度（デフォルト: 0.7、0.0 = greedy）
--speculative-k <n>       投機的デコード: K個のトークンをドラフト（デフォルト: 0 = 無効）
--draft-layers <n>        ドラフトモデルのレイヤー数（デフォルト: 8）
--ternary                 ターナリ量子化重みを使用
--ternary-threshold <f>   ターナリスパース性閾値（デフォルト: 0.7）
```

## Cargo Features

| Feature | 説明 |
|---|---|
| `gguf` | GGUFファイル読み込みとマルチアーキテクチャ推論 |
| `gpu` | wgpu GPUコンピュート（Metal/Vulkan/DX12）、`gguf` 必須 |
| `server` | HTTP推論サーバー（axum）、`gpu` + `gguf` を含む |
| `parallel` | RayonベースのマルチスレッドCPU matvec |

## ライセンス

**デュアルライセンス: AGPL-3.0+ または商用**

ALICE-LLMは **GNU Affero General Public License v3.0以降 (AGPL-3.0+)** の下でオープンソースかつ無償で利用できます。これにより、このエンジンを使用した変更やネットワークベースのSaaSデプロイメントがオープンソースコミュニティに還元されることが保証されます。

**商用ライセンス**
ALICE-LLMをプロプライエタリSaaS環境、クラウドインフラ、またはクローズドソース商用製品でバックエンドソースコードを開示せずに使用する場合は、**商用ライセンス**の取得が必要です。

商用ライセンスのお問い合わせとエンタープライズサポート: licensing@alicelaw.net
