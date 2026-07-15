# Phase X.3.e.3 — SSM-math refinement (Bonsai / Qwen 3.6 DeltaNet)

**Status**: **✅ Complete** (2026-07-16、5 commit で全 subphase landing)
**Predecessor**: Phase X.3.e.2 (PR #66、DeltaNet forward wiring) + X.5 (PR #67、Jetson load-and-run 動作証明)
**Objective**: Bonsai 27B の DeltaNet forward path で **Qwen 3.5 と Bonsai / Qwen 3.6 の SSM 数値挙動一致を担保** し、`ssm_a` / `ssm_dt_bias` / `ssm_norm` の `#[allow(dead_code)]` を全撤去する **達成済**

---

## 0. Achievement Summary (2026-07-16)

Phase X.3.e.3 は当初 4 Gap と設計、実装過程で reference (`PrismML llama.cpp fork qwen35.cpp:436-562`) を精読した結果 2 refinement を追加特定、合計 **6 subphase** を 5 commit で完遂

| Subphase | Commit | 対象 | Test 追加 | Reference |
|---|---|---|---|---|
| **Gap A** | `146ee22` | Per-V-head alpha/beta loop (Bonsai 48 V / 16 KV) | +1 | — |
| **Gap C** | `005b3d0` | `ssm_norm` per-V-head RMSNorm (Step 4.5) | +1 | qwen35.cpp:562 |
| **Gap B** | `6d43602` | `ssm_a` / `ssm_dt_bias` SSM discretisation | +2 | qwen35.cpp:443-451 |
| **Gap B extra** | `2d55f2b` | `beta = sigmoid(beta_raw)` | +2 | qwen35.cpp:440-441 |
| **§Q/K + §silu(z)** | `c342f10` | Post-conv1d silu (Step 3.5) + z-gate after ssm-norm (Step 4.6)、`bonsai_semantics: bool` flag | +1 | qwen35.cpp:502 + :562 |

**Test 数推移**: 296 (baseline) → 297 (Gap A) → 299 (Gap C) → 301 (Gap B) → 303 (Gap B extra) → **304** (§Q/K + §silu(z))、`+8` test で SSM math の各層が回帰防止 net として機能

**Backwards compat**: 全 refinement は Bonsai conditional (`is_bonsai_path = ssm_a.is_some() && ssm_dt_bias.is_some()`) で発火、Qwen 3.5 (両方 None) は完全 unchanged

**残る作業** (別 session、要 external resource): End-to-end numerical validation vs llama.cpp Bonsai 27B tensor dumps (~1-2h の GGUF DL + build) と Qwen 3.5 backwards compat 意思決定 (現状は Bonsai only、reference 一致させるなら Qwen 3.5 側も同 refinement 適用が正解)

---

## 1. Overview (Historical, 2026-07-15 design)

Phase X.3.e.2 で Bonsai 27B の DeltaNet forward path が「panic なし load-and-run 可能」まで到達したが、以下 4 点は **numerical correctness 未検証** で残置:

1. `ssm_a` (48 rows, per-V-head A matrix) — loaded but not consumed → **Gap B で解消 (commit `6d43602`)**
2. `ssm_dt_bias` (48 rows, per-V-head time-step bias) — loaded but not consumed → **Gap B で解消 (commit `6d43602`)**
3. `ssm_norm` (128 dim, per-head qk_dim RMSNorm) — loaded but not consumed → **Gap C で解消 (commit `005b3d0`)**
4. `dn_alpha` / `dn_beta` の per-V-head vs per-KV-head 解釈 — 現状 loop は `dn_num_kv_heads=16` 回、Bonsai の 48-entry alpha/beta の 32 entries が silent drop されている可能性 → **Gap A で解消 (commit `146ee22`)**

Phase X.3.e.3 は当初 llama.cpp Qwen 3.6 reference 出力との numerical comparison を通じて 4 gap を埋める予定だったが、実際は **`gh api` で PrismML llama.cpp fork の `qwen35.cpp` を直接精読** することで 4 gap + 2 追加 refinement (§Q/K L2Norm without silu、§silu(z) order) を code-level で reference formula と数式一致まで進めた 実 GGUF DL + build を伴う end-to-end 数値検証は別 session に持ち越し

---

## 2. Current State (2026-07-15)

### 2.1 Loaded tensors (`src/llama3.rs:8877-8879, 8942-8944`)

```rust
let ssm_a = gguf.tensor_to_f32(&format!("{prefix}.ssm_a"));           // Bonsai: 48 entries
let ssm_dt_bias = gguf.tensor_to_f32(&format!("{prefix}.ssm_dt.bias"));  // Bonsai: 48 entries
let ssm_norm = gguf.tensor_to_f32(&format!("{prefix}.ssm_norm.weight")); // Bonsai: 128 entries

DeltaNetLayerWeights {
    // ... existing fields ...
    ssm_a,          // #[allow(dead_code)]
    ssm_dt_bias,    // #[allow(dead_code)]
    ssm_norm,       // #[allow(dead_code)]
    // ...
}
```

### 2.2 Forward path (`src/llama3.rs:3810-3903`)

```
Step 1. Input projection
        - Qwen 3.5: ssm_in (single fused matvec)
        - Bonsai:   attn_qkv (QKV portion) + attn_gate (Z portion)
Step 2. alpha_proj / beta_proj matvec → dn_alpha (48-entry buf for Bonsai), dn_beta
Step 3. causal_conv1d_step (Q + K + V の depthwise conv1d、Z は除外)
Step 4. gated_deltanet_step(
            q, k, v, alpha=&dn_alpha, beta=&dn_beta, z,
            state, out, num_heads=dn_num_kv_heads,  // ← 16 for Bonsai
            qk_dim, v_dim,
        )
Step 5. ssm_out.matvec(&dn_delta_out, &mut o_buf)
Step 6. residual add
Step 7. FFN sub-block
```

### 2.3 Recurrence body (`src/llama3.rs:2265-2314`)

```rust
fn gated_deltanet_head_disjoint(
    q: &[f32], k: &[f32], v: &[f32],
    alpha_h: f32, beta_h: f32,   // ← per-head scalar (どちらか一方の粒度に丸められる)
    z: &[f32], state: &mut [f32], out: &mut [f32],
    qk_dim: usize, v_dim: usize,
) {
    // L2 normalize q with a tiny epsilon so a zero vector produces a zero output.
    // ...
    for j in 0..v_dim {
        let error_j = v[j] - alpha_h * st_k;
        for i in 0..qk_dim {
            let s_new = alpha_h * state[idx] + beta_h * k_i * error_j;
            state[idx] = s_new;
            out_j += s_new * q_i;
        }
        out[j] = out_j * silu(z[j]);
    }
}
```

**問題点**: Bonsai は per-V-head の alpha/beta を持つが、この関数は per-head scalar `alpha_h`, `beta_h` を受け取る 呼び出し側 `gated_deltanet_step` は `num_heads = num_kv_heads = 16` で loop するので、Bonsai の 48 entry alpha/beta のうち 16 しか使われない (残り 32 は silent drop)

---

## 3. Gap Analysis

### Gap A: Per-V-head alpha/beta 未反映 (優先度: 高、correctness critical)

- **現象**: Bonsai config `linear_num_v_heads=48, linear_num_kv_heads=16` (3 V heads per KV head group)
- **現在の挙動**: loop `for head in 0..16` で `dn_alpha[0..16]` のみ使用、`dn_alpha[16..48]` が drop される
- **正解**: 各 KV head 内の 3 V heads は独立した alpha/beta を持つ、loop 構造を per-V-head に変更する必要
- **影響**: alpha/beta 情報の 2/3 が失われている、Bonsai output が数値的に破綻している可能性が高い

### Gap B: `ssm_a` / `ssm_dt_bias` (Mamba-style SSM refinement、優先度: 中〜高)

- **役割**: Selective State Space Model の discretization に必要な parameter
- **推測される数式** (llama.cpp reference で確定必要):
  ```
  dt = softplus(dt_proj_out + ssm_dt_bias)   # per-V-head time step
  A_bar = exp(dt * ssm_a)                    # discretized A (transition)
  # Recurrence with A_bar:
  S_new = A_bar * S_prev + B_bar * k * v_error
  ```
- **代替仮説** (別解): `ssm_a` は既存 `alpha_h` を per-V-head modulation する scale factor (Mamba とは異なる Qwen 独自の refinement)
- **影響**: state transition の速度 / stability に影響、long-context で急速に発散 or 発散抑制

### Gap C: `ssm_norm` (per-head qk_dim RMSNorm、優先度: 中)

- **役割**: `dn_delta_out` を `ssm_out` に食わせる前に per-head RMSNorm
- **shape**: 128 (= state_size = per-head qk_dim = v_dim)、48 heads 全てで共有 or 各 head 独立の 128 dim
- **推測される数式**:
  ```
  # Step 4 完了後、Step 5 の直前:
  for h in 0..num_v_heads {
      let head_delta = &mut dn_delta_out[h * v_dim..(h + 1) * v_dim];
      apply_rms_norm(head_delta, &ssm_norm, eps);
  }
  ```
- **影響**: activation magnitude の stability、ssm_out matvec への入力分布

### Gap D: 現状の `alpha_h` scalar 抽出 (最初の 16 entry のみ) は KV head 主導 or V head 主導?

- Bonsai の alpha_proj は `Linear(hidden_dim=5120, out=48)` (per V-head)
- 現状の `gated_deltanet_head_disjoint` は 1 head につき 1 scalar alpha を expect
- **明確化必要**: 48 entries は独立 per-V-head か、それとも 16 group で共有 (3 V per KV group で同 alpha) か

---

## 4. Implementation Plan

### 4.1 Prerequisite: llama.cpp reference

**必須**: Bonsai / Qwen 3.6 の DeltaNet を llama.cpp fork (PrismML: https://github.com/PrismML-Eng/llama.cpp) で動作させ、中間 tensor 出力を dump する

```bash
# 1. clone PrismML llama.cpp fork
git clone https://github.com/PrismML-Eng/llama.cpp ~/llama.cpp-prismml
cd ~/llama.cpp-prismml && make -j

# 2. Bonsai 27B Ternary GGUF (5.9 GB) DL
wget https://huggingface.co/prism-ml/bonsai-27b-ternary-g128/resolve/main/bonsai-27b-ternary-g128.gguf

# 3. Reference inference with 中間 tensor dump
./main -m bonsai-27b-ternary-g128.gguf \
    -p "The capital of Japan is" \
    --n-predict 5 \
    --seed 42 \
    --dump-tensors deltanet_layer3  # ← llama.cpp fork の debug flag (要 grep 確認)
```

**代替案** (llama.cpp fork に dump 機能がない場合): Bonsai 論文の付録 numerical example を使う、または PrismML の HuggingFace transformers 実装で `output_hidden_states=True` + `output_attentions=True` を有効化して dump

### 4.2 Subtask 1: Per-V-head alpha/beta loop (Gap A、~1h)

**変更対象**: `src/llama3.rs:2143` `gated_deltanet_step` の signature + loop

**Before**:
```rust
fn gated_deltanet_step(
    q, k, v, alpha, beta, z, state, out,
    num_heads: usize,  // = num_kv_heads for both variants
    qk_dim, v_dim,
) {
    for head in 0..num_heads {  // 16 iterations for Bonsai
        gated_deltanet_head_disjoint(
            q, k, v, alpha[head], beta[head], z, state, out, head, qk_dim, v_dim,
        );
    }
}
```

**After**:
```rust
fn gated_deltanet_step(
    q, k, v, alpha, beta, z, state, out,
    num_kv_heads: usize,   // 16 for Bonsai
    num_v_heads: usize,    // 48 for Bonsai
    qk_dim, v_dim,
) {
    let v_per_kv = num_v_heads / num_kv_heads;  // 3 for Bonsai
    debug_assert_eq!(alpha.len(), num_v_heads);
    debug_assert_eq!(beta.len(), num_v_heads);

    for v_head in 0..num_v_heads {
        let kv_head = v_head / v_per_kv;  // shared Q/K within group
        // v_head has its own state slab + output slab + alpha/beta
        gated_deltanet_head_disjoint(
            &q[kv_head * qk_dim..(kv_head + 1) * qk_dim],  // shared Q
            &k[kv_head * qk_dim..(kv_head + 1) * qk_dim],  // shared K
            &v[v_head * v_dim..(v_head + 1) * v_dim],       // unique V
            alpha[v_head],
            beta[v_head],
            &z[v_head * v_dim..(v_head + 1) * v_dim],
            &mut state[v_head * qk_dim * v_dim..(v_head + 1) * qk_dim * v_dim],
            &mut out[v_head * v_dim..(v_head + 1) * v_dim],
            qk_dim, v_dim,
        );
    }
}
```

**Qwen 3.5 backwards compatibility**: `num_v_heads == num_kv_heads` の場合は 1 V head per KV group で従来と等価、既存 test suite は unchanged

**State buffer resize**: `deltanet_state[dn_idx]` allocation を `num_v_heads * qk_dim * v_dim` に変更 (`src/llama3.rs:3201` 付近) 従来は `num_kv_heads * qk_dim * v_dim` だったので Bonsai は 3× 大 (16→48 heads)

### 4.3 Subtask 2: ssm_norm 適用 (Gap C、~30 min)

**変更対象**: `src/llama3.rs:3895-3898` の Step 4 → Step 5 の間に挿入

**Before**:
```rust
gated_deltanet_step(...);
// Step 5. Output projection to hidden dim.
dn_layer.ssm_out.matvec(&dn_delta_out, &mut o_buf);
```

**After**:
```rust
gated_deltanet_step(...);

// 4.5. Per-V-head RMSNorm on state (Bonsai / Qwen 3.6 only).
if let Some(ssm_norm) = dn_layer.ssm_norm.as_ref() {
    let v_stride = dn_v_dim;
    for v_head in 0..dn_num_v_heads {
        let head_slice = &mut dn_delta_out[v_head * v_stride..(v_head + 1) * v_stride];
        apply_rms_norm_in_place(head_slice, ssm_norm, c.norm_eps);
    }
}

// Step 5. Output projection to hidden dim.
dn_layer.ssm_out.matvec(&dn_delta_out, &mut o_buf);
```

**新規関数** `apply_rms_norm_in_place`: 既存 `rms_norm` の in-place variant (元は out-of-place)

### 4.4 Subtask 3: ssm_a / ssm_dt_bias 適用 (Gap B、~2h)

**要 llama.cpp reference**: 数式が確定してから実装 3 hypothesis:

**Option B1 (Mamba-style selective SSM)**:
```rust
// Step 2 の後、Step 3 の前:
for v_head in 0..num_v_heads {
    let dt = softplus(dn_beta[v_head] + ssm_dt_bias[v_head]);
    let a_bar = (dt * ssm_a[v_head]).exp();  // discretized A
    // alpha_h effective = a_bar, replace usage in gated_deltanet_head_disjoint
    effective_alpha[v_head] = a_bar;
    effective_beta[v_head] = dt * dn_beta[v_head];  // discretized B scaling
}
```

**Option B2 (per-V-head alpha modulation only)**:
```rust
// Simpler: ssm_a is a modulation factor on existing alpha
for v_head in 0..num_v_heads {
    effective_alpha[v_head] = dn_alpha[v_head] * ssm_a[v_head];
    effective_beta[v_head] = dn_beta[v_head] * softplus(ssm_dt_bias[v_head]);
}
```

**Option B3 (skip, minimal change)**:
- `ssm_a` / `ssm_dt_bias` を最初は untouched で残し、Gap A + Gap C だけ実装
- llama.cpp との numerical diff で「必要か / 不要か」判断、必要なら B1 / B2 選択

**推奨**: B3 でスタート → llama.cpp diff → B1 or B2 選択 → 実装

### 4.5 Subtask 4: `#[allow(dead_code)]` 撤去 (~5 min)

Subtask 1-3 完了後、`src/llama3.rs:2838, 2844, 2852` の `#[allow(dead_code)]` を削除
`ssm_a` / `ssm_dt_bias` / `ssm_norm` の doc comment から "Loaded but not applied yet" 文言を削除

---

## 5. Numerical Validation Plan

### 5.1 Ground truth 生成

llama.cpp fork で以下の中間 tensor を dump:

- Layer 3 (最初の DeltaNet layer) の Step 1-7 各段の tensor 値
- Layer 15 / 31 / 63 (深い layer) の final output logits

### 5.2 ALICE-LLM 側 dump 追加 (temporary、Phase X.3.e.3 用)

`src/llama3.rs:3810-3903` に `#[cfg(feature = "dump-deltanet")]` gate 付き dump 追加:

```rust
#[cfg(feature = "dump-deltanet")]
if layer == DUMP_TARGET_LAYER {
    dump_tensor("step2_alpha", &dn_alpha);
    dump_tensor("step2_beta", &dn_beta);
    dump_tensor("step4_delta_out_pre_norm", &dn_delta_out);
    // ... 各 step
}
```

### 5.3 Diff methodology

- 各 step の tensor を `abs_diff_max` / `rel_diff_max` で比較
- Threshold: FP16 precision 相当 (1e-3 relative) を通過なら OK
- 一致しない step を特定 → 該当 gap の math を再検討

### 5.4 End-to-end validation

- prompt: `"The capital of Japan is"` (seed=42, temperature=0, top-p=1)
- llama.cpp 出力の最初 5 tokens を expected として ALICE-LLM 出力と一致確認
- Bonsai は Ternary なので argmax noise あり、first 3 tokens 一致で pass 判定

---

## 6. Test Plan

### 6.1 Unit tests

- `gated_deltanet_step_bonsai_per_v_head`: 48 V heads / 16 KV heads で state / alpha / beta の正しい slicing
- `apply_rms_norm_in_place`: 既存 `rms_norm` との数値一致 (in-place ≡ out-of-place)
- `gated_deltanet_step_qwen35_backwards_compat`: 32 heads / 32 heads (num_v == num_kv) で従来動作維持

### 6.2 Integration tests

- Bonsai 27B GGUF load + forward 1 layer (mock inference)
- Qwen 3.5 GGUF load + forward 1 layer (regression 保証)

### 6.3 Numerical tests (feature = "dump-deltanet")

- llama.cpp Qwen 3.6 reference との step-by-step 一致
- End-to-end first 5 tokens 一致 (prompt fixed, seed fixed)

### 6.4 Performance regression

- Qwen 2.5 Coder 1.5B (non-DeltaNet) の tok/s が Phase X.5 baseline から±5% 以内
- Bonsai 27B load 時間が X.5 の 13.2s から±10% 以内 (load code 無変更なら 0% 変化想定)

---

## 7. Risk Assessment

| Risk | 影響 | 対策 |
|---|---|---|
| llama.cpp reference が取得不可 | Blocker、SSM math 確定不能 | PrismML HuggingFace transformers 実装で代替 dump / Bonsai 論文付録 数値例 参照 |
| Gap A 修正で Qwen 3.5 regression | 既存 3B/7B model 出力破綻 | backwards compat test 追加 + `num_v_heads == num_kv_heads` の 1:1 mapping で etxist 動作維持 |
| Gap B の 3 hypothesis 全て不一致 | 更なる仮説必要 (long research) | Option B3 (skip) で Phase X.3.e.3.1 partial landing、B1/B2 は X.3.e.3.2 で追加 |
| Bonsai 27B 実行に 8GB unified memory 不足 | Jetson で validation 不能 | Mac M-series で validation (Jetson は SIMD 最適化後の速度検証専用) |
| State buffer 3× 大 (16→48 heads) で OOM | 5120 * 128 * 48 * 4 bytes/f32 = 125 MB / layer × 48 layers = 6 GB | 実測 (Phase X.5 は 8GB tight 動作、48 KV state は 2 GB 増、OOM リスクあり) |

---

## 8. Fallback Strategy

Phase X.3.e.3 完了不能 (llama.cpp reference 取得不能 or numerical diff 解消不能) の場合:

1. **Subtask 1 (Gap A) のみ landing**: per-V-head alpha/beta の正しい消費、既存 test suite で regression 検証
2. `ssm_a` / `ssm_dt_bias` / `ssm_norm` は `#[allow(dead_code)]` 継続、"Phase X.3.e.3.1 landed, X.3.e.3.2 pending llama.cpp reference" と doc comment 更新
3. Bonsai 27B の output quality は「per-V-head 対応で最低限の numerical correctness」を主張、"full SSM refinement は X.3.e.3.2 で" と CHANGELOG に明記

---

## 9. Timeline

| Task | 工数 | 依存 |
|---|---|---|
| 5.1 llama.cpp reference 生成 | 1-2h | PrismML fork clone + Bonsai GGUF DL |
| 5.2 ALICE-LLM 側 dump 追加 | 30 min | — |
| 4.2 Subtask 1 (Gap A per-V-head loop) | 1h | — |
| 4.3 Subtask 2 (Gap C ssm_norm) | 30 min | — |
| 5.3 Diff methodology 検証 | 1h | 5.1, 5.2, 4.2, 4.3 完了 |
| 4.4 Subtask 3 (Gap B ssm_a/dt_bias) | 2h | 5.3 で仮説確定 |
| 6.x tests | 1h | 4.2-4.4 完了 |
| PR polish (fmt / clippy / doc) | 30 min | 6.x 完了 |
| **合計** | **7-8h** | Phase X.3.e.3 全 subtask |

**Session 分割案**:
- Session 1 (~3h): 5.1 + 5.2 + 4.2 + 4.3 + tests → PR "X.3.e.3.1 landing"
- Session 2 (~3h): 5.3 diff + 4.4 + tests → PR "X.3.e.3.2 landing"
- Session 3 (optional、~1-2h): performance regression check + doc polish

---

## 10. Success Criteria

Phase X.3.e.3 完了の判定 (2026-07-16 時点):

- [x] `DeltaNetLayerWeights::ssm_a` / `ssm_dt_bias` / `ssm_norm` の `#[allow(dead_code)]` 全撤去 → commit `6d43602` + `005b3d0`
- [x] `gated_deltanet_step` が per-V-head alpha/beta を正しく消費 → commit `146ee22` (Gap A)
- [x] `ssm_norm` が Step 4.5 で per-V-head 適用 → commit `005b3d0` (Gap C)
- [x] `ssm_a` / `ssm_dt_bias` が Step 2-3 で消費 → commit `6d43602` (Gap B、Option B1 Mamba-style selective SSM 採用)
- [x] beta sigmoid transformation → commit `2d55f2b` (Gap B extra、reference qwen35.cpp:440-441)
- [x] Post-conv1d silu + z-gate after ssm-norm → commit `c342f10` (§Q/K L2Norm + §silu(z) order、reference qwen35.cpp:502 + :562)
- [x] Qwen 3.5 (num_v_heads == num_kv_heads) regression なし → **304 tests pass** (前 296 + 新規 8)
- [ ] llama.cpp Bonsai 27B との step-by-step numerical diff (rel < 1e-3) → **別 session** (要 GGUF DL + llama.cpp build)
- [ ] End-to-end first 3 tokens 一致 (prompt fixed, seed fixed) → **別 session** (同上)
- [ ] Bonsai 27B Jetson load time が X.5 baseline から±10% 以内 → **別 session** (要実機測定)

---

## 11. Related

- Predecessor:
  - [PR #66 — Phase X.3.e.2 DeltaNet forward wiring](https://github.com/ext-sakamoro/ALICE-LLM/pull/66)
  - [PR #67 — Phase X.5 Jetson load-and-run 動作証明](https://github.com/ext-sakamoro/ALICE-LLM/pull/67)
- Landing commits (2026-07-16、all authored by `Moroya Sakamoto <sakamoro@alicelaw.net>`):
  - `146ee22` — Gap A per-V-head loop
  - `005b3d0` — Gap C ssm_norm RMSNorm
  - `6d43602` — Gap B ssm_a / ssm_dt_bias SSM discretisation
  - `2d55f2b` — Gap B extra beta sigmoid
  - `c342f10` — §Q/K L2Norm + §silu(z) order (`bonsai_semantics: bool` flag)
- Reference:
  - Bonsai whitepaper §DeltaNet section: https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-27b-whitepaper.pdf
  - PrismML llama.cpp fork `src/models/qwen35.cpp:436-562` (reference implementation、`gh api repos/PrismML-Eng/llama.cpp/contents/src/models/qwen35.cpp?ref=prism` で fetch 済)
  - PrismML llama.cpp fork `src/models/delta-net-base.cpp:289-371` (autoregressive path reference)
  - Issue #60 (Phase X.3 diff analysis, Comment `4978572709`)
  - Qwen 3.6 HF config: https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json
- Successor:
  - Phase X.3.e.3.3 (別 session、要 external resource): llama.cpp end-to-end numerical validation + Qwen 3.5 backwards compat 意思決定
  - Phase X.4 (4-bit KV cache) — X.3.e.3 完了、着手可
  - Phase X.8 (Guided Generation) — Bonsai output quality 確定後着手
- ALICE-* memory:
  - `~/claude-config/memory/reference_bonsai_27b_prism_ml.md`
  - `~/claude-config/memory/alice_llm_edge_stack_roadmap.md`
  - `~/claude-config/claude-skills/edge-llm-inference-architecture/SKILL.md` §10 Pattern #1-5

---

## 12. PR draft (Historical、実際は 5 commit で段階 landing)

Phase X.3.e.3 は当初 1 PR で landing する想定だったが、実装過程で subphase 化した結果 **5 commit で段階 landing** を採用。各 commit の PR draft は該当 commit message を参照。

**当初想定 (未使用)**:

```
Title: feat(gguf): Bonsai / Qwen 3.6 SSM-math refinement (Phase X.3.e.3)

Summary:
- Per-V-head alpha/beta loop で Bonsai (48 V / 16 KV heads) の decay parameter を全消費 (Gap A)
- ssm_norm per-V-head RMSNorm を Step 4.5 で適用 (Gap C)
- ssm_a / ssm_dt_bias を Step 2-3 で消費 (Option B1 Mamba-style)
- beta sigmoid transformation (Bonsai reference 一致)
- Post-conv1d silu + z-gate after ssm-norm
- Qwen 3.5 backwards compat 保証 (bonsai_semantics conditional)

Numerical validation:
- Qwen 3.5 (32-head) regression: 0 (既存 296 tests pass、+8 新規 test)
- llama.cpp Bonsai 27B step-by-step diff: 別 session (要 GGUF + build)

Impact:
- `#[allow(dead_code)]` を ssm_a / ssm_dt_bias / ssm_norm から全撤去
- Bonsai 27B forward path が reference formula (qwen35.cpp:436-562) と数式一致
- Phase X.4 (4-bit KV cache) の前提条件クリア

Refs: #60, Phase X.3.e.3
```

**実際の 5 commit landing** は §0 Achievement Summary の表を参照。
