# Phase X.3.e.3.3 — End-to-end numerical validation (execution plan)

**Status**: 🟡 **Partially executed** (2026-07-16、Bonsai 1.7B forward validated、DeltaNet SSM 検証は 27B / Qwen3.5-4B DL 要)

## Execution Log (2026-07-16)

Session で実行して判明した事項:

### ✅ Achieved

- **Mac disk cleanup**: 900 MB → **66 GB free** (`cargo clean` on ALICE-CodeTracker + ALICE-Eco-System で 65 GB 回復)
- **PrismML llama.cpp fork build**: `~/llama.cpp-prismml/build/bin/llama-cli` (Metal enabled)
- **Bonsai 1.7B Q2_0 DL**: `~/models/bonsai/Ternary-Bonsai-1.7B-Q2_0.gguf` (450 MB)
- **ALICE-LLM Bonsai 1.7B forward path validated**:
  - Load 成功 (310 tensors, 35 metadata keys, arch=qwen3, config Llama3Config)
  - Forward pass が sensible logits 生成:
    - pos=-1: top-1 "The" (21.93), Tokyo 系 "Tok" が top-2 (14.09)
    - pos=0: top-1 " capital" (26.70)
    - pos=1: top-1 " of" (26.60)
    - pos=2: top-1 " Japan" (31.27), " Tokyo" top-4 (17.22)
  - Speed: 4.0 tok/s (Q2_0 scalar fallback、SIMD kernel 未実装)
- **llama.cpp fork Bonsai 1.7B 動作確認**:
  - Prefill 322 t/s、Generation 167 t/s (Metal accelerated)
  - Interactive chat mode で "The capital of Japan is" → "| The capital of" 応答生成

### 🟡 Discovered Blockers

- **Bonsai 1.7B / 4B / 8B is std GQA (no DeltaNet)** — HF README で「GQA, SwiGLU MLP, RoPE, RMSNorm」明記、DeltaNet SSM 検証には使用不可
- **Bonsai 27B のみ hybrid-attention DeltaNet** (`Qwen3.6-27B backbone, ~75% linear attention`)、DL は 7.17 GB (~4h @ 500 KB/s)
- **PrismML fork の `llama-cli` は interactive-only** — `-no-cnv` flag 非対応、`echo prompt\n/exit` で pipe すると chat mode 経由で動作するが、raw prompt (no chat template) では不可
- **ALICE-LLM Bonsai 1.7B は `arch=qwen3` として load** (`ssm: None`) — DeltaNet path を通らず、SSM refinement (Phase X.3.e.3 全 6 subphase) は unused

### 🎯 Alternative: Qwen3.5-4B for DeltaNet validation

`unsloth/Qwen3.5-4B-GGUF` に Q3_K_S (2.11 GB) / Q4_K_M (2.74 GB) 等の GGUF 存在、Qwen3.5-4B safetensors は `linear_attn.A_log` / `dt_bias` / `norm.weight` 等 SSM tensors 全て含む (HF API で確認済)、Q4_K quantization は ALICE-LLM 実装済 = **Qwen3.5-4B Q4_K_M で DeltaNet SSM validation 可能** (2.74 GB DL、~40 min)

将来 session の推奨手順:
1. Qwen3.5-4B Q4_K_M DL
2. llama.cpp fork で reference 生成 (interactive で prompt + /exit の流れ、chat template 依存)
3. ALICE-LLM elyza_gguf 例で同 prompt 生成 (chat template 一致必要、要確認)
4. `--logits-dump` mode で top-5 logits を JSONL 出力、Python diff script で cos-sim + top-k rank agreement 計算

---

## Original design (2026-07-15、Bonsai 27B 想定)
**Predecessor**: Phase X.3.e.3 (5 commit landing、`146ee22` + `005b3d0` + `6d43602` + `2d55f2b` + `c342f10`)
**Objective**: PrismML llama.cpp fork Bonsai 27B の tensor dump と ALICE-LLM `gated_deltanet_step` の出力を step-by-step で数値比較し、Phase X.3.e.3 の 6 subphase (Gap A + C + B + B extra + §Q/K + §silu(z)) が reference formula (`qwen35.cpp:436-562`) と bit-exact 圏内 (`rel_diff < 1e-3`) で一致することを実証する

Phase X.3.e.3 の code-level reference alignment は完了済 (design doc §0 Achievement Summary 参照)、本 doc は **実 GGUF を用いた end-to-end 数値検証** の実行手順を documented

---

## 1. Prerequisites

### 1.1 Disk (Mac 側)

Bonsai 27B Ternary GGUF は 5.9 GB、build directory + 一時 tensor dumps で **最低 10-12 GB free** が必要

```bash
# 現状確認
df -h ~

# 実行前 checklist
# - Mac disk usage < 85%
# - ~/Downloads / ~/CTW cleanup done (memory 済 dir を rm -rf)
# - `~/bin/disk-check` で cross-machine 集計確認
```

不足なら大 file cleanup を先行:

```bash
# 候補
du -sh ~/CTW/* ~/Downloads/* 2>/dev/null | sort -h | tail -20
# CTW / hololive で memory 化済 repo は rm -rf 可
```

### 1.2 Build environment

- CMake 3.20+ (`brew install cmake`)
- Xcode Command Line Tools
- (optional) Homebrew `openblas` / `libomp` for Mac Metal acceleration

---

## 2. Reference (llama.cpp side) 準備

### 2.1 PrismML fork clone + build

```bash
cd ~
git clone --depth 1 --branch prism https://github.com/PrismML-Eng/llama.cpp.git llama.cpp-prismml
cd llama.cpp-prismml
mkdir build && cd build
cmake .. -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j
# → ./bin/llama-cli が生成される (~/llama.cpp-prismml/build/bin/llama-cli)
```

**予想時間**: cmake config 30s + build 5-10 min (M-series Mac)

### 2.2 Bonsai 27B Ternary GGUF DL

```bash
# HuggingFace CLI (要 pip install huggingface_hub)
huggingface-cli download prism-ml/bonsai-27b-ternary-g128 \
    bonsai-27b-ternary-g128.gguf \
    --local-dir ~/models/bonsai
# または direct DL
mkdir -p ~/models/bonsai
curl -L -o ~/models/bonsai/bonsai-27b-ternary-g128.gguf \
    https://huggingface.co/prism-ml/bonsai-27b-ternary-g128/resolve/main/bonsai-27b-ternary-g128.gguf
```

**予想時間**: 5.9 GB @ ~50 MB/s = 2 min (fast connection)、~30 GB free 消費 (DL + 一時 buffer)

### 2.3 Reference tensor dump 生成

PrismML fork の `llama-cli` を patched build して中間 tensor を dump する必要あり (fork main branch には dump flag なし)。以下 3 option:

**Option A (推奨)**: llama.cpp の既存 debug callback を activate

```bash
# ggml debug 環境変数を活用
export GGML_LOG_LEVEL=DEBUG
export LLAMA_DUMP_TENSORS=blk.3.ssm_a,blk.3.ssm_dt.bias,blk.3.ssm_norm.weight
~/llama.cpp-prismml/build/bin/llama-cli \
    -m ~/models/bonsai/bonsai-27b-ternary-g128.gguf \
    -p "The capital of Japan is" \
    -n 5 \
    --seed 42 \
    --temp 0 \
    --n-predict 5 \
    2>&1 | tee ~/models/bonsai/reference_dump.log
```

**Option B**: fork に `--dump-tensors <name>` flag を patch (要 C++ 実装、~1h)

**Option C**: HuggingFace transformers 実装で `output_hidden_states=True` + hook で dump (PyTorch、実装最小だが quantization 差異あり)

**推奨**: **Option A** で std log から抽出、足りなければ **Option C** で参照値 supplement

---

## 3. ALICE-LLM 側 dump infrastructure (未実装、要 patch)

### 3.1 Feature flag 設計

`Cargo.toml` に追加:

```toml
[features]
dump-deltanet = []
```

### 3.2 Dump code (llama3.rs forward path に patch)

```rust
// forward path 内、Step 4 gated_deltanet_step の前後で挿入
#[cfg(feature = "dump-deltanet")]
{
    use std::io::Write;
    let dump_layer: usize = std::env::var("DUMP_LAYER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    if dn_idx == dump_layer {
        let dump_path = std::env::var("DUMP_PATH")
            .unwrap_or_else(|_| "/tmp/alice_dump".to_string());
        std::fs::create_dir_all(&dump_path).ok();
        let mut f = std::fs::File::create(
            format!("{}/layer{}_step2_alpha.bin", dump_path, dump_layer),
        )
        .unwrap();
        for &v in dn_alpha.iter() {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        // 同様に beta, ssm_a, ssm_dt_bias, dn_conv_out, dn_delta_out, o_buf を dump
    }
}
```

### 3.3 Dump 対象 tensors (Step 別)

| Step | Tensor | Shape | 対応 llama.cpp callback name |
|---|---|---|---|
| 2a | `dn_alpha` (raw) | `[num_v_heads]` | `alpha` |
| 2b | `dn_beta` (raw) | `[num_v_heads]` | `beta` |
| 2c | `dn_alpha` (after softplus + ssm_a exp) | `[num_v_heads]` | `gate_exp` |
| 2d | `dn_beta` (after sigmoid) | `[num_v_heads]` | `beta_sigmoid` |
| 3 | `dn_conv_out` (post-conv1d) | `[qkv_len]` | `conv_output_raw` |
| 3.5 | `dn_conv_out` (post-silu) | `[qkv_len]` | `conv_output_silu` |
| 4 | `dn_delta_out` (post-recurrence) | `[num_v_heads * v_dim]` | `attn_out` |
| 4.5 | `dn_delta_out` (post-ssm_norm) | `[num_v_heads * v_dim]` | `attn_out_norm` |
| 4.6 | `dn_delta_out` (post-z-gate) | `[num_v_heads * v_dim]` | `gated_output` |
| 5 | `o_buf` (post-ssm_out matvec) | `[hidden_dim]` | `linear_attn_out` |

### 3.4 Run

```bash
cd ~/ALICE-LLM
cargo build --release --features dump-deltanet
DUMP_LAYER=3 DUMP_PATH=/tmp/alice_dump \
    ./target/release/alice-llm-inference \
    --model ~/models/bonsai/bonsai-27b-ternary-g128.gguf \
    --prompt "The capital of Japan is" \
    --n-predict 5 \
    --seed 42 \
    --temperature 0
```

---

## 4. Comparison script

`~/ALICE-LLM/scripts/compare_deltanet_dumps.py`:

```python
#!/usr/bin/env python3
"""Phase X.3.e.3.3: llama.cpp reference tensor dumps vs ALICE-LLM dumps.

Reads binary f32 dumps (per-step) and reports per-step abs/rel diff stats.
Usage:
    ./compare_deltanet_dumps.py <alice_dump_dir> <llama_dump_dir> [--threshold 1e-3]

Exit 0 if all steps within threshold, 1 otherwise.
"""
import struct
import sys
from pathlib import Path

def read_f32(path):
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))

def diff_stats(a, b):
    assert len(a) == len(b), f"length mismatch: {len(a)} vs {len(b)}"
    n = len(a)
    abs_diffs = [abs(x - y) for x, y in zip(a, b)]
    abs_max = max(abs_diffs)
    abs_mean = sum(abs_diffs) / n
    max_ref = max(abs(y) for y in b)
    rel_max = abs_max / max_ref if max_ref > 1e-12 else 0.0
    return abs_max, abs_mean, rel_max

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    alice_dir = Path(sys.argv[1])
    llama_dir = Path(sys.argv[2])
    threshold = 1e-3
    if "--threshold" in sys.argv:
        threshold = float(sys.argv[sys.argv.index("--threshold") + 1])

    all_ok = True
    for name in sorted(alice_dir.glob("*.bin")):
        stem = name.stem
        llama_path = llama_dir / f"{stem}.bin"
        if not llama_path.exists():
            print(f"[skip] {stem}: no reference dump")
            continue
        alice = read_f32(name)
        llama = read_f32(llama_path)
        abs_max, abs_mean, rel_max = diff_stats(alice, llama)
        status = "OK" if rel_max < threshold else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(
            f"[{status}] {stem}: abs_max={abs_max:.6e} abs_mean={abs_mean:.6e} rel_max={rel_max:.6e}"
        )
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
```

**実行**:

```bash
python3 ~/ALICE-LLM/scripts/compare_deltanet_dumps.py \
    /tmp/alice_dump /tmp/llama_dump \
    --threshold 1e-3
```

**Expected output**:

```
[OK] layer3_step2_alpha: abs_max=1.234e-06 abs_mean=5.678e-08 rel_max=1.234e-06
[OK] layer3_step2_beta_sigmoid: abs_max=... rel_max=...
[OK] layer3_step3_conv_output_silu: ...
[OK] layer3_step4_attn_out: ...
[OK] layer3_step4.5_attn_out_norm: ...
[OK] layer3_step4.6_gated_output: ...
[OK] layer3_step5_linear_attn_out: ...
```

---

## 5. Validation methodology

### 5.1 Success criteria (per step)

| Step | Tolerance | Rationale |
|---|---|---|
| `alpha` / `beta` raw | `< 1e-6` | Linear matvec (BF16 precision) |
| `gate_exp` (softplus × ssm_a → exp) | `< 1e-5` | Exp amplification of f32 rounding |
| `beta_sigmoid` | `< 1e-6` | Sigmoid closed form |
| `conv_output_silu` | `< 1e-5` | conv1d + silu chain |
| `attn_out` (recurrence) | `< 1e-3` | State accumulation over long dimension |
| `attn_out_norm` | `< 1e-3` | RMSNorm on accumulated state |
| `gated_output` | `< 1e-3` | Same as attn_out_norm × silu(z) |
| `linear_attn_out` | `< 1e-3` | Final output projection |

### 5.2 End-to-end success

- First 3 tokens 一致 (prompt=`The capital of Japan is`, seed=42, temp=0)
- Both engines must produce the same argmax tokens 3 in a row
- Bonsai is Ternary, so slight rounding may cause 4-th+ token divergence — first 3 是準

### 5.3 Regression check (Qwen 3.5)

Qwen3.5-4B (HuggingFace has SSM tensors confirmed 2026-07-16) を GGUF 化し、同 procedure で llama.cpp Qwen 3.5 と ALICE-LLM Qwen 3.5 の numerical diff を測る:
- 同 threshold 満たすなら Bonsai conditional は正しく Qwen 3.5 でも auto-trigger、意思決定不要
- 満たさないなら Qwen 3.5 GGUF converter が SSM tensors を write していない可能性、別途調査

---

## 6. Failure modes (期待される差異)

### 6.1 GGUF quantization boundary

Ternary weights は `{-1, 0, +1}` × FP16 scale (group-128) で保存、dequantize は decidable だが、順序 (accumulate order) の違いで最終桁が変わる

**対策**: `abs_max < 1e-4` を許容基準として採用

### 6.2 SIMD backend 差 (NEON vs scalar)

NEON kernel の FMA accumulation vs scalar accumulation で微差、大きく蓄積すると `rel_diff` が閾値超え

**対策**: SIMD 無効 build で対照実験 (`cargo build --release --no-default-features --features dump-deltanet`)

### 6.3 llama.cpp `ggml_sum_rows` の order vs ALICE-LLM `for i in 0..n { sum += x[i]; }` の order

llama.cpp は SIMD accumulate、順序が並列 chunk 単位で異なる

**対策**: 大 dim reduction は `rel_diff < 5e-3` に緩和 (`attn_out` / `linear_attn_out`)

---

## 7. Post-validation actions

### 7.1 All steps within threshold

- Update `docs/PHASE_X_3_E_3_DESIGN.md` §10 の残 3 checkbox を chek
- `README.md` に "Bonsai 27B reference-aligned" badge 追加
- Phase X.3.e.3 を fully close、Phase X.4 (4-bit KV cache) 着手可

### 7.2 Some step exceeds threshold

- 該当 step の math を再検討、reference qwen35.cpp / delta-net-base.cpp 該当行 diff
- 差異が small (< 5e-3) なら threshold 緩和 + doc justify
- 差異が large なら code fix + new commit + re-validate

### 7.3 End-to-end tokens 不一致

- Argmax difference は per-step small diff の累積、まず per-step で追跡
- 累積誤差が inevitable (Ternary quantize による) と判断されれば first token 一致で妥協

---

## 8. Estimated timeline

| Task | 工数 | 前提 |
|---|---|---|
| Mac disk cleanup | 30 min - 2h | 何を消せるか user 判断次第 |
| llama.cpp fork clone + build | 15 min | disk cleanup 済 |
| Bonsai GGUF DL | 2-5 min | HuggingFace access |
| llama.cpp reference tensor dump | 30 min | Option A / B / C の選定 + 実行 |
| ALICE-LLM dump feature 実装 | 1h | patch + build + test |
| ALICE-LLM Bonsai tensor dump | 15 min | Feature build 済 |
| Comparison script 実行 + 分析 | 30 min | 両 dump 揃った状態 |
| **合計** | **3-5h** | + disk cleanup 時間 |

**Session 分割案**:
- **Session A (~1h)**: disk cleanup + llama.cpp fork clone + build + Bonsai GGUF DL
- **Session B (~1h)**: llama.cpp reference dump 生成 (Option A / B / C 決定)
- **Session C (~1.5h)**: ALICE-LLM dump feature 実装 + build + Bonsai dump
- **Session D (~30min)**: comparison + 差異分析 + Phase X.3.e.3 fully close

---

## 9. Related

- **Predecessor**: `docs/PHASE_X_3_E_3_DESIGN.md` (Phase X.3.e.3 の 6 subphase 完了 doc、reference qwen35.cpp との数式一致を code-level 実現)
- **Reference**: PrismML llama.cpp fork `src/models/qwen35.cpp:436-562` + `src/models/delta-net-base.cpp:289-371`
- **Bonsai GGUF**: https://huggingface.co/prism-ml/bonsai-27b-ternary-g128
- **Qwen3.5-4B safetensors** (SSM tensor 存在確認 2026-07-16): https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/model.safetensors.index.json
- **ALICE-* memory**:
  - `~/claude-config/memory/alice_llm_edge_stack_roadmap.md`
  - `~/claude-config/memory/reference_bonsai_27b_prism_ml.md`
  - `~/claude-config/claude-skills/edge-llm-inference-architecture/SKILL.md`

---

## 10. Notes for future executor

- **Qwen 3.5 backwards compat 意思決定 (X.3.e.3.4) は既に解決済**: HuggingFace API で Qwen3.5-4B safetensors を確認、`linear_attn.A_log` / `dt_bias` / `norm.weight` / `in_proj_a` / `in_proj_b` / `in_proj_qkv` / `in_proj_z` / `conv1d.weight` 全て存在。ALICE-LLM の loader は unconditional で `tensor_to_f32(...)` で Option<Vec<f32>> を読むので、Qwen 3.5 GGUF (正しく converted) なら SSM tensors も Some で load される。よって `is_bonsai_path = ssm_a.is_some() && ssm_dt_bias.is_some()` は自動的に Qwen 3.5 でも true → SSM refinement 適用 → reference formula と一致 = **意思決定不要、現行 code が正解**

- **命名の混乱**: `bonsai_semantics` / `is_bonsai_path` は misleading (actual meaning は "GGUF に SSM refinement tensor が入っている")。将来 followup commit で `use_ssm_refinement` / `has_ssm_math` へ rename 推奨、ただし機能は unchanged

- **Bonsai と Qwen 3.5 の実質的違い**: (1) linear layer 数 vs full-attention layer 数の比率 (Bonsai 48/16、Qwen 3.5 は arch config 次第) (2) `attn_output_gate: true` (Bonsai/Qwen 3.6) vs false (Qwen 3.5) (3) 量子化 scheme (Bonsai Ternary g128、Qwen 3.5 は Q4_K_M / Q8_0 等が典型) — DeltaNet の SSM math 自体は同一
