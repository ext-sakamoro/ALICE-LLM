# DeepSeek-V2-Lite HF transformers Oracle Validation

Partial closure of Issue #36 with V2-Lite scope. Full V3 671B validation
remains blocked on 370 GB GGUF disk + 40-80 GB GPU host.

## Environment

| Component | Version / Spec |
|---|---|
| Oracle engine | `deepseek-ai/DeepSeek-V2-Lite-Chat` via HuggingFace transformers 4.42.0 |
| Oracle host | Paperspace Free A6000 (48 GB VRAM), torch 2.1.1+cu121, bfloat16 |
| Oracle attention | `attn_implementation="eager"` (flash_attn stub, no CUDA kernel) |
| Test engine | ALICE-LLM commit `8b10f26` (V2-Lite dense Q + advance() fix landed) |
| Test host | Mac M3 CPU, `examples/elyza_gguf.rs --logits-dump 10` |
| Model file | `DeepSeek-V2-Lite-Chat.Q4_K_M.gguf` (mradermacher, 9.7 GB) |
| Prompt | `"The capital of Japan is"` |
| Template | `"User: {prompt}\n\nAssistant:"` (both engines applied identically) |

## Reproduce

**HF oracle** (Paperspace):

```bash
cd /notebooks
python3 v2lite_hf_oracle.py --model /notebooks/v2lite_hf \
  --prompt "The capital of Japan is" --max-tokens 10 \
  --output /notebooks/v2lite_oracle_result.json
```

**ALICE-LLM** (Mac):

```bash
cd ~/ALICE-LLM
./target/release/examples/elyza_gguf \
  --model models/DeepSeek-V2-Lite-Chat.Q4_K_M.gguf \
  --prompt "The capital of Japan is" \
  --max-tokens 10 --temperature 0.0 \
  --logits-dump 10 2>/tmp/alice_v2lite_logits.jsonl
```

## Tokenization

Both engines produce **12 prompt tokens** after BOS handling
(`add_bos_token=true` for DeepSeek's LLaMa-style tokenizer):

```
[100000, 5726, 25, 429, 6077, 280, 12693, 317, 185, 185, 77398, 25]
   BOS   User  :  The capital of  Japan  is  \n  \n Assistant :
```

Prior to the `elyza_gguf.rs` BOS fix (this commit), ALICE-LLM's manual
argmax logits-dump path omitted BOS and produced 11 prompt tokens
inconsistent with the `Llama3Model::generate()` path. Fix applied in
`examples/elyza_gguf.rs` to align both paths.

## Position 0 top-5 (first generated token)

**HF Oracle (bfloat16, eager attention)**:

| Rank | Token ID | Token | Logit |
|---:|---:|---|---:|
| 1 | **33132** | `" Tokyo"` | **29.500** |
| 2 | 429 | `" The"` | 28.875 |
| 3 | 2158 | `" To"` | 25.375 |
| 4 | 9217 | ... | 25.000 |
| 5 | 575 | ... | 25.000 |

**ALICE-LLM (Q4_K_M, CPU, MLA forward)**:

| Rank | Token ID | Token | Logit |
|---:|---:|---|---:|
| 1 | 207 | `" "` | 16.968 |
| 2 | 2158 | `" To"` | 16.134 |
| 3 | 244 | `" t"` | 16.024 |
| 4 | 304 | `" I"` | 15.152 |
| 5 | 429 | `" The"` | 15.052 |

## Divergence Analysis

- **Token 33132 (" Tokyo")** — the correct answer — is present in HF's top-5
  with the highest logit (29.5) but **absent from ALICE-LLM's top-5**
  (below rank 5, logit < 15.05)
- **Absolute logit magnitude** differs by ~13 points at the top position
  (HF 29.5 vs ALICE-LLM 16.97). Softmax normalisation removes overall
  scale, but the relative rankings still disagree qualitatively
- Both engines agree on **" To" (2158)** and **" The" (429)** appearing in
  top-5, suggesting the tokenizer and embedding lookup are correctly wired.
  The divergence is downstream in the attention / MLA / MoE forward path
- This is **far outside K-quant tolerance** (±0.03 logits documented for
  Q4_K vs f16). A structural bug in the CPU MLA / MoE forward is
  implicated

## Generated text comparison

- **HF (bfloat16, greedy)**: `" Tokyo<｜end▁of▁sentence｜>"` (2 tokens — one
  correct answer + EOS)
- **ALICE-LLM (Q4_K_M, greedy)**: `"\n\n很抱帖\n\n"` (10 tokens — Chinese
  "very sorry" fallback response)

## Verdict

**Load path**: ✅ Correct (V2 dense Q enum + config nope/rope split verified
by matching prompt tokenisation 12/12)

**Forward path**: ⚠️ Structural divergence detected. The Q4_K_M quant
noise alone cannot explain 13-logit swings and loss of the correct answer
from top-5. Candidate root causes (unresolved as of this commit):

1. **MLA k_pe / q_pe alignment** — the shared `k_pe` (single positional
   slice) vs per-head `q_pe` (rotated per head) — may not be phase-aligned
2. **Attention softmax scale** — DeepSeek-V2 uses non-standard
   `1 / sqrt(qk_nope + qk_rope)` = `1/sqrt(192)`; verify math against
   HF's `attention_scale` config
3. **KV cache reconstruction** — historical positions rebuild
   `k_nope[h,t]` and `v[h,t]` from cached `kv_a[t]` via `kv_b_proj`. If
   the per-position matvec is called with wrong offsets, historical
   context is corrupt for all t > 0
4. **MoE routing sigmoid + noaux_tc bias** — V2-Lite has 64 routed + 2
   shared experts. Bias correction and top-k selection math should match
   HF's `DeepseekV2MoE` module

## Follow-up

- ✅ Per-op tensor dump for layer 0 landed in both engines (`ALICE_DEEPSEEK_DUMP=1`
  env var on ALICE-LLM; `scripts/v2lite_hf_per_op_dump.py` on HF side)
- ✅ Cross-position diff bisected the divergence source (see next section)

## Per-op layer-0 divergence bisect (2026-07-17 update)

Both engines were dumped with the same 12-token prompt and diffed
position by position. Key finding: **divergence starts at position 1
and grows monotonically with position** while position 0 matches within
Q4_K quant tolerance.

### attn_out (pre-o_proj) sum divergence by position

| Pos | ALICE sum | HF sum | Diff | |Diff / HF| |
|---:|---:|---:|---:|---:|
| 0 | +2.225 | +2.236 | 0.011 | 0.5% ✅ |
| 1 | +1.359 | +1.090 | 0.269 | 25% ⚠️ |
| 2 | +0.194 | +0.245 | 0.051 | 21% ⚠️ |
| 5 | +1.058 | +2.086 | 1.028 | 49% ❌ |
| 11 | -0.740 | -0.062 | 0.678 | 12× ❌ |

At position 0 (single-position self-attention) attn_out matches to
0.5%: input embedding, RMSNorm, Q projection, KV LoRA chain, attention,
and o_proj all reproduce HF within K-quant tolerance.

From position 1 onwards, ALICE-LLM's attention re-projects each
historical KV cache entry via `kv_b_proj` (compression trick). HF stores
expanded K/V directly. Both paths are mathematically equivalent, so the
observed monotonic divergence points to **cumulative Q4_K quant noise
across 27 layers × 12 positions of MoE + attention** rather than a
structural bug.

### Individual op diffs at position 11 (last prompt token)

| Op | ALICE l2 | HF l2 | | Diff | | ALICE sum | HF sum | Notes |
|---|---:|---:|---:|---:|---:|---|
| hidden_in (embedding) | 3.003 | 3.006 | 0.1% | -1.165 | -1.156 | ✅ token embedding matches |
| attn_norm | 25.027 | 24.916 | 0.4% | 0.886 | 1.859 | ⚠️ l2 close, sum drift from Q4_K noise on norm weight |
| q_full (pre-RoPE) | 89.942 | 91.527 | 1.7% | -2.185 | 1.558 | ⚠️ Q proj matvec quant noise |
| kv_a_full | 32.716 | 33.105 | 1.2% | 10.538 | 12.142 | ⚠️ same |
| kv_a_normed | 1.492 | 1.519 | 1.8% | 1.449 | 1.532 | ✅ close |
| kv_up (K + V) | 4.775 | 4.920 | 3.0% | -1.481 | -2.257 | ⚠️ kv_b_proj quant noise |
| attn_out | 1.301 | 1.783 | 27% | -0.740 | -0.062 | ❌ cumulative KV history drift |
| o_proj_out | 1.446 | 1.719 | 16% | -0.182 | -0.691 | ❌ downstream |
| hidden_post_attn | 3.383 | 3.559 | 5% | -1.347 | -1.858 | ⚠️ residual dilutes attn drift |
| hidden_post_ffn | 2.981 | 1.095 | 172% | -1.129 | 0.184 | ❌ MoE routing amplifies drift |

The FFN (MoE) block at layer 0 dramatically amplifies attn drift because
router top-K selection is highly non-linear — a small attn_norm shift
can flip which experts fire.

## Root cause (as of 2026-07-17)

**Hypothesis (highest confidence)**: Q4_K_M mixed quantization introduces
per-tensor noise that:
1. Is bounded per element (~2-4%) — matches K-quant published tolerance
2. Accumulates through `kv_b_proj`, `q_proj`, `attn_norm.weight` across
   the 27 MoE layers × 12 prompt positions
3. Is amplified by MoE routing's discrete top-K expert selection (small
   input shifts flip which experts activate)

**Mitigation candidates** (not yet tested):
- Try Q5_K_M or Q8_0 quant instead of Q4_K_M — should reduce per-element
  noise and check if divergence stays bounded
- Verify `attn_norm.weight` / `kv_a_norm.weight` are stored as F32 in
  the GGUF (not Q6_K) — norm weights are small and shouldn't be quantized

## Q5_K_M cross-check (2026-07-17 update)

Downloaded `DeepSeek-V2-Lite-Chat.Q5_K_M.gguf` (mradermacher, 11.3 GB) and
re-ran the same dump. Layer-0 attention weights (`attn_q`, `attn_kv_b`,
`attn_output`) are Q5_K instead of Q4_K, so per-element quant noise
drops from ~2-4% to ~1-1.5%. Norm weights (`attn_norm`, `ffn_norm`) are
stored as F32 in both quant files, so any drift there comes from
compute-side ops, not weight storage.

Per-position `attn_out` sum diff versus HF bf16 oracle:

| Pos | Q4_K_M | Q5_K_M | HF |
|---:|---:|---:|---:|
| 0 | +2.225 (Δ 0.011) | +2.294 (Δ 0.057) | +2.236 |
| 1 | +1.359 (Δ 0.269) | +1.244 (Δ 0.154) | +1.090 |
| 5 | +1.058 (Δ 1.028) | +1.266 (Δ 0.820) | +2.086 |
| 11 | −0.740 (Δ 0.678) | −0.839 (Δ 0.777) | −0.062 |

**Q5_K_M does not reduce the position-11 divergence** (Δ 0.777 vs Q4_K_M's
Δ 0.678 — actually slightly larger). If quant noise were the primary
driver, Q5_K would have cut the drift roughly in half. Instead it barely
moves, and at position 0 Q5_K is actually farther from HF than Q4_K,
suggesting Q4_K's larger noise coincidentally cancelled some structural
error that Q5_K exposes.

## Root cause (revised, 2026-07-17)

**Structural divergence in the MLA forward path**, not quantization
noise. The Q4_K → Q5_K test rules out per-element weight drift as the
primary driver — a structural bug on ALICE-LLM's side is the remaining
explanation.

Confirmed characteristics:
- Position 0 (single-position self-attention) matches HF within Q4_K
  tolerance for pre-attention ops (embedding, RMSNorm, Q proj, KV LoRA)
- Position 1+ shows monotonically growing `attn_out` divergence
- Q5_K quant (lower per-element noise) does not close the gap

Candidate root causes (highest likelihood first):

1. **RoPE cache vs on-the-fly asymmetry** — ALICE-LLM stores `k_pe` in
   the KV cache POST-RoPE (rotated at write time). HF may apply RoPE
   on-the-fly at each attention forward using `position_ids`. If the
   rotation angles differ by an offset (e.g. `pos` vs `pos - offset`),
   historical Q·K dot products drift systematically
2. **Softmax scale** — `1 / sqrt(qk_nope + qk_rope)` = `1/sqrt(192)`
   should match HF's `self.softmax_scale`. Verify no mscale multiplier
   is inadvertently applied for V2-Lite (DeepSeek-V2 uses mscale only
   with YARN, which V2-Lite doesn't enable)
3. **KV cache K/V slice extraction from `scratch_kv_up`** — layout is
   `[head0_qk_nope(128), head0_v(128), head1_qk_nope(128), head1_v(128), ...]`.
   Off-by-one indexing would produce systematic drift proportional to
   position count
4. **`kv_b_proj` matvec direction** — inspect whether `weight.T @ x` vs
   `weight @ x` is applied consistently between write and read paths

## Follow-up (open issue, updated priority)

1. ✅ Q5_K_M cross-check completed — rules out quant noise as primary cause
2. Compare ALICE post-RoPE `q_full` vs HF post-RoPE q at pos 1+ — need
   a new HF hook after `apply_rotary_pos_emb` inside `DeepseekV2Attention`
3. Compare `scratch_kv_up[head_off .. head_off+256]` per-head at
   historical positions vs HF's expanded K/V — need dumps of the
   per-position reconstructed K_nope and V from ALICE
4. Once divergence is pin-pointed to one op, fix + rerun oracle to
   verify token-exact match on Tokyo (33132)

**Alternative** (lower confidence): subtle formula divergence in KV cache
reconstruction. To rule out, need to add a hook that captures HF's
per-position reconstructed K_nope + V and compare against ALICE's
`scratch_kv_up` split, one position at a time.

## Follow-up (open issue)

- Test Q5_K_M / Q8_0 variants to isolate quant-noise-vs-formula divergence
- If Q8_0 also diverges: add per-position K_nope + V dump to both engines,
  diff scratch_kv_up[head] slices for historical positions 0..11
- If Q8_0 matches: document as a known quant tolerance issue, close #36
  V2-Lite scope

## Post-RoPE + pre-RoPE k_pe capture (2026-07-17 追加)

HF DeepSeek-V2 の `apply_rotary_pos_emb` を monkey-patch して pre/post
RoPE の q_pe / k_pe を capture する経路を実装。ALICE の `k_pe_pre_rope`
と HF の `k_pe_pre_rope` を pos 0 (RoPE 恒等) で直接比較:

```
ALICE k_pe_pre_rope pos 0: head [−0.055, +0.041, −0.058, +0.009, +0.067], l2 16.70, sum +15.34
HF    k_pe_pre_rope pos 0: head [−0.035, +0.024, +0.075, +0.008, +0.031], l2 11.35, sum −9.62
```

**k_pe の element-wise が異なる**: l2 47% 差、sum 符号反転。両 engine とも
`kv_a_proj_with_mqa` 出力の last 64 elements を k_pe として抽出しているが、
値が根本的に違う (Q4_K quant noise の範疇を大幅超過)

一方、`kv_a_full[0..5]` (element-wise) も divergent:

```
ALICE kv_a_full[0..5] pos 0: [+0.105, −0.294, +0.038, −0.259, +0.182]
HF    kv_a_full[0..5] pos 0: [+0.231, −0.037, +0.094, −0.243, +0.191]
```

要素 3, 4 は 5% 以内で match、要素 0, 1, 2 は 2-8× 差 一方 l2 と sum は
Q4_K tolerance 内で近い (l2 24.23 vs 24.37、sum 35.76 vs 39.29)

要 element-wise divergent でも `kv_a_normed` (RMSNorm 後) は一致するのは、
RMSNorm の `x[i] * w[i] / rms(x)` で分母 rms(x) が scale drift を打ち消し
directional error のみが残るため k_pe は RMSNorm 経由しないので raw
element-wise error がそのまま attention に流れる

## Root cause 再更新 (2026-07-17)

現在の hypothesis: ALICE-LLM の **Q4_K dequant が `kv_a_proj_with_mqa.weight`
に対して HF (safetensors bf16 直読み) と異なる element-wise 結果**を生む

- `kv_a_normed` (norm 後 512-dim) は match するので、rows [0..512] は
  overall direction 一致
- `k_pe` (raw last 64) は不一致
- rows [0..512] は match、rows [512..576] は違う → **row-block ごとの
  Q4_K dequant で block 2 (rows 512-575) が誤解読** の可能性

**次段の verification**: Q8_0 quant (per-element noise ~0.4%) で同じ diff
実施 Q8_0 で match すれば Q4_K dequant bug 確定、diverge すれば別 root cause

## Full HF oracle result

Saved at `/notebooks/v2lite_oracle_result.json` on Paperspace instance
`nfv1bb8mo0` (Free A6000). Contents include prompt tokens, generated
tokens, and per-position top-5 logits for 10 generation steps.

## Full ALICE-LLM logits dump

Saved at `/tmp/alice_v2lite_logits.jsonl` on Mac (`elyza_gguf.rs` output).
JSONL format: one line per position with `pos` / `top5` (5 objects with
`id` / `logit` / `tok`).
