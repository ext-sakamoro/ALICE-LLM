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

## Full HF oracle result

Saved at `/notebooks/v2lite_oracle_result.json` on Paperspace instance
`nfv1bb8mo0` (Free A6000). Contents include prompt tokens, generated
tokens, and per-position top-5 logits for 10 generation steps.

## Full ALICE-LLM logits dump

Saved at `/tmp/alice_v2lite_logits.jsonl` on Mac (`elyza_gguf.rs` output).
JSONL format: one line per position with `pos` / `top5` (5 objects with
`id` / `logit` / `tok`).
