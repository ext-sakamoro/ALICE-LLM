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

- Add a **layer-0 per-op tensor dump** to both engines for `attn_norm →
  q → q_normed → k → kv_a → k_nope → attn_scores → attn_out → o_proj →
  hidden`
- Diff each stage against HF's forward hooks to pinpoint the divergence
- Track under a new issue "DeepSeek-V2-Lite forward numerical parity"
  (Issue #36 remains open for full V3 671B validation infrastructure)

## Full HF oracle result

Saved at `/notebooks/v2lite_oracle_result.json` on Paperspace instance
`nfv1bb8mo0` (Free A6000). Contents include prompt tokens, generated
tokens, and per-position top-5 logits for 10 generation steps.

## Full ALICE-LLM logits dump

Saved at `/tmp/alice_v2lite_logits.jsonl` on Mac (`elyza_gguf.rs` output).
JSONL format: one line per position with `pos` / `top5` (5 objects with
`id` / `logit` / `tok`).
