# data/ — datasets & measurement logs

This directory holds datasets, measurement logs, and other ephemeral artifacts
produced by ALICE-LLM examples (bench, perplexity, dump). The directory itself
is `.gitignore`'d except this README so nothing is committed by default; users
prepare local artifacts on demand.

## ⚠ Known limitation: `examples/perplexity` diverges from llama.cpp (2026-07-24)

Cross-validation on Mac M3 Metal, WikiText-2 test first 500 tokens, ctx=512:

| Implementation | Qwen 3.5-4B Q4_K_M PPL |
|---|---|
| ALICE-LLM `examples/perplexity` | 16.38 |
| llama.cpp `llama-perplexity` (--chunks 1) | 6.09 ± 1.05 |
| **divergence** | **2.68× (ALICE-LLM inflated)** |

Cause is not BOS handling (Qwen 3.5 GGUF defines no `bos_token_id` and
neither tool prepends BOS in this config). The gap is attributable to Qwen
3.5 forward path numerical drift in ALICE-LLM (see Phase X.3.e.3.30-35
diagnostic in `memory/alice_llm_future_work.md`) and possibly to Q4_K
dequant precision differences.

Bonsai 27B Q1_0 cannot be validated against llama.cpp: mainline llama.cpp
rejects Bonsai's custom Q1_0 quant type (ggml type 41).

**Use `examples/perplexity` for internal diagnostics only** until the
ALICE-LLM Qwen forward matches llama.cpp within ~5%.

## WikiText-2 raw (for `examples/perplexity.rs`)

Standard benchmark for LLM perplexity. Downloaded from the Hugging Face
`Salesforce/wikitext` dataset repository (raw variant, test split).

```bash
mkdir -p data/wikitext-2
curl -sL "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet" \
  -o /tmp/wt2-test.parquet

python3 - <<'PY'
import pyarrow.parquet as pq
tbl = pq.read_table('/tmp/wt2-test.parquet')
texts = tbl['text'].to_pylist()
combined = ''.join(texts)  # rows are already prefixed with their own newlines
with open('data/wikitext-2/wiki.test.raw', 'w', encoding='utf-8') as f:
    f.write(combined)
print(f"wrote {len(combined)} chars ({tbl.num_rows} rows, {sum(1 for t in texts if t.strip())} non-empty)")
PY
```

Expected size: ~1.29 MB, ~4358 rows, ~2891 non-empty.

## Perplexity measurement logs (`data/ppl_logs/`)

Optional destination for `perplexity` example runs. Convention:

- `<model>_<dataset>_<n>tok.log` — full stderr + stdout of a single run
- The final line of every log is a one-line JSON with the summary metrics

Example:

```bash
mkdir -p data/ppl_logs
./target/release/examples/perplexity \
  --model models/Qwen3.5-4B-Q4_K_M.gguf \
  --dataset data/wikitext-2/wiki.test.raw \
  --ctx 512 --n-samples 500 --mode chunked --progress-every 100 \
  2>&1 | tee data/ppl_logs/qwen4b_wt2_500tok.log
```

## Smoke test samples (`data/smoke_test/`)

Small ad-hoc text samples for verifying `perplexity` end-to-end without a full
dataset. Not tracked; recreate as needed for local experiments.
