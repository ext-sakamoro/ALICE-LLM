#!/usr/bin/env bash
# DSpark Phase 11: 実測 pipeline
#
# Step 1: PositionConfidenceHead 学習 (dspark_train_confidence_head example)
# Step 2: A/B/C/D 比較実行 (speculative_dspark_dual example、baseline + DSpark 3 strength + confidence-gated 3 threshold)
#
# 環境変数で main / draft / prompt / max_tokens 等を切替
# デフォルトは Llama-3.2-1B draft (flow 検証、~数分完走)
# K3 DSpark は環境変数上書きで指定
#
# Usage:
#   # Flow 検証 (Llama-3.2-1B draft、~数分)
#   DSPARK_MAIN=/path/to/main.gguf \
#   DSPARK_DRAFT=/path/to/llama-3.2-1b.gguf \
#     ~/ALICE-LLM/scripts/phase11_measure.sh
#
#   # K3 本測定 (USB4 化後 ~1-2 hours、K3 DSpark 2.2B GGUF 変換完了後)
#   DSPARK_MAIN=/path/to/kimi-k3-iq1_s.gguf \
#   DSPARK_DRAFT=/path/to/kimi-k3-dspark-2.2b.gguf \
#   DSPARK_MAX_TOKENS=20 \
#     ~/ALICE-LLM/scripts/phase11_measure.sh
#
#   # Phase 12+12b: snapshot mode 切替 (full / compact / delta)
#   DSPARK_MAIN=... DSPARK_DRAFT=... \
#   DSPARK_SNAPSHOT_MODE=compact \
#     ~/ALICE-LLM/scripts/phase11_measure.sh
#   # Delta mode = ~30-48MB overhead (Phase 12b Part 3c2 完成、KimiK3Model draft 用)
#   DSPARK_SNAPSHOT_MODE=delta ...

set -euo pipefail
IFS=$'\n\t'

# ---- 必須引数 ----
MAIN_MODEL="${DSPARK_MAIN:?Usage: DSPARK_MAIN=<main.gguf> DSPARK_DRAFT=<draft.gguf> $0}"
DRAFT_MODEL="${DSPARK_DRAFT:?Usage: DSPARK_MAIN=<main.gguf> DSPARK_DRAFT=<draft.gguf> $0}"

if [[ ! -f "${MAIN_MODEL}" ]]; then
    echo "error: DSPARK_MAIN not found: ${MAIN_MODEL}" >&2
    exit 1
fi
if [[ ! -f "${DRAFT_MODEL}" ]]; then
    echo "error: DSPARK_DRAFT not found: ${DRAFT_MODEL}" >&2
    exit 1
fi

# ---- 任意引数 (デフォルト値) ----
PROMPT="${DSPARK_PROMPT:-The capital of Japan is}"
TRAIN_MAX_TOKENS="${DSPARK_TRAIN_MAX_TOKENS:-50}"
MEASURE_MAX_TOKENS="${DSPARK_MAX_TOKENS:-20}"
SPEC_K="${DSPARK_SPEC_K:-4}"
BIGRAM_RANK="${DSPARK_BIGRAM_RANK:-256}"
EPOCHS="${DSPARK_EPOCHS:-30}"
LR="${DSPARK_LR:-0.05}"
TEMPERATURE="${DSPARK_TEMPERATURE:-0.0}"
# Phase 12+12b: snapshot mode 選択 (full / compact / delta)
# full: default、~290MB、正確、O(1) rollback
# compact: ~145MB、精度 loss ~1e-3 (f16)、O(1) rollback + f16 変換
# delta: ~30-48MB、bit-exact、O(N) rollback (base + N updates replay)
# 注: 現状 Llama3Model draft では no-op、KimiK3Model draft (Phase 13+) で actionable
SNAPSHOT_MODE="${DSPARK_SNAPSHOT_MODE:-full}"
if [[ ! "${SNAPSHOT_MODE}" =~ ^(full|compact|delta)$ ]]; then
    echo "error: DSPARK_SNAPSHOT_MODE must be one of: full, compact, delta (got: ${SNAPSHOT_MODE})" >&2
    exit 1
fi

# ---- 出力先 ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="${LOG_DIR}/phase11_train_${TIMESTAMP}.log"
MEASURE_LOG="${LOG_DIR}/phase11_measure_${TIMESTAMP}.log"
HEAD_OUTPUT="${LOG_DIR}/phase11_trained_head_${TIMESTAMP}.bincode"

# ---- 表示 ----
echo "=== DSpark Phase 11 Measurement Pipeline ==="
echo "  main model:   ${MAIN_MODEL}"
echo "  draft model:  ${DRAFT_MODEL}"
echo "  prompt:       ${PROMPT}"
echo "  train tokens: ${TRAIN_MAX_TOKENS}"
echo "  measure tokens: ${MEASURE_MAX_TOKENS}"
echo "  spec_k:       ${SPEC_K}"
echo "  bigram_rank:  ${BIGRAM_RANK}"
echo "  epochs:       ${EPOCHS}"
echo "  lr:           ${LR}"
echo "  temperature:  ${TEMPERATURE}"
echo "  snapshot mode: ${SNAPSHOT_MODE}"
echo "  train log:    ${TRAIN_LOG}"
echo "  measure log:  ${MEASURE_LOG}"
echo "  head output:  ${HEAD_OUTPUT}"
echo

cd "${REPO_ROOT}"

# ---- Step 1: PositionConfidenceHead 学習 ----
echo "=== Step 1: dspark_train_confidence_head ==="
echo "  (expect: label collection → position 別 accept rate → SGD BCE 学習 → bincode save)"
echo

cargo run --release --example dspark_train_confidence_head \
    --features "dspark,dspark-serde,gguf,parallel" -- \
    --model "${MAIN_MODEL}" \
    --draft-model "${DRAFT_MODEL}" \
    --prompt "${PROMPT}" \
    --max-tokens "${TRAIN_MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --speculative-k "${SPEC_K}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --output "${HEAD_OUTPUT}" 2>&1 | tee "${TRAIN_LOG}"

if [[ ! -f "${HEAD_OUTPUT}" ]]; then
    echo "error: trained head bincode not created at ${HEAD_OUTPUT}" >&2
    exit 1
fi

echo
echo "=== Step 1 完了: trained head saved to ${HEAD_OUTPUT} ($(stat -f%z "${HEAD_OUTPUT}" 2>/dev/null || stat -c%s "${HEAD_OUTPUT}") bytes) ==="
echo

# ---- Step 2: A/B/C/D 比較 ----
echo "=== Step 2: speculative_dspark_dual with --confidence-head ==="
echo "  (expect: baseline + DSpark 3 strength + DSpark bigram+confidence 3 threshold = 7 variant)"
echo

cargo run --release --example speculative_dspark_dual \
    --features "dspark,gguf,dspark-serde,parallel" -- \
    --model "${MAIN_MODEL}" \
    --draft-model "${DRAFT_MODEL}" \
    --prompt "${PROMPT}" \
    --max-tokens "${MEASURE_MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --speculative-k "${SPEC_K}" \
    --bigram-rank "${BIGRAM_RANK}" \
    --confidence-head "${HEAD_OUTPUT}" \
    --snapshot-mode "${SNAPSHOT_MODE}" 2>&1 | tee "${MEASURE_LOG}"

echo
echo "=== Phase 11 完了 ==="
echo "  train log:   ${TRAIN_LOG}"
echo "  measure log: ${MEASURE_LOG}"
echo "  head bincode: ${HEAD_OUTPUT}"
echo
echo "次: 記事 kimi-k3-dspark-sequel.md §7 の table に log から数字を転記"
echo "  tok/s / accept rate / speedup vs baseline を variant 別に埋める"
