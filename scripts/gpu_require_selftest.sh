#!/usr/bin/env bash
# Check that ALICE_LLM_REQUIRE_GPU has teeth: with the adapter hidden
# (ALICE_LLM_TEST_FORCE_NO_ADAPTER=1) a GPU test must fail on the requirement
# panic. Passes only when the run fails AND the requirement message is in the
# output (a run that fails for another reason does not count).
#
# usage: scripts/gpu_require_selftest.sh <cargo test arguments...>
set -uo pipefail
log=$(mktemp)
trap 'rm -f "$log"' EXIT
if ALICE_LLM_REQUIRE_GPU=1 ALICE_LLM_TEST_FORCE_NO_ADAPTER=1 cargo test "$@" >"$log" 2>&1; then
  tail -n 20 "$log"
  echo "error: self-test: the run passed with the adapter hidden" >&2
  exit 1
fi
hits=$(grep -c 'ALICE_LLM_REQUIRE_GPU is set but no GPU adapter is available' "$log" || true)
echo "self-test: tests failed on the missing adapter: $hits"
if [ "$hits" -eq 0 ]; then
  tail -n 40 "$log"
  echo "error: self-test: the run failed for another reason" >&2
  exit 1
fi
