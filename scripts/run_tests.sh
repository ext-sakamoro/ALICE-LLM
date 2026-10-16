#!/usr/bin/env bash
# scripts/run_tests.sh <required-test-binary>[=<min>][,<more>...] -- <cargo test command...>
#
# Runs the given cargo test command and fails unless
#   - the command itself succeeds,
#   - at least one test passed in total, and
#   - every named integration test binary ran and passed at least its minimum
#     number of tests (1 when no minimum is given).
# A feature or target mismatch otherwise turns a test step into a silent
# no-op that still reports success.
#
# Several names are given comma separated. Naming a binary is what keeps a
# suite from disappearing: `cargo test` without `--test` runs every target, so
# a suite that stops being built reports nothing rather than failing.
#
# A minimum (`float_environment=2`) is for a binary whose every test carries
# the claim: with only "at least one passed", an `#[ignore]`, a `#[cfg]` that
# drops a test, or a rename would leave the step green with fewer tests than
# the claim needs. The minimum is a positive integer; `name=0`, `name=`,
# `name=x` and an empty name are refused before the command runs (exit 2).
set -euo pipefail

IFS=',' read -r -a specs <<< "$1"
shift
[[ "${1:-}" == "--" ]] && shift

# ⚠️ Indexed arrays, not an associative one: `declare -A` needs bash 4, and the
# macOS runner's /bin/bash is 3.2 — the CI step failed with
# `declare: -A: invalid option` while a Homebrew bash 5 ran it locally.
required=()
required_min=()
for spec in "${specs[@]}"; do
  name="${spec%%=*}"
  if [[ "$spec" == *=* ]]; then
    min="${spec#*=}"
    if [[ ! "$min" =~ ^[1-9][0-9]*$ ]]; then
      echo "run_tests: '${spec}': the minimum after '=' must be a positive integer" >&2
      exit 2
    fi
  else
    min=1
  fi
  if [[ -z "$name" ]]; then
    echo "run_tests: '${spec}': empty test binary name" >&2
    exit 2
  fi
  required+=("$name")
  required_min+=("$min")
done

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

"$@" 2>&1 | tee "$log"

total=0
required_passed=()
for i in "${!required[@]}"; do required_passed[$i]=0; done
current=""
while IFS= read -r line; do
  case "$line" in
    *"Running "*)
      current="$line"
      ;;
    "test result: "*)
      n=$(printf '%s\n' "$line" | sed -E 's/.* ([0-9]+) passed.*/\1/')
      total=$((total + n))
      for i in "${!required[@]}"; do
        name="${required[$i]}"
        if [[ "$current" == *"tests/${name}.rs"* || "$current" == *"tests\\${name}.rs"* ]]; then
          required_passed[$i]=$((required_passed[$i] + n))
        fi
      done
      ;;
  esac
# strip CR (Windows) and ANSI colour codes (CARGO_TERM_COLOR=always puts one
# right after "Running", which would hide the binary name from the match above)
done < <(tr -d '\r' < "$log" | sed "s/$(printf '\033')\[[0-9;]*m//g")

summary=""
for i in "${!required[@]}"; do
  summary="${summary} ${required_passed[$i]} in ${required[$i]},"
done
echo "run_tests: ${total} tests passed in total,${summary%,}"
if [[ "$total" -eq 0 ]]; then
  echo "run_tests: no test ran" >&2
  exit 1
fi
missing=0
for i in "${!required[@]}"; do
  if [[ "${required_passed[$i]}" -eq 0 ]]; then
    echo "run_tests: ${required[$i]} did not run any test" >&2
    missing=1
  elif [[ "${required_passed[$i]}" -lt "${required_min[$i]}" ]]; then
    echo "run_tests: ${required[$i]} passed ${required_passed[$i]} test(s), needs at least ${required_min[$i]}" >&2
    missing=1
  fi
done
[[ "$missing" -eq 0 ]] || exit 1
