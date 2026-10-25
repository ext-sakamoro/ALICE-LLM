#!/usr/bin/env bash
# scripts/preflight.sh — local reproduction of the CI gates before `git push`.
# Every command is the one CI runs. A step this file does not cover is a step
# that can only fail remotely — when a workflow step is added, add it here in
# the same commit. Run `--quick` before every push.
#
# usage: scripts/preflight.sh [--quick]   (--quick skips the test / bench suites)
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
[[ "${1:-}" == "--quick" ]] && quick=1

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
# `cargo clippy` reuses fresh `cargo check` artifacts and then lints nothing;
# touching the crate roots invalidates only this repo's fingerprints.
relint() { git ls-files | grep -E '(^|/)src/(lib|main)\.rs$' | xargs -r touch; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1 ($2)" >&2; exit 1; }; }
has_toolchain() { rustup toolchain list | grep -q "^$1"; }

# Steps CI runs that this file cannot reproduce locally (they can only fail remotely):
#   - security-audit.yml:audit:Install cargo-audit (needs network / runner-only)
#   - security-audit.yml:deny:Install cargo-deny (needs network / runner-only)
#   - security-audit.yml:coverage (job is continue-on-error: informational in CI)
#   - fuzz.yml:fuzz:Install cargo-fuzz (needs network / runner-only)
#   - fuzz.yml:fuzz:Build fuzz target (needs network / runner-only)
#   - fuzz.yml:fuzz:Set fuzz duration (no cargo / grep)
#   - fuzz.yml:fuzz:Run fuzz target (time-boxed) (continue-on-error)
#   - fuzz.yml:fuzz:Report crash (informational) (no cargo / grep)

need actionlint "brew install actionlint"
need cargo-audit "cargo install cargo-audit --locked"
need cargo-deny "cargo install cargo-deny --locked"
need cargo-machete "cargo install cargo-machete --locked"

step "ci.yml / fmt: Check formatting"
( export CARGO_TERM_COLOR="always"; cargo fmt --all -- --check )

step "ci.yml / actionlint: actionlint"
actionlint .github/workflows/*.yml

step "ci.yml / actionlint: workflow-timeouts (every job has timeout-minutes)"
# text files are LF in the index and the working tree (.github/workflows/eol.yml, shared by the ALICE crates)
python3 scripts/test_check_eol.py
python3 scripts/check_eol.py
python3 scripts/test_workflow_timeouts.py
python3 scripts/workflow_timeouts.py
python3 scripts/test_changed_paths.py
python3 scripts/changed_paths.py sync

step "ci.yml / docs_lint: docs lint (tests + public documents)"
python3 scripts/test_docs_lint.py
python3 scripts/test_ci_test_coverage_check.py
python3 scripts/ci_test_coverage_check.py
python3 scripts/docs_lint.py --check

step "ci.yml / float-environment: run_tests.sh and build configuration guard (tests, then check)"
python3 scripts/test_run_tests.py
python3 scripts/test_entropy_guard.py
python3 scripts/entropy_guard.py
python3 scripts/test_build_config_guard.py
python3 scripts/build_config_guard.py
python3 scripts/test_build_flags_check.py

step "ci.yml / float-environment: Shared files equal ALICE-Law's table"
python3 scripts/test_shared_files.py
scripts/shared_files.sh

step "ci.yml / clippy: Identifier moves (tests, then against the published release)"
python3 scripts/test_id_move_check.py
python3 scripts/id_move_check.py

step "ci.yml / clippy: cargo clippy (default features, all targets)"
relint
( export CARGO_TERM_COLOR="always"; cargo clippy --all-targets -- -D warnings )

step "ci.yml / clippy: cargo clippy (simd, all targets)"
relint
( export CARGO_TERM_COLOR="always"; cargo clippy --all-targets --features simd -- -D warnings )

step "ci.yml / clippy: cargo clippy (full feature set, all targets)"
relint
( export CARGO_TERM_COLOR="always"; cargo clippy --all-targets --features "gguf,parallel,simd,grammar,quant,hf-config,dspark,dspark-serde,imatrix,gpu,server" -- -D warnings )

step "ci.yml / clippy: cargo clippy (full feature set + internal-kernel-oracle, all targets)"
relint
( export CARGO_TERM_COLOR="always"; cargo clippy --all-targets --features "gguf,parallel,simd,grammar,quant,hf-config,dspark,dspark-serde,imatrix,gpu,server,internal-kernel-oracle" -- -D warnings )

step "ci.yml / g1: Clippy canary data (generator tests, generated from clippy.toml, no drift)"
python3 scripts/test_gen_clippy_canary_calls.py
python3 scripts/gen_clippy_canary_calls.py --check

step "ci.yml / g1: Clippy canary (tests, then every planted banned call is flagged)"
python3 scripts/test_clippy_canary.py
python3 scripts/clippy_canary.py

step "ci.yml / g1: G1 call pin (tests, then every banned call is listed)"
rustup target list --installed | grep -q aarch64-unknown-linux-gnu || rustup target add aarch64-unknown-linux-gnu
rustup target list --installed | grep -q x86_64-unknown-linux-gnu || rustup target add x86_64-unknown-linux-gnu
python3 scripts/test_g1_debt_pin.py
python3 scripts/g1_debt_pin.py

# Hand-added (kept across regeneration): the CI clippy runner is x86_64, this
# Mac is aarch64 — arch-gated bodies (AVX2 / AVX-512 kernels in gguf.rs) only
# lint against the x86_64 target (2026-09-16: 5 findings reached CI this way)
step "clippy: x86_64 target (Linux runner arch, full feature set)"
rustup target list --installed | grep -q x86_64-unknown-linux-gnu || rustup target add x86_64-unknown-linux-gnu
relint
( export CARGO_TERM_COLOR="always"; cargo clippy --all-targets --target x86_64-unknown-linux-gnu --features "gguf,parallel,simd,grammar,quant,hf-config,dspark,dspark-serde,imatrix,gpu,server" -- -D warnings )

step "ci.yml / doc: cargo doc"
( export CARGO_TERM_COLOR="always" RUSTDOCFLAGS="-D warnings"; cargo doc --no-deps --features "gguf,parallel,simd,grammar,quant,hf-config,dspark,dspark-serde,imatrix" )

step "ci.yml / build_release: cargo build --release"
( export CARGO_TERM_COLOR="always"; cargo build --release --lib )

step "ci.yml / examples_build: cargo build --example lol_gen --features "grammar gguf""
( export CARGO_TERM_COLOR="always"; cargo build --example lol_gen --features "grammar gguf" )

step "ci.yml / examples_build: cargo build --bin alice-llm-server --features server"
( export CARGO_TERM_COLOR="always"; cargo build --bin alice-llm-server --features server )

step "ci.yml / examples_build: server lifecycle tests"
( export CARGO_TERM_COLOR="always"; scripts/run_tests.sh server_lifecycle=3 -- cargo test --features server --test server_lifecycle )
( export CARGO_TERM_COLOR="always"; scripts/run_tests.sh lib=6 -- cargo test --lib server_cli )

step "security-audit.yml / deny: Run cargo deny check all"
( export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"; cargo deny --all-features check all )

step "security-audit.yml / unused-deps: cargo machete"
cargo machete

step "security-audit.yml / stub-guard: Block panic!(STUB) in src/**"
(
  export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -eo pipefail
  hits=$(grep -rnE 'panic!\([^)]*STUB' \
    src/ --include="*.rs" \
    --exclude-dir=bin \
    | grep -v ':[[:space:]]*//' \
    | grep -vE ':[[:space:]]*/\*' \
    || true)
  if [ -n "$hits" ]; then
    echo "Explicit STUB panic detected in src/ (production path):"
    echo "$hits"
    exit 1
  fi
  echo "OK: No panic!(STUB) in src/"
)

step "security-audit.yml / stub-guard: Detect todo! / unimplemented! (informational, not blocking)"
(
  export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -eo pipefail
  hits=$(grep -rnE 'todo!\(|unimplemented!\(' \
    src/ --include="*.rs" \
    --exclude-dir=bin \
    | grep -v ':[[:space:]]*//' \
    | grep -vE ':[[:space:]]*/\*' \
    || true)
  if [ -n "$hits" ]; then
    count=$(echo "$hits" | wc -l | tr -d ' ')
    echo "::warning::${count} todo!()/unimplemented!() marker(s) in src/ (informational, intentional fail-fast idiom):"
    echo "$hits" | head -20
  else
    echo "OK: No todo!/unimplemented! markers in src/"
  fi
)

step "security-audit.yml / stub-guard: Detect dbg!() residual in src/**"
(
  export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -eo pipefail
  hits=$(grep -rn 'dbg!(' src/ --include="*.rs" || true)
  if [ -n "$hits" ]; then
    echo "dbg!() macro left in src/:"
    echo "$hits"
    exit 1
  fi
  echo "OK: No dbg!() in src/"
)

step "security-audit.yml / stub-guard: Detect TODO / FIXME / XXX / HACK (informational)"
(
  export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -eo pipefail
  hits=$(grep -rnE 'TODO|FIXME|XXX|HACK' src/ --include="*.rs" || true)
  if [ -n "$hits" ]; then
    echo "::warning::TODO/FIXME/XXX/HACK found in src/ (informational, not blocking):"
    echo "$hits" | head -50
  else
    echo "OK: No TODO/FIXME/XXX/HACK in src/"
  fi
)

step "fuzz.yml / build every fuzz target (nightly; the replay needs the runner)"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (cd fuzz && cargo +nightly fuzz build)
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "public-text.yml / public-text: Public text (ALICE-Law checker at the pin; tree, then the commits)"
python3 scripts/test_public_text.py
scripts/public_text.sh

if [[ $quick -eq 1 ]]; then
  echo; echo "preflight --quick OK (test / bench suites skipped)"; exit 0
fi

step "ci.yml / float-environment: Float environment (debug and release)"
scripts/run_tests.sh float_environment=2 -- cargo test --test float_environment
scripts/run_tests.sh float_environment=2 -- cargo test --release --test float_environment

step "ci.yml / float-environment: Effective rustc flags (default, docs.rs feature set, no default features)"
python3 scripts/build_flags_check.py -- cargo check --release --lib
python3 scripts/build_flags_check.py -- cargo check --release --lib --features "gguf,parallel,simd,grammar,quant,hf-config,dspark,dspark-serde,imatrix"
python3 scripts/build_flags_check.py -- cargo check --release --lib --no-default-features

step "ci.yml / float-environment: No negated fma result (syntax tree, tests then check)"
cargo fmt --manifest-path tools/no-outer-fneg-of-fma/Cargo.toml -- --check
cargo test --quiet --locked --manifest-path tools/no-outer-fneg-of-fma/Cargo.toml
cargo run --quiet --locked --release --manifest-path tools/no-outer-fneg-of-fma/Cargo.toml -- --root .

step "g3-ir.yml / ir: probe tests + no negated fma result in alice-llm's release IR (host build; the per-target floors are checked in CI)"
python3 scripts/test_fneg_fma_ir_check.py
python3 scripts/fneg_fma_ir_check.py --package alice-llm --features "gguf,parallel,simd,grammar,quant,hf-config,dspark,dspark-serde,imatrix" --findings-only

step "ci.yml / test: cargo test (default features)"
( export CARGO_TERM_COLOR="always"; cargo test --lib )

step "ci.yml / test: cargo test (--features simd)"
( export CARGO_TERM_COLOR="always"; cargo test --lib --features simd )

step "ci.yml / test: Analytic oracle tests (closed-form laws)"
( export CARGO_TERM_COLOR="always"; cargo test --test analytic_oracle )

step "ci.yml / test: Law tests (lawmath ln_1p, max / min ties, semantics id, order independence)"
scripts/run_tests.sh lawmath_ln_1p=5,law_max_min_ties=5,semantics_golden=8,llm_semantics_id=9,float_edge_golden=6,running_semantics_id=1 -- cargo test --test lawmath_ln_1p --test law_max_min_ties --test semantics_golden --test llm_semantics_id --test float_edge_golden --test running_semantics_id
scripts/run_tests.sh order_independence=3 -- cargo test --features gguf,parallel,dspark --test order_independence
scripts/run_tests.sh gguf_untrusted_counts=4 -- cargo test --test gguf_untrusted_counts
scripts/run_tests.sh lib=12 -- cargo test --lib llama3::tests::kv_cache

step "ci.yml / release: release profile tests (library, oracles, Law tests)"
cargo test --release --lib
cargo test --release --test analytic_oracle
scripts/run_tests.sh lawmath_ln_1p=5,law_max_min_ties=5,semantics_golden=8,llm_semantics_id=9,float_edge_golden=6 -- cargo test --release --test lawmath_ln_1p --test law_max_min_ties --test semantics_golden --test llm_semantics_id --test float_edge_golden
scripts/run_tests.sh order_independence=3 -- cargo test --release --features gguf,parallel,dspark --test order_independence
scripts/run_tests.sh lib=12 -- cargo test --release --lib llama3::tests::kv_cache

step "ci.yml / test: Non-Law build compiles (platform-math)"
cargo check --all-targets --features platform-math

step "ci.yml / test: Running identifier under platform-math"
scripts/run_tests.sh running_semantics_id=1 -- cargo test --features platform-math --test running_semantics_id

step "ci.yml / test: Doc tests"
( export CARGO_TERM_COLOR="always"; cargo test --doc --features "gguf,parallel,simd,grammar,quant,hf-config,dspark,dspark-serde,imatrix" )

step "ci.yml / gpu_pipeline_smoke: cargo test --features gpu (pipeline smoke test)"
( export CARGO_TERM_COLOR="always" ALICE_LLM_REQUIRE_GPU=1; cargo test --lib --features gpu smoke_test_gpu_pipeline_creation )

step "ci.yml / gpu_pipeline_smoke: self-test (a hidden adapter fails the GPU tests)"
bash scripts/gpu_require_selftest.sh --lib --features gpu smoke_test_gpu_pipeline_creation

step "security-audit.yml / audit: Run cargo audit"
( export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"; cargo audit --deny yanked )

step "semver-checks.yml / semver-checks: semver gate (tests)"
python3 scripts/test_semver_gate.py
python3 scripts/test_release_baseline.py

step "semver-checks.yml / semver-checks: semver gate (latest crates.io release, every consumer feature set)"
need cargo-semver-checks "cargo install cargo-semver-checks --version 0.50.0 --locked"
python3 scripts/semver_gate.py --config .github/semver-gate.toml --target-dir "${CARGO_TARGET_DIR:-target}/semver"

echo; echo "preflight OK"
