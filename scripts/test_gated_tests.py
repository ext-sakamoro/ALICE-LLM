#!/usr/bin/env python3
"""Tests of scripts/gated_tests.py on hand-written repositories (Cargo.toml features, test files,
workflows, preflight and the per-repository scripts/gated_tests.toml)."""

import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).with_name("gated_tests.py")

CARGO = """[package]
name = "x"
version = "0.1.0"

[features]
default = ["std"]
std = []
gpu = []
physics = ["std"]
full = ["gpu", "physics"]
"""
SELF_DEV = '\n[dev-dependencies]\nx = { path = ".", default-features = false, features = ["physics"] }\n'
OTHER_DEV = '\n[dev-dependencies]\ny = { path = "../y", features = ["physics"] }\n'
REQUIRED = '\n[[test]]\nname = "r"\nrequired-features = ["gpu"]\n'
GPU = '#[cfg(feature = "gpu")]\nmod gpu {\n    #[test]\n    #[ignore = "needs a working GPU adapter"]\n    fn a() {}\n    #[test]\n    #[ignore = "needs a working GPU adapter"]\n    fn b() {}\n}\n'
PHYS = '#![cfg(feature = "physics")]\n#[test]\nfn a() {}\n'
BOTH = '#![cfg(all(feature = "gpu", feature = "physics"))]\n#[test]\nfn a() {}\n'
STD = '#![cfg(feature = "std")]\n#[test]\nfn a() {}\n'
ANY = '#![cfg(any(feature = "gpu", feature = "physics"))]\n#[test]\nfn a() {}\n'
MIXED = '#[cfg(feature = "gpu")]\n#[test]\nfn a() {}\n#[cfg(feature = "physics")]\n#[test]\nfn b() {}\n'
FILE_AND_ITEM = '#![cfg(feature = "gpu")]\n#[test]\nfn a() {}\n#[cfg(feature = "physics")]\n#[test]\nfn b() {}\n'
PLAIN = "#[test]\nfn a() {}\n"

GPU_COUNTED = "cargo test --features gpu --test g -- --include-ignored 2>&1 | python3 scripts/min_passed.py --no-ignored g=2"
PHYS_COUNTED = "cargo test --features physics 2>&1 | python3 scripts/min_passed.py lib=1 p=1"
COUNTED = GPU_COUNTED + "\n" + PHYS_COUNTED + "\n"
RUN = "cargo test --features gpu --test g -- --include-ignored\ncargo test --features physics\n"

CONF_COUNTED = 'mode = "counted"\n[[cover]]\nname = "ci"\nglobs = [".github/workflows/*.yml"]\n[[cover]]\nname = "preflight"\nglobs = ["scripts/preflight.sh"]\n'
CONF_COUNTED_CI = 'mode = "counted"\n[[cover]]\nname = "ci"\nglobs = [".github/workflows/*.yml"]\n'
CONF_RUN = 'mode = "run"\n[[cover]]\nname = "ci"\nglobs = [".github/workflows/*.yml"]\n'


def run(files, workflows, pre=None, conf=CONF_COUNTED, cargo=CARGO):
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "tests").mkdir()
        (root / ".github" / "workflows").mkdir(parents=True)
        (root / "scripts").mkdir()
        (root / "Cargo.toml").write_text(cargo)
        for name, text in files.items():
            (root / "tests" / f"{name}.rs").write_text(text)
        for name, text in workflows.items():
            (root / ".github" / "workflows" / f"{name}.yml").write_text(text)
        if pre is not None:
            (root / "scripts" / "preflight.sh").write_text(pre)
        if conf is not None:
            (root / "scripts" / "gated_tests.toml").write_text(conf)
        r = subprocess.run([sys.executable, str(SCRIPT), "--root", d], capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr


FILES = {"g": GPU, "p": PHYS, "plain": PLAIN}
CI = {"ci": COUNTED}

CASES = [
    # (description, files, workflows, preflight, config, expected: 0 (passes) or the problem text
    # it must report (fails with exit 1), [Cargo.toml])
    # counted mode (v1's cases)
    ("every gated file run and counted in both", FILES, CI, COUNTED, CONF_COUNTED, 0),
    ("a gated file missing from the workflows", FILES, {"ci": PHYS_COUNTED}, COUNTED, CONF_COUNTED, "ci: tests/g.rs (features ['gpu'], 2 tests): no min_passed command names it"),
    ("a gated file missing from preflight", FILES, CI, GPU_COUNTED, CONF_COUNTED, "preflight: tests/p.rs (features ['physics'], 1 tests): no min_passed command names it"),
    ("a floor below the test count", FILES, {"ci": COUNTED.replace("g=2", "g=1")}, COUNTED, CONF_COUNTED, "ci: tests/g.rs (features ['gpu'], 2 tests): floor 1, the file has 2 tests"),
    ("a floor above the test count", FILES, {"ci": COUNTED.replace("g=2", "g=3")}, COUNTED, CONF_COUNTED, "ci: tests/g.rs (features ['gpu'], 2 tests): floor 3, the file has 2 tests"),
    ("the feature not enabled", FILES, {"ci": COUNTED.replace("--features gpu", "--features physics")}, COUNTED, CONF_COUNTED, "ci: tests/g.rs (features ['gpu'], 2 tests): features ['gpu'] not enabled"),
    ("named but filtered out by --test", FILES, {"ci": COUNTED.replace("--test g", "--test other")}, COUNTED, CONF_COUNTED, "ci: tests/g.rs (features ['gpu'], 2 tests): not among the --test filters"),
    ("ignored tests without --include-ignored", FILES, {"ci": COUNTED.replace(" -- --include-ignored", "")}, COUNTED, CONF_COUNTED, "ci: tests/g.rs (features ['gpu'], 2 tests): has ignored tests but the command does not pass --include-ignored"),
    ("ignored tests without --no-ignored", FILES, {"ci": COUNTED.replace("--no-ignored ", "")}, COUNTED, CONF_COUNTED, "ci: tests/g.rs (features ['gpu'], 2 tests): has ignored tests but min_passed is not given --no-ignored"),
    ("a command over lines with backslashes", FILES, {"ci": COUNTED.replace("--test g", "\\\n    --test g")}, COUNTED, CONF_COUNTED, 0),
    ("the workflows spread over two files", FILES, {"a": GPU_COUNTED, "b": PHYS_COUNTED}, COUNTED, CONF_COUNTED, 0),
    ("no gated file at all", {"plain": PLAIN}, CI, COUNTED, CONF_COUNTED, 'no readable feature-gated test file found under tests/'),
    ("no cargo test command in a cover", FILES, {"ci": "cargo build\n"}, COUNTED, CONF_COUNTED, 'ci: no `cargo test` command found'),
    ("a cover whose globs match no file", FILES, CI, None, CONF_COUNTED, "preflight: globs ['scripts/preflight.sh'] match no file"),
    # feature resolution
    ("--features full enables gpu and physics through the graph", FILES, {"ci": COUNTED.replace("--features gpu", "--features full")}, COUNTED, CONF_COUNTED, 0),
    ("--all-features enables everything", FILES, {"ci": COUNTED.replace("--features gpu", "--all-features")}, COUNTED, CONF_COUNTED, 0),
    ("-F is --features", FILES, {"ci": COUNTED.replace("--features gpu", "-F gpu")}, COUNTED, CONF_COUNTED, 0),
    ("all(...) needs every feature", {"b": BOTH}, {"ci": "cargo test --features gpu --test b 2>&1 | python3 scripts/min_passed.py b=1\n"}, "cargo test --features full --test b 2>&1 | python3 scripts/min_passed.py b=1\n", CONF_COUNTED, "ci: tests/b.rs (features ['gpu', 'physics'], 1 tests): features ['physics'] not enabled"),
    ("all(...) with every feature", {"b": BOTH}, {"ci": "cargo test --features gpu,physics --test b 2>&1 | python3 scripts/min_passed.py b=1\n"}, "cargo test --features full --test b 2>&1 | python3 scripts/min_passed.py b=1\n", CONF_COUNTED, 0),
    ("a default feature is on in a plain cargo test", {"s": STD}, {"ci": "cargo test\n"}, None, CONF_RUN, 0),
    ("--no-default-features turns the default off", {"s": STD}, {"ci": "cargo test --no-default-features\n"}, None, CONF_RUN, "ci: tests/s.rs (features ['std'], 1 tests): features ['std'] not enabled"),
    # what runs tests and what does not
    ("--lib alone runs no test file", {"p": PHYS}, {"ci": "cargo test --lib --features physics\n"}, None, CONF_RUN, "ci: tests/p.rs (features ['physics'], 1 tests): runs no test file (--lib / --doc / ... only)"),
    ("--lib --tests runs the test files", {"p": PHYS}, {"ci": "cargo test --lib --tests --features physics\n"}, None, CONF_RUN, 0),
    ("--tests with a --test filter runs every test file", {"p": PHYS}, {"ci": "cargo test --features physics --test other --tests\n"}, None, CONF_RUN, 0),
    ("cargo hack check runs no test", {"p": PHYS}, {"ci": "cargo hack check --lib --feature-powerset\n"}, None, CONF_RUN, 'ci: no `cargo test` command found'),
    ("browser tests are counted", {"w": '#![cfg(all(target_arch = "wasm32", feature = "gpu"))]\n#[wasm_bindgen_test]\nfn a() {}\n#[wasm_bindgen_test]\nasync fn b() {}\n'}, {"ci": "cargo test --target wasm32-unknown-unknown --features gpu --test w 2>&1 | python3 scripts/min_passed.py w=2\n"}, "cargo test --target wasm32-unknown-unknown --features gpu --test w 2>&1 | python3 scripts/min_passed.py w=2\n", CONF_COUNTED, 0),
    ("a floor below the browser test count", {"w": '#![cfg(all(target_arch = "wasm32", feature = "gpu"))]\n#[wasm_bindgen_test]\nfn a() {}\n#[wasm_bindgen_test]\nasync fn b() {}\n'}, {"ci": "cargo test --target wasm32-unknown-unknown --features gpu --test w 2>&1 | python3 scripts/min_passed.py w=1\n"}, "cargo test --target wasm32-unknown-unknown --features gpu --test w 2>&1 | python3 scripts/min_passed.py w=2\n", CONF_COUNTED, "ci: tests/w.rs (features ['gpu'], 2 tests): floor 1, the file has 2 tests"),
    # run mode: no floors
    ("run mode: covered without min_passed", FILES, {"ci": RUN}, None, CONF_RUN, 0),
    ("run mode: a gated file never run", FILES, {"ci": "cargo test --features physics\n"}, None, CONF_RUN, "ci: tests/g.rs (features ['gpu'], 2 tests): features ['gpu'] not enabled"),
    # gates it cannot read
    ("an any() gate not listed as unread", {"x": ANY, "p": PHYS}, {"ci": RUN}, None, CONF_RUN, 'tests/x.rs: gate with any()/not() this script cannot read; list it in [unread] with the reason'),
    ("an any() gate listed as unread", {"x": ANY, "p": PHYS}, {"ci": RUN}, None, CONF_RUN + '[unread]\nx = "any(): run by the full job"\n', 0),
    ("a stale unread entry (now readable)", {"p": PHYS, "g": GPU}, {"ci": RUN}, None, CONF_RUN + '[unread]\np = "was any()"\n', '[unread] p: its gate is readable now (stale), remove the entry'),
    ("an unread entry for a file that does not exist", {"p": PHYS, "g": GPU}, {"ci": RUN}, None, CONF_RUN + '[unread]\nnope = "gone"\n', '[unread] nope: no such file tests/nope.rs (stale)'),
    ("a YAML folded block is one command", {"p": PHYS}, {"ci": "      - run: >-\n          cargo test\n          --features physics\n      - run: cargo build\n"}, None, CONF_RUN, 0),
    ("a YAML literal block keeps its lines", {"p": PHYS}, {"ci": "      - run: |\n          cargo test\n          echo --features physics\n"}, None, CONF_RUN, "ci: tests/p.rs (features ['physics'], 1 tests): features ['physics'] not enabled"),
    # shell variables set in the same file
    ("a feature list from an env variable", {"p": PHYS}, {"ci": 'env:\n  FEATS: "gpu,physics"\njobs:\n  t:\n    steps:\n      - run: cargo test --features "$FEATS"\n'}, None, CONF_RUN, 0),
    ("${VAR} braces", {"p": PHYS}, {"ci": 'env:\n  FEATS: physics\nrun: cargo test --features "${FEATS}"\n'}, None, CONF_RUN, 0),
    ("a shell assignment", {"p": PHYS}, {"ci": "FEATS=physics\ncargo test --features $FEATS\n"}, None, CONF_RUN, 0),
    ("an unresolved variable enables nothing", {"p": PHYS}, {"ci": 'run: cargo test --features "$NOPE"\n'}, None, CONF_RUN, "ci: tests/p.rs (features ['physics'], 1 tests): features ['physics'] not enabled"),
    # several assignments on one shell line, with export / local / readonly / declare
    ("export with several assignments (DB's preflight form)", {"p": PHYS}, {"ci": 'export CARGO_TERM_COLOR=always RUSTFLAGS="-Dwarnings" FEATS="gpu,physics"\ncargo test --features "$FEATS"\n'}, None, CONF_RUN, 0),
    ("export with quoted values (Sign's preflight form)", {"p": PHYS}, {"ci": 'export CARGO_TERM_COLOR="always" FEATS="physics"\ncargo test --features "$FEATS"\n'}, None, CONF_RUN, 0),
    # each variable gates its own file, so each prefix is needed
    ("local, readonly and declare -x", {"p": PHYS, "g": GPU, "b": BOTH}, {"ci": "local A=physics\nreadonly B='gpu'\ndeclare -x C=\"gpu,physics\"\ncargo test --features \"$A\" --test p\ncargo test --features $B --test g -- --include-ignored\ncargo test --features \"$C\" --test b\n"}, None, CONF_RUN, 0),
    ("a NAME=value word in a command is not an assignment", {"p": PHYS}, {"ci": 'echo FEATS=physics\ncargo test --features "$FEATS"\n'}, None, CONF_RUN, "ci: tests/p.rs (features ['physics'], 1 tests): features ['physics'] not enabled"),
    ("a later assignment on one line wins over an earlier line", {"p": PHYS}, {"ci": 'FEATS=gpu\nexport X=1 FEATS=physics\ncargo test --features "$FEATS"\n'}, None, CONF_RUN, 0),
    # gates on items: each item needs its own and the file's features
    ("run mode: item gates met by two commands", {"m": MIXED}, {"ci": "cargo test --features gpu\ncargo test --features physics\n"}, None, CONF_RUN, 0),
    ("run mode: an item gate met by none", {"m": MIXED}, {"ci": "cargo test --features gpu\n"}, None, CONF_RUN, "ci: tests/m.rs (features ['physics'], 2 tests): features ['physics'] not enabled"),
    ("run mode: the file gate with every item", {"m": FILE_AND_ITEM}, {"ci": "cargo test --features gpu\n"}, None, CONF_RUN, "ci: tests/m.rs (features ['gpu', 'physics'], 2 tests): features ['physics'] not enabled"),
    ("run mode: the item met without the file gate", {"m": FILE_AND_ITEM}, {"ci": "cargo test --features physics\n"}, None, CONF_RUN, "ci: tests/m.rs (features ['gpu', 'physics'], 2 tests): features ['gpu'] not enabled"),
    ("run mode: the file gate and the item met", {"m": FILE_AND_ITEM}, {"ci": "cargo test --features gpu,physics\n"}, None, CONF_RUN, 0),
    ("counted mode: item gates need one command with all", {"m": MIXED}, {"ci": "cargo test --features gpu 2>&1 | python3 scripts/min_passed.py m=2\ncargo test --features physics 2>&1 | python3 scripts/min_passed.py m=2\n"}, None, CONF_COUNTED_CI, "ci: tests/m.rs (features ['gpu', 'physics'], 2 tests): features ['physics'] not enabled"),
    ("counted mode: item gates in one command", {"m": MIXED}, {"ci": "cargo test --features full 2>&1 | python3 scripts/min_passed.py m=2\n"}, None, CONF_COUNTED_CI, 0),
    # ignored tests that are not meant to run (a known defect, not a missing GPU)
    ("ignored = \"skip\": ignored tests need not run", FILES, {"ci": RUN.replace(" -- --include-ignored", "")}, None, 'ignored = "skip"\n' + CONF_RUN, 0),
    ("an unknown ignored value", FILES, {"ci": RUN}, None, 'ignored = "maybe"\n' + CONF_RUN, 'ignored \'maybe\': must be "run" (ignored tests must run) or "skip"'),
    # Cargo.toml: a dev-dependency on the crate itself enables its features in every cargo test
    ("a self dev-dependency enables its features", {"p": PHYS}, {"ci": "cargo test\n"}, None, CONF_RUN, 0, CARGO + SELF_DEV),
    ("a dev-dependency on another crate does not", {"p": PHYS}, {"ci": "cargo test\n"}, None, CONF_RUN, "ci: tests/p.rs (features ['physics'], 1 tests): features ['physics'] not enabled", CARGO + OTHER_DEV),
    # [[test]] required-features is a gate of the whole file
    ("required-features gates the file", {"r": PLAIN}, {"ci": "cargo test\n"}, None, CONF_RUN, "ci: tests/r.rs (features ['gpu'], 1 tests): features ['gpu'] not enabled", CARGO + REQUIRED),
    ("required-features met", {"r": PLAIN}, {"ci": "cargo test --features gpu\n"}, None, CONF_RUN, 0, CARGO + REQUIRED),
    # the configuration
    ("no configuration file", FILES, CI, COUNTED, None, 'no configuration scripts/gated_tests.toml'),
    ("an unknown mode", FILES, CI, COUNTED, CONF_COUNTED.replace('"counted"', '"maybe"'), 'mode \'maybe\': must be "counted" or "run"'),
    ("no cover", FILES, CI, COUNTED, 'mode = "run"\n', 'no [[cover]] in the configuration'),
]


def main() -> int:
    bad = 0
    for case in CASES:
        what, files, workflows, pre, conf, want = case[:6]
        code, out = run(files, workflows, pre, conf, *case[6:])
        # a crash is never the expected failure: the check must have found the problem itself
        if "Traceback" in out:
            why = "crashed"
        elif want == 0:
            why = None if code == 0 else f"exit {code}, expected 0"
        elif code != 1:
            why = f"exit {code}, expected 1"
        elif f"gated_tests: {want}" not in out:
            why = f"did not report {want!r}"
        else:
            why = None
        if why:
            print(f"test_gated_tests: {what}: {why}\n{out}", file=sys.stderr)
            bad += 1
    print(f"test_gated_tests: {len(CASES) - bad}/{len(CASES)} cases", file=sys.stderr)
    return 1 if bad or not CASES else 0


if __name__ == "__main__":
    sys.exit(main())
