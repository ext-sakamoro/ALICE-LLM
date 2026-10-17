//! `LLM_SEMANTICS_ID` is the fold of the recorded-bit goldens of
//! `tests/semantics_golden.rs` and the `alice-det-math` identifier; the
//! `platform-math` build reports another identifier.
//!
//! Every digest is read from the golden test that records it, so changing a
//! recorded scenario without re-recording the entry (or the entry without the
//! identifier) fails here.

use std::collections::BTreeMap;
use std::path::Path;

use alice_llm::semantics::{LLM_PLATFORM_SEMANTICS_ID, LLM_SEMANTICS_ID, RUNNING_SEMANTICS_ID};
use alice_llm::semantics::{
    LLM_SEMANTICS_BY_DET_MATH, LLM_SEMANTICS_CHECKED_AT, LLM_SEMANTICS_PINS,
};
use sha2::{Digest, Sha256};

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

fn unhex32(s: &str) -> [u8; 32] {
    assert_eq!(s.len(), 64, "a digest is 64 hex digits: {s}");
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex digit");
    }
    out
}

/// The fold of `alice-det-math`: names sorted, u32 big-endian name length,
/// name bytes, digest bytes, all under one SHA-256.
fn fold(entries: &[(&str, [u8; 32])]) -> [u8; 32] {
    let mut sorted: Vec<_> = entries.to_vec();
    sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    let mut h = Sha256::new();
    for (name, digest) in &sorted {
        h.update(u32::try_from(name.len()).expect("short name").to_be_bytes());
        h.update(name.as_bytes());
        h.update(digest);
    }
    h.finalize().into()
}

/// `scenario name -> recorded digest` of every `assert_golden(` call in
/// `tests/semantics_golden.rs`
fn recorded_goldens() -> BTreeMap<String, [u8; 32]> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/semantics_golden.rs");
    let src = std::fs::read_to_string(&path).expect("tests/semantics_golden.rs");
    let mut out = BTreeMap::new();
    let mut rest = src.as_str();
    while let Some(at) = rest.find("assert_golden(") {
        rest = &rest[at + "assert_golden(".len()..];
        let call = &rest[..rest.find(");").expect("call ends")];
        let quoted: Vec<&str> = call.split('"').skip(1).step_by(2).collect();
        if let (Some(name), Some(digest)) = (quoted.first(), quoted.last()) {
            if digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()) {
                out.insert((*name).to_string(), unhex32(digest));
            }
        }
    }
    out
}

fn check_entries(pins: &[(&str, [u8; 32])], minimum: usize) {
    let goldens = recorded_goldens();
    let mut compared = 0;
    for (name, digest) in pins {
        if *name == "alice-det-math" {
            continue;
        }
        let recorded = goldens.get(*name).unwrap_or_else(|| {
            panic!("no recorded golden named {name:?} in tests/semantics_golden.rs")
        });
        assert_eq!(
            hex(digest),
            hex(recorded),
            "entry {name:?} is not the recorded golden"
        );
        compared += 1;
    }
    assert!(
        compared >= minimum,
        "compared {compared} entries: the table lost its pins"
    );
    // and the other way: every recorded golden is folded (a scenario added
    // to tests/semantics_golden.rs without an entry here would not move the
    // identifier)
    for name in goldens.keys() {
        assert!(
            pins.iter().any(|(n, _)| n == name),
            "the golden {name:?} of tests/semantics_golden.rs is not folded"
        );
    }
}

#[test]
fn every_semantics_entry_is_the_golden_its_scenario_records() {
    check_entries(LLM_SEMANTICS_PINS, 7);
}

#[test]
fn the_det_math_entry_is_the_det_math_identifier() {
    let entry = LLM_SEMANTICS_PINS
        .iter()
        .find(|(name, _)| *name == "alice-det-math")
        .expect("an alice-det-math entry");
    assert_eq!(entry.1, alice_det_math::SEMANTICS_ID);
}

#[test]
fn the_identifiers_are_the_folds_of_their_tables() {
    assert_eq!(hex(&fold(LLM_SEMANTICS_PINS)), hex(&LLM_SEMANTICS_ID));
    let row = LLM_SEMANTICS_BY_DET_MATH
        .iter()
        .find(|(det, _)| *det == alice_det_math::SEMANTICS_ID)
        .expect("a row for the resolved alice-det-math");
    assert_eq!(row.1, LLM_SEMANTICS_ID);
}

#[test]
fn the_identifiers_are_the_recorded_values() {
    assert_eq!(
        hex(&LLM_SEMANTICS_ID),
        "32957f18dac2838d705c034d0e27bcfc23f9680f40967338ea13d8e9d25c3f9d"
    );
}

#[test]
fn one_changed_bit_in_any_entry_changes_the_identifier() {
    for table in [LLM_SEMANTICS_PINS] {
        let id = fold(table);
        for i in 0..table.len() {
            let mut entries = table.to_vec();
            entries[i].1[0] ^= 1;
            assert_ne!(
                fold(&entries),
                id,
                "entry {} does not move the identifier",
                entries[i].0
            );
        }
    }
}

#[test]
fn the_platform_identifier_is_the_hash_of_its_sentence_and_differs() {
    let want: [u8; 32] = Sha256::digest(
        b"alice-llm platform-math build: the platform libm computes the transcendentals; not Law-bearing",
    )
    .into();
    assert_eq!(LLM_PLATFORM_SEMANTICS_ID, want);
    assert_ne!(LLM_PLATFORM_SEMANTICS_ID, LLM_SEMANTICS_ID);
}

#[test]
fn the_running_identifier_follows_the_build_mode() {
    if cfg!(feature = "platform-math") {
        assert_eq!(RUNNING_SEMANTICS_ID, LLM_PLATFORM_SEMANTICS_ID);
    } else {
        assert_eq!(RUNNING_SEMANTICS_ID, LLM_SEMANTICS_ID);
    }
}

#[test]
fn the_tables_and_identifiers_are_not_behind_a_cfg() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/semantics.rs");
    let src = std::fs::read_to_string(path).expect("src/semantics.rs");
    assert!(
        !src.contains("#[cfg"),
        "src/semantics.rs must not gate anything behind cfg"
    );
}

/// The pins were checked on this crate version and toolchain; a release or a
/// toolchain move without checking them again fails here
#[test]
fn the_pins_were_checked_on_this_version_and_toolchain() {
    assert_eq!(
        LLM_SEMANTICS_CHECKED_AT.0,
        env!("CARGO_PKG_VERSION"),
        "crate version"
    );
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("rust-toolchain.toml");
    let toml = std::fs::read_to_string(path).expect("rust-toolchain.toml");
    let channel = toml
        .lines()
        .find_map(|l| l.trim().strip_prefix("channel = "))
        .expect("a channel line")
        .trim_matches('"');
    assert_eq!(LLM_SEMANTICS_CHECKED_AT.1, channel, "toolchain channel");
    // the rustc that runs the test, so an override of the file is caught too
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let out = std::process::Command::new(rustc)
        .arg("-V")
        .output()
        .expect("rustc -V");
    let version = String::from_utf8(out.stdout).expect("utf-8");
    assert!(
        version.starts_with(&format!("rustc {} ", LLM_SEMANTICS_CHECKED_AT.1)),
        "running {version:?}, pins checked on {}",
        LLM_SEMANTICS_CHECKED_AT.1
    );
}
