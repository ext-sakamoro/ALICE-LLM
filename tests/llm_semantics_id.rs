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

/// 64-hex literals of the golden file that are not folded digests, each
/// with its reason
const UNFOLDED_DIGESTS: &[&str] = &[];

/// Goldens whose digest is not a literal, with the exact number of such calls
/// (none of them is a current output folded into the identifier)
const UNFOLDED_GOLDENS: &[(&str, usize)] = &[];

/// `scenario name -> recorded digest` of every `assert_golden(` call in
/// `tests/semantics_golden.rs`
fn recorded_goldens() -> BTreeMap<String, [u8; 32]> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/semantics_golden.rs");
    let src = std::fs::read_to_string(&path).expect("tests/semantics_golden.rs");
    // every use of the identifier other than its definition must be a call
    // this reads: a golden passed through a const, a macro or any other form
    // fails here instead of being skipped (so it cannot escape the fold)
    let src = code_only(&src);
    // the golden file pulls in no other source: `include!` (or `#[path]` / a `mod` item) would let a call
    // live in a file this scan does not read
    for banned in ["include!", "include_str!", "include_bytes!", "#[path"] {
        assert!(!src.contains(banned), "the golden file uses {banned}");
    }
    assert!(
        !src.lines().any(|l| {
            let t = l.trim_start();
            t.starts_with("mod ") || t.starts_with("pub mod ")
        }),
        "the golden file declares a module"
    );
    let mut out = BTreeMap::new();
    let mut unfolded: BTreeMap<String, usize> = BTreeMap::new();
    let bytes = src.as_bytes();
    let ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut uses = 0;
    let mut from = 0;
    while let Some(off) = src[from..].find("assert_golden") {
        let at = from + off;
        from = at + "assert_golden".len();
        if (at > 0 && ident(bytes[at - 1])) || bytes.get(from).is_some_and(|&b| ident(b)) {
            continue;
        }
        let before = src[..at].trim_end();
        if before.ends_with("fn")
            && !before[..before.len() - 2]
                .bytes()
                .next_back()
                .is_some_and(ident)
        {
            continue;
        }
        uses += 1;
        let args = call_args(&src[from..]).unwrap_or_else(|| {
            panic!("the assert_golden use at byte {at} is not a call this test can read")
        });
        let (name, digest) = match args.as_slice() {
            [name, .., digest] => (literal(name), literal(digest)),
            _ => (None, None),
        };
        let Some(name) = name else {
            panic!("the assert_golden call at byte {at} does not name its scenario with a literal")
        };
        let Some(digest) = digest else {
            // a digest that is not a literal is allowed only for a listed name,
            // counted exactly below
            assert!(
                UNFOLDED_GOLDENS.iter().any(|(n, _)| *n == name),
                "the assert_golden call {name:?} at byte {at} does not pass its digest \
                 as a string literal"
            );
            *unfolded.entry(name.to_string()).or_insert(0usize) += 1;
            continue;
        };
        assert!(
            digest.len() == 64 && digest.bytes().all(|b| b.is_ascii_hexdigit()),
            "the digest of {name:?} is not 64 hex digits"
        );
        assert!(
            out.insert(name.to_string(), unhex32(digest)).is_none(),
            "two goldens are named {name:?}"
        );
    }
    assert!(uses > 0, "no assert_golden call found");
    for (name, count) in UNFOLDED_GOLDENS {
        assert_eq!(
            unfolded.get(*name).copied().unwrap_or(0),
            *count,
            "calls of {name:?} with a digest that is not a literal"
        );
    }
    assert_eq!(
        out.len() + unfolded.values().sum::<usize>(),
        uses,
        "every assert_golden use is read"
    );
    // and a digest written anywhere else in the file (a helper that compares
    // without naming assert_golden) is one of the folded ones or listed
    let folded: Vec<String> = out.values().map(|d| hex(d)).collect();
    let mut extra: Vec<&str> = string_literals(&src)
        .into_iter()
        .filter(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
        .filter(|s| !folded.iter().any(|f| f == s))
        .collect();
    extra.sort_unstable();
    extra.dedup();
    let mut listed = UNFOLDED_DIGESTS.to_vec();
    listed.sort_unstable();
    assert_eq!(
        extra, listed,
        "the 64-hex literals of the golden file that are not folded digests"
    );
    out
}

/// The top-level arguments of the call whose `(` starts `rest` (after
/// whitespace), None when `rest` is not a call
fn call_args(rest: &str) -> Option<Vec<&str>> {
    let rest = rest.trim_start();
    let body = rest.strip_prefix('(')?;
    let (mut depth, mut in_str, mut escaped, mut start) = (0usize, false, false, 0);
    let mut args = Vec::new();
    for (i, c) in body.char_indices() {
        if in_str {
            match (escaped, c) {
                (true, _) => escaped = false,
                (false, '\\') => escaped = true,
                (false, '"') => in_str = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '(' | '[' | '{' => depth += 1,
            ')' if depth == 0 => {
                let last = body[start..i].trim();
                if !last.is_empty() {
                    args.push(last);
                }
                return Some(args);
            }
            ')' | ']' | '}' => depth = depth.checked_sub(1)?,
            ',' if depth == 0 => {
                args.push(body[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    None
}

/// The contents of the string literals of `code` (comments already blank)
fn string_literals(code: &str) -> Vec<&str> {
    let b = code.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'"' => {
                let start = i + 1;
                i = start;
                while i < b.len() && b[i] != b'"' {
                    i += if b[i] == b'\\' { 2 } else { 1 };
                }
                out.extend(code.get(start..i));
                i += 1;
            }
            b'\'' if b.get(i + 1) == Some(&b'\\') => {
                i += 2;
                while i < b.len() && b[i] != b'\'' {
                    i += 1;
                }
                i += 1;
            }
            b'\'' if b.get(i + 2) == Some(&b'\'') => i += 3,
            _ => i += 1,
        }
    }
    out
}

/// `src` with every comment replaced by spaces (same byte offsets), so a
/// mention in a comment is not a use and a comment cannot hide a call; string
/// and char literals are kept as they are
fn code_only(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = b.to_vec();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'"' => {
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    i += if b[i] == b'\\' { 2 } else { 1 };
                }
                i += 1;
            }
            b'\'' if b.get(i + 1) == Some(&b'\\') => {
                i += 2;
                while i < b.len() && b[i] != b'\'' {
                    i += 1;
                }
                i += 1;
            }
            b'\'' if b.get(i + 2) == Some(&b'\'') => i += 3,
            b'/' if b.get(i + 1) == Some(&b'/') => {
                while i < b.len() && b[i] != b'\n' {
                    out[i] = b' ';
                    i += 1;
                }
            }
            b'/' if b.get(i + 1) == Some(&b'*') => {
                let mut depth = 0usize;
                while i < b.len() {
                    if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
                        depth += 1;
                        out[i] = b' ';
                        out[i + 1] = b' ';
                        i += 2;
                    } else if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
                        depth -= 1;
                        out[i] = b' ';
                        out[i + 1] = b' ';
                        i += 2;
                        if depth == 0 {
                            break;
                        }
                    } else {
                        if b[i] != b'\n' {
                            out[i] = b' ';
                        }
                        i += 1;
                    }
                }
            }
            _ => i += 1,
        }
    }
    String::from_utf8(out).expect("only ASCII bytes were replaced")
}

/// The content of a plain string literal (no escapes), None for anything else
fn literal(arg: &str) -> Option<&str> {
    arg.strip_prefix('"')?
        .strip_suffix('"')
        .filter(|s| !s.contains('"') && !s.contains('\\'))
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

/// Goldens live in `tests/semantics_golden.rs` only: no other file under `tests/` (any extension, any depth: a
/// fixture can be pulled in by `include!`) uses `assert_golden`, so a scenario cannot be recorded where the
/// fold does not look
///
/// What a source scan cannot see, in this file or any other: a digest a helper builds at run time
/// (`format!` / `concat` of parts) or a comparison against a byte array; such a golden is not folded and
/// must not be written (the convention is `assert_golden` with literal arguments)
#[test]
fn no_other_test_file_records_a_golden() {
    fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("tests/") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else {
                out.push(path);
            }
        }
    }
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut files = Vec::new();
    walk(&dir, &mut files);
    let mut seen = 0;
    for path in files {
        let name = path
            .strip_prefix(&dir)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();
        if name == "semantics_golden.rs" || name == "llm_semantics_id.rs" {
            continue;
        }
        seen += 1;
        let raw = String::from_utf8_lossy(&std::fs::read(&path).expect("test file")).into_owned();
        let is_rs = path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("rs"));
        let code = if is_rs { code_only(&raw) } else { raw };
        let bytes = code.as_bytes();
        let ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
        let mut from = 0;
        while let Some(off) = code[from..].find("assert_golden") {
            let at = from + off;
            from = at + "assert_golden".len();
            let part_of_another =
                (at > 0 && ident(bytes[at - 1])) || bytes.get(from).is_some_and(|&b| ident(b));
            assert!(
                part_of_another,
                "tests/{name} uses assert_golden at byte {at}"
            );
        }
    }
    assert!(seen > 0, "no other file under tests/ was read");
}
