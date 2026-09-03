//! Fuzz target: GGUF binary parser を任意 byte 列で fuzz
//!
//! GGUF file は attacker-controlled input (HuggingFace / crates.io からの DL 経路含む)
//! となり得るため、GgufFile::parse は panic-free / OOM-free / bounded-memory であること
//! が必須
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 罠 catalog #6 準拠
//! (length prefix 上限なしで capacity overflow panic → DoS 脆弱性)
//!
//! 起こり得る危険:
//! - malformed header で `Vec::with_capacity(huge)` → capacity overflow panic
//! - tensor count / metadata count の bogus 値で allocator abort
//! - string length prefix 巨大で OOM
//! - integer overflow in offset arithmetic
//!
//! 期待 behavior: どんな入力でも `Option<GgufFile>` を返す (panic なし)

#![no_main]

use alice_llm::gguf::GgufFile;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // GgufFile::parse must never panic regardless of input bytes.
    // Malformed / truncated / adversarial inputs should return None gracefully.
    let _ = GgufFile::parse(data);
});
