//! Fuzz target: GBNF grammar text parser を任意 UTF-8 文字列で fuzz
//!
//! GBNF grammar は constrained decoding (OpenAI /v1/completions の `grammar` field 等)
//! で user-supplied な文字列として渡ってくる経路のため、任意入力で panic しないことが必須
//!
//! 起こり得る危険:
//! - 深い再帰 grammar (`a ::= a a a ...`) で stack overflow
//! - unicode escape / char class の巨大範囲で allocator abort
//! - malformed syntax で index-out-of-bounds
//! - 無限ループ相当の parse (arbitrary crate は timeout 済のため直接 concern ではないが記録)
//!
//! 期待 behavior: どんな UTF-8 入力でも `Result<Grammar, GbnfError>` を返す (panic なし)

#![no_main]

use alice_llm::grammar::gbnf::parse_gbnf;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // parse_gbnf takes &str, so only valid UTF-8 slices are exercised.
    // Invalid UTF-8 is discarded here rather than panicking — that path
    // is covered by GgufFile parse (binary) fuzz target instead.
    let Ok(src) = std::str::from_utf8(data) else {
        return;
    };
    // Cap length to keep memory bounded; grammar too large is a separate
    // (application-level) concern, not a security property of the parser.
    if src.len() > 64 * 1024 {
        return;
    }
    let _ = parse_gbnf(src);
});
