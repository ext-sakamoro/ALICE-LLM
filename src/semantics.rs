//! Content hash of the arithmetic this build computes with:
//! [`LLM_SEMANTICS_ID`] (ALICE-Law G6), and the identifier a KV cache file
//! carries ([`RUNNING_SEMANTICS_ID`], G8)
//!
//! The recorded-bit tests in `tests/semantics_golden.rs` pin the kernels the
//! inference computes with: the `lawmath` transcendentals, the activations
//! and norms, softmax and the sampling filters, the choice of a token, rotary embeddings, the Kimi
//! delta gates, quantization and a forward pass of the reference transformer
//! (`TransformerModel::new_test`). [`LLM_SEMANTICS_ID`] folds those digests and
//! [`DET_MATH_PINS`], the identifier of each `alice-det-math` function this
//! crate calls ([`DET_MATH_FUNCTIONS`]). Two builds with the same identifier
//! compute every pinned scenario to the same bits. A function this crate does
//! not call moves nothing; one it calls stops the compile.
//!
//! # Hex form
//!
//! ```text
//! LLM_SEMANTICS_ID           5e093a0494ac6bc0c7d07cee7de741ff63246e6c1b9925710d5257663079d1c6
//! LLM_PLATFORM_SEMANTICS_ID  269ade8fc103bb500c378ba0b10b61f2da534968ffd4c987a818e577aa033fac
//! ```
//!
//! # The non-Law build
//!
//! With the `platform-math` feature the transcendentals come from the
//! platform libm, so the pinned scenarios do not hold. That build reports
//! [`LLM_PLATFORM_SEMANTICS_ID`] as [`RUNNING_SEMANTICS_ID`]: the SHA-256 of
//! a fixed sentence, which no fold produces, so a KV cache written in one
//! mode is refused by the other.
//!
//! # Fold
//!
//! The procedure of the `SEMANTICS_ID` fold of `alice-det-math`: sort the
//! entries by name (ascending byte order), then SHA-256 over, for each entry
//! in that order, the name length as 4 bytes big-endian (`u32`), the name
//! bytes and the 32 digest bytes. A [`DET_MATH_PINS`] entry goes in under the
//! name `det-math <function>`.
//!
//! # Staleness
//!
//! [`LLM_SEMANTICS_CHECKED_AT`] names the crate version and toolchain the pins
//! were last checked on; a different version or toolchain fails
//! `tests/llm_semantics_id.rs` until they are checked again.

/// Decodes 64 hex digits into 32 bytes at compile time; anything else is a
/// compile error.
const fn hex32(s: &str) -> [u8; 32] {
    const fn nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => panic!("not a lower-case hex digit"),
        }
    }
    let b = s.as_bytes();
    assert!(b.len() == 64, "a digest is 64 hex digits");
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = nibble(b[2 * i]) * 16 + nibble(b[2 * i + 1]);
        i += 1;
    }
    out
}

/// The entries [`LLM_SEMANTICS_ID`] folds with [`DET_MATH_PINS`]: one per
/// recorded-bit scenario of `tests/semantics_golden.rs`, named as that file
/// names it
pub const LLM_SEMANTICS_PINS: &[(&str, [u8; 32])] = &[
    (
        "lawmath functions",
        hex32("28a35ff713e3e39b11ebbfa5931662fd0f80f160ed341f90266058d915697058"),
    ),
    (
        "activations and norms",
        hex32("c8d3214d779643678054c268879a0b28e31355560f476a5c4684ecc6cb7e6af5"),
    ),
    (
        "softmax and sampling",
        hex32("766bc8cd3d681be4e4ef057528c811fdd46ee7de504de3b33d0b9bdc839bd2a2"),
    ),
    (
        "sampling choices",
        hex32("aa3282680998a20f1999057d57f21ecf162eb455ec3f3493e8ce16b25e6df1f5"),
    ),
    (
        "rope",
        hex32("5abde8f7b2926a6a0932c048cd37ca439470f5da6ddbd72862617afcbcdfd9d5"),
    ),
    (
        "kimi delta gates",
        hex32("8ffdc9af40926cb1a03e7d32c3f842ec84936bac1e3acee5bfd86d35579d7744"),
    ),
    (
        "quantization",
        hex32("6e1c07817d9a764ef413e63ffd9d8780bbcaba2352860c91796a671227df4c62"),
    ),
    (
        "reference transformer forward",
        hex32("849a960d7721d3fc2e34ef7196a2c18222bb81cfa0b00c5810678945b16d911e"),
    ),
];

/// The crate version and toolchain the pins were last checked against
pub const LLM_SEMANTICS_CHECKED_AT: (&str, &str) = ("4.0.0", "1.98.1");

/// The functions of `alice-det-math` this crate calls (ALICE-Law's
/// `scripts/fold_set_check.py` compares this with the calls in the source and
/// writes it)
pub const DET_MATH_FUNCTIONS: &[&str] = &[
    "clamp", "cos", "exp", "ln", "max", "max64", "min", "min64", "powf", "powi", "sin", "sin_cos",
    "tanh",
];

/// The identifier of each function of [`DET_MATH_FUNCTIONS`], in that order.
///
/// As recorded when [`LLM_SEMANTICS_ID`] was computed. A function whose
/// identifier moves in the resolved `alice-det-math` stops the compile, rather
/// than reporting an identifier computed from other bits.
pub const DET_MATH_PINS: &[(&str, [u8; 32])] = &[
    (
        "clamp",
        hex32("8a35afd329a733bcd0c771a8921fbf21a82491954fc8293881220944dc820455"),
    ),
    (
        "cos",
        hex32("f7627941400acabe93c8dca4090eee12f8910fe4c14a8735ad456cef3c48ed7b"),
    ),
    (
        "exp",
        hex32("606918f2d637f46f9a792a29b3ee31a991ef3196064456caa7b3a7ca81700a63"),
    ),
    (
        "ln",
        hex32("6fe18bc0fba8b7e3e871c4cbe555ddb40a460be7b3c7690303ee30accdcb7291"),
    ),
    (
        "max",
        hex32("8bd32fcb313f7fb3d7a699fc3075273de7e5a4e083912d8dc7f4cc4016fc2286"),
    ),
    (
        "max64",
        hex32("aac586eacc9acd69cf5988ffe736283c71fad8146027e86034d3e9b881db28b6"),
    ),
    (
        "min",
        hex32("2939e83813b26c868ec3113fb1c25102f16f28d8f24c343a727d4b1bec5904fe"),
    ),
    (
        "min64",
        hex32("1bef418073152f5e1e76f41718aa29b2517de8966a8c8ea8c1c2b5ba92fab3cb"),
    ),
    (
        "powf",
        hex32("0f1f85adcfa801b4afbbf858cf4c30f4ed0abdd3b7163d71aa9aa2b6031d9117"),
    ),
    (
        "powi",
        hex32("d3d76ba59208f7b0151ccde9b63043f538bc6b7737d1f16ff02d918f5806be45"),
    ),
    (
        "sin",
        hex32("ee1b80d12df18ec0de5bce5a6bf05432484dabfc1cee1bb9b3f3291dd862cfec"),
    ),
    (
        "sin_cos",
        hex32("025acd2cdc900eff683e39ef1e71de115dafe8da8a1a4b02847b41d75cbd130f"),
    ),
    (
        "tanh",
        hex32("d8a147acf5f34b1072d5409f94cd6b9cadd06d77e484634d26c2a113011c0ea3"),
    ),
];

const _: () = alice_det_math::assert_function_pins(DET_MATH_PINS);

/// Content hash of the Law build's arithmetic: the fold of
/// [`LLM_SEMANTICS_PINS`] and [`DET_MATH_PINS`]
pub const LLM_SEMANTICS_ID: [u8; 32] =
    hex32("5e093a0494ac6bc0c7d07cee7de741ff63246e6c1b9925710d5257663079d1c6");

/// The identifier the `platform-math` build reports: SHA-256 of the sentence
/// "alice-llm platform-math build: the platform libm computes the
/// transcendentals; not Law-bearing"
pub const LLM_PLATFORM_SEMANTICS_ID: [u8; 32] =
    hex32("269ade8fc103bb500c378ba0b10b61f2da534968ffd4c987a818e577aa033fac");

/// The identifier of the running build: [`LLM_SEMANTICS_ID`], or
/// [`LLM_PLATFORM_SEMANTICS_ID`] with the `platform-math` feature
pub const RUNNING_SEMANTICS_ID: [u8; 32] = if cfg!(feature = "platform-math") {
    LLM_PLATFORM_SEMANTICS_ID
} else {
    LLM_SEMANTICS_ID
};

/// Earlier values of [`LLM_SEMANTICS_ID`] whose KV cache files this build reads
///
/// Their builds compute what a cache holds to the same bits. `c1eee96c` differs only in the fold,
/// which then gained the "sampling choices" scenario; the choice of a token is not in a cache.
/// `cd0f624f` differs only in the fold too: it took alice-det-math's whole identifier.
pub const LLM_PREVIOUS_SEMANTICS_IDS: &[[u8; 32]] = &[
    hex32("c1eee96c2de49cfd114ac48ab8fef3d8e7bd72522b8d1160ef426fdcb904eadf"),
    hex32("cd0f624fc7210862703502a5ead1b6baafe65af5f9240ab63ab2e954bb7875dc"),
];

/// Byte equality usable in a const context
#[must_use]
pub const fn same32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut i = 0;
    while i < 32 {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}
