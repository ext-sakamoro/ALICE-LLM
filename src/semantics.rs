//! Content hash of the arithmetic this build computes with:
//! [`LLM_SEMANTICS_ID`] (ALICE-Law G6), and the identifier a KV cache file
//! carries ([`RUNNING_SEMANTICS_ID`], G8)
//!
//! The recorded-bit tests in `tests/semantics_golden.rs` pin the kernels the
//! inference computes with: the `lawmath` transcendentals, the activations
//! and norms, softmax and the sampling filters, rotary embeddings, the Kimi
//! delta gates, quantization and a forward pass of the reference transformer
//! (`TransformerModel::new_test`). [`LLM_SEMANTICS_ID`] folds those digests and
//! the semantics identifier of `alice-det-math`. Two builds with the same
//! identifier compute every pinned scenario to the same bits.
//!
//! # Hex form (with `alice-det-math` at the git rev 2d01c63, its `SEMANTICS_ID` `a2a11b9b`)
//!
//! ```text
//! LLM_SEMANTICS_ID           c1eee96c2de49cfd114ac48ab8fef3d8e7bd72522b8d1160ef426fdcb904eadf
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
//! bytes and the 32 digest bytes.
//!
//! # Staleness
//!
//! [`LLM_SEMANTICS_CHECKED_AT`] names the crate version and toolchain the pins
//! were last checked on; a different version or toolchain fails
//! `tests/llm_semantics_id.rs` until they are checked again. An
//! `alice-det-math` generation with no row in [`LLM_SEMANTICS_BY_DET_MATH`]
//! does not compile.

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

/// The entries [`LLM_SEMANTICS_ID`] folds: one per recorded-bit scenario of
/// `tests/semantics_golden.rs`, named as that file names it, plus
/// `"alice-det-math"` holding `alice_det_math::SEMANTICS_ID`
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
    // the resolved alice-det-math's own identifier, not a copy of it: the
    // identifier below is looked up by it
    ("alice-det-math", alice_det_math::SEMANTICS_ID),
];

/// The crate version and toolchain the pins were last checked against
pub const LLM_SEMANTICS_CHECKED_AT: (&str, &str) = ("3.0.0", "1.98.1");

/// [`LLM_SEMANTICS_ID`] for each `alice-det-math` generation this crate
/// knows: `(alice_det_math::SEMANTICS_ID, fold of LLM_SEMANTICS_PINS with that
/// det-math entry)`
pub const LLM_SEMANTICS_BY_DET_MATH: &[([u8; 32], [u8; 32])] = &[
    // alice-det-math 0.5.1
    (
        hex32("85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315"),
        hex32("32957f18dac2838d705c034d0e27bcfc23f9680f40967338ea13d8e9d25c3f9d"),
    ),
    // alice-det-math at the git rev 2d01c63 (sin / cos at |x| >= 2^13, powf of a negative
    // base, powf64 as fdlibm e_pow and hypot of two subnormals change bits; erf / erf64 added):
    // the "rope" scenario moved with it (position 32767 reaches angles past 2^13; the
    // angle is the position times a frequency of at most 1, so 4095 stays below)
    (
        hex32("a2a11b9b826227fa7a9ed8ab2c1ce63dbb07ad4c39aa2c46a363ddea6b47dbe9"),
        hex32("c1eee96c2de49cfd114ac48ab8fef3d8e7bd72522b8d1160ef426fdcb904eadf"),
    ),
];

/// Content hash of the Law build's arithmetic: the fold of
/// [`LLM_SEMANTICS_PINS`], taken from [`LLM_SEMANTICS_BY_DET_MATH`] by the
/// resolved `alice-det-math`
pub const LLM_SEMANTICS_ID: [u8; 32] = semantics_id_for(&alice_det_math::SEMANTICS_ID);

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

/// Earlier values of [`LLM_SEMANTICS_ID`] whose builds compute every pinned
/// scenario to the same bits as this one (none yet)
pub const LLM_PREVIOUS_SEMANTICS_IDS: &[[u8; 32]] = &[];

/// The [`LLM_SEMANTICS_ID`] of a build that resolved the `alice-det-math`
/// with identifier `det_math`.
///
/// # Panics
///
/// When [`LLM_SEMANTICS_BY_DET_MATH`] has no row for `det_math` (in a const
/// context this stops the compile).
#[must_use]
pub const fn semantics_id_for(det_math: &[u8; 32]) -> [u8; 32] {
    let mut i = 0;
    while i < LLM_SEMANTICS_BY_DET_MATH.len() {
        if same32(&LLM_SEMANTICS_BY_DET_MATH[i].0, det_math) {
            return LLM_SEMANTICS_BY_DET_MATH[i].1;
        }
        i += 1;
    }
    panic!("alice-det-math SEMANTICS_ID has no row in LLM_SEMANTICS_BY_DET_MATH: add its fold");
}

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
