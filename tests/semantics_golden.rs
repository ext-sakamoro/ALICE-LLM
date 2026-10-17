//! Recorded bits of the kernels the inference computes with
//!
//! Each scenario runs fixed inputs through public entry points of the default
//! (Law) build and hashes the output bits with SHA-256. The digests are the
//! entries `LLM_SEMANTICS_PINS` (src/semantics.rs) folds; `tests/llm_semantics_id.rs`
//! reads them from this file. These are change detectors: the closed-form
//! checks of the same functions are in `tests/analytic_oracle.rs` and
//! `tests/lawmath_ln_1p.rs`.
//!
//! Only the default features are Law-bearing; under `platform-math` the
//! transcendentals come from the platform libm and these digests do not hold.

#![cfg(not(feature = "platform-math"))]

use alice_llm::batch_inference::{ModelConfig, TransformerModel};
use alice_llm::kv_cache::KvCache;
use alice_llm::lawmath;
use alice_llm::linalg::{gelu, layer_norm, rms_norm, silu};
use alice_llm::llama3::{kimi_delta_lower_bounded_decay, kimi_delta_output_gate};
use alice_llm::quantization::{QuantizedInt4, QuantizedInt8};
use alice_llm::rope::apply_rope;
use alice_llm::sampling::{apply_temperature, softmax, top_p_filter};
use sha2::{Digest, Sha256};

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

fn assert_golden(name: &str, bytes: &[u8], expected: &str) {
    let got = hex(&Sha256::digest(bytes));
    assert_eq!(
        got, expected,
        "scenario {name:?} moved: re-record only for an intended change"
    );
}

fn bits(out: &mut Vec<u8>, values: impl IntoIterator<Item = f32>) {
    for v in values {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
}

/// 4001 points over [-20, 20] (step 0.01, as f32 from an integer)
fn grid() -> Vec<f32> {
    (-2000_i16..=2000).map(|i| f32::from(i) / 100.0).collect()
}

#[test]
fn lawmath_functions_are_the_recorded_bits() {
    let g = grid();
    let mut out = Vec::new();
    bits(&mut out, g.iter().map(|&x| lawmath::exp(x)));
    bits(&mut out, g.iter().map(|&x| lawmath::tanh(x)));
    bits(&mut out, g.iter().map(|&x| lawmath::ln(x.abs() + 1e-3)));
    bits(&mut out, g.iter().map(|&x| lawmath::ln_1p(lawmath::exp(x))));
    bits(
        &mut out,
        g.iter().map(|&x| lawmath::powf(10_000.0, x / 20.0)),
    );
    bits(
        &mut out,
        g.iter().flat_map(|&x| {
            let (s, c) = lawmath::sin_cos(x * 7.0);
            [s, c]
        }),
    );
    bits(&mut out, (-40..=40).map(|n| lawmath::powi(0.9, n)));
    assert_golden(
        "lawmath functions",
        &out,
        "28a35ff713e3e39b11ebbfa5931662fd0f80f160ed341f90266058d915697058",
    );
}

#[test]
fn activations_and_norms_are_the_recorded_bits() {
    let g = grid();
    let mut out = Vec::new();
    bits(&mut out, g.iter().map(|&x| silu(x)));
    bits(&mut out, g.iter().map(|&x| gelu(x)));
    for chunk in g.chunks(64) {
        bits(&mut out, rms_norm(chunk, 1e-6));
        bits(&mut out, layer_norm(chunk, 1e-5));
    }
    assert_golden(
        "activations and norms",
        &out,
        "c8d3214d779643678054c268879a0b28e31355560f476a5c4684ecc6cb7e6af5",
    );
}

#[test]
fn softmax_and_sampling_filters_are_the_recorded_bits() {
    let g = grid();
    let mut out = Vec::new();
    for chunk in g.chunks(97) {
        bits(&mut out, softmax(chunk));
        let mut t = chunk.to_vec();
        apply_temperature(&mut t, 0.7);
        top_p_filter(&mut t, 0.9);
        bits(&mut out, t);
    }
    assert_golden(
        "softmax and sampling",
        &out,
        "766bc8cd3d681be4e4ef057528c811fdd46ee7de504de3b33d0b9bdc839bd2a2",
    );
}

#[test]
fn rope_is_the_recorded_bits() {
    let v: Vec<f32> = (0..128_u8).map(|i| f32::from(i) / 64.0 - 1.0).collect();
    let mut out = Vec::new();
    for pos in [0_usize, 1, 2, 7, 63, 511, 4095, 32_767] {
        bits(&mut out, apply_rope(&v, pos, 10_000.0));
        bits(&mut out, apply_rope(&v, pos, 500_000.0));
    }
    assert_golden(
        "rope",
        &out,
        "e615f1f3efe0eb51a280d57dcd462f331a01612703a5a27f2d06792b18ba8647",
    );
}

#[test]
fn kimi_delta_gates_are_the_recorded_bits() {
    let z: Vec<f32> = (0..64_u8).map(|i| f32::from(i) / 8.0 - 4.0).collect();
    let mut out = Vec::new();
    for log_a in [-2.0_f32, 0.0, 1.5] {
        bits(&mut out, kimi_delta_lower_bounded_decay(&z, log_a, -5.0));
    }
    let o_bar: Vec<f32> = (0..16_u8).map(|i| f32::from(i) / 4.0 - 2.0).collect();
    let gate: Vec<f32> = (0..16_u8).map(|i| 2.0 - f32::from(i) / 4.0).collect();
    let w: Vec<f32> = (0..64_u8).map(|i| f32::from(i % 7) / 7.0 - 0.5).collect();
    bits(
        &mut out,
        kimi_delta_output_gate(&o_bar, &gate, None, 1e-6, &w, 4),
    );
    assert_golden(
        "kimi delta gates",
        &out,
        "8ffdc9af40926cb1a03e7d32c3f842ec84936bac1e3acee5bfd86d35579d7744",
    );
}

#[test]
fn quantization_is_the_recorded_bits() {
    let g = grid();
    let mut out = Vec::new();
    for chunk in g.chunks(128) {
        let q8 = QuantizedInt8::quantize(chunk);
        bits(&mut out, [q8.scale, q8.zero_point]);
        bits(&mut out, q8.dequantize());
        let q4 = QuantizedInt4::quantize(chunk);
        bits(&mut out, [q4.scale, q4.zero_point]);
        bits(&mut out, q4.dequantize());
    }
    assert_golden(
        "quantization",
        &out,
        "6e1c07817d9a764ef413e63ffd9d8780bbcaba2352860c91796a671227df4c62",
    );
}

/// The reference transformer of `TransformerModel::new_test` (fixed weights):
/// a 32-token prefill, then 8 single-token steps on the same cache
#[test]
fn reference_transformer_forward_is_the_recorded_bits() {
    let config = ModelConfig {
        vocab_size: 256,
        hidden_dim: 64,
        num_heads: 4,
        num_layers: 2,
        max_seq_len: 64,
        rope_base: 10_000.0,
    };
    let model = TransformerModel::new_test(config);
    let mut cache = KvCache::new(2, 64, 64);
    let prompt: Vec<u32> = (0..32_u32).map(|i| (i * 37 + 5) % 256).collect();
    let mut out = Vec::new();
    bits(&mut out, model.forward(&prompt, &mut cache));
    for step in 0..8_u32 {
        bits(
            &mut out,
            model.forward(&[(step * 53 + 11) % 256], &mut cache),
        );
    }
    assert_golden(
        "reference transformer forward",
        &out,
        "849a960d7721d3fc2e34ef7196a2c18222bb81cfa0b00c5810678945b16d911e",
    );
}
