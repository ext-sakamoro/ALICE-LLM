// MXFP4 (OCP Microscaling FP4, MX v1.0) block dequantization compute
// shader. Phase X.4.g.1 landing.
//
// Reference: `dequantize_mxfp4_block` in `~/ALICE-LLM/src/gguf.rs`
// (native path) + `pack_mxfp4_for_gpu_upload` in same file (upload
// padding). Native MXFP4 block layout is 17 bytes (1 E8M0 scale + 16
// packed E2M1). GPU upload padding pads each block to 20 bytes = 5
// u32 words so the storage buffer is word-aligned (mirrors the Q8_0
// pipeline's 34 → 36 padded layout).
//
// Padded block layout (per `pack_mxfp4_for_gpu_upload`):
//   word 0: [scale(u8, low 8 bits) | 0 (padding, high 24 bits)]
//   word 1: bytes 0-3 of the 16 packed E2M1 nibbles
//   word 2: bytes 4-7 ...
//   word 3: bytes 8-11 ...
//   word 4: bytes 12-15 ...
//
// Each E2M1 byte packs 2 elements: `low nibble = element 2k`,
// `high nibble = element 2k+1`, per OCP MX v1.0 §5.5.
//
// Per-element math:
//   scale_f32 = exp2(scale_byte - 127)   [E8M0, with 255 → NaN]
//   value_f32 = E2M1_TABLE[nibble] * scale_f32
//
// Workgroup layout: 32 threads = one output element per thread within
// one block. One workgroup per block; caller dispatches
// `blocks_per_row × rows` workgroups (or restructures to
// row-major-friendly layouts once fused matvec lands at Phase
// X.4.g.2).

struct Params {
    // Number of 32-element MXFP4 blocks in the input weight buffer.
    n_blocks: u32,
    // Padding to keep the struct 16-byte aligned (WGSL uniform
    // requirement). Consumed as 0 by the shader.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_vec: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// Padded block size: 20 bytes = 5 u32 words per 32 elements.
const BLOCK_WORDS: u32 = 5u;

// OCP MX v1.0 §Table 1 — E2M1 decoding table. Nibble index 0..7 =
// positive values, 8..15 = negative (including signed zero at
// index 8). Constant array is baked into the shader binary; no
// runtime branch overhead.
const E2M1_TABLE: array<f32, 16> = array<f32, 16>(
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    // -0.0 encoded as its bit-negation of +0.0 (WGSL treats them
    // equivalent for arithmetic; the sign bit round-trips through
    // multiplication).
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
);

// Decode an E8M0 scale byte to an f32 multiplier.
//
// Per OCP MX v1.0 §5.4, E8M0 encodes an unsigned 8-bit exponent with
// bias 127, so the scale is `2^(byte - 127)`. Reserved codepoints:
// `byte == 0` → 2^(-127) (denormal-scale edge case) and
// `byte == 255` → NaN (per spec).
fn decode_e8m0_scale(byte: u32) -> f32 {
    if (byte == 255u) {
        // Produce a NaN. WGSL does not have a nan() intrinsic in
        // MSL/GLSL portable form, so use 0/0 which every backend
        // canonicalises to NaN.
        return 0.0 / 0.0;
    }
    return exp2(f32(byte) - 127.0);
}

// Extract the k-th byte (0-indexed, 0..4) from a u32 word.
fn extract_byte(word: u32, k: u32) -> u32 {
    return (word >> (k * 8u)) & 0xFFu;
}

@compute @workgroup_size(32)
fn dequant_mxfp4(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let block_idx = wid.x;
    if (block_idx >= params.n_blocks) {
        return;
    }
    let elem_idx = lid.x; // 0..32

    let block_base_word = block_idx * BLOCK_WORDS;
    // Word 0 low byte = E8M0 scale.
    let scale_byte = weights[block_base_word] & 0xFFu;
    let scale_f32 = decode_e8m0_scale(scale_byte);

    // Determine which packed byte contains this element and which
    // nibble half. Byte b in bytes 0..16 contains elements 2b and 2b+1.
    let byte_offset_within_block = elem_idx / 2u; // 0..16
    let nibble_half = elem_idx & 1u; // 0 = low, 1 = high

    // Bytes 0..16 of the E2M1 payload live in words 1..5 (4 bytes each).
    let word_offset = byte_offset_within_block / 4u; // 0..4
    let byte_in_word = byte_offset_within_block & 3u; // 0..4
    let packed_word = weights[block_base_word + 1u + word_offset];
    let packed_byte = extract_byte(packed_word, byte_in_word);
    let nibble = select(packed_byte & 0x0Fu, (packed_byte >> 4u) & 0x0Fu, nibble_half == 1u);

    let value = E2M1_TABLE[nibble] * scale_f32;
    let out_idx = block_idx * 32u + elem_idx;
    output_vec[out_idx] = value;
}
