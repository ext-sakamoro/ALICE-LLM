// Q8_0 dequantized matrix-vector multiply compute shader.
//
// Reference: `dequantize_q8_0` in `~/ALICE-LLM/src/gguf.rs:778-792`.
// Q8_0 block layout (34 bytes / 32 elements in the GGUF file):
//   bytes 0..2:  d (f16 super-scale)
//   bytes 2..34: qs (32 × signed int8 quantized weights)
//
// GPU upload path pads each block to 36 bytes (9 u32 words) so the layout
// becomes word-aligned:
//   word 0: [d(f16, low 16 bits) | 0(pad, high 16 bits)]
//   words 1..8: qs (8 u32 × 4 signed int8 bytes = 32 total)
//
// Per-element math: out[k] = d * i8(qs[k]).
//
// Workgroup layout: 32 threads = one element per block per thread. One
// workgroup per output row, iterating `blocks_per_row` blocks. Reduction
// via subgroup ops (small workgroup so 1-2 subgroups typical).

struct Params {
    rows: u32,
    cols: u32,
    blocks_per_row: u32,
    grid_x: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input_vec: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_vec: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// Padded block size: 36 bytes = 9 u32 words per 32 elements.
const BLOCK_WORDS: u32 = 9u;

// Reinterpret the low byte of `u` as a signed int8 and return as f32.
// bytes 128..255 map to -128..-1 (subtract 256 when sign bit is set).
fn sign_extend_i8(u: u32) -> f32 {
    let sign = f32((u >> 7u) & 1u);
    return f32(u) - sign * 256.0;
}

// Small shared array — max 2 subgroups on typical GPUs (workgroup_size=32,
// subgroup_size >= 16).
var<workgroup> sg_partial: array<f32, 2>;

@compute @workgroup_size(32)
fn matvec_q8_0(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let row = wid.y * params.grid_x + wid.x;
    if (row >= params.rows) { return; }

    let tid = lid.x;
    let row_word_offset = row * params.blocks_per_row * BLOCK_WORDS;

    // Per-thread byte position within each padded block:
    //   qs starts at word 1 (byte 4), so element `tid` is at byte 4+tid.
    //   qs_word_idx = 1 + tid / 4, byte_shift = (tid & 3) * 8.
    let qs_word_idx = 1u + (tid >> 2u);
    let byte_shift = (tid & 3u) * 8u;

    var acc: f32 = 0.0;

    for (var blk = 0u; blk < params.blocks_per_row; blk = blk + 1u) {
        let bw = row_word_offset + blk * BLOCK_WORDS;

        // d occupies low 16 bits of word 0 (padded upload leaves high 16 = 0).
        let d = unpack2x16float(weights[bw]).x;

        let qs_byte = (weights[bw + qs_word_idx] >> byte_shift) & 0xFFu;
        let q = sign_extend_i8(qs_byte);

        let input_idx = blk * 32u + tid;
        acc = fma(d * q, input_vec[input_idx], acc);
    }

    // Subgroup reduction — small workgroup so at most 2 subgroups.
    let sg_sum = subgroupAdd(acc);
    let sg_id = tid / sg_size;
    if (sg_lane == 0u) {
        sg_partial[sg_id] = sg_sum;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var total: f32 = 0.0;
        let n_sg = 32u / sg_size;
        for (var s = 0u; s < n_sg; s = s + 1u) {
            total += sg_partial[s];
        }
        output_vec[row] = total;
    }
}
