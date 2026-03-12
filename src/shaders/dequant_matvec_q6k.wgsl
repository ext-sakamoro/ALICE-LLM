// Q6_K dequantized matrix-vector multiply compute shader.
// Each workgroup computes one output row: output[row] = dot(dequant(W[row,:]), input).
// Workgroup size 256 = one thread per element in a Q6_K block (256 elements/block).
// Uses subgroup (SIMD-group) reduction to minimize shared memory and barriers.
//
// Q6_K block layout (210 bytes):
//   ql[128]    — low 4 bits of 6-bit quants
//   qh[64]     — high 2 bits of 6-bit quants
//   scales[16] — per-group signed 8-bit scales
//   d[2]       — super-block scale (f16)

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

var<workgroup> sg_partial: array<f32, 8>;

const BLOCK_BYTES: u32 = 210u;

// Inline byte read: precomputed block word base + byte fraction
fn rb(bb_word: u32, bb_frac: u32, offset: u32) -> u32 {
    let a = bb_frac + offset;
    return (weights[bb_word + (a >> 2u)] >> ((a & 3u) << 3u)) & 0xFFu;
}

@compute @workgroup_size(256)
fn matvec_q6k(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let row = wid.y * params.grid_x + wid.x;
    if (row >= params.rows) { return; }

    let tid = lid.x;
    let row_bytes = params.blocks_per_row * BLOCK_BYTES;
    let row_byte_offset = row * row_bytes;

    // Hoist all tid-derived constants (loop-invariant)
    let half = tid >> 7u;
    let in_half = tid & 127u;
    let group = in_half >> 5u;
    let lane = in_half & 31u;
    let is_val = lane >> 4u;

    // Precompute byte offsets within block (loop-invariant)
    let ql_off = half << 6u;
    let p_ql = ql_off + lane + (group & 1u) * 32u;
    let p_qh = 128u + (half << 5u) + lane;
    let p_scale = 192u + (half << 3u) + is_val + group * 2u;

    // Branchless nibble/shift constants
    let ql_nibble_shift = (group >> 1u) * 4u;
    let qh_shift = group * 2u;

    var acc: f32 = 0.0;

    for (var blk = 0u; blk < params.blocks_per_row; blk = blk + 1u) {
        let bb = row_byte_offset + blk * BLOCK_BYTES;
        let bb_word = bb >> 2u;
        let bb_frac = bb & 3u;

        // Read d (f16 at offset 208)
        let d_pair = unpack2x16float(weights[bb_word + 52u]);
        let d = select(d_pair.x, d_pair.y, bb_frac != 0u);

        // Read ql, qh, scale bytes
        let ql_byte = rb(bb_word, bb_frac, p_ql);
        let qh_byte = rb(bb_word, bb_frac, p_qh);
        let scale_byte = rb(bb_word, bb_frac, p_scale);

        // Branchless 6-bit quant assembly
        let q_from_ql = (ql_byte >> ql_nibble_shift) & 0xFu;
        let q_from_qh = ((qh_byte >> qh_shift) & 3u) << 4u;
        let q_signed = i32(q_from_ql | q_from_qh) - 32;

        // Branchless signed i8 scale (XOR trick)
        let scale = i32(scale_byte ^ 128u) - 128;

        // FMA dequantize and dot with input
        let input_idx = blk * 256u + tid;
        acc = fma(d * f32(scale) * f32(q_signed), input_vec[input_idx], acc);
    }

    // --- Subgroup reduction ---
    let sg_sum = subgroupAdd(acc);

    let sg_id = tid / sg_size;
    if (sg_lane == 0u) {
        sg_partial[sg_id] = sg_sum;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var total: f32 = 0.0;
        let n_sg = 256u / sg_size;
        for (var s = 0u; s < n_sg; s = s + 1u) {
            total += sg_partial[s];
        }
        output_vec[row] = total;
    }
}
