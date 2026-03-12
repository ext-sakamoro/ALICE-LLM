// Q6_K batch-4 matvec: weight decoded ONCE, 4 scalar FMAs in registers.
// Zero dynamic array indexing — acc0..acc3 guaranteed in physical registers.

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

const BLOCK_BYTES: u32 = 210u;

var<workgroup> sg0: array<f32, 8>;
var<workgroup> sg1: array<f32, 8>;
var<workgroup> sg2: array<f32, 8>;
var<workgroup> sg3: array<f32, 8>;

fn rb(bb_word: u32, bb_frac: u32, offset: u32) -> u32 {
    let a = bb_frac + offset;
    return (weights[bb_word + (a >> 2u)] >> ((a & 3u) << 3u)) & 0xFFu;
}

@compute @workgroup_size(256)
fn matvec_q6k_batch4(
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
    let input_stride = params.blocks_per_row * 256u;

    // Hoist all tid-derived constants (loop-invariant)
    let half = tid >> 7u;
    let in_half = tid & 127u;
    let group = in_half >> 5u;
    let lane = in_half & 31u;
    let is_val = lane >> 4u;

    let ql_off = half << 6u;
    let p_ql = ql_off + lane + (group & 1u) * 32u;
    let p_qh = 128u + (half << 5u) + lane;
    let p_scale = 192u + (half << 3u) + is_val + group * 2u;

    let ql_nibble_shift = (group >> 1u) * 4u;
    let qh_shift = group * 2u;

    // 4 scalar accumulators — guaranteed in GPU registers
    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    for (var blk = 0u; blk < params.blocks_per_row; blk = blk + 1u) {
        let bb = row_byte_offset + blk * BLOCK_BYTES;
        let bb_word = bb >> 2u;
        let bb_frac = bb & 3u;

        // Weight dequantization — done ONCE, reused 4×
        let d_pair = unpack2x16float(weights[bb_word + 52u]);
        let d = select(d_pair.x, d_pair.y, bb_frac != 0u);

        let ql_byte = rb(bb_word, bb_frac, p_ql);
        let qh_byte = rb(bb_word, bb_frac, p_qh);
        let scale_byte = rb(bb_word, bb_frac, p_scale);

        let q_from_ql = (ql_byte >> ql_nibble_shift) & 0xFu;
        let q_from_qh = ((qh_byte >> qh_shift) & 3u) << 4u;
        let q_signed = i32(q_from_ql | q_from_qh) - 32;
        let scale = i32(scale_byte ^ 128u) - 128;
        let w = d * f32(scale) * f32(q_signed);

        // 4 input reads + 4 FMAs — pure ALU, zero spill
        let base_idx = blk * 256u + tid;
        acc0 = fma(w, input_vec[base_idx], acc0);
        acc1 = fma(w, input_vec[input_stride + base_idx], acc1);
        acc2 = fma(w, input_vec[2u * input_stride + base_idx], acc2);
        acc3 = fma(w, input_vec[3u * input_stride + base_idx], acc3);
    }

    // 4 subgroup reductions
    let s0 = subgroupAdd(acc0);
    let s1 = subgroupAdd(acc1);
    let s2 = subgroupAdd(acc2);
    let s3 = subgroupAdd(acc3);

    let sg_id = tid / sg_size;
    if (sg_lane == 0u) {
        sg0[sg_id] = s0;
        sg1[sg_id] = s1;
        sg2[sg_id] = s2;
        sg3[sg_id] = s3;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var t0: f32 = 0.0;
        var t1: f32 = 0.0;
        var t2: f32 = 0.0;
        var t3: f32 = 0.0;
        let n_sg = 256u / sg_size;
        for (var s = 0u; s < n_sg; s = s + 1u) {
            t0 += sg0[s];
            t1 += sg1[s];
            t2 += sg2[s];
            t3 += sg3[s];
        }
        let rows = params.rows;
        output_vec[row] = t0;
        output_vec[rows + row] = t1;
        output_vec[2u * rows + row] = t2;
        output_vec[3u * rows + row] = t3;
    }
}
