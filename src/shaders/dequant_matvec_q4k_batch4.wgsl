// Q4_K batch-4 matvec: weight decoded ONCE, 4 scalar FMAs in registers.
// Zero dynamic array indexing — acc0..acc3 guaranteed in physical registers.
// Weight read: constant. FMA overhead: 4× ALU only. No memory spill.

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

const BLOCK_WORDS: u32 = 36u;

var<workgroup> sg0: array<f32, 8>;
var<workgroup> sg1: array<f32, 8>;
var<workgroup> sg2: array<f32, 8>;
var<workgroup> sg3: array<f32, 8>;

fn extract_byte(word: u32, pos: u32) -> u32 {
    return (word >> (pos * 8u)) & 0xFFu;
}

fn get_scale_min_k4(j: u32, s0: u32, s1: u32, s2: u32) -> vec2<u32> {
    var sc: u32;
    var mn: u32;
    if (j < 4u) {
        sc = extract_byte(s0, j) & 63u;
        mn = extract_byte(s1, j) & 63u;
    } else {
        let jj = j - 4u;
        let b_jp4 = extract_byte(s2, jj);
        let b_jm4 = extract_byte(s0, jj);
        let b_j = extract_byte(s1, jj);
        sc = (b_jp4 & 0xFu) | ((b_jm4 >> 6u) << 4u);
        mn = (b_jp4 >> 4u) | ((b_j >> 6u) << 4u);
    }
    return vec2<u32>(sc, mn);
}

@compute @workgroup_size(256)
fn matvec_q4k_batch4(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let row = wid.y * params.grid_x + wid.x;
    if (row >= params.rows) { return; }

    let tid = lid.x;
    let row_word_offset = row * params.blocks_per_row * BLOCK_WORDS;
    let input_stride = params.blocks_per_row * 256u;

    // Hoist all tid-derived constants (loop-invariant)
    let group = tid >> 6u;
    let in_group = tid & 63u;
    let is_high = in_group >> 5u;
    let lane = in_group & 31u;
    let sub = group * 2u + is_high;
    let qs_flat = group * 32u + lane;
    let qs_rel = qs_flat >> 2u;
    let nibble_shift = (qs_flat & 3u) * 8u + is_high * 4u;

    // 4 scalar accumulators — guaranteed in GPU registers
    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    for (var blk = 0u; blk < params.blocks_per_row; blk = blk + 1u) {
        let bw = row_word_offset + blk * BLOCK_WORDS;

        // Weight dequantization — done ONCE, reused 4×
        let dd = unpack2x16float(weights[bw]);
        let sm = get_scale_min_k4(sub, weights[bw + 1u], weights[bw + 2u], weights[bw + 3u]);
        let d_sc = dd.x * f32(sm.x);
        let bias = dd.y * f32(sm.y);
        let q = (weights[bw + 4u + qs_rel] >> nibble_shift) & 0xFu;
        let w = fma(d_sc, f32(q), -bias);

        // 4 input reads + 4 FMAs — pure ALU, zero spill
        let base_idx = blk * 256u + tid;
        acc0 = fma(w, input_vec[base_idx], acc0);
        acc1 = fma(w, input_vec[input_stride + base_idx], acc1);
        acc2 = fma(w, input_vec[2u * input_stride + base_idx], acc2);
        acc3 = fma(w, input_vec[3u * input_stride + base_idx], acc3);
    }

    // 4 subgroup reductions (register-to-register SIMD shuffles)
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
