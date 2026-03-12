// Fused SwiGLU Q4_K batch-4: weight decoded ONCE, 8 scalar FMAs in registers.
// gate0..gate3 + up0..up3 = 8 accumulators, all physical registers.
// output[k][row] = silu(gate_k) × up_k for k=0..3.

struct Params {
    rows: u32,
    cols: u32,
    blocks_per_row: u32,
    grid_x: u32,
}

@group(0) @binding(0) var<storage, read> gate_weights: array<u32>;
@group(0) @binding(1) var<storage, read> up_weights: array<u32>;
@group(0) @binding(2) var<storage, read> input_vec: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_vec: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const BLOCK_WORDS: u32 = 36u;

// 8 separate workgroup arrays — no dynamic indexing
var<workgroup> sg_g0: array<f32, 8>;
var<workgroup> sg_g1: array<f32, 8>;
var<workgroup> sg_g2: array<f32, 8>;
var<workgroup> sg_g3: array<f32, 8>;
var<workgroup> sg_u0: array<f32, 8>;
var<workgroup> sg_u1: array<f32, 8>;
var<workgroup> sg_u2: array<f32, 8>;
var<workgroup> sg_u3: array<f32, 8>;

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
fn swiglu_fused_q4k_batch4(
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

    // Hoist all tid-derived constants
    let group = tid >> 6u;
    let in_group = tid & 63u;
    let is_high = in_group >> 5u;
    let lane = in_group & 31u;
    let sub = group * 2u + is_high;
    let qs_flat = group * 32u + lane;
    let qs_rel = qs_flat >> 2u;
    let nibble_shift = (qs_flat & 3u) * 8u + is_high * 4u;

    // 8 scalar accumulators — all in physical registers
    var gate0: f32 = 0.0;
    var gate1: f32 = 0.0;
    var gate2: f32 = 0.0;
    var gate3: f32 = 0.0;
    var up0: f32 = 0.0;
    var up1: f32 = 0.0;
    var up2: f32 = 0.0;
    var up3: f32 = 0.0;

    for (var blk = 0u; blk < params.blocks_per_row; blk = blk + 1u) {
        let bw = row_word_offset + blk * BLOCK_WORDS;
        let qs_word_off = bw + 4u + qs_rel;

        // Gate weight dequant — done ONCE
        let g_dd = unpack2x16float(gate_weights[bw]);
        let g_sm = get_scale_min_k4(sub, gate_weights[bw + 1u], gate_weights[bw + 2u], gate_weights[bw + 3u]);
        let gw = fma(g_dd.x * f32(g_sm.x), f32((gate_weights[qs_word_off] >> nibble_shift) & 0xFu), -(g_dd.y * f32(g_sm.y)));

        // Up weight dequant — done ONCE
        let u_dd = unpack2x16float(up_weights[bw]);
        let u_sm = get_scale_min_k4(sub, up_weights[bw + 1u], up_weights[bw + 2u], up_weights[bw + 3u]);
        let uw = fma(u_dd.x * f32(u_sm.x), f32((up_weights[qs_word_off] >> nibble_shift) & 0xFu), -(u_dd.y * f32(u_sm.y)));

        // 4 input reads (shared between gate and up)
        let base_idx = blk * 256u + tid;
        let i0 = input_vec[base_idx];
        let i1 = input_vec[input_stride + base_idx];
        let i2 = input_vec[2u * input_stride + base_idx];
        let i3 = input_vec[3u * input_stride + base_idx];

        // 8 FMAs — pure ALU, zero spill
        gate0 = fma(gw, i0, gate0);
        gate1 = fma(gw, i1, gate1);
        gate2 = fma(gw, i2, gate2);
        gate3 = fma(gw, i3, gate3);
        up0 = fma(uw, i0, up0);
        up1 = fma(uw, i1, up1);
        up2 = fma(uw, i2, up2);
        up3 = fma(uw, i3, up3);
    }

    // 8 subgroup reductions (register-to-register SIMD shuffles)
    let g0s = subgroupAdd(gate0);
    let g1s = subgroupAdd(gate1);
    let g2s = subgroupAdd(gate2);
    let g3s = subgroupAdd(gate3);
    let u0s = subgroupAdd(up0);
    let u1s = subgroupAdd(up1);
    let u2s = subgroupAdd(up2);
    let u3s = subgroupAdd(up3);

    let sg_id = tid / sg_size;
    if (sg_lane == 0u) {
        sg_g0[sg_id] = g0s;
        sg_g1[sg_id] = g1s;
        sg_g2[sg_id] = g2s;
        sg_g3[sg_id] = g3s;
        sg_u0[sg_id] = u0s;
        sg_u1[sg_id] = u1s;
        sg_u2[sg_id] = u2s;
        sg_u3[sg_id] = u3s;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var tg0: f32 = 0.0; var tg1: f32 = 0.0; var tg2: f32 = 0.0; var tg3: f32 = 0.0;
        var tu0: f32 = 0.0; var tu1: f32 = 0.0; var tu2: f32 = 0.0; var tu3: f32 = 0.0;
        let n_sg = 256u / sg_size;
        for (var s = 0u; s < n_sg; s = s + 1u) {
            tg0 += sg_g0[s]; tg1 += sg_g1[s]; tg2 += sg_g2[s]; tg3 += sg_g3[s];
            tu0 += sg_u0[s]; tu1 += sg_u1[s]; tu2 += sg_u2[s]; tu3 += sg_u3[s];
        }

        let rows = params.rows;
        output_vec[row]              = (tg0 / (1.0 + exp(-tg0))) * tu0;
        output_vec[rows + row]       = (tg1 / (1.0 + exp(-tg1))) * tu1;
        output_vec[2u * rows + row]  = (tg2 / (1.0 + exp(-tg2))) * tu2;
        output_vec[3u * rows + row]  = (tg3 / (1.0 + exp(-tg3))) * tu3;
    }
}
