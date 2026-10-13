// Fused SwiGLU Q4_K kernel: output[row] = silu(gate·input) × up·input.
// Each workgroup computes one output row from two Q4_K weight matrices.
// Eliminates intermediate gate/up buffers and the silu_mul dispatch.
// Workgroup size 256 = one thread per element in a Q4_K block (256 elements/block).
// Uses subgroup (SIMD-group) reduction to minimize shared memory and barriers.

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

var<workgroup> sg_gate: array<f32, 8>;
var<workgroup> sg_up: array<f32, 8>;

const BLOCK_WORDS: u32 = 36u;

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
fn swiglu_fused_q4k(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let row = wid.y * params.grid_x + wid.x;
    if (row >= params.rows) { return; }

    let tid = lid.x;
    let row_word_offset = row * params.blocks_per_row * BLOCK_WORDS;

    // Hoist all tid-derived constants (loop-invariant)
    let group = tid >> 6u;
    let in_group = tid & 63u;
    let is_high = in_group >> 5u;
    let lane = in_group & 31u;
    let sub = group * 2u + is_high;
    let qs_flat = group * 32u + lane;
    let qs_rel = qs_flat >> 2u;
    let nibble_shift = (qs_flat & 3u) * 8u + is_high * 4u;

    var gate_acc: f32 = 0.0;
    var up_acc: f32 = 0.0;

    for (var blk = 0u; blk < params.blocks_per_row; blk = blk + 1u) {
        let bw = row_word_offset + blk * BLOCK_WORDS;
        let input_idx = blk * 256u + tid;
        let inp = input_vec[input_idx];
        let qs_word_off = bw + 4u + qs_rel;

        // --- Gate weight: dequant + FMA ---
        let g_dd = unpack2x16float(gate_weights[bw]);
        let g_sm = get_scale_min_k4(sub, gate_weights[bw + 1u], gate_weights[bw + 2u], gate_weights[bw + 3u]);
        let g_d_sc = g_dd.x * f32(g_sm.x);
        let g_bias = g_dd.y * f32(g_sm.y);
        let g_q = (gate_weights[qs_word_off] >> nibble_shift) & 0xFu;
        gate_acc = fma(fma(g_d_sc, f32(g_q), -g_bias), inp, gate_acc);

        // --- Up weight: dequant + FMA ---
        let u_dd = unpack2x16float(up_weights[bw]);
        let u_sm = get_scale_min_k4(sub, up_weights[bw + 1u], up_weights[bw + 2u], up_weights[bw + 3u]);
        let u_d_sc = u_dd.x * f32(u_sm.x);
        let u_bias = u_dd.y * f32(u_sm.y);
        let u_q = (up_weights[qs_word_off] >> nibble_shift) & 0xFu;
        up_acc = fma(fma(u_d_sc, f32(u_q), -u_bias), inp, up_acc);
    }

    // --- Subgroup reduction for both accumulators ---
    let gate_sg = subgroupAdd(gate_acc);
    let up_sg = subgroupAdd(up_acc);

    let sg_id = tid / sg_size;
    if (sg_lane == 0u) {
        sg_gate[sg_id] = gate_sg;
        sg_up[sg_id] = up_sg;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var gate_val: f32 = 0.0;
        var up_val: f32 = 0.0;
        let n_sg = 256u / sg_size;
        for (var s = 0u; s < n_sg; s = s + 1u) {
            gate_val += sg_gate[s];
            up_val += sg_up[s];
        }
        let silu_gate = gate_val / (1.0 + exp(-gate_val));
        output_vec[row] = silu_gate * up_val;
    }
}
