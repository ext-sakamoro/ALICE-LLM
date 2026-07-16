// Q5_K dequantized matrix-vector multiply compute shader.
//
// Reference: `dequantize_q5_k` in `~/ALICE-LLM/src/gguf.rs:737-776`.
// Block layout (176 bytes = 44 u32 words per 256 elements):
//   word  0:       d(f16) | dmin(f16)
//   words 1-3:     scales[12 bytes] — 6-bit packed scales and mins (same as Q4_K)
//   words 4-11:    qh[32 bytes]     — high bit for each element (1 bit / element)
//   words 12-43:   qs[128 bytes]    — low 4 bits (2 elements per byte, same layout as Q4_K)
//
// Per-element math (matches CPU reference):
//   q_low  = (qs[q_offset + lane] >> (is_high * 4)) & 0xF          // 4-bit
//   q_high = (qh[lane] >> sub) & 1                                  // 1-bit
//   q      = q_low | (q_high << 4)                                  // 5-bit 0..31
//   sub    = group * 2 + is_high  (0..7, index into 8 sub-block scales/mins)
//   out    = d * scale[sub] * q - dmin * min[sub]
//
// Workgroup layout mirrors Q4_K (workgroup_size=256, one row per workgroup,
// tid iterates 1 element per outer block iteration, blocks_per_row iterations
// covering `cols = 256 * blocks_per_row` input columns).

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

// One slot per subgroup (max 8 subgroups on Metal / Vulkan with 256/32 threads).
var<workgroup> sg_partial: array<f32, 8>;

const BLOCK_WORDS: u32 = 44u;
// Q5_K layout: word 0 is d/dmin, words 1..3 are scales, words 4..11 are qh
// (32 bytes = 8 u32), words 12..43 are qs (128 bytes = 32 u32).
const QH_WORD_OFFSET: u32 = 4u;
const QS_WORD_OFFSET: u32 = 12u;

fn extract_byte(word: u32, pos: u32) -> u32 {
    return (word >> (pos * 8u)) & 0xFFu;
}

// Same 6-bit scale/min packing as Q4_K (see `dequant_matvec_q4k.wgsl`).
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
fn matvec_q5k(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let row = wid.y * params.grid_x + wid.x;
    if (row >= params.rows) { return; }

    let tid = lid.x;
    let row_word_offset = row * params.blocks_per_row * BLOCK_WORDS;

    // Indexing shared with Q4_K.
    let group = tid >> 6u;
    let in_group = tid & 63u;
    let is_high = in_group >> 5u;
    let lane = in_group & 31u;
    let sub = group * 2u + is_high;
    let qs_flat = group * 32u + lane;
    let qs_rel = qs_flat >> 2u;
    let nibble_shift = (qs_flat & 3u) * 8u + is_high * 4u;

    // qh: 32 bytes = 8 u32. Byte index = `lane` (0..31).
    // qh_word_idx = lane / 4, qh_byte_shift = (lane & 3) * 8.
    let qh_word_idx = lane >> 2u;
    let qh_byte_shift = (lane & 3u) * 8u;

    var acc: f32 = 0.0;

    for (var blk = 0u; blk < params.blocks_per_row; blk = blk + 1u) {
        let bw = row_word_offset + blk * BLOCK_WORDS;

        let dd = unpack2x16float(weights[bw]);
        let d = dd.x;
        let dmin = dd.y;

        let s0 = weights[bw + 1u];
        let s1 = weights[bw + 2u];
        let s2 = weights[bw + 3u];

        let sm = get_scale_min_k4(sub, s0, s1, s2);
        let d_sc = d * f32(sm.x);
        let bias = dmin * f32(sm.y);

        // Low 4 bits from qs (same nibble packing as Q4_K).
        let q_low = (weights[bw + QS_WORD_OFFSET + qs_rel] >> nibble_shift) & 0xFu;

        // High 1 bit from qh (bit `sub` of qh[lane]).
        let qh_byte = (weights[bw + QH_WORD_OFFSET + qh_word_idx] >> qh_byte_shift) & 0xFFu;
        let qh_bit = (qh_byte >> sub) & 1u;

        let q = q_low | (qh_bit << 4u);

        let input_idx = blk * 256u + tid;
        acc = fma(fma(d_sc, f32(q), -bias), input_vec[input_idx], acc);
    }

    // Subgroup reduction (same structure as Q4_K).
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
