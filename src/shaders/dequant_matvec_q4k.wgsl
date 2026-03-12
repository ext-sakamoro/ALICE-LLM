// Q4_K dequantized matrix-vector multiply compute shader.
// Each workgroup computes one output row: output[row] = dot(dequant(W[row,:]), input).
// Workgroup size 256 = one thread per element in a Q4_K block (256 elements/block).
// Uses subgroup (SIMD-group) reduction to minimize shared memory and barriers.

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

// Only 8 slots needed (one per subgroup) instead of 256
var<workgroup> sg_partial: array<f32, 8>;

// Q4_K block layout (144 bytes = 36 words):
//   word 0:       d(f16) | dmin(f16)
//   words 1-3:    scales[12 bytes] — 6-bit packed scales and mins
//   words 4-35:   qs[128 bytes] — 4-bit quantized values (2 per byte)

const BLOCK_WORDS: u32 = 36u;

fn extract_byte(word: u32, pos: u32) -> u32 {
    return (word >> (pos * 8u)) & 0xFFu;
}

// Extract 6-bit scale and min for sub-block j (0..7) from 12-byte scales array.
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
fn matvec_q4k(
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

        let q = (weights[bw + 4u + qs_rel] >> nibble_shift) & 0xFu;

        let input_idx = blk * 256u + tid;
        acc = fma(fma(d_sc, f32(q), -bias), input_vec[input_idx], acc);
    }

    // --- Subgroup reduction: 8 barriers → 1 barrier ---
    // Step 1: Reduce within each subgroup (register-to-register, zero shared mem)
    let sg_sum = subgroupAdd(acc);

    // Step 2: Subgroup leader writes partial sum to shared memory
    let sg_id = tid / sg_size;
    if (sg_lane == 0u) {
        sg_partial[sg_id] = sg_sum;
    }
    workgroupBarrier();

    // Step 3: Thread 0 reduces 8 partial sums
    if (tid == 0u) {
        var total: f32 = 0.0;
        let n_sg = 256u / sg_size;
        for (var s = 0u; s < n_sg; s = s + 1u) {
            total += sg_partial[s];
        }
        output_vec[row] = total;
    }
}
