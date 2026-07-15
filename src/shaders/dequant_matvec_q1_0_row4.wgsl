// Q1_0 row4 matvec (single batch, 4 output rows per workgroup).
//
// Mirrors `dequant_matvec_q1_0_row4_batch4.wgsl` but for the standard
// batch=1 decode path where speculative decoding is not in use. Same
// 4× workgroup-count reduction (10240 rows → 2560 workgroups on Bonsai
// attn_qkv) and same 4× input-read amortization across rows, without the
// batch4 output-dimension expansion.
//
// This is the kernel used by the real Bonsai inference path in single-
// token decode mode. Step 4's row4×batch4 kernel requires a 4-token input
// buffer (speculative decoding); this single-batch variant is the natural
// companion for interactive prompt/decode.
//
// Layout: 4 scalar accumulators (one per output row) in registers, 4
// subgroup reduction slots, single thread-0 writes 4 outputs.

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

// One partial-sum slot per subgroup per output row (max 16 subgroups for
// workgroup_size 128).
var<workgroup> sg0: array<f32, 16>;
var<workgroup> sg1: array<f32, 16>;
var<workgroup> sg2: array<f32, 16>;
var<workgroup> sg3: array<f32, 16>;

const BLOCK_BYTES: u32 = 18u;

fn byte_at(byte_offset: u32) -> u32 {
    let word = weights[byte_offset / 4u];
    let shift = (byte_offset % 4u) * 8u;
    return (word >> shift) & 0xFFu;
}

fn read_fp16(byte_offset: u32) -> f32 {
    let lo = byte_at(byte_offset);
    let hi = byte_at(byte_offset + 1u);
    let bits = lo | (hi << 8u);
    return unpack2x16float(bits).x;
}

@compute @workgroup_size(128)
fn matvec_q1_0_row4(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let row_group = wid.y * params.grid_x + wid.x;
    let row_base = row_group * 4u;
    if (row_base >= params.rows) { return; }

    let tid = lid.x;
    let byte_idx_in_block = 2u + tid / 8u;
    let bit_shift = tid & 7u;

    let row_bytes = params.blocks_per_row * BLOCK_BYTES;

    let last = params.rows - 1u;
    let row0 = row_base;
    let row1 = min(row_base + 1u, last);
    let row2 = min(row_base + 2u, last);
    let row3 = min(row_base + 3u, last);
    let row0_off = row0 * row_bytes;
    let row1_off = row1 * row_bytes;
    let row2_off = row2 * row_bytes;
    let row3_off = row3 * row_bytes;

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    for (var blk = 0u; blk < params.blocks_per_row; blk = blk + 1u) {
        // Single shared input read — reused across all 4 output rows.
        let in_val = input_vec[blk * 128u + tid];

        let blk_off = blk * BLOCK_BYTES;

        let off0 = row0_off + blk_off;
        let d0 = read_fp16(off0);
        let b0 = byte_at(off0 + byte_idx_in_block);
        let bit0 = (b0 >> bit_shift) & 1u;
        let w0 = fma(2.0 * d0, f32(bit0), -d0);
        acc0 = fma(w0, in_val, acc0);

        let off1 = row1_off + blk_off;
        let d1 = read_fp16(off1);
        let b1 = byte_at(off1 + byte_idx_in_block);
        let bit1 = (b1 >> bit_shift) & 1u;
        let w1 = fma(2.0 * d1, f32(bit1), -d1);
        acc1 = fma(w1, in_val, acc1);

        let off2 = row2_off + blk_off;
        let d2 = read_fp16(off2);
        let b2 = byte_at(off2 + byte_idx_in_block);
        let bit2 = (b2 >> bit_shift) & 1u;
        let w2 = fma(2.0 * d2, f32(bit2), -d2);
        acc2 = fma(w2, in_val, acc2);

        let off3 = row3_off + blk_off;
        let d3 = read_fp16(off3);
        let b3 = byte_at(off3 + byte_idx_in_block);
        let bit3 = (b3 >> bit_shift) & 1u;
        let w3 = fma(2.0 * d3, f32(bit3), -d3);
        acc3 = fma(w3, in_val, acc3);
    }

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
        let n_sg = 128u / sg_size;
        for (var s = 0u; s < n_sg; s = s + 1u) {
            t0 += sg0[s];
            t1 += sg1[s];
            t2 += sg2[s];
            t3 += sg3[s];
        }

        // Output: single batch, 4 output rows. Layout matches `matvec_q1_0`
        // (output_vec[row]), just 4 outputs from this workgroup.
        let rows = params.rows;
        let r0 = row_base;
        let r1 = row_base + 1u;
        let r2 = row_base + 2u;
        let r3 = row_base + 3u;

        if (r0 < rows) { output_vec[r0] = t0; }
        if (r1 < rows) { output_vec[r1] = t1; }
        if (r2 < rows) { output_vec[r2] = t2; }
        if (r3 < rows) { output_vec[r3] = t3; }
    }
}
