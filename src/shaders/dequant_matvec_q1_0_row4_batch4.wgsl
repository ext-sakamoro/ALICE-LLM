// Q1_0 row4×batch4 matvec: 1 workgroup produces 4 rows × 4 batches = 16 outputs.
//
// Extends the Step 2 batch4 kernel by ALSO batching 4 output rows per
// workgroup. This yields two compounding wins over batch4 alone:
//   1. Input reads shared across 4 rows: instead of every workgroup reading
//      the same input tensor once, only every 4th workgroup does. Total
//      input read volume across all workgroups drops 4× → less L2 pressure.
//   2. Workgroup count drops 4× (10240 rows → 2560 workgroups on Bonsai
//      attn_qkv), which amortizes any per-workgroup overhead (compute
//      pipeline setup, subgroup init, workgroup barriers).
//
// Weight decode work per workgroup grows 4× (one row-decode → four
// row-decodes per block), but the *total* weight decode work across the
// dispatch stays constant. In exchange for that extra register / decode
// activity per workgroup, we buy 4× fewer input reads (the dominant memory
// traffic bottleneck at 11% bandwidth utilization on Jetson Vulkan).
//
// Layout: 16 scalar accumulators (4 rows × 4 batches) in registers, 16
// subgroup reduction slots, single thread-0 writes 16 outputs.

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

// 16 partial-sum slots per subgroup for the 16 (row × batch) accumulators.
// workgroup_size 128 / subgroup_size 8 = 16 subgroups worst case, so 16 slots
// per accumulator = 16 × 16 = 256 f32 = 1024 bytes shared memory. Trivially
// under the 32 KB / 48 KB per-workgroup shared-memory limits on Metal and
// Vulkan integrated GPUs.
var<workgroup> sg00: array<f32, 16>;
var<workgroup> sg01: array<f32, 16>;
var<workgroup> sg02: array<f32, 16>;
var<workgroup> sg03: array<f32, 16>;
var<workgroup> sg10: array<f32, 16>;
var<workgroup> sg11: array<f32, 16>;
var<workgroup> sg12: array<f32, 16>;
var<workgroup> sg13: array<f32, 16>;
var<workgroup> sg20: array<f32, 16>;
var<workgroup> sg21: array<f32, 16>;
var<workgroup> sg22: array<f32, 16>;
var<workgroup> sg23: array<f32, 16>;
var<workgroup> sg30: array<f32, 16>;
var<workgroup> sg31: array<f32, 16>;
var<workgroup> sg32: array<f32, 16>;
var<workgroup> sg33: array<f32, 16>;

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
fn matvec_q1_0_row4_batch4(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let row_group = wid.y * params.grid_x + wid.x;
    let row_base = row_group * 4u;
    // Early exit if this row-group is entirely past the row count. Individual
    // rows within a group are guarded during the output write step below.
    if (row_base >= params.rows) { return; }

    let tid = lid.x;                              // 0..128, one thread per element
    let byte_idx_in_block = 2u + tid / 8u;
    let bit_shift = tid & 7u;

    let row_bytes = params.blocks_per_row * BLOCK_BYTES;
    let input_stride = params.blocks_per_row * 128u;

    // Row byte-offsets. Rows past `params.rows` are guarded at output write
    // time — reading beyond the tensor into cargo-allocated storage bytes is
    // safe (storage buffer is sized to `rows × row_bytes`; extra rows would
    // require the launcher to over-allocate. In practice Bonsai's attn_qkv
    // has 10240 rows = divisible by 4, so `row_base + 3 < rows` always holds
    // for that tensor. For non-multiples of 4 we still avoid reading past
    // the buffer by clamping.
    let row0 = row_base;
    let row1 = min(row_base + 1u, params.rows - 1u);
    let row2 = min(row_base + 2u, params.rows - 1u);
    let row3 = min(row_base + 3u, params.rows - 1u);
    let row0_off = row0 * row_bytes;
    let row1_off = row1 * row_bytes;
    let row2_off = row2 * row_bytes;
    let row3_off = row3 * row_bytes;

    // 16 scalar accumulators = 4 rows × 4 batches. Guaranteed in physical
    // registers (no dynamic indexing on any of them), which prevents the
    // compiler from spilling them to local memory.
    var acc00: f32 = 0.0; var acc01: f32 = 0.0; var acc02: f32 = 0.0; var acc03: f32 = 0.0;
    var acc10: f32 = 0.0; var acc11: f32 = 0.0; var acc12: f32 = 0.0; var acc13: f32 = 0.0;
    var acc20: f32 = 0.0; var acc21: f32 = 0.0; var acc22: f32 = 0.0; var acc23: f32 = 0.0;
    var acc30: f32 = 0.0; var acc31: f32 = 0.0; var acc32: f32 = 0.0; var acc33: f32 = 0.0;

    for (var blk = 0u; blk < params.blocks_per_row; blk = blk + 1u) {
        // Shared input reads — 4 batches, but each read serves 4 rows.
        let base_idx = blk * 128u + tid;
        let in0 = input_vec[base_idx];
        let in1 = input_vec[input_stride + base_idx];
        let in2 = input_vec[2u * input_stride + base_idx];
        let in3 = input_vec[3u * input_stride + base_idx];

        let blk_off = blk * BLOCK_BYTES;

        // Row 0 weight decode
        let off0 = row0_off + blk_off;
        let d0 = read_fp16(off0);
        let b0 = byte_at(off0 + byte_idx_in_block);
        let bit0 = (b0 >> bit_shift) & 1u;
        let w0 = fma(2.0 * d0, f32(bit0), -d0);
        acc00 = fma(w0, in0, acc00);
        acc01 = fma(w0, in1, acc01);
        acc02 = fma(w0, in2, acc02);
        acc03 = fma(w0, in3, acc03);

        // Row 1 weight decode
        let off1 = row1_off + blk_off;
        let d1 = read_fp16(off1);
        let b1 = byte_at(off1 + byte_idx_in_block);
        let bit1 = (b1 >> bit_shift) & 1u;
        let w1 = fma(2.0 * d1, f32(bit1), -d1);
        acc10 = fma(w1, in0, acc10);
        acc11 = fma(w1, in1, acc11);
        acc12 = fma(w1, in2, acc12);
        acc13 = fma(w1, in3, acc13);

        // Row 2 weight decode
        let off2 = row2_off + blk_off;
        let d2 = read_fp16(off2);
        let b2 = byte_at(off2 + byte_idx_in_block);
        let bit2 = (b2 >> bit_shift) & 1u;
        let w2 = fma(2.0 * d2, f32(bit2), -d2);
        acc20 = fma(w2, in0, acc20);
        acc21 = fma(w2, in1, acc21);
        acc22 = fma(w2, in2, acc22);
        acc23 = fma(w2, in3, acc23);

        // Row 3 weight decode
        let off3 = row3_off + blk_off;
        let d3 = read_fp16(off3);
        let b3 = byte_at(off3 + byte_idx_in_block);
        let bit3 = (b3 >> bit_shift) & 1u;
        let w3 = fma(2.0 * d3, f32(bit3), -d3);
        acc30 = fma(w3, in0, acc30);
        acc31 = fma(w3, in1, acc31);
        acc32 = fma(w3, in2, acc32);
        acc33 = fma(w3, in3, acc33);
    }

    // 16 subgroup reductions (SIMD shuffles, register-to-register).
    let s00 = subgroupAdd(acc00); let s01 = subgroupAdd(acc01); let s02 = subgroupAdd(acc02); let s03 = subgroupAdd(acc03);
    let s10 = subgroupAdd(acc10); let s11 = subgroupAdd(acc11); let s12 = subgroupAdd(acc12); let s13 = subgroupAdd(acc13);
    let s20 = subgroupAdd(acc20); let s21 = subgroupAdd(acc21); let s22 = subgroupAdd(acc22); let s23 = subgroupAdd(acc23);
    let s30 = subgroupAdd(acc30); let s31 = subgroupAdd(acc31); let s32 = subgroupAdd(acc32); let s33 = subgroupAdd(acc33);

    let sg_id = tid / sg_size;
    if (sg_lane == 0u) {
        sg00[sg_id] = s00; sg01[sg_id] = s01; sg02[sg_id] = s02; sg03[sg_id] = s03;
        sg10[sg_id] = s10; sg11[sg_id] = s11; sg12[sg_id] = s12; sg13[sg_id] = s13;
        sg20[sg_id] = s20; sg21[sg_id] = s21; sg22[sg_id] = s22; sg23[sg_id] = s23;
        sg30[sg_id] = s30; sg31[sg_id] = s31; sg32[sg_id] = s32; sg33[sg_id] = s33;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var t00: f32 = 0.0; var t01: f32 = 0.0; var t02: f32 = 0.0; var t03: f32 = 0.0;
        var t10: f32 = 0.0; var t11: f32 = 0.0; var t12: f32 = 0.0; var t13: f32 = 0.0;
        var t20: f32 = 0.0; var t21: f32 = 0.0; var t22: f32 = 0.0; var t23: f32 = 0.0;
        var t30: f32 = 0.0; var t31: f32 = 0.0; var t32: f32 = 0.0; var t33: f32 = 0.0;
        let n_sg = 128u / sg_size;
        for (var s = 0u; s < n_sg; s = s + 1u) {
            t00 += sg00[s]; t01 += sg01[s]; t02 += sg02[s]; t03 += sg03[s];
            t10 += sg10[s]; t11 += sg11[s]; t12 += sg12[s]; t13 += sg13[s];
            t20 += sg20[s]; t21 += sg21[s]; t22 += sg22[s]; t23 += sg23[s];
            t30 += sg30[s]; t31 += sg31[s]; t32 += sg32[s]; t33 += sg33[s];
        }

        // Output layout matches batch4: output_vec[batch * rows + row].
        // Row-guard: only write rows that are actually within the tensor.
        let rows = params.rows;
        let r0 = row_base;
        let r1 = row_base + 1u;
        let r2 = row_base + 2u;
        let r3 = row_base + 3u;

        if (r0 < rows) {
            output_vec[             r0] = t00;
            output_vec[     rows + r0] = t01;
            output_vec[2u * rows + r0] = t02;
            output_vec[3u * rows + r0] = t03;
        }
        if (r1 < rows) {
            output_vec[             r1] = t10;
            output_vec[     rows + r1] = t11;
            output_vec[2u * rows + r1] = t12;
            output_vec[3u * rows + r1] = t13;
        }
        if (r2 < rows) {
            output_vec[             r2] = t20;
            output_vec[     rows + r2] = t21;
            output_vec[2u * rows + r2] = t22;
            output_vec[3u * rows + r2] = t23;
        }
        if (r3 < rows) {
            output_vec[             r3] = t30;
            output_vec[     rows + r3] = t31;
            output_vec[2u * rows + r3] = t32;
            output_vec[3u * rows + r3] = t33;
        }
    }
}
