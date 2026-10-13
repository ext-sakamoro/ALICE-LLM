// Q1_0 row8×batch4 matvec: 1 workgroup produces 8 rows × 4 batches = 32 outputs.
//
// Compounds Step 4's row4×batch4 win by doubling the row batching factor.
// Workgroup count drops another 2× (2560 → 1280 on Bonsai attn_qkv 10240 rows),
// and total input-read traffic drops another 2× (input is now shared across
// 8 output rows within a workgroup instead of 4).
//
// Cost side:
//   - 32 f32 accumulators per thread (up from 16 in row4×batch4). This is
//     the primary concern — Ampere iGPU / Metal Apple GPU / Vulkan MoltenVK
//     typically support 32-64 registers per thread without spilling to
//     local memory. The scalar `varN...` variable form (no dynamic
//     indexing) ensures the compiler keeps every accumulator in a
//     physical register.
//   - Per-block work per thread: 4 shared input loads + 8 weight decodes
//     + 32 FMAs (was 4 + 4 + 16 in row4×batch4). This is more arithmetic
//     per thread → higher ALU utilization but same total work per
//     dispatch (workgroup count halved).
//   - Shared memory: 32 subgroup partial-sum arrays × 16 slots each = 2 KB
//     smem per workgroup (was 1 KB in row4×batch4). Still comfortably under
//     the 32 KB / 48 KB per-workgroup limits.

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

// Single consolidated threadgroup array — 32 accumulator slots (8 rows ×
// 4 batches) × 16 subgroups = 512 f32 = 2 KB smem. Indexed as
// `sg_all[(row * 4 + batch) * 16 + sg_id]`.
//
// Metal has a per-shader hard limit on the *number* of distinct
// `threadgroup` resources (each `var<workgroup>` becomes one Metal
// threadgroup slot). 32 separate arrays hit that limit ("no 'threadgroup'
// resource location available"), so we consolidate into one flat array.
// WGSL naga transpiles this into a single Metal `threadgroup type_4&`,
// staying under the resource-slot budget while preserving the same
// per-accumulator 16 subgroup partial-sum layout.
var<workgroup> sg_all: array<f32, 512>;

// Convenience: base index of accumulator `(row, batch)` in `sg_all`.
fn sg_base(row: u32, batch: u32) -> u32 {
    return (row * 4u + batch) * 16u;
}

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
fn matvec_q1_0_row8_batch4(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let row_group = wid.y * params.grid_x + wid.x;
    let row_base = row_group * 8u;
    if (row_base >= params.rows) { return; }

    let tid = lid.x;
    let byte_idx_in_block = 2u + tid / 8u;
    let bit_shift = tid & 7u;

    let row_bytes = params.blocks_per_row * BLOCK_BYTES;
    let input_stride = params.blocks_per_row * 128u;

    // Row byte-offsets. Rows past `params.rows` are guarded at output write
    // time. For non-multiples of 8, clamp trailing rows to row_base so we
    // don't read past the tensor buffer.
    let last = params.rows - 1u;
    let row0 = row_base;
    let row1 = min(row_base + 1u, last);
    let row2 = min(row_base + 2u, last);
    let row3 = min(row_base + 3u, last);
    let row4 = min(row_base + 4u, last);
    let row5 = min(row_base + 5u, last);
    let row6 = min(row_base + 6u, last);
    let row7 = min(row_base + 7u, last);
    let row0_off = row0 * row_bytes;
    let row1_off = row1 * row_bytes;
    let row2_off = row2 * row_bytes;
    let row3_off = row3 * row_bytes;
    let row4_off = row4 * row_bytes;
    let row5_off = row5 * row_bytes;
    let row6_off = row6 * row_bytes;
    let row7_off = row7 * row_bytes;

    // 32 scalar accumulators = 8 rows × 4 batches. Each is a distinct named
    // variable (no dynamic indexing) so the compiler keeps them in
    // physical registers instead of spilling to local memory.
    var acc00: f32 = 0.0; var acc01: f32 = 0.0; var acc02: f32 = 0.0; var acc03: f32 = 0.0;
    var acc10: f32 = 0.0; var acc11: f32 = 0.0; var acc12: f32 = 0.0; var acc13: f32 = 0.0;
    var acc20: f32 = 0.0; var acc21: f32 = 0.0; var acc22: f32 = 0.0; var acc23: f32 = 0.0;
    var acc30: f32 = 0.0; var acc31: f32 = 0.0; var acc32: f32 = 0.0; var acc33: f32 = 0.0;
    var acc40: f32 = 0.0; var acc41: f32 = 0.0; var acc42: f32 = 0.0; var acc43: f32 = 0.0;
    var acc50: f32 = 0.0; var acc51: f32 = 0.0; var acc52: f32 = 0.0; var acc53: f32 = 0.0;
    var acc60: f32 = 0.0; var acc61: f32 = 0.0; var acc62: f32 = 0.0; var acc63: f32 = 0.0;
    var acc70: f32 = 0.0; var acc71: f32 = 0.0; var acc72: f32 = 0.0; var acc73: f32 = 0.0;

    for (var blk = 0u; blk < params.blocks_per_row; blk = blk + 1u) {
        // Shared input reads — 4 batches, shared across all 8 output rows.
        let base_idx = blk * 128u + tid;
        let in0 = input_vec[base_idx];
        let in1 = input_vec[input_stride + base_idx];
        let in2 = input_vec[2u * input_stride + base_idx];
        let in3 = input_vec[3u * input_stride + base_idx];

        let blk_off = blk * BLOCK_BYTES;

        // Row 0
        let off0 = row0_off + blk_off;
        let d0 = read_fp16(off0);
        let b0 = byte_at(off0 + byte_idx_in_block);
        let bit0 = (b0 >> bit_shift) & 1u;
        let w0 = fma(2.0 * d0, f32(bit0), -d0);
        acc00 = fma(w0, in0, acc00); acc01 = fma(w0, in1, acc01); acc02 = fma(w0, in2, acc02); acc03 = fma(w0, in3, acc03);

        // Row 1
        let off1 = row1_off + blk_off;
        let d1 = read_fp16(off1);
        let b1 = byte_at(off1 + byte_idx_in_block);
        let bit1 = (b1 >> bit_shift) & 1u;
        let w1 = fma(2.0 * d1, f32(bit1), -d1);
        acc10 = fma(w1, in0, acc10); acc11 = fma(w1, in1, acc11); acc12 = fma(w1, in2, acc12); acc13 = fma(w1, in3, acc13);

        // Row 2
        let off2 = row2_off + blk_off;
        let d2 = read_fp16(off2);
        let b2 = byte_at(off2 + byte_idx_in_block);
        let bit2 = (b2 >> bit_shift) & 1u;
        let w2 = fma(2.0 * d2, f32(bit2), -d2);
        acc20 = fma(w2, in0, acc20); acc21 = fma(w2, in1, acc21); acc22 = fma(w2, in2, acc22); acc23 = fma(w2, in3, acc23);

        // Row 3
        let off3 = row3_off + blk_off;
        let d3 = read_fp16(off3);
        let b3 = byte_at(off3 + byte_idx_in_block);
        let bit3 = (b3 >> bit_shift) & 1u;
        let w3 = fma(2.0 * d3, f32(bit3), -d3);
        acc30 = fma(w3, in0, acc30); acc31 = fma(w3, in1, acc31); acc32 = fma(w3, in2, acc32); acc33 = fma(w3, in3, acc33);

        // Row 4
        let off4 = row4_off + blk_off;
        let d4 = read_fp16(off4);
        let b4 = byte_at(off4 + byte_idx_in_block);
        let bit4 = (b4 >> bit_shift) & 1u;
        let w4 = fma(2.0 * d4, f32(bit4), -d4);
        acc40 = fma(w4, in0, acc40); acc41 = fma(w4, in1, acc41); acc42 = fma(w4, in2, acc42); acc43 = fma(w4, in3, acc43);

        // Row 5
        let off5 = row5_off + blk_off;
        let d5 = read_fp16(off5);
        let b5 = byte_at(off5 + byte_idx_in_block);
        let bit5 = (b5 >> bit_shift) & 1u;
        let w5 = fma(2.0 * d5, f32(bit5), -d5);
        acc50 = fma(w5, in0, acc50); acc51 = fma(w5, in1, acc51); acc52 = fma(w5, in2, acc52); acc53 = fma(w5, in3, acc53);

        // Row 6
        let off6 = row6_off + blk_off;
        let d6 = read_fp16(off6);
        let b6 = byte_at(off6 + byte_idx_in_block);
        let bit6 = (b6 >> bit_shift) & 1u;
        let w6 = fma(2.0 * d6, f32(bit6), -d6);
        acc60 = fma(w6, in0, acc60); acc61 = fma(w6, in1, acc61); acc62 = fma(w6, in2, acc62); acc63 = fma(w6, in3, acc63);

        // Row 7
        let off7 = row7_off + blk_off;
        let d7 = read_fp16(off7);
        let b7 = byte_at(off7 + byte_idx_in_block);
        let bit7 = (b7 >> bit_shift) & 1u;
        let w7 = fma(2.0 * d7, f32(bit7), -d7);
        acc70 = fma(w7, in0, acc70); acc71 = fma(w7, in1, acc71); acc72 = fma(w7, in2, acc72); acc73 = fma(w7, in3, acc73);
    }

    // 32 subgroup reductions.
    let s00 = subgroupAdd(acc00); let s01 = subgroupAdd(acc01); let s02 = subgroupAdd(acc02); let s03 = subgroupAdd(acc03);
    let s10 = subgroupAdd(acc10); let s11 = subgroupAdd(acc11); let s12 = subgroupAdd(acc12); let s13 = subgroupAdd(acc13);
    let s20 = subgroupAdd(acc20); let s21 = subgroupAdd(acc21); let s22 = subgroupAdd(acc22); let s23 = subgroupAdd(acc23);
    let s30 = subgroupAdd(acc30); let s31 = subgroupAdd(acc31); let s32 = subgroupAdd(acc32); let s33 = subgroupAdd(acc33);
    let s40 = subgroupAdd(acc40); let s41 = subgroupAdd(acc41); let s42 = subgroupAdd(acc42); let s43 = subgroupAdd(acc43);
    let s50 = subgroupAdd(acc50); let s51 = subgroupAdd(acc51); let s52 = subgroupAdd(acc52); let s53 = subgroupAdd(acc53);
    let s60 = subgroupAdd(acc60); let s61 = subgroupAdd(acc61); let s62 = subgroupAdd(acc62); let s63 = subgroupAdd(acc63);
    let s70 = subgroupAdd(acc70); let s71 = subgroupAdd(acc71); let s72 = subgroupAdd(acc72); let s73 = subgroupAdd(acc73);

    let sg_id = tid / sg_size;
    if (sg_lane == 0u) {
        sg_all[sg_base(0u, 0u) + sg_id] = s00; sg_all[sg_base(0u, 1u) + sg_id] = s01;
        sg_all[sg_base(0u, 2u) + sg_id] = s02; sg_all[sg_base(0u, 3u) + sg_id] = s03;
        sg_all[sg_base(1u, 0u) + sg_id] = s10; sg_all[sg_base(1u, 1u) + sg_id] = s11;
        sg_all[sg_base(1u, 2u) + sg_id] = s12; sg_all[sg_base(1u, 3u) + sg_id] = s13;
        sg_all[sg_base(2u, 0u) + sg_id] = s20; sg_all[sg_base(2u, 1u) + sg_id] = s21;
        sg_all[sg_base(2u, 2u) + sg_id] = s22; sg_all[sg_base(2u, 3u) + sg_id] = s23;
        sg_all[sg_base(3u, 0u) + sg_id] = s30; sg_all[sg_base(3u, 1u) + sg_id] = s31;
        sg_all[sg_base(3u, 2u) + sg_id] = s32; sg_all[sg_base(3u, 3u) + sg_id] = s33;
        sg_all[sg_base(4u, 0u) + sg_id] = s40; sg_all[sg_base(4u, 1u) + sg_id] = s41;
        sg_all[sg_base(4u, 2u) + sg_id] = s42; sg_all[sg_base(4u, 3u) + sg_id] = s43;
        sg_all[sg_base(5u, 0u) + sg_id] = s50; sg_all[sg_base(5u, 1u) + sg_id] = s51;
        sg_all[sg_base(5u, 2u) + sg_id] = s52; sg_all[sg_base(5u, 3u) + sg_id] = s53;
        sg_all[sg_base(6u, 0u) + sg_id] = s60; sg_all[sg_base(6u, 1u) + sg_id] = s61;
        sg_all[sg_base(6u, 2u) + sg_id] = s62; sg_all[sg_base(6u, 3u) + sg_id] = s63;
        sg_all[sg_base(7u, 0u) + sg_id] = s70; sg_all[sg_base(7u, 1u) + sg_id] = s71;
        sg_all[sg_base(7u, 2u) + sg_id] = s72; sg_all[sg_base(7u, 3u) + sg_id] = s73;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var t00: f32 = 0.0; var t01: f32 = 0.0; var t02: f32 = 0.0; var t03: f32 = 0.0;
        var t10: f32 = 0.0; var t11: f32 = 0.0; var t12: f32 = 0.0; var t13: f32 = 0.0;
        var t20: f32 = 0.0; var t21: f32 = 0.0; var t22: f32 = 0.0; var t23: f32 = 0.0;
        var t30: f32 = 0.0; var t31: f32 = 0.0; var t32: f32 = 0.0; var t33: f32 = 0.0;
        var t40: f32 = 0.0; var t41: f32 = 0.0; var t42: f32 = 0.0; var t43: f32 = 0.0;
        var t50: f32 = 0.0; var t51: f32 = 0.0; var t52: f32 = 0.0; var t53: f32 = 0.0;
        var t60: f32 = 0.0; var t61: f32 = 0.0; var t62: f32 = 0.0; var t63: f32 = 0.0;
        var t70: f32 = 0.0; var t71: f32 = 0.0; var t72: f32 = 0.0; var t73: f32 = 0.0;
        let n_sg = 128u / sg_size;
        for (var s = 0u; s < n_sg; s = s + 1u) {
            t00 += sg_all[sg_base(0u, 0u) + s]; t01 += sg_all[sg_base(0u, 1u) + s];
            t02 += sg_all[sg_base(0u, 2u) + s]; t03 += sg_all[sg_base(0u, 3u) + s];
            t10 += sg_all[sg_base(1u, 0u) + s]; t11 += sg_all[sg_base(1u, 1u) + s];
            t12 += sg_all[sg_base(1u, 2u) + s]; t13 += sg_all[sg_base(1u, 3u) + s];
            t20 += sg_all[sg_base(2u, 0u) + s]; t21 += sg_all[sg_base(2u, 1u) + s];
            t22 += sg_all[sg_base(2u, 2u) + s]; t23 += sg_all[sg_base(2u, 3u) + s];
            t30 += sg_all[sg_base(3u, 0u) + s]; t31 += sg_all[sg_base(3u, 1u) + s];
            t32 += sg_all[sg_base(3u, 2u) + s]; t33 += sg_all[sg_base(3u, 3u) + s];
            t40 += sg_all[sg_base(4u, 0u) + s]; t41 += sg_all[sg_base(4u, 1u) + s];
            t42 += sg_all[sg_base(4u, 2u) + s]; t43 += sg_all[sg_base(4u, 3u) + s];
            t50 += sg_all[sg_base(5u, 0u) + s]; t51 += sg_all[sg_base(5u, 1u) + s];
            t52 += sg_all[sg_base(5u, 2u) + s]; t53 += sg_all[sg_base(5u, 3u) + s];
            t60 += sg_all[sg_base(6u, 0u) + s]; t61 += sg_all[sg_base(6u, 1u) + s];
            t62 += sg_all[sg_base(6u, 2u) + s]; t63 += sg_all[sg_base(6u, 3u) + s];
            t70 += sg_all[sg_base(7u, 0u) + s]; t71 += sg_all[sg_base(7u, 1u) + s];
            t72 += sg_all[sg_base(7u, 2u) + s]; t73 += sg_all[sg_base(7u, 3u) + s];
        }

        // Output layout matches batch4 family: output_vec[batch * rows + row].
        // Row-guard: only write rows that are actually within the tensor.
        let rows = params.rows;
        let r0 = row_base;
        let r1 = row_base + 1u;
        let r2 = row_base + 2u;
        let r3 = row_base + 3u;
        let r4 = row_base + 4u;
        let r5 = row_base + 5u;
        let r6 = row_base + 6u;
        let r7 = row_base + 7u;

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
        if (r4 < rows) {
            output_vec[             r4] = t40;
            output_vec[     rows + r4] = t41;
            output_vec[2u * rows + r4] = t42;
            output_vec[3u * rows + r4] = t43;
        }
        if (r5 < rows) {
            output_vec[             r5] = t50;
            output_vec[     rows + r5] = t51;
            output_vec[2u * rows + r5] = t52;
            output_vec[3u * rows + r5] = t53;
        }
        if (r6 < rows) {
            output_vec[             r6] = t60;
            output_vec[     rows + r6] = t61;
            output_vec[2u * rows + r6] = t62;
            output_vec[3u * rows + r6] = t63;
        }
        if (r7 < rows) {
            output_vec[             r7] = t70;
            output_vec[     rows + r7] = t71;
            output_vec[2u * rows + r7] = t72;
            output_vec[3u * rows + r7] = t73;
        }
    }
}
