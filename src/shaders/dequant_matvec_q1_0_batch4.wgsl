// Q1_0 batch-4 matvec: weight decoded ONCE, 4 scalar FMAs in registers.
// Mirrors `dequant_matvec_q4k_batch4.wgsl` for Bonsai 27B / PrismML Q1_0
// (128 elements/block, 18 bytes/block, binary g128).
//
// For batch decode this halves the total wgpu dispatch count: the same weight
// row is combined against 4 different input vectors (4 tokens' worth of
// activations) in a single kernel launch, keeping the weight decode work
// off the ALU/memory-bandwidth critical path.

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

// One partial-sum slot per subgroup (max 16 subgroups for workgroup_size 128).
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
fn matvec_q1_0_batch4(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let row = wid.y * params.grid_x + wid.x;
    if (row >= params.rows) { return; }

    let tid = lid.x;                            // 0..128, one thread per element
    let row_byte_offset = row * params.blocks_per_row * BLOCK_BYTES;
    let input_stride = params.blocks_per_row * 128u;

    // Loop-invariant per-thread constants.
    let byte_idx_in_block = 2u + tid / 8u;
    let bit_shift = tid & 7u;

    // 4 scalar accumulators — guaranteed in physical registers, no dynamic
    // indexing prevents them from spilling to memory.
    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    for (var blk = 0u; blk < params.blocks_per_row; blk = blk + 1u) {
        let block_byte_offset = row_byte_offset + blk * BLOCK_BYTES;

        // Weight decode — done ONCE, reused across 4 batch elements.
        let d = read_fp16(block_byte_offset);
        let b = byte_at(block_byte_offset + byte_idx_in_block);
        let bit = (b >> bit_shift) & 1u;
        let w = fma(2.0 * d, f32(bit), -d);   // = (2 * bit − 1) * d = ±d

        // 4 input reads + 4 FMAs — pure ALU, no weight-decode duplication.
        let base_idx = blk * 128u + tid;
        acc0 = fma(w, input_vec[base_idx], acc0);
        acc1 = fma(w, input_vec[input_stride + base_idx], acc1);
        acc2 = fma(w, input_vec[2u * input_stride + base_idx], acc2);
        acc3 = fma(w, input_vec[3u * input_stride + base_idx], acc3);
    }

    // 4 subgroup reductions (register-to-register SIMD shuffles).
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
        let rows = params.rows;
        output_vec[row] = t0;
        output_vec[rows + row] = t1;
        output_vec[2u * rows + row] = t2;
        output_vec[3u * rows + row] = t3;
    }
}
