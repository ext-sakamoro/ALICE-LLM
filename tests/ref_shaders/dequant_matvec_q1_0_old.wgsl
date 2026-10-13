// PrismML `Q1_0` (Bonsai 27B binary g128) dequantized matrix-vector multiply
// compute shader.
//
// Each workgroup computes one output row: output[row] = dot(dequant(W[row,:]), input).
// Workgroup size 128 = one thread per element in a Q1_0 block (128 elements/block).
// Uses subgroup (SIMD-group) reduction to minimize shared memory and barriers,
// mirroring the pattern already used in `dequant_matvec_q4k.wgsl`.
//
// Q1_0 block layout (18 bytes / 128 elements):
//   bytes 0..2   FP16 scale `d`
//   bytes 2..18  16 bytes of packed 1-bit values (LSB-first, 8 elements per byte)
//
// Value per element: `bit == 1 → +d`, `bit == 0 → −d`.
//
// Because 18 bytes is not a multiple of 4, blocks do not align to `u32` word
// boundaries — a block that starts on an even block index sits at byte offset
// `blk * 18` which alternates between `% 4 == 0` and `% 4 == 2`. We therefore
// use byte-level indexing (`byte_at`) rather than word-level indexing, so a
// single kernel handles either alignment.

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

// Only enough slots for the worst-case number of subgroups per workgroup
// (workgroup_size 128 / subgroup_size 8 = 16 slots; NVIDIA Ampere has 32-lane
// subgroups so 128/32 = 4 slots suffice, but keeping 16 makes the shader
// portable across Apple Silicon (32-lane) and older Intel iGPUs (8-lane).)
var<workgroup> sg_partial: array<f32, 16>;

const BLOCK_BYTES: u32 = 18u;

// Extract byte at absolute byte offset from the `u32` storage buffer.
// Assumes little-endian byte order in the buffer, which matches every
// backend wgpu currently ships (Metal, Vulkan, DX12).
fn byte_at(byte_offset: u32) -> u32 {
    let word = weights[byte_offset / 4u];
    let shift = (byte_offset % 4u) * 8u;
    return (word >> shift) & 0xFFu;
}

// Read an FP16 value at the given byte offset and widen to f32.
// `unpack2x16float` interprets a `u32` as [low16 : f16, high16 : f16]; we
// build the low 16 bits from the two little-endian scale bytes and take `.x`.
fn read_fp16(byte_offset: u32) -> f32 {
    let lo = byte_at(byte_offset);
    let hi = byte_at(byte_offset + 1u);
    let bits = lo | (hi << 8u);
    return unpack2x16float(bits).x;
}

@compute @workgroup_size(128)
fn matvec_q1_0(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let row = wid.y * params.grid_x + wid.x;
    if (row >= params.rows) { return; }

    let tid = lid.x;                         // 0..128 — one thread per element in a block
    let row_byte_offset = row * params.blocks_per_row * BLOCK_BYTES;

    // Per-thread constants derived from `tid`, hoisted outside the loop.
    let byte_idx_in_block = 2u + tid / 8u;   // 2..17: the 16 bit-packed bytes after the FP16 scale
    let bit_shift = tid & 7u;                // 0..7

    var acc: f32 = 0.0;

    for (var blk = 0u; blk < params.blocks_per_row; blk = blk + 1u) {
        let block_byte_offset = row_byte_offset + blk * BLOCK_BYTES;

        let d = read_fp16(block_byte_offset);
        let b = byte_at(block_byte_offset + byte_idx_in_block);
        let bit = (b >> bit_shift) & 1u;

        // `bit == 1 → +d`, `bit == 0 → −d`.
        // Written as `2*d*bit − d` to fold into a single FMA-friendly form.
        let val = fma(2.0 * d, f32(bit), -d);

        let input_idx = blk * 128u + tid;
        acc = fma(val, input_vec[input_idx], acc);
    }

    // Subgroup reduction: register-to-register within each SIMD group,
    // then one barrier + workgroup fanout for the final sum. Matches the
    // pattern used by `dequant_matvec_q4k.wgsl` / `dequant_matvec_q6k.wgsl`.
    let sg_sum = subgroupAdd(acc);
    let sg_id = tid / sg_size;
    if (sg_lane == 0u) {
        sg_partial[sg_id] = sg_sum;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var total: f32 = 0.0;
        let n_sg = 128u / sg_size;
        for (var s = 0u; s < n_sg; s = s + 1u) {
            total += sg_partial[s];
        }
        output_vec[row] = total;
    }
}
