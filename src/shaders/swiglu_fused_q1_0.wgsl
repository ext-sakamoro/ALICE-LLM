// Fused SwiGLU Q1_0 kernel: output[row] = silu(gate · input) × up · input.
// Each workgroup computes one output row from two Q1_0 weight matrices.
//
// Q1_0 layout (matches `dequant_matvec_q1_0_row4.wgsl` and `gguf.rs`
// `q1_0_matvec_preq`):
//   - 128 elements per block
//   - 18 bytes per block: 2 bytes f16 scale `d`, 16 bytes packed 1-bit
//     weights (128 bits, one per element)
//   - value at position `i` = `+d` if bit_i == 1 else `-d`
//     (compact ternary sign encoding around zero)
//
// Blocks straddle `u32` word boundaries, so all reads go through
// `byte_at()` helper (byte-level indexing into the storage buffer).
//
// Workgroup layout mirrors the standard Q1_0 matvec (workgroup_size = 128,
// one thread per element in a block). Two accumulators per thread —
// `acc_gate` and `acc_up` — running the two matvecs simultaneously.
// Subgroup reduction produces the row-level gate and up sums, then
// thread 0 applies `silu` to the gate sum and writes the fused product.

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

// One partial-sum slot per subgroup (max 16 subgroups for workgroup_size 128).
var<workgroup> sg_gate: array<f32, 16>;
var<workgroup> sg_up: array<f32, 16>;

const BLOCK_BYTES: u32 = 18u;

fn gate_byte_at(byte_offset: u32) -> u32 {
    let word = gate_weights[byte_offset / 4u];
    let shift = (byte_offset % 4u) * 8u;
    return (word >> shift) & 0xFFu;
}

fn up_byte_at(byte_offset: u32) -> u32 {
    let word = up_weights[byte_offset / 4u];
    let shift = (byte_offset % 4u) * 8u;
    return (word >> shift) & 0xFFu;
}

fn gate_read_fp16(byte_offset: u32) -> f32 {
    let lo = gate_byte_at(byte_offset);
    let hi = gate_byte_at(byte_offset + 1u);
    let bits = lo | (hi << 8u);
    return unpack2x16float(bits).x;
}

fn up_read_fp16(byte_offset: u32) -> f32 {
    let lo = up_byte_at(byte_offset);
    let hi = up_byte_at(byte_offset + 1u);
    let bits = lo | (hi << 8u);
    return unpack2x16float(bits).x;
}

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

@compute @workgroup_size(128)
fn swiglu_fused_q1_0(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let row = wid.y * params.grid_x + wid.x;
    if (row >= params.rows) { return; }

    let tid = lid.x;
    // Which byte within the 16-byte packed qs and which bit inside it
    // the current thread reads. Matches the base Q1_0 kernel.
    let byte_idx_in_block = 2u + tid / 8u;
    let bit_shift = tid & 7u;

    let row_bytes = params.blocks_per_row * BLOCK_BYTES;
    let row_off = row * row_bytes;

    var acc_gate: f32 = 0.0;
    var acc_up: f32 = 0.0;

    for (var blk = 0u; blk < params.blocks_per_row; blk = blk + 1u) {
        let in_val = input_vec[blk * 128u + tid];
        let blk_off = row_off + blk * BLOCK_BYTES;

        let d_gate = gate_read_fp16(blk_off);
        let bg = gate_byte_at(blk_off + byte_idx_in_block);
        let bit_g = (bg >> bit_shift) & 1u;
        let w_gate = fma(2.0 * d_gate, f32(bit_g), -d_gate);
        acc_gate = fma(w_gate, in_val, acc_gate);

        let d_up = up_read_fp16(blk_off);
        let bu = up_byte_at(blk_off + byte_idx_in_block);
        let bit_u = (bu >> bit_shift) & 1u;
        let w_up = fma(2.0 * d_up, f32(bit_u), -d_up);
        acc_up = fma(w_up, in_val, acc_up);
    }

    let sg_gate_sum = subgroupAdd(acc_gate);
    let sg_up_sum = subgroupAdd(acc_up);

    let sg_id = tid / sg_size;
    if (sg_lane == 0u) {
        sg_gate[sg_id] = sg_gate_sum;
        sg_up[sg_id] = sg_up_sum;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var g: f32 = 0.0;
        var u: f32 = 0.0;
        let n_sg = 128u / sg_size;
        for (var s = 0u; s < n_sg; s = s + 1u) {
            g += sg_gate[s];
            u += sg_up[s];
        }
        output_vec[row] = silu(g) * u;
    }
}
