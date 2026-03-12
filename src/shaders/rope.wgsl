// RoPE (Rotary Position Embedding) — in-place rotation (batch-capable).
// Each thread handles one (cos, sin) pair across all heads.
// wid.y = batch index. Token k uses position = base_position + k.

struct Params {
    position: u32,
    num_heads: u32,
    head_dim: u32,
    theta: f32,
}

@group(0) @binding(0) var<storage, read_write> vec_data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn rope(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let batch_idx = wid.y;
    let pair_id = gid.x;
    let half_dim = params.head_dim / 2u;
    let total_pairs = params.num_heads * half_dim;
    if (pair_id >= total_pairs) { return; }

    let head = pair_id / half_dim;
    let pair_in_head = pair_id % half_dim;
    let dim_idx = pair_in_head * 2u;

    // Offset data by batch index (K vectors concatenated)
    let data_offset = batch_idx * params.num_heads * params.head_dim;
    let buf_idx = data_offset + head * params.head_dim + dim_idx;

    // Position incremented per token in batch
    let position = params.position + batch_idx;
    let freq = 1.0 / pow(params.theta, f32(dim_idx) / f32(params.head_dim));
    let angle = f32(position) * freq;
    let s = sin(angle);
    let c = cos(angle);

    let x0 = vec_data[buf_idx];
    let x1 = vec_data[buf_idx + 1u];
    vec_data[buf_idx] = x0 * c - x1 * s;
    vec_data[buf_idx + 1u] = x0 * s + x1 * c;
}
