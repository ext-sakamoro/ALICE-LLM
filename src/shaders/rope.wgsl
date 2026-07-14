// RoPE (Rotary Position Embedding) — in-place rotation (batch-capable).
// Supports both LLAMA-style (pair (i, i+1)) and NEOX-style (pair (i, i+half))
// rotation conventions. Selected via `params.neox`:
//   * 0 = LLAMA-style — used by Llama-1/2/3, Mistral, Gemma 4.
//   * 1 = NEOX-style  — used by Qwen 2/2.5/3/3.5, Gemma 2.
// Each thread handles one (cos, sin) pair across all heads.
// wid.y = batch index. Token k uses position = base_position + k.

struct Params {
    position: u32,
    num_heads: u32,
    head_dim: u32,
    theta: f32,
    batch_size: u32,
    neox: u32,
    _pad2: u32,
    _pad3: u32,
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

    // Offset data by batch index (K vectors concatenated)
    let data_offset = batch_idx * params.num_heads * params.head_dim;
    let head_base = data_offset + head * params.head_dim;

    // Position incremented per token in batch
    let position = params.position + batch_idx;

    // Frequency: for both conventions, freq[i] = 1 / theta^(2i / head_dim) where
    // i is the pair index within the head (0..half_dim). Equivalently for LLAMA
    // this is `theta^(dim_idx / head_dim)` with `dim_idx = 2 * pair_in_head`
    // (since dim_idx / head_dim == 2 * pair_in_head / head_dim). Same value.
    let freq = 1.0 / pow(params.theta, f32(pair_in_head * 2u) / f32(params.head_dim));
    let angle = f32(position) * freq;
    let s = sin(angle);
    let c = cos(angle);

    // Pair index selection:
    //   LLAMA: (dim_idx, dim_idx + 1) — consecutive floats.
    //   NEOX:  (i, i + half_dim)      — first-half + second-half floats.
    // NEOX mirrors `apply_rope_neox` in src/llama3.rs (line 1826) which Qwen
    // 2/2.5/3/3.5 and Gemma 2 require because their GGUF Q/K weights are stored
    // in raw HF layout (no LLaMA-style permute). Applying LLAMA-style rotation
    // to those weights silently produces the wrong rotated vector — the bug
    // that Issue #40 chased through six diagnostic rounds before this fix.
    var buf_idx_0: u32;
    var buf_idx_1: u32;
    if (params.neox != 0u) {
        buf_idx_0 = head_base + pair_in_head;
        buf_idx_1 = buf_idx_0 + half_dim;
    } else {
        let dim_idx = pair_in_head * 2u;
        buf_idx_0 = head_base + dim_idx;
        buf_idx_1 = buf_idx_0 + 1u;
    }

    let x0 = vec_data[buf_idx_0];
    let x1 = vec_data[buf_idx_1];
    vec_data[buf_idx_0] = x0 * c - x1 * s;
    vec_data[buf_idx_1] = x0 * s + x1 * c;
}
