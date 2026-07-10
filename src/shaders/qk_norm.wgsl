// Per-head RMSNorm applied in-place on Q or K after projection, before RoPE.
// One workgroup per head. `wid.x` = head index.
// `dim` = head_dim (typically 128 for Qwen 3). Weight has `head_dim` elements
// shared across all heads.
//
// Used by Qwen 3 architecture family (`attn_q_norm.weight` / `attn_k_norm.weight`).
// Qwen 2 / 2.5 / Llama / Mistral / Gemma do not have per-head QK norm and skip this.

struct Params {
    dim: u32,
    eps: f32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn qk_norm(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let head_idx = wid.x;
    let tid = lid.x;
    let n = params.dim;
    let base = head_idx * n;

    // Phase 1: each thread accumulates local sum of squares (strided over dim).
    var local_sum: f32 = 0.0;
    var i = tid;
    while (i < n) {
        let v = buf[base + i];
        local_sum = local_sum + v * v;
        i = i + 256u;
    }

    // Phase 2: parallel reduction to workgroup sum.
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + s];
        }
        workgroupBarrier();
    }

    // Thread 0 computes the RMSNorm scale (inverse sqrt of mean of squares + eps).
    if (tid == 0u) {
        shared_sum[0u] = inverseSqrt(shared_sum[0u] / f32(n) + params.eps);
    }
    workgroupBarrier();

    let scale = shared_sum[0u];

    // Phase 3: in-place normalize + scale by shared per-dim weight.
    i = tid;
    while (i < n) {
        buf[base + i] = buf[base + i] * scale * weight[i];
        i = i + 256u;
    }
}
