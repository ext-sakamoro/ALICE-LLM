// KV cache append (batch-capable): copy K and V vectors into cache.
// wid.y = batch index. Token k writes to cache position = base_position + k.
// k_cache[(position+k) * kv_dim + i] = k_in[k * kv_dim + i]

struct Params {
    position: u32,
    kv_dim: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> k_in: array<f32>;
@group(0) @binding(1) var<storage, read> v_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn kv_cache_append(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let batch_idx = wid.y;
    let i = gid.x;
    if (i >= params.kv_dim) { return; }

    let in_off = batch_idx * params.kv_dim + i;
    let cache_off = (params.position + batch_idx) * params.kv_dim + i;
    k_cache[cache_off] = k_in[in_off];
    v_cache[cache_off] = v_in[in_off];
}
