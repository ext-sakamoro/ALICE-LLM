// GQA (Grouped Query Attention) compute shader (batch-capable).
// 1 workgroup per (head, token_in_batch). wg.x = head, wg.y = batch index.
// Causal: token k attends to seq_len + k positions.
//
// For each head h, token k:
//   kv_h = h / heads_per_kv
//   attend_len = seq_len + k  (causal mask)
//   score[t] = dot(Q[k,h], K_cache[t, kv_h]) * inv_sqrt_d   for t in 0..attend_len
//   softmax(score)
//   out[k,h] = sum_t score[t] * V_cache[t, kv_h]

struct Params {
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    inv_sqrt_d: f32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> q_buf: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const MAX_SEQ: u32 = 4096u;
var<workgroup> scores: array<f32, MAX_SEQ>;
var<workgroup> shared_reduce: array<f32, 256>;

@compute @workgroup_size(256)
fn attention(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let h = wg.x;
    let batch_idx = wg.y;
    let tid = lid.x;

    if (h >= params.num_heads) { return; }

    // Causal: token k attends to seq_len + k positions
    let seq_len = params.seq_len + batch_idx;
    if (seq_len == 0u) { return; }

    let heads_per_kv = params.num_heads / params.num_kv_heads;
    let kv_h = h / heads_per_kv;

    // Q offset includes batch index (K Q-vectors concatenated)
    let q_off = batch_idx * params.num_heads * params.head_dim + h * params.head_dim;

    // Phase 1: Compute dot product scores
    var t = tid;
    while (t < seq_len) {
        let k_off = t * params.kv_dim + kv_h * params.head_dim;
        var dot_val: f32 = 0.0;
        for (var d = 0u; d < params.head_dim; d = d + 1u) {
            dot_val = dot_val + q_buf[q_off + d] * k_cache[k_off + d];
        }
        scores[t] = dot_val * params.inv_sqrt_d;
        t = t + 256u;
    }
    workgroupBarrier();

    // Phase 2a: Find max score (parallel reduction)
    var local_max: f32 = -1e38;
    t = tid;
    while (t < seq_len) {
        local_max = max(local_max, scores[t]);
        t = t + 256u;
    }
    shared_reduce[tid] = local_max;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            shared_reduce[tid] = max(shared_reduce[tid], shared_reduce[tid + s]);
        }
        workgroupBarrier();
    }
    let max_val = shared_reduce[0u];
    workgroupBarrier();

    // Phase 2b: Exponentiate and sum
    var local_sum: f32 = 0.0;
    t = tid;
    while (t < seq_len) {
        let e = exp(scores[t] - max_val);
        scores[t] = e;
        local_sum = local_sum + e;
        t = t + 256u;
    }
    shared_reduce[tid] = local_sum;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            shared_reduce[tid] = shared_reduce[tid] + shared_reduce[tid + s];
        }
        workgroupBarrier();
    }
    let inv_sum = 1.0 / shared_reduce[0u];
    workgroupBarrier();

    // Phase 2c: Normalize scores
    t = tid;
    while (t < seq_len) {
        scores[t] = scores[t] * inv_sum;
        t = t + 256u;
    }
    workgroupBarrier();

    // Phase 3: Weighted sum of V (output offset includes batch index)
    let out_off = batch_idx * params.num_heads * params.head_dim + h * params.head_dim;
    var d_idx = tid;
    while (d_idx < params.head_dim) {
        var weighted_val: f32 = 0.0;
        for (var tt = 0u; tt < seq_len; tt = tt + 1u) {
            let v_off = tt * params.kv_dim + kv_h * params.head_dim;
            weighted_val = weighted_val + scores[tt] * v_cache[v_off + d_idx];
        }
        out_buf[out_off + d_idx] = weighted_val;
        d_idx = d_idx + 256u;
    }
}
