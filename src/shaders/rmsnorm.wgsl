// RMSNorm compute shader (batch-capable).
// out[batch*dim + i] = input[batch*dim + i] * weight[i] * rsqrt(mean(input^2) + eps)
// One workgroup per token in batch. wid.x = batch index.

struct Params {
    dim: u32,
    eps: f32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn rmsnorm(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let batch_idx = wid.x;
    let tid = lid.x;
    let n = params.dim;
    let base = batch_idx * n;

    // Phase 1: Each thread sums squares for its elements
    var local_sum: f32 = 0.0;
    var i = tid;
    while (i < n) {
        let v = input_data[base + i];
        local_sum = local_sum + v * v;
        i = i + 256u;
    }

    // Phase 2: Parallel reduction to get total sum of squares
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + s];
        }
        workgroupBarrier();
    }

    // Thread 0 computes the normalization scale
    if (tid == 0u) {
        shared_sum[0u] = inverseSqrt(shared_sum[0u] / f32(n) + params.eps);
    }
    workgroupBarrier();

    let scale = shared_sum[0u];

    // Phase 3: Apply normalization and weight (weight is NOT batched)
    i = tid;
    while (i < n) {
        output_data[base + i] = input_data[base + i] * scale * weight[i];
        i = i + 256u;
    }
}
