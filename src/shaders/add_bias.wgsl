// Element-wise bias add (broadcasts bias vector across an optional batch dim).
//
// acc_buf layout for batched matvec output: [bs][bias_len] contiguous, row-major.
// For single-token forward, dispatch with y = 1 (bs = 1).
// For batch-4 forward, dispatch with y = 4 (bs = 4).
//
// dispatch: workgroups = (ceil(bias_len / 256), bs, 1)

@group(0) @binding(0) var<storage, read_write> acc_buf: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;

@compute @workgroup_size(256)
fn add_bias(@builtin(global_invocation_id) gid: vec3<u32>) {
    let elem = gid.x;
    let bs_idx = gid.y;
    let bias_len = arrayLength(&bias);
    if (elem >= bias_len) { return; }
    let idx = bs_idx * bias_len + elem;
    acc_buf[idx] = acc_buf[idx] + bias[elem];
}
