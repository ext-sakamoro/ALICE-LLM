// Residual connection: acc_buf[i] += addend[i]

@group(0) @binding(0) var<storage, read_write> acc_buf: array<f32>;
@group(0) @binding(1) var<storage, read> addend: array<f32>;

@compute @workgroup_size(256)
fn residual_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&acc_buf)) { return; }
    acc_buf[i] = acc_buf[i] + addend[i];
}
