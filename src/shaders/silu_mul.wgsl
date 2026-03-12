// SiLU(gate) * up — element-wise, in-place on gate buffer.
// gate[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]

@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;

@compute @workgroup_size(256)
fn silu_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&gate)) { return; }
    let a = gate[i];
    gate[i] = a / (1.0 + exp(-a)) * up[i];
}
