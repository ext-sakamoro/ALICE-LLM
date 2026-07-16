// Element-wise SiLU (Swish) applied in-place on the input buffer:
//   buf[i] = buf[i] / (1 + exp(-buf[i])) = buf[i] * sigmoid(buf[i])
//
// Bonsai / Qwen 3.6-27B DeltaNet post-conv1d silu (Phase X.3.e.3 Gap A):
// applied to the causal-conv1d output before delta-rule integration.
//
// Distinct from `silu_mul.wgsl` (SwiGLU FFN: `gate[i] = silu(gate[i]) * up[i]`,
// two-input) and `silu_gate_apply.wgsl` (Bonsai attention gate:
// `attn_out[i] *= silu(gate[i])`, two-buffer). This is the single-input,
// single-buffer variant needed for plain silu activation.
//
// Equivalent CPU reference:
//   for x in &mut buf { *x = *x / (1.0 + (-*x).exp()); }

@group(0) @binding(0) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(256)
fn silu_inplace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&buf)) { return; }
    let x = buf[i];
    // silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
    buf[i] = x / (1.0 + exp(-x));
}
