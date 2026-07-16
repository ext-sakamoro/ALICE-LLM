// Qwen 3.5 / 3.6 / Bonsai 27B gated attention: attn_out *= sigmoid(gate)
// in-place on attn_out buffer.
//
// Reference qwen35.cpp:401-404 applies `ggml_sigmoid(gate)` (not silu) after
// the attention output computation, then multiplies elementwise with the
// attention output before the output projection (`o_proj`) matvec.
//
// Phase X.3.e.3.14 fix: previously used `silu_gate_apply.wgsl` (silu, i.e.
// `g / (1 + exp(-g))`) which mismatched reference and caused dramatic
// divergence propagating from attention layer onwards (sign-flipped
// attn_norm on next DeltaNet layer, garbage generation end-to-end).
//
// Equivalent CPU reference:
//   for i in 0..n {
//       attn_out[i] = attn_out[i] / (1.0 + (-gate[i]).exp());
//   }
//
// Distinct from `silu_gate_apply.wgsl` which computes `silu(g) = g * sigmoid(g)`
// and is still used for the Bonsai DeltaNet z-gate (post-ssm-norm) where the
// reference kernel wraps `silu(z)` around the recurrence output.

@group(0) @binding(0) var<storage, read_write> attn_out: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;

@compute @workgroup_size(256)
fn sigmoid_gate_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&attn_out)) { return; }
    let g = gate[i];
    // sigmoid(g) = 1 / (1 + exp(-g))
    // attn_out[i] *= sigmoid(g)
    attn_out[i] = attn_out[i] / (1.0 + exp(-g));
}
