// Bonsai / Qwen 3.6-27B gated attention: attn_out *= silu(gate) in-place on
// attn_out buffer.
//
// In Bonsai full-attention layers, `attn_q [q_dim*2, hidden]` projects both Q
// (upper `q_dim` rows) and the swish gate (lower `q_dim` rows). After the
// standard scaled-dot-product attention output computation, the raw attention
// output is multiplied element-wise by silu(gate) before the output projection
// (`o_proj`) matvec.
//
// Equivalent CPU reference:
//   for i in 0..n {
//       attn_out[i] = attn_out[i] * (gate[i] / (1.0 + (-gate[i]).exp()));
//   }
//
// Distinct from `silu_mul.wgsl` (used for SwiGLU FFN), which computes
// `gate[i] = silu(gate[i]) * up[i]` in-place on `gate`. Here the write target
// is `attn_out` and the silu-transformed operand is `gate` (read-only). Naming
// mirrors that distinction to avoid confusion at the call site.

@group(0) @binding(0) var<storage, read_write> attn_out: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;

@compute @workgroup_size(256)
fn silu_gate_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&attn_out)) { return; }
    let g = gate[i];
    // silu(g) = g / (1 + exp(-g)) = g * sigmoid(g)
    // attn_out[i] *= silu(g)
    attn_out[i] = attn_out[i] * g / (1.0 + exp(-g));
}
