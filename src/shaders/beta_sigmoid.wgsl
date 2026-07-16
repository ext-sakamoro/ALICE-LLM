// Bonsai / Qwen 3.6-27B DeltaNet beta transformation: beta[i] = sigmoid(beta_raw[i])
// in-place on beta buffer.
//
// Phase X.3.e.3 Gap B extra (see commit 2d55f2b for CPU implementation).
//
// The DeltaNet recurrence's update rate `beta` must be constrained to (0, 1)
// for the delta rule to remain stable. Standard Qwen 3.5 uses raw beta from
// `beta_proj` directly; Bonsai / Qwen 3.6-27B applies sigmoid to bring it
// into the valid range. Reference: llama.cpp PrismML fork
// `qwen35.cpp:440-441` (`ggml_sigmoid(beta)`).
//
// Equivalent CPU reference:
//   for i in 0..n {
//       beta[i] = 1.0 / (1.0 + (-beta[i]).exp());
//   }
//
// Only applied when the GGUF is Bonsai-flavored (detected by `ssm_a` +
// `ssm_dt_bias` tensor presence); standard Qwen 3.5 skips this shader.

@group(0) @binding(0) var<storage, read_write> beta: array<f32>;

@compute @workgroup_size(256)
fn beta_sigmoid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&beta)) { return; }
    let b = beta[i];
    // sigmoid(b) = 1 / (1 + exp(-b))
    beta[i] = 1.0 / (1.0 + exp(-b));
}
