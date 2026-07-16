// Bonsai / Qwen 3.6-27B DeltaNet SSM discretisation: per-V-head decay factor
// computation from raw alpha projection + ssm_dt_bias + ssm_a.
//
// Phase X.3.e.3 Gap B (see commit 6d43602 for CPU implementation).
//
// Reference formula from llama.cpp PrismML fork `qwen35.cpp:443-451`:
//
//   alpha_biased[h]  = alpha[h] + ssm_dt_bias[h]
//   alpha_softplus[h] = softplus(alpha_biased[h])    // > 0
//   gate[h]           = alpha_softplus[h] * ssm_a[h] // < 0 (ssm_a stored ≈ -exp(A_log), Mamba convention)
//   decay[h]          = exp(gate[h])                 // ∈ (0, 1]
//
// Where `softplus(x) = ln(1 + exp(x))` with numerical stability for large |x|:
//   softplus(x) ≈ x     when x > 20  (avoid exp overflow → inf)
//   softplus(x) ≈ exp(x) when x < -20 (avoid ln(1) underflow)
//   otherwise: standard ln(1 + exp(x))
//
// Output: writes decay factor back into alpha buffer in-place (alpha is
// consumed and replaced by decay before delta-rule integration).
//
// Only applied when GGUF is Bonsai-flavored (ssm_a + ssm_dt_bias both
// present); standard Qwen 3.5 skips this shader and uses raw alpha directly.

@group(0) @binding(0) var<storage, read_write> alpha: array<f32>;
@group(0) @binding(1) var<storage, read> ssm_dt_bias: array<f32>;
@group(0) @binding(2) var<storage, read> ssm_a: array<f32>;

fn softplus_stable(x: f32) -> f32 {
    // Numerically stable softplus, avoiding exp overflow / ln(1) underflow.
    if (x > 20.0) {
        return x;
    }
    if (x < -20.0) {
        return exp(x);
    }
    return log(1.0 + exp(x));
}

@compute @workgroup_size(256)
fn ssm_discretisation(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= arrayLength(&alpha)) { return; }
    let a_biased = alpha[h] + ssm_dt_bias[h];
    let a_softplus = softplus_stable(a_biased);
    let gate = a_softplus * ssm_a[h];
    // decay = exp(gate) ∈ (0, 1] since gate ≤ 0 (ssm_a is negative)
    alpha[h] = exp(gate);
}
