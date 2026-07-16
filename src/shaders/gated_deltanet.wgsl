// Gated DeltaNet recurrent update + output for Qwen3.5 linear attention.
//
// Per-head recurrent state S of shape [qk_dim, v_dim].
// Single-token decode step:
//
// Standard Qwen 3.5 path (is_bonsai == 0):
//   q, k = l2_normalize(silu(q)), l2_normalize(silu(k))
//   error = v - alpha * (S^T @ k)
//   S_new = alpha * S + beta * outer(k, error)
//   output = S_new^T @ q
//   output = output * silu(z)    // gated output
//
// Bonsai / Qwen 3.6-27B path (is_bonsai == 1):
//   Phase X.3.e.3 §Q/K L2Norm: use raw q/k instead of silu(q)/silu(k) inside
//   the L2-norm multiplication (silu was already applied externally via
//   silu_inplace on the conv1d output).
//   q, k = q / ||q||, k / ||k||
//   error = v - alpha * (S^T @ k)
//   S_new = alpha * S + beta * outer(k, error)
//   output = S_new^T @ q
//   // silu(z) multiplication is SKIPPED inside the shader — it is applied
//   // externally by silu_gate_apply AFTER the ssm_norm per-V-head RMSNorm
//   // (Phase X.3.e.3 §silu(z) order).
//
// Dispatch: 1 workgroup per head. Each workgroup handles one [qk_dim, v_dim] state.

struct Params {
    num_heads: u32,
    qk_dim: u32,
    v_dim: u32,
    is_bonsai: u32, // 0 = standard Qwen 3.5, 1 = Bonsai / Qwen 3.6-27B path
}

@group(0) @binding(0) var<storage, read> q_buf: array<f32>;      // [num_heads, qk_dim]
@group(0) @binding(1) var<storage, read> k_buf: array<f32>;      // [num_heads, qk_dim]
@group(0) @binding(2) var<storage, read> v_buf: array<f32>;      // [num_heads, v_dim]
@group(0) @binding(3) var<storage, read> alpha_buf: array<f32>;  // [num_heads] decay gate
@group(0) @binding(4) var<storage, read> beta_buf: array<f32>;   // [num_heads] update rate
@group(0) @binding(5) var<storage, read> z_buf: array<f32>;      // [num_heads, v_dim] output gate
@group(0) @binding(6) var<storage, read_write> state: array<f32>; // [num_heads, qk_dim, v_dim]
@group(0) @binding(7) var<storage, read_write> out_buf: array<f32>; // [num_heads, v_dim]
@group(0) @binding(8) var<uniform> params: Params;

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

// Apply silu(x) unless we're on the Bonsai path (where the outer silu was
// already applied to q/k externally via silu_inplace on the conv1d output).
fn q_k_activation(x: f32, is_bonsai: u32) -> f32 {
    if (is_bonsai == 1u) {
        return x;
    }
    return silu(x);
}

@compute @workgroup_size(1)
fn gated_deltanet(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head = gid.x;
    if head >= params.num_heads { return; }

    let qk_dim = params.qk_dim;
    let v_dim = params.v_dim;
    let is_bonsai = params.is_bonsai;
    let q_off = head * qk_dim;
    let k_off = head * qk_dim;
    let v_off = head * v_dim;
    let z_off = head * v_dim;
    let s_off = head * qk_dim * v_dim;

    let alpha = alpha_buf[head];
    let beta = beta_buf[head];

    // Q normalisation follows the reference `qwen35.cpp:319-321`:
    //   q = q * (1/sqrt(qk_dim))   // scale
    //   q = q / ||q||               // L2-normalise
    // These fold into a single per-element multiplier `q_scale / ||q||`,
    // matching the CPU implementation in `llama3::gated_deltanet_head_disjoint`
    // (see commit 8af6ffb — Phase X.3.e.3.5 Q scale fix).
    var q_sum_sq: f32 = 0.0;
    for (var i = 0u; i < qk_dim; i += 1u) {
        let val = q_buf[q_off + i];
        q_sum_sq += val * val;
    }
    let q_scale = 1.0 / sqrt(f32(qk_dim));
    let q_norm = q_scale / max(sqrt(q_sum_sq), 1e-12);

    // L2 norm for k (same treatment as q).
    var k_sum_sq: f32 = 0.0;
    for (var i = 0u; i < qk_dim; i += 1u) {
        let val = k_buf[k_off + i];
        k_sum_sq += val * val;
    }
    let k_norm = 1.0 / max(sqrt(k_sum_sq), 1e-12);

    for (var j = 0u; j < v_dim; j += 1u) {
        // S^T @ k for column j — activation branched by is_bonsai.
        var st_k: f32 = 0.0;
        for (var i = 0u; i < qk_dim; i += 1u) {
            let k_i = q_k_activation(k_buf[k_off + i], is_bonsai) * k_norm;
            st_k += state[s_off + i * v_dim + j] * k_i;
        }

        let error_j = v_buf[v_off + j] - alpha * st_k;

        // Update state column j and compute output.
        var out_j: f32 = 0.0;
        for (var i = 0u; i < qk_dim; i += 1u) {
            let k_i = q_k_activation(k_buf[k_off + i], is_bonsai) * k_norm;
            let q_i = q_k_activation(q_buf[q_off + i], is_bonsai) * q_norm;
            let idx = s_off + i * v_dim + j;

            let s_new = alpha * state[idx] + beta * k_i * error_j;
            state[idx] = s_new;

            out_j += s_new * q_i;
        }

        // Output modulation branched by is_bonsai:
        //   Standard Qwen 3.5: output = out_j * silu(z)
        //   Bonsai: output = out_j (silu(z) applied externally by
        //     silu_gate_apply AFTER ssm_norm per-V-head, Phase X.3.e.3
        //     §silu(z) order).
        if (is_bonsai == 1u) {
            out_buf[v_off + j] = out_j;
        } else {
            let gate = silu(z_buf[z_off + j]);
            out_buf[v_off + j] = out_j * gate;
        }
    }
}
