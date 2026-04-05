// Gated DeltaNet recurrent update + output for Qwen3.5 linear attention.
//
// Per-head recurrent state S of shape [qk_dim, v_dim].
// Single-token decode step:
//
//   q, k = l2_normalize(silu(q)), l2_normalize(silu(k))
//   error = v - alpha * (S^T @ k)
//   S_new = alpha * S + beta * outer(k, error)
//   output = S_new^T @ q
//   output = output * silu(z)    // gated output
//
// Dispatch: 1 workgroup per head. Each workgroup handles one [qk_dim, v_dim] state.

struct Params {
    num_heads: u32,
    qk_dim: u32,
    v_dim: u32,
    _pad: u32,
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

@compute @workgroup_size(1)
fn gated_deltanet(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head = gid.x;
    if head >= params.num_heads { return; }

    let qk_dim = params.qk_dim;
    let v_dim = params.v_dim;
    let q_off = head * qk_dim;
    let k_off = head * qk_dim;
    let v_off = head * v_dim;
    let z_off = head * v_dim;
    let s_off = head * qk_dim * v_dim;

    let alpha = alpha_buf[head];
    let beta = beta_buf[head];

    // L2 norm for q
    var q_sum_sq: f32 = 0.0;
    for (var i = 0u; i < qk_dim; i += 1u) {
        let val = q_buf[q_off + i];
        q_sum_sq += val * val;
    }
    let q_norm = 1.0 / max(sqrt(q_sum_sq), 1e-12);

    // L2 norm for k
    var k_sum_sq: f32 = 0.0;
    for (var i = 0u; i < qk_dim; i += 1u) {
        let val = k_buf[k_off + i];
        k_sum_sq += val * val;
    }
    let k_norm = 1.0 / max(sqrt(k_sum_sq), 1e-12);

    for (var j = 0u; j < v_dim; j += 1u) {
        // S^T @ k for column j
        var st_k: f32 = 0.0;
        for (var i = 0u; i < qk_dim; i += 1u) {
            let k_i = silu(k_buf[k_off + i]) * k_norm;
            st_k += state[s_off + i * v_dim + j] * k_i;
        }

        let error_j = v_buf[v_off + j] - alpha * st_k;

        // Update state column j and compute output
        var out_j: f32 = 0.0;
        for (var i = 0u; i < qk_dim; i += 1u) {
            let k_i = silu(k_buf[k_off + i]) * k_norm;
            let q_i = silu(q_buf[q_off + i]) * q_norm;
            let idx = s_off + i * v_dim + j;

            let s_new = alpha * state[idx] + beta * k_i * error_j;
            state[idx] = s_new;

            out_j += s_new * q_i;
        }

        // Gated output
        let gate = silu(z_buf[z_off + j]);
        out_buf[v_off + j] = out_j * gate;
    }
}
