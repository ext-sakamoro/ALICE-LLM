// Causal Conv1d for Gated DeltaNet preprocessing.
// Applies depthwise 1D convolution with kernel_size=4 per channel.
// Uses a ring buffer of previous activations (kernel_size - 1 = 3).
//
// Input:  x[batch, dim]          — current timestep activation
// State:  conv_state[3, dim]     — previous 3 activations (ring buffer)
// Weight: conv_weight[4, dim]    — depthwise conv kernel
// Bias:   conv_bias[dim]         — per-channel bias
// Output: out[batch, dim]        — convolved activation
//
// For decode (single token): out[d] = sum_k(conv_weight[k, d] * history[k, d]) + bias[d]
// where history = [conv_state[0..2], current_x] (4 timesteps total)
//
// Storage note (Phase X.3.e.3.5): GGUF `ssm_conv1d.weight` has ggml shape
// `{kernel_size, dim}` with `ne[0] = kernel_size` fastest-varying, so the
// storage layout is dim-outer × kernel-inner. Access element `(k, d)` via
// `conv_weight[d * kernel_size + k]`, NOT `conv_weight[k * dim + d]` — the
// latter is transposed and produces garbage for every hybrid arch that ships
// an `ssm_conv1d` tensor (Qwen 3.5, Bonsai, Qwen 3.6, etc.).

struct Params {
    dim: u32,
    ring_pos: u32,   // current position in ring buffer (0..2)
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x_buf: array<f32>;
@group(0) @binding(1) var<storage, read_write> conv_state: array<f32>;  // [3, dim]
@group(0) @binding(2) var<storage, read> conv_weight: array<f32>;       // [4, dim]
@group(0) @binding(3) var<storage, read> conv_bias: array<f32>;         // [dim]
@group(0) @binding(4) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn conv1d_causal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if d >= params.dim { return; }

    let dim = params.dim;
    let rp = params.ring_pos;

    // Read the 3 previous activations from ring buffer + current
    // Ring buffer order: [rp+1, rp+2, rp] maps to [t-3, t-2, t-1]
    let h0 = conv_state[((rp + 1u) % 3u) * dim + d]; // t-3
    let h1 = conv_state[((rp + 2u) % 3u) * dim + d]; // t-2
    let h2 = conv_state[rp * dim + d];                // t-1
    let h3 = x_buf[d];                                // t (current)

    // Depthwise conv: sum over kernel dimension. GGUF stores conv_weight as
    // dim-outer × kernel-inner (`ne[0] = kernel_size = 4`), so for each
    // channel `d` the 4 kernel positions are contiguous at `d * 4 + k`.
    let w_base = d * 4u;
    let w0 = conv_weight[w_base + 0u];
    let w1 = conv_weight[w_base + 1u];
    let w2 = conv_weight[w_base + 2u];
    let w3 = conv_weight[w_base + 3u];

    out_buf[d] = w0 * h0 + w1 * h1 + w2 * h2 + w3 * h3 + conv_bias[d];

    // Update ring buffer: write current x into the oldest slot
    conv_state[((rp + 1u) % 3u) * dim + d] = h3;
}
