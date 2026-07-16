//! GPU compute engine for quantized inference via wgpu (Metal/Vulkan/DX12).
//!
//! Provides GPU-accelerated matvec, RMSNorm, SiLU, RoPE, and residual add.
//! All operations can be chained via `GpuPass` into a single command buffer,
//! eliminating per-operation submission overhead.

use std::sync::Arc;
use wgpu::util::DeviceExt;

const MATVEC_Q4K_SHADER: &str = include_str!("shaders/dequant_matvec_q4k.wgsl");
const MATVEC_Q5K_SHADER: &str = include_str!("shaders/dequant_matvec_q5k.wgsl");
const MATVEC_Q6K_SHADER: &str = include_str!("shaders/dequant_matvec_q6k.wgsl");
const MATVEC_Q8_0_SHADER: &str = include_str!("shaders/dequant_matvec_q8_0.wgsl");
const MATVEC_Q1_0_SHADER: &str = include_str!("shaders/dequant_matvec_q1_0.wgsl");
const MATVEC_Q1_0_BATCH4_SHADER: &str = include_str!("shaders/dequant_matvec_q1_0_batch4.wgsl");
const MATVEC_Q1_0_ROW4_SHADER: &str = include_str!("shaders/dequant_matvec_q1_0_row4.wgsl");
const MATVEC_Q1_0_ROW4_BATCH4_SHADER: &str =
    include_str!("shaders/dequant_matvec_q1_0_row4_batch4.wgsl");
const MATVEC_Q1_0_ROW8_BATCH4_SHADER: &str =
    include_str!("shaders/dequant_matvec_q1_0_row8_batch4.wgsl");
const RMSNORM_SHADER: &str = include_str!("shaders/rmsnorm.wgsl");
const SILU_MUL_SHADER: &str = include_str!("shaders/silu_mul.wgsl");
const SILU_GATE_APPLY_SHADER: &str = include_str!("shaders/silu_gate_apply.wgsl");
const SIGMOID_GATE_APPLY_SHADER: &str = include_str!("shaders/sigmoid_gate_apply.wgsl");
const SILU_INPLACE_SHADER: &str = include_str!("shaders/silu_inplace.wgsl");
const BETA_SIGMOID_SHADER: &str = include_str!("shaders/beta_sigmoid.wgsl");
const SSM_DISCRETISATION_SHADER: &str = include_str!("shaders/ssm_discretisation.wgsl");
const RESIDUAL_ADD_SHADER: &str = include_str!("shaders/residual_add.wgsl");
const ADD_BIAS_SHADER: &str = include_str!("shaders/add_bias.wgsl");
const QK_NORM_SHADER: &str = include_str!("shaders/qk_norm.wgsl");
const ROPE_SHADER: &str = include_str!("shaders/rope.wgsl");
const ATTENTION_SHADER: &str = include_str!("shaders/attention.wgsl");
const KV_CACHE_APPEND_SHADER: &str = include_str!("shaders/kv_cache_append.wgsl");
const SWIGLU_FUSED_Q4K_SHADER: &str = include_str!("shaders/swiglu_fused_q4k.wgsl");
const SWIGLU_FUSED_Q1_0_SHADER: &str = include_str!("shaders/swiglu_fused_q1_0.wgsl");
const MATVEC_Q4K_BATCH4_SHADER: &str = include_str!("shaders/dequant_matvec_q4k_batch4.wgsl");
const MATVEC_Q6K_BATCH4_SHADER: &str = include_str!("shaders/dequant_matvec_q6k_batch4.wgsl");
const SWIGLU_FUSED_Q4K_BATCH4_SHADER: &str = include_str!("shaders/swiglu_fused_q4k_batch4.wgsl");
const CONV1D_CAUSAL_SHADER: &str = include_str!("shaders/conv1d_causal.wgsl");
const GATED_DELTANET_SHADER: &str = include_str!("shaders/gated_deltanet.wgsl");

// --- CPU reference helpers for GPU shaders ---

/// CPU reference implementation of the `silu_gate_apply` WGSL shader.
///
/// Computes `attn_out[i] *= silu(gate[i])` in-place on `attn_out`, matching
/// `src/shaders/silu_gate_apply.wgsl` line-by-line. Used by Bonsai / Qwen
/// 3.6-27B full-attention layers to modulate the raw attention output by the
/// swish gate (from `attn_q [q_dim*2, hidden]` lower half) before the
/// `o_proj` matvec.
///
/// This is the definitive CPU parity reference for the GPU shader — any
/// future integration test comparing GPU output to CPU should call this
/// function to produce the expected values.
///
/// # Panics
///
/// Panics if `attn_out.len() != gate.len()`.
pub fn silu_gate_apply_cpu(attn_out: &mut [f32], gate: &[f32]) {
    assert_eq!(
        attn_out.len(),
        gate.len(),
        "silu_gate_apply_cpu: attn_out and gate must have same length"
    );
    for (a, &g) in attn_out.iter_mut().zip(gate.iter()) {
        // silu(g) = g / (1 + exp(-g)) = g * sigmoid(g)
        // attn_out[i] *= silu(g)
        *a = *a * g / (1.0 + (-g).exp());
    }
}

/// CPU reference implementation of the `silu_inplace` WGSL shader.
///
/// Computes `buf[i] = silu(buf[i])` in-place, matching
/// `src/shaders/silu_inplace.wgsl` line-by-line. Used by Bonsai / Qwen
/// 3.6-27B DeltaNet post-conv1d silu (Phase X.3.e.3 Gap A).
pub fn silu_inplace_cpu(buf: &mut [f32]) {
    for x in buf.iter_mut() {
        // silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
        *x = *x / (1.0 + (-*x).exp());
    }
}

/// CPU reference implementation of the `beta_sigmoid` WGSL shader.
///
/// Computes `beta[i] = sigmoid(beta_raw[i])` in-place on `beta`, matching
/// `src/shaders/beta_sigmoid.wgsl` line-by-line. Used by Bonsai / Qwen
/// 3.6-27B DeltaNet layers to bring the raw beta projection into the
/// stable delta-rule integration range (0, 1). Reference: llama.cpp
/// PrismML fork `qwen35.cpp:440-441` (`ggml_sigmoid(beta)`).
pub fn beta_sigmoid_cpu(beta: &mut [f32]) {
    for b in beta.iter_mut() {
        // sigmoid(x) = 1 / (1 + exp(-x))
        *b = 1.0 / (1.0 + (-*b).exp());
    }
}

/// CPU reference implementation of the `ssm_discretisation` WGSL shader.
///
/// Computes per-V-head decay factor from raw alpha projection, ssm_dt_bias,
/// and ssm_a, matching `src/shaders/ssm_discretisation.wgsl` line-by-line.
/// Writes decay back into `alpha` buffer in-place.
///
/// Formula (per llama.cpp PrismML fork `qwen35.cpp:443-451`):
/// ```text
///   alpha_biased[h]   = alpha[h] + ssm_dt_bias[h]
///   alpha_softplus[h] = softplus(alpha_biased[h])   // > 0
///   gate[h]           = alpha_softplus[h] * ssm_a[h] // < 0 (Mamba convention)
///   decay[h]          = exp(gate[h])                 // ∈ (0, 1]
/// ```
///
/// Softplus is numerically stable for large |x|:
/// - `x > 20`: `softplus(x) ≈ x` (avoid exp overflow)
/// - `x < -20`: `softplus(x) ≈ exp(x)` (asymptotic)
/// - otherwise: `ln(1 + exp(x))`
///
/// # Panics
///
/// Panics if `alpha.len()`, `ssm_dt_bias.len()`, and `ssm_a.len()` are not
/// all equal.
/// CPU reference implementation of `GpuPass::ssm_norm_per_head` — per-V-head
/// RMSNorm on `dn_delta_out` with `ssm_norm` weight broadcast across V heads.
///
/// Bonsai / Qwen 3.6-27B DeltaNet Phase X.3.e.3 Gap C. Applies RMSNorm
/// independently to each `v_dim`-sized slab of `input`, with the shared
/// `weight` vector broadcast across all heads. In-place operation.
///
/// This is the f32-accumulation variant matching the WGSL shader's shared-
/// memory reduction. Numerically identical (within FP summation order noise)
/// to `llama3::apply_qk_norm` which accumulates in f64 for a bit more
/// precision on very large head dims — for Bonsai's v_dim=128 the difference
/// is < 1e-6.
///
/// Formula per V head h:
/// ```text
///   ss[h]    = sum(input[h*v_dim..(h+1)*v_dim]^2)
///   scale[h] = 1.0 / sqrt(ss[h] / v_dim + eps)
///   for i in 0..v_dim:
///       input[h*v_dim + i] = input[h*v_dim + i] * scale[h] * weight[i]
/// ```
///
/// # Panics
///
/// Panics if `input.len()` is not a multiple of `v_dim`, or if
/// `weight.len() != v_dim`.
pub fn ssm_norm_per_head_cpu(input: &mut [f32], weight: &[f32], v_dim: usize, eps: f32) {
    assert_eq!(
        input.len() % v_dim,
        0,
        "ssm_norm_per_head_cpu: input.len() ({}) must be multiple of v_dim ({})",
        input.len(),
        v_dim
    );
    assert_eq!(
        weight.len(),
        v_dim,
        "ssm_norm_per_head_cpu: weight.len() ({}) must equal v_dim ({})",
        weight.len(),
        v_dim
    );
    let num_heads = input.len() / v_dim;
    for h in 0..num_heads {
        let start = h * v_dim;
        let slice = &mut input[start..start + v_dim];
        let mut ss = 0.0f32;
        for &v in slice.iter() {
            ss += v * v;
        }
        let scale = 1.0f32 / (ss / v_dim as f32 + eps).sqrt();
        for (i, w) in weight.iter().enumerate() {
            slice[i] = slice[i] * scale * w;
        }
    }
}

pub fn ssm_discretisation_cpu(alpha: &mut [f32], ssm_dt_bias: &[f32], ssm_a: &[f32]) {
    assert_eq!(
        alpha.len(),
        ssm_dt_bias.len(),
        "ssm_discretisation_cpu: alpha and ssm_dt_bias must have same length"
    );
    assert_eq!(
        alpha.len(),
        ssm_a.len(),
        "ssm_discretisation_cpu: alpha and ssm_a must have same length"
    );
    for h in 0..alpha.len() {
        let a_biased = alpha[h] + ssm_dt_bias[h];
        // Numerically stable softplus, avoiding exp overflow / ln(1) underflow.
        let a_softplus = if a_biased > 20.0 {
            a_biased
        } else if a_biased < -20.0 {
            a_biased.exp()
        } else {
            (1.0 + a_biased.exp()).ln()
        };
        let gate = a_softplus * ssm_a[h];
        alpha[h] = gate.exp();
    }
}

// --- Uniform param structs (must match WGSL) ---

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatvecParams {
    rows: u32,
    cols: u32,
    blocks_per_row: u32,
    grid_x: u32,
    batch_size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsnormParams {
    dim: u32,
    eps: f32,
    batch_size: u32,
    _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeParams {
    position: u32,
    num_heads: u32,
    head_dim: u32,
    theta: f32,
    batch_size: u32,
    /// RoPE rotation convention. `0` = LLAMA-style pair `(i, i+1)`,
    /// `1` = NEOX-style pair `(i, i + head_dim/2)`. See Issue #40.
    neox: u32,
    _pad2: u32,
    _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AttentionParams {
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    inv_sqrt_d: f32,
    batch_size: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct KvCacheParams {
    position: u32,
    kv_dim: u32,
    batch_size: u32,
    _pad2: u32,
}

// --- Public types ---

/// GPU buffer wrapping a `wgpu::Buffer`.
pub struct GpuBuffer {
    pub(crate) buffer: wgpu::Buffer,
    /// Number of f32 elements (or raw bytes for weight buffers).
    pub len: usize,
}

/// Quantization type for GPU weight buffers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GpuQuantType {
    Q4K,
    /// Q5_K (256 elements / block, 176 bytes). Mixed with Q4_K in Q4_K_M
    /// GGUFs — Qwen 3.5-4B uses this for `attn_qkv` and `ssm_out` weights.
    Q5K,
    Q6K,
    /// Q8_0 (32 elements / block, 34 bytes packed to 36 bytes on GPU upload
    /// for u32 alignment). Used for small tensors like `ssm_alpha` and
    /// `ssm_beta` in Qwen 3.5-4B (which need higher precision on the tiny
    /// 32-row projection matrices).
    Q8_0,
    /// PrismML Q1_0 (Bonsai 27B binary g128) — 128 elements per block,
    /// 18 bytes per block (2 FP16 scale + 16 × 1-bit packed values).
    Q1_0,
}

/// Quantized weight tensor on GPU (Q4_K or Q6_K).
pub struct GpuWeightBuffer {
    buffer: wgpu::Buffer,
    pub rows: u32,
    pub cols: u32,
    blocks_per_row: u32,
    pub quant: GpuQuantType,
}

/// Pre-allocated context for repeated single matvec with readback.
pub struct MatvecContext {
    input_buf: wgpu::Buffer,
    output_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    rows: u32,
}

/// GPU inference engine.
pub struct GpuEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Nanoseconds per GPU timestamp tick (0.0 if timestamps not supported).
    pub timestamp_period: f32,
    /// Adapter type — used by `GpuModel::load` for OOM prevention heuristics.
    /// `IntegratedGpu` = unified memory (Jetson / Apple Silicon / AMD APU / Intel iGPU),
    /// requires higher peak memory factor since wgpu Vulkan / Metal double-allocates.
    pub device_type: wgpu::DeviceType,
    matvec_q4k_pipeline: wgpu::ComputePipeline,
    /// Q5_K matvec (176 byte blocks with 32-byte qh + 128-byte qs). Same
    /// bind-group layout as Q4_K so `MatvecBG` structures are reused.
    matvec_q5k_pipeline: wgpu::ComputePipeline,
    matvec_q6k_pipeline: wgpu::ComputePipeline,
    /// Q8_0 matvec (32 elements/block, 36-byte padded upload for word
    /// alignment — see `upload_weights_q8_0`). Workgroup size 32 (one
    /// thread per element in a block).
    matvec_q8_0_pipeline: wgpu::ComputePipeline,
    /// PrismML Q1_0 (Bonsai 27B binary g128) matvec, 128 elements/block,
    /// 18 bytes/block, byte-level indexed (blocks straddle `u32` word boundaries).
    matvec_q1_0_pipeline: wgpu::ComputePipeline,
    /// Q1_0 batch-4 matvec — one weight decode per block, 4 FMAs against
    /// 4 batch input vectors. Halves per-token dispatch cost when combined
    /// with 4-token speculative decoding.
    matvec_q1_0_batch4_pipeline: wgpu::ComputePipeline,
    /// Q1_0 row4 matvec (single batch) — 1 workgroup produces 4 rows × 1
    /// batch = 4 outputs. Used by the standard batch=1 decode path where
    /// speculative decoding is not in use. Same 4× workgroup-count
    /// amortization as row4×batch4 without the batch dimension.
    matvec_q1_0_row4_pipeline: wgpu::ComputePipeline,
    /// Q1_0 row4×batch4 matvec — 1 workgroup produces 4 rows × 4 batches =
    /// 16 outputs. Compounds Step 2's batch4 win with 4× fewer workgroups
    /// and 4× less total input-read traffic. Register / smem cost: 16 f32
    /// accumulators + 16 subgroup partial-sum arrays (1 KB smem).
    matvec_q1_0_row4_batch4_pipeline: wgpu::ComputePipeline,
    /// Q1_0 row8×batch4 matvec — 1 workgroup produces 8 rows × 4 batches =
    /// 32 outputs. Doubles the row-batching factor over row4×batch4:
    /// workgroup count drops another 2× (2560 → 1280 on Bonsai attn_qkv)
    /// and total input-read traffic drops another 2×. Register cost: 32 f32
    /// accumulators (scalar named vars, no dynamic indexing → no spill).
    /// Shared memory: 2 KB (32 subgroup partial-sum arrays).
    matvec_q1_0_row8_batch4_pipeline: wgpu::ComputePipeline,
    rmsnorm_pipeline: wgpu::ComputePipeline,
    silu_mul_pipeline: wgpu::ComputePipeline,
    /// Bonsai / Qwen 3.6-27B gated attention output modulation:
    /// `attn_out[i] *= silu(gate[i])` in-place on `attn_out`. Building block
    /// for Phase 2 Step 3-6 forward-path integration (see
    /// `docs/BONSAI_GPU_SUPPORT.md`). Not yet wired into `encode_forward_impl`
    /// — exposed via `GpuPass::silu_gate_apply` for future integration and
    /// standalone parity validation.
    silu_gate_apply_pipeline: wgpu::ComputePipeline,
    /// Element-wise sigmoid gate apply: `attn_out[i] *= sigmoid(gate[i])`.
    /// Used by Qwen 3.5 / 3.6 / Bonsai 27B gated attention (reference
    /// qwen35.cpp:401-404 `ggml_sigmoid(gate)` + `ggml_mul(attn, gate_sigmoid)`).
    /// Phase X.3.e.3.14 introduced sigmoid variant to fix silu vs sigmoid
    /// mismatch on GPU forward path (mirroring CPU b616219 lineage fix).
    sigmoid_gate_apply_pipeline: wgpu::ComputePipeline,
    /// Element-wise SiLU in-place: `buf[i] = buf[i] * sigmoid(buf[i])`.
    /// Bonsai / Qwen 3.6-27B post-conv1d silu (Phase X.3.e.3 Gap A). Distinct
    /// from `silu_mul` (SwiGLU two-input) and `silu_gate_apply` (Bonsai attn
    /// output modulation, two-buffer): this is the single-input variant.
    silu_inplace_pipeline: wgpu::ComputePipeline,
    /// Bonsai / Qwen 3.6-27B DeltaNet beta transformation: element-wise
    /// `beta[i] = sigmoid(beta_raw[i])` in-place on beta buffer. Phase X.3.e.3
    /// Gap B extra (commit `2d55f2b` CPU implementation reference). Only
    /// applied when GGUF has `ssm_a` + `ssm_dt_bias` (Bonsai-flavored).
    beta_sigmoid_pipeline: wgpu::ComputePipeline,
    /// Bonsai / Qwen 3.6-27B DeltaNet SSM discretisation: per-V-head
    /// `decay = exp(softplus(alpha + ssm_dt_bias) * ssm_a)`. Phase X.3.e.3
    /// Gap B (commit `6d43602` CPU reference). Writes decay back into alpha
    /// buffer in-place. Mamba convention: ssm_a stored ≈ -exp(A_log) < 0
    /// → gate < 0 → decay ∈ (0, 1].
    ssm_discretisation_pipeline: wgpu::ComputePipeline,
    residual_add_pipeline: wgpu::ComputePipeline,
    /// Element-wise bias add (Qwen 2 / 2.5 attention QKV biases).
    add_bias_pipeline: wgpu::ComputePipeline,
    /// Per-head RMSNorm on Q / K (Qwen 3 architecture family).
    qk_norm_pipeline: wgpu::ComputePipeline,
    rope_pipeline: wgpu::ComputePipeline,
    attention_pipeline: wgpu::ComputePipeline,
    kv_cache_append_pipeline: wgpu::ComputePipeline,
    swiglu_fused_q4k_pipeline: wgpu::ComputePipeline,
    /// Fused SwiGLU pipeline for Q1_0 weights (Bonsai 27B FFN). Same
    /// bind-group layout as `swiglu_fused_q4k_pipeline` — only the shader
    /// differs to handle the 18-byte / 128-element Q1_0 block layout with
    /// byte-level indexing. Selected by `build_swiglu_bg` via the quant
    /// type of `dlw.gate_proj` / `dlw.up_proj`.
    swiglu_fused_q1_0_pipeline: wgpu::ComputePipeline,
    // Batch-4 specialized pipelines (K=4 unrolled scalar accumulators)
    matvec_q4k_batch4_pipeline: wgpu::ComputePipeline,
    matvec_q6k_batch4_pipeline: wgpu::ComputePipeline,
    swiglu_batch4_pipeline: wgpu::ComputePipeline,
    // DeltaNet (Qwen3.5) pipelines
    conv1d_causal_pipeline: wgpu::ComputePipeline,
    gated_deltanet_pipeline: wgpu::ComputePipeline,
}

/// Command builder — chains GPU operations into a single command buffer.
pub struct GpuPass<'a> {
    engine: &'a GpuEngine,
    encoder: wgpu::CommandEncoder,
}

// --- GpuEngine implementation ---

impl GpuEngine {
    /// Initialize GPU engine (blocking). Selects high-performance adapter.
    #[must_use]
    pub fn new() -> Self {
        pollster::block_on(Self::init_async())
    }

    async fn init_async() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("no GPU adapter found");

        let info = adapter.get_info();
        eprintln!("[gpu] adapter: {} ({:?})", info.name, info.backend);
        let device_type = info.device_type;
        eprintln!("[gpu] device_type: {device_type:?}");

        let adapter_limits = adapter.limits();
        let has_timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let has_subgroups = adapter.features().contains(wgpu::Features::SUBGROUP);
        if has_timestamps {
            eprintln!("[gpu] TIMESTAMP_QUERY supported");
        }
        eprintln!("[gpu] SUBGROUP supported={has_subgroups}");

        let mut required_features = wgpu::Features::empty();
        if has_timestamps {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if has_subgroups {
            required_features |= wgpu::Features::SUBGROUP;
        }

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features,
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: adapter_limits
                            .max_storage_buffer_binding_size,
                        max_buffer_size: adapter_limits.max_buffer_size,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .await
            .expect("failed to create GPU device");
        eprintln!(
            "[gpu] max_storage_buffer: {}MB",
            device.limits().max_storage_buffer_binding_size / (1024 * 1024)
        );

        let make_pipeline = |source: &str, entry: &str, label: &str| -> wgpu::ComputePipeline {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let timestamp_period = queue.get_timestamp_period();

        // Create original pipelines first
        let matvec_q4k_pipeline = make_pipeline(MATVEC_Q4K_SHADER, "matvec_q4k", "matvec_q4k");
        let matvec_q5k_pipeline = make_pipeline(MATVEC_Q5K_SHADER, "matvec_q5k", "matvec_q5k");
        let matvec_q6k_pipeline = make_pipeline(MATVEC_Q6K_SHADER, "matvec_q6k", "matvec_q6k");
        let matvec_q8_0_pipeline = make_pipeline(MATVEC_Q8_0_SHADER, "matvec_q8_0", "matvec_q8_0");
        let matvec_q1_0_pipeline = make_pipeline(MATVEC_Q1_0_SHADER, "matvec_q1_0", "matvec_q1_0");
        let matvec_q1_0_batch4_pipeline = make_pipeline(
            MATVEC_Q1_0_BATCH4_SHADER,
            "matvec_q1_0_batch4",
            "matvec_q1_0_batch4",
        );
        let matvec_q1_0_row4_pipeline = make_pipeline(
            MATVEC_Q1_0_ROW4_SHADER,
            "matvec_q1_0_row4",
            "matvec_q1_0_row4",
        );
        let matvec_q1_0_row4_batch4_pipeline = make_pipeline(
            MATVEC_Q1_0_ROW4_BATCH4_SHADER,
            "matvec_q1_0_row4_batch4",
            "matvec_q1_0_row4_batch4",
        );
        let matvec_q1_0_row8_batch4_pipeline = make_pipeline(
            MATVEC_Q1_0_ROW8_BATCH4_SHADER,
            "matvec_q1_0_row8_batch4",
            "matvec_q1_0_row8_batch4",
        );
        let swiglu_fused_q1_0_pipeline = make_pipeline(
            SWIGLU_FUSED_Q1_0_SHADER,
            "swiglu_fused_q1_0",
            "swiglu_fused_q1_0",
        );
        let swiglu_fused_q4k_pipeline = make_pipeline(
            SWIGLU_FUSED_Q4K_SHADER,
            "swiglu_fused_q4k",
            "swiglu_fused_q4k",
        );

        // Batch4 pipelines: share bind group layout with original pipelines
        // so existing bind groups work with both.
        let make_batch4 = |source: &str,
                           entry: &str,
                           label: &str,
                           base_pipeline: &wgpu::ComputePipeline|
         -> wgpu::ComputePipeline {
            let layout = base_pipeline.get_bind_group_layout(0);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&layout],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let matvec_q4k_batch4_pipeline = make_batch4(
            MATVEC_Q4K_BATCH4_SHADER,
            "matvec_q4k_batch4",
            "matvec_q4k_batch4",
            &matvec_q4k_pipeline,
        );
        let matvec_q6k_batch4_pipeline = make_batch4(
            MATVEC_Q6K_BATCH4_SHADER,
            "matvec_q6k_batch4",
            "matvec_q6k_batch4",
            &matvec_q6k_pipeline,
        );
        let swiglu_batch4_pipeline = make_batch4(
            SWIGLU_FUSED_Q4K_BATCH4_SHADER,
            "swiglu_fused_q4k_batch4",
            "swiglu_batch4",
            &swiglu_fused_q4k_pipeline,
        );

        Self {
            matvec_q4k_pipeline,
            matvec_q5k_pipeline,
            matvec_q6k_pipeline,
            matvec_q8_0_pipeline,
            matvec_q1_0_pipeline,
            matvec_q1_0_batch4_pipeline,
            matvec_q1_0_row4_pipeline,
            matvec_q1_0_row4_batch4_pipeline,
            matvec_q1_0_row8_batch4_pipeline,
            rmsnorm_pipeline: make_pipeline(RMSNORM_SHADER, "rmsnorm", "rmsnorm"),
            silu_mul_pipeline: make_pipeline(SILU_MUL_SHADER, "silu_mul", "silu_mul"),
            silu_gate_apply_pipeline: make_pipeline(
                SILU_GATE_APPLY_SHADER,
                "silu_gate_apply",
                "silu_gate_apply",
            ),
            sigmoid_gate_apply_pipeline: make_pipeline(
                SIGMOID_GATE_APPLY_SHADER,
                "sigmoid_gate_apply",
                "sigmoid_gate_apply",
            ),
            silu_inplace_pipeline: make_pipeline(
                SILU_INPLACE_SHADER,
                "silu_inplace",
                "silu_inplace",
            ),
            beta_sigmoid_pipeline: make_pipeline(
                BETA_SIGMOID_SHADER,
                "beta_sigmoid",
                "beta_sigmoid",
            ),
            ssm_discretisation_pipeline: make_pipeline(
                SSM_DISCRETISATION_SHADER,
                "ssm_discretisation",
                "ssm_discretisation",
            ),
            residual_add_pipeline: make_pipeline(
                RESIDUAL_ADD_SHADER,
                "residual_add",
                "residual_add",
            ),
            add_bias_pipeline: make_pipeline(ADD_BIAS_SHADER, "add_bias", "add_bias"),
            qk_norm_pipeline: make_pipeline(QK_NORM_SHADER, "qk_norm", "qk_norm"),
            rope_pipeline: make_pipeline(ROPE_SHADER, "rope", "rope"),
            attention_pipeline: make_pipeline(ATTENTION_SHADER, "attention", "attention"),
            kv_cache_append_pipeline: make_pipeline(
                KV_CACHE_APPEND_SHADER,
                "kv_cache_append",
                "kv_cache_append",
            ),
            swiglu_fused_q4k_pipeline,
            swiglu_fused_q1_0_pipeline,
            matvec_q4k_batch4_pipeline,
            matvec_q6k_batch4_pipeline,
            swiglu_batch4_pipeline,
            conv1d_causal_pipeline: make_pipeline(
                CONV1D_CAUSAL_SHADER,
                "conv1d_causal",
                "conv1d_causal",
            ),
            gated_deltanet_pipeline: make_pipeline(
                GATED_DELTANET_SHADER,
                "gated_deltanet",
                "gated_deltanet",
            ),
            timestamp_period,
            device_type,
            device,
            queue,
        }
    }

    // --- Buffer management ---

    /// Allocate a GPU buffer for `len` f32 elements.
    pub fn alloc_f32(&self, len: usize) -> GpuBuffer {
        GpuBuffer {
            buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (len * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            len,
        }
    }

    /// Upload f32 data to a new GPU buffer.
    pub fn upload_f32(&self, data: &[f32]) -> GpuBuffer {
        GpuBuffer {
            buffer: self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                }),
            len: data.len(),
        }
    }

    /// Upload Q4_K weight tensor to GPU.
    pub fn upload_weights(&self, data: &[u8], rows: usize, cols: usize) -> GpuWeightBuffer {
        self.upload_weights_typed(data, rows, cols, GpuQuantType::Q4K)
    }

    /// Upload Q5_K weight tensor to GPU. Layout is 176 bytes / 256 elements
    /// per block — u32-aligned so no repacking is needed.
    pub fn upload_weights_q5k(&self, data: &[u8], rows: usize, cols: usize) -> GpuWeightBuffer {
        self.upload_weights_typed(data, rows, cols, GpuQuantType::Q5K)
    }

    /// Upload Q6_K weight tensor to GPU.
    pub fn upload_weights_q6k(&self, data: &[u8], rows: usize, cols: usize) -> GpuWeightBuffer {
        self.upload_weights_typed(data, rows, cols, GpuQuantType::Q6K)
    }

    /// Upload Q8_0 weight tensor to GPU. GGUF stores Q8_0 as 34-byte blocks
    /// which are not u32-aligned; we pad each block to 36 bytes (adding two
    /// zero bytes after `d`) so the WGSL shader can read the block layout as
    /// 9 u32 words. Overhead: 5.88 % memory per Q8_0 tensor (negligible for
    /// small projection matrices like `ssm_alpha` / `ssm_beta`).
    pub fn upload_weights_q8_0(&self, data: &[u8], rows: usize, cols: usize) -> GpuWeightBuffer {
        const RAW_BLOCK_BYTES: usize = 34;
        const PADDED_BLOCK_BYTES: usize = 36;
        assert!(
            cols.is_multiple_of(32),
            "Q8_0 requires cols to be a multiple of 32 (got {cols})"
        );
        let blocks_per_row = cols / 32;
        let total_blocks = rows * blocks_per_row;
        assert_eq!(
            data.len(),
            total_blocks * RAW_BLOCK_BYTES,
            "Q8_0 data length mismatch: rows={rows} cols={cols} \
             expected {} raw bytes, got {}",
            total_blocks * RAW_BLOCK_BYTES,
            data.len(),
        );
        let mut padded = vec![0u8; total_blocks * PADDED_BLOCK_BYTES];
        for b in 0..total_blocks {
            let src = &data[b * RAW_BLOCK_BYTES..(b + 1) * RAW_BLOCK_BYTES];
            let dst = &mut padded[b * PADDED_BLOCK_BYTES..b * PADDED_BLOCK_BYTES + 2];
            dst.copy_from_slice(&src[0..2]);
            // Bytes 2..3 of the padded block are zero (word 0 upper 16 bits).
            padded[b * PADDED_BLOCK_BYTES + 4..b * PADDED_BLOCK_BYTES + 36]
                .copy_from_slice(&src[2..34]);
        }
        GpuWeightBuffer {
            buffer: self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("q8_0_weights"),
                    contents: &padded,
                    usage: wgpu::BufferUsages::STORAGE,
                }),
            rows: rows as u32,
            cols: cols as u32,
            blocks_per_row: blocks_per_row as u32,
            quant: GpuQuantType::Q8_0,
        }
    }

    /// Upload Q1_0 (PrismML Bonsai binary g128) weight tensor to GPU.
    pub fn upload_weights_q1_0(&self, data: &[u8], rows: usize, cols: usize) -> GpuWeightBuffer {
        self.upload_weights_typed(data, rows, cols, GpuQuantType::Q1_0)
    }

    fn upload_weights_typed(
        &self,
        data: &[u8],
        rows: usize,
        cols: usize,
        quant: GpuQuantType,
    ) -> GpuWeightBuffer {
        // Q8_0 has a non-u32-aligned 34-byte block layout that must go
        // through `upload_weights_q8_0` for padding — reject here to catch
        // any accidental call path that skipped the padded helper.
        assert!(
            !matches!(quant, GpuQuantType::Q8_0),
            "upload_weights_typed does not support Q8_0 directly — use upload_weights_q8_0"
        );
        let (label, elements_per_block) = match quant {
            GpuQuantType::Q4K => ("q4k_weights", 256usize),
            GpuQuantType::Q5K => ("q5k_weights", 256),
            GpuQuantType::Q6K => ("q6k_weights", 256),
            GpuQuantType::Q8_0 => unreachable!("gated by assert above"),
            GpuQuantType::Q1_0 => ("q1_0_weights", 128),
        };
        GpuWeightBuffer {
            buffer: self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: data,
                    usage: wgpu::BufferUsages::STORAGE,
                }),
            rows: rows as u32,
            cols: cols as u32,
            blocks_per_row: (cols / elements_per_block) as u32,
            quant,
        }
    }

    /// Write f32 data into an existing GPU buffer.
    pub fn write_f32(&self, buf: &GpuBuffer, data: &[f32]) {
        self.queue
            .write_buffer(&buf.buffer, 0, bytemuck::cast_slice(data));
    }

    /// Read f32 data back from GPU (blocking).
    pub fn read_f32(&self, buf: &GpuBuffer) -> Vec<f32> {
        let size = (buf.len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&buf.buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));
        self.map_staging(&staging, buf.len)
    }

    // --- Command pass ---

    /// Begin a new GPU command pass. Chain operations, then call `execute()`.
    pub fn begin_pass(&self) -> GpuPass<'_> {
        GpuPass {
            engine: self,
            encoder: self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default()),
        }
    }

    // --- Standalone operations (for benchmarking) ---

    /// Single matvec with readback (convenience method).
    pub fn matvec(&self, weights: &GpuWeightBuffer, input: &[f32]) -> Vec<f32> {
        let input_buf = self.upload_f32(input);
        let output_buf = self.alloc_f32(weights.rows as usize);
        let mut pass = self.begin_pass();
        pass.matvec_q4k(weights, &input_buf, &output_buf);
        pass.execute();
        self.read_f32(&output_buf)
    }

    /// Create pre-allocated context for repeated matvec with readback.
    pub fn create_matvec_context(&self, weights: &GpuWeightBuffer) -> MatvecContext {
        let input_buf = self.alloc_f32(weights.cols as usize);
        let output_buf = self.alloc_f32(weights.rows as usize);
        let output_size = (weights.rows as u64) * 4;
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ctx_staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = MatvecParams {
            rows: weights.rows,
            cols: weights.cols,
            blocks_per_row: weights.blocks_per_row,
            grid_x: weights.rows.min(65535),
            batch_size: 1,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ctx_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let layout = self.matvec_q4k_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ctx_bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weights.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        MatvecContext {
            input_buf: input_buf.buffer,
            output_buf: output_buf.buffer,
            staging_buf,
            bind_group,
            rows: weights.rows,
        }
    }

    /// Fast matvec using pre-allocated context (with readback).
    pub fn matvec_fast(&self, ctx: &MatvecContext, input: &[f32]) -> Vec<f32> {
        self.queue
            .write_buffer(&ctx.input_buf, 0, bytemuck::cast_slice(input));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.matvec_q4k_pipeline);
            pass.set_bind_group(0, &ctx.bind_group, &[]);
            pass.dispatch_workgroups(ctx.rows, 1, 1);
        }
        let size = (ctx.rows as u64) * 4;
        encoder.copy_buffer_to_buffer(&ctx.output_buf, 0, &ctx.staging_buf, 0, size);
        self.queue.submit(Some(encoder.finish()));
        self.map_staging(&ctx.staging_buf, ctx.rows as usize)
    }

    /// Dispatch without readback (for benchmarking).
    pub fn dispatch_matvec_only(&self, ctx: &MatvecContext, input: &[f32]) {
        self.queue
            .write_buffer(&ctx.input_buf, 0, bytemuck::cast_slice(input));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.matvec_q4k_pipeline);
            pass.set_bind_group(0, &ctx.bind_group, &[]);
            pass.dispatch_workgroups(ctx.rows, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Batch N matvec dispatches in one command buffer (for benchmarking).
    pub fn bench_dispatch_batch(&self, ctx: &MatvecContext, input: &[f32], n: u32) {
        self.queue
            .write_buffer(&ctx.input_buf, 0, bytemuck::cast_slice(input));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.matvec_q4k_pipeline);
            pass.set_bind_group(0, &ctx.bind_group, &[]);
            for _ in 0..n {
                pass.dispatch_workgroups(ctx.rows, 1, 1);
            }
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }

    /// Wait for all GPU work to complete.
    pub fn sync(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }

    // --- Internal helpers ---

    fn make_uniform<T: bytemuck::Pod>(&self, data: &T) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(data),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Create a uniform buffer that can be updated via queue.write_buffer.
    fn make_persistent_uniform<T: bytemuck::Pod>(&self, data: &T) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn map_staging(&self, staging: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("channel closed").expect("map failed");
        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data)[..n].to_vec();
        drop(data);
        staging.unmap();
        result
    }
}

// --- GpuPass implementation ---

/// Compute 2D dispatch dimensions for row count that may exceed 65535.
fn matvec_dispatch(rows: u32) -> (u32, u32, u32) {
    if rows <= 65535 {
        (rows, 1, rows)
    } else {
        let grid_x = 65535u32;
        let grid_y = (rows + grid_x - 1) / grid_x;
        (grid_x, grid_y, grid_x)
    }
}

impl<'a> GpuPass<'a> {
    /// Q4_K matvec: output = weights × input (GPU buffers, no readback).
    /// PrismML `Q1_0` (Bonsai 27B binary g128) GPU matvec — one workgroup per
    /// output row, workgroup_size 128 = one thread per element in a Q1_0 block.
    /// Uses byte-level indexing inside the shader because 18 bytes/block does
    /// not align to `u32` word boundaries.
    pub fn matvec_q1_0(
        &mut self,
        weights: &GpuWeightBuffer,
        input: &GpuBuffer,
        output: &GpuBuffer,
    ) {
        let (dispatch_x, dispatch_y, grid_x) = matvec_dispatch(weights.rows);
        let params = self.engine.make_uniform(&MatvecParams {
            rows: weights.rows,
            cols: weights.cols,
            blocks_per_row: weights.blocks_per_row,
            grid_x,
            batch_size: 1,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        });
        let layout = self.engine.matvec_q1_0_pipeline.get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: weights.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.matvec_q1_0_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
    }

    /// Q1_0 batch-4 GPU matvec — same weight tensor, 4 input vectors of size
    /// `weights.cols` packed back-to-back, 4 output vectors of size
    /// `weights.rows` packed back-to-back. Halves per-token wgpu dispatch
    /// overhead when combined with 4-token speculative decode.
    pub fn matvec_q1_0_batch4(
        &mut self,
        weights: &GpuWeightBuffer,
        input: &GpuBuffer,
        output: &GpuBuffer,
    ) {
        let (dispatch_x, dispatch_y, grid_x) = matvec_dispatch(weights.rows);
        let params = self.engine.make_uniform(&MatvecParams {
            rows: weights.rows,
            cols: weights.cols,
            blocks_per_row: weights.blocks_per_row,
            grid_x,
            batch_size: 4,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        });
        let layout = self
            .engine
            .matvec_q1_0_batch4_pipeline
            .get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: weights.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.matvec_q1_0_batch4_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
    }

    /// Q1_0 row4×batch4 GPU matvec — 1 workgroup produces 4 rows × 4 batches
    /// = 16 outputs. Compounds `matvec_q1_0_batch4` with 4× fewer workgroups
    /// (input reads are shared across the 4 output rows), buying an extra
    /// 4× reduction in total input-read traffic on top of the batch-4 weight
    /// decode amortization.
    ///
    /// Input / output layout matches `matvec_q1_0_batch4`:
    ///   input[batch * cols  + col]   for batch ∈ [0..4)
    ///   output[batch * rows + row]   for batch ∈ [0..4), row ∈ [0..rows)
    ///
    /// The shader itself clamps to `params.rows` internally, so callers that
    /// pass a row count that isn't a multiple of 4 still produce correct
    /// output (the trailing 1..3 rows of the last row-group are guarded).
    /// Q1_0 row4 GPU matvec (single batch) — 1 workgroup produces 4 rows × 1
    /// batch = 4 outputs. The natural companion to `matvec_q1_0_row4_batch4`
    /// for the standard batch=1 decode path (where speculative decoding is
    /// not in use).
    ///
    /// Same workgroup-count amortization as row4×batch4 (4× fewer workgroups
    /// than the base `matvec_q1_0` single-row kernel), same 4× reduction in
    /// total input-read traffic (input is shared across 4 output rows within
    /// a workgroup), but without the batch dimension.
    ///
    /// Input / output layout matches `matvec_q1_0`:
    ///   input[col]                 for col ∈ [0..cols)
    ///   output[row]                for row ∈ [0..rows)
    pub fn matvec_q1_0_row4(
        &mut self,
        weights: &GpuWeightBuffer,
        input: &GpuBuffer,
        output: &GpuBuffer,
    ) {
        let row_groups = weights.rows.div_ceil(4);
        let (dispatch_x, dispatch_y, grid_x) = matvec_dispatch(row_groups);
        let params = self.engine.make_uniform(&MatvecParams {
            rows: weights.rows,
            cols: weights.cols,
            blocks_per_row: weights.blocks_per_row,
            grid_x,
            batch_size: 1,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        });
        let layout = self
            .engine
            .matvec_q1_0_row4_pipeline
            .get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: weights.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.matvec_q1_0_row4_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
    }

    /// Q1_0 row8×batch4 GPU matvec — 1 workgroup produces 8 rows × 4 batches
    /// = 32 outputs. Compounds `matvec_q1_0_row4_batch4` by doubling the row
    /// batching factor, dropping workgroup count another 2× and total
    /// input-read traffic another 2× on top of Step 4's row4×batch4.
    ///
    /// Register cost: 32 f32 accumulators per thread (scalar named vars,
    /// no dynamic indexing) — verified compiles clean on Vulkan iGPU
    /// (Jetson Orin Nano, Ampere) and Metal (Apple Silicon).
    ///
    /// Input / output layout matches `matvec_q1_0_row4_batch4`:
    ///   input[batch * cols  + col]   for batch ∈ [0..4)
    ///   output[batch * rows + row]   for batch ∈ [0..4), row ∈ [0..rows)
    pub fn matvec_q1_0_row8_batch4(
        &mut self,
        weights: &GpuWeightBuffer,
        input: &GpuBuffer,
        output: &GpuBuffer,
    ) {
        // 8 rows per workgroup → workgroup count is ceil(rows / 8).
        let row_groups = weights.rows.div_ceil(8);
        let (dispatch_x, dispatch_y, grid_x) = matvec_dispatch(row_groups);
        let params = self.engine.make_uniform(&MatvecParams {
            rows: weights.rows,
            cols: weights.cols,
            blocks_per_row: weights.blocks_per_row,
            grid_x,
            batch_size: 4,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        });
        let layout = self
            .engine
            .matvec_q1_0_row8_batch4_pipeline
            .get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: weights.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.matvec_q1_0_row8_batch4_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
    }

    pub fn matvec_q1_0_row4_batch4(
        &mut self,
        weights: &GpuWeightBuffer,
        input: &GpuBuffer,
        output: &GpuBuffer,
    ) {
        // 4 rows per workgroup → workgroup count is ceil(rows / 4).
        let row_groups = weights.rows.div_ceil(4);
        let (dispatch_x, dispatch_y, grid_x) = matvec_dispatch(row_groups);
        let params = self.engine.make_uniform(&MatvecParams {
            rows: weights.rows,
            cols: weights.cols,
            blocks_per_row: weights.blocks_per_row,
            grid_x,
            batch_size: 4,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        });
        let layout = self
            .engine
            .matvec_q1_0_row4_batch4_pipeline
            .get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: weights.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.matvec_q1_0_row4_batch4_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
    }

    pub fn matvec_q4k(&mut self, weights: &GpuWeightBuffer, input: &GpuBuffer, output: &GpuBuffer) {
        let (dispatch_x, dispatch_y, grid_x) = matvec_dispatch(weights.rows);
        let params = self.engine.make_uniform(&MatvecParams {
            rows: weights.rows,
            cols: weights.cols,
            blocks_per_row: weights.blocks_per_row,
            grid_x,
            batch_size: 1,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        });
        let layout = self.engine.matvec_q4k_pipeline.get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: weights.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.matvec_q4k_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
    }

    /// Bonsai DeltaNet per-V-head RMSNorm on `dn_delta_out`, applying
    /// `ssm_norm` weight (broadcast [v_dim]) across all V heads.
    ///
    /// Phase X.3.e.3 Gap C (see commit `005b3d0` for CPU implementation via
    /// `apply_qk_norm`). Applied before `ssm_out` projection in the Bonsai
    /// DeltaNet forward path.
    ///
    /// Reuses the existing `rmsnorm_pipeline` (which already supports batched
    /// dispatch via `wid.x = batch_idx`) by treating each V head as a "batch"
    /// element: `params.dim = v_dim`, dispatch `num_v_heads` workgroups. This
    /// is exactly equivalent to the per-head loop in `llama3::apply_qk_norm`.
    ///
    /// Input layout: `[num_v_heads * v_dim]` (V heads concatenated).
    /// Output layout: same. Must be a separate buffer from input.
    ///
    /// Not yet wired into `encode_forward_impl` — awaiting Phase 2 Steps
    /// 1+4 (Bonsai loader + forward path wiring).
    pub fn ssm_norm_per_head(
        &mut self,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        output: &GpuBuffer,
        v_dim: u32,
        num_v_heads: u32,
        eps: f32,
    ) {
        let params = self.engine.make_uniform(&RmsnormParams {
            dim: v_dim,
            eps,
            batch_size: num_v_heads,
            _pad3: 0,
        });
        let layout = self.engine.rmsnorm_pipeline.get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: weight.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.rmsnorm_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(num_v_heads, 1, 1);
        }
    }

    /// RMSNorm: output[i] = input[i] * weight[i] * rsqrt(mean(input^2) + eps).
    pub fn rmsnorm(&mut self, input: &GpuBuffer, weight: &GpuBuffer, output: &GpuBuffer, eps: f32) {
        let params = self.engine.make_uniform(&RmsnormParams {
            dim: input.len as u32,
            eps,
            batch_size: 1,
            _pad3: 0,
        });
        let layout = self.engine.rmsnorm_pipeline.get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: weight.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.rmsnorm_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    /// SiLU(gate) * up — in-place on gate buffer.
    pub fn silu_mul(&mut self, gate: &GpuBuffer, up: &GpuBuffer) {
        let layout = self.engine.silu_mul_pipeline.get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: gate.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: up.buffer.as_entire_binding(),
                    },
                ],
            });
        let dispatch_x = ((gate.len as u32) + 255) / 256;
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.silu_mul_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }
    }

    /// Element-wise SiLU in-place: `buf[i] = buf[i] * sigmoid(buf[i])`.
    ///
    /// Bonsai / Qwen 3.6-27B DeltaNet post-conv1d silu (Phase X.3.e.3 Gap A):
    /// applied to the causal-conv1d output for the Q+K+V portion before
    /// delta-rule integration. Distinct from `silu_mul` (SwiGLU two-input)
    /// and `silu_gate_apply` (Bonsai attn output modulation).
    ///
    /// Not yet wired into `encode_forward_impl` — awaiting Phase 2 Step 4
    /// (Bonsai DeltaNet forward path wiring).
    pub fn silu_inplace(&mut self, buf: &GpuBuffer) {
        let layout = self.engine.silu_inplace_pipeline.get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.buffer.as_entire_binding(),
                }],
            });
        let dispatch_x = ((buf.len as u32) + 255) / 256;
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.silu_inplace_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }
    }

    /// Bonsai DeltaNet beta transformation: `beta[i] = sigmoid(beta_raw[i])`
    /// in-place on beta buffer.
    ///
    /// Phase X.3.e.3 Gap B extra (see commit `2d55f2b` for CPU reference).
    /// The DeltaNet recurrence's update rate `beta` must be constrained to
    /// (0, 1) for the delta rule to remain stable. Bonsai / Qwen 3.6-27B
    /// applies sigmoid to bring raw `beta_proj` output into the valid range.
    ///
    /// Not yet wired into `encode_forward_impl` — awaiting Phase 2 Steps
    /// 1+4 (Bonsai loader + forward path wiring). Exposed publicly for
    /// standalone parity validation and future integration.
    pub fn beta_sigmoid(&mut self, beta: &GpuBuffer) {
        let layout = self.engine.beta_sigmoid_pipeline.get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: beta.buffer.as_entire_binding(),
                }],
            });
        let dispatch_x = ((beta.len as u32) + 255) / 256;
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.beta_sigmoid_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }
    }

    /// Bonsai DeltaNet SSM discretisation: writes per-V-head decay factor
    /// into `alpha` buffer in-place, consuming raw alpha projection +
    /// `ssm_dt_bias` + `ssm_a`.
    ///
    /// Computes `decay[h] = exp(softplus(alpha[h] + ssm_dt_bias[h]) * ssm_a[h])`.
    /// Phase X.3.e.3 Gap B (see commit `6d43602` for CPU reference).
    /// Reference: llama.cpp PrismML fork `qwen35.cpp:443-451`.
    ///
    /// Mamba convention: `ssm_a` stored ≈ -exp(A_log) < 0 → gate < 0 →
    /// decay ∈ (0, 1] (real decay factor). Softplus is numerically stable
    /// for |x| > 20 via closed-form approximation.
    ///
    /// Not yet wired into `encode_forward_impl` — awaiting Phase 2 Steps
    /// 1+4 (Bonsai loader + forward path wiring).
    pub fn ssm_discretisation(
        &mut self,
        alpha: &GpuBuffer,
        ssm_dt_bias: &GpuBuffer,
        ssm_a: &GpuBuffer,
    ) {
        let layout = self
            .engine
            .ssm_discretisation_pipeline
            .get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: alpha.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ssm_dt_bias.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ssm_a.buffer.as_entire_binding(),
                    },
                ],
            });
        let dispatch_x = ((alpha.len as u32) + 255) / 256;
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.ssm_discretisation_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }
    }

    /// Bonsai gated attention output modulation: `attn_out *= silu(gate)`
    /// in-place on `attn_out` buffer.
    ///
    /// Distinct from `silu_mul` (which writes to `gate` for SwiGLU FFN): here
    /// the write target is `attn_out` and the silu-transformed operand is
    /// `gate` (read-only). Used by Bonsai / Qwen 3.6-27B full-attention
    /// layers where `attn_q [q_dim*2, hidden]` produces both Q and the swish
    /// gate; after standard scaled-dot-product attention completes, the raw
    /// output is modulated by silu(gate) before the `o_proj` matvec.
    ///
    /// Not yet wired into `encode_forward_impl` — awaiting Phase 2 Steps 3-6
    /// (see `docs/BONSAI_GPU_SUPPORT.md`). Exposed publicly for standalone
    /// parity validation against the CPU reference and for future forward
    /// path integration.
    pub fn silu_gate_apply(&mut self, attn_out: &GpuBuffer, gate: &GpuBuffer) {
        let layout = self
            .engine
            .silu_gate_apply_pipeline
            .get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: attn_out.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: gate.buffer.as_entire_binding(),
                    },
                ],
            });
        let dispatch_x = ((attn_out.len as u32) + 255) / 256;
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.silu_gate_apply_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }
    }

    /// Residual add: inout[i] += addend[i].
    pub fn residual_add(&mut self, inout: &GpuBuffer, addend: &GpuBuffer) {
        let layout = self.engine.residual_add_pipeline.get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: inout.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: addend.buffer.as_entire_binding(),
                    },
                ],
            });
        let dispatch_x = ((inout.len as u32) + 255) / 256;
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.residual_add_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }
    }

    /// RoPE rotation: in-place on vec buffer.
    pub fn rope(
        &mut self,
        vec_buf: &GpuBuffer,
        position: u32,
        num_heads: u32,
        head_dim: u32,
        theta: f32,
    ) {
        let params = self.engine.make_uniform(&RopeParams {
            position,
            num_heads,
            head_dim,
            theta,
            batch_size: 1,
            // Default LLAMA-style — this path is only used by the legacy
            // one-shot `rope` benchmark method and does not carry Qwen
            // routing. NEOX callers should set this via the persistent
            // RopeParams buffer created in `GpuModel::load`.
            neox: 0,
            _pad2: 0,
            _pad3: 0,
        });
        let layout = self.engine.rope_pipeline.get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: vec_buf.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        let total_pairs = num_heads * head_dim / 2;
        let dispatch_x = (total_pairs + 255) / 256;
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.rope_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }
    }

    /// Q6_K matvec: output = weights × input (GPU buffers, no readback).
    /// Q5_K matvec (standalone). Same bind group layout as Q4_K, only pipeline
    /// differs — see `matvec_q5k` shader for the block-layout math.
    pub fn matvec_q5k(&mut self, weights: &GpuWeightBuffer, input: &GpuBuffer, output: &GpuBuffer) {
        let (dispatch_x, dispatch_y, grid_x) = matvec_dispatch(weights.rows);
        let params = self.engine.make_uniform(&MatvecParams {
            rows: weights.rows,
            cols: weights.cols,
            blocks_per_row: weights.blocks_per_row,
            grid_x,
            batch_size: 1,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        });
        let layout = self.engine.matvec_q5k_pipeline.get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: weights.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.matvec_q5k_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
    }

    /// Q8_0 matvec (standalone). Same bind group layout as Q4_K, only pipeline
    /// differs — see `matvec_q8_0` shader for the padded 9-word block layout.
    pub fn matvec_q8_0(
        &mut self,
        weights: &GpuWeightBuffer,
        input: &GpuBuffer,
        output: &GpuBuffer,
    ) {
        let (dispatch_x, dispatch_y, grid_x) = matvec_dispatch(weights.rows);
        let params = self.engine.make_uniform(&MatvecParams {
            rows: weights.rows,
            cols: weights.cols,
            blocks_per_row: weights.blocks_per_row,
            grid_x,
            batch_size: 1,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        });
        let layout = self.engine.matvec_q8_0_pipeline.get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: weights.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.matvec_q8_0_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
    }

    pub fn matvec_q6k(&mut self, weights: &GpuWeightBuffer, input: &GpuBuffer, output: &GpuBuffer) {
        let (dispatch_x, dispatch_y, grid_x) = matvec_dispatch(weights.rows);
        let params = self.engine.make_uniform(&MatvecParams {
            rows: weights.rows,
            cols: weights.cols,
            blocks_per_row: weights.blocks_per_row,
            grid_x,
            batch_size: 1,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        });
        let layout = self.engine.matvec_q6k_pipeline.get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: weights.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.matvec_q6k_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
    }

    /// Dispatch matvec using the weight buffer's quantization type (Q4_K or Q6_K).
    pub fn matvec_auto(
        &mut self,
        weights: &GpuWeightBuffer,
        input: &GpuBuffer,
        output: &GpuBuffer,
    ) {
        match weights.quant {
            GpuQuantType::Q4K => self.matvec_q4k(weights, input, output),
            GpuQuantType::Q5K => self.matvec_q5k(weights, input, output),
            GpuQuantType::Q6K => self.matvec_q6k(weights, input, output),
            GpuQuantType::Q8_0 => self.matvec_q8_0(weights, input, output),
            GpuQuantType::Q1_0 => self.matvec_q1_0(weights, input, output),
        }
    }

    /// GQA attention: q_buf × k_cache → softmax → × v_cache → out_buf.
    pub fn attention(
        &mut self,
        q_buf: &GpuBuffer,
        k_cache: &GpuBuffer,
        v_cache: &GpuBuffer,
        out_buf: &GpuBuffer,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) {
        let kv_dim = num_kv_heads * head_dim;
        let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();
        let params = self.engine.make_uniform(&AttentionParams {
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_dim,
            inv_sqrt_d,
            batch_size: 1,
            _pad2: 0,
        });
        let layout = self.engine.attention_pipeline.get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: q_buf.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: k_cache.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: v_cache.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_buf.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.attention_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(num_heads, 1, 1);
        }
    }

    /// Append K,V vectors to their cache positions.
    pub fn kv_cache_append(
        &mut self,
        k_in: &GpuBuffer,
        v_in: &GpuBuffer,
        k_cache: &GpuBuffer,
        v_cache: &GpuBuffer,
        position: u32,
        kv_dim: u32,
    ) {
        let params = self.engine.make_uniform(&KvCacheParams {
            position,
            kv_dim,
            batch_size: 1,
            _pad2: 0,
        });
        let layout = self
            .engine
            .kv_cache_append_pipeline
            .get_bind_group_layout(0);
        let bg = self
            .engine
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: k_in.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: v_in.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: k_cache.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: v_cache.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
        let dispatch_x = (kv_dim + 255) / 256;
        {
            let mut pass = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.engine.kv_cache_append_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(dispatch_x, 1, 1);
        }
    }

    /// Submit all encoded operations and wait for completion.
    pub fn execute(self) {
        self.engine.queue.submit(Some(self.encoder.finish()));
        self.engine.device.poll(wgpu::Maintain::Wait);
    }

    /// Submit and read back one buffer's contents.
    pub fn execute_and_read(mut self, buf: &GpuBuffer) -> Vec<f32> {
        let size = (buf.len * 4) as u64;
        let staging = self.engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.encoder
            .copy_buffer_to_buffer(&buf.buffer, 0, &staging, 0, size);
        self.engine.queue.submit(Some(self.encoder.finish()));
        self.engine.map_staging(&staging, buf.len)
    }
}

// === GpuModel: zero per-token allocation forward pass ===

/// Configuration for GpuModel.
pub struct GpuModelConfig {
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub rope_theta: f32,
    pub eps: f32,
    pub max_seq_len: usize,
    /// Qwen3.5 DeltaNet: full attention interval (None = all attention).
    pub full_attention_interval: Option<usize>,
    /// Qwen3.5 DeltaNet: number of QK heads for linear attention layers.
    pub linear_num_kv_heads: Option<u32>,
    /// Qwen3.5 DeltaNet: QK head dimension.
    pub linear_qk_head_dim: Option<u32>,
    /// Qwen3.5 DeltaNet: V head dimension.
    pub linear_kv_head_dim: Option<u32>,
    /// Qwen3.5 DeltaNet: number of V heads.
    pub linear_num_v_heads: Option<u32>,
    /// Qwen3.5 DeltaNet: causal conv1d kernel size (typically 4).
    pub linear_conv_kernel_dim: Option<u32>,
    /// RoPE rotation convention. `false` = LLAMA-style (pair `(i, i+1)`),
    /// `true` = NEOX-style (pair `(i, i + head_dim/2)`). Qwen 2/3, Gemma 2
    /// require NEOX; Llama/Mistral/Gemma 4 use LLAMA-style. Mirrors
    /// `Llama3Config::is_neox_rope()` on the CPU side.
    pub neox_rope: bool,
}

/// Pre-cached matvec bind group with dispatch dimensions.
struct MatvecBG {
    bg: wgpu::BindGroup,
    dispatch_x: u32,
    dispatch_y: u32,
    quant: GpuQuantType,
}

/// Pre-cached SwiGLU fused bind group (gate + up + silu×mul in one kernel).
struct SwigluBG {
    bg: wgpu::BindGroup,
    dispatch_x: u32,
    dispatch_y: u32,
    /// Selects the fused SwiGLU pipeline: Q4_K uses `swiglu_fused_q4k`
    /// (workgroup_size 256, one row per workgroup, 256-elem block), Q1_0
    /// uses `swiglu_fused_q1_0` (workgroup_size 128, byte-level 18-byte
    /// block indexing). Q6_K and Q5_K FFN weights fall back to the un-fused
    /// path — none of the shipping Bonsai / Qwen 3.5 GGUFs use those
    /// quants for `ffn_gate` / `ffn_up` so the fused path covers the
    /// production workload.
    quant: GpuQuantType,
}

/// Per-layer pre-cached bind groups.
struct LayerBGs {
    attn_norm_bg: wgpu::BindGroup,
    q_proj: MatvecBG,
    k_proj: MatvecBG,
    v_proj: MatvecBG,
    o_proj: MatvecBG,
    /// Bonsai / Qwen 3.6-27B full-attention layer swish gate projection.
    /// Same shape as `q_proj` ([q_dim, hidden]) — it's the second half of
    /// the Bonsai `attn_q [q_dim*2, hidden]` tensor, split at load time.
    /// The gate output is silu-multiplied into `attn_out` after the
    /// scaled-dot-product attention, before the `o_proj` matvec.
    /// `None` on standard Qwen 3.5 / Llama / Qwen 2/3 / Gemma layers.
    bonsai_attn_q_gate_proj: Option<MatvecBG>,
    /// Bonsai / Qwen 3.6-27B full-attention swish gate modulation:
    /// `silu_gate_apply(attn_out, gate_buf)` applied after standard
    /// scaled-dot-product attention, before `o_proj`. Reuses the FFN SwiGLU
    /// `gate_buf` (size `intermediate_dim` ≥ `hidden_dim`) — the shader's
    /// `arrayLength(&attn_out)` guard confines the iteration to the actual
    /// gate output length. `None` on non-Bonsai layers.
    bonsai_full_attn_silu_gate_bg: Option<wgpu::BindGroup>,
    /// Qwen 2 / 2.5 QKV bias-add bind groups (None for arch without biases).
    q_bias_bg: Option<wgpu::BindGroup>,
    k_bias_bg: Option<wgpu::BindGroup>,
    v_bias_bg: Option<wgpu::BindGroup>,
    /// Qwen 3 per-head QK RMSNorm bind groups (None for arch without QK norm).
    q_norm_bg: Option<wgpu::BindGroup>,
    k_norm_bg: Option<wgpu::BindGroup>,
    kv_append_bg: wgpu::BindGroup,
    attention_bg: wgpu::BindGroup,
    ffn_norm_bg: wgpu::BindGroup,
    swiglu: SwigluBG, // fused gate + up + silu×mul
    down_proj: MatvecBG,
}

/// Weight buffer ownership (bind groups reference these internally).
struct LayerWeightBufs {
    attn_norm: GpuBuffer,
    q_proj: GpuWeightBuffer,
    k_proj: GpuWeightBuffer,
    v_proj: GpuWeightBuffer,
    o_proj: GpuWeightBuffer,
    /// Bonsai / Qwen 3.6-27B full-attention swish gate weights. Shape
    /// [q_dim, hidden] — the second half of Bonsai's `attn_q [q_dim*2, hidden]`
    /// after load-time byte-level split. `None` on non-Bonsai layers.
    bonsai_attn_q_gate: Option<GpuWeightBuffer>,
    /// Qwen 2 / 2.5 attention QKV projection biases (absent for Llama / Mistral /
    /// Gemma / Qwen 3+ which removed biases). Added element-wise to q/k/v projection
    /// outputs before RoPE.
    q_bias: Option<GpuBuffer>,
    k_bias: Option<GpuBuffer>,
    v_bias: Option<GpuBuffer>,
    /// Qwen 3 per-head Q / K RMSNorm weights (`head_dim` elements each, shared across heads).
    /// Absent for Llama / Mistral / Gemma / Qwen 2 / Qwen 2.5.
    q_norm: Option<GpuBuffer>,
    k_norm: Option<GpuBuffer>,
    ffn_norm: GpuBuffer,
    gate_proj: GpuWeightBuffer,
    up_proj: GpuWeightBuffer,
    down_proj: GpuWeightBuffer,
}

// ─── DeltaNet (Qwen3.5 Gated DeltaNet) layer structures ───────────────���────

/// DeltaNet layer weight buffers (for linear attention layers in Qwen3.5).
#[allow(dead_code)]
struct DeltaNetLayerWeightBufs {
    attn_norm: GpuBuffer,
    /// Standard Qwen 3.5 fused in_proj for q, k, v, z: `[qk_dim*2 + v_dim*2, hidden]`.
    /// `None` on Bonsai / Qwen 3.6-27B (which splits this into `attn_qkv` +
    /// `attn_gate`); exactly one of `ssm_in` or `attn_qkv` is `Some` per layer.
    ssm_in: Option<GpuWeightBuffer>,
    /// Bonsai / Qwen 3.6-27B DeltaNet Q+K+V fused projection:
    /// `[qk_dim * num_kv_heads * 2 + v_dim * num_v_heads, hidden]`
    /// (10240 rows for Bonsai 27B). Together with `attn_gate` replaces the
    /// standard Qwen 3.5 `ssm_in`.
    attn_qkv: Option<GpuWeightBuffer>,
    /// Bonsai / Qwen 3.6-27B DeltaNet Z (output-gate) projection:
    /// `[v_dim * num_v_heads, hidden]` (6144 rows for Bonsai 27B).
    attn_gate: Option<GpuWeightBuffer>,
    /// Bonsai / Qwen 3.6 learnable SSM state-transition parameter, `[num_v_heads]` f32,
    /// stored ≈ `-exp(A_log)` (Mamba convention, negative). Consumed by
    /// `GpuPass::ssm_discretisation` in the Bonsai DeltaNet forward path.
    ssm_a: Option<GpuBuffer>,
    /// Bonsai / Qwen 3.6 SSM discretisation-step bias, `[num_v_heads]` f32.
    /// Added to raw alpha projection before softplus, together with `ssm_a`.
    ssm_dt_bias: Option<GpuBuffer>,
    /// Bonsai / Qwen 3.6 SSM state RMSNorm weight, `[v_dim]` f32. Broadcast
    /// across V heads and applied to `dn_delta_out` via
    /// `GpuPass::ssm_norm_per_head` in the Bonsai DeltaNet forward path.
    ssm_norm: Option<GpuBuffer>,
    /// Causal conv1d kernel: `[kernel_size, conv_dim]`
    conv1d_weight: GpuBuffer,
    /// Causal conv1d bias: `[conv_dim]`. `None` on Bonsai (which omits the
    /// bias tensor); Standard Qwen 3.5 GGUFs always have it.
    conv1d_bias: Option<GpuBuffer>,
    /// Alpha (decay gate) projection
    alpha_proj: GpuWeightBuffer,
    /// Beta (update rate) projection
    beta_proj: GpuWeightBuffer,
    /// Output projection
    ssm_out: GpuWeightBuffer,
    ffn_norm: GpuBuffer,
    gate_proj: GpuWeightBuffer,
    up_proj: GpuWeightBuffer,
    down_proj: GpuWeightBuffer,
}

/// DeltaNet layer bind groups.
#[allow(dead_code)]
struct DeltaNetLayerBGs {
    attn_norm_bg: wgpu::BindGroup,
    /// Standard Qwen 3.5 fused in_proj bind group. `None` on Bonsai / Qwen 3.6-27B.
    ssm_in_proj: Option<MatvecBG>,
    /// Bonsai / Qwen 3.6-27B fused Q+K+V projection bind group. `None` on
    /// standard Qwen 3.5.
    attn_qkv_proj: Option<MatvecBG>,
    /// Bonsai / Qwen 3.6-27B Z (output-gate) projection bind group. `None`
    /// on standard Qwen 3.5.
    attn_gate_proj: Option<MatvecBG>,
    /// Bonsai `beta_sigmoid` dispatch bind group: beta_buf in-place.
    /// `None` on standard Qwen 3.5.
    bonsai_beta_sigmoid_bg: Option<wgpu::BindGroup>,
    /// Bonsai `ssm_discretisation` dispatch bind group: alpha_buf +
    /// ssm_dt_bias + ssm_a. Per-layer because ssm_dt_bias and ssm_a are
    /// per-layer tensors. `None` on standard Qwen 3.5.
    bonsai_ssm_discretisation_bg: Option<wgpu::BindGroup>,
    /// Bonsai `silu_inplace` on conv output (Q+K+V portion of k_buf).
    /// `None` on standard Qwen 3.5.
    bonsai_silu_inplace_kbuf_bg: Option<wgpu::BindGroup>,
    /// Bonsai `ssm_norm_per_head` on attn_out with per-layer `ssm_norm`
    /// weight. Uses the shared `rmsnorm_pipeline`; per-layer because
    /// `ssm_norm` is a per-layer tensor. `None` on standard Qwen 3.5.
    bonsai_ssm_norm_per_head_bg: Option<wgpu::BindGroup>,
    /// Bonsai `silu_gate_apply(attn_out, v_buf)` where v_buf carries the
    /// z projection from `attn_gate_proj`. `None` on standard Qwen 3.5.
    bonsai_silu_gate_apply_bg: Option<wgpu::BindGroup>,
    /// alpha decay-gate projection: `norm_buf → alpha_buf` (`[num_kv_heads]`).
    alpha_proj_mv: MatvecBG,
    /// beta update-rate projection: `norm_buf → beta_buf` (`[num_kv_heads]`).
    beta_proj_mv: MatvecBG,
    conv1d_bg: wgpu::BindGroup,
    deltanet_bg: wgpu::BindGroup,
    ssm_out_proj: MatvecBG,
    ffn_norm_bg: wgpu::BindGroup,
    swiglu: SwigluBG,
    down_proj: MatvecBG,
}

/// Indicates the type of a layer (attention or DeltaNet).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    Attention,
    DeltaNet,
}

/// Pre-compiled GPU model with zero per-token allocations.
///
/// All `wgpu::BindGroup` and uniform buffers are created once at load time.
/// Per-token `forward()` only updates 4 persistent uniform buffers via
/// `queue.write_buffer` (4 bytes each) and encodes dispatches with
/// pre-cached bind groups — no dynamic allocation.
pub struct GpuModel {
    engine: Arc<GpuEngine>,
    config: GpuModelConfig,

    // Weight ownership (prevents GPU buffer deallocation)
    _layer_weights: Vec<LayerWeightBufs>,
    _output_norm_weight: GpuBuffer,
    _output_proj_weight: GpuWeightBuffer,

    // Scratch buffers (owned to keep GPU buffers alive for bind groups)
    hidden: GpuBuffer,
    _norm_buf: GpuBuffer,
    _q_buf: GpuBuffer,
    _k_buf: GpuBuffer,
    _v_buf: GpuBuffer,
    _attn_out: GpuBuffer,
    /// Post-`ssm_norm_per_head` scratch for the Bonsai DeltaNet path (see
    /// scratch alloc in `GpuModel::load`). Kept as an owned field so its
    /// storage buffer outlives the bind groups that reference it via
    /// `attn_out_normed.buffer`.
    _attn_out_normed: GpuBuffer,
    _o_buf: GpuBuffer,
    _gate_buf: GpuBuffer,
    _down_buf: GpuBuffer,
    // DeltaNet scratch: alpha decay-gate `[num_kv_heads]`, beta update-rate
    // `[num_kv_heads]`. Populated by `alpha_proj_mv` / `beta_proj_mv` on each
    // DeltaNet layer step. Kept as owned scratch to prevent GPU buffer
    // deallocation while `deltanet_bg` still references them.
    _alpha_buf: GpuBuffer,
    _beta_buf: GpuBuffer,
    logits: GpuBuffer,
    staging: wgpu::Buffer,
    /// Issue #40 diagnostic. Small staging buffer for reading `_norm_buf`
    /// (post-final-RMSNorm hidden state) back to CPU. Sized `hidden_dim * 4`.
    /// Lazily allocated the first time `forward_and_read_hidden_and_logits`
    /// is called.
    hidden_staging: std::cell::OnceCell<wgpu::Buffer>,

    // KV caches (attention layers only)
    k_caches: Vec<GpuBuffer>,
    v_caches: Vec<GpuBuffer>,

    // DeltaNet state (Qwen3.5 linear attention layers)
    #[allow(dead_code)]
    deltanet_states: Vec<GpuBuffer>, // [num_heads, qk_dim, v_dim] per DeltaNet layer
    #[allow(dead_code)]
    deltanet_conv_states: Vec<GpuBuffer>, // [3, conv_dim] per DeltaNet layer
    #[allow(dead_code)]
    _deltanet_layer_weights: Vec<DeltaNetLayerWeightBufs>,
    #[allow(dead_code)]
    deltanet_layer_bgs: Vec<DeltaNetLayerBGs>,
    /// Layer type map: layer_types[i] = Attention or DeltaNet
    layer_types: Vec<LayerType>,

    // Pre-cached bind groups (attention layers)
    layer_bgs: Vec<LayerBGs>,
    rope_q_bg: wgpu::BindGroup,
    rope_k_bg: wgpu::BindGroup,
    residual_attn_bg: wgpu::BindGroup,
    residual_ffn_bg: wgpu::BindGroup,
    output_norm_bg: wgpu::BindGroup,
    output_proj_bg: MatvecBG,

    // Persistent uniform buffers (updated per-token via write_buffer)
    rope_q_params_buf: wgpu::Buffer,
    rope_k_params_buf: wgpu::Buffer,
    kv_append_params_buf: wgpu::Buffer,
    attn_params_buf: wgpu::Buffer,
    /// DeltaNet conv1d params — layout `[dim, ring_pos, _pad, _pad]`.
    /// `ring_pos` is updated per-token to `pos % (kernel_size - 1)`
    /// (Phase X.3.e.3.25 fix). `None` when the model has no DeltaNet
    /// layers (e.g., pure-attention Qwen 2 / 2.5 / 3).
    dn_conv_params_buf: Option<wgpu::Buffer>,

    // Dispatch constants
    residual_dispatch_x: u32,
    rope_q_dispatch_x: u32,
    rope_k_dispatch_x: u32,
    kv_dispatch_x: u32,
    /// Q bias add dispatch: `ceil(hidden_dim / 256)`.
    q_bias_dispatch_x: u32,
    /// K / V bias add dispatch: `ceil(kv_dim / 256)`.
    kv_bias_dispatch_x: u32,

    // CPU data
    embedding: Vec<f32>,
    vocab_size: usize,

    // Auto-tracking state
    seq_len: u32,
}

/// Per-category GPU timing results (microseconds).
pub struct ProfileResult {
    pub rmsnorm_us: f64,
    pub matvec_q4k_us: f64,
    pub matvec_q6k_us: f64,
    pub swiglu_us: f64,
    pub rope_us: f64,
    pub kv_append_us: f64,
    pub attention_us: f64,
    pub residual_us: f64,
    pub rmsnorm_n: u32,
    pub matvec_q4k_n: u32,
    pub matvec_q6k_n: u32,
    pub swiglu_n: u32,
    pub rope_n: u32,
    pub kv_append_n: u32,
    pub attention_n: u32,
    pub residual_n: u32,
}

/// Submit a single dispatch, sync, return elapsed microseconds.
fn timed_dispatch(
    engine: &GpuEngine,
    pipeline: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    dx: u32,
    dy: u32,
    dz: u32,
) -> f64 {
    let mut encoder = engine
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cp.set_pipeline(pipeline);
        cp.set_bind_group(0, bg, &[]);
        cp.dispatch_workgroups(dx, dy, dz);
    }
    let t0 = std::time::Instant::now();
    engine.queue.submit(Some(encoder.finish()));
    engine.device.poll(wgpu::Maintain::Wait);
    t0.elapsed().as_nanos() as f64 / 1000.0
}

/// Timed matvec dispatch (auto-selects pipeline by quant type).
///
/// Q1_0 uses the row4 kernel to match the production `dispatch_mv` path
/// (Step 6, PR #74) — timing reflects the same kernel the forward pass
/// actually runs.
fn timed_mv(engine: &GpuEngine, mv: &MatvecBG) -> (f64, GpuQuantType) {
    let pipeline = match mv.quant {
        GpuQuantType::Q4K => &engine.matvec_q4k_pipeline,
        GpuQuantType::Q5K => &engine.matvec_q5k_pipeline,
        GpuQuantType::Q6K => &engine.matvec_q6k_pipeline,
        GpuQuantType::Q8_0 => &engine.matvec_q8_0_pipeline,
        GpuQuantType::Q1_0 => &engine.matvec_q1_0_row4_pipeline,
    };
    let us = timed_dispatch(engine, pipeline, &mv.bg, mv.dispatch_x, mv.dispatch_y, 1);
    (us, mv.quant)
}

/// Memory estimate for GPU model load — used by `GpuModel::estimate_load_memory`.
///
/// Fields are in bytes. Reported via `[GpuModel] estimated peak memory:` log line at load time.
#[derive(Debug, Clone, Copy)]
pub struct GpuMemoryEstimate {
    /// Sum of GGUF tensor byte sizes (weights + biases + norms + embeddings).
    pub weight_bytes: u64,
    /// KV cache in device memory: `2 (K, V) × num_layers × num_kv_heads × head_dim × max_seq_len × 4 bytes`.
    pub kv_cache_bytes: u64,
    /// Intermediate / staging / bind group overhead estimate.
    pub overhead_bytes: u64,
    /// Projected peak memory. For `IntegratedGpu`, weight bytes counted at 2× (mmap + Vulkan buffer copy).
    /// For `DiscreteGpu`, weight bytes counted at 1.2× (device memory + small transient staging).
    pub peak_bytes: u64,
    /// Whether the adapter uses unified memory (peak factor is higher).
    pub is_integrated_gpu: bool,
}

/// Query system available memory in bytes on Linux via `/proc/meminfo`.
/// Returns `None` on other platforms or on read failure.
#[cfg(target_os = "linux")]
fn query_available_memory_bytes() -> Option<u64> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.trim().split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn query_available_memory_bytes() -> Option<u64> {
    None
}

/// Compute memory estimate for loading a GGUF model on the given adapter type.
#[cfg(feature = "gguf")]
#[must_use]
pub fn estimate_gpu_load_memory(
    device_type: wgpu::DeviceType,
    gguf: &crate::gguf::GgufFile,
    config: &GpuModelConfig,
) -> GpuMemoryEstimate {
    let is_integrated_gpu = matches!(
        device_type,
        wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu
    );

    // Sum tensor bytes from GGUF metadata.
    let weight_bytes: u64 = gguf
        .tensors
        .values()
        .map(|info| info.data_size() as u64)
        .sum();

    // KV cache: 2 (K + V) × num_layers × num_kv_heads × head_dim × max_seq_len × 4 (f32).
    let kv_cache_bytes = 2u64
        * config.num_layers as u64
        * u64::from(config.num_kv_heads)
        * u64::from(config.head_dim)
        * config.max_seq_len as u64
        * 4;

    // Fixed overhead: hidden buffers, intermediate SwiGLU / attention output, bind groups.
    let overhead_bytes = 200 * 1024 * 1024;

    // Peak factor. iGPU on wgpu Vulkan / Metal typically 2× (GGUF mmap + Vulkan buffer copy).
    // Discrete GPU 1.2× (device allocation + transient staging).
    let peak_bytes = if is_integrated_gpu {
        weight_bytes * 2 + kv_cache_bytes + overhead_bytes
    } else {
        weight_bytes * 12 / 10 + kv_cache_bytes + overhead_bytes
    };

    GpuMemoryEstimate {
        weight_bytes,
        kv_cache_bytes,
        overhead_bytes,
        peak_bytes,
        is_integrated_gpu,
    }
}

impl GpuModel {
    // --- Static helpers for bind group construction ---

    fn build_matvec_bg(
        engine: &GpuEngine,
        w: &GpuWeightBuffer,
        input: &GpuBuffer,
        output: &GpuBuffer,
    ) -> MatvecBG {
        // Q1_0 uses the row4 kernel by default in the batch=1 decode path:
        // 1 workgroup produces 4 output rows (input read shared across the
        // 4 rows), cutting workgroup count 4× and total input-read traffic
        // 4×. Empirically 1.82× vs the base `matvec_q1_0` kernel on
        // Jetson Vulkan iGPU for Bonsai 27B attn_qkv (Step 6, PR #74).
        let dispatch_rows = match w.quant {
            GpuQuantType::Q1_0 => w.rows.div_ceil(4),
            GpuQuantType::Q4K | GpuQuantType::Q5K | GpuQuantType::Q6K | GpuQuantType::Q8_0 => {
                w.rows
            }
        };
        let (dispatch_x, dispatch_y, grid_x) = matvec_dispatch(dispatch_rows);
        let params_buf = engine.make_persistent_uniform(&MatvecParams {
            rows: w.rows,
            cols: w.cols,
            blocks_per_row: w.blocks_per_row,
            grid_x,
            batch_size: 1,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        });
        let pipeline = match w.quant {
            GpuQuantType::Q4K => &engine.matvec_q4k_pipeline,
            GpuQuantType::Q5K => &engine.matvec_q5k_pipeline,
            GpuQuantType::Q6K => &engine.matvec_q6k_pipeline,
            GpuQuantType::Q8_0 => &engine.matvec_q8_0_pipeline,
            GpuQuantType::Q1_0 => &engine.matvec_q1_0_row4_pipeline,
        };
        let layout = pipeline.get_bind_group_layout(0);
        let bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: w.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        drop(params_buf); // BindGroup holds Arc ref to GPU resource
        MatvecBG {
            bg,
            dispatch_x,
            dispatch_y,
            quant: w.quant,
        }
    }

    /// Build bind group for element-wise `acc_buf[i] += bias[i]`.
    /// Layout: binding 0 read_write acc_buf, binding 1 read bias.
    fn build_add_bias_bg(
        engine: &GpuEngine,
        acc_buf: &GpuBuffer,
        bias: &GpuBuffer,
    ) -> wgpu::BindGroup {
        let layout = engine.add_bias_pipeline.get_bind_group_layout(0);
        engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: acc_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bias.buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Build bind group for per-head RMSNorm (Qwen 3 QK norm).
    /// In-place on `buf` (both Q and K reuse this pattern with different dispatch counts).
    /// Layout: binding 0 read_write buf, binding 1 read weight, binding 2 uniform params.
    fn build_qk_norm_bg(
        engine: &GpuEngine,
        buf: &GpuBuffer,
        weight: &GpuBuffer,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let layout = engine.qk_norm_pipeline.get_bind_group_layout(0);
        engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    fn build_rmsnorm_bg(
        engine: &GpuEngine,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        output: &GpuBuffer,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let layout = engine.rmsnorm_pipeline.get_bind_group_layout(0);
        engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    fn build_swiglu_bg(
        engine: &GpuEngine,
        gate_w: &GpuWeightBuffer,
        up_w: &GpuWeightBuffer,
        input: &GpuBuffer,
        output: &GpuBuffer,
    ) -> SwigluBG {
        assert_eq!(
            gate_w.quant, up_w.quant,
            "SwigluBG requires matching quant types for gate and up projections"
        );
        let (dispatch_x, dispatch_y, grid_x) = matvec_dispatch(gate_w.rows);
        let params_buf = engine.make_persistent_uniform(&MatvecParams {
            rows: gate_w.rows,
            cols: gate_w.cols,
            blocks_per_row: gate_w.blocks_per_row,
            grid_x,
            batch_size: 1,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        });
        // Select the fused SwiGLU pipeline by quant type — Q4_K uses the
        // 256-elem block kernel, Q1_0 uses the 18-byte block kernel with
        // byte-level indexing. Other quants are not supported by the
        // fused path (would need a new shader per quant); the loader
        // panics upstream when it sees such a FFN configuration.
        let pipeline = match gate_w.quant {
            GpuQuantType::Q4K => &engine.swiglu_fused_q4k_pipeline,
            GpuQuantType::Q1_0 => &engine.swiglu_fused_q1_0_pipeline,
            other => panic!(
                "build_swiglu_bg: unsupported quant type {other:?} for fused \
                 SwiGLU — only Q4_K and Q1_0 have fused shaders. Add a new \
                 `swiglu_fused_<quant>.wgsl` or fall back to un-fused \
                 `gate_proj` + `up_proj` + silu_mul + `down_proj`."
            ),
        };
        let layout = pipeline.get_bind_group_layout(0);
        let bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gate_w.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: up_w.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        drop(params_buf); // BindGroup holds Arc ref to GPU resource
        SwigluBG {
            bg,
            dispatch_x,
            dispatch_y,
            quant: gate_w.quant,
        }
    }

    /// Load model from GGUF, upload all weights, pre-create all bind groups.
    /// Load model from GGUF (takes ownership of engine).
    #[cfg(feature = "gguf")]
    pub fn load(engine: GpuEngine, gguf: &crate::gguf::GgufFile, config: GpuModelConfig) -> Self {
        Self::load_shared(Arc::new(engine), gguf, config)
    }

    /// Load model from GGUF with shared engine (for dual-model speculative decoding).
    #[cfg(feature = "gguf")]
    pub fn load_shared(
        engine: Arc<GpuEngine>,
        gguf: &crate::gguf::GgufFile,
        config: GpuModelConfig,
    ) -> Self {
        use crate::gguf::GgmlType;

        // Bonsai / Qwen 3.6-27B detection — the GGUF splits attention QKV into
        // `blk.0.attn_qkv.weight` (Q + swish gate for full-attention layers, or
        // Q + K + V for DeltaNet layers) + `blk.0.attn_gate.weight` (Z for
        // DeltaNet layers). This layout requires gated attention shader support
        // and DeltaNet SSM refinement pipelines (`ssm_a` / `ssm_dt_bias` /
        // `ssm_norm` consumers) that GpuModel does not yet implement.
        //
        // Bonsai / Qwen 3.6-27B tensor layout detection.
        //
        // Bonsai splits the DeltaNet fused in_proj into `attn_qkv` + `attn_gate`
        // (unlike standard Qwen 3.5 which uses `ssm_in`), and adds SSM
        // refinement tensors `ssm_a` / `ssm_dt_bias` / `ssm_norm`. The GPU
        // loader can now handle this layout — Phase 2 Step 1 (this PR) makes
        // load succeed by loading Bonsai tensors into `Option<>` fields on
        // `DeltaNetLayerWeightBufs` / `DeltaNetLayerBGs`.
        //
        // Forward path integration (Phase 2 Step 4) is separate work: the
        // `encode_forward_impl` DeltaNet arm still fails fast when it
        // encounters a Bonsai layer, pointing to the specific building block
        // that needs to be wired (see `docs/BONSAI_GPU_SUPPORT.md`).
        //
        // The previous PR #77 unconditional fail-fast is removed in favor of
        // this partial support (load succeeds, forward panics with clear
        // per-op message).
        // For Bonsai 27B (full_attention_interval=4) blk.0 is always a
        // DeltaNet layer, so checking `blk.0.attn_qkv.weight` cleanly
        // distinguishes the Bonsai layout from standard Qwen 3.5 (which
        // exports `blk.0.ssm_in.weight` instead).
        // Bonsai SSM-refinement path is gated on the presence of BOTH
        // `attn_qkv` layout (split QKV) AND `ssm_a`/`ssm_dt_bias` tensors —
        // matching CPU semantics at `llama3.rs:3975`:
        //     is_bonsai_path = dn_layer.ssm_a.is_some() && dn_layer.ssm_dt_bias.is_some()
        //
        // Qwen 3.5-4B has the `attn_qkv` split layout (loader dispatches two
        // matvecs instead of fused ssm_in) BUT NO `ssm_a`/`ssm_dt_bias`, so
        // the gated_deltanet shader must run the standard silu(q)/silu(k) +
        // silu(z) semantics (is_bonsai=0). Only Qwen 3.6-27B ("real" Bonsai)
        // ships the SSM refinement tensors and requires is_bonsai=1.
        // GGUF tensor names for these refinements omit the `.weight` suffix
        // (`blk.N.ssm_a`) and use dot-separated `ssm_dt.bias` — matching the
        // CPU loader at `llama3.rs:9477-9478`. Previous GPU-side names
        // (`ssm_a.weight` / `ssm_dt_bias.weight`) never matched any GGUF
        // tensor, so `is_bonsai_deltanet_gguf` was always false for
        // Qwen 3.5-4B / Bonsai 27B, causing:
        //   - `dn_params_buf.is_bonsai = 0` → shader used raw `alpha`
        //     values from `alpha_proj` (~7 in magnitude) as decay
        //     factors, blowing up recurrent state 7× per token.
        //   - `ssm_a_opt` / `ssm_dt_bias_opt` load returned None → Gap B
        //     SSM discretisation dispatch was skipped, so no
        //     `alpha → exp(softplus(...) * ssm_a)` transform to bring
        //     alpha into [0, 1].
        // Root cause of Phase X.3.e.3.23 layer-0 L2 = 2328 (CPU: 2.43).
        let is_bonsai_deltanet_gguf = gguf.tensor_info("blk.0.attn_qkv.weight").is_some()
            && gguf.tensor_info("blk.0.ssm_a").is_some()
            && gguf.tensor_info("blk.0.ssm_dt.bias").is_some();
        if is_bonsai_deltanet_gguf {
            eprintln!(
                "[GpuModel] Bonsai / Qwen 3.6-27B tensor layout detected \
                 (attn_qkv + attn_gate + SSM refinement tensors). Load will \
                 succeed, but forward path is not yet fully wired — expect a \
                 panic at DeltaNet layer forward with details about the \
                 remaining Phase 2 Step 4 work. See docs/BONSAI_GPU_SUPPORT.md."
            );
        }
        // Full-attention Bonsai detection — Bonsai stores Q and the swish gate
        // as a single stacked tensor `attn_q [q_dim*2, hidden]`; the loader
        // splits the byte data at the midpoint into two separate
        // `GpuWeightBuffer`s (`q_proj` + `bonsai_attn_q_gate`) so both halves
        // can be dispatched as separate matvecs during forward.
        //
        // Load succeeds; forward path for full-attn Bonsai still requires
        // wiring the gate multiplication (deferred to Phase 2 Step 4b —
        // this PR only completes the loader side).
        let is_bonsai_full_attn_gguf = if let Some(interval) = config.full_attention_interval {
            let full_attn_layer = interval - 1;
            let expected_q_rows = config.num_heads * config.head_dim;
            gguf.tensor_info(&format!("blk.{full_attn_layer}.attn_q.weight"))
                .is_some_and(|info| info.dims[1] as u32 == expected_q_rows * 2)
        } else {
            false
        };
        if is_bonsai_full_attn_gguf {
            eprintln!(
                "[GpuModel] Bonsai / Qwen 3.6-27B full-attention layer layout \
                 detected (attn_q with 2×q_dim rows). Loader will split into \
                 q_proj + bonsai_attn_q_gate. Forward path for full-attn \
                 Bonsai (silu(gate) multiplication into attn_out before \
                 o_proj) is Phase 2 Step 4b — expect a panic at full-attn \
                 layer forward if not yet wired."
            );
        }

        // OOM prevention: estimate peak memory usage and log warning if projected
        // to exceed available system memory. On integrated GPU (Jetson / Apple Silicon /
        // AMD APU / Intel iGPU), wgpu Vulkan / Metal typically double-allocates weight
        // memory (GGUF mmap + Vulkan buffer copy), so peak is ~2x GGUF size.
        let estimate = estimate_gpu_load_memory(engine.device_type, gguf, &config);
        let peak_gb = estimate.peak_bytes as f64 / 1e9;
        eprintln!(
            "[GpuModel] estimated peak memory: {peak_gb:.2} GB (weights {:.2} GB{}, KV cache {:.2} GB, overhead {:.2} GB)",
            estimate.weight_bytes as f64 / 1e9,
            if estimate.is_integrated_gpu {
                " x2 iGPU"
            } else {
                " x1.2 dGPU"
            },
            estimate.kv_cache_bytes as f64 / 1e9,
            estimate.overhead_bytes as f64 / 1e9,
        );
        if let Some(available) = query_available_memory_bytes() {
            let available_gb = available as f64 / 1e9;
            eprintln!("[GpuModel] system available memory: {available_gb:.2} GB");
            if estimate.peak_bytes > available * 90 / 100 {
                eprintln!(
                    "[GpuModel] WARNING: projected peak ({peak_gb:.2} GB) exceeds 90% of available ({available_gb:.2} GB)"
                );
                eprintln!(
                    "[GpuModel] WARNING: kernel OOM Kill likely during weight upload. Consider:"
                );
                eprintln!(
                    "[GpuModel] WARNING:   1) Smaller model (Q4_K 1.5B-3B on 8 GB unified memory)"
                );
                eprintln!(
                    "[GpuModel] WARNING:   2) CPU inference via `Llama3Model::from_gguf` (mmap zero-copy)"
                );
                eprintln!(
                    "[GpuModel] WARNING:   3) llama.cpp with Vulkan backend (unified memory zero-copy)"
                );
            }
        }

        let kv_dim = (config.num_kv_heads * config.head_dim) as usize;

        // Upload weight with auto quant type detection
        let upload_w = |name: &str, rows: usize, cols: usize| -> GpuWeightBuffer {
            let info = gguf.tensor_info(name).unwrap();
            let data = gguf.tensor_data(name).unwrap();
            match info.qtype {
                GgmlType::Q4_K => engine.upload_weights(data, rows, cols),
                GgmlType::Q5_K => engine.upload_weights_q5k(data, rows, cols),
                GgmlType::Q6_K => engine.upload_weights_q6k(data, rows, cols),
                GgmlType::Q8_0 => engine.upload_weights_q8_0(data, rows, cols),
                GgmlType::Q1_0 => engine.upload_weights_q1_0(data, rows, cols),
                other => panic!(
                    "GPU upload for tensor `{name}` unsupported quant type {other:?} \
                     (row=[{rows} col={cols}]) — extend GpuQuantType + upload_weights_typed"
                ),
            }
        };

        // Split a Bonsai `attn_q [q_dim*2, hidden]` tensor at load time into
        // (Q half, gate half). Each half becomes an independent
        // `GpuWeightBuffer` with shape `[q_dim, hidden]` and the same
        // quantisation type.
        //
        // Reference `qwen35.cpp:347-370` shows the GGUF stride for gated
        // attention Q is `nb1 = element_size * n_embd_head * 2`, i.e. the
        // rows are stored **per-head interleaved**:
        //   `[q_h0, gate_h0, q_h1, gate_h1, ..., q_h(N-1), gate_h(N-1)]`
        // where each `q_hi` / `gate_hi` is `head_dim` rows. CPU
        // (`llama3.rs:4373-4394`) matvecs the raw tensor and de-interleaves
        // the output vector after the fact. The GPU couldn't do the same
        // without extra pipeline glue, so we de-interleave at load time
        // by shuffling byte-aligned per-row blocks into two separate
        // buffers. This works uniformly for Q4_K / Q5_K / Q6_K / Q8_0 /
        // Q1_0 because each row occupies a fixed byte size (row-major
        // GGUF storage).
        //
        // Previous naive midpoint split was correct only for consecutive
        // `[Q(q_dim), Gate(q_dim)]` layout, which no known Qwen / Bonsai
        // GGUF ships. On Qwen 3.5-4B the wrong split fed heads 0..7 of
        // interleaved (q, gate, q, gate, ...) rows into the Q buffer,
        // producing garbage attention outputs at layer 3 and cascading
        // through the network (Phase X.3.e.3.26 diagnostic finding).
        let upload_w_bonsai_split = |name: &str,
                                     half_rows: usize,
                                     cols: usize,
                                     num_heads: usize,
                                     head_dim: usize|
         -> (GpuWeightBuffer, GpuWeightBuffer) {
            let info = gguf.tensor_info(name).unwrap();
            let data = gguf.tensor_data(name).unwrap();
            let total_rows = half_rows * 2;
            let bytes_per_row = data.len() / total_rows;
            assert_eq!(
                half_rows,
                num_heads * head_dim,
                "upload_w_bonsai_split: half_rows ({half_rows}) must equal \
                 num_heads * head_dim ({num_heads} * {head_dim})"
            );
            let head_pair_rows = head_dim * 2;
            let head_pair_bytes = head_pair_rows * bytes_per_row;
            let half_bytes_per_head = head_dim * bytes_per_row;
            let mut q_bytes: Vec<u8> = Vec::with_capacity(half_rows * bytes_per_row);
            let mut gate_bytes: Vec<u8> = Vec::with_capacity(half_rows * bytes_per_row);
            for h in 0..num_heads {
                let head_start = h * head_pair_bytes;
                let q_end = head_start + half_bytes_per_head;
                let gate_end = q_end + half_bytes_per_head;
                q_bytes.extend_from_slice(&data[head_start..q_end]);
                gate_bytes.extend_from_slice(&data[q_end..gate_end]);
            }
            let upload_half = |bytes: &[u8]| -> GpuWeightBuffer {
                match info.qtype {
                    GgmlType::Q4_K => engine.upload_weights(bytes, half_rows, cols),
                    GgmlType::Q5_K => engine.upload_weights_q5k(bytes, half_rows, cols),
                    GgmlType::Q6_K => engine.upload_weights_q6k(bytes, half_rows, cols),
                    GgmlType::Q8_0 => engine.upload_weights_q8_0(bytes, half_rows, cols),
                    GgmlType::Q1_0 => engine.upload_weights_q1_0(bytes, half_rows, cols),
                    other => panic!(
                        "GPU upload for tensor `{name}` (bonsai split) unsupported quant \
                         type {other:?} — extend GpuQuantType + upload_weights_typed"
                    ),
                }
            };
            let q = upload_half(&q_bytes);
            let gate = upload_half(&gate_bytes);
            (q, gate)
        };

        eprintln!("[GpuModel] uploading weights...");
        let t0 = std::time::Instant::now();

        // Determine layer types from config
        let layer_types: Vec<LayerType> = (0..config.num_layers)
            .map(|i| match config.full_attention_interval {
                Some(interval) if (i + 1) % interval != 0 => LayerType::DeltaNet,
                _ => LayerType::Attention,
            })
            .collect();

        // DeltaNet config (Qwen3.5)
        let dn_qk_dim = config.linear_qk_head_dim.unwrap_or(128) as usize;
        let dn_v_dim = config.linear_kv_head_dim.unwrap_or(128) as usize;
        let dn_num_kv_heads = config.linear_num_kv_heads.unwrap_or(config.num_kv_heads) as usize;
        let dn_num_v_heads = config.linear_num_v_heads.unwrap_or(config.num_heads) as usize;
        let dn_conv_kernel = config.linear_conv_kernel_dim.unwrap_or(4) as usize;
        // Fused in_proj output size: q(qk_dim*num_kv) + k(qk_dim*num_kv) + v(v_dim*num_v) + z(v_dim*num_v)
        let dn_in_proj_out = dn_qk_dim * dn_num_kv_heads * 2 + dn_v_dim * dn_num_v_heads * 2;
        let dn_conv_dim = dn_qk_dim * dn_num_kv_heads * 2 + dn_v_dim * dn_num_v_heads; // q+k+v (not z)

        let mut layer_weights = Vec::with_capacity(config.num_layers);
        let mut deltanet_layer_weights: Vec<DeltaNetLayerWeightBufs> = Vec::new();
        let mut deltanet_states: Vec<GpuBuffer> = Vec::new();
        let mut deltanet_conv_states: Vec<GpuBuffer> = Vec::new();

        for i in 0..config.num_layers {
            match layer_types[i] {
                LayerType::Attention => {
                    // Bonsai full-attn detection per-layer: `attn_q` has
                    // 2×hidden_dim rows (Q + swish gate stacked). Split at
                    // load time so both halves are queryable individually.
                    let attn_q_name = format!("blk.{i}.attn_q.weight");
                    let attn_q_actual_rows = gguf
                        .tensor_info(&attn_q_name)
                        .map(|info| info.dims[1] as usize)
                        .unwrap_or(config.hidden_dim);
                    // Phase X.3.e.3.19 fix: gated attention layer detection uses
                    // q_dim = num_heads * head_dim (NOT hidden_dim). For Qwen
                    // 3.5-4B, hidden_dim=2560 but q_dim=16*256=4096, and
                    // attn_q rows = 2 * q_dim = 8192. Previous condition
                    // `attn_q_actual_rows == hidden_dim * 2` (== 5120) missed
                    // the actual `== q_dim * 2` (== 8192) pattern, causing
                    // Qwen 3.5-4B gated attention layers to load through the
                    // non-Bonsai path with `upload_w(name, hidden_dim, hidden_dim)`
                    // = wrong row count = 31% of Q4_K tensor bytes only =
                    // massive weight data corruption on GPU forward.
                    let q_dim_attn = (config.num_heads as usize) * (config.head_dim as usize);
                    let is_bonsai_full_attn_layer = attn_q_actual_rows == q_dim_attn * 2;
                    let (q_proj_val, bonsai_attn_q_gate_val) = if is_bonsai_full_attn_layer {
                        let (q, gate) = upload_w_bonsai_split(
                            &attn_q_name,
                            q_dim_attn,
                            config.hidden_dim,
                            config.num_heads as usize,
                            config.head_dim as usize,
                        );
                        (q, Some(gate))
                    } else {
                        (upload_w(&attn_q_name, q_dim_attn, config.hidden_dim), None)
                    };
                    layer_weights.push(LayerWeightBufs {
                        attn_norm: engine.upload_f32(
                            &gguf
                                .tensor_to_f32(&format!("blk.{i}.attn_norm.weight"))
                                .unwrap(),
                        ),
                        q_proj: q_proj_val,
                        bonsai_attn_q_gate: bonsai_attn_q_gate_val,
                        k_proj: upload_w(
                            &format!("blk.{i}.attn_k.weight"),
                            kv_dim,
                            config.hidden_dim,
                        ),
                        v_proj: upload_w(
                            &format!("blk.{i}.attn_v.weight"),
                            kv_dim,
                            config.hidden_dim,
                        ),
                        o_proj: upload_w(
                            &format!("blk.{i}.attn_output.weight"),
                            config.hidden_dim,
                            config.hidden_dim,
                        ),
                        q_bias: gguf
                            .tensor_to_f32(&format!("blk.{i}.attn_q.bias"))
                            .map(|v| engine.upload_f32(&v)),
                        k_bias: gguf
                            .tensor_to_f32(&format!("blk.{i}.attn_k.bias"))
                            .map(|v| engine.upload_f32(&v)),
                        v_bias: gguf
                            .tensor_to_f32(&format!("blk.{i}.attn_v.bias"))
                            .map(|v| engine.upload_f32(&v)),
                        q_norm: gguf
                            .tensor_to_f32(&format!("blk.{i}.attn_q_norm.weight"))
                            .map(|v| engine.upload_f32(&v)),
                        k_norm: gguf
                            .tensor_to_f32(&format!("blk.{i}.attn_k_norm.weight"))
                            .map(|v| engine.upload_f32(&v)),
                        ffn_norm: engine.upload_f32(
                            // Standard Qwen 3.5 / Llama / Qwen 2/3 / Gemma
                            // ship `ffn_norm.weight`; Bonsai / Qwen 3.6-27B
                            // exports it as `post_attention_norm.weight`.
                            // Mirrors `llama3::load_ffn_norm` fallback.
                            &gguf
                                .tensor_to_f32(&format!("blk.{i}.ffn_norm.weight"))
                                .or_else(|| {
                                    gguf.tensor_to_f32(&format!(
                                        "blk.{i}.post_attention_norm.weight"
                                    ))
                                })
                                .expect(
                                    "neither blk.N.ffn_norm.weight nor \
                                     blk.N.post_attention_norm.weight found",
                                ),
                        ),
                        gate_proj: upload_w(
                            &format!("blk.{i}.ffn_gate.weight"),
                            config.intermediate_dim,
                            config.hidden_dim,
                        ),
                        up_proj: upload_w(
                            &format!("blk.{i}.ffn_up.weight"),
                            config.intermediate_dim,
                            config.hidden_dim,
                        ),
                        down_proj: upload_w(
                            &format!("blk.{i}.ffn_down.weight"),
                            config.hidden_dim,
                            config.intermediate_dim,
                        ),
                    });
                }
                LayerType::DeltaNet => {
                    // Load DeltaNet-specific tensors
                    let conv1d_w = gguf
                        .tensor_to_f32(&format!("blk.{i}.ssm_conv1d.weight"))
                        .unwrap();
                    // Bonsai / Qwen 3.6-27B GGUFs omit the conv1d bias tensor
                    // (see PR #62 for the CPU-side treatment). Standard
                    // Qwen 3.5 GGUFs always ship it. Load as Option and let
                    // the forward path decide how to handle absence.
                    let conv1d_b_opt = gguf
                        .tensor_to_f32(&format!("blk.{i}.ssm_conv1d.bias"))
                        .map(|v| engine.upload_f32(&v));

                    // Bonsai / Qwen 3.6-27B split the fused `ssm_in` into
                    // `attn_qkv` + `attn_gate`. Exactly one of these layouts
                    // is present in the GGUF; the loader picks accordingly.
                    let has_attn_qkv = gguf
                        .tensor_info(&format!("blk.{i}.attn_qkv.weight"))
                        .is_some();
                    let (ssm_in_opt, attn_qkv_opt, attn_gate_opt) = if has_attn_qkv {
                        // Bonsai layout: two-tensor split.
                        // Compute Bonsai-specific row counts:
                        //   attn_qkv rows = qk_dim*num_kv_heads*2 + v_dim*num_v_heads
                        //   attn_gate rows = v_dim * num_v_heads
                        let qkv_rows =
                            (dn_qk_dim * dn_num_kv_heads) * 2 + dn_v_dim * dn_num_v_heads;
                        let gate_rows = dn_v_dim * dn_num_v_heads;
                        (
                            None,
                            Some(upload_w(
                                &format!("blk.{i}.attn_qkv.weight"),
                                qkv_rows,
                                config.hidden_dim,
                            )),
                            Some(upload_w(
                                &format!("blk.{i}.attn_gate.weight"),
                                gate_rows,
                                config.hidden_dim,
                            )),
                        )
                    } else {
                        // Standard Qwen 3.5 layout: single fused ssm_in tensor.
                        (
                            Some(upload_w(
                                &format!("blk.{i}.ssm_in.weight"),
                                dn_in_proj_out,
                                config.hidden_dim,
                            )),
                            None,
                            None,
                        )
                    };

                    // Bonsai SSM refinement tensors — all optional. Standard
                    // Qwen 3.5 GGUFs omit these; Bonsai GGUFs include them.
                    // Tensor names (no `.weight` suffix on `ssm_a`;
                    // dot-separated `ssm_dt.bias`) mirror the CPU loader at
                    // `llama3.rs:9477-9478`. See root-cause note above
                    // (Phase X.3.e.3.23).
                    let ssm_a_opt = gguf
                        .tensor_to_f32(&format!("blk.{i}.ssm_a"))
                        .map(|v| engine.upload_f32(&v));
                    let ssm_dt_bias_opt = gguf
                        .tensor_to_f32(&format!("blk.{i}.ssm_dt.bias"))
                        .map(|v| engine.upload_f32(&v));
                    let ssm_norm_opt = gguf
                        .tensor_to_f32(&format!("blk.{i}.ssm_norm.weight"))
                        .map(|v| engine.upload_f32(&v));

                    deltanet_layer_weights.push(DeltaNetLayerWeightBufs {
                        attn_norm: engine.upload_f32(
                            &gguf
                                .tensor_to_f32(&format!("blk.{i}.attn_norm.weight"))
                                .unwrap(),
                        ),
                        ssm_in: ssm_in_opt,
                        attn_qkv: attn_qkv_opt,
                        attn_gate: attn_gate_opt,
                        ssm_a: ssm_a_opt,
                        ssm_dt_bias: ssm_dt_bias_opt,
                        ssm_norm: ssm_norm_opt,
                        conv1d_weight: engine.upload_f32(&conv1d_w),
                        conv1d_bias: conv1d_b_opt,
                        // Phase X.3.e.3.12 fix: ssm_alpha/beta rows count is
                        // `num_v_heads` (32 for Qwen 3.5, 48 for Bonsai 27B),
                        // NOT `num_kv_heads` (16). Previous code loaded only
                        // half the weight for GGUFs where the tensor has 32
                        // rows, producing undefined alpha/beta for V-heads
                        // 16..31. CPU loader uses `load_weight_ref_any_rows`
                        // which reads the actual GGUF row count (see
                        // llama3.rs:9351-9360).
                        alpha_proj: upload_w(
                            &format!("blk.{i}.ssm_alpha.weight"),
                            dn_num_v_heads,
                            config.hidden_dim,
                        ),
                        beta_proj: upload_w(
                            &format!("blk.{i}.ssm_beta.weight"),
                            dn_num_v_heads,
                            config.hidden_dim,
                        ),
                        ssm_out: upload_w(
                            &format!("blk.{i}.ssm_out.weight"),
                            config.hidden_dim,
                            dn_v_dim * dn_num_v_heads,
                        ),
                        ffn_norm: engine.upload_f32(
                            // Standard Qwen 3.5 / Llama / Qwen 2/3 / Gemma
                            // ship `ffn_norm.weight`; Bonsai / Qwen 3.6-27B
                            // exports it as `post_attention_norm.weight`.
                            // Mirrors `llama3::load_ffn_norm` fallback.
                            &gguf
                                .tensor_to_f32(&format!("blk.{i}.ffn_norm.weight"))
                                .or_else(|| {
                                    gguf.tensor_to_f32(&format!(
                                        "blk.{i}.post_attention_norm.weight"
                                    ))
                                })
                                .expect(
                                    "neither blk.N.ffn_norm.weight nor \
                                     blk.N.post_attention_norm.weight found",
                                ),
                        ),
                        gate_proj: upload_w(
                            &format!("blk.{i}.ffn_gate.weight"),
                            config.intermediate_dim,
                            config.hidden_dim,
                        ),
                        up_proj: upload_w(
                            &format!("blk.{i}.ffn_up.weight"),
                            config.intermediate_dim,
                            config.hidden_dim,
                        ),
                        down_proj: upload_w(
                            &format!("blk.{i}.ffn_down.weight"),
                            config.hidden_dim,
                            config.intermediate_dim,
                        ),
                    });

                    // Recurrent state: [num_v_heads, qk_dim, v_dim].
                    // Phase X.3.e.3.11 fix: previous `dn_num_kv_heads` sizing
                    // allocated half the required memory when num_v_heads >
                    // num_kv_heads (Qwen 3.5: 32/16, Bonsai: 48/16). Under
                    // cyclic V→KV mapping each V-head has its own recurrent
                    // state slab even though V-heads share Q/K per KV group.
                    // Phase X.3.e.3.20 fix: explicitly zero-initialize both
                    // buffers. `alloc_f32` returns uninitialized memory on
                    // Metal (wgpu-hal doesn't guarantee zero-init unless
                    // `mapped_at_creation` is true) → first forward reads
                    // NaN/Inf from uninitialized DeltaNet state → propagates
                    // through recurrence → hidden state = NaN → PAD248319
                    // (argmax picks last valid token).
                    let dn_state_len = dn_num_v_heads * dn_qk_dim * dn_v_dim;
                    let dn_state_buf = engine.alloc_f32(dn_state_len);
                    engine.write_f32(&dn_state_buf, &vec![0.0f32; dn_state_len]);
                    deltanet_states.push(dn_state_buf);
                    // Conv1d ring buffer: [kernel_size - 1, conv_dim]
                    let conv_state_len = (dn_conv_kernel - 1) * dn_conv_dim;
                    let conv_state_buf = engine.alloc_f32(conv_state_len);
                    engine.write_f32(&conv_state_buf, &vec![0.0f32; conv_state_len]);
                    deltanet_conv_states.push(conv_state_buf);

                    // Push a dummy LayerWeightBufs placeholder so indexing stays aligned
                    // (DeltaNet layers use deltanet_layer_weights instead)
                    layer_weights.push(LayerWeightBufs {
                        attn_norm: engine.alloc_f32(1),
                        q_proj: engine.upload_weights(&[0u8; 256], 1, 1),
                        bonsai_attn_q_gate: None,
                        k_proj: engine.upload_weights(&[0u8; 256], 1, 1),
                        v_proj: engine.upload_weights(&[0u8; 256], 1, 1),
                        o_proj: engine.upload_weights(&[0u8; 256], 1, 1),
                        q_bias: None,
                        k_bias: None,
                        v_bias: None,
                        q_norm: None,
                        k_norm: None,
                        ffn_norm: engine.alloc_f32(1),
                        gate_proj: engine.upload_weights(&[0u8; 256], 1, 1),
                        up_proj: engine.upload_weights(&[0u8; 256], 1, 1),
                        down_proj: engine.upload_weights(&[0u8; 256], 1, 1),
                    });
                }
            }
        }

        let n_attn = layer_types
            .iter()
            .filter(|t| **t == LayerType::Attention)
            .count();
        let n_delta = layer_types
            .iter()
            .filter(|t| **t == LayerType::DeltaNet)
            .count();
        if n_delta > 0 {
            eprintln!("[GpuModel] hybrid: {n_delta} DeltaNet + {n_attn} Attention layers");
        }

        // Output projection — fallback to tied embedding
        let output_norm_weight =
            engine.upload_f32(&gguf.tensor_to_f32("output_norm.weight").unwrap());
        let (out_name, vocab_size) = if let Some(info) = gguf.tensor_info("output.weight") {
            ("output.weight", info.dims[1] as usize)
        } else {
            let info = gguf.tensor_info("token_embd.weight").unwrap();
            ("token_embd.weight", info.dims[1] as usize)
        };
        let output_proj_weight = upload_w(out_name, vocab_size, config.hidden_dim);
        let embedding = gguf.tensor_to_f32("token_embd.weight").unwrap();

        eprintln!(
            "[GpuModel] weights uploaded: {:.0}ms (vocab={vocab_size}, output={:?})",
            t0.elapsed().as_millis(),
            output_proj_weight.quant,
        );

        // Scratch buffers (MAX_BATCH sized for batch forward support).
        // Phase X.3.e.3.20 fix: explicit zero-initialization to prevent NaN
        // propagation from uninitialized memory (wgpu-hal Metal does not
        // guarantee zero-init for `alloc_f32`; DeltaNet recurrence + RMSNorm
        // + matvec on uninitialized garbage → NaN → PAD248319 output).
        const MAX_BATCH: usize = 8;
        let zero_init = |sz: usize| -> GpuBuffer {
            let b = engine.alloc_f32(sz);
            engine.write_f32(&b, &vec![0.0f32; sz]);
            b
        };
        let hidden = zero_init(MAX_BATCH * config.hidden_dim);
        let norm_buf = zero_init(MAX_BATCH * config.hidden_dim);
        let q_buf = zero_init(MAX_BATCH * config.hidden_dim);
        // Scratch buffers `k_buf` / `v_buf` are shared between the attention
        // path (holds `k_proj` / `v_proj` output, sized `kv_dim`) and the
        // DeltaNet path (holds `conv1d` output at `dn_conv_dim` = q+k+v
        // packed, and `attn_gate` output at `dn_v_dim * dn_num_v_heads` for
        // the Bonsai z-gate). Take the max of both consumers so hybrid
        // models (Qwen 3.5 / Bonsai 27B) don't overflow the buffer inside
        // the DeltaNet bind group's `BufferBinding { offset, size }`.
        //
        // Bonsai 27B example: `dn_conv_dim = 128*16*2 + 128*48 = 10240` f32
        // (40960 bytes/token), which exceeds the attention `kv_dim = 1024`
        // (4096 bytes/token). Without this max the layer-load bind group
        // creation panicked with `Buffer bound range 16384..40960 overflows
        // its size (32768)` (Phase X.3.e.3.26 Bonsai 27B verification).
        let dn_conv_dim_usize = dn_conv_dim as usize;
        let dn_v_out = dn_v_dim * dn_num_v_heads;
        let k_buf_dim = kv_dim.max(dn_conv_dim_usize);
        let v_buf_dim = kv_dim.max(dn_v_out);
        let k_buf = zero_init(MAX_BATCH * k_buf_dim);
        let v_buf = zero_init(MAX_BATCH * v_buf_dim);
        let attn_out = zero_init(MAX_BATCH * config.hidden_dim);
        // Bonsai DeltaNet writes the post-`ssm_norm_per_head` result into
        // this scratch instead of aliasing `attn_out` — WebGPU forbids
        // binding the same buffer as both `read` and `read_write` inside a
        // single compute pass. Sized to hold the `attn_out` slice used by
        // the DeltaNet pipeline (`num_v_heads * v_dim`), fallback to the
        // legacy `hidden_dim` size when the model is not Bonsai.
        let attn_out_normed_size = {
            let dn_v_dim = config.linear_kv_head_dim.unwrap_or(0) as usize;
            let dn_num_v_heads = config.linear_num_v_heads.unwrap_or(config.num_heads) as usize;
            let bonsai_size = dn_num_v_heads * dn_v_dim;
            MAX_BATCH * bonsai_size.max(config.hidden_dim)
        };
        let attn_out_normed = zero_init(attn_out_normed_size);
        let o_buf = zero_init(MAX_BATCH * config.hidden_dim);
        let gate_buf = zero_init(MAX_BATCH * config.intermediate_dim);
        let down_buf = zero_init(MAX_BATCH * config.hidden_dim);
        // DeltaNet decay-gate / update-rate scratch (Qwen 3.5 hybrid). Sized to
        // `linear_num_v_heads` (per-V-head scalar under cyclic V→KV mapping,
        // Phase X.3.e.3.11 fix) with MAX_BATCH lanes reserved for future
        // batch-inference support. Reused across all DeltaNet layers within
        // a single forward pass. Previously sized to `linear_num_kv_heads`
        // which left V-heads 16..31 (Qwen 3.5) or 16..47 (Bonsai) with
        // undefined alpha/beta.
        let dn_num_v_heads_scratch = config.linear_num_v_heads.unwrap_or(config.num_heads) as usize;
        let alpha_buf = engine.alloc_f32(MAX_BATCH * dn_num_v_heads_scratch);
        let beta_buf = engine.alloc_f32(MAX_BATCH * dn_num_v_heads_scratch);
        let logits = engine.alloc_f32(MAX_BATCH * vocab_size);
        let staging = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("logits_staging"),
            size: (MAX_BATCH * vocab_size * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // KV caches (only for attention layers)
        let n_attn_layers = layer_types
            .iter()
            .filter(|t| **t == LayerType::Attention)
            .count();
        let mut k_caches = Vec::with_capacity(n_attn_layers);
        let mut v_caches = Vec::with_capacity(n_attn_layers);
        for _ in 0..n_attn_layers {
            k_caches.push(engine.alloc_f32(config.max_seq_len * kv_dim));
            v_caches.push(engine.alloc_f32(config.max_seq_len * kv_dim));
        }

        // --- Persistent uniform buffers (updated per-token via write_buffer) ---
        let make_persistent = |data: &[u8]| -> wgpu::Buffer {
            engine
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: data,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                })
        };

        let neox_flag: u32 = u32::from(config.neox_rope);
        let rope_q_params_buf = make_persistent(bytemuck::bytes_of(&RopeParams {
            position: 0,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            theta: config.rope_theta,
            batch_size: 1,
            neox: neox_flag,
            _pad2: 0,
            _pad3: 0,
        }));
        let rope_k_params_buf = make_persistent(bytemuck::bytes_of(&RopeParams {
            position: 0,
            num_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            theta: config.rope_theta,
            batch_size: 1,
            neox: neox_flag,
            _pad2: 0,
            _pad3: 0,
        }));
        let kv_append_params_buf = make_persistent(bytemuck::bytes_of(&KvCacheParams {
            position: 0,
            kv_dim: kv_dim as u32,
            batch_size: 1,
            _pad2: 0,
        }));
        let inv_sqrt_d = 1.0 / (config.head_dim as f32).sqrt();
        let attn_params_buf = make_persistent(bytemuck::bytes_of(&AttentionParams {
            seq_len: 1,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            kv_dim: kv_dim as u32,
            inv_sqrt_d,
            batch_size: 1,
            _pad2: 0,
        }));

        // --- Shared bind groups (layer-independent) ---
        let rope_layout = engine.rope_pipeline.get_bind_group_layout(0);
        let rope_q_bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &rope_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rope_q_params_buf.as_entire_binding(),
                },
            ],
        });
        let rope_k_bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &rope_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: k_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rope_k_params_buf.as_entire_binding(),
                },
            ],
        });

        let residual_layout = engine.residual_add_pipeline.get_bind_group_layout(0);
        let residual_attn_bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &residual_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: hidden.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: o_buf.buffer.as_entire_binding(),
                },
            ],
        });
        let residual_ffn_bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &residual_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: hidden.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: down_buf.buffer.as_entire_binding(),
                },
            ],
        });

        // RMSNorm params (shared by all norm dispatches)
        let rmsnorm_params_buf = engine.make_persistent_uniform(&RmsnormParams {
            dim: config.hidden_dim as u32,
            eps: config.eps,
            batch_size: 1,
            _pad3: 0,
        });

        // Per-head QK RMSNorm params (Qwen 3): shared across all layers, `dim = head_dim`.
        let qk_norm_params_buf = engine.make_persistent_uniform(&RmsnormParams {
            dim: config.head_dim,
            eps: config.eps,
            batch_size: 1,
            _pad3: 0,
        });

        // Output head: separate rmsnorm + matvec
        let output_norm_bg = Self::build_rmsnorm_bg(
            &engine,
            &hidden,
            &output_norm_weight,
            &norm_buf,
            &rmsnorm_params_buf,
        );
        let output_proj_bg =
            Self::build_matvec_bg(&engine, &output_proj_weight, &norm_buf, &logits);

        // --- Per-layer bind groups ---
        let kv_append_layout = engine.kv_cache_append_pipeline.get_bind_group_layout(0);
        let attn_layout = engine.attention_pipeline.get_bind_group_layout(0);
        let conv1d_layout = engine.conv1d_causal_pipeline.get_bind_group_layout(0);
        let deltanet_layout = engine.gated_deltanet_pipeline.get_bind_group_layout(0);

        let mut layer_bgs = Vec::with_capacity(config.num_layers);
        let mut deltanet_bgs: Vec<DeltaNetLayerBGs> = Vec::new();
        // Bonsai GGUFs omit the conv1d bias tensor; store zero-filled fallback
        // buffers here so their lifetime extends past bind group creation.
        // Populated only for Bonsai DeltaNet layers, empty for standard Qwen 3.5.
        let mut bonsai_conv1d_bias_zero_buffers: Vec<GpuBuffer> = Vec::new();
        let mut dn_idx: usize = 0; // index into deltanet_layer_weights
        let mut attn_kv_idx: usize = 0; // index into k_caches/v_caches

        // Persistent DeltaNet uniform params
        let dn_conv_params_buf = make_persistent(bytemuck::cast_slice(&[
            dn_conv_dim as u32,
            0u32,
            0u32,
            0u32, // dim, ring_pos, pad, pad
        ]));
        // Layout: (num_heads, qk_dim, v_dim, is_bonsai, num_kv_heads, pad, pad, pad).
        // Phase X.3.e.3.11 fix: `num_heads` is num_v_heads (32 for Qwen 3.5,
        // 48 for Bonsai 27B); shader iterates one workgroup per V-head and
        // computes cyclic KV slice via `head % num_kv_heads` to match
        // reference qwen35.cpp:547-548 `ggml_repeat_4d` + fused GDN
        // `iv1 % neq1`. Previous code passed dn_num_kv_heads (16) which
        // meant only half the V-heads were processed.
        //
        // `is_bonsai` = 1 when the GGUF is Bonsai / Qwen 3.6-27B (detected
        // by `blk.0.attn_qkv.weight` presence — see
        // `is_bonsai_deltanet_gguf` above). Consumed by `gated_deltanet.wgsl`
        // to skip internal silu(q) / silu(k) / silu(z) — those are applied
        // externally in the Bonsai path (silu_inplace on conv output before
        // the recurrence, silu_gate_apply after ssm_norm).
        let dn_params_buf = make_persistent(bytemuck::cast_slice(&[
            dn_num_v_heads as u32,
            dn_qk_dim as u32,
            dn_v_dim as u32,
            u32::from(is_bonsai_deltanet_gguf),
            dn_num_kv_heads as u32,
            0u32,
            0u32,
            0u32,
        ]));

        for i in 0..config.num_layers {
            match layer_types[i] {
                LayerType::Attention => {
                    let lw = &layer_weights[i];

                    let attn_norm_bg = Self::build_rmsnorm_bg(
                        &engine,
                        &hidden,
                        &lw.attn_norm,
                        &norm_buf,
                        &rmsnorm_params_buf,
                    );
                    let q_proj = Self::build_matvec_bg(&engine, &lw.q_proj, &norm_buf, &q_buf);
                    let k_proj = Self::build_matvec_bg(&engine, &lw.k_proj, &norm_buf, &k_buf);
                    let v_proj = Self::build_matvec_bg(&engine, &lw.v_proj, &norm_buf, &v_buf);
                    let o_proj = Self::build_matvec_bg(&engine, &lw.o_proj, &attn_out, &o_buf);

                    // Bonsai / Qwen 3.6-27B full-attention gate: (a) gate matvec
                    // dispatches `bonsai_attn_q_gate` weight → gate_buf (reusing
                    // the FFN SwiGLU gate_buf, which is intermediate_dim ≥
                    // hidden_dim so it can hold the gate output); (b) silu
                    // modulation bind group for `silu_gate_apply(attn_out,
                    // gate_buf)` applied after standard attention, before
                    // o_proj. Constructed only when `bonsai_attn_q_gate` is
                    // loaded (Bonsai full-attn layer).
                    let (bonsai_attn_q_gate_proj, bonsai_full_attn_silu_gate_bg) =
                        if let Some(gate_weight) = lw.bonsai_attn_q_gate.as_ref() {
                            let gate_proj_mv =
                                Self::build_matvec_bg(&engine, gate_weight, &norm_buf, &gate_buf);
                            // Phase X.3.e.3.16 fix: use sigmoid_gate_apply
                            // pipeline layout (Qwen 3.5 / 3.6 / Bonsai gated
                            // attention uses sigmoid, not silu). Field name
                            // `bonsai_full_attn_silu_gate_bg` kept for
                            // continuity but semantic is sigmoid now.
                            let silu_gate_layout =
                                engine.sigmoid_gate_apply_pipeline.get_bind_group_layout(0);
                            let silu_gate_bg =
                                engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: None,
                                    layout: &silu_gate_layout,
                                    entries: &[
                                        wgpu::BindGroupEntry {
                                            binding: 0,
                                            resource: attn_out.buffer.as_entire_binding(),
                                        },
                                        wgpu::BindGroupEntry {
                                            binding: 1,
                                            resource: gate_buf.buffer.as_entire_binding(),
                                        },
                                    ],
                                });
                            (Some(gate_proj_mv), Some(silu_gate_bg))
                        } else {
                            (None, None)
                        };

                    // Qwen 2 / 2.5 attention QKV biases (None for Llama / Mistral /
                    // Gemma / Qwen 3+ which do not carry attention biases).
                    let q_bias_bg = lw
                        .q_bias
                        .as_ref()
                        .map(|b| Self::build_add_bias_bg(&engine, &q_buf, b));
                    let k_bias_bg = lw
                        .k_bias
                        .as_ref()
                        .map(|b| Self::build_add_bias_bg(&engine, &k_buf, b));
                    let v_bias_bg = lw
                        .v_bias
                        .as_ref()
                        .map(|b| Self::build_add_bias_bg(&engine, &v_buf, b));

                    // Qwen 3 per-head QK RMSNorm (None for arch without QK norms).
                    let q_norm_bg = lw
                        .q_norm
                        .as_ref()
                        .map(|w| Self::build_qk_norm_bg(&engine, &q_buf, w, &qk_norm_params_buf));
                    let k_norm_bg = lw
                        .k_norm
                        .as_ref()
                        .map(|w| Self::build_qk_norm_bg(&engine, &k_buf, w, &qk_norm_params_buf));

                    let ffn_norm_bg = Self::build_rmsnorm_bg(
                        &engine,
                        &hidden,
                        &lw.ffn_norm,
                        &norm_buf,
                        &rmsnorm_params_buf,
                    );
                    let swiglu = Self::build_swiglu_bg(
                        &engine,
                        &lw.gate_proj,
                        &lw.up_proj,
                        &norm_buf,
                        &gate_buf,
                    );
                    let down_proj =
                        Self::build_matvec_bg(&engine, &lw.down_proj, &gate_buf, &down_buf);

                    let kv_append_bg =
                        engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &kv_append_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: k_buf.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: v_buf.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: k_caches[attn_kv_idx].buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: v_caches[attn_kv_idx].buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: kv_append_params_buf.as_entire_binding(),
                                },
                            ],
                        });
                    let attention_bg =
                        engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &attn_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: q_buf.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: k_caches[attn_kv_idx].buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: v_caches[attn_kv_idx].buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: attn_out.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: attn_params_buf.as_entire_binding(),
                                },
                            ],
                        });
                    attn_kv_idx += 1;

                    layer_bgs.push(LayerBGs {
                        attn_norm_bg,
                        q_proj,
                        k_proj,
                        v_proj,
                        o_proj,
                        bonsai_attn_q_gate_proj,
                        bonsai_full_attn_silu_gate_bg,
                        q_bias_bg,
                        k_bias_bg,
                        v_bias_bg,
                        q_norm_bg,
                        k_norm_bg,
                        kv_append_bg,
                        attention_bg,
                        ffn_norm_bg,
                        swiglu,
                        down_proj,
                    });
                }
                LayerType::DeltaNet => {
                    let dlw = &deltanet_layer_weights[dn_idx];

                    // DeltaNet attention sub-block bind groups
                    let attn_norm_bg = Self::build_rmsnorm_bg(
                        &engine,
                        &hidden,
                        &dlw.attn_norm,
                        &norm_buf,
                        &rmsnorm_params_buf,
                    );
                    // Standard Qwen 3.5 uses the fused `ssm_in`; Bonsai splits
                    // it into `attn_qkv` + `attn_gate`. Build the corresponding
                    // MatvecBGs only for the tensors that are present.
                    let ssm_in_proj = dlw
                        .ssm_in
                        .as_ref()
                        .map(|w| Self::build_matvec_bg(&engine, w, &norm_buf, &q_buf));
                    // Bonsai: `attn_qkv` produces Q+K+V into q_buf, `attn_gate`
                    // produces Z into v_buf. The output-buffer choice for
                    // `attn_gate` follows the standard forward path's `z` slice
                    // pattern (v_buf holds z for the deltanet recurrence).
                    let attn_qkv_proj = dlw
                        .attn_qkv
                        .as_ref()
                        .map(|w| Self::build_matvec_bg(&engine, w, &norm_buf, &q_buf));
                    let attn_gate_proj = dlw
                        .attn_gate
                        .as_ref()
                        .map(|w| Self::build_matvec_bg(&engine, w, &norm_buf, &v_buf));

                    let alpha_proj_mv =
                        Self::build_matvec_bg(&engine, &dlw.alpha_proj, &norm_buf, &alpha_buf);
                    let beta_proj_mv =
                        Self::build_matvec_bg(&engine, &dlw.beta_proj, &norm_buf, &beta_buf);

                    // Conv1d bind group: x=q_buf, state=conv_state, weight/bias, out=k_buf.
                    //
                    // Bonsai GGUFs omit the conv1d bias tensor (dlw.conv1d_bias
                    // is None). Since the WGSL shader expects a bias buffer at
                    // binding 3, we allocate a zero-filled fallback of size
                    // `conv_dim` for those layers. The Bonsai forward path is
                    // not yet wired anyway (Phase 2 Step 4) — this allocation
                    // is a no-op cost at load time, and the zero bias produces
                    // mathematically equivalent output to omitting the bias
                    // step in the CPU implementation.
                    // Bonsai GGUFs omit the conv1d bias tensor. Push a
                    // zero-filled fallback to the persistent Vec first, then
                    // borrow from the Vec so the buffer outlives the bind
                    // group creation.
                    if dlw.conv1d_bias.is_none() {
                        let zero = vec![0.0f32; dn_conv_dim as usize];
                        bonsai_conv1d_bias_zero_buffers.push(engine.upload_f32(&zero));
                    }
                    let conv1d_bias_buf = dlw
                        .conv1d_bias
                        .as_ref()
                        .unwrap_or_else(|| bonsai_conv1d_bias_zero_buffers.last().unwrap());

                    let conv1d_bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &conv1d_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: q_buf.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: deltanet_conv_states[dn_idx].buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: dlw.conv1d_weight.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: conv1d_bias_buf.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: k_buf.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: dn_conv_params_buf.as_entire_binding(),
                            },
                        ],
                    });

                    // DeltaNet recurrence bind group (Phase X.3.e.3.18 rework).
                    //
                    // Reference (CPU llama3.rs:4133-4136, after conv1d):
                    //   q_slice = dn_conv_out[0..q_dim]              // post-conv1d Q
                    //   k_slice = dn_conv_out[q_dim..2*q_dim]        // post-conv1d K
                    //   v_slice = dn_conv_out[2*q_dim..qkv_len]      // post-conv1d V
                    //   z_slice = dn_in_proj[qkv_len..qkv_len+v_out] // pre-conv1d Z
                    //
                    // GPU mapping:
                    //   - `k_buf` receives conv1d output (Q+K+V packed, Z portion
                    //     unchanged from prior contents)
                    //   - For standard Qwen 3.5 (`ssm_in.is_some()`): Z lives at
                    //     q_buf offset qkv_len (fused matvec output layout)
                    //   - For Bonsai (`attn_qkv.is_some() + attn_gate.is_some()`):
                    //     Z lives at v_buf offset 0 (`attn_gate_proj` output)
                    //
                    // Prior code bound all four (q/k/v/z) to entire buffers,
                    // meaning shader indexing `kv_head*qk_dim` picked wrong
                    // slices (K, V, Z reads were nonsense). This was the root
                    // cause of GPU DeltaNet garbage output.
                    let q_dim_bytes = (dn_qk_dim * dn_num_kv_heads * 4) as u64;
                    let v_out_bytes = (dn_v_dim * dn_num_v_heads * 4) as u64;
                    let qkv_len_bytes = 2 * q_dim_bytes + v_out_bytes;
                    let z_buffer_ref = if dlw.attn_gate.is_some() {
                        &v_buf.buffer // Bonsai: attn_gate output in v_buf
                    } else {
                        &q_buf.buffer // Standard Qwen 3.5: fused ssm_in output in q_buf
                    };
                    let z_offset_bytes = if dlw.attn_gate.is_some() {
                        0
                    } else {
                        qkv_len_bytes
                    };
                    let deltanet_bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &deltanet_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &k_buf.buffer,
                                    offset: 0,
                                    size: std::num::NonZeroU64::new(q_dim_bytes),
                                }),
                            }, // q = post-conv1d Q
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &k_buf.buffer,
                                    offset: q_dim_bytes,
                                    size: std::num::NonZeroU64::new(q_dim_bytes),
                                }),
                            }, // k = post-conv1d K
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &k_buf.buffer,
                                    offset: 2 * q_dim_bytes,
                                    size: std::num::NonZeroU64::new(v_out_bytes),
                                }),
                            }, // v = post-conv1d V
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: alpha_buf.buffer.as_entire_binding(),
                            }, // alpha (populated by alpha_proj_mv)
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: beta_buf.buffer.as_entire_binding(),
                            }, // beta (populated by beta_proj_mv)
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: z_buffer_ref,
                                    offset: z_offset_bytes,
                                    size: std::num::NonZeroU64::new(v_out_bytes),
                                }),
                            }, // z = pre-conv1d Z (from ssm_in_proj tail or attn_gate)
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: deltanet_states[dn_idx].buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: attn_out.buffer.as_entire_binding(),
                            }, // output
                            wgpu::BindGroupEntry {
                                binding: 8,
                                resource: dn_params_buf.as_entire_binding(),
                            },
                        ],
                    });

                    // Detect the Bonsai / Qwen 3.6 SSM refinement path.
                    // Gated on the presence of `ssm_a` + `ssm_dt_bias` +
                    // `ssm_norm` tensors — Qwen 3.5-4B and Qwen 3.6-27B both
                    // ship these (Phase X.3.e.3.23 tensor-name fix moved
                    // Qwen 3.5-4B into the same Bonsai path as the 27B
                    // variant), matching the CPU `is_bonsai_path` gate at
                    // `llama3.rs:4082` `ssm_a.is_some() && ssm_dt_bias.is_some()`.
                    // Determined here (moved up from below `ssm_out_proj`
                    // build) so the Bonsai flow can route `ssm_out_proj` to
                    // read from the post-`ssm_norm`/post-silu(z) scratch
                    // `attn_out_normed` — while non-Bonsai layers keep
                    // reading `attn_out` directly.
                    let is_bonsai_layer = dlw.ssm_a.is_some() && dlw.ssm_dt_bias.is_some();

                    let ssm_out_proj = if is_bonsai_layer {
                        Self::build_matvec_bg(&engine, &dlw.ssm_out, &attn_out_normed, &o_buf)
                    } else {
                        Self::build_matvec_bg(&engine, &dlw.ssm_out, &attn_out, &o_buf)
                    };
                    let bonsai_beta_sigmoid_bg = if is_bonsai_layer {
                        let layout = engine.beta_sigmoid_pipeline.get_bind_group_layout(0);
                        Some(engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &layout,
                            entries: &[wgpu::BindGroupEntry {
                                binding: 0,
                                resource: beta_buf.buffer.as_entire_binding(),
                            }],
                        }))
                    } else {
                        None
                    };
                    let bonsai_ssm_discretisation_bg = if is_bonsai_layer {
                        let layout = engine.ssm_discretisation_pipeline.get_bind_group_layout(0);
                        // Both ssm_a and ssm_dt_bias are present when has_attn_qkv is
                        // true (Bonsai GGUFs ship all three refinement tensors).
                        let ssm_dt_bias = dlw
                            .ssm_dt_bias
                            .as_ref()
                            .expect("Bonsai layer requires ssm_dt_bias");
                        let ssm_a = dlw.ssm_a.as_ref().expect("Bonsai layer requires ssm_a");
                        Some(engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: alpha_buf.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: ssm_dt_bias.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: ssm_a.buffer.as_entire_binding(),
                                },
                            ],
                        }))
                    } else {
                        None
                    };
                    let bonsai_silu_inplace_kbuf_bg = if is_bonsai_layer {
                        let layout = engine.silu_inplace_pipeline.get_bind_group_layout(0);
                        Some(engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &layout,
                            entries: &[wgpu::BindGroupEntry {
                                binding: 0,
                                resource: k_buf.buffer.as_entire_binding(),
                            }],
                        }))
                    } else {
                        None
                    };
                    let bonsai_ssm_norm_per_head_bg = if is_bonsai_layer {
                        // Per-V-head RMSNorm on attn_out with ssm_norm weight.
                        // Uses the shared rmsnorm_pipeline; params are the
                        // per-Bonsai-layer RmsnormParams with dim=v_dim +
                        // batch_size=num_v_heads (allocated per layer to
                        // reference by the bind group's uniform slot).
                        let layout = engine.rmsnorm_pipeline.get_bind_group_layout(0);
                        let ssm_norm = dlw
                            .ssm_norm
                            .as_ref()
                            .expect("Bonsai layer requires ssm_norm");
                        // Persistent uniform: dim=v_dim, eps=norm_eps,
                        // batch_size=num_v_heads. Reused across every forward
                        // (the values are constant at load time).
                        let params_buf = engine.make_persistent_uniform(&RmsnormParams {
                            dim: dn_v_dim as u32,
                            eps: config.eps,
                            batch_size: dn_num_v_heads as u32,
                            _pad3: 0,
                        });
                        let bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: attn_out.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: ssm_norm.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    // Output to a dedicated scratch — WebGPU
                                    // rejects aliasing `read` binding 0 and
                                    // `read_write` binding 2 on the same
                                    // `attn_out` buffer within a compute
                                    // pass. `attn_out_normed` is sized to
                                    // hold `num_v_heads * v_dim` (see
                                    // scratch alloc above).
                                    resource: attn_out_normed.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: params_buf.as_entire_binding(),
                                },
                            ],
                        });
                        drop(params_buf); // BindGroup holds Arc ref
                        Some(bg)
                    } else {
                        None
                    };
                    let bonsai_silu_gate_apply_bg = if is_bonsai_layer {
                        let layout = engine.silu_gate_apply_pipeline.get_bind_group_layout(0);
                        Some(engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    // Read from the post-ssm_norm scratch
                                    // (`attn_out_normed`) so the gate
                                    // applies to the correct DeltaNet path.
                                    resource: attn_out_normed.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: v_buf.buffer.as_entire_binding(),
                                },
                            ],
                        }))
                    } else {
                        None
                    };

                    // FFN bind groups (same pattern as attention layers)
                    let ffn_norm_bg = Self::build_rmsnorm_bg(
                        &engine,
                        &hidden,
                        &dlw.ffn_norm,
                        &norm_buf,
                        &rmsnorm_params_buf,
                    );
                    let swiglu = Self::build_swiglu_bg(
                        &engine,
                        &dlw.gate_proj,
                        &dlw.up_proj,
                        &norm_buf,
                        &gate_buf,
                    );
                    let down_proj =
                        Self::build_matvec_bg(&engine, &dlw.down_proj, &gate_buf, &down_buf);

                    deltanet_bgs.push(DeltaNetLayerBGs {
                        attn_norm_bg,
                        ssm_in_proj,
                        attn_qkv_proj,
                        attn_gate_proj,
                        bonsai_beta_sigmoid_bg,
                        bonsai_ssm_discretisation_bg,
                        bonsai_silu_inplace_kbuf_bg,
                        bonsai_ssm_norm_per_head_bg,
                        bonsai_silu_gate_apply_bg,
                        alpha_proj_mv,
                        beta_proj_mv,
                        conv1d_bg,
                        deltanet_bg,
                        ssm_out_proj,
                        ffn_norm_bg,
                        swiglu,
                        down_proj,
                    });

                    // Push dummy attention LayerBGs (unused but keeps index alignment)
                    layer_bgs.push(LayerBGs {
                        attn_norm_bg: engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &engine.rmsnorm_pipeline.get_bind_group_layout(0),
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: hidden.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: hidden.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: norm_buf.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: rmsnorm_params_buf.as_entire_binding(),
                                },
                            ],
                        }),
                        q_proj: Self::build_matvec_bg(
                            &engine,
                            &layer_weights[i].q_proj,
                            &norm_buf,
                            &q_buf,
                        ),
                        k_proj: Self::build_matvec_bg(
                            &engine,
                            &layer_weights[i].k_proj,
                            &norm_buf,
                            &k_buf,
                        ),
                        v_proj: Self::build_matvec_bg(
                            &engine,
                            &layer_weights[i].v_proj,
                            &norm_buf,
                            &v_buf,
                        ),
                        o_proj: Self::build_matvec_bg(
                            &engine,
                            &layer_weights[i].o_proj,
                            &attn_out,
                            &o_buf,
                        ),
                        // DeltaNet layers do not carry attention biases (unused placeholder).
                        q_bias_bg: None,
                        k_bias_bg: None,
                        v_bias_bg: None,
                        // DeltaNet layers do not carry per-head QK norms (unused placeholder).
                        bonsai_attn_q_gate_proj: None,
                        bonsai_full_attn_silu_gate_bg: None,
                        q_norm_bg: None,
                        k_norm_bg: None,
                        kv_append_bg: engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &kv_append_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: k_buf.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: v_buf.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: k_caches[0].buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: v_caches[0].buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: kv_append_params_buf.as_entire_binding(),
                                },
                            ],
                        }),
                        attention_bg: engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &attn_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: q_buf.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: k_caches[0].buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: v_caches[0].buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: attn_out.buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: attn_params_buf.as_entire_binding(),
                                },
                            ],
                        }),
                        ffn_norm_bg: Self::build_rmsnorm_bg(
                            &engine,
                            &hidden,
                            &dlw.ffn_norm,
                            &norm_buf,
                            &rmsnorm_params_buf,
                        ),
                        swiglu: Self::build_swiglu_bg(
                            &engine,
                            &dlw.gate_proj,
                            &dlw.up_proj,
                            &norm_buf,
                            &gate_buf,
                        ),
                        down_proj: Self::build_matvec_bg(
                            &engine,
                            &dlw.down_proj,
                            &gate_buf,
                            &down_buf,
                        ),
                    });

                    dn_idx += 1;
                }
            }
        }

        // Dispatch constants
        let residual_dispatch_x = ((config.hidden_dim as u32) + 255) / 256;
        let rope_q_dispatch_x = (config.num_heads * config.head_dim / 2 + 255) / 256;
        let rope_k_dispatch_x = (config.num_kv_heads * config.head_dim / 2 + 255) / 256;
        let kv_dispatch_x = ((kv_dim as u32) + 255) / 256;
        // Q bias is added over hidden_dim elements; K/V bias over kv_dim.
        let q_bias_dispatch_x = ((config.hidden_dim as u32) + 255) / 256;
        let kv_bias_dispatch_x = ((kv_dim as u32) + 255) / 256;

        let total_bgs = layer_bgs.len() * 10 + 7;
        eprintln!("[GpuModel] {total_bgs} BGs pre-cached (fused SwiGLU), 4 persistent UBs");

        // Capture the "has DeltaNet layers" flag before `deltanet_bgs` is
        // moved into the struct — the struct literal below moves the Vec
        // and we still need the boolean afterwards to gate
        // `dn_conv_params_buf`.
        let has_deltanet = !deltanet_bgs.is_empty();

        Self {
            engine,
            config,
            _layer_weights: layer_weights,
            _output_norm_weight: output_norm_weight,
            _output_proj_weight: output_proj_weight,
            hidden,
            _norm_buf: norm_buf,
            _q_buf: q_buf,
            _k_buf: k_buf,
            _v_buf: v_buf,
            _attn_out: attn_out,
            _attn_out_normed: attn_out_normed,
            _o_buf: o_buf,
            _gate_buf: gate_buf,
            _down_buf: down_buf,
            _alpha_buf: alpha_buf,
            _beta_buf: beta_buf,
            logits,
            staging,
            hidden_staging: std::cell::OnceCell::new(),
            k_caches,
            v_caches,
            deltanet_states,
            deltanet_conv_states,
            _deltanet_layer_weights: deltanet_layer_weights,
            deltanet_layer_bgs: deltanet_bgs,
            layer_types,
            layer_bgs,
            rope_q_bg,
            rope_k_bg,
            residual_attn_bg,
            residual_ffn_bg,
            output_norm_bg,
            output_proj_bg,
            rope_q_params_buf,
            rope_k_params_buf,
            kv_append_params_buf,
            attn_params_buf,
            dn_conv_params_buf: if has_deltanet {
                Some(dn_conv_params_buf)
            } else {
                None
            },
            residual_dispatch_x,
            rope_q_dispatch_x,
            rope_k_dispatch_x,
            kv_dispatch_x,
            q_bias_dispatch_x,
            kv_bias_dispatch_x,
            embedding,
            vocab_size,
            seq_len: 0,
        }
    }

    // --- Forward pass (zero per-token allocation) ---

    /// Encode the full forward pass into a command encoder.
    fn encode_forward(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_forward_impl(encoder, None);
    }

    /// Encode the standard forward but stop after the given layer's FFN
    /// residual add. Skips the final norm + output projection so the caller
    /// can inspect the `hidden` buffer at a specific layer boundary. Used
    /// exclusively by the Issue #40 layer bisection diagnostic.
    fn encode_forward_stop_after(&self, encoder: &mut wgpu::CommandEncoder, stop_after: usize) {
        self.encode_forward_impl(encoder, Some(stop_after));
    }

    fn encode_forward_impl(&self, encoder: &mut wgpu::CommandEncoder, stop_after: Option<usize>) {
        let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        let mut dn_fwd_idx: usize = 0; // index into deltanet_layer_bgs

        for i in 0..self.config.num_layers {
            match self.layer_types[i] {
                LayerType::Attention => {
                    let lbg = &self.layer_bgs[i];

                    // --- Attention sub-block ---
                    // RMSNorm: hidden → norm_buf
                    cp.set_pipeline(&self.engine.rmsnorm_pipeline);
                    cp.set_bind_group(0, &lbg.attn_norm_bg, &[]);
                    cp.dispatch_workgroups(1, 1, 1);

                    // Q/K/V projections from norm_buf
                    Self::dispatch_mv(&self.engine, &mut cp, &lbg.q_proj);
                    Self::dispatch_mv(&self.engine, &mut cp, &lbg.k_proj);
                    Self::dispatch_mv(&self.engine, &mut cp, &lbg.v_proj);

                    // Bonsai / Qwen 3.6-27B full-attention gate projection:
                    // dispatch `bonsai_attn_q_gate` matvec into gate_buf
                    // (reused FFN SwiGLU buffer; larger than hidden_dim so
                    // holds the gate output). The silu(gate) multiplication
                    // into attn_out is applied after the attention core
                    // (see silu_gate_apply dispatch below, before o_proj).
                    if let Some(gate_proj) = &lbg.bonsai_attn_q_gate_proj {
                        Self::dispatch_mv(&self.engine, &mut cp, gate_proj);
                    }

                    // Qwen 2 / 2.5 attention biases: add pointwise to Q/K/V before RoPE.
                    // No-op for arch without biases (Llama / Mistral / Gemma / Qwen 3+).
                    if lbg.q_bias_bg.is_some() || lbg.k_bias_bg.is_some() || lbg.v_bias_bg.is_some()
                    {
                        cp.set_pipeline(&self.engine.add_bias_pipeline);
                        if let Some(bg) = &lbg.q_bias_bg {
                            cp.set_bind_group(0, bg, &[]);
                            cp.dispatch_workgroups(self.q_bias_dispatch_x, 1, 1);
                        }
                        if let Some(bg) = &lbg.k_bias_bg {
                            cp.set_bind_group(0, bg, &[]);
                            cp.dispatch_workgroups(self.kv_bias_dispatch_x, 1, 1);
                        }
                        if let Some(bg) = &lbg.v_bias_bg {
                            cp.set_bind_group(0, bg, &[]);
                            cp.dispatch_workgroups(self.kv_bias_dispatch_x, 1, 1);
                        }
                    }

                    // Qwen 3 per-head QK RMSNorm: applied after bias, before RoPE.
                    // No-op for arch without QK norms.
                    if lbg.q_norm_bg.is_some() || lbg.k_norm_bg.is_some() {
                        cp.set_pipeline(&self.engine.qk_norm_pipeline);
                        if let Some(bg) = &lbg.q_norm_bg {
                            cp.set_bind_group(0, bg, &[]);
                            cp.dispatch_workgroups(self.config.num_heads, 1, 1);
                        }
                        if let Some(bg) = &lbg.k_norm_bg {
                            cp.set_bind_group(0, bg, &[]);
                            cp.dispatch_workgroups(self.config.num_kv_heads, 1, 1);
                        }
                    }

                    cp.set_pipeline(&self.engine.rope_pipeline);
                    cp.set_bind_group(0, &self.rope_q_bg, &[]);
                    cp.dispatch_workgroups(self.rope_q_dispatch_x, 1, 1);
                    cp.set_bind_group(0, &self.rope_k_bg, &[]);
                    cp.dispatch_workgroups(self.rope_k_dispatch_x, 1, 1);

                    cp.set_pipeline(&self.engine.kv_cache_append_pipeline);
                    cp.set_bind_group(0, &lbg.kv_append_bg, &[]);
                    cp.dispatch_workgroups(self.kv_dispatch_x, 1, 1);

                    cp.set_pipeline(&self.engine.attention_pipeline);
                    cp.set_bind_group(0, &lbg.attention_bg, &[]);
                    cp.dispatch_workgroups(self.config.num_heads, 1, 1);

                    // Qwen 3.5 / 3.6 / Bonsai 27B full-attention sigmoid gate
                    // modulation: `attn_out[i] *= sigmoid(gate_buf[i])` (applied
                    // AFTER the scaled-dot-product attention, BEFORE `o_proj`).
                    // Reference qwen35.cpp:401-404 uses ggml_sigmoid (not silu).
                    // Phase X.3.e.3.14 fix mirrored from CPU (`b616219` lineage).
                    // No-op for non-gated layers (bind group is None).
                    if let Some(bg) = &lbg.bonsai_full_attn_silu_gate_bg {
                        cp.set_pipeline(&self.engine.sigmoid_gate_apply_pipeline);
                        cp.set_bind_group(0, bg, &[]);
                        let attn_out_len = self.config.num_heads * self.config.head_dim;
                        cp.dispatch_workgroups((attn_out_len + 255) / 256, 1, 1);
                    }

                    Self::dispatch_mv(&self.engine, &mut cp, &lbg.o_proj);

                    cp.set_pipeline(&self.engine.residual_add_pipeline);
                    cp.set_bind_group(0, &self.residual_attn_bg, &[]);
                    cp.dispatch_workgroups(self.residual_dispatch_x, 1, 1);
                }
                LayerType::DeltaNet => {
                    // Gated DeltaNet linear attention (Qwen3.5)
                    let dbg = &self.deltanet_layer_bgs[dn_fwd_idx];

                    // 1. RMSNorm: hidden → norm_buf
                    cp.set_pipeline(&self.engine.rmsnorm_pipeline);
                    cp.set_bind_group(0, &dbg.attn_norm_bg, &[]);
                    cp.dispatch_workgroups(1, 1, 1);

                    // 2. Fused in_proj: norm_buf → q_buf (contains q, k, v, z packed).
                    //
                    // Standard Qwen 3.5: single `ssm_in_proj` produces Q+K+V+Z.
                    // Bonsai / Qwen 3.6-27B: `attn_qkv_proj` produces Q+K+V into
                    // q_buf; separate `attn_gate_proj` produces Z into v_buf.
                    let is_bonsai_layer = dbg.ssm_in_proj.is_none();
                    match (&dbg.ssm_in_proj, &dbg.attn_qkv_proj, &dbg.attn_gate_proj) {
                        (Some(mv), _, _) => {
                            Self::dispatch_mv(&self.engine, &mut cp, mv);
                        }
                        (None, Some(qkv), Some(gate)) => {
                            Self::dispatch_mv(&self.engine, &mut cp, qkv);
                            Self::dispatch_mv(&self.engine, &mut cp, gate);
                        }
                        (None, _, _) => {
                            panic!(
                                "DeltaNet layer has neither ssm_in_proj nor \
                                 (attn_qkv_proj + attn_gate_proj) — GGUF load \
                                 detection is inconsistent."
                            );
                        }
                    }

                    // 2a. Alpha decay-gate projection: norm_buf → alpha_buf ([num_v_heads]).
                    Self::dispatch_mv(&self.engine, &mut cp, &dbg.alpha_proj_mv);
                    // 2b. Beta update-rate projection: norm_buf → beta_buf ([num_v_heads]).
                    Self::dispatch_mv(&self.engine, &mut cp, &dbg.beta_proj_mv);

                    // 2c. (Bonsai only) beta_sigmoid: beta_buf = sigmoid(beta_buf)
                    //     Phase X.3.e.3 Gap B extra.
                    if let Some(bg) = &dbg.bonsai_beta_sigmoid_bg {
                        cp.set_pipeline(&self.engine.beta_sigmoid_pipeline);
                        cp.set_bind_group(0, bg, &[]);
                        let dn_num_v_heads = self
                            .config
                            .linear_num_v_heads
                            .unwrap_or(self.config.num_heads);
                        cp.dispatch_workgroups((dn_num_v_heads + 255) / 256, 1, 1);
                    }

                    // 2d. (Bonsai only) ssm_discretisation: alpha_buf =
                    //     exp(softplus(alpha + ssm_dt_bias) * ssm_a).
                    //     Phase X.3.e.3 Gap B.
                    if let Some(bg) = &dbg.bonsai_ssm_discretisation_bg {
                        cp.set_pipeline(&self.engine.ssm_discretisation_pipeline);
                        cp.set_bind_group(0, bg, &[]);
                        let dn_num_v_heads = self
                            .config
                            .linear_num_v_heads
                            .unwrap_or(self.config.num_heads);
                        cp.dispatch_workgroups((dn_num_v_heads + 255) / 256, 1, 1);
                    }

                    // 3. Causal conv1d: q_buf → k_buf (preprocessed q, k, v)
                    let dn_conv_dim = self.config.linear_qk_head_dim.unwrap_or(128)
                        * self
                            .config
                            .linear_num_kv_heads
                            .unwrap_or(self.config.num_kv_heads)
                        * 2
                        + self.config.linear_kv_head_dim.unwrap_or(128)
                            * self
                                .config
                                .linear_num_v_heads
                                .unwrap_or(self.config.num_heads);
                    let conv1d_dispatch = (dn_conv_dim + 255) / 256;
                    cp.set_pipeline(&self.engine.conv1d_causal_pipeline);
                    cp.set_bind_group(0, &dbg.conv1d_bg, &[]);
                    cp.dispatch_workgroups(conv1d_dispatch, 1, 1);

                    // 3.5. (Bonsai only) post-conv1d silu on k_buf (Q+K+V portion).
                    //      Phase X.3.e.3 §silu(z) order (applied to conv output
                    //      before delta-rule integration).
                    if let Some(bg) = &dbg.bonsai_silu_inplace_kbuf_bg {
                        cp.set_pipeline(&self.engine.silu_inplace_pipeline);
                        cp.set_bind_group(0, bg, &[]);
                        cp.dispatch_workgroups(conv1d_dispatch, 1, 1);
                    }

                    // 4. DeltaNet recurrence: state update + gated output → attn_out.
                    //    The shader handles the Bonsai path via the `is_bonsai`
                    //    uniform (dn_params_buf fourth u32) — Phase 2 Step 4
                    //    shader-side PR #84.
                    //    Phase X.3.e.3.11 fix: dispatch one workgroup per V-head
                    //    (not KV-head) so all V-heads are processed. Cyclic
                    //    KV mapping is applied inside the shader via
                    //    `head % num_kv_heads` (params.num_kv_heads slot).
                    let dn_num_heads = self
                        .config
                        .linear_num_v_heads
                        .unwrap_or(self.config.num_heads);
                    cp.set_pipeline(&self.engine.gated_deltanet_pipeline);
                    cp.set_bind_group(0, &dbg.deltanet_bg, &[]);
                    cp.dispatch_workgroups(dn_num_heads, 1, 1);

                    // 4.5. (Bonsai only) per-V-head RMSNorm on attn_out with
                    //      ssm_norm weight. Phase X.3.e.3 Gap C.
                    if let Some(bg) = &dbg.bonsai_ssm_norm_per_head_bg {
                        cp.set_pipeline(&self.engine.rmsnorm_pipeline);
                        cp.set_bind_group(0, bg, &[]);
                        let dn_num_v_heads = self
                            .config
                            .linear_num_v_heads
                            .unwrap_or(self.config.num_heads);
                        cp.dispatch_workgroups(dn_num_v_heads, 1, 1);
                    }

                    // 4.6. (Bonsai only) silu(z) multiply: attn_out *= silu(z_buf).
                    //      Phase X.3.e.3 §silu(z) order — applied AFTER ssm_norm
                    //      (in contrast to standard Qwen 3.5 where it happens
                    //      inside gated_deltanet).
                    if let Some(bg) = &dbg.bonsai_silu_gate_apply_bg {
                        cp.set_pipeline(&self.engine.silu_gate_apply_pipeline);
                        cp.set_bind_group(0, bg, &[]);
                        let dn_num_v_heads = self
                            .config
                            .linear_num_v_heads
                            .unwrap_or(self.config.num_heads);
                        let attn_out_len =
                            dn_num_v_heads * self.config.linear_kv_head_dim.unwrap_or(128);
                        cp.dispatch_workgroups((attn_out_len + 255) / 256, 1, 1);
                    }

                    // Compile-check: silence unused warning when Bonsai path is
                    // not exercised. `is_bonsai_layer` is retained as a signal
                    // to future maintainers about the branching semantics of
                    // the pipeline sequence above.
                    let _ = is_bonsai_layer;

                    // 5. Output projection: attn_out → o_buf
                    Self::dispatch_mv(&self.engine, &mut cp, &dbg.ssm_out_proj);

                    // 6. Residual add: hidden += o_buf
                    cp.set_pipeline(&self.engine.residual_add_pipeline);
                    cp.set_bind_group(0, &self.residual_attn_bg, &[]);
                    cp.dispatch_workgroups(self.residual_dispatch_x, 1, 1);

                    dn_fwd_idx += 1;
                }
            }

            // --- FFN sub-block ---
            // Use the correct source of FFN bind groups based on layer type.
            let (ffn_norm, ffn_swiglu_bg, ffn_swiglu_dx, ffn_swiglu_dy, ffn_swiglu_quant, ffn_down) =
                match self.layer_types[i] {
                    LayerType::Attention => {
                        let lbg = &self.layer_bgs[i];
                        (
                            &lbg.ffn_norm_bg,
                            &lbg.swiglu.bg,
                            lbg.swiglu.dispatch_x,
                            lbg.swiglu.dispatch_y,
                            lbg.swiglu.quant,
                            &lbg.down_proj,
                        )
                    }
                    LayerType::DeltaNet => {
                        let dbg = &self.deltanet_layer_bgs[dn_fwd_idx - 1]; // already incremented above
                        (
                            &dbg.ffn_norm_bg,
                            &dbg.swiglu.bg,
                            dbg.swiglu.dispatch_x,
                            dbg.swiglu.dispatch_y,
                            dbg.swiglu.quant,
                            &dbg.down_proj,
                        )
                    }
                };

            cp.set_pipeline(&self.engine.rmsnorm_pipeline);
            cp.set_bind_group(0, ffn_norm, &[]);
            cp.dispatch_workgroups(1, 1, 1);

            // Phase X.3.e.3.27 fix: SwiGLU dispatch selects the fused pipeline
            // by the FFN weight quant type. Q4_K (Qwen 3.5) uses the 256-elem
            // block kernel; Q1_0 (Bonsai 27B) uses the 18-byte block kernel.
            // Bind group layouts are compatible across the two shaders (same
            // storage buffer / uniform layout), so switching pipelines just
            // reinterprets the weight bytes.
            let ffn_swiglu_pipeline = match ffn_swiglu_quant {
                GpuQuantType::Q4K => &self.engine.swiglu_fused_q4k_pipeline,
                GpuQuantType::Q1_0 => &self.engine.swiglu_fused_q1_0_pipeline,
                other => panic!(
                    "encode_forward: unexpected SwiGLU quant {other:?} — \
                     build_swiglu_bg should have panicked at load time."
                ),
            };
            cp.set_pipeline(ffn_swiglu_pipeline);
            cp.set_bind_group(0, ffn_swiglu_bg, &[]);
            cp.dispatch_workgroups(ffn_swiglu_dx, ffn_swiglu_dy, 1);

            Self::dispatch_mv(&self.engine, &mut cp, ffn_down);

            cp.set_pipeline(&self.engine.residual_add_pipeline);
            cp.set_bind_group(0, &self.residual_ffn_bg, &[]);
            cp.dispatch_workgroups(self.residual_dispatch_x, 1, 1);

            // Issue #40 layer bisection: if the caller requested a stop after
            // this layer, skip the output head. The `hidden` buffer now holds
            // this layer's post-FFN-residual output.
            if stop_after == Some(i) {
                return;
            }
        }

        // --- Output head ---
        // RMSNorm: hidden → norm_buf
        cp.set_pipeline(&self.engine.rmsnorm_pipeline);
        cp.set_bind_group(0, &self.output_norm_bg, &[]);
        cp.dispatch_workgroups(1, 1, 1);

        // Output projection from norm_buf
        Self::dispatch_mv(&self.engine, &mut cp, &self.output_proj_bg);
    }

    /// Dispatch a pre-cached matvec bind group.
    fn dispatch_mv(engine: &GpuEngine, cp: &mut wgpu::ComputePass, mv: &MatvecBG) {
        // Q1_0 uses the row4 kernel by default (Step 6, PR #74): 1 workgroup
        // produces 4 output rows, cutting workgroup count 4× and total
        // input-read traffic 4×. Empirically 1.82× vs base `matvec_q1_0`
        // on Jetson Vulkan iGPU for Bonsai 27B attn_qkv. Dispatch dims
        // pre-computed in `build_matvec_bg` for ceil(rows / 4).
        let pipeline = match mv.quant {
            GpuQuantType::Q4K => &engine.matvec_q4k_pipeline,
            GpuQuantType::Q5K => &engine.matvec_q5k_pipeline,
            GpuQuantType::Q6K => &engine.matvec_q6k_pipeline,
            GpuQuantType::Q8_0 => &engine.matvec_q8_0_pipeline,
            GpuQuantType::Q1_0 => &engine.matvec_q1_0_row4_pipeline,
        };
        cp.set_pipeline(pipeline);
        cp.set_bind_group(0, &mv.bg, &[]);
        cp.dispatch_workgroups(mv.dispatch_x, mv.dispatch_y, 1);
    }

    /// Dispatch a batch-4 matvec (K=4 unrolled scalar accumulators).
    /// Same bind group layout — only pipeline differs.
    fn dispatch_mv_batch4(engine: &GpuEngine, cp: &mut wgpu::ComputePass, mv: &MatvecBG) {
        let pipeline = match mv.quant {
            GpuQuantType::Q4K => &engine.matvec_q4k_batch4_pipeline,
            GpuQuantType::Q6K => &engine.matvec_q6k_batch4_pipeline,
            GpuQuantType::Q1_0 => &engine.matvec_q1_0_batch4_pipeline,
            // Q5_K and Q8_0 batch4 kernels are not yet implemented — the
            // speculative-decoding path (`encode_forward_batch4`) is not
            // supported on Qwen 3.5-4B-style Q4_K_M GGUFs that mix these
            // quantisations into the projection matmuls. The single-token
            // forward path (`encode_forward_impl`, `dispatch_mv`) does
            // handle them via the new pipelines.
            other @ (GpuQuantType::Q5K | GpuQuantType::Q8_0) => panic!(
                "dispatch_mv_batch4 is not implemented for {other:?} \
                 (speculative decoding with Q4_K_M mixed-quant GGUFs). \
                 Use single-token `encode_forward_impl` or implement \
                 dequant_matvec_q5k_batch4.wgsl / dequant_matvec_q8_0_batch4.wgsl."
            ),
        };
        cp.set_pipeline(pipeline);
        cp.set_bind_group(0, &mv.bg, &[]);
        cp.dispatch_workgroups(mv.dispatch_x, mv.dispatch_y, 1);
    }

    /// Update persistent uniform buffers for the current position.
    fn update_uniforms(&self, pos: u32, seq_len: u32) {
        self.engine
            .queue
            .write_buffer(&self.rope_q_params_buf, 0, bytemuck::bytes_of(&pos));
        self.engine
            .queue
            .write_buffer(&self.rope_k_params_buf, 0, bytemuck::bytes_of(&pos));
        self.engine
            .queue
            .write_buffer(&self.kv_append_params_buf, 0, bytemuck::bytes_of(&pos));
        self.engine
            .queue
            .write_buffer(&self.attn_params_buf, 0, bytemuck::bytes_of(&seq_len));
        // Phase X.3.e.3.25 fix: advance the DeltaNet conv1d ring position
        // for the current token. The CPU implementation stores `ring_pos`
        // as per-layer state and advances it after each `causal_conv1d_step`
        // (see `llama3.rs:2183` `*ring_pos = write_slot;`). On GPU we
        // computed `pos % ring_size` (= `pos % (kernel_size - 1)`) directly
        // and write it into the shared uniform before the compute pass. The
        // shader at `shaders/conv1d_causal.wgsl` reads the same slot from
        // `params.ring_pos`, so all DeltaNet layers see the correct history
        // window. Without this update `ring_pos` stayed at 0 forever, so
        // the shader wrote every token's activation into the same slot,
        // effectively discarding the conv1d history — the primary source
        // of GPU vs CPU state-accumulation divergence starting at layer 6.
        if let Some(buf) = self.dn_conv_params_buf.as_ref() {
            // Layout offset: word 1 (bytes 4..8) holds `ring_pos`.
            let ring_pos: u32 = pos % 3;
            self.engine
                .queue
                .write_buffer(buf, 4, bytemuck::bytes_of(&ring_pos));
        }
    }

    /// Upload token embedding to the hidden buffer.
    fn upload_embedding(&self, token_id: u32) {
        let start = token_id as usize * self.config.hidden_dim;
        let end = start + self.config.hidden_dim;
        self.engine.queue.write_buffer(
            &self.hidden.buffer,
            0,
            bytemuck::cast_slice(&self.embedding[start..end]),
        );
    }

    /// Phase X.3.e.3.21 diagnostic: read back the current contents of a
    /// scratch buffer for NaN source hunting. Executes a full forward first,
    /// then dumps the requested buffer's first N f32 values. Call after
    /// `forward(token)` to inspect end-of-forward state.
    pub fn debug_read_scratch(&mut self, which: &str, n_head: usize) -> Vec<f32> {
        let (buf, dim) = match which {
            "norm_buf" => (&self._norm_buf, self.config.hidden_dim),
            "q_buf" => (&self._q_buf, self.config.hidden_dim),
            "k_buf" => (
                &self._k_buf,
                (self.config.num_kv_heads * self.config.head_dim) as usize,
            ),
            "v_buf" => (
                &self._v_buf,
                (self.config.num_kv_heads * self.config.head_dim) as usize,
            ),
            "attn_out" => (&self._attn_out, self.config.hidden_dim),
            "attn_out_normed" => (&self._attn_out_normed, self.config.hidden_dim),
            "o_buf" => (&self._o_buf, self.config.hidden_dim),
            "hidden" => (&self.hidden, self.config.hidden_dim),
            "alpha_buf" => (
                &self._alpha_buf,
                self.config.linear_num_v_heads.unwrap_or(0) as usize,
            ),
            "beta_buf" => (
                &self._beta_buf,
                self.config.linear_num_v_heads.unwrap_or(0) as usize,
            ),
            _ => panic!("unknown buffer: {which}"),
        };
        let read_len = n_head.min(dim);
        let bytes = (read_len * 4) as u64;
        let staging = self.engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug_staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&buf.buffer, 0, &staging, 0, bytes);
        self.engine.queue.submit(Some(encoder.finish()));
        self.engine.map_staging(&staging, read_len)
    }

    /// Phase X.3.e.3.24 diagnostic: upload the embedding for `token_id` into
    /// `hidden` and read the first 5 dimensions. No RMSNorm applied — used
    /// to check that the embedding weight table (GGUF `token_embd.weight`)
    /// matches the CPU implementation's expected values byte-for-byte.
    pub fn debug_read_embedding_head(&mut self, token_id: u32) -> Vec<f32> {
        self.upload_embedding(token_id);
        self.debug_read_scratch("hidden", 5)
    }

    /// Phase X.3.e.3.21 diagnostic: run only layer 0's RMSNorm (attn_norm)
    /// and return the resulting norm_buf. Used to isolate whether RMSNorm
    /// or a downstream op is the NaN source.
    pub fn debug_forward_layer0_attn_norm_only(&mut self, token_id: u32) -> Vec<f32> {
        let pos = self.seq_len;
        let seq_len = pos + 1;
        self.update_uniforms(pos, seq_len);
        self.upload_embedding(token_id);

        let mut encoder = self
            .engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            // Run only the first layer's RMSNorm.
            let dbg = &self.deltanet_layer_bgs[0];
            cp.set_pipeline(&self.engine.rmsnorm_pipeline);
            cp.set_bind_group(0, &dbg.attn_norm_bg, &[]);
            cp.dispatch_workgroups(1, 1, 1);
        }
        self.engine.queue.submit(Some(encoder.finish()));

        self.debug_read_scratch("norm_buf", 5)
    }

    /// Phase X.3.e.3.23 diagnostic: run one full layer 0 forward and read
    /// every DeltaNet-pipeline intermediate scratch buffer at head-5 depth.
    /// Returned in stage order so a caller can eyeball where amplification
    /// starts:
    ///
    ///   (norm_buf, q_buf_head_after_qkv, k_buf_head_after_conv1d,
    ///    v_buf_head, alpha_buf, beta_buf,
    ///    attn_out_head_after_deltanet, o_buf_head_after_ssm_out,
    ///    hidden_after_residual)
    ///
    /// Each Vec<f32> has 5 elements. Executes the full first-token forward
    /// (not just stage-by-stage) then reads back — accurate for a healthy
    /// pipeline, useful signal even for a broken one.
    ///
    /// **Not thread-safe** and mutates `seq_len`. Reset before running any
    /// subsequent forward.
    #[allow(clippy::type_complexity)]
    pub fn debug_dump_layer0_deltanet_stages(
        &mut self,
        token_id: u32,
    ) -> (
        Vec<f32>, // norm_buf head 5
        Vec<f32>, // q_buf head 5 (after attn_qkv, pre-conv1d)
        Vec<f32>, // k_buf head 5 (after conv1d, DeltaNet Q slice)
        Vec<f32>, // v_buf head 5 (attn_gate output = z)
        Vec<f32>, // alpha_buf head 5
        Vec<f32>, // beta_buf head 5
        Vec<f32>, // attn_out head 5 (post gated_deltanet)
        Vec<f32>, // attn_out_normed head 5 (post ssm_norm + silu(z))
        Vec<f32>, // o_buf head 5 (post ssm_out_proj)
        Vec<f32>, // hidden head 5 (post residual add for layer 0)
    ) {
        // Run one full layer 0 forward.
        let hidden0 = self.forward_stop_after_layer_and_read_hidden(token_id, 0);
        // Now read the various scratch buffers that were populated in the
        // layer 0 forward. `debug_read_scratch` does a copy_buffer_to_buffer
        // → submit → map_staging cycle each call — inefficient but fine for a
        // one-off diagnostic.
        let norm = self.debug_read_scratch("norm_buf", 5);
        let q = self.debug_read_scratch("q_buf", 5);
        let k = self.debug_read_scratch("k_buf", 5);
        let v = self.debug_read_scratch("v_buf", 5);
        let alpha = self.debug_read_scratch("alpha_buf", 5);
        let beta = self.debug_read_scratch("beta_buf", 5);
        let attn_out = self.debug_read_scratch("attn_out", 5);
        let attn_out_normed = self.debug_read_scratch("attn_out_normed", 5);
        let o_buf_head = self.debug_read_scratch("o_buf", 5);
        // The `hidden0` return of `forward_stop_after_layer_and_read_hidden`
        // is the entire hidden vector — take the first 5.
        let hidden_head = hidden0.into_iter().take(5).collect();
        (
            norm,
            q,
            k,
            v,
            alpha,
            beta,
            attn_out,
            attn_out_normed,
            o_buf_head,
            hidden_head,
        )
    }

    /// Phase X.3.e.3.21 diagnostic: run RMSNorm + attn_qkv projection,
    /// return q_buf head (first N values). Isolates whether matvec Q4_K
    /// dequantization is producing NaN.
    pub fn debug_forward_layer0_qkv_only(&mut self, token_id: u32) -> Vec<f32> {
        let pos = self.seq_len;
        let seq_len = pos + 1;
        self.update_uniforms(pos, seq_len);
        self.upload_embedding(token_id);

        let mut encoder = self
            .engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            let dbg = &self.deltanet_layer_bgs[0];
            cp.set_pipeline(&self.engine.rmsnorm_pipeline);
            cp.set_bind_group(0, &dbg.attn_norm_bg, &[]);
            cp.dispatch_workgroups(1, 1, 1);
            // Fused in_proj (Standard) or attn_qkv (Bonsai) writes to q_buf.
            if let Some(ref mv) = dbg.ssm_in_proj {
                Self::dispatch_mv(&self.engine, &mut cp, mv);
            } else if let Some(ref qkv) = dbg.attn_qkv_proj {
                Self::dispatch_mv(&self.engine, &mut cp, qkv);
            }
        }
        self.engine.queue.submit(Some(encoder.finish()));

        self.debug_read_scratch("q_buf", 5)
    }

    /// Execute forward pass without logits readback.
    ///
    /// Does NOT poll — GPU commands are queued in-order, so subsequent
    /// `write_buffer` + `submit` calls are safe without explicit sync.
    /// Call `sync()` if you need to ensure completion.
    pub fn forward(&mut self, token_id: u32) {
        let pos = self.seq_len;
        let seq_len = pos + 1;

        self.update_uniforms(pos, seq_len);
        self.upload_embedding(token_id);

        let mut encoder = self
            .engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_forward(&mut encoder);
        self.engine.queue.submit(Some(encoder.finish()));

        self.seq_len = seq_len;
    }

    /// Wait for all queued GPU work to complete.
    pub fn sync(&self) {
        self.engine.device.poll(wgpu::Maintain::Wait);
    }

    /// Execute forward pass with logits readback.
    pub fn forward_and_read(&mut self, token_id: u32) -> Vec<f32> {
        let pos = self.seq_len;
        let seq_len = pos + 1;

        self.update_uniforms(pos, seq_len);
        self.upload_embedding(token_id);

        let mut encoder = self
            .engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_forward(&mut encoder);

        let size = (self.vocab_size * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.logits.buffer, 0, &self.staging, 0, size);

        self.engine.queue.submit(Some(encoder.finish()));
        self.seq_len = seq_len;
        self.engine.map_staging(&self.staging, self.vocab_size)
    }

    /// Issue #40 layer bisection diagnostic. Executes forward but stops
    /// after the given layer's FFN residual add, then returns that layer's
    /// output hidden state. Skips the output head entirely so downstream
    /// state is untouched.
    ///
    /// **KV cache side effect**: only layers `0..=stop_after` append K/V
    /// at the current position. The caller MUST call `reset()` before any
    /// subsequent forward — otherwise later attention layers will read
    /// stale K/V at this position. This method advances `seq_len` regardless.
    pub fn forward_stop_after_layer_and_read_hidden(
        &mut self,
        token_id: u32,
        stop_after: usize,
    ) -> Vec<f32> {
        assert!(
            stop_after < self.config.num_layers,
            "stop_after ({stop_after}) must be < num_layers ({})",
            self.config.num_layers,
        );

        let pos = self.seq_len;
        let seq_len = pos + 1;

        self.update_uniforms(pos, seq_len);
        self.upload_embedding(token_id);

        let hidden_dim = self.config.hidden_dim;
        let hidden_size = (hidden_dim * 4) as u64;
        let hidden_staging = self.hidden_staging.get_or_init(|| {
            self.engine.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hidden_staging"),
                size: hidden_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let mut encoder = self
            .engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_forward_stop_after(&mut encoder, stop_after);

        encoder.copy_buffer_to_buffer(&self.hidden.buffer, 0, hidden_staging, 0, hidden_size);

        self.engine.queue.submit(Some(encoder.finish()));
        self.seq_len = seq_len;

        self.engine.map_staging(hidden_staging, hidden_dim)
    }

    /// Issue #40 diagnostic. Executes forward and returns both the
    /// post-final-RMSNorm hidden state (the buffer that feeds output_proj)
    /// AND the final logits. Used to isolate whether the CPU/GPU logit gap
    /// originates in the layer stack (hidden state drift) or in the Q6_K
    /// output projection.
    ///
    /// The hidden buffer is `hidden_dim` f32 values; logits are `vocab_size`.
    /// Adds two `copy_buffer_to_buffer` steps to the encoder but is otherwise
    /// the same as `forward_and_read`.
    pub fn forward_and_read_hidden_and_logits(&mut self, token_id: u32) -> (Vec<f32>, Vec<f32>) {
        let pos = self.seq_len;
        let seq_len = pos + 1;

        self.update_uniforms(pos, seq_len);
        self.upload_embedding(token_id);

        // Lazy-allocate the hidden staging buffer on first call.
        let hidden_dim = self.config.hidden_dim;
        let hidden_size = (hidden_dim * 4) as u64;
        let hidden_staging = self.hidden_staging.get_or_init(|| {
            self.engine.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hidden_staging"),
                size: hidden_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let mut encoder = self
            .engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_forward(&mut encoder);

        // Capture post-final-RMSNorm state BEFORE output projection reads
        // it. Safe because output_proj only reads norm_buf (line 2687-2692 in
        // encode_forward) — no subsequent writer clobbers it.
        encoder.copy_buffer_to_buffer(&self._norm_buf.buffer, 0, hidden_staging, 0, hidden_size);

        let logits_size = (self.vocab_size * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.logits.buffer, 0, &self.staging, 0, logits_size);

        self.engine.queue.submit(Some(encoder.finish()));
        self.seq_len = seq_len;

        let hidden = self.engine.map_staging(hidden_staging, hidden_dim);
        let logits = self.engine.map_staging(&self.staging, self.vocab_size);
        (hidden, logits)
    }

    /// Reset KV caches and position counter.
    ///
    /// Phase X.3.e.3.20 fix: KV caches are only allocated for attention
    /// layers (`n_attn_layers`), not all layers. Previous code iterated
    /// `0..num_layers` (e.g., 32 for Qwen 3.5-4B) but `k_caches.len() ==
    /// n_attn_layers` (8) → index-out-of-bounds panic on any reset() call
    /// (encountered via `--layer-bisect` mode).
    pub fn reset(&mut self) {
        self.seq_len = 0;
        let kv_dim = (self.config.num_kv_heads * self.config.head_dim) as usize;
        let zeros = vec![0.0f32; self.config.max_seq_len * kv_dim];
        for i in 0..self.k_caches.len() {
            self.engine.write_f32(&self.k_caches[i], &zeros);
            self.engine.write_f32(&self.v_caches[i], &zeros);
        }
        // Phase X.3.e.3.25 fix: also zero out the DeltaNet recurrent /
        // conv1d states so multi-run tests (`--layer-bisect` prefill × 5)
        // don't carry leftover state from a prior forward. The conv1d
        // ring position resets naturally on the next `update_uniforms`
        // (pos = 0 → ring_pos = 0), but the state buffers themselves
        // need to be zeroed here.
        for state in &self.deltanet_states {
            let zeros = vec![0.0f32; state.len];
            self.engine.write_f32(state, &zeros);
        }
        for state in &self.deltanet_conv_states {
            let zeros = vec![0.0f32; state.len];
            self.engine.write_f32(state, &zeros);
        }
    }

    /// Rollback KV cache to a previous position (for speculative decoding).
    /// GPU KV memory beyond `pos` is stale but harmless — attention shader
    /// only reads up to `seq_len`, and future appends overwrite old entries.
    pub fn rollback_to(&mut self, pos: u32) {
        debug_assert!(
            pos <= self.seq_len,
            "rollback_to({pos}) > seq_len({})",
            self.seq_len
        );
        self.seq_len = pos;
    }

    /// Current sequence position.
    pub fn position(&self) -> u32 {
        self.seq_len
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Profiled forward pass: individual dispatch + sync per operation.
    /// Returns per-category timing breakdown.
    pub fn forward_profiled(&mut self, token_id: u32) -> ProfileResult {
        let pos = self.seq_len;
        let seq_len = pos + 1;

        self.update_uniforms(pos, seq_len);
        self.upload_embedding(token_id);

        let mut r = ProfileResult {
            rmsnorm_us: 0.0,
            matvec_q4k_us: 0.0,
            matvec_q6k_us: 0.0,
            swiglu_us: 0.0,
            rope_us: 0.0,
            kv_append_us: 0.0,
            attention_us: 0.0,
            residual_us: 0.0,
            rmsnorm_n: 0,
            matvec_q4k_n: 0,
            matvec_q6k_n: 0,
            swiglu_n: 0,
            rope_n: 0,
            kv_append_n: 0,
            attention_n: 0,
            residual_n: 0,
        };

        let e = &self.engine;

        for i in 0..self.config.num_layers {
            let lbg = &self.layer_bgs[i];

            // --- Attention sub-block ---
            r.rmsnorm_us += timed_dispatch(e, &e.rmsnorm_pipeline, &lbg.attn_norm_bg, 1, 1, 1);
            r.rmsnorm_n += 1;

            for mv in [&lbg.q_proj, &lbg.k_proj, &lbg.v_proj] {
                let (us, qt) = timed_mv(e, mv);
                match qt {
                    GpuQuantType::Q4K => {
                        r.matvec_q4k_us += us;
                        r.matvec_q4k_n += 1;
                    }
                    GpuQuantType::Q6K => {
                        r.matvec_q6k_us += us;
                        r.matvec_q6k_n += 1;
                    }
                    GpuQuantType::Q5K | GpuQuantType::Q8_0 | GpuQuantType::Q1_0 => {
                        // Bin Q5_K / Q8_0 / Q1_0 timings into the Q6K
                        // accumulator until dedicated counters are added —
                        // keeps the profile summary total accurate for
                        // mixed-quant models.
                        r.matvec_q6k_us += us;
                        r.matvec_q6k_n += 1;
                    }
                }
            }

            // Qwen 2 / 2.5 attention biases (add-bias timing absorbed into residual_us
            // since add_bias and residual_add share the same shape / cost profile).
            if let Some(bg) = &lbg.q_bias_bg {
                r.residual_us +=
                    timed_dispatch(e, &e.add_bias_pipeline, bg, self.q_bias_dispatch_x, 1, 1);
                r.residual_n += 1;
            }
            if let Some(bg) = &lbg.k_bias_bg {
                r.residual_us +=
                    timed_dispatch(e, &e.add_bias_pipeline, bg, self.kv_bias_dispatch_x, 1, 1);
                r.residual_n += 1;
            }
            if let Some(bg) = &lbg.v_bias_bg {
                r.residual_us +=
                    timed_dispatch(e, &e.add_bias_pipeline, bg, self.kv_bias_dispatch_x, 1, 1);
                r.residual_n += 1;
            }

            // Qwen 3 per-head QK RMSNorm (timing absorbed into rmsnorm_us).
            if let Some(bg) = &lbg.q_norm_bg {
                r.rmsnorm_us +=
                    timed_dispatch(e, &e.qk_norm_pipeline, bg, self.config.num_heads, 1, 1);
                r.rmsnorm_n += 1;
            }
            if let Some(bg) = &lbg.k_norm_bg {
                r.rmsnorm_us +=
                    timed_dispatch(e, &e.qk_norm_pipeline, bg, self.config.num_kv_heads, 1, 1);
                r.rmsnorm_n += 1;
            }

            r.rope_us += timed_dispatch(
                e,
                &e.rope_pipeline,
                &self.rope_q_bg,
                self.rope_q_dispatch_x,
                1,
                1,
            );
            r.rope_n += 1;
            r.rope_us += timed_dispatch(
                e,
                &e.rope_pipeline,
                &self.rope_k_bg,
                self.rope_k_dispatch_x,
                1,
                1,
            );
            r.rope_n += 1;

            r.kv_append_us += timed_dispatch(
                e,
                &e.kv_cache_append_pipeline,
                &lbg.kv_append_bg,
                self.kv_dispatch_x,
                1,
                1,
            );
            r.kv_append_n += 1;

            r.attention_us += timed_dispatch(
                e,
                &e.attention_pipeline,
                &lbg.attention_bg,
                self.config.num_heads,
                1,
                1,
            );
            r.attention_n += 1;

            let (us, qt) = timed_mv(e, &lbg.o_proj);
            match qt {
                GpuQuantType::Q4K => {
                    r.matvec_q4k_us += us;
                    r.matvec_q4k_n += 1;
                }
                GpuQuantType::Q6K => {
                    r.matvec_q6k_us += us;
                    r.matvec_q6k_n += 1;
                }
                GpuQuantType::Q5K | GpuQuantType::Q8_0 | GpuQuantType::Q1_0 => {
                    r.matvec_q6k_us += us;
                    r.matvec_q6k_n += 1;
                }
            }

            r.residual_us += timed_dispatch(
                e,
                &e.residual_add_pipeline,
                &self.residual_attn_bg,
                self.residual_dispatch_x,
                1,
                1,
            );
            r.residual_n += 1;

            // --- FFN sub-block ---
            r.rmsnorm_us += timed_dispatch(e, &e.rmsnorm_pipeline, &lbg.ffn_norm_bg, 1, 1, 1);
            r.rmsnorm_n += 1;

            let swiglu_pipeline = match lbg.swiglu.quant {
                GpuQuantType::Q4K => &e.swiglu_fused_q4k_pipeline,
                GpuQuantType::Q1_0 => &e.swiglu_fused_q1_0_pipeline,
                other => panic!("forward_profiled: unexpected SwiGLU quant {other:?}"),
            };
            r.swiglu_us += timed_dispatch(
                e,
                swiglu_pipeline,
                &lbg.swiglu.bg,
                lbg.swiglu.dispatch_x,
                lbg.swiglu.dispatch_y,
                1,
            );
            r.swiglu_n += 1;

            let (us, qt) = timed_mv(e, &lbg.down_proj);
            match qt {
                GpuQuantType::Q4K => {
                    r.matvec_q4k_us += us;
                    r.matvec_q4k_n += 1;
                }
                GpuQuantType::Q6K => {
                    r.matvec_q6k_us += us;
                    r.matvec_q6k_n += 1;
                }
                GpuQuantType::Q5K | GpuQuantType::Q8_0 | GpuQuantType::Q1_0 => {
                    r.matvec_q6k_us += us;
                    r.matvec_q6k_n += 1;
                }
            }

            r.residual_us += timed_dispatch(
                e,
                &e.residual_add_pipeline,
                &self.residual_ffn_bg,
                self.residual_dispatch_x,
                1,
                1,
            );
            r.residual_n += 1;
        }

        // --- Output head ---
        r.rmsnorm_us += timed_dispatch(e, &e.rmsnorm_pipeline, &self.output_norm_bg, 1, 1, 1);
        r.rmsnorm_n += 1;

        let (us, qt) = timed_mv(e, &self.output_proj_bg);
        match qt {
            GpuQuantType::Q4K => {
                r.matvec_q4k_us += us;
                r.matvec_q4k_n += 1;
            }
            GpuQuantType::Q6K => {
                r.matvec_q6k_us += us;
                r.matvec_q6k_n += 1;
            }
            GpuQuantType::Q5K | GpuQuantType::Q8_0 | GpuQuantType::Q1_0 => {
                r.matvec_q6k_us += us;
                r.matvec_q6k_n += 1;
            }
        }

        self.seq_len = seq_len;
        r
    }

    // --- Batch-4 forward pass (K=4 unrolled scalar accumulators) ---

    /// Upload K token embeddings concatenated into the hidden buffer.
    fn upload_embeddings_batch(&self, token_ids: &[u32]) {
        for (k, &token) in token_ids.iter().enumerate() {
            let start = token as usize * self.config.hidden_dim;
            let end = start + self.config.hidden_dim;
            let buf_offset = (k * self.config.hidden_dim * 4) as u64;
            self.engine.queue.write_buffer(
                &self.hidden.buffer,
                buf_offset,
                bytemuck::cast_slice(&self.embedding[start..end]),
            );
        }
    }

    /// Encode the full forward pass for K=4 tokens using batch4 pipelines.
    /// matvec/swiglu use K=4 unrolled scalar shaders (zero register spill).
    /// rmsnorm/rope/kv_append/attention/residual use batch-aware standard shaders.
    fn encode_forward_batch4(&self, encoder: &mut wgpu::CommandEncoder) {
        let bs: u32 = 4;
        let hidden_dim = self.config.hidden_dim as u32;
        let res_batch_dispatch = (bs * hidden_dim + 255) / 256;

        let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());

        for i in 0..self.config.num_layers {
            let lbg = &self.layer_bgs[i];

            // RMSNorm: 4 workgroups (1 per token)
            cp.set_pipeline(&self.engine.rmsnorm_pipeline);
            cp.set_bind_group(0, &lbg.attn_norm_bg, &[]);
            cp.dispatch_workgroups(bs, 1, 1);

            // Q/K/V matvec — batch4 pipelines (same bind groups!)
            Self::dispatch_mv_batch4(&self.engine, &mut cp, &lbg.q_proj);
            Self::dispatch_mv_batch4(&self.engine, &mut cp, &lbg.k_proj);
            Self::dispatch_mv_batch4(&self.engine, &mut cp, &lbg.v_proj);

            // Qwen 2 / 2.5 attention biases: broadcast bias across the batch dim
            // (wid.y iterates 0..bs, add_bias shader indexes acc_buf[bs_idx * bias_len + elem]).
            if lbg.q_bias_bg.is_some() || lbg.k_bias_bg.is_some() || lbg.v_bias_bg.is_some() {
                cp.set_pipeline(&self.engine.add_bias_pipeline);
                if let Some(bg) = &lbg.q_bias_bg {
                    cp.set_bind_group(0, bg, &[]);
                    cp.dispatch_workgroups(self.q_bias_dispatch_x, bs, 1);
                }
                if let Some(bg) = &lbg.k_bias_bg {
                    cp.set_bind_group(0, bg, &[]);
                    cp.dispatch_workgroups(self.kv_bias_dispatch_x, bs, 1);
                }
                if let Some(bg) = &lbg.v_bias_bg {
                    cp.set_bind_group(0, bg, &[]);
                    cp.dispatch_workgroups(self.kv_bias_dispatch_x, bs, 1);
                }
            }

            // Qwen 3 per-head QK RMSNorm: dispatch head_count * bs workgroups (each covers
            // one head's head_dim elements across all batches). Assumes data layout
            // [bs, heads, head_dim] contiguous.
            if lbg.q_norm_bg.is_some() || lbg.k_norm_bg.is_some() {
                cp.set_pipeline(&self.engine.qk_norm_pipeline);
                if let Some(bg) = &lbg.q_norm_bg {
                    cp.set_bind_group(0, bg, &[]);
                    cp.dispatch_workgroups(self.config.num_heads * bs, 1, 1);
                }
                if let Some(bg) = &lbg.k_norm_bg {
                    cp.set_bind_group(0, bg, &[]);
                    cp.dispatch_workgroups(self.config.num_kv_heads * bs, 1, 1);
                }
            }

            // RoPE: wid.y = batch index
            cp.set_pipeline(&self.engine.rope_pipeline);
            cp.set_bind_group(0, &self.rope_q_bg, &[]);
            cp.dispatch_workgroups(self.rope_q_dispatch_x, bs, 1);
            cp.set_bind_group(0, &self.rope_k_bg, &[]);
            cp.dispatch_workgroups(self.rope_k_dispatch_x, bs, 1);

            // KV cache append: wid.y = batch index
            cp.set_pipeline(&self.engine.kv_cache_append_pipeline);
            cp.set_bind_group(0, &lbg.kv_append_bg, &[]);
            cp.dispatch_workgroups(self.kv_dispatch_x, bs, 1);

            // Attention: wg.y = batch index (causal: token k attends seq_len+k)
            cp.set_pipeline(&self.engine.attention_pipeline);
            cp.set_bind_group(0, &lbg.attention_bg, &[]);
            cp.dispatch_workgroups(self.config.num_heads, bs, 1);

            // O projection — batch4
            Self::dispatch_mv_batch4(&self.engine, &mut cp, &lbg.o_proj);

            // Residual: 4 × hidden_dim elements
            cp.set_pipeline(&self.engine.residual_add_pipeline);
            cp.set_bind_group(0, &self.residual_attn_bg, &[]);
            cp.dispatch_workgroups(res_batch_dispatch, 1, 1);

            // FFN: RMSNorm
            cp.set_pipeline(&self.engine.rmsnorm_pipeline);
            cp.set_bind_group(0, &lbg.ffn_norm_bg, &[]);
            cp.dispatch_workgroups(bs, 1, 1);

            // SwiGLU — batch4 pipeline (same bind group!)
            cp.set_pipeline(&self.engine.swiglu_batch4_pipeline);
            cp.set_bind_group(0, &lbg.swiglu.bg, &[]);
            cp.dispatch_workgroups(lbg.swiglu.dispatch_x, lbg.swiglu.dispatch_y, 1);

            // Down projection — batch4
            Self::dispatch_mv_batch4(&self.engine, &mut cp, &lbg.down_proj);

            // Residual: 4 × hidden_dim elements
            cp.set_pipeline(&self.engine.residual_add_pipeline);
            cp.set_bind_group(0, &self.residual_ffn_bg, &[]);
            cp.dispatch_workgroups(res_batch_dispatch, 1, 1);
        }

        // Output head
        cp.set_pipeline(&self.engine.rmsnorm_pipeline);
        cp.set_bind_group(0, &self.output_norm_bg, &[]);
        cp.dispatch_workgroups(bs, 1, 1);

        Self::dispatch_mv_batch4(&self.engine, &mut cp, &self.output_proj_bg);
    }

    /// Execute batch-4 forward pass: 4 tokens, K=4 unrolled scalar shaders.
    ///
    /// Weight decoded ONCE, 4 scalar FMAs in GPU registers — zero spill.
    /// No uniform batch_size updates needed (compile-time constant in shaders).
    pub fn forward_batch(&mut self, token_ids: &[u32]) {
        assert!(
            token_ids.len() == 4,
            "forward_batch requires exactly 4 tokens"
        );
        let base_pos = self.seq_len;

        // Update per-position uniforms
        self.engine
            .queue
            .write_buffer(&self.rope_q_params_buf, 0, bytemuck::bytes_of(&base_pos));
        self.engine
            .queue
            .write_buffer(&self.rope_k_params_buf, 0, bytemuck::bytes_of(&base_pos));
        self.engine.queue.write_buffer(
            &self.kv_append_params_buf,
            0,
            bytemuck::bytes_of(&base_pos),
        );
        let attn_seq_len = base_pos + 1;
        self.engine
            .queue
            .write_buffer(&self.attn_params_buf, 0, bytemuck::bytes_of(&attn_seq_len));

        self.upload_embeddings_batch(token_ids);

        let mut encoder = self
            .engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_forward_batch4(&mut encoder);
        self.engine.queue.submit(Some(encoder.finish()));

        self.seq_len = base_pos + 4;
    }

    /// Execute batch-4 forward pass with logits readback for all 4 tokens.
    pub fn forward_batch_and_read(&mut self, token_ids: &[u32]) -> Vec<f32> {
        assert!(
            token_ids.len() == 4,
            "forward_batch_and_read requires exactly 4 tokens"
        );
        let base_pos = self.seq_len;

        self.engine
            .queue
            .write_buffer(&self.rope_q_params_buf, 0, bytemuck::bytes_of(&base_pos));
        self.engine
            .queue
            .write_buffer(&self.rope_k_params_buf, 0, bytemuck::bytes_of(&base_pos));
        self.engine.queue.write_buffer(
            &self.kv_append_params_buf,
            0,
            bytemuck::bytes_of(&base_pos),
        );
        let attn_seq_len = base_pos + 1;
        self.engine
            .queue
            .write_buffer(&self.attn_params_buf, 0, bytemuck::bytes_of(&attn_seq_len));
        self.upload_embeddings_batch(token_ids);

        let mut encoder = self
            .engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_forward_batch4(&mut encoder);

        let total_logits = 4 * self.vocab_size;
        let size = (total_logits * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.logits.buffer, 0, &self.staging, 0, size);

        self.engine.queue.submit(Some(encoder.finish()));
        self.seq_len = base_pos + 4;
        self.engine.map_staging(&self.staging, total_logits)
    }

    /// GPU Timestamp Query profiled forward pass.
    /// Each dispatch is a separate compute pass with `timestamp_writes`.
    /// Single submit — no CPU sync overhead. Returns true GPU execution times.
    #[allow(unused_assignments)]
    pub fn forward_gpu_profiled(&mut self, token_id: u32) -> ProfileResult {
        let pos = self.seq_len;
        let seq_len = pos + 1;

        self.update_uniforms(pos, seq_len);
        self.upload_embedding(token_id);

        // 14 dispatches per layer × num_layers + 2 output head + optional
        // Qwen 2 / 2.5 QKV bias adds + optional Qwen 3 per-head QK norms.
        let ops_per_layer: u32 = 14;
        let n_layers = self.config.num_layers as u32;
        let bias_dispatches: u32 = self
            .layer_bgs
            .iter()
            .map(|lbg| {
                u32::from(lbg.q_bias_bg.is_some())
                    + u32::from(lbg.k_bias_bg.is_some())
                    + u32::from(lbg.v_bias_bg.is_some())
            })
            .sum();
        let qk_norm_dispatches: u32 = self
            .layer_bgs
            .iter()
            .map(|lbg| u32::from(lbg.q_norm_bg.is_some()) + u32::from(lbg.k_norm_bg.is_some()))
            .sum();
        let total_dispatches = ops_per_layer * n_layers + bias_dispatches + qk_norm_dispatches + 2;
        // Each dispatch uses 2 query slots (beginning + end of compute pass)
        let num_queries = total_dispatches * 2;

        let query_set = self
            .engine
            .device
            .create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("timestamp_queries"),
                ty: wgpu::QueryType::Timestamp,
                count: num_queries,
            });

        let resolve_buf = self.engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp_resolve"),
            size: (num_queries as u64) * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = self.engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp_readback"),
            size: (num_queries as u64) * 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Operation category tags (parallel array to dispatches)
        // 0=rmsnorm, 1=matvec_q4k, 2=matvec_q6k, 3=swiglu, 4=rope, 5=kv_append, 6=attention, 7=residual
        let mut op_tags: Vec<u8> = Vec::with_capacity(total_dispatches as usize);

        let mut encoder = self
            .engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let mut dispatch_idx: u32 = 0;

        // Each dispatch gets its own compute pass with timestamp_writes (begin + end)
        macro_rules! ts_dispatch {
            ($pipeline:expr, $bg:expr, $dx:expr, $dy:expr, $dz:expr, $tag:expr) => {{
                let begin_idx = dispatch_idx * 2;
                let end_idx = begin_idx + 1;
                {
                    let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                            query_set: &query_set,
                            beginning_of_pass_write_index: Some(begin_idx),
                            end_of_pass_write_index: Some(end_idx),
                        }),
                    });
                    cp.set_pipeline($pipeline);
                    cp.set_bind_group(0, $bg, &[]);
                    cp.dispatch_workgroups($dx, $dy, $dz);
                }
                op_tags.push($tag);
                dispatch_idx += 1;
            }};
        }

        macro_rules! ts_dispatch_mv {
            ($mv:expr) => {{
                // Q1_0 uses the row4 kernel (Step 6, PR #74) to stay
                // consistent with the production `dispatch_mv` path.
                let pipeline = match $mv.quant {
                    GpuQuantType::Q4K => &self.engine.matvec_q4k_pipeline,
                    GpuQuantType::Q5K => &self.engine.matvec_q5k_pipeline,
                    GpuQuantType::Q6K => &self.engine.matvec_q6k_pipeline,
                    GpuQuantType::Q8_0 => &self.engine.matvec_q8_0_pipeline,
                    GpuQuantType::Q1_0 => &self.engine.matvec_q1_0_row4_pipeline,
                };
                let tag = match $mv.quant {
                    GpuQuantType::Q4K => 1u8,
                    GpuQuantType::Q5K => 4u8,
                    GpuQuantType::Q6K => 2u8,
                    GpuQuantType::Q8_0 => 5u8,
                    GpuQuantType::Q1_0 => 3u8,
                };
                ts_dispatch!(pipeline, &$mv.bg, $mv.dispatch_x, $mv.dispatch_y, 1, tag);
            }};
        }

        for i in 0..self.config.num_layers {
            let lbg = &self.layer_bgs[i];

            // Attention sub-block
            ts_dispatch!(
                &self.engine.rmsnorm_pipeline,
                &lbg.attn_norm_bg,
                1,
                1,
                1,
                0u8
            );
            ts_dispatch_mv!(lbg.q_proj);
            ts_dispatch_mv!(lbg.k_proj);
            ts_dispatch_mv!(lbg.v_proj);
            // Qwen 2 / 2.5 attention biases (add-bias tagged as `7u8` residual).
            if let Some(bg) = &lbg.q_bias_bg {
                ts_dispatch!(
                    &self.engine.add_bias_pipeline,
                    bg,
                    self.q_bias_dispatch_x,
                    1,
                    1,
                    7u8
                );
            }
            if let Some(bg) = &lbg.k_bias_bg {
                ts_dispatch!(
                    &self.engine.add_bias_pipeline,
                    bg,
                    self.kv_bias_dispatch_x,
                    1,
                    1,
                    7u8
                );
            }
            if let Some(bg) = &lbg.v_bias_bg {
                ts_dispatch!(
                    &self.engine.add_bias_pipeline,
                    bg,
                    self.kv_bias_dispatch_x,
                    1,
                    1,
                    7u8
                );
            }
            // Qwen 3 per-head QK RMSNorm (tagged as `0u8` rmsnorm).
            if let Some(bg) = &lbg.q_norm_bg {
                ts_dispatch!(
                    &self.engine.qk_norm_pipeline,
                    bg,
                    self.config.num_heads,
                    1,
                    1,
                    0u8
                );
            }
            if let Some(bg) = &lbg.k_norm_bg {
                ts_dispatch!(
                    &self.engine.qk_norm_pipeline,
                    bg,
                    self.config.num_kv_heads,
                    1,
                    1,
                    0u8
                );
            }
            ts_dispatch!(
                &self.engine.rope_pipeline,
                &self.rope_q_bg,
                self.rope_q_dispatch_x,
                1,
                1,
                4u8
            );
            ts_dispatch!(
                &self.engine.rope_pipeline,
                &self.rope_k_bg,
                self.rope_k_dispatch_x,
                1,
                1,
                4u8
            );
            ts_dispatch!(
                &self.engine.kv_cache_append_pipeline,
                &lbg.kv_append_bg,
                self.kv_dispatch_x,
                1,
                1,
                5u8
            );
            ts_dispatch!(
                &self.engine.attention_pipeline,
                &lbg.attention_bg,
                self.config.num_heads,
                1,
                1,
                6u8
            );
            ts_dispatch_mv!(lbg.o_proj);
            ts_dispatch!(
                &self.engine.residual_add_pipeline,
                &self.residual_attn_bg,
                self.residual_dispatch_x,
                1,
                1,
                7u8
            );

            // FFN sub-block
            ts_dispatch!(
                &self.engine.rmsnorm_pipeline,
                &lbg.ffn_norm_bg,
                1,
                1,
                1,
                0u8
            );
            let ts_swiglu_pipeline = match lbg.swiglu.quant {
                GpuQuantType::Q4K => &self.engine.swiglu_fused_q4k_pipeline,
                GpuQuantType::Q1_0 => &self.engine.swiglu_fused_q1_0_pipeline,
                other => panic!("forward_timestamped: unexpected SwiGLU quant {other:?}"),
            };
            ts_dispatch!(
                ts_swiglu_pipeline,
                &lbg.swiglu.bg,
                lbg.swiglu.dispatch_x,
                lbg.swiglu.dispatch_y,
                1,
                3u8
            );
            ts_dispatch_mv!(lbg.down_proj);
            ts_dispatch!(
                &self.engine.residual_add_pipeline,
                &self.residual_ffn_bg,
                self.residual_dispatch_x,
                1,
                1,
                7u8
            );
        }

        // Output head
        ts_dispatch!(
            &self.engine.rmsnorm_pipeline,
            &self.output_norm_bg,
            1,
            1,
            1,
            0u8
        );
        ts_dispatch_mv!(self.output_proj_bg);

        // Resolve timestamps to buffer
        encoder.resolve_query_set(&query_set, 0..num_queries, &resolve_buf, 0);
        encoder.copy_buffer_to_buffer(&resolve_buf, 0, &readback_buf, 0, (num_queries as u64) * 8);

        self.engine.queue.submit(Some(encoder.finish()));
        self.engine.device.poll(wgpu::Maintain::Wait);

        // Readback timestamps
        let slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.engine.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("channel closed").expect("map failed");
        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);

        let period_us = self.engine.timestamp_period as f64 / 1000.0; // ns→µs

        let mut r = ProfileResult {
            rmsnorm_us: 0.0,
            matvec_q4k_us: 0.0,
            matvec_q6k_us: 0.0,
            swiglu_us: 0.0,
            rope_us: 0.0,
            kv_append_us: 0.0,
            attention_us: 0.0,
            residual_us: 0.0,
            rmsnorm_n: 0,
            matvec_q4k_n: 0,
            matvec_q6k_n: 0,
            swiglu_n: 0,
            rope_n: 0,
            kv_append_n: 0,
            attention_n: 0,
            residual_n: 0,
        };

        for i in 0..total_dispatches as usize {
            let begin_ts = timestamps[i * 2];
            let end_ts = timestamps[i * 2 + 1];
            let delta_us = (end_ts.wrapping_sub(begin_ts)) as f64 * period_us;
            let tag = op_tags[i];
            match tag {
                0 => {
                    r.rmsnorm_us += delta_us;
                    r.rmsnorm_n += 1;
                }
                1 => {
                    r.matvec_q4k_us += delta_us;
                    r.matvec_q4k_n += 1;
                }
                2 => {
                    r.matvec_q6k_us += delta_us;
                    r.matvec_q6k_n += 1;
                }
                3 => {
                    r.swiglu_us += delta_us;
                    r.swiglu_n += 1;
                }
                4 => {
                    r.rope_us += delta_us;
                    r.rope_n += 1;
                }
                5 => {
                    r.kv_append_us += delta_us;
                    r.kv_append_n += 1;
                }
                6 => {
                    r.attention_us += delta_us;
                    r.attention_n += 1;
                }
                7 => {
                    r.residual_us += delta_us;
                    r.residual_n += 1;
                }
                _ => {}
            }
        }

        drop(data);
        readback_buf.unmap();

        self.seq_len = seq_len;
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify `silu_gate_apply_cpu` produces the mathematically correct
    /// values for the Bonsai gated attention output modulation.
    ///
    /// This is the CPU reference for the WGSL `silu_gate_apply` shader —
    /// they must produce bit-exact identical outputs (modulo FP summation
    /// order which is irrelevant here since the operation is fully
    /// element-wise). Any change to the shader must have a corresponding
    /// change to `silu_gate_apply_cpu` and this test must still pass.
    #[test]
    fn silu_gate_apply_cpu_matches_math() {
        // Key values of silu(x) = x / (1 + exp(-x)) = x * sigmoid(x):
        //   silu(0)         = 0
        //   silu(1)         ≈ 0.7310586
        //   silu(-1)        ≈ -0.2689414
        //   silu(large +)   ≈ x (asymptotic)
        //   silu(large -)   ≈ 0 (asymptotic)
        let mut attn_out = vec![1.0f32, 1.0, 1.0, 1.0, 1.0];
        let gate = vec![0.0f32, 1.0, -1.0, 20.0, -20.0];

        silu_gate_apply_cpu(&mut attn_out, &gate);

        // attn_out[i] *= silu(gate[i]), initial attn_out = 1 → attn_out = silu(gate)
        assert!(
            (attn_out[0] - 0.0).abs() < 1e-6,
            "silu(0) = 0 expected, got {}",
            attn_out[0]
        );
        assert!(
            (attn_out[1] - 0.7310586).abs() < 1e-5,
            "silu(1) ≈ 0.7310586 expected, got {}",
            attn_out[1]
        );
        assert!(
            (attn_out[2] - (-0.2689414)).abs() < 1e-5,
            "silu(-1) ≈ -0.2689414 expected, got {}",
            attn_out[2]
        );
        // silu(20) ≈ 20 (sigmoid(20) ≈ 1.0)
        assert!(
            (attn_out[3] - 20.0).abs() < 1e-4,
            "silu(20) ≈ 20 expected, got {}",
            attn_out[3]
        );
        // silu(-20) ≈ 0 (sigmoid(-20) ≈ 0)
        assert!(
            attn_out[4].abs() < 1e-4,
            "silu(-20) ≈ 0 expected, got {}",
            attn_out[4]
        );
    }

    /// Verify `silu_gate_apply_cpu` respects multiplicative structure:
    /// `attn_out *= silu(gate)` with non-unit `attn_out`.
    #[test]
    fn silu_gate_apply_cpu_multiplicative() {
        let mut attn_out = vec![2.0f32, -3.0, 0.5, 0.0];
        let gate = vec![1.0f32, 1.0, -1.0, 5.0];

        // Expected: attn_out[i] = attn_out[i] * silu(gate[i])
        // silu(1)  ≈ 0.7310586
        // silu(-1) ≈ -0.2689414
        // silu(5)  ≈ 4.9665358
        let expected = [
            2.0 * 0.7310586,  // ≈  1.4621172
            -3.0 * 0.7310586, // ≈ -2.1931758
            0.5 * -0.2689414, // ≈ -0.1344707
            0.0 * 4.9665358,  // =  0.0
        ];

        silu_gate_apply_cpu(&mut attn_out, &gate);

        for (i, (&got, &want)) in attn_out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "silu_gate_apply_cpu[{i}]: got {got}, want {want}"
            );
        }
    }

    /// Verify `silu_gate_apply_cpu` panics on mismatched lengths (contract test).
    #[test]
    #[should_panic(expected = "attn_out and gate must have same length")]
    fn silu_gate_apply_cpu_panics_on_length_mismatch() {
        let mut attn_out = vec![1.0f32; 4];
        let gate = vec![1.0f32; 5];
        silu_gate_apply_cpu(&mut attn_out, &gate);
    }

    /// Verify `silu_inplace_cpu` produces standard silu values for key anchor points.
    #[test]
    fn silu_inplace_cpu_matches_math() {
        // silu(x) = x * sigmoid(x)
        //   silu(0)         = 0
        //   silu(1)         ≈ 0.7310586
        //   silu(-1)        ≈ -0.2689414
        //   silu(large +)   ≈ x (asymptotic)
        //   silu(large -)   ≈ 0 (asymptotic)
        let mut buf = vec![0.0f32, 1.0, -1.0, 20.0, -20.0];
        silu_inplace_cpu(&mut buf);

        assert!(buf[0].abs() < 1e-6, "silu(0) = 0, got {}", buf[0]);
        assert!(
            (buf[1] - 0.7310586).abs() < 1e-5,
            "silu(1) ≈ 0.7310586, got {}",
            buf[1]
        );
        assert!(
            (buf[2] - (-0.2689414)).abs() < 1e-5,
            "silu(-1) ≈ -0.2689414, got {}",
            buf[2]
        );
        assert!(
            (buf[3] - 20.0).abs() < 1e-4,
            "silu(20) ≈ 20, got {}",
            buf[3]
        );
        assert!(buf[4].abs() < 1e-4, "silu(-20) ≈ 0, got {}", buf[4]);
    }

    /// Verify `silu_inplace_cpu` matches `silu_gate_apply_cpu` when the
    /// gate operand of the latter is a unit vector (attn_out=silu(gate)).
    #[test]
    fn silu_inplace_cpu_matches_silu_gate_apply_with_unit_gate() {
        let src = vec![0.5f32, -1.2, 3.7, 0.0, 100.0, -100.0];

        // silu_inplace on src
        let mut inplace = src.clone();
        silu_inplace_cpu(&mut inplace);

        // silu_gate_apply with attn_out = [1, 1, ..., 1], gate = src
        // → attn_out[i] = 1 * silu(src[i]) = silu(src[i])
        let mut gate_apply = vec![1.0f32; src.len()];
        silu_gate_apply_cpu(&mut gate_apply, &src);

        for (i, (&a, &b)) in inplace.iter().zip(gate_apply.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "silu_inplace[{i}] = {a}, silu_gate_apply[{i}] = {b} (should match)",
            );
        }
    }

    /// Verify `beta_sigmoid_cpu` produces standard sigmoid values.
    #[test]
    fn beta_sigmoid_cpu_matches_math() {
        // Key sigmoid values:
        //   sigmoid(0)      = 0.5
        //   sigmoid(1)      ≈ 0.7310586
        //   sigmoid(-1)     ≈ 0.2689414
        //   sigmoid(large+) ≈ 1.0
        //   sigmoid(large-) ≈ 0.0
        let mut beta = vec![0.0f32, 1.0, -1.0, 20.0, -20.0];
        beta_sigmoid_cpu(&mut beta);

        assert!(
            (beta[0] - 0.5).abs() < 1e-6,
            "sigmoid(0) = 0.5, got {}",
            beta[0]
        );
        assert!(
            (beta[1] - 0.7310586).abs() < 1e-5,
            "sigmoid(1) ≈ 0.7310586, got {}",
            beta[1]
        );
        assert!(
            (beta[2] - 0.2689414).abs() < 1e-5,
            "sigmoid(-1) ≈ 0.2689414, got {}",
            beta[2]
        );
        assert!(
            (beta[3] - 1.0).abs() < 1e-4,
            "sigmoid(20) ≈ 1.0, got {}",
            beta[3]
        );
        assert!(beta[4].abs() < 1e-4, "sigmoid(-20) ≈ 0.0, got {}", beta[4]);
    }

    /// Verify `beta_sigmoid_cpu` output is always in (0, 1) for any input.
    #[test]
    fn beta_sigmoid_cpu_range_invariant() {
        let mut beta = vec![
            -1000.0f32, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0,
        ];
        beta_sigmoid_cpu(&mut beta);
        for (i, &b) in beta.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&b),
                "beta_sigmoid_cpu[{i}] out of (0, 1] range: {b}"
            );
        }
    }

    /// Verify `ssm_discretisation_cpu` produces decay values in (0, 1] when
    /// `ssm_a` is negative (Mamba convention: ssm_a stored ≈ -exp(A_log) < 0).
    #[test]
    fn ssm_discretisation_cpu_produces_valid_decay() {
        // 3 V heads with typical Bonsai / Mamba values:
        //   ssm_a in [-2, -0.1] (stored as -exp(A_log), A_log in [-2, 0.7])
        //   ssm_dt_bias in [-1, 1]
        //   alpha (raw projection) in [-3, 3]
        let mut alpha = vec![0.0f32, 1.5, -2.0];
        let ssm_dt_bias = vec![0.5f32, -0.2, 0.1];
        let ssm_a = vec![-1.0f32, -0.5, -1.5];

        ssm_discretisation_cpu(&mut alpha, &ssm_dt_bias, &ssm_a);

        // Verify each decay ∈ (0, 1]
        for (h, &d) in alpha.iter().enumerate() {
            assert!(
                d > 0.0 && d <= 1.0,
                "ssm_discretisation_cpu[{h}] decay = {d} not in (0, 1]"
            );
        }
    }

    /// Verify `ssm_discretisation_cpu` hand-computed reference for h=0.
    ///
    /// alpha=0.0, ssm_dt_bias=0.0, ssm_a=-1.0:
    ///   alpha_biased  = 0.0
    ///   alpha_softplus = ln(1 + exp(0)) = ln(2) ≈ 0.6931472
    ///   gate          = 0.6931472 * -1.0 = -0.6931472
    ///   decay         = exp(-0.6931472) = 0.5
    #[test]
    fn ssm_discretisation_cpu_hand_computed_anchor() {
        let mut alpha = vec![0.0f32];
        let ssm_dt_bias = vec![0.0f32];
        let ssm_a = vec![-1.0f32];

        ssm_discretisation_cpu(&mut alpha, &ssm_dt_bias, &ssm_a);

        assert!(
            (alpha[0] - 0.5).abs() < 1e-6,
            "ssm_discretisation_cpu(0, 0, -1) should give decay ≈ 0.5, got {}",
            alpha[0]
        );
    }

    /// Verify softplus numerical stability for large positive input.
    ///
    /// alpha=25.0 (biased): softplus should saturate to ~25 without exp overflow.
    /// ssm_a=-0.1: gate = 25 * -0.1 = -2.5, decay = exp(-2.5) ≈ 0.0820850
    #[test]
    fn ssm_discretisation_cpu_softplus_stable_large_positive() {
        let mut alpha = vec![25.0f32];
        let ssm_dt_bias = vec![0.0f32];
        let ssm_a = vec![-0.1f32];

        ssm_discretisation_cpu(&mut alpha, &ssm_dt_bias, &ssm_a);

        assert!(
            alpha[0].is_finite(),
            "softplus overflow: got non-finite decay {}",
            alpha[0]
        );
        assert!(
            (alpha[0] - 0.0820850).abs() < 1e-4,
            "expected decay ≈ 0.0820850 for alpha=25, ssm_a=-0.1, got {}",
            alpha[0]
        );
    }

    /// Verify `ssm_discretisation_cpu` panics on mismatched lengths.
    #[test]
    #[should_panic(expected = "alpha and ssm_a must have same length")]
    fn ssm_discretisation_cpu_panics_on_length_mismatch() {
        let mut alpha = vec![1.0f32; 4];
        let ssm_dt_bias = vec![1.0f32; 4];
        let ssm_a = vec![1.0f32; 5];
        ssm_discretisation_cpu(&mut alpha, &ssm_dt_bias, &ssm_a);
    }

    /// Verify `ssm_norm_per_head_cpu` computes per-head RMSNorm independently.
    ///
    /// 2 V heads × 4 v_dim = 8 elements. Head 0 has all 1.0, Head 1 has all 2.0.
    /// Weight is [1, 1, 1, 1] (identity), eps = 1e-6.
    ///
    /// Expected:
    ///   Head 0: mean(1^2) = 1.0 → scale = 1/sqrt(1.0 + eps) ≈ 1.0
    ///           → output ≈ [1, 1, 1, 1]
    ///   Head 1: mean(2^2) = 4.0 → scale = 1/sqrt(4.0 + eps) ≈ 0.5
    ///           → output ≈ [1, 1, 1, 1]
    ///
    /// The normalization brings both heads to the same magnitude — which is
    /// exactly the point of RMSNorm (magnitude-invariant unit output).
    #[test]
    fn ssm_norm_per_head_cpu_normalizes_each_head_independently() {
        let mut input = vec![1.0f32, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let weight = vec![1.0f32; 4];
        let v_dim = 4;
        let eps = 1e-6;

        ssm_norm_per_head_cpu(&mut input, &weight, v_dim, eps);

        // Both heads normalized to ~[1, 1, 1, 1]
        for (i, &v) in input.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-3,
                "input[{i}] = {v}, expected ≈ 1.0 after per-head RMSNorm",
            );
        }
    }

    /// Verify `ssm_norm_per_head_cpu` respects weight broadcast across V heads.
    ///
    /// 2 V heads × 3 v_dim, uniform input, non-uniform weight [1, 2, 3]:
    /// - Both heads normalize to unit magnitude vector [s, s, s]
    /// - Then multiplied by weight → [s, 2s, 3s]
    /// - Same weight applied to both heads (broadcast).
    #[test]
    fn ssm_norm_per_head_cpu_broadcasts_weight() {
        let mut input = vec![3.0f32, 3.0, 3.0, 5.0, 5.0, 5.0];
        let weight = vec![1.0f32, 2.0, 3.0];
        let v_dim = 3;
        let eps = 1e-6;

        ssm_norm_per_head_cpu(&mut input, &weight, v_dim, eps);

        // Head 0: mean(9) = 9, scale = 1/3, unit vector = [1, 1, 1]
        //         after weight: [1, 2, 3]
        // Head 1: mean(25) = 25, scale = 1/5, unit vector = [1, 1, 1]
        //         after weight: [1, 2, 3]
        for h in 0..2 {
            let start = h * v_dim;
            assert!((input[start] - 1.0).abs() < 1e-3, "head {h}, elem 0");
            assert!((input[start + 1] - 2.0).abs() < 1e-3, "head {h}, elem 1");
            assert!((input[start + 2] - 3.0).abs() < 1e-3, "head {h}, elem 2");
        }
    }

    /// Verify `ssm_norm_per_head_cpu` result magnitude is close to sqrt(v_dim)
    /// (RMSNorm invariant: normalizes to unit RMS = 1.0, so magnitude = sqrt(v_dim)).
    #[test]
    fn ssm_norm_per_head_cpu_rms_invariant() {
        let v_dim = 8;
        let num_heads = 3;
        // Random-ish input
        let mut input: Vec<f32> = (0..num_heads * v_dim)
            .map(|i| ((i as f32 * 0.7).sin() + 1.5) * ((i / v_dim + 1) as f32))
            .collect();
        let weight = vec![1.0f32; v_dim]; // identity weight

        ssm_norm_per_head_cpu(&mut input, &weight, v_dim, 1e-6);

        // After normalization with identity weight, each head should have
        // RMS ≈ 1.0 (sum-of-squares / v_dim ≈ 1.0)
        for h in 0..num_heads {
            let start = h * v_dim;
            let ss: f32 = input[start..start + v_dim].iter().map(|x| x * x).sum();
            let rms = (ss / v_dim as f32).sqrt();
            assert!(
                (rms - 1.0).abs() < 1e-3,
                "head {h} RMS = {rms}, expected ≈ 1.0",
            );
        }
    }

    /// Verify `ssm_norm_per_head_cpu` matches `llama3::apply_qk_norm` semantics
    /// (which is the CPU forward path reference implementation).
    ///
    /// The only difference is CPU forward uses f64 accumulation while our
    /// GPU-parity ref uses f32 — should agree within FP summation noise for
    /// v_dim=128 (Bonsai standard).
    #[test]
    fn ssm_norm_per_head_cpu_matches_llama3_apply_qk_norm() {
        // Simulate Bonsai-like tensor: 3 V heads × 128 v_dim
        let v_dim = 128;
        let num_heads = 3;
        let eps = 1e-6f32;
        let input: Vec<f32> = (0..num_heads * v_dim)
            .map(|i| ((i as f32 * 0.031).cos() + 0.3) * (h_scale(i / v_dim)))
            .collect();
        let weight: Vec<f32> = (0..v_dim)
            .map(|i| 0.5 + (i as f32 * 0.017).sin() * 0.1)
            .collect();

        // Compute via our f32 GPU-parity ref
        let mut input_gpu_ref = input.clone();
        ssm_norm_per_head_cpu(&mut input_gpu_ref, &weight, v_dim, eps);

        // Compute via llama3::apply_qk_norm-equivalent f64 accumulation
        let mut input_llama3_ref = input.clone();
        for h in 0..num_heads {
            let start = h * v_dim;
            let slice = &mut input_llama3_ref[start..start + v_dim];
            let mut ss = 0.0f64;
            for &v in slice.iter() {
                ss += (v as f64) * (v as f64);
            }
            let mean = (ss / v_dim as f64) as f32;
            let scale = 1.0f32 / (mean + eps).sqrt();
            for (i, w) in weight.iter().enumerate() {
                slice[i] = slice[i] * scale * w;
            }
        }

        // Both should agree within FP summation noise for v_dim=128.
        for (i, (&g, &l)) in input_gpu_ref
            .iter()
            .zip(input_llama3_ref.iter())
            .enumerate()
        {
            assert!(
                (g - l).abs() < 1e-5,
                "index {i}: gpu_ref={g}, llama3_ref={l}, diff={}",
                (g - l).abs(),
            );
        }
    }

    fn h_scale(h: usize) -> f32 {
        match h {
            0 => 1.0,
            1 => 0.3,
            _ => 2.5,
        }
    }

    /// Verify `ssm_norm_per_head_cpu` panics on invalid input length.
    #[test]
    #[should_panic(expected = "input.len()")]
    fn ssm_norm_per_head_cpu_panics_on_input_not_multiple_of_v_dim() {
        let mut input = vec![1.0f32; 7]; // 7 not divisible by v_dim=4
        let weight = vec![1.0f32; 4];
        ssm_norm_per_head_cpu(&mut input, &weight, 4, 1e-6);
    }

    /// Verify `ssm_norm_per_head_cpu` panics on invalid weight length.
    #[test]
    #[should_panic(expected = "weight.len()")]
    fn ssm_norm_per_head_cpu_panics_on_weight_length_mismatch() {
        let mut input = vec![1.0f32; 8];
        let weight = vec![1.0f32; 3]; // must be v_dim=4
        ssm_norm_per_head_cpu(&mut input, &weight, 4, 1e-6);
    }

    /// GPU pipeline creation smoke test — instantiates `GpuEngine::new()` to
    /// force all compute pipelines to compile on the runtime backend (Metal
    /// on macOS, Vulkan on Linux with lavapipe, DirectX on Windows).
    ///
    /// This is the sentinel that catches shader-side regressions that only
    /// surface at Metal / Vulkan pipeline-creation time and are missed by
    /// naga transpile (which validates WGSL syntax but does not exercise
    /// the target backend's shader compiler).
    ///
    /// # Historical context
    ///
    /// PR #73 (row8×batch4 shader) introduced 32 separate `var<workgroup>`
    /// arrays which exceeded Metal's per-shader threadgroup resource slot
    /// limit. WGSL syntax was valid, naga transpiled cleanly, ubuntu Vulkan
    /// tests passed — but any Mac Metal user hit `GpuEngine::new()` panic:
    ///
    /// ```text
    /// Internal error: Metal: program_source:89:23: error:
    /// no 'threadgroup' resource location available for 'sg73_'
    /// ```
    ///
    /// PR #76 fixed the shader by consolidating the 32 arrays into a single
    /// `array<f32, 512>`. This smoke test exists to prevent regression: if
    /// any shader change re-triggers a Metal / Vulkan / DirectX validation
    /// error at pipeline creation time, this test fails at CI green-gate
    /// time rather than in production.
    ///
    /// # Behavior on GPU-less CI runners
    ///
    /// If no GPU adapter is available (headless Linux CI without lavapipe),
    /// `GpuEngine::new()` panics with "no GPU adapter found" during
    /// `.expect()`. The test uses `std::panic::catch_unwind` to distinguish:
    /// - "no GPU adapter found" → SKIP (test passes with note)
    /// - Any other panic (Metal validation, etc.) → FAIL
    ///
    /// This lets the test compile-check on any CI runner and actually
    /// exercise pipeline creation whenever a GPU (real or software) exists.
    #[test]
    fn smoke_test_gpu_pipeline_creation_all_pipelines_valid() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _engine = GpuEngine::new();
            // If we reach this line, all pipelines compiled successfully on
            // the runtime backend. GpuEngine::new() creates all pipelines
            // eagerly, so any Metal / Vulkan / DirectX validation error would
            // have panicked above.
        }));

        match result {
            Ok(()) => {
                eprintln!("[smoke_test] GpuEngine::new() succeeded — all pipelines compiled");
            }
            Err(err) => {
                let msg = if let Some(s) = err.downcast_ref::<String>() {
                    s.as_str()
                } else if let Some(s) = err.downcast_ref::<&str>() {
                    *s
                } else {
                    "(non-string panic payload)"
                };

                // Distinguish "no GPU available" (acceptable on headless CI)
                // from actual shader / pipeline compilation failure.
                if msg.contains("no GPU adapter found") {
                    eprintln!(
                        "[smoke_test] SKIP: no GPU adapter available on this runner \
                         (this is expected on headless Linux CI without lavapipe). \
                         Pipeline creation validation deferred to macOS / Linux+lavapipe runners."
                    );
                    // Test passes — the intent is to catch pipeline compilation
                    // errors, not adapter availability.
                } else {
                    // Any other panic message is a real shader / pipeline
                    // compilation error — fail the test with the diagnostic.
                    panic!(
                        "GpuEngine::new() failed with an unexpected panic \
                         (likely shader / pipeline compilation error on this backend): {msg}"
                    );
                }
            }
        }
    }

    /// Try to instantiate `GpuEngine::new()`. Returns `None` if no adapter is
    /// available on the host (typical for headless CI without lavapipe) so
    /// callers can `return` to skip the test cleanly.
    fn try_gpu_engine() -> Option<GpuEngine> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(GpuEngine::new));
        match result {
            Ok(e) => Some(e),
            Err(err) => {
                let msg = if let Some(s) = err.downcast_ref::<String>() {
                    s.as_str()
                } else if let Some(s) = err.downcast_ref::<&str>() {
                    *s
                } else {
                    "(non-string panic payload)"
                };
                if msg.contains("no GPU adapter found") {
                    eprintln!("[test] SKIP: no GPU adapter available");
                    None
                } else {
                    panic!("GpuEngine::new() failed unexpectedly: {msg}");
                }
            }
        }
    }

    /// Encode f16 (IEEE 754 half) bits from an f32. Round-to-nearest without
    /// bit-perfect edge case handling — sufficient for test scale/min
    /// constants which are well within the normalised f16 range.
    fn f32_to_f16_bits(x: f32) -> u16 {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mant = bits & 0x7fffff;
        if exp == 0 && mant == 0 {
            return sign;
        }
        if exp == 0xff {
            // inf or nan
            return sign | 0x7c00 | (if mant != 0 { 0x200 } else { 0 });
        }
        let new_exp = exp - 127 + 15;
        if new_exp <= 0 {
            // subnormal / underflow — round to zero for simplicity.
            return sign;
        }
        if new_exp >= 0x1f {
            // overflow → inf
            return sign | 0x7c00;
        }
        sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
    }

    /// Build a minimal Q5_K test block (176 bytes) with `d = 1.0`,
    /// `dmin = 0.0`, all 8 sub-block scales = 1 (so per-element output =
    /// f32(nibble)), and all mins = 0. `qh` bits and `qs` nibbles are
    /// supplied by the caller. This is enough coverage to test the shader's
    /// dequant + accumulation math against `dequantize_q5_k` without
    /// implementing a full Q5_K quantiser.
    fn build_q5k_block(qh: &[u8; 32], qs: &[u8; 128]) -> Vec<u8> {
        let mut block = Vec::with_capacity(176);
        block.extend_from_slice(&f32_to_f16_bits(1.0).to_le_bytes()); // d
        block.extend_from_slice(&f32_to_f16_bits(0.0).to_le_bytes()); // dmin
                                                                      // scales / mins packed for 8 sub-blocks. `get_scale_min_k4` for `j<4`
                                                                      // reads sc = s0[j] & 63, mn = s1[j] & 63. For `j>=4` it uses the
                                                                      // combined layout across s0, s1, s2. Set:
                                                                      //   s0[0..4] = 1, s1[0..4] = 0    (sc[0..4] = 1, mn[0..4] = 0)
                                                                      //   s0[4..8] = 0, s1[4..8] = 0, s2 encodes sc[4..8]=1 mn[4..8]=0.
                                                                      // For j>=4 (from the shader):
                                                                      //   sc = (s2[j-4] & 0xF) | ((s0[j-4] >> 6) << 4)
                                                                      //   mn = (s2[j-4] >> 4)  | ((s1[j-4] >> 6) << 4)
                                                                      // Setting s0/s1/s2 zero except:
                                                                      //   s0[0..4] = 1  → sc[0..4] = 1
                                                                      //   s2[0..4] = 1  → sc[4..8] = 1
                                                                      // and mn stays 0.
        let mut scales = [0u8; 12];
        for i in 0..4 {
            scales[i] = 1; // s0[i]
        }
        for i in 0..4 {
            scales[8 + i] = 1; // s2[i]
        }
        block.extend_from_slice(&scales);
        block.extend_from_slice(qh); // 32 bytes
        block.extend_from_slice(qs); // 128 bytes
        assert_eq!(block.len(), 176);
        block
    }

    /// Build a Q8_0 test block (34 bytes). Caller supplies `d` (super-scale)
    /// and 32 signed int8 quantised values (represented as `i8` — packed as
    /// two's-complement bytes on write).
    fn build_q8_0_block(d: f32, qs: &[i8; 32]) -> Vec<u8> {
        let mut block = Vec::with_capacity(34);
        block.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        for &q in qs.iter() {
            block.push(q as u8);
        }
        assert_eq!(block.len(), 34);
        block
    }

    /// GPU Q5_K matvec produces the same output as the naive CPU reference
    /// (dequantize → f32 matmul) for a small hand-crafted single-block
    /// weight tensor. Verifies both the shader's dequant math (5-bit
    /// nibble+qh combination, sub-block scale/min lookup) and the workgroup
    /// reduction against the CPU dequant reference (`dequantize_q5_k` in
    /// `gguf.rs`).
    #[test]
    fn q5k_matvec_matches_cpu_dequant_reference() {
        let engine = match try_gpu_engine() {
            Some(e) => e,
            None => return,
        };

        let rows = 8usize;
        let cols = 256usize; // Exactly 1 Q5_K block per row.

        // Deterministic per-row qs / qh patterns — enough coverage over the
        // 4-bit and 5-bit branches without needing PRNG.
        let mut data = Vec::with_capacity(rows * 176);
        for r in 0..rows {
            let mut qs = [0u8; 128];
            let mut qh = [0u8; 32];
            for k in 0..128 {
                let low = ((r + k) & 0xF) as u8;
                let high = ((r + k + 3) & 0xF) as u8;
                qs[k] = low | (high << 4);
            }
            for k in 0..32 {
                qh[k] = ((r + k) & 0xFF) as u8;
            }
            data.extend_from_slice(&build_q5k_block(&qh, &qs));
        }

        // CPU: dequantise to f32 then dense matvec against the input.
        let mut cpu_weights = vec![0.0f32; rows * cols];
        crate::gguf::dequantize_weight_row(&data, crate::gguf::GgmlType::Q5_K, &mut cpu_weights);
        let input: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.001 - 0.128).collect();
        let mut expected = vec![0.0f32; rows];
        for r in 0..rows {
            let mut acc = 0.0f32;
            for c in 0..cols {
                acc += cpu_weights[r * cols + c] * input[c];
            }
            expected[r] = acc;
        }

        // GPU: upload weights + input, run Q5_K matvec.
        let weights = engine.upload_weights_q5k(&data, rows, cols);
        let input_buf = engine.upload_f32(&input);
        let output_buf = engine.alloc_f32(rows);
        {
            let mut pass = engine.begin_pass();
            pass.matvec_q5k(&weights, &input_buf, &output_buf);
            pass.execute();
        }
        let gpu_out = engine.read_f32(&output_buf);

        for r in 0..rows {
            let diff = (gpu_out[r] - expected[r]).abs();
            let scale = expected[r].abs().max(1.0);
            assert!(
                diff < 5e-3 * scale,
                "row {r}: gpu={} expected={} diff={} scale={scale}",
                gpu_out[r],
                expected[r],
                diff
            );
        }
    }

    /// GPU Q8_0 matvec matches the naive CPU reference. Verifies the padded
    /// 9-word block layout and int8 sign extension against `dequantize_q8_0`.
    #[test]
    fn q8_0_matvec_matches_cpu_dequant_reference() {
        let engine = match try_gpu_engine() {
            Some(e) => e,
            None => return,
        };

        // Q8_0 blocks: 32 elements each. Use 8 blocks/row = cols=256, rows=8.
        let rows = 8usize;
        let blocks_per_row = 8usize;
        let cols = blocks_per_row * 32;

        let mut data = Vec::with_capacity(rows * blocks_per_row * 34);
        for r in 0..rows {
            for b in 0..blocks_per_row {
                let d = 0.05 + (r as f32 + b as f32) * 0.01;
                let mut qs = [0i8; 32];
                for k in 0..32 {
                    // Cover both positive and negative i8 values.
                    let v = ((r * 3 + b * 7 + k * 5) as i32 % 200) - 100;
                    qs[k] = v as i8;
                }
                data.extend_from_slice(&build_q8_0_block(d, &qs));
            }
        }

        let mut cpu_weights = vec![0.0f32; rows * cols];
        crate::gguf::dequantize_weight_row(&data, crate::gguf::GgmlType::Q8_0, &mut cpu_weights);
        let input: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.007).sin()).collect();
        let mut expected = vec![0.0f32; rows];
        for r in 0..rows {
            let mut acc = 0.0f32;
            for c in 0..cols {
                acc += cpu_weights[r * cols + c] * input[c];
            }
            expected[r] = acc;
        }

        let weights = engine.upload_weights_q8_0(&data, rows, cols);
        let input_buf = engine.upload_f32(&input);
        let output_buf = engine.alloc_f32(rows);
        {
            let mut pass = engine.begin_pass();
            pass.matvec_q8_0(&weights, &input_buf, &output_buf);
            pass.execute();
        }
        let gpu_out = engine.read_f32(&output_buf);

        for r in 0..rows {
            let diff = (gpu_out[r] - expected[r]).abs();
            let scale = expected[r].abs().max(1.0);
            assert!(
                diff < 5e-3 * scale,
                "row {r}: gpu={} expected={} diff={} scale={scale}",
                gpu_out[r],
                expected[r],
                diff
            );
        }
    }
}
