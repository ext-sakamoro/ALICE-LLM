//! GPU compute engine for quantized inference via wgpu (Metal/Vulkan/DX12).
//!
//! Provides GPU-accelerated matvec, RMSNorm, SiLU, RoPE, and residual add.
//! All operations can be chained via `GpuPass` into a single command buffer,
//! eliminating per-operation submission overhead.

use std::sync::Arc;
use wgpu::util::DeviceExt;

const MATVEC_Q4K_SHADER: &str = include_str!("shaders/dequant_matvec_q4k.wgsl");
const MATVEC_Q6K_SHADER: &str = include_str!("shaders/dequant_matvec_q6k.wgsl");
const RMSNORM_SHADER: &str = include_str!("shaders/rmsnorm.wgsl");
const SILU_MUL_SHADER: &str = include_str!("shaders/silu_mul.wgsl");
const RESIDUAL_ADD_SHADER: &str = include_str!("shaders/residual_add.wgsl");
const ROPE_SHADER: &str = include_str!("shaders/rope.wgsl");
const ATTENTION_SHADER: &str = include_str!("shaders/attention.wgsl");
const KV_CACHE_APPEND_SHADER: &str = include_str!("shaders/kv_cache_append.wgsl");
const SWIGLU_FUSED_Q4K_SHADER: &str = include_str!("shaders/swiglu_fused_q4k.wgsl");
const MATVEC_Q4K_BATCH4_SHADER: &str = include_str!("shaders/dequant_matvec_q4k_batch4.wgsl");
const MATVEC_Q6K_BATCH4_SHADER: &str = include_str!("shaders/dequant_matvec_q6k_batch4.wgsl");
const SWIGLU_FUSED_Q4K_BATCH4_SHADER: &str = include_str!("shaders/swiglu_fused_q4k_batch4.wgsl");

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
    _pad1: u32,
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
    Q6K,
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
    matvec_q4k_pipeline: wgpu::ComputePipeline,
    matvec_q6k_pipeline: wgpu::ComputePipeline,
    rmsnorm_pipeline: wgpu::ComputePipeline,
    silu_mul_pipeline: wgpu::ComputePipeline,
    residual_add_pipeline: wgpu::ComputePipeline,
    rope_pipeline: wgpu::ComputePipeline,
    attention_pipeline: wgpu::ComputePipeline,
    kv_cache_append_pipeline: wgpu::ComputePipeline,
    swiglu_fused_q4k_pipeline: wgpu::ComputePipeline,
    // Batch-4 specialized pipelines (K=4 unrolled scalar accumulators)
    matvec_q4k_batch4_pipeline: wgpu::ComputePipeline,
    matvec_q6k_batch4_pipeline: wgpu::ComputePipeline,
    swiglu_batch4_pipeline: wgpu::ComputePipeline,
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

        let make_pipeline =
            |source: &str, entry: &str, label: &str| -> wgpu::ComputePipeline {
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
        let matvec_q6k_pipeline = make_pipeline(MATVEC_Q6K_SHADER, "matvec_q6k", "matvec_q6k");
        let swiglu_fused_q4k_pipeline = make_pipeline(SWIGLU_FUSED_Q4K_SHADER, "swiglu_fused_q4k", "swiglu_fused_q4k");

        // Batch4 pipelines: share bind group layout with original pipelines
        // so existing bind groups work with both.
        let make_batch4 = |source: &str, entry: &str, label: &str, base_pipeline: &wgpu::ComputePipeline| -> wgpu::ComputePipeline {
            let layout = base_pipeline.get_bind_group_layout(0);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                })),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let matvec_q4k_batch4_pipeline = make_batch4(MATVEC_Q4K_BATCH4_SHADER, "matvec_q4k_batch4", "matvec_q4k_batch4", &matvec_q4k_pipeline);
        let matvec_q6k_batch4_pipeline = make_batch4(MATVEC_Q6K_BATCH4_SHADER, "matvec_q6k_batch4", "matvec_q6k_batch4", &matvec_q6k_pipeline);
        let swiglu_batch4_pipeline = make_batch4(SWIGLU_FUSED_Q4K_BATCH4_SHADER, "swiglu_fused_q4k_batch4", "swiglu_batch4", &swiglu_fused_q4k_pipeline);

        Self {
            matvec_q4k_pipeline,
            matvec_q6k_pipeline,
            rmsnorm_pipeline: make_pipeline(RMSNORM_SHADER, "rmsnorm", "rmsnorm"),
            silu_mul_pipeline: make_pipeline(SILU_MUL_SHADER, "silu_mul", "silu_mul"),
            residual_add_pipeline: make_pipeline(RESIDUAL_ADD_SHADER, "residual_add", "residual_add"),
            rope_pipeline: make_pipeline(ROPE_SHADER, "rope", "rope"),
            attention_pipeline: make_pipeline(ATTENTION_SHADER, "attention", "attention"),
            kv_cache_append_pipeline: make_pipeline(KV_CACHE_APPEND_SHADER, "kv_cache_append", "kv_cache_append"),
            swiglu_fused_q4k_pipeline,
            matvec_q4k_batch4_pipeline,
            matvec_q6k_batch4_pipeline,
            swiglu_batch4_pipeline,
            timestamp_period,
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

    /// Upload Q6_K weight tensor to GPU.
    pub fn upload_weights_q6k(&self, data: &[u8], rows: usize, cols: usize) -> GpuWeightBuffer {
        self.upload_weights_typed(data, rows, cols, GpuQuantType::Q6K)
    }

    fn upload_weights_typed(
        &self,
        data: &[u8],
        rows: usize,
        cols: usize,
        quant: GpuQuantType,
    ) -> GpuWeightBuffer {
        let label = match quant {
            GpuQuantType::Q4K => "q4k_weights",
            GpuQuantType::Q6K => "q6k_weights",
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
            blocks_per_row: (cols / 256) as u32,
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
            _pad1: 0, _pad2: 0, _pad3: 0,
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
        rx.recv()
            .expect("channel closed")
            .expect("map failed");
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
    pub fn matvec_q4k(
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
            _pad1: 0, _pad2: 0, _pad3: 0,
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

    /// RMSNorm: output[i] = input[i] * weight[i] * rsqrt(mean(input^2) + eps).
    pub fn rmsnorm(
        &mut self,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        output: &GpuBuffer,
        eps: f32,
    ) {
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
            _pad1: 0, _pad2: 0, _pad3: 0,
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
    pub fn matvec_q6k(
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
            _pad1: 0, _pad2: 0, _pad3: 0,
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
            GpuQuantType::Q6K => self.matvec_q6k(weights, input, output),
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
}

/// Per-layer pre-cached bind groups.
struct LayerBGs {
    attn_norm_bg: wgpu::BindGroup,
    q_proj: MatvecBG,
    k_proj: MatvecBG,
    v_proj: MatvecBG,
    o_proj: MatvecBG,
    kv_append_bg: wgpu::BindGroup,
    attention_bg: wgpu::BindGroup,
    ffn_norm_bg: wgpu::BindGroup,
    swiglu: SwigluBG,       // fused gate + up + silu×mul
    down_proj: MatvecBG,
}

/// Weight buffer ownership (bind groups reference these internally).
struct LayerWeightBufs {
    attn_norm: GpuBuffer,
    q_proj: GpuWeightBuffer,
    k_proj: GpuWeightBuffer,
    v_proj: GpuWeightBuffer,
    o_proj: GpuWeightBuffer,
    ffn_norm: GpuBuffer,
    gate_proj: GpuWeightBuffer,
    up_proj: GpuWeightBuffer,
    down_proj: GpuWeightBuffer,
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
    _o_buf: GpuBuffer,
    _gate_buf: GpuBuffer,
    _down_buf: GpuBuffer,
    logits: GpuBuffer,
    staging: wgpu::Buffer,

    // KV caches
    k_caches: Vec<GpuBuffer>,
    v_caches: Vec<GpuBuffer>,

    // Pre-cached bind groups
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

    // Dispatch constants
    residual_dispatch_x: u32,
    rope_q_dispatch_x: u32,
    rope_k_dispatch_x: u32,
    kv_dispatch_x: u32,

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
fn timed_mv(engine: &GpuEngine, mv: &MatvecBG) -> (f64, GpuQuantType) {
    let pipeline = match mv.quant {
        GpuQuantType::Q4K => &engine.matvec_q4k_pipeline,
        GpuQuantType::Q6K => &engine.matvec_q6k_pipeline,
    };
    let us = timed_dispatch(engine, pipeline, &mv.bg, mv.dispatch_x, mv.dispatch_y, 1);
    (us, mv.quant)
}

impl GpuModel {
    // --- Static helpers for bind group construction ---

    fn build_matvec_bg(
        engine: &GpuEngine,
        w: &GpuWeightBuffer,
        input: &GpuBuffer,
        output: &GpuBuffer,
    ) -> MatvecBG {
        let (dispatch_x, dispatch_y, grid_x) = matvec_dispatch(w.rows);
        let params_buf = engine.make_persistent_uniform(&MatvecParams {
            rows: w.rows,
            cols: w.cols,
            blocks_per_row: w.blocks_per_row,
            grid_x,
            batch_size: 1,
            _pad1: 0, _pad2: 0, _pad3: 0,
        });
        let pipeline = match w.quant {
            GpuQuantType::Q4K => &engine.matvec_q4k_pipeline,
            GpuQuantType::Q6K => &engine.matvec_q6k_pipeline,
        };
        let layout = pipeline.get_bind_group_layout(0);
        let bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });
        drop(params_buf); // BindGroup holds Arc ref to GPU resource
        MatvecBG { bg, dispatch_x, dispatch_y, quant: w.quant }
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
                wgpu::BindGroupEntry { binding: 0, resource: input.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weight.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
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
        let (dispatch_x, dispatch_y, grid_x) = matvec_dispatch(gate_w.rows);
        let params_buf = engine.make_persistent_uniform(&MatvecParams {
            rows: gate_w.rows,
            cols: gate_w.cols,
            blocks_per_row: gate_w.blocks_per_row,
            grid_x,
            batch_size: 1,
            _pad1: 0, _pad2: 0, _pad3: 0,
        });
        let layout = engine.swiglu_fused_q4k_pipeline.get_bind_group_layout(0);
        let bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gate_w.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: up_w.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: input.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: output.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });
        drop(params_buf); // BindGroup holds Arc ref to GPU resource
        SwigluBG { bg, dispatch_x, dispatch_y }
    }

    /// Load model from GGUF, upload all weights, pre-create all bind groups.
    /// Load model from GGUF (takes ownership of engine).
    #[cfg(feature = "gguf")]
    pub fn load(
        engine: GpuEngine,
        gguf: &crate::gguf::GgufFile,
        config: GpuModelConfig,
    ) -> Self {
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

        let kv_dim = (config.num_kv_heads * config.head_dim) as usize;

        // Upload weight with auto quant type detection
        let upload_w = |name: &str, rows: usize, cols: usize| -> GpuWeightBuffer {
            let info = gguf.tensor_info(name).unwrap();
            let data = gguf.tensor_data(name).unwrap();
            match info.qtype {
                GgmlType::Q6_K => engine.upload_weights_q6k(data, rows, cols),
                _ => engine.upload_weights(data, rows, cols),
            }
        };

        eprintln!("[GpuModel] uploading weights...");
        let t0 = std::time::Instant::now();

        let mut layer_weights = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            layer_weights.push(LayerWeightBufs {
                attn_norm: engine.upload_f32(
                    &gguf.tensor_to_f32(&format!("blk.{i}.attn_norm.weight")).unwrap(),
                ),
                q_proj: upload_w(
                    &format!("blk.{i}.attn_q.weight"), config.hidden_dim, config.hidden_dim,
                ),
                k_proj: upload_w(
                    &format!("blk.{i}.attn_k.weight"), kv_dim, config.hidden_dim,
                ),
                v_proj: upload_w(
                    &format!("blk.{i}.attn_v.weight"), kv_dim, config.hidden_dim,
                ),
                o_proj: upload_w(
                    &format!("blk.{i}.attn_output.weight"), config.hidden_dim, config.hidden_dim,
                ),
                ffn_norm: engine.upload_f32(
                    &gguf.tensor_to_f32(&format!("blk.{i}.ffn_norm.weight")).unwrap(),
                ),
                gate_proj: upload_w(
                    &format!("blk.{i}.ffn_gate.weight"), config.intermediate_dim, config.hidden_dim,
                ),
                up_proj: upload_w(
                    &format!("blk.{i}.ffn_up.weight"), config.intermediate_dim, config.hidden_dim,
                ),
                down_proj: upload_w(
                    &format!("blk.{i}.ffn_down.weight"), config.hidden_dim, config.intermediate_dim,
                ),
            });
        }

        // Output projection — fallback to tied embedding
        let output_norm_weight = engine.upload_f32(
            &gguf.tensor_to_f32("output_norm.weight").unwrap(),
        );
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

        // Scratch buffers (MAX_BATCH sized for batch forward support)
        const MAX_BATCH: usize = 8;
        let hidden = engine.alloc_f32(MAX_BATCH * config.hidden_dim);
        let norm_buf = engine.alloc_f32(MAX_BATCH * config.hidden_dim);
        let q_buf = engine.alloc_f32(MAX_BATCH * config.hidden_dim);
        let k_buf = engine.alloc_f32(MAX_BATCH * kv_dim);
        let v_buf = engine.alloc_f32(MAX_BATCH * kv_dim);
        let attn_out = engine.alloc_f32(MAX_BATCH * config.hidden_dim);
        let o_buf = engine.alloc_f32(MAX_BATCH * config.hidden_dim);
        let gate_buf = engine.alloc_f32(MAX_BATCH * config.intermediate_dim);
        let down_buf = engine.alloc_f32(MAX_BATCH * config.hidden_dim);
        let logits = engine.alloc_f32(MAX_BATCH * vocab_size);
        let staging = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("logits_staging"),
            size: (MAX_BATCH * vocab_size * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // KV caches
        let mut k_caches = Vec::with_capacity(config.num_layers);
        let mut v_caches = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            k_caches.push(engine.alloc_f32(config.max_seq_len * kv_dim));
            v_caches.push(engine.alloc_f32(config.max_seq_len * kv_dim));
        }

        // --- Persistent uniform buffers (updated per-token via write_buffer) ---
        let make_persistent = |data: &[u8]| -> wgpu::Buffer {
            engine.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: data,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };

        let rope_q_params_buf = make_persistent(bytemuck::bytes_of(&RopeParams {
            position: 0,
            num_heads: config.num_heads,
            head_dim: config.head_dim,
            theta: config.rope_theta,
            batch_size: 1,
            _pad1: 0, _pad2: 0, _pad3: 0,
        }));
        let rope_k_params_buf = make_persistent(bytemuck::bytes_of(&RopeParams {
            position: 0,
            num_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            theta: config.rope_theta,
            batch_size: 1,
            _pad1: 0, _pad2: 0, _pad3: 0,
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
                wgpu::BindGroupEntry { binding: 0, resource: q_buf.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: rope_q_params_buf.as_entire_binding() },
            ],
        });
        let rope_k_bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &rope_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: k_buf.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: rope_k_params_buf.as_entire_binding() },
            ],
        });

        let residual_layout = engine.residual_add_pipeline.get_bind_group_layout(0);
        let residual_attn_bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &residual_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: hidden.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: o_buf.buffer.as_entire_binding() },
            ],
        });
        let residual_ffn_bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &residual_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: hidden.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: down_buf.buffer.as_entire_binding() },
            ],
        });

        // RMSNorm params (shared by all norm dispatches)
        let rmsnorm_params_buf = engine.make_persistent_uniform(&RmsnormParams {
            dim: config.hidden_dim as u32,
            eps: config.eps,
            batch_size: 1,
            _pad3: 0,
        });

        // Output head: separate rmsnorm + matvec
        let output_norm_bg = Self::build_rmsnorm_bg(
            &engine, &hidden, &output_norm_weight, &norm_buf, &rmsnorm_params_buf,
        );
        let output_proj_bg = Self::build_matvec_bg(&engine, &output_proj_weight, &norm_buf, &logits);

        // --- Per-layer bind groups ---
        let kv_append_layout = engine.kv_cache_append_pipeline.get_bind_group_layout(0);
        let attn_layout = engine.attention_pipeline.get_bind_group_layout(0);

        let mut layer_bgs = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let lw = &layer_weights[i];

            // Attention: rmsnorm → norm_buf, then plain matvec from norm_buf
            let attn_norm_bg = Self::build_rmsnorm_bg(
                &engine, &hidden, &lw.attn_norm, &norm_buf, &rmsnorm_params_buf,
            );
            let q_proj = Self::build_matvec_bg(&engine, &lw.q_proj, &norm_buf, &q_buf);
            let k_proj = Self::build_matvec_bg(&engine, &lw.k_proj, &norm_buf, &k_buf);
            let v_proj = Self::build_matvec_bg(&engine, &lw.v_proj, &norm_buf, &v_buf);
            let o_proj = Self::build_matvec_bg(&engine, &lw.o_proj, &attn_out, &o_buf);

            // FFN: rmsnorm → norm_buf, then fused swiglu (gate+up+silu×mul)
            let ffn_norm_bg = Self::build_rmsnorm_bg(
                &engine, &hidden, &lw.ffn_norm, &norm_buf, &rmsnorm_params_buf,
            );
            let swiglu = Self::build_swiglu_bg(
                &engine, &lw.gate_proj, &lw.up_proj, &norm_buf, &gate_buf,
            );
            let down_proj = Self::build_matvec_bg(&engine, &lw.down_proj, &gate_buf, &down_buf);

            let kv_append_bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &kv_append_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: k_buf.buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: v_buf.buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: k_caches[i].buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: v_caches[i].buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: kv_append_params_buf.as_entire_binding() },
                ],
            });
            let attention_bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &attn_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: q_buf.buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: k_caches[i].buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: v_caches[i].buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: attn_out.buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: attn_params_buf.as_entire_binding() },
                ],
            });

            layer_bgs.push(LayerBGs {
                attn_norm_bg,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                kv_append_bg,
                attention_bg,
                ffn_norm_bg,
                swiglu,
                down_proj,
            });
        }

        // Dispatch constants
        let residual_dispatch_x = ((config.hidden_dim as u32) + 255) / 256;
        let rope_q_dispatch_x = (config.num_heads * config.head_dim / 2 + 255) / 256;
        let rope_k_dispatch_x = (config.num_kv_heads * config.head_dim / 2 + 255) / 256;
        let kv_dispatch_x = ((kv_dim as u32) + 255) / 256;

        let total_bgs = layer_bgs.len() * 10 + 7;
        eprintln!(
            "[GpuModel] {total_bgs} BGs pre-cached (fused SwiGLU), 4 persistent UBs"
        );

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
            _o_buf: o_buf,
            _gate_buf: gate_buf,
            _down_buf: down_buf,
            logits,
            staging,
            k_caches,
            v_caches,
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
            residual_dispatch_x,
            rope_q_dispatch_x,
            rope_k_dispatch_x,
            kv_dispatch_x,
            embedding,
            vocab_size,
            seq_len: 0,
        }
    }

    // --- Forward pass (zero per-token allocation) ---

    /// Encode the full forward pass into a command encoder.
    fn encode_forward(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());

        for i in 0..self.config.num_layers {
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

            Self::dispatch_mv(&self.engine, &mut cp, &lbg.o_proj);

            cp.set_pipeline(&self.engine.residual_add_pipeline);
            cp.set_bind_group(0, &self.residual_attn_bg, &[]);
            cp.dispatch_workgroups(self.residual_dispatch_x, 1, 1);

            // --- FFN sub-block ---
            // RMSNorm: hidden → norm_buf
            cp.set_pipeline(&self.engine.rmsnorm_pipeline);
            cp.set_bind_group(0, &lbg.ffn_norm_bg, &[]);
            cp.dispatch_workgroups(1, 1, 1);

            // Fused SwiGLU: gate·norm + up·norm → silu(gate)×up → gate_buf
            cp.set_pipeline(&self.engine.swiglu_fused_q4k_pipeline);
            cp.set_bind_group(0, &lbg.swiglu.bg, &[]);
            cp.dispatch_workgroups(lbg.swiglu.dispatch_x, lbg.swiglu.dispatch_y, 1);

            Self::dispatch_mv(&self.engine, &mut cp, &lbg.down_proj);

            cp.set_pipeline(&self.engine.residual_add_pipeline);
            cp.set_bind_group(0, &self.residual_ffn_bg, &[]);
            cp.dispatch_workgroups(self.residual_dispatch_x, 1, 1);
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
        let pipeline = match mv.quant {
            GpuQuantType::Q4K => &engine.matvec_q4k_pipeline,
            GpuQuantType::Q6K => &engine.matvec_q6k_pipeline,
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
        };
        cp.set_pipeline(pipeline);
        cp.set_bind_group(0, &mv.bg, &[]);
        cp.dispatch_workgroups(mv.dispatch_x, mv.dispatch_y, 1);
    }

    /// Update persistent uniform buffers for the current position.
    fn update_uniforms(&self, pos: u32, seq_len: u32) {
        self.engine.queue.write_buffer(
            &self.rope_q_params_buf, 0, bytemuck::bytes_of(&pos),
        );
        self.engine.queue.write_buffer(
            &self.rope_k_params_buf, 0, bytemuck::bytes_of(&pos),
        );
        self.engine.queue.write_buffer(
            &self.kv_append_params_buf, 0, bytemuck::bytes_of(&pos),
        );
        self.engine.queue.write_buffer(
            &self.attn_params_buf, 0, bytemuck::bytes_of(&seq_len),
        );
    }

    /// Upload token embedding to the hidden buffer.
    fn upload_embedding(&self, token_id: u32) {
        let start = token_id as usize * self.config.hidden_dim;
        let end = start + self.config.hidden_dim;
        self.engine.queue.write_buffer(
            &self.hidden.buffer, 0,
            bytemuck::cast_slice(&self.embedding[start..end]),
        );
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

        let mut encoder = self.engine.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor::default(),
        );
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

        let mut encoder = self.engine.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor::default(),
        );
        self.encode_forward(&mut encoder);

        let size = (self.vocab_size * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.logits.buffer, 0, &self.staging, 0, size);

        self.engine.queue.submit(Some(encoder.finish()));
        self.seq_len = seq_len;
        self.engine.map_staging(&self.staging, self.vocab_size)
    }

    /// Reset KV caches and position counter.
    pub fn reset(&mut self) {
        self.seq_len = 0;
        let kv_dim = (self.config.num_kv_heads * self.config.head_dim) as usize;
        let zeros = vec![0.0f32; self.config.max_seq_len * kv_dim];
        for i in 0..self.config.num_layers {
            self.engine.write_f32(&self.k_caches[i], &zeros);
            self.engine.write_f32(&self.v_caches[i], &zeros);
        }
    }

    /// Rollback KV cache to a previous position (for speculative decoding).
    /// GPU KV memory beyond `pos` is stale but harmless — attention shader
    /// only reads up to `seq_len`, and future appends overwrite old entries.
    pub fn rollback_to(&mut self, pos: u32) {
        debug_assert!(pos <= self.seq_len, "rollback_to({pos}) > seq_len({})", self.seq_len);
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
            rmsnorm_us: 0.0, matvec_q4k_us: 0.0, matvec_q6k_us: 0.0,
            swiglu_us: 0.0, rope_us: 0.0, kv_append_us: 0.0,
            attention_us: 0.0, residual_us: 0.0,
            rmsnorm_n: 0, matvec_q4k_n: 0, matvec_q6k_n: 0,
            swiglu_n: 0, rope_n: 0, kv_append_n: 0,
            attention_n: 0, residual_n: 0,
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
                    GpuQuantType::Q4K => { r.matvec_q4k_us += us; r.matvec_q4k_n += 1; }
                    GpuQuantType::Q6K => { r.matvec_q6k_us += us; r.matvec_q6k_n += 1; }
                }
            }

            r.rope_us += timed_dispatch(e, &e.rope_pipeline, &self.rope_q_bg, self.rope_q_dispatch_x, 1, 1);
            r.rope_n += 1;
            r.rope_us += timed_dispatch(e, &e.rope_pipeline, &self.rope_k_bg, self.rope_k_dispatch_x, 1, 1);
            r.rope_n += 1;

            r.kv_append_us += timed_dispatch(e, &e.kv_cache_append_pipeline, &lbg.kv_append_bg, self.kv_dispatch_x, 1, 1);
            r.kv_append_n += 1;

            r.attention_us += timed_dispatch(e, &e.attention_pipeline, &lbg.attention_bg, self.config.num_heads, 1, 1);
            r.attention_n += 1;

            let (us, qt) = timed_mv(e, &lbg.o_proj);
            match qt {
                GpuQuantType::Q4K => { r.matvec_q4k_us += us; r.matvec_q4k_n += 1; }
                GpuQuantType::Q6K => { r.matvec_q6k_us += us; r.matvec_q6k_n += 1; }
            }

            r.residual_us += timed_dispatch(e, &e.residual_add_pipeline, &self.residual_attn_bg, self.residual_dispatch_x, 1, 1);
            r.residual_n += 1;

            // --- FFN sub-block ---
            r.rmsnorm_us += timed_dispatch(e, &e.rmsnorm_pipeline, &lbg.ffn_norm_bg, 1, 1, 1);
            r.rmsnorm_n += 1;

            r.swiglu_us += timed_dispatch(e, &e.swiglu_fused_q4k_pipeline, &lbg.swiglu.bg, lbg.swiglu.dispatch_x, lbg.swiglu.dispatch_y, 1);
            r.swiglu_n += 1;

            let (us, qt) = timed_mv(e, &lbg.down_proj);
            match qt {
                GpuQuantType::Q4K => { r.matvec_q4k_us += us; r.matvec_q4k_n += 1; }
                GpuQuantType::Q6K => { r.matvec_q6k_us += us; r.matvec_q6k_n += 1; }
            }

            r.residual_us += timed_dispatch(e, &e.residual_add_pipeline, &self.residual_ffn_bg, self.residual_dispatch_x, 1, 1);
            r.residual_n += 1;
        }

        // --- Output head ---
        r.rmsnorm_us += timed_dispatch(e, &e.rmsnorm_pipeline, &self.output_norm_bg, 1, 1, 1);
        r.rmsnorm_n += 1;

        let (us, qt) = timed_mv(e, &self.output_proj_bg);
        match qt {
            GpuQuantType::Q4K => { r.matvec_q4k_us += us; r.matvec_q4k_n += 1; }
            GpuQuantType::Q6K => { r.matvec_q6k_us += us; r.matvec_q6k_n += 1; }
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
                &self.hidden.buffer, buf_offset,
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
        assert!(token_ids.len() == 4, "forward_batch requires exactly 4 tokens");
        let base_pos = self.seq_len;

        // Update per-position uniforms
        self.engine.queue.write_buffer(
            &self.rope_q_params_buf, 0, bytemuck::bytes_of(&base_pos),
        );
        self.engine.queue.write_buffer(
            &self.rope_k_params_buf, 0, bytemuck::bytes_of(&base_pos),
        );
        self.engine.queue.write_buffer(
            &self.kv_append_params_buf, 0, bytemuck::bytes_of(&base_pos),
        );
        let attn_seq_len = base_pos + 1;
        self.engine.queue.write_buffer(
            &self.attn_params_buf, 0, bytemuck::bytes_of(&attn_seq_len),
        );

        self.upload_embeddings_batch(token_ids);

        let mut encoder = self.engine.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor::default(),
        );
        self.encode_forward_batch4(&mut encoder);
        self.engine.queue.submit(Some(encoder.finish()));

        self.seq_len = base_pos + 4;
    }

    /// Execute batch-4 forward pass with logits readback for all 4 tokens.
    pub fn forward_batch_and_read(&mut self, token_ids: &[u32]) -> Vec<f32> {
        assert!(token_ids.len() == 4, "forward_batch_and_read requires exactly 4 tokens");
        let base_pos = self.seq_len;

        self.engine.queue.write_buffer(
            &self.rope_q_params_buf, 0, bytemuck::bytes_of(&base_pos),
        );
        self.engine.queue.write_buffer(
            &self.rope_k_params_buf, 0, bytemuck::bytes_of(&base_pos),
        );
        self.engine.queue.write_buffer(
            &self.kv_append_params_buf, 0, bytemuck::bytes_of(&base_pos),
        );
        let attn_seq_len = base_pos + 1;
        self.engine.queue.write_buffer(
            &self.attn_params_buf, 0, bytemuck::bytes_of(&attn_seq_len),
        );
        self.upload_embeddings_batch(token_ids);

        let mut encoder = self.engine.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor::default(),
        );
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

        // 14 dispatches per layer × num_layers + 2 output head = total dispatches
        let ops_per_layer: u32 = 14;
        let n_layers = self.config.num_layers as u32;
        let total_dispatches = ops_per_layer * n_layers + 2;
        // Each dispatch uses 2 query slots (beginning + end of compute pass)
        let num_queries = total_dispatches * 2;

        let query_set = self.engine.device.create_query_set(&wgpu::QuerySetDescriptor {
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

        let mut encoder = self.engine.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor::default(),
        );

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
                let pipeline = match $mv.quant {
                    GpuQuantType::Q4K => &self.engine.matvec_q4k_pipeline,
                    GpuQuantType::Q6K => &self.engine.matvec_q6k_pipeline,
                };
                let tag = match $mv.quant {
                    GpuQuantType::Q4K => 1u8,
                    GpuQuantType::Q6K => 2u8,
                };
                ts_dispatch!(pipeline, &$mv.bg, $mv.dispatch_x, $mv.dispatch_y, 1, tag);
            }};
        }

        for i in 0..self.config.num_layers {
            let lbg = &self.layer_bgs[i];

            // Attention sub-block
            ts_dispatch!(&self.engine.rmsnorm_pipeline, &lbg.attn_norm_bg, 1, 1, 1, 0u8);
            ts_dispatch_mv!(lbg.q_proj);
            ts_dispatch_mv!(lbg.k_proj);
            ts_dispatch_mv!(lbg.v_proj);
            ts_dispatch!(&self.engine.rope_pipeline, &self.rope_q_bg, self.rope_q_dispatch_x, 1, 1, 4u8);
            ts_dispatch!(&self.engine.rope_pipeline, &self.rope_k_bg, self.rope_k_dispatch_x, 1, 1, 4u8);
            ts_dispatch!(&self.engine.kv_cache_append_pipeline, &lbg.kv_append_bg, self.kv_dispatch_x, 1, 1, 5u8);
            ts_dispatch!(&self.engine.attention_pipeline, &lbg.attention_bg, self.config.num_heads, 1, 1, 6u8);
            ts_dispatch_mv!(lbg.o_proj);
            ts_dispatch!(&self.engine.residual_add_pipeline, &self.residual_attn_bg, self.residual_dispatch_x, 1, 1, 7u8);

            // FFN sub-block
            ts_dispatch!(&self.engine.rmsnorm_pipeline, &lbg.ffn_norm_bg, 1, 1, 1, 0u8);
            ts_dispatch!(&self.engine.swiglu_fused_q4k_pipeline, &lbg.swiglu.bg, lbg.swiglu.dispatch_x, lbg.swiglu.dispatch_y, 1, 3u8);
            ts_dispatch_mv!(lbg.down_proj);
            ts_dispatch!(&self.engine.residual_add_pipeline, &self.residual_ffn_bg, self.residual_dispatch_x, 1, 1, 7u8);
        }

        // Output head
        ts_dispatch!(&self.engine.rmsnorm_pipeline, &self.output_norm_bg, 1, 1, 1, 0u8);
        ts_dispatch_mv!(self.output_proj_bg);

        // Resolve timestamps to buffer
        encoder.resolve_query_set(&query_set, 0..num_queries, &resolve_buf, 0);
        encoder.copy_buffer_to_buffer(&resolve_buf, 0, &readback_buf, 0, (num_queries as u64) * 8);

        self.engine.queue.submit(Some(encoder.finish()));
        self.engine.device.poll(wgpu::Maintain::Wait);

        // Readback timestamps
        let slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        self.engine.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("channel closed").expect("map failed");
        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);

        let period_us = self.engine.timestamp_period as f64 / 1000.0; // ns→µs

        let mut r = ProfileResult {
            rmsnorm_us: 0.0, matvec_q4k_us: 0.0, matvec_q6k_us: 0.0,
            swiglu_us: 0.0, rope_us: 0.0, kv_append_us: 0.0,
            attention_us: 0.0, residual_us: 0.0,
            rmsnorm_n: 0, matvec_q4k_n: 0, matvec_q6k_n: 0,
            swiglu_n: 0, rope_n: 0, kv_append_n: 0,
            attention_n: 0, residual_n: 0,
        };

        for i in 0..total_dispatches as usize {
            let begin_ts = timestamps[i * 2];
            let end_ts = timestamps[i * 2 + 1];
            let delta_us = (end_ts.wrapping_sub(begin_ts)) as f64 * period_us;
            let tag = op_tags[i];
            match tag {
                0 => { r.rmsnorm_us += delta_us; r.rmsnorm_n += 1; }
                1 => { r.matvec_q4k_us += delta_us; r.matvec_q4k_n += 1; }
                2 => { r.matvec_q6k_us += delta_us; r.matvec_q6k_n += 1; }
                3 => { r.swiglu_us += delta_us; r.swiglu_n += 1; }
                4 => { r.rope_us += delta_us; r.rope_n += 1; }
                5 => { r.kv_append_us += delta_us; r.kv_append_n += 1; }
                6 => { r.attention_us += delta_us; r.attention_n += 1; }
                7 => { r.residual_us += delta_us; r.residual_n += 1; }
                _ => {}
            }
        }

        drop(data);
        readback_buf.unmap();

        self.seq_len = seq_len;
        r
    }
}
