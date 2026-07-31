//! GPU (wgpu) backend for the KV-outer sparse-attention forward pass.
//!
//! Compiles the `sparse_attention_forward.wgsl` compute shader and dispatches
//! `total_partials = idx.edges.len() * qhead` workgroups. Each workgroup owns
//! one `(edge, qhead_lane)` partial and internally parallelizes across the
//! KV block positions (up to `MAX_BLOCK_SIZE = 128` in the shader) with 64
//! threads.
//!
//! MVP scope:
//! * `causal = false` only (right-aligned causal mask is not yet in the
//!   shader; the CPU forward stays the reference for causal workloads).
//! * `block_size <= 128`, `pages_per_block` limited to what fits in
//!   `MAX_BLOCK_SIZE`.
//! * `head_dim` arbitrary — each thread handles multiple output dims via
//!   stride 64.
//! * Numerically bit-close to the CPU forward (rel err < 1e-3 in tests).
//!
//! Everything is gated on `feature = "gpu"` (which pulls in `wgpu`,
//! `pollster`, and `bytemuck`, matching the rest of the crate).

#![cfg(feature = "gpu")]

use std::borrow::Cow;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::forward::ForwardPartials;
use super::scheduler::enumerate_work_units;
use super::types::{BlockTables, CuSeqlensQ, KvOuterIndex, SparseAttentionError};

const SHADER_WGSL: &str = include_str!("../shaders/sparse_attention_forward.wgsl");
const SHADER_MAX_BLOCK_SIZE: usize = 128;

// GPU-side push-constant / uniform block. Must match the WGSL `Params` struct
// byte-for-byte (12 × u32 = 48 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuParams {
    total_partials: u32,
    qhead: u32,
    head_dim: u32,
    block_size: u32,
    page_size: u32,
    pages_per_block: u32,
    hkv: u32,
    hq: u32,
    msb: u32,
    max_pages: u32,
    softmax_scale: f32,
    _pad: u32,
}

// Public engine
// ---------------------------------------------------------------------------

/// wgpu-backed sparse-attention forward engine. Own the wgpu device + queue
/// once at process startup and reuse across calls; adapter / pipeline
/// creation is expensive.
pub struct SparseAttentionGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
}

impl SparseAttentionGpu {
    /// Initialize a default GPU backend (high-performance adapter, whichever
    /// wgpu picks).
    pub fn new_blocking() -> Result<Self, SparseAttentionError> {
        pollster::block_on(Self::new_async())
    }

    /// Async version of [`Self::new_blocking`].
    pub async fn new_async() -> Result<Self, SparseAttentionError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok_or(SparseAttentionError::GpuInitFailed { what: "no adapter" })?;
        let adapter_limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
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
            .map_err(|_| SparseAttentionError::GpuInitFailed {
                what: "device request",
            })?;
        Self::from_device_queue(Arc::new(device), Arc::new(queue))
    }

    /// Reuse an existing wgpu device + queue (for callers that already
    /// maintain a `GpuEngine`).
    pub fn from_device_queue(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Result<Self, SparseAttentionError> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sparse_attention_forward"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_WGSL)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sparse_attention_forward"),
            layout: None,
            module: &module,
            entry_point: Some("sparse_forward"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Ok(Self {
            device,
            queue,
            pipeline,
        })
    }

    /// Run the KV-outer forward on the GPU.
    ///
    /// Signature mirrors [`super::forward::kvouter_forward`] except `causal`
    /// is unsupported in the MVP shader — pass `false`. The `used_kv_lens`
    /// argument accepts `None` (interpreted as full block coverage).
    #[allow(clippy::too_many_arguments)]
    pub fn kvouter_forward_gpu(
        &self,
        q: &[f32],
        k_pages: &[f32],
        v_pages: &[f32],
        idx: &KvOuterIndex,
        block_tables: &BlockTables,
        cu_seqlens_q: &CuSeqlensQ,
        used_kv_lens: Option<&[i32]>,
        hq: usize,
        hkv: usize,
        head_dim: usize,
        block_size: usize,
        page_size: usize,
        softmax_scale: f32,
    ) -> Result<ForwardPartials, SparseAttentionError> {
        // --- Argument sanity (mirrors the CPU forward) --------------------
        if hkv == 0 || hq == 0 || hq % hkv != 0 {
            return Err(SparseAttentionError::HeadCountMismatch { hq, hkv });
        }
        let qhead = hq / hkv;
        if head_dim == 0 || block_size == 0 || page_size == 0 || block_size % page_size != 0 {
            return Err(SparseAttentionError::BlockPageMismatch {
                block_size,
                page_size,
            });
        }
        if block_size > SHADER_MAX_BLOCK_SIZE {
            return Err(SparseAttentionError::BlockPageMismatch {
                block_size,
                page_size,
            });
        }
        if idx.hkv != hkv || idx.pages_per_block != block_size / page_size {
            return Err(SparseAttentionError::ShapeMismatch {
                what: "idx (hkv or pages_per_block)",
                expected: hkv,
                got: idx.hkv,
            });
        }
        let tq = cu_seqlens_q.total_tq();
        let expected_q = tq * hq * head_dim;
        if q.len() != expected_q {
            return Err(SparseAttentionError::ShapeMismatch {
                what: "q",
                expected: expected_q,
                got: q.len(),
            });
        }
        let page_stride = hkv * page_size * head_dim;
        if k_pages.len() % page_stride != 0 || v_pages.len() != k_pages.len() {
            return Err(SparseAttentionError::ShapeMismatch {
                what: "k_pages / v_pages page stride",
                expected: page_stride,
                got: k_pages.len(),
            });
        }

        let total_edges = idx.edges.len();
        let total_partials = total_edges * qhead;

        // Zero-edge case: nothing to dispatch, return empty partials.
        if total_partials == 0 {
            return Ok(ForwardPartials {
                o_partial: Vec::new(),
                m_partial: Vec::new(),
                l_partial: Vec::new(),
                total_edges,
                qhead,
                head_dim,
            });
        }

        // --- Resolve used_kv_lens on the host (default = full msb range) --
        let default_lk = (idx.msb * block_size) as i32;
        let used: Vec<i32> = used_kv_lens
            .map(<[i32]>::to_vec)
            .unwrap_or_else(|| vec![default_lk; cu_seqlens_q.batch_size()]);
        if used.len() != cu_seqlens_q.batch_size() {
            return Err(SparseAttentionError::ShapeMismatch {
                what: "used_kv_lens",
                expected: cu_seqlens_q.batch_size(),
                got: used.len(),
            });
        }

        // --- Build edge_meta [q_idx, h, blk, batch, block_len] × total_edges
        // `block_len` is precomputed here so the shader does not need a
        // separate `used_kv_lens` binding (which would push us over the
        // Metal default limit of 8 storage buffers per compute stage).
        let mut edge_meta: Vec<i32> = Vec::with_capacity(total_edges * 5);
        for unit in enumerate_work_units(idx) {
            let raw_slot = idx.raw_slot(unit.head, unit.compact_slot) as usize;
            let b = raw_slot / idx.msb;
            let blk = raw_slot % idx.msb;
            let lk = used[b].max(0) as usize;
            let block_start = blk * block_size;
            let block_end = ((blk + 1) * block_size).min(lk);
            let block_len = block_end.saturating_sub(block_start) as i32;
            let (start, end) = idx.edge_range(unit.head, unit.compact_slot);
            for abs in start..end {
                let (q_idx, _rank) = idx.edges[abs];
                edge_meta.push(q_idx);
                edge_meta.push(unit.head as i32);
                edge_meta.push(blk as i32);
                edge_meta.push(b as i32);
                edge_meta.push(block_len);
            }
        }
        debug_assert_eq!(edge_meta.len(), total_edges * 5);

        let params = GpuParams {
            total_partials: total_partials as u32,
            qhead: qhead as u32,
            head_dim: head_dim as u32,
            block_size: block_size as u32,
            page_size: page_size as u32,
            pages_per_block: (block_size / page_size) as u32,
            hkv: hkv as u32,
            hq: hq as u32,
            msb: idx.msb as u32,
            max_pages: block_tables.max_pages as u32,
            softmax_scale,
            _pad: 0,
        };

        // --- Upload input buffers ----------------------------------------
        let device = &self.device;

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sparse.params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sparse.q"),
            contents: bytemuck::cast_slice(q),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let k_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sparse.k"),
            contents: bytemuck::cast_slice(k_pages),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sparse.v"),
            contents: bytemuck::cast_slice(v_pages),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let edge_meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sparse.edge_meta"),
            contents: bytemuck::cast_slice(&edge_meta),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let block_tables_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sparse.block_tables"),
            contents: bytemuck::cast_slice(&block_tables.table),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Output buffers.
        let o_len_bytes = (total_partials * head_dim * 4) as u64;
        let m_len_bytes = (total_partials * 4) as u64;
        let l_len_bytes = (total_partials * 4) as u64;

        let o_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sparse.o_out"),
            size: o_len_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let m_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sparse.m_out"),
            size: m_len_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let l_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sparse.l_out"),
            size: l_len_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Staging buffers for readback.
        let o_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sparse.o_stage"),
            size: o_len_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let m_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sparse.m_stage"),
            size: m_len_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let l_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sparse.l_stage"),
            size: l_len_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // --- Bind group + dispatch ---------------------------------------
        let bind_group_layout = self.pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sparse.bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: q_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: k_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: v_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: edge_meta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: block_tables_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: o_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: m_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: l_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sparse.encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sparse.forward"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(total_partials as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&o_buf, 0, &o_stage, 0, o_len_bytes);
        encoder.copy_buffer_to_buffer(&m_buf, 0, &m_stage, 0, m_len_bytes);
        encoder.copy_buffer_to_buffer(&l_buf, 0, &l_stage, 0, l_len_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        // --- Readback -----------------------------------------------------
        let o_partial = read_f32_buffer(&self.device, &o_stage, total_partials * head_dim)?;
        let m_partial = read_f32_buffer(&self.device, &m_stage, total_partials)?;
        let l_partial = read_f32_buffer(&self.device, &l_stage, total_partials)?;

        Ok(ForwardPartials {
            o_partial,
            m_partial,
            l_partial,
            total_edges,
            qhead,
            head_dim,
        })
    }
}

fn read_f32_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    n_elements: usize,
) -> Result<Vec<f32>, SparseAttentionError> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        // Discard the mapping error type here; the outer channel receive
        // detects the case.
        let _ = tx.send(res.is_ok());
    });
    device.poll(wgpu::Maintain::Wait);
    let ok = rx
        .recv()
        .map_err(|_| SparseAttentionError::GpuRuntimeError { what: "map recv" })?;
    if !ok {
        return Err(SparseAttentionError::GpuRuntimeError { what: "map_async" });
    }
    let data = slice.get_mapped_range();
    let out: Vec<f32> = bytemuck::cast_slice(&data)[..n_elements].to_vec();
    drop(data);
    buffer.unmap();
    Ok(out)
}

// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::{
        build_kvouter_index,
        forward::kvouter_forward,
        types::{BlockTables, CuSeqlensQ, SparseSelection},
    };
    use super::*;

    fn setup_gpu() -> Option<SparseAttentionGpu> {
        // CI / headless environments may not have a GPU adapter. Skip the
        // test gracefully rather than fail.
        SparseAttentionGpu::new_blocking().ok()
    }

    #[test]
    fn gpu_forward_matches_cpu_forward_small() {
        let Some(gpu) = setup_gpu() else {
            eprintln!("skip: no GPU adapter available");
            return;
        };

        // Same fixture as combine::sparse_full_matches_dense_reference_no_causal
        // but with `causal=false` end-to-end. The GPU MVP shader does not do
        // causal masking.
        let tq = 2;
        let hq = 1;
        let hkv = 1;
        let head_dim = 4;
        let page_size = 2;
        let block_size = 2;
        let max_pages = 3;
        let num_pages = 3;

        let q: Vec<f32> = (0..tq * hq * head_dim).map(|i| (i as f32) * 0.11).collect();
        let k: Vec<f32> = (0..num_pages * hkv * page_size * head_dim)
            .map(|i| ((i as f32) * 0.07).sin())
            .collect();
        let v: Vec<f32> = (0..num_pages * hkv * page_size * head_dim)
            .map(|i| ((i as f32) * 0.13).cos())
            .collect();

        let tbl = BlockTables::new(vec![0, 1, 2], 1, max_pages).unwrap();
        let cu = CuSeqlensQ::new(vec![0, tq as i64]).unwrap();
        let used = vec![6i32];
        let mut sel_flat = Vec::new();
        for _i in 0..tq {
            for _h in 0..hkv {
                for r in 0..max_pages {
                    sel_flat.push(r as i32);
                }
            }
        }
        let sel = SparseSelection::new(sel_flat, tq, hkv, max_pages).unwrap();
        let idx = build_kvouter_index(&sel, &tbl, &cu, Some(&used), block_size, page_size).unwrap();

        let softmax_scale = 1.0f32 / (head_dim as f32).sqrt();

        let cpu = kvouter_forward(
            &q,
            &k,
            &v,
            &idx,
            &tbl,
            &cu,
            Some(&used),
            hq,
            hkv,
            head_dim,
            block_size,
            page_size,
            softmax_scale,
            false,
        )
        .unwrap();
        let gpu_out = gpu
            .kvouter_forward_gpu(
                &q,
                &k,
                &v,
                &idx,
                &tbl,
                &cu,
                Some(&used),
                hq,
                hkv,
                head_dim,
                block_size,
                page_size,
                softmax_scale,
            )
            .unwrap();

        assert_eq!(cpu.o_partial.len(), gpu_out.o_partial.len());
        for (i, (c, g)) in cpu.o_partial.iter().zip(&gpu_out.o_partial).enumerate() {
            let diff = (c - g).abs();
            let rel = diff / c.abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "o_partial mismatch at {i}: cpu={c}, gpu={g}, rel={rel}"
            );
        }
        for (i, (c, g)) in cpu.m_partial.iter().zip(&gpu_out.m_partial).enumerate() {
            assert!(
                (c - g).abs() < 1e-3,
                "m_partial mismatch at {i}: cpu={c}, gpu={g}"
            );
        }
        for (i, (c, g)) in cpu.l_partial.iter().zip(&gpu_out.l_partial).enumerate() {
            let diff = (c - g).abs();
            let rel = diff / c.abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "l_partial mismatch at {i}: cpu={c}, gpu={g}, rel={rel}"
            );
        }
    }

    #[test]
    fn gpu_forward_gqa_multi_batch() {
        let Some(gpu) = setup_gpu() else {
            eprintln!("skip: no GPU adapter available");
            return;
        };
        // batch_size=2, cu_seqlens_q=[0,2,3], Hq=4, Hkv=2, qhead=2,
        // head_dim=8, page_size=4, block_size=4, max_pages=3.
        let tq = 3;
        let hq = 4;
        let hkv = 2;
        let head_dim = 8;
        let page_size = 4;
        let block_size = 4;
        let max_pages = 3;
        let num_pages = 3;
        let batch_size = 2;

        let q: Vec<f32> = (0..tq * hq * head_dim)
            .map(|i| ((i as f32) * 0.05).sin())
            .collect();
        let k: Vec<f32> = (0..num_pages * hkv * page_size * head_dim)
            .map(|i| ((i as f32) * 0.03).cos())
            .collect();
        let v: Vec<f32> = (0..num_pages * hkv * page_size * head_dim)
            .map(|i| ((i as f32) * 0.09).sin())
            .collect();

        // Two batches share the same paged K/V, but each batch's block_tables
        // covers a different subset. Build block_tables shape [batch_size, max_pages].
        let mut tbl_flat: Vec<i32> = Vec::new();
        for _ in 0..batch_size {
            for p in 0..max_pages {
                tbl_flat.push(p as i32);
            }
        }
        let tbl = BlockTables::new(tbl_flat, batch_size, max_pages).unwrap();
        let cu = CuSeqlensQ::new(vec![0, 2, 3]).unwrap();
        let used = vec![12i32, 12];
        // Select all blocks for every (query, kv_head).
        let topk = max_pages;
        let mut sel_flat = Vec::new();
        for _i in 0..tq {
            for _h in 0..hkv {
                for r in 0..topk {
                    sel_flat.push(r as i32);
                }
            }
        }
        let sel = SparseSelection::new(sel_flat, tq, hkv, topk).unwrap();
        let idx = build_kvouter_index(&sel, &tbl, &cu, Some(&used), block_size, page_size).unwrap();

        let softmax_scale = 1.0f32 / (head_dim as f32).sqrt();

        let cpu = kvouter_forward(
            &q,
            &k,
            &v,
            &idx,
            &tbl,
            &cu,
            Some(&used),
            hq,
            hkv,
            head_dim,
            block_size,
            page_size,
            softmax_scale,
            false,
        )
        .unwrap();
        let gpu_out = gpu
            .kvouter_forward_gpu(
                &q,
                &k,
                &v,
                &idx,
                &tbl,
                &cu,
                Some(&used),
                hq,
                hkv,
                head_dim,
                block_size,
                page_size,
                softmax_scale,
            )
            .unwrap();

        for (i, (c, g)) in cpu.o_partial.iter().zip(&gpu_out.o_partial).enumerate() {
            let diff = (c - g).abs();
            let rel = diff / c.abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "gqa o_partial mismatch at {i}: cpu={c}, gpu={g}, rel={rel}"
            );
        }
    }
}
