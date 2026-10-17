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

// `#[cfg(feature = "gpu")]` is applied on the `mod gpu;` line in `mod.rs`;
// duplicating it here would trip `clippy::duplicated_attributes`.

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
        if hkv == 0 || hq == 0 || !hq.is_multiple_of(hkv) {
            return Err(SparseAttentionError::HeadCountMismatch { hq, hkv });
        }
        let qhead = hq / hkv;
        if head_dim == 0
            || block_size == 0
            || page_size == 0
            || !block_size.is_multiple_of(page_size)
        {
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
        if !k_pages.len().is_multiple_of(page_stride) || v_pages.len() != k_pages.len() {
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
        let used: Vec<i32> = used_kv_lens.map_or_else(
            || vec![default_lk; cu_seqlens_q.batch_size()],
            <[i32]>::to_vec,
        );
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
        // Without an adapter the test is skipped, except under
        // ALICE_LLM_REQUIRE_GPU (see `crate::gpu::test_adapter`).
        if crate::gpu::test_adapter::hidden() {
            crate::gpu::test_adapter::no_adapter("sparse_attention setup_gpu");
            return None;
        }
        match SparseAttentionGpu::new_blocking() {
            Ok(gpu) => {
                eprintln!("[gpu] sparse attention device acquired");
                Some(gpu)
            }
            Err(e) => {
                crate::gpu::test_adapter::no_adapter(&format!("sparse_attention setup_gpu: {e}"));
                None
            }
        }
    }

    #[test]
    #[expect(
        clippy::disallowed_methods,
        reason = "G1 reference: test oracle or example code, not Law output; each call is listed in scripts/g1_reference_sites.tsv"
    )]
    fn gpu_forward_matches_cpu_forward_small() {
        let Some(gpu) = setup_gpu() else {
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
    #[expect(
        clippy::disallowed_methods,
        reason = "G1 reference: test oracle or example code, not Law output; each call is listed in scripts/g1_reference_sites.tsv"
    )]
    fn gpu_forward_gqa_multi_batch() {
        let Some(gpu) = setup_gpu() else {
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

    /// Bit-identical old-vs-new oracle for `sparse_attention_forward.wgsl`
    /// (the no-early-return rewrite): covers the out-of-range-workgroup
    /// padding case (the removed `partial_idx >= total_partials` guard) and
    /// the legitimate in-range `block_len == 0` case (the removed
    /// early-bail-and-write-sentinel block) separately, since they are
    /// different removed guards with different replacement strategies — see
    /// `sparse_attention_forward.wgsl`'s module doc. `qhead = 1` so
    /// `abs_edge == partial_idx` directly, simplifying `edge_meta`
    /// construction to one 5-i32 record per partial. `page_id` is always -1
    /// (never dereferenced as a valid page — robustness-safe, and the point
    /// here is old-vs-new control-flow agreement, not realistic V-weighting).
    #[cfg(feature = "internal-kernel-oracle")]
    #[test]
    fn sparse_attention_forward_old_vs_new_bit_identical() {
        use wgpu::util::DeviceExt as _;

        const OLD_SHADER: &str =
            include_str!("../../tests/ref_shaders/sparse_attention_forward_old.wgsl");

        let Some(gpu) = setup_gpu() else {
            return;
        };
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();

        let old_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sparse_attention_forward_old"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(OLD_SHADER)),
        });
        let old_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sparse_attention_forward_old"),
            layout: None,
            module: &old_module,
            entry_point: Some("sparse_forward"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        const HEAD_DIM: u32 = 4;
        const BLOCK_SIZE: u32 = 8;
        const PAGE_SIZE: u32 = 8;
        const PAD: u32 = 3;

        // (total_partials_real, block_len for every edge, label)
        let cases: &[(u32, i32, &str)] = &[
            (0, 0, "zero partials (every workgroup is padding)"),
            (1, 0, "one partial, legitimate block_len=0"),
            (1, 1, "one partial, block_len=1"),
            (3, 0, "several partials, all block_len=0"),
            (
                3,
                (BLOCK_SIZE - 1) as i32,
                "several partials, block_len=tile-1",
            ),
            (5, BLOCK_SIZE as i32, "several partials, block_len=tile"),
        ];

        let mut table: Vec<(u32, i32, usize, usize)> = Vec::new(); // (total_partials_real, block_len, compared, mismatched)

        for &(total_partials_real, block_len, _label) in cases {
            let dispatch_x = total_partials_real + PAD;

            let mut edge_meta: Vec<i32> = Vec::with_capacity((total_partials_real * 5) as usize);
            for p in 0..total_partials_real {
                edge_meta.push(0); // q_idx
                edge_meta.push(0); // h
                edge_meta.push(0); // blk
                edge_meta.push((p % 2) as i32); // batch — varied, just to touch indexing
                edge_meta.push(block_len); // block_len
            }
            let params = GpuParams {
                total_partials: total_partials_real,
                qhead: 1,
                head_dim: HEAD_DIM,
                block_size: BLOCK_SIZE,
                page_size: PAGE_SIZE,
                pages_per_block: 1,
                hkv: 1,
                hq: 1,
                msb: 1,
                max_pages: 4,
                softmax_scale: 0.5,
                _pad: 0,
            };

            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[0.1f32; 16]),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let k_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[0.2f32; 64]),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[0.3f32; 64]),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let edge_meta_bytes: &[i32] = if edge_meta.is_empty() {
                &[0i32; 5]
            } else {
                &edge_meta
            };
            let edge_meta_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(edge_meta_bytes),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let block_tables_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[-1i32; 16]),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let o_len = ((total_partials_real.max(1) * HEAD_DIM) as usize).max(1);
            let m_len = total_partials_real.max(1) as usize;
            let l_len = total_partials_real.max(1) as usize;

            let run = |pipeline: &wgpu::ComputePipeline| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
                let o_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (o_len * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let m_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (m_len * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let l_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (l_len * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let layout = pipeline.get_bind_group_layout(0);
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layout,
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
                let o_stage = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (o_len * 4) as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let m_stage = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (m_len * 4) as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let l_stage = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (l_len * 4) as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                {
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(dispatch_x.max(1), 1, 1);
                }
                encoder.copy_buffer_to_buffer(&o_buf, 0, &o_stage, 0, (o_len * 4) as u64);
                encoder.copy_buffer_to_buffer(&m_buf, 0, &m_stage, 0, (m_len * 4) as u64);
                encoder.copy_buffer_to_buffer(&l_buf, 0, &l_stage, 0, (l_len * 4) as u64);
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::Maintain::Wait);
                (
                    read_f32_buffer(&device, &o_stage, o_len).unwrap(),
                    read_f32_buffer(&device, &m_stage, m_len).unwrap(),
                    read_f32_buffer(&device, &l_stage, l_len).unwrap(),
                )
            };

            let (old_o, old_m, old_l) = run(&old_pipeline);
            let (new_o, new_m, new_l) = run(&gpu.pipeline);

            // `page_id` is always -1 here (never dereferenced as valid) and
            // every other input is plain deterministic f32 (no fp16/
            // quantized decode anywhere in this shader), so finite is
            // asserted outright, not counted.
            for (side, out) in [
                ("old o_partial", &old_o),
                ("new o_partial", &new_o),
                ("old m_partial", &old_m),
                ("new m_partial", &new_m),
                ("old l_partial", &old_l),
                ("new l_partial", &new_l),
            ] {
                for (i, v) in out.iter().enumerate() {
                    assert!(
                        v.is_finite(),
                        "total_partials_real={total_partials_real} block_len={block_len} {side}[{i}]={v}: expected finite"
                    );
                }
            }
            let expected_o_len = (total_partials_real.max(1) * HEAD_DIM) as usize;
            let expected_ml_len = total_partials_real.max(1) as usize;
            assert_eq!(old_o.len(), expected_o_len, "o_partial length mismatch");
            assert_eq!(old_m.len(), expected_ml_len, "m_partial length mismatch");
            assert_eq!(old_l.len(), expected_ml_len, "l_partial length mismatch");
            assert!(
                expected_o_len > 0 && expected_ml_len > 0,
                "total_partials_real={total_partials_real}: compared nothing"
            );

            let bits_mismatch = |a: &[f32], b: &[f32]| -> usize {
                a.iter()
                    .zip(b.iter())
                    .filter(|(x, y)| x.to_bits() != y.to_bits())
                    .count()
            };
            let mismatched = bits_mismatch(&old_o, &new_o)
                + bits_mismatch(&old_m, &new_m)
                + bits_mismatch(&old_l, &new_l);
            let compared = old_o.len() + old_m.len() + old_l.len();
            table.push((total_partials_real, block_len, compared, mismatched));
            assert_eq!(
                mismatched, 0,
                "total_partials_real={total_partials_real} block_len={block_len} dispatch_x={dispatch_x}: \
                 {mismatched}/{compared} values not bit-identical \
                 old=(o={old_o:?} m={old_m:?} l={old_l:?}) new=(o={new_o:?} m={new_m:?} l={new_l:?})"
            );
        }

        eprintln!("=== sparse_attention_forward old-vs-new oracle: (total_partials, block_len) x size table ===");
        for (total_partials_real, block_len, compared, mismatched) in &table {
            eprintln!(
                "total_partials={total_partials_real:>4} block_len={block_len:>4} compared={compared:>6} mismatched={mismatched:>6}"
            );
        }
        assert!(!table.is_empty(), "compared nothing");
    }

    /// New-kernel-vs-CPU-reference at large sizes (the ruling's "129 class"),
    /// safe on every backend including WARP: `kvouter_forward_gpu` always
    /// compiles from the CURRENT `sparse_attention_forward.wgsl` (this
    /// module's `SHADER_WGSL` const), never the pre-rewrite one, so unlike
    /// `old_vs_new_oracle::sparse_attention_forward_old_vs_new_bit_identical`
    /// this test has nothing WARP-unsafe to exclude — it's the same shape as
    /// the existing small-scale `gpu_forward_matches_cpu_forward_small`
    /// above, just scaled up.
    ///
    /// Tolerance, not bitwise: this involves the same softmax +
    /// sequential-vs-parallel-reduction mismatch as `attention`'s CPU
    /// reference above, so exact equality is not the right bar. The bound
    /// is pinned to 10x the worst relative error actually measured on this
    /// machine (Metal) at tq=129/head_dim=37:
    /// `1.231e-4 * 10 = 1.231e-3`.
    ///
    /// The legitimate in-range `block_len == 0` case (the removed
    /// early-bail-and-write-sentinel block's former special case) is
    /// deliberately NOT exercised here: tried `used_kv_lens = 0` for a whole
    /// batch (the index builder emits zero edges for that batch — a
    /// different, index-level case, never reaching the shader at all) and
    /// `used_kv_lens` less than one block (every edge that produced still
    /// came back with a nonzero `block_len` after `build_kvouter_index`'s
    /// own filtering). Neither reaches the shader's `block_len == 0` path
    /// through this public `build_kvouter_index`/`kvouter_forward_gpu`
    /// pipeline — it is covered instead, directly, by
    /// `old_vs_new_oracle::sparse_attention_forward_old_vs_new_bit_identical`,
    /// which builds `edge_meta` by hand specifically to force that value
    /// regardless of what the index builder would ever produce.
    #[test]
    #[expect(
        clippy::disallowed_methods,
        reason = "G1 reference: test oracle or example code, not Law output; each call is listed in scripts/g1_reference_sites.tsv"
    )]
    fn gpu_forward_matches_cpu_forward_large() {
        let Some(gpu) = setup_gpu() else {
            return;
        };

        let tq = 129; // matches the ruling's "129" size class
        let hq = 1;
        let hkv = 1;
        let head_dim = 37; // matches the ruling's "37" size class
        let page_size = 4;
        let block_size = 8; // 2 pages/block
        let max_pages = 6; // 3 blocks' worth of pages per batch
        let num_pages = max_pages + 1; // +1 so page index 0 is never page_slot's only option

        let q: Vec<f32> = (0..tq * hq * head_dim)
            .map(|i| (i as f32 * 0.011).sin())
            .collect();
        let k: Vec<f32> = (0..num_pages * hkv * page_size * head_dim)
            .map(|i| (i as f32 * 0.007).sin())
            .collect();
        let v: Vec<f32> = (0..num_pages * hkv * page_size * head_dim)
            .map(|i| (i as f32 * 0.013).cos())
            .collect();

        let tbl = BlockTables::new((0..max_pages as i32).collect(), 1, max_pages).unwrap();
        let cu = CuSeqlensQ::new(vec![0, tq as i64]).unwrap();
        // `used = max_pages * page_size` -> every page fully valid (this is
        // the size/throughput leg; see the function doc for why the
        // degenerate `block_len == 0` case is not attempted here).
        let used = vec![(max_pages * page_size) as i32];
        let pages_per_block = block_size / page_size;
        let msb = max_pages / pages_per_block; // max selected *blocks* (not pages)
        let mut sel_flat = Vec::new();
        for _i in 0..tq {
            for _h in 0..hkv {
                for r in 0..msb {
                    sel_flat.push(r as i32);
                }
            }
        }
        let sel = SparseSelection::new(sel_flat, tq, hkv, msb).unwrap();
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

        const MEASURED_WORST_REL: f32 = 1.231e-4; // Metal, tq=129/head_dim=37
        const TOL: f32 = MEASURED_WORST_REL * 10.0;

        assert_eq!(cpu.o_partial.len(), gpu_out.o_partial.len());
        assert!(!cpu.o_partial.is_empty(), "compared nothing (o_partial)");
        let mut worst_rel: f32 = 0.0;
        for (c, g) in cpu.o_partial.iter().zip(&gpu_out.o_partial) {
            assert!(c.is_finite(), "cpu reference o_partial not finite: {c}");
            assert!(g.is_finite(), "gpu o_partial not finite: {g}");
            let rel = (c - g).abs() / c.abs().max(1e-6);
            worst_rel = worst_rel.max(rel);
            assert!(
                rel < TOL,
                "o_partial mismatch: cpu={c}, gpu={g}, rel={rel}, tol={TOL}"
            );
        }
        for (c, g) in cpu.m_partial.iter().zip(&gpu_out.m_partial) {
            assert!(c.is_finite(), "cpu reference m_partial not finite: {c}");
            assert!(g.is_finite(), "gpu m_partial not finite: {g}");
            assert!(
                (c - g).abs() < TOL,
                "m_partial mismatch: cpu={c}, gpu={g}, tol={TOL}"
            );
        }
        for (c, g) in cpu.l_partial.iter().zip(&gpu_out.l_partial) {
            assert!(c.is_finite(), "cpu reference l_partial not finite: {c}");
            assert!(g.is_finite(), "gpu l_partial not finite: {g}");
            let rel = (c - g).abs() / c.abs().max(1e-6);
            assert!(
                rel < TOL,
                "l_partial mismatch: cpu={c}, gpu={g}, rel={rel}, tol={TOL}"
            );
        }
        eprintln!(
            "=== sparse_attention_forward large new-vs-CPU-reference: tq={tq} head_dim={head_dim} \
             partials={} worst_rel={worst_rel} (tol {TOL}) ===",
            cpu.o_partial.len() / head_dim
        );
    }
}
