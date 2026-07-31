//! KV-outer forward pass: emit unnormalized partials + online-softmax stats.
//!
//! For each `(kv_head, compact_slot)` work unit we load the selected KV block
//! once and compute attention for every `(query, selected_rank, qhead_lane)`
//! edge that pointed at it. This mirrors the persistent SM100 forward
//! kernel's outer loop (see `fw-ai/minimax-kernels/docs/m3-sparse-attention.md`
//! § "KV-outer forward").
//!
//! With the `parallel` feature enabled the outer loop over work units is
//! walked with Rayon. Each unit writes into a disjoint slice of the flat
//! partial buffers (guaranteed by the CSR ordering of `enumerate_work_units`),
//! so the parallel version stays entirely inside safe Rust — we split
//! `o_partial` / `m_partial` / `l_partial` into per-unit sub-slices up front
//! and hand each sub-slice to a task.
//!
//! Partials are emitted in *head-local* order matching
//! `KvOuterIndex::edges`. The combine pass then merges them with the
//! standard FlashAttention log-sum-exp update.

use super::scheduler::{enumerate_work_units, WorkSplit};
use super::types::{BlockTables, CuSeqlensQ, KvOuterIndex, SparseAttentionError};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// Types
// ---------------------------------------------------------------------------

/// Forward-pass partial outputs + online-softmax bookkeeping.
///
/// Layout is `[total_edges, qhead, ...]` where `total_edges = idx.edges.len()`
/// and `qhead = Hq / Hkv`.
#[derive(Debug, Clone)]
pub struct ForwardPartials {
    /// `[total_edges * qhead, head_dim]` — unnormalized `Σ exp(s-m) · V`.
    pub o_partial: Vec<f32>,
    /// `[total_edges * qhead]` — row max `m`.
    pub m_partial: Vec<f32>,
    /// `[total_edges * qhead]` — softmax sum `Σ exp(s-m)`.
    pub l_partial: Vec<f32>,
    /// Convenience mirror of `idx.edges.len()`.
    pub total_edges: usize,
    /// GQA group size (`Hq / Hkv`).
    pub qhead: usize,
    /// Head dimension.
    pub head_dim: usize,
}

impl ForwardPartials {
    /// Position of the `(abs_edge, qh)` partial in the flat buffers.
    #[inline]
    #[must_use]
    pub fn partial_index(&self, abs_edge: usize, qh: usize) -> usize {
        abs_edge * self.qhead + qh
    }
}

// Shared context for a single work-unit computation
// ---------------------------------------------------------------------------

/// Read-only inputs passed to each work-unit call.
struct WorkCtx<'a> {
    q: &'a [f32],
    k_pages: &'a [f32],
    v_pages: &'a [f32],
    idx: &'a KvOuterIndex,
    block_tables: &'a BlockTables,
    cu_seqlens_q: &'a CuSeqlensQ,
    used_kv_lens: Option<&'a [i32]>,
    batch_of_tq: &'a [usize],
    q_local_pos: &'a [usize],
    hq: usize,
    qhead: usize,
    head_dim: usize,
    block_size: usize,
    page_size: usize,
    page_stride: usize,
    num_pages: usize,
    softmax_scale: f32,
    causal: bool,
}

// Public API
// ---------------------------------------------------------------------------

/// KV-outer forward pass.
///
/// * `q` — `[Tq, Hq, head_dim]` packed queries.
/// * `k_pages`, `v_pages` — `[num_pages, Hkv, page_size, head_dim]` paged
///   K / V caches.
/// * `idx` — CSR inverse index from [`build_kvouter_index`].
/// * `block_tables`, `cu_seqlens_q`, `used_kv_lens` — as in
///   [`kvouter_attention`]'s public contract.
/// * `hq`, `hkv`, `head_dim`, `block_size`, `page_size` — geometry
///   constants; must satisfy `hq % hkv == 0` and
///   `block_size % page_size == 0`.
/// * `softmax_scale` — usually `1.0 / sqrt(head_dim)`.
/// * `causal` — mask out `k > q` positions inside the block. Uses the
///   right-aligned per-sequence causal position:
///   `q_seq_pos = q_idx - cu_seqlens_q.prefix[b]`,
///   `kv_seq_pos_limit = used_kv_lens[b] - qo_len_b + q_seq_pos`.
///   With no `used_kv_lens`, we fall back to `q_seq_pos`.
///
/// [`build_kvouter_index`]: super::build_kvouter_index
/// [`kvouter_attention`]: super::kvouter_attention
pub fn kvouter_forward(
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
    causal: bool,
) -> Result<ForwardPartials, SparseAttentionError> {
    // Argument sanity.
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
    let num_pages = k_pages.len() / page_stride;

    // Precompute per-query batch + local query position (needed for causal).
    let batch_size = cu_seqlens_q.batch_size();
    let mut batch_of_tq: Vec<usize> = vec![0; tq];
    let mut q_local_pos: Vec<usize> = vec![0; tq];
    {
        let mut b = 0usize;
        for i in 0..tq {
            while b + 1 < batch_size && (i as i64) >= cu_seqlens_q.prefix[b + 1] {
                b += 1;
            }
            batch_of_tq[i] = b;
            q_local_pos[i] = i - cu_seqlens_q.prefix[b] as usize;
        }
    }

    let total_edges = idx.edges.len();
    let mut o_partial = vec![0.0f32; total_edges * qhead * head_dim];
    let mut m_partial = vec![f32::NEG_INFINITY; total_edges * qhead];
    let mut l_partial = vec![0.0f32; total_edges * qhead];

    let work_units = enumerate_work_units(idx);

    // --- Pre-split partial buffers into disjoint per-unit slices. --------
    // `enumerate_work_units` walks edges in CSR order, so consecutive units
    // occupy consecutive ranges in the flat buffers → we can split with
    // `split_at_mut` linearly.
    let mut o_rest: &mut [f32] = o_partial.as_mut_slice();
    let mut m_rest: &mut [f32] = m_partial.as_mut_slice();
    let mut l_rest: &mut [f32] = l_partial.as_mut_slice();
    let mut tasks: Vec<(WorkSplit, &mut [f32], &mut [f32], &mut [f32])> =
        Vec::with_capacity(work_units.len());
    for unit in &work_units {
        let n = unit.edge_end - unit.edge_start;
        let (o_head, o_tail) = std::mem::take(&mut o_rest).split_at_mut(n * qhead * head_dim);
        let (m_head, m_tail) = std::mem::take(&mut m_rest).split_at_mut(n * qhead);
        let (l_head, l_tail) = std::mem::take(&mut l_rest).split_at_mut(n * qhead);
        tasks.push((*unit, o_head, m_head, l_head));
        o_rest = o_tail;
        m_rest = m_tail;
        l_rest = l_tail;
    }

    let ctx = WorkCtx {
        q,
        k_pages,
        v_pages,
        idx,
        block_tables,
        cu_seqlens_q,
        used_kv_lens,
        batch_of_tq: &batch_of_tq,
        q_local_pos: &q_local_pos,
        hq,
        qhead,
        head_dim,
        block_size,
        page_size,
        page_stride,
        num_pages,
        softmax_scale,
        causal,
    };

    // --- Walk work units. Parallelized under `parallel` feature. ---------
    #[cfg(feature = "parallel")]
    let result: Result<(), SparseAttentionError> =
        tasks
            .par_iter_mut()
            .try_for_each(|(unit, o_slice, m_slice, l_slice)| {
                process_work_unit(&ctx, unit, o_slice, m_slice, l_slice)
            });
    #[cfg(not(feature = "parallel"))]
    let result: Result<(), SparseAttentionError> =
        tasks
            .iter_mut()
            .try_for_each(|(unit, o_slice, m_slice, l_slice)| {
                process_work_unit(&ctx, unit, o_slice, m_slice, l_slice)
            });
    result?;

    Ok(ForwardPartials {
        o_partial,
        m_partial,
        l_partial,
        total_edges,
        qhead,
        head_dim,
    })
}

// Per-unit worker
// ---------------------------------------------------------------------------

fn process_work_unit(
    ctx: &WorkCtx<'_>,
    unit: &WorkSplit,
    o_slice: &mut [f32],
    m_slice: &mut [f32],
    l_slice: &mut [f32],
) -> Result<(), SparseAttentionError> {
    let h = unit.head;
    let j = unit.compact_slot;
    let raw_slot = ctx.idx.raw_slot(h, j) as usize;
    let b = raw_slot / ctx.idx.msb;
    let blk = raw_slot % ctx.idx.msb;

    let lk = ctx
        .used_kv_lens
        .map(|u| u[b].max(0) as usize)
        .unwrap_or(ctx.idx.msb * ctx.block_size);
    let block_start = blk * ctx.block_size;
    let block_end = ((blk + 1) * ctx.block_size).min(lk);
    if block_end <= block_start {
        return Ok(());
    }
    let seq_len_b = lk;
    let b_q_start = ctx.cu_seqlens_q.prefix[b] as usize;
    let b_q_end = ctx.cu_seqlens_q.prefix[b + 1] as usize;
    let qo_len_b = b_q_end - b_q_start;

    // Gather this KV block (K, V) into contiguous scratch buffers.
    let mut k_block: Vec<f32> = Vec::with_capacity(ctx.block_size * ctx.head_dim);
    let mut v_block: Vec<f32> = Vec::with_capacity(ctx.block_size * ctx.head_dim);
    let mut block_positions: Vec<usize> = Vec::with_capacity(ctx.block_size);
    for page_local in 0..ctx.idx.pages_per_block {
        let page_seq_start = blk * ctx.block_size + page_local * ctx.page_size;
        let page_seq_end = page_seq_start + ctx.page_size;
        let page_kv_start = block_start.max(page_seq_start);
        let page_kv_end = block_end.min(page_seq_end);
        if page_kv_end <= page_kv_start {
            continue;
        }
        let page_slot = blk * ctx.idx.pages_per_block + page_local;
        let page_id = ctx.block_tables.get(b, page_slot);
        if page_id < 0 {
            continue;
        }
        let page_id_u = page_id as usize;
        if page_id_u >= ctx.num_pages {
            return Err(SparseAttentionError::ShapeMismatch {
                what: "block_tables page id",
                expected: ctx.num_pages,
                got: page_id_u,
            });
        }
        let base_k = page_id_u * ctx.page_stride + h * ctx.page_size * ctx.head_dim;
        let base_v = base_k;
        let pos_start = page_kv_start - page_seq_start;
        let pos_end = page_kv_end - page_seq_start;
        for pos in pos_start..pos_end {
            let k_off = base_k + pos * ctx.head_dim;
            let v_off = base_v + pos * ctx.head_dim;
            k_block.extend_from_slice(&ctx.k_pages[k_off..k_off + ctx.head_dim]);
            v_block.extend_from_slice(&ctx.v_pages[v_off..v_off + ctx.head_dim]);
            block_positions.push(page_seq_start + pos);
        }
    }
    let block_len = block_positions.len();
    if block_len == 0 {
        return Ok(());
    }

    // Iterate over edges owned by this unit. `unit.edge_start` is absolute;
    // per-slice offsets are edge-local (edge_local = abs_edge - unit.edge_start).
    let mut scores: Vec<f32> = Vec::with_capacity(block_len);
    for local_edge in 0..(unit.edge_end - unit.edge_start) {
        let abs_edge = unit.edge_start + local_edge;
        let (q_idx_i32, _rank) = ctx.idx.edges[abs_edge];
        let q_idx = q_idx_i32 as usize;
        let causal_limit = if ctx.causal {
            let q_local = ctx.q_local_pos[q_idx];
            if seq_len_b >= qo_len_b {
                Some(seq_len_b - qo_len_b + q_local)
            } else {
                Some(q_local)
            }
        } else {
            None
        };

        for qh in 0..ctx.qhead {
            let hq_abs = h * ctx.qhead + qh;
            let q_base = (q_idx * ctx.hq + hq_abs) * ctx.head_dim;
            let q_vec = &ctx.q[q_base..q_base + ctx.head_dim];

            scores.clear();
            let mut m = f32::NEG_INFINITY;
            for (pos, &kv_pos_abs) in block_positions.iter().enumerate() {
                let alive = causal_limit.map_or(true, |limit| kv_pos_abs <= limit);
                let s = if alive {
                    let k_off = pos * ctx.head_dim;
                    let k_row = &k_block[k_off..k_off + ctx.head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..ctx.head_dim {
                        dot += q_vec[d] * k_row[d];
                    }
                    dot * ctx.softmax_scale
                } else {
                    f32::NEG_INFINITY
                };
                if s > m {
                    m = s;
                }
                scores.push(s);
            }

            let local_partial = local_edge * ctx.qhead + qh;
            let o_off = local_partial * ctx.head_dim;

            if !m.is_finite() {
                m_slice[local_partial] = f32::NEG_INFINITY;
                l_slice[local_partial] = 0.0;
                for d in 0..ctx.head_dim {
                    o_slice[o_off + d] = 0.0;
                }
                continue;
            }

            let mut l = 0.0f32;
            for (pos, &s) in scores.iter().enumerate() {
                let p = if s.is_finite() { (s - m).exp() } else { 0.0 };
                l += p;
                let v_off = pos * ctx.head_dim;
                let v_row = &v_block[v_off..v_off + ctx.head_dim];
                for d in 0..ctx.head_dim {
                    o_slice[o_off + d] += p * v_row[d];
                }
            }
            m_slice[local_partial] = m;
            l_slice[local_partial] = l;
        }
    }

    Ok(())
}
