//! KV-outer forward pass: emit unnormalized partials + online-softmax stats.
//!
//! For each `(kv_head, compact_slot)` work unit we load the selected KV block
//! once and compute attention for every `(query, selected_rank, qhead_lane)`
//! edge that pointed at it. This mirrors the persistent SM100 forward
//! kernel's outer loop (see `fw-ai/minimax-kernels/docs/m3-sparse-attention.md`
//! § "KV-outer forward") but runs sequentially on the CPU.
//!
//! Partials are emitted in *head-local* order matching
//! `KvOuterIndex::edges`. The combine pass then merges them with the
//! standard FlashAttention log-sum-exp update.

use super::scheduler::enumerate_work_units;
use super::types::{BlockTables, CuSeqlensQ, KvOuterIndex, SparseAttentionError};

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

    // Scratch buffers reused across work units.
    let mut scores = Vec::<f32>::with_capacity(block_size);
    let mut k_block = Vec::<f32>::with_capacity(block_size * head_dim);
    let mut v_block = Vec::<f32>::with_capacity(block_size * head_dim);

    for unit in enumerate_work_units(idx) {
        let h = unit.head;
        let j = unit.compact_slot;
        let raw_slot = idx.raw_slot(h, j) as usize;
        let b = raw_slot / idx.msb;
        let blk = raw_slot % idx.msb;

        // Determine real KV range covered by this sparse block for batch b.
        let lk = used_kv_lens
            .map(|u| u[b].max(0) as usize)
            .unwrap_or(idx.msb * block_size);
        let block_start = blk * block_size;
        let block_end = ((blk + 1) * block_size).min(lk);
        if block_end <= block_start {
            continue;
        }
        // Sequence length (per batch) — used for right-aligned causal.
        let seq_len_b = lk;
        // qo_len_b = number of packed queries in this batch.
        let b_q_start = cu_seqlens_q.prefix[b] as usize;
        let b_q_end = cu_seqlens_q.prefix[b + 1] as usize;
        let qo_len_b = b_q_end - b_q_start;

        // Gather this KV block (K, V) into contiguous scratch buffers.
        k_block.clear();
        v_block.clear();
        let mut block_positions: Vec<usize> = Vec::new(); // absolute KV pos
        for page_local in 0..idx.pages_per_block {
            let page_seq_start = blk * block_size + page_local * page_size;
            let page_seq_end = page_seq_start + page_size;
            let page_kv_start = block_start.max(page_seq_start);
            let page_kv_end = block_end.min(page_seq_end);
            if page_kv_end <= page_kv_start {
                continue;
            }
            let page_slot = blk * idx.pages_per_block + page_local;
            let page_id = block_tables.get(b, page_slot);
            if page_id < 0 {
                continue;
            }
            let page_id_u = page_id as usize;
            if page_id_u >= num_pages {
                return Err(SparseAttentionError::ShapeMismatch {
                    what: "block_tables page id",
                    expected: num_pages,
                    got: page_id_u,
                });
            }
            let base_k = page_id_u * page_stride + h * page_size * head_dim;
            let base_v = base_k; // same layout for V
            let pos_start = page_kv_start - page_seq_start;
            let pos_end = page_kv_end - page_seq_start;
            for pos in pos_start..pos_end {
                let k_off = base_k + pos * head_dim;
                let v_off = base_v + pos * head_dim;
                k_block.extend_from_slice(&k_pages[k_off..k_off + head_dim]);
                v_block.extend_from_slice(&v_pages[v_off..v_off + head_dim]);
                block_positions.push(page_seq_start + pos);
            }
        }
        let block_len = block_positions.len();
        if block_len == 0 {
            continue;
        }

        // For each edge in this slot × each qhead lane, compute attention
        // and store the partial.
        let (edge_start, edge_end) = idx.edge_range(h, j);
        for abs_edge in edge_start..edge_end {
            let (q_idx_i32, _rank) = idx.edges[abs_edge];
            let q_idx = q_idx_i32 as usize;
            // Right-aligned causal position: kv_seq_limit = seq_len_b - qo_len_b + q_local_pos
            // (matches the fw-ai convention used in tests).
            let causal_limit = if causal {
                let q_local = q_local_pos[q_idx];
                // Underflow-safe: if q_local exceeds seq_len_b - qo_len_b something is wrong upstream.
                if seq_len_b >= qo_len_b {
                    Some(seq_len_b - qo_len_b + q_local)
                } else {
                    Some(q_local)
                }
            } else {
                None
            };

            for qh in 0..qhead {
                let hq_abs = h * qhead + qh;
                let q_base = (q_idx * hq + hq_abs) * head_dim;
                let q_vec = &q[q_base..q_base + head_dim];

                // Compute raw dot scores over the block.
                scores.clear();
                let mut m = f32::NEG_INFINITY;
                for (pos, &kv_pos_abs) in block_positions.iter().enumerate() {
                    let mut alive = true;
                    if let Some(limit) = causal_limit {
                        if kv_pos_abs > limit {
                            alive = false;
                        }
                    }
                    let s = if alive {
                        let k_off = pos * head_dim;
                        let k_row = &k_block[k_off..k_off + head_dim];
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_vec[d] * k_row[d];
                        }
                        dot * softmax_scale
                    } else {
                        f32::NEG_INFINITY
                    };
                    if s > m {
                        m = s;
                    }
                    scores.push(s);
                }

                let partial_idx = abs_edge * qhead + qh;
                let o_off = partial_idx * head_dim;

                if !m.is_finite() {
                    // All positions masked. Leave l = 0, m = -inf, o = 0.
                    m_partial[partial_idx] = f32::NEG_INFINITY;
                    l_partial[partial_idx] = 0.0;
                    for d in 0..head_dim {
                        o_partial[o_off + d] = 0.0;
                    }
                    continue;
                }

                let mut l = 0.0f32;
                for (pos, &s) in scores.iter().enumerate() {
                    let p = if s.is_finite() { (s - m).exp() } else { 0.0 };
                    l += p;
                    let v_off = pos * head_dim;
                    let v_row = &v_block[v_off..v_off + head_dim];
                    for d in 0..head_dim {
                        o_partial[o_off + d] += p * v_row[d];
                    }
                }
                m_partial[partial_idx] = m;
                l_partial[partial_idx] = l;
            }
        }
    }

    Ok(ForwardPartials {
        o_partial,
        m_partial,
        l_partial,
        total_edges,
        qhead,
        head_dim,
    })
}
