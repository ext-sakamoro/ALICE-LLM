//! LSE (log-sum-exp) combine: merge per-`(query, rank)` forward partials
//! into final attention outputs.
//!
//! Standard FlashAttention combine: given `k` partials `(m_i, l_i, o_i)`
//! where `o_i = Σ exp(s - m_i) · V` (unnormalized), we form
//!
//! ```text
//! M      = max_i m_i
//! l_out  = Σ  exp(m_i - M) · l_i
//! o_out  = Σ  exp(m_i - M) · o_i   /  l_out
//! ```
//!
//! Queries with zero valid partials receive an all-zero output row.

use super::forward::ForwardPartials;
use super::simd;
use super::types::{KvOuterIndex, SparseAttentionError};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// Public API
// ---------------------------------------------------------------------------

/// LSE-combine the forward partials into `[Tq, Hq, head_dim]` outputs.
///
/// * `partials` — from [`kvouter_forward`].
/// * `idx` — the same CSR index consumed by the forward.
/// * `hq` — total query heads (`hkv * qhead`).
///
/// Returns a flat `[Tq * Hq * head_dim]` FP32 output buffer.
///
/// [`kvouter_forward`]: super::forward::kvouter_forward
pub fn lse_combine(
    partials: &ForwardPartials,
    idx: &KvOuterIndex,
    hq: usize,
) -> Result<Vec<f32>, SparseAttentionError> {
    if hq == 0 || !hq.is_multiple_of(idx.hkv) {
        return Err(SparseAttentionError::HeadCountMismatch { hq, hkv: idx.hkv });
    }
    let qhead_expected = hq / idx.hkv;
    if partials.qhead != qhead_expected {
        return Err(SparseAttentionError::ShapeMismatch {
            what: "partials.qhead vs hq/hkv",
            expected: qhead_expected,
            got: partials.qhead,
        });
    }
    let head_dim = partials.head_dim;
    let tq = idx.tq;
    let topk = idx.topk;
    let qhead = partials.qhead;
    let hkv = idx.hkv;

    let mut out = vec![0.0f32; tq * hq * head_dim];

    // Combine walks per-query rows (`i`) independently: each row writes into a
    // disjoint `hq * head_dim` slice of `out`, so we parallelize over `i` by
    // chunking the output buffer into `hq * head_dim`-sized rows.
    let row_stride = hq * head_dim;

    // Closure: fill a single output row `i`.
    let fill_row = |i: usize, row: &mut [f32]| {
        for h in 0..hkv {
            for qh in 0..qhead {
                let hq_abs = h * qhead + qh;
                // First pass: find M = max m_i across valid partials.
                let mut m_final = f32::NEG_INFINITY;
                for r in 0..topk {
                    let inv_local = idx.inv_of(h, i, r);
                    if inv_local < 0 {
                        continue;
                    }
                    let abs_edge = idx.edge_head_base[h] + inv_local as usize;
                    let p_idx = partials.partial_index(abs_edge, qh);
                    let m_p = partials.m_partial[p_idx];
                    if m_p > m_final {
                        m_final = m_p;
                    }
                }
                if !m_final.is_finite() {
                    // No valid partials → leave zero output for this head lane.
                    continue;
                }
                // Second pass: accumulate l_final and unnormalized o_final.
                let mut l_final = 0.0f32;
                let row_off = hq_abs * head_dim;
                for r in 0..topk {
                    let inv_local = idx.inv_of(h, i, r);
                    if inv_local < 0 {
                        continue;
                    }
                    let abs_edge = idx.edge_head_base[h] + inv_local as usize;
                    let p_idx = partials.partial_index(abs_edge, qh);
                    let m_p = partials.m_partial[p_idx];
                    let l_p = partials.l_partial[p_idx];
                    if !m_p.is_finite() {
                        continue;
                    }
                    let scale = (m_p - m_final).exp();
                    l_final += scale * l_p;
                    let o_off = p_idx * head_dim;
                    simd::axpy(
                        &mut row[row_off..row_off + head_dim],
                        scale,
                        &partials.o_partial[o_off..o_off + head_dim],
                    );
                }
                if l_final > 0.0 {
                    let inv_l = 1.0 / l_final;
                    simd::scale_in_place(&mut row[row_off..row_off + head_dim], inv_l);
                }
            }
        }
    };

    #[cfg(feature = "parallel")]
    out.par_chunks_mut(row_stride)
        .enumerate()
        .for_each(|(i, row)| fill_row(i, row));
    #[cfg(not(feature = "parallel"))]
    for (i, row) in out.chunks_mut(row_stride).enumerate() {
        fill_row(i, row);
    }

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

    /// Reference dense scaled-dot-product attention over a paged K/V cache
    /// with `used_kv_lens` and optional right-aligned causal mask. Used as
    /// the ground truth for the sparse pipeline when every KV block is
    /// selected.
    #[allow(clippy::too_many_arguments)]
    fn dense_reference(
        q: &[f32],
        k_pages: &[f32],
        v_pages: &[f32],
        block_tables: &BlockTables,
        cu_seqlens_q: &CuSeqlensQ,
        used_kv_lens: &[i32],
        hq: usize,
        hkv: usize,
        head_dim: usize,
        page_size: usize,
        causal: bool,
    ) -> Vec<f32> {
        let tq = cu_seqlens_q.total_tq();
        let qhead = hq / hkv;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let page_stride = hkv * page_size * head_dim;
        let mut out = vec![0.0f32; tq * hq * head_dim];

        let batch_size = cu_seqlens_q.batch_size();
        // Precompute batch id + local q pos per packed query.
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

        for i in 0..tq {
            let b = batch_of_tq[i];
            let seq_len_b = used_kv_lens[b].max(0) as usize;
            let b_q_start = cu_seqlens_q.prefix[b] as usize;
            let b_q_end = cu_seqlens_q.prefix[b + 1] as usize;
            let qo_len_b = b_q_end - b_q_start;

            for h in 0..hkv {
                for qh in 0..qhead {
                    let hq_abs = h * qhead + qh;
                    let q_off = (i * hq + hq_abs) * head_dim;
                    let q_vec = &q[q_off..q_off + head_dim];

                    // Gather every KV position ≤ causal limit.
                    let causal_limit = if causal {
                        seq_len_b.saturating_sub(qo_len_b) + q_local_pos[i]
                    } else {
                        seq_len_b.saturating_sub(1)
                    };
                    let effective_len = seq_len_b.min(causal_limit + 1);

                    let mut scores = Vec::with_capacity(effective_len);
                    let mut m = f32::NEG_INFINITY;
                    // K collected as `positions × head_dim`.
                    let mut k_flat = Vec::with_capacity(effective_len * head_dim);
                    let mut v_flat = Vec::with_capacity(effective_len * head_dim);

                    for kv_pos in 0..effective_len {
                        let page_local = kv_pos / page_size;
                        let pos_in_page = kv_pos % page_size;
                        let page_id = block_tables.get(b, page_local);
                        if page_id < 0 {
                            scores.push(f32::NEG_INFINITY);
                            k_flat.extend(std::iter::repeat_n(0.0, head_dim));
                            v_flat.extend(std::iter::repeat_n(0.0, head_dim));
                            continue;
                        }
                        let base_k = page_id as usize * page_stride + h * page_size * head_dim;
                        let k_row = &k_pages[base_k + pos_in_page * head_dim
                            ..base_k + (pos_in_page + 1) * head_dim];
                        let v_row = &v_pages[base_k + pos_in_page * head_dim
                            ..base_k + (pos_in_page + 1) * head_dim];
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_vec[d] * k_row[d];
                        }
                        let s = dot * scale;
                        if s > m {
                            m = s;
                        }
                        scores.push(s);
                        k_flat.extend_from_slice(k_row);
                        v_flat.extend_from_slice(v_row);
                    }

                    if !m.is_finite() {
                        continue;
                    }
                    let mut l = 0.0f32;
                    let mut o = vec![0.0f32; head_dim];
                    for (pos, &s) in scores.iter().enumerate() {
                        if !s.is_finite() {
                            continue;
                        }
                        let p = (s - m).exp();
                        l += p;
                        let v_row = &v_flat[pos * head_dim..(pos + 1) * head_dim];
                        for d in 0..head_dim {
                            o[d] += p * v_row[d];
                        }
                    }
                    for d in 0..head_dim {
                        out[q_off + d] = o[d] / l;
                    }
                }
            }
        }
        out
    }

    fn make_selection_all_blocks(tq: usize, hkv: usize, msb: usize) -> SparseSelection {
        // topk = msb, every rank r picks block r.
        let topk = msb;
        let mut sel = Vec::with_capacity(tq * hkv * topk);
        for _i in 0..tq {
            for _h in 0..hkv {
                for r in 0..topk {
                    sel.push(r as i32);
                }
            }
        }
        SparseSelection::new(sel, tq, hkv, topk).unwrap()
    }

    #[test]
    fn sparse_full_matches_dense_reference_no_causal() {
        // Tq=2 (single batch), Hq=Hkv=1, head_dim=4, page_size=block_size=2,
        // 3 pages populated (used_kv_lens=6).
        let tq = 2;
        let hq = 1;
        let hkv = 1;
        let head_dim = 4;
        let page_size = 2;
        let block_size = 2;
        let max_pages = 3;
        let num_pages = 3;

        // Deterministic small values.
        let q: Vec<f32> = (0..tq * hq * head_dim).map(|i| (i as f32) * 0.11).collect();
        let k: Vec<f32> = (0..num_pages * hkv * page_size * head_dim)
            .map(|i| ((i as f32) * 0.07).sin())
            .collect();
        let v: Vec<f32> = (0..num_pages * hkv * page_size * head_dim)
            .map(|i| ((i as f32) * 0.13).cos())
            .collect();

        let tbl = BlockTables::new(vec![0, 1, 2], 1, max_pages).unwrap();
        let cu = CuSeqlensQ::new(vec![0, tq as i64]).unwrap();
        let used = vec![6i32]; // full 3 blocks × 2 = 6 kv tokens.
        let sel = make_selection_all_blocks(tq, hkv, max_pages);
        let idx = build_kvouter_index(&sel, &tbl, &cu, Some(&used), block_size, page_size).unwrap();

        let softmax_scale = 1.0 / (head_dim as f32).sqrt();
        let partials = kvouter_forward(
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

        let sparse_out = lse_combine(&partials, &idx, hq).unwrap();
        let dense_out = dense_reference(
            &q, &k, &v, &tbl, &cu, &used, hq, hkv, head_dim, page_size, false,
        );

        for (i, (s, d)) in sparse_out.iter().zip(dense_out.iter()).enumerate() {
            let diff = (s - d).abs();
            let rel = diff / d.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "mismatch at {i}: sparse={s}, dense={d}, rel={rel}"
            );
        }
    }

    #[test]
    fn sparse_full_matches_dense_reference_gqa() {
        // GQA: Hq=4, Hkv=2, qhead=2.
        let tq = 3;
        let hq = 4;
        let hkv = 2;
        let head_dim = 4;
        let page_size = 4;
        let block_size = 4;
        let max_pages = 3;
        let num_pages = 3;

        let q: Vec<f32> = (0..tq * hq * head_dim)
            .map(|i| ((i as f32) * 0.09).sin())
            .collect();
        let k: Vec<f32> = (0..num_pages * hkv * page_size * head_dim)
            .map(|i| ((i as f32) * 0.05).cos())
            .collect();
        let v: Vec<f32> = (0..num_pages * hkv * page_size * head_dim)
            .map(|i| ((i as f32) * 0.03).sin())
            .collect();

        let tbl = BlockTables::new(vec![0, 1, 2], 1, max_pages).unwrap();
        let cu = CuSeqlensQ::new(vec![0, tq as i64]).unwrap();
        let used = vec![12i32]; // 3 blocks × 4 = 12.
        let sel = make_selection_all_blocks(tq, hkv, max_pages);
        let idx = build_kvouter_index(&sel, &tbl, &cu, Some(&used), block_size, page_size).unwrap();

        let softmax_scale = 1.0 / (head_dim as f32).sqrt();
        let partials = kvouter_forward(
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
        let sparse_out = lse_combine(&partials, &idx, hq).unwrap();
        let dense_out = dense_reference(
            &q, &k, &v, &tbl, &cu, &used, hq, hkv, head_dim, page_size, false,
        );

        for (i, (s, d)) in sparse_out.iter().zip(dense_out.iter()).enumerate() {
            let diff = (s - d).abs();
            let rel = diff / d.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "gqa mismatch at {i}: sparse={s}, dense={d}, rel={rel}"
            );
        }
    }

    #[test]
    fn sparse_full_matches_dense_reference_causal() {
        // Same shape as the no-causal test but with `causal=true` and Tq=3
        // so multiple queries in a single batch exercise the right-aligned
        // causal mask.
        let tq = 3;
        let hq = 2;
        let hkv = 1;
        let head_dim = 4;
        let page_size = 2;
        let block_size = 2;
        let max_pages = 3;
        let num_pages = 3;

        let q: Vec<f32> = (0..tq * hq * head_dim)
            .map(|i| ((i as f32) * 0.17).cos())
            .collect();
        let k: Vec<f32> = (0..num_pages * hkv * page_size * head_dim)
            .map(|i| ((i as f32) * 0.11).sin())
            .collect();
        let v: Vec<f32> = (0..num_pages * hkv * page_size * head_dim)
            .map(|i| ((i as f32) * 0.19).cos())
            .collect();
        let tbl = BlockTables::new(vec![0, 1, 2], 1, max_pages).unwrap();
        let cu = CuSeqlensQ::new(vec![0, tq as i64]).unwrap();
        let used = vec![6i32];
        let sel = make_selection_all_blocks(tq, hkv, max_pages);
        let idx = build_kvouter_index(&sel, &tbl, &cu, Some(&used), block_size, page_size).unwrap();

        let softmax_scale = 1.0 / (head_dim as f32).sqrt();
        let partials = kvouter_forward(
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
            true,
        )
        .unwrap();
        let sparse_out = lse_combine(&partials, &idx, hq).unwrap();
        let dense_out = dense_reference(
            &q, &k, &v, &tbl, &cu, &used, hq, hkv, head_dim, page_size, true,
        );

        for (i, (s, d)) in sparse_out.iter().zip(dense_out.iter()).enumerate() {
            let diff = (s - d).abs();
            let rel = diff / d.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "causal mismatch at {i}: sparse={s}, dense={d}, rel={rel}"
            );
        }
    }

    #[test]
    fn zero_valid_partials_yields_zero_output_row() {
        // Selection is all -1 → no partials → combined output must be all zero.
        let tq = 1;
        let hq = 1;
        let hkv = 1;
        let head_dim = 2;
        let sel = SparseSelection::new(vec![-1, -1], tq, hkv, 2).unwrap();
        let tbl = BlockTables::new(vec![0, 1], 1, 2).unwrap();
        let cu = CuSeqlensQ::new(vec![0, tq as i64]).unwrap();
        let idx = build_kvouter_index(&sel, &tbl, &cu, None, 2, 2).unwrap();

        let q = vec![1.0; tq * hq * head_dim];
        let k = vec![1.0; 2 * hkv * 2 * head_dim];
        let v = vec![1.0; 2 * hkv * 2 * head_dim];
        let softmax_scale = 1.0f32 / (head_dim as f32).sqrt();
        let partials = kvouter_forward(
            &q,
            &k,
            &v,
            &idx,
            &tbl,
            &cu,
            None,
            hq,
            hkv,
            head_dim,
            2,
            2,
            softmax_scale,
            false,
        )
        .unwrap();
        let out = lse_combine(&partials, &idx, hq).unwrap();
        assert!(out.iter().all(|&x| x == 0.0));
    }
}
