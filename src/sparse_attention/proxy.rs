//! Dense proxy pass: per-block max attention score from a cheap query slice.
//!
//! The MSA algorithm chooses top-K KV blocks by scoring them cheaply with a
//! small `proxy_q` (typically one or a handful of query-head lanes) against
//! the full paged K cache. The output feeds [`sparse_topk_select`] to pick
//! blocks for the sparse attention pass.
//!
//! [`sparse_topk_select`]: super::topk::sparse_topk_select

use super::simd;
use super::types::{BlockTables, CuSeqlensQ, SparseAttentionError};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// Public API
// ---------------------------------------------------------------------------

/// Compute per-`(query, kv_head, sparse_block)` maximum attention score.
///
/// * `proxy_q` — `[Tq, Hkv, head_dim]` flat FP32. One proxy query lane per
///   KV head; upstream MSA uses a cheap slice of query heads (often 1).
/// * `k_pages` — `[num_pages, Hkv, page_size, head_dim]` flat FP32.
/// * `block_tables` — `[B, max_pages]` logical→physical page ids. `-1`
///   marks an unused page slot.
/// * `cu_seqlens_q` — packed query prefix sums.
/// * `used_kv_lens` — optional `[B]` real KV lengths. Positions past
///   `used_kv_lens[b]` are masked out.
/// * `head_dim`, `block_size`, `page_size` — geometry constants.
///
/// Returns a `[Tq, Hkv, msb]` flat FP32 buffer, where
/// `msb = max_pages / (block_size / page_size)`. Blocks that fall beyond
/// `used_kv_lens[b]` (or that address a `-1` physical page) receive
/// `f32::NEG_INFINITY` so they are naturally excluded by top-K selection.
///
/// This is a straight `O(Tq · Hkv · num_valid_positions · head_dim)` port
/// and does no SIMD / rayon parallelism — those come in Phase MSA.5.
pub fn compute_proxy_block_max_scores(
    proxy_q: &[f32],
    k_pages: &[f32],
    block_tables: &BlockTables,
    cu_seqlens_q: &CuSeqlensQ,
    used_kv_lens: Option<&[i32]>,
    hkv: usize,
    head_dim: usize,
    block_size: usize,
    page_size: usize,
) -> Result<Vec<f32>, SparseAttentionError> {
    // Argument sanity.
    let tq = cu_seqlens_q.total_tq();
    if tq == 0 || hkv == 0 || head_dim == 0 {
        return Err(SparseAttentionError::EmptyInput);
    }
    if page_size == 0 || block_size == 0 || block_size % page_size != 0 {
        return Err(SparseAttentionError::BlockPageMismatch {
            block_size,
            page_size,
        });
    }
    let pages_per_block = block_size / page_size;
    let msb = block_tables.max_pages / pages_per_block;
    if msb == 0 {
        return Err(SparseAttentionError::BlockPageMismatch {
            block_size,
            page_size,
        });
    }
    let expected_q = tq * hkv * head_dim;
    if proxy_q.len() != expected_q {
        return Err(SparseAttentionError::ShapeMismatch {
            what: "proxy_q",
            expected: expected_q,
            got: proxy_q.len(),
        });
    }
    // `num_pages` is inferred from k_pages length.
    let page_stride = hkv * page_size * head_dim;
    if k_pages.len() % page_stride != 0 {
        return Err(SparseAttentionError::ShapeMismatch {
            what: "k_pages (not a whole number of pages)",
            expected: page_stride,
            got: k_pages.len(),
        });
    }
    let num_pages = k_pages.len() / page_stride;
    if cu_seqlens_q.batch_size() != block_tables.batch_size {
        return Err(SparseAttentionError::ShapeMismatch {
            what: "cu_seqlens_q.batch_size vs block_tables.batch_size",
            expected: block_tables.batch_size,
            got: cu_seqlens_q.batch_size(),
        });
    }
    if let Some(used) = used_kv_lens {
        if used.len() != cu_seqlens_q.batch_size() {
            return Err(SparseAttentionError::ShapeMismatch {
                what: "used_kv_lens",
                expected: cu_seqlens_q.batch_size(),
                got: used.len(),
            });
        }
    }

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut out = vec![f32::NEG_INFINITY; tq * hkv * msb];

    // Precompute batch id per packed query index.
    let batch_size = cu_seqlens_q.batch_size();
    let mut batch_of_tq: Vec<usize> = vec![0; tq];
    {
        let mut b = 0usize;
        for i in 0..tq {
            while b + 1 < batch_size && (i as i64) >= cu_seqlens_q.prefix[b + 1] {
                b += 1;
            }
            batch_of_tq[i] = b;
        }
    }

    // Per-query row = `hkv * msb` scores. Filling each row is independent, so
    // parallelize by chunking `out` into rows and mapping in parallel under
    // the `parallel` feature.
    let row_stride = hkv * msb;

    let fill_row = |i: usize, row: &mut [f32]| -> Result<(), SparseAttentionError> {
        let b = batch_of_tq[i];
        let lk = used_kv_lens
            .map(|u| u[b].max(0) as usize)
            .unwrap_or(msb * block_size);
        let valid_slots = lk.div_ceil(block_size).min(msb);
        for h in 0..hkv {
            let q_base = (i * hkv + h) * head_dim;
            let q = &proxy_q[q_base..q_base + head_dim];
            for blk in 0..valid_slots {
                let mut block_max = f32::NEG_INFINITY;
                let block_start = blk * block_size;
                let block_end = ((blk + 1) * block_size).min(lk);
                if block_end <= block_start {
                    continue;
                }
                for page_local in 0..pages_per_block {
                    let page_slot = blk * pages_per_block + page_local;
                    let page_seq_start = blk * block_size + page_local * page_size;
                    let page_seq_end = page_seq_start + page_size;
                    let page_kv_start = block_start.max(page_seq_start);
                    let page_kv_end = block_end.min(page_seq_end);
                    if page_kv_end <= page_kv_start {
                        continue;
                    }
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
                    let base_page = page_id_u * page_stride + h * page_size * head_dim;
                    let pos_start = page_kv_start - page_seq_start;
                    let pos_end = page_kv_end - page_seq_start;
                    for pos in pos_start..pos_end {
                        let k_base = base_page + pos * head_dim;
                        let k = &k_pages[k_base..k_base + head_dim];
                        let s = simd::dot(q, k) * scale;
                        if s > block_max {
                            block_max = s;
                        }
                    }
                }
                row[h * msb + blk] = block_max;
            }
        }
        Ok(())
    };

    #[cfg(feature = "parallel")]
    out.par_chunks_mut(row_stride)
        .enumerate()
        .try_for_each(|(i, row)| fill_row(i, row))?;
    #[cfg(not(feature = "parallel"))]
    for (i, row) in out.chunks_mut(row_stride).enumerate() {
        fill_row(i, row)?;
    }

    Ok(out)
}

// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::topk::sparse_topk_select;
    use super::*;

    fn tiny_setup() -> (Vec<f32>, Vec<f32>, BlockTables, CuSeqlensQ) {
        // 1 batch, Tq=1, Hkv=1, head_dim=2, page_size=2, block_size=2 →
        // pages_per_block=1 → msb=4 → nbs=4.
        // 4 physical pages, each with 2 positions of 2-dim K.
        // Q = [1, 0].
        // K page 0: [[1, 0], [0, 1]]      → dots [1, 0] → max = 1
        // K page 1: [[0.5, 0.5], [0, 0]]  → dots [0.5, 0] → max = 0.5
        // K page 2: [[2, 0], [0, 0]]      → dots [2, 0] → max = 2
        // K page 3: [[0, 0], [0, 0]]      → dots [0, 0] → max = 0
        let q = vec![1.0, 0.0];
        let k = vec![
            1.0, 0.0, 0.0, 1.0, // page 0
            0.5, 0.5, 0.0, 0.0, // page 1
            2.0, 0.0, 0.0, 0.0, // page 2
            0.0, 0.0, 0.0, 0.0, // page 3
        ];
        let tbl = BlockTables::new(vec![0, 1, 2, 3], 1, 4).unwrap();
        let cu = CuSeqlensQ::new(vec![0, 1]).unwrap();
        (q, k, tbl, cu)
    }

    #[test]
    fn proxy_computes_per_block_max_score() {
        let (q, k, tbl, cu) = tiny_setup();
        let scores = compute_proxy_block_max_scores(&q, &k, &tbl, &cu, None, 1, 2, 2, 2).unwrap();
        // scale = 1/sqrt(2). Multiply expected raw dots by scale.
        let s = 1.0f32 / 2.0f32.sqrt();
        // Order: [q0, h0, blk0..3].
        assert!((scores[0] - 1.0 * s).abs() < 1e-6);
        assert!((scores[1] - 0.5 * s).abs() < 1e-6);
        assert!((scores[2] - 2.0 * s).abs() < 1e-6);
        assert!((scores[3] - 0.0 * s).abs() < 1e-6);
    }

    #[test]
    fn proxy_masks_beyond_used_kv_lens() {
        // used_kv_lens = 3 → ceil(3/2) = 2 valid sparse blocks → blk 2, 3
        // must be NEG_INFINITY (block 2 fully past, block 3 too), and
        // block 1 sees only its first position (pos 0 valid, pos 1 masked).
        let (q, k, tbl, cu) = tiny_setup();
        let scores =
            compute_proxy_block_max_scores(&q, &k, &tbl, &cu, Some(&[3]), 1, 2, 2, 2).unwrap();
        let s = 1.0f32 / 2.0f32.sqrt();
        // Block 0: full page → max 1.0.
        assert!((scores[0] - 1.0 * s).abs() < 1e-6);
        // Block 1: only pos 0 (dot 0.5) survives — pos 1 masked (kv index 3
        // is inside the block but kv index 3 is past used_kv_lens=3 too,
        // wait: block_start=2, block_end=min(4, 3) = 3, so pos 0 (kv=2)
        // valid, pos 1 (kv=3) masked. max = 0.5.
        assert!((scores[1] - 0.5 * s).abs() < 1e-6);
        assert!(scores[2].is_infinite() && scores[2] < 0.0);
        assert!(scores[3].is_infinite() && scores[3] < 0.0);
    }

    #[test]
    fn proxy_negative_page_id_is_skipped() {
        // Same setup but block_tables page slot 1 is -1 (unmapped).
        // Block 1 has no valid physical page → score stays NEG_INFINITY.
        let q = vec![1.0, 0.0];
        let k = vec![
            1.0, 0.0, 0.0, 1.0, // page 0
            2.0, 0.0, 0.0, 0.0, // page 1 (still exists physically)
        ];
        let tbl = BlockTables::new(vec![0, -1], 1, 2).unwrap();
        let cu = CuSeqlensQ::new(vec![0, 1]).unwrap();
        let scores = compute_proxy_block_max_scores(&q, &k, &tbl, &cu, None, 1, 2, 2, 2).unwrap();
        let s = 1.0f32 / 2.0f32.sqrt();
        assert!((scores[0] - 1.0 * s).abs() < 1e-6);
        assert!(scores[1].is_infinite() && scores[1] < 0.0);
    }

    #[test]
    fn proxy_and_topk_pipeline_end_to_end() {
        // Verify the two pieces compose: run proxy → feed sparse_topk_select
        // → expect [blk 2 (score 2.0), blk 0 (score 1.0)] (top-2).
        let (q, k, tbl, cu) = tiny_setup();
        let scores = compute_proxy_block_max_scores(&q, &k, &tbl, &cu, None, 1, 2, 2, 2).unwrap();
        // For (q=0, h=0) row of length msb=4:
        let row = &scores[0..4];
        let top = sparse_topk_select(row, 2, 4);
        assert_eq!(top, vec![2, 0]);
    }
}
