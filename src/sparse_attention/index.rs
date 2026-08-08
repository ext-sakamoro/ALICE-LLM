//! KV-outer CSR inverse index builder.
//!
//! Converts a query-outer top-K `SparseSelection` into a KV-outer CSR
//! representation: for each `(kv_head, sparse_block)` slot we obtain the
//! list of `(query, selected_rank)` edges that pointed at it. This is the
//! Rust-from-scratch analogue of the five CuTe-DSL kernels described in
//! `fw-ai/minimax-kernels/docs/m3-sparse-attention.md`:
//!
//! 1. init slots + counters,
//! 2. count edges (drop masked / out-of-range),
//! 3. reduce replica counters,
//! 4. scan counts → CSR offsets + compact nonempty slots,
//! 5. scatter `(query, rank)` into CSR order + emit inverse map.
//!
//! Here they collapse into two linear passes (count + scatter).

use super::types::{BlockTables, CuSeqlensQ, KvOuterIndex, SparseAttentionError, SparseSelection};

// Public API
// ---------------------------------------------------------------------------

/// Build a KV-outer CSR inverse index from a `[Tq, Hkv, topK]` selection tensor.
///
/// * `selected` — top-K sparse-block ids per `(query, kv_head)`. `-1` marks
///   unused ranks.
/// * `block_tables` — `[B, max_pages]` logical→physical page ids. Only the
///   shape is consumed here; physical page addresses are looked up later by
///   the forward kernel.
/// * `cu_seqlens_q` — packed query prefix sums.
/// * `used_kv_lens` — optional `[B]` real KV lengths. Selections at slots
///   past `ceil(used_kv_lens[b] / block_size)` are dropped, matching the
///   `used_kv_lens` semantics of `kvouter_attention`.
/// * `block_size`, `page_size` — sparse block size and physical page size.
///   Must satisfy `block_size % page_size == 0`.
///
/// Returns a fully populated [`KvOuterIndex`].
pub fn build_kvouter_index(
    selected: &SparseSelection,
    block_tables: &BlockTables,
    cu_seqlens_q: &CuSeqlensQ,
    used_kv_lens: Option<&[i32]>,
    block_size: usize,
    page_size: usize,
) -> Result<KvOuterIndex, SparseAttentionError> {
    // Argument sanity.
    if selected.tq == 0 || selected.hkv == 0 || selected.topk == 0 {
        return Err(SparseAttentionError::EmptyInput);
    }
    if selected.tq != cu_seqlens_q.total_tq() {
        return Err(SparseAttentionError::ShapeMismatch {
            what: "cu_seqlens_q.total_tq vs selected.tq",
            expected: selected.tq,
            got: cu_seqlens_q.total_tq(),
        });
    }
    if cu_seqlens_q.batch_size() != block_tables.batch_size {
        return Err(SparseAttentionError::ShapeMismatch {
            what: "cu_seqlens_q.batch_size vs block_tables.batch_size",
            expected: block_tables.batch_size,
            got: cu_seqlens_q.batch_size(),
        });
    }
    if page_size == 0 || block_size == 0 || !block_size.is_multiple_of(page_size) {
        return Err(SparseAttentionError::BlockPageMismatch {
            block_size,
            page_size,
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

    let pages_per_block = block_size / page_size;
    // `msb` = max sparse blocks in one block-table row. When `max_pages`
    // doesn't cover a whole sparse block we treat the tail as unaddressable.
    let msb = block_tables.max_pages / pages_per_block;
    if msb == 0 {
        return Err(SparseAttentionError::BlockPageMismatch {
            block_size,
            page_size,
        });
    }

    let batch_size = cu_seqlens_q.batch_size();
    let hkv = selected.hkv;
    let tq = selected.tq;
    let topk = selected.topk;
    let nbs = batch_size * msb;

    // Per-batch valid-slot cap (from used_kv_lens) — inclusive upper bound.
    let per_batch_valid_slots: Vec<usize> = (0..batch_size)
        .map(|b| {
            used_kv_lens.map_or(msb, |used| {
                let lk = used[b].max(0) as usize;
                // ceil(lk / block_size).
                lk.div_ceil(block_size).min(msb)
            })
        })
        .collect();

    // --- Pass 1: count edges into a dense [hkv, nbs] slot histogram. -------
    let mut slot_counts = vec![0i32; hkv * nbs];

    // Precompute batch id per packed query index once (cheap linear scan).
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

    for i in 0..tq {
        let b = batch_of_tq[i];
        let valid_slots_b = per_batch_valid_slots[b];
        for h in 0..hkv {
            for r in 0..topk {
                let blk = selected.get(i, h, r);
                if blk < 0 {
                    continue;
                }
                let blk_u = blk as usize;
                if blk_u >= msb {
                    return Err(SparseAttentionError::SelectedOutOfRange { got: blk, msb });
                }
                if blk_u >= valid_slots_b {
                    // Masked by used_kv_lens.
                    continue;
                }
                let real_slot = b * msb + blk_u;
                slot_counts[h * nbs + real_slot] += 1;
            }
        }
    }

    // --- Build compact CSR offsets + compact slot list per head. -----------
    let sel_offsets_len = hkv * (nbs + 1);
    let mut sel_offsets = vec![0i32; sel_offsets_len];
    let mut sel_slots = vec![0i32; hkv * nbs];
    let mut num_sel = vec![0i32; hkv];
    let mut edge_head_base = vec![0usize; hkv + 1];

    for h in 0..hkv {
        let off_base = h * (nbs + 1);
        let mut compact_j: usize = 0;
        let mut cum: i32 = 0;
        sel_offsets[off_base] = 0;
        for raw_slot in 0..nbs {
            let cnt = slot_counts[h * nbs + raw_slot];
            if cnt > 0 {
                sel_slots[h * nbs + compact_j] = raw_slot as i32;
                cum += cnt;
                sel_offsets[off_base + compact_j + 1] = cum;
                compact_j += 1;
            }
        }
        num_sel[h] = compact_j as i32;
        // Plateau the rest of sel_offsets at `cum`.
        for j in (compact_j + 1)..=nbs {
            sel_offsets[off_base + j] = cum;
        }
        edge_head_base[h + 1] = edge_head_base[h] + cum as usize;
    }

    let total_edges = edge_head_base[hkv];

    // --- Pass 2: scatter edges into CSR order + build inverse map. ---------
    let mut edges = vec![(0i32, 0i32); total_edges];
    let mut inv = vec![-1i32; hkv * tq * topk];

    // Per-head write cursor, initialized at each compact slot's start.
    // We can reuse sel_offsets for cursor bookkeeping via a small
    // per-head Vec that tracks how many edges we've written into each
    // compact slot so far.
    let mut write_cursor: Vec<i32> = Vec::with_capacity(hkv * nbs);
    for h in 0..hkv {
        let off_base = h * (nbs + 1);
        for j in 0..nbs {
            write_cursor.push(sel_offsets[off_base + j]);
        }
    }

    // Reverse map: raw_slot → compact_j, per head. Simple dense LUT.
    let mut raw_to_compact: Vec<i32> = vec![-1; hkv * nbs];
    for h in 0..hkv {
        for j in 0..(num_sel[h] as usize) {
            let raw = sel_slots[h * nbs + j] as usize;
            raw_to_compact[h * nbs + raw] = j as i32;
        }
    }

    for i in 0..tq {
        let b = batch_of_tq[i];
        let valid_slots_b = per_batch_valid_slots[b];
        for h in 0..hkv {
            for r in 0..topk {
                let blk = selected.get(i, h, r);
                if blk < 0 {
                    continue;
                }
                let blk_u = blk as usize;
                if blk_u >= valid_slots_b {
                    continue;
                }
                let real_slot = b * msb + blk_u;
                let compact_j = raw_to_compact[h * nbs + real_slot];
                if compact_j < 0 {
                    // Shouldn't happen — counted in pass 1.
                    continue;
                }
                let cj = compact_j as usize;
                let cursor_idx = h * nbs + cj;
                let write_pos = write_cursor[cursor_idx] as usize;
                edges[edge_head_base[h] + write_pos] = (i as i32, r as i32);
                write_cursor[cursor_idx] += 1;
                // Inverse map: partial position is within-head, so we
                // store the head-local position directly.
                inv[(h * tq + i) * topk + r] = write_pos as i32;
            }
        }
    }

    Ok(KvOuterIndex {
        sel_slots,
        sel_offsets,
        num_sel,
        edges,
        edge_head_base,
        inv,
        hkv,
        tq,
        topk,
        nbs,
        msb,
        batch_size,
        pages_per_block,
    })
}

// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_selection() -> (SparseSelection, BlockTables, CuSeqlensQ) {
        // 1 batch, 3 packed queries (Tq=3), 1 KV head, topk=2, block_size=128,
        // page_size=128 → pages_per_block=1 → msb = max_pages.
        // block_tables shape [1, 4] → msb = 4 → nbs = 4.
        // selected[Tq=3, Hkv=1, topK=2]:
        //   q0: [0, 2]
        //   q1: [2, -1]         (rank 1 padding)
        //   q2: [3, 0]
        // Expected CSR (head 0):
        //   raw slot 0 → edges [(q=0,r=0), (q=2,r=1)]     count 2
        //   raw slot 2 → edges [(q=0,r=1), (q=1,r=0)]     count 2
        //   raw slot 3 → edges [(q=2,r=0)]                count 1
        //   compact_j: 0→raw0, 1→raw2, 2→raw3            num_sel=3
        let sel = SparseSelection::new(vec![0, 2, 2, -1, 3, 0], 3, 1, 2).unwrap();
        let tbl = BlockTables::new(vec![0, 1, 2, 3], 1, 4).unwrap();
        let cu = CuSeqlensQ::new(vec![0, 3]).unwrap();
        (sel, tbl, cu)
    }

    #[test]
    fn build_index_matches_hand_computed_csr() {
        let (sel, tbl, cu) = small_selection();
        let idx = build_kvouter_index(&sel, &tbl, &cu, None, 128, 128).unwrap();

        assert_eq!(idx.hkv, 1);
        assert_eq!(idx.tq, 3);
        assert_eq!(idx.topk, 2);
        assert_eq!(idx.nbs, 4);
        assert_eq!(idx.msb, 4);
        assert_eq!(idx.pages_per_block, 1);

        // num_sel[0] == 3.
        assert_eq!(idx.num_sel_of(0), 3);

        // compact_j → raw slot.
        assert_eq!(idx.raw_slot(0, 0), 0);
        assert_eq!(idx.raw_slot(0, 1), 2);
        assert_eq!(idx.raw_slot(0, 2), 3);

        // sel_offsets[h=0]: [0, 2, 4, 5, 5]  (plateaued after num_sel entries).
        assert_eq!(&idx.sel_offsets[0..5], &[0, 2, 4, 5, 5]);

        // Compact slot 0 (raw 0): edges (q0,r0), (q2,r1).
        let (s0, e0) = idx.edge_range(0, 0);
        assert_eq!(&idx.edges[s0..e0], &[(0, 0), (2, 1)]);

        // Compact slot 1 (raw 2): edges (q0,r1), (q1,r0).
        let (s1, e1) = idx.edge_range(0, 1);
        assert_eq!(&idx.edges[s1..e1], &[(0, 1), (1, 0)]);

        // Compact slot 2 (raw 3): edges (q2,r0).
        let (s2, e2) = idx.edge_range(0, 2);
        assert_eq!(&idx.edges[s2..e2], &[(2, 0)]);

        // Inverse map spot checks: partial position is head-local.
        //   (q=0, r=0) → position 0 in slot 0 → 0.
        //   (q=0, r=1) → position 0 in slot 1 → 2.
        //   (q=2, r=1) → position 1 in slot 0 → 1.
        //   (q=1, r=1) → dropped → -1.
        assert_eq!(idx.inv_of(0, 0, 0), 0);
        assert_eq!(idx.inv_of(0, 0, 1), 2);
        assert_eq!(idx.inv_of(0, 2, 1), 1);
        assert_eq!(idx.inv_of(0, 1, 1), -1);
    }

    #[test]
    fn build_index_drops_padding_ranks() {
        // All ranks padding → no edges emitted.
        let sel = SparseSelection::new(vec![-1, -1, -1, -1], 2, 1, 2).unwrap();
        let tbl = BlockTables::new(vec![0, 1], 1, 2).unwrap();
        let cu = CuSeqlensQ::new(vec![0, 2]).unwrap();
        let idx = build_kvouter_index(&sel, &tbl, &cu, None, 64, 64).unwrap();
        assert_eq!(idx.num_sel_of(0), 0);
        assert_eq!(idx.edges.len(), 0);
        assert!(idx.inv.iter().all(|&v| v == -1));
    }

    #[test]
    fn build_index_respects_used_kv_lens() {
        // 1 batch, Tq=1, Hkv=1, topK=3, block=64, page=64, max_pages=4 → msb=4.
        // selected: [0, 1, 3].
        // used_kv_lens = 128 → ceil(128/64) = 2 → valid slots 0..2 → drop 3.
        let sel = SparseSelection::new(vec![0, 1, 3], 1, 1, 3).unwrap();
        let tbl = BlockTables::new(vec![10, 11, 12, 13], 1, 4).unwrap();
        let cu = CuSeqlensQ::new(vec![0, 1]).unwrap();
        let idx = build_kvouter_index(&sel, &tbl, &cu, Some(&[128]), 64, 64).unwrap();
        // Only ranks 0 (→raw 0) and 1 (→raw 1) survive.
        assert_eq!(idx.num_sel_of(0), 2);
        // Rank 2 (blk=3) was dropped.
        assert_eq!(idx.inv_of(0, 0, 2), -1);
        // inv stores the head-local edge index (position in `edges` for the
        // owning head), not a compact-slot-local offset. Each surviving edge
        // is the first (and only) entry in its slot, so head-local indexes
        // are 0 and 1.
        assert_eq!(idx.inv_of(0, 0, 0), 0);
        assert_eq!(idx.inv_of(0, 0, 1), 1);
    }

    #[test]
    fn multi_batch_multi_head_index() {
        // batch_size=2, cu_seqlens_q=[0,2,3] → Tq=3.
        // Hkv=2, topK=2, block=128 page=128 max_pages=2 → msb=2 → nbs=4.
        // selected[Tq=3, Hkv=2, topK=2]:
        //   q0 (b=0): head0 [0,1], head1 [1,-1]
        //   q1 (b=0): head0 [1,0], head1 [0, 1]
        //   q2 (b=1): head0 [0,-1], head1 [1,0]
        // real_slot = b*msb + blk.
        //   Head 0 raw slots hit:  b=0 slot 0 (q0r0, q1r1), b=0 slot 1 (q0r1, q1r0),
        //                          b=1 slot 2 (q2r0).
        //   Head 1 raw slots hit:  b=0 slot 0 (q1r0), b=0 slot 1 (q0r0, q1r1),
        //                          b=1 slot 2 (q2r1), b=1 slot 3 (q2r0).
        let sel = SparseSelection::new(
            vec![
                0, 1, 1, -1, // q0 head0,head1
                1, 0, 0, 1, // q1 head0,head1
                0, -1, 1, 0, // q2 head0,head1
            ],
            3,
            2,
            2,
        )
        .unwrap();
        let tbl = BlockTables::new(vec![0, 1, 2, 3], 2, 2).unwrap();
        let cu = CuSeqlensQ::new(vec![0, 2, 3]).unwrap();
        let idx = build_kvouter_index(&sel, &tbl, &cu, None, 128, 128).unwrap();

        assert_eq!(idx.nbs, 4);

        // Head 0: raw slots {0,1,2} → num_sel=3.
        assert_eq!(idx.num_sel_of(0), 3);
        assert_eq!(idx.raw_slot(0, 0), 0);
        assert_eq!(idx.raw_slot(0, 1), 1);
        assert_eq!(idx.raw_slot(0, 2), 2);

        // Head 1: raw slots {0,1,2,3} → num_sel=4.
        assert_eq!(idx.num_sel_of(1), 4);
        assert_eq!(idx.raw_slot(1, 0), 0);
        assert_eq!(idx.raw_slot(1, 1), 1);
        assert_eq!(idx.raw_slot(1, 2), 2);
        assert_eq!(idx.raw_slot(1, 3), 3);

        // Head 0 slot 0 edges: (q0,r0), (q1,r1).
        let (s, e) = idx.edge_range(0, 0);
        assert_eq!(&idx.edges[s..e], &[(0, 0), (1, 1)]);

        // Head 1 slot 3 edges: (q2, r0).
        let (s, e) = idx.edge_range(1, 3);
        assert_eq!(&idx.edges[s..e], &[(2, 0)]);
    }

    #[test]
    fn build_index_rejects_out_of_range_selection() {
        let sel = SparseSelection::new(vec![5], 1, 1, 1).unwrap();
        let tbl = BlockTables::new(vec![0, 1], 1, 2).unwrap();
        let cu = CuSeqlensQ::new(vec![0, 1]).unwrap();
        let err = build_kvouter_index(&sel, &tbl, &cu, None, 128, 128).unwrap_err();
        matches!(err, SparseAttentionError::SelectedOutOfRange { got: 5, .. });
    }

    #[test]
    fn build_index_rejects_bad_block_page_ratio() {
        let sel = SparseSelection::new(vec![0], 1, 1, 1).unwrap();
        let tbl = BlockTables::new(vec![0], 1, 1).unwrap();
        let cu = CuSeqlensQ::new(vec![0, 1]).unwrap();
        // block_size % page_size != 0.
        let err = build_kvouter_index(&sel, &tbl, &cu, None, 100, 64).unwrap_err();
        matches!(err, SparseAttentionError::BlockPageMismatch { .. });
    }
}
