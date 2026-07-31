//! Load-balance scheduler for the KV-outer forward pass.
//!
//! Upstream (fw-ai/minimax-kernels) partitions the CSR-ordered edge stream
//! `(kv_head, compact_slot, query_within_slot)` into `num_splits`
//! contiguous runs and hands each split to a persistent SM100 CTA. Splits
//! beyond the real work receive `(-1, -1, -1)` sentinels.
//!
//! On CPU we don't need thousands of splits — the scheduler here is a
//! forward-compatible scaffold that returns a flat `Vec<WorkSplit>` covering
//! every real edge. Phase MSA.5 hooks a `rayon`-based parallel walk over the
//! same splits.

use super::types::KvOuterIndex;

// Types
// ---------------------------------------------------------------------------

/// One unit of KV-outer work: a contiguous sub-range of edges within a
/// single `(head, compact_slot)` cell.
///
/// A `WorkSplit` never crosses a `(head, compact_slot)` boundary so that
/// each split can load its KV block once and reuse it across every edge
/// in `edge_start..edge_end`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkSplit {
    /// KV head owning the slot.
    pub head: usize,
    /// Compact slot rank within the head (`0 .. num_sel[head]`).
    pub compact_slot: usize,
    /// First edge (head-local, absolute in `idx.edges` after adding
    /// `edge_head_base[head]`).
    pub edge_start: usize,
    /// One past the last edge.
    pub edge_end: usize,
}

// Public API
// ---------------------------------------------------------------------------

/// Enumerate every KV-outer work unit as a flat vector of [`WorkSplit`].
///
/// Each `(head, compact_slot)` cell contributes exactly one `WorkSplit`
/// covering all its edges. This ordering matches the persistent-kernel
/// traversal described in `fw-ai/minimax-kernels/docs/m3-sparse-attention.md`
/// (§ "Compact slot rank is the loop variable").
#[must_use]
pub fn enumerate_work_units(idx: &KvOuterIndex) -> Vec<WorkSplit> {
    let mut out = Vec::new();
    for h in 0..idx.hkv {
        let off_base = h * (idx.nbs + 1);
        let num_sel = idx.num_sel_of(h);
        for j in 0..num_sel {
            let head_local_start = idx.sel_offsets[off_base + j] as usize;
            let head_local_end = idx.sel_offsets[off_base + j + 1] as usize;
            if head_local_end <= head_local_start {
                continue;
            }
            let base = idx.edge_head_base[h];
            out.push(WorkSplit {
                head: h,
                compact_slot: j,
                edge_start: base + head_local_start,
                edge_end: base + head_local_end,
            });
        }
    }
    out
}

/// Build a fixed-`num_splits` schedule by chunking [`enumerate_work_units`]
/// evenly. Unused splits are padded with sentinel `WorkSplit` values whose
/// `edge_start == edge_end` (no work).
///
/// This is deliberately simple — the CPU walk in Phase MSA.3 doesn't rely on
/// splits — but it mirrors the upstream contract (fixed-count splits with
/// sentinel tails) so a future `rayon` / GPU path drops in cleanly.
#[must_use]
pub fn build_fixed_schedule(idx: &KvOuterIndex, num_splits: usize) -> Vec<WorkSplit> {
    let units = enumerate_work_units(idx);
    if num_splits == 0 {
        return units;
    }
    let mut out = Vec::with_capacity(num_splits.max(units.len()));
    if units.is_empty() {
        return sentinel_vec(num_splits);
    }
    // Even-ish partition of `units` across `num_splits` buckets. Each
    // WorkSplit is atomic here — bigger cells are not further sliced. This
    // is intentional: the KV-block reuse breaks if we split within a cell.
    let n = units.len();
    let base = n / num_splits.max(1);
    let rem = n % num_splits.max(1);
    let mut cursor = 0usize;
    for s in 0..num_splits.max(1) {
        let count = base + usize::from(s < rem);
        if count == 0 {
            out.push(sentinel_split());
            continue;
        }
        // We flatten by emitting the first unit of the bucket verbatim;
        // additional units within the same bucket are appended sequentially.
        // Upstream keeps one WorkSplit per SM; on CPU we don't gain from
        // fewer bookkeeping entries, so we keep the full unit list.
        for u in &units[cursor..cursor + count] {
            out.push(*u);
        }
        cursor += count;
    }
    // Any trailing capacity (n < num_splits) is already filled above; guard
    // just in case.
    while out.len() < num_splits {
        out.push(sentinel_split());
    }
    out
}

// Helpers
// ---------------------------------------------------------------------------

const fn sentinel_split() -> WorkSplit {
    WorkSplit {
        head: 0,
        compact_slot: 0,
        edge_start: 0,
        edge_end: 0,
    }
}

fn sentinel_vec(n: usize) -> Vec<WorkSplit> {
    vec![sentinel_split(); n]
}

// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::{
        build_kvouter_index,
        types::{BlockTables, CuSeqlensQ, SparseSelection},
    };
    use super::*;

    fn make_multi_head_idx() -> super::super::types::KvOuterIndex {
        // Same fixture as index::multi_batch_multi_head_index — reused so
        // scheduler correctness locks against a known-good index build.
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
        build_kvouter_index(&sel, &tbl, &cu, None, 128, 128).unwrap()
    }

    #[test]
    fn enumerate_work_units_covers_every_edge() {
        let idx = make_multi_head_idx();
        let units = enumerate_work_units(&idx);
        let total_edges: usize = units.iter().map(|w| w.edge_end - w.edge_start).sum();
        assert_eq!(total_edges, idx.edges.len());
        // Head 0 has 3 nonempty slots, head 1 has 4 → 7 work units.
        assert_eq!(units.len(), 7);
    }

    #[test]
    fn enumerate_work_units_stays_within_head_offsets() {
        let idx = make_multi_head_idx();
        for u in enumerate_work_units(&idx) {
            let base = idx.edge_head_base[u.head];
            let cap = idx.edge_head_base[u.head + 1];
            assert!(u.edge_start >= base);
            assert!(u.edge_end <= cap);
        }
    }

    #[test]
    fn build_fixed_schedule_pads_with_sentinels() {
        let idx = make_multi_head_idx();
        let schedule = build_fixed_schedule(&idx, 32);
        assert!(schedule.len() >= 32);
        // Sentinel splits have edge_start == edge_end.
        let sentinel_count = schedule
            .iter()
            .filter(|w| w.edge_start == w.edge_end)
            .count();
        assert!(sentinel_count >= 32 - 7);
    }

    #[test]
    fn build_fixed_schedule_num_splits_zero_returns_units() {
        let idx = make_multi_head_idx();
        assert_eq!(build_fixed_schedule(&idx, 0), enumerate_work_units(&idx));
    }
}
