//! sparse attention types.

// Types
// ---------------------------------------------------------------------------

/// Errors returned by the `sparse_attention` module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseAttentionError {
    /// `Hq % Hkv != 0` — query heads must be a multiple of KV heads.
    HeadCountMismatch { hq: usize, hkv: usize },
    /// `block_size` is not a positive multiple of `page_size`.
    BlockPageMismatch { block_size: usize, page_size: usize },
    /// `cu_seqlens_q` must be `[B+1]` and monotonically non-decreasing.
    CuSeqlensInvalid,
    /// A shape argument disagrees with the flat buffer length.
    ShapeMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    /// A `selected` block id is outside `[-1, msb)`.
    SelectedOutOfRange { got: i32, msb: usize },
    /// Head dim mismatch across `q` / `k_cache` / `v_cache`.
    HeadDimMismatch,
    /// Empty input (`Tq == 0` or `Hkv == 0` or `topk == 0`).
    EmptyInput,
}

impl core::fmt::Display for SparseAttentionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::HeadCountMismatch { hq, hkv } => {
                write!(f, "Hq ({hq}) must be a positive multiple of Hkv ({hkv})")
            }
            Self::BlockPageMismatch {
                block_size,
                page_size,
            } => {
                write!(
                    f,
                    "block_size ({block_size}) must be a positive multiple of page_size ({page_size})"
                )
            }
            Self::CuSeqlensInvalid => f.write_str("cu_seqlens_q invalid (empty or not monotonic)"),
            Self::ShapeMismatch {
                what,
                expected,
                got,
            } => {
                write!(
                    f,
                    "shape mismatch for {what}: expected {expected}, got {got}"
                )
            }
            Self::SelectedOutOfRange { got, msb } => {
                write!(f, "selected block id {got} out of range [-1, {msb})")
            }
            Self::HeadDimMismatch => f.write_str("head dim mismatch across q/k_cache/v_cache"),
            Self::EmptyInput => f.write_str("empty input"),
        }
    }
}

// std::error::Error impl is gated on the std feature (which is on by default,
// but the crate can be compiled `--no-default-features` for wasm work).
#[cfg(feature = "std")]
impl std::error::Error for SparseAttentionError {}

/// Per-query top-K KV block selection tensor.
///
/// Logical shape: `[tq, hkv, topk]` stored row-major in `selected`.
/// A value of `-1` marks an unused selection rank (padding).
#[derive(Debug, Clone)]
pub struct SparseSelection {
    /// Flat buffer with length `tq * hkv * topk`.
    pub selected: Vec<i32>,
    /// Total query tokens (packed, not per-batch).
    pub tq: usize,
    /// KV heads.
    pub hkv: usize,
    /// Per-`(query, kv-head)` top-K depth.
    pub topk: usize,
}

impl SparseSelection {
    /// Construct from a flat buffer. Returns an error when the length
    /// disagrees with `tq * hkv * topk`.
    pub fn new(
        selected: Vec<i32>,
        tq: usize,
        hkv: usize,
        topk: usize,
    ) -> Result<Self, SparseAttentionError> {
        let expected = tq
            .checked_mul(hkv)
            .and_then(|v| v.checked_mul(topk))
            .ok_or(SparseAttentionError::ShapeMismatch {
                what: "selected (overflow)",
                expected: usize::MAX,
                got: selected.len(),
            })?;
        if selected.len() != expected {
            return Err(SparseAttentionError::ShapeMismatch {
                what: "selected",
                expected,
                got: selected.len(),
            });
        }
        Ok(Self {
            selected,
            tq,
            hkv,
            topk,
        })
    }

    /// Access `selected[tq_idx, hkv_idx, rank]` without bounds checks.
    ///
    /// # Safety
    /// Caller guarantees `tq_idx < self.tq`, `hkv_idx < self.hkv`, `rank < self.topk`.
    #[inline]
    #[must_use]
    pub unsafe fn get_unchecked(&self, tq_idx: usize, hkv_idx: usize, rank: usize) -> i32 {
        let idx = (tq_idx * self.hkv + hkv_idx) * self.topk + rank;
        *self.selected.get_unchecked(idx)
    }

    /// Safe accessor for `selected[tq_idx, hkv_idx, rank]`.
    #[inline]
    #[must_use]
    pub fn get(&self, tq_idx: usize, hkv_idx: usize, rank: usize) -> i32 {
        self.selected[(tq_idx * self.hkv + hkv_idx) * self.topk + rank]
    }
}

/// Logical-to-physical page table per sequence.
///
/// Logical shape: `[batch_size, max_pages]` stored row-major in `table`.
/// A value of `-1` may pad rows (unused physical pages).
#[derive(Debug, Clone)]
pub struct BlockTables {
    /// Flat buffer with length `batch_size * max_pages`.
    pub table: Vec<i32>,
    /// Sequences per call.
    pub batch_size: usize,
    /// Maximum physical pages addressable per sequence.
    pub max_pages: usize,
}

impl BlockTables {
    /// Construct from a flat buffer.
    pub fn new(
        table: Vec<i32>,
        batch_size: usize,
        max_pages: usize,
    ) -> Result<Self, SparseAttentionError> {
        let expected = batch_size.saturating_mul(max_pages);
        if table.len() != expected {
            return Err(SparseAttentionError::ShapeMismatch {
                what: "block_tables",
                expected,
                got: table.len(),
            });
        }
        Ok(Self {
            table,
            batch_size,
            max_pages,
        })
    }

    /// Access `table[batch_idx, page_idx]`.
    #[inline]
    #[must_use]
    pub fn get(&self, batch_idx: usize, page_idx: usize) -> i32 {
        self.table[batch_idx * self.max_pages + page_idx]
    }
}

/// Cumulative packed-query lengths: `[B+1]`, `cu_seqlens_q[0] == 0`,
/// `cu_seqlens_q[B] == Tq`, monotonically non-decreasing.
#[derive(Debug, Clone)]
pub struct CuSeqlensQ {
    /// Underlying `[B+1]` prefix-sum buffer.
    pub prefix: Vec<i64>,
}

impl CuSeqlensQ {
    /// Construct + validate.
    pub fn new(prefix: Vec<i64>) -> Result<Self, SparseAttentionError> {
        if prefix.len() < 2 || prefix[0] != 0 {
            return Err(SparseAttentionError::CuSeqlensInvalid);
        }
        for w in prefix.windows(2) {
            if w[1] < w[0] {
                return Err(SparseAttentionError::CuSeqlensInvalid);
            }
        }
        Ok(Self { prefix })
    }

    /// Batch count `B`.
    #[inline]
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.prefix.len() - 1
    }

    /// Total packed query tokens `Tq`.
    #[inline]
    #[must_use]
    pub fn total_tq(&self) -> usize {
        // Safe by construction: prefix has at least 2 entries and prefix[0] == 0.
        self.prefix[self.prefix.len() - 1] as usize
    }

    /// Map a packed query index to its sequence id via binary search
    /// (right-open intervals). Returns `batch_size() - 1` for the last token
    /// of the last sequence.
    #[inline]
    #[must_use]
    pub fn batch_of(&self, tq_idx: usize) -> usize {
        // partition_point returns the first index where prefix[i] > tq_idx,
        // so subtracting 1 gives the batch containing tq_idx.
        let idx = self.prefix.partition_point(|&p| p <= tq_idx as i64);
        idx.saturating_sub(1)
    }
}

/// KV-outer compressed sparse row (CSR) inverted index.
///
/// Layout is per-KV-head; each head has its own compact list of nonempty
/// `(kv_head, sparse_block)` slots plus a CSR-ordered edge list.
#[derive(Debug, Clone)]
pub struct KvOuterIndex {
    /// `[hkv, num_sel_max]` — compact rank → raw slot id (0..nbs). Rows are
    /// padded to `nbs` length but only the first `num_sel[h]` entries are
    /// meaningful.
    pub sel_slots: Vec<i32>,
    /// `[hkv, nbs+1]` — compact CSR offsets (plateaued after
    /// `num_sel[h]`).
    pub sel_offsets: Vec<i32>,
    /// `[hkv]` — number of nonempty compact slots per head.
    pub num_sel: Vec<i32>,
    /// Concatenated CSR edge payload across all heads: each entry is
    /// `(query_index, selected_rank)` in CSR order.
    pub edges: Vec<(i32, i32)>,
    /// `[hkv+1]` — cumulative base into `edges` per head (so head `h` owns
    /// `edges[edge_head_base[h] .. edge_head_base[h+1]]`).
    pub edge_head_base: Vec<usize>,
    /// `[hkv, tq, topk]` — `(query, rank)` → partial position (within head).
    /// `-1` means the entry was dropped (masked or out-of-range).
    pub inv: Vec<i32>,
    /// KV heads.
    pub hkv: usize,
    /// Packed query tokens.
    pub tq: usize,
    /// Per-`(query, kv-head)` top-K depth.
    pub topk: usize,
    /// `nbs = batch_size * msb` — dense slot capacity per head.
    pub nbs: usize,
    /// `msb` — maximum sparse blocks addressable in one block-table row.
    pub msb: usize,
    /// `batch_size = cu_seqlens_q.batch_size()`.
    pub batch_size: usize,
    /// Physical pages per sparse block (`block_size / page_size`).
    pub pages_per_block: usize,
}

impl KvOuterIndex {
    /// Number of nonempty compact slots for head `h`.
    #[inline]
    #[must_use]
    pub fn num_sel_of(&self, h: usize) -> usize {
        self.num_sel[h] as usize
    }

    /// Edge range in `self.edges` for compact slot `j` under head `h`.
    #[inline]
    #[must_use]
    pub fn edge_range(&self, h: usize, j: usize) -> (usize, usize) {
        let base = h * (self.nbs + 1);
        let start = self.sel_offsets[base + j] as usize;
        let end = self.sel_offsets[base + j + 1] as usize;
        let head_base = self.edge_head_base[h];
        (head_base + start, head_base + end)
    }

    /// Raw slot id for compact slot `j` under head `h`.
    #[inline]
    #[must_use]
    pub fn raw_slot(&self, h: usize, j: usize) -> i32 {
        self.sel_slots[h * self.nbs + j]
    }

    /// Inverse map entry: `(query, rank)` → partial position under head `h`.
    /// Returns `-1` when the entry was dropped.
    #[inline]
    #[must_use]
    pub fn inv_of(&self, h: usize, tq_idx: usize, rank: usize) -> i32 {
        self.inv[(h * self.tq + tq_idx) * self.topk + rank]
    }
}

// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_selection_shape_mismatch() {
        let err = SparseSelection::new(vec![0, 1, 2], 2, 2, 2).unwrap_err();
        matches!(err, SparseAttentionError::ShapeMismatch { .. });
    }

    #[test]
    fn sparse_selection_get_indexes_row_major() {
        // tq=2, hkv=2, topk=2 → 8 entries.
        // At (tq=1, hkv=0, rank=1) index = (1*2 + 0)*2 + 1 = 5 → value 55.
        let sel = SparseSelection::new(vec![0, 1, 2, 3, 10, 55, 20, 21], 2, 2, 2).unwrap();
        assert_eq!(sel.get(1, 0, 1), 55);
        assert_eq!(sel.get(0, 1, 0), 2);
    }

    #[test]
    fn block_tables_shape_mismatch() {
        let err = BlockTables::new(vec![0, 1, 2], 2, 2).unwrap_err();
        matches!(err, SparseAttentionError::ShapeMismatch { .. });
    }

    #[test]
    fn cu_seqlens_validation() {
        // Empty / not starting at 0 / non-monotonic → error.
        assert!(CuSeqlensQ::new(vec![]).is_err());
        assert!(CuSeqlensQ::new(vec![1, 2, 3]).is_err());
        assert!(CuSeqlensQ::new(vec![0, 3, 2]).is_err());
        // Well-formed.
        let c = CuSeqlensQ::new(vec![0, 3, 8]).unwrap();
        assert_eq!(c.batch_size(), 2);
        assert_eq!(c.total_tq(), 8);
    }

    #[test]
    fn batch_of_maps_packed_query_indexes() {
        // cu_seqlens = [0, 3, 8] → batch 0 = [0,1,2], batch 1 = [3..8).
        let c = CuSeqlensQ::new(vec![0, 3, 8]).unwrap();
        assert_eq!(c.batch_of(0), 0);
        assert_eq!(c.batch_of(2), 0);
        assert_eq!(c.batch_of(3), 1);
        assert_eq!(c.batch_of(7), 1);
    }
}
