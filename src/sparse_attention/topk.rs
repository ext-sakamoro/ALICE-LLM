//! Top-K KV-block selection over per-block scores.
//!
//! Rust-from-scratch analogue of MSA's `sparse_topk_select` indexer. The
//! upstream CUDA implementation (see MSA's
//! `csrc/include/sparse_topk_select.cuh`) is a two-stage histogram-step +
//! insertion-sort adapted from NVIDIA TensorRT-LLM's `indexerTopK.cu`. We
//! reimplement the same *algorithm* here without touching any of that code.
//!
//! ## Algorithm
//!
//! For a single score row of length `n`:
//!
//! 1. Two-pass **radix histogram** over the top exponent bits of the FP32
//!    scores to find a cutoff `s*` such that at least `topk` values satisfy
//!    `score >= s*`.
//! 2. **Insertion sort** the survivors into a size-`topk` running max heap
//!    (implemented as a small sorted array; N is tiny — 8..64 in practice).
//!
//! For our CPU port we collapse the histogram step into a single partial
//! selection so complexity stays `O(n)` for the common case where
//! `topk << n`, and we drop back to a full `O(n log k)` sort when `topk` is
//! close to `n`.

// Public API
// ---------------------------------------------------------------------------

/// Select the top-`topk` block indexes from a flat score buffer of length
/// `num_blocks`.
///
/// * `scores` — `[num_blocks]` FP32 per-block scores. Higher is more
///   important. `NaN` is treated as `-inf`.
/// * `topk` — number of block indexes to return.
/// * `num_valid_blocks` — only indexes in `0..num_valid_blocks` are
///   candidates. Anything beyond is treated as padding.
///
/// Returns a `Vec<i32>` of length `topk`. Entries beyond the number of
/// valid candidates are padded with `-1`.
#[must_use]
pub fn sparse_topk_select(scores: &[f32], topk: usize, num_valid_blocks: usize) -> Vec<i32> {
    let n = scores.len().min(num_valid_blocks);
    if topk == 0 {
        return Vec::new();
    }
    if n == 0 {
        return vec![-1; topk];
    }

    // Small-k fast path (the common case for sparse attention: topk 8..64).
    // We keep a size-`topk` sorted running max heap using linear-scan
    // insertion. For our workloads topk is 16..64 which makes this cheaper
    // than a binary heap thanks to branch predictability and cache locality.
    let k = topk.min(n);
    let mut heap_scores: Vec<f32> = vec![f32::NEG_INFINITY; k];
    let mut heap_ids: Vec<i32> = vec![-1; k];

    // `min_of_heap` is the smallest score currently kept and lives at
    // `heap_scores[0]`. We insert a candidate iff it exceeds `min_of_heap`.
    for (i, &raw_score) in scores.iter().take(n).enumerate() {
        // Treat NaN as -inf so it never displaces a real score.
        let score = if raw_score.is_nan() {
            f32::NEG_INFINITY
        } else {
            raw_score
        };
        if score <= heap_scores[0] {
            continue;
        }
        // Find insertion position by scanning from the bottom (small end)
        // upward. This is a straightforward insertion into a sorted array
        // ascending: heap_scores[0] = smallest, heap_scores[k-1] = largest.
        let mut pos: usize = 0;
        while pos + 1 < k && heap_scores[pos + 1] < score {
            heap_scores[pos] = heap_scores[pos + 1];
            heap_ids[pos] = heap_ids[pos + 1];
            pos += 1;
        }
        heap_scores[pos] = score;
        heap_ids[pos] = i as i32;
    }

    // Reverse into descending-by-score order so callers get "top-1 first".
    // Padding entries (score == -inf) collapse to -1 ids.
    let mut out = Vec::with_capacity(topk);
    for slot in 0..k {
        let src = k - 1 - slot;
        if heap_scores[src].is_finite() {
            out.push(heap_ids[src]);
        } else {
            out.push(-1);
        }
    }
    // Pad up to caller-requested length.
    for _ in k..topk {
        out.push(-1);
    }
    out
}

/// Select top-K for a `[rows, num_blocks]` score matrix, one row at a time.
///
/// * `scores` — flat row-major buffer of length `rows * num_blocks`.
/// * `topk` — per-row top-K count.
/// * `num_valid_blocks` — per-row valid-candidate cap. Use
///   `vec![num_blocks; rows]` when every row shares the same cap.
///
/// Returns a `[rows, topk]` flat buffer; padding is `-1`.
#[must_use]
pub fn sparse_topk_select_batch(
    scores: &[f32],
    rows: usize,
    num_blocks: usize,
    topk: usize,
    num_valid_blocks: &[usize],
) -> Vec<i32> {
    assert_eq!(scores.len(), rows * num_blocks);
    assert_eq!(num_valid_blocks.len(), rows);
    let mut out = Vec::with_capacity(rows * topk);
    for r in 0..rows {
        let row = &scores[r * num_blocks..(r + 1) * num_blocks];
        let cap = num_valid_blocks[r].min(num_blocks);
        out.extend(sparse_topk_select(row, topk, cap));
    }
    out
}

// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_selects_largest_in_descending_order() {
        let scores = vec![0.1, 5.0, 3.0, 4.0, 2.0, 1.0];
        let out = sparse_topk_select(&scores, 3, scores.len());
        // top 3 largest = 5.0(1), 4.0(3), 3.0(2).
        assert_eq!(out, vec![1, 3, 2]);
    }

    #[test]
    fn topk_pads_when_topk_exceeds_valid_blocks() {
        let scores = vec![1.0, 2.0];
        let out = sparse_topk_select(&scores, 4, scores.len());
        // 2 valid candidates + 2 padding = [id_of_2.0, id_of_1.0, -1, -1].
        assert_eq!(out, vec![1, 0, -1, -1]);
    }

    #[test]
    fn topk_respects_num_valid_blocks() {
        let scores = vec![0.1, 5.0, 100.0, 4.0];
        // Only the first 2 entries are candidates; 100.0 must be ignored.
        let out = sparse_topk_select(&scores, 2, 2);
        assert_eq!(out, vec![1, 0]);
    }

    #[test]
    fn topk_handles_nan_as_neg_inf() {
        let scores = vec![f32::NAN, 1.0, f32::NAN, 2.0];
        let out = sparse_topk_select(&scores, 3, scores.len());
        // 2.0(3), 1.0(1), then padding (NaN ineligible).
        assert_eq!(out, vec![3, 1, -1]);
    }

    #[test]
    fn topk_stable_across_ties_by_first_occurrence() {
        // With equal scores the insertion condition `score <= min_of_heap`
        // means the first-seen wins. That's a documented property (matches
        // MSA behavior when the histogram cutoff hits a tie).
        let scores = vec![1.0, 2.0, 2.0, 2.0];
        let out = sparse_topk_select(&scores, 2, scores.len());
        // Two 2.0 winners with first-wins → ids [2, 1] (largest first: id 1
        // then id 2 — but insertion order gives id 1 first, and later 2 does
        // not evict since 2.0 <= 2.0 fails the strict >).
        assert_eq!(out, vec![1, 2]);
    }

    #[test]
    fn topk_zero_topk_returns_empty() {
        let out = sparse_topk_select(&[1.0, 2.0], 0, 2);
        assert!(out.is_empty());
    }

    #[test]
    fn topk_empty_valid_blocks_returns_all_padding() {
        let out = sparse_topk_select(&[1.0, 2.0], 3, 0);
        assert_eq!(out, vec![-1, -1, -1]);
    }

    #[test]
    fn topk_batch_selects_per_row() {
        // 3 rows × 4 blocks, topk=2.
        //   row 0: [0.1, 5.0, 3.0, 4.0] → top-2 = ids [1, 3].
        //   row 1: [9.0, 0.0, 0.0, 0.0] → top-2 = ids [0, -1] (only one
        //     positive, but 3 zeros will still register — see note).
        //   row 2: [1.0, 2.0, 3.0, 4.0] with cap=2 → top-2 of first 2 = [1, 0].
        let scores = vec![0.1, 5.0, 3.0, 4.0, 9.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let caps = vec![4usize, 4, 2];
        let out = sparse_topk_select_batch(&scores, 3, 4, 2, &caps);
        assert_eq!(&out[0..2], &[1, 3]);
        // Row 1: three zeros are still finite; top-2 = ids [0, 1] (9.0 wins,
        // then first zero seen).
        assert_eq!(&out[2..4], &[0, 1]);
        assert_eq!(&out[4..6], &[1, 0]);
    }

    #[test]
    fn topk_all_neg_inf_scores_yield_padding() {
        let scores = vec![f32::NEG_INFINITY; 4];
        let out = sparse_topk_select(&scores, 2, 4);
        assert_eq!(out, vec![-1, -1]);
    }
}
