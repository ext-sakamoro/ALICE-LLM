//! sparse attention (KV-outer block-sparse).
//!
//! Pure-Rust from-scratch port of the algorithm described in MiniMax Sparse
//! Attention (MSA; MiniMax-AI/MSA, MIT) and Fireworks AI's M3 KV-outer sparse
//! attention (fw-ai/minimax-kernels, Apache-2.0). See
//! `docs/m3-sparse-attention.md` in the latter repo for the reference
//! pipeline; the algorithm is reimplemented here without vendoring any of the
//! upstream CUDA / CuTe-DSL kernels — we only take the mathematical
//! formulation and the tensor contracts. See the crate NOTICE for
//! attribution.
//!
//! # Pipeline (implemented in phases)
//!
//! * `types`           — tensor contracts (`SparseSelection`, `BlockTables`,
//!                       `CuSeqlensQ`, `KvOuterIndex`) and error type.
//! * `index`           — `build_kvouter_index`: KV-outer CSR inverse index
//!                       (Phase MSA.1).
//! * `topk` / `proxy`  — top-K KV-block selector + dense proxy pass
//!                       (Phase MSA.2, not yet present).
//! * `scheduler` /
//!   `forward` /
//!   `combine`         — load-balance scheduler, KV-outer forward
//!                       (partial output + online softmax), and LSE
//!                       combine (Phase MSA.3, not yet present).

pub mod combine;
pub mod forward;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod index;
pub mod proxy;
pub mod scheduler;
mod simd;
pub mod topk;
pub mod types;

pub use combine::lse_combine;
pub use forward::{kvouter_forward, ForwardPartials};
pub use index::build_kvouter_index;
pub use proxy::compute_proxy_block_max_scores;
pub use scheduler::{build_fixed_schedule, enumerate_work_units, WorkSplit};
pub use topk::{sparse_topk_select, sparse_topk_select_batch};
pub use types::{BlockTables, CuSeqlensQ, KvOuterIndex, SparseAttentionError, SparseSelection};

// Public one-shot API
// ---------------------------------------------------------------------------

/// End-to-end KV-outer sparse attention: index → forward → combine.
///
/// Rust-CPU one-shot equivalent of the upstream `kvouter_attention` entry
/// point in `fw-ai/minimax-kernels`. Callers that already own a top-K
/// `SparseSelection` can invoke this in one call instead of chaining
/// [`build_kvouter_index`], [`kvouter_forward`], and [`lse_combine`]
/// manually.
///
/// # Arguments
///
/// * `q` — `[Tq, Hq, head_dim]` packed queries.
/// * `k_pages`, `v_pages` — `[num_pages, Hkv, page_size, head_dim]` paged K/V.
/// * `selected` — top-K sparse-block ids per `(query, kv_head)`.
/// * `block_tables` — `[B, max_pages]` logical→physical page ids.
/// * `cu_seqlens_q` — packed query prefix sums.
/// * `used_kv_lens` — optional `[B]` real KV lengths.
/// * `hq`, `hkv`, `head_dim`, `block_size`, `page_size` — geometry.
/// * `softmax_scale` — usually `1.0 / sqrt(head_dim)`.
/// * `causal` — apply right-aligned per-sequence causal mask.
///
/// Returns a `[Tq * Hq * head_dim]` FP32 output buffer. Query rows with no
/// valid partial are zero-filled.
#[allow(clippy::too_many_arguments)]
pub fn kvouter_attention(
    q: &[f32],
    k_pages: &[f32],
    v_pages: &[f32],
    selected: &SparseSelection,
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
) -> Result<Vec<f32>, SparseAttentionError> {
    let idx = build_kvouter_index(
        selected,
        block_tables,
        cu_seqlens_q,
        used_kv_lens,
        block_size,
        page_size,
    )?;
    let partials = kvouter_forward(
        q,
        k_pages,
        v_pages,
        &idx,
        block_tables,
        cu_seqlens_q,
        used_kv_lens,
        hq,
        hkv,
        head_dim,
        block_size,
        page_size,
        softmax_scale,
        causal,
    )?;
    lse_combine(&partials, &idx, hq)
}

#[cfg(test)]
mod api_tests {
    use super::*;

    #[test]
    fn kvouter_attention_one_shot_matches_manual_chain() {
        // Smallest possible case: Tq=1, Hq=Hkv=1, head_dim=2, block=2, page=2,
        // 2 pages, topk=2 selects everything.
        let tq = 1;
        let hq = 1;
        let hkv = 1;
        let head_dim = 2;
        let page_size = 2;
        let block_size = 2;
        let num_pages = 2;

        let q = vec![1.0f32, 0.0];
        let k = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0, 0.0];
        let v = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let sel = SparseSelection::new(vec![0, 1], tq, hkv, 2).unwrap();
        let tbl = BlockTables::new(vec![0, 1], 1, 2).unwrap();
        let cu = CuSeqlensQ::new(vec![0, tq as i64]).unwrap();
        let used = vec![num_pages as i32 * page_size as i32];

        let softmax_scale = 1.0f32 / (head_dim as f32).sqrt();

        let one_shot = kvouter_attention(
            &q,
            &k,
            &v,
            &sel,
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

        // Manual chain — should produce identical output.
        let idx = build_kvouter_index(&sel, &tbl, &cu, Some(&used), block_size, page_size).unwrap();
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
        let manual = lse_combine(&partials, &idx, hq).unwrap();

        assert_eq!(one_shot.len(), manual.len());
        for (a, b) in one_shot.iter().zip(manual.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
