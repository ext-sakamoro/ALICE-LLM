//! High-level adapter that lets llama3-style dense KV-cache callers drive
//! `kvouter_attention` without hand-rolling the paged-cache repack, block
//! selection, and one-shot API plumbing themselves.
//!
//! ## Architectural fit
//!
//! This adapter targets **standard Llama-3-style attention** — models that
//! keep K / V as *dense* tensors of shape `[seq_len, hkv, head_dim]` per
//! layer, accessed one KV token at a time. In `src/llama3.rs` this is the
//! shape consumed by `gqa_attention` (Qwen 3.5, Llama 3, Bonsai, Elyza,
//! Gemma, and every other GQA / MQA / MHA arch that uses the standard
//! `KvCache`).
//!
//! **Not fit** for Kimi K3 (`kimi_k3_gated_mla_step`): K3 uses MLA
//! (Multi-head Latent Attention), which stores a LoRA-compressed *latent*
//! KV that gets decompressed on the fly per attention step. A separate
//! `mla_bridge` would be needed to sparse-attend over MLA's latent
//! representation; keeping K3's compression advantage while sparsifying
//! is a redesign problem, not an adapter problem. See
//! `memory/project_alice_llm_sparse_attention.md` for the rationale.
//!
//! ## Why this exists (Phase MSA.5.5 / 5.6)
//!
//! Wiring dense KV cache to [`super::kvouter_attention`] requires four
//! steps every call site would otherwise repeat:
//!
//! 1. repack dense K / V into a paged layout
//!    `[num_pages, hkv, page_size, head_dim]` with block tables;
//! 2. build a `SparseSelection` (either dense = every block, or top-K via
//!    the dense proxy pass);
//! 3. build the ancillary `CuSeqlensQ` / `BlockTables` view;
//! 4. invoke `kvouter_attention` with the correct softmax scale.
//!
//! [`llama3_sparse_attention`] does all four in one call and returns the
//! `[Tq, Hq, head_dim]` output ready to be consumed by the residual add.
//!
//! ## Env-gated activation
//!
//! `gqa_attention` in `llama3.rs` reads `ALICE_SPARSE_TOPK` at call time
//! and, when set to a positive integer, dispatches this adapter with
//! `topk = <env>`. `ALICE_SPARSE_TOPK=0` selects every sparse block and
//! is arithmetically equivalent to dense attention modulo FP
//! re-association (the parity guarantee tested below). Larger values pick
//! only the top-K KV blocks per query.
//!
//! ```ignore
//! // Actual code in llama3.rs::gqa_attention (roughly):
//! if let Some(topk) = std::env::var("ALICE_SPARSE_TOPK")
//!     .ok().and_then(|s| s.parse::<usize>().ok())
//! {
//!     let cfg = BridgeConfig { hq, hkv, head_dim, block_size: 64, page_size: 64,
//!                              softmax_scale, causal: false, topk };
//!     let kv_view = DenseKvCacheView { k: &k_dense, v: &v_dense,
//!                                       seq_len: used_len, hkv, head_dim };
//!     if let Ok(out) = llama3_sparse_attention(q_buf, &kv_view, 1, &cfg) {
//!         attn_out.copy_from_slice(&out);
//!         return;
//!     }
//!     // adapter rejected (softcap / geometry) → fall back to dense
//! }
//! ```

use super::index::build_kvouter_index;
use super::proxy::compute_proxy_block_max_scores;
use super::topk::sparse_topk_select_batch;
use super::types::{BlockTables, CuSeqlensQ, SparseAttentionError, SparseSelection};
use super::{kvouter_attention, kvouter_forward, lse_combine};

// Types
// ---------------------------------------------------------------------------

/// Read-only view into a dense KV cache slice for one layer.
///
/// Layout: `[seq_len, hkv, head_dim]` row-major.
#[derive(Debug, Clone, Copy)]
pub struct DenseKvCacheView<'a> {
    /// Key tensor of length `seq_len * hkv * head_dim`.
    pub k: &'a [f32],
    /// Value tensor of length `seq_len * hkv * head_dim`.
    pub v: &'a [f32],
    /// KV positions written so far.
    pub seq_len: usize,
    /// KV heads.
    pub hkv: usize,
    /// Head dimension.
    pub head_dim: usize,
}

/// Adapter configuration.
///
/// Only `block_size % page_size == 0` and `hq % hkv == 0` are validated here;
/// the deeper geometry checks happen inside [`kvouter_attention`].
#[derive(Debug, Clone, Copy)]
pub struct BridgeConfig {
    /// Total query heads.
    pub hq: usize,
    /// KV heads.
    pub hkv: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Sparse-block size (must be a multiple of `page_size`).
    pub block_size: usize,
    /// Physical page size for the paged KV cache.
    pub page_size: usize,
    /// Attention scale (usually `1.0 / sqrt(head_dim)`).
    pub softmax_scale: f32,
    /// Apply right-aligned causal mask.
    pub causal: bool,
    /// Number of KV blocks each query attends to. `0` (or `>= num_blocks`)
    /// selects every block — the "dense fallback" that lets the adapter
    /// return the same output as a naive scaled-dot-product attention.
    pub topk: usize,
}

// Public API
// ---------------------------------------------------------------------------

/// Run sparse KV-outer attention over a dense (llama3-style) KV cache.
///
/// * `q` — flat `[tq * hq * head_dim]` packed queries.
/// * `kv` — dense K / V cache view (`[seq_len, hkv, head_dim]`).
/// * `tq` — packed query length (typically `1` for autoregressive decode,
///   `prefill_len` for prefill).
/// * `cfg` — geometry + sparsity knobs.
///
/// Returns `[tq * hq * head_dim]` FP32 attention output.
pub fn llama3_sparse_attention(
    q: &[f32],
    kv: &DenseKvCacheView<'_>,
    tq: usize,
    cfg: &BridgeConfig,
) -> Result<Vec<f32>, SparseAttentionError> {
    if cfg.hq == 0 || cfg.hkv == 0 || cfg.hq % cfg.hkv != 0 {
        return Err(SparseAttentionError::HeadCountMismatch {
            hq: cfg.hq,
            hkv: cfg.hkv,
        });
    }
    if cfg.head_dim == 0 || cfg.block_size == 0 || cfg.page_size == 0 {
        return Err(SparseAttentionError::BlockPageMismatch {
            block_size: cfg.block_size,
            page_size: cfg.page_size,
        });
    }
    if cfg.block_size % cfg.page_size != 0 {
        return Err(SparseAttentionError::BlockPageMismatch {
            block_size: cfg.block_size,
            page_size: cfg.page_size,
        });
    }
    if kv.hkv != cfg.hkv || kv.head_dim != cfg.head_dim {
        return Err(SparseAttentionError::HeadDimMismatch);
    }
    let expected_kv = kv.seq_len * kv.hkv * kv.head_dim;
    if kv.k.len() != expected_kv || kv.v.len() != expected_kv {
        return Err(SparseAttentionError::ShapeMismatch {
            what: "DenseKvCacheView k/v",
            expected: expected_kv,
            got: kv.k.len().max(kv.v.len()),
        });
    }
    if q.len() != tq * cfg.hq * cfg.head_dim {
        return Err(SparseAttentionError::ShapeMismatch {
            what: "q",
            expected: tq * cfg.hq * cfg.head_dim,
            got: q.len(),
        });
    }

    // --- Repack dense K/V into paged layout ---------------------------
    let num_pages = kv.seq_len.div_ceil(cfg.page_size).max(1);
    let msb = num_pages / (cfg.block_size / cfg.page_size);
    if msb == 0 {
        return Err(SparseAttentionError::BlockPageMismatch {
            block_size: cfg.block_size,
            page_size: cfg.page_size,
        });
    }
    let page_stride = cfg.hkv * cfg.page_size * cfg.head_dim;
    let mut k_pages = vec![0.0f32; num_pages * page_stride];
    let mut v_pages = vec![0.0f32; num_pages * page_stride];
    for seq_pos in 0..kv.seq_len {
        let page_id = seq_pos / cfg.page_size;
        let pos_in_page = seq_pos % cfg.page_size;
        for h in 0..cfg.hkv {
            let dense_off = (seq_pos * cfg.hkv + h) * cfg.head_dim;
            let paged_off = page_id * page_stride
                + h * cfg.page_size * cfg.head_dim
                + pos_in_page * cfg.head_dim;
            k_pages[paged_off..paged_off + cfg.head_dim]
                .copy_from_slice(&kv.k[dense_off..dense_off + cfg.head_dim]);
            v_pages[paged_off..paged_off + cfg.head_dim]
                .copy_from_slice(&kv.v[dense_off..dense_off + cfg.head_dim]);
        }
    }

    // --- Build block_tables (single-batch, identity mapping) ----------
    let block_tables_data: Vec<i32> = (0..num_pages as i32).collect();
    let block_tables = BlockTables::new(block_tables_data, 1, num_pages)?;
    let cu_seqlens_q = CuSeqlensQ::new(vec![0, tq as i64])?;
    let used_kv_lens = vec![kv.seq_len as i32];

    // --- Build SparseSelection ---------------------------------------
    let effective_topk = if cfg.topk == 0 || cfg.topk >= msb {
        msb
    } else {
        cfg.topk
    };

    let selection = if effective_topk == msb {
        // Dense fallback: every query picks every block.
        let mut sel_flat = Vec::with_capacity(tq * cfg.hkv * msb);
        for _i in 0..tq {
            for _h in 0..cfg.hkv {
                for r in 0..msb {
                    sel_flat.push(r as i32);
                }
            }
        }
        SparseSelection::new(sel_flat, tq, cfg.hkv, msb)?
    } else {
        // Cheap proxy = the first Q lane of each KV head.
        let qhead = cfg.hq / cfg.hkv;
        let mut proxy_q = Vec::with_capacity(tq * cfg.hkv * cfg.head_dim);
        for i in 0..tq {
            for h in 0..cfg.hkv {
                let hq_abs = h * qhead;
                let base = (i * cfg.hq + hq_abs) * cfg.head_dim;
                proxy_q.extend_from_slice(&q[base..base + cfg.head_dim]);
            }
        }
        let scores = compute_proxy_block_max_scores(
            &proxy_q,
            &k_pages,
            &block_tables,
            &cu_seqlens_q,
            Some(&used_kv_lens),
            cfg.hkv,
            cfg.head_dim,
            cfg.block_size,
            cfg.page_size,
        )?;
        let rows = tq * cfg.hkv;
        let caps = vec![msb; rows];
        let sel_flat = sparse_topk_select_batch(&scores, rows, msb, effective_topk, &caps);
        SparseSelection::new(sel_flat, tq, cfg.hkv, effective_topk)?
    };

    // --- Run the sparse attention pipeline (via the one-shot API when the
    // caller wants dense parity; otherwise chain index → forward → combine).
    if effective_topk == msb {
        kvouter_attention(
            q,
            &k_pages,
            &v_pages,
            &selection,
            &block_tables,
            &cu_seqlens_q,
            Some(&used_kv_lens),
            cfg.hq,
            cfg.hkv,
            cfg.head_dim,
            cfg.block_size,
            cfg.page_size,
            cfg.softmax_scale,
            cfg.causal,
        )
    } else {
        let idx = build_kvouter_index(
            &selection,
            &block_tables,
            &cu_seqlens_q,
            Some(&used_kv_lens),
            cfg.block_size,
            cfg.page_size,
        )?;
        let partials = kvouter_forward(
            q,
            &k_pages,
            &v_pages,
            &idx,
            &block_tables,
            &cu_seqlens_q,
            Some(&used_kv_lens),
            cfg.hq,
            cfg.hkv,
            cfg.head_dim,
            cfg.block_size,
            cfg.page_size,
            cfg.softmax_scale,
            cfg.causal,
        )?;
        lse_combine(&partials, &idx, cfg.hq)
    }
}

// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::scaled_dot_product_attention;

    /// Naive dense scaled-dot-product attention for a single query row and
    /// GQA head layout `[tq, hq, head_dim]` × KV `[seq_len, hkv, head_dim]`.
    /// Delegates each per-head slice to
    /// `crate::attention::scaled_dot_product_attention`.
    fn naive_dense_reference(
        q: &[f32],
        kv: &DenseKvCacheView<'_>,
        tq: usize,
        hq: usize,
    ) -> Vec<f32> {
        let head_dim = kv.head_dim;
        let hkv = kv.hkv;
        let qhead = hq / hkv;
        let mut out = vec![0.0f32; tq * hq * head_dim];

        for h in 0..hkv {
            // Gather k / v rows for this KV head as [seq_len][head_dim].
            let k_rows: Vec<Vec<f32>> = (0..kv.seq_len)
                .map(|s| {
                    let off = (s * hkv + h) * head_dim;
                    kv.k[off..off + head_dim].to_vec()
                })
                .collect();
            let v_rows: Vec<Vec<f32>> = (0..kv.seq_len)
                .map(|s| {
                    let off = (s * hkv + h) * head_dim;
                    kv.v[off..off + head_dim].to_vec()
                })
                .collect();
            for qh in 0..qhead {
                let hq_abs = h * qhead + qh;
                // Gather q rows for this query head as [tq][head_dim].
                let q_rows: Vec<Vec<f32>> = (0..tq)
                    .map(|i| {
                        let off = (i * hq + hq_abs) * head_dim;
                        q[off..off + head_dim].to_vec()
                    })
                    .collect();
                let out_rows = scaled_dot_product_attention(&q_rows, &k_rows, &v_rows, None);
                for (i, row) in out_rows.into_iter().enumerate() {
                    let dst_off = (i * hq + hq_abs) * head_dim;
                    out[dst_off..dst_off + head_dim].copy_from_slice(&row);
                }
            }
        }
        out
    }

    fn fixture_gqa() -> (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        usize,
        usize,
        usize,
        usize,
        usize,
    ) {
        // tq=3, hq=4, hkv=2 (qhead=2), head_dim=8, seq_len=12.
        let tq = 3;
        let hq = 4;
        let hkv = 2;
        let head_dim = 8;
        let seq_len = 12;
        let q: Vec<f32> = (0..tq * hq * head_dim)
            .map(|i| ((i as f32) * 0.05).sin())
            .collect();
        let k: Vec<f32> = (0..seq_len * hkv * head_dim)
            .map(|i| ((i as f32) * 0.07).cos())
            .collect();
        let v: Vec<f32> = (0..seq_len * hkv * head_dim)
            .map(|i| ((i as f32) * 0.03).sin())
            .collect();
        (q, k, v, tq, hq, hkv, head_dim, seq_len)
    }

    #[test]
    fn bridge_dense_fallback_matches_naive_dense() {
        let (q, k, v, tq, hq, hkv, head_dim, seq_len) = fixture_gqa();
        let kv = DenseKvCacheView {
            k: &k,
            v: &v,
            seq_len,
            hkv,
            head_dim,
        };
        let cfg = BridgeConfig {
            hq,
            hkv,
            head_dim,
            block_size: 4,
            page_size: 4,
            softmax_scale: 1.0f32 / (head_dim as f32).sqrt(),
            causal: false,
            topk: 0, // full fallback
        };
        let bridge_out = llama3_sparse_attention(&q, &kv, tq, &cfg).unwrap();
        let dense_out = naive_dense_reference(&q, &kv, tq, hq);
        assert_eq!(bridge_out.len(), dense_out.len());
        for (i, (a, b)) in bridge_out.iter().zip(&dense_out).enumerate() {
            let rel = (a - b).abs() / b.abs().max(1e-6);
            assert!(rel < 1e-4, "elem {i}: bridge={a}, dense={b}, rel={rel}");
        }
    }

    #[test]
    fn bridge_topk_equal_to_num_blocks_matches_naive_dense() {
        // Explicit topk == msb should be treated identically to topk == 0.
        let (q, k, v, tq, hq, hkv, head_dim, seq_len) = fixture_gqa();
        let kv = DenseKvCacheView {
            k: &k,
            v: &v,
            seq_len,
            hkv,
            head_dim,
        };
        // seq_len=12, block_size=4 → 3 blocks.
        let cfg = BridgeConfig {
            hq,
            hkv,
            head_dim,
            block_size: 4,
            page_size: 4,
            softmax_scale: 1.0f32 / (head_dim as f32).sqrt(),
            causal: false,
            topk: 3,
        };
        let bridge_out = llama3_sparse_attention(&q, &kv, tq, &cfg).unwrap();
        let dense_out = naive_dense_reference(&q, &kv, tq, hq);
        for (i, (a, b)) in bridge_out.iter().zip(&dense_out).enumerate() {
            let rel = (a - b).abs() / b.abs().max(1e-6);
            assert!(rel < 1e-4, "elem {i}: bridge={a}, dense={b}, rel={rel}");
        }
    }

    #[test]
    fn bridge_topk_partial_produces_bounded_output() {
        // With topk < num_blocks the sparse output diverges from dense; we
        // only verify it runs and returns finite values in the expected shape.
        let (q, k, v, tq, hq, hkv, head_dim, seq_len) = fixture_gqa();
        let kv = DenseKvCacheView {
            k: &k,
            v: &v,
            seq_len,
            hkv,
            head_dim,
        };
        let cfg = BridgeConfig {
            hq,
            hkv,
            head_dim,
            block_size: 4,
            page_size: 4,
            softmax_scale: 1.0f32 / (head_dim as f32).sqrt(),
            causal: false,
            topk: 2, // top-2 of 3 blocks
        };
        let out = llama3_sparse_attention(&q, &kv, tq, &cfg).unwrap();
        assert_eq!(out.len(), tq * hq * head_dim);
        for (i, v) in out.iter().enumerate() {
            assert!(v.is_finite(), "non-finite output at {i}: {v}");
        }
    }

    #[test]
    fn bridge_rejects_shape_mismatch() {
        // q length doesn't match tq * hq * head_dim.
        let kv_k = vec![0.0f32; 4 * 2 * 8]; // seq_len=4, hkv=2, head_dim=8
        let kv_v = kv_k.clone();
        let kv = DenseKvCacheView {
            k: &kv_k,
            v: &kv_v,
            seq_len: 4,
            hkv: 2,
            head_dim: 8,
        };
        let cfg = BridgeConfig {
            hq: 4,
            hkv: 2,
            head_dim: 8,
            block_size: 4,
            page_size: 4,
            softmax_scale: 1.0,
            causal: false,
            topk: 0,
        };
        // Correct q length would be 1 * 4 * 8 = 32; supply 30.
        let err = llama3_sparse_attention(&vec![0.0f32; 30], &kv, 1, &cfg).unwrap_err();
        matches!(err, SparseAttentionError::ShapeMismatch { .. });
    }

    #[test]
    fn bridge_rejects_head_dim_mismatch() {
        let kv_k = vec![0.0f32; 4 * 2 * 8];
        let kv_v = kv_k.clone();
        let kv = DenseKvCacheView {
            k: &kv_k,
            v: &kv_v,
            seq_len: 4,
            hkv: 2,
            head_dim: 8,
        };
        let cfg = BridgeConfig {
            hq: 4,
            hkv: 2,
            head_dim: 16, // mismatch with kv.head_dim = 8
            block_size: 4,
            page_size: 4,
            softmax_scale: 1.0,
            causal: false,
            topk: 0,
        };
        let err = llama3_sparse_attention(&vec![0.0f32; 1 * 4 * 16], &kv, 1, &cfg).unwrap_err();
        matches!(err, SparseAttentionError::HeadDimMismatch);
    }
}
