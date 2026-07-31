//! Sparse attention (KV-outer) end-to-end demo.
//!
//! Runs the dense-proxy → top-K → sparse-forward → LSE-combine pipeline
//! on a synthetic single-batch workload and prints the resulting attention
//! output alongside a naïve dense reference so a reader can eyeball the
//! agreement.
//!
//! ```
//! cargo run --example sparse_attention_demo --release
//! ```

use alice_llm::sparse_attention::{
    build_kvouter_index, compute_proxy_block_max_scores, kvouter_forward, lse_combine,
    sparse_topk_select_batch, BlockTables, CuSeqlensQ, SparseSelection,
};

fn main() {
    // Geometry.
    let tq = 4usize;
    let hq = 4usize;
    let hkv = 2usize;
    let head_dim = 8usize;
    let page_size = 4usize;
    let block_size = 4usize;
    let max_pages = 8usize;
    let num_pages = 8usize;
    // Selecting every sparse block (topk == msb) makes the pipeline
    // arithmetically equivalent to dense attention. Set topk < msb to see
    // how much accuracy the top-K approximation gives up.
    let topk = 8usize;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // Deterministic tensors.
    let q: Vec<f32> = (0..tq * hq * head_dim)
        .map(|i| ((i as f32) * 0.09).sin())
        .collect();
    let k: Vec<f32> = (0..num_pages * hkv * page_size * head_dim)
        .map(|i| ((i as f32) * 0.11).cos())
        .collect();
    let v: Vec<f32> = (0..num_pages * hkv * page_size * head_dim)
        .map(|i| ((i as f32) * 0.07).sin())
        .collect();

    // One batch, all 8 KV pages populated.
    let tbl = BlockTables::new((0..max_pages as i32).collect(), 1, max_pages).unwrap();
    let cu = CuSeqlensQ::new(vec![0, tq as i64]).unwrap();
    let used = vec![(num_pages * page_size) as i32];

    // ------ Stage 1: dense proxy pass (cheap Q slice = one lane per KV head).
    // Real deployments use a low-rank / low-precision proxy; here we take
    // the *first* Q lane of each KV head as the "cheap" slice.
    let qhead = hq / hkv;
    let mut proxy_q = Vec::with_capacity(tq * hkv * head_dim);
    for i in 0..tq {
        for h in 0..hkv {
            let hq_abs = h * qhead;
            let base = (i * hq + hq_abs) * head_dim;
            proxy_q.extend_from_slice(&q[base..base + head_dim]);
        }
    }
    let msb = max_pages / (block_size / page_size);
    let scores = compute_proxy_block_max_scores(
        &proxy_q,
        &k,
        &tbl,
        &cu,
        Some(&used),
        hkv,
        head_dim,
        block_size,
        page_size,
    )
    .unwrap();
    assert_eq!(scores.len(), tq * hkv * msb);

    // ------ Stage 2: top-K KV block selection per (query, kv_head).
    // Flatten scores to `[tq*hkv, msb]` and select topk per row.
    let rows = tq * hkv;
    let caps = vec![msb; rows];
    let selected_flat = sparse_topk_select_batch(&scores, rows, msb, topk, &caps);
    let selection = SparseSelection::new(selected_flat, tq, hkv, topk).unwrap();

    // ------ Stage 3: KV-outer sparse attention.
    let idx =
        build_kvouter_index(&selection, &tbl, &cu, Some(&used), block_size, page_size).unwrap();
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
        scale,
        false,
    )
    .unwrap();
    let sparse_out = lse_combine(&partials, &idx, hq).unwrap();

    // ------ Reference: dense scaled-dot-product over the full KV.
    let dense_out =
        naive_dense_reference(&q, &k, &v, &tbl, &cu, &used, hq, hkv, head_dim, page_size);

    // Report.
    println!("--- KV-outer sparse attention demo ---");
    println!("Tq={tq}, Hq={hq}, Hkv={hkv}, head_dim={head_dim}");
    println!("block_size={block_size}, page_size={page_size}, num_pages={num_pages}");
    println!("selected top-{topk} of {msb} sparse blocks per (query, kv_head)");
    println!();
    let mut max_diff = 0.0f32;
    let mut max_rel = 0.0f32;
    for (s, d) in sparse_out.iter().zip(dense_out.iter()) {
        let diff = (s - d).abs();
        let rel = diff / d.abs().max(1e-6);
        max_diff = max_diff.max(diff);
        max_rel = max_rel.max(rel);
    }
    println!("max abs diff vs dense reference: {max_diff:.6e}");
    println!("max rel diff vs dense reference: {max_rel:.6e}");
    // With topk == msb we should be bit-close to dense.
    assert!(
        max_rel < 1e-4,
        "sparse output diverged from dense reference (rel={max_rel})"
    );
    println!("OK — sparse attention pipeline matches dense reference.");
    println!();
    println!("first output row (query 0, head 0):");
    for d in 0..head_dim {
        println!(
            "  d{d}: sparse={:+.6}  dense={:+.6}",
            sparse_out[d], dense_out[d]
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn naive_dense_reference(
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
) -> Vec<f32> {
    let tq = cu_seqlens_q.total_tq();
    let qhead = hq / hkv;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let page_stride = hkv * page_size * head_dim;
    let mut out = vec![0.0f32; tq * hq * head_dim];

    for i in 0..tq {
        // Single batch demo → batch id 0.
        let b = 0usize;
        let seq_len_b = used_kv_lens[b].max(0) as usize;
        for h in 0..hkv {
            for qh in 0..qhead {
                let hq_abs = h * qhead + qh;
                let q_off = (i * hq + hq_abs) * head_dim;
                let q_vec = &q[q_off..q_off + head_dim];

                let mut scores = Vec::with_capacity(seq_len_b);
                let mut m = f32::NEG_INFINITY;
                let mut k_flat = Vec::with_capacity(seq_len_b * head_dim);
                let mut v_flat = Vec::with_capacity(seq_len_b * head_dim);
                for kv_pos in 0..seq_len_b {
                    let page_local = kv_pos / page_size;
                    let pos_in_page = kv_pos % page_size;
                    let page_id = block_tables.get(b, page_local);
                    if page_id < 0 {
                        continue;
                    }
                    let base = page_id as usize * page_stride + h * page_size * head_dim;
                    let k_row = &k_pages
                        [base + pos_in_page * head_dim..base + (pos_in_page + 1) * head_dim];
                    let v_row = &v_pages
                        [base + pos_in_page * head_dim..base + (pos_in_page + 1) * head_dim];
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
