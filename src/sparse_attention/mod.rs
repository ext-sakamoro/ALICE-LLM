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

pub mod index;
pub mod proxy;
pub mod topk;
pub mod types;

pub use index::build_kvouter_index;
pub use proxy::compute_proxy_block_max_scores;
pub use topk::{sparse_topk_select, sparse_topk_select_batch};
pub use types::{BlockTables, CuSeqlensQ, KvOuterIndex, SparseAttentionError, SparseSelection};
