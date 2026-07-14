//! ALICE-LLM: Pure Rust LLM inference engine.

#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::doc_markdown,
    clippy::wildcard_imports,
    clippy::too_many_lines,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::similar_names,
    clippy::return_self_not_must_use,
    clippy::float_cmp,
    clippy::suboptimal_flops,
    clippy::items_after_statements,
    clippy::many_single_char_names,
    clippy::manual_midpoint,
    clippy::approx_constant,
    clippy::unreadable_literal,
    clippy::unnecessary_wraps,
    clippy::match_same_arms,
    clippy::needless_range_loop,
    clippy::manual_clamp,
    clippy::useless_conversion,
    clippy::used_underscore_binding,
    clippy::needless_pass_by_value,
    clippy::missing_const_for_fn,
    clippy::unused_self,
    clippy::option_if_let_else,
    clippy::inline_always,
    clippy::trivially_copy_pass_by_ref,
    clippy::or_fun_call,
    clippy::too_many_arguments,
    clippy::unnecessary_map_or,
    clippy::redundant_pub_crate,
    clippy::branches_sharing_code,
    clippy::doc_link_with_quotes,
    clippy::assigning_clones,
    clippy::redundant_closure_for_method_calls,
    clippy::significant_drop_tightening,
    clippy::useless_let_if_seq,
    clippy::use_self,
    clippy::collection_is_never_read
)]
#![allow(
    rustdoc::broken_intra_doc_links,
    rustdoc::invalid_html_tags,
    rustdoc::bare_urls
)]

pub mod attention;
pub mod batch_inference;
pub mod deepseek_streaming;
pub mod gguf;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod kv_cache;
pub mod linalg;
pub mod llama3;
pub mod matrix;
pub mod prelude;
pub mod quantization;
pub mod rope;
pub mod sampling;
pub mod tokenizer;
pub mod training;

#[cfg(test)]
mod integration_tests;

pub use matrix::{dot_flat, Matrix};

pub use crate::attention::*;
pub use crate::batch_inference::*;
pub use crate::kv_cache::*;
pub use crate::linalg::*;
pub use crate::quantization::*;
pub use crate::rope::*;
pub use crate::sampling::*;
pub use crate::tokenizer::*;
