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
    clippy::unreadable_literal
)]

pub mod attention;
pub mod batch_inference;
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
