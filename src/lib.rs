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
pub mod diag_clock;
pub mod gguf;
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "grammar")]
pub mod grammar;
#[cfg(feature = "imatrix")]
pub mod imatrix;
pub mod iq1_s;
pub mod kv_cache;
pub mod lawmath;
pub mod linalg;
pub mod llama3;
pub mod matrix;
pub mod prelude;
pub mod quantization;
pub mod rope;
pub mod sampling;
pub mod semantics;
pub mod server_cli;
pub mod sparse_attention;
pub mod speculative_dspark;
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

/// A KV cache stamped with the identifier before the fold gained "sampling choices" loads, since
/// what a cache holds is computed to the same bits. `scripts/id_move_check.py` runs this test, and
/// runs it again with the value taken out of the previous list, where it has to fail. The tool
/// selects a library test by its exact name, so this one stays at the crate root.
#[cfg(test)]
#[test]
fn a_stamp_of_the_previous_identifier_is_accepted() {
    use crate::llama3::{kv_check_semantics, KvCacheSemantics};
    let previous: [u8; 32] = [
        0xc1, 0xee, 0xe9, 0x6c, 0x2d, 0xe4, 0x9c, 0xfd, 0x11, 0x4a, 0xc4, 0x8a, 0xb8, 0xfe, 0xf3,
        0xd8, 0xe7, 0xbd, 0x72, 0x52, 0x2b, 0x8d, 0x11, 0x60, 0xef, 0x42, 0x6f, 0xdc, 0xb9, 0x04,
        0xea, 0xdf,
    ];
    let got = kv_check_semantics(previous, &crate::semantics::LLM_SEMANTICS_ID);
    assert!(matches!(got, Ok(KvCacheSemantics::VerifiedPrevious(p)) if p == previous));
}

#[cfg(test)]
#[test]
fn a_stamp_of_the_identifier_before_the_function_identifiers_is_accepted() {
    use crate::llama3::{kv_check_semantics, KvCacheSemantics};
    let previous: [u8; 32] = [
        0xcd, 0x0f, 0x62, 0x4f, 0xc7, 0x21, 0x08, 0x62, 0x70, 0x35, 0x02, 0xa5, 0xea, 0xd1, 0xb6,
        0xba, 0xaf, 0xe6, 0x5a, 0xf5, 0xf9, 0x24, 0x0a, 0xb6, 0x3a, 0xb2, 0xe9, 0x54, 0xbb, 0x78,
        0x75, 0xdc,
    ];
    let got = kv_check_semantics(previous, &crate::semantics::LLM_SEMANTICS_ID);
    assert!(matches!(got, Ok(KvCacheSemantics::VerifiedPrevious(p)) if p == previous));
}
