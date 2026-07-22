//! Grammar-constrained decoding (Phase X.8 Guided Generation).
//!
//! Opt-in via the `grammar` feature. Provides the building blocks for
//! constraining LLM sampling to follow a formal grammar so that the model can
//! only emit strings recognised by the target language (JSON schema, an
//! embedded DSL such as ALICE-LOL, tool-call payloads, etc.).
//!
//! Currently implemented (B-1):
//!
//! - [`gbnf`] — llama.cpp-compatible GBNF parser (subset).
//!
//! Planned:
//!
//! - `fsm` (B-2) — `Grammar` → finite-state machine builder.
//! - Sampling integration (B-3) — `logits_mask(&fsm, tokenizer, logits)` that
//!   zero-masks tokens which cannot advance the current parse state.

pub mod gbnf;

pub use gbnf::{parse_gbnf, Alternative, CharClass, GbnfError, Grammar, Symbol};
