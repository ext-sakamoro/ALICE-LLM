//! Grammar-constrained decoding (Phase X.8 Guided Generation).
//!
//! Opt-in via the `grammar` feature. Provides the building blocks for
//! constraining LLM sampling to follow a formal grammar so that the model can
//! only emit strings recognised by the target language (JSON schema, an
//! embedded DSL such as ALICE-LOL, tool-call payloads, etc.).
//!
//! Currently implemented:
//!
//! - [`gbnf`] (B-1) — llama.cpp-compatible GBNF parser (subset).
//! - [`fsm`] (B-2) — `Grammar` → finite-state machine driving constrained
//!   decoding.
//!
//! Planned:
//!
//! - Sampling integration (B-3) — `mask_logits_by_grammar(&fsm, tokenizer,
//!   logits)` that zero-masks tokens which cannot advance the current parse
//!   state.

pub mod fsm;
pub mod gbnf;

pub use fsm::{CharSet, Fsm, FsmError, DEFAULT_MAX_DEPTH};
pub use gbnf::{parse_gbnf, Alternative, CharClass, GbnfError, Grammar, Symbol};
