//! ALICE-Dynamic-v1 imatrix pipeline (Phase I.0 + I.3).
//!
//! # Scope
//!
//! This module implements the *tier-decision* portion of the imatrix pipeline
//! only. Given a GGUF model and a target output size, it emits a
//! `layer_assignments` map describing which quant tier each tensor would
//! receive if the model were re-packed as ALICE-Dynamic-v1.
//!
//! Future phases (not implemented here):
//!
//! - I.1 corpus loader (JSONL calibration input)
//! - I.2 activation capture hook (forward-pass instrumentation)
//! - I.4 GGUF repack writer
//! - I.5 end-to-end integration harness
//! - I.6 ALICE-LLM Studio badge integration
//!
//! # Heuristic source
//!
//! The tier-decision rules are the Unsloth-baseline heuristics extracted in
//! `memory/success_unsloth_v3_sensitivity_oracle_2026_09_12.md` from a
//! metadata-only analysis of Unsloth Dynamic v3.0 Qwen3.8-27B GGUFs. See
//! [`heuristic`] for the eight rules and their derivation.
//!
//! # Non-goals
//!
//! - No activation statistics: this module cannot beat Unsloth without the
//!   Phase I.2 hook. It reproduces Unsloth's per-layer *structure* using the
//!   Heuristic 8 rules but does not tune tier choices to a specific corpus.
//! - No I-quant emission: the tier menu ([`tier::QuantTier`]) is restricted
//!   to F32 / Q8_0 / Q_K family. IQ2/3/4 tensors are out of scope for
//!   ALICE-Dynamic-v1 (see design memo alternative (d)).
//! - No GGUF write: assignments are emitted as JSON only. Repack lands in
//!   Phase I.4.

pub mod heuristic;
pub mod tier;

pub use heuristic::{
    assign_tier, classify_layers, Assignment, BlockKind, LayerClassification, MTP_DROP_BYTES,
};
pub use tier::{QuantTier, SizeBudget, TierBand, TierMenu};
