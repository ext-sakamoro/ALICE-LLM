//! Unsloth-baseline heuristic rules for per-tensor quant tier assignment.
//!
//! Implements the eight heuristics extracted from Unsloth Dynamic v3.0's
//! per-layer decisions in
//! [[success_unsloth_v3_sensitivity_oracle_2026_09_12]]:
//!
//! | # | Rule | Priority |
//! |---|------|----------|
//! | 1 | `token_embd` = Q2_K; `output.weight` = mid tier | P1 |
//! | 2 | norm + `ssm_conv1d` always F32 | P0 |
//! | 3 | DeltaNet `ssm_alpha` / `ssm_beta` = Q8_0 mandatory | P1 |
//! | 4 | `attn_v` two bands above `attn_q` / `attn_k` | P2 |
//! | 5 | FFN `ffn_gate` / `ffn_up` = low band | P2 |
//! | 6 | last 1/3 of blocks bump one tier (`TierMenu::bump`) | P2 |
//! | 7 | `nextn.*` (MTP) dropped when target < 8.37 GB | P3 |
//! | 8 | pure-attention blocks (split Q/K/V) > fused DeltaNet QKV | P3 |
//!
//! The rules never allocate and never `unwrap`. Layer classification is
//! performed once up-front by [`classify_layers`]; per-tensor decisions run
//! in `O(1)` after that.

use std::collections::BTreeMap;

use crate::gguf::TensorInfo;
use crate::imatrix::tier::{QuantTier, SizeBudget, TierBand, TierMenu};

/// MTP drop threshold in bytes, matching the 8.37 GB Unsloth boundary.
///
/// Both UD-Q2_K_XL (9.83 GB) and larger tiers keep the `nextn.*` module;
/// UD-IQ1_S (6.2 GB) drops it entirely. The threshold is expressed in
/// base-10 GB to align with Unsloth's own file-size labelling.
pub const MTP_DROP_BYTES: u64 = 8_370_000_000;

/// Kind of transformer block, inferred from per-block tensor names.
///
/// Qwen3.5 hybrid architecture (`arch=qwen35`) mixes pure attention blocks
/// and DeltaNet blocks; ALICE-Dynamic-v1 preserves the pure-attention ones
/// at a higher tier per Heuristic 8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// Block has split `attn_q` / `attn_k` / `attn_v` tensors.
    PureAttention,
    /// Block has fused `attn_qkv` and `ssm_*` tensors.
    DeltaNetHybrid,
    /// No attention tensors detected (norm-only helper block).
    NormOnly,
}

/// Classified layout of the model's transformer stack.
#[derive(Debug, Clone)]
pub struct LayerClassification {
    /// `block_idx → kind` map. Missing indices are treated as `NormOnly`.
    pub kinds: BTreeMap<u32, BlockKind>,
    /// Highest block index observed (inclusive). Used to compute the late-
    /// layer depth-bonus threshold.
    pub max_block: u32,
}

impl LayerClassification {
    /// Compute the depth-bonus cutoff (last third of blocks receive a bump).
    #[must_use]
    pub const fn late_layer_threshold(&self) -> u32 {
        // Late = last third. For 65 blocks, indices 44..=64 qualify.
        self.max_block.saturating_sub(self.max_block / 3)
    }

    #[must_use]
    pub fn kind_of(&self, block_idx: u32) -> BlockKind {
        self.kinds
            .get(&block_idx)
            .copied()
            .unwrap_or(BlockKind::NormOnly)
    }
}

/// Inspect every tensor in the model and classify each block index.
///
/// A block is `PureAttention` if any of its tensors ends with `.attn_q.weight`,
/// `.attn_k.weight`, or `.attn_v.weight`. It is `DeltaNetHybrid` if it has
/// `.attn_qkv.weight` or any `.ssm_*` tensor. If both signatures appear (rare)
/// the DeltaNet classification wins because SSM state is the more sensitive
/// path. Blocks with neither signature are `NormOnly`.
#[must_use]
pub fn classify_layers<'a, I>(tensors: I) -> LayerClassification
where
    I: IntoIterator<Item = &'a TensorInfo>,
{
    let mut kinds: BTreeMap<u32, BlockKind> = BTreeMap::new();
    let mut max_block: u32 = 0;

    for info in tensors {
        let Some((block_idx, part)) = parse_block_tensor(&info.name) else {
            continue;
        };
        if block_idx > max_block {
            max_block = block_idx;
        }
        let new_kind = if part.starts_with("ssm_") || part == "attn_qkv" {
            BlockKind::DeltaNetHybrid
        } else if matches!(part, "attn_q" | "attn_k" | "attn_v") {
            BlockKind::PureAttention
        } else {
            continue;
        };
        kinds
            .entry(block_idx)
            .and_modify(|existing| {
                // DeltaNet wins over PureAttention if both are seen.
                if new_kind == BlockKind::DeltaNetHybrid {
                    *existing = BlockKind::DeltaNetHybrid;
                }
            })
            .or_insert(new_kind);
    }

    LayerClassification { kinds, max_block }
}

/// Parse `blk.<N>.<part>.weight` → `(N, part)`. Returns `None` for tensors
/// that don't match the block pattern.
fn parse_block_tensor(name: &str) -> Option<(u32, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    let (idx_str, tail) = rest.split_at(dot);
    let block_idx = idx_str.parse::<u32>().ok()?;
    let part = tail.strip_prefix('.')?.strip_suffix(".weight")?;
    Some((block_idx, part))
}

/// Outcome of applying the heuristic to a single tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Assignment {
    /// Emit this tensor at the given quant tier.
    Keep(QuantTier),
    /// Drop this tensor from the re-packed GGUF (currently only MTP).
    Drop,
}

/// Apply the eight heuristics to a single tensor. Pure function: no state,
/// no allocation. `classification` and `menu` are shared across a whole
/// model.
#[must_use]
pub fn assign_tier(
    name: &str,
    classification: &LayerClassification,
    menu: &TierMenu,
    budget: SizeBudget,
) -> Assignment {
    // Heuristic 7: drop MTP under the size-budget threshold.
    if name.starts_with("nextn.")
        || name.contains(".nextn.")
        || name.starts_with("mtp.")
        || name.contains(".mtp.")
    {
        if budget.total_bytes() < MTP_DROP_BYTES {
            return Assignment::Drop;
        }
        // Retained MTP tensors: preserve at high tier (Unsloth kept eh_proj Q6_K).
        return Assignment::Keep(TierMenu::bump(menu.high));
    }

    // Heuristic 2: norm + conv1d always F32.
    if is_norm_or_conv1d(name) {
        return Assignment::Keep(QuantTier::F32);
    }

    // Heuristic 1: global tensors.
    if name == "token_embd.weight" {
        return Assignment::Keep(QuantTier::Q2_K);
    }
    if name == "output.weight" {
        return Assignment::Keep(menu.mid);
    }

    // Block tensors: extract idx + part, apply per-slot rules. Unknown
    // top-level tensors fall through to the mid-low band.
    let Some((block_idx, part)) = parse_block_tensor(name) else {
        return Assignment::Keep(menu.mid_low);
    };

    // Heuristic 3: DeltaNet gates always Q8_0 (bypass menu).
    if matches!(part, "ssm_alpha" | "ssm_beta") {
        return Assignment::Keep(QuantTier::Q8_0);
    }

    let is_late = block_idx >= classification.late_layer_threshold();
    let kind = classification.kind_of(block_idx);
    let base_band = slot_to_band(part, kind);
    let base_tier = menu.resolve(base_band);

    // Heuristic 6: late-layer bump (skip if already at Q8_0 or if slot is
    // aggressive-compress-by-design like ffn_gate/ffn_up at low band).
    let tier = if is_late && should_bump_late(part) {
        TierMenu::bump(base_tier)
    } else {
        base_tier
    };
    Assignment::Keep(tier)
}

/// True if the tensor is a normalization or 1-D convolution weight that must
/// stay in F32 for numerical stability (Heuristic 2). Handles both dotted
/// (`.conv1d.weight`) and merged (`.ssm_conv1d.weight`) naming conventions
/// used by the Qwen3.5 hybrid DeltaNet blocks.
fn is_norm_or_conv1d(name: &str) -> bool {
    name.ends_with("_norm.weight")
        || name.ends_with(".norm.weight")
        || name.ends_with(".conv1d.weight")
        || name.ends_with("_conv1d.weight")
        || name == "output_norm.weight"
}

/// Map a per-block slot name (e.g. `attn_v`, `ffn_gate`, `attn_qkv`) to its
/// heuristic tier band. Encodes Heuristics 4, 5, and 8.
fn slot_to_band(part: &str, kind: BlockKind) -> TierBand {
    match (part, kind) {
        // Heuristic 4: value projection preserved (High band).
        ("attn_v", _) => TierBand::High,
        // Heuristic 8: pure attention Q/K/output preserved at mid tier.
        ("attn_q" | "attn_k", BlockKind::PureAttention) => TierBand::Mid,
        ("attn_output", BlockKind::PureAttention) => TierBand::Mid,
        // Fused DeltaNet QKV is compressed aggressively (Heuristic 8 reverse).
        ("attn_qkv", _) => TierBand::Low,
        // FFN down slightly preserved.
        ("ffn_down", _) => TierBand::MidLow,
        // Heuristic 5: FFN gate/up = Low.
        ("ffn_gate" | "ffn_up", _) => TierBand::Low,
        // DeltaNet state output.
        ("ssm_out", _) => TierBand::MidLow,
        // Attention Q/K in a DeltaNet-hybrid block (unusual): default mid-low.
        ("attn_q" | "attn_k" | "attn_output", _) => TierBand::MidLow,
        // Anything else (Q gate, etc.) → mid.
        _ => TierBand::Mid,
    }
}

/// Whether Heuristic 6 (late-layer bump) should fire for this slot. FFN
/// gate/up remain aggressive even in late layers because the sensitivity
/// oracle showed only late-layer *variance*, not systematic preservation.
fn should_bump_late(part: &str) -> bool {
    !matches!(part, "ssm_alpha" | "ssm_beta" | "ffn_gate" | "ffn_up")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgmlType;

    fn ti(name: &str) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            n_dims: 2,
            dims: vec![1024, 1024],
            qtype: GgmlType::F32,
            offset: 0,
        }
    }

    fn qwen35_layout() -> Vec<TensorInfo> {
        // 3 pure-attention blocks (0, 1, 2), 3 DeltaNet blocks (3, 4, 5).
        let mut ts = Vec::new();
        for i in 0..3 {
            ts.push(ti(&format!("blk.{i}.attn_q.weight")));
            ts.push(ti(&format!("blk.{i}.attn_k.weight")));
            ts.push(ti(&format!("blk.{i}.attn_v.weight")));
            ts.push(ti(&format!("blk.{i}.attn_output.weight")));
            ts.push(ti(&format!("blk.{i}.ffn_gate.weight")));
        }
        for i in 3..6 {
            ts.push(ti(&format!("blk.{i}.attn_qkv.weight")));
            ts.push(ti(&format!("blk.{i}.ssm_alpha.weight")));
            ts.push(ti(&format!("blk.{i}.ssm_beta.weight")));
            ts.push(ti(&format!("blk.{i}.ffn_gate.weight")));
        }
        ts
    }

    fn menu_9g() -> TierMenu {
        TierMenu::for_budget(SizeBudget::from_bytes(9_000_000_000).expect("nonzero"))
    }

    fn budget_9g() -> SizeBudget {
        SizeBudget::from_bytes(9_000_000_000).expect("nonzero")
    }

    fn budget_5g() -> SizeBudget {
        SizeBudget::from_bytes(5_000_000_000).expect("nonzero")
    }

    #[test]
    fn parse_block_tensor_ok() {
        assert_eq!(
            parse_block_tensor("blk.7.attn_v.weight"),
            Some((7, "attn_v"))
        );
        assert_eq!(
            parse_block_tensor("blk.64.nextn.eh_proj.weight"),
            Some((64, "nextn.eh_proj"))
        );
    }

    #[test]
    fn parse_block_tensor_rejects_globals() {
        assert!(parse_block_tensor("output.weight").is_none());
        assert!(parse_block_tensor("blk.NaN.attn_v.weight").is_none());
    }

    #[test]
    fn classify_detects_pure_and_deltanet_blocks() {
        let ts = qwen35_layout();
        let c = classify_layers(&ts);
        assert_eq!(c.kind_of(0), BlockKind::PureAttention);
        assert_eq!(c.kind_of(2), BlockKind::PureAttention);
        assert_eq!(c.kind_of(3), BlockKind::DeltaNetHybrid);
        assert_eq!(c.kind_of(5), BlockKind::DeltaNetHybrid);
        assert_eq!(c.max_block, 5);
    }

    #[test]
    fn h2_norms_always_f32() {
        let c = classify_layers(&qwen35_layout());
        let m = menu_9g();
        for name in [
            "output_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_k_norm.weight",
            "blk.0.attn_q_norm.weight",
            "blk.0.post_attention_norm.weight",
            "blk.0.ssm_norm.weight",
            "blk.0.ssm_conv1d.weight",
        ] {
            assert_eq!(
                assign_tier(name, &c, &m, budget_9g()),
                Assignment::Keep(QuantTier::F32),
                "tensor {name} should be F32"
            );
        }
    }

    #[test]
    fn h3_deltanet_gates_always_q8_0() {
        let c = classify_layers(&qwen35_layout());
        let m = menu_9g();
        assert_eq!(
            assign_tier("blk.3.ssm_alpha.weight", &c, &m, budget_9g()),
            Assignment::Keep(QuantTier::Q8_0)
        );
        assert_eq!(
            assign_tier("blk.4.ssm_beta.weight", &c, &m, budget_9g()),
            Assignment::Keep(QuantTier::Q8_0)
        );
    }

    #[test]
    fn h1_token_embd_always_q2k_output_at_mid() {
        let c = classify_layers(&qwen35_layout());
        let m = menu_9g();
        assert_eq!(
            assign_tier("token_embd.weight", &c, &m, budget_9g()),
            Assignment::Keep(QuantTier::Q2_K)
        );
        assert_eq!(
            assign_tier("output.weight", &c, &m, budget_9g()),
            Assignment::Keep(m.mid)
        );
    }

    #[test]
    fn h4_attn_v_higher_than_attn_qk_in_pure_attention_block() {
        let c = classify_layers(&qwen35_layout());
        let m = menu_9g();
        let v = assign_tier("blk.0.attn_v.weight", &c, &m, budget_9g());
        let q = assign_tier("blk.0.attn_q.weight", &c, &m, budget_9g());
        let Assignment::Keep(v_t) = v else {
            panic!("attn_v dropped")
        };
        let Assignment::Keep(q_t) = q else {
            panic!("attn_q dropped")
        };
        assert!(
            v_t.bpw() >= q_t.bpw(),
            "attn_v ({v_t:?}) should be >= attn_q ({q_t:?})",
        );
    }

    #[test]
    fn h5_ffn_gate_up_at_low_band() {
        let c = classify_layers(&qwen35_layout());
        let m = menu_9g();
        assert_eq!(
            assign_tier("blk.0.ffn_gate.weight", &c, &m, budget_9g()),
            Assignment::Keep(m.low)
        );
        assert_eq!(
            assign_tier("blk.0.ffn_up.weight", &c, &m, budget_9g()),
            Assignment::Keep(m.low)
        );
    }

    #[test]
    fn h6_late_layer_bumps_attn_v_but_not_ffn_gate() {
        // Build a layout with 30 blocks; late threshold = 30 - 10 = 20.
        let mut ts = Vec::new();
        for i in 0..30 {
            ts.push(ti(&format!("blk.{i}.attn_q.weight")));
            ts.push(ti(&format!("blk.{i}.attn_k.weight")));
            ts.push(ti(&format!("blk.{i}.attn_v.weight")));
            ts.push(ti(&format!("blk.{i}.ffn_gate.weight")));
        }
        let c = classify_layers(&ts);
        let m = menu_9g();
        // late block 29: attn_v should be bumped one tier above menu.high.
        let late = assign_tier("blk.29.attn_v.weight", &c, &m, budget_9g());
        let early = assign_tier("blk.0.attn_v.weight", &c, &m, budget_9g());
        let (Assignment::Keep(late_t), Assignment::Keep(early_t)) = (late, early) else {
            panic!("attn_v dropped");
        };
        assert!(
            late_t.bpw() > early_t.bpw(),
            "late attn_v ({late_t:?}) should exceed early ({early_t:?})",
        );

        // ffn_gate must NOT bump (Heuristic 5 preservation-through-depth).
        let late_gate = assign_tier("blk.29.ffn_gate.weight", &c, &m, budget_9g());
        let early_gate = assign_tier("blk.0.ffn_gate.weight", &c, &m, budget_9g());
        assert_eq!(late_gate, early_gate, "ffn_gate must not receive late bump");
    }

    #[test]
    fn h7_mtp_dropped_below_threshold_kept_above() {
        let c = classify_layers(&qwen35_layout());
        let m_low = TierMenu::for_budget(budget_5g());
        assert_eq!(
            assign_tier("blk.64.nextn.eh_proj.weight", &c, &m_low, budget_5g()),
            Assignment::Drop,
            "MTP must be dropped when budget < 8.37 GB"
        );

        let m_hi = menu_9g();
        let kept = assign_tier("blk.64.nextn.eh_proj.weight", &c, &m_hi, budget_9g());
        assert!(
            matches!(kept, Assignment::Keep(_)),
            "MTP must be kept when budget >= 8.37 GB"
        );
    }

    #[test]
    fn h8_pure_attention_qkv_preserved_over_deltanet_qkv() {
        let c = classify_layers(&qwen35_layout());
        let m = menu_9g();
        // Pure-attention block 0: attn_q at mid band.
        let pure_q = assign_tier("blk.0.attn_q.weight", &c, &m, budget_9g());
        // DeltaNet block 3: fused attn_qkv at low band.
        let delta_qkv = assign_tier("blk.3.attn_qkv.weight", &c, &m, budget_9g());
        let (Assignment::Keep(pure_t), Assignment::Keep(delta_t)) = (pure_q, delta_qkv) else {
            panic!("dropped");
        };
        assert!(
            pure_t.bpw() > delta_t.bpw(),
            "pure attention Q ({pure_t:?}) should exceed DeltaNet fused QKV ({delta_t:?})",
        );
    }

    #[test]
    fn late_layer_threshold_matches_qwen38_layout() {
        // Simulate Unsloth Qwen3.8-27B: 65 blocks (max_block = 64).
        let mut ts = Vec::new();
        for i in 0..65 {
            ts.push(ti(&format!("blk.{i}.attn_qkv.weight")));
            ts.push(ti(&format!("blk.{i}.ssm_alpha.weight")));
        }
        let c = classify_layers(&ts);
        assert_eq!(c.max_block, 64);
        // late = max - max/3 = 64 - 21 = 43. blocks 43..=64 qualify (22 blocks).
        assert_eq!(c.late_layer_threshold(), 43);
    }
}
