//! Quantization tier types and per-target-size tier menus for ALICE-Dynamic-v1.
//!
//! `TierMenu` selects the concrete quant types available for each priority
//! band (high / mid / low) given the user-specified `SizeBudget`. Heuristic
//! rules in [`crate::imatrix::heuristic`] read this menu instead of hardcoding
//! `GgmlType` values so the same rule tree scales from a 3 GB target down to
//! ~1.5 GB and up to 10+ GB without duplicated match arms.
//!
//! Only Q_K family + Q8_0 + F32 are exposed here. I-quant family (IQ2/3/4)
//! is intentionally excluded — see [[project_alice_llm_imatrix_cli_design]]
//! alternative (d) for the rationale.

use crate::gguf::GgmlType;

/// Concrete quant type an ALICE-Dynamic-v1 pipeline is willing to emit.
///
/// Wraps a subset of [`GgmlType`] restricted to types ALICE-LLM can already
/// dequantize (Q_K + Q8_0 + F32). `IQ1_S` is intentionally excluded from the
/// output menu even though ALICE-LLM supports read: repack quality at 1-bit
/// requires I-quant family support (unimplemented) to be competitive.
#[allow(non_camel_case_types)] // matches upstream GgmlType naming (Q4_0 etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantTier {
    F32,
    Q8_0,
    Q6_K,
    Q5_K,
    Q4_K,
    Q3_K,
    Q2_K,
}

impl QuantTier {
    /// Return the canonical GGML wire type for this tier.
    #[must_use]
    pub const fn to_ggml(self) -> GgmlType {
        match self {
            Self::F32 => GgmlType::F32,
            Self::Q8_0 => GgmlType::Q8_0,
            Self::Q6_K => GgmlType::Q6_K,
            Self::Q5_K => GgmlType::Q5_K,
            Self::Q4_K => GgmlType::Q4_K,
            Self::Q3_K => GgmlType::Q3_K,
            Self::Q2_K => GgmlType::Q2_K,
        }
    }

    /// Bits-per-weight approximation for size accounting.
    #[must_use]
    pub const fn bpw(self) -> f32 {
        match self {
            Self::F32 => 32.0,
            Self::Q8_0 => 8.5,
            Self::Q6_K => 6.5625,
            Self::Q5_K => 5.5,
            Self::Q4_K => 4.5,
            Self::Q3_K => 3.4375,
            Self::Q2_K => 2.625,
        }
    }

    /// Human-readable tag used in `layer_assignments.json`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::Q8_0 => "Q8_0",
            Self::Q6_K => "Q6_K",
            Self::Q5_K => "Q5_K",
            Self::Q4_K => "Q4_K",
            Self::Q3_K => "Q3_K",
            Self::Q2_K => "Q2_K",
        }
    }
}

/// User-supplied target size (bytes). Parsed from CLI flags like `3.5G`.
#[derive(Debug, Clone, Copy)]
pub struct SizeBudget {
    bytes: u64,
}

impl SizeBudget {
    /// Construct from raw byte count. Zero is rejected as a malformed budget.
    #[must_use]
    pub const fn from_bytes(bytes: u64) -> Option<Self> {
        if bytes == 0 {
            None
        } else {
            Some(Self { bytes })
        }
    }

    /// Total target size in gigabytes (base-10, matches Unsloth's tier labels).
    #[must_use]
    pub fn total_gb(self) -> f32 {
        self.bytes as f32 / 1_000_000_000.0
    }

    /// Total target size in bytes.
    #[must_use]
    pub const fn total_bytes(self) -> u64 {
        self.bytes
    }
}

/// Priority band emitted by the heuristic rules. The `TierMenu` maps each
/// band to a concrete [`QuantTier`] based on the overall size budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierBand {
    /// Highest-preservation band. Used for `attn_v` and depth-bonused layers.
    High,
    /// Standard preservation. Used for `attn_output` and `output.weight`.
    Mid,
    /// Slightly compressed. Used for `ffn_down` and `attn_q`/`attn_k`.
    MidLow,
    /// Aggressive compression. Used for `ffn_gate`/`ffn_up` and DeltaNet QKV.
    Low,
}

/// Per-band tier selection derived from [`SizeBudget`].
///
/// Three brackets map directly to Unsloth's UD-* naming:
/// - `>= 8.0 GB`   → mid/high tier (Q5_K, Q6_K, Q4_K, Q3_K)
/// - `4.0..8.0 GB` → 1-bit-adjacent tier (Q4_K, Q3_K, Q3_K, Q2_K)
/// - `< 4.0 GB`    → extreme tier (Q3_K, Q2_K, Q2_K, Q2_K)
#[derive(Debug, Clone, Copy)]
pub struct TierMenu {
    pub high: QuantTier,
    pub mid: QuantTier,
    pub mid_low: QuantTier,
    pub low: QuantTier,
}

impl TierMenu {
    /// Derive a per-band tier from the overall size budget.
    #[must_use]
    pub fn for_budget(budget: SizeBudget) -> Self {
        let gb = budget.total_gb();
        if gb >= 8.0 {
            // UD-Q2_K_XL 9.83 GB class — retain Q5_K/Q6_K headroom
            Self {
                high: QuantTier::Q6_K,
                mid: QuantTier::Q5_K,
                mid_low: QuantTier::Q4_K,
                low: QuantTier::Q3_K,
            }
        } else if gb >= 4.0 {
            // Mid-range: 4-8 GB target (Qwen 3.5-4B typical envelope)
            Self {
                high: QuantTier::Q5_K,
                mid: QuantTier::Q4_K,
                mid_low: QuantTier::Q3_K,
                low: QuantTier::Q3_K,
            }
        } else {
            // Extreme 1-bit-adjacent: UD-IQ1_S 6.2 GB analogue for Q_K path
            Self {
                high: QuantTier::Q4_K,
                mid: QuantTier::Q3_K,
                mid_low: QuantTier::Q2_K,
                low: QuantTier::Q2_K,
            }
        }
    }

    /// Look up the concrete tier for a heuristic-emitted band.
    #[must_use]
    pub const fn resolve(&self, band: TierBand) -> QuantTier {
        match band {
            TierBand::High => self.high,
            TierBand::Mid => self.mid,
            TierBand::MidLow => self.mid_low,
            TierBand::Low => self.low,
        }
    }

    /// Bump a tier one step higher (used by the late-layer depth bonus).
    /// Saturates at Q8_0.
    #[must_use]
    pub const fn bump(tier: QuantTier) -> QuantTier {
        match tier {
            QuantTier::F32 | QuantTier::Q8_0 => QuantTier::Q8_0,
            QuantTier::Q6_K => QuantTier::Q8_0,
            QuantTier::Q5_K => QuantTier::Q6_K,
            QuantTier::Q4_K => QuantTier::Q5_K,
            QuantTier::Q3_K => QuantTier::Q4_K,
            QuantTier::Q2_K => QuantTier::Q3_K,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_from_bytes_rejects_zero() {
        assert!(SizeBudget::from_bytes(0).is_none());
        assert!(SizeBudget::from_bytes(1).is_some());
    }

    #[test]
    fn budget_gb_conversion_matches_unsloth_labels() {
        let b = SizeBudget::from_bytes(9_830_000_000).expect("nonzero");
        assert!((b.total_gb() - 9.83).abs() < 0.01);
    }

    #[test]
    fn menu_high_budget_reserves_q6k_headroom() {
        let menu = TierMenu::for_budget(SizeBudget::from_bytes(10_000_000_000).expect("nonzero"));
        assert_eq!(menu.high, QuantTier::Q6_K);
        assert_eq!(menu.low, QuantTier::Q3_K);
    }

    #[test]
    fn menu_mid_budget_uses_q5k_high() {
        let menu = TierMenu::for_budget(SizeBudget::from_bytes(5_000_000_000).expect("nonzero"));
        assert_eq!(menu.high, QuantTier::Q5_K);
        assert_eq!(menu.mid, QuantTier::Q4_K);
    }

    #[test]
    fn menu_low_budget_collapses_to_q2k_floor() {
        let menu = TierMenu::for_budget(SizeBudget::from_bytes(3_500_000_000).expect("nonzero"));
        assert_eq!(menu.high, QuantTier::Q4_K);
        assert_eq!(menu.low, QuantTier::Q2_K);
    }

    #[test]
    fn bump_saturates_at_q8_0() {
        assert_eq!(TierMenu::bump(QuantTier::Q8_0), QuantTier::Q8_0);
        assert_eq!(TierMenu::bump(QuantTier::Q6_K), QuantTier::Q8_0);
        assert_eq!(TierMenu::bump(QuantTier::Q5_K), QuantTier::Q6_K);
        assert_eq!(TierMenu::bump(QuantTier::Q2_K), QuantTier::Q3_K);
    }

    #[test]
    fn tier_bpw_is_monotonic() {
        let tiers = [
            QuantTier::Q2_K,
            QuantTier::Q3_K,
            QuantTier::Q4_K,
            QuantTier::Q5_K,
            QuantTier::Q6_K,
            QuantTier::Q8_0,
            QuantTier::F32,
        ];
        for pair in tiers.windows(2) {
            assert!(
                pair[0].bpw() < pair[1].bpw(),
                "bpw not monotonic between {pair:?}",
            );
        }
    }
}
