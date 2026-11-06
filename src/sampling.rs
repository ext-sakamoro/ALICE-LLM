//! sampling.

#[cfg(feature = "grammar")]
use crate::grammar::{Fsm, FsmError, TokenTrie};

// Sampling
// ---------------------------------------------------------------------------

/// Apply temperature scaling to logits (in-place).
///
/// A `NaN` temperature or a scale past the range of `f32` leaves `NaN` or `+inf` in the logits;
/// the next function they reach refuses them.
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature < f32::EPSILON {
        return;
    }
    let inv_t = 1.0 / temperature;
    for l in logits.iter_mut() {
        *l *= inv_t;
    }
}

/// Why a sampling function refused its input. Nothing was changed when it did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SamplingError {
    /// This logit is `NaN` or `+inf`: the model failed numerically, and no order or softmax exists.
    NonFinite { index: usize },
    /// Every token is masked (`-inf`) or has probability 0, so there is nothing to sample.
    AllMasked,
    /// This probability is not finite and at least 0.
    InvalidProbability { index: usize },
    /// `p` or the random value is `NaN`.
    InvalidParameter,
}

impl core::fmt::Display for SamplingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFinite { index } => write!(f, "logit {index} is NaN or +inf"),
            Self::AllMasked => f.write_str("every token is masked"),
            Self::InvalidProbability { index } => write!(f, "probability {index} is invalid"),
            Self::InvalidParameter => f.write_str("a sampling parameter is NaN"),
        }
    }
}

impl std::error::Error for SamplingError {}

/// Logits are finite or `-inf` (a masked token), and at least one is not masked.
pub(crate) fn check_logits(logits: &[f32]) -> Result<(), SamplingError> {
    if let Some(index) = logits
        .iter()
        .position(|l| l.is_nan() || *l == f32::INFINITY)
    {
        return Err(SamplingError::NonFinite { index });
    }
    if !logits.is_empty() && logits.iter().all(|l| *l == f32::NEG_INFINITY) {
        return Err(SamplingError::AllMasked);
    }
    Ok(())
}

/// Top-k filtering: keep only the top k logits, set the rest to -inf.
///
/// Uses `select_nth_unstable_by` (partial sort, O(N)) to find the K-th
/// largest logit without fully sorting the whole vocabulary — matters
/// once `vocab_size` grows past 10k.
///
/// # Errors
///
/// [`SamplingError::NonFinite`] or [`SamplingError::AllMasked`], also when `k` keeps every token.
pub fn top_k_filter(logits: &mut [f32], k: usize) -> Result<(), SamplingError> {
    check_logits(logits)?;
    if k == 0 || k >= logits.len() {
        return Ok(());
    }
    // O(N) partial sort: locate the K-th largest value.
    let mut vals: Vec<f32> = logits.to_vec();
    let kth = k - 1;
    vals.select_nth_unstable_by(kth, f32_descending);
    let threshold = vals[kth];
    for l in logits.iter_mut() {
        if *l < threshold {
            *l = f32::NEG_INFINITY;
        }
    }
    Ok(())
}

/// Descending order of values already checked to have no `NaN`.
fn f32_descending(a: &f32, b: &f32) -> std::cmp::Ordering {
    descending_nan_last(*a, *b)
}

/// A total order for sorting scores: descending, `NaN` last. Finite values and `±inf` order exactly as
/// `partial_cmp` does, `-0` and `+0` as equal (`f32::total_cmp` would part them and reorder ties).
#[must_use]
pub fn descending_nan_last(a: f32, b: f32) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal),
    }
}

/// Top-p (nucleus) filtering: keep smallest set of tokens whose
/// cumulative probability exceeds p.
///
/// # Errors
///
/// [`SamplingError::NonFinite`], [`SamplingError::AllMasked`], or
/// [`SamplingError::InvalidParameter`] when `p` is `NaN`.
pub fn top_p_filter(logits: &mut [f32], p: f32) -> Result<(), SamplingError> {
    check_logits(logits)?;
    if p.is_nan() {
        return Err(SamplingError::InvalidParameter);
    }
    if p >= 1.0 {
        return Ok(());
    }

    let mut probs = logits.to_vec();
    softmax_of_checked(&mut probs);

    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| f32_descending(&a.1, &b.1));

    let mut cumsum = 0.0f32;
    let mut cutoff_idx = indexed.len();
    for (rank, &(_, prob)) in indexed.iter().enumerate() {
        cumsum += prob;
        if cumsum > p {
            cutoff_idx = rank + 1;
            break;
        }
    }

    let keep: std::collections::HashSet<usize> =
        indexed[..cutoff_idx].iter().map(|&(i, _)| i).collect();

    for (i, l) in logits.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *l = f32::NEG_INFINITY;
        }
    }
    Ok(())
}

/// Softmax in-place.
///
/// # Errors
///
/// [`SamplingError::NonFinite`] or [`SamplingError::AllMasked`].
pub fn softmax_inplace(v: &mut [f32]) -> Result<(), SamplingError> {
    check_logits(v)?;
    softmax_of_checked(v);
    Ok(())
}

/// The softmax itself. Attention scores use it directly: they are not logits to sample from.
pub(crate) fn softmax_of_checked(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let max_val = v
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, crate::lawmath::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = crate::lawmath::exp(*x - max_val);
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// Softmax returning a new vector.
///
/// # Errors
///
/// [`SamplingError::NonFinite`] or [`SamplingError::AllMasked`].
pub fn softmax(logits: &[f32]) -> Result<Vec<f32>, SamplingError> {
    let mut v = logits.to_vec();
    softmax_inplace(&mut v)?;
    Ok(v)
}

/// Argmax sampling (greedy). No tokens gives 0.
///
/// # Errors
///
/// [`SamplingError::NonFinite`] or [`SamplingError::AllMasked`].
pub fn sample_argmax(logits: &[f32]) -> Result<usize, SamplingError> {
    check_logits(logits)?;
    Ok(logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i))
}

/// Deterministic weighted sampling using a provided random value in `[0, 1)`. No tokens gives 0.
/// A value past the sum of the probabilities gives the last token with any probability.
///
/// # Errors
///
/// [`SamplingError::InvalidProbability`] for a value that is not finite and at least 0,
/// [`SamplingError::AllMasked`] when all are 0, [`SamplingError::InvalidParameter`] when `rand_val`
/// is `NaN`.
pub fn sample_with_random(probs: &[f32], rand_val: f32) -> Result<usize, SamplingError> {
    if let Some(index) = probs.iter().position(|p| !p.is_finite() || *p < 0.0) {
        return Err(SamplingError::InvalidProbability { index });
    }
    if !probs.is_empty() && probs.iter().all(|p| *p == 0.0) {
        return Err(SamplingError::AllMasked);
    }
    if rand_val.is_nan() {
        return Err(SamplingError::InvalidParameter);
    }
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if rand_val < cumsum {
            return Ok(i);
        }
    }
    // past the sum, which rounding can leave a little under 1: never a token with no probability
    Ok(probs.iter().rposition(|&p| p > 0.0).unwrap_or(0))
}

// ---------------------------------------------------------------------------
// Grammar-constrained sampling (Phase X.8 B-3)
// ---------------------------------------------------------------------------

/// Tokenizer facade used by grammar-constrained sampling.
///
/// Kept as a trait so that both the real `GgufTokenizer` and lightweight
/// test doubles can drive the mask, and so the grammar module doesn't
/// have to link against `gguf` when not needed.
#[cfg(feature = "grammar")]
pub trait GrammarTokenizer {
    /// Return the textual form of `id` as an owned `String`.
    /// Returns `""` for control / meta tokens that contribute no visible
    /// text (BOS, `<|im_start|>`, etc.); the mask treats such tokens as
    /// non-advancing and disallows them mid-parse.
    fn text_of(&self, id: u32) -> String;

    /// End-of-sequence token id. The mask allows EOS only when the FSM
    /// reports [`Fsm::is_final`].
    fn eos_id(&self) -> u32;
}

#[cfg(feature = "grammar")]
impl GrammarTokenizer for crate::gguf::GgufTokenizer {
    fn text_of(&self, id: u32) -> String {
        self.decode(&[id])
    }
    fn eos_id(&self) -> u32 {
        self.eos_id
    }
}

/// Set to `-inf` every logit whose token would violate `fsm`.
///
/// - Tokens whose decoded text the FSM refuses are masked.
/// - Tokens that decode to `""` (control tokens) are masked so the
///   sampler cannot emit invisible glyphs mid-parse.
/// - The EOS token is preserved iff the FSM is already in a final state.
/// - Logits already at `-inf` (or `NaN`) are left untouched.
///
/// This does not mutate the FSM. Use [`advance_fsm_on_emit`] once the
/// sampler picks a winner.
///
/// # Complexity
///
/// O(vocab_size × cost(accepts_str)). `accepts_str` clones the FSM per
/// call, which is bounded by the number of live cursors. Measured at
/// ~8 s/step for a 130 k vocab against the LOL grammar root — kept as the
/// reference implementation for parity tests; production decoding uses
/// [`mask_logits_by_grammar_trie`].
#[cfg(feature = "grammar")]
pub fn mask_logits_by_grammar<T: GrammarTokenizer + ?Sized>(
    fsm: &Fsm<'_>,
    tokenizer: &T,
    logits: &mut [f32],
) {
    let eos = tokenizer.eos_id();
    let final_state = fsm.is_final();
    for (id, logit) in logits.iter_mut().enumerate() {
        // Preserve prior masks and NaN sentinels — no need to reclassify.
        if !logit.is_finite() {
            continue;
        }
        let id_u32 = u32::try_from(id).unwrap_or(u32::MAX);
        if id_u32 == eos {
            if !final_state {
                *logit = f32::NEG_INFINITY;
            }
            continue;
        }
        let text = tokenizer.text_of(id_u32);
        if text.is_empty() {
            *logit = f32::NEG_INFINITY;
            continue;
        }
        if !fsm.accepts_str(&text) {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// Trie-backed equivalent of [`mask_logits_by_grammar`].
///
/// Produces the identical mask (same admitted set, same EOS rule, same
/// treatment of already non-finite logits) but computes the admitted set
/// once per step via [`TokenTrie::allowed_tokens`], sharing FSM work across
/// every token with a common prefix. `scratch` is reused between calls to
/// avoid per-step allocation; pass any `Vec` (it is cleared).
///
/// Returns the number of FSM advances the trie walk performed.
///
/// # Panics
///
/// Panics if `logits.len() != trie.vocab_size()` — the trie must be built
/// for the same tokenizer that produced `logits`.
#[cfg(feature = "grammar")]
pub fn mask_logits_by_grammar_trie<T: GrammarTokenizer + ?Sized>(
    fsm: &Fsm<'_>,
    trie: &TokenTrie,
    tokenizer: &T,
    logits: &mut [f32],
    scratch: &mut Vec<u32>,
) -> usize {
    assert_eq!(
        logits.len(),
        trie.vocab_size(),
        "logits length must match the trie vocabulary"
    );
    let advances = trie.allowed_tokens(fsm, scratch);
    // Mark admitted ids, then sweep. A bitmap over the vocab is cheaper
    // than sorting `scratch` and merging.
    let mut allowed = vec![false; logits.len()];
    for &id in scratch.iter() {
        allowed[id as usize] = true;
    }
    let eos = tokenizer.eos_id() as usize;
    let final_state = fsm.is_final();
    for (id, logit) in logits.iter_mut().enumerate() {
        if !logit.is_finite() {
            continue;
        }
        if id == eos {
            if !final_state {
                *logit = f32::NEG_INFINITY;
            }
            continue;
        }
        if !allowed[id] {
            *logit = f32::NEG_INFINITY;
        }
    }
    advances
}

/// Feed the text of the just-sampled `token_id` into the FSM so its
/// state matches the running emission.
///
/// EOS is a no-op (the caller terminates the loop). Returns
/// [`FsmError::NoTransition`] if a char of the token's text is rejected,
/// which indicates the sampler drew a token that the mask should have
/// excluded (i.e. a bug in the driver).
#[cfg(feature = "grammar")]
pub fn advance_fsm_on_emit<T: GrammarTokenizer + ?Sized>(
    fsm: &mut Fsm<'_>,
    tokenizer: &T,
    token_id: u32,
) -> Result<(), FsmError> {
    if token_id == tokenizer.eos_id() {
        return Ok(());
    }
    let text = tokenizer.text_of(token_id);
    for ch in text.chars() {
        fsm.advance(ch)?;
    }
    Ok(())
}

#[cfg(all(test, feature = "grammar"))]
mod grammar_tests {
    use super::{advance_fsm_on_emit, mask_logits_by_grammar, GrammarTokenizer};
    use crate::grammar::{parse_gbnf, Fsm};

    struct MockTokenizer {
        tokens: Vec<String>,
        eos: u32,
    }

    impl MockTokenizer {
        fn new(tokens: &[&str], eos: u32) -> Self {
            Self {
                tokens: tokens.iter().map(|s| (*s).to_string()).collect(),
                eos,
            }
        }
    }

    impl GrammarTokenizer for MockTokenizer {
        fn text_of(&self, id: u32) -> String {
            self.tokens.get(id as usize).cloned().unwrap_or_default()
        }
        fn eos_id(&self) -> u32 {
            self.eos
        }
    }

    #[test]
    fn masks_non_matching_prefix() {
        // 0=EOS, 1="y" ok, 2="n" rejected, 3="hello" rejected.
        let tok = MockTokenizer::new(&["", "y", "n", "hello"], 0);
        let grammar = parse_gbnf(r#"root ::= "yes""#).unwrap();
        let fsm = Fsm::start(&grammar).unwrap();
        let mut logits = vec![0.0f32; 4];
        mask_logits_by_grammar(&fsm, &tok, &mut logits);
        assert!(logits[0].is_infinite() && logits[0] < 0.0); // EOS
        assert!(logits[1].is_finite()); // "y"
        assert!(logits[2].is_infinite() && logits[2] < 0.0); // "n"
        assert!(logits[3].is_infinite() && logits[3] < 0.0); // "hello"
    }

    #[test]
    fn allows_eos_when_final() {
        // `root ::= "a"?` — final immediately.
        let tok = MockTokenizer::new(&["", "a", "b"], 0);
        let grammar = parse_gbnf(r#"root ::= "a"?"#).unwrap();
        let fsm = Fsm::start(&grammar).unwrap();
        assert!(fsm.is_final());
        let mut logits = vec![0.0f32; 3];
        mask_logits_by_grammar(&fsm, &tok, &mut logits);
        assert!(logits[0].is_finite()); // EOS ok
        assert!(logits[1].is_finite()); // "a" (take fork)
        assert!(logits[2].is_infinite()); // "b" rejected
    }

    #[test]
    fn masks_eos_when_not_final() {
        let tok = MockTokenizer::new(&["", "a"], 0);
        let grammar = parse_gbnf(r#"root ::= "a""#).unwrap();
        let fsm = Fsm::start(&grammar).unwrap();
        assert!(!fsm.is_final());
        let mut logits = vec![0.0f32; 2];
        mask_logits_by_grammar(&fsm, &tok, &mut logits);
        assert!(logits[0].is_infinite() && logits[0] < 0.0);
        assert!(logits[1].is_finite());
    }

    #[test]
    fn masks_control_tokens_with_empty_text() {
        // id 2 is a control token: empty decoded text.
        let tok = MockTokenizer::new(&["", "y", ""], 0);
        let grammar = parse_gbnf(r#"root ::= "yes""#).unwrap();
        let fsm = Fsm::start(&grammar).unwrap();
        let mut logits = vec![0.0f32; 3];
        mask_logits_by_grammar(&fsm, &tok, &mut logits);
        assert!(logits[0].is_infinite()); // EOS
        assert!(logits[1].is_finite()); // "y"
        assert!(logits[2].is_infinite()); // control
    }

    #[test]
    fn skips_already_masked_logits() {
        let tok = MockTokenizer::new(&["", "y"], 0);
        let grammar = parse_gbnf(r#"root ::= "y""#).unwrap();
        let fsm = Fsm::start(&grammar).unwrap();
        let mut logits = vec![f32::NEG_INFINITY; 2];
        mask_logits_by_grammar(&fsm, &tok, &mut logits);
        // Still masked — the pass never re-examines dead logits.
        assert!(logits[0].is_infinite());
        assert!(logits[1].is_infinite());
    }

    #[test]
    fn advance_on_emit_advances_state() {
        // Multi-char token "es" tests UTF-8 char-by-char advance.
        let tok = MockTokenizer::new(&["", "y", "es"], 0);
        let grammar = parse_gbnf(r#"root ::= "yes""#).unwrap();
        let mut fsm = Fsm::start(&grammar).unwrap();
        advance_fsm_on_emit(&mut fsm, &tok, 1).unwrap();
        assert!(!fsm.is_final());
        advance_fsm_on_emit(&mut fsm, &tok, 2).unwrap();
        assert!(fsm.is_final());
    }

    #[test]
    fn advance_on_emit_eos_is_noop() {
        let tok = MockTokenizer::new(&["", "y"], 0);
        let grammar = parse_gbnf(r#"root ::= "y""#).unwrap();
        let mut fsm = Fsm::start(&grammar).unwrap();
        advance_fsm_on_emit(&mut fsm, &tok, 0).unwrap();
        assert!(!fsm.is_final());
    }

    #[test]
    fn advance_on_emit_rejects_invalid_token() {
        // The mask should have removed "x", but if the driver misuses
        // the API we must fail loudly rather than corrupt state.
        let tok = MockTokenizer::new(&["", "x"], 0);
        let grammar = parse_gbnf(r#"root ::= "y""#).unwrap();
        let mut fsm = Fsm::start(&grammar).unwrap();
        assert!(advance_fsm_on_emit(&mut fsm, &tok, 1).is_err());
    }

    #[test]
    fn utf8_multibyte_token_ok() {
        // Terminal contains a multibyte char.
        let tok = MockTokenizer::new(&["", "あ"], 0);
        let grammar = parse_gbnf(r#"root ::= "あ""#).unwrap();
        let fsm = Fsm::start(&grammar).unwrap();
        let mut logits = vec![0.0f32; 2];
        mask_logits_by_grammar(&fsm, &tok, &mut logits);
        assert!(logits[0].is_infinite()); // EOS
        assert!(logits[1].is_finite()); // "あ" accepted
        let mut fsm2 = Fsm::start(&grammar).unwrap();
        advance_fsm_on_emit(&mut fsm2, &tok, 1).unwrap();
        assert!(fsm2.is_final());
    }
}

#[cfg(test)]
mod fold_tests {
    /// The tie law of the max fold: a NaN is skipped, so the others are compared against the finite
    /// maximum (the NaN entry and the sum stay NaN, and the row is then not normalised).
    #[test]
    fn the_softmax_fold_skips_a_nan_in_the_maximum() {
        let mut v = [1.0_f32, 2.0, f32::from_bits(0x7fc0_dead)];
        super::softmax_of_checked(&mut v);
        assert_eq!(v[0].to_bits(), crate::lawmath::exp(-1.0).to_bits());
        assert_eq!(v[1].to_bits(), 1.0_f32.to_bits());
        assert!(v[2].is_nan());
    }
}
