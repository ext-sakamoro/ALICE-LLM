//! sampling.

#[cfg(feature = "grammar")]
use crate::grammar::{Fsm, FsmError};

// Sampling
// ---------------------------------------------------------------------------

/// Apply temperature scaling to logits (in-place).
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature < f32::EPSILON {
        return;
    }
    let inv_t = 1.0 / temperature;
    for l in logits.iter_mut() {
        *l *= inv_t;
    }
}

/// Top-k filtering: keep only the top k logits, set the rest to -inf.
///
/// Uses `select_nth_unstable_by` (partial sort, O(N)) to find the K-th
/// largest logit without fully sorting the whole vocabulary — matters
/// once `vocab_size` grows past 10k.
pub fn top_k_filter(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
    // O(N) partial sort: locate the K-th largest value.
    let mut vals: Vec<f32> = logits.to_vec();
    let kth = k - 1;
    vals.select_nth_unstable_by(kth, |a, b| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = vals[kth];
    for l in logits.iter_mut() {
        if *l < threshold {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Top-p (nucleus) filtering: keep smallest set of tokens whose
/// cumulative probability exceeds p.
pub fn top_p_filter(logits: &mut [f32], p: f32) {
    if p >= 1.0 {
        return;
    }

    let mut probs = logits.to_vec();
    softmax_inplace(&mut probs);

    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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
}

/// Softmax in-place.
pub fn softmax_inplace(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let max_val = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// Softmax returning a new vector.
#[must_use]
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut v = logits.to_vec();
    softmax_inplace(&mut v);
    v
}

/// Argmax sampling (greedy).
#[must_use]
pub fn sample_argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i)
}

/// Deterministic weighted sampling using a provided random value in `[0, 1)`.
#[must_use]
pub fn sample_with_random(probs: &[f32], rand_val: f32) -> usize {
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if rand_val < cumsum {
            return i;
        }
    }
    probs.len().saturating_sub(1)
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

#[cfg(all(feature = "grammar", feature = "gguf"))]
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
/// call, which is bounded by the number of live cursors. Adequate for
/// small grammars (JSON, LOL DSL) and MVP; heavy grammars may need a
/// pre-computed token → grammar-legal cache in a future revision.
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
