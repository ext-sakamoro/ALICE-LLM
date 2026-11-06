//! Sampling refuses logits it cannot sample from, and keeps every result on the ones it can.
//!
//! oracle: `*_ref` below are the implementations these functions had before they refused anything,
//! kept word for word, and the results on finite logits and `-inf` masks must have the same bits.
//! (`softmax` keeps its bits under the recorded hash in `tests/semantics_golden.rs`.)
//!
//! What is refused follows from IEEE 754: `NaN` has no order, so no threshold or maximum exists, and
//! `+inf - +inf` is `NaN`, so `softmax` has no value. `-inf` is how a token is masked and stays legal;
//! when every token is masked, `exp(-inf - -inf)` is `NaN` and there is nothing to sample.

use alice_llm::sampling::{
    apply_temperature, sample_argmax, sample_with_random, softmax, softmax_inplace, top_k_filter,
    top_p_filter, SamplingError,
};
use SamplingError::{AllMasked, InvalidParameter, InvalidProbability, NonFinite};

const NAN: f32 = f32::NAN;
const INF: f32 = f32::INFINITY;
const MASK: f32 = f32::NEG_INFINITY;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn top_k_ref(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
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

fn top_p_ref(logits: &mut [f32], p: f32) {
    if p >= 1.0 {
        return;
    }
    let probs = softmax(logits).unwrap();
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

fn argmax_ref(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i)
}

/// Word for word, but for the last line: the fallback changed on purpose, and only past the sum, and
/// only when the last tokens have no probability.
fn sample_ref(probs: &[f32], rand_val: f32) -> usize {
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if rand_val < cumsum {
            return i;
        }
    }
    probs.iter().rposition(|&p| p > 0.0).unwrap_or(0)
}

/// A fixed sequence (splitmix64), so every run compares the same inputs.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Finite logits of mixed scale, with ties and at least one token not masked.
    fn logits(&mut self) -> Vec<f32> {
        let n = 1 + (self.next() % 64) as usize;
        let mut v: Vec<f32> = (0..n)
            .map(|_| match self.next() % 8 {
                0 => MASK,
                1 => 0.0,
                2 => -0.0,
                3 => 1.0,
                4 => f32::MAX,
                5 => f32::MIN,
                _ => (self.unit() - 0.5) * 40.0,
            })
            .collect();
        if v.iter().all(|x| *x == MASK) {
            v[0] = 1.0;
        }
        v
    }
}

#[test]
fn finite_logits_and_masks_keep_their_bits() {
    let mut rng = Rng(0x4c4f_4749);
    let mut compared = 0usize;
    for _ in 0..20_000 {
        let logits = rng.logits();
        let k = (rng.next() % 70) as usize;
        let (mut a, mut b) = (logits.clone(), logits.clone());
        top_k_filter(&mut a, k).unwrap();
        top_k_ref(&mut b, k);
        assert_eq!(bits(&a), bits(&b), "top_k {k} {logits:?}");

        let p = rng.unit() * 1.2;
        let (mut a, mut b) = (logits.clone(), logits.clone());
        top_p_filter(&mut a, p).unwrap();
        top_p_ref(&mut b, p);
        assert_eq!(bits(&a), bits(&b), "top_p {p} {logits:?}");

        assert_eq!(
            sample_argmax(&logits),
            Ok(argmax_ref(&logits)),
            "{logits:?}"
        );

        let probs = softmax(&logits).unwrap();
        let mut in_place = logits.clone();
        softmax_inplace(&mut in_place).unwrap();
        assert_eq!(bits(&probs), bits(&in_place));
        let r = rng.unit();
        assert_eq!(
            sample_with_random(&probs, r),
            Ok(sample_ref(&probs, r)),
            "{probs:?}"
        );
        compared += 1;
    }
    assert_eq!(compared, 20_000);
}

/// `(what, logits, the error)`: every entry point that takes logits.
fn refused() -> Vec<(&'static str, Vec<f32>, SamplingError)> {
    vec![
        ("NaN first", vec![NAN, 1.0, 2.0], NonFinite { index: 0 }),
        ("NaN last", vec![1.0, 2.0, NAN], NonFinite { index: 2 }),
        ("all NaN", vec![NAN, NAN, NAN], NonFinite { index: 0 }),
        ("+inf", vec![1.0, INF, 2.0], NonFinite { index: 1 }),
        (
            "NaN beside a mask",
            vec![MASK, NAN, 2.0],
            NonFinite { index: 1 },
        ),
        ("all masked", vec![MASK, MASK, MASK], AllMasked),
        ("one, masked", vec![MASK], AllMasked),
    ]
}

#[test]
fn every_entry_point_refuses() {
    let rows = refused();
    assert!(rows.len() >= 7);
    for (what, logits, err) in &rows {
        // k and p cover the early returns: k = 0, k >= len, p >= 1
        for k in [0, 1, 2, 3, 100] {
            let mut l = logits.clone();
            assert_eq!(top_k_filter(&mut l, k), Err(*err), "top_k {k} {what}");
            assert_eq!(
                bits(&l),
                bits(logits),
                "top_k {k} {what}: changed what it refused"
            );
        }
        for p in [0.0, 0.5, 1.0, 2.0] {
            let mut l = logits.clone();
            assert_eq!(top_p_filter(&mut l, p), Err(*err), "top_p {p} {what}");
            assert_eq!(
                bits(&l),
                bits(logits),
                "top_p {p} {what}: changed what it refused"
            );
        }
        assert_eq!(sample_argmax(logits), Err(*err), "argmax {what}");
        assert_eq!(softmax(logits), Err(*err), "softmax {what}");
        let mut l = logits.clone();
        assert_eq!(softmax_inplace(&mut l), Err(*err), "softmax_inplace {what}");
        assert_eq!(
            bits(&l),
            bits(logits),
            "softmax_inplace {what}: changed what it refused"
        );
    }
}

#[test]
fn a_mask_is_legal_and_stays_a_mask() {
    let logits = [MASK, 1.0, MASK, 3.0, 2.0];
    let mut l = logits;
    top_k_filter(&mut l, 2).unwrap();
    assert_eq!(bits(&l), bits(&[MASK, MASK, MASK, 3.0, 2.0]));
    let mut l = logits;
    top_p_filter(&mut l, 0.5).unwrap();
    assert_eq!(bits(&l), bits(&[MASK, MASK, MASK, 3.0, MASK]));
    assert_eq!(sample_argmax(&logits), Ok(3));
    let probs = softmax(&logits).unwrap();
    assert_eq!((probs[0], probs[2]), (0.0, 0.0));
    assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6, "{probs:?}");
}

#[test]
fn no_tokens_is_not_an_error() {
    assert_eq!(top_k_filter(&mut [], 1), Ok(()));
    assert_eq!(top_p_filter(&mut [], 0.5), Ok(()));
    assert_eq!(softmax(&[]), Ok(vec![]));
    assert_eq!(sample_argmax(&[]), Ok(0));
    assert_eq!(sample_with_random(&[], 0.5), Ok(0));
}

#[test]
fn probabilities_and_parameters() {
    assert_eq!(
        sample_with_random(&[0.5, NAN], 0.1),
        Err(InvalidProbability { index: 1 })
    );
    assert_eq!(
        sample_with_random(&[INF, 0.5], 0.1),
        Err(InvalidProbability { index: 0 })
    );
    assert_eq!(
        sample_with_random(&[0.5, MASK], 0.1),
        Err(InvalidProbability { index: 1 })
    );
    assert_eq!(
        sample_with_random(&[0.5, -0.1], 0.1),
        Err(InvalidProbability { index: 1 })
    );
    assert_eq!(sample_with_random(&[0.0, 0.0], 0.1), Err(AllMasked));
    assert_eq!(sample_with_random(&[0.5, 0.5], NAN), Err(InvalidParameter));
    assert_eq!(sample_with_random(&[-0.0, 1.0], 0.5), Ok(1));
    assert_eq!(sample_with_random(&[0.5, 0.5], 1.0), Ok(1));
    let mut l = [1.0, 2.0];
    assert_eq!(top_p_filter(&mut l, NAN), Err(InvalidParameter));
}

/// A random value past the sum of the probabilities (they sum to a little under 1, or the value is
/// not in `[0, 1)`) picks the last token that has any probability: a token with probability 0, such
/// as one masked by a grammar, is never the answer.
#[test]
fn past_the_sum_is_the_last_token_with_any_probability() {
    assert_eq!(sample_with_random(&[0.3, 0.3, 0.3, 0.0], 0.95), Ok(2));
    assert_eq!(sample_with_random(&[0.3, 0.3, 0.3, 0.0], 1.0), Ok(2));
    assert_eq!(sample_with_random(&[0.5, 0.0, 0.0], 0.75), Ok(0));
    assert_eq!(sample_with_random(&[0.0, 0.5, -0.0], 0.75), Ok(1));
    assert_eq!(sample_with_random(&[0.25, 0.25], 2.0), Ok(1));
    // a negative value is below the first cumulative sum that is above it
    assert_eq!(sample_with_random(&[0.0, 0.5], -1.0), Ok(0));
}

#[test]
fn a_token_with_no_probability_is_never_sampled() {
    let mut rng = Rng(0x5a45_524f);
    let mut masked_seen = 0usize;
    for _ in 0..200_000 {
        let n = 1 + (rng.next() % 16) as usize;
        let mut logits: Vec<f32> = (0..n).map(|_| (rng.unit() - 0.5) * 20.0).collect();
        for l in &mut logits {
            if rng.next() % 3 == 0 {
                *l = MASK;
                masked_seen += 1;
            }
        }
        if logits.iter().all(|l| *l == MASK) {
            continue;
        }
        let probs = softmax(&logits).unwrap();
        // the top of the range is where rounding leaves the sum short
        let r = if rng.next() % 2 == 0 { 1.0 - f32::EPSILON / 2.0 } else { rng.unit() };
        let i = sample_with_random(&probs, r).unwrap();
        assert!(probs[i] > 0.0, "token {i} of {probs:?} at {r}");
    }
    assert!(masked_seen > 100_000, "masked {masked_seen}");
}

#[test]
fn a_temperature_that_breaks_the_logits_is_refused_at_the_next_step() {
    // apply_temperature keeps its signature: what it produces meets the check of the next call
    for t in [NAN, 1e-3] {
        let mut l = [1.0, f32::MAX];
        apply_temperature(&mut l, t);
        assert!(
            matches!(top_k_filter(&mut l, 1), Err(NonFinite { .. })),
            "temperature {t}: {l:?}"
        );
    }
}
