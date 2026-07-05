//! sampling.

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
