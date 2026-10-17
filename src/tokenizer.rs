//! tokenizer.

use std::collections::BTreeMap;

use crate::lookup::LookupMap;

// ---------------------------------------------------------------------------
// BPE Tokenizer
// ---------------------------------------------------------------------------

/// A byte-pair encoding merge rule.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MergeRule {
    pub left: String,
    pub right: String,
    pub merged: String,
}

/// BPE tokenizer that encodes text into token IDs and decodes back.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    merges: Vec<MergeRule>,
    vocab: LookupMap<String, u32>,
    inverse_vocab: LookupMap<u32, String>,
}

impl BpeTokenizer {
    /// Create a new tokenizer from merge rules and `(string, id)` pairs (a
    /// `HashMap<String, u32>` or any other map or list of pairs).
    ///
    /// The result does not depend on the order of the pairs: a string given
    /// more than once keeps its smallest id, and an id shared by several
    /// strings decodes to the smallest string.
    #[must_use]
    pub fn new(merges: Vec<MergeRule>, vocab: impl IntoIterator<Item = (String, u32)>) -> Self {
        let mut ordered: BTreeMap<String, u32> = BTreeMap::new();
        for (k, v) in vocab {
            ordered
                .entry(k)
                .and_modify(|id| *id = (*id).min(v))
                .or_insert(v);
        }
        // strings in ascending order, so the first one stored for an id is the smallest
        let mut inverse_vocab: LookupMap<u32, String> = LookupMap::with_capacity(ordered.len());
        for (k, &v) in &ordered {
            inverse_vocab.entry(v).or_insert_with(|| k.clone());
        }
        Self {
            merges,
            vocab: ordered.into_iter().collect(),
            inverse_vocab,
        }
    }

    /// Build a simple character-level tokenizer with optional merges.
    #[must_use]
    pub fn from_chars(text: &str, merges: Vec<MergeRule>) -> Self {
        let mut vocab = BTreeMap::new();
        let mut id = 0u32;
        for ch in text.chars() {
            let s = ch.to_string();
            if let std::collections::btree_map::Entry::Vacant(e) = vocab.entry(s) {
                e.insert(id);
                id += 1;
            }
        }
        for rule in &merges {
            if !vocab.contains_key(&rule.merged) {
                vocab.insert(rule.merged.clone(), id);
                id += 1;
            }
        }
        Self::new(merges, vocab)
    }

    /// Encode text into a sequence of token IDs.
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        for rule in &self.merges {
            let mut i = 0;
            while i + 1 < tokens.len() {
                if tokens[i] == rule.left && tokens[i + 1] == rule.right {
                    tokens[i].clone_from(&rule.merged);
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        tokens
            .iter()
            .map(|t| self.vocab.get(t).copied().unwrap_or(0))
            .collect()
    }

    /// Decode a sequence of token IDs back to text.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|id| self.inverse_vocab.get(id))
            .cloned()
            .collect()
    }

    /// Vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}
