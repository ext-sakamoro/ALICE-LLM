//! tokenizer.

use std::collections::HashMap;

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
    vocab: HashMap<String, u32>,
    inverse_vocab: HashMap<u32, String>,
}

impl BpeTokenizer {
    /// Create a new tokenizer from merge rules and a vocabulary map.
    #[must_use]
    pub fn new(merges: Vec<MergeRule>, vocab: HashMap<String, u32>) -> Self {
        let inverse_vocab: HashMap<u32, String> =
            vocab.iter().map(|(k, &v)| (v, k.clone())).collect();
        Self {
            merges,
            vocab,
            inverse_vocab,
        }
    }

    /// Build a simple character-level tokenizer with optional merges.
    #[must_use]
    pub fn from_chars(text: &str, merges: Vec<MergeRule>) -> Self {
        let mut vocab = HashMap::new();
        let mut id = 0u32;
        for ch in text.chars() {
            let s = ch.to_string();
            if let std::collections::hash_map::Entry::Vacant(e) = vocab.entry(s) {
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
