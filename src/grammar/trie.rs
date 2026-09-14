//! Token trie for grammar-constrained decoding (Phase X.8 B-10).
//!
//! The naïve mask ([`crate::sampling::mask_logits_by_grammar`]) probes
//! every vocabulary entry independently: `vocab_size` × (decode the token
//! text + clone the FSM + advance it char by char). Measured on
//! MiniCPM5-2B (vocab 130,560) against `lol.gbnf` that is ~8 s per decode
//! step at the grammar root and ~95 % of end-to-end latency — the model
//! forward is ~60 ms.
//!
//! This module shares work across tokens with a common prefix. All token
//! texts are inserted once into a character trie; per step the trie is
//! walked depth-first from the current FSM state, and a subtree is pruned
//! as soon as the FSM rejects its edge char. The FSM is cloned only along
//! *accepted* edges, so the cost is proportional to the number of distinct
//! prefixes the grammar admits at that state, not to the vocabulary size.
//!
//! # Parity contract
//!
//! [`TokenTrie::allowed_tokens`] admits exactly the set of ids for which
//! `fsm.accepts_str(text_of(id))` is true and `text_of(id)` is non-empty
//! (empty / control tokens sit at the trie root and are never admitted).
//! EOS is handled by the mask layer, not here.
//!
//! # Grammar states with (near-)universal char acceptance
//!
//! A state that accepts almost any char (e.g. inside a `//` line comment,
//! `noteol*`) admits most of the vocabulary; the walk then visits most of
//! the trie and degrades toward the naïve cost. Grammars intended for LLM
//! emission should avoid such rules.

use super::fsm::Fsm;
use crate::sampling::GrammarTokenizer;

/// One trie node: sorted child edges + the token ids whose text ends here.
#[derive(Debug, Clone, Default)]
struct Node {
    /// `(edge char, child index)`, sorted by char for binary search.
    children: Vec<(char, u32)>,
    /// Token ids whose full text terminates at this node (usually 0 or 1;
    /// >1 when several ids decode to identical text).
    tokens: Vec<u32>,
}

/// Character trie over a tokenizer's vocabulary.
///
/// Build once per tokenizer (≈ tens of ms for 130 k tokens) and reuse for
/// every decode step and every grammar.
#[derive(Debug, Clone)]
pub struct TokenTrie {
    nodes: Vec<Node>,
    vocab_size: usize,
    /// Number of ids with empty text (control tokens) — never admitted.
    empty_tokens: usize,
}

impl TokenTrie {
    /// Insert the text of every id in `0..vocab_size`.
    ///
    /// Ids whose `text_of` is empty are counted in `empty_tokens` and
    /// attached to the root, which the walk never reports.
    #[must_use]
    pub fn build<T: GrammarTokenizer + ?Sized>(tokenizer: &T, vocab_size: usize) -> Self {
        let mut nodes: Vec<Node> = vec![Node::default()];
        let mut empty_tokens = 0usize;
        for id in 0..vocab_size {
            let id_u32 = u32::try_from(id).expect("vocab_size exceeds u32");
            let text = tokenizer.text_of(id_u32);
            if text.is_empty() {
                empty_tokens += 1;
                continue;
            }
            let mut cur = 0usize;
            for ch in text.chars() {
                let pos = nodes[cur].children.binary_search_by_key(&ch, |&(c, _)| c);
                cur = match pos {
                    Ok(i) => nodes[cur].children[i].1 as usize,
                    Err(i) => {
                        let next = nodes.len();
                        let next_u32 = u32::try_from(next).expect("trie exceeds u32 nodes");
                        nodes[cur].children.insert(i, (ch, next_u32));
                        nodes.push(Node::default());
                        next
                    }
                };
            }
            nodes[cur].tokens.push(id_u32);
        }
        Self {
            nodes,
            vocab_size,
            empty_tokens,
        }
    }

    /// Vocabulary size the trie was built for.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Total trie nodes (root included).
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of ids with empty text (never admitted by the walk).
    #[must_use]
    pub const fn empty_token_count(&self) -> usize {
        self.empty_tokens
    }

    /// Collect every token id whose full text `fsm` would consume from its
    /// current state. `out` is cleared first. Does not mutate `fsm`.
    ///
    /// Returns the number of FSM advances performed (a cost metric for
    /// benchmarks / diagnostics).
    pub fn allowed_tokens(&self, fsm: &Fsm<'_>, out: &mut Vec<u32>) -> usize {
        out.clear();
        let mut advances = 0usize;
        // Root tokens (empty text) are intentionally skipped.
        self.walk(0, fsm, out, &mut advances);
        advances
    }

    fn walk(&self, node: usize, fsm: &Fsm<'_>, out: &mut Vec<u32>, advances: &mut usize) {
        for &(ch, child) in &self.nodes[node].children {
            if !fsm.accepts(ch) {
                continue;
            }
            let mut next = fsm.clone();
            if next.advance(ch).is_err() {
                // `accepts` is a head-only probe; `advance` additionally runs
                // expansion, which can fail on depth cap. Treat as rejection.
                continue;
            }
            *advances += 1;
            let child = child as usize;
            out.extend_from_slice(&self.nodes[child].tokens);
            self.walk(child, &next, out, advances);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::gbnf::parse_gbnf;
    use super::*;

    struct MockTokenizer {
        tokens: Vec<String>,
        eos: u32,
    }

    impl GrammarTokenizer for MockTokenizer {
        fn text_of(&self, id: u32) -> String {
            self.tokens.get(id as usize).cloned().unwrap_or_default()
        }
        fn eos_id(&self) -> u32 {
            self.eos
        }
    }

    fn mock(words: &[&str]) -> MockTokenizer {
        MockTokenizer {
            tokens: words.iter().map(|s| (*s).to_string()).collect(),
            eos: 0,
        }
    }

    fn naive_allowed(fsm: &Fsm<'_>, tok: &MockTokenizer) -> Vec<u32> {
        (0..tok.tokens.len() as u32)
            .filter(|&id| {
                let t = tok.text_of(id);
                !t.is_empty() && fsm.accepts_str(&t)
            })
            .collect()
    }

    #[test]
    fn build_shares_prefixes_and_counts_empties() {
        let tok = mock(&["", "sp", "sph", "sphere", "s", "box", ""]);
        let trie = TokenTrie::build(&tok, tok.tokens.len());
        // root + s,p,h,e,r,e + b,o,x = 10 nodes
        assert_eq!(trie.node_count(), 10);
        assert_eq!(trie.empty_token_count(), 2);
        assert_eq!(trie.vocab_size(), 7);
    }

    #[test]
    fn allowed_matches_naive_on_terminal_grammar() {
        let tok = mock(&["", "y", "ye", "yes", "yesterday", "n", "s", "es"]);
        let trie = TokenTrie::build(&tok, tok.tokens.len());
        let grammar = parse_gbnf(r#"root ::= "yes""#).unwrap();
        let mut fsm = Fsm::start(&grammar).unwrap();

        let mut got = Vec::new();
        trie.allowed_tokens(&fsm, &mut got);
        got.sort_unstable();
        assert_eq!(got, naive_allowed(&fsm, &tok));
        assert_eq!(got, vec![1, 2, 3]);

        fsm.advance('y').unwrap();
        trie.allowed_tokens(&fsm, &mut got);
        got.sort_unstable();
        assert_eq!(got, naive_allowed(&fsm, &tok));
        assert_eq!(got, vec![7]); // "es"
    }

    #[test]
    fn allowed_matches_naive_across_lol_like_walk() {
        let tok = mock(&[
            "", "sphere", "sp", "s", "(", "1", ".", "5", ")", "1.5", "(1", "1.", ".5", "))", " ",
            "box", "b", "e(", "here(", "5)", "0", "-", "-1",
        ]);
        let trie = TokenTrie::build(&tok, tok.tokens.len());
        let grammar = parse_gbnf(
            r#"root ::= ws expr ws
ws ::= [ \t]*
expr ::= "sphere" "(" number ")" | "box" "(" number ")"
number ::= "-"? [0-9]+ ("." [0-9]+)?"#,
        )
        .unwrap();
        let mut fsm = Fsm::start(&grammar).unwrap();
        let mut got = Vec::new();
        for ch in "sphere(1.5)".chars() {
            trie.allowed_tokens(&fsm, &mut got);
            got.sort_unstable();
            assert_eq!(got, naive_allowed(&fsm, &tok), "state before {ch:?}");
            fsm.advance(ch).unwrap();
        }
        // Final state: only ws tokens remain.
        trie.allowed_tokens(&fsm, &mut got);
        got.sort_unstable();
        assert_eq!(got, naive_allowed(&fsm, &tok));
        assert_eq!(got, vec![14]);
    }

    #[test]
    fn duplicate_texts_report_every_id() {
        let tok = mock(&["", "a", "a", "ab"]);
        let trie = TokenTrie::build(&tok, tok.tokens.len());
        let grammar = parse_gbnf(r#"root ::= "ab""#).unwrap();
        let fsm = Fsm::start(&grammar).unwrap();
        let mut got = Vec::new();
        trie.allowed_tokens(&fsm, &mut got);
        got.sort_unstable();
        assert_eq!(got, vec![1, 2, 3]);
    }
}
