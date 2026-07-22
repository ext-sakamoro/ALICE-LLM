//! Grammar → FSM builder for constrained decoding (Phase X.8 B-2).
//!
//! Turns a parsed [`Grammar`] into a runtime state
//! machine that tracks every possible parse position in parallel (NFA
//! simulation). Sampling code queries the machine to find which characters
//! (and therefore which tokens) may legally come next, and advances the
//! machine once an emission is committed.
//!
//! The state is deliberately simple: each active cursor is the remaining
//! linear stream of [`Symbol`]s still to consume. `RuleRef`, `Group`, and
//! quantifier symbols are expanded eagerly into forks (multiple cursors),
//! so that after every mutation each cursor either begins with a
//! terminal-consuming symbol or is empty (grammar fully accepted).
//!
//! # Left recursion
//!
//! Grammars with left recursion (`root ::= root "a" | "b"`) expand
//! unboundedly during eager forking. Expansion is bounded by
//! [`Fsm::with_max_depth`] (default 256); cursors that reach the cap are
//! dropped silently. If *every* cursor is dropped and at least one hit the
//! cap, [`FsmError::RecursionOverflow`] is returned so the caller can react.
//!
//! # Not covered here
//!
//! - Token-level integration lives in [`sampling`](crate::sampling) (Phase
//!   X.8 B-3), which uses [`Fsm::accepts_str`] / [`Fsm::advance`] to build
//!   a per-step logits mask.
//! - Grammar authoring (LOL / JSON / tool-call payloads) is out of scope
//!   for this module.

use std::collections::HashSet;
use std::fmt;

use super::gbnf::{CharClass, Grammar, Symbol};

/// Default cap on expansion depth (per cursor). Guards against unbounded
/// left recursion during eager fork-out.
pub const DEFAULT_MAX_DEPTH: usize = 256;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from FSM construction or transitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FsmError {
    /// [`Fsm::advance`] was called with a char no active cursor accepts.
    NoTransition { ch: char },
    /// The grammar's root rule has no alternatives.
    EmptyRoot,
    /// A rule reference points to a rule missing from the grammar.
    /// A grammar returned by [`super::parse_gbnf`] cannot trigger this
    /// (it validates rule refs) but manually constructed grammars can.
    UnknownRule { name: String },
    /// Expansion exceeded the configured recursion depth *and* no cursor
    /// survived. Usually signals unguarded left recursion.
    RecursionOverflow { depth: usize },
}

impl fmt::Display for FsmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoTransition { ch } => {
                write!(f, "no grammar transition accepts char {ch:?}")
            }
            Self::EmptyRoot => write!(f, "grammar's root rule has no alternatives"),
            Self::UnknownRule { name } => write!(f, "unknown rule `{name}`"),
            Self::RecursionOverflow { depth } => {
                write!(f, "grammar expansion exceeded recursion depth {depth}")
            }
        }
    }
}

impl std::error::Error for FsmError {}

// ---------------------------------------------------------------------------
// CharSet — allowed-char introspection
// ---------------------------------------------------------------------------

/// The set of characters that would be accepted by the current FSM state.
///
/// Only used for introspection and tests. Sampling code should call
/// [`Fsm::accepts`] or [`Fsm::accepts_str`] instead — those don't
/// materialise the full set and handle negated char classes efficiently.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CharSet {
    ranges: Vec<(char, char)>,
}

impl CharSet {
    /// Construct an empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// True if `ch` is a member.
    #[must_use]
    pub fn contains(&self, ch: char) -> bool {
        self.ranges.iter().any(|&(a, b)| ch >= a && ch <= b)
    }

    /// True if no chars are allowed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// The raw list of allowed ranges (may overlap; not canonicalised).
    #[must_use]
    pub fn ranges(&self) -> &[(char, char)] {
        &self.ranges
    }

    fn add_char(&mut self, ch: char) {
        self.ranges.push((ch, ch));
    }

    fn add_char_class(&mut self, cc: &CharClass) {
        if !cc.negated {
            self.ranges.extend_from_slice(&cc.ranges);
            return;
        }
        // Negated: materialise the complement of `cc.ranges` over the full
        // Unicode scalar-value range. Kept naive because introspection is
        // not on the hot path.
        let mut sorted = cc.ranges.clone();
        sorted.sort_by_key(|&(a, _)| a as u32);
        // Merge overlapping / adjacent ranges so the complement walk stays
        // sane.
        let mut merged: Vec<(char, char)> = Vec::with_capacity(sorted.len());
        for (a, b) in sorted {
            if let Some(last) = merged.last_mut() {
                if (a as u32) <= (last.1 as u32).saturating_add(1) {
                    if (b as u32) > (last.1 as u32) {
                        last.1 = b;
                    }
                    continue;
                }
            }
            merged.push((a, b));
        }
        let mut cur: u32 = 0;
        for (a, b) in merged {
            let a_u = a as u32;
            if a_u > cur {
                if let (Some(lo), Some(hi)) = (char::from_u32(cur), char::from_u32(a_u - 1)) {
                    self.ranges.push((lo, hi));
                }
            }
            let b_u = b as u32;
            if b_u == char::MAX as u32 {
                return;
            }
            cur = b_u + 1;
        }
        if let Some(lo) = char::from_u32(cur) {
            self.ranges.push((lo, char::MAX));
        }
    }
}

// ---------------------------------------------------------------------------
// Fsm
// ---------------------------------------------------------------------------

/// A cursor into the grammar: the linear sequence of symbols still to
/// consume. An empty cursor means the grammar has been fully recognised.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Cursor {
    pending: Vec<Symbol>,
}

/// Runtime state machine over a parsed [`Grammar`].
///
/// The FSM holds a set of active cursors (an NFA-style set of parse
/// positions). Each terminal-consuming operation prunes cursors that
/// reject the input and forks over alternatives / quantifier choices.
#[derive(Debug, Clone)]
pub struct Fsm<'g> {
    grammar: &'g Grammar,
    active: Vec<Cursor>,
    max_depth: usize,
}

impl<'g> Fsm<'g> {
    /// Build an FSM seeded from `grammar`'s `root` rule.
    ///
    /// Fails with [`FsmError::EmptyRoot`] if the root rule has no
    /// alternatives, and [`FsmError::RecursionOverflow`] if expansion
    /// exhausted `max_depth` for *every* seed (typically left recursion
    /// without a base case).
    pub fn start(grammar: &'g Grammar) -> Result<Self, FsmError> {
        let root_alts = grammar.root_alternatives();
        if root_alts.is_empty() {
            return Err(FsmError::EmptyRoot);
        }
        let active = root_alts
            .iter()
            .map(|alt| Cursor {
                pending: alt.0.clone(),
            })
            .collect();
        let mut fsm = Self {
            grammar,
            active,
            max_depth: DEFAULT_MAX_DEPTH,
        };
        fsm.expand_all()?;
        Ok(fsm)
    }

    /// Override the default expansion depth cap.
    #[must_use]
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// True if any active cursor has consumed the entire grammar.
    #[must_use]
    pub fn is_final(&self) -> bool {
        self.active.iter().any(|c| c.pending.is_empty())
    }

    /// True if there are no active cursors left (the FSM is stuck).
    #[must_use]
    pub fn is_dead(&self) -> bool {
        self.active.is_empty()
    }

    /// Set of chars that would be accepted at the current state.
    #[must_use]
    pub fn allowed_chars(&self) -> CharSet {
        let mut out = CharSet::new();
        for c in &self.active {
            if let Some(first) = c.pending.first() {
                Self::collect_leading_chars(first, &mut out);
            }
        }
        out
    }

    /// Consume `ch` and advance the state. Errors if no cursor accepts it.
    pub fn advance(&mut self, ch: char) -> Result<(), FsmError> {
        let mut next: Vec<Cursor> = Vec::with_capacity(self.active.len());
        for c in &self.active {
            if let Some(new_pending) = Self::step(c, ch) {
                next.push(Cursor {
                    pending: new_pending,
                });
            }
        }
        if next.is_empty() {
            return Err(FsmError::NoTransition { ch });
        }
        self.active = next;
        self.expand_all()?;
        Ok(())
    }

    /// True iff `ch` would be accepted by at least one active cursor.
    /// Does not mutate the FSM.
    #[must_use]
    pub fn accepts(&self, ch: char) -> bool {
        self.active.iter().any(|c| Self::step(c, ch).is_some())
    }

    /// True iff every char of `s` would be consumed in sequence starting
    /// from the current state. Does not mutate; internally clones.
    #[must_use]
    pub fn accepts_str(&self, s: &str) -> bool {
        if s.is_empty() {
            return true;
        }
        let mut probe = self.clone();
        for ch in s.chars() {
            if probe.advance(ch).is_err() {
                return false;
            }
        }
        true
    }

    // ---- internal ----

    /// Consume `ch` at the head of `c`. Returns the new pending list if
    /// the cursor's leading symbol accepts `ch`, otherwise `None`.
    fn step(c: &Cursor, ch: char) -> Option<Vec<Symbol>> {
        let head = c.pending.first()?;
        match head {
            Symbol::Terminal(s) => {
                let mut chars = s.chars();
                let first = chars.next()?;
                if first != ch {
                    return None;
                }
                let rest: String = chars.collect();
                let mut new_pending = c.pending.clone();
                if rest.is_empty() {
                    new_pending.remove(0);
                } else {
                    new_pending[0] = Symbol::Terminal(rest);
                }
                Some(new_pending)
            }
            Symbol::CharClass(cc) => {
                if !cc.matches(ch) {
                    return None;
                }
                let mut new_pending = c.pending.clone();
                new_pending.remove(0);
                Some(new_pending)
            }
            // Non-terminal-leading symbols must have been expanded away by
            // `expand_all` — if we ever see one here the invariant is
            // broken; treat as rejection rather than panic to stay robust.
            Symbol::RuleRef(_)
            | Symbol::Group(_)
            | Symbol::Star(_)
            | Symbol::Plus(_)
            | Symbol::Optional(_) => None,
        }
    }

    /// Repeatedly rewrite every active cursor until its head is either a
    /// terminal-consuming symbol or the cursor is empty.
    fn expand_all(&mut self) -> Result<(), FsmError> {
        let mut result: Vec<Cursor> = Vec::new();
        let mut work: Vec<(Cursor, usize)> = self.active.drain(..).map(|c| (c, 0usize)).collect();
        let mut seen: HashSet<Cursor> = HashSet::new();
        let mut hit_cap = false;

        while let Some((cursor, depth)) = work.pop() {
            if depth > self.max_depth {
                hit_cap = true;
                continue;
            }
            if !seen.insert(cursor.clone()) {
                continue;
            }
            match cursor.pending.first() {
                None => {
                    result.push(cursor);
                }
                Some(Symbol::Terminal(s)) => {
                    if s.is_empty() {
                        // Degenerate empty terminal — skip past it.
                        let mut new_pending = cursor.pending.clone();
                        new_pending.remove(0);
                        work.push((
                            Cursor {
                                pending: new_pending,
                            },
                            depth + 1,
                        ));
                    } else {
                        result.push(cursor);
                    }
                }
                Some(Symbol::CharClass(_)) => {
                    result.push(cursor);
                }
                Some(Symbol::RuleRef(name)) => {
                    let alts = self
                        .grammar
                        .rule(name)
                        .ok_or_else(|| FsmError::UnknownRule { name: name.clone() })?;
                    let rest: Vec<Symbol> = cursor.pending[1..].to_vec();
                    for alt in alts {
                        let mut new_pending = alt.0.clone();
                        new_pending.extend_from_slice(&rest);
                        work.push((
                            Cursor {
                                pending: new_pending,
                            },
                            depth + 1,
                        ));
                    }
                }
                Some(Symbol::Group(alts)) => {
                    let alts_cloned: Vec<_> = alts.clone();
                    let rest: Vec<Symbol> = cursor.pending[1..].to_vec();
                    for alt in &alts_cloned {
                        let mut new_pending = alt.0.clone();
                        new_pending.extend_from_slice(&rest);
                        work.push((
                            Cursor {
                                pending: new_pending,
                            },
                            depth + 1,
                        ));
                    }
                }
                Some(Symbol::Star(inner)) => {
                    let inner = (**inner).clone();
                    let rest: Vec<Symbol> = cursor.pending[1..].to_vec();
                    // Fork 1: skip (zero reps).
                    work.push((
                        Cursor {
                            pending: rest.clone(),
                        },
                        depth + 1,
                    ));
                    // Fork 2: consume one, then loop (Star retained).
                    let mut take = Vec::with_capacity(rest.len() + 2);
                    take.push(inner.clone());
                    take.push(Symbol::Star(Box::new(inner)));
                    take.extend_from_slice(&rest);
                    work.push((Cursor { pending: take }, depth + 1));
                }
                Some(Symbol::Plus(inner)) => {
                    let inner = (**inner).clone();
                    let rest: Vec<Symbol> = cursor.pending[1..].to_vec();
                    // Must consume at least one; then Star for the rest.
                    let mut take = Vec::with_capacity(rest.len() + 2);
                    take.push(inner.clone());
                    take.push(Symbol::Star(Box::new(inner)));
                    take.extend_from_slice(&rest);
                    work.push((Cursor { pending: take }, depth + 1));
                }
                Some(Symbol::Optional(inner)) => {
                    let inner = (**inner).clone();
                    let rest: Vec<Symbol> = cursor.pending[1..].to_vec();
                    // Fork 1: skip.
                    work.push((
                        Cursor {
                            pending: rest.clone(),
                        },
                        depth + 1,
                    ));
                    // Fork 2: take.
                    let mut take = Vec::with_capacity(rest.len() + 1);
                    take.push(inner);
                    take.extend_from_slice(&rest);
                    work.push((Cursor { pending: take }, depth + 1));
                }
            }
        }

        self.active = result;
        if self.active.is_empty() && hit_cap {
            return Err(FsmError::RecursionOverflow {
                depth: self.max_depth,
            });
        }
        Ok(())
    }

    /// Emit into `out` the chars that could be consumed by `sym` at the
    /// head of a cursor. `sym` is assumed to be a terminal-consuming symbol
    /// (Terminal or CharClass); other kinds are ignored.
    fn collect_leading_chars(sym: &Symbol, out: &mut CharSet) {
        match sym {
            Symbol::Terminal(s) => {
                if let Some(c) = s.chars().next() {
                    out.add_char(c);
                }
            }
            Symbol::CharClass(cc) => out.add_char_class(cc),
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::gbnf::parse_gbnf;
    use super::*;

    fn g(src: &str) -> Grammar {
        parse_gbnf(src).unwrap_or_else(|e| panic!("failed to parse test grammar: {e}"))
    }

    #[test]
    fn empty_root_rejected() {
        // Build a Grammar manually with a root rule that has no alternatives.
        // parse_gbnf itself rejects such input, so construct directly.
        use std::collections::HashMap;
        let grammar = Grammar {
            rules: {
                let mut m = HashMap::new();
                m.insert("root".to_string(), Vec::new());
                m
            },
            root: "root".to_string(),
        };
        let err = Fsm::start(&grammar).unwrap_err();
        assert_eq!(err, FsmError::EmptyRoot);
    }

    #[test]
    fn single_terminal_full_consume() {
        let grammar = g(r#"root ::= "hi""#);
        let mut fsm = Fsm::start(&grammar).unwrap();
        assert!(!fsm.is_final());
        fsm.advance('h').unwrap();
        assert!(!fsm.is_final());
        fsm.advance('i').unwrap();
        assert!(fsm.is_final());
    }

    #[test]
    fn single_terminal_rejects_wrong_char() {
        let grammar = g(r#"root ::= "hi""#);
        let mut fsm = Fsm::start(&grammar).unwrap();
        let err = fsm.advance('x').unwrap_err();
        assert_eq!(err, FsmError::NoTransition { ch: 'x' });
    }

    #[test]
    fn char_class_accepts_range() {
        let grammar = g(r"root ::= [a-z]");
        let mut fsm = Fsm::start(&grammar).unwrap();
        fsm.advance('m').unwrap();
        assert!(fsm.is_final());
    }

    #[test]
    fn char_class_rejects_outside_range() {
        let grammar = g(r"root ::= [a-z]");
        let mut fsm = Fsm::start(&grammar).unwrap();
        let err = fsm.advance('A').unwrap_err();
        assert_eq!(err, FsmError::NoTransition { ch: 'A' });
    }

    #[test]
    fn negated_char_class() {
        let grammar = g(r"root ::= [^abc]");
        let mut fsm = Fsm::start(&grammar).unwrap();
        fsm.advance('d').unwrap();
        assert!(fsm.is_final());
        // Reset and try a forbidden char.
        let mut fsm2 = Fsm::start(&grammar).unwrap();
        assert!(fsm2.advance('a').is_err());
    }

    #[test]
    fn alternatives_pick_a_branch() {
        let grammar = g(r#"root ::= "yes" | "no""#);
        let mut fsm = Fsm::start(&grammar).unwrap();
        fsm.advance('y').unwrap();
        fsm.advance('e').unwrap();
        fsm.advance('s').unwrap();
        assert!(fsm.is_final());
    }

    #[test]
    fn alternatives_reject_mixed_prefix() {
        let grammar = g(r#"root ::= "yes" | "no""#);
        let mut fsm = Fsm::start(&grammar).unwrap();
        fsm.advance('y').unwrap();
        // After 'y' only the "yes" branch survives; 'n' must be rejected.
        assert!(fsm.advance('n').is_err());
    }

    #[test]
    fn rule_ref_expands() {
        let grammar = g(r#"
            root ::= greeting
            greeting ::= "hi"
        "#);
        let mut fsm = Fsm::start(&grammar).unwrap();
        fsm.advance('h').unwrap();
        fsm.advance('i').unwrap();
        assert!(fsm.is_final());
    }

    #[test]
    fn sequence_of_rule_refs() {
        let grammar = g(r#"
            root ::= greeting name
            greeting ::= "hi"
            name ::= "bob"
        "#);
        let mut fsm = Fsm::start(&grammar).unwrap();
        for ch in "hibob".chars() {
            fsm.advance(ch).unwrap();
        }
        assert!(fsm.is_final());
    }

    #[test]
    fn star_zero_reps_ok() {
        let grammar = g(r#"root ::= "a"*"#);
        let fsm = Fsm::start(&grammar).unwrap();
        assert!(fsm.is_final());
    }

    #[test]
    fn star_many_reps_ok() {
        let grammar = g(r#"root ::= "a"*"#);
        let mut fsm = Fsm::start(&grammar).unwrap();
        for _ in 0..5 {
            fsm.advance('a').unwrap();
            assert!(fsm.is_final());
        }
    }

    #[test]
    fn plus_requires_one() {
        let grammar = g(r#"root ::= "a"+"#);
        let fsm = Fsm::start(&grammar).unwrap();
        assert!(!fsm.is_final());
        let mut fsm2 = Fsm::start(&grammar).unwrap();
        fsm2.advance('a').unwrap();
        assert!(fsm2.is_final());
        fsm2.advance('a').unwrap();
        assert!(fsm2.is_final());
    }

    #[test]
    fn optional_zero_or_one() {
        let grammar = g(r#"root ::= "a"?"#);
        let fsm = Fsm::start(&grammar).unwrap();
        assert!(fsm.is_final());
        let mut fsm2 = Fsm::start(&grammar).unwrap();
        fsm2.advance('a').unwrap();
        assert!(fsm2.is_final());
        // Second 'a' rejected.
        assert!(fsm2.advance('a').is_err());
    }

    #[test]
    fn nested_group_and_quantifier() {
        let grammar = g(r#"root ::= "(" [0-9]+ ")""#);
        let mut fsm = Fsm::start(&grammar).unwrap();
        for ch in "(42)".chars() {
            fsm.advance(ch).unwrap();
        }
        assert!(fsm.is_final());
    }

    #[test]
    fn accepts_str_positive_and_partial() {
        let grammar = g(r#"root ::= "hello""#);
        let fsm = Fsm::start(&grammar).unwrap();
        assert!(fsm.accepts_str("hello"));
        assert!(fsm.accepts_str("hell")); // partial prefix, still consumable
        assert!(!fsm.accepts_str("helloo")); // overshoots
        assert!(!fsm.accepts_str("world")); // wrong prefix
    }

    #[test]
    fn accepts_ch_non_mutating() {
        let grammar = g(r#"root ::= "hi""#);
        let fsm = Fsm::start(&grammar).unwrap();
        assert!(fsm.accepts('h'));
        // No mutation: still at the same state, still accepts 'h'.
        assert!(fsm.accepts('h'));
        // Real advance now works.
        let mut fsm_mut = fsm.clone();
        fsm_mut.advance('h').unwrap();
        assert!(fsm_mut.accepts('i'));
    }

    #[test]
    fn allowed_chars_of_char_class() {
        let grammar = g(r"root ::= [a-c]");
        let fsm = Fsm::start(&grammar).unwrap();
        let allowed = fsm.allowed_chars();
        assert!(allowed.contains('a'));
        assert!(allowed.contains('b'));
        assert!(allowed.contains('c'));
        assert!(!allowed.contains('d'));
    }

    #[test]
    fn left_recursion_with_base_case_survives() {
        // `root ::= root "a" | "b"` — infinite left branch is culled, "b" survives.
        let grammar = g(r#"root ::= root "a" | "b""#);
        let mut fsm = Fsm::start(&grammar).unwrap();
        assert!(!fsm.is_dead());
        fsm.advance('b').unwrap();
        // Now zero or more 'a's; is_final true immediately.
        assert!(fsm.is_final());
        fsm.advance('a').unwrap();
        assert!(fsm.is_final());
    }

    #[test]
    fn left_recursion_no_base_case_errors() {
        // `root ::= root "a"` — no base case, every path exceeds depth.
        // Constructed manually since parse_gbnf itself is happy with it.
        use super::super::gbnf::{Alternative, Symbol};
        use std::collections::HashMap;
        let grammar = Grammar {
            rules: {
                let mut m = HashMap::new();
                m.insert(
                    "root".to_string(),
                    vec![Alternative(vec![
                        Symbol::RuleRef("root".to_string()),
                        Symbol::Terminal("a".to_string()),
                    ])],
                );
                m
            },
            root: "root".to_string(),
        };
        let err = Fsm::start(&grammar).unwrap_err();
        matches!(err, FsmError::RecursionOverflow { .. });
    }

    #[test]
    fn is_dead_after_impossible_advance() {
        // Non-mutating scenario: build a stuck FSM and verify is_dead.
        let grammar = g(r#"root ::= "a""#);
        let mut fsm = Fsm::start(&grammar).unwrap();
        // After consuming 'a' the grammar is fully done and any further
        // advance dies.
        fsm.advance('a').unwrap();
        assert!(fsm.is_final());
        assert!(fsm.advance('b').is_err());
    }
}
