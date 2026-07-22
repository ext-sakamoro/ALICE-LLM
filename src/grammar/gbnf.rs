//! GBNF (llama.cpp-compatible) grammar parser — subset used for constrained
//! decoding under Phase X.8 Guided Generation.
//!
//! # Supported syntax
//!
//! ```text
//! # comments run to end of line
//! root     ::= greeting name
//! greeting ::= "hi" | "hello"
//! name     ::= [A-Za-z_] [A-Za-z0-9_]*
//! ```
//!
//! - Rule definitions: `name ::= alt1 | alt2 ...`
//! - Terminal literals: `"..."` with escapes `\n \t \r \" \\ \xNN`
//! - Character classes: `[a-z0-9]`, `[^abc]` (negation, ranges, escapes)
//! - Rule references: identifier
//! - Grouping: `(...)`
//! - Quantifiers: `*` (0+), `+` (1+), `?` (0-1)
//! - Comments: `# ...` to end of line
//! - A rule named `root` is required and used as the start symbol.
//!
//! # Not supported (v1)
//!
//! - `.` (any char) — use explicit `[^\n]`.
//! - `{n,m}` repeat — express with `*`/`+`/`?`.
//! - Lookahead / lookbehind.

use std::collections::HashMap;
use std::fmt;

/// A character class such as `[a-z]` or `[^abc]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CharClass {
    pub negated: bool,
    pub ranges: Vec<(char, char)>,
}

impl CharClass {
    /// Test membership.
    pub fn matches(&self, ch: char) -> bool {
        let hit = self.ranges.iter().any(|&(a, b)| ch >= a && ch <= b);
        hit ^ self.negated
    }
}

/// One grammar symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Symbol {
    Terminal(String),
    CharClass(CharClass),
    RuleRef(String),
    Group(Vec<Alternative>),
    Star(Box<Symbol>),
    Plus(Box<Symbol>),
    Optional(Box<Symbol>),
}

/// A sequence of symbols. A rule body is a list of alternatives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Alternative(pub Vec<Symbol>);

/// A parsed grammar. The rule named `root` is the entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grammar {
    pub rules: HashMap<String, Vec<Alternative>>,
    pub root: String,
}

impl Grammar {
    /// Alternatives of the root rule.
    pub fn root_alternatives(&self) -> &[Alternative] {
        self.rules.get(&self.root).map_or(&[], Vec::as_slice)
    }

    /// Look up a rule by name.
    pub fn rule(&self, name: &str) -> Option<&[Alternative]> {
        self.rules.get(name).map(Vec::as_slice)
    }
}

/// Errors that can occur while parsing a GBNF source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GbnfError {
    UnexpectedChar {
        pos: usize,
        ch: char,
        ctx: &'static str,
    },
    UnexpectedEof {
        ctx: &'static str,
    },
    UnknownEscape {
        pos: usize,
        ch: char,
    },
    EmptyGrammar,
    MissingRoot,
    DuplicateRule {
        name: String,
    },
    UndefinedRule {
        name: String,
    },
    UnsupportedSyntax {
        pos: usize,
        feature: &'static str,
    },
    InvalidCharRange {
        pos: usize,
        start: char,
        end: char,
    },
    EmptyCharClass {
        pos: usize,
    },
    EmptyAlternative {
        pos: usize,
    },
    ExpectedIdentifier {
        pos: usize,
    },
    ExpectedAssign {
        pos: usize,
    },
}

impl fmt::Display for GbnfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedChar { pos, ch, ctx } => {
                write!(f, "unexpected char {ch:?} at byte {pos} ({ctx})")
            }
            Self::UnexpectedEof { ctx } => write!(f, "unexpected end of input ({ctx})"),
            Self::UnknownEscape { pos, ch } => {
                write!(f, "unknown escape \\{ch} at byte {pos}")
            }
            Self::EmptyGrammar => write!(f, "grammar contains no rules"),
            Self::MissingRoot => write!(f, "grammar is missing a `root` rule"),
            Self::DuplicateRule { name } => write!(f, "rule `{name}` is defined more than once"),
            Self::UndefinedRule { name } => {
                write!(f, "rule `{name}` is referenced but not defined")
            }
            Self::UnsupportedSyntax { pos, feature } => {
                write!(f, "unsupported syntax at byte {pos}: {feature}")
            }
            Self::InvalidCharRange { pos, start, end } => {
                write!(f, "invalid char range {start:?}-{end:?} at byte {pos}")
            }
            Self::EmptyCharClass { pos } => write!(f, "empty character class at byte {pos}"),
            Self::EmptyAlternative { pos } => write!(f, "empty alternative at byte {pos}"),
            Self::ExpectedIdentifier { pos } => write!(f, "expected identifier at byte {pos}"),
            Self::ExpectedAssign { pos } => write!(f, "expected `::=` at byte {pos}"),
        }
    }
}

impl std::error::Error for GbnfError {}

/// Parse a GBNF source string into a validated [`Grammar`].
///
/// All rule references are checked to exist as rule definitions.
pub fn parse_gbnf(src: &str) -> Result<Grammar, GbnfError> {
    let mut p = Parser::new(src);
    let g = p.parse_grammar()?;
    for alts in g.rules.values() {
        for alt in alts {
            for sym in &alt.0 {
                validate_symbol(sym, &g.rules)?;
            }
        }
    }
    Ok(g)
}

fn validate_symbol(
    sym: &Symbol,
    rules: &HashMap<String, Vec<Alternative>>,
) -> Result<(), GbnfError> {
    match sym {
        Symbol::RuleRef(name) => {
            if !rules.contains_key(name) {
                return Err(GbnfError::UndefinedRule { name: name.clone() });
            }
        }
        Symbol::Group(alts) => {
            for alt in alts {
                for s in &alt.0 {
                    validate_symbol(s, rules)?;
                }
            }
        }
        Symbol::Star(s) | Symbol::Plus(s) | Symbol::Optional(s) => {
            validate_symbol(s, rules)?;
        }
        Symbol::Terminal(_) | Symbol::CharClass(_) => {}
    }
    Ok(())
}

struct Parser<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn peek(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn peek2(&self) -> Option<char> {
        let mut it = self.src[self.pos..].chars();
        it.next();
        it.next()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        Some(c)
    }

    fn consume_char(&mut self, expected: char) -> Result<(), GbnfError> {
        match self.peek() {
            Some(c) if c == expected => {
                self.advance();
                Ok(())
            }
            Some(c) => Err(GbnfError::UnexpectedChar {
                pos: self.pos,
                ch: c,
                ctx: "consume_char",
            }),
            None => Err(GbnfError::UnexpectedEof {
                ctx: "consume_char",
            }),
        }
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            match self.peek() {
                Some(c) if c.is_whitespace() => {
                    self.advance();
                }
                Some('#') => {
                    while let Some(c) = self.advance() {
                        if c == '\n' {
                            break;
                        }
                    }
                }
                _ => break,
            }
        }
    }

    /// Returns `true` if the parser is positioned at the start of a new rule
    /// definition (`ident (ws)* ::=`). Used to detect rule boundaries.
    fn at_rule_start(&self) -> bool {
        let mut probe = self.pos;
        let bytes = self.src.as_bytes();
        // Identifier must start with alpha or `_`.
        let first = self.src[probe..].chars().next();
        match first {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
            _ => return false,
        }
        // Consume identifier characters.
        while probe < bytes.len() {
            let Some(c) = self.src[probe..].chars().next() else {
                break;
            };
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                probe += c.len_utf8();
            } else {
                break;
            }
        }
        // Skip inline whitespace (spaces / tabs).
        while probe < bytes.len() && (bytes[probe] == b' ' || bytes[probe] == b'\t') {
            probe += 1;
        }
        // Look for `::=`.
        let rest = &self.src[probe..];
        rest.starts_with("::=")
    }

    fn parse_identifier(&mut self) -> Result<String, GbnfError> {
        let start = self.pos;
        match self.peek() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
            _ => return Err(GbnfError::ExpectedIdentifier { pos: start }),
        }
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }
        Ok(s)
    }

    fn parse_grammar(&mut self) -> Result<Grammar, GbnfError> {
        let mut rules: HashMap<String, Vec<Alternative>> = HashMap::new();
        loop {
            self.skip_ws_and_comments();
            if self.peek().is_none() {
                break;
            }
            let (name, alts) = self.parse_rule()?;
            if rules.contains_key(&name) {
                return Err(GbnfError::DuplicateRule { name });
            }
            rules.insert(name, alts);
        }
        if rules.is_empty() {
            return Err(GbnfError::EmptyGrammar);
        }
        if !rules.contains_key("root") {
            return Err(GbnfError::MissingRoot);
        }
        Ok(Grammar {
            rules,
            root: "root".to_string(),
        })
    }

    fn parse_rule(&mut self) -> Result<(String, Vec<Alternative>), GbnfError> {
        let name = self.parse_identifier()?;
        // Skip inline whitespace only; leave newlines untouched so `at_rule_start`
        // logic upstream stays well-defined.
        while let Some(c) = self.peek() {
            if c == ' ' || c == '\t' {
                self.advance();
            } else {
                break;
            }
        }
        let assign_pos = self.pos;
        if !self.src[assign_pos..].starts_with("::=") {
            return Err(GbnfError::ExpectedAssign { pos: assign_pos });
        }
        self.pos += 3;
        let alts = self.parse_alternatives(true)?;
        Ok((name, alts))
    }

    fn parse_alternatives(&mut self, top_level: bool) -> Result<Vec<Alternative>, GbnfError> {
        let mut alts = vec![self.parse_alternative(top_level)?];
        loop {
            self.skip_ws_and_comments();
            if self.peek() == Some('|') {
                self.advance();
                alts.push(self.parse_alternative(top_level)?);
            } else {
                break;
            }
        }
        Ok(alts)
    }

    fn parse_alternative(&mut self, top_level: bool) -> Result<Alternative, GbnfError> {
        let start_pos = self.pos;
        let mut syms = Vec::new();
        loop {
            self.skip_ws_and_comments();
            match self.peek() {
                None | Some('|' | ')') => break,
                _ => {}
            }
            // Inside a group, `::=` cannot legally appear, so we only apply the
            // rule-start heuristic at top level.
            if top_level && self.at_rule_start() {
                break;
            }
            syms.push(self.parse_symbol_with_quantifier()?);
        }
        if syms.is_empty() {
            return Err(GbnfError::EmptyAlternative { pos: start_pos });
        }
        Ok(Alternative(syms))
    }

    fn parse_symbol_with_quantifier(&mut self) -> Result<Symbol, GbnfError> {
        let base = self.parse_symbol()?;
        let symbol = match self.peek() {
            Some('*') => {
                self.advance();
                Symbol::Star(Box::new(base))
            }
            Some('+') => {
                self.advance();
                Symbol::Plus(Box::new(base))
            }
            Some('?') => {
                self.advance();
                Symbol::Optional(Box::new(base))
            }
            _ => base,
        };
        Ok(symbol)
    }

    fn parse_symbol(&mut self) -> Result<Symbol, GbnfError> {
        match self.peek() {
            Some('"') => Ok(Symbol::Terminal(self.parse_terminal()?)),
            Some('[') => Ok(Symbol::CharClass(self.parse_char_class()?)),
            Some('(') => {
                self.advance();
                let alts = self.parse_alternatives(false)?;
                self.skip_ws_and_comments();
                self.consume_char(')')?;
                Ok(Symbol::Group(alts))
            }
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                let name = self.parse_identifier()?;
                Ok(Symbol::RuleRef(name))
            }
            Some('.') => Err(GbnfError::UnsupportedSyntax {
                pos: self.pos,
                feature: "any-char '.'",
            }),
            Some('{') => Err(GbnfError::UnsupportedSyntax {
                pos: self.pos,
                feature: "repeat {n,m}",
            }),
            Some(c) => Err(GbnfError::UnexpectedChar {
                pos: self.pos,
                ch: c,
                ctx: "parse_symbol",
            }),
            None => Err(GbnfError::UnexpectedEof {
                ctx: "parse_symbol",
            }),
        }
    }

    fn parse_terminal(&mut self) -> Result<String, GbnfError> {
        self.consume_char('"')?;
        let mut s = String::new();
        loop {
            let esc_pos = self.pos;
            match self.advance() {
                Some('"') => return Ok(s),
                Some('\\') => {
                    let ch = self.parse_escape(esc_pos, /*in_class=*/ false)?;
                    s.push(ch);
                }
                Some(c) => s.push(c),
                None => return Err(GbnfError::UnexpectedEof { ctx: "terminal" }),
            }
        }
    }

    fn parse_char_class(&mut self) -> Result<CharClass, GbnfError> {
        let start = self.pos;
        self.consume_char('[')?;
        let negated = self.peek() == Some('^');
        if negated {
            self.advance();
        }
        let mut ranges = Vec::new();
        loop {
            match self.peek() {
                Some(']') => {
                    self.advance();
                    break;
                }
                None => {
                    return Err(GbnfError::UnexpectedEof { ctx: "char class" });
                }
                _ => {}
            }
            let range_pos = self.pos;
            let lo = self.parse_class_char()?;
            let hi = if self.peek() == Some('-') && self.peek2() != Some(']') {
                self.advance();
                self.parse_class_char()?
            } else {
                lo
            };
            if lo > hi {
                return Err(GbnfError::InvalidCharRange {
                    pos: range_pos,
                    start: lo,
                    end: hi,
                });
            }
            ranges.push((lo, hi));
        }
        if ranges.is_empty() {
            return Err(GbnfError::EmptyCharClass { pos: start });
        }
        Ok(CharClass { negated, ranges })
    }

    fn parse_class_char(&mut self) -> Result<char, GbnfError> {
        let esc_pos = self.pos;
        match self.advance() {
            Some('\\') => self.parse_escape(esc_pos, /*in_class=*/ true),
            Some(c) => Ok(c),
            None => Err(GbnfError::UnexpectedEof { ctx: "class char" }),
        }
    }

    fn parse_escape(&mut self, esc_pos: usize, in_class: bool) -> Result<char, GbnfError> {
        match self.advance() {
            Some('n') => Ok('\n'),
            Some('t') => Ok('\t'),
            Some('r') => Ok('\r'),
            Some('"') => Ok('"'),
            Some('\\') => Ok('\\'),
            Some(']') if in_class => Ok(']'),
            Some('[') if in_class => Ok('['),
            Some('-') if in_class => Ok('-'),
            Some('^') if in_class => Ok('^'),
            Some('x') => {
                let h1 = self
                    .advance()
                    .ok_or(GbnfError::UnexpectedEof { ctx: "hex escape" })?;
                let h2 = self
                    .advance()
                    .ok_or(GbnfError::UnexpectedEof { ctx: "hex escape" })?;
                let hex: String = [h1, h2].iter().collect();
                let byte = u8::from_str_radix(&hex, 16).map_err(|_| GbnfError::UnknownEscape {
                    pos: esc_pos,
                    ch: 'x',
                })?;
                Ok(byte as char)
            }
            Some(c) => Err(GbnfError::UnknownEscape {
                pos: esc_pos,
                ch: c,
            }),
            None => Err(GbnfError::UnexpectedEof {
                ctx: "escape sequence",
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_terminal() {
        let g = parse_gbnf("root ::= \"hello\"").unwrap();
        assert_eq!(g.root, "root");
        assert_eq!(
            g.rules["root"],
            vec![Alternative(vec![Symbol::Terminal("hello".to_string())])],
        );
    }

    #[test]
    fn parses_alternatives() {
        let g = parse_gbnf("root ::= \"a\" | \"b\"").unwrap();
        assert_eq!(g.rules["root"].len(), 2);
    }

    #[test]
    fn parses_char_class_basic() {
        let g = parse_gbnf("root ::= [0-9]").unwrap();
        match &g.rules["root"][0].0[0] {
            Symbol::CharClass(cc) => {
                assert!(!cc.negated);
                assert_eq!(cc.ranges, vec![('0', '9')]);
                assert!(cc.matches('5'));
                assert!(!cc.matches('a'));
            }
            other => panic!("expected CharClass, got {:?}", other),
        }
    }

    #[test]
    fn parses_char_class_negated_multi_range() {
        let g = parse_gbnf("root ::= [^A-Za-z0-9]").unwrap();
        match &g.rules["root"][0].0[0] {
            Symbol::CharClass(cc) => {
                assert!(cc.negated);
                assert_eq!(cc.ranges.len(), 3);
                assert!(!cc.matches('a'));
                assert!(!cc.matches('Z'));
                assert!(!cc.matches('7'));
                assert!(cc.matches('!'));
            }
            other => panic!("expected CharClass, got {:?}", other),
        }
    }

    #[test]
    fn parses_quantifiers() {
        let g = parse_gbnf("root ::= \"a\"* \"b\"+ \"c\"?").unwrap();
        let syms = &g.rules["root"][0].0;
        assert!(matches!(syms[0], Symbol::Star(_)));
        assert!(matches!(syms[1], Symbol::Plus(_)));
        assert!(matches!(syms[2], Symbol::Optional(_)));
    }

    #[test]
    fn parses_multi_rule_grammar() {
        let src = r#"
            root     ::= greeting name
            greeting ::= "hi" | "hello"
            name     ::= [A-Za-z_] [A-Za-z0-9_]*
        "#;
        let g = parse_gbnf(src).unwrap();
        assert_eq!(g.root, "root");
        assert_eq!(g.rules.len(), 3);
        assert_eq!(
            g.rules["root"][0].0[0],
            Symbol::RuleRef("greeting".to_string())
        );
    }

    #[test]
    fn parses_group_with_quantifier() {
        let g = parse_gbnf("root ::= (\"a\" | \"b\")+").unwrap();
        match &g.rules["root"][0].0[0] {
            Symbol::Plus(inner) => match inner.as_ref() {
                Symbol::Group(alts) => assert_eq!(alts.len(), 2),
                other => panic!("expected Group inside Plus, got {:?}", other),
            },
            other => panic!("expected Plus, got {:?}", other),
        }
    }

    #[test]
    fn skips_line_comments() {
        let src = "# top-level\nroot ::= \"a\"  # trailing";
        let g = parse_gbnf(src).unwrap();
        assert_eq!(g.rules["root"][0].0[0], Symbol::Terminal("a".to_string()));
    }

    #[test]
    fn requires_root_rule() {
        assert_eq!(parse_gbnf("start ::= \"a\""), Err(GbnfError::MissingRoot),);
    }

    #[test]
    fn rejects_duplicate_rule() {
        let src = "root ::= \"a\"\nroot ::= \"b\"";
        assert!(matches!(
            parse_gbnf(src),
            Err(GbnfError::DuplicateRule { .. }),
        ));
    }

    #[test]
    fn rejects_undefined_rule_reference() {
        assert!(matches!(
            parse_gbnf("root ::= foo"),
            Err(GbnfError::UndefinedRule { .. }),
        ));
    }

    #[test]
    fn parses_escape_sequences_in_terminal() {
        let g = parse_gbnf(r#"root ::= "a\nb\tc\"d\\e""#).unwrap();
        assert_eq!(
            g.rules["root"][0].0[0],
            Symbol::Terminal("a\nb\tc\"d\\e".to_string()),
        );
    }

    #[test]
    fn parses_hex_escape() {
        let g = parse_gbnf(r#"root ::= "\x41""#).unwrap();
        assert_eq!(g.rules["root"][0].0[0], Symbol::Terminal("A".to_string()),);
    }

    #[test]
    fn rejects_dot_wildcard() {
        assert!(matches!(
            parse_gbnf("root ::= ."),
            Err(GbnfError::UnsupportedSyntax {
                feature: "any-char '.'",
                ..
            }),
        ));
    }

    #[test]
    fn rejects_brace_quantifier() {
        assert!(matches!(
            parse_gbnf("root ::= \"a\"{2,3}"),
            Err(GbnfError::UnsupportedSyntax {
                feature: "repeat {n,m}",
                ..
            }),
        ));
    }

    #[test]
    fn rejects_empty_alternative() {
        assert!(matches!(
            parse_gbnf("root ::= \"a\" |"),
            Err(GbnfError::EmptyAlternative { .. }),
        ));
    }

    /// Golden test: a small LOL-flavoured grammar covering the pieces the
    /// B-5 handwritten `lol.gbnf` will need — sequence, alternatives, groups,
    /// quantifiers, character classes and rule references all interacting.
    #[test]
    fn parses_lol_flavoured_grammar() {
        let src = r#"
            # LOL DSL mini subset for tests
            root       ::= expr
            expr       ::= sphere | box3d | union-expr
            sphere     ::= "sphere(" number ")"
            box3d      ::= "box3d(" number "," ws number "," ws number ")"
            union-expr ::= "union(" expr "," ws expr ")"
            number     ::= "-"? [0-9]+ ("." [0-9]+)?
            ws         ::= " "*
        "#;
        let g = parse_gbnf(src).unwrap();
        assert_eq!(g.root, "root");
        for name in ["expr", "sphere", "box3d", "union-expr", "number", "ws"] {
            assert!(
                g.rules.contains_key(name),
                "expected rule `{}` to be present",
                name,
            );
        }
        // `number` should start with an optional "-"
        let number_alt = &g.rules["number"][0].0;
        assert!(matches!(&number_alt[0], Symbol::Optional(_)));
    }
}
