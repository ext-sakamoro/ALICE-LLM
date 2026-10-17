//! ALICE-Law G13: `LookupMap` / `LookupSet` (src/lookup.rs) expose no
//! iteration order
//!
//! src/lookup.rs is the one module allowed to name the std hash collections
//! (clippy.toml bans them elsewhere). This reads it and fails when it gains a
//! method outside the keyed-access list, an iterating trait (IntoIterator,
//! Extend, Deref, AsRef, Borrow of the inner map, Serialize), a field visible
//! outside the module, or a Debug that prints the entries.

const SRC: &str = include_str!("../src/lookup.rs");

/// Methods that look up, insert or remove by key, or count, and the trait
/// methods of the allowed impls (Default, PartialEq, FromIterator, Debug)
const ALLOWED_FNS: &[&str] = &[
    "new",
    "with_capacity",
    "get",
    "contains",
    "insert",
    "remove",
    "entry",
    "len",
    "is_empty",
    "default",
    "eq",
    "from_iter",
    "fmt",
];

/// Traits a wrapper may implement: none of them hands out the entries
const ALLOWED_IMPLS: &[&str] = &["PartialEq", "Eq", "Default", "FromIterator", "fmt::Debug"];

/// Source with `//` comments removed (doc comments mention the forbidden names)
fn code() -> String {
    SRC.lines()
        .map(|l| l.find("//").map_or(l, |i| &l[..i]))
        .collect::<Vec<_>>()
        .join("\n")
}

fn words_after(code: &str, keyword: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = code;
    while let Some(i) = rest.find(keyword) {
        let before_ok = i == 0
            || !rest.as_bytes()[i - 1].is_ascii_alphanumeric() && rest.as_bytes()[i - 1] != b'_';
        rest = &rest[i + keyword.len()..];
        if !before_ok {
            continue;
        }
        let name: String = rest
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == ':')
            .collect();
        out.push(name);
    }
    out
}

#[test]
fn the_wrappers_have_only_keyed_access() {
    let code = code();
    let fns = words_after(&code, "fn ");
    assert!(
        fns.len() >= ALLOWED_FNS.len(),
        "read {} fn names: compared too little",
        fns.len()
    );
    let extra: Vec<&String> = fns
        .iter()
        .filter(|f| !ALLOWED_FNS.contains(&f.as_str()))
        .collect();
    assert!(
        extra.is_empty(),
        "src/lookup.rs gained methods outside keyed access: {extra:?}"
    );

    // `impl<..> Trait for LookupX` or `impl<..> LookupX` (inherent)
    let mut impls = 0;
    for line in code
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("impl"))
    {
        impls += 1;
        let head = line.split('{').next().unwrap_or(line);
        let after_generics = match head.find('>') {
            Some(i) if head.starts_with("impl<") => &head[i + 1..],
            _ => &head[4..],
        };
        let after_generics = after_generics.trim();
        if let Some((tr, _)) = after_generics.split_once(" for ") {
            let tr = tr.split('<').next().unwrap_or(tr).trim();
            assert!(
                ALLOWED_IMPLS.contains(&tr),
                "src/lookup.rs implements `{tr}`: {line}"
            );
        } else {
            assert!(
                after_generics.starts_with("LookupMap") || after_generics.starts_with("LookupSet"),
                "unexpected impl: {line}"
            );
        }
    }
    assert!(impls >= 10, "read {impls} impl blocks: compared too little");

    let derives: Vec<&str> = code
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("#[derive"))
        .collect();
    assert_eq!(
        derives,
        ["#[derive(Clone)]", "#[derive(Clone)]"],
        "derives on the wrappers changed"
    );

    for (name, inner) in [
        ("LookupMap<K, V>", "HashMap<K, V>"),
        ("LookupSet<T>", "HashSet<T>"),
    ] {
        let decl = format!("pub(crate) struct {name}({inner});");
        assert!(
            code.contains(&decl),
            "the inner collection of {name} must stay private: `{decl}` not found"
        );
    }
    // Debug: the impl heads carry no `Debug` bound on the keys or values, so
    // no entry can be formatted, and the body reads the inner collection only
    // for its length
    for head in [
        "impl<K, V> fmt::Debug for LookupMap<K, V> {",
        "impl<T> fmt::Debug for LookupSet<T> {",
    ] {
        assert!(
            code.contains(head),
            "Debug impl head changed: `{head}` not found"
        );
    }
    let mut fmt_bodies = 0;
    for (i, _) in code.match_indices("fn fmt(") {
        let body = &code[i..];
        let end = body.find("\n    }").expect("end of fmt");
        let body = &body[..end];
        fmt_bodies += 1;
        let uses = body.matches("self.0").count();
        let len_uses = body.matches("self.0.len()").count();
        assert_eq!(
            uses, len_uses,
            "a Debug body reads the inner collection beyond its length: {body}"
        );
    }
    assert_eq!(fmt_bodies, 2, "expected the two Debug bodies");
    for forbidden in [
        ".iter(",
        ".keys(",
        ".values(",
        ".drain(",
        ".into_iter()",
        ".retain(",
        "debug_map",
        "debug_set",
    ] {
        assert!(
            !code.contains(forbidden),
            "src/lookup.rs uses `{forbidden}`"
        );
    }
}
