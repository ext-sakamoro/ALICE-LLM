//! Forbid negating the whole result of a `mul_add` call (`-(x.mul_add(y, z))`,
//! and its unparenthesised form `-x.mul_add(y, z)` — in Rust method calls bind
//! tighter than unary minus, so these are the *same* AST: `ExprUnary{Neg}`
//! wrapping an `ExprMethodCall` named `mul_add` once parens/groups are peeled).
//!
//! Replaces an earlier line-oriented regex scanner that only matched the
//! parenthesised spelling and silently passed `-a.mul_add(b, c)`,
//! `-(a).mul_add(b, c)`, `-x.y.mul_add(b, c)` and `-v[0].mul_add(b, c)` — all
//! four are the identical miscompile-prone form (see the module doc of the
//! Law fix this gate exists for: cancellation input makes `-C opt-level=0`
//! and `>=1` disagree on the sign of zero, because `fneg(fma(x,y,z)) ->
//! fma(x,-y,-z)` is an `nsz`-only-valid LLVM canonicalisation that fires
//! anyway). This tool parses real Rust syntax (`syn`) instead of guessing at
//! paren/bracket balance, so it cannot be fooled by the receiver's shape:
//! identifier, `.field`, `[index]`, a parenthesised sub-expression, or any
//! chain of those.
//!
//! The negation itself is also checked in every syntactically-reachable
//! spelling, not just `-expr`: all of these negate the whole `mul_add`
//! result exactly as `-expr` would, and the generated code is identical
//! since Rust lowers each to the same negate-the-FMA-result operation —
//!
//!   * `a.mul_add(b, c).neg()` — the `Neg` trait's method form
//!   * `-{ a.mul_add(b, c) }` — unary minus on a block whose only statement
//!     is the call (no semicolon); `peel()` sees through single-tail-expr
//!     blocks the same as it sees through parens
//!   * `-0.0 - a.mul_add(b, c)` — subtracting from the literal `-0.0` is the
//!     textbook substitute for negation on a target with no native negate,
//!     and is bit-for-bit equivalent to `-x` for every `x` including both
//!     signed zeros (IEEE 754 round-to-nearest); `0.0 - x` (positive zero)
//!     is a *different*, incidentally safer operation and is not flagged
//!   * `a.mul_add(b, c) * -1.0` (either operand order) — multiplying by the
//!     literal `-1.0` is the same substitute, also bit-exact for all `x`
//!   * `let t = a.mul_add(b, c); /* ... */ -t;` — binding the call to a
//!     variable and negating the variable later, in any of the four forms
//!     above and in **any syntactically-reachable position** within the
//!     enclosing block — a `return` expression, a function-call argument, an
//!     array/struct-literal field, an index-assignment right-hand side, an
//!     `if`/`match`/loop branch, or buried inside a larger unrelated
//!     expression (`(-t) * 0.5` still negates `t` as a subexpression, even
//!     though the statement as a whole computes something else); this is a
//!     best-effort textual-scope check, not full dataflow — it recurses into
//!     every nested block reachable from the binding (so shadowing is
//!     tracked independently per nested scope: a `let`/reassignment of the
//!     same name inside an `if`-branch only shadows within that branch, not
//!     in sibling branches or after the `if` ends) but does not otherwise
//!     prove the binding was never mutated through a reference or alias, so
//!     prefer writing the direct forms where possible
//!
//! ⚠️ the miscompile this gate exists for only reproduces on a target with a
//! native FMA instruction (`mul_add` lowers to one `fma`/`fmadd` op the
//! `nsz`-unsound canonicalisation can rewrite); the default `x86_64-apple-
//! darwin` / `x86_64-unknown-linux-gnu` targets have no FMA enabled by
//! default, so `mul_add` there lowers to a plain libm `fmaf`/`fma` call
//! instead, and the split does not show up. The gate itself is still
//! correct to run everywhere (it is a syntactic check, not a runtime one),
//! but a CI leg meant to *demonstrate* the split (not just lint for the
//! form) needs an FMA-enabled target — `aarch64-apple-darwin` has FMA
//! unconditionally, which is why that release leg is the one that actually
//! shows debug/release disagreeing and must stay required.
//!
//! `mul_add` itself is not forbidden (one correctly-rounded operation); the
//! `f32::mul_add` / `f64::mul_add` **path**-call spelling (`f32::mul_add(a,
//! b, c)`, including fully-qualified `<f32>::mul_add(...)`) is covered the
//! same as the method-call spelling. Negating an *argument* before the call
//! (`a.mul_add(b, -c)`, FMSUB-equivalent) or a nested argument
//! (`a.mul_add(b, -(c * d))`) is the Law-adopted safe form and is not
//! flagged — only negation of the call's own result is.
//!
//! ## What this does **not** see: macro invocation bodies
//!
//! `syn` does not expand macros; a macro invocation (`foo!(...)`,
//! `assert_eq!(...)`, a custom proc-macro call, etc.) is parsed as an opaque
//! token stream, not as Rust expressions, so the AST walk above cannot tell
//! whether a `mul_add` mention inside one is negated, safely negated-operand,
//! or unrelated — unlike everywhere else this tool looks, it genuinely
//! cannot verify the site. Rather than leave that as a silent blind spot,
//! any macro invocation whose token stream contains the text `mul_add` is a
//! **hard failure by default**, exactly like an unreviewed violation: it
//! must be listed in the allowlist (`<file>:<line>` of the macro invocation,
//! with a reason) before it passes, and — the same rule as everywhere else
//! — a `src/`-scoped site has **no allowlist escape at all**: the only fix
//! there is to pull the expression out of the macro call so the AST walk can
//! verify it directly. The allowlist entry only ever means "a human looked
//! at this specific macro invocation and confirmed it is safe", never
//! "this form is fine in general".
//!
//! ⚠️ `src/` has no escape hatch: a `src/`-scoped finding with an allowlist
//! entry is still a hard failure — rewriting to the Law form is the only fix
//! there. The allowlist exists only for `tests/` / `examples/` oracle files
//! that deliberately keep the old form as a comparison reference.
//!
//! ## The fma family and the other spellings of a negation (ALICE-Law G3)
//!
//! The same check covers every fused multiply-add the crates call, not only
//! `mul_add`: a call (method or path, however imported) whose name is
//! `mul_add`, `fma`, `fmaf`, `fmaf32` or `fmaf64` (`libm`, `core::intrinsics`),
//! an aarch64 `vfma*` / `vfms*` intrinsic, or an x86 `_mm*_fmadd_*` /
//! `_fmsub_*` / `_fnmadd_*` / `_fnmsub_*` / `_fmaddsub_*` / `_fmsubadd_*`
//! intrinsic. `vfms*` and `_fnmadd_*` negate inside the single rounding (the
//! safe operand form) and are not findings themselves; negating their result is.
//! Besides the forms above, a negation is also `Neg::neg(x)` (any path ending in
//! `neg` with one argument), `x / -1.0`, `x *= -1.0` / `x /= -1.0` on a tracked
//! binding, the sign-bit flip `f32::from_bits(x.to_bits() ^ SIGN)` (or
//! `f64::...`, with `0x8000_0000`, `1 << 31`, `0x8000_0000_0000_0000` or
//! `1 << 63`), and the SIMD spellings `vneg*(x)`, `vsub*(vdup*_n_*(-0.0), x)`,
//! `vmul*_n_*(x, -1.0)`, `_mm*_xor_p[sd](x, _mm*_set1_p[sd](-0.0))`,
//! `_mm*_sub_p[sd](_mm*_set1_p[sd](-0.0), x)` and
//! `_mm*_mul_p[sd](x, _mm*_set1_p[sd](-1.0))` (either operand order where the
//! operation commutes; a vector of negative zeros may also be the sign-bit
//! integer mask reinterpreted, `_mm*_castsi*_p[sd](_mm*_set1_epi32(SIGN))` or
//! `vreinterpret*_f32_u32(vdup*_n_u32(SIGN))`). Each constant (`-0.0`, `-1.0`,
//! a sign mask) is matched as a literal or through a `const` / `static` / `let`
//! name that the file binds to that constant only (`const SIGN: u32 = 1 << 31;`);
//! a name the file also binds to anything else is not followed. A fma result is also seen through `as` casts,
//! `T::from(x)` / `x.into()`, `*&x`, a block's tail expression, and any arm of
//! an `if` / `match`. A binding is tracked through `let u = t;` aliases and a
//! deferred `let t; t = <fma>;`.
//!
//! Renaming a fma-family item on import (`use libm::fma as fused;`) and using a
//! fma-family path as a value (`let f = f32::mul_add;`, `.map(f32::mul_add)`)
//! are hard failures anywhere: the check reads names at call sites, so either
//! would hide the call (and leave it out of the counts). Import it under its own
//! name and call it directly.
//!
//! What stays out of reach of a syntax tree (a fma returned from a helper
//! function or a closure and negated by the caller, a fma stored in a tuple or
//! struct field and read back, generic code instantiated in another crate) is
//! what the IR half of the gate, `scripts/fneg_fma_ir_check.py`, reads in the
//! optimised LLVM IR. This tool is the fast pre-check that reports source lines.
//!
//! ## Counts and the allowlist
//!
//! `scripts/no-outer-fneg-of-fma-allowlist.toml` holds `expected-fma-calls`
//! (every fma-family call in the scanned files) and `expected-src-fma-calls`
//! (the library's: those in a `src/` directory outside a test-only item, i.e. an
//! item under `#[test]` or a `#[cfg(..)]` that holds only under `test` (`test`,
//! `all(.., test, ..)`, an `any` of such); `cfg(not(test))` and feature or target
//! predicates are library code). A statement with such an attribute inside a
//! library function is test-only too, and so is a whole file reached through a
//! test-only `mod y;` (`y.rs` / `y/mod.rs` / `#[path]`, under an inline module's
//! directory, or a non-root file's `<stem>/`) or declared from such a file.
//! A test-only item is compiled out of the library,
//! so its calls are not in the release IR the IR check reads; the output prints
//! both counts. Both must equal what the tool finds; a missing
//! header fails, and a repository without any fma call declares 0. Reading no
//! file fails whatever the header says. `[[allow]]` entries name a finding by
//! `path`, the exact trimmed `line` text and the `count` of findings with that
//! text, with a `reason`; only `tests/` and `examples/` files can carry one, and
//! an entry that matches nothing or a different count fails. `[[ir]]` entries are
//! read by the IR check only.
//!
//! Usage: `cargo run --release -- [--root DIR] [--src DIR [DIR...]]`
//! (default `--src`: the `src` / `tests` / `examples` / `benches` of every member
//! of the root `Cargo.toml`'s `[workspace]`, or of the root package when there is
//! no workspace)

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{
    BinOp, Block, Expr, ExprBinary, ExprCall, ExprMethodCall, ExprUnary, ItemUse, Lit, Macro, Pat,
    Stmt, UnOp, UseTree,
};

const ALLOWLIST_REL: &str = "scripts/no-outer-fneg-of-fma-allowlist.toml";
const ALLOWLIST_SCOPE_DIRS: [&str; 2] = ["tests", "examples"];
/// Directories scanned inside each workspace member.
const MEMBER_SUBDIRS: [&str; 4] = ["src", "tests", "examples", "benches"];

/// Fixed directory set used only by this tool's own unit tests (their
/// fixtures have no workspace `Cargo.toml`); the binary scans every
/// workspace member instead, see [`workspace_src_dirs`].
#[cfg(test)]
const DEFAULT_SRC_DIRS: [&str; 5] = [
    "alice-lol/src",
    "alice-lol/tests",
    "alice-lol/examples",
    "alice-lol-ui/src",
    "alice-lol-ui/examples",
];

/// `src` / `tests` / `examples` / `benches` of every member listed in the
/// root `Cargo.toml`'s `[workspace] members` (read from the file, not
/// hand-maintained, so a new member is scanned the moment it joins), or of the
/// root package itself when the manifest has `[package]` and no workspace. An
/// empty result means the manifest could not be read, has neither, or names a
/// member by a glob (not expanded); the caller fails on it.
fn workspace_src_dirs(root: &Path) -> Vec<String> {
    let Ok(text) = fs::read_to_string(root.join("Cargo.toml")) else {
        return Vec::new();
    };
    let Ok(manifest) = text.parse::<toml::Table>() else {
        return Vec::new();
    };
    let members: Vec<String> = match manifest.get("workspace").and_then(|w| w.get("members")) {
        Some(toml::Value::Array(list)) => list
            .iter()
            .filter_map(|m| m.as_str().map(str::to_string))
            .collect(),
        _ if manifest.contains_key("package") => vec![String::new()],
        _ => return Vec::new(),
    };
    if members.iter().any(|m| m.contains('*')) {
        return Vec::new();
    }
    let mut dirs = Vec::new();
    for member in members {
        for sub in MEMBER_SUBDIRS {
            dirs.push(if member.is_empty() {
                sub.to_string()
            } else {
                format!("{member}/{sub}")
            });
        }
    }
    dirs
}

enum FindingKind {
    /// `-x.mul_add(y, z)` (any receiver shape, any fma-family callee, any
    /// negation spelling) found directly by the AST walk.
    NegatedMulAdd,
    /// a macro invocation's token stream mentions a fma-family name —
    /// AST-unverified, hard fail unless allowlisted (never allowlistable in `src/`).
    MacroMulAdd,
    /// `use <fma-family path> as <other name>`, or a fma-family path used as a
    /// value (`let f = f32::mul_add;`): the check reads names at call sites, so
    /// either would hide the call. Never allowlistable.
    AliasedFma,
}

struct Finding {
    rel: String,
    line: usize,
    excerpt: String,
    kind: FindingKind,
}

#[derive(Default)]
struct FileCounts {
    unary_neg_exprs: usize,
    mul_add_calls: usize,
    scalar_fma_calls: usize,
    simd_fma_calls: usize,
    /// fma-family calls inside a test-only item (see [`is_test_only`])
    test_fma_calls: usize,
}

impl FileCounts {
    fn fma_calls(&self) -> usize {
        self.mul_add_calls + self.scalar_fma_calls + self.simd_fma_calls
    }
}

struct Visitor<'a> {
    rel: &'a str,
    path: &'a Path,
    src: &'a str,
    findings: Vec<Finding>,
    counts: FileCounts,
    /// how many enclosing items are test-only (`#[test]`, `#[cfg(test)]`, ...)
    test_depth: usize,
    /// the names of the enclosing inline `mod x { .. }`, for where a nested `mod y;` file lives
    mod_stack: Vec<String>,
    /// every `mod y;` declared in this file: its candidate files and whether it is declared test-only
    mod_decls: Vec<ModDecl>,
}

/// a `mod y;` whose body is another file: where that file can be, and whether the declaration is test-only
struct ModDecl {
    candidates: Vec<PathBuf>,
    test_only: bool,
}

/// the directory the `mod y;` declarations of the file at `path` resolve against: the file's own directory
/// for a crate root or a `mod.rs`, `<dir>/<stem>/` for any other file
fn module_dir(path: &Path) -> PathBuf {
    let dir = path.parent().unwrap_or(Path::new("")).to_path_buf();
    match path.file_name().and_then(|n| n.to_str()) {
        Some("lib.rs" | "main.rs" | "mod.rs") => dir,
        _ => dir.join(path.file_stem().and_then(|s| s.to_str()).unwrap_or("")),
    }
}

/// `#[path = "..."]` on a module declaration
fn path_attr(attrs: &[syn::Attribute]) -> Option<String> {
    attrs.iter().find_map(|a| match &a.meta {
        syn::Meta::NameValue(nv) if nv.path.is_ident("path") => match &nv.value {
            Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(s),
                ..
            }) => Some(s.value()),
            _ => None,
        },
        _ => None,
    })
}

/// the files whose every item is test-only: reached through a `mod y;` declared test-only, or declared from
/// such a file (a test-only parent compiles its children out too)
fn test_only_files(decls: &BTreeMap<PathBuf, Vec<ModDecl>>) -> BTreeSet<PathBuf> {
    let mut out = BTreeSet::new();
    loop {
        let before = out.len();
        for (parent, list) in decls {
            let parent_test = out.contains(parent);
            for d in list {
                if d.test_only || parent_test {
                    if let Some(c) = d.candidates.iter().find(|c| decls.contains_key(*c)) {
                        out.insert(c.clone());
                    }
                }
            }
        }
        if out.len() == before {
            return out;
        }
    }
}

/// `#[test]`, or a `#[cfg(..)]` whose predicate holds only under `cfg(test)`: `test`, `all(.., test, ..)`,
/// `any(..)` whose every branch is test-only. `not(..)` never is (`cfg(not(test))` is library code), nor is a
/// feature or target predicate. Such an item is compiled out of the library, so its fma calls are not in the
/// release IR and do not count towards `expected-src-fma-calls`.
fn is_test_only(attrs: &[syn::Attribute]) -> bool {
    fn test_only(meta: &syn::Meta) -> bool {
        match meta {
            syn::Meta::Path(p) => p.is_ident("test"),
            syn::Meta::List(l) => {
                let Ok(nested) = l.parse_args_with(
                    syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
                ) else {
                    return false;
                };
                if l.path.is_ident("all") {
                    nested.iter().any(test_only)
                } else if l.path.is_ident("any") {
                    !nested.is_empty() && nested.iter().all(test_only)
                } else {
                    false
                }
            }
            syn::Meta::NameValue(_) => false,
        }
    }
    attrs.iter().any(|a| {
        a.path().is_ident("test")
            || (a.path().is_ident("cfg")
                && a.parse_args::<syn::Meta>().is_ok_and(|m| test_only(&m)))
    })
}

/// the outer attributes of an expression statement (`#[cfg(test)] { .. }`, `#[cfg(test)] f();`)
fn expr_attrs(e: &Expr) -> &[syn::Attribute] {
    match e {
        Expr::Block(b) => &b.attrs,
        Expr::Call(c) => &c.attrs,
        Expr::MethodCall(m) => &m.attrs,
        Expr::Assign(a) => &a.attrs,
        Expr::Macro(m) => &m.attrs,
        Expr::If(i) => &i.attrs,
        Expr::Match(m) => &m.attrs,
        Expr::ForLoop(f) => &f.attrs,
        Expr::While(w) => &w.attrs,
        Expr::Loop(l) => &l.attrs,
        Expr::Unsafe(u) => &u.attrs,
        _ => &[],
    }
}

fn excerpt_at(src: &str, line: usize) -> String {
    src.lines()
        .nth(line.saturating_sub(1))
        .unwrap_or("")
        .trim()
        .to_string()
}

/// The kind of fma-family callee a name denotes, if any.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FmaKind {
    MulAdd,
    Scalar,
    Simd,
}

fn fma_kind(name: &str) -> Option<FmaKind> {
    const X86: [&str; 6] = [
        "_fmadd_",
        "_fmsub_",
        "_fnmadd_",
        "_fnmsub_",
        "_fmaddsub_",
        "_fmsubadd_",
    ];
    match name {
        "mul_add" => Some(FmaKind::MulAdd),
        "fma" | "fmaf" | "fmaf32" | "fmaf64" => Some(FmaKind::Scalar),
        _ if name.starts_with("vfma") || name.starts_with("vfms") => Some(FmaKind::Simd),
        _ if name.starts_with("_mm") && X86.iter().any(|k| name.contains(k)) => Some(FmaKind::Simd),
        _ => None,
    }
}

fn path_last(e: &Expr) -> Option<String> {
    match e {
        Expr::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()),
        _ => None,
    }
}

fn call_name(call: &ExprCall) -> Option<String> {
    path_last(&call.func)
}

fn method_fma_kind(call: &ExprMethodCall) -> Option<FmaKind> {
    fma_kind(&call.method.to_string())
}

fn path_call_fma_kind(call: &ExprCall) -> Option<FmaKind> {
    call_name(call).and_then(|n| fma_kind(&n))
}

/// Peel `(expr)`, `{ expr }`-less groups (syn's `Expr::Group`, inserted by
/// some macro-expansion contexts), and a block whose only statement is a
/// semicolon-less tail expression, down to the real inner expression, so
/// `-(a.mul_add(b, c))`, `-((a.mul_add(b, c)))` and `-{ a.mul_add(b, c) }`
/// are all treated identically to `-a.mul_add(b, c)`.
fn peel(mut e: &Expr) -> &Expr {
    loop {
        match e {
            Expr::Paren(p) => e = &p.expr,
            Expr::Group(g) => e = &g.expr,
            Expr::Block(b) if b.label.is_none() && b.block.stmts.len() == 1 => {
                match &b.block.stmts[0] {
                    Stmt::Expr(inner, None) => e = inner,
                    _ => return e,
                }
            }
            _ => return e,
        }
    }
}

/// The tail expression of a block (its last statement, without a semicolon).
fn block_tail(b: &Block) -> Option<&Expr> {
    match b.stmts.last() {
        Some(Stmt::Expr(e, None)) => Some(e),
        _ => None,
    }
}

/// `e` is the value of a fma-family call: directly, or through an `as` cast,
/// `T::from(x)` / `From::from(x)` / `x.into()`, `*&x`, a block's tail, or any
/// arm of an `if` / `match` (any arm: the negation then negates a fma result on
/// that path).
fn is_fma_result(e: &Expr) -> bool {
    match peel(e) {
        Expr::MethodCall(m) => {
            method_fma_kind(m).is_some()
                || (m.method == "into" && m.args.is_empty() && is_fma_result(&m.receiver))
        }
        Expr::Call(c) => {
            path_call_fma_kind(c).is_some()
                || (call_name(c).as_deref() == Some("from")
                    && c.args.len() == 1
                    && is_fma_result(&c.args[0]))
        }
        Expr::Cast(c) => is_fma_result(&c.expr),
        Expr::Reference(r) => is_fma_result(&r.expr),
        Expr::Unary(u) if matches!(u.op, UnOp::Deref(_)) => is_fma_result(&u.expr),
        Expr::Block(b) => block_tail(&b.block).is_some_and(is_fma_result),
        Expr::If(i) => {
            block_tail(&i.then_branch).is_some_and(is_fma_result)
                || i.else_branch
                    .as_ref()
                    .is_some_and(|(_, e)| is_fma_result(e))
        }
        Expr::Match(m) => m.arms.iter().any(|a| is_fma_result(&a.body)),
        _ => false,
    }
}

/// The `Neg` trait's method-call spelling: `x.neg()` means the same as `-x`.
fn is_neg_method_call(call: &ExprMethodCall) -> bool {
    call.method == "neg" && call.args.is_empty() && call.turbofish.is_none()
}

/// A float literal that parses to exactly `0.0` (matches `0.0`, `0.`, `0e0`, ...).
fn is_float_zero_literal(e: &Expr) -> bool {
    matches!(peel(e), Expr::Lit(l) if matches!(&l.lit, Lit::Float(f) if matches!(f.base10_parse::<f64>(), Ok(v) if v == 0.0)))
}

/// A float literal that parses to exactly `1.0`.
fn is_float_one_literal(e: &Expr) -> bool {
    matches!(peel(e), Expr::Lit(l) if matches!(&l.lit, Lit::Float(f) if matches!(f.base10_parse::<f64>(), Ok(v) if v == 1.0)))
}

/// The constants a negation is written with, as a literal or through a name.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum ConstKind {
    NegZero,
    NegOne,
    SignMask,
}

thread_local! {
    /// Names bound to one of the [`ConstKind`] constants in the file being
    /// checked (see [`named_constants`]).
    static NAMED: std::cell::RefCell<BTreeMap<String, ConstKind>> =
        const { std::cell::RefCell::new(BTreeMap::new()) };
}

fn named_kind(e: &Expr) -> Option<ConstKind> {
    let Expr::Path(p) = peel(e) else { return None };
    let name = p.path.segments.last()?.ident.to_string();
    NAMED.with(|n| n.borrow().get(&name).copied())
}

fn literal_kind(e: &Expr) -> Option<ConstKind> {
    match peel(e) {
        Expr::Unary(u) if matches!(u.op, UnOp::Neg(_)) && is_float_zero_literal(&u.expr) => {
            Some(ConstKind::NegZero)
        }
        Expr::Unary(u) if matches!(u.op, UnOp::Neg(_)) && is_float_one_literal(&u.expr) => {
            Some(ConstKind::NegOne)
        }
        e if is_sign_mask_literal(e) => Some(ConstKind::SignMask),
        _ => None,
    }
}

/// `const` / `static` items and `let` bindings of the file, by name, whose every
/// binding is the same [`ConstKind`] constant: `const SIGN: u32 = 1 << 31;` and
/// `let z = -0.0;` are then read as the literal they name. A name also bound to
/// anything else in the file is not followed (no scope analysis), so a reused
/// name cannot turn an unrelated value into a negation.
fn named_constants(file: &syn::File) -> BTreeMap<String, ConstKind> {
    struct Collect(BTreeMap<String, BTreeSet<Option<ConstKind>>>);
    impl<'ast> Visit<'ast> for Collect {
        fn visit_item_const(&mut self, node: &'ast syn::ItemConst) {
            self.0
                .entry(node.ident.to_string())
                .or_default()
                .insert(literal_kind(&node.expr));
            visit::visit_item_const(self, node);
        }
        fn visit_item_static(&mut self, node: &'ast syn::ItemStatic) {
            self.0
                .entry(node.ident.to_string())
                .or_default()
                .insert(literal_kind(&node.expr));
            visit::visit_item_static(self, node);
        }
        fn visit_local(&mut self, node: &'ast syn::Local) {
            let pat = match &node.pat {
                Pat::Type(t) => &*t.pat,
                p => p,
            };
            if let Pat::Ident(pi) = pat {
                let kind = node.init.as_ref().and_then(|i| literal_kind(&i.expr));
                self.0.entry(pi.ident.to_string()).or_default().insert(kind);
            }
            visit::visit_local(self, node);
        }
    }
    let mut c = Collect(BTreeMap::new());
    c.visit_file(file);
    c.0.into_iter()
        .filter_map(
            |(name, kinds)| match kinds.into_iter().collect::<Vec<_>>().as_slice() {
                [Some(k)] => Some((name, *k)),
                _ => None,
            },
        )
        .collect()
}

/// `-0.0` specifically (not `0.0`): `0.0 - x` is a different, incidentally
/// safer operation (it forces `+0` for a `+0` input instead of propagating
/// `x`'s own sign), so only the negative-zero spelling is treated as negation.
/// A name bound to `-0.0` counts too ([`named_constants`]).
fn is_negative_zero_literal(e: &Expr) -> bool {
    literal_kind(e) == Some(ConstKind::NegZero) || named_kind(e) == Some(ConstKind::NegZero)
}

/// `-1.0` specifically, as a literal or through a name.
fn is_negative_one_literal(e: &Expr) -> bool {
    literal_kind(e) == Some(ConstKind::NegOne) || named_kind(e) == Some(ConstKind::NegOne)
}

/// A SIMD splat of a literal: `vdup*_n_*(lit)` / `vmov*_n_*(lit)` /
/// `_mm*_set1_*(lit)` (float or integer lanes), true when `lit` satisfies `pred`.
fn is_simd_splat_of(e: &Expr, pred: &dyn Fn(&Expr) -> bool) -> bool {
    let Expr::Call(c) = peel(e) else { return false };
    let Some(name) = call_name(c) else {
        return false;
    };
    let splat = (name.starts_with("vdup") || name.starts_with("vmov")) && name.contains("_n_")
        || (name.starts_with("_mm") && name.contains("_set1_"));
    splat && c.args.len() == 1 && pred(&c.args[0])
}

/// A vector of negative zeros: a float splat of `-0.0`, or the sign-bit
/// integer mask reinterpreted as floats (`_mm*_castsi*_p[sd](_mm*_set1_epi32(SIGN))`,
/// `vreinterpret*_f32_u32(vdup*_n_u32(SIGN))` and the 64-bit forms).
fn is_simd_negative_zero(e: &Expr) -> bool {
    if is_simd_splat_of(e, &|x| is_negative_zero_literal(x)) {
        return true;
    }
    let Expr::Call(c) = peel(e) else { return false };
    let Some(name) = call_name(c) else {
        return false;
    };
    let cast = (name.starts_with("_mm") && name.contains("_castsi"))
        || (name.starts_with("vreinterpret") && (name.contains("_f32_") || name.contains("_f64_")));
    cast && c.args.len() == 1 && is_simd_splat_of(&c.args[0], &is_sign_mask)
}

/// The sign bit of an f32 or f64 as an integer literal, shift, or a name bound
/// to one: `0x8000_0000`, `1 << 31`, `0x8000_0000_0000_0000`, `1 << 63`.
fn is_sign_mask(e: &Expr) -> bool {
    is_sign_mask_literal(e) || named_kind(e) == Some(ConstKind::SignMask)
}

fn is_sign_mask_literal(e: &Expr) -> bool {
    match peel(e) {
        Expr::Lit(l) => matches!(&l.lit, Lit::Int(i)
            if matches!(i.base10_parse::<u64>(), Ok(0x8000_0000) | Ok(0x8000_0000_0000_0000))),
        Expr::Binary(b) if matches!(b.op, BinOp::Shl(_)) => {
            matches!(peel(&b.left), Expr::Lit(l) if matches!(&l.lit, Lit::Int(i) if matches!(i.base10_parse::<u64>(), Ok(1))))
                && matches!(peel(&b.right), Expr::Lit(l) if matches!(&l.lit, Lit::Int(i) if matches!(i.base10_parse::<u64>(), Ok(31) | Ok(63))))
        }
        // `0x8000_0000u32 as i32`, and `i32::MIN` / `i64::MIN` (the same bits)
        Expr::Cast(c) => is_sign_mask_literal(&c.expr),
        Expr::Path(p) => {
            let segs: Vec<String> = p
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            matches!(segs.as_slice(), [t, m] if m == "MIN" && (t == "i32" || t == "i64"))
        }
        _ => false,
    }
}

/// The call spellings of a negation: `Neg::neg(x)` (any path ending in `neg`
/// with one argument), the sign-bit flip `f32::from_bits(x.to_bits() ^ SIGN)`,
/// and the SIMD forms listed in the module doc.
fn is_negation_call_of(c: &ExprCall, target: &dyn Fn(&Expr) -> bool) -> bool {
    let Some(name) = call_name(c) else {
        return false;
    };
    let args: Vec<&Expr> = c.args.iter().collect();
    let neg_one = |e: &Expr| is_negative_one_literal(e);
    match (name.as_str(), args.as_slice()) {
        ("neg", [x]) => target(x),
        ("from_bits", [x]) => match peel(x) {
            Expr::Binary(b) if matches!(b.op, BinOp::BitXor(_)) => {
                let bits_of = |e: &Expr| {
                    matches!(peel(e), Expr::MethodCall(m)
                    if m.method == "to_bits" && m.args.is_empty() && target(&m.receiver))
                };
                (bits_of(&b.left) && is_sign_mask(&b.right))
                    || (bits_of(&b.right) && is_sign_mask(&b.left))
            }
            _ => false,
        },
        (n, [x]) if n.starts_with("vneg") => target(x),
        (n, [z, x]) if n.starts_with("vsub") => is_simd_negative_zero(z) && target(x),
        (n, [x, k]) if n.starts_with("vmul") && n.contains("_n_") => neg_one(k) && target(x),
        (n, [x, y]) if n.starts_with("vmul") => {
            (is_simd_splat_of(y, &neg_one) && target(x))
                || (is_simd_splat_of(x, &neg_one) && target(y))
        }
        (n, [x, y])
            if n.starts_with("_mm") && (n.ends_with("_xor_ps") || n.ends_with("_xor_pd")) =>
        {
            (is_simd_negative_zero(y) && target(x)) || (is_simd_negative_zero(x) && target(y))
        }
        (n, [z, x])
            if n.starts_with("_mm") && (n.ends_with("_sub_ps") || n.ends_with("_sub_pd")) =>
        {
            is_simd_negative_zero(z) && target(x)
        }
        (n, [x, y])
            if n.starts_with("_mm") && (n.ends_with("_mul_ps") || n.ends_with("_mul_pd")) =>
        {
            (is_simd_splat_of(y, &neg_one) && target(x))
                || (is_simd_splat_of(x, &neg_one) && target(y))
        }
        _ => false,
    }
}

/// Every non-let-binding spelling of "negate the whole value of `target`":
/// `-target`, `target.neg()`, `-0.0 - target`, `target * -1.0` / `-1.0 * target`,
/// `target / -1.0`, and the call spellings of [`is_negation_call_of`].
fn is_negation_of(e: &Expr, target: &dyn Fn(&Expr) -> bool) -> bool {
    match e {
        Expr::Unary(u) if matches!(u.op, UnOp::Neg(_)) => target(&u.expr),
        Expr::MethodCall(m) if is_neg_method_call(m) => target(&m.receiver),
        Expr::Binary(b) => is_negation_of_binary(b, target),
        Expr::Call(c) => is_negation_call_of(c, target),
        _ => false,
    }
}

/// The `ExprBinary`-specific half of [`is_negation_of`] (forms D/E), usable
/// directly from `visit_expr_binary` without needing an owned `Expr::Binary`
/// to match against (syn's types here are not `Clone`).
fn is_negation_of_binary(b: &ExprBinary, target: &dyn Fn(&Expr) -> bool) -> bool {
    if matches!(b.op, BinOp::Sub(_)) && is_negative_zero_literal(&b.left) {
        return target(&b.right);
    }
    if matches!(b.op, BinOp::Mul(_)) {
        return (is_negative_one_literal(&b.left) && target(&b.right))
            || (is_negative_one_literal(&b.right) && target(&b.left));
    }
    // `x / -1.0`, and on a binding `x *= -1.0` / `x /= -1.0`
    if matches!(
        b.op,
        BinOp::Div(_) | BinOp::MulAssign(_) | BinOp::DivAssign(_)
    ) {
        return is_negative_one_literal(&b.right) && target(&b.left);
    }
    false
}

fn is_path_ident(e: &Expr, ident: &str) -> bool {
    matches!(peel(e), Expr::Path(p) if p.path.get_ident().is_some_and(|i| i == ident))
}

fn ident_name(e: &Expr) -> Option<String> {
    match peel(e) {
        Expr::Path(p) => p.path.get_ident().map(|i| i.to_string()),
        _ => None,
    }
}

fn is_negation_of_ident(e: &Expr, ident: &str) -> bool {
    is_negation_of(e, &|t| is_path_ident(t, ident))
}

impl<'a> Visitor<'a> {
    fn record_finding_at(&mut self, span: proc_macro2::Span) {
        self.record_kind_at(span, FindingKind::NegatedMulAdd);
    }

    fn record_kind_at(&mut self, span: proc_macro2::Span, kind: FindingKind) {
        let line = span.start().line;
        self.findings.push(Finding {
            rel: self.rel.to_string(),
            line,
            excerpt: excerpt_at(self.src, line),
            kind,
        });
    }

    fn count(&mut self, kind: Option<FmaKind>) {
        if kind.is_some() && self.test_depth > 0 {
            self.counts.test_fma_calls += 1;
        }
        match kind {
            Some(FmaKind::MulAdd) => self.counts.mul_add_calls += 1,
            Some(FmaKind::Scalar) => self.counts.scalar_fma_calls += 1,
            Some(FmaKind::Simd) => self.counts.simd_fma_calls += 1,
            None => {}
        }
    }

    fn record_unary(&mut self, u: &ExprUnary) {
        if !matches!(u.op, UnOp::Neg(_)) {
            return;
        }
        self.counts.unary_neg_exprs += 1;
        if is_fma_result(&u.expr) {
            self.record_finding_at(u.op.span());
        }
    }

    fn record_binary(&mut self, b: &ExprBinary) {
        if is_negation_of_binary(b, &is_fma_result) {
            self.record_finding_at(b.op.span());
        }
    }

    /// Form F: `let t = <mul_add>; ... -t;` — a recursive, shadow-aware
    /// best-effort textual-scope check (see module doc), not full dataflow.
    /// Finds a negation of `ident` anywhere reachable from the statements
    /// following the binding, at any nesting depth, not just a statement
    /// whose entire top-level expression IS the negation.
    fn record_let_then_negate(&mut self, block: &Block) {
        // names bound to a fma value so far in this block: `let t = <fma>`,
        // an alias `let u = t` of one, or a deferred `let t; ... t = <fma>`
        let mut fma_names: BTreeSet<String> = BTreeSet::new();
        let mut uninit: BTreeSet<String> = BTreeSet::new();
        for (i, stmt) in block.stmts.iter().enumerate() {
            let bound = match stmt {
                Stmt::Local(local) => {
                    let Pat::Ident(pi) = &local.pat else { continue };
                    let ident = pi.ident.to_string();
                    match &local.init {
                        None => {
                            uninit.insert(ident);
                            continue;
                        }
                        Some(init) if init.diverge.is_none() => {
                            let alias = matches!(peel(&init.expr), Expr::Path(p)
                                if p.path.get_ident().is_some_and(|n| fma_names.contains(&n.to_string())));
                            if is_fma_result(&init.expr) || alias {
                                ident
                            } else {
                                fma_names.remove(&ident);
                                continue;
                            }
                        }
                        Some(_) => continue,
                    }
                }
                Stmt::Expr(Expr::Assign(a), _) => {
                    let Some(ident) = ident_name(&a.left) else {
                        continue;
                    };
                    if uninit.contains(&ident) && is_fma_result(&a.right) {
                        uninit.remove(&ident);
                        ident
                    } else {
                        continue;
                    }
                }
                _ => continue,
            };
            fma_names.insert(bound.clone());
            let mut finder = IdentNegationFinder {
                ident: &bound,
                findings: Vec::new(),
            };
            finder.scan_block_from(&block.stmts[i + 1..]);
            for span in finder.findings {
                self.record_finding_at(span);
            }
        }
    }
}

/// Recursively scans a sequence of statements (and every nested block
/// reachable from them — `if`/`match`/loop bodies, closures, any
/// expression position) for a negation of `ident`, in any of the forms
/// [`is_negation_of`] recognises, at any syntactic depth: a `return`
/// expression, a function-call argument, an array/struct-literal field, the
/// right-hand side of an index assignment, a branch of an `if`/`match`, or
/// buried inside a larger unrelated expression. Shadow-aware: once a
/// (sub)block rebinds (`let ident = ...`) or reassigns (`ident = ...`) the
/// same name, tracking stops for the remainder of *that* block only — a
/// sibling branch (e.g. the other arm of the `if` that caused the rebind)
/// and anything after the enclosing block ends are unaffected, since each
/// nested block gets its own, independent scan starting fresh from its own
/// first statement.
struct IdentNegationFinder<'a> {
    ident: &'a str,
    findings: Vec<proc_macro2::Span>,
}

impl<'a> IdentNegationFinder<'a> {
    fn scan_block_from(&mut self, stmts: &[Stmt]) {
        for stmt in stmts {
            let shadowed = match stmt {
                Stmt::Local(l) => {
                    if let Some(init) = &l.init {
                        self.visit_expr(&init.expr);
                        if let Some((_, diverge_expr)) = &init.diverge {
                            self.visit_expr(diverge_expr);
                        }
                    }
                    matches!(&l.pat, Pat::Ident(pi) if pi.ident == self.ident)
                }
                Stmt::Expr(Expr::Assign(a), _) => {
                    self.visit_expr(&a.right);
                    let reassigns = is_path_ident(&a.left, self.ident);
                    if !reassigns {
                        self.visit_expr(&a.left);
                    }
                    reassigns
                }
                Stmt::Expr(e, _) => {
                    self.visit_expr(e);
                    false
                }
                _ => false,
            };
            if shadowed {
                break;
            }
        }
    }
}

impl<'a, 'ast> Visit<'ast> for IdentNegationFinder<'a> {
    fn visit_expr(&mut self, node: &'ast Expr) {
        if is_negation_of_ident(node, self.ident) {
            let span = match node {
                Expr::Unary(u) => u.op.span(),
                Expr::MethodCall(m) => m.method.span(),
                Expr::Binary(b) => b.op.span(),
                _ => node.span(),
            };
            self.findings.push(span);
            return;
        }
        visit::visit_expr(self, node);
    }

    /// Every nested block (`if`/`match`/loop bodies, a bare `{ }` expression)
    /// gets its own independent scan via [`IdentNegationFinder::scan_block_from`]
    /// instead of the default per-statement descent, so shadowing introduced
    /// inside one branch cannot leak into a sibling branch or past the block.
    fn visit_block(&mut self, node: &'ast Block) {
        self.scan_block_from(&node.stmts);
    }
}

impl<'a, 'ast> Visit<'ast> for Visitor<'a> {
    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        let name = node.ident.to_string();
        if node.content.is_some() {
            self.mod_stack.push(name);
            visit::visit_item_mod(self, node);
            self.mod_stack.pop();
            return;
        }
        let dir = module_dir(self.path);
        let candidates = match path_attr(&node.attrs) {
            // outside an inline module a `#[path]` is relative to the declaring file's directory
            Some(p) if self.mod_stack.is_empty() => {
                vec![self.path.parent().unwrap_or(Path::new("")).join(p)]
            }
            Some(p) => vec![self.mod_stack.iter().fold(dir, |d, m| d.join(m)).join(p)],
            None => {
                let base = self.mod_stack.iter().fold(dir, |d, m| d.join(m));
                vec![
                    base.join(format!("{name}.rs")),
                    base.join(&name).join("mod.rs"),
                ]
            }
        };
        self.mod_decls.push(ModDecl {
            candidates,
            test_only: self.test_depth > 0,
        });
        visit::visit_item_mod(self, node);
    }

    fn visit_stmt(&mut self, node: &'ast syn::Stmt) {
        // a statement compiled only under test (`#[cfg(test)] let ..;` / `#[cfg(test)] { .. }`) inside a library fn
        let attrs: &[syn::Attribute] = match node {
            syn::Stmt::Local(l) => &l.attrs,
            syn::Stmt::Macro(m) => &m.attrs,
            syn::Stmt::Expr(e, _) => expr_attrs(e),
            syn::Stmt::Item(_) => &[],
        };
        let test = is_test_only(attrs);
        self.test_depth += usize::from(test);
        visit::visit_stmt(self, node);
        self.test_depth -= usize::from(test);
    }

    fn visit_item(&mut self, node: &'ast syn::Item) {
        let attrs: &[syn::Attribute] = match node {
            syn::Item::Const(i) => &i.attrs,
            syn::Item::Enum(i) => &i.attrs,
            syn::Item::ExternCrate(i) => &i.attrs,
            syn::Item::Fn(i) => &i.attrs,
            syn::Item::ForeignMod(i) => &i.attrs,
            syn::Item::Impl(i) => &i.attrs,
            syn::Item::Macro(i) => &i.attrs,
            syn::Item::Mod(i) => &i.attrs,
            syn::Item::Static(i) => &i.attrs,
            syn::Item::Struct(i) => &i.attrs,
            syn::Item::Trait(i) => &i.attrs,
            syn::Item::TraitAlias(i) => &i.attrs,
            syn::Item::Type(i) => &i.attrs,
            syn::Item::Union(i) => &i.attrs,
            syn::Item::Use(i) => &i.attrs,
            _ => &[],
        };
        let test = is_test_only(attrs);
        self.test_depth += usize::from(test);
        visit::visit_item(self, node);
        self.test_depth -= usize::from(test);
    }

    fn visit_impl_item(&mut self, node: &'ast syn::ImplItem) {
        let attrs: &[syn::Attribute] = match node {
            syn::ImplItem::Const(i) => &i.attrs,
            syn::ImplItem::Fn(i) => &i.attrs,
            syn::ImplItem::Type(i) => &i.attrs,
            syn::ImplItem::Macro(i) => &i.attrs,
            _ => &[],
        };
        let test = is_test_only(attrs);
        self.test_depth += usize::from(test);
        visit::visit_impl_item(self, node);
        self.test_depth -= usize::from(test);
    }

    fn visit_trait_item(&mut self, node: &'ast syn::TraitItem) {
        let attrs: &[syn::Attribute] = match node {
            syn::TraitItem::Const(i) => &i.attrs,
            syn::TraitItem::Fn(i) => &i.attrs,
            syn::TraitItem::Type(i) => &i.attrs,
            syn::TraitItem::Macro(i) => &i.attrs,
            _ => &[],
        };
        let test = is_test_only(attrs);
        self.test_depth += usize::from(test);
        visit::visit_trait_item(self, node);
        self.test_depth -= usize::from(test);
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        self.count(method_fma_kind(node));
        if is_neg_method_call(node) && is_fma_result(&node.receiver) {
            self.record_finding_at(node.method.span());
        }
        visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        let kind = path_call_fma_kind(node);
        self.count(kind);
        if is_negation_call_of(node, &is_fma_result) {
            self.record_finding_at(node.func.span());
        }
        if kind.is_some() {
            // the callee path is the call itself: only the arguments are visited,
            // so `visit_expr_path` sees a fma-family path only where it is a value
            for arg in &node.args {
                self.visit_expr(arg);
            }
        } else {
            visit::visit_expr_call(self, node);
        }
    }

    fn visit_expr_path(&mut self, node: &'ast syn::ExprPath) {
        if node
            .path
            .segments
            .last()
            .is_some_and(|s| fma_kind(&s.ident.to_string()).is_some())
        {
            self.record_kind_at(node.span(), FindingKind::AliasedFma);
        }
        visit::visit_expr_path(self, node);
    }

    fn visit_item_use(&mut self, node: &'ast ItemUse) {
        fn renames(tree: &UseTree, out: &mut Vec<proc_macro2::Span>) {
            match tree {
                UseTree::Path(p) => renames(&p.tree, out),
                UseTree::Rename(r)
                    if fma_kind(&r.ident.to_string()).is_some() && r.ident != r.rename =>
                {
                    out.push(r.rename.span());
                }
                UseTree::Group(g) => g.items.iter().for_each(|t| renames(t, out)),
                _ => {}
            }
        }
        let mut spans = Vec::new();
        renames(&node.tree, &mut spans);
        for span in spans {
            self.record_kind_at(span, FindingKind::AliasedFma);
        }
        visit::visit_item_use(self, node);
    }

    fn visit_expr_unary(&mut self, node: &'ast ExprUnary) {
        self.record_unary(node);
        visit::visit_expr_unary(self, node);
    }

    fn visit_expr_binary(&mut self, node: &'ast ExprBinary) {
        self.record_binary(node);
        visit::visit_expr_binary(self, node);
    }

    fn visit_block(&mut self, node: &'ast Block) {
        self.record_let_then_negate(node);
        visit::visit_block(self, node);
    }

    fn visit_macro(&mut self, node: &'ast Macro) {
        // syn does not expand macros: the tokens inside are opaque to the
        // AST walk above, so this cannot tell negated from safe — hard fail
        // by default (see module doc), never allowlistable in src/.
        // identifiers only: a fma name inside a string literal (an assert
        // message) is not a call
        fn mentions_fma(tokens: proc_macro2::TokenStream) -> bool {
            tokens.into_iter().any(|t| match t {
                proc_macro2::TokenTree::Ident(i) => fma_kind(&i.to_string()).is_some(),
                proc_macro2::TokenTree::Group(g) => mentions_fma(g.stream()),
                _ => false,
            })
        }
        if mentions_fma(node.tokens.clone()) {
            let line = node.span().start().line;
            self.findings.push(Finding {
                rel: self.rel.to_string(),
                line,
                excerpt: excerpt_at(self.src, line),
                kind: FindingKind::MacroMulAdd,
            });
        }
        visit::visit_macro(self, node);
    }
}

fn is_allowlist_eligible(rel: &str) -> bool {
    let parts: Vec<&str> = rel.split(['/', '\\']).collect();
    ALLOWLIST_SCOPE_DIRS
        .iter()
        .any(|d| parts.iter().any(|p| p == d))
}

fn walk_rs_files(base: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(base) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            walk_rs_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

struct Report {
    errors: Vec<String>,
    files: usize,
    unary_neg_exprs: usize,
    /// `mul_add` calls (method and path spellings)
    mul_add_calls: usize,
    /// `fma` / `fmaf` / `fmaf32` / `fmaf64` calls
    scalar_fma_calls: usize,
    /// SIMD fused multiply-add intrinsic calls
    simd_fma_calls: usize,
    /// every fma-family call
    fma_calls: usize,
    /// fma-family calls of the library: in files under a `src/` directory, outside test-only items
    src_fma_calls: usize,
    /// fma-family calls in test-only items (`#[test]`, `#[cfg(test)]`, ...) of files under `src/`
    src_test_fma_calls: usize,
    violations: usize,
    allowed: usize,
    macro_hits: usize,
    macro_allowed: usize,
    aliases: usize,
}

/// `[[allow]]`: a finding by path, exact trimmed line text and count.
struct AllowEntry {
    path: String,
    line: String,
    count: usize,
    reason: String,
}

struct Allowlist {
    expected_fma_calls: Option<usize>,
    expected_src_fma_calls: Option<usize>,
    entries: Vec<AllowEntry>,
    errors: Vec<String>,
}

fn read_allowlist(root: &Path) -> Allowlist {
    let mut out = Allowlist {
        expected_fma_calls: None,
        expected_src_fma_calls: None,
        entries: Vec::new(),
        errors: Vec::new(),
    };
    let path = root.join(ALLOWLIST_REL);
    let Ok(text) = fs::read_to_string(&path) else {
        return out;
    };
    let table = match text.parse::<toml::Table>() {
        Ok(t) => t,
        Err(e) => {
            out.errors
                .push(format!("error: {ALLOWLIST_REL}: TOML として読めない ({e})"));
            return out;
        }
    };
    let count = |key: &str| {
        table
            .get(key)
            .and_then(toml::Value::as_integer)
            .and_then(|v| usize::try_from(v).ok())
    };
    out.expected_fma_calls = count("expected-fma-calls");
    out.expected_src_fma_calls = count("expected-src-fma-calls");
    if let Some(toml::Value::Array(list)) = table.get("allow") {
        for (i, item) in list.iter().enumerate() {
            let field = |k: &str| {
                item.get(k)
                    .and_then(toml::Value::as_str)
                    .map(str::to_string)
            };
            let n = item
                .get("count")
                .and_then(toml::Value::as_integer)
                .and_then(|v| usize::try_from(v).ok());
            match (field("path"), field("line"), n, field("reason")) {
                (Some(path), Some(line), Some(count), Some(reason)) if !reason.trim().is_empty() && count > 0 => {
                    out.entries.push(AllowEntry { path, line: line.trim().to_string(), count, reason });
                }
                _ => out.errors.push(format!(
                    "error: {ALLOWLIST_REL}: allow[{i}] には path / line / count (1 以上) / reason (空でない) が要る"
                )),
            }
        }
    }
    out
}

/// The count headers against what the tool found: both are required, both
/// must be equal (a repository without fma calls declares 0).
fn check_expected(report: &Report, allow: &Allowlist) -> Vec<String> {
    let mut errors = Vec::new();
    for (key, want, got) in [
        (
            "expected-fma-calls",
            allow.expected_fma_calls,
            report.fma_calls,
        ),
        (
            "expected-src-fma-calls",
            allow.expected_src_fma_calls,
            report.src_fma_calls,
        ),
    ] {
        match want {
            None => errors.push(format!(
                "error: {ALLOWLIST_REL} に {key} が無い (見つけた数は {got})"
            )),
            Some(n) if n != got => errors.push(format!(
                "error: {key} = {n} だが見つけた数は {got} — 呼び出しを足した / 消した commit で値を更新する (走査範囲が変わったなら原因を確かめる)"
            )),
            Some(_) => {}
        }
    }
    errors
}

fn check(root: &Path, src_dirs: &[String]) -> Report {
    let mut errors = Vec::new();
    let mut files = 0usize;
    let mut totals = FileCounts::default();
    let mut found: Vec<Finding> = Vec::new();
    // per src/ file: (fma calls, test-only fma calls), and its `mod y;` declarations
    let mut src_counts: BTreeMap<PathBuf, (usize, usize)> = BTreeMap::new();
    let mut decls: BTreeMap<PathBuf, Vec<ModDecl>> = BTreeMap::new();

    for src_dir in src_dirs {
        let base = root.join(src_dir);
        if !base.is_dir() {
            continue;
        }
        let mut rs_files = Vec::new();
        walk_rs_files(&base, &mut rs_files);
        for path in rs_files {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            let Ok(src) = fs::read_to_string(&path) else {
                errors.push(format!("error: {rel}: 読めなかった"));
                continue;
            };
            files += 1;
            let parsed = match syn::parse_file(&src) {
                Ok(f) => f,
                Err(e) => {
                    errors.push(format!(
                        "error: {rel}: syn が parse できない — 検査ができていない ({e})"
                    ));
                    continue;
                }
            };
            let mut visitor = Visitor {
                rel: &rel,
                path: &path,
                src: &src,
                findings: Vec::new(),
                counts: FileCounts::default(),
                test_depth: 0,
                mod_stack: Vec::new(),
                mod_decls: Vec::new(),
            };
            NAMED.with(|n| *n.borrow_mut() = named_constants(&parsed));
            visitor.visit_file(&parsed);
            NAMED.with(|n| n.borrow_mut().clear());
            totals.unary_neg_exprs += visitor.counts.unary_neg_exprs;
            totals.mul_add_calls += visitor.counts.mul_add_calls;
            totals.scalar_fma_calls += visitor.counts.scalar_fma_calls;
            totals.simd_fma_calls += visitor.counts.simd_fma_calls;
            if rel.split('/').any(|p| p == "src") {
                src_counts.insert(
                    path.clone(),
                    (visitor.counts.fma_calls(), visitor.counts.test_fma_calls),
                );
            }
            decls.insert(path.clone(), visitor.mod_decls);
            found.extend(visitor.findings);
        }
    }
    // the library's own calls: a test-only item in src/, or a whole file reached only through a test-only
    // `mod y;`, is not in the release build
    let test_files = test_only_files(&decls);
    let mut src_fma_calls = 0usize;
    let mut src_test_fma_calls = 0usize;
    for (path, (all, test)) in &src_counts {
        let test = if test_files.contains(path) {
            *all
        } else {
            *test
        };
        src_fma_calls += all - test;
        src_test_fma_calls += test;
    }

    let mut report = Report {
        errors: Vec::new(),
        files,
        unary_neg_exprs: totals.unary_neg_exprs,
        mul_add_calls: totals.mul_add_calls,
        scalar_fma_calls: totals.scalar_fma_calls,
        simd_fma_calls: totals.simd_fma_calls,
        fma_calls: totals.fma_calls(),
        src_fma_calls,
        src_test_fma_calls,
        violations: 0,
        allowed: 0,
        macro_hits: 0,
        macro_allowed: 0,
        aliases: 0,
    };
    if files == 0 {
        errors.push(format!(
            "error: src file を 1 つも読めなかった — 検査が空振りしている (対象: {})",
            src_dirs.join(", ")
        ));
        report.errors = errors;
        return report;
    }

    let allowlist = read_allowlist(root);
    errors.extend(allowlist.errors.iter().cloned());
    // findings grouped by (path, exact trimmed line text)
    let mut groups: BTreeMap<(String, String), Vec<&Finding>> = BTreeMap::new();
    for f in &found {
        match f.kind {
            FindingKind::NegatedMulAdd => report.violations += 1,
            FindingKind::MacroMulAdd => report.macro_hits += 1,
            FindingKind::AliasedFma => report.aliases += 1,
        }
        groups
            .entry((f.rel.clone(), f.excerpt.clone()))
            .or_default()
            .push(f);
    }
    let mut used: BTreeSet<usize> = BTreeSet::new();
    for ((rel, text), fs_) in &groups {
        let entry = allowlist
            .entries
            .iter()
            .position(|e| &e.path == rel && &e.line == text);
        let aliased = fs_
            .iter()
            .any(|f| matches!(f.kind, FindingKind::AliasedFma));
        if let Some(idx) = entry {
            used.insert(idx);
            let e = &allowlist.entries[idx];
            if aliased {
                errors.push(format!(
                    "error: {rel}:{}: {text}  (fma-family の改名 import は allowlist 不可、元の名前で import する — allowlist の行も消す)",
                    fs_[0].line
                ));
            } else if !is_allowlist_eligible(rel) {
                let label = if fs_
                    .iter()
                    .any(|f| matches!(f.kind, FindingKind::MacroMulAdd))
                {
                    "src/ 配下は allowlist 不可、macro 呼び出しの外に出して書き直す — allowlist の行も消す"
                } else {
                    "src/ 配下は allowlist 不可、Law form への書き換えのみ — allowlist の行も消す"
                };
                errors.push(format!("error: {rel}:{}: {text}  ({label})", fs_[0].line));
            } else if e.count != fs_.len() {
                errors.push(format!(
                    "error: {rel}: \"{text}\": allowlist の count = {} だが見つけた数は {} — 実物に合わせる",
                    e.count,
                    fs_.len()
                ));
            } else {
                for f in fs_ {
                    match f.kind {
                        FindingKind::NegatedMulAdd => report.allowed += 1,
                        FindingKind::MacroMulAdd => report.macro_allowed += 1,
                        FindingKind::AliasedFma => {}
                    }
                }
            }
            continue;
        }
        for f in fs_ {
            let label = match f.kind {
                FindingKind::NegatedMulAdd => "呼び出し結果全体を negate する fma、禁止形",
                FindingKind::MacroMulAdd => {
                    "macro 呼び出しの中に fma-family の名前がある、AST 未検証 — allowlist に理由付きで明記する"
                }
                FindingKind::AliasedFma => {
                    "fma-family を別名で import している、または値として使っている、検査が呼び出しの名前で見るので見えなくなる — 元の名前で直接呼ぶ"
                }
            };
            errors.push(format!(
                "error: {}:{}: {}  ({label})",
                f.rel, f.line, f.excerpt
            ));
        }
    }
    for (idx, e) in allowlist.entries.iter().enumerate() {
        if !used.contains(&idx) {
            errors.push(format!(
                "error: {}: \"{}\": allowlist にあるが実物が無い — 行を消す ({})",
                e.path, e.line, e.reason
            ));
        }
    }
    report.errors = errors;
    report
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut root = std::env::current_dir().expect("cwd");
    // tools/no-outer-fneg-of-fma/ -> repo root is two levels up when run via
    // `cargo run` from inside this crate; `--root` overrides unconditionally.
    if let Some(i) = args.iter().position(|a| a == "--root") {
        root = PathBuf::from(&args[i + 1]);
    }
    let mut src_dirs: Vec<String> = workspace_src_dirs(&root);
    if let Some(i) = args.iter().position(|a| a == "--src") {
        let rest: Vec<String> = args[i + 1..]
            .iter()
            .take_while(|a| !a.starts_with("--"))
            .cloned()
            .collect();
        if rest.is_empty() {
            eprintln!("error: --src には 1 つ以上の dir を渡す");
            std::process::exit(2);
        }
        src_dirs = rest;
    }
    if src_dirs.is_empty() {
        eprintln!(
            "error: {}/Cargo.toml から走査する dir を決められなかった ([workspace] members も [package] も無い、または members に glob がある)",
            root.display()
        );
        std::process::exit(2);
    }

    let mut report = check(&root, &src_dirs);
    if report.files > 0 {
        let allow = read_allowlist(&root);
        let expected = check_expected(&report, &allow);
        report.errors.extend(expected);
    }
    for e in &report.errors {
        println!("{e}");
    }
    println!(
        "compared: files parsed {}, fma calls {} (mul_add {}, scalar fma {}, simd {}; library in src/ {}, test-only in src/ {}), unary-neg exprs examined {}, fma under neg {}, allowed (tests/examples) {}, macro fma mentions {}, macro allowed {}, renamed fma imports {}",
        report.files,
        report.fma_calls,
        report.mul_add_calls,
        report.scalar_fma_calls,
        report.simd_fma_calls,
        report.src_fma_calls,
        report.src_test_fma_calls,
        report.unary_neg_exprs,
        report.violations,
        report.allowed,
        report.macro_hits,
        report.macro_allowed,
        report.aliases
    );
    if !report.errors.is_empty() {
        println!("no-outer-fneg-of-fma: {} violation(s)", report.errors.len());
        std::process::exit(1);
    }
    println!("no-outer-fneg-of-fma: ok");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn run(files: &[(&str, &str)], allowlist: Option<&str>) -> Report {
        let dir = tempfile::tempdir().expect("tmpdir");
        for (rel, text) in files {
            let path = dir.path().join(rel);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            let mut f = fs::File::create(&path).unwrap();
            f.write_all(text.as_bytes()).unwrap();
        }
        if let Some(a) = allowlist {
            let path = dir.path().join(ALLOWLIST_REL);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            let mut f = fs::File::create(&path).unwrap();
            f.write_all(a.as_bytes()).unwrap();
        }
        let src_dirs: Vec<String> = DEFAULT_SRC_DIRS.iter().map(|s| s.to_string()).collect();
        check(dir.path(), &src_dirs)
    }

    const SRC: &str = "alice-lol/src/lib.rs";

    /// One `[[allow]]` entry in the TOML allowlist format.
    fn allow(path: &str, line: &str, count: usize) -> String {
        format!("[[allow]]\npath = \"{path}\"\nline = \"{line}\"\ncount = {count}\nreason = \"comparison baseline\"\n")
    }

    const CLEAN: &str = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    a.mul_add(b, -c)\n}\n";

    #[test]
    fn a_clean_tree_passes() {
        let r = run(&[(SRC, CLEAN)], None);
        assert_eq!(r.errors, Vec::<String>::new());
        assert_eq!(r.violations, 0);
        assert!(r.mul_add_calls > 0);
    }

    #[test]
    fn parenthesised_outer_negation_is_caught() {
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    -(a.mul_add(b, c))\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
        assert!(r.errors.iter().any(|e| e.contains("lib.rs:2")));
    }

    #[test]
    fn bare_unparenthesised_outer_negation_is_caught() {
        // the exact form 53 found missed in both LOL and Physics: no parens,
        // method call binds tighter than unary minus, same AST as above.
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    -a.mul_add(b, c)\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn space_before_receiver_is_still_caught() {
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    - a.mul_add(b, c)\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn parenthesised_receiver_then_chained_call_is_caught() {
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    -(a).mul_add(b, c)\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn field_access_receiver_is_caught() {
        let src =
            "struct S { y: f32 }\nfn f(x: S, b: f32, c: f32) -> f32 {\n    -x.y.mul_add(b, c)\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn index_receiver_is_caught() {
        let src = "fn f(v: [f32; 3], b: f32, c: f32) -> f32 {\n    -v[0].mul_add(b, c)\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn path_call_form_is_caught() {
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    -f32::mul_add(a, b, c)\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn doubly_parenthesised_is_caught() {
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    -((a.mul_add(b, c)))\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn inside_a_closure_is_caught() {
        let src = "fn f() -> f32 {\n    let g = |a: f32, b: f32, c: f32| -a.mul_add(b, c);\n    g(1.0, 2.0, 3.0)\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn inside_a_const_is_caught() {
        // not evaluable as a real const (a/b/c undefined) but syn parses it
        // as a const item body and this tool only parses, never evaluates
        let src = "const X: fn(f32, f32, f32) -> f32 = |a, b, c| -a.mul_add(b, c);\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn binary_subtraction_before_mul_add_is_not_flagged() {
        let src = "fn f(x: f32, a: f32, b: f32, c: f32) -> f32 {\n    x - a.mul_add(b, c)\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 0);
    }

    #[test]
    fn negating_an_argument_before_the_call_is_not_flagged() {
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    (-a).mul_add(b, -c)\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 0);
    }

    #[test]
    fn negating_a_nested_argument_is_not_flagged() {
        let src = "fn f(a: f32, b: f32, c: f32, d: f32) -> f32 {\n    a.mul_add(b, -(c * d))\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 0);
    }

    #[test]
    fn negating_something_chained_after_mul_add_is_not_flagged() {
        // negates the result of .abs(), not of mul_add — a different
        // operation entirely, must not be flagged
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    -a.mul_add(b, c).abs()\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 0);
    }

    #[test]
    fn extra_addend_inside_the_outer_parens_is_not_flagged() {
        let src = "fn f(a: f32, b: f32, c: f32, d: f32) -> f32 {\n    -(a.mul_add(b, c) + d)\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 0);
    }

    #[test]
    fn a_src_site_fails_even_when_the_allowlist_carries_it() {
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    -(a.mul_add(b, c))\n}\n";
        let r = run(&[(SRC, src)], Some(&allow(SRC, "-(a.mul_add(b, c))", 1)));
        assert_eq!(r.violations, 1);
        assert_eq!(r.allowed, 0);
        assert!(r.errors.iter().any(|e| e.contains("allowlist 不可")));
    }

    #[test]
    fn a_tests_dir_site_passes_when_the_allowlist_carries_it() {
        let oracle = "alice-lol/tests/oracle.rs";
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    -(a.mul_add(b, c))\n}\n";
        let r = run(
            &[(oracle, src)],
            Some(&allow(oracle, "-(a.mul_add(b, c))", 1)),
        );
        assert_eq!(r.errors, Vec::<String>::new());
        assert_eq!(r.violations, 1);
        assert_eq!(r.allowed, 1);
    }

    #[test]
    fn an_examples_dir_site_passes_when_the_allowlist_carries_it() {
        let oracle = "alice-lol/examples/compare_forms.rs";
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    -(a.mul_add(b, c))\n}\n";
        let r = run(
            &[(oracle, src)],
            Some(&allow(oracle, "-(a.mul_add(b, c))", 1)),
        );
        assert_eq!(r.errors, Vec::<String>::new());
        assert_eq!(r.violations, 1);
        assert_eq!(r.allowed, 1);
    }

    #[test]
    fn an_allowlist_entry_without_the_site_fails() {
        let r = run(&[(SRC, CLEAN)], Some(&allow(SRC, "-(a.mul_add(b, c))", 1)));
        assert!(r
            .errors
            .iter()
            .any(|e| e.contains("allowlist にあるが実物が無い")));
    }

    #[test]
    fn no_readable_src_file_fails() {
        let r = run(&[], None);
        assert_eq!(r.files, 0);
        assert!(r.errors.iter().any(|e| e.contains("1 つも読めなかった")));
    }

    fn header(all: usize, src: usize) -> String {
        format!("expected-fma-calls = {all}\nexpected-src-fma-calls = {src}\n")
    }

    fn expected_errors(files: &[(&str, &str)], allowlist: Option<&str>) -> Vec<String> {
        let dir = tempfile::tempdir().expect("tmpdir");
        for (rel, text) in files {
            let path = dir.path().join(rel);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, text).unwrap();
        }
        if let Some(a) = allowlist {
            let path = dir.path().join(ALLOWLIST_REL);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, a).unwrap();
        }
        let src_dirs: Vec<String> = DEFAULT_SRC_DIRS.iter().map(|s| s.to_string()).collect();
        let r = check(dir.path(), &src_dirs);
        check_expected(&r, &read_allowlist(dir.path()))
    }

    #[test]
    fn a_missing_count_header_fails() {
        let e = expected_errors(&[(SRC, CLEAN)], None);
        assert_eq!(e.len(), 2, "{e:?}");
        assert!(e.iter().all(|x| x.contains("が無い")));
    }

    #[test]
    fn a_count_header_that_differs_fails_with_both_numbers() {
        let e = expected_errors(&[(SRC, CLEAN)], Some(&header(2, 1)));
        assert!(
            e.iter()
                .any(|x| x.contains("expected-fma-calls = 2") && x.contains("1")),
            "{e:?}"
        );
    }

    #[test]
    fn a_matching_count_header_passes() {
        let e = expected_errors(&[(SRC, CLEAN)], Some(&header(1, 1)));
        assert_eq!(e, Vec::<String>::new());
    }

    #[test]
    fn zero_calls_pass_only_when_declared_zero() {
        // a repository without any fma call declares 0 explicitly
        let files = [(SRC, "fn f() {}\n")];
        assert_eq!(
            expected_errors(&files, Some(&header(0, 0))),
            Vec::<String>::new()
        );
        // without the declaration the same tree fails (a broken walk looks the same)
        assert!(!expected_errors(&files, None).is_empty());
    }

    #[test]
    fn a_call_found_while_zero_is_declared_fails() {
        let e = expected_errors(&[(SRC, CLEAN)], Some(&header(0, 0)));
        assert_eq!(e.len(), 2, "{e:?}");
    }

    /// (library, test-only) fma counts of one src file
    fn lib_test(src: &str) -> (usize, usize) {
        let r = run(&[(SRC, src)], None);
        (r.src_fma_calls, r.src_test_fma_calls)
    }

    #[test]
    fn a_cfg_test_module_in_src_is_not_library_code() {
        let src = "pub fn lib(a: f32, b: f32, c: f32) -> f32 { a.mul_add(b, c) }\n\
                   #[cfg(test)]\nmod tests {\n    fn t(a: f32) -> f32 { a.mul_add(a, a) }\n    \
                   mod inner { fn u(a: f32) -> f32 { a.mul_add(a, a) } }\n}\n";
        assert_eq!(lib_test(src), (1, 2));
    }

    #[test]
    fn test_only_predicates_and_attributes() {
        for (attr, test_only) in [
            ("#[test]", true),
            ("#[cfg(test)]", true),
            ("#[cfg(all(test, feature = \"x\"))]", true),
            ("#[cfg(all(feature = \"x\", test))]", true),
            ("#[cfg(any(test, all(test, unix)))]", true),
            ("#[cfg(not(test))]", false),
            ("#[cfg(feature = \"x\")]", false),
            ("#[cfg(any(test, feature = \"x\"))]", false),
            ("#[cfg(all(not(test), unix))]", false),
            ("#[cfg_attr(test, allow(dead_code))]", false),
        ] {
            let src = format!("{attr}\nfn f(a: f32) -> f32 {{ a.mul_add(a, a) }}\n");
            let want = if test_only { (0, 1) } else { (1, 0) };
            assert_eq!(lib_test(&src), want, "{attr}");
        }
    }

    #[test]
    fn a_test_only_impl_item_and_trait_item_are_not_library_code() {
        let src = "struct S;\nimpl S {\n    pub fn lib(a: f32) -> f32 { a.mul_add(a, a) }\n    \
                   #[cfg(test)]\n    fn t(a: f32) -> f32 { a.mul_add(a, a) }\n}\n\
                   trait T {\n    #[cfg(test)]\n    fn d(a: f32) -> f32 { a.mul_add(a, a) }\n}\n";
        assert_eq!(lib_test(src), (1, 2));
    }

    #[test]
    fn a_test_only_impl_block_is_not_library_code() {
        let src = "struct S;\n#[cfg(test)]\nimpl S { fn t(a: f32) -> f32 { a.mul_add(a, a) } }\n";
        assert_eq!(lib_test(src), (0, 1));
    }

    #[test]
    fn the_header_compares_the_library_count_and_the_output_prints_both() {
        let src = "pub fn lib(a: f32) -> f32 { a.mul_add(a, a) }\n#[cfg(test)]\nmod tests { fn t(a: f32) -> f32 { a.mul_add(a, a) } }\n";
        // every call 2, library calls 1: the src header is the library's
        assert_eq!(
            expected_errors(&[(SRC, src)], Some(&header(2, 1))),
            Vec::<String>::new()
        );
        assert!(!expected_errors(&[(SRC, src)], Some(&header(2, 2))).is_empty());
    }

    /// (library, test-only) fma counts of several src files
    fn lib_test_files(files: &[(&str, &str)]) -> (usize, usize) {
        let r = run(files, None);
        (r.src_fma_calls, r.src_test_fma_calls)
    }

    const FMA_FN: &str = "pub fn f(a: f32) -> f32 { a.mul_add(a, a) }\n";

    #[test]
    fn a_file_declared_by_a_test_only_mod_is_not_library_code() {
        let lib = "#[cfg(test)]\nmod t;\nmod l;\n";
        let files = [
            (SRC, lib),
            ("alice-lol/src/t.rs", FMA_FN),
            ("alice-lol/src/l.rs", FMA_FN),
        ];
        assert_eq!(lib_test_files(&files), (1, 1));
    }

    #[test]
    fn the_children_of_a_test_only_file_and_the_mod_rs_form_are_test_only() {
        let files = [
            (SRC, "#[cfg(test)]\nmod t;\n"),
            ("alice-lol/src/t/mod.rs", "mod deeper;\n"),
            ("alice-lol/src/t/deeper.rs", FMA_FN),
        ];
        assert_eq!(lib_test_files(&files), (0, 1));
    }

    #[test]
    fn a_mod_inside_a_test_only_inline_module_and_a_path_attribute_are_followed() {
        let files = [
            (SRC, "#[cfg(test)]\nmod outer {\n    mod inner;\n}\n#[cfg(test)]\n#[path = \"elsewhere.rs\"]\nmod p;\n"),
            ("alice-lol/src/outer/inner.rs", FMA_FN),
            ("alice-lol/src/elsewhere.rs", FMA_FN),
        ];
        assert_eq!(lib_test_files(&files), (0, 2));
    }

    #[test]
    fn a_mod_of_a_non_root_file_resolves_under_its_stem() {
        // a.rs declares `#[cfg(test)] mod b;` -> a/b.rs (2 calls, test-only), not src/b.rs (1 call, library)
        let two = "pub fn f(a: f32) -> f32 { a.mul_add(a, a) }\npub fn g(a: f32) -> f32 { a.mul_add(a, a) }\n";
        let files = [
            (SRC, "mod a;\nmod b;\n"),
            ("alice-lol/src/a.rs", "#[cfg(test)]\nmod b;\n"),
            ("alice-lol/src/a/b.rs", two),
            ("alice-lol/src/b.rs", FMA_FN),
        ];
        assert_eq!(lib_test_files(&files), (1, 2));
    }

    #[test]
    fn a_test_only_statement_in_a_library_fn_is_not_library_code() {
        let src = "pub fn f(a: f32) -> f32 {\n    #[cfg(test)]\n    let _t = a.mul_add(a, a);\n    \
                   #[cfg(test)]\n    { let _u = a.mul_add(a, a); }\n    #[cfg(not(test))]\n    let _l = a.mul_add(a, a);\n    \
                   a.mul_add(a, a)\n}\n";
        assert_eq!(lib_test(src), (2, 2));
    }

    #[test]
    fn a_fma_in_a_test_only_macro_in_src_still_fails() {
        // a macro's tokens are opaque: a fma mentioned in one stays a hard failure in src/, test-only or not
        let src = "#[cfg(test)]\nmod tests {\n    macro_rules! m { ($a:expr) => { $a.mul_add($a, $a) } }\n}\n";
        let r = run(&[(SRC, src)], None);
        assert!(
            r.errors.iter().any(|e| e.contains("macro")),
            "{:?}",
            r.errors
        );
    }

    #[test]
    fn only_src_calls_count_towards_the_src_header() {
        let files = [(SRC, CLEAN), ("alice-lol/tests/t.rs", CLEAN)];
        assert_eq!(
            expected_errors(&files, Some(&header(2, 1))),
            Vec::<String>::new()
        );
    }

    #[test]
    fn an_allowlist_count_that_differs_fails() {
        let oracle = "alice-lol/tests/oracle.rs";
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    -(a.mul_add(b, c))\n}\n";
        let r = run(
            &[(oracle, src)],
            Some(&allow(oracle, "-(a.mul_add(b, c))", 2)),
        );
        assert!(
            r.errors.iter().any(|e| e.contains("count = 2")),
            "{:?}",
            r.errors
        );
        assert_eq!(r.allowed, 0);
    }

    #[test]
    fn an_allowlist_entry_without_a_reason_fails() {
        let oracle = "alice-lol/tests/oracle.rs";
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    -(a.mul_add(b, c))\n}\n";
        let entry = format!("[[allow]]\npath = \"{oracle}\"\nline = \"-(a.mul_add(b, c))\"\ncount = 1\nreason = \"\"\n");
        let r = run(&[(oracle, src)], Some(&entry));
        assert!(
            r.errors.iter().any(|e| e.contains("reason")),
            "{:?}",
            r.errors
        );
    }

    #[test]
    fn macro_mul_add_mention_is_a_hard_fail_by_default() {
        // documented blind spot: the AST walk cannot verify a macro's
        // argument tokens, so any mul_add mention there fails by default,
        // not just a note — the gate must not stay silently blind to new
        // code that hides a negation inside a macro call.
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    a.mul_add(b, c)\n}\nfn g(a: f32, b: f32, c: f32) {\n    println!(\"{}\", a.mul_add(b, c));\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(
            r.violations, 0,
            "the macro mention itself is not an AST-level negation"
        );
        assert_eq!(r.macro_hits, 1);
        assert_eq!(r.macro_allowed, 0);
        assert!(
            !r.errors.is_empty(),
            "unreviewed macro mul_add mention must fail"
        );
    }

    #[test]
    fn macro_hidden_negation_is_also_a_hard_fail() {
        // the specific danger case: a negated mul_add hidden inside a macro
        // invocation — the AST walk cannot see it is negated at all (it is
        // opaque tokens), so it must still fail, same as the plain mention.
        let src = "fn g(a: f32, b: f32, c: f32) {\n    println!(\"{}\", -a.mul_add(b, c));\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.macro_hits, 1);
        assert!(r
            .errors
            .iter()
            .any(|e| e.contains("macro 呼び出しの中に fma-family")));
    }

    #[test]
    fn macro_mul_add_in_src_fails_even_with_an_allowlist_entry() {
        // same no-escape-hatch rule as a direct src/ violation: a macro
        // mention in src/ cannot be allowlisted away, only rewritten out of
        // the macro so the AST walk can verify it directly.
        let src = "fn g(a: f32, b: f32, c: f32) {\n    println!(\"{}\", a.mul_add(b, c));\n}\n";
        let r = run(
            &[(SRC, src)],
            Some(&allow(SRC, "println!(\\\"{}\\\", a.mul_add(b, c));", 1)),
        );
        assert_eq!(r.macro_hits, 1);
        assert_eq!(r.macro_allowed, 0);
        assert!(r
            .errors
            .iter()
            .any(|e| e.contains("macro 呼び出しの外に出して書き直す")));
    }

    #[test]
    fn macro_mul_add_in_tests_dir_passes_with_an_allowlist_entry() {
        let oracle = "alice-lol/tests/oracle.rs";
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    a.mul_add(b, c)\n}\nfn g(a: f32, b: f32, c: f32) {\n    println!(\"{}\", a.mul_add(b, c));\n}\n";
        let r = run(
            &[(oracle, src)],
            Some(&allow(oracle, "println!(\\\"{}\\\", a.mul_add(b, c));", 1)),
        );
        assert_eq!(r.errors, Vec::<String>::new());
        assert_eq!(r.macro_hits, 1);
        assert_eq!(r.macro_allowed, 1);
    }

    #[test]
    fn neg_method_call_is_caught() {
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    a.mul_add(b, c).neg()\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn unary_neg_of_a_single_statement_block_is_caught() {
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    -{ a.mul_add(b, c) }\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn subtraction_from_negative_zero_is_caught() {
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    -0.0 - a.mul_add(b, c)\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn subtraction_from_positive_zero_is_not_flagged() {
        // a genuinely different (incidentally safer) operation, not negation
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    0.0 - a.mul_add(b, c)\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 0);
    }

    #[test]
    fn multiplication_by_negative_one_is_caught_either_operand_order() {
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    a.mul_add(b, c) * -1.0\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
        let src2 = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    -1.0 * a.mul_add(b, c)\n}\n";
        let r2 = run(&[(SRC, src2)], None);
        assert_eq!(r2.violations, 1);
    }

    #[test]
    fn multiplication_by_a_different_constant_is_not_flagged() {
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    a.mul_add(b, c) * 2.0\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 0);
    }

    #[test]
    fn let_binding_then_unary_negate_in_the_same_block_is_caught() {
        let src =
            "fn f(a: f32, b: f32, c: f32) -> f32 {\n    let t = a.mul_add(b, c);\n    -t\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn let_binding_then_neg_method_in_the_same_block_is_caught() {
        let src =
            "fn f(a: f32, b: f32, c: f32) -> f32 {\n    let t = a.mul_add(b, c);\n    t.neg()\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn let_binding_then_sub_from_neg_zero_in_the_same_block_is_caught() {
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    let t = a.mul_add(b, c);\n    -0.0 - t\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn let_binding_then_mul_by_neg_one_in_the_same_block_is_caught() {
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    let t = a.mul_add(b, c);\n    t * -1.0\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn let_binding_used_without_negation_is_not_flagged() {
        let src =
            "fn f(a: f32, b: f32, c: f32) -> f32 {\n    let t = a.mul_add(b, c);\n    t + 1.0\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 0);
    }

    #[test]
    fn let_binding_shadowed_before_any_negation_is_not_flagged() {
        // the second `let t` rebinds the name to something else; a later
        // `-t` negates the NEW binding, not the mul_add one, so this must
        // not be flagged (and is a case this checker deliberately bails out
        // on conservatively rather than risk tracking it wrong)
        let src = "fn f(a: f32, b: f32, c: f32, d: f32) -> f32 {\n    let t = a.mul_add(b, c);\n    let t = d;\n    -t\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 0);
    }

    #[test]
    fn let_binding_reassigned_before_any_negation_is_not_flagged() {
        let src = "fn f(a: f32, b: f32, c: f32, d: f32) -> f32 {\n    let mut t = a.mul_add(b, c);\n    t = d;\n    -t\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 0);
    }

    #[test]
    fn let_binding_negated_as_a_buried_subexpression_is_flagged() {
        // `-t` appearing deep inside a larger, syntactically unrelated
        // expression (here `(-t) * 0.5`) still constructs the same
        // fneg(fma(...)) IR subexpression LLVM's nsz-gated canonicalisation
        // matches, regardless of what the enclosing expression goes on to
        // compute with the negated value — this must be flagged, not waved
        // through because the statement as a whole "computes something new"
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    let t = a.mul_add(b, c);\n    -t * 0.5\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn let_binding_negated_inside_a_return_expression_is_caught() {
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    let t = a.mul_add(b, c);\n    return -t;\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn let_binding_negated_as_a_function_call_argument_is_caught() {
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    let t = a.mul_add(b, c);\n    f32::max(-t, 0.0)\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn let_binding_negated_in_an_array_literal_element_is_caught() {
        let src = "fn f(a: f32, b: f32, c: f32) -> [f32; 2] {\n    let t = a.mul_add(b, c);\n    [t, -t]\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn let_binding_negated_in_a_struct_literal_field_is_caught() {
        let src = "struct S { x: f32 }\nfn f(a: f32, b: f32, c: f32) -> S {\n    let t = a.mul_add(b, c);\n    S { x: -t }\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn let_binding_negated_in_an_index_assignment_rhs_is_caught() {
        let src = "fn f(a: f32, b: f32, c: f32, v: &mut [f32; 2]) {\n    let t = a.mul_add(b, c);\n    v[0] = -t;\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn let_binding_negated_in_one_if_branch_is_caught() {
        let src = "fn f(a: f32, b: f32, c: f32, cond: bool) -> f32 {\n    let t = a.mul_add(b, c);\n    if cond {\n        -t\n    } else {\n        t\n    }\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn let_binding_shadowed_in_one_if_branch_does_not_suppress_the_other_branch() {
        // the `else` branch rebinds `t` to an unrelated value and negates
        // THAT — must not be flagged — but the `if` branch still negates the
        // original mul_add-bound `t` and must be. Each branch's shadow
        // tracking must be independent: the rebind in one arm must not leak
        // into, or be bled out of, the sibling arm.
        let src = "fn f(a: f32, b: f32, c: f32, d: f32, cond: bool) -> f32 {\n    let t = a.mul_add(b, c);\n    if cond {\n        -t\n    } else {\n        let t = d;\n        -t\n    }\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn let_binding_negated_in_a_different_function_is_not_flagged() {
        // deliberately out of scope: this checker only looks within the same
        // block, not across function boundaries
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    a.mul_add(b, c)\n}\nfn g(x: f32) -> f32 {\n    -x\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 0);
    }

    #[test]
    fn multi_line_receiver_chain_is_caught() {
        let src = "struct S { y: f32 }\nfn f(x: S, b: f32, c: f32) -> f32 {\n    -x\n        .y\n        .mul_add(b, c)\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn every_workspace_member_is_scanned_by_default() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("Cargo.toml"),
            "[workspace]\nmembers = [\"a\", \"b-crate\"]\nresolver = \"2\"\n",
        )
        .unwrap();
        let dirs = workspace_src_dirs(dir.path());
        assert_eq!(dirs.len(), 8);
        assert!(dirs.contains(&"b-crate/src".to_string()));
        assert!(dirs.contains(&"a/benches".to_string()));
        // a violation in a member that is not alice-lol is found
        let path = dir.path().join("b-crate/src/lib.rs");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            "fn f(a: f32, b: f32, c: f32) -> f32 {\n    -a.mul_add(b, c)\n}\n",
        )
        .unwrap();
        let r = check(dir.path(), &dirs);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn unreadable_members_give_no_dirs() {
        let dir = tempfile::tempdir().unwrap();
        assert!(workspace_src_dirs(dir.path()).is_empty());
        // neither a workspace nor a package
        fs::write(
            dir.path().join("Cargo.toml"),
            "[profile.release]\nopt-level = 3\n",
        )
        .unwrap();
        assert!(workspace_src_dirs(dir.path()).is_empty());
        // a glob member is not expanded: refuse rather than scan less
        fs::write(
            dir.path().join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/*\"]\n",
        )
        .unwrap();
        assert!(workspace_src_dirs(dir.path()).is_empty());
    }

    #[test]
    fn a_single_crate_repository_scans_its_root_package() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"x\"\nversion = \"0.1.0\"\n",
        )
        .unwrap();
        let dirs = workspace_src_dirs(dir.path());
        assert_eq!(dirs, vec!["src", "tests", "examples", "benches"]);
        fs::create_dir_all(dir.path().join("src")).unwrap();
        fs::write(
            dir.path().join("src/lib.rs"),
            "fn f(a: f32, b: f32, c: f32) -> f32 {\n    -a.mul_add(b, c)\n}\n",
        )
        .unwrap();
        let r = check(dir.path(), &dirs);
        assert_eq!(r.violations, 1);
        assert_eq!(r.src_fma_calls, 1);
    }

    #[test]
    fn every_fma_family_callee_is_counted_by_kind() {
        let src = "fn f(a: f32, b: f32, c: f32, v: V) {\n    let _ = a.mul_add(b, c);\n    let _ = f32::mul_add(a, b, c);\n    let _ = libm::fma(1.0, 2.0, 3.0);\n    let _ = libm::fmaf(a, b, c);\n    let _ = core::intrinsics::fmaf32(a, b, c);\n    let _ = vfmaq_f32(v, v, v);\n    let _ = vfmsq_laneq_f32::<1>(v, v, v);\n    let _ = _mm256_fmadd_ps(v, v, v);\n    let _ = _mm_fnmsub_sd(v, v, v);\n    let _ = _mm512_mask_fmaddsub_pd(v, 1, v, v);\n    let _ = a.max(b);\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(
            (r.mul_add_calls, r.scalar_fma_calls, r.simd_fma_calls),
            (2, 3, 5)
        );
        assert_eq!(r.fma_calls, 10);
    }

    #[test]
    fn a_negated_libm_fma_is_caught() {
        for call in [
            "libm::fma(a, b, c)",
            "fmaf(a, b, c)",
            "core::intrinsics::fmaf64(a, b, c)",
        ] {
            let src = format!("fn f(a: f64, b: f64, c: f64) -> f64 {{\n    -{call}\n}}\n");
            let r = run(&[(SRC, &src)], None);
            assert_eq!(r.violations, 1, "{call}");
        }
    }

    #[test]
    fn a_wide_vector_receiver_is_caught_by_the_method_name() {
        let src = "use wide::f32x8;\nfn f(v: f32x8, a: f32x8, b: f32x8) -> f32x8 {\n    -v.mul_add(a, b)\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.violations, 1);
    }

    #[test]
    fn renaming_a_fma_family_item_on_import_fails_even_with_an_allowlist_entry() {
        let src = "use libm::fma as fused;\nfn f(a: f64, b: f64, c: f64) -> f64 {\n    -fused(a, b, c)\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.aliases, 1);
        assert!(
            r.errors.iter().any(|e| e.contains("別名で import")),
            "{:?}",
            r.errors
        );
        let nested = "use core::arch::aarch64::{vdupq_n_f32, vfmaq_f32 as f};\n";
        assert_eq!(run(&[(SRC, nested)], None).aliases, 1);
        let oracle = "alice-lol/tests/oracle.rs";
        let r = run(
            &[(oracle, src)],
            Some(&allow(oracle, "use libm::fma as fused;", 1)),
        );
        assert!(
            r.errors
                .iter()
                .any(|e| e.contains("改名 import は allowlist 不可")),
            "{:?}",
            r.errors
        );
    }

    #[test]
    fn importing_a_fma_family_item_under_its_own_name_is_fine() {
        let src = "use libm::fma;\nuse core::arch::aarch64::vfmaq_f32 as vfmaq_f32;\nfn f(a: f64, b: f64, c: f64) -> f64 {\n    fma(a, b, -c)\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.aliases, 0);
        assert_eq!(r.errors, Vec::<String>::new());
    }

    #[test]
    fn a_fma_name_inside_a_string_literal_of_a_macro_is_not_a_mention() {
        let src = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    assert!(a > 0.0, \"the fma and mul_add forms differ\");\n    a.mul_add(b, c)\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.macro_hits, 0);
    }

    #[test]
    fn a_simd_fma_name_in_a_macro_is_a_mention() {
        let src = "fn f(v: V) {\n    println!(\"{:?}\", vfmaq_f32(v, v, v));\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.macro_hits, 1);
    }

    #[test]
    fn a_name_bound_to_negative_zero_and_to_something_else_is_not_followed() {
        let followed = "fn f(a: f32, b: f32, c: f32) -> f32 {\n    let z = -0.0f32;\n    z - a.mul_add(b, c)\n}\n";
        assert_eq!(run(&[(SRC, followed)], None).violations, 1);
        let reused = "fn g() -> f32 {\n    let z = 3.0;\n    z\n}\nfn f(a: f32, b: f32, c: f32) -> f32 {\n    let z = -0.0f32;\n    z - a.mul_add(b, c)\n}\n";
        assert_eq!(run(&[(SRC, reused)], None).violations, 0);
    }

    #[test]
    fn a_fma_family_path_used_as_a_value_is_refused() {
        let src = "fn f(v: &[(f32, f32, f32)]) -> Vec<f32> {\n    v.iter().map(|&(a, b, c)| a).map(f32::sqrt).collect()\n}\nfn g(a: f32, b: f32, c: f32) -> f32 {\n    [a].iter().map(|&x| x).fold(c, |acc, x| f32::mul_add(x, b, acc))\n}\nfn h() -> fn(f64, f64, f64) -> f64 {\n    libm::fma\n}\n";
        let r = run(&[(SRC, src)], None);
        assert_eq!(r.aliases, 1, "{:?}", r.errors);
        assert_eq!(r.scalar_fma_calls + r.mul_add_calls, 1);
    }

    /// Every probe of `tools/fneg-fma-ir-probes` (shared with the IR check's
    /// tests) with its `expect-syn` verdict: `flag` must be found, `clean` must
    /// not, `refused` must fail as a renamed or value-used fma, and `miss` (a
    /// shape beyond a syntax tree: a closure, a wrapper function, a tuple field)
    /// must stay unfound so the documented limit of this pre-check is pinned; the
    /// IR check covers those.
    #[test]
    fn every_shared_probe_has_its_expected_syntax_tree_verdict() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../fneg-fma-ir-probes");
        let mut probes: Vec<PathBuf> = fs::read_dir(&dir)
            .expect("tools/fneg-fma-ir-probes")
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "rs"))
            .collect();
        probes.sort();
        assert!(probes.len() >= 40, "only {} probes", probes.len());
        let mut wrong = Vec::new();
        for probe in &probes {
            let text = fs::read_to_string(probe).unwrap();
            let expect = text
                .lines()
                .find_map(|l| l.strip_prefix("// expect-syn: "))
                .expect("expect-syn line")
                .trim()
                .to_string();
            let r = run(&[(SRC, &text)], None);
            let found = r.violations > 0;
            let ok = match expect.as_str() {
                "flag" => found,
                "clean" | "miss" => !found && r.aliases == 0,
                "refused" => r.aliases > 0,
                other => panic!("{}: unknown expect-syn {other}", probe.display()),
            };
            if !ok {
                wrong.push(format!(
                    "{}: expect {expect}, violations {}",
                    probe.display(),
                    r.violations
                ));
            }
            assert!(
                r.fma_calls > 0
                    || expect == "refused"
                    || probe.to_string_lossy().contains("no_fma"),
                "{}: no fma call counted",
                probe.display()
            );
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    #[test]
    fn src_dirs_can_be_overridden_for_other_repos() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("src/lib.rs");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, CLEAN).unwrap();
        let default_dirs: Vec<String> = DEFAULT_SRC_DIRS.iter().map(|s| s.to_string()).collect();
        let r_default = check(dir.path(), &default_dirs);
        assert_eq!(r_default.files, 0);
        let r_override = check(dir.path(), &["src".to_string()]);
        assert_eq!(r_override.errors, Vec::<String>::new());
        assert_eq!(r_override.files, 1);
    }

    /// Mutation teeth (2026-10-04 discipline: break the fix, see red):
    /// if the Paren/Group-peeling in `peel()` is removed (i.e. only the
    /// bare-identifier unparenthesised branch survives, or vice versa), one
    /// of `parenthesised_outer_negation_is_caught` /
    /// `bare_unparenthesised_outer_negation_is_caught` goes red because each
    /// exercises a different code path through `is_fma_result`. Verified
    /// by hand: commenting out the `Expr::Paren`/`Expr::Group` arms of
    /// `peel()` turns `parenthesised_outer_negation_is_caught` red while
    /// `bare_unparenthesised_outer_negation_is_caught` stays green, and vice
    /// versa for commenting out the direct `Expr::MethodCall`/`Expr::Call`
    /// match arms in `is_fma_result`.
    #[test]
    fn mutation_teeth_marker_both_branches_have_dedicated_tests() {
        // this test exists only as a signpost to the doc comment above;
        // the actual teeth are the two tests it names.
    }

    /// Mutation teeth for the 5 additionally-caught shapes (2026-10-13):
    /// verified by hand, each neutered independently then reverted.
    /// - Neutering the `.neg()` check in `visit_expr_method_call` turns
    ///   `neg_method_call_is_caught` red while
    ///   `let_binding_then_neg_method_in_the_same_block_is_caught` stays
    ///   green (it goes through the separate `is_negation_of_ident` path,
    ///   not `visit_expr_method_call`) — proving the two are independent
    ///   teeth, not one check exercised twice.
    /// - Neutering `record_binary` (the block/`-0.0 -`/`* -1.0` forms) turns
    ///   `subtraction_from_negative_zero_is_caught` and
    ///   `multiplication_by_negative_one_is_caught_either_operand_order` red,
    ///   while their Form-F (`let`-then-negate) counterparts stay green for
    ///   the same reason.
    /// - Neutering `record_let_then_negate` entirely (Form F) turns all 4
    ///   `let_binding_then_*_in_the_same_block_is_caught` tests red, as well
    ///   as every `let_binding_negated_*_is_caught` buried-position test
    ///   below (return expression, call argument, array/struct literal
    ///   field, index-assignment RHS, `if` branch, buried subexpression).
    #[test]
    fn mutation_teeth_marker_five_new_shapes_have_dedicated_tests() {
        // signpost only; see the doc comment above.
    }

    /// Correction (2026-10-13, independent review of this file's Form F
    /// scan): the first landing of Form F detection used a recursive,
    /// `syn::visit::Visit`-based scan (an early version of what is now
    /// [`IdentNegationFinder`]) that flagged 18 real sites across LOL
    /// source. Those 18 were misdiagnosed as false positives on the theory
    /// that a negation buried inside a larger, syntactically unrelated
    /// expression (`(-t) * 0.5`) "isn't really undoing the mul_add result" —
    /// and the recursive scan was replaced with a narrower check that only
    /// looked at a statement's own top-level expression, with
    /// `let_binding_negated_as_a_buried_subexpression_is_not_flagged` landed
    /// as a regression fixture codifying that (wrong) call.
    ///
    /// That theory does not hold: LLVM's `fneg(fma(x,y,z)) ->
    /// fma(x,-y,-z)` canonicalisation (the module doc above) matches the
    /// `fneg`-over-`fma` IR subexpression pattern wherever it occurs — it
    /// has no notion of "the enclosing statement computes something else
    /// afterward". All 18 original findings were genuine. This file now
    /// restores the recursive scan ([`IdentNegationFinder`]) and the
    /// fixture above has been corrected to assert `violations == 1`
    /// (renamed `..._is_flagged`), alongside new tests for the other
    /// syntactic positions a buried negation can occupy.
    #[test]
    fn mutation_teeth_marker_form_f_is_a_full_recursive_same_function_scan() {
        // signpost only; see the doc comment above and `IdentNegationFinder`.
    }
}
