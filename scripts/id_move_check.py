#!/usr/bin/env python3
"""An identifier moves only with a breaking version and naming both sides (ALICE-Law G9).

The identifiers of scripts/id_move.toml (`SEMANTICS_ID`, `*_SEMANTICS_ID`: constants a
stored value is stamped with) are read as values, not as text: most are folded by a
`const fn`, so the source can stay the same while the value moves (a resolved upstream
crate changed) and change while the value stays. For the base ref and for the working
tree, a probe binary that depends on the crate by path prints each identifier and each
previous-identifier list; the two are then compared.

The base is the newest release tag reachable from HEAD that matches a pattern of the
config (`v{version}`, `<crate>-v{version}`, ...), so an identifier moved twice between two
releases counts once, against what was published. `--base <ref>` overrides it.

For each identifier that exists at the base and has another value now:

- when the base's value is in the identifier's previous list, the move claims the old bits
  are reproduced (not a change of meaning) and needs no version bump, but the claim is tested
  by behaviour: the [[id]] names `previous-tests` (`<path>.rs::<fn>`, each a `#[test]` fn, not
  `#[ignore]`d), and the tool runs each of them twice in a copy of the working tree next to it:
  as it is (it must pass, one test run) and with the old value taken out of the previous list's
  definition (located by its rustdoc span; the value written as `hex32("…")` / a hex string, as
  32 byte literals, or as `[0xNN; 32]`) (it must fail). A test that passes without the old value
  does not depend on it, and naming none fails, so the list cannot be filled to dodge a bump;
- otherwise (not in the list, or no list) the move is a change of meaning and the version must
  be a breaking step from the base's (a major for 1.x and later, a minor or a major for 0.x);
- in both cases CHANGELOG.md must name both values (their first 8 hex digits).

An identifier missing at the base is new and passes. Comparing no identifier fails, and
so does a config naming a constant the working tree does not have.

Completeness: every `const` / `static` item of the crate whose type resolves to `[u8; 32]`
(the form every identifier of these crates takes; any visibility, through `&`, a type alias,
a length written as `0x20`, `32usize` or a constant) must be an [[id]] of the config or be
listed under [exempt] with a reason, so a second identifier (a format id next to
SEMANTICS_ID) cannot be left out. The items come from rustdoc's JSON output (RUSTC_BOOTSTRAP=1,
--document-private-items, format_version 60, as scripts/api_boundary.py reads it), where the
compiler has resolved the types: a text scan misses a type on the next line or behind an
alias. A public re-export of another crate's constant or static counts too (rustdoc does not
give its type), so a crate that stamps an upstream identifier lists it; a public glob re-export of
another crate (`pub use other::*`) is listed by its source (`other::*`) under [exempt] with a reason. Associated consts count too (an inherent impl's, a trait's and a trait impl's), named
`Type::NAME` / `Trait::NAME`. rustdoc runs with every feature (`--all-features`, or the
config's `doc-features` when a feature cannot build here), so an identifier behind a feature
is seen; the values are still read with the config's `features`. An [exempt] entry that names
no such item fails, and finding none fails as having compared nothing. A `const` inside a
function body is not an item rustdoc lists, and an item behind a target cfg (`target_os`) is
seen only on that target.

The base's identifiers are read from the version as published (crates.io, `={version}`),
which is what a stored value was stamped by; a version never published is read from the tag's
tree instead (`--base-from tag` forces that, `--base-from registry` refuses the fallback).

A registry that cannot be read fails the run on the default branch, on tags and locally, and
is a warning on ci/ and wip/ branches and pull requests (an outage does not turn every branch
red; the default branch still does not pass a check it did not run).

usage: scripts/id_move_check.py [--root DIR] [--base REF] [--base-from auto|registry|tag] [--print-base]
config: scripts/id_move.toml
    package = "alice-zip"                 # the cargo package
    manifest = "Cargo.toml"               # its manifest, relative to the root
    features = ["std"]                    # features the probe builds it with (optional)
    default-features = true               # (optional)
    tags = ["v{version}"]                 # release tag patterns
    doc-features = ["std", "lzma"]        # rustdoc's features for completeness (default: all)
    probe-dependencies = ["alice-det-math"]   # crates a path below names besides the package (the
                                          # probe takes the version the package resolves)
    [exempt]                              # [u8; 32] constants that are not identifiers
    SOME_KEY = "why it is not an identifier"
    [[id]]
    path = "alice_zip::law::SEMANTICS_ID"
    previous = "alice_zip::semantics::PREVIOUS_SEMANTICS_IDS"   # optional
    previous-tests = ["tests/golden.rs::previous_ids_reproduce"]  # optional, see above
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG = "scripts/id_move.toml"
VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$")
MISSING_RE = re.compile(r"error\[E0(?:425|432|433)\]|cannot find|unresolved import|could not find")


class CheckError(Exception):
    pass


@dataclass(frozen=True)
class IdSpec:
    path: str
    previous: str | None
    previous_tests: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    package: str
    manifest: str
    features: tuple[str, ...]
    default_features: bool
    tags: tuple[str, ...]
    ids: tuple[IdSpec, ...]
    exempt: tuple[tuple[str, str], ...] = ()
    doc_features: tuple[str, ...] | None = None
    probe_deps: tuple[str, ...] = ()


MIN_REASON = 12
SUPPORTED_FORMATS = {60}


def rustdoc_json(crate_dir: Path, cfg: Config) -> dict:
    env = dict(os.environ, RUSTC_BOOTSTRAP="1", RUSTDOCFLAGS="")
    target = Path(os.environ.get("ID_MOVE_TARGET_DIR") or tempfile.mkdtemp(prefix="id-move-doc-")) / "rustdoc"
    try:
        cmd = ["cargo", "rustdoc", "-q", "--lib", "--manifest-path", str(crate_dir / "Cargo.toml"),
               "--target-dir", str(target)]
        # every feature: an identifier behind one must be listed too
        if cfg.doc_features is None:
            cmd.append("--all-features")
        elif cfg.doc_features:
            cmd += ["--features", ",".join(cfg.doc_features)]
        cmd += ["--", "-Z", "unstable-options", "--output-format", "json",
                "--document-private-items", "--document-hidden-items"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise CheckError(f"rustdoc failed in {crate_dir}: {r.stderr.strip()[-800:]}")
        lib = cfg.package.replace("-", "_")
        doc = target / "doc" / f"{lib}.json"
        if not doc.exists():
            docs = list((target / "doc").glob("*.json"))
            if len(docs) != 1:
                raise CheckError(f"rustdoc wrote no JSON for {cfg.package} ({[d.name for d in docs]})")
            doc = docs[0]
        data = json.loads(doc.read_text(encoding="utf-8"))
        doc.unlink()
    finally:
        if not os.environ.get("ID_MOVE_TARGET_DIR"):
            shutil.rmtree(target.parent, ignore_errors=True)
    if data.get("format_version") not in SUPPORTED_FORMATS:
        raise CheckError(f"rustdoc JSON format_version {data.get('format_version')} is not one of {sorted(SUPPORTED_FORMATS)}")
    return data


def is_id_type(ty, index: dict, depth: int = 0) -> bool:
    """the type resolves to [u8; 32] (through &, type aliases); a length that is not a number
    counts as a hit, so an identifier written with a computed length is not skipped"""
    if not isinstance(ty, dict) or depth > 16:
        return False
    if "borrowed_ref" in ty:
        return is_id_type(ty["borrowed_ref"].get("type"), index, depth + 1)
    if "array" in ty:
        arr = ty["array"]
        if arr.get("type") != {"primitive": "u8"}:
            return False
        length = str(arr.get("len", "")).strip()
        try:
            return int(re.sub(r"(?:usize|u\d+)$", "", length).replace("_", ""), 0) == 32
        except ValueError:
            return True
    if "resolved_path" in ty:
        target = index.get(str(ty["resolved_path"].get("id")))
        inner = (target or {}).get("inner", {})
        if "type_alias" in inner:
            return is_id_type(inner["type_alias"].get("type"), index, depth + 1)
    return False


def id_constants_from_doc(doc: dict) -> dict[str, list[str]]:
    """{name: [file:line, ...]} of every const / static item of the documented crate whose type
    resolves to [u8; 32]"""
    index = doc.get("index", {})
    root_crate = index.get(str(doc.get("root")), {}).get("crate_id", 0)
    # the owner of an associated const: the impl's self type, or the trait
    owner: dict[str, str] = {}
    for item in index.values():
        inner = item.get("inner", {})
        if "impl" in inner:
            for_ty = inner["impl"].get("for") or {}
            name = (for_ty.get("resolved_path") or {}).get("path", "").rsplit("::", 1)[-1] or "?"
            for child in inner["impl"].get("items", []):
                owner[str(child)] = name
        elif "trait" in inner:
            for child in inner["trait"].get("items", []):
                owner[str(child)] = item.get("name") or "?"
    out: dict[str, list[str]] = {}
    for key, item in index.items():
        if item.get("crate_id") != root_crate or not item.get("name"):
            continue
        inner = item.get("inner", {})
        kind = next((k for k in ("constant", "static", "assoc_const") if k in inner), None)
        if kind is None or not is_id_type(inner[kind].get("type"), index):
            continue
        name = item["name"] if kind != "assoc_const" else f"{owner.get(str(key), '?')}::{item['name']}"
        span = item.get("span") or {}
        where = f"{span.get('filename', '?')}:{(span.get('begin') or ['?'])[0]}"
        out.setdefault(name, []).append(where)
    # a public re-export of another crate's constant or static (`pub use alice_core::law::SEMANTICS_ID`):
    # the JSON carries the target's kind but not its type, so every such re-export counts
    paths = doc.get("paths", {})
    for item in index.values():
        use = item.get("inner", {}).get("use")
        if item.get("crate_id") != root_crate or not use or item.get("visibility") != "public":
            continue
        target = paths.get(str(use.get("id")), {})
        if use.get("is_glob"):
            # `pub use other_crate::*` (or a module of it): whatever [u8; 32] it holds becomes public
            # here unseen, so the glob itself must be listed under [exempt] with a reason
            if target.get("crate_id") != root_crate:
                span = item.get("span") or {}
                where = f"{span.get('filename', '?')}:{(span.get('begin') or ['?'])[0]} (glob re-export)"
                out.setdefault(f"{use.get('source') or '?'}::*", []).append(where)
            continue
        if target.get("crate_id") == root_crate or target.get("kind") not in ("constant", "static"):
            continue
        span = item.get("span") or {}
        where = f"{span.get('filename', '?')}:{(span.get('begin') or ['?'])[0]} (re-export of {'::'.join(target.get('path', []))})"
        out.setdefault(use.get("name") or "?", []).append(where)
    return {k: sorted(v) for k, v in out.items()}


_DOC: dict[str, dict] = {}


def id_constants(crate_dir: Path, cfg: Config, root: Path | None = None) -> dict[str, list[str]]:
    doc = rustdoc_json(crate_dir, cfg)
    if root is not None:
        _DOC[str(root)] = doc
    return id_constants_from_doc(doc)


def completeness(cfg: Config, found: dict[str, list[str]]) -> list[str]:
    problems = []
    if not found:
        return ["no const / static item of type [u8; 32] found in the crate (compared nothing)"]
    # a constant by its name, an associated const by Type::NAME (the last two segments)
    listed = {spec.path.rsplit("::", 1)[1] for spec in cfg.ids} | {"::".join(spec.path.split("::")[-2:]) for spec in cfg.ids}
    exempt = dict(cfg.exempt)
    for name, where in sorted(found.items()):
        if name not in listed and name not in exempt:
            problems.append(f"{name} ({', '.join(where)}): a [u8; 32] item that is neither an [[id]] of "
                            f"{CONFIG} nor under [exempt] with a reason")
    for name, reason in sorted(exempt.items()):
        if name not in found:
            problems.append(f"[exempt] {name}: names no const / static item of type [u8; 32] (stale)")
        if len(reason.strip()) < MIN_REASON:
            problems.append(f"[exempt] {name}: reason shorter than {MIN_REASON} characters")
        if name in listed:
            problems.append(f"[exempt] {name}: also an [[id]]")
    return problems


def load_config(root: Path) -> Config:
    p = root / CONFIG
    if not p.exists():
        raise CheckError(f"{CONFIG} is missing")
    doc = tomllib.loads(p.read_text(encoding="utf-8"))
    ids = []
    for e in doc.get("id", []):
        path = e.get("path")
        if not isinstance(path, str) or not re.fullmatch(r"[A-Za-z_]\w*(::[A-Za-z_]\w*)+", path):
            raise CheckError(f"{CONFIG}: [[id]] path {path!r} is not a Rust path")
        prev = e.get("previous")
        if prev is not None and (not isinstance(prev, str) or not re.fullmatch(r"[A-Za-z_]\w*(::[A-Za-z_]\w*)+", prev)):
            raise CheckError(f"{CONFIG}: [[id]] previous {prev!r} is not a Rust path")
        ptests = e.get("previous-tests", [])
        if not (isinstance(ptests, list) and all(isinstance(t, str) and re.fullmatch(r"[\w./-]+\.rs::\w+", t) for t in ptests)):
            raise CheckError(f"{CONFIG}: [[id]] previous-tests must be a list of `<path>.rs::<fn>`")
        if ptests and prev is None:
            raise CheckError(f"{CONFIG}: [[id]] {path}: previous-tests without a previous list")
        ids.append(IdSpec(path, prev, tuple(ptests)))
    tags = doc.get("tags", [])
    if not ids:
        raise CheckError(f"{CONFIG}: no [[id]] (compared nothing)")
    if not tags or not all(isinstance(t, str) and "{version}" in t for t in tags):
        raise CheckError(f"{CONFIG}: tags must be patterns containing {{version}}")
    for key in ("package", "manifest"):
        if not isinstance(doc.get(key), str) or not doc[key]:
            raise CheckError(f"{CONFIG}: {key} is missing")
    exempt = doc.get("exempt", {})
    if not isinstance(exempt, dict) or not all(isinstance(v, str) for v in exempt.values()):
        raise CheckError(f"{CONFIG}: [exempt] must map names to reasons")
    doc_features = doc.get("doc-features")
    if doc_features is not None and not (isinstance(doc_features, list) and all(isinstance(f, str) for f in doc_features)):
        raise CheckError(f"{CONFIG}: doc-features must be a list of feature names")
    probe_deps = doc.get("probe-dependencies", [])
    if not (isinstance(probe_deps, list) and all(isinstance(d, str) and re.fullmatch(r"[A-Za-z0-9_-]+", d) for d in probe_deps)):
        raise CheckError(f"{CONFIG}: probe-dependencies must be a list of crate names")
    return Config(doc["package"], doc["manifest"], tuple(doc.get("features", [])),
                  bool(doc.get("default-features", True)), tuple(tags), tuple(ids), tuple(sorted(exempt.items())),
                  None if doc_features is None else tuple(doc_features), tuple(probe_deps))


def parse_version(v: str) -> tuple[int, int, int]:
    m = VERSION_RE.match(v)
    if not m:
        raise CheckError(f"version {v!r} is not semver")
    return int(m[1]), int(m[2]), int(m[3])


def is_breaking(old: str, new: str) -> bool:
    """`new` is a semver-breaking step from `old`: a major, or a minor while the major is 0."""
    a, b = parse_version(old), parse_version(new)
    if b[0] != a[0]:
        return b[0] > a[0]
    if a[0] == 0:
        return b[1] > a[1]
    return False


def next_breaking(v: str) -> str:
    """the smallest breaking step from `v`: the next minor in 0.x, the next major from 1.0"""
    a = parse_version(v)
    return f"0.{a[1] + 1}.0" if a[0] == 0 else f"{a[0] + 1}.0.0"


def package_version(manifest_text: str, package: str) -> str:
    doc = tomllib.loads(manifest_text)
    pkg = doc.get("package", {})
    if pkg.get("name") != package:
        raise CheckError(f"manifest names package {pkg.get('name')!r}, the config {package!r}")
    v = pkg.get("version")
    if isinstance(v, dict):
        raise CheckError("version.workspace = true is not read; give the member's version")
    if not isinstance(v, str):
        raise CheckError("the manifest has no version")
    return v


def git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise CheckError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def release_base(root: Path, cfg: Config) -> tuple[str, str]:
    """(tag, version) of the newest release tag reachable from HEAD matching a pattern."""
    candidates = []
    for tag in git(root, "tag", "--merged", "HEAD").split():
        for pat in cfg.tags:
            pre, post = pat.split("{version}", 1)
            if tag.startswith(pre) and tag.endswith(post):
                v = tag[len(pre):len(tag) - len(post) if post else None]
                if VERSION_RE.match(v):
                    candidates.append((parse_version(v), tag, v))
    # newest first; a tag counts only when the manifest at it is this package at that
    # version (a repository can carry tags of another package or of an earlier language)
    for _, tag, v in sorted(candidates, reverse=True):
        r = subprocess.run(["git", "-C", str(root), "show", f"{tag}:{cfg.manifest}"], capture_output=True, text=True)
        if r.returncode != 0:
            continue
        try:
            if package_version(r.stdout, cfg.package) == v:
                return tag, v
        except (CheckError, tomllib.TOMLDecodeError):
            continue
    raise CheckError(f"no release tag reachable from HEAD matches {list(cfg.tags)} with {cfg.package} "
                     f"at that version in {cfg.manifest} (nothing published to compare with; pass --base to name one)")


def probe_source(cfg: Config, spec: IdSpec, with_previous: bool = True) -> str:
    lines = ["fn hex(b: &[u8; 32]) -> String { b.iter().map(|x| format!(\"{:02x}\", x)).collect() }",
             "fn main() {",
             f"    let id: [u8; 32] = {spec.path};",
             "    print!(\"{{\\\"id\\\": \\\"{}\\\", \\\"previous\\\": \", hex(&id));"]
    if spec.previous and with_previous:
        lines += [f"    let prev: &[[u8; 32]] = {spec.previous};",
                  "    let p: Vec<String> = prev.iter().map(hex).collect();",
                  "    println!(\"[{}]}}\", p.iter().map(|h| format!(\"\\\"{}\\\"\", h)).collect::<Vec<_>>().join(\", \"));"]
    else:
        lines.append("    println!(\"null}}\");")
    lines.append("}")
    return "\n".join(lines) + "\n"


class NotPublished(CheckError):
    pass


class RegistryUnreachable(CheckError):
    pass


UNREACHABLE_RE = re.compile(r"failed to (?:download|get|query|update)|spurious network error|Couldn't resolve host|"
                            r"Could not resolve host|timed out|SSL|failed to fetch|network failure|HTTP status (?:5\d\d|429)")


def unreadable_base_policy(env: dict[str, str]) -> str:
    """"fail" or "warn" when the base cannot be read (the registry is unreachable): a registry
    outage must not turn every branch red, but the default branch and release tags must not
    pass a check they did not run. Local runs (no GITHUB_REF) fail."""
    ref = env.get("GITHUB_REF", "")
    if env.get("GITHUB_EVENT_NAME") in ("pull_request", "pull_request_target"):
        return "warn"
    if ref.startswith("refs/heads/ci/") or ref.startswith("refs/heads/wip/"):
        return "warn"
    return "fail"


def read_ids(source: Path | str, cfg: Config, target_dir: Path, with_previous: bool = True) -> dict[str, dict | None]:
    """{path: {"id": hex, "previous": [hex] | None} or None when the crate does not have it}, of the
    crate in the tree `source` (a Path) or of the published version `source` (a str) from the
    registry; the previous lists are read only where they are judged (the working tree): the base
    may predate them"""
    out: dict[str, dict | None] = {}
    for i, spec in enumerate(cfg.ids):
        probe = Path(tempfile.mkdtemp(prefix="id-move-probe-"))
        try:
            (probe / "src").mkdir()
            if isinstance(source, Path):
                dep = {"path": str((source / cfg.manifest).resolve().parent)}
            else:
                dep = {"version": f"={source}"}
            dep["default-features"] = cfg.default_features
            if cfg.features:
                dep["features"] = list(cfg.features)
            dep_toml = "{ " + ", ".join(f"{k} = {json.dumps(v)}" for k, v in dep.items()) + " }"
            (probe / "Cargo.toml").write_text(
                f"[package]\nname = \"id-move-probe-{i}\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n"
                f"[dependencies]\n{json.dumps(cfg.package)} = {dep_toml}\n"
                # "*" unifies with the package's own requirement, so the probe reads the same version
                + "".join(f"{json.dumps(d)} = {{ version = \"*\", default-features = false }}\n" for d in cfg.probe_deps)
                + "\n[workspace]\n", encoding="utf-8")
            (probe / "src" / "main.rs").write_text(probe_source(cfg, spec, with_previous), encoding="utf-8")
            env = dict(os.environ, CARGO_TARGET_DIR=str(target_dir), RUSTFLAGS="")
            env.pop("RUSTC_WRAPPER", None)
            r = subprocess.run(["cargo", "run", "--quiet", "--manifest-path", str(probe / "Cargo.toml")],
                               capture_output=True, text=True, env=env)
            if r.returncode != 0 and isinstance(source, str) and re.search(
                    r"failed to select a version|no matching package named|could not find .* in registry", r.stderr):
                raise NotPublished(f"{cfg.package} {source} is not in the registry")
            if r.returncode != 0 and isinstance(source, str) and UNREACHABLE_RE.search(r.stderr):
                raise RegistryUnreachable(f"cannot read the base {cfg.package} {source} from the registry: "
                                          f"{r.stderr.strip()[-400:]}")
            if r.returncode != 0:
                # absent only when the error is the probe's path not resolving (a segment of
                # it named in a resolution error); any other build failure is an error, so a
                # broken base cannot pass as "the identifier is new"
                segs = spec.path.split("::")[1:]
                errs = [ln for ln in r.stderr.splitlines() if MISSING_RE.search(ln)]
                if errs and "id-move-probe" in r.stderr and any(
                        re.search(rf"`{re.escape(s)}`", ln) for ln in errs for s in segs):
                    out[spec.path] = None
                    continue
                raise CheckError(f"the probe for {spec.path} did not run against {source}: {r.stderr.strip()[-800:]}")
            line = r.stdout.strip().splitlines()[-1]
            out[spec.path] = json.loads(line)
        finally:
            shutil.rmtree(probe, ignore_errors=True)
    return out


def _test_fn_problem(crate_dir: Path, ref: str, list_name: str) -> str | None:
    rel, fn = ref.split("::")
    f = crate_dir / rel
    if not f.exists():
        return f"{rel} does not exist"
    text = f.read_text(encoding="utf-8")
    m = re.search(r"((?:[ \t]*#\[[^\n]*\]\s*\n)*)[ \t]*(?:pub\s+)?fn\s+" + re.escape(fn) + r"\s*\(", text)
    if not m:
        return f"no fn {fn} in {rel}"
    if "#[test]" not in m.group(1):
        return f"fn {fn} has no #[test]"
    if re.search(r"#\[\s*ignore", m.group(1)):
        return f"fn {fn} is #[ignore]d"
    return None


def remove_value(source: str, old_hex: str) -> str | None:
    """`source` (a previous list's definition) with the value `old_hex` changed so it no longer
    matches, or None when the value is not written there in a form this tool reads"""
    if old_hex in source.lower():
        i = source.lower().index(old_hex)
        last = source[i + 63]
        flipped = "0" if last.lower() != "0" else "1"
        return source[:i + 63] + flipped + source[i + 64:]
    byte_list = [old_hex[k:k + 2] for k in range(0, 64, 2)]
    rx = r"\s*,\s*".join(rf"0[xX]{b}(?:_?u8)?" for b in byte_list)
    m = re.search(rx, source, re.I)
    if m:
        tail = re.search(r"0[xX][0-9a-fA-F]{2}(?:_?u8)?$", m.group(0))
        flipped = "0x00" if byte_list[-1] != "00" else "0x01"
        return source[:m.start() + tail.start()] + flipped + source[m.end():]
    if len(set(byte_list)) == 1:
        m = re.search(rf"\[\s*0[xX]{byte_list[0]}(?:_?u8)?\s*;\s*32\s*\]", source, re.I)
        if m:
            flipped = "0x00" if byte_list[0] != "00" else "0x01"
            return source[:m.start()] + f"[{flipped}; 32]" + source[m.end():]
    return None


def run_named_test(crate_dir: Path, ref: str, target_dir: Path) -> tuple[bool, int, str]:
    """(passed, tests run, output tail) of one named test in the crate at `crate_dir`"""
    rel, fn = ref.split("::")
    sel = ["--test", Path(rel).stem] if rel.startswith("tests/") else ["--lib"]
    env = dict(os.environ, CARGO_TARGET_DIR=str(target_dir))
    r = subprocess.run(["cargo", "test", "--quiet", "--manifest-path", str(crate_dir / "Cargo.toml"), "--all-features",
                        *sel, fn, "--", "--exact"], capture_output=True, text=True, env=env)
    ran = sum(int(x) for x in re.findall(r"test result: \w+\. (\d+) passed", r.stdout)) + \
        sum(int(x) for x in re.findall(r"(\d+) failed", r.stdout))
    return r.returncode == 0, ran, (r.stdout + r.stderr).strip()[-400:]


class PreviousVerifier:
    """runs each [[id]]'s previous-tests against a copy of the working tree with the old value
    taken out of the previous list"""

    def __init__(self, root: Path, cfg: Config, doc: dict | None, target_dir: Path):
        self.root, self.cfg, self.doc, self.target = root, cfg, doc, target_dir

    def _list_span(self, spec: IdSpec) -> tuple[str, int, int] | None:
        name = (spec.previous or "").rsplit("::", 1)[-1]
        for item in (self.doc or {}).get("index", {}).values():
            if item.get("name") == name and "constant" in item.get("inner", {}) and item.get("span"):
                sp = item["span"]
                return sp["filename"], sp["begin"][0], sp["end"][0]
        return None

    def __call__(self, spec: IdSpec, old: str) -> tuple[list[str], list[str]]:
        problems, passed_refs = [], []
        if not spec.previous_tests:
            return [], [f"{spec.path}: the move keeps {old[:8]} in {spec.previous} but the [[id]] names no "
                        f"previous-tests: name the tests that reproduce the old generation, or treat the move as a "
                        f"change of meaning (breaking version, value out of the list)"]
        crate_dir = (self.root / self.cfg.manifest).resolve().parent
        list_name = spec.previous.rsplit("::", 1)[-1]
        for ref in spec.previous_tests:
            why = _test_fn_problem(crate_dir, ref, list_name)
            if why:
                problems.append(f"{spec.path}: previous-tests {ref}: {why}")
        if problems:
            return [], problems
        span = self._list_span(spec)
        if span is None:
            return [], [f"{spec.path}: {spec.previous} has no source span in the rustdoc JSON, so the old value cannot be taken out of it"]
        rel, begin, end = span
        # the copy sits next to the repository root so sibling path dependencies resolve
        top = self.root.resolve()
        copy = top.parent / f".{top.name}.id-move-previous-{os.getpid()}"
        try:
            files = subprocess.run(["git", "-C", str(top), "ls-files", "-co", "--exclude-standard"],
                                   capture_output=True, text=True, check=True).stdout.split("\n")
            for f in files:
                if not f or f.startswith("target/"):
                    continue
                src = top / f
                if src.is_file():
                    dst = copy / f
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    # new mtimes: a build left in the target dir from an earlier (changed) copy
                    # must not look newer than these sources
                    shutil.copy(src, dst)
            copy_crate = copy / crate_dir.relative_to(top)
            for ref in spec.previous_tests:
                ok, ran, out = run_named_test(copy_crate, ref, self.target)
                if not ok or ran < 1:
                    problems.append(f"{spec.path}: previous-tests {ref} does not pass as it is ({ran} run): {out}")
            if problems:
                return [], problems
            path = copy_crate / rel
            lines = path.read_text(encoding="utf-8").split("\n")
            block = "\n".join(lines[begin - 1:end])
            changed = remove_value(block, old)
            if changed is None:
                return [], [f"{spec.path}: {old[:8]} is not written in {rel}:{begin}-{end} ({list_name}) as a hex string, "
                            f"32 byte literals or [0xNN; 32], so the tool cannot take it out to test the claim"]
            lines[begin - 1:end] = changed.split("\n")
            path.write_text("\n".join(lines), encoding="utf-8")
            for ref in spec.previous_tests:
                ok, ran, out = run_named_test(copy_crate, ref, self.target)
                if ok:
                    problems.append(f"{spec.path}: previous-tests {ref} still passes with {old[:8]} taken out of "
                                    f"{list_name}: it does not depend on the old value")
                else:
                    passed_refs.append(ref)
        finally:
            shutil.rmtree(copy, ignore_errors=True)
        return passed_refs, problems


def judge(base_version: str, head_version: str, base: dict[str, dict | None],
          head: dict[str, dict | None], changelog: str, cfg: Config,
          verify_previous=None) -> tuple[list[str], list[str]]:
    """(problems, lines of what was compared); `verify_previous(spec, old)` returns (tests that
    passed with the old value and failed without it, problems)"""
    problems, report = [], []
    for spec in cfg.ids:
        now, then = head.get(spec.path), base.get(spec.path)
        if now is None:
            problems.append(f"{spec.path}: not found in the working tree ({CONFIG} names it)")
            continue
        if then is None:
            report.append(f"{spec.path}: new since the base ({now['id'][:8]})")
            continue
        if now["id"] == then["id"]:
            report.append(f"{spec.path}: unchanged ({now['id'][:8]})")
            continue
        old, new = then["id"], now["id"]
        in_previous = bool(spec.previous) and old in (now.get("previous") or [])
        if in_previous:
            # same bits claimed: no bump, but the old generation must still be tested
            if verify_previous is None:
                refs, why = [], [f"{spec.path}: the move keeps {old[:8]} in {spec.previous}, and nothing tested the claim"]
            else:
                refs, why = verify_previous(spec, old)
            problems.extend(why)
            report.append(f"{spec.path}: moved {old[:8]} -> {new[:8]}, the old value kept in {spec.previous}"
                          + (f" and exercised by {', '.join(refs)}" if refs else ""))
            if not refs and not why:
                problems.append(f"{spec.path}: moved {old[:8]} -> {new[:8]} keeping {old[:8]} in {spec.previous}, "
                                f"but no previous-tests entry showed it depends on the old value")
        else:
            report.append(f"{spec.path}: moved {old[:8]} -> {new[:8]}")
        if not in_previous and not is_breaking(base_version, head_version):
            fix = next_breaking(base_version)
            if head_version == base_version:
                problems.append(f"{spec.path}: moved {old[:8]} -> {new[:8]} at {head_version}, the version of the "
                                f"release it is compared with: bump the version first (set version = \"{fix}\" in "
                                f"{cfg.manifest}) and name {old[:8]} and {new[:8]} in CHANGELOG.md")
            else:
                problems.append(f"{spec.path}: moved {old[:8]} -> {new[:8]} but the version {base_version} -> "
                                f"{head_version} is not a breaking step (a major, or a minor in 0.x): set "
                                f"version = \"{fix}\" in {cfg.manifest}")
        missing = [h[:8] for h in (old, new) if h[:8] not in changelog]
        if missing:
            problems.append(f"{spec.path}: moved, and CHANGELOG.md does not name {', '.join(missing)}")
    return problems, report


def check(root: Path, base_ref: str | None, base_from: str = "auto") -> tuple[list[str], list[str]]:
    cfg = load_config(root)
    missing = completeness(cfg, id_constants((root / cfg.manifest).resolve().parent, cfg, root))
    problems, report = check_values(root, base_ref, base_from)
    return missing + problems, report


def check_values(root: Path, base_ref: str | None, base_from: str = "auto") -> tuple[list[str], list[str]]:
    """`base_from`: "registry" reads the base version as published (what G9 is about: a path
    dependency of the tag's tree on a sibling repository would resolve against today's sibling,
    not the released one), "tag" builds the tag's tree, "auto" the registry and the tag when the
    version was never published"""
    cfg = load_config(root)
    head_version = package_version((root / cfg.manifest).read_text(encoding="utf-8"), cfg.package)
    if base_ref is None:
        base_ref, _ = release_base(root, cfg)
    base_version = package_version(git(root, "show", f"{base_ref}:{cfg.manifest}"), cfg.package)
    changelog_path = (root / cfg.manifest).parent / "CHANGELOG.md"
    if not changelog_path.exists():
        changelog_path = root / "CHANGELOG.md"
    changelog = changelog_path.read_text(encoding="utf-8") if changelog_path.exists() else ""
    work = Path(tempfile.mkdtemp(prefix="id-move-"))
    target = Path(os.environ.get("ID_MOVE_TARGET_DIR") or (work / "target"))
    source = base_ref
    if base_from in ("registry", "auto"):
        try:
            base = read_ids(base_version, cfg, target, with_previous=False)
            head = read_ids(root, cfg, target)
            source = f"{cfg.package} {base_version} from the registry"
        except NotPublished:
            if base_from == "registry":
                shutil.rmtree(work, ignore_errors=True)
                raise
            base = None
    else:
        base = None
    if base is not None:
        try:
            problems, report = judge(base_version, head_version, base, head, changelog, cfg,
                                    PreviousVerifier(root, cfg, _DOC.get(str(root)), target))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        report.insert(0, f"base {base_ref}: {source}, working tree {head_version}")
        return problems, report
    # the base tree sits next to the root, at the same depth, so a path dependency on a
    # sibling repository (`../ALICE-SDF`) resolves from it as it does from the root
    wt = root.resolve().parent / f".{root.resolve().name}.id-move-base-{os.getpid()}"
    try:
        # every file of the tag, written through a throwaway index: the repository's own index and
        # checkout are not touched, nothing is registered, and unlike `git archive` no file marked
        # export-ignore in .gitattributes is left out of the build
        wt.mkdir()
        env = dict(os.environ, GIT_INDEX_FILE=str(work / "base.index"))
        for args in (["read-tree", base_ref], ["checkout-index", "--all", "--force", f"--prefix={wt}/"]):
            r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, env=env)
            if r.returncode != 0:
                raise CheckError(f"git {args[0]} {base_ref}: {r.stderr.strip()}")
        base = read_ids(wt, cfg, target, with_previous=False)
        shutil.rmtree(wt, ignore_errors=True)
        head = read_ids(root, cfg, target)
        problems, report = judge(base_version, head_version, base, head, changelog, cfg,
                                PreviousVerifier(root, cfg, _DOC.get(str(root)), target))
    finally:
        shutil.rmtree(wt, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
    report.insert(0, f"base {base_ref}: the tag's tree ({base_version}), working tree {head_version}")
    return problems, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--base")
    ap.add_argument("--base-from", choices=["auto", "registry", "tag"], default="auto")
    ap.add_argument("--print-base", action="store_true",
                    help="print <package>-<version> of the base release (a cache key for its .crate) and exit")
    a = ap.parse_args(argv)
    try:
        if a.print_base:
            cfg = load_config(Path(a.root))
            version = release_base(Path(a.root), cfg)[1] if a.base is None else package_version(
                git(Path(a.root), "show", f"{a.base}:{cfg.manifest}"), cfg.package)
            print(f"{cfg.package}-{version}")
            return 0
        problems, report = check(Path(a.root), a.base, a.base_from)
    except RegistryUnreachable as e:
        if unreadable_base_policy(dict(os.environ)) == "warn":
            print(f"id_move_check: WARNING (not checked on this ref): {e}", file=sys.stderr)
            return 0
        print(f"id_move_check: {e}", file=sys.stderr)
        return 1
    except CheckError as e:
        print(f"id_move_check: {e}", file=sys.stderr)
        return 1
    for line in report:
        print(f"id_move_check: {line}")
    for p in problems:
        print(f"id_move_check: {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"id_move_check: {len(report) - 1} identifier(s) compared, 0 problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
