# alice-router — Design Specification (Draft)

**Status**: Spec only. Crate does not exist yet. This document exists to
lock the design surface before any Rust code is written, per
`comprehensive-spec-templates` skill guidance.

**Version**: 0.0 (pre-crate, spec-first)
**Author**: Moroya Sakamoto
**Last update**: 2026-07-21
**Related crates**: ALICE-LLM (v1.1.0+, engine); ALICE-Verify (proposed, TBD).

---

## §1. Vision, Scope, Non-scope

### Vision

`alice-router` is a **model orchestration layer** that sits above one or
more inference backends and decides, per request, which backend answers,
whether the answer is trustworthy, and how the outcome is cached and
observed. It complements ALICE-LLM (single-model inference engine) by
turning a collection of engines and external APIs into a **single
addressable "meta-model"**.

The design is influenced by Sakana AI's "multi-agent system as a model"
framing (Sakana Fugu, 2026-07) but rebuilt from first principles for
Rust, edge deployment, and explicit cost accounting.

### Scope

- Route a single prompt/context to one or more backends
- Enforce a verification policy on backend outputs
- Track per-backend cost, latency, and quality signals
- Cache prompt segments and expert selections where the underlying
  backend supports it (e.g. Kimi K3 cached input at $0.30/1M)
- Provide a stable Rust trait surface downstream crates can implement
  and swap backends without touching the router core
- Support both **synchronous** and **streaming** request modes

### Non-scope

- **Training or fine-tuning models** (that lives in ALICE-Train)
- **Model weight quantization** (ALICE-LLM concern)
- **Prompt engineering libraries** (out of scope; users bring their own)
- **UI or REPL frontends** (a possible sibling crate, not this one)
- **Guaranteed byte-exact reproducibility across backends** (physically
  impossible when backends include external stochastic APIs; a strict
  deterministic sub-mode is a Phase R.5 stretch goal)
- **Multi-agent tool-use planning** (LangGraph territory; router
  provides the substrate but does not embed the planner)

---

## §2. Positioning

| Project | Layer | Openness | Rust-native | Verification built-in | Edge target | Cost accounting |
|---|---|---|---|---|---|---|
| **alice-router** (this) | Orchestrator | Open (planned AGPL-3.0) | ✓ | ✓ (Phase R.3) | ✓ | ✓ per-backend |
| Sakana Fugu | Orchestrator + verifier | Closed | ✗ (Python API) | ✓ | ✗ (cloud only) | ✗ (bundled pricing) |
| LangChain / LangGraph | Orchestrator + planner + tool-use | Open | ✗ (Python primary) | Partial (via chains) | ✗ | Partial |
| LiteLLM | Backend abstraction (proxy) | Open | ✗ (Python) | ✗ | ✗ | ✓ per-backend |
| OpenRouter | Hosted routing service | Closed (hosted) | ✗ (HTTP) | ✗ | ✗ (SaaS) | ✓ |

**Where alice-router differs from all of the above**: it is a *library*
(no runtime dependency on Python, no hosted control plane), it treats
verification as a first-class layer (not a chain node), and it can bind
to ALICE-LLM engines running on the same host with zero network hop.

---

## §3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client                               │
└──────────────────────────┬──────────────────────────────────┘
                           │ Request { messages, hints, budget }
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  alice-router core                          │
│                                                             │
│   ┌───────────────┐  ┌───────────────┐  ┌─────────────┐   │
│   │ RoutePolicy   │→ │  Backend[s]   │→ │  Verifier   │   │
│   │  (§6)         │  │  (§8)         │  │  (§7)       │   │
│   └───────────────┘  └───────────────┘  └─────────────┘   │
│                                                             │
│   ┌─────────────────────┐   ┌─────────────────────────┐   │
│   │  Cache (§9)         │   │  Observability (§10)    │   │
│   └─────────────────────┘   └─────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │ Response { text, votes, cost, trace }
                           ▼
                        Client
```

Key invariants:

- **Router core is transport-agnostic** (no `tokio`/`std::net`
  imports; async runtime is caller-supplied via an executor trait).
- **Backends implement a single `Backend` trait** — swapping ALICE-LLM
  Bonsai for an HTTP OpenAI proxy is a config change, not a code change.
- **Verification runs in a bounded time budget** and can be turned off
  per-request for low-latency paths.

---

## §4. Data model

```rust
/// A single conversational turn.
pub struct Message {
    pub role: Role,              // System / User / Assistant / Tool
    pub content: Content,        // Text | MultiModal (image / audio bytes)
    pub name: Option<String>,    // Optional attribution for multi-agent
}

/// The full request the router acts on.
pub struct Request {
    pub messages: Vec<Message>,
    pub hints: RouteHints,       // Explicit preferences: {min_context, mode: "code"|"reasoning"|...}
    pub budget: Budget,          // Max cost + max latency
    pub stream: bool,
    pub verify_policy: Option<VerifyPolicy>, // None => use default policy
}

/// The router's decision on how to fulfil a Request.
pub struct RouteDecision {
    pub primary: BackendId,
    pub verifiers: Vec<BackendId>,      // may be empty; §7
    pub cache_key: Option<CacheKey>,
    pub reason: RouteReason,            // trace: why this backend?
}

/// A binding between a logical name (e.g. "reasoning") and a concrete backend.
pub struct ModelBinding {
    pub id: BackendId,
    pub kind: BackendKind,              // AliceLLM | Http { provider } | Local
    pub cost: CostModel,                // per-token input/output rates + cache tier
    pub capabilities: Capabilities,     // context_len, modalities, tool_use, ...
}

/// The verifier's judgement on a candidate response.
pub struct VerifyResult {
    pub verdict: Verdict,               // Accept | AcceptWithConcerns | Reject
    pub confidence: f32,                // in [0.0, 1.0]
    pub notes: Vec<String>,             // human-readable rationale
    pub cost: CostTally,
}

/// User-facing response after routing + (optional) verification.
pub struct Response {
    pub text: String,                   // or streamed via ResponseStream
    pub route: RouteDecision,
    pub verify: Option<VerifyResult>,
    pub cost_total: CostTally,
    pub trace_id: TraceId,
}
```

Design notes:

- `Content` is a sum type so a future multimodal Kimi K3 or Gemini call
  can carry image/audio bytes without a schema change.
- `Budget` is enforced at every backend boundary — a router that
  ignores `Budget::max_cost_usd` is a bug, not a feature.
- `TraceId` is required, not optional — traceability is a §10 invariant.

---

## §5. API surface (Rust traits)

```rust
/// Entry point.
pub trait Router: Send + Sync {
    async fn respond(&self, req: Request) -> Result<Response, RouteError>;
    fn stream(&self, req: Request) -> ResponseStream;    // if req.stream
    fn bindings(&self) -> &[ModelBinding];
    fn configure(&mut self, cfg: RouteConfig) -> Result<(), ConfigError>;
}

/// Decide which backend(s) handle a Request.
pub trait RoutePolicy: Send + Sync {
    fn decide(&self, req: &Request, bindings: &[ModelBinding]) -> RouteDecision;
}

/// Verify a candidate response against the request.
pub trait Verifier: Send + Sync {
    async fn verify(
        &self,
        req: &Request,
        candidate: &str,
        peers: &[String],           // other backends' answers, if any
    ) -> VerifyResult;
}

/// Actual model runtime (ALICE-LLM engine, HTTP API, local, ...).
pub trait Backend: Send + Sync {
    fn id(&self) -> BackendId;
    fn capabilities(&self) -> Capabilities;
    async fn generate(&self, req: BackendRequest) -> Result<BackendResponse, BackendError>;
    async fn stream(&self, req: BackendRequest) -> BackendStream;
    fn cost_of(&self, tokens_in: u32, tokens_out: u32) -> CostTally;
}
```

Error taxonomy (top-level `RouteError`):

- `NoBackendMatchesHints(RouteHints)`
- `AllBackendsFailed(Vec<BackendError>)`
- `BudgetExceeded { spent: CostTally, cap: Budget }`
- `VerifyDivergence { candidates: Vec<Candidate>, verdicts: Vec<VerifyResult> }`
- `CacheCorrupt(CacheKey)`
- `ConfigError(ConfigError)`

The router core NEVER swallows a `BackendError` silently — either it is
retried against another backend, or it surfaces as a `RouteError`. Per
CLAUDE.md "薄っぺらい対応禁止" rule (fallback loops that hide the root
cause are banned).

---

## §6. Routing strategies

Four routing policies ship in the initial spec:

1. **`FixedPolicy`** — always pick a single backend. Baseline, no
   decision logic. Useful for tests and single-backend deployments.
2. **`HeuristicPolicy`** — dispatch by `RouteHints::mode` and a set of
   TOML-configured rules (regex over prompt, token count, modality).
   Fast (<1 ms decision) and explainable.
3. **`CostAwarePolicy`** — score `(quality_prior × cache_hit_prob) /
   (cost_per_token × expected_tokens)` and pick argmax within
   `Budget::max_cost_usd`. Requires a `CostModel` per binding.
4. **`EnsemblePolicy`** — send the request to N backends in parallel;
   verifier chooses the winner. Highest quality, highest cost. Wraps
   any of the above three as the "primary decision" and adds
   `verifiers`.

Extension point: `RoutePolicy` is a trait, so users can plug an
ML-based policy (embed the prompt, run kNN over a labelled history) as
Phase R.5.

**Deterministic sub-mode** (Phase R.5 stretch): `FixedPolicy` +
`FixedSeed` + non-streaming backend = byte-exact reproducibility. Any
policy with a stochastic branch (parallel `EnsemblePolicy`) forfeits
this guarantee and must document it.

---

## §7. Verification layer

Verification answers: **does the candidate response deserve to reach
the client?** Three modes:

1. **`SelfVerify`** — feed `(request, candidate)` back to the same
   backend with a verifier prompt. Cheapest, weakest signal (backends
   are bad at judging their own output for correctness — see the LLM
   self-eval literature).
2. **`CrossCheck { peer: BackendId }`** — ask a second, independent
   backend to judge. Stronger signal, higher cost. Preferred for
   safety-critical routing (e.g. `mode = "medical"` in hints).
3. **`MajorityVote { peers: Vec<BackendId> }`** — poll N backends,
   emit the modal answer + confidence based on agreement rate. The
   "Sakana Fugu-style verification" reimplemented in Rust. Requires
   `EnsemblePolicy` for the underlying dispatch.

Guardrails:

- Verifier calls are subject to the same `Budget` — a runaway verifier
  cannot bankrupt the request.
- A `Verdict::Reject` triggers a documented policy: `FailFast` (return
  `RouteError::VerifyDivergence`), `RetryWithPrimary`, or
  `ReturnWithWarning`. Configurable per route.
- Verifier prompts are NOT hard-coded here — the crate ships a default
  template but users can inject their own via config.

---

## §8. Backend integration

Two backend kinds ship in v0.1:

### `AliceLLMBackend`

Wraps `alice_llm::Llama3Model` (or a future `KimiK3Model`). Binds
in-process, zero network overhead. Loads a GGUF, exposes `generate` /
`stream`. Selection between Bonsai / Qwen 3.5 / DeepSeek V2-Lite / Kimi
K3 is done at `ModelBinding` config time.

```toml
[[backend]]
id = "kimi-k3-local"
kind = "alice_llm"
model_path = "/models/Kimi-K3-Q4_K_M.gguf"
device = "hybrid"    # cpu, gpu, hybrid, hybrid-per-layer
```

### `HttpBackend`

HTTP APIs: Anthropic, OpenAI, Google Gemini, Moonshot Kimi API. One
struct with pluggable request/response adapters. Rate limiting,
retry-with-backoff, and cost accounting live here.

```toml
[[backend]]
id = "kimi-k3-api"
kind = "http"
provider = "moonshot"
base_url = "https://api.moonshot.cn/v1"
api_key_env = "MOONSHOT_API_KEY"    # never hard-coded (§13)
cost_in_cache_miss = 3.00           # $/1M tokens
cost_in_cache_hit = 0.30
cost_out = 15.00
```

Extension point: `Backend` is a trait, users add their own for
Bedrock, Cohere, in-house services, etc. `router-contrib-*` crate
namespace reserved for community adapters.

---

## §9. Caching

Two orthogonal cache layers:

1. **Prompt cache** — a hash of the leading N messages maps to a
   backend's cached-input tier. For Kimi K3 API, this drops input from
   $3.00/1M to $0.30/1M (10× cheaper). For ALICE-LLM local, it maps to
   a prewarmed KV cache slice.
2. **Route decision cache** — memoise `(hint_signature, budget_class)
   → BackendId` for a bounded LRU window. Skips policy computation for
   hot paths. Invalidated on `configure()`.

Neither cache is on by default in v0.1 — they are opt-in via
`RouteConfig::caching`. Explicit-in, no surprise costs.

---

## §10. Observability

Non-negotiable requirements:

- **Every request emits a `TraceId`** (ULID). Downstream logs and
  metrics MUST carry it.
- **Structured logging** via `tracing`. Backend calls, verify calls,
  cache hits/misses all emit spans.
- **Metrics** (via `metrics` crate): counters (`router_requests_total`,
  `router_backend_calls_total{backend}`, `router_verify_verdicts_total{verdict}`),
  histograms (`router_latency_seconds{stage}`, `router_cost_usd{backend}`).
- **Cost accounting** is a first-class trace attribute, not a
  best-effort log line. Every backend call attaches
  `CostTally { tokens_in, tokens_out, cache_tier, usd }`.

No default exporter — users wire `tracing-subscriber` /
`metrics-exporter-prometheus` themselves. Router core stays lean.

---

## §11. Configuration

TOML schema (illustrative):

```toml
[router]
default_verify = "self_verify"
default_policy = "heuristic"

[[backend]]
id = "bonsai-27b"
kind = "alice_llm"
model_path = "/models/Bonsai-27B-Q4_K_M.gguf"
capabilities = { context_len = 32768, modalities = ["text"], tool_use = false }

[[backend]]
id = "kimi-k3-api"
kind = "http"
provider = "moonshot"
api_key_env = "MOONSHOT_API_KEY"
capabilities = { context_len = 1048576, modalities = ["text","image","video"] }

[[policy.heuristic.rule]]
match_hints = { mode = "code" }
prefer = ["kimi-k3-api"]

[[policy.heuristic.rule]]
match_regex = "(?i)proof|theorem|integrate"
prefer = ["bonsai-27b"]

[verifier.default]
kind = "cross_check"
peer = "kimi-k3-api"

[cache]
prompt = true
decision = true
prompt_ttl_seconds = 900
decision_ttl_seconds = 60
```

Validation:

- Every `backend.id` must be unique.
- `capabilities.context_len` must be ≥ observed prompt tokens or the
  binding is skipped in policy.
- `api_key_env` MUST resolve to a non-empty env var at
  `Router::configure` time — fail-fast, not fail-later.

---

## §12. Error handling

Per §5 taxonomy plus these rules:

- **No `unwrap` / `expect` in router core.** Every error path is a
  typed variant of `RouteError`.
- **Backend retries** are bounded (`max_retries: u8`, default 2), and
  each attempt burns budget. The last error is preserved and returned.
- **Verify divergence** does not hide the underlying candidates — the
  error variant carries all of them so downstream code can escalate
  to a human or log for later analysis.
- **Timeout policy**: `Budget::max_latency_ms` is enforced at every
  await point. Backends that don't honour it are cancelled with
  extreme prejudice.

---

## §13. Security

- **API keys**: only via env vars declared in config (`api_key_env`).
  Router MUST refuse to serialise or log any resolved key value.
- **Prompt injection**: the router is transport-neutral and does not
  parse tool-calling directives. Downstream tool-use planners are
  responsible for their own guardrails.
- **Backend isolation**: a compromised or misbehaving backend cannot
  leak state to other backends via the router core — each backend
  gets a fresh `BackendRequest` and no shared mutable context.
- **Audit**: every request/response pair can be optionally persisted
  to an append-only journal (Phase R.4). Off by default.

Reference: CLAUDE.md § "AI-Tencho 情報の隔離ルール" — the same
isolation posture applies: multiple concurrent tenants MUST NOT share
router state without explicit tagging.

---

## §14. Performance targets (v0.1)

| Metric | Target | Measurement |
|---|---|---|
| Route decision time (Heuristic, cached) | <1 ms | criterion bench, Mac M3 |
| Route decision time (Heuristic, uncached) | <5 ms | criterion bench |
| Verifier overhead (SelfVerify, ALICE-LLM local) | <200 ms | end-to-end integration bench |
| Cache hit rate (steady state, HeuristicPolicy) | >70% | 24 h synthetic replay |
| Memory overhead vs raw backend call | <20 MB / router instance | RSS delta measurement |
| Panic rate | 0 / 10^9 requests | fuzzed integration + property test |

These are targets, not contracts — v0.1 will publish a benchmark
harness (`criterion` + `divan`) and users can verify locally.

---

## §15. Testing plan

| Layer | Tool | Coverage focus |
|---|---|---|
| Unit | `#[test]` | Every trait impl, error variant construction, config parse |
| Integration | `tests/*.rs` with `MockBackend` | Policy × Verifier × Cache combinatorics |
| Property | `proptest` | Budget invariant (spent ≤ cap), Verdict monotonicity |
| Fuzz | `cargo fuzz` | Config parser, prompt cache key derivation |
| Benchmark | `criterion` | §14 metrics |
| End-to-end | GitHub Actions | Real ALICE-LLM backend on Bonsai + mock HTTP backend |

**Mock backend requirement** (per CLAUDE.md rule): external APIs
(`Anthropic`, `Moonshot`, `OpenAI`) are never called in unit or
integration tests. Real calls live behind
`--features integration-live` and are opt-in.

---

## §16. Roadmap

| Phase | Scope | Blockers |
|---|---|---|
| **R.0** | Spec approval + crate scaffold (`Cargo.toml`, module skeleton, no `todo!()` outside stub functions) | — (this doc) |
| R.1 | `Router` + `FixedPolicy` + `AliceLLMBackend` + baseline tests. Users can wrap a single ALICE-LLM model behind the router surface. | R.0 |
| R.2 | `HeuristicPolicy` + `HttpBackend` (Moonshot / Anthropic / OpenAI adapters) + `RouteConfig` TOML parser. | R.1 |
| R.3 | `Verifier` layer: `SelfVerify`, `CrossCheck`. Divergence handling. | R.2 |
| R.4 | Cost-aware policy + prompt cache + decision cache + observability (tracing spans, metrics counters). Audit journal (off-default). | R.3 |
| R.5 | `EnsemblePolicy` + `MajorityVote`. ML-based policy (embed similarity kNN). Deterministic sub-mode. | R.4 + a stable embedding backend |
| R.6 | Streaming interleave: cancel in-flight verifiers on early primary completion. | R.5 |
| R.7 | Multi-tenant isolation + resource quotas per tenant. | R.6 + audit journal (R.4) |

Each phase ends with a `karikari-review` §9 gate (fmt / clippy / test /
doc) and a `CHANGELOG.md` entry.

---

## §17. Non-functional requirements

- **Backpressure**: If a backend queue is saturated, router MUST
  reflect that in `RouteError::BackendUnavailable` rather than block
  indefinitely. Callers implement retry logic; router does not
  buffer.
- **Reproducibility**: Non-goal in general. Sub-mode only (§6).
- **Deterministic verification**: When `Verifier` is `SelfVerify` with
  `temperature = 0` and a fixed backend seed, verification is
  reproducible. Documented, not enforced.
- **Cancellation safety**: Every async fn MUST be cancel-safe.
  Backends that hold external resources (HTTP connections, GPU
  buffers) release them on drop.

---

## §18. Prior art

- **Sakana AI Fugu** (2026-07) — "multi-agent system as a model".
  Primary influence on §7. Public architecture is a blackbox; we
  reimplement the *idea* (verifier over ensemble) without reverse
  engineering internals.
- **LangChain / LangGraph** — chain-based orchestration. Rejected for
  Rust ergonomics reasons (chains do not compose well in a
  strongly-typed trait system) and because tool-use planning is out
  of scope.
- **LiteLLM** — backend abstraction proxy. Closest in spirit to §8;
  differs in language and in scope (LiteLLM does not verify).
- **OpenRouter** — hosted routing service. Different deployment model
  (SaaS vs library). We treat OpenRouter as a possible upstream
  `HttpBackend`.
- **Nathan Lambert, "Kimi K3: the open-weights escalation"
  (Interconnects, 2026-07-17)** — the current open-weight landscape
  context that motivates routing over one open + one closed backend
  as a common case.

---

## §19. Open questions

1. **Verifier LLM selection**: for `CrossCheck`, does the peer default
   to "second-cheapest capable backend" or "highest-quality-prior
   backend"? Impacts cost model in §14. Recommend the former for v0.1
   (predictable cost), revisit at R.4.
2. **Cost accounting precision**: sub-cent rounding — round-half-up
   (banker's default) vs floor? Cross-provider comparisons hinge on
   this. Recommend floor + carry to next request for now.
3. **Streaming interleave**: if primary streams and a verifier
   contradicts mid-stream, do we (a) let primary finish and annotate,
   (b) cut the stream, (c) inject an inline correction token stream?
   Decision deferred to R.6 with prototype spike.
4. **Backend identity across upgrades**: `BackendId = "kimi-k3-api"`
   spans model minor versions (e.g. K3 → K3.1). Should the router key
   caches by (BackendId, ModelVersion) tuple? Recommend yes — schema
   change in v0.2.
5. **AGPL vs Apache-2.0 licence**: ALICE-LLM ships AGPL-3.0. Same for
   this crate, or dual-licence to widen adoption? Deferred, user
   judgement call at R.0 commit.

---

## §20. Related ALICE work

| Component | Version | Relation to alice-router |
|---|---|---|
| `alice-llm` (this repo) | v1.1.0 | Primary in-process `Backend` implementation via `AliceLLMBackend` |
| Phase X.4 Kimi K3 (in progress, `docs/KIMI_K3_INTEGRATION.md`) | X.4.a (2026-07-17 skeleton), X.4.b blocked on 2026-07-27 open weight release | Prime router backend candidate once GGUF conversion lands; API path via `HttpBackend` is available today |
| Phase X.3.e.3.27 Bonsai 27B GPU coherent generation | v1.1.0 | Local heavyweight backend (32K context, reasoning-strong) |
| Phase X.3.e.3.30 V2-Lite validation | v1.1.0 | Reference "small, cheap, fast" backend for routing baseline |
| `alice-verify` (proposed, not started) | — | Standalone verifier crate the router optionally delegates to |
| ALICE-CodeTracker | v0.1.0-mvp | Stub tracking discipline applies to alice-router the same as ALICE-LLM |

---

## §21. Glossary

| Term | Definition |
|---|---|
| **Router** | The orchestration layer this crate provides. |
| **Backend** | Anything that turns a `BackendRequest` into a `BackendResponse` (local model, HTTP API, mock). |
| **Policy** | The decision function `Request → RouteDecision`. |
| **Verifier** | A function `(Request, Candidate) → VerifyResult`. |
| **Binding** | A named, typed reference to a Backend with cost and capability metadata. |
| **Budget** | A hard cap on cost and latency for a single Request. |
| **Ensemble** | A RouteDecision that dispatches to ≥2 backends in parallel. |
| **Verdict** | Verifier output: `Accept`, `AcceptWithConcerns`, `Reject`. |
| **Trace ID** | ULID attached to every Request; propagated through logs/metrics. |
| **CostTally** | Structured record of `(tokens_in, tokens_out, cache_tier, usd)` per backend call. |
| **Cache tier** | For HTTP backends, the pricing tier (`cache-miss` vs `cache-hit`); for local, prewarmed vs cold KV. |
| **Cancel-safe** | An async fn that leaves observable state consistent when its future is dropped mid-execution. |

---

**End of spec.** The next document in this thread is the R.0 crate
scaffold PR: `Cargo.toml`, `src/lib.rs` (module declarations only), and
a `CHANGELOG.md` entry. That PR is out of scope for this document.
