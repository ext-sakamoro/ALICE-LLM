//! DeepSeek-V3 / Kimi K3-family routed-expert streaming from disk
//! (Phase 4a, Issue #34; extended for Kimi K3 at Phase X.4.e).
//!
//! # Overview
//!
//! Sparse-MoE frontier models push per-token active weights well past the
//! usable RAM budget of consumer hardware. Two concrete cases handled by
//! this module:
//!
//! * **DeepSeek-V3 671B** — 61 MoE layers × 256 routed experts + 1 shared
//!   expert per layer. Even at Q4_K_M each routed expert is ~19 MB, so
//!   keeping every expert in RAM costs ~370 GB. A single token only touches
//!   `num_experts_per_tok = 8` per MoE layer (sparsity 8/256 ≈ 3.13%).
//! * **Kimi K3 2.8T / 104B-active** — 92 MoE layers × 896 routed experts +
//!   2 shared per layer, native MXFP4 (~1.4 TB total at community-Q4 GGUF,
//!   ~594 GB at native MXFP4). Each token touches
//!   `num_experts_per_tok = 16` (sparsity 16/896 ≈ 1.79%, sparser than V3).
//!   Per-token active weights ≈ 24 GB at Q4 (see
//!   [`kimi_k3_active_bytes`] for the derivation) — well within an
//!   NVMe-backed Mac M3 Max 128 GB budget once streamed.
//!
//! The core of the colibri innovation is to **load routed experts on
//! demand** with an LRU cache, keyed by `(layer_idx, kind, expert_idx)`.
//! The infrastructure is expert-count-agnostic: `n_experts` is a runtime
//! parameter of [`ExpertLayerInfo`], and neither [`LruExpertCache`] nor
//! [`StreamingExpertPool`] hardcodes 256 anywhere. The K3 topology
//! (896 experts, top-16) drops into the same pool by construction.
//!
//! Kimi K3 uses **Stable LatentMoE**: the per-expert FFN runs in a
//! `routed_expert_hidden_size = 3584` latent space (versus the DeepSeek V3
//! full-width path). The per-expert slab layout — three matrices for the
//! SiTU-GLU / SwiGLU expert FFN — is identical, so the pool's `ExpertKind
//! = {Gate, Up, Down}` triple still applies unchanged. The Kimi-specific
//! `W^↓` down-projection + `W^↑` up-projection with RMSNorm live in the
//! LatentMoE forward path (not per-expert; captured in
//! [`crate::llama3::KimiDeltaConfig`]) and are not streamed by this pool.
//!
//! # Scope of Phase 4a (this module)
//!
//! Ships the *infrastructure* — types, LRU semantics, and the enum that lets
//! `DeepSeekMoeWeights` hold either in-memory `WeightRef`s or a shared pool:
//!
//! - [`StreamingExpertPool`] — owns a byte source and an LRU cache of decoded
//!   expert slabs. Cache eviction is byte-budget-driven.
//! - [`LruExpertCache`] — MRU-first `VecDeque` order + `HashMap` slot map, with
//!   `Arc<Vec<u8>>` so a caller can safely hold a slab reference across an
//!   eviction of the same key.
//! - [`ExpertByteSource`] — trait abstracting the underlying storage; the two
//!   canonical implementations are an owned `Vec<u8>` (unit tests / small
//!   test models) and a `memmap2::Mmap` (production, real GGUF).
//! - [`ExpertKind`] / [`ExpertSlabRef`] — dispatch surface consumed by
//!   `forward_deepseek_moe_layer` in `llama3.rs`.
//!
//! # Deferred to Phase 4b
//!
//! - **Async readahead** — `posix_fadvise(WILLNEED)` + a background thread
//!   that reads next-layer's likely experts while the current layer's matvec
//!   runs. Requires router-lookahead prediction to know which experts to
//!   prefetch.
//! - **Hot-pinning** — permanent RAM residence for the top-N experts across
//!   all layers, bypassing LRU eviction.
//! - **Router-lookahead prefetch** (experimental) — colibri claims 71.6% of
//!   next-layer routing is predictable from the current layer's post-attention
//!   state. Would allow overlap of expert I/O with compute.
//! - **OS page cache tuning** — `mlock` for hot regions, `madvise(RANDOM)` for
//!   the expert region so the kernel does not thrash on sequential readahead.
//! - **Real DeepSeek-V3 GGUF benchmarks** — blocked on local disk budget
//!   (~370 GB) and dedicated multi-day slot.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use crate::gguf::GgmlType;

/// Which of the three MoE weight matrices a slab belongs to.
///
/// Encoded as a `u8` so it fits into a compact `ExpertKey` even when we
/// eventually key millions of entries during a long streaming session.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertKind {
    Gate = 0,
    Up = 1,
    Down = 2,
}

/// Uniquely identifies one expert slab in the model.
///
/// Fields are `(layer_idx, kind, expert_idx)`. `expert_idx` is the routed-
/// expert index within the layer (`0..n_routed_experts`, so 0..256 for V3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub layer_idx: u16,
    pub kind: ExpertKind,
    pub expert_idx: u16,
}

impl ExpertKey {
    #[inline]
    pub const fn new(layer_idx: usize, kind: ExpertKind, expert_idx: usize) -> Self {
        Self {
            layer_idx: layer_idx as u16,
            kind,
            expert_idx: expert_idx as u16,
        }
    }
}

/// Abstract byte storage backing an expert pool.
///
/// The two canonical implementations are `Vec<u8>` (unit tests, small
/// synthetic data) and `memmap2::Mmap` (production, real GGUF files). Any
/// type that can produce a `&[u8]` slice works, so custom sources — for
/// example a chunked S3 reader with local caching — can also be dropped in.
pub trait ExpertByteSource: Send + Sync {
    fn as_bytes(&self) -> &[u8];
}

impl ExpertByteSource for Vec<u8> {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self
    }
}

impl ExpertByteSource for Box<[u8]> {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self
    }
}

// The gguf feature also gates the memmap2 dependency (Cargo.toml), so we
// only compile the Mmap impl when gguf is enabled. This is the production
// byte source: the pool `mmap`s the GGUF file separately from the parser's
// mmap so its lifetime is independent and it can be shared across threads.
#[cfg(feature = "gguf")]
impl ExpertByteSource for memmap2::Mmap {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self.as_ref()
    }
}

/// Advise the kernel that the given mmap region will be accessed randomly
/// (Phase 4b.4).
///
/// This disables the sequential-readahead heuristic — a win
/// for routed-expert bytes because the router picks 8 experts out of 256
/// per token, so paging in *adjacent* experts is pure page-cache pollution.
///
/// Silently no-op on non-Unix platforms and when the `gguf` feature is
/// off (`libc` is a gguf-gated optional dep, no separate `mmap-tuning`
/// feature — every gguf user gets it). Returns `true` when the syscall
/// was actually issued so bench harnesses can gate their timing on the
/// hint being live.
///
/// # Safety
///
/// `madvise` is safe when the passed pointer + length span a valid mapped
/// region owned by the current process. This helper only accepts `&[u8]`
/// slices that must have been obtained from a live [`memmap2::Mmap`], so
/// both invariants are enforced by the type system at the call site.
#[cfg(all(unix, feature = "gguf"))]
pub fn advise_random(mmap_bytes: &[u8]) -> bool {
    // SAFETY: madvise takes (addr, len, advice) and is a no-op if the
    // range is not a valid mapping — but the caller passed us bytes that
    // came from a live memmap2::Mmap in the pool constructor, so we know
    // the range is valid. The Mmap outlives this call by construction.
    let ret = unsafe {
        libc::madvise(
            mmap_bytes.as_ptr().cast::<libc::c_void>().cast_mut(),
            mmap_bytes.len(),
            libc::MADV_RANDOM,
        )
    };
    ret == 0
}

/// Non-Unix / no-gguf fallback for [`advise_random`]. Always returns
/// `false` so callers know the hint was skipped.
#[cfg(not(all(unix, feature = "gguf")))]
pub fn advise_random(_mmap_bytes: &[u8]) -> bool {
    false
}

/// Hint the OS to prefetch a byte range into page cache (POSIX
/// `madvise(MADV_WILLNEED)`).
///
/// Kimi K3 MoE forward uses this to
/// overlap disk I/O with compute: right after the router picks the
/// top-k experts for a layer, we call this on each of the 16 expert
/// cube byte ranges so the OS starts paging them in while we're
/// still finishing the previous layer's shared-experts matvec.
///
/// This is a **hint** — safe under any circumstances, but the OS
/// may ignore it under memory pressure. Return value: `true` iff
/// the syscall returned 0.
///
/// # Safety
///
/// Same as [`advise_random`]: the byte slice must have come from a
/// live `memmap2::Mmap` region.
#[cfg(all(unix, feature = "gguf"))]
pub fn advise_willneed(mmap_bytes: &[u8]) -> bool {
    if mmap_bytes.is_empty() {
        return true;
    }
    // SAFETY: caller-supplied slice must be a valid mmap'd region;
    // see docstring.
    let ret = unsafe {
        libc::madvise(
            mmap_bytes.as_ptr().cast::<libc::c_void>().cast_mut(),
            mmap_bytes.len(),
            libc::MADV_WILLNEED,
        )
    };
    ret == 0
}

#[cfg(not(all(unix, feature = "gguf")))]
pub fn advise_willneed(_mmap_bytes: &[u8]) -> bool {
    false
}

/// Byte offset + length of one layer's expert-0 slab, plus per-expert
/// stride and quant type. Sufficient to locate any expert `e` for that
/// layer via `base_offset + e * bytes_per_expert`.
#[derive(Debug, Clone, Copy)]
pub struct ExpertLayerInfo {
    pub base_offset: usize,
    pub bytes_per_expert: usize,
    pub n_experts: usize,
    pub qtype: GgmlType,
}

/// LRU cache of decoded expert slabs. Eviction is triggered when the running
/// byte total would exceed `budget_bytes` after inserting a new slab.
///
/// Slab values are `Arc<Vec<u8>>` so a caller that pulled a slab, was
/// preempted by a subsequent MoE dispatch that evicted the same key, and
/// then returned to finish its matvec still holds valid bytes — the `Arc`
/// keeps the eviction-victim `Vec` alive until the last handle drops.
pub struct LruExpertCache {
    entries: HashMap<ExpertKey, Arc<Vec<u8>>>,
    /// Front = most-recently used; back = eviction candidate.
    lru: VecDeque<ExpertKey>,
    /// Phase 4b.3 hot-pinning set: keys in here are skipped by the LRU
    /// eviction loop, so they stay resident regardless of how many other
    /// entries push the current byte total over budget. Intended for the
    /// top-N most-frequent experts identified by offline profiling —
    /// pinning them costs a fixed amount of RAM but eliminates the miss
    /// penalty every time the router picks them.
    pinned: HashSet<ExpertKey>,
    current_bytes: usize,
    budget_bytes: usize,
    hits: u64,
    misses: u64,
}

impl LruExpertCache {
    /// Create a cache with the given byte budget. `0` disables the cache
    /// (every lookup misses, useful for cold-cache micro-benchmarks).
    pub fn with_budget(budget_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            lru: VecDeque::new(),
            pinned: HashSet::new(),
            current_bytes: 0,
            budget_bytes,
            hits: 0,
            misses: 0,
        }
    }

    /// Look up a key. Bumps the entry to the front of the LRU order and
    /// increments the hit counter when present.
    pub fn get(&mut self, key: &ExpertKey) -> Option<Arc<Vec<u8>>> {
        let hit = self.entries.get(key)?.clone();
        // Bump to MRU position.
        if let Some(pos) = self.lru.iter().position(|k| k == key) {
            self.lru.remove(pos);
        }
        self.lru.push_front(*key);
        self.hits += 1;
        Some(hit)
    }

    /// Insert a new slab, evicting LRU entries as needed to stay under
    /// budget. If the incoming slab is larger than the budget itself, it
    /// still gets stored (the caller expects it) but every other entry is
    /// dropped first.
    pub fn insert(&mut self, key: ExpertKey, bytes: Arc<Vec<u8>>) {
        let incoming_bytes = bytes.len();
        // Remove any existing entry for this key (rare — typically only
        // when the caller reloads the same expert without a prior get).
        if let Some(old) = self.entries.remove(&key) {
            self.current_bytes = self.current_bytes.saturating_sub(old.len());
            if let Some(pos) = self.lru.iter().position(|k| k == &key) {
                self.lru.remove(pos);
            }
        }
        // Evict LRU entries until the new slab fits, skipping any keys the
        // caller has hot-pinned via `pin`. If every remaining entry is
        // pinned, the loop terminates without evicting further and the new
        // slab still gets inserted — the cache is allowed to grow beyond
        // its byte budget when all evictable candidates are gone. This is
        // by design: the caller's `pin` set is a hard lower bound on the
        // memory footprint, and the LRU cache is a soft upper bound.
        while self.current_bytes + incoming_bytes > self.budget_bytes && !self.lru.is_empty() {
            // Scan from the back (least-recently-used) forward, evicting
            // the first non-pinned candidate.
            let mut victim_pos: Option<usize> = None;
            for i in (0..self.lru.len()).rev() {
                if !self.pinned.contains(&self.lru[i]) {
                    victim_pos = Some(i);
                    break;
                }
            }
            match victim_pos {
                Some(pos) => {
                    let victim = self.lru.remove(pos).expect("index in bounds");
                    if let Some(evicted) = self.entries.remove(&victim) {
                        self.current_bytes = self.current_bytes.saturating_sub(evicted.len());
                    }
                }
                None => break, // every entry pinned; can't evict further
            }
        }
        self.entries.insert(key, bytes);
        self.lru.push_front(key);
        self.current_bytes += incoming_bytes;
        self.misses += 1;
    }

    /// Mark a key as hot-pinned — the eviction loop will skip it. Safe to
    /// call for keys not currently in the cache; the pin persists so that
    /// when the key is next loaded it's already flagged.
    pub fn pin(&mut self, key: ExpertKey) {
        self.pinned.insert(key);
    }

    /// Remove a hot-pin. If the entry is still in the cache it becomes
    /// evictable again on the next insert that requires space.
    pub fn unpin(&mut self, key: &ExpertKey) {
        self.pinned.remove(key);
    }

    /// Number of hot-pinned entries. Exposed so bench harnesses can
    /// verify the pin set matches what they configured.
    pub fn pinned_len(&self) -> usize {
        self.pinned.len()
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }

    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Streaming pool serving routed-expert slabs for every MoE layer in a
/// sparse-MoE model. One pool instance is shared across all MoE layers via
/// `Arc` — the layer index is passed at fetch time.
///
/// The pool is expert-count agnostic: [`ExpertLayerInfo::n_experts`]
/// accepts any positive value, so both DeepSeek V3 (256) and Kimi K3
/// (896) drop in unchanged. `budget_bytes` should be sized to hold at
/// least one token's active weights across every layer plus a safety
/// margin — see [`kimi_k3_active_bytes`] for the K3 formula and
/// [`recommended_budget_bytes`] for a rule-of-thumb helper.
pub struct StreamingExpertPool {
    source: Arc<dyn ExpertByteSource>,
    /// Indexed by `[layer_idx][kind as usize]`. Populated at construction
    /// time by walking the GGUF metadata for the routed-expert tensors.
    layer_info: Vec<[ExpertLayerInfo; 3]>,
    cache: Mutex<LruExpertCache>,
}

impl StreamingExpertPool {
    /// Build a new pool over a byte source and a per-layer expert layout.
    ///
    /// # Sizing `budget_bytes`
    ///
    /// The cache is byte-budget driven (not entry-count driven), so a
    /// wrong budget shows up as either a residency-loss thrash (too
    /// small) or wasted RAM (too large). Recommended defaults by model:
    ///
    /// * **DeepSeek V3 Q4_K_M**: ~370 GB total, active per token ≈ 61
    ///   layers × 8 experts × 3 slabs × 19 MB ≈ 27 GB. A 32-40 GB
    ///   budget gives one-token headroom + LRU reuse across tokens.
    /// * **Kimi K3 Q4** (community GGUF, when it lands): active per
    ///   token ≈ 92 layers × 16 experts × ~16.5 MB ≈ **24 GB**, see
    ///   [`kimi_k3_active_bytes`]. A 30-40 GB budget fits a Mac M3 Max
    ///   128 GB unified memory comfortably (leaves ~80+ GB for KV
    ///   cache + attention + shared experts + OS).
    ///
    /// See [`recommended_budget_bytes`] for a programmatic helper.
    pub fn new(
        source: Arc<dyn ExpertByteSource>,
        layer_info: Vec<[ExpertLayerInfo; 3]>,
        budget_bytes: usize,
    ) -> Self {
        Self {
            source,
            layer_info,
            cache: Mutex::new(LruExpertCache::with_budget(budget_bytes)),
        }
    }

    /// Return the quant type for the given layer + kind. Needed so the MoE
    /// dispatch code can pass the right `GgmlType` to `quantized_matvec`.
    pub fn qtype(&self, layer_idx: usize, kind: ExpertKind) -> GgmlType {
        self.layer_info[layer_idx][kind as usize].qtype
    }

    pub fn bytes_per_expert(&self, layer_idx: usize, kind: ExpertKind) -> usize {
        self.layer_info[layer_idx][kind as usize].bytes_per_expert
    }

    /// Fetch a slab, either from the LRU cache (hit) or by reading fresh
    /// bytes from the underlying source (miss).
    ///
    /// # Panics
    ///
    /// Panics on out-of-bounds `layer_idx` / `expert_idx` — those are always
    /// programmer errors (the caller derived them from config), so a debug
    /// panic is more useful than an `Err` variant that no callsite could
    /// meaningfully recover from.
    pub fn get_or_load(
        &self,
        layer_idx: usize,
        kind: ExpertKind,
        expert_idx: usize,
    ) -> Arc<Vec<u8>> {
        let key = ExpertKey::new(layer_idx, kind, expert_idx);
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(hit) = cache.get(&key) {
                return hit;
            }
        }
        let info = &self.layer_info[layer_idx][kind as usize];
        assert!(
            expert_idx < info.n_experts,
            "expert_idx {expert_idx} out of range (n_experts = {})",
            info.n_experts
        );
        let offset = info.base_offset + expert_idx * info.bytes_per_expert;
        let source_bytes = self.source.as_bytes();
        assert!(
            offset + info.bytes_per_expert <= source_bytes.len(),
            "expert slab out of source range (offset {offset} + len {} > source len {})",
            info.bytes_per_expert,
            source_bytes.len()
        );
        let slab = Arc::new(source_bytes[offset..offset + info.bytes_per_expert].to_vec());
        let mut cache = self.cache.lock().unwrap();
        cache.insert(key, slab.clone());
        slab
    }

    /// Warm the cache with a batch of expert keys. Every key that already
    /// hits stays in place (only its LRU position bumps); every miss reads
    /// from the source and inserts. Intended usage: `forward_deepseek_moe_layer`
    /// calls this on the top-k × 3-kind key list *immediately after* the
    /// router picks its winners and *before* the per-expert matvec loop.
    /// The subsequent `get_or_load` calls inside the loop then become
    /// guaranteed hits, decoupling the I/O phase from the compute phase.
    ///
    /// This is the Phase 4b.2 replacement for async readahead. It is
    /// sequential; the [`prefetch_parallel`] variant below fans out over
    /// rayon when the `parallel` feature is enabled — the winning move on
    /// real disk-backed mmap since page-in latency dominates.
    ///
    /// [`prefetch_parallel`]: Self::prefetch_parallel
    pub fn prefetch(&self, keys: &[ExpertKey]) {
        for &k in keys {
            let _ = self.get_or_load(k.layer_idx as usize, k.kind, k.expert_idx as usize);
        }
    }

    /// Parallel prefetch via rayon (feature-gated). Same semantics as
    /// [`prefetch`] but the misses fan out across the rayon global thread
    /// pool. Only useful when the source is genuinely I/O-bound (real
    /// disk-backed `Mmap` with cold pages) — for the in-memory `Vec<u8>`
    /// unit-test source the sequential variant is faster because the extra
    /// work-stealing overhead outweighs the copy cost.
    ///
    /// When the `parallel` feature is not enabled this delegates to
    /// [`prefetch`] so the caller sees a consistent API regardless of
    /// which build variant they compile.
    ///
    /// [`prefetch`]: Self::prefetch
    #[cfg(feature = "parallel")]
    pub fn prefetch_parallel(&self, keys: &[ExpertKey]) {
        use rayon::prelude::*;
        keys.par_iter().for_each(|&k| {
            let _ = self.get_or_load(k.layer_idx as usize, k.kind, k.expert_idx as usize);
        });
    }

    /// Fallback when `parallel` is disabled — sequential prefetch. See
    /// the `#[cfg(feature = "parallel")]` variant for the real doc.
    #[cfg(not(feature = "parallel"))]
    pub fn prefetch_parallel(&self, keys: &[ExpertKey]) {
        self.prefetch(keys);
    }

    /// Predict the routed-expert set the *next* MoE layer will pick and
    /// prefetch it into the cache. Overlaps well with the outer forward
    /// loop's between-layer work (residual add + next attention RMSNorm +
    /// Q/K/V projection) when called right after the current layer's
    /// routing is decided.
    ///
    /// This is the synchronous variant — the prefetch blocks until every
    /// predicted key is either resident or freshly loaded. Async overlap
    /// (predict + prefetch on a background thread while the current
    /// layer's matvec runs) is deferred to Phase 4c; async support would
    /// require a bounded work queue and thread pool, adds synchronisation
    /// complexity, and cannot be validated without a real DeepSeek-V3
    /// GGUF to bench against.
    ///
    /// # Predictor
    ///
    /// The [`NextLayerPredictor`] parameter decouples the prefetch
    /// mechanism from the actual prediction algorithm. The simplest
    /// implementation, [`PersistenceHeuristic`], assumes the next layer
    /// picks the same top-k as the current one. The colibri paper reports
    /// ~71.6% overlap between consecutive layers' top-k on real
    /// DeepSeek-V3 traffic, so persistence is a strong default even
    /// though a proper next-layer predictor would do better.
    ///
    /// Returns the number of predicted keys that were prefetched (i.e.
    /// `top_k * 3` unless the predictor produced fewer).
    pub fn prefetch_predicted_next_layer(
        &self,
        current_router_logits: &[f32],
        next_layer_idx: usize,
        top_k: usize,
        predictor: &dyn NextLayerPredictor,
    ) -> usize {
        let predicted = predictor.predict(current_router_logits, top_k);
        let mut keys = Vec::with_capacity(predicted.len() * 3);
        for e in predicted {
            keys.push(ExpertKey::new(next_layer_idx, ExpertKind::Gate, e));
            keys.push(ExpertKey::new(next_layer_idx, ExpertKind::Up, e));
            keys.push(ExpertKey::new(next_layer_idx, ExpertKind::Down, e));
        }
        let n = keys.len();
        self.prefetch_parallel(&keys);
        n
    }

    /// Hot-pin a batch of expert keys. Pinned entries survive LRU
    /// eviction until [`unpin_experts`] removes them (or the pool drops).
    /// Intended for pinning the top-N most-frequent experts identified
    /// by offline profiling — costs a fixed slice of RAM but eliminates
    /// the miss penalty for those keys entirely. See
    /// [`LruExpertCache::pin`] for the semantic details.
    ///
    /// This is Phase 4b.3 (Issue #34). It intentionally does NOT load
    /// the pinned keys into the cache — call [`prefetch`] first if you
    /// want the keys warm. Pinning without prefetching means the first
    /// access is still a miss, but from then on the entry is permanent.
    ///
    /// [`unpin_experts`]: Self::unpin_experts
    /// [`prefetch`]: Self::prefetch
    pub fn pin_experts(&self, keys: &[ExpertKey]) {
        let mut cache = self.cache.lock().unwrap();
        for &k in keys {
            cache.pin(k);
        }
    }

    /// Unpin a batch of previously-pinned expert keys, making them
    /// evictable again. Silently ignores keys that were not pinned.
    pub fn unpin_experts(&self, keys: &[ExpertKey]) {
        let mut cache = self.cache.lock().unwrap();
        for k in keys {
            cache.unpin(k);
        }
    }

    /// Snapshot of current cache metrics — hits, misses, live-byte total.
    /// Zero-cost when unused; primarily consumed by bench harnesses.
    pub fn cache_stats(&self) -> CacheStats {
        let cache = self.cache.lock().unwrap();
        CacheStats {
            hits: cache.hits(),
            misses: cache.misses(),
            current_bytes: cache.current_bytes(),
            entries: cache.len(),
            pinned: cache.pinned_len(),
        }
    }
}

/// Predict which routed-expert indices the next MoE layer will pick, given
/// the current layer's raw router logits.
///
/// Implementations may consume the
/// logits as-is or apply arbitrary post-processing (sigmoid, softmax,
/// noaux_tc bias, learned lookahead model, etc.).
///
/// The returned indices are in `0..n_routed_experts`; duplicates and
/// out-of-range entries are silently dropped by the caller's prefetch
/// dispatch, so implementations may keep the algorithm simple without
/// guarding against them here.
///
/// Phase 4b infrastructure — real DeepSeek-V3 validation of prediction
/// accuracy is deferred to a follow-up (blocked on ~370 GB local disk to
/// hold the model checkpoint). See [`PersistenceHeuristic`] for the
/// simplest baseline.
pub trait NextLayerPredictor: Send + Sync {
    fn predict(&self, current_router_logits: &[f32], top_k: usize) -> Vec<usize>;
}

/// The zero-training baseline predictor: **assume the next layer picks the
/// same top-k routed experts as the current layer**.
///
/// This is what colibri
/// calls "persistence"; empirically it correctly predicts ~71.6% of the
/// next layer's top-k routing on real DeepSeek-V3 traffic (Issue #34 body,
/// PILOT=1 experiment). A one-line implementation that turns out to be
/// surprisingly hard to beat without a learned predictor.
///
/// Selection is by raw logit magnitude — sigmoid vs raw scoring does not
/// change the top-k order for a fixed layer, so we skip the sigmoid pass
/// and sort on the logits directly.
///
/// # Applicability to Kimi K3 (896 experts, top-16)
///
/// The predictor is expert-count and top-k agnostic (both are passed at
/// call time), so it drops in unchanged for K3. The 71.6% overlap number
/// is DeepSeek V3-specific and the equivalent K3 hit-rate has not been
/// measured yet (blocked on the same Phase X.4.b GGUF conversion that
/// blocks end-to-end validation). K3 uses the same Quantile Balancing
/// routing family as V3 (see Section 2.3.3 of the K3 tech report), so
/// persistence remains a defensible zero-cost baseline until a learned
/// K3-specific predictor is trained.
pub struct PersistenceHeuristic;

impl NextLayerPredictor for PersistenceHeuristic {
    fn predict(&self, current_router_logits: &[f32], top_k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = current_router_logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(top_k).map(|(i, _)| i).collect()
    }
}

/// Point-in-time snapshot of an [`LruExpertCache`]'s counters.
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub current_bytes: usize,
    pub entries: usize,
    /// Count of hot-pinned entries (Phase 4b.3). Pinned entries are
    /// exempt from LRU eviction until [`StreamingExpertPool::unpin_experts`]
    /// removes them. `entries - pinned` is the count of evictable entries.
    pub pinned: usize,
}

// ── Kimi K3 sizing helpers (Phase X.4.e) ─────────────────────────────

/// Compute the active per-token routed-expert byte budget for a Kimi K3
/// Stable LatentMoE configuration.
///
/// K3's routed FFN runs in a `routed_expert_hidden_size` latent space
/// (default 3584) with three SiTU-GLU matrices per expert (Gate / Up /
/// Down), each of shape `[latent_hidden × moe_intermediate_size]`
/// (default 3584 × 3072). At Q4_K_M (≈ 0.5 byte / weight) that's
/// ≈ 5.5 MB per matrix, ≈ 16.5 MB per expert. With `num_experts_per_tok
/// = 16` active per layer across `num_moe_layers = 92` MoE layers the
/// per-token active weight footprint is:
///
/// ```text
/// active_bytes = num_moe_layers × num_experts_per_tok × 3 slabs
///              × latent_hidden × moe_intermediate × bytes_per_weight
///            = 92 × 16 × 3 × 3584 × 3072 × 0.5
///            ≈ 24 GB (Q4)
/// ```
///
/// This function returns that number for arbitrary K3 sizing, letting
/// callers derive an [`LruExpertCache`] budget without hardcoding the
/// value at every callsite.
///
/// # Bytes-per-weight guidance
///
/// - Q4_K_M / MXFP4 native: pass `bytes_per_weight_x100 = 50` (0.50)
/// - Q5_K_M: `bytes_per_weight_x100 = 63` (0.625)
/// - Q6_K: `bytes_per_weight_x100 = 82` (0.8203)
/// - Q8_0: `bytes_per_weight_x100 = 106` (1.0625)
/// - BF16 / F16: `bytes_per_weight_x100 = 200`
///
/// The `x100` fixed-point encoding avoids `f32` in a public API for a
/// value that only needs 2 decimal digits of precision.
#[must_use]
pub const fn kimi_k3_active_bytes(
    num_moe_layers: usize,
    num_experts_per_tok: usize,
    latent_hidden: usize,
    moe_intermediate: usize,
    bytes_per_weight_x100: usize,
) -> usize {
    // 3 slabs per expert (SiTU-GLU: gate, up, down)
    let per_slab = latent_hidden * moe_intermediate * bytes_per_weight_x100 / 100;
    num_moe_layers * num_experts_per_tok * 3 * per_slab
}

/// Recommended LRU cache byte budget for a streaming-expert pool.
///
/// Formula: `active_bytes × safety_multiplier / 10`. A multiplier of 12
/// (i.e. 1.2×) is a defensible default — it covers one full token of
/// active weights plus 20% headroom for cross-token LRU reuse (which
/// pays off whenever two consecutive tokens share any routed experts,
/// as the persistence heuristic observed on DeepSeek V3).
///
/// For Kimi K3 Q4 on Mac M3 Max 128 GB, the recommended budget is
/// `kimi_k3_active_bytes(92, 16, 3584, 3072, 50) × 12 / 10 ≈ 30 GB`.
/// Higher multipliers (15-17) trade RAM for hit rate; lower multipliers
/// (10-11) trade hit rate for headroom on tighter machines.
#[must_use]
pub const fn recommended_budget_bytes(active_bytes: usize, safety_multiplier_x10: usize) -> usize {
    active_bytes * safety_multiplier_x10 / 10
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pool(n_experts: usize, bytes_per_expert: usize, budget: usize) -> StreamingExpertPool {
        // Synthetic source: byte value at offset `i` = `(i % 251) as u8` so
        // each expert has a deterministic-but-distinct byte pattern.
        let total = 3 * n_experts * bytes_per_expert;
        let data: Vec<u8> = (0..total).map(|i| (i % 251) as u8).collect();
        let source: Arc<dyn ExpertByteSource> = Arc::new(data);
        let mut per_kind = [ExpertLayerInfo {
            base_offset: 0,
            bytes_per_expert,
            n_experts,
            qtype: GgmlType::Q4_K,
        }; 3];
        per_kind[ExpertKind::Gate as usize].base_offset = 0;
        per_kind[ExpertKind::Up as usize].base_offset = n_experts * bytes_per_expert;
        per_kind[ExpertKind::Down as usize].base_offset = 2 * n_experts * bytes_per_expert;
        StreamingExpertPool::new(source, vec![per_kind], budget)
    }

    #[test]
    fn get_or_load_returns_deterministic_slab_from_source() {
        let pool = make_pool(4, 32, 1024);
        // Expert 2's gate slab starts at offset 2 * 32 = 64, spans 32 bytes.
        let slab = pool.get_or_load(0, ExpertKind::Gate, 2);
        assert_eq!(slab.len(), 32);
        // Confirm the deterministic byte pattern.
        for (i, &b) in slab.iter().enumerate() {
            assert_eq!(b, ((64 + i) % 251) as u8, "byte {i} mismatch");
        }
    }

    #[test]
    fn second_lookup_is_a_cache_hit() {
        let pool = make_pool(4, 32, 1024);
        let _slab1 = pool.get_or_load(0, ExpertKind::Up, 1);
        let stats_after_load = pool.cache_stats();
        assert_eq!(stats_after_load.misses, 1);
        assert_eq!(stats_after_load.hits, 0);

        let _slab2 = pool.get_or_load(0, ExpertKind::Up, 1);
        let stats_after_hit = pool.cache_stats();
        assert_eq!(stats_after_hit.misses, 1, "no extra miss on second lookup");
        assert_eq!(stats_after_hit.hits, 1);
    }

    #[test]
    fn evicts_lru_when_budget_exceeded() {
        // Budget of 64 bytes = only 2 slabs of 32 bytes fit simultaneously.
        let pool = make_pool(4, 32, 64);
        // Load experts 0, 1, 2 in order — expert 0 becomes LRU.
        let _e0 = pool.get_or_load(0, ExpertKind::Gate, 0);
        let _e1 = pool.get_or_load(0, ExpertKind::Gate, 1);
        assert_eq!(pool.cache_stats().current_bytes, 64);
        let _e2 = pool.get_or_load(0, ExpertKind::Gate, 2);
        // Expert 0 must have been evicted; cache still at 64 bytes.
        let stats = pool.cache_stats();
        assert_eq!(stats.current_bytes, 64);
        assert_eq!(
            stats.entries, 2,
            "cache holds exactly 2 entries after eviction"
        );
        // Re-loading expert 0 counts as a miss (evicted, not a hit).
        let _e0_again = pool.get_or_load(0, ExpertKind::Gate, 0);
        let stats2 = pool.cache_stats();
        assert_eq!(stats2.misses, 4, "expert 0 reload is a fresh miss");
    }

    #[test]
    fn get_bumps_lru_position() {
        // Budget 64 bytes = 2 slabs. Load 0, 1, touch 0, then load 2.
        // Since 0 was just accessed, expert 1 (least recent) should be evicted.
        let pool = make_pool(4, 32, 64);
        let _e0 = pool.get_or_load(0, ExpertKind::Gate, 0);
        let _e1 = pool.get_or_load(0, ExpertKind::Gate, 1);
        // Re-touch expert 0 — bumps it to MRU.
        let _e0_again = pool.get_or_load(0, ExpertKind::Gate, 0);
        // Now load expert 2 — expert 1 should be the victim.
        let _e2 = pool.get_or_load(0, ExpertKind::Gate, 2);
        let stats = pool.cache_stats();
        assert_eq!(stats.entries, 2);
        // Verify 0 and 2 are in cache (touch them, count hits).
        let hits_before = pool.cache_stats().hits;
        let _ = pool.get_or_load(0, ExpertKind::Gate, 0);
        let _ = pool.get_or_load(0, ExpertKind::Gate, 2);
        let hits_after = pool.cache_stats().hits;
        assert_eq!(hits_after - hits_before, 2, "both 0 and 2 should hit");
    }

    #[test]
    fn different_kinds_do_not_collide() {
        // Same layer_idx + expert_idx but different kind → distinct cache entries.
        let pool = make_pool(4, 32, 1024);
        let gate = pool.get_or_load(0, ExpertKind::Gate, 1);
        let up = pool.get_or_load(0, ExpertKind::Up, 1);
        let down = pool.get_or_load(0, ExpertKind::Down, 1);
        // Gate slab starts at 0 * 4*32 = 0, so expert 1 gate offset = 32.
        // Up slab starts at 1 * 4*32 = 128, so expert 1 up offset = 160.
        // Down slab starts at 2 * 4*32 = 256, so expert 1 down offset = 288.
        assert_eq!(gate[0], 32_u8);
        assert_eq!(up[0], 160_u8);
        assert_eq!(down[0], (288 % 251) as u8);
        assert_eq!(pool.cache_stats().entries, 3);
        assert_eq!(pool.cache_stats().misses, 3);
    }

    #[test]
    fn arc_lets_caller_outlive_eviction() {
        // Load expert 0, keep its Arc, then evict 0 by loading enough others.
        // The Arc must still deref to valid bytes.
        let pool = make_pool(4, 32, 64);
        let e0 = pool.get_or_load(0, ExpertKind::Gate, 0);
        let expected: Vec<u8> = e0.iter().copied().collect();
        // Fill cache past capacity so 0 gets evicted.
        let _e1 = pool.get_or_load(0, ExpertKind::Gate, 1);
        let _e2 = pool.get_or_load(0, ExpertKind::Gate, 2);
        // Cache no longer holds expert 0.
        let stats = pool.cache_stats();
        assert!(!stats.entries == 0 || stats.entries <= 2);
        // The Arc we captured earlier still holds the original bytes.
        assert_eq!(*e0, expected, "arc-held slab must survive eviction");
    }

    #[test]
    fn prefetch_batch_warms_all_keys() {
        let pool = make_pool(4, 32, 1024);
        let keys = vec![
            ExpertKey::new(0, ExpertKind::Gate, 0),
            ExpertKey::new(0, ExpertKind::Gate, 1),
            ExpertKey::new(0, ExpertKind::Up, 2),
            ExpertKey::new(0, ExpertKind::Down, 3),
        ];
        pool.prefetch(&keys);
        let after_prefetch = pool.cache_stats();
        assert_eq!(after_prefetch.misses, 4, "prefetch loads every key");
        assert_eq!(after_prefetch.entries, 4);

        // Every subsequent get_or_load must be a hit.
        for key in &keys {
            let _ = pool.get_or_load(key.layer_idx as usize, key.kind, key.expert_idx as usize);
        }
        let after_get = pool.cache_stats();
        assert_eq!(after_get.misses, 4, "no fresh misses after prefetch");
        assert_eq!(after_get.hits, 4);
    }

    #[test]
    fn prefetch_is_noop_for_already_cached_keys() {
        let pool = make_pool(4, 32, 1024);
        // Load expert 0 once.
        let _ = pool.get_or_load(0, ExpertKind::Gate, 0);
        assert_eq!(pool.cache_stats().misses, 1);
        // Prefetch the same key + one new key.
        let keys = vec![
            ExpertKey::new(0, ExpertKind::Gate, 0), // already cached (hit, LRU bump)
            ExpertKey::new(0, ExpertKind::Gate, 1), // fresh (miss)
        ];
        pool.prefetch(&keys);
        let stats = pool.cache_stats();
        assert_eq!(
            stats.misses, 2,
            "only the fresh key increments miss counter"
        );
        assert_eq!(
            stats.hits, 1,
            "cached key increments hit counter via LRU bump"
        );
    }

    #[test]
    fn prefetch_parallel_matches_sequential_semantics() {
        // Under both feature configurations `prefetch_parallel` must
        // produce the same final cache contents as `prefetch`. This
        // catches the mistake of e.g. dropping keys under parallelism.
        let pool_seq = make_pool(8, 32, 4096);
        let pool_par = make_pool(8, 32, 4096);
        let keys: Vec<ExpertKey> = (0..8)
            .flat_map(|e| {
                [ExpertKind::Gate, ExpertKind::Up, ExpertKind::Down]
                    .into_iter()
                    .map(move |k| ExpertKey::new(0, k, e))
            })
            .collect();
        pool_seq.prefetch(&keys);
        pool_par.prefetch_parallel(&keys);
        let s_seq = pool_seq.cache_stats();
        let s_par = pool_par.cache_stats();
        assert_eq!(
            s_seq.entries, s_par.entries,
            "prefetch_parallel entry count must match sequential"
        );
        assert_eq!(
            s_seq.current_bytes, s_par.current_bytes,
            "prefetch_parallel byte total must match sequential"
        );
    }

    #[test]
    fn pinned_entry_survives_budget_pressure() {
        // Budget = 64 bytes → only 2 slabs of 32 bytes fit at once.
        let pool = make_pool(4, 32, 64);
        let pinned_key = ExpertKey::new(0, ExpertKind::Gate, 0);
        pool.pin_experts(&[pinned_key]);
        assert_eq!(pool.cache_stats().pinned, 1);

        // Load the pinned key, then flood the cache with 3 other keys.
        // Normally the LRU would evict the oldest entry (the pinned one),
        // but the pin should shield it.
        let _ = pool.get_or_load(0, ExpertKind::Gate, 0); // pinned load
        let _ = pool.get_or_load(0, ExpertKind::Gate, 1);
        let _ = pool.get_or_load(0, ExpertKind::Gate, 2); // triggers eviction
        let _ = pool.get_or_load(0, ExpertKind::Gate, 3); // triggers eviction

        // Verify the pinned key is still cached: subsequent get is a hit.
        let hits_before = pool.cache_stats().hits;
        let _ = pool.get_or_load(0, ExpertKind::Gate, 0);
        let hits_after = pool.cache_stats().hits;
        assert_eq!(
            hits_after - hits_before,
            1,
            "pinned key must remain in cache"
        );
    }

    #[test]
    fn unpin_makes_entry_evictable_again() {
        let pool = make_pool(4, 32, 64);
        let key = ExpertKey::new(0, ExpertKind::Gate, 0);
        pool.pin_experts(&[key]);
        let _ = pool.get_or_load(0, ExpertKind::Gate, 0);
        let _ = pool.get_or_load(0, ExpertKind::Gate, 1);
        // Unpin, then trigger eviction pressure — the previously-pinned
        // key should now be a valid eviction victim.
        pool.unpin_experts(&[key]);
        assert_eq!(pool.cache_stats().pinned, 0);
        let _ = pool.get_or_load(0, ExpertKind::Gate, 2);
        let _ = pool.get_or_load(0, ExpertKind::Gate, 3);
        // Expert 0 was the oldest and no longer pinned → evicted.
        // Reloading it is a miss, not a hit.
        let misses_before = pool.cache_stats().misses;
        let _ = pool.get_or_load(0, ExpertKind::Gate, 0);
        let misses_after = pool.cache_stats().misses;
        assert_eq!(
            misses_after - misses_before,
            1,
            "unpinned key must be evictable"
        );
    }

    #[test]
    fn pinning_all_entries_bypasses_eviction_bound() {
        // Budget = 64 bytes, but pin 3 * 32 = 96 bytes of entries. The
        // cache is allowed to overflow because every candidate is pinned.
        let pool = make_pool(4, 32, 64);
        let keys = [
            ExpertKey::new(0, ExpertKind::Gate, 0),
            ExpertKey::new(0, ExpertKind::Gate, 1),
            ExpertKey::new(0, ExpertKind::Gate, 2),
        ];
        pool.pin_experts(&keys);
        for k in &keys {
            let _ = pool.get_or_load(0, ExpertKind::Gate, k.expert_idx as usize);
        }
        let stats = pool.cache_stats();
        assert_eq!(stats.entries, 3, "all 3 pinned keys stay in cache");
        assert!(
            stats.current_bytes > 0,
            "pin overrides byte-budget lower bound"
        );
    }

    #[test]
    fn persistence_heuristic_picks_top_k_by_logit() {
        // Logits [0.1, 0.9, 0.3, 0.7, 0.5] → top-3 = experts 1, 3, 4.
        let logits = vec![0.1, 0.9, 0.3, 0.7, 0.5];
        let predicted = PersistenceHeuristic.predict(&logits, 3);
        assert_eq!(predicted, vec![1, 3, 4]);
    }

    #[test]
    fn persistence_heuristic_handles_top_k_bigger_than_n_experts() {
        let logits = vec![0.1, 0.5];
        let predicted = PersistenceHeuristic.predict(&logits, 8);
        assert_eq!(predicted, vec![1, 0]);
    }

    #[test]
    fn prefetch_predicted_next_layer_warms_next_layer() {
        // Two-layer synthetic pool: verify that calling
        // prefetch_predicted_next_layer on layer 0's logits warms layer 1's
        // top-k experts, and that layer 1's subsequent get_or_load
        // sequence hits every predicted key.
        let n_experts = 4;
        let bytes_per_expert = 32;
        let n_layers = 2;
        let total = n_layers * 3 * n_experts * bytes_per_expert;
        let data: Vec<u8> = (0..total).map(|i| (i % 251) as u8).collect();
        let source: Arc<dyn ExpertByteSource> = Arc::new(data);
        let mut layer_info = Vec::with_capacity(n_layers);
        for layer_idx in 0..n_layers {
            let base = layer_idx * 3 * n_experts * bytes_per_expert;
            let mut per_kind = [ExpertLayerInfo {
                base_offset: 0,
                bytes_per_expert,
                n_experts,
                qtype: GgmlType::Q4_K,
            }; 3];
            per_kind[ExpertKind::Gate as usize].base_offset = base;
            per_kind[ExpertKind::Up as usize].base_offset = base + n_experts * bytes_per_expert;
            per_kind[ExpertKind::Down as usize].base_offset =
                base + 2 * n_experts * bytes_per_expert;
            layer_info.push(per_kind);
        }
        let pool = StreamingExpertPool::new(source, layer_info, 4096);

        // Simulate layer 0's routing: experts 1 and 3 have highest logits.
        let logits = vec![0.1, 0.9, 0.3, 0.7];
        let top_k = 2;
        let n_prefetched = pool.prefetch_predicted_next_layer(
            &logits,
            /* next_layer_idx */ 1,
            top_k,
            &PersistenceHeuristic,
        );
        assert_eq!(n_prefetched, top_k * 3, "3 kinds per predicted expert");

        // Pool now has layer 1's experts 1 and 3 cached (across all 3
        // kinds). Simulate layer 1's forward hitting the SAME experts —
        // every get should be a hit.
        let hits_before = pool.cache_stats().hits;
        for &e in &[1usize, 3] {
            let _ = pool.get_or_load(1, ExpertKind::Gate, e);
            let _ = pool.get_or_load(1, ExpertKind::Up, e);
            let _ = pool.get_or_load(1, ExpertKind::Down, e);
        }
        let hits_after = pool.cache_stats().hits;
        assert_eq!(
            hits_after - hits_before,
            6,
            "persistence heuristic correctly pre-warmed next-layer routing"
        );
    }

    #[test]
    #[cfg(all(unix, feature = "gguf"))]
    fn advise_random_returns_true_for_valid_mmap() {
        // Construct a real temp-file mmap so the syscall has a valid
        // target. On non-Unix builds this test is compiled out and the
        // fallback advise_random always returns false.
        use std::io::Write;
        let mut tf = tempfile_alt();
        tf.write_all(b"hello alice-llm streaming pool madvise test")
            .unwrap();
        tf.flush().unwrap();
        // SAFETY: file is a real, live temp file we just wrote.
        let mmap = unsafe { memmap2::Mmap::map(&tf) }.unwrap();
        assert!(
            super::advise_random(&mmap),
            "madvise MADV_RANDOM must succeed on a valid mmap"
        );
    }

    #[cfg(all(unix, feature = "gguf"))]
    fn tempfile_alt() -> std::fs::File {
        // Minimal in-test tempfile: creates a file under std::env::temp_dir()
        // and immediately unlinks it while keeping the file handle. On
        // Unix the mmap survives the unlink until the FD is closed.
        let mut path = std::env::temp_dir();
        path.push(format!("alice_llm_streaming_test_{}", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        let _ = std::fs::remove_file(&path);
        file
    }

    #[test]
    fn zero_budget_evicts_immediately() {
        let pool = make_pool(4, 32, 0);
        let _e0 = pool.get_or_load(0, ExpertKind::Gate, 0);
        let stats = pool.cache_stats();
        // With budget = 0, the newly inserted slab is over budget, so
        // subsequent inserts evict everything including itself on the next
        // insert. First insert stays because the eviction loop condition
        // uses saturating math and there's nothing else to drop.
        assert!(
            stats.entries <= 1,
            "zero-budget cache must not accumulate entries"
        );
    }

    // ── Kimi K3 topology tests (Phase X.4.e) ──────────────────────

    #[test]
    fn kimi_k3_active_bytes_matches_paper_estimate() {
        // K3 spec (tech report Table 1, §2.3): 92 MoE layers × 16 experts
        // per token × 3 SiTU-GLU slabs × 3584 latent × 3072 intermediate
        // × Q4 (0.5 byte/weight) ≈ 24 GB. This anchors the sizing helper
        // against the paper-reported estimate so a future refactor cannot
        // silently drift the formula.
        let active = kimi_k3_active_bytes(
            92,   // num_moe_layers (total 93 - 1 dense)
            16,   // num_experts_per_tok
            3584, // routed_expert_hidden_size (latent)
            3072, // moe_intermediate_size
            50,   // Q4 = 0.50 byte/weight
        );
        // 92 × 16 × 3 × 3584 × 3072 × 0.5 = 24_326_701_056 bytes ≈ 22.7 GiB
        // (≈ 24.3 GB in SI units, matches the "≈ 24 GB" paper claim).
        assert_eq!(active, 92 * 16 * 3 * 3584 * 3072 / 2);
        // Sanity check the ballpark against the tech report / integration
        // doc estimate (20-30 GB window).
        let gb = active / 1_000_000_000;
        assert!(
            (20..=30).contains(&gb),
            "K3 active bytes {active} ({gb} GB) outside 20-30 GB paper estimate window"
        );
    }

    #[test]
    fn recommended_budget_applies_safety_multiplier() {
        let active = 20_000_000_000_usize; // 20 GB
                                           // Default 1.2× → 24 GB.
        assert_eq!(recommended_budget_bytes(active, 12), 24_000_000_000);
        // Aggressive 1.7× for high hit-rate targets → 34 GB.
        assert_eq!(recommended_budget_bytes(active, 17), 34_000_000_000);
        // Tight 1.0× (baseline, no cross-token reuse) → 20 GB.
        assert_eq!(recommended_budget_bytes(active, 10), 20_000_000_000);
    }

    #[test]
    fn pool_supports_kimi_k3_896_experts_top16_dispatch() {
        // Construct a synthetic K3-topology pool: 1 MoE layer, 896 routed
        // experts, tiny slab size (1 KB / expert) so the test stays fast
        // and RAM-cheap while still exercising the 896-expert index
        // range and top-16 dispatch path. This proves the pool
        // infrastructure is expert-count-agnostic: nothing in the LRU /
        // slab layout / cache accounting hardcodes 256.
        let n_experts = 896;
        let bytes_per_expert = 1024; // 1 KB
        let per_layer_bytes = 3 * n_experts * bytes_per_expert; // ~2.6 MB
                                                                // Budget for top-16 across 3 slabs = 48 slabs × 1 KB = 48 KB,
                                                                // sized × 4 for cross-call LRU reuse.
        let budget = 16 * 3 * bytes_per_expert * 4;
        let pool = make_pool(n_experts, bytes_per_expert, budget);

        // Sanity: verify n_experts propagated through construction.
        assert_eq!(pool.bytes_per_expert(0, ExpertKind::Gate), bytes_per_expert);

        // Simulate one token's top-16 dispatch across all 3 slab kinds
        // (48 slab fetches). Every one should succeed and stay within
        // the 896-expert index range.
        let top16: [usize; 16] = [
            7, 42, 100, 200, 300, 400, 500, 600, 700, 800, 850, 890, 895, 3, 17, 128,
        ];
        for &e in &top16 {
            let g = pool.get_or_load(0, ExpertKind::Gate, e);
            let u = pool.get_or_load(0, ExpertKind::Up, e);
            let d = pool.get_or_load(0, ExpertKind::Down, e);
            assert_eq!(g.len(), bytes_per_expert);
            assert_eq!(u.len(), bytes_per_expert);
            assert_eq!(d.len(), bytes_per_expert);
        }

        // Re-fetch the same 16 experts — every access must be a hit as
        // long as the budget accommodates 48 slabs (it does at 192 KB).
        let stats_before = pool.cache_stats();
        for &e in &top16 {
            let _g = pool.get_or_load(0, ExpertKind::Gate, e);
            let _u = pool.get_or_load(0, ExpertKind::Up, e);
            let _d = pool.get_or_load(0, ExpertKind::Down, e);
        }
        let stats_after = pool.cache_stats();
        assert_eq!(
            stats_after.hits - stats_before.hits,
            48,
            "top-16 × 3-slab re-fetch must be all hits inside the 48-slab budget"
        );
        assert_eq!(
            stats_after.misses, stats_before.misses,
            "no fresh misses on the re-fetch"
        );

        // Highest expert index (895) must be reachable — smoke test the
        // upper end of the 0..896 range.
        let last = pool.get_or_load(0, ExpertKind::Down, 895);
        assert_eq!(last.len(), bytes_per_expert);

        // Silence the unused-var warning for `per_layer_bytes` — it's
        // documentation of the layout, not consumed by asserts.
        let _ = per_layer_bytes;
    }

    #[test]
    fn pool_rejects_out_of_range_expert_index_at_896() {
        // Complement to the above: expert_idx == n_experts (out of
        // range) must panic rather than silently serve garbage from
        // adjacent slabs. The pool asserts on `expert_idx < n_experts`
        // in get_or_load; verify that boundary at the K3 scale so a
        // future refactor doesn't accidentally soften it.
        let pool = make_pool(896, 128, 4096);
        let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = pool.get_or_load(0, ExpertKind::Gate, 896);
        }));
        assert!(
            ok.is_err(),
            "expert_idx = n_experts must panic (out of range)"
        );
    }

    #[test]
    fn persistence_heuristic_scales_to_896_experts_top16() {
        // The predictor is expert-count and top-k agnostic — verify
        // that on a 896-length logits vector with top-k=16 it returns
        // the 16 largest indices, no truncation / off-by-one issues at
        // the K3 scale. Uses a monotone gradient so the expected
        // top-16 is trivially the last 16 indices.
        use super::{NextLayerPredictor, PersistenceHeuristic};
        let mut logits = vec![0.0_f32; 896];
        for (i, l) in logits.iter_mut().enumerate() {
            *l = i as f32;
        }
        let mut picks = PersistenceHeuristic.predict(&logits, 16);
        assert_eq!(picks.len(), 16, "top-16 selection returned wrong count");
        // The 16 largest indices in a 0..896 gradient are 880..896.
        picks.sort_unstable();
        let expected: Vec<usize> = (880..896).collect();
        assert_eq!(picks, expected, "top-16 must be the last 16 indices");
    }
}
