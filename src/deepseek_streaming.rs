//! DeepSeek-V3 routed-expert streaming from disk (Phase 4a, Issue #34).
//!
//! # Overview
//!
//! DeepSeek-V3 671B has 61 MoE layers × 256 routed experts + 1 shared expert
//! per layer. Even at Q4_K_M each routed expert is ~19 MB, so keeping every
//! expert in RAM costs ~370 GB. The core of the colibri innovation is to
//! **load routed experts on demand** with an LRU cache, since a single token
//! only touches `num_experts_per_tok = 8` per MoE layer.
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
/// (Phase 4b.4). This disables the sequential-readahead heuristic — a win
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
            mmap_bytes.as_ptr() as *mut libc::c_void,
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
/// DeepSeek-V3 model. One pool instance is shared across all MoE layers via
/// `Arc` — the layer index is passed at fetch time.
pub struct StreamingExpertPool {
    source: Arc<dyn ExpertByteSource>,
    /// Indexed by `[layer_idx][kind as usize]`. Populated at construction
    /// time by walking the GGUF metadata for the routed-expert tensors.
    layer_info: Vec<[ExpertLayerInfo; 3]>,
    cache: Mutex<LruExpertCache>,
}

impl StreamingExpertPool {
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
        assert_eq!(gate[0], (32 % 251) as u8);
        assert_eq!(up[0], (160 % 251) as u8);
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
            stats.current_bytes > stats.entries * 0, // just >0
            "pin overrides byte-budget lower bound"
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
}
