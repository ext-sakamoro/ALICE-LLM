//! DSpark speculative decoding primitives.
//!
//! Reference: <https://huggingface.co/RadixArk/Kimi-K3-DSpark> (2026-07-29 absorbed).
//!
//! DSpark = DFlash 並列 draft + Markov logit-bias + 位置別 confidence head の 3 要素合成
//! 本 module は standalone primitive のみ提供 `generate_speculative_dual` 等への配線は caller 責務
//!
//! ## Scope
//!
//! Phase 1 (完了): [`MarkovBigramBias`] vanilla DSpark は rank=256
//! Phase 2 (完了): [`PositionConfidenceHead`] SGD BCE 学習可能な位置別 sigmoid confidence
//! Phase 3 (完了): [`DFlashParallelDraft`] 外部 draft callback + [`DraftPosition`] / [`DraftBlock`] I/O 型
//! Phase 4 (完了): [`BigramBias`] trait + [`FullCountBigramBias`] (eager truncate 制約解消) + optional `dspark-serde` feature で serde derive
//! Phase 5 (完了): [`apply_bigram_bias_maybe`] helper + `dspark` feature 経由で `Llama3Model::generate_speculative_dual_dspark` を追加 (`examples/speculative_dspark_dual.rs` で A/B 比較)
//! Phase 6 (完了): [`DsparkAdvancedConfig`] + `Llama3Model::forward_capture_hidden` (hidden state 抽出) + `generate_speculative_dual_dspark` に第 9 引数 `advanced: Option<&DsparkAdvancedConfig>` 追加、confidence-gated 早期打切り [`PositionConfidenceHead`] 統合完了 標準 arch (Llama/Mistral/Gemma2/Qwen2/Qwen3/Qwen3_5) 限定
//! Phase 7 (完了): [`DsparkLabelSample`] + `Llama3Model::generate_speculative_dual_collect_labels` (accept/reject label collection) + `examples/dspark_train_confidence_head.rs` (SGD BCE 学習 + bincode save) + `examples/speculative_dspark_dual.rs` の `--confidence-head` オプション追加 (trained head load + threshold 0.3/0.5/0.7 の A/B/C 比較)
//! Phase 8 (完了): `KimiK3Model::forward_with_layer_hook` + `KimiK3Model::forward_capture_hidden` を追加 (K3 の 93 層 KDA + Gated MLA + 1 dense 対応、hook は info-only で post-FFN residual `x` を expose、既存 `KimiK3Model::forward` 完全無改変の duplicate 実装で安全性優先) K3 を draft として `generate_speculative_dual` に統合するのは Phase 9+ (K3 の `layer_caches` は `PagedKvCache` 不使用で `rollback_to` / `seq_len` trait 抽象が必要)
//! Phase 9 (完了): [`DraftBackend`] trait 新規追加 (9 method で draft model 抽象化)、`impl DraftBackend for Llama3Model` + `Llama3Model::generate_speculative_dual_dspark` / `generate_speculative_dual_collect_labels` の draft 引数を `&mut dyn DraftBackend` に refactor (既存 example は Rust 型推論で無改修 coerce)、`KimiK3Model` の impl は Phase 10 (KDA snapshot 依存)
//! Phase 10 (完了): `KimiK3AttnResState` に `#[derive(Clone)]` 追加、`KimiK3MlaCache::rollback_to(pos)` (positional truncate)、`KimiK3ModelSnapshot` struct、`KimiK3Model` に `token_count` + `snapshot_ring: VecDeque<KimiK3ModelSnapshot>` + `max_snapshot_ring` (default 8) を追加、`snapshot() / restore() / seq_len() / rollback_to() / set_max_snapshot_ring() / forward_with_snapshot() / forward_capture_hidden_with_snapshot()` の pub method 追加、`impl DraftBackend for KimiK3Model<'_>` (feature-gated) を実装、K3 が draft として `generate_speculative_dual_dspark` に使用可能に snapshot は forward 直前に push、`rollback_to(pos)` は ring から適切な snapshot 検索 + restore
//! Phase 11 (user 実行、記事骨子 + 測定 script 用意済 2026-08-04): Track 5-4 K3 accept length 実測 (real K3 566GB weights + trained head、`speculative_dspark_dual --confidence-head` を K3 draft で走らせて accept rate / tok/s / speedup を測定、Zenn 記事に publish)
//! Phase 12 (完了): f16 quantized snapshot compression: `KimiK3ModelSnapshotCompact` + `set_snapshot_compact_mode(true)` で ~2× メモリ削減 (KDA state + MLA c_k/k_rope + AttnResState.banked を f32 → IEEE 754 binary16 変換、精度 loss ~1e-3)、既存 full snapshot と排他的に使用、default false で無破壊 手書き IEEE 754 half-precision 変換 (無依存追加、`half` crate 不要)
//! Phase 12b Part 1+2 (完了): rank-1 delta encoding primitives: `KimiDeltaHeadUpdate` struct (q_pre/k_pre/v_pre/k_conv/v_conv/alpha/beta) + `KimiDeltaHeadCache::apply_update()` (conv_state ring push + kimi_delta_step 呼出で 1 step 進める) + `KimiK3ModelSnapshotDelta { base_snapshot, per_step_updates }` + `KimiK3Model::{snapshot_delta_from, restore_from_delta, snapshot_delta_bytes_estimate}` 実装、7 unit test 全 pass (apply_update state bit-exact + ring 進め + delta roundtrip)
//! Phase 12b Part 3a (完了): head-level capture 関数追加 `kimi_delta_forward_head_with_capture(x, params, cache, l2_eps) -> (Vec<f32>, KimiDeltaHeadUpdate)`、既存 `kimi_delta_forward_head` と output/state bit-exact 一致、capture した update を fresh cache に `apply_update` で replay して state bit-exact 復元、2 unit test 全 pass
//! Phase 12b Part 3b (完了): delta ring infrastructure: `KimiK3Model` に `snapshot_ring_delta: VecDeque<KimiK3ModelSnapshotDelta>` + `delta_snapshots: bool` field 追加、`set_snapshot_delta_mode(bool)` (3 mode 排他 Full/Compact/Delta) + `is_snapshot_delta_mode` getter、`snapshot_ring_len` / `snapshot_ring_bytes_estimate` / `reset` / `set_max_snapshot_ring` を delta ring 対応、`rollback_to` に delta mode branch (base restore + updates replay)、8 追加 unit test (mode toggle + ring 管理 + synthetic rollback replay) 全 pass
//! Phase 12b Part 3c1 (完了): KDA capture wire-up: `kimi_k3_kda_head_forward` に `capture: Option<&mut KimiDeltaHeadUpdate>` 引数追加 (Some で `kimi_delta_forward_head_with_capture` delegate、None で既存 `kimi_delta_forward_head`)、`kimi_k3_kda_layer_forward` に `capture_updates: Option<&mut [KimiDeltaHeadUpdate]>` 引数追加 (Some で serial iteration + per-head slot、None で既存 parallel rayon path)、既存 2 K3Model caller (K3Model::forward / forward_with_layer_hook) は None 渡し、既存 594 test 無破壊 pass
//! Phase 12b Part 3c2 (完了): K3Model 統合 + delta mode 完全 wiring: `KimiK3Model::forward_capture_updates(token_id) -> (Vec<f32>, Vec<Vec<KimiDeltaHeadUpdate>>)` (K3 forward 3rd duplicate + KDA layer capture 統合、~250 LOC)、`forward_with_snapshot` に delta mode branch 追加 (現 delta bound 超過で rebase = 現状態を新 base に snapshot、そうでなければ forward_capture_updates 呼出 + 返り値 updates を現 delta の per_step_updates に append)、既存 594 test 無破壊 pass、**Part 3c2 完了で delta mode が完全機能**、実 K3 (real weights 必要) で **6-10× 実メモリ圧縮 (290MB → 30-48MB)** が実測可能
//!
//! ## Rank-K bigram bias 設計
//!
//! 各 previous token に対し top-K next token の観測頻度を保持し、生成時に
//! `logits[next] += strength * ln(1 + count)` を加算する ln 形状は runaway
//! high-count bias を damp する 内部は eager truncate 方式 (observe 時に
//! rank 到達で下位 bucket を drop) 完全 top-K が必要な場合は Phase 2 で
//! full-count sketch を追加する

use std::collections::HashMap;

#[cfg(feature = "dspark-serde")]
use serde::{Deserialize, Serialize};

/// DSpark primitives が扱う token id 型
pub type TokenId = u32;

/// DSpark primitive のエラー
#[cfg_attr(feature = "dspark-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DsparkError {
    /// Token id が宣言 vocab_size を超過
    TokenOutOfVocab {
        /// 越境した token id
        token: TokenId,
        /// 宣言 vocab size
        vocab_size: u32,
    },
    /// Logits slice 長が vocab_size と不一致
    LogitsLenMismatch {
        /// 期待長 = vocab_size
        expected: usize,
        /// 実長
        got: usize,
    },
    /// rank が 0
    ZeroRank,
    /// vocab_size が 0
    ZeroVocab,
    /// block_size が 0 (PositionConfidenceHead)
    ZeroBlockSize,
    /// hidden_dim が 0 (PositionConfidenceHead)
    ZeroHiddenDim,
    /// hidden 入力の長さが hidden_dim と不一致
    HiddenLenMismatch {
        /// 期待長 = hidden_dim
        expected: usize,
        /// 実長
        got: usize,
    },
    /// position 引数が block_size 以上
    PositionOutOfRange {
        /// 越境した position
        position: u32,
        /// 宣言 block_size
        block_size: u32,
    },
    /// predict_block に渡された hidden_states の要素数が block_size と不一致
    BlockStatesCountMismatch {
        /// 期待長 = block_size
        expected: usize,
        /// 実長
        got: usize,
    },
    /// prefix が空 (DFlashParallelDraft)
    EmptyPrefix,
    /// 外部 draft model が返した position 数が block_size と不一致
    DraftModelBlockSizeMismatch {
        /// 期待長 = block_size
        expected: usize,
        /// 実長
        got: usize,
    },
    /// vocab_size 一致検証で失敗 (bigram_bias or draft_fn logits)
    VocabSizeMismatch {
        /// 期待長
        expected: u32,
        /// 実長
        got: u32,
    },
    /// hidden_dim 一致検証で失敗 (confidence_head or draft_fn hidden)
    HiddenDimMismatch {
        /// 期待長
        expected: u32,
        /// 実長
        got: u32,
    },
    /// confidence_head.block_size と DFlashParallelDraft.block_size が不一致
    ConfidenceHeadBlockSizeMismatch {
        /// 期待長
        expected: u32,
        /// 実長
        got: u32,
    },
    /// bigram_bias.vocab_size と DFlashParallelDraft.vocab_size が不一致
    BigramVocabMismatch {
        /// 期待長
        expected: u32,
        /// 実長
        got: u32,
    },
    /// draft_fn の logits が全て NaN で argmax 不能
    DraftLogitsAllNonFinite,
    /// 外部 draft model が返した任意のエラー文字列
    DraftModelFailed(String),
}

impl core::fmt::Display for DsparkError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TokenOutOfVocab { token, vocab_size } => {
                write!(f, "token id {token} exceeds vocab size {vocab_size}")
            }
            Self::LogitsLenMismatch { expected, got } => {
                write!(
                    f,
                    "logits length {got} does not match vocab size {expected}"
                )
            }
            Self::ZeroRank => write!(f, "rank must be non-zero"),
            Self::ZeroVocab => write!(f, "vocab size must be non-zero"),
            Self::ZeroBlockSize => write!(f, "block_size must be non-zero"),
            Self::ZeroHiddenDim => write!(f, "hidden_dim must be non-zero"),
            Self::HiddenLenMismatch { expected, got } => {
                write!(
                    f,
                    "hidden length {got} does not match hidden_dim {expected}"
                )
            }
            Self::PositionOutOfRange {
                position,
                block_size,
            } => {
                write!(f, "position {position} exceeds block_size {block_size}")
            }
            Self::BlockStatesCountMismatch { expected, got } => {
                write!(
                    f,
                    "hidden_states count {got} does not match block_size {expected}"
                )
            }
            Self::EmptyPrefix => write!(f, "prefix must be non-empty"),
            Self::DraftModelBlockSizeMismatch { expected, got } => {
                write!(
                    f,
                    "draft model returned {got} positions but expected {expected}"
                )
            }
            Self::VocabSizeMismatch { expected, got } => {
                write!(f, "vocab_size {got} does not match expected {expected}")
            }
            Self::HiddenDimMismatch { expected, got } => {
                write!(f, "hidden_dim {got} does not match expected {expected}")
            }
            Self::ConfidenceHeadBlockSizeMismatch { expected, got } => {
                write!(
                    f,
                    "confidence_head block_size {got} does not match expected {expected}"
                )
            }
            Self::BigramVocabMismatch { expected, got } => {
                write!(
                    f,
                    "bigram_bias vocab_size {got} does not match expected {expected}"
                )
            }
            Self::DraftLogitsAllNonFinite => {
                write!(f, "draft logits are all non-finite (NaN); cannot argmax")
            }
            Self::DraftModelFailed(msg) => {
                write!(f, "draft model failed: {msg}")
            }
        }
    }
}

impl std::error::Error for DsparkError {}

/// bigram bias 実装が実装すべき最小 API
///
/// `DFlashParallelDraft::draft` に渡す bigram_bias 側から要求される trait
/// [`MarkovBigramBias`] (eager truncate、高速) と [`FullCountBigramBias`]
/// (完全 top-K、apply 時 sort) の両方が実装する
pub trait BigramBias {
    /// 宣言 vocab size
    fn vocab_size(&self) -> u32;
    /// prev の top-K bucket から `logits[next] += strength * ln_1p(count)` を加算する
    ///
    /// # Errors
    /// - `logits.len() != vocab_size` の場合
    /// - `prev >= vocab_size` の場合
    fn apply(&self, prev: TokenId, logits: &mut [f32], strength: f32) -> Result<(), DsparkError>;
}

/// Rank-K Markov bigram bias (DSpark vanilla rank=256)
///
/// 各 prev token に対して観測頻度 top-K の (next, count) を eager truncate で保持する
/// 完全 top-K が必要な場合は [`FullCountBigramBias`] を使う
#[cfg_attr(feature = "dspark-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct MarkovBigramBias {
    vocab_size: u32,
    rank: u32,
    entries: HashMap<TokenId, Vec<(TokenId, u32)>>,
}

impl MarkovBigramBias {
    /// 空の bias table を構築する
    ///
    /// # Errors
    /// `vocab_size` または `rank` が 0 の場合
    pub fn new(vocab_size: u32, rank: u32) -> Result<Self, DsparkError> {
        if vocab_size == 0 {
            return Err(DsparkError::ZeroVocab);
        }
        if rank == 0 {
            return Err(DsparkError::ZeroRank);
        }
        Ok(Self {
            vocab_size,
            rank,
            entries: HashMap::new(),
        })
    }

    /// token 列から一括構築する 隣接 pair を全部 observe する
    ///
    /// # Errors
    /// [`new`](Self::new) と [`observe_sequence`](Self::observe_sequence) と同じ
    pub fn from_sequence(
        vocab_size: u32,
        rank: u32,
        tokens: &[TokenId],
    ) -> Result<Self, DsparkError> {
        let mut bias = Self::new(vocab_size, rank)?;
        bias.observe_sequence(tokens)?;
        Ok(bias)
    }

    /// 宣言 vocab size
    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    /// top-K の K
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// 1 pair も観測していない場合 true
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// これまでに 1 度でも prev 側に出現した token の種類数
    pub fn observed_prev_count(&self) -> usize {
        self.entries.len()
    }

    /// 特定 prev の top-K bucket 現在サイズ (未観測 prev は 0)
    pub fn bucket_len(&self, prev: TokenId) -> usize {
        self.entries.get(&prev).map_or(0, Vec::len)
    }

    /// prev → next の bigram 頻度を +1 する
    ///
    /// 既存 (next, count) があれば count を saturating_add で 1 増やす
    /// 新規なら (next, 1) を push し、bucket サイズが rank を超えたら
    /// count 降順 → token id 昇順で sort して下位を drop する
    ///
    /// # Errors
    /// `prev` または `next` が vocab_size 以上の場合
    pub fn observe(&mut self, prev: TokenId, next: TokenId) -> Result<(), DsparkError> {
        if prev >= self.vocab_size {
            return Err(DsparkError::TokenOutOfVocab {
                token: prev,
                vocab_size: self.vocab_size,
            });
        }
        if next >= self.vocab_size {
            return Err(DsparkError::TokenOutOfVocab {
                token: next,
                vocab_size: self.vocab_size,
            });
        }
        let rank = self.rank as usize;
        let bucket = self
            .entries
            .entry(prev)
            .or_insert_with(|| Vec::with_capacity(rank + 1));
        if let Some(pos) = bucket.iter().position(|(t, _)| *t == next) {
            bucket[pos].1 = bucket[pos].1.saturating_add(1);
        } else {
            bucket.push((next, 1));
        }
        bucket.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        if bucket.len() > rank {
            bucket.truncate(rank);
        }
        Ok(())
    }

    /// token 列の隣接 pair を全部 observe する 最初の out-of-vocab で fail
    ///
    /// # Errors
    /// [`observe`](Self::observe) と同じ
    pub fn observe_sequence(&mut self, tokens: &[TokenId]) -> Result<(), DsparkError> {
        for pair in tokens.windows(2) {
            self.observe(pair[0], pair[1])?;
        }
        Ok(())
    }

    /// prev の top-K bucket に対して `logits[next] += strength * ln(1 + count)` を加算する
    ///
    /// - `strength = 0.0` は no-op で早期 return
    /// - `prev` が未観測なら no-op で早期 return
    ///
    /// # Errors
    /// - `logits.len() != vocab_size` の場合
    /// - `prev >= vocab_size` の場合
    pub fn apply(
        &self,
        prev: TokenId,
        logits: &mut [f32],
        strength: f32,
    ) -> Result<(), DsparkError> {
        if logits.len() != self.vocab_size as usize {
            return Err(DsparkError::LogitsLenMismatch {
                expected: self.vocab_size as usize,
                got: logits.len(),
            });
        }
        if prev >= self.vocab_size {
            return Err(DsparkError::TokenOutOfVocab {
                token: prev,
                vocab_size: self.vocab_size,
            });
        }
        if strength == 0.0 {
            return Ok(());
        }
        let Some(bucket) = self.entries.get(&prev) else {
            return Ok(());
        };
        for &(next, count) in bucket {
            let idx = next as usize;
            let bias = strength * (count as f32).ln_1p();
            logits[idx] += bias;
        }
        Ok(())
    }
}

impl BigramBias for MarkovBigramBias {
    fn vocab_size(&self) -> u32 {
        Self::vocab_size(self)
    }

    fn apply(&self, prev: TokenId, logits: &mut [f32], strength: f32) -> Result<(), DsparkError> {
        Self::apply(self, prev, logits, strength)
    }
}

/// Full-count Markov bigram bias
///
/// [`MarkovBigramBias`] の eager truncate 制約 (rank 到達後の tie で新規が drop される)
/// を解消するため、全ての観測 (prev, next, count) を保持し、apply 時に count 降順で
/// top-K を選ぶ apply コストは `O(N_unique_next * log N_unique_next)` per call で
/// [`MarkovBigramBias`] より遅いが、正確な top-K を保証する
///
/// storage: `HashMap<TokenId, HashMap<TokenId, u32>>` (sparse)
#[cfg_attr(feature = "dspark-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct FullCountBigramBias {
    vocab_size: u32,
    rank: u32,
    counts: HashMap<TokenId, HashMap<TokenId, u32>>,
}

impl FullCountBigramBias {
    /// 空の table を構築する
    ///
    /// # Errors
    /// `vocab_size` または `rank` が 0 の場合
    pub fn new(vocab_size: u32, rank: u32) -> Result<Self, DsparkError> {
        if vocab_size == 0 {
            return Err(DsparkError::ZeroVocab);
        }
        if rank == 0 {
            return Err(DsparkError::ZeroRank);
        }
        Ok(Self {
            vocab_size,
            rank,
            counts: HashMap::new(),
        })
    }

    /// token 列から一括構築する 隣接 pair を全部 observe する
    ///
    /// # Errors
    /// [`new`](Self::new) と [`observe_sequence`](Self::observe_sequence) と同じ
    pub fn from_sequence(
        vocab_size: u32,
        rank: u32,
        tokens: &[TokenId],
    ) -> Result<Self, DsparkError> {
        let mut bias = Self::new(vocab_size, rank)?;
        bias.observe_sequence(tokens)?;
        Ok(bias)
    }

    /// 宣言 vocab size
    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    /// apply 時に取り出す top-K の K
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// 1 pair も観測していない場合 true
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// prev 側に 1 度でも出現した token の種類数
    pub fn observed_prev_count(&self) -> usize {
        self.counts.len()
    }

    /// 特定 prev に対して観測された unique next の種類数 (未観測 prev は 0)
    pub fn unique_next_count(&self, prev: TokenId) -> usize {
        self.counts.get(&prev).map_or(0, HashMap::len)
    }

    /// prev → next の観測 count を返す (未観測は 0)
    pub fn count(&self, prev: TokenId, next: TokenId) -> u32 {
        self.counts
            .get(&prev)
            .and_then(|inner| inner.get(&next).copied())
            .unwrap_or(0)
    }

    /// prev → next の観測を +1 する (saturating)
    ///
    /// # Errors
    /// `prev` または `next` が vocab_size 以上の場合
    pub fn observe(&mut self, prev: TokenId, next: TokenId) -> Result<(), DsparkError> {
        if prev >= self.vocab_size {
            return Err(DsparkError::TokenOutOfVocab {
                token: prev,
                vocab_size: self.vocab_size,
            });
        }
        if next >= self.vocab_size {
            return Err(DsparkError::TokenOutOfVocab {
                token: next,
                vocab_size: self.vocab_size,
            });
        }
        let inner = self.counts.entry(prev).or_default();
        let c = inner.entry(next).or_insert(0_u32);
        *c = c.saturating_add(1);
        Ok(())
    }

    /// token 列の隣接 pair を全部 observe する 最初の out-of-vocab で fail
    ///
    /// # Errors
    /// [`observe`](Self::observe) と同じ
    pub fn observe_sequence(&mut self, tokens: &[TokenId]) -> Result<(), DsparkError> {
        for pair in tokens.windows(2) {
            self.observe(pair[0], pair[1])?;
        }
        Ok(())
    }

    /// prev の観測 counts から top-K を count 降順 → token id 昇順で選び、
    /// `logits[next] += strength * ln_1p(count)` を加算する
    ///
    /// - `strength = 0.0` は no-op で早期 return
    /// - `prev` が未観測なら no-op で早期 return
    ///
    /// # Errors
    /// - `logits.len() != vocab_size` の場合
    /// - `prev >= vocab_size` の場合
    pub fn apply(
        &self,
        prev: TokenId,
        logits: &mut [f32],
        strength: f32,
    ) -> Result<(), DsparkError> {
        if logits.len() != self.vocab_size as usize {
            return Err(DsparkError::LogitsLenMismatch {
                expected: self.vocab_size as usize,
                got: logits.len(),
            });
        }
        if prev >= self.vocab_size {
            return Err(DsparkError::TokenOutOfVocab {
                token: prev,
                vocab_size: self.vocab_size,
            });
        }
        if strength == 0.0 {
            return Ok(());
        }
        let Some(inner) = self.counts.get(&prev) else {
            return Ok(());
        };
        let rank = self.rank as usize;
        let mut ranked: Vec<(TokenId, u32)> = inner.iter().map(|(&t, &c)| (t, c)).collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.truncate(rank);
        for (next, count) in ranked {
            let idx = next as usize;
            let bias = strength * (count as f32).ln_1p();
            logits[idx] += bias;
        }
        Ok(())
    }
}

impl BigramBias for FullCountBigramBias {
    fn vocab_size(&self) -> u32 {
        Self::vocab_size(self)
    }

    fn apply(&self, prev: TokenId, logits: &mut [f32], strength: f32) -> Result<(), DsparkError> {
        Self::apply(self, prev, logits, strength)
    }
}

/// 位置別 confidence head (DSpark 3 要素の 2 番目)
///
/// draft position i ∈ [0, block_size) ごとに per-position 重み `w_i ∈ R^H` と bias `b_i ∈ R`
/// を持ち、`confidence_i = sigmoid(w_i · hidden_i + b_i) ∈ [0, 1]` を返す
/// BCE 学習は target 受理ラベル (y ∈ {0, 1}) で `loss = -[y·ln(p) + (1-y)·ln(1-p)]`、
/// sigmoid + BCE の canonical form `dL/dz = p - y` で SGD 1 step
///
/// zero-init 時は全 position で sigmoid(0) = 0.5 (uninformative prior)
#[cfg_attr(feature = "dspark-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct PositionConfidenceHead {
    block_size: u32,
    hidden_dim: u32,
    // row-major (block_size × hidden_dim)、index = pos * hidden_dim + h
    weights: Vec<f32>,
    // per-position bias、長さ = block_size
    biases: Vec<f32>,
}

impl PositionConfidenceHead {
    /// zero-init で新規構築 全 position の初期 confidence = 0.5
    ///
    /// # Errors
    /// `block_size` または `hidden_dim` が 0 の場合
    pub fn new(block_size: u32, hidden_dim: u32) -> Result<Self, DsparkError> {
        if block_size == 0 {
            return Err(DsparkError::ZeroBlockSize);
        }
        if hidden_dim == 0 {
            return Err(DsparkError::ZeroHiddenDim);
        }
        let total = block_size as usize * hidden_dim as usize;
        Ok(Self {
            block_size,
            hidden_dim,
            weights: vec![0.0_f32; total],
            biases: vec![0.0_f32; block_size as usize],
        })
    }

    /// [`new`](Self::new) の alias 明示的に zero-init を意図する場合に使う
    ///
    /// # Errors
    /// [`new`](Self::new) と同じ
    pub fn zeros(block_size: u32, hidden_dim: u32) -> Result<Self, DsparkError> {
        Self::new(block_size, hidden_dim)
    }

    /// 宣言 block size
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// 宣言 hidden dim
    pub fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }

    fn row_range(&self, position: u32) -> (usize, usize) {
        let h = self.hidden_dim as usize;
        let start = position as usize * h;
        (start, start + h)
    }

    fn validate_position(&self, position: u32) -> Result<(), DsparkError> {
        if position >= self.block_size {
            return Err(DsparkError::PositionOutOfRange {
                position,
                block_size: self.block_size,
            });
        }
        Ok(())
    }

    fn validate_hidden(&self, hidden: &[f32]) -> Result<(), DsparkError> {
        if hidden.len() != self.hidden_dim as usize {
            return Err(DsparkError::HiddenLenMismatch {
                expected: self.hidden_dim as usize,
                got: hidden.len(),
            });
        }
        Ok(())
    }

    /// 単一 position の confidence を返す
    ///
    /// # Errors
    /// - `position >= block_size` の場合
    /// - `hidden.len() != hidden_dim` の場合
    pub fn predict(&self, position: u32, hidden: &[f32]) -> Result<f32, DsparkError> {
        self.validate_position(position)?;
        self.validate_hidden(hidden)?;
        let (start, end) = self.row_range(position);
        let w = &self.weights[start..end];
        let b = self.biases[position as usize];
        let mut z = b;
        for (wi, hi) in w.iter().zip(hidden.iter()) {
            z += wi * hi;
        }
        Ok(stable_sigmoid(z))
    }

    /// block 全 position の confidence を一括で返す
    ///
    /// # Errors
    /// - `hidden_states.len() != block_size` の場合
    /// - どれかの `hidden_states[i].len()` が `hidden_dim` と不一致の場合
    pub fn predict_block(&self, hidden_states: &[&[f32]]) -> Result<Vec<f32>, DsparkError> {
        if hidden_states.len() != self.block_size as usize {
            return Err(DsparkError::BlockStatesCountMismatch {
                expected: self.block_size as usize,
                got: hidden_states.len(),
            });
        }
        let mut out = Vec::with_capacity(hidden_states.len());
        for (i, h) in hidden_states.iter().enumerate() {
            let conf = self.predict(i as u32, h)?;
            out.push(conf);
        }
        Ok(out)
    }

    /// SGD 1 step 学習 BCE loss を返す
    ///
    /// `label = true` → target y = 1、`false` → y = 0
    /// 更新式: `dL/dz = p - y`、`grad_w = (p - y) * hidden`、`grad_b = (p - y)`
    ///
    /// # Errors
    /// [`predict`](Self::predict) と同じ
    pub fn train_step(
        &mut self,
        position: u32,
        hidden: &[f32],
        label: bool,
        lr: f32,
    ) -> Result<f32, DsparkError> {
        self.validate_position(position)?;
        self.validate_hidden(hidden)?;
        let (start, end) = self.row_range(position);
        let b = self.biases[position as usize];
        let mut z = b;
        {
            let w = &self.weights[start..end];
            for (wi, hi) in w.iter().zip(hidden.iter()) {
                z += wi * hi;
            }
        }
        let p = stable_sigmoid(z);
        let y = if label { 1.0_f32 } else { 0.0_f32 };
        let dz = p - y;
        let w = &mut self.weights[start..end];
        for (wi, hi) in w.iter_mut().zip(hidden.iter()) {
            *wi -= lr * dz * hi;
        }
        self.biases[position as usize] -= lr * dz;
        Ok(stable_bce(p, y))
    }

    /// confidence 列 → accept/reject bool 列 (`conf >= threshold` で accept)
    ///
    /// NaN confidence は Rust の f32 `>=` semantics (NaN 比較は false) により自動 reject
    /// self を取らない associated function
    pub fn accept_mask(confidences: &[f32], threshold: f32) -> Vec<bool> {
        confidences.iter().map(|c| *c >= threshold).collect()
    }
}

// 数値安定 sigmoid: z >= 0 は 1/(1+exp(-z))、z < 0 は exp(z)/(1+exp(z))
fn stable_sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        let e = (-z).exp();
        1.0 / (1.0 + e)
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

// BCE loss: p を [eps, 1-eps] で clamp してから log (log(0) 回避)
fn stable_bce(p: f32, y: f32) -> f32 {
    let eps = 1e-7_f32;
    let p_clamped = p.clamp(eps, 1.0 - eps);
    -(y * p_clamped.ln() + (1.0 - y) * (1.0 - p_clamped).ln())
}

// NaN skip 版 argmax +inf は正当な argmax として通す、全 NaN で Err
fn argmax_finite(logits: &[f32]) -> Result<u32, DsparkError> {
    let mut best_idx: Option<u32> = None;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        if v > best_val {
            best_val = v;
            best_idx = Some(i as u32);
        }
    }
    best_idx.ok_or(DsparkError::DraftLogitsAllNonFinite)
}

/// Phase 6 拡張 config `Llama3Model::generate_speculative_dual_dspark` の第 9 引数
///
/// `advanced = None` の場合は Phase 5 と bit-exact 同一動作
/// `advanced = Some(cfg)` の場合は draft position ごとに hidden state を抽出し、
/// `PositionConfidenceHead::predict` で confidence を算出、`confidence < confidence_threshold`
/// なら draft loop を早期打切りして target verify 計算を省略する
///
/// 実 K3 accept length 実測には trained `PositionConfidenceHead` が必要 (accept/reject
/// label collection example が Phase 7 相当) 現状は zero-init head で全 position が
/// confidence = 0.5 になるため、threshold = 0.5 未満なら打切りなし、0.5 以上なら全打切りとなる
// serde derive は付けない — reference field を持つため serialize 不能
// (weight を配布したい場合は confidence_head 単体を serialize する)
#[derive(Debug, Clone, Copy)]
pub struct DsparkAdvancedConfig<'a> {
    /// draft position ごとの confidence 算出に使う位置別 sigmoid head
    ///
    /// `block_size >= spec_k` を満たす必要がある (method entry で検証)
    /// `hidden_dim` は `draft_model.config.hidden_dim` と一致 (method entry で検証)
    pub confidence_head: &'a PositionConfidenceHead,
    /// confidence がこの値未満なら draft 早期打切り
    ///
    /// `0.0` = 打切りなし (Phase 5 と同じ behavior)
    /// `1.0` = 全 draft 打切り (spec_k = 0 相当、target のみで生成)
    /// 通常は `0.3` 〜 `0.7` の範囲
    pub confidence_threshold: f32,
    /// draft model からどの layer の hidden state を抽出するか
    ///
    /// `None` = 最終層 (num_layers - 1)
    /// `Some(n)` = layer n の hidden state (RMSNorm 適用後)
    /// 範囲外の場合は最終層 fallback
    pub hidden_capture_layer: Option<usize>,
}

impl<'a> DsparkAdvancedConfig<'a> {
    /// 標準構成 confidence_threshold=0.5, hidden_capture_layer=None (最終層)
    pub fn new(confidence_head: &'a PositionConfidenceHead) -> Self {
        Self {
            confidence_head,
            confidence_threshold: 0.5,
            hidden_capture_layer: None,
        }
    }
}

/// bigram bias を条件付で logits に in-place 加算する DSpark llama3.rs 配線 (Phase 5) 用 helper
///
/// Phase 9: draft model 抽象 trait
///
/// [`Llama3Model::generate_speculative_dual_dspark`] と `generate_speculative_dual_collect_labels`
/// の draft 引数を型消去して、将来的な K3 / DeepSeek / Hy3 draft も同じ pipeline で
/// 使えるようにする trait 現状の impl は `Llama3Model` のみ (Phase 10 で K3 の KDA
/// snapshot 実装後に `KimiK3Model` も対応予定)
///
/// **KDA snapshot 問題**: KimiK3Model の KDA layer は recurrent state (`S_t = f(S_{t-1}, ...)`)
/// で positional cache を持たないため、`rollback_to` は state snapshot save/restore が
/// 必要 これが Phase 10 の主要作業
#[cfg(feature = "dspark")]
pub trait DraftBackend {
    /// 1 token forward、logits (長さ = vocab_size) を返す
    fn forward(&mut self, token_id: TokenId) -> Vec<f32>;

    /// forward + specified layer の hidden state を capture して返す
    ///
    /// `layer_idx = None` は最終層 (num_layers - 1)、範囲外は最終層 fallback
    fn forward_capture_hidden(
        &mut self,
        token_id: TokenId,
        layer_idx: Option<usize>,
    ) -> (Vec<f32>, Vec<f32>);

    /// 現在の KV cache 内 token 数 (最後の `clear_cache` / `rollback_to` 以降に append された数)
    fn seq_len(&self) -> usize;

    /// KV cache を指定位置まで rollback する speculative verify で reject された draft を破棄する
    fn rollback_to(&mut self, pos: usize);

    /// KV cache を全部クリアする (`rollback_to(0)` と等価だが reset 意図を明示)
    fn clear_cache(&mut self);

    /// vocab size
    fn vocab_size(&self) -> u32;

    /// hidden dim
    fn hidden_dim(&self) -> u32;

    /// draft model の layer 数 (spec_stats.draft_layers に転記)
    fn num_layers(&self) -> u32;
}

/// [`Llama3Model::generate_speculative_dual_collect_labels`] (Phase 7) が collect する 1 サンプル
///
/// vanilla speculative dual pipeline を走らせて各 draft position の
/// `(hidden_state, was_accepted)` を集める [`PositionConfidenceHead::train_step`] で
/// SGD BCE 学習する input
///
/// **注**: verify で reject された position 以降は verify されないため label 非付与
/// `bonus` は draft でなく main sample なので label 非付与
#[cfg_attr(feature = "dspark-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct DsparkLabelSample {
    /// 0..spec_k の draft position
    pub position: u32,
    /// draft model の hidden state (長さ = draft_model.config.hidden_dim)
    pub hidden: Vec<f32>,
    /// target model の verify で accept されたか (Leviathan sampling)
    pub was_accepted: bool,
}

/// `bigram_bias = None` または `strength = 0.0` の場合は no-op で `Ok(())` を返す
/// それ以外は `bigram_bias.apply(prev, logits, strength)` を呼び、error はそのまま伝播する
///
/// # Errors
/// [`BigramBias::apply`] と同じ
pub fn apply_bigram_bias_maybe(
    logits: &mut [f32],
    prev: TokenId,
    bigram_bias: Option<&dyn BigramBias>,
    strength: f32,
) -> Result<(), DsparkError> {
    if let Some(b) = bigram_bias {
        if strength != 0.0 {
            b.apply(prev, logits, strength)?;
        }
    }
    Ok(())
}

/// 外部 draft model からの 1 位置分の出力 (DFlashParallelDraft 契約)
#[cfg_attr(feature = "dspark-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct DraftPosition {
    /// draft model の hidden state (長さ = hidden_dim)
    pub hidden: Vec<f32>,
    /// vocab 上の logits (長さ = vocab_size)
    pub logits: Vec<f32>,
}

/// [`DFlashParallelDraft::draft`] の結果 block
#[cfg_attr(feature = "dspark-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct DraftBlock {
    /// argmax で選ばれた draft token 列 (長さ = block_size)
    pub tokens: Vec<TokenId>,
    /// [`PositionConfidenceHead`] が出した位置別 confidence (長さ = block_size)
    pub confidences: Vec<f32>,
    /// draft model が返した hidden state 列 (長さ = block_size)
    pub hidden_states: Vec<Vec<f32>>,
}

/// DFlash 並列 draft (DSpark 3 要素の 3 番目)
///
/// 外部 draft model callback を closure で受け取り、block_size 個の (hidden, logits) を
/// 並列に取得する 各 position の logits には optional の [`MarkovBigramBias`] を適用し、
/// argmax で token を確定、[`PositionConfidenceHead`] で位置別 confidence を算出する
///
/// llama3.rs との配線は本 struct のスコープ外 caller が `Fn(prefix, block_size) -> Result<Vec<DraftPosition>>`
/// を実装して渡す
#[cfg_attr(feature = "dspark-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct DFlashParallelDraft {
    block_size: u32,
    vocab_size: u32,
    hidden_dim: u32,
    bigram_strength: f32,
}

impl DFlashParallelDraft {
    /// 新規構築 `bigram_strength = 0.0` で MarkovBigramBias を無効化できる
    ///
    /// # Errors
    /// `block_size` / `vocab_size` / `hidden_dim` のいずれかが 0 の場合
    pub fn new(
        block_size: u32,
        vocab_size: u32,
        hidden_dim: u32,
        bigram_strength: f32,
    ) -> Result<Self, DsparkError> {
        if block_size == 0 {
            return Err(DsparkError::ZeroBlockSize);
        }
        if vocab_size == 0 {
            return Err(DsparkError::ZeroVocab);
        }
        if hidden_dim == 0 {
            return Err(DsparkError::ZeroHiddenDim);
        }
        Ok(Self {
            block_size,
            vocab_size,
            hidden_dim,
            bigram_strength,
        })
    }

    /// 宣言 block size
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// 宣言 vocab size
    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    /// 宣言 hidden dim
    pub fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }

    /// 現在の bigram strength (0.0 で bigram apply を skip)
    pub fn bigram_strength(&self) -> f32 {
        self.bigram_strength
    }

    /// bigram strength を更新する 値の finite 性は検証しない
    pub fn set_bigram_strength(&mut self, strength: f32) {
        self.bigram_strength = strength;
    }

    /// prefix と外部 draft callback を受けて DraftBlock を返す
    ///
    /// アルゴリズム:
    /// 1. prefix 非空 + confidence_head / bigram 一致検証
    /// 2. `draft_fn(prefix, block_size)` 呼出、returned positions の shape 検証
    /// 3. 各 position i:
    ///    - prev = if i == 0 { prefix.last() } else { tokens\[i-1\] }
    ///    - bigram_bias 提供 && strength != 0 → `bigram.apply(prev, &mut logits, strength)`
    ///    - `token = argmax_finite(logits)`
    ///    - `conf = confidence_head.predict(i, &hidden)`
    /// 4. `DraftBlock { tokens, confidences, hidden_states }` を返却
    ///
    /// # Errors
    /// - `EmptyPrefix`: prefix が空
    /// - `ConfidenceHeadBlockSizeMismatch` / `HiddenDimMismatch`: 構成不一致
    /// - `BigramVocabMismatch`: bigram_bias.vocab_size 不一致
    /// - `DraftModelBlockSizeMismatch` / `VocabSizeMismatch` / `HiddenDimMismatch`: draft_fn output shape 不一致
    /// - `DraftLogitsAllNonFinite`: 全 NaN logits で argmax 不能
    /// - `DraftModelFailed(msg)`: draft_fn 自身が返したエラー (`draft_fn` が返す `DsparkError` はそのまま伝播)
    pub fn draft<F>(
        &self,
        prefix: &[TokenId],
        bigram_bias: Option<&dyn BigramBias>,
        confidence_head: &PositionConfidenceHead,
        draft_fn: F,
    ) -> Result<DraftBlock, DsparkError>
    where
        F: FnOnce(&[TokenId], u32) -> Result<Vec<DraftPosition>, DsparkError>,
    {
        // (1) prefix 検証
        let last_prefix_token = *prefix.last().ok_or(DsparkError::EmptyPrefix)?;

        // (2) config 一致検証
        if confidence_head.block_size() != self.block_size {
            return Err(DsparkError::ConfidenceHeadBlockSizeMismatch {
                expected: self.block_size,
                got: confidence_head.block_size(),
            });
        }
        if confidence_head.hidden_dim() != self.hidden_dim {
            return Err(DsparkError::HiddenDimMismatch {
                expected: self.hidden_dim,
                got: confidence_head.hidden_dim(),
            });
        }
        if let Some(b) = bigram_bias {
            if b.vocab_size() != self.vocab_size {
                return Err(DsparkError::BigramVocabMismatch {
                    expected: self.vocab_size,
                    got: b.vocab_size(),
                });
            }
        }

        // (3) draft_fn 呼出
        let mut positions = draft_fn(prefix, self.block_size)?;
        if positions.len() != self.block_size as usize {
            return Err(DsparkError::DraftModelBlockSizeMismatch {
                expected: self.block_size as usize,
                got: positions.len(),
            });
        }
        for pos in &positions {
            if pos.hidden.len() != self.hidden_dim as usize {
                return Err(DsparkError::HiddenDimMismatch {
                    expected: self.hidden_dim,
                    got: pos.hidden.len() as u32,
                });
            }
            if pos.logits.len() != self.vocab_size as usize {
                return Err(DsparkError::VocabSizeMismatch {
                    expected: self.vocab_size,
                    got: pos.logits.len() as u32,
                });
            }
        }

        // (4) 各 position の argmax + confidence
        let mut tokens = Vec::with_capacity(self.block_size as usize);
        let mut confidences = Vec::with_capacity(self.block_size as usize);
        let mut hidden_states = Vec::with_capacity(self.block_size as usize);

        let apply_bigram = bigram_bias.is_some() && self.bigram_strength != 0.0;

        for (i, pos) in positions.iter_mut().enumerate() {
            let prev = if i == 0 {
                last_prefix_token
            } else {
                tokens[i - 1]
            };
            if apply_bigram {
                if let Some(b) = bigram_bias {
                    b.apply(prev, &mut pos.logits, self.bigram_strength)?;
                }
            }
            let token = argmax_finite(&pos.logits)?;
            let conf = confidence_head.predict(i as u32, &pos.hidden)?;
            tokens.push(token);
            confidences.push(conf);
            hidden_states.push(core::mem::take(&mut pos.hidden));
        }

        Ok(DraftBlock {
            tokens,
            confidences,
            hidden_states,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BigramBias, DFlashParallelDraft, DraftBlock, DraftPosition, DsparkError,
        FullCountBigramBias, MarkovBigramBias, PositionConfidenceHead, TokenId,
    };

    #[test]
    fn new_rejects_zero_vocab() {
        let err = MarkovBigramBias::new(0, 256).unwrap_err();
        assert_eq!(err, DsparkError::ZeroVocab);
    }

    #[test]
    fn new_rejects_zero_rank() {
        let err = MarkovBigramBias::new(100, 0).unwrap_err();
        assert_eq!(err, DsparkError::ZeroRank);
    }

    #[test]
    fn new_defaults_are_empty() {
        let bias = MarkovBigramBias::new(100, 256).expect("valid");
        assert_eq!(bias.vocab_size(), 100);
        assert_eq!(bias.rank(), 256);
        assert!(bias.is_empty());
        assert_eq!(bias.observed_prev_count(), 0);
        assert_eq!(bias.bucket_len(0), 0);
    }

    #[test]
    fn observe_rejects_prev_out_of_vocab() {
        let mut bias = MarkovBigramBias::new(10, 4).expect("valid");
        let err = bias.observe(10, 5).unwrap_err();
        assert_eq!(
            err,
            DsparkError::TokenOutOfVocab {
                token: 10,
                vocab_size: 10
            }
        );
    }

    #[test]
    fn observe_rejects_next_out_of_vocab() {
        let mut bias = MarkovBigramBias::new(10, 4).expect("valid");
        let err = bias.observe(5, 99).unwrap_err();
        assert_eq!(
            err,
            DsparkError::TokenOutOfVocab {
                token: 99,
                vocab_size: 10
            }
        );
    }

    #[test]
    fn observe_increments_count() {
        let mut bias = MarkovBigramBias::new(10, 4).expect("valid");
        bias.observe(1, 2).expect("valid");
        bias.observe(1, 2).expect("valid");
        bias.observe(1, 2).expect("valid");
        // apply で count 反映を確認 count=3 → bias = 1.0 * ln(4)
        let mut logits = vec![0.0_f32; 10];
        bias.apply(1, &mut logits, 1.0).expect("valid");
        let expected = 4.0_f32.ln();
        assert!(
            (logits[2] - expected).abs() < 1e-6,
            "logits[2] = {}",
            logits[2]
        );
        assert_eq!(bias.bucket_len(1), 1);
    }

    #[test]
    fn observe_truncates_to_rank() {
        // 高頻度側を先に投入 → eager truncate でも count 差が保たれる形で top-K が確定する
        let mut bias = MarkovBigramBias::new(20, 3).expect("valid");
        for _ in 0..10 {
            bias.observe(0, 11).expect("valid");
        }
        for _ in 0..8 {
            bias.observe(0, 12).expect("valid");
        }
        for _ in 0..6 {
            bias.observe(0, 13).expect("valid");
        }
        // ここまでで bucket = [(11,10),(12,8),(13,6)] (rank=3 埋)
        // 以降の低頻度観測は eager truncate で全部 drop される
        for _ in 0..4 {
            bias.observe(0, 14).expect("valid");
        }
        for _ in 0..2 {
            bias.observe(0, 15).expect("valid");
        }
        assert_eq!(bias.bucket_len(0), 3);
        let mut logits = vec![0.0_f32; 20];
        bias.apply(0, &mut logits, 1.0).expect("valid");
        assert!(logits[14] == 0.0, "14 must be truncated (eager)");
        assert!(logits[15] == 0.0, "15 must be truncated (eager)");
        assert!(logits[11] > logits[12], "11 (count 10) > 12 (count 8)");
        assert!(logits[12] > logits[13], "12 (count 8) > 13 (count 6)");
    }

    #[test]
    fn observe_eager_truncate_drops_late_tied_arrivals() {
        // eager truncate の既知制約: rank 到達後の同 count 新規は id tie で drop される
        // 完全 top-K が必要なら Phase 2 full-count sketch を待つ
        let mut bias = MarkovBigramBias::new(20, 2).expect("valid");
        bias.observe(0, 10).expect("valid");
        bias.observe(0, 11).expect("valid");
        // bucket 満杯 [(10,1),(11,1)]
        // 12 を何度 observe しても push 直後の tie で id 大 (12) が drop される
        for _ in 0..5 {
            bias.observe(0, 12).expect("valid");
        }
        assert_eq!(bias.bucket_len(0), 2);
        let mut logits = vec![0.0_f32; 20];
        bias.apply(0, &mut logits, 1.0).expect("valid");
        assert!(
            logits[12] == 0.0,
            "12 dropped by eager truncate (known limitation)"
        );
    }

    #[test]
    fn observe_sequence_from_stream() {
        let mut bias = MarkovBigramBias::new(20, 8).expect("valid");
        // 1 2 3 1 2 3 → pairs: (1,2), (2,3), (3,1), (1,2), (2,3)
        bias.observe_sequence(&[1, 2, 3, 1, 2, 3]).expect("valid");
        assert_eq!(bias.bucket_len(1), 1); // next: {2}
        assert_eq!(bias.bucket_len(2), 1); // next: {3}
        assert_eq!(bias.bucket_len(3), 1); // next: {1}
        assert_eq!(bias.observed_prev_count(), 3);
    }

    #[test]
    fn observe_sort_tie_break_by_token_id() {
        let mut bias = MarkovBigramBias::new(20, 4).expect("valid");
        // 同 count で 3 個 push
        bias.observe(0, 15).expect("valid");
        bias.observe(0, 10).expect("valid");
        bias.observe(0, 12).expect("valid");
        // count は全部 1 → token id 昇順で並ぶ (10, 12, 15)
        let mut logits = vec![0.0_f32; 20];
        bias.apply(0, &mut logits, 1.0).expect("valid");
        // 全部同 bias なので値の順位確認は無意味、bucket_len だけ確認
        assert_eq!(bias.bucket_len(0), 3);
    }

    #[test]
    fn from_sequence_constructor() {
        let bias = MarkovBigramBias::from_sequence(20, 8, &[1, 2, 3, 4]).expect("valid");
        assert_eq!(bias.observed_prev_count(), 3);
        assert_eq!(bias.bucket_len(1), 1);
        assert_eq!(bias.bucket_len(2), 1);
        assert_eq!(bias.bucket_len(3), 1);
    }

    #[test]
    fn apply_strength_zero_is_noop() {
        let mut bias = MarkovBigramBias::new(10, 4).expect("valid");
        bias.observe(1, 2).expect("valid");
        let mut logits = vec![0.5_f32; 10];
        bias.apply(1, &mut logits, 0.0).expect("valid");
        assert!(logits.iter().all(|&v| (v - 0.5).abs() < 1e-9));
    }

    #[test]
    fn apply_unobserved_prev_is_noop() {
        let bias = MarkovBigramBias::new(10, 4).expect("valid");
        let mut logits = vec![0.0_f32; 10];
        bias.apply(5, &mut logits, 1.0).expect("valid");
        assert!(logits.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn apply_rejects_logits_len_mismatch() {
        let bias = MarkovBigramBias::new(10, 4).expect("valid");
        let mut logits = vec![0.0_f32; 5];
        let err = bias.apply(0, &mut logits, 1.0).unwrap_err();
        assert_eq!(
            err,
            DsparkError::LogitsLenMismatch {
                expected: 10,
                got: 5
            }
        );
    }

    #[test]
    fn apply_rejects_prev_out_of_vocab() {
        let bias = MarkovBigramBias::new(10, 4).expect("valid");
        let mut logits = vec![0.0_f32; 10];
        let err = bias.apply(10, &mut logits, 1.0).unwrap_err();
        assert_eq!(
            err,
            DsparkError::TokenOutOfVocab {
                token: 10,
                vocab_size: 10
            }
        );
    }

    #[test]
    fn dspark_error_display() {
        let e1 = DsparkError::TokenOutOfVocab {
            token: 5,
            vocab_size: 3,
        };
        assert_eq!(format!("{e1}"), "token id 5 exceeds vocab size 3");
        let e2 = DsparkError::LogitsLenMismatch {
            expected: 10,
            got: 5,
        };
        assert_eq!(
            format!("{e2}"),
            "logits length 5 does not match vocab size 10"
        );
        assert_eq!(
            format!("{}", DsparkError::ZeroRank),
            "rank must be non-zero"
        );
        assert_eq!(
            format!("{}", DsparkError::ZeroVocab),
            "vocab size must be non-zero"
        );
    }

    // ---- PositionConfidenceHead tests ----

    #[test]
    fn confidence_head_new_rejects_zero_block_size() {
        let err = PositionConfidenceHead::new(0, 16).unwrap_err();
        assert_eq!(err, DsparkError::ZeroBlockSize);
    }

    #[test]
    fn confidence_head_new_rejects_zero_hidden_dim() {
        let err = PositionConfidenceHead::new(7, 0).unwrap_err();
        assert_eq!(err, DsparkError::ZeroHiddenDim);
    }

    #[test]
    fn confidence_head_zero_init_confidence_is_half() {
        let head = PositionConfidenceHead::new(7, 4).expect("valid");
        assert_eq!(head.block_size(), 7);
        assert_eq!(head.hidden_dim(), 4);
        for pos in 0..7 {
            let conf = head.predict(pos, &[1.0, -2.0, 3.5, 0.0]).expect("valid");
            assert!(
                (conf - 0.5).abs() < 1e-6,
                "position {pos} confidence should be 0.5 at zero-init, got {conf}"
            );
        }
    }

    #[test]
    fn confidence_head_zeros_alias_works() {
        let head = PositionConfidenceHead::zeros(3, 2).expect("valid");
        let conf = head.predict(1, &[0.5, -0.5]).expect("valid");
        assert!((conf - 0.5).abs() < 1e-6);
    }

    #[test]
    fn confidence_head_predict_rejects_hidden_len_mismatch() {
        let head = PositionConfidenceHead::new(4, 8).expect("valid");
        let err = head.predict(0, &[0.0; 4]).unwrap_err();
        assert_eq!(
            err,
            DsparkError::HiddenLenMismatch {
                expected: 8,
                got: 4
            }
        );
    }

    #[test]
    fn confidence_head_predict_rejects_position_out_of_range() {
        let head = PositionConfidenceHead::new(4, 8).expect("valid");
        let err = head.predict(4, &[0.0; 8]).unwrap_err();
        assert_eq!(
            err,
            DsparkError::PositionOutOfRange {
                position: 4,
                block_size: 4
            }
        );
    }

    #[test]
    fn confidence_head_predict_block_rejects_state_count_mismatch() {
        let head = PositionConfidenceHead::new(3, 4).expect("valid");
        let h = [0.0_f32; 4];
        let states: Vec<&[f32]> = vec![&h, &h];
        let err = head.predict_block(&states).unwrap_err();
        assert_eq!(
            err,
            DsparkError::BlockStatesCountMismatch {
                expected: 3,
                got: 2
            }
        );
    }

    #[test]
    fn confidence_head_predict_block_returns_all_half_at_init() {
        let head = PositionConfidenceHead::new(5, 3).expect("valid");
        let h = [1.0_f32, -1.0, 2.0];
        let states: Vec<&[f32]> = vec![&h, &h, &h, &h, &h];
        let confs = head.predict_block(&states).expect("valid");
        assert_eq!(confs.len(), 5);
        for c in confs {
            assert!((c - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn confidence_head_train_step_returns_bce_loss_ln2_at_init() {
        // p = 0.5、y = 1 → loss = -ln(0.5) = ln(2) ≈ 0.6931
        let mut head = PositionConfidenceHead::new(2, 3).expect("valid");
        let loss = head
            .train_step(0, &[0.1, 0.2, 0.3], true, 0.01)
            .expect("valid");
        let expected = 2.0_f32.ln();
        assert!(
            (loss - expected).abs() < 1e-5,
            "loss = {loss}, expected {expected}"
        );
    }

    #[test]
    fn confidence_head_train_step_moves_toward_label_positive() {
        let mut head = PositionConfidenceHead::new(2, 3).expect("valid");
        let hidden = [1.0_f32, 1.0, 1.0];
        let before = head.predict(0, &hidden).expect("valid");
        for _ in 0..50 {
            let _ = head.train_step(0, &hidden, true, 0.1).expect("valid");
        }
        let after = head.predict(0, &hidden).expect("valid");
        assert!(
            after > before + 0.1,
            "confidence must increase toward label=1: before={before}, after={after}"
        );
        assert!(after <= 1.0);
    }

    #[test]
    fn confidence_head_train_step_moves_toward_label_negative() {
        let mut head = PositionConfidenceHead::new(2, 3).expect("valid");
        let hidden = [1.0_f32, 1.0, 1.0];
        let before = head.predict(0, &hidden).expect("valid");
        for _ in 0..50 {
            let _ = head.train_step(0, &hidden, false, 0.1).expect("valid");
        }
        let after = head.predict(0, &hidden).expect("valid");
        assert!(
            after < before - 0.1,
            "confidence must decrease toward label=0: before={before}, after={after}"
        );
        assert!(after >= 0.0);
    }

    #[test]
    fn confidence_head_train_step_isolates_position() {
        // position 0 の更新が position 1 に影響しないこと
        let mut head = PositionConfidenceHead::new(3, 2).expect("valid");
        let hidden = [1.0_f32, 1.0];
        for _ in 0..30 {
            let _ = head.train_step(0, &hidden, true, 0.1).expect("valid");
        }
        let conf0 = head.predict(0, &hidden).expect("valid");
        let conf1 = head.predict(1, &hidden).expect("valid");
        let conf2 = head.predict(2, &hidden).expect("valid");
        assert!(conf0 > 0.7);
        assert!((conf1 - 0.5).abs() < 1e-6);
        assert!((conf2 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn confidence_head_accept_mask_threshold_boundary() {
        let confidences = [0.3_f32, 0.5, 0.7, 0.9];
        let mask = PositionConfidenceHead::accept_mask(&confidences, 0.5);
        assert_eq!(mask, vec![false, true, true, true]);
    }

    #[test]
    fn confidence_head_accept_mask_nan_confidence_is_false() {
        let confidences = [f32::NAN, 0.6];
        let mask = PositionConfidenceHead::accept_mask(&confidences, 0.5);
        assert_eq!(mask, vec![false, true]);
    }

    #[test]
    fn confidence_head_numerical_stability_large_positive() {
        // 巨大 hidden で sigmoid が 1.0 に飽和、panic せず [0,1] 範囲
        let mut head = PositionConfidenceHead::new(1, 1).expect("valid");
        // bias を大きくして z ≈ +1e5
        head.biases[0] = 1.0e5;
        let conf = head.predict(0, &[0.0]).expect("valid");
        assert!(conf >= 0.0 && conf <= 1.0);
        assert!(conf > 0.999);
    }

    #[test]
    fn confidence_head_numerical_stability_large_negative() {
        let mut head = PositionConfidenceHead::new(1, 1).expect("valid");
        head.biases[0] = -1.0e5;
        let conf = head.predict(0, &[0.0]).expect("valid");
        assert!(conf >= 0.0 && conf <= 1.0);
        assert!(conf < 1.0e-3);
    }

    #[test]
    fn dspark_error_display_confidence_variants() {
        assert_eq!(
            format!("{}", DsparkError::ZeroBlockSize),
            "block_size must be non-zero"
        );
        assert_eq!(
            format!("{}", DsparkError::ZeroHiddenDim),
            "hidden_dim must be non-zero"
        );
        assert_eq!(
            format!(
                "{}",
                DsparkError::HiddenLenMismatch {
                    expected: 8,
                    got: 4
                }
            ),
            "hidden length 4 does not match hidden_dim 8"
        );
        assert_eq!(
            format!(
                "{}",
                DsparkError::PositionOutOfRange {
                    position: 7,
                    block_size: 5
                }
            ),
            "position 7 exceeds block_size 5"
        );
        assert_eq!(
            format!(
                "{}",
                DsparkError::BlockStatesCountMismatch {
                    expected: 7,
                    got: 3
                }
            ),
            "hidden_states count 3 does not match block_size 7"
        );
    }

    // ---- DFlashParallelDraft tests ----

    // vocab_size=8, hidden_dim=4, block_size=3 の draft position を生成する helper
    // pos i の logits は index=(i * 2) が最大、それ以外は 0.0 (bigram なしで argmax=i*2)
    fn mock_positions(block_size: u32) -> Vec<DraftPosition> {
        let vocab = 8usize;
        let hidden = 4usize;
        let mut out = Vec::with_capacity(block_size as usize);
        for i in 0..block_size {
            let mut logits = vec![0.0_f32; vocab];
            let target = (i as usize * 2) % vocab;
            logits[target] = 5.0;
            let h = vec![0.1_f32; hidden];
            out.push(DraftPosition { hidden: h, logits });
        }
        out
    }

    #[test]
    fn dfp_new_rejects_zero_block_size() {
        let err = DFlashParallelDraft::new(0, 100, 4, 1.0).unwrap_err();
        assert_eq!(err, DsparkError::ZeroBlockSize);
    }

    #[test]
    fn dfp_new_rejects_zero_vocab() {
        let err = DFlashParallelDraft::new(3, 0, 4, 1.0).unwrap_err();
        assert_eq!(err, DsparkError::ZeroVocab);
    }

    #[test]
    fn dfp_new_rejects_zero_hidden() {
        let err = DFlashParallelDraft::new(3, 100, 0, 1.0).unwrap_err();
        assert_eq!(err, DsparkError::ZeroHiddenDim);
    }

    #[test]
    fn dfp_getters_and_setter() {
        let mut dfp = DFlashParallelDraft::new(7, 128, 16, 0.5).expect("valid");
        assert_eq!(dfp.block_size(), 7);
        assert_eq!(dfp.vocab_size(), 128);
        assert_eq!(dfp.hidden_dim(), 16);
        assert!((dfp.bigram_strength() - 0.5).abs() < 1e-9);
        dfp.set_bigram_strength(2.0);
        assert!((dfp.bigram_strength() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn dfp_draft_rejects_empty_prefix() {
        let dfp = DFlashParallelDraft::new(3, 8, 4, 0.0).expect("valid");
        let head = PositionConfidenceHead::new(3, 4).expect("valid");
        let err = dfp
            .draft(&[], None, &head, |_p, bs| Ok(mock_positions(bs)))
            .unwrap_err();
        assert_eq!(err, DsparkError::EmptyPrefix);
    }

    #[test]
    fn dfp_draft_rejects_confidence_head_block_mismatch() {
        let dfp = DFlashParallelDraft::new(3, 8, 4, 0.0).expect("valid");
        let head = PositionConfidenceHead::new(5, 4).expect("valid");
        let prefix: [TokenId; 1] = [1];
        let err = dfp
            .draft(&prefix, None, &head, |_p, bs| Ok(mock_positions(bs)))
            .unwrap_err();
        assert_eq!(
            err,
            DsparkError::ConfidenceHeadBlockSizeMismatch {
                expected: 3,
                got: 5
            }
        );
    }

    #[test]
    fn dfp_draft_rejects_confidence_head_hidden_mismatch() {
        let dfp = DFlashParallelDraft::new(3, 8, 4, 0.0).expect("valid");
        let head = PositionConfidenceHead::new(3, 8).expect("valid");
        let prefix: [TokenId; 1] = [1];
        let err = dfp
            .draft(&prefix, None, &head, |_p, bs| Ok(mock_positions(bs)))
            .unwrap_err();
        assert_eq!(
            err,
            DsparkError::HiddenDimMismatch {
                expected: 4,
                got: 8
            }
        );
    }

    #[test]
    fn dfp_draft_rejects_bigram_vocab_mismatch() {
        let dfp = DFlashParallelDraft::new(3, 8, 4, 1.0).expect("valid");
        let head = PositionConfidenceHead::new(3, 4).expect("valid");
        let bigram = MarkovBigramBias::new(16, 4).expect("valid");
        let prefix: [TokenId; 1] = [1];
        let err = dfp
            .draft(&prefix, Some(&bigram), &head, |_p, bs| {
                Ok(mock_positions(bs))
            })
            .unwrap_err();
        assert_eq!(
            err,
            DsparkError::BigramVocabMismatch {
                expected: 8,
                got: 16
            }
        );
    }

    #[test]
    fn dfp_draft_rejects_draft_fn_position_count() {
        let dfp = DFlashParallelDraft::new(3, 8, 4, 0.0).expect("valid");
        let head = PositionConfidenceHead::new(3, 4).expect("valid");
        let prefix: [TokenId; 1] = [1];
        let err = dfp
            .draft(&prefix, None, &head, |_p, _bs| Ok(mock_positions(2)))
            .unwrap_err();
        assert_eq!(
            err,
            DsparkError::DraftModelBlockSizeMismatch {
                expected: 3,
                got: 2
            }
        );
    }

    #[test]
    fn dfp_draft_rejects_draft_fn_hidden_len_mismatch() {
        let dfp = DFlashParallelDraft::new(3, 8, 4, 0.0).expect("valid");
        let head = PositionConfidenceHead::new(3, 4).expect("valid");
        let prefix: [TokenId; 1] = [1];
        let err = dfp
            .draft(&prefix, None, &head, |_p, bs| {
                let mut ps = mock_positions(bs);
                ps[1].hidden = vec![0.0; 2]; // 意図的に不一致
                Ok(ps)
            })
            .unwrap_err();
        assert_eq!(
            err,
            DsparkError::HiddenDimMismatch {
                expected: 4,
                got: 2
            }
        );
    }

    #[test]
    fn dfp_draft_rejects_draft_fn_logits_len_mismatch() {
        let dfp = DFlashParallelDraft::new(3, 8, 4, 0.0).expect("valid");
        let head = PositionConfidenceHead::new(3, 4).expect("valid");
        let prefix: [TokenId; 1] = [1];
        let err = dfp
            .draft(&prefix, None, &head, |_p, bs| {
                let mut ps = mock_positions(bs);
                ps[0].logits = vec![0.0; 3]; // 意図的に不一致
                Ok(ps)
            })
            .unwrap_err();
        assert_eq!(
            err,
            DsparkError::VocabSizeMismatch {
                expected: 8,
                got: 3
            }
        );
    }

    #[test]
    fn dfp_draft_propagates_draft_fn_error() {
        let dfp = DFlashParallelDraft::new(3, 8, 4, 0.0).expect("valid");
        let head = PositionConfidenceHead::new(3, 4).expect("valid");
        let prefix: [TokenId; 1] = [1];
        let err = dfp
            .draft(&prefix, None, &head, |_p, _bs| {
                Err(DsparkError::DraftModelFailed(
                    "external draft crashed".to_string(),
                ))
            })
            .unwrap_err();
        assert_eq!(
            err,
            DsparkError::DraftModelFailed("external draft crashed".to_string())
        );
    }

    #[test]
    fn dfp_draft_rejects_all_nan_logits() {
        let dfp = DFlashParallelDraft::new(3, 8, 4, 0.0).expect("valid");
        let head = PositionConfidenceHead::new(3, 4).expect("valid");
        let prefix: [TokenId; 1] = [1];
        let err = dfp
            .draft(&prefix, None, &head, |_p, bs| {
                let mut ps = mock_positions(bs);
                for p in &mut ps {
                    for v in &mut p.logits {
                        *v = f32::NAN;
                    }
                }
                Ok(ps)
            })
            .unwrap_err();
        assert_eq!(err, DsparkError::DraftLogitsAllNonFinite);
    }

    #[test]
    fn dfp_draft_argmax_and_half_confidence_at_init() {
        let dfp = DFlashParallelDraft::new(3, 8, 4, 0.0).expect("valid");
        let head = PositionConfidenceHead::new(3, 4).expect("valid");
        let prefix: [TokenId; 1] = [1];
        let block = dfp
            .draft(&prefix, None, &head, |_p, bs| Ok(mock_positions(bs)))
            .expect("valid");
        // mock_positions で pos i の argmax = i*2 (mod vocab)
        assert_eq!(block.tokens, vec![0, 2, 4]);
        assert_eq!(block.confidences.len(), 3);
        for c in &block.confidences {
            assert!((c - 0.5).abs() < 1e-6);
        }
        assert_eq!(block.hidden_states.len(), 3);
        assert_eq!(block.hidden_states[0].len(), 4);
    }

    #[test]
    fn dfp_draft_applies_bigram_bias_shifts_argmax() {
        // vocab=8、position 0 で mock_positions は logits[0] = 5.0 (他 0.0)
        // bigram を prev=1 → next=3 で count=10 学習し strength=100 apply すれば
        // logits[3] = 0.0 + 100 * ln_1p(10) = 100 * 2.398 ≈ 239.8 > 5.0 で argmax=3 に変わる
        let dfp = DFlashParallelDraft::new(1, 8, 4, 100.0).expect("valid");
        let head = PositionConfidenceHead::new(1, 4).expect("valid");
        let mut bigram = MarkovBigramBias::new(8, 4).expect("valid");
        for _ in 0..10 {
            bigram.observe(1, 3).expect("valid");
        }
        let prefix: [TokenId; 1] = [1];
        let block = dfp
            .draft(&prefix, Some(&bigram), &head, |_p, bs| {
                Ok(mock_positions(bs))
            })
            .expect("valid");
        assert_eq!(
            block.tokens,
            vec![3],
            "bigram bias should shift argmax to 3"
        );
    }

    #[test]
    fn dfp_draft_ignores_bigram_when_strength_zero() {
        // strength=0.0 なら bigram_bias があっても apply されず argmax は mock 通り 0
        let dfp = DFlashParallelDraft::new(1, 8, 4, 0.0).expect("valid");
        let head = PositionConfidenceHead::new(1, 4).expect("valid");
        let mut bigram = MarkovBigramBias::new(8, 4).expect("valid");
        for _ in 0..10 {
            bigram.observe(1, 3).expect("valid");
        }
        let prefix: [TokenId; 1] = [1];
        let block = dfp
            .draft(&prefix, Some(&bigram), &head, |_p, bs| {
                Ok(mock_positions(bs))
            })
            .expect("valid");
        assert_eq!(block.tokens, vec![0], "strength=0 should skip bigram apply");
    }

    #[test]
    fn dfp_draft_uses_previous_drafted_token_as_bigram_prev() {
        // block_size=2、pos 0 draft = 0、pos 1 draft は prev=0 で bigram を引く
        // bigram に (0, 6) を count 10 積み、strength=100 で pos 1 の argmax が 6 に変わることを確認
        let dfp = DFlashParallelDraft::new(2, 8, 4, 100.0).expect("valid");
        let head = PositionConfidenceHead::new(2, 4).expect("valid");
        let mut bigram = MarkovBigramBias::new(8, 4).expect("valid");
        for _ in 0..10 {
            bigram.observe(0, 6).expect("valid");
        }
        // prev token は vocab 内 (< 8) で bigram に未登録の値
        let prefix: [TokenId; 1] = [7];
        let block = dfp
            .draft(&prefix, Some(&bigram), &head, |_p, bs| {
                Ok(mock_positions(bs))
            })
            .expect("valid");
        // pos 0: prev=7、bigram で 7 は未観測 → apply no-op → argmax = 0
        // pos 1: prev=0 (drafted[0])、bigram で (0,6) が引かれる → argmax = 6
        assert_eq!(block.tokens, vec![0, 6]);
    }

    #[test]
    fn dspark_error_display_draft_variants() {
        assert_eq!(
            format!("{}", DsparkError::EmptyPrefix),
            "prefix must be non-empty"
        );
        assert_eq!(
            format!(
                "{}",
                DsparkError::DraftModelBlockSizeMismatch {
                    expected: 7,
                    got: 5
                }
            ),
            "draft model returned 5 positions but expected 7"
        );
        assert_eq!(
            format!(
                "{}",
                DsparkError::VocabSizeMismatch {
                    expected: 8,
                    got: 4
                }
            ),
            "vocab_size 4 does not match expected 8"
        );
        assert_eq!(
            format!(
                "{}",
                DsparkError::HiddenDimMismatch {
                    expected: 16,
                    got: 8
                }
            ),
            "hidden_dim 8 does not match expected 16"
        );
        assert_eq!(
            format!(
                "{}",
                DsparkError::ConfidenceHeadBlockSizeMismatch {
                    expected: 7,
                    got: 3
                }
            ),
            "confidence_head block_size 3 does not match expected 7"
        );
        assert_eq!(
            format!(
                "{}",
                DsparkError::BigramVocabMismatch {
                    expected: 8,
                    got: 16
                }
            ),
            "bigram_bias vocab_size 16 does not match expected 8"
        );
        assert_eq!(
            format!("{}", DsparkError::DraftLogitsAllNonFinite),
            "draft logits are all non-finite (NaN); cannot argmax"
        );
        assert_eq!(
            format!("{}", DsparkError::DraftModelFailed("boom".to_string())),
            "draft model failed: boom"
        );
    }

    // DraftBlock を tests で touch していないと unused import 警告になるので確認テスト
    #[test]
    fn dfp_draft_block_field_visibility() {
        let block = DraftBlock {
            tokens: vec![0, 1],
            confidences: vec![0.5, 0.7],
            hidden_states: vec![vec![0.0]],
        };
        assert_eq!(block.tokens.len(), 2);
        assert_eq!(block.confidences.len(), 2);
        assert_eq!(block.hidden_states.len(), 1);
    }

    // ---- FullCountBigramBias tests ----

    #[test]
    fn full_count_new_rejects_zero_vocab() {
        let err = FullCountBigramBias::new(0, 256).unwrap_err();
        assert_eq!(err, DsparkError::ZeroVocab);
    }

    #[test]
    fn full_count_new_rejects_zero_rank() {
        let err = FullCountBigramBias::new(100, 0).unwrap_err();
        assert_eq!(err, DsparkError::ZeroRank);
    }

    #[test]
    fn full_count_new_defaults_are_empty() {
        let bias = FullCountBigramBias::new(100, 4).expect("valid");
        assert_eq!(bias.vocab_size(), 100);
        assert_eq!(bias.rank(), 4);
        assert!(bias.is_empty());
        assert_eq!(bias.observed_prev_count(), 0);
        assert_eq!(bias.unique_next_count(0), 0);
        assert_eq!(bias.count(0, 0), 0);
    }

    #[test]
    fn full_count_observe_rejects_prev_out_of_vocab() {
        let mut bias = FullCountBigramBias::new(10, 4).expect("valid");
        let err = bias.observe(10, 5).unwrap_err();
        assert_eq!(
            err,
            DsparkError::TokenOutOfVocab {
                token: 10,
                vocab_size: 10,
            }
        );
    }

    #[test]
    fn full_count_observe_rejects_next_out_of_vocab() {
        let mut bias = FullCountBigramBias::new(10, 4).expect("valid");
        let err = bias.observe(5, 99).unwrap_err();
        assert_eq!(
            err,
            DsparkError::TokenOutOfVocab {
                token: 99,
                vocab_size: 10,
            }
        );
    }

    #[test]
    fn full_count_observe_increments_count() {
        let mut bias = FullCountBigramBias::new(20, 4).expect("valid");
        for _ in 0..7 {
            bias.observe(1, 2).expect("valid");
        }
        assert_eq!(bias.count(1, 2), 7);
        assert_eq!(bias.unique_next_count(1), 1);
        assert_eq!(bias.observed_prev_count(), 1);
    }

    #[test]
    fn full_count_observe_sequence_from_stream() {
        let mut bias = FullCountBigramBias::new(20, 8).expect("valid");
        // 1 2 3 1 2 3 → pairs: (1,2), (2,3), (3,1), (1,2), (2,3)
        bias.observe_sequence(&[1, 2, 3, 1, 2, 3]).expect("valid");
        assert_eq!(bias.count(1, 2), 2);
        assert_eq!(bias.count(2, 3), 2);
        assert_eq!(bias.count(3, 1), 1);
        assert_eq!(bias.observed_prev_count(), 3);
    }

    #[test]
    fn full_count_from_sequence_constructor() {
        let bias = FullCountBigramBias::from_sequence(20, 8, &[1, 2, 3, 4]).expect("valid");
        assert_eq!(bias.observed_prev_count(), 3);
        assert_eq!(bias.count(1, 2), 1);
        assert_eq!(bias.count(2, 3), 1);
        assert_eq!(bias.count(3, 4), 1);
    }

    #[test]
    fn full_count_apply_strength_zero_is_noop() {
        let mut bias = FullCountBigramBias::new(10, 4).expect("valid");
        bias.observe(1, 2).expect("valid");
        let mut logits = vec![0.5_f32; 10];
        bias.apply(1, &mut logits, 0.0).expect("valid");
        assert!(logits.iter().all(|&v| (v - 0.5).abs() < 1e-9));
    }

    #[test]
    fn full_count_apply_unobserved_prev_is_noop() {
        let bias = FullCountBigramBias::new(10, 4).expect("valid");
        let mut logits = vec![0.0_f32; 10];
        bias.apply(5, &mut logits, 1.0).expect("valid");
        assert!(logits.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn full_count_apply_rejects_logits_len_mismatch() {
        let bias = FullCountBigramBias::new(10, 4).expect("valid");
        let mut logits = vec![0.0_f32; 5];
        let err = bias.apply(0, &mut logits, 1.0).unwrap_err();
        assert_eq!(
            err,
            DsparkError::LogitsLenMismatch {
                expected: 10,
                got: 5,
            }
        );
    }

    #[test]
    fn full_count_apply_rejects_prev_out_of_vocab() {
        let bias = FullCountBigramBias::new(10, 4).expect("valid");
        let mut logits = vec![0.0_f32; 10];
        let err = bias.apply(10, &mut logits, 1.0).unwrap_err();
        assert_eq!(
            err,
            DsparkError::TokenOutOfVocab {
                token: 10,
                vocab_size: 10,
            }
        );
    }

    #[test]
    fn full_count_apply_uses_ln_1p() {
        // count=3 → bias = 1.0 * ln(4) を厳密確認
        let mut bias = FullCountBigramBias::new(10, 4).expect("valid");
        for _ in 0..3 {
            bias.observe(1, 2).expect("valid");
        }
        let mut logits = vec![0.0_f32; 10];
        bias.apply(1, &mut logits, 1.0).expect("valid");
        let expected = 4.0_f32.ln();
        assert!(
            (logits[2] - expected).abs() < 1e-6,
            "logits[2] = {}",
            logits[2]
        );
    }

    #[test]
    fn full_count_apply_top_k_by_count_desc() {
        // rank=3、5 個の unique next を count 差付きで観測 → count 上位 3 のみ apply
        let mut bias = FullCountBigramBias::new(20, 3).expect("valid");
        for _ in 0..10 {
            bias.observe(0, 11).expect("valid");
        }
        for _ in 0..8 {
            bias.observe(0, 12).expect("valid");
        }
        for _ in 0..6 {
            bias.observe(0, 13).expect("valid");
        }
        for _ in 0..4 {
            bias.observe(0, 14).expect("valid");
        }
        for _ in 0..2 {
            bias.observe(0, 15).expect("valid");
        }
        // unique_next は 5 個保存されているが apply は top-3
        assert_eq!(bias.unique_next_count(0), 5);
        let mut logits = vec![0.0_f32; 20];
        bias.apply(0, &mut logits, 1.0).expect("valid");
        assert!(logits[14] == 0.0, "14 (count 4) truncated by top-3");
        assert!(logits[15] == 0.0, "15 (count 2) truncated by top-3");
        assert!(logits[11] > logits[12]);
        assert!(logits[12] > logits[13]);
    }

    #[test]
    fn full_count_preserves_late_tied_arrivals() {
        // ★ eager truncate 制約の解消を実証 ★
        // [`MarkovBigramBias`] の同型テスト `observe_eager_truncate_drops_late_tied_arrivals`
        // では 12 が全て drop されるが、FullCountBigramBias では count が積み上がり
        // apply 時に top-K で選ばれる
        let mut bias = FullCountBigramBias::new(20, 2).expect("valid");
        bias.observe(0, 10).expect("valid");
        bias.observe(0, 11).expect("valid");
        // 12 を 5 回 observe → count=5 で 10 (count=1) を追い抜く
        for _ in 0..5 {
            bias.observe(0, 12).expect("valid");
        }
        assert_eq!(bias.count(0, 12), 5);
        assert_eq!(bias.unique_next_count(0), 3);
        let mut logits = vec![0.0_f32; 20];
        bias.apply(0, &mut logits, 1.0).expect("valid");
        // rank=2 → top 2 は 12 (count=5) と (10 or 11 の count=1、id 昇順 tie-break で 10)
        assert!(logits[12] > 0.0, "12 must be in top-K (count=5)");
        assert!(
            logits[10] > 0.0,
            "10 must be in top-K (count=1, id tiebreak)"
        );
        assert!(
            logits[11] == 0.0,
            "11 dropped by top-K (id 11 > 10 for tie)"
        );
        assert!(logits[12] > logits[10], "12 count 5 > 10 count 1");
    }

    // ---- BigramBias trait tests ----

    fn assert_bigram_trait_applies_bias<B: BigramBias>(bias: &B, prev: TokenId) -> f32 {
        let vocab = bias.vocab_size() as usize;
        let mut logits = vec![0.0_f32; vocab];
        bias.apply(prev, &mut logits, 1.0).expect("valid");
        logits.iter().copied().fold(0.0_f32, f32::max)
    }

    #[test]
    fn bigram_bias_trait_markov_impl() {
        let mut bias = MarkovBigramBias::new(10, 4).expect("valid");
        for _ in 0..5 {
            bias.observe(1, 2).expect("valid");
        }
        let peak = assert_bigram_trait_applies_bias(&bias, 1);
        let expected = 6.0_f32.ln();
        assert!((peak - expected).abs() < 1e-6);
    }

    #[test]
    fn bigram_bias_trait_full_count_impl() {
        let mut bias = FullCountBigramBias::new(10, 4).expect("valid");
        for _ in 0..5 {
            bias.observe(1, 2).expect("valid");
        }
        let peak = assert_bigram_trait_applies_bias(&bias, 1);
        let expected = 6.0_f32.ln();
        assert!((peak - expected).abs() < 1e-6);
    }

    #[test]
    fn dfp_draft_accepts_full_count_bigram_via_trait() {
        // Phase 3 test `dfp_draft_applies_bigram_bias_shifts_argmax` の FullCount 版
        let dfp = DFlashParallelDraft::new(1, 8, 4, 100.0).expect("valid");
        let head = PositionConfidenceHead::new(1, 4).expect("valid");
        let mut bigram = FullCountBigramBias::new(8, 4).expect("valid");
        for _ in 0..10 {
            bigram.observe(1, 3).expect("valid");
        }
        // vocab=8, hidden=4, block=1 の mock position (Phase 3 test 準拠)
        let mut logits = vec![0.0_f32; 8];
        logits[0] = 5.0;
        let position = DraftPosition {
            hidden: vec![0.1_f32; 4],
            logits,
        };
        let prefix: [TokenId; 1] = [1];
        let block = dfp
            .draft(&prefix, Some(&bigram), &head, |_p, _bs| {
                Ok(vec![position.clone()])
            })
            .expect("valid");
        assert_eq!(
            block.tokens,
            vec![3],
            "FullCount bigram bias should shift argmax to 3 via trait"
        );
    }

    // ---- serde roundtrip tests (dspark-serde feature 有効時のみ) ----

    #[cfg(feature = "dspark-serde")]
    #[test]
    fn serde_roundtrip_markov_bigram_bias() {
        let mut bias = MarkovBigramBias::new(100, 4).expect("valid");
        bias.observe_sequence(&[1, 2, 3, 1, 2, 4]).expect("valid");
        let encoded = bincode::serialize(&bias).expect("serialize");
        let back: MarkovBigramBias = bincode::deserialize(&encoded).expect("deserialize");
        assert_eq!(back.vocab_size(), 100);
        assert_eq!(back.rank(), 4);
        assert_eq!(back.observed_prev_count(), 3);
        // apply 出力が一致することで内部状態の bit-level 保存を確認
        let mut logits_orig = vec![0.0_f32; 100];
        let mut logits_back = vec![0.0_f32; 100];
        bias.apply(1, &mut logits_orig, 1.0).expect("valid");
        back.apply(1, &mut logits_back, 1.0).expect("valid");
        assert_eq!(logits_orig, logits_back);
    }

    #[cfg(feature = "dspark-serde")]
    #[test]
    fn serde_roundtrip_full_count_bigram_bias() {
        let mut bias = FullCountBigramBias::new(100, 4).expect("valid");
        for _ in 0..3 {
            bias.observe(1, 2).expect("valid");
        }
        bias.observe(5, 10).expect("valid");
        let encoded = bincode::serialize(&bias).expect("serialize");
        let back: FullCountBigramBias = bincode::deserialize(&encoded).expect("deserialize");
        assert_eq!(back.vocab_size(), 100);
        assert_eq!(back.rank(), 4);
        assert_eq!(back.count(1, 2), 3);
        assert_eq!(back.count(5, 10), 1);
        assert_eq!(back.observed_prev_count(), 2);
    }

    #[cfg(feature = "dspark-serde")]
    #[test]
    fn serde_roundtrip_position_confidence_head() {
        let mut head = PositionConfidenceHead::new(3, 4).expect("valid");
        let hidden = [1.0_f32, 1.0, 1.0, 1.0];
        for _ in 0..10 {
            let _ = head.train_step(1, &hidden, true, 0.1).expect("valid");
        }
        let encoded = bincode::serialize(&head).expect("serialize");
        let back: PositionConfidenceHead = bincode::deserialize(&encoded).expect("deserialize");
        assert_eq!(back.block_size(), 3);
        assert_eq!(back.hidden_dim(), 4);
        let orig = head.predict(1, &hidden).expect("valid");
        let back_pred = back.predict(1, &hidden).expect("valid");
        assert!((orig - back_pred).abs() < 1e-6);
    }

    #[cfg(feature = "dspark-serde")]
    #[test]
    fn serde_roundtrip_dspark_error() {
        let cases = [
            DsparkError::ZeroVocab,
            DsparkError::EmptyPrefix,
            DsparkError::TokenOutOfVocab {
                token: 42,
                vocab_size: 10,
            },
            DsparkError::DraftModelFailed("external boom".to_string()),
        ];
        for err in cases {
            let encoded = bincode::serialize(&err).expect("serialize");
            let back: DsparkError = bincode::deserialize(&encoded).expect("deserialize");
            assert_eq!(err, back);
        }
    }

    // ---- apply_bigram_bias_maybe (Phase 5 helper) tests ----

    #[test]
    fn apply_bigram_bias_maybe_returns_ok_when_none() {
        let mut logits = vec![0.5_f32; 10];
        super::apply_bigram_bias_maybe(&mut logits, 3, None, 1.0).expect("valid");
        assert!(logits.iter().all(|&v| (v - 0.5).abs() < 1e-9));
    }

    #[test]
    fn apply_bigram_bias_maybe_returns_ok_when_strength_zero() {
        let mut bias = MarkovBigramBias::new(10, 4).expect("valid");
        for _ in 0..5 {
            bias.observe(3, 7).expect("valid");
        }
        let mut logits = vec![0.5_f32; 10];
        super::apply_bigram_bias_maybe(&mut logits, 3, Some(&bias), 0.0).expect("valid");
        assert!(logits.iter().all(|&v| (v - 0.5).abs() < 1e-9));
    }

    #[test]
    fn apply_bigram_bias_maybe_applies_bias_when_enabled() {
        let mut bias = MarkovBigramBias::new(10, 4).expect("valid");
        for _ in 0..5 {
            bias.observe(3, 7).expect("valid");
        }
        let mut logits = vec![0.0_f32; 10];
        super::apply_bigram_bias_maybe(&mut logits, 3, Some(&bias), 1.0).expect("valid");
        let expected = 6.0_f32.ln();
        assert!(
            (logits[7] - expected).abs() < 1e-6,
            "logits[7] = {} expected {}",
            logits[7],
            expected
        );
    }

    #[test]
    fn apply_bigram_bias_maybe_propagates_error() {
        let bias = MarkovBigramBias::new(10, 4).expect("valid");
        let mut logits = vec![0.0_f32; 5]; // 意図的に vocab 不一致
        let err = super::apply_bigram_bias_maybe(&mut logits, 3, Some(&bias), 1.0).unwrap_err();
        assert_eq!(
            err,
            DsparkError::LogitsLenMismatch {
                expected: 10,
                got: 5,
            }
        );
    }

    // ---- DsparkAdvancedConfig (Phase 6) tests ----

    #[test]
    fn dspark_advanced_config_new_defaults() {
        let head = PositionConfidenceHead::new(7, 4).expect("valid");
        let cfg = super::DsparkAdvancedConfig::new(&head);
        assert_eq!(cfg.confidence_head.block_size(), 7);
        assert_eq!(cfg.confidence_head.hidden_dim(), 4);
        assert!((cfg.confidence_threshold - 0.5).abs() < 1e-9);
        assert!(cfg.hidden_capture_layer.is_none());
    }

    #[test]
    fn dspark_advanced_config_custom_fields() {
        let head = PositionConfidenceHead::new(3, 8).expect("valid");
        let cfg = super::DsparkAdvancedConfig {
            confidence_head: &head,
            confidence_threshold: 0.7,
            hidden_capture_layer: Some(5),
        };
        assert!((cfg.confidence_threshold - 0.7).abs() < 1e-9);
        assert_eq!(cfg.hidden_capture_layer, Some(5));
    }

    #[test]
    fn dspark_advanced_config_is_copy() {
        // DsparkAdvancedConfig は Copy trait を持つので clone せず値渡しできる
        let head = PositionConfidenceHead::new(3, 4).expect("valid");
        let cfg = super::DsparkAdvancedConfig::new(&head);
        let cfg2 = cfg;
        // cfg も cfg2 も使える (Copy semantics)
        assert!((cfg.confidence_threshold - cfg2.confidence_threshold).abs() < 1e-9);
    }

    // ---- DsparkLabelSample (Phase 7) tests ----

    #[test]
    fn dspark_label_sample_construction() {
        let sample = super::DsparkLabelSample {
            position: 3,
            hidden: vec![0.1, 0.2, 0.3, 0.4],
            was_accepted: true,
        };
        assert_eq!(sample.position, 3);
        assert_eq!(sample.hidden.len(), 4);
        assert!(sample.was_accepted);
        // Clone test
        let cloned = sample.clone();
        assert_eq!(cloned.position, sample.position);
        assert_eq!(cloned.hidden, sample.hidden);
        assert_eq!(cloned.was_accepted, sample.was_accepted);
    }

    #[test]
    fn dspark_label_sample_trains_confidence_head() {
        // collect した labels で train_step を呼び、confidence が label 方向に動くか確認
        let mut head = PositionConfidenceHead::new(4, 3).expect("valid");
        let hidden = [1.0_f32, 1.0, 1.0];

        // 全部 accepted な label 100 個で train
        let samples: Vec<super::DsparkLabelSample> = (0..100)
            .map(|_| super::DsparkLabelSample {
                position: 0,
                hidden: hidden.to_vec(),
                was_accepted: true,
            })
            .collect();

        let before = head.predict(0, &hidden).expect("valid");
        for s in &samples {
            head.train_step(s.position, &s.hidden, s.was_accepted, 0.1)
                .expect("train_step");
        }
        let after = head.predict(0, &hidden).expect("valid");

        assert!(
            after > before + 0.2,
            "predict should move toward 1.0 after training on accepted labels: before={before}, after={after}"
        );
    }

    #[cfg(feature = "dspark-serde")]
    #[test]
    fn serde_roundtrip_dspark_label_sample() {
        let sample = super::DsparkLabelSample {
            position: 5,
            hidden: vec![0.1, -0.2, 0.3, -0.4, 0.5],
            was_accepted: false,
        };
        let encoded = bincode::serialize(&sample).expect("serialize");
        let back: super::DsparkLabelSample = bincode::deserialize(&encoded).expect("deserialize");
        assert_eq!(back.position, 5);
        assert_eq!(back.hidden, sample.hidden);
        assert!(!back.was_accepted);
    }
}
