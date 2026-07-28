//! DSpark speculative decoding primitives.
//!
//! Reference: <https://huggingface.co/RadixArk/Kimi-K3-DSpark> (2026-07-29 absorbed).
//!
//! DSpark = DFlash 並列 draft + Markov logit-bias + 位置別 confidence head の 3 要素合成
//! 本 module は standalone primitive のみ提供 `generate_speculative_dual` 等への配線は caller 責務
//!
//! ## Scope
//!
//! Phase 1 (現状): [`MarkovBigramBias`] のみ vanilla DSpark は rank=256
//! Phase 2 (次 session): `PositionConfidenceHead`
//! Phase 3 (次々 session): `DFlashParallelDraft` + `generate_speculative_dual` 配線
//!
//! ## Rank-K bigram bias 設計
//!
//! 各 previous token に対し top-K next token の観測頻度を保持し、生成時に
//! `logits[next] += strength * ln(1 + count)` を加算する ln 形状は runaway
//! high-count bias を damp する 内部は eager truncate 方式 (observe 時に
//! rank 到達で下位 bucket を drop) 完全 top-K が必要な場合は Phase 2 で
//! full-count sketch を追加する

use std::collections::HashMap;

/// DSpark primitives が扱う token id 型
pub type TokenId = u32;

/// DSpark primitive のエラー
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
        }
    }
}

impl std::error::Error for DsparkError {}

/// Rank-K Markov bigram bias (DSpark vanilla rank=256)
///
/// 各 prev token に対して観測頻度 top-K の (next, count) を eager truncate で保持する
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

#[cfg(test)]
mod tests {
    use super::{DsparkError, MarkovBigramBias};

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
}
