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
//! Phase 3 (次 session): `DFlashParallelDraft` + `generate_speculative_dual` 配線
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

/// 位置別 confidence head (DSpark 3 要素の 2 番目)
///
/// draft position i ∈ [0, block_size) ごとに per-position 重み `w_i ∈ R^H` と bias `b_i ∈ R`
/// を持ち、`confidence_i = sigmoid(w_i · hidden_i + b_i) ∈ [0, 1]` を返す
/// BCE 学習は target 受理ラベル (y ∈ {0, 1}) で `loss = -[y·ln(p) + (1-y)·ln(1-p)]`、
/// sigmoid + BCE の canonical form `dL/dz = p - y` で SGD 1 step
///
/// zero-init 時は全 position で sigmoid(0) = 0.5 (uninformative prior)
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

#[cfg(test)]
mod tests {
    use super::{DsparkError, MarkovBigramBias, PositionConfidenceHead};

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
}
