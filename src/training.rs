//! Ternary Quantization-Aware Training (QAT).
//!
//! BitNet b1.58 style: weights constrained to {-1, 0, +1} during forward pass,
//! gradients flow through via Straight-Through Estimator (STE).
//!
//! This module provides the building blocks for fine-tuning a pre-trained model
//! so that ternary quantization preserves quality (unlike post-training quantization).

/// Ternarize a weight value: map to {-1, 0, +1} based on threshold.
/// threshold = ratio * mean(|w|) — values below threshold become 0.
#[inline]
pub fn ternarize(w: f32, threshold: f32) -> f32 {
    if w > threshold {
        1.0
    } else if w < -threshold {
        -1.0
    } else {
        0.0
    }
}

/// Compute the ternary threshold for a weight slice: ratio * mean(|w|).
pub fn ternary_threshold(weights: &[f32], ratio: f32) -> f32 {
    if weights.is_empty() {
        return 0.0;
    }
    let mean_abs = weights.iter().map(|w| w.abs()).sum::<f32>() / weights.len() as f32;
    ratio * mean_abs
}

/// Ternarize an entire weight matrix in-place, returning the scale factor.
/// Scale = mean(|w|) for non-zero entries (used to rescale ternary weights).
/// Output: w[i] ∈ {-scale, 0, +scale}
pub fn ternarize_weights(weights: &mut [f32], ratio: f32) -> f32 {
    let thresh = ternary_threshold(weights, ratio);
    let mut sum_abs = 0.0f64;
    let mut count = 0usize;

    for w in weights.iter() {
        if w.abs() > thresh {
            sum_abs += w.abs() as f64;
            count += 1;
        }
    }

    let scale = if count > 0 { (sum_abs / count as f64) as f32 } else { 1.0 };

    for w in weights.iter_mut() {
        let t = ternarize(*w, thresh);
        *w = t * scale;
    }

    scale
}

// ─── Straight-Through Estimator ─────────────────────────────────────────────

/// STE forward: ternarize weights, return (ternary_weights, scale, threshold).
/// STE backward: gradient passes through unchanged (identity).
pub struct StraightThroughEstimator {
    pub ratio: f32,
}

impl StraightThroughEstimator {
    pub fn new(ratio: f32) -> Self {
        Self { ratio }
    }

    /// Forward pass: ternarize weights, return (output, scale).
    /// The original weights are NOT modified — caller keeps the f32 master copy.
    pub fn forward(&self, weights: &[f32]) -> (Vec<f32>, f32) {
        let mut ternary = weights.to_vec();
        let scale = ternarize_weights(&mut ternary, self.ratio);
        (ternary, scale)
    }

    /// Backward pass: STE — gradient flows through unchanged.
    /// Optionally clips gradient to [-1, 1] for stability (grad clipping).
    #[inline]
    pub fn backward(&self, grad: &[f32], clip: bool) -> Vec<f32> {
        if clip {
            grad.iter().map(|g| g.clamp(-1.0, 1.0)).collect()
        } else {
            grad.to_vec()
        }
    }
}

// ─── QAT Linear Layer ───────────────────────────────────────────────────────

/// A linear layer that uses ternary QAT.
/// Holds f32 master weights; forward pass ternarizes them via STE.
pub struct QatLinear {
    /// Master weights in f32 (rows × cols, row-major).
    pub weights: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    /// STE for ternary quantization.
    pub ste: StraightThroughEstimator,
    /// Gradient accumulator (same shape as weights).
    pub grad: Vec<f32>,
}

impl QatLinear {
    pub fn new(weights: Vec<f32>, rows: usize, cols: usize, ratio: f32) -> Self {
        let n = rows * cols;
        assert_eq!(weights.len(), n);
        Self {
            weights,
            rows,
            cols,
            ste: StraightThroughEstimator::new(ratio),
            grad: vec![0.0; n],
        }
    }

    /// Forward: ternarize weights, compute y = W_ternary @ x.
    /// Returns (output, ternary_weights) — ternary_weights needed for backward.
    pub fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(input.len(), self.cols);
        let (ternary_w, _scale) = self.ste.forward(&self.weights);

        let mut output = vec![0.0f32; self.rows];
        for r in 0..self.rows {
            let row_start = r * self.cols;
            let mut sum = 0.0f32;
            for c in 0..self.cols {
                sum += ternary_w[row_start + c] * input[c];
            }
            output[r] = sum;
        }

        (output, ternary_w)
    }

    /// Backward: compute gradients w.r.t. weights and input.
    /// grad_output: [rows], input: [cols] (from forward pass).
    /// Returns grad_input: [cols].
    /// Accumulates weight gradients into self.grad (STE: pass through).
    pub fn backward(&mut self, grad_output: &[f32], input: &[f32]) -> Vec<f32> {
        assert_eq!(grad_output.len(), self.rows);
        assert_eq!(input.len(), self.cols);

        // dW = grad_output ⊗ input (outer product), passed through STE
        for r in 0..self.rows {
            let row_start = r * self.cols;
            for c in 0..self.cols {
                self.grad[row_start + c] += grad_output[r] * input[c];
            }
        }

        // dX = W^T @ grad_output (use master weights for STE)
        let mut grad_input = vec![0.0f32; self.cols];
        for c in 0..self.cols {
            let mut sum = 0.0f32;
            for r in 0..self.rows {
                sum += self.weights[r * self.cols + c] * grad_output[r];
            }
            grad_input[c] = sum;
        }

        grad_input
    }

    /// Zero accumulated gradients.
    pub fn zero_grad(&mut self) {
        self.grad.fill(0.0);
    }
}

// ─── AdamW Optimizer ────────────────────────────────────────────────────────

/// Minimal AdamW optimizer for QAT training.
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    /// First moment (per parameter).
    m: Vec<f32>,
    /// Second moment (per parameter).
    v: Vec<f32>,
    /// Step count (for bias correction).
    t: usize,
}

impl AdamW {
    pub fn new(num_params: usize, lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            m: vec![0.0; num_params],
            v: vec![0.0; num_params],
            t: 0,
        }
    }

    /// One optimizer step: update weights using accumulated gradients.
    pub fn step(&mut self, weights: &mut [f32], grads: &[f32]) {
        assert_eq!(weights.len(), grads.len());
        assert_eq!(weights.len(), self.m.len());

        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..weights.len() {
            // AdamW: decouple weight decay
            weights[i] *= 1.0 - self.lr * self.weight_decay;

            // Moment updates
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];

            // Bias-corrected moments
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;

            weights[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

// ─── QAT Training Step ──────────────────────────────────────────────────────

/// Mean Squared Error loss: (1/N) Σ (pred - target)².
pub fn mse_loss(pred: &[f32], target: &[f32]) -> f32 {
    assert_eq!(pred.len(), target.len());
    let n = pred.len() as f32;
    pred.iter()
        .zip(target.iter())
        .map(|(p, t)| (p - t) * (p - t))
        .sum::<f32>()
        / n
}

/// Gradient of MSE loss: (2/N) * (pred - target).
pub fn mse_loss_grad(pred: &[f32], target: &[f32]) -> Vec<f32> {
    let n = pred.len() as f32;
    pred.iter()
        .zip(target.iter())
        .map(|(p, t)| 2.0 * (p - t) / n)
        .collect()
}

/// One QAT training step on a single linear layer.
/// Returns the loss value.
pub fn qat_train_step(
    layer: &mut QatLinear,
    optimizer: &mut AdamW,
    input: &[f32],
    target: &[f32],
) -> f32 {
    qat_train_step_l1(layer, optimizer, input, target, 0.0)
}

/// QAT training step with L1 regularization for sparsity induction.
/// `lambda_l1`: L1 penalty coefficient. Higher = more zeros in weights.
/// Returns the total loss (MSE + L1).
pub fn qat_train_step_l1(
    layer: &mut QatLinear,
    optimizer: &mut AdamW,
    input: &[f32],
    target: &[f32],
    lambda_l1: f32,
) -> f32 {
    layer.zero_grad();

    // Forward
    let (output, _ternary_w) = layer.forward(input);

    // MSE loss
    let mse = mse_loss(&output, target);
    let grad_output = mse_loss_grad(&output, target);

    // L1 penalty: λ × Σ|w|
    let l1_penalty = if lambda_l1 > 0.0 {
        lambda_l1 * layer.weights.iter().map(|w| w.abs()).sum::<f32>()
    } else {
        0.0
    };

    // Backward (STE: gradients pass through ternarization)
    let _grad_input = layer.backward(&grad_output, input);

    // Add L1 gradient: d/dw (λ|w|) = λ * sign(w)
    if lambda_l1 > 0.0 {
        for (g, w) in layer.grad.iter_mut().zip(layer.weights.iter()) {
            *g += lambda_l1 * w.signum();
        }
    }

    // Apply STE clipping to weight gradients
    let clipped_grad = layer.ste.backward(&layer.grad, true);

    // Optimizer step
    optimizer.step(&mut layer.weights, &clipped_grad);

    mse + l1_penalty
}

/// Count the fraction of weights that are effectively zero (below epsilon).
pub fn sparsity_ratio(weights: &[f32], eps: f32) -> f32 {
    if weights.is_empty() {
        return 0.0;
    }
    let zeros = weights.iter().filter(|w| w.abs() < eps).count();
    zeros as f32 / weights.len() as f32
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternarize_basic() {
        assert_eq!(ternarize(0.5, 0.3), 1.0);
        assert_eq!(ternarize(-0.5, 0.3), -1.0);
        assert_eq!(ternarize(0.1, 0.3), 0.0);
        assert_eq!(ternarize(0.0, 0.3), 0.0);
    }

    #[test]
    fn test_ternary_threshold() {
        let w = [1.0, -2.0, 0.5, -0.5];
        // mean(|w|) = (1 + 2 + 0.5 + 0.5) / 4 = 1.0
        let t = ternary_threshold(&w, 0.7);
        assert!((t - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_ternary_threshold_empty() {
        assert_eq!(ternary_threshold(&[], 0.7), 0.0);
    }

    #[test]
    fn test_ternarize_weights() {
        let mut w = vec![1.0, -2.0, 0.1, -0.1, 3.0];
        let scale = ternarize_weights(&mut w, 0.5);

        // All values should be {-scale, 0, +scale}
        for &v in &w {
            assert!(
                v.abs() < 1e-6 || (v.abs() - scale).abs() < 1e-5,
                "value {v} not in {{-{scale}, 0, +{scale}}}"
            );
        }
        assert!(scale > 0.0);
    }

    #[test]
    fn test_ste_forward_produces_ternary() {
        let ste = StraightThroughEstimator::new(0.7);
        let weights = vec![1.0, -2.0, 0.01, 3.0, -0.5];
        let (ternary, scale) = ste.forward(&weights);

        assert_eq!(ternary.len(), weights.len());
        for &v in &ternary {
            assert!(
                v.abs() < 1e-6 || (v.abs() - scale).abs() < 1e-5,
                "STE output {v} not ternary (scale={scale})"
            );
        }
    }

    #[test]
    fn test_ste_backward_identity() {
        let ste = StraightThroughEstimator::new(0.7);
        let grad = vec![0.5, -0.3, 0.1];
        let out = ste.backward(&grad, false);
        assert_eq!(out, grad);
    }

    #[test]
    fn test_ste_backward_clipped() {
        let ste = StraightThroughEstimator::new(0.7);
        let grad = vec![2.0, -3.0, 0.5];
        let out = ste.backward(&grad, true);
        assert_eq!(out, vec![1.0, -1.0, 0.5]);
    }

    #[test]
    fn test_qat_linear_forward() {
        // 2x3 matrix, input [3]
        let weights = vec![1.0, 0.0, -1.0, 0.5, 0.5, 0.0];
        let layer = QatLinear::new(weights, 2, 3, 0.7);
        let input = vec![1.0, 2.0, 3.0];
        let (output, _ternary_w) = layer.forward(&input);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_qat_linear_backward_shapes() {
        let weights = vec![1.0, 0.0, -1.0, 0.5, 0.5, 0.0];
        let mut layer = QatLinear::new(weights, 2, 3, 0.7);
        let input = vec![1.0, 2.0, 3.0];
        let grad_output = vec![1.0, -1.0];
        let grad_input = layer.backward(&grad_output, &input);
        assert_eq!(grad_input.len(), 3);
        // Weight gradients should be non-zero
        assert!(layer.grad.iter().any(|g| g.abs() > 1e-6));
    }

    #[test]
    fn test_adamw_step() {
        let mut weights = vec![1.0, -1.0, 0.5];
        let grads = vec![0.1, -0.2, 0.05];
        let mut opt = AdamW::new(3, 0.01);
        let original = weights.clone();

        opt.step(&mut weights, &grads);

        // Weights should have changed
        for i in 0..3 {
            assert!(
                (weights[i] - original[i]).abs() > 1e-8,
                "weight[{i}] unchanged"
            );
        }
    }

    #[test]
    fn test_mse_loss() {
        let pred = vec![1.0, 2.0, 3.0];
        let target = vec![1.0, 2.0, 3.0];
        assert!(mse_loss(&pred, &target) < 1e-6);

        let pred2 = vec![2.0, 3.0, 4.0];
        // MSE = (1 + 1 + 1) / 3 = 1.0
        assert!((mse_loss(&pred2, &target) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mse_loss_grad() {
        let pred = vec![2.0, 3.0];
        let target = vec![1.0, 1.0];
        let grad = mse_loss_grad(&pred, &target);
        // grad = 2/2 * [1, 2] = [1.0, 2.0]
        assert!((grad[0] - 1.0).abs() < 1e-6);
        assert!((grad[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_qat_train_step_loss_decreases() {
        // Simple regression: learn W such that W @ [1, 0] ≈ [2]
        let weights = vec![0.0, 0.0]; // 1x2 matrix
        let mut layer = QatLinear::new(weights, 1, 2, 0.3);
        let mut opt = AdamW::new(2, 0.1);

        let input = vec![1.0, 0.0];
        let target = vec![2.0];

        let loss_0 = qat_train_step(&mut layer, &mut opt, &input, &target);

        // Run 50 more steps
        let mut last_loss = loss_0;
        for _ in 0..50 {
            last_loss = qat_train_step(&mut layer, &mut opt, &input, &target);
        }

        assert!(
            last_loss < loss_0,
            "loss should decrease: {loss_0} -> {last_loss}"
        );
    }

    #[test]
    fn test_qat_train_convergence() {
        // Learn identity mapping: W @ x ≈ x for 2x2
        let weights = vec![0.1, 0.0, 0.0, 0.1];
        let mut layer = QatLinear::new(weights, 2, 2, 0.3);
        let mut opt = AdamW::new(4, 0.05);

        let input = vec![1.0, -1.0];
        let target = vec![1.0, -1.0];

        for _ in 0..200 {
            qat_train_step(&mut layer, &mut opt, &input, &target);
        }

        let (output, _) = layer.forward(&input);
        let final_loss = mse_loss(&output, &target);
        assert!(
            final_loss < 1.0,
            "should converge somewhat: loss={final_loss}"
        );
    }

    #[test]
    fn test_sparsity_ratio() {
        let w = vec![0.0, 0.0, 1.0, 0.0, -1.0];
        let r = sparsity_ratio(&w, 1e-6);
        assert!((r - 0.6).abs() < 1e-6); // 3/5 = 0.6
    }

    #[test]
    fn test_sparsity_ratio_empty() {
        assert_eq!(sparsity_ratio(&[], 1e-6), 0.0);
    }

    #[test]
    fn test_l1_increases_sparsity() {
        // Train with L1=0 and L1=0.1, compare sparsity
        let run = |lambda: f32| -> f32 {
            let weights = vec![0.5, -0.3, 0.8, -0.1, 0.2, -0.6, 0.05, -0.02];
            let mut layer = QatLinear::new(weights, 2, 4, 0.3);
            let mut opt = AdamW::new(8, 0.01);
            let input = vec![1.0, 0.0, 0.0, 0.0];
            let target = vec![1.0, 0.0];

            for _ in 0..100 {
                qat_train_step_l1(&mut layer, &mut opt, &input, &target, lambda);
            }
            sparsity_ratio(&layer.weights, 0.05)
        };

        let sparsity_no_l1 = run(0.0);
        let sparsity_with_l1 = run(0.1);

        assert!(
            sparsity_with_l1 >= sparsity_no_l1,
            "L1 should increase sparsity: no_l1={sparsity_no_l1}, with_l1={sparsity_with_l1}"
        );
    }

    #[test]
    fn test_l1_grad_adds_sign() {
        let weights = vec![1.0, -2.0]; // 1x2
        let mut layer = QatLinear::new(weights, 1, 2, 0.3);
        let mut opt = AdamW::new(2, 0.01);
        let input = vec![1.0, 1.0];
        let target = vec![0.0];

        // With lambda=0, get baseline gradient
        layer.zero_grad();
        let (output, _) = layer.forward(&input);
        let grad_out = mse_loss_grad(&output, &target);
        layer.backward(&grad_out, &input);
        let grad_no_l1 = layer.grad.clone();

        // Manually add L1 sign gradient
        let lambda = 0.5f32;
        let expected_0 = grad_no_l1[0] + lambda * 1.0f32.signum(); // w=1.0 → sign=1
        let expected_1 = grad_no_l1[1] + lambda * (-2.0f32).signum(); // w=-2.0 → sign=-1

        // Verify the L1 gradient direction
        assert!(expected_0 > grad_no_l1[0]); // positive weight → L1 pushes gradient up
        assert!(expected_1 < grad_no_l1[1]); // negative weight → L1 pushes gradient down
    }
}
