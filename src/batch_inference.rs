//! batch inference.

use crate::attention::*;
use crate::kv_cache::*;
use crate::linalg::*;
use crate::rope::*;
use crate::sampling::*;

// Batch Inference
// ---------------------------------------------------------------------------

/// Configuration for a minimal transformer model.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub rope_base: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256,
            hidden_dim: 64,
            num_heads: 4,
            num_layers: 2,
            max_seq_len: 128,
            rope_base: 10000.0,
        }
    }
}

/// A minimal transformer model for inference (weights are random/zero for testing).
#[derive(Debug, Clone)]
pub struct TransformerModel {
    pub config: ModelConfig,
    pub embedding: Vec<Vec<f32>>,
    pub layers: Vec<TransformerLayer>,
    pub output_proj: Vec<Vec<f32>>,
}

/// A single transformer layer.
#[derive(Debug, Clone)]
pub struct TransformerLayer {
    pub q_proj: Vec<Vec<f32>>,
    pub k_proj: Vec<Vec<f32>>,
    pub v_proj: Vec<Vec<f32>>,
    pub o_proj: Vec<Vec<f32>>,
    pub ff_up: Vec<Vec<f32>>,
    pub ff_down: Vec<Vec<f32>>,
}

impl TransformerModel {
    /// Create a model with identity-like initialization for testing.
    #[must_use]
    pub fn new_test(config: ModelConfig) -> Self {
        let d = config.hidden_dim;
        let v = config.vocab_size;

        let embedding = (0..v)
            .map(|i| {
                let mut row = vec![0.0f32; d];
                row[i % d] = 1.0;
                row
            })
            .collect();

        let make_identity = || -> Vec<Vec<f32>> {
            (0..d)
                .map(|i| {
                    let mut row = vec![0.0f32; d];
                    row[i] = 0.1;
                    row
                })
                .collect()
        };

        let layers = (0..config.num_layers)
            .map(|_| TransformerLayer {
                q_proj: make_identity(),
                k_proj: make_identity(),
                v_proj: make_identity(),
                o_proj: make_identity(),
                ff_up: make_identity(),
                ff_down: make_identity(),
            })
            .collect();

        let output_proj = (0..v)
            .map(|i| {
                let mut row = vec![0.0f32; d];
                row[i % d] = 1.0;
                row
            })
            .collect();

        Self {
            config,
            embedding,
            layers,
            output_proj,
        }
    }

    /// Run a forward pass on a single token sequence, returning logits.
    ///
    /// # Panics
    ///
    /// Panics if `token_ids` is empty.
    #[must_use]
    pub fn forward(&self, token_ids: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let d = self.config.hidden_dim;
        let head_dim = d / self.config.num_heads;

        // Embedding lookup
        let mut hidden: Vec<Vec<f32>> = token_ids
            .iter()
            .map(|&id| {
                let idx = id as usize;
                if idx < self.embedding.len() {
                    self.embedding[idx].clone()
                } else {
                    vec![0.0f32; d]
                }
            })
            .collect();

        // Process each layer
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let seq_len = hidden.len();
            let cache_offset = cache.seq_len(layer_idx);

            // Project Q, K, V
            let queries: Vec<Vec<f32>> = hidden.iter().map(|h| matvec(&layer.q_proj, h)).collect();
            let keys: Vec<Vec<f32>> = hidden.iter().map(|h| matvec(&layer.k_proj, h)).collect();
            let values: Vec<Vec<f32>> = hidden.iter().map(|h| matvec(&layer.v_proj, h)).collect();

            // Apply RoPE to Q and K (simplified: first head only)
            let queries_rope: Vec<Vec<f32>> = queries
                .iter()
                .enumerate()
                .map(|(i, q)| {
                    let mut roped = q.clone();
                    let pos = cache_offset + i;
                    let chunk: Vec<f32> = roped[..head_dim].to_vec();
                    let rotated = apply_rope(&chunk, pos, self.config.rope_base);
                    roped[..head_dim].copy_from_slice(&rotated);
                    roped
                })
                .collect();

            let keys_rope: Vec<Vec<f32>> = keys
                .iter()
                .enumerate()
                .map(|(i, k)| {
                    let mut roped = k.clone();
                    let pos = cache_offset + i;
                    let chunk: Vec<f32> = roped[..head_dim].to_vec();
                    let rotated = apply_rope(&chunk, pos, self.config.rope_base);
                    roped[..head_dim].copy_from_slice(&rotated);
                    roped
                })
                .collect();

            // Append to KV cache
            for i in 0..seq_len {
                cache.append(layer_idx, keys_rope[i].clone(), values[i].clone());
            }

            // Attention with full cached K, V
            let all_keys = &cache.keys[layer_idx];
            let all_values = &cache.values[layer_idx];
            let total_len = all_keys.len();

            // Build causal mask for the query positions
            let mut mask = vec![vec![0.0f32; total_len]; seq_len];
            for (i, mask_row) in mask.iter_mut().enumerate() {
                let query_pos = cache_offset + i;
                for (j, val) in mask_row.iter_mut().enumerate() {
                    if j > query_pos {
                        *val = f32::NEG_INFINITY;
                    }
                }
            }

            let attn_out =
                scaled_dot_product_attention(&queries_rope, all_keys, all_values, Some(&mask));

            // Output projection + residual
            for i in 0..seq_len {
                let projected = matvec(&layer.o_proj, &attn_out[i]);
                for j in 0..d {
                    hidden[i][j] += projected[j];
                }
            }

            // Feedforward + residual
            for h in hidden.iter_mut().take(seq_len) {
                let normed = rms_norm(h, 1e-6);
                let up = matvec(&layer.ff_up, &normed);
                let activated: Vec<f32> = up.iter().map(|&x| silu(x)).collect();
                let down = matvec(&layer.ff_down, &activated);
                for (hj, &dj) in h.iter_mut().zip(down.iter()) {
                    *hj += dj;
                }
            }
        }

        // Output projection: take last token
        let last = hidden.last().unwrap();
        matvec(&self.output_proj, last)
    }

    /// Batch inference: run forward pass on multiple sequences.
    #[must_use]
    pub fn forward_batch(&self, batch: &[Vec<u32>]) -> Vec<Vec<f32>> {
        batch
            .iter()
            .map(|seq| {
                let mut cache = KvCache::new(
                    self.config.num_layers,
                    self.config.hidden_dim / self.config.num_heads,
                    self.config.max_seq_len,
                );
                self.forward(seq, &mut cache)
            })
            .collect()
    }

    /// Generate tokens autoregressively.
    #[must_use]
    pub fn generate(
        &self,
        prompt: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> Vec<u32> {
        let mut cache = KvCache::new(
            self.config.num_layers,
            self.config.hidden_dim / self.config.num_heads,
            self.config.max_seq_len,
        );

        let mut tokens = prompt.to_vec();

        // Prefill
        let mut logits = self.forward(&tokens, &mut cache);

        for _ in 0..max_new_tokens {
            apply_temperature(&mut logits, temperature);
            top_k_filter(&mut logits, top_k);
            let next_token = sample_argmax(&logits) as u32;
            tokens.push(next_token);

            // Decode step (single token)
            logits = self.forward(&[next_token], &mut cache);
        }

        tokens
    }
}
