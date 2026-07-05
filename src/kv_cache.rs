//! kv cache.

// KV Cache
// ---------------------------------------------------------------------------

/// Key-value cache for transformer attention layers.
#[derive(Debug, Clone)]
pub struct KvCache {
    /// Cached keys: `[layer][seq_pos][head_dim]`
    pub keys: Vec<Vec<Vec<f32>>>,
    /// Cached values: `[layer][seq_pos][head_dim]`
    pub values: Vec<Vec<Vec<f32>>>,
    num_layers: usize,
    head_dim: usize,
    max_seq_len: usize,
}

impl KvCache {
    /// Create a new empty KV cache.
    #[must_use]
    pub fn new(num_layers: usize, head_dim: usize, max_seq_len: usize) -> Self {
        Self {
            keys: vec![Vec::new(); num_layers],
            values: vec![Vec::new(); num_layers],
            num_layers,
            head_dim,
            max_seq_len,
        }
    }

    /// Append a key-value pair for a given layer.
    /// Returns `false` if the cache is full.
    pub fn append(&mut self, layer: usize, key: Vec<f32>, value: Vec<f32>) -> bool {
        if layer >= self.num_layers {
            return false;
        }
        if self.keys[layer].len() >= self.max_seq_len {
            return false;
        }
        self.keys[layer].push(key);
        self.values[layer].push(value);
        true
    }

    /// Current sequence length for a given layer.
    #[must_use]
    pub fn seq_len(&self, layer: usize) -> usize {
        if layer >= self.num_layers {
            return 0;
        }
        self.keys[layer].len()
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        for layer in 0..self.num_layers {
            self.keys[layer].clear();
            self.values[layer].clear();
        }
    }

    /// Head dimension.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Maximum sequence length.
    #[must_use]
    pub const fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Number of layers.
    #[must_use]
    pub const fn num_layers(&self) -> usize {
        self.num_layers
    }
}
