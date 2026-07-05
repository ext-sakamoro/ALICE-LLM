//! Integration tests.

#![allow(
    clippy::wildcard_imports,
    clippy::too_many_lines,
    clippy::unwrap_used,
    clippy::indexing_slicing
)]

use crate::attention::*;
use crate::batch_inference::*;
use crate::kv_cache::*;
use crate::linalg::*;
use crate::quantization::*;
use crate::rope::*;
use crate::sampling::*;
use crate::tokenizer::*;
use std::collections::HashMap;

use super::*;

// ---- BPE Tokenizer tests ----

#[test]
fn test_tokenizer_encode_chars() {
    let tok = BpeTokenizer::from_chars("abc", vec![]);
    let ids = tok.encode("abc");
    assert_eq!(ids.len(), 3);
}

#[test]
fn test_tokenizer_decode_roundtrip() {
    let tok = BpeTokenizer::from_chars("hello", vec![]);
    let ids = tok.encode("hello");
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, "hello");
}

#[test]
fn test_tokenizer_with_merge() {
    let merges = vec![MergeRule {
        left: "h".into(),
        right: "e".into(),
        merged: "he".into(),
    }];
    let tok = BpeTokenizer::from_chars("hello", merges);
    let ids = tok.encode("hello");
    assert_eq!(ids.len(), 4); // "he", "l", "l", "o"
}

#[test]
fn test_tokenizer_empty() {
    let tok = BpeTokenizer::from_chars("a", vec![]);
    let ids = tok.encode("");
    assert!(ids.is_empty());
}

#[test]
fn test_tokenizer_vocab_size() {
    let tok = BpeTokenizer::from_chars("abcdef", vec![]);
    assert_eq!(tok.vocab_size(), 6);
}

#[test]
fn test_tokenizer_unknown_char() {
    let tok = BpeTokenizer::from_chars("ab", vec![]);
    let ids = tok.encode("abc");
    assert_eq!(ids.len(), 3);
    // 'c' is unknown, maps to 0
}

#[test]
fn test_tokenizer_multiple_merges() {
    let merges = vec![
        MergeRule {
            left: "a".into(),
            right: "b".into(),
            merged: "ab".into(),
        },
        MergeRule {
            left: "ab".into(),
            right: "c".into(),
            merged: "abc".into(),
        },
    ];
    let tok = BpeTokenizer::from_chars("abcabc", merges);
    let ids = tok.encode("abcabc");
    assert_eq!(ids.len(), 2); // "abc", "abc"
}

#[test]
fn test_tokenizer_decode_unknown_id() {
    let tok = BpeTokenizer::from_chars("ab", vec![]);
    let decoded = tok.decode(&[999]);
    assert_eq!(decoded, "");
}

#[test]
fn test_tokenizer_single_char() {
    let tok = BpeTokenizer::from_chars("x", vec![]);
    let ids = tok.encode("x");
    assert_eq!(ids.len(), 1);
    assert_eq!(tok.decode(&ids), "x");
}

#[test]
fn test_tokenizer_repeated_chars() {
    let tok = BpeTokenizer::from_chars("aaa", vec![]);
    let ids = tok.encode("aaa");
    assert_eq!(ids.len(), 3);
}

// ---- KV Cache tests ----

#[test]
fn test_kv_cache_new() {
    let cache = KvCache::new(2, 16, 64);
    assert_eq!(cache.num_layers(), 2);
    assert_eq!(cache.head_dim(), 16);
    assert_eq!(cache.max_seq_len(), 64);
}

#[test]
fn test_kv_cache_append() {
    let mut cache = KvCache::new(1, 4, 10);
    assert!(cache.append(0, vec![1.0; 4], vec![2.0; 4]));
    assert_eq!(cache.seq_len(0), 1);
}

#[test]
fn test_kv_cache_full() {
    let mut cache = KvCache::new(1, 2, 2);
    assert!(cache.append(0, vec![1.0; 2], vec![1.0; 2]));
    assert!(cache.append(0, vec![1.0; 2], vec![1.0; 2]));
    assert!(!cache.append(0, vec![1.0; 2], vec![1.0; 2]));
}

#[test]
fn test_kv_cache_clear() {
    let mut cache = KvCache::new(2, 4, 10);
    cache.append(0, vec![1.0; 4], vec![1.0; 4]);
    cache.append(1, vec![1.0; 4], vec![1.0; 4]);
    cache.clear();
    assert_eq!(cache.seq_len(0), 0);
    assert_eq!(cache.seq_len(1), 0);
}

#[test]
fn test_kv_cache_invalid_layer() {
    let mut cache = KvCache::new(1, 4, 10);
    assert!(!cache.append(5, vec![1.0; 4], vec![1.0; 4]));
    assert_eq!(cache.seq_len(5), 0);
}

#[test]
fn test_kv_cache_multiple_layers() {
    let mut cache = KvCache::new(3, 4, 10);
    cache.append(0, vec![1.0; 4], vec![1.0; 4]);
    cache.append(1, vec![2.0; 4], vec![2.0; 4]);
    cache.append(1, vec![3.0; 4], vec![3.0; 4]);
    assert_eq!(cache.seq_len(0), 1);
    assert_eq!(cache.seq_len(1), 2);
    assert_eq!(cache.seq_len(2), 0);
}

// ---- Attention tests ----

#[test]
fn test_attention_basic() {
    let q = vec![vec![1.0, 0.0]];
    let k = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let v = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let out = scaled_dot_product_attention(&q, &k, &v, None);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].len(), 2);
}

#[test]
fn test_attention_identity() {
    let q = vec![vec![1.0, 0.0, 0.0, 0.0]];
    let k = vec![vec![1.0, 0.0, 0.0, 0.0]];
    let v = vec![vec![0.5, 0.5, 0.0, 0.0]];
    let out = scaled_dot_product_attention(&q, &k, &v, None);
    assert!((out[0][0] - 0.5).abs() < 1e-5);
}

#[test]
fn test_attention_empty() {
    let out = scaled_dot_product_attention(&[], &[], &[], None);
    assert!(out.is_empty());
}

#[test]
fn test_attention_with_causal_mask() {
    let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let k = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let v = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let mask = causal_mask(2);
    let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask));
    assert_eq!(out.len(), 2);
    // First token can only attend to itself
    assert!((out[0][0] - 1.0).abs() < 1e-5);
    assert!(out[0][1].abs() < 1e-5);
}

#[test]
fn test_causal_mask_shape() {
    let mask = causal_mask(4);
    assert_eq!(mask.len(), 4);
    assert_eq!(mask[0].len(), 4);
    assert_eq!(mask[0][0], 0.0);
    assert!(mask[0][1].is_infinite());
}

#[test]
fn test_attention_preserves_sum() {
    let q = vec![vec![0.5, 0.5]];
    let k = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let v = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let out = scaled_dot_product_attention(&q, &k, &v, None);
    let sum: f32 = out[0].iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
}

// ---- RoPE tests ----

#[test]
fn test_rope_position_zero() {
    let v = vec![1.0, 0.0, 0.0, 1.0];
    let r = apply_rope(&v, 0, 10000.0);
    // At position 0, angles are 0, so cos=1, sin=0
    assert!((r[0] - 1.0).abs() < 1e-5);
    assert!((r[1] - 0.0).abs() < 1e-5);
}

#[test]
fn test_rope_preserves_norm() {
    let v = vec![1.0, 0.0, 0.0, 1.0];
    let r = apply_rope(&v, 5, 10000.0);
    let norm_orig: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_roped: f32 = r.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm_orig - norm_roped).abs() < 1e-4);
}

#[test]
fn test_rope_different_positions() {
    let v = vec![1.0, 0.0, 1.0, 0.0];
    let r1 = apply_rope(&v, 1, 10000.0);
    let r2 = apply_rope(&v, 2, 10000.0);
    assert_ne!(r1, r2);
}

#[test]
fn test_rope_batch() {
    let vecs = vec![vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 1.0, 1.0, 0.0]];
    let result = apply_rope_batch(&vecs, 10000.0);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].len(), 4);
}

#[test]
fn test_rope_even_dim() {
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let r = apply_rope(&v, 3, 10000.0);
    assert_eq!(r.len(), 6);
}

// ---- Quantization INT8 tests ----

#[test]
fn test_int8_quantize_roundtrip() {
    let values = vec![0.0, 0.5, 1.0, -1.0, -0.5];
    let q = QuantizedInt8::quantize(&values);
    let dq = q.dequantize();
    for (orig, deq) in values.iter().zip(dq.iter()) {
        assert!((orig - deq).abs() < 0.05);
    }
}

#[test]
fn test_int8_len() {
    let q = QuantizedInt8::quantize(&[1.0, 2.0, 3.0]);
    assert_eq!(q.len(), 3);
    assert!(!q.is_empty());
}

#[test]
fn test_int8_empty() {
    let q = QuantizedInt8::quantize(&[]);
    assert!(q.is_empty());
}

#[test]
fn test_int8_constant() {
    let q = QuantizedInt8::quantize(&[5.0, 5.0, 5.0]);
    let dq = q.dequantize();
    for v in &dq {
        assert!((v - 5.0).abs() < 0.1);
    }
}

#[test]
fn test_int8_negative_values() {
    let values = vec![-10.0, -5.0, 0.0, 5.0, 10.0];
    let q = QuantizedInt8::quantize(&values);
    let dq = q.dequantize();
    for (orig, deq) in values.iter().zip(dq.iter()) {
        assert!((orig - deq).abs() < 0.2);
    }
}

// ---- Quantization INT4 tests ----

#[test]
fn test_int4_quantize_roundtrip() {
    let values = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let q = QuantizedInt4::quantize(&values);
    let dq = q.dequantize();
    for (orig, deq) in values.iter().zip(dq.iter()) {
        assert!((orig - deq).abs() < 0.1);
    }
}

#[test]
fn test_int4_len() {
    let q = QuantizedInt4::quantize(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(q.len(), 5);
    assert!(!q.is_empty());
}

#[test]
fn test_int4_empty() {
    let q = QuantizedInt4::quantize(&[]);
    assert!(q.is_empty());
}

#[test]
fn test_int4_packing() {
    let q = QuantizedInt4::quantize(&[0.0, 1.0, 2.0]);
    // 3 values packed into 2 bytes
    assert_eq!(q.data.len(), 2);
}

#[test]
fn test_int4_odd_count() {
    let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let q = QuantizedInt4::quantize(&values);
    let dq = q.dequantize();
    assert_eq!(dq.len(), 7);
}

#[test]
fn test_int4_single_value() {
    let q = QuantizedInt4::quantize(&[42.0]);
    let dq = q.dequantize();
    assert!((dq[0] - 42.0).abs() < 0.1);
}

// ---- Sampling tests ----

#[test]
fn test_temperature_scaling() {
    let mut logits = vec![1.0, 2.0, 3.0];
    apply_temperature(&mut logits, 2.0);
    assert!((logits[0] - 0.5).abs() < 1e-6);
    assert!((logits[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_temperature_zero() {
    let mut logits = vec![1.0, 2.0, 3.0];
    let original = logits.clone();
    apply_temperature(&mut logits, 0.0);
    assert_eq!(logits, original);
}

#[test]
fn test_top_k_filter() {
    let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
    top_k_filter(&mut logits, 2);
    let finite_count = logits.iter().filter(|x| x.is_finite()).count();
    assert!(finite_count <= 3); // top-2 plus ties
}

#[test]
fn test_top_k_zero() {
    let mut logits = vec![1.0, 2.0, 3.0];
    let original = logits.clone();
    top_k_filter(&mut logits, 0);
    assert_eq!(logits, original);
}

#[test]
fn test_top_k_larger_than_len() {
    let mut logits = vec![1.0, 2.0];
    let original = logits.clone();
    top_k_filter(&mut logits, 10);
    assert_eq!(logits, original);
}

#[test]
fn test_top_p_filter() {
    let mut logits = vec![1.0, 2.0, 10.0, 0.5];
    top_p_filter(&mut logits, 0.9);
    // The highest logit should remain
    assert!(logits[2].is_finite());
}

#[test]
fn test_top_p_one() {
    let mut logits = vec![1.0, 2.0, 3.0];
    let original = logits.clone();
    top_p_filter(&mut logits, 1.0);
    assert_eq!(logits, original);
}

#[test]
fn test_softmax_basic() {
    let probs = softmax(&[0.0, 0.0, 0.0]);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!((probs[0] - probs[1]).abs() < 1e-5);
}

#[test]
fn test_softmax_large_values() {
    let probs = softmax(&[1000.0, 1000.0, 0.0]);
    assert!((probs[0] - 0.5).abs() < 1e-3);
    assert!((probs[1] - 0.5).abs() < 1e-3);
}

#[test]
fn test_softmax_empty() {
    let probs = softmax(&[]);
    assert!(probs.is_empty());
}

#[test]
fn test_softmax_single() {
    let probs = softmax(&[5.0]);
    assert!((probs[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_sample_argmax() {
    assert_eq!(sample_argmax(&[1.0, 5.0, 3.0]), 1);
    assert_eq!(sample_argmax(&[10.0, 2.0, 3.0]), 0);
}

#[test]
fn test_sample_argmax_single() {
    assert_eq!(sample_argmax(&[42.0]), 0);
}

#[test]
fn test_sample_with_random() {
    let probs = vec![0.1, 0.2, 0.7];
    assert_eq!(sample_with_random(&probs, 0.05), 0);
    assert_eq!(sample_with_random(&probs, 0.25), 1);
    assert_eq!(sample_with_random(&probs, 0.95), 2);
}

#[test]
fn test_sample_with_random_boundary() {
    let probs = vec![0.5, 0.5];
    assert_eq!(sample_with_random(&probs, 0.0), 0);
    assert_eq!(sample_with_random(&probs, 0.49), 0);
    assert_eq!(sample_with_random(&probs, 0.51), 1);
}

// ---- Linear algebra tests ----

#[test]
fn test_dot_product() {
    assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-5);
}

#[test]
fn test_dot_product_zero() {
    assert!((dot(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-5);
}

#[test]
fn test_matvec() {
    let mat = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let v = vec![3.0, 4.0];
    let out = matvec(&mat, &v);
    assert!((out[0] - 3.0).abs() < 1e-5);
    assert!((out[1] - 4.0).abs() < 1e-5);
}

#[test]
fn test_matmul_identity() {
    let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let b = vec![vec![3.0, 4.0], vec![5.0, 6.0]];
    let c = matmul(&a, &b);
    assert!((c[0][0] - 3.0).abs() < 1e-5);
    assert!((c[1][1] - 6.0).abs() < 1e-5);
}

#[test]
fn test_matmul_empty() {
    let out = matmul(&[], &[]);
    assert!(out.is_empty());
}

#[test]
fn test_transpose() {
    let m = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let t = transpose(&m);
    assert_eq!(t.len(), 3);
    assert_eq!(t[0].len(), 2);
    assert!((t[0][0] - 1.0).abs() < 1e-5);
    assert!((t[0][1] - 4.0).abs() < 1e-5);
}

#[test]
fn test_transpose_empty() {
    let t = transpose(&[]);
    assert!(t.is_empty());
}

#[test]
fn test_rms_norm() {
    let v = vec![1.0, 1.0, 1.0, 1.0];
    let n = rms_norm(&v, 1e-6);
    // RMS of [1,1,1,1] is 1, so output ~ [1,1,1,1]
    for x in &n {
        assert!((x - 1.0).abs() < 0.01);
    }
}

#[test]
fn test_rms_norm_scaling() {
    let v = vec![2.0, 2.0];
    let n = rms_norm(&v, 1e-6);
    // RMS = 2, so output ~ [1, 1]
    for x in &n {
        assert!((x - 1.0).abs() < 0.01);
    }
}

#[test]
fn test_layer_norm() {
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let n = layer_norm(&v, 1e-6);
    let mean: f32 = n.iter().sum::<f32>() / n.len() as f32;
    assert!(mean.abs() < 1e-4);
}

#[test]
fn test_layer_norm_constant() {
    let v = vec![5.0, 5.0, 5.0];
    let n = layer_norm(&v, 1e-6);
    for x in &n {
        assert!(x.abs() < 1e-3);
    }
}

#[test]
fn test_silu() {
    assert!((silu(0.0)).abs() < 1e-5);
    assert!(silu(1.0) > 0.5);
    assert!(silu(-10.0).abs() < 0.01);
}

#[test]
fn test_gelu() {
    assert!((gelu(0.0)).abs() < 1e-5);
    assert!(gelu(1.0) > 0.5);
    assert!(gelu(-3.0).abs() < 0.01);
}

#[test]
fn test_silu_monotonic() {
    let a = silu(1.0);
    let b = silu(2.0);
    let c = silu(3.0);
    assert!(a < b);
    assert!(b < c);
}

#[test]
fn test_gelu_monotonic() {
    let a = gelu(1.0);
    let b = gelu(2.0);
    let c = gelu(3.0);
    assert!(a < b);
    assert!(b < c);
}

// ---- Model tests ----

#[test]
fn test_model_creation() {
    let config = ModelConfig::default();
    let model = TransformerModel::new_test(config);
    assert_eq!(model.config.vocab_size, 256);
    assert_eq!(model.config.hidden_dim, 64);
}

#[test]
fn test_model_forward() {
    let config = ModelConfig {
        vocab_size: 32,
        hidden_dim: 16,
        num_heads: 2,
        num_layers: 1,
        max_seq_len: 32,
        rope_base: 10000.0,
    };
    let model = TransformerModel::new_test(config);
    let mut cache = KvCache::new(1, 8, 32);
    let logits = model.forward(&[0, 1, 2], &mut cache);
    assert_eq!(logits.len(), 32);
}

#[test]
fn test_model_forward_single_token() {
    let config = ModelConfig {
        vocab_size: 16,
        hidden_dim: 8,
        num_heads: 2,
        num_layers: 1,
        max_seq_len: 16,
        rope_base: 10000.0,
    };
    let model = TransformerModel::new_test(config);
    let mut cache = KvCache::new(1, 4, 16);
    let logits = model.forward(&[0], &mut cache);
    assert_eq!(logits.len(), 16);
}

#[test]
fn test_model_forward_cache_grows() {
    let config = ModelConfig {
        vocab_size: 16,
        hidden_dim: 8,
        num_heads: 2,
        num_layers: 1,
        max_seq_len: 32,
        rope_base: 10000.0,
    };
    let model = TransformerModel::new_test(config);
    let mut cache = KvCache::new(1, 4, 32);
    let _ = model.forward(&[0, 1], &mut cache);
    assert_eq!(cache.seq_len(0), 2);
    let _ = model.forward(&[2], &mut cache);
    assert_eq!(cache.seq_len(0), 3);
}

#[test]
fn test_batch_inference() {
    let config = ModelConfig {
        vocab_size: 16,
        hidden_dim: 8,
        num_heads: 2,
        num_layers: 1,
        max_seq_len: 16,
        rope_base: 10000.0,
    };
    let model = TransformerModel::new_test(config);
    let batch = vec![vec![0, 1, 2], vec![3, 4], vec![5]];
    let results = model.forward_batch(&batch);
    assert_eq!(results.len(), 3);
    for r in &results {
        assert_eq!(r.len(), 16);
    }
}

#[test]
fn test_generate() {
    let config = ModelConfig {
        vocab_size: 16,
        hidden_dim: 8,
        num_heads: 2,
        num_layers: 1,
        max_seq_len: 32,
        rope_base: 10000.0,
    };
    let model = TransformerModel::new_test(config);
    let output = model.generate(&[0, 1], 3, 1.0, 5);
    assert_eq!(output.len(), 5); // 2 prompt + 3 generated
}

#[test]
fn test_generate_deterministic() {
    let config = ModelConfig {
        vocab_size: 16,
        hidden_dim: 8,
        num_heads: 2,
        num_layers: 1,
        max_seq_len: 32,
        rope_base: 10000.0,
    };
    let model = TransformerModel::new_test(config.clone());
    let out1 = model.generate(&[0], 5, 1.0, 1);
    let model2 = TransformerModel::new_test(config);
    let out2 = model2.generate(&[0], 5, 1.0, 1);
    assert_eq!(out1, out2);
}

#[test]
fn test_model_multi_layer() {
    let config = ModelConfig {
        vocab_size: 16,
        hidden_dim: 8,
        num_heads: 2,
        num_layers: 3,
        max_seq_len: 16,
        rope_base: 10000.0,
    };
    let model = TransformerModel::new_test(config);
    let mut cache = KvCache::new(3, 4, 16);
    let logits = model.forward(&[0, 1, 2], &mut cache);
    assert_eq!(logits.len(), 16);
    assert_eq!(cache.seq_len(0), 3);
    assert_eq!(cache.seq_len(1), 3);
    assert_eq!(cache.seq_len(2), 3);
}

#[test]
fn test_model_oov_token() {
    let config = ModelConfig {
        vocab_size: 8,
        hidden_dim: 8,
        num_heads: 2,
        num_layers: 1,
        max_seq_len: 16,
        rope_base: 10000.0,
    };
    let model = TransformerModel::new_test(config);
    let mut cache = KvCache::new(1, 4, 16);
    // Token ID 999 is out of vocab
    let logits = model.forward(&[999], &mut cache);
    assert_eq!(logits.len(), 8);
}

// ---- Integration tests ----

#[test]
fn test_tokenize_and_infer() {
    let tok = BpeTokenizer::from_chars("abcd", vec![]);
    let ids = tok.encode("abcd");
    let config = ModelConfig {
        vocab_size: 8,
        hidden_dim: 8,
        num_heads: 2,
        num_layers: 1,
        max_seq_len: 16,
        rope_base: 10000.0,
    };
    let model = TransformerModel::new_test(config);
    let mut cache = KvCache::new(1, 4, 16);
    let logits = model.forward(&ids, &mut cache);
    assert_eq!(logits.len(), 8);
    let next_token = sample_argmax(&logits);
    assert!(next_token < 8);
}

#[test]
fn test_quantize_and_use() {
    let weights = vec![0.1, -0.5, 0.3, 0.8, -0.2];
    let q8 = QuantizedInt8::quantize(&weights);
    let dq8 = q8.dequantize();
    let q4 = QuantizedInt4::quantize(&weights);
    let dq4 = q4.dequantize();
    // Both should roughly match
    for i in 0..weights.len() {
        assert!((dq8[i] - weights[i]).abs() < 0.1);
        assert!((dq4[i] - weights[i]).abs() < 0.2);
    }
}

#[test]
fn test_full_pipeline() {
    let merges = vec![MergeRule {
        left: "a".into(),
        right: "b".into(),
        merged: "ab".into(),
    }];
    let tok = BpeTokenizer::from_chars("abcde", merges);
    let ids = tok.encode("abcde");

    let config = ModelConfig {
        vocab_size: 8,
        hidden_dim: 8,
        num_heads: 2,
        num_layers: 1,
        max_seq_len: 16,
        rope_base: 10000.0,
    };
    let model = TransformerModel::new_test(config);
    let results = model.forward_batch(&[ids.clone(), ids]);
    assert_eq!(results.len(), 2);
}

#[test]
fn test_softmax_then_sample() {
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let probs = softmax(&logits);
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    let idx = sample_with_random(&probs, 0.5);
    assert!(idx < probs.len());
}

#[test]
fn test_temperature_affects_distribution() {
    let mut logits_hot = vec![1.0, 2.0, 10.0];
    let mut logits_cold = logits_hot.clone();
    apply_temperature(&mut logits_hot, 100.0);
    apply_temperature(&mut logits_cold, 0.01);
    let probs_hot = softmax(&logits_hot);
    let probs_cold = softmax(&logits_cold);
    // Hot temperature -> more uniform
    assert!((probs_hot[0] - probs_hot[2]).abs() < 0.1);
    // Cold temperature -> more peaked
    assert!(probs_cold[2] > 0.99);
}

#[test]
fn test_rope_orthogonality() {
    // Two orthogonal vectors should remain orthogonal after RoPE
    let a = vec![1.0, 0.0, 0.0, 0.0];
    let b = vec![0.0, 0.0, 1.0, 0.0];
    let ra = apply_rope(&a, 3, 10000.0);
    let rb = apply_rope(&b, 3, 10000.0);
    let d = dot(&ra, &rb);
    assert!(d.abs() < 1e-4);
}

#[test]
fn test_matmul_associative() {
    let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
    let c = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let ab_c = matmul(&matmul(&a, &b), &c);
    let a_bc = matmul(&a, &matmul(&b, &c));
    for i in 0..2 {
        for j in 0..2 {
            assert!((ab_c[i][j] - a_bc[i][j]).abs() < 1e-3);
        }
    }
}

#[test]
fn test_transpose_involution() {
    let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let tt = transpose(&transpose(&m));
    assert_eq!(m, tt);
}

#[test]
fn test_kv_cache_with_model() {
    let config = ModelConfig {
        vocab_size: 8,
        hidden_dim: 8,
        num_heads: 2,
        num_layers: 2,
        max_seq_len: 32,
        rope_base: 10000.0,
    };
    let model = TransformerModel::new_test(config);
    let mut cache = KvCache::new(2, 4, 32);
    let l1 = model.forward(&[0, 1], &mut cache);
    let l2 = model.forward(&[2], &mut cache);
    assert_eq!(l1.len(), 8);
    assert_eq!(l2.len(), 8);
    assert_eq!(cache.seq_len(0), 3);
    assert_eq!(cache.seq_len(1), 3);
}

#[test]
fn test_int8_large_range() {
    let values: Vec<f32> = (-100..=100).map(|x| x as f32).collect();
    let q = QuantizedInt8::quantize(&values);
    let dq = q.dequantize();
    assert_eq!(dq.len(), 201);
    for (orig, deq) in values.iter().zip(dq.iter()) {
        assert!((orig - deq).abs() < 2.0);
    }
}

#[test]
fn test_int4_large_range() {
    let values: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let q = QuantizedInt4::quantize(&values);
    let dq = q.dequantize();
    assert_eq!(dq.len(), 16);
}

#[test]
fn test_default_model_config() {
    let config = ModelConfig::default();
    assert_eq!(config.vocab_size, 256);
    assert_eq!(config.hidden_dim, 64);
    assert_eq!(config.num_heads, 4);
    assert_eq!(config.num_layers, 2);
    assert_eq!(config.max_seq_len, 128);
    assert!((config.rope_base - 10000.0).abs() < 1e-5);
}

// ---- Additional tests to reach 100+ ----

#[test]
fn test_softmax_inplace_uniform() {
    let mut v = vec![0.0; 4];
    softmax_inplace(&mut v);
    for x in &v {
        assert!((x - 0.25).abs() < 1e-5);
    }
}

#[test]
fn test_softmax_inplace_peaked() {
    let mut v = vec![0.0, 0.0, 100.0];
    softmax_inplace(&mut v);
    assert!(v[2] > 0.99);
}

#[test]
fn test_dot_empty() {
    assert!((dot(&[], &[])).abs() < 1e-9);
}

#[test]
fn test_matvec_single_row() {
    let mat = vec![vec![2.0, 3.0]];
    let result = matvec(&mat, &[4.0, 5.0]);
    assert!((result[0] - 23.0).abs() < 1e-5);
}

#[test]
fn test_matmul_non_square() {
    let a = vec![vec![1.0, 2.0, 3.0]];
    let b = vec![vec![4.0], vec![5.0], vec![6.0]];
    let c = matmul(&a, &b);
    assert_eq!(c.len(), 1);
    assert_eq!(c[0].len(), 1);
    assert!((c[0][0] - 32.0).abs() < 1e-5);
}

#[test]
fn test_rope_large_position() {
    let v = vec![1.0, 0.5, 0.5, 1.0];
    let r = apply_rope(&v, 1000, 10000.0);
    let norm: f32 = r.iter().map(|x| x * x).sum::<f32>().sqrt();
    let orig_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - orig_norm).abs() < 1e-3);
}

#[test]
fn test_rope_batch_empty() {
    let result = apply_rope_batch(&[], 10000.0);
    assert!(result.is_empty());
}

#[test]
fn test_int8_single_value() {
    let q = QuantizedInt8::quantize(&[3.14]);
    let dq = q.dequantize();
    assert!((dq[0] - 3.14).abs() < 0.1);
}

#[test]
fn test_int4_constant() {
    let q = QuantizedInt4::quantize(&[7.0, 7.0, 7.0]);
    let dq = q.dequantize();
    for v in &dq {
        assert!((v - 7.0).abs() < 0.1);
    }
}

#[test]
fn test_top_k_keeps_highest() {
    let mut logits = vec![1.0, 10.0, 2.0, 3.0, 4.0];
    top_k_filter(&mut logits, 1);
    assert!(logits[1].is_finite());
    assert_eq!(sample_argmax(&logits), 1);
}

#[test]
fn test_generate_prompt_preserved() {
    let config = ModelConfig {
        vocab_size: 16,
        hidden_dim: 8,
        num_heads: 2,
        num_layers: 1,
        max_seq_len: 32,
        rope_base: 10000.0,
    };
    let model = TransformerModel::new_test(config);
    let prompt = vec![3, 7, 11];
    let output = model.generate(&prompt, 2, 1.0, 5);
    assert_eq!(&output[..3], &prompt);
    assert_eq!(output.len(), 5);
}

#[test]
fn test_layer_norm_two_elements() {
    let v = vec![0.0, 2.0];
    let n = layer_norm(&v, 1e-6);
    assert!((n[0] + n[1]).abs() < 1e-4);
}

#[test]
fn test_causal_mask_single() {
    let mask = causal_mask(1);
    assert_eq!(mask.len(), 1);
    assert_eq!(mask[0][0], 0.0);
}

#[test]
fn test_kv_cache_seq_len_after_clear() {
    let mut cache = KvCache::new(2, 8, 16);
    cache.append(0, vec![0.0; 8], vec![0.0; 8]);
    cache.append(0, vec![0.0; 8], vec![0.0; 8]);
    assert_eq!(cache.seq_len(0), 2);
    cache.clear();
    assert_eq!(cache.seq_len(0), 0);
    assert!(cache.append(0, vec![0.0; 8], vec![0.0; 8]));
    assert_eq!(cache.seq_len(0), 1);
}
