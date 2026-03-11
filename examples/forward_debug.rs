//! Forward pass debugging — check weights, dims, intermediate values.

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::llama3::Llama3Model;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_path = args.get(1).expect("Usage: forward_debug <model.gguf>");
    let data = fs::read(model_path).expect("Failed to read");
    let gguf = GgufFile::parse(&data).expect("Failed to parse");

    // Check tensor dimensions
    println!("=== Tensor dimensions ===");
    let check = [
        "token_embd.weight", "output.weight", "output_norm.weight",
        "blk.0.attn_norm.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight",
        "blk.0.attn_v.weight", "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
    ];
    for name in check {
        if let Some(info) = gguf.tensor_info(name) {
            println!("  {}: dims={:?}, qtype={:?}, n_elements={}",
                name, info.dims, info.qtype, info.n_elements());
        } else {
            println!("  {}: NOT FOUND", name);
        }
    }

    // Check embedding values
    println!("\n=== Embedding check ===");
    let emb = gguf.tensor_to_f32("token_embd.weight").expect("embedding");
    let emb_info = gguf.tensor_info("token_embd.weight").unwrap();
    println!("  Shape: {:?}, total elements: {}", emb_info.dims, emb.len());

    // For Llama-3: dims should be [hidden_dim=4096, vocab_size=128256] in GGUF
    // But we access as emb[token_id * hidden_dim .. (token_id+1) * hidden_dim]
    // So we need dims[0] = hidden_dim, dims[1] = vocab_size
    let dim0 = emb_info.dims[0] as usize;
    let dim1 = if emb_info.dims.len() > 1 { emb_info.dims[1] as usize } else { 1 };
    println!("  dim0={dim0}, dim1={dim1}");

    // Show first few elements of token 0 embedding
    println!("  token 0 (if row-major, stride=dim0): {:?}", &emb[0..8.min(emb.len())]);
    println!("  token 1 (if row-major, stride=dim0): {:?}", &emb[dim0..dim0+8.min(emb.len())]);

    // Check if embedding looks reasonable (not all zeros, not all same)
    let sum: f32 = emb[0..dim0.min(emb.len())].iter().sum();
    let max = emb[0..dim0.min(emb.len())].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = emb[0..dim0.min(emb.len())].iter().cloned().fold(f32::INFINITY, f32::min);
    println!("  token 0 stats: sum={sum:.4}, min={min:.4}, max={max:.4}");

    // Check norm weights
    println!("\n=== Norm weights ===");
    let norm = gguf.tensor_to_f32("output_norm.weight").expect("output_norm");
    println!("  output_norm: len={}, first_8={:?}", norm.len(), &norm[0..8]);
    let norm0 = gguf.tensor_to_f32("blk.0.attn_norm.weight").expect("attn_norm");
    println!("  blk.0.attn_norm: len={}, first_8={:?}", norm0.len(), &norm0[0..8]);

    // Load model and run single forward
    println!("\n=== Forward pass ===");
    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("tokenizer");
    let mut model = Llama3Model::from_gguf(&gguf).expect("model");

    // Forward BOS token
    let logits = model.forward(tokenizer.bos_id);
    println!("  logits len: {}", logits.len());
    let logits_sum: f32 = logits.iter().sum();
    let logits_max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let logits_min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let argmax = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
    println!("  logits stats: sum={logits_sum:.2}, min={logits_min:.4}, max={logits_max:.4}");
    println!("  argmax: token_id={}, logit={:.4}", argmax.0, argmax.1);
    println!("  argmax decode: '{}'", tokenizer.decode(&[argmax.0 as u32]));

    // Top-10 tokens
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\n  Top-10 after BOS:");
    for (idx, logit) in indexed.iter().take(10) {
        let decoded = tokenizer.decode(&[*idx as u32]);
        println!("    [{idx}] {logit:.4} '{decoded}'");
    }
}
