//! Debug intermediate values at each layer of the forward pass.

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::llama3::Llama3Model;
use std::env;
use std::fs;

fn vec_stats(v: &[f32]) -> (f32, f32, f32, f32) {
    let sum: f32 = v.iter().sum();
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let l2 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    (sum, min, max, l2)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_path = args.get(1).expect("Usage: layer_debug <model.gguf>");
    let data = fs::read(model_path).expect("Failed to read");
    let gguf = GgufFile::parse(&data).expect("Failed to parse");
    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("tokenizer");
    let mut model = Llama3Model::from_gguf(&gguf).expect("model");

    // Check for double BOS issue
    let prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    let encoded = tokenizer.encode(prompt);
    println!("BOS ID: {}", tokenizer.bos_id);
    println!("Encoded prompt: {:?}", &encoded);
    println!("First token == BOS? {}", encoded.first() == Some(&tokenizer.bos_id));
    println!();

    // Test: forward BOS, then check logits
    println!("=== Forward single BOS ===");
    let logits = model.forward(tokenizer.bos_id);
    let (sum, min, max, l2) = vec_stats(&logits);
    println!("  logits: sum={sum:.2}, min={min:.4}, max={max:.4}, L2={l2:.2}");

    // Forward a few more tokens from the prompt and check logits
    println!("\n=== Forward prompt tokens sequentially ===");
    model.clear_cache();

    // Don't double BOS - if prompt starts with <|begin_of_text|>, don't add BOS
    let tokens: Vec<u32> = if encoded.first() == Some(&tokenizer.bos_id) {
        encoded.clone()
    } else {
        let mut t = vec![tokenizer.bos_id];
        t.extend_from_slice(&encoded);
        t
    };

    println!("Tokens to process: {} total", tokens.len());
    let mut last_logits = Vec::new();
    for (i, &tok) in tokens.iter().enumerate() {
        last_logits = model.forward(tok);
        let (sum, min, max, l2) = vec_stats(&last_logits);
        let argmax = last_logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        let decoded = tokenizer.decode(&[argmax.0 as u32]);
        println!("  [{i}] tok={tok:6} logits: sum={sum:8.1}, min={min:8.3}, max={max:8.3}, L2={l2:8.1}, argmax={} '{decoded}'",
            argmax.0);
    }

    // Now generate 3 tokens from the last logits
    println!("\n=== Generate from last logits ===");
    for i in 0..3 {
        let argmax = last_logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        let next_id = argmax.0 as u32;
        let decoded = tokenizer.decode(&[next_id]);
        println!("  gen[{i}] token={next_id} logit={:.4} '{decoded}'", argmax.1);
        last_logits = model.forward(next_id);
    }
}
