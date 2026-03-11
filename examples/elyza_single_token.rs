//! Single-token test for ELYZA model — verify correctness.

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::llama3::Llama3Model;
use std::env;
use std::fs;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_path = args.get(1).expect("Usage: elyza_single_token <model.gguf>");

    println!("Loading model...");
    let load_start = Instant::now();
    let data = fs::read(model_path).expect("Failed to read");
    let gguf = GgufFile::parse(&data).expect("Failed to parse");
    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("tokenizer");
    let mut model = Llama3Model::from_gguf(&gguf).expect("model");
    println!("Loaded in {}ms", load_start.elapsed().as_millis());

    // Test: generate 3 tokens
    println!("\nGenerating 3 tokens...");
    let result = model.generate(&tokenizer, "Hello", 3, 0.0, 1);
    println!("Output: {:?}", result.text);
    println!("Tokens: {}, Speed: {:.2} tok/s", result.tokens_generated, result.tokens_per_sec);
    println!("Prefill: {}ms, Decode: {}ms", result.prefill_ms, result.decode_ms);
}
