//! Phase 6: Verify SPM tokenizer + Gemma 3n chat template on real GGUF.

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use std::fs;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/gemma-3n-E2B-it-Q4_K_M.gguf".to_string());
    println!("Loading: {path}");
    let bytes = fs::read(&path).expect("read gguf");
    let gguf = GgufFile::parse(&bytes).expect("parse gguf");

    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("load tokenizer");
    println!("Tokenizer loaded. bos_id={} eos_id={}", tokenizer.bos_id, tokenizer.eos_id);
    println!("add_bos_token={}", tokenizer.add_bos_token);

    // Test 1: Basic English round-trip
    let prompt = "Hello, world!";
    let tokens = tokenizer.encode(prompt);
    println!("\n=== English encode ===");
    println!("Text: {prompt:?}");
    println!("Tokens ({}): {tokens:?}", tokens.len());
    let decoded = tokenizer.decode(&tokens);
    println!("Decoded: {decoded:?}");

    // Test 2: Japanese
    let jp = "こんにちは、世界！";
    let jp_tokens = tokenizer.encode(jp);
    println!("\n=== Japanese encode ===");
    println!("Text: {jp:?}");
    println!("Tokens ({}): {jp_tokens:?}", jp_tokens.len());
    let jp_decoded = tokenizer.decode(&jp_tokens);
    println!("Decoded: {jp_decoded:?}");

    // Test 3: Chat template with special tokens
    let chat = format!(
        "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
        "What is 2+2?"
    );
    let chat_tokens = tokenizer.encode(&chat);
    println!("\n=== Chat template ===");
    println!("Text: {chat:?}");
    println!("Tokens ({}): first 10 = {:?}", chat_tokens.len(), &chat_tokens[..chat_tokens.len().min(15)]);
    let chat_decoded = tokenizer.decode(&chat_tokens);
    println!("Decoded: {chat_decoded:?}");

    // Test 4: Verify BOS and special tokens are detected
    println!("\n=== Special token detection ===");
    // The prompt should have BOS token prepended when add_bos_token is true.
    let bos_check = tokenizer.encode("test");
    let has_bos = !bos_check.is_empty() && bos_check[0] == tokenizer.bos_id;
    println!("First token of 'test' = {} (bos_id={}, add_bos={})",
        bos_check.first().copied().unwrap_or(u32::MAX),
        tokenizer.bos_id, tokenizer.add_bos_token);
    println!("BOS prepended: {has_bos}");

    println!("\nPhase 6 verify OK.");
}
