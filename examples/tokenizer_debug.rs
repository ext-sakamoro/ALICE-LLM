//! Tokenizer debugging tool.

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_path = args.get(1).expect("Usage: tokenizer_debug <model.gguf>");

    let data = fs::read(model_path).expect("Failed to read");
    let gguf = GgufFile::parse(&data).expect("Failed to parse");
    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("tokenizer");

    println!("Vocab size: {}", tokenizer.vocab_size());
    println!("BOS: {}, EOS: {}", tokenizer.bos_id, tokenizer.eos_id);

    // Check if single-byte tokens exist
    println!("\n--- Single byte tokens ---");
    for b in [b'H', b'e', b'l', b'o', b' ', b'\n'] {
        let encoded = tokenizer.encode(&String::from(b as char));
        println!("  byte 0x{:02X} '{}' → encode={:?}", b, b as char, encoded);
    }

    // Check some token entries by looking at what IDs are produced
    println!("\n--- Encode tests ---");
    let tests = ["Hello", "Hello world", "日本", "the", " the",
                  "<|begin_of_text|>", "<|start_header_id|>"];
    for t in tests {
        let ids = tokenizer.encode(t);
        let decoded = tokenizer.decode(&ids);
        println!("  '{}' → ids={:?} → decode='{}'", t, ids, decoded);
    }

    // Show what some token IDs look like
    println!("\n--- Token samples ---");
    for id in [0u32, 1, 2, 128000, 128001, 128006, 128007, 128008, 128009] {
        let decoded = tokenizer.decode(&[id]);
        println!("  token[{}] → '{}'", id, decoded);
    }

    // Test the instruction format
    println!("\n--- Instruction format ---");
    let prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    let ids = tokenizer.encode(prompt);
    println!("  {} tokens: {:?}", ids.len(), &ids[..ids.len().min(30)]);
    let decoded = tokenizer.decode(&ids);
    println!("  decode: '{}'", decoded);
}
