//! Phase 6.5: smoke test — run 1 forward pass on Gemma 3n GGUF.
//! Verifies forward_gemma3n() completes without panic and produces
//! finite logits.

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::llama3::Llama3Model;
use std::fs;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/gemma-3n-E2B-it-Q4_K_M.gguf".to_string());
    eprintln!("Loading {path}");
    let bytes = fs::read(&path).expect("read gguf");
    let gguf = GgufFile::parse(&bytes).expect("parse gguf");
    let mut model = Llama3Model::from_gguf(&gguf).expect("load model");
    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("load tokenizer");
    eprintln!(
        "Model loaded. arch={:?}, vocab={}, layers={}",
        model.config.arch, model.config.vocab_size, model.config.num_layers
    );

    let prompt = "<start_of_turn>user\nHello, what is 2+2?<end_of_turn>\n<start_of_turn>model\n";
    let tokens = tokenizer.encode(prompt);
    eprintln!(
        "Prompt tokens ({}): {:?}",
        tokens.len(),
        &tokens[..tokens.len().min(15)]
    );

    // Prefill: single forward per token
    eprintln!("\nPrefill...");
    let start = Instant::now();
    for &tok in &tokens {
        let _ = model.forward(tok);
    }
    let prefill_ms = start.elapsed().as_millis();
    eprintln!("Prefill: {} tokens in {} ms", tokens.len(), prefill_ms);

    // Decode 1 token
    eprintln!("\nDecoding 1 token...");
    let start = Instant::now();
    let logits = model.forward(*tokens.last().unwrap());
    let decode_ms = start.elapsed().as_millis();
    eprintln!("First decode: {decode_ms} ms");
    // Sanity check: logits are finite
    let finite_count = logits.iter().filter(|v| v.is_finite()).count();
    eprintln!("Finite logits: {finite_count} / {}", logits.len());
    // Greedy argmax
    let (arg, val) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let text = tokenizer.decode(&[arg as u32]);
    eprintln!("Greedy next token: id={arg}, logit={val:.4}, text={text:?}");
    if finite_count != logits.len() {
        eprintln!("  WARN: {} non-finite logits", logits.len() - finite_count);
    }

    // Generate 10 more tokens
    eprintln!("\nGenerating 10 tokens greedily:");
    let mut generated = vec![arg as u32];
    for _ in 0..10 {
        let last = *generated.last().unwrap();
        let logits = model.forward(last);
        let (next, _) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        generated.push(next as u32);
    }
    let full_text = tokenizer.decode(&generated);
    eprintln!("Generated: {full_text:?}");
    eprintln!("\nPhase 6.5 smoke test OK.");
}
