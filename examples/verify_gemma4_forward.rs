//! Phase M3: smoke test — run forward on Gemma 4 E2B QAT Q4_0 GGUF.

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::llama3::Llama3Model;
use std::fs;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/gemma-4-E2B_q4_0-it.gguf".to_string());
    eprintln!("Loading {path}");
    let bytes = fs::read(&path).expect("read gguf");
    let gguf = GgufFile::parse(&bytes).expect("parse gguf");
    let mut model = Llama3Model::from_gguf(&gguf).expect("load model");
    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("load tokenizer");
    eprintln!(
        "Model loaded. arch={:?}, vocab={}, layers={}",
        model.config.arch, model.config.vocab_size, model.config.num_layers
    );

    let prompt =
        "<start_of_turn>user\nWhat is the capital of Japan? Answer in one word.<end_of_turn>\n<start_of_turn>model\n";
    let tokens = tokenizer.encode(prompt);
    eprintln!(
        "Prompt tokens ({}): {:?}",
        tokens.len(),
        &tokens[..tokens.len().min(15)]
    );

    eprintln!("\nPrefill (n-1 tokens, save last for decode)...");
    let start = Instant::now();
    for &tok in &tokens[..tokens.len() - 1] {
        let _ = model.forward(tok);
    }
    let prefill_ms = start.elapsed().as_millis();
    eprintln!("Prefill: {} tokens in {} ms", tokens.len() - 1, prefill_ms);

    eprintln!("\nDecoding 1 token (feeding last prompt token)...");
    let start = Instant::now();
    let logits = model.forward(*tokens.last().unwrap());
    let decode_ms = start.elapsed().as_millis();
    eprintln!("First decode: {decode_ms} ms");
    // Top 5
    let mut idx_val: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    idx_val.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("Top 5:");
    for &(i, v) in idx_val.iter().take(5) {
        eprintln!(
            "  id={i}, logit={v:.4}, text={:?}",
            tokenizer.decode(&[i as u32])
        );
    }
    let finite_count = logits.iter().filter(|v| v.is_finite()).count();
    eprintln!("Finite logits: {finite_count} / {}", logits.len());
    let (arg, val) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let text = tokenizer.decode(&[arg as u32]);
    eprintln!("Greedy next token: id={arg}, logit={val:.4}, text={text:?}");

    eprintln!("\nGenerating 15 tokens greedily:");
    let mut generated = vec![arg as u32];
    for _ in 0..15 {
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
    eprintln!("\nPhase M3 smoke test OK.");
}
