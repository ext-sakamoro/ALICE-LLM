//! GPU-accelerated autoregressive text generation with Llama-3.
//!
//! End-to-end: GGUF load → GPU upload → prefill → decode → streaming output.
//!
//! Usage:
//!   cargo run --example generate_gpu --features gpu,gguf --release -- \
//!     --prompt "The meaning of life is"
//!   cargo run --example generate_gpu --features gpu,gguf --release -- \
//!     --model path/to/model.gguf --prompt "Hello" --max-tokens 128 --temperature 0.8

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::{sample_argmax, apply_temperature, top_k_filter, softmax_inplace, sample_with_random};
use std::io::Write;
use std::time::Instant;

#[cfg(feature = "gpu")]
use alice_llm::gpu::{GpuEngine, GpuModel, GpuModelConfig};

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let model_path = parse_arg::<String>(&args, "--model").unwrap_or_else(|| {
        "/Users/ys/models/llama-3.2-1b-gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_string()
    });

    let raw_prompt = parse_arg::<String>(&args, "--prompt")
        .unwrap_or_else(|| "The meaning of life is".to_string());

    let max_tokens: usize = parse_arg(&args, "--max-tokens").unwrap_or(256);
    let temperature: f32 = parse_arg(&args, "--temperature").unwrap_or(0.0);
    let top_k: usize = parse_arg(&args, "--top-k").unwrap_or(40);

    #[cfg(feature = "gpu")]
    {
        // --- Load GGUF ---
        println!("Loading model: {model_path}");
        let t0 = Instant::now();
        let data = std::fs::read(&model_path).expect("Failed to read GGUF file");
        let gguf = GgufFile::parse(&data).expect("Failed to parse GGUF");
        let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");
        println!("  GGUF parsed: {}ms (vocab={})", t0.elapsed().as_millis(), tokenizer.vocab_size());

        // --- Model config (Llama-3.2-1B) ---
        let config = GpuModelConfig {
            num_layers: 16,
            hidden_dim: 2048,
            intermediate_dim: 8192,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            rope_theta: 500000.0,
            eps: 1e-5,
            max_seq_len: 2048,
        };

        // --- GPU init ---
        let t_gpu = Instant::now();
        let engine = GpuEngine::new();
        let mut model = GpuModel::load(engine, &gguf, config);
        println!("  GPU model ready: {}ms", t_gpu.elapsed().as_millis());

        // --- Format as Llama-3 instruct ---
        let formatted = format!(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{raw_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );

        // --- Tokenize ---
        let prompt_tokens = tokenizer.encode(&formatted);
        println!("  Prompt: {} tokens", prompt_tokens.len());
        println!("  Generating (max {max_tokens} tokens, temp={temperature}, top_k={top_k})");
        println!("---");

        // --- Prefill: process all prompt tokens ---
        let t_prefill = Instant::now();
        for &tok in &prompt_tokens[..prompt_tokens.len() - 1] {
            model.forward(tok);
        }
        // Last prompt token: read logits to get first generated token
        let last_prompt = *prompt_tokens.last().unwrap();
        let mut logits = model.forward_and_read(last_prompt);
        let prefill_ms = t_prefill.elapsed().as_millis();

        // --- Decode: autoregressive loop ---
        let t_decode = Instant::now();
        let mut generated_tokens: Vec<u32> = Vec::new();
        let eos_id = tokenizer.eos_id;
        // Llama-3 instruct stop tokens
        let eot_id: u32 = 128009; // <|eot_id|>

        // Simple xorshift64 PRNG for sampling
        let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_1234;
        let mut next_rand = || -> f32 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state as f32) / (u64::MAX as f32)
        };

        for _ in 0..max_tokens {
            let next_token = if temperature < 1e-6 {
                sample_argmax(&logits) as u32
            } else {
                apply_temperature(&mut logits, temperature);
                top_k_filter(&mut logits, top_k);
                softmax_inplace(&mut logits);
                sample_with_random(&logits, next_rand()) as u32
            };

            if next_token == eos_id || next_token == eot_id {
                break;
            }

            generated_tokens.push(next_token);

            // Streaming output: decode and print this token immediately
            let text = tokenizer.decode(&[next_token]);
            print!("{text}");
            std::io::stdout().flush().ok();

            // Next forward pass
            logits = model.forward_and_read(next_token);
        }

        let decode_ms = t_decode.elapsed().as_millis();
        let total_ms = prefill_ms + decode_ms;
        let n_gen = generated_tokens.len();
        let tok_per_sec = if decode_ms > 0 {
            n_gen as f64 / (decode_ms as f64 / 1000.0)
        } else {
            0.0
        };

        println!();
        println!("---");
        println!(
            "{n_gen} tokens generated, {:.1} tok/s ({prefill_ms}ms prefill + {decode_ms}ms decode = {total_ms}ms)",
            tok_per_sec,
        );
        println!(
            "Avg: {:.1}ms/token (decode only), position={}",
            if n_gen > 0 { decode_ms as f64 / n_gen as f64 } else { 0.0 },
            model.position(),
        );
    }

    #[cfg(not(feature = "gpu"))]
    println!("GPU feature not enabled. Run with: cargo run --example generate_gpu --features gpu,gguf --release");
}
