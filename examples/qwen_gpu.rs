//! GPU-accelerated Qwen 2 / 2.5 / 3 inference example (wgpu: Metal / Vulkan / DX12).
//!
//! Loads model config automatically from GGUF metadata via `Llama3Config::from_gguf`,
//! applies Qwen ChatML template, streams output token-by-token.
//!
//! Usage:
//!   cargo run --release --example qwen_gpu --features "gguf,gpu" -- \
//!     --model models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
//!     --prompt "fn fibonacci(n: u32) -> u32 {" \
//!     --max-tokens 100 --temperature 0.0

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::llama3::Llama3Config;
use alice_llm::{
    apply_temperature, sample_argmax, sample_with_random, softmax_inplace, top_k_filter,
};
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

#[cfg(feature = "gpu")]
fn gpu_config_from_llama3(cfg: &Llama3Config) -> GpuModelConfig {
    GpuModelConfig {
        num_layers: cfg.num_layers,
        hidden_dim: cfg.hidden_dim,
        intermediate_dim: cfg.intermediate_dim,
        num_heads: cfg.num_heads as u32,
        num_kv_heads: cfg.num_kv_heads as u32,
        head_dim: cfg.head_dim as u32,
        rope_theta: cfg.rope_theta,
        eps: cfg.norm_eps,
        max_seq_len: cfg.max_seq_len,
        // Qwen3.5 DeltaNet-only fields — None for standard attention (Qwen 2 / 2.5 / 3).
        // GpuModel::load treats None `full_attention_interval` as pure-attention model.
        full_attention_interval: cfg.full_attention_interval,
        linear_num_kv_heads: cfg.linear_num_kv_heads.map(|v| v as u32),
        linear_qk_head_dim: cfg.linear_qk_head_dim.map(|v| v as u32),
        linear_kv_head_dim: cfg.linear_kv_head_dim.map(|v| v as u32),
        linear_num_v_heads: cfg.linear_num_v_heads.map(|v| v as u32),
        linear_conv_kernel_dim: cfg.linear_conv_kernel_dim.map(|v| v as u32),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let model_path = parse_arg::<String>(&args, "--model").expect("Usage: --model <path.gguf>");
    let raw_prompt =
        parse_arg::<String>(&args, "--prompt").unwrap_or_else(|| "Hello".to_string());
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
        println!(
            "  GGUF parsed: {}ms (vocab={})",
            t0.elapsed().as_millis(),
            tokenizer.vocab_size()
        );

        if let Some(arch) = gguf.meta_str("general.architecture") {
            println!("  Architecture: {arch}");
        }
        if let Some(name) = gguf.meta_str("general.name") {
            println!("  Model name: {name}");
        }

        // --- Auto-detect config from GGUF metadata ---
        let llama_cfg =
            Llama3Config::from_gguf(&gguf).expect("Failed to load Llama3Config from GGUF");
        println!(
            "  Config: layers={} hidden={} heads={}/{} head_dim={} rope_theta={}",
            llama_cfg.num_layers,
            llama_cfg.hidden_dim,
            llama_cfg.num_heads,
            llama_cfg.num_kv_heads,
            llama_cfg.head_dim,
            llama_cfg.rope_theta
        );

        let config = gpu_config_from_llama3(&llama_cfg);

        // --- GPU init ---
        let t_gpu = Instant::now();
        let engine = GpuEngine::new();
        let mut model = GpuModel::load(engine, &gguf, config);
        println!("  GPU model ready: {}ms", t_gpu.elapsed().as_millis());

        // --- Qwen ChatML template ---
        let formatted =
            format!("<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n");

        // --- Tokenize ---
        let prompt_tokens = tokenizer.encode(&formatted);
        println!("  Prompt: {} tokens", prompt_tokens.len());
        println!(
            "  Generating (max {max_tokens} tokens, temp={temperature}, top_k={top_k})"
        );
        println!("---");

        // --- Prefill: process all prompt tokens except the last ---
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
        // Qwen 2 / 2.5 / 3 ChatML stop tokens (hardcoded IDs are stable in Qwen tokenizer)
        let im_end_id: u32 = 151645; // <|im_end|>
        let endoftext_id: u32 = 151643; // <|endoftext|>

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

            if next_token == eos_id || next_token == im_end_id || next_token == endoftext_id {
                break;
            }

            generated_tokens.push(next_token);

            let text = tokenizer.decode(&[next_token]);
            print!("{text}");
            std::io::stdout().flush().ok();

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
            "{n_gen} tokens generated, {tok_per_sec:.1} tok/s ({prefill_ms}ms prefill + {decode_ms}ms decode = {total_ms}ms)"
        );
        println!(
            "Avg: {avg:.1}ms/token (decode only), position={pos}",
            avg = if n_gen > 0 {
                decode_ms as f64 / n_gen as f64
            } else {
                0.0
            },
            pos = model.position(),
        );
    }

    #[cfg(not(feature = "gpu"))]
    {
        let _ = (&model_path, &raw_prompt, max_tokens, temperature, top_k);
        println!(
            "GPU feature not enabled. Run with: cargo run --release --example qwen_gpu --features \"gguf,gpu\""
        );
    }
}
