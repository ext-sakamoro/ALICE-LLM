//! GPU speculative decoding: K=3 draft (autoregressive) + batch-4 verify.
//!
//! Single-model self-speculative decoding to validate the loop structure.
//! With same model, acceptance ≈ 100% (greedy) — real speedup requires
//! a larger verifier model (8B/70B) with the 1B as draft.
//!
//! Flow per round:
//!   1. Draft: K=1 forward × 3 → draft_tokens[0..2], draft_logits[0..2]
//!   2. Rollback to pre-draft position
//!   3. Verify: batch-4 forward([current, d0, d1, d2]) → verify_logits[0..3]
//!   4. Accept/reject (Leviathan et al. probabilistic sampling)
//!   5. Rollback to last accepted position
//!   6. Output: up to 4 tokens (3 accepted + 1 bonus)
//!
//! Usage:
//!   cargo run --example speculative_gpu --features gpu,gguf --release -- \
//!     --prompt "The meaning of life is" --max-tokens 128

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::{sample_argmax, softmax, sample_with_random};
use std::time::Instant;

#[cfg(feature = "gpu")]
use alice_llm::gpu::{GpuEngine, GpuModel, GpuModelConfig};

const K_DRAFT: usize = 3;
const EOT_ID: u32 = 128009; // <|eot_id|>

/// Simple xorshift64 PRNG.
struct Rng64(u64);
impl Rng64 {
    fn new(seed: u64) -> Self { Self(seed) }
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 as f32) / (u64::MAX as f32)
    }
}

fn is_stop_token(tok: u32, eos_id: u32) -> bool {
    tok == eos_id || tok == EOT_ID
}

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
    let max_tokens: usize = parse_arg(&args, "--max-tokens").unwrap_or(128);

    #[cfg(feature = "gpu")]
    {
        println!("=== GPU Speculative Decoding (K_draft={K_DRAFT}, batch-4 verify) ===\n");

        // --- Load GGUF ---
        println!("Loading model: {model_path}");
        let data = std::fs::read(&model_path).expect("Failed to read GGUF file");
        let gguf = GgufFile::parse(&data).expect("Failed to parse GGUF");
        let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");

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

        let engine = GpuEngine::new();
        let mut model = GpuModel::load(engine, &gguf, config);
        let vocab = model.vocab_size();

        // --- Format prompt ---
        let formatted = format!(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{raw_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        let prompt_tokens = tokenizer.encode(&formatted);
        println!("  Prompt: {} tokens, vocab={vocab}", prompt_tokens.len());

        // =====================================================
        // Baseline: naive autoregressive (K=1)
        // =====================================================
        println!("\n--- Baseline (naive K=1 autoregressive) ---");
        model.reset();
        let t_baseline = Instant::now();
        // Prefill
        for &tok in &prompt_tokens[..prompt_tokens.len() - 1] {
            model.forward(tok);
        }
        let mut logits = model.forward_and_read(*prompt_tokens.last().unwrap());

        let mut baseline_tokens: Vec<u32> = Vec::new();
        for _ in 0..max_tokens {
            let next = sample_argmax(&logits) as u32;
            if is_stop_token(next, tokenizer.eos_id) { break; }
            baseline_tokens.push(next);
            logits = model.forward_and_read(next);
        }
        let baseline_ms = t_baseline.elapsed().as_millis();
        let baseline_text = tokenizer.decode(&baseline_tokens);
        println!("{baseline_text}");
        println!("---");
        let baseline_tps = if baseline_ms > 0 {
            baseline_tokens.len() as f64 / (baseline_ms as f64 / 1000.0)
        } else { 0.0 };
        println!(
            "{} tokens, {:.1} tok/s ({baseline_ms}ms)",
            baseline_tokens.len(), baseline_tps,
        );

        // =====================================================
        // Speculative: K=3 draft + batch-4 verify
        // =====================================================
        println!("\n--- Speculative (K={K_DRAFT} draft + batch-4 verify) ---");
        model.reset();
        let mut rng = Rng64::new(42);

        let t_spec = Instant::now();
        // Prefill
        for &tok in &prompt_tokens[..prompt_tokens.len() - 1] {
            model.forward(tok);
        }
        logits = model.forward_and_read(*prompt_tokens.last().unwrap());

        let mut spec_tokens: Vec<u32> = Vec::new();
        let mut total_drafted: usize = 0;
        let mut total_accepted: usize = 0;
        let mut total_rounds: usize = 0;

        'outer: while spec_tokens.len() < max_tokens {
            // Sample current token from logits
            let current_token = sample_argmax(&logits) as u32;
            if is_stop_token(current_token, tokenizer.eos_id) { break; }
            spec_tokens.push(current_token);

            let remaining = max_tokens - spec_tokens.len();
            if remaining == 0 {
                logits = model.forward_and_read(current_token);
                continue;
            }

            let k = K_DRAFT.min(remaining);

            // --- Draft phase: K=1 forward × k ---
            let saved_pos = model.position();
            let mut draft_tokens: Vec<u32> = Vec::with_capacity(k);
            let mut draft_logits: Vec<Vec<f32>> = Vec::with_capacity(k);
            let mut draft_input = current_token;

            for _ in 0..k {
                let dl = model.forward_and_read(draft_input);
                draft_input = sample_argmax(&dl) as u32;
                draft_tokens.push(draft_input);
                draft_logits.push(dl);
            }
            total_drafted += draft_tokens.len();

            // --- Rollback to pre-draft position ---
            model.rollback_to(saved_pos);

            // --- Verify phase: batch-4 forward ---
            // Pad to 4 tokens if k < 3
            let mut verify_input = [current_token, 0u32, 0, 0];
            for i in 0..k {
                verify_input[i + 1] = draft_tokens[i];
            }
            // Fill padding slots with last valid token (harmless, logits ignored)
            for i in (k + 1)..4 {
                verify_input[i] = verify_input[k];
            }

            let verify_logits_flat = model.forward_batch_and_read(&verify_input);
            // verify_logits_flat: [batch0_vocab..., batch1_vocab..., batch2_vocab..., batch3_vocab...]

            // --- Accept/reject (Leviathan et al.) ---
            let mut num_accepted: usize = 0;
            let mut all_accepted = true;

            for i in 0..k {
                let p = softmax(&verify_logits_flat[i * vocab..(i + 1) * vocab]);
                let q = softmax(&draft_logits[i]);
                let x = draft_tokens[i] as usize;

                let p_x = if x < p.len() { p[x] } else { 0.0 };
                let q_x = if x < q.len() { q[x] } else { 1e-10 };
                let accept_prob = (p_x / q_x.max(1e-10)).min(1.0);

                let r = rng.next_f32();
                if r < accept_prob {
                    // Accepted
                    spec_tokens.push(draft_tokens[i]);
                    num_accepted += 1;
                    total_accepted += 1;
                } else {
                    // Rejected: resample from max(0, p - q)
                    let mut adjusted = vec![0.0f32; vocab];
                    let mut adj_sum = 0.0f32;
                    for j in 0..vocab {
                        adjusted[j] = (p[j] - q[j]).max(0.0);
                        adj_sum += adjusted[j];
                    }
                    let resampled = if adj_sum > 0.0 {
                        let inv = 1.0 / adj_sum;
                        for a in &mut adjusted { *a *= inv; }
                        sample_with_random(&adjusted, rng.next_f32()) as u32
                    } else {
                        sample_with_random(&p, rng.next_f32()) as u32
                    };
                    if is_stop_token(resampled, tokenizer.eos_id) {
                        break 'outer;
                    }
                    spec_tokens.push(resampled);
                    all_accepted = false;
                    break;
                }
            }

            // Rollback to correct position:
            // KV cache from verify batch is valid at positions saved_pos..saved_pos+3
            // We keep: current_token (pos saved_pos) + num_accepted drafts
            let keep_pos = saved_pos + 1 + num_accepted as u32;
            model.rollback_to(keep_pos);

            // Get logits for next round
            if all_accepted && k == K_DRAFT {
                // All K drafts accepted → sample bonus from verify_logits[k]
                let bonus_logits = &verify_logits_flat[k * vocab..(k + 1) * vocab];
                let bonus = sample_argmax(bonus_logits) as u32;
                if !is_stop_token(bonus, tokenizer.eos_id) && spec_tokens.len() < max_tokens {
                    spec_tokens.push(bonus);
                }
                // Feed the last accepted/bonus token to get logits for next round
                let last = *spec_tokens.last().unwrap();
                logits = model.forward_and_read(last);
            } else {
                // Feed the last token (accepted draft or resampled) to get logits
                let last = *spec_tokens.last().unwrap();
                logits = model.forward_and_read(last);
            }

            total_rounds += 1;
        }

        let spec_ms = t_spec.elapsed().as_millis();
        let spec_text = tokenizer.decode(&spec_tokens);
        println!("{spec_text}");
        println!("---");
        let spec_tps = if spec_ms > 0 {
            spec_tokens.len() as f64 / (spec_ms as f64 / 1000.0)
        } else { 0.0 };
        let accept_rate = if total_drafted > 0 {
            total_accepted as f64 / total_drafted as f64 * 100.0
        } else { 0.0 };
        println!(
            "{} tokens, {:.1} tok/s ({spec_ms}ms)",
            spec_tokens.len(), spec_tps,
        );
        println!(
            "Speculation: {total_accepted}/{total_drafted} accepted ({accept_rate:.0}%), {total_rounds} rounds"
        );

        // --- Comparison ---
        println!("\n--- Comparison ---");
        let speedup = if baseline_tps > 0.0 { spec_tps / baseline_tps } else { 0.0 };
        println!("Baseline:    {:.1} tok/s ({baseline_ms}ms)", baseline_tps);
        println!("Speculative: {:.1} tok/s ({spec_ms}ms)", spec_tps);
        println!("Speedup:     {speedup:.2}×");
        let text_match = baseline_text == spec_text;
        println!("Text match:  {}", if text_match { "PASS" } else { "MISMATCH" });
        if !text_match {
            println!("  Baseline ({} tokens): {}...", baseline_tokens.len(),
                &baseline_text[..baseline_text.len().min(80)]);
            println!("  Speculative ({} tokens): {}...", spec_tokens.len(),
                &spec_text[..spec_text.len().min(80)]);
        }
    }

    #[cfg(not(feature = "gpu"))]
    println!("GPU feature not enabled. Run with: cargo run --example speculative_gpu --features gpu,gguf --release");
}
