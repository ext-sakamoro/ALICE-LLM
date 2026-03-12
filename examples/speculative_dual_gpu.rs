//! True dual-model GPU speculative decoding: 1B draft + 8B verify.
//!
//! Draft model (Llama-3.2-1B): autoregressive K=1 × 3 tokens.
//! Verifier model (Llama-3.1-8B): batch-4 verification in one pass.
//! Both models share a single GpuEngine (same wgpu Device/Queue).
//!
//! Usage:
//!   cargo run --example speculative_dual_gpu --features gpu,gguf --release -- \
//!     --prompt "The meaning of life is" --max-tokens 64
//!
//!   # Custom models:
//!   cargo run --example speculative_dual_gpu --features gpu,gguf --release -- \
//!     --draft-model path/to/1B.gguf --verify-model path/to/8B.gguf \
//!     --prompt "Hello" --max-tokens 128

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::{sample_argmax, softmax, sample_with_random};
use std::sync::Arc;
use std::time::Instant;

#[cfg(feature = "gpu")]
use alice_llm::gpu::{GpuEngine, GpuModel, GpuModelConfig};

const K_DRAFT: usize = 3;
const EOT_ID: u32 = 128009;

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

fn is_stop(tok: u32, eos: u32) -> bool { tok == eos || tok == EOT_ID }

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let draft_path = parse_arg::<String>(&args, "--draft-model").unwrap_or_else(|| {
        "/Users/ys/models/llama-3.2-1b-gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_string()
    });
    let verify_path = parse_arg::<String>(&args, "--verify-model").unwrap_or_else(|| {
        "/Users/ys/models/llama-3.1-8b-gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf".to_string()
    });
    let raw_prompt = parse_arg::<String>(&args, "--prompt")
        .unwrap_or_else(|| "The meaning of life is".to_string());
    let max_tokens: usize = parse_arg(&args, "--max-tokens").unwrap_or(64);

    #[cfg(feature = "gpu")]
    {
        println!("=== GPU Dual-Model Speculative Decoding ===");
        println!("    Draft:    1B (K=1 × {K_DRAFT})");
        println!("    Verifier: 8B (batch-4)\n");

        // --- Load GGUF files ---
        println!("Loading draft GGUF: {draft_path}");
        let t0 = Instant::now();
        let draft_data = std::fs::read(&draft_path).expect("Failed to read draft GGUF");
        let draft_gguf = GgufFile::parse(&draft_data).expect("Failed to parse draft GGUF");
        let tokenizer = GgufTokenizer::from_gguf(&draft_gguf).expect("Failed to load tokenizer");
        println!("  parsed: {}ms", t0.elapsed().as_millis());

        println!("Loading verify GGUF: {verify_path}");
        let t1 = Instant::now();
        let verify_data = std::fs::read(&verify_path).expect("Failed to read verify GGUF");
        let verify_gguf = GgufFile::parse(&verify_data).expect("Failed to parse verify GGUF");
        println!("  parsed: {}ms", t1.elapsed().as_millis());

        // --- Shared GPU engine ---
        let t_gpu = Instant::now();
        let engine = Arc::new(GpuEngine::new());

        // --- Draft model: Llama-3.2-1B ---
        let draft_config = GpuModelConfig {
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
        let mut draft = GpuModel::load_shared(Arc::clone(&engine), &draft_gguf, draft_config);

        // --- Verifier model: Llama-3.1-8B ---
        let verify_config = GpuModelConfig {
            num_layers: 32,
            hidden_dim: 4096,
            intermediate_dim: 14336,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            rope_theta: 500000.0,
            eps: 1e-5,
            max_seq_len: 2048,
        };
        let mut verifier = GpuModel::load_shared(Arc::clone(&engine), &verify_gguf, verify_config);
        println!(
            "GPU models ready: {}ms (draft=1B + verifier=8B, shared engine)",
            t_gpu.elapsed().as_millis(),
        );

        let draft_vocab = draft.vocab_size();
        let verify_vocab = verifier.vocab_size();
        println!("  draft vocab={draft_vocab}, verify vocab={verify_vocab}");

        // --- Format prompt ---
        let formatted = format!(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{raw_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        let prompt_tokens = tokenizer.encode(&formatted);
        println!("  Prompt: {} tokens\n", prompt_tokens.len());

        // ======================================================
        // Baseline: 8B only (naive K=1)
        // ======================================================
        println!("--- Baseline (8B naive K=1) ---");
        verifier.reset();

        let t_base = Instant::now();
        for &tok in &prompt_tokens[..prompt_tokens.len() - 1] {
            verifier.forward(tok);
        }
        let mut logits = verifier.forward_and_read(*prompt_tokens.last().unwrap());

        let mut baseline_tokens: Vec<u32> = Vec::new();
        for _ in 0..max_tokens {
            let next = sample_argmax(&logits) as u32;
            if is_stop(next, tokenizer.eos_id) { break; }
            baseline_tokens.push(next);
            logits = verifier.forward_and_read(next);
        }
        let base_ms = t_base.elapsed().as_millis();
        let baseline_text = tokenizer.decode(&baseline_tokens);
        println!("{baseline_text}");
        println!("---");
        let base_tps = if base_ms > 0 {
            baseline_tokens.len() as f64 / (base_ms as f64 / 1000.0)
        } else { 0.0 };
        println!("{} tokens, {:.1} tok/s ({base_ms}ms)\n", baseline_tokens.len(), base_tps);

        // ======================================================
        // Speculative: 1B draft + 8B batch-4 verify
        // ======================================================
        println!("--- Speculative (1B draft + 8B batch-4 verify) ---");
        draft.reset();
        verifier.reset();
        let mut rng = Rng64::new(42);

        let t_spec = Instant::now();

        // Prefill both models
        for &tok in &prompt_tokens[..prompt_tokens.len() - 1] {
            draft.forward(tok);
            verifier.forward(tok);
        }
        let last_prompt = *prompt_tokens.last().unwrap();
        // Draft needs logits for first token sampling
        let mut draft_prev_logits = draft.forward_and_read(last_prompt);
        // Verifier prefills but doesn't read logits yet (will be done in batch)
        verifier.forward(last_prompt);

        let mut spec_tokens: Vec<u32> = Vec::new();
        let mut total_drafted: usize = 0;
        let mut total_accepted: usize = 0;
        let mut total_rounds: usize = 0;

        'outer: while spec_tokens.len() < max_tokens {
            // Sample current token from draft's logits
            let current_token = sample_argmax(&draft_prev_logits) as u32;
            if is_stop(current_token, tokenizer.eos_id) { break; }
            spec_tokens.push(current_token);

            let remaining = max_tokens - spec_tokens.len();
            if remaining == 0 {
                draft_prev_logits = draft.forward_and_read(current_token);
                verifier.forward(current_token);
                continue;
            }

            let k = K_DRAFT.min(remaining);

            // --- Draft phase: K=1 forward × k on 1B ---
            let draft_saved_pos = draft.position();
            let verify_saved_pos = verifier.position();
            let mut draft_tokens: Vec<u32> = Vec::with_capacity(k);
            let mut draft_logits: Vec<Vec<f32>> = Vec::with_capacity(k);
            let mut draft_input = current_token;

            for _ in 0..k {
                let dl = draft.forward_and_read(draft_input);
                draft_input = sample_argmax(&dl) as u32;
                draft_tokens.push(draft_input);
                draft_logits.push(dl);
            }
            total_drafted += draft_tokens.len();

            // --- Verify phase: batch-4 on 8B ---
            // Verifier hasn't seen current_token yet — feed it along with drafts
            let mut verify_input = [current_token, 0u32, 0, 0];
            for i in 0..k {
                verify_input[i + 1] = draft_tokens[i];
            }
            for i in (k + 1)..4 {
                verify_input[i] = verify_input[k];
            }
            let verify_logits_flat = verifier.forward_batch_and_read(&verify_input);

            // --- Accept/reject (Leviathan et al.) ---
            let mut num_accepted: usize = 0;
            let mut all_accepted = true;

            for i in 0..k {
                let p = softmax(&verify_logits_flat[i * verify_vocab..(i + 1) * verify_vocab]);
                let q = softmax(&draft_logits[i][..draft_vocab]);
                let x = draft_tokens[i] as usize;

                let p_x = if x < p.len() { p[x] } else { 0.0 };
                let q_x = if x < q.len() { q[x] } else { 1e-10 };
                let accept_prob = (p_x / q_x.max(1e-10)).min(1.0);

                let r = rng.next_f32();
                if r < accept_prob {
                    spec_tokens.push(draft_tokens[i]);
                    num_accepted += 1;
                    total_accepted += 1;
                } else {
                    let mut adjusted = vec![0.0f32; verify_vocab];
                    let mut adj_sum = 0.0f32;
                    for j in 0..verify_vocab.min(draft_vocab) {
                        adjusted[j] = (p[j] - q[j]).max(0.0);
                        adj_sum += adjusted[j];
                    }
                    // Verifier vocab may be larger — keep p for extra tokens
                    for j in draft_vocab..verify_vocab {
                        adjusted[j] = p[j];
                        adj_sum += p[j];
                    }
                    let resampled = if adj_sum > 0.0 {
                        let inv = 1.0 / adj_sum;
                        for a in &mut adjusted { *a *= inv; }
                        sample_with_random(&adjusted, rng.next_f32()) as u32
                    } else {
                        sample_with_random(&p, rng.next_f32()) as u32
                    };
                    if is_stop(resampled, tokenizer.eos_id) {
                        break 'outer;
                    }
                    spec_tokens.push(resampled);
                    all_accepted = false;
                    break;
                }
            }

            // --- Sync KV caches ---
            // Verifier: batch wrote positions verify_saved_pos..verify_saved_pos+3
            let verify_keep = verify_saved_pos + 1 + num_accepted as u32;
            verifier.rollback_to(verify_keep);

            // Draft: wrote positions draft_saved_pos..draft_saved_pos+k-1
            let draft_keep = draft_saved_pos + 1 + num_accepted as u32;
            draft.rollback_to(draft_keep);

            // Feed the last token to both models for next round
            let last = *spec_tokens.last().unwrap();
            if all_accepted && k == K_DRAFT {
                // All accepted — bonus token from verify_logits[k]
                let bonus_logits = &verify_logits_flat[k * verify_vocab..(k + 1) * verify_vocab];
                let bonus = sample_argmax(bonus_logits) as u32;
                if !is_stop(bonus, tokenizer.eos_id) && spec_tokens.len() < max_tokens {
                    spec_tokens.push(bonus);
                }
                let final_tok = *spec_tokens.last().unwrap();
                draft_prev_logits = draft.forward_and_read(final_tok);
                verifier.forward(final_tok);
            } else {
                draft_prev_logits = draft.forward_and_read(last);
                verifier.forward(last);
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
            "Speculation: {total_accepted}/{total_drafted} accepted ({accept_rate:.0}%), {total_rounds} rounds\n"
        );

        // --- Summary ---
        println!("=== Summary ===");
        let speedup = if base_tps > 0.0 { spec_tps / base_tps } else { 0.0 };
        println!("8B Baseline:    {:.1} tok/s ({base_ms}ms, {} tokens)", base_tps, baseline_tokens.len());
        println!("1B+8B Speculative: {:.1} tok/s ({spec_ms}ms, {} tokens)", spec_tps, spec_tokens.len());
        println!("Speedup:        {speedup:.2}×");
        println!("Accept rate:    {accept_rate:.0}% ({total_accepted}/{total_drafted})");
    }

    #[cfg(not(feature = "gpu"))]
    println!("GPU feature not enabled. Run with: cargo run --example speculative_dual_gpu --features gpu,gguf --release");
}
