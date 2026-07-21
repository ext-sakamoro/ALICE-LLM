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
use alice_llm::llama3::{Llama3Config, Llama3Model};
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

/// Emit one JSONL line of top-5 logits to stderr for CPU/GPU divergence diagnostics
/// (Issue #40). Format is intentionally identical between `qwen_gpu.rs` (GPU) and
/// `elyza_gguf.rs` (CPU) so the two streams can be diffed line-by-line.
fn dump_logits_jsonl(backend: &str, pos: i64, logits: &[f32], tokenizer: &GgufTokenizer) {
    // Argpartition-style top-5 without extra deps: track (idx, val).
    let mut top: [(u32, f32); 5] = [(u32::MAX, f32::NEG_INFINITY); 5];
    for (i, &v) in logits.iter().enumerate() {
        // Find slot to displace (smallest current). Linear over 5 is fine.
        let mut min_idx = 0usize;
        for k in 1..5 {
            if top[k].1 < top[min_idx].1 {
                min_idx = k;
            }
        }
        if v > top[min_idx].1 {
            top[min_idx] = (i as u32, v);
        }
    }
    // Sort descending by logit value.
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Build JSONL manually (no serde dep for examples).
    let mut line = String::with_capacity(256);
    line.push_str(&format!(
        "{{\"backend\":\"{backend}\",\"pos\":{pos},\"top5\":["
    ));
    for (k, (tid, logit)) in top.iter().enumerate() {
        if k > 0 {
            line.push(',');
        }
        let decoded = tokenizer.decode(&[*tid]);
        // JSON-escape decoded token (only \, ", control chars matter for our vocabs).
        let mut esc = String::with_capacity(decoded.len() + 2);
        for ch in decoded.chars() {
            match ch {
                '"' => esc.push_str("\\\""),
                '\\' => esc.push_str("\\\\"),
                '\n' => esc.push_str("\\n"),
                '\r' => esc.push_str("\\r"),
                '\t' => esc.push_str("\\t"),
                c if (c as u32) < 0x20 => esc.push_str(&format!("\\u{:04x}", c as u32)),
                c => esc.push(c),
            }
        }
        line.push_str(&format!(
            "{{\"id\":{tid},\"logit\":{logit:.6},\"tok\":\"{esc}\"}}"
        ));
    }
    line.push_str("]}");
    eprintln!("{line}");
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
        full_attention_interval: cfg.full_attention_interval(),
        linear_num_kv_heads: cfg.linear_num_kv_heads().map(|v| v as u32),
        linear_qk_head_dim: cfg.linear_qk_head_dim().map(|v| v as u32),
        linear_kv_head_dim: cfg.linear_kv_head_dim().map(|v| v as u32),
        linear_num_v_heads: cfg.linear_num_v_heads().map(|v| v as u32),
        linear_conv_kernel_dim: cfg.linear_conv_kernel_dim().map(|v| v as u32),
        // Qwen 2/2.5/3 require NEOX-style RoPE. Route from Llama3Config
        // (Issue #40 root-cause fix).
        neox_rope: cfg.arch.use_neox_rope(),
        // Full GPU forward — the per-layer hybrid orchestrator toggles
        // this to `true` for its own load path (see `run_hybrid_per_layer`).
        attention_only_load: false,
    }
}

/// Phase X.3.e.3.17 Option C MVP: hybrid mode fallback path.
///
/// Loads the model via `Llama3Model::from_gguf` (CPU-only, no GPU allocation),
/// applies Qwen ChatML template, runs autoregressive generation. Purpose is
/// to avoid the wgpu-hal Vulkan weight 2× duplication on Jetson unified
/// memory (Bonsai 27B Q1_0: 3.6 GB weight × 2 = 7.2 GB, tight in Jetson 8 GB;
/// hybrid mode uses 3.6 GB total).
///
/// True per-layer hybrid (DeltaNet on CPU + Attention on GPU) requires
/// `Llama3Model::forward_layers_range()` refactor and is Phase A2 (next
/// session) work — landing here is the memory-fix MVP that ships now.
fn run_hybrid_cpu_delegate(
    model_path: &str,
    raw_prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    max_seq_len_override: Option<usize>,
) {
    println!("Loading model: {model_path} (hybrid mode: CPU delegate)");
    let t0 = Instant::now();
    let data = std::fs::read(model_path).expect("Failed to read GGUF file");
    let gguf = GgufFile::parse(&data).expect("Failed to parse GGUF");
    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");
    println!(
        "  GGUF parsed: {}ms (vocab={})",
        t0.elapsed().as_millis(),
        tokenizer.vocab_size()
    );

    let config = Llama3Config::from_gguf(&gguf).expect("Failed to load Llama3Config from GGUF");
    if let Some(override_len) = max_seq_len_override {
        eprintln!(
            "  Note: --max-seq-len {} (hybrid CPU forward uses GGUF context_length {} directly)",
            override_len, config.max_seq_len
        );
    }
    println!(
        "  Config: layers={} hidden={} heads={}/{} head_dim={}",
        config.num_layers,
        config.hidden_dim,
        config.num_heads,
        config.num_kv_heads,
        config.head_dim
    );

    let t_load = Instant::now();
    let mut model = Llama3Model::from_gguf(&gguf).expect("Failed to load Llama3Model");
    println!("  Model loaded (CPU): {}ms", t_load.elapsed().as_millis());
    println!("  [hybrid] GPU allocation skipped; weights on CPU only (mmap'd)");

    let formatted = format!("<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n");
    let prompt_tokens = tokenizer.encode(&formatted);
    println!("  Prompt: {} tokens", prompt_tokens.len());
    println!("  Generating (max {max_tokens} tokens, temp={temperature}, top_k={top_k})");
    println!("---");

    let t_prefill = Instant::now();
    for &tok in &prompt_tokens[..prompt_tokens.len() - 1] {
        let _ = model.forward(tok);
    }
    let last_prompt = *prompt_tokens.last().unwrap();
    let mut logits = model.forward(last_prompt);
    let prefill_ms = t_prefill.elapsed().as_millis();

    let t_decode = Instant::now();
    let mut generated_tokens: Vec<u32> = Vec::new();
    let eos_id = tokenizer.eos_id;
    let im_end_id: u32 = 151645;
    let endoftext_id: u32 = 151643;

    let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_1234;
    let mut next_rand = || -> f32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f32) / (u64::MAX as f32)
    };

    for _step in 0..max_tokens {
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
        logits = model.forward(next_token);
    }
    println!();
    println!("---");

    let decode_ms = t_decode.elapsed().as_millis();
    let n_tokens = generated_tokens.len();
    println!(
        "{n_tokens} tokens generated, {:.1} tok/s ({prefill_ms}ms prefill + {decode_ms}ms decode = {}ms)",
        (n_tokens as f64 * 1000.0) / (prefill_ms + decode_ms) as f64,
        prefill_ms + decode_ms
    );
}

/// Phase A2 per-layer hybrid: CPU processes DeltaNet layers, GPU
/// processes attention layers. Uses `Llama3Model::forward_with_layer_hook`
/// on the CPU side and `GpuModel::run_attention_layer_only` on the GPU
/// side. Intended for Jetson (Cortex-A78AE 6-core + Tegra Orin iGPU)
/// where full GPU forward OOMs due to `wgpu-hal` Vulkan weight
/// duplication.
///
/// Memory footprint (Bonsai 27B Q1_0 on 8 GB unified Jetson):
///   - CPU: full model via mmap (kernel page cache), effectively free
///   - GPU: attention layer weights (~16/64 layers) + KV cache with
///     `--max-seq-len 512` = ~2 GB with `x2` duplication, fits in ~2.5 GB
///     available.
///
/// The orchestrator layer-kind routing mirrors `Llama3Config::
/// build_kv_layer_map`: every 4th layer is Attention (interval = 4) for
/// Qwen 3.5 hybrid schedules; other layers are DeltaNet.
#[cfg(feature = "gpu")]
fn run_hybrid_per_layer(
    model_path: &str,
    raw_prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    max_seq_len_override: Option<usize>,
) {
    println!("Loading model: {model_path} (hybrid-per-layer mode)");
    let t0 = Instant::now();
    let data = std::fs::read(model_path).expect("Failed to read GGUF file");
    let gguf = GgufFile::parse(&data).expect("Failed to parse GGUF");
    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");
    println!(
        "  GGUF parsed: {}ms (vocab={})",
        t0.elapsed().as_millis(),
        tokenizer.vocab_size()
    );

    let config = Llama3Config::from_gguf(&gguf).expect("Failed to load Llama3Config from GGUF");
    let gpu_cfg = gpu_config_from_llama3(&config);
    if let Some(override_len) = max_seq_len_override {
        eprintln!(
            "  Note: --max-seq-len {} — CPU forward respects GGUF context_length {} (mmap zero-copy),\n         GPU load respects the override for KV cache sizing.",
            override_len, config.max_seq_len
        );
    }

    let t_load = Instant::now();
    let mut cpu_model = Llama3Model::from_gguf(&gguf).expect("Failed to load Llama3Model");
    println!("  CPU model loaded: {}ms", t_load.elapsed().as_millis());

    let engine = GpuEngine::new();
    let mut gpu_cfg = gpu_cfg;
    if let Some(override_len) = max_seq_len_override {
        gpu_cfg.max_seq_len = override_len;
    }
    // Phase X.3.e.3.29: per-layer hybrid only ever touches Attention layers
    // on the GPU, so tell the loader to skip DeltaNet weights entirely.
    // This is what makes hybrid-per-layer fit under Jetson's ~2-3 GB usable
    // GPU budget after `wgpu-hal` Vulkan 2× duplication.
    gpu_cfg.attention_only_load = true;
    let t_gpu = Instant::now();
    let mut gpu_model = GpuModel::load(engine, &gguf, gpu_cfg);
    println!("  GPU model loaded: {}ms", t_gpu.elapsed().as_millis());

    // Build the attention-layer bitmap once so the CPU hook can decide
    // per-layer whether to defer to GPU without probing the model on
    // every layer step.
    let interval = config.full_attention_interval();
    let num_layers = config.num_layers;
    let is_attention_layer: Vec<bool> = (0..num_layers)
        .map(|i| match interval {
            Some(iv) if (i + 1) % iv != 0 => false,
            _ => true,
        })
        .collect();

    let formatted = format!(
        "<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let mut prompt_tokens = tokenizer.encode(&formatted);
    if tokenizer.add_bos_token && (prompt_tokens.is_empty() || prompt_tokens[0] != tokenizer.bos_id)
    {
        prompt_tokens.insert(0, tokenizer.bos_id);
    }
    println!("  Prompt: {} tokens", prompt_tokens.len());
    println!(
        "  Generating (max {max_tokens} tokens, temp={temperature}, top_k={top_k}) [per-layer hybrid]"
    );
    println!("---");

    let step_forward =
        |cpu_model: &mut Llama3Model, gpu_model: &mut GpuModel, token: u32| -> Vec<f32> {
            let pos = cpu_model.kv_cache_seq_len() as u32;
            let is_attention_layer = &is_attention_layer;
            cpu_model.forward_with_layer_hook(token, |layer_idx, hidden| {
                if !is_attention_layer[layer_idx] {
                    return false;
                }
                let new_hidden = gpu_model.run_attention_layer_only(layer_idx, hidden, pos);
                hidden.copy_from_slice(&new_hidden);
                true
            })
        };

    let t_prefill = Instant::now();
    for &tok in &prompt_tokens[..prompt_tokens.len() - 1] {
        let _ = step_forward(&mut cpu_model, &mut gpu_model, tok);
    }
    let last_prompt = *prompt_tokens.last().unwrap();
    let mut logits = step_forward(&mut cpu_model, &mut gpu_model, last_prompt);
    let prefill_ms = t_prefill.elapsed().as_millis();

    let t_decode = Instant::now();
    let mut generated_tokens: Vec<u32> = Vec::new();
    let eos_id = tokenizer.eos_id;
    let im_end_id: u32 = 151645;
    let endoftext_id: u32 = 151643;

    let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_1234;
    let mut next_rand = || -> f32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f32) / (u64::MAX as f32)
    };

    for _step in 0..max_tokens {
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
        logits = step_forward(&mut cpu_model, &mut gpu_model, next_token);
    }
    println!();
    println!("---");

    let decode_ms = t_decode.elapsed().as_millis();
    let n_tokens = generated_tokens.len();
    println!(
        "{n_tokens} tokens generated, {:.1} tok/s ({prefill_ms}ms prefill + {decode_ms}ms decode = {}ms) [per-layer hybrid]",
        (n_tokens as f64 * 1000.0) / (prefill_ms + decode_ms) as f64,
        prefill_ms + decode_ms
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let model_path = parse_arg::<String>(&args, "--model").expect("Usage: --model <path.gguf>");
    let raw_prompt = parse_arg::<String>(&args, "--prompt").unwrap_or_else(|| "Hello".to_string());
    let max_tokens: usize = parse_arg(&args, "--max-tokens").unwrap_or(256);
    let temperature: f32 = parse_arg(&args, "--temperature").unwrap_or(0.0);
    let top_k: usize = parse_arg(&args, "--top-k").unwrap_or(40);
    // Override the GGUF-provided `context_length` to shrink the KV cache
    // allocation. Useful on memory-constrained targets (Jetson 8 GB
    // unified) where the default 8192 causes OOM at load time.
    let max_seq_len_override: Option<usize> = parse_arg(&args, "--max-seq-len");
    // Issue #40 diagnostic: dump top-5 logits per position (JSONL to stderr) for
    // the first N decoded tokens plus the prompt-end position (pos=-1).
    let logits_dump: usize = parse_arg(&args, "--logits-dump").unwrap_or(0);
    // Issue #40 diagnostic: dump post-final-RMSNorm hidden state (pre-output-
    // projection) for pos=-1 to stderr as JSONL. Same schema as CPU side so
    // the two streams can be diffed for cos-sim / L2 to isolate whether the
    // logit gap comes from the layer stack or from output_proj.
    let dump_final_hidden = args.iter().any(|a| a == "--dump-final-hidden");
    // Issue #40 layer bisection: run prefill and dump hidden state after
    // layers 0, 6, 13, 20, 27 on the last prompt token, then exit. Requires
    // one full prefill per checkpoint (5x prefill cost) because each stop
    // point mutates KV cache differently and needs a clean reset.
    let layer_bisect = args.iter().any(|a| a == "--layer-bisect");
    let debug_nan = args.iter().any(|a| a == "--debug-nan");
    // Phase X.3.e.3.17 Option C MVP: `--hybrid` skips GPU entirely and falls
    // back to CPU-only `Llama3Model::forward()`. This delivers the Jetson OOM
    // fix (no GPU weight allocation) at the cost of speed (~0.05-0.1 tok/s on
    // Jetson Cortex-A78AE for Bonsai 27B Q1_0).
    let hybrid_mode = args.iter().any(|a| a == "--hybrid");
    // Phase A2 per-layer hybrid: CPU processes DeltaNet layers, GPU
    // processes attention layers, hidden state is shuttled between them
    // via write_f32 / read_f32. Uses `Llama3Model::forward_with_layer_hook`
    // (CPU) + `GpuModel::run_attention_layer_only` (GPU).
    let hybrid_per_layer = args.iter().any(|a| a == "--hybrid-per-layer");

    #[cfg(feature = "gpu")]
    {
        // Phase X.3.e.3.17 Option C MVP: `--hybrid` bypasses GPU entirely and
        // uses `Llama3Model::forward()` (CPU) so weights are not duplicated on
        // GPU. Delivers Jetson OOM fix (no GPU weight allocation) at speed cost.
        if hybrid_mode {
            run_hybrid_cpu_delegate(
                &model_path,
                &raw_prompt,
                max_tokens,
                temperature,
                top_k,
                max_seq_len_override,
            );
            return;
        }
        if hybrid_per_layer {
            run_hybrid_per_layer(
                &model_path,
                &raw_prompt,
                max_tokens,
                temperature,
                top_k,
                max_seq_len_override,
            );
            return;
        }

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

        let mut config = gpu_config_from_llama3(&llama_cfg);
        if let Some(override_len) = max_seq_len_override {
            eprintln!(
                "  Overriding max_seq_len: {} → {} (--max-seq-len)",
                config.max_seq_len, override_len
            );
            config.max_seq_len = override_len;
        }

        // --- GPU init ---
        let t_gpu = Instant::now();
        let engine = GpuEngine::new();
        let mut model = GpuModel::load(engine, &gguf, config);
        println!("  GPU model ready: {}ms", t_gpu.elapsed().as_millis());

        // --- Qwen 3 / 3.5 ChatML template with thinking-mode disabled ---
        // Same template as CPU (`examples/elyza_gguf.rs:163-165`) — pre-fills
        // empty `<think></think>` so greedy sampling doesn't loop the
        // thinking block. Required for apples-to-apples CPU/GPU diff.
        let formatted = format!(
            "<|im_start|>user\n{raw_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );

        // --- Tokenize ---
        // Mirror CPU `Llama3Model::generate` (llama3.rs:5974-5979): prepend
        // BOS if the tokenizer requests it and the encoded prompt doesn't
        // already start with the BOS ID. Without this, GPU processes
        // `<|im_start|>` (token 248045 for Qwen 3.5) as the first token,
        // while CPU processes BOS (token 1) first — producing entirely
        // different KV cache / DeltaNet state, which is the primary source
        // of GPU vs CPU numerical divergence at layers 6+ (Phase X.3.e.3.24
        // Step 1 diagnostic finding).
        let mut prompt_tokens = tokenizer.encode(&formatted);
        if tokenizer.add_bos_token
            && (prompt_tokens.is_empty() || prompt_tokens[0] != tokenizer.bos_id)
        {
            prompt_tokens.insert(0, tokenizer.bos_id);
        }
        println!("  Prompt: {} tokens", prompt_tokens.len());
        println!("  Generating (max {max_tokens} tokens, temp={temperature}, top_k={top_k})");
        println!("---");

        // --- Phase X.3.e.3.21 NaN source hunting ---
        if debug_nan {
            let token = *prompt_tokens.first().unwrap();
            eprintln!("[debug-nan] token = {token}");

            // Step 0: raw embedding (no RMSNorm applied)
            let emb_head = model.debug_read_embedding_head(token);
            eprintln!("[debug-nan] embedding hidden head 5 = {emb_head:?}");
            model.reset();

            // Step 1: only RMSNorm
            let norm_head = model.debug_forward_layer0_attn_norm_only(token);
            eprintln!("[debug-nan] after RMSNorm norm_buf head 5 = {norm_head:?}");
            let has_nan = norm_head.iter().any(|v| v.is_nan());
            eprintln!("[debug-nan] RMSNorm output has NaN: {has_nan}");

            // Step 2: RMSNorm + attn_qkv
            model.reset();
            let q_head = model.debug_forward_layer0_qkv_only(token);
            eprintln!("[debug-nan] after attn_qkv q_buf head 5 = {q_head:?}");
            let has_nan = q_head.iter().any(|v| v.is_nan());
            eprintln!("[debug-nan] attn_qkv output has NaN: {has_nan}");

            // Phase X.3.e.3.23: full layer 0 DeltaNet pipeline dump.
            model.reset();
            let (norm, q, k, v, alpha, beta, attn_out, attn_normed, o_buf, hidden) =
                model.debug_dump_layer0_deltanet_stages(token);
            eprintln!("[debug-dn0] norm_buf head 5      = {norm:?}");
            eprintln!("[debug-dn0] q_buf head 5 (pre-conv1d) = {q:?}");
            eprintln!("[debug-dn0] k_buf head 5 (post-conv1d, Q slice) = {k:?}");
            eprintln!("[debug-dn0] v_buf head 5 (attn_gate=z) = {v:?}");
            eprintln!("[debug-dn0] alpha_buf head 5    = {alpha:?}");
            eprintln!("[debug-dn0] beta_buf head 5     = {beta:?}");
            eprintln!("[debug-dn0] attn_out head 5     = {attn_out:?}");
            eprintln!(
                "[debug-dn0] attn_out_normed head 5 (post ssm_norm+silu(z)) = {attn_normed:?}"
            );
            eprintln!("[debug-dn0] o_buf head 5        = {o_buf:?}");
            eprintln!("[debug-dn0] hidden head 5 (post residual) = {hidden:?}");

            return;
        }

        // --- Phase X.3.e.3.36 diagnostic: dump v_buf / k_buf at layer 3 pos 0 ---
        if args.iter().any(|a| a == "--dump-l3") {
            model.reset();
            let tok0 = prompt_tokens[0];
            let kv_dim = (model.config().num_kv_heads * model.config().head_dim) as usize;
            let v = model.forward_stop_after_layer_and_read_v_buf(tok0, 3, kv_dim);
            alice_llm::llama3::dump_hidden_jsonl_stderr("gpu_attn3_v_full", &v);
            model.reset();
            let k = model.forward_stop_after_layer_and_read_k_buf(tok0, 3, kv_dim);
            alice_llm::llama3::dump_hidden_jsonl_stderr("gpu_attn3_k_full", &k);
            return;
        }

        // --- Issue #40 layer bisection mode ---
        if layer_bisect {
            let checkpoints = [0usize, 6, 13, 20, 27];
            eprintln!(
                "[layer-bisect] Running 5 prefill passes to dump hidden state at layers {checkpoints:?}"
            );
            for &stop_at in &checkpoints {
                model.reset();
                // Prefill all prompt tokens except the last (full 28-layer path)
                for &tok in &prompt_tokens[..prompt_tokens.len() - 1] {
                    model.forward(tok);
                }
                // Last prompt token: stop after `stop_at` and read hidden
                let last = *prompt_tokens.last().unwrap();
                let hidden = model.forward_stop_after_layer_and_read_hidden(last, stop_at);
                alice_llm::llama3::dump_hidden_jsonl_stderr(
                    &format!("gpu_layer_{stop_at}"),
                    &hidden,
                );
            }
            eprintln!("[layer-bisect] Done");
            return;
        }

        // --- Prefill: process all prompt tokens except the last ---
        let t_prefill = Instant::now();
        for &tok in &prompt_tokens[..prompt_tokens.len() - 1] {
            model.forward(tok);
        }
        // Last prompt token: read logits (and optionally the pre-output-
        // projection hidden state) to get first generated token + Issue #40
        // diagnostic sample.
        let last_prompt = *prompt_tokens.last().unwrap();
        let mut logits = if dump_final_hidden {
            let (hidden, logits) = model.forward_and_read_hidden_and_logits(last_prompt);
            alice_llm::llama3::dump_hidden_jsonl_stderr("gpu", &hidden);
            logits
        } else {
            model.forward_and_read(last_prompt)
        };
        let prefill_ms = t_prefill.elapsed().as_millis();

        // Issue #40 diagnostic: dump prompt-end position logits (labelled pos=-1)
        // before any decoded token is sampled. This is the layer input that
        // determines the very first generated token.
        if logits_dump > 0 {
            dump_logits_jsonl("gpu", -1, &logits, &tokenizer);
        }

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

        for step in 0..max_tokens {
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

            // Issue #40 diagnostic: dump per-step logits AFTER forwarding the
            // just-sampled token, so `pos` matches the "logits that would
            // sample decode position pos+1" convention.
            if logits_dump > 0 && step < logits_dump {
                dump_logits_jsonl("gpu", step as i64, &logits, &tokenizer);
            }
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
