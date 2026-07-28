//! Load a Kimi K3 GGUF and run 1 forward pass, printing top-k logits.
//!
//! Usage:
//! ```sh
//! cargo run --release --features hf-config --example kimi_k3_forward -- <path/to/merged.gguf> [token_id]
//! ```
//!
//! For split GGUF (94 shards), merge first via
//! `llama-gguf-split --merge <path/to/Kimi-K3-IQ1_S-00001-of-00094.gguf>
//! <path/to/merged.gguf>`.
//!
//! Phase X.4.b + X.4.c.3.3.d: real K3 GGUF end-to-end forward test.
//! Emits argmax token id + top-5 (id, logit) pairs. Does NOT tokenize
//! or detokenize (would need a K3 tokenizer implementation); the
//! logits are the raw model output.

use std::fs::File;
use std::path::Path;
use std::time::Instant;

use alice_llm::gguf::GgufFile;
use alice_llm::llama3::{load_kimi_k3_model_weights, KimiK3Model, Llama3Config, ModelArch};

use memmap2::Mmap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: {} <gguf-path> [token_id]\n\
             \n\
             Loads a Kimi K3 GGUF (must be single-file, use\n\
             `llama-gguf-split --merge` on split files first) and runs\n\
             one forward pass. Prints top-5 logits + argmax.",
            args[0]
        );
        return Err("missing argument".into());
    }
    let path = Path::new(&args[1]);
    let token_id: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    println!("=== Loading GGUF: {} ===", path.display());
    let t0 = Instant::now();
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mmap = unsafe { Mmap::map(&file)? };
    println!(
        "mmap'd {:.2} GB in {:?}",
        file_len as f64 / 1e9,
        t0.elapsed()
    );

    let t1 = Instant::now();
    let gguf = GgufFile::parse(&mmap).ok_or("GGUF parse failed")?;
    println!("parsed GGUF in {:?}", t1.elapsed());

    println!("=== Detecting architecture ===");
    let arch = ModelArch::from_gguf(&gguf);
    println!("arch: {arch:?}");
    if arch != ModelArch::KimiK3 {
        return Err(format!("expected KimiK3 arch, got {arch:?}").into());
    }

    println!("=== Building Llama3Config ===");
    let t2 = Instant::now();
    let config = Llama3Config::from_gguf(&gguf).ok_or("Llama3Config::from_gguf returned None")?;
    println!(
        "hidden={} layers={} heads={} vocab={} loaded in {:?}",
        config.hidden_dim,
        config.num_layers,
        config.num_heads,
        config.vocab_size,
        t2.elapsed()
    );
    if let Some(kd) = &config.kimi_delta {
        println!(
            "K3 subconfig: kda_head_dim={:?} kv_lora={:?} q_lora={:?} \
             qk_nope={:?} qk_rope={:?} n_experts={:?} top_k={:?} \
             attn_res_block={:?} moe_intermediate={:?} latent_hidden={:?}",
            kd.kda_head_dim,
            kd.kv_lora_rank,
            kd.q_lora_rank,
            kd.qk_nope_head_dim,
            kd.qk_rope_head_dim,
            kd.n_routed_experts,
            kd.num_experts_per_tok,
            kd.attn_res_block_size,
            kd.moe_intermediate_size,
            kd.routed_expert_hidden_size,
        );
    }

    println!("=== Loading weights ===");
    let t3 = Instant::now();
    let weights = load_kimi_k3_model_weights(&gguf, &config)
        .map_err(|e| format!("load_kimi_k3_model_weights failed: {e}"))?;
    println!("weights loaded in {:?}", t3.elapsed());

    println!("=== Constructing KimiK3Model ===");
    let t4 = Instant::now();
    let mut model =
        KimiK3Model::new(weights, config).map_err(|e| format!("KimiK3Model::new failed: {e}"))?;
    println!("model constructed in {:?}", t4.elapsed());
    println!("num_layers: {}", model.num_layers());

    println!("=== Forward pass (token_id = {token_id}) ===");
    let t5 = Instant::now();
    let logits = model.forward(token_id);
    println!("forward in {:?}", t5.elapsed());
    println!("logits.len() = {}", logits.len());

    // Argmax + top-5.
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("\nTop-5 (token_id, logit):");
    for (i, (idx, logit)) in indexed.iter().take(5).enumerate() {
        println!("  #{i}: token_id={idx:>6} logit={logit:>10.4}");
    }
    println!("\nargmax token_id = {}", indexed[0].0);

    let finite = logits.iter().filter(|v| v.is_finite()).count();
    println!(
        "\nsanity: {finite}/{total} logits finite, nan/inf count = {}",
        logits.len() - finite,
        total = logits.len()
    );

    Ok(())
}
