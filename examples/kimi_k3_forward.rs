//! Load a Kimi K3 GGUF and run 1 forward pass, printing top-k logits.
//!
//! Usage:
//! ```sh
//! # Single-file GGUF
//! cargo run --release --features gguf,hf-config --example kimi_k3_forward -- \
//!     <path/to/model.gguf> [token_id]
//!
//! # Split GGUF (94 shards): pass shard 1, sibling shards auto-discovered
//! cargo run --release --features gguf,hf-config --example kimi_k3_forward -- \
//!     <path/to/Kimi-K3-IQ1_S-00001-of-00094.gguf> [token_id]
//! ```
//!
//! Phase X.4.b + X.4.c.3.3.d + X.4.b.6: real K3 GGUF end-to-end forward
//! test. Emits argmax token id + top-5 (id, logit) pairs. Does NOT
//! tokenize / detokenize (would need a K3 tokenizer implementation);
//! the logits are the raw model output.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use alice_llm::gguf::{GgufFile, GgufMultiFile, GgufSource};
use alice_llm::llama3::{load_kimi_k3_model_weights, KimiK3Model, Llama3Config, ModelArch};

use memmap2::Mmap;

/// Detect if `path` matches the split GGUF naming pattern
/// `<prefix>-<NNNNN>-of-<TTTTT>.gguf`. Returns `Some((prefix, total))`
/// if it does, `None` otherwise. The caller uses this to auto-discover
/// sibling shards.
fn detect_split(path: &Path) -> Option<(String, u32)> {
    let stem = path.file_stem()?.to_str()?;
    // Expect `...-NNNNN-of-TTTTT`.
    let parts: Vec<&str> = stem.rsplitn(4, '-').collect();
    // parts = [TTTTT, "of", NNNNN, prefix]
    if parts.len() != 4 || parts[1] != "of" {
        return None;
    }
    let total: u32 = parts[0].parse().ok()?;
    let _nn: u32 = parts[2].parse().ok()?;
    let prefix = parts[3].to_string();
    Some((prefix, total))
}

/// Discover all sibling shards of a split GGUF by walking `parent_dir`.
fn discover_shards(first_shard: &Path, prefix: &str, total: u32) -> Vec<PathBuf> {
    let dir = first_shard.parent().unwrap_or(Path::new("."));
    (1..=total)
        .map(|i| dir.join(format!("{prefix}-{i:05}-of-{total:05}.gguf")))
        .collect()
}

/// Enum wrapping single-file or multi-shard GGUF, both satisfying `GgufSource`.
enum GgufKind<'a> {
    Single(GgufFile<'a>),
    Split(GgufMultiFile<'a>),
}

impl<'a> GgufSource<'a> for GgufKind<'a> {
    fn meta(&self, key: &str) -> Option<&alice_llm::gguf::MetaValue> {
        match self {
            Self::Single(g) => GgufSource::meta(g, key),
            Self::Split(g) => GgufSource::meta(g, key),
        }
    }
    fn tensor_data(&self, name: &str) -> Option<&'a [u8]> {
        match self {
            Self::Single(g) => GgufSource::tensor_data(g, name),
            Self::Split(g) => GgufSource::tensor_data(g, name),
        }
    }
    fn tensor_info(&self, name: &str) -> Option<&alice_llm::gguf::TensorInfo> {
        match self {
            Self::Single(g) => GgufSource::tensor_info(g, name),
            Self::Split(g) => GgufSource::tensor_info(g, name),
        }
    }
    fn tensor_to_f32(&self, name: &str) -> Option<Vec<f32>> {
        match self {
            Self::Single(g) => GgufSource::tensor_to_f32(g, name),
            Self::Split(g) => GgufSource::tensor_to_f32(g, name),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: {} <gguf-path> [token_id]\n\
             \n\
             Loads a Kimi K3 GGUF (single file, or shard 1 of a\n\
             split GGUF — sibling shards auto-discovered).",
            args[0]
        );
        return Err("missing argument".into());
    }
    let path = Path::new(&args[1]);
    let token_id: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);

    println!("=== Loading GGUF: {} ===", path.display());
    let t0 = Instant::now();
    // Detect split GGUF via filename pattern.
    let split_info = detect_split(path);

    // Own the mmaps for the entire program lifetime so borrows into
    // the byte slices stay valid.
    let mmaps: Vec<Mmap>;
    let gguf: GgufKind<'_>;

    if let Some((prefix, total)) = split_info {
        println!("split GGUF detected: prefix={prefix} total={total} shards");
        let shard_paths = discover_shards(path, &prefix, total);
        mmaps = shard_paths
            .iter()
            .map(|p| {
                let f = File::open(p).map_err(|e| format!("open {p:?}: {e}"))?;
                Ok::<Mmap, String>(unsafe {
                    Mmap::map(&f).map_err(|e| format!("mmap {p:?}: {e}"))?
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let total_gb: f64 = mmaps.iter().map(|m| m.len() as f64).sum::<f64>() / 1e9;
        println!(
            "mmap'd {total} shards ({:.2} GB total) in {:?}",
            total_gb,
            t0.elapsed()
        );
        let shard_bytes: Vec<&[u8]> = mmaps.iter().map(|m| m.as_ref()).collect();
        let multi =
            GgufMultiFile::parse_shards(shard_bytes).ok_or("GgufMultiFile::parse_shards failed")?;
        println!(
            "parsed {} shards ({} tensors)",
            multi.shard_count(),
            multi.tensor_count()
        );
        gguf = GgufKind::Split(multi);
    } else {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mmap = unsafe { Mmap::map(&file)? };
        mmaps = vec![mmap];
        println!(
            "mmap'd single-file {:.2} GB in {:?}",
            file_len as f64 / 1e9,
            t0.elapsed()
        );
        let single = GgufFile::parse(mmaps[0].as_ref()).ok_or("GgufFile::parse failed")?;
        gguf = GgufKind::Single(single);
    }

    let t1 = Instant::now();
    println!("=== Detecting architecture ===");
    let arch = ModelArch::from_gguf(&gguf);
    println!("arch: {arch:?} in {:?}", t1.elapsed());
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
