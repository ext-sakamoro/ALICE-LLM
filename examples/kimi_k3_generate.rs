//! K3 multi-token generate loop with prompt encoding + decoded text output.
//!
//! Usage:
//! ```sh
//! cargo run --release --features gguf,hf-config,parallel --example kimi_k3_generate -- \
//!     <path/to/shard-00001-of-NNNNN.gguf> "<prompt text>" [max_new_tokens]
//! ```
//!
//! Prompt is BPE-encoded via the tokenizer embedded in shard 0 (K3 uses
//! GPT-2 style BPE from tiktoken.model, vocab 163840). Each generated
//! token is decoded immediately and printed to stdout with `print!` +
//! flush, so long-running generations produce visible streaming output.
//!
//! Note on Kimi K3 performance: single-token forward is ~23 min on Mac
//! mini M2 Pro + USB 2.0 external SSD (I/O bound on ~3 GB of expert
//! cube data per token). Setting `max_new_tokens = 1` for baseline
//! validation is recommended; longer runs require faster storage.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use alice_llm::gguf::{GgufFile, GgufMultiFile, GgufSource, GgufTokenizer};
use alice_llm::llama3::{load_kimi_k3_model_weights, KimiK3Model, Llama3Config, ModelArch};

use memmap2::Mmap;

/// K3 end-of-message token id (per GrEarl GGUF metadata:
/// `tokenizer.ggml.eos_token_id = 163586`, string `<|end_of_msg|>`).
const K3_EOS_TOKEN_ID: u32 = 163586;

fn detect_split(path: &Path) -> Option<(String, u32)> {
    let stem = path.file_stem()?.to_str()?;
    let parts: Vec<&str> = stem.rsplitn(4, '-').collect();
    if parts.len() != 4 || parts[1] != "of" {
        return None;
    }
    let total: u32 = parts[0].parse().ok()?;
    let _nn: u32 = parts[2].parse().ok()?;
    let prefix = parts[3].to_string();
    Some((prefix, total))
}

fn discover_shards(first: &Path, prefix: &str, total: u32) -> Vec<PathBuf> {
    let dir = first.parent().unwrap_or(Path::new("."));
    (1..=total)
        .map(|i| dir.join(format!("{prefix}-{i:05}-of-{total:05}.gguf")))
        .collect()
}

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
    if args.len() < 3 {
        eprintln!(
            "usage: {} <gguf-path> <prompt> [max_new_tokens]\n\
             \n\
             Encodes prompt via K3 tokenizer, runs prefill (one forward per\n\
             prompt token to build KV cache), then decodes max_new_tokens\n\
             greedy-argmax tokens. Stops early on EOS ({K3_EOS_TOKEN_ID}).",
            args[0]
        );
        return Err("missing args".into());
    }
    let path = Path::new(&args[1]);
    let prompt = &args[2];
    let max_new_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(4);

    // ── Load GGUF (single or split) ──
    println!("=== Loading GGUF: {} ===", path.display());
    let t0 = Instant::now();
    let split_info = detect_split(path);
    let mmaps: Vec<Mmap>;
    let gguf: GgufKind<'_>;

    if let Some((prefix, total)) = split_info {
        println!("split GGUF detected: prefix={prefix} total={total}");
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
        let multi = GgufMultiFile::parse_shards(shard_bytes).ok_or("parse_shards failed")?;
        gguf = GgufKind::Split(multi);
    } else {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        mmaps = vec![mmap];
        let single = GgufFile::parse(mmaps[0].as_ref()).ok_or("GgufFile::parse failed")?;
        gguf = GgufKind::Single(single);
    }

    // ── Build config + weights + model ──
    let arch = ModelArch::from_gguf(&gguf);
    if arch != ModelArch::KimiK3 {
        return Err(format!("expected KimiK3 arch, got {arch:?}").into());
    }
    let config = Llama3Config::from_gguf(&gguf).ok_or("config from_gguf None")?;
    println!(
        "config: hidden={} layers={} heads={} vocab={}",
        config.hidden_dim, config.num_layers, config.num_heads, config.vocab_size
    );

    let t_load = Instant::now();
    let weights =
        load_kimi_k3_model_weights(&gguf, &config).map_err(|e| format!("load weights: {e}"))?;
    println!("weights loaded in {:?}", t_load.elapsed());

    let mut model =
        KimiK3Model::new(weights, config).map_err(|e| format!("KimiK3Model::new: {e}"))?;

    // ── Load tokenizer + encode prompt ──
    let tokenizer: GgufTokenizer = match &gguf {
        GgufKind::Single(g) => GgufTokenizer::from_gguf(g),
        GgufKind::Split(_) => {
            let g0 = GgufFile::parse(mmaps[0].as_ref()).ok_or("shard 0 reparse for tokenizer")?;
            GgufTokenizer::from_gguf(&g0)
        }
    }
    .ok_or("tokenizer load returned None")?;
    let prompt_tokens = tokenizer.encode(prompt);
    println!(
        "prompt encoded: {} tokens (raw ids: {:?})",
        prompt_tokens.len(),
        &prompt_tokens[..prompt_tokens.len().min(10)]
    );

    if prompt_tokens.is_empty() {
        return Err("prompt encoded to empty token list".into());
    }

    // ── Prefill: build KV cache by running forward once per prompt token ──
    println!(
        "\n=== Prefill ({} tokens, ~{:.1} min at 23 min/token) ===",
        prompt_tokens.len(),
        prompt_tokens.len() as f64 * 23.0
    );
    let t_prefill = Instant::now();
    // last prompt logits become the sampling seed for the first generated token
    let mut last_logits: Vec<f32> = Vec::new();
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let t_tok = Instant::now();
        last_logits = model.forward(tok);
        eprintln!(
            "[prefill] token {}/{} id={tok} in {:.2} min",
            i + 1,
            prompt_tokens.len(),
            t_tok.elapsed().as_secs_f64() / 60.0
        );
    }
    println!(
        "prefill total: {:.2} min",
        t_prefill.elapsed().as_secs_f64() / 60.0
    );

    // ── Decode: greedy argmax up to max_new_tokens or EOS ──
    println!("\n=== Decode (up to {max_new_tokens} tokens or EOS={K3_EOS_TOKEN_ID}) ===");
    print!("{prompt}");
    let _ = std::io::stdout().flush();

    let t_decode = Instant::now();
    let mut generated: Vec<u32> = Vec::new();
    for step in 0..max_new_tokens {
        let next_id = argmax(&last_logits) as u32;
        let decoded = tokenizer.decode(&[next_id]);
        print!("{decoded}");
        let _ = std::io::stdout().flush();
        generated.push(next_id);
        if next_id == K3_EOS_TOKEN_ID {
            eprintln!(
                "\n[decode] EOS at step {} (id={next_id}, cumulative decode {:.2} min)",
                step + 1,
                t_decode.elapsed().as_secs_f64() / 60.0
            );
            break;
        }
        let t_step = Instant::now();
        last_logits = model.forward(next_id);
        eprintln!(
            "\n[decode] step {}/{max_new_tokens} id={next_id} decoded={decoded:?} \
             forward {:.2} min (cum {:.2} min)",
            step + 1,
            t_step.elapsed().as_secs_f64() / 60.0,
            t_decode.elapsed().as_secs_f64() / 60.0
        );
    }
    println!();
    println!(
        "\ndecode total: {:.2} min for {} tokens ({:.3} tok/min)",
        t_decode.elapsed().as_secs_f64() / 60.0,
        generated.len(),
        generated.len() as f64 / (t_decode.elapsed().as_secs_f64() / 60.0)
    );
    println!("generated ids: {generated:?}");
    println!("generated text: {:?}", tokenizer.decode(&generated));

    Ok(())
}

fn argmax(logits: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best {
            best = v;
            best_idx = i;
        }
    }
    best_idx
}
