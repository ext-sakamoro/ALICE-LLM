//! Grammar-constrained generation demo (Phase X.8 B-8).
//!
//! Loads a GGUF model + tokenizer, parses a `.gbnf` grammar file, and
//! runs [`alice_llm::llama3::Llama3Model::generate_grammar`] so the
//! sampled tokens are guaranteed to satisfy the grammar. This is the
//! low-level, DSL-agnostic entry point: point `--grammar` at any GBNF
//! file (LOL, JSON schema, tool-call, custom) and the model will only
//! emit strings the grammar accepts.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example lol_gen --features "grammar gguf" -- \
//!     --model  path/to/model.gguf \
//!     --grammar path/to/lol.gbnf \
//!     --prompt "a chess knight for 3D print" \
//!     [--max-tokens 128]
//! ```
//!
//! The GGUF file is not shipped; download separately (e.g.
//! `huggingface-cli download ...`). Grammar files live in the
//! consuming project — for LOL the file is at `~/ALICE-LOL/lol.gbnf`.

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::grammar::parse_gbnf;
use alice_llm::llama3::Llama3Model;
use std::env;
use std::fs;
use std::process;
use std::time::Instant;

fn arg_after<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    arg_after(args, flag).and_then(|s| s.parse().ok())
}

fn usage_and_exit(msg: &str) -> ! {
    eprintln!("error: {msg}");
    eprintln!();
    eprintln!("Usage:");
    eprintln!(
        "  cargo run --example lol_gen --features \"grammar gguf\" -- \\\n\
         \x20   --model <path.gguf> --grammar <path.gbnf> \\\n\
         \x20   --prompt \"<text>\" [--max-tokens 128]"
    );
    process::exit(2);
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let model_path =
        arg_after(&args, "--model").unwrap_or_else(|| usage_and_exit("--model missing"));
    let grammar_path =
        arg_after(&args, "--grammar").unwrap_or_else(|| usage_and_exit("--grammar missing"));
    let prompt = arg_after(&args, "--prompt").unwrap_or_else(|| usage_and_exit("--prompt missing"));
    let max_new_tokens: usize = parse_arg(&args, "--max-tokens").unwrap_or(128);

    // --- Model + tokenizer ---
    println!("Loading GGUF: {model_path}");
    let t0 = Instant::now();
    let data = fs::read(model_path).unwrap_or_else(|e| {
        eprintln!("failed to read GGUF file {model_path}: {e}");
        process::exit(1);
    });
    let gguf = GgufFile::parse(&data).unwrap_or_else(|| {
        eprintln!("failed to parse GGUF (malformed header or truncated)");
        process::exit(1);
    });
    let tokenizer = GgufTokenizer::from_gguf(&gguf).unwrap_or_else(|| {
        eprintln!("failed to load tokenizer from GGUF");
        process::exit(1);
    });
    let mut model = Llama3Model::from_gguf(&gguf).unwrap_or_else(|| {
        eprintln!("failed to load model from GGUF");
        process::exit(1);
    });
    println!(
        "  loaded in {}ms — vocab={}, arch={:?}",
        t0.elapsed().as_millis(),
        tokenizer.vocab_size(),
        gguf.meta_str("general.architecture").unwrap_or("unknown")
    );

    // --- Grammar ---
    println!("Loading grammar: {grammar_path}");
    let gbnf_src = fs::read_to_string(grammar_path).unwrap_or_else(|e| {
        eprintln!("failed to read grammar file {grammar_path}: {e}");
        process::exit(1);
    });
    let grammar = parse_gbnf(&gbnf_src).unwrap_or_else(|e| {
        eprintln!("failed to parse grammar: {e}");
        process::exit(1);
    });
    println!("  {} rules loaded", grammar.rules.len());

    // --- Generate ---
    println!();
    println!("Prompt: {prompt:?}");
    println!("max_new_tokens: {max_new_tokens}");
    println!("Generating (greedy, temperature=1.0 top_k=1)...");
    println!();

    let result = model
        .generate_grammar(&tokenizer, prompt, max_new_tokens, &grammar, 1.0, 1)
        .unwrap_or_else(|e| {
            eprintln!("generate_grammar failed: {e}");
            process::exit(1);
        });

    println!("--- generated ---");
    println!("{}", result.text);
    println!("--- /generated ---");
    println!();
    println!(
        "prompt_tokens={} generated={} prefill={}ms decode={}ms total={}ms {:.2} tok/s",
        result.prompt_tokens,
        result.tokens_generated,
        result.prefill_ms,
        result.decode_ms,
        result.total_ms,
        result.tokens_per_sec,
    );
}
