//! Grammar-constrained decode の時間内訳計測 (A1-0)
//!
//! `Llama3Model::generate_grammar` と同じ loop を手で回し、1 step ごとの
//! `forward` 時間と `mask_logits_by_grammar` 時間を分離して出力する
//! B-9-A (Qwen3.5-4B、477 s / ~8 token) の遅さが model forward なのか
//! grammar mask (vocab 線形 + alloc) なのかを数値で確定するための計測用
//!
//! ```text
//! cargo run --release --example bench_grammar_mask --features "grammar gguf" -- \
//!     --model models/MiniCPM5-2B-Q4_K_M.gguf \
//!     --grammar ../ALICE-LOL/lol.gbnf \
//!     --prompt "generate lol: sphere(1.5)" \
//!     [--max-tokens 32]
//! ```

#![allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)] // 計測用 example、ms 表示のみ

use std::env;
use std::fs;
use std::process;
use std::time::Instant;

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::grammar::{parse_gbnf, Fsm, TokenTrie};
use alice_llm::llama3::Llama3Model;
use alice_llm::sampling::{
    advance_fsm_on_emit, mask_logits_by_grammar, mask_logits_by_grammar_trie, GrammarTokenizer,
};

fn arg_after<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_path = arg_after(&args, "--model").unwrap_or_else(|| {
        eprintln!("--model missing");
        process::exit(2);
    });
    let grammar_path = arg_after(&args, "--grammar").unwrap_or_else(|| {
        eprintln!("--grammar missing");
        process::exit(2);
    });
    let prompt = arg_after(&args, "--prompt").unwrap_or("generate lol: sphere(1.5)");
    let max_new_tokens: usize = arg_after(&args, "--max-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    // `--naive` selects the per-token reference mask; default is the trie mask.
    let naive = args.iter().any(|a| a == "--naive");

    let t0 = Instant::now();
    let data = fs::read(model_path).expect("read gguf");
    let gguf = GgufFile::parse(&data).expect("parse gguf");
    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("tokenizer");
    let mut model = Llama3Model::from_gguf(&gguf).expect("model");
    println!(
        "loaded in {}ms vocab={} arch={}",
        t0.elapsed().as_millis(),
        tokenizer.vocab_size(),
        gguf.meta_str("general.architecture").unwrap_or("unknown")
    );
    let grammar = parse_gbnf(&fs::read_to_string(grammar_path).expect("read gbnf")).expect("gbnf");

    let mut tokens = tokenizer.encode(prompt);
    if tokenizer.add_bos_token && (tokens.is_empty() || tokens[0] != tokenizer.bos_id) {
        tokens.insert(0, tokenizer.bos_id);
    }
    model.clear_cache();

    let t_prefill = Instant::now();
    let mut logits = Vec::new();
    for &tok in &tokens {
        logits = model.forward(tok);
    }
    println!(
        "prefill {} tokens: {}ms",
        tokens.len(),
        t_prefill.elapsed().as_millis()
    );

    let mut fsm = Fsm::start(&grammar)
        .expect("fsm start")
        .with_max_depth(4096);
    let t_trie = Instant::now();
    let trie = TokenTrie::build(&tokenizer, tokenizer.vocab_size());
    println!(
        "trie: {} nodes, {} empty tokens, built in {}ms (mask mode: {})",
        trie.node_count(),
        trie.empty_token_count(),
        t_trie.elapsed().as_millis(),
        if naive { "naive" } else { "trie" }
    );
    let mut scratch = Vec::new();
    let mut generated = Vec::new();
    let (mut sum_fwd, mut sum_mask, mut sum_argmax) = (0u128, 0u128, 0u128);

    println!("step  mask_ms  advances  argmax_ms  forward_ms  token");
    for step in 0..max_new_tokens {
        let t = Instant::now();
        let advances = if naive {
            mask_logits_by_grammar(&fsm, &tokenizer, &mut logits);
            0
        } else {
            mask_logits_by_grammar_trie(&fsm, &trie, &tokenizer, &mut logits, &mut scratch)
        };
        let mask_ms = t.elapsed().as_millis();

        let t = Instant::now();
        let next = logits
            .iter()
            .enumerate()
            .filter(|(_, l)| l.is_finite())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32);
        let argmax_ms = t.elapsed().as_millis();
        let Some(next) = next else {
            println!("step {step}: no valid token");
            break;
        };
        if next == tokenizer.eos_id {
            println!("step {step}: eos");
            break;
        }
        advance_fsm_on_emit(&mut fsm, &tokenizer, next).expect("fsm advance");
        generated.push(next);

        let t = Instant::now();
        logits = model.forward(next);
        let fwd_ms = t.elapsed().as_millis();

        sum_mask += mask_ms;
        sum_argmax += argmax_ms;
        sum_fwd += fwd_ms;
        println!(
            "{step:>4}  {mask_ms:>7}  {advances:>8}  {argmax_ms:>9}  {fwd_ms:>10}  {:?}",
            tokenizer.text_of(next)
        );
    }
    let n = generated.len().max(1) as u128;
    println!();
    println!("generated: {:?}", tokenizer.decode(&generated));
    println!(
        "tokens={} mask total={}ms (avg {}ms) argmax total={}ms forward total={}ms (avg {}ms) => mask share {:.1}%",
        generated.len(),
        sum_mask,
        sum_mask / n,
        sum_argmax,
        sum_fwd,
        sum_fwd / n,
        100.0 * sum_mask as f64 / (sum_mask + sum_argmax + sum_fwd).max(1) as f64
    );
}
