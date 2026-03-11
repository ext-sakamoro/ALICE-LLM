use alice_llm::gguf::GgufFile;
use std::fs;

fn main() {
    let path = "/Users/ys/models/llama-3.2-1b-q3k/Llama-3.2-1B-Instruct-Q3_K_L.gguf";
    println!("Loading Q3_K_M model...");
    let data = fs::read(path).unwrap();
    let gguf = GgufFile::parse(&data).unwrap();

    // Print tensor types to verify Q3_K is present
    let tensors = ["blk.0.attn_q.weight", "blk.0.ffn_gate.weight", "blk.0.attn_k.weight", "token_embd.weight"];
    for name in &tensors {
        if let Some(info) = gguf.tensor_info(name) {
            println!("  {name}: {:?} dims={:?}", info.qtype, info.dims);
        }
    }

    // Load full model and generate
    let mut model = alice_llm::llama3::Llama3Model::from_gguf(&gguf).unwrap();
    let tok = alice_llm::gguf::GgufTokenizer::from_gguf(&gguf).unwrap();

    println!("\nGenerating with Q3_K_M model...");
    let prompt = "<|start_header_id|>user<|end_header_id|>\n\nWhat is 1+1?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    let result = model.generate(&tok, prompt, 30, 0.0, 40);
    println!("Greedy: \"{}\"", result.text);
    println!("{} tokens, {:.1} tok/s", result.tokens_generated, result.tokens_per_sec);

    // Try with temperature
    model.clear_cache();
    let result2 = model.generate(&tok, prompt, 50, 0.7, 40);
    println!("\nTemp=0.7: \"{}\"", result2.text);

    // BOS token top-10
    model.clear_cache();
    let logits = model.forward(128000);
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nSingle BOS top-5:");
    for (idx, val) in indexed.iter().take(5) {
        println!("  token={idx} logit={val:.4}");
    }
}
