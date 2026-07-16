//! Verify Gemma 3n Config loads correctly from real GGUF.
//! Phase 2 validation.

use alice_llm::gguf::GgufFile;
use alice_llm::llama3::{Llama3Config, Llama3Model, ModelArch};
use std::fs;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/gemma-3n-E2B-it-Q4_K_M.gguf".to_string());
    println!("Loading: {path}");
    let bytes = fs::read(&path).expect("read gguf");
    let gguf = GgufFile::parse(&bytes).expect("parse gguf");

    let arch = ModelArch::from_gguf(&gguf);
    println!("Arch: {arch:?}");

    let config = Llama3Config::from_gguf(&gguf).expect("load config");
    println!("hidden_dim: {}", config.hidden_dim);
    println!("num_heads: {}", config.num_heads);
    println!("num_kv_heads: {}", config.num_kv_heads);
    println!("num_layers: {}", config.num_layers);
    println!("head_dim: {}", config.head_dim);
    println!("intermediate_dim: {}", config.intermediate_dim);
    println!("rope_theta: {}", config.rope_theta);
    println!("sliding_window: {:?}", config.sliding_window());
    println!("shared_kv_layers: {:?}", config.shared_kv_layers());
    println!(
        "per_layer_input_embedding_dim: {:?}",
        config.per_layer_input_embedding_dim()
    );
    println!("altup_num_inputs: {:?}", config.altup_num_inputs());
    println!("altup_active_idx: {:?}", config.altup_active_idx());
    println!(
        "sliding_window_pattern: {:?}",
        config.sliding_window_pattern()
    );
    println!(
        "activation_sparsity_scale (first 12): {:?}",
        config
            .activation_sparsity_scale()
            .map(|v| &v[..v.len().min(12)])
    );
    println!("use_neox_rope: {}", config.use_neox_rope());
    println!(
        "sliding_window_for_layer(0..10): {:?}",
        (0..10)
            .map(|i| config.sliding_window_for_layer(i))
            .collect::<Vec<_>>()
    );
    println!(
        "apply_ffn_act(layer=0, x=1.0)  = {}",
        config.apply_ffn_act(0, 1.0)
    );
    println!(
        "apply_ffn_act(layer=15, x=1.0) = {}",
        config.apply_ffn_act(15, 1.0)
    );
    println!("kv_from_start_layers: {}", config.kv_from_start_layers());
    let map = config.build_kv_layer_map();
    println!("kv_layer_map: {map:?}");

    println!("\n=== Phase 5: Model load + per_layer_embedding ===");
    let model = Llama3Model::from_gguf(&gguf).expect("load model");
    println!("Model loaded successfully.");
    // Test per-layer embedding for token_id = 100 (arbitrary).
    let emb = model
        .per_layer_embedding_for_token(100)
        .expect("per-layer emb");
    println!("per_layer_embedding_for_token(100): {} elements", emb.len());
    println!("  first 8: {:?}", &emb[..8]);
    let expected = model.config.num_layers * model.config.per_layer_input_embedding_dim().unwrap();
    assert_eq!(emb.len(), expected, "size mismatch");
    println!("\nPhase 2+4+5 verify OK.");
}
