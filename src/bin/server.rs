//! ALICE-LLM inference server — OpenAI-compatible API.
//!
//! Exposes GPU-accelerated Llama inference via HTTP.
//!
//! Usage:
//!   cargo run --bin alice-llm-server --features server --release -- \
//!     --model path/to/model.gguf --port 8090
//!
//! Endpoints:
//!   POST /v1/completions   — OpenAI-compatible text completion
//!   GET  /v1/models        — List loaded models
//!   GET  /health           — Health check

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::gpu::{GpuEngine, GpuModel, GpuModelConfig};
use alice_llm::{sample_argmax, apply_temperature, top_k_filter, softmax_inplace, sample_with_random};
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const EOT_ID: u32 = 128009;

// --- Request / Response types (OpenAI-compatible) ---

#[derive(Deserialize)]
struct CompletionRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    temperature: f32,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

fn default_max_tokens() -> usize { 256 }
fn default_top_k() -> usize { 40 }

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: String,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    text: String,
    index: usize,
    finish_reason: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
    #[serde(rename = "tokens_per_second")]
    tokens_per_sec: f64,
    decode_ms: u64,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    owned_by: String,
}

#[derive(Serialize)]
struct ModelList {
    object: String,
    data: Vec<ModelInfo>,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    model: String,
    vocab_size: usize,
}

// --- Application State ---

struct AppState {
    model: Mutex<GpuModel>,
    tokenizer: GgufTokenizer,
    model_name: String,
}

// --- Handlers ---

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let vocab = state.model.lock().unwrap().vocab_size();
    Json(HealthResponse {
        status: "ok".to_string(),
        model: state.model_name.clone(),
        vocab_size: vocab,
    })
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_name.clone(),
            object: "model".to_string(),
            owned_by: "alice-llm".to_string(),
        }],
    })
}

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, StatusCode> {
    let formatted = format!(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        req.prompt,
    );
    let prompt_tokens = state.tokenizer.encode(&formatted);
    let eos_id = state.tokenizer.eos_id;

    // Lock model for inference (single-request at a time)
    let mut model = state.model.lock().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    model.reset();

    // Prefill
    for &tok in &prompt_tokens[..prompt_tokens.len() - 1] {
        model.forward(tok);
    }
    let mut logits = model.forward_and_read(*prompt_tokens.last().unwrap());

    // Decode
    let t_decode = Instant::now();
    let mut generated: Vec<u32> = Vec::new();
    let mut finish_reason = "length".to_string();

    let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_1234;
    let mut next_rand = || -> f32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f32) / (u64::MAX as f32)
    };

    for _ in 0..req.max_tokens {
        let next = if req.temperature < 1e-6 {
            sample_argmax(&logits) as u32
        } else {
            apply_temperature(&mut logits, req.temperature);
            top_k_filter(&mut logits, req.top_k);
            softmax_inplace(&mut logits);
            sample_with_random(&logits, next_rand()) as u32
        };

        if next == eos_id || next == EOT_ID {
            finish_reason = "stop".to_string();
            break;
        }
        generated.push(next);
        logits = model.forward_and_read(next);
    }

    let decode_ms = t_decode.elapsed().as_millis() as u64;
    let text = state.tokenizer.decode(&generated);
    let n_gen = generated.len();
    let tps = if decode_ms > 0 { n_gen as f64 / (decode_ms as f64 / 1000.0) } else { 0.0 };

    // Generate a simple request ID
    let req_id = format!("cmpl-{:016x}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos());

    Ok(Json(CompletionResponse {
        id: req_id,
        object: "text_completion".to_string(),
        model: state.model_name.clone(),
        choices: vec![Choice {
            text,
            index: 0,
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: n_gen,
            total_tokens: prompt_tokens.len() + n_gen,
            tokens_per_sec: tps,
            decode_ms,
        },
    }))
}

// --- Main ---

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let model_path = args.iter().position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            "/Users/ys/models/llama-3.2-1b-gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_string()
        });

    let port: u16 = args.iter().position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(8090);

    println!("ALICE-LLM Server v1.0");
    println!("Loading model: {model_path}");

    let t0 = Instant::now();
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
    let model = GpuModel::load(engine, &gguf, config);
    println!("Model ready: {}ms (vocab={})", t0.elapsed().as_millis(), model.vocab_size());

    // Extract model name from path
    let model_name = std::path::Path::new(&model_path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "alice-llm".to_string());

    let state = Arc::new(AppState {
        model: Mutex::new(model),
        tokenizer,
        model_name,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/completions", post(completions))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    println!("\nListening on http://{addr}");
    println!("  POST /v1/completions  — Generate text");
    println!("  GET  /v1/models       — List models");
    println!("  GET  /health          — Health check\n");

    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(&addr).await.expect("Failed to bind");
        axum::serve(listener, app).await.expect("Server error");
    });
}
