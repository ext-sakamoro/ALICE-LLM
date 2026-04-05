//! ALICE-LLM inference server — OpenAI-compatible API.
//!
//! Serves GPU-accelerated LLM inference with auto-config from GGUF.
//! Supports both completions and chat completions endpoints.
//!
//! Usage:
//!   cargo run --bin alice-llm-server --features server --release -- \
//!     --model path/to/model.gguf --port 8000
//!
//! Endpoints:
//!   POST /v1/chat/completions — OpenAI chat format
//!   POST /v1/completions      — OpenAI text completion
//!   GET  /v1/models           — List loaded models
//!   GET  /health              — Health check

use alice_llm::gguf::{GgufFile, GgufTokenizer};
use alice_llm::gpu::{GpuEngine, GpuModel, GpuModelConfig};
use alice_llm::llama3::Llama3Config;
use alice_llm::{apply_temperature, sample_argmax, sample_with_random, softmax_inplace, top_k_filter};
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Chat template
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum ChatTemplate {
    Llama3,
    Qwen2,
    Mistral,
    Generic,
}

impl ChatTemplate {
    fn detect(gguf: &GgufFile, arch: alice_llm::llama3::ModelArch) -> Self {
        if let Some(tmpl) = gguf.meta_str("tokenizer.chat_template") {
            if tmpl.contains("im_start") {
                return Self::Qwen2;
            }
            if tmpl.contains("start_header_id") {
                return Self::Llama3;
            }
            if tmpl.contains("[INST]") {
                return Self::Mistral;
            }
        }
        match arch {
            alice_llm::llama3::ModelArch::Mistral => Self::Mistral,
            alice_llm::llama3::ModelArch::Qwen2 | alice_llm::llama3::ModelArch::Qwen3_5 => Self::Qwen2,
            _ => Self::Llama3,
        }
    }

    fn format_messages(&self, messages: &[ChatMessage]) -> String {
        match self {
            Self::Llama3 => {
                let mut out = String::from("<|begin_of_text|>");
                for m in messages {
                    out.push_str(&format!(
                        "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                        m.role, m.content
                    ));
                }
                out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                out
            }
            Self::Qwen2 => {
                let mut out = String::new();
                for m in messages {
                    out.push_str(&format!(
                        "<|im_start|>{}\n{}<|im_end|>\n",
                        m.role, m.content
                    ));
                }
                out.push_str("<|im_start|>assistant\n");
                out
            }
            Self::Mistral => {
                let mut out = String::new();
                for m in messages {
                    if m.role == "user" {
                        out.push_str(&format!("[INST] {} [/INST]", m.content));
                    } else if m.role == "assistant" {
                        out.push_str(&format!("{}</s>", m.content));
                    } else {
                        out.push_str(&format!("[INST] {} [/INST]", m.content));
                    }
                }
                out
            }
            Self::Generic => {
                let mut out = String::new();
                for m in messages {
                    out.push_str(&format!("### {}:\n{}\n\n", m.role, m.content));
                }
                out.push_str("### assistant:\n");
                out
            }
        }
    }

    fn stop_token_strs(&self) -> Vec<&'static str> {
        match self {
            Self::Llama3 => vec!["<|eot_id|>", "<|end_of_text|>"],
            Self::Qwen2 => vec!["<|im_end|>", "<|endoftext|>"],
            Self::Mistral => vec!["</s>"],
            Self::Generic => vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

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

#[derive(Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    #[serde(default)]
    _model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    temperature: f32,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default)]
    _stream: bool,
}

fn default_max_tokens() -> usize { 2048 }
fn default_top_k() -> usize { 40 }

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: String,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct CompletionChoice {
    text: String,
    index: usize,
    finish_reason: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessage,
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
    arch: String,
    vocab_size: usize,
    chat_template: String,
}

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

struct AppState {
    model: Mutex<GpuModel>,
    tokenizer: GgufTokenizer,
    model_name: String,
    chat_template: ChatTemplate,
    stop_token_ids: Vec<u32>,
    vocab_size: usize,
}

// ---------------------------------------------------------------------------
// Inference core (shared between completions and chat)
// ---------------------------------------------------------------------------

fn generate(
    state: &AppState,
    prompt_text: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
) -> Result<(String, usize, usize, u64, String), StatusCode> {
    let prompt_tokens = state.tokenizer.encode(prompt_text);

    let mut model = state.model.lock().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    model.reset();

    // Prefill
    for &tok in &prompt_tokens[..prompt_tokens.len().saturating_sub(1)] {
        model.forward(tok);
    }
    let last = *prompt_tokens.last().ok_or(StatusCode::BAD_REQUEST)?;
    let mut logits = model.forward_and_read(last);

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

    for _ in 0..max_tokens {
        let next = if temperature < 1e-6 {
            sample_argmax(&logits) as u32
        } else {
            apply_temperature(&mut logits, temperature);
            top_k_filter(&mut logits, top_k);
            softmax_inplace(&mut logits);
            sample_with_random(&logits, next_rand()) as u32
        };

        if next == state.tokenizer.eos_id || state.stop_token_ids.contains(&next) {
            finish_reason = "stop".to_string();
            break;
        }
        generated.push(next);
        logits = model.forward_and_read(next);
    }

    let decode_ms = t_decode.elapsed().as_millis() as u64;
    let text = state.tokenizer.decode(&generated);

    Ok((text, prompt_tokens.len(), generated.len(), decode_ms, finish_reason))
}

fn make_req_id() -> String {
    format!(
        "cmpl-{:016x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    )
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model: state.model_name.clone(),
        arch: format!("{:?}", state.chat_template),
        vocab_size: state.vocab_size,
        chat_template: format!("{:?}", state.chat_template),
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
    let (text, n_prompt, n_gen, decode_ms, finish_reason) =
        generate(&state, &req.prompt, req.max_tokens, req.temperature, req.top_k)?;

    let tps = if decode_ms > 0 { n_gen as f64 / (decode_ms as f64 / 1000.0) } else { 0.0 };

    Ok(Json(CompletionResponse {
        id: make_req_id(),
        object: "text_completion".to_string(),
        model: state.model_name.clone(),
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens: n_prompt,
            completion_tokens: n_gen,
            total_tokens: n_prompt + n_gen,
            tokens_per_sec: tps,
            decode_ms,
        },
    }))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, StatusCode> {
    let prompt = state.chat_template.format_messages(&req.messages);
    let (text, n_prompt, n_gen, decode_ms, finish_reason) =
        generate(&state, &prompt, req.max_tokens, req.temperature, req.top_k)?;

    let tps = if decode_ms > 0 { n_gen as f64 / (decode_ms as f64 / 1000.0) } else { 0.0 };

    Ok(Json(ChatCompletionResponse {
        id: make_req_id(),
        object: "chat.completion".to_string(),
        model: state.model_name.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: text,
            },
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens: n_prompt,
            completion_tokens: n_gen,
            total_tokens: n_prompt + n_gen,
            tokens_per_sec: tps,
            decode_ms,
        },
    }))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let model_path = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            eprintln!("Usage: alice-llm-server --model <path.gguf> [--port 8000]");
            std::process::exit(1);
        });

    let port: u16 = args
        .iter()
        .position(|a| a == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(8000);

    println!("ALICE-LLM Server v1.1");
    println!("Loading: {model_path}");

    let t0 = Instant::now();
    let data = std::fs::read(&model_path).expect("Failed to read GGUF file");
    let gguf = GgufFile::parse(&data).expect("Failed to parse GGUF");
    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");

    // Auto-detect model config from GGUF metadata
    let llm_config = Llama3Config::from_gguf(&gguf).expect("Failed to detect model config from GGUF");
    println!(
        "  arch: {:?}, layers: {}, hidden: {}, heads: {}/{}, vocab: {}",
        llm_config.arch, llm_config.num_layers, llm_config.hidden_dim,
        llm_config.num_heads, llm_config.num_kv_heads, llm_config.vocab_size,
    );

    let config = GpuModelConfig {
        num_layers: llm_config.num_layers,
        hidden_dim: llm_config.hidden_dim,
        intermediate_dim: llm_config.intermediate_dim,
        num_heads: llm_config.num_heads as u32,
        num_kv_heads: llm_config.num_kv_heads as u32,
        head_dim: llm_config.head_dim as u32,
        rope_theta: llm_config.rope_theta,
        eps: llm_config.norm_eps,
        max_seq_len: llm_config.max_seq_len,
        full_attention_interval: llm_config.full_attention_interval,
        linear_num_kv_heads: llm_config.linear_num_kv_heads.map(|v| v as u32),
        linear_qk_head_dim: llm_config.linear_qk_head_dim.map(|v| v as u32),
        linear_kv_head_dim: llm_config.linear_kv_head_dim.map(|v| v as u32),
        linear_num_v_heads: llm_config.linear_num_v_heads.map(|v| v as u32),
        linear_conv_kernel_dim: llm_config.linear_conv_kernel_dim.map(|v| v as u32),
    };

    if llm_config.is_hybrid() {
        let n_attn = (0..llm_config.num_layers).filter(|i| !llm_config.is_deltanet_layer(*i)).count();
        let n_delta = llm_config.num_layers - n_attn;
        println!(
            "  hybrid: {} DeltaNet + {} Attention (interval={})",
            n_delta, n_attn, llm_config.full_attention_interval.unwrap_or(0),
        );
    }

    // Detect chat template
    let chat_template = ChatTemplate::detect(&gguf, llm_config.arch);
    println!("  chat template: {:?}", chat_template);

    // Resolve stop token IDs
    let mut stop_token_ids = vec![tokenizer.eos_id];
    for stop_str in chat_template.stop_token_strs() {
        let ids = tokenizer.encode(stop_str);
        if ids.len() == 1 {
            let id = ids[0];
            if !stop_token_ids.contains(&id) {
                stop_token_ids.push(id);
            }
        }
    }
    println!("  stop tokens: {:?}", stop_token_ids);

    // Load GPU model
    let engine = GpuEngine::new();
    let model = GpuModel::load(engine, &gguf, config);
    let vocab_size = model.vocab_size();
    println!("Model ready: {}ms (vocab={vocab_size})", t0.elapsed().as_millis());

    let model_name = std::path::Path::new(&model_path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "alice-llm".to_string());

    let state = Arc::new(AppState {
        model: Mutex::new(model),
        tokenizer,
        model_name,
        chat_template,
        stop_token_ids,
        vocab_size,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    println!("\nListening on http://{addr}");
    println!("  POST /v1/chat/completions — Chat");
    println!("  POST /v1/completions      — Text completion");
    println!("  GET  /v1/models           — List models");
    println!("  GET  /health              — Health check\n");

    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    rt.block_on(async {
        let listener = tokio::net::TcpListener::bind(&addr).await.expect("Failed to bind");
        axum::serve(listener, app).await.expect("Server error");
    });
}
