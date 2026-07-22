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
use alice_llm::{
    apply_temperature, sample_argmax, sample_with_random, softmax_inplace, top_k_filter,
};
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

// ---------------------------------------------------------------------------
// Chat template
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum ChatTemplate {
    Llama3,
    Qwen2,
    Mistral,
    /// Gemma 2 / Gemma 3n: `<start_of_turn>role\n...<end_of_turn>\n` format.
    Gemma,
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
            if tmpl.contains("start_of_turn") {
                return Self::Gemma;
            }
        }
        match arch {
            alice_llm::llama3::ModelArch::Mistral => Self::Mistral,
            alice_llm::llama3::ModelArch::Qwen2 | alice_llm::llama3::ModelArch::Qwen3_5 => {
                Self::Qwen2
            }
            alice_llm::llama3::ModelArch::Gemma2 | alice_llm::llama3::ModelArch::Gemma3n => {
                Self::Gemma
            }
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
            Self::Gemma => {
                let mut out = String::new();
                for m in messages {
                    // Gemma uses "model" instead of "assistant".
                    let role = if m.role == "assistant" {
                        "model"
                    } else {
                        m.role.as_str()
                    };
                    out.push_str(&format!(
                        "<start_of_turn>{}\n{}<end_of_turn>\n",
                        role, m.content
                    ));
                }
                out.push_str("<start_of_turn>model\n");
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
            Self::Gemma => vec!["<end_of_turn>"],
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
    stream: bool,
}

fn default_max_tokens() -> usize {
    2048
}
fn default_top_k() -> usize {
    40
}

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

// Streaming chunk shape — mirrors OpenAI's `chat.completion.chunk` so any
// OpenAI-compatible client (Voicebox, llama.cpp examples, LangChain, etc.)
// reads the frames without a special code path.
#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChoiceDelta>,
}

#[derive(Serialize)]
struct ChoiceDelta {
    index: usize,
    delta: DeltaContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Serialize, Default)]
struct DeltaContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
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

    let mut model = state
        .model
        .lock()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
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

    Ok((
        text,
        prompt_tokens.len(),
        generated.len(),
        decode_ms,
        finish_reason,
    ))
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
    let (text, n_prompt, n_gen, decode_ms, finish_reason) = generate(
        &state,
        &req.prompt,
        req.max_tokens,
        req.temperature,
        req.top_k,
    )?;

    let tps = if decode_ms > 0 {
        n_gen as f64 / (decode_ms as f64 / 1000.0)
    } else {
        0.0
    };

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
) -> Result<Response, StatusCode> {
    let prompt = state.chat_template.format_messages(&req.messages);

    if req.stream {
        // OpenAI-compatible SSE. Bridges the sync generation loop (which
        // has to hold the GpuModel mutex) to the async response body via a
        // per-request mpsc channel: `spawn_blocking` drives the model and
        // pushes deltas, the response body drains the receiver.
        return Ok(stream_chat_completions(state, prompt, req).into_response());
    }

    let (text, n_prompt, n_gen, decode_ms, finish_reason) =
        generate(&state, &prompt, req.max_tokens, req.temperature, req.top_k)?;

    let tps = if decode_ms > 0 {
        n_gen as f64 / (decode_ms as f64 / 1000.0)
    } else {
        0.0
    };

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
    })
    .into_response())
}

fn stream_chat_completions(
    state: Arc<AppState>,
    prompt: String,
    req: ChatCompletionRequest,
) -> impl IntoResponse {
    let (tx, mut rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
    let req_id = make_req_id();
    let model_name = state.model_name.clone();

    // Emit the leading role delta so OpenAI SDKs that expect it can seed
    // the assistant message before any content arrives.
    let initial = ChatCompletionChunk {
        id: req_id.clone(),
        object: "chat.completion.chunk".to_string(),
        created: now_secs(),
        model: model_name.clone(),
        choices: vec![ChoiceDelta {
            index: 0,
            delta: DeltaContent {
                role: Some("assistant".to_string()),
                content: None,
            },
            finish_reason: None,
        }],
    };
    let _ = tx.send(Ok(sse_data(&initial)));

    // The tokenizer / GpuModel loop is synchronous, so drive it on a
    // blocking thread and drip completions back through the channel.
    let tx_gen = tx.clone();
    let req_id_gen = req_id.clone();
    let model_name_gen = model_name.clone();
    tokio::task::spawn_blocking(move || {
        run_stream_generation(state, prompt, req, req_id_gen, model_name_gen, tx_gen);
    });

    Sse::new(async_stream::stream! {
        while let Some(item) = rx.recv().await {
            yield item;
        }
    })
    .keep_alive(KeepAlive::default())
}

fn run_stream_generation(
    state: Arc<AppState>,
    prompt_text: String,
    req: ChatCompletionRequest,
    req_id: String,
    model_name: String,
    tx: mpsc::UnboundedSender<Result<Event, Infallible>>,
) {
    let prompt_tokens = state.tokenizer.encode(&prompt_text);

    let mut model = match state.model.lock() {
        Ok(m) => m,
        Err(_) => {
            // Poison — end the stream cleanly so the client isn't left hanging.
            let _ = tx.send(Ok(sse_done()));
            return;
        }
    };
    model.reset();

    for &tok in &prompt_tokens[..prompt_tokens.len().saturating_sub(1)] {
        model.forward(tok);
    }
    let Some(&last) = prompt_tokens.last() else {
        let _ = tx.send(Ok(sse_done()));
        return;
    };
    let mut logits = model.forward_and_read(last);

    let mut generated: Vec<u32> = Vec::new();
    // Running decoded text — we ship the suffix on each token so the client
    // sees incremental content instead of one big blob at the end. BPE
    // decode isn't strictly incremental byte-for-byte (multi-token merges
    // can re-render the tail), so `starts_with` guards the suffix slice.
    let mut last_decoded = String::new();
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

        if next == state.tokenizer.eos_id || state.stop_token_ids.contains(&next) {
            finish_reason = "stop".to_string();
            break;
        }

        generated.push(next);
        let full = state.tokenizer.decode(&generated);
        if full.starts_with(&last_decoded) && full.len() > last_decoded.len() {
            let delta = &full[last_decoded.len()..];
            let chunk = ChatCompletionChunk {
                id: req_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: now_secs(),
                model: model_name.clone(),
                choices: vec![ChoiceDelta {
                    index: 0,
                    delta: DeltaContent {
                        role: None,
                        content: Some(delta.to_string()),
                    },
                    finish_reason: None,
                }],
            };
            if tx.send(Ok(sse_data(&chunk))).is_err() {
                // Client disconnected — stop wasting GPU cycles.
                return;
            }
        }
        last_decoded = full;
        logits = model.forward_and_read(next);
    }

    // Terminal chunk carries the finish reason so clients that look for it
    // (finish_reason: "stop" | "length") don't have to synthesise one.
    let final_chunk = ChatCompletionChunk {
        id: req_id,
        object: "chat.completion.chunk".to_string(),
        created: now_secs(),
        model: model_name,
        choices: vec![ChoiceDelta {
            index: 0,
            delta: DeltaContent::default(),
            finish_reason: Some(finish_reason),
        }],
    };
    let _ = tx.send(Ok(sse_data(&final_chunk)));
    let _ = tx.send(Ok(sse_done()));
}

fn sse_data<T: Serialize>(chunk: &T) -> Event {
    // Fall back to a minimal error payload if serde ever chokes — we've
    // already committed to a streamed response, so bailing with 500 isn't
    // an option here.
    let payload = serde_json::to_string(chunk)
        .unwrap_or_else(|_| String::from(r#"{"error":"serialization failed"}"#));
    Event::default().data(payload)
}

fn sse_done() -> Event {
    // OpenAI sentinel — literal `data: [DONE]` (not JSON).
    Event::default().data("[DONE]")
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
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
    let llm_config =
        Llama3Config::from_gguf(&gguf).expect("Failed to detect model config from GGUF");
    println!(
        "  arch: {:?}, layers: {}, hidden: {}, heads: {}/{}, vocab: {}",
        llm_config.arch,
        llm_config.num_layers,
        llm_config.hidden_dim,
        llm_config.num_heads,
        llm_config.num_kv_heads,
        llm_config.vocab_size,
    );

    // Gemma 3n uses per-layer input embeddings, AltUp residual streams,
    // Laurel low-rank branch, shared KV cache, and activation sparsity, none
    // of which are implemented in the GPU inference path (`GpuModel`).
    // Serving Gemma 3n through the generic GPU pipeline yields incoherent
    // output. Fail-fast to prevent misleading responses; use the CPU examples
    // (`verify_gemma3n_forward`) instead until GPU support lands.
    if llm_config.arch == alice_llm::llama3::ModelArch::Gemma3n {
        eprintln!("\nError: Gemma 3n is not yet supported by the GPU inference server.");
        eprintln!("The GPU path (`GpuModel`) lacks the AltUp, Laurel, per-layer input");
        eprintln!("embedding, shared-KV, and activation-sparsity mechanisms that Gemma 3n");
        eprintln!("requires. Use the CPU example instead:");
        eprintln!("\n  cargo run --release --example verify_gemma3n_forward -- {model_path}\n");
        std::process::exit(2);
    }

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
        neox_rope: llm_config.use_neox_rope(),
        full_attention_interval: llm_config.full_attention_interval(),
        linear_num_kv_heads: llm_config.linear_num_kv_heads().map(|v| v as u32),
        linear_qk_head_dim: llm_config.linear_qk_head_dim().map(|v| v as u32),
        linear_kv_head_dim: llm_config.linear_kv_head_dim().map(|v| v as u32),
        linear_num_v_heads: llm_config.linear_num_v_heads().map(|v| v as u32),
        linear_conv_kernel_dim: llm_config.linear_conv_kernel_dim().map(|v| v as u32),
        attention_only_load: false,
    };

    if llm_config.is_hybrid() {
        let n_attn = (0..llm_config.num_layers)
            .filter(|i| !llm_config.is_deltanet_layer(*i))
            .count();
        let n_delta = llm_config.num_layers - n_attn;
        println!(
            "  hybrid: {} DeltaNet + {} Attention (interval={})",
            n_delta,
            n_attn,
            llm_config.full_attention_interval().unwrap_or(0),
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
    println!(
        "Model ready: {}ms (vocab={vocab_size})",
        t0.elapsed().as_millis()
    );

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
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .expect("Failed to bind");
        axum::serve(listener, app).await.expect("Server error");
    });
}
