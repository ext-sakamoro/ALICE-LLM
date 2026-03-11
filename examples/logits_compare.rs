//! Compare forward pass intermediate values against llama.cpp reference.
//! Traces BOS token through layer 0 step-by-step to locate divergence.

use alice_llm::gguf::{quantized_matvec, GgufFile, GgufTokenizer};
use alice_llm::llama3::{Llama3Config, Llama3Model};
use std::env;
use std::fs;

// Reference logits[0:10] from llama.cpp after BOS token
const REF_LOGITS: [f32; 10] = [7.79, 7.03, 6.98, 6.67, 5.41, 6.44, 4.10, 8.04, 8.78, 5.11];

fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    let mut ss = 0.0f64;
    for &v in x {
        ss += (v as f64) * (v as f64);
    }
    let mean = (ss / n as f64) as f32;
    let scale = 1.0f32 / (mean + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
}

fn apply_rope(vec: &mut [f32], position: usize, head_dim: usize, theta: f32) {
    for i in (0..head_dim).step_by(2) {
        let freq = 1.0 / theta.powf(i as f32 / head_dim as f32);
        let angle = position as f32 * freq;
        let (sin_val, cos_val) = angle.sin_cos();
        let x0 = vec[i];
        let x1 = vec[i + 1];
        vec[i] = x0 * cos_val - x1 * sin_val;
        vec[i + 1] = x0 * sin_val + x1 * cos_val;
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn stats(v: &[f32]) -> (f32, f32, f32, f32) {
    let sum: f32 = v.iter().sum();
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let l2 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    (sum, min, max, l2)
}

fn print_stats(label: &str, v: &[f32]) {
    let (sum, min, max, l2) = stats(v);
    print!("  {label}: sum={sum:.2}, min={min:.6}, max={max:.6}, L2={l2:.2}");
    if v.len() >= 10 {
        print!(", [0:10]={:.4?}", &v[0..10].iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>());
    }
    println!();
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_path = args.get(1).expect("Usage: logits_compare <model.gguf>");
    let data = fs::read(model_path).expect("Failed to read");
    let gguf = GgufFile::parse(&data).expect("Failed to parse");
    let config = Llama3Config::from_gguf(&gguf).expect("config");
    let tokenizer = GgufTokenizer::from_gguf(&gguf).expect("tokenizer");

    println!("BOS token ID: {}", tokenizer.bos_id);
    println!("Config: hidden={}, heads={}, kv_heads={}, layers={}, vocab={}",
        config.hidden_dim, config.num_heads, config.num_kv_heads, config.num_layers, config.vocab_size);

    // ─── Step 1: Embedding ─────────────────────────────────────────────
    println!("\n=== Step 1: BOS Embedding ===");
    let emb_info = gguf.tensor_info("token_embd.weight").expect("embd info");
    println!("  token_embd.weight: dims={:?}, qtype={:?}", emb_info.dims, emb_info.qtype);

    let embedding = gguf.tensor_to_f32("token_embd.weight").expect("embedding");
    let bos = tokenizer.bos_id as usize;
    let emb_start = bos * config.hidden_dim;
    let hidden_init: Vec<f32> = embedding[emb_start..emb_start + config.hidden_dim].to_vec();
    print_stats("BOS embedding", &hidden_init);

    // ─── Step 2: Layer 0 attn_norm ─────────────────────────────────────
    println!("\n=== Step 2: Layer 0 Attention Norm ===");
    let attn_norm_w = gguf.tensor_to_f32("blk.0.attn_norm.weight").expect("attn_norm");
    print_stats("attn_norm weight", &attn_norm_w);

    let mut norm_buf = vec![0.0f32; config.hidden_dim];
    rms_norm(&hidden_init, &attn_norm_w, config.norm_eps, &mut norm_buf);
    print_stats("after attn_norm", &norm_buf);

    // ─── Step 3: Q, K, V projections ──────────────────────────────────
    println!("\n=== Step 3: Q, K, V Projections (Layer 0) ===");
    let kv_dim = config.num_kv_heads * config.head_dim;

    let q_info = gguf.tensor_info("blk.0.attn_q.weight").expect("q info");
    let k_info = gguf.tensor_info("blk.0.attn_k.weight").expect("k info");
    let v_info = gguf.tensor_info("blk.0.attn_v.weight").expect("v info");
    println!("  Q: dims={:?}, qtype={:?}", q_info.dims, q_info.qtype);
    println!("  K: dims={:?}, qtype={:?}", k_info.dims, k_info.qtype);
    println!("  V: dims={:?}, qtype={:?}", v_info.dims, v_info.qtype);

    let q_data = gguf.tensor_data("blk.0.attn_q.weight").expect("q data");
    let k_data = gguf.tensor_data("blk.0.attn_k.weight").expect("k data");
    let v_data = gguf.tensor_data("blk.0.attn_v.weight").expect("v data");

    let mut q_buf = vec![0.0f32; config.hidden_dim];
    let mut k_buf = vec![0.0f32; kv_dim];
    let mut v_buf = vec![0.0f32; kv_dim];

    quantized_matvec(&norm_buf, q_data, q_info.qtype, config.hidden_dim, config.hidden_dim, &mut q_buf);
    quantized_matvec(&norm_buf, k_data, k_info.qtype, kv_dim, config.hidden_dim, &mut k_buf);
    quantized_matvec(&norm_buf, v_data, v_info.qtype, kv_dim, config.hidden_dim, &mut v_buf);

    print_stats("Q (raw)", &q_buf);
    print_stats("K (raw)", &k_buf);
    print_stats("V (raw)", &v_buf);

    // ─── Step 4: RoPE (position 0 = identity) ────────────────────────
    println!("\n=== Step 4: RoPE at position 0 ===");
    for h in 0..config.num_heads {
        let start = h * config.head_dim;
        apply_rope(&mut q_buf[start..start + config.head_dim], 0, config.head_dim, config.rope_theta);
    }
    for h in 0..config.num_kv_heads {
        let start = h * config.head_dim;
        apply_rope(&mut k_buf[start..start + config.head_dim], 0, config.head_dim, config.rope_theta);
    }
    print_stats("Q (after RoPE pos=0)", &q_buf);
    print_stats("K (after RoPE pos=0)", &k_buf);

    // ─── Step 5: Self-attention (single token = V passthrough) ────────
    println!("\n=== Step 5: Attention (seq_len=1, softmax=[1.0]) ===");
    let heads_per_kv = config.num_heads / config.num_kv_heads;
    let mut attn_out = vec![0.0f32; config.hidden_dim];
    for h in 0..config.num_heads {
        let kv_h = h / heads_per_kv;
        let q_start = h * config.head_dim;
        let k_start = kv_h * config.head_dim;

        // Score = Q·K / sqrt(head_dim)
        let mut score = 0.0f32;
        for d in 0..config.head_dim {
            score += q_buf[q_start + d] * k_buf[k_start + d];
        }
        score /= (config.head_dim as f32).sqrt();

        // With seq_len=1, softmax([score]) = [1.0], so attn_out = V
        for d in 0..config.head_dim {
            attn_out[q_start + d] = v_buf[kv_h * config.head_dim + d];
        }

        if h < 2 {
            println!("  head {h}: score={score:.6}, kv_h={kv_h}");
        }
    }
    print_stats("attn_out (= V broadcast)", &attn_out);

    // ─── Step 6: Output projection + residual ─────────────────────────
    println!("\n=== Step 6: Output Projection + Residual ===");
    let o_info = gguf.tensor_info("blk.0.attn_output.weight").expect("o info");
    let o_data = gguf.tensor_data("blk.0.attn_output.weight").expect("o data");
    println!("  O: dims={:?}, qtype={:?}", o_info.dims, o_info.qtype);

    let mut o_buf = vec![0.0f32; config.hidden_dim];
    quantized_matvec(&attn_out, o_data, o_info.qtype, config.hidden_dim, config.hidden_dim, &mut o_buf);
    print_stats("o_proj output", &o_buf);

    let mut hidden = hidden_init.clone();
    for i in 0..config.hidden_dim {
        hidden[i] += o_buf[i];
    }
    print_stats("hidden (after attn residual)", &hidden);

    // ─── Step 7: FFN ──────────────────────────────────────────────────
    println!("\n=== Step 7: FFN (Layer 0) ===");
    let ffn_norm_w = gguf.tensor_to_f32("blk.0.ffn_norm.weight").expect("ffn_norm");
    rms_norm(&hidden, &ffn_norm_w, config.norm_eps, &mut norm_buf);
    print_stats("after ffn_norm", &norm_buf);

    let gate_info = gguf.tensor_info("blk.0.ffn_gate.weight").expect("gate info");
    let up_info = gguf.tensor_info("blk.0.ffn_up.weight").expect("up info");
    let down_info = gguf.tensor_info("blk.0.ffn_down.weight").expect("down info");
    println!("  gate: dims={:?}, qtype={:?}", gate_info.dims, gate_info.qtype);
    println!("  up:   dims={:?}, qtype={:?}", up_info.dims, up_info.qtype);
    println!("  down: dims={:?}, qtype={:?}", down_info.dims, down_info.qtype);

    let gate_data = gguf.tensor_data("blk.0.ffn_gate.weight").expect("gate data");
    let up_data = gguf.tensor_data("blk.0.ffn_up.weight").expect("up data");
    let down_data = gguf.tensor_data("blk.0.ffn_down.weight").expect("down data");

    let mut gate_buf = vec![0.0f32; config.intermediate_dim];
    let mut up_buf = vec![0.0f32; config.intermediate_dim];
    quantized_matvec(&norm_buf, gate_data, gate_info.qtype, config.intermediate_dim, config.hidden_dim, &mut gate_buf);
    quantized_matvec(&norm_buf, up_data, up_info.qtype, config.intermediate_dim, config.hidden_dim, &mut up_buf);

    print_stats("gate_proj", &gate_buf);
    print_stats("up_proj", &up_buf);

    for i in 0..config.intermediate_dim {
        gate_buf[i] = silu(gate_buf[i]) * up_buf[i];
    }
    print_stats("silu(gate)*up", &gate_buf);

    let mut down_buf = vec![0.0f32; config.hidden_dim];
    quantized_matvec(&gate_buf, down_data, down_info.qtype, config.hidden_dim, config.intermediate_dim, &mut down_buf);
    print_stats("down_proj", &down_buf);

    for i in 0..config.hidden_dim {
        hidden[i] += down_buf[i];
    }
    print_stats("hidden (after layer 0)", &hidden);

    // ─── Step 8: Skip to output (full model) ──────────────────────────
    println!("\n=== Step 8: Full Model forward(BOS) ===");
    let mut model = Llama3Model::from_gguf(&gguf).expect("model");
    let logits = model.forward(tokenizer.bos_id);
    let (sum, min, max, l2) = stats(&logits);
    println!("  logits: sum={sum:.2}, min={min:.4}, max={max:.4}, L2={l2:.2}");
    println!("  logits[0:10] (ours):  {:?}", &logits[0..10].iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>());
    println!("  logits[0:10] (ref):   {:?}", REF_LOGITS.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>());

    // Diff
    println!("\n  Element-wise diff (ours - ref):");
    for i in 0..10 {
        let diff = logits[i] - REF_LOGITS[i];
        println!("    [{i}] ours={:.4}, ref={:.4}, diff={diff:+.4}", logits[i], REF_LOGITS[i]);
    }

    // Top-10
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\n  Top-10 (ours):");
    for (idx, logit) in indexed.iter().take(10) {
        let decoded = tokenizer.decode(&[*idx as u32]);
        println!("    [{idx}] {logit:.4} '{decoded}'");
    }
    println!("\n  Top-10 (ref): [15024] 14.5643 'し', [5486] 14.0293 '、', [1811] 13.9260 '。'");

    // ─── Step 9: 1-layer logits (skip layers 1-31) ────────────────────
    println!("\n=== Step 9: 1-layer logits (layer 0 → output_norm → output_proj) ===");
    let output_norm_w = gguf.tensor_to_f32("output_norm.weight").expect("output_norm");
    let out_info = gguf.tensor_info("output.weight").expect("output info");
    let out_data = gguf.tensor_data("output.weight").expect("output data");
    println!("  output.weight: dims={:?}, qtype={:?}", out_info.dims, out_info.qtype);

    let mut norm_out = vec![0.0f32; config.hidden_dim];
    rms_norm(&hidden, &output_norm_w, config.norm_eps, &mut norm_out);
    print_stats("output_norm(layer0_hidden)", &norm_out);

    let mut logits_1layer = vec![0.0f32; config.vocab_size];
    quantized_matvec(&norm_out, out_data, out_info.qtype, config.vocab_size, config.hidden_dim, &mut logits_1layer);
    let (sum1, min1, max1, l2_1) = stats(&logits_1layer);
    println!("  1-layer logits: sum={sum1:.2}, min={min1:.4}, max={max1:.4}, L2={l2_1:.2}");
    println!("  1-layer logits[0:10]: {:?}", &logits_1layer[0..10].iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>());

    let mut idx1: Vec<(usize, f32)> = logits_1layer.iter().copied().enumerate().collect();
    idx1.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("  1-layer Top-5:");
    for (idx, logit) in idx1.iter().take(5) {
        let decoded = tokenizer.decode(&[*idx as u32]);
        println!("    [{idx}] {logit:.4} '{decoded}'");
    }

    // ─── Step 10: Manual 2-layer forward ──────────────────────────────
    println!("\n=== Step 10: Layer 1 (manual) ===");
    // Re-use hidden from layer 0 output
    let attn_norm1_w = gguf.tensor_to_f32("blk.1.attn_norm.weight").expect("blk1 attn_norm");
    rms_norm(&hidden, &attn_norm1_w, config.norm_eps, &mut norm_buf);
    print_stats("layer1 after attn_norm", &norm_buf);

    let q1_info = gguf.tensor_info("blk.1.attn_q.weight").expect("q1");
    let k1_info = gguf.tensor_info("blk.1.attn_k.weight").expect("k1");
    let v1_info = gguf.tensor_info("blk.1.attn_v.weight").expect("v1");
    let o1_info = gguf.tensor_info("blk.1.attn_output.weight").expect("o1");
    println!("  Layer 1 Q: qtype={:?}, K: qtype={:?}, V: qtype={:?}", q1_info.qtype, k1_info.qtype, v1_info.qtype);

    let q1_data = gguf.tensor_data("blk.1.attn_q.weight").expect("q1d");
    let k1_data = gguf.tensor_data("blk.1.attn_k.weight").expect("k1d");
    let v1_data = gguf.tensor_data("blk.1.attn_v.weight").expect("v1d");
    let o1_data = gguf.tensor_data("blk.1.attn_output.weight").expect("o1d");

    quantized_matvec(&norm_buf, q1_data, q1_info.qtype, config.hidden_dim, config.hidden_dim, &mut q_buf);
    quantized_matvec(&norm_buf, k1_data, k1_info.qtype, kv_dim, config.hidden_dim, &mut k_buf);
    quantized_matvec(&norm_buf, v1_data, v1_info.qtype, kv_dim, config.hidden_dim, &mut v_buf);
    print_stats("L1 Q", &q_buf);
    print_stats("L1 K", &k_buf);
    print_stats("L1 V", &v_buf);

    // RoPE at position 0 (still position 0 since we're re-doing from scratch)
    for h in 0..config.num_heads {
        let start = h * config.head_dim;
        apply_rope(&mut q_buf[start..start + config.head_dim], 0, config.head_dim, config.rope_theta);
    }
    for h in 0..config.num_kv_heads {
        let start = h * config.head_dim;
        apply_rope(&mut k_buf[start..start + config.head_dim], 0, config.head_dim, config.rope_theta);
    }

    // Self-attention (single token, seq_len=1 for this layer's cache too)
    attn_out.fill(0.0);
    for h in 0..config.num_heads {
        let kv_h = h / heads_per_kv;
        let q_start = h * config.head_dim;
        for d in 0..config.head_dim {
            attn_out[q_start + d] = v_buf[kv_h * config.head_dim + d];
        }
    }

    // O projection + residual
    quantized_matvec(&attn_out, o1_data, o1_info.qtype, config.hidden_dim, config.hidden_dim, &mut o_buf);
    for i in 0..config.hidden_dim { hidden[i] += o_buf[i]; }

    // FFN layer 1
    let ffn_norm1_w = gguf.tensor_to_f32("blk.1.ffn_norm.weight").expect("ffn1_norm");
    rms_norm(&hidden, &ffn_norm1_w, config.norm_eps, &mut norm_buf);

    let g1_info = gguf.tensor_info("blk.1.ffn_gate.weight").expect("g1");
    let u1_info = gguf.tensor_info("blk.1.ffn_up.weight").expect("u1");
    let d1_info = gguf.tensor_info("blk.1.ffn_down.weight").expect("d1");
    let g1_data = gguf.tensor_data("blk.1.ffn_gate.weight").expect("g1d");
    let u1_data = gguf.tensor_data("blk.1.ffn_up.weight").expect("u1d");
    let d1_data = gguf.tensor_data("blk.1.ffn_down.weight").expect("d1d");

    quantized_matvec(&norm_buf, g1_data, g1_info.qtype, config.intermediate_dim, config.hidden_dim, &mut gate_buf);
    quantized_matvec(&norm_buf, u1_data, u1_info.qtype, config.intermediate_dim, config.hidden_dim, &mut up_buf);
    for i in 0..config.intermediate_dim { gate_buf[i] = silu(gate_buf[i]) * up_buf[i]; }
    quantized_matvec(&gate_buf, d1_data, d1_info.qtype, config.hidden_dim, config.intermediate_dim, &mut down_buf);
    for i in 0..config.hidden_dim { hidden[i] += down_buf[i]; }

    print_stats("hidden (after layer 1)", &hidden);

    // 2-layer logits
    rms_norm(&hidden, &output_norm_w, config.norm_eps, &mut norm_out);
    quantized_matvec(&norm_out, out_data, out_info.qtype, config.vocab_size, config.hidden_dim, &mut logits_1layer);
    let (sum2, _, max2, _) = stats(&logits_1layer);
    println!("  2-layer logits: sum={sum2:.2}, max={max2:.4}");
    println!("  2-layer logits[0:10]: {:?}", &logits_1layer[0..10].iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>());

    // ─── Step 11: Manual N-layer forward to find divergence point ─────
    println!("\n=== Step 11: Layer-by-layer hidden state L2 growth ===");
    // Restart from embedding
    let mut h = hidden_init.clone();
    let mut nb = vec![0.0f32; config.hidden_dim];
    let mut qb = vec![0.0f32; config.hidden_dim];
    let mut kb = vec![0.0f32; kv_dim];
    let mut vb = vec![0.0f32; kv_dim];
    let mut ab = vec![0.0f32; config.hidden_dim];
    let mut ob = vec![0.0f32; config.hidden_dim];
    let mut gb = vec![0.0f32; config.intermediate_dim];
    let mut ub = vec![0.0f32; config.intermediate_dim];
    let mut db = vec![0.0f32; config.hidden_dim];

    for layer_idx in 0..config.num_layers {
        let prefix = format!("blk.{layer_idx}");
        let an_w = gguf.tensor_to_f32(&format!("{prefix}.attn_norm.weight")).unwrap();
        rms_norm(&h, &an_w, config.norm_eps, &mut nb);

        let qi = gguf.tensor_info(&format!("{prefix}.attn_q.weight")).unwrap();
        let ki = gguf.tensor_info(&format!("{prefix}.attn_k.weight")).unwrap();
        let vi = gguf.tensor_info(&format!("{prefix}.attn_v.weight")).unwrap();
        let oi = gguf.tensor_info(&format!("{prefix}.attn_output.weight")).unwrap();
        let qd = gguf.tensor_data(&format!("{prefix}.attn_q.weight")).unwrap();
        let kd = gguf.tensor_data(&format!("{prefix}.attn_k.weight")).unwrap();
        let vd = gguf.tensor_data(&format!("{prefix}.attn_v.weight")).unwrap();
        let od = gguf.tensor_data(&format!("{prefix}.attn_output.weight")).unwrap();

        quantized_matvec(&nb, qd, qi.qtype, config.hidden_dim, config.hidden_dim, &mut qb);
        quantized_matvec(&nb, kd, ki.qtype, kv_dim, config.hidden_dim, &mut kb);
        quantized_matvec(&nb, vd, vi.qtype, kv_dim, config.hidden_dim, &mut vb);

        // RoPE at position 0
        for hh in 0..config.num_heads {
            let s = hh * config.head_dim;
            apply_rope(&mut qb[s..s+config.head_dim], 0, config.head_dim, config.rope_theta);
        }
        for hh in 0..config.num_kv_heads {
            let s = hh * config.head_dim;
            apply_rope(&mut kb[s..s+config.head_dim], 0, config.head_dim, config.rope_theta);
        }

        // Attention (seq_len=1 → V passthrough)
        ab.fill(0.0);
        for hh in 0..config.num_heads {
            let kv_h = hh / heads_per_kv;
            let qs = hh * config.head_dim;
            for d in 0..config.head_dim {
                ab[qs + d] = vb[kv_h * config.head_dim + d];
            }
        }

        // O projection + residual
        quantized_matvec(&ab, od, oi.qtype, config.hidden_dim, config.hidden_dim, &mut ob);
        for i in 0..config.hidden_dim { h[i] += ob[i]; }

        // FFN
        let fn_w = gguf.tensor_to_f32(&format!("{prefix}.ffn_norm.weight")).unwrap();
        rms_norm(&h, &fn_w, config.norm_eps, &mut nb);

        let gi = gguf.tensor_info(&format!("{prefix}.ffn_gate.weight")).unwrap();
        let ui = gguf.tensor_info(&format!("{prefix}.ffn_up.weight")).unwrap();
        let di = gguf.tensor_info(&format!("{prefix}.ffn_down.weight")).unwrap();
        let gd = gguf.tensor_data(&format!("{prefix}.ffn_gate.weight")).unwrap();
        let ud = gguf.tensor_data(&format!("{prefix}.ffn_up.weight")).unwrap();
        let dd = gguf.tensor_data(&format!("{prefix}.ffn_down.weight")).unwrap();

        quantized_matvec(&nb, gd, gi.qtype, config.intermediate_dim, config.hidden_dim, &mut gb);
        quantized_matvec(&nb, ud, ui.qtype, config.intermediate_dim, config.hidden_dim, &mut ub);
        for i in 0..config.intermediate_dim { gb[i] = silu(gb[i]) * ub[i]; }
        quantized_matvec(&gb, dd, di.qtype, config.hidden_dim, config.intermediate_dim, &mut db);
        for i in 0..config.hidden_dim { h[i] += db[i]; }

        let (s, mn, mx, l) = stats(&h);
        println!("  layer {layer_idx:2}: sum={s:10.2}, L2={l:8.2}, min={mn:10.4}, max={mx:10.4}");
    }

    // Final logits from manual N-layer forward
    rms_norm(&h, &output_norm_w, config.norm_eps, &mut norm_out);
    quantized_matvec(&norm_out, out_data, out_info.qtype, config.vocab_size, config.hidden_dim, &mut logits_1layer);
    let (sf, _, mxf, _) = stats(&logits_1layer);
    println!("\n  Manual 32-layer logits: sum={sf:.2}, max={mxf:.4}");
    println!("  Manual logits[0:10]: {:?}", &logits_1layer[0..10].iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>());
    println!("  Model  logits[0:10]: {:?}", &logits[0..10].iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>());
    println!("  Ref    logits[0:10]: {:?}", REF_LOGITS.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>());

    // Check if manual == model (should be identical if logic is same)
    let mut max_diff = 0.0f32;
    for i in 0..config.vocab_size {
        let diff = (logits_1layer[i] - logits[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    println!("  Max diff (manual vs model): {max_diff:.6e}");
}
