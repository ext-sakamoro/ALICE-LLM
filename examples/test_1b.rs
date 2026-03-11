use alice_llm::gguf::{GgufFile, dequantize_weight_row};
use std::fs;

fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ss = 0.0f64;
    for &v in x { ss += (v as f64) * (v as f64); }
    let mean = (ss / n as f64) as f32;
    let scale = 1.0f32 / (mean + eps).sqrt();
    x.iter().zip(weight).map(|(x, w)| x * scale * w).collect()
}

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

fn f32_matvec_deq(data: &[u8], qtype: alice_llm::gguf::GgmlType, rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    let epb = qtype.elements_per_block();
    let bpb = qtype.block_bytes();
    let bpr = cols / epb;
    let rb = bpr * bpb;
    let mut out = vec![0.0f32; rows];
    let mut row_buf = vec![0.0f32; cols];
    for r in 0..rows {
        dequantize_weight_row(&data[r * rb..(r + 1) * rb], qtype, &mut row_buf);
        out[r] = row_buf.iter().zip(input).map(|(w, x)| w * x).sum();
    }
    out
}

fn main() {
    let data = fs::read("/Users/ys/models/llama-3.2-1b-gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf").unwrap();
    let gguf = GgufFile::parse(&data).unwrap();

    let h = 2048; // hidden_dim
    let kv = 512; // kv_dim
    let inter = 8192; // intermediate_dim
    let nheads = 32; let nkv = 8; let hd = 64;
    let eps = 1e-5f32;
    let num_layers = 3; // Only first 3 layers for speed

    let emb_all = gguf.tensor_to_f32("token_embd.weight").unwrap();
    let mut hidden: Vec<f32> = emb_all[128000 * h..(128000 + 1) * h].to_vec();
    let emb_l2: f32 = hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("Embedding[128000] l2={emb_l2:.6}");
    println!("Embedding[128000][0:10] = {:?}", &hidden[..10]);

    // After layer 0 attn_norm, print normed values
    {
        let an = gguf.tensor_to_f32("blk.0.attn_norm.weight").unwrap();
        let normed = rms_norm(&hidden, &an, eps);
        println!("After L0 attn_norm[0:10] = {:?}", &normed[..10]);
        let n_l2: f32 = normed.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("After L0 attn_norm l2={n_l2:.6}");
    }

    for layer in 0..num_layers {
        let prefix = format!("blk.{layer}");
        let t = std::time::Instant::now();

        // Attention norm
        let an = gguf.tensor_to_f32(&format!("{prefix}.attn_norm.weight")).unwrap();
        let normed = rms_norm(&hidden, &an, eps);

        // Q, K, V projections (pure f32 dequantized)
        let qi = gguf.tensor_info(&format!("{prefix}.attn_q.weight")).unwrap();
        let qd = gguf.tensor_data(&format!("{prefix}.attn_q.weight")).unwrap();
        let _q_buf = f32_matvec_deq(qd, qi.qtype, h, h, &normed);

        let ki = gguf.tensor_info(&format!("{prefix}.attn_k.weight")).unwrap();
        let kd = gguf.tensor_data(&format!("{prefix}.attn_k.weight")).unwrap();
        let _k_buf = f32_matvec_deq(kd, ki.qtype, kv, h, &normed);

        let vi = gguf.tensor_info(&format!("{prefix}.attn_v.weight")).unwrap();
        let vd = gguf.tensor_data(&format!("{prefix}.attn_v.weight")).unwrap();
        let v_buf = f32_matvec_deq(vd, vi.qtype, kv, h, &normed);

        // Attention at pos=layer (single position, softmax=1)
        // For simplicity, single-token: attn_out = V with GQA repeat
        let hpk = nheads / nkv;
        let mut attn_out = vec![0.0f32; h];
        for head in 0..nheads {
            let kvh = head / hpk;
            for d in 0..hd {
                attn_out[head * hd + d] = v_buf[kvh * hd + d];
            }
        }

        // O projection
        let oi = gguf.tensor_info(&format!("{prefix}.attn_output.weight")).unwrap();
        let od = gguf.tensor_data(&format!("{prefix}.attn_output.weight")).unwrap();
        let o_buf = f32_matvec_deq(od, oi.qtype, h, h, &attn_out);

        for i in 0..h { hidden[i] += o_buf[i]; }

        // FFN norm
        let fn_ = gguf.tensor_to_f32(&format!("{prefix}.ffn_norm.weight")).unwrap();
        let fn_normed = rms_norm(&hidden, &fn_, eps);

        // Gate, Up projections
        let gi = gguf.tensor_info(&format!("{prefix}.ffn_gate.weight")).unwrap();
        let gd = gguf.tensor_data(&format!("{prefix}.ffn_gate.weight")).unwrap();
        let mut gate = f32_matvec_deq(gd, gi.qtype, inter, h, &fn_normed);

        let ui = gguf.tensor_info(&format!("{prefix}.ffn_up.weight")).unwrap();
        let ud = gguf.tensor_data(&format!("{prefix}.ffn_up.weight")).unwrap();
        let up = f32_matvec_deq(ud, ui.qtype, inter, h, &fn_normed);

        // SwiGLU
        for i in 0..inter { gate[i] = silu(gate[i]) * up[i]; }

        // Down projection
        let di = gguf.tensor_info(&format!("{prefix}.ffn_down.weight")).unwrap();
        let dd = gguf.tensor_data(&format!("{prefix}.ffn_down.weight")).unwrap();
        let down = f32_matvec_deq(dd, di.qtype, h, inter, &gate);

        for i in 0..h { hidden[i] += down[i]; }

        let l2: f32 = hidden.iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("Layer {layer}: l2={l2:.4} ({:.1}s)", t.elapsed().as_secs_f64());
    }

    // Output norm + tied embedding logits
    let on = gguf.tensor_to_f32("output_norm.weight").unwrap();
    let out_normed = rms_norm(&hidden, &on, eps);

    let vocab = 128256;
    let mut logits = vec![0.0f32; vocab];
    for row in 0..vocab {
        let off = row * h;
        logits[row] = emb_all[off..off + h].iter().zip(&out_normed).map(|(w, x)| w * x).sum();
    }

    println!("\nF32 {num_layers}-layer logits[0:5]: {:?}", &logits[..5]);
    println!("The (791): {:.4}", logits[791]);

    // Compare with model's quantized output limited to same 3 layers
    let mut model = alice_llm::llama3::Llama3Model::from_gguf(&gguf).unwrap();
    let orig_layers = model.config.num_layers;
    model.config.num_layers = num_layers;
    let model_3l_logits = model.forward(128000);
    println!("\nModel {num_layers}-layer QUANTIZED logits[0:5]: {:?}", &model_3l_logits[..5]);
    println!("Model The (791): {:.4}", model_3l_logits[791]);

    // Element-wise diff for first 10
    println!("\nDiff (f32 - quantized) first 10:");
    for i in 0..10 {
        println!("  [{i}] f32={:.6} quant={:.6} diff={:.6}", logits[i], model_3l_logits[i], logits[i] - model_3l_logits[i]);
    }

    // Max absolute diff
    let max_diff: f32 = logits.iter().zip(&model_3l_logits).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let mean_diff: f32 = logits.iter().zip(&model_3l_logits).map(|(a, b)| (a - b).abs()).sum::<f32>() / logits.len() as f32;
    println!("Max abs diff: {max_diff:.6}, Mean abs diff: {mean_diff:.6}");

    // Also run full 16-layer model
    model.config.num_layers = orig_layers;
    model.clear_cache();
    let model_full_logits = model.forward(128000);
    println!("\nModel {orig_layers}-layer FULL logits[0:5]: {:?}", &model_full_logits[..5]);
    println!("Model The (791): {:.4}", model_full_logits[791]);

    // Argmax and top-5 for 16-layer output
    let mut indexed: Vec<(usize, f32)> = model_full_logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTop-10 tokens (16-layer):");
    for (idx, val) in indexed.iter().take(10) {
        println!("  token={idx} logit={val:.4}");
    }

    // Single BOS token test
    model.clear_cache();
    let bos_logits = model.forward(128000);
    let mut bos_indexed: Vec<(usize, f32)> = bos_logits.iter().copied().enumerate().collect();
    bos_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nSingle BOS (128000) top-10:");
    for (idx, val) in bos_indexed.iter().take(10) {
        println!("  token={idx} logit={val:.4}");
    }

    // Test actual generation with the model
    let tok = alice_llm::gguf::GgufTokenizer::from_gguf(&gguf).unwrap();

    // Check: does encode() already produce BOS from <|begin_of_text|>?
    let prompt_with_bos = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is 1+1?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    let encoded = tok.encode(prompt_with_bos);
    println!("\nTokenizer debug:");
    println!("  bos_id = {}", tok.bos_id);
    println!("  eos_id = {}", tok.eos_id);
    println!("  encoded[0:10] = {:?}", &encoded[..encoded.len().min(10)]);
    println!("  encoded len = {}", encoded.len());
    println!("  BOS duplicate? generate() prepends bos_id={}, and encode produces [0]={}", tok.bos_id, encoded[0]);

    // Test with prompt that already includes <|begin_of_text|> (will get double BOS)
    let result = model.generate(&tok, prompt_with_bos, 30, 0.0, 40);
    println!("\nWith <|begin_of_text|> (double BOS): \"{}\"", result.text);

    // Test WITHOUT <|begin_of_text|> in prompt (generate() adds BOS)
    model.clear_cache();
    let prompt_no_bos = "<|start_header_id|>user<|end_header_id|>\n\nWhat is 1+1?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    let result2 = model.generate(&tok, prompt_no_bos, 30, 0.0, 40);
    println!("Without <|begin_of_text|> (single BOS): \"{}\"", result2.text);
    println!("  {} tokens, {:.1} tok/s", result2.tokens_generated, result2.tokens_per_sec);
}
