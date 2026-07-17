#!/usr/bin/env python3
"""
Per-op tensor dump for DeepSeek-V2-Lite HF transformers forward pass.

Uses forward hooks on layer 0's attention + MLP submodules to capture
the same intermediates ALICE-LLM's `forward_deepseek_v3` emits under
ALICE_DEEPSEEK_DUMP=1. Output format matches (JSONL, one line per
op x position) so both dumps can be diffed side-by-side.

Requires: torch, transformers >=4.36, deepseek-ai/DeepSeek-V2-Lite-Chat
loaded at /notebooks/v2lite_hf.
"""

import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def summary(tensor, engine, pos, layer, op):
    """One JSONL line with head/l2/sum for parity with ALICE-LLM's dump."""
    flat = tensor.detach().to(torch.float32).flatten().cpu()
    n = flat.numel()
    head = flat[:5].tolist()
    l2 = float(torch.linalg.norm(flat))
    total = float(flat.sum())
    return json.dumps({
        "engine": engine,
        "pos": pos,
        "layer": layer,
        "op": op,
        "len": n,
        "head": [float(x) for x in head],
        "l2": round(l2, 6),
        "sum": round(total, 6),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/notebooks/v2lite_hf")
    parser.add_argument("--prompt", default="The capital of Japan is")
    parser.add_argument("--output", default="/notebooks/v2lite_hf_dump.jsonl")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    prompt = f"User: {args.prompt}\n\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_tokens = inputs["input_ids"][0].tolist()
    n = len(prompt_tokens)
    print(f"prompt tokens ({n}): {prompt_tokens}")

    # Storage keyed by op name; we'll expand to per-position JSONL after
    # the forward completes.
    caps = {}

    layer0 = model.model.layers[0]
    attn = layer0.self_attn
    mlp = layer0.mlp
    input_norm = layer0.input_layernorm
    post_attn_norm = layer0.post_attention_layernorm

    def hook_hidden_in(module, inputs, outputs):
        # input_layernorm input = the hidden after embedding + optional norm
        # inputs is a tuple; V2-Lite embedding lookup is upstream
        caps["hidden_in_layer0"] = inputs[0].detach()

    def hook_attn_norm(module, inputs, outputs):
        caps["attn_norm"] = outputs.detach()

    def hook_attn_module(module, inputs, outputs):
        # outputs may be (attn_output, attn_weights, past_key_value) tuple
        if isinstance(outputs, tuple):
            caps["attn_output"] = outputs[0].detach()
        else:
            caps["attn_output"] = outputs.detach()

    def hook_q_proj(module, inputs, outputs):
        caps["q_full"] = outputs.detach()

    def hook_kv_a_mqa(module, inputs, outputs):
        caps["kv_a_full"] = outputs.detach()

    def hook_kv_a_layernorm(module, inputs, outputs):
        caps["kv_a_normed"] = outputs.detach()

    def hook_kv_b(module, inputs, outputs):
        caps["kv_up"] = outputs.detach()

    def hook_o_proj(module, inputs, outputs):
        # inputs[0] is the pre-o_proj attention output (ALICE-LLM's `attn_out`),
        # outputs is the post-o_proj (equivalent to ALICE-LLM's `o_proj_out`).
        caps["attn_out"] = inputs[0].detach()
        caps["o_proj_out"] = outputs.detach()

    def hook_post_attn_norm(module, inputs, outputs):
        # inputs[0] is hidden after residual add (attn output + input)
        caps["hidden_post_attn"] = inputs[0].detach()

    def hook_mlp(module, inputs, outputs):
        caps["mlp_out"] = outputs.detach()

    handles = [
        layer0.input_layernorm.register_forward_hook(hook_hidden_in),
        layer0.input_layernorm.register_forward_hook(hook_attn_norm),
        attn.register_forward_hook(hook_attn_module),
        attn.o_proj.register_forward_hook(hook_o_proj),
        attn.kv_a_proj_with_mqa.register_forward_hook(hook_kv_a_mqa),
        attn.kv_a_layernorm.register_forward_hook(hook_kv_a_layernorm),
        attn.kv_b_proj.register_forward_hook(hook_kv_b),
        layer0.post_attention_layernorm.register_forward_hook(hook_post_attn_norm),
        mlp.register_forward_hook(hook_mlp),
    ]
    # V2-Lite dense Q: single q_proj attribute
    if hasattr(attn, "q_proj"):
        handles.append(attn.q_proj.register_forward_hook(hook_q_proj))
    elif hasattr(attn, "q_b_proj"):
        # V2.5 / V3 / R1 LoRA q chain — dump q_b_proj output (post-LoRA)
        handles.append(attn.q_b_proj.register_forward_hook(hook_q_proj))

    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    with open(args.output, "w") as f:
        for op_name, tensor in caps.items():
            # tensor shape (batch=1, seq=n, ...) — dump every position so
            # ALICE-LLM's per-pos dumps line up with matching pos values.
            if tensor.dim() >= 2 and tensor.shape[0] == 1 and tensor.shape[1] == n:
                for pos in range(n):
                    line = summary(tensor[0, pos], "hf", pos, 0, op_name)
                    f.write(line + "\n")
            else:
                # scalar or non-per-position tensor — dump once at pos=-1
                f.write(summary(tensor, "hf", -1, 0, op_name) + "\n")
    print(f"wrote per-op dump: {args.output}")


if __name__ == "__main__":
    main()
