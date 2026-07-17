#!/usr/bin/env python3
"""
DeepSeek-V2-Lite HF transformers oracle for ALICE-LLM validation (Issue #36).

Runs greedy generation on the same prompt as ALICE-LLM's test:
- Prompt: "User: The capital of Japan is\n\nAssistant:" (deepseek2 template from
  `examples/elyza_gguf.rs`)
- Greedy decoding (do_sample=False, temperature=0)
- Output: prompt tokens + generated tokens + per-position top-5 logits

Result saved to /notebooks/v2lite_oracle_result.json for comparison against
ALICE-LLM's `--logits-dump` output.
"""

import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/notebooks/v2lite_hf")
    parser.add_argument("--prompt", default="The capital of Japan is")
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--output", default="/notebooks/v2lite_oracle_result.json")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading model from {args.model} (bfloat16, cuda, eager attention)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="eager",  # avoid flash_attn requirement
    )
    model.eval()

    prompt = f"User: {args.prompt}\n\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    prompt_tokens = inputs["input_ids"][0].tolist()
    print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            return_dict_in_generate=True,
            output_scores=True,
        )

    seq = output.sequences[0].tolist()
    generated_tokens = seq[len(prompt_tokens):]
    print(f"Generated tokens ({len(generated_tokens)}): {generated_tokens}")
    print(f"Generated text: {repr(tokenizer.decode(generated_tokens))}")

    logits_dump = []
    for step_idx, scores in enumerate(output.scores):
        top5 = torch.topk(scores[0], 5)
        logits_dump.append({
            "position": step_idx,
            "top5_token_ids": top5.indices.tolist(),
            "top5_logits": [float(v) for v in top5.values.float().tolist()],
        })

    result = {
        "engine": "hf-transformers-oracle",
        "model": args.model,
        "torch_dtype": "bfloat16",
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "generated_text": tokenizer.decode(generated_tokens),
        "per_position_top5": logits_dump,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
