"""Compare reference logits from llama.cpp against our implementation."""
import numpy as np
from llama_cpp import Llama

model_path = "/Users/ys/models/elyza-gguf/Llama-3-ELYZA-JP-8B-q4_k_m.gguf"
print(f"Loading model: {model_path}")
llm = Llama(model_path=model_path, n_ctx=256, n_gpu_layers=0, verbose=False, logits_all=True)

# Process BOS token and get logits
bos_id = llm.token_bos()
print(f"BOS token: {bos_id}")

# Eval BOS
llm.eval([bos_id])
logits = llm.scores[0]  # logits after BOS
top10_idx = np.argsort(logits)[::-1][:10]

print("\nTop-10 after BOS (llama.cpp reference):")
for idx in top10_idx:
    tok = llm.detokenize([idx]).decode('utf-8', errors='replace')
    print(f"  [{idx}] {logits[idx]:.4f} '{tok}'")

# Print first 10 logits values for comparison
print(f"\nLogits[0:10]: {logits[0:10]}")
print(f"Logits sum: {logits.sum():.2f}")
print(f"Logits min: {logits.min():.4f}, max: {logits.max():.4f}")
print(f"Logits L2: {np.sqrt((logits**2).sum()):.2f}")

# Now process instruction prompt
llm.reset()
prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n日本の首都はどこですか？<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
tokens = llm.tokenize(prompt.encode('utf-8'), special=True)
print(f"\nPrompt tokens ({len(tokens)}): {tokens[:20]}")

llm.eval(tokens)
last_logits = llm.scores[len(tokens) - 1]
top10_idx2 = np.argsort(last_logits)[::-1][:10]

print("\nTop-10 after instruction prompt (llama.cpp reference):")
for idx in top10_idx2:
    tok = llm.detokenize([idx]).decode('utf-8', errors='replace')
    print(f"  [{idx}] {last_logits[idx]:.4f} '{tok}'")

# Generate a few tokens
print("\nGeneration (greedy, 10 tokens):")
output = llm.create_completion(prompt, max_tokens=10, temperature=0.0)
print(f"  Output: {output['choices'][0]['text']}")
