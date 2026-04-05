"""Extract and verify BOS embedding from GGUF file.

Implements Q4_K dequantization in Python to cross-check against Rust.
Also compares with llama-cpp-python reference.
"""
import struct
import sys
import numpy as np

GGUF_MAGIC = 0x46554747

# GGML types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14

TYPE_NAMES = {0:"F32",1:"F16",8:"Q8_0",12:"Q4_K",13:"Q5_K",14:"Q6_K"}
QK_K = 256


def read_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')


def skip_meta_value(f, vtype):
    sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:8,8:8,9:8,10:8,11:8,12:8}
    if vtype == 8:
        read_string(f)
    elif vtype == 9:
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        for _ in range(arr_len):
            skip_meta_value(f, arr_type)
    elif vtype in sizes:
        f.read(sizes[vtype])
    else:
        raise ValueError(f"Unknown type {vtype}")


def read_gguf_header(f):
    magic = struct.unpack('<I', f.read(4))[0]
    assert magic == GGUF_MAGIC
    version = struct.unpack('<I', f.read(4))[0]
    tensor_count = struct.unpack('<Q', f.read(8))[0]
    metadata_count = struct.unpack('<Q', f.read(8))[0]
    print(f"GGUF v{version}: {tensor_count} tensors, {metadata_count} metadata")

    for _ in range(metadata_count):
        read_string(f)
        vtype = struct.unpack('<I', f.read(4))[0]
        skip_meta_value(f, vtype)

    tensors = {}
    for _ in range(tensor_count):
        name = read_string(f)
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        qtype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        tensors[name] = {"dims": dims, "qtype": qtype, "offset": offset}

    alignment = 32
    data_start = (f.tell() + alignment - 1) // alignment * alignment
    return tensors, data_start


def get_scale_min_k4(j, scales):
    """Get scale and min for Q4_K sub-block j (0-7)."""
    if j < 4:
        sc = scales[j] & 63
        m = scales[j + 4] & 63
    else:
        sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return sc, m


def dequantize_q4_k_block(block_data):
    """Dequantize a single Q4_K block (256 elements from 144 bytes)."""
    d = np.frombuffer(block_data[0:2], dtype=np.float16)[0].astype(np.float32)
    dmin = np.frombuffer(block_data[2:4], dtype=np.float16)[0].astype(np.float32)
    scales = np.frombuffer(block_data[4:16], dtype=np.uint8)
    qs = np.frombuffer(block_data[16:144], dtype=np.uint8)

    out = np.zeros(256, dtype=np.float32)
    is_idx = 0
    q_offset = 0
    elem = 0

    for _ in range(4):
        sc1, m1 = get_scale_min_k4(is_idx, scales)
        sc2, m2 = get_scale_min_k4(is_idx + 1, scales)
        d1 = d * float(sc1)
        m1f = dmin * float(m1)
        d2 = d * float(sc2)
        m2f = dmin * float(m2)

        for l in range(32):
            out[elem] = d1 * float(qs[q_offset + l] & 0xF) - m1f
            elem += 1
        for l in range(32):
            out[elem] = d2 * float(qs[q_offset + l] >> 4) - m2f
            elem += 1

        q_offset += 32
        is_idx += 2

    return out


def dequantize_q4_k_row(row_data, ne0):
    """Dequantize a full row of Q4_K data."""
    block_bytes = 144
    blocks_per_row = ne0 // QK_K
    out = np.zeros(ne0, dtype=np.float32)
    for bi in range(blocks_per_row):
        block = row_data[bi * block_bytes:(bi + 1) * block_bytes]
        vals = dequantize_q4_k_block(block)
        out[bi * QK_K:(bi + 1) * QK_K] = vals
    return out


def dequantize_q8_0_row(row_data, ne0):
    """Dequantize a full row of Q8_0 data."""
    block_bytes = 34
    block_size = 32
    blocks = ne0 // block_size
    out = np.zeros(ne0, dtype=np.float32)
    for bi in range(blocks):
        off = bi * block_bytes
        d = np.frombuffer(row_data[off:off+2], dtype=np.float16)[0].astype(np.float32)
        quants = np.frombuffer(row_data[off+2:off+34], dtype=np.int8).astype(np.float32)
        out[bi * block_size:(bi + 1) * block_size] = d * quants
    return out


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/ys/models/elyza-gguf/Llama-3-ELYZA-JP-8B-q4_k_m.gguf"
    print(f"Reading: {model_path}")

    with open(model_path, 'rb') as f:
        tensors, data_start = read_gguf_header(f)

    emb_info = tensors["token_embd.weight"]
    ne0 = emb_info['dims'][0]
    ne1 = emb_info['dims'][1]
    qtype = emb_info['qtype']
    print(f"\ntoken_embd.weight: dims=[{ne0}, {ne1}], qtype={TYPE_NAMES.get(qtype, qtype)}")

    bos_id = 128000

    if qtype == GGML_TYPE_Q4_K:
        blocks_per_row = ne0 // QK_K
        block_bytes = 144
        row_bytes = blocks_per_row * block_bytes
    elif qtype == GGML_TYPE_Q8_0:
        blocks_per_row = ne0 // 32
        block_bytes = 34
        row_bytes = blocks_per_row * block_bytes
    else:
        print(f"Unsupported qtype: {qtype}")
        return

    row_offset = data_start + emb_info['offset'] + bos_id * row_bytes
    print(f"BOS row: offset={row_offset}, row_bytes={row_bytes}")

    with open(model_path, 'rb') as f:
        f.seek(row_offset)
        row_data = f.read(row_bytes)

    if qtype == GGML_TYPE_Q4_K:
        embedding = dequantize_q4_k_row(row_data, ne0)
    elif qtype == GGML_TYPE_Q8_0:
        embedding = dequantize_q8_0_row(row_data, ne0)

    print(f"\n=== BOS Embedding (Python Q4_K dequantization) ===")
    print(f"  First 10: {embedding[:10]}")
    print(f"  First 20: {embedding[:20]}")
    print(f"  sum={embedding.sum():.6f}, L2={np.linalg.norm(embedding):.6f}")
    print(f"  min={embedding.min():.6f}, max={embedding.max():.6f}")
    print(f"  nonzero count: {np.count_nonzero(embedding)}")
    print(f"  unique values (first block): {len(np.unique(embedding[:256]))}")

    # Reference from Rust output for comparison
    print(f"\n=== Expected from Rust ===")
    print(f"  First 10: [-0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001, 0.0041, -0.0001, -0.0001, -0.0001]")
    print(f"  sum=0.23, L2=0.43")

    # Check if they match
    rust_first10 = [-0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001, 0.0041, -0.0001, -0.0001, -0.0001]
    match = np.allclose(embedding[:10], rust_first10, atol=1e-3)
    print(f"\n  Python vs Rust first 10 match (atol=1e-3): {match}")

    # Also check blk.0.attn_q.weight first row
    q_info = tensors.get("blk.0.attn_q.weight")
    if q_info:
        q_ne0 = q_info['dims'][0]
        q_qtype = q_info['qtype']
        print(f"\n=== blk.0.attn_q.weight ===")
        print(f"  dims=[{q_ne0}, {q_info['dims'][1]}], qtype={TYPE_NAMES.get(q_qtype, q_qtype)}")

        if q_qtype == GGML_TYPE_Q4_K:
            q_blocks_per_row = q_ne0 // QK_K
            q_row_bytes = q_blocks_per_row * 144
            q_offset = data_start + q_info['offset']
            with open(model_path, 'rb') as f:
                f.seek(q_offset)
                q_row_data = f.read(q_row_bytes)
            q_row0 = dequantize_q4_k_row(q_row_data, q_ne0)
            print(f"  Row 0 first 10: {q_row0[:10]}")
            print(f"  Row 0 sum={q_row0.sum():.6f}, L2={np.linalg.norm(q_row0):.6f}")

    # llama-cpp-python reference
    print(f"\n=== llama-cpp-python Reference ===")
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=model_path, n_ctx=256, n_gpu_layers=0, verbose=False, logits_all=True)
        bos = llm.token_bos()
        llm.eval([bos])
        ref = llm.scores[0]
        print(f"  Ref logits[0:10]: {ref[:10]}")
        print(f"  Ref sum={ref.sum():.2f}, L2={np.linalg.norm(ref):.2f}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    main()
