"""Compute layer 0 Q projection in Python and compare against Rust."""
import struct
import sys
import numpy as np

GGUF_MAGIC = 0x46554747
GGML_TYPE_F32 = 0
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14
QK_K = 256


def read_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')


def skip_meta_value(f, vtype):
    sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:8,8:8,9:8,10:8,11:8,12:8}
    if vtype == 8: read_string(f)
    elif vtype == 9:
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        for _ in range(arr_len): skip_meta_value(f, arr_type)
    elif vtype in sizes: f.read(sizes[vtype])
    else: raise ValueError(f"Unknown type {vtype}")


def read_gguf(path):
    with open(path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        assert magic == GGUF_MAGIC
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_meta = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n_meta):
            read_string(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            skip_meta_value(f, vtype)
        tensors = {}
        for _ in range(n_tensors):
            name = read_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            qtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensors[name] = {"dims": dims, "qtype": qtype, "offset": offset}
        data_start = (f.tell() + 31) // 32 * 32
    return tensors, data_start


def get_scale_min_k4(j, scales):
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    else:
        sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
        return sc, m


def dequantize_q4_k_row(data, ne0):
    blocks_per_row = ne0 // QK_K
    out = np.zeros(ne0, dtype=np.float32)
    for bi in range(blocks_per_row):
        block = data[bi*144:(bi+1)*144]
        d = np.frombuffer(block[0:2], dtype=np.float16)[0].astype(np.float32)
        dmin = np.frombuffer(block[2:4], dtype=np.float16)[0].astype(np.float32)
        scales = np.frombuffer(block[4:16], dtype=np.uint8)
        qs = np.frombuffer(block[16:144], dtype=np.uint8)
        is_idx = 0
        q_off = 0
        elem = bi * QK_K
        for _ in range(4):
            sc1, m1 = get_scale_min_k4(is_idx, scales)
            sc2, m2 = get_scale_min_k4(is_idx + 1, scales)
            d1 = d * float(sc1); m1f = dmin * float(m1)
            d2 = d * float(sc2); m2f = dmin * float(m2)
            for l in range(32):
                out[elem] = d1 * float(qs[q_off + l] & 0xF) - m1f
                elem += 1
            for l in range(32):
                out[elem] = d2 * float(qs[q_off + l] >> 4) - m2f
                elem += 1
            q_off += 32
            is_idx += 2
    return out


def read_f32_tensor(path, data_start, info):
    ne0 = info['dims'][0]
    n_elements = 1
    for d in info['dims']: n_elements *= d
    with open(path, 'rb') as f:
        f.seek(data_start + info['offset'])
        data = f.read(n_elements * 4)
    return np.frombuffer(data, dtype=np.float32)


def q4k_matvec(path, data_start, info, input_vec):
    """Compute matrix-vector product for Q4_K weight."""
    ne0 = info['dims'][0]  # cols (input dim)
    ne1 = info['dims'][1]  # rows (output dim)
    blocks_per_row = ne0 // QK_K
    row_bytes = blocks_per_row * 144
    total_bytes = ne1 * row_bytes

    with open(path, 'rb') as f:
        f.seek(data_start + info['offset'])
        all_data = f.read(total_bytes)

    output = np.zeros(ne1, dtype=np.float32)
    for row in range(ne1):
        row_data = all_data[row * row_bytes:(row + 1) * row_bytes]
        row_weights = dequantize_q4_k_row(row_data, ne0)
        output[row] = np.dot(row_weights, input_vec)
    return output


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/Users/ys/models/elyza-gguf/Llama-3-ELYZA-JP-8B-q4_k_m.gguf"
    print(f"Reading: {path}")
    tensors, data_start = read_gguf(path)

    # Step 1: BOS embedding
    emb_info = tensors["token_embd.weight"]
    ne0 = emb_info['dims'][0]  # 4096
    bos_id = 128000
    blocks_per_row = ne0 // QK_K
    row_bytes = blocks_per_row * 144
    with open(path, 'rb') as f:
        f.seek(data_start + emb_info['offset'] + bos_id * row_bytes)
        emb_data = f.read(row_bytes)
    embedding = dequantize_q4_k_row(emb_data, ne0)
    print(f"BOS embedding: sum={embedding.sum():.6f}, L2={np.linalg.norm(embedding):.6f}")
    print(f"  first 10: {embedding[:10]}")

    # Step 2: RMSNorm with attn_norm.weight (F32)
    norm_w = read_f32_tensor(path, data_start, tensors["blk.0.attn_norm.weight"])
    # RMSNorm: same as llama.cpp (f64 sum, f32 mean/scale)
    ss = np.sum(embedding.astype(np.float64) ** 2)
    mean = np.float32(ss / ne0)
    scale = np.float32(1.0 / np.sqrt(mean + 1e-5))
    norm_out = embedding * scale * norm_w
    print(f"\nAfter attn_norm: sum={norm_out.sum():.6f}, L2={np.linalg.norm(norm_out):.6f}")
    print(f"  first 10: {norm_out[:10]}")

    # Compare with Rust
    rust_norm_sum = -9.31
    print(f"  Rust sum: {rust_norm_sum:.2f}, Python sum: {norm_out.sum():.2f}, match: {abs(norm_out.sum() - rust_norm_sum) < 0.1}")

    # Step 3: Q projection (Q4_K matvec)
    print(f"\nComputing Q = attn_q @ norm_out (4096x4096 matvec)...")
    q_info = tensors["blk.0.attn_q.weight"]
    q_out = q4k_matvec(path, data_start, q_info, norm_out)
    print(f"Q: sum={q_out.sum():.6f}, L2={np.linalg.norm(q_out):.6f}")
    print(f"  first 10: {q_out[:10]}")

    # Compare with Rust
    rust_q = [-0.0128, -0.1323, 0.4867, -0.3151, 0.4097, -0.6568, -0.2572, 0.4960, 0.6160, -0.6184]
    print(f"\n  Rust Q first 10: {rust_q}")
    print(f"  Python Q first 10: {q_out[:10].tolist()}")
    diffs = [abs(a - b) for a, b in zip(q_out[:10], rust_q)]
    print(f"  Max diff: {max(diffs):.6e}")
    print(f"  Match (atol=0.01): {all(d < 0.01 for d in diffs)}")

if __name__ == "__main__":
    main()
