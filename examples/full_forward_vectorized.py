"""Full 32-layer Llama-3 forward pass using fully vectorized numpy dequantization.
Determines whether dequantize+f32 approach is fundamentally broken."""
import struct, sys, time
import numpy as np

GGUF_MAGIC = 0x46554747
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


def read_gguf(path):
    with open(path, 'rb') as f:
        assert struct.unpack('<I', f.read(4))[0] == GGUF_MAGIC
        struct.unpack('<I', f.read(4))
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


def dequant_q4k_vectorized(raw_blocks, n_blocks):
    """Fully vectorized Q4_K dequantization. raw_blocks: [n_blocks, 144]"""
    # Extract d, dmin (f16)
    d_bytes = raw_blocks[:, 0:2].copy()
    d = np.frombuffer(d_bytes.tobytes(), dtype=np.float16).astype(np.float32)
    dmin_bytes = raw_blocks[:, 2:4].copy()
    dmin = np.frombuffer(dmin_bytes.tobytes(), dtype=np.float16).astype(np.float32)

    scales = raw_blocks[:, 4:16]   # [n_blocks, 12]
    qs = raw_blocks[:, 16:144]     # [n_blocks, 128]

    out = np.zeros((n_blocks, QK_K), dtype=np.float32)

    # Process 4 sub-blocks per block, each producing 64 elements
    for sb in range(4):
        is_idx = sb * 2
        q_off = sb * 32

        # Compute scale/min for sub-block pair
        if is_idx < 4:
            sc1 = (scales[:, is_idx] & 63).astype(np.float32)
            m1 = (scales[:, is_idx + 4] & 63).astype(np.float32)
        else:
            sc1 = ((scales[:, is_idx + 4] & 0xF) | ((scales[:, is_idx - 4] >> 6) << 4)).astype(np.float32)
            m1 = ((scales[:, is_idx + 4] >> 4) | ((scales[:, is_idx] >> 6) << 4)).astype(np.float32)

        is_idx2 = is_idx + 1
        if is_idx2 < 4:
            sc2 = (scales[:, is_idx2] & 63).astype(np.float32)
            m2 = (scales[:, is_idx2 + 4] & 63).astype(np.float32)
        else:
            sc2 = ((scales[:, is_idx2 + 4] & 0xF) | ((scales[:, is_idx2 - 4] >> 6) << 4)).astype(np.float32)
            m2 = ((scales[:, is_idx2 + 4] >> 4) | ((scales[:, is_idx2] >> 6) << 4)).astype(np.float32)

        d1 = d * sc1   # [n_blocks]
        m1f = dmin * m1
        d2 = d * sc2
        m2f = dmin * m2

        # Low nibbles -> first 32 elements
        elem_start = sb * 64
        for l in range(32):
            out[:, elem_start + l] = d1 * (qs[:, q_off + l] & 0xF).astype(np.float32) - m1f
        # High nibbles -> next 32 elements
        for l in range(32):
            out[:, elem_start + 32 + l] = d2 * (qs[:, q_off + l] >> 4).astype(np.float32) - m2f

    return out


def dequant_q6k_vectorized(raw_blocks, n_blocks):
    """Fully vectorized Q6_K dequantization. raw_blocks: [n_blocks, 210]"""
    ql = raw_blocks[:, 0:128]      # [n_blocks, 128]
    qh = raw_blocks[:, 128:192]    # [n_blocks, 64]
    sc = raw_blocks[:, 192:208]    # [n_blocks, 16]
    d_bytes = raw_blocks[:, 208:210].copy()
    d = np.frombuffer(d_bytes.tobytes(), dtype=np.float16).astype(np.float32)  # [n_blocks]

    out = np.zeros((n_blocks, QK_K), dtype=np.float32)

    # Two groups of 128 elements each
    for grp in range(2):
        ql_off = grp * 64
        qh_off = grp * 32
        is_base = grp * 8  # n=grp*128, is=n/16

        for l in range(32):
            q1 = ((ql[:, ql_off + l] & 0xF) | (((qh[:, qh_off + l] >> 0) & 3) << 4)).astype(np.int8) - np.int8(32)
            q2 = ((ql[:, ql_off + l + 32] & 0xF) | (((qh[:, qh_off + l] >> 2) & 3) << 4)).astype(np.int8) - np.int8(32)
            q3 = ((ql[:, ql_off + l] >> 4) | (((qh[:, qh_off + l] >> 4) & 3) << 4)).astype(np.int8) - np.int8(32)
            q4 = ((ql[:, ql_off + l + 32] >> 4) | (((qh[:, qh_off + l] >> 6) & 3) << 4)).astype(np.int8) - np.int8(32)

            sc1 = sc[:, is_base].view(np.int8).astype(np.float32)
            sc2 = sc[:, is_base + 2].view(np.int8).astype(np.float32)
            sc3 = sc[:, is_base + 4].view(np.int8).astype(np.float32)
            sc4 = sc[:, is_base + 6].view(np.int8).astype(np.float32)

            elem = grp * 128
            out[:, elem + l] = d * sc1 * q1.astype(np.float32)
            out[:, elem + l + 32] = d * sc2 * q2.astype(np.float32)
            out[:, elem + l + 64] = d * sc3 * q3.astype(np.float32)
            out[:, elem + l + 96] = d * sc4 * q4.astype(np.float32)

    return out


def load_matrix(path, data_start, info):
    """Load and dequantize a weight matrix. Returns [rows, cols] f32."""
    ne0 = info['dims'][0]  # cols
    ne1 = info['dims'][1] if len(info['dims']) > 1 else 1
    qtype = info['qtype']

    if qtype == 0:  # F32
        with open(path, 'rb') as f:
            f.seek(data_start + info['offset'])
            return np.frombuffer(f.read(ne0 * ne1 * 4), dtype=np.float32).reshape(ne1, ne0)
    elif qtype == 12:  # Q4_K
        blocks_per_row = ne0 // QK_K
        block_bytes = 144
        total_bytes = ne1 * blocks_per_row * block_bytes
        with open(path, 'rb') as f:
            f.seek(data_start + info['offset'])
            raw = np.frombuffer(f.read(total_bytes), dtype=np.uint8)
        n_blocks = ne1 * blocks_per_row
        blocks = raw.reshape(n_blocks, block_bytes)
        out = dequant_q4k_vectorized(blocks, n_blocks)
        return out.reshape(ne1, ne0)
    elif qtype == 14:  # Q6_K
        blocks_per_row = ne0 // QK_K
        block_bytes = 210
        total_bytes = ne1 * blocks_per_row * block_bytes
        with open(path, 'rb') as f:
            f.seek(data_start + info['offset'])
            raw = np.frombuffer(f.read(total_bytes), dtype=np.uint8)
        n_blocks = ne1 * blocks_per_row
        blocks = raw.reshape(n_blocks, block_bytes)
        out = dequant_q6k_vectorized(blocks, n_blocks)
        return out.reshape(ne1, ne0)
    else:
        raise ValueError(f"Unsupported qtype: {qtype}")


def load_f32_vec(path, data_start, info):
    """Load F32 vector (norm weights)."""
    ne0 = info['dims'][0]
    with open(path, 'rb') as f:
        f.seek(data_start + info['offset'])
        return np.frombuffer(f.read(ne0 * 4), dtype=np.float32).copy()


def rms_norm(x, weight, eps=1e-5):
    ss = np.sum(x.astype(np.float64) ** 2)
    mean = np.float32(ss / len(x))
    scale = np.float32(1.0 / np.sqrt(mean + eps))
    return x * scale * weight


def silu(x):
    return x / (1.0 + np.exp(-x.clip(-88, 88)))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/Users/ys/models/elyza-gguf/Llama-3-ELYZA-JP-8B-q4_k_m.gguf"
    print(f"Loading GGUF: {path}")
    tensors, data_start = read_gguf(path)

    hidden_dim = 4096
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    intermediate_dim = 14336
    num_layers = 32
    kv_dim = num_kv_heads * head_dim
    heads_per_kv = num_heads // num_kv_heads
    bos_id = 128000

    # BOS embedding (dequantize single row)
    emb_info = tensors["token_embd.weight"]
    blocks_per_row = hidden_dim // QK_K
    row_bytes = blocks_per_row * 144
    with open(path, 'rb') as f:
        f.seek(data_start + emb_info['offset'] + bos_id * row_bytes)
        emb_raw = np.frombuffer(f.read(row_bytes), dtype=np.uint8)
    emb_blocks = emb_raw.reshape(blocks_per_row, 144)
    hidden = dequant_q4k_vectorized(emb_blocks, blocks_per_row).flatten()
    print(f"BOS embedding: sum={hidden.sum():.4f}, L2={np.linalg.norm(hidden):.4f}")

    t0 = time.time()
    for layer_idx in range(num_layers):
        prefix = f"blk.{layer_idx}"
        t1 = time.time()

        # Attention norm
        an_w = load_f32_vec(path, data_start, tensors[f"{prefix}.attn_norm.weight"])
        norm_buf = rms_norm(hidden, an_w)

        # Q, K, V projections (dequantize full matrix, then matvec)
        print(f"  layer {layer_idx:2d}: loading Q...", end="", flush=True)
        q_w = load_matrix(path, data_start, tensors[f"{prefix}.attn_q.weight"])
        print(f" K...", end="", flush=True)
        k_w = load_matrix(path, data_start, tensors[f"{prefix}.attn_k.weight"])
        print(f" V...", end="", flush=True)
        v_w = load_matrix(path, data_start, tensors[f"{prefix}.attn_v.weight"])

        q_buf = q_w @ norm_buf
        k_buf = k_w @ norm_buf
        v_buf = v_w @ norm_buf
        del q_w, k_w, v_w

        # RoPE at position 0 is identity (cos(0)=1, sin(0)=0)

        # Attention: seq_len=1, softmax([x])=[1.0], output = V broadcast
        attn_out = np.zeros(hidden_dim, dtype=np.float32)
        for h in range(num_heads):
            kv_h = h // heads_per_kv
            qs = h * head_dim
            vs = kv_h * head_dim
            attn_out[qs:qs+head_dim] = v_buf[vs:vs+head_dim]

        # Output projection + residual
        print(f" O...", end="", flush=True)
        o_w = load_matrix(path, data_start, tensors[f"{prefix}.attn_output.weight"])
        o_buf = o_w @ attn_out
        del o_w
        hidden = hidden + o_buf

        # FFN
        fn_w = load_f32_vec(path, data_start, tensors[f"{prefix}.ffn_norm.weight"])
        norm_buf = rms_norm(hidden, fn_w)

        print(f" gate...", end="", flush=True)
        g_w = load_matrix(path, data_start, tensors[f"{prefix}.ffn_gate.weight"])
        print(f" up...", end="", flush=True)
        u_w = load_matrix(path, data_start, tensors[f"{prefix}.ffn_up.weight"])
        print(f" down...", end="", flush=True)
        d_w = load_matrix(path, data_start, tensors[f"{prefix}.ffn_down.weight"])

        gate = g_w @ norm_buf
        up = u_w @ norm_buf
        ffn_out = silu(gate) * up
        down = d_w @ ffn_out
        del g_w, u_w, d_w
        hidden = hidden + down

        dt = time.time() - t1
        l2 = np.linalg.norm(hidden)
        print(f" L2={l2:8.2f}, sum={hidden.sum():10.2f} ({dt:.1f}s)")

    # Final output
    on_w = load_f32_vec(path, data_start, tensors["output_norm.weight"])
    norm_out = rms_norm(hidden, on_w)
    print("Computing output logits...")
    out_w = load_matrix(path, data_start, tensors["output.weight"])
    logits = out_w @ norm_out

    print(f"\n=== Vectorized Python 32-layer logits ===")
    print(f"  sum={logits.sum():.2f}, L2={np.linalg.norm(logits):.2f}")
    print(f"  logits[0:10]: {logits[:10]}")
    print(f"  Total time: {time.time() - t0:.1f}s")

    top10 = np.argsort(logits)[::-1][:10]
    print(f"  Top-10: {[(int(i), float(logits[i])) for i in top10]}")

    print(f"\n  Rust   logits[0:10]: [4.94, 3.55, 5.33, 2.20, 5.22, 4.10, 3.12, 0.28, 2.34, 4.10]")
    print(f"  Ref    logits[0:10]: [7.79, 7.03, 6.98, 6.67, 5.41, 6.44, 4.10, 8.04, 8.78, 5.11]")
    print(f"  Rust   sum=14938.75")
    print(f"  Ref    sum=-72694.65")


if __name__ == "__main__":
    main()
