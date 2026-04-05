"""Full 32-layer Llama-3 forward pass in Python using dequantize+f32 matmul.
Tests whether our f32-input approach is fundamentally broken."""
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
        struct.unpack('<I', f.read(4))  # version
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
    if j < 4: return int(scales[j] & 63), int(scales[j + 4] & 63)
    sc = int((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4))
    m = int((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4))
    return sc, m


def dequantize_q4_k_matrix(path, data_start, info):
    """Dequantize full Q4_K matrix using numpy vectorization."""
    ne0 = info['dims'][0]
    ne1 = info['dims'][1] if len(info['dims']) > 1 else 1
    blocks_per_row = ne0 // QK_K
    block_bytes = 144
    total_bytes = ne1 * blocks_per_row * block_bytes

    with open(path, 'rb') as f:
        f.seek(data_start + info['offset'])
        raw = np.frombuffer(f.read(total_bytes), dtype=np.uint8)

    n_blocks = ne1 * blocks_per_row
    blocks = raw.reshape(n_blocks, block_bytes)

    # Extract d, dmin from each block (f16)
    d_raw = blocks[:, 0:2].copy()
    d = np.frombuffer(d_raw.tobytes(), dtype=np.float16).astype(np.float32)
    dmin_raw = blocks[:, 2:4].copy()
    dmin = np.frombuffer(dmin_raw.tobytes(), dtype=np.float16).astype(np.float32)

    scales_all = blocks[:, 4:16]  # [n_blocks, 12]
    qs_all = blocks[:, 16:144]    # [n_blocks, 128]

    out = np.zeros((n_blocks, QK_K), dtype=np.float32)

    for bi in range(n_blocks):
        scales = scales_all[bi]
        qs = qs_all[bi]
        is_idx = 0
        q_off = 0
        elem = 0
        for _ in range(4):
            sc1, m1 = get_scale_min_k4(is_idx, scales)
            sc2, m2 = get_scale_min_k4(is_idx + 1, scales)
            d1 = float(d[bi]) * sc1
            m1f = float(dmin[bi]) * m1
            d2 = float(d[bi]) * sc2
            m2f = float(dmin[bi]) * m2
            for l in range(32):
                out[bi, elem] = d1 * float(qs[q_off + l] & 0xF) - m1f
                elem += 1
            for l in range(32):
                out[bi, elem] = d2 * float(qs[q_off + l] >> 4) - m2f
                elem += 1
            q_off += 32
            is_idx += 2

    return out.reshape(ne1, ne0)


def dequantize_q6_k_matrix(path, data_start, info):
    """Dequantize full Q6_K matrix."""
    ne0 = info['dims'][0]
    ne1 = info['dims'][1] if len(info['dims']) > 1 else 1
    blocks_per_row = ne0 // QK_K
    block_bytes = 210
    total_bytes = ne1 * blocks_per_row * block_bytes

    with open(path, 'rb') as f:
        f.seek(data_start + info['offset'])
        raw = np.frombuffer(f.read(total_bytes), dtype=np.uint8)

    n_blocks = ne1 * blocks_per_row
    blocks = raw.reshape(n_blocks, block_bytes)

    out = np.zeros((n_blocks, QK_K), dtype=np.float32)

    for bi in range(n_blocks):
        block = blocks[bi]
        ql = block[0:128]
        qh = block[128:192]
        scales = block[192:208]
        d_raw = block[208:210].copy()
        d = float(np.frombuffer(d_raw.tobytes(), dtype=np.float16)[0])

        ql_off = 0
        qh_off = 0
        elem = 0

        for n in range(0, QK_K, 128):
            is_idx = n // 16
            for l in range(32):
                q1 = int((ql[ql_off+l] & 0xF) | (((qh[qh_off+l] >> 0) & 3) << 4)) - 32
                q2 = int((ql[ql_off+l+32] & 0xF) | (((qh[qh_off+l] >> 2) & 3) << 4)) - 32
                q3 = int((ql[ql_off+l] >> 4) | (((qh[qh_off+l] >> 4) & 3) << 4)) - 32
                q4 = int((ql[ql_off+l+32] >> 4) | (((qh[qh_off+l] >> 6) & 3) << 4)) - 32

                sc_val = np.int8(scales[is_idx]).item()
                sc2_val = np.int8(scales[is_idx + 2]).item()
                sc4_val = np.int8(scales[is_idx + 4]).item()
                sc6_val = np.int8(scales[is_idx + 6]).item()

                out[bi, elem + l] = d * sc_val * q1
                out[bi, elem + l + 32] = d * sc2_val * q2
                out[bi, elem + l + 64] = d * sc4_val * q3
                out[bi, elem + l + 96] = d * sc6_val * q4

            elem += 128
            ql_off += 64
            qh_off += 32

    return out.reshape(ne1, ne0)


def load_f32_tensor(path, data_start, info):
    ne0 = info['dims'][0]
    n_el = 1
    for d in info['dims']: n_el *= d
    with open(path, 'rb') as f:
        f.seek(data_start + info['offset'])
        return np.frombuffer(f.read(n_el * 4), dtype=np.float32)


def load_weight_matrix(path, data_start, info):
    qtype = info['qtype']
    if qtype == 12:  # Q4_K
        return dequantize_q4_k_matrix(path, data_start, info)
    elif qtype == 14:  # Q6_K
        return dequantize_q6_k_matrix(path, data_start, info)
    elif qtype == 0:  # F32
        ne0 = info['dims'][0]
        ne1 = info['dims'][1] if len(info['dims']) > 1 else 1
        return load_f32_tensor(path, data_start, info).reshape(ne1, ne0)
    else:
        raise ValueError(f"Unsupported qtype: {qtype}")


def rms_norm(x, weight, eps=1e-5):
    ss = np.sum(x.astype(np.float64) ** 2)
    mean = np.float32(ss / len(x))
    scale = np.float32(1.0 / np.sqrt(mean + eps))
    return x * scale * weight


def silu(x):
    return x / (1.0 + np.exp(-x))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/Users/ys/models/elyza-gguf/Llama-3-ELYZA-JP-8B-q4_k_m.gguf"
    print(f"Loading: {path}")
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

    # Load embedding row
    emb_info = tensors["token_embd.weight"]
    blocks_per_row = hidden_dim // QK_K
    row_bytes = blocks_per_row * 144
    with open(path, 'rb') as f:
        f.seek(data_start + emb_info['offset'] + bos_id * row_bytes)
        emb_raw = f.read(row_bytes)
    # Reuse dequant function on single row
    emb_data_arr = np.frombuffer(emb_raw, dtype=np.uint8)
    h_info_tmp = {"dims": [hidden_dim, 1], "qtype": 12, "offset": 0}  # dummy
    # Just dequantize manually
    hidden = np.zeros(hidden_dim, dtype=np.float32)
    for bi in range(blocks_per_row):
        block = emb_data_arr[bi*144:(bi+1)*144]
        d = np.frombuffer(block[0:2].tobytes(), dtype=np.float16)[0].astype(np.float32)
        dmin_val = np.frombuffer(block[2:4].tobytes(), dtype=np.float16)[0].astype(np.float32)
        scales = block[4:16]
        qs = block[16:144]
        is_idx = 0; q_off = 0; elem = bi * QK_K
        for _ in range(4):
            sc1, m1 = get_scale_min_k4(is_idx, scales)
            sc2, m2 = get_scale_min_k4(is_idx+1, scales)
            d1 = float(d) * sc1; m1f = float(dmin_val) * m1
            d2 = float(d) * sc2; m2f = float(dmin_val) * m2
            for l in range(32):
                hidden[elem] = d1 * float(qs[q_off+l] & 0xF) - m1f; elem += 1
            for l in range(32):
                hidden[elem] = d2 * float(qs[q_off+l] >> 4) - m2f; elem += 1
            q_off += 32; is_idx += 2

    print(f"BOS embedding: sum={hidden.sum():.4f}, L2={np.linalg.norm(hidden):.4f}")

    t0 = time.time()
    for layer_idx in range(num_layers):
        prefix = f"blk.{layer_idx}"
        t1 = time.time()

        # Attention norm
        an_w = load_f32_tensor(path, data_start, tensors[f"{prefix}.attn_norm.weight"])
        norm_buf = rms_norm(hidden, an_w)

        # Q, K, V projections
        q_w = load_weight_matrix(path, data_start, tensors[f"{prefix}.attn_q.weight"])
        k_w = load_weight_matrix(path, data_start, tensors[f"{prefix}.attn_k.weight"])
        v_w = load_weight_matrix(path, data_start, tensors[f"{prefix}.attn_v.weight"])

        q_buf = q_w @ norm_buf
        k_buf = k_w @ norm_buf
        v_buf = v_w @ norm_buf

        # RoPE at position 0 is identity (no-op)

        # Attention: seq_len=1, softmax=[1.0], output = V broadcast
        attn_out = np.zeros(hidden_dim, dtype=np.float32)
        for h in range(num_heads):
            kv_h = h // heads_per_kv
            qs = h * head_dim
            vs = kv_h * head_dim
            attn_out[qs:qs+head_dim] = v_buf[vs:vs+head_dim]

        # Output projection + residual
        o_w = load_weight_matrix(path, data_start, tensors[f"{prefix}.attn_output.weight"])
        o_buf = o_w @ attn_out
        hidden = hidden + o_buf

        # FFN
        fn_w = load_f32_tensor(path, data_start, tensors[f"{prefix}.ffn_norm.weight"])
        norm_buf = rms_norm(hidden, fn_w)

        g_w = load_weight_matrix(path, data_start, tensors[f"{prefix}.ffn_gate.weight"])
        u_w = load_weight_matrix(path, data_start, tensors[f"{prefix}.ffn_up.weight"])
        d_w = load_weight_matrix(path, data_start, tensors[f"{prefix}.ffn_down.weight"])

        gate = g_w @ norm_buf
        up = u_w @ norm_buf
        ffn_out = silu(gate) * up
        down = d_w @ ffn_out
        hidden = hidden + down

        dt = time.time() - t1
        l2 = np.linalg.norm(hidden)
        print(f"  layer {layer_idx:2d}: L2={l2:8.2f}, sum={hidden.sum():10.2f} ({dt:.1f}s)")

        # Free weight matrices
        del q_w, k_w, v_w, o_w, g_w, u_w, d_w

    # Final output
    on_w = load_f32_tensor(path, data_start, tensors["output_norm.weight"])
    norm_out = rms_norm(hidden, on_w)
    out_w = load_weight_matrix(path, data_start, tensors["output.weight"])
    logits = out_w @ norm_out

    print(f"\n=== Python 32-layer logits ===")
    print(f"  sum={logits.sum():.2f}, L2={np.linalg.norm(logits):.2f}")
    print(f"  logits[0:10]: {logits[:10]}")
    print(f"  Total time: {time.time() - t0:.1f}s")

    # Top-10
    top10 = np.argsort(logits)[::-1][:10]
    print(f"  Top-10: {[(int(i), float(logits[i])) for i in top10]}")

    # Reference
    print(f"\n  Rust   logits[0:10]: [4.94, 3.55, 5.33, 2.20, 5.22, 4.10, 3.12, 0.28, 2.34, 4.10]")
    print(f"  Ref    logits[0:10]: [7.79, 7.03, 6.98, 6.67, 5.41, 6.44, 4.10, 8.04, 8.78, 5.11]")


if __name__ == "__main__":
    main()
