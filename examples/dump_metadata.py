"""Dump all GGUF metadata keys and values."""
import struct
import sys

GGUF_MAGIC = 0x46554747
TYPE_NAMES = {0:"F32",1:"F16",8:"Q8_0",12:"Q4_K",13:"Q5_K",14:"Q6_K"}

def read_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8', errors='replace')

def read_meta_value(f, vtype):
    if vtype == 0: return struct.unpack('<B', f.read(1))[0]  # u8
    elif vtype == 1: return struct.unpack('<b', f.read(1))[0]  # i8
    elif vtype == 2: return struct.unpack('<H', f.read(2))[0]  # u16
    elif vtype == 3: return struct.unpack('<h', f.read(2))[0]  # i16
    elif vtype == 4: return struct.unpack('<I', f.read(4))[0]  # u32
    elif vtype == 5: return struct.unpack('<i', f.read(4))[0]  # i32
    elif vtype == 6: return struct.unpack('<f', f.read(4))[0]  # f32
    elif vtype == 7: return struct.unpack('<?', f.read(1))[0]  # bool
    elif vtype == 8: return read_string(f)  # string
    elif vtype == 9:  # array
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        if arr_len > 20:
            # Skip large arrays (token lists etc), just show count
            for _ in range(arr_len):
                read_meta_value(f, arr_type)
            return f"[array of {arr_len} elements, type={arr_type}]"
        return [read_meta_value(f, arr_type) for _ in range(arr_len)]
    elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]  # u64
    elif vtype == 11: return struct.unpack('<q', f.read(8))[0]  # i64
    elif vtype == 12: return struct.unpack('<d', f.read(8))[0]  # f64
    else:
        raise ValueError(f"Unknown type {vtype}")

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/Users/ys/models/elyza-gguf/Llama-3-ELYZA-JP-8B-q4_k_m.gguf"
    with open(path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_meta = struct.unpack('<Q', f.read(8))[0]
        print(f"GGUF v{version}, {n_tensors} tensors, {n_meta} metadata\n")

        print("=== ALL METADATA ===")
        for _ in range(n_meta):
            key = read_string(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            val = read_meta_value(f, vtype)
            # Skip huge arrays
            if isinstance(val, str) and val.startswith("[array"):
                print(f"  {key}: {val}")
            elif isinstance(val, list) and len(val) > 10:
                print(f"  {key}: [list of {len(val)} items]")
            else:
                print(f"  {key}: {val}")

        # Read tensor info and show first few + qtypes
        print(f"\n=== TENSOR TYPES (first 20 + last 5) ===")
        tensor_infos = []
        for _ in range(n_tensors):
            name = read_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            qtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensor_infos.append((name, dims, qtype))

        for i, (name, dims, qtype) in enumerate(tensor_infos):
            if i < 20 or i >= len(tensor_infos) - 5:
                print(f"  {name}: dims={dims}, qtype={TYPE_NAMES.get(qtype, qtype)}")
            elif i == 20:
                print(f"  ... ({len(tensor_infos) - 25} more) ...")

if __name__ == "__main__":
    main()
