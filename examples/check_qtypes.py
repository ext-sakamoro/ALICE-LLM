"""Check all tensor quantization types, especially layers 29-31."""
import struct, sys

GGUF_MAGIC = 0x46554747
TYPE_NAMES = {0:"F32",1:"F16",8:"Q8_0",12:"Q4_K",13:"Q5_K",14:"Q6_K"}

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

path = sys.argv[1] if len(sys.argv) > 1 else "/Users/ys/models/elyza-gguf/Llama-3-ELYZA-JP-8B-q4_k_m.gguf"
with open(path, 'rb') as f:
    struct.unpack('<I', f.read(4))
    struct.unpack('<I', f.read(4))
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_meta = struct.unpack('<Q', f.read(8))[0]
    for _ in range(n_meta):
        read_string(f); vtype = struct.unpack('<I', f.read(4))[0]; skip_meta_value(f, vtype)
    for _ in range(n_tensors):
        name = read_string(f)
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        qtype = struct.unpack('<I', f.read(4))[0]
        struct.unpack('<Q', f.read(8))
        # Show layers 28-31 and any non-standard types
        if any(f"blk.{i}" in name for i in range(28,32)) or qtype not in (0, 12, 14):
            print(f"  {name}: dims={dims}, qtype={TYPE_NAMES.get(qtype, f'UNKNOWN({qtype})')}")
