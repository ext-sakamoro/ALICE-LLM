# MXFP4 / MXFP8 Integration Plan (Phase X.4.f / X.4.g)

**Status**: Skeleton **landed** 2026-07-24. Full matvec + SIMD + GPU
still blocked on Kimi K3 open weight release (2026-07-27) for validation,
but the format spec is public (OCP MX v1.0, Sep 2023) and reference
implementations exist (`microsoft/microxcaling`, llama.cpp/ik_llama.cpp).

**Landed 2026-07-24** (see commit + `src/gguf.rs` MXFP4 section):

- ✅ `GgmlType::Mxfp4` variant (id=39, `block_bytes()=17`,
  `elements_per_block()=32=QK_MXFP4`)
- ✅ `E2M1_DECODE_TABLE: [f32; 16]` const (bit-exact per OCP §Table 1)
- ✅ `decode_e8m0_scale(byte: u8) -> f32` (with 0xFF → NaN handling)
- ✅ `dequantize_mxfp4_block(block: &[u8]) -> [f32; 32]` (scalar, spec
  §5.5 layout: byte 0 = E8M0 scale, bytes 1..17 = packed E2M1)
- ✅ `dequantize_row_mxfp4(data, out)` — row-level dequant loop
- ✅ `MxfP4Row` / `MxfP4Matrix` structs mirroring `TernaryRow` pattern
- ✅ `MxfP4Row::dequantize()` — struct-level dequant with padding-aware
  tail handling
- ✅ Routing: `tensor_to_f32` and `dequantize_weight_row` dispatch MXFP4
  to `dequantize_row_mxfp4`
- ✅ Routing: `quantized_matvec` dispatches MXFP4 to
  `mxfp4_matvec_fallback` (correctness-first dequant + f32_matvec)
- ✅ Free function `mxfp4_matvec()` is `todo!()` fail-fast pointing to
  this document, so callers who reach the fused-kernel API prematurely
  get an explicit panic (compliant with CLAUDE.md 「仮実装完了偽装の禁止」)
- ✅ 11 unit tests: E2M1 table bit-exact + no-NaN/Inf, E8M0 endpoints +
  NaN reserved, block dequant identity/scale/signed-symmetric, row
  dequant multi-block, struct roundtrip, `GgmlType::Mxfp4` metadata,
  matvec fail-fast `#[should_panic]`
- ✅ CI-level clippy clean (0 new warnings on `src/gguf.rs`), 316 total
  tests pass, `cargo doc --lib --no-deps` clean

**Depends on**: `docs/KIMI_K3_INTEGRATION.md` (Phase X.4 sub-phases),
`docs/HEGEMONY_THESIS.md` (why MXFP4 native is enabling, not optional).

**Depends on**: `docs/KIMI_K3_INTEGRATION.md` (Phase X.4 sub-phases),
`docs/HEGEMONY_THESIS.md` (why MXFP4 native is enabling, not optional).

## Why MXFP4 support is load-bearing for Phase X.4

Kimi K3 was trained with **MXFP4 quantization-aware training from the SFT
stage onward**. This means:

1. MXFP4 weights are the **native representation**, not a post-hoc lossy
   compression. Any Q4_K_M-style dequantize-then-requantize path
   introduces accuracy loss that would not exist in the reference model.
2. The native weight file is **~594 GB** (vs the estimated 1.4 TB for
   community Q4 GGUF conversion). Fitting on H100 8× (640 GB) requires
   the native format.
3. Without MXFP4 support, ALICE-LLM's Kimi K3 path depends on community
   Q4 GGUF conversion (mradermacher / bartowski), which:
   - Doubles disk footprint (~1.4 TB)
   - Requires H200 8× (1.13 TB) or heavy CPU offload
   - Loses the QAT-native accuracy that Moonshot trained for

MXFP4 support is therefore the enabling factor for the H100 8× target
and, on the consumer side, keeps the ~594 GB stream size within reach
of a 2 TB NVMe on Mac M3 Max.

Same logic applies to MXFP8 for activations (see §MXFP8 note below).

## OCP Microscaling MXFP4 format (v1.0, Sep 2023)

### Block layout

MXFP4 groups **32 elements per block**, each encoded as a shared scale
plus 32 individual 4-bit values.

| Field | Format | Bits | Purpose |
|---|---|---|---|
| Shared scale | E8M0 (8-bit exponent, 0 mantissa) | 8 | Pure power-of-2 multiplier for all 32 elements |
| Element × 32 | E2M1 (1 sign + 2 exp + 1 mantissa) | 4 × 32 = 128 | Per-element values in [-6, 6] range after scale |
| **Total per block** | | **136 bits = 17 bytes** | Effective 4.25 bits/weight |

### E2M1 decoding table (per-element)

Element bit pattern (S EE M) — 4 bits total:

| Pattern | Sign | Exp | Mant | Decoded value |
|---|---|---|---|---|
| 0000 | + | 0 | 0 | +0.0 |
| 0001 | + | 0 | 1 | +0.5 |
| 0010 | + | 1 | 0 | +1.0 |
| 0011 | + | 1 | 1 | +1.5 |
| 0100 | + | 2 | 0 | +2.0 |
| 0101 | + | 2 | 1 | +3.0 |
| 0110 | + | 3 | 0 | +4.0 |
| 0111 | + | 3 | 1 | +6.0 |
| 1000 | − | 0 | 0 | −0.0 |
| ... | | | | (mirror of positive) |
| 1111 | − | 3 | 1 | −6.0 |

**Note**: E2M1 has no NaN or Inf representation; it uses all 16 patterns
for finite values with subnormal 0.5 and the largest finite ±6.0.

### E8M0 scale decoding

E8M0 stores an 8-bit unsigned exponent representing 2^(x - 127), giving
a scale range of 2^(-127) to 2^127. The value 0xFF is reserved for NaN.

### Full dequantization formula (per block)

For each of the 32 elements in a block:

```
scale = 2^(E8M0_scale - 127)
elem_f32 = decode_E2M1(elem_bits) * scale
```

**Total ops per block dequant**: 1 scale conversion (bit shift + subtract)
+ 32 table lookups + 32 multiplies. Trivially SIMD-vectorizable.

## MXFP8 note (activations)

Kimi K3 uses **MXFP4 for weights + MXFP8 for activations**. MXFP8 has
the same 32-element block structure but uses E4M3 or E5M2 as the element
format (8 bits each). This means activation matmul must dequantize the
activation-side operand on-the-fly or maintain MXFP8-native compute
kernels.

For CPU forward (Phase X.4.f), we dequantize activations to F32 at each
layer boundary. GPU forward (Phase X.4.g) needs MXFP8-native compute
kernels for full efficiency, but a first pass can also dequantize to F32
in shader before matmul.

## Reference implementations (do not reinvent)

| Implementation | Language | Status | Reuse strategy |
|---|---|---|---|
| **`microsoft/microxcaling`** | PyTorch | Public reference, follows OCP spec strictly | Use for numerical validation oracle (dequant one block, compare to ALICE-LLM byte-for-byte) |
| **llama.cpp (ggml)** | C++ | MXFP4 native handling landed 2025-11 across CUDA/Vulkan/Metal/CPU | **Primary port source**. Study ggml MXFP4 dequant kernel + memory layout, translate to Rust idioms |
| **ik_llama.cpp fork** | C++ | More advanced MXFP variants (MXFP6, NVFP4 experiments) | Reference for future Phase X.11+ extensions (Hy3 FP8, LongCat variants) |
| **OCP spec PDF** | English | Authoritative | Read once, keep as reference; do not skim |

## ALICE-LLM integration approach

### Placement in existing quant enum

Add to `src/gguf.rs` `QuantType` enum:

```rust
pub enum QuantType {
    // ... existing variants (F32, F16, Q4_0, Q4_K_M, Q5_K_M, Q8_0, Ternary, IQ4_XS, ...)
    Mxfp4,   // OCP MX v1.0, 32-element blocks, E2M1 elem + E8M0 scale, 17 bytes/block
    Mxfp8E4M3, // OCP MX v1.0, 32-element blocks, E4M3 elem + E8M0 scale, 33 bytes/block
    Mxfp8E5M2, // Same but E5M2 element format
}
```

### Files modified (Phase X.4.f, CPU-side) — 2026-07-24 landing

| File | Change | Status |
|---|---|---|
| `src/gguf.rs` | `GgmlType::Mxfp4` variant + `block_bytes()=17` + `elements_per_block()=QK_MXFP4=32` accessor updates + `QK_MXFP4` const | ✅ Landed |
| `src/gguf.rs` | `MxfP4Row` / `MxfP4Matrix` structs mirroring `TernaryRow` pattern + `n_blocks()` + `dequantize()` method | ✅ Landed |
| `src/gguf.rs` | Fused `mxfp4_matvec()` optimized kernel | 🚧 `todo!()` fail-fast (Phase X.4.f matvec) |
| `src/gguf.rs` | Correctness-first `mxfp4_matvec_fallback()` (dequant + f32_matvec) for `quantized_matvec` routing | ✅ Landed |
| `src/gguf.rs` | Block dequant primitive `dequantize_mxfp4_block` + `dequantize_row_mxfp4` + `E2M1_DECODE_TABLE` (16-entry const) + `decode_e8m0_scale` | ✅ Landed |
| `src/gguf.rs` | Route `GgmlType::Mxfp4` in `tensor_to_f32` + `dequantize_weight_row` + `quantized_matvec` | ✅ Landed |
| `src/gguf.rs` | Unit tests: 11 total (E2M1 table + E8M0 endpoints + block dequant × 3 variants + row dequant + struct roundtrip + GgmlType metadata + matvec fail-fast should_panic) | ✅ Landed |
| `src/gguf.rs` | SIMD variants (NEON / AVX2 / AVX-512) for fused matvec | 🚧 `todo!()` fail-fast (deferred until scalar fused kernel + weight validation) |
| `tests/mxfp4_dequant.rs` (integration) | Block dequant byte-exact vs `microxcaling` PyTorch oracle | 🚧 Pending weight release + Python oracle setup |

**Landed 2026-07-24**: ~250 LOC production code + ~180 LOC tests = ~430 LOC in `src/gguf.rs`.
**Remaining**: fused matvec kernel + SIMD paths + PyTorch oracle validation (~2-3 days once weights land).

### Files to modify (Phase X.4.g, GPU-side)

| File | Change | LOC estimate |
|---|---|---|
| `src/shaders/dequant_mxfp4.wgsl` | New: WGSL block dequant kernel, 32-element workgroup | +80-120 |
| `src/shaders/matvec_mxfp4.wgsl` | New: fused dequant+matvec, workgroup per output row | +150-200 |
| `src/gpu.rs` | Pipeline registration + `MatvecMxfp4Pipeline` struct | +100-150 |
| `src/gpu.rs` | Buffer layout for MXFP4 weights (host→GPU upload path) | +60-100 |
| `src/shaders/dequant_mxfp8_e4m3.wgsl` | Same pattern for MXFP8 activations | +80-120 |
| `tests/gpu_mxfp4_parity.rs` | GPU parity vs CPU MXFP4 vs F32 baseline | +100 |

**Total GPU-side**: ~600-900 LOC, estimated 7-10 days.

## Numerical validation strategy

Mirror the methodology used in Phase X.3.e.3.30 V2-Lite validation
(see `docs/DEEPSEEK_V2_LITE_VALIDATION.md`):

### Step 1: Block-level dequant byte-exact

Take one arbitrary MXFP4 block from a real Kimi K3 GGUF tensor (post
2026-07-27). Dequantize it with `microxcaling` PyTorch reference and
with ALICE-LLM CPU implementation. Assert exact match at F32 level.

### Step 2: Full-tensor dequant statistics

Dequantize all weights of one layer's `attn_q_proj` in both
implementations. Compute:
- `max_abs_diff`
- `mean_abs_diff`
- `relative_error_p99`

For a correct implementation these are all zero (byte-exact); any
non-zero value indicates a bit-manipulation bug.

### Step 3: Matvec parity

Compare `mxfp4_matvec(W, x)` vs `f32_matvec(dequant(W), x)` for random
`x` vectors. Should be exactly equal because both dequantize to F32
before the multiply-accumulate. Any non-zero delta indicates a
matvec-side accumulation bug (order-of-operations, SIMD reduction, etc.).

### Step 4: End-to-end forward

Once Phase X.4.b (weight loading) + X.4.c (CPU forward) land with MXFP4
weights, compare full 27-layer forward pass output vs HF Mac-mainlined
oracle for a fixed prompt. Same methodology as Phase X.3.e.3.30 V2-Lite.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| llama.cpp MXFP4 kernel has subtle bug ALICE-LLM inherits | Wrong outputs, hard to detect | Independent validation via `microxcaling` PyTorch oracle (Step 1) |
| GGUF tensor type numeric value for MXFP4 changes | Load fails post-release | Wait for community GGUF conversion + inspect actual byte pattern before hardcoding |
| MXFP8 activation compute needed for full efficiency (not just weights) | Slower than expected on GPU | First-pass: dequant to F32 in shader; Phase X.4.g.2 for MXFP8-native kernels if profiling shows bottleneck |
| E2M1 does not represent common weight values well → QAT-time accuracy loss | Would show up in Moonshot's own benchmarks; if their model is competitive with Claude Fable 5, E2M1 is sufficient | Trust Moonshot's QAT; validate against their reported benchmarks in Step 4 |
| Consumer NVMe cannot sustain 24 GB/token streaming | Below 0.5 tok/s target | Phase X.4.e expert LRU cache (96 GB hot residency of top-64 experts) covers >90% of forwards without disk hit |

## Success criteria (Phase X.4.f / X.4.g exit)

- [ ] `microxcaling` PyTorch dequant byte-exact match on ≥3 real Kimi K3 tensors
- [ ] CPU `mxfp4_matvec` matches F32 reference exactly for ≥100 random inputs
- [ ] SIMD paths (NEON on Mac, AVX2/AVX-512 on x86_64) match scalar path byte-exact
- [ ] GPU `matvec_mxfp4` matches CPU implementation within F32 rounding tolerance
- [ ] End-to-end Kimi K3 forward on a real prompt matches Moonshot API output at top-1 argmax
- [ ] Throughput: Mac M3 Max full MXFP4 in-memory ≥ 1 tok/s for a small prompt
- [ ] Zero regression on existing Q1_0 / Q4_K_M / Q8_0 / BitNet paths (all existing tests pass)

## Timing

| Phase | Earliest start | Blockers | Estimated landing |
|---|---|---|---|
| ✅ Skeleton (`GgmlType::Mxfp4` variant + `QK_MXFP4` const + `E2M1_DECODE_TABLE` + `decode_e8m0_scale` + `dequantize_mxfp4_block` + `dequantize_row_mxfp4` + `MxfP4Row/Matrix` + `mxfp4_matvec_fallback` correctness-first path + 11 tests) | Landed **2026-07-24** | None — spec is public | Same session |
| Scalar fused `mxfp4_matvec` (replaces `todo!()`) | 2026-07-27 (first real weights) | Weight release + PyTorch oracle | 2-3 days |
| CPU SIMD paths (NEON + AVX2 + AVX-512) | After scalar landing | — | 2-3 days |
| GPU shader + pipeline (Metal + wgpu) | 2026-07-27 (first real weights) | Weight release | 7-10 days |
| Full validation vs Moonshot API | Phase X.4.j | X.4.c + X.4.d + X.4.f complete | Same as X.4.j |

**Remaining**: ~10-15 days from 2026-07-27 for fused kernels + SIMD + GPU
integrated with the rest of Phase X.4. Fallback path already works for
correctness (dequant + f32_matvec).

## Related documents

- `docs/KIMI_K3_INTEGRATION.md` — Phase X.4 sub-phase breakdown
- `docs/HEGEMONY_THESIS.md` — Why MXFP4 native is load-bearing (not optional)
- `docs/DEEPSEEK_V2_LITE_VALIDATION.md` — Numerical validation methodology template
- `docs/PHASE_X_3_E_3_3_VALIDATION.md` — GPU numerical parity approach reference
- `src/gguf.rs:TernaryRow` — Pattern for row-level quantization struct + matvec integration
- `src/gguf.rs:ternary_matvec_impl` — SIMD dispatch pattern to mirror

## References

- OCP Microscaling Formats (MX) Specification v1.0, September 2023:
  https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- Microsoft microxcaling (reference PyTorch impl):
  https://github.com/microsoft/microxcaling
- FPRox — OCP MX Scaling Formats explainer:
  https://fprox.substack.com/p/ocp-mx-scaling-formats
- Microscaling Data Formats for Deep Learning (2023 paper):
  https://arxiv.org/pdf/2310.10537
- llama.cpp MXFP4 discussion:
  https://github.com/ggml-org/llama.cpp/discussions/22498
- InsiderLLM — FP4 in llama.cpp (NVFP4 vs MXFP4):
  https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/
- Spheron — MXFP4 GPU cloud deployment guide:
  https://www.spheron.network/blog/mxfp4-microscaling-quantization-gpu-cloud/
