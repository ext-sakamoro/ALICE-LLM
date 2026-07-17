# Bonsai / Qwen 3.6-27B GPU Forward Support Scope

**Status**: **Coherent generation on Mac Metal (Phase X.3.e.3.27, 2026-07-17)**; Jetson Orin Nano 8GB still hits the `wgpu-hal` Vulkan 2× duplication ceiling.
**Current path**:
- Mac M3 Metal: `GpuModel::load` + `qwen_gpu` example generates coherent English + LaTeX at 1.1 tok/s using the Q1_0 fused SwiGLU shader (`swiglu_fused_q1_0.wgsl`), Q5_K / Q8_0 dequant kernels, and the per-head interleaved `attn_q` de-interleave in `upload_w_bonsai_split` (Phase X.3.e.3.22-3.27).
- Jetson Orin Nano 8GB: `--hybrid` (CPU delegate MVP, Phase X.3.e.3.17) generates `"The capital of Japan is Tokyo."` at ~0.09 tok/s. Attempts to use `--hybrid-per-layer` with `attention_only_load` (Phase X.3.e.3.29) still exceed the memory budget because 3.6 GB CPU model + 3.8 GB attention-only GPU × 2 duplication = 11+ GB; llama.cpp Vulkan with unified-memory zero-copy is the recommended path until `wgpu-hal` upstream ships zero-copy on Vulkan.

Historical note: earlier versions of `GpuModel::load` panicked on Bonsai GGUFs. That block has been removed as of Phase X.3.e.3.23; the loader now flows through the same DeltaNet + Attention path used by Qwen 3.5-4B and reports layout details up front instead of failing early.

## Why is CPU-only right now

Bonsai 27B departs from the standard Qwen 3.5 tensor layout that `GpuModel`
was built for:

| Tensor | Standard Qwen 3.5 | Bonsai 27B / Qwen 3.6-27B |
|--------|-------------------|----------------------------|
| Full-attention QKV | `blk.N.attn_q.weight [q_dim, hidden]` | `blk.N.attn_q.weight [q_dim * 2, hidden]` — Q + swish gate |
| DeltaNet in-projection | `blk.N.ssm_in.weight [qkv+z, hidden]` fused | `blk.N.attn_qkv.weight [qkv, hidden]` + `blk.N.attn_gate.weight [z, hidden]` split |
| DeltaNet SSM discretization | (missing) | `blk.N.ssm_a.weight [v_dim]` + `blk.N.ssm_dt_bias.weight [v_dim]` |
| DeltaNet per-V-head norm | (missing) | `blk.N.ssm_norm.weight [v_dim]` |

These tensors drive additional forward-path math that only exists in the CPU
implementation today:

- **Gated attention** (full-attention layers): `attn_out *= silu(gate)` after
  the standard scaled-dot-product attention output projection.
- **DeltaNet SSM discretization** (Phase X.3.e.3 Gap B): `decay = exp(softplus(alpha + dt_bias) * ssm_a)` before delta-rule integration.
- **DeltaNet per-V-head RMSNorm** (Phase X.3.e.3 Gap C): `ssm_out = rms_norm(dn_delta_out, ssm_norm)` before `ssm_out_proj` matvec.
- **Beta sigmoid transformation** (Phase X.3.e.3 Gap B extra): `beta = sigmoid(beta_raw)` before delta rule.
- **Q/K L2 normalisation without silu** (Phase X.3.e.3 §Q/K L2Norm): plain
  L2-normalise Q and K instead of `silu(x) * L2Norm(x)`.
- **Silu(z) after ssm_norm** (Phase X.3.e.3 §silu(z) order): `out = build_norm_gated(rms_norm(x, w) * silu(z))` semantics.

## Implementation Scope for GPU Support

Rough size estimate: **5-10 person-days** of shader + Rust work.

### 1. `GpuModel::load` — Bonsai arch detection + tensor loading (~1 day)

- Detect `attn_qkv` / `attn_gate` presence and switch loader path
- Load `attn_qkv [q_dim*2, hidden]` and split into two `MatvecBG` (Q half + gate half)
- Load `ssm_a` / `ssm_dt_bias` / `ssm_norm` as `GpuBuffer` and wire into `DeltaNetLayerBGs`
- Handle both full-attention layers (Q + gate) and DeltaNet layers (Q + K + V + Z split)

### 2. Full-attention gated attention shader / pipeline (~1 day)

- Extend attention path to compute `silu(gate)` and multiply into `attn_out`
- Option A: new fused shader `attention_gated.wgsl`
- Option B: reuse existing `attention` shader + separate `silu_mul` shader post-attention
- Prefer B for incremental landing; A for fewer dispatches

### 3. DeltaNet SSM refinement WGSL shaders (~2-3 days)

- **`ssm_discretisation.wgsl`**: `decay = exp(softplus(alpha + dt_bias) * ssm_a)` — per-V-head
- **`ssm_norm_per_head.wgsl`**: per-V-head RMSNorm on `dn_delta_out`
- **`beta_sigmoid.wgsl`**: `beta = sigmoid(beta_raw)` (or fold into existing `beta_proj` pipeline as post-processing)
- **Update `gated_deltanet.wgsl`**: accept decay + normalised state, skip silu(q) / silu(k), skip out *= silu(z)
- **Update `residual_add.wgsl` sequence**: insert `silu(z)` multiply between `ssm_norm` and `ssm_out_proj`

### 4. Wire DeltaNet forward path (`encode_forward_impl`) (~1 day)

- Dispatch new pipelines in correct order:
  1. RMSNorm → norm_buf
  2. attn_qkv → q_buf/k_buf/v_buf (or fused ssm_in for legacy Qwen 3.5)
  3. attn_gate → z_buf (Bonsai only)
  4. alpha_proj + beta_proj
  5. ssm_discretisation (Bonsai only)
  6. beta_sigmoid (Bonsai only)
  7. conv1d + post-conv1d silu (Bonsai only)
  8. gated_deltanet (with is_bonsai_path flag propagated via uniform)
  9. ssm_norm per V head (Bonsai only)
  10. silu(z) multiply (Bonsai only)
  11. ssm_out projection
  12. residual add

### 5. End-to-end validation (~1-2 days)

- Load Bonsai 27B Q1_0 GGUF via `GpuModel::load` on Jetson (8GB tight, may need weight sharding)
- Compare per-layer output vs CPU reference (bit-exact within FP summation noise, L2_rel < 1e-4)
- Compare final logits distribution (top-K overlap > 90% at temperature 0)
- Measure end-to-end tok/s vs CPU baseline (target: 5-10× improvement given Q1_0 GPU 8.4× vs CPU)

### 6. CI matrix additions (~0.5 day)

- Add Bonsai load smoke test to `.github/workflows/ci.yml` on both ubuntu and macos runners
- Add pipeline creation smoke test (`GpuEngine::new()` — Metal validation) — this
  would have caught the PR #73 row8×batch4 Metal regression before landing

## Related work

- **Phase X.1** — Q1_0 / Q2_0 GGUF parser (PR #59, landed)
- **Phase X.3.a-e** — DeltaNet CPU forward path (PR #61-#66, all landed)
- **Phase X.3.e.3.1-3.4** — SSM refinements (Gap A-C + §Q/K + §silu(z), commits `146ee22` / `005b3d0` / `6d43602` / `2d55f2b` / `c342f10`, all landed)
- **Phase X.5** — Bonsai 27B Jetson load-and-run demo (PR #67, landed, CPU forward)
- **Q1_0 wgpu Step 1-6** — matvec kernels (PR #70-#74, all landed, in GpuModel via `dispatch_mv` for any Q1_0 tensor)
- **Q1_0 wgpu integration** — GpuModel::dispatch_mv Q1_0 → row4 default (PR #76, landed)
- **Bonsai numerical validation** — `docs/PHASE_X_3_E_3_3_VALIDATION.md` (blocked on Mac disk space for GGUF DL)

## Priorities

Given the CPU forward path already runs Bonsai end-to-end on Jetson 8GB at
~10 s / token, GPU forward is a **speedup effort, not a correctness blocker**.
Priority ordering:

1. **(This effort)** — Fail-fast error at `GpuModel::load` so users don't hit
   a random `.unwrap()` panic. ✅ Landed in this PR.
2. **CI smoke test** — Add `GpuEngine::new()` pipeline creation smoke test to
   catch Metal regressions like PR #73 before they land. High leverage, low
   effort. (See `docs/QGPU_CI_SMOKE.md` — TBD.)
3. **Bonsai GPU forward implementation** — 5-10 days, deferred until CPU
   forward speed becomes a hard product blocker. Target: ~1 s / token on
   Jetson (matches Apple Foundation Models edge target).
