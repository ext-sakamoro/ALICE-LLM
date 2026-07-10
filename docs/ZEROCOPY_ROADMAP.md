---
name: alice-llm-zerocopy-roadmap
description: ALICE-LLM の wgpu Vulkan unified memory zero-copy 実装 (v1.0.2 candidate、D2 相当) の技術 roadmap 現状 wgpu-hal 24 の external_memory_fd 未 support 判明、upstream contribution 経路 or 他 backend への段階的移行を提案 Jetson で 7B Q4_K を動かすための工程を 3 stage で構造化
metadata: 
  node_type: memory
  type: reference
  originSessionId: f93c467d-7bb8-4825-a0a7-919a5d2de7ae
---

# ALICE-LLM wgpu Vulkan unified memory zero-copy 実装 roadmap (v1.0.2 candidate)

**Why**: 2026-07-10 セッションで [[feedback_jetson_wgpu_vulkan_memory_limit]] の問題を根本解決する D2 (wgpu-hal + `VK_KHR_external_memory_fd`) を実装しようとしたところ **wgpu-hal 24.0.4 が Linux fd import を native support していない**技術 blocker に遭遇 覚悟して着手 (A3) しても本 session では完遂不可能と判明 v1.0.1 の OOM prevention warning ([[feedback_alice_llm_oom_prevention]]) で緩和した先の、真の zero-copy 実装への段階的 roadmap を記録

## 現状の壁 (2026-07-10 時点)

### wgpu-hal 24.0.4 実装状況

```bash
# 検証コマンド (再検証時に叩く)
grep -rn "external_memory_fd\|KHR_EXTERNAL_MEMORY_FD" \
  ~/.cargo/registry/src/index.crates.io-*/wgpu-hal-24.0.4/
# → 0 hit

grep -rn "external_memory_win32" ~/.cargo/registry/src/index.crates.io-*/wgpu-hal-24.0.4/
# → 5 hit (Windows は実装済)
```

- **Linux fd import (`VK_KHR_external_memory_fd`) は unsupported**
- Windows (`VK_KHR_external_memory_win32`) は 5 箇所実装
- Comment `We don't use VK_KHR_external_memory` (device.rs:325) が upstream 方針を示す

### wgpu 24.0.5 の as_hal 経路

- `wgpu::Device::as_hal::<Vulkan>()` で `ash::Device` 取得可 (device.rs:388)
- `Device::create_buffer_from_hal::<Vulkan>()` で raw VkBuffer を wgpu Buffer にラップ可 (device.rs:281)
- ただし wgpu-hal が **extension を device creation 時に有効化していない**ため、runtime に `vkImportMemoryFdKHR` を呼んでも失敗 (Vulkan spec 準拠)

## 3 経路の評価

### 経路 1: wgpu-hal を fork/patch

**内容**: `~/ALICE-LLM/Cargo.toml` の wgpu-hal を fork patch に差し替え、`external_memory_fd` extension を有効化

**工数実測**: 3-5 日 (wgpu-hal 100+ files の全体構造把握 + Vulkan CTS レベル test + upstream compatibility 維持)

**リスク**: fork 保守負担、wgpu 25 upgrade 時に merge 地獄

**判定**: ❌ 個人 crate の maintainability に悪影響大

### 経路 2: as_hal で raw ash::Device 取って runtime に extension 呼ぶ

**内容**: wgpu-hal は extension enable せずに device 作るが、`vkGetDeviceProcAddr("vkImportMemoryFdKHR")` で function pointer だけ取って呼ぶ

**工数実測**: 1-2 日

**リスク**: Vulkan spec 違反 (extension は VkDeviceCreateInfo で enable 必須)、driver 実装依存で `nullptr` or UB (undefined behavior) crash 可能性大 Nvidia driver は fp を返すかもしれないが、Vulkan validation layers で reject される

**判定**: ❌ Undefined behavior、production 使用不可

### 経路 3: wgpu を bypass して並列 Vulkan device

**内容**: `ash` で独立 VkDevice + VkQueue 作成、weight buffer だけそこで管理、wgpu には compute pipeline だけ担当させる

**工数実測**: 5-7 日 (2 device 併存、queue sync、cross-device buffer share 実装は極めて困難)

**リスク**: 実質 ALICE-LLM v2 相当、既存アーキ全捨て

**判定**: ❌ 現行 codebase の delta が大きすぎる

## 推奨 roadmap (Stage 制)

### Stage 1: upstream contribution (推奨、~1-2 日)

**目標**: wgpu 本家に `VK_KHR_external_memory_fd` support を追加する PR を送る

**具体 work items**:
1. `wgpu-hal/src/vulkan/adapter.rs:1002-1010` (Windows external_memory_win32 の下) に fd 版を追加
2. `wgpu-hal/src/vulkan/device.rs:721` (`external_memory_image_create_info`) に fd 版 helper 追加
3. `wgpu-types` に `Features::EXTERNAL_MEMORY_FD` variant 追加
4. `wgpu-hal/src/vulkan/adapter.rs:895 get_required_extensions` で feature request 時に extension push
5. wgpu example/docs で使用法を示す

**参考実装**:
- Windows 版 (`external_memory_win32`) の diff を Linux 版 (`external_memory_fd`) に翻案
- Vulkan spec: https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_external_memory_fd.html
- ash crate 側は `ash::khr::external_memory_fd::Device::new()` extension helper 存在済

**submit 先**: https://github.com/gfx-rs/wgpu/issues (Feature request → PR)

**依存タイムライン**: upstream merge まで 2 週間 - 3 ヶ月 (wgpu チーム reviews)

### Stage 2: ALICE-LLM v1.0.2 実装 (upstream merge 後、~1 日)

**前提**: wgpu 25 (or 26) で `Features::EXTERNAL_MEMORY_FD` が使えるようになった時点

**具体 work items**:
1. `GpuEngine::new` で `Features::EXTERNAL_MEMORY_FD` を request
2. `GpuEngine::upload_weights_mmap` 新規 fn を追加
   - Signature: `pub fn upload_weights_mmap(&self, fd: RawFd, offset: u64, size: u64, rows: usize, cols: usize) -> GpuWeightBuffer`
   - 内部で `Device::as_hal::<Vulkan>()` で ash::Device 取得
   - `ash::khr::external_memory_fd::Device` extension を wrap
   - `vkImportMemoryFdKHR` + `vkCreateBuffer` + `vkBindBufferMemory` で imported memory buffer 作成
   - `create_buffer_from_hal` で wgpu Buffer 化 (unsafe)
3. `GpuModel::load_shared` で adapter が IntegratedGpu + linux + Vulkan の時に `upload_weights_mmap` を使う分岐追加
4. GGUF mmap fd の Passing: `GgufFile` に file descriptor accessor 追加
5. fallback path: fd import 失敗時に既存 `upload_weights` に戻る
6. Mac Metal / Windows / non-integrated GPU では既存経路そのまま

### Stage 3: 検証 + release (~1 日)

1. Jetson Orin Nano 8GB で Qwen2.5-Coder-7B Q4_K の weight upload 完走確認
2. 生成品質 CPU 版と bit-identical (or 極めて近似) 確認
3. memory peak が weight size × 1.0 (± 300MB overhead) に収まる実測
4. Mac Metal で regression 0 確認
5. ALICE-LLM v1.0.2 tag + CHANGELOG 更新

**成功条件**:
- Jetson Orin Nano 8GB で Qwen2.5-Coder-7B Q4_K が動く (~5-15 tok/s 期待)
- Jetson で ALICE-LLM が真の unified memory zero-copy を実現、Metal と feature parity
- ALICE-LLM が local coding LLM (Claude Code 代替) として Jetson で実用に耐える

## 中間 workaround (v1.0.1 で既に対応済)

Stage 1-3 の upstream 依存を待つ間、user は以下で緩和可能:

1. **v1.0.1 の OOM warning** ([[feedback_alice_llm_oom_prevention]]) で silent OOM Kill 回避、pre-load で peak memory 予測 + 3 案代替提示
2. **CPU inference** (`Llama3Model::from_gguf`): Jetson で 7B 動作 (2.4 tok/s、mmap zero-copy)
3. **llama.cpp Vulkan backend**: 既に Linux fd import 実装済、Jetson で 7B 動く

## Blocker 解除の signal

- wgpu 本家 issue tracker で `external_memory_fd` を search
- wgpu-hal 25.x release notes で `VK_KHR_external_memory_fd` mention 出るか watch
- wgpu 25 (or 26) upgrade 時に本 roadmap 再検証

## 判断軸

- **今すぐ Jetson で 7B が必要**: llama.cpp を並行運用、ALICE-LLM は 1.5B / CPU 経路
- **v1.0.2 として ALICE-LLM で完結したい**: Stage 1 の upstream PR に投資、~1-3 ヶ月 wait
- **skip**: v1.0.1 の warning + user 判断で運用継続、D2 は future work として塩漬け

## 関連メモ

- [[feedback_jetson_wgpu_vulkan_memory_limit]]: 元問題 (silent OOM)
- [[feedback_alice_llm_oom_prevention]]: v1.0.1 の緩和策 (PR #7)
- [[feedback_alice_llm_gpu_qwen2_bias]]: v1.0.1 の Qwen 2 bias fix (PR #6)
- ALICE-LLM PR #6 (fix/gpu-qkv-bias-qwen2)、PR #7 (feat/oom-prevention)、PR #8 (feat/qwen3-qk-norm)
- 検証機体: extoria-jetson (Jetson Orin Nano 8GB、JetPack R36.4.3、nvgpu Vulkan)
