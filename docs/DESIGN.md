# Design Notes

## 分层

| 层 | 位置 | 说明 |
|---|---|---|
| 公共 API | `include/inferop/ops.h` | 纯 C++，`void*` + `dtype` + opaque stream，不暴露 CUDA 类型；未来可被 Python/ffi 绑定直接使用 |
| dtype 分发 | `kernels/*.cu` 末尾的 `*_impl` | 把 `dtype` switch 到模板实例（fp16/bf16/fp32） |
| 架构选择 | kernel 内 `if (sm >= 90)` 或编译期 `__CUDA_ARCH__` | v0.1 用运行时 SM 查询选 tile 变体（如 GEMM 64x64 vs 128x64），避免每个 arch 一套代码的维护成本；特化路径（wgmma 等）会用 `#if __CUDA_ARCH__ == 900` 守卫 |
| CPU 黄金参考 | `reference/` | fp32 存储、fp64 累加（GEMM），与 kernel 逐位可比 |

## 数值正确性策略

1. **FP8 软件转换**（`include/inferop/fp8.h`）：纯整数实现，host/device 位一致，
   不依赖 sm_89+ 的硬件 cvt 指令 → A100 上也能跑 FP8 量化研究。
   已用独立 oracle（枚举 256 网格点 + RNE tie 规则）验证 20 万样本。
2. **量化算子要求逐字节一致**：GPU kernel 与 CPU 参考跑同一算法（absmax→scale→RN 编码），
   max 是精确运算、编码是整数运算 → bit-exact，测试直接 memcmp。
3. **GEMM 用 fp32 FMA 累加**：与 CPU 参考（fp64 累加）的容差来自求和顺序差异，
   实测 rtol 2e-3 内；tensor core 路径上线后按各 arch 的累加精度单独定容差。
4. **attention 用 `__expf`**（SFU 快速指数）：输出 bf16 容差 2e-2 覆盖其 ~2^-21 相对误差。

## 内存布局约定（与 vLLM/DeepSeek 生态对齐）

| 算子 | 布局 |
|---|---|
| GEMM | A[M,K] row-major（B 同为 [N,K]，即 "TN"，与 DeepGEMM 一致），scale 按 [row, k/128] / [n/128, k/128] |
| Paged KV | [num_blocks, block_size, num_kv_heads, head_dim]，block_table int32 |
| RoPE | GPT-NeoX 半维旋转（非交错），`head_dim` 内 `rotary_dim` 前缀生效 |
| MoE 路由 | logits fp32 [tokens, experts] ≤ 1024 experts |

## 每算子的 kernel 结构速记

- **rmsnorm**：block-per-row，fp32 归约，256 线程。
- **rope**：thread-per-pair，`sincosf`；每线程处理 `(t, h, i)` 与 `(t, h, i+rd/2)`。
- **quant**：block-per-group，两遍（absmax → 编码），与 CPU 参考算法逐位对齐。
- **w8a8 GEMM**：BK=GS=128（k-tile == scale group，scale 提升到外层循环），
  smem 分块 [BM][132] 字节带 4B pad 防 bank conflict，per-thread 4x4/8x4 FMA tile，
  fp8→fp32 整数解码即时进行。
- **paged attn**：block-per-(head, seq)；A 相 4 lane/key 的 warp 归约点积、
  B 相 block 一致 online softmax、C 相每线程固定 head_dim 切片累加 V（coalesced）。
- **topk_softmax**：block-per-token，共享内存概率表 + 位图去重，迭代 top-k。

## 为什么 v0.1 不直接上 tensor core

盲写 wgmma/tcgen05（无 GPU 验证）的错误率高且难以排查。v0.1 的 CUDA-core 路径
把**接口、布局、黄金参考、CI 全部打通**，tensor core 路径按 arch 逐个替换内核，
每一步都有 bit-level 参照物。这也是 FlashInfer/DeepGEMM 实际的工程路径。
