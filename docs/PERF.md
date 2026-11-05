# Performance Methodology

## Roofline 参考表（理论峰值）

| GPU | SM | HBM 带宽 | FP16 TC | FP8 TC | FP4 TC |
|---|---|---|---|---|---|
| A100 80G | 80 | 2039 GB/s | 312 TFLOPS | – | – |
| A10 | 86 | 600 GB/s | 125 TFLOPS | – | – |
| L4 | 89 | 300 GB/s | 121 TFLOPS | (存储) | – |
| RTX 4090 | 89 | 1008 GB/s | 165 TFLOPS | (存储) | – |
| H100 SXM | 90 | 3350 GB/s | 990 TFLOPS | 1979 TFLOPS | – |
| B200 | 100 | 8000 GB/s | ~2250 TFLOPS | ~4500 TFLOPS | ~9000 TFLOPS |

## 基准方法

`bench/bench_kernels.cpp`：CUDA events 计时，20 次迭代取均值，warmup 一次。
逐算子输出 ms + GB/s（带宽型）或 TFLOPS（计算型），与上表对照即可得到"距峰值百分比"。

## v0.1 基线优化清单（CUDA-core 路径）

### rmsnorm / silu_and_mul / rope（带宽型，目标 ≥ 70% HBM 带宽）
- [ ] rmsnorm: uint4 向量化读写（cols%8==0 时 8×half/次）
- [ ] rmsnorm: 小 cols 用 warp-per-row，减少 block 启动开销
- [ ] rope: half2 成对处理，预计算 inv_freq 表
- [ ] 全部: `-maxrregcount` 调优 + occupancy 对比

### quant_fp8（带宽型）
- [ ] 向量化加载 + `__vimax`/warp-level absmax
- [ ] scale 用 `__fdividef` 常量折叠（×(1/448) 已做）

### w8a8 block-fp8 GEMM（计算型，v0.1 CUDA-core 预期 ~5-15% 峰值）
- [x] smem 分块 + pad、scale hoisting、边界 tail 处理
- [ ] 双缓冲（cp.async，sm_80+）隐藏全局加载
- [ ] split-K（小 M 场景，M=1..8 decode GEMM）
- [ ] sm_89+: 硬件 cvt 替换整数 fp8 解码
- [ ] sm_90: wgmma + TMA 路径（目标 >60% FP8 峰值）
- [ ] sm_100: tcgen05 + 2-CTA MMA；NVFP4 变体（GEMM + 反量化 epilogue）

### paged_attn decode（带宽型，GQA 8:1 时目标 ≥ 60%）
- [x] 分页寻址 + online softmax + GQA head 分组
- [ ] head-pair 处理提高带宽利用（2 heads/block 共享 K/V 加载）
- [x] `__expf` 快速指数
- [ ] FP8 KV cache 读取路径（sm_89+ 硬件 cvt）
- [ ] sm_90: wgmma 变体（参考 FlashMLA）

### topk_softmax（小 kernel，延迟型）
- [x] 单 block 完成 softmax + top-k
- [ ] 与 MoE align/pack 融合（roadmap: fused routing 全家桶）

## 测量纪律

1. 每次 PR 只改一个优化点，bench 输出贴进 PR 描述
2. 关闭 ECC/持久时钟（`nvidia-smi -lgc`）后再对比
3. 报告带机器型号 + 驱动 + CUDA 版本
4. 带宽型算子同时报告 GB/s 与 %峰值；计算型报告 TFLOPS 与 %峰值
