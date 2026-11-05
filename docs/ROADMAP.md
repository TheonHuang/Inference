# Roadmap

依据：2025–2026 社区公开讨论（vLLM / DeepGEMM / FlashInfer issues、HuggingFace
kernel 工程博客、NVIDIA 开发者论坛）中的算子痛点，按"热度 × 可行性"排序。

## v0.1 ✅（本版本）—— 推理逐层算子 + FP8 量化基线

- [x] rmsnorm / fused_add_rmsnorm / silu_and_mul / rope_neox
- [x] FP8 per-token-group 量化/反量化（位一致软件实现，全 SM 可用）
- [x] W8A8 blockwise FP8 GEMM（DeepSeek 配方，CUDA-core 基线，多 tile 按架构选择）
- [x] paged attention decode（GQA、block tables、head_dim 64/128/256）
- [x] MoE topk_softmax 路由
- [x] CPU 黄金参考 + standalone 测试 + CI 编译矩阵（sm_80..120）+ Docker

## v0.2 —— 带宽型算子深度优化（无 tensor core 依赖，全架构受益）

- [ ] 全部逐元素算子向量化（uint4 / half2）+ occupancy 调优
- [ ] w8a8 GEMM 双缓冲（cp.async）+ split-K 小 M 路径
- [ ] paged attn head-pair 共享 KV 加载、FP8 KV cache 读取
- [ ] bench 进 CI artifact（自托管 GPU runner 接入点已留好）

## v0.3 —— Hopper 专项（sm_90）

- [ ] w8a8 GEMM wgmma + TMA 路径（对齐 DeepGEMM 思路：UMMA + softmax-recoverable 累加）
- [ ] attention prefill kernel（split-KV，大 seq_len 吞吐）
- [ ] CUDA graph 友好接口（静态 shape 特化）

## v0.4 —— Blackwell 专项（sm_100/120）

- [ ] tcgen05 MMA 路径 + 2-CTA cluster
- [ ] NVFP4 分组 GEMM（MoE 权重压缩主战场）+ FP4→FP16 epilogue
- [ ] sm_120（DGX Spark）内存一致模型适配（cva）

## v0.5 —— 新兴算子（社区热点）

- [ ] 稀疏 attention：NSA / MoBA top-k block selection kernel
- [ ] 线性注意力：Gated DeltaNet / GLA chunk-wise kernel（训练+推理）
- [ ] 投机解码 verify kernel（tree top-k 批量验证）
- [ ] MoE 全家桶：grouped GEMM（contiguous / masked 两种布局）+ dispatch/combine 融合

## 持续

- [ ] 每算子：GB/s 与 %峰值 进 README 徽章表
- [ ] Python 绑定（nanobind）+ torch.compile custom op 注册
- [ ] CUTLASS Python DSL 对照实现（作为第二参考与教学对照）
