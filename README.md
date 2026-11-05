# Inference / inferop

面向多代 NVIDIA GPU 架构（Ampere / Ada / Hopper / Blackwell）的 **LLM 推理算子库**：
覆盖当前推理栈中占耗时大头、且对整体吞吐影响最直接的算子，用"一个算子 × 多代架构 × 带黄金基准测试"的方式打磨。

## 为什么是这些算子

综合主流开源推理框架与低精度计算库的公开实践、论文与业界工程趋势，
当前 LLM 推理的算子瓶颈集中在：

| 痛点 | 本仓库对应算子 | 状态 |
|---|---|---|
| DeepSeek 式 W8A8 blockwise FP8 GEMM（MoE 权重吞吐瓶颈） | `w8a8_block_fp8_gemm` | ✅ v0.1 基线（CUDA core），tensor core 路线图中 |
| FP8 per-token-group 激活量化（FP8 GEMM 的前置算子） | `quant_fp8_per_token_group` | ✅ |
| Paged attention decode（KV cache 分页注意力） | `paged_attn_decode` | ✅ GQA + block table |
| MoE 路由（softmax + top-k，路由 kernel 是 MoE 融合的基础） | `topk_softmax` | ✅ |
| 逐元素融合算子（每层必跑、纯带宽型） | `rmsnorm` / `fused_add_rmsnorm` / `silu_and_mul` / `rope_neox` | ✅ |

路线图（`docs/ROADMAP.md`）：NVFP4/FP4 分组 GEMM、sm_90 wgmma+TMA、sm_100 tcgen05、
FP8 KV cache、稀疏 attention（NSA/MoBA）、线性注意力（Gated DeltaNet/GLA chunk kernel）、
投机解码 verify kernel、CUTLASS Python DSL 对比基线。

## 架构策略

```
                ┌────────────────────────────────────────────┐
  host API      │ include/inferop/ops.h   (dtype/void* 接口) │
                ├────────────────────────────────────────────┤
  dispatch      │ kernels/*.cu  dtype 分发 + 运行时按 SM 选 tile │
                ├────────────────────────────────────────────┤
  kernels       │ 每个算子: 全架构可跑的基线 + 按代际特化路径      │
                ├────────────────────────────────────────────┤
  correctness   │ reference/*.cpp CPU 黄金实现 (fp32/fp64)      │
                │ + fp8 软件转换 (host/device 位一致, 任意 SM)   │
                └────────────────────────────────────────────┘
```

- **sm_80/86/89（A100/A10/30xx/L4/4090）**：mma-free 快速路径，所有算子可用
- **sm_90（H100/H200）**：GEMM 大 tile 变体自动启用；wgmma/TMA 路径开发中
- **sm_100/120（B200/DGX Spark）**：CUDA core 路径直接可用；tcgen05/NVFP4 路线图中
- FP8 转换为纯整数软件实现，**不依赖硬件 cvt**，从 sm_70 起全架构位一致

## 快速开始

```bash
# 纯 CPU 构建（黄金参考 + fp8 oracle 测试，无需 CUDA）
cmake -B build -DINFEROP_BUILD_CUDA=OFF -DINFEROP_BUILD_BENCH=OFF
cmake --build build -j && ctest --test-dir build --output-on-failure

# 有 nvcc 的机器：全架构编译
cmake -B build -DINFEROP_CUDA_ARCHES="80;86;89;90;100;120"
cmake --build build -j

# GPU benchmark
./build/bench_kernels
```

### Docker

```bash
docker build -t inferop -f docker/Dockerfile --build-arg INFEROP_CUDA_ARCHES="89" .
docker run --rm --gpus all inferop     # GPU 主机上跑全量测试
docker run --rm inferop                # 无 GPU 也能构建 + CPU 测试
```

## 测试模型（没有 GPU 也能开发）

每个算子一个 standalone 测试二进制（`tests/test_*.cpp`），两层结构：

1. **CPU 黄金检查**（永远运行）：独立性质验证 + 对照 CPU 参考
2. **GPU 交叉验证**（编译了 CUDA 时自动启用；无设备时运行期自动跳过）：
   kernel 输出 vs CPU 参考，按 dtype 给容差；FP8 量化要求**逐字节一致**

数值算法先经本地验证：`python scripts/check_fp8.py` 用独立 oracle（枚举 256 个
E4M3 网格点）验证 20 万随机样本 + 全部 RNE tie 中点。

CI（`.github/workflows/ci.yml`）：
- `cpu-tests`：免费 runner 上跑全部 CPU 黄金测试
- `cuda-build`：CUDA 12.6/12.8/13.0 × `{80,86,89}` / `{90,100,120}` 编译矩阵
- `format`：clang-format 检查

## 目录结构

```
include/inferop/   公共 API（ops.h）+ fp8/host 转换头
kernels/           CUDA 算子（每算子一个 .cu + dtype 分发）
reference/         CPU 黄金参考实现
tests/             standalone 测试（每算子一个可执行文件）
bench/             GPU 基准（带宽 GB/s / 吞吐 TFLOPS）
docker/            部署镜像
docs/              DESIGN.md / PERF.md / ROADMAP.md
scripts/           check_fp8.py（数值验证）/ format.sh
```

## 设计文档

- `docs/DESIGN.md` —— 分层、dispatch、多架构策略、内存布局约定
- `docs/PERF.md` —— 各代 GPU roofline 参考表、优化方法论、逐算子优化清单
- `docs/ROADMAP.md` —— 版本规划与业界趋势对照

## License

MIT
