# 可复用性能诊断工具

本目录只保留**通用、可复用**的带宽、步级 Roofline、冷编译与前端开销诊断工具。一次性 PR 调试探针与下游模型（Diffusers / SAM2 / 视觉等）专属验收脚本不再留存本目录；端到端对拍请使用 [`benchmarks/torch_compat/`](../torch_compat/README.md)，下游生态库验收请到 [JittorEcosystem](https://github.com/JittorEcosystem) 对应仓库执行。

`_paths.py` 统一提供源码定位与外部输出目录校验（只依赖标准库）；`run_perf_env.sh` 提供隔离缓存的运行包装。

| 工具 | 用途 | 设备 / 前端 |
| --- | --- | --- |
| `benchmark_npu_frontends.py` | 同尺寸矩阵乘的原生 / 独立 compat 前端延迟与零回退校验 | ACL NPU |
| `profile_single_kernel.py` | 合成 C++ kernel 的编译、链接与加载各阶段冷启动耗时剖析 | CPU / CUDA |
| `roofline/roofline_copy.py` | 设备内存 copy 带宽上界采样（Roofline 基准） | CUDA / NPU |
| `roofline/profile_step.py` | Jittor 训练/推理单步耗时与 Roofline 采样 | CUDA / NPU |
| `roofline/profile_step_torch.py` | 独立 PyTorch 单步耗时与 Roofline 对照采样 | CUDA / NPU |
| `roofline/classify.py` | 将 Roofline 采样结果归类为 compute-bound / memory-bound / launch-bound | 离线分析 |

## 运行环境

从仓库根目录执行。为每个任务指定独立 `cache_name`，首次 JIT 串行预热，不与单测共享缓存：

```bash
cache_name=cold-compile CUDA_VISIBLE_DEVICES="$JITTOR_GPU" \
  benchmarks/profiling/run_perf_env.sh python \
  benchmarks/profiling/profile_single_kernel.py
```

`run_perf_env.sh` 保留用户 `HOME`，将 JIT、临时文件与结果放到 `$JITTOR_LAB_ROOT/_state/profiling/<cache_name>/`；需要指定特定 CUDA SDK 时设置 `CUDA_HOME` 或 `JTCUDA`。

NPU 前端对比先按[硬件指南](../../docs/guides/hardware/ascend-910b.md)加载 CANN：

```bash
export JITTOR_LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}"
run_dir="$JITTOR_LAB_ROOT/_state/profiling/npu-matmul"
JITTOR_HOME="$run_dir/native-cache" python benchmarks/profiling/benchmark_npu_frontends.py \
  --mode native --output "$run_dir/native.json"
JITTOR_HOME="$run_dir/compat-cache" python benchmarks/profiling/benchmark_npu_frontends.py \
  --mode compat --output "$run_dir/compat.json"
```

完整计时与留存规则见[性能基准规范](../../docs/performance/benchmarking.md)。
