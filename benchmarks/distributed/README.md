# Torch 兼容层分布式基准（`benchmarks/distributed/`）

本目录提供**纯标准 PyTorch（`import torch` + `torch.distributed`）**编写的分布式通信与多卡训练扩展基准，同一份脚本可直接分别在 **Jittor Torch 兼容层**（`jtrun`）与**独立原生 PyTorch**（`torchrun`）上执行并做同口径对比。

## 覆盖项

| 工作负载 | 测量内容 | 指标 |
| --- | --- | --- |
| `all_reduce` | 多尺寸（64 KB / 1 MB / 16 MB / 64 MB）`SUM` 集合归约 + 数值校验 | `median_ms`、`algbw_gb_s`、`busbw_gb_s` |
| `all_gather` | 多尺寸 `all_gather_into_tensor` 聚合 + 数值校验 | `median_ms`、`algbw_gb_s`、`busbw_gb_s` |
| `reduce_scatter` | 多尺寸 `reduce_scatter_tensor` + 数值校验 | `median_ms`、`algbw_gb_s`、`busbw_gb_s` |
| `broadcast` | 多尺寸 rank 0 广播 + 数值校验 | `median_ms`、`algbw_gb_s`、`busbw_gb_s` |
| `ddp_transformer_step` | 4 层 Decoder Transformer（RMSNorm + Causal SDPA + SwiGLU + AdamW）多卡同步训练 | `median_ms`、`comm_ms`、`comm_fraction`、`tokens_per_s` |

## 运行示例

### 多卡 CUDA（NCCL）

```bash
# Jittor Torch 兼容层（无 MPI）
CUDA_VISIBLE_DEVICES=0,1 JITTOR_TORCH_SHIM=1 \
  jtrun -n 2 --backend nccl -- \
  python benchmarks/distributed/bench_collectives.py \
  --device cuda --backend nccl --dtype bfloat16

# 原生 PyTorch 对照
CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 \
  benchmarks/distributed/bench_collectives.py \
  --device cuda --backend nccl --dtype bfloat16
```

### 多进程 CPU（Gloo，无需 MPI）

```bash
# Jittor Torch 兼容层（原生 TCP Gloo，无需 mpirun）
CUDA_VISIBLE_DEVICES="" JITTOR_TORCH_SHIM=1 \
  jtrun -n 2 --backend gloo -- \
  python benchmarks/distributed/bench_collectives.py \
  --device cpu --backend gloo --size tiny
```

查看工作负载清单（不导入框架）：

```bash
python benchmarks/distributed/bench_collectives.py --list --json
```
