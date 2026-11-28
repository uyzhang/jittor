#!/usr/bin/env python3
"""Pure PyTorch distributed collective & DDP scaling benchmark.

Written strictly against standard ``import torch`` and ``torch.distributed`` so
the exact same script runs under:

* Jittor Torch compat (launched with ``jtrun``):
    CUDA_VISIBLE_DEVICES=0,1 jtrun -n 2 --backend nccl -- \\
        python benchmarks/distributed/bench_collectives.py --device cuda --backend nccl

    jtrun -n 2 --backend gloo -- \\
        python benchmarks/distributed/bench_collectives.py --device cpu --backend gloo

* Independent binary PyTorch (launched with ``torchrun``):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \\
        benchmarks/distributed/bench_collectives.py --device cuda --backend nccl

    torchrun --standalone --nproc_per_node=2 \\
        benchmarks/distributed/bench_collectives.py --device cpu --backend gloo
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time


WORKLOAD_NAMES = (
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "ddp_transformer_step",
)

DEFAULT_SIZES_BYTES = (64 * 1024, 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024)


def _sync_device(torch, device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _busbw_factor(collective, world_size):
    n = float(world_size)
    if n <= 1.0:
        return 1.0
    if collective == "all_reduce":
        return 2.0 * (n - 1.0) / n
    if collective in ("all_gather", "reduce_scatter"):
        return (n - 1.0) / n
    return 1.0


def _runtime_label(torch):
    if hasattr(torch, "_torch_compat_install_context"):
        return "jittor-torch-compat"
    return "pytorch"


def _bench_collective(torch, dist, name, nbytes, dtype, device, warmup, repeats):
    rank = dist.get_rank()
    world = dist.get_world_size()
    elem_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    numel = max(world, (nbytes // elem_bytes // world) * world)
    actual_bytes = numel * elem_bytes

    if name == "all_reduce":
        src = torch.full((numel,), float(rank + 1), device=device, dtype=dtype)
        def step():
            buf = src.clone()
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            return buf
        expected = float(world * (world + 1) // 2)
    elif name == "broadcast":
        src = torch.full((numel,), 7.0 if rank == 0 else -1.0, device=device, dtype=dtype)
        def step():
            buf = src.clone()
            dist.broadcast(buf, src=0)
            return buf
        expected = 7.0
    elif name == "all_gather":
        shard = numel // world
        local = torch.full((shard,), float(rank + 1), device=device, dtype=dtype)
        out = torch.empty((numel,), device=device, dtype=dtype)
        def step():
            dist.all_gather_into_tensor(out, local)
            return out
        expected = float(world * (world + 1)) / (2.0 * world)
    elif name == "reduce_scatter":
        shard = numel // world
        inp = torch.full((numel,), float(rank + 1), device=device, dtype=dtype)
        out = torch.empty((shard,), device=device, dtype=dtype)
        def step():
            dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)
            return out
        expected = float(world * (world + 1) // 2)
    else:
        raise ValueError("unknown collective: " + name)

    # Warmup + numerical check
    last = None
    for _ in range(warmup):
        last = step()
        _sync_device(torch, device)
    observed = float(last.float().mean().cpu().item())
    if abs(observed - expected) > max(1e-2, 1e-2 * abs(expected)):
        raise RuntimeError(
            "%s numerical mismatch on rank %d: expected %.4f, got %.4f"
            % (name, rank, expected, observed)
        )

    dist.barrier()
    _sync_device(torch, device)
    times_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = step()
        _sync_device(torch, device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    dist.barrier()

    median_ms = statistics.median(times_ms)
    algbw = (actual_bytes / 1e9) / (median_ms / 1e3) if median_ms > 0 else 0.0
    busbw = algbw * _busbw_factor(name, world)
    return {
        "workload": name,
        "bytes": actual_bytes,
        "dtype": str(dtype).replace("torch.", ""),
        "world_size": world,
        "median_ms": round(median_ms, 4),
        "min_ms": round(min(times_ms), 4),
        "algbw_gb_s": round(algbw, 3),
        "busbw_gb_s": round(busbw, 3),
    }


def _bench_ddp_transformer(torch, dist, dtype, device, size, warmup, repeats):
    nn = torch.nn
    F = nn.functional
    rank = dist.get_rank()
    world = dist.get_world_size()

    tiny = size == "tiny"
    layers = 2 if tiny else 4
    batch = 2 if tiny else 4
    seq = 32 if tiny else 512
    hidden = 128 if tiny else 1024
    heads = 4 if tiny else 16
    inter = 256 if tiny else 2816

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.heads = heads
            self.head_dim = hidden // heads
            self.norm1 = nn.RMSNorm(hidden, eps=1e-6)
            self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
            self.o_proj = nn.Linear(hidden, hidden, bias=False)
            self.norm2 = nn.RMSNorm(hidden, eps=1e-6)
            self.gate = nn.Linear(hidden, inter, bias=False)
            self.up = nn.Linear(hidden, inter, bias=False)
            self.down = nn.Linear(inter, hidden, bias=False)

        def forward(self, x):
            b, s, d = x.shape
            h = self.norm1(x)
            qkv = self.qkv(h).reshape(b, s, 3, self.heads, self.head_dim)
            q = qkv[:, :, 0].transpose(1, 2)
            k = qkv[:, :, 1].transpose(1, 2)
            v = qkv[:, :, 2].transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.o_proj(attn.transpose(1, 2).reshape(b, s, d))
            h2 = self.norm2(x)
            return x + self.down(F.silu(self.gate(h2)) * self.up(h2))

    torch.manual_seed(42)
    model = nn.Sequential(*[_Block() for _ in range(layers)]).to(device=device, dtype=dtype).train()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    torch.manual_seed(1000 + rank)
    x = torch.randn((batch, seq, hidden), device=device, dtype=dtype)

    def step():
        loss = model(x).float().pow(2).mean()
        loss.backward()
        t_comm0 = time.perf_counter()
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        _sync_device(torch, device)
        comm_ms = (time.perf_counter() - t_comm0) * 1000.0
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _sync_device(torch, device)
        return float(loss.detach().cpu().item()), comm_ms

    for _ in range(warmup):
        step()

    dist.barrier()
    step_ms_list, comm_ms_list = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _loss, comm_ms = step()
        step_ms_list.append((time.perf_counter() - t0) * 1000.0)
        comm_ms_list.append(comm_ms)
    dist.barrier()

    med_step = statistics.median(step_ms_list)
    med_comm = statistics.median(comm_ms_list)
    total_tokens = batch * seq * world
    return {
        "workload": "ddp_transformer_step",
        "size": size,
        "dtype": str(dtype).replace("torch.", ""),
        "world_size": world,
        "median_ms": round(med_step, 3),
        "comm_ms": round(med_comm, 3),
        "comm_fraction": round(med_comm / med_step, 4) if med_step > 0 else 0.0,
        "tokens_per_s": round(total_tokens / (med_step / 1e3), 1) if med_step > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu", "npu"), default="cuda")
    parser.add_argument("--backend", choices=("nccl", "gloo", "hccl"), default="nccl")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--workloads", default="all",
                        help="comma-separated subset of %s, or 'all'" % ",".join(WORKLOAD_NAMES))
    parser.add_argument("--sizes-kb", default=None,
                        help="comma-separated payload sizes in KB (default: 64,1024,16384,65536)")
    parser.add_argument("--size", choices=("full", "tiny"), default="full")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", default=None, help="optional JSON output path (rank 0)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.list:
        rows = [{"name": name} for name in WORKLOAD_NAMES]
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for row in rows:
                print(row["name"])
        return 0

    import torch
    import torch.distributed as dist

    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    if args.device == "cuda" and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank < torch.cuda.device_count():
            torch.cuda.set_device(local_rank)

    dtype = getattr(torch, args.dtype)
    if args.sizes_kb:
        sizes = [int(tok.strip()) * 1024 for tok in args.sizes_kb.split(",") if tok.strip()]
    elif args.size == "tiny":
        sizes = [4 * 1024, 64 * 1024]
    else:
        sizes = list(DEFAULT_SIZES_BYTES)

    selected = (
        list(WORKLOAD_NAMES) if args.workloads == "all"
        else [w.strip() for w in args.workloads.split(",") if w.strip()]
    )

    results = []
    for name in selected:
        if name == "ddp_transformer_step":
            results.append(_bench_ddp_transformer(
                torch, dist, dtype, args.device, args.size, args.warmup, args.repeats
            ))
        else:
            for nbytes in sizes:
                results.append(_bench_collective(
                    torch, dist, name, nbytes, dtype, args.device, args.warmup, args.repeats
                ))

    payload = {
        "runtime": _runtime_label(torch),
        "backend": args.backend,
        "device": args.device,
        "world_size": world,
        "results": results,
    }
    if rank == 0:
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("DIST_BENCH_RESULT " + json.dumps(payload), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
