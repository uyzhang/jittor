#!/usr/bin/env python3
"""Sweep the full 230-operator OpInfo registry through Torch compat vs PyTorch.

Discovers every operator entry declared in ``tests/opinfo/definitions/*.py`` via
AST (zero framework import during discovery) and benchmarks every PyTorch-facing
operator at realistic scale in two isolated processes -- one running independent
binary PyTorch, one running Jittor's Torch compat layer.

Produces:
  * Per-domain summary (geometric mean & median ratio)
  * Top 20 slowest operators relative to PyTorch
  * Top 15 fastest operators relative to PyTorch
  * Numerical agreement check per operator
  * ``opinfo_results.json`` and ``opinfo_report.md``

Examples::

    # List all 230 OpInfo operators without importing any framework
    python benchmarks/torch_compat/bench_opinfo.py --list --json

    # Full 230-operator RTX 4090 sweep (medium scale, ~1-2 min)
    python benchmarks/torch_compat/bench_opinfo.py \\
        --torch-python /path/to/torch/bin/python \\
        --jittor-python /path/to/jittor/bin/python \\
        --device cuda --gpu 0 --scale medium

    # Focused domain sweep
    python benchmarks/torch_compat/bench_opinfo.py \\
        --domains core_ops,nn_conv_pool,nn_loss,ordering_ops \\
        --device cuda --gpu 0
"""

import argparse
import ast
import datetime
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OPINFO_DIR = REPO / "tests" / "opinfo" / "definitions"

# Jittor-native-only primitives with no public PyTorch counterpart.
NATIVE_ONLY_OPS = frozenset(("reindex", "reindex_reduce"))

JITTOR_VARIABLES = (
    "JITTOR_SOURCE_ROOT", "JITTOR_HOME", "JITTOR_TORCH_CACHE_ROOT",
    "JITTOR_TORCH_SHIM", "JITTOR_TORCH_KEEP_HOME", "JITTOR_TORCH_ISOLATE_HOME",
    "JITTOR_SHARED_CACHE", "JT_BACKEND", "JT_USE_CUDA",
    "use_cuda", "cache_name", "JT_BUILD_NVCC_PATH", "nvcc_path",
)


def discover_opinfo_inventory(root=REPO):
    """Read every OpInfo entry from ``tests/opinfo/definitions/*.py`` via AST."""
    defn_dir = root / "tests" / "opinfo" / "definitions"
    rows = []
    for path in sorted(defn_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        domain = path.stem
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            kind = node.func.id
            if kind not in ("OpInfo", "UnaryUfuncInfo", "BinaryUfuncInfo", "ReductionOpInfo"):
                continue
            if not (node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                continue
            name = node.args[0].value
            variant = ""
            for kw in node.keywords:
                if kw.arg == "variant_test_name" and isinstance(kw.value, ast.Constant):
                    variant = str(kw.value.value)
            full_name = f"{name}:{variant}" if variant else name
            rows.append({
                "name": full_name,
                "op": name,
                "variant": variant,
                "domain": domain,
                "kind": kind,
            })
    return rows


# ---------------------------------------------------------------------------
# Scale presets & pure-PyTorch operator workload builder
# ---------------------------------------------------------------------------

def _scale_params(scale):
    if scale == "tiny":
        return dict(vec=4096, mat=64, batch=4, seq=32, dim=64,
                    ch=16, hw=16, d3=8, vocab=256, linalg_b=2, linalg_n=16)
    if scale == "large":
        return dict(vec=4 * 1024 * 1024, mat=512, batch=16, seq=512, dim=512,
                    ch=128, hw=64, d3=16, vocab=16384, linalg_b=8, linalg_n=128)
    # Default "medium": fast enough to sweep 228 ops in ~1-2 min while large
    # enough to expose real GPU kernel + host overhead differences.
    return dict(vec=1024 * 1024, mat=256, batch=8, seq=128, dim=256,
                ch=64, hw=32, d3=12, vocab=4096, linalg_b=4, linalg_n=64)


def _extract_primary_tensor(torch, out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for item in out:
            found = _extract_primary_tensor(torch, item)
            if found is not None:
                return found
    return None


def _build_op_runner(torch, np, entry, device, dtype, scale):
    """Return ``(callable_fn, shape_desc)`` using strictly public PyTorch APIs."""
    F = torch.nn.functional
    name = entry["op"]
    variant = entry["variant"]
    domain = entry["domain"]
    full = entry["name"]
    p = _scale_params(scale)
    rng = np.random.default_rng(20261128)

    if name in NATIVE_ONLY_OPS:
        return None, "native-only Jittor op (no public torch counterpart)"

    def t_rand(shape, low=-1.0, high=1.0, dt=None):
        arr = rng.uniform(low, high, size=shape).astype(np.float32)
        return torch.tensor(arr, device=device).to(dt or dtype)

    def t_pos(shape, low=0.25, high=2.5, dt=None):
        return t_rand(shape, low=low, high=high, dt=dt)

    def t_int(shape, low=0, high=100, dt=None):
        arr = rng.integers(low, high, size=shape).astype(np.int64)
        return torch.tensor(arr, device=device, dtype=dt or torch.int64)

    def t_bool(shape):
        arr = rng.integers(0, 2, size=shape).astype(np.bool_)
        return torch.tensor(arr, device=device, dtype=torch.bool)

    # 1. Unary elementwise & special functions
    _UNIT_DOMAIN = {"asin", "acos", "atanh", "erfinv"}
    _POS_DOMAIN = {"log", "log2", "log1p", "sqrt", "rsqrt", "reciprocal", "lgamma", "digamma"}
    _ACOSH_DOMAIN = {"acosh"}
    if domain in ("pointwise_unary", "special_unary") or name in (
        "exp", "log", "sin", "cos", "tanh", "sqrt", "abs", "negative",
        "sigmoid", "relu", "gelu", "silu",
    ):
        shape = (p["vec"],)
        if name in _UNIT_DOMAIN:
            x = t_rand(shape, -0.8, 0.8)
        elif name in _POS_DOMAIN:
            x = t_pos(shape, 0.25, 3.0)
        elif name in _ACOSH_DOMAIN:
            x = t_pos(shape, 1.2, 4.0)
        else:
            x = t_rand(shape, -2.0, 2.0)

        if name == "prelu":
            w = t_pos((1,), 0.1, 0.3)
            return (lambda: F.prelu(x, w)), str(shape)
        if name == "leaky_relu":
            return (lambda: F.leaky_relu(x, 0.1)), str(shape)
        if name == "elu":
            return (lambda: F.elu(x, 1.0)), str(shape)
        if name == "celu":
            return (lambda: F.celu(x, 1.0)), str(shape)
        if name == "softplus":
            return (lambda: F.softplus(x)), str(shape)
        if name in ("relu", "gelu", "silu", "relu6", "hardswish", "hardsigmoid",
                    "selu", "mish", "tanhshrink", "softsign"):
            fn = getattr(F, name)
            return (lambda f=fn: f(x)), str(shape)
        if name == "negative":
            return (lambda: torch.neg(x)), str(shape)
        fn = getattr(torch, name, None) or getattr(torch.special, name, None)
        if fn is None:
            return None, f"torch.{name} unavailable"
        return (lambda f=fn: f(x)), str(shape)

    # 2. Extra NN activations
    if domain == "nn_activations_extra":
        shape = (p["batch"], p["seq"], p["dim"])
        x = t_rand(shape, -2.0, 2.0)
        if name == "log_sigmoid":
            return (lambda: F.logsigmoid(x)), str(shape)
        if name == "hardtanh":
            return (lambda: F.hardtanh(x)), str(shape)
        if name == "glu":
            return (lambda: F.glu(x, dim=-1)), str(shape)
        if name == "normalize":
            return (lambda: F.normalize(x, p=2.0, dim=-1)), str(shape)
        if name == "cosine_similarity":
            y = t_rand(shape, -2.0, 2.0)
            return (lambda: F.cosine_similarity(x, y, dim=-1)), str(shape)

    # 3. Binary elementwise
    if domain == "pointwise_binary" or name in ("add", "sub", "mul", "div", "maximum", "minimum"):
        shape = (p["vec"],)
        if name in ("logical_and", "logical_or"):
            a, b = t_bool(shape), t_bool(shape)
            fn = getattr(torch, name)
            return (lambda f=fn: f(a, b)), str(shape)
        if name == "clamp":
            a = t_rand(shape, -3.0, 3.0)
            return (lambda: torch.clamp(a, -1.0, 1.0)), str(shape)
        if name == "pow":
            a, b = t_pos(shape, 0.5, 2.5), t_rand(shape, -1.5, 2.0)
        elif name in ("div", "fmod", "remainder", "floor_divide"):
            a, b = t_rand(shape, -4.0, 4.0), t_pos(shape, 0.5, 3.0)
        else:
            a, b = t_rand(shape, -2.0, 2.0), t_rand(shape, -2.0, 2.0)
        if name == "not_equal":
            return (lambda: torch.ne(a, b)), str(shape)
        if name == "equal":
            return (lambda: torch.eq(a, b)), str(shape)
        fn = getattr(torch, name)
        return (lambda f=fn: f(a, b)), str(shape)

    # 4. Bitwise & predicates
    if domain == "bitwise_ops":
        shape = (p["vec"],)
        if name == "logical_not":
            a = t_bool(shape)
            return (lambda: torch.logical_not(a)), str(shape)
        if name == "logical_xor":
            a, b = t_bool(shape), t_bool(shape)
            return (lambda: torch.logical_xor(a, b)), str(shape)
        a = t_int(shape, 0, 1024, dt=torch.int32)
        if name == "bitwise_not":
            return (lambda: torch.bitwise_not(a)), str(shape)
        if name in ("left_shift", "right_shift"):
            s = t_int(shape, 1, 5, dt=torch.int32)
            fn = getattr(torch, "bitwise_" + name)
            return (lambda f=fn: f(a, s)), str(shape)
        b = t_int(shape, 0, 1024, dt=torch.int32)
        fn = getattr(torch, name)
        return (lambda f=fn: f(a, b)), str(shape)

    if domain == "predicates":
        shape = (p["vec"],)
        x = t_rand(shape, -2.0, 2.0)
        fn = getattr(torch, name)
        return (lambda f=fn: f(x)), str(shape)

    # 5. Reductions
    if domain in ("reductions_extra", "reduce_dtypes") or name in ("sum", "mean", "prod"):
        shape = (p["batch"] * 4, p["seq"], p["dim"])
        if variant == "int_reduce":
            x = t_int(shape, 1, 5, dt=torch.int32)
            if name == "prod":
                x = t_int((p["batch"] * 4, p["seq"], 8), 1, 3, dt=torch.int32)
            fn = getattr(torch, name)
            return (lambda f=fn: f(x, dim=-1)), str(tuple(x.shape))
        if variant == "bool_reduce" or name in ("all", "any"):
            x = t_bool(shape)
            fn = getattr(torch, name)
            return (lambda f=fn: f(x, dim=-1)), str(shape)
        x = t_pos(shape, 0.9, 1.1) if name in ("prod", "cumprod") else t_rand(shape, -2.0, 2.0)
        if name == "norm":
            return (lambda: torch.linalg.vector_norm(x, ord=2, dim=-1)), str(shape)
        if name in ("max", "min"):
            fn = getattr(torch, name)
            return (lambda f=fn: f(x, dim=-1).values), str(shape)
        if name in ("cumsum", "cumprod"):
            x_small = x[:, :, :64].contiguous()
            fn = getattr(torch, name)
            return (lambda f=fn: f(x_small, dim=-1)), str(tuple(x_small.shape))
        if name == "count_nonzero":
            return (lambda: torch.count_nonzero(x > 0, dim=-1)), str(shape)
        fn = getattr(torch, name)
        return (lambda f=fn: f(x, dim=-1)), str(shape)

    # 6. Matmul / Softmax / Norms / Attention
    if name == "matmul":
        a = t_rand((p["batch"], p["mat"], p["mat"]))
        b = t_rand((p["batch"], p["mat"], p["mat"]))
        return (lambda: torch.matmul(a, b)), f"({p['batch']},{p['mat']},{p['mat']})"
    if name == "bmm":
        a = t_rand((p["batch"], p["mat"], p["mat"]))
        b = t_rand((p["batch"], p["mat"], p["mat"]))
        return (lambda: torch.bmm(a, b)), f"({p['batch']},{p['mat']},{p['mat']})"
    if name in ("softmax", "log_softmax"):
        shape = (p["batch"], p["seq"], p["dim"] * 2)
        x = t_rand(shape)
        fn = getattr(F, name)
        return (lambda f=fn: f(x, dim=-1)), str(shape)
    if name == "layer_norm":
        shape = (p["batch"], p["seq"], p["dim"])
        x = t_rand(shape)
        w = torch.ones((p["dim"],), device=device, dtype=dtype)
        b = torch.zeros((p["dim"],), device=device, dtype=dtype)
        return (lambda: F.layer_norm(x, (p["dim"],), w, b, 1e-5)), str(shape)
    if name == "rms_norm":
        shape = (p["batch"], p["seq"], p["dim"])
        x = t_rand(shape)
        w = torch.ones((p["dim"],), device=device, dtype=dtype)
        return (lambda: F.rms_norm(x, (p["dim"],), w, 1e-6)), str(shape)
    if name == "group_norm":
        shape = (p["batch"], p["ch"], p["hw"], p["hw"])
        g = max(1, min(8, p["ch"]))
        x = t_rand(shape)
        w = torch.ones((p["ch"],), device=device, dtype=dtype)
        b = torch.zeros((p["ch"],), device=device, dtype=dtype)
        return (lambda: F.group_norm(x, g, w, b, 1e-5)), str(shape)
    if name == "scaled_dot_product_attention":
        heads = 4 if scale == "tiny" else 8
        hd = 32 if scale == "tiny" else 64
        shape = (p["batch"], heads, p["seq"], hd)
        q, k, v = t_rand(shape), t_rand(shape), t_rand(shape)
        causal = variant == "causal"
        return (lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal)), str(shape)

    # 7. Conv & Pooling
    if name == "conv1d":
        x = t_rand((p["batch"], p["ch"], p["seq"]))
        w = t_rand((p["ch"], p["ch"], 3), -0.1, 0.1)
        return (lambda: F.conv1d(x, w, padding=1)), str(tuple(x.shape))
    if name == "conv2d":
        x = t_rand((p["batch"], p["ch"], p["hw"], p["hw"]))
        w = t_rand((p["ch"], p["ch"], 3, 3), -0.1, 0.1)
        return (lambda: F.conv2d(x, w, padding=1)), str(tuple(x.shape))
    if name == "conv_transpose2d":
        x = t_rand((p["batch"], p["ch"], p["hw"], p["hw"]))
        w = t_rand((p["ch"], p["ch"], 4, 4), -0.1, 0.1)
        return (lambda: F.conv_transpose2d(x, w, stride=2, padding=1)), str(tuple(x.shape))
    if name == "conv3d":
        c3 = max(8, p["ch"] // 2)
        x = t_rand((max(1, p["batch"] // 2), c3, p["d3"], p["d3"], p["d3"]))
        w = t_rand((c3, c3, 3, 3, 3), -0.1, 0.1)
        return (lambda: F.conv3d(x, w, padding=1)), str(tuple(x.shape))
    if name in ("max_pool1d", "avg_pool1d"):
        x = t_rand((p["batch"], p["ch"], p["seq"]))
        fn = getattr(F, name)
        return (lambda f=fn: f(x, kernel_size=2, stride=2)), str(tuple(x.shape))
    if name in ("max_pool2d", "avg_pool2d"):
        x = t_rand((p["batch"], p["ch"], p["hw"], p["hw"]))
        fn = getattr(F, name)
        return (lambda f=fn: f(x, kernel_size=2, stride=2)), str(tuple(x.shape))
    if name == "adaptive_avg_pool2d":
        x = t_rand((p["batch"], p["ch"], p["hw"], p["hw"]))
        return (lambda: F.adaptive_avg_pool2d(x, (4, 4))), str(tuple(x.shape))
    if name in ("max_pool3d", "avg_pool3d"):
        x = t_rand((max(1, p["batch"] // 2), max(8, p["ch"] // 2), p["d3"], p["d3"], p["d3"]))
        fn = getattr(F, name)
        return (lambda f=fn: f(x, kernel_size=2, stride=2)), str(tuple(x.shape))

    # 8. Embedding & Losses
    if name == "embedding":
        w = t_rand((p["vocab"], p["dim"]), -0.2, 0.2)
        idx = t_int((p["batch"], p["seq"]), 0, p["vocab"])
        if variant == "max_norm":
            return (lambda: F.embedding(idx, w, max_norm=1.0)), str(tuple(idx.shape))
        if variant == "padding_idx":
            return (lambda: F.embedding(idx, w, padding_idx=0)), str(tuple(idx.shape))
        return (lambda: F.embedding(idx, w)), str(tuple(idx.shape))
    if domain == "nn_loss":
        if name in ("cross_entropy", "nll_loss"):
            n = p["batch"] * p["seq"]
            logits = t_rand((n, p["vocab"]))
            if name == "nll_loss":
                logits = F.log_softmax(logits, dim=-1)
            target = t_int((n,), 0, p["vocab"])
            fn = getattr(F, name)
            return (lambda f=fn: f(logits, target)), f"({n},{p['vocab']})"
        shape = (p["batch"] * p["seq"], p["dim"])
        if name == "binary_cross_entropy":
            a = t_pos(shape, 0.05, 0.95)
            b = t_pos(shape, 0.0, 1.0)
            return (lambda: F.binary_cross_entropy(a, b)), str(shape)
        if name == "binary_cross_entropy_with_logits":
            a = t_rand(shape, -2.0, 2.0)
            b = t_pos(shape, 0.0, 1.0)
            return (lambda: F.binary_cross_entropy_with_logits(a, b)), str(shape)
        if name == "kl_div":
            a = F.log_softmax(t_rand(shape), dim=-1)
            b = F.softmax(t_rand(shape), dim=-1)
            return (lambda: F.kl_div(a, b, reduction="batchmean")), str(shape)
        a, b = t_rand(shape), t_rand(shape)
        fn = getattr(F, name)
        return (lambda f=fn: f(a, b)), str(shape)

    # 9. Interpolate, Grid Sample, Pad
    if name == "interpolate":
        x = t_rand((p["batch"], p["ch"], p["hw"], p["hw"]))
        mode = variant or "bilinear"
        kw = {"align_corners": False} if mode == "bilinear" else {}
        return (lambda: F.interpolate(x, scale_factor=2.0, mode=mode, **kw)), str(tuple(x.shape))
    if name == "grid_sample":
        x = t_rand((p["batch"], p["ch"], p["hw"], p["hw"]))
        grid = t_rand((p["batch"], p["hw"], p["hw"], 2), -1.0, 1.0)
        return (lambda: F.grid_sample(x, grid, align_corners=False)), str(tuple(x.shape))
    if name == "affine_grid":
        theta = t_rand((p["batch"], 2, 3), -0.5, 0.5)
        sz = (p["batch"], p["ch"], p["hw"], p["hw"])
        return (lambda: F.affine_grid(theta, sz, align_corners=False)), str(sz)
    if name == "pad":
        x = t_rand((p["batch"], p["ch"], p["hw"], p["hw"]))
        mode = variant or "constant"
        return (lambda: F.pad(x, (2, 2, 2, 2), mode=mode)), str(tuple(x.shape))

    # 10. Indexing & Advanced Indexing
    if domain == "indexing":
        rows, cols = p["batch"] * 8, p["dim"] * 2
        x = t_rand((rows, cols))
        if name == "gather":
            idx = t_int((rows, cols), 0, cols)
            return (lambda: torch.gather(x, 1, idx)), str(tuple(x.shape))
        if name == "take":
            idx = t_int((rows * cols // 2,), 0, rows * cols)
            return (lambda: torch.take(x, idx)), str(tuple(x.shape))
        if name in ("scatter", "scatter_add"):
            idx = t_int((rows, cols), 0, cols)
            src = t_rand((rows, cols))
            fn = getattr(torch, name)
            return (lambda f=fn: f(x, 1, idx, src)), str(tuple(x.shape))
        if name == "index_select":
            idx = t_int((cols // 2,), 0, cols)
            return (lambda: torch.index_select(x, 1, idx)), str(tuple(x.shape))
        if name == "index_add":
            idx = t_int((rows // 2,), 0, rows)
            src = t_rand((rows // 2, cols))
            return (lambda: torch.index_add(x, 0, idx, src)), str(tuple(x.shape))
        if name == "masked_select":
            mask = t_bool((rows, cols))
            return (lambda: torch.masked_select(x, mask)), str(tuple(x.shape))
        if name == "masked_fill":
            mask = t_bool((rows, cols))
            return (lambda: x.masked_fill(mask, 0.0)), str(tuple(x.shape))
        if name == "where":
            mask = t_bool((rows, cols))
            y = t_rand((rows, cols))
            return (lambda: torch.where(mask, x, y)), str(tuple(x.shape))

    if name == "getitem":
        rows, cols = p["batch"] * 8, p["dim"]
        x = t_rand((rows, cols, 8))
        mask = t_bool((rows,))
        idx1 = t_int((rows // 2,), 0, rows)
        idx2 = t_int((rows // 2,), 0, cols)
        neg_idx = t_int((rows // 2,), -rows, 0)
        cases = {
            "slice": lambda: x[1:rows - 1, 2:cols - 2],
            "int": lambda: x[1],
            "step": lambda: x[::2, ::2],
            "mask": lambda: x[mask],
            "fancy_1d": lambda: x[idx1],
            "fancy_2d": lambda: x[idx1, idx2],
            "ellipsis": lambda: x[..., 1:5],
            "newaxis": lambda: x[:, None, :],
            "negative_int": lambda: x[-1],
            "negative_slice": lambda: x[-rows // 2:-1],
            "fancy_negative": lambda: x[neg_idx],
        }
        return cases.get(variant, lambda: x[:rows // 2]), str(tuple(x.shape))

    # 11. Ordering & Search / Histogram
    if domain == "ordering_ops":
        rows, cols = p["batch"] * 4, p["dim"] * 4
        x = t_rand((rows, cols))
        if name == "sort":
            return (lambda: torch.sort(x, dim=-1).values), str(tuple(x.shape))
        if name == "topk":
            return (lambda: torch.topk(x, min(32, cols // 2), dim=-1).values), str(tuple(x.shape))
        if name == "argsort":
            return (lambda: torch.argsort(x, dim=-1)), str(tuple(x.shape))
        if name == "kthvalue":
            return (lambda: torch.kthvalue(x, cols // 2, dim=-1).values), str(tuple(x.shape))
        if name == "median":
            return (lambda: torch.median(x, dim=-1).values), str(tuple(x.shape))

    if domain == "search_histogram":
        if name == "searchsorted":
            n = p["dim"] * 4
            boundaries = torch.linspace(-2.0, 2.0, n, device=device, dtype=torch.float32)
            vals = t_rand((p["batch"] * 16, n // 2), -2.5, 2.5, dt=torch.float32)
            return (lambda: torch.searchsorted(boundaries, vals)), str(tuple(vals.shape))
        if name == "histc":
            x = t_rand((p["vec"] // 4,), -3.0, 3.0, dt=torch.float32)
            return (lambda: torch.histc(x, bins=64, min=-3.0, max=3.0)), str(tuple(x.shape))

    # 12. Shape / Reindex-alias / Structure / Public API gaps
    if domain in ("shape", "reindex_ops"):
        shape = (p["batch"], p["seq"], p["dim"])
        x = t_rand(shape)
        if name == "reshape":
            return (lambda: x.reshape(p["batch"] * p["seq"], p["dim"]).contiguous()), str(shape)
        if name == "flatten":
            return (lambda: torch.flatten(x, 1).contiguous()), str(shape)
        if name == "squeeze":
            x1 = x.unsqueeze(1)
            return (lambda: x1.squeeze(1)), str(tuple(x1.shape))
        if name == "unsqueeze":
            return (lambda: x.unsqueeze(1)), str(shape)
        if name == "permute":
            return (lambda: x.permute(0, 2, 1).contiguous()), str(shape)
        if name == "transpose":
            return (lambda: x.transpose(1, 2).contiguous()), str(shape)
        if name == "flip":
            return (lambda: torch.flip(x, dims=(-1,))), str(shape)
        if name == "roll":
            return (lambda: torch.roll(x, shifts=7, dims=-1)), str(shape)
        if name == "cat":
            xs = [t_rand(shape) for _ in range(4)]
            return (lambda: torch.cat(xs, dim=1)), f"4x{shape}"
        if name == "stack":
            xs = [t_rand(shape) for _ in range(4)]
            return (lambda: torch.stack(xs, dim=0)), f"4x{shape}"
        if name == "split":
            return (lambda: torch.split(x, p["dim"] // 4, dim=-1)[0]), str(shape)
        if name == "chunk":
            return (lambda: torch.chunk(x, 4, dim=-1)[0]), str(shape)
        if name == "unbind":
            return (lambda: torch.unbind(x, dim=0)[0]), str(shape)
        if name == "repeat":
            xs = t_rand((p["batch"], p["seq"] // 2, p["dim"] // 2))
            return (lambda: xs.repeat(1, 2, 2)), str(tuple(xs.shape))
        if name == "tile":
            xs = t_rand((p["batch"], p["seq"] // 2, p["dim"] // 2))
            return (lambda: torch.tile(xs, (1, 2, 2))), str(tuple(xs.shape))
        if name in ("expand", "broadcast_to"):
            xs = t_rand((p["batch"], 1, p["dim"]))
            return (lambda: xs.expand(shape).contiguous()), str(shape)

    if domain == "structure_ops":
        m = p["mat"]
        x = t_rand((p["batch"], m, m))
        if name == "tril":
            return (lambda: torch.tril(x)), str(tuple(x.shape))
        if name == "triu":
            return (lambda: torch.triu(x)), str(tuple(x.shape))
        if name == "cross":
            a = t_rand((p["vec"] // 8, 3))
            b = t_rand((p["vec"] // 8, 3))
            return (lambda: torch.linalg.cross(a, b, dim=-1)), str(tuple(a.shape))
        if name == "kron":
            k = 16 if scale == "tiny" else 32
            a, b = t_rand((k, k)), t_rand((k, k))
            return (lambda: torch.kron(a, b)), f"({k},{k})"
        if name == "tensordot":
            a = t_rand((p["batch"], m // 2, m // 2))
            b = t_rand((m // 2, m // 2, p["batch"]))
            return (lambda: torch.tensordot(a, b, dims=([1, 2], [0, 1]))), str(tuple(a.shape))
        if name in ("cummax", "cummin"):
            xs = t_rand((p["batch"] * 4, p["seq"], 64))
            fn = getattr(torch, name)
            return (lambda f=fn: f(xs, dim=-1).values), str(tuple(xs.shape))
        if name == "meshgrid":
            a = t_rand((m,))
            b = t_rand((m,))
            return (lambda: torch.meshgrid(a, b, indexing="ij")[0]), f"({m},{m})"

    if domain == "public_api_gaps":
        if name == "nonzero":
            x = t_bool((p["batch"] * 4, p["dim"]))
            return (lambda: torch.nonzero(x)), str(tuple(x.shape))
        if name == "unique":
            x = t_int((p["vec"] // 16,), 0, 1024, dt=torch.int32)
            return (lambda: torch.unique(x)), str(tuple(x.shape))
        if name == "einsum":
            a = t_rand((p["batch"], p["mat"], p["mat"]))
            b = t_rand((p["batch"], p["mat"], p["mat"]))
            return (lambda: torch.einsum("bik,bkj->bij", a, b)), str(tuple(a.shape))

    # 13. FFT
    if domain == "fft":
        shape = (p["batch"] * 2, p["seq"], p["dim"])
        x = t_rand(shape, dt=torch.float32)
        if name == "rfft":
            return (lambda: torch.fft.rfft(x, dim=-1)), str(shape)
        if name == "irfft":
            spec = torch.fft.rfft(x, dim=-1)
            return (lambda: torch.fft.irfft(spec, n=p["dim"], dim=-1)), str(shape)
        if name == "fft":
            return (lambda: torch.fft.fft(x, dim=-1)), str(shape)
        if name == "ifft":
            spec = torch.fft.fft(x, dim=-1)
            return (lambda: torch.fft.ifft(spec, dim=-1)), str(shape)

    # 14. Linear algebra
    if domain == "linalg":
        lb, ln = p["linalg_b"], p["linalg_n"]
        a_f32 = t_rand((lb, ln, ln), -0.5, 0.5, dt=torch.float32)
        eye = torch.eye(ln, device=device, dtype=torch.float32)
        spd = torch.matmul(a_f32, a_f32.transpose(-1, -2)) + eye * float(ln)
        if name == "dot":
            u, v = t_rand((p["vec"] // 4,)), t_rand((p["vec"] // 4,))
            return (lambda: torch.dot(u, v)), str(tuple(u.shape))
        if name == "outer":
            u, v = t_rand((p["mat"] * 2,)), t_rand((p["mat"] * 2,))
            return (lambda: torch.outer(u, v)), str(tuple(u.shape))
        if name == "mv":
            m = t_rand((p["mat"] * 2, p["mat"] * 2))
            v = t_rand((p["mat"] * 2,))
            return (lambda: torch.mv(m, v)), str(tuple(m.shape))
        if name == "inner":
            u = t_rand((p["batch"], p["mat"]))
            v = t_rand((p["batch"], p["mat"]))
            return (lambda: torch.inner(u, v)), str(tuple(u.shape))
        if name == "trace":
            m = t_rand((p["mat"] * 2, p["mat"] * 2))
            return (lambda: torch.trace(m)), str(tuple(m.shape))
        if name == "diag":
            v = t_rand((p["mat"] * 2,))
            return (lambda: torch.diag(v)), str(tuple(v.shape))
        if name == "diagonal":
            return (lambda: torch.diagonal(a_f32, dim1=-2, dim2=-1)), str(tuple(a_f32.shape))
        if name == "inv":
            return (lambda: torch.linalg.inv(spd)), str(tuple(spd.shape))
        if name == "det":
            return (lambda: torch.linalg.det(spd)), str(tuple(spd.shape))
        if name == "slogdet":
            return (lambda: torch.linalg.slogdet(spd).logabsdet), str(tuple(spd.shape))
        if name == "solve":
            rhs = t_rand((lb, ln, ln // 2), dt=torch.float32)
            return (lambda: torch.linalg.solve(spd, rhs)), str(tuple(spd.shape))
        if name == "cholesky":
            return (lambda: torch.linalg.cholesky(spd)), str(tuple(spd.shape))
        if name == "qr":
            return (lambda: torch.linalg.qr(spd).R), str(tuple(spd.shape))
        if name == "svd":
            return (lambda: torch.linalg.svd(spd, full_matrices=False).S), str(tuple(spd.shape))

    return None, f"no benchmark recipe for {full} ({domain})"


# ---------------------------------------------------------------------------
# Worker (runs all selected ops in ONE fresh process for one runtime)
# ---------------------------------------------------------------------------

def _sync(torch, runtime, device, out):
    if runtime == "jittor":
        import jittor as jt
        t = _extract_primary_tensor(torch, out)
        if t is not None:
            jt.sync([t], device_sync=device != "cpu")
        jt.sync_all(device != "cpu")
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _summary_scalar(torch, out):
    t = _extract_primary_tensor(torch, out)
    if t is None:
        return None
    try:
        v = t.detach()
        if hasattr(v, "is_complex") and (v.is_complex() if callable(v.is_complex) else v.is_complex):
            v = v.real
        v = v.float()
        return float((v if v.numel() == 1 else v.mean()).cpu().item())
    except Exception:
        return None


def _run_worker(args):
    import numpy as np

    os.environ.pop("JITTOR_TORCH_SHIM", None)
    import torch
    if args.runtime == "jittor":
        if not hasattr(torch, "_torch_compat_install_context"):
            raise SystemExit("torch did not resolve to the Jittor shim")
        import jittor as jt
        if args.device == "cpu":
            jt.flags.use_cuda = 0
        elif args.device == "cuda":
            jt.flags.use_cuda = 1
    else:
        if not hasattr(torch, "_C") or hasattr(torch, "_torch_compat_install_context"):
            raise SystemExit("torch did not resolve to independent binary PyTorch")

    dtype = getattr(torch, args.dtype)
    inventory = _select_entries(discover_opinfo_inventory(), args)

    rows = []
    for entry in inventory:
        row = {
            "name": entry["name"],
            "domain": entry["domain"],
            "runtime": args.runtime,
            "status": "ok",
        }
        try:
            fn, shape_desc = _build_op_runner(torch, np, entry, args.device, dtype, args.scale)
            row["shape"] = shape_desc
            if fn is None:
                row["status"] = "skipped"
                row["reason"] = shape_desc
                rows.append(row)
                continue

            # First call (includes JIT compile for Jittor)
            t_first = time.perf_counter()
            out = fn()
            _sync(torch, args.runtime, args.device, out)
            row["first_ms"] = (time.perf_counter() - t_first) * 1000.0
            row["value"] = _summary_scalar(torch, out)

            for _ in range(args.warmup):
                out = fn()
                _sync(torch, args.runtime, args.device, out)

            times_ms = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                out = fn()
                _sync(torch, args.runtime, args.device, out)
                times_ms.append((time.perf_counter() - t0) * 1000.0)
            row["median_ms"] = statistics.median(times_ms)
            row["min_ms"] = min(times_ms)
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    out_path = Path(args.worker_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"runtime": args.runtime, "rows": rows}, indent=1),
                        encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# Orchestrator & leaderboard report
# ---------------------------------------------------------------------------

def _select_entries(inventory, args):
    rows = list(inventory)
    if args.domains:
        wanted = {d.strip() for d in args.domains.split(",") if d.strip()}
        rows = [r for r in rows if r["domain"] in wanted]
    if args.ops:
        wanted = {o.strip() for o in args.ops.split(",") if o.strip()}
        rows = [r for r in rows if r["name"] in wanted or r["op"] in wanted]
    return rows


def _ensure_shim(state):
    shim_dir = state / "shim"
    if (shim_dir / "torch" / "__init__.py").is_file():
        return shim_dir
    import importlib.util
    deploy_path = REPO / "compat" / "shim" / "deploy.py"
    spec = importlib.util.spec_from_file_location("_opinfo_shim_deploy", deploy_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.deploy(shim_dir, resource_root=REPO / "compat" / "shim" / "resources")
    return shim_dir


def _worker_env(runtime, args, state):
    env = os.environ.copy()
    if runtime == "torch":
        env["PYTHONPATH"] = ""
        for k in JITTOR_VARIABLES:
            env.pop(k, None)
    else:
        shim_dir = _ensure_shim(state)
        env["PYTHONPATH"] = os.pathsep.join([str(shim_dir), str(REPO / "python")])
        env["JITTOR_HOME"] = str(state / "jittor-home")
        env["JITTOR_TORCH_CACHE_ROOT"] = str(state / "torch-shim")
        env["cache_name"] = "bench-opinfo"
        if args.device == "cpu":
            env["JT_BACKEND"] = "cpu"
            env["JT_USE_CUDA"] = "0"
            env["CUDA_VISIBLE_DEVICES"] = ""
    if args.gpu is not None:
        key = "ASCEND_RT_VISIBLE_DEVICES" if args.device == "npu" else "CUDA_VISIBLE_DEVICES"
        env[key] = args.gpu
    env["TMPDIR"] = str(state / "tmp")
    return env


def _render_report(compared, skipped, errors, meta):
    ratios = [r["ratio"] for r in compared if r["ratio"] is not None]
    geomean = math.exp(sum(math.log(max(x, 1e-6)) for x in ratios) / len(ratios)) if ratios else None
    med_ratio = statistics.median(ratios) if ratios else None
    faster = sum(1 for x in ratios if x <= 1.0)
    slow_2x = sum(1 for x in ratios if x >= 2.0)
    mismatches = [r for r in compared if not r["agree"]]

    lines = [
        "# Torch Compat Full OpInfo Operator Benchmark Report",
        "",
        f"- Device: `{meta['device']}` | Dtype: `{meta['dtype']}` | Scale: `{meta['scale']}`",
        f"- Total OpInfo entries: **{meta['total']}** | Compared: **{len(compared)}** "
        f"| Skipped: **{len(skipped)}** | Errors: **{len(errors)}**",
    ]
    if geomean is not None:
        lines.append(
            f"- Overall Jittor / PyTorch ratio: **geomean {geomean:.2f}x**, "
            f"**median {med_ratio:.2f}x** (<=1.0x faster/parity: {faster}, "
            f">=2.0x slower: {slow_2x}, numerical mismatches: {len(mismatches)})"
        )

    # Per-domain breakdown
    by_domain = {}
    for r in compared:
        by_domain.setdefault(r["domain"], []).append(r)
    lines += [
        "",
        "## Per-Domain Summary",
        "",
        "| Domain | Ops | Geomean (JT/PT) | Median (JT/PT) | Slowest Op |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for domain in sorted(by_domain):
        items = by_domain[domain]
        dr = [i["ratio"] for i in items if i["ratio"] is not None]
        dg = math.exp(sum(math.log(max(x, 1e-6)) for x in dr) / len(dr)) if dr else 0.0
        dm = statistics.median(dr) if dr else 0.0
        worst = max(items, key=lambda i: i["ratio"] or 0.0)
        lines.append(
            f"| `{domain}` | {len(items)} | {dg:.2f}x | {dm:.2f}x | "
            f"`{worst['name']}` ({worst['ratio']:.2f}x) |"
        )

    # Slowest 20 operators vs PyTorch
    slowest = sorted([r for r in compared if r["ratio"] is not None],
                     key=lambda r: r["ratio"], reverse=True)[:20]
    lines += [
        "",
        "## Slowest 20 Operators vs PyTorch",
        "",
        "| Operator | Domain | Shape | PyTorch (ms) | Jittor (ms) | Ratio | Agree |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in slowest:
        lines.append(
            f"| `{r['name']}` | `{r['domain']}` | `{r['shape']}` | "
            f"{r['torch_ms']:.3f} | {r['jittor_ms']:.3f} | **{r['ratio']:.2f}x** | "
            f"{'ok' if r['agree'] else 'DIFF'} |"
        )

    # Fastest 15 operators vs PyTorch
    fastest = sorted([r for r in compared if r["ratio"] is not None],
                     key=lambda r: r["ratio"])[:15]
    lines += [
        "",
        "## Fastest 15 Operators vs PyTorch",
        "",
        "| Operator | Domain | Shape | PyTorch (ms) | Jittor (ms) | Ratio | Agree |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in fastest:
        lines.append(
            f"| `{r['name']}` | `{r['domain']}` | `{r['shape']}` | "
            f"{r['torch_ms']:.3f} | {r['jittor_ms']:.3f} | **{r['ratio']:.2f}x** | "
            f"{'ok' if r['agree'] else 'DIFF'} |"
        )

    if errors:
        lines += ["", "## Errors", ""]
        for e in errors[:30]:
            lines.append(f"- `{e['name']}` ({e['runtime']}): `{e['error']}`")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--torch-python", default=os.environ.get("REAL_TORCH_PYTHON"))
    parser.add_argument("--jittor-python", default=sys.executable)
    parser.add_argument("--device", choices=("cuda", "cpu", "npu"), default="cuda")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--scale", choices=("tiny", "medium", "large"), default="medium")
    parser.add_argument("--domains", default=None, help="comma-separated OpInfo domain names")
    parser.add_argument("--ops", default=None, help="comma-separated OpInfo operator names")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out", default=None)
    parser.add_argument("--state", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--json", action="store_true")
    # Internal worker mode
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--runtime", choices=("torch", "jittor"), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        return _run_worker(args)

    inventory = _select_entries(discover_opinfo_inventory(), args)
    if args.list:
        if args.json:
            print(json.dumps(inventory, indent=2))
        else:
            for row in inventory:
                print(f"{row['name']:36s} {row['domain']:24s} {row['kind']}")
        return 0

    if not args.torch_python:
        parser.error("--torch-python (or REAL_TORCH_PYTHON) is required")

    lab = Path(os.environ.get("JITTOR_LAB_ROOT", str(REPO.parent / "jittor-lab"))).expanduser()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else (lab / "_state" / "bench-opinfo" / stamp)
    state_dir = Path(args.state) if args.state else (lab / "_state" / "bench-opinfo" / "cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "tmp").mkdir(parents=True, exist_ok=True)

    worker_results = {}
    for runtime, py_exe in (("torch", args.torch_python), ("jittor", args.jittor_python)):
        w_out = out_dir / f"{runtime}_raw.json"
        cmd = [
            py_exe, str(Path(__file__).resolve()),
            "--worker", "--runtime", runtime, "--worker-out", str(w_out),
            "--device", args.device, "--dtype", args.dtype, "--scale", args.scale,
            "--warmup", str(args.warmup), "--repeats", str(args.repeats),
        ]
        if args.domains:
            cmd += ["--domains", args.domains]
        if args.ops:
            cmd += ["--ops", args.ops]
        print(f"Running {runtime} OpInfo sweep ({len(inventory)} ops, scale={args.scale})...",
              flush=True)
        proc = subprocess.run(cmd, env=_worker_env(runtime, args, state_dir),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (out_dir / f"{runtime}.log").write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0 or not w_out.is_file():
            raise SystemExit(f"{runtime} worker failed (rc={proc.returncode}); see {out_dir / (runtime + '.log')}")
        worker_results[runtime] = {
            r["name"]: r for r in json.loads(w_out.read_text(encoding="utf-8"))["rows"]
        }

    compared, skipped, errors = [], [], []
    tol = 5e-2 if args.dtype in ("bfloat16", "float16") else 1e-2
    for entry in inventory:
        name = entry["name"]
        tr = worker_results["torch"].get(name, {})
        jr = worker_results["jittor"].get(name, {})
        if tr.get("status") == "skipped" or jr.get("status") == "skipped":
            skipped.append({"name": name, "domain": entry["domain"],
                            "reason": tr.get("reason") or jr.get("reason")})
            continue
        if tr.get("status") != "ok":
            errors.append({"name": name, "runtime": "torch", "error": tr.get("error", "unknown")})
            continue
        if jr.get("status") != "ok":
            errors.append({"name": name, "runtime": "jittor", "error": jr.get("error", "unknown")})
            continue
        t_ms, j_ms = tr["median_ms"], jr["median_ms"]
        ratio = (j_ms / t_ms) if t_ms > 0 else None
        tv, jv = tr.get("value"), jr.get("value")
        agree = True
        if tv is not None and jv is not None:
            if math.isnan(tv) and math.isnan(jv):
                agree = True
            else:
                rel = abs(tv - jv) / max(1.0, abs(tv))
                agree = rel <= tol
        compared.append({
            "name": name,
            "domain": entry["domain"],
            "shape": tr.get("shape", ""),
            "torch_ms": t_ms,
            "jittor_ms": j_ms,
            "jittor_first_ms": jr.get("first_ms"),
            "ratio": ratio,
            "torch_val": tv,
            "jittor_val": jv,
            "agree": agree,
        })

    meta = {
        "total": len(inventory),
        "device": args.device,
        "dtype": args.dtype,
        "scale": args.scale,
    }
    report_md = _render_report(compared, skipped, errors, meta)
    (out_dir / "opinfo_results.json").write_text(
        json.dumps({"meta": meta, "compared": compared, "skipped": skipped, "errors": errors},
                   indent=2),
        encoding="utf-8",
    )
    (out_dir / "opinfo_report.md").write_text(report_md, encoding="utf-8")
    print(report_md)
    print(f"Saved report to {out_dir / 'opinfo_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
