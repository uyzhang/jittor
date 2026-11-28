"""Real multi-process tests for the MPI-free native TCP CPU (gloo) backend."""

import os
from pathlib import Path
import tempfile
import unittest

from _helpers.child_process import PYTHON, run_python_child


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LAUNCH = _REPO_ROOT / "python" / "jittor" / "distributed" / "launch.py"


_NATIVE_GLOO_WORKER = r"""
import os
import numpy as np
import jittor as jt
from jittor import distributed as dist

assert not jt.compile_extern.has_mpi, "test must run without MPI"
dist.init_process_group(backend="gloo")
rank = dist.get_rank()
world = dist.get_world_size()
assert world == 3, f"expected world_size=3, got {world}"
assert dist.get_backend() == "gloo"

# 1. all_reduce (sum / mean / max / min / product) on float32
x = jt.array(np.array([rank + 1.0, (rank + 1.0) * 10.0], dtype=np.float32))
s = dist.all_reduce(x, op="sum").numpy()
np.testing.assert_allclose(s, [6.0, 60.0], rtol=1e-5, atol=1e-5)

m = dist.all_reduce(x, op="mean").numpy()
np.testing.assert_allclose(m, [2.0, 20.0], rtol=1e-5, atol=1e-5)

mx = dist.all_reduce(x, op="max").numpy()
np.testing.assert_allclose(mx, [3.0, 30.0], rtol=1e-5, atol=1e-5)

mn = dist.all_reduce(x, op="min").numpy()
np.testing.assert_allclose(mn, [1.0, 10.0], rtol=1e-5, atol=1e-5)

pr = dist.all_reduce(x, op="product").numpy()
np.testing.assert_allclose(pr, [6.0, 6000.0], rtol=1e-4, atol=1e-4)

# 2. bfloat16 all_reduce
bf = jt.array(np.array([rank + 1.0, 2.0], dtype=np.float32)).cast("bfloat16")
bf_sum = dist.all_reduce(bf, op="sum")
assert str(bf_sum.dtype) == "bfloat16"
np.testing.assert_allclose(bf_sum.cast("float32").numpy(), [6.0, 6.0], rtol=1e-2, atol=1e-2)

# 3. broadcast from non-zero root
b = jt.array(np.array([rank * 100 + 7, rank * 100 + 9], dtype=np.int32))
b_out = dist.broadcast(b, src=2).numpy()
np.testing.assert_array_equal(b_out, [207, 209])

# 4. all_gather & reduce_scatter
ag = dist.all_gather(jt.array(np.array([[rank + 1.0, rank + 2.0]], dtype=np.float32))).numpy()
np.testing.assert_allclose(ag, [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], rtol=1e-5)

rs_in = jt.array(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32) * (rank + 1))
rs = dist.reduce_scatter(rs_in).numpy()
expected_rs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32) * 6.0
np.testing.assert_allclose(rs, expected_rs[rank * 2:(rank + 1) * 2], rtol=1e-5)

# 5. Autograd through all_reduce
w = jt.array(np.array([rank + 1.0, 2.0], dtype=np.float32))
w.start_grad()
y = dist.all_reduce(w, op="sum")
loss = (y * (rank + 1.0)).sum()
grad_w = jt.grad(loss, w).numpy()
# d(sum_r (r+1) * (sum_k w_k)) / d w_j = sum_r (r+1) = 6.0
np.testing.assert_allclose(grad_w, [6.0, 6.0], rtol=1e-5)

# 6. Subgroup [1, 2]
sub = dist.new_group([1, 2], name="sub_12")
if rank in (1, 2):
    sub_val = dist.all_reduce(jt.array(np.array([float(rank)], dtype=np.float32)), op="sum", group=sub).numpy()
    np.testing.assert_allclose(sub_val, [3.0], rtol=1e-5)

# 7. Point-to-point send/recv and bidirectional batch_isend_irecv
if rank == 0:
    dist.send(jt.array(np.array([42.0, 99.0], dtype=np.float32)), dst=1, tag=7)
elif rank == 1:
    buf = jt.zeros((2,), dtype="float32")
    dist.recv(buf, src=0, tag=7)
    np.testing.assert_allclose(buf.numpy(), [42.0, 99.0], rtol=1e-5)

if rank in (0, 2):
    peer = 2 if rank == 0 else 0
    send_t = jt.array(np.array([rank + 10.0, rank + 20.0], dtype=np.float32))
    recv_t = jt.zeros((2,), dtype="float32")
    dist.batch_isend_irecv([(True, send_t, peer, 11), (False, recv_t, peer, 11)])
    np.testing.assert_allclose(recv_t.numpy(), [peer + 10.0, peer + 20.0], rtol=1e-5)

dist.barrier()
dist.destroy_process_group()
print(f"NATIVE_GLOO_RANK_{rank}_OK", flush=True)
"""


_TORCH_GLOO_DDP_WORKER = r"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

assert dist.is_gloo_available()
assert dist.is_backend_available("gloo")

dist.init_process_group(backend="gloo")
rank = dist.get_rank()
world = dist.get_world_size()
assert world == 2
assert dist.get_backend() == "gloo"

# Tensor & object collectives
t = torch.tensor([float(rank + 1), float(rank + 3)], dtype=torch.float32)
dist.all_reduce(t, op=dist.ReduceOp.SUM)
np.testing.assert_allclose(t.numpy(), [3.0, 7.0], rtol=1e-5)

objs = [None, None]
dist.all_gather_object(objs, {"rank": rank, "msg": "gloo"})
assert objs == [{"rank": 0, "msg": "gloo"}, {"rank": 1, "msg": "gloo"}], objs

# CPU DDP-style training parity across 2 ranks without MPI
torch.manual_seed(1234)
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
# Broadcast initial parameters from rank 0
for p in model.parameters():
    dist.broadcast(p.data, src=0)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
x = torch.full((2, 4), float(rank + 1), dtype=torch.float32)
loss = model(x).pow(2).mean()
loss.backward()
for p in model.parameters():
    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
opt.step()

# Verify updated weights match identically across both ranks
w_sum = model[0].weight.detach().clone()
dist.all_reduce(w_sum, op=dist.ReduceOp.AVG)
np.testing.assert_allclose(model[0].weight.detach().numpy(), w_sum.numpy(), rtol=1e-5, atol=1e-5)

dist.barrier()
dist.destroy_process_group()
print(f"TORCH_GLOO_RANK_{rank}_OK", flush=True)
"""


class TestNativeGlooCpuBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Warm jit_utils_core / jittor_core once before spawning multi-rank
        # children so rank 0 does not exit with JIT_UTILS_UPDATED_EXIT_CODE (3)
        # on a brand-new JITTOR_HOME.
        import jittor  # noqa: F401

    def test_jtrun_gloo_three_rank_collectives_and_autograd_without_mpi(self):
        with tempfile.TemporaryDirectory() as tmp:
            done = run_python_child(
                [os.fspath(_LAUNCH), "-n", "3", "--backend", "gloo",
                 "--logdir", tmp, "--", PYTHON, "-c", _NATIVE_GLOO_WORKER],
                cwd=_REPO_ROOT,
                env={
                    "CUDA_VISIBLE_DEVICES": "",
                    "JT_BUILD_USE_MPI": "0",
                    "JT_BUILD_USE_NCCL": "0",
                    "JT_BUILD_USE_MKL": "0",
                },
                without_torch_mode=True,
                merge_stderr=True,
                timeout=180,
            )
            logs = {
                r: (Path(tmp) / f"rank{r}.log").read_text(encoding="utf-8", errors="replace")
                if (Path(tmp) / f"rank{r}.log").is_file() else ""
                for r in range(3)
            }
            self.assertEqual(done.returncode, 0, f"{done.stdout[-2000:]}\nRANK0:\n{logs[0][-2000:]}")
            for r in range(3):
                self.assertIn(f"NATIVE_GLOO_RANK_{r}_OK", logs[r])

    def test_torch_distributed_gloo_two_rank_cpu_training_without_mpi(self):
        with tempfile.TemporaryDirectory() as tmp:
            done = run_python_child(
                [os.fspath(_LAUNCH), "-n", "2", "--backend", "gloo",
                 "--logdir", tmp, "--", PYTHON, "-c", _TORCH_GLOO_DDP_WORKER],
                cwd=_REPO_ROOT,
                env={
                    "CUDA_VISIBLE_DEVICES": "",
                    "JITTOR_TORCH_SHIM": "1",
                    "JT_BUILD_USE_MPI": "0",
                    "JT_BUILD_USE_NCCL": "0",
                    "JT_BUILD_USE_MKL": "0",
                },
                merge_stderr=True,
                timeout=180,
            )
            logs = {
                r: (Path(tmp) / f"rank{r}.log").read_text(encoding="utf-8", errors="replace")
                if (Path(tmp) / f"rank{r}.log").is_file() else ""
                for r in range(2)
            }
            self.assertEqual(done.returncode, 0, f"{done.stdout[-2000:]}\nRANK0:\n{logs[0][-2000:]}")
            for r in range(2):
                self.assertIn(f"TORCH_GLOO_RANK_{r}_OK", logs[r])


if __name__ == "__main__":
    unittest.main()
