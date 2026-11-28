"""Regression tests for OpInfo-surfaced Torch APIs, namedtuples, and CUDA placement."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F


def _devices():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


@pytest.mark.parametrize("device", _devices())
def test_missing_public_torch_apis_parity(device):
    # 1. torch.neg
    x = torch.tensor([-2.0, 0.0, 3.5], device=device)
    np.testing.assert_allclose(torch.neg(x).cpu().numpy(), [2.0, 0.0, -3.5])

    # 2. torch.dot (1-D -> 0-D scalar)
    a = torch.tensor([1.0, 2.0, 3.0], device=device)
    b = torch.tensor([4.0, 5.0, 6.0], device=device)
    d = torch.dot(a, b)
    assert tuple(d.shape) == ()
    np.testing.assert_allclose(float(d.cpu().item()), 32.0, rtol=1e-5)

    # 3. torch.inner
    u = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
    v = torch.tensor([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]], device=device)
    np.testing.assert_allclose(
        torch.inner(u, v).cpu().numpy(),
        np.inner(u.cpu().numpy(), v.cpu().numpy()),
        rtol=1e-5,
    )

    # 4-5. torch.bitwise_left_shift / torch.bitwise_right_shift
    xi = torch.tensor([1, 4, 16], dtype=torch.int32, device=device)
    si = torch.tensor([1, 2, 1], dtype=torch.int32, device=device)
    np.testing.assert_array_equal(torch.bitwise_left_shift(xi, si).cpu().numpy(), [2, 16, 32])
    np.testing.assert_array_equal(torch.bitwise_right_shift(xi, si).cpu().numpy(), [0, 1, 8])

    # 6-7. F.max_pool1d / F.avg_pool1d
    sig = torch.tensor([[[1.0, 3.0, 2.0, 4.0, 0.0, 5.0]]], device=device)
    np.testing.assert_allclose(
        F.max_pool1d(sig, kernel_size=2, stride=2).cpu().numpy(),
        [[[3.0, 4.0, 5.0]]],
    )
    np.testing.assert_allclose(
        F.avg_pool1d(sig, kernel_size=2, stride=2).cpu().numpy(),
        [[[2.0, 3.0, 2.5]]],
    )
    np.testing.assert_allclose(
        F.avg_pool1d(sig, kernel_size=2, stride=2, padding=1, count_include_pad=False).cpu().numpy(),
        [[[1.0, 2.5, 2.0, 5.0]]],
    )

    # 8. torch.linalg.cross
    c1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
    c2 = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device)
    np.testing.assert_allclose(
        torch.linalg.cross(c1, c2, dim=-1).cpu().numpy(),
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        rtol=1e-5,
    )


@pytest.mark.parametrize("device", _devices())
def test_namedtuple_return_fields_for_kthvalue_and_slogdet(device):
    x = torch.tensor([[3.0, 1.0, 4.0, 2.0], [9.0, 5.0, 8.0, 6.0]], device=device)
    kv = torch.kthvalue(x, 2, dim=-1)
    assert hasattr(kv, "values") and hasattr(kv, "indices")
    np.testing.assert_allclose(kv.values.cpu().numpy(), [2.0, 6.0])
    np.testing.assert_array_equal(kv.indices.cpu().numpy(), [3, 3])
    # Tuple unpacking still works
    v, i = kv
    np.testing.assert_allclose(v.cpu().numpy(), [2.0, 6.0])

    # slogdet on CPU (or CUDA if CuPy/cuSOLVER available)
    if device == "cpu":
        m = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float32, device=device)
        sd = torch.linalg.slogdet(m)
        assert hasattr(sd, "sign") and hasattr(sd, "logabsdet")
        np.testing.assert_allclose(float(sd.sign.cpu().item()), 1.0, rtol=1e-5)
        np.testing.assert_allclose(float(sd.logabsdet.cpu().item()), np.log(11.0), rtol=1e-5)


@pytest.mark.parametrize("device", _devices())
def test_irfft_histc_cummax_cummin_preserve_device(device):
    # 1. fft.irfft round-trip on CPU & CUDA
    x = torch.randn(4, 16, device=device, dtype=torch.float32)
    spec = torch.fft.rfft(x, dim=-1)
    back = torch.fft.irfft(spec, n=16, dim=-1)
    assert back.device.type == device
    np.testing.assert_allclose(back.cpu().numpy(), x.cpu().numpy(), rtol=1e-4, atol=1e-4)

    # 2. histc on CPU & CUDA
    h_in = torch.tensor([0.1, 0.4, 0.6, 1.2, 1.9, 2.0], device=device, dtype=torch.float32)
    h = torch.histc(h_in, bins=4, min=0.0, max=2.0)
    assert h.device.type == device
    np.testing.assert_allclose(h.cpu().numpy(), [2.0, 1.0, 1.0, 2.0], atol=1e-5)

    # 3-4. cummax / cummin on CPU & CUDA
    c_in = torch.tensor([[1.0, 3.0, 2.0, 5.0], [4.0, 2.0, 6.0, 1.0]], device=device)
    cm = torch.cummax(c_in, dim=-1)
    assert cm.values.device.type == device
    np.testing.assert_allclose(cm.values.cpu().numpy(), [[1.0, 3.0, 3.0, 5.0], [4.0, 4.0, 6.0, 6.0]])
    np.testing.assert_array_equal(cm.indices.cpu().numpy(), [[0, 1, 1, 3], [0, 0, 2, 2]])

    cn = torch.cummin(c_in, dim=-1)
    assert cn.values.device.type == device
    np.testing.assert_allclose(cn.values.cpu().numpy(), [[1.0, 1.0, 1.0, 1.0], [4.0, 2.0, 2.0, 1.0]])
    np.testing.assert_array_equal(cn.indices.cpu().numpy(), [[0, 0, 0, 0], [0, 1, 1, 3]])
