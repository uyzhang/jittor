"""Regression tests for GPU grid-stride lgamma/digamma, FP32 pow, fast_clamp, and selu."""

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F


_DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lgamma_and_digamma_1d_and_2d_parity(device, dtype):
    # Exercise both 1-D (previously launched <<<1, 16>>> on CUDA) and 2-D shapes
    # across positive values and negative non-integers.
    rtol = 1e-5 if dtype == torch.float32 else 1e-11
    vals_1d = np.array([0.25, 0.5, 1.0, 2.5, 10.0, 25.0, -0.5, -2.5], dtype=np.float64)
    vals_1d = np.tile(vals_1d, 256)
    x = torch.tensor(vals_1d, device=device, dtype=dtype)
    x_cpu = x.cpu()
    np.testing.assert_allclose(torch.lgamma(x).cpu().numpy(), torch.lgamma(x_cpu).numpy(), rtol=rtol)
    np.testing.assert_allclose(torch.digamma(x).cpu().numpy(), torch.digamma(x_cpu).numpy(), rtol=rtol)

    x2 = x.reshape(32, -1)
    np.testing.assert_allclose(torch.lgamma(x2).cpu().numpy(), torch.lgamma(x2.cpu()).numpy(), rtol=rtol)
    np.testing.assert_allclose(torch.digamma(x2).cpu().numpy(), torch.digamma(x2.cpu()).numpy(), rtol=rtol)


@pytest.mark.parametrize("device", _DEVICES)
def test_signed_pow_f32_and_f64_parity(device):
    for dtype, rtol in ((torch.float32, 1e-5), (torch.float64, 1e-11)):
        base = torch.tensor([-2.0, -3.0, 2.0, 0.5, 4.0], device=device, dtype=dtype)
        exp = torch.tensor([3.0, 2.0, 3.0, 0.5, -0.5], device=device, dtype=dtype)
        expected = np.array([-8.0, 9.0, 8.0, math.sqrt(0.5), 0.5], dtype=np.float64)
        np.testing.assert_allclose(torch.pow(base, exp).cpu().numpy(), expected, rtol=rtol)


@pytest.mark.parametrize("device", _DEVICES)
def test_fast_clamp_and_activations_on_torch_tensor(device):
    x = torch.tensor([-4.0, -1.0, 0.25, 2.5, 4.0, float("nan")], device=device, dtype=torch.float32)
    clamped = torch.clamp(x, -1.0, 1.0)
    assert isinstance(clamped, torch.Tensor)
    assert clamped.device.type == device
    arr = clamped.cpu().numpy()
    np.testing.assert_allclose(arr[:5], [-1.0, -1.0, 0.25, 1.0, 1.0])
    assert np.isnan(arr[5])

    # Hardswish, hardsigmoid, hardtanh, selu, atan2 parity vs CPU
    x_fin = torch.tensor([-4.0, -1.0, -1e-6, 0.0, 0.5, 3.5], device=device, dtype=torch.float32)
    y_fin = torch.tensor([1.0, 2.0, 0.5, 1.5, 0.25, 2.0], device=device, dtype=torch.float32)
    for fn in (F.hardswish, F.hardsigmoid, F.hardtanh, F.selu):
        out = fn(x_fin)
        assert out.device.type == device
        np.testing.assert_allclose(out.cpu().numpy(), fn(x_fin.cpu()).numpy(), rtol=1e-5, atol=1e-6)
    a_out = torch.atan2(x_fin, y_fin)
    assert a_out.device.type == device
    np.testing.assert_allclose(a_out.cpu().numpy(), np.arctan2(x_fin.cpu().numpy(), y_fin.cpu().numpy()), rtol=1e-5, atol=1e-6)
