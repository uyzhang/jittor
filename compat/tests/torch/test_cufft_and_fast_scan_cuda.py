"""Regression tests for CUDA cuFFT dispatch, warp/strided cumsum, and cumprod."""

import numpy as np
import pytest
import torch


def _devices():
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


@pytest.mark.parametrize("device", _devices())
def test_fft_ifft_rfft_irfft_fft2_parity(device):
    rng = np.random.RandomState(42)
    x_np = rng.randn(4, 16, 64).astype(np.float32)
    x = torch.tensor(x_np, device=device)

    f1 = torch.fft.fft(x)
    ref1 = np.fft.fft(x_np, axis=-1)
    np.testing.assert_allclose(f1.real.cpu().numpy(), ref1.real, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(f1.imag.cpu().numpy(), ref1.imag, atol=1e-3, rtol=1e-3)

    inv1 = torch.fft.ifft(f1)
    np.testing.assert_allclose(inv1.real.cpu().numpy(), x_np, atol=1e-4, rtol=1e-4)

    rf = torch.fft.rfft(x, norm="ortho")
    ref_rf = np.fft.rfft(x_np, axis=-1, norm="ortho")
    np.testing.assert_allclose(rf.real.cpu().numpy(), ref_rf.real, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(rf.imag.cpu().numpy(), ref_rf.imag, atol=1e-3, rtol=1e-3)

    irf = torch.fft.irfft(rf, norm="ortho")
    np.testing.assert_allclose(irf.cpu().numpy(), x_np, atol=1e-4, rtol=1e-4)

    f2 = torch.fft.fft2(x)
    ref2 = np.fft.fft2(x_np, axes=(-2, -1))
    np.testing.assert_allclose(f2.real.cpu().numpy(), ref2.real, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(f2.imag.cpu().numpy(), ref2.imag, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dim", [-1, 1, 0])
def test_cumprod_and_cumsum_forward_backward_parity(device, dim):
    rng = np.random.RandomState(7)
    x_np = rng.randn(4, 10, 12).astype(np.float32) * 0.4 + 0.8
    # Include negatives and a zero entry to verify exact zero/sign handling
    x_np[1, 3, 2] = -1.25
    x_np[2, 4, 5] = 0.0
    w_np = rng.randn(*x_np.shape).astype(np.float32)

    x_s = torch.tensor(x_np, device=device, requires_grad=True)
    y_s = torch.cumsum(x_s, dim=dim)
    np.testing.assert_allclose(y_s.cpu().numpy(), np.cumsum(x_np, axis=dim), atol=1e-4, rtol=1e-4)
    (y_s * torch.tensor(w_np, device=device)).sum().backward()
    ref_g_s = np.flip(np.cumsum(np.flip(w_np, axis=dim), axis=dim), axis=dim)
    np.testing.assert_allclose(x_s.grad.cpu().numpy(), ref_g_s, atol=1e-4, rtol=1e-4)

    x_p = torch.tensor(x_np, device=device, requires_grad=True)
    y_p = torch.cumprod(x_p, dim=dim)
    np.testing.assert_allclose(y_p.cpu().numpy(), np.cumprod(x_np, axis=dim), atol=1e-4, rtol=1e-4)
    if device == "cuda":
        # CUDA cumprod uses the exact zero-safe O(D) backward recurrence
        (y_p * torch.tensor(w_np, device=device)).sum().backward()
        x_m = np.moveaxis(x_np, dim, -1)
        y_m = np.cumprod(x_m, axis=-1)
        w_m = np.moveaxis(w_np, dim, -1)
        g_m = np.zeros_like(x_m)
        L = x_m.shape[-1]
        suffix = np.zeros_like(x_m[..., 0])
        for k in range(L - 1, -1, -1):
            suffix = w_m[..., k] if k == L - 1 else w_m[..., k] + x_m[..., k + 1] * suffix
            excl = np.ones_like(suffix) if k == 0 else y_m[..., k - 1]
            g_m[..., k] = excl * suffix
        ref_g_p = np.moveaxis(g_m, -1, dim)
        np.testing.assert_allclose(x_p.grad.cpu().numpy(), ref_g_p, atol=1e-3, rtol=1e-3)
