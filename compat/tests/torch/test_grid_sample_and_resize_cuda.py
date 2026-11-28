"""Regression tests for fused CUDA grid_sample, bilinear resize, and affine_grid."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [False, True])
def test_grid_sample_2d_cuda_cpu_parity(mode, padding_mode, align_corners):
    rng = np.random.RandomState(31)
    x_np = rng.randn(2, 8, 9, 11).astype(np.float32)
    g_np = rng.uniform(-1.3, 1.3, size=(2, 7, 8, 2)).astype(np.float32)
    w_np = rng.randn(2, 8, 7, 8).astype(np.float32)

    xc = torch.tensor(x_np, requires_grad=True)
    gc = torch.tensor(g_np, requires_grad=True)
    yc = F.grid_sample(xc, gc, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    (yc * torch.tensor(w_np)).sum().backward()

    xg = torch.tensor(x_np, device="cuda", requires_grad=True)
    gg = torch.tensor(g_np, device="cuda", requires_grad=True)
    yg = F.grid_sample(xg, gg, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    (yg * torch.tensor(w_np, device="cuda")).sum().backward()

    np.testing.assert_allclose(yg.cpu().numpy(), yc.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(xg.grad.cpu().numpy(), xc.grad.numpy(), rtol=1e-3, atol=1e-3)
    if mode == "bilinear":
        np.testing.assert_allclose(gg.grad.cpu().numpy(), gc.grad.numpy(), rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("out_size", [(21, 23), (5, 7)])
@pytest.mark.parametrize("align_corners", [False, True])
def test_interpolate_bilinear_cuda_cpu_parity(out_size, align_corners):
    rng = np.random.RandomState(32)
    x_np = rng.randn(2, 8, 11, 13).astype(np.float32)
    w_np = rng.randn(2, 8, out_size[0], out_size[1]).astype(np.float32)

    xc = torch.tensor(x_np, requires_grad=True)
    yc = F.interpolate(xc, size=out_size, mode="bilinear", align_corners=align_corners)
    (yc * torch.tensor(w_np)).sum().backward()

    xg = torch.tensor(x_np, device="cuda", requires_grad=True)
    yg = F.interpolate(xg, size=out_size, mode="bilinear", align_corners=align_corners)
    (yg * torch.tensor(w_np, device="cuda")).sum().backward()

    np.testing.assert_allclose(yg.cpu().numpy(), yc.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(xg.grad.cpu().numpy(), xc.grad.numpy(), rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("align_corners", [False, True])
def test_affine_grid_4d_cuda_cpu_parity(align_corners):
    rng = np.random.RandomState(33)
    th_np = rng.randn(4, 2, 3).astype(np.float32)
    w_np = rng.randn(4, 11, 13, 2).astype(np.float32)

    tc = torch.tensor(th_np, requires_grad=True)
    gc = F.affine_grid(tc, (4, 8, 11, 13), align_corners=align_corners)
    (gc * torch.tensor(w_np)).sum().backward()

    tg = torch.tensor(th_np, device="cuda", requires_grad=True)
    gg = F.affine_grid(tg, (4, 8, 11, 13), align_corners=align_corners)
    (gg * torch.tensor(w_np, device="cuda")).sum().backward()

    np.testing.assert_allclose(gg.cpu().numpy(), gc.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(tg.grad.cpu().numpy(), tc.grad.numpy(), rtol=1e-3, atol=1e-3)
