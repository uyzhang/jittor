"""Regression tests for fused CUDA cross_entropy, nll_loss, bce_with_logits, and avg pools."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("weighted", [False, True])
def test_cross_entropy_and_nll_cuda_cpu_parity(reduction, weighted):
    rng = np.random.RandomState(11)
    x_np = rng.randn(32, 64).astype(np.float32)
    t_np = rng.randint(0, 64, size=(32,)).astype(np.int64)
    t_np[0] = -100
    w_np = (rng.rand(64).astype(np.float32) + 0.1) if weighted else None

    for fn, inp_np, t_arr, ign in [
        (F.cross_entropy, x_np, t_np, -100),
        (F.nll_loss, x_np - np.log(np.exp(x_np - x_np.max(-1, keepdims=True)).sum(-1, keepdims=True)) - x_np.max(-1, keepdims=True),
         np.where(t_np < 0, 0, t_np), 0),
    ]:
        xc = torch.tensor(inp_np, requires_grad=True)
        tc = torch.tensor(t_arr)
        wc = torch.tensor(w_np) if weighted else None
        lc = fn(xc, tc, weight=wc, ignore_index=ign, reduction=reduction)
        (lc.sum() if reduction == "none" else lc).backward()

        xg = torch.tensor(inp_np, device="cuda", requires_grad=True)
        tg = torch.tensor(t_arr, device="cuda")
        wg = torch.tensor(w_np, device="cuda") if weighted else None
        lg = fn(xg, tg, weight=wg, ignore_index=ign, reduction=reduction)
        (lg.sum() if reduction == "none" else lg).backward()

        np.testing.assert_allclose(lg.cpu().numpy(), lc.numpy(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(xg.grad.cpu().numpy(), xc.grad.numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_bce_with_logits_cuda_cpu_parity(reduction):
    rng = np.random.RandomState(12)
    x_np = rng.randn(64, 32).astype(np.float32)
    t_np = rng.rand(64, 32).astype(np.float32)

    xc = torch.tensor(x_np, requires_grad=True)
    lc = F.binary_cross_entropy_with_logits(xc, torch.tensor(t_np), reduction=reduction)
    (lc.sum() if reduction == "none" else lc).backward()

    xg = torch.tensor(x_np, device="cuda", requires_grad=True)
    lg = F.binary_cross_entropy_with_logits(xg, torch.tensor(t_np, device="cuda"), reduction=reduction)
    (lg.sum() if reduction == "none" else lg).backward()

    np.testing.assert_allclose(lg.cpu().numpy(), lc.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(xg.grad.cpu().numpy(), xc.grad.numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_adaptive_and_avg_pool_cuda_cpu_parity():
    rng = np.random.RandomState(13)
    img2 = rng.randn(3, 8, 13, 15).astype(np.float32)
    for out_sz in ((5, 7), (4, 4)):
        xc = torch.tensor(img2, requires_grad=True)
        yc = F.adaptive_avg_pool2d(xc, out_sz)
        w = np.arange(1, yc.numel() + 1, dtype=np.float32).reshape(yc.shape)
        (yc * torch.tensor(w)).sum().backward()

        xg = torch.tensor(img2, device="cuda", requires_grad=True)
        yg = F.adaptive_avg_pool2d(xg, out_sz)
        (yg * torch.tensor(w, device="cuda")).sum().backward()

        np.testing.assert_allclose(yg.cpu().numpy(), yc.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(xg.grad.cpu().numpy(), xc.grad.numpy(), rtol=1e-4, atol=1e-4)

    xc2 = torch.tensor(img2, requires_grad=True)
    yc2 = F.avg_pool2d(xc2, 3, stride=2, padding=1, count_include_pad=False)
    yc2.sum().backward()
    xg2 = torch.tensor(img2, device="cuda", requires_grad=True)
    yg2 = F.avg_pool2d(xg2, 3, stride=2, padding=1, count_include_pad=False)
    yg2.sum().backward()
    np.testing.assert_allclose(yg2.cpu().numpy(), yc2.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(xg2.grad.cpu().numpy(), xc2.grad.numpy(), rtol=1e-4, atol=1e-4)

    vol3 = rng.randn(2, 4, 7, 8, 9).astype(np.float32)
    xc = torch.tensor(vol3, requires_grad=True)
    yc = F.avg_pool3d(xc, 3, stride=2, padding=1, count_include_pad=False)
    yc.sum().backward()
    xg = torch.tensor(vol3, device="cuda", requires_grad=True)
    yg = F.avg_pool3d(xg, 3, stride=2, padding=1, count_include_pad=False)
    yg.sum().backward()
    np.testing.assert_allclose(yg.cpu().numpy(), yc.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(xg.grad.cpu().numpy(), xc.grad.numpy(), rtol=1e-4, atol=1e-4)
