"""Regression tests for CUDA fast reductions, triu/tril/trace, and pointwise ops."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_reductions_var_std_prod_max_min_cuda_cpu_parity():
    rng = np.random.RandomState(41)
    x_np = rng.randn(8, 16, 32).astype(np.float32)
    xp_np = rng.uniform(0.8, 1.2, size=(8, 16, 32)).astype(np.float32)

    for dim in (-1, 1, 0):
        for keepdim in (False, True):
            xc = torch.tensor(x_np, requires_grad=True)
            xg = torch.tensor(x_np, device="cuda", requires_grad=True)
            vc = torch.var(xc, dim=dim, keepdim=keepdim)
            vg = torch.var(xg, dim=dim, keepdim=keepdim)
            np.testing.assert_allclose(vg.cpu().numpy(), vc.numpy(), rtol=1e-4, atol=1e-5)

            sc = torch.std(xc, dim=dim, keepdim=keepdim)
            sg = torch.std(xg, dim=dim, keepdim=keepdim)
            np.testing.assert_allclose(sg.cpu().numpy(), sc.numpy(), rtol=1e-4, atol=1e-5)

            mc = torch.max(xc, dim=dim, keepdim=keepdim)
            mg = torch.max(xg, dim=dim, keepdim=keepdim)
            np.testing.assert_allclose(mg.values.cpu().numpy(), mc.values.numpy(), rtol=1e-5)
            np.testing.assert_array_equal(mg.indices.cpu().numpy(), mc.indices.numpy())
            mc.values.sum().backward()
            mg.values.sum().backward()
            np.testing.assert_allclose(xg.grad.cpu().numpy(), xc.grad.numpy(), rtol=1e-5)

            xc2 = torch.tensor(x_np, requires_grad=True)
            xg2 = torch.tensor(x_np, device="cuda", requires_grad=True)
            nc = torch.min(xc2, dim=dim, keepdim=keepdim)
            ng = torch.min(xg2, dim=dim, keepdim=keepdim)
            np.testing.assert_allclose(ng.values.cpu().numpy(), nc.values.numpy(), rtol=1e-5)
            np.testing.assert_array_equal(ng.indices.cpu().numpy(), nc.indices.numpy())
            nc.values.sum().backward()
            ng.values.sum().backward()
            np.testing.assert_allclose(xg2.grad.cpu().numpy(), xc2.grad.numpy(), rtol=1e-5)

            pc = torch.prod(torch.tensor(xp_np), dim=dim, keepdim=keepdim)
            pg = torch.prod(torch.tensor(xp_np, device="cuda"), dim=dim, keepdim=keepdim)
            np.testing.assert_allclose(pg.cpu().numpy(), pc.numpy(), rtol=1e-4, atol=1e-5)

    # Backward parity for last-dim var, std, and prod (including a zero entry in prod)
    for fn in (lambda t: torch.var(t, dim=-1), lambda t: torch.std(t, dim=-1)):
        xc = torch.tensor(x_np, requires_grad=True)
        xg = torch.tensor(x_np, device="cuda", requires_grad=True)
        fn(xc).sum().backward()
        fn(xg).sum().backward()
        np.testing.assert_allclose(xg.grad.cpu().numpy(), xc.grad.numpy(), rtol=1e-3, atol=1e-4)

    # Non-zero prod backward matches CPU, and zero-entry prod backward matches exact numpy formula
    xc = torch.tensor(xp_np, requires_grad=True)
    xg = torch.tensor(xp_np, device="cuda", requires_grad=True)
    torch.prod(xc, dim=-1).sum().backward()
    torch.prod(xg, dim=-1).sum().backward()
    np.testing.assert_allclose(xg.grad.cpu().numpy(), xc.grad.numpy(), rtol=1e-3, atol=1e-4)

    xp_z = xp_np.copy()
    xp_z[0, 1, 3] = 0.0
    xg_z = torch.tensor(xp_z, device="cuda", requires_grad=True)
    torch.prod(xg_z, dim=-1).sum().backward()
    got_row = xg_z.grad.cpu().numpy()[0, 1]
    want_zero_grad = float(np.prod(np.delete(xp_z[0, 1], 3)))
    assert np.isfinite(got_row).all()
    np.testing.assert_allclose(got_row[3], want_zero_grad, rtol=1e-4)
    np.testing.assert_allclose(np.delete(got_row, 3), 0.0, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_triu_tril_trace_cuda_cpu_parity():
    rng = np.random.RandomState(42)
    x_np = rng.randn(4, 16, 16).astype(np.float32)
    w_np = rng.randn(4, 16, 16).astype(np.float32)
    for diag in (-2, 0, 1, 5):
        for fn in (torch.triu, torch.tril):
            xc = torch.tensor(x_np, requires_grad=True)
            yc = fn(xc, diagonal=diag)
            (yc * torch.tensor(w_np)).sum().backward()
            xg = torch.tensor(x_np, device="cuda", requires_grad=True)
            yg = fn(xg, diagonal=diag)
            (yg * torch.tensor(w_np, device="cuda")).sum().backward()
            np.testing.assert_allclose(yg.cpu().numpy(), yc.numpy(), rtol=1e-6)
            np.testing.assert_allclose(xg.grad.cpu().numpy(), xc.grad.numpy(), rtol=1e-6)

    m_np = rng.randn(17, 23).astype(np.float32)
    mc = torch.tensor(m_np, requires_grad=True)
    mg = torch.tensor(m_np, device="cuda", requires_grad=True)
    tc = torch.trace(mc); tc.backward()
    tg = torch.trace(mg); tg.backward()
    np.testing.assert_allclose(tg.cpu().numpy(), tc.numpy(), rtol=1e-5)
    np.testing.assert_allclose(mg.grad.cpu().numpy(), mc.grad.numpy(), rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pointwise_and_activation_cuda_cpu_parity():
    rng = np.random.RandomState(43)
    x_np = rng.randn(128, 32).astype(np.float32)
    x_pos = np.abs(x_np) + 0.25
    y_np = rng.randn(128, 32).astype(np.float32)
    w_np = rng.randn(128, 32).astype(np.float32)

    cases = [
        ("rsqrt", torch.rsqrt, x_pos),
        ("expm1", torch.expm1, np.clip(x_np, -5, 5)),
        ("reciprocal", torch.reciprocal, x_pos),
        ("softplus", lambda t: F.softplus(t, beta=1.5, threshold=10.0), x_np),
        ("mish", F.mish, x_np),
        ("logsigmoid", F.logsigmoid, x_np),
        ("elu", lambda t: F.elu(t, alpha=1.2), x_np),
        ("celu", lambda t: F.celu(t, alpha=1.3), x_np),
    ]
    for name, fn, inp in cases:
        xc = torch.tensor(inp, requires_grad=True)
        yc = fn(xc)
        (yc * torch.tensor(w_np)).sum().backward()
        xg = torch.tensor(inp, device="cuda", requires_grad=True)
        yg = fn(xg)
        (yg * torch.tensor(w_np, device="cuda")).sum().backward()
        np.testing.assert_allclose(yg.cpu().numpy(), yc.numpy(), rtol=1e-4, atol=1e-5, err_msg=name)
        np.testing.assert_allclose(xg.grad.cpu().numpy(), xc.grad.numpy(), rtol=1e-3, atol=1e-4, err_msg=f"{name} grad")

    yc_in = torch.tensor(y_np, requires_grad=True)
    xc_in = torch.tensor(x_pos, requires_grad=True)
    ac = torch.atan2(yc_in, xc_in)
    (ac * torch.tensor(w_np)).sum().backward()

    yg_in = torch.tensor(y_np, device="cuda", requires_grad=True)
    xg_in = torch.tensor(x_pos, device="cuda", requires_grad=True)
    ag = torch.atan2(yg_in, xg_in)
    (ag * torch.tensor(w_np, device="cuda")).sum().backward()

    np.testing.assert_allclose(ag.cpu().numpy(), ac.numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(yg_in.grad.cpu().numpy(), yc_in.grad.numpy(), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(xg_in.grad.cpu().numpy(), xc_in.grad.numpy(), rtol=1e-3, atol=1e-4)
