"""Fused RMSNorm against Qwen2RMSNorm's PyTorch math, forward and input gradient."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def reference(x, weight, eps):
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 77, 1536), (2, 5, 1000)])
def test_rmsnorm_matches_pytorch(dtype, shape):
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.rand(shape[-1], device="cuda") + 0.5  # fp32 and frozen, as in the run
    grad = torch.randn(shape, device="cuda", dtype=torch.promote_types(dtype, torch.float32))

    x_ref = x.clone().requires_grad_()
    ref = reference(x_ref, weight, 1e-6)
    ref.backward(grad)
    x_ours = x.clone().requires_grad_()
    ours = rmsnorm(x_ours, weight, 1e-6)
    ours.backward(grad)

    tol = 1e-5 if dtype == torch.float32 else 2e-3
    assert ours.dtype == ref.dtype
    torch.testing.assert_close(ours, ref, rtol=tol, atol=tol)
    torch.testing.assert_close(x_ours.grad.float(), x_ref.grad.float(), rtol=tol, atol=tol)
