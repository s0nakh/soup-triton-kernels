"""The fused LoRA + rotary + cast prologue against Qwen2's PyTorch ops, forward and backward."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


@pytest.mark.parametrize("has_lora,has_rope", [(True, True), (False, True), (True, False)])
def test_prologue_matches_pytorch(has_lora, has_rope):
    from solution.rope import qkv_prologue

    torch.manual_seed(0)
    tokens, heads, dim = 57, 12, 128
    base = torch.randn(tokens, heads * dim, device="cuda", dtype=torch.float16)
    lora = torch.randn_like(base) * 0.1 if has_lora else None
    angles = torch.rand(tokens, dim // 2, device="cuda") * 6.0
    cos = torch.cat([angles.cos()] * 2, -1) if has_rope else None
    sin = torch.cat([angles.sin()] * 2, -1) if has_rope else None
    grad = torch.randn_like(base)

    b_ref = base.clone().requires_grad_()
    l_ref = lora.clone().requires_grad_() if has_lora else None
    x = b_ref.float() + (l_ref.float() if has_lora else 0)
    if has_rope:
        x4 = x.view(tokens, heads, dim)
        x = (x4 * cos[:, None] + rotate_half(x4) * sin[:, None]).reshape(tokens, -1)
    ref = x.half()
    ref.backward(grad)

    b = base.clone().requires_grad_()
    l = lora.clone().requires_grad_() if has_lora else None
    out = qkv_prologue(b, l, cos, sin, dim)
    out.backward(grad)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(b.grad, b_ref.grad, rtol=1e-3, atol=1e-3)
    if has_lora:
        torch.testing.assert_close(l.grad, l_ref.grad, rtol=1e-3, atol=1e-3)
