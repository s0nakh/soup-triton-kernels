"""The recomputing MLP against the plain fp16 path over real bitsandbytes NF4 layers."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def nf4_linear(n_in, n_out):
    import bitsandbytes as bnb

    layer = bnb.nn.Linear4bit(n_in, n_out, bias=False, compute_dtype=torch.float16,
                              compress_statistics=True, quant_type="nf4")
    with torch.no_grad():
        layer.weight = bnb.nn.Params4bit(torch.randn(n_out, n_in) * 0.02, requires_grad=False,
                                         compress_statistics=True, quant_type="nf4")
    return layer.cuda()


def test_recomputed_mlp_matches_the_plain_path():
    import bitsandbytes.functional as bnbf
    from solution.mlp import RecomputedMLP, _nf4_weight

    torch.manual_seed(0)
    gate, up, down = nf4_linear(256, 640), nf4_linear(256, 640), nf4_linear(640, 256)
    weights = [_nf4_weight(m) for m in (gate, up, down)]
    assert all(w is not None for w in weights)
    x = torch.randn(2, 37, 256, device="cuda", dtype=torch.float16)
    grad = torch.randn(2, 37, 256, device="cuda", dtype=torch.float16)

    def dense(m):
        return bnbf.dequantize_4bit(m.weight, m.weight.quant_state).half()

    x_ref = x.clone().requires_grad_()
    g, u = x_ref @ dense(gate).t(), x_ref @ dense(up).t()
    ref = (torch.nn.functional.silu(g.float()) * u.float()).half() @ dense(down).t()
    ref.backward(grad)

    x_ours = x.clone().requires_grad_()
    ours = RecomputedMLP.apply(x_ours, *weights)
    ours.backward(grad)
    torch.testing.assert_close(ours, ref, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(x_ours.grad, x_ref.grad, rtol=2e-2, atol=2e-3)
