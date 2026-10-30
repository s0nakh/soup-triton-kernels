"""Our NF4 dequantization and 4-bit linear against bitsandbytes itself."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def quantized(n_out, n_in, seed=0):
    import bitsandbytes.functional as bnbf

    torch.manual_seed(seed)
    weight = (torch.randn(n_out, n_in, device="cuda") * 0.02).half()
    packed, state = bnbf.quantize_4bit(weight, blocksize=64, compress_statistics=True,
                                       quant_type="nf4", quant_storage=torch.uint8)
    return packed, state


@pytest.mark.parametrize("shape", [(1536, 1536), (8960, 1536), (256, 1536), (1536, 8960)])
def test_dequant_matches_bitsandbytes(shape):
    import bitsandbytes.functional as bnbf
    from solution.nf4 import dequantize_nf4, supported

    packed, state = quantized(*shape)
    assert supported(state)
    ref = bnbf.dequantize_4bit(packed, state)
    ours = dequantize_nf4(packed, state)
    assert ours.shape == ref.shape and ours.dtype == ref.dtype
    exact = (ours == ref).float().mean().item()
    print(f"bit-exact fraction {exact:.6f}")
    torch.testing.assert_close(ours, ref, rtol=1e-3, atol=1e-6)
    assert exact > 0.999


def test_linear_forward_and_input_grad_match_bitsandbytes():
    import bitsandbytes as bnb
    from solution.nf4 import NF4Linear

    packed, state = quantized(1536, 1536, seed=1)
    bias = (torch.randn(1536, device="cuda") * 0.1).half()
    x = torch.randn(4, 77, 1536, device="cuda", dtype=torch.float16)
    grad = torch.randn(4, 77, 1536, device="cuda", dtype=torch.float16)

    x_ref = x.clone().requires_grad_()
    ref = bnb.matmul_4bit(x_ref, packed.t(), quant_state=state, bias=bias)
    ref.backward(grad)
    x_ours = x.clone().requires_grad_()
    ours = NF4Linear.apply(x_ours, packed.t(), bias, state)
    ours.backward(grad)
    torch.testing.assert_close(ours, ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x_ours.grad, x_ref.grad, rtol=2e-3, atol=2e-3)
