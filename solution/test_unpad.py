"""Packed (unpadded) Qwen2 forward/backward against the ordinary padded one on a tiny model."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def tiny_qwen2():
    from transformers import Qwen2Config, Qwen2Model

    config = Qwen2Config(vocab_size=1000, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256, attn_implementation="sdpa")
    torch.manual_seed(0)
    return Qwen2Model(config).cuda().train()


def test_packed_forward_and_backward_match_padded():
    from solution.unpad import make_pack, packing, patch_qwen2_unpadded

    patch_qwen2_unpadded()
    model = tiny_qwen2()
    ids = torch.randint(0, 1000, (3, 40), device="cuda")
    mask = torch.ones_like(ids)
    mask[0, 25:] = 0  # right padding of different lengths
    mask[2, 33:] = 0
    keep = mask.bool()

    padded = model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    (padded[keep] ** 2).sum().backward()
    ref_grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()

    pack = make_pack(mask)
    with packing(pack):
        packed = model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    assert packed.shape == (1, int(keep.sum()), 64)
    (packed ** 2).sum().backward()

    torch.testing.assert_close(packed[0], padded[keep], rtol=1e-4, atol=1e-5)
    for got, ref in zip((p.grad for p in model.parameters()), ref_grads):
        torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-5)


def test_dropout_draws_the_same_mask_as_on_the_padded_layout():
    from solution.unpad import make_pack, packing, patch_qwen2_unpadded

    patch_qwen2_unpadded()
    mask = torch.ones(2, 9, dtype=torch.long, device="cuda")
    mask[1, 6:] = 0
    pack = make_pack(mask)
    x = torch.randn(2, 9, 16, device="cuda")
    dropout = torch.nn.Dropout(0.3).train()

    torch.manual_seed(7)
    ref = dropout(x)
    torch.manual_seed(7)
    with packing(pack):
        got = dropout(pack.pack(x))
    torch.testing.assert_close(got, pack.pack(ref))


def test_no_pack_without_padding():
    from solution.unpad import make_pack

    assert make_pack(torch.ones(4, 12, device="cuda")) is None
