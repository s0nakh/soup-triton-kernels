"""fp16 storage of the tied embedding must give the same lookups, bit for bit."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


class TinyLM(torch.nn.Module):
    def __init__(self, vocab=1000, width=64):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, width)
        self.lm_head = torch.nn.Linear(width, vocab, bias=False)
        self.lm_head.weight = self.embed.weight  # tied, as in Qwen2.5-1.5B

    def get_input_embeddings(self):
        return self.embed


def test_fp16_storage_is_bit_exact_and_stays_tied():
    from solution.tied_embedding import store_tied_embedding_fp16

    model = TinyLM().cuda()
    with torch.no_grad():  # fp32 weights holding fp16 values, as after peft's upcast
        model.embed.weight.copy_(model.embed.weight.half().float())
    model.embed.weight.requires_grad_(False)
    ids = torch.randint(0, 1000, (3, 17), device="cuda")
    before = model.embed(ids).clone()

    assert store_tied_embedding_fp16(model)
    after = model.embed(ids)
    assert model.embed.weight.dtype == torch.float16
    assert model.lm_head.weight is model.embed.weight
    assert after.dtype == torch.float32 and torch.equal(after, before)


def test_keeps_fp32_when_values_are_not_fp16():
    from solution.tied_embedding import store_tied_embedding_fp16

    model = TinyLM().cuda()
    model.embed.weight.requires_grad_(False)  # random fp32 values: fp16 would lose bits
    assert not store_tied_embedding_fp16(model)
    assert model.embed.weight.dtype == torch.float32
