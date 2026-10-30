"""The fused LM-head loss against plain PyTorch: loss, gradient, entropy, per-token loss, accuracy."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def reference(hidden, weight, labels, divisor):
    logits = (hidden.to(torch.float16) @ weight.to(torch.float16).t()).float()
    targets = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    token_loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), ignore_index=-100, reduction="none"
    ).view_as(targets)
    logp = torch.log_softmax(logits, dim=-1)
    entropy = -(logp.exp() * logp).sum(-1)
    mask = targets != -100
    correct = ((logits.argmax(-1) == targets) & mask).sum()
    return token_loss.sum() / divisor, entropy, token_loss, correct, mask.sum()


def make_batch(batch, seq, width, vocab, seed=0):
    torch.manual_seed(seed)
    hidden = torch.randn(batch, seq, width, device="cuda") * 0.5
    weight = torch.randn(vocab, width, device="cuda") * 0.05
    labels = torch.randint(0, vocab, (batch, seq), device="cuda")
    labels[:, : seq // 3] = -100  # an unsupervised prompt
    labels[-1, -seq // 4:] = -100  # padding at the end of the last row
    return hidden, weight, labels


@pytest.mark.parametrize("shape", [(2, 37, 64, 1000), (4, 300, 128, 9001), (2, 520, 64, 4096)])
@pytest.mark.parametrize("given_divisor", [False, True])
def test_fused_head_loss_matches_pytorch(shape, given_divisor):
    from solution.head_loss import fused_head_loss

    hidden, weight, labels = make_batch(*shape)
    n_labelled = int((labels[:, 1:] != -100).sum())
    num_items = torch.tensor(n_labelled + 7, device="cuda") if given_divisor else None
    divisor = float(n_labelled + 7) if given_divisor else float(n_labelled)

    ref_hidden = hidden.clone().requires_grad_()
    ref_loss, ref_entropy, ref_token_loss, ref_correct, ref_labelled = reference(
        ref_hidden, weight, labels, divisor)
    (ref_loss * 3.0).backward()

    fused_hidden = hidden.clone().requires_grad_()
    loss, step = fused_head_loss(fused_hidden, weight, labels, num_items)
    (loss * 3.0).backward()

    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(fused_hidden.grad, ref_hidden.grad, rtol=2e-3, atol=2e-5)
    torch.testing.assert_close(step.token_entropy, ref_entropy, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(step.token_loss, ref_token_loss, rtol=1e-4, atol=1e-4)
    assert int(step.labelled) == int(ref_labelled)
    assert abs(int(step.correct) - int(ref_correct)) <= 1  # an fp16 tie may break either way


def test_argmax_takes_the_first_maximum_like_torch():
    from solution.head_loss import fused_head_loss

    hidden = torch.zeros(1, 3, 8, device="cuda")
    weight = torch.zeros(5000, 8, device="cuda")  # every logit is 0: argmax is index 0
    labels = torch.tensor([[-100, 0, 4321]], device="cuda")
    _, step = fused_head_loss(hidden, weight, labels)
    assert int(step.correct) == 1 and int(step.labelled) == 2
