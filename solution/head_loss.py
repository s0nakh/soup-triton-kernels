"""Fused LM-head cross-entropy over the 151,936-token vocabulary, without materializing the logits.

The base run keeps the full (batch, seq, vocab) logits alive: fp16 from the LM head, an fp32 copy
for the loss, its gradient, a contiguous copy for TRL's token accuracy and a per-row recompute for
Soup's rewind log. Here the tokens are processed in chunks: a chunk's logits are computed with the
same fp16 matmul the base uses, one Triton kernel reads them once to get everything every consumer
needs (log-sum-exp, target logit, entropy, argmax) and overwrites the chunk in place with the
gradient softmax - one_hot, and a second matmul turns that into the gradient of the hidden states.
Only a (tokens, hidden) gradient survives until backward, so the logits never exist in full.

Idea credit: "Cut Your Losses in Large-Vocabulary Language Models" (Wijmans et al., 2024, Cut
Cross-Entropy) and the chunked fused linear cross-entropy of Liger-Kernel; no code was taken
from either.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100
CHUNK_TOKENS = 512
VOCAB_BLOCK = 4096
NUM_WARPS = 8


@triton.jit
def head_ce_stats_grad_kernel(
    logits_ptr, labels_ptr, loss_ptr, entropy_ptr, correct_ptr,
    n_cols, row_stride, ignore_index,
    BLOCK: tl.constexpr,
):
    """One program per token row of a logits chunk (fp16).

    Pass 1 streams the row once with an online log-sum-exp and also keeps sum(p * z) for the
    entropy and the first index of the maximum for the accuracy. Pass 2 (labelled rows only)
    overwrites the row with softmax(z) - one_hot(label), the gradient of the row's loss.
    """
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * row_stride
    label = tl.load(labels_ptr + row)
    cols = tl.arange(0, BLOCK)

    run_max = -float("inf")
    run_sum = 0.0
    run_dot = 0.0  # sum(exp(z - run_max) * z), rescaled with run_sum
    best_val = -float("inf")
    best_idx = 0
    for start in range(0, n_cols, BLOCK):
        offs = start + cols
        mask = offs < n_cols
        z = tl.load(row_ptr + offs, mask=mask, other=-float("inf")).to(tl.float32)
        block_max = tl.max(z, axis=0)
        # first index of this block's maximum; strict ">" keeps the earliest block on ties
        block_idx = tl.min(tl.where(z == block_max, offs, n_cols), axis=0)
        if block_max > best_val:
            best_val = block_max
            best_idx = block_idx
        new_max = tl.maximum(run_max, block_max)
        rescale = tl.exp(run_max - new_max)
        e = tl.where(mask, tl.exp(z - new_max), 0.0)
        run_sum = run_sum * rescale + tl.sum(e, axis=0)
        run_dot = run_dot * rescale + tl.sum(tl.where(mask, e * z, 0.0), axis=0)
        run_max = new_max

    lse = run_max + tl.log(run_sum)
    tl.store(entropy_ptr + row, lse - run_dot / run_sum)

    labelled = label != ignore_index
    if labelled:
        target = tl.load(row_ptr + label).to(tl.float32)
        tl.store(loss_ptr + row, lse - target)
        tl.store(correct_ptr + row, (best_idx == label).to(tl.int32))
        for start in range(0, n_cols, BLOCK):
            offs = start + cols
            mask = offs < n_cols
            z = tl.load(row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            grad = tl.exp(z - lse) - tl.where(offs == label, 1.0, 0.0)
            tl.store(row_ptr + offs, grad.to(logits_ptr.dtype.element_ty), mask=mask)
    else:
        tl.store(loss_ptr + row, 0.0)
        tl.store(correct_ptr + row, 0)


@dataclass(frozen=True)
class HeadStep:
    """What a fused step leaves for TRL's metrics and Soup's rewind log (no logits)."""

    token_entropy: torch.Tensor  # (batch, seq) fp32, entropy of every position's distribution
    token_loss: torch.Tensor  # (batch, seq) fp32, cross-entropy of labels[:, t + 1], 0 if none
    correct: torch.Tensor  # scalar: labelled positions whose argmax is the label
    labelled: torch.Tensor  # scalar: positions with a label


def shift_labels(labels):
    """labels[:, t + 1] at position t and IGNORE_INDEX at the last one, as ForCausalLMLoss does."""
    return torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]


class FusedHeadLoss(torch.autograd.Function):
    """loss = sum of token cross-entropies / divisor, with d(loss)/d(hidden) computed in forward."""

    @staticmethod
    def forward(ctx, hidden, weight16, targets, divisor):
        batch, seq, width = hidden.shape
        flat = hidden.reshape(-1, width)
        flat_targets = targets.reshape(-1).contiguous()
        n_tokens, n_vocab = flat.shape[0], weight16.shape[0]

        token_loss = torch.empty(batch, seq, device=hidden.device, dtype=torch.float32)
        token_entropy = torch.empty_like(token_loss)
        correct = torch.empty(batch, seq, device=hidden.device, dtype=torch.int32)
        flat_loss, flat_entropy, flat_correct = (t.view(-1) for t in (token_loss, token_entropy,
                                                                      correct))
        grad_hidden = torch.zeros(n_tokens, width, device=hidden.device, dtype=torch.float32)

        for start in range(0, n_tokens, CHUNK_TOKENS):
            stop = min(start + CHUNK_TOKENS, n_tokens)
            logits = flat[start:stop].to(torch.float16) @ weight16.t()  # the base's fp16 head
            head_ce_stats_grad_kernel[(stop - start,)](
                logits, flat_targets[start:stop], flat_loss[start:stop],
                flat_entropy[start:stop], flat_correct[start:stop],
                n_vocab, logits.stride(0), IGNORE_INDEX,
                BLOCK=VOCAB_BLOCK, num_warps=NUM_WARPS,
            )
            rows = (flat_targets[start:stop] != IGNORE_INDEX).nonzero().squeeze(1)
            if rows.numel():
                grad_hidden[start + rows] = (logits.index_select(0, rows) @ weight16).float()

        grad_hidden.div_(divisor)
        loss = token_loss.sum() / divisor
        ctx.save_for_backward(grad_hidden)
        ctx.hidden_shape, ctx.hidden_dtype = hidden.shape, hidden.dtype
        ctx.mark_non_differentiable(token_entropy, token_loss)
        n_correct = correct.sum()
        n_labelled = (flat_targets != IGNORE_INDEX).sum()
        return loss, token_entropy, token_loss, n_correct, n_labelled

    @staticmethod
    def backward(ctx, grad_loss, *unused):
        (grad_hidden,) = ctx.saved_tensors
        grad = (grad_hidden * grad_loss).view(ctx.hidden_shape).to(ctx.hidden_dtype)
        return grad, None, None, None


_WEIGHT16 = {}


def fp16_weight(weight):
    """The LM head weight in fp16, as autocast would cast it; cached because it is frozen."""
    if weight.dtype == torch.float16:
        return weight
    key = (weight.data_ptr(), weight._version, tuple(weight.shape))
    cached = _WEIGHT16.get("value")
    if _WEIGHT16.get("key") != key:
        cached = weight.detach().to(torch.float16)
        _WEIGHT16.clear()
        _WEIGHT16.update(key=key, value=cached)
    return cached


def fused_head_loss(hidden, weight, labels, num_items_in_batch=None):
    """(loss, HeadStep) for a causal LM head, matching ForCausalLMLoss's reduction."""
    targets = shift_labels(labels)
    if num_items_in_batch is None:
        divisor = (targets != IGNORE_INDEX).sum().clamp(min=1).to(torch.float32)
    elif torch.is_tensor(num_items_in_batch):
        divisor = num_items_in_batch.to(device=hidden.device, dtype=torch.float32)
    else:
        divisor = torch.tensor(float(num_items_in_batch), device=hidden.device)
    loss, entropy, token_loss, correct, labelled = FusedHeadLoss.apply(
        hidden, fp16_weight(weight), targets, divisor)
    return loss, HeadStep(token_entropy=entropy, token_loss=token_loss, correct=correct,
                          labelled=labelled)
