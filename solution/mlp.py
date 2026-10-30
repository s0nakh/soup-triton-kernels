"""Qwen2 MLP: fp16 intermediates, and gate/up recomputed in backward instead of kept.

1. bitsandbytes' Linear4bit casts its fp16 GEMM result back to the input dtype. The MLP input is
   the fp32 RMSNorm output, so gate and up would come back as fp32 and be kept in fp32 for the
   SwiGLU backward. Casting the input to fp16 once (the cast every Linear4bit does inside) gives
   the same values in half the bytes.
2. gate and up (tokens x 8960 each) are most of the activation memory of a step, and the peak
   memory of the run is set by its largest batch alone. For batches above RECOMPUTE_MIN_TOKENS
   real tokens the MLP is one autograd function that keeps only its fp16 input: backward
   rebuilds the three frozen NF4 weights with our dequant kernel and recomputes gate and up with
   the same GEMMs (same numbers, two extra GEMMs per layer). Smaller batches, most of the run,
   keep gate and up and pay nothing.

SwiGLU itself uses the organizers' example kernels (allowed; they do not count as ours).
"""

import torch
import torch.nn.functional as F

from .nf4 import dequantize_nf4, supported
from .swiglu import BLOCK, example_swiglu_bwd_kernel, swiglu

import triton

RECOMPUTE_MIN_TOKENS = 800  # real tokens in the batch; the large batches set the memory peak


def _nf4_weight(linear):
    """(packed codes, quant state) of a bitsandbytes Linear4bit we can dequantize, else None."""
    weight = getattr(linear, "weight", None)
    state = getattr(weight, "quant_state", None)
    if (getattr(linear, "bias", None) is not None or state is None or not supported(state)
            or getattr(linear, "compute_dtype", None) != torch.float16):
        return None
    return weight, state


class RecomputedMLP(torch.autograd.Function):
    """down(silu(gate(x)) * up(x)) with only x (fp16) kept for backward."""

    @staticmethod
    def forward(ctx, x16, gate, up, down):
        (gw, gs), (uw, us), (dw, ds) = gate, up, down
        g = F.linear(x16, dequantize_nf4(gw, gs))
        u = F.linear(x16, dequantize_nf4(uw, us))
        out = F.linear(swiglu(g, u), dequantize_nf4(dw, ds))
        ctx.save_for_backward(x16)
        ctx.weights = (gate, up, down)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x16,) = ctx.saved_tensors
        (gw, gs), (uw, us), (dw, ds) = ctx.weights
        w_gate, w_up = dequantize_nf4(gw, gs), dequantize_nf4(uw, us)
        g = F.linear(x16, w_gate).contiguous()
        u = F.linear(x16, w_up).contiguous()
        grad_act = torch.matmul(grad_out.to(torch.float16), dequantize_nf4(dw, ds)).contiguous()
        grad_g, grad_u = torch.empty_like(g), torch.empty_like(u)
        n = g.numel()
        example_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](grad_act, g, u, grad_g, grad_u, n,
                                                           BLOCK=BLOCK)
        width = w_gate.shape[1]
        grad_x = torch.mm(grad_g.view(-1, grad_g.shape[-1]), w_gate)
        grad_x.addmm_(grad_u.view(-1, grad_u.shape[-1]), w_up)
        return grad_x.view(*x16.shape[:-1], width), None, None, None


def patch_qwen2_mlp_fp16():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2MLP
    if getattr(cls.forward, "_mindstream", False):
        return
    previous = cls.forward

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if act != "silu" or not x.is_cuda:
            return previous(self, x)
        weights = [_nf4_weight(m) for m in (self.gate_proj, self.up_proj, self.down_proj)]
        x16 = x.to(torch.float16)
        tokens = x.numel() // x.shape[-1]
        if (tokens > RECOMPUTE_MIN_TOKENS and torch.is_grad_enabled()
                and all(w is not None for w in weights)):
            return RecomputedMLP.apply(x16, *weights)
        if (getattr(self.gate_proj, "compute_dtype", None) != torch.float16
                or getattr(self.up_proj, "compute_dtype", None) != torch.float16):
            return previous(self, x)
        return self.down_proj(swiglu(self.gate_proj(x16), self.up_proj(x16)))

    forward._mindstream = True
    cls.forward = forward
