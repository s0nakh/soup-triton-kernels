"""Fused Qwen2 RMSNorm (frozen weight) in Triton, forward and backward.

PyTorch runs Qwen2RMSNorm as six separate kernels (cast, pow, mean, add + rsqrt, two multiplies)
and twice as many in backward, each a full pass over the (tokens, 1536) fp32 activations. Here one
program per row does the forward in one read and one write and keeps only the per-row 1/rms for
backward; backward is one pass as well. LoRA leaves the norm weights frozen, so no weight
gradient is needed; a trainable weight falls back to the original module.

The math follows Qwen2RMSNorm exactly: y = weight * (x * rsqrt(mean(x^2) + eps)).to(x.dtype).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, n_cols, eps,
                       BLOCK: tl.constexpr, CAST_TO_INPUT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    normed = x * rstd
    if CAST_TO_INPUT:  # Qwen2RMSNorm casts back to the input dtype before the weight
        normed = normed.to(x_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * n_cols + cols, (normed * w).to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def rmsnorm_bwd_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, n_cols, BLOCK: tl.constexpr):
    """dx = rstd * (g - x_hat * mean(g * x_hat)), g = dy * weight, x_hat = x * rstd."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    dy = tl.load(dy_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    g = dy * w
    x_hat = x * rstd
    dx = rstd * (g - x_hat * (tl.sum(g * x_hat, axis=0) / n_cols))
    tl.store(dx_ptr + row * n_cols + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def _num_warps(block):
    return 8 if block >= 2048 else 4


class FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        n_rows, n_cols = flat.shape
        out_dtype = torch.promote_types(weight.dtype, x.dtype)
        y = torch.empty((n_rows, n_cols), dtype=out_dtype, device=x.device)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
        block = triton.next_power_of_2(n_cols)
        rmsnorm_fwd_kernel[(n_rows,)](flat, weight, y, rstd, n_cols, eps, BLOCK=block,
                                      CAST_TO_INPUT=x.dtype != torch.float32,
                                      num_warps=_num_warps(block))
        ctx.save_for_backward(flat, weight, rstd)
        ctx.shape, ctx.x_dtype = shape, x.dtype
        return y.view(*shape[:-1], n_cols)

    @staticmethod
    def backward(ctx, grad_y):
        flat, weight, rstd = ctx.saved_tensors
        n_rows, n_cols = flat.shape
        dy = grad_y.reshape(n_rows, n_cols).contiguous()
        dx = torch.empty((n_rows, n_cols), dtype=ctx.x_dtype, device=flat.device)
        block = triton.next_power_of_2(n_cols)
        rmsnorm_bwd_kernel[(n_rows,)](dy, flat, weight, rstd, dx, n_cols, BLOCK=block,
                                      num_warps=_num_warps(block))
        return dx.view(ctx.shape), None, None


def rmsnorm(x, weight, eps):
    return FusedRMSNorm.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    if getattr(cls.forward, "_mindstream", False):
        return
    original = cls.forward

    def forward(self, hidden_states):
        weight = self.weight
        if (not hidden_states.is_cuda or weight.requires_grad or not weight.is_cuda
                or hidden_states.dtype not in (torch.float32, torch.float16)):
            return original(self, hidden_states)
        return rmsnorm(hidden_states, weight, self.variance_epsilon)

    forward._mindstream = True
    cls.forward = forward
