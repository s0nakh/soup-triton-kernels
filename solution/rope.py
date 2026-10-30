"""Fused attention prologue in Triton: LoRA merge + rotary embedding + cast, forward and backward.

In the base run each of q, k, v goes through a chain of small ops before attention: the 4-bit
GEMM result is cast to fp32 and cloned, the LoRA delta (fp16) is added in fp32, q and k get the
rotary embedding as six separate fp32 ops (two multiplies, a slice-and-concatenate rotate_half,
a negate, a multiply, an add), and autocast casts all three back to fp16 for attention.

Here one program per (token, head) does all of it in registers: it loads the fp16 base output
and the fp16 LoRA delta, adds them in fp32, applies the rotation in fp32 and stores the fp16
tensor attention reads. The numbers are the base's: the same fp32 sum and rotation, rounded to
fp16 at the same point. Backward is the transposed rotation in one pass, and its result is the
gradient of both the base output and the LoRA delta.

Rotary embedding: Su et al., "RoFormer" (2021), in the rotate_half form Qwen2 uses.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def qkv_prologue_fwd_kernel(base_ptr, lora_ptr, cos_ptr, sin_ptr, out_ptr, n_heads,
                            HALF: tl.constexpr, HAS_LORA: tl.constexpr, HAS_ROPE: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    cols = tl.arange(0, HALF)
    row = (token * n_heads + head) * (2 * HALF)
    a = tl.load(base_ptr + row + cols).to(tl.float32)
    b = tl.load(base_ptr + row + HALF + cols).to(tl.float32)
    if HAS_LORA:
        a += tl.load(lora_ptr + row + cols).to(tl.float32)
        b += tl.load(lora_ptr + row + HALF + cols).to(tl.float32)
    if HAS_ROPE:
        c1 = tl.load(cos_ptr + token * 2 * HALF + cols)
        c2 = tl.load(cos_ptr + token * 2 * HALF + HALF + cols)
        s1 = tl.load(sin_ptr + token * 2 * HALF + cols)
        s2 = tl.load(sin_ptr + token * 2 * HALF + HALF + cols)
        a, b = a * c1 - b * s1, b * c2 + a * s2  # x * cos + rotate_half(x) * sin
    tl.store(out_ptr + row + cols, a.to(out_ptr.dtype.element_ty))
    tl.store(out_ptr + row + HALF + cols, b.to(out_ptr.dtype.element_ty))


@triton.jit
def qkv_prologue_bwd_kernel(grad_ptr, cos_ptr, sin_ptr, dx_ptr, n_heads,
                            HALF: tl.constexpr, HAS_ROPE: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    cols = tl.arange(0, HALF)
    row = (token * n_heads + head) * (2 * HALF)
    ga = tl.load(grad_ptr + row + cols).to(tl.float32)
    gb = tl.load(grad_ptr + row + HALF + cols).to(tl.float32)
    if HAS_ROPE:
        c1 = tl.load(cos_ptr + token * 2 * HALF + cols)
        c2 = tl.load(cos_ptr + token * 2 * HALF + HALF + cols)
        s1 = tl.load(sin_ptr + token * 2 * HALF + cols)
        s2 = tl.load(sin_ptr + token * 2 * HALF + HALF + cols)
        ga, gb = ga * c1 + gb * s2, gb * c2 - ga * s1  # the transposed rotation
    tl.store(dx_ptr + row + cols, ga.to(dx_ptr.dtype.element_ty))
    tl.store(dx_ptr + row + HALF + cols, gb.to(dx_ptr.dtype.element_ty))


class QKVPrologue(torch.autograd.Function):
    """out = rope(base + lora) in fp16, for (tokens, heads * head_dim) inputs."""

    @staticmethod
    def forward(ctx, base, lora, cos, sin, head_dim):
        tokens = base.shape[0]
        n_heads = base.shape[1] // head_dim
        base = base.contiguous()
        lora_arg = base if lora is None else lora.contiguous()
        has_rope = cos is not None
        cos_arg = cos.contiguous() if has_rope else base
        sin_arg = sin.contiguous() if has_rope else base
        out = torch.empty_like(base, dtype=torch.float16)
        qkv_prologue_fwd_kernel[(tokens, n_heads)](
            base, lora_arg, cos_arg, sin_arg, out, n_heads, HALF=head_dim // 2,
            HAS_LORA=lora is not None, HAS_ROPE=has_rope, num_warps=1)
        ctx.save_for_backward(cos_arg, sin_arg)
        ctx.meta = (n_heads, head_dim, has_rope, lora is not None, base.dtype)
        return out

    @staticmethod
    def backward(ctx, grad):
        cos, sin = ctx.saved_tensors
        n_heads, head_dim, has_rope, has_lora, base_dtype = ctx.meta
        grad = grad.contiguous()
        dx = torch.empty(grad.shape, dtype=base_dtype, device=grad.device)
        qkv_prologue_bwd_kernel[(grad.shape[0], n_heads)](
            grad, cos, sin, dx, n_heads, HALF=head_dim // 2, HAS_ROPE=has_rope, num_warps=1)
        return dx, (dx if has_lora else None), None, None, None


def qkv_prologue(base, lora, cos, sin, head_dim):
    return QKVPrologue.apply(base, lora, cos, sin, head_dim)
