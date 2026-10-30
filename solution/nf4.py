"""Our NF4 dequantization in Triton for the 4-bit linear layers, forward and backward.

bitsandbytes reconstructs every frozen weight twice per step (forward, and backward for the input
gradient) with two CUDA kernels: one turns the 8-bit double-quantized absmax back into fp32, the
second unpacks the 4-bit NF4 codes. This kernel does both in one launch: it reads each packed
byte once, rebuilds that block's absmax on the fly (code2[absmax_q] * absmax2 + offset) and
writes the two fp16 weights. The matmul stays the same cuBLAS fp16 GEMM bitsandbytes uses on this
path (dequantize + F.linear), so the layer computes the same thing.

Format (bitsandbytes 0.50.2, NF4, compress_statistics=True): two codes per byte, high nibble
first; absmax per 64 weights stored as uint8 codes into a 256-entry table, with an fp32 scale per
256 absmax values and a global offset. Format reference: Dettmers et al., "QLoRA" (2023).
"""

import torch
import triton
import triton.language as tl

BYTES_PER_PROGRAM = 1024
NUM_WARPS = 4


@triton.jit
def nf4_dequant_kernel(packed_ptr, absmax_q_ptr, absmax2_ptr, code2_ptr, nf4_ptr, offset_ptr,
                       out_ptr, n_bytes,
                       BLOCK: tl.constexpr, BYTES_PER_ABSMAX: tl.constexpr,
                       ABSMAX_PER_SCALE: tl.constexpr):
    """Each program unpacks BLOCK bytes into 2 * BLOCK fp16 weights."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_bytes
    byte = tl.load(packed_ptr + offs, mask=mask, other=0).to(tl.int32)
    block = offs // BYTES_PER_ABSMAX
    absmax_code = tl.load(absmax_q_ptr + block, mask=mask, other=0).to(tl.int32)
    scale = tl.load(absmax2_ptr + block // ABSMAX_PER_SCALE, mask=mask, other=0.0)
    absmax = tl.load(code2_ptr + absmax_code) * scale + tl.load(offset_ptr)
    high = tl.load(nf4_ptr + (byte >> 4)) * absmax
    low = tl.load(nf4_ptr + (byte & 15)) * absmax
    pairs = tl.reshape(tl.join(high, low), (2 * BLOCK,))  # high nibble first
    out_offs = 2 * pid * BLOCK + tl.arange(0, 2 * BLOCK)
    tl.store(out_ptr + out_offs, pairs.to(out_ptr.dtype.element_ty), mask=out_offs < 2 * n_bytes)


def supported(quant_state):
    state2 = getattr(quant_state, "state2", None)
    return (quant_state is not None and getattr(quant_state, "quant_type", None) == "nf4"
            and getattr(quant_state, "nested", False) and state2 is not None
            and state2.blocksize == 256 and quant_state.blocksize % 2 == 0
            and len(quant_state.shape) == 2 and quant_state.absmax.dtype == torch.uint8)


def _offset_tensor(quant_state, device):
    offset = quant_state.offset
    if not torch.is_tensor(offset):
        offset = torch.tensor(float(offset))
    return offset.to(device=device, dtype=torch.float32).reshape(1)


def dequantize_nf4(packed, quant_state, dtype=torch.float16):
    """The (N, K) weight from bitsandbytes' packed NF4 codes and nested absmax."""
    out = torch.empty(tuple(quant_state.shape), dtype=dtype, device=packed.device)
    n_bytes = out.numel() // 2
    state2 = quant_state.state2
    nf4_dequant_kernel[(triton.cdiv(n_bytes, BYTES_PER_PROGRAM),)](
        packed.reshape(-1), quant_state.absmax, state2.absmax, state2.code,
        quant_state.code.to(device=packed.device, dtype=torch.float32),
        _offset_tensor(quant_state, packed.device), out, n_bytes,
        BLOCK=BYTES_PER_PROGRAM, BYTES_PER_ABSMAX=quant_state.blocksize // 2,
        ABSMAX_PER_SCALE=state2.blocksize, num_warps=NUM_WARPS,
    )
    return out


class NF4Linear(torch.autograd.Function):
    """y = A @ dequant(B).T + bias, as bitsandbytes' MatMul4Bit on its dequantize path."""

    @staticmethod
    def forward(ctx, A, B, bias, quant_state):
        weight = dequantize_nf4(B, quant_state, A.dtype)
        ctx.quant_state, ctx.bias_dtype = quant_state, None if bias is None else bias.dtype
        ctx.save_for_backward(B)
        return torch.nn.functional.linear(A, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        (B,) = ctx.saved_tensors
        need_a, _, need_bias, _ = ctx.needs_input_grad
        grad_a = grad_bias = None
        if need_bias:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0, dtype=ctx.bias_dtype)
        if need_a:
            weight = dequantize_nf4(B, ctx.quant_state, grad_output.dtype)
            grad_a = torch.matmul(grad_output, weight)
        return grad_a, None, grad_bias, None


def patch_bnb_matmul_4bit():
    """Route bitsandbytes' 4-bit matmul through our kernel when the weight is NF4 with nested
    absmax and the input is fp16 on CUDA; anything else keeps the original."""
    import bitsandbytes

    if getattr(bitsandbytes.matmul_4bit, "_mindstream", False):
        return
    original = bitsandbytes.matmul_4bit

    def matmul_4bit(A, B, quant_state, out=None, bias=None):
        if (out is None and A.is_cuda and A.dtype == torch.float16 and A.numel() > 0
                and supported(quant_state) and A.shape[-1] == quant_state.shape[1]):
            return NF4Linear.apply(A, B, bias, quant_state)
        return original(A, B, quant_state, out=out, bias=bias)

    matmul_4bit._mindstream = True
    bitsandbytes.matmul_4bit = matmul_4bit
