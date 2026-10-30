"""Compile our Triton kernels in a background thread while Soup loads the model.

Every measured run starts with empty compiler caches, so the first launch of each kernel stops
the first training step for a compile (about 3 s for all of ours on a T4). `apply()` runs before
Soup loads the tokenizer, the dataset and the 4-bit model, which keeps the GPU idle for many
seconds. Starting the same compiles then, with the argument types and specializations the
training step will use, lets the first step find them ready. Nothing is cached across runs and
nothing is launched: `warmup` only compiles.
"""

import threading

import torch

HIDDEN = 1536
VOCAB = 151936


def _compile_all():
    from .head_loss import IGNORE_INDEX, NUM_WARPS, VOCAB_BLOCK, head_ce_stats_grad_kernel
    from .nf4 import BYTES_PER_PROGRAM, NUM_WARPS as NF4_WARPS, nf4_dequant_kernel
    from .rmsnorm import _num_warps, rmsnorm_bwd_kernel, rmsnorm_fwd_kernel

    cuda = torch.device("cuda")
    f32, f16 = torch.float32, torch.float16
    rows = torch.empty(16, HIDDEN, device=cuda, dtype=f32)
    weight = torch.empty(HIDDEN, device=cuda, dtype=f32)
    rstd = torch.empty(16, device=cuda, dtype=f32)
    block = 2048
    rmsnorm_fwd_kernel.warmup(rows, weight, rows, rstd, HIDDEN, 1e-6, BLOCK=block,
                              CAST_TO_INPUT=False, num_warps=_num_warps(block), grid=(1,))
    rmsnorm_bwd_kernel.warmup(rows, rows, weight, rstd, rows, HIDDEN, BLOCK=block,
                              num_warps=_num_warps(block), grid=(1,))

    n_bytes = HIDDEN * HIDDEN // 2
    packed = torch.empty(n_bytes, device=cuda, dtype=torch.uint8)
    absmax_q = torch.empty(n_bytes // 32, device=cuda, dtype=torch.uint8)
    table = torch.empty(256, device=cuda, dtype=f32)
    weights16 = torch.empty(2 * n_bytes, device=cuda, dtype=f16)
    nf4_dequant_kernel.warmup(packed, absmax_q, table, table, table, table, weights16, n_bytes,
                              BLOCK=BYTES_PER_PROGRAM, BYTES_PER_ABSMAX=32, ABSMAX_PER_SCALE=256,
                              num_warps=NF4_WARPS, grid=(1,))

    from .rope import qkv_prologue_bwd_kernel, qkv_prologue_fwd_kernel

    qkv = torch.empty(16, HIDDEN, device=cuda, dtype=f16)
    angles = torch.empty(16, 128, device=cuda, dtype=f32)
    for has_lora, has_rope in ((True, True), (False, True), (True, False)):
        table = angles if has_rope else qkv  # without rotary the kernel gets the base tensor
        qkv_prologue_fwd_kernel.warmup(qkv, qkv, table, table, qkv, 12, HALF=64,
                                       HAS_LORA=has_lora, HAS_ROPE=has_rope, num_warps=1,
                                       grid=(1, 1))
    for has_rope in (True, False):
        table = angles if has_rope else qkv
        qkv_prologue_bwd_kernel.warmup(qkv, table, table, qkv, 12, HALF=64, HAS_ROPE=has_rope,
                                       num_warps=1, grid=(1, 1))

    logits = torch.empty(16, VOCAB, device=cuda, dtype=f16)
    labels = torch.empty(16, device=cuda, dtype=torch.int64)
    per_row = torch.empty(16, device=cuda, dtype=f32)
    correct = torch.empty(16, device=cuda, dtype=torch.int32)
    head_ce_stats_grad_kernel.warmup(logits, labels, per_row, per_row, correct, VOCAB, VOCAB,
                                     IGNORE_INDEX, BLOCK=VOCAB_BLOCK, num_warps=NUM_WARPS,
                                     grid=(1,))


def _run():
    try:
        _compile_all()
    except Exception:  # noqa: BLE001 - a failed warm-up only means the first step compiles
        pass


def start_background_compile():
    if not torch.cuda.is_available():
        return None
    thread = threading.Thread(target=_run, name="mindstream-triton-warmup", daemon=True)
    thread.start()
    return thread
