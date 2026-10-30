"""MindStream AI solution for the Soup Kernel Challenge.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer.

- `head_loss.py`: our fused LM-head cross-entropy (Triton): loss, entropy, token accuracy and
  per-row losses from one pass over each chunk of logits, which never exist in full.
- `patches.py`: routes Qwen2's labelled training forward and TRL's metric logging through it.
- `nf4.py`: our NF4 dequantization (Triton) for every 4-bit linear, forward and backward: absmax
  and weights rebuilt in one launch instead of bitsandbytes' two.
- `rmsnorm.py`: fused Qwen2 RMSNorm forward and backward (Triton).
- `mlp.py`: keeps the MLP's gate/up in fp16, the dtype its matmuls produce (same values).
- `unpad.py`: drops the padding inside each batch: token-wise ops run on real tokens only,
  attention unpacks to the base's padded layout and mask.
- `warmup.py`: compiles our kernels in a background thread while Soup loads the model.
- `tied_embedding.py`: stores the tied embedding / LM head in fp16, bit-exactly (the fp32 upcast
  carries no extra information on a T4), so neither the fp32 copy nor autocast's fp16 copy exists.
- `swiglu.py`: the organizers' example kernel, kept running (it does not count as ours).
"""

from .mlp import patch_qwen2_mlp_fp16
from .nf4 import patch_bnb_matmul_4bit
from .patches import patch_qwen2_causal_lm, patch_sft_trainer
from .rmsnorm import patch_qwen2_rmsnorm
from .swiglu import patch_qwen2_mlp
from .tied_embedding import patch_soup_train
from .unpad import patch_qwen2_unpadded
from .warmup import start_background_compile


def apply():
    start_background_compile()  # overlaps the first-step Triton compiles with model loading
    patch_qwen2_mlp()  # the example: does not count as our own kernel
    patch_qwen2_mlp_fp16()
    patch_bnb_matmul_4bit()
    patch_qwen2_rmsnorm()
    patch_qwen2_unpadded()
    patch_qwen2_causal_lm()
    patch_sft_trainer()
    patch_soup_train()
