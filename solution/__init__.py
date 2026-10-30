"""MindStream AI solution for the Soup Kernel Challenge.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer.

- `head_loss.py`: our fused LM-head cross-entropy (Triton): loss, entropy, token accuracy and
  per-row losses from one pass over each chunk of logits, which never exist in full.
- `patches.py`: routes Qwen2's labelled training forward and TRL's metric logging through it.
- `swiglu.py`: the organizers' example kernel, kept running (it does not count as ours).
"""

from .patches import patch_qwen2_causal_lm, patch_sft_trainer
from .swiglu import patch_qwen2_mlp


def apply():
    patch_qwen2_mlp()  # the example: does not count as our own kernel
    patch_qwen2_causal_lm()
    patch_sft_trainer()
