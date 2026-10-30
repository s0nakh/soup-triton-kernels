"""Keep the tied embedding / LM head in fp16 storage instead of fp32, bit-exactly.

On a T4 Soup loads the base in fp16, then peft's `prepare_model_for_kbit_training` upcasts the
tied `embed_tokens` / `lm_head` weight to fp32 (151,936 x 1,536 = 934 MB). The upcast adds no
information: every value is still an fp16 value. The base also makes a 467 MB fp16 copy of it on
every step (autocast casts the LM head weight for its fp16 matmul).

Here the weight goes back to fp16 storage (467 MB) after a check that this loses nothing, and the
embedding lookup returns fp32 again, so every number the model computes stays the same. The fused
LM head reuses the same fp16 tensor, so neither copy exists any more.
"""

import torch


def fp16_is_exact(weight):
    """True when every value of an fp32 tensor is an fp16 value (the cast back loses nothing)."""
    for rows in weight.detach().split(8192):
        if not torch.equal(rows, rows.to(torch.float16).to(rows.dtype)):
            return False
    return True


def store_tied_embedding_fp16(model):
    """Store the input embedding weight in fp16 when that is bit-exact; returns True if done."""
    get_embeddings = getattr(model, "get_input_embeddings", None)
    embedding = get_embeddings() if callable(get_embeddings) else None
    weight = getattr(embedding, "weight", None)
    if (not isinstance(embedding, torch.nn.Embedding) or not isinstance(weight, torch.Tensor)
            or weight.dtype != torch.float32 or weight.requires_grad or not weight.is_cuda
            or getattr(embedding.forward, "_mindstream", False) or not fp16_is_exact(weight)):
        return False
    weight.data = weight.data.to(torch.float16)  # the same Parameter: lm_head stays tied to it
    lookup = embedding.forward

    def forward(input_ids):
        return lookup(input_ids).to(torch.float32)  # the base's fp32 values, bit for bit

    forward._mindstream = True
    embedding.forward = forward
    return True


def patch_soup_train():
    """Convert right before training: Soup has built the final model and the trainer."""
    from soup_cli.trainer.sft import SFTTrainerWrapper

    if getattr(SFTTrainerWrapper.train, "_mindstream", False):
        return
    original = SFTTrainerWrapper.train

    def train(self, *args, **kwargs):
        model = getattr(self, "model", None)
        if isinstance(model, torch.nn.Module):
            store_tied_embedding_fp16(model)
        return original(self, *args, **kwargs)

    train._mindstream = True
    SFTTrainerWrapper.train = train
