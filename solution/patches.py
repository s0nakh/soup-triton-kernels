"""Patches that route Soup's training step through the fused LM-head loss.

1. `Qwen2ForCausalLM.forward`: on a labelled training forward on CUDA, the loss comes from
   `fused_head_loss` and the output carries no logits; what TRL and Soup need instead is left in
   `_STEP`. Every other call (eval, generation, CPU, a trainable head) runs the original.
2. `trl.SFTTrainer.compute_loss`: when the forward left a step, log `entropy`, `num_tokens` and
   `mean_token_accuracy` exactly by TRL's formulas from it, and hand the per-token losses to
   Soup's rewind recorder under the attribute it reads for logit-free steps. Otherwise the
   original method runs untouched.
"""

import torch

from .head_loss import HeadStep, fused_head_loss
from .unpad import make_pack, packing

REWIND_STEP_ATTR = "_soup_chunked_step"  # soup_cli.utils.stream_loss_chunk.STEP_ATTR (0.77.0)
_STEP = {"value": None}


def _take_step():
    step, _STEP["value"] = _STEP["value"], None
    return step


def _fused_forward_applies(model, labels, logits_to_keep, kwargs):
    weight = getattr(getattr(model, "lm_head", None), "weight", None)
    return (labels is not None and model.training and isinstance(weight, torch.Tensor)
            and weight.is_cuda and not weight.requires_grad and logits_to_keep == 0
            and "shift_labels" not in kwargs)


def patch_qwen2_causal_lm():
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2ForCausalLM
    if getattr(cls.forward, "_mindstream", False):
        return
    original = cls.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        if not _fused_forward_applies(self, labels, logits_to_keep, kwargs):
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                            logits_to_keep=logits_to_keep, **kwargs)
        pack = (make_pack(attention_mask) if position_ids is None and past_key_values is None
                and not use_cache else None)
        with packing(pack):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 position_ids=position_ids, past_key_values=past_key_values,
                                 inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        hidden = outputs.last_hidden_state
        if pack is not None and tuple(hidden.shape[:2]) != (1, pack.n_real):
            pack = None  # some layer did not run packed: the output is still padded
        loss, step = fused_head_loss(hidden, self.lm_head.weight, labels,
                                     kwargs.get("num_items_in_batch"), pack)
        _STEP["value"] = step
        return CausalLMOutputWithPast(loss=loss, logits=None,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)

    forward._mindstream = True
    cls.forward = forward


def _log_trl_metrics(trainer, mode, inputs, step):
    """entropy, num_tokens and mean_token_accuracy, as trl 0.29.1 SFTTrainer.compute_loss logs
    them from the logits."""
    gather = trainer.accelerator.gather_for_metrics
    attention_mask = inputs["attention_mask"]
    entropy = torch.sum(step.token_entropy * attention_mask) / attention_mask.sum()
    trainer._metrics[mode]["entropy"].append(gather(entropy).mean().item())
    if mode == "train":
        trainer._total_train_tokens += gather(attention_mask.sum()).sum().item()
    trainer._metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    total = gather(step.labelled).sum()
    accuracy = (gather(step.correct).sum() / total).item() if total > 0 else 0.0
    trainer._metrics[mode]["mean_token_accuracy"].append(accuracy)


def _step_from_logits(logits, labels):
    """The same metric inputs from full logits, by TRL's own formulas (fallback path only)."""
    from trl.trainer.utils import entropy_from_logits

    with torch.no_grad():
        shift_logits = logits[..., :-1, :]
        shift_targets = labels[..., 1:]
        mask = shift_targets != -100
        correct = ((shift_logits.argmax(dim=-1) == shift_targets) & mask).sum()
        return HeadStep(token_entropy=entropy_from_logits(logits), token_loss=None,
                        correct=correct, labelled=mask.sum())


def _fused_step_applies(trainer, model, inputs):
    args = getattr(trainer, "args", None)
    return (isinstance(model, torch.nn.Module) and model.training
            and not getattr(trainer, "aux_loss_enabled", False)
            and getattr(trainer, "model", None) is not None and trainer.model.training
            and args is not None and not getattr(args, "use_liger_kernel", False)
            and getattr(trainer, "num_virtual_tokens", 0) == 0
            and isinstance(inputs, dict) and "labels" in inputs
            and "attention_mask" in inputs and "shift_labels" not in inputs
            and hasattr(trainer, "_metrics") and hasattr(trainer, "_total_train_tokens"))


def patch_sft_trainer():
    import transformers
    from trl import SFTTrainer

    if getattr(SFTTrainer.compute_loss, "_mindstream", False):
        return
    original = SFTTrainer.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not _fused_step_applies(self, model, inputs):
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        _take_step()  # never reuse a stale step
        loss, outputs = transformers.Trainer.compute_loss(
            self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        step = _take_step()
        if step is None:  # the forward fell back to the original and returned logits
            step = _step_from_logits(outputs.logits, inputs["labels"])
        else:
            setattr(self, REWIND_STEP_ATTR, step)
        _log_trl_metrics(self, "train", inputs, step)
        return (loss, outputs) if return_outputs else loss

    compute_loss._mindstream = True
    SFTTrainer.compute_loss = compute_loss
