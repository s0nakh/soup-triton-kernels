"""Skip the padding inside a batch: every token-wise op runs on the real tokens only.

A batch is padded to its longest row, and every linear layer, norm, MLP and the LM head then
works on the pad positions too, though nothing uses their results. Here the real tokens of the
batch are packed into one (1, n_real, hidden) sequence right after the embedding and stay packed
through all 28 layers. Only attention, the one op that mixes positions, unpacks its q, k and v
back into the padded (batch, heads, seq, dim) layout and runs on the same mask the base builds,
so no row ever sees another row. LoRA's dropout also runs on the padded layout, so it draws the
same random mask as the base. Rows of different batches are never mixed (the task allows removing
padding inside a batch, not packing across batches).
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch

_ACTIVE = {"pack": None}


@dataclass(frozen=True)
class Pack:
    index: torch.Tensor  # (n_real,) positions of the real tokens in the flattened (batch * seq)
    batch: int
    seq: int
    lengths: tuple = ()  # real tokens per row when each row's tokens are contiguous, else ()

    @property
    def n_real(self):
        return self.index.numel()

    def pack(self, padded):
        """(batch, seq, ...) -> (1, n_real, ...)"""
        return padded.reshape(self.batch * self.seq, *padded.shape[2:])[self.index].unsqueeze(0)

    def unpack(self, packed, fill_pads=True):
        """(1, n_real, ...) -> (batch, seq, ...) with zeros at the pad positions (or, with
        fill_pads=False, whatever the fresh memory holds: for consumers that ignore them)."""
        rows = packed.squeeze(0)
        size = (self.batch * self.seq, *rows.shape[1:])
        out = rows.new_zeros(size) if fill_pads else rows.new_empty(size)
        out[self.index] = rows
        return out.view(self.batch, self.seq, *rows.shape[1:])


def make_pack(attention_mask):
    """A Pack when the batch has padding to drop, else None."""
    if attention_mask is None or attention_mask.dim() != 2 or attention_mask.shape[0] < 2:
        return None
    keep = attention_mask.reshape(-1) != 0
    index = keep.nonzero().squeeze(1)
    if index.numel() == keep.numel() or index.numel() == 0:
        return None
    batch, seq = attention_mask.shape
    return Pack(index=index, batch=batch, seq=seq, lengths=_contiguous_lengths(attention_mask))


def _contiguous_lengths(attention_mask):
    """Real tokens per row if every row's real tokens form one contiguous block (right or left
    padding), else (). One host sync per forward."""
    mask = attention_mask != 0
    lengths = mask.sum(dim=1)
    first = mask.int().argmax(dim=1)
    positions = torch.arange(mask.shape[1], device=mask.device)
    block = (positions >= first[:, None]) & (positions < (first + lengths)[:, None])
    if not bool(torch.equal(block, mask)) or bool((lengths == 0).any()):
        return ()
    return tuple(int(n) for n in lengths.tolist())


def active_pack():
    return _ACTIVE["pack"]


@contextmanager
def packing(pack):
    _ACTIVE["pack"] = pack
    try:
        yield
    finally:
        _ACTIVE["pack"] = None


def _is_padded(pack, tensor):
    return tensor.dim() >= 3 and tuple(tensor.shape[:2]) == (pack.batch, pack.seq)


def _is_packed(pack, tensor):
    return tensor.dim() == 3 and tensor.shape[0] == 1 and tensor.shape[1] == pack.n_real


def _packed_rotary(pack, position_embeddings):
    cos, sin = position_embeddings
    if not _is_padded(pack, cos.expand(pack.batch, *cos.shape[1:])):
        return None
    return tuple(pack.pack(t.expand(pack.batch, *t.shape[1:])) for t in (cos, sin))


def _lora_adapter(proj):
    """"" for a bare projection, the adapter name for a plain single-adapter peft LoRA layer, or
    None for anything else. Decided before any dropout runs, so a fallback never draws twice."""
    if getattr(proj, "base_layer", None) is None:
        return ""
    adapters = list(getattr(proj, "active_adapters", []) or [])
    if (len(adapters) != 1 or getattr(proj, "disable_adapters", True)
            or getattr(proj, "merged", True) or adapters[0] not in proj.lora_A
            or adapters[0] in getattr(proj, "lora_variant", {})):
        return None
    return adapters[0]


def _lora_parts(proj, adapter, x16, x32):
    """(base output, LoRA delta or None) of a q/k/v projection, as peft computes them under
    autocast: the base layer on the fp16 input, LoRA on the fp32 input through its dropout."""
    if not adapter:
        return proj(x16), None
    lora = proj.lora_B[adapter](proj.lora_A[adapter](proj.lora_dropout[adapter](x32)))
    return proj.base_layer(x16), lora * proj.scaling[adapter]


def _fused_qkv(module, hidden_states, rotary):
    """q, k, v as fp16 (tokens, heads * dim) from our fused LoRA + rotary + cast kernel, or None
    to use the original ops."""
    projections = (module.q_proj, module.k_proj, module.v_proj)  # peft's order: same dropout RNG
    adapters = [_lora_adapter(p) for p in projections]
    if (not torch.is_autocast_enabled("cuda") or hidden_states.dtype != torch.float32
            or any(a is None for a in adapters)):
        return None
    from .rope import qkv_prologue

    x16 = hidden_states.to(torch.float16)  # the cast every 4-bit layer makes inside, done once
    tokens, dim = hidden_states.shape[1], module.head_dim
    cos, sin = (t.reshape(tokens, -1) for t in rotary)
    flat = []
    for proj, adapter in zip(projections, adapters):
        base, lora = _lora_parts(proj, adapter, x16, hidden_states)
        flat.append((base.reshape(tokens, -1), None if lora is None else lora.reshape(tokens, -1)))
    (qb, ql), (kb, kl), (vb, vl) = flat
    return (qkv_prologue(qb, ql, cos, sin, dim), qkv_prologue(kb, kl, cos, sin, dim),
            qkv_prologue(vb, vl, None, None, dim))


def _attention_per_row(module, pack, query, key, value):
    """Causal attention row by row on the packed tokens: each row's real tokens are contiguous,
    so a plain causal mask on its own block is exactly the base's causal + padding mask for the
    real positions, with no work on padding and no copies into the padded layout."""
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv

    groups, lengths = module.num_key_value_groups, list(pack.lengths)
    # split (not slicing): its backward is one concatenation, not a zero-filled copy per row
    outputs = [
        torch.nn.functional.scaled_dot_product_attention(
            q, repeat_kv(k, groups), repeat_kv(v, groups), is_causal=True, scale=module.scaling)
        for q, k, v in zip(query.split(lengths, dim=2), key.split(lengths, dim=2),
                           value.split(lengths, dim=2))
    ]
    return torch.cat(outputs, dim=2)


def patch_qwen2_unpadded():
    """Patch Qwen2's decoder layer (pack at the first layer), attention (unpack around the
    attention kernel) and LoRA's dropout (draw on the padded layout). All three run their
    original code unless a pack is active for the current forward."""
    from transformers.models.qwen2 import modeling_qwen2

    layer_cls, attn_cls = modeling_qwen2.Qwen2DecoderLayer, modeling_qwen2.Qwen2Attention
    if getattr(attn_cls.forward, "_mindstream", False):
        return
    layer_forward, attn_forward = layer_cls.forward, attn_cls.forward
    dropout_forward = torch.nn.Dropout.forward

    def decoder_layer(self, hidden_states, *args, **kwargs):
        pack = active_pack()
        if (pack is not None and _is_padded(pack, hidden_states)
                and kwargs.get("past_key_values") is None and not kwargs.get("use_cache")):
            hidden_states = pack.pack(hidden_states)  # the first layer packs; the rest stay packed
        return layer_forward(self, hidden_states, *args, **kwargs)

    def attention(self, hidden_states, position_embeddings, attention_mask,
                  past_key_values=None, **kwargs):
        pack = active_pack()
        rotary = None if pack is None else _packed_rotary(pack, position_embeddings)
        if (pack is None or rotary is None or past_key_values is not None
                or not _is_packed(pack, hidden_states)):
            return attn_forward(self, hidden_states, position_embeddings, attention_mask,
                                past_key_values=past_key_values, **kwargs)
        shape = (1, pack.n_real, -1, self.head_dim)
        fused = _fused_qkv(self, hidden_states, rotary)
        if fused is not None:
            query, key, value = (t.view(shape).transpose(1, 2) for t in fused)
        else:
            query = self.q_proj(hidden_states).view(shape).transpose(1, 2)
            key = self.k_proj(hidden_states).view(shape).transpose(1, 2)
            value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
            query, key = modeling_qwen2.apply_rotary_pos_emb(query, key, *rotary)

        if pack.lengths and self.sliding_window is None and getattr(self, "is_causal", True):
            output = _attention_per_row(self, pack, query, key, value)
            return self.o_proj(output.transpose(1, 2).reshape(1, pack.n_real, -1)), None

        def padded(states):  # (1, heads, n_real, dim) -> (batch, heads, seq, dim)
            return pack.unpack(states.transpose(1, 2)).transpose(1, 2)

        interface = modeling_qwen2.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, modeling_qwen2.eager_attention_forward)
        output, weights = interface(
            self, padded(query), padded(key), padded(value), attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
        output = pack.pack(output.reshape(pack.batch, pack.seq, -1).contiguous())
        return self.o_proj(output), weights

    def dropout(self, input):
        pack = active_pack()
        if pack is None or not self.training or self.p == 0 or not _is_packed(pack, input):
            return dropout_forward(self, input)
        # the mask does not depend on the values, and pad positions are dropped again right after
        return pack.pack(dropout_forward(self, pack.unpack(input, fill_pads=False)))

    for fn in (decoder_layer, attention, dropout):
        fn._mindstream = True
    layer_cls.forward, attn_cls.forward = decoder_layer, attention
    torch.nn.Dropout.forward = dropout
