"""Keep the Qwen2 MLP's intermediates in fp16, the dtype its matmuls already produce.

bitsandbytes' Linear4bit casts its fp16 GEMM result back to the input dtype. The MLP input is the
fp32 output of RMSNorm, so gate and up come back as fp32 (2 x tokens x 8960 x 4 bytes per layer),
are kept in fp32 for the SwiGLU backward, and the SwiGLU output is cast to fp16 again inside
down_proj. Casting the input to fp16 once up front (exactly the cast every Linear4bit does
inside) gives gate and up in fp16 with identical values: SwiGLU still computes in fp32 from the
same numbers and rounds where down_proj would round. Half the saved memory, five fewer casts.
"""

import torch

from .swiglu import swiglu


def patch_qwen2_mlp_fp16():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2MLP
    if getattr(cls.forward, "_mindstream", False):
        return
    previous = cls.forward

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        compute_dtype = getattr(self.gate_proj, "compute_dtype", None)
        if (act != "silu" or not x.is_cuda or compute_dtype != torch.float16
                or getattr(self.up_proj, "compute_dtype", None) != torch.float16):
            return previous(self, x)
        x16 = x.to(torch.float16)
        return self.down_proj(swiglu(self.gate_proj(x16), self.up_proj(x16)))

    forward._mindstream = True
    cls.forward = forward
