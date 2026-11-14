# Submission: MindStream AI

## Links
- Repository, final commit: https://github.com/BAITC-Hacks/hack-65172d00-mindstream-ai/commit/b1c97d9ca02276b955732f38699bc868ccf8d3d1
  (the code; later commits only add links to this file and the README). That team repository was private to
  the hackathon; this public repository is its copy with the full history, so commit hashes differ.
  The same commit here: https://github.com/s0nakh/soup-triton-kernels/commit/eacdde26cb961f462119f846a4097bc7046bf382
  (`eacdde2`; the code in `solution/` is unchanged since).
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  the README): https://www.kaggle.com/code/sophieda/mindstream-soup-kernels?scriptVersionId=363824217
- After the hackathon (10.11.2026), version 5 of the same notebook ran from this public repository
  (commit `ca77dcc`, same `solution/` code): https://www.kaggle.com/code/sophieda/mindstream-soup-kernels?scriptVersionId=368085963
  — 93.1 s → 58.8 s (1.58 x), 14,345 → 3,329 MiB (4.31 x), eval loss +0.002 %, ALL CHECKS PASSED.
- Demo video (up to 3 minutes): https://youtu.be/fNUtKUW9vSs (voiceover: ElevenLabs free plan, voice Victoriya Voloshina — Curious Educator)
- Presentation: https://drive.google.com/file/d/1kJC7frksNmOO7wqf6DHUjxd_cLn_TOz8/view?usp=sharing (slides generated in Google NotebookLM (Gemini) from our text, numbers checked by hand)

## Outcome
Average jury score **61.88 / 100**; Decentrathon 6.0 [participant certificate](./assets/decentrathon6-certificate.pdf) (AI Engineering by Soup track, issued 14 Nov 2026).

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Sonia Akhsen | captain, solo participant · Aktau | [s0nakh](https://github.com/s0nakh) | [sophieda](https://www.kaggle.com/sophieda) | [GitHub profile](https://github.com/s0nakh) |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `nf4_dequant_kernel` | `solution/nf4.py` | bitsandbytes' two dequantization kernels (nested absmax + NF4 codes) in every forward and backward of the 196 frozen 4-bit linears | 7.4–7.5 % |
| `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | Qwen2RMSNorm's 6 forward / ~12 backward PyTorch ops | 0.7 % + 0.8–0.9 % |
| `qkv_prologue_fwd_kernel`, `qkv_prologue_bwd_kernel` | `solution/rope.py` | the cast + clone + LoRA add + rotary embedding + cast chain around q, k, v | 0.7 % + 0.4 % |
| `head_ce_stats_grad_kernel` | `solution/head_loss.py` | the full (batch, seq, 151,936) logits: fp32 loss copy, its gradient, TRL's accuracy copy, Soup's rewind recompute | 0.5 % |
| **Total (ours)** | | | **10.7–10.8 %** (check 4: 10.5 % lowest run) |

The organizers' example SwiGLU kernels keep running and are not counted.

Ideas taken from other kernels, with links (we wrote the code ourselves):
- Cut Cross-Entropy — E. Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models*, 2024: https://arxiv.org/abs/2411.09009
- Liger-Kernel fused linear cross-entropy / fused RMSNorm: https://github.com/linkedin/Liger-Kernel
- Online softmax — M. Milakov, N. Gimelshein, 2018: https://arxiv.org/abs/1805.02867
- NF4 and double quantization — T. Dettmers et al., *QLoRA*, 2023: https://arxiv.org/abs/2305.14314
- Rotary embedding — J. Su et al., *RoFormer*, 2021: https://arxiv.org/abs/2104.09864
- Fusing LoRA with neighbouring ops and recomputing activations in own code — Unsloth (https://unsloth.ai/blog/long-context), Axolotl (https://docs.axolotl.ai/docs/lora_optims.html)

## Results on Kaggle T4

The saved public notebook above (organizers' template, `COMMIT = "b1c97d9"`, 1 run per side):

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 95.7 | 64.1 | speedup ≈ 1.49 x |
| Peak GPU memory, MiB | 14,345 | 3,329 | base/ours ≈ 4.31 x |
| Eval loss (public rows) | 2.04969 | 2.04965 | −0.002 % vs base |
| Own kernels, share of the training time | | 10.1 % | threshold 5 % |

ALL CHECKS PASSED (10/10).

Memory-limit check ("fits a 4 GB GPU"): on the same T4, `soup train` with our solution and a hard PyTorch
memory cap (`torch.cuda.set_per_process_memory_fraction`, no change to the solution) of 3,500 MiB and of 3,072 MiB
finished without running out of memory, eval loss 2.04966 / 2.04964, whole process 3,332 / 3,218 MiB by nvidia-smi.
This emulates a small GPU on the T4; we did not run on a physical 4 GB card.

Our development run with `harness/run_task.py --repeats 2` (2 runs per side, means):

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 83.5 | 55.5 | speedup ≈ 1.50 x |
| Peak GPU memory, MiB | 14,345 | 3,329 | base/ours ≈ 4.31 x |
| Eval loss (public rows) | 2.04966 | 2.04962 | −0.002 % vs base |

- [x] All 10 checks of `harness/check_rules.py` passed
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why (logits of the 151,936-token vocabulary, NF4 dequantization,
element-wise chains around norms and attention, padding), how the kernels work, results against
the base (speed, memory, quality with the accuracy report), limits and next steps.

## Limitations
- About 19 s of the ~55 s are not ours to cut: library imports (~13 s), loading the 4-bit model
  (~3 s), data preparation and saving. So the whole-command speedup (≈1.5×) is lower than the
  training-loop speedup (≈1.73×).
- The first step is ~2.8 s vs ~1.2 s in the base (first loading of CUDA/cuBLAS kernels for new
  shapes); we did not remove it.
- The matmuls stay on cuBLAS: we did not write a Triton NF4 GEMM for Turing.
- The memory floor (~1.6 GB) is the 4-bit weights, the fp16 embedding and the CUDA context.
- Built for this task's setup (Qwen2, bitsandbytes NF4 with double quantization, fp16 on a T4);
  anything else falls back to the original code.
- Next: a Triton NF4 GEMM with LoRA fused in, offloading checkpointed activations to RAM for
  GPUs below 3 GB.
