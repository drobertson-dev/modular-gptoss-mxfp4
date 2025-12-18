# GPT-OSS Overview

Here’s the nuts-and-bolts view of GPT-OSS (both **120b** and **20b**)—what’s inside a block, how MoE is wired, and what actually runs at inference.

## High-level shape

* **Family & sizes.** 120b = **36 layers**, ~**116.8B** total params with **~5.1B active** per token; 20b = **24 layers**, ~**20.9B** total with **~3.6B active**. Checkpoints are ~60.8 GiB (120b) and ~12.8 GiB (20b).
* **Transformer core.** Decoder-only, Pre-LN with **RMSNorm**; residual (model) width **d_model = 2880**.
* **Attention strategy.** Layers **alternate** between **dense** and **banded-window** attention (window **128**). **GQA**: **64 query heads** (head dim **64**) sharing **8 KV heads** (group size 8). **RoPE**, and dense attention is context-extended to **131,072** tokens via **YaRN**. Each head adds a **learned bias in the softmax denominator**, permitting “pay attention to nothing.”
* **Tokenizer & chat framing.** **o200k_harmony** BPE (201,088 tokens) plus the **Harmony** chat format (roles/channels) used in post-training for tool-use and CoT.
* **Active vs total compute.** The **MoE** dominates the parameter count; only a **subset of experts** is executed per token, which is why active params (runtime FLOPs/memory) are far below totals. ([OpenAI][1])

## The Transformer layer (what a single block does)

```text
for each layer ℓ:
  # attention sub-block
  x = x + Attnℓ( RMSNormℓ,a(x) )

  # feed-forward sub-block (Mixture-of-Experts)
  x = x + MoEℓ( RMSNormℓ,m(x) )
```

* **Attentionℓ.** Q: linear(2880→64×64), K/V: linear(2880→8×64) with **GQA**; apply **banded 128** (on sparse layers) or full attention (on dense layers), **RoPE**, softmax(**Q·Kᵀ / √d + bias**). Output projects back to 2880.
* **MoEℓ.** See below.

## Mixture-of-Experts (MoE) block

* **Experts per layer.** **120b:** **128 experts**; **20b:** **32 experts**. Each expert is a **gated SwiGLU MLP**; OpenAI notes an “**unconventional**” SwiGLU variant with **clamping + residual** inside the expert.
* **Router.** A **linear projection** maps the residual stream (2880) to **scores over all experts**. Routing is **Top-k = 4** per token. The outputs of the chosen experts are **weighted by a softmax computed over the selected 4** (not all experts), then summed and residually added.
* **Parameter accounting.** 120b’s table shows **~114.7B** params in MLP (i.e., MoE) vs **~0.96B** in attention and **~1.16B** in embed/unembed—hence >90% of weights sit in MoE, which is why quantizing MoE weights moves the dial.
* **Per-expert math details.** In NVIDIA’s modelcard, MoE **GEMMs include per-expert biases** (implemented in fused kernels). ([NVIDIA NIM APIs][2])

## Attention details that matter operationally

* **Alternating patterns.** Every other layer is band-limited attention with **bandwidth 128**; adjacent layers are full dense attention. This reduces KV-cache bandwidth while preserving periodic global mixing.
* **GQA (64Q / 8KV).** Eight KV streams each serve a group of 8 Q heads—**8× reuse** of KV per group—cutting memory traffic and cache size with negligible quality loss.
* **Long context.** **YaRN** extends **dense** layers’ max context to **131,072** tokens; the alternating scheme keeps compute tractable at that length.

## Quantization & runtime format

* **What’s quantized.** After pretraining, **MoE linear projection weights** are **post-trained to MXFP4** (microscaling FP4). This single move makes **120b fit on one 80 GB GPU** and **20b run in ~16 GB**, because MoE weights are >90% of the total.
* **Storage layout (open-sourced).** MXFP4 tensors are stored as packed **fp4 blocks** plus **per-block scales**; packing is along the **last dimension** to align with GEMM access. (Reference impls show this explicitly.) ([GitHub][3])
* **Execution.** In GPU kernels, MoE weights remain packed in HBM and are **decoded in-kernel** just before MMA (often to FP8/BF16), while attention math is standard FP8/BF16. (The repo ships **PyTorch**, **Triton** (single-GPU 120b), and **Metal** reference backends that match this architecture.) ([GitHub][3])

## Putting it together (token path)

1. **Embed** token → residual stream (2880).
2. **Layer ℓ attention:** compute Q (64×64), K/V (8×64) with GQA; apply **dense** or **banded-128** attention; project back and residual-add.
3. **Layer ℓ MoE:** router scores all experts → pick **top-4**; run **4 gated-SwiGLU experts**; softmax-weight only those 4; sum; residual-add.
4. Repeat for **36** (or **24**) layers; **unembed** to logits.

## Why this design (engineering motives)

* **Sparse MLP FLOPs** at large width—MoE gives the accuracy of a very wide FFN while keeping **active params** small (**5.1B / 3.6B**), which materially lowers memory bandwidth and latency. ([OpenAI][1])
* **Alternating attention** provides periodic global mixing at long context with **bounded KV cost** every other layer.
* **GQA (8 KV)** slashes cache/reads yet preserves Q-granularity; **learned softmax bias** allows true “null attention,” stabilizing training/inference at long context.
* **MXFP4 on MoE** hits the biggest weight bucket with a hardware-friendly decode path; attention & embeddings stay higher-precision where it matters.

## Useful implementation breadcrumbs

* **OpenAI model card** (exact layer counts, d_model, heads, GQA, window size, MoE k, quantization scope, tokenizer, YaRN).
* **OpenAI blog / models page** (active/total parameters, alternating attention summary). ([OpenAI][1])
* **Repo reference code** (Torch/Triton/Metal; MXFP4 packing along last dim; MoE kernels). ([GitHub][3])
* **NVIDIA modelcard** (per-expert bias in MoE GEMMs; long-context settings). ([NVIDIA NIM APIs][2])

[1]: https://openai.com/index/introducing-gpt-oss/?utm_source=chatgpt.com "Introducing gpt-oss"
[2]: https://build.nvidia.com/openai/gpt-oss-120b/modelcard?utm_source=chatgpt.com "gpt-oss-120b Model by OpenAI"
[3]: https://github.com/openai/gpt-oss "GitHub - openai/gpt-oss: gpt-oss-120b and gpt-oss-20b are two open-weight language models by OpenAI"
