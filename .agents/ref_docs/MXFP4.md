# MXFP4 Technical Overview

MXFP4 is a **microscaling** block format: groups of **k=32** FP4 elements share a single power-of-two scale. Each
element is **FP4 E2M1** (1 sign, 2-bit exponent, 1-bit mantissa); the shared scale is **E8M0** (8-bit exponent-only
scale). A block encodes values (v_i = X \times P_i), where (X) is the E8M0 scale and (P_i) is an FP4(E2M1) element. This
is standardized by the Open Compute Project (OCP).

Key numeric details (from the OCP spec):

* **Block size** (k=32); **element bits** (d=4); **scale bits** (w=8). Bits per value = (4 + \frac{8}{32} = 4.25) bits →
  **0.53125 bytes/value**. Versus BF16/FP16 (2 bytes), MXFP4 is **~3.76× smaller**; versus FP8 (1 byte), **~1.88×
  smaller**.
* **FP4(E2M1)**: exponent bias **1**, no Inf/NaN encodings; **max normal = ±6.0**, **min normal = ±1.0**, the only
  subnormal magnitude is **±0.5**.
* **Scale E8M0**: unsigned, exponent-only with bias **127**; represents powers of two from (2^{-127}) to (2^{127}); one
  code reserved for NaN (marks the whole block as NaN).

## How conversion/rounding works (post-training quantization)

The spec’s reference semantics for converting a float vector ({V_i}) into an MX block:

1. Choose scale (X) as the **largest power-of-two** (\le \max_i |V_i| / \text{(FP4 max normal)}) → for FP4, divide by 6
   and round the resulting scale **down to power-of-two**.
2. Quantize (P_i = \text{roundTiesToEven}(;V_i / X;)) into FP4(E2M1); clamp out-of-range to ±6; values below the **min
   subnormal** after rounding become **0**.
   Ties must use **round-to-nearest-even**, per OCP.

## Why MXFP4 is special (and different)

### 1) Block-floating **with power-of-two scaling**

Unlike INT4 or nonuniform codebooks (NF4), MXFP4’s scale is **E8M0 power-of-two**. That means decoding is cheap:
multiply by (2^e) (or equivalently add an exponent during conversion to the compute dtype). This keeps **weights packed
in HBM** and enables **fused dequant inside the GEMM kernel** with almost free scaling. The spec even factors scales out
of dot products:
[
\text{Dot}(\mathbf{A},\mathbf{B})=X^{(A)}X^{(B)}\sum_i P^{(A)}_i P^{(B)}_i
]
so kernels can handle products of tiny FP4 integers and multiply the two block-scales once.

### 2) Extremely compact but still floating-point

E2M1 gives a tiny per-element dynamic range ([0.5,6]) (plus 0), but the **shared E8M0** scale restores **huge global
range** (up to (2^{±127})). You get the **log-like spacing** of floats (better for heavy-tailed weight distributions)
with the memory footprint of 4-bit. Contrast: INT4 is linear per-group and often needs tighter group sizes or
per-channel scales to keep accuracy.

### 3) Hardware friendliness

Because (X) is a power-of-two, the scale step is coarse but **fast**: on GPUs it’s just exponent shifting when
converting to the math dtype. On **Hopper (H100)**, compute is in **FP8/BF16**, so you typically **keep FP4 packed** and
**dequant inside the kernel** before Tensor Core MMA; on **Blackwell**, NVIDIA also introduces **NVFP4** (see
below). ([NVIDIA Docs][1])

### 4) Clean edge-case rules

* **No Inf/NaN in FP4 elements**; NaN at the **block scale** wipes the block.
* **Clamping** to ±6 on overflow; **underflow to zero** below 0.5 after rounding; **roundTiesToEven** is required.
  These rules make implementations deterministic and vectorizable.

## MXFP4 vs. other 4-bit approaches

**Plain FP4 (no block scale):** poor global range—rarely viable alone for LLM weights. MXFP4 fixes this with per-32
microexponents.

**INT4 + per-group scale/zero-point:** linear quantization; good bandwidth, but less robust on outliers; usually needs
small group sizes or per-channel scales. MXFP4’s log spacing often preserves accuracy better at the same bit budget (
especially for weights). (General background; INT4 isn’t in OCP MX.)

**NF4 (QLoRA):** a **codebook** optimized to normal distributions; great for fine-tuning with LoRA and weight-only PTQ,
but it’s not block-floating and uses a non-power-of-two mapping. MXFP4 is more **kernel-friendly** and often faster to
decode; NF4 can be more accurate when the codebook assumption matches the weight stats. ([arXiv][2])

**NVFP4 (Blackwell):** NVIDIA’s evolution: **k=16** (smaller block), **FP8(E4M3) fractional scale** per block **plus** a
second-level **FP32 per-tensor** scale. That reduces quantization error versus MXFP4’s E8M0 power-of-two scale and
improves accuracy at 4-bit—while keeping low memory. On Blackwell, NVFP4 is the native 4-bit
target. ([NVIDIA Developer][3])

## Practical implementation notes

**Packing & memory math.** One 32-value block: **16 bytes** for FP4 nibbles + **1 byte** for scale = **17 bytes**. Align
blocks (e.g., to 16/32B boundaries) for coalesced loads.

**Kernel decode (Hopper-class GPUs).** Typical fused path:

1. Load 32 nibbles and the scale byte.
2. Convert nibbles (\rightarrow) FP4(E2M1) base (tiny LUT of 16 values or bitfield ops).
3. Apply (X) by exponent shifting (power-of-two) while converting into the compute dtype (FP8 or BF16), then feed into *
   *(w)GMMA**.
   You thereby keep HBM traffic at 4.25 bits/value and only expand in registers/shared memory just before MMA. (H100
   supports FP8 math; MXFP4 itself is decoded in software.) ([NVIDIA Docs][1])

**Calibration/scale selection.** Use the OCP recipe (power-of-two **floor** against (\max|V|/6)); it’s robust and
deterministic. If you see many zeros, your scale is too small (underflow to zero). If you see saturation at ±6, scale is
too large. The spec’s dot-product and conversion sections are the anchor.

**Where to quantize.** For LLMs, MXFP4 is commonly **weight-only**; keep embeddings/LM head/attention softmax in higher
precision. OpenAI’s GPT-OSS checkpoints ship **natively in MXFP4**, and HF Transformers can load them with MXFP4-aware
kernels. ([Hugging Face][4])

**Training with MXFP4.** If you need fine-tuning while preserving FP4 deployment, do **QAT** (fake quant during forward)
and export back to MXFP4—shown for GPT-OSS with NVIDIA ModelOpt + SGLang. ([LMSYS][5])

## When to pick MXFP4

* You’re **memory/bandwidth bound** (LLM inference) and can tolerate tiny per-block coarseness.
* You want **fast, simple decode** (shift-based scaling) and **fused kernels** without heavy codebooks.
* Your hardware path is FP8/BF16 compute (Hopper); keep FP4 packed and decode in-kernel. On **Blackwell**, consider *
  *NVFP4** for best 4-bit accuracy. ([NVIDIA Developer][3])

## TL;DR

MXFP4 = **FP4(E2M1) values + one E8M0 power-of-two scale per 32-element block**. It gives **~3.8×** BF16 memory
reduction with a simple, hardware-friendly decode (exponent shift), solid accuracy for **weight-only** LLM inference,
and clean, standardized semantics (round-to-even, clamp, block-NaN). NVFP4 refines it with smaller blocks and fractional
FP8 scales for higher accuracy on Blackwell; NF4 is an alternative when a learned codebook fits your statistics.

[1]: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html?utm_source=chatgpt.com "Using FP8 and FP4 with Transformer Engine"

[2]: https://arxiv.org/abs/2305.14314?utm_source=chatgpt.com "QLoRA: Efficient Finetuning of Quantized LLMs"

[3]: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference | NVIDIA Technical Blog"

[4]: https://huggingface.co/docs/transformers/main/quantization/mxfp4 "MXFP4"

[5]: https://lmsys.org/blog/2025-08-28-gpt-oss-qat/ "Fine-tune and deploy gpt-oss MXFP4: ModelOpt + SGLang | LMSYS Org"
