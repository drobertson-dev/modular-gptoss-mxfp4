# MXFP4 GPT-OSS Custom Architecture & Kernel Development

Fork of the Modular Max repo for building a native MXFP4 architecture for GPT-OSS LLM (20B and 120B models).

## Background

GPT-OSS was trained in MXFP4, making it the model's native quantization format. Current Modular integrations use BF16 as a stopgap. This project creates a proper native architecture that handles MXFP4 natively.

**Critical:** MXFP4 kernels require dequantization in shared memory before passing data to tensor cores. Standard kernels will run *slower* with MXFP4 if data transfer isn't minimized. See the Triton reference implementation for the correct approach.

## Environment

- **Hardware:** Hopper architecture with H100 GPU (SM90)
- **Purpose:** Testing and debugging MXFP4 kernels

## ⚠️ Important Constraint

**DO NOT modify existing Modular kernel code.** Import existing kernels if needed. All new code goes in the `max/examples/` directory.

---

## Directory Structure

### Primary Development Directories

| Path | Description |
|------|-------------|
| `max/examples/custom_ops/kernels/` | Mojo kernel implementations |
| `max/examples/custom-models/gpt_oss_mxfp4/` | GPT-OSS model architecture and Python bindings |
| `max/examples/custom-models/triton_example/` | Reference Triton implementation (kernel pattern to follow) |

### Planning & Documentation (`.agents/`)

| Path | Description |
|------|-------------|
| `.agents/OVERVIEW.md` | High-level design and SM90 kernel architecture |
| `.agents/PLANS.md` | Master implementation plan |
| `.agents/exec_plans/` | Execution plans |
| `.agents/ref_docs/` | Technical documentation |
| `.agents/skills/` | Step-by-step guides |

### Reference Documentation (`.agents/ref_docs/`)

| File | Description |
|------|-------------|
| `MXFP4_KEY_TAKEAWAYS.md` | **Critical performance requirements** — must follow exactly |
| `MXFP4.md` | MXFP4 format specification |
| `GPT_OSS_OVERVIEW.md` | GPT-OSS model architecture details |

### Skills & Guides

| Path | Description |
|------|-------------|
| `.agents/skills/debugging-mojo/` | GPU debugging instructions for Mojo programs |
| `.agents/skills/testing-mojo/` | How to test mojo code correctly (Its not with python) |

---

## Workflow: Creating an Execution Plan

When writing an execution plan, follow this sequence:

1. **Read the Overview** — `.agents/OVERVIEW.md`
   - Understand the SM90 kernel design, tile layouts, and warp mapping
   - Note the MXFP4→FP16 dequant strategy (decode in registers, not shared memory)

2. **Study MXFP4 Key Takeaways** — `.agents/ref_docs/MXFP4_KEY_TAKEAWAYS.md`
   - Contains critical performance constraints that must be followed exactly
   - Covers scale factor handling, vector sizes, and decode patterns

3. **Review the Triton Reference** — `max/examples/custom-models/triton_example/`
   - Working SM90 implementation showing the correct kernel pattern
   - Pay attention to: tile sizes, K-loop structure, fragment decode, epilogue fusion

4. **Write the Execution Plan** — output to `.agents/exec_plans/`
   - Break down into concrete implementation steps
   - Reference specific files and functions from the Triton example
   - Include test/validation checkpoints

---

## Quick Reference

1. **Start here:** Read `.agents/OVERVIEW.md` for kernel architecture
2. **Critical constraints:** Read `.agents/ref_docs/MXFP4_KEY_TAKEAWAYS.md`
3. **Kernel pattern:** Study `max/examples/custom-models/triton_example/` for working SM90 code
4. **Write kernels:** Add Mojo kernels in `max/examples/custom_ops/kernels/`
5. **Model integration:** Update architecture in `max/examples/custom-models/gpt_oss_mxfp4/`
6. **Debug:** Follow instructions in `.agents/skills/debugging-mojo/`

## ExecPlans

When writing complex features or significant refactors, use an ExecPlan (as described in `.agents/PLANS.md`) from design to implementation.
