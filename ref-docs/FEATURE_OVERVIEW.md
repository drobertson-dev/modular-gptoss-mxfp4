# Feature Overview

**Porting The Kernel**
Porting the GPT-OSS-120b MXFP4 implementation `ref-docs/gpt-oss` to Modular Max to run on a single H100 is not only possible but is the ideal use case for Mojo’s `MAX` platform.

**Reasoning:**

1. **Memory Constraints:** The 120b model in BF16 requires ~240GB of VRAM. An H100 has 80GB. The current Modular implementation (BF16) physically cannot run on one H100. In MXFP4 (approx 4.25 bits/param), the weights drop to ~60GB, fitting comfortably on a single H100 with room for KV cache.
2. **Compute Capability:** Mojo provides direct access to low-level GPU intrinsics (via MLIR/NVVM). You can write the exact equivalent of the OpenAI Triton kernel—loading packed `int4`/`int8` data, dequantizing in registers, and feeding Tensor Cores—without the overhead of Python or the rigidity of pre-compiled C++ CUDA libraries.
3. **Modular Architecture:** Max allows for `CustomOp` insertion. You do not need to rewrite the entire pipeline; you only need to replace the specific MoE matrix multiplication node in the graph with a custom Mojo kernel.

---

### The Plan

To achieve `max serve openai/gpt-oss-120b` with on-the-fly dequantization, we must move from "pure graph composition" (Python) to "custom kernel injection" (Mojo).

### Phase 1: Weight Adapter & Data Loading (Python)

The current `weight_adapters.py` likely converts the safetensors to standard float types. We must stop this.

1. **Modify Weight Loading:** Update `weight_adapters.py` to preserve the MXFP4 structure. Instead of returning one `BF16` tensor, return two `uint8` tensors per weight:
   - `blocks`: The packed 4-bit weights (stored as `uint8`).
   - `scales`: The 8-bit scales (stored as `uint8`).
2. **Graph Inputs:** In `moe.py`, the inputs for the experts must change from `TensorValue` (BF16) to `TensorValue` (uint8) to represent the raw bytes in memory.

### Phase 2: The Custom Mojo Kernel (Mojo)

This is the heavy lifting. We need a Mojo `CustomOp` that replicates the logic of `triton/moe.py`.

1. **Kernel Signature:** Accepts `activations` (BF16), `packed_weights` (uint8), `scales` (uint8), and `indices` (int32).
2. **Tiling Strategy:** Use Hopper-specific features (Tensor Memory Accelerator - TMA) to load compressed blocks efficiently.
3. **Register Dequantization:**
   - Load 32-element blocks into registers.
   - Bit-shift/mask to separate nibbles.
   - Apply the E8M0 scale (this is a fast bitwise exponent addition on floating point representations).
   - Convert to BF16 (or FP8 if targeting H100 specifically for higher throughput).
4. **Gemm:** Feed the dequantized registers into `mma.sync` (Matrix Multiply Accumulate) intrinsics.

### Phase 3: Integration (Python/Graph)

1. **Inject Custom Op:** In `layers/moe.py`, replace `max.nn.kernels.grouped_matmul_ragged` with your new `moe_mxfp4` custom op.
2. **Config Update:** Update `GptOssConfig` to accept a `quantization_format` flag to toggle between the BF16 path and the MXFP4 path.

---

### Detailed Walkthrough

### 1. Modify Weight Adapters (Python)

Do not dequantize to BF16. Keep raw bytes.

```python
# max/pipelines/architectures/gpt_oss/weight_adapters.py

def convert_safetensor_state_dict(state_dict: dict[str, Weights], **kwargs) -> dict[str, WeightData]:
    new_state_dict = {}

    for name, weight in state_dict.items():
        # If this is an MoE weight (identified by name pattern), keep it packed
        if "mlp1_weight" in name or "mlp2_weight" in name:
            # In the safetensors, these are split into .blocks and .scales

            # We pass them through as uint8 tensors
            new_state_dict[name] = weight.data() # Assuming raw bytes
        else:
            # Standard conversion for non-MoE layers (Attention, Norms)
            # ... existing logic ...
            pass

    return new_state_dict

```

### 2. The Mojo Kernel Skeleton

Create a file `moe_mxfp4.mojo`. This uses the Max Graph Custom Op API.

```mojo
import max.graph.custom_op as ops
from max.tensor import Tensor
from max.dtype import DType

# This struct defines the interface between Python and the GPU Kernel
@value
struct MXFP4MoEOp(ops.CustomOp):
    fn name(self) -> String:
        return "mxfp4_moe_matmul"

    # Define output shape logic
    fn shape(self, inputs: ops.TensorShapeArray) -> ops.TensorShapeArray:
        # Calculate output shape based on input activation shape and expert count
        var activation_shape = inputs[0]
        # ... logic to determine output shape ...
        return ops.TensorShapeArray(activation_shape)

    # The CPU-side dispatcher that launches the GPU kernel
    fn execute(self, inputs: ops.TensorArray, outputs: ops.TensorArray):
        var activations = inputs[0]
        var packed_weights = inputs[1]
        var scales = inputs[2]
        var indices = inputs[3]

        # Launch parameters
        var grid = ...
        var block = ...

        # Call the GPU kernel
        mxfp4_kernel[grid, block](
            activations.data,
            packed_weights.data,
            scales.data,
            indices.data,
            outputs[0].data
        )

# The actual GPU Kernel logic
@always_inline("nodebug")
fn mxfp4_kernel(
    act_ptr: UnsafePointer[bfloat16],
    weights_ptr: UnsafePointer[uint8],
    scales_ptr: UnsafePointer[uint8],
    ...
):
    # 1. Compute Thread ID and offsets

    # 2. Load Packed Data (Global Memory -> Shared Memory/Registers)
    #    Load 16 bytes (32 packed FP4 weights) + 1 byte scale

    # 3. Dequantize (The "Secret Sauce")
    #    Iterate over the packed byte
    #    Extract nibbles: (byte & 0x0F), (byte >> 4)
    #    Look up FP4 value in constant table or compute via bit manipulation
    #    Multiply by Scale: result = fp4_val * (2.0 ** (scale - 127))

    # 4. Compute GEMM
    #    Use SIMD instructions or Tensor Core intrinsics to multiply
    #    dequantized_weights * activations

    pass

```

### 3. Integrate into Python Layer (`moe.py`)

Modify `GptOssMoE` to use the custom op.

```python
# max/pipelines/architectures/gpt_oss/layers/moe.py

from max.graph import ops, TensorValue
# You will compile the mojo file and import the registered op
# This assumes the custom op is registered with the session

class GptOssMoE(MoE, Shardable):
    def _init_experts(self) -> None:
        if self.config.quantization == "mxfp4":
            # Initialize weights as uint8 tensors to hold packed data
            self._experts_gate_up_proj_blocks = Weight(
                "experts.gate_up_proj.blocks", dtype=DType.uint8, ...
            )
            self._experts_gate_up_proj_scales = Weight(
                "experts.gate_up_proj.scales", dtype=DType.uint8, ...
            )
            # ... repeat for down proj ...
        else:
            # ... existing BF16 initialization ...

    def __call__(self, x: TensorValue) -> TensorValue:
        # ... routing logic (keeps existing router) ...

        if self.config.quantization == "mxfp4":
            # Instead of grouped_matmul_ragged, call the custom op
            gate_up_output = ops.custom_op(
                "mxfp4_moe_matmul",
                inputs=[
                    permutated_states,
                    self._experts_gate_up_proj_blocks,
                    self._experts_gate_up_proj_scales,
                    expert_ids
                ],
                out_types=[x.dtype]
            )
        else:
            # ... existing BF16 matmul ...

        # ... swiglu and rest of logic ...

```

### Critical Implementation Details for H100

1. **Layout:** The Triton reference uses `HopperMXValueLayout`. You must ensure your Mojo kernel expects the data in the exact layout provided by the OpenAI checkpoints (packed along the last dimension), or transpose it once during loading in Python.
2. **Block Size:** MXFP4 uses a block size of 32. Your kernel threads should process data in multiples of 32 to align with the scaling factor.
3. **Performance:** To match vLLM/Triton performance, you cannot process one element at a time. You must load `int4` vectors, dequantize into `bfloat16x2` or `float32x4` SIMD vectors, and immediately issue FMA instructions.

### Conclusion

This plan aligns with Modular's ethos: keep high-level graph definitions in Python for flexibility, but drop into Mojo for the hardware-critical "hot path" (the MoE GEMM). This allows you to run the 120b model on a single H100 by keeping weights compressed in memory and only expanding them in the GPU registers during compute.
