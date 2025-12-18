# MXFP4 Key Takeaways

This document is a synopsys of a presentation at PyTorch Con 2025 and it contains key insights on how to properly handle quanizations like MXFP4 and what it takes to get them to operate at peak performance.  This is a very imporant document when it comes to implementation.

## Unlocking Low-Precision Speed: Fusion and Compute Dominance

Kernel fusion and utilizing sufficiently large tensor sizes are critical for realizing maximal speedup in low-precision training and inference because they mitigate memory bandwidth bottlenecks and amortize the overhead of quantization operations. Without these optimizations, the theoretical throughput gains of formats like MXFP4 or FP4 are often lost to data movement costs.

### **1. The Criticality of Kernel Fusion**

Kernel fusion, combining multiple operations into a single GPU kernel call, is essential because operations like scaling and quantization are typically memory-bound rather than compute-bound.

- **Eliminating Memory Roundtrips:** Unfused scaling operations require multiple roundtrips to and from global memory. For example, a naive implementation might load a tensor, calculate a scale, write it back, load it again to quantize, and write it back again. By fusing these steps, data remains in the GPU's faster memory hierarchies (registers or shared memory), drastically reducing bandwidth consumption.
- **Magnitude of Impact:** The performance difference is extreme. Benchmarks comparing an unfused kernel for casting BF16 to MXFP8 against a fused implementation (using tools like `torch.compile` or Triton) show that the fused kernel can be **34 times faster**.
- **Complex Transformation Pipelines:** Advanced quantization recipes often involve multiple steps, such as the Random Hadamard Transform (RHT), scaling, and rounding. To make these viable, they must be fused. For instance, efficient implementations of the "Quartet" algorithm fuse the Hadamard transform, quantization, scale calculation, and clipping mask generation into a single kernel to optimize data movement. Without fusion, the overhead of these preparatory steps would negate the speedup gained from the faster matrix multiplication.

### **2. The Necessity of Sufficiently Large Tensor Sizes**

Large tensor sizes are required to ensure that the workload is compute-bound, allowing the faster math operations of low-precision formats to dominate the total execution time.

- **Compute vs. Memory Scaling:** Matrix multiplications (GEMMs) scale cubically (compute heavy), whereas scaling and quantization kernels scale quadratically (memory heavy). When tensor shapes are small, the quadratic overhead of calculating scales and casting data dominates the execution time. As tensor shapes increase, the cubic term (the actual matrix math) dominates, allowing the speedup from low-precision tensor cores to become visible.
- **Amortizing Overhead:** The overhead of pre-processing steps (like RHT or scaling) decreases relative to the total runtime as matrix dimensions grow. For example, applying the RHT adds approximately 9.7% overhead for matrices sized for a 7B parameter model, but this drops to only 1.6% for 70B-sized matrices.
- **The Crossover Point:** There is a specific threshold where the benefit of quantizing a GEMM (e.g., getting a 2x or 4x speedup on the math) exceeds the cost of the overhead paid to cast the tensors. If the gem sizes are too small, the system may never reach this crossover point, and the low-precision implementation might actually be slower than standard BF16.

Kernel fusion prevents the system from stalling on memory bandwidth during quantization, while large tensor sizes ensure that the workload spends the majority of its time in the accelerated compute phase rather than in overhead preparation.
