---
title: "Bank Conflicts"
description: "This puzzle contains two complementary sections that build your expertise progressively:"
---

# Bank Conflicts

This puzzle contains two complementary sections that build your expertise progressively:

## Why this puzzle matters

**Completing the performance trilogy:**You've learned GPU profiling tools in Puzzle 30 and understood occupancy optimization in Puzzle 31. Now you're ready for the final piece of the performance optimization puzzle: **shared memory efficiency**.

**The hidden performance trap:**You can write GPU kernels with perfect occupancy, optimal global memory coalescing, and identical mathematical operations - yet still experience dramatic performance differences due to **how threads access shared memory**. Bank conflicts represent one of the most subtle but impactful performance pitfalls in GPU programming.

**The learning journey:**

- **Puzzle 30**taught you to **measure and diagnose**performance with NSight profiling
- **Puzzle 31**taught you to **predict and control**resource usage through occupancy analysis
- **Puzzle 32**teaches you to **optimize shared memory access patterns**for maximum efficiency

**Why this matters beyond GPU programming:**The principles of memory banking, conflict detection, and systematic access pattern optimization apply across many parallel computing systems - from CPU cache hierarchies to distributed memory architectures.

> **Note: This puzzle is specific to NVIDIA GPUs**
>
> Bank conflict analysis uses NVIDIA's 32-bank shared memory architecture and NSight Compute profiling tools. While the optimization principles apply broadly, the specific techniques and measurements are NVIDIA CUDA-focused.

## Overview

**Shared memory bank conflicts**occur when multiple threads in a warp simultaneously access different addresses within the same memory bank, forcing the hardware to serialize these accesses. This can transform what should be a single-cycle memory operation into multiple cycles of serialized access.

**What you'll discover:**

- How GPU shared memory banking works at the hardware level
- Why identical kernels can have vastly different shared memory efficiency
- How to predict and measure bank conflicts before they impact performance
- Professional optimization strategies for designing conflict-free algorithms

**The detective methodology:**This puzzle follows the same evidence-based approach as previous performance puzzles - you'll use profiling tools to uncover hidden inefficiencies, then apply systematic optimization principles to eliminate them.

## Key concepts

**Shared memory architecture fundamentals:**

- **32-bank design**: NVIDIA GPUs organize shared memory into 32 independent banks
- **Conflict types**: No conflict (optimal), N-way conflicts (serialized), broadcast (optimized)
- **Access pattern mathematics**: Bank assignment formulas and conflict prediction
- **Performance impact**: From optimal 1-cycle access to worst-case 32-cycle serialization

**Professional optimization skills:**

- **Pattern analysis**: Mathematical prediction of banking behavior
- **Profiling methodology**: NSight Compute metrics for conflict measurement
- **Design principles**: Conflict-free algorithm patterns and prevention strategies
- **Performance validation**: Evidence-based optimization using systematic measurement

## Puzzle structure

This puzzle contains two complementary sections that build your expertise progressively:

### **[ Understanding Shared Memory Banks](#understanding-shared-memory-banks)**

Learn the theoretical foundations of GPU shared memory banking through clear explanations and practical examples.

**You'll learn:**

- How NVIDIA's 32-bank architecture enables parallel access
- The mathematics of bank assignment and conflict prediction
- Types of conflicts and their performance implications
- Connection to previous concepts (warp execution, occupancy, profiling)

**Key insight:**Understanding the hardware enables you to predict performance before writing code.

### **[Conflict-Free Patterns](#conflict-free-patterns)**

Apply your banking knowledge to solve a performance mystery using professional profiling techniques.

**The detective challenge:**Two kernels compute identical results but have dramatically different shared memory access efficiency. Use NSight Compute to uncover why one kernel experiences systematic bank conflicts while the other achieves optimal performance.

**Skills developed:**Pattern analysis, conflict measurement, systematic optimization, and evidence-based performance improvement.

## Getting started

**Learning path:**

1. **[Understanding Shared Memory Banks](#understanding-shared-memory-banks)**- Build theoretical foundation
2. **[Conflict-Free Patterns](#conflict-free-patterns)**- Apply detective skills to real optimization

**Prerequisites:**

- GPU profiling experience from Puzzle 30
- Resource optimization understanding from Puzzle 31
- Shared memory programming experience from Puzzle 8 and Puzzle 16

**Hardware requirements:**

- NVIDIA GPU with CUDA toolkit
- NSight Compute profiling tools
- The dependencies such as profiling are managed by `pixi`
- [Compatible GPU architecture](https://docs.modular.com/max/packages/#gpu-compatibility)

## The optimization impact

**When bank conflicts matter most:**

- **Matrix multiplication**with shared memory tiling
- **Stencil computations**using shared memory caching
- **Parallel reductions**with stride-based memory patterns

**Professional development value:**

- **Systematic optimization**: Evidence-based performance improvement methodology
- **Hardware awareness**: Understanding how software maps to hardware constraints
- **Pattern recognition**: Identifying problematic access patterns in algorithm design

**Learning outcome:**Complete your GPU performance optimization toolkit with the ability to design, measure, and optimize shared memory access patterns - the final piece for professional-level GPU programming expertise.

This puzzle demonstrates that **optimal GPU performance requires understanding hardware at multiple levels**- from global memory coalescing through occupancy management to shared memory banking efficiency.

##  Understanding Shared Memory Banks

### Building on what you've learned

You've come a long way in your GPU optimization journey. In Puzzle 8, you discovered how shared memory provides fast, block-local storage that dramatically outperforms global memory. Puzzle 16 showed you how matrix multiplication kernels use shared memory to cache data tiles, reducing expensive global memory accesses.

But there's a hidden performance trap lurking in shared memory that can serialize your parallel operations: **bank conflicts**.

**The performance mystery:**You can write two kernels that access shared memory in seemingly identical ways - both use the same amount of data, both have perfect occupancy, both avoid race conditions. Yet one runs 32 slower than the other. The culprit? How threads access shared memory banks.

### What are shared memory banks?

Think of shared memory as a collection of 32 independent memory units called **banks**, each capable of serving one memory request per clock cycle. This banking system exists for a fundamental reason: **hardware parallelism**.

When a warp of 32 threads needs to access shared memory simultaneously, the GPU can serve all 32 requests in parallel, **if each thread accesses a different bank**. When multiple threads try to access the same bank, the hardware must **serialize**these accesses, turning what should be a 1-cycle operation into multiple cycles.

#### Bank address mapping

Each 4-byte word in shared memory belongs to a specific bank according to this formula:

```
bank_id = (byte_address / 4) % 32
```

Here's how the first 128 bytes of shared memory map to banks:

| Address Range | Bank ID | Example `float32` Elements |
|---------------|---------|---------------------------|
| 0-3 bytes     | Bank 0  | `shared[0]` |
| 4-7 bytes     | Bank 1  | `shared[1]` |
| 8-11 bytes    | Bank 2  | `shared[2]` |
| ...           | ...     | ... |
| 124-127 bytes | Bank 31 | `shared[31]` |
| 128-131 bytes | Bank 0  | `shared[32]` |
| 132-135 bytes | Bank 1  | `shared[33]` |

**Key insight:**The banking pattern repeats every 32 elements for `float32` arrays, which perfectly matches the 32-thread warp size. This is not a coincidence - it's designed for optimal parallel access.

### Types of bank conflicts

#### No conflict: the ideal case

When each thread in a warp accesses a different bank, all 32 accesses complete in 1 cycle:

```mojo
# Perfect case: each thread accesses a different bank
shared[thread_idx.x]  # Thread 0->Bank 0, Thread 1->Bank 1, ..., Thread 31->Bank 31
```

**Result:**32 parallel accesses, 1 cycle total

#### N-way bank conflicts

When N threads access different addresses in the same bank, the hardware serializes these accesses:

```mojo
# 2-way conflict: stride-2 access pattern
shared[thread_idx.x * 2]  # Thread 0,16->Bank 0; Thread 1,17->Bank 1; etc.
```

**Result:**2 accesses per bank, 2 cycles total (50% efficiency)

```mojo
# Worst case: all threads access different addresses in Bank 0
shared[thread_idx.x * 32]  # All threads->Bank 0
```

**Result:**32 serialized accesses, 32 cycles total (3% efficiency)

#### The broadcast exception

There's one important exception to the conflict rule: **broadcast access**. When all threads read the **same address**, the hardware optimizes this into a single memory access:

```mojo
# Broadcast: all threads read the same value
constant = shared[0]  # All threads read shared[0]
```

**Result:**1 access broadcasts to 32 threads, 1 cycle total

This optimization exists because broadcasting is a common pattern (loading constants, reduction operations), and the hardware can duplicate a single value to all threads without additional memory bandwidth.

### Why bank conflicts matter

#### Performance impact

Bank conflicts directly multiply your shared memory access time:

| Conflict Type | Access Time | Efficiency | Performance Impact |
|---------------|-------------|------------|-------------------|
| No conflict | 1 cycle | 100% | Baseline |
| 2-way conflict | 2 cycles | 50% | 2 slower |
| 4-way conflict | 4 cycles | 25% | 4 slower |
| 32-way conflict | 32 cycles | 3% | **32 slower**|

#### Real-world context

From Puzzle 30, you learned that memory access patterns can create dramatic performance differences. Bank conflicts are another example of this principle operating at the shared memory level.

Just as global memory coalescing affects DRAM bandwidth utilization, bank conflicts affect shared memory throughput. The difference is scale: global memory latency is hundreds of cycles, while shared memory conflicts add only a few cycles per access. However, in compute-intensive kernels that heavily use shared memory, these "few cycles" accumulate quickly.

#### Connection to warp execution

Remember from Puzzle 24 that warps execute in SIMT (Single Instruction, Multiple Thread) fashion. When a warp encounters a bank conflict, **all 32 threads must wait**for the serialized memory accesses to complete. This waiting time affects the entire warp's progress, not just the conflicting threads.

This connects to the occupancy concepts from Puzzle 31: bank conflicts can prevent warps from hiding memory latency effectively, reducing the practical benefit of high occupancy.

### Detecting bank conflicts

#### Visual pattern recognition

You can often predict bank conflicts by analyzing access patterns:

**Sequential access (no conflicts):**
```mojo
# Thread ID:  0  1  2  3  ...  31
# Address:    0  4  8 12  ... 124
# Bank:       0  1  2  3  ...  31  OK All different banks
```

**Stride-2 access (2-way conflicts):**
```mojo
# Thread ID:  0  1  2  3  ...  15 16 17 18 ... 31
# Address:    0  8 16 24  ... 120  4 12 20 ... 124
# Bank:       0  2  4  6  ...  30  1  3  5 ...  31
# Conflict:   Banks 0,2,4... have 2 threads each  X
```

**Stride-32 access (32-way conflicts):**
```mojo
# Thread ID:  0   1   2   3  ...  31
# Address:    0  128 256 384 ... 3968
# Bank:       0   0   0   0  ...   0  X All threads->Bank 0
```

#### Profiling with NSight Compute (`ncu`)

Building on the profiling methodology from Puzzle 30, you can measure bank conflicts quantitatively:

```bash
# Key metrics for shared memory bank conflicts
ncu --metrics=l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st your_kernel

# Additional context metrics
ncu --metrics=smsp__sass_average_branch_targets_threads_uniform.pct your_kernel
ncu --metrics=smsp__warps_issue_stalled_membar_per_warp_active.pct your_kernel
```

The `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` and `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` metrics directly count the number of bank conflicts for load and store operations during kernel execution. Combined with the number of shared memory accesses, these give you the conflict ratio - a critical performance indicator.

### When bank conflicts matter most

#### Compute-intensive kernels

Bank conflicts have the greatest impact on kernels where:
- Shared memory is accessed frequently within tight loops
- Computational work per shared memory access is minimal
- The kernel is compute-bound rather than memory-bound

**Example scenarios:**
- Matrix multiplication inner loops (like the tiled versions in Puzzle 16)
- Stencil computations with shared memory caching
- Parallel reduction operations

#### Memory-bound vs compute-bound trade-offs

Just as Puzzle 31 showed that occupancy matters less for memory-bound workloads, bank conflicts matter less when your kernel is bottlenecked by global memory bandwidth or arithmetic intensity is very low.

However, many kernels that use shared memory do so precisely **because**they want to shift from memory-bound to compute-bound execution. In these cases, bank conflicts can prevent you from achieving the performance gains that motivated using shared memory in the first place.

### The path forward

Understanding shared memory banking gives you the foundation to:
1. **Predict performance**before writing code by analyzing access patterns
2. **Diagnose slowdowns**using systematic profiling approaches
3. **Design conflict-free algorithms**that maintain high shared memory throughput
4. **Make informed trade-offs**between algorithm complexity and memory efficiency

In the next section, you'll apply this knowledge through hands-on exercises that demonstrate common conflict patterns and their solutions - turning theoretical understanding into practical optimization skills.

## Conflict-Free Patterns

> **Note: This section is specific to NVIDIA GPUs**
>
> Bank conflict analysis and profiling techniques covered here apply specifically to NVIDIA GPUs. The profiling commands use NSight Compute tools that are part of the NVIDIA CUDA toolkit.

### Building on your profiling skills

You've learned GPU profiling fundamentals in Puzzle 30 and understood resource optimization in Puzzle 31. Now you're ready to apply those detective skills to a new performance mystery: **shared memory bank conflicts**.

**The detective challenge:**You have two GPU kernels that perform identical mathematical operations (`(input + 10) * 2`). Both produce exactly the same results. Both use the same amount of shared memory. Both have identical occupancy. Yet one experiences systematic performance degradation due to **how**it accesses shared memory.

**Your mission:**Use the profiling methodology you've learned to uncover this hidden performance trap and understand when bank conflicts matter in real-world GPU programming.

### Overview

Shared memory bank conflicts occur when multiple threads in a warp simultaneously access different addresses within the same memory bank. This detective case explores two kernels with contrasting access patterns:

```mojo
comptime SIZE = 8 * 1024  # 8K elements - small enough to focus on shared memory patterns
comptime TPB = 256  # Threads per block - divisible by 32 (warp size)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime BLOCKS_PER_GRID = (SIZE // TPB, 1)
comptime dtype = DType.float32
comptime layout = Layout.row_major(SIZE)

fn no_conflict_kernel
    layout: Layout
:
    """Perfect shared memory access - no bank conflicts.

    Each thread accesses a different bank: thread_idx.x maps to bank thread_idx.x % 32.
    This achieves optimal shared memory bandwidth utilization.
    """

    # Shared memory buffer - each thread loads one element
    shared_buf = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = thread_idx.x

    # Load from global memory to shared memory - no conflicts
    if global_i < size:
        shared_buf[local_i] = (
            input[global_i] + 10.0
        )  # Add 10 as simple operation

    barrier()  # Synchronize shared memory writes

    # Read back from shared memory and write to output - no conflicts
    if global_i < size:
        output[global_i] = shared_buf[local_i] * 2.0  # Multiply by 2

    barrier()  # Ensure completion

```

```mojo
fn two_way_conflict_kernel
    layout: Layout
:
    """Stride-2 shared memory access - creates 2-way bank conflicts.

    Threads 0,16 -> Bank 0, Threads 1,17 -> Bank 1, etc.
    Each bank serves 2 threads, doubling access time.
    """

    # Shared memory buffer - stride-2 access pattern creates conflicts
    shared_buf = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = thread_idx.x

    # CONFLICT: stride-2 access creates 2-way bank conflicts
    conflict_index = (local_i * 2) % TPB

    # Load with bank conflicts
    if global_i < size:
        shared_buf[conflict_index] = (
            input[global_i] + 10.0
        )  # Same operation as no-conflict

    barrier()  # Synchronize shared memory writes

    # Read back with same conflicts
    if global_i < size:
        output[global_i] = (
            shared_buf[conflict_index] * 2.0
        )  # Same operation as no-conflict

    barrier()  # Ensure completion

```

**The mystery:**These kernels compute identical results but have dramatically different shared memory access efficiency. Your job is to discover why using systematic profiling analysis.

### Configuration

**Requirements:**

- NVIDIA GPU with CUDA toolkit and NSight Compute from Puzzle 30
- Understanding of shared memory banking concepts from the [previous section](#understanding-shared-memory-banks)

**Kernel specifications:**

```mojo
comptime SIZE = 8 * 1024      # 8K elements - focus on shared memory patterns
comptime TPB = 256            # 256 threads per block (8 warps)
comptime BLOCKS_PER_GRID = (SIZE // TPB, 1)  # 32 blocks
```

**Key insight:**The problem size is deliberately smaller than previous puzzles to highlight shared memory effects rather than global memory bandwidth limitations.

### The investigation

#### Step 1: Verify correctness

```bash
pixi shell -e nvidia
mojo problems/p32/p32.mojo --test
```

Both kernels should produce identical results. This confirms that bank conflicts affect **performance**but not **correctness**.

#### Step 2: Benchmark performance baseline

```bash
mojo problems/p32/p32.mojo --benchmark
```

Record the execution times. You may notice similar performance due to the workload being dominated by global memory access, but bank conflicts will be revealed through profiling metrics.

#### Step 3: Build for profiling

```bash
mojo build --debug-level=full problems/p32/p32.mojo -o problems/p32/p32_profiler
```

#### Step 4: Profile bank conflicts

Use NSight Compute to measure shared memory bank conflicts quantitatively:

```bash
# Profile no-conflict kernel
ncu --metrics=l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st problems/p32/p32_profiler --no-conflict

```

and

```bash
# Profile two-way conflict kernel
ncu --metrics=l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st problems/p32/p32_profiler --two-way
```

**Key metrics to record:**

- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` - Load conflicts
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` - Store conflicts

#### Step 5: Analyze access patterns

Based on your profiling results, analyze the mathematical access patterns:

**No-conflict kernel access pattern:**

```mojo
# Thread mapping: thread_idx.x directly maps to shared memory index
shared_buf[thread_idx.x]  # Thread 0->Index 0, Thread 1->Index 1, etc.
# Bank mapping: Index % 32 = Bank ID
# Result: Thread 0->Bank 0, Thread 1->Bank 1, ..., Thread 31->Bank 31
```

**Two-way conflict kernel access pattern:**

```mojo
# Thread mapping with stride-2 modulo operation
shared_buf[(thread_idx.x * 2) % TPB]
# For threads 0-31: Index 0,2,4,6,...,62, then wraps to 64,66,...,126, then 0,2,4...
# Bank mapping examples:
# Thread 0  -> Index 0   -> Bank 0
# Thread 16 -> Index 32  -> Bank 0  (conflict!)
# Thread 1  -> Index 2   -> Bank 2
# Thread 17 -> Index 34  -> Bank 2  (conflict!)
```

### Example goal: solve the bank conflict mystery

**After completing the investigation steps above, answer these analysis questions:**

#### Performance analysis (Steps 1-2)

1. Do both kernels produce identical mathematical results?
2. What are the execution time differences (if any) between the kernels?
3. Why might performance be similar despite different access patterns?

#### Bank conflict profiling (Step 4)

4. How many bank conflicts does the no-conflict kernel generate for loads and stores?
5. How many bank conflicts does the two-way conflict kernel generate for loads and stores?
6. What is the total conflict count difference between the kernels?

#### Access pattern analysis (Step 5)

7. In the no-conflict kernel, which bank does Thread 0 access? Thread 31?
8. In the two-way conflict kernel, which threads access Bank 0? Which access Bank 2?
9. How many threads compete for the same bank in the conflict kernel?

#### The bank conflict detective work

10. Why does the two-way conflict kernel show measurable conflicts while the no-conflict kernel shows zero?
11. How does the stride-2 access pattern `(thread_idx.x * 2) % TPB` create systematic conflicts?
12. Why do bank conflicts matter more in compute-intensive kernels than memory-bound kernels?

#### Real-world implications

13. When would you expect bank conflicts to significantly impact application performance?
14. How can you predict bank conflict patterns before implementing shared memory algorithms?
15. What design principles help avoid bank conflicts in matrix operations and stencil computations?

#### Tips

**Bank conflict detective toolkit:**

- **NSight Compute metrics**- Quantify conflicts with precise measurements
- **Access pattern visualization**- Map thread indices to banks systematically
- **Mathematical analysis**- Use modulo arithmetic to predict conflicts
- **Workload characteristics**- Understand when conflicts matter vs when they don't

**Key investigation principles:**

- **Measure systematically:**Use profiling tools rather than guessing about conflicts
- **Visualize access patterns:**Draw thread-to-bank mappings for complex algorithms
- **Consider workload context:**Bank conflicts matter most in compute-intensive shared memory algorithms
- **Think prevention:**Design algorithms with conflict-free access patterns from the start

**Access pattern analysis approach:**

1. **Map threads to indices:**Understand the mathematical address calculation
2. **Calculate bank assignments:**Use the formula `bank_id = (address / 4) % 32`
3. **Identify conflicts:**Look for multiple threads accessing the same bank
4. **Validate with profiling:**Confirm theoretical analysis with NSight Compute measurements

**Common conflict-free patterns:**

- **Sequential access:**`shared[thread_idx.x]` - each thread different bank
- **Broadcast access:**`shared[0]` for all threads - hardware optimization
- **Power-of-2 strides:**Stride-32 often maps cleanly to banking patterns
- **Padded arrays:**Add padding to shift problematic access patterns

### Reference implementation (example)

#### Complete Solution with Bank Conflict Analysis

This bank conflict detective case demonstrates how shared memory access patterns affect GPU performance and reveals the importance of systematic profiling for optimization.

### **Investigation results from profiling**

**Step 1: Correctness Verification**
Both kernels produce identical mathematical results:

```
OK No-conflict kernel: PASSED
OK Two-way conflict kernel: PASSED
OK Both kernels produce identical results
```

**Step 2: Performance Baseline**
Benchmark results show similar execution times:

```
| name             | met (ms)           | iters |
| ---------------- | ------------------ | ----- |
| no_conflict      | 2.1930616745886655 | 547   |
| two_way_conflict | 2.1978922967032966 | 546   |
```

**Key insight:**Performance is nearly identical (~2.19ms vs ~2.20ms) because this workload is **global memory bound**rather than shared memory bound. Bank conflicts become visible through profiling metrics rather than execution time.

### **Bank conflict profiling evidence**

**No-Conflict Kernel (Optimal Access Pattern):**

```
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum    0
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum    0
```

**Result:**Zero conflicts for both loads and stores - perfect shared memory efficiency.

**Two-Way Conflict Kernel (Problematic Access Pattern):**

```
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum    256
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum    256
```

**Result:**256 conflicts each for loads and stores - clear evidence of systematic banking problems.

**Total conflict difference:**512 conflicts (256 + 256) demonstrate measurable shared memory inefficiency.

### **Access pattern mathematical analysis**

#### No-conflict kernel access pattern

**Thread-to-index mapping:**

```mojo
shared_buf[thread_idx.x]
```

**Bank assignment analysis:**

```
Thread 0  -> Index 0   -> Bank 0 % 32 = 0
Thread 1  -> Index 1   -> Bank 1 % 32 = 1
Thread 2  -> Index 2   -> Bank 2 % 32 = 2
...
Thread 31 -> Index 31  -> Bank 31 % 32 = 31
```

**Result:**Perfect bank distribution - each thread accesses a different bank within each warp, enabling parallel access.

#### Two-way conflict kernel access pattern

**Thread-to-index mapping:**

```mojo
shared_buf[(thread_idx.x * 2) % TPB]  # TPB = 256
```

**Bank assignment analysis for first warp (threads 0-31):**

```
Thread 0  -> Index (0*2)%256 = 0   -> Bank 0
Thread 1  -> Index (1*2)%256 = 2   -> Bank 2
Thread 2  -> Index (2*2)%256 = 4   -> Bank 4
...
Thread 16 -> Index (16*2)%256 = 32 -> Bank 0  <- CONFLICT with Thread 0
Thread 17 -> Index (17*2)%256 = 34 -> Bank 2  <- CONFLICT with Thread 1
Thread 18 -> Index (18*2)%256 = 36 -> Bank 4  <- CONFLICT with Thread 2
...
```

**Conflict pattern:**Each bank serves exactly 2 threads, creating systematic 2-way conflicts across all 32 banks.

**Mathematical explanation:**The stride-2 pattern with modulo 256 creates a repeating access pattern where:

- Threads 0-15 access banks 0,2,4,...,30
- Threads 16-31 access the **same banks**0,2,4,...,30
- Each bank collision requires hardware serialization

### **Why this matters: workload context analysis**

#### Memory-bound vs compute-bound implications

**This workload characteristics:**

- **Global memory dominant:**Each thread performs minimal computation relative to memory transfer
- **Shared memory secondary:**Bank conflicts add overhead but don't dominate total execution time
- **Identical performance:**Global memory bandwidth saturation masks shared memory inefficiency

**When bank conflicts matter most:**

1. **Compute-intensive shared memory algorithms**- Matrix multiplication, stencil computations, FFT
2. **Tight computational loops**- Repeated shared memory access within inner loops
3. **High arithmetic intensity**- Significant computation per memory access
4. **Large shared memory working sets**- Algorithms that heavily utilize shared memory caching

#### Real-world performance implications

**Applications where bank conflicts significantly impact performance:**

**Matrix Multiplication:**

```mojo
# Problematic: All threads in warp access same column
for k in range(tile_size):
    acc += a_shared[local_row, k] * b_shared[k, local_col]  # b_shared[k, 0] conflicts
```

**Stencil Computations:**

```mojo
# Problematic: Stride access in boundary handling
shared_buf[thread_idx.x * stride]  # Creates systematic conflicts
```

**Parallel Reductions:**

```mojo
# Problematic: Power-of-2 stride patterns
if thread_idx.x < stride:
    shared_buf[thread_idx.x] += shared_buf[thread_idx.x + stride]  # Conflict potential
```

### **Conflict-free design principles**

#### Prevention strategies

**1. Sequential access patterns:**

```mojo
shared[thread_idx.x]  # Optimal - each thread different bank
```

**2. Broadcast optimization:**

```mojo
constant = shared[0]  # All threads read same address - hardware optimized
```

**3. Padding techniques:**

```mojo
shared = LayoutTensor[dtype, Layout.row_major(TPB + 1), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()  # Shift access patterns
```

**4. Access pattern analysis:**

- Calculate bank assignments before implementation
- Use modulo arithmetic: `bank_id = (address_bytes / 4) % 32`
- Visualize thread-to-bank mappings for complex algorithms

#### Systematic optimization workflow

**Design Phase:**

1. **Plan access patterns**- Sketch thread-to-memory mappings
2. **Calculate bank assignments**- Use mathematical analysis
3. **Predict conflicts**- Identify problematic access patterns
4. **Design alternatives**- Consider padding, transpose, or algorithm changes

**Implementation Phase:**

1. **Profile systematically**- Use NSight Compute conflict metrics
2. **Measure impact**- Compare conflict counts across implementations
3. **Validate performance**- Ensure optimizations improve end-to-end performance
4. **Document patterns**- Record successful conflict-free algorithms for reuse

### **Key takeaways: from detective work to optimization expertise**

**The Bank Conflict Investigation revealed:**

1. **Measurement trumps intuition**- Profiling tools reveal conflicts invisible to performance timing
2. **Pattern analysis works**- Mathematical prediction accurately matched NSight Compute results
3. **Context matters**- Bank conflicts matter most in compute-intensive shared memory workloads
4. **Prevention beats fixing**- Designing conflict-free patterns easier than retrofitting optimizations

**Universal shared memory optimization principles:**

**When to worry about bank conflicts:**

- **High-computation kernels**using shared memory for data reuse
- **Iterative algorithms**with repeated shared memory access in tight loops
- **Performance-critical code**where every cycle matters
- **Memory-intensive operations**that are compute-bound rather than bandwidth-bound

**When bank conflicts are less critical:**

- **Memory-bound workloads**where global memory dominates performance
- **Simple caching scenarios**with minimal shared memory reuse
- **One-time access patterns**without repeated conflict-prone operations

**Professional development methodology:**

1. **Profile before optimizing**- Measure conflicts quantitatively with NSight Compute
2. **Understand access mathematics**- Use bank assignment formulas to predict problems
3. **Design systematically**- Consider banking in algorithm design, not as afterthought
4. **Validate optimizations**- Confirm that conflict reduction improves actual performance

This detective case demonstrates that **systematic profiling reveals optimization opportunities invisible to performance timing alone**- bank conflicts are a perfect example of where measurement-driven optimization beats guesswork.
