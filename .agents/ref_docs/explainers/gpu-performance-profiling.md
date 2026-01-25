---
title: "GPU Performance Profiling"
description: "GPU performance profiling transforms correct code into high-performance code through systematic analysis. This chapter explores professional profiling tools and detective methodologies used in production GPU development."
---

# GPU Performance Profiling

GPU performance profiling transforms correct code into high-performance code through systematic analysis. This chapter explores professional profiling tools and detective methodologies used in production GPU development.

> **Beyond Correct Code**
>
> Note: **This part is specific to compatible NVIDIA GPUs**
>
> This chapter introduces **systematic performance analysis**that transforms working GPU code into high-performance code. Unlike previous puzzles that focused on correctness and GPU features, these challenges explore **profiling methodologies**used in production GPU software development.
>
>
> **What you'll learn:**
>
> - **Professional profiling tools**: NSight Systems and NSight Compute for comprehensive performance analysis
> - **Performance detective work**: Using profiler data to identify bottlenecks and optimization opportunities
> - **Memory system insights**: Understanding how memory access patterns dramatically impact performance
> - **Counter-intuitive discoveries**: Learning when "good" metrics actually indicate performance problems
> - **Evidence-based optimization**: Making optimization decisions based on profiler data, not assumptions
>
> **Why this matters:**Most GPU tutorials teach basic performance concepts, but real-world GPU development requires **systematic profiling methodologies**to identify actual bottlenecks, understand memory system behavior, and make informed optimization decisions. These skills bridge the gap between academic examples and production GPU computing.

## Overview

GPU performance profiling transforms correct code into high-performance code through systematic analysis. This chapter explores professional profiling tools and detective methodologies used in production GPU development.

**Core learning objectives:**

- **Learn profiling tool selection**and understand when to use NSight Systems vs NSight Compute
- **Develop performance detective skills**using real profiler output to identify bottlenecks
- **Discover counter-intuitive insights**about GPU memory systems and caching behavior
- **Learn evidence-based optimization**based on profiler data rather than assumptions

## Key concepts

**Professional profiling tools:**

- **[NSight Systems](https://developer.nvidia.com/nsight-systems) (`nsys`)**: System-wide timeline analysis for CPU-GPU coordination and memory transfers
- **[NSight Compute](https://developer.nvidia.com/nsight-compute) (`ncu`)**: Detailed kernel analysis for memory efficiency and compute utilization
- **Systematic methodology**: Evidence-based bottleneck identification and optimization validation

**Key insights you'll discover:**

- **Counter-intuitive behavior**: When high cache hit rates actually indicate poor performance
- **Memory access patterns**: How coalescing dramatically impacts bandwidth utilization
- **Tool-guided optimization**: Using profiler data to make decisions rather than performance assumptions

## Configuration

**Requirements:**

- **NVIDIA GPU**: CUDA-compatible hardware with profiling enabled
- **CUDA Toolkit**: NSight Systems and NSight Compute tools
- **Build setup**: Optimized code with debug info (`--debug-level=full`)

**Methodology:**

1. **System-wide analysis**with NSight Systems to identify major bottlenecks
2. **Kernel deep-dives**with NSight Compute for memory system analysis
3. **Evidence-based conclusions**using profiler data to guide optimization

## Puzzle structure

This chapter contains two interconnected components that build upon each other:

### **[NVIDIA Profiling Basics Tutorial](#nvidia-profiling-basics)**

Learn the essential NVIDIA profiling ecosystem through hands-on examples with actual profiler output.

**You'll learn:**

- NSight Systems for system-wide timeline analysis and bottleneck identification
- NSight Compute for detailed kernel analysis and memory system insights
- Professional profiling workflows and best practices from production GPU development

### **[The Cache Hit Paradox Detective Case](#the-cache-hit-paradox)**

Apply profiling skills to solve a performance mystery where three identical vector addition kernels have dramatically different performance.

**Example scenario:**Discover why the kernel with the **highest cache hit rates**has the **worst performance**- a counter-intuitive insight that challenges traditional CPU-based performance thinking.

**Detective skills:**Use real NSight Systems and NSight Compute data to understand memory coalescing effects and evidence-based optimization.

## Getting started

**Learning path:**

1. **[Profiling Basics Tutorial](#nvidia-profiling-basics)**- Learn NSight Systems and NSight Compute
2. **[Cache Hit Paradox Detective Case](#the-cache-hit-paradox)**- Apply skills to solve performance mysteries

**Prerequisites:**

- GPU memory hierarchies and access patterns
- GPU programming fundamentals (threads, blocks, warps, shared memory)
- Command-line profiling tools experience

**Learning outcome:**Professional-level profiling skills for systematic bottleneck identification and evidence-based optimization used in production GPU development.

This chapter teaches that **systematic profiling reveals truths that intuition misses**- GPU performance optimization requires tool-guided discovery rather than assumptions.

**Additional resources:**

- [NVIDIA CUDA Best Practices Guide - Profiling](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#profiling)
- [NSight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [NSight Compute CLI User Guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)

##  NVIDIA Profiling Basics

### Overview

You've learned GPU programming fundamentals and advanced patterns. Part II taught you debugging techniques for *
*correctness**using `compute-sanitizer` and `cuda-gdb`, while other parts covered different GPU features like warp
programming, memory systems, and block-level operations. Your kernels work correctly - but are they **fast**?

> This tutorial follows NVIDIA's recommended profiling methodology from
> the [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#profiling).

**Key Insight**: A correct kernel can still be orders of magnitude slower than optimal. Profiling bridges the gap
between working code and high-performance code.

### The profiling toolkit

Since you have `cuda-toolkit` via pixi, you have access to NVIDIA's professional profiling suite:

#### NSight Systems (`nsys`) - the "big picture" tool

**Purpose**: System-wide performance analysis ([NSight Systems Documentation](https://docs.nvidia.com/nsight-systems/))

- Timeline view of CPU-GPU interaction
- Memory transfer bottlenecks
- Kernel launch overhead
- Multi-GPU coordination
- API call tracing

**Available interfaces**: Command-line (`nsys`) and GUI (`nsys-ui`)

**Use when**:

- Understanding overall application flow
- Identifying CPU-GPU synchronization issues
- Analyzing memory transfer patterns
- Finding kernel launch bottlenecks

```bash
# See the help
pixi run nsys --help

# Basic system-wide profiling
pixi run nsys profile --trace=cuda,nvtx --output=timeline mojo your_program.mojo

# Interactive analysis
pixi run nsys stats --force-export=true timeline.nsys-rep
```

#### NSight Compute (`ncu`) - the "kernel deep-dive" tool

**Purpose**: Detailed single-kernel performance
analysis ([NSight Compute Documentation](https://docs.nvidia.com/nsight-compute/))

- Roofline model analysis
- Memory hierarchy utilization
- Warp execution efficiency
- Register/shared memory usage
- Compute unit utilization

**Available interfaces**: Command-line (`ncu`) and GUI (`ncu-ui`)

**Use when**:

- Optimizing specific kernel performance
- Understanding memory access patterns
- Analyzing compute vs memory bound kernels
- Identifying warp divergence issues

```bash
# See the help
pixi run ncu --help

# Detailed kernel profiling
pixi run ncu --set full --output kernel_profile mojo your_program.mojo

# Focus on specific kernels
pixi run ncu --kernel-name regex:your_kernel_name mojo your_program.mojo
```

### Tool selection decision tree

```
Performance Problem
        |
        v
Know which kernel?
    |           |
   No          Yes
    |           |
    v           v
NSight    Kernel-specific issue?
Systems       |           |
    |        No          Yes
    v         |           |
Timeline      |           v
Analysis <----+     NSight Compute
                          |
                          v
                   Kernel Deep-Dive
```

**Quick Decision Guide**:

- **Start with NSight Systems (`nsys`)**if you're unsure where the bottleneck is
- **Use NSight Compute (`ncu`)**when you know exactly which kernel to optimize
- **Use both**for comprehensive analysis (common workflow)

### Hands-on: system-wide profiling with NSight Systems

Let's profile the Matrix Multiplication implementations from Puzzle 16 to understand
performance differences.

> **GUI Note**: The NSight Systems and Compute GUIs (`nsys-ui`, `ncu-ui`) require a display and OpenGL support. On
> headless servers or remote systems without X11 forwarding, use the command-line versions (`nsys`, `ncu`) with text-based
> analysis via `nsys stats` and `ncu --import --page details`. You can also transfer `.nsys-rep` and `.ncu-rep` files to
> local machines for GUI analysis.

#### Step 1: Prepare your code for profiling

**Critical**: For accurate profiling, build with full debug information while keeping optimizations enabled:

```bash
pixi shell -e nvidia
# Build with full debug info (for comprehensive source mapping) with optimizations enabled
mojo build --debug-level=full solutions/p16/p16.mojo -o solutions/p16/p16_optimized

# Test the optimized build
./solutions/p16/p16_optimized --naive
```

**Why this matters**:

- **Full debug info**: Provides complete symbol tables, variable names, and source line mapping for profilers
- **Comprehensive analysis**: Enables NSight tools to correlate performance data with specific code locations
- **Optimizations enabled**: Ensures realistic performance measurements that match production builds

#### Step 2: Capture system-wide profile

```bash
# Profile the optimized build with comprehensive tracing
nsys profile \
  --trace=cuda,nvtx \
  --output=matmul_naive \
  --force-overwrite=true \
  ./solutions/p16/p16_optimized --naive
```

**Command breakdown**:

- `--trace=cuda,nvtx`: Capture CUDA API calls and custom annotations
- `--output=matmul_naive`: Save profile as `matmul_naive.nsys-rep`
- `--force-overwrite=true`: Replace existing profiles
- Final argument: Your Mojo program

#### Step 3: Analyze the timeline

```bash
# Generate text-based statistics
nsys stats --force-export=true matmul_naive.nsys-rep

# Key metrics to look for:
# - GPU utilization percentage
# - Memory transfer times
# - Kernel execution times
# - CPU-GPU synchronization gaps
```

**What you'll see**(actual output from a 22 matrix multiplication):

```txt
** CUDA API Summary (cuda_api_sum):
 Time (%)  Total Time (ns)  Num Calls  Avg (ns)   Med (ns)  Min (ns)  Max (ns)  StdDev (ns)          Name
 --------  ---------------  ---------  ---------  --------  --------  --------  -----------  --------------------
     81.9          8617962          3  2872654.0    2460.0      1040   8614462    4972551.6  cuMemAllocAsync
     15.1          1587808          4   396952.0    5965.5      3810   1572067     783412.3  cuMemAllocHost_v2
      0.6            67152          1    67152.0   67152.0     67152     67152          0.0  cuModuleLoadDataEx
      0.4            44961          1    44961.0   44961.0     44961     44961          0.0  cuLaunchKernelEx

** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):
 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                    Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------
    100.0             1920          1    1920.0    1920.0      1920      1920          0.0  p16_naive_matmul_Layout_Int6A6AcB6A6AsA6A6A

** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):
 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)           Operation
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ----------------------------
     49.4             4224      3    1408.0    1440.0      1312      1472         84.7  [CUDA memcpy Device-to-Host]
     36.0             3072      4     768.0     528.0       416      1600        561.0  [CUDA memset]
     14.6             1248      3     416.0     416.0       416       416          0.0  [CUDA memcpy Host-to-Device]
```

**Key Performance Insights**:

- **Memory allocation dominates**: 81.9% of total time spent on `cuMemAllocAsync`
- **Kernel is lightning fast**: Only 1,920 ns (0.000001920 seconds) execution time
- **Memory transfer breakdown**: 49.4% DeviceHost, 36.0% memset, 14.6% HostDevice
- **Tiny data sizes**: All memory operations are < 0.001 MB (4 float32 values = 16 bytes)

#### Step 4: Compare implementations

Profile different versions and compare:

```bash
# Make sure you've in pixi shell still `pixi run -e nvidia`

# Profile shared memory version
nsys profile --trace=cuda,nvtx --force-overwrite=true --output=matmul_shared ./solutions/p16/p16_optimized --single-block

# Profile tiled version
nsys profile --trace=cuda,nvtx --force-overwrite=true --output=matmul_tiled ./solutions/p16/p16_optimized --tiled

# Profile idiomatic tiled version
nsys profile --trace=cuda,nvtx --force-overwrite=true --output=matmul_idiomatic_tiled ./solutions/p16/p16_optimized --idiomatic-tiled

# Analyze each implementation separately (nsys stats processes one file at a time)
nsys stats --force-export=true matmul_shared.nsys-rep
nsys stats --force-export=true matmul_tiled.nsys-rep
nsys stats --force-export=true matmul_idiomatic_tiled.nsys-rep
```

**How to compare the results**:

1. **Look at GPU Kernel Summary**- Compare execution times between implementations
2. **Check Memory Operations**- See if shared memory reduces global memory traffic
3. **Compare API overhead**- All should have similar memory allocation patterns

**Manual comparison workflow**:

```bash
# Run each analysis and save output for comparison
nsys stats --force-export=true matmul_naive.nsys-rep > naive_stats.txt
nsys stats --force-export=true matmul_shared.nsys-rep > shared_stats.txt
nsys stats --force-export=true matmul_tiled.nsys-rep > tiled_stats.txt
nsys stats --force-export=true matmul_idiomatic_tiled.nsys-rep > idiomatic_tiled_stats.txt
```

**Fair Comparison Results**(actual output from profiling):

#### Comparison 1: 2 x 2 matrices

| Implementation                | Memory Allocation     | Kernel Execution | Performance      |
|-------------------------------|-----------------------|------------------|------------------|
| **Naive**| 81.9% cuMemAllocAsync |  1,920 ns       | Baseline         |
| **Shared**(`--single-block`) | 81.8% cuMemAllocAsync |  1,984 ns       | **+3.3% slower**|

#### Comparison 2: 9 x 9 matrices

| Implementation      | Memory Allocation     | Kernel Execution | Performance       |
|---------------------|-----------------------|------------------|-------------------|
| **Tiled**(manual)  | 81.1% cuMemAllocAsync |  2,048 ns       | Baseline          |
| **Idiomatic Tiled**| 81.6% cuMemAllocAsync |  2,368 ns       | **+15.6% slower**|

**Key Insights from Fair Comparisons**:

**Both Matrix Sizes Are Tiny for GPU Work!**:

- **22 matrices**: 4 elements - completely overhead-dominated
- **99 matrices**: 81 elements - still completely overhead-dominated
- **Real GPU workloads**: Thousands to millions of elements per dimension

**What These Results Actually Show**:

- **All variants dominated by memory allocation**(>81% of time)
- **Kernel execution is irrelevant**compared to setup costs
- **"Optimizations" can hurt**: Shared memory adds 3.3% overhead, async_copy adds 15.6%
- **The real lesson**: For tiny workloads, algorithm choice doesn't matter - overhead dominates everything

**Why This Happens**:

- GPU setup cost (memory allocation, kernel launch) is fixed regardless of problem size
- For tiny problems, this fixed cost dwarfs computation time
- Optimizations designed for large problems become overhead for small ones

**Real-World Profiling Lessons**:

- **Problem size context matters**: Both 22 and 99 are tiny for GPUs
- **Fixed costs dominate small problems**: Memory allocation, kernel launch overhead
- **"Optimizations" can hurt tiny workloads**: Shared memory, async operations add overhead
- **Don't optimize tiny problems**: Focus on algorithms that scale to real workloads
- **Always benchmark**: Assumptions about "better" code are often wrong

**Understanding Small Kernel Profiling**:
This 22 matrix example demonstrates a **classic small-kernel pattern**:

- The actual computation (matrix multiply) is extremely fast (1,920 ns)
- Memory setup overhead dominates the total time (97%+ of execution)
- This is why **real-world GPU optimization**focuses on:
  - **Batching operations**to amortize setup costs
  - **Memory reuse**to reduce allocation overhead
  - **Larger problem sizes**where compute becomes the bottleneck

### Hands-on: kernel deep-dive with NSight Compute

Now let's dive deep into a specific kernel's performance characteristics.

#### Step 1: Profile a specific kernel

```bash
# Make sure you're in an active shell
pixi shell -e nvidia

# Profile the naive MatMul kernel in detail (using our optimized build)
ncu \
  --set full \
  -o kernel_analysis \
  --force-overwrite \
  ./solutions/p16/p16_optimized --naive
```

> **Common Issue: Permission Error**
>
> If you get `ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters`, try
> these > solutions:
>
> ```bash
> # Add NVIDIA driver option (safer than rmmod)
> echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | sudo tee -a /etc/modprobe.d/nvidia-kernel-common.conf
>
> # Set kernel parameter
> sudo sysctl -w kernel.perf_event_paranoid=0
>
> # Make permanent
> echo 'kernel.perf_event_paranoid=0' | sudo tee -a /etc/sysctl.conf
>
> # Reboot required for driver changes to take effect
> sudo reboot
>
> # Then run the ncu command again
> ncu \
>   --set full \
>   -o kernel_analysis \
>   --force-overwrite \
>   ./solutions/p16/p16_optimized --naive
> ```

#### Step 2: Analyze key metrics

```bash
# Generate detailed report (correct syntax)
ncu --import kernel_analysis.ncu-rep --page details
```

**Real NSight Compute Output**(from your 22 naive MatMul):

```txt
GPU Speed Of Light Throughput
----------------------- ----------- ------------
DRAM Frequency              Ghz         6.10
SM Frequency                Ghz         1.30
Elapsed Cycles            cycle         3733
Memory Throughput             %         1.02
DRAM Throughput               %         0.19
Duration                     us         2.88
Compute (SM) Throughput       %         0.00
----------------------- ----------- ------------

Launch Statistics
-------------------------------- --------------- ---------------
Block Size                                                     9
Grid Size                                                      1
Threads                           thread               9
Waves Per SM                                                0.00
-------------------------------- --------------- ---------------

Occupancy
------------------------------- ----------- ------------
Theoretical Occupancy                 %        33.33
Achieved Occupancy                    %         2.09
------------------------------- ----------- ------------
```

**Critical Insights from Real Data**:

##### Performance analysis - the brutal truth

- **Compute Throughput: 0.00%**- GPU is completely idle computationally
- **Memory Throughput: 1.02%**- Barely touching memory bandwidth
- **Achieved Occupancy: 2.09%**- Using only 2% of GPU capability
- **Grid Size: 1 block**- Completely underutilizing 80 multiprocessors!

##### Why performance is so poor

- **Tiny problem size**: 22 matrix = 4 elements total
- **Poor launch configuration**: 9 threads in 1 block (should be multiples of 32)
- **Massive underutilization**: 0.00 waves per SM (need thousands for efficiency)

##### Key optimization recommendations from NSight Compute

- **"Est. Speedup: 98.75%"**- Increase grid size to use all 80 SMs
- **"Est. Speedup: 71.88%"**- Use thread blocks as multiples of 32
- **"Kernel grid is too small"**- Need much larger problems for GPU efficiency

#### Step 3: The reality check

**What This Profiling Data Teaches Us**:

1. **Tiny problems are GPU poison**: 22 matrices completely waste GPU resources
2. **Launch configuration matters**: Wrong thread/block sizes kill performance
3. **Scale matters more than algorithm**: No optimization can fix a fundamentally tiny problem
4. **NSight Compute is honest**: It tells us when our kernel performance is poor

**The Real Lesson**:

- **Don't optimize toy problems**- they're not representative of real GPU workloads
- **Focus on realistic workloads**- 10001000+ matrices where optimizations actually matter
- **Use profiling to guide optimization**- but only on problems worth optimizing

**For our tiny 22 example**: All the sophisticated algorithms (shared memory, tiling) just add overhead to an already
overhead-dominated workload.

### Reading profiler output like a performance detective

#### Common performance patterns

##### Pattern 1: Memory-bound kernel

**NSight Systems shows**: Long memory transfer times
**NSight Compute shows**: High memory throughput, low compute utilization
**Solution**: Optimize memory access patterns, use shared memory

##### Pattern 2: Low occupancy

**NSight Systems shows**: Short kernel execution with gaps
**NSight Compute shows**: Low achieved occupancy
**Solution**: Reduce register usage, optimize block size

##### Pattern 3: Warp divergence

**NSight Systems shows**: Irregular kernel execution patterns
**NSight Compute shows**: Low warp execution efficiency
**Solution**: Minimize conditional branches, restructure algorithms

#### Profiling detective workflow

```
Performance Issue
        |
        v
NSight Systems: Big Picture
        |
        v
GPU Well Utilized?
    |           |
   No          Yes
    |           |
    v           v
Fix CPU-GPU    NSight Compute: Kernel Detail
Pipeline            |
                    v
            Memory or Compute Bound?
                |       |       |
             Memory  Compute  Neither
                |       |       |
                v       v       v
           Optimize  Optimize  Check
           Memory    Arithmetic Occupancy
           Access
```

### Profiling best practices

For comprehensive profiling guidelines, refer to
the [Best Practices Guide - Performance Metrics](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#performance-metrics).

#### Do's

1. **Profile representative workloads**: Use realistic data sizes and patterns
2. **Build with full debug info**: Use `--debug-level=full` for comprehensive profiling data and source mapping with
   optimizations
3. **Warm up the GPU**: Run kernels multiple times, profile later iterations
4. **Compare alternatives**: Always profile multiple implementations
5. **Focus on hotspots**: Optimize the kernels that take the most time

#### Don'ts

1. **Don't profile without debug info**: You won't be able to map performance back to source code (`mojo build --help`)
2. **Don't profile single runs**: GPU performance can vary between runs
3. **Don't ignore memory transfers**: CPU-GPU transfers often dominate
4. **Don't optimize prematurely**: Profile first, then optimize

#### Common pitfalls and solutions

##### Pitfall 1: Cold start effects

```bash
# Wrong: Profile first run
nsys profile mojo your_program.mojo

# Right: Warm up, then profile
nsys profile --delay=5 mojo your_program.mojo  # Let GPU warm up
```

##### Pitfall 2: Wrong build configuration

```bash
# Wrong: Full debug build (disables optimizations) i.e. `--no-optimization`
mojo build -O0 your_program.mojo -o your_program

# Wrong: No debug info (can't map to source)
mojo build your_program.mojo -o your_program

# Right: Optimized build with full debug info for profiling
mojo build --debug-level=full your_program.mojo -o optimized_program
nsys profile ./optimized_program
```

##### Pitfall 3: Ignoring memory transfers

```txt
# Look for this pattern in NSight Systems:
CPU -> GPU transfer: 50ms
Kernel execution: 2ms
GPU -> CPU transfer: 48ms
# Total: 100ms (kernel is only 2%!)
```

**Solution**: Overlap transfers with compute, reduce transfer frequency (covered in Part IX)

##### Pitfall 4: Single kernel focus

```bash
# Wrong: Only profile the "slow" kernel
ncu --kernel-name regex:slow_kernel program

# Right: Profile the whole application first
nsys profile mojo program.mojo  # Find real bottlenecks
```

### Best practices and advanced options

#### Advanced NSight Systems profiling

For comprehensive system-wide analysis, use these advanced `nsys` flags:

```bash
# Production-grade profiling command
nsys profile \
  --gpu-metrics-devices=all \
  --trace=cuda,osrt,nvtx \
  --trace-fork-before-exec=true \
  --cuda-memory-usage=true \
  --cuda-um-cpu-page-faults=true \
  --cuda-um-gpu-page-faults=true \
  --opengl-gpu-workload=false \
  --delay=2 \
  --duration=30 \
  --sample=cpu \
  --cpuctxsw=process-tree \
  --output=comprehensive_profile \
  --force-overwrite=true \
  ./your_program
```

**Flag explanations**:

- `--gpu-metrics-devices=all`: Collect GPU metrics from all devices
- `--trace=cuda,osrt,nvtx`: Comprehensive API tracing
- `--cuda-memory-usage=true`: Track memory allocation/deallocation
- `--cuda-um-cpu/gpu-page-faults=true`: Monitor Unified Memory page faults
- `--delay=2`: Wait 2 seconds before profiling (avoid cold start)
- `--duration=30`: Profile for 30 seconds max
- `--sample=cpu`: Include CPU sampling for hotspot analysis
- `--cpuctxsw=process-tree`: Track CPU context switches

#### Advanced NSight Compute profiling

For detailed kernel analysis with comprehensive metrics:

```bash
# Full kernel analysis with all metric sets
ncu \
  --set full \
  --import-source=on \
  --kernel-id=:::1 \
  --launch-skip=0 \
  --launch-count=1 \
  --target-processes=all \
  --replay-mode=kernel \
  --cache-control=all \
  --clock-control=base \
  --apply-rules=yes \
  --check-exit-code=yes \
  --export=detailed_analysis \
  --force-overwrite \
  ./your_program

# Focus on specific performance aspects
ncu \
  --set=@roofline \
  --section=InstructionStats \
  --section=LaunchStats \
  --section=Occupancy \
  --section=SpeedOfLight \
  --section=WarpStateStats \
  --metrics=sm__cycles_elapsed.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  --kernel-name regex:your_kernel_.* \
  --export=targeted_analysis \
  ./your_program
```

**Key NSight Compute flags**:

- `--set full`: Collect all available metrics (comprehensive but slow)
- `--set @roofline`: Optimized set for roofline analysis
- `--import-source=on`: Map results back to source code
- `--replay-mode=kernel`: Replay kernels for accurate measurements
- `--cache-control=all`: Control GPU caches for consistent results
- `--clock-control=base`: Lock clocks to base frequencies
- `--section=SpeedOfLight`: Include Speed of Light analysis
- `--metrics=...`: Collect specific metrics only
- `--kernel-name regex:pattern`: Target kernels using regex patterns (not `--kernel-regex`)

#### Profiling workflow best practices

##### 1. Progressive profiling strategy

```bash
# Step 1: Quick overview (fast)
nsys profile --trace=cuda --duration=10 --output=quick_look ./program

# Step 2: Detailed system analysis (medium)
nsys profile --trace=cuda,osrt,nvtx --cuda-memory-usage=true --output=detailed ./program

# Step 3: Kernel deep-dive (slow but comprehensive)
ncu --set=@roofline --kernel-name regex:hotspot_kernel ./program
```

##### 2. Multi-run analysis for reliability

```bash
# Profile multiple runs and compare
for i in {1..5}; do
  nsys profile --output=run_${i} ./program
  nsys stats run_${i}.nsys-rep > stats_${i}.txt
done

# Compare results
diff stats_1.txt stats_2.txt
```

##### 3. Targeted kernel profiling

```bash
# First, identify hotspot kernels
nsys profile --trace=cuda,nvtx --output=overview ./program
nsys stats overview.nsys-rep | grep -A 10 "GPU Kernel Summary"

# Then profile specific kernels
ncu --kernel-name="identified_hotspot_kernel" --set full ./program
```

#### Environment and build best practices

##### Optimal build configuration

```bash
# For profiling: optimized with full debug info
mojo build --debug-level=full --optimization-level=3 program.mojo -o program_profile

# Verify build settings
mojo build --help | grep -E "(debug|optimization)"
```

##### Profiling environment setup

```bash
# Disable GPU boost for consistent results
sudo nvidia-smi -ac 1215,1410  # Lock memory and GPU clocks

# Set deterministic behavior
export CUDA_LAUNCH_BLOCKING=1  # Synchronous launches for accurate timing

# Increase driver limits for profiling
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | sudo tee -a /etc/modprobe.d/nvidia-kernel-common.conf
```

##### Memory and performance isolation

```bash
# Clear GPU memory before profiling
nvidia-smi --gpu-reset

# Disable other GPU processes
sudo fuser -v /dev/nvidia*  # Check what's using GPU
sudo pkill -f cuda  # Kill CUDA processes if needed

# Run with high priority
sudo nice -n -20 nsys profile ./program
```

#### Analysis and reporting best practices

##### Comprehensive report generation

```bash
# Generate multiple report formats
nsys stats --report=cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format=csv --output=. profile.nsys-rep

# Export for external analysis
nsys export --type=sqlite profile.nsys-rep
nsys export --type=json profile.nsys-rep

# Generate comparison reports
nsys stats --report=cuda_gpu_kern_sum baseline.nsys-rep > baseline_kernels.txt
nsys stats --report=cuda_gpu_kern_sum optimized.nsys-rep > optimized_kernels.txt
diff -u baseline_kernels.txt optimized_kernels.txt
```

##### Performance regression testing

```bash
#!/bin/bash
# Automated profiling script for CI/CD
BASELINE_TIME=$(nsys stats baseline.nsys-rep | grep "Total Time" | awk '{print $3}')
CURRENT_TIME=$(nsys stats current.nsys-rep | grep "Total Time" | awk '{print $3}')

REGRESSION_THRESHOLD=1.10  # 10% slowdown threshold
if (( $(echo "$CURRENT_TIME > $BASELINE_TIME * $REGRESSION_THRESHOLD" | bc -l) )); then
    echo "Performance regression detected: ${CURRENT_TIME}ns vs ${BASELINE_TIME}ns"
    exit 1
fi
```

Now that you understand profiling fundamentals:

1. **Practice with your existing kernels**: Profile puzzles you've already solved
2. **Prepare for optimization**: Puzzle 31 will use these insights for occupancy optimization
3. **Understand the tools**: Experiment with different NSight Systems and NSight Compute options

**Remember**: Profiling is not just about finding slow code - it's about understanding your program's behavior and
making informed optimization decisions.

For additional profiling resources, see:

- [NVIDIA Profiler User's Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)
- [NSight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [NSight Compute CLI User Guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)

##  The Cache Hit Paradox

### Overview

Welcome to your first **profiling detective case**! You have three GPU kernels that all compute the same simple vector addition: `output[i] = a[i] + b[i]`. They should all perform identically, right?

**Wrong!**These kernels have dramatically different performance - one is **orders of magnitude slower**than the others. Your mission: use the [profiling tools](#nvidia-profiling-basics) you just learned to discover **why**.

### Example scenario

Welcome to a **performance mystery**that will challenge everything you think you know about GPU optimization! You're confronted with three seemingly identical vector addition kernels that compute the exact same mathematical operation:

```
output[i] = a[i] + b[i]  // Simple arithmetic - what could go wrong?
```

**The shocking reality:**

- **All three kernels produce identical, correct results**
- **One kernel runs ~50x slower than the others**
- **The slowest kernel has the highest cache hit rates**(counterintuitive!)
- **Standard performance intuition completely fails**

**Your detective mission:**

1. **Identify the performance culprit**- Which kernel is catastrophically slow?
2. **Uncover the cache paradox**- Why do high cache hits indicate poor performance?
3. **Decode memory access patterns**- What makes identical operations behave so differently?
4. **Learn profiling methodology**- Use NSight tools to gather evidence, not guesses

**Why this matters:**This puzzle reveals a fundamental GPU performance principle that challenges CPU-based intuition. The skills you develop here apply to real-world GPU optimization where memory access patterns often matter more than algorithmic complexity.

**The twist:**We approach this **without looking at the source code first**- using only profiling tools as your guide, just like debugging production performance issues. After we obtained the profiling results, we look at the code for further analysis.

### Your detective toolkit

From the profiling tutorial, you have:

- **NSight Systems (`nsys`)**- Find which kernels are slow
- **NSight Compute (`ncu`)**- Analyze why kernels are slow
- **Memory efficiency metrics**- Detect poor access patterns

### Getting started

#### Step 1: Run the benchmark

```bash
pixi shell -e nvidia
mojo problems/p30/p30.mojo --benchmark
```

You'll see dramatic timing differences between kernels! One kernel is **much slower**than the others. Your job is to figure out why using profiling tools **without**looking at the code.

**Example output:**

```
| name    | met (ms)  | iters |
| ------- | --------- | ----- |
| kernel1 | 171.85    | 11    |
| kernel2 | 1546.68   | 11    |  <- This one is much slower!
| kernel3 | 172.18    | 11    |
```

#### Step 2: Prepare your code for profiling

**Critical**: For accurate profiling, build with full debug information while keeping optimizations enabled:

```bash
mojo build --debug-level=full problems/p30/p30.mojo -o problems/p30/p30_profiler
```

**Why this matters**:

- **Full debug info**: Provides complete symbol tables, variable names, and source line mapping for profilers
- **Comprehensive analysis**: Enables NSight tools to correlate performance data with specific code locations
- **Optimizations enabled**: Ensures realistic performance measurements that match production builds

### Step 3: System-wide investigation (NSight Systems)

Profile each kernel to see the big picture:

```bash
# Profile each kernel individually using the optimized build (with warmup to avoid cold start effects)
nsys profile --trace=cuda,osrt,nvtx --delay=2 --output=./problems/p30/kernel1_profile ./problems/p30/p30_profiler --kernel1
nsys profile --trace=cuda,osrt,nvtx --delay=2 --output=./problems/p30/kernel2_profile ./problems/p30/p30_profiler --kernel2
nsys profile --trace=cuda,osrt,nvtx --delay=2 --output=./problems/p30/kernel3_profile ./problems/p30/p30_profiler --kernel3

# Analyze the results
nsys stats --force-export=true ./problems/p30/kernel1_profile.nsys-rep > ./problems/p30/kernel1_profile.txt
nsys stats --force-export=true ./problems/p30/kernel2_profile.nsys-rep > ./problems/p30/kernel2_profile.txt
nsys stats --force-export=true ./problems/p30/kernel3_profile.nsys-rep > ./problems/p30/kernel3_profile.txt
```

**Look for:**

- **GPU Kernel Summary**- Which kernels take longest?
- **Kernel execution times**- How much do they vary?
- **Memory transfer patterns**- Are they similar across implementations?

### Step 4: Kernel deep-dive (NSight Compute)

Once you identify the slow kernel, analyze it with NSight Compute:

```bash
# Deep-dive into memory patterns for each kernel using the optimized build
ncu --set=@roofline --section=MemoryWorkloadAnalysis -f -o ./problems/p30/kernel1_analysis ./problems/p30/p30_profiler --kernel1
ncu --set=@roofline --section=MemoryWorkloadAnalysis -f -o ./problems/p30/kernel2_analysis ./problems/p30/p30_profiler --kernel2
ncu --set=@roofline --section=MemoryWorkloadAnalysis -f -o ./problems/p30/kernel3_analysis ./problems/p30/p30_profiler --kernel3

# View the results
ncu --import ./problems/p30/kernel1_analysis.ncu-rep --page details
ncu --import ./problems/p30/kernel2_analysis.ncu-rep --page details
ncu --import ./problems/p30/kernel3_analysis.ncu-rep --page details
```

**When you run these commands, you'll see output like this:**

```
Kernel1: Memory Throughput: ~308 Gbyte/s, Max Bandwidth: ~51%
Kernel2: Memory Throughput: ~6 Gbyte/s,   Max Bandwidth: ~12%
Kernel3: Memory Throughput: ~310 Gbyte/s, Max Bandwidth: ~52%
```

**Key metrics to investigate:**

- **Memory Throughput (Gbyte/s)**- Actual memory bandwidth achieved
- **Max Bandwidth (%)**- Percentage of theoretical peak bandwidth utilized
- **L1/TEX Hit Rate (%)**- L1 cache efficiency
- **L2 Hit Rate (%)**- L2 cache efficiency

** The Counterintuitive Result**: You'll notice Kernel2 has the **highest**cache hit rates but the **lowest**performance! This is the key mystery to solve.

### Step 5: Detective questions

Use your profiling evidence to answer these questions by looking at the kernel code problems/p30/p30.mojo:

#### Performance analysis

1. **Which kernel achieves the highest Memory Throughput?**(Look at Gbyte/s values)
2. **Which kernel has the lowest Max Bandwidth utilization?**(Compare percentages)
3. **What's the performance gap in memory throughput?**(Factor difference between fastest and slowest)

#### The cache paradox

4. **Which kernel has the highest L1/TEX Hit Rate?**
5. **Which kernel has the highest L2 Hit Rate?**
6. ** Why does the kernel with the BEST cache hit rates perform the WORST?**

#### Memory access detective work

7. **Can high cache hit rates actually indicate a performance problem?**
8. **What memory access pattern would cause high cache hits but low throughput?**
9. **Why might "efficient caching" be a symptom of "inefficient memory access"?**

#### The "Aha!" Moment

10. **Based on the profiling evidence, what fundamental GPU memory principle does this demonstrate?**

**Key insight to discover**: Sometimes **high cache hit rates are a red flag**, not a performance victory!

### Reference implementation (example)

The mystery reveals a fundamental GPU performance principle: **memory access patterns dominate performance for memory-bound operations**, even when kernels perform identical computations.

**The profiling evidence reveals:**

1. **Performance hierarchy**: Kernel1 and Kernel3 are fast, Kernel2 is catastrophically slow (orders of magnitude difference)
2. **Memory throughput tells the story**: Fast kernels achieve high bandwidth utilization, slow kernel achieves minimal utilization
3. **The cache paradox**: The slowest kernel has the **highest**cache hit rates - revealing that high cache hits can indicate **poor**memory access patterns
4. **Memory access patterns matter more than algorithmic complexity**for memory-bound GPU workloads

#### Complete Solution with Enhanced Explanation

This profiling detective case demonstrates how memory access patterns create orders-of-magnitude performance differences, even when kernels perform identical mathematical operations.

### **Performance evidence from profiling**

**NSight Systems Timeline Analysis:**

- **Kernel 1**: Short execution time - **EFFICIENT**
- **Kernel 3**: Similar to Kernel 1 - **EFFICIENT**
- **Kernel 2**: Dramatically longer execution time - **INEFFICIENT**

**NSight Compute Memory Analysis (Hardware-Agnostic Patterns):**

- **Efficient kernels (1 & 3)**: High memory throughput, good bandwidth utilization, moderate cache hit rates
- **Inefficient kernel (2)**: Very low memory throughput, poor bandwidth utilization, **extremely high cache hit rates**

### **The cache paradox revealed**

** The Counterintuitive Discovery:**

- **Kernel2 has the HIGHEST cache hit rates**but **WORST performance**
- **This challenges conventional wisdom**: "High cache hits = good performance"
- **The truth**: High cache hit rates can be a **symptom of inefficient memory access patterns**

**Why the Cache Paradox Occurs:**

**Traditional CPU intuition (INCORRECT for GPUs):**

- Higher cache hit rates always mean better performance
- Cache hits reduce memory traffic, improving efficiency

**GPU memory reality (CORRECT understanding):**

- **Coalescing matters more than caching**for memory-bound workloads
- **Poor access patterns**can cause artificial cache hit inflation
- **Memory bandwidth utilization**is the real performance indicator

### **Root cause analysis - memory access patterns**

**Actual Kernel Implementations from p30.mojo:**

**Kernel 1 - Efficient Coalesced Access:**

```mojo
fn kernel1
    layout: Layout
:
    i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i < size:
        output[i] = a[i] + b[i]

```

*Standard thread indexing - adjacent threads access adjacent memory*

**Kernel 2 - Inefficient Strided Access:**

```mojo
fn kernel2
    layout: Layout
:
    tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    stride = 512

    i = tid
    while i < size:
        output[i] = a[i] + b[i]
        i += stride

```

*Large stride=512 creates memory access gaps - same operation but scattered access*

**Kernel 3 - Efficient Reverse Access:**

```mojo
fn kernel3
    layout: Layout
:
    tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    total_threads = (SIZE // 1024) * 1024

    for step in range(0, size, total_threads):
        forward_i = step + tid
        if forward_i < size:
            reverse_i = size - 1 - forward_i
            output[reverse_i] = a[reverse_i] + b[reverse_i]

```

*Reverse indexing but still predictable - adjacent threads access adjacent addresses (just backwards)*

**Pattern Analysis:**

- **Kernel 1**: Classic coalesced access - adjacent threads access adjacent memory
- **Kernel 2**: Catastrophic strided access - threads jump by 512 elements
- **Kernel 3**: Reverse but still coalesced within warps - predictable pattern

### **Understanding the memory system**

**GPU Memory Architecture Fundamentals:**

- **Warp execution**: 32 threads execute together
- **Cache line size**: 128 bytes (32 float32 values)
- **Coalescing requirement**: Adjacent threads should access adjacent memory

**p30.mojo Configuration Details:**

```mojo
comptime SIZE = 16 * 1024 * 1024          # 16M elements (64MB of float32 data)
comptime THREADS_PER_BLOCK = (1024, 1)    # 1024 threads per block
comptime BLOCKS_PER_GRID = (SIZE // 1024, 1)  # 16,384 blocks total
comptime dtype = DType.float32             # 4 bytes per element
```

**Why these settings matter:**

- **Large dataset (16M)**: Makes memory access patterns clearly visible
- **1024 threads/block**: Maximum CUDA threads per block
- **32 warps/block**: Each block contains 32 warps of 32 threads each

**Memory Access Efficiency Visualization:**

```
KERNEL 1 (Coalesced):           KERNEL 2 (Strided by 512):
Warp threads 0-31:             Warp threads 0-31:
  Thread 0: Memory[0]            Thread 0: Memory[0]
  Thread 1: Memory[1]            Thread 1: Memory[512]
  Thread 2: Memory[2]            Thread 2: Memory[1024]
  ...                           ...
  Thread 31: Memory[31]          Thread 31: Memory[15872]

Result: 1 cache line fetch       Result: 32 separate cache line fetches
Status: ~308 GB/s throughput     Status: ~6 GB/s throughput
Cache: Efficient utilization     Cache: Same lines hit repeatedly!
```

**KERNEL 3 (Reverse but Coalesced):**

```
Warp threads 0-31 (first iteration):
  Thread 0: Memory[SIZE-1]     (reverse_i = SIZE-1-0)
  Thread 1: Memory[SIZE-2]     (reverse_i = SIZE-1-1)
  Thread 2: Memory[SIZE-3]     (reverse_i = SIZE-1-2)
  ...
  Thread 31: Memory[SIZE-32]   (reverse_i = SIZE-1-31)

Result: Adjacent addresses (just backwards)
Status: ~310 GB/s throughput (nearly identical to Kernel 1)
Cache: Efficient utilization despite reverse order
```

### **The cache paradox explained**

**Why Kernel2 (stride=512) has high cache hit rates but poor performance:**

**The stride=512 disaster explained:**

```mojo
# Each thread processes multiple elements with huge gaps:
Thread 0: elements [0, 512, 1024, 1536, 2048, ...]
Thread 1: elements [1, 513, 1025, 1537, 2049, ...]
Thread 2: elements [2, 514, 1026, 1538, 2050, ...]
...
```

**Why this creates the cache paradox:**

1. **Cache line repetition**: Each 512-element jump stays within overlapping cache line regions
2. **False efficiency illusion**: Same cache lines accessed repeatedly = artificially high "hit rates"
3. **Bandwidth catastrophe**: 32 threads  32 separate cache lines = massive memory traffic
4. **Warp execution mismatch**: GPU designed for coalesced access, but getting scattered access

**Concrete example with float32 (4 bytes each):**

- **Cache line**: 128 bytes = 32 float32 values
- **Stride 512**: Thread jumps by 5124 = 2048 bytes = 16 cache lines apart!
- **Warp impact**: 32 threads need 32 different cache lines instead of 1

**The key insight**: High cache hits in Kernel2 are **repeated access to inefficiently fetched data**, not smart caching!

### **Profiling methodology insights**

**Systematic Detective Approach:**

**Phase 1: NSight Systems (Big Picture)**

- Identify which kernels are slow
- Rule out obvious bottlenecks (memory transfers, API overhead)
- Focus on kernel execution time differences

**Phase 2: NSight Compute (Deep Analysis)**

- Analyze memory throughput metrics
- Compare bandwidth utilization percentages
- Investigate cache hit rates and patterns

**Phase 3: Connect Evidence to Theory**

```
PROFILING EVIDENCE -> CODE ANALYSIS:

NSight Compute Results:           Actual Code Pattern:
- Kernel1: ~308 GB/s            -> i = block_idx*block_dim + thread_idx (coalesced)
- Kernel2: ~6 GB/s, 99% L2 hits -> i += 512 (catastrophic stride)
- Kernel3: ~310 GB/s            -> reverse_i = size-1-forward_i (reverse coalesced)

The profiler data directly reveals the memory access efficiency!
```

**Evidence-to-Code Connection:**

- **High throughput + normal cache rates**= Coalesced access (Kernels 1 & 3)
- **Low throughput + high cache rates**= Inefficient strided access (Kernel 2)
- **Memory bandwidth utilization**reveals true efficiency regardless of cache statistics

### **Real-world performance implications**

**This pattern affects many GPU applications:**

**Scientific Computing:**

- **Stencil computations**: Neighbor access patterns in grid simulations
- **Linear algebra**: Matrix traversal order (row-major vs column-major)
- **PDE solvers**: Grid point access patterns in finite difference methods

**Graphics and Image Processing:**

- **Texture filtering**: Sample access patterns in shaders
- **Image convolution**: Filter kernel memory access
- **Color space conversion**: Channel interleaving strategies

**Machine Learning:**

- **Matrix operations**: Memory layout optimization in GEMM
- **Tensor contractions**: Multi-dimensional array access patterns
- **Data loading**: Batch processing and preprocessing pipelines

### **Fundamental GPU optimization principles**

**Memory-First Optimization Strategy:**

1. **Memory patterns dominate**: Access patterns often matter more than algorithmic complexity
2. **Coalescing is critical**: Design for adjacent threads accessing adjacent memory
3. **Measure bandwidth utilization**: Focus on actual throughput, not just cache statistics
4. **Profile systematically**: Use NSight tools to identify real bottlenecks

**Key Technical Insights:**

- **Memory-bound workloads**: Bandwidth utilization determines performance
- **Cache metrics can mislead**: High hit rates don't always indicate efficiency
- **Warp-level thinking**: Design access patterns for 32-thread execution groups
- **Hardware-aware programming**: Understanding GPU memory hierarchy is essential

### **Key takeaways**

This detective case reveals that **GPU performance optimization requires abandoning CPU intuition**for **memory-centric thinking**:

**Critical insights:**

- High cache hit rates can indicate poor memory access patterns (not good performance)
- Memory bandwidth utilization matters more than cache statistics
- Simple coalesced patterns often outperform complex algorithms
- Profiling tools reveal counterintuitive performance truths

**Practical methodology:**

- Profile systematically with NSight Systems and NSight Compute
- Design for adjacent threads accessing adjacent memory (coalescing)
- Let profiler evidence guide optimization decisions, not intuition

The cache paradox demonstrates that **high-level metrics can mislead without architectural understanding**- applicable far beyond GPU programming.
