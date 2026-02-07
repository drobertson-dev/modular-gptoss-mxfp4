---
title: "Memory Error Detection & Race Conditions with Sanitizers"
description: "You've written what looks like perfect GPU code. Your algorithm is sound, your memory management seems correct, and your thread coordination appears flawless. You run your tests with confidence and..."
---

# Memory Error Detection & Race Conditions with Sanitizers

You've written what looks like perfect GPU code. Your algorithm is sound, your memory management seems correct, and your thread coordination appears flawless. You run your tests with confidence and...

>  This puzzle works on compatible **NVIDIA GPU**only. We are working to enable tooling support for other GPU vendors.

## The moment every GPU developer dreads

You've written what looks like perfect GPU code. Your algorithm is sound, your memory management seems correct, and your thread coordination appears flawless. You run your tests with confidence and...

- ** ALL TESTS PASS**
- ** Performance looks great**
- ** Output matches expected results**

You ship your code to production, feeling proud of your work. Then weeks later, you get the call:

- **"The application crashed in production"**
- **"Results are inconsistent between runs"**
- **"Memory corruption detected"**

Welcome to the insidious world of **silent GPU bugs**- errors that hide in the shadows of massive parallelism, waiting to strike when you least expect them. These bugs can pass all your tests, produce correct results 99% of the time, and then catastrophically fail when it matters most.

**Important note**: This puzzle requires NVIDIA GPU hardware and is only available through `pixi`, as `compute-sanitizer` is part of NVIDIA's CUDA toolkit.

## Why GPU bugs are uniquely sinister

Unlike CPU programs where bugs usually announce themselves with immediate crashes or wrong results, GPU bugs are **experts at hiding**:

**Silent corruption patterns:**

- **Memory violations that don't crash**: Out-of-bounds access to "lucky" memory locations
- **Race conditions that work "most of the time"**: Timing-dependent bugs that appear random
- **Thread coordination failures**: Deadlocks that only trigger under specific load conditions

**Massive scale amplification:**

- **One thread's bug affects thousands**: A single memory violation can corrupt entire warps
- **Race conditions multiply exponentially**: More threads = more opportunities for corruption
- **Hardware variations mask problems**: Same bug behaves differently across GPU architectures

But here's the exciting part: **once you learn GPU sanitization tools, you'll catch these elusive bugs before they ever reach production**.

## Your sanitization toolkit: NVIDIA compute-sanitizer

**NVIDIA compute-sanitizer**is your specialized weapon against GPU bugs. It can detect:

- **Memory violations**: Out-of-bounds access, invalid pointers, memory leaks
- **Race conditions**: Shared memory hazards between threads
- **Synchronization bugs**: Deadlocks, barrier misuse, improper thread coordination
- **And more**: Check `pixi run compute-sanitizer --help`

 **Official documentation**: [NVIDIA Compute Sanitizer User Guide](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)

Think of it as **X-ray vision for your GPU programs**- revealing hidden problems that normal testing can't see.

## What you'll learn in this puzzle

This puzzle teaches you to systematically find and fix the most elusive GPU bugs. You'll learn the detective skills that distinguish competent GPU developers from exceptional ones.

### **Critical skills you'll develop**

1. **Silent bug detection**- Find problems that tests don't catch
2. **Memory corruption investigation**- Track down undefined behavior before it strikes
3. **Race condition detection**- Identify and eliminate concurrency hazards
4. **Tool selection expertise**- Know exactly which sanitizer to use when
5. **Production debugging confidence**- Catch bugs before they reach users

### **Real-world bug hunting scenarios**

You'll investigate the two most dangerous classes of GPU bugs:

- **Memory violations**- The silent killers that corrupt data without warning
- **Race conditions**- The chaos creators that make results unpredictable

Each scenario teaches you to think like a GPU bug detective, following clues that are invisible to normal testing.

## Your bug hunting journey

This puzzle takes you through a carefully designed progression from discovering silent corruption to learning parallel debugging:

###  [The Silent Corruption Mystery](#the-silent-memory-corruption)

**Memory violation investigation**- When tests pass but memory lies

- Investigate programs that pass tests while committing memory crimes
- Learn to spot the telltale signs of undefined behavior (UB)
- Learn `memcheck` - your memory violation detector
- Understand why GPU hardware masks memory errors
- Practice systematic memory access validation

**Key outcome**: Ability to detect memory violations that would otherwise go unnoticed until production

###  [The Race Condition Hunt](#debugging-race-conditions)

**Concurrency bug investigation**- When threads turn against each other

- Investigate programs that fail randomly due to thread timing
- Learn to identify shared memory hazards before they corrupt data
- Learn `racecheck` - your race condition detector
- Compare `racecheck` vs `synccheck` for different concurrency bugs
- Practice thread synchronization strategies

**Key outcome**: Advanced concurrency debugging - the ability to tame thousands of parallel threads

## The GPU detective mindset

GPU sanitization requires you to become a **parallel program detective**investigating crimes where:

- **The evidence is hidden**- Bugs occur in parallel execution you can't directly observe
- **Multiple suspects exist**- Thousands of threads, any combination could be guilty
- **The crime is intermittent**- Race conditions and timing-dependent failures
- **The tools are specialized**- Sanitizers that see what normal debugging can't

But like any good detective, you'll learn to:

- **Follow invisible clues**- Memory access patterns, thread timing, synchronization points
- **Think in parallel**- Consider how thousands of threads interact simultaneously
- **Prevent future crimes**- Build sanitization into your development workflow
- **Trust your tools**- Let sanitizers reveal what manual testing cannot

## Prerequisites and expectations

**What you need to know**:

- GPU programming concepts from Puzzles 1-8 (memory management, thread coordination, barriers)
- **[Compatible NVIDIA GPU hardware](https://docs.modular.com/max/faq#gpu-requirements)**
- Environment setup with `pixi` package manager for accessing `compute-sanitizer`
- **Prior puzzles**: Familiarity with Puzzle 4 and Puzzle 8 are recommended

**What you'll gain**:

- **Production-ready debugging skills**used by professional GPU development teams
- **Silent bug detection skills**that prevent costly production failures
- **Parallel debugging confidence**for the most challenging concurrency scenarios
- **Tool expertise**that will serve you throughout your GPU programming career

##  The Silent Memory Corruption

### Overview

Learn how to detect memory violations that can silently corrupt GPU programs, even when tests appear to pass. Using NVIDIA's `compute-sanitizer` (available through `pixi`) with the `memcheck` tool, you'll discover hidden memory bugs that could cause unpredictable behavior in your GPU code.

**Key insight**: A GPU program can produce "correct" results while simultaneously performing illegal memory accesses.

**Prerequisites**: Understanding of Puzzle 4 LayoutTensor and basic GPU memory concepts.

### The silent memory bug discovery

#### Test passes, but is my code actually correct?

Let's start with a seemingly innocent program that appears to work perfectly (this is Puzzle 04 without guards):

```mojo
fn add_10_2d(
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    a: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    size: UInt,
):
    row = thread_idx.y
    col = thread_idx.x
    output[row, col] = a[row, col] + 10.0

```

When you run this program normally, everything looks fine:

```bash
pixi run p10 --memory-bug
```

```txt
out shape: 2 x 2
Running memory bug example (bounds checking issue)...
out: HostBuffer([10.0, 11.0, 12.0, 13.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
OK Memory test PASSED! (memcheck may find bounds violations)
```

 **Test PASSED!**The output matches expected results perfectly. Case closed, right?

**Wrong!**Let's see what `compute-sanitizer` reveals:

```bash
MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=0 pixi run compute-sanitizer --tool memcheck mojo problems/p10/p10.mojo --memory-bug
```

**Note**: `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=0` is a command-line environment variable setting that disables a device context's buffer cache. This setting can reveal memory issues, like bounds violations, that are otherwise masked by the normal caching behavior.

```txt
========= COMPUTE-SANITIZER
out shape: 2 x 2
Running memory bug example (bounds checking issue)...

========= Invalid __global__ read of size 4 bytes
=========     at p10_add_10_2d_...+0x80
=========     by thread (2,1,0) in block (0,0,0)
=========     Access at 0xe0c000210 is out of bounds
=========     and is 513 bytes after the nearest allocation at 0xe0c000000 of size 16 bytes

========= Invalid __global__ read of size 4 bytes
=========     at p10_add_10_2d_...+0x80
=========     by thread (0,2,0) in block (0,0,0)
=========     Access at 0xe0c000210 is out of bounds
=========     and is 513 bytes after the nearest allocation at 0xe0c000000 of size 16 bytes

========= Invalid __global__ read of size 4 bytes
=========     at p10_add_10_2d_...+0x80
=========     by thread (1,2,0) in block (0,0,0)
=========     Access at 0xe0c000214 is out of bounds
=========     and is 517 bytes after the nearest allocation at 0xe0c000000 of size 16 bytes

========= Invalid __global__ read of size 4 bytes
=========     at p10_add_10_2d_...+0x80
=========     by thread (2,2,0) in block (0,0,0)
=========     Access at 0xe0c000218 is out of bounds
=========     and is 521 bytes after the nearest allocation at 0xe0c000000 of size 16 bytes

========= Program hit CUDA_ERROR_LAUNCH_FAILED (error 719) due to "unspecified launch failure" on CUDA API call to cuStreamSynchronize.
========= Program hit CUDA_ERROR_LAUNCH_FAILED (error 719) due to "unspecified launch failure" on CUDA API call to cuEventCreate.
========= Program hit CUDA_ERROR_LAUNCH_FAILED (error 719) due to "unspecified launch failure" on CUDA API call to cuMemFreeAsync.

========= ERROR SUMMARY: 7 errors
```

The program has **7 total errors**despite passing all tests:
- **4 memory violations**(Invalid __global__ read)
- **3 runtime errors**(caused by the memory violations)

### Understanding the hidden bug

#### Root cause analysis

**The Problem:**
- **Tensor size**: 22 (valid indices: 0, 1)
- **Thread grid**: 33 (thread indices: 0, 1, 2)
- **Out-of-bounds threads**: `(2,1)`, `(0,2)`, `(1,2)`, `(2,2)` access invalid memory
- **Missing bounds check**: No validation of `thread_idx` against tensor dimensions

#### Understanding the 7 total errors

**4 Memory Violations:**
- Each out-of-bounds thread `(2,1)`, `(0,2)`, `(1,2)`, `(2,2)` caused an "Invalid __global__ read"

**3 CUDA Runtime Errors:**
- `cuStreamSynchronize` failed due to kernel launch failure
- `cuEventCreate` failed during cleanup
- `cuMemFreeAsync` failed during memory deallocation

**Key Insight**: Memory violations have cascading effects - one bad memory access causes multiple downstream CUDA API failures.

**Why tests still passed:**
- Valid threads `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)` wrote correct results
- Test only checked valid output locations
- Out-of-bounds accesses didn't immediately crash the program

### Understanding undefined behavior (UB)

#### What is undefined behavior?

**Undefined Behavior (UB)**occurs when a program performs operations that have no defined meaning according to the language specification. Out-of-bounds memory access is a classic example of undefined behavior.

**Key characteristics of UB:**
- The program can do **literally anything**: crash, produce wrong results, appear to work, or corrupt memory
- **No guarantees**: Behavior may change between compilers, hardware, drivers, or even different runs

#### Why undefined behavior is especially dangerous

**Correctness issues:**
- **Unpredictable results**: Your program may work during testing but fail in production
- **Non-deterministic behavior**: Same code can produce different results on different runs
- **Silent corruption**: UB can corrupt data without any visible errors
- **Compiler optimizations**: Compilers assume no UB occurs and may optimize in unexpected ways

**Security vulnerabilities:**
- **Buffer overflows**: Classic source of security exploits in systems programming
- **Memory corruption**: Can lead to privilege escalation and code injection attacks
- **Information leakage**: Out-of-bounds reads can expose sensitive data
- **Control flow hijacking**: UB can be exploited to redirect program execution

#### GPU-specific undefined behavior dangers

**Massive scale impact:**
- **Thread divergence**: One thread's UB can affect entire warps (32 threads)
- **Memory coalescing**: Out-of-bounds access can corrupt neighboring threads' data
- **Kernel failures**: UB can cause entire GPU kernels to fail catastrophically

**Hardware variations:**
- **Different GPU architectures**: UB may manifest differently on different GPU models
- **Driver differences**: Same UB may behave differently across driver versions
- **Memory layout changes**: GPU memory allocation patterns can change UB manifestation

### Fixing the memory violation

#### The solution

As we saw in Puzzle 04, we need to bound-check as follows:

```mojo
fn add_10_2d(
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    a: LayoutTensor[dtype, layout, MutAnyOrigin],
    size: UInt,
):
    row = thread_idx.y
    col = thread_idx.x
    if col < size and row < size:
        output[row, col] = a[row, col] + 10.0

```

The fix is simple: **always validate thread indices against data dimensions**before accessing memory.

#### Verification with compute-sanitizer

```bash
# Fix the bounds checking in your copy of p10.mojo, then run:
MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=0 pixi run compute-sanitizer --tool memcheck mojo problems/p10/p10.mojo --memory-bug
```

```txt
========= COMPUTE-SANITIZER
out shape: 2 x 2
Running memory bug example (bounds checking issue)...
out: HostBuffer([10.0, 11.0, 12.0, 13.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
OK Memory test PASSED! (memcheck may find bounds violations)
========= ERROR SUMMARY: 0 errors
```

** SUCCESS:**No memory violations detected!

### Key learning points

#### Why manual bounds checking matters

1. **Clarity**: Makes the safety requirements explicit in the code
2. **Control**: You decide exactly what happens for out-of-bounds cases
3. **Debugging**: Easier to reason about when memory violations occur

#### GPU memory safety rules

1. **Always validate thread indices**against data dimensions
2. **Avoid undefined behavior (UB) at all costs**- out-of-bounds access is UB and can break everything
3. **Use compute-sanitizer**during development and testing
4. **Never assume "it works" without memory checking**
5. **Test with different grid/block configurations**to catch undefined behavior (UB) that manifests inconsistently

#### Compute-sanitizer best practices

```bash
MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=0 pixi run compute-sanitizer --tool memcheck mojo your_code.mojo
```

**Note**: You may see Mojo runtime warnings in the sanitizer output. Focus on the `========= Invalid` and `========= ERROR SUMMARY` lines for actual memory violations.

##  Debugging Race Conditions

### Overview

Debug failing GPU programs using NVIDIA's `compute-sanitizer` to identify race conditions that cause incorrect results. You'll learn to use the `racecheck` tool to find concurrency bugs in shared memory operations.

You have a GPU kernel that should accumulate values from multiple threads using shared memory. The test fails, but the logic seems correct. Example goal:  identify and fix the race condition causing the failure.

### Configuration

```mojo
comptime SIZE = 2
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = (3, 3)  # 9 threads, but only 4 are active
comptime dtype = DType.float32
```

### The failing kernel

```mojo

comptime SIZE = 2
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = (3, 3)
comptime dtype = DType.float32
comptime layout = Layout.row_major(SIZE, SIZE)

fn shared_memory_race(
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    a: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    size: UInt,
):
    row = thread_idx.y
    col = thread_idx.x

    shared_sum = LayoutTensor[
        dtype,
        Layout.row_major(1),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    if row < size and col < size:
        shared_sum[0] += a[row, col]

    barrier()

    if row < size and col < size:
        output[row, col] = shared_sum[0]

```

### Running the code

```bash
pixi run p10 --race-condition
```

and the output will look like

```txt
out shape: 2 x 2
Running race condition example...
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([6.0, 6.0, 6.0, 6.0])
stack trace was not collected. Enable stack trace collection with environment variable `MOJO_ENABLE_STACK_TRACE_ON_ERROR`
Unhandled exception caught during execution: At /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p10/p10.mojo:122:33: AssertionError: `left == right` comparison failed:
   left: 0.0
  right: 6.0
```

Let's see how `compute-sanitizer` can help us detection issues in our GPU code.

### Debugging with `compute-sanitizer`

#### Step 1: Identify the race condition with `racecheck`

Use `compute-sanitizer` with the `racecheck` tool to identify race conditions:

```bash
pixi run compute-sanitizer --tool racecheck mojo problems/p10/p10.mojo --race-condition
```

the output will look like

```txt
========= COMPUTE-SANITIZER
out shape: 2 x 2
Running race condition example...
========= Error: Race reported between Write access at p10_shared_memory_race_...+0x140
=========     and Read access at p10_shared_memory_race_...+0xe0 [4 hazards]
=========     and Write access at p10_shared_memory_race_...+0x140 [5 hazards]
=========
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([6.0, 6.0, 6.0, 6.0])
AssertionError: `left == right` comparison failed:
  left: 0.0
  right: 6.0
========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)
```

**Analysis**: The program has **1 race condition**with **9 individual hazards**:
- **4 read-after-write hazards**(threads reading while others write)
- **5 write-after-write hazards**(multiple threads writing simultaneously)

#### Step 2: Compare with `synccheck`

Verify this is a race condition, not a synchronization issue:

```bash
pixi run compute-sanitizer --tool synccheck mojo problems/p10/p10.mojo --race-condition
```

and the output will be like

```txt
========= COMPUTE-SANITIZER
out shape: 2 x 2
Running race condition example...
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([6.0, 6.0, 6.0, 6.0])
AssertionError: `left == right` comparison failed:
  left: 0.0
  right: 6.0
========= ERROR SUMMARY: 0 errors
```

**Key insight**: `synccheck` found **0 errors**- there are no synchronization issues like deadlocks. The problem is **race conditions**, not synchronization bugs.

### Deadlock vs Race Condition: Understanding the Difference

| Aspect | Deadlock | Race Condition |
|--------|----------|----------------|
| **Symptom**| Program hangs forever | Program produces wrong results |
| **Execution**| Never completes | Completes successfully |
| **Timing**| Deterministic hang | Non-deterministic results |
| **Root Cause**| Synchronization logic error | Unsynchronized data access |
| **Detection Tool**| `synccheck` | `racecheck` |
| **Example**| Puzzle 09: Third case barrier deadlock | Our shared memory `+=` operation |

**In our specific case:**
- **Program completes** No deadlock (threads don't get stuck)
- **Wrong results** Race condition (threads corrupt each other's data)
- **Tool confirms** `synccheck` reports 0 errors, `racecheck` reports 9 hazards

**Why this distinction matters for debugging:**
- **Deadlock debugging**: Focus on barrier placement, conditional synchronization, thread coordination
- **Race condition debugging**: Focus on shared memory access patterns, atomic operations, data dependencies

### Investigation prompt

Equiped with these tools, fix the kernel failing kernel.

#### Tips

#### Understanding the hazard breakdown

The `shared_sum[0] += a[row, col]` operation creates hazards because it's actually **three separate memory operations**:
1. **READ**`shared_sum[0]`
2. **ADD**`a[row, col]` to the read value
3. **WRITE**the result back to `shared_sum[0]`

With 4 active threads (positions (0,0), (0,1), (1,0), (1,1)), these operations can interleave:
- **Thread timing overlap** Multiple threads read the same initial value (0.0)
- **Lost updates** Each thread writes back `0.0 + their_value`, overwriting others' work
- **Non-atomic operation** The `+=` compound assignment isn't atomic in GPU shared memory

**Why we get exactly 9 hazards:**
- Each thread tries to perform read-modify-write
- 4 threads  2-3 hazards per thread = 9 total hazards
- `compute-sanitizer` tracks every conflicting memory access pair

#### Race condition debugging tips

1. **Use racecheck for data races**: Detects shared memory hazards and data corruption
2. **Use synccheck for deadlocks**: Detects synchronization bugs (barrier issues, deadlocks)
3. **Focus on shared memory access**: Look for unsynchronized `+=`, `=` operations to shared variables
4. **Identify the pattern**: Read-modify-write operations are common race condition sources
5. **Check barrier placement**: Barriers must be placed BEFORE conflicting operations, not after

**Why this distinction matters for debugging:**
- **Deadlock debugging**: Focus on barrier placement, conditional synchronization, thread coordination
- **Race condition debugging**: Focus on shared memory access patterns, atomic operations, data dependencies

**Common race condition patterns to avoid:**
- Multiple threads writing to the same shared memory location
- Unsynchronized read-modify-write operations (`+=`, `++`, etc.)
- Barriers placed after the race condition instead of before

### Reference implementation (example)


```mojo

comptime SIZE = 2
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = (3, 3)
comptime dtype = DType.float32
comptime layout = Layout.row_major(SIZE, SIZE)

fn shared_memory_race(
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    a: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    size: UInt,
):
    """Fixed: sequential access with barriers eliminates race conditions."""
    row = thread_idx.y
    col = thread_idx.x

    shared_sum = LayoutTensor[
        dtype,
        Layout.row_major(1),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Only thread 0 does all the accumulation work to prevent races
    if row == 0 and col == 0:
        # Use local accumulation first, then single write to shared memory
        local_sum = Scalardtype
        for r in range(size):
            for c in range(size):
                local_sum += rebind[Scalar[dtype]](a[r, c])

        shared_sum[0] = local_sum  # Single write operation

    barrier()  # Ensure thread 0 completes before others read

    # All threads read the safely accumulated result after synchronization
    if row < size and col < size:
        output[row, col] = shared_sum[0]

```

#### Understanding what went wrong

##### The race condition problem pattern

The original failing code had this critical line:

```mojo
shared_sum[0] += a[row, col]  # RACE CONDITION!
```

This single line creates multiple hazards among the 4 valid threads:
1. **Thread (0,0) reads**`shared_sum[0]` (value: 0.0)
2. **Thread (0,1) reads**`shared_sum[0]` (value: 0.0)  **Read-after-write hazard!**
3. **Thread (0,0) writes**back `0.0 + 0`
4. **Thread (1,0) writes**back `0.0 + 2`  **Write-after-write hazard!**

##### Why the test failed

- Multiple threads corrupt each other's writes during the `+=` operation
- The `+=` operation gets interrupted, causing lost updates
- Expected sum of 6.0 (0+1+2+3), but race conditions resulted in 0.0
- The `barrier()` comes too late - after the race condition already occurred

##### What are race conditions?

**Race conditions**occur when multiple threads access shared data concurrently, and the result depends on the unpredictable timing of thread execution.

**Key characteristics:**
- **Non-deterministic behavior**: Same code can produce different results on different runs
- **Timing-dependent**: Results depend on which thread "wins the race"
- **Hard to reproduce**: May only manifest under specific conditions or hardware

##### GPU-specific dangers

**Massive parallelism impact:**
- **Warp-level corruption**: Race conditions can affect entire warps (32 threads)
- **Memory coalescing issues**: Races can disrupt efficient memory access patterns
- **Kernel-wide failures**: Shared memory corruption can affect the entire GPU kernel

**Hardware variations:**
- **Different GPU architectures**: Race conditions may manifest differently across GPU models
- **Memory hierarchy**: L1 cache, L2 cache, and global memory can all exhibit different race behaviors
- **Warp scheduling**: Different thread scheduling can expose different race condition scenarios

#### Strategy: Single writer pattern

The key insight is to eliminate concurrent writes to shared memory:

1. **Single writer**: Only one thread (thread at position (0,0)) does all accumulation work
2. **Local accumulation**: Thread at position (0,0) uses a local variable to avoid repeated shared memory access
3. **Single shared memory write**: One write operation eliminates write-write races
4. **Barrier synchronization**: Ensures writer completes before others read
5. **Multiple readers**: All threads safely read the final result

##### Step-by-step solution breakdown

**Step 1: Thread identification**
```mojo
if row == 0 and col == 0:
```
Use direct coordinate check to identify thread at position (0,0).

**Step 2: Single-threaded accumulation**
```mojo
if row == 0 and col == 0:
    local_sum = Scalardtype
    for r in range(size):
        for c in range(size):
            local_sum += rebind[Scalar[dtype]](a[r, c])
    shared_sum[0] = local_sum  # Single write operation
```
Only thread at position (0,0) performs all accumulation work, eliminating race conditions.

**Step 3: Synchronization barrier**
```mojo
barrier()  # Ensure thread (0,0) completes before others read
```
All threads wait for thread at position (0,0) to finish accumulation.

**Step 4: Safe parallel reads**
```mojo
if row < size and col < size:
    output[row, col] = shared_sum[0]
```
All threads can safely read the result after synchronization.

#### Important note on efficiency

**This solution prioritizes correctness over efficiency**. While it eliminates race conditions, using only thread at position (0,0) for accumulation is **not optimal**for GPU performance - we're essentially doing serial computation on a massively parallel device.

#### Verification

```bash
pixi run compute-sanitizer --tool racecheck mojo solutions/p10/p10.mojo --race-condition
```

**Expected output:**
```txt
========= COMPUTE-SANITIZER
out shape: 2 x 2
Running race condition example...
out: HostBuffer([6.0, 6.0, 6.0, 6.0])
expected: HostBuffer([6.0, 6.0, 6.0, 6.0])
OK Race condition test PASSED! (racecheck will find hazards)
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

** SUCCESS:**Test passes and no race conditions detected!
