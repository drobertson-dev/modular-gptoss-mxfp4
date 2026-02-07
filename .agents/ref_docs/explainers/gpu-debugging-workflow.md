---
title: "GPU Debugging Workflow"
description: "You've written GPU kernels, worked with shared memory, and coordinated thousands of parallel threads. Your code compiles. You run it expecting correct results, and then:"
---

# GPU Debugging Workflow

You've written GPU kernels, worked with shared memory, and coordinated thousands of parallel threads. Your code compiles. You run it expecting correct results, and then:

>  This puzzle works on compatible **NVIDIA GPU**only. We are working to enable tooling support for other GPU vendors.

## When GPU programs fail

You've written GPU kernels, worked with shared memory, and coordinated thousands of parallel threads. Your code compiles. You run it expecting correct results, and then:

- **CRASH**
- **Wrong results**
- **Infinite hang**

This is GPU programming reality: **debugging parallel code running on thousands of threads simultaneously**. This is where theory meets practice, where algorithmic knowledge meets investigative skills.

## Why GPU debugging is challenging

Unlike traditional CPU debugging where you follow a single thread through sequential execution, GPU debugging requires you to:

- **Think in parallel**: Thousands of threads executing simultaneously, each potentially doing something different
- **Navigate multiple memory spaces**: Global memory, shared memory, registers, constant memory
- **Handle coordination failures**: Race conditions, barrier deadlocks, memory access violations
- **Debug optimized code**: JIT compilation, variable optimization, limited symbol information
- **Use specialized tools**: CUDA-GDB for kernel inspection, thread navigation, parallel state analysis

**GPU debugging skills provide deep understanding of parallel computing fundamentals**.

## What you'll learn in this puzzle

This puzzle teaches you to debug GPU code systematically. You'll learn the approaches, tools, and techniques that GPU developers use daily to solve complex parallel programming challenges.

### **Essential skills you'll develop**

1. **Professional debugging workflow**- The systematic approach professionals use
2. **Tool proficiency**- LLDB for host code, CUDA-GDB for GPU kernels
3. **Pattern recognition**- Common GPU bug types and symptoms
4. **Investigation techniques**- Finding root causes when variables are optimized out
5. **Thread coordination debugging**- Advanced GPU debugging skills

### **Real-world debugging scenarios**

You'll tackle the three most common GPU programming failures:

- **Memory crashes**- Null pointers, illegal memory access, segmentation faults
- **Logic bugs**- Correct execution with wrong results, algorithmic errors
- **Coordination deadlocks**- Barrier synchronization failures, infinite hangs

Each scenario teaches different investigation techniques and builds debugging intuition.

## Your debugging journey

This puzzle takes you through a carefully designed progression from basic debugging concepts to advanced parallel coordination failures:

###  **Step 1: [Mojo GPU Debugging Essentials](#mojo-gpu-debugging-essentials)**

**Foundation building**- Learn the tools and workflow

- Set up your debugging environment with `pixi` and CUDA-GDB
- Learn the four debugging approaches: JIT vs binary, CPU vs GPU
- Learn essential CUDA-GDB commands for GPU kernel inspection
- Practice with hands-on examples using familiar code from previous puzzles
- Understand when to use each debugging approach

**Key outcome**: Professional debugging workflow and tool proficiency

###  **Step 2: [Detective Work: First Case](#detective-work-first-case)**

**Memory crash investigation**- Debug a GPU program that crashes

- Investigate `CUDA_ERROR_ILLEGAL_ADDRESS` crashes
- Learn systematic pointer inspection techniques
- Learn null pointer detection and validation
- Practice professional crash analysis workflow
- Understand GPU memory access failures

**Key outcome**: Ability to debug GPU memory crashes and pointer issues

###  **Step 3: [Detective Work: Second Case](#detective-work-second-case)**

**Logic bug investigation**- Debug a program with wrong results

- Investigate LayoutTensor-based algorithmic errors
- Learn execution flow analysis when variables are optimized out
- Learn loop boundary analysis and iteration counting
- Practice pattern recognition in incorrect results
- Debug without direct variable inspection

**Key outcome**: Ability to debug algorithmic errors and logic bugs in GPU kernels

###  **Step 4: [Detective Work: Third Case](#detective-work-third-case)**

**Barrier deadlock investigation**- Debug a program that hangs forever

- Investigate barrier synchronization failures
- Learn multi-thread state analysis across parallel execution
- Learn conditional execution path tracing
- Practice thread coordination debugging
- Understand the most challenging GPU debugging scenario

**Key outcome**: Advanced thread coordination debugging - the pinnacle of GPU debugging skills

## The detective mindset

GPU debugging requires a different mindset than traditional programming. You become a **detective**investigating a crime scene where:

- **The evidence is limited**- Variables are optimized out, symbols are mangled
- **Multiple suspects exist**- Thousands of threads, any could be the culprit
- **The timeline is complex**- Parallel execution, race conditions, timing dependencies
- **The tools are specialized**- CUDA-GDB, thread navigation, GPU memory inspection

But like any good detective, you'll learn to:

- **Follow the clues systematically**- Error messages, crash patterns, thread states
- **Form hypotheses**- What could cause this specific behavior?
- **Test theories**- Use debugging commands to verify or disprove ideas
- **Trace back to root causes**- From symptoms to the actual source of problems

## Prerequisites and expectations

**What you need to know**:

- GPU programming concepts from Puzzles 1-8 (thread indexing, memory management, barriers)
- Basic command-line comfort (you'll use terminal-based debugging tools)
- Patience and systematic thinking (GPU debugging requires methodical investigation)

**What you'll gain**:

- **Professional debugging skills**used in GPU development teams
- **Deep parallel computing understanding**that comes from seeing execution at the thread level
- **Problem-solving confidence**for the most challenging GPU programming scenarios
- **Tool proficiency**that will serve you throughout your GPU programming career

## Ready to begin?

GPU debugging is where you transition from *writing* GPU programs to *understanding* them deeply. Every professional GPU developer has spent countless hours debugging parallel code, learning to think in thousands of simultaneous threads, and developing the patience to investigate complex coordination failures.

This is your opportunity to join that elite group.

**Start your debugging journey**: [Mojo GPU Debugging Essentials](#mojo-gpu-debugging-essentials)

---

*"Debugging is twice as hard as writing the code in the first place. Therefore, if you write the code as cleverly as possible, you are, by definition, not smart enough to debug it."* - Brian Kernighan

*In GPU programming, this wisdom is amplified by a factor of thousands - the number of parallel threads you're debugging simultaneously.*

##  Mojo GPU Debugging Essentials

Welcome to the world of GPU debugging! After learning GPU programming concepts through puzzles 1-8, you're now ready to learn the most critical skill for any GPU programmer: **how to debug when things go wrong**.

GPU debugging can seem intimidating at first - you're dealing with thousands of threads running in parallel, different memory spaces, and hardware-specific behaviors. But with the right tools and workflow, debugging GPU code becomes systematic and manageable.

In this guide, you'll learn to debug both the **CPU host code**(where you set up your GPU operations) and the **GPU kernel code**(where the parallel computation happens). We'll use real examples, actual debugger output, and step-by-step workflows that you can immediately apply to your own projects.

**Note**: The following content focuses on command-line debugging for universal IDE compatibility. If you prefer VS Code debugging, refer to the [Mojo debugging documentation](https://docs.modular.com/mojo/tools/debugging) for VS Code-specific setup and workflows.

### Why GPU debugging is different

Before diving into tools, consider what makes GPU debugging unique:

- **Traditional CPU debugging**: One thread, sequential execution, straightforward memory model
- **GPU debugging**: Thousands of threads, parallel execution, multiple memory spaces, race conditions

This means you need specialized tools that can:

- Switch between different GPU threads
- Inspect thread-specific variables and memory
- Handle the complexity of parallel execution
- Debug both CPU setup code and GPU kernel code

### Your debugging toolkit

Mojo's GPU debugging capabilities currently is limited to NVIDIA GPUs. The [Mojo debugging documentation](https://docs.modular.com/mojo/tools/debugging) explains that the Mojo package includes:

- **LLDB debugger**with Mojo plugin for CPU-side debugging
- **CUDA-GDB integration**for GPU kernel debugging
- **Command-line interface**via `mojo debug` for universal IDE compatibility

For GPU-specific debugging, the [Mojo GPU debugging guide](https://docs.modular.com/mojo/tools/gpu-debugging) provides additional technical details.

This architecture provides the best of both worlds: familiar debugging commands with GPU-specific capabilities.

### The debugging workflow: From problem to solution

When your GPU program crashes, produces wrong results, or behaves unexpectedly, follow this systematic approach:

1. **Prepare your code for debugging**(disable optimizations, add debug symbols)
2. **Choose the right debugger**(CPU host code vs GPU kernel debugging)
3. **Set strategic breakpoints**(where you suspect the problem lies)
4. **Execute and inspect**(step through code, examine variables)
5. **Analyze patterns**(memory access, thread behavior, race conditions)

This workflow works whether you're debugging a simple array operation from Puzzle 01 or complex shared memory code from Puzzle 08.

### Step 1: Preparing your code for debugging

** The golden rule**: Never debug _optimized_ code. Optimizations can reorder instructions, eliminate variables, and inline functions, making debugging nearly impossible.

#### Building with debug information

When building Mojo programs for debugging, always include debug symbols:

```bash
# Build with full debug information
mojo build -O0 -g your_program.mojo -o your_program_debug
```

**What these flags do:**

- `-O0`: Disables all optimizations, preserving your original code structure
- `-g`: Includes debug symbols so the debugger can map machine code back to your Mojo source
- `-o`: Creates a named output file for easier identification

#### Why this matters

Without debug symbols, your debugging session looks like this:

```
(lldb) print my_variable
error: use of undeclared identifier 'my_variable'
```

With debug symbols, you get:

```
(lldb) print my_variable
(int) $0 = 42
```

### Step 2: Choosing your debugging approach

Here's where GPU debugging gets interesting. You have **four different combinations**to choose from, and picking the right one saves you time:

#### The four debugging combinations

**Quick reference:**

```bash
# 1. JIT + LLDB: Debug CPU host code directly from source
pixi run mojo debug your_gpu_program.mojo

# 2. JIT + CUDA-GDB: Debug GPU kernels directly from source
pixi run mojo debug --cuda-gdb --break-on-launch your_gpu_program.mojo

# 3. Binary + LLDB: Debug CPU host code from pre-compiled binary
pixi run mojo build -O0 -g your_gpu_program.mojo -o your_program_debug
pixi run mojo debug your_program_debug

# 4. Binary + CUDA-GDB: Debug GPU kernels from pre-compiled binary
pixi run mojo debug --cuda-gdb --break-on-launch your_program_debug
```

#### When to use each approach

**For learning and quick experiments:**

- Use **JIT debugging**- no build step required, faster iteration

**For serious debugging sessions:**

- Use **binary debugging**- more predictable, cleaner debugger output

**For CPU-side issues**(buffer allocation, host memory, program logic):

- Use **LLDB mode**- perfect for debugging your `main()` function and setup code

**For GPU kernel issues**(thread behavior, GPU memory, kernel crashes):

- Use **CUDA-GDB mode**- the only way to inspect individual GPU threads

The beauty is that you can mix and match. Start with JIT + LLDB to debug your setup code, then switch to JIT + CUDA-GDB to debug the actual kernel.

---

### Understanding GPU kernel debugging with CUDA-GDB

Next comes GPU kernel debugging - the most powerful (and complex) part of your debugging toolkit.

When you use `--cuda-gdb`, Mojo integrates with NVIDIA's [CUDA-GDB debugger](https://docs.nvidia.com/cuda/cuda-gdb/index.html). This isn't just another debugger - it's specifically designed for the parallel, multi-threaded world of GPU computing.

#### What makes CUDA-GDB special

**Regular GDB**debugs one thread at a time, stepping through sequential code.
**CUDA-GDB**debugs thousands of GPU threads simultaneously, each potentially executing different instructions.

This means you can:

- **Set breakpoints inside GPU kernels**- pause execution when any thread hits your breakpoint
- **Switch between GPU threads**- examine what different threads are doing at the same moment
- **Inspect thread-specific data**- see how the same variable has different values across threads
- **Debug memory access patterns**- catch out-of-bounds access, race conditions, and memory corruption (more on detecting such issues in the Puzzle 10)
- **Analyze parallel execution**- understand how your threads interact and synchronize

#### Connecting to concepts from previous puzzles

Remember the GPU programming concepts you learned in puzzles 1-8? CUDA-GDB lets you inspect all of them at runtime:

##### Thread hierarchy debugging

Back in puzzles 1-8, you wrote code like this:

```mojo
# From puzzle 1: Basic thread indexing
i = thread_idx.x  # Each thread gets a unique index

# From puzzle 7: 2D thread indexing
row = thread_idx.y  # 2D grid of threads
col = thread_idx.x
```

With CUDA-GDB, you can **actually see these thread coordinates in action**:

```gdb
(cuda-gdb) info cuda threads
```

outputs

```
  BlockIdx ThreadIdx To BlockIdx To ThreadIdx Count                 PC                                                       Filename  Line
Kernel 0
*  (0,0,0)   (0,0,0)     (0,0,0)      (3,0,0)     4 0x00007fffcf26fed0 /home/ubuntu/workspace/mojo-gpu-puzzles/solutions/p01/p01.mojo    13
```

and jump to a specific thread to see what it's doing

```gdb
(cuda-gdb) cuda thread (1,0,0)
```

shows

```
[Switching to CUDA thread (1,0,0)]
```

This is incredibly powerful - you can literally **watch your parallel algorithm execute across different threads**.

##### Memory space debugging

Remember puzzle 8 where you learned about different types of GPU memory? CUDA-GDB lets you inspect all of them:

```gdb
# Examine global memory (the arrays from puzzles 1-5)
(cuda-gdb) print input_array[0]@4
$1 = {{1}, {2}, {3}, {4}}   # Mojo scalar format

# Examine shared memory using local variables (thread_idx.x doesn't work)
(cuda-gdb) print shared_data[i]   # Use local variable 'i' instead
$2 = {42}
```

The debugger shows you exactly what each thread sees in memory - perfect for catching race conditions or memory access bugs.

##### Strategic breakpoint placement

CUDA-GDB breakpoints are much more powerful than regular breakpoints because they work with parallel execution:

```gdb
# Break when ANY thread enters your kernel
(cuda-gdb) break add_kernel

# Break only for specific threads (great for isolating issues)
(cuda-gdb) break add_kernel if thread_idx.x == 0

# Break on memory access violations
(cuda-gdb) watch input_array[thread_idx.x]

# Break on specific data conditions
(cuda-gdb) break add_kernel if input_array[thread_idx.x] > 100.0
```

This lets you focus on exactly the threads and conditions you care about, instead of drowning in output from thousands of threads.

---

### Getting your environment ready

Before you can start debugging, ensure your development environment is properly configured. If you've been working through the earlier puzzles, most of this is already set up!

**Note**: Without `pixi`, you would need to manually install CUDA Toolkit from [NVIDIA's official resources](https://developer.nvidia.com/cuda-toolkit), manage driver compatibility, configure environment variables, and handle version conflicts between components. `pixi` eliminates this complexity by automatically managing all CUDA dependencies, versions, and environment configuration for you.

#### Why `pixi` matters for debugging

**Example scenario**: GPU debugging requires precise coordination between CUDA toolkit, GPU drivers, Mojo compiler, and debugger components. Version mismatches can lead to frustrating "debugger not found" errors.

**The solution**: Using `pixi` ensures all these components work together harmoniously. When you run `pixi run mojo debug --cuda-gdb`, pixi automatically:

- Sets up CUDA toolkit paths
- Loads the correct GPU drivers
- Configures Mojo debugging plugins
- Manages environment variables consistently

#### Verifying your setup

Let's check that everything is working:

```bash
# 1. Verify GPU hardware is accessible
pixi run nvidia-smi
# Should show your GPU(s) and driver version

# 2. Set up CUDA-GDB integration (required for GPU debugging)
pixi run setup-cuda-gdb
# Links system CUDA-GDB binaries to conda environment

# 3. Verify Mojo debugger is available
pixi run mojo debug --help
# Should show debugging options including --cuda-gdb

# 4. Test CUDA-GDB integration
pixi run cuda-gdb --version
# Should show NVIDIA CUDA-GDB version information
```

If any of these commands fail, double-check your `pixi.toml` configuration and ensure the CUDA toolkit feature is enabled.

**Important**: The `pixi run setup-cuda-gdb` command is required because conda's `cuda-gdb` package only provides a wrapper script. This command auto-detects and links the actual CUDA-GDB binaries from your system CUDA installation to the conda environment, enabling full GPU debugging capabilities.

**What this command does:**

The script automatically detects CUDA from multiple common locations:

- `$CUDA_HOME` environment variable
- `/usr/local/cuda` (Ubuntu/Debian default)
- `/opt/cuda` (ArchLinux and other distributions)
- System PATH (via `which cuda-gdb`)

See [`scripts/setup-cuda-gdb.sh`](https://github.com/modular/mojo-gpu-puzzles/blob/main/scripts/setup-cuda-gdb.sh) for implementation details.

**Special note for WSL users**: Both debug tools we will use in Part II (namely cuda-gdb and compute-sanatizer) do support debugging CUDA applications on WSL, but require you to add the registry key `HKEY_LOCAL_MACHINE\SOFTWARE\NVIDIA Corporation\GPUDebugger\EnableInterface` and set it to `(DWORD) 1`. More details on supported platforms and their OS specific behavior can be found here: [cuda-gdb](https://docs.nvidia.com/cuda/cuda-gdb/index.html#supported-platforms) and [compute-sanatizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#operating-system-specific-behavior)

---

### Hands-on tutorial: Your first GPU debugging session

Theory is great, but nothing beats hands-on experience. Let's debug a real program using Puzzle 01 - the simple "add 10 to each array element" kernel you know well.

**Why Puzzle 01?**It's the perfect debugging tutorial because:

- **Simple enough**to understand what _should_ happen
- **Real GPU code**with actual kernel execution
- **Contains both**CPU setup code and GPU kernel code
- **Short execution time**so you can iterate quickly

By the end of this tutorial, you'll have debugged the same program using all four debugging approaches, seen real debugger output, and learned the essential debugging commands you'll use daily.

#### Learning path through the debugging approaches

We'll explore the [four debugging combinations](#the-four-debugging-combinations) using Puzzle 01 as our example. **Learning path**: We'll start with JIT + LLDB (easiest), then progress to CUDA-GDB (most powerful).

** Important for GPU debugging**:

- The `--break-on-launch` flag is **required**for CUDA-GDB approaches
- **Pre-compiled binaries**(Approaches 3 & 4) preserve local variables like `i` for debugging
- **JIT compilation**(Approaches 1 & 2) optimizes away most local variables
- For serious GPU debugging, use **Approach 4**(Binary + CUDA-GDB)

### Tutorial step 1: CPU debugging with LLDB

Let's begin with the most common debugging scenario: **your program crashes or behaves unexpectedly, and you need to see what's happening in your `main()` function**.

**The mission**: Debug the CPU-side setup code in Puzzle 01 to understand how Mojo initializes GPU memory and launches kernels.

#### Launch the debugger

Fire up the LLDB debugger with JIT compilation:

```bash
# This compiles and debugs p01.mojo in one step
pixi run mojo debug solutions/p01/p01.mojo
```

You'll see the LLDB prompt: `(lldb)`. You're now inside the debugger, ready to inspect your program's execution!

#### Your first debugging commands

Let's trace through what happens when Puzzle 01 runs. **Type these commands exactly as shown**and observe the output:

**Step 1: Set a breakpoint at the main function**

```bash
(lldb) br set -n main
```

Output:

```
Breakpoint 1: where = mojo`main, address = 0x00000000027d7530
```

The debugger found your main function and will pause execution there.

**Step 2: Start your program**

```bash
(lldb) run
```

Output:

```
Process 186951 launched: '/home/ubuntu/workspace/mojo-gpu-puzzles/.pixi/envs/default/bin/mojo' (x86_64)
Process 186951 stopped
* thread #1, name = 'mojo', stop reason = breakpoint 1.1
    frame #0: 0x0000555557d2b530 mojo`main
mojo`main:
->  0x555557d2b530 <+0>: pushq  %rbp
    0x555557d2b531 <+1>: movq   %rsp, %rbp
    ...
```

The program has stopped at your breakpoint. You're currently viewing **assembly code**, which is normal - the debugger starts at the low-level machine code before reaching your high-level Mojo source.

**Step 3: Navigate through the startup process**

```bash
# Try stepping through one instruction
(lldb) next
```

Output:

```
Process 186951 stopped
* thread #1, name = 'mojo', stop reason = instruction step over
    frame #0: 0x0000555557d2b531 mojo`main + 1
mojo`main:
->  0x555557d2b531 <+1>: movq   %rsp, %rbp
    0x555557d2b534 <+4>: pushq  %r15
    ...
```

Stepping through assembly can be tedious. Let's proceed to the more relevant parts.

**Step 4: Continue to reach your Mojo source code**

```bash
# Skip through the startup assembly to get to your actual code
(lldb) continue
```

Output:

```
Process 186951 resuming
Process 186951 stopped and restarted: thread 1 received signal: SIGCHLD
2 locations added to breakpoint 1
Process 186951 stopped
* thread #1, name = 'mojo', stop reason = breakpoint 1.3
    frame #0: 0x00007fff5c01e841 JIT(0x7fff5c075000)`stdlib::builtin::_startup::__mojo_main_prototype(argc=([0] = 1), argv=0x00007fffffffa858) at _startup.mojo:95:4
```

Mojo's runtime is initializing. The `_startup.mojo` indicates Mojo's internal startup code. The `SIGCHLD` signal is normal - it's how Mojo manages its internal processes.

**Step 5: Continue to your actual code**

```bash
# One more continue to reach your p01.mojo code!
(lldb) continue
```

Output:

```
Process 186951 resuming
Process 186951 stopped
* thread #1, name = 'mojo', stop reason = breakpoint 1.2
    frame #0: 0x00007fff5c014040 JIT(0x7fff5c075000)`p01::main(__error__=<unavailable>) at p01.mojo:24:23
   21
   22
   23   def main():
-> 24       with DeviceContext() as ctx:
   25           out = ctx.enqueue_create_bufferdtype
   26           out.enqueue_fill(0)
   27           a = ctx.enqueue_create_bufferdtype
```

You can now view your actual Mojo source code. Notice:

- **Line numbers 21-27**from your p01.mojo file
- **Current line 24**: `with DeviceContext() as ctx:`
- **JIT compilation**: The `JIT(0x7fff5c075000)` indicates Mojo compiled your code just-in-time

**Step 6: Let the program complete**

```bash
# Let the program run to completion
(lldb) continue
```

Output:

```
Process 186951 resuming
out: HostBuffer([10.0, 11.0, 12.0, 13.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
Process 186951 exited with status = 0 (0x00000000)
```

#### What you just learned

 **Congratulations!**You've just completed your first GPU program debugging session. Here's what happened:

**The debugging journey you took:**

1. **Started with assembly**- Normal for low-level debugging, shows how the debugger works at machine level
2. **Navigated through Mojo startup**- Learned that Mojo has internal initialization code
3. **Reached your source code**- Saw your actual p01.mojo lines 21-27 with syntax highlighting
4. **Watched JIT compilation**- Observed Mojo compiling your code on-the-fly
5. **Verified successful execution**- Confirmed your program produces the expected output

**LLDB debugging provides:**

-  **CPU-side visibility**: See your `main()` function, buffer allocation, memory setup
-  **Source code inspection**: View your actual Mojo code with line numbers
-  **Variable examination**: Check values of host-side variables (CPU memory)
-  **Program flow control**: Step through your setup logic line by line
-  **Error investigation**: Debug crashes in device setup, memory allocation, etc.

**What LLDB cannot do:**

-  **GPU kernel inspection**: Cannot step into `add_10` function execution
-  **Thread-level debugging**: Cannot see individual GPU thread behavior
-  **GPU memory access**: Cannot examine data as GPU threads see it
-  **Parallel execution analysis**: Cannot debug race conditions or synchronization

**When to use LLDB debugging:**

- Your program crashes before the GPU code runs
- Buffer allocation or memory setup issues
- Understanding program initialization and flow
- Learning how Mojo applications start up
- Quick prototyping and experimenting with code changes

**Key insight**: LLDB is perfect for **host-side debugging**- everything that happens on your CPU before and after GPU execution. For the actual GPU kernel debugging, you need our next approach...

### Tutorial step 2: Binary debugging

You've learned JIT debugging - now let's explore the **professional approach**used in production environments.

**The scenario**: You're debugging a complex application with multiple files, or you need to debug the same program repeatedly. Building a binary first provides more control and faster debugging iterations.

#### Build your debug binary

**Step 1: Compile with debug information**

```bash
# Create a debug build (notice the clear naming)
pixi run mojo build -O0 -g solutions/p01/p01.mojo -o solutions/p01/p01_debug
```

**What happens here:**

-  **`-O0`**: Disables optimizations (critical for accurate debugging)
-  **`-g`**: Includes debug symbols mapping machine code to source code
-  **`-o p01_debug`**: Creates a clearly named debug binary

**Step 2: Debug the binary**

```bash
# Debug the pre-built binary
pixi run mojo debug solutions/p01/p01_debug
```

#### What's different (and better)

**Startup comparison:**

| JIT Debugging | Binary Debugging |
|---------------|------------------|
| Compile + debug in one step | Build once, debug many times |
| Slower startup (compilation overhead) | Faster startup |
| Compilation messages mixed with debug output | Clean debugger output |
| Debug symbols generated during debugging | Fixed debug symbols |

**When you run the same LLDB commands**(`br set -n main`, `run`, `continue`), you'll notice:

- **Faster startup**- no compilation delay
- **Cleaner output**- no JIT compilation messages
- **More predictable**- debug symbols don't change between runs
- **Professional workflow**- this is how production debugging works

---

### Tutorial step 3: Debugging the GPU kernel

So far, you've debugged the **CPU host code**- the setup, memory allocation, and initialization. But what about the actual **GPU kernel**where the parallel computation happens?

**Example scenario**: Your `add_10` kernel runs on the GPU with potentially thousands of threads executing simultaneously. LLDB can't reach into the GPU's parallel execution environment.

**The solution**: CUDA-GDB - a specialized debugger that understands GPU threads, GPU memory, and parallel execution.

#### Why you need CUDA-GDB

Let's understand what makes GPU debugging fundamentally different:

**CPU debugging (LLDB):**

- One thread executing sequentially
- Single call stack to follow
- Straightforward memory model
- Variables have single values

**GPU debugging (CUDA-GDB):**

- Thousands of threads executing in parallel
- Multiple call stacks (one per thread)
- Complex memory hierarchy (global, shared, local, registers)
- Same variable has different values across threads

**Real example**: In your `add_10` kernel, the variable `thread_idx.x` has a **different value in every thread**- thread 0 sees `0`, thread 1 sees `1`, etc. Only CUDA-GDB can show you this parallel reality.

#### Launch CUDA-GDB debugger

**Step 1: Start GPU kernel debugging**

Choose your approach:

```bash
# Make sure you've run this already (once is enough)
pixi run setup-cuda-gdb

# We'll use JIT + CUDA-GDB (Approach 2 from above)
pixi run mojo debug --cuda-gdb --break-on-launch solutions/p01/p01.mojo
```

We'll use the **JIT + CUDA-GDB approach**since it's perfect for learning and quick iterations.

**Step 2: Launch and automatically stop at GPU kernel entry**

The CUDA-GDB prompt looks like: `(cuda-gdb)`. Start the program:

```gdb
# Run the program - it automatically stops when the GPU kernel launches
(cuda-gdb) run
```

Output:

```
Starting program: /home/ubuntu/workspace/mojo-gpu-puzzles/.pixi/envs/default/bin/mojo...
[Thread debugging using libthread_db enabled]
...
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0)]

CUDA thread hit application kernel entry function breakpoint, p01_add_10_UnsafePointer...
   <<<(1,1,1),(4,1,1)>>> (output=0x302000000, a=0x302000200) at p01.mojo:16
16          i = thread_idx.x
```

**Success! You're automatically stopped inside the GPU kernel!**The `--break-on-launch` flag caught the kernel launch and you're now at line 16 where `i = thread_idx.x` executes.

**Important**: You **don't**need to manually set breakpoints like `break add_10` - the kernel entry breakpoint is automatic. GPU kernel functions have mangled names in CUDA-GDB (like `p01_add_10_UnsafePointer...`), but you're already inside the kernel and can start debugging immediately.

**Step 3: Explore the parallel execution**

```gdb
# See all the GPU threads that are paused at your breakpoint
(cuda-gdb) info cuda threads
```

Output:

```
  BlockIdx ThreadIdx To BlockIdx To ThreadIdx Count                 PC                                                       Filename  Line
Kernel 0
*  (0,0,0)   (0,0,0)     (0,0,0)      (3,0,0)     4 0x00007fffd326fb70 /home/ubuntu/workspace/mojo-gpu-puzzles/solutions/p01/p01.mojo    16
```

Perfect! This shows you **all 4 parallel GPU threads**from Puzzle 01:

- **`*` marks your current thread**: `(0,0,0)` - the thread you're debugging
- **Thread range**: From `(0,0,0)` to `(3,0,0)` - all 4 threads in the block
- **Count**: `4` - matches `THREADS_PER_BLOCK = 4` from the code
- **Same location**: All threads are paused at line 16 in `p01.mojo`

**Step 4: Step through the kernel and examine variables**

```gdb
# Use 'next' to step through code (not 'step' which goes into internals)
(cuda-gdb) next
```

Output:

```
p01_add_10_UnsafePointer... at p01.mojo:17
17          output[i] = a[i] + 10.0
```

```gdb
# Local variables work with pre-compiled binaries!
(cuda-gdb) print i
```

Output:

```
$1 = 0                    # This thread's index (captures thread_idx.x value)
```

```gdb
# GPU built-ins don't work, but you don't need them
(cuda-gdb) print thread_idx.x
```

Output:

```
No symbol "thread_idx" in current context.
```

```gdb
# Access thread-specific data using local variables
(cuda-gdb) print a[i]     # This thread's input: a[0]
```

Output:

```
$2 = {0}                  # Input value (Mojo scalar format)
```

```gdb
(cuda-gdb) print output[i] # This thread's output BEFORE computation
```

Output:

```
$3 = {0}                  # Still zero - computation hasn't executed yet!
```

```gdb
# Execute the computation line
(cuda-gdb) next
```

Output:

```
13      fn add_10(         # Steps to function signature line after computation
```

```gdb
# Now check the result
(cuda-gdb) print output[i]
```

Output:

```
$4 = {10}                 # Now shows the computed result: 0 + 10 = 10
```

```gdb
# Function parameters are still available
(cuda-gdb) print a
```

Output:

```
$5 = (!pop.scalar<f32> * @register) 0x302000200
```

**Step 5: Navigate between parallel threads**

```gdb
# Switch to a different thread to see its execution
(cuda-gdb) cuda thread (1,0,0)
```

Output:

```
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (1,0,0), device 0, sm 0, warp 0, lane 1]
13      fn add_10(         # Thread 1 is also at function signature
```

```gdb
# Check the thread's local variable
(cuda-gdb) print i
```

Output:

```
$5 = 1                    # Thread 1's index (different from Thread 0!)
```

```gdb
# Examine what this thread processes
(cuda-gdb) print a[i]     # This thread's input: a[1]
```

Output:

```
$6 = {1}                  # Input value for thread 1
```

```gdb
# Thread 1's computation is already done (parallel execution!)
(cuda-gdb) print output[i] # This thread's output: output[1]
```

Output:

```
$7 = {11}                 # 1 + 10 = 11 (already computed)
```

```gdb
# BEST TECHNIQUE: View all thread results at once
(cuda-gdb) print output[0]@4
```

Output:

```
$8 = {{10}, {11}, {12}, {13}}     # All 4 threads' results in one command!
```

```gdb
(cuda-gdb) print a[0]@4
```

Output:

```
$9 = {{0}, {1}, {2}, {3}}         # All input values for comparison
```

```gdb
# Don't step too far or you'll lose CUDA context
(cuda-gdb) next
```

Output:

```
[Switching to Thread 0x7ffff7e25840 (LWP 306942)]  # Back to host thread
0x00007fffeca3f831 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
```

```gdb
(cuda-gdb) print output[i]
```

Output:

```
No symbol "output" in current context.  # Lost GPU context!
```

**Key insights from this debugging session:**

-  **Parallel execution is real**- when you switch to thread (1,0,0), its computation is already done!
- **Each thread has different data**- `i=0` vs `i=1`, `a[i]={0}` vs `a[i]={1}`, `output[i]={10}` vs `output[i]={11}`
- **Array inspection is powerful**- `print output[0]@4` shows all threads' results: `{{10}, {11}, {12}, {13}}`
- **GPU context is fragile**- stepping too far switches back to host thread and loses GPU variables

This demonstrates the fundamental nature of parallel computing: **same code, different data per thread, executing simultaneously.**

#### What you've learned with CUDA-GDB

You've completed GPU kernel execution debugging with **pre-compiled binaries**. Here's what actually works:

**GPU debugging capabilities you gained:**

-  **Debug GPU kernels automatically**- `--break-on-launch` stops at kernel entry
-  **Navigate between GPU threads**- switch contexts with `cuda thread`
-  **Access local variables**- `print i` works with `-O0 -g` compiled binaries
-  **Inspect thread-specific data**- each thread shows different `i`, `a[i]`, `output[i]` values
-  **View all thread results**- `print output[0]@4` shows `{{10}, {11}, {12}, {13}}` in one command
-  **Step through GPU code**- `next` executes computation and shows results
-  **See parallel execution**- threads execute simultaneously (other threads already computed when you switch)
-  **Access function parameters**- examine `output` and `a` pointers
-  **GPU built-ins unavailable**- `thread_idx.x`, `blockIdx.x` etc. don't work (but local variables do!)
-  **Mojo scalar format**- values display as `{10}` instead of `10.0`
-  **Fragile GPU context**- stepping too far loses access to GPU variables

**Key insights**:

- **Pre-compiled binaries**(`mojo build -O0 -g`) are essential - local variables preserved
- **Array inspection with `@N`**- most efficient way to see all parallel results at once
- **GPU built-ins are missing**- but local variables like `i` capture what you need
- **Mojo uses `{value}` format**- scalars display as `{10}` instead of `10.0`
- **Be careful with stepping**- easy to lose GPU context and return to host thread

**Real-world debugging techniques**

Now let's explore practical debugging scenarios you'll encounter in real GPU programming:

##### Technique 1: Verifying thread boundaries

```gdb
# Check if all 4 threads computed correctly
(cuda-gdb) print output[0]@4
```

Output:

```
$8 = {{10}, {11}, {12}, {13}}    # All 4 threads computed correctly
```

```gdb
# Check beyond valid range to detect out-of-bounds issues
(cuda-gdb) print output[0]@5
```

Output:

```
$9 = {{10}, {11}, {12}, {13}, {0}}  # Element 4 is uninitialized (good!)
```

```gdb
# Compare with input to verify computation
(cuda-gdb) print a[0]@4
```

Output:

```
$10 = {{0}, {1}, {2}, {3}}       # Input values: 0+10=10, 1+10=11, etc.
```

**Why this matters**: Out-of-bounds access is the #1 cause of GPU crashes. These debugging steps catch it early.

##### Technique 2: Understanding thread organization

```gdb
# See how your threads are organized into blocks
(cuda-gdb) info cuda blocks
```

Output:

```
  BlockIdx To BlockIdx Count   State
Kernel 0
*  (0,0,0)     (0,0,0)     1 running
```

```gdb
# See all threads in the current block
(cuda-gdb) info cuda threads
```

Output shows which threads are active, stopped, or have errors.

**Why this matters**: Understanding thread block organization helps debug synchronization and shared memory issues.

##### Technique 3: Memory access pattern analysis

```gdb
# Check GPU memory addresses:
(cuda-gdb) print a               # Input array GPU pointer
```

Output:

```
$9 = (!pop.scalar<f32> * @register) 0x302000200
```

```gdb
(cuda-gdb) print output          # Output array GPU pointer
```

Output:

```
$10 = (!pop.scalar<f32> * @register) 0x302000000
```

```gdb
# Verify memory access pattern using local variables:
(cuda-gdb) print a[i]            # Each thread accesses its own element using 'i'
```

Output:

```
$11 = {0}                        # Thread's input data
```

**Why this matters**: Memory access patterns affect performance and correctness. Wrong patterns cause race conditions or crashes.

##### Technique 4: Results verification and completion

```gdb
# After stepping through kernel execution, verify the final results
(cuda-gdb) print output[0]@4
```

Output:

```
$11 = {10.0, 11.0, 12.0, 13.0}    # Perfect! Each element increased by 10
```

```gdb
# Let the program complete normally
(cuda-gdb) continue
```

Output:

```
...Program output shows success...
```

```gdb
# Exit the debugger
(cuda-gdb) exit
```

You've completed debugging a GPU kernel execution from setup to results.

### Your GPU debugging progress: key insights

You've completed a comprehensive GPU debugging tutorial. Here's what you discovered about parallel computing:

#### Deep insights about parallel execution

1. **Thread indexing in action**: You **saw**`thread_idx.x` have different values (0, 1, 2, 3...) across parallel threads - not just read about it in theory

2. **Memory access patterns revealed**: Each thread accesses `a[thread_idx.x]` and writes to `output[thread_idx.x]`, creating perfect data parallelism with no conflicts

3. **Parallel execution demystified**: Thousands of threads executing the **same kernel code**simultaneously, but each processing **different data elements**

4. **GPU memory hierarchy**: Arrays live in global GPU memory, accessible by all threads but with thread-specific indexing

#### Debugging techniques that transfer to all puzzles

**From Puzzle 01 to Puzzle 08 and beyond**, you now have techniques that work universally:

- **Start with LLDB**for CPU-side issues (device setup, memory allocation)
- **Switch to CUDA-GDB**for GPU kernel issues (thread behavior, memory access)
- **Use conditional breakpoints**to focus on specific threads or data conditions
- **Navigate between threads**to understand parallel execution patterns
- **Verify memory access patterns**to catch race conditions and out-of-bounds errors

**Scalability**: These same techniques work whether you're debugging:

- **Puzzle 01**: 4-element arrays with simple addition
- **Puzzle 08**: Complex shared memory operations with thread synchronization
- **Production code**: Million-element arrays with sophisticated algorithms

---

### Essential debugging commands reference

Now that you've learned the debugging workflow, here's your **quick reference guide**for daily debugging sessions. Bookmark this section!

#### GDB command abbreviations (save time!)

**Most commonly used shortcuts**for faster debugging:

| Abbreviation | Full Command | Function |
|-------------|-------------|----------|
| `r` | `run` | Start/launch the program |
| `c` | `continue` | Resume execution |
| `n` | `next` | Step over (same level) |
| `s` | `step` | Step into functions |
| `b` | `break` | Set breakpoint |
| `p` | `print` | Print variable value |
| `l` | `list` | Show source code |
| `q` | `quit` | Exit debugger |

**Examples:**

```bash
(cuda-gdb) r                    # Instead of 'run'
(cuda-gdb) b 39                 # Instead of 'break 39'
(cuda-gdb) p thread_id          # Instead of 'print thread_id'
(cuda-gdb) n                    # Instead of 'next'
(cuda-gdb) c                    # Instead of 'continue'
```

** Pro tip**: Use abbreviations for 3-5x faster debugging sessions!

### LLDB commands (CPU host code debugging)

**When to use**: Debugging device setup, memory allocation, program flow, host-side crashes

#### Execution control

```bash
(lldb) run                    # Launch your program
(lldb) continue              # Resume execution (alias: c)
(lldb) step                  # Step into functions (source level)
(lldb) next                  # Step over functions (source level)
(lldb) finish                # Step out of current function
```

#### Breakpoint management

```bash
(lldb) br set -n main        # Set breakpoint at main function
(lldb) br set -n function_name     # Set breakpoint at any function
(lldb) br list               # Show all breakpoints
(lldb) br delete 1           # Delete breakpoint #1
(lldb) br disable 1          # Temporarily disable breakpoint #1
```

#### Variable inspection

```bash
(lldb) print variable_name   # Show variable value
(lldb) print pointer[offset]        # Dereference pointer
(lldb) print array[0]@4      # Show first 4 array elements
```

### CUDA-GDB commands (GPU kernel debugging)

**When to use**: Debugging GPU kernels, thread behavior, parallel execution, GPU memory issues

#### GPU state inspection

```bash
(cuda-gdb) info cuda threads    # Show all GPU threads and their state
(cuda-gdb) info cuda blocks     # Show all thread blocks
(cuda-gdb) cuda kernel          # List active GPU kernels
```

#### Thread navigation (The most powerful feature!)

```bash
(cuda-gdb) cuda thread (0,0,0)  # Switch to specific thread coordinates
(cuda-gdb) cuda block (0,0)     # Switch to specific block
(cuda-gdb) cuda thread          # Show current thread coordinates
```

#### Thread-specific variable inspection

```bash
# Local variables and function parameters:
(cuda-gdb) print i              # Local thread index variable
(cuda-gdb) print output         # Function parameter pointers
(cuda-gdb) print a              # Function parameter pointers
```

#### GPU memory access

```bash
# Array inspection using local variables (what actually works):
(cuda-gdb) print array[i]       # Thread-specific array access using local variable
(cuda-gdb) print array[0]@4     # View multiple elements: {{val1}, {val2}, {val3}, {val4}}
```

#### Advanced GPU debugging

```bash
# Memory watching
(cuda-gdb) watch array[i]     # Break on memory changes
(cuda-gdb) rwatch array[i]    # Break on memory reads
```

---

### Quick reference: Debugging decision tree

** What type of issue are you debugging?**

#### Program crashes before GPU code runs

 **Use LLDB debugging**

```bash
pixi run mojo debug your_program.mojo
```

#### GPU kernel produces wrong results

 **Use CUDA-GDB with conditional breakpoints**

```bash
pixi run mojo debug --cuda-gdb --break-on-launch your_program.mojo
```

#### Performance issues or race conditions

 **Use binary debugging for repeatability**

```bash
pixi run mojo build -O0 -g your_program.mojo -o debug_binary
pixi run mojo debug --cuda-gdb --break-on-launch debug_binary
```

---

### You've learned the essentials of GPU debugging

You've completed a comprehensive tutorial on GPU debugging fundamentals. Here's what you've accomplished:

#### Skills you've learned

**Multi-level debugging knowledge**:

-  **CPU host debugging**with LLDB - debug device setup, memory allocation, program flow
-  **GPU kernel debugging**with CUDA-GDB - debug parallel threads, GPU memory, race conditions
-  **JIT vs binary debugging**- choose the right approach for different scenarios
-  **Environment management**with pixi - ensure consistent, reliable debugging setups

**Real parallel programming insights**:

- **Saw threads in action**- witnessed `thread_idx.x` having different values across parallel threads
- **Understood memory hierarchy**- debugged global GPU memory, shared memory, thread-local variables
- **Learned thread navigation**- jumped between thousands of parallel threads efficiently

#### From theory to practice

You didn't just read about GPU debugging - you **experienced it**:

- **Debugged real code**: Puzzle 01's `add_10` kernel with actual GPU execution
- **Saw real debugger output**: LLDB assembly, CUDA-GDB thread states, memory addresses
- **Used professional tools**: The same CUDA-GDB used in production GPU development
- **Solved real scenarios**: Out-of-bounds access, race conditions, kernel launch failures

#### Your debugging toolkit

**Quick decision guide**(keep this handy!):

| Problem Type | Tool | Command |
|-------------|------|---------|
| **Program crashes before GPU**| LLDB | `pixi run mojo debug program.mojo` |
| **GPU kernel issues**| CUDA-GDB | `pixi run mojo debug --cuda-gdb --break-on-launch program.mojo` |
| **Race conditions**| CUDA-GDB + thread nav | `(cuda-gdb) cuda thread (0,0,0)` |

**Essential commands**(for daily debugging):

```bash
# GPU thread inspection
(cuda-gdb) info cuda threads          # See all threads
(cuda-gdb) cuda thread (0,0,0)        # Switch threads
(cuda-gdb) print i                    # Local thread index (thread_idx.x equivalent)

# Smart breakpoints (using local variables since GPU built-ins don't work)
(cuda-gdb) break kernel if i == 0      # Focus on thread 0
(cuda-gdb) break kernel if array[i] > 100  # Focus on data conditions

# Memory debugging
(cuda-gdb) print array[i]              # Thread-specific data using local variable
(cuda-gdb) print array[0]@4            # Array segments: {{val1}, {val2}, {val3}, {val4}}
```

---

#### Summary

GPU debugging involves thousands of parallel threads, complex memory hierarchies, and specialized tools. You now have:

- **Systematic workflows**that work for any GPU program
- **Professional tools**familiarity with LLDB and CUDA-GDB
- **Real experience**debugging actual parallel code
- **Practical strategies**for handling complex scenarios
- **Foundation**to tackle GPU debugging challenges

---

### Additional resources

- [Mojo Debugging Documentation](https://docs.modular.com/mojo/tools/debugging)
- [Mojo GPU Debugging Guide](https://docs.modular.com/mojo/tools/gpu-debugging)
- [NVIDIA CUDA-GDB User Guide](https://docs.nvidia.com/cuda/cuda-gdb/index.html)
- [CUDA-GDB Command Reference](https://docs.nvidia.com/cuda/cuda-gdb/index.html#command-reference)

**Note**: GPU debugging requires patience and systematic investigation. The workflow and commands in this puzzle provide the foundation for debugging complex GPU issues you'll encounter in real applications.

##  Detective Work: First Case

### Overview

This puzzle presents a crashing GPU program where Example goal:  identify the issue using only `(cuda-gdb)` debugging tools, without examining the source code. Apply your debugging skills to solve the mystery!

**Prerequisites**: Complete [Mojo GPU Debugging Essentials](#mojo-gpu-debugging-essentials) to understand CUDA-GDB setup and basic debugging commands. Make sure you've run:

```bash
pixi run -e nvidia setup-cuda-gdb
```

This auto-detects your CUDA installation and sets up the necessary links for GPU debugging.

### Key concepts

In this debugging challenge, you'll learn about:

- **Systematic debugging**: Using error messages as clues to find root causes
- **Error analysis**: Reading crash messages and stack traces
- **Hypothesis formation**: Making educated guesses about the problem
- **Debugging workflow**: Step-by-step investigation process

### Running the code

First, examine the kernel without looking at the complete code:

```mojo
fn add_10(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    i = thread_idx.x
    output[i] = a[i] + 10.0

```

To experience the bug firsthand, run the following command in your terminal (`pixi` only):

```bash
pixi run -e nvidia p09 --first-case
```

You'll see output like this when the program crashes:

```txt
First Case: Try to identify what's wrong without looking at the code!

stack trace was not collected. Enable stack trace collection with environment variable `MOJO_ENABLE_STACK_TRACE_ON_ERROR`
Unhandled exception caught during execution: At open-source/max/mojo/stdlib/stdlib/gpu/host/device_context.mojo:2082:17: CUDA call failed: CUDA_ERROR_INVALID_IMAGE (device kernel image is invalid)
To get more accurate error information, set MODULAR_DEVICE_CONTEXT_SYNC_MODE=true.
/home/ubuntu/workspace/mojo-gpu-puzzles/.pixi/envs/nvidia/bin/mojo: error: execution exited with a non-zero result: 1
```

### Example goal: detective work

**Design prompt**: Without looking at the code yet, what would be your debugging strategy to investigate this crash?

Start with:

```bash
pixi run -e nvidia mojo debug --cuda-gdb --break-on-launch problems/p09/p09.mojo --first-case
```

#### Tips

1. **Read the crash message carefully**- `CUDA_ERROR_ILLEGAL_ADDRESS` means the GPU tried to access invalid memory
2. **Check the breakpoint information**- Look at the function parameters shown when CUDA-GDB stops
3. **Inspect all pointers systematically**- Use `print` to examine each pointer parameter
4. **Look for suspicious addresses**- Valid GPU addresses are typically large hex numbers (what does `0x0` mean?)
5. **Test memory access**- Try accessing the data through each pointer to see which one fails
6. **Apply the systematic approach**- Like a detective, follow the evidence from symptom to root cause
7. **Compare valid vs invalid patterns**- If one pointer works and another doesn't, focus on the broken one

####  Investigation & Solution

### Step-by-Step Investigation with CUDA-GDB

#### Launch the Debugger

```bash
pixi run -e nvidia mojo debug --cuda-gdb --break-on-launch problems/p09/p09.mojo --first-case
```

#### Examine the Breakpoint Information

When CUDA-GDB stops, it immediately shows valuable clues:

```
(cuda-gdb) run
CUDA thread hit breakpoint, p09_add_10_... (output=0x302000000, a=0x0)
    at /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p09/p09.mojo:31
31          i = thread_idx.x
```

** First Clue**: The function signature shows `(output=0x302000000, a=0x0)`

- `output` has a valid GPU memory address
- `a` is `0x0` - this is a null pointer!

#### Systematic variable inspection

```
(cuda-gdb) next
32          output[i] = a[i] + 10.0
(cuda-gdb) print i
$1 = 0
(cuda-gdb) print output
$2 = (!pop.scalar<f32> * @register) 0x302000000
(cuda-gdb) print a
$3 = (!pop.scalar<f32> * @register) 0x0
```

**Evidence Gathering**:

-  Thread index `i=0` is valid
-  Result pointer `0x302000000` is a proper GPU address
-  Input pointer `0x0` is null

#### Confirm the Problem

```
(cuda-gdb) print a[i]
Cannot access memory at address 0x0
```

**Smoking Gun**: Cannot access memory at null address - this confirms the crash cause!

### Root cause analysis

**The Problem**: Now if we look at the code for `--first-crash`, we see that the host code creates a null pointer instead of allocating proper GPU memory:

```mojo
 input_buf = ctx.enqueue_create_bufferdtype  # Creates a `DeviceBuffer` with 0 elements. Since there are zero elements, no memory is allocated, which results in a NULL pointer!
```

**Why This Crashes**:

1. `ctx.enqueue_create_bufferdtype` creates a `DeviceBuffer` with zero (0) elements.
2. since there are no elements for which to allocate memory, this returns a null pointer.
3. This null pointer gets passed to the GPU kernel
5. When kernel tries `a[i]`, it dereferences null  `CUDA_ERROR_ILLEGAL_ADDRESS`

### The fix

Replace null pointer creation with proper buffer allocation:

```mojo
# Wrong: Creates null pointer
input_buf = ctx.enqueue_create_bufferdtype

# Correct: Allocates and initialize actual GPU memory for safe processing
input_buf = ctx.enqueue_create_bufferdtype
input_bufenqueue_fill(0)
```

### Key debugging lessons

**Pattern Recognition**:

- `0x0` addresses are always null pointers
- Valid GPU addresses are large hex numbers (e.g., `0x302000000`)

**Debugging Strategy**:

1. **Read crash messages**- They often hint at the problem type
2. **Check function parameters**- CUDA-GDB shows them at breakpoint entry
3. **Inspect all pointers**- Compare addresses to identify null/invalid ones
4. **Test memory access**- Try dereferencing suspicious pointers
5. **Trace back to allocation**- Find where the problematic pointer was created

** Key Insight**: This type of null pointer bug is extremely common in GPU programming. The systematic CUDA-GDB investigation approach you learned here applies to debugging many other GPU memory issues, race conditions, and kernel crashes.

**You've learned crash debugging!**You can now:

- **Systematically investigate GPU crashes**using error messages as clues
- **Identify null pointer bugs**through pointer address inspection
- **Use CUDA-GDB effectively**for memory-related debugging

#### Your next challenge: [Detective Work: Second Case](#detective-work-second-case)

**But what if your program doesn't crash?**What if it runs perfectly but produces **wrong results**?

The [Second Case](#detective-work-second-case) presents a completely different debugging challenge:

- **No crash messages**to guide you
- **No obvious pointer problems**to investigate
- **No stack traces**pointing to the issue
- **Just wrong results**that need systematic investigation

**New skills you'll develop:**

- **Logic bug detection**- Finding algorithmic errors without crashes
- **Pattern analysis**- Using incorrect output to trace back to root causes
- **Execution flow debugging**- When variable inspection fails due to optimizations

The systematic investigation approach you learned here - reading clues, forming hypotheses, testing systematically - forms the foundation for debugging the more subtle logic errors ahead.

##  Detective Work: Second Case

### Overview

Building on your [crash debugging skills from the First Case](#detective-work-first-case), you'll now face a completely different challenge: a **logic bug**that produces incorrect results without crashing.

**The debugging shift:**

- **First Case**: Clear crash signals (`CUDA_ERROR_ILLEGAL_ADDRESS`) guided your investigation
- **Second Case**: No crashes, no error messages - just subtly wrong results that require detective work

This intermediate-level debugging challenge covers investigating **algorithmic errors**using `LayoutTensor` operations, where the program runs successfully but produces wrong output - a much more common (and trickier) real-world debugging scenario.

**Prerequisites**: Complete [Mojo GPU Debugging Essentials](#mojo-gpu-debugging-essentials) and [Detective Work: First Case](#detective-work-first-case) to understand CUDA-GDB workflow and systematic debugging techniques. Make sure you run the setup:

```bash
pixi run -e nvidia setup-cuda-gdb
```

### Key concepts

In this debugging challenge, you'll learn about:

- **LayoutTensor debugging**: Investigating structured data access patterns
- **Logic bug detection**: Finding algorithmic errors that don't crash
- **Loop boundary analysis**: Understanding iteration count problems
- **Result pattern analysis**: Using output data to trace back to root causes

### Running the code

First, examine the kernel without looking at the complete code:

```mojo
fn process_sliding_window(
    output: LayoutTensor[dtype, vector_layout, MutAnyOrigin],
    a: LayoutTensor[dtype, vector_layout, ImmutAnyOrigin],
):
    thread_id = thread_idx.x

    # Each thread processes a sliding window of 3 elements
    window_sum = Scalardtype

    # Sum elements in sliding window: [i-1, i, i+1]
    for offset in range(ITER):
        idx = Int(thread_id) + offset - 1
        if 0 <= idx < SIZE:
            value = rebind[Scalar[dtype]](a[idx])
            window_sum += value

    output[thread_id] = window_sum

```

To experience the bug firsthand, run the following command in your terminal (`pixi` only):

```bash
pixi run -e nvidia p09 --second-case
```

You'll see output like this - **no crash, but wrong results**:

```txt
This program computes sliding window sums for each position...

Input array: [0, 1, 2, 3]
Computing sliding window sums (window size = 3)...
Each position should sum its neighbors: [left + center + right]
stack trace was not collected. Enable stack trace collection with environment variable `MOJO_ENABLE_STACK_TRACE_ON_ERROR`
Unhandled exception caught during execution: At open-source/max/mojo/stdlib/stdlib/gpu/host/device_context.mojo:2082:17: CUDA call failed: CUDA_ERROR_INVALID_IMAGE (device kernel image is invalid)
To get more accurate error information, set MODULAR_DEVICE_CONTEXT_SYNC_MODE=true.
/home/ubuntu/workspace/mojo-gpu-puzzles/.pixi/envs/nvidia/bin/mojo: error: execution exited with a non-zero result: 1
```

### Example goal: detective work

**Design prompt**: The program runs without crashing but produces consistently wrong results. Without looking at the code, what would be your systematic approach to investigate this logic bug?

**Think about:**

- What pattern do you see in the wrong results?
- How would you investigate a loop that might not be running correctly?
- What debugging strategy works when you can't inspect variables directly?
- How can you apply the systematic investigation approach from [First Case](#detective-work-first-case) when there are no crash signals to guide you?

Start with:

```bash
pixi run -e nvidia mojo debug --cuda-gdb --break-on-launch problems/p09/p09.mojo --second-case
```

#### GDB command shortcuts (faster debugging)

**Use these abbreviations**to speed up your debugging session:

| Short | Full | Usage Example |
|-------|------|---------------|
| `r` | `run` | `(cuda-gdb) r` |
| `n` | `next` | `(cuda-gdb) n` |
| `c` | `continue` | `(cuda-gdb) c` |
| `b` | `break` | `(cuda-gdb) b 39` |
| `p` | `print` | `(cuda-gdb) p thread_id` |
| `q` | `quit` | `(cuda-gdb) q` |

**All debugging commands below use these shortcuts for efficiency!**

#### Tips

1. **Pattern analysis first**- Look at the relationship between expected and actual results (what's the mathematical pattern in the differences?)
2. **Focus on execution flow**- Count loop iterations when variables aren't accessible
3. **Use simple breakpoints**- Complex debugging commands often fail with optimized code
4. **Mathematical reasoning**- Work out what each thread should access vs what it actually accesses
5. **Missing data investigation**- If results are consistently smaller than expected, what might be missing?
6. **Host output verification**- The final results often reveal the pattern of the bug
7. **Algorithm boundary analysis**- Check if loops are processing the right number of elements
8. **Cross-validate with working cases**- Why does thread 3 work correctly but others don't?

####  Investigation & Solution

### Step-by-step investigation with CUDA-GDB

#### Phase 1: Launch and initial analysis

##### Step 1: Start the debugger

```bash
pixi run -e nvidia mojo debug --cuda-gdb --break-on-launch problems/p09/p09.mojo --second-case
```

##### Step 2: analyze the symptoms first

Before diving into the debugger, examine what we know:

```txt
Actual result: [0.0, 1.0, 3.0, 5.0]
Expected: [1.0, 3.0, 6.0, 5.0]
```

** Pattern Recognition**:

- Thread 0: Got 0.0, Expected 1.0  Missing 1.0
- Thread 1: Got 1.0, Expected 3.0  Missing 2.0
- Thread 2: Got 3.0, Expected 6.0  Missing 3.0
- Thread 3: Got 5.0, Expected 5.0   Correct

**Initial Hypothesis**: Each thread is missing some data, but thread 3 works correctly.

#### Phase 2: Entering the kernel

##### Step 3: Observe the breakpoint entry

Based on the real debugging session, here's what happens:

```bash
(cuda-gdb) r
Starting program: .../mojo run problems/p09/p09.mojo --second-case

This program computes sliding window sums for each position...
Input array: [0, 1, 2, 3]
Computing sliding window sums (window size = 3)...
Each position should sum its neighbors: [left + center + right]

[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 0, warp 0, lane 0]

CUDA thread hit application kernel entry function breakpoint, p09_process_sliding_window_...
   <<<(1,1,1),(4,1,1)>>> (output=..., input=...)
    at /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p09/p09.mojo:30
30          input: LayoutTensor[mut=False, dtype, vector_layout],
```

##### Step 4: Navigate to the main logic

```bash
(cuda-gdb) n
29          output: LayoutTensor[mut=True, dtype, vector_layout],
(cuda-gdb) n
32          thread_id = thread_idx.x
(cuda-gdb) n
38          for offset in range(ITER):
```

##### Step 5: Test variable accessibility - crucial discovery

```bash
(cuda-gdb) p thread_id
$1 = 0
```

** Good**: Thread ID is accessible.

```bash
(cuda-gdb) p window_sum
Cannot access memory at address 0x0
```

** Problem**: `window_sum` is not accessible.

```bash
(cuda-gdb) p a[0]
Attempt to take address of value not located in memory.
```

** Problem**: Direct LayoutTensor indexing doesn't work.

```bash
(cuda-gdb) p a.ptr[0]
$2 = {0}
(cuda-gdb) p a.ptr[0]@4
$3 = {{0}, {1}, {2}, {3}}
```

** BREAKTHROUGH**: `a.ptr[0]@4` shows the full input array! This is how we can inspect LayoutTensor data.

#### Phase 3: The critical loop investigation

##### Step 6: Set up loop monitoring

```bash
(cuda-gdb) b 42
Breakpoint 1 at 0x7fffd326ffd0: file problems/p09/p09.mojo, line 42.
(cuda-gdb) c
Continuing.

CUDA thread hit Breakpoint 1, p09_process_sliding_window_...
   <<<(1,1,1),(4,1,1)>>> (output=..., input=...)
    at /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p09/p09.mojo:42
42              idx = thread_id + offset - 1
```

** We're now inside the loop body. Let's count iterations manually.**

##### Step 7: First loop iteration (offset = 0)

```bash
(cuda-gdb) n
43              if 0 <= idx < SIZE:
(cuda-gdb) n
41          for offset in range(ITER):
```

**First iteration complete**: Loop went from line 42  43  back to 41. The loop continues.

##### Step 8: Second loop iteration (offset = 1)

```bash
(cuda-gdb) n

CUDA thread hit Breakpoint 1, p09_process_sliding_window_...
42              idx = thread_id + offset - 1
(cuda-gdb) n
43              if 0 <= idx < SIZE:
(cuda-gdb) n
44                  value = rebind[Scalar[dtype]](input[idx])
(cuda-gdb) n
45                  window_sum += value
(cuda-gdb) n
43              if 0 <= idx < SIZE:
(cuda-gdb) n
41          for offset in range(ITER):
```

**Second iteration complete**: This time it went through the if-block (lines 44-45).

##### Step 9: testing for third iteration

```bash
(cuda-gdb) n
47          output[thread_id] = window_sum
```

**CRITICAL DISCOVERY**: The loop exited after only 2 iterations! It went directly to line 47 instead of hitting our breakpoint at line 42 again.

**Conclusion**: The loop ran exactly **2 iterations**and then exited.

##### Step 10: Complete kernel execution and context loss

```bash
(cuda-gdb) n
31      fn process_sliding_window(
(cuda-gdb) n
[Switching to Thread 0x7ffff7cc0e00 (LWP 110927)]
0x00007ffff064f84a in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
(cuda-gdb) p output.ptr[0]@4
No symbol "output" in current context.
(cuda-gdb) p offset
No symbol "offset" in current context.
```

** Context Lost**: After kernel completion, we lose access to kernel variables. This is normal behavior.

#### Phase 4: Root cause analysis

##### Step 11: Algorithm analysis from observed execution

From our debugging session, we observed:

1. **Loop Iterations**: Only 2 iterations (offset = 0, offset = 1)
2. **Expected**: A sliding window of size 3 should require 3 iterations (offset = 0, 1, 2)
3. **Missing**: The third iteration (offset = 2)

Looking at what each thread should compute:

- **Thread 0**: window_sum = input[-1] + input[0] + input[1] = (boundary) + 0 + 1 = 1.0
- **Thread 1**: window_sum = input[0] + input[1] + input[2] = 0 + 1 + 2 = 3.0
- **Thread 2**: window_sum = input[1] + input[2] + input[3] = 1 + 2 + 3 = 6.0
- **Thread 3**: window_sum = input[2] + input[3] + input[4] = 2 + 3 + (boundary) = 5.0

##### Step 12: Trace the actual execution for thread 0

With only 2 iterations (offset = 0, 1):

**Iteration 1 (offset = 0)**:

- `idx = thread_id + offset - 1 = 0 + 0 - 1 = -1`
- `if 0 <= idx < SIZE:`  `if 0 <= -1 < 4:`  **False**
- Skip the sum operation

**Iteration 2 (offset = 1)**:

- `idx = thread_id + offset - 1 = 0 + 1 - 1 = 0`
- `if 0 <= idx < SIZE:`  `if 0 <= 0 < 4:`  **True**
- `window_sum += input[0]`  `window_sum += 0`

**Missing Iteration 3 (offset = 2)**:

- `idx = thread_id + offset - 1 = 0 + 2 - 1 = 1`
- `if 0 <= idx < SIZE:`  `if 0 <= 1 < 4:`  **True**
- `window_sum += input[1]`  `window_sum += 1`  **THIS NEVER HAPPENS**

**Result**: Thread 0 gets `window_sum = 0` instead of `window_sum = 0 + 1 = 1`

#### Phase 5: Bug confirmation

Looking at the problem code, we find:

```mojo

for offset in range(ITER):           # <- Only 2 iterations: [0, 1]
    idx = Int(thread_id) + offset - 1     # <- Missing offset = 2
    if 0 <= idx < SIZE:
        value = rebind[Scalar[dtype]](a[idx])
        window_sum += value
```

** ROOT CAUSE IDENTIFIED**: `ITER = 2` should be `ITER = 3` for a sliding window of size 3.

**The Fix**: Change `comptime ITER = 2` to `comptime ITER = 3` in the source code.

### Key debugging lessons

**When Variables Are Inaccessible**:

1. **Focus on execution flow**- Count breakpoint hits and loop iterations
2. **Use mathematical reasoning**- Work out what should happen vs what does happen
3. **Pattern analysis**- Let the wrong results guide your investigation
4. **Cross-validation**- Test your hypothesis against multiple data points

**Professional GPU Debugging Reality**:

- **Variable inspection often fails**due to compiler optimizations
- **Execution flow analysis**is more reliable than data inspection
- **Host output patterns**provide crucial debugging clues
- **Source code reasoning**complements limited debugger capabilities

**LayoutTensor Debugging**:

- Even with LayoutTensor abstractions, underlying algorithmic bugs still manifest
- Focus on the algorithm logic rather than trying to inspect tensor contents
- Use systematic reasoning to trace what each thread should vs actually accesses

**Key Insight**: This type of off-by-one loop bug is extremely common in GPU programming. The systematic approach you learned here - combining limited debugger info with mathematical analysis and pattern recognition - is exactly how professional GPU developers debug when tools have limitations.

**You've learned logic bug debugging!**You can now:

-  **Investigate algorithmic errors**without crashes or obvious symptoms
-  **Use pattern analysis**to trace wrong results back to root causes
-  **Debug with limited variable access**using execution flow analysis
-  **Apply mathematical reasoning**when debugger tools have limitations

#### Your final challenge: [Detective Work: Third Case](#detective-work-third-case)

**But what if your program doesn't crash AND doesn't finish?**What if it just **hangs forever**?

The [Third Case](#detective-work-third-case) presents the ultimate debugging challenge:

-  **No crash messages**(like First Case)
-  **No wrong results**(like Second Case)
-  **No completion at all**- just infinite hanging
-  **Silent deadlock**requiring advanced thread coordination analysis

**New skills you'll develop:**

- **Barrier deadlock detection**- Finding coordination failures in parallel threads
- **Multi-thread state analysis**- Examining all threads simultaneously
- **Synchronization debugging**- Understanding thread cooperation breakdowns

**The debugging evolution:**

1. **First Case**: Follow crash signals  Find memory bugs
2. **Second Case**: Analyze result patterns  Find logic bugs
3. **Third Case**: Investigate thread states  Find coordination bugs

The systematic investigation skills from both previous cases - hypothesis formation, evidence gathering, pattern analysis - become crucial when debugging the most challenging GPU issue: threads that coordinate incorrectly and wait forever.

##  Detective Work: Third Case

### Overview

You've learned debugging [memory crashes](#detective-work-first-case) and [logic bugs](#detective-work-second-case). Now face the ultimate GPU debugging challenge: a **barrier deadlock**that causes the program to hang indefinitely with no error messages, no wrong results - just eternal silence.

**The complete debugging journey:**
- **[First Case](#detective-work-first-case)**: Program crashes  Follow error signals  Find memory bugs
- **[Second Case](#detective-work-second-case)**: Program produces wrong results  Analyze patterns  Find logic bugs
- **[Third Case]**: Program hangs forever  Investigate thread states  Find coordination bugs

This advanced-level debugging challenge teaches you to investigate **thread coordination failures**using shared memory, LayoutTensor operations, and barrier synchronization - combining all the systematic investigation skills from the previous cases.

**Prerequisites**: Complete [Mojo GPU Debugging Essentials](#mojo-gpu-debugging-essentials), [Detective Work: First Case](#detective-work-first-case), and [Detective Work: Second Case](#detective-work-second-case) to understand CUDA-GDB workflow, variable inspection limitations, and systematic debugging approaches. Make sure you run the setup:

```bash
pixi run -e nvidia setup-cuda-gdb
```

### Key concepts

In this debugging challenge, you'll learn about:
- **Barrier deadlock detection**: Identifying when threads wait forever at synchronization points
- **Shared memory coordination**: Understanding thread cooperation patterns with LayoutTensor
- **Conditional execution analysis**: Debugging when some threads take different code paths
- **Thread coordination debugging**: Using CUDA-GDB to analyze multi-thread synchronization failures

### Running the code

First, examine the kernel without looking at the complete code:

```mojo
fn collaborative_filter(
    output: LayoutTensor[dtype, vector_layout, MutAnyOrigin],
    a: LayoutTensor[dtype, vector_layout, ImmutAnyOrigin],
):
    thread_id = thread_idx.x

    # Shared memory workspace for collaborative processing
    shared_workspace = LayoutTensor[
        dtype,
        Layout.row_major(SIZE - 1),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Phase 1: Initialize shared workspace (all threads participate)
    if thread_id < SIZE - 1:
        shared_workspace[thread_id] = rebind[Scalar[dtype]](a[thread_id])
    barrier()

    # Phase 2: Collaborative processing
    if thread_id < SIZE - 1:
        # Apply collaborative filter with neighbors
        if thread_id > 0:
            shared_workspace[thread_id] += shared_workspace[thread_id - 1] * 0.5
        barrier()

    # Phase 3: Final synchronization and output
    barrier()

    # Write filtered results back to output
    if thread_id < SIZE - 1:
        output[thread_id] = shared_workspace[thread_id]
    else:
        output[thread_id] = rebind[Scalar[dtype]](a[thread_id])

```

To experience the bug firsthand, run the following command in your terminal (`pixi` only):

```bash
pixi run -e nvidia p09 --third-case
```

You'll see output like this - **the program hangs indefinitely**:
```txt
Third Case: Advanced collaborative filtering with shared memory...
WARNING: This may hang - use Ctrl+C to stop if needed

Input array: [1, 2, 3, 4]
Applying collaborative filter using shared memory...
Each thread cooperates with neighbors for smoothing...
Waiting for GPU computation to complete...
[HANGS FOREVER - Use Ctrl+C to stop]
```

 **Warning**: This program will hang and never complete. Use `Ctrl+C` to stop it.

### Example goal: detective work

**Design prompt**: The program launches successfully but hangs during GPU computation and never returns. Without looking at the complete code, what would be your systematic approach to investigate this deadlock?

**Think about:**
- What could cause a GPU kernel to never complete?
- How would you investigate thread coordination issues?
- What debugging strategy works when the program just "freezes" with no error messages?
- How do you debug when threads might not be cooperating correctly?
- How can you combine systematic investigation ([First Case](#detective-work-first-case)) with execution flow analysis ([Second Case](#detective-work-second-case)) to debug coordination failures?

Start with:

```bash
pixi run -e nvidia mojo debug --cuda-gdb --break-on-launch problems/p09/p09.mojo --third-case
```

#### GDB command shortcuts (faster debugging)

**Use these abbreviations**to speed up your debugging session:

| Short | Full | Usage Example |
|-------|------|---------------|
| `r` | `run` | `(cuda-gdb) r` |
| `n` | `next` | `(cuda-gdb) n` |
| `c` | `continue` | `(cuda-gdb) c` |
| `b` | `break` | `(cuda-gdb) b 62` |
| `p` | `print` | `(cuda-gdb) p thread_id` |
| `q` | `quit` | `(cuda-gdb) q` |

**All debugging commands below use these shortcuts for efficiency!**

#### Tips

1. **Silent hang investigation**- When programs freeze without error messages, what GPU primitives could cause infinite waiting?
2. **Thread state inspection**- Use `info cuda threads` to see where different threads are stopped
3. **Conditional execution analysis**- Check which threads execute which code paths (do all threads follow the same path?)
4. **Synchronization point investigation**- Look for places where threads might need to coordinate
5. **Thread divergence detection**- Are all threads at the same program location, or are some elsewhere?
6. **Coordination primitive analysis**- What happens if threads don't all participate in the same synchronization operations?
7. **Execution flow tracing**- Follow the path each thread takes through conditional statements
8. **Thread ID impact analysis**- How do different thread IDs affect which code paths execute?

####  Investigation & Solution

### Step-by-step investigation with CUDA-GDB

#### Phase 1: launch and initial setup

##### Step 1: start the debugger
```bash
pixi run -e nvidia mojo debug --cuda-gdb --break-on-launch problems/p09/p09.mojo --third-case
```

##### Step 2: analyze the hanging behavior
Before diving into debugging, let's understand what we know:

```txt
Expected: Program completes and shows filtered results
Actual: Program hangs at "Waiting for GPU computation to complete..."
```

** Initial Hypothesis**: The GPU kernel is deadlocked - some synchronization primitive is causing threads to wait forever.

#### Phase 2: entering the kernel

##### Step 3: launch and observe kernel entry
```bash
(cuda-gdb) r
Starting program: .../mojo run problems/p09/p09.mojo --third-case

Third Case: Advanced collaborative filtering with shared memory...
WARNING: This may hang - use Ctrl+C to stop if needed

Input array: [1, 2, 3, 4]
Applying collaborative filter using shared memory...
Each thread cooperates with neighbors for smoothing...
Waiting for GPU computation to complete...

[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 0, warp 0, lane 0]

CUDA thread hit application kernel entry function breakpoint, p09_collaborative_filter_Orig6A6AcB6A6A_1882ca334fc2d34b2b9c4fa338df6c07<<<(1,1,1),(4,1,1)>>> (
    output=..., a=...)
    at /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p09/p09.mojo:56
56          a: LayoutTensor[mut=False, dtype, vector_layout],
```

** Key Observations**:
- **Grid**: (1,1,1) - single block
- **Block**: (4,1,1) - 4 threads total (0, 1, 2, 3)
- **Current thread**: (0,0,0) - debugging thread 0
- **Function**: collaborative_filter with shared memory operations

##### Step 4: navigate through initialization
```bash
(cuda-gdb) n
55          output: LayoutTensor[mut=True, dtype, vector_layout],
(cuda-gdb) n
58          thread_id = thread_idx.x
(cuda-gdb) n
66          ].stack_allocation()
(cuda-gdb) n
69          if thread_id < SIZE - 1:
(cuda-gdb) p thread_id
$1 = 0
```

** Thread 0 state**: `thread_id = 0`, about to check condition `0 < 3`  **True**

##### Step 5: trace through phase 1
```bash
(cuda-gdb) n
70              shared_workspace[thread_id] = rebind[Scalar[dtype]](a[thread_id])
(cuda-gdb) n
69          if thread_id < SIZE - 1:
(cuda-gdb) n
71          barrier()
```

**Phase 1 Complete**: Thread 0 executed the initialization and reached the first barrier.

#### Phase 3: the critical barrier investigation

##### Step 6: examine the first barrier
```bash
(cuda-gdb) n
74          if thread_id < SIZE - 1:
(cuda-gdb) info cuda threads
  BlockIdx ThreadIdx To BlockIdx To ThreadIdx Count                 PC                                                       Filename  Line
Kernel 0
*  (0,0,0)   (0,0,0)     (0,0,0)      (3,0,0)     4 0x00007fffd3272180 /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p09/p09.mojo    74
```

** Good**: All 4 threads are at line 74 (after the first barrier). The first barrier worked correctly.

** Critical Point**: Now we're entering Phase 2 with another conditional statement.

##### Step 7: trace through phase 2 - thread 0 perspective
```bash
(cuda-gdb) n
76              if thread_id > 0:
```

**Thread 0 Analysis**: `0 < 3`  **True** Thread 0 enters the Phase 2 block

```bash
(cuda-gdb) n
78              barrier()
```

**Thread 0 Path**: `0 > 0`  **False** Thread 0 skips the inner computation but reaches the barrier at line 78

**CRITICAL MOMENT**: Thread 0 is now waiting at the barrier on line 78.

```bash
(cuda-gdb) n # <-- if you run it the program hangs!
[HANGS HERE - Program never proceeds beyond this point]
```

##### Step 8: investigate other threads
```bash
(cuda-gdb) cuda thread (1,0,0)
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (1,0,0), device 0, sm 0, warp 0, lane 1]
78              barrier()
(cuda-gdb) p thread_id
$2 = 1
(cuda-gdb) info cuda threads
  BlockIdx ThreadIdx To BlockIdx To ThreadIdx Count                 PC                                                       Filename  Line
Kernel 0
*  (0,0,0)   (0,0,0)     (0,0,0)      (2,0,0)     3 0x00007fffd3273aa0 /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p09/p09.mojo    78
   (0,0,0)   (3,0,0)     (0,0,0)      (3,0,0)     1 0x00007fffd3273b10 /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p09/p09.mojo    81
```

**SMOKING GUN DISCOVERED**:
- **Threads 0, 1, 2**: All waiting at line 78 (barrier inside the conditional block)
- **Thread 3**: At line 81 (after the conditional block, never reached the barrier!)

##### Step 9: analyze thread 3's execution path

** Thread 3 Analysis from the info output**:
- **Thread 3**: Located at line 81 (PC: 0x00007fffd3273b10)
- **Phase 2 condition**: `thread_id < SIZE - 1`  `3 < 3`  **False**
- **Result**: Thread 3 **NEVER entered**the Phase 2 block (lines 74-78)
- **Consequence**: Thread 3 **NEVER reached**the barrier at line 78
- **Current state**: Thread 3 is at line 81 (final barrier), while threads 0,1,2 are stuck at line 78

#### Phase 4: root cause analysis

##### Step 10: deadlock mechanism identified
```mojo
# Phase 2: Collaborative processing
if thread_id < SIZE - 1:        # <- Only threads 0, 1, 2 enter this block
    # Apply collaborative filter with neighbors
    if thread_id > 0:
        shared_workspace[thread_id] += shared_workspace[thread_id - 1] * 0.5
    barrier()                   # <- DEADLOCK: Only 3 out of 4 threads reach here!
```

** Deadlock Mechanism**:
1. **Thread 0**: `0 < 3`  **True** Enters block  **Waits at barrier**(line 69)
2. **Thread 1**: `1 < 3`  **True** Enters block  **Waits at barrier**(line 69)
3. **Thread 2**: `2 < 3`  **True** Enters block  **Waits at barrier**(line 69)
4. **Thread 3**: `3 < 3`  **False** **NEVER enters block** **Continues to line 72**

**Result**: 3 threads wait forever for the 4th thread, but thread 3 never arrives at the barrier.

#### Phase 5: bug confirmation and solution

##### Step 11: the fundamental barrier rule violation
**GPU Barrier Rule**: ALL threads in a thread block must reach the SAME barrier for synchronization to complete.

**What went wrong**:
```mojo
# X WRONG: Barrier inside conditional
if thread_id < SIZE - 1:    # Not all threads enter
    # ... some computation ...
    barrier()               # Only some threads reach this

# OK CORRECT: Barrier outside conditional
if thread_id < SIZE - 1:    # Not all threads enter
    # ... some computation ...
 barrier()                # ALL threads reach this
```

**The Fix**: Move the barrier outside the conditional block:
```mojo
fn collaborative_filter(
    output: LayoutTensor[mut=True, dtype, vector_layout],
    a: LayoutTensor[mut=False, dtype, vector_layout],
):
    thread_id = thread_idx.x
    shared_workspace = LayoutTensor[
        dtype,
        Layout.row_major(SIZE-1),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # Phase 1: Initialize shared workspace (all threads participate)
    if thread_id < SIZE - 1:
        shared_workspace[thread_id] = rebind[Scalar[dtype]](a[thread_id])
    barrier()

    # Phase 2: Collaborative processing
    if thread_id < SIZE - 1:
        if thread_id > 0:
            shared_workspace[thread_id] += shared_workspace[thread_id - 1] * 0.5
    barrier()

    # Phase 3: Final synchronization and output
    barrier()

    if thread_id < SIZE - 1:
        output[thread_id] = shared_workspace[thread_id]
    else:
        output[thread_id] = rebind[Scalar[dtype]](a[thread_id])
```

### Key debugging lessons

**Barrier deadlock detection**:
1. **Use `info cuda threads`**- Shows which threads are at which lines
2. **Look for thread state divergence**- Some threads at different program locations
3. **Trace conditional execution paths**- Check if all threads reach the same barriers
4. **Verify barrier reachability**- Ensure no thread can skip a barrier that others reach

**Professional GPU debugging reality**:
- **Deadlocks are silent killers**- programs just hang with no error messages
- **Thread coordination debugging requires patience**- systematic analysis of each thread's path
- **Conditional barriers are the #1 deadlock cause**- always verify all threads reach the same sync points
- **CUDA-GDB thread inspection is essential**- the only way to see thread coordination failures

**Advanced GPU synchronization**:
- **Barrier rule**: ALL threads in a block must reach the SAME barrier
- **Conditional execution pitfalls**: Any if-statement can cause thread divergence
- **Shared memory coordination**: Requires careful barrier placement for correct synchronization
- **LayoutTensor doesn't prevent deadlocks**: Higher-level abstractions still need correct synchronization

** Key Insight**: Barrier deadlocks are among the hardest GPU bugs to debug because:
- **No visible error**- just infinite waiting
- **Requires multi-thread analysis**- can't debug by examining one thread
- **Silent failure mode**- looks like performance issue, not correctness bug
- **Complex thread coordination**- need to trace execution paths across all threads

This type of debugging - using CUDA-GDB to analyze thread states, identify divergent execution paths, and verify barrier reachability - is exactly what professional GPU developers do when facing deadlock issues in production systems.

**You've completed the GPU debugging trilogy!**

#### Your complete GPU debugging arsenal

**From the [First Case](#detective-work-first-case) - Crash debugging:**
-  **Systematic crash investigation**using error messages as guides
-  **Memory bug detection**through pointer address inspection
-  **CUDA-GDB fundamentals**for memory-related issues

**From the [Second Case](#detective-work-second-case) - Logic bug debugging:**
-  **Algorithm error investigation**without obvious symptoms
-  **Pattern analysis techniques**for tracing wrong results to root causes
-  **Execution flow debugging**when variable inspection fails

**From the [Third Case](#detective-work-third-case) - Coordination debugging:**
-  **Barrier deadlock investigation**for thread coordination failures
-  **Multi-thread state analysis**using advanced CUDA-GDB techniques
-  **Synchronization verification**for complex parallel programs

#### The professional GPU debugging methodology

You've learned the systematic approach used by professional GPU developers:

1. **Read the symptoms**- Crashes? Wrong results? Infinite hangs?
2. **Form hypotheses**- Memory issue? Logic error? Coordination problem?
3. **Gather evidence**- Use CUDA-GDB strategically based on the bug type
4. **Test systematically**- Verify each hypothesis through targeted investigation
5. **Trace to root cause**- Follow the evidence chain to the source

**Achievement Unlocked**: You can now debug the three most common GPU programming issues:
- **Memory crashes**([First Case](#detective-work-first-case)) - null pointers, out-of-bounds access
- **Logic bugs**([Second Case](#detective-work-second-case)) - algorithmic errors, incorrect results
- **Coordination deadlocks**([Third Case](#detective-work-third-case)) - barrier synchronization failures
