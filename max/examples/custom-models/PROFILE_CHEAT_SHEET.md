# Profiling Cheat Sheet (SM90/H100, MXFP4 MoE + grouped)

Goal
Prove the pipeline is exactly:
  decode in regs → ONE vector st.shared into final SWIZZLE_128B tile → wgmma_async
Then use profiler evidence to choose the next lever (writer lane-map vs global-load coalescing vs small-M persistence).

0) Build/run prerequisites (so results map to code)
- Build optimized WITH symbols:
  mojo build --debug-level=full <…>   (keep -O3 / default optimizations on)
- Warm-up: run once, profile later iterations (or use ncu launch-skip).
- Don’t set CUDA_LAUNCH_BLOCKING for perf profiling.

1) “Big picture” with NSight Systems (nsys) — find real hotspots + CPU gaps
Run (works even if you launch via pixi, just profile the whole process tree):
  nsys profile --trace=cuda,nvtx --trace-fork-before-exec=true --delay=2 --duration=15 --force-overwrite=true --output=nsys_moe20 <YOUR_BENCH_CMD>
Analyze:
  nsys stats --force-export=true nsys_moe20.nsys-rep

What to look for (decision):
- If big CPU gaps between kernels → fix launch cadence / batching / sync points.
- If memcpy/alloc dominates → reuse buffers, avoid per-iter allocs, overlap transfers.
- If kernels dominate → move to ncu on the top 1–2 kernels.

2) Kernel deep-dive with NSight Compute (ncu) — confirm “single-pass shared store + WGMMA”
Run one kernel at a time (W1 then W2):
  ncu --target-processes all --replay-mode=kernel --clock-control=base --cache-control=all --import-source=on --kernel-name regex:mxfp4_moe_w1_swiglu --launch-skip 10 --launch-count 1 --section SpeedOfLight --section LaunchStats --section Occupancy --section InstructionStats --section WarpStateStats --section MemoryWorkloadAnalysis --section SharedMemory --force-overwrite -o ncu_w1 <YOUR_BENCH_CMD>

Dump readable text:
  ncu --import ncu_w1.ncu-rep --page details > ncu_w1.txt

Hard “evidence checklist” (PASS/FAIL):
A) WGMMA present + dominant
- PASS: InstructionStats shows wgmma_async as a major instruction class.
- FAIL: no wgmma or it’s tiny → we’re not hitting the intended TC path.

B) Exactly one shared-store pass into B tile
- PASS: SharedMemory shows a single strong shared-store stream; minimal shared-load traffic for the same tile.
- FAIL: shared store + another shared store/load phase → we accidentally reformat/stage twice.

C) Vectorized st.shared (not scalar storm)
- PASS: shared store transactions are wide/efficient; low instruction count per element stored.
- FAIL: lots of tiny shared stores → force wider SIMD stores and align tile pointers.

D) No decode spills
- PASS: Occupancy shows sane regs/thread (no big spills); WarpStateStats not dominated by “local memory” or spill-related stalls.
- FAIL: high regs + low occupancy + spill indicators → tighten types (u32/u16 + bitcast), reduce live ranges, split decode/store.

E) Shared bank conflicts low
- PASS: SharedMemory/bank-conflict metrics low.
- FAIL: bank conflicts high → fix lane→address mapping inside the swizzled tile (writer mapping issue).

3) If W1/W2 are good but Small-M collapses: treat it as a launch/occupancy problem
NCU tells you immediately:
- Look at Waves Per SM / Achieved Occupancy / Eligible warps per cycle.
If waves-per-SM << 1 or occupancy low only for small-M:
- Implement a small-M path: persistent CTA over multiple groups OR fuse more work per CTA.
- Don’t waste time micro-optimizing decode in that regime.

4) Grouped ragged is slow? Profile it separately (it’s usually NOT WGMMA)
Run:
  ncu --target-processes all --replay-mode=kernel --clock-control=base --cache-control=all --import-source=on --kernel-name regex:grouped_matmul --launch-skip 10 --launch-count 1 --section SpeedOfLight --section MemoryWorkloadAnalysis --section MemoryCharts --section WarpStateStats --section Occupancy --force-overwrite -o ncu_grouped <YOUR_BENCH_CMD>
Then decide based on evidence:
- If DRAM/L2 throughput is low and “global load efficiency” is poor → ragged gather/indirection is the bottleneck.
  Fix: reorder tokens/expert groups to improve coalescing, vectorize A loads, precompute/pack indices, reduce pointer chasing.
- Cache hit paradox applies: high L2 hit + low bandwidth often means serialized/uncoalesced accesses, not “good caching”.

5) Quick SASS spot-check (only if we need to confirm store width / unexpected extra passes)
Dump and grep:
  cuobjdump --dump-sass <YOUR_BINARY_OR_CUBIN> | rg -n "wgmma|st\\.shared|ld\\.shared|cvt"
What you want to see in the hot loop:
  bit ops + bf16 ops + st.shared.v* + wgmma
What you don’t want:
  a second shared staging loop, lots of cvt chains, or scalar st.shared.

6) Counter permissions (only if ncu errors)
If you hit ERR_NVGPUCTRPERM on the box:
- You need profiling counters enabled by the driver/admin; fix once per machine, then proceed.

Deliverable format for reporting back (so we can act fast)
For each kernel (W1, W2, grouped):
- SpeedOfLight: Compute % vs Memory %
- LaunchStats: waves-per-SM, block size/grid size
- Occupancy: regs/thread, smem/block, achieved occupancy
- SharedMemory: shared store/load throughput + bank conflicts
- InstructionStats: presence/weight of wgmma + st.shared

Interpretation rule:
- If WGMMA + vector st.shared look clean → stop touching decode; attack small-M persistence or ragged global access next.
- If shared stores are scalar or bank-conflicted → fix writer mapping/store width before anything else.
