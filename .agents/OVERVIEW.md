# MXFP4 GPT-OSS Custom arch and kernels

For SM90 we don’t want “generic grouped_matmul_ragged with some dequant glued on the side”, you want **a real Hopper FP4→FP16 GEMM** that behaves like the Triton `matmul_ogs` path and then wire that into a Max custom arch.

Let’s zoom all the way in on **SM90 only** and design something you can realistically implement.

## IMPORTANT: DO NOT TOUCH EXISTING KERNELS, IMPORT THEM IF YOU WANT TO USE THEM AND WRITE NEW FILES IN THE EXAMPLES AREA. WE ARE NOT MODIFYING MODULAR EXISTING CODE

---

## 0. Quick alignment

* Target: **H100 (SM90)**.
* Hardware fact: SM90 has tensor cores for **BF16/FP16/FP8/INT8**, but **no native FP4**. So, just like the OpenAI Triton code, the only sensible play is:

  * Keep weights in **MXFP4 in global**,
  * **Dequantize to BF16 inside the GEMM tile loop**,
  * Feed BF16 fragments into **wgmma**.

* **Decode reality (important):** SM90 **WGMMA is fixed `M=64`** (instruction-level). For MoE decode, each expert segment often has **tiny M** (few tokens), so a WGMMA-only design wastes most work and tanks single-request TPS. Production engines (and Triton-style implementations) avoid this by using a **kernel family**:

  * **Small-M:** HMMA / warp-MMA (e.g. 16×16/16×8 tiles), BF16 inputs, FP32 accum in registers.
  * **Large-M:** WGMMA (M=64), BF16 tiles in shared, FP32 accum in registers.
  * **Deterministic dispatch:** one hard threshold (e.g. `M <= 64 → HMMA`, else WGMMA). No env flags.

We’re going to design that GEMM, with:

* A: BF16 activations.
* B: MXFP4 weights (uint8 blocks + uint8 E8M0 scales).
* C: BF16 output (FP32 accumulation in registers only).
* Fused SWIGLU epilogue for the **MoE MLP1** path.

### Precision policy (hard rule)

* **FP32 only in registers:** accumulator fragments + a few scalar epilogue temporaries (bias/scale/activation math).
* **Everything else:** BF16 (or U8/U32 metadata). **No FP32 shared tiles, no FP32 intermediate tensors.**

Once that kernel exists, everything else (Python arch, loader, etc.) is just plumbing.

---

## 1. SM90 kernel shape: what we want it to do

For each GEMM `C = A @ B`:

* Shapes:

  * `A`: `[M, K]` BF16.
  * `B`: `[K, N]` in **logical** BF16, but **stored as MXFP4 blocks**:

    * K is chunked into groups of 32 values,
    * Each group of 32 is stored as 16 bytes (2 FP4 / byte),
    * Each 32-wide group has a **scale exponent** stored as `uint8` (E8M0).
  * `C`: `[M, N]` BF16.

* Kernel mapping:

  * Each **CTA** computes one `BLOCK_M × BLOCK_N` tile of `C`.
  * It loops over the K dimension in chunks of `BLOCK_K`.
  * For each chunk:

    * Load `A_tile[BLOCK_M, BLOCK_K]` (BF16) into shared.
    * Load **packed B tile** and corresponding **scales** into shared.
    * Each warp:

      * Decodes a **FP4→FP16 B fragment** from these bytes + scales into registers.
      * Loads an A fragment from shared.
      * Runs `wgmma` (BF16 × BF16 → FP32 accum).

* After K loop:

  * Each thread has a **FP32 accumulator tile** in registers.
  * It:

    * Adds bias,
    * Applies SWIGLU in-place (`alpha=1.702`, clamp `limit=7.0`, interleaved layout),
    * Casts FP32 → BF16,
    * Stores to global.

So you get **exactly** what Triton’s `matmul_ogs(…, fused_activation=swiglu)` does, but in Mojo on SM90.

---

## 2. Tile and warp layout (SM90)

Let’s pick a plausible matmul shape that maps nicely to SM90 wgmma and isn’t insane for a first version. You can tune later.

### 2.1 CTA-level tiling

Start with something like:

* `BLOCK_M = 128`
* `BLOCK_N = 128`
* `BLOCK_K = 64`

So one CTA computes a `128×128` tile of `C` and marches through K in `64`-wide slices.

### 2.2 Warp mapping inside CTA

Example mapping (simple and works):

* `num_warps_per_CTA = 8`
* `warp_size = 32`
* `threads_per_block = 256`

Warp responsibilities:

* Warp (0,1): can handle A tile loads, some of B tile loads.
* Warp (2,3): can handle B tile loads / decode.
* Warp (4–7): primarily compute (but in practice all warps will share compute).

For each `BLOCK_K` slice, we’ll split it into **smaller fragment-K** chunks that match the wgmma K dimension, like `FRAG_K = 16` or `32`. So:

* `BLOCK_K = 64` → 2–4 `FRAG_K` steps per K-slice.

This is very similar to what they do in `max/kernels/src/linalg/matmul/gpu/sm90/matmul_kernels.mojo` (you’ll mirror that structure, not guess blindly).

---

## 3. Staging A and B on SM90

### 3.1 A: BF16 activations

* Global layout: `[M, K]`, row-major.
* For `BLOCK_M=128, BLOCK_K=64`, the A tile for one CTA is a contiguous-ish chunk.

Per K-slice (`k0..k0+BLOCK_K`):

1. Each warp cooperates to:

   * Load stripes of `A` from global into `smem_A[BLOCK_M, BLOCK_K]`.

2. The load pattern should match whatever SM90 tensor-core loader wants, but for a first version you can:

   * Load in row-major into shared,
   * Use the same “A tile loader” they use in `test_matmul_sm90_bf16.mojo` to convert from row-major shared → wgmma fragment.

### 3.2 B: MXFP4 weights

Logical shape: `[K, N]` BF16.

Stored as two arrays:

* `B_blocks`: packed int8 / uint8, something like:

  ```text
  [K_groups, N]   with K_groups = K / 32, BYTES_PER_GROUP = 16
  ```

* `B_scales`: e.g.

  ```text
  [K_groups, N_scale_groups] or [K_groups, N] depending on your grouping
  MXFP4_SF_DTYPE = float8_e8m0fnu
  ```

For the kernel we care about:

* Given `(k, n)`, we can compute:

  * Which 32‑group along K: `k_group = k // 32`,
  * Index inside the group: `k_in_group = k % 32`,
  * Which byte in the group: `byte_idx_in_group = k_in_group // 2`,
  * Low or high nibble: `lo_or_hi = k_in_group % 2`,
  * Where the scale lives: `scale = _get_scale_factor_mxfp4(B_scales, k, n)`.

Per `BLOCK_K` slice:

1. We load **B_blocks tile** corresponding to `[k0..k0+BLOCK_K, col_start..col_start+BLOCK_N]` into `smem_B_packed`.

2. We load **B_scales tile** for that same region into `smem_B_scales`.

Important: we’re **not** going to dequant B to BF16 in shared and then run a “normal” BF16 matmul; that would be too memory-heavy. Instead:

* `smem_B_packed` and `smem_B_scales` stay packed,
* Each warp decodes just the bytes it needs into its **register fragment** as part of the GEMM inner loop.

Exactly what your NF4 kernel does, but the target is now a `wgmma` fragment instead of a simple `[2]` half pair.

---

## 4. Warp-level decode + MMA (the core)

Let’s drill into one **warp** inside one `BLOCK_K` slice.

Assume:

* Each warp computes a sub-tile of `C` of shape `WR_M × WR_N` (e.g. 64×64 or 32×64).
* For each `kk` in `0..BLOCK_K step FRAG_K`:

  We want to:

  1. Load `A` fragment: `FRAG_M × FRAG_K` (BF16).
  2. Decode `B` fragment: `FRAG_K × FRAG_N` (FP4→FP16).
  3. Run `wgmma` on these fragments.

### 4.1 Per-lane mapping

You can reuse the same conceptual pattern as your NF4 warp kernel:

* In that kernel:

  * 1 warp handled 1 NF4 block (32 bytes = 64 weights),
  * Each lane handled 1 byte = 2 weights.

Here, for a warp’s **B fragment**:

* Suppose `FRAG_K = 32, FRAG_N = 16` (just as an example).
* That’s `FRAG_K × FRAG_N = 512` FP4 values → 256 bytes.

You can map it as:

* Each warp handles these 256 bytes.
* Each lane (0–31) is responsible for `256 / 32 = 8` bytes = 16 FP4 values.

So in decode:

```mojo
var lane_id = Int(thread_idx.x & 31)
# for this warp's fragment (kk, n_range), compute the 8 byte indices each lane owns

for b in range(0, 8):
    var global_k = ...  # determined by kk + some offset
    var global_n = ...  # determined by warp_n + some offset
    var packed_byte: U8 = smem_B_packed[smem_index_for(global_k, global_n)]
    var scale: MXFP4_SF_DTYPE = smem_B_scales[scale_index_for(global_k, global_n)]

    var lo = packed_byte & 0x0F
    var hi = packed_byte >> 4

    var v_lo_f32 = MXFP4_LUT[Int(lo)]
    var v_hi_f32 = MXFP4_LUT[Int(hi)]

    # convert scale to float32
    var s = scale.cast[DType.float32]()

    var v0 = v_lo_f32 * s
    var v1 = v_hi_f32 * s

    # cast to fp16 and assign into correct position in B fragment
    assign_into_b_frag(b_frag, lane_id, b, v0, v1)
```

The **only annoying part** is `assign_into_b_frag(...)`: you need to follow the unpack order that SM90 wgmma expects. That’s where the existing Max tests help.

### 4.2 Using Max’s SM90 fragment helpers

In your tree you have:

* `max/kernels/src/test/gpu/layout/test_wgmma_layouts.mojo`
* `max/kernels/src/linalg/matmul/gpu/sm90/tile_loader.mojo`
* `max/kernels/src/linalg/matmul/gpu/sm90/matmul_kernels.mojo`

The pattern is usually:

* There is some `FragmentA`, `FragmentB`, `FragmentC` types with methods like:

  ```mojo
  alias FragB = sm90_mma.BFragment[FRAG_M, FRAG_N, FRAG_K, ...]
  var b_frag = FragB()

  b_frag.load_from_smem(smem, base_row, base_col)
  ```

* Instead of `load_from_smem`, you can implement:

  ```mojo
  fn load_b_frag_from_mxfp4(
      frag: inout FragB,
      smem_packed: RawPointer[U8],
      smem_scales: RawPointer[MXFP4_SF_DTYPE],
      global_k0: Int,
      global_n0: Int,
      ...
  ):
      # warp-cooperative, each lane decodes some bytes and writes into frag
  ```

So inside the SM90 matmul micro-kernel, when they currently do:

```mojo
a_frag.load_from_smem(...)
b_frag.load_from_smem(...)
acc = sm90_mma.mma_sync(acc, a_frag, b_frag)
```

You create a **new path**:

```mojo
a_frag.load_from_smem(...)
load_b_frag_from_mxfp4(b_frag, smem_B_packed, smem_B_scales, ...)
acc = sm90_mma.mma_sync(acc, a_frag, b_frag)
```

That lets you reuse their entire tile scheduler, ring buffers, TMA prefetch, etc., and only customize the B load.

---

## 5. SWIGLU epilogue in the SM90 kernel

We’re only fusing SWIGLU into the **MLP1** GEMM (`[M, 2D]` output). So:

* Each thread has an accumulator tile `C_frag` in registers, FP32.
* We know each output row is `[..., glu, lin, glu, lin, ...]` interleaved.

Inside the epilogue:

```mojo
fn swiglu_epilogue_and_store[
    c_type: DType,
    accum_type: DType,
    FRAG_M: Int,
    FRAG_N: Int,
](
    C: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    acc: AccumulatorFragment[accum_type, FRAG_M, FRAG_N],
    bias: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    row_start: Int,
    col_start: Int,
    M: Int,
    N: Int,
    alpha: Float32,
    limit: Float32,
):
    # per-thread: operate on its fragment only
    @parameter
    for mi in range(FRAG_M):
        var global_m = row_start + warp_row_offset + mi
        if global_m >= M:
            continue

        @parameter
        for nj in range(0, FRAG_N, 2):
            var global_n0 = col_start + warp_col_offset + nj
            var global_n1 = global_n0 + 1
            if global_n1 >= N:
                continue

            var x_glu = acc[mi, nj]      # FP32
            var x_lin = acc[mi, nj + 1]  # FP32

            # add bias (broadcasted per col)
            x_glu += bias[0, global_n0].cast[accum_type]()
            x_lin += bias[0, global_n1].cast[accum_type]()

            # clamp
            if x_glu > limit:
                x_glu = limit
            if x_lin > limit:
                x_lin = limit
            if x_lin < -limit:
                x_lin = -limit

            # GLU
            var glu = x_glu * sigmoid(alpha * x_glu)

            var out = glu * (x_lin + 1.0)

            # cast to BF16 and store
            var out_bf16 = Scalar[DType.bfloat16](out)
            C[global_m, global_n0] = out_bf16
            # Optionally store something to global_n1 or discard, depending on layout
```

You’ll adapt this to whatever **fragment type** and storage layout they’re using; conceptually it’s exactly the Triton `swiglu` code, just operating directly on register tiles.

---

## 6. Concrete implementation plan for SM90

Let’s turn this into a to-do list in your Max tree.

### 6.1 Mojo side (kernels)

## **(1) MXFP4 decode utilities**

Create or extend something like `max/kernels/src/linalg/fp4_utils.mojo`:

* Add MXFP4-specific constants:

  ```mojo
  comptime MXFP4_LUT = SIMD[DType.float32, 16](...)  # your FP4 values
  comptime MXFP4_SF_VECTOR_SIZE = 32
  comptime MXFP4_SF_DTYPE = DType.float8_e8m0fnu
  ```

* Implement:

  ```mojo
  fn mxfp4_decode_16bytes_to_fp16x32(
      bytes: SIMD[DType.uint8, 16],
      scale: Scalar[MXFP4_SF_DTYPE],
  ) -> SIMD[DType.float16, 32]:
      # (very similar to your NF4 SIMD logic)
  ```

* Implement `_get_scale_factor_mxfp4` just like `_get_scale_factor` for NVFP4, but using `MXFP4_SF_VECTOR_SIZE = 32`.

## **(2) Warp-level B fragment loader for SM90**

In something like `max/kernels/src/linalg/matmul/gpu/sm90/blockwise_mxfp4.mojo`:

* Define a helper:

  ```mojo
  fn load_b_frag_from_mxfp4[
      b_scales_type: DType,
      SF_VECTOR_SIZE: Int = MXFP4_SF_VECTOR_SIZE,
  ](
      b_frag: inout FragB,
      smem_B_packed: RawPointer[DType.uint8],
      smem_B_scales: RawPointer[b_scales_type],
      k_base: Int,
      n_base: Int,
      K: Int,
      N: Int,
  ):
      var lane_id = Int(thread_idx.x & 31)
      # compute which bytes in smem this lane is responsible for,
      # decode them with mxfp4_decode_*,
      # and write into b_frag in the order expected by FragB
  ```

Look at:

* `test_wgmma_layouts.mojo`
* `sm90/matmul_kernels.mojo` and `sm90/tile_loader.mojo`

to see how they normally fill `FragB` for BF16/FP8 and mirror that pattern but with FP4 decode.

## **(3) MXFP4 matmul micro-kernel**

Create a matmul kernel like:

```mojo
fn mxfp4_matmul_sm90_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,           # uint8
    b_scales_type: DType,    # MXFP4_SF_DTYPE
    ...
](
    C: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    A: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    B_packed: LayoutTensor[b_type, b_layout, MutAnyOrigin],
    B_scales: LayoutTensor[b_scales_type, b_scales_layout, MutAnyOrigin],
    bias: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    alpha: Scalar[DType.float32],
    limit: Scalar[DType.float32],
):
    # 1. Determine block_m, block_n from block_idx
    # 2. Loop over K in BLOCK_K steps:
    #    - TMA/ldg: move A_tile and B_packed + scales to smem
    #    - barrier
    #    - inner loop over kk: load a_frag, call load_b_frag_from_mxfp4, wgmma
    # 3. Epilogue: swiglu_epilogue_and_store(...)
```

You can base this directly off `test_matmul_sm90_bf16.mojo` / `matmul_sm90_bf16.mojo`. The only big difference: B fragment load uses your MXFP4 decode helper.

## **(4) Register kernel with `@compiler.register`**

In `max/kernels/src/register/register.mojo`:

```mojo
@compiler.register("mo.mxfp4.matmul.sm90")
struct StructMxfp4MatmulSm90:
    @always_inline
    @staticmethod
    fn execute[
        c_type: DType,
        a_type: DType,
        b_type: DType,
        b_scales_type: DType,
        ...
    ](
        C: OutputTensor[dtype=c_type, rank=2],
        A: InputTensor[dtype=a_type, rank=2],
        B_packed: InputTensor[dtype=b_type, rank=2],
        B_scales: InputTensor[dtype=b_scales_type, rank=2],
        bias: InputTensor[dtype=c_type, rank=1],
        alpha: Float32,
        limit: Float32,
        context: DeviceContextPtr,
    ) raises:
        mxfp4_matmul_sm90_kernel(...)
```

So on the Python side you can call it via `ops.custom("mo.mxfp4.matmul.sm90", ...)`.

---

### 6.2 Python side (MAX / architecture)

Now, build the GPT‑OSS‑style arch around this.

## **(1) Weight loader for MXFP4 MoE weights**

In your custom arch folder (say `max/python/max/nn/architectures/gpt_oss_mxfp4/weights.py`):

* Similar to the OpenAI `Checkpoint._get_mxfp4_tensor`, but:

  * Instead of fully dequantizing to BF16,
  * You load *exactly* the packed blocks and scales into tensors:

  ```python
  class GptOssMxfp4Checkpoint:
      def __init__(self, path: str, device: DeviceRef):
          ...

      def get_moe_weights(self, name: str):
          blocks = self._get_tensor(name + ".blocks")  # uint8
          scales = self._get_tensor(name + ".scales")  # float32 / float8
          # maybe cast to MXFP4_SF_DTYPE
          return blocks, scales
  ```

* Arrange shapes to match what your kernel expects: e.g. `[K_groups, N]` for blocks and `[K_groups, scale_cols]` for scales.

## **(2) Python wrapper op for MXFP4 matmul**

In `max/python/max/nn/kernels.py` (or a local helper file):

```python
from max.graph import ops, TensorType, TensorValue

def mxfp4_matmul_sm90(
    x: TensorValue,           # [M, K] bf16
    w_blocks: TensorValue,    # packed mx4
    w_scales: TensorValue,    # scales
    bias: TensorValue,        # [N] bf16
    alpha: float = 1.702,
    limit: float = 7.0,
) -> TensorValue:
    M, K = x.shape
    # infer N from blocks
    N = ...
    out = ops.custom(
        "mo.mxfp4.matmul.sm90",
        device=x.device,
        values=[x, w_blocks, w_scales, bias, alpha, limit],
        out_types=[
            TensorType(
                dtype=x.dtype,
                shape=[M, N],
                device=x.device,
            )
        ],
    )[0].tensor
    return out
```

## **(3) MoE MLP1 using MXFP4 matmul**

In your MoE layer file (e.g. `gpt_oss_mxfp4/moe.py`):

* For the MoE experts:

  ```python
  class GptOssMxfp4MoE(...):
      def __init__(self, config: GptOssConfig):
          ...
          self.mlp1_blocks, self.mlp1_scales = ...  # from checkpoint
          self.mlp1_bias = Weight(...)
          self.mlp2_blocks, self.mlp2_scales = ...
          self.mlp2_bias = Weight(...)
      ...
      def _mlp1(self, x: TensorValue) -> TensorValue:
          # x is [tokens, hidden]
          return mxfp4_matmul_sm90(
              x,
              self.mlp1_blocks,
              self.mlp1_scales,
              self.mlp1_bias,
              alpha=self.alpha,
              limit=self.limit,
          )
  ```
