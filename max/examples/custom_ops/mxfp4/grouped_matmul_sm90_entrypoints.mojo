import compiler

from .grouped_matmul_sm90_common import *
from .grouped_matmul_sm90_wgmma_swload import (
    grouped_matmul_mxfp4_bf16_wgmma_sm90_pipeline_swload,
)
from .grouped_matmul_sm90_wgmma_swload_transpose import (
    grouped_matmul_mxfp4_bf16_wgmma_sm90_pipeline_swload_transpose,
)


@compiler.register("mxfp4_grouped_matmul_ragged_bf16")
struct MXFP4GroupedMatmulRaggedBF16:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        c: OutputTensor[dtype=BF16, rank=2],
        a: InputTensor[dtype=BF16, rank=2],
        w_blocks: InputTensor[dtype=U8, rank=4],
        w_scales: InputTensor[dtype=U8, rank=3],
        expert_start_indices: InputTensor[dtype=U32, rank=1],
        expert_ids: InputTensor[dtype=I32, rank=1],
        max_num_tokens_per_expert: UInt32,
        num_active_experts: UInt32,
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if is_cpu[target]():
            raise Error("mxfp4_grouped_matmul_ragged_bf16: GPU only")

        var P = a.dim_size(0)
        var K = a.dim_size(1)
        var N = c.dim_size(1)
        var Kblocks = w_blocks.dim_size(1)
        var a_stride0 = K
        var a_stride1 = 1
        var out_stride0 = N
        var out_stride1 = 1
        var w_blocks_stride0 = Kblocks * N * BYTES_PER_BLOCK
        var w_blocks_stride1 = N * BYTES_PER_BLOCK
        var w_blocks_stride2 = BYTES_PER_BLOCK
        var w_blocks_stride3 = 1
        var scales_m2 = w_scales.dim_size(1)
        var scales_k = w_scales.dim_size(2)
        var w_scales_stride0 = scales_m2 * scales_k
        var w_scales_stride1 = scales_k
        var w_scales_stride2 = 1

        if P == 0 or K == 0 or N == 0:
            return
        # NOTE: This kernel expects packed MXFP4 layouts and contiguous
        # row-major tensors; stride validation is omitted for speed.

        if Kblocks * VALUES_PER_BLOCK != K:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16: K must be divisible by 32"
                " and match w_blocks Kblocks"
            )
        if w_blocks_stride3 != 1:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16: w_blocks last dimension"
                " must be contiguous (stride 1)"
            )
        if (Kblocks % HOPPER_SCALE_ALIGN_K) != 0:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16: K must be divisible by 64"
                " for Hopper scale swizzle"
            )
        if scales_k != K:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16: w_scales K dimension"
                " must match K"
            )
        if scales_m2 * 32 < N:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16: w_scales dim1 must"
                " cover N (>= N/32 with padding)"
            )
        if (scales_m2 % HOPPER_SCALE_NUM_WARPS) != 0:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16: w_scales dim1 must be"
                " a multiple of Hopper num_warps"
            )

        var gpu_ctx = ctx.get_device_context()
        var c_dev = DeviceBuffer[c.dtype](
            gpu_ctx, c.unsafe_ptr(), c.size(), owning=False
        )
        var w_blocks_dev = DeviceBuffer[w_blocks.dtype](
            gpu_ctx, w_blocks.unsafe_ptr(), w_blocks.size(), owning=False
        )
        var w_scales_dev = DeviceBuffer[w_scales.dtype](
            gpu_ctx, w_scales.unsafe_ptr(), w_scales.size(), owning=False
        )
        var expert_start_dev = DeviceBuffer[expert_start_indices.dtype](
            gpu_ctx,
            expert_start_indices.unsafe_ptr(),
            expert_start_indices.size(),
            owning=False,
        )
        var expert_ids_dev = DeviceBuffer[expert_ids.dtype](
            gpu_ctx, expert_ids.unsafe_ptr(), expert_ids.size(), owning=False
        )

        # IMPORTANT: `mo.moe.create.indices` compacts `expert_start_indices` and
        # `expert_ids` so only the first `num_active_experts` entries are valid.
        # Reading beyond that yields undefined values and can cause OOB reads on
        # expert weights (NaNs -> invalid router indices -> gather OOB).
        var max_M = Int(max_num_tokens_per_expert)
        var n_active = Int(num_active_experts)
        if max_M <= 0 or n_active <= 0:
            return
        if max_M > P:
            max_M = P

        var grid_z = n_active
        var max_experts = expert_ids.dim_size(0)
        var start_len = expert_start_indices.dim_size(0)
        if start_len <= 1:
            return
        var start_max = start_len - 1
        if max_experts > start_max:
            max_experts = start_max
        if grid_z > max_experts:
            grid_z = max_experts
        if grid_z <= 0:
            return

        # SM90 WGMMA pipeline kernels (warp-group, BF16 accum).
        # Use a transpose trick for very small M to avoid wasting WGMMA M=64.
        comptime SMALL_M_TRANSPOSE_THRESHOLD = 64

        # w_blocks: [E, K/32, N, 16] row-major contiguous
        # w_scales: [E, N/32 (padded), K] Hopper scale swizzled

        var a_dev = DeviceBuffer[a.dtype](
            gpu_ctx, a.unsafe_ptr(), a.size(), owning=False
        )

        if max_M <= SMALL_M_TRANSPOSE_THRESHOLD:
            # Transpose path: treat N as WGMMA-M and M as WGMMA-N.
            comptime BM = 256  # N-tile
            comptime BN = 128  # M-tile (keep >= 128 for SM90 WGMMA)
            comptime BK = 128
            comptime NUM_WARP_GROUPS = 1
            comptime NUM_PIPELINE_STAGES = 2

            var grid_x = ceildiv(N, BM)
            var grid_y = ceildiv(max_M, BN)
            if grid_x == 0 or grid_y == 0:
                return

            comptime kernel = (
                grouped_matmul_mxfp4_bf16_wgmma_sm90_pipeline_swload_transpose[
                    BM=BM,
                    BN=BN,
                    BK=BK,
                    WGMMA_M=64,
                    WGMMA_N=64,
                    WGMMA_K=16,
                    NUM_WARP_GROUPS=NUM_WARP_GROUPS,
                    NUM_PIPELINE_STAGES=NUM_PIPELINE_STAGES,
                ]
            )
            comptime b_smem_layout = tile_layout_k_major[
                BF16, BN, BK, TensorMapSwizzle.SWIZZLE_128B
            ]()
            comptime b_bytes = b_smem_layout.size() * 2
            comptime a_stage_bytes = 0
            comptime b_stage_bytes = ((b_bytes + 255) // 256) * 256
            comptime stage_bytes = a_stage_bytes + b_stage_bytes
            comptime smem_use = stage_bytes * NUM_PIPELINE_STAGES

            gpu_ctx.enqueue_function_experimental[kernel](
                a_dev,
                a_stride0,
                a_stride1,
                P,
                K,
                expert_start_dev,
                expert_ids_dev,
                grid_z,
                w_blocks_dev,
                w_blocks_stride0,
                w_blocks_stride1,
                w_blocks_stride2,
                w_scales_dev,
                w_scales_stride0,
                w_scales_stride1,
                w_scales_stride2,
                Kblocks,
                c_dev,
                out_stride0,
                out_stride1,
                N,
                grid_dim=(grid_x, grid_y, grid_z),
                block_dim=(WARPGROUP_SIZE * (NUM_WARP_GROUPS + 1), 1, 1),
                shared_mem_bytes=Int(smem_use),
            )
        else:
            # RS path for larger M: decode weights into regs, activations in shared.
            comptime BM = 256  # N-tile
            comptime BN = 128  # M-tile
            comptime BK = 128
            comptime NUM_WARP_GROUPS = 1
            comptime NUM_PIPELINE_STAGES = 2
            var grid_x = ceildiv(N, BM)
            var grid_y = ceildiv(max_M, BN)
            if grid_x == 0 or grid_y == 0:
                return

            comptime kernel = (
                grouped_matmul_mxfp4_bf16_wgmma_sm90_pipeline_swload_transpose[
                    BM=BM,
                    BN=BN,
                    BK=BK,
                    WGMMA_M=64,
                    WGMMA_N=64,
                    WGMMA_K=16,
                    NUM_WARP_GROUPS=NUM_WARP_GROUPS,
                    NUM_PIPELINE_STAGES=NUM_PIPELINE_STAGES,
                ]
            )
            comptime b_smem_layout = tile_layout_k_major[
                BF16, BN, BK, TensorMapSwizzle.SWIZZLE_128B
            ]()
            comptime b_bytes = b_smem_layout.size() * 2
            comptime a_stage_bytes = 0
            comptime b_stage_bytes = ((b_bytes + 255) // 256) * 256
            comptime stage_bytes = a_stage_bytes + b_stage_bytes
            comptime smem_use = stage_bytes * NUM_PIPELINE_STAGES

            gpu_ctx.enqueue_function_experimental[kernel](
                a_dev,
                a_stride0,
                a_stride1,
                P,
                K,
                expert_start_dev,
                expert_ids_dev,
                grid_z,
                w_blocks_dev,
                w_blocks_stride0,
                w_blocks_stride1,
                w_blocks_stride2,
                w_scales_dev,
                w_scales_stride0,
                w_scales_stride1,
                w_scales_stride2,
                Kblocks,
                c_dev,
                out_stride0,
                out_stride1,
                N,
                grid_dim=(grid_x, grid_y, grid_z),
                block_dim=(WARPGROUP_SIZE * (NUM_WARP_GROUPS + 1), 1, 1),
                shared_mem_bytes=Int(smem_use),
            )


@compiler.register("mxfp4_grouped_matmul_ragged_bf16_swizzled")
struct MXFP4GroupedMatmulRaggedBF16Swizzled:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        c: OutputTensor[dtype=BF16, rank=2],
        a: InputTensor[dtype=BF16, rank=2],
        w_blocks: InputTensor[dtype=U8, rank=3],
        w_scales: InputTensor[dtype=U8, rank=3],
        expert_start_indices: InputTensor[dtype=U32, rank=1],
        expert_ids: InputTensor[dtype=I32, rank=1],
        max_num_tokens_per_expert: UInt32,
        num_active_experts: UInt32,
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if is_cpu[target]():
            raise Error("mxfp4_grouped_matmul_ragged_bf16_swizzled: GPU only")

        var P = a.dim_size(0)
        var K = a.dim_size(1)
        var N = c.dim_size(1)
        var Kblocks = K // VALUES_PER_BLOCK
        var kbytes = K // 2
        var a_stride0 = K
        var a_stride1 = 1
        var out_stride0 = N
        var out_stride1 = 1
        var w_m2 = w_blocks.dim_size(1)
        var w_k2 = w_blocks.dim_size(2)
        var w_blocks_stride0 = w_m2 * w_k2
        var w_blocks_stride1 = w_k2
        var w_blocks_stride2 = 1
        var scales_m2 = w_scales.dim_size(1)
        var scales_k = w_scales.dim_size(2)
        var w_scales_stride0 = scales_m2 * scales_k
        var w_scales_stride1 = scales_k
        var w_scales_stride2 = 1

        if P == 0 or K == 0 or N == 0:
            return
        # NOTE: This kernel expects swizzled Hopper value layout for w_blocks
        # and swizzled Hopper scale layout for w_scales.

        if Kblocks * VALUES_PER_BLOCK != K:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16_swizzled: K must be divisible"
                " by 32"
            )
        if (Kblocks % HOPPER_SCALE_ALIGN_K) != 0:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16_swizzled: K must be divisible"
                " by 64 for Hopper swizzles"
            )
        if w_m2 * 4 < N:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16_swizzled: w_blocks dim1 must"
                " cover N (>= N/4 with padding)"
            )
        if w_k2 < kbytes * 4:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16_swizzled: w_blocks dim2 must"
                " cover K bytes (>= K/2 * 4)"
            )
        if scales_k != K:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16_swizzled: w_scales K"
                " dimension must match K"
            )
        if scales_m2 * 32 < N:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16_swizzled: w_scales dim1 must"
                " cover N (>= N/32 with padding)"
            )
        if (scales_m2 % HOPPER_SCALE_NUM_WARPS) != 0:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16_swizzled: w_scales dim1 must"
                " be a multiple of Hopper num_warps"
            )

        var gpu_ctx = ctx.get_device_context()
        var c_dev = DeviceBuffer[c.dtype](
            gpu_ctx, c.unsafe_ptr(), c.size(), owning=False
        )
        var w_blocks_dev = DeviceBuffer[w_blocks.dtype](
            gpu_ctx, w_blocks.unsafe_ptr(), w_blocks.size(), owning=False
        )
        var w_scales_dev = DeviceBuffer[w_scales.dtype](
            gpu_ctx, w_scales.unsafe_ptr(), w_scales.size(), owning=False
        )
        var expert_start_dev = DeviceBuffer[expert_start_indices.dtype](
            gpu_ctx,
            expert_start_indices.unsafe_ptr(),
            expert_start_indices.size(),
            owning=False,
        )
        var expert_ids_dev = DeviceBuffer[expert_ids.dtype](
            gpu_ctx, expert_ids.unsafe_ptr(), expert_ids.size(), owning=False
        )

        var max_M = Int(max_num_tokens_per_expert)
        var n_active = Int(num_active_experts)
        if max_M <= 0 or n_active <= 0:
            return
        if max_M > P:
            max_M = P

        var grid_z = n_active
        var max_experts = expert_ids.dim_size(0)
        var start_len = expert_start_indices.dim_size(0)
        if start_len <= 1:
            return
        var start_max = start_len - 1
        if max_experts > start_max:
            max_experts = start_max
        if grid_z > max_experts:
            grid_z = max_experts
        if grid_z <= 0:
            return

        # SM90 WGMMA pipeline kernels (warp-group, BF16 accum).
        # Keep the swizzled path on a single RS kernel variant for now.
        comptime SMALL_M_TRANSPOSE_THRESHOLD = 0

        # w_blocks: [E, N/4 (padded), Kbytes*4] Hopper value swizzled
        # w_scales: [E, N/32 (padded), K] Hopper scale swizzled

        var a_dev = DeviceBuffer[a.dtype](
            gpu_ctx, a.unsafe_ptr(), a.size(), owning=False
        )

        if max_M <= SMALL_M_TRANSPOSE_THRESHOLD:
            comptime BM = 256  # N-tile
            comptime BN = 128  # M-tile (keep >= 128 for SM90 WGMMA)
            comptime BK = 128
            comptime NUM_WARP_GROUPS = 1
            comptime NUM_PIPELINE_STAGES = 2

            var grid_x = ceildiv(N, BM)
            var grid_y = ceildiv(max_M, BN)
            if grid_x == 0 or grid_y == 0:
                return

            comptime kernel = (
                grouped_matmul_mxfp4_bf16_wgmma_sm90_pipeline_swload_transpose[
                    BM=BM,
                    BN=BN,
                    BK=BK,
                    WGMMA_M=64,
                    WGMMA_N=64,
                    WGMMA_K=16,
                    NUM_WARP_GROUPS=NUM_WARP_GROUPS,
                    NUM_PIPELINE_STAGES=NUM_PIPELINE_STAGES,
                    USE_VALUE_SWIZZLE=True,
                ]
            )
            comptime b_smem_layout = tile_layout_k_major[
                BF16, BN, BK, TensorMapSwizzle.SWIZZLE_128B
            ]()
            comptime b_bytes = b_smem_layout.size() * 2
            comptime a_stage_bytes = 0
            comptime b_stage_bytes = ((b_bytes + 255) // 256) * 256
            comptime stage_bytes = b_stage_bytes
            comptime smem_use = stage_bytes * NUM_PIPELINE_STAGES

            gpu_ctx.enqueue_function_experimental[kernel](
                a_dev,
                a_stride0,
                a_stride1,
                P,
                K,
                expert_start_dev,
                expert_ids_dev,
                grid_z,
                w_blocks_dev,
                w_blocks_stride0,
                w_blocks_stride1,
                w_blocks_stride2,
                w_scales_dev,
                w_scales_stride0,
                w_scales_stride1,
                w_scales_stride2,
                Kblocks,
                c_dev,
                out_stride0,
                out_stride1,
                N,
                grid_dim=(grid_x, grid_y, grid_z),
                block_dim=(WARPGROUP_SIZE * (NUM_WARP_GROUPS + 1), 1, 1),
                shared_mem_bytes=Int(smem_use),
            )
        else:
            # RS path for larger M: decode weights into regs, activations in shared.
            comptime BM = 256  # N-tile
            comptime BN = 128  # M-tile
            comptime BK = 128
            comptime NUM_WARP_GROUPS = 1
            comptime NUM_PIPELINE_STAGES = 2
            var grid_x = ceildiv(N, BM)
            var grid_y = ceildiv(max_M, BN)
            if grid_x == 0 or grid_y == 0:
                return

            comptime kernel = (
                grouped_matmul_mxfp4_bf16_wgmma_sm90_pipeline_swload_transpose[
                    BM=BM,
                    BN=BN,
                    BK=BK,
                    WGMMA_M=64,
                    WGMMA_N=64,
                    WGMMA_K=16,
                    NUM_WARP_GROUPS=NUM_WARP_GROUPS,
                    NUM_PIPELINE_STAGES=NUM_PIPELINE_STAGES,
                    USE_VALUE_SWIZZLE=True,
                ]
            )
            comptime b_smem_layout = tile_layout_k_major[
                BF16, BN, BK, TensorMapSwizzle.SWIZZLE_128B
            ]()
            comptime b_bytes = b_smem_layout.size() * 2
            comptime a_stage_bytes = 0
            comptime b_stage_bytes = ((b_bytes + 255) // 256) * 256
            comptime stage_bytes = b_stage_bytes
            comptime smem_use = stage_bytes * NUM_PIPELINE_STAGES

            gpu_ctx.enqueue_function_experimental[kernel](
                a_dev,
                a_stride0,
                a_stride1,
                P,
                K,
                expert_start_dev,
                expert_ids_dev,
                grid_z,
                w_blocks_dev,
                w_blocks_stride0,
                w_blocks_stride1,
                w_blocks_stride2,
                w_scales_dev,
                w_scales_stride0,
                w_scales_stride1,
                w_scales_stride2,
                Kblocks,
                c_dev,
                out_stride0,
                out_stride1,
                N,
                grid_dim=(grid_x, grid_y, grid_z),
                block_dim=(WARPGROUP_SIZE * (NUM_WARP_GROUPS + 1), 1, 1),
                shared_mem_bytes=Int(smem_use),
            )


@compiler.register("mxfp4_grouped_matmul_ragged_bf16_swizzled_no_small_m")
struct MXFP4GroupedMatmulRaggedBF16SwizzledNoSmallM:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        c: OutputTensor[dtype=BF16, rank=2],
        a: InputTensor[dtype=BF16, rank=2],
        w_blocks: InputTensor[dtype=U8, rank=3],
        w_scales: InputTensor[dtype=U8, rank=3],
        expert_start_indices: InputTensor[dtype=U32, rank=1],
        expert_ids: InputTensor[dtype=I32, rank=1],
        max_num_tokens_per_expert: UInt32,
        num_active_experts: UInt32,
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if is_cpu[target]():
            raise Error("mxfp4_grouped_matmul_ragged_bf16_swizzled: GPU only")

        var P = a.dim_size(0)
        var K = a.dim_size(1)
        var N = c.dim_size(1)
        var Kblocks = K // VALUES_PER_BLOCK
        var kbytes = K // 2
        var a_stride0 = K
        var a_stride1 = 1
        var out_stride0 = N
        var out_stride1 = 1
        var w_m2 = w_blocks.dim_size(1)
        var w_k2 = w_blocks.dim_size(2)
        var w_blocks_stride0 = w_m2 * w_k2
        var w_blocks_stride1 = w_k2
        var w_blocks_stride2 = 1
        var scales_m2 = w_scales.dim_size(1)
        var scales_k = w_scales.dim_size(2)
        var w_scales_stride0 = scales_m2 * scales_k
        var w_scales_stride1 = scales_k
        var w_scales_stride2 = 1

        if P == 0 or K == 0 or N == 0:
            return
        # NOTE: This kernel expects swizzled Hopper value layout for w_blocks
        # and swizzled Hopper scale layout for w_scales.

        if Kblocks * VALUES_PER_BLOCK != K:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16_swizzled: K must be divisible"
                " by 32"
            )
        if (Kblocks % HOPPER_SCALE_ALIGN_K) != 0:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16_swizzled: K must be divisible"
                " by 64 for Hopper swizzles"
            )
        if w_m2 * 4 < N:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16_swizzled: w_blocks dim1 must"
                " cover N (>= N/4 with padding)"
            )
        if w_k2 < kbytes * 4:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16_swizzled: w_blocks dim2 must"
                " cover K bytes (>= K/2 * 4)"
            )
        if scales_k != K:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16_swizzled: w_scales K"
                " dimension must match K"
            )
        if scales_m2 * 32 < N:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16_swizzled: w_scales dim1 must"
                " cover N (>= N/32 with padding)"
            )
        if (scales_m2 % HOPPER_SCALE_NUM_WARPS) != 0:
            raise Error(
                "mxfp4_grouped_matmul_ragged_bf16_swizzled: w_scales dim1 must"
                " be a multiple of Hopper num_warps"
            )

        var gpu_ctx = ctx.get_device_context()
        var c_dev = DeviceBuffer[c.dtype](
            gpu_ctx, c.unsafe_ptr(), c.size(), owning=False
        )
        var w_blocks_dev = DeviceBuffer[w_blocks.dtype](
            gpu_ctx, w_blocks.unsafe_ptr(), w_blocks.size(), owning=False
        )
        var w_scales_dev = DeviceBuffer[w_scales.dtype](
            gpu_ctx, w_scales.unsafe_ptr(), w_scales.size(), owning=False
        )
        var expert_start_dev = DeviceBuffer[expert_start_indices.dtype](
            gpu_ctx,
            expert_start_indices.unsafe_ptr(),
            expert_start_indices.size(),
            owning=False,
        )
        var expert_ids_dev = DeviceBuffer[expert_ids.dtype](
            gpu_ctx, expert_ids.unsafe_ptr(), expert_ids.size(), owning=False
        )

        var max_M = Int(max_num_tokens_per_expert)
        var n_active = Int(num_active_experts)
        if max_M <= 0 or n_active <= 0:
            return
        if max_M > P:
            max_M = P

        var grid_z = n_active
        var max_experts = expert_ids.dim_size(0)
        var start_len = expert_start_indices.dim_size(0)
        if start_len <= 1:
            return
        var start_max = start_len - 1
        if max_experts > start_max:
            max_experts = start_max
        if grid_z > max_experts:
            grid_z = max_experts
        if grid_z <= 0:
            return

        # SM90 WGMMA pipeline kernels (warp-group, BF16 accum).
        # Use transpose RS path for very small M to avoid wasting WGMMA M=64.
        comptime SMALL_M_TRANSPOSE_THRESHOLD = 0

        # w_blocks: [E, N/4 (padded), Kbytes*4] Hopper value swizzled
        # w_scales: [E, N/32 (padded), K] Hopper scale swizzled

        var a_dev = DeviceBuffer[a.dtype](
            gpu_ctx, a.unsafe_ptr(), a.size(), owning=False
        )

        if max_M <= SMALL_M_TRANSPOSE_THRESHOLD:
            comptime BM = 256  # N-tile
            comptime BN = 128  # M-tile (keep >= 128 for SM90 WGMMA)
            comptime BK = 128
            comptime NUM_WARP_GROUPS = 1
            comptime NUM_PIPELINE_STAGES = 2

            var grid_x = ceildiv(N, BM)
            var grid_y = ceildiv(max_M, BN)
            if grid_x == 0 or grid_y == 0:
                return

            comptime kernel = (
                grouped_matmul_mxfp4_bf16_wgmma_sm90_pipeline_swload_transpose[
                    BM=BM,
                    BN=BN,
                    BK=BK,
                    WGMMA_M=64,
                    WGMMA_N=64,
                    WGMMA_K=16,
                    NUM_WARP_GROUPS=NUM_WARP_GROUPS,
                    NUM_PIPELINE_STAGES=NUM_PIPELINE_STAGES,
                    USE_VALUE_SWIZZLE=True,
                ]
            )
            comptime b_smem_layout = tile_layout_k_major[
                BF16, BN, BK, TensorMapSwizzle.SWIZZLE_128B
            ]()
            comptime b_bytes = b_smem_layout.size() * 2
            comptime a_stage_bytes = 0
            comptime b_stage_bytes = ((b_bytes + 255) // 256) * 256
            comptime stage_bytes = b_stage_bytes
            comptime smem_use = stage_bytes * NUM_PIPELINE_STAGES

            gpu_ctx.enqueue_function_experimental[kernel](
                a_dev,
                a_stride0,
                a_stride1,
                P,
                K,
                expert_start_dev,
                expert_ids_dev,
                grid_z,
                w_blocks_dev,
                w_blocks_stride0,
                w_blocks_stride1,
                w_blocks_stride2,
                w_scales_dev,
                w_scales_stride0,
                w_scales_stride1,
                w_scales_stride2,
                Kblocks,
                c_dev,
                out_stride0,
                out_stride1,
                N,
                grid_dim=(grid_x, grid_y, grid_z),
                block_dim=(WARPGROUP_SIZE * (NUM_WARP_GROUPS + 1), 1, 1),
                shared_mem_bytes=Int(smem_use),
            )
        else:
            # RS path for larger M: decode weights into regs, activations in shared.
            comptime BM = 256  # N-tile
            comptime BN = 128  # M-tile
            comptime BK = 128
            comptime NUM_WARP_GROUPS = 1
            comptime NUM_PIPELINE_STAGES = 2
            var grid_x = ceildiv(N, BM)
            var grid_y = ceildiv(max_M, BN)
            if grid_x == 0 or grid_y == 0:
                return

            comptime kernel = (
                grouped_matmul_mxfp4_bf16_wgmma_sm90_pipeline_swload_transpose[
                    BM=BM,
                    BN=BN,
                    BK=BK,
                    WGMMA_M=64,
                    WGMMA_N=64,
                    WGMMA_K=16,
                    NUM_WARP_GROUPS=NUM_WARP_GROUPS,
                    NUM_PIPELINE_STAGES=NUM_PIPELINE_STAGES,
                    USE_VALUE_SWIZZLE=True,
                ]
            )
            comptime b_smem_layout = tile_layout_k_major[
                BF16, BN, BK, TensorMapSwizzle.SWIZZLE_128B
            ]()
            comptime b_bytes = b_smem_layout.size() * 2
            comptime a_stage_bytes = 0
            comptime b_stage_bytes = ((b_bytes + 255) // 256) * 256
            comptime stage_bytes = b_stage_bytes
            comptime smem_use = stage_bytes * NUM_PIPELINE_STAGES

            gpu_ctx.enqueue_function_experimental[kernel](
                a_dev,
                a_stride0,
                a_stride1,
                P,
                K,
                expert_start_dev,
                expert_ids_dev,
                grid_z,
                w_blocks_dev,
                w_blocks_stride0,
                w_blocks_stride1,
                w_blocks_stride2,
                w_scales_dev,
                w_scales_stride0,
                w_scales_stride1,
                w_scales_stride2,
                Kblocks,
                c_dev,
                out_stride0,
                out_stride1,
                N,
                grid_dim=(grid_x, grid_y, grid_z),
                block_dim=(WARPGROUP_SIZE * (NUM_WARP_GROUPS + 1), 1, 1),
                shared_mem_bytes=Int(smem_use),
            )
