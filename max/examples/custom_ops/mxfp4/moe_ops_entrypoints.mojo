# moe_mxfp4_ops_entrypoints.mojo
#
# Custom op entrypoints for MXFP4 MoE ops (W1/W2).
#
# Contract (matches `examples/custom-models/triton_example/moe.py` at a high level):
# - Routing is provided as:
#     token_expert_order : [P] u32  (pair_idx sorted by expert)
#     expert_start_indices: [E+1] u32 (segment starts for each expert id 0..E-1)
# - W1 kernel computes: h_sorted[pair, :] = SwiGLU(x[token] @ W1_expert + b1_expert)
# - W2 kernel computes per-pair outputs:
#     y_pairs[pair, :] = gate_weight[pair_idx] * (h_sorted[pair] @ W2_expert + b2_expert)
# - TOPK reduction kernel maps y_pairs -> y[token, :].

from math import ceildiv

import compiler
from gpu.host import DeviceBuffer
from gpu.host.info import is_cpu
from layout.tensor_core_async import tile_layout_k_major
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor

from .moe_ops_common import (
    BF16,
    F32,
    U8,
    U32,
    I32,
    BYTES_PER_BLOCK,
    VALUES_PER_BLOCK,
)

from .moe_ops_kernels import (
    moe_topk_reduce_pairs_bf16,
    moe_w1_mxfp4_swiglu_wgmma,
    moe_w2_mxfp4_scatter_wgmma,
)

@compiler.register("mxfp4_moe_w1_swiglu")
struct MXFP4MoEW1SwiGlu:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        h_sorted: OutputTensor[dtype=BF16, rank=2],
        x: InputTensor[dtype=BF16, rank=2],
        token_expert_order: InputTensor[dtype=U32, rank=1],
        expert_start_indices: InputTensor[dtype=U32, rank=1],
        expert_ids: InputTensor[dtype=I32, rank=1],
        expert_usage_stats: InputTensor[dtype=U32, rank=1],
        w_blocks: InputTensor[dtype=U8, rank=4],
        w_scales: InputTensor[dtype=U8, rank=3],
        bias: InputTensor[dtype=F32, rank=2],
        alpha: Float32,
        limit: Float32,
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if is_cpu[target]():
            raise Error("mxfp4_moe_w1_swiglu: GPU only")

        var T = x.dim_size(0)
        var D = x.dim_size(1)
        var P = token_expert_order.dim_size(0)

        var num_experts = w_blocks.dim_size(0)
        var kblocks = w_blocks.dim_size(2)

        var n_raw_total = bias.dim_size(1)
        var I = n_raw_total // 2

        if P == 0:
            return

        var grid_x = ceildiv(I, 64)  # BN_ACT = 64 (BN_RAW = 128)
        # Tune launch geometry without CPU sync:
        # - For tiny P (decode/small batches), launching all experts dominates overhead.
        #   We can cap grid_z by P since `num_active_experts <= P` for TOPK routing.
        # - For small P, extra Y blocks are pure overhead (they immediately return).
        var grid_y = 2
        if P <= 128:
            grid_y = 1
        var grid_z = num_experts
        if grid_z > P:
            grid_z = P

        var gpu_ctx = ctx.get_device_context()

        var x_dev = DeviceBuffer[x.dtype](
            gpu_ctx, x.unsafe_ptr(), x.size(), owning=False
        )
        var token_expert_order_dev = DeviceBuffer[token_expert_order.dtype](
            gpu_ctx,
            token_expert_order.unsafe_ptr(),
            token_expert_order.size(),
            owning=False,
        )
        var expert_start_indices_dev = DeviceBuffer[expert_start_indices.dtype](
            gpu_ctx,
            expert_start_indices.unsafe_ptr(),
            expert_start_indices.size(),
            owning=False,
        )
        var expert_ids_dev = DeviceBuffer[expert_ids.dtype](
            gpu_ctx,
            expert_ids.unsafe_ptr(),
            expert_ids.size(),
            owning=False,
        )
        var expert_usage_stats_dev = DeviceBuffer[expert_usage_stats.dtype](
            gpu_ctx,
            expert_usage_stats.unsafe_ptr(),
            expert_usage_stats.size(),
            owning=False,
        )
        var w_blocks_dev = DeviceBuffer[w_blocks.dtype](
            gpu_ctx, w_blocks.unsafe_ptr(), w_blocks.size(), owning=False
        )
        var w_scales_dev = DeviceBuffer[w_scales.dtype](
            gpu_ctx, w_scales.unsafe_ptr(), w_scales.size(), owning=False
        )
        var bias_dev = DeviceBuffer[bias.dtype](
            gpu_ctx, bias.unsafe_ptr(), bias.size(), owning=False
        )
        var h_sorted_dev = DeviceBuffer[h_sorted.dtype](
            gpu_ctx, h_sorted.unsafe_ptr(), h_sorted.size(), owning=False
        )

        comptime w1_kernel = moe_w1_mxfp4_swiglu_wgmma[
            BM=128,
            BN_RAW=128,
            BK=64,
            WGMMA_M=64,
            WGMMA_N=128,
            WGMMA_K=16,
            NUM_WARP_GROUPS=2,
        ]
        comptime a_smem_layout = tile_layout_k_major[BF16, 128, 64]()
        comptime b_smem_layout = tile_layout_k_major[BF16, 128, 64]()
        comptime a_bytes = a_smem_layout.size() * 2
        comptime b_bytes = b_smem_layout.size() * 2
        comptime a1_off = ((a_bytes + 255) // 256) * 256
        comptime b0_off = ((a1_off + a_bytes + 255) // 256) * 256
        comptime b1_off = ((b0_off + b_bytes + 255) // 256) * 256
        comptime blocks_per_tile = 64 // VALUES_PER_BLOCK
        comptime pack_bytes = 128 * blocks_per_tile * BYTES_PER_BLOCK
        comptime pack0_off = ((b1_off + b_bytes + 255) // 256) * 256
        comptime pack1_off = ((pack0_off + pack_bytes + 255) // 256) * 256
        comptime smem_use = pack1_off + pack_bytes
        gpu_ctx.enqueue_function_experimental[w1_kernel](
            x_dev,
            T,
            D,
            token_expert_order_dev,
            P,
            expert_start_indices_dev,
            expert_ids_dev,
            expert_usage_stats_dev,
            w_blocks_dev,
            w_scales_dev,
            kblocks,
            n_raw_total,
            bias_dev,
            h_sorted_dev,
            I,
            Scalar[F32](alpha),
            Scalar[F32](limit),
            grid_dim=(grid_x, grid_y, grid_z),
            block_dim=(256, 1, 1),
            shared_mem_bytes=Int(smem_use),
        )


@compiler.register("mxfp4_moe_w2_pairs_bf16")
struct MXFP4MoEW2PairsBF16:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        y_pairs: OutputTensor[dtype=BF16, rank=2],
        h_sorted: InputTensor[dtype=BF16, rank=2],
        token_expert_order: InputTensor[dtype=U32, rank=1],
        expert_start_indices: InputTensor[dtype=U32, rank=1],
        expert_ids: InputTensor[dtype=I32, rank=1],
        expert_usage_stats: InputTensor[dtype=U32, rank=1],
        gate_weights: InputTensor[dtype=BF16, rank=1],
        w_blocks: InputTensor[dtype=U8, rank=4],
        w_scales: InputTensor[dtype=U8, rank=3],
        bias: InputTensor[dtype=F32, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if is_cpu[target]():
            raise Error("mxfp4_moe_w2_pairs_bf16: GPU only")

        var P = h_sorted.dim_size(0)
        var I = h_sorted.dim_size(1)
        var D = y_pairs.dim_size(1)
        var P_out = y_pairs.dim_size(0)

        var kblocks = w_blocks.dim_size(2)

        var grid_x = ceildiv(D, 128)  # BN = 128
        var grid_y = 2
        if P <= 128:
            grid_y = 1
        var grid_z = w_blocks.dim_size(0)
        if grid_z > P:
            grid_z = P

        var gpu_ctx = ctx.get_device_context()

        var h_sorted_dev = DeviceBuffer[h_sorted.dtype](
            gpu_ctx, h_sorted.unsafe_ptr(), h_sorted.size(), owning=False
        )
        var token_expert_order_dev = DeviceBuffer[token_expert_order.dtype](
            gpu_ctx,
            token_expert_order.unsafe_ptr(),
            token_expert_order.size(),
            owning=False,
        )
        var expert_start_indices_dev = DeviceBuffer[expert_start_indices.dtype](
            gpu_ctx,
            expert_start_indices.unsafe_ptr(),
            expert_start_indices.size(),
            owning=False,
        )
        var expert_ids_dev = DeviceBuffer[expert_ids.dtype](
            gpu_ctx,
            expert_ids.unsafe_ptr(),
            expert_ids.size(),
            owning=False,
        )
        var expert_usage_stats_dev = DeviceBuffer[expert_usage_stats.dtype](
            gpu_ctx,
            expert_usage_stats.unsafe_ptr(),
            expert_usage_stats.size(),
            owning=False,
        )
        var gate_weights_dev = DeviceBuffer[gate_weights.dtype](
            gpu_ctx,
            gate_weights.unsafe_ptr(),
            gate_weights.size(),
            owning=False,
        )
        var w_blocks_dev = DeviceBuffer[w_blocks.dtype](
            gpu_ctx, w_blocks.unsafe_ptr(), w_blocks.size(), owning=False
        )
        var w_scales_dev = DeviceBuffer[w_scales.dtype](
            gpu_ctx, w_scales.unsafe_ptr(), w_scales.size(), owning=False
        )
        var bias_dev = DeviceBuffer[bias.dtype](
            gpu_ctx, bias.unsafe_ptr(), bias.size(), owning=False
        )

        if y_pairs.size() % 2 != 0:
            raise Error(
                "mxfp4_moe_w2_pairs_bf16: y_pairs size must be even to alias"
                " as f32"
            )

        var y_pairs_ptr_f32 = rebind[UnsafePointer[Float32, MutAnyOrigin]](
            y_pairs.unsafe_ptr()
        )
        var y_pairs_dev = DeviceBuffer[F32](
            gpu_ctx, y_pairs_ptr_f32, y_pairs.size() // 2, owning=False
        )

        if P == 0 or P_out == 0:
            return

        comptime w2_kernel = moe_w2_mxfp4_scatter_wgmma[
            BM=128,
            BN=128,
            BK=64,
            WGMMA_M=64,
            WGMMA_N=128,
            WGMMA_K=16,
            NUM_WARP_GROUPS=2,
            WRITE_PAIRS=True,
            PAIR_OUT_BF16=True,
        ]
        comptime a_smem_layout = tile_layout_k_major[BF16, 128, 64]()
        comptime b_smem_layout = tile_layout_k_major[BF16, 128, 64]()
        comptime a_bytes = a_smem_layout.size() * 2
        comptime b_bytes = b_smem_layout.size() * 2
        comptime a1_off = ((a_bytes + 255) // 256) * 256
        comptime b0_off = ((a1_off + a_bytes + 255) // 256) * 256
        comptime b1_off = ((b0_off + b_bytes + 255) // 256) * 256
        comptime blocks_per_tile = 64 // VALUES_PER_BLOCK
        comptime pack_bytes = 128 * blocks_per_tile * BYTES_PER_BLOCK
        comptime pack0_off = ((b1_off + b_bytes + 255) // 256) * 256
        comptime pack1_off = ((pack0_off + pack_bytes + 255) // 256) * 256
        comptime smem_use = pack1_off + pack_bytes
        gpu_ctx.enqueue_function_experimental[w2_kernel](
            h_sorted_dev,
            P,
            I,
            token_expert_order_dev,
            expert_start_indices_dev,
            expert_ids_dev,
            expert_usage_stats_dev,
            gate_weights_dev,
            w_blocks_dev,
            w_scales_dev,
            kblocks,
            bias_dev,
            y_pairs_dev,
            P_out,
            D,
            grid_dim=(grid_x, grid_y, grid_z),
            block_dim=(256, 1, 1),
            shared_mem_bytes=Int(smem_use),
        )


@compiler.register("mxfp4_moe_topk_reduce_bf16")
struct MXFP4MoETopKReduceBF16:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        y: OutputTensor[dtype=BF16, rank=2],
        y_pairs: InputTensor[dtype=BF16, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if is_cpu[target]():
            raise Error("mxfp4_moe_topk_reduce_bf16: GPU only")

        var P = y_pairs.dim_size(0)
        var T = y.dim_size(0)
        var D = y.dim_size(1)

        if T == 0 or D == 0 or P == 0:
            return

        var gpu_ctx = ctx.get_device_context()
        var y_pairs_dev = DeviceBuffer[y_pairs.dtype](
            gpu_ctx, y_pairs.unsafe_ptr(), y_pairs.size(), owning=False
        )
        var y_dev = DeviceBuffer[y.dtype](
            gpu_ctx, y.unsafe_ptr(), y.size(), owning=False
        )

        comptime reduce_kernel = moe_topk_reduce_pairs_bf16[BN=256]
        gpu_ctx.enqueue_function_experimental[reduce_kernel](
            y_pairs_dev,
            P,
            y_dev,
            T,
            D,
            grid_dim=(ceildiv(D, 256), T, 1),
            block_dim=(256, 1, 1),
        )
