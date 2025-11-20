# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import compiler_internal as compiler

from .mxfp4_kernel import mxfp4_grouped_matmul, mxfp4_grouped_matmul_swiglu
from builtin.dtype import DType
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from tensor.transitional import managed_tensor_slice_to_ndbuffer


@compiler.register("custom.moe.mx4.matmul")
struct MXFP4GroupedMatMul:
    @always_inline
    @staticmethod
    fn execute[
        c_type: DType,
        a_type: DType,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2],
        a: InputTensor[dtype=a_type, rank=2],
        packed_weights: InputTensor[dtype = DType.uint8, rank=3],
        packed_scales: InputTensor[dtype = DType.uint8, rank=3],
        bias: InputTensor[dtype = c_type, rank=2],
        expert_start_indices: InputTensor[dtype = DType.uint32, rank=1],
        expert_ids: InputTensor[dtype = DType.int32, rank=1],
        max_num_tokens_per_expert: UInt32,
        num_active_experts: UInt32,
        context: DeviceContextPtr,
    ) raises:
        var device_ctx = context.get_device_context()
        mxfp4_grouped_matmul[c_type, a_type, target](
            managed_tensor_slice_to_ndbuffer(c),
            managed_tensor_slice_to_ndbuffer(a),
            managed_tensor_slice_to_ndbuffer(packed_weights),
            managed_tensor_slice_to_ndbuffer(packed_scales),
            managed_tensor_slice_to_ndbuffer(bias),
            managed_tensor_slice_to_ndbuffer(expert_start_indices),
            managed_tensor_slice_to_ndbuffer(expert_ids),
            Int(max_num_tokens_per_expert),
            Int(num_active_experts),
            device_ctx,
        )


@compiler.register("custom.moe.mx4.matmul_swiglu")
struct MXFP4GroupedMatMulSwiGLU:
    @always_inline
    @staticmethod
    fn execute[
        c_type: DType,
        a_type: DType,
        target: StaticString,
    ](
        c: OutputTensor[dtype=c_type, rank=2],
        a: InputTensor[dtype=a_type, rank=2],
        packed_weights: InputTensor[dtype = DType.uint8, rank=3],
        packed_scales: InputTensor[dtype = DType.uint8, rank=3],
        bias: InputTensor[dtype = c_type, rank=2],
        expert_start_indices: InputTensor[dtype = DType.uint32, rank=1],
        expert_ids: InputTensor[dtype = DType.int32, rank=1],
        max_num_tokens_per_expert: UInt32,
        num_active_experts: UInt32,
        alpha: Float32,
        limit: Float32,
        context: DeviceContextPtr,
    ) raises:
        var device_ctx = context.get_device_context()
        mxfp4_grouped_matmul_swiglu[c_type, a_type, target](
            managed_tensor_slice_to_ndbuffer(c),
            managed_tensor_slice_to_ndbuffer(a),
            managed_tensor_slice_to_ndbuffer(packed_weights),
            managed_tensor_slice_to_ndbuffer(packed_scales),
            managed_tensor_slice_to_ndbuffer(bias),
            managed_tensor_slice_to_ndbuffer(expert_start_indices),
            managed_tensor_slice_to_ndbuffer(expert_ids),
            Int(max_num_tokens_per_expert),
            Int(num_active_experts),
            alpha,
            limit,
            device_ctx,
        )
