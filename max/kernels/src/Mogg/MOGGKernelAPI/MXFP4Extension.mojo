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
from runtime.asyncrt import DeviceContextPtr
from nn.moe_mxfp4 import mxfp4_grouped_matmul, mxfp4_grouped_matmul_swiglu
from tensor import InputTensor, OutputTensor
from tensor.transitional import managed_tensor_slice_to_ndbuffer


@compiler.register("mo.moe.mx4.matmul")
struct Mxfp4GroupedMatmul:
    """MXFP4 Grouped MatMul kernel."""

    @staticmethod
    fn execute[
        target: StaticString,
        c_type: DType,
        a_type: DType,
    ](
        c: OutputTensor[dtype=c_type, rank=2],
        a: InputTensor[dtype=a_type, rank=2],
        packed_b: InputTensor[dtype = DType.uint8, rank=3],
        scales: InputTensor[dtype = DType.uint8, rank=3],
        bias: InputTensor[dtype=c_type, rank=2],
        expert_offsets: InputTensor[dtype = DType.uint32, rank=1],
        expert_ids: InputTensor[dtype = DType.int32, rank=1],
        max_num_tokens_per_expert: Scalar[DType.int64],
        num_active_experts: Scalar[DType.int64],
        ctx: DeviceContextPtr,
    ) raises:
        """Execute the MXFP4 Grouped MatMul kernel.

        Parameters:
            target: The target architecture.
            c_type: The output data type.
            a_type: The input data type.

        Args:
            c: The output tensor.
            a: The input tensor.
            packed_b: The packed weights tensor.
            scales: The packed scales tensor.
            bias: The bias tensor.
            expert_offsets: The expert offsets tensor.
            expert_ids: The expert IDs tensor.
            max_num_tokens_per_expert: The maximum number of tokens per expert.
            num_active_experts: The number of active experts.
            ctx: The device context.

        Raises:
            Error: If the kernel execution fails.
        """
        mxfp4_grouped_matmul[c_type, a_type, target](
            managed_tensor_slice_to_ndbuffer(c),
            managed_tensor_slice_to_ndbuffer(a),
            managed_tensor_slice_to_ndbuffer(packed_b),
            managed_tensor_slice_to_ndbuffer(scales),
            managed_tensor_slice_to_ndbuffer(bias),
            managed_tensor_slice_to_ndbuffer(expert_offsets),
            managed_tensor_slice_to_ndbuffer(expert_ids),
            Int(max_num_tokens_per_expert),
            Int(num_active_experts),
            ctx.get_device_context(),
        )


@compiler.register("mo.moe.mx4.matmul.swiglu")
struct Mxfp4GroupedMatmulSwiglu:
    """MXFP4 Grouped MatMul SwiGLU kernel."""

    @staticmethod
    fn execute[
        target: StaticString,
        c_type: DType,
        a_type: DType,
    ](
        c: OutputTensor[dtype=c_type, rank=2],
        a: InputTensor[dtype=a_type, rank=2],
        packed_b: InputTensor[dtype = DType.uint8, rank=3],
        scales: InputTensor[dtype = DType.uint8, rank=3],
        bias: InputTensor[dtype=c_type, rank=2],
        expert_offsets: InputTensor[dtype = DType.uint32, rank=1],
        expert_ids: InputTensor[dtype = DType.int32, rank=1],
        max_num_tokens_per_expert: Scalar[DType.int64],
        num_active_experts: Scalar[DType.int64],
        alpha: Scalar[DType.float32],
        limit: Scalar[DType.float32],
        ctx: DeviceContextPtr,
    ) raises:
        """Execute the MXFP4 Grouped MatMul SwiGLU kernel.

        Parameters:
            target: The target architecture.
            c_type: The output data type.
            a_type: The input data type.

        Args:
            c: The output tensor.
            a: The input tensor.
            packed_b: The packed weights tensor.
            scales: The packed scales tensor.
            bias: The bias tensor.
            expert_offsets: The expert offsets tensor.
            expert_ids: The expert IDs tensor.
            max_num_tokens_per_expert: The maximum number of tokens per expert.
            num_active_experts: The number of active experts.
            alpha: The alpha parameter for SwiGLU.
            limit: The limit parameter for SwiGLU.
            ctx: The device context.

        Raises:
            Error: If the kernel execution fails.
        """
        mxfp4_grouped_matmul_swiglu[c_type, a_type, target](
            managed_tensor_slice_to_ndbuffer(c),
            managed_tensor_slice_to_ndbuffer(a),
            managed_tensor_slice_to_ndbuffer(packed_b),
            managed_tensor_slice_to_ndbuffer(scales),
            managed_tensor_slice_to_ndbuffer(bias),
            managed_tensor_slice_to_ndbuffer(expert_offsets),
            managed_tensor_slice_to_ndbuffer(expert_ids),
            Int(max_num_tokens_per_expert),
            Int(num_active_experts),
            alpha,
            limit,
            ctx.get_device_context(),
        )
