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

from __future__ import annotations

import importlib
import os
from pathlib import Path

from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType


def test_modular_home_bootstrap_allows_loading_custom_ops() -> None:
    old_modular_home = os.environ.pop("MODULAR_HOME", None)
    try:
        import gpt_oss_mxfp4

        importlib.reload(gpt_oss_mxfp4)

        modular_home = os.environ.get("MODULAR_HOME")
        assert modular_home is not None
        assert Path(modular_home).exists()

        from gpt_oss_mxfp4.kernels import (
            get_mxfp4_kernels_path,
            mxfp4_matmul_swiglu,
        )

        m, k, n_full = 2, 32, 8
        cpu = CPU()
        device_ref = DeviceRef.from_device(cpu)
        with Graph(
            "bootstrap_custom_ops",
            input_types=[
                TensorType(DType.float32, shape=[m, k], device=device_ref),
                TensorType(
                    DType.uint8, shape=[k // 32, n_full, 16], device=device_ref
                ),
                TensorType(
                    DType.float32, shape=[k // 32, n_full], device=device_ref
                ),
                TensorType(DType.float32, shape=[n_full], device=device_ref),
            ],
            custom_extensions=[get_mxfp4_kernels_path()],
        ) as graph:
            x, blocks, scales, bias = graph.inputs
            out = mxfp4_matmul_swiglu(x, blocks, scales, bias, target="cpu")
            graph.output(out)
    finally:
        if old_modular_home is None:
            os.environ.pop("MODULAR_HOME", None)
        else:
            os.environ["MODULAR_HOME"] = old_modular_home
