import pytest

pytest.importorskip("max")

from gpt_oss_mxfp4 import arch as arch_mod
from gpt_oss_mxfp4 import model as model_mod
from gpt_oss_mxfp4 import weight_adapters


def test_arch_registration_contract():
    # SupportedArchitecture must exist and carry the expected wiring.
    assert hasattr(arch_mod, "gpt_oss_arch"), "gpt_oss_arch must be defined"
    arch = arch_mod.gpt_oss_arch

    assert arch.name == "GptOssForCausalLM"
    assert arch.pipeline_model is model_mod.GptOssModel

    # Encoding contract: default and supported encodings should include BF16 + paged KV.
    assert hasattr(arch, "default_encoding")
    assert hasattr(arch, "supported_encodings")
    assert arch.default_encoding in arch.supported_encodings
    assert arch.supported_encodings[arch.default_encoding], (
        "Default encoding must have KV strategies"
    )

    # Weight adapter registration.
    assert hasattr(arch, "weight_adapters")
    adapters = arch.weight_adapters
    key = getattr(arch_mod, "WeightsFormat", None)
    if key is not None and hasattr(key, "safetensors"):
        key = key.safetensors
    else:
        key = arch.default_weights_format
    assert hasattr(weight_adapters, "convert_safetensor_state_dict")
    assert adapters.get(key) is weight_adapters.convert_safetensor_state_dict
