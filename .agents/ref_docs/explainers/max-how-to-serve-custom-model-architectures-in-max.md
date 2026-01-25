---
title: "How to Serve Custom Model Architectures in MAX"
description: "Learn to extend existing MAX model implementations to support and serve custom architectures using the OpenAI-compatible API."
---

# How to Serve Custom Model Architectures in MAX

Learn to extend existing MAX model implementations to support and serve custom architectures using the OpenAI-compatible API.

MAX comes with built-in support for popular model architectures like `Gemma3ForCausalLM`, `Qwen2ForCausalLM`, and
`LlamaForCausalLM`, so you can instantly deploy them by passing a specific Hugging Face model name to the `max serve`
command. You can also use MAX to serve a custom model architecture with the `max serve` command, which provides an
OpenAI-compatible API.

In this tutorial, you'll implement a custom architecture based on the Qwen2 model by extending MAX's existing Llama3
implementation. This approach demonstrates how to leverage MAX's built-in architectures to quickly support new models
with similar structures. By the end of this tutorial, you'll understand how to:

- Set up the required file structure for custom architectures.
- Extend existing MAX model implementations.
- Register your model architecture with MAX.
- Serve your model and make inference requests.

## Set up your environment

Create a Python project and install the necessary dependencies.

### Using pixi

1. If you don't have it, install `pixi`:

```sh
curl -fsSL https://pixi.sh/install.sh | sh
```

Then restart your terminal for the changes to take effect.

2. Create a project:

```sh
pixi init qwen2 \
     -c https://conda.modular.com/max-nightly/ -c conda-forge \
     && cd qwen2
```

3. Install the `modular` conda package:

```sh
pixi add modular
```

4. Start the virtual environment:

```sh
pixi shell
```

### Using uv

1. If you don't have it, install `uv`:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your terminal to make `uv` accessible.

2. Create a project:

```sh
uv init qwen2 && cd qwen2
```

3. Create and start a virtual environment:

```sh
uv venv && source .venv/bin/activate
```

4. Install the `modular` Python package:

```sh
uv pip install modular \
  --index https://whl.modular.com/nightly/simple/ \
  --prerelease allow
```

### Using pip

1. Create a project folder:

```sh
mkdir qwen2 && cd qwen2
```

2. Create and activate a virtual environment:

```sh
python3 -m venv .venv/qwen2 \
     && source .venv/qwen2/bin/activate
```

3. Install the `modular` Python package:

```sh
pip install --pre modular \
  --extra-index-url https://whl.modular.com/nightly/simple/
```

### Using conda

1. If you don't have it, install conda.

1. Initialize `conda` for shell interaction:

```sh
conda init
```

3. Create a project:

```sh
conda create -n qwen2
```

4. Start the virtual environment:

```sh
conda activate qwen2
```

5. Install the `modular` conda package:

```sh
conda install -c conda-forge -c https://conda.modular.com/max-nightly/ modular
```

## Understand the architecture structure

Before creating your custom architecture, let's understand how to organize your custom model project. Create the
following structure in your project directory:

```text
qwen2/
  |-- __init__.py
  |-- arch.py
  `-- model.py
```

Here's what each file does:

- **`__init__.py`**: Makes your architecture discoverable by MAX.
- **`arch.py`**: Registers your model with MAX, specifying supported encodings, capabilities, and which existing
  components to reuse.
- **`model.py`**: Contains your model implementation that extends an existing MAX model class.

When extending an existing architecture, you can often reuse configuration handling and weight adapters from the parent
model, significantly reducing the amount of code you need to write.

## Implement the main model class

When your model is similar to an existing architecture, you can extend that model class instead of building from
scratch. In this example, we'll extend the `Llama3Model` class to implement the `Qwen2Model` class:

**model.py**

```python
from __future__ import annotations

from max.driver import Device
from max.engine import InferenceSession
from max.graph.weights import Weights, WeightsAdapter
from max.nn.legacy import ReturnLogits
from max.pipelines.architectures.llama3.model import Llama3Model
from max.pipelines.lib import KVCacheConfig, PipelineConfig, SupportedEncoding
from transformers import AutoConfig

class Qwen2Model(Llama3Model):
    """Qwen2 pipeline model implementation."""

    attention_bias: bool = True
    """Whether to use attention bias."""

    def __init__(
            self,
            pipeline_config: PipelineConfig,
            session: InferenceSession,
            huggingface_config: AutoConfig,
            encoding: SupportedEncoding,
            devices: list[Device],
            kv_cache_config: KVCacheConfig,
            weights: Weights,
            adapter: WeightsAdapter | None = None,
            return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )
```

By inheriting from `Llama3Model`, the Qwen2 implementation automatically gets:

- The `execute`, `prepare_initial_token_inputs`, and `prepare_next_token_inputs` methods required by MAX.
- Graph building logic for transformer architectures.
- Configuration handling from Hugging Face models.
- Weight loading and conversion capabilities.

The only modification needed is setting `attention_bias = True` to match Qwen2's architecture specifics.

## Define your architecture registration

The `arch.py` file tells MAX about your model's capabilities. When extending an existing architecture, you can reuse
many components:

**arch.py**

```python
from max.graph.weights import WeightsFormat
from max.interfaces import PipelineTask
from max.nn.legacy.kv_cache import KVCacheStrategy
from max.pipelines.architectures.llama3 import weight_adapters
from max.pipelines.lib import (
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
)

from .model import Qwen2Model

qwen2_arch = SupportedArchitecture(
    name="Qwen2ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["Qwen/Qwen2.5-7B-Instruct", "Qwen/QwQ-32B"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: [KVCacheStrategy.PAGED],
        SupportedEncoding.bfloat16: [KVCacheStrategy.PAGED],
    },
    pipeline_model=Qwen2Model,
    tokenizer=TextTokenizer,
    rope_type=RopeType.normal,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
)
```

The `name` parameter must match the model class name in Hugging Face configs, while `task` specifies the pipeline task
type. The `rope_type` parameter specifies the type of rotary position embeddings used by the model. We are reusing
Llama3's weight adapters to handle conversion between formats like SafeTensors and GGUF.

## Load your architecture

Create an `__init__.py` file to make your architecture discoverable by MAX:

**\_\_init\_\_.py**

```python
from .arch import qwen2_arch

ARCHITECTURES = [qwen2_arch]

__all__ = ["qwen2_arch", "ARCHITECTURES"]
```

MAX automatically loads any architectures listed in the `ARCHITECTURES` variable when you specify your module with the
`--custom-architectures` flag.

## Test your custom architecture

You can now test your custom architecture using the `--custom-architectures` flag. From your project directory, run the
following command:

```bash
max serve \
  --model Qwen/Qwen2.5-7B-Instruct \
  --custom-architectures qwen2
```

The `--model` flag tells MAX to use a specified model. The `--custom-architectures` flag tells MAX to load custom
architectures from the specified Python module.

### Trust remote code

Some models require executing custom code from their repository. If you encounter an error about "trust_remote_code",
add the `--trust-remote-code` flag:

```bash
max serve \
  --model Qwen/Qwen2.5-7B-Instruct \
  --custom-architectures qwen2 \
  --trust-remote-code
```

The server is ready when you see this message:

```output
Server ready on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Now you can test your custom architecture by sending a request to the endpoint.

#### Using cURL

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [\
      {"role": "user", "content": "Hello! Can you help me with a simple task?"}\
    ],
    "max_tokens": 100
  }'
```

#### Using Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",  # Required by API but not used by MAX
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "Hello! Can you help me with a simple task?"}
        ],
    max_tokens=100,
)

print(response.choices[0].message.content)
```
