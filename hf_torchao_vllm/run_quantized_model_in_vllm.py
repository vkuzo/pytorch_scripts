# SPDX-License-Identifier: Apache-2.0

import os
import random

import numpy as np
from rich import print
from vllm import LLM, SamplingParams

import torch
from torchao.quantization.quant_api import _quantization_type


# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Set seed before creating the LLM
set_seed(42)

# enable torch profiler, can also be set on cmd line
# os.environ["VLLM_TORCH_PROFILER_DIR"] = "data/flex_profile"
os.environ["VLLM_USE_V1"] = "1"
# os.environ["VLLM_ATTENTION_BACKEND"] = "FLEX_ATTENTION_VLLM_V1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
os.environ["VLLM_TEST_STANDALONE_COMPILE"] = "1"

# TODO: remove this after https://github.com/pytorch/ao/issues/2478 is fixed
torch.serialization.add_safe_globals([getattr])


def print_vllm_torchao_quant_info(model: torch.nn.Module):
    # vLLM model layers do not print any information about torchao
    # quantization, because the tensor subclasses wrap the weights and
    # the custom vLLM linear modules do not print information about the
    # weights. For now, hack it. In the future, we should fix this
    # in vLLM.

    seen_types = set()
    for name, mod in model.named_modules():
        if "Linear" not in str(type(mod)):
            continue
        if not hasattr(mod, "weight"):
            continue
        mod_and_weight_type = type(mod), type(mod.weight)
        if mod_and_weight_type in seen_types:
            continue
        seen_types.add(mod_and_weight_type)
        print(
            f"first seen torchao quantization for {mod}:\n  path {name}, quant_type {_quantization_type(mod.weight)}"
        )


def main(
    # model_name: str = "Qwen/Qwen2-7B-Instruct",
    model_name: str = "data/torchao/fp8-opt-125m",
    max_tokens=64,
    tp_size: int = 1,
    compile: bool = True,
    print_configs: bool = False,
    print_model: bool = True,
    gpu_memory_utilization: float = 0.9,
):
    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, seed=42, max_tokens=max_tokens
    )
    # Create an LLM.
    print(f"Using Model name: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        enforce_eager=not compile,
        max_model_len=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # Print diagnostic information
    # model config
    model_config = llm.llm_engine.model_config
    print(f"model_name: {model_config.model}")
    print(f"architecture: {model_config.architecture}")
    if print_configs:
        print(f"model_config: {model_config}")
        print(f"hf_config: {model_config.hf_config}")
    if print_model:
        # TODO: fix this for latest vllm, lines below crash when building from
        # source with https://www.internalfb.com/phabricator/paste/view/P2028278010
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        print(f"model: {model}")

        print_vllm_torchao_quant_info(model)

    prompts = [
        "Why is Pytorch 2.0 the best machine learning compiler?",
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
