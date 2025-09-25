#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Script for quantizing LLM models with TorchAO.
Supports various quantization configurations and model types.

Copy of Driss's https://www.internalfb.com/phabricator/paste/view/P1838316614
"""


import random
import numpy as np
import torch
import time
from pathlib import Path
from typing import Callable, Optional, Literal

from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
import torchao
from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
    Int4WeightOnlyConfig,
    Int8WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    PerRow,
    PerTensor,
    GemliteUIntXWeightOnlyConfig,
    Int4DynamicActivationInt4WeightConfig,
    Int8DynamicActivationInt4WeightConfig,
    CutlassInt4PackedLayout,
)
from torchao.prototype.mx_formats.inference_workflow import MXFPInferenceConfig
from torchao.prototype.mx_formats import MXGemmKernelChoice
from jsonargparse import CLI, Namespace
from rich import print

from torch._inductor.utils import do_bench_using_profiling

def benchmark_cuda_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    """Thin wrapper around do_bench_using_profiling"""
    no_args = lambda: func(*args, **kwargs)
    time = do_bench_using_profiling(no_args)
    return time * 1e3


# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_quantization_config(args):
    """Create TorchAo quantization config based on provided args."""
    granularity_mapping = {
        "per_row": PerRow(),
        "per_tensor": PerTensor(),
    }

    gran = granularity_mapping[args.granularity]

    match args.quant_type:
        case "autoquant":
            return TorchAoConfig("autoquant", min_sqnr=args.min_sqnr)
        case "fp8":
            single_config = Float8DynamicActivationFloat8WeightConfig(granularity=gran)
            if args.experts_only_qwen_1_5_moe_a_2_7b:
                from torchao.quantization import ModuleFqnToConfig
                expert_fqn_to_config = {}
                # TODO(future PR): this is annoying, I should be able to use a regex here
                for layer_idx in range(24):
                    for expert_idx in range(60):
                        expert_fqn_to_config[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj"] = single_config
                        expert_fqn_to_config[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj"] = single_config
                        expert_fqn_to_config[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj"] = single_config
                module_fqn_to_config = ModuleFqnToConfig({
                    "_default": None,
                    **expert_fqn_to_config,
                })
                return TorchAoConfig(
                    quant_type=module_fqn_to_config,
                )
            else:
                return TorchAoConfig(single_config)
        case "int4_weight_only":
            return TorchAoConfig(Int4WeightOnlyConfig(group_size=128))
        case "int8_weight_only":
            return TorchAoConfig(Int8WeightOnlyConfig())
        case "int8_dynamic_act_int8_weight":
            return TorchAoConfig(Int8DynamicActivationInt8WeightConfig())
        case "gemlite":
            return TorchAoConfig(GemliteUIntXWeightOnlyConfig())
        case "A4W4":
            return TorchAoConfig(Int4DynamicActivationInt4WeightConfig())
        case "A8W4":
            return TorchAoConfig(
                Int8DynamicActivationInt4WeightConfig(layout=CutlassInt4PackedLayout())
            )
        case "mxfp8":
            return TorchAoConfig(MXFPInferenceConfig())
        case "mxfp4":
            return TorchAoConfig(
                MXFPInferenceConfig(
                    activation_dtype=torch.float4_e2m1fn_x2,
                    weight_dtype=torch.float4_e2m1fn_x2,
                    block_size=32,
                    gemm_kernel_choice=MXGemmKernelChoice.CUTLASS,
                )
            )
        case _:
            raise ValueError(f"Unsupported quantization type: {args.quant_type}")


def benchmark_model(model, input_ids, max_new_tokens, name=""):
    """Benchmark model generation speed."""
    try:
        time_ms = benchmark_cuda_function_in_microseconds(
            model.generate,
            **input_ids,
            max_new_tokens=max_new_tokens,
            cache_implementation="static",
        )
        tokens_per_second = max_new_tokens / (time_ms / 1000)
        print(
            f"{name} model: {time_ms:.2f}ms for {max_new_tokens} tokens ({tokens_per_second:.2f} tokens/sec)"
        )
        return time_ms
    except ImportError:
        # Fallback to simple timing if inductor utils not available
        print("torch._inductor.utils not available, using simple timing")
        start = time.time()
        model.generate(
            **input_ids, max_new_tokens=max_new_tokens, cache_implementation="static"
        )
        elapsed = (time.time() - start) * 1000  # ms
        tokens_per_second = max_new_tokens / (elapsed / 1000)
        print(
            f"{name} model: {elapsed:.2f}ms for {max_new_tokens} tokens ({tokens_per_second:.2f} tokens/sec)"
        )
        return elapsed


def main(
    model_name: str = "facebook/opt-125m",
    output_dir: Optional[str] = None,
    push_to_hub: bool = False,
    quant_type: Literal[
        "fp8",
        "int4_weight_only",
        "int8_weight_only",
        "int8_dynamic_act_int8_weight",
        "autoquant",
        "gemlite",
        "A4W4",
        "A8W4",
        "fp8",
        "mxfp4",
    ] = "fp8",
    granularity: Literal["per_row", "per_tensor"] = "per_row",
    min_sqnr: Optional[float] = None,
    max_new_tokens: int = 64,
    benchmark: bool = False,
    bench_tokens: int = 100,
    device_map: str = "cuda",
    experts_only_qwen_1_5_moe_a_2_7b: bool = False,
    save_model_to_disk: bool = True,
):
    """
    Quantize a model with TorchAO and test its performance.

    Args:
        model_name: Model to quantize (e.g., meta-llama/Meta-Llama-3-8B, facebook/opt-125m, Qwen/Qwen1.5-MoE-A2.7B)
        output_dir: Directory to save the quantized model
        push_to_hub: HF Hub repo name to push the model (e.g., 'your-username/model-name')
        quant_type: Quantization type to use
        granularity: Quantization granularity
        min_sqnr: Minimum SQNR for autoquant
        max_new_tokens: Max tokens to generate for testing
        benchmark: Run benchmarking comparison
        bench_tokens: Number of tokens to generate for benchmarking
        device_map: Device mapping strategy
        experts_only_qwen_1_5_moe_a_2_7b: if True, quantizes experts only for Qwen1.5-MoE-A2.7B model
        save_model_to_disk: if True, saves quantized model to local disk
    """
    # Set seed before creating the model
    set_seed(42)

    # Set default output directory based on model base name if not provided
    if output_dir is None:
        model_base_name = model_name.split("/")[-1]
        output_dir = f"data/{quant_type}-{model_base_name}"

    # Convert to args-like object for compatibility with the rest of the code
    args = Namespace(
        model_name=model_name,
        output_dir=output_dir,
        push_to_hub=push_to_hub,
        quant_type=quant_type,
        granularity=granularity,
        min_sqnr=min_sqnr,
        max_new_tokens=max_new_tokens,
        benchmark=benchmark,
        bench_tokens=bench_tokens,
        device_map=device_map,
        experts_only_qwen_1_5_moe_a_2_7b=experts_only_qwen_1_5_moe_a_2_7b,
        save_model_to_disk=save_model_to_disk,
    )
    print(f"{args=}")

    if args.experts_only_qwen_1_5_moe_a_2_7b:
        assert args.quant_type == "fp8", "unsupported"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get quantization config
    quantization_config = get_quantization_config(args)

    # Load and quantize model
    print("Loading and quantizing model...")
    quantized_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="bfloat16",
        device_map=args.device_map,
        quantization_config=quantization_config,
    )
    print(quantized_model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Test prompts
    prompts = [
        "Why is Pytorch 2.0 the best machine learning compiler?",
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Test generation
    print("\nTesting quantized model generation...")
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True).to(
        quantized_model.device
    )
    outputs = quantized_model.generate(**input_ids, max_new_tokens=args.max_new_tokens)

    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    if args.save_model_to_disk:
        # Save quantized model
        print(f"\nSaving quantized model to: {output_dir}")
        quantized_model.save_pretrained(output_dir, safe_serialization=False)
        tokenizer.save_pretrained(output_dir)

    # Push to HuggingFace hub if requested
    if args.push_to_hub:
        # Get model name from output_dir
        model_name = output_dir.name
        hub_path = f"vkuzometa/ao_models/{model_name}"
        print(f"Pushing model to HuggingFace Hub: {hub_path}")
        quantized_model.push_to_hub(model_name, safe_serialization=False)
        tokenizer.push_to_hub(model_name)

    if args.save_model_to_disk:
        # Load saved model to verify
        print("\nLoading saved quantized model to verify...")
        # TODO: do we really need `weights_only=False` here?
        loaded_model = AutoModelForCausalLM.from_pretrained(
            output_dir, device_map=args.device_map, torch_dtype="auto", weights_only=False,
        )

        # Test loaded model with first prompt
        test_prompt = prompts[0]
        input_ids = tokenizer(test_prompt, return_tensors="pt").to(loaded_model.device)
        output = loaded_model.generate(**input_ids, max_new_tokens=args.max_new_tokens)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Verification - Prompt: {test_prompt!r}, Generated text: {generated_text!r}")

    # Benchmark if requested
    if args.benchmark:
        assert args.save_model_to_disk, "unsupported"
        print("\nBenchmarking models...")
        # Benchmark quantized model
        print("Benchmarking quantized model:")
        quant_time = benchmark_model(
            loaded_model, input_ids, args.bench_tokens, f"Quantized ({args.quant_type})"
        )

        # Load and benchmark original model in BF16
        print("\nLoading original model in BF16 for comparison...")
        bf16_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, device_map=args.device_map, torch_dtype=torch.bfloat16
        )

        # Benchmark original model
        print("Benchmarking original BF16 model:")
        bf16_time = benchmark_model(bf16_model, input_ids, args.bench_tokens, "BF16")

        # Calculate speedup
        speedup = bf16_time / quant_time if quant_time > 0 else 0
        print(f"\nSpeedup: {speedup:.2f}x")

    print("\nQuantization process completed successfully.")


if __name__ == "__main__":
    CLI(main)
