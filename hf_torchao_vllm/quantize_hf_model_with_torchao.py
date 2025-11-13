#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Script for quantizing HuggingFace models with TorchAO.
Supports various quantization configurations and model types.
"""

import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from jsonargparse import CLI, Namespace
from rich import print

import torch
import transformers
from torch._inductor.utils import do_bench_using_profiling
from torchao.prototype.mx_formats import MXGemmKernelChoice
from torchao.prototype.mx_formats.inference_workflow import (
    MXFPInferenceConfig,
    NVFP4InferenceConfig,
    NVFP4MMConfig,
)
from torchao.quantization import (
    ModuleFqnToConfig,
    PerBlock,
    PerRow,
    PerTensor,
)
from torchao.quantization.quant_api import (
    CutlassInt4PackedLayout,
    Float8DynamicActivationFloat8WeightConfig,
    GemliteUIntXWeightOnlyConfig,
    Int4DynamicActivationInt4WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt4WeightConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
)
from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
    Float8Tensor,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig


def _assert_transformers_version_supports_regex():
    v = transformers.__version__
    assert str(v) >= "5", f"transformers version {v} not greater than 5"


def benchmark_cuda_function_in_microseconds(
    func: Callable, *args, **kwargs
) -> float:
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
        "a1x128_w128x128": [PerBlock([1, 128]), PerBlock([128, 128])],
    }

    gran = granularity_mapping[args.granularity]

    match args.quant_type:
        case "autoquant":
            return TorchAoConfig("autoquant", min_sqnr=args.min_sqnr)
        case "fp8":
            single_config = Float8DynamicActivationFloat8WeightConfig(
                granularity=gran,
                # the 125m model has a lot of activation zeroes for some
                # prompts, need to set a lower bound to prevent scales from
                # being 0.
                # TODO seems like torchao should do this for me.
                # TODO tool to find this (I used bisect on this tiny model).
                activation_value_lb=1.0e-12,
            )

            if args.experts_only_qwen_1_5_moe_a_2_7b:
                _assert_transformers_version_supports_regex()
                module_fqn_to_config = ModuleFqnToConfig(
                    {
                        r"re:.*experts.*gate_proj.*": single_config,
                        r"re:.*experts.*up_proj.*": single_config,
                        r"re:.*experts.*down_proj.*": single_config,
                    }
                )
                return TorchAoConfig(
                    quant_type=module_fqn_to_config,
                )

            elif args.ffn_only_llama_4_scout:
                # TODO gate this properly
                expert_3d_weight_single_config = Float8DynamicActivationFloat8WeightConfig(
                    # the weights of this model are stored in (B, K, N) layout, and we
                    # need to quantize rowwise across the K axis, which is `PerRow(1)`.
                    granularity=[PerRow(), PerRow(1)],
                    # the 125m model has a lot of activation zeroes for some
                    # prompts, need to set a lower bound to prevent scales from
                    # being 0.
                    # TODO seems like torchao should do this for me.
                    # TODO tool to find this (I used bisect on this tiny model).
                    activation_value_lb=1.0e-12,
                )
                _assert_transformers_version_supports_regex()
                module_fqn_to_config = ModuleFqnToConfig(
                    {
                        r"re:.*\.feed_forward\.experts\.gate_up_proj": expert_3d_weight_single_config,
                        r"re:.*\.feed_forward\.experts\.down_proj": expert_3d_weight_single_config,
                        # r"re:.*\.shared_expert\.down_proj": single_config,
                        # r"re:.*\.shared_expert\.up_proj": single_config,
                        # r"re:.*\.shared_expert\.gate_proj": single_config,
                    }
                )
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
                Int8DynamicActivationInt4WeightConfig(
                    layout=CutlassInt4PackedLayout()
                )
            )
        case "mxfp8":
            return TorchAoConfig(MXFPInferenceConfig())
        case "mxfp4":
            single_config = MXFPInferenceConfig(
                activation_dtype=torch.float4_e2m1fn_x2,
                weight_dtype=torch.float4_e2m1fn_x2,
                block_size=32,
                # gemm_kernel_choice=MXGemmKernelChoice.CUTLASS,
                gemm_kernel_choice=MXGemmKernelChoice.EMULATED,
            )
            if args.experts_only_qwen_1_5_moe_a_2_7b:
                expert_fqn_to_config = {}
                # TODO(future PR): this is annoying, I should be able to use a regex here
                for layer_idx in range(24):
                    expert_fqn_to_config[
                        f"model.layers.{layer_idx}.self_attn.q_proj"
                    ] = None
                    expert_fqn_to_config[
                        f"model.layers.{layer_idx}.self_attn.k_proj"
                    ] = None
                    expert_fqn_to_config[
                        f"model.layers.{layer_idx}.self_attn.v_proj"
                    ] = None
                    expert_fqn_to_config[
                        f"model.layers.{layer_idx}.self_attn.o_proj"
                    ] = None
                    expert_fqn_to_config[
                        f"model.layers.{layer_idx}.mlp.gate"
                    ] = None
                    expert_fqn_to_config[
                        f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj"
                    ] = None
                    expert_fqn_to_config[
                        f"model.layers.{layer_idx}.mlp.shared_expert.up_proj"
                    ] = None
                    expert_fqn_to_config[
                        f"model.layers.{layer_idx}.mlp.shared_expert.down_proj"
                    ] = None
                    expert_fqn_to_config[
                        f"model.layers.{layer_idx}.mlp.shared_expert_gate"
                    ] = None
                    expert_fqn_to_config["lm_head"] = None
                module_fqn_to_config = ModuleFqnToConfig(
                    {
                        "_default": single_config,
                        **expert_fqn_to_config,
                    }
                )

                return TorchAoConfig(
                    quant_type=module_fqn_to_config,
                )
            else:
                modules_to_not_convert = []
                if args.skip_gate_qwen_1_5_moe_a_2_7b:
                    for layer_idx in range(24):
                        modules_to_not_convert.append(
                            f"model.layers.{layer_idx}.mlp.gate"
                        )
                        modules_to_not_convert.append(
                            f"model.layers.{layer_idx}.mlp.shared_expert_gate"
                        )
                modules_to_not_convert.append("lm_head")
                return TorchAoConfig(
                    single_config,
                    modules_to_not_convert=modules_to_not_convert,
                )
        case "nvfp4":
            single_config = NVFP4InferenceConfig(
                mm_config=NVFP4MMConfig.WEIGHT_ONLY,
                use_triton_kernel=False,
                #
                # weight_only and use_dynamic_per_tensor_scale=True works here
                # but garbage output in vLLM, probably because we currently don't have a way
                # in torchao to enforce the scales for attention and ffn weights that
                # are going to be fused for inference to be the same
                # TODO: file a torchao issue about this, and fix in torchao
                #
                # dynamic and use_dynamic_per_tensor_scale=False not supported in torch._scaled_mm
                #
                use_dynamic_per_tensor_scale=False,
            )
            if args.experts_only_qwen_1_5_moe_a_2_7b:
                expert_fqn_to_config = {}
                # TODO(future PR): this is annoying, I should be able to use a regex here
                for layer_idx in range(24):
                    expert_fqn_to_config[
                        f"model.layers.{layer_idx}.self_attn.q_proj"
                    ] = None
                    expert_fqn_to_config[
                        f"model.layers.{layer_idx}.self_attn.k_proj"
                    ] = None
                    expert_fqn_to_config[
                        f"model.layers.{layer_idx}.self_attn.v_proj"
                    ] = None
                    expert_fqn_to_config[
                        f"model.layers.{layer_idx}.self_attn.o_proj"
                    ] = None
                    expert_fqn_to_config[
                        f"model.layers.{layer_idx}.mlp.gate"
                    ] = None
                    expert_fqn_to_config[
                        f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj"
                    ] = None
                    expert_fqn_to_config[
                        f"model.layers.{layer_idx}.mlp.shared_expert.up_proj"
                    ] = None
                    expert_fqn_to_config[
                        f"model.layers.{layer_idx}.mlp.shared_expert.down_proj"
                    ] = None
                    expert_fqn_to_config[
                        f"model.layers.{layer_idx}.mlp.shared_expert_gate"
                    ] = None
                    expert_fqn_to_config["lm_head"] = None
                module_fqn_to_config = ModuleFqnToConfig(
                    {
                        "_default": single_config,
                        **expert_fqn_to_config,
                    }
                )
                return TorchAoConfig(
                    quant_type=module_fqn_to_config,
                )
            else:
                modules_to_not_convert = []
                if args.skip_gate_qwen_1_5_moe_a_2_7b:
                    for layer_idx in range(24):
                        modules_to_not_convert.append(
                            f"model.layers.{layer_idx}.mlp.gate"
                        )
                        modules_to_not_convert.append(
                            f"model.layers.{layer_idx}.mlp.shared_expert_gate"
                        )
                modules_to_not_convert.append("lm_head")
                return TorchAoConfig(
                    single_config,
                    modules_to_not_convert=modules_to_not_convert,
                )
        case "none":
            return None
        case _:
            raise ValueError(
                f"Unsupported quantization type: {args.quant_type}"
            )


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
            **input_ids,
            max_new_tokens=max_new_tokens,
            cache_implementation="static",
        )
        elapsed = (time.time() - start) * 1000  # ms
        tokens_per_second = max_new_tokens / (elapsed / 1000)
        print(
            f"{name} model: {elapsed:.2f}ms for {max_new_tokens} tokens ({tokens_per_second:.2f} tokens/sec)"
        )
        return elapsed


def main(
    model_name: str = "facebook/opt-125m",
    output_dir: str | None = None,
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
        "nvfp4",
        "none",
    ] = "fp8",
    granularity: Literal[
        "per_row", "per_tensor", "a1x128_w128x128"
    ] = "per_row",
    min_sqnr: float | None = None,
    max_new_tokens: int = 64,
    benchmark: bool = False,
    bench_tokens: int = 100,
    device_map: str = "cuda",
    experts_only_qwen_1_5_moe_a_2_7b: bool = False,
    skip_gate_qwen_1_5_moe_a_2_7b: bool = False,
    ffn_only_llama_4_scout: bool = False,
    convert_llama_4_expert_weights_to_mnk: bool = False,
    save_model_to_disk: bool = True,
    load_model_from_disk: bool = True,
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
        skip_gate_qwen_1_5_moe_a_2_7b: if True, skips gate quantization for Qwen1.5-MoE-A2.7B model
        ffn_only_llama_4_scout: if True, FFN only for meta-llama/Llama-4-Scout-17B-16E-Instruct
        convert_llama_4_expert_weights_to_mnk: if True, converts LLaMa 4 Scout expert weights from MKN to MNK memory layout
        save_model_to_disk: if True, saves quantized model to local disk
        load_model_from_disk: if True, reloads model from disk to test it again
    """
    # Test prompts
    prompts = [
        "Why is Pytorch 2.0 the best machine learning compiler?",
        # "Hello, my name is",
        # "The president of the United States is",
        # "The capital of France is",
        # "The future of AI is",
    ]

    # Set seed before creating the model
    set_seed(42)

    # Set default output directory based on model base name if not provided
    if output_dir is None:
        model_base_name = model_name.split("/")[-1]
        output_dir = f"data/torchao/{quant_type}-{model_base_name}"

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
        load_model_from_disk=load_model_from_disk,
        skip_gate_qwen_1_5_moe_a_2_7b=skip_gate_qwen_1_5_moe_a_2_7b,
        ffn_only_llama_4_scout=ffn_only_llama_4_scout,
        convert_llama_4_expert_weights_to_mnk=convert_llama_4_expert_weights_to_mnk,
    )
    print(f"{args=}")

    if args.experts_only_qwen_1_5_moe_a_2_7b:
        assert args.quant_type in ("fp8", "mxfp4", "nvfp4"), "unsupported"

    assert not (
        args.skip_gate_qwen_1_5_moe_a_2_7b
        and args.experts_only_qwen_1_5_moe_a_2_7b
    ), "unsupported"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get quantization config
    quantization_config = get_quantization_config(args)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # breakpoint()

    # Load and quantize model
    print("Loading and quantizing model...")
    quantized_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="bfloat16",
        device_map=args.device_map,
        quantization_config=quantization_config,
    )
    print(quantized_model)

    if args.model_name == "meta-llama/Llama-4-Scout-17B-16E-Instruct":
        print(
            "quantized_model.model.layers[47].feed_forward.experts.down_proj",
            type(
                quantized_model.model.layers[47].feed_forward.experts.down_proj
            ),
        )
        print(
            "quantized_model.model.layers[47].feed_forward.experts.gate_up_proj",
            type(
                quantized_model.model.layers[
                    47
                ].feed_forward.experts.gate_up_proj
            ),
        )

    if args.convert_llama_4_expert_weights_to_mnk:
        if args.model_name != "meta-llama/Llama-4-Scout-17B-16E-Instruct":
            raise AssertionError("unimplemented")
        print("\nConverting LLaMa 4 expert weights from MKN to MNK layout")
        for name, param in quantized_model.named_parameters():
            if (
                ("feed_forward.experts.down_proj" in name)
                or ("feed_forward.experts.gate_up_proj" in name)
            ) and isinstance(param, Float8Tensor):
                # convert memory layout mkn -> mnk
                param.qdata = (
                    param.qdata.transpose(-2, -1).contiguous().transpose(-2, -1)
                )
        # TODO(future): investigate why this memory layout transformation does
        # not survive saving the checkpoint to disk

    # Test generation
    print("\nTesting quantized model generation...")
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True).to(
        quantized_model.device
    )
    outputs = quantized_model.generate(
        **input_ids, max_new_tokens=args.max_new_tokens
    )

    for i, (prompt, output) in enumerate(zip(prompts, outputs, strict=False)):
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

    if args.load_model_from_disk:
        assert args.save_model_to_disk, "unimplemented"
        # Load saved model to verify
        # TODO: do we really need `weights_only=False` here?
        loaded_model = AutoModelForCausalLM.from_pretrained(
            output_dir,
            device_map=args.device_map,
            torch_dtype="auto",
            weights_only=False,
        )

        # Test loaded model with first prompt
        test_prompt = prompts[0]
        input_ids = tokenizer(test_prompt, return_tensors="pt").to(
            loaded_model.device
        )
        output = loaded_model.generate(
            **input_ids, max_new_tokens=args.max_new_tokens
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(
            f"Verification - Prompt: {test_prompt!r}, Generated text: {generated_text!r}"
        )

    # Benchmark if requested
    if args.benchmark:
        assert args.save_model_to_disk, "unsupported"
        print("\nBenchmarking models...")
        # Benchmark quantized model
        print("Benchmarking quantized model:")
        quant_time = benchmark_model(
            loaded_model,
            input_ids,
            args.bench_tokens,
            f"Quantized ({args.quant_type})",
        )

        # Load and benchmark original model in BF16
        print("\nLoading original model in BF16 for comparison...")
        bf16_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map=args.device_map,
            torch_dtype=torch.bfloat16,
        )

        # Benchmark original model
        print("Benchmarking original BF16 model:")
        bf16_time = benchmark_model(
            bf16_model, input_ids, args.bench_tokens, "BF16"
        )

        # Calculate speedup
        speedup = bf16_time / quant_time if quant_time > 0 else 0
        print(f"\nSpeedup: {speedup:.2f}x")

    print("\nQuantization process completed successfully.")


if __name__ == "__main__":
    CLI(main)
