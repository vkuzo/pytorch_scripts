# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a collection of PyTorch experimental scripts and utilities focused on quantization, distributed training, model optimization, and debugging. The repository contains standalone scripts and focused projects, not a traditional Python package.

## Key Projects

### hf_torchao_vllm
HuggingFace model quantization with TorchAO and vLLM integration. Core workflow:

```bash
# Quantize HF model with TorchAO and save to disk
python quantize_hf_model_with_torchao.py --model_name "Qwen/Qwen1.5-MoE-A2.7B" --experts_only_qwen_1_5_moe_a_2_7b True --save_model_to_disk True --quant_type nvfp4

# Run quantized model in vLLM
python run_quantized_model_in_vllm.py --model_name "data/torchao/nvfp4-Qwen1.5-MoE-A2.7B" --compile False

# Convert to compressed_tensors format for vLLM compatibility
python convert_torchao_checkpoint_to_compressed_tensors.py --dir_source data/torchao/fp8-opt-125m --dir_target data/torchao_compressed_tensors/fp8-opt-125m
```

**Code Quality:**
```bash
ruff format .
ruff check . --fix
```

Configuration in `.ruff.toml` (target Python 3.12, line length 80).

### graph_viz
Visualizes AOT graphs and Triton kernels from `torch.compile`:

```bash
# Generate logs
TORCH_LOGS_FORMAT=short TORCH_LOGS=aot_graphs,output_code your_script.py > your_logs.txt 2>&1

# Create HTML visualization
python main.py your_logs.txt ~/local/tmp/your_output_dir
```

Outputs interactive HTML with AOT joint graphs, partitioned graphs, and Triton kernel visualizations.

### dp2ep
Data Parallel to Expert Parallel (DP->EP) with DTensor for MoE models. Uses PyTorch DTensor and custom `ExpertParallel` parallelization strategy from torchtitan.

```bash
torchrun --nproc_per_node 2 main.py
```

Demonstrates sharding experts across devices using DTensor's `parallelize_module`.

### dtensor_parallelisms
Basic DTensor parallelism examples (FSDP, Tensor Parallel):

```bash
torchrun --nproc_per_node 2 main.py --mode fsdp
```

## Utility Scripts

- **add_inductor_metadata_to_perf_trace.py**: Merges Inductor metadata from `TORCH_LOGS=output_code` into `torch.profiler` traces for enriched visualization in chrome://tracing
- **test_ao.py**: TorchAO quantization testing with `quantize_()` and `autoquant()`
- **llvm_e8m0/**: Verifies LLVM APFloat E8M0 rounding behavior (requires LLVM v20.0+)
- **20240617_gpu_profile_debug.py**: PyTorch profiler debugging for `torch.compile` GPU events
- **build_vllm_with_existing_torch.sh**: Builds vLLM using an existing PyTorch installation

## Architecture Notes

### TorchAO Quantization Patterns
- Uses `TorchAoConfig` with HuggingFace transformers for quantization
- Supports per-module quantization via `ModuleFqnToConfig` with regex patterns (e.g., `r"re:.*experts.*gate_proj.*"`)
- Common quantization types: `fp8` (FP8 rowwise), `nvfp4`, `int8_weight_only`, `int4_weight_only`, `autoquant`
- Granularity options: `per_row`, `per_tensor`, `a1x128_w128x128`

### DTensor Parallelism
- Uses `DeviceMesh`, `distribute_tensor`, `Shard`, `Replicate` from `torch.distributed._tensor`
- Expert Parallel sharding via `parallelize_module` with custom `ExpertParallel()` plan
- Data Parallel typically implicit (manual slicing in torchtitan style)

### vLLM Integration
- Requires vLLM v1 (`VLLM_USE_V1=1`)
- Standalone compile mode: `VLLM_TEST_STANDALONE_COMPILE=1`
- Model weights wrapped in TorchAO tensor subclasses are not introspectable by default - use custom printing utilities

## Development Workflow

Most scripts use `fire` for CLI argument parsing and can be run directly:

```bash
python script_name.py --arg1 value1 --arg2 value2
```

Distributed scripts use `torchrun`:

```bash
torchrun --nproc_per_node N script.py
```

For debugging `torch.compile`:
- Use `TORCH_LOGS=aot_graphs,output_code` with `TORCH_LOGS_FORMAT=short`
- Capture both stdout and stderr: `script.py > logs.txt 2>&1`
- Use graph_viz for visualization or add_inductor_metadata_to_perf_trace for profiler integration

## Common Patterns

- **Seeding**: Many scripts use `set_seed()` for reproducibility with numpy, torch, and CUDA
- **Precision**: `torch.backends.fp32_precision = "ieee"` used in distributed scripts
- **Benchmarking**: Uses `torch._inductor.utils.do_bench_using_profiling` for GPU timing
- **Distributed setup**: `init_device_mesh("cuda", (world_size,))` with manual rank-based CUDA device setting
