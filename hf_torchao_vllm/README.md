# HF -> torchao -> vLLM convenience scripts

## hf -> torchao -> vLLM

```bash
# save a quantized model to data/torchao/nvfp4-Qwen1.5-MoE-A2.7B
# models tested: Qwen1.5-MoE-A2.7B (small MoE), facebook/opt-125m (tiny dense)
# quant_type tested: fp8 (which means fp8 rowwise here), nvfp4 (without any calibration)
python quantize_hf_model_with_torchao.py --model_name "Qwen/Qwen1.5-MoE-A2.7B" --experts_only_qwen_1_5_moe_a_2_7b True --save_model_to_disk True --quant_type nvfp4

# run the model from above in vLLM
# requires https://github.com/vllm-project/vllm/pull/26095 (currently not landed)
python run_quantized_model_in_vllm.py --model_name "data/torchao/nvfp4-Qwen1.5-MoE-A2.7B" --compile False
```

## hf -> torchao -> compressed_tensors checkpoint -> vLLM

```bash
# save a quantized model to data/torchao/fp8-opt-125m
# models tested: Qwen1.5-MoE-A2.7B (small MoE), facebook/opt-125m (tiny dense)
# quant_type tested: fp8 (which means fp8 rowwise here), nvfp4 (without any calibration). Note that nvfp4 on the MoE model leads to an error in vLLM.
python quantize_hf_model_with_torchao.py --model_name "facebook/opt-125m" --experts_only_qwen_1_5_moe_a_2_7b False --save_model_to_disk True --quant_type fp8

# (optional) save a quantized model with llm-compressor to data/llmcompressor/fp8-opt-125m
python quantize_hf_model_with_llm_compressor.py --model_name facebook/opt-125m --quant_type fp8

# (optional) inspect the torchao and compressed-tensors checkpoints
python inspect_torchao_output.py --dir_name data/torchao/fp8-opt-125m
python inspect_llm_compressor_output.py --dir_name data/llmcompressor/fp8-opt-125m

# convert the torchao checkpoint to compressed-tensors format
python convert_torchao_checkpoint_to_compressed_tensors.py --dir_source data/torchao/fp8-opt-125m --dir_target data/torchao_compressed_tensors/fp8-opt-125m --dir_validation data/llmcompressor/fp8-opt-125m

# run the converted model vLLM
python run_quantized_model_in_vllm.py --model_name "data/torchao_compressed_tensors/fp8-opt-125m" --compile False
```

## Code Quality & Linting

```bash
ruff format .
ruff check . --fix
```
