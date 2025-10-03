# HF -> torchao -> vLLM convenience scripts

Example

```bash
# save a quantized model ot data/nvfp4-Qwen1.5-MoE-A2.7B
python quantize_hf_model_with_torchao.py --model_name "Qwen/Qwen1.5-MoE-A2.7B" --experts_only_qwen_1_5_moe_a_2_7b True --save_model_to_disk True --quant_type nvfp4

# run the model from above in vLLM
python run_quantized_model_in_vllm.py --model_name "data/torchao/nvfp4-Qwen1.5-MoE-A2.7B" --compile False
```
