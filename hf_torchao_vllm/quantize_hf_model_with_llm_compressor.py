# https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w8a8_fp8/llama3_example.py

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

import fire

def run(model_name: str = 'facebook/opt-125m'):
    # Load model.
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure the quantization algorithm and scheme.
    # In this case, we:
    #   * quantize the weights to fp8 with per channel via ptq
    #   * quantize the activations to fp8 with dynamic per token
    recipe = QuantizationModifier(
        targets="Linear", 
        scheme="FP8_DYNAMIC", 
        ignore=[
            "lm_head",
            # for Qwen MoE, but ok to just hardcode here for now
            # https://github.com/vllm-project/llm-compressor/blob/33ef5f497a9801893764c6a2c880cb1f560067fa/examples/quantizing_moe/qwen_example.py#L10
            "re:.*mlp.gate$", 
            "re:.*mlp.shared_expert_gate$",
            # also skip attention and shared expert, to focus on MoE for now
            "re:.*self_attn.*",
            "re:.*shared_expert.*",
        ],
    )

    # Apply quantization.
    oneshot(model=model, recipe=recipe)

    # Confirm generations of the quantized model look sane.
    print("========== SAMPLE GENERATION ==============")
    dispatch_for_generation(model)
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
        model.device
    )
    output = model.generate(input_ids, max_new_tokens=20)
    print(tokenizer.decode(output[0]))
    print("==========================================")

    # Save to disk in compressed-tensors format.
    SAVE_DIR = "data/llmcompressor/" + "fp8-" + model_name.rstrip("/").split("/")[-1]
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

if __name__ == '__main__':
    fire.Fire(run)
