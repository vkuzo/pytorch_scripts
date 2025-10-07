# https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w8a8_fp8/llama3_example.py

import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

import fire

def run(
    model_name: str = 'facebook/opt-125m',
    quant_type: str = 'fp8',
):
    assert quant_type in ('fp8', 'nvfp4'), 'unsupported'

    # Load model.
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if quant_type == 'fp8':
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

    else:
        assert quant_type == 'nvfp4', 'unsupported'
        DATASET_ID = "HuggingFaceH4/ultrachat_200k"
        DATASET_SPLIT = "train_sft"
        NUM_CALIBRATION_SAMPLES = 20
        MAX_SEQUENCE_LENGTH = 2048
        ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
        ds = ds.shuffle(seed=42)

        def preprocess(example):
            chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
            return {
                "text": tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    chat_template=chat_template,
                )
            }

        ds = ds.map(preprocess)

        # Tokenize inputs.
        def tokenize(sample):
            return tokenizer(
                sample["text"],
                padding=False,
                max_length=MAX_SEQUENCE_LENGTH,
                truncation=True,
                add_special_tokens=False,
            )


        ds = ds.map(tokenize, remove_columns=ds.column_names)

        # Configure the quantization algorithm and scheme.
        # In this case, we:
        #   * quantize the weights to fp4 with per group 16 via ptq
        #   * calibrate a global_scale for activations, which will be used to
        #       quantize activations to fp4 on the fly
        recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])

        # Apply quantization.
        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=MAX_SEQUENCE_LENGTH,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        )

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
    SAVE_DIR = "data/llmcompressor/" + quant_type + "-" + model_name.rstrip("/").split("/")[-1]
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

if __name__ == '__main__':
    fire.Fire(run)
