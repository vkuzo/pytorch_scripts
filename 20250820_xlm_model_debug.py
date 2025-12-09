import copy
import time
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn as nn

from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
    PerRow,
    quantize_,
)

from torchao.quantization.quantize_.common import KernelPreference


def run():
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    model = AutoModelForMaskedLM.from_pretrained(
        "FacebookAI/xlm-roberta-base",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    print(model)

    log_shapes = False
    if log_shapes:

        def print_input_shape(module, input, output):
            """Hook function that prints the shape of the input tensor"""
            # input is a tuple of tensors, get the first one
            input_tensor = input[0]
            print(f"Input shape to {module.__class__.__name__}: {input_tensor.shape}")

        def register_hooks_to_linear_modules(model):
            hook_handles = []

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    handle = module.register_forward_hook(
                        lambda module, input, output, name=name: print(
                            f"Linear module '{name}' input shape: {input[0].shape}"
                        )
                    )
                    hook_handles.append(handle)

            return hook_handles

        register_hooks_to_linear_modules(model)

    test_with_token = False
    if test_with_token:
        # Usage
        register_hooks_to_linear_modules(model)
        # Prepare input[[transformers.XLMRobertaConfig]]
        inputs = tokenizer(
            "Bonjour, je suis un modèle <mask>.", return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.logits

        masked_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
        predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
        predicted_token = tokenizer.decode(predicted_token_id)
        print(f"The predicted token is: {predicted_token}")

    # can't use `do_bench_using_profiling` due to hitting
    # https://github.com/pytorch/pytorch/issues/153587
    def get_time_s(model, inputs):
        # warm up
        model(*inputs)
        model(*inputs)
        torch.cuda.synchronize()

        start_time = time.time()
        n_iter = 10
        for _ in range(n_iter):
            model(*inputs)
        torch.cuda.synchronize()
        end_time = time.time()

        final_time_s = (end_time - start_time) / n_iter
        return final_time_s

    test_encoder_performance = True
    if test_encoder_performance:
        # now, hand craft a tensor with seq_len 256 and batch_size 32 and pass it through the encoder
        bsz, seq_len, dim = 32, 256, 768
        input_tensor = torch.randn(
            bsz, seq_len, dim, dtype=torch.bfloat16, device="cuda"
        )
        inputs = (input_tensor,)
        # m_ref = model.roberta.encoder.layer[0].attention
        m_ref = model.roberta.encoder

        measure_bf16 = True
        if measure_bf16:
            m_ref_c = torch.compile(copy.deepcopy(m_ref))
            with torch.no_grad():
                bf16_time_sec = get_time_s(m_ref_c, inputs)
            print("bf16_time_sec", bf16_time_sec)
        else:
            bf16_time_sec = -1

        m_q = copy.deepcopy(m_ref)
        config = Float8DynamicActivationFloat8WeightConfig(
            granularity=PerRow(),
            # TODO: I should be able to use a string here
            kernel_preference=KernelPreference.TORCH,
        )
        print(config)

        def filter_fn(mod, name):
            return isinstance(mod, torch.nn.Linear) and "attention" not in name

        quantize_(m_q, config, filter_fn)
        print(m_q)
        m_q_c = torch.compile(m_q, mode="default")
        with torch.no_grad():
            fp8_time_sec = get_time_s(m_q_c, inputs)
        print("fp8_time_sec", fp8_time_sec, "speedup", bf16_time_sec / fp8_time_sec)


if __name__ == "__main__":
    run()
