"""
Test that https://huggingface.co/docs/diffusers/en/quantization/torchao is not
broken by https://github.com/pytorch/ao/issues/1690
"""

import fire

def run():
    # copy-pasted from https://huggingface.co/docs/diffusers/en/quantization/torchao

    import torch
    import diffusers
    import torchao
    from diffusers import FluxPipeline, FluxTransformer2DModel, TorchAoConfig

    print(f"torch version: {torch.__version__}")
    print(f"torchao version: {torchao.__version__}")
    print(f"diffusers version: {diffusers.__version__}")

    model_id = "black-forest-labs/FLUX.1-dev"
    dtype = torch.bfloat16

    quantization_config = TorchAoConfig("int8wo")
    print(quantization_config)
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )
    print(transformer)
    pipe = FluxPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=dtype,
    )
    pipe.to("cuda")

    # Without quantization: ~31.447 GB
    # With quantization: ~20.40 GB
    print(f"Pipeline memory usage: {torch.cuda.max_memory_reserved() / 1024**3:.3f} GB")

    prompt = "A cat holding a sign that says hello world"
    image = pipe(
        prompt, num_inference_steps=50, guidance_scale=4.5, max_sequence_length=512
    ).images[0]
    image.save("output.png")    

if __name__ == '__main__':
    fire.Fire(run)
