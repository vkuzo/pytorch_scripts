"""
Simple example demonstrating quant_logger parameter and activation logging.

Usage:
    python example.py
"""

import torch
from diffusers import DiffusionPipeline

from quant_logger import add_activation_loggers, log_parameter_info

# Load model
pipe = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Log parameter statistics
print("=" * 70)
print("Parameter statistics:")
print("=" * 70)
log_parameter_info(pipe.transformer)

# Add activation loggers
add_activation_loggers(pipe.transformer)

# Generate one image
print("=" * 70)
print("Activation statistics during inference:")
print("=" * 70)
result = pipe(
    prompt="A cat holding a sign that says hello world",
    height=1024,
    width=1024,
    num_inference_steps=20,
    generator=torch.manual_seed(0),
)
