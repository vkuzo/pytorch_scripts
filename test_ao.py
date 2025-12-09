"""
Testing torchao for user empathy day
"""

import fire
import torch
import torch.nn as nn

import torchao

from torchao.quantization.quant_api import (
    quantize_,
    int8_weight_only,
)


def get_model(large_model: bool):
    if large_model:
        m = (
            nn.Sequential(
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.LayerNorm(2048),
                nn.Linear(2048, 1024),
                nn.Sigmoid(),
            )
            .cuda()
            .bfloat16()
        )
    else:
        m = (
            nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
            )
            .cuda()
            .bfloat16()
        )
    return m


def run(
    large_model: bool = False,
    use_autoquant: bool = False,
):
    if large_model:
        x = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    else:
        x = torch.randn(32, 32, dtype=torch.bfloat16, device="cuda")
    m = get_model(large_model)

    if use_autoquant:
        # try using autoquant_

        m = torchao.autoquant(torch.compile(m, mode="max-autotune"))
        m(x)
        print(m)

    else:
        # try using quantize_

        # when I try below, no quantization seems to be applied, and
        # there are no logs to tell me why
        # quantize_(m, int4_weight_only())

        # when I try below, I see the weights become `AffineQuantizedTensor`
        # but printing them out tells me nothing about the precision or type
        # of quantization that was applied
        quantize_(m, int8_weight_only())
        print(m)

    pass


if __name__ == "__main__":
    fire.Fire(run)
