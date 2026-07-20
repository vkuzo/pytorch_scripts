"""Standalone correctness tests for the golden quant-cast recipes.

Each `QuantCastSingleKernelGold` must be internally consistent: running its `correctness_fn`
on `pt_ref_fn`'s own outputs has to pass. That's a gold-package concern (no flex_tile_map
involved), so it lives here rather than in flexquant_v3/test.py. Kept independent of
flexquant_v3 -- inputs (and any aux args) come from each recipe's own `example_input_fn`.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_cast_gold.recipes import (
    ColwiseFp8Gold,
    ColwisePrecalcGold,
    Deepseek1x128DimMGold,
    Deepseek1x128Gold,
    Deepseek128x128Gold,
    Float8TensorwiseGold,
    HadamardRht,
    Mxfp8BiasGold,
    Mxfp8FloorGold,
    Mxfp8FloorSwizzleGold,
    Nvfp4BlockedOuterGold,
    Nvfp4GsSwizzleGold,
    RowwiseFp8Gold,
    RowwisePrecalcGold,
    SrF32ToBf16,
    SrF32ToBf16Global,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


GOLD_CASES = [
    ("deepseek_1x128", Deepseek1x128Gold),
    ("deepseek_128x128", Deepseek128x128Gold),
    ("deepseek_1x128_dim_m", Deepseek1x128DimMGold),
    ("rowwise_fp8", RowwiseFp8Gold),
    ("colwise_fp8", ColwiseFp8Gold),
    ("rowwise_precalc", RowwisePrecalcGold),
    ("colwise_precalc", ColwisePrecalcGold),
    ("mxfp8_floor", Mxfp8FloorGold),
    ("mxfp8_floor_swizzle", Mxfp8FloorSwizzleGold),
    ("float8_tensorwise", Float8TensorwiseGold),
    ("nvfp4_gs_swizzle", Nvfp4GsSwizzleGold),
    ("nvfp4_blocked_outer", Nvfp4BlockedOuterGold),
    ("mxfp8_bias", Mxfp8BiasGold),
    ("hadamard_rht", HadamardRht),
    ("sr_bf16", SrF32ToBf16),
    ("sr_bf16_global", SrF32ToBf16Global),
]


@pytest.mark.parametrize(
    "name, gold",
    GOLD_CASES,
    ids=[name for name, _ in GOLD_CASES],
)
def test_ref_correctness(name, gold):
    # each gold recipe is internally consistent: pt_ref_fn's own outputs clear its correctness_fn.
    # example_input_fn builds the full positional inputs (x, *aux). Calls pt_ref_fn directly on
    # the whole tensor (no flex_tile_map). The whole tensor is one tile, so we pass the origin
    # position kwargs a REFERENCE-style whole-tensor call would -- recipes that ignore them accept
    # **kwargs; sr_bf16_global needs them for its per-element global-position dither.
    torch.manual_seed(0)
    inputs = gold.example_input_fn(512, 512)

    outputs = gold.pt_ref_fn(*inputs, global_row=0, global_col=0, num_col=inputs[0].shape[1])
    gold.correctness_fn(inputs, outputs)  # raises AssertionError on failure
