"""Correctness tests for the Triton quant-cast recipes: each `triton_fn` must reproduce its
gold `pt_ref_fn`'s outputs. Inputs come from the recipe's (inherited) `example_input_fn`.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_cast_triton.recipes import ALL_RECIPES

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _qdata_equal(a, b):
    # fp8 has no packing, and fp8 -> fp32 is lossless, so a bit-exact compare is a fp32-cast compare.
    return torch.equal(a.to(torch.float32), b.to(torch.float32))


@pytest.mark.parametrize("name, recipe", ALL_RECIPES, ids=[n for n, _ in ALL_RECIPES])
def test_triton_matches_reference(name, recipe):
    # the Triton kernel must reproduce the gold reference bit-for-bit (identical fp32 math + RNE
    # cast). example_input_fn builds the full positional inputs (x, *aux).
    torch.manual_seed(0)
    inputs = recipe.example_input_fn(512, 512)

    ref_outs = recipe.pt_ref_fn(*inputs)
    tri_outs = recipe.triton_fn(*inputs)

    assert len(tri_outs) == len(ref_outs), f"{name}: output count {len(tri_outs)} != {len(ref_outs)}"
    for i, (t, r) in enumerate(zip(tri_outs, ref_outs)):
        assert r.shape == t.shape and r.dtype == t.dtype, (
            f"{name} output {i}: shape/dtype mismatch ({t.shape}/{t.dtype} vs {r.shape}/{r.dtype})"
        )
        assert _qdata_equal(t, r), f"{name} output {i}: triton differs from reference"
