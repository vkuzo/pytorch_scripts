"""Triton implementations of the quant_cast_gold recipes.

Each recipe is a `QuantCastTritonRecipe` -- it inherits the gold reference
(`pt_ref_fn`/`correctness_fn`/`example_input_fn`/`perf_description`) from a
`QuantCastSingleKernelGold` and adds `triton_fn`, a Triton-backed implementation of the same
cast. Mirrors flexquant_v3's `RecipeV2` (inherit-from-gold + `from_gold`). test.py grades each
`triton_fn` against its gold `pt_ref_fn`.
"""

import os
import sys
from dataclasses import dataclass
from typing import Callable

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_cast_gold.recipes import Float8TensorwiseGold, QuantCastSingleKernelGold


@dataclass(frozen=True)
class QuantCastTritonRecipe(QuantCastSingleKernelGold):
    """A gold recipe plus a Triton implementation of its `pt_ref_fn`. Mirrors flexquant_v3's
    RecipeV2: inherits pt_ref_fn/correctness_fn/example_input_fn/perf_description from the gold,
    and adds `triton_fn` (same `(inputs) -> outputs` signature as `pt_ref_fn`)."""

    triton_fn: Callable | None = None

    @classmethod
    def from_gold(cls, gold: QuantCastSingleKernelGold, triton_fn: Callable) -> "QuantCastTritonRecipe":
        """Build a QuantCastTritonRecipe from a gold recipe, attaching its Triton implementation."""
        return cls(
            pt_ref_fn=gold.pt_ref_fn,
            correctness_fn=gold.correctness_fn,
            example_input_fn=gold.example_input_fn,
            perf_description=gold.perf_description,
            triton_fn=triton_fn,
        )


# ---------------------------------------------------------------------------
# fp8 tensorwise with a precomputed per-tensor scale. The scale is an input (a global reduction
# done outside), so the kernel is a pure elementwise cast: qdata = (x * (1/scale)).to(fp8_e4m3).
# ---------------------------------------------------------------------------
@triton.jit
def _fp8_tensorwise_kernel(x_ptr, scale_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    scale = tl.load(scale_ptr)  # precomputed per-tensor scalar
    y = (x * (1.0 / scale)).to(tl.float8e4nv)  # mirror float8_tensorwise_f exactly
    tl.store(y_ptr + offs, y, mask=mask)


def float8_tensorwise_triton(x, scale, **kwargs):
    """Triton impl matching float8_tensorwise_f: elementwise (x / scale) -> fp8_e4m3. `scale` is
    the precomputed per-tensor scalar. Returns a 1-tuple `(qdata,)`."""
    assert x.is_contiguous() and x.dim() == 2
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    n = x.numel()

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK"]),)

    _fp8_tensorwise_kernel[grid](x, scale, y, n, BLOCK=1024)
    return (y,)


FP8_TENSORWISE_PRECALC_SCALE = QuantCastTritonRecipe.from_gold(
    Float8TensorwiseGold, triton_fn=float8_tensorwise_triton
)

ALL_RECIPES = [
    ("fp8_tensorwise_precalc_scale", FP8_TENSORWISE_PRECALC_SCALE),
]
