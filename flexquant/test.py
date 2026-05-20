import pytest
import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck

from api import _HopMode, flex_cast_quant_dense
from api_triton_for_debugging import flex_cast_quant_dense_triton
from recipes import (
    Recipe,
    deepseek_fp8_1_128,
    deepseek_fp8_1_128_dim_m,
    deepseek_fp8_1_128_dim_m_triton,
    deepseek_fp8_128_128,
    deepseek_fp8_128_128_triton,
)

# (label, recipe, hop_mode). Recipes that support both HOP and non-HOP routes
# are listed twice with the two modes; triton-only recipes carry NO_HOP since
# the flag is unused for them.
RECIPES: list[tuple[str, Recipe, _HopMode]] = [
    ("deepseek_fp8_1_128", deepseek_fp8_1_128, _HopMode.NO_HOP),
    ("deepseek_fp8_1_128_dim_m", deepseek_fp8_1_128_dim_m, _HopMode.NO_HOP),
    ("deepseek_fp8_1_128_dim_m_hop", deepseek_fp8_1_128_dim_m, _HopMode.HOP),
    ("deepseek_fp8_1_128_dim_m_triton", deepseek_fp8_1_128_dim_m_triton, _HopMode.NO_HOP),
    ("deepseek_fp8_128_128", deepseek_fp8_128_128, _HopMode.NO_HOP),
    ("deepseek_fp8_128_128_hop", deepseek_fp8_128_128, _HopMode.HOP),
    ("deepseek_fp8_128_128_triton", deepseek_fp8_128_128_triton, _HopMode.NO_HOP),
]


def _call(recipe: Recipe, hop_mode: _HopMode, x: torch.Tensor, fn=None):
    if recipe._use_triton_kernel:
        triton_fn = fn if fn is not None else flex_cast_quant_dense_triton
        return triton_fn(
            x,
            block_size=recipe.block_size,
            dim=recipe.dim,
            qdata_dtype=recipe.qdata_dtype,
            scale_dtype=recipe.scale_dtype,
            amax_to_scale_fn_triton=recipe._amax_to_scale_fn_triton,
            cast_to_dtype_fn_triton=recipe._cast_to_dtype_fn_triton,
        )
    pt_fn = fn if fn is not None else flex_cast_quant_dense
    return pt_fn(
        x,
        block_size=recipe.block_size,
        dim=recipe.dim,
        qdata_dtype=recipe.qdata_dtype,
        scale_dtype=recipe.scale_dtype,
        amax_to_scale_fn=recipe.amax_to_scale_fn,
        cast_to_dtype_fn=recipe.cast_to_dtype_fn,
        _hop_mode=hop_mode,
    )


@pytest.mark.parametrize(
    "label,recipe,hop_mode", RECIPES, ids=[label for label, _, _ in RECIPES]
)
def test_eager_vs_reference(label: str, recipe: Recipe, hop_mode: _HopMode):
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    qdata, scale = _call(recipe, hop_mode, x)
    qdata_ref, scale_ref = recipe._reference_fn(x)

    assert torch.equal(qdata.to(torch.float32), qdata_ref.to(torch.float32))
    assert torch.equal(scale, scale_ref)


@pytest.mark.parametrize(
    "label,recipe,hop_mode",
    [(l, r, m) for l, r, m in RECIPES if not r._use_triton_kernel],
    ids=[l for l, r, _ in RECIPES if not r._use_triton_kernel],
)
def test_eager_vs_compile(label: str, recipe: Recipe, hop_mode: _HopMode):
    # Skipped for triton-backed recipes; they bypass torch.compile.
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    qdata_eager, scale_eager = _call(recipe, hop_mode, x)

    compiled = torch.compile(flex_cast_quant_dense, fullgraph=True)
    qdata_compiled, scale_compiled = _call(recipe, hop_mode, x, fn=compiled)

    # Inductor may reorder fp ops, causing tiny scale drift that flips a small
    # fraction of fp8 values by one quantization bin. Compare with tolerances
    # that accept one-bin drift but not algorithmic divergence.
    torch.testing.assert_close(scale_eager, scale_compiled, rtol=1e-2, atol=1e-6)
    bin_flip_frac = (
        qdata_eager.to(torch.float32) != qdata_compiled.to(torch.float32)
    ).float().mean().item()
    assert bin_flip_frac < 0.05, f"too many bin flips: {bin_flip_frac}"


def test_fuses_with_preceding_pointwise():
    recipe = deepseek_fp8_1_128

    def fn(x):
        flex_cast_quant_dense_c = torch.compile(flex_cast_quant_dense)
        return flex_cast_quant_dense_c(
            F.relu(x),
            block_size=recipe.block_size,
            dim=recipe.dim,
            qdata_dtype=recipe.qdata_dtype,
            scale_dtype=recipe.scale_dtype,
            amax_to_scale_fn=recipe.amax_to_scale_fn,
            cast_to_dtype_fn=recipe.cast_to_dtype_fn,
        )

    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    compiled = torch.compile(fn, fullgraph=True)
    _, code = run_and_get_code(compiled, x)
    triton_code = "\n".join(code)

    # If relu fuses into the quant kernel, there is exactly one @triton.jit
    # kernel emitted. A second kernel would mean Inductor failed to fuse.
    FileCheck().check_count("@triton.jit", 1, exactly=True).run(triton_code)
