import pytest
import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck

from api import flex_cast_quant_dense
from recipes import (
    Recipe,
    deepseek_fp8_1_128,
    deepseek_fp8_1_128_dim_m,
    deepseek_fp8_1_128_dim_m_triton,
    deepseek_fp8_1_128_triton,
    deepseek_fp8_128_128,
    deepseek_fp8_128_128_hop,
    deepseek_fp8_128_128_triton,
)

RECIPES = [
    deepseek_fp8_1_128,
    deepseek_fp8_1_128_triton,
    deepseek_fp8_1_128_dim_m,
    deepseek_fp8_1_128_dim_m_triton,
    deepseek_fp8_128_128,
    deepseek_fp8_128_128_triton,
    deepseek_fp8_128_128_hop,
]


def _call(recipe: Recipe, x: torch.Tensor, fn=flex_cast_quant_dense):
    return fn(
        x,
        block_size=recipe.block_size,
        dim=recipe.dim,
        qdata_dtype=recipe.qdata_dtype,
        scale_dtype=recipe.scale_dtype,
        amax_to_scale_fn=recipe.amax_to_scale_fn,
        cast_to_dtype_fn=recipe.cast_to_dtype_fn,
        use_triton_kernel=recipe.use_triton_kernel,
        amax_to_scale_fn_triton=recipe.amax_to_scale_fn_triton,
        cast_to_dtype_fn_triton=recipe.cast_to_dtype_fn_triton,
        use_hop_path=recipe.use_hop_path,
    )


@pytest.mark.parametrize("recipe", RECIPES, ids=[r.name for r in RECIPES])
def test_eager_vs_reference(recipe: Recipe):
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    qdata, scale = _call(recipe, x)
    qdata_ref, scale_ref = recipe.reference_fn(x)

    assert torch.equal(qdata.to(torch.float32), qdata_ref.to(torch.float32))
    assert torch.equal(scale, scale_ref)


@pytest.mark.parametrize(
    "recipe",
    [r for r in RECIPES if not r.use_triton_kernel],
    ids=[r.name for r in RECIPES if not r.use_triton_kernel],
)
def test_eager_vs_compile(recipe: Recipe):
    # Skipped for triton-backed recipes; they bypass torch.compile.
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    qdata_eager, scale_eager = _call(recipe, x)

    compiled = torch.compile(flex_cast_quant_dense, fullgraph=True)
    qdata_compiled, scale_compiled = _call(recipe, x, fn=compiled)

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
        return flex_cast_quant_dense(
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
