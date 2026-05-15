from typing import Callable, NamedTuple

import pytest
import torch
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck

from api import flex_quant_dense


class Recipe(NamedTuple):
    name: str
    block_size: int | tuple[int, int]
    dim: int | tuple[int, int]
    qdata_dtype: torch.dtype
    scale_dtype: torch.dtype
    calc_scale_fn: Callable
    cast_to_dtype_fn: Callable
    reference_fn: Callable


def _deepseek_fp8_1_128_calc_scale_fn(tile: torch.Tensor) -> torch.Tensor:
    return tile.abs().amax(dim=-1, keepdim=True) / torch.finfo(torch.float8_e4m3fn).max


def _deepseek_fp8_1_128_cast_to_dtype_fn(tile: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (tile / scale).to(torch.float8_e4m3fn)


def _deepseek_fp8_1_128_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, last = x.shape
    n_blocks = last // 128
    x_b = x.reshape(*lead, n_blocks, 128)
    scale = x_b.abs().amax(dim=-1, keepdim=True) / torch.finfo(torch.float8_e4m3fn).max
    qdata_b = (x_b / scale).to(torch.float8_e4m3fn)
    qdata = qdata_b.reshape(*lead, last)
    return qdata, scale.squeeze(-1).to(torch.float32)


def _deepseek_fp8_128_128_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    *lead, D1, D2 = x.shape
    n1, n2 = D1 // 128, D2 // 128
    x_b = (
        x.reshape(*lead, n1, 128, n2, 128)
        .transpose(-3, -2)
        .contiguous()
        .reshape(*lead, n1, n2, 128 * 128)
    )
    scale = x_b.abs().amax(dim=-1, keepdim=True) / fp8_max
    qdata_b = (x_b / scale).to(torch.float8_e4m3fn)
    qdata = (
        qdata_b.reshape(*lead, n1, n2, 128, 128)
        .transpose(-3, -2)
        .contiguous()
        .reshape(*lead, D1, D2)
    )
    return qdata, scale.squeeze(-1).to(torch.float32)


def _rowwise_fp8_calc_scale_fn(tile: torch.Tensor) -> torch.Tensor:
    return tile.abs().amax(dim=-1, keepdim=True) / torch.finfo(torch.float8_e4m3fn).max


def _rowwise_fp8_cast_to_dtype_fn(tile: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    return (tile / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)


def _rowwise_fp8_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = x.abs().amax(dim=-1, keepdim=True) / fp8_max
    qdata = (x / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return qdata, scale.squeeze(-1).to(torch.float32)


RECIPES = [
    Recipe(
        name="deepseek_fp8_1_128",
        block_size=128,
        dim=-1,
        qdata_dtype=torch.float8_e4m3fn,
        scale_dtype=torch.float32,
        calc_scale_fn=_deepseek_fp8_1_128_calc_scale_fn,
        cast_to_dtype_fn=_deepseek_fp8_1_128_cast_to_dtype_fn,
        reference_fn=_deepseek_fp8_1_128_reference,
    ),
    Recipe(
        name="deepseek_fp8_128_128",
        block_size=(128, 128),
        dim=(-2, -1),
        qdata_dtype=torch.float8_e4m3fn,
        scale_dtype=torch.float32,
        calc_scale_fn=_deepseek_fp8_1_128_calc_scale_fn,
        cast_to_dtype_fn=_deepseek_fp8_1_128_cast_to_dtype_fn,
        reference_fn=_deepseek_fp8_128_128_reference,
    ),
    Recipe(
        name="rowwise_fp8",
        block_size=-1,
        dim=-1,
        qdata_dtype=torch.float8_e4m3fn,
        scale_dtype=torch.float32,
        calc_scale_fn=_rowwise_fp8_calc_scale_fn,
        cast_to_dtype_fn=_rowwise_fp8_cast_to_dtype_fn,
        reference_fn=_rowwise_fp8_reference,
    ),
]


@pytest.mark.parametrize("recipe", RECIPES, ids=[r.name for r in RECIPES])
def test_eager_vs_reference(recipe: Recipe):
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    qdata, scale = flex_quant_dense(
        x,
        block_size=recipe.block_size,
        dim=recipe.dim,
        qdata_dtype=recipe.qdata_dtype,
        scale_dtype=recipe.scale_dtype,
        calc_scale_fn=recipe.calc_scale_fn,
        cast_to_dtype_fn=recipe.cast_to_dtype_fn,
    )
    qdata_ref, scale_ref = recipe.reference_fn(x)

    assert torch.equal(qdata.to(torch.float32), qdata_ref.to(torch.float32))
    assert torch.equal(scale, scale_ref)


@pytest.mark.parametrize("recipe", RECIPES, ids=[r.name for r in RECIPES])
def test_eager_vs_compile(recipe: Recipe):
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    qdata_eager, scale_eager = flex_quant_dense(
        x,
        block_size=recipe.block_size,
        dim=recipe.dim,
        qdata_dtype=recipe.qdata_dtype,
        scale_dtype=recipe.scale_dtype,
        calc_scale_fn=recipe.calc_scale_fn,
        cast_to_dtype_fn=recipe.cast_to_dtype_fn,
    )

    compiled = torch.compile(flex_quant_dense, fullgraph=True)
    qdata_compiled, scale_compiled = compiled(
        x,
        block_size=recipe.block_size,
        dim=recipe.dim,
        qdata_dtype=recipe.qdata_dtype,
        scale_dtype=recipe.scale_dtype,
        calc_scale_fn=recipe.calc_scale_fn,
        cast_to_dtype_fn=recipe.cast_to_dtype_fn,
    )

    # Inductor may reorder fp ops, causing tiny scale drift that flips a small
    # fraction of fp8 values by one quantization bin. Compare with tolerances
    # that accept one-bin drift but not algorithmic divergence.
    torch.testing.assert_close(scale_eager, scale_compiled, rtol=1e-2, atol=1e-6)
    bin_flip_frac = (
        qdata_eager.to(torch.float32) != qdata_compiled.to(torch.float32)
    ).float().mean().item()
    assert bin_flip_frac < 0.05, f"too many bin flips: {bin_flip_frac}"


def test_fuses_with_preceding_pointwise():
    recipe = next(r for r in RECIPES if r.name == "deepseek_fp8_1_128")

    def fn(x):
        return flex_quant_dense(
            F.relu(x),
            block_size=recipe.block_size,
            dim=recipe.dim,
            qdata_dtype=recipe.qdata_dtype,
            scale_dtype=recipe.scale_dtype,
            calc_scale_fn=recipe.calc_scale_fn,
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
