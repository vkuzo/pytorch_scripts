"""Battle-test flex_tile_map against a plain-PyTorch reference, recipe by recipe.

Comparison discipline mirrors flexquant v1/v2 test.py: bit-exact `torch.equal` on both
qdata (compared as fp32) and scale. Recipes live in recipes.py.
"""

import os
import sys

import pytest
import torch
import torch.func._random as prng
import torch.nn.functional as F

from api import AuxKind, FlexTileMapBackend, OutputKind, flex_tile_map
from recipes import (
    DEEPSEEK_1X128,
    DEEPSEEK_1X128_DIM_M,
    MXFP8_FLOOR,
    MXFP8_FLOOR_SWIZZLE,
    RECIPES_V2,
)

# The SR variants (`sr_bf16_f` tile-local, `sr_bf16_global_f` tiling-invariant) live in
# quant_cast_gold (SrF32ToBf16 / SrF32ToBf16Global); the dedicated SR tests use the fns directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_cast_gold.recipes import sr_bf16_f, sr_bf16_global_f

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _recipe_input(name):
    # the input `x` the generic RECIPES_V2 tests feed each recipe. The SR recipes (both the
    # tile-local sr_bf16 and the tiling-invariant sr_bf16_global) are the exception: their `f`
    # asserts fp32 (not the default bf16), and their correctness_fn checks unbiasedness on a
    # CONSTANT value strictly between two bf16 grid points (spacing 2**-7 near 1.0). Everything
    # else uses the default bf16 randn.
    if name in ("sr_bf16", "sr_bf16_global"):
        return torch.full((512, 512), 1.0 + 0.003, dtype=torch.float32, device="cuda")
    return torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")


def _qdata_equal(a, b):
    # dtype-aware bit-exact qdata compare: packed fp4 (float4_e2m1fn_x2) has no float cast,
    # so compare its raw bytes via the uint8 view; everything else compares as fp32.
    if a.dtype == torch.float4_e2m1fn_x2:
        return torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    return torch.equal(a.to(torch.float32), b.to(torch.float32))


@pytest.mark.parametrize(
    "name, recipe",
    RECIPES_V2,
    ids=[name for name, _ in RECIPES_V2],
)
def test_ref_correctness(name, recipe):
    # tests that running correctness_fn on the outputs of the reference fn itself passes --
    # i.e. the gold recipe is internally consistent (pt_ref_fn's outputs clear its own
    # correctness_fn), independent of flex_tile_map. Mirrors test_flex_tile_map_correctness
    # but calls pt_ref_fn directly on the whole tensor instead of tiling it.
    torch.manual_seed(0)
    x = _recipe_input(name)

    aux = recipe.aux_fn(x) if recipe.aux_fn is not None else ()
    inputs = (x, *aux)

    # the whole tensor is one tile: pass the same position kwargs the REFERENCE backend would
    # (origin (0,0), full row stride). Recipes that ignore them (all but sr_bf16_global) accept
    # **kwargs; sr_bf16_global needs them to key its per-element global-position dither.
    outputs = recipe.pt_ref_fn(*inputs, global_row=0, global_col=0, num_col=x.shape[1])
    recipe.correctness_fn(inputs, outputs)  # raises AssertionError on failure


@pytest.mark.parametrize(
    "name, recipe",
    RECIPES_V2,
    ids=[name for name, _ in RECIPES_V2],
)
def test_flex_tile_map_ref_correctness(name, recipe):
    # tests that running correctness_fn on the outputs of flex_tile_map passes

    torch.manual_seed(0)
    x = _recipe_input(name)

    # aux_fn (if any) precomputes extra positional args pt_ref_fn takes beyond `x` (e.g. a
    # precalculated scale); inputs is the full tuple correctness_fn compares outputs against.
    aux = recipe.aux_fn(x) if recipe.aux_fn is not None else ()
    inputs = (x, *aux)

    outputs = flex_tile_map(
        x,
        recipe.pt_ref_fn,
        aux_inputs=aux,
        aux_kinds=recipe.aux_kinds,
        output_kinds=recipe.output_kinds,
        valid_tile_size_fn=recipe.valid_tile_size_fn,
    )
    recipe.correctness_fn(inputs, outputs)  # raises AssertionError on failure

@pytest.mark.parametrize(
    "name, recipe",
    RECIPES_V2,
    ids=[name for name, _ in RECIPES_V2],
)
def test_flex_tile_map_backends_keep_numerics(name, recipe):
    # every RecipeV2 is tile-invariant, so the MANUAL_TILE backend must produce bit-identical
    # outputs to REFERENCE. Compares every output tensor (qdata + any scale/aux outputs)
    # exactly via _qdata_equal (packed fp4 via its uint8 view; everything else -- fp8_e4m3,
    # e8m0, fp32, 4D swizzle grids -- as a bit-exact fp32 compare).
    #
    # the SR recipes are skipped here; both keep their REFERENCE-vs-MANUAL_TILE behavior in
    # dedicated tests. sr_bf16 is the NON-tile-invariant counterexample (dither keyed on
    # tile-local order, so MANUAL_TILE != REFERENCE by design -- test_sr_bf16_tiling_changes_rounding).
    # sr_bf16_global IS tile-invariant (keyed on global position); that equality is asserted by
    # test_sr_bf16_global_tiling_invariant, so it's skipped here too rather than duplicated.
    if name in ("sr_bf16", "sr_bf16_global"):
        pytest.skip(f"{name}: REFERENCE-vs-MANUAL_TILE behavior is covered by a dedicated SR test")

    torch.manual_seed(0)
    x = _recipe_input(name)

    aux = recipe.aux_fn(x) if recipe.aux_fn is not None else ()
    kw = dict(
        aux_inputs=aux,
        aux_kinds=recipe.aux_kinds,
        output_kinds=recipe.output_kinds,
        valid_tile_size_fn=recipe.valid_tile_size_fn,
    )
    ref = flex_tile_map(x, recipe.pt_ref_fn, _backend=FlexTileMapBackend.REFERENCE, **kw)
    tile = flex_tile_map(x, recipe.pt_ref_fn, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)

    assert len(ref) == len(tile), f"{name}: output count {len(tile)} != {len(ref)}"
    for i, (r, t) in enumerate(zip(ref, tile)):
        assert r.shape == t.shape and r.dtype == t.dtype, (
            f"{name} output {i}: shape/dtype mismatch ({t.shape}/{t.dtype} vs {r.shape}/{r.dtype})"
        )
        assert _qdata_equal(t, r), f"{name} output {i}: MANUAL_TILE differs from REFERENCE"


# dim-M deepseek: `f` transposes the tile + reduces last dim, and OutputKind.SWAP_TILE_INDEX
# grid-transposes the placement. Together they reproduce deepseek_1x128_f(x.t()) -- the dim-M
# layout that used to be expressed by the removed global_input_transform=SWAP_0_AND_1_AXES.
_DIM_M_SWAP = (OutputKind.SWAP_TILE_INDEX, OutputKind.SWAP_TILE_INDEX)


# dim-M whole-tensor correctness and REFERENCE == MANUAL_TILE (square) are covered by the
# generic RECIPES_V2 suite (DEEPSEEK_1X128_DIM_M carries output_kinds=SWAP_TILE_INDEX). The
# non-square case below is kept: it uniquely exercises the grid-transpose with P != Q.
def test_deepseek_dim_m_non_square():
    # non-square input exercises the grid-transpose (P != Q): a 384x512 input produces a
    # (512, 384) qdata / (512, 3) scale swapped-grid output; REFERENCE == MANUAL_TILE bit-exact.
    torch.manual_seed(0)
    x = torch.randn(384, 512, dtype=torch.bfloat16, device="cuda")

    kernel = DEEPSEEK_1X128_DIM_M.pt_ref_fn
    kw = dict(output_kinds=_DIM_M_SWAP, valid_tile_size_fn=DEEPSEEK_1X128_DIM_M.valid_tile_size_fn)
    qr, sr = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.REFERENCE, **kw)
    qt, st = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)
    assert qr.shape == (512, 384)  # grid-transposed
    assert sr.shape == (512, 384 // 128)
    assert _qdata_equal(qt, qr)
    assert torch.equal(st, sr)


# input padding (`pad_input_to_multiple_of`): a ragged input (e.g. LLM decode/prefill token
# dim) is zero-padded up to a multiple so the tile-invariant recipe sees an aligned shape.
# Outputs are returned at the PADDED shape (the swizzle scale grid is 128-row-atom-structured
# and can't be sliced back to an arbitrary original M). Pad multiples are chosen to satisfy
# each recipe's block/atom so the padded shape passes the existing constraint asserts.
def _ceil_to(v, m):
    return ((v + m - 1) // m) * m


def test_valid_tile_size_fn_unsatisfiable_raises_then_pad_fixes():
    # deepseek's predicate (actual[1] % 128 == 0) can't be satisfied on a ragged 512x300 (the
    # 44-wide edge fails, and spanning 300 fails too) -> the tile-size search raises. Padding the
    # columns up to a multiple of 128 makes it satisfiable.
    torch.manual_seed(0)
    x = torch.randn(512, 300, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(ValueError):
        flex_tile_map(
            x,
            DEEPSEEK_1X128.pt_ref_fn,
            valid_tile_size_fn=DEEPSEEK_1X128.valid_tile_size_fn,
            _backend=FlexTileMapBackend.MANUAL_TILE,
        )

    # pad N 300 -> 384 (multiple of 128); now every tile's column extent is 128-aligned.
    qdata, scale = flex_tile_map(
        x,
        DEEPSEEK_1X128.pt_ref_fn,
        valid_tile_size_fn=DEEPSEEK_1X128.valid_tile_size_fn,
        pad_input_to_multiple_of=(1, 128),
        _backend=FlexTileMapBackend.MANUAL_TILE,
    )
    assert qdata.shape == (512, 384)  # returned at the padded shape


def test_pad_ref_shapes_swizzle():
    # ragged 200x300 padded to (128,128)-multiple -> (256, 384); swizzle grid nrb=2, ncb=3.
    torch.manual_seed(0)
    x = torch.randn(200, 300, dtype=torch.bfloat16, device="cuda")
    qdata, scale = flex_tile_map(
        x,
        MXFP8_FLOOR_SWIZZLE.pt_ref_fn,
        pad_input_to_multiple_of=(128, 128),
        valid_tile_size_fn=MXFP8_FLOOR_SWIZZLE.valid_tile_size_fn,
    )
    assert qdata.shape == (256, 384)
    assert scale.shape == (2, 3, 32, 16)


@pytest.mark.parametrize(
    "recipe, pad_to",
    [
        (MXFP8_FLOOR, (1, 32)),
        (MXFP8_FLOOR_SWIZZLE, (128, 128)),
        (DEEPSEEK_1X128, (1, 128)),
    ],
    ids=["mxfp8_floor", "mxfp8_floor_swizzle", "deepseek_1x128"],
)
def test_pad_backends_match(recipe, pad_to):
    # padded ragged input: MANUAL_TILE must match REFERENCE bit-exact (padding happens before
    # tiling in both paths, so the two backends see the identical padded tensor).
    torch.manual_seed(0)
    x = torch.randn(200, 300, dtype=torch.bfloat16, device="cuda")
    kernel = recipe.pt_ref_fn
    kw = dict(
        pad_input_to_multiple_of=pad_to,
        valid_tile_size_fn=recipe.valid_tile_size_fn,
    )
    qdata_ref, scale_ref = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.REFERENCE, **kw)
    qdata_tile, scale_tile = flex_tile_map(x, kernel, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)
    assert _qdata_equal(qdata_tile, qdata_ref)
    assert scale_tile.shape == scale_ref.shape
    assert torch.equal(scale_tile, scale_ref)


def test_pad_matches_manual_pad():
    # padding inside the API == padding the input outside it, then running the recipe.
    torch.manual_seed(0)
    x = torch.randn(200, 300, dtype=torch.bfloat16, device="cuda")
    kernel = MXFP8_FLOOR.pt_ref_fn
    qdata, scale = flex_tile_map(
        x,
        kernel,
        pad_input_to_multiple_of=(1, 32),
        valid_tile_size_fn=MXFP8_FLOOR.valid_tile_size_fn,
    )
    # manual pad: 200 stays (mult of 1), 300 -> 320 (mult of 32); high-edge zero pad.
    x_padded = F.pad(x, (0, _ceil_to(300, 32) - 300, 0, 0))
    qdata_ref, scale_ref = kernel(x_padded)
    assert _qdata_equal(qdata, qdata_ref)
    assert torch.equal(scale, scale_ref)


def test_sr_bf16_tiling_changes_rounding():
    # documents the accepted non-invariance: REFERENCE vs MANUAL_TILE differ bit-for-bit
    # (tile-local offsets repeat), yet both stay unbiased (mean ~= input).
    v = 1.0 + 0.003
    x = torch.full((512, 512), v, dtype=torch.float32, device="cuda")

    key0 = prng.key(0, device="cuda")
    kw = dict(aux_inputs=(key0,), aux_kinds=(AuxKind.REPLICATE,))
    (out_ref,) = flex_tile_map(x, sr_bf16_f, _backend=FlexTileMapBackend.REFERENCE, **kw)
    (out_tile,) = flex_tile_map(x, sr_bf16_f, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)

    assert not torch.equal(out_ref, out_tile)
    assert abs(out_ref.float().mean().item() - v) < 2e-3
    assert abs(out_tile.float().mean().item() - v) < 2e-3


def test_sr_bf16_global_tiling_invariant():
    # the tiling-invariant SR: keyed on GLOBAL element position, so REFERENCE == MANUAL_TILE
    # bit-for-bit (contrast test_sr_bf16_tiling_changes_rounding, which uses the tile-local key).
    v = 1.0 + 0.003
    x = torch.full((512, 512), v, dtype=torch.float32, device="cuda")

    key0 = prng.key(0, device="cuda")
    kw = dict(aux_inputs=(key0,), aux_kinds=(AuxKind.REPLICATE,))
    (out_ref,) = flex_tile_map(x, sr_bf16_global_f, _backend=FlexTileMapBackend.REFERENCE, **kw)
    (out_tile,) = flex_tile_map(x, sr_bf16_global_f, _backend=FlexTileMapBackend.MANUAL_TILE, **kw)

    assert torch.equal(out_ref, out_tile)  # global-position keying is tiling-invariant
    assert abs(out_ref.float().mean().item() - v) < 2e-3  # still unbiased
