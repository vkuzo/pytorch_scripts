"""Battle-test flex_cast_quant against a plain-PyTorch reference, recipe by recipe.

Comparison discipline mirrors flexquant v1/v2 test.py: bit-exact `torch.equal` on both
qdata (compared as fp32) and scale. Recipes live in recipes.py.
"""

import pytest
import torch

from api import FlexCastQuantBackend, flex_cast_quant
from recipes import (
    DEEPSEEK_1X128,
    DEEPSEEK_1X128_DIM_M,
    DEEPSEEK_128X128,
    MXFP8_FLOOR,
    MXFP8_FLOOR_SWIZZLE,
    float8_tensorwise_scale,
    hadamard_rht_matrix,
    make_float8_tensorwise_recipe,
    make_hadamard_rht_f,
    make_nvfp4_gs_swizzle_recipe,
    make_sr_bf16_f,
    nvfp4_gs_scale,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


# (recipe_name, recipe, swap_axes, scale shape+dtype, qdata_dtype, flat_compare, sqnr_min).
# swap_axes: pass _swap_input_axes=True (transpose-on-load) so `quant` sees the (K, M)
# orientation -- this is how the dim-M recipe is expressed (plain deepseek_1x128_f on the
# swapped input). Because the swap happens BEFORE tiling, dim-M stays tile-invariant and
# runs under MANUAL_TILE like the rest.
# qdata_dtype: fp4 packs two values per byte (float4_e2m1fn_x2) and is compared via its
# uint8 view; everything else compares as fp32 (see _qdata_equal).
# flat_compare: retained per recipe but now always False -- the swizzled scale is a 4D
# block grid (n_row_blocks, n_col_blocks, 32, 16), which is tile-invariant, so REFERENCE
# and MANUAL_TILE produce the SAME shape and compare bit-exact (no flatten needed).
# sqnr_min: fp8 e4m3 recipes clear ~20 dB; the mxfp8 e8m0 pow2 scale is coarser (15 dB).
# mxfp8_floor_swizzle scale: (256, 8) block scale -> nrb=2, ncb=2 -> (2, 2, 32, 16).
RECIPES = [
    ("deepseek_1x128", DEEPSEEK_1X128, False, (256, 256 // 128), torch.float32, torch.float8_e4m3fn, False, 20.0),
    ("deepseek_128x128", DEEPSEEK_128X128, False, (256 // 128, 256 // 128), torch.float32, torch.float8_e4m3fn, False, 20.0),
    ("deepseek_1x128_dim_m", DEEPSEEK_1X128_DIM_M, True, (256, 256 // 128), torch.float32, torch.float8_e4m3fn, False, 20.0),
    ("mxfp8_floor", MXFP8_FLOOR, False, (256, 256 // 32), torch.float8_e8m0fnu, torch.float8_e4m3fn, False, 15.0),
    ("mxfp8_floor_swizzle", MXFP8_FLOOR_SWIZZLE, False, (2, 2, 32, 16), torch.float8_e8m0fnu, torch.float8_e4m3fn, False, 15.0),
]


def _compute_error(x, y):
    # torchao's `compute_error` (quantization/utils.py:63) -- SQNR in dB.
    return 20 * torch.log10(
        torch.linalg.vector_norm(x) / torch.linalg.vector_norm(x - y)
    )


def _qdata_equal(a, b):
    # dtype-aware bit-exact qdata compare: packed fp4 (float4_e2m1fn_x2) has no float cast,
    # so compare its raw bytes via the uint8 view; everything else compares as fp32.
    if a.dtype == torch.float4_e2m1fn_x2:
        return torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    return torch.equal(a.to(torch.float32), b.to(torch.float32))


def test_float8_tensorwise_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    # scale computed outside flex_cast_quant; the Recipe wraps only the cast kernel.
    scale = float8_tensorwise_scale(x)
    recipe = make_float8_tensorwise_recipe(scale)
    qdata, scale_out = flex_cast_quant(x, recipe.quant)
    qdata_ref, scale_ref = recipe.quant(x)

    # shapes / dtypes: scale is a single per-tensor scalar
    assert qdata.shape == (256, 256)
    assert qdata.dtype == torch.float8_e4m3fn
    assert scale_out.shape == ()
    assert scale_out.dtype == torch.float32

    # bit-exact vs reference (matches v1/v2 discipline)
    assert torch.equal(qdata.to(torch.float32), qdata_ref.to(torch.float32))
    assert torch.equal(scale_out, scale_ref)


def test_float8_tensorwise_sqnr_vs_high_precision():
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    scale = float8_tensorwise_scale(x)
    recipe = make_float8_tensorwise_recipe(scale)
    qdata, scale_out = flex_cast_quant(x, recipe.quant)
    x_hat = recipe.dequant(qdata, scale_out)
    assert _compute_error(x.float(), x_hat.float()) > 20.0


# nvfp4 with global scale is factory-bound (needs the runtime outer scale), so -- like
# tensorwise -- it lives in dedicated tests rather than the static RECIPES table.
def test_nvfp4_gs_swizzle_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    # outer scale computed outside; the Recipe wraps the inner cast + swizzle.
    outer = nvfp4_gs_scale(x)
    recipe = make_nvfp4_gs_swizzle_recipe(outer)
    qdata, scale = flex_cast_quant(x, recipe.quant)
    qdata_ref, scale_ref = recipe.quant(x)

    # shapes / dtypes: packed fp4 qdata + swizzled e4m3 inner scale as a 4D block grid.
    # inner scale is (256, 256//16) = (256, 16) -> nrb=2, ncb=4 -> (2, 4, 32, 16).
    assert qdata.shape == (256, 128)
    assert qdata.dtype == torch.float4_e2m1fn_x2
    assert scale.shape == (2, 4, 32, 16)
    assert scale.dtype == torch.float8_e4m3fn

    assert _qdata_equal(qdata, qdata_ref)
    assert torch.equal(scale, scale_ref)


def test_nvfp4_gs_swizzle_backends_match():
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    outer = nvfp4_gs_scale(x)
    recipe = make_nvfp4_gs_swizzle_recipe(outer)
    qdata_ref, scale_ref = flex_cast_quant(x, recipe.quant, _backend=FlexCastQuantBackend.REFERENCE)
    qdata_tile, scale_tile = flex_cast_quant(x, recipe.quant, _backend=FlexCastQuantBackend.MANUAL_TILE)

    # exercises the _manual_tile packed-fp4 cat (via uint8 view).
    assert _qdata_equal(qdata_tile, qdata_ref)
    # swizzled scale is a 4D block grid: tile-invariant, so same shape AND bit-exact.
    assert scale_tile.shape == scale_ref.shape
    assert torch.equal(scale_tile, scale_ref)


def test_nvfp4_gs_swizzle_sqnr_vs_high_precision():
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    outer = nvfp4_gs_scale(x)
    recipe = make_nvfp4_gs_swizzle_recipe(outer)
    qdata, scale = flex_cast_quant(x, recipe.quant)
    x_hat = recipe.dequant(qdata, scale)
    # nvfp4 is 4-bit, coarser than fp8/mxfp8, so a lower SQNR floor.
    assert _compute_error(x.float(), x_hat.float()) > 12.0


# randomized Hadamard transform (RHT): a non-quant example. `f` returns a 1-tuple `(out,)`
# (no scale) and closes over a fixed sign vector so it's identical across backends.
def _rht_sign_vector():
    torch.manual_seed(0)
    return torch.randint(0, 2, (16,), device="cuda") * 2 - 1  # length-16 +/-1


def test_hadamard_rht_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    f = make_hadamard_rht_f(_rht_sign_vector())
    (out,) = flex_cast_quant(x, f)
    (out_ref,) = f(x)

    assert out.shape == (256, 256)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, out_ref)


def test_hadamard_rht_backends_match():
    # single-output f: 256 // 2 == 128 is a multiple of 16, so quadrants don't sever a
    # 16-group -> tile invariant. Exercises the generalized single-output _manual_tile.
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    f = make_hadamard_rht_f(_rht_sign_vector())
    (out_ref,) = flex_cast_quant(x, f, _backend=FlexCastQuantBackend.REFERENCE)
    (out_tile,) = flex_cast_quant(x, f, _backend=FlexCastQuantBackend.MANUAL_TILE)

    assert torch.equal(out_tile, out_ref)


def test_hadamard_rht_roundtrip_sqnr():
    # RHT is orthogonal, so its inverse is its transpose (NOT applying it twice).
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    signs = _rht_sign_vector()
    f = make_hadamard_rht_f(signs)
    (y,) = flex_cast_quant(x, f)

    rht = hadamard_rht_matrix(signs, x.device, x.dtype)
    M, N = x.shape
    x_rec = (y.reshape(M, N // 16, 16) @ rht.t()).reshape(M, N)
    assert _compute_error(x.float(), x_rec.float()) > 25.0


# stochastic rounding fp32 -> bf16: non-quant, single-output `(out,)`, and by design NOT
# tile-invariant (tile-local RNG offsets repeat across tiles, so tiling changes rounding).
def test_sr_bf16_matches_reference():
    # determinism: same seed -> bit-exact; different seed -> differs.
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.float32, device="cuda")

    (out,) = flex_cast_quant(x, make_sr_bf16_f(0))
    (out_ref,) = make_sr_bf16_f(0)(x)

    assert out.shape == (256, 256)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, out_ref)

    (out_other,) = make_sr_bf16_f(1)(x)
    assert not torch.equal(out, out_other)


def test_sr_bf16_unbiased():
    # the defining SR property: outputs land on the two bracketing bf16 grid points and
    # E[SR(x)] ~= x. Pick a value strictly between two bf16 values (spacing near 1.0 is
    # 2**-7 ~= 0.0078).
    v = 1.0 + 0.003
    x = torch.full((1024, 1024), v, dtype=torch.float32, device="cuda")

    (out,) = flex_cast_quant(x, make_sr_bf16_f(0))

    lo = torch.tensor(v, dtype=torch.bfloat16).float().item()  # RTN neighbor (round down)
    hi = torch.tensor(v + 2**-7, dtype=torch.bfloat16).float().item()
    assert lo < v < hi
    uniq = set(out.float().unique().tolist())
    assert uniq <= {lo, hi}, f"unexpected values {uniq - {lo, hi}}"
    assert abs(out.float().mean().item() - v) < 1e-3


def test_sr_bf16_tiling_changes_rounding():
    # documents the accepted non-invariance: REFERENCE vs MANUAL_TILE differ bit-for-bit
    # (tile-local offsets repeat), yet both stay unbiased (mean ~= input).
    v = 1.0 + 0.003
    x = torch.full((256, 256), v, dtype=torch.float32, device="cuda")

    (out_ref,) = flex_cast_quant(x, make_sr_bf16_f(0), _backend=FlexCastQuantBackend.REFERENCE)
    (out_tile,) = flex_cast_quant(x, make_sr_bf16_f(0), _backend=FlexCastQuantBackend.MANUAL_TILE)

    assert not torch.equal(out_ref, out_tile)
    assert abs(out_ref.float().mean().item() - v) < 2e-3
    assert abs(out_tile.float().mean().item() - v) < 2e-3


@pytest.mark.parametrize(
    "recipe, swap_axes, scale_shape, scale_dtype, qdata_dtype",
    [(r, swap_axes, scale_shape, scale_dtype, qdata_dtype) for _, r, swap_axes, scale_shape, scale_dtype, qdata_dtype, _, _ in RECIPES],
    ids=[name for name, *_ in RECIPES],
)
def test_matches_reference(recipe, swap_axes, scale_shape, scale_dtype, qdata_dtype):
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    qdata, scale = flex_cast_quant(x, recipe.quant, _swap_input_axes=swap_axes)
    # reference applies the same axis swap before running `quant`.
    x_ref = x.t().contiguous() if swap_axes else x
    qdata_ref, scale_ref = recipe.quant(x_ref)

    # shapes / dtypes
    assert qdata.dtype == qdata_dtype
    assert scale.shape == scale_shape
    assert scale.dtype == scale_dtype

    # bit-exact vs reference (matches v1/v2 discipline)
    assert _qdata_equal(qdata, qdata_ref)
    assert torch.equal(scale, scale_ref)


@pytest.mark.parametrize(
    "recipe, swap_axes, flat_compare",
    [(r, swap_axes, flat_compare) for _, r, swap_axes, _, _, _, flat_compare, _ in RECIPES],
    ids=[name for name, *_ in RECIPES],
)
def test_backends_match(recipe, swap_axes, flat_compare):
    # every recipe is tile-invariant, so the MANUAL_TILE backend must match REFERENCE
    # exactly. 256 // 2 == 128 keeps the quadrant split on a 128x128 tile boundary.
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    qdata_ref, scale_ref = flex_cast_quant(
        x, recipe.quant, _swap_input_axes=swap_axes, _backend=FlexCastQuantBackend.REFERENCE
    )
    qdata_tile, scale_tile = flex_cast_quant(
        x, recipe.quant, _swap_input_axes=swap_axes, _backend=FlexCastQuantBackend.MANUAL_TILE
    )

    assert _qdata_equal(qdata_tile, qdata_ref)
    # every scale (incl. the 4D swizzled block grid) is tile-invariant: same shape, bit-exact.
    del flat_compare  # retained in RECIPES for column layout; no longer needed here
    assert scale_tile.shape == scale_ref.shape
    assert torch.equal(scale_tile, scale_ref)


@pytest.mark.parametrize(
    "recipe, swap_axes, sqnr_min",
    [(r, swap_axes, sqnr_min) for _, r, swap_axes, _, _, _, _, sqnr_min in RECIPES],
    ids=[name for name, *_ in RECIPES],
)
def test_sqnr_vs_high_precision(recipe, swap_axes, sqnr_min):
    # dequantizing (qdata, scale) should recover the input with high SQNR.
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    qdata, scale = flex_cast_quant(x, recipe.quant, _swap_input_axes=swap_axes)
    x_hat = recipe.dequant(qdata, scale)
    # swap_axes quantized the transposed (K, M) tensor, so transpose the dequant back to
    # (M, N) to align with the original x.
    if swap_axes:
        x_hat = x_hat.t()
    sqnr = _compute_error(x.float(), x_hat.float())
    assert sqnr > sqnr_min, f"{sqnr=} below {sqnr_min}"


# Directed regression for swizzle tile-invariance. The swizzled scale is a 4D block grid
# (n_row_blocks, n_col_blocks, 32, 16); its leading two axes are the block grid, so a
# column split reassembles on dim=1 and a row split on dim=0. This is the property the old
# flat-2D layout LACKED: serializing to (nrb*32, ncb*16) folded the (row-block, col-block)
# walk into the axes, so a column split silently reordered the buffer (and a 384-row input,
# split at 192 -- not a 128 multiple -- corrupted it even while "128-aligned"). We exercise
# column, row, and quadrant splits on shapes >256 so more than one col-block exists per
# band (the case that made column vs row disagree).
def _swizzle_recipes():
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
    mx = ("mxfp8_swizzle", MXFP8_FLOOR_SWIZZLE.quant)
    nv = ("nvfp4_swizzle", make_nvfp4_gs_swizzle_recipe(nvfp4_gs_scale(x)).quant)
    return x, [mx, nv]


@pytest.mark.parametrize("split", ["column", "row", "quadrant"])
def test_swizzle_scale_tile_invariant(split):
    x, recipes = _swizzle_recipes()
    for name, quant in recipes:
        _, scale_ref = quant(x)  # whole-tensor reference (the correct buffer)

        if split == "column":  # two 512x256 tiles, glue on the col-block axis (dim=1)
            a = quant(x[:, :256])[1]
            b = quant(x[:, 256:])[1]
            recomposed = torch.cat([a, b], dim=1)
        elif split == "row":  # two 256x512 tiles, glue on the row-block axis (dim=0)
            a = quant(x[:256, :])[1]
            b = quant(x[256:, :])[1]
            recomposed = torch.cat([a, b], dim=0)
        else:  # 2x2 quadrant (contains a column split -- the old buffer's worst case)
            ul = quant(x[:256, :256])[1]
            ur = quant(x[:256, 256:])[1]
            ll = quant(x[256:, :256])[1]
            lr = quant(x[256:, 256:])[1]
            top = torch.cat([ul, ur], dim=1)
            bottom = torch.cat([ll, lr], dim=1)
            recomposed = torch.cat([top, bottom], dim=0)

        assert recomposed.shape == scale_ref.shape, f"{name} {split}: shape mismatch"
        assert torch.equal(recomposed, scale_ref), f"{name} {split}: buffer mismatch"


def test_swizzle_384_aligned_split_invariant():
    # What the 4D grid DOES fix: buffer-order invariance under any atom-ALIGNED split. A
    # 384-row input splits cleanly at 256 (= 2*128) into a 256-row and a 128-row tile; both
    # are whole 128-row atoms, so the row-block axes concatenate to match the reference.
    torch.manual_seed(0)
    x = torch.randn(384, 256, dtype=torch.bfloat16, device="cuda")
    _, s_ref = MXFP8_FLOOR_SWIZZLE.quant(x)
    a = MXFP8_FLOOR_SWIZZLE.quant(x[:256, :])[1]
    b = MXFP8_FLOOR_SWIZZLE.quant(x[256:, :])[1]
    recomposed = torch.cat([a, b], dim=0)
    assert recomposed.shape == s_ref.shape
    assert torch.equal(recomposed, s_ref)


def test_swizzle_non_atom_aligned_split_is_out_of_scope():
    # What the 4D grid does NOT (and cannot) fix: a split that severs a 128-row atom. The
    # default MANUAL_TILE quadrant split of 384 lands at 192 (not a 128 multiple), so each
    # 192-row tile pads its partial row-block to 256 independently -> nrb=2+2=4 vs the
    # reference's ceil(384/128)=3. This is the option-(2) alignment contract (tile rows must
    # be a multiple of 128), orthogonal to buffer order, and is left unenforced for now.
    torch.manual_seed(0)
    x = torch.randn(384, 256, dtype=torch.bfloat16, device="cuda")
    _, s_ref = flex_cast_quant(x, MXFP8_FLOOR_SWIZZLE.quant, _backend=FlexCastQuantBackend.REFERENCE)
    _, s_tile = flex_cast_quant(x, MXFP8_FLOOR_SWIZZLE.quant, _backend=FlexCastQuantBackend.MANUAL_TILE)
    assert s_tile.shape != s_ref.shape  # documents the known, out-of-scope limitation


def test_swizzle_flatten_is_hardware_buffer():
    # the 4D grid must still serialize (once, outside f) to torchao's to_blocked buffer:
    # .reshape(-1) of the grid == the old flat layout. Guards the "serialize last" contract.
    from recipes import _to_blocked_4d, mxfp8_floor_f

    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
    _, s = mxfp8_floor_f(x)
    grid = _to_blocked_4d(s)
    nrb, ncb = grid.shape[0], grid.shape[1]
    # reference flat buffer built by the pre-refactor 2D path.
    blocks = s.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    flat_ref = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(nrb * 32, ncb * 16).reshape(-1)
    assert torch.equal(grid.reshape(-1), flat_ref)
