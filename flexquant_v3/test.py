"""Battle-test flex_cast_quant against a plain-PyTorch reference, recipe by recipe.

Comparison discipline mirrors flexquant v1/v2 test.py: bit-exact `torch.equal` on both
qdata (compared as fp32) and scale. Recipes live in recipes.py.
"""

import pytest
import torch
import torch.func._random as prng
import torch.nn.functional as F

from api import AuxKind, FlexCastQuantBackend, GlobalInputTransform, flex_cast_quant
from recipes import (
    DEEPSEEK_1X128,
    DEEPSEEK_1X128_DIM_M,
    DEEPSEEK_128X128,
    FLOAT8_TENSORWISE,
    MXFP8_BIAS,
    MXFP8_FLOOR,
    MXFP8_FLOOR_SWIZZLE,
    NVFP4_BLOCKED_OUTER,
    NVFP4_GS_SWIZZLE,
    float8_tensorwise_f,
    float8_tensorwise_scale,
    hadamard_rht_f,
    hadamard_rht_matrix,
    mxfp8_bias_f,
    nvfp4_blocked_outer_f,
    nvfp4_blocked_outer_scale,
    nvfp4_gs_scale,
    nvfp4_gs_swizzle_f,
    sr_bf16_f,
    sr_bf16_global_f,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


# (recipe_name, recipe, transform, scale shape+dtype, qdata_dtype, flat_compare, sqnr_min).
# transform: the global_input_transform enum. SWAP_0_AND_1_AXES (transpose-on-load) is how the
# dim-M recipe is expressed (plain deepseek_1x128_f on the swapped input) so `quant` sees the
# (K, M) orientation. Because the swap happens BEFORE tiling, dim-M stays tile-invariant and
# runs under MANUAL_TILE like the rest.
# qdata_dtype: fp4 packs two values per byte (float4_e2m1fn_x2) and is compared via its
# uint8 view; everything else compares as fp32 (see _qdata_equal).
# flat_compare: retained per recipe but now always False -- the swizzled scale is a 4D
# block grid (n_row_blocks, n_col_blocks, 32, 16), which is tile-invariant, so REFERENCE
# and MANUAL_TILE produce the SAME shape and compare bit-exact (no flatten needed).
# sqnr_min: fp8 e4m3 recipes clear ~20 dB; the mxfp8 e8m0 pow2 scale is coarser (15 dB).
# mxfp8_floor_swizzle scale: (256, 8) block scale -> nrb=2, ncb=2 -> (2, 2, 32, 16).
RECIPES = [
    ("deepseek_1x128", DEEPSEEK_1X128, GlobalInputTransform.NONE, (512, 512 // 128), torch.float32, torch.float8_e4m3fn, False, 20.0),
    ("deepseek_128x128", DEEPSEEK_128X128, GlobalInputTransform.NONE, (512 // 128, 512 // 128), torch.float32, torch.float8_e4m3fn, False, 20.0),
    ("deepseek_1x128_dim_m", DEEPSEEK_1X128_DIM_M, GlobalInputTransform.SWAP_0_AND_1_AXES, (512, 512 // 128), torch.float32, torch.float8_e4m3fn, False, 20.0),
    ("mxfp8_floor", MXFP8_FLOOR, GlobalInputTransform.NONE, (512, 512 // 32), torch.float8_e8m0fnu, torch.float8_e4m3fn, False, 15.0),
    ("mxfp8_floor_swizzle", MXFP8_FLOOR_SWIZZLE, GlobalInputTransform.NONE, (4, 4, 32, 16), torch.float8_e8m0fnu, torch.float8_e4m3fn, False, 15.0),
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
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    # scale computed outside flex_cast_quant, passed in as a REPLICATE aux input.
    scale = float8_tensorwise_scale(x)
    qdata, scale_out = flex_cast_quant(
        x,
        FLOAT8_TENSORWISE.quant,
        aux_inputs=(scale,),
        aux_kinds=(AuxKind.REPLICATE,),
        tile_multiple_of=FLOAT8_TENSORWISE.tile_multiple_of,
    )
    qdata_ref, scale_ref = float8_tensorwise_f(x, scale)

    # shapes / dtypes: scale is a single per-tensor scalar
    assert qdata.shape == (512, 512)
    assert qdata.dtype == torch.float8_e4m3fn
    assert scale_out.shape == ()
    assert scale_out.dtype == torch.float32

    # bit-exact vs reference (matches v1/v2 discipline)
    assert torch.equal(qdata.to(torch.float32), qdata_ref.to(torch.float32))
    assert torch.equal(scale_out, scale_ref)


def test_float8_tensorwise_sqnr_vs_high_precision():
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    scale = float8_tensorwise_scale(x)
    qdata, scale_out = flex_cast_quant(
        x, FLOAT8_TENSORWISE.quant, aux_inputs=(scale,), aux_kinds=(AuxKind.REPLICATE,)
    )
    x_hat = FLOAT8_TENSORWISE.dequant(qdata, scale_out)
    assert _compute_error(x.float(), x_hat.float()) > 20.0


# nvfp4 with global scale needs the runtime outer scale (a REPLICATE aux input), so -- like
# tensorwise -- it lives in dedicated tests rather than the static RECIPES table.
def test_nvfp4_gs_swizzle_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    # outer scale computed outside, passed in as a REPLICATE aux input.
    outer = nvfp4_gs_scale(x)
    qdata, scale = flex_cast_quant(
        x,
        NVFP4_GS_SWIZZLE.quant,
        aux_inputs=(outer,),
        aux_kinds=(AuxKind.REPLICATE,),
        tile_multiple_of=NVFP4_GS_SWIZZLE.tile_multiple_of,
        full_tile_multiple_of=NVFP4_GS_SWIZZLE.full_tile_multiple_of,
    )
    qdata_ref, scale_ref = nvfp4_gs_swizzle_f(x, outer)

    # shapes / dtypes: packed fp4 qdata + swizzled e4m3 inner scale as a 4D block grid.
    # inner scale is (256, 256//16) = (256, 16) -> nrb=2, ncb=4 -> (2, 4, 32, 16).
    assert qdata.shape == (512, 256)
    assert qdata.dtype == torch.float4_e2m1fn_x2
    assert scale.shape == (4, 8, 32, 16)
    assert scale.dtype == torch.float8_e4m3fn

    assert _qdata_equal(qdata, qdata_ref)
    assert torch.equal(scale, scale_ref)


def test_nvfp4_gs_swizzle_backends_match():
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    outer = nvfp4_gs_scale(x)
    kw = dict(
        aux_inputs=(outer,),
        aux_kinds=(AuxKind.REPLICATE,),
        tile_multiple_of=NVFP4_GS_SWIZZLE.tile_multiple_of,
        full_tile_multiple_of=NVFP4_GS_SWIZZLE.full_tile_multiple_of,
    )
    qdata_ref, scale_ref = flex_cast_quant(
        x, NVFP4_GS_SWIZZLE.quant, _backend=FlexCastQuantBackend.REFERENCE, **kw
    )
    qdata_tile, scale_tile = flex_cast_quant(
        x, NVFP4_GS_SWIZZLE.quant, _backend=FlexCastQuantBackend.MANUAL_TILE, **kw
    )

    # exercises the _manual_tile packed-fp4 cat (via uint8 view).
    assert _qdata_equal(qdata_tile, qdata_ref)
    # swizzled scale is a 4D block grid: tile-invariant, so same shape AND bit-exact.
    assert scale_tile.shape == scale_ref.shape
    assert torch.equal(scale_tile, scale_ref)


def test_nvfp4_gs_swizzle_sqnr_vs_high_precision():
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    outer = nvfp4_gs_scale(x)
    qdata, scale = flex_cast_quant(
        x,
        NVFP4_GS_SWIZZLE.quant,
        aux_inputs=(outer,),
        aux_kinds=(AuxKind.REPLICATE,),
        tile_multiple_of=NVFP4_GS_SWIZZLE.tile_multiple_of,
        full_tile_multiple_of=NVFP4_GS_SWIZZLE.full_tile_multiple_of,
    )
    x_hat = NVFP4_GS_SWIZZLE.dequant(qdata, scale, outer)
    # nvfp4 is 4-bit, coarser than fp8/mxfp8, so a lower SQNR floor.
    assert _compute_error(x.float(), x_hat.float()) > 12.0


# randomized Hadamard transform (RHT): a non-quant example. `f` returns a 1-tuple `(out,)`
# (no scale) and takes the RHT matrix as a REPLICATE aux input, identical across backends.
def _rht_sign_vector():
    torch.manual_seed(0)
    return torch.randint(0, 2, (16,), device="cuda") * 2 - 1  # length-16 +/-1


def _rht_matrix():
    return hadamard_rht_matrix(_rht_sign_vector(), "cuda", torch.bfloat16)


def test_hadamard_rht_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    rht = _rht_matrix()
    (out,) = flex_cast_quant(x, hadamard_rht_f, aux_inputs=(rht,), aux_kinds=(AuxKind.REPLICATE,))
    (out_ref,) = hadamard_rht_f(x, rht)

    assert out.shape == (512, 512)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, out_ref)


def test_hadamard_rht_backends_match():
    # single-output f: 256 // 2 == 128 is a multiple of 16, so quadrants don't sever a
    # 16-group -> tile invariant. Exercises the generalized single-output _manual_tile.
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    rht = _rht_matrix()
    kw = dict(aux_inputs=(rht,), aux_kinds=(AuxKind.REPLICATE,))
    (out_ref,) = flex_cast_quant(x, hadamard_rht_f, _backend=FlexCastQuantBackend.REFERENCE, **kw)
    (out_tile,) = flex_cast_quant(x, hadamard_rht_f, _backend=FlexCastQuantBackend.MANUAL_TILE, **kw)

    assert torch.equal(out_tile, out_ref)


def test_hadamard_rht_roundtrip_sqnr():
    # RHT is orthogonal, so its inverse is its transpose (NOT applying it twice).
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    rht = _rht_matrix()
    (y,) = flex_cast_quant(x, hadamard_rht_f, aux_inputs=(rht,), aux_kinds=(AuxKind.REPLICATE,))

    M, N = x.shape
    x_rec = (y.reshape(M, N // 16, 16) @ rht.t()).reshape(M, N)
    assert _compute_error(x.float(), x_rec.float()) > 25.0


# stochastic rounding fp32 -> bf16: non-quant, single-output `(out,)`, and by design NOT
# tile-invariant (tile-local RNG offsets repeat across tiles, so tiling changes rounding).
# The PRNG key is a tensor passed as a REPLICATE aux input (built via prng.key(seed)).
def test_sr_bf16_matches_reference():
    # determinism: same key -> bit-exact; different key -> differs.
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.float32, device="cuda")

    key0 = prng.key(0, device="cuda")
    (out,) = flex_cast_quant(x, sr_bf16_f, aux_inputs=(key0,), aux_kinds=(AuxKind.REPLICATE,))
    (out_ref,) = sr_bf16_f(x, key0)

    assert out.shape == (512, 512)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, out_ref)

    (out_other,) = sr_bf16_f(x, prng.key(1, device="cuda"))
    assert not torch.equal(out, out_other)


def test_sr_bf16_unbiased():
    # the defining SR property: outputs land on the two bracketing bf16 grid points and
    # E[SR(x)] ~= x. Pick a value strictly between two bf16 values (spacing near 1.0 is
    # 2**-7 ~= 0.0078).
    v = 1.0 + 0.003
    x = torch.full((1024, 1024), v, dtype=torch.float32, device="cuda")

    key0 = prng.key(0, device="cuda")
    (out,) = flex_cast_quant(x, sr_bf16_f, aux_inputs=(key0,), aux_kinds=(AuxKind.REPLICATE,))

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
    x = torch.full((512, 512), v, dtype=torch.float32, device="cuda")

    key0 = prng.key(0, device="cuda")
    kw = dict(aux_inputs=(key0,), aux_kinds=(AuxKind.REPLICATE,))
    (out_ref,) = flex_cast_quant(x, sr_bf16_f, _backend=FlexCastQuantBackend.REFERENCE, **kw)
    (out_tile,) = flex_cast_quant(x, sr_bf16_f, _backend=FlexCastQuantBackend.MANUAL_TILE, **kw)

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
    (out_ref,) = flex_cast_quant(x, sr_bf16_global_f, _backend=FlexCastQuantBackend.REFERENCE, **kw)
    (out_tile,) = flex_cast_quant(x, sr_bf16_global_f, _backend=FlexCastQuantBackend.MANUAL_TILE, **kw)

    assert torch.equal(out_ref, out_tile)  # global-position keying is tiling-invariant
    assert abs(out_ref.float().mean().item() - v) < 2e-3  # still unbiased


@pytest.mark.parametrize(
    "recipe, transform, scale_shape, scale_dtype, qdata_dtype",
    [(r, transform, scale_shape, scale_dtype, qdata_dtype) for _, r, transform, scale_shape, scale_dtype, qdata_dtype, _, _ in RECIPES],
    ids=[name for name, *_ in RECIPES],
)
def test_matches_reference(recipe, transform, scale_shape, scale_dtype, qdata_dtype):
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    qdata, scale = flex_cast_quant(
        x,
        recipe.quant,
        global_input_transform=transform,
        tile_multiple_of=recipe.tile_multiple_of,
        full_tile_multiple_of=recipe.full_tile_multiple_of,
    )
    # reference applies the same axis swap before running `quant`.
    swap = transform is GlobalInputTransform.SWAP_0_AND_1_AXES
    x_ref = x.t().contiguous() if swap else x
    qdata_ref, scale_ref = recipe.quant(x_ref)

    # shapes / dtypes
    assert qdata.dtype == qdata_dtype
    assert scale.shape == scale_shape
    assert scale.dtype == scale_dtype

    # bit-exact vs reference (matches v1/v2 discipline)
    assert _qdata_equal(qdata, qdata_ref)
    assert torch.equal(scale, scale_ref)


@pytest.mark.parametrize(
    "recipe, transform, flat_compare",
    [(r, transform, flat_compare) for _, r, transform, _, _, _, flat_compare, _ in RECIPES],
    ids=[name for name, *_ in RECIPES],
)
def test_backends_match(recipe, transform, flat_compare):
    # every recipe is tile-invariant, so the MANUAL_TILE backend must match REFERENCE
    # exactly. 256 // 2 == 128 keeps the quadrant split on a 128x128 tile boundary.
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    qdata_ref, scale_ref = flex_cast_quant(
        x,
        recipe.quant,
        global_input_transform=transform,
        _backend=FlexCastQuantBackend.REFERENCE,
        tile_multiple_of=recipe.tile_multiple_of,
        full_tile_multiple_of=recipe.full_tile_multiple_of,
    )
    qdata_tile, scale_tile = flex_cast_quant(
        x,
        recipe.quant,
        global_input_transform=transform,
        _backend=FlexCastQuantBackend.MANUAL_TILE,
        tile_multiple_of=recipe.tile_multiple_of,
        full_tile_multiple_of=recipe.full_tile_multiple_of,
    )

    assert _qdata_equal(qdata_tile, qdata_ref)
    # every scale (incl. the 4D swizzled block grid) is tile-invariant: same shape, bit-exact.
    del flat_compare  # retained in RECIPES for column layout; no longer needed here
    assert scale_tile.shape == scale_ref.shape
    assert torch.equal(scale_tile, scale_ref)


@pytest.mark.parametrize(
    "recipe, transform, sqnr_min",
    [(r, transform, sqnr_min) for _, r, transform, _, _, _, _, sqnr_min in RECIPES],
    ids=[name for name, *_ in RECIPES],
)
def test_sqnr_vs_high_precision(recipe, transform, sqnr_min):
    # dequantizing (qdata, scale) should recover the input with high SQNR.
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")

    qdata, scale = flex_cast_quant(
        x,
        recipe.quant,
        global_input_transform=transform,
        tile_multiple_of=recipe.tile_multiple_of,
        full_tile_multiple_of=recipe.full_tile_multiple_of,
    )
    x_hat = recipe.dequant(qdata, scale)
    # a swap quantized the transposed (K, M) tensor, so transpose the dequant back to
    # (M, N) to align with the original x.
    if transform is GlobalInputTransform.SWAP_0_AND_1_AXES:
        x_hat = x_hat.t()
    sqnr = _compute_error(x.float(), x_hat.float())
    assert sqnr > sqnr_min, f"{sqnr=} below {sqnr_min}"


# input padding (`pad_input_to_multiple_of`): a ragged input (e.g. LLM decode/prefill token
# dim) is zero-padded up to a multiple so the tile-invariant recipe sees an aligned shape.
# Outputs are returned at the PADDED shape (the swizzle scale grid is 128-row-atom-structured
# and can't be sliced back to an arbitrary original M). Pad multiples are chosen to satisfy
# each recipe's block/atom so the padded shape passes the existing constraint asserts.
def _ceil_to(v, m):
    return ((v + m - 1) // m) * m


def test_pad_ref_shapes_swizzle():
    # ragged 200x300 padded to (128,128)-multiple -> (256, 384); swizzle grid nrb=2, ncb=3.
    torch.manual_seed(0)
    x = torch.randn(200, 300, dtype=torch.bfloat16, device="cuda")
    qdata, scale = flex_cast_quant(
        x,
        MXFP8_FLOOR_SWIZZLE.quant,
        pad_input_to_multiple_of=(128, 128),
        tile_multiple_of=MXFP8_FLOOR_SWIZZLE.tile_multiple_of,
        full_tile_multiple_of=MXFP8_FLOOR_SWIZZLE.full_tile_multiple_of,
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
    kw = dict(
        pad_input_to_multiple_of=pad_to,
        tile_multiple_of=recipe.tile_multiple_of,
        full_tile_multiple_of=recipe.full_tile_multiple_of,
    )
    qdata_ref, scale_ref = flex_cast_quant(x, recipe.quant, _backend=FlexCastQuantBackend.REFERENCE, **kw)
    qdata_tile, scale_tile = flex_cast_quant(x, recipe.quant, _backend=FlexCastQuantBackend.MANUAL_TILE, **kw)
    assert _qdata_equal(qdata_tile, qdata_ref)
    assert scale_tile.shape == scale_ref.shape
    assert torch.equal(scale_tile, scale_ref)


def test_pad_matches_manual_pad():
    # padding inside the API == padding the input outside it, then running the recipe.
    torch.manual_seed(0)
    x = torch.randn(200, 300, dtype=torch.bfloat16, device="cuda")
    qdata, scale = flex_cast_quant(
        x,
        MXFP8_FLOOR.quant,
        pad_input_to_multiple_of=(1, 32),
        tile_multiple_of=MXFP8_FLOOR.tile_multiple_of,
    )
    # manual pad: 200 stays (mult of 1), 300 -> 320 (mult of 32); high-edge zero pad.
    x_padded = F.pad(x, (0, _ceil_to(300, 32) - 300, 0, 0))
    qdata_ref, scale_ref = MXFP8_FLOOR.quant(x_padded)
    assert _qdata_equal(qdata, qdata_ref)
    assert torch.equal(scale, scale_ref)


# aux inputs: REPLICATE and TILE are implemented; ROW/COL must still raise.
@pytest.mark.parametrize("kind", [AuxKind.ROW, AuxKind.COL])
def test_aux_unimplemented_kind_raises(kind):
    torch.manual_seed(0)
    x = torch.randn(512, 512, dtype=torch.bfloat16, device="cuda")
    scale = float8_tensorwise_scale(x)
    with pytest.raises(NotImplementedError):
        flex_cast_quant(x, FLOAT8_TENSORWISE.quant, aux_inputs=(scale,), aux_kinds=(kind,))


# AuxKind.TILE: nvfp4 with a 128x128-blocked outer scale. The framework slices the outer scale
# (2, 2) to the sub-block covering each tile; `f` block-broadcasts it. Divisor = 256//2 = 128.
def test_nvfp4_blocked_outer_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    outer = nvfp4_blocked_outer_scale(x)  # (2, 2)
    assert outer.shape == (2, 2)
    qdata, scale = flex_cast_quant(
        x,
        NVFP4_BLOCKED_OUTER.quant,
        aux_inputs=(outer,),
        aux_kinds=(AuxKind.TILE,),
        tile_multiple_of=NVFP4_BLOCKED_OUTER.tile_multiple_of,
        full_tile_multiple_of=NVFP4_BLOCKED_OUTER.full_tile_multiple_of,
    )
    qdata_ref, scale_ref = nvfp4_blocked_outer_f(x, outer)

    assert qdata.shape == (256, 128)
    assert qdata.dtype == torch.float4_e2m1fn_x2
    assert scale.shape == (2, 4, 32, 16)
    assert _qdata_equal(qdata, qdata_ref)
    assert torch.equal(scale, scale_ref)


def test_nvfp4_blocked_outer_backends_match():
    # the real proof TILE slicing + f's block-broadcast compose: REFERENCE == MANUAL_TILE.
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    outer = nvfp4_blocked_outer_scale(x)
    kw = dict(
        aux_inputs=(outer,),
        aux_kinds=(AuxKind.TILE,),
        tile_multiple_of=NVFP4_BLOCKED_OUTER.tile_multiple_of,
        full_tile_multiple_of=NVFP4_BLOCKED_OUTER.full_tile_multiple_of,
    )
    qdata_ref, scale_ref = flex_cast_quant(x, NVFP4_BLOCKED_OUTER.quant, _backend=FlexCastQuantBackend.REFERENCE, **kw)
    qdata_tile, scale_tile = flex_cast_quant(x, NVFP4_BLOCKED_OUTER.quant, _backend=FlexCastQuantBackend.MANUAL_TILE, **kw)

    assert _qdata_equal(qdata_tile, qdata_ref)
    assert scale_tile.shape == scale_ref.shape
    assert torch.equal(scale_tile, scale_ref)


def test_nvfp4_blocked_outer_sqnr_vs_high_precision():
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    outer = nvfp4_blocked_outer_scale(x)
    qdata, scale = flex_cast_quant(
        x,
        NVFP4_BLOCKED_OUTER.quant,
        aux_inputs=(outer,),
        aux_kinds=(AuxKind.TILE,),
        tile_multiple_of=NVFP4_BLOCKED_OUTER.tile_multiple_of,
        full_tile_multiple_of=NVFP4_BLOCKED_OUTER.full_tile_multiple_of,
    )
    x_hat = NVFP4_BLOCKED_OUTER.dequant(qdata, scale, outer)
    assert _compute_error(x.float(), x_hat.float()) > 12.0


# AuxKind.TILE with divisor (1, 1): an elementwise bias (same shape as input) added before mxfp8.
def test_mxfp8_bias_backends_match():
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")

    kw = dict(aux_inputs=(bias,), aux_kinds=(AuxKind.TILE,), tile_multiple_of=MXFP8_BIAS.tile_multiple_of)
    qdata_ref, scale_ref = flex_cast_quant(x, MXFP8_BIAS.quant, _backend=FlexCastQuantBackend.REFERENCE, **kw)
    qdata_tile, scale_tile = flex_cast_quant(x, MXFP8_BIAS.quant, _backend=FlexCastQuantBackend.MANUAL_TILE, **kw)

    # matches a direct (whole-tensor) bias-add + quant, and REFERENCE == MANUAL_TILE.
    qdata_direct, scale_direct = mxfp8_bias_f(x, bias)
    assert _qdata_equal(qdata_ref, qdata_direct)
    assert torch.equal(scale_ref, scale_direct)
    assert _qdata_equal(qdata_tile, qdata_ref)
    assert torch.equal(scale_tile, scale_ref)
