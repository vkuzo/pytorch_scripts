"""Plain Helion kernel for deepseek fp8 1x128 quantization (no user callbacks).

Baseline for the flexquant abstraction work: a hand-written Helion kernel for the
deepseek 1x128 cast, to check that plain Helion can hit good memory bandwidth on
this cast before layering a higher-order `tile_map` on top.

Recipe (from flexquant v1 recipes.py): for each 1x128 block along K,
    amax  = block.abs().amax().clamp(min=EPS)
    scale = amax / FP8_MAX            # forward scale (on-disk format)
    qdata = (block / scale).to(fp8)
Output: qdata (M, K) fp8_e4m3, scale (M, K//128) fp32.

Tiling: nested. Outer tile is autotuned over (M, K); the inner loop is pinned to
BLOCK_N so each quant block is one concrete-extent reduction. We do NOT reshape a
tile to expose the block group -- Helion cannot reshape a symbolic tile extent.
"""

import torch
import helion
import helion.language as hl

# From flexquant v1 recipes.py:15-16
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
EPS = 1e-12
BLOCK_N = 128


def _make_kernel(config=None):
    """Build the deepseek 1x128 kernel. config=None -> autotune (bench);
    a fixed helion.Config -> skip autotuning (fast tests)."""

    @helion.kernel(config=config)
    def deepseek_quant_1x128(x: torch.Tensor):
        M, K = x.size()
        qdata = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
        scale = torch.empty((M, K // BLOCK_N), dtype=torch.float32, device=x.device)
        # Constrain the outer K tile to be a power-of-two multiple of BLOCK_N: min
        # 128 -> the autotuner searches {128, 256, ...} and can never pick a
        # tile_n < BLOCK_N (which would silently reduce over a partial block).
        # NOTE 1: must be split into two 1D `hl.tile`s -- passing a registered
        # block size inside the 2D `hl.tile([M, K], block_size=[None, block_k])`
        # list form mis-parses (IncorrectTileUsage: min got u0).
        # NOTE 2: the `min` arg must be a LITERAL int (128), not the module global
        # BLOCK_N -- a global traces as a symbol (u0) and register_block_size
        # requires a constant min.
        block_k = hl.register_block_size(128, K)
        for tile_m in hl.tile(M):
            for tile_n in hl.tile(K, block_size=block_k):
                # inner loop pinned to the quant block -> concrete-extent
                # reduction, no symbolic in-tile reshape.
                for blk in hl.tile(tile_n.begin, tile_n.end, block_size=BLOCK_N):
                    acc = x[tile_m, blk]  # (tile_m, BLOCK_N)
                    amax = (
                        acc.abs()
                        .amax(dim=-1, keepdim=True)
                        .clamp(min=EPS)
                        .to(torch.float32)
                    )
                    s = amax / FP8_MAX
                    # IEEE-correct reciprocal. Triton lowers a plain `1.0 / s` to
                    # a reduced-precision reciprocal (~1 fp32 ULP high), which
                    # flips fp8 round-to-nearest-even on tie-adjacent values (1 in
                    # ~16.7M here). `tl.fdiv(..., ieee_rounding=True)` forces
                    # div.rn and makes qdata bit-exact vs the eager reference.
                    one = torch.ones_like(s)
                    recip = hl.inline_triton(
                        "tl.fdiv({a}, {b}, ieee_rounding=True)",
                        {"a": one, "b": s},
                        s,
                    )
                    qdata[tile_m, blk] = (acc.to(torch.float32) * recip).to(
                        torch.float8_e4m3fn
                    )
                    scale[tile_m, blk.begin // BLOCK_N] = s.squeeze(-1)
        return qdata, scale

    return deepseek_quant_1x128


# Autotuned kernel (used by bench and __main__).
deepseek_quant_1x128 = _make_kernel()

# Fixed-config kernel for fast tests (skips autotuning).
deepseek_quant_1x128_fixed = _make_kernel(
    helion.Config(block_sizes=[32, BLOCK_N], num_warps=4)
)


def _make_kernel_reshape(config=None):
    """Build a variant that exposes the 128-block as a real reduction AXIS.

    The block dim is introduced by a HOST-side reshape (x -> (M, NB, 128)), which
    is a free view on a concrete-shape tensor -- legal, unlike an in-kernel
    reshape of a symbolic tile. The kernel then tiles the (M, NB) output grid and
    reduces over the full last dim with `.amax(dim=-1)`. Because that axis is a
    real tensor dim Helion owns (not a hand-written loop), it enters the
    `reduction_loops` autotuner dimension -- the autotuner can choose a persistent
    reduction, like inductor does.
    """

    @helion.kernel(config=config)
    def _kernel(x3: torch.Tensor):  # x3: (M, NB, 128)
        M, NB, B = x3.size()
        qdata = torch.empty((M, NB, B), dtype=torch.float8_e4m3fn, device=x3.device)
        scale = torch.empty((M, NB), dtype=torch.float32, device=x3.device)
        for tile_m, tile_b in hl.tile([M, NB]):
            block = x3[tile_m, tile_b, :]  # (BM, BB, 128) -- full block via `:`
            amax = (
                block.abs()
                .amax(dim=-1, keepdim=True)
                .clamp(min=EPS)
                .to(torch.float32)
            )
            s = amax / FP8_MAX  # (BM, BB, 1)
            one = torch.ones_like(s)
            recip = hl.inline_triton(
                "tl.fdiv({a}, {b}, ieee_rounding=True)",
                {"a": one, "b": s},
                s,
            )
            qdata[tile_m, tile_b, :] = (block.to(torch.float32) * recip).to(
                torch.float8_e4m3fn
            )
            scale[tile_m, tile_b] = s.squeeze(-1)
        return qdata, scale

    def wrapper(x: torch.Tensor):
        # Host-side reshape to expose the block dim (free view; K contiguous).
        M, K = x.shape
        x3 = x.reshape(M, K // BLOCK_N, BLOCK_N)
        qdata3, scale = _kernel(x3)
        return qdata3.reshape(M, K), scale

    return wrapper


# Reshape-axis variant: autotuned + fixed-config.
deepseek_quant_1x128_reshape = _make_kernel_reshape()
deepseek_quant_1x128_reshape_fixed = _make_kernel_reshape(
    helion.Config(block_sizes=[32, 1], num_warps=4)
)


def deepseek_quant_1x128_reference(x: torch.Tensor):
    """Plain-PyTorch reference (v1 recipes.py:60-70 _deepseek_fp8_1_128_reference)."""
    M, K = x.shape
    n_blocks = K // BLOCK_N
    x_b = x.reshape(M, n_blocks, BLOCK_N)
    amax = x_b.abs().amax(dim=-1, keepdim=True).clamp(min=EPS).to(torch.float32)
    scale = (amax / FP8_MAX).to(torch.float32)
    y = x_b.to(torch.float32) * (1.0 / scale)
    qdata = y.to(torch.float8_e4m3fn).reshape(M, K)
    return qdata, scale.squeeze(-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
    qdata, scale = deepseek_quant_1x128_fixed(x)
    print(f"x:     {tuple(x.shape)} {x.dtype}")
    print(f"qdata: {tuple(qdata.shape)} {qdata.dtype}")
    print(f"scale: {tuple(scale.shape)} {scale.dtype}")
