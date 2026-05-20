"""Helion port of the deepseek 128x128 weight-quant kernel.

Mirrors the reference body in `recipes.py:_deepseek_fp8_128_128_reference`:
each (128, 128) tile of the input is reduced to its abs-max, that scalar is
divided by FP8_MAX to produce the forward scale, and each element of the tile
is cast to fp8_e4m3fn after multiplying by the reciprocal of the scale.

Outputs:
- `qdata` shape (M, K), dtype fp8_e4m3fn, row-major.
- `scale` shape (M // 128, K // 128), dtype fp32, row-major.
"""

from typing import Callable, Tuple

import helion
import helion.language as hl
import torch

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
EPS = 1e-12


# Config picked by helion's autotuner on B200 for shape (16384, 16384) bf16.
# Hard-coded so benchmark runs aren't paying the ~90s autotune cost each
# invocation. Re-run with `@helion.kernel` (no config) to re-tune.
_AUTOTUNED_CONFIG = helion.Config(
    block_sizes=[],
    indexing=["pointer", "pointer", "tensor_descriptor"],
    l2_groupings=[8],
    load_eviction_policies=[""],
    loop_orders=[[1, 0]],
    num_stages=8,
    num_warps=32,
    pid_type="flat",
    range_flattens=[None],
    range_multi_buffers=[None],
    range_num_stages=[0],
    range_unroll_factors=[0],
    range_warp_specializes=[None],
)


@helion.kernel(config=_AUTOTUNED_CONFIG, static_shapes=True)
def deepseek_fp8_128_128_helion(
    x: torch.Tensor,
    amax_to_scale_fn: Callable,
    cast_to_dtype_fn: Callable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, K = x.shape
    n1, n2 = M // 128, K // 128
    qdata = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty(n1, n2, dtype=torch.float32, device=x.device)

    for tile_m, tile_n in hl.tile([M, K], block_size=[128, 128]):
        x_tile = x[tile_m, tile_n]
        # Reduce in two steps; helion's lowering doesn't accept reduction over
        # multiple dims in one call.
        amax_row = torch.amax(torch.abs(x_tile), dim=1)
        amax = torch.amax(amax_row, dim=0)
        s = amax_to_scale_fn(amax)
        qdata[tile_m, tile_n] = cast_to_dtype_fn(x_tile, s)
        scale[tile_m.id, tile_n.id] = s

    return qdata, scale
