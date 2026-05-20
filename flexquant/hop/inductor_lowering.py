"""Inductor lowering for FlexQuantHOP.

Picks the FlexQuant Triton template, builds subgraph buffers from the two
captured callbacks, and registers a kernel choice via maybe_append_choice.
Modeled after torch._inductor.kernel.flex.flex_attention:flex_attention.
"""

import os
from typing import Any

import torch
from torch._inductor.ir import FixedLayout
from torch._inductor.kernel.flex.common import (
    build_subgraph_buffer,
    create_placeholder,
    freeze_irnodes,
)
from torch._inductor.lowering import empty_strided, register_lowering
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    SymbolicGridFn,
    TritonTemplate,
)

from .hop import flex_quant


_HERE = os.path.dirname(__file__)
with open(os.path.join(_HERE, "template.py.jinja")) as _f:
    _TEMPLATE_SOURCE = _f.read()


@SymbolicGridFn
def _flex_quant_grid(M, N, meta, *, cdiv):
    return (cdiv(M, meta["BLOCK_SIZE"]), cdiv(N, meta["BLOCK_SIZE"]), 1)


flex_quant_template = TritonTemplate(
    name="flex_quant",
    grid=_flex_quant_grid,
    source=_TEMPLATE_SOURCE,
)


_AUTOTUNE_CONFIGS = [
    {"BLOCK_SIZE": 128, "num_warps": w, "num_stages": s}
    for w in (4, 8)
    for s in (2, 4)
]


@register_lowering(flex_quant, type_promotion_kind=None)
def _flex_quant_lowering(
    x,
    amax_subgraph,
    cast_subgraph,
    block_size,
    qdata_dtype,
    scale_dtype,
):
    """Lowering for the FlexQuantHOP. Only handles the 128x128 e4m3fn+fp32
    case; raises otherwise (no fallback for now)."""

    # Static config sanity checks.
    assert tuple(block_size) == (128, 128), f"block_size {block_size!r} unsupported"
    # Note: qdata_dtype is parameterized so we can A/B fp8 vs fp32 output during
    # debugging; production scope is fp8_e4m3fn / fp32.
    assert qdata_dtype in (torch.float8_e4m3fn, torch.float32)
    assert scale_dtype == torch.float32

    device = x.get_device()
    M = x.get_size()[0]
    N = x.get_size()[1]
    B1, B2 = block_size

    # Output layouts. qdata is (M, N) row-major in qdata_dtype; scale is
    # (M//B1, N//B2) row-major in scale_dtype.
    qdata_layout = FixedLayout(
        device,
        qdata_dtype,
        [M, N],
        stride=[N, 1],
    )
    n1 = (M + B1 - 1) // B1
    n2 = (N + B2 - 1) // B2
    # Allocate a scale buffer that will be mutated in-place by the kernel.
    scale = empty_strided([n1, n2], None, dtype=scale_dtype, device=device)

    # Build the subgraph buffers. The placeholder names ("amax", "tile",
    # "scale") match the kwargs to {{ modification(...) }} in the template.
    amax_placeholders = [create_placeholder("amax", x.get_dtype(), device)]
    cast_placeholders = [
        create_placeholder("tile", x.get_dtype(), device),
        create_placeholder("scale", scale_dtype, device),
    ]

    amax_buffer = build_subgraph_buffer(amax_placeholders, amax_subgraph)
    freeze_irnodes(amax_buffer)
    cast_buffer = build_subgraph_buffer(cast_placeholders, cast_subgraph)
    freeze_irnodes(cast_buffer)

    choices: list[Any] = []
    for cfg in _AUTOTUNE_CONFIGS:
        flex_quant_template.maybe_append_choice(
            choices=choices,
            input_nodes=[x, scale],
            layout=qdata_layout,
            subgraphs=[amax_buffer, cast_buffer],
            mutated_inputs=[scale],
            call_sizes=[M, N],
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    qdata, _ = autotune_select_algorithm(
        "flex_quant",
        choices,
        [x, scale],
        qdata_layout,
    )

    return (qdata, scale)
