"""
A simple triton kernel to calculate max(tensor) with atomics. This avoids the
need to do a two-stage reduction.
"""

import torch
import triton
from triton import language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 8192}),
        triton.Config({"BLOCK_SIZE": 16384}),
    ],
    key=["n_elements"],
)
@triton.jit
def max_with_atomics_kernel(
    in_ptr0,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    x_max = tl.max(x)
    tl.atomic_max(out_ptr, x_max)


def max_with_atomics(x: torch.Tensor):
    # Note: need to initialize to zero for numerical correctness of max
    output = torch.zeros(1, dtype=x.dtype, device=x.device)
    n_elements = x.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    max_with_atomics_kernel[grid](x, output, n_elements)
    return output.squeeze()
