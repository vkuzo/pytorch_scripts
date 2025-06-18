"""
Further debugging for https://github.com/pytorch/pytorch/issues/149982
"""

from typing import Callable, Tuple

import torch
from torch._inductor.utils import do_bench_using_profiling
import fire

import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

torch.manual_seed(0)

def benchmark_cuda_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    """Thin wrapper around do_bench_using_profiling"""
    no_args = lambda: func(*args, **kwargs)
    time = do_bench_using_profiling(no_args)
    return time * 1e3

def compute_error(x, y):
    Ps = torch.linalg.norm(x)
    Pn = torch.linalg.norm(x - y)
    return 20 * torch.log10(Ps / Pn)



@triton.jit
def _triton_calculate_scale(x, axis):
    # There is no good support for accessing globals from a jit'ed triton
    # function, so we redefine them here. Since this is prototype code which
    # we plan to remove after torch.compile catches up, this is fine.
    target_max_pow2 = 8
    e8m0_exponent_bias = 127
    bf16_mbits = 7
    bf16_exp_bias = 127
    fp32_mbits = 23
    # We use a small epsilon to avoid division by zero
    epsilon = 1e-10

    # Find the maximum absolute value for each row
    max_abs = tl.max(x, axis=axis)

    # Real code would calculate the MX scale here, but for toy example
    # just return the max(abs(val)) to normalize
    return max_abs, max_abs


def _get_mxfp8_dim1_kernel_autotune_configs():
    # Values to sweep over here were determined by a manual
    # sweep over a small set of shapes, it's likely that this
    # can be improved in the future.
    results = []
    for ROW_TILE_SIZE in (64, 128):
        for COL_TILE_SIZE in (64, 128):
            for num_warps in (1, 2, 4):
                config = triton.Config(
                    {
                        "ROW_TILE_SIZE": ROW_TILE_SIZE,
                        "COL_TILE_SIZE": COL_TILE_SIZE,
                    },
                    num_warps=num_warps,
                )
                results.append(config)
    return results

@triton.autotune(
    configs=_get_mxfp8_dim1_kernel_autotune_configs(),
    key=["n_rows", "n_cols", "INNER_BLOCK_SIZE"],
)
@triton.jit
def to_mxfp8_dim1_kernel(
    x_ptr,  # pointer to input tensor
    output_col_major_ptr,  # pointer to column-major output tensor (column-normalized)
    col_scale_ptr,  # pointer to store column-wise maximum absolute values
    n_rows,  # number of rows in the tensor
    n_cols,  # number of columns in the tensor
    ROW_TILE_SIZE: tl.constexpr,
    COL_TILE_SIZE: tl.constexpr,
    INNER_BLOCK_SIZE: tl.constexpr,  # should be 32 for MX
):
    """
    Example tiling for n_rows==8, n_cols=8, ROW_TILE_SIZE=4, COL_TILE_SIZE=4, INNER_BLOCK_SIZE=2,
    pid_row=0, pid_col=0:

    Input (row-major)

    cols      0  1  2  3  4  5  6  7
    --------------------------------
    rows 0 |  0  1  2  3
         1 |  8  9 10 11
         2 | 16 17 18 19
         3 | 24 25 26 27
         4 |
         5 |
         6 |
         7 |

    Output (row-major of transpose), ids are from input

    cols      0  1  2  3  4  5  6  7
    --------------------------------
    rows 0 |  0  8 16 24
         1 |  1  9 17 25
         2 |  2 10 18 26
         3 |  3 11 19 27
         4 |
         5 |
         6 |
         7 |

    Output (scales), s(0, 8) means the scale used to cast elements 0 and 8

    rows           0          1  ...      4  ...       31
    ------------------------------------------------------
              s(0, 8)  s(16, 24) ... s(1, 9) ... s(19, 27)
    """

    BLOCKS_PER_ROW_TILE: tl.constexpr = ROW_TILE_SIZE // INNER_BLOCK_SIZE

    # Get program ID
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    # Calculate starting row and column for this tile
    start_row = pid_row * ROW_TILE_SIZE
    start_col = pid_col * COL_TILE_SIZE

    # Create offsets for the block
    row_offsets = tl.arange(0, ROW_TILE_SIZE)
    col_offsets = tl.arange(0, COL_TILE_SIZE)

    # Compute global row/col positions
    rows = start_row + row_offsets[:, None]  # Convert to 2D for proper broadcasting
    cols = start_col + col_offsets[None, :]

    # Create masks for out-of-bounds accesses
    row_mask = rows < n_rows
    col_mask = cols < n_cols
    mask = row_mask & col_mask

    # Compute memory offsets for row-major layout (rows, cols)
    row_major_offsets = (rows * n_cols + cols).to(tl.int32)

    # Compute memory offsets for column-major layout (cols, rows)
    col_major_offsets = (cols * n_rows + rows).to(tl.int32)

    # Load the entire block in a single operation
    # shape: (ROW_TILE_SIZE, COL_TILE_SIZE)
    x_block = tl.load(x_ptr + row_major_offsets, mask=mask)

    # Transpose dim0 and dim1
    # shape: (COL_TILE_SIZE, ROW_TILE_SIZE)
    x_block_t = tl.trans(x_block)

    # Reshape to inner tile size
    # shape: (COL_TILE_SIZE, ROW_TILE_SIZE) -> (COL_TILE_SIZE * BLOCKS_PER_ROW_TILE, INNER_BLOCK_SIZE)
    x_block_t_r = x_block_t.reshape(
        COL_TILE_SIZE * BLOCKS_PER_ROW_TILE, INNER_BLOCK_SIZE
    )

    # Calculate the absolute values of elements in the block
    x_block_abs_t_r = tl.abs(x_block_t_r)

    # Find the maximum absolute value for each column
    # shape: (COL_TILE_SIZE * BLOCKS_PER_ROW_TILE,)
    col_scale_r, col_scale_e8m0_r = _triton_calculate_scale(x_block_abs_t_r, axis=1)

    # Divide each column by scale
    # Broadcasting col_scale to match x_block's shape
    # x_block_t_r shape (COL_TILE_SIZE * BLOCKS_PER_ROW_TILE, INNER_BLOCK_SIZE)
    # col_scale shape (COL_TILE_SIZE * BLOCKS_PER_ROW_TILE,) -> (COL_TILE_SIZE * BLOCKS_PER_ROW_TILE, 1)
    col_normalized_t_r = x_block_t_r / col_scale_r[:, None]
    # col_normalized_t_r = x_block_t_r

    # Reshape back to original tile size
    col_normalized_t = tl.reshape(col_normalized_t_r, COL_TILE_SIZE, ROW_TILE_SIZE)

    # Undo the transpose
    col_normalized = tl.trans(col_normalized_t)

    # Quantize to float8
    col_normalized = col_normalized.to(tl.float8e4nv)

    # Store the column-normalized result in column-major format
    # TODO(future): this mask is for row-major likely need to transpose it for col-major
    tl.store(output_col_major_ptr + col_major_offsets, col_normalized, mask=mask)

    # reshape col_scale_e8m0_r to col_scale_e8m0
    # shape: (COL_TILE_SIZE * BLOCKS_PER_ROW_TILE,) -> (COL_TILE_SIZE, BLOCKS_PER_ROW_TILE,)
    col_scale_e8m0 = col_scale_e8m0_r.reshape(COL_TILE_SIZE * BLOCKS_PER_ROW_TILE)

    col_scale_start_offsets = (
        (pid_col * COL_TILE_SIZE * (n_rows // ROW_TILE_SIZE))
        * BLOCKS_PER_ROW_TILE  # number of blocks seen so far
        + pid_row * BLOCKS_PER_ROW_TILE  # increment BLOCKS_PER_ROW_TILE
    )

    col_scale_start_ptr = col_scale_ptr + col_scale_start_offsets

    # calculate col_scale_indices
    col_scale_indices = tl.arange(0, COL_TILE_SIZE * BLOCKS_PER_ROW_TILE)

    # How many values are in all the other columns for this row_pid, need to jump
    # over them for every BLOCKS_PER_ROW_TILE values
    jump_vals_per_col = (n_rows - ROW_TILE_SIZE) // INNER_BLOCK_SIZE

    # example transformation (specifics depend on tile sizes):
    # [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 1, 4, 5, 8, 9, 12, 13]
    col_scale_indices = col_scale_indices + (
        (col_scale_indices // BLOCKS_PER_ROW_TILE) * jump_vals_per_col
    )

    # TODO(future): mask this store
    tl.store(col_scale_start_ptr + col_scale_indices, col_scale_e8m0)

# @triton_op("torchao::triton_to_mxfp8_dim1", mutates_args={})
def triton_to_mxfp8_dim1(
    x: torch.Tensor, inner_block_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Input:
    * `x` - input tensor, in row major memory layout
    * `inner_block_size` - size of tiles to scale across, default is 32 for MX recipes

    Output:
    * `output_col_major`: the `float8_e4m3fn` values of `x` cast to mxfp8 across dim1
    * `col_scale`: the `e8m0` values of `x_scale` used to cast `x` to mxfp8 across dim1
    """
    assert x.is_contiguous(), "`x` must be contiguous"
    assert x.dtype == torch.bfloat16
    assert inner_block_size <= 32

    # Get tensor shape
    n_rows, n_cols = x.shape

    # Masking of loads and stores is not well tested yet, so for now enforce
    # shapes which do not need masking. Note that this condition depends on max values of
    # ROW_TILE_SIZE and COL_TILE_SIZE, which are autotuned above.
    # TODO(future): implement and test masking and remove this restriction
    max_row_tile_size = 128
    max_col_tile_size = 128
    assert n_rows % max_row_tile_size == 0, "unsupported"
    assert n_cols % max_col_tile_size == 0, "unsupported"

    # Create output tensors
    output_col_major = torch.empty(
        (n_cols, n_rows), dtype=torch.float8_e4m3fn, device=x.device
    )

    # Create scale tensors
    col_scale = torch.empty(
        (n_cols * n_rows // inner_block_size, 1), dtype=torch.bfloat16, device=x.device
    )

    # Calculate grid dimensions based on tile size
    grid = lambda META: (
        triton.cdiv(n_rows, META["ROW_TILE_SIZE"]),
        triton.cdiv(n_cols, META["COL_TILE_SIZE"]),
    )

    # Launch the kernel
    wrap_triton(to_mxfp8_dim1_kernel)[grid](
        x_ptr=x,
        output_col_major_ptr=output_col_major,
        col_scale_ptr=col_scale,
        n_rows=n_rows,
        n_cols=n_cols,
        INNER_BLOCK_SIZE=inner_block_size,
    )

    return (
        output_col_major.t(),
        col_scale,
    )


def scale_dim1_reference(x_hp: torch.Tensor, block_size) -> Tuple[torch.Tensor, torch.Tensor]:
    # normalize across dim1
    x_hp_d1 = x_hp.t().contiguous()
    x_hp_d1_block = x_hp_d1.reshape(-1, block_size)
    x_hp_d1_block_abs = x_hp_d1_block.abs()
    amax_dim1 = torch.amax(x_hp_d1_block_abs, dim=1).unsqueeze(1)
    x_hp_d1_block_normalized = (x_hp_d1_block.float() / amax_dim1.float()).to(torch.float8_e4m3fn)
    # x_hp_d1_block_normalized = (x_hp_d1_block).to(torch.float8_e4m3fn)
    x_hp_d1_normalized = x_hp_d1_block_normalized.reshape(x_hp_d1.shape)
    return x_hp_d1_normalized.t(), amax_dim1

bytes_per_el_bf16 = 2
bytes_per_el_fp8 = 1


def run(
    M: int = 16384, 
    K: int = 16384, 
    BLOCK_SIZE: int=32,
    mode: str = "compile",
):
    print(f'M {M} K {K} BLOCK_SIZE {BLOCK_SIZE}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'torch version: {torch.__version__}')
    print(f'triton version: {triton.__version__}')
    print(f'mode: {mode}')

    assert mode in ("perf_compile", "perf_triton", "validate_numerics")

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    if mode == "validate_numerics":
        x_d1_ref, amax_d1_ref = scale_dim1_reference(x, BLOCK_SIZE)
        x_d1_triton, amax_d1_triton = triton_to_mxfp8_dim1(x, BLOCK_SIZE)

        # not an exact match likely due to some subtle numerical differences
        # in type promotion, but close enough for this benchmarking script
        sqnr_data = compute_error(x_d1_ref.float(), x_d1_triton.float())
        print('sqnr_data', sqnr_data)
        sqnr_amax = compute_error(amax_d1_ref, amax_d1_triton)
        print('sqnr_amax', sqnr_amax)

        assert sqnr_data > 50.0
        assert sqnr_amax > 50.0
        return

    if mode == "perf_compile":
        scale_dim1_c = torch.compile(scale_dim1_reference)
    else:
        scale_dim1_c = triton_to_mxfp8_dim1

    x_d1, amax_d1 = scale_dim1_c(x, BLOCK_SIZE)

    # warm up
    for _ in range(2):
        __ = scale_dim1_reference(x, BLOCK_SIZE)
    time_us = benchmark_cuda_function_in_microseconds(lambda x, b: scale_dim1_c(x, b), x, BLOCK_SIZE)

    assert x.dtype == torch.bfloat16
    assert x_d1.dtype == torch.float8_e4m3fn
    assert amax_d1.dtype == torch.bfloat16
    # bytes_rw = (x.numel() + x_d1.numel() + amax_d1.numel()) * 2
    bytes_rw = (
        (x.numel() + amax_d1.numel()) * bytes_per_el_bf16 +
        x_d1.numel() * bytes_per_el_fp8
    )
    bytes_per_second = bytes_rw / (time_us / 1e6)
    gbytes_per_second = bytes_per_second / 1e9

    print('time_us', time_us)
    print('mem_bw_gbps', gbytes_per_second)


if __name__ == '__main__':
    fire.Fire(run)
