"""
See if we can optimize https://github.com/pytorch/ao/pull/1932/files by
swizzling the input memory
"""

import torch

# must be a power of 2 - 1, i.e. 1, 3, 7, 15, ...
SWIZZLE_CONSTANT = 3

def slow_index_load(tensor, indices_2d):
    # there is probably a better way to do this...
    assert tensor.numel() == indices_2d.numel()
    t_1d = tensor.reshape(-1)
    i_1d = indices_2d.reshape(-1)
    r_1d = t_1d[i_1d]
    return r_1d.reshape(tensor.shape)

def slow_index_write(tensor, indices_2d):
    new_tensor = torch.empty_like(tensor).reshape(-1)
    for row in range(tensor.shape[0]):
        for col in range(tensor.shape[1]):
            val = tensor[row][col]
            new_index = indices_2d[row][col]
            new_tensor[new_index] = val
    return new_tensor.reshape(tensor.shape)
            

def run():

    n_rows = 8
    n_cols = 8
    n = n_rows * n_cols

    x = torch.arange(n).reshape(n_rows, n_cols)
    print('x\n', x)

    row_offsets = torch.arange(n_rows)
    col_offsets = torch.arange(n_cols)

    rows = row_offsets[:, None]

    # the swizzle - use powers of 2 minus 1 - 1, 3, 7, etc
    col_offsets = col_offsets ^ (rows & 7)

    cols = col_offsets[None, :]
    print('rows\n', rows)
    print('cols\n', cols)

    row_major_offsets = (rows * n_cols + cols)
    col_major_offsets = (cols * n_rows + rows).squeeze(0)
    print('row_major_offsets\n', row_major_offsets)
    print('col_major_offsets\n', col_major_offsets)

    # read from row_major_offsets
    x_loaded_row_major = slow_index_load(x, row_major_offsets)
    print('x_loaded_row_major\n', x_loaded_row_major)

    # write to col_major_offsets
    y = slow_index_write(x_loaded_row_major, col_major_offsets)
    print('y\n', y)


    # indices = torch.arange(16).reshape(4, 4)
    # print('indices\n', indices)

    # indices_swizzled = indices ^ SWIZZLE_CONSTANT
    # print('indices_swizzled', indices_swizzled)

    # read swizzled indices
    # x_loaded = slow_index_load(x, indices_swizzled)
    # print('x_loaded', x_loaded)

    # transpose

if __name__ == '__main__':
    run()
