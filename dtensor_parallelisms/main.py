"""
Working through basic parallelisms with DTensor

run with

  torchrun --nproc_per_node 2 main.py
"""

import copy
import fire
import torch
import torch.nn as nn
import os

from torch.distributed import get_rank
from torch.distributed._tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

# torch.set_float32_matmul_precision("high")
torch.backends.fp32_precision = "ieee"

# this is set to 1 by default, set it here to avoid warning in stdout
os.environ['OMP_NUM_THREADS'] = '1'

def print0(*args, **kwargs):
    if not get_rank() == 0:
        return
    print(*args, **kwargs)

def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    device_mesh = init_device_mesh("cuda", (world_size,))
    # seed must be the same in all processes
    torch.manual_seed(1)
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    return device_mesh

def run(mode: str):

    assert mode == 'fsdp', f'unsupported mode {mode}'

    device_mesh = setup_distributed()


    #
    # Data Parallel - use FSDP2 for a ready-made example
    #
    # Original weight: (N, K)
    # New weight: (N // world_size, K) on each shard, sharded on dim0
    #

    M, K, N = 4, 8, 16
    world_size = torch.distributed.get_world_size()
    assert world_size == 2, "unsupported"

    # create a toy linear
    m_unsharded = nn.Sequential(nn.Linear(K, N, bias=False, device='cuda'))

    m_sharded = copy.deepcopy(m_unsharded)
    m_sharded = torch.distributed.fsdp.fully_shard(m_sharded, mesh=device_mesh)

    w_us = m_unsharded[0].weight
    w_s = m_sharded[0].weight
    w_s_f = w_s.full_tensor()

    # local tensor on rank 0 equals to a slice of the original tensor
    if get_rank() == 0:
        shape_us = N, K
        shape_s = N // world_size, K
        torch.testing.assert_close(w_us[:N//world_size], w_s.to_local(), atol=0, rtol=0)

    # original tensor equals the reassembled tensor
    torch.testing.assert_close(w_us, w_s_f, atol=0, rtol=0)

    # passing an input through results in equal outputs and gradients
    x = torch.randn(M, K).cuda()
    x_copy = copy.deepcopy(x)
    x.requires_grad_()
    x_copy.requires_grad_()

    y_us = m_unsharded(x)
    y_us.sum().backward()

    y_s = m_sharded(x_copy)
    y_s.sum().backward()

    # output
    torch.testing.assert_close(y_us, y_s, atol=0, rtol=0)
    # grad_input
    torch.testing.assert_close(x.grad, x_copy.grad, atol=0, rtol=0)
    # grad_weight
    torch.testing.assert_close(w_us.grad, w_s.grad.full_tensor(), atol=0, rtol=0)


    # torch.distributed.breakpoint(0)

    print0('done')

    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    fire.Fire(run)
