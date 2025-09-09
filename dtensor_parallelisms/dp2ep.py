"""
Working through dp2ep with DTensor

run with

  torchrun --nproc_per_node 2 dp2ep.py
"""

import copy
import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torch.distributed import get_rank
from torch.distributed._tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

from torchtitan_moe import MoE, MoEArgs
from expert_parallel import ExpertParallel

torch.backends.fp32_precision = "ieee"

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

# copied from https://github.com/pytorch/torchtitan/pull/1324/files and simplified
def apply_moe_ep(
    model: nn.Module,
    ep_mesh: DeviceMesh,
):
    parallelize_module(
        module=model.experts,
        device_mesh=ep_mesh,
        parallelize_plan=ExpertParallel(),
    )

def run():

    device_mesh = setup_distributed()
    world_size = torch.distributed.get_world_size()
    local_rank = torch.distributed.get_rank()

    moe_args = MoEArgs(
        num_experts=2,
        # turn off shared expert for now, for simplicity
        num_shared_experts=0,
        # turn off expert bias for now, for simplicity
        load_balance_coeff=None,
    )
    dim, hidden_dim = 512, 1024
    batch, seq, dim = 8, 2028, dim
    with torch.device('cuda'):
        moe = MoE(moe_args, dim, hidden_dim).to(torch.bfloat16)
        moe.init_weights(init_std=0.1, buffer_device='cuda')
        x = torch.randn(batch, seq, dim, dtype=torch.bfloat16)

    # in torchtitan, DP is implicit, so we keep the input tensor as a
    # regular tensor instead of converting to DTensor
    batch_local_size = batch // world_size
    batch_local_start = batch_local_size * local_rank
    batch_local_end = batch_local_start + batch_local_size
    x_local = x[batch_local_start:batch_local_end]

    print0(moe)
    # print0(moe.experts.w1)

    y_ref = moe(x_local)

    apply_moe_ep(moe, device_mesh)
    # print0(moe)
    # print0(moe.experts.w1)

    y2 = moe(x_local)

    # exact match
    torch.testing.assert_close(y_ref, y2, rtol=0, atol=0)

    # torch.distributed.breakpoint(0)

    print0('done')

    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    fire.Fire(run)
