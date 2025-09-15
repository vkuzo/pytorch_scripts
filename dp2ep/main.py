"""
Working through dp2ep with DTensor

run with

  torchrun --nproc_per_node 2 main.py
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

from utils import print0

torch.backends.fp32_precision = "ieee"

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
    assert world_size == 2, "unsupported"

    # turn off shared expert for now, for simplicity
    num_experts, num_shared_experts = 2, 0
    top_k = 1
    moe_args = MoEArgs(
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        # turn off expert bias for now, for simplicity
        load_balance_coeff=None,
        top_k=top_k,
        # set score_before_experts to False for simplicity
        score_before_experts=False
    )
    dim, hidden_dim = 8, 16
    batch, seq = 2, 4

    print0(f'{dim=}, {hidden_dim=}, {batch=}, {seq=}, {num_experts=}, {top_k=}')

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
    print0('x.shape', x.shape, 'x_local.shape', x_local.shape)

    print0(moe)
    # print0('before moe.experts.w1', moe.experts.w1)

    print0('\nstarting ep==1\n')
    y_ref = moe(x_local)

    print0('\nstarting ep==2\n')
    apply_moe_ep(moe, device_mesh)
    # print('local_rank', local_rank, 'after moe.experts.w1', moe.experts.w1)
    # print0(moe.experts.w1)

    y2 = moe(x_local)

    # exact match
    torch.testing.assert_close(y_ref, y2, rtol=0, atol=0)

    # torch.distributed.breakpoint(0)

    print0('dp2ep done')

    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    fire.Fire(run)
