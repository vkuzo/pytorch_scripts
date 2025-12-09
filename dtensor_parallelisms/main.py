"""
Working through basic parallelisms with DTensor

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
from torch.distributed.device_mesh import init_device_mesh

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


def run(mode: str):
    device_mesh = setup_distributed()
    world_size = torch.distributed.get_world_size()
    print0("mode", mode)

    if mode == "fsdp":
        #
        # Data Parallel - use FSDP2 for a ready-made example
        #
        # Original weight: (N, K)
        # New weight: (N // world_size, K) on each shard, sharded on dim0
        #

        M, K, N = 4, 8, 16
        assert world_size == 2, "unsupported"

        # create a toy linear
        m_unsharded = nn.Sequential(nn.Linear(K, N, bias=False, device="cuda"))

        m_sharded = copy.deepcopy(m_unsharded)
        m_sharded = torch.distributed.fsdp.fully_shard(m_sharded, mesh=device_mesh)

        w_us = m_unsharded[0].weight
        w_s = m_sharded[0].weight
        w_s_f = w_s.full_tensor()

        # local tensor on rank 0 equals to a slice of the original tensor
        if get_rank() == 0:
            torch.testing.assert_close(
                w_us[: N // world_size], w_s.to_local(), atol=0, rtol=0
            )

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

    else:
        assert mode == "tp", f"unsupported mode {mode}"

        #
        # Tensor parallel
        # * https://arxiv.org/pdf/1909.08053 (TP)
        # * https://arxiv.org/pdf/2205.05198 section 4.2.2 figure 5 (paper about TP + SP)
        # * https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html
        #
        # unsharded fwd:
        #   input: (M, dim1)
        #   w1 and w3 are (dim1, dim2): (M, dim1) -> (M, dim2)
        #   w2 is (dim2, dim1): (M, dim2) -> (M, dim1)
        #
        # sharded fwd:
        #   input: (M, dim1)
        #   w1 and w3 are (dim1, dim2 // world_size): (M, dim1) -> (M, dim2 // world_size)
        #   w2 is (dim2 // world_size, dim1): (M, dim2 // world_size) -> (M, dim1)
        #   all-reduce(sum) the output

        class FFN(nn.Module):
            def __init__(self, dim1, dim2):
                super().__init__()
                self.w1 = nn.Linear(dim1, dim2, bias=False)
                self.w2 = nn.Linear(dim2, dim1, bias=False)
                self.w3 = nn.Linear(dim1, dim2, bias=False)

            def forward(self, x):
                # self.w2(F.silu(self.w1(x)) * self.w3(x))
                # print0('x', x.shape)
                w1 = self.w1(x)  # M, dim1 -> M, dim2
                w3 = self.w3(x)  # M, dim1 -> M, dim2
                # print0('w1.shape', w1.shape)
                # print0('w3.shape', w3.shape)
                tmp = F.silu(w1) * w3  # (M, dim2), (M, dim2) -> M, dim2
                w2 = self.w2(tmp)  # M, dim2 -> M, dim1
                # print0('w2.shape', w2.shape)
                return w2

        M, K, N = 4, 8, 16
        print0("MKN", M, K, N)

        x_ref = torch.randn(M, K, device="cuda")
        m_ref = FFN(K, N).cuda()

        # print0('unsharded forward')
        y = m_ref(x_ref)
        # print0('y', y)

        # apply TP APIs from https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            RowwiseParallel,
            parallelize_module,
        )

        layer_tp_plan = {
            "w1": ColwiseParallel(),
            "w2": RowwiseParallel(),
            "w3": ColwiseParallel(),
        }
        m_dtensor = copy.deepcopy(m_ref)

        parallelize_module(m_dtensor, device_mesh, layer_tp_plan)

        # print0('sharded forward')
        y2 = m_dtensor(x_ref)
        # print0('y2', y2)

        # not exact because we are comparing a matmul in one shot to two matmuls + all-reduce add
        torch.testing.assert_close(y, y2)

        # now, reimplement the above by hand for world_size == 2
        assert world_size == 2, "unsupported"
        w1_col_size = m_ref.w1.weight.shape[0] // world_size
        w2_row_size = m_ref.w2.weight.shape[1] // world_size
        rank = get_rank()

        w1_col_start = w1_col_size * rank
        w1_col_end = w1_col_start + w1_col_size

        w2_row_start = w2_row_size * rank
        w2_row_end = w2_row_start + w2_row_size

        w1_sharded = m_ref.w1.weight[w1_col_start:w1_col_end]
        w3_sharded = m_ref.w3.weight[w1_col_start:w1_col_end]
        w2_sharded = m_ref.w2.weight[:, w2_row_start:w2_row_end]

        w1_out = torch.mm(x_ref, w1_sharded.t())
        w3_out = torch.mm(x_ref, w3_sharded.t())
        tmp_out = F.silu(w1_out) * w3_out
        w2_out = torch.mm(tmp_out, w2_sharded.t())
        torch.distributed.all_reduce(w2_out)
        torch.testing.assert_close(y, w2_out)

    # torch.distributed.breakpoint(0)

    print0("done")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(run)
