# modified copy of https://github.com/pytorch/torchtitan/blob/d14f1e3bcb4570be461b4bb70e0be522b4bc9a1c/torchtitan/distributed/expert_parallel.py
# with non-ep code removed

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
from typing import Callable, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import get_rank
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement
from utils import print0, print1


# from torch.distributed._functional_collectives import all_to_all_single_autograd
# TODO: there is memory leak issue with AC + all_to_all_single_autograd
# This is a temporary fix by @rakkit https://github.com/pytorch/torchtitan/issues/1467
class _A2A(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out_splits, in_splits, group):
        if isinstance(out_splits, torch.Tensor):
            out_splits = out_splits.tolist()
        if isinstance(in_splits, torch.Tensor):
            in_splits = in_splits.tolist()
        T_out = int(sum(out_splits))

        y = x.new_empty((T_out,) + tuple(x.shape[1:]))  # allocate by output splits
        dist.all_to_all_single(y, x.contiguous(), out_splits, in_splits, group=group)

        ctx.in_splits = in_splits
        ctx.out_splits = out_splits
        ctx.group = group
        return y

    @staticmethod
    def backward(ctx, grad_y):
        # grad wrt input has length sum(in_splits)
        T_in = int(sum(ctx.in_splits))
        grad_x = grad_y.new_empty((T_in,) + tuple(grad_y.shape[1:]))
        dist.all_to_all_single(
            grad_x, grad_y.contiguous(), ctx.in_splits, ctx.out_splits, group=ctx.group
        )
        return grad_x, None, None, None


def all_to_all_single_autograd(x, out_splits, in_splits, group):
    return _A2A.apply(x, out_splits, in_splits, group)


TOKEN_GROUP_ALIGN_SIZE_M = 8
ValidTokenGroupAlignmentSize = Literal[8, 16, 32]


def set_token_group_alignment_size_m(
    alignment_size: ValidTokenGroupAlignmentSize,
) -> None:
    """
    Set the token group alignment size for token groups in MoE. This is implemented by
    padding each token group size to the next multiple of TOKEN_GROUP_ALIGN_SIZE_M.

    Valid values are: 8, 16, or 32.
    Different values are needed for different cases:

    * For bf16, 8 is enough (16 byte alignment / 2 bytes per elem = 8 elements).
    * For fp8, 16 byte alignment / 1 byte per elem = 16 elements.
    * For mxfp8, we need 32 (or block_size) because scaling block size is (1 x 32),
      so when doing per-token-group quantization on each logically distinct subtensor,
      we need to ensure the contracting dim is divisible by block_size.
      In the backward pass, grad_weight = (grad_output_t @ input).t() has gemm dims
      of (N, M) @ (M, K) so M is the contracting dim, and group offsets are along M,
      so we need 32 element alignment.
    """
    global TOKEN_GROUP_ALIGN_SIZE_M
    TOKEN_GROUP_ALIGN_SIZE_M = alignment_size


class ExpertParallel(ParallelStyle):
    def __init__(self):
        super().__init__()
        self.input_splits = None
        self.output_splits = None

    # performing all-to-all dispatch on the input
    def _token_dispatch(self, mod, inputs, device_mesh):
        print0('start token dispatch')
        # annotate module input placements/sharding with input_layouts
        routed_input, num_tokens_per_expert = inputs
        print0('routed_input', routed_input)

        # print('rank', get_rank(), 'num_tokens_per_expert', num_tokens_per_expert, '\n')

        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
            num_tokens_per_expert_group = num_tokens_per_expert.new_empty(
                num_tokens_per_expert.shape[0]
            )
            dist.all_to_all_single(
                num_tokens_per_expert_group,
                num_tokens_per_expert,
                group=device_mesh.get_group(),
            )
            # example with actual values
            #
            # before all_to_all_single 
            #   num_tokens_per_expert rank 0: tensor([1, 0, 2, 1], device='cuda:0')
            #   num_tokens_per_expert rank 1: tensor([0, 0, 1, 3], device='cuda:1')
            # after all_to_all_single 
            #   num_tokens_per_expert_group rank 0: tensor([1, 0, 0, 0], device='cuda:0')
            #   num_tokens_per_expert_group rank 1: tensor([2, 1, 1, 3], device='cuda:1')
            # 
            # example with variables
            #
            # before all_to_all_single
            #   rank 0: [n0, n1, n2, n3]
            #   rank 1: [n4, n5, n6, n7]
            # after all_to_all_single
            #   rank 0: [n0, n1, n4, n5]
            #   rank 1: [n2, n3, n6, n7]

            # print('rank', get_rank(), 'num_tokens_per_expert_group', num_tokens_per_expert_group)

            # NOTE: this would incur a device-to-host sync
            self.input_splits = (
                num_tokens_per_expert.view(device_mesh.shape[0], -1).sum(dim=1).tolist()
            )

            # example input_splits (how many tokens each rank is sending in a2a)
            #   rank 0: [1, 3]  # 1 token for experts 0:1, 3 tokens for experts 2:3
            #   rank 1: [0, 4]  # 0 token for experts 0:1, 4 tokens for experts 2:3
            # print('rank', get_rank(), 'input_splits', self.input_splits)
            # print0('input_splits', self.input_splits)

            # example output_splits (how many tokens each rank is receiving in a2a)
            #   rank 0: [1, 0]  # 1 token for expert 0, 0 tokens for expert 1
            #   rank 1: [3, 4]  # 3 token for expert 2, 4 tokens for expert 3
            self.output_splits = (
                num_tokens_per_expert_group.view(device_mesh.shape[0], -1)
                .sum(dim=1)
                .tolist()
            )
            # print('rank', get_rank(), 'output_splits', self.output_splits)
            # print0('output_splits', self.output_splits)

        # perform all-to-all

        # example routed_input, rank0:
        # 
        #   before a2a
        #
        # tensor([[ 0.1045,  1.0703, -0.6953,  0.8125,  0.2461, -0.0194, -1.8516, -0.8750],
        #         [ 0.9258,  0.2041,  1.1016, -0.0518, -1.1953,  0.2773, -0.9492, -0.3457],
        #         [-0.3789,  0.8242,  0.0203,  0.3945, -0.1377, -0.3984, -0.3633, -0.1514],
        #         [ 1.5000, -1.3125, -0.8594, -0.5820, -0.5039,  0.2100, -1.3281, -0.3789]],)
        # 
        #   after a2a (1 token for expert 0, 0 tokens for expert 1)
        #
        # tensor([[ 0.1045,  1.0703, -0.6953,  0.8125,  0.2461, -0.0194, -1.8516, -0.8750]])
        #
        #   num_tokens_per_expert_group: tensor([1, 0, 0, 0], device='cuda:0')
        #
        # rank1:
        #   
        #   before a2a
				#
				# tensor([[-1.0781e+00,  1.6602e-01,  4.1602e-01,  1.3672e+00, -2.3750e+00, 1.1172e+00, -7.0703e-01, -1.6504e-01],
				#         [ 3.6523e-01,  2.5940e-03,  5.4688e-01,  3.0078e-01,  1.8984e+00, -2.4531e+00, -2.1562e+00, -2.9102e-01],
				#         [ 1.1016e+00,  1.0469e+00, -9.7266e-01, -2.0469e+00,  9.8633e-02, -6.4844e-01, -1.1094e+00,  3.9844e-01],
				#         [ 2.6250e+00, -6.7383e-02, -2.3594e+00,  1.5938e+00,  5.6250e-01, -1.1406e+00,  6.6797e-01,  1.3477e-01]], device='cuda:1',)
				#
				#   after a2a (3 token for expert 2, 4 tokens for expert 3)
				#
        # tensor([[ 9.2578e-01,  2.0410e-01,  1.1016e+00, -5.1758e-02, -1.1953e+00, 2.7734e-01, -9.4922e-01, -3.4570e-01],
        #         [-3.7891e-01,  8.2422e-01,  2.0264e-02,  3.9453e-01, -1.3770e-01, -3.9844e-01, -3.6328e-01, -1.5137e-01],
        #         [ 1.5000e+00, -1.3125e+00, -8.5938e-01, -5.8203e-01, -5.0391e-01, 2.0996e-01, -1.3281e+00, -3.7891e-01],
        #         [-1.0781e+00,  1.6602e-01,  4.1602e-01,  1.3672e+00, -2.3750e+00, 1.1172e+00, -7.0703e-01, -1.6504e-01],
        #         [ 3.6523e-01,  2.5940e-03,  5.4688e-01,  3.0078e-01,  1.8984e+00, -2.4531e+00, -2.1562e+00, -2.9102e-01],
        #         [ 1.1016e+00,  1.0469e+00, -9.7266e-01, -2.0469e+00,  9.8633e-02, -6.4844e-01, -1.1094e+00,  3.9844e-01],
        #         [ 2.6250e+00, -6.7383e-02, -2.3594e+00,  1.5938e+00,  5.6250e-01, -1.1406e+00,  6.6797e-01,  1.3477e-01]], device='cuda:1', dtype=torch.bfloat16)
        #
        #   num_tokens_per_expert_group: tensor([2, 1, 1, 3], device='cuda:1')
        #

        # print1('r1 routed_input before a2a', routed_input, routed_input.shape)
        routed_input = all_to_all_single_autograd(
            routed_input,
            self.output_splits,
            self.input_splits,
            device_mesh.get_group(),
        )
        # print1('r1 routed_input after a2a', routed_input, routed_input.shape)
        # print1('r1 num_tokens_per_expert_group', num_tokens_per_expert_group)

        # NOTE: After this all-to-all, the routed input is put on proper EP rank.
        # However, the num_tokens_per_expert_group is not of the final target format
        # [#tokens for local expert 0, #tokens for local expert 1, ...]
        # Rather, it is of the format
        # [#tokens for local expert 0 from EP rank 0, #tokens for local expert 1 from EP rank 0, ...,
        #  #tokens for local expert 0 from EP rank 1, #tokens for local expert 1 from EP rank 1, ...]
        # We need to perform another shuffle to get the correct format -- this is done via the function
        # generate_permute_indices in moe.py, which also does padding to make sure the number of tokens
        # each expert gets locally is a multiple of ALIGN_SIZE_M.

        return routed_input, num_tokens_per_expert_group

    @staticmethod
    def _partition_fn(name, mod, device_mesh):
        # shard on the expert dimension
        for name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(name, dist_param)

    # performing all-to-all combine on the output
    def _token_combine(self, mod, routed_output, device_mesh):
        print0('start token combine')
        routed_output = all_to_all_single_autograd(
            routed_output,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group(),
        )
        return routed_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=ExpertParallel._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


def expert_parallel(func: Callable) -> Callable:
    """
    This is a wrapper applied to the GroupedExperts computation, serving
    the following three purposes:
    1. Convert parameters from DTensors to plain Tensors, to work with
    dynamic-shape inputs which cannot be easily expressed as DTensors.
    2. In Expert Parallel, apply the generate_permute_indices kernel to
    permute the inputs to be ordered by local experts (see the _token_dispatch
    function in ExpertParallel) and permute the outputs back.
    3. In order to use torch._grouped_mm, we need to make sure the number of
    tokens each expert gets is a multiple of ALIGN_SIZE_M. The generate_permute_indices
    kernel also helps achieve this via padding, without incurring synchronization
    between device and host. Note that this will create side effects when wrapping
    the for-loop implementation of GroupedExperts, as it does not need padding.

    Among the above:
    1 and 2 are needed only when expert_parallel_degree > 1.
    3 is needed even for single-device computation.
    2 can be moved to ExpertParallel _token_dispatch if not coupled with 3.
    """

    def wrapper(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        print0('start expert_parallel wrapper')
        global TOKEN_GROUP_ALIGN_SIZE_M
        if isinstance(w1, DTensor):
            w1 = w1.to_local()
            w2 = w2.to_local()
            w3 = w3.to_local()

        if num_tokens_per_expert is not None:
            # from torchtitan.experiments.kernels.moe.indices import (
            #     generate_permute_indices,
            # )
            from indices import generate_permute_indices

            experts_per_ep_rank = w1.shape[0]
            num_ep_ranks = num_tokens_per_expert.shape[0] // experts_per_ep_rank
            # print1('experts_per_ep_rank', experts_per_ep_rank, 'num_ep_ranks', num_ep_ranks)

            # print1('before generate_permute_indices')
            # print1('num_tokens_per_expert', num_tokens_per_expert, 'x.shape[0]', x.shape[0], 'TOKEN_GROUP_ALIGN_SIZE_M', TOKEN_GROUP_ALIGN_SIZE_M)
            # print1('x.shape[0] + experts_per_ep_rank * TOKEN_GROUP_ALIGN_SIZE_M', x.shape[0] + experts_per_ep_rank * TOKEN_GROUP_ALIGN_SIZE_M)

            # example
            #   rank 0
            #     inputs
            #       num_tokens_per_expert = tensor([1, 0, 0, 0], device='cuda:0')
            #       experts_per_ep_rank = 2
            #       num_ep_ranks = 2
            #       max_len = 17
            #       TOKEN_GROUP_ALIGN_SIZE_M = 16
            #     outputs
            #       permuted_indices = tensor([ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], device='cuda:0', dtype=torch.int32) torch.Size([17])
            #       num_tokens_per_expert = tensor([8, 8], device='cuda:0', dtype=torch.int32)
            #       _ = tensor([ 8, 16], device='cuda:0', dtype=torch.int32)
            #   rank 1
            #     inputs
            #       num_tokens_per_expert = tensor([2, 1, 1, 3], device='cuda:1')
            #       experts_per_ep_rank = 2
            #       num_ep_ranks = 2
            #       max_len = 23
            #       TOKEN_GROUP_ALIGN_SIZE_M = 16
            #     outputs
            #       permuted_indices = tensor([ 0,  1,  3, -1, -1, -1, -1, -1,  2,  4,  5,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], device='cuda:1', dtype=torch.int32) torch.Size([23])
            #       num_tokens_per_expert = num_tokens_per_expert tensor([8, 8], device='cuda:1', dtype=torch.int32)
            #       _ = tensor([ 8, 16], device='cuda:1', dtype=torch.int32)

            with torch.no_grad():
                # note that this creates `-1` indices for entries which should be padded with zeroes
                (
                    permuted_indices,
                    num_tokens_per_expert,
                    _,  # offsets,
                ) = generate_permute_indices(
                    num_tokens_per_expert,
                    experts_per_ep_rank,
                    num_ep_ranks,
                    x.shape[0] + experts_per_ep_rank * TOKEN_GROUP_ALIGN_SIZE_M,
                    TOKEN_GROUP_ALIGN_SIZE_M,
                )
            # print1('after generate_permute_indices')
            # print1('permuted_indices', permuted_indices, permuted_indices.shape)
            # print1('num_tokens_per_expert', num_tokens_per_expert)
            # print1('_', _)

            # print1('old x', x.shape, x)
            # add a row of zeros to the end, we use this to generate padding
            x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
            # print1('intermediate x', x.shape, x)
            input_shape = x.shape
            # this pads the tensor with zeros, using the padding we generated above
            x = x[permuted_indices, :]
            # print1('new x', x.shape, x)

        out = func(w1, w2, w3, x, num_tokens_per_expert)
        print0('after func in expert_parallel wrapper')
        print0('out.shape', out.shape)

        # example:
        #   rank 0 shape: (17, 8) -> (1, 8)
        #   rank 1 shape: (23, 8) -> (7, 8)
        if num_tokens_per_expert is not None:
            out_unpermuted = out.new_empty(input_shape)
            out_unpermuted[permuted_indices, :] = out
            out = out_unpermuted[:-1]
            print0('new out.shape', out.shape)

        print0('end expert_parallel wrapper')
        return out

    return wrapper
