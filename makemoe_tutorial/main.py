"""
Working through https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import fire

torch.manual_seed(0)


# Creating an Expert module i.e. a simple Multi Layer Perceptron
class Expert(nn.Module):
    """ An MLP is a simple linear layer followed by a non-linearity i.e. each Expert """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(TopkRouter, self).__init__()
        self.top_k = top_k
        self.linear = nn.Linear(n_embed, num_experts)
    
    def forward(self, mh_output):
        # input: (*leading_dims, n_embed)
        # output:
        #   router_output: (*leading_dims, num_experts)
        #   indices: (*leading_dims, top_k)

        # mh_ouput is the output tensor from multihead self attention block

        # (*leading_dims, n_embed) -> (*leading_dims, num_experts)
        logits = self.linear(mh_output)
        # print('logits', logits)

        # (*leading_dims, num_experts) -> (*leading_dims, top_k)
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        # print('top_k_logits', top_k_logits.shape, top_k_logits)

        # Obtain the sparse gating output by only keeping the top k values in their 
        # respective index along the last dimension. Fill the rest with '-inf' and 
        # pass through a softmax activation. This pushes '-inf' values to zero, 
        # makes the top two values more accentuated and sum to 1. This summation 
        # to 1 helps with the weighting of expert outputs
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        # print('sparse_logits', sparse_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        # print('router_output', router_output.shape, router_output)
        # print('indices', indices.shape, indices)

        return router_output, indices


class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k, dropout):
        super(SparseMoE, self).__init__()
        self.router = TopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed, dropout) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        # x: (*dims, n_embed)

        # gating_output: (*dims, num_experts)
        # indices: (*dims, top_k)
        gating_output, indices = self.router(x)

        # final_output: (*dims, n_embed)
        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))  # (-1, n_embed)
        # print('flat_x.shape', flat_x.shape)
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))  # (-1, num_experts)

        # print('indices', indices)

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # print('i', i)
            # Create a mask for the inputs where the current expert is in top-k
            # print('indices == i', indices == i)

            # shape (*dims)
            expert_mask = (indices == i).any(dim=-1)
            # print('expert_mask', expert_mask.shape, expert_mask)
            flat_mask = expert_mask.view(-1)
            # print('flat_mask', flat_mask.shape, flat_mask)

            if flat_mask.any():
                # Select the original data which needs to be sent to this expert
                # flat_x shape: (prod(dims), n_embed)
                # flat_mask shape: (prod(dims))
                # expert_input shape: (tokens_this_expert, n_embed)
                expert_input = flat_x[flat_mask]
                # expert_output shape: (tokens_this_expert, n_embed)
                expert_output = expert(expert_input)
                # print('in.shape', expert_input.shape, 'out.shape', expert_output.shape)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output


def run():

    # Understanding how gating works    

    num_experts = 4
    top_k = 2
    n_embed = 32
    dropout = 0.1

    # example MHA output
    mh_output = torch.randn(2, 4, n_embed)  # (2, 4, n_embed)
    print('mh_output.shape', mh_output.shape)

    # router = TopkRouter(n_embed, num_experts, top_k)
    # output, indices = router(mh_output)
    sparse_moe = SparseMoE(n_embed, num_experts, top_k, dropout)
    final_output = sparse_moe(mh_output)
    print('final_output.shape', final_output.shape)


if __name__ == '__main__':
    fire.Fire(run)
