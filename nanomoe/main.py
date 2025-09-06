"""
Studying https://cameronrwolfe.substack.com/p/nano-moe
and https://github.com/wolfecameron/nanoMoE/blob/master/model.py in detail
"""

import math
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# model code is copy-pasta from https://github.com/wolfecameron/nanoMoE/blob/master/model.py,
# with slight modifications


class Router(nn.Module):
    def __init__(self, config):
        super().__init__()

        # router settings
        self.top_k = config.top_k
        self.n_exp = config.n_exp
        assert self.top_k >= 1 and self.top_k <= config.n_exp
        self.use_noisy_top_k = config.use_noisy_top_k
        self.train_capacity = config.train_capacity
        self.eval_capacity = config.eval_capacity
        self.min_capacity = config.min_capacity
        self.router_use_full_prec = config.router_use_full_prec

        # auxiliary / load balancing loss settings
        self.use_aux_loss = config.use_aux_loss
        self.use_router_z_loss = config.use_router_z_loss

        # linear projection for (noisy) softmax gating
        # no bias is used, see page 4 eq (4) in (https://arxiv.org/abs/1701.06538)
        self.w_g = nn.Linear(config.n_embd, config.n_exp, bias=False)
        self.w_noise = nn.Linear(config.n_embd, config.n_exp, bias=False) if self.use_noisy_top_k else None
    
    def forward(self, x):
        # optionally run the router in full precision to avoid instability during training
        # see discussion on pg. 9 here: https://arxiv.org/abs/2101.03961
        # setting enabled to False in autocast automatically puts everything in float32
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu' # for later use in torch.autocast
        ctx = nullcontext() if not self.router_use_full_prec else torch.amp.autocast(device_type=device_type, enabled=False)

        with ctx:
            B, T, _ = x.size()
            num_tokens = B * T

            # eq (4) in (https://arxiv.org/abs/1701.06538)
            logits = self.w_g(x)  # [B, T, n_exp]
            if self.use_noisy_top_k:
                # optionally add noise into the router
                noise = F.softplus(self.w_noise(x))
                noise *= torch.randn_like(noise)
                logits += noise

            # router z loss, computed on logits (before softmax)
            # this loss prevents router logits from becoming too large
            if self.use_router_z_loss:
                z_loss = self.compute_router_z_loss(logits)
                MANAGER.add_router_z_loss(z_loss)

            # find top k experts for each token
            top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1) # [B, T, k]
            print('top_k_indices', top_k_indices, top_k_indices.shape)

            # normalize expert probabilities
            # Question: should we normalize over all experts or just top-k?
            # we choose to normalize over top-k, other option is commented out below

            # Shazeer et al (https://arxiv.org/abs/1701.06538) does only topk
            # see page 4 eq (3)-(5), the code for this is commented out below
            router_probs = torch.full_like(logits, float('-inf'))  # [B, T, n_exp]
            router_probs.scatter_(-1, top_k_indices, top_k_logits)
            router_probs = F.softmax(router_probs, dim=-1)
            print('router_probs', router_probs, router_probs.shape)

            # # normalize all router logits (not just top-k) via softmax      
            # router_probs = F.softmax(logits, dim=-1)

            # compute auxiliary load balancing loss
            # this loss encourages equal probability assigned to each expert
            # and equal load balancing of tokens assigned to each expert
            if self.use_aux_loss:
                aux_loss = self.compute_aux_loss(router_probs, top_k_indices)
                MANAGER.add_aux_loss(aux_loss)

            # compute expert capacity
            exp_capacity = self.get_capacity(num_tokens)
            # print('exp_capacity', exp_capacity)

            # make a multi-hot mask of chosen experts, size [B, T, n_exp]
            # entries are 0 if expert not chosen and 1 if expert chosen
            exp_mask = F.one_hot(top_k_indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            # print('exp_mask B, T, k, n_exp', exp_mask, exp_mask.shape)
            exp_mask = exp_mask.view(num_tokens, self.top_k, self.n_exp)  # [B * T, k, n_exp]
            exp_mask = exp_mask.permute(1, 0, 2) # [k, B * T, n_exp]
            # print('exp_mask k, B*T, n_exp', exp_mask, exp_mask.shape)

            # compute cumulative sum of each token over experts, this stores
            # the index of each token within the batch of each expert
            # NOTE: cumsum should count all top-1 first, top-2 second, etc.
            # so that we prioritize top experts when dropping tokens (this is
            # done by putting k dimension first for the reshape operation)
            exp_rank = exp_mask.reshape(self.top_k * num_tokens, self.n_exp)  # [k * B * T, n_exp]
            exp_rank = torch.cumsum(exp_rank, dim=0) - 1  # cumulative sum of expert selections [k * B * T, n_exp]
            exp_rank = exp_rank.reshape(self.top_k, num_tokens, self.n_exp)  # [k, B * T, n_exp]

            # mask out (set to zero) entries that go beyond expert capacity
            # compute amount of used capacity by taking a sum over mask
            exp_mask *= torch.lt(exp_rank, exp_capacity) # [k, B * T, n_exp]
            used_capacity = torch.sum(exp_mask, dim=(0, 1)) # [n_exp]

            # mask rank to only include tokens that are selected
            # perform a sum so each row only contains index of token
            # for the expert that is selected in that row
            # result is a matrix that contains the position of each token
            # in the batch of its corresponding expert
            exp_rank = torch.sum(exp_mask * exp_rank, dim=-1)  # [k, B * T]

            # mask probabilities to only include selected experts
            router_probs = router_probs.view(num_tokens, self.n_exp)[None, :] # [1, B * T, n_exp]
            exp_weights = exp_mask * router_probs # [k, B * T, n_exp]

            # convert rank into one-hot vectors over the available capacity
            # stores the position of each token within the capacity of the selected expert
            exp_rank_sc = F.one_hot(exp_rank, num_classes=exp_capacity) # [k, B * T, exp_capacity]

            # create a vector that stores, for each token, the weight of selected
            # experts at token's position in the capacity of that expert
            # size of tensor is [B * T, n_exp, exp_capacity]
            cb_weight = torch.sum(exp_weights.unsqueeze(3) * exp_rank_sc.unsqueeze(2), dim=0)
            sec_mask = cb_weight.bool() # binary mask of selected experts for each token
            return used_capacity, cb_weight, sec_mask
    
    def compute_aux_loss(self, expert_probs: torch.Tensor, indices: torch.Tensor):
        """
        Computes Switch Transformer auxiliary loss (https://arxiv.org/abs/2101.03961)
        See equations (4)-(6) on page 7
        """

        # equation (5): compute ratio of tokens allocated to each expert
        # total number of tokens is defined as total tokens in batch * k
        # (k = 1) for the Switch Transformer
        with torch.no_grad():
            one_hot_indices = F.one_hot(indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            one_hot_indices = torch.sum(one_hot_indices.float(), dim=2)  # [B, T, n_exp] (sum over k dimension)
            tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0, 1))

        # equation (6): compute ratio of router probability allocated to each expert
        prob_per_expert = torch.mean(expert_probs.float(), dim=(0, 1))

        # equation (4): take a scaled dot product between prob/token allocation vectors
        # multiply the result by the number of experts
        return self.n_exp * torch.sum(prob_per_expert * tokens_per_expert)
    
    def compute_router_z_loss(self, logits: torch.Tensor):
        """
        Computes ST-MoE router z loss (https://arxiv.org/abs/2202.08906)
        See equation (5) on page 7
        """
    
        # exponentiate logits, sum logits of each expert, take log, and square
        # code below is the same as:
        # > z_loss = torch.exp(logits)
        # > z_loss = torch.sum(z_loss, dim=-1)
        # > z_loss = torch.log(z_loss) ** 2.0
        z_loss = torch.logsumexp(logits, dim=-1) ** 2.0  # [B, T, n_exp]

        # sum over all tokens and divide by total number of tokens
        return torch.mean(z_loss)

    def get_capacity(self, tokens_per_batch):
        # expert capacity is given by (tokens_per_batch / num_experts) * capacity_factor
        # see eq (3) in Switch Transformer (https://arxiv.org/abs/2101.03961)
        capacity_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = math.floor(self.top_k * capacity_factor * tokens_per_batch / self.n_exp)
        capacity += capacity % 2 # make sure capacity is an even number
        capacity = max(capacity, self.min_capacity) # use min capacity
        assert capacity > 0
        return int(capacity)

class MLPExperts(nn.Module):
    """
    implementation of multiple MLP-based experts that can process input
    in batch -- based upon ColossalAI OpenMoE but simple, has optional bias, and
    uses a bmm instead of a loop over a mm for each expert to improve efficiency
    link: https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/moe/experts.py
    """
    def __init__(self, config):
        # TODO: add param init
        super().__init__()
        self.bias = config.bias

        self.c_fc = nn.Parameter(torch.empty(config.n_exp, config.n_embd, 4 * config.n_embd))
        self.c_proj = nn.Parameter(torch.empty(config.n_exp, 4 * config.n_embd, config.n_embd))
        self.fc_bias = nn.Parameter(torch.empty(config.n_exp, 1, 4 * config.n_embd)) if self.bias else None
        self.proj_bias = nn.Parameter(torch.empty(config.n_exp, 1, config.n_embd)) if self.bias else None
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
    

    def forward(self, x):
        x = torch.bmm(x, self.c_fc)
        if self.bias:
            x += self.fc_bias
        x = self.gelu(x)
        x = torch.bmm(x, self.c_proj)
        if self.bias:
            x += self.proj_bias
        x = self.dropout(x)
        return x

class MOELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = Router(config) # (noisy) top k router
        self.experts = MLPExperts(config) # group of MLPs (experts)

    def forward(self, x: torch.Tensor):
        B, T, n_embd = x.size() # track original shape of input
        num_tokens = (B * T)

        # pass each token through the router
        used_capacity, exp_weight, exp_mask = self.router(x)
        # print('used_capacity', used_capacity, used_capacity.shape)
        # B * T, n_exp, exp_capacity
        # print('exp_weight', exp_weight, exp_weight.shape)
        # B * T, n_exp, exp_capacity
        # print('exp_mask', exp_mask, exp_mask.shape)
        # print('exp_mask permuted', exp_mask.permute(1, 2, 0))

        # flatten out the input
        x = x.view(num_tokens, n_embd)

        print('x', x)

        # example x:
        # x tensor(
        #   [[-0.6136,  0.0316, -0.4927,  0.2484,  0.4397,  0.1124,  0.6408,  0.4412],
        #    [-0.1023,  0.7924, -0.2897,  0.0525,  0.5229,  2.3022, -1.4689, -1.5867]])

        # reshape tokens into batches for each expert
        # [n_exp, exp_capacity, B * T] * [B * T, n_embd] -> [n_exp, exp_capacity, n_embd]
        exp_batches = exp_mask.permute(1, 2, 0).type_as(x) @ x
        print('exp_batches', exp_batches)

        # example exp_batches sending to expert 0 and expert 2 only:
        #
        # exp_batches tensor([[[-0.6136,  0.0316, -0.4927,  0.2484,  0.4397,  0.1124,  0.6408, 0.4412],
        #   [-0.1023,  0.7924, -0.2897,  0.0525,  0.5229,  2.3022, -1.4689, -1.5867],
        #   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000],
        #   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000]],
        # [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000],
        #   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000],
        #   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000],
        #   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000]],
        # [[-0.6136,  0.0316, -0.4927,  0.2484,  0.4397,  0.1124,  0.6408, 0.4412],
        #   [-0.1023,  0.7924, -0.2897,  0.0525,  0.5229,  2.3022, -1.4689, -1.5867],
        #   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000],
        #   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000]],
        # [[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000],
        #   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000],
        #   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000],
        #   [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, 0.0000]]])
        #
        # note that:
        # 1. tokens are duplicated (unavoidable with top_k > 1)
        # 2. non selected experts waste computation as they are multiplying matrices of 0s

        # compute expert output
        exp_out = self.experts(exp_batches) # [n_exp, exp_capacity, n_embd]

        # aggregate expert outputs based on router weights
        # eq (2) on page 4 of ST-MoE (https://arxiv.org/abs/2202.08906)
        # similar equations are used for other MoE papers
        exp_weight = exp_weight.view(num_tokens, -1) # [B * T, n_exp * exp_capacity]
        exp_out = exp_out.view(-1, n_embd) # [n_exp * exp_capacity, n_embd] 
        output = exp_weight @ exp_out # [B * T, n_embd]
        
        # resize output before return
        return output.view(B, T, n_embd)


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    # MoE-related configs 
    n_exp: int = 1 # if n_exp = 1 we just use regular MLP layers
    top_k: int = 2
    use_aux_loss: bool = False # apply auxiliary loss (from Switch Transformer) in router
    use_router_z_loss: bool = False # apply router z loss (from ST-MoE)
    use_noisy_top_k: bool = False
    aux_loss_weight: float = 0.01 # default setting from Switch Transformer (see top of page 8)
    router_z_loss_weight: float = 0.001 # default setting from ST-MoE (see page 8 eq. 6)
    train_capacity: float = 1.25  # default setting from ST-MoE (see top of page 6)
    eval_capacity: float = 2.0
    min_capacity: int = 4  # minimum batch size to send to any single expert
    stride: int = 2 # one in every stride layers are converted to an MoE
    use_switch_tfm_init: bool = False  # use weight init scheme from Switch Transformer
    switch_tfm_init_scale: float = 1.0
    router_use_full_prec: bool = False  # use float32 precision in the router

def run():

    B, T, n_embd = 1, 2, 8
    n_exp = 4
    config = GPTConfig(n_exp=n_exp, n_embd=n_embd)
    moe = MOELayer(config)
    x = torch.randn(B, T, n_embd)
    print('x.shape', x.shape)

    y = moe(x)

    print('done')


if __name__ == '__main__':
    run()
