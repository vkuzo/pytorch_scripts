"""Router for the reference dropless MoE block.

Holds ``DeepseekRouter``, the group-limited sigmoid top-k router that replicates
HuggingFace's ``DeepseekV3MoE.route_tokens_to_experts`` exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepseekRouter(nn.Module):
    """Group-limited sigmoid top-k router, replicating ``DeepseekV3TopkRouter`` +
    ``DeepseekV3MoE.route_tokens_to_experts`` exactly (computed in float32).

    Returns ``(topk_indices, topk_weights)`` with shape ``[T, top_k]`` each.
    """

    def __init__(
        self,
        hidden_size: int,
        n_routed_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool,
        routed_scaling_factor: float,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor

        # Names match HF: router projection weight and the score correction bias.
        self.weight = nn.Parameter(torch.empty(n_routed_experts, hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(n_routed_experts))

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32)
        )

        # --- DeepseekV3MoE.route_tokens_to_experts, verbatim ---
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias

        # Group-limited routing: score each group by its top-2 experts, keep the best
        # `topk_group` groups, then pick the top-k experts among the kept groups.
        group_scores = (
            scores_for_choice.view(
                -1, self.n_group, self.n_routed_experts // self.n_group
            )
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        masked = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(masked, k=self.top_k, dim=-1, sorted=False)[1]

        # Weights are the *un-biased* sigmoid scores, gathered at the chosen experts.
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights
