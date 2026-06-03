"""Reference dropless MoE block (shared expert + grouped_mm).

This is a clean, plain-PyTorch reference for a Mixture-of-Experts layer whose
expert compute is expressed as a *grouped GEMM* over expert-sorted tokens, plus an
additive (ungated) shared expert. It mirrors the math of HuggingFace's
``DeepseekV3MoE`` building block, and the unit test in ``test.py`` asserts numerical
equivalence against it.

Properties:
  * **Dropless**: every routed (token, expert) pair is computed. There is no capacity
    factor and no token dropping.
  * **grouped_mm**: instead of the readable-but-slow per-expert ``torch.where`` +
    ``F.linear`` loop that HF uses, we sort tokens into expert-contiguous groups and
    run a single ``grouped_mm`` (here emulated with a Python for-loop, but laid out
    exactly the way a real fused grouped-GEMM kernel consumes data: a 2D token tensor
    plus an ``offs`` array of per-expert group boundaries).
  * **NOT sync-less**: the ``argsort`` / ``bincount`` / Python ``for`` loop introduce
    device-to-host syncs and dynamic shapes. That is intentional here for clarity;
    making it sync-less (so it is CUDA-graphable) is a separate exercise.

The routed-expert weights use the same 3D-parameter layout and names as
``DeepseekV3NaiveMoe`` (``gate_up_proj``, ``down_proj``) so weights can be copied
across by name. Note the transpose caveat documented on ``RoutedExperts``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from router import DeepseekRouter


def grouped_mm_2d_3d(x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Emulated grouped GEMM: a separate matmul per expert group.

    For each expert ``e``, computes ``x[start:end] @ w[e]`` where ``[start, end)`` is
    the contiguous block of rows assigned to that expert, as delimited by ``offs``.
    This is the for-loop reference for what a fused grouped-GEMM kernel (e.g.
    ``torch._grouped_mm`` / ``torch._scaled_grouped_mm``) does in a single launch.

    Args:
        x: ``[T, K]`` tokens, pre-sorted so each expert's tokens are contiguous.
        w: ``[E, K, N]`` per-expert weight matrices.
        offs: ``[E]`` int tensor of cumulative *end* row index of each group (i.e.
            ``offs[e]`` is the first row of group ``e+1``; ``offs[-1] == T``).

    Returns:
        ``[T, N]`` output, row-aligned with ``x``.
    """
    out = x.new_empty(x.shape[0], w.shape[-1])
    start = 0
    for e in range(w.shape[0]):
        end = int(offs[e])
        if end > start:  # skip experts that received no tokens
            out[start:end] = x[start:end] @ w[e]
        start = end
    return out


def grouped_mm_2d_2d(
    x: torch.Tensor, grad_out: torch.Tensor, offs: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """Weight-gradient grouped GEMM (the "varlen-K" contraction over tokens).

    For ``out = grouped_mm(x, w, offs)`` with ``x:[T,K]``, ``w:[E,K,N]``, the weight
    gradient for each expert is ``gw[e] = x[grp].T @ grad_out[grp]`` (shape ``[K, N]``),
    contracting over that expert's tokens.

    Args:
        x: ``[T, K]`` the (expert-sorted) inputs that were fed to the forward GEMM.
        grad_out: ``[T, N]`` upstream gradient w.r.t. the GEMM output.
        offs: ``[E]`` cumulative end-row per expert.
        num_experts: ``E``.

    Returns:
        ``[E, K, N]`` gradient w.r.t. the grouped weight tensor ``w``.
    """
    gw = x.new_zeros(num_experts, x.shape[1], grad_out.shape[1])
    start = 0
    for e in range(num_experts):
        end = int(offs[e])
        if end > start:
            gw[e] = x[start:end].transpose(0, 1) @ grad_out[start:end]
        start = end
    return gw


class GroupedExpertGEMM(torch.autograd.Function):
    """Expert compute kernel: grouped GEMM -> SwiGLU -> grouped GEMM, with a manual
    backward.

    This stands in for a fused MoE grouped-GEMM kernel (cf. TorchAO's
    ``_MXFP8GroupedMM``). It receives tokens that have *already* been permuted into
    expert-contiguous order plus the group ``offs``; token dispatch (permute/gather)
    and combine (routing-weight scaling + scatter-add) live outside, in
    ``RoutedExperts.forward``.

    The activation is hardcoded to SwiGLU with SiLU (``a = silu(gate) * up``), matching
    DeepSeek-V3, so the manual backward stays valid.

    Shapes (E experts, H hidden, I intermediate, M = total routed tokens):
        x_sorted:     [M, H]
        gate_up_proj: [E, 2I, H]   (HF layout, [out, in] for F.linear)
        down_proj:    [E, H, I]    (HF layout)
        offs:         [E]
        output y:     [M, H]
    """

    @staticmethod
    def forward(ctx, x_sorted, gate_up_proj, down_proj, offs):

        # * x_sorted quant: ideally `x_sorted` would be fused with
        #   dispatch (quant -> a2a -> write out to final layout on each rank).
        #   This is "EP with fused flexible quantization".
        # * gate_up_proj.transpose(-2, -1) quant: ~flex_cast_quant_grouped 
        h = grouped_mm_2d_3d(x_sorted, gate_up_proj.transpose(-2, -1), offs)  # [M, 2I]
        gate, up = h.chunk(2, dim=-1)  # [M, I] each
        a = F.silu(gate) * up  # [M, I]

        # * a quant: either ~flex_cast_quant_grouped, or 
        #   the same thing as an epilogue of the first grouped_gemm
        # * down_proj.transpose(-2, -1) quant: ~flex_cast_quant_grouped
        y = grouped_mm_2d_3d(a, down_proj.transpose(-2, -1), offs)  # [M, H]

        ctx.save_for_backward(x_sorted, gate_up_proj, down_proj, offs, gate, up, a)
        ctx.num_experts = gate_up_proj.shape[0]
        return y

    @staticmethod
    def backward(ctx, dy):
        x_sorted, gate_up_proj, down_proj, offs, gate, up, a = ctx.saved_tensors
        E = ctx.num_experts

        # --- down-proj: y = grouped_mm_2d_3d(a, down_proj) ---

        # * dy quant: ideally EP fused with flexible quantization
        # * down_proj quant: ~flex_cast_quant_grouped, we either
        #   - dequant quantized down_proj and requant along the other axis
        #   - save down_proj for bw in bf16
        da = grouped_mm_2d_3d(dy, down_proj, offs)  # [M, I]
        # * dy quant: ideally EP fused with flexible quant, need to switch quant axis
        #   vs the line above
        # * a quant: ~flex_cast_quant_grouped (the fwd can calculate along both axes)
        d_down_proj = grouped_mm_2d_2d(dy, a, offs, E)  # [E, H, I]

        # --- SwiGLU: a = silu(gate) * up ---
        sig = torch.sigmoid(gate)
        silu = gate * sig
        dsilu_dgate = sig * (1 + gate * (1 - sig))  # d/dgate [gate * sigmoid(gate)]
        dgate = da * up * dsilu_dgate  # [M, I]
        dup = da * silu  # [M, I]
        dh = torch.cat([dgate, dup], dim=-1)  # [M, 2I]

        # --- up-proj: h = grouped_mm_2d_3d(x_sorted, gate_up_proj) ---
        # * dh quant: either epilogue of preceding grouped gemm or 
        #   ~flex_cast_quant_grouped
        # * others - similar to above so skip
        dx = grouped_mm_2d_3d(dh, gate_up_proj, offs)  # [M, H]
        d_gate_up_proj = grouped_mm_2d_2d(dh, x_sorted, offs, E)  # [E, 2I, H]

        # No gradient flows to offs (non-differentiable indexing tensor).
        return dx, d_gate_up_proj, d_down_proj, None


class RoutedExperts(nn.Module):
    """Dropless routed experts via ``grouped_mm`` over expert-sorted tokens.

    Weights mirror ``DeepseekV3NaiveMoe``:
      * ``gate_up_proj``: ``[E, 2*I, H]`` (fused gate+up; stored ``[out, in]`` for
        ``F.linear`` in HF)
      * ``down_proj``:    ``[E, H, I]``   (stored ``[out, in]``)

    NOTE on layout: HF applies these with ``F.linear`` (which does ``x @ W.T``), so the
    parameters are stored ``[out_features, in_features]``. Our ``grouped_mm`` computes
    ``x @ w`` directly, so we transpose the last two dims before feeding them in.

    The expert compute (two grouped GEMMs + SwiGLU) is delegated to the
    ``GroupedExpertGEMM`` autograd Function; dispatch and combine stay here.

    Notes on SonicMoe:
    * fuse the local gather into prologue of grouped_gemm 1, fuse the local scatter into epilogue of grouped_gemm 2
    * reformulate the backward to remove the need to save some of the activations from fwd to bwd

    Notes on ScatterMoe:
    * same as SonicMoe except the backward reformulation, and less efficient kernels

    Notes on MoMoE:
    * similar to ScatterMoe, but can trade memory vs compute via AC (user can choose to recompute a grouped_gemm)

    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        n_routed_experts: int,
        hidden_act: str,
    ):
        super().__init__()
        self.num_experts = n_routed_experts
        self.hidden_dim = hidden_size
        self.intermediate_dim = moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(n_routed_experts, 2 * moe_intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(n_routed_experts, hidden_size, moe_intermediate_size)
        )
        # The manual backward in GroupedExpertGEMM hardcodes SwiGLU/SiLU.
        if hidden_act != "silu":
            raise ValueError(
                f"RoutedExperts only supports hidden_act='silu' (got {hidden_act!r}); "
                "the GroupedExpertGEMM backward is SwiGLU-specific."
            )
        self.act_fn = _get_activation(hidden_act)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: [T, H]   topk_indices/weights: [T, k]
        T, H = hidden_states.shape
        k = topk_indices.shape[1]


        #
        # --- DISPATCH ---
        #

        is_ep = False
        if is_ep:
            # Multi-GPU expert parallelism: experts are sharded across `ep_size` ranks,
            # E_local = E // ep_size experts per rank. A token's expert usually lives on
            # another rank, so the local gather below becomes a cross-GPU all-to-all.
            #
            # (local) flatten (token, expert) pairs:
            #   flat_tok = arange(T).repeat_interleave(k)   # [T*k]
            #   flat_exp = topk_indices.reshape(-1)          # [T*k] GLOBAL expert id
            #   flat_w   = topk_weights.reshape(-1)          # [T*k]
            # (local) decide which RANK each pair goes to, then bucket by rank:
            #   dest_rank  = flat_exp // E_local             # [T*k] in [0, ep_size)
            #   send_order = argsort(dest_rank)              # contiguous per dest rank
            #   (apply send_order to flat_tok/flat_exp/flat_w/dest_rank)
            #   send_counts = bincount(dest_rank, minlength=ep_size)   # [ep_size]
            #   send_tokens = hidden_states[flat_tok]        # [T*k, H], in send order
            # >>> metadata a2a: tell every rank how many rows it will receive
            #   recv_counts = all_to_all(send_counts)        # [ep_size]
            #   M_local     = int(recv_counts.sum())
            # >>> dispatch a2a (variable-sized): redistribute tokens + their metadata
            #   recv_tokens = all_to_all_v(send_tokens, send_counts, recv_counts)  # [M_local, H]
            #   recv_exp    = all_to_all_v(flat_exp,    send_counts, recv_counts)  # [M_local]
            #   recv_w      = all_to_all_v(flat_w,      send_counts, recv_counts)  # [M_local]
            # (local) received experts all belong to MY shard; re-sort by LOCAL expert:
            #   local_exp = recv_exp - ep_rank * E_local     # [M_local] in [0, E_local)
            #   perm      = argsort(local_exp)
            #   x_sorted  = recv_tokens[perm]                # [M_local, H] -> into GEMM
            #   (apply perm to local_exp/recv_w; keep send_order/perm/counts for combine)
            #   counts = bincount(local_exp, minlength=E_local)
            #   offs   = counts.cumsum(0)                    # [E_local], offs[-1] == M_local
            raise AssertionError("not implemented")
        else:
            # Single-GPU: all experts are local, so "dispatch" is a local sort + gather.
            # permute tokens into expert-contiguous order and build group offsets
            flat_tok = (
                torch.arange(T, device=hidden_states.device).repeat_interleave(k)
            )  # [T*k]
            flat_exp = topk_indices.reshape(-1)  # [T*k]
            flat_w = topk_weights.reshape(-1)  # [T*k]

            # Sort by expert id so each expert's tokens are contiguous (the "permute").
            order = torch.argsort(flat_exp)
            flat_tok = flat_tok[order]
            flat_exp = flat_exp[order]
            flat_w = flat_w[order]

            # Gather token rows into expert-major layout; build group offsets.
            x_sorted = hidden_states[flat_tok]  # [T*k, H]
            counts = torch.bincount(flat_exp, minlength=self.num_experts)
            offs = counts.cumsum(0)  # [E], cumulative end-row per expert

        #
        # --- EXPERT COMPUTE ---
        #

        # grouped GEMM + SwiGLU + grouped GEMM
        y = GroupedExpertGEMM.apply(x_sorted, self.gate_up_proj, self.down_proj, offs)

        #
        # --- COMBINE ---
        #

        if is_ep:
            # Mirror of dispatch: send each expert output back to the rank the token
            # came from. Same partner ranks, send/recv counts swapped.
            #
            # y is expert-major output for MY local experts, shape [M_local, H].
            # (local) undo the local expert-sort so rows line up with recv order:
            #   y_recv_order = empty_like(y); y_recv_order[perm] = y
            # >>> combine a2a (counts swapped relative to dispatch):
            #   y_back = all_to_all_v(y_recv_order, recv_counts, send_counts)  # [T*k, H]
            # (local) undo the by-rank send_order to restore original (token, expert) order:
            #   inv = argsort(send_order); y_back = y_back[inv]; w = flat_w[inv]
            # (local) scale by routing weights and scatter-add the k outputs per token:
            #   out = zeros_like(hidden_states)
            #   out.index_add_(0, original_flat_tok, y_back * w.unsqueeze(-1))
            raise AssertionError("not implemented")
        else:
            # Single-GPU: scale by routing weights, scatter-add back to token positions.
            y = y * flat_w.unsqueeze(-1).to(y.dtype)
            out = torch.zeros_like(hidden_states)
            out.index_add_(0, flat_tok, y)
            return out


class SharedExpert(nn.Module):
    """Plain SwiGLU MLP, names matching ``DeepseekV3MLP``.

    Intermediate size is ``moe_intermediate_size * n_shared_experts`` (DeepSeek stacks
    its shared experts into a single wider MLP).
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        n_shared_experts: int,
        hidden_act: str,
    ):
        super().__init__()
        intermediate = moe_intermediate_size * n_shared_experts
        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)
        self.act_fn = _get_activation(hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class RefMoE(nn.Module):
    """Dropless MoE block: routed experts (via grouped_mm) + additive shared expert.

    Mirrors ``DeepseekV3MoE.forward``:
        out = routed_experts(x) + shared_experts(x)
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        n_group: int,
        topk_group: int,
        n_shared_experts: int,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.gate = DeepseekRouter(
            hidden_size=hidden_size,
            n_routed_experts=n_routed_experts,
            top_k=num_experts_per_tok,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.experts = RoutedExperts(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            n_routed_experts=n_routed_experts,
            hidden_act=hidden_act,
        )
        self.shared_experts = SharedExpert(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            n_shared_experts=n_shared_experts,
            hidden_act=hidden_act,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        x = hidden_states.view(-1, hidden_states.shape[-1])
        routed = self.experts(x, topk_indices, topk_weights).view(*orig_shape)
        return routed + self.shared_experts(residuals)


def _get_activation(name: str):
    """Minimal activation lookup so this module stays transformers-independent."""
    acts = {
        "silu": F.silu,
        "gelu": F.gelu,
        "relu": F.relu,
    }
    if name not in acts:
        raise ValueError(f"unsupported hidden_act={name!r}; add it to _get_activation")
    return acts[name]
