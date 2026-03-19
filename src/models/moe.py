"""
Mixture of Experts (MoE)
========================
Implements a sparse MoE layer with Top-K routing and load-balancing loss.

Each "expert" is a small feed-forward network. A learned gating network
selects the top-K experts per token; only those experts process the token
and their outputs are weighted-summed.

Reference: "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer"
           (Shazeer et al., 2017)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single feed-forward expert network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixtureOfExperts(nn.Module):
    """
    Sparse MoE layer.

    Parameters
    ----------
    d_model        : token dimension
    num_experts    : total number of expert networks (E)
    top_k          : number of active experts per token
    d_ff           : feed-forward hidden size inside each expert
    capacity_factor: fraction of tokens each expert can handle
    dropout        : dropout inside experts
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        top_k: int = 2,
        d_ff: int | None = None,
        capacity_factor: float = 1.25,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
        self.num_experts    = num_experts
        self.top_k          = min(top_k, num_experts)
        self.capacity_factor = capacity_factor

        self.gate    = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x      : (B, L, d_model)
        returns: (out, aux_loss)
          out      : (B, L, d_model)
          aux_loss : scalar – load-balancing auxiliary loss
        """
        B, L, D = x.shape
        # Flatten tokens
        x_flat = x.reshape(B * L, D)                          # (N, D)
        N = x_flat.shape[0]

        # Gating logits & probabilities
        logits = self.gate(x_flat)                             # (N, E)
        probs  = F.softmax(logits, dim=-1)                     # (N, E)

        # Top-K selection
        topk_vals, topk_idx = torch.topk(probs, self.top_k, dim=-1)  # (N, K)
        topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)  # renormalise

        # Load-balancing auxiliary loss
        # fraction of tokens routed to each expert × mean gate probability
        dispatch_mask = torch.zeros(N, self.num_experts, device=x.device)
        dispatch_mask.scatter_(1, topk_idx, 1.0)
        tokens_per_expert = dispatch_mask.mean(dim=0)          # (E,)
        mean_gate_prob    = probs.mean(dim=0)                  # (E,)
        aux_loss = (tokens_per_expert * mean_gate_prob).sum() * self.num_experts

        # Dispatch tokens to experts
        out = torch.zeros_like(x_flat)                        # (N, D)
        for k in range(self.top_k):
            expert_indices = topk_idx[:, k]                   # (N,)
            weights        = topk_weights[:, k].unsqueeze(-1) # (N, 1)
            for e in range(self.num_experts):
                mask = (expert_indices == e)
                if mask.any():
                    out[mask] += weights[mask] * self.experts[e](x_flat[mask])

        return out.reshape(B, L, D), aux_loss
