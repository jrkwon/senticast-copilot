"""
News Encoder
============
Projects pre-computed sentence embeddings (short / medium / long summaries)
for each mineral into a context vector that can be fused with the time-series
encoder via cross-attention.

Input per sample:
  news_embeds: (n_minerals, 3, embed_dim)
    axis-1 = [summary_short, summary_medium, summary_long]

Output:
  news_context: (n_minerals, d_model)   – one vector per mineral
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NewsEncoder(nn.Module):
    """
    Encodes multi-granularity news summaries for each mineral.

    Processing pipeline:
    1. Linear projection: embed_dim → proj_dim per summary type.
    2. Temporal self-attention across the three summary granularities
       (short / medium / long) to weight their importance.
    3. Weighted sum → single context vector per mineral (B, M, proj_dim).
    """

    def __init__(
        self,
        embed_dim: int = 384,
        proj_dim: int = 128,
        n_minerals: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(embed_dim, proj_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(proj_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, news_embeds: torch.Tensor) -> torch.Tensor:
        """
        news_embeds: (B, M, 3, embed_dim)
        returns:     (B, M, proj_dim)
        """
        B, M, T, E = news_embeds.shape

        # Merge batch and mineral dims for parallel processing
        x = news_embeds.reshape(B * M, T, E)        # (B*M, 3, E)
        x = self.proj(x)                             # (B*M, 3, proj_dim)

        # Self-attention across the 3 summary types
        attn_out, _ = self.attn(x, x, x)            # (B*M, 3, proj_dim)
        x = self.norm(x + self.dropout(attn_out))   # residual

        # Aggregate over the 3 summary types (mean-pool)
        context = x.mean(dim=1)                      # (B*M, proj_dim)
        return context.reshape(B, M, -1)             # (B, M, proj_dim)


class NewsCrossAttention(nn.Module):
    """
    Fuses time-series features with news context via cross-attention.

    The time-series features act as queries; news context as keys/values.

    Inputs:
      ts_feat    : (B, L, d_model)  – time-series encoder output
      news_ctx   : (B, M, d_model)  – news context from NewsEncoder
    Output:
      fused      : (B, L, d_model)
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        ts_feat: torch.Tensor,
        news_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """
        ts_feat  : (B, L, d_model)
        news_ctx : (B, M, d_model)  – M = n_minerals (key/value sequence)
        returns  : (B, L, d_model)
        """
        attn_out, _ = self.cross_attn(
            query=ts_feat,
            key=news_ctx,
            value=news_ctx,
        )
        return self.norm(ts_feat + self.dropout(attn_out))
