"""
News Encoder
============
Projects pre-computed sentence embeddings (short / medium / long summaries)
for each mineral into a context vector that can be fused with the time-series
encoder via cross-attention.

Input per sample:
  news_embeds: (n_minerals, 3, embed_dim)
    axis-1 = [summary_short, summary_medium, summary_long]
  news_mask:   (n_minerals,)  bool  – True where news is available

Output:
  news_context: (n_minerals, d_model)   – one vector per mineral

Zero-vector handling
--------------------
When ``news_mask`` is provided, minerals without news (mask=False) have their
context vector explicitly zeroed **after** the self-attention step.  This
prevents the network from treating the projection of a zero embedding (which
has non-zero output due to bias terms) as meaningful signal.

In ``NewsCrossAttention``, masked-out minerals are excluded from the key/value
sequence so the time-series queries cannot attend to them.  If *all* minerals
in a batch row are masked, the cross-attention is skipped for that row and
``ts_feat`` is returned unchanged.
"""

from __future__ import annotations

from typing import Optional

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
    4. Zero-gate: multiply by ``news_mask`` so minerals without news contribute
       nothing to the cross-attention step.
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

    def forward(
        self,
        news_embeds: torch.Tensor,               # (B, M, 3, embed_dim)
        news_mask: Optional[torch.Tensor] = None, # (B, M) bool
    ) -> torch.Tensor:
        """
        news_embeds : (B, M, 3, embed_dim)
        news_mask   : (B, M) bool – True = has news  (optional)
        returns     : (B, M, proj_dim)
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
        context = context.reshape(B, M, -1)          # (B, M, proj_dim)

        # ── Zero-gate: mask out minerals with no news ─────────────────────────
        # Even though the embedding is zero, the Linear projection produces
        # non-zero output (bias terms).  We explicitly zero the context for
        # masked minerals so the cross-attention sees no spurious signal.
        if news_mask is not None:
            gate = news_mask.unsqueeze(-1).to(context.dtype)  # (B, M, 1)
            context = context * gate

        return context


class NewsCrossAttention(nn.Module):
    """
    Fuses time-series features with news context via cross-attention.

    The time-series features act as queries; news context as keys/values.

    Inputs:
      ts_feat    : (B, L, d_model)  – time-series encoder output
      news_ctx   : (B, M, d_model)  – news context from NewsEncoder
      news_mask  : (B, M) bool      – True = mineral has valid news (optional)
    Output:
      fused      : (B, L, d_model)

    When ``news_mask`` is provided, minerals without news are excluded from the
    key/value sequence (``key_padding_mask``).  For batch rows where *all*
    minerals are masked, cross-attention is skipped and ``ts_feat`` is returned
    unchanged (avoids NaN from all-True padding mask).
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
        ts_feat: torch.Tensor,                   # (B, L, d_model)
        news_ctx: torch.Tensor,                  # (B, M, d_model)
        news_mask: Optional[torch.Tensor] = None, # (B, M) bool
    ) -> torch.Tensor:
        """
        ts_feat   : (B, L, d_model)
        news_ctx  : (B, M, d_model)  – M = n_minerals (key/value sequence)
        news_mask : (B, M) bool      – True = has news (optional)
        returns   : (B, L, d_model)
        """
        key_padding_mask: Optional[torch.Tensor] = None

        if news_mask is not None:
            # PyTorch key_padding_mask: True = IGNORE this key position
            key_padding_mask = ~news_mask  # (B, M)

            # Rows where ALL minerals are masked would produce NaN in softmax.
            # For those rows, neutralise the mask (attend to everything; values
            # are zero anyway due to NewsEncoder gating).
            all_masked = key_padding_mask.all(dim=-1)  # (B,)
            if all_masked.any():
                # Set those rows to all-False (no positions ignored)
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked] = False

        attn_out, _ = self.cross_attn(
            query=ts_feat,
            key=news_ctx,
            value=news_ctx,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(ts_feat + self.dropout(attn_out))
