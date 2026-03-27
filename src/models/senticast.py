"""
SentiCast – Main Model
======================
Architecture overview
---------------------
1. **GLAFF encoder** (Global-Local Adaptive Feature Fusion)
   - Input:  (B, L, M) price time-series
   - Output: (B, L, d_model) enriched temporal features

2. **Transformer encoder** (optional, n_layers > 0)
   - Self-attention over the L time steps
   - Output: (B, L, d_model)

3. **News fusion** (NewsEncoder + cross-attention)
   - news_embeds: (B, M, 3, embed_dim) → news context (B, M, d_model)
   - Cross-attention: ts_feat × news_ctx → (B, L, d_model)

4. **MoE feed-forward layer**
   - Applied after news fusion for capacity-efficient processing
   - Output: (B, L, d_model)

5. **Diffusion backbone**
   - Training: computes ε-prediction loss over noised targets
   - Inference: DDIM reverse sampling → (B, H, M) point predictions
     + multiple samples for confidence intervals

6. **Quantile head** (optional, for direct interval estimation)
   - Direct regression head producing (B, H, M, Q) outputs
   - Used alongside diffusion for training stability
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import DiffusionBackbone
from .glaff import GLAFF
from .moe import MixtureOfExperts
from .news_encoder import NewsCrossAttention, NewsEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# SentiCast
# ─────────────────────────────────────────────────────────────────────────────

class SentiCast(nn.Module):
    """
    Mineral price forecasting model.

    Parameters
    ----------
    n_minerals     : number of mineral channels (default 3)
    n_horizons     : number of prediction horizons (default 3 → 30/90/180 days)
    lookback       : input sequence length (default 180)
    d_model        : core model dimension
    n_heads        : attention heads
    n_layers       : transformer encoder layers
    dropout        : dropout probability
    news_embed_dim : pre-computed news embedding dimension
    glaff_cfg      : dict with top_k_freqs, local_kernel_sizes
    diffusion_cfg  : dict with num_steps, beta_start, beta_end, etc.
    moe_cfg        : dict with num_experts, top_k, capacity_factor
    quantiles      : list of quantiles for direct prediction head
    """

    def __init__(
        self,
        n_minerals: int = 3,
        n_horizons: int = 3,
        lookback: int = 180,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        news_embed_dim: int = 384,
        glaff_cfg: Optional[Dict] = None,
        diffusion_cfg: Optional[Dict] = None,
        moe_cfg: Optional[Dict] = None,
        quantiles: Optional[List[float]] = None,
    ):
        super().__init__()
        glaff_cfg     = glaff_cfg     or {}
        diffusion_cfg = diffusion_cfg or {}
        moe_cfg       = moe_cfg       or {}
        self.quantiles = quantiles or [0.05, 0.50, 0.95]
        Q = len(self.quantiles)

        self.n_minerals = n_minerals
        self.n_horizons = n_horizons
        self.lookback   = lookback
        self.d_model    = d_model

        # ── 1. GLAFF encoder ──────────────────────────────────────────────────
        self.glaff = GLAFF(
            seq_len=lookback,
            in_channels=n_minerals,
            d_model=d_model,
            top_k_freqs=glaff_cfg.get("top_k_freqs", 5),
            local_kernel_sizes=glaff_cfg.get("local_kernel_sizes", [3, 7, 14]),
        )

        # ── 2. Positional encoding + Transformer encoder ───────────────────────
        self.pos_enc = PositionalEncoding(d_model, max_len=lookback + 10, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── 3. News encoder + cross-attention ─────────────────────────────────
        self.news_enc = NewsEncoder(
            embed_dim=news_embed_dim,
            proj_dim=d_model,
            n_minerals=n_minerals,
            num_heads=n_heads,
            dropout=dropout,
        )
        self.news_cross_attn = NewsCrossAttention(
            d_model=d_model,
            num_heads=n_heads,
            dropout=dropout,
        )

        # ── 4. MoE feed-forward ───────────────────────────────────────────────
        self.moe = MixtureOfExperts(
            d_model=d_model,
            num_experts=moe_cfg.get("num_experts", 4),
            top_k=moe_cfg.get("top_k", 2),
            capacity_factor=moe_cfg.get("capacity_factor", 1.25),
            dropout=dropout,
        )
        self.moe_norm = nn.LayerNorm(d_model)

        # ── 5. Diffusion backbone ─────────────────────────────────────────────
        self.diffusion = DiffusionBackbone(
            n_minerals=n_minerals,
            n_horizons=n_horizons,
            d_model=d_model,
            context_len=lookback,
            num_steps=diffusion_cfg.get("num_steps", 100),
            beta_start=diffusion_cfg.get("beta_start", 1e-4),
            beta_end=diffusion_cfg.get("beta_end", 0.02),
            schedule=diffusion_cfg.get("schedule", "linear"),
            inference_steps=diffusion_cfg.get("inference_steps", 20),
            n_heads=n_heads,
            n_layers=max(n_layers // 2, 1),
            dropout=dropout,
        )

        # ── 6. Quantile regression head ───────────────────────────────────────
        # Lightweight MLP on the pooled context → direct predictions
        self.quant_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_horizons * n_minerals * Q),
        )
        self.Q = Q

    # ── Forward ───────────────────────────────────────────────────────────────

    def encode(
        self,
        price_series: torch.Tensor,                   # (B, L, M)
        news_embeds:  torch.Tensor,                   # (B, M, 3, embed_dim)
        news_mask:    Optional[torch.Tensor] = None,  # (B, M) bool
    ) -> torch.Tensor:
        """Return context tensor (B, L, d_model) for conditioning."""
        # GLAFF
        x = self.glaff(price_series)                          # (B, L, d_model)
        # Positional encoding + Transformer
        x = self.pos_enc(x)
        x = self.transformer(x)                               # (B, L, d_model)
        # News fusion – pass mask so missing-news minerals are gated to zero
        news_ctx = self.news_enc(news_embeds, news_mask)      # (B, M, d_model)
        x = self.news_cross_attn(x, news_ctx, news_mask)      # (B, L, d_model)
        # MoE
        moe_out, aux_loss = self.moe(x)
        x = self.moe_norm(x + moe_out)                        # (B, L, d_model)
        # Store aux loss as attribute so training loop can access it
        self._moe_aux_loss = aux_loss
        return x

    def forward(
        self,
        price_series: torch.Tensor,                   # (B, L, M)
        news_embeds:  torch.Tensor,                   # (B, M, 3, embed_dim)
        targets:      Optional[torch.Tensor] = None,  # (B, H, M) normalised targets
        news_mask:    Optional[torch.Tensor] = None,  # (B, M) bool
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Returns dict with keys:
          - "diffusion_loss" : scalar diffusion ε-prediction loss
          - "quant_preds"    : (B, H, M, Q) quantile predictions
          - "moe_aux_loss"   : scalar MoE load-balancing loss
        """
        context = self.encode(price_series, news_embeds, news_mask)   # (B, L, d_model)

        out: Dict[str, torch.Tensor] = {}

        # Diffusion loss (training only)
        if targets is not None:
            out["diffusion_loss"] = self.diffusion.diffusion_loss(targets, context)

        # Quantile head (always)
        # Use the last time-step context (most recent market state) instead of
        # the mean-pool which discards the current price level.
        pooled = context[:, -1, :]                        # (B, d_model)
        q_flat = self.quant_head(pooled)                  # (B, H*M*Q)
        B = price_series.shape[0]
        q_preds = q_flat.reshape(B, self.n_horizons, self.n_minerals, self.Q)
        # Last-price residual anchor: the head predicts deviations from the
        # last observed (RevIN-normalised) price so that even an untrained
        # model produces reasonable starting predictions.
        last_price = price_series[:, -1, :].unsqueeze(1).unsqueeze(-1)  # (B,1,M,1)
        out["quant_preds"] = q_preds + last_price

        out["moe_aux_loss"] = self._moe_aux_loss
        return out

    @torch.no_grad()
    def predict(
        self,
        price_series: torch.Tensor,                   # (B, L, M)
        news_embeds:  torch.Tensor,                   # (B, M, 3, embed_dim)
        n_samples:    int = 20,
        news_mask:    Optional[torch.Tensor] = None,  # (B, M) bool
    ) -> Dict[str, torch.Tensor]:
        """
        Inference: return point predictions and confidence intervals.

        Returns dict with keys:
          - "mean"   : (B, H, M)    – ensemble mean (diffusion)
          - "lower"  : (B, H, M)    – 5th percentile
          - "upper"  : (B, H, M)    – 95th percentile
          - "median" : (B, H, M)    – quantile head median prediction
        """
        context = self.encode(price_series, news_embeds, news_mask)

        # Diffusion ensemble
        samples = self.diffusion.sample(context, n_samples=n_samples, return_all=True)
        # samples: (B, S, H, M)
        mean  = samples.mean(dim=1)                        # (B, H, M)
        lower = samples.quantile(0.05, dim=1)
        upper = samples.quantile(0.95, dim=1)

        # Quantile head median
        pooled  = context[:, -1, :]
        q_flat  = self.quant_head(pooled)
        B = price_series.shape[0]
        q_preds = q_flat.reshape(B, self.n_horizons, self.n_minerals, self.Q)
        last_price = price_series[:, -1, :].unsqueeze(1).unsqueeze(-1)  # (B,1,M,1)
        q_preds = q_preds + last_price
        median_idx = self.quantiles.index(0.50) if 0.50 in self.quantiles else self.Q // 2
        median = q_preds[..., median_idx]

        return {"mean": mean, "lower": lower, "upper": upper, "median": median}


# ─────────────────────────────────────────────────────────────────────────────
# Factory helper
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict, n_minerals: int = 3) -> SentiCast:
    """Build a SentiCast model from a config dict."""
    mcfg = cfg.get("model", {})
    return SentiCast(
        n_minerals=n_minerals,
        n_horizons=len(cfg.get("data", {}).get("horizons", [30, 90, 180])),
        lookback=cfg.get("data", {}).get("lookback", 180),
        d_model=mcfg.get("d_model", 128),
        n_heads=mcfg.get("n_heads", 8),
        n_layers=mcfg.get("n_layers", 4),
        dropout=mcfg.get("dropout", 0.1),
        news_embed_dim=cfg.get("data", {}).get("news_embed_dim", 384),
        glaff_cfg=mcfg.get("glaff", {}),
        diffusion_cfg=mcfg.get("diffusion", {}),
        moe_cfg=mcfg.get("moe", {}),
        quantiles=cfg.get("training", {}).get("quantiles", [0.05, 0.50, 0.95]),
    )
