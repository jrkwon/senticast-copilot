"""
Diffusion backbone for SentiCast
==================================
Implements a DDPM-style conditional denoising diffusion model for
multi-horizon time-series prediction.

Forward (training):
  Given clean future prices y_0 ∈ ℝ^(H, M), add Gaussian noise to
  produce y_t at diffusion step t.

Reverse (inference):
  Starting from Gaussian noise y_T, iteratively denoise conditioned on
  the encoded history context c ∈ ℝ^(B, L, d_model).

The noise-prediction network (ε-network) is a small Transformer that
attends over both the noisy future sequence and the conditioning context.

Supports both DDPM (full T steps) and DDIM (accelerated, inference_steps < T).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Noise schedule
# ─────────────────────────────────────────────────────────────────────────────

def _linear_beta_schedule(num_steps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps)


def _cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    steps = num_steps + 1
    x = torch.linspace(0, num_steps, steps)
    alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.9999)


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal timestep embedding
# ─────────────────────────────────────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.d_model = d_model

    @staticmethod
    def _sinusoidal(t: torch.Tensor, d_model: int) -> torch.Tensor:
        half = d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self._sinusoidal(t, self.d_model)
        return self.linear2(F.silu(self.linear1(emb)))


# ─────────────────────────────────────────────────────────────────────────────
# Noise prediction network
# ─────────────────────────────────────────────────────────────────────────────

class NoisePredictor(nn.Module):
    """
    Transformer-based ε-network for diffusion.

    Architecture:
      - Project noisy future y_t  → d_model
      - Add timestep embedding
      - Cross-attend over conditioning context c
      - Project → output shape (H, M)
    """

    def __init__(
        self,
        n_minerals: int,
        n_horizons: int,
        d_model: int,
        context_len: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj   = nn.Linear(n_minerals, d_model)
        self.time_embed   = TimestepEmbedding(d_model)
        self.pos_emb      = nn.Embedding(n_horizons, d_model)

        encoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder    = nn.TransformerDecoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, n_minerals)
        self.n_horizons  = n_horizons

    def forward(
        self,
        y_t: torch.Tensor,      # (B, H, M)  noisy future prices
        t: torch.Tensor,        # (B,)        diffusion timestep
        context: torch.Tensor,  # (B, L, d_model)  encoder output
    ) -> torch.Tensor:
        """Returns predicted noise ε, shape (B, H, M)."""
        B, H, M = y_t.shape

        x = self.input_proj(y_t)                              # (B, H, d_model)
        t_emb = self.time_embed(t).unsqueeze(1)               # (B, 1, d_model)
        pos = self.pos_emb(torch.arange(H, device=y_t.device))
        x = x + t_emb + pos                                   # (B, H, d_model)

        x = self.decoder(tgt=x, memory=context)               # (B, H, d_model)
        return self.output_proj(x)                            # (B, H, M)


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion backbone
# ─────────────────────────────────────────────────────────────────────────────

class DiffusionBackbone(nn.Module):
    """
    DDPM-style conditional diffusion backbone.

    Parameters
    ----------
    n_minerals    : number of mineral price channels
    n_horizons    : number of prediction horizons
    d_model       : model dimension
    context_len   : length of conditioning sequence (lookback)
    num_steps     : total diffusion steps T
    beta_start/end: noise schedule end-points
    schedule      : "linear" | "cosine"
    inference_steps: DDIM steps at inference (< num_steps for speed)
    """

    def __init__(
        self,
        n_minerals: int,
        n_horizons: int,
        d_model: int,
        context_len: int,
        num_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
        inference_steps: int = 20,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.T               = num_steps
        self.inference_steps = min(inference_steps, num_steps)
        self.n_horizons      = n_horizons
        self.n_minerals      = n_minerals

        if schedule == "cosine":
            betas = _cosine_beta_schedule(num_steps)
        else:
            betas = _linear_beta_schedule(num_steps, beta_start, beta_end)

        alphas      = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (moved with .to(device))
        self.register_buffer("betas",               betas)
        self.register_buffer("alphas",              alphas)
        self.register_buffer("alpha_cumprod",       alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev",  alpha_cumprod_prev)
        self.register_buffer("sqrt_alpha_cumprod",  alpha_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_cumprod", (1.0 - alpha_cumprod).sqrt())
        # Posterior variance
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod),
        )

        self.eps_net = NoisePredictor(
            n_minerals=n_minerals,
            n_horizons=n_horizons,
            d_model=d_model,
            context_len=context_len,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def q_sample(
        self,
        y_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: add noise to y_0 at timestep t.
        Returns (y_t, noise).
        """
        if noise is None:
            noise = torch.randn_like(y_0)
        sqrt_ac   = self.sqrt_alpha_cumprod[t][:, None, None]
        sqrt_omac = self.sqrt_one_minus_alpha_cumprod[t][:, None, None]
        y_t = sqrt_ac * y_0 + sqrt_omac * noise
        return y_t, noise

    def diffusion_loss(
        self,
        y_0: torch.Tensor,      # (B, H, M)
        context: torch.Tensor,  # (B, L, d_model)
    ) -> torch.Tensor:
        """L2 loss between predicted and actual noise."""
        B = y_0.shape[0]
        t = torch.randint(0, self.T, (B,), device=y_0.device)
        y_t, noise = self.q_sample(y_0, t)
        noise_pred = self.eps_net(y_t, t, context)
        return F.mse_loss(noise_pred, noise)

    # ── Inference (DDIM) ──────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,        # (B, L, d_model)
        n_samples: int = 1,
        return_all: bool = False,
    ) -> torch.Tensor:
        """
        DDIM-accelerated reverse diffusion.

        Returns predicted y_0 of shape (B, H, M), or (B, S, H, M) if
        return_all=True (S = n_samples for ensemble averaging).
        """
        B = context.shape[0]
        device = context.device

        # Build DDIM timestep subsequence
        step_size = self.T // self.inference_steps
        ddim_steps = list(range(0, self.T, step_size))[::-1]  # high → low

        all_samples: List[torch.Tensor] = []

        for _ in range(n_samples):
            y_t = torch.randn(B, self.n_horizons, self.n_minerals, device=device)

            for t_idx in ddim_steps:
                t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)
                eps = self.eps_net(y_t, t_batch, context)

                alpha_t    = self.alpha_cumprod[t_idx]
                alpha_prev = self.alpha_cumprod[max(t_idx - step_size, 0)]

                x0_pred = (y_t - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
                x0_pred = x0_pred.clamp(-10, 10)

                # DDIM update (deterministic)
                y_t = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * eps

            all_samples.append(y_t)

        stacked = torch.stack(all_samples, dim=1)  # (B, S, H, M)
        if return_all:
            return stacked
        return stacked.mean(dim=1)                 # (B, H, M)
