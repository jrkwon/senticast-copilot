"""
GLAFF – Global-Local Adaptive Feature Fusion
============================================
Based on the core idea of "GLAFF: Robust Time Series Forecasting":
  1. **Global branch** – FFT-based spectral feature extraction retaining
     the top-K dominant frequencies to capture long-range periodicity.
  2. **Local branch** – Multi-scale depthwise convolutions for short-range
     patterns (trend, recent anomalies).
  3. **Adaptive gate** – learned sigmoid gate merges global + local features.

Reference concept: Adaptive Frequency Filtering for Robust Time-Series Forecasting.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralBlock(nn.Module):
    """
    Extracts global periodicity features via Fast Fourier Transform.

    Keeps only the ``top_k`` dominant (highest-amplitude) frequency components
    and reconstructs a clean, de-noised version of the input sequence.
    The reconstructed signal is then projected channel-wise (C → d_model) at
    every time step so that the full temporal axis is preserved.
    """

    def __init__(self, seq_len: int, in_channels: int, d_model: int, top_k: int = 5):
        super().__init__()
        self.seq_len = seq_len
        self.top_k   = top_k
        # Project C → d_model independently at each time step (temporal axis intact)
        self.proj = nn.Linear(in_channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)  →  out: (B, L, d_model)
        """
        # FFT along time dimension
        x_f = torch.fft.rfft(x, dim=1, norm="ortho")         # (B, L//2+1, C)

        amp = x_f.abs()                                       # (B, F, C)
        # Keep only top-K frequencies per channel
        topk_vals, topk_idx = torch.topk(amp, self.top_k, dim=1)  # (B, K, C)
        mask = torch.zeros_like(amp)
        mask.scatter_(1, topk_idx, 1.0)
        x_f_filtered = x_f * mask

        # Inverse FFT → filtered time-domain signal (B, L, C) — temporal axis preserved
        x_filtered = torch.fft.irfft(x_f_filtered, n=self.seq_len, dim=1, norm="ortho")

        # Project channel dimension: (B, L, C) → (B, L, d_model)
        return self.proj(x_filtered)


class LocalBlock(nn.Module):
    """
    Multi-scale depthwise convolution to capture local temporal patterns.
    The temporal dimension is fully preserved so the Transformer downstream
    can attend across meaningful, distinct time steps.
    """

    def __init__(self, in_channels: int, d_model: int, kernel_sizes: List[int] = (3, 7, 14)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels, in_channels,
                    kernel_size=k,
                    padding=k // 2,
                    groups=in_channels,  # depthwise
                    bias=False,
                ),
                nn.Conv1d(in_channels, d_model, kernel_size=1),  # pointwise
                nn.GELU(),
            )
            for k in kernel_sizes
        ])
        self.scale_weight = nn.Parameter(torch.ones(len(kernel_sizes)) / len(kernel_sizes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)  →  out: (B, L, d_model)
        """
        x_t = x.permute(0, 2, 1)                             # (B, C, L)
        L = x_t.shape[-1]
        weights = F.softmax(self.scale_weight, dim=0)
        # Trim to L to handle even kernel sizes where padding may add 1 extra step
        outs = [w * conv(x_t)[..., :L] for w, conv in zip(weights, self.convs)]
        out = sum(outs)                                        # (B, d_model, L)
        # Transpose to keep temporal axis: (B, d_model, L) → (B, L, d_model)
        return out.permute(0, 2, 1)


class GLAFF(nn.Module):
    """
    Global-Local Adaptive Feature Fusion module.

    Combines global (spectral) and local (multi-scale conv) features using
    a learned adaptive gate, then projects to *d_model* per time step.

    Input:  (B, L, C)   – L time steps, C minerals/features
    Output: (B, L, d_model)  – enriched temporal features

    Both the SpectralBlock and the LocalBlock now preserve the full temporal
    axis (shape ``(B, L, d_model)``), so the downstream Transformer receives
    L genuinely distinct tokens rather than L copies of the same summary.
    """

    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        d_model: int,
        top_k_freqs: int = 5,
        local_kernel_sizes: List[int] = (3, 7, 14),
    ):
        super().__init__()
        self.global_branch = SpectralBlock(seq_len, in_channels, d_model, top_k=top_k_freqs)
        self.local_branch  = LocalBlock(in_channels, d_model, kernel_sizes=local_kernel_sizes)

        # Gate: learned sigmoid weight per time step deciding how much global vs local to use
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Layer norm for stable training
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)  →  out: (B, L, d_model)
        """
        g  = self.global_branch(x)              # (B, L, d_model)
        lo = self.local_branch(x)               # (B, L, d_model)

        # Adaptive gate per time step
        combined = torch.cat([g, lo], dim=-1)   # (B, L, 2*d_model)
        alpha = self.gate(combined)             # (B, L, 1)
        fused = alpha * g + (1 - alpha) * lo   # (B, L, d_model)

        return self.norm(fused)
