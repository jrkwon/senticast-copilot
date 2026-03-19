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
    """

    def __init__(self, seq_len: int, d_model: int, top_k: int = 5):
        super().__init__()
        self.seq_len = seq_len
        self.top_k   = top_k
        # Learnable projection from reconstructed sequence → d_model
        self.proj = nn.Linear(seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)  →  out: (B, C, d_model)
        """
        # FFT along time dimension
        x_f = torch.fft.rfft(x, dim=1, norm="ortho")         # (B, L//2+1, C)

        amp = x_f.abs()                                       # (B, F, C)
        # Keep only top-K frequencies per channel
        topk_vals, topk_idx = torch.topk(amp, self.top_k, dim=1)  # (B, K, C)
        mask = torch.zeros_like(amp)
        mask.scatter_(1, topk_idx, 1.0)
        x_f_filtered = x_f * mask

        # Inverse FFT → filtered time-domain signal
        x_filtered = torch.fft.irfft(x_f_filtered, n=self.seq_len, dim=1, norm="ortho")  # (B, L, C)

        # Aggregate over time: transpose so projection is over L
        x_filtered = x_filtered.permute(0, 2, 1)             # (B, C, L)
        out = self.proj(x_filtered)                           # (B, C, d_model)
        return out


class LocalBlock(nn.Module):
    """
    Multi-scale depthwise convolution to capture local temporal patterns.
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
        x: (B, L, C)  →  out: (B, C, d_model)
        """
        x_t = x.permute(0, 2, 1)                             # (B, C, L)
        L = x_t.shape[-1]
        weights = F.softmax(self.scale_weight, dim=0)
        # Trim to L to handle even kernel sizes where padding may add 1 extra step
        outs = [w * conv(x_t)[..., :L] for w, conv in zip(weights, self.convs)]
        out = sum(outs)                                        # (B, d_model, L)
        # Average-pool over time
        out = out.mean(dim=-1)                                 # (B, d_model)
        # Expand to (B, C, d_model) by repeating so shapes match SpectralBlock
        B, C, _ = x_t.shape
        out = out.unsqueeze(1).expand(-1, C, -1)              # (B, C, d_model)
        return out


class GLAFF(nn.Module):
    """
    Global-Local Adaptive Feature Fusion module.

    Combines global (spectral) and local (multi-scale conv) features using
    a learned adaptive gate, then projects to *d_model* per channel.

    Input:  (B, L, C)   – L time steps, C minerals/features
    Output: (B, L, d_model)  – enriched temporal features
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
        self.global_branch = SpectralBlock(seq_len, d_model, top_k=top_k_freqs)
        self.local_branch  = LocalBlock(in_channels, d_model, kernel_sizes=local_kernel_sizes)

        # Gate: scalar weight per channel deciding how much of global vs local to use
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Final projection from fused (B, C, d_model) → (B, L, d_model)
        self.output_proj = nn.Linear(in_channels * d_model, seq_len * d_model)
        self.seq_len = seq_len
        self.d_model = d_model
        self.in_channels = in_channels

        # Layer norm for stable training
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)  →  out: (B, L, d_model)
        """
        B, L, C = x.shape

        g = self.global_branch(x)   # (B, C, d_model)
        lo = self.local_branch(x)   # (B, C, d_model)

        # Adaptive gate per channel
        combined = torch.cat([g, lo], dim=-1)   # (B, C, 2*d_model)
        alpha = self.gate(combined)              # (B, C, 1)
        fused = alpha * g + (1 - alpha) * lo    # (B, C, d_model)

        # Reshape to (B, L, d_model) by projecting flattened channels
        fused_flat = fused.reshape(B, C * self.d_model)       # (B, C*d_model)
        out = self.output_proj(fused_flat)                     # (B, L*d_model)
        out = out.reshape(B, L, self.d_model)                  # (B, L, d_model)
        return self.norm(out)
