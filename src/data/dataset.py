"""
PyTorch Dataset for SentiCast.

Each sample corresponds to a reference date T and contains:
  - price_series : (lookback, n_minerals)          – normalised prices
  - news_embeds  : (n_minerals, 3, embed_dim)       – short/medium/long news
  - targets      : (n_horizons, n_minerals)         – future prices (normalised)
  - target_dates : list of date strings (for logging)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MineralDataset(Dataset):
    """
    Sliding-window dataset over a contiguous index range [start, end) of the
    full price / news arrays.

    Parameters
    ----------
    prices_norm : np.ndarray, shape (T_total, n_minerals)
        Normalised price matrix (full timeline).
    news_tensor : np.ndarray, shape (T_total, n_minerals, 3, embed_dim)
        Pre-computed news embeddings (full timeline).
    dates : pd.DatetimeIndex
        Full sorted date index corresponding to rows in prices_norm.
    horizons : List[int]
        Prediction horizons in days, e.g. [30, 90, 180].
    lookback : int
        Number of past days fed as input (default 180).
    start : int
        First valid index in the full timeline for this split.
    end : int
        One-past-last valid index for this split.
    """

    def __init__(
        self,
        prices_norm: np.ndarray,
        news_tensor: np.ndarray,
        dates: pd.DatetimeIndex,
        horizons: List[int],
        lookback: int = 180,
        start: int = 0,
        end: Optional[int] = None,
    ):
        self.prices = prices_norm
        self.news   = news_tensor
        self.dates  = dates
        self.horizons = sorted(horizons)
        self.lookback = lookback
        self.max_horizon = max(horizons)

        total = len(prices_norm)
        end = end if end is not None else total
        end = min(end, total)

        # Valid reference indices: need lookback days before and max_horizon after
        self.indices = [
            i for i in range(max(start, lookback), end)
            if i + self.max_horizon <= total
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.indices[idx]

        # Input: prices from [t-lookback, t)
        price_series = self.prices[t - self.lookback : t]          # (L, M)

        # News at reference date t (or last available)
        news_embeds = self.news[t - 1]                             # (M, 3, E)

        # Targets: prices at t + h for each horizon h
        targets = np.stack([self.prices[t + h - 1] for h in self.horizons], axis=0)  # (H, M)

        return {
            "price_series": torch.tensor(price_series, dtype=torch.float32),
            "news_embeds":  torch.tensor(news_embeds,  dtype=torch.float32),
            "targets":      torch.tensor(targets,      dtype=torch.float32),
            "ref_idx":      torch.tensor(t,            dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(
    prices_norm: np.ndarray,
    news_tensor: np.ndarray,
    dates: pd.DatetimeIndex,
    split,          # DataSplit namedtuple (train/val/test index ranges)
    horizons: List[int],
    lookback: int = 180,
) -> Tuple["MineralDataset", "MineralDataset", "MineralDataset"]:
    """Return (train_ds, val_ds, test_ds) for a given DataSplit."""
    kwargs = dict(
        prices_norm=prices_norm,
        news_tensor=news_tensor,
        dates=dates,
        horizons=horizons,
        lookback=lookback,
    )
    train_ds = MineralDataset(**kwargs, start=split.train[0], end=split.train[1])
    val_ds   = MineralDataset(**kwargs, start=split.val[0],   end=split.val[1])
    test_ds  = MineralDataset(**kwargs, start=split.test[0],  end=split.test[1])
    return train_ds, val_ds, test_ds
