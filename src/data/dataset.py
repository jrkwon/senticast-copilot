"""
PyTorch Dataset for SentiCast.

Each sample corresponds to a reference date T and contains:
  - price_series : (lookback, n_minerals)          – normalised prices
  - news_embeds  : (n_minerals, 3, embed_dim)       – short/medium/long news
  - news_mask    : (n_minerals,)  bool              – True where news is available
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
    news_mask : np.ndarray or None, shape (T_total, n_minerals), dtype bool
        ``True`` where a real news row exists.  When ``None``, all entries are
        treated as available (backward-compatible with the sample-data path).
    use_revin : bool
        When True (default) each sample is re-normalised by its own lookback-
        window mean/std (Reversible Instance Normalisation).  This eliminates
        the distribution shift that occurs when test-period prices lie far
        outside the training-period range.  The batch then also carries
        ``revin_mean`` and ``revin_std`` tensors so that predictions can be
        inverse-transformed back to the original price space.
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
        news_mask: Optional[np.ndarray] = None,
        use_revin: bool = True,
    ):
        self.prices = prices_norm
        self.news   = news_tensor
        self.dates  = dates
        self.horizons = sorted(horizons)
        self.lookback = lookback
        self.max_horizon = max(horizons)
        self.use_revin = use_revin
        # All-True sentinel when no mask is supplied (sample-data path)
        n_minerals = prices_norm.shape[1]
        if news_mask is None:
            news_mask = np.ones((len(prices_norm), n_minerals), dtype=bool)
        self.news_mask = news_mask

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
        price_series = self.prices[t - self.lookback : t].copy()  # (L, M)

        # Targets: prices at t + h for each horizon h
        targets = np.stack([self.prices[t + h - 1] for h in self.horizons], axis=0)  # (H, M)

        # RevIN – per-window instance normalisation.
        # Normalises both the input window and the targets by the window's own
        # mean/std so that the model always sees a stationary, unit-variance
        # input regardless of where in the price history the window falls.
        # This eliminates the train→test distribution shift caused by long-term
        # price trends (e.g. gold rising from $1 300 to $5 300 over the dataset).
        if self.use_revin:
            revin_mean = price_series.mean(axis=0)                       # (M,)
            revin_std  = np.maximum(price_series.std(axis=0), 1e-6)     # (M,)
            price_series = (price_series - revin_mean) / revin_std
            targets      = (targets      - revin_mean) / revin_std

        # News at reference date t (or last available)
        news_embeds = self.news[t - 1]                             # (M, 3, E)
        news_mask   = self.news_mask[t - 1]                        # (M,) bool

        result: Dict[str, torch.Tensor] = {
            "price_series": torch.tensor(price_series, dtype=torch.float32),
            "news_embeds":  torch.tensor(news_embeds,  dtype=torch.float32),
            "news_mask":    torch.tensor(news_mask,    dtype=torch.bool),
            "targets":      torch.tensor(targets,      dtype=torch.float32),
            "ref_idx":      torch.tensor(t,            dtype=torch.long),
        }
        if self.use_revin:
            result["revin_mean"] = torch.tensor(revin_mean, dtype=torch.float32)
            result["revin_std"]  = torch.tensor(revin_std,  dtype=torch.float32)
        return result


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
    news_mask: Optional[np.ndarray] = None,
    use_revin: bool = True,
) -> Tuple["MineralDataset", "MineralDataset", "MineralDataset"]:
    """Return (train_ds, val_ds, test_ds) for a given DataSplit."""
    kwargs = dict(
        prices_norm=prices_norm,
        news_tensor=news_tensor,
        dates=dates,
        horizons=horizons,
        lookback=lookback,
        news_mask=news_mask,
        use_revin=use_revin,
    )
    train_ds = MineralDataset(**kwargs, start=split.train[0], end=split.train[1])
    val_ds   = MineralDataset(**kwargs, start=split.val[0],   end=split.val[1])
    test_ds  = MineralDataset(**kwargs, start=split.test[0],  end=split.test[1])
    return train_ds, val_ds, test_ds
