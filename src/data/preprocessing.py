"""
Data loading, normalisation and rolling-forward train/val/test splitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Normaliser
# ─────────────────────────────────────────────────────────────────────────────

class Normalizer:
    """Per-mineral z-score or min-max normaliser fitted on training data only."""

    def __init__(self, method: str = "zscore"):
        if method not in ("zscore", "minmax"):
            raise ValueError(f"Unknown normalization method: {method!r}")
        self.method = method
        self._params: dict[str, dict[str, float]] = {}

    def fit(self, df: pd.DataFrame, minerals: List[str]) -> "Normalizer":
        for m in minerals:
            if self.method == "zscore":
                self._params[m] = {"mean": float(df[m].mean()), "std": float(df[m].std()) or 1.0}
            else:
                self._params[m] = {"min": float(df[m].min()), "max": float(df[m].max()) or 1.0}
        return self

    def transform(self, df: pd.DataFrame, minerals: List[str]) -> pd.DataFrame:
        df = df.copy()
        for m in minerals:
            p = self._params[m]
            if self.method == "zscore":
                df[m] = (df[m] - p["mean"]) / p["std"]
            else:
                df[m] = (df[m] - p["min"]) / (p["max"] - p["min"])
        return df

    def inverse_transform(self, arr: np.ndarray, mineral_idx: int, mineral: str) -> np.ndarray:
        """Invert normalisation for a single mineral column."""
        p = self._params[mineral]
        if self.method == "zscore":
            return arr * p["std"] + p["mean"]
        return arr * (p["max"] - p["min"]) + p["min"]

    def inverse_transform_all(self, arr: np.ndarray, minerals: List[str]) -> np.ndarray:
        """
        Invert normalisation for all minerals.

        arr: shape (..., len(minerals))
        """
        out = arr.copy()
        for i, m in enumerate(minerals):
            out[..., i] = self.inverse_transform(arr[..., i], i, m)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_prices(path: str, minerals: List[str]) -> pd.DataFrame:
    """Load and minimally validate the price CSV."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for m in minerals:
        if m not in df.columns:
            raise ValueError(f"Column '{m}' not found in prices file")
    # Fill weekends / holidays with linear interpolation
    df[minerals] = df[minerals].interpolate(method="linear", limit_direction="both")
    return df


def load_news(path: str) -> pd.DataFrame:
    """Load news embeddings. Each cell is a comma-separated float string."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["date", "mineral"]).reset_index(drop=True)
    return df


def _parse_embed(cell: str) -> np.ndarray:
    """Convert a comma-separated float string to a numpy array."""
    return np.fromstring(cell, sep=",", dtype=np.float32)


def build_news_tensor(
    news_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    minerals: List[str],
    embed_dim: int,
) -> np.ndarray:
    """
    Build a news embedding array aligned to *dates*.

    Returns shape (len(dates), n_minerals, 3, embed_dim)
    where axis-2 is [short, medium, long].
    """
    n = len(dates)
    n_min = len(minerals)
    tensor = np.zeros((n, n_min, 3, embed_dim), dtype=np.float32)

    # Index news by (date, mineral) for fast lookup
    news_idx = news_df.set_index(["date", "mineral"])

    for i, d in enumerate(dates):
        for j, m in enumerate(minerals):
            try:
                row = news_idx.loc[(d, m)]
            except KeyError:
                continue
            tensor[i, j, 0] = _parse_embed(row["summary_short"])
            tensor[i, j, 1] = _parse_embed(row["summary_medium"])
            tensor[i, j, 2] = _parse_embed(row["summary_long"])

    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Rolling-forward split
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataSplit:
    """Indices (into the full sorted date array) for one train/val/test split."""
    train: Tuple[int, int]  # [start, end)
    val:   Tuple[int, int]
    test:  Tuple[int, int]


def make_rolling_splits(
    n_total: int,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    roll_step: int = 30,
    lookback: int = 180,
    max_horizon: int = 180,
) -> List[DataSplit]:
    """
    Generate a list of rolling-forward splits.

    The first split uses the first *train_ratio* of data for training, the
    next *val_ratio* for validation, and the next *test_ratio* for testing.
    Subsequent splits advance by *roll_step* days.

    Only the first split is returned unless there is enough data to advance.
    """
    splits: List[DataSplit] = []

    train_len = int(n_total * train_ratio)
    val_len   = int(n_total * val_ratio)
    test_len  = int(n_total * test_ratio)

    offset = 0
    while True:
        tr_start = offset
        tr_end   = offset + train_len
        va_start = tr_end
        va_end   = va_start + val_len
        te_start = va_end
        te_end   = te_start + test_len

        if te_end > n_total:
            break

        # Ensure the test window is large enough to form at least one sample
        if (te_end - te_start) < (lookback + max_horizon):
            break

        splits.append(DataSplit(
            train=(tr_start, tr_end),
            val=(va_start, va_end),
            test=(te_start, te_end),
        ))

        offset += roll_step
        # Stop if the initial split offset would exceed data bounds
        if tr_start + roll_step + train_len + val_len + test_len > n_total:
            break

    # Always return at least one split (the initial one)
    if not splits:
        tr_end  = train_len
        va_end  = tr_end + val_len
        te_end  = min(va_end + test_len, n_total)
        splits.append(DataSplit(
            train=(0, tr_end),
            val=(tr_end, va_end),
            test=(va_end, te_end),
        ))

    return splits
