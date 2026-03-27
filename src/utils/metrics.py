"""
Evaluation metrics for SentiCast
==================================
ICP  – Interval Coverage Probability
MIW  – Mean Interval Width
Pearson – Pearson correlation coefficient between predicted and actual values

All functions operate on numpy arrays and return per-horizon, per-mineral results.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def icp(
    y_true: np.ndarray,
    lower:  np.ndarray,
    upper:  np.ndarray,
) -> np.ndarray:
    """
    Interval Coverage Probability.

    Parameters
    ----------
    y_true : (...,) actual values
    lower  : (...,) lower bound of prediction interval
    upper  : (...,) upper bound of prediction interval

    Returns
    -------
    coverage : float
        Fraction of actual values within [lower, upper] (scalar between 0 and 1).
    """
    covered = (y_true >= lower) & (y_true <= upper)
    return covered.mean(axis=0)


def miw(
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """
    Mean Interval Width.

    Returns
    -------
    width : float
        Mean absolute width of the prediction interval (scalar).
    """
    return (upper - lower).mean(axis=0)


def pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Pearson correlation coefficient, computed column-wise.

    Parameters
    ----------
    y_true : (N, ...) actual values
    y_pred : (N, ...) predicted values

    Returns
    -------
    r : (...) correlation coefficients
    """
    # Flatten to 2D: (N, *features)
    orig_shape = y_true.shape[1:]
    y_true = y_true.reshape(len(y_true), -1)
    y_pred = y_pred.reshape(len(y_pred), -1)

    r_vals = np.array([
        _pearson_1d(y_true[:, c], y_pred[:, c])
        for c in range(y_true.shape[1])
    ])
    if not orig_shape or orig_shape == (1,):
        return float(r_vals[0])
    return r_vals.reshape(orig_shape)


def _pearson_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(y_true - y_pred).mean(axis=0)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return ((y_true - y_pred) ** 2).mean(axis=0)


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (np.abs((y_true - y_pred) / (np.abs(y_true) + eps))).mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metric report
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true:   np.ndarray,   # (N, H, M)  actual (denormalised)
    y_pred:   np.ndarray,   # (N, H, M)  point predictions (denormalised)
    y_lower:  np.ndarray,   # (N, H, M)  lower CI bound (denormalised)
    y_upper:  np.ndarray,   # (N, H, M)  upper CI bound (denormalised)
    horizons: List[int],
    minerals: List[str],
) -> Dict[str, Dict]:
    """
    Compute all metrics for each horizon × mineral combination.

    Returns a nested dict:
      results[horizon_str][mineral_str] = {metric: value}
    """
    results: Dict[str, Dict] = {}
    N, H, M = y_true.shape

    for h_idx, h in enumerate(horizons):
        h_key = f"{h}d"
        results[h_key] = {}
        for m_idx, m in enumerate(minerals):
            yt = y_true[:, h_idx, m_idx]   # (N,)
            yp = y_pred[:, h_idx, m_idx]
            yl = y_lower[:, h_idx, m_idx]
            yu = y_upper[:, h_idx, m_idx]

            results[h_key][m] = {
                "icp":     float(icp(yt, yl, yu)),
                "miw":     float(miw(yl, yu)),
                "pearson": float(pearson(yt.reshape(-1, 1), yp.reshape(-1, 1))),
                "mae":     float(mae(yt, yp)),
                "mse":     float(mse(yt, yp)),
                "mape":    float(mape(yt, yp)),
            }

    return results


def print_metrics(results: Dict[str, Dict], minerals: List[str]) -> None:
    """Pretty-print a metrics dict."""
    for h_key, h_val in results.items():
        print(f"\n── Horizon: {h_key} ──")
        header = f"{'Mineral':>8}  {'ICP':>8}  {'MIW':>10}  {'Pearson':>8}  {'MAE':>10}  {'MAPE':>8}"
        print(header)
        print("─" * len(header))
        for m in minerals:
            v = h_val.get(m, {})
            print(
                f"{m:>8}  "
                f"{v.get('icp', float('nan')):>8.4f}  "
                f"{v.get('miw', float('nan')):>10.4f}  "
                f"{v.get('pearson', float('nan')):>8.4f}  "
                f"{v.get('mae', float('nan')):>10.4f}  "
                f"{v.get('mape', float('nan')):>8.4f}"
            )
