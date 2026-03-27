"""
Visualization for SentiCast
============================
Generates plots showing:
  1. Actual vs. predicted prices for each mineral × horizon combination
     over the **test period only** (no training data shown).
  2. 90% confidence intervals (lower / upper bounds).

Usage
-----
    python src/visualize.py --config config.yaml --eval_file results/eval_split0.npz
    python src/visualize.py --config config.yaml --eval_file results/eval_split0.npz --show
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

matplotlib.use("Agg")   # headless by default; overridden if --show is passed


MINERAL_COLORS = {
    "gold":   "#FFB300",
    "silver": "#90A4AE",
    "copper": "#BF360C",
}

HORIZON_LABELS = {
    30:  "30일",
    90:  "90일",
    180: "180일",
}


def _setup_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=8, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_predictions(
    y_true:   np.ndarray,    # (N, H, M)  actual (denormalised)
    y_mean:   np.ndarray,    # (N, H, M)
    y_lower:  np.ndarray,    # (N, H, M)
    y_upper:  np.ndarray,    # (N, H, M)
    minerals: List[str],
    horizons: List[int],
    out_path: str,
    ref_idx: Optional[np.ndarray] = None,
    show: bool = False,
) -> None:
    """
    One figure per horizon; each figure has M subplots (one per mineral).
    Shows actual vs. predicted with 90% CI shading.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    N = len(y_true)
    x_axis = ref_idx if ref_idx is not None else np.arange(N)

    for h_idx, h in enumerate(horizons):
        fig, axes = plt.subplots(1, len(minerals), figsize=(6 * len(minerals), 4))
        if len(minerals) == 1:
            axes = [axes]

        for m_idx, m in enumerate(minerals):
            ax = axes[m_idx]
            color = MINERAL_COLORS.get(m, "steelblue")

            yt = y_true[:, h_idx, m_idx]
            yp = y_mean[:, h_idx, m_idx]
            yl = y_lower[:, h_idx, m_idx]
            yu = y_upper[:, h_idx, m_idx]

            ax.plot(x_axis, yt, color="black",  linewidth=1.2, label="실제값 (Actual)", zorder=3)
            ax.plot(x_axis, yp, color=color,    linewidth=1.2, label="예측값 (Predicted)", linestyle="--", zorder=3)
            ax.fill_between(x_axis, yl, yu, color=color, alpha=0.25, label="90% CI")

            unit = "USD/t oz" if m in ("gold", "silver") else "USD/MT"
            _setup_axes(
                ax,
                title=f"{m.capitalize()} – {HORIZON_LABELS.get(h, f'{h}d')} 예측",
                xlabel="기준일 (Reference date index)",
                ylabel=f"가격 ({unit})",
            )

        fig.suptitle(
            f"SentiCast 예측 결과 – 시험 구간 (Test Period)  |  예측 지평: {HORIZON_LABELS.get(h, f'{h}d')}",
            fontsize=13,
            y=1.02,
        )
        fig.tight_layout()

        # Save per-horizon figure
        base, ext = os.path.splitext(out_path)
        h_path = f"{base}_horizon{h}d{ext}"
        fig.savefig(h_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {h_path}")

        if show:
            plt.show()
        plt.close(fig)


def plot_all_horizons_single_mineral(
    y_true:   np.ndarray,
    y_mean:   np.ndarray,
    y_lower:  np.ndarray,
    y_upper:  np.ndarray,
    mineral:  str,
    m_idx:    int,
    horizons: List[int],
    out_path: str,
    ref_idx: Optional[np.ndarray] = None,
    show: bool = False,
) -> None:
    """
    One figure per mineral showing all three horizons as subplots.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    N = len(y_true)
    x_axis = ref_idx if ref_idx is not None else np.arange(N)
    color = MINERAL_COLORS.get(mineral, "steelblue")

    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 4), sharey=False)
    if len(horizons) == 1:
        axes = [axes]

    for h_idx, h in enumerate(horizons):
        ax = axes[h_idx]
        yt = y_true[:, h_idx, m_idx]
        yp = y_mean[:, h_idx, m_idx]
        yl = y_lower[:, h_idx, m_idx]
        yu = y_upper[:, h_idx, m_idx]

        ax.plot(x_axis, yt, color="black", linewidth=1.2, label="실제값", zorder=3)
        ax.plot(x_axis, yp, color=color,   linewidth=1.2, label="예측값", linestyle="--", zorder=3)
        ax.fill_between(x_axis, yl, yu, color=color, alpha=0.25, label="90% CI")

        unit = "USD/t oz" if mineral in ("gold", "silver") else "USD/MT"
        _setup_axes(
            ax,
            title=f"{HORIZON_LABELS.get(h, f'{h}d')} 예측",
            xlabel="기준일 인덱스",
            ylabel=f"가격 ({unit})",
        )

    fig.suptitle(
        f"{mineral.capitalize()} – 전체 예측 지평 비교 (시험 구간)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    base, ext = os.path.splitext(out_path)
    m_path = f"{base}_{mineral}{ext}"
    fig.savefig(m_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {m_path}")

    if show:
        plt.show()
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    minerals = cfg["data"]["minerals"]
    horizons = cfg["data"]["horizons"]

    data = np.load(args.eval_file)
    y_true  = data["y_true"]
    y_mean  = data["y_mean"]
    y_lower = data["y_lower"]
    y_upper = data["y_upper"]
    ref_idx = data.get("ref_idx", None)

    out_dir = cfg["output"]["figures_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print("Generating per-horizon plots …")
    plot_predictions(
        y_true, y_mean, y_lower, y_upper,
        minerals, horizons,
        out_path=os.path.join(out_dir, "prediction.png"),
        ref_idx=ref_idx,
        show=args.show,
    )

    print("Generating per-mineral plots …")
    for m_idx, m in enumerate(minerals):
        plot_all_horizons_single_mineral(
            y_true, y_mean, y_lower, y_upper,
            mineral=m,
            m_idx=m_idx,
            horizons=horizons,
            out_path=os.path.join(out_dir, "mineral.png"),
            ref_idx=ref_idx,
            show=args.show,
        )

    print("Done.")


if __name__ == "__main__":
    if not matplotlib.rcParams.get("backend"):
        matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description="Visualise SentiCast predictions")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--eval_file", default="results/eval_split0.npz")
    parser.add_argument("--show",      action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    if args.show:
        matplotlib.use("TkAgg")

    main(args)
