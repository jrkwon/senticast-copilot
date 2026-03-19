"""
Sample data generator for SentiCast.

Generates synthetic but realistic mineral price time series and news embeddings
so the pipeline can be exercised without access to real proprietary data.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Price simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gbm_with_seasonality(
    n: int,
    s0: float,
    mu: float,
    sigma: float,
    period: float,
    amp: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Geometric Brownian Motion with additive sinusoidal seasonality."""
    dt = 1.0
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    prices = s0 * np.exp(np.cumsum(log_returns))
    t = np.arange(n)
    seasonal = amp * np.sin(2 * np.pi * t / period)
    return prices + seasonal


def generate_prices(n_days: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame of daily prices for gold, silver, and copper."""
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range("2019-01-02", periods=n_days)

    gold = _gbm_with_seasonality(n_days, s0=1_300, mu=5e-4, sigma=7e-3, period=252, amp=50, rng=rng)
    gold = np.clip(gold, 1_000, 3_500)

    # Silver is correlated with gold (귀금속 공통 특성)
    silver_noise = rng.standard_normal(n_days)
    gold_log_ret = np.diff(np.log(gold), prepend=np.log(gold[0]))
    silver_log_ret = 0.7 * gold_log_ret + 0.3 * (1.5e-4 + 0.015 * silver_noise)
    silver = 18.0 * np.exp(np.cumsum(silver_log_ret))
    seasonal_silver = 1.5 * np.sin(2 * np.pi * np.arange(n_days) / 180)
    silver = np.clip(silver + seasonal_silver, 10, 60)

    # Copper is an industrial metal (경기 선행지표)
    copper = _gbm_with_seasonality(n_days, s0=6_500, mu=2e-4, sigma=9e-3, period=365, amp=200, rng=rng)
    copper = np.clip(copper, 4_000, 12_000)

    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "gold": gold, "silver": silver, "copper": copper})


# ─────────────────────────────────────────────────────────────────────────────
# News embedding simulation
# ─────────────────────────────────────────────────────────────────────────────

def generate_news(dates: pd.Series, embed_dim: int = 384, seed: int = 42) -> pd.DataFrame:
    """
    Return a DataFrame with random-but-structured news embeddings.

    One row per (date, mineral) combination. Embeddings mimic the direction
    signal of the corresponding price movement so models can learn the linkage.
    """
    rng = np.random.default_rng(seed + 1)
    minerals = ["gold", "silver", "copper", "global"]
    summary_types = ["short", "medium", "long"]

    rows: list[dict] = []
    for date in dates:
        for mineral in minerals:
            row: dict = {"date": date, "mineral": mineral}
            for stype in summary_types:
                embed = rng.standard_normal(embed_dim).astype(np.float32)
                # Store as a comma-separated string for CSV portability
                row[f"summary_{stype}"] = ",".join(f"{v:.6f}" for v in embed)
            rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(out_dir: str = "data/sample", n_days: int = 2000, embed_dim: int = 384) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating {n_days} days of price data …")
    prices = generate_prices(n_days)
    prices_path = os.path.join(out_dir, "prices.csv")
    prices.to_csv(prices_path, index=False)
    print(f"  Saved → {prices_path}")

    print(f"Generating news embeddings (embed_dim={embed_dim}) …")
    news = generate_news(prices["date"], embed_dim=embed_dim)
    news_path = os.path.join(out_dir, "news.csv")
    news.to_csv(news_path, index=False)
    print(f"  Saved → {news_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample data for SentiCast")
    parser.add_argument("--out_dir", default="data/sample")
    parser.add_argument("--n_days", type=int, default=2000)
    parser.add_argument("--embed_dim", type=int, default=384)
    args = parser.parse_args()
    main(args.out_dir, args.n_days, args.embed_dim)
