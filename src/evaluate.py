"""
Standalone evaluation script for SentiCast.

Loads a trained checkpoint and evaluates it on the test split,
printing per-horizon / per-mineral metrics.

Usage
-----
    python src/evaluate.py --config config.yaml --checkpoint checkpoints/best_split0.pt
    python src/evaluate.py --config config.yaml --checkpoint checkpoints/best_split0.pt --split_idx 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch
import yaml

from src.data.preprocessing import (
    Normalizer,
    build_news_tensor,
    load_news,
    load_prices,
    make_rolling_splits,
)
from src.data.dataset import build_datasets
from src.models.senticast import build_model
from src.utils.metrics import compute_all_metrics, print_metrics
from src.train import evaluate_on_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    minerals = cfg["data"]["minerals"]
    lookback  = cfg["data"]["lookback"]
    horizons  = cfg["data"]["horizons"]
    embed_dim = cfg["data"]["news_embed_dim"]

    # ── Load data ──────────────────────────────────────────────────────────────
    prices_df = load_prices(cfg["data"]["prices_path"], minerals)
    news_df   = load_news(cfg["data"]["news_path"])
    dates     = prices_df["date"]
    n_total   = len(prices_df)

    # ── Split ─────────────────────────────────────────────────────────────────
    splits = make_rolling_splits(
        n_total=n_total,
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
        roll_step=cfg["split"]["roll_step"],
        lookback=lookback,
        max_horizon=max(horizons),
    )
    split = splits[args.split_idx]
    log.info(f"Using split {args.split_idx}: test={split.test}")

    # ── Normaliser (fitted on train) ──────────────────────────────────────────
    train_df = prices_df.iloc[split.train[0] : split.train[1]]
    normalizer = Normalizer(method=cfg["data"]["normalization"])
    normalizer.fit(train_df, minerals)
    prices_norm = normalizer.transform(prices_df, minerals)[minerals].values.astype(np.float32)

    # ── News tensor ───────────────────────────────────────────────────────────
    news_tensor = build_news_tensor(news_df, dates, minerals, embed_dim)

    # ── Datasets ──────────────────────────────────────────────────────────────
    _, _, test_ds = build_datasets(prices_norm, news_tensor, dates, split, horizons, lookback)
    log.info(f"Test samples: {len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg, n_minerals=len(minerals)).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    log.info(f"Loaded checkpoint: {args.checkpoint}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    result = evaluate_on_test(model, test_ds, cfg, device, normalizer, minerals)

    # ── Save metrics ──────────────────────────────────────────────────────────
    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    out_path = os.path.join(
        cfg["output"]["results_dir"],
        f"metrics_eval_split{args.split_idx}.json",
    )
    with open(out_path, "w") as f:
        json.dump(result["metrics"], f, indent=2)
    log.info(f"Metrics saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SentiCast checkpoint")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split_idx",  type=int, default=0)
    args = parser.parse_args()
    main(args)
