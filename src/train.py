"""
Training pipeline for SentiCast.

Supports rolling-forward evaluation with train/val/test splits and
early stopping on validation loss.

Usage
-----
    python src/train.py --config config.yaml
    python src/train.py --config config.yaml --split_idx 0  # first rolling window
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.dataset import MineralDataset, build_datasets
from src.data.preprocessing import (
    Normalizer,
    build_news_tensor,
    build_news_tensor_real,
    load_news,
    load_news_real,
    load_prices,
    load_prices_real,
    make_rolling_splits,
)
from src.utils.metrics import compute_all_metrics, print_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(
    out: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    quantiles: List[float],
    horizon_weights: List[float],
    moe_aux_coeff: float,
    diff_weight: float = 1.0,
) -> torch.Tensor:
    """Combine diffusion loss, quantile regression loss, and MoE aux loss."""
    H = targets.shape[1]
    hw = torch.tensor(horizon_weights[:H], device=targets.device)

    # ── Diffusion loss ────────────────────────────────────────────────────────
    diff_loss = out.get("diffusion_loss", torch.tensor(0.0, device=targets.device))

    # ── Quantile regression (pinball) loss ────────────────────────────────────
    q_preds = out["quant_preds"]                    # (B, H, M, Q)
    targets_exp = targets.unsqueeze(-1)             # (B, H, M, 1)
    errors = targets_exp - q_preds                  # (B, H, M, Q)
    q_tensor = torch.tensor(quantiles, device=targets.device).reshape(1, 1, 1, -1)
    pinball = torch.where(errors >= 0, q_tensor * errors, (q_tensor - 1.0) * errors)
    # weight by horizon
    pinball_weighted = (pinball.mean(dim=[-1, -2]) * hw).mean()

    # ── MoE auxiliary loss ────────────────────────────────────────────────────
    moe_aux = out.get("moe_aux_loss", torch.tensor(0.0, device=targets.device))

    total = diff_weight * diff_loss + pinball_weighted + moe_aux_coeff * moe_aux
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Epoch runners
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: SentiCast,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    cfg: dict,
    training: bool = True,
) -> float:
    model.train(training)
    total_loss = 0.0
    t_cfg = cfg["training"]
    quantiles    = t_cfg["quantiles"]
    hw           = t_cfg["horizon_weights"]
    moe_coeff    = cfg["model"]["moe"].get("aux_loss_coeff", 0.01)
    grad_clip    = t_cfg.get("gradient_clip", 1.0)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            ps  = batch["price_series"].to(device)
            ne  = batch["news_embeds"].to(device)
            tgt = batch["targets"].to(device)
            nm  = batch.get("news_mask")
            if nm is not None:
                nm = nm.to(device)

            out = model(ps, ne, targets=tgt, news_mask=nm)
            loss = compute_loss(out, tgt, quantiles, hw, moe_coeff)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * len(ps)

    return total_loss / max(len(loader.dataset), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop for one split
# ─────────────────────────────────────────────────────────────────────────────

def train_one_split(
    model: SentiCast,
    train_ds: MineralDataset,
    val_ds: MineralDataset,
    cfg: dict,
    device: torch.device,
    ckpt_path: str,
) -> Dict[str, list]:
    t_cfg = cfg["training"]
    batch_size = t_cfg["batch_size"]
    epochs     = t_cfg["epochs"]
    patience   = t_cfg.get("early_stopping_patience", 15)
    warmup     = t_cfg.get("warmup_epochs", 5)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    optimizer = AdamW(model.parameters(), lr=t_cfg["learning_rate"], weight_decay=t_cfg["weight_decay"])

    if t_cfg.get("scheduler", "cosine") == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup, eta_min=1e-6)
    else:
        scheduler = ReduceLROnPlateau(optimizer, patience=patience // 2, factor=0.5)

    history: Dict[str, list] = {"train_loss": [], "val_loss": []}
    best_val   = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = run_epoch(model, train_loader, optimizer, device, cfg, training=True)
        vl_loss = run_epoch(model, val_loader,   None,      device, cfg, training=False)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)

        # Warmup: skip scheduler for first few epochs
        if epoch > warmup:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(vl_loss)
            else:
                scheduler.step()

        elapsed = time.time() - t0
        log.info(f"Epoch {epoch:3d}/{epochs}  train={tr_loss:.5f}  val={vl_loss:.5f}  [{elapsed:.1f}s]")

        if vl_loss < best_val:
            best_val   = vl_loss
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"Early stopping at epoch {epoch} (best val={best_val:.5f})")
                break

    # Restore best
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    log.info(f"Restored best model (val={best_val:.5f}) from {ckpt_path}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation on test split
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_on_test(
    model: SentiCast,
    test_ds: MineralDataset,
    cfg: dict,
    device: torch.device,
    normalizer: Normalizer,
    minerals: List[str],
) -> Dict:
    model.eval()
    batch_size = cfg["training"]["batch_size"]
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_true, all_mean, all_lower, all_upper, all_median = [], [], [], [], []
    all_ref_idx = []

    for batch in test_loader:
        ps  = batch["price_series"].to(device)
        ne  = batch["news_embeds"].to(device)
        tgt = batch["targets"].cpu().numpy()
        nm  = batch.get("news_mask")
        if nm is not None:
            nm = nm.to(device)

        pred = model.predict(ps, ne, n_samples=20, news_mask=nm)
        all_true.append(tgt)
        all_mean.append(pred["mean"].cpu().numpy())
        all_lower.append(pred["lower"].cpu().numpy())
        all_upper.append(pred["upper"].cpu().numpy())
        all_median.append(pred["median"].cpu().numpy())
        all_ref_idx.append(batch["ref_idx"].numpy())

    y_true   = np.concatenate(all_true,   axis=0)
    y_mean   = np.concatenate(all_mean,   axis=0)
    y_lower  = np.concatenate(all_lower,  axis=0)
    y_upper  = np.concatenate(all_upper,  axis=0)
    y_median = np.concatenate(all_median, axis=0)
    ref_idx  = np.concatenate(all_ref_idx, axis=0)

    # Denormalise
    y_true_dn   = normalizer.inverse_transform_all(y_true,   minerals)
    y_mean_dn   = normalizer.inverse_transform_all(y_mean,   minerals)
    y_lower_dn  = normalizer.inverse_transform_all(y_lower,  minerals)
    y_upper_dn  = normalizer.inverse_transform_all(y_upper,  minerals)
    y_median_dn = normalizer.inverse_transform_all(y_median, minerals)

    horizons = cfg["data"]["horizons"]
    metrics = compute_all_metrics(y_true_dn, y_mean_dn, y_lower_dn, y_upper_dn, horizons, minerals)
    print_metrics(metrics, minerals)

    return {
        "metrics":  metrics,
        "y_true":   y_true_dn,
        "y_mean":   y_mean_dn,
        "y_lower":  y_lower_dn,
        "y_upper":  y_upper_dn,
        "y_median": y_median_dn,
        "ref_idx":  ref_idx,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    minerals = cfg["data"]["minerals"]
    lookback = cfg["data"]["lookback"]
    horizons = cfg["data"]["horizons"]
    embed_dim = cfg["data"]["news_embed_dim"]

    # ── Load data ──────────────────────────────────────────────────────────────
    log.info("Loading data …")
    data_dir = cfg["data"].get("data_dir")
    news_mask = None
    if data_dir:
        log.info(f"Using real dataset from '{data_dir}'")
        prices_df = load_prices_real(data_dir, minerals)
        news_df   = load_news_real(data_dir, minerals)
    else:
        prices_df = load_prices(cfg["data"]["prices_path"], minerals)
        news_df   = load_news(cfg["data"]["news_path"])

    dates = prices_df["date"]
    n_total = len(prices_df)

    # ── Build news tensor (done once; cached across splits) ───────────────────
    log.info("Building news embeddings …")
    if data_dir:
        encoder_name = cfg["data"].get("news_encoder", "auto")
        cache_path = cfg["data"].get("news_cache_path", "data/cache/news_tensor.npy")
        news_tensor, news_mask = build_news_tensor_real(
            news_df, dates, minerals, embed_dim, cache_path, encoder_name
        )
        # Sync embed_dim with what the encoder actually produced
        embed_dim = news_tensor.shape[-1]
        cfg["data"]["news_embed_dim"] = embed_dim
    else:
        news_tensor = build_news_tensor(news_df, dates, minerals, embed_dim)

    # ── Rolling splits ─────────────────────────────────────────────────────────
    splits = make_rolling_splits(
        n_total=n_total,
        train_ratio=cfg["split"]["train_ratio"],
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
        roll_step=cfg["split"]["roll_step"],
        lookback=lookback,
        max_horizon=max(horizons),
    )
    log.info(f"Total rolling splits: {len(splits)}")

    split_range = [args.split_idx] if args.split_idx >= 0 else range(len(splits))

    all_results = {}
    for si in split_range:
        split = splits[si]
        log.info(f"\n{'='*60}")
        log.info(f"Rolling split {si}  train={split.train}  val={split.val}  test={split.test}")

        # ── Normalise using training statistics only ───────────────────────────
        train_df = prices_df.iloc[split.train[0] : split.train[1]]
        normalizer = Normalizer(method=cfg["data"]["normalization"])
        normalizer.fit(train_df, minerals)
        prices_norm = normalizer.transform(prices_df, minerals)[minerals].values.astype(np.float32)

        # ── Datasets ──────────────────────────────────────────────────────────
        train_ds, val_ds, test_ds = build_datasets(
            prices_norm, news_tensor, dates, split, horizons, lookback, news_mask
        )
        log.info(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

        if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
            log.warning("Skipping split with empty dataset.")
            continue

        # ── Model ─────────────────────────────────────────────────────────────
        model = build_model(cfg, n_minerals=len(minerals)).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"  Model parameters: {n_params:,}")

        # ── Train ─────────────────────────────────────────────────────────────
        os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)
        ckpt_path = os.path.join(cfg["output"]["checkpoint_dir"], f"best_split{si}.pt")
        history = train_one_split(model, train_ds, val_ds, cfg, device, ckpt_path)

        # ── Evaluate ──────────────────────────────────────────────────────────
        log.info("Evaluating on test split …")
        eval_result = evaluate_on_test(model, test_ds, cfg, device, normalizer, minerals)
        eval_result["history"] = history

        all_results[f"split_{si}"] = eval_result["metrics"]

        # ── Save eval arrays for visualisation ────────────────────────────────
        os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
        np.savez(
            os.path.join(cfg["output"]["results_dir"], f"eval_split{si}.npz"),
            y_true=eval_result["y_true"],
            y_mean=eval_result["y_mean"],
            y_lower=eval_result["y_lower"],
            y_upper=eval_result["y_upper"],
            y_median=eval_result["y_median"],
            ref_idx=eval_result["ref_idx"],
        )

    # ── Save summary metrics ───────────────────────────────────────────────────
    metrics_path = os.path.join(cfg["output"]["results_dir"], "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nMetrics summary saved → {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SentiCast")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--split_idx", type=int, default=0,
                        help="Index of rolling split to train; -1 = all splits")
    args = parser.parse_args()
    main(args)
