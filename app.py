"""
SentiCast Web Interface
=======================
Gradio-based web UI for training and evaluating the SentiCast mineral-price
forecasting model.

Launch
------
    # with uv (recommended)
    uv run python app.py

    # with plain Python (after installing dependencies)
    python app.py
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Generator, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
import gradio as gr

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.yaml"
CHECKPOINTS_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Training process manager
# ─────────────────────────────────────────────────────────────────────────────

class _TrainingProcess:
    """Thread-safe wrapper around a single background training subprocess."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def start(self, cmd: List[str]) -> subprocess.Popen:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("A training job is already running.")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd=str(ROOT),
            )
            return self._proc

    def stop(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()


_training_proc = _TrainingProcess()


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def _save_temp_config(cfg: dict) -> str:
    """Write *cfg* to a temporary YAML file and return its path."""
    tmp_dir = ROOT / "data"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="run_cfg_", dir=str(tmp_dir))
    with os.fdopen(fd, "w") as fh:
        yaml.dump(cfg, fh)
    return path


def _build_config(
    news_encoder: str,
    lookback: int,
    horizons_str: str,
    normalization: str,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    num_diffusion_steps: int,
    inference_steps: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    scheduler: str,
    warmup_epochs: int,
    patience: int,
    gradient_clip: float,
    seed: int,
) -> dict:
    """Merge UI values into the base config dict."""
    cfg = _load_config()

    # Parse horizon list
    try:
        horizons = [int(h.strip()) for h in horizons_str.split(",") if h.strip()]
    except ValueError:
        horizons = [30, 90, 180]

    # Embedding dim is set automatically at runtime; we expose it as a hint
    encoder_dims = {"finbert": 768, "sentence-transformers": 384}
    embed_dim = encoder_dims.get(news_encoder, cfg["data"].get("news_embed_dim", 384))

    cfg["data"]["news_encoder"]   = news_encoder
    cfg["data"]["lookback"]       = lookback
    cfg["data"]["horizons"]       = horizons
    cfg["data"]["news_embed_dim"] = embed_dim
    cfg["data"]["normalization"]  = normalization

    cfg["model"]["d_model"]  = d_model
    cfg["model"]["n_heads"]  = n_heads
    cfg["model"]["n_layers"] = n_layers
    cfg["model"]["dropout"]  = round(dropout, 3)

    cfg["model"]["news"]["embed_dim"] = embed_dim
    cfg["model"]["news"]["proj_dim"]  = d_model

    cfg["model"]["diffusion"]["num_steps"]       = num_diffusion_steps
    cfg["model"]["diffusion"]["inference_steps"] = inference_steps

    cfg["training"]["epochs"]                   = epochs
    cfg["training"]["batch_size"]               = batch_size
    cfg["training"]["learning_rate"]            = learning_rate
    cfg["training"]["weight_decay"]             = weight_decay
    cfg["training"]["scheduler"]                = scheduler
    cfg["training"]["warmup_epochs"]            = warmup_epochs
    cfg["training"]["early_stopping_patience"]  = patience
    cfg["training"]["gradient_clip"]            = gradient_clip
    cfg["training"]["seed"]                     = seed

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_loss_chart(
    train_losses: List[float],
    val_losses: List[float],
) -> Optional[plt.Figure]:
    if not train_losses:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ep = list(range(1, len(train_losses) + 1))
    ax.plot(ep, train_losses, "b-", lw=1.5, label="Train Loss")
    if val_losses:
        ax.plot(ep[: len(val_losses)], val_losses, "r-", lw=1.5, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    ax.legend()
    ax.grid(alpha=0.3, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _make_prediction_chart(
    npz_path: str,
    minerals: List[str],
    horizons: List[int],
) -> Optional[plt.Figure]:
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    y_true  = data["y_true"]
    y_mean  = data["y_mean"]
    y_lower = data["y_lower"]
    y_upper = data["y_upper"]

    COLORS = {"gold": "#FFB300", "silver": "#90A4AE", "copper": "#BF360C"}
    H = len(horizons)
    M = len(minerals)
    N = len(y_true)

    fig, axes = plt.subplots(
        H, M,
        figsize=(5 * M, 3.5 * H),
        squeeze=False,
    )
    for h_idx, h in enumerate(horizons):
        for m_idx, m in enumerate(minerals):
            ax = axes[h_idx][m_idx]
            color = COLORS.get(m, "steelblue")
            ax.plot(y_true[:, h_idx, m_idx], "k-", lw=1.0, label="Actual")
            ax.plot(y_mean[:, h_idx, m_idx], "--", color=color, lw=1.0, label="Predicted")
            ax.fill_between(
                range(N),
                y_lower[:, h_idx, m_idx],
                y_upper[:, h_idx, m_idx],
                color=color, alpha=0.2, label="90% CI",
            )
            ax.set_title(f"{m.capitalize()} – {h}d", fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)

    fig.suptitle("SentiCast – Test-Period Predictions", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Training generator
# ─────────────────────────────────────────────────────────────────────────────

def _training_generator(
    news_encoder: str,
    lookback: int,
    horizons_str: str,
    normalization: str,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    num_diffusion_steps: int,
    inference_steps: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    scheduler: str,
    warmup_epochs: int,
    patience: int,
    gradient_clip: float,
    seed: int,
    split_idx: int,
) -> Generator:
    """
    Gradio streaming generator: runs training as a subprocess and yields
    ``(log_text, loss_figure, status_text)`` tuples after every log line.
    """
    split_idx = int(split_idx)

    if _training_proc.is_running:
        yield "⚠️ A training job is already running!", None, "⚠️ Already running"
        return

    cfg = _build_config(
        news_encoder, lookback, horizons_str, normalization,
        d_model, n_heads, n_layers, dropout,
        num_diffusion_steps, inference_steps,
        epochs, batch_size, learning_rate, weight_decay,
        scheduler, warmup_epochs, patience, gradient_clip, seed,
    )
    config_path = _save_temp_config(cfg)

    cmd = [
        sys.executable, "-m", "src.train",
        "--config", config_path,
        "--split_idx", str(split_idx),
    ]

    yield (
        f"🚀 Starting training…\n"
        f"   Encoder : {news_encoder}\n"
        f"   d_model : {d_model}  |  layers : {n_layers}  |  heads : {n_heads}\n"
        f"   Epochs  : {epochs}  |  batch : {batch_size}  |  lr : {learning_rate}\n"
        f"   Split   : {split_idx}\n\n",
        None,
        "⏳ Starting…",
    )

    try:
        proc = _training_proc.start(cmd)
    except RuntimeError as exc:
        os.unlink(config_path)
        yield str(exc), None, "❌ Error"
        return

    logs = ""
    train_losses: List[float] = []
    val_losses:   List[float] = []
    status = "⏳ Running…"

    for line in iter(proc.stdout.readline, ""):
        logs += line

        # Parse epoch loss line: "Epoch  5/100  train=X.XXXXX  val=Y.YYYYY  [Zs]"
        m = re.search(r"train=(\d+\.\d+)\s+val=(\d+\.\d+)", line)
        if m:
            train_losses.append(float(m.group(1)))
            val_losses.append(float(m.group(2)))
            ep = len(train_losses)
            status = f"⏳ Epoch {ep}/{epochs}  train={m.group(1)}  val={m.group(2)}"

        # Emit at most 6 000 chars to avoid overloading the browser
        display = logs[-6000:] if len(logs) > 6000 else logs
        yield display, _make_loss_chart(train_losses, val_losses), status

    proc.wait()
    try:
        os.unlink(config_path)
    except FileNotFoundError:
        pass

    rc = proc.returncode
    final_status = "✅ Training complete!" if rc == 0 else f"❌ Process exited with code {rc}"
    display = logs[-6000:] if len(logs) > 6000 else logs
    yield display + f"\n\n{final_status}", _make_loss_chart(train_losses, val_losses), final_status


def _stop_training() -> str:
    _training_proc.stop()
    return "⏹ Stop requested – waiting for current epoch to finish…"


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation generator
# ─────────────────────────────────────────────────────────────────────────────

def _evaluation_generator(
    checkpoint_path: str,
    split_idx: int,
) -> Generator:
    """Run ``src.evaluate`` as a subprocess and stream its output."""
    ckpt = checkpoint_path.strip()
    split_idx = int(split_idx)
    if not ckpt:
        ckpt = str(CHECKPOINTS_DIR / f"best_split{split_idx}.pt")

    if not os.path.exists(ckpt):
        yield f"❌ Checkpoint not found: {ckpt}", None, None, "❌ Not found"
        return

    cmd = [
        sys.executable, "-m", "src.evaluate",
        "--config", str(CONFIG_PATH),
        "--checkpoint", ckpt,
        "--split_idx", str(split_idx),
    ]

    yield "🔍 Running evaluation…\n", None, None, "⏳ Running…"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(ROOT),
    )

    logs = ""
    for line in iter(proc.stdout.readline, ""):
        logs += line
        display = logs[-4000:] if len(logs) > 4000 else logs
        yield display, None, None, "⏳ Running…"

    proc.wait()

    cfg = _load_config()
    minerals = cfg["data"]["minerals"]
    horizons = cfg["data"]["horizons"]

    # Load metrics JSON if it exists
    metrics_path = (
        ROOT / cfg["output"]["results_dir"] / f"metrics_eval_split{split_idx}.json"
    )
    metrics_table = None
    if metrics_path.exists():
        with open(metrics_path) as fh:
            raw = json.load(fh)
        rows = []
        for h_key, h_val in raw.items():
            for mineral, vals in h_val.items():
                rows.append({
                    "Horizon": h_key,
                    "Mineral": mineral,
                    "ICP":     round(vals.get("icp",     float("nan")), 4),
                    "MIW":     round(vals.get("miw",     float("nan")), 2),
                    "Pearson": round(vals.get("pearson", float("nan")), 4),
                    "MAE":     round(vals.get("mae",     float("nan")), 2),
                    "MAPE":    round(vals.get("mape",    float("nan")), 4),
                })
        import pandas as pd
        metrics_table = pd.DataFrame(rows)

    # Load prediction chart if eval npz exists
    npz_path = str(ROOT / cfg["output"]["results_dir"] / f"eval_split{split_idx}.npz")
    pred_chart = _make_prediction_chart(npz_path, minerals, horizons)

    rc = proc.returncode
    status = "✅ Evaluation complete!" if rc == 0 else f"❌ Exited with code {rc}"
    display = logs[-4000:] if len(logs) > 4000 else logs
    yield display + f"\n\n{status}", metrics_table, pred_chart, status


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def _list_checkpoints() -> List[str]:
    """Return available checkpoint files."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(str(p) for p in CHECKPOINTS_DIR.glob("*.pt"))


def build_ui() -> gr.Blocks:
    cfg = _load_config()

    # ── Default values pulled from config ─────────────────────────────────────
    D = cfg["data"]
    M = cfg["model"]
    T = cfg["training"]

    with gr.Blocks(
        title="SentiCast",
    ) as demo:

        gr.Markdown(
            """# 📈 SentiCast — Mineral Price Forecasting
            Multi-horizon price prediction for **Gold**, **Silver**, and **Copper**
            using a diffusion backbone fused with financial news sentiment.
            """
        )

        # ═══════════════════════════════════════════════════════════════════
        # TAB 1 – TRAINING
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("🚀 Training"):
            with gr.Row(equal_height=False):

                # ── Left column: configuration ─────────────────────────────
                with gr.Column(scale=4, min_width=380):
                    gr.Markdown("### ⚙️ Hyperparameters")

                    with gr.Accordion("📰 Data & News Encoder", open=True):
                        news_encoder = gr.Dropdown(
                            choices=["sentence-transformers", "finbert", "tfidf-svd", "auto"],
                            value=D.get("news_encoder", "sentence-transformers"),
                            label="News Encoder",
                            info=(
                                "sentence-transformers: fast, 384-dim  |  "
                                "finbert: finance-tuned, 768-dim (needs ~500MB download)  |  "
                                "tfidf-svd: CPU fallback"
                            ),
                        )
                        lookback = gr.Slider(
                            60, 365, value=D.get("lookback", 180), step=10,
                            label="Lookback window (days)",
                            info="Number of past price days fed as input",
                        )
                        horizons_str = gr.Textbox(
                            value=", ".join(str(h) for h in D.get("horizons", [30, 90, 180])),
                            label="Prediction horizons (comma-separated days)",
                            info="e.g.  30, 90, 180",
                        )
                        normalization = gr.Radio(
                            choices=["zscore", "minmax"],
                            value=D.get("normalization", "zscore"),
                            label="Price normalisation",
                        )

                    with gr.Accordion("🧠 Model Architecture", open=True):
                        with gr.Row():
                            d_model = gr.Slider(
                                32, 256, value=M.get("d_model", 128), step=32,
                                label="d_model",
                                info="Core transformer dimension",
                            )
                            n_heads = gr.Dropdown(
                                choices=[1, 2, 4, 8],
                                value=M.get("n_heads", 4),
                                label="Attention heads",
                                info="Must divide d_model",
                            )
                        with gr.Row():
                            n_layers = gr.Slider(
                                1, 8, value=M.get("n_layers", 3), step=1,
                                label="Transformer layers",
                            )
                            dropout = gr.Slider(
                                0.0, 0.5, value=M.get("dropout", 0.1), step=0.05,
                                label="Dropout",
                            )
                        with gr.Row():
                            num_diffusion_steps = gr.Slider(
                                10, 500, value=M["diffusion"].get("num_steps", 100), step=10,
                                label="Diffusion steps (training)",
                            )
                            inference_steps = gr.Slider(
                                5, 100, value=M["diffusion"].get("inference_steps", 20), step=5,
                                label="DDIM inference steps",
                                info="Fewer = faster inference",
                            )

                    with gr.Accordion("🏋️ Training", open=True):
                        with gr.Row():
                            epochs = gr.Slider(
                                10, 300, value=T.get("epochs", 100), step=10,
                                label="Epochs",
                            )
                            batch_size = gr.Dropdown(
                                choices=[8, 16, 32, 64, 128],
                                value=T.get("batch_size", 32),
                                label="Batch size",
                            )
                        with gr.Row():
                            learning_rate = gr.Number(
                                value=T.get("learning_rate", 3e-4),
                                label="Learning rate",
                                precision=6,
                            )
                            weight_decay = gr.Number(
                                value=T.get("weight_decay", 1e-4),
                                label="Weight decay",
                                precision=6,
                            )
                        with gr.Row():
                            scheduler = gr.Radio(
                                choices=["cosine", "plateau"],
                                value=T.get("scheduler", "cosine"),
                                label="LR Scheduler",
                            )
                            warmup_epochs = gr.Slider(
                                0, 20, value=T.get("warmup_epochs", 5), step=1,
                                label="Warmup epochs",
                            )
                        with gr.Row():
                            patience = gr.Slider(
                                5, 50, value=T.get("early_stopping_patience", 15), step=1,
                                label="Early-stop patience (epochs)",
                            )
                            gradient_clip = gr.Slider(
                                0.1, 5.0, value=T.get("gradient_clip", 1.0), step=0.1,
                                label="Gradient clip norm",
                            )
                        seed = gr.Number(
                            value=T.get("seed", 42), label="Random seed", precision=0,
                        )

                    with gr.Row():
                        split_idx_train = gr.Slider(
                            -1, 5, value=0, step=1,
                            label="Split index  (−1 = all splits)",
                            info="Rolling-forward window to train",
                        )

                    with gr.Row():
                        start_btn = gr.Button("▶  Start Training", variant="primary", size="lg")
                        stop_btn  = gr.Button("⏹  Stop",            variant="stop",    size="lg")

                # ── Right column: live outputs ──────────────────────────────
                with gr.Column(scale=5, min_width=420):
                    train_status = gr.Markdown("*Ready — configure hyperparameters and click **Start Training***")
                    log_box = gr.Textbox(
                        label="Training Log",
                        lines=22,
                        max_lines=22,
                    )
                    loss_chart = gr.Plot(label="Loss Curve")

            # ── Wire up events ──────────────────────────────────────────────
            train_inputs = [
                news_encoder, lookback, horizons_str, normalization,
                d_model, n_heads, n_layers, dropout,
                num_diffusion_steps, inference_steps,
                epochs, batch_size, learning_rate, weight_decay,
                scheduler, warmup_epochs, patience, gradient_clip, seed,
                split_idx_train,
            ]
            train_outputs = [log_box, loss_chart, train_status]

            start_event = start_btn.click(
                fn=_training_generator,
                inputs=train_inputs,
                outputs=train_outputs,
            )
            stop_btn.click(
                fn=_stop_training,
                outputs=[train_status],
                cancels=[start_event],
            )

        # ═══════════════════════════════════════════════════════════════════
        # TAB 2 – EVALUATION
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("🔍 Evaluation"):
            with gr.Row(equal_height=False):

                with gr.Column(scale=3, min_width=300):
                    gr.Markdown("### ⚙️ Evaluation Settings")

                    ckpt_choices = _list_checkpoints()
                    checkpoint_path = gr.Dropdown(
                        choices=ckpt_choices,
                        value=ckpt_choices[0] if ckpt_choices else "",
                        label="Checkpoint file",
                        info="Select a saved .pt checkpoint from the checkpoints/ folder",
                        allow_custom_value=True,
                    )
                    refresh_btn = gr.Button("🔄 Refresh checkpoint list", size="sm")

                    split_idx_eval = gr.Slider(
                        0, 5, value=0, step=1,
                        label="Split index",
                        info="Must match the split the checkpoint was trained on",
                    )

                    eval_btn = gr.Button("▶  Run Evaluation", variant="primary", size="lg")

                with gr.Column(scale=6, min_width=480):
                    eval_status  = gr.Markdown("*Select a checkpoint and click **Run Evaluation***")
                    eval_log     = gr.Textbox(label="Evaluation Log", lines=10, max_lines=12)
                    metrics_df   = gr.Dataframe(label="Metrics (test split)", interactive=False)
                    pred_chart   = gr.Plot(label="Predictions vs Actual (test period)")

            # Wire up
            refresh_btn.click(
                fn=lambda: gr.update(choices=_list_checkpoints()),
                outputs=[checkpoint_path],
            )
            eval_event = eval_btn.click(
                fn=_evaluation_generator,
                inputs=[checkpoint_path, split_idx_eval],
                outputs=[eval_log, metrics_df, pred_chart, eval_status],
            )

        # ═══════════════════════════════════════════════════════════════════
        # TAB 3 – CONFIGURATION
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("📄 Config"):
            gr.Markdown(
                "Current `config.yaml` is shown below. "
                "Edit the file directly on disk to make permanent changes; "
                "those changes take effect on the next training run."
            )
            with open(CONFIG_PATH) as fh:
                cfg_text_initial = fh.read()

            cfg_display = gr.Code(
                value=cfg_text_initial,
                language="yaml",
                label="config.yaml",
                interactive=False,
            )

            def _reload_config_text() -> str:
                with open(CONFIG_PATH) as fh:
                    return fh.read()

            gr.Button("🔄 Reload", size="sm").click(
                fn=_reload_config_text, outputs=[cfg_display]
            )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    demo = build_ui()
    demo.queue()          # enable queue for streaming / cancel support
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
