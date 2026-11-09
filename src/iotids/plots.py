"""Report figures (saved as PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


def roc_curves(y: np.ndarray, scores: dict[str, np.ndarray], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=1.8, label=name)
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set(xscale="log", xlim=(1e-4, 1), ylim=(0, 1.01), xlabel="False-positive rate (log)",
           ylabel="True-positive rate", title=title)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def family_heatmap(df: pd.DataFrame, path: Path, title: str, fmt: str = ".2f") -> None:
    """df: rows = models, cols = families."""
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * df.shape[1], 0.8 + 0.5 * df.shape[0]))
    im = ax.imshow(df.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=35, ha="right")
    ax.set_yticks(range(df.shape[0]), df.index)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.values[i, j]
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8,
                    color="white" if v < 0.6 else "black")
    ax.grid(False)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def score_hist(scores: np.ndarray, families: np.ndarray, thr: float, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    s = np.log10(np.maximum(scores - scores.min() + 1e-6, 1e-6))
    t = np.log10(max(thr - scores.min() + 1e-6, 1e-6))
    bins = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.8), 70)
    for fam in sorted(set(families)):
        ax.hist(s[families == fam], bins=bins, histtype="step", lw=1.5, density=True, label=fam)
    ax.axvline(t, c="red", ls="--", lw=1.2, label="threshold")
    ax.set(xlabel="log10 anomaly score (shifted)", ylabel="density", title=title)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def loss_curves(histories: dict[str, list[dict]], path: Path) -> None:
    histories = {k: v for k, v in histories.items() if v}
    if not histories:
        return
    fig, axes = plt.subplots(1, len(histories), figsize=(4 * len(histories), 3.2), squeeze=False)
    for ax, (name, h) in zip(axes[0], histories.items()):
        h = pd.DataFrame(h)
        ax.plot(h["epoch"], h["train_loss"], label="train")
        ax.plot(h["epoch"], h["val_loss"], label="val (benign)")
        ax.set(title=name, xlabel="epoch", ylabel="loss")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def stream_plot(df: pd.DataFrame, path: Path, title: str, drift_window: int | None = None) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    for mode, c in [("static", "#d95f02"), ("adaptive", "#1b9e77")]:
        d = df[df["mode"] == mode]
        axes[0].plot(d["window"], d["fpr"].rolling(3, min_periods=1).mean(), c=c, label=mode)
        axes[1].plot(d["window"], d["recall"].rolling(3, min_periods=1).mean(), c=c, label=mode)
        if mode == "adaptive":
            for w in d.loc[d["adapted"], "window"]:
                for ax in axes:
                    ax.axvline(w, c=c, ls=":", lw=1)
    if drift_window is not None:
        for ax in axes:
            ax.axvline(drift_window, c="black", ls="--", lw=1.2)
        axes[0].text(drift_window, axes[0].get_ylim()[1] * 0.9, " true drift", fontsize=8)
    axes[0].set(ylabel="False-positive rate", title=title)
    axes[1].set(ylabel="Attack recall", xlabel="stream window", ylim=(0, 1.02))
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
