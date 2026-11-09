"""Streaming evaluation with drift detection and self-adaptation.

Labels are used ONLY to compute metrics — never by the adaptation logic.

Per window of incoming flows:
 1. score flows, flag those above the current threshold
 2. take the "likely benign" part of the window: scores <= trim_quantile of the window
    (robust as long as attacks are a minority of the window)
 3. feed those scores to a drift detector (ADWIN / Page-Hinkley) and keep the rows in a buffer
 4. on drift (after warm-up / cooldown): fine-tune the model on buffer + replay of the
    original benign reference (avoids catastrophic forgetting), then re-calibrate the
    threshold robustly (median + z0 * MAD-sigma on recent windows, where z0 reproduces the
    original benign-quantile threshold)
"""
from __future__ import annotations

import copy
import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import benign_threshold, robust_stats
from .models.base import Detector

log = logging.getLogger(__name__)


def make_drift_detector(kind: str = "adwin", **kw):
    from river import drift

    if kind == "adwin":
        return drift.ADWIN(delta=kw.get("delta", 0.002))
    if kind == "page_hinkley":
        return drift.PageHinkley(threshold=kw.get("threshold", 50.0), delta=kw.get("ph_delta", 0.005))
    raise ValueError(f"unknown drift detector {kind}")


@dataclass
class StreamConfig:
    window: int = 1000
    threshold_quantile: float = 0.99
    trim_quantile: float = 0.6
    detector: str = "adwin"
    delta: float = 0.002
    buffer_size: int = 8000
    min_buffer: int = 3000
    replay_ratio: float = 0.3
    cooldown_windows: int = 5
    recalib_windows: int = 3

    @classmethod
    def from_dict(cls, d: dict) -> "StreamConfig":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


class AdaptiveStream:
    def __init__(self, model: Detector, X_ref: np.ndarray, X_cal: np.ndarray, cfg: StreamConfig,
                 adaptive: bool = True, seed: int = 42):
        self.model = model
        self.cfg = cfg
        self.adaptive = adaptive
        self.rng = np.random.default_rng(seed)
        self.X_ref = X_ref
        s_cal = model.score(X_cal)
        self.thr = benign_threshold(s_cal, cfg.threshold_quantile)
        med, sig = robust_stats(s_cal)
        self.z0 = (self.thr - med) / sig  # standardized threshold to preserve after adaptation
        self.dd = make_drift_detector(cfg.detector, delta=cfg.delta)
        self.buffer: deque = deque(maxlen=cfg.buffer_size)
        self.recent_X: deque = deque(maxlen=cfg.recalib_windows)
        self.since_adapt = 0
        self.n_adaptations = 0

    def _adapt(self) -> None:
        buf = np.asarray(self.buffer)
        n_rep = int(len(buf) * self.cfg.replay_ratio / (1 - self.cfg.replay_ratio))
        rep = self.X_ref[self.rng.choice(len(self.X_ref), min(n_rep, len(self.X_ref)), replace=False)]
        Xa = np.concatenate([buf, rep])
        self.rng.shuffle(Xa)
        self.model.adapt(Xa)
        recent = self.model.score(np.concatenate(list(self.recent_X)))
        med, sig = robust_stats(recent)
        self.thr = med + self.z0 * sig
        self.dd = make_drift_detector(self.cfg.detector, delta=self.cfg.delta)
        self.since_adapt = 0
        self.n_adaptations += 1

    def run(self, X: np.ndarray, y: np.ndarray, families: np.ndarray, drift_index: int | None = None) -> pd.DataFrame:
        rows = []
        w = self.cfg.window
        for i, start in enumerate(range(0, len(X), w)):
            Xw, yw = X[start : start + w], y[start : start + w]
            s = self.model.score(Xw)
            pred = s > self.thr
            drift_flag = adapted = False

            cut = np.quantile(s, self.cfg.trim_quantile)
            likely_benign = s <= cut
            for v in s[likely_benign]:
                self.dd.update(float(v))
                if self.dd.drift_detected:
                    drift_flag = True
            self.buffer.extend(Xw[likely_benign])
            self.recent_X.append(Xw)
            self.since_adapt += 1

            if (self.adaptive and drift_flag and self.since_adapt >= self.cfg.cooldown_windows
                    and len(self.buffer) >= self.cfg.min_buffer):
                log.info("window %d: drift detected -> adapting (#%d)", i, self.n_adaptations + 1)
                self._adapt()
                adapted = True

            b, a = yw == 0, yw == 1
            rows.append({
                "window": i,
                "start": start,
                "n": len(yw),
                "attack_frac": float(a.mean()),
                "fpr": float(pred[b].mean()) if b.any() else np.nan,
                "recall": float(pred[a].mean()) if a.any() else np.nan,
                "threshold": self.thr,
                "median_score": float(np.median(s)),
                "drift_detected": drift_flag,
                "adapted": adapted,
                "after_true_drift": drift_index is not None and start >= drift_index,
            })
        return pd.DataFrame(rows)


def compare_static_vs_adaptive(model: Detector, X_ref, X_cal, X_stream, y_stream, fam_stream,
                               cfg: StreamConfig, drift_index: int | None, seed: int = 42) -> pd.DataFrame:
    out = []
    for mode, adaptive in [("static", False), ("adaptive", True)]:
        runner = AdaptiveStream(copy.deepcopy(model), X_ref, X_cal, cfg, adaptive=adaptive, seed=seed)
        df = runner.run(X_stream, y_stream, fam_stream, drift_index)
        df["mode"] = mode
        df["n_adaptations"] = runner.n_adaptations
        out.append(df)
    return pd.concat(out, ignore_index=True)


def summarize_stream(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["mode", "after_true_drift"]).agg(
        fpr=("fpr", "mean"), recall=("recall", "mean"), windows=("window", "count")
    ).reset_index()
    g["period"] = np.where(g["after_true_drift"], "post-drift", "pre-drift")
    return g.drop(columns="after_true_drift")[["mode", "period", "fpr", "recall", "windows"]]
