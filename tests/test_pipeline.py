import json

import numpy as np
import pandas as pd

from iotids.cli import predict
from iotids.drift import StreamConfig, compare_static_vs_adaptive
from iotids.metrics import evaluate, robust_stats
from iotids.models import build
from iotids.pipeline import run_batch, run_stream


def test_metrics_perfect_and_random():
    y = np.r_[np.zeros(100), np.ones(100)].astype(int)
    m = evaluate(y, y.astype(float), 0.5)
    assert m["roc_auc"] == 1.0 and m["f1"] == 1.0 and m["fpr"] == 0.0
    rng = np.random.default_rng(0)
    assert 0.35 < evaluate(y, rng.random(200), 0.5)["roc_auc"] < 0.65


def test_robust_stats_resist_contamination():
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 1, 9000)
    dirty = np.r_[clean, rng.normal(50, 1, 3000)]  # 25% outliers
    med, sig = robust_stats(dirty)
    assert abs(med) < 0.5 and sig < 2.0


def test_adaptation_reduces_fpr_under_drift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, (3000, 8)).astype(np.float32)
    cal = rng.normal(0, 1, (1000, 8)).astype(np.float32)
    benign = rng.normal(0, 1, (12000, 8)).astype(np.float32)
    benign[4000:, :3] += 2.0  # drift
    attacks = rng.normal(0, 1, (1500, 8)).astype(np.float32) + 4.0
    X = np.r_[benign, attacks]
    y = np.r_[np.zeros(len(benign)), np.ones(len(attacks))].astype(int)
    order = np.argsort(np.r_[np.arange(len(benign)), rng.uniform(0, len(benign), len(attacks))], kind="stable")
    X, y = X[order], y[order]
    drift_idx = int(np.searchsorted(np.cumsum(y == 0), 4000))
    det = build("iforest", {"n_estimators": 100}, seed=0).fit(ref)
    cfg = StreamConfig(window=500, min_buffer=1000, cooldown_windows=2)
    res = compare_static_vs_adaptive(det, ref, cal, X, y, np.where(y, "A", "Benign"), cfg, drift_idx)
    post = res[res["after_true_drift"]].groupby("mode")["fpr"].mean()
    assert res.loc[res["mode"] == "adaptive", "adapted"].any()
    assert post["adaptive"] < post["static"]


def test_end_to_end(cfg, tmp_path):
    run = run_batch(cfg)
    for f in ["metrics.csv", "per_family_detection.csv", "summary.json", "test_scores.csv.gz",
              "figures/roc_zero_day.png", "models/preprocessor.joblib", "models/ae.pt"]:
        assert (run / f).exists(), f
    summ = json.load(open(run / "summary.json"))
    assert summ["splits"]["train"] == {"Benign": summ["splits"]["train"]["Benign"]}

    sample = pd.read_csv(sorted((tmp_path / "data").glob("*.csv"))[0]).head(200)
    sample.to_csv(tmp_path / "in.csv", index=False)
    out = predict(run, "ae", tmp_path / "in.csv")
    assert {"anomaly_score", "is_attack_pred"} <= set(out.columns) and len(out) == 200

    srun = run_stream(cfg)
    w = pd.read_csv(srun / "stream_windows.csv")
    assert set(w["mode"]) == {"static", "adaptive"}
