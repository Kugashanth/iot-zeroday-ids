"""End-to-end experiments.

* run_batch  — zero-day benchmark: train every model on benign, evaluate on unseen attack families
* run_stream — drift experiment: static vs self-adapting detector over a time-ordered stream
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import models as M
from . import plots
from .config import set_seed
from .data import BENIGN, Preprocessor, is_attack, load_dataset, make_splits
from .drift import StreamConfig, compare_static_vs_adaptive, summarize_stream
from .metrics import benign_threshold, evaluate, family_auc, per_family

log = logging.getLogger(__name__)


def _new_run_dir(cfg: dict, kind: str) -> Path:
    root = Path(cfg.get("output_dir", "runs"))
    name = f"{cfg.get('experiment', 'exp')}-{kind}-{datetime.now():%Y%m%d-%H%M%S}"
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (root / "LATEST").write_text(str(d.resolve()))
    with open(d / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return d


def _preprocessor(cfg: dict) -> Preprocessor:
    d = cfg["data"]
    drop = list(d.get("drop_cols", []))
    if d.get("time_col"):
        drop.append(d["time_col"])
    p = cfg.get("preprocess", {})
    return Preprocessor(drop_cols=drop, log1p=p.get("log1p", True), clip=p.get("clip", 10.0))


def _mlflow(cfg: dict):
    if not cfg.get("mlflow", {}).get("enabled"):
        return None
    try:
        import mlflow

        mlflow.set_experiment(cfg.get("experiment", "iot-ids"))
        return mlflow
    except ImportError:
        log.warning("mlflow not installed — skipping tracking")
        return None


# ============================================================================ batch
def run_batch(cfg: dict) -> Path:
    set_seed(cfg.get("seed", 42))
    run = _new_run_dir(cfg, "batch")
    (run / "models").mkdir(exist_ok=True)
    df = load_dataset(cfg)
    sp = make_splits(df, cfg)
    json.dump(sp.summary(), open(run / "splits.json", "w"), indent=2)

    pre = _preprocessor(cfg).fit(sp.train)
    pre.save(run / "models" / "preprocessor.joblib")
    Xtr, Xva, Xte = pre.transform(sp.train), pre.transform(sp.val), pre.transform(sp.test)
    Xkn = pre.transform(sp.known) if len(sp.known) else np.empty((0, Xtr.shape[1]), np.float32)
    y_te = is_attack(sp.test)
    b_te = sp.test[sp.test["family"] == BENIGN]
    q = cfg["threshold"]["quantile"]
    log.info("features: %d | train=%d val=%d test=%d known=%d", Xtr.shape[1], len(Xtr), len(Xva), len(Xte), len(Xkn))

    mlf = _mlflow(cfg)
    rows, fam_rows, auc_rows, test_scores, histories = [], {}, {}, {}, {}
    for spec in cfg["models"]:
        name, params = spec["name"], spec.get("params", {})
        log.info("=== %s ===", name)
        det = M.build(name, params, seed=cfg.get("seed", 42))
        t0 = time.perf_counter()
        det.fit(Xtr, Xva)
        fit_s = time.perf_counter() - t0
        s_va = det.score(Xva)
        thr = benign_threshold(s_va, q)
        t0 = time.perf_counter()
        s_te = det.score(Xte)
        infer_us = (time.perf_counter() - t0) / len(Xte) * 1e6

        m = {"model": name, "eval": "zero-day (unknown families)", **evaluate(y_te, s_te, thr),
             "fit_seconds": fit_s, "infer_us_per_flow": infer_us}
        rows.append(m)
        fams = sp.test["family"].to_numpy()
        all_scores = s_te
        if len(Xkn):
            s_kn = det.score(Xkn)
            n_b = len(b_te)
            yk = np.r_[np.zeros(n_b), np.ones(len(s_kn))].astype(int)
            sk = np.r_[s_te[: n_b], s_kn]  # benign test rows come first in sp.test
            rows.append({"model": name, "eval": "known families", **evaluate(yk, sk, thr)})
            fams = np.r_[fams, sp.known["family"].to_numpy()]
            all_scores = np.r_[s_te, s_kn]
        fam_rows[name] = per_family(fams, all_scores, thr)
        auc_rows[name] = family_auc(fams, all_scores)
        test_scores[name] = s_te
        histories[name] = getattr(det, "history", [])
        det.save(run / "models" / f"{name}{M.FILE_EXT[name]}")
        json.dump({"threshold": thr, "quantile": q}, open(run / "models" / f"{name}.threshold.json", "w"))
        log.info("%s: ROC-AUC=%.4f PR-AUC=%.4f recall=%.3f FPR=%.4f (fit %.1fs)",
                 name, m.get("roc_auc", np.nan), m.get("pr_auc", np.nan), m["recall"], m["fpr"], fit_s)
        if mlf:
            with mlf.start_run(run_name=f"{run.name}-{name}"):
                mlf.log_params({"model": name, **{k: str(v) for k, v in params.items()}})
                mlf.log_metrics({k.replace("@", "_at_").replace("%", "pct").replace("=", ""): v
                                 for k, v in m.items() if isinstance(v, (int, float))})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(run / "metrics.csv", index=False)
    fam_df = pd.DataFrame(fam_rows).T
    fam_df = fam_df[[BENIGN] + sorted(c for c in fam_df.columns if c != BENIGN)]
    fam_df.to_csv(run / "per_family_detection.csv")
    auc_df = pd.DataFrame(auc_rows).T
    auc_df.to_csv(run / "per_family_auc.csv")

    # scores for the dashboard
    sc = pd.DataFrame({"family": sp.test["family"].values, "label": sp.test["label"].values, **test_scores})
    sc.to_csv(run / "test_scores.csv.gz", index=False)

    fig_dir = run / "figures"
    fig_dir.mkdir(exist_ok=True)
    plots.roc_curves(y_te, test_scores, fig_dir / "roc_zero_day.png", "Zero-day ROC (unknown families)")
    plots.family_heatmap(fam_df, fig_dir / "per_family_detection.png",
                         f"Detection rate @ benign q={q} (Benign column = FPR)")
    plots.family_heatmap(auc_df, fig_dir / "per_family_auc.png", "Per-family ROC-AUC vs benign", ".3f")
    best = metrics[metrics["eval"].str.startswith("zero")].sort_values("roc_auc", ascending=False).iloc[0]["model"]
    thr_best = json.load(open(run / "models" / f"{best}.threshold.json"))["threshold"]
    plots.score_hist(test_scores[best], sp.test["family"].to_numpy(), thr_best,
                     fig_dir / f"score_dist_{best}.png", f"Score distribution — {best}")
    plots.loss_curves(histories, fig_dir / "training_curves.png")

    summary = {"run_dir": str(run), "best_model_by_roc_auc": best, "n_features": int(Xtr.shape[1]),
               "splits": sp.summary(), "metrics": metrics.round(5).to_dict(orient="records")}
    json.dump(summary, open(run / "summary.json", "w"), indent=2, default=float)
    print("\n" + metrics.drop(columns=["tp", "fp", "fn", "tn"], errors="ignore").round(4).to_string(index=False))
    print(f"\nPer-family detection rate:\n{fam_df.round(3).to_string()}")
    print(f"\nArtifacts -> {run}")
    return run


# ============================================================================ stream
def build_stream(df: pd.DataFrame, cfg: dict):
    """Returns ref (benign, fit), cal (benign, threshold), stream (mixed, time-ordered), drift_index.

    * With a time column: split by time; drift is whatever really happens in the data.
    * Without one (e.g. CIC-IoT2023 CSVs): random order + SIMULATED drift injected later.
    """
    s = cfg["stream"]
    tcol = cfg["data"].get("time_col")
    rng = np.random.default_rng(cfg.get("seed", 42))
    benign = df[df["family"] == BENIGN]
    if tcol and tcol in df.columns:
        t = df[tcol]
        lo, hi = t.quantile(s["fit_until"]), t.quantile(s["calibrate_until"])
        ref = benign[benign[tcol] < lo]
        cal = benign[(benign[tcol] >= lo) & (benign[tcol] < hi)]
        stream = df[df[tcol] >= hi].reset_index(drop=True)
        drift_index = None
        if s.get("known_drift_time") is not None:
            drift_index = int(np.searchsorted(stream[tcol].to_numpy(), s["known_drift_time"]))
        return ref, cal, stream, drift_index, False
    b = benign.sample(frac=1.0, random_state=cfg.get("seed", 42))
    n_ref, n_cal = int(len(b) * s["fit_until"]), int(len(b) * (s["calibrate_until"] - s["fit_until"]))
    ref, cal = b.iloc[:n_ref], b.iloc[n_ref : n_ref + n_cal]
    rest = pd.concat([b.iloc[n_ref + n_cal :], df[df["family"] != BENIGN]])
    stream = rest.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    drift_index = int(len(stream) * s.get("simulate_drift_at", 0.4))
    return ref, cal, stream, drift_index, True


def simulate_drift(X: np.ndarray, start: int, magnitude: float, n_features: int, seed: int) -> np.ndarray:
    """Shift a random subset of (standardized) features from `start` onward — e.g. a firmware
    update changing packet sizes/rates for every flow on the network."""
    rng = np.random.default_rng(seed)
    X = X.copy()
    feats = rng.choice(X.shape[1], min(n_features, X.shape[1]), replace=False)
    X[start:, feats] += magnitude * rng.choice([-1, 1], len(feats))
    return X


def run_stream(cfg: dict, model_name: str | None = None) -> Path:
    set_seed(cfg.get("seed", 42))
    s = cfg["stream"]
    model_name = model_name or s.get("model", "ae")
    run = _new_run_dir(cfg, f"stream-{model_name}")
    df = load_dataset(cfg)
    ref, cal, stream, drift_idx, simulated = build_stream(df, cfg)
    pre = _preprocessor(cfg).fit(ref)
    X_ref, X_cal, X_st = pre.transform(ref), pre.transform(cal), pre.transform(stream)
    if simulated:
        X_st = simulate_drift(X_st, drift_idx, s.get("drift_magnitude", 1.5),
                              s.get("drift_features", 6), cfg.get("seed", 42))
    log.info("stream: ref=%d cal=%d stream=%d drift_index=%s (simulated=%s)",
             len(X_ref), len(X_cal), len(X_st), drift_idx, simulated)

    spec = next((m for m in cfg["models"] if m["name"] == model_name), {"name": model_name})
    det = M.build(model_name, spec.get("params", {}), seed=cfg.get("seed", 42))
    n_v = int(0.15 * len(X_ref))
    det.fit(X_ref[n_v:], X_ref[:n_v])

    scfg = StreamConfig.from_dict({**s, "threshold_quantile": cfg["threshold"]["quantile"]})
    res = compare_static_vs_adaptive(det, X_ref, X_cal, X_st, is_attack(stream),
                                     stream["family"].to_numpy(), scfg, drift_idx, cfg.get("seed", 42))
    res.to_csv(run / "stream_windows.csv", index=False)
    summ = summarize_stream(res)
    summ.to_csv(run / "stream_summary.csv", index=False)
    (run / "figures").mkdir(exist_ok=True)
    dw = None if drift_idx is None else drift_idx // scfg.window
    plots.stream_plot(res, run / "figures" / "stream_static_vs_adaptive.png",
                      f"Streaming detection — {model_name} ({'simulated' if simulated else 'natural'} drift)", dw)
    n_ad = int(res.loc[res["mode"] == "adaptive", "adapted"].sum())
    print(f"\n{summ.round(4).to_string(index=False)}\nadaptations: {n_ad}\nArtifacts -> {run}")
    return run
