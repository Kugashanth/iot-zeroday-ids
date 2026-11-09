"""Command-line interface.

    iotids synth   --out data/synthetic
    iotids train   --config configs/synthetic.yaml
    iotids stream  --config configs/synthetic.yaml --model ae
    iotids all     --config configs/synthetic.yaml
    iotids predict --run runs/<dir> --model ae --input flows.csv --output scored.csv
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from . import models as M
from .config import load_config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


def predict(run_dir: str | Path, model: str, input_csv: str | Path) -> pd.DataFrame:
    from .data import Preprocessor

    run_dir = Path(run_dir)
    pre = Preprocessor.load(run_dir / "models" / "preprocessor.joblib")
    det = M.load(model, run_dir / "models" / f"{model}{M.FILE_EXT[model]}")
    thr = json.load(open(run_dir / "models" / f"{model}.threshold.json"))["threshold"]
    df = pd.read_csv(input_csv, low_memory=False)
    s = det.score(pre.transform(df))
    return df.assign(anomaly_score=s, is_attack_pred=(s > thr).astype(int))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="iotids", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sy = sub.add_parser("synth", help="generate synthetic CIC-IoT2023-like CSVs")
    sy.add_argument("--out", default="data/synthetic")
    sy.add_argument("--n-benign", type=int, default=60_000)
    sy.add_argument("--n-per-attack", type=int, default=1_500)
    sy.add_argument("--seed", type=int, default=42)

    for name in ("train", "stream", "all"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default="configs/default.yaml")
        if name != "train":
            sp.add_argument("--model", default=None, help="model for the stream experiment")

    pr = sub.add_parser("predict", help="score a CSV with a trained run")
    pr.add_argument("--run", required=True)
    pr.add_argument("--model", default="ae")
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", default="scored.csv")

    a = p.parse_args(argv)
    _setup_logging(a.verbose)

    if a.cmd == "synth":
        from .synthetic import write_csvs

        paths = write_csvs(a.out, n_benign=a.n_benign, n_per_attack=a.n_per_attack, seed=a.seed)
        print(f"wrote {len(paths)} files to {a.out}")
        return
    if a.cmd == "predict":
        out = predict(a.run, a.model, a.input)
        out.to_csv(a.output, index=False)
        print(f"scored {len(out)} flows -> {a.output} ({out['is_attack_pred'].mean():.2%} flagged)")
        return

    from .pipeline import run_batch, run_stream

    cfg = load_config(a.config)
    if a.cmd in ("train", "all"):
        run_batch(cfg)
    if a.cmd in ("stream", "all"):
        run_stream(cfg, a.model)


if __name__ == "__main__":
    main()
