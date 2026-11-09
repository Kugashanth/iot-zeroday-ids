"""Data loading, family mapping, preprocessing and zero-day splits."""
from __future__ import annotations

import glob
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BENIGN = "Benign"

# CIC-IoT2023 has 34 labels -> 8 families (Neto et al., 2023)
_FAMILY_RULES = [
    ("benign", BENIGN),
    ("ddos", "DDoS"),
    ("dos", "DoS"),
    ("mirai", "Mirai"),
    ("recon", "Recon"),
    ("vulnerabilityscan", "Recon"),
    ("mitm", "Spoofing"),
    ("dns_spoofing", "Spoofing"),
    ("dictionarybruteforce", "BruteForce"),
    ("sqlinjection", "Web"),
    ("xss", "Web"),
    ("commandinjection", "Web"),
    ("backdoor_malware", "Web"),
    ("uploading_attack", "Web"),
    ("browserhijacking", "Web"),
]


def family_of(label: str, custom_map: dict | None = None) -> str:
    """Map a raw dataset label to an attack family."""
    if custom_map and label in custom_map:
        return custom_map[label]
    low = str(label).strip().lower()
    for prefix, fam in _FAMILY_RULES:
        if low.startswith(prefix):
            return fam
    return str(label)


# --------------------------------------------------------------------------- loading
def load_dataset(cfg: dict) -> pd.DataFrame:
    """Load CSV/parquet files described by cfg['data'] into one DataFrame.

    Adds `family` and keeps `label`. Caps rows per family to keep memory bounded
    (CIC-IoT2023 has ~46M rows; you rarely need all of them).
    """
    d = cfg["data"]
    files = sorted(glob.glob(str(Path(d["path"]) / d.get("pattern", "*.csv"))))
    if not files:
        raise FileNotFoundError(
            f"No files matching {d.get('pattern', '*.csv')} in {d['path']}. "
            "See data/README.md for download instructions, or run `iotids synth` first."
        )
    if d.get("max_files"):
        files = files[: d["max_files"]]
    rng = np.random.default_rng(cfg.get("seed", 42))
    frac = d.get("sample_frac", 1.0)

    parts = []
    for f in files:
        part = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f, low_memory=False)
        if frac < 1.0:
            part = part.sample(frac=frac, random_state=int(rng.integers(1 << 31)))
        parts.append(part)
        log.info("loaded %s (%d rows)", Path(f).name, len(part))
    df = pd.concat(parts, ignore_index=True)

    label_col = d.get("label_col", "label")
    if label_col not in df.columns:
        raise KeyError(f"label column '{label_col}' not in data; columns: {list(df.columns)[:10]}...")
    df = df.rename(columns={label_col: "label"})
    benign_values = set(d.get("benign_values", []))
    custom = d.get("family_map") or {}
    df["family"] = [
        BENIGN if lab in benign_values else family_of(lab, custom) for lab in df["label"].astype(str)
    ]

    cap = d.get("max_rows_per_family")
    if cap:
        df = (
            df.sample(frac=1.0, random_state=cfg.get("seed", 42))
            .groupby("family", sort=False).head(cap)
            .sort_index().reset_index(drop=True)
        )
    time_col = d.get("time_col")
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
    log.info("dataset: %d rows | families: %s", len(df), df["family"].value_counts().to_dict())
    return df


# --------------------------------------------------------------------------- preprocessing
@dataclass
class Preprocessor:
    """Fit on BENIGN TRAIN ONLY (no attack info leaks into scaling)."""

    drop_cols: list[str] = field(default_factory=list)
    log1p: bool = True
    clip: float = 10.0
    feature_names_: list[str] = field(default_factory=list)
    log_cols_: list[str] = field(default_factory=list)
    median_: pd.Series | None = None
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def _numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        reserved = {"label", "family", *self.drop_cols}
        cols = [c for c in df.columns if c not in reserved and pd.api.types.is_numeric_dtype(df[c])]
        return df[cols].replace([np.inf, -np.inf], np.nan)

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        X = self._numeric(df)
        X = X.loc[:, X.nunique(dropna=True) > 1]  # drop constant columns
        self.feature_names_ = list(X.columns)
        self.median_ = X.median()
        X = X.fillna(self.median_)
        if self.log1p:
            self.log_cols_ = [c for c in X.columns if (X[c] >= 0).all() and X[c].skew() > 1.0]
            X[self.log_cols_] = np.log1p(X[self.log_cols_])
        self.mean_ = X.mean().to_numpy()
        self.std_ = X.std().replace(0, 1.0).to_numpy()
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df.reindex(columns=self.feature_names_).replace([np.inf, -np.inf], np.nan)
        X = X.fillna(self.median_)
        if self.log_cols_:
            X[self.log_cols_] = np.log1p(X[self.log_cols_].clip(lower=0))
        Z = (X.to_numpy(dtype=np.float64) - self.mean_) / self.std_
        return np.clip(Z, -self.clip, self.clip).astype(np.float32)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "Preprocessor":
        return joblib.load(path)


# --------------------------------------------------------------------------- splits
@dataclass
class Splits:
    train: pd.DataFrame        # benign only  -> model fitting
    val: pd.DataFrame          # benign only  -> early stopping + threshold
    test: pd.DataFrame         # benign + UNKNOWN attack families (zero-day evaluation)
    known: pd.DataFrame        # KNOWN attack families (reported separately, never trained on)

    def summary(self) -> dict:
        return {k: getattr(self, k)["family"].value_counts().to_dict() for k in ("train", "val", "test", "known")}


def make_splits(df: pd.DataFrame, cfg: dict) -> Splits:
    """Zero-day protocol.

    * Models see benign traffic only.
    * Attack families in `unknown_families` appear only in the test set.
    * `known_families` are kept apart for reporting (and optional tuning), so you can
      compare "attacks like ones we have seen" vs "truly novel attacks".
    """
    s = cfg["split"]
    seed = cfg.get("seed", 42)
    benign = df[df["family"] == BENIGN]
    attacks = df[df["family"] != BENIGN]

    if s.get("mode", "random") == "chronological":
        b = benign.reset_index(drop=True)  # already time-sorted by load_dataset
    else:
        b = benign.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(b)
    n_tr, n_va = int(n * s["train_frac"]), int(n * s["val_frac"])
    train, val, b_test = b.iloc[:n_tr], b.iloc[n_tr : n_tr + n_va], b.iloc[n_tr + n_va :]

    all_fams = sorted(attacks["family"].unique())
    unknown = s.get("unknown_families") or all_fams
    unknown = [f for f in unknown if f in all_fams]
    known = [f for f in all_fams if f not in unknown]

    test = pd.concat([b_test, attacks[attacks["family"].isin(unknown)]], ignore_index=True)
    known_df = attacks[attacks["family"].isin(known)].reset_index(drop=True)
    sp = Splits(train.reset_index(drop=True), val.reset_index(drop=True), test, known_df)
    log.info("splits: %s", sp.summary())
    return sp


def is_attack(df: pd.DataFrame) -> np.ndarray:
    return (df["family"] != BENIGN).to_numpy().astype(int)
