"""Synthetic IoT flow generator.

Produces CSVs that mimic the CIC-IoT2023 schema (same column style and label names)
so the full pipeline can be run and tested without downloading the real dataset.
It is NOT a substitute for real data — use it for smoke tests, CI and demos only.

Built-in realism:
* 3 benign device profiles (camera, sensor, smart plug) with heavy-tailed features
* 7 attack families with different "visibility" (DDoS is loud, Spoofing/Web are subtle)
* concept drift: after `drift_at` of the timeline, benign traffic shifts
  (simulating a firmware update / new device type)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ack_count", "syn_count", "fin_count", "urg_count",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Variance",
]

# label -> (family-level tweak function key)
ATTACK_LABELS = {
    "DDoS-UDP_Flood": "ddos", "DDoS-SYN_Flood": "ddos", "DDoS-ICMP_Flood": "ddos",
    "DoS-TCP_Flood": "dos", "DoS-HTTP_Flood": "dos",
    "Mirai-greeth_flood": "mirai", "Mirai-udpplain": "mirai",
    "Recon-PortScan": "recon", "Recon-OSScan": "recon", "VulnerabilityScan": "recon",
    "MITM-ArpSpoofing": "spoof", "DNS_Spoofing": "spoof",
    "DictionaryBruteForce": "brute",
    "SqlInjection": "web", "XSS": "web", "CommandInjection": "web",
}


def _benign(rng: np.random.Generator, n: int, drifted: np.ndarray) -> np.ndarray:
    d = len(FEATURES)
    profiles = np.array([
        rng.normal(0.0, 1.0, d),   # camera
        rng.normal(0.5, 1.0, d),   # sensor
        rng.normal(-0.5, 1.0, d),  # smart plug
    ])
    which = rng.integers(0, 3, n)
    z = profiles[which] + rng.normal(0, 0.6, (n, d))
    # correlated structure: a few latent factors
    W = rng.normal(0, 0.4, (4, d))
    z += rng.normal(0, 1, (n, 4)) @ W
    # drift: shift + rescale a subset of features for drifted rows
    shift = np.zeros(d)
    shift[[0, 4, 5, 15, 20, 21]] = [1.4, 1.2, 1.2, 1.0, 1.0, -1.1]
    z[drifted] += shift
    return z


def _attack(rng: np.random.Generator, z: np.ndarray, kind: str) -> np.ndarray:
    z = z.copy()
    idx = {f: i for i, f in enumerate(FEATURES)}
    if kind == "ddos":
        for f in ["Rate", "Srate", "Number", "Tot sum"]:
            z[:, idx[f]] += rng.normal(3.0, 0.7, len(z))
        z[:, idx["IAT"]] -= 2.5
    elif kind == "dos":
        for f in ["Rate", "Srate", "syn_count"]:
            z[:, idx[f]] += rng.normal(2.2, 0.8, len(z))
    elif kind == "mirai":
        for f in ["Tot size", "AVG", "Max", "Header_Length"]:
            z[:, idx[f]] += rng.normal(2.0, 0.8, len(z))
        z[:, idx["Protocol Type"]] += 1.5
    elif kind == "recon":
        z[:, idx["syn_flag_number"]] += rng.normal(2.0, 0.6, len(z))
        z[:, idx["rst_flag_number"]] += rng.normal(1.8, 0.6, len(z))
        z[:, idx["Tot size"]] -= 1.5
    elif kind == "spoof":  # subtle
        z[:, idx["ack_count"]] += rng.normal(1.0, 0.6, len(z))
        z[:, idx["Variance"]] += rng.normal(0.8, 0.5, len(z))
    elif kind == "brute":
        z[:, idx["psh_flag_number"]] += rng.normal(1.6, 0.6, len(z))
        z[:, idx["flow_duration"]] += rng.normal(1.2, 0.5, len(z))
    elif kind == "web":  # subtle
        z[:, idx["psh_flag_number"]] += rng.normal(0.9, 0.5, len(z))
        z[:, idx["Std"]] += rng.normal(0.9, 0.5, len(z))
    return z


def _to_flow_space(z: np.ndarray) -> np.ndarray:
    """Map latent gaussian space to positive heavy-tailed 'flow' values."""
    return np.round(np.expm1(np.clip(z, -5, 8) * 0.9 + 3.0), 4)


def generate(n_benign: int = 60_000, n_per_attack: int = 1_500, drift_at: float = 0.6,
             seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t_benign = np.sort(rng.uniform(0, 1, n_benign))
    zb = _benign(rng, n_benign, drifted=t_benign >= drift_at)
    frames = [pd.DataFrame(_to_flow_space(zb), columns=FEATURES).assign(label="BenignTraffic", ts=t_benign)]

    for label, kind in ATTACK_LABELS.items():
        t = np.sort(rng.uniform(0, 1, n_per_attack))
        base = _benign(rng, n_per_attack, drifted=t >= drift_at)
        za = _attack(rng, base, kind)
        frames.append(pd.DataFrame(_to_flow_space(za), columns=FEATURES).assign(label=label, ts=t))

    df = pd.concat(frames, ignore_index=True)
    # inject a few dirty values like the real data has
    dirty = rng.choice(len(df), size=max(1, len(df) // 2000), replace=False)
    df.loc[dirty, "Rate"] = np.inf
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def write_csvs(out_dir: str | Path, n_files: int = 4, **kwargs) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate(**kwargs)
    paths = []
    for i, part in enumerate(np.array_split(np.arange(len(df)), n_files)):
        p = out_dir / f"part-{i:05d}.csv"
        df.iloc[part].to_csv(p, index=False)
        paths.append(p)
    return paths
