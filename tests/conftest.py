import pytest

from iotids.synthetic import generate


@pytest.fixture(scope="session")
def small_df():
    df = generate(n_benign=4000, n_per_attack=120, seed=0)
    from iotids.data import family_of

    df["family"] = [family_of(x) for x in df["label"]]
    return df


@pytest.fixture
def cfg(tmp_path):
    from iotids.synthetic import write_csvs

    write_csvs(tmp_path / "data", n_files=2, n_benign=4000, n_per_attack=120, seed=0)
    return {
        "experiment": "test", "seed": 0, "output_dir": str(tmp_path / "runs"),
        "data": {"path": str(tmp_path / "data"), "pattern": "*.csv", "label_col": "label", "time_col": "ts"},
        "preprocess": {"log1p": True, "clip": 10.0},
        "split": {"mode": "random", "train_frac": 0.6, "val_frac": 0.2,
                  "unknown_families": ["Mirai", "Web", "BruteForce", "Spoofing"]},
        "threshold": {"quantile": 0.99},
        "models": [{"name": "iforest", "params": {"n_estimators": 50}},
                   {"name": "ae", "params": {"hidden": [32], "latent": 8, "epochs": 3}}],
        "stream": {"model": "ae", "fit_until": 0.4, "calibrate_until": 0.5, "known_drift_time": 0.6,
                   "window": 300, "min_buffer": 500, "buffer_size": 2000, "cooldown_windows": 2},
    }
