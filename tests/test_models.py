import numpy as np
import pytest

from iotids import models as M
from iotids.data import BENIGN, Preprocessor
from iotids.metrics import benign_threshold, evaluate

FAST = {
    "iforest": {"n_estimators": 50},
    "ocsvm": {"n_components": 50},
    "ae": {"hidden": [32], "latent": 8, "epochs": 5},
    "vae": {"hidden": [32], "latent": 8, "epochs": 5},
    "contrastive": {"hidden": [32], "latent": 16, "epochs": 3, "bank_size": 1000},
}


@pytest.fixture(scope="module")
def arrays(small_df):
    b = small_df[small_df["family"] == BENIGN].sample(frac=1, random_state=0)
    pre = Preprocessor(drop_cols=["ts"]).fit(b.iloc[:2500])
    Xtr, Xva = pre.transform(b.iloc[:2500]), pre.transform(b.iloc[2500:3200])
    test = small_df.drop(b.iloc[:3200].index)
    return Xtr, Xva, pre.transform(test), (test["family"] != BENIGN).to_numpy().astype(int)


@pytest.mark.parametrize("name", list(M.REGISTRY))
def test_detector_learns_and_roundtrips(name, arrays, tmp_path):
    Xtr, Xva, Xte, y = arrays
    det = M.build(name, FAST[name], seed=0).fit(Xtr, Xva)
    s = det.score(Xte)
    assert s.shape == (len(Xte),) and np.isfinite(s).all()
    m = evaluate(y, s, benign_threshold(det.score(Xva), 0.99))
    assert m["roc_auc"] > 0.75, f"{name} failed to separate attacks: {m['roc_auc']:.3f}"

    path = tmp_path / f"m{M.FILE_EXT[name]}"
    det.save(path)
    s2 = M.load(name, path).score(Xte)
    assert np.allclose(s, s2, atol=1e-4)


@pytest.mark.parametrize("name", ["ae", "contrastive", "iforest"])
def test_adapt_runs(name, arrays):
    Xtr, Xva, _, _ = arrays
    det = M.build(name, FAST[name], seed=0).fit(Xtr, Xva)
    det.adapt(Xtr[:800] + 0.5)
    assert np.isfinite(det.score(Xva)).all()
