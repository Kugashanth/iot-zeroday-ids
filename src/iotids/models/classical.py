"""Classical one-class baselines."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline

from .base import Detector


class _SklearnDetector(Detector):
    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "Detector":
        return joblib.load(path)


class IsolationForestDetector(_SklearnDetector):
    name = "iforest"

    def __init__(self, n_estimators: int = 300, max_samples: int | str = 1024, seed: int = 42, **_):
        self.n_estimators, self.max_samples, self.seed = n_estimators, max_samples, seed

    def fit(self, X, X_val=None):
        ms = min(self.max_samples, len(X)) if isinstance(self.max_samples, int) else self.max_samples
        self.model = IsolationForest(
            n_estimators=self.n_estimators, max_samples=ms, random_state=self.seed, n_jobs=-1
        )
        self.model.fit(X)
        return self

    def score(self, X):
        return -self.model.score_samples(X)


class OCSVMDetector(_SklearnDetector):
    """Scalable One-Class SVM: RBF kernel approximated with Nystroem + linear SGD OC-SVM.

    Exact sklearn OneClassSVM is O(n^2) and unusable on millions of flows.
    """

    name = "ocsvm"

    def __init__(self, nu: float = 0.05, gamma: float | str = "scale", n_components: int = 300,
                 max_fit: int = 50_000, seed: int = 42, **_):
        self.nu, self.gamma, self.n_components = nu, gamma, n_components
        self.max_fit, self.seed = max_fit, seed

    def fit(self, X, X_val=None):
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_fit:
            X = X[rng.choice(len(X), self.max_fit, replace=False)]
        gamma = 1.0 / (X.shape[1] * X.var()) if self.gamma == "scale" else self.gamma
        self.model = make_pipeline(
            Nystroem(gamma=gamma, n_components=self.n_components, random_state=self.seed),
            SGDOneClassSVM(nu=self.nu, random_state=self.seed, max_iter=2000, tol=1e-4),
        )
        self.model.fit(X)
        return self

    def score(self, X):
        return -self.model.decision_function(X)
