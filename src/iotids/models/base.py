"""Common detector interface.

Every detector is trained on benign data only and returns an anomaly score where
HIGHER = MORE ANOMALOUS. This makes thresholds, metrics and drift handling model-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Detector(ABC):
    name: str = "detector"

    @abstractmethod
    def fit(self, X: np.ndarray, X_val: np.ndarray | None = None) -> "Detector":
        """Fit on benign samples. X_val (benign) may be used for early stopping."""

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score per row (higher = more anomalous)."""

    def adapt(self, X: np.ndarray) -> "Detector":
        """Update on recent (assumed benign) data after drift. Default: refit."""
        return self.fit(X)

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "Detector": ...
