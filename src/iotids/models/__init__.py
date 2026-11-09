from .base import Detector
from .classical import IsolationForestDetector, OCSVMDetector
from .deep import AutoencoderDetector, ContrastiveDetector, VAEDetector

REGISTRY: dict[str, type[Detector]] = {
    "iforest": IsolationForestDetector,
    "ocsvm": OCSVMDetector,
    "ae": AutoencoderDetector,
    "vae": VAEDetector,
    "contrastive": ContrastiveDetector,
}

FILE_EXT = {"iforest": ".joblib", "ocsvm": ".joblib", "ae": ".pt", "vae": ".pt", "contrastive": ".pt"}


def build(name: str, params: dict | None = None, seed: int = 42) -> Detector:
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'. Options: {list(REGISTRY)}")
    return REGISTRY[name](seed=seed, **(params or {}))


def load(name: str, path) -> Detector:
    return REGISTRY[name].load(path)


__all__ = ["Detector", "REGISTRY", "FILE_EXT", "build", "load"]
