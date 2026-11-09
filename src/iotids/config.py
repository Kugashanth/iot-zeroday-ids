"""Config loading and reproducibility helpers."""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict[str, Any]:
    """Load a YAML config. Supports `inherit: other.yaml` (relative to the file)."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = deep_update(load_config(path.parent / parent), cfg)
    return deep_update(cfg, overrides or {})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
