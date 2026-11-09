"""Self-supervised deep detectors (PyTorch).

* AutoencoderDetector   — reconstruction error
* VAEDetector           — reconstruction error under a variational bottleneck
* ContrastiveDetector   — SCARF-style contrastive pretraining (random feature corruption),
                          scored by kNN distance to benign embeddings
All train on benign traffic only.
"""
from __future__ import annotations

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .base import Detector

log = logging.getLogger(__name__)


def _device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def mlp(dims: list[int], dropout: float = 0.0, last_act: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or last_act:
            layers += [nn.BatchNorm1d(dims[i + 1]), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- base trainer
class TorchDetector(Detector):
    def __init__(self, hidden: list[int] | None = None, latent: int = 16, dropout: float = 0.1,
                 lr: float = 1e-3, weight_decay: float = 1e-5, epochs: int = 50,
                 batch_size: int = 512, patience: int = 5, adapt_epochs: int = 5,
                 adapt_lr_mult: float = 0.3, device: str = "auto", seed: int = 42, **_):
        self.hidden = hidden or [128, 64]
        self.latent, self.dropout = latent, dropout
        self.lr, self.weight_decay = lr, weight_decay
        self.epochs, self.batch_size, self.patience = epochs, batch_size, patience
        self.adapt_epochs, self.adapt_lr_mult = adapt_epochs, adapt_lr_mult
        self.device_pref, self.seed = device, seed
        self.device = _device(device)
        self.net: nn.Module | None = None
        self.history: list[dict] = []

    # subclasses implement
    def build(self, in_dim: int) -> nn.Module: ...
    def loss(self, xb: torch.Tensor) -> torch.Tensor: ...
    def _score_batch(self, xb: torch.Tensor) -> torch.Tensor: ...

    def _loader(self, X: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32))
        drop_last = shuffle and len(ds) > self.batch_size  # avoid size-1 batches breaking BatchNorm
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last)

    def _eval_loss(self, X: np.ndarray) -> float:
        self.net.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for (xb,) in self._loader(X, shuffle=False):
                xb = xb.to(self.device)
                tot += self.loss(xb).item() * len(xb)
                n += len(xb)
        return tot / max(n, 1)

    def _train(self, X: np.ndarray, X_val: np.ndarray | None, epochs: int, lr: float) -> None:
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        best, best_state, bad = np.inf, None, 0
        for ep in range(epochs):
            self.net.train()
            tot, n = 0.0, 0
            for (xb,) in self._loader(X, shuffle=True):
                xb = xb.to(self.device)
                loss = self.loss(xb)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                tot += loss.item() * len(xb)
                n += len(xb)
            sched.step()
            tr = tot / max(n, 1)
            va = self._eval_loss(X_val) if X_val is not None and len(X_val) else tr
            self.history.append({"epoch": len(self.history) + 1, "train_loss": tr, "val_loss": va})
            log.debug("[%s] epoch %d train=%.4f val=%.4f", self.name, ep + 1, tr, va)
            if va < best - 1e-5:
                best, best_state, bad = va, copy.deepcopy(self.net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    log.info("[%s] early stop at epoch %d (best val=%.4f)", self.name, ep + 1, best)
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)

    def fit(self, X, X_val=None):
        torch.manual_seed(self.seed)
        self.in_dim = X.shape[1]
        self.net = self.build(self.in_dim).to(self.device)
        self.history = []
        self._train(X, X_val, self.epochs, self.lr)
        self._after_fit(X)
        return self

    def adapt(self, X):
        """Fine-tune on recent benign-looking data (low LR, few epochs) — keeps prior knowledge."""
        n_val = max(1, int(0.15 * len(X)))
        self._train(X[n_val:], X[:n_val], self.adapt_epochs, self.lr * self.adapt_lr_mult)
        self._after_fit(X)
        return self

    def _after_fit(self, X: np.ndarray) -> None:
        pass

    def score(self, X):
        self.net.eval()
        out = []
        with torch.no_grad():
            for (xb,) in self._loader(X, shuffle=False):
                out.append(self._score_batch(xb.to(self.device)).cpu())
        return torch.cat(out).numpy().astype(np.float64)

    # ---- persistence
    def _params(self) -> dict:
        keys = ["hidden", "latent", "dropout", "lr", "weight_decay", "epochs", "batch_size",
                "patience", "adapt_epochs", "adapt_lr_mult", "device_pref", "seed"]
        p = {k: getattr(self, k) for k in keys}
        p["device"] = p.pop("device_pref")
        return p

    def _extra_state(self) -> dict:
        return {}

    def _load_extra(self, state: dict) -> None:
        pass

    def save(self, path):
        torch.save({"params": self._params(), "in_dim": self.in_dim, "state": self.net.state_dict(),
                    "history": self.history, "extra": self._extra_state()}, path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(**ck["params"])
        obj.in_dim = ck["in_dim"]
        obj.net = obj.build(obj.in_dim).to(obj.device)
        obj.net.load_state_dict(ck["state"])
        obj.history = ck.get("history", [])
        obj._load_extra(ck.get("extra", {}))
        obj.net.eval()
        return obj


# --------------------------------------------------------------------------- autoencoder
class _AE(nn.Module):
    def __init__(self, d, hidden, latent, dropout):
        super().__init__()
        self.enc = mlp([d, *hidden, latent], dropout)
        self.dec = mlp([latent, *hidden[::-1], d], dropout)

    def forward(self, x):
        return self.dec(self.enc(x))


class AutoencoderDetector(TorchDetector):
    name = "ae"

    def build(self, d):
        return _AE(d, self.hidden, self.latent, self.dropout)

    def loss(self, xb):
        return F.mse_loss(self.net(xb), xb)

    def _score_batch(self, xb):
        return ((self.net(xb) - xb) ** 2).mean(dim=1)


# --------------------------------------------------------------------------- VAE
class _VAE(nn.Module):
    def __init__(self, d, hidden, latent, dropout):
        super().__init__()
        self.body = mlp([d, *hidden], dropout, last_act=True)
        self.mu = nn.Linear(hidden[-1], latent)
        self.logvar = nn.Linear(hidden[-1], latent)
        self.dec = mlp([latent, *hidden[::-1], d], dropout)

    def forward(self, x):
        h = self.body(x)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-10, 10)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.dec(z), mu, logvar


class VAEDetector(TorchDetector):
    name = "vae"

    def __init__(self, beta: float = 0.5, **kw):
        super().__init__(**kw)
        self.beta = beta

    def _params(self):
        return {**super()._params(), "beta": self.beta}

    def build(self, d):
        return _VAE(d, self.hidden, self.latent, self.dropout)

    def loss(self, xb):
        recon, mu, logvar = self.net(xb)
        rec = F.mse_loss(recon, xb, reduction="none").sum(1)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
        return (rec + self.beta * kl).mean() / xb.shape[1]

    def _score_batch(self, xb):
        recon, _, _ = self.net(xb)
        return ((recon - xb) ** 2).mean(dim=1)


# --------------------------------------------------------------------------- contrastive (SCARF)
class _Contrastive(nn.Module):
    def __init__(self, d, hidden, latent, dropout):
        super().__init__()
        self.encoder = mlp([d, *hidden, latent], dropout)
        self.head = mlp([latent, latent, latent])

    def forward(self, x):
        return self.head(self.encoder(x))


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / tau
    n = z1.shape[0]
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


class ContrastiveDetector(TorchDetector):
    """SCARF (Bahri et al., ICLR 2022): corrupt a random subset of features by sampling
    them from their empirical marginal, learn embeddings invariant to that corruption.
    Anomaly score = mean distance to the k nearest benign embeddings."""

    name = "contrastive"

    def __init__(self, corruption: float = 0.3, temperature: float = 0.5, k: int = 5,
                 bank_size: int = 20_000, noise_std: float = 0.05, **kw):
        super().__init__(**kw)
        self.corruption, self.temperature, self.k = corruption, temperature, k
        self.bank_size, self.noise_std = bank_size, noise_std
        self.bank: torch.Tensor | None = None

    def _params(self):
        return {**super()._params(), "corruption": self.corruption, "temperature": self.temperature,
                "k": self.k, "bank_size": self.bank_size, "noise_std": self.noise_std}

    def build(self, d):
        return _Contrastive(d, self.hidden, self.latent, self.dropout)

    def _corrupt(self, xb):
        mask = torch.rand_like(xb) < self.corruption
        # sample each feature from its marginal = same column, random row of the batch
        idx = torch.randint(0, xb.shape[0], xb.shape, device=xb.device)
        rand = torch.gather(xb, 0, idx)
        return torch.where(mask, rand, xb) + self.noise_std * torch.randn_like(xb)

    def loss(self, xb):
        return nt_xent(self.net(self._corrupt(xb)), self.net(self._corrupt(xb)), self.temperature)

    def _embed(self, X: np.ndarray) -> torch.Tensor:
        self.net.eval()
        out = []
        with torch.no_grad():
            for (xb,) in self._loader(X, shuffle=False):
                out.append(F.normalize(self.net.encoder(xb.to(self.device)), dim=1))
        return torch.cat(out)

    def _after_fit(self, X):
        rng = np.random.default_rng(self.seed)
        if len(X) > self.bank_size:
            X = X[rng.choice(len(X), self.bank_size, replace=False)]
        self.bank = self._embed(X)

    def _score_batch(self, xb):
        e = F.normalize(self.net.encoder(xb), dim=1)
        d = torch.cdist(e, self.bank)
        return d.topk(self.k, largest=False).values.mean(1)

    def _extra_state(self):
        return {"bank": self.bank.cpu()}

    def _load_extra(self, state):
        self.bank = state["bank"].to(self.device)
