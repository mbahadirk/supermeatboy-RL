"""The world model: S(t+1) = f(S(t), A(t)) on structured game state (wm/features.py), no images.

An ensemble of E small MLPs. Each predicts, for one agent and one step:
  - position change and new velocity: a mean and a log-variance each (Gaussian), so the model says how sure
    it is (aleatoric); the spread between ensemble members says how much it has seen such a case (epistemic)
  - next flags (airborne, on a wall, facing), death, finish, landing, blocked by a wall (logits)
  - next movement state (classes)
  - the step's reward
The heads are slices of one output layer, kept apart in `split` / `predict` and in the loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm.features import FEAT_DIM, N_STATES, BIN_NAMES

N_CONT = 4
OUT = 2 * N_CONT + len(BIN_NAMES) + N_STATES + 1
LOGVAR_MIN, LOGVAR_MAX = -10.0, 4.0


class WorldModel(nn.Module):
    def __init__(self, n_in: int = FEAT_DIM, members: int = 5, hidden: int = 256):
        super().__init__()
        self.members = nn.ModuleList(
            nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, OUT))
            for _ in range(members))
        self.register_buffer("mu", torch.zeros(n_in))
        self.register_buffer("sd", torch.ones(n_in))
        self.register_buffer("pos_weight", torch.ones(len(BIN_NAMES)))

    def set_norm(self, x: torch.Tensor):
        """Input normalisation from training data (binary inputs, with std ~0.5 or less, are left as they are)."""
        mu, sd = x.mean(0), x.std(0)
        binary = ((x == 0) | (x == 1)).all(0)
        self.mu.copy_(torch.where(binary, torch.zeros_like(mu), mu))
        self.sd.copy_(torch.where(binary | (sd < 1e-4), torch.ones_like(sd), sd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(members, batch, OUT)"""
        z = (x - self.mu) / self.sd
        return torch.stack([m(z) for m in self.members])

    @staticmethod
    def split(out: torch.Tensor) -> dict:
        i = 0
        d = {}
        for k, n in (("mean", N_CONT), ("logvar", N_CONT), ("bin", len(BIN_NAMES)), ("state", N_STATES), ("reward", 1)):
            d[k] = out[..., i:i + n]
            i += n
        d["logvar"] = d["logvar"].clamp(LOGVAR_MIN, LOGVAR_MAX)
        d["reward"] = d["reward"][..., 0]
        return d

    def loss(self, x, yc, yb, ys, yr) -> tuple[torch.Tensor, dict]:
        """Gaussian NLL (position / velocity), BCE (flags, events; rare ones weighted up), CE (state), MSE
        (reward), averaged over the ensemble members."""
        h = self.split(self.forward(x))
        nll = (0.5 * (h["logvar"] + (yc - h["mean"]) ** 2 / h["logvar"].exp())).mean()
        bce = F.binary_cross_entropy_with_logits(h["bin"], yb.expand_as(h["bin"]), pos_weight=self.pos_weight)
        ce = F.cross_entropy(h["state"].reshape(-1, N_STATES), ys.repeat(len(self.members)))
        mse = F.mse_loss(h["reward"], yr.expand_as(h["reward"]))
        total = nll + bce + 0.5 * ce + 0.2 * mse
        return total, {"nll": nll.item(), "bce": bce.item(), "ce": ce.item(), "mse": mse.item()}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        """Ensemble prediction: means, total std (aleatoric + spread between members), member spread alone,
        probabilities of the binary events, state probabilities, reward."""
        h = self.split(self.forward(x))
        mean = h["mean"].mean(0)
        alea = h["logvar"].exp().mean(0)
        epi = h["mean"].var(0, unbiased=False)
        return {"mean": mean, "std": (alea + epi).sqrt(), "spread": epi.sqrt(),
                "p": torch.sigmoid(h["bin"]).mean(0), "state": F.softmax(h["state"], -1).mean(0),
                "reward": h["reward"].mean(0)}

    def save(self, path: str, info: dict | None = None):
        torch.save({"state": self.state_dict(), "members": len(self.members), "n_in": self.mu.numel(),
                    "info": info or {}}, path)

    @classmethod
    def load(cls, path: str) -> "WorldModel":
        d = torch.load(path, map_location="cpu")
        m = cls(d["n_in"], d["members"])
        m.load_state_dict(d["state"])
        m.info = d.get("info", {})
        return m
