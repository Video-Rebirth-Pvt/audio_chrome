"""Language balance: temperature sampling over the speech tasks, capped in epochs.

By default a source is drawn in proportion to its train rows, so Japanese (84k rows) is 0.6% of the
samples next to Chinese (5.3M, 37%). With balance on, every matching task (one language: ``speech_ja``)
gets a share ∝ rows^alpha of the mass those tasks hold together; alpha = 1 is the plain mix, smaller
alpha moves share from the large languages to the small ones. Sources inside a task keep their
proportions, and the other tasks (music, sfx) keep their share.

``max_epochs`` caps how often a raised task is repeated over the planned run (train.max_steps × world ×
grad_accum batches of max_tokens); what a capped task cannot take goes to the others, by the same
temperature. A task is never pushed below its plain share. Weights are not in the index fingerprint and
not in the sampler state, so switching this on or off needs no rebuild and resumes as usual.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LanguageBalance:
    tasks: tuple[str, ...] = ("speech_*",)
    alpha: float = 0.8
    max_epochs: float | None = 10.0

    @classmethod
    def from_config(cls, cfg) -> "LanguageBalance | None":
        """None unless ``data.loader.language_balance.enabled``."""
        spec = cfg["data"]["loader"].get("language_balance") or {}
        if not spec.get("enabled"):
            return None
        max_epochs = spec.get("max_epochs", 10.0)
        balance = cls(
            tasks=tuple(str(p) for p in spec.get("tasks") or ["speech_*"]),
            alpha=float(spec.get("alpha", 0.8)),
            max_epochs=None if max_epochs in (None, 0) else float(max_epochs),
        )
        if not 0.0 < balance.alpha <= 1.0:
            raise ValueError(f"language_balance.alpha must be in (0, 1], got {balance.alpha}")
        if balance.max_epochs is not None and balance.max_epochs <= 0:
            raise ValueError(f"language_balance.max_epochs must be > 0 or null, got {balance.max_epochs}")
        return balance

    def matches(self, task: str) -> bool:
        return any(fnmatch.fnmatch(task, p) for p in self.tasks)

    def weights(self, tasks: Sequence[str], weights: np.ndarray, rows: np.ndarray, cost: np.ndarray,
                budget: float | None) -> tuple[np.ndarray, list[dict]]:
        """New source weights (same total) and one report entry per balanced task.

        ``weights``: plain mixture weights; ``rows``: train rows; ``cost``: mean packed cost of a sample;
        ``budget``: packed cost of the planned run (None: no epoch cap).
        """
        weights = np.asarray(weights, dtype=np.float64)
        rows = np.asarray(rows, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        total = weights.sum()
        plain = weights / total
        groups: dict[str, list[int]] = {}
        for i, task in enumerate(tasks):
            if self.matches(task) and weights[i] > 0:
                groups.setdefault(task, []).append(i)
        if not groups:
            return weights.copy(), []
        names = list(groups)
        base = np.array([plain[groups[g]].sum() for g in names])
        size = np.array([rows[groups[g]].sum() for g in names])
        mass = base.sum()
        temp = base ** self.alpha
        fixed = np.full(len(names), np.nan)

        def expand(share_g: np.ndarray) -> np.ndarray:
            p = plain.copy()
            for g, s in zip(names, share_g):
                idx = groups[g]
                p[idx] = s * plain[idx] / plain[idx].sum()
            return p

        for _ in range(len(names) + 1):
            free = np.isnan(fixed)
            rest = mass - np.nansum(fixed)
            share = np.where(free, rest * temp / max(temp[free].sum(), 1e-300), fixed)
            if self.max_epochs is None or budget is None:
                break
            p = expand(share)
            samples = budget / float((p * cost).sum())
            cap = np.maximum(self.max_epochs * size / samples, base)
            over = free & (share > cap * (1 + 1e-9))
            if not over.any():
                break
            fixed[over] = cap[over]
        p = expand(share)
        p /= p.sum()
        samples = None if budget is None else budget / float((p * cost).sum())
        report = [{
            "task": g, "sources": len(groups[g]), "rows": int(size[k]), "plain": float(base[k]),
            "share": float(p[groups[g]].sum()),
            "epochs": None if samples is None else float(p[groups[g]].sum() * samples / max(size[k], 1)),
        } for k, g in enumerate(names)]
        return p * total, report

    def describe(self, report: list[dict]) -> str:
        cap = "no epoch cap" if self.max_epochs is None else f"≤ {self.max_epochs:g} epochs"
        parts = [f"{r['task']} {r['plain']:.2%}→{r['share']:.2%}"
                 + ("" if r["epochs"] is None else f" ({r['epochs']:.1f} ep)")
                 for r in sorted(report, key=lambda r: -r["share"])]
        return f"language_balance alpha={self.alpha:g}, {cap}: " + ", ".join(parts)


def run_budget(cfg, world_size: int, grad_accum: int | None = None, steps: int | None = None) -> float | None:
    """Packed cost of the whole planned run; None without a train section (no epoch cap)."""
    train = cfg.get("train")
    if steps is None and (train is None or train.get("max_steps") is None):
        return None
    steps = int(steps if steps is not None else train["max_steps"])
    accum = int(grad_accum if grad_accum is not None else (train.get("grad_accum") or 1))
    return float(steps) * accum * int(world_size) * int(cfg["data"]["loader"]["max_tokens"])
