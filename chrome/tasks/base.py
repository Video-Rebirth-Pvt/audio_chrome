"""A task is one training objective over one model: its data, how conditions are produced, the loss, and what
a preview renders. The trainer (DDP / FSDP, optimizer, EMA, checkpoints, logging) is shared by every task;
T2A is the first, S4 distillation and S5 video register their own.

    task = build_task(cfg, device, log)                # cfg["task"]["kind"], default t2a
    with torch.device("meta"):
        model = task.build_model()                     # the parallel strategy materialises and shards it
    task.init_model(model)                             # after materialisation
    loader, sampler = task.train_loader(rank, world, state)
    inputs = task.prepare(batch)                       # no-grad conditioning
    out = task.loss(model, inputs, generators)         # LossOutput

Collective rule: under FSDP every rank must run the same number of model forwards. The trainer pads
validation and preview so ranks match; a task's ``loss`` and ``preview`` must run a fixed number of forwards
for a given input count.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_TASKS: dict[str, type["Task"]] = {}


def register_task(name: str):
    def wrap(cls):
        _TASKS[name] = cls
        return cls

    return wrap


@dataclass
class Generators:
    """Seeded per (step, micro-batch, rank) by the trainer, so a resumed run replays the same σ and ε."""

    cpu: torch.Generator
    device: torch.Generator


@dataclass
class LossOutput:
    loss: torch.Tensor
    per_sample: torch.Tensor  # [N] detached, NaN where a sample has no target rows
    groups: torch.Tensor  # [N] index into Task.group_names (per-source metrics)


@dataclass
class Prepared:
    inputs: Any
    num_samples: int
    # Packed rows used / budget, seconds spent in each conditioning stage.
    fill: float = 0.0
    timing: dict[str, float] = field(default_factory=dict)


class Task(ABC):
    group_names: list[str]

    def __init__(self, cfg: Mapping, device: torch.device, log=print):
        self.cfg, self.device, self.log = cfg, torch.device(device), log

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Called under ``torch.device("meta")``; freeze parameters here (before sharding)."""

    @abstractmethod
    def init_model(self, model: nn.Module) -> None:
        """Initialise every parameter and buffer of the materialised (possibly sharded) model."""

    @abstractmethod
    def train_loader(self, rank: int, world_size: int, state: dict | None):
        """(DataLoader, sampler with ``state_dict(batch_id)``)."""

    @abstractmethod
    def val_loader(self, rank: int, world_size: int):
        """DataLoader over fixed holdout batches; ``batch["batch_id"]`` identifies a batch."""

    @abstractmethod
    def prepare(self, batch: dict) -> Prepared: ...

    @abstractmethod
    def loss(self, model: nn.Module, inputs: Any, generators: Generators) -> LossOutput: ...

    def grad_groups(self, model: nn.Module) -> dict[str, list[nn.Parameter]]:
        """Named parameter subsets whose gradient norm is logged next to the total."""
        return {}

    def preview_items(self) -> list[Any]:
        return []

    def preview(self, model: nn.Module, items: list[Any], out_dir: Path, keep: int) -> None:
        """Render ``items`` and save the first ``keep`` (the rest is padding that keeps ranks in step)."""


def build_task(cfg: Mapping, device, log=print) -> Task:
    import chrome.tasks.t2a  # noqa: F401  (registration)

    kind = str((cfg.get("task") or {}).get("kind", "t2a"))
    if kind not in _TASKS:
        raise KeyError(f"unknown task {kind!r}; known: {sorted(_TASKS)}")
    return _TASKS[kind](cfg, device, log)
