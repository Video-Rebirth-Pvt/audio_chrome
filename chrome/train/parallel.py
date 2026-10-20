"""Data-parallel strategies behind one interface: replicate (DDP) or shard (FSDP2, optionally hybrid).

    parallel = build_parallel(cfg["train"]["parallel"], env)
    module = parallel.setup(model, init_fn)          # model built on meta → materialised, initialised, wrapped
    with parallel.accumulate(module, sync=last_micro):
        loss.backward()
    grad_norm = parallel.clip_grad_norm(params, 1.0)

``ddp``: fp32 weights replicated on every rank, bf16 autocast; fits up to ~1B.
``fsdp``: FSDP2 ``fully_shard`` on every unit of ``model.shard_units()`` (DiT and refiner blocks) and the
root. fp32 master weights, gradients and Adam states are sharded; forward / backward run on bf16 copies,
gradients are reduced in fp32. ``shard_size`` < world gives HSDP: shard within groups of that many ranks
(one node), replicate across groups (multi-node). 10B on 8×H200: ~22 GB of states per rank.

Model inputs are never cast by FSDP (``cast_forward_inputs=False``): RoPE positions (fp64) and timesteps
must keep their precision; the model casts what it projects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn

from chrome.train.dist import DistEnv

_PARALLEL: dict[str, type["Parallel"]] = {}


def register_parallel(name: str):
    def wrap(cls):
        _PARALLEL[name] = cls
        return cls

    return wrap


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if hasattr(t, "to_local") else t


class Parallel(ABC):
    kind: str

    def __init__(self, env: DistEnv, seed: int = 0):
        self.env, self.seed = env, int(seed)

    @abstractmethod
    def setup(self, model: nn.Module, init_fn: Callable[[nn.Module], None]) -> nn.Module:
        """``model`` lives on the meta device. Returns the module to call for training; ``model`` itself keeps
        the parameter names used by checkpoints (FSDP shards it in place, DDP wraps it)."""

    @abstractmethod
    def accumulate(self, module: nn.Module, sync: bool):
        """Context for one micro-batch backward; gradients are reduced across ranks only when ``sync``."""

    def _norm_group(self):
        return None

    @torch.no_grad()
    def grad_sq_norm(self, params: Iterable[nn.Parameter]) -> torch.Tensor:
        """Σ‖g‖² over ``params`` across all shards, fp32 on the device (same value on every rank)."""
        grads = [_local(p.grad) for p in params if p.grad is not None]
        total = torch.zeros((), device=self.env.device, dtype=torch.float32)
        if grads:
            total = torch.stack([g.float().pow(2).sum() for g in grads]).sum()
        group = self._norm_group()
        if group is not None:
            dist.all_reduce(total, group=group)
        return total

    @torch.no_grad()
    def clip_grad_norm(self, params: list[nn.Parameter], max_norm: float) -> torch.Tensor:
        norm = self.grad_sq_norm(params).sqrt()
        coef = (max_norm / (norm + 1e-6)).clamp(max=1.0)
        grads = [_local(p.grad) for p in params if p.grad is not None]
        if grads and torch.isfinite(norm):
            torch._foreach_mul_(grads, coef.to(grads[0].dtype))
        return norm

    def reshard(self, module: nn.Module) -> None:
        """Drop gathered weights left by no-grad forwards (FSDP units that do not reshard after forward), so
        the next forward gathers the current shards. Called around validation, preview and the EMA swap."""

    def describe(self) -> str:
        return self.kind


@register_parallel("ddp")
class DDPParallel(Parallel):
    kind = "ddp"

    def setup(self, model, init_fn):
        from torch.nn.parallel import DistributedDataParallel

        model.to_empty(device=self.env.device)
        init_fn(model)
        if not self.env.enabled:
            return model
        # Broadcasts rank 0's parameters and buffers, so per-rank init seeds do not matter.
        ids = [self.env.device.index] if self.env.device.type == "cuda" else None
        # Buffers (RoPE table, text stats) never change after init; skip the per-forward broadcast.
        return DistributedDataParallel(model, device_ids=ids, broadcast_buffers=False)

    def accumulate(self, module, sync):
        if self.env.enabled and not sync:
            return module.no_sync()
        return nullcontext()


@register_parallel("fsdp")
class FSDPParallel(Parallel):
    kind = "fsdp"

    def __init__(
        self,
        env: DistEnv,
        seed: int = 0,
        shard_size: int | None = None,
        reshard_after_forward: bool = True,
        param_dtype: str = "bfloat16",
        reduce_dtype: str = "float32",
        # True: reduce-scatter after every micro-batch (sharded grads, more traffic). False: keep unsharded
        # grads until the last micro-batch (one reduce, full-size grads in memory).
        sync_every_micro: bool = True,
    ):
        super().__init__(env, seed)
        world = env.world_size
        self.shard_size = int(shard_size or world)
        if world % self.shard_size:
            raise ValueError(f"world size {world} is not a multiple of shard_size {self.shard_size}")
        self.reshard_after_forward = bool(reshard_after_forward)
        self.param_dtype, self.reduce_dtype = getattr(torch, param_dtype), getattr(torch, reduce_dtype)
        self.sync_every_micro = bool(sync_every_micro)
        self.mesh = None

    def _build_mesh(self):
        from torch.distributed.device_mesh import init_device_mesh

        device = self.env.device.type
        replicate = self.env.world_size // self.shard_size
        if replicate == 1:
            return init_device_mesh(device, (self.shard_size,), mesh_dim_names=("shard",))
        return init_device_mesh(device, (replicate, self.shard_size), mesh_dim_names=("replicate", "shard"))

    def _norm_group(self):
        # Gradients are identical across replicas after reduction; sum the squares over one shard group.
        return self.mesh["shard"].get_group() if self.mesh is not None and self.env.enabled else None

    def setup(self, model, init_fn):
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        if not self.env.enabled:
            raise RuntimeError("parallel.kind=fsdp needs torchrun with more than one rank; use ddp for one GPU")
        self.mesh = self._build_mesh()
        mp = MixedPrecisionPolicy(
            param_dtype=self.param_dtype, reduce_dtype=self.reduce_dtype, cast_forward_inputs=False
        )
        units = model.shard_units() if hasattr(model, "shard_units") else []
        for unit in units:
            fully_shard(unit, mesh=self.mesh, mp_policy=mp, reshard_after_forward=self.reshard_after_forward)
        # The root also reshards after forward: a no-grad forward (validation, preview) would otherwise leave
        # a gathered copy that later forwards reuse, hiding the EMA swap and the next optimizer step.
        fully_shard(model, mesh=self.mesh, mp_policy=mp, reshard_after_forward=True)
        model.to_empty(device=self.env.device)
        # DTensor random init draws from one stream synced across ranks, so every shard gets its own slice
        # of the same full-tensor init. Seed identically; the trainer re-seeds per rank afterwards.
        torch.manual_seed(self.seed)
        init_fn(model)
        return model

    def accumulate(self, module, sync):
        if self.sync_every_micro:
            return nullcontext()
        return self._gradient_sync(module, sync)

    def reshard(self, module):
        from torch.distributed.fsdp import FSDPModule

        for m in module.modules():
            if isinstance(m, FSDPModule):
                m.reshard()

    @contextmanager
    def _gradient_sync(self, module, sync: bool):
        module.set_requires_gradient_sync(sync)
        try:
            yield
        finally:
            module.set_requires_gradient_sync(True)

    def describe(self) -> str:
        replicate = self.env.world_size // self.shard_size
        mode = "fsdp" if replicate == 1 else f"hsdp {replicate}×{self.shard_size}"
        return f"{mode}, params {str(self.param_dtype)[6:]}, reduce {str(self.reduce_dtype)[6:]}"


def build_parallel(cfg: Mapping | None, env: DistEnv, seed: int = 0) -> Parallel:
    """From ``train.parallel``: ``kind`` (ddp | fsdp) plus the class arguments."""
    cfg = dict(cfg or {})
    kind = str(cfg.pop("kind", "ddp"))
    if kind not in _PARALLEL:
        raise KeyError(f"unknown parallel strategy {kind!r}; known: {sorted(_PARALLEL)}")
    return _PARALLEL[kind](env, seed, **cfg)


class EMA:
    """fp32 shadow of every parameter in the model's own layout (sharded under FSDP, so no rank holds a full
    copy). ``swapped`` puts the shadow into the model for previews and restores the live weights after."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        named = list(model.named_parameters())
        self.names = [n for n, _ in named]
        self.params = [p for _, p in named]
        with torch.no_grad():
            self.shadow = [_local(p.detach()).float().clone() for p in self.params]

    @torch.no_grad()
    def update(self, step: int) -> None:
        decay = min(self.decay, (1 + step) / (10 + step))
        live = [_local(p.detach()).float() for p in self.params]
        torch._foreach_lerp_(self.shadow, live, 1.0 - decay)

    def copy_from_model(self) -> None:
        """Pull the current (possibly sharded) model weights into the shadow. Used when loading a full EMA file."""
        with torch.no_grad():
            for param, shadow in zip(self.params, self.shadow):
                shadow.copy_(_local(param.detach()).float())

    def state_dict(self) -> dict[str, torch.Tensor]:
        """DTensor views of the shards under FSDP, so torch.distributed.checkpoint saves them sharded."""
        out = {}
        for name, param, shadow in zip(self.names, self.params, self.shadow):
            if hasattr(param, "to_local"):
                from torch.distributed.tensor import DTensor

                shadow = DTensor.from_local(shadow, param.device_mesh, param.placements, shape=param.shape,
                                            stride=param.stride(), run_check=False)
            out[name] = shadow
        return out

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        for name, shadow in zip(self.names, self.shadow):
            shadow.copy_(_local(state[name]))

    @contextmanager
    def swapped(self):
        live = [_local(p.detach()).clone() for p in self.params]
        with torch.no_grad():
            for p, s in zip(self.params, self.shadow):
                _local(p.data).copy_(s)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, s in zip(self.params, live):
                    _local(p.data).copy_(s)
