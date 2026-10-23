"""torchrun environment: rank / world / device, and no-op fallbacks for single-process runs."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistEnv:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def init_distributed(timeout_minutes: float = 30.0) -> DistEnv:
    """Single node or multi-node alike: torchrun sets RANK / WORLD_SIZE / LOCAL_RANK. The timeout covers
    collectives that wait on one rank's slow work (a Qwen load on a cache miss). Rank-0 checkpoint
    writes wait on a file poll, not this NCCL timeout — see ``train.ckpt_timeout_minutes``."""
    from datetime import timedelta

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
            device_id=device if device.type == "cuda" else None,
            timeout=timedelta(minutes=timeout_minutes),
        )
    return DistEnv(rank, world, local, device)


def all_reduce_max(env: DistEnv, value: int) -> int:
    if not env.enabled:
        return int(value)
    t = torch.tensor([int(value)], device=env.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())


def barrier(env: DistEnv) -> None:
    if env.enabled:
        dist.barrier()


def all_reduce_sum(env: DistEnv, tensor: torch.Tensor) -> torch.Tensor:
    if env.enabled:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def cleanup(env: DistEnv) -> None:
    if env.enabled and dist.is_initialized():
        dist.destroy_process_group()
