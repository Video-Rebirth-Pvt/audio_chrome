"""Run directory layout and checkpoints.

    <family>/                                  outputs/train/<run_name>  (t2a_10b stamps children)
    <run_dir>/                                 family itself (legacy) or <family>/<YYYYMMDD_HHMMSS>/
    <run_dir>/config.yaml                      resolved config of the run (inference rebuilds from it)
    <run_dir>/log.jsonl                        one JSON line per train / val event
    <run_dir>/tb/                              TensorBoard scalars + preview audio (train.tensorboard)
    <run_dir>/checkpoints/latest               name of the newest complete checkpoint
    <run_dir>/checkpoints/step_0001000/
        model.safetensors / ema.safetensors    full fp32 weights (10B default; infer reads ema)
        optimizer.pt                           full Adam state (same snapshot; needed to resume)
        trainer.pt                             step, LR schedule, DiT config, strategy; written last
        data_rank{r}.json                      sampler state of every rank (resume the data stream)
        state/                                 optional DCP shards (train.save_sharded=true, smoke / 1B)
    <run_dir>/samples/step_0001000/*.wav       previews

``state/`` is written collectively, sharded as trained (FSDP) or deduplicated (DDP), and loads into any
world size or strategy. ``ema`` is a complete model state dict (EMA parameters plus the model's buffers).
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn

from chrome.dit.config import DiTConfig
from chrome.dit.model import ChromeDiT

LATEST = "latest"
STATE = "state"
TRAINER = "trainer.pt"
OPTIMIZER = "optimizer.pt"
MODEL_FILE = "model.safetensors"
EMA_FILE = "ema.safetensors"
STAMP_FMT = "%Y%m%d_%H%M%S"
STAMP_RE = re.compile(r"^\d{8}_\d{6}$")


def dit_config(cfg) -> DiTConfig:
    """``dit`` section; ``text_dim: null`` takes the width of the configured text encoder."""
    from chrome.text.base import text_dim

    section = dict(cfg["dit"])
    if section.get("text_dim") is None:
        section["text_dim"] = text_dim(cfg["text"])
    return DiTConfig.from_config(section)


def build_dit(cfg) -> ChromeDiT:
    return ChromeDiT(dit_config(cfg))


def run_stamp(when: float | None = None) -> str:
    return time.strftime(STAMP_FMT, time.localtime(when if when is not None else time.time()))


def is_run_dir(path: str | Path) -> bool:
    path = Path(path)
    return has_checkpoint(path) or (path / "config.yaml").is_file()


def stamped_runs(family: str | Path) -> list[Path]:
    family = Path(family)
    if not family.is_dir():
        return []
    found = [p for p in family.iterdir() if p.is_dir() and STAMP_RE.match(p.name) and is_run_dir(p)]
    return sorted(found, key=lambda p: p.name)


def resolve_run_dir(output_dir: str | Path, resume="auto", stamp: bool = False, now: str | None = None) -> Path:
    """Family dir ``outputs/train/<run_name>`` → a concrete run.

    ``stamp``: new experiments go to ``<family>/<YYYYMMDD_HHMMSS>/``. ``resume=auto`` continues the newest
    stamped child, or the family itself if it is a legacy flat run. ``resume=null`` always opens a new stamp.
    """
    family = Path(output_dir)
    if STAMP_RE.match(family.name):
        return family
    resume_on = resume not in (None, "", "none", False)
    auto = str(resume) == "auto" if resume_on else False
    if not stamp:
        return family
    if auto:
        children = stamped_runs(family)
        if children:
            return children[-1]
        if has_checkpoint(family):
            return family
        return family / (now or run_stamp())
    if not resume_on:
        return family / (now or run_stamp())
    return family


def find_run_dir(path: str | Path) -> Path:
    """Infer / export: a concrete run, or the newest stamped child under a family dir."""
    path = Path(path)
    children = stamped_runs(path)
    if children:
        return children[-1]
    if is_run_dir(path):
        return path
    raise FileNotFoundError(f"no run under {path}")


def checkpoint_root(run_dir: str | Path) -> Path:
    return Path(run_dir) / "checkpoints"


def step_dir(run_dir: str | Path, step: int) -> Path:
    return checkpoint_root(run_dir) / f"step_{step:07d}"


def resolve_checkpoint(run_dir: str | Path, which: str | int = LATEST) -> Path:
    """``latest``, a step number, a step directory name, or a path."""
    root = checkpoint_root(run_dir)
    if isinstance(which, int) or str(which).isdigit():
        path = step_dir(run_dir, int(which))
    elif str(which) == LATEST:
        marker = root / LATEST
        if not marker.is_file():
            raise FileNotFoundError(f"no checkpoint under {root}")
        path = root / marker.read_text().strip()
    else:
        path = Path(which) if Path(which).is_dir() else root / str(which)
    if not (path / TRAINER).is_file():
        raise FileNotFoundError(f"incomplete checkpoint {path}")
    return path


def has_checkpoint(run_dir: str | Path) -> bool:
    return (checkpoint_root(run_dir) / LATEST).is_file()


def write_latest(run_dir: str | Path, step: int, keep: int) -> None:
    root = checkpoint_root(run_dir)
    tmp = root / f".{LATEST}.tmp"
    tmp.write_text(step_dir(run_dir, step).name + "\n")
    tmp.replace(root / LATEST)
    if keep > 0:
        steps = sorted(p for p in root.glob("step_*") if p.is_dir())
        for old in steps[:-keep]:
            shutil.rmtree(old, ignore_errors=True)


def save_data_state(path: Path, rank: int, state: dict) -> None:
    (path / f"data_rank{rank}.json").write_text(json.dumps(state))


def load_data_state(path: Path, rank: int) -> dict | None:
    file = path / f"data_rank{rank}.json"
    return json.loads(file.read_text()) if file.is_file() else None


# ---------------------------------------------------------------- sharded state (collective)


def _is_sharded(model: nn.Module) -> bool:
    return any(hasattr(p, "device_mesh") for p in model.parameters())


def stored_keys(ckpt: Path) -> set[str]:
    """Top-level entries (model / ema / optimizer) present in ``state/``."""
    from torch.distributed.checkpoint import FileSystemReader

    if not (ckpt / STATE / ".metadata").is_file():
        return set()
    keys = FileSystemReader(str(ckpt / STATE)).read_metadata().state_dict_metadata
    return {k.split(".", 1)[0] for k in keys}


def _rank0() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def _full_options():
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    return StateDictOptions(full_state_dict=True, cpu_offload=True)


def has_full(ckpt: Path) -> bool:
    ckpt = Path(ckpt)
    return (ckpt / MODEL_FILE).is_file() or (ckpt / EMA_FILE).is_file()


def save_full(ckpt: Path, model: nn.Module, optimizer=None, ema=None) -> None:
    """Every rank calls this. Rank 0 writes one unsharded snapshot: model + ema safetensors and optimizer.pt."""
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict

    options = _full_options()
    model_state = get_model_state_dict(model, options=options)
    opt_state = get_optimizer_state_dict(model, optimizer, options=options) if optimizer is not None else None
    ema_state = None
    if ema is not None:
        with ema.swapped():
            ema_state = get_model_state_dict(model, options=options)
    if not _rank0():
        return
    ckpt = Path(ckpt)
    save_file({k: v.float().contiguous() for k, v in model_state.items()}, str(ckpt / MODEL_FILE))
    if ema_state is not None:
        save_file({k: v.float().contiguous() for k, v in ema_state.items()}, str(ckpt / EMA_FILE))
    if opt_state is not None:
        torch.save(opt_state, ckpt / OPTIMIZER)


def load_full(ckpt: Path, model: nn.Module, optimizer=None, ema=None) -> None:
    """Restore a ``save_full`` snapshot into the current sharding (or a single process)."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict

    ckpt = Path(ckpt)
    options = StateDictOptions(full_state_dict=True)
    if ema is not None and (ckpt / EMA_FILE).is_file():
        load_weights(model, ckpt, "ema")
        ema.copy_from_model()
    if (ckpt / MODEL_FILE).is_file():
        load_weights(model, ckpt, "model")
    elif ema is None:
        load_weights(model, ckpt, "ema")
    if optimizer is not None and (ckpt / OPTIMIZER).is_file():
        state = torch.load(ckpt / OPTIMIZER, map_location="cpu", weights_only=False)
        set_optimizer_state_dict(model, optimizer, state, options=options)


def save_state(ckpt: Path, model: nn.Module, optimizer=None, ema=None) -> None:
    """Every rank calls this. ``model`` is the unwrapped module (FSDP-sharded or plain)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict

    model_state = get_model_state_dict(model)
    state = {"model": model_state}
    if optimizer is not None:
        state["optimizer"] = get_optimizer_state_dict(model, optimizer)
    if ema is not None:
        state["ema"] = {**model_state, **ema.state_dict()}
    dcp.save(state, checkpoint_id=str(ckpt / STATE))


def load_state(ckpt: Path, model: nn.Module, optimizer=None, ema=None) -> None:
    """Every rank calls this; restores into the current sharding whatever the saving world size was."""
    ckpt = Path(ckpt)
    if not (ckpt / STATE / ".metadata").is_file():
        if has_full(ckpt):
            load_full(ckpt, model, optimizer, ema)
            return
        raise FileNotFoundError(f"no state/ or full snapshot in {ckpt}")
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        get_optimizer_state_dict,
        set_model_state_dict,
        set_optimizer_state_dict,
    )

    present = stored_keys(ckpt)
    state = {"model": get_model_state_dict(model)}
    if optimizer is not None:
        state["optimizer"] = get_optimizer_state_dict(model, optimizer)
    if ema is not None and "ema" in present:
        state["ema"] = ema.state_dict()
    dcp.load(state, checkpoint_id=str(ckpt / STATE))
    set_model_state_dict(model, state["model"])
    if optimizer is not None:
        set_optimizer_state_dict(model, optimizer, state["optimizer"])
    if "ema" in state:
        ema.load_state_dict(state["ema"])


def load_weights(model: nn.Module, ckpt: Path, weights: str = "ema") -> str:
    """Load ``ema`` (falls back to ``model`` when the run kept no EMA) or ``model`` weights into a materialised
    model, plain or sharded; from ``<weights>.safetensors`` if exported, else from ``state/``. Returns which.
    Sharded models need every rank to call this."""
    ckpt = Path(ckpt)
    present = stored_keys(ckpt)
    for name in dict.fromkeys((weights, "model")):
        file = ckpt / f"{name}.safetensors"
        if file.is_file():
            from safetensors.torch import load_file

            state = load_file(str(file))
            if _is_sharded(model):
                from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

                set_model_state_dict(model, state, options=StateDictOptions(full_state_dict=True))
            else:
                model.load_state_dict(state, strict=True)
            return name
        if name in present:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

            state = {name: get_model_state_dict(model)}
            dcp.load(state, checkpoint_id=str(ckpt / STATE))
            set_model_state_dict(model, state[name])
            return name
    raise FileNotFoundError(f"no {weights} or model weights in {ckpt}")


def export_weights(ckpt: Path, model: nn.Module, weights: str = "ema", out: Path | None = None) -> Path:
    """Single-process: ``state/`` → one fp32 safetensors file (default ``<ckpt>/<weights>.safetensors``)."""
    from safetensors.torch import save_file

    used = load_weights(model, ckpt, weights)
    out = Path(out) if out else Path(ckpt) / f"{used}.safetensors"
    save_file({k: v.detach().to("cpu", torch.float32).contiguous() for k, v in model.state_dict().items()}, str(out))
    return out
