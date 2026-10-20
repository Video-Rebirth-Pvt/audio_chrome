"""YAML configs with ``_base_`` inheritance, ``${...}`` interpolation and dotlist overrides."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf


def _load_with_bases(path: Path, seen: tuple[Path, ...] = ()) -> DictConfig:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"_base_ cycle: {' -> '.join(str(p) for p in (*seen, path))}")
    raw = OmegaConf.load(path)
    bases = raw.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    elif not isinstance(bases, (list, ListConfig)):
        raise ValueError(f"_base_ in {path} must be a path or a list of paths")
    merged = OmegaConf.create()
    for base in bases:
        merged = OmegaConf.merge(merged, _load_with_bases(path.parent / str(base), (*seen, path)))
    return OmegaConf.merge(merged, raw)


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> DictConfig:
    """Bases merge in order, then the file itself, then ``key=value`` overrides."""
    cfg = _load_with_bases(Path(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg
