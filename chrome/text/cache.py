"""Caption embedding cache in front of any text encoder: memory LRU, then disk, then the encoder.

The encoder itself is built on the first miss, so a run whose captions are all cached (see
``bash scripts/text/cache_text.sh``) never loads Qwen. Entries are keyed by the encoder fingerprint and the
token ids, one safetensors file each under ``<dir>/<fingerprint hash>/``; writes are atomic, so ranks can
share the directory.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from chrome.text.base import TextEncoder


def _digest(*parts: bytes, size: int = 16) -> str:
    h = hashlib.blake2b(digest_size=size)
    for part in parts:
        h.update(part)
    return h.hexdigest()


class CachedTextEncoder(TextEncoder):
    def __init__(
        self,
        factory: Callable[[], TextEncoder],
        fingerprint: str,
        dim: int,
        cache_dir: str | Path | None = None,
        memory_items: int = 0,
        device: str | torch.device = "cpu",
    ):
        self._factory = factory
        self._encoder: TextEncoder | None = None
        self._fingerprint = fingerprint
        self.dim = int(dim)
        self.device = torch.device(device)
        self.dir = None if not cache_dir else Path(cache_dir) / _digest(fingerprint.encode(), size=6)
        self.memory_items = int(memory_items)
        self._memory: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.hits = {"memory": 0, "disk": 0, "encoded": 0}
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "fingerprint.txt").write_text(fingerprint + "\n")

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def encoder(self) -> TextEncoder:
        if self._encoder is None:
            self._encoder = self._factory()
            if self._encoder.dim != self.dim:
                raise ValueError(f"encoder dim {self._encoder.dim} != described {self.dim}")
        return self._encoder

    @property
    def loaded(self) -> bool:
        return self._encoder is not None

    def key(self, ids: torch.Tensor) -> str:
        return _digest(self._fingerprint.encode(), ids.to(torch.int32).cpu().numpy().tobytes())

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.safetensors"

    def _remember(self, key: str, value: torch.Tensor) -> None:
        if self.memory_items <= 0:
            return
        self._memory[key] = value
        self._memory.move_to_end(key)
        while len(self._memory) > self.memory_items:
            self._memory.popitem(last=False)

    def _load(self, key: str) -> torch.Tensor | None:
        if key in self._memory:
            self._memory.move_to_end(key)
            self.hits["memory"] += 1
            return self._memory[key]
        if self.dir is None:
            return None
        path = self._path(key)
        if not path.is_file():
            return None
        from safetensors.torch import load_file

        try:
            value = load_file(str(path))["hidden"]
        except Exception:
            return None
        self.hits["disk"] += 1
        self._remember(key, value)
        return value

    def _store(self, key: str, value: torch.Tensor) -> None:
        value = value.detach().to("cpu").contiguous()
        self._remember(key, value)
        if self.dir is None:
            return
        from safetensors.torch import save_file

        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        save_file({"hidden": value}, str(tmp))
        os.replace(tmp, path)

    def contains(self, ids: torch.Tensor) -> bool:
        key = self.key(ids)
        return key in self._memory or (self.dir is not None and self._path(key).is_file())

    def encode(self, ids: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        keys = [self.key(i) for i in ids]
        out: list[torch.Tensor | None] = [self._load(k) for k in keys]
        missing = [i for i, value in enumerate(out) if value is None]
        if missing:
            unique: dict[str, int] = {}
            for i in missing:
                unique.setdefault(keys[i], i)
            encoded = self.encoder.encode([ids[i] for i in unique.values()])
            self.hits["encoded"] += len(encoded)
            fresh = dict(zip(unique, encoded))
            for key, value in fresh.items():
                self._store(key, value)
            for i in missing:
                out[i] = fresh[keys[i]]
        return [value.to(self.device, non_blocking=True) for value in out]
