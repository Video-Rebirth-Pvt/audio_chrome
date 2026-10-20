"""Text encoder interface: caption token ids in, per-caption hidden states out.

Encoders are registered by name and built from the ``text`` config section, so swapping the Qwen size for the
S2 ablation (or a random table in unit tests) is a config change:

    encoder = build_text_encoder(cfg["text"], device="cuda")
    embeds = encoder.encode([ids_a, ids_b])        # [L_a, dim], [L_b, dim]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence

import torch

_REGISTRY: dict[str, Callable[..., "TextEncoder"]] = {}


def register_text_encoder(name: str):
    def wrap(cls):
        if name in _REGISTRY:
            raise ValueError(f"text encoder {name!r} registered twice")
        _REGISTRY[name] = cls
        return cls

    return wrap


class TextEncoder(ABC):
    """Maps token-id sequences to hidden states of width ``dim`` (no pooling, one row per token).

    Registered classes also provide ``describe(**kwargs) -> (fingerprint, dim)`` as a static method that
    does not load weights, so a cache can answer without building the encoder.
    """

    dim: int

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        """Identifies the weights and settings; embedding caches are keyed by it."""

    @abstractmethod
    def encode(self, ids: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Token ids (1-D long tensors, any device) → [L_i, dim] on the encoder's device."""

    def to(self, device) -> "TextEncoder":
        return self


def _resolve(cfg: Mapping) -> tuple[type, dict]:
    """Class for ``kind`` and the config keys its constructor accepts (keys left over from a base config
    written for another encoder are dropped)."""
    import inspect

    import chrome.text.qwen  # noqa: F401  (registration)
    import chrome.text.random  # noqa: F401

    kind = str(cfg["kind"])
    if kind not in _REGISTRY:
        raise KeyError(f"unknown text encoder {kind!r}; known: {sorted(_REGISTRY)}")
    cls = _REGISTRY[kind]
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self", "device", "log"}
    return cls, {str(k): v for k, v in dict(cfg).items() if k in accepted}


def build_text_encoder(cfg: Mapping, device: str | torch.device = "cpu", log=print) -> TextEncoder:
    """``cfg`` is the ``text`` section: ``kind`` picks the class, ``cache`` the optional embedding cache,
    the remaining keys are the class arguments. With a cache the encoder is only built on the first miss."""
    from chrome.text.cache import CachedTextEncoder

    cls, kwargs = _resolve(cfg)
    cache = dict(cfg.get("cache") or {})
    if not cache.get("dir") and not int(cache.get("memory_items") or 0):
        return cls(**kwargs, device=device, log=log)
    fingerprint, dim = cls.describe(**kwargs)
    return CachedTextEncoder(
        lambda: cls(**kwargs, device=device, log=log),
        fingerprint,
        dim,
        cache_dir=cache.get("dir"),
        memory_items=int(cache.get("memory_items") or 0),
        device=device,
    )


def text_dim(cfg: Mapping) -> int:
    """Hidden width of the configured encoder, without loading it."""
    cls, kwargs = _resolve(cfg)
    return cls.describe(**kwargs)[1]
