"""Which segments a sample contributes to the packed sequence. Training and inference both build sequences here.

T2A is ``[text | audio L | audio R]`` with the ears on the w positions of a ``StereoW`` policy. S5 registers a
layout with video segments (keyframes, targets); the packer and the DiT stay unchanged.

    layout = build_layout(cfg["layout"])
    seq = layout.pack(text_lens=[12, 40], frames=[400, 160], channels=2)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from chrome.dit.packing import PackedSequence, Segment, pack, t2a_segments
from chrome.dit.rope import build_stereo_w

_LAYOUTS: dict[str, type["Layout"]] = {}


def register_layout(name: str):
    def wrap(cls):
        _LAYOUTS[name] = cls
        return cls

    return wrap


class Layout(ABC):
    @abstractmethod
    def segments(self, text_len: int, frames: int, channels: int) -> list[Segment]: ...

    def pack(self, text_lens: Sequence[int], frames: Sequence[int], channels: int | Sequence[int]) -> PackedSequence:
        if isinstance(channels, int):
            channels = [channels] * len(frames)
        return pack([self.segments(int(l), int(f), int(c)) for l, f, c in zip(text_lens, frames, channels)])


@register_layout("t2a")
class T2ALayout(Layout):
    def __init__(self, stereo_w: Mapping):
        self.stereo = build_stereo_w(stereo_w)

    def segments(self, text_len, frames, channels):
        return t2a_segments(text_len, frames, self.stereo.values(channels))


def build_layout(cfg: Mapping) -> Layout:
    """From the ``layout`` section: ``kind`` (default t2a) plus the class arguments."""
    cfg = dict(cfg)
    kind = str(cfg.pop("kind", "t2a"))
    if kind not in _LAYOUTS:
        raise KeyError(f"unknown layout {kind!r}; known: {sorted(_LAYOUTS)}")
    return _LAYOUTS[kind](**cfg)
