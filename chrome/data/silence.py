"""Silence padding: speech clips stretched into the spare rows of their packed batch.

Speech clips are cut to their own length, so a short caption always fills its window and the model
learns to drag the words over it. When a batch closes, the rows it could not fill (the next sample did
not fit) go to some of its speech clips as whole seconds of silence: their window grows, the speech
itself keeps its natural pace, and the rest is zeros before, after or around it. No real sample is
displaced; the cost is at most the batch's spare rows.

Off (``data.loader.silence_pad.enabled: false``) the sampler emits the same batches as before and the
dataset takes the same path. The extra seconds come from a hash of (seed, rank, batch id), not from the
sampler's random stream, so switching it on or off never changes which rows are drawn, and a resumed run
replays it. Validation and preview never pad.
"""

from __future__ import annotations

import fnmatch
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

PAD_TAG = 0x51_1E_7C_E5


@dataclass(frozen=True)
class SilencePad:
    eligible: frozenset[int]
    rows_per_second: int
    max_seconds: int
    max_fill: float = 1.0
    min_extra_seconds: int = 1
    max_extra_seconds: int = 3
    max_ratio: float = 1.6
    placement: tuple[float, float, float] = (0.7, 0.15, 0.15)
    onset_seconds: tuple[float, float] = (0.0, 0.7)
    fade_ms: float = 20.0

    @classmethod
    def from_config(cls, cfg, tasks: Sequence[str], rows_per_second: int) -> "SilencePad | None":
        """None unless ``data.loader.silence_pad.enabled``; ``tasks`` is the task of every train stream."""
        pad = cfg["data"]["loader"].get("silence_pad") or {}
        if not pad.get("enabled"):
            return None
        patterns = [str(p) for p in pad.get("tasks") or ["speech_*"]]
        eligible = frozenset(i for i, task in enumerate(tasks) if any(fnmatch.fnmatch(task, p) for p in patterns))
        place = pad.get("placement") or {}
        onset = pad.get("onset_seconds") or [0.0, 0.7]
        spec = cls(
            eligible=eligible,
            rows_per_second=int(rows_per_second),
            max_seconds=int(cfg["audio"]["max_seconds"]),
            max_fill=float(pad.get("max_fill", 1.0)),
            min_extra_seconds=int(pad.get("min_extra_seconds", 1)),
            max_extra_seconds=int(pad.get("max_extra_seconds", 3)),
            max_ratio=float(pad.get("max_ratio", 1.6)),
            placement=(float(place.get("after", 0.7)), float(place.get("before", 0.15)), float(place.get("both", 0.15))),
            onset_seconds=(float(onset[0]), float(onset[1])),
            fade_ms=float(pad.get("fade_ms", 20.0)),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not 0.0 <= self.max_fill <= 1.0:
            raise ValueError(f"silence_pad.max_fill must be in [0, 1], got {self.max_fill}")
        if not 1 <= self.min_extra_seconds <= self.max_extra_seconds:
            raise ValueError("silence_pad needs 1 <= min_extra_seconds <= max_extra_seconds")
        if self.max_ratio <= 1.0:
            raise ValueError(f"silence_pad.max_ratio must be > 1, got {self.max_ratio}")
        if min(self.placement) < 0 or sum(self.placement) <= 0:
            raise ValueError(f"silence_pad.placement weights must be >= 0 with a positive sum: {self.placement}")
        lo, hi = self.onset_seconds
        if not 0.0 <= lo <= hi:
            raise ValueError(f"silence_pad.onset_seconds must be 0 <= lo <= hi, got {self.onset_seconds}")

    def cap(self, seconds: int) -> int:
        """Most extra seconds a clip of ``seconds`` may take (0: none)."""
        cap = min(self.max_extra_seconds, int(seconds * (self.max_ratio - 1.0) + 1e-6), self.max_seconds - seconds)
        return cap if cap >= self.min_extra_seconds else 0

    def extras(self, batch: Sequence[tuple[int, int, int]], seconds: Sequence[np.ndarray | None], spare: int,
               seed: int) -> list[int]:
        """Extra seconds per item of a closed batch, within ``max_fill`` of its ``spare`` rows."""
        extra = [0] * len(batch)
        budget = int(spare * self.max_fill) // self.rows_per_second
        if budget < self.min_extra_seconds:
            return extra
        candidates = [i for i, (source, _, _) in enumerate(batch) if source in self.eligible]
        rng = random.Random(seed)
        rng.shuffle(candidates)
        for i in candidates:
            source, row, _ = batch[i]
            cap = min(self.cap(int(seconds[source][row])), budget)
            if cap < self.min_extra_seconds:
                continue
            extra[i] = rng.randint(self.min_extra_seconds, cap)
            budget -= extra[i]
            if budget < self.min_extra_seconds:
                break
        return extra

    def place(self, waveform: torch.Tensor, valid: int, extra_seconds: int, sample_rate: int,
              rng: random.Random) -> tuple[torch.Tensor, int]:
        """The clip's ``valid`` samples inside a window ``extra_seconds`` longer; (wave, lead samples)."""
        target = waveform.shape[-1] + extra_seconds * sample_rate
        speech = waveform[:, :valid].clone()
        silence = target - valid
        mode = rng.choices(("after", "before", "both"), weights=self.placement)[0]
        onset = int(rng.uniform(*self.onset_seconds) * sample_rate)
        if mode == "after":
            lead = min(onset, silence)
        elif mode == "before":
            lead = max(0, silence - onset)
        else:
            lead = rng.randint(0, silence)
        fade = min(int(self.fade_ms * sample_rate / 1000), valid // 4)
        if fade > 0:
            ramp = torch.linspace(0.0, 1.0, fade, dtype=speech.dtype)
            if lead > 0:
                speech[:, :fade] *= ramp
            if silence - lead > 0:
                speech[:, -fade:] *= ramp.flip(0)
        out = torch.zeros(waveform.shape[0], target, dtype=waveform.dtype)
        out[:, lead:lead + valid] = speech
        return out, lead


def describe(pad: SilencePad | None, names: Sequence[str]) -> str:
    if pad is None:
        return "silence_pad off"
    return (f"silence_pad on for {len(pad.eligible)}/{len(names)} sources (+{pad.min_extra_seconds}–"
            f"{pad.max_extra_seconds} s, ≤ ×{pad.max_ratio}, ≤ {pad.max_fill:.0%} of spare rows, "
            f"after/before/both {'/'.join(f'{w:g}' for w in pad.placement)})")
