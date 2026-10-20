"""Worker side: one sampled row → a fixed-length 32 kHz stereo clip and its caption token ids."""

from __future__ import annotations

import fcntl
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from chrome.data.audio_io import limit_peak, read_audio, read_audio_window, resample, to_stereo
from chrome.data.captions import NULL_CAPTION, caption_ids, load_caption_tokenizer
from chrome.data.index import SourceIndex, read_path_list
from chrome.spec import AudioSpec


class BadAudioList:
    """Undecodable paths, appended by any worker under a file lock and skipped from then on."""

    def __init__(self, path: str | Path, extra: list[str | Path] = ()):
        self.path = Path(path)
        self.extra = [Path(p) for p in extra]
        self._known: set[str] | None = None

    @property
    def known(self) -> set[str]:
        if self._known is None:
            self._known = set()
            for path in [self.path, *self.extra]:
                self._known |= read_path_list(path)
        return self._known

    def add(self, audio: str, reason: str) -> None:
        self.known.add(audio)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        reason = " ".join(str(reason).split())[:300]
        with self.path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(f"{audio}\t{reason}\n")
            fcntl.flock(handle, fcntl.LOCK_UN)
        print(f"[data] bad audio {audio}: {reason}", flush=True)


class T2ADataset(Dataset):
    """Indexed by ``(batch_id, source, row, sample_seed)`` from the batch samplers.

    The window start and the caption dropout come from ``sample_seed``, so a resumed run reproduces them.
    A clip that fails to decode is recorded and replaced by a random train row of the same source.
    """

    def __init__(
        self,
        indexes: list[SourceIndex],
        spec: AudioSpec,
        tokenizer_dir: str,
        bad_audio: BadAudioList,
        cfg_drop: float = 0.0,
        peak: float | None = 0.95,
        crop: str = "random",
        retries: int = 8,
    ):
        if crop not in ("random", "center", "start"):
            raise ValueError(f"crop must be random|center|start, got {crop}")
        self.indexes = indexes
        self.spec = spec
        self.tokenizer_dir = str(tokenizer_dir)
        self.bad_audio = bad_audio
        self.cfg_drop = float(cfg_drop)
        self.peak = peak
        self.crop = crop
        self.retries = int(retries)
        self._tokenizer = None

    def __len__(self) -> int:
        return sum(len(index) for index in self.indexes)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = load_caption_tokenizer(self.tokenizer_dir)
        return self._tokenizer

    def load_clip(self, path: str, duration: float, seconds: int, rng: random.Random) -> tuple[torch.Tensor, int]:
        """[2, seconds·sr] and the number of real (unpadded) samples."""
        rate = self.spec.sample_rate
        target = seconds * rate
        if duration > seconds + 1e-3:
            span = duration - seconds
            start = {"random": rng.uniform(0.0, span), "center": span / 2, "start": 0.0}[self.crop]
            waveform, file_rate = read_audio_window(path, start, seconds)
        else:
            waveform, file_rate = read_audio(path)
        waveform = resample(to_stereo(waveform), file_rate, rate)
        valid = min(waveform.shape[-1], target)
        if valid < self.spec.hop_length:
            raise RuntimeError(f"decoded only {waveform.shape[-1]} samples")
        if waveform.shape[-1] >= target:
            waveform = waveform[:, :target]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, target - waveform.shape[-1]))
        return limit_peak(waveform, self.peak).contiguous(), valid

    def __getitem__(self, item: tuple[int, int, int, int]) -> dict:
        batch_id, source, row, seed = item
        rng = random.Random(seed)
        index = self.indexes[source]
        errors = []
        for _ in range(self.retries):
            path, caption, duration = index.row(row)
            seconds = int(index.seconds[row])
            if path not in self.bad_audio.known:
                try:
                    waveform, valid = self.load_clip(path, duration, seconds, rng)
                    break
                except Exception as exc:
                    errors.append(f"{path}: {exc}")
                    self.bad_audio.add(path, str(exc))
            row = self._replacement(index, rng)
        else:
            raise RuntimeError(f"{index.name}: {self.retries} clips failed to decode; " + " | ".join(errors[:3]))
        dropped = rng.random() < self.cfg_drop
        ids = caption_ids(self.tokenizer, NULL_CAPTION if dropped else caption)
        return {
            "batch_id": batch_id,
            "wave": waveform,
            "seconds": seconds,
            "valid_samples": valid,
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "cfg_dropped": dropped,
            "source": source,
            "row": row,
            "caption": caption,
            "path": path,
        }

    @staticmethod
    def _replacement(index: SourceIndex, rng: random.Random) -> int:
        for _ in range(64):
            row = rng.randrange(len(index))
            if not index.holdout[row]:
                return row
        return row
