"""TensorBoard writer for a Chrome run: scalars from log.jsonl records, audio from samples/.

    <run_dir>/tb/                 SummaryWriter event files
    <run_dir>/tb/cursor.json      last imported train/val step and preview steps (idempotent sync)

Live training writes each ``_record`` / preview. ``python -m chrome.train.tensorboard <run_dir>``
imports anything not yet in the cursor, so a run that started before this module still plots.
"""

from __future__ import annotations

import io
import json
import math
import sys
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

SKIP = frozenset({"event", "step"})
CURSOR = "cursor.json"
AUDIO_LAYOUT = "stereo"


def flatten_scalars(record: dict[str, Any]) -> dict[str, float]:
    """``{event: train, loss: 1.2, loss/music: 1.1, text_cache: {memory: 3}}`` → ``train/loss``, …"""
    event = str(record.get("event") or "train")
    out: dict[str, float] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                walk(f"{prefix}/{key}" if prefix else str(key), inner)
            return
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number):
                out[prefix] = number

    for key, value in record.items():
        if key in SKIP:
            continue
        walk(f"{event}/{key}", value)
    return out


def encode_wav(waveform: torch.Tensor, sample_rate: int) -> bytes:
    """``[C, T]`` or ``[T]`` float → PCM16 WAV. Keeps every channel (PyTorch ``add_audio`` cannot)."""
    array = waveform.detach().float().cpu().numpy()
    if array.ndim == 1:
        array = array[None]
    if array.ndim != 2:
        raise ValueError(f"audio must be [C, T] or [T], got {tuple(array.shape)}")
    peak = float(np.abs(array).max()) if array.size else 0.0
    if peak > 1.0:
        array = array / peak
    pcm = np.ascontiguousarray(
        np.clip(array.T * np.iinfo(np.int16).max, -32768, 32767).astype("<i2")
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(int(array.shape[0]))
        writer.setsampwidth(2)
        writer.setframerate(int(sample_rate))
        writer.writeframes(pcm.tobytes())
    return buffer.getvalue()


def audio_summary(tag: str, waveform: torch.Tensor, sample_rate: int):
    """Stereo-capable replacement for ``torch.utils.tensorboard.summary.audio`` (that helper is mono-only)."""
    from tensorboard.compat.proto.summary_pb2 import Summary

    array = waveform.detach().float().cpu()
    if array.ndim == 1:
        array = array.unsqueeze(0)
    encoded = encode_wav(array, sample_rate)
    return Summary(value=[Summary.Value(
        tag=tag,
        audio=Summary.Audio(
            sample_rate=int(sample_rate),
            num_channels=int(array.shape[0]),
            length_frames=int(array.shape[-1]),
            encoded_audio_string=encoded,
            content_type="audio/wav",
        ),
    )])


class TensorBoardLogger:
    """Rank-0 writer. ``enabled=False`` or a missing tensorboard install is a no-op."""

    def __init__(self, run_dir: Path, enabled: bool = True, audio: bool = True, log: Callable[[str], None] | None = None):
        self.run_dir = Path(run_dir)
        self.tb_dir = self.run_dir / "tb"
        self.audio = bool(audio)
        self.writer = None
        self.cursor: dict[str, Any] = {"scalars": {}, "audio_steps": []}
        self._log = log or (lambda message: print(message, flush=True))
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            self._log("[tb] tensorboard not installed; scalars stay in log.jsonl")
            return
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.cursor = self._load_cursor()
        if self.audio:
            self._upgrade_audio_layout()
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        n_train, n_val, n_wav = self.sync()
        self._log(f"[tb] {self.tb_dir} (sync train={n_train} val={n_val} wav_steps={n_wav})")

    def log_record(self, record: dict[str, Any]) -> None:
        if self.writer is None or "step" not in record:
            return
        step = int(record["step"])
        for tag, value in flatten_scalars(record).items():
            self.writer.add_scalar(tag, value, step)
        event = str(record.get("event") or "train")
        seen = self.cursor.setdefault("scalars", {})
        seen[event] = max(int(seen.get(event, -1)), step)
        self._save_cursor()
        self.writer.flush()

    def log_previews(self, step: int, sample_dir: Path) -> None:
        if self.writer is None or not self.audio or not sample_dir.is_dir():
            return
        done = self.cursor.setdefault("audio_steps", [])
        if int(step) in {int(s) for s in done}:
            return
        import torchaudio

        wrote = 0
        file_writer = self.writer._get_file_writer()
        for wav in sorted(sample_dir.glob("*.wav")):
            audio, rate = torchaudio.load(str(wav))
            file_writer.add_summary(audio_summary(f"preview/{wav.stem}", audio.float(), int(rate)), global_step=int(step))
            wrote += 1
        if wrote:
            done.append(int(step))
            self.cursor["audio_layout"] = AUDIO_LAYOUT
            self._save_cursor()
            self.writer.flush()

    def sync(self) -> tuple[int, int, int]:
        """Import jsonl rows and sample dirs newer than the cursor. Returns (train, val, audio steps) written."""
        if self.writer is None:
            return 0, 0, 0
        n_train = n_val = 0
        log_path = self.run_dir / "log.jsonl"
        if log_path.is_file():
            seen = self.cursor.setdefault("scalars", {})
            with log_path.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    event = str(record.get("event") or "train")
                    step = int(record.get("step", -1))
                    if step <= int(seen.get(event, -1)):
                        continue
                    self.log_record(record)
                    if event == "val":
                        n_val += 1
                    else:
                        n_train += 1
        n_wav = 0
        samples = self.run_dir / "samples"
        if self.audio and samples.is_dir():
            done = {int(s) for s in self.cursor.get("audio_steps", [])}
            for folder in sorted(samples.glob("step_*")):
                try:
                    step = int(folder.name.split("_", 1)[1])
                except ValueError:
                    continue
                if step in done:
                    continue
                before = len(self.cursor.get("audio_steps", []))
                self.log_previews(step, folder)
                if len(self.cursor.get("audio_steps", [])) > before:
                    n_wav += 1
        return n_train, n_val, n_wav

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def _upgrade_audio_layout(self) -> None:
        """Old events used PyTorch's mono ``add_audio``. Rebuild from jsonl + wavs as stereo."""
        if self.cursor.get("audio_layout") == AUDIO_LAYOUT:
            return
        stale = list(self.tb_dir.glob("events.out.tfevents*"))
        if stale:
            for path in stale:
                path.unlink()
            self._log("[tb] rewriting events with stereo preview wavs")
        self.cursor = {"scalars": {}, "audio_steps": [], "audio_layout": AUDIO_LAYOUT}
        self._save_cursor()

    def _load_cursor(self) -> dict[str, Any]:
        path = self.tb_dir / CURSOR
        if not path.is_file():
            return {"scalars": {}, "audio_steps": [], "audio_layout": AUDIO_LAYOUT}
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return {"scalars": {}, "audio_steps": [], "audio_layout": AUDIO_LAYOUT}
        data.setdefault("scalars", {})
        data.setdefault("audio_steps", [])
        return data

    def _save_cursor(self) -> None:
        (self.tb_dir / CURSOR).write_text(json.dumps(self.cursor) + "\n")


def sync_tree(root: Path, log: Callable[[str], None] | None = None) -> int:
    """Sync ``root`` if it is a run, otherwise every descendant that has ``log.jsonl``."""
    root = Path(root)
    if (root / "log.jsonl").is_file():
        runs = [root]
    else:
        runs = sorted({path.parent for path in root.glob("**/log.jsonl")})
    if not runs:
        (log or print)(f"[tb] no log.jsonl under {root}")
        return 0
    for run in runs:
        TensorBoardLogger(run, enabled=True, log=log).close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print("usage: python -m chrome.train.tensorboard <run_dir|family_dir>", file=sys.stderr)
        return 0 if args else 2
    return sync_tree(Path(args[0]))


if __name__ == "__main__":
    sys.exit(main())
