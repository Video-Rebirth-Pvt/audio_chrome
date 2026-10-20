"""Decode audio files (or ``<file>.arrow#<row>`` refs to repacked bytes) into float32 [C, N]."""

from __future__ import annotations

import io
import math
from collections.abc import Callable
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from chrome.data.blobs import parse_blob_ref, read_blob

_RESAMPLERS: dict[tuple[int, int], torchaudio.transforms.Resample] = {}


def audio_source(path: str | Path) -> Callable[[], str | io.BytesIO]:
    """Opener for a decoder: the path itself, or a fresh in-memory file per call for a blob ref."""
    if parse_blob_ref(str(path)) is None:
        return lambda: str(path)
    data = read_blob(str(path))
    return lambda: io.BytesIO(data)


def resample(waveform: torch.Tensor, orig_rate: int, new_rate: int) -> torch.Tensor:
    """``torchaudio.functional.resample`` (within 1e-4) with the sinc kernel built once per rate pair."""
    if orig_rate == new_rate:
        return waveform
    key = (int(orig_rate), int(new_rate))
    if key not in _RESAMPLERS:
        _RESAMPLERS[key] = torchaudio.transforms.Resample(*key)
    return _RESAMPLERS[key](waveform)


def read_audio_window(path: str | Path, start_seconds: float, num_seconds: float) -> tuple[torch.Tensor, int]:
    """[C, n] float32 covering [start, start + num) seconds, decoding only that span when the format seeks."""
    errors: list[Exception] = []
    source = audio_source(path)
    try:
        with sf.SoundFile(source()) as handle:
            rate = handle.samplerate
            start = min(int(round(start_seconds * rate)), handle.frames)
            frames = int(math.ceil(num_seconds * rate))
            if handle.seekable():
                handle.seek(start)
                data = handle.read(frames, dtype="float32", always_2d=True)
            else:
                data = handle.read(dtype="float32", always_2d=True)[start : start + frames]
        if len(data) > 0:
            return torch.from_numpy(data.T.copy()), int(rate)
        errors.append(RuntimeError("empty window"))
    except Exception as exc:
        errors.append(exc)
    try:
        rate = torchaudio.info(source()).sample_rate
        waveform, rate = torchaudio.load(
            source(), frame_offset=int(round(start_seconds * rate)), num_frames=int(math.ceil(num_seconds * rate))
        )
        if waveform.numel() > 0:
            return waveform.float(), int(rate)
        errors.append(RuntimeError("empty window"))
    except Exception as exc:
        errors.append(exc)
    raise RuntimeError(f"failed to decode {path}: " + "; ".join(str(e) for e in errors)) from errors[-1]


def read_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    """[C, N] float32 and the file's sample rate. soundfile first, torchaudio (ffmpeg) for the rest."""
    errors: list[Exception] = []
    source = audio_source(path)
    try:
        data, rate = sf.read(source(), always_2d=True, dtype="float32")
        if len(data) > 0:
            return torch.from_numpy(data.T.copy()), int(rate)
        errors.append(RuntimeError("empty audio"))
    except Exception as exc:
        errors.append(exc)
    try:
        waveform, rate = torchaudio.load(source())
        if waveform.numel() > 0:
            return waveform.float(), int(rate)
        errors.append(RuntimeError("empty audio"))
    except Exception as exc:
        errors.append(exc)
    raise RuntimeError(f"failed to decode {path}: " + "; ".join(str(e) for e in errors)) from errors[-1]


def to_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """Mono is copied to both ears. Extra channels fold into L/R."""
    channels = waveform.shape[0]
    if channels == 1:
        return waveform.repeat(2, 1)
    if channels == 2:
        return waveform
    left, right = waveform[0].clone(), waveform[1].clone()
    extra = waveform[2:]
    left = left + 0.7071 * extra[0]
    right = right + (0.7071 * extra[0] if extra.shape[0] == 1 else extra[1:].mean(dim=0))
    return torch.stack([left, right], dim=0)


def load_audio(
    path: str | Path, sample_rate: int = 32000, channels: int = 2, peak: float | None = None
) -> torch.Tensor:
    """[channels, N] float32 at ``sample_rate``. With ``peak``, louder clips are scaled down to it."""
    waveform, rate = read_audio(path)
    if channels == 2:
        waveform = to_stereo(waveform)
    elif channels == 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    else:
        raise ValueError(f"channels must be 1 or 2, got {channels}")
    return limit_peak(resample(waveform, rate, sample_rate), peak)


def limit_peak(waveform: torch.Tensor, peak: float | None) -> torch.Tensor:
    """Scale down only when louder than ``peak``, then clamp to [-1, 1]."""
    if peak is not None:
        max_abs = float(waveform.abs().amax())
        if max_abs > peak:
            waveform = waveform * (peak / max_abs)
    return waveform.clamp(-1.0, 1.0)


def save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """[C, N] float tensor → 16-bit PCM wav."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = waveform.detach().float().clamp(-1.0, 1.0).cpu().numpy().T
    sf.write(str(path), data, sample_rate, subtype="PCM_16")
