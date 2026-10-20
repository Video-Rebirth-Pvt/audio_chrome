"""Packed single-stream sequences: many samples, one varlen sequence, padded to a multiple of 64.

A sample is a list of ``Segment``s (modality, (t, h, w) positions, whether its rows are a denoising target,
whether their AdaLN time is pinned to clean). ``pack`` lays samples end to end and derives everything the DiT
needs: modality tags, RoPE positions, ``cu_seqlens`` (one attention segment per sample, plus one for the
padding), and per-modality row indices. T2A is ``[text | audio L | audio R]`` per sample; S5 adds video
segments without touching the packer or the model.

    seq = pack([t2a_segments(len(ids), frames, stereo.values(2)) for ids, frames in ...])
    t_rows = seq.row_timesteps(t_per_sample)        # text rows follow their sample's audio t
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import IntEnum

import torch


class Modality(IntEnum):
    VIDEO = 0
    TEXT = 1
    AUDIO = 2


PAD_TAG = -1
SEQ_ALIGN = 64


def pack_audio(latent: torch.Tensor) -> torch.Tensor:
    """[C, D, T] → [C·T, D], channel-major: all T rows of the left ear, then all of the right."""
    channels, dim, frames = latent.shape
    return latent.permute(0, 2, 1).reshape(channels * frames, dim)


def unpack_audio(rows: torch.Tensor, channels: int) -> torch.Tensor:
    """[C·T, D] → [C, D, T]."""
    return rows.reshape(channels, rows.shape[0] // channels, rows.shape[-1]).permute(0, 2, 1).contiguous()


@dataclass
class Segment:
    modality: Modality
    positions: torch.Tensor
    target: bool = False
    clean: bool = False

    @property
    def rows(self) -> int:
        return int(self.positions.shape[0])


def text_segment(text_len: int, t0: float = 0.0) -> Segment:
    pos = torch.zeros(text_len, 3, dtype=torch.float64)
    pos[:, 0] = t0 + torch.arange(text_len, dtype=torch.float64)
    return Segment(Modality.TEXT, pos)


def audio_segment(frames: int, w_values: list[float], t0: float, target: bool = True, clean: bool = False) -> Segment:
    """Both ears share t = t0 + k; each ear gets its own w; h = 0."""
    channels = len(w_values)
    pos = torch.zeros(channels * frames, 3, dtype=torch.float64)
    pos[:, 0] = (t0 + torch.arange(frames, dtype=torch.float64)).repeat(channels)
    pos[:, 2] = torch.tensor(w_values, dtype=torch.float64).repeat_interleave(frames)
    return Segment(Modality.AUDIO, pos, target=target, clean=clean)


def t2a_segments(text_len: int, frames: int, w_values: list[float]) -> list[Segment]:
    return [text_segment(text_len), audio_segment(frames, w_values, t0=float(text_len))]


@dataclass
class PackedSequence:
    seq_len: int
    used: int
    num_samples: int
    token_tags: torch.Tensor  # [S] long, PAD_TAG on padding
    positions: torch.Tensor  # [S, 3] float64
    sample_index: torch.Tensor  # [S] long, -1 on padding
    clean: torch.Tensor  # [S] bool, AdaLN t pinned to 1.0 (also padding)
    cu_seqlens: torch.Tensor  # int32
    max_seqlen: int
    text_pos: torch.Tensor
    audio_pos: torch.Tensor
    video_pos: torch.Tensor
    text_cu_seqlens: torch.Tensor  # int32, over the concatenated text rows
    text_max_seqlen: int
    audio_target: torch.Tensor  # [Na] bool
    audio_counts: list[int] = field(default_factory=list)  # audio rows per sample
    text_counts: list[int] = field(default_factory=list)

    def to(self, device) -> "PackedSequence":
        moved = {
            f.name: getattr(self, f.name).to(device) if torch.is_tensor(getattr(self, f.name)) else getattr(self, f.name)
            for f in fields(self)
        }
        return PackedSequence(**moved)

    def row_timesteps(self, t_sample: torch.Tensor) -> torch.Tensor:
        """Per-sample AdaLN t → per-row t; clean rows and padding get 1.0."""
        t = t_sample.to(torch.float32)[self.sample_index.clamp(min=0)]
        return torch.where(self.clean, torch.ones_like(t), t)

    def split_audio(self, rows: torch.Tensor) -> list[torch.Tensor]:
        return list(rows.split(self.audio_counts, dim=0))


def pack(samples: list[list[Segment]], align: int = SEQ_ALIGN) -> PackedSequence:
    tags, positions, sample_index, clean, audio_target = [], [], [], [], []
    lengths, audio_counts, text_counts = [], [], []
    for i, segments in enumerate(samples):
        length = audio = text = 0
        for seg in segments:
            tags.append(torch.full((seg.rows,), int(seg.modality), dtype=torch.long))
            positions.append(seg.positions.to(torch.float64))
            sample_index.append(torch.full((seg.rows,), i, dtype=torch.long))
            clean.append(torch.full((seg.rows,), seg.clean, dtype=torch.bool))
            if seg.modality == Modality.AUDIO:
                audio_target.append(torch.full((seg.rows,), seg.target, dtype=torch.bool))
                audio += seg.rows
            elif seg.modality == Modality.TEXT:
                text += seg.rows
            length += seg.rows
        if length == 0:
            raise ValueError(f"sample {i} has no rows")
        lengths.append(length)
        audio_counts.append(audio)
        text_counts.append(text)
    used = sum(lengths)
    seq_len = -(-used // align) * align
    pad = seq_len - used
    token_tags = torch.cat([*tags, torch.full((pad,), PAD_TAG, dtype=torch.long)])
    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)
    if pad:
        bounds.append(seq_len)
    text_bounds = [0]
    for count in text_counts:
        if count:
            text_bounds.append(text_bounds[-1] + count)
    return PackedSequence(
        seq_len=seq_len,
        used=used,
        num_samples=len(samples),
        token_tags=token_tags,
        positions=torch.cat([*positions, torch.zeros(pad, 3, dtype=torch.float64)]),
        sample_index=torch.cat([*sample_index, torch.full((pad,), -1, dtype=torch.long)]),
        clean=torch.cat([*clean, torch.ones(pad, dtype=torch.bool)]),
        cu_seqlens=torch.tensor(bounds, dtype=torch.int32),
        max_seqlen=max(max(lengths), pad),
        text_pos=torch.nonzero(token_tags == Modality.TEXT).flatten(),
        audio_pos=torch.nonzero(token_tags == Modality.AUDIO).flatten(),
        video_pos=torch.nonzero(token_tags == Modality.VIDEO).flatten(),
        text_cu_seqlens=torch.tensor(text_bounds, dtype=torch.int32),
        text_max_seqlen=max(text_counts) if text_counts else 0,
        audio_target=torch.cat(audio_target) if audio_target else torch.zeros(0, dtype=torch.bool),
        audio_counts=audio_counts,
        text_counts=text_counts,
    )
