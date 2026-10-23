"""Trace preview samples back to the source clips in the original datasets.

Training only saves generated wavs under ``samples/step_*``. The preview set already lists the
dataset path (a file, or a ``<file>.arrow#<row>`` blob). This writes those clips once, with the
same stems as the generated previews, so you can A/B later.

    bash scripts/train/dump_preview_gt.sh outputs/train/t2a_10b_s20_ws/20261021_210505
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch

from chrome.data.audio_io import load_audio, save_audio
from chrome.spec import AudioSpec


def preview_name(label: str) -> str:
    return f"preview_{str(label).replace(' ', '_')}"


def preview_seconds(duration: float, spec: AudioSpec, fixed: int | None = None) -> int:
    """Whole-second length the trainer asked the sampler for."""
    if fixed is not None:
        return int(fixed)
    lo, hi = int(spec.min_seconds), int(spec.max_seconds)
    return min(hi, max(lo, round(float(duration))))


def preview_catalog(cfg: Mapping, spec: AudioSpec) -> list[dict]:
    """One record per preview-set clip: name, caption, seconds, original audio path, duration."""
    p = cfg["preview"]
    items: list[dict] = []
    labels = set(p.get("labels") or [])
    if not p.get("set"):
        return items
    for entry in json.loads(Path(p["set"]).read_text()):
        if labels and entry["label"] not in labels:
            continue
        items.append(
            {
                "name": preview_name(entry["label"]),
                "label": entry["label"],
                "domain": entry.get("domain"),
                "caption": entry["caption"],
                "seconds": preview_seconds(
                    entry["duration"], spec, int(p["seconds"]) if p.get("seconds") else None
                ),
                "audio": entry.get("audio"),
                "duration": float(entry["duration"]),
            }
        )
    return items


def dump_preview_gt(
    cfg: Mapping,
    spec: AudioSpec,
    out_dir: str | Path,
    force: bool = False,
    log=print,
) -> list[dict]:
    """Decode each preview-set source clip → 32 kHz stereo wav named like the generated sample."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    peak = cfg["data"].get("peak")
    records = []
    for item in preview_catalog(cfg, spec):
        rec = dict(item)
        rec["wav"] = f"{item['name']}.wav"
        audio = item.get("audio")
        wav_path = out_dir / rec["wav"]
        txt_path = out_dir / f"{item['name']}.txt"
        if not audio:
            rec["status"] = "no_audio"
            records.append(rec)
            continue
        if wav_path.is_file() and not force:
            txt_path.write_text(item["caption"] + "\n")
            rec["status"] = "exists"
            records.append(rec)
            continue
        try:
            wave = load_audio(audio, spec.sample_rate, spec.channels, peak)
            target = int(item["seconds"]) * spec.sample_rate
            if wave.shape[-1] >= target:
                wave = wave[:, :target]
            else:
                wave = torch.nn.functional.pad(wave, (0, target - wave.shape[-1]))
            save_audio(wav_path, wave, spec.sample_rate)
            txt_path.write_text(item["caption"] + "\n")
            rec["status"] = "wrote"
        except Exception as exc:
            rec["status"] = f"error: {exc}"
            log(f"[preview-gt] skip {item['name']}: {exc}")
        records.append(rec)
    (out_dir / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "sources.tsv").write_text(
        "name\tseconds\tduration\tstatus\taudio\n"
        + "".join(
            f"{r['name']}\t{r['seconds']}\t{r['duration']}\t{r['status']}\t{r.get('audio') or ''}\n"
            for r in records
        )
    )
    return records
