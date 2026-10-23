"""Self-check of the T2A data pipeline on synthetic audio (CPU, about a minute).

Run from anywhere:
    bash scripts/data/check_data_pipeline.sh
    bash scripts/data/check_data_pipeline.sh --keep
    python tests/test_data_pipeline.py --keep

Checks:
  1. length policy: slack crop, pad up, crop down, too short / too long, long crop
  2. JSONL byte-range chunks see every row exactly once
  3. index build: drop reasons, probed durations, parquet manifests, dedup, exclude list
  4. fingerprints: unchanged sources are skipped; weight edits are free, row-changing edits must rebuild
  5. holdout by path hash is stable when rows are added
  6. sampler: deterministic, exact resume, disjoint rank shards, token budget, weight shares
  6b. max_records caps train rows; max_hours caps train clip hours (hash prefix, raising only adds rows);
      omitted weight mixes by kept row count
  7. dataset: fixed lengths, padding, peak limit, channel folding, seeded window reads, caption dropout,
     bad files replaced and recorded
  8. VAE batch grouping keeps order and never pads
  9. DataLoader with workers: resume continues with the next batch; validation uses holdout rows only
  9b. val quota is source-balanced: a small budget still keeps every holdout source; prefix-cutting cannot drop the last one
  10. HF parquet with embedded audio: include glob, sidecar join (+ misaligned sidecar rejected), list keep,
      repack to Arrow (skip when current), refs decode to the same clip, stale repack rejected
  11. migrate in place: dry run writes nothing, parquet deleted only when its sidecar matches, resumes;
      sidecar as manifest + Arrow audio gives the same rows; misaligned sidecar / missing Arrow rejected;
      null audio (metadata-only shards) is reported, keeps its parquet and is dropped as no_audio
Exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chrome.config import load_config  # noqa: E402
from chrome.data.batch import collate_t2a, encode_waves  # noqa: E402
from chrome.data.captions import NULL_CAPTION, caption_ids, load_caption_tokenizer  # noqa: E402
from chrome.data.dataset import BadAudioList, T2ADataset  # noqa: E402
from chrome.data.index import (  # noqa: E402
    LengthPolicy,
    SourceIndex,
    SourceSpec,
    build_source,
    check_index,
    config_fingerprint,
    mixture_weight,
    source_specs,
)
from chrome.data.loader import (  # noqa: E402
    allocate_quota,
    even_split,
    holdout_rows,
    build_train_loader,
    build_val_loader,
    open_sources,
    train_streams,
    validation_batches,
)
from chrome.data.manifest import Chunk, iter_rows, plan_chunks  # noqa: E402
from chrome.data.sampler import MixtureBatchSampler, shard  # noqa: E402
from chrome.spec import AudioSpec  # noqa: E402

FAILED: list[str] = []


def check(name: str, passed: bool, **details) -> bool:
    shown = ", ".join(f"{k}={v}" for k, v in details.items())
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  ({shown})" if shown else ""), flush=True)
    if not passed:
        FAILED.append(name)
    return bool(passed)


def write_tone(path: Path, seconds: float, rate: int, channels: int = 1, amplitude: float = 0.5) -> str:
    t = np.arange(int(round(seconds * rate))) / rate
    tone = amplitude * np.sin(2 * np.pi * 440.0 * t)
    data = np.stack([tone * (1 - 0.1 * c) for c in range(channels)], axis=1).astype(np.float32)
    sf.write(str(path), data, rate)
    return str(path)


def write_ramp(path: Path, seconds: float, rate: int) -> str:
    """Sample value = time / 40, so a decoded window reveals where it starts."""
    sf.write(str(path), (np.arange(int(seconds * rate)) / rate / 40.0).astype(np.float32), rate, subtype="FLOAT")
    return str(path)


def write_jsonl(path: Path, rows: list, raw_lines: list[str] = ()) -> str:
    lines = [json.dumps(r, ensure_ascii=False) for r in rows] + list(raw_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def make_corpus(tmp: Path) -> dict:
    audio = tmp / "audio"
    audio.mkdir()
    f = {
        "short": write_tone(audio / "short.wav", 3.5, 16000),
        "slack": write_tone(audio / "slack.flac", 4.05, 44100, channels=2),
        "pad": write_tone(audio / "pad.wav", 6.5, 48000, amplitude=1.0),
        "six": write_tone(audio / "six.wav", 12.3, 32000, channels=6),
        "full": write_tone(audio / "full.flac", 15.05, 44100),
        "long": write_tone(audio / "long.wav", 16.0, 16000),
        "nodur": write_tone(audio / "nodur.wav", 5.5, 22050),
        "excluded": write_tone(audio / "excluded.wav", 5.0, 32000),
        "lowcer": write_tone(audio / "lowcer.wav", 5.0, 32000),
        "ramp": write_ramp(audio / "ramp.wav", 20.0, 48000),
        "music": write_tone(audio / "music.flac", 6.5, 44100, channels=2),
    }
    corrupt = audio / "corrupt.wav"
    corrupt.write_bytes(b"RIFF\x00\x00\x00\x00not audio at all" * 10)
    f["corrupt"] = str(corrupt)

    def row(key, seconds, caption="[SPEECH]: <d>[English] hello</d>", **extra):
        return {"path": f[key], "resp": caption, "duration": seconds, **extra}

    speech = write_jsonl(
        tmp / "speech.jsonl",
        [
            row("short", 3.5),
            row("slack", 4.05),
            row("pad", 6.5),
            row("six", 12.3),
            row("full", 15.05),
            row("long", 16.0),
            {"path": f["nodur"], "resp": "a bird sings"},
            row("excluded", 5.0),
            row("lowcer", 5.0, cer="0.9"),
            row("corrupt", 5.0),
            {"path": f["pad"], "resp": "", "duration": 6.5},
            row("slack", 4.05),
        ],
        raw_lines=['{"path": "broken'],
    )
    music = tmp / "music.parquet"
    pq.write_table(
        pa.table({"path": [f["ramp"], f["music"]], "resp_new": ["[MUSIC]: ramp", "[MUSIC]: tones"], "duration": [20.0, 6.5]}),
        music,
    )
    (tmp / "preview.json").write_text(json.dumps([{"audio": f["excluded"]}]))

    base = {s: write_tone(audio / f"base_{s}.wav", s, 44100) for s in (4.2, 5.0, 7.5, 11.0)}
    links = tmp / "links"
    links.mkdir()
    bulk_rows = []
    for i in range(2000):
        seconds = (4.2, 5.0, 7.5, 11.0)[i % 4]
        link = links / f"clip_{i:05d}.wav"
        link.symlink_to(base[seconds])
        bulk_rows.append({"path": str(link), "resp": f"sound number {i} " * (1 + i % 5), "duration": seconds})
    bulk = write_jsonl(tmp / "bulk.jsonl", bulk_rows)
    return {"files": f, "speech": speech, "music": str(music), "bulk": bulk, "bulk_rows": bulk_rows, "links": links,
            "base": base}


def make_config(tmp: Path, corpus: dict):
    cfg = load_config(ROOT / "configs/data/t2a_base.yaml")
    override = {
        "data": {
            "index_root": str(tmp / "index"),
            "exclude": [str(tmp / "preview.json")],
            "bad_list": str(tmp / "bad_audio.txt"),
            "extra_bad_lists": [],
            "build": {"workers": 4, "chunk_mb": 0.0004, "force": False},
            "loader": {"max_tokens": 4096, "num_workers": 2, "prefetch_factor": 2},
            "val": {"per_source": 16, "num_workers": 1},
            "sources": [
                {"name": "speech", "task": "speech_en", "weight": 1.0, "paths": [corpus["speech"]],
                 "keep": {"cer": {"max": 0.3}}},
                {"name": "music", "task": "music", "weight": 1.0, "fit": "crop", "long": "crop",
                 "columns": {"caption": "resp_new"}, "paths": [corpus["music"]]},
                {"name": "bulk", "task": "sfx", "weight": 3.0, "holdout_ppm": 100000, "paths": [corpus["bulk"]]},
            ],
        }
    }
    merged = OmegaConf.merge(cfg, override)
    merged.data.sources = override["data"]["sources"]
    return merged


def check_policy() -> None:
    policy = LengthPolicy(4, 15, 0.1)
    cases = [
        (3.5, "pad", "drop", "too_short"), (4.05, "pad", "drop", 4), (6.5, "pad", "drop", 7),
        (6.5, "crop", "drop", 6), (12.3, "pad", "drop", 13), (14.5, "pad", "drop", 15), (15.05, "pad", "drop", 15),
        (16.0, "pad", "drop", "too_long"), (20.0, "crop", "crop", 15), (4.95, "crop", "drop", 4),
    ]
    got = [policy.seconds_for(d, fit, long) for d, fit, long, _ in cases]
    check("length policy", got == [c[3] for c in cases], got=got)


def check_chunks(corpus: dict) -> None:
    path = corpus["bulk"]
    whole = [r["path"] for r in iter_rows(Chunk(path, "jsonl", 0, Path(path).stat().st_size))]
    chunked = [r["path"] for c in plan_chunks([path], 97) for r in iter_rows(c)]
    check("jsonl byte-range chunks", whole == chunked and len(whole) == 2000, rows=len(whole), chunks=len(plan_chunks([path], 97)))


def check_build(cfg, corpus: dict) -> dict:
    specs = {s.name: s for s in source_specs(cfg)}
    metas = {name: build_source(spec, cfg, workers=4, log=lambda *_: None) for name, spec in specs.items()}
    f = corpus["files"]
    speech = SourceIndex(Path(cfg.data.index_root) / "speech")
    seconds = {speech.row(i)[0]: int(speech.seconds[i]) for i in range(len(speech))}
    expected = {f["slack"]: 4, f["pad"]: 7, f["six"]: 13, f["full"]: 15, f["nodur"]: 6, f["corrupt"]: 5}
    check("index rows and lengths", seconds == expected, seconds={Path(k).stem: v for k, v in seconds.items()})
    drops = metas["speech"]["drops"]
    want = {"too_short": 1, "too_long": 1, "no_caption": 1, "bad_json": 1, "duplicate": 1, "excluded": 1, "keep_filter": 1}
    check("drop reasons", all(drops.get(k) == v for k, v in want.items()), drops=drops)
    music = SourceIndex(Path(cfg.data.index_root) / "music")
    got = {Path(music.row(i)[0]).stem: int(music.seconds[i]) for i in range(len(music))}
    check("parquet manifest, fit/long crop", got == {"ramp": 15, "music": 6}, got=got)
    return metas


def check_fingerprints(cfg, metas: dict) -> None:
    spec = next(s for s in source_specs(cfg) if s.name == "speech")
    messages: list[str] = []
    again = build_source(spec, cfg, workers=4, log=messages.append)
    check("unchanged source is skipped", any("skipped" in m for m in messages)
          and again["config_fingerprint"] == metas["speech"]["config_fingerprint"])
    index = SourceIndex(Path(cfg.data.index_root) / "speech")
    weighted = OmegaConf.merge(cfg, {})
    weighted.data.sources[0].weight = 7.0
    try:
        check_index(next(s for s in source_specs(weighted) if s.name == "speech"), index, weighted, log=lambda *_: None)
        check("weight change needs no rebuild", True)
    except RuntimeError as exc:
        check("weight change needs no rebuild", False, error=str(exc))
    refit = OmegaConf.merge(cfg, {})
    refit.data.sources[0].fit = "crop"
    try:
        check_index(next(s for s in source_specs(refit) if s.name == "speech"), index, refit, log=lambda *_: None)
        check("fit change demands a rebuild", False)
    except RuntimeError:
        check("fit change demands a rebuild", True)


def check_holdout(cfg, corpus: dict, tmp: Path) -> None:
    root = Path(cfg.data.index_root)
    before = SourceIndex(root / "bulk")
    held = {before.row(i)[0] for i in np.flatnonzero(before.holdout)}
    rows = list(corpus["bulk_rows"])
    for i in range(2000, 2200):
        link = corpus["links"] / f"clip_{i:05d}.wav"
        link.symlink_to(corpus["base"][5.0])
        rows.append({"path": str(link), "resp": f"new sound {i}", "duration": 5.0})
    grown = tmp / "bulk_grown.jsonl"
    write_jsonl(grown, rows)
    cfg2 = OmegaConf.merge(cfg, {"data": {"index_root": str(tmp / "index_grown")}})
    cfg2.data.sources[2].paths = [str(grown)]
    build_source(next(s for s in source_specs(cfg2) if s.name == "bulk"), cfg2, workers=4, log=lambda *_: None)
    after = SourceIndex(tmp / "index_grown" / "bulk")
    held_after = {after.row(i)[0] for i in np.flatnonzero(after.holdout)}
    old = {r["path"] for r in corpus["bulk_rows"]}
    check("holdout stable when rows are added", held == (held_after & old) and 150 < len(held) < 250,
          holdout=len(held), after=len(held_after))


def check_sampler(cfg) -> None:
    specs, indexes = open_sources(cfg, log=lambda *_: None)
    streams = train_streams(specs, indexes, AudioSpec.from_config(cfg))
    make = lambda rank=0, world=1: MixtureBatchSampler(streams, 4096, 64, rank, world, seed=5)  # noqa: E731
    a, b = make(), make()
    first = [a.next_batch() for _ in range(40)]
    check("sampler is deterministic", first == [b.next_batch() for _ in range(40)])
    state = a.state_dict(25)
    resumed = make()
    resumed.load_state_dict(state, log=lambda *_: None)
    check("resume continues with batch 26", [resumed.next_batch() for _ in range(15)] == first[25:40])
    costs = [sum(int(streams[s].cost[r]) for _, s, r, _ in batch) for batch in first]
    check("token budget", max(costs) <= 4096 and min(costs) > 4096 * 0.5, max=max(costs), min=min(costs))
    bulk = next(s for s in streams if s.name == "bulk")
    r0, r1 = shard(bulk, 5, 0, 0, 2), shard(bulk, 5, 0, 1, 2)
    check("rank shards are disjoint", not set(r0) & set(r1) and len(set(r0) | set(r1)) == len(bulk.rows))
    draws = [s for batch in (a.next_batch() for _ in range(400)) for _, s, _, _ in batch]
    share = np.bincount(draws, minlength=3) / len(draws)
    check("weight shares", abs(share[2] - 0.6) < 0.03, share=np.round(share, 3).tolist())


def check_max_records(cfg, tmp: Path) -> None:
    spec = SourceSpec(name="n", paths=("/x",), weight=None, max_records=80)
    check("omitted weight uses train rows", mixture_weight(spec, 80) == 80.0)
    check("explicit weight still wins", mixture_weight(SourceSpec(name="n", paths=("/x",), weight=3.0), 80) == 3.0)

    capped = OmegaConf.merge(cfg, {"data": {"index_root": str(tmp / "index_capped")}})
    capped.data.sources[2].max_records = 80
    meta = build_source(next(s for s in source_specs(capped) if s.name == "bulk"), capped, workers=4, log=lambda *_: None)
    check("max_records caps train rows", meta["train_rows"] == 80, train=meta["train_rows"], holdout=meta["holdout_rows"])
    index = SourceIndex(tmp / "index_capped" / "bulk")
    try:
        check_index(next(s for s in source_specs(cfg) if s.name == "bulk"), index, cfg, log=lambda *_: None)
        check("max_records change demands a rebuild", False)
    except RuntimeError:
        check("max_records change demands a rebuild", True)

    full = build_source(next(s for s in source_specs(cfg) if s.name == "bulk"), cfg, workers=4, log=lambda *_: None)
    picked = {}
    for hours in (0.05, 0.1):
        root = tmp / f"index_hours_{hours}"
        limited = OmegaConf.merge(cfg, {"data": {"index_root": str(root)}})
        limited.data.sources[2].max_hours = hours
        meta = build_source(next(s for s in source_specs(limited) if s.name == "bulk"), limited, workers=4,
                            log=lambda *_: None)
        index = SourceIndex(root / "bulk")
        train = ~index.holdout
        picked[hours] = (set(index.keys()[train].tolist()), int(index.seconds[train].sum()), meta)
    small, large = picked[0.05], picked[0.1]
    check("max_hours caps train clip seconds", small[1] <= 0.05 * 3600 < small[1] + 20 and large[1] <= 0.1 * 3600
          and small[2]["holdout_rows"] == full["holdout_rows"], seconds=(small[1], large[1]))
    check("raising max_hours only adds rows", small[0] < large[0], rows=(len(small[0]), len(large[0])))
    check("max_hours dropped rows are counted", small[2]["drops"].get("max_hours", 0) == full["train_rows"] - len(small[0]),
          drops=small[2]["drops"].get("max_hours"))
    try:
        check_index(next(s for s in source_specs(cfg) if s.name == "bulk"), SourceIndex(tmp / "index_hours_0.05" / "bulk"),
                    cfg, log=lambda *_: None)
        check("max_hours change demands a rebuild", False)
    except RuntimeError:
        check("max_hours change demands a rebuild", True)

    equal = OmegaConf.merge(cfg, {})
    for source in equal.data.sources:
        source.weight = None
    specs, indexes = open_sources(equal, log=lambda *_: None)
    streams = train_streams(specs, indexes, AudioSpec.from_config(equal))
    check(
        "omitted weight mixes by kept rows",
        all(abs(s.weight - len(s.rows)) < 1e-6 for s in streams),
        weights=[s.weight for s in streams],
        rows=[len(s.rows) for s in streams],
    )


def check_min_seconds(cfg, corpus: dict, metas: dict, tmp: Path) -> None:
    short = corpus["files"]["short"]
    lowered = OmegaConf.merge(cfg, {"data": {"index_root": str(tmp / "index_min3")}})
    lowered.data.sources[0].min_seconds = 3
    meta = build_source(next(s for s in source_specs(lowered) if s.name == "speech"), lowered, workers=4,
                        log=lambda *_: None)
    index = SourceIndex(tmp / "index_min3" / "speech")
    seconds = {index.row(i)[0]: int(index.seconds[i]) for i in range(len(index))}
    check("source min_seconds keeps shorter clips", seconds.get(short) == 4 and "too_short" not in meta["drops"],
          short=seconds.get(short), drops=meta["drops"])
    try:
        check_index(next(s for s in source_specs(lowered) if s.name == "speech"),
                    SourceIndex(Path(cfg.data.index_root) / "speech"), lowered, log=lambda *_: None)
        check("min_seconds change demands a rebuild", False)
    except RuntimeError:
        check("min_seconds change demands a rebuild", True)
    pinned = OmegaConf.merge(cfg, {"audio": {"min_seconds": 3}})
    pinned.data.sources[0].min_seconds = int(cfg.audio.min_seconds)
    same = config_fingerprint(next(s for s in source_specs(pinned) if s.name == "speech"), pinned)
    check("source pinned to the old minimum keeps its fingerprint", same == metas["speech"]["config_fingerprint"])
    try:
        SourceSpec.from_config({"name": "n", "paths": ["/x"], "min_seconds": 0}, {})
        check("min_seconds < 1 is rejected", False)
    except ValueError:
        check("min_seconds < 1 is rejected", True)


def check_dataset(cfg, corpus: dict) -> None:
    specs, indexes = open_sources(cfg, log=lambda *_: None)
    spec = AudioSpec.from_config(cfg)
    tokenizer = load_caption_tokenizer(cfg.data.tokenizer)
    bad = BadAudioList(cfg.data.bad_list)
    dataset = T2ADataset(indexes, spec, cfg.data.tokenizer, bad, cfg_drop=0.0, peak=0.95, crop="random")
    speech = indexes[0]
    names = {Path(speech.row(i)[0]).stem: i for i in range(len(speech))}
    pad = dataset[(1, 0, names["pad"], 7)]
    check("fixed length + padding", tuple(pad["wave"].shape) == (2, 7 * 32000) and abs(pad["valid_samples"] - 6.5 * 32000) <= 800,
          shape=tuple(pad["wave"].shape), valid=pad["valid_samples"])
    check("peak limited to 0.95", float(pad["wave"].abs().max()) <= 0.95 + 1e-5, peak=round(float(pad["wave"].abs().max()), 4))
    six = dataset[(1, 0, names["six"], 7)]
    check("6-channel file folds to stereo", tuple(six["wave"].shape) == (2, 13 * 32000))
    check("H3 caption ids", pad["input_ids"].tolist() == caption_ids(tokenizer, "[SPEECH]: <d>[English] hello</d>")
          and 151669 in pad["input_ids"].tolist())
    music = indexes[1]
    ramp_row = next(i for i in range(len(music)) if Path(music.row(i)[0]).stem == "ramp")
    ramp = dataset[(1, 1, ramp_row, 11)]
    start = float(ramp["wave"][0, 400]) * 40 - 400 / 32000
    again = dataset[(1, 1, ramp_row, 11)]
    check("seeded window read", 0 <= start <= 5.05 and torch.equal(ramp["wave"], again["wave"]), start_s=round(start, 3))
    dropped = T2ADataset(indexes, spec, cfg.data.tokenizer, bad, cfg_drop=1.0)[(1, 0, names["pad"], 7)]
    check("caption dropout → ' '", dropped["input_ids"].tolist() == caption_ids(tokenizer, NULL_CAPTION) == [220])
    replaced = dataset[(1, 0, names["corrupt"], 3)]
    listed = corpus["files"]["corrupt"] in Path(cfg.data.bad_list).read_text()
    check("bad file replaced and recorded", listed and replaced["path"] != corpus["files"]["corrupt"], got=Path(replaced["path"]).stem)


class FakeVAE:
    def __init__(self):
        self.calls = 0

    def encode_audio(self, x):
        self.calls += 1
        return x[..., ::800].unsqueeze(-2).expand(*x.shape[:-1], 32, x.shape[-1] // 800)


def check_grouping() -> None:
    waves = [torch.randn(2, s * 32000) for s in (4, 7, 4, 15, 7, 4)]
    vae = FakeVAE()
    latents = encode_waves(vae, waves, device="cpu")
    single = [FakeVAE().encode_audio(w[None])[0] for w in waves]
    check("VAE grouping keeps order", all(torch.equal(a, b) for a, b in zip(latents, single)) and vae.calls == 3,
          calls=vae.calls)
    batch = collate_t2a([{"batch_id": 3, "wave": w, "seconds": w.shape[-1] // 32000, "valid_samples": w.shape[-1],
                          "input_ids": torch.tensor([1, 2]), "cfg_dropped": False, "source": 0, "row": i,
                          "caption": "x", "path": "p"} for i, w in enumerate(waves)])
    check("collate", batch["batch_id"] == 3 and batch["seconds"].tolist() == [4, 7, 4, 15, 7, 4])


def check_val_quota(cfg, specs, indexes) -> None:
    names = [s.name for s in specs]
    alive = [s.name for s, index in zip(specs, indexes) if int(index.holdout.sum()) > 0]
    tight = OmegaConf.merge(cfg, {"train": {"val_batches": 2}, "data": {"val": {"per_source": 16, "max_total_clips": 0}}})
    batches = validation_batches(tight, specs, indexes, log=lambda *_: None)
    seen = {names[batch[0][1]] for batch in batches}
    check("small val_batches still covers every holdout source", seen == set(alive),
          seen=sorted(seen), alive=alive, batches=len(batches))
    capped = OmegaConf.merge(cfg, {"train": {"val_batches": 0}, "data": {"val": {"per_source": 64, "max_total_clips": 6}}})
    packed = validation_batches(capped, specs, indexes, log=lambda *_: None)
    covered = {names[batch[0][1]] for batch in packed}
    n_clips = sum(len(batch) for batch in packed)
    check("max_total_clips scales but keeps every holdout source",
          covered == set(alive) and 1 <= n_clips <= max(6, len(alive)), covered=sorted(covered), clips=n_clips)
    sizes = [10, 8, 10, 9, 8]
    got = allocate_quota(sizes, 32)
    check("allocate_quota never drops a later source", all(n > 0 for n in got) and sum(got) == 32 and got[-1] > 0,
          got=got)
    check("allocate_quota budget 0 keeps all", allocate_quota(sizes, 0) == sizes)


def check_val_groups() -> None:
    from chrome.train.trainer import metric_keys, top_metrics

    class Holdout:
        def __init__(self, keys, held):
            self._keys, self.holdout = np.asarray(keys, np.uint64), np.asarray(held, bool)

        def keys(self):
            return self._keys

    rng = np.random.default_rng(0)
    def index(n_rows, n_held):
        keys = rng.permutation(10_000)[:n_rows].astype(np.uint64)
        return Holdout(keys, np.arange(n_rows) < n_held)
    specs = [SourceSpec(name="solo", paths=("/x",), task="sfx"),
             SourceSpec(name="small", paths=("/x",), task="t1", group="g"),
             SourceSpec(name="big_a", paths=("/x",), task="t2", group="g"),
             SourceSpec(name="big_b", paths=("/x",), task="t2", group="g")]
    indexes = [index(30, 10), index(30, 3), index(60, 20), index(60, 20)]
    rows = holdout_rows(specs, indexes, 8)
    solo = np.flatnonzero(indexes[0].holdout)
    check("ungrouped source: smallest hashes, as before",
          rows[0].tolist() == solo[np.argsort(indexes[0].keys()[solo])][:8].tolist())
    t2 = sorted((int(indexes[i].keys()[r]), i, int(r)) for i in (2, 3) for r in np.flatnonzero(indexes[i].holdout))[:5]
    got_t2 = sorted((int(indexes[i].keys()[r]), i, int(r)) for i in (2, 3) for r in rows[i])
    check("group shares per_source evenly over tasks, spare to the rest",
          len(rows[1]) == 3 and got_t2 == t2, sizes=[len(r) for r in rows])
    check("even_split", even_split([3, 40, 40], 30) == [3, 14, 13] and even_split([5, 5], 30) == [5, 5]
          and even_split([0, 9], 4) == [0, 4], got=even_split([3, 40, 40], 30))
    names = list(dict.fromkeys(s.metric for s in specs))
    check("group metrics roll up", names == ["solo", "g/t1", "g/t2"] and metric_keys("g/t1") == ("g/t1", "g")
          and top_metrics(names) == ["solo", "g"], names=names)
    check("group is not in the fingerprint", specs[1].content() == SourceSpec(name="small", paths=("/x",)).content())


def check_loader(cfg) -> None:
    loader, sampler = build_train_loader(cfg, log=lambda *_: None)
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) == 6:
            break
    state = sampler.state_dict(batches[2]["batch_id"])
    del loader
    resumed, _ = build_train_loader(cfg, state=state, log=lambda *_: None)
    nxt = next(iter(resumed))
    same = nxt["paths"] == batches[3]["paths"] and all(torch.equal(a, b) for a, b in zip(nxt["waves"], batches[3]["waves"]))
    check("loader resume (2 workers)", same and nxt["batch_id"] == batches[3]["batch_id"], batch_id=nxt["batch_id"])
    specs, indexes = open_sources(cfg, log=lambda *_: None)
    val = list(build_val_loader(cfg, log=lambda *_: None))
    names = [s.name for s in specs]
    rows = [(names[s], r) for b in val for s, r in zip(b["source"].tolist(), b["row"].tolist())]
    only_holdout = all(indexes[names.index(n)].holdout[r] for n, r in rows)
    check("validation = holdout rows, no dropout", rows and only_holdout and not any(b["cfg_dropped"].any() for b in val),
          clips=len(rows))
    check_val_quota(cfg, specs, indexes)


SHARD_ROWS = {
    "train-00000": ([5.0, 6.5, 3.0, 8.05], ["clean", "high", "medium", "clean"]),
    "train-00001": ([7.0, 4.4], ["medium", "clean"]),
    "test-00000": ([5.0], ["clean"]),
}
KEPT_SEGMENTS = {"train-00000_seg0", "train-00000_seg3", "train-00001_seg0", "train-00001_seg1"}


def _clip(seconds: float, rate: int, fmt: str) -> bytes:
    buffer = io.BytesIO()
    t = np.arange(int(seconds * rate)) / rate
    sf.write(buffer, (0.3 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), rate, format=fmt)
    return buffer.getvalue()


def _write_hf_shards(shards: Path, sidecars: Path, full_sidecar: bool = False) -> dict[str, list[bytes]]:
    """HF-style shards (audio struct, row groups of 3) and their sidecars; returns the clip bytes per shard."""
    shards.mkdir(parents=True)
    sidecars.mkdir(parents=True)
    clips = {}
    for stem, (durations, levels) in SHARD_ROWS.items():
        ids = [f"{stem}_seg{i}" for i in range(len(durations))]
        clips[stem] = [_clip(d, 24000, "OGG" if i % 2 else "FLAC") for i, d in enumerate(durations)]
        audio = [{"bytes": b, "path": f"{s}.opus"} for b, s in zip(clips[stem], ids)]
        table = pa.table({"audio": audio, "duration": durations, "segment_id": ids, "cer": [0.0] * len(ids)})
        pq.write_table(table, shards / f"{stem}.parquet", row_group_size=3)
        side = {"row_index": list(range(len(ids))), "segment_id": ids, "qa_risk_level": levels,
                "resp": [f"[SPEECH]: <d>[Test] {s}</d>" for s in ids]}
        if full_sidecar:
            side.update(duration=durations, cer=[0.0] * len(ids), audio_path=[f"{s}.opus" for s in ids])
        pq.write_table(pa.table(side), sidecars / f"{stem}.parquet", row_group_size=3)
    return clips


def check_embedded(base_cfg, tmp: Path) -> None:
    from chrome.data.audio_io import read_audio
    from chrome.data.blobs import blob_path, parse_blob_ref, repack

    shards, sidecars = tmp / "hf" / "xx_yy", tmp / "sidecar" / "xx_yy"
    _write_hf_shards(shards, sidecars)

    cfg = OmegaConf.merge(base_cfg, {"data": {"index_root": str(tmp / "index_hf")}})
    cfg.data.sources = [{"name": "hf", "task": "speech", "holdout_ppm": 0, "include": "train-*.parquet",
                         "sidecar": str(tmp / "sidecar"), "blobs": str(tmp / "blobs"),
                         "columns": {"caption": "resp"}, "keep": {"qa_risk_level": ["clean", "medium"]},
                         "paths": [str(shards)]}]
    spec = source_specs(cfg)[0]
    try:
        build_source(spec, cfg, workers=2, log=lambda *_: None)
        check("embedded audio needs a repack first", False)
    except FileNotFoundError:
        check("embedded audio needs a repack first", True)

    pairs = [(f, blob_path(spec.blobs, f)) for f in sorted(shards.glob("train-*.parquet"))]
    first = repack(pairs, workers=2, log=lambda *_: None)
    again = repack(pairs, workers=2, log=lambda *_: None)
    check("repack writes once, then skips", first["rows"] == 6 and first["skipped"] == 0 and again["skipped"] == 2,
          first=first, again=again)

    meta = build_source(spec, cfg, workers=2, log=lambda *_: None)
    index = SourceIndex(Path(cfg.data.index_root) / "hf")
    rows = {index.row(i)[1].split("] ")[-1][:-4]: index.row(i)[0] for i in range(len(index))}
    check("include + sidecar join + list keep", set(rows) == KEPT_SEGMENTS
          and meta["drops"].get("keep_filter") == 1 and meta["drops"].get("too_short") == 1,
          kept=sorted(rows), drops=meta["drops"])
    file, row = parse_blob_ref(rows["train-00000_seg3"])
    check("ref points at the shard row", file.endswith("xx_yy/train-00000.arrow") and row == 3, ref=rows["train-00000_seg3"])

    wave, rate = read_audio(rows["train-00001_seg1"])
    check("ref decodes to the embedded clip", rate == 24000 and abs(wave.shape[-1] / rate - 4.4) < 0.05,
          seconds=round(wave.shape[-1] / rate, 3))
    dataset = T2ADataset([index], AudioSpec.from_config(cfg), cfg.data.tokenizer, BadAudioList(tmp / "bad_hf.txt"))
    names = [index.row(i)[1] for i in range(len(index))]
    item = dataset[(1, 0, names.index("[SPEECH]: <d>[Test] train-00000_seg3</d>"), 9)]
    check("dataset crops a window from bytes", tuple(item["wave"].shape) == (2, 8 * 32000) and item["valid_samples"] == 8 * 32000,
          shape=tuple(item["wave"].shape))

    good = pq.read_table(sidecars / "train-00001.parquet")
    bad = good.to_pydict()
    bad["segment_id"] = bad["segment_id"][::-1]
    pq.write_table(pa.table(bad), sidecars / "train-00001.parquet")
    try:
        build_source(spec, cfg, workers=2, log=lambda *_: None)
        check("misaligned sidecar is rejected", False)
    except ValueError:
        check("misaligned sidecar is rejected", True)
    pq.write_table(good, sidecars / "train-00001.parquet")

    table = pq.read_table(shards / "train-00000.parquet")
    pq.write_table(table.slice(0, 3), shards / "train-00000.parquet")
    try:
        build_source(spec, cfg, workers=2, force=True, log=lambda *_: None)
        check("stale repack is rejected", False)
    except FileNotFoundError:
        check("stale repack is rejected", True)


def check_migrated(base_cfg, tmp: Path) -> None:
    from chrome.data.audio_io import read_audio
    from chrome.data.blobs import parse_blob_ref
    from chrome.data.cli import cmd_migrate

    root, side_root = tmp / "ws" / "data", tmp / "ws" / "qa_sidecar"
    shards, sidecars = root / "xx_yy", side_root / "xx_yy"
    clips = _write_hf_shards(shards, sidecars, full_sidecar=True)
    # Metadata-only shard (HF writes a null audio struct) and a shard with one null clip.
    holder = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    for stem, payload in {"train-00002": [None, None], "train-00003": [_clip(6.0, 24000, "FLAC"), None]}.items():
        ids = [f"{stem}_seg{i}" for i in range(len(payload))]
        audio = pa.array([None if b is None else {"bytes": b, "path": f"{s}.opus"} for b, s in zip(payload, ids)], holder)
        common = {"duration": [6.0, 7.0], "segment_id": ids, "cer": [0.0, 0.0]}
        pq.write_table(pa.table({"audio": audio, **common}), shards / f"{stem}.parquet")
        pq.write_table(pa.table({"row_index": [0, 1], **common, "qa_risk_level": ["clean", "clean"],
                                 "audio_path": [f"{s}.opus" for s in ids],
                                 "resp": [f"[SPEECH]: <d>[Test] {s}</d>" for s in ids]}), sidecars / f"{stem}.parquet")
        clips[stem] = payload
    settings = {"root": str(root), "sidecar": str(side_root), "locales": None, "include": "*.parquet", "workers": 2,
                "delete": False, "dry_run": True, "force": False, "report": str(tmp / "ws" / "migrate.jsonl")}

    def run(**changes) -> int:
        return cmd_migrate(OmegaConf.create({"migrate": {**settings, **changes}}))

    code = run()
    check("migrate dry run writes nothing", code == 0 and not list(shards.glob("*.arrow"))
          and len(list(shards.glob("*.parquet"))) == 5)

    good = pq.read_table(sidecars / "train-00001.parquet")
    bad = good.to_pydict()
    bad["segment_id"] = bad["segment_id"][::-1]
    pq.write_table(pa.table(bad), sidecars / "train-00001.parquet")
    code = run(dry_run=False, delete=True)
    left = sorted(p.name for p in shards.glob("*.parquet"))
    check("migrate deletes only shards whose sidecar matches and that have audio",
          code == 1 and left == ["train-00001.parquet", "train-00002.parquet"]
          and len(list(shards.glob("*.arrow"))) == 5, left=left)
    pq.write_table(good, sidecars / "train-00001.parquet")
    code = run(dry_run=False, delete=True)
    again = run(dry_run=False, delete=True)
    left = sorted(p.name for p in shards.glob("*.parquet"))
    check("migrate resumes; only the shard without audio keeps its parquet",
          code == 0 and again == 0 and left == ["train-00002.parquet"], left=left)
    reports = [json.loads(line) for line in Path(settings["report"]).read_text().splitlines()]
    flagged = {Path(r["file"]).stem: (r["no_audio"], r["null_rows"]) for r in reports}
    check("migrate reports null audio", flagged.get("train-00002") == (True, 2) and flagged.get("train-00003") == (False, 1),
          flagged={k: v for k, v in flagged.items() if k in ("train-00002", "train-00003")})

    same = True
    for stem, want in clips.items():
        table = pa.ipc.open_file(pa.memory_map(str(shards / f"{stem}.arrow"))).read_all()
        same &= table["audio"].to_pylist() == want
        same &= table["segment_id"].to_pylist() == [f"{stem}_seg{i}" for i in range(len(want))]
    check("migrated Arrow = parquet bytes + segment_id", same)

    cfg = OmegaConf.merge(base_cfg, {"data": {"index_root": str(tmp / "index_ws")}})
    cfg.data.sources = [{"name": "ws", "task": "speech", "holdout_ppm": 0, "include": "train-*.parquet",
                         "blobs": str(root), "columns": {"audio": "audio", "caption": "resp", "duration": "duration"},
                         "keep": {"qa_risk_level": ["clean", "medium"]}, "paths": [str(sidecars)]}]
    spec = source_specs(cfg)[0]
    meta = build_source(spec, cfg, workers=2, log=lambda *_: None)
    index = SourceIndex(Path(cfg.data.index_root) / "ws")
    rows = {index.row(i)[1].split("] ")[-1][:-4]: index.row(i)[0] for i in range(len(index))}
    file, row = parse_blob_ref(rows["train-00000_seg3"])
    check("sidecar manifest + Arrow audio: same rows as the parquet source, null clips dropped",
          set(rows) == KEPT_SEGMENTS | {"train-00003_seg0"}
          and file == str(shards / "train-00000.arrow") and row == 3 and meta["drops"].get("no_audio") == 3
          and meta["drops"].get("keep_filter") == 1 and meta["drops"].get("too_short") == 1,
          kept=sorted(rows), drops=meta["drops"])
    wave, rate = read_audio(rows["train-00001_seg1"])
    check("migrated ref decodes to the clip", rate == 24000 and abs(wave.shape[-1] / rate - 4.4) < 0.05,
          seconds=round(wave.shape[-1] / rate, 3))
    check("Arrow files are part of the manifest stamps", any(s[0].endswith("train-00000.arrow") for s in spec.stamps()))

    pq.write_table(pa.table(bad), sidecars / "train-00001.parquet")
    try:
        build_source(spec, cfg, workers=2, force=True, log=lambda *_: None)
        check("sidecar out of step with its Arrow file is rejected", False)
    except ValueError:
        check("sidecar out of step with its Arrow file is rejected", True)
    pq.write_table(good, sidecars / "train-00001.parquet")
    (shards / "train-00001.arrow").rename(shards / "train-00001.arrow.bak")
    try:
        build_source(spec, cfg, workers=2, force=True, log=lambda *_: None)
        check("missing Arrow file is rejected", False)
    except FileNotFoundError:
        check("missing Arrow file is rejected", True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", action="store_true", help="keep the temp dir")
    args = parser.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="chrome_data_check_"))
    try:
        corpus = make_corpus(tmp)
        cfg = make_config(tmp, corpus)
        check_policy()
        check_chunks(corpus)
        metas = check_build(cfg, corpus)
        check_fingerprints(cfg, metas)
        check_holdout(cfg, corpus, tmp)
        check_sampler(cfg)
        check_max_records(cfg, tmp)
        check_min_seconds(cfg, corpus, metas, tmp)
        check_dataset(cfg, corpus)
        check_grouping()
        check_val_groups()
        check_loader(cfg)
        check_embedded(cfg, tmp)
        check_migrated(cfg, tmp)
    finally:
        if args.keep:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL PASSED' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
