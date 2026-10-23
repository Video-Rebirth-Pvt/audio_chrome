"""Data pipeline tools: repack embedded audio, build the per-source indexes, inspect the mixture, benchmark the loader.

Run from the Chrome-10B root (the shell launchers do this):
    bash scripts/data/repack_audio.sh --config configs/data/t2a_worldspeech.yaml
    bash scripts/data/migrate_audio.sh migrate.locales=[yue_hk]
    bash scripts/data/build_index.sh                       # all sources; unchanged ones are skipped
    bash scripts/data/build_index.sh data.only=[music] data.build.force=true
    bash scripts/data/inspect_data.sh inspect.world_size=64
    CUDA_VISIBLE_DEVICES=0 bash scripts/data/bench_loader.sh bench.vae=true
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from chrome.config import load_config
from chrome.data.index import META_FILE, SourceIndex, build_source, source_specs

ROOT = Path(__file__).resolve().parents[2]


def _log(message: str) -> None:
    print(message, flush=True)


def _selected(cfg):
    specs = source_specs(cfg)
    only = cfg["data"].get("only")
    if only:
        names = {str(n) for n in only}
        unknown = names - {s.name for s in specs}
        if unknown:
            raise SystemExit(f"unknown sources in data.only: {sorted(unknown)}")
        specs = [s for s in specs if s.name in names]
    return specs


def cmd_repack(cfg) -> int:
    import pyarrow.parquet as pq

    from chrome.data.blobs import blob_path, embedded_audio_column, repack
    from chrome.data.manifest import manifest_files

    settings = cfg["data"]["repack"]
    started = time.monotonic()
    for spec in _selected(cfg):
        if not spec.blobs:
            continue
        files = [f for p in spec.paths for f in manifest_files(p, spec.include) if f.suffix == ".parquet"]
        files = [f for f in files if embedded_audio_column(pq.ParquetFile(f).schema_arrow) is not None]
        _log(f"[repack] {spec.name}: {len(files)} parquet file(s) with embedded audio → {spec.blobs}")
        result = repack([(f, blob_path(spec.blobs, f)) for f in files], int(settings["workers"]),
                        bool(settings["force"]), log=_log)
        _log(f"[repack] {spec.name}: {result['files'] - result['skipped']} written, {result['skipped']} up to date")
    print(f"[repack] done in {time.monotonic() - started:.0f}s")
    return 0


def cmd_migrate(cfg) -> int:
    from chrome.data.blobs import BLOB_SUFFIX, blob_path, migrate
    from chrome.data.manifest import sidecar_path

    settings = cfg["migrate"]
    root = Path(settings["root"])
    sidecar = settings.get("sidecar")
    delete, dry_run = bool(settings["delete"]), bool(settings["dry_run"])
    if delete and not sidecar:
        raise SystemExit("migrate.delete needs migrate.sidecar: the sidecar keeps every non-audio column")
    locales = [str(n) for n in settings.get("locales") or []] or sorted(p.name for p in root.iterdir() if p.is_dir())
    jobs, migrated = [], 0
    for locale in locales:
        folder = root / locale
        if not folder.is_dir():
            raise SystemExit(f"no locale dir {folder}")
        shards = sorted(folder.glob(str(settings["include"])))
        parquets = [f for f in shards if f.suffix == ".parquet"]
        stems = {f.stem for f in parquets}
        migrated += sum(1 for f in folder.glob(f"*{BLOB_SUFFIX}") if f.stem not in stems)
        jobs += [
            {
                "parquet": str(f),
                "blob": str(blob_path(root, f)),
                "sidecar": str(sidecar_path(sidecar, f)) if sidecar else None,
                "delete": delete,
                "dry_run": dry_run,
                "force": bool(settings["force"]),
            }
            for f in parquets
        ]
    mode = "dry run" if dry_run else ("repack + delete parquet" if delete else "repack only")
    _log(f"[migrate] {len(locales)} locale(s), {len(jobs)} parquet shard(s) to process, {migrated} already "
         f"migrated (Arrow without parquet); mode: {mode}")
    if not jobs:
        return 0
    result = migrate(jobs, int(settings["workers"]), settings.get("report"), log=_log)
    if dry_run:
        _log(f"[migrate] dry run: {result['repacked']} shard(s) need a repack, {result['files'] - result['repacked']} "
             f"current; parquet total {result['bytes'] / 2**30:.1f} GiB" + (" would be deleted" if delete else ""))
    else:
        _log(f"[migrate] {result['repacked']} repacked ({result['bytes'] / 2**30:.1f} GiB), {result['deleted']} parquet "
             f"deleted ({result['freed'] / 2**30:.1f} GiB freed), {result['failed']} failed"
             + (f"; log: {settings['report']}" if settings.get("report") else ""))
    if result["nulls"]:
        _log(f"[migrate] null audio: {result['no_audio']} shard(s) without any audio keep their parquet; "
             f"{result['null_rows']} null clip(s) in all, dropped as no_audio at index build")
        for locale, (empty, partial, clips) in result["nulls"].items():
            _log(f"[migrate]   {locale}: {empty} shard(s) without audio, {partial} with some null clips, {clips} null clips")
    for failure in result["failures"][:20]:
        _log(f"[migrate]   {failure['file']}: {failure['error']}")
    return 1 if result["failed"] else 0


def cmd_build(cfg) -> int:
    build = cfg["data"]["build"]
    started = time.monotonic()
    for spec in _selected(cfg):
        build_source(spec, cfg, workers=int(build["workers"]), force=bool(build["force"]), log=_log)
    print(f"[index] done in {time.monotonic() - started:.0f}s → {cfg['data']['index_root']}")
    return 0


def _table(rows: list[list], header: list[str]) -> str:
    cells = [header] + [[str(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    lines = ["  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths))) for r in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def cmd_inspect(cfg) -> int:
    data, inspect = cfg["data"], cfg["inspect"]
    root = Path(data["index_root"])
    specs = _selected(cfg)
    metas = {}
    for spec in specs:
        path = root / spec.name / META_FILE
        if not path.is_file():
            print(f"[inspect] {spec.name}: no index yet (bash scripts/data/build_index.sh)")
            continue
        metas[spec.name] = json.loads(path.read_text())
    specs = [s for s in specs if s.name in metas]
    if not specs:
        return 1

    rows = []
    for spec in specs:
        meta = metas[spec.name]
        train = meta["train"]
        rates = train.get("sample_rate_hist", {})
        low = sum(c for r, c in rates.items() if 0 < int(r) <= 24000)
        tokens = train.get("text_tokens", {})
        rows.append([
            spec.name, spec.task, meta["scanned"], meta["train_rows"], meta["holdout_rows"],
            f"{train.get('clip_hours', 0):.0f}", f"{train.get('mean_seconds', 0):.1f}",
            f"{tokens.get('p50', 0)}/{tokens.get('p95', 0)}/{tokens.get('max', 0)}",
            f"{low / max(meta['train_rows'], 1):.0%}",
            f"{train.get('padded_rows', 0) / max(meta['train_rows'], 1):.0%}",
        ])
    print(_table(rows, ["source", "task", "scanned", "train", "holdout", "hours", "mean s",
                        "tokens p50/p95/max", "≤24k sr", "padded"]))

    print("\nDropped rows (reason: count)")
    for spec in specs:
        drops = ", ".join(f"{k}: {v}" for k, v in metas[spec.name]["drops"].items()) or "-"
        print(f"  {spec.name}: {drops}")

    print("\nLength histogram of train clips (seconds: share)")
    for spec in specs:
        hist = metas[spec.name]["train"].get("seconds_hist", {})
        total = max(sum(hist.values()), 1)
        print(f"  {spec.name}: " + " ".join(f"{s}:{c / total:.0%}" for s, c in sorted(hist.items(), key=lambda kv: int(kv[0]))))

    from chrome.data.index import mixture_weight
    from chrome.data.loader import rows_per_second
    from chrome.spec import AudioSpec

    train_rows = np.array([metas[s.name]["train_rows"] for s in specs], dtype=np.float64)
    weights = np.array([mixture_weight(s, int(n)) for s, n in zip(specs, train_rows)], dtype=np.float64)
    if weights.sum() <= 0:
        print("\nAll mixture weights are 0; nothing would be sampled.")
        return 0
    share = weights / weights.sum()
    mean_s = np.array([metas[s.name]["train"].get("mean_seconds", 0.0) for s in specs])
    mean_t = np.array([metas[s.name]["train"].get("text_tokens", {}).get("mean", 0.0) for s in specs])

    cost = rows_per_second(AudioSpec.from_config(cfg)) * mean_s + mean_t
    per_batch = int(data["loader"]["max_tokens"]) / float((share * cost).sum())
    per_step = per_batch * int(inspect["world_size"]) * int(inspect["grad_accum"])
    steps = int(inspect["steps"])
    hours = share * mean_s
    rows = []
    for i, spec in enumerate(specs):
        epochs = share[i] * per_step * steps / max(train_rows[i], 1)
        caps = ([f"{spec.max_records:g}"] if spec.max_records is not None else []) + (
            [f"{spec.max_hours:g} h"] if spec.max_hours is not None else [])
        cap = " / ".join(caps) or "all"
        mix = "rows" if spec.weight is None else f"{spec.weight:g}"
        rows.append([spec.name, cap, mix, f"{share[i]:.1%}", f"{hours[i] / hours.sum():.1%}",
                     f"{share[i] * cost[i] / (share * cost).sum():.1%}", f"{epochs:.2f}"])
    print(f"\nMixture: ≈{per_batch:.1f} clips per batch of {data['loader']['max_tokens']} tokens, "
          f"{per_step:.0f} clips per step (world {inspect['world_size']} × accum {inspect['grad_accum']})")
    print(_table(rows, ["source", "cap", "weight", "clips", "audio hours", "tokens", f"epochs / {steps} steps"]))

    if inspect.get("duplicates", True) and len(specs) > 1:
        keys = {s.name: SourceIndex(root / s.name).keys() for s in specs}
        pairs = []
        for i, a in enumerate(specs):
            for b in specs[i + 1 :]:
                n = np.intersect1d(keys[a.name], keys[b.name], assume_unique=True).size
                if n:
                    pairs.append(f"{a.name}∩{b.name}: {n}")
        print("\nSame audio path in two sources: " + (", ".join(pairs) if pairs else "none"))
    return 0


def cmd_bench(cfg) -> int:
    import torch

    from chrome.data.audio_io import save_audio
    from chrome.data.batch import encode_waves
    from chrome.data.loader import build_train_loader, rows_per_second
    from chrome.spec import AudioSpec

    bench = cfg["bench"]
    if int(bench["batches"]) <= int(bench["warmup"]):
        raise SystemExit("bench.batches must exceed bench.warmup")
    per_second = rows_per_second(AudioSpec.from_config(cfg))
    loader, sampler = build_train_loader(cfg, int(bench["rank"]), int(bench["world_size"]))
    names = [s.name for s in sampler.streams]
    weights = np.array([s.weight if len(s.rows) else 0.0 for s in sampler.streams])
    expected = weights / weights.sum()
    vae = None
    if bench["vae"]:
        from chrome.codec import load_h3_audio_vae

        vae = load_h3_audio_vae(bench["vae_checkpoint"], device=bench["device"])
    out = Path(bench["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    dump = int(bench["dump"])
    total, warmup = int(bench["batches"]), int(bench["warmup"])
    max_tokens = int(cfg["data"]["loader"]["max_tokens"])
    seen: Counter = Counter()
    seconds_hist: Counter = Counter()
    clips = tokens = audio_seconds = dropped = 0
    encode_time = 0.0
    dumped = 0
    started = time.monotonic()
    first_batch = None
    for step, batch in enumerate(loader, start=1):
        if step == 1:
            first_batch = time.monotonic() - started
        if step == warmup + 1:
            started_timed = time.monotonic()
            clips = tokens = audio_seconds = dropped = 0
            encode_time = 0.0
        seconds = batch["seconds"]
        clips += len(seconds)
        audio_seconds += int(seconds.sum())
        tokens += int((seconds * per_second + batch["text_len"]).sum())
        dropped += int(batch["cfg_dropped"].sum())
        seen.update(names[i] for i in batch["source"].tolist())
        seconds_hist.update(seconds.tolist())
        if vae is not None:
            torch.cuda.synchronize()
            t = time.monotonic()
            encode_waves(vae, batch["waves"], bench["device"], allow_tf32=bool(bench["allow_tf32"]))
            torch.cuda.synchronize()
            encode_time += time.monotonic() - t
        for i in range(len(batch["waves"])):
            if dumped >= dump:
                break
            name = f"{dumped:02d}_{names[int(batch['source'][i])]}_{int(seconds[i])}s"
            save_audio(out / f"{name}.wav", batch["waves"][i], cfg["audio"]["sample_rate"])
            (out / f"{name}.txt").write_text(f"{batch['paths'][i]}\n{batch['captions'][i]}\n", encoding="utf-8")
            dumped += 1
        if step % 20 == 0:
            print(f"[bench] {step}/{total} batches", flush=True)
        if step >= total:
            break
    elapsed = time.monotonic() - started_timed
    timed = total - warmup
    print(f"\n[bench] first batch after {first_batch:.1f}s (workers start + index open)")
    print(f"[bench] {timed} timed batches in {elapsed:.1f}s: {timed / elapsed:.2f} batches/s, "
          f"{clips / elapsed:.1f} clips/s, {audio_seconds / elapsed:.0f} audio-s/s, {tokens / elapsed:.0f} rows/s")
    print(f"[bench] per batch: {clips / timed:.1f} clips, {tokens / timed:.0f} tokens "
          f"({tokens / timed / max_tokens:.0%} of max_tokens={max_tokens}); caption dropout {dropped / max(clips, 1):.1%}")
    total_seen = max(sum(seen.values()), 1)
    print("[bench] source share seen / expected: " + ", ".join(
        f"{n} {seen[n] / total_seen:.1%}/{expected[i]:.1%}" for i, n in enumerate(names)))
    hist_total = max(sum(seconds_hist.values()), 1)
    print("[bench] clip seconds: " + " ".join(f"{s}:{c / hist_total:.0%}" for s, c in sorted(seconds_hist.items())))
    if vae is not None:
        print(f"[bench] VAE encode {encode_time / timed * 1000:.0f} ms/batch, "
              f"{encode_time / max(audio_seconds, 1) * 1000:.1f} ms per clip-second (tf32={bool(bench['allow_tf32'])})")
    print(f"[bench] {dumped} clips written to {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["repack", "migrate", "build", "inspect", "bench"])
    parser.add_argument("--config", default="configs/data/t2a_base.yaml")
    parser.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. data.only=[music]")
    args = parser.parse_intermixed_args()
    cfg = load_config(ROOT / args.config, args.overrides)
    commands = {"repack": cmd_repack, "migrate": cmd_migrate, "build": cmd_build, "inspect": cmd_inspect,
                "bench": cmd_bench}
    return commands[args.command](cfg)


if __name__ == "__main__":
    sys.exit(main())
