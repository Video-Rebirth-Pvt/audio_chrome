"""S2 entry points. Run from the Chrome-10B root (the shell launchers do this):

    torchrun --nproc_per_node 8 -m chrome.train.cli train configs/train/t2a_smoke.yaml [key=value ...]
    python -m chrome.train.cli infer --ckpt outputs/train/t2a_10b --text "Rain on a tin roof." --seconds 10
    python -m chrome.train.cli infer outputs/train/t2a_smoke [infer.steps=50 'infer.prompts=[...]' ...]
    python -m chrome.train.cli text-cache configs/train/t2a_smoke.yaml [text_cache.max_rows=...]
    python -m chrome.train.cli export outputs/train/t2a_10b [--checkpoint latest] [--weights ema]
    python -m chrome.train.cli dump-gt outputs/train/t2a_10b_s20_ws/20261021_210505
    python -m chrome.train.tensorboard outputs/train/t2a_10b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def cmd_train(args) -> int:
    from chrome.config import load_config
    from chrome.train.dist import cleanup
    from chrome.train.trainer import Trainer

    trainer = Trainer(load_config(args.config, args.overrides))
    try:
        trainer.train()
    finally:
        trainer.close()
        cleanup(trainer.env)
    return 0


def _slug(text: str, fallback: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return (slug[:48] or fallback).replace("/", "_")


def resolve_infer_source(run_dir: str | None, ckpt: str | None) -> tuple[Path, str | None]:
    """Run directory plus an optional checkpoint hint (step dir, step number, or None → config)."""
    from chrome.train.checkpoint import TRAINER, find_run_dir

    if ckpt:
        path = Path(ckpt)
        if path.is_file() and path.suffix == ".safetensors":
            path = path.parent
        if path.is_dir() and (path / TRAINER).is_file():
            run = path.parent.parent if path.parent.name == "checkpoints" else path
            return find_run_dir(run), str(path.resolve())
        if ckpt in ("latest",) or str(ckpt).isdigit() or str(path.name).startswith("step_"):
            if not run_dir:
                raise SystemExit(f"--ckpt {ckpt} is a step id; also pass the run dir")
            return find_run_dir(run_dir), str(ckpt)
        return find_run_dir(path), None
    if run_dir:
        return find_run_dir(run_dir), None
    raise SystemExit("pass a run dir or --ckpt")


def _prompts(cfg, spec, captions: list[str] | None = None, name: str | None = None) -> list[tuple[str, str, int]]:
    """infer.prompts / infer.prompt_file / --text → (name, caption, seconds) clamped to the run's length range."""
    infer = cfg["infer"]
    if captions:
        entries = list(captions)
    else:
        entries = list(infer.get("prompts") or [])
        if not entries and infer.get("prompt_file"):
            text = Path(infer["prompt_file"]).read_text()
            entries = json.loads(text) if text.lstrip().startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
    if not entries:
        raise SystemExit("no prompts: --text, infer.prompts=[...], or infer.prompt_file=...")
    lo, hi = int(spec.min_seconds), int(spec.max_seconds)
    items = []
    for i, entry in enumerate(entries):
        if isinstance(entry, str):
            entry = {"caption": entry}
        entry = dict(entry)
        requested = entry.get("seconds") or entry.get("duration") or infer["seconds"]
        seconds = min(hi, max(lo, round(float(requested))))
        if seconds != round(float(requested)):
            print(f"[infer] seconds {requested} clamped to {seconds} (run range {lo}–{hi})", flush=True)
        label = entry.get("name") or entry.get("label")
        if not label:
            if name:
                label = name if len(entries) == 1 else f"{name}_{i:03d}"
            elif captions:
                label = _slug(entry["caption"], f"{i:03d}")
            else:
                label = f"{i:03d}"
        items.append((str(label).replace(" ", "_").replace("/", "_"), entry["caption"], seconds))
    return items


def _infer_overrides(args) -> list[str]:
    extra = list(args.overrides or [])
    if args.prompt_file:
        extra.append(f"infer.prompt_file={args.prompt_file}")
    if args.seconds is not None:
        extra.append(f"infer.seconds={args.seconds}")
    if args.steps is not None:
        extra.append(f"infer.steps={args.steps}")
    if args.cfg_scale is not None:
        extra.append(f"infer.cfg_scale={args.cfg_scale}")
    if args.seed is not None:
        extra.append(f"infer.seed={args.seed}")
    if args.sampler:
        extra.append(f"infer.sampler={args.sampler}")
    if args.weights:
        extra.append(f"infer.weights={args.weights}")
    if args.shift is not None:
        extra.append(f"flow.shift={args.shift}")
    if args.out:
        out = Path(args.out)
        extra.append(f"infer.output_dir={out.parent if out.suffix == '.wav' else out}")
    return extra


def cmd_infer(args) -> int:
    from chrome.config import load_config
    from chrome.data.audio_io import save_audio
    from chrome.pipeline.t2a import T2APipeline

    if args.run_dir and "=" in args.run_dir and not Path(args.run_dir).exists():
        args.overrides = [args.run_dir, *(args.overrides or [])]
        args.run_dir = None
    run_dir, ckpt = resolve_infer_source(args.run_dir, args.ckpt)
    overrides = _infer_overrides(args)
    print(f"[infer] run {run_dir}", flush=True)
    cfg = load_config(run_dir / "config.yaml", overrides)
    infer = cfg["infer"]
    which = ckpt if ckpt is not None else str(infer["checkpoint"])
    weights = str(infer["weights"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = T2APipeline.from_run(run_dir, which, weights, device, overrides)
    out_arg = Path(args.out) if args.out else None
    name = args.name or (out_arg.stem if out_arg is not None and out_arg.suffix == ".wav" else None)
    items = _prompts(cfg, pipe.spec, captions=list(args.text) if args.text else None, name=name)
    names, captions, seconds = zip(*items)
    started = time.perf_counter()
    waves = pipe.generate(
        list(captions), list(seconds), int(infer["steps"]), float(infer["cfg_scale"]), int(infer["seed"]),
        batch=int(infer["batch"]), sampler=str(infer.get("sampler", "euler")),
    )
    out = Path(infer["output_dir"]) if infer.get("output_dir") else run_dir / "infer"
    for name, caption, wave in zip(names, captions, waves):
        save_audio(out / f"{name}.wav", wave, pipe.spec.sample_rate)
        (out / f"{name}.txt").write_text(caption + "\n")
    print(f"[infer] {len(waves)} clips in {time.perf_counter() - started:.0f}s → {out}")
    return 0


def cmd_text_cache(args) -> int:
    """Encode every caption a run will need (train subset, holdout, preview, " ") and report hidden stats."""
    import numpy as np

    from chrome.config import load_config
    from chrome.data.captions import NULL_CAPTION, caption_ids, load_caption_tokenizer
    from chrome.data.loader import open_sources, train_rows
    from chrome.text.base import build_text_encoder
    from chrome.text.cache import CachedTextEncoder

    cfg = load_config(args.config, args.overrides)
    tc = cfg["text_cache"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = build_text_encoder(cfg["text"], device=device)
    if not isinstance(encoder, CachedTextEncoder) or encoder.dir is None:
        raise SystemExit("text.cache.dir is not set; nothing to fill")
    tokenizer = load_caption_tokenizer(cfg["data"]["tokenizer"])
    subset = cfg["data"]["loader"].get("subset")
    max_rows = tc.get("max_rows")
    captions = [NULL_CAPTION]
    _, indexes = open_sources(cfg)
    for index in indexes:
        rows = list(train_rows(index, None if subset is None else int(subset)))
        rows += list(np.flatnonzero(index.holdout)[np.argsort(index.keys()[index.holdout], kind="stable")][: int(cfg["data"]["val"]["per_source"])])
        if max_rows:
            rows = rows[: int(max_rows)]
        captions += [index.row(int(r))[1] for r in rows]
    if cfg["preview"].get("set"):
        captions += [e["caption"] for e in json.loads(Path(cfg["preview"]["set"]).read_text())]
    ids = [torch.tensor(caption_ids(tokenizer, c), dtype=torch.long) for c in dict.fromkeys(captions)]
    todo = [i for i in ids if not encoder.contains(i)]
    print(f"[text-cache] {len(ids)} unique captions, {len(todo)} to encode → {encoder.dir}")
    started = time.perf_counter()
    batch = int(tc["batch"])
    for start in range(0, len(todo), batch):
        encoder.encode(todo[start : start + batch])
        print(f"[text-cache] {min(start + batch, len(todo))}/{len(todo)} ({time.perf_counter() - started:.0f}s)", flush=True)
    write_text_stats(encoder, ids[1 : 1 + int(tc["stats_captions"])])
    return 0


def write_text_stats(encoder, ids: list[torch.Tensor]) -> None:
    """Per-dim mean / std over every token of ``ids`` (token 0 included, so its massive activation widens the
    std of its dims instead of exploding after normalisation) → ``<cache dir>/stats.safetensors`` for
    ``dit.text_norm=stats``. Also prints how extreme the raw hidden states are."""
    from safetensors.torch import save_file

    from chrome.tasks.t2a import TEXT_STATS_FILE

    if not ids:
        return
    hidden = [h.float() for h in encoder.encode(ids)]
    rows = torch.cat(hidden)
    first = torch.stack([h[0] for h in hidden])
    rest = torch.cat([h[1:] for h in hidden if h.shape[0] > 1])
    mean, std = rows.mean(dim=0), rows.std(dim=0)
    save_file({"mean": mean.contiguous(), "std": std.contiguous()}, str(encoder.dir / TEXT_STATS_FILE))
    absmax = rows.abs().amax(dim=0)
    top = torch.topk(absmax, 8)
    row_rms = lambda x: x.pow(2).mean(dim=1).sqrt().median().item()  # noqa: E731
    print(
        f"[text-stats] {len(ids)} captions, {rows.shape[0]} tokens: row RMS median token 0 {row_rms(first):.1f}, "
        f"others {row_rms(rest):.2f}; per-dim std median {std.median():.2f} max {std.max():.1f}; |x| max {absmax.max():.0f}"
    )
    print("[text-stats] largest |x| dims: " + ", ".join(f"{d}:{v:.0f}" for v, d in zip(top.values.tolist(), top.indices.tolist())))
    print(f"[text-stats] → {encoder.dir / TEXT_STATS_FILE}")


def cmd_dump_gt(args) -> int:
    """Trace preview samples back to the original dataset clips and write them once."""
    from chrome.config import load_config
    from chrome.data.preview_gt import dump_preview_gt
    from chrome.spec import AudioSpec
    from chrome.train.checkpoint import find_run_dir

    src = Path(args.source)
    if src.is_file() and src.suffix in {".yaml", ".yml"}:
        cfg = load_config(src, args.overrides)
        out = Path(args.out) if args.out else Path(cfg["train"]["output_dir"]) / "samples" / "gt"
        print(f"[preview-gt] config {src}", flush=True)
    else:
        run_dir = find_run_dir(src)
        cfg = load_config(run_dir / "config.yaml", args.overrides)
        out = Path(args.out) if args.out else run_dir / "samples" / "gt"
        print(f"[preview-gt] run {run_dir}", flush=True)
    spec = AudioSpec.from_config(cfg)
    started = time.perf_counter()
    records = dump_preview_gt(cfg, spec, out, force=bool(args.force))
    wrote = sum(1 for r in records if r["status"] == "wrote")
    kept = sum(1 for r in records if r["status"] == "exists")
    bad = [r for r in records if r["status"] not in ("wrote", "exists")]
    print(
        f"[preview-gt] {len(records)} clips ({wrote} wrote, {kept} existed) in "
        f"{time.perf_counter() - started:.0f}s → {out}",
        flush=True,
    )
    for rec in bad:
        print(f"[preview-gt] {rec['name']}: {rec['status']}", flush=True)
    return 1 if bad else 0


def cmd_export(args) -> int:
    """Sharded ``state/`` → one fp32 safetensors file, for sharing or other tools (inference reads both)."""
    from chrome.config import load_config
    from chrome.train.checkpoint import build_dit, export_weights, find_run_dir, resolve_checkpoint

    run_dir = find_run_dir(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    path = resolve_checkpoint(run_dir, args.checkpoint)
    with torch.device("meta"):
        model = build_dit(cfg)
    model.to_empty(device="cpu")
    started = time.perf_counter()
    out = export_weights(path, model, args.weights, args.out)
    size = sum(p.numel() for p in model.parameters())
    print(f"[export] {size / 1e9:.3f}B params → {out} ({time.perf_counter() - started:.0f}s)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="chrome.train.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("config")
    p.add_argument("overrides", nargs="*")
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("infer", help="text → wav from a training run (single GPU)")
    p.add_argument("run_dir", nargs="?", default=None, help="run or family dir; optional if --ckpt is a path")
    p.add_argument("overrides", nargs="*", help="OmegaConf key=value, e.g. infer.steps=50")
    p.add_argument("--ckpt", default=None, help="run dir, step dir, step number, latest, or ema.safetensors")
    p.add_argument("--text", action="append", default=None, help="caption; repeat for several clips")
    p.add_argument("--prompt-file", default=None, help="JSON / JSONL of captions")
    p.add_argument("--seconds", type=float, default=None, help="clip length; clamped to the run's min/max")
    p.add_argument("--steps", type=int, default=None, help="ODE steps (default 50)")
    p.add_argument("--cfg", "--cfg-scale", dest="cfg_scale", type=float, default=None, help="CFG scale (default 4.0)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--sampler", default=None, help="euler")
    p.add_argument("--weights", choices=["ema", "model"], default=None)
    p.add_argument("--shift", type=float, default=None, help="flow shift (default 3.0)")
    p.add_argument("--name", default=None, help="output stem; default is a slug of the caption")
    p.add_argument("--out", default=None, help="output directory, or a .wav path for a single clip")
    p.set_defaults(func=cmd_infer)
    p = sub.add_parser("text-cache")
    p.add_argument("config")
    p.add_argument("overrides", nargs="*")
    p.set_defaults(func=cmd_text_cache)
    p = sub.add_parser("export")
    p.add_argument("run_dir")
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--weights", default="ema", choices=["ema", "model"])
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_export)
    p = sub.add_parser("dump-gt", help="write original preview-set clips once (same names as samples/step_*)")
    p.add_argument("source", help="run dir, family dir, or a train yaml")
    p.add_argument("overrides", nargs="*", help="OmegaConf key=value")
    p.add_argument("--out", default=None, help="default: <run_dir>/samples/gt")
    p.add_argument("--force", action="store_true", help="overwrite existing gt wavs")
    p.set_defaults(func=cmd_dump_gt)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
