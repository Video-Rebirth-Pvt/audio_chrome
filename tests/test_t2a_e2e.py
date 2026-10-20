"""End-to-end check of the S2 T2A chain on the 500-clip smoke subset (real Qwen text cache, frozen H3 VAE,
tiny DiT), through the real launchers. Needs 8 idle GPUs (``--gpus`` to choose), ~10 minutes.

Run from anywhere:
    bash scripts/train/check_t2a_smoke.sh
    bash scripts/train/check_t2a_smoke.sh --gpus 0,1,2,3,4,5,6,7 --keep

Checks:
  1. text cache filled for every caption of the run, text stats written (stats.safetensors)
  2. DDP and FSDP from the same initial weights follow the same loss curve (bf16 tolerance)
  3. FSDP stopped at step 10 and resumed matches the uninterrupted run: same batches, same losses
  4. HSDP (4 ranks, shard groups of 2) trains; text_norm=stats trains
  5. validation / preview under FSDP (uneven holdout and preview shares across ranks do not hang)
  6. inference from the sharded checkpoint = inference from the exported safetensors; wav shapes
Exits 1 if any check fails. Runs live under outputs/check/t2a_e2e/ (removed unless --keep).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIG = "configs/train/t2a_smoke.yaml"
OUT = ROOT / "outputs/check/t2a_e2e"
FAILED: list[str] = []
COMMON = [
    "train.resume=auto", "train.max_steps=20", "train.log_every=2", "train.val_every=10", "train.val_batches=8",
    "train.preview_every=0", "train.ckpt_every=10", "train.keep_ckpts=3", "train.warmup_steps=5",
    "data.loader.num_workers=4", "data.val.num_workers=2",
]


def check(name: str, passed: bool, **details) -> bool:
    shown = ", ".join(f"{k}={v}" for k, v in details.items())
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  ({shown})" if shown else ""), flush=True)
    if not passed:
        FAILED.append(name)
    return bool(passed)


def launch(log: str, gpus: list[str], script: str, args: list[str]) -> subprocess.Popen:
    out = (OUT / f"{log}.log").open("w")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
    print(f"[run] {log} on GPU {','.join(gpus)}: {script} {' '.join(args)}", flush=True)
    return subprocess.Popen(["bash", script, *args], cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)


def train(run: str, gpus: list[str], *overrides: str, log: str | None = None) -> subprocess.Popen:
    return launch(log or run, gpus, "scripts/train/train_t2a.sh",
                  [CONFIG, f"train.output_dir={OUT / run}", *COMMON, *overrides])


def wait(*procs: tuple[str, subprocess.Popen]) -> bool:
    ok = True
    for name, proc in procs:
        code = proc.wait()
        if code != 0:
            print(f"[run] {name} exited {code}; tail of {OUT / f'{name}.log'}:")
            print("".join((OUT / f"{name}.log").read_text().splitlines(keepends=True)[-30:]))
            ok = False
    return ok


def events(name: str, kind: str = "train") -> dict[int, dict]:
    path = OUT / name / "log.jsonl"
    if not path.is_file():
        return {}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["step"]: r for r in records if r.get("event") == kind}


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-8)


def make_init() -> Path:
    """One set of initial weights for every run that compares strategies (FSDP init draws differently)."""
    import torch
    from safetensors.torch import save_file

    from chrome.config import load_config
    from chrome.train.checkpoint import build_dit

    cfg = load_config(ROOT / CONFIG)
    torch.manual_seed(7)
    model = build_dit(cfg)
    model.init_weights()
    # Open the zero-initialised gates and heads a little, so the curves depend on every weight.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "adaln" in name or "audio_out" in name:
                p.normal_(std=1e-3)
    path = OUT / "init.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(gpus) < 8:
        raise SystemExit("needs 8 GPUs (four 2-rank runs side by side, then a 4-rank HSDP run)")
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    started = time.perf_counter()

    # 1. text cache + stats
    if not wait(("cache", launch("cache", gpus[:1], "scripts/text/cache_text.sh", [CONFIG]))):
        FAILED.append("text cache")
        return 1
    cache_log = (OUT / "cache.log").read_text()
    stats_line = next((l for l in cache_log.splitlines() if l.startswith("[text-stats] →")), "")
    check("text cache filled, stats.safetensors written", bool(stats_line) and Path(stats_line.split("→ ")[1]).is_file())
    init = make_init()

    # 2–5. four 2-rank runs side by side, then resume and HSDP
    first = [
        ("ddp", train("ddp", gpus[0:2], f"train.init_from={init}", "train.init_weights=model")),
        ("fsdp", train("fsdp", gpus[2:4], f"train.init_from={init}", "train.init_weights=model",
                       "train.parallel.kind=fsdp", "train.preview_every=20", "preview.steps=5")),
        ("fsdp_cut", train("fsdp_cut", gpus[4:6], f"train.init_from={init}", "train.init_weights=model",
                           "train.parallel.kind=fsdp", "train.max_steps=10")),
        ("stats", train("stats", gpus[6:8], "train.parallel.kind=fsdp", "dit.text_norm=stats", "dit.text_clip=8",
                        "train.max_steps=10")),
    ]
    ok = wait(*first)
    second = [
        ("fsdp_cut_resume", train("fsdp_cut", gpus[4:6], f"train.init_from={init}", "train.init_weights=model",
                                  "train.parallel.kind=fsdp", log="fsdp_cut_resume")),
        ("hsdp", train("hsdp", gpus[0:4], "train.parallel.kind=fsdp", "train.parallel.shard_size=2", "train.max_steps=10")),
    ]
    ok = wait(*second) and ok
    if not ok:
        FAILED.append("training runs")

    ddp, fsdp = events("ddp"), events("fsdp")
    steps = sorted(set(ddp) & set(fsdp))
    worst = max((rel(fsdp[s]["loss"], ddp[s]["loss"]) for s in steps), default=math.inf)
    check("DDP = FSDP from the same init (loss per logged step, bf16 tolerance)", len(steps) == 10 and worst < 5e-3,
          steps=len(steps), worst_rel=f"{worst:.1e}", ddp_last=ddp.get(20, {}).get("loss"), fsdp_last=fsdp.get(20, {}).get("loss"))
    dv, fv = events("ddp", "val"), events("fsdp", "val")
    check("DDP = FSDP validation loss", 20 in dv and 20 in fv and rel(fv[20]["loss"], dv[20]["loss"]) < 5e-3,
          ddp=dv.get(20, {}).get("loss"), fsdp=fv.get(20, {}).get("loss"))

    cut = events("fsdp_cut")
    later = [s for s in sorted(cut) if s > 10]
    same_batches = all(cut[s]["batch_id"] == fsdp[s]["batch_id"] for s in later if s in fsdp)
    worst = max((rel(cut[s]["loss"], fsdp[s]["loss"]) for s in later if s in fsdp), default=math.inf)
    resumed = "[resume]" in (OUT / "fsdp_cut_resume.log").read_text()
    check("FSDP resume at step 10 replays the uninterrupted run", resumed and len(later) == 5 and same_batches and worst < 2e-3,
          batches_equal=same_batches, worst_rel=f"{worst:.1e}")

    for name in ("hsdp", "stats"):
        run = events(name)
        losses = [run[s]["loss"] for s in sorted(run)]
        check(f"{name} trains (finite, decreasing loss)", len(losses) == 5 and all(map(math.isfinite, losses)) and losses[-1] < losses[0],
              first=losses[0] if losses else None, last=losses[-1] if losses else None)
    hsdp_log = (OUT / "hsdp.log").read_text()
    check("HSDP mesh 2×2", "hsdp 2×2" in hsdp_log)

    samples = sorted((OUT / "fsdp/samples/step_0000020").glob("*.wav"))
    check("FSDP preview: every item rendered once across ranks (9 items, 2 ranks)", len(samples) == 9, wavs=len(samples))

    # 6. inference from state/ vs exported safetensors
    prompt = 'infer.prompts=[{caption: "Heavy rain on a tin roof.", seconds: 4, name: rain}]'
    a = launch("infer_state", gpus[:1], "scripts/infer/infer_t2a.sh",
               [str(OUT / "fsdp"), "infer.steps=5", prompt, f"infer.output_dir={OUT / 'infer_state'}"])
    ok = wait(("infer_state", a))
    ok = wait(("export", launch("export", gpus[:1], "scripts/train/export_t2a.sh", [str(OUT / "fsdp")]))) and ok
    b = launch("infer_export", gpus[:1], "scripts/infer/infer_t2a.sh",
               [str(OUT / "fsdp"), "infer.steps=5", prompt, f"infer.output_dir={OUT / 'infer_export'}"])
    ok = wait(("infer_export", b)) and ok
    if ok:
        import soundfile as sf

        x, sr = sf.read(OUT / "infer_state/rain.wav")
        y, _ = sf.read(OUT / "infer_export/rain.wav")
        exported = list((OUT / "fsdp/checkpoints").glob("step_*/ema.safetensors"))
        check("inference: sharded state = exported weights, [N, 2] at 32 kHz", x.shape == (128000, 2) and sr == 32000
              and (x == y).all() and bool(exported), shape=x.shape)
    else:
        FAILED.append("inference / export")

    print(f"\n{time.perf_counter() - started:.0f}s; {'ALL PASSED' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
    if not args.keep and not FAILED:
        shutil.rmtree(OUT, ignore_errors=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
