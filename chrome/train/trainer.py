"""Task-agnostic trainer: DDP or FSDP (``train.parallel``), bf16 autocast over fp32 master weights, AdamW, EMA,
resumable data stream. What the batches are, the loss and the previews come from the task (``task.kind``).

One step = ``grad_accum`` packed micro-batches per rank. Every ``log_every`` steps: loss (overall and per
group, averaged over all ranks), LR, grad norm (total and per task group), throughput, and seconds per
micro-batch in loading (data), conditioning (cond, split by the task's stages) and forward + backward (model).
``val_every``: loss on the source-balanced holdout set at fixed σ / noise. ``preview_every``: every rank renders its
share of the preview items. ``ckpt_every``: sharded weights, EMA, optimizer and every rank's sampler state,
so ``train.resume=auto`` continues with the next batch (at any world size; the data stream restarts if the
world size changed).
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from chrome.tasks.base import Generators, build_task
from chrome.train import checkpoint as ckpt
from chrome.train.dist import DistEnv, all_reduce_max, all_reduce_sum, barrier, init_distributed
from chrome.train.parallel import EMA, build_parallel
from chrome.train.tensorboard import TensorBoardLogger


class Meter:
    """Sums and counts per metric; ``means`` averages them over every rank (collective when distributed)."""

    def __init__(self):
        self.sums: dict[str, float] = defaultdict(float)
        self.counts: dict[str, float] = defaultdict(float)

    def add(self, name: str, value: float, n: float = 1.0) -> None:
        if math.isfinite(value):
            self.sums[name] += value * n
            self.counts[name] += n

    def means(self, env: DistEnv | None = None) -> dict[str, float]:
        keys = sorted(self.sums)
        if env is not None and env.enabled:
            gathered: list[list[str]] = [[] for _ in range(env.world_size)]
            dist.all_gather_object(gathered, keys)
            keys = sorted(set().union(*gathered))
            values = torch.tensor(
                [self.sums.get(k, 0.0) for k in keys] + [self.counts.get(k, 0.0) for k in keys],
                device=env.device, dtype=torch.float64,
            )
            all_reduce_sum(env, values)
            sums, counts = values[: len(keys)].tolist(), values[len(keys) :].tolist()
        else:
            sums, counts = [self.sums[k] for k in keys], [self.counts[k] for k in keys]
        return {k: s / c for k, s, c in zip(keys, sums, counts) if c > 0}


def lr_lambda(warmup: int, total: int, schedule: str, min_ratio: float):
    def fn(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if schedule == "constant":
            return 1.0
        if schedule == "cosine":
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"unknown lr schedule {schedule!r}")

    return fn


def _numel(p: torch.Tensor) -> int:
    return p.numel()  # global size, also for DTensor shards


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.t = t = cfg["train"]
        self.env: DistEnv = init_distributed(float(t.get("dist_timeout_minutes", 30)))
        self.device = self.env.device
        self.seed = int(t["seed"])
        torch.backends.cuda.matmul.allow_tf32 = bool(t["allow_tf32"])
        torch.backends.cudnn.allow_tf32 = bool(t["allow_tf32"])

        decided = None
        if self.env.is_main:
            decided = str(
                ckpt.resolve_run_dir(Path(t["output_dir"]), t.get("resume"), bool(t.get("stamp_run", False)))
            )
        if self.env.enabled:
            payload = [decided]
            dist.broadcast_object_list(payload, src=0)
            decided = payload[0]
        self.run_dir = Path(decided)
        OmegaConf.update(cfg, "train.output_dir", str(self.run_dir))
        if ckpt.STAMP_RE.match(self.run_dir.name):
            OmegaConf.update(cfg, "train.started_at", self.run_dir.name)
        if self.env.is_main:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(cfg, self.run_dir / "config.yaml")
        barrier(self.env)
        self._log_file = (self.run_dir / "log.jsonl").open("a") if self.env.is_main else None
        self._tb = TensorBoardLogger(
            self.run_dir,
            enabled=self.env.is_main and bool(t.get("tensorboard", True)),
            audio=bool(t.get("tensorboard_audio", True)),
            log=self.log,
        )

        torch.manual_seed(self.seed)
        self.task = build_task(cfg, self.device, log=self.log)
        self.parallel = build_parallel(t.get("parallel"), self.env, self.seed)
        started = time.perf_counter()
        with torch.device("meta"):
            model = self.task.build_model()
        self.module = self.parallel.setup(model, self.task.init_model)
        self.model = model
        torch.manual_seed(self.seed + self.env.rank)
        self.params = [p for p in model.parameters() if p.requires_grad]
        self._log_model(time.perf_counter() - started)

        self.step = 0
        resume_from = self._resume_path()
        if resume_from is None and t.get("init_from"):
            self._init_from(str(t["init_from"]))
        self.ema = EMA(model, float(t["ema_decay"])) if float(t["ema_decay"]) > 0 else None
        self.optimizer = self._optimizer()
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda(int(t["warmup_steps"]), int(t["max_steps"]), str(t["lr_schedule"]), float(t["min_lr_ratio"])),
        )
        data_state = None
        if resume_from is not None:
            ckpt.load_state(resume_from, model, self.optimizer, self.ema)
            state = torch.load(resume_from / ckpt.TRAINER, map_location="cpu", weights_only=False)
            self.scheduler.load_state_dict(state["scheduler"])
            self.step = int(state["step"])
            data_state = ckpt.load_data_state(resume_from, self.env.rank)
            if data_state is not None and int(data_state.get("world_size", -1)) != self.env.world_size:
                self.log(f"[resume] world size changed ({data_state.get('world_size')} → {self.env.world_size}); fresh data stream")
                data_state = None
            self.log(f"[resume] {resume_from} at step {self.step}")

        self.loader, self.sampler = self.task.train_loader(self.env.rank, self.env.world_size, data_state)
        self.data_state = data_state
        self.val_loader = None
        self._val_forwards = 0
        self._last_inputs = None

    # ------------------------------------------------------------ setup helpers

    def log(self, message: str) -> None:
        if self.env.is_main:
            print(message, flush=True)

    def _record(self, record: dict) -> None:
        if self._log_file is not None:
            self._log_file.write(json.dumps(record) + "\n")
            self._log_file.flush()
        if self._tb is not None:
            self._tb.log_record(record)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
            self._tb = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _log_model(self, seconds: float) -> None:
        total = sum(_numel(p) for p in self.model.parameters())
        trainable = sum(_numel(p) for p in self.params)
        parts = [f"total {total / 1e6:.2f}M", f"trainable {trainable / 1e6:.2f}M"]
        detail = ""
        from chrome.dit.model import ChromeDiT, parameter_report

        if isinstance(self.model, ChromeDiT):
            report = parameter_report(self.model)
            parts += [f"adaln {report['adaln'] / 1e6:.2f}M", f"backbone {report['backbone'] / 1e6:.2f}M"]
            detail = " | " + json.dumps(self.model.config.to_dict())
        self.log(f"[model] {', '.join(parts)} | {self.parallel.describe()}, world {self.env.world_size}, "
                 f"built in {seconds:.0f}s{detail}")

    def _resume_path(self) -> Path | None:
        resume = self.t.get("resume")
        if resume in (None, "", "none", False):
            return None
        if str(resume) == "auto":
            return ckpt.resolve_checkpoint(self.run_dir) if ckpt.has_checkpoint(self.run_dir) else None
        return ckpt.resolve_checkpoint(self.run_dir, resume)

    def _init_from(self, source: str) -> None:
        """Weights only (no optimizer / data state): a run dir (latest), a step dir, or a .safetensors file."""
        path = Path(source)
        weights = str(self.t.get("init_weights") or "ema")
        if path.is_file():
            path, weights = path.parent, path.stem
        elif ckpt.has_checkpoint(path):
            path = ckpt.resolve_checkpoint(path)
        used = ckpt.load_weights(self.model, path, weights)
        self.log(f"[init] {used} weights from {path}")

    def _optimizer(self) -> torch.optim.Optimizer:
        t = self.t
        decay = [p for p in self.params if p.ndim >= 2]
        no_decay = [p for p in self.params if p.ndim < 2]
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": float(t["weight_decay"])}, {"params": no_decay, "weight_decay": 0.0}],
            lr=float(t["lr"]),
            betas=tuple(float(b) for b in t["betas"]),
            eps=float(t["eps"]),
            fused=self.device.type == "cuda",
        )

    def _autocast(self):
        precision = str(self.t["precision"])
        if precision == "fp32" or self.device.type != "cuda":
            return nullcontext()
        return torch.autocast("cuda", dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[precision])

    def _generators(self, *key: int) -> Generators:
        """σ (CPU) and ε (device) generators seeded from the step, so resumes replay them."""
        seed = hash((self.seed, *key)) & 0x7FFF_FFFF_FFFF_FFFF
        cpu = torch.Generator().manual_seed(seed)
        dev = torch.Generator(device=self.device).manual_seed(seed ^ 0x5DEECE66D)
        return Generators(cpu, dev)

    # ------------------------------------------------------------ train

    def train(self) -> None:
        t = self.t
        max_steps, accum = int(t["max_steps"]), int(t["grad_accum"])
        log_every = int(t["log_every"])
        meter = Meter()
        data_iter = iter(self.loader)
        self.module.train()
        window_start = time.perf_counter()
        self.log(f"[train] {self.run_dir} from step {self.step} to {max_steps}, world {self.env.world_size}")
        while self.step < max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            for micro in range(accum):
                t0 = time.perf_counter()
                batch = next(data_iter)
                t1 = time.perf_counter()
                prepared = self.task.prepare(batch)
                t2 = time.perf_counter()
                with self.parallel.accumulate(self.module, sync=micro == accum - 1):
                    with self._autocast():
                        out = self.task.loss(self.module, prepared.inputs, self._generators(self.step, micro, self.env.rank))
                    (out.loss / accum).backward()
                meter.add("loss", out.loss.item())
                meter.add("data_s", t1 - t0)
                meter.add("cond_s", t2 - t1)
                meter.add("model_s", time.perf_counter() - t2)
                for name, seconds in prepared.timing.items():
                    meter.add(name, seconds)
                meter.add("samples", prepared.num_samples)
                meter.add("fill", prepared.fill)
                for value, group in zip(out.per_sample.tolist(), out.groups.tolist()):
                    meter.add(f"loss/{self.task.group_names[group]}", value)
                last_batch = batch["batch_id"]
                self._last_inputs = prepared.inputs

            if (self.step + 1) % log_every == 0:
                for name, params in self.task.grad_groups(self.model).items():
                    meter.add(f"gn/{name}", self.parallel.grad_sq_norm(params).sqrt().item())
            grad_norm = self.parallel.clip_grad_norm(self.params, float(t["grad_clip"]))
            if torch.isfinite(grad_norm):
                self.optimizer.step()
            else:
                self.log(f"[train] step {self.step}: non-finite grad norm, update skipped")
            self.scheduler.step()
            self.step += 1
            if self.ema is not None:
                self.ema.update(self.step)
            self.data_state = self.sampler.state_dict(last_batch)
            meter.add("grad_norm", grad_norm.item())

            if self.step % log_every == 0 or self.step == max_steps:
                self._log_train(meter, time.perf_counter() - window_start, last_batch)
                meter, window_start = Meter(), time.perf_counter()
            if int(t["val_every"]) and self.step % int(t["val_every"]) == 0:
                self.validate()
            if int(t["preview_every"]) and self.step % int(t["preview_every"]) == 0:
                self.preview()
            if self.step % int(t["ckpt_every"]) == 0 or self.step == max_steps:
                self.save()
        if int(t["preview_every"]) and self.step % int(t["preview_every"]) != 0:
            self.preview()
        self.close()

    def _log_train(self, meter: Meter, elapsed: float, last_batch: int) -> None:
        steps = max(1, int(meter.counts["grad_norm"]))
        stats = meter.means(self.env)
        record = {
            "event": "train",
            "step": self.step,
            "batch_id": int(last_batch),
            "lr": self.scheduler.get_last_lr()[0],
            "sec_per_step": elapsed / steps,
            **{k: round(v, 6) for k, v in stats.items()},
        }
        hits = getattr(self.task, "text_encoder", None)
        if hits is not None and hasattr(hits, "hits"):
            record["text_cache"] = dict(hits.hits)
        if self.device.type == "cuda":
            record["mem_gb"] = round(torch.cuda.max_memory_allocated(self.device) / 2**30, 1)
        self._record(record)
        groups = " ".join(f"{n}={stats[f'loss/{n}']:.3f}" for n in self.task.group_names if f"loss/{n}" in stats)
        stages = " ".join(
            f"{k[:-2]} {v:.2f}" for k, v in stats.items() if k.endswith("_s") and k not in ("data_s", "cond_s", "model_s")
        )
        gn = " ".join(f"{k[3:]} {v:.3g}" for k, v in stats.items() if k.startswith("gn/"))
        self.log(
            f"[train] step {self.step} loss {stats.get('loss', float('nan')):.4f} lr {record['lr']:.2e} "
            f"gn {stats.get('grad_norm', 0):.3g}{f' ({gn})' if gn else ''} | {record['sec_per_step']:.2f}s/step "
            f"(data {stats.get('data_s', 0):.2f} cond {stats.get('cond_s', 0):.2f} [{stages}] "
            f"model {stats.get('model_s', 0):.2f}) "
            f"{stats.get('samples', 0):.1f} clips/rank fill {stats.get('fill', 0):.2f} "
            f"mem {record.get('mem_gb', 0)}G | {groups}"
        )

    # ------------------------------------------------------------ validation / preview / checkpoint

    @torch.no_grad()
    def validate(self) -> dict:
        """Eval forwards go through the unwrapped model (DDP would sync buffers per forward). Under FSDP every
        rank runs the same number of forwards: ranks with fewer holdout batches repeat one, uncounted."""
        if self.val_loader is None:
            self.val_loader = self.task.val_loader(self.env.rank, self.env.world_size)
            self._val_forwards = all_reduce_max(self.env, len(self.val_loader))
        names = self.task.group_names
        self.parallel.reshard(self.model)
        self.model.eval()
        totals = torch.zeros(2 + 2 * len(names), device=self.device, dtype=torch.float64)
        done, last = 0, self._last_inputs
        for batch in self.val_loader:
            prepared = self.task.prepare(batch)
            with self._autocast():
                out = self.task.loss(self.model, prepared.inputs, self._generators(-1, int(batch["batch_id"])))
            for value, group in zip(out.per_sample.tolist(), out.groups.tolist()):
                if math.isfinite(value):
                    totals[0] += value
                    totals[1] += 1
                    totals[2 + group] += value
                    totals[2 + len(names) + group] += 1
            done, last = done + 1, prepared.inputs
        while getattr(self.parallel, "kind", "") == "fsdp" and done < self._val_forwards and last is not None:
            with self._autocast():
                self.task.loss(self.model, last, self._generators(-1, 0))
            done += 1
        all_reduce_sum(self.env, totals)
        self.parallel.reshard(self.model)
        self.model.train()
        result = {"event": "val", "step": self.step, "loss": (totals[0] / totals[1].clamp(min=1)).item()}
        for i, name in enumerate(names):
            if totals[2 + len(names) + i] > 0:
                result[f"loss/{name}"] = (totals[2 + i] / totals[2 + len(names) + i]).item()
        self._record(result)
        self.log(
            f"[val] step {self.step} loss {result['loss']:.4f} ({int(totals[1].item())} clips) | "
            + " ".join(f"{n}={result[f'loss/{n}']:.3f}" for n in names if f"loss/{n}" in result)
        )
        return result

    @torch.no_grad()
    def preview(self) -> None:
        """Every rank renders an equal share (the last item repeats as padding), with EMA weights swapped in."""
        items = self.task.preview_items()
        if not items:
            return
        world, rank = self.env.world_size, self.env.rank
        per = -(-len(items) // world)
        mine = items[rank * per : (rank + 1) * per]
        keep = len(mine)
        mine += [items[-1]] * (per - keep)
        out = self.run_dir / "samples" / f"step_{self.step:07d}"
        started = time.perf_counter()
        use_ema = self.ema is not None and bool(self.cfg["preview"]["use_ema"])
        self.parallel.reshard(self.model)
        self.model.eval()
        with self.ema.swapped() if use_ema else nullcontext():
            self.task.preview(self.model, mine, out, keep)
            self.parallel.reshard(self.model)
        self.model.train()
        barrier(self.env)
        if self._tb is not None:
            self._tb.log_previews(self.step, out)
        self.log(f"[preview] {len(items)} clips in {time.perf_counter() - started:.0f}s → {out}")

    def save(self) -> None:
        path = ckpt.step_dir(self.run_dir, self.step)
        started = time.perf_counter()
        if self.env.is_main:
            path.mkdir(parents=True, exist_ok=True)
        barrier(self.env)
        sharded = bool(self.t.get("save_sharded", True))
        if sharded:
            ckpt.save_state(path, self.model, self.optimizer, self.ema)
        else:
            ckpt.save_full(path, self.model, self.optimizer, self.ema)
        if self.data_state is not None:
            ckpt.save_data_state(path, self.env.rank, self.data_state)
        export = str(self.t.get("export") or "none")
        if sharded and export != "none":
            self._export(path, export)
        barrier(self.env)
        if self.env.is_main:
            torch.save(
                {
                    "scheduler": self.scheduler.state_dict(),
                    "step": self.step,
                    "dit": self.model.config.to_dict() if hasattr(self.model, "config") else None,
                    "parallel": self.parallel.describe(),
                    "world_size": self.env.world_size,
                },
                path / ckpt.TRAINER,
            )
            ckpt.write_latest(self.run_dir, self.step, int(self.t["keep_ckpts"]))
        barrier(self.env)
        self.log(f"[ckpt] step {self.step} → {path} ({time.perf_counter() - started:.0f}s)")

    def _export(self, path: Path, which: str) -> None:
        """Collective: gather full fp32 weights onto rank 0's CPU and write ``<which>.safetensors``."""
        from safetensors.torch import save_file
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        which = which if which == "model" or self.ema is not None else "model"
        with self.ema.swapped() if which == "ema" else nullcontext():
            state = get_model_state_dict(self.model, options=options)
        if self.env.is_main:
            save_file({k: v.float().contiguous() for k, v in state.items()}, str(path / f"{which}.safetensors"))
