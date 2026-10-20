"""T2A: caption → 32 kHz stereo clip. Waves go through the frozen VAE, caption ids through the (cached) text
encoder, both into the packed ``[text | audio L | audio R]`` layout; the DiT predicts v = ε − x0 per audio row.

A latent cache would replace ``encode_waves`` in ``prepare`` only (online VAE encoding is ~0.9 s per
16k-row batch, almost all of the conditioning time; text from the cache takes ~7 ms).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from chrome.codec import load_h3_audio_vae
from chrome.data.audio_io import save_audio
from chrome.data.batch import encode_waves
from chrome.data.index import source_specs
from chrome.dit.layout import Layout, build_layout
from chrome.dit.model import ChromeDiT
from chrome.dit.packing import PackedSequence, pack_audio
from chrome.flow.objective import add_noise, flow_loss, velocity_target
from chrome.flow.schedule import build_timestep_sampler
from chrome.spec import AudioSpec
from chrome.tasks.base import Generators, LossOutput, Prepared, Task, register_task
from chrome.text.base import TextEncoder, build_text_encoder
from chrome.text.cache import CachedTextEncoder

TEXT_STATS_FILE = "stats.safetensors"


@dataclass
class T2AInputs:
    seq: PackedSequence
    latents: list[torch.Tensor]  # [C, 32, T_i] whitened
    text_rows: torch.Tensor  # [Σ L_i, text_dim]
    sources: torch.Tensor  # [N] source index per sample


@torch.no_grad()
def encode_t2a(batch: dict, vae, text_encoder: TextEncoder, layout: Layout, device, allow_tf32=False):
    """Loader batch → (T2AInputs, seconds per stage)."""
    t0 = time.perf_counter()
    latents = encode_waves(vae, batch["waves"], device, allow_tf32)
    t1 = time.perf_counter()
    texts = text_encoder.encode(batch["input_ids"])
    t2 = time.perf_counter()
    seq = layout.pack([len(i) for i in batch["input_ids"]], [z.shape[-1] for z in latents], [z.shape[0] for z in latents])
    inputs = T2AInputs(seq.to(device), latents, torch.cat(texts).to(device), batch["source"])
    return inputs, {"vae_s": t1 - t0, "text_s": t2 - t1, "pack_s": time.perf_counter() - t2}


def t2a_flow_loss(
    dit, inputs: T2AInputs, sigma: torch.Tensor, generator: torch.Generator | None = None, reduction: str = "token"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Noise every clip at its own σ, predict v = ε − x0 on the audio rows. Returns (loss, per-sample loss)."""
    seq = inputs.seq
    noise = [torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype) for z in inputs.latents]
    noisy = [add_noise(z, e, s) for z, e, s in zip(inputs.latents, noise, sigma.tolist())]
    rows = torch.cat([pack_audio(x) for x in noisy])
    target = torch.cat([pack_audio(velocity_target(z, e)) for z, e in zip(inputs.latents, noise)])
    t_rows = seq.row_timesteps((1.0 - sigma).to(rows.device))
    pred, _ = dit(seq, rows, inputs.text_rows, t_rows)
    row_sample = seq.sample_index.index_select(0, seq.audio_pos)
    return flow_loss(pred, target, seq.audio_target, row_sample, seq.num_samples, reduction)


@register_task("t2a")
class T2ATask(Task):
    def __init__(self, cfg, device, log=print):
        super().__init__(cfg, device, log)
        self.spec = AudioSpec.from_config(cfg)
        self.vae = load_h3_audio_vae(cfg["vae"]["checkpoint"], device=self.device).eval().requires_grad_(False)
        self.spec.check_codec(self.vae.config)
        self.allow_tf32 = bool(cfg["vae"]["allow_tf32"])
        self.text_encoder = build_text_encoder(cfg["text"], device=self.device, log=log)
        self.layout = build_layout(cfg["layout"])
        self.sigma_sampler = build_timestep_sampler(cfg["flow"])
        self.reduction = str(cfg["flow"]["loss_reduction"])
        self.max_tokens = int(cfg["data"]["loader"]["max_tokens"])
        self.group_names = [s.name for s in source_specs(cfg)]
        self._train = None

    # ------------------------------------------------------------ model

    def build_model(self) -> ChromeDiT:
        from chrome.train.checkpoint import build_dit

        model = build_dit(self.cfg)
        model.set_video_trainable(False)
        return model

    def init_model(self, model: ChromeDiT) -> None:
        model.init_weights()
        if model.config.text_norm == "stats":
            mean, std = self.text_stats()
            model.set_text_stats(mean, std)
            self.log(f"[model] text stats: |mean| max {mean.abs().max():.1f}, std range {std.min():.2f}–{std.max():.1f}")

    def text_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-dim mean / std of the encoder output, written by ``scripts/text/cache_text.sh``."""
        from safetensors.torch import load_file

        path = self.cfg["dit"].get("text_stats")
        if not path:
            if not isinstance(self.text_encoder, CachedTextEncoder) or self.text_encoder.dir is None:
                raise ValueError("text_norm=stats needs dit.text_stats or a text cache directory")
            path = self.text_encoder.dir / TEXT_STATS_FILE
        if not Path(path).is_file():
            raise FileNotFoundError(f"{path} missing; run bash scripts/text/cache_text.sh <config> first")
        stats = load_file(str(path))
        return stats["mean"], stats["std"]

    def grad_groups(self, model: ChromeDiT) -> dict[str, list[nn.Parameter]]:
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        return {
            "condition_proj": [p for n, p in named if n.startswith("condition_proj.")],
            "adaln": [p for n, p in named if "adaln" in n],
        }

    # ------------------------------------------------------------ data

    def train_loader(self, rank, world_size, state):
        from chrome.data.loader import build_train_loader

        loader, sampler = build_train_loader(self.cfg, rank, world_size, state=state, log=self.log)
        self._train = (loader, sampler)
        return loader, sampler

    def val_loader(self, rank, world_size):
        from chrome.data.loader import build_val_loader

        return build_val_loader(self.cfg, rank, world_size, log=self.log)

    def prepare(self, batch: dict) -> Prepared:
        inputs, timing = encode_t2a(batch, self.vae, self.text_encoder, self.layout, self.device, self.allow_tf32)
        return Prepared(inputs, inputs.seq.num_samples, inputs.seq.used / self.max_tokens, timing)

    def loss(self, model, inputs: T2AInputs, generators: Generators) -> LossOutput:
        sigma = self.sigma_sampler.sample(inputs.seq.num_samples, generator=generators.cpu)
        loss, per_sample = t2a_flow_loss(model, inputs, sigma, generators.device, self.reduction)
        return LossOutput(loss, per_sample, inputs.sources)

    # ------------------------------------------------------------ preview

    def preview_items(self) -> list[tuple[str, str, int, int]]:
        """(name, caption, seconds, seed): preview-set clips plus the first train rows (fitting check)."""
        p = self.cfg["preview"]
        lo, hi = int(self.spec.min_seconds), int(self.spec.max_seconds)
        items = []
        labels = set(p.get("labels") or [])
        if p.get("set"):
            for entry in json.loads(Path(p["set"]).read_text()):
                if labels and entry["label"] not in labels:
                    continue
                seconds = int(p["seconds"]) if p.get("seconds") else min(hi, max(lo, round(entry["duration"])))
                items.append((f"preview_{entry['label'].replace(' ', '_')}", entry["caption"], seconds))
        if self._train is not None and int(p.get("train_captions") or 0):
            loader, sampler = self._train
            indexes, streams = loader.dataset.indexes, sampler.streams
            for k in range(int(p["train_captions"])):
                stream = streams[k % len(streams)]
                if len(stream.rows) == 0:
                    continue
                row = int(stream.rows[(k // len(streams)) % len(stream.rows)])
                index = indexes[k % len(streams)]
                _, caption, _ = index.row(row)
                items.append((f"train_{stream.name}_{row}", caption, int(index.seconds[row])))
        return [(*item, int(p["seed"]) + i) for i, item in enumerate(items)]

    def pipeline(self, model: ChromeDiT):
        from chrome.data.captions import load_caption_tokenizer
        from chrome.pipeline.t2a import T2APipeline

        return T2APipeline(
            model, self.text_encoder, self.vae, load_caption_tokenizer(self.cfg["data"]["tokenizer"]),
            self.spec, self.layout, float(self.cfg["flow"]["shift"]), device=self.device,
        )

    @torch.no_grad()
    def preview(self, model, items, out_dir: Path, keep: int) -> None:
        if not items:
            return
        p = self.cfg["preview"]
        names, captions, seconds, seeds = zip(*items)
        waves = self.pipeline(model).generate(
            list(captions), list(seconds), int(p["steps"]), float(p["cfg_scale"]), batch=int(p["batch"]),
            seeds=list(seeds), sampler=str(p.get("sampler", "euler")),
        )
        for name, caption, wave in list(zip(names, captions, waves))[:keep]:
            save_audio(out_dir / f"{name}.wav", wave, self.spec.sample_rate)
            (out_dir / f"{name}.txt").write_text(caption + "\n")
