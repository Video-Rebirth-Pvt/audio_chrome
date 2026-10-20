"""Text → 32 kHz stereo audio: caption ids → text encoder → sampler over the DiT (with CFG) → VAE decode.

    pipe = T2APipeline.from_run("outputs/train/t2a_smoke", device="cuda")
    waves = pipe.generate(["Rain on a tin roof."], seconds=[8], steps=50, cfg_scale=4.0, seed=0)  # [2, N]

CFG puts the conditional and unconditional copy of every sample into the same packed sequence, so one
forward gives both velocities. The unconditional caption is a single space (H3's negative prompt).
Item i is seeded ``seed + i`` (or ``seeds[i]``), independent of how items are batched.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from chrome.data.captions import NULL_CAPTION, caption_ids, load_caption_tokenizer
from chrome.dit.layout import Layout
from chrome.dit.model import ChromeDiT
from chrome.dit.packing import pack_audio, unpack_audio
from chrome.flow.sampler import build_sampler
from chrome.spec import AudioSpec
from chrome.text.base import TextEncoder


class T2APipeline:
    def __init__(
        self,
        dit: ChromeDiT,
        text_encoder: TextEncoder,
        vae,
        tokenizer,
        spec: AudioSpec,
        layout: Layout,
        shift: float,
        device: str | torch.device = "cuda",
        autocast_dtype: torch.dtype | None = torch.bfloat16,
    ):
        self.dit, self.text_encoder, self.vae, self.tokenizer = dit, text_encoder, vae, tokenizer
        self.spec, self.layout, self.shift = spec, layout, float(shift)
        self.device = torch.device(device)
        self.autocast_dtype = autocast_dtype

    @classmethod
    def from_run(
        cls, run_dir: str | Path, checkpoint: str = "latest", weights: str = "ema", device="cuda", overrides=(), log=print
    ) -> "T2APipeline":
        from chrome.codec import load_h3_audio_vae
        from chrome.config import load_config
        from chrome.dit.layout import build_layout
        from chrome.text.base import build_text_encoder
        from chrome.train.checkpoint import build_dit, load_weights, resolve_checkpoint

        run_dir = Path(run_dir)
        cfg = load_config(run_dir / "config.yaml", overrides)
        with torch.device("meta"):
            dit = build_dit(cfg)
        dit.to_empty(device=device)
        ckpt = resolve_checkpoint(run_dir, checkpoint)
        used = load_weights(dit, ckpt, weights)
        log(f"[infer] {used} weights from {ckpt}")
        return cls(
            dit.eval(),
            build_text_encoder(cfg["text"], device=device, log=log),
            load_h3_audio_vae(cfg["vae"]["checkpoint"], device=device).eval(),
            load_caption_tokenizer(cfg["data"]["tokenizer"]),
            AudioSpec.from_config(cfg),
            build_layout(cfg["layout"]),
            float(cfg["flow"]["shift"]),
            device=device,
        )

    def token_ids(self, captions: Sequence[str]) -> list[torch.Tensor]:
        return [torch.tensor(caption_ids(self.tokenizer, c), dtype=torch.long) for c in captions]

    def _autocast(self):
        enabled = self.autocast_dtype is not None and self.device.type == "cuda"
        return torch.autocast(self.device.type, dtype=self.autocast_dtype or torch.bfloat16, enabled=enabled)

    @torch.no_grad()
    def generate_latents(
        self,
        captions: Sequence[str],
        seconds: Sequence[float],
        steps: int = 50,
        cfg_scale: float = 4.0,
        seeds: Sequence[int] = (),
        negative: str = NULL_CAPTION,
        sampler: str = "euler",
    ) -> list[torch.Tensor]:
        """Whitened latents [C, 32, T] per caption."""
        n = len(captions)
        if len(seconds) != n or len(seeds) != n:
            raise ValueError(f"{n} captions but {len(seconds)} durations and {len(seeds)} seeds")
        channels = self.spec.channels
        frames = [self.spec.frames_for_seconds(s) for s in seconds]
        ids = self.token_ids(captions)
        texts = self.text_encoder.encode(ids)
        text_lens = [len(i) for i in ids]
        use_cfg = cfg_scale != 1.0
        all_frames = list(frames)
        if use_cfg:
            neg_ids = self.token_ids([negative])[0]
            neg = self.text_encoder.encode([neg_ids])[0]
            text_lens += [len(neg_ids)] * n
            all_frames += frames
            texts = [*texts, *([neg] * n)]
        seq = self.layout.pack(text_lens, all_frames, channels).to(self.device)
        text_rows = torch.cat(texts).to(self.device)

        noise = []
        for seed, f in zip(seeds, frames):
            generator = torch.Generator().manual_seed(int(seed))
            noise.append(torch.randn(channels, self.spec.latent_channels, f, generator=generator).to(self.device))

        def velocity(xs: list[torch.Tensor], sigma: float) -> list[torch.Tensor]:
            rows = torch.cat([pack_audio(x) for x in xs] * (2 if use_cfg else 1))
            t_rows = seq.row_timesteps(torch.full((seq.num_samples,), 1.0 - sigma, device=self.device))
            with self._autocast():
                audio, _ = self.dit(seq, rows, text_rows, t_rows)
            v = [unpack_audio(part, channels) for part in seq.split_audio(audio.float())]
            if use_cfg:
                v = [vu + cfg_scale * (vc - vu) for vc, vu in zip(v[:n], v[n:])]
            return v

        return build_sampler({"kind": sampler, "steps": steps, "shift": self.shift}).sample(velocity, noise)

    @torch.no_grad()
    def decode(self, latents: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self.vae.decode_audio(z[None].float())[0].cpu() for z in latents]

    def generate(
        self, captions, seconds, steps=50, cfg_scale=4.0, seed=0, negative=NULL_CAPTION, batch: int = 8,
        seeds: Sequence[int] | None = None, sampler: str = "euler",
    ):
        """Waveforms [C, 40·s·800] on CPU, generated ``batch`` captions per packed sequence."""
        seeds = list(seeds) if seeds is not None else [int(seed) + i for i in range(len(captions))]
        waves = []
        for start in range(0, len(captions), batch):
            stop = start + batch
            latents = self.generate_latents(
                captions[start:stop], seconds[start:stop], steps, cfg_scale, seeds[start:stop], negative, sampler
            )
            waves += self.decode(latents)
        return waves
