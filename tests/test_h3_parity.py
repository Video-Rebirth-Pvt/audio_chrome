"""Check that chrome.codec reproduces the released H3 audio VAE.

Run from the Chrome-10B root:
    CUDA_VISIBLE_DEVICES=0 bash scripts/vae/check_h3_parity.sh
    CUDA_VISIBLE_DEVICES=0 bash scripts/vae/check_h3_parity.sh parity.clip_seconds=4 parity.device=cpu
    CUDA_VISIBLE_DEVICES=0 python tests/test_h3_parity.py --config configs/vae/h3_frozen.yaml

Checks:
  1. both checkpoint layouts load with strict=True and match configs/vae/model_h3.yaml
  2. decode equals the official bundle on random latents
  3. encode equals the official modules and the DiffSynth port (raw and whitened)
  4. stereo folding: identical ears give identical latents, batched equals per-clip
  5. export → reload round trip is bit-exact
  6. real preview clips: mel_l1 gate; si_sdr, posterior σ and whitened latent statistics are reported
Writes ``report.json`` and wav pairs under parity.output_dir; exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import tempfile
import types
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chrome.codec import (  # noqa: E402
    AudioVAEConfig,
    ChromeAudioVAE,
    load_h3_audio_vae,
    read_h3_component,
    save_h3_audio_vae,
)
from chrome.config import load_config  # noqa: E402
from chrome.data.audio_io import load_audio, save_audio  # noqa: E402
from chrome.eval.recon import clip_metrics  # noqa: E402
from chrome.spec import AudioSpec  # noqa: E402


class Report:
    def __init__(self):
        self.checks: list[dict] = []
        self.clips: list[dict] = []

    def check(self, name: str, passed: bool, **details) -> bool:
        self.checks.append({"name": name, "passed": bool(passed), **details})
        shown = ", ".join(f"{k}={_fmt(v)}" for k, v in details.items())
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  ({shown})" if shown else ""), flush=True)
        return bool(passed)

    def close(self, atol: float) -> Callable[[str, torch.Tensor, torch.Tensor], bool]:
        def compare(name: str, ours: torch.Tensor, ref: torch.Tensor) -> bool:
            if ours.shape != ref.shape:
                return self.check(name, False, ours=list(ours.shape), ref=list(ref.shape))
            diff = (ours.float() - ref.float()).abs()
            scale = ref.float().abs().amax().clamp_min(1e-12)
            return self.check(
                name, float(diff.amax()) <= atol, max_abs=float(diff.amax()), rel=float(diff.amax() / scale)
            )

        return compare

    @property
    def passed(self) -> bool:
        return all(c["passed"] for c in self.checks)


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3e}" if value and abs(value) < 1e-2 else f"{value:.4g}"
    return str(value)


def load_official(bundle_dir: Path):
    sys.path.insert(0, str(bundle_dir.parent))
    module = importlib.import_module(f"{bundle_dir.name}.minimax_h3_audio_vae")
    return module.MiniMaxH3AudioVAE.from_pretrained(str(bundle_dir)).model


def load_diffsynth(diffsynth_root: Path, weights: Path):
    """Import only minimax_h3_audio_vae.py; the diffsynth package pulls in modelscope and friends."""
    from einops import rearrange
    from safetensors.torch import load_file

    def attention_forward(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims=None,
                          attn_mask=None, scale=None, is_causal=False, **_):
        dims = dims or {}
        q = rearrange(q, f"{q_pattern} -> b n s d", **dims)
        k = rearrange(k, f"{k_pattern} -> b n s d", **dims)
        v = rearrange(v, f"{v_pattern} -> b n s d", **dims)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale, is_causal=is_causal)
        return rearrange(out, f"b n s d -> {out_pattern}", **dims)

    for name in ("diffsynth", "diffsynth.core", "diffsynth.models"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    attention = types.ModuleType("diffsynth.core.attention")
    attention.attention_forward = attention_forward
    sys.modules["diffsynth.core.attention"] = attention

    name = "diffsynth.models.minimax_h3_audio_vae"
    spec = importlib.util.spec_from_file_location(name, diffsynth_root / "diffsynth/models/minimax_h3_audio_vae.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    renamed = {"pre_block.attn.q_bias", "pre_block.attn.v_bias", "pre_block.attn.zero_k_bias"}
    state = {(k + ".weight" if k in renamed else k): v for k, v in load_file(str(weights), device="cpu").items()}
    model = module.MiniMaxH3AudioVAE()
    model.load_state_dict(state, strict=True)
    return model.eval()


def count_params(model: ChromeAudioVAE) -> dict:
    groups = {
        "encoder": [model.encoder],
        "pre_block": [model.pre_block] if model.pre_block is not None else [],
        "heads": [model.mean_proj, model.logs_proj, model.dec_in_proj],
        "decoder": [model.decoder],
    }
    counts = {k: sum(p.numel() for m in ms for p in m.parameters()) for k, ms in groups.items()}
    counts["total"] = sum(p.numel() for p in model.parameters())
    return {k: round(v / 1e6, 3) for k, v in counts.items()}


def check_loading(cfg, report: Report, device) -> ChromeAudioVAE:
    p = cfg.parity
    expected = AudioVAEConfig.from_dict(OmegaConf.to_container(cfg.vae))
    spec = AudioSpec.from_config(cfg)

    model = load_h3_audio_vae(p.bundle_dir, device=device)
    report.check("bundle strict load + config == model_h3.yaml", model.config == expected, **count_params(model))
    try:
        spec.check_codec(model.config)
        report.check("codec matches audio spec", True, hop=model.hop_length, latent_rate=model.config.latent_rate)
    except ValueError as exc:
        report.check("codec matches audio spec", False, error=str(exc))

    diffusers = load_h3_audio_vae(p.diffusers_dir, device=device)
    same_config = diffusers.config == expected and diffusers.latent_stats == model.latent_stats
    ours_state, ref_state = model.state_dict(), diffusers.state_dict()
    same_weights = all(torch.equal(ours_state[k], ref_state[k]) for k in ours_state)
    report.check("diffusers strict load, same config/stats/weights", same_config and same_weights)
    del diffusers
    return model


@torch.no_grad()
def check_decode(cfg, report: Report, model: ChromeAudioVAE, official, device):
    p = cfg.parity
    compare = report.close(p.atol)
    frames = round(p.probe_seconds * model.config.latent_rate)
    generator = torch.Generator(device="cpu").manual_seed(p.seed)
    z = torch.randn(2, model.config.latent_channels, frames, generator=generator).to(device)
    compare("decode vs official bundle (random z)", model.decode(z), official.decode(z))


@torch.no_grad()
def check_encode(cfg, report: Report, model: ChromeAudioVAE, official, diffsynth, probe: torch.Tensor):
    p = cfg.parity
    compare = report.close(p.atol)
    audio = model.preprocess(probe.unsqueeze(1))

    ours = model.encode_posterior(audio)
    z = official.encoder(audio)
    z = official.pre_block(z.transpose(1, 2)).transpose(1, 2)
    compare("encode mean vs official modules", ours.mean, official.mean_proj(z))
    compare("encode logs vs official modules", ours.logs, official.logs_proj(z))

    compare("encode vs DiffSynth.encode", model.encode(audio), diffsynth.encode(audio))
    whitened = model.encode_audio(probe)
    compare("encode_audio (whitened) vs DiffSynth", whitened, diffsynth.encode_audio(probe))
    compare("decode_audio vs DiffSynth", model.decode_audio(whitened), diffsynth.decode_audio(whitened)[0])


@torch.no_grad()
def check_stereo(cfg, report: Report, model: ChromeAudioVAE, probe: torch.Tensor):
    compare = report.close(cfg.parity.atol)
    twin = probe[:1].repeat(2, 1)
    latents = model.encode_audio(twin)
    compare("identical ears give identical latents", latents[0], latents[1])

    swapped = probe.flip(0)
    batch = torch.stack([probe, swapped])
    batched = model.encode_audio(batch)
    single = torch.stack([model.encode_audio(probe), model.encode_audio(swapped)])
    compare("batched [B, C, N] encode == per-clip encode", batched, single)


@torch.no_grad()
def check_export(cfg, report: Report, model: ChromeAudioVAE, probe: torch.Tensor):
    with tempfile.TemporaryDirectory() as tmp:
        reloaded = load_h3_audio_vae(save_h3_audio_vae(model, tmp), device=probe.device)
    ours, ref = model.state_dict(), reloaded.state_dict()
    same = ours.keys() == ref.keys() and all(torch.equal(ours[k], ref[k]) for k in ours)
    same = same and reloaded.config == model.config and reloaded.latent_stats == model.latent_stats
    same = same and torch.equal(model.encode_audio(probe), reloaded.encode_audio(probe))
    report.check("export → reload is bit-exact", same)


def _quantiles(x: torch.Tensor) -> dict:
    x = x.flatten().float().cpu()
    q = torch.quantile(x[torch.randperm(x.numel())[:1_000_000]], torch.tensor([0.05, 0.5, 0.95]))
    return {"p05": float(q[0]), "p50": float(q[1]), "p95": float(q[2])}


@torch.no_grad()
def check_clips(cfg, report: Report, model: ChromeAudioVAE, clips: list[dict], device):
    p = cfg.parity
    out_dir = ROOT / p.output_dir
    sr = model.sample_rate
    whitened_all = []
    for item in clips:
        waveform = load_audio(item["audio"], sample_rate=sr, channels=2)
        waveform = waveform[:, : round(p.clip_seconds * sr)].to(device)
        posterior = model.encode_posterior(model.preprocess(waveform.unsqueeze(1)))
        recon = model.decode(posterior.mean).squeeze(1)[:, : waveform.shape[-1]]
        whitened = model.latent_stats.whiten(posterior.mean)
        whitened_all.append(whitened.transpose(0, 1).reshape(model.config.latent_channels, -1))

        metrics = clip_metrics(waveform, recon, sr)
        metrics.update(
            label=item["label"],
            domain=item["domain"],
            seconds=round(waveform.shape[-1] / sr, 2),
            sigma=_quantiles(posterior.std),
            kl_per_dim=float(posterior.kl().mean()),
            whitened_mean=float(whitened.mean()),
            whitened_std=float(whitened.std()),
        )
        report.clips.append(metrics)
        stem = item["label"].replace(" ", "_")
        save_audio(out_dir / f"{stem}_src.wav", waveform, sr)
        save_audio(out_dir / f"{stem}_rec.wav", recon, sr)
        hb = metrics["highband_ratio"]
        print(
            f"  {item['label']:<26} si_sdr {metrics['si_sdr_db']:6.2f} dB  mel_l1 {metrics['mel_l1']:.3f}  "
            f"hb {'n/a' if hb is None else f'{hb:.2f}'}  σ_p50 {metrics['sigma']['p50']:.3f}  "
            f"KL/dim {metrics['kl_per_dim']:.3f}  z̃ μ {metrics['whitened_mean']:+.2f} σ {metrics['whitened_std']:.2f}",
            flush=True,
        )

    pooled = torch.cat(whitened_all, dim=-1)
    channel_std = pooled.std(dim=-1)
    report.check(
        "whitened latents of real clips are O(1)",
        bool(channel_std.amax() < 3.0) and bool(pooled.mean(dim=-1).abs().amax() < 1.5),
        frames=pooled.shape[-1],
        channel_std_min=float(channel_std.amin()),
        channel_std_max=float(channel_std.amax()),
        channel_mean_absmax=float(pooled.mean(dim=-1).abs().amax()),
    )
    worst = max(report.clips, key=lambda c: c["mel_l1"])
    report.check(
        f"real clips reconstruct (mel_l1 < {p.max_mel_l1})",
        worst["mel_l1"] < p.max_mel_l1,
        worst=worst["label"],
        worst_mel_l1=worst["mel_l1"],
    )


def select_clips(cfg) -> list[dict]:
    with (ROOT / cfg.parity.preview_set).open("r", encoding="utf-8") as f:
        items = {item["label"]: item for item in json.load(f)}
    missing = [label for label in cfg.parity.clip_labels if label not in items]
    if missing:
        raise KeyError(f"labels not in {cfg.parity.preview_set}: {missing}")
    return [items[label] for label in cfg.parity.clip_labels]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/vae/h3_frozen.yaml")
    parser.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. parity.clip_seconds=4")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config, args.overrides)
    p = cfg.parity
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(p.seed)
    device = torch.device(p.device if p.device != "cuda" or torch.cuda.is_available() else "cpu")
    print(f"device={device}  atol={p.atol}  config={args.config}", flush=True)

    report = Report()
    model = check_loading(cfg, report, device)
    official = load_official(Path(p.bundle_dir)).to(device)
    _, _, weights = read_h3_component(p.bundle_dir)
    diffsynth = load_diffsynth(Path(p.diffsynth_root), weights).to(device)

    clips = select_clips(cfg)
    probe = load_audio(clips[0]["audio"], sample_rate=model.sample_rate, channels=2)
    probe = probe[:, : round(p.probe_seconds * model.sample_rate) + 123].to(device)

    check_decode(cfg, report, model, official, device)
    check_encode(cfg, report, model, official, diffsynth, probe)
    del official, diffsynth
    check_stereo(cfg, report, model, probe)
    check_export(cfg, report, model, probe)
    check_clips(cfg, report, model, clips, device)

    out_dir = ROOT / p.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": report.passed,
        "device": str(device),
        "config": OmegaConf.to_container(cfg),
        "checks": report.checks,
        "clips": report.clips,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    failed = [c["name"] for c in report.checks if not c["passed"]]
    print(f"\n{len(report.checks) - len(failed)}/{len(report.checks)} checks passed → {out_dir / 'report.json'}")
    for name in failed:
        print(f"  FAILED: {name}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
