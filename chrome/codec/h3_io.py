"""Read and write audio VAE checkpoints in the two H3 layouts.

* Bundle (``FL2VA/audio_vae``, ``Ref2VA/audio_vae``): config.json points at config.yaml,
  metadata.json and model.safetensors.
* Diffusers (``audio_vae``): one config.json with every hyperparameter plus
  diffusion_pytorch_model.safetensors. Exports use this layout.

Both store latents_mean / latents_std in config.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

from chrome.codec.audio_vae import AudioVAEConfig, ChromeAudioVAE
from chrome.codec.whitening import LatentStats

DIFFUSERS_CLASS = "AutoencoderKLMiniMaxH3Audio"
DIFFUSERS_WEIGHTS = "diffusion_pytorch_model.safetensors"

# The bundle picks its BigVGAN table from the sample rate; metadata.json carries no kernel sizes.
_BUNDLE_DECODER_KERNELS = {16000: (9, 9, 4, 4, 4, 4), 32000: (9, 9, 4, 4, 4, 4, 4)}

_CHROME_KEYS = ("attn_proj", "snake_logscale", "use_tanh_at_final", "use_bias_at_final")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _bundle_config(directory: Path, config: dict) -> tuple[AudioVAEConfig, Path]:
    meta = _read_json(directory / config["source_metadata_path"])["metadata"]["kwargs"]
    with (directory / config["source_config_path"]).open("r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)["model_config"]
    if meta.get("decoder_type", "bigvgan") != "bigvgan":
        raise ValueError(f"unsupported decoder_type {meta['decoder_type']!r}")
    sample_rate = int(meta["sample_rate"])
    vae_config = AudioVAEConfig(
        sample_rate=sample_rate,
        encoder_dim=meta.get("encoder_dim", 64),
        encoder_rates=meta["encoder_rates"],
        latent_dim=meta.get("latent_dim"),
        latent_channels=model_config["vae_latent_channels"],
        attn_proj=meta["attn_proj"],
        decoder_dim=model_config["decoder_dim"],
        decoder_rates=meta["decoder_rates"],
        decoder_kernel_sizes=_BUNDLE_DECODER_KERNELS[sample_rate],
    )
    return vae_config, directory / config["source_safetensors_path"]


def _diffusers_config(directory: Path, config: dict) -> tuple[AudioVAEConfig, Path]:
    vae_config = AudioVAEConfig(
        sample_rate=config["sampling_rate"],
        encoder_dim=config["encoder_dim"],
        encoder_rates=config["encoder_rates"],
        latent_dim=config["latent_dim"],
        latent_channels=config["latent_channels"],
        attn_heads=config.get("num_attention_heads", 8),
        decoder_dim=config["decoder_dim"],
        decoder_rates=config["decoder_rates"],
        decoder_kernel_sizes=config["decoder_kernel_sizes"],
        resblock_kernel_sizes=config["resblock_kernel_sizes"],
        resblock_dilation_sizes=config["resblock_dilation_sizes"],
        **{key: config[key] for key in _CHROME_KEYS if key in config},
    )
    return vae_config, directory / DIFFUSERS_WEIGHTS


def read_h3_component(directory: str | Path) -> tuple[AudioVAEConfig, LatentStats, Path]:
    """Architecture, latent statistics and weight file of one audio_vae directory."""
    directory = Path(directory)
    config = _read_json(directory / "config.json")
    if "source_metadata_path" in config:
        vae_config, weights = _bundle_config(directory, config)
    else:
        vae_config, weights = _diffusers_config(directory, config)
    if "latents_mean" in config:
        stats = LatentStats(config["latents_mean"], config["latents_std"])
    else:
        stats = LatentStats.identity(vae_config.latent_channels)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    return vae_config, stats, weights


def load_h3_audio_vae(
    directory: str | Path, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
) -> ChromeAudioVAE:
    vae_config, stats, weights = read_h3_component(directory)
    model = ChromeAudioVAE(vae_config, stats)
    model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    return model.to(device=device, dtype=dtype).eval()


def save_h3_audio_vae(model: ChromeAudioVAE, directory: str | Path) -> Path:
    """Write the diffusers layout: config.json + diffusion_pytorch_model.safetensors (float32)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    c = model.config
    config = {
        "_class_name": DIFFUSERS_CLASS,
        "encoder_dim": c.encoder_dim,
        "encoder_rates": list(c.encoder_rates),
        "latent_dim": c.latent_dim,
        "latent_channels": c.latent_channels,
        "decoder_dim": c.decoder_dim,
        "decoder_rates": list(c.decoder_rates),
        "decoder_kernel_sizes": list(c.decoder_kernel_sizes),
        "num_attention_heads": c.attn_heads,
        "resblock_kernel_sizes": list(c.resblock_kernel_sizes),
        "resblock_dilation_sizes": [list(group) for group in c.resblock_dilation_sizes],
        "sampling_rate": c.sample_rate,
        **{key: getattr(c, key) for key in _CHROME_KEYS},
        "latents_mean": list(model.latent_stats.mean),
        "latents_std": list(model.latent_stats.std),
    }
    state = {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(directory / DIFFUSERS_WEIGHTS))
    with (directory / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    return directory
