"""Reconstruction and stereo metrics for one [C, N] clip pair."""

from __future__ import annotations

import torch
import torchaudio

MEL_SCALES = ((512, 64), (1024, 128), (2048, 128))


def si_sdr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR in dB, one value per channel."""
    reference = reference - reference.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    dot = (estimate * reference).sum(dim=-1, keepdim=True)
    energy = reference.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)
    projection = dot * reference / energy
    noise = estimate - projection
    return 10.0 * torch.log10(projection.pow(2).sum(-1).clamp_min(eps) / noise.pow(2).sum(-1).clamp_min(eps))


def log_stft_l1(reference: torch.Tensor, estimate: torch.Tensor, n_ffts: tuple[int, ...] = (512, 1024, 2048)) -> float:
    losses = []
    for n_fft in n_ffts:
        window = torch.hann_window(n_fft, device=reference.device)
        ref = torch.stft(reference, n_fft, n_fft // 4, window=window, return_complex=True).abs()
        est = torch.stft(estimate, n_fft, n_fft // 4, window=window, return_complex=True).abs()
        losses.append((ref.clamp_min(1e-5).log() - est.clamp_min(1e-5).log()).abs().mean())
    return float(torch.stack(losses).mean())


def mel_l1(reference: torch.Tensor, estimate: torch.Tensor, sample_rate: int) -> float:
    """Mean |log10 mel| distance over three resolutions."""
    losses = []
    for n_fft, n_mels in MEL_SCALES:
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=n_fft // 4, n_mels=n_mels, power=1.0
        ).to(reference.device)
        ref = mel(reference).clamp_min(1e-5).log10()
        est = mel(estimate).clamp_min(1e-5).log10()
        losses.append((ref - est).abs().mean())
    return float(torch.stack(losses).mean())


def band_energy(waveform: torch.Tensor, sample_rate: int, low_hz: float, high_hz: float, n_fft: int = 2048) -> float:
    window = torch.hann_window(n_fft, device=waveform.device)
    power = torch.stft(waveform, n_fft, n_fft // 4, window=window, return_complex=True).abs().pow(2)
    freqs = torch.fft.rfftfreq(n_fft, 1.0 / sample_rate, device=waveform.device)
    mask = (freqs >= low_hz) & (freqs < high_hz)
    return float(power[:, mask, :].mean()) if bool(mask.any()) else 0.0


def highband_ratio(
    reference: torch.Tensor, estimate: torch.Tensor, sample_rate: int, low_hz: float = 8000.0
) -> float | None:
    """Reconstructed energy above ``low_hz`` over the source's. None when the source has no high band."""
    ref = band_energy(reference, sample_rate, low_hz, sample_rate / 2)
    if ref < 1e-10:
        return None
    return band_energy(estimate, sample_rate, low_hz, sample_rate / 2) / ref


def stereo_report(waveform: torch.Tensor) -> dict:
    """L/R correlation, side-to-mid energy and ILD of a [2, N] clip."""
    left, right = waveform[0], waveform[1]
    corr = float(torch.corrcoef(torch.stack([left, right]))[0, 1].nan_to_num(1.0))
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    side_ratio = float(side.pow(2).mean() / mid.pow(2).mean().clamp_min(1e-8))
    ild_db = float(10.0 * torch.log10(left.pow(2).mean().clamp_min(1e-8) / right.pow(2).mean().clamp_min(1e-8)))
    return {"lr_corr": corr, "side_ratio": side_ratio, "ild_db": ild_db}


def clip_metrics(reference: torch.Tensor, estimate: torch.Tensor, sample_rate: int) -> dict:
    length = min(reference.shape[-1], estimate.shape[-1])
    reference = reference[..., :length].float()
    estimate = estimate[..., :length].float()
    sdr = si_sdr(reference, estimate)
    metrics = {
        "si_sdr_db": float(sdr.mean()),
        "mel_l1": mel_l1(reference, estimate, sample_rate),
        "log_stft_l1": log_stft_l1(reference, estimate),
        "highband_ratio": highband_ratio(reference, estimate, sample_rate),
    }
    if reference.shape[0] == 2:
        src, rec = stereo_report(reference), stereo_report(estimate)
        metrics.update(
            {
                "src_side_ratio": src["side_ratio"],
                "rec_side_ratio": rec["side_ratio"],
                "src_ild_db": src["ild_db"],
                "rec_ild_db": rec["ild_db"],
            }
        )
    return metrics
