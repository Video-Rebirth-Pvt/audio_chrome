"""Self-check of the S2 T2A stack: packing, MM-RoPE, DiT, flow, text cache, pipeline (CPU, under a minute).

Run from anywhere:
    bash scripts/dit/check_t2a_stack.sh
    python tests/test_t2a_stack.py

Checks:
  1. packing: per-sample layout, tags, RoPE positions, cu_seqlens, pad to 64, audio pack / unpack round trip
  2. stereo w: H3 canvas ends for 16:9 / 1:1 / 9:16
  3. parity with DiffSynth MiniMaxH3DiT (random tiny weights, strict state-dict load), one and two samples
  4. varlen isolation: a packed pair gives the same outputs as each sample alone; flash = sdpa on CUDA
  5. parameter counts of the 1b / 10b presets match skills.md; shared AdaLN trains
  6. flow: noise / target identities, Euler is exact on straight paths, σ table, loss reductions and masks;
     sampler and layout registries
  7. text cache: duplicates encoded once, disk hits never build the encoder
  8. meta-device build + init_weights sets every tensor; text_norm=stats; a tiny model fits one fixed batch;
     sharded-format (torch.distributed.checkpoint) round trip of model / EMA / optimizer, EMA swap, pruning
  9. pipeline: shapes, CFG path, seeded determinism (random text encoder, stub VAE)
  10. tensorboard: flatten scalars, import log.jsonl + preview wav, cursor skips duplicates
Multi-GPU behaviour (DDP = FSDP, resume, previews) is covered by scripts/train/check_t2a_smoke.sh.
Exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chrome.config import load_config  # noqa: E402
from chrome.dit.config import DiTConfig  # noqa: E402
from chrome.dit.layout import build_layout  # noqa: E402
from chrome.dit.model import ChromeDiT, parameter_report  # noqa: E402
from chrome.dit.packing import Modality, pack, pack_audio, t2a_segments, unpack_audio  # noqa: E402
from chrome.dit.rope import CanvasStereoW, FixedStereoW  # noqa: E402
from chrome.flow.objective import add_noise, flow_loss, velocity_target  # noqa: E402
from chrome.flow.sampler import EulerSampler, build_sampler  # noqa: E402
from chrome.flow.schedule import ShiftedUniform, inference_sigmas, shift_sigma  # noqa: E402
from chrome.pipeline.t2a import T2APipeline  # noqa: E402
from chrome.spec import AudioSpec  # noqa: E402
from chrome.tasks.t2a import T2AInputs, t2a_flow_loss  # noqa: E402
from chrome.text.cache import CachedTextEncoder  # noqa: E402
from chrome.text.random import RandomTextEncoder  # noqa: E402
from chrome.train import checkpoint as ckpt  # noqa: E402
from chrome.train.parallel import EMA  # noqa: E402

FAILED: list[str] = []
W = CanvasStereoW().values(2)


def check(name: str, passed: bool, **details) -> bool:
    shown = ", ".join(f"{k}={v}" for k, v in details.items())
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  ({shown})" if shown else ""), flush=True)
    if not passed:
        FAILED.append(name)
    return bool(passed)


def tiny_config(**over) -> DiTConfig:
    base = dict(num_layers=2, hidden_size=256, num_heads=2, head_dim=128, ffn_hidden_size=384, time_embed_dim=128,
                text_dim=64, attention="sdpa")
    return DiTConfig(**{**base, **over})


def randomize(model: torch.nn.Module, seed: int = 0, std: float = 0.05) -> torch.nn.Module:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(std * torch.randn(p.shape, generator=g))
    return model


def make_case(text_lens, frames, text_dim=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    seq = pack([t2a_segments(l, f, W) for l, f in zip(text_lens, frames)])
    text = torch.randn(sum(text_lens), text_dim, generator=g)
    audio = torch.randn(sum(2 * f for f in frames), 32, generator=g)
    t = torch.rand(len(frames), generator=g)
    return seq, text, audio, seq.row_timesteps(t)


# ---------------------------------------------------------------- 1, 2


def check_packing() -> None:
    seq = pack([t2a_segments(3, 5, W), t2a_segments(2, 4, W)])
    used = 3 + 10 + 2 + 8
    tags = seq.token_tags.tolist()
    layout = tags[:used] == [1] * 3 + [2] * 10 + [1] * 2 + [2] * 8 and set(tags[used:]) == {-1}
    check("pack layout [text | L | R] per sample, pad tag -1", layout and seq.seq_len == 64 and seq.used == used)
    check("cu_seqlens: one segment per sample + padding", seq.cu_seqlens.tolist() == [0, 13, 23, 64]
          and seq.text_cu_seqlens.tolist() == [0, 3, 5] and seq.max_seqlen == 41)
    p = seq.positions
    audio0 = p[3:13]
    t_ok = audio0[:, 0].tolist() == [3, 4, 5, 6, 7] * 2 and p[13:15, 0].tolist() == [0, 1]
    w_ok = torch.allclose(audio0[:5, 2], torch.tensor(W[0], dtype=torch.float64)) and \
        torch.allclose(audio0[5:, 2], torch.tensor(W[1], dtype=torch.float64))
    check("RoPE positions: audio t = text_len + k shared by ears, w per ear, text (k, 0, 0)",
          t_ok and w_ok and p[:3, 1:].abs().sum() == 0 and p[:, 1].abs().sum() == 0)
    rt = seq.row_timesteps(torch.tensor([0.25, 0.75]))
    check("row timesteps: text follows its sample's audio t, padding 1.0",
          rt[:13].eq(0.25).all() and rt[13:23].eq(0.75).all() and rt[23:].eq(1.0).all())
    lat = torch.randn(2, 32, 7)
    rows = pack_audio(lat)
    check("pack_audio channel-major, unpack round trip",
          torch.equal(rows[:7], lat[0].T) and torch.equal(rows[7:], lat[1].T) and torch.equal(unpack_audio(rows, 2), lat))
    parts = seq.split_audio(torch.arange(18).float()[:, None])
    check("split_audio by sample", [len(x) for x in parts] == [10, 8])


def check_stereo_w() -> None:
    cases = {(48, 84): (-5.17, 36.16), (64, 64): (0.0, 31.0), (84, 48): (3.91, 27.09)}
    got = {k: tuple(round(v, 2) for v in CanvasStereoW(*k).values(2)) for k in cases}
    check("canvas stereo w = H3 values (16:9, 1:1, 9:16)", got == cases, got=got)
    check("fixed stereo w", FixedStereoW([0.0, 8.0]).values(2) == [0.0, 8.0])


# ---------------------------------------------------------------- 3, 4, 5


def load_diffsynth_dit():
    cfg = load_config(ROOT / "configs/_base_/paths.yaml")
    root = Path(cfg["paths"]["diffsynth_root"])
    for name in ("diffsynth", "diffsynth.core", "diffsynth.models"):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules.setdefault(name, package)
    attention = types.ModuleType("diffsynth.core.attention")
    attention.attention_forward = lambda q, k, v, scale=None, **_: F.scaled_dot_product_attention(q, k, v, scale=scale)
    sys.modules["diffsynth.core.attention"] = attention
    gradient = types.ModuleType("diffsynth.core.gradient")
    gradient.gradient_checkpoint_forward = lambda block, use, offload, *a, **kw: block(*a, **kw)
    sys.modules["diffsynth.core.gradient"] = gradient
    name = "diffsynth.models.minimax_h3_dit"
    spec = importlib.util.spec_from_file_location(name, root / "diffsynth/models/minimax_h3_dit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_parity() -> None:
    try:
        h3 = load_diffsynth_dit()
    except Exception as exc:  # pragma: no cover
        check("DiffSynth MiniMaxH3DiT importable", False, error=repr(exc))
        return
    c = tiny_config()
    ours = randomize(ChromeDiT(c)).eval()
    ref = h3.MiniMaxH3DiT(
        num_layers=c.num_layers, token_refiner_num_layers=c.refiner_layers, hidden_size=c.hidden_size,
        num_attention_heads=c.num_heads, attention_head_dim=c.head_dim, ffn_hidden_size=c.ffn_hidden_size,
        text_dim=c.text_dim, time_embed_hidden_size=c.time_hidden, time_embed_dim=c.time_embed_dim,
        adaln_out_features=18 * c.hidden_size, final_adaln_out_features=2 * c.hidden_size,
    ).eval()
    missing, unexpected = ref.load_state_dict(ours.state_dict(), strict=True)
    check("state dict loads into MiniMaxH3DiT with strict=True", not missing and not unexpected)
    for label, lens, frames in (("one sample", [7], [12]), ("two samples", [7, 3], [12, 9])):
        seq, text, audio, t_rows = make_case(lens, frames)
        with torch.no_grad():
            mine, _ = ours(seq, audio, text, t_rows)
            audio_x = torch.zeros(1, seq.seq_len, 32)
            audio_x[0, seq.audio_pos] = audio
            unique_t, inverse = torch.unique(t_rows, sorted=True, return_inverse=True)
            _, theirs = ref(
                x=torch.zeros(1, seq.seq_len, 96), audio_x=audio_x, img_position_ids=seq.positions[None],
                unique_timesteps=unique_t, inverse_indices=inverse, update_mask=None, token_tags=seq.token_tags,
                prompt_embeds=text, img_pos_info={"position_ids": seq.video_pos},
                audio_pos_info={"position_ids": seq.audio_pos}, text_pos_info={"position_ids": seq.text_pos},
                img_pos_for_infer_output_info={"position_ids": seq.video_pos},
                packed_seq_params={"cu_seqlens_q": seq.cu_seqlens, "max_seqlen_q": seq.max_seqlen},
                refiner_packed_seq_params={"cu_seqlens_q": seq.text_cu_seqlens, "max_seqlen_q": seq.text_max_seqlen},
                skip_mask_out_condition=True,
            )
        err = (mine - theirs).abs().max().item()
        check(f"parity vs DiffSynth ({label})", err < 1e-4 * max(1.0, theirs.abs().max().item()),
              max_abs=f"{err:.2e}", scale=f"{theirs.abs().max().item():.2f}")


def check_isolation() -> None:
    model = randomize(ChromeDiT(tiny_config()), seed=1).eval()
    seq, text, audio, t_rows = make_case([5, 4], [10, 6], seed=3)
    t_sample = torch.tensor([t_rows[0].item(), t_rows[5 + 20].item()])
    with torch.no_grad():
        both, _ = model(seq, audio, text, t_rows)
        outs = []
        for i, (l, f) in enumerate(((5, 10), (4, 6))):
            s = pack([t2a_segments(l, f, W)])
            ta = text[:5] if i == 0 else text[5:]
            au = audio[:20] if i == 0 else audio[20:]
            out, _ = model(s, au, ta, s.row_timesteps(t_sample[i : i + 1]))
            outs.append(out)
    err = (both - torch.cat(outs)).abs().max().item()
    check("packed pair = each sample alone (varlen isolation)", err < 1e-5, max_abs=f"{err:.2e}")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        m = model.to(dev)
        s = seq.to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outs = {}
            for backend in ("sdpa", "flash"):
                for mod in m.modules():
                    if hasattr(mod, "backend"):
                        mod.backend = backend
                outs[backend], _ = m(s, audio.to(dev), text.to(dev), t_rows.to(dev))
        err = (outs["sdpa"].float() - outs["flash"].float()).abs().max().item()
        check("flash-attn varlen = per-segment SDPA (bf16, CUDA)", err < 5e-2 * max(1.0, outs["sdpa"].abs().max().item()),
              max_abs=f"{err:.2e}")
        model.cpu()


def check_presets() -> None:
    expected = {"1b": (1.320e9, 0.513e9), "10b": (10.822e9, 4.247e9)}
    for name, (total, adaln) in expected.items():
        cfg = load_config(ROOT / f"configs/dit/{name}.yaml")
        dit_cfg = DiTConfig.from_config({**cfg["dit"], "text_dim": 5120})
        with torch.device("meta"):
            report = parameter_report(ChromeDiT(dit_cfg))
        ok = abs(report["total"] - total) / total < 0.002 and abs(report["adaln"] - adaln) / adaln < 0.002
        check(f"{name} preset parameter count", ok, total=f"{report['total'] / 1e9:.3f}B",
              adaln=f"{report['adaln'] / 1e9:.3f}B", backbone=f"{report['backbone'] / 1e9:.3f}B")
    model = ChromeDiT(tiny_config(adaln="shared"))
    model.init_weights()
    seq, text, audio, t_rows = make_case([4, 6], [8, 5])
    # Zero heads give zero output at init; open the head so gradients reach the shared AdaLN.
    with torch.no_grad():
        model.final_layer.audio_out.weight.normal_(std=0.02)
    out, _ = model(seq, audio, text, t_rows)
    out.pow(2).mean().backward()
    grads = [model.shared_adaln.linear.weight.grad, model.blocks[0].adaln_proj.table.grad]
    check("shared AdaLN: one stack-wide projection + per-block tables receive gradients",
          all(g is not None and g.abs().sum() > 0 for g in grads) and "blocks.0.adaln_proj.linear.weight" not in model.state_dict())


# ---------------------------------------------------------------- 6


def check_flow() -> None:
    g = torch.Generator().manual_seed(0)
    x0, eps = torch.randn(2, 32, 9, generator=g), torch.randn(2, 32, 9, generator=g)
    sigma = 0.3
    xs = add_noise(x0, eps, sigma)
    v = velocity_target(x0, eps)
    check("x_σ − σ·v = x0 and x_σ + (1−σ)·v = ε", torch.allclose(xs - sigma * v, x0, atol=1e-6)
          and torch.allclose(xs + (1 - sigma) * v, eps, atol=1e-6))
    out = EulerSampler(steps=7, shift=3.0).sample(lambda xs_, s: [velocity_target(x0, eps)], [eps])[0]
    check("Euler on the exact straight-path velocity lands on x0", torch.allclose(out, x0, atol=1e-5))
    sig = inference_sigmas(10, 3.0)
    check("σ table: 1 → 0, monotone, shifted towards noise", sig[0] == 1 and sig[-1] == 0 and bool((sig[1:] < sig[:-1]).all())
          and sig[5].item() > 0.5, mid=f"{sig[5].item():.3f}")
    u = torch.linspace(0, 1, 11)
    check("shift 1 is the identity; training sampler stays in [0, 1]", torch.allclose(shift_sigma(u, 1.0), u)
          and bool(((s := ShiftedUniform(3.0).sample(1000, torch.Generator().manual_seed(1))) >= 0).all() and (s <= 1).all()))
    pred = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [9.0, 9.0]])
    target = torch.zeros_like(pred)
    mask = torch.tensor([True, True, True, False])
    rows = torch.tensor([0, 0, 1, 1])
    tok, per = flow_loss(pred, target, mask, rows, 2, "token")
    smp, _ = flow_loss(pred, target, mask, rows, 2, "sample")
    check("loss: token / sample reductions, masked rows ignored",
          abs(tok.item() - (1 + 4 + 9) / 3) < 1e-6 and abs(smp.item() - (2.5 + 9) / 2) < 1e-6
          and torch.allclose(per, torch.tensor([2.5, 9.0])))
    sampler = build_sampler({"kind": "euler", "steps": 7, "shift": 3.0})
    layout = build_layout({"kind": "t2a", "stereo_w": {"kind": "canvas"}})
    seq = layout.pack([3, 2], [5, 4], 2)
    ref = pack([t2a_segments(3, 5, W), t2a_segments(2, 4, W)])
    check("registries: build_sampler(euler), build_layout(t2a) = t2a_segments packing",
          isinstance(sampler, EulerSampler) and torch.equal(seq.positions, ref.positions)
          and torch.equal(seq.token_tags, ref.token_tags))


# ---------------------------------------------------------------- 7


def check_text_cache(tmp: Path) -> None:
    built = []

    def factory():
        built.append(1)
        return RandomTextEncoder(dim=16, seed=5)

    fp, dim = RandomTextEncoder.describe(dim=16, seed=5)
    ids = [torch.tensor([1, 2, 3]), torch.tensor([220]), torch.tensor([1, 2, 3])]
    first = CachedTextEncoder(factory, fp, dim, cache_dir=tmp / "cache", memory_items=0)
    a = first.encode(ids)
    second = CachedTextEncoder(lambda: (_ for _ in ()).throw(RuntimeError("should not load")), fp, dim,
                               cache_dir=tmp / "cache", memory_items=4)
    b = second.encode(ids)
    # The repeated caption is served by the memory LRU on the second pass, the other two from disk.
    check("text cache: duplicates encoded once; disk hits never build the encoder",
          len(built) == 1 and first.hits["encoded"] == 2 and second.hits["disk"] == 2
          and second.hits["memory"] == 1 and not second.loaded and all(torch.equal(x, y) for x, y in zip(a, b)),
          hits=second.hits)


# ---------------------------------------------------------------- 8, 9


def check_meta_init() -> None:
    eager = ChromeDiT(tiny_config(adaln="shared", text_norm="stats"))
    torch.manual_seed(3)
    eager.init_weights()
    with torch.device("meta"):
        lazy = ChromeDiT(tiny_config(adaln="shared", text_norm="stats"))
    lazy.to_empty(device="cpu")
    with torch.no_grad():
        for t in [*lazy.parameters(), *lazy.buffers()]:
            t.fill_(float("nan"))
    torch.manual_seed(3)
    lazy.init_weights()
    tensors = dict(lazy.state_dict())
    same = all(torch.equal(v, eager.state_dict()[k]) for k, v in tensors.items())
    check("meta build + to_empty + init_weights sets every parameter and buffer (= eager init)",
          same and all(torch.isfinite(v).all() for v in tensors.values()))

    model = randomize(ChromeDiT(tiny_config(text_dim=64, text_norm="stats", text_clip=4.0)), seed=4).eval()
    mean, std = torch.randn(64), torch.rand(64) + 0.5
    model.set_text_stats(mean, std)
    seq, text, audio, t_rows = make_case([5], [6], seed=5)
    big = text.clone()
    big[0, 7] = 1e4
    with torch.no_grad():
        out_a, _ = model(seq, audio, big, t_rows)
        clipped = big.clone()
        clipped[0, 7] = mean[7] + 4.0 * std[7]
        out_b, _ = model(seq, audio, clipped, t_rows)
    check("text_norm=stats: (x − mean) / std clamped to ±text_clip; stats are saved buffers",
          torch.allclose(out_a, out_b, atol=1e-5) and "text_mean" in model.state_dict())


def check_fit_and_checkpoint(tmp: Path) -> None:
    torch.manual_seed(0)
    model = ChromeDiT(tiny_config(text_dim=16))
    model.init_weights()
    model.set_video_trainable(False)
    text_enc = RandomTextEncoder(dim=16)
    ids = [torch.tensor([5, 6, 7]), torch.tensor([9, 10])]
    latents = [torch.randn(2, 32, 8), torch.randn(2, 32, 6)]
    texts = text_enc.encode(ids)
    inputs = T2AInputs(pack([t2a_segments(3, 8, W), t2a_segments(2, 6, W)]), latents, torch.cat(texts), torch.tensor([0, 1]))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-3)
    ema = EMA(model, 0.9)
    losses = []

    def step(i: int) -> float:
        g = torch.Generator().manual_seed(i % 4)
        loss, _ = t2a_flow_loss(model, inputs, torch.tensor([0.3, 0.7]), g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(i + 1)
        return loss.item()

    for i in range(60):
        losses.append(step(i))
    check("tiny DiT fits a fixed batch (loss drops)", losses[-1] < 0.5 * losses[0],
          first=f"{losses[0]:.3f}", last=f"{losses[-1]:.3f}")
    frozen = not model.video_patch_proj.weight.requires_grad and model.video_patch_proj.weight.grad is None
    check("video heads frozen in T2A (no unused DDP params)", frozen)

    live = {k: v.clone() for k, v in model.state_dict().items()}
    with ema.swapped():
        swapped = {k: v.clone() for k, v in model.state_dict().items()}
    restored = all(torch.equal(v, live[k]) for k, v in model.state_dict().items())
    ema_ok = all(torch.equal(swapped[n], s) for n, s in zip(ema.names, ema.shadow))
    check("EMA swap puts the shadow in and restores the live weights", restored and ema_ok
          and not torch.equal(swapped["condition_proj.weight"], live["condition_proj.weight"]))

    run = tmp / "run"
    for s in (10, 20, 30):
        path = ckpt.step_dir(run, s)
        path.mkdir(parents=True)
        ckpt.save_state(path, model, opt, ema)
        ckpt.save_data_state(path, 0, {"batch_id": s})
        torch.save({"step": s}, path / ckpt.TRAINER)
        ckpt.write_latest(run, s, keep=2)
    latest = ckpt.resolve_checkpoint(run)
    kept = sorted(p.name for p in (run / "checkpoints").glob("step_*"))
    saved_ema = {n: s.clone() for n, s in zip(ema.names, ema.shadow)}
    check("latest pointer and pruning", latest.name == "step_0000030" and kept == ["step_0000020", "step_0000030"]
          and ckpt.load_data_state(latest, 0) == {"batch_id": 30} and ckpt.stored_keys(latest) == {"model", "ema", "optimizer"})

    with torch.device("meta"):
        clone = ChromeDiT(tiny_config(text_dim=16))
    clone.to_empty(device="cpu")
    clone.set_video_trainable(False)
    used = ckpt.load_weights(clone, latest, "ema")
    ema_same = all(torch.equal(clone.state_dict()[n], s) for n, s in zip(ema.names, ema.shadow))
    used_model = ckpt.load_weights(clone, latest, "model")
    model_same = all(torch.equal(v, live[k]) for k, v in clone.state_dict().items())
    check("load_weights from sharded state/: ema and model", used == "ema" and ema_same and used_model == "model" and model_same)

    clone_opt = torch.optim.AdamW([p for p in clone.parameters() if p.requires_grad], lr=2e-3)
    clone_ema = EMA(clone, 0.9)
    with torch.no_grad():
        for p in clone.parameters():
            p.zero_()
    ckpt.load_state(latest, clone, clone_opt, clone_ema)
    expected = step(60)
    model, opt, ema = clone, clone_opt, clone_ema
    resumed = step(60)
    check("resume: model + optimizer + EMA restored, next step identical", expected == resumed,
          expected=f"{expected:.6f}", resumed=f"{resumed:.6f}")

    family = tmp / "t2a_10b"
    (family / "checkpoints").mkdir(parents=True)
    (family / "checkpoints" / "latest").write_text("step_0002000\n")
    check("resolve_run_dir: legacy flat family + auto stays put",
          ckpt.resolve_run_dir(family, "auto", stamp=True, now="20261013_180000") == family)
    fresh = tmp / "t2a_10b_family"
    stamped = ckpt.resolve_run_dir(fresh, None, stamp=True, now="20261013_194512")
    check("resolve_run_dir: resume=null opens <family>/<stamp>",
          stamped == fresh / "20261013_194512")
    (stamped / "config.yaml").write_text("train: {}\n")
    (stamped / "checkpoints").mkdir(parents=True)
    (stamped / "checkpoints" / "latest").write_text("step_0002000\n")
    later = fresh / "20261014_090000"
    later.mkdir()
    (later / "config.yaml").write_text("train: {}\n")
    check("find_run_dir: family → newest stamp", ckpt.find_run_dir(fresh) == later)
    check("resolve_run_dir: auto continues newest stamp",
          ckpt.resolve_run_dir(fresh, "auto", stamp=True, now="20990101_000000") == later)
    check("stamp_run off leaves output_dir unchanged",
          ckpt.resolve_run_dir(fresh, None, stamp=False, now="20261013_194512") == fresh)

    full_dir = tmp / "full"
    full_dir.mkdir()
    ckpt.save_full(full_dir, model, opt, ema)
    check("save_full writes unsharded model + ema + optimizer, no state/",
          (full_dir / ckpt.MODEL_FILE).is_file() and (full_dir / ckpt.EMA_FILE).is_file()
          and (full_dir / ckpt.OPTIMIZER).is_file() and not (full_dir / ckpt.STATE).exists())
    with torch.device("meta"):
        full_model = ChromeDiT(tiny_config(text_dim=16))
    full_model.to_empty(device="cpu")
    full_model.set_video_trainable(False)
    full_opt = torch.optim.AdamW([p for p in full_model.parameters() if p.requires_grad], lr=2e-3)
    full_ema = EMA(full_model, 0.9)
    with torch.no_grad():
        for p in full_model.parameters():
            p.zero_()
    ckpt.load_full(full_dir, full_model, full_opt, full_ema)
    check("load_full restores live weights", all(torch.equal(v, live[k]) for k, v in full_model.state_dict().items()))

    out = ckpt.export_weights(latest, clone, "ema", tmp / "ema.safetensors")
    from safetensors.torch import load_file

    exported = load_file(str(out))
    check("export: sharded state → single safetensors with buffers", set(exported) == set(clone.state_dict())
          and all(torch.equal(exported[n], s) for n, s in saved_ema.items()))


class StubVAE:
    def decode_audio(self, latent):
        return latent.mean(dim=-2).repeat_interleave(800, dim=-1).clamp(-1, 1)


def check_pipeline() -> None:
    torch.manual_seed(0)
    model = randomize(ChromeDiT(tiny_config(text_dim=16)), seed=2).eval()

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return types.SimpleNamespace(ids=[ord(c) % 1000 for c in text])

    spec = AudioSpec()
    layout = build_layout({"kind": "t2a", "stereo_w": {"kind": "canvas"}})
    pipe = T2APipeline(model, RandomTextEncoder(dim=16), StubVAE(), Tok(), spec, layout, 3.0, device="cpu")
    waves = pipe.generate(["rain", "a dog barks twice"], [4, 5], steps=3, cfg_scale=3.0, seed=7)
    again = pipe.generate(["rain", "a dog barks twice"], [4, 5], steps=3, cfg_scale=3.0, seed=7)
    split = pipe.generate(["rain", "a dog barks twice"], [4, 5], steps=3, cfg_scale=3.0, seed=7, batch=1)
    second = pipe.generate(["a dog barks twice"], [5], steps=3, cfg_scale=3.0, seeds=[8])
    nocfg = pipe.generate(["rain"], [4], steps=3, cfg_scale=1.0, seed=7)
    check("pipeline: [2, 32000·s] per caption, seeded per item (batching / sharding independent), CFG changes it",
          [tuple(w.shape) for w in waves] == [(2, 128000), (2, 160000)]
          and all(torch.equal(a, b) for a, b in zip(waves, again))
          and all(torch.allclose(a, b, atol=1e-5) for a, b in zip(waves, split))
          and torch.allclose(second[0], waves[1], atol=1e-5) and not torch.equal(nocfg[0], waves[0]))


def check_tensorboard(tmp: Path) -> None:
    import io
    import wave

    from chrome.train.tensorboard import TensorBoardLogger, encode_wav, flatten_scalars

    tags = flatten_scalars({
        "event": "train", "step": 20, "loss": 1.25, "loss/music": 1.2,
        "text_cache": {"memory": 3}, "lr": 1e-4, "batch_id": 20,
    })
    check("tb flatten: event prefix + nested dicts, drop step",
          tags.get("train/loss") == 1.25 and tags.get("train/loss/music") == 1.2
          and tags.get("train/text_cache/memory") == 3.0 and tags.get("train/lr") == 1e-4
          and "step" not in tags and "train/step" not in tags)

    stereo = torch.stack([torch.full((1600,), 0.5), torch.full((1600,), -0.25)])
    with wave.open(io.BytesIO(encode_wav(stereo, 16000))) as handle:
        check("tb encode_wav keeps 2 channels and L/R",
              handle.getnchannels() == 2 and handle.getnframes() == 1600
              and handle.getframerate() == 16000)

    run = tmp / "tb_run"
    run.mkdir()
    rows = [
        {"event": "train", "step": 20, "loss": 1.2, "lr": 1e-4},
        {"event": "train", "step": 40, "loss": 1.1, "lr": 2e-4},
        {"event": "val", "step": 40, "loss": 1.15, "loss/music": 1.0},
    ]
    (run / "log.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    wav_dir = run / "samples" / "step_0000040"
    wav_dir.mkdir(parents=True)
    torchaudio.save(str(wav_dir / "preview_rain.wav"), stereo, 16000)

    first = TensorBoardLogger(run, enabled=True, log=lambda _: None)
    first.close()
    cursor = json.loads((run / "tb" / "cursor.json").read_text())
    events = list((run / "tb").glob("events.out.tfevents*"))
    check("tb sync: tfevents + cursor from jsonl/wav",
          bool(events) and cursor.get("scalars", {}).get("train") == 40
          and cursor.get("scalars", {}).get("val") == 40
          and 40 in {int(s) for s in cursor.get("audio_steps", [])})

    again = TensorBoardLogger(run, enabled=True, log=lambda _: None)
    n_train, n_val, n_wav = again.sync()
    again.log_record({"event": "train", "step": 60, "loss": 1.05})
    again.close()
    cursor = json.loads((run / "tb" / "cursor.json").read_text())
    check("tb sync: cursor skips duplicates, live step advances",
          (n_train, n_val, n_wav) == (0, 0, 0) and cursor["scalars"]["train"] == 60)

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        check("tb event_accumulator reads train/val/audio tags", True, skipped=True)
        return
    acc = EventAccumulator(str(run / "tb"))
    acc.Reload()
    scalars = acc.Tags().get("scalars", [])
    audio = acc.Tags().get("audio", [])
    losses = [event.value for event in acc.Scalars("train/loss")]
    clips = acc.Audio("preview/preview_rain")
    wav_channels = 0
    if clips:
        with wave.open(io.BytesIO(clips[-1].encoded_audio_string)) as handle:
            wav_channels = handle.getnchannels()
    check("tb events: train/val tags and stereo preview wav",
          "train/loss" in scalars and "val/loss" in scalars and "val/loss/music" in scalars
          and torch.allclose(torch.tensor(losses), torch.tensor([1.2, 1.1, 1.05]), atol=1e-5)
          and "preview/preview_rain" in audio and wav_channels == 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tmp = Path(tempfile.mkdtemp(prefix="chrome_t2a_check_"))
    try:
        check_packing()
        check_stereo_w()
        check_parity()
        check_isolation()
        check_presets()
        check_flow()
        check_text_cache(tmp)
        check_meta_init()
        check_fit_and_checkpoint(tmp)
        check_pipeline()
        check_tensorboard(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL PASSED' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
