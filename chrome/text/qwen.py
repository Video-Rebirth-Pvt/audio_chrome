"""Qwen3-VL language model truncated to its first N layers, as the H3 text encoder.

Same features as DiffSynth's ``MiniMaxH3TextEncoder`` for text-only prompts: the language model alone (no
vision tower, no lm_head), ``norm`` replaced by Identity, hidden states of the last kept layer. Text-only
MRoPE has equal t/h/w positions, so it reduces to 1-D RoPE and the default ``arange`` positions are exact.

Only the weights of kept layers are read: build on the meta device, then assign tensors straight from the
safetensors shards onto the target device.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import torch

from chrome.text.base import TextEncoder, register_text_encoder

_PREFIX = "model.language_model."


def _text_config(path: Path) -> dict:
    config = json.loads((path / "config.json").read_text())
    return dict(config.get("text_config", config))


@register_text_encoder("qwen3_vl")
class Qwen3VLTextEncoder(TextEncoder):
    def __init__(
        self,
        path: str,
        num_layers: int = 50,
        dtype: str = "bfloat16",
        batch_tokens: int = 32768,
        pad_id: int = 151643,
        device: str | torch.device = "cpu",
        log=print,
    ):
        from transformers import Qwen3VLTextConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLTextRotaryEmbedding

        self.path = Path(path)
        self.num_layers = int(num_layers)
        self.dtype = getattr(torch, dtype)
        self.batch_tokens = int(batch_tokens)
        self.pad_id = int(pad_id)
        self.device = torch.device(device)
        full = _text_config(self.path)
        if not 0 < self.num_layers <= int(full["num_hidden_layers"]):
            raise ValueError(f"num_layers={self.num_layers} outside 1..{full['num_hidden_layers']}")
        self.dim = int(full["hidden_size"])

        started = time.monotonic()
        config = Qwen3VLTextConfig(**{**full, "num_hidden_layers": self.num_layers, "dtype": dtype})
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = Qwen3VLTextModel(config)
        model.norm = torch.nn.Identity()
        model.load_state_dict(self._read_weights(), strict=True, assign=True)
        model.rotary_emb = Qwen3VLTextRotaryEmbedding(config, device=self.device)
        leftover = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
        if leftover:
            raise RuntimeError(f"tensors left on meta after loading: {leftover[:5]}")
        self.model = model.eval().requires_grad_(False)
        log(
            f"[text] Qwen3-VL {self.num_layers}/{full['num_hidden_layers']} layers, dim {self.dim}, "
            f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params on {self.device} "
            f"in {time.monotonic() - started:.0f}s"
        )

    @staticmethod
    def describe(path: str, num_layers: int = 50, dtype: str = "bfloat16", **_) -> tuple[str, int]:
        return f"qwen3_vl:{Path(path).resolve()}:L{int(num_layers)}:{dtype}", int(_text_config(Path(path))["hidden_size"])

    @property
    def fingerprint(self) -> str:
        return self.describe(str(self.path), self.num_layers, str(self.dtype).removeprefix("torch."))[0]

    def _wanted(self, key: str) -> str | None:
        if not key.startswith(_PREFIX):
            return None
        name = key[len(_PREFIX) :]
        if name.startswith("layers."):
            return name if int(name.split(".")[1]) < self.num_layers else None
        return name if name.startswith("embed_tokens.") else None

    def _read_weights(self) -> dict[str, torch.Tensor]:
        from safetensors import safe_open

        weight_map = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        by_file: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for key, file in weight_map.items():
            name = self._wanted(key)
            if name is not None:
                by_file[file].append((key, name))
        state = {}
        for file, keys in sorted(by_file.items()):
            with safe_open(str(self.path / file), framework="pt", device=str(self.device)) as handle:
                for key, name in keys:
                    state[name] = handle.get_tensor(key).to(self.dtype)
        return state

    @torch.no_grad()
    def encode(self, ids: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """Right-padded batches of similar length; causal attention keeps real tokens blind to the padding."""
        out: list[torch.Tensor | None] = [None] * len(ids)
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and len(ids[order[stop]]) * (stop - start + 1) <= self.batch_tokens:
                stop += 1
            members = order[start:stop]
            longest = max(1, max(len(ids[i]) for i in members))
            batch = torch.full((len(members), longest), self.pad_id, dtype=torch.long)
            for row, i in enumerate(members):
                batch[row, : len(ids[i])] = ids[i].to(torch.long).cpu()
            hidden = self.model(input_ids=batch.to(self.device), use_cache=False).last_hidden_state
            for row, i in enumerate(members):
                out[i] = hidden[row, : len(ids[i])].clone()
            start = stop
        return out
