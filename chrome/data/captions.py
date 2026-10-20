"""Caption tokenization with the H3 tokenizer, exactly as the H3 T2VA presentation builds its ids."""

from __future__ import annotations

from pathlib import Path

# Unconditional caption for CFG. H3's default negative_prompt; "" would encode to zero tokens.
NULL_CAPTION = " "


def load_caption_tokenizer(path: str | Path):
    """Fast tokenizer with H3's added tokens (151669–151675: <d>, </d>, lyrics/caption markers).

    The added tokens live in tokenizer_config.json, so tokenizer.json on its own splits ``<d>`` into
    characters. Load through transformers once and keep its backend for batch encoding.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path)).backend_tokenizer


def caption_ids(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def caption_lengths(tokenizer, texts: list[str], batch: int = 8192) -> list[int]:
    # Same ids as encode_batch without the offset bookkeeping, about 2.7x faster.
    encode = getattr(tokenizer, "encode_batch_fast", tokenizer.encode_batch)
    lengths: list[int] = []
    for start in range(0, len(texts), batch):
        lengths += [len(e.ids) for e in encode(texts[start : start + batch], add_special_tokens=False)]
    return lengths
