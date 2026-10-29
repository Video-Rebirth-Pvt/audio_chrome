"""Manifest rows: JSONL (byte-range chunks) or Parquet (row-group chunks), column lookup and keep filters.

Parquet with the audio embedded as bytes (HF ``datasets``) reads every column except the bytes; the audio
field of each row becomes a ref into the repacked Arrow file (``chrome/data/blobs.py``). A sidecar parquet
with the same row order (``<sidecar>/<parent dir>/<file name>``) adds or overrides columns, e.g. captions
and QA scores produced after the download.

A parquet without audio bytes in a source with ``blobs`` (a sidecar whose shard was migrated) is the
manifest itself: row i gets ``audio = <blobs>/<parent dir>/<stem>.arrow#i``, checked by row count and
``segment_id``. Rows whose blob audio is null get no audio field and are dropped as ``no_audio``.
"""

from __future__ import annotations

import fnmatch
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import orjson

from chrome.data.blobs import (
    KEY_COLUMNS,
    blob_path,
    blob_ref,
    check_blob,
    check_blob_rows,
    embedded_audio_column,
    null_audio_rows,
)

# Field that receives the blob ref when the manifest itself holds no audio.
BLOB_AUDIO_KEY = "audio"
AUDIO_KEYS = ("path", "audio", "audio_filepath", "wav")
CAPTION_KEYS = ("resp", "caption", "normalized_text", "text")
DURATION_KEYS = ("duration", "audio_duration")
SAMPLE_RATE_KEYS = ("sample_rate",)
MANIFEST_SUFFIXES = (".jsonl", ".parquet")


def _ordered(preferred, defaults: tuple[str, ...]) -> tuple[str, ...]:
    keys = [str(preferred)] if preferred else []
    return tuple(dict.fromkeys(keys + list(defaults)))


@dataclass(frozen=True)
class Columns:
    """Candidate keys per field; the configured name comes first, then the VR-Foley fallbacks."""

    audio: tuple[str, ...] = AUDIO_KEYS
    caption: tuple[str, ...] = CAPTION_KEYS
    duration: tuple[str, ...] = DURATION_KEYS
    sample_rate: tuple[str, ...] = SAMPLE_RATE_KEYS

    @classmethod
    def from_config(cls, columns: Mapping | None) -> "Columns":
        columns = columns or {}
        return cls(
            audio=_ordered(columns.get("audio"), AUDIO_KEYS),
            caption=_ordered(columns.get("caption"), CAPTION_KEYS),
            duration=_ordered(columns.get("duration"), DURATION_KEYS),
            sample_rate=_ordered(columns.get("sample_rate"), SAMPLE_RATE_KEYS),
        )


def first_value(row: Mapping, keys: tuple[str, ...]):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _is_range(spec) -> bool:
    return isinstance(spec, Mapping) and any(key in spec for key in ("min", "max", "allow_missing"))


def keep_check(row: Mapping, keep: Mapping) -> tuple[str | None, list[str]]:
    """(first field that fails ``keep``, or None; range fields that passed only because the value is missing).

    Exact match or any of a list (compared as strings), or ``{min, max, allow_missing}``: null, "" and NaN
    are missing and pass a range unless ``allow_missing: false``; a null min / max is no bound.
    """
    missing: list[str] = []
    for key, spec in keep.items():
        value = row.get(key)
        if isinstance(spec, Sequence) and not isinstance(spec, str):
            if str(value) not in {str(s) for s in spec}:
                return key, missing
        elif _is_range(spec):
            try:
                number = None if value in (None, "") else float(value)
            except (TypeError, ValueError):
                return key, missing
            if number is None or math.isnan(number):
                if not spec.get("allow_missing", True):
                    return key, missing
                missing.append(key)
                continue
            if spec.get("min") is not None and number < float(spec["min"]):
                return key, missing
            if spec.get("max") is not None and number > float(spec["max"]):
                return key, missing
        elif str(value) != str(spec):
            return key, missing
    return None, missing


@dataclass(frozen=True)
class Chunk:
    """JSONL: rows whose first byte lies in [start, stop). Parquet: row groups [start, stop).

    ``sidecar``: parquet joined row by row. ``blob``: repacked audio of a parquet with embedded bytes, or the
    audio of a parquet manifest without any.
    """

    path: str
    kind: str
    start: int
    stop: int
    sidecar: str | None = None
    blob: str | None = None


def manifest_files(path: str | Path, include: str | None = None) -> list[Path]:
    """A file, or the manifests in a directory whose names match ``include`` (a glob, e.g. ``train-*``)."""
    root = Path(path)
    if root.is_dir():
        files = sorted(
            p for p in root.iterdir()
            if p.suffix in MANIFEST_SUFFIXES and (include is None or fnmatch.fnmatch(p.name, include))
        )
        if not files:
            raise FileNotFoundError(f"no .jsonl or .parquet manifests in {root}" + (f" matching {include}" if include else ""))
        return files
    if not root.is_file():
        raise FileNotFoundError(root)
    if root.suffix not in MANIFEST_SUFFIXES:
        raise ValueError(f"unsupported manifest {root}; expected .jsonl or .parquet")
    return [root]


def sidecar_path(sidecar: str | Path, file: str | Path) -> Path:
    file = Path(file)
    return Path(sidecar) / file.parent.name / file.name


def plan_chunks(
    paths, chunk_bytes: int, include: str | None = None, sidecar: str | None = None, blobs: str | None = None
) -> list[Chunk]:
    chunks = []
    for path in paths:
        for file in manifest_files(path, include):
            if file.suffix == ".parquet":
                import pyarrow.parquet as pq

                handle = pq.ParquetFile(file)
                side = str(sidecar_path(sidecar, file)) if sidecar else None
                if embedded_audio_column(handle.schema_arrow) is not None and not blobs:
                    raise ValueError(f"{file} embeds audio bytes; set the source's `blobs` dir and repack it")
                blob = str(blob_path(blobs, file)) if blobs else None
                chunks += [
                    Chunk(str(file), "parquet", g, g + 1, side, blob) for g in range(handle.num_row_groups)
                ]
            else:
                if sidecar:
                    raise ValueError(f"sidecar is only supported for parquet manifests, not {file}")
                size = file.stat().st_size
                chunks += [
                    Chunk(str(file), "jsonl", start, min(start + chunk_bytes, size))
                    for start in range(0, max(size, 1), chunk_bytes)
                ]
    return chunks


def _sidecar_rows(chunk: Chunk, first: int, count: int) -> list[dict]:
    import pyarrow.parquet as pq

    path = Path(chunk.sidecar)
    if not path.is_file():
        raise FileNotFoundError(f"no sidecar {path} for {chunk.path}")
    table = pq.read_table(path)
    total = pq.ParquetFile(chunk.path).metadata.num_rows
    if table.num_rows != total:
        raise ValueError(f"sidecar {path} has {table.num_rows} rows, {chunk.path} has {total}")
    return table.slice(first, count).to_pylist()


def _parquet_rows(chunk: Chunk) -> Iterator[dict]:
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(chunk.path)
    meta = handle.metadata
    first = sum(meta.row_group(g).num_rows for g in range(chunk.start))
    audio = embedded_audio_column(handle.schema_arrow)
    columns = None if audio is None else [n for n in handle.schema_arrow.names if n != audio]
    rows = handle.read_row_groups(list(range(chunk.start, chunk.stop)), columns=columns).to_pylist()
    if audio is not None or chunk.blob:
        if audio is not None:
            check_blob(chunk.path, chunk.blob, meta.num_rows)
        else:
            keys = {k: [row.get(k) for row in rows] for k in KEY_COLUMNS if k in handle.schema_arrow.names}
            check_blob_rows(chunk.blob, meta.num_rows, first, keys)
        empty = set(null_audio_rows(chunk.blob, first, len(rows)))
        key = audio if audio is not None else BLOB_AUDIO_KEY
        for i, row in enumerate(rows):
            row[key] = None if i in empty else blob_ref(chunk.blob, first + i)
    if chunk.sidecar:
        for i, (row, extra) in enumerate(zip(rows, _sidecar_rows(chunk, first, len(rows)))):
            index = extra.get("row_index")
            if index is not None and int(index) != first + i:
                raise ValueError(f"sidecar {chunk.sidecar} row {first + i} has row_index {index}")
            for key in ("segment_id", "id"):
                if key in row and key in extra and row[key] != extra[key]:
                    raise ValueError(f"sidecar {chunk.sidecar} row {first + i}: {key} {extra[key]!r} != {row[key]!r}")
            row.update(extra)
    yield from rows


def iter_rows(chunk: Chunk) -> Iterator[dict | None]:
    """Rows of one chunk; ``None`` marks a line that is not a JSON object."""
    if chunk.kind == "parquet":
        yield from _parquet_rows(chunk)
        return
    with open(chunk.path, "rb") as handle:
        position = chunk.start
        if position > 0:
            # Land on the first line that starts at or after ``start``.
            handle.seek(position - 1)
            position += len(handle.readline()) - 1
        while position < chunk.stop:
            line = handle.readline()
            if not line:
                break
            position += len(line)
            line = line.strip()
            if not line:
                continue
            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError:
                yield None
                continue
            yield row if isinstance(row, dict) else None


def manifest_stamp(
    paths, include: str | None = None, sidecar: str | None = None, blobs: str | None = None
) -> list[tuple[str, int, int]]:
    """(file, size, mtime_ns) per manifest file, its sidecar and its blob; part of the index fingerprint.

    A blob not written yet is left out; the build then fails on it, and the stamps change once it exists.
    """
    stamps = []
    for path in paths:
        for file in manifest_files(path, include):
            parts = [file, sidecar_path(sidecar, file) if sidecar else None]
            if blobs and file.suffix == ".parquet":
                blob = blob_path(blobs, file)
                parts.append(blob if blob.is_file() else None)
            for part in parts:
                if part is None:
                    continue
                stat = part.stat()
                stamps.append((str(part), stat.st_size, stat.st_mtime_ns))
    return stamps
