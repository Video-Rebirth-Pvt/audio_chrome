"""Manifest rows: JSONL (byte-range chunks) or Parquet (row-group chunks), column lookup and keep filters."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import orjson

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


def row_kept(row: Mapping, keep: Mapping) -> bool:
    """Exact match (compared as strings), or ``{min, max, allow_missing}``; missing numbers pass by default."""
    for key, spec in keep.items():
        value = row.get(key)
        if _is_range(spec):
            if value in (None, ""):
                if not spec.get("allow_missing", True):
                    return False
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                return False
            if spec.get("min") is not None and number < float(spec["min"]):
                return False
            if spec.get("max") is not None and number > float(spec["max"]):
                return False
        elif str(value) != str(spec):
            return False
    return True


@dataclass(frozen=True)
class Chunk:
    """JSONL: rows whose first byte lies in [start, stop). Parquet: row groups [start, stop)."""

    path: str
    kind: str
    start: int
    stop: int


def manifest_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_dir():
        files = sorted(p for p in root.iterdir() if p.suffix in MANIFEST_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"no .jsonl or .parquet manifests in {root}")
        return files
    if not root.is_file():
        raise FileNotFoundError(root)
    if root.suffix not in MANIFEST_SUFFIXES:
        raise ValueError(f"unsupported manifest {root}; expected .jsonl or .parquet")
    return [root]


def plan_chunks(paths, chunk_bytes: int) -> list[Chunk]:
    chunks = []
    for path in paths:
        for file in manifest_files(path):
            if file.suffix == ".parquet":
                import pyarrow.parquet as pq

                groups = pq.ParquetFile(file).num_row_groups
                chunks += [Chunk(str(file), "parquet", g, g + 1) for g in range(groups)]
            else:
                size = file.stat().st_size
                chunks += [
                    Chunk(str(file), "jsonl", start, min(start + chunk_bytes, size))
                    for start in range(0, max(size, 1), chunk_bytes)
                ]
    return chunks


def iter_rows(chunk: Chunk) -> Iterator[dict | None]:
    """Rows of one chunk; ``None`` marks a line that is not a JSON object."""
    if chunk.kind == "parquet":
        import pyarrow.parquet as pq

        table = pq.ParquetFile(chunk.path).read_row_groups(list(range(chunk.start, chunk.stop)))
        yield from table.to_pylist()
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


def manifest_stamp(paths) -> list[tuple[str, int, int]]:
    """(file, size, mtime_ns) per manifest file; part of the index fingerprint."""
    stamps = []
    for path in paths:
        for file in manifest_files(path):
            stat = file.stat()
            stamps.append((str(file), stat.st_size, stat.st_mtime_ns))
    return stamps
