"""Audio embedded in Parquet (HF ``datasets`` style), repacked once into memory-mappable Arrow files.

HF shards keep every clip of a file in one ZSTD dictionary page, so a single row costs decoding the whole
~100 MB column. ``repack_file`` copies the encoded bytes untouched into an uncompressed Arrow IPC file next to
nothing else; a training row then reads only its own ~100 KB through the page cache.

An index row points at a clip with a ref string ``<file>.arrow#<row>``; ``read_blob`` resolves it.

    <blobs>/<parent dir of the parquet>/<parquet stem>.arrow     audio: large_binary
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
import time
from collections import OrderedDict
from pathlib import Path

import pyarrow as pa

BLOB_SUFFIX = ".arrow"
BLOB_VERSION = "1"
_REF = re.compile(r"^(?P<file>.+\.arrow)#(?P<row>\d+)$")
_MAX_OPEN = 512


def blob_ref(file: str | Path, row: int) -> str:
    return f"{file}#{int(row)}"


def parse_blob_ref(ref: str) -> tuple[str, int] | None:
    match = _REF.match(str(ref))
    return (match["file"], int(match["row"])) if match else None


def blob_path(blobs: str | Path, parquet: str | Path) -> Path:
    """``<blobs>/<locale>/<stem>.arrow``: the parent dir keeps same-named shards of different dirs apart."""
    parquet = Path(parquet)
    return Path(os.path.abspath(blobs)) / parquet.parent.name / f"{parquet.stem}{BLOB_SUFFIX}"


def _holds_bytes(kind: pa.DataType) -> bool:
    if pa.types.is_binary(kind) or pa.types.is_large_binary(kind):
        return True
    return pa.types.is_struct(kind) and any(_holds_bytes(kind.field(i).type) for i in range(kind.num_fields))


def embedded_audio_column(schema: pa.Schema) -> str | None:
    """The one top-level column that carries bytes (``binary`` or HF's ``struct<bytes, path>``)."""
    found = [field.name for field in schema if _holds_bytes(field.type)]
    if len(found) > 1:
        raise ValueError(f"more than one binary column {found}; cannot tell which is the audio")
    return found[0] if found else None


def _bytes_of(column: pa.ChunkedArray) -> pa.Array:
    array = column.combine_chunks()
    if pa.types.is_struct(array.type):
        array = array.field("bytes")
    return array.cast(pa.large_binary())


def source_stamp(path: str | Path) -> dict[str, str]:
    stat = Path(path).stat()
    return {"source": str(path), "size": str(stat.st_size), "mtime_ns": str(stat.st_mtime_ns), "version": BLOB_VERSION}


def blob_is_current(parquet: str | Path, blob: str | Path) -> bool:
    blob = Path(blob)
    if not blob.is_file():
        return False
    try:
        meta = pa.ipc.open_file(pa.memory_map(str(blob))).schema.metadata or {}
    except (OSError, pa.ArrowInvalid):
        return False
    stored = {k.decode(): v.decode() for k, v in meta.items()}
    want = source_stamp(parquet)
    return all(stored.get(k) == v for k, v in want.items() if k != "source")


def check_blob(parquet: str | Path, blob: str | Path, rows: int) -> None:
    """Index build: the blob must come from this exact parquet and hold every one of its rows."""
    if not blob_is_current(parquet, blob):
        raise FileNotFoundError(
            f"{blob} is missing or older than {parquet}; run: bash scripts/data/repack_audio.sh (same config)"
        )
    held = pa.ipc.open_file(pa.memory_map(str(blob))).read_all().num_rows
    if held != rows:
        raise ValueError(f"{blob} holds {held} rows, {parquet} has {rows}")


def repack_file(job: tuple[str, str, bool]) -> dict:
    """One parquet → one Arrow file with only the audio bytes, one record batch per row group."""
    import pyarrow.parquet as pq

    parquet, blob, force = job
    started = time.monotonic()
    if not force and blob_is_current(parquet, blob):
        return {"file": parquet, "skipped": True, "rows": 0, "bytes": 0, "seconds": 0.0}
    handle = pq.ParquetFile(parquet)
    column = embedded_audio_column(handle.schema_arrow)
    if column is None:
        raise ValueError(f"{parquet} has no binary audio column")
    schema = pa.schema([("audio", pa.large_binary())], metadata=source_stamp(parquet))
    dest = Path(blob)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + f".partial-{os.getpid()}")
    rows = size = 0
    try:
        with pa.OSFile(str(partial), "wb") as sink, pa.ipc.new_file(sink, schema) as writer:
            for group in range(handle.num_row_groups):
                audio = _bytes_of(handle.read_row_group(group, columns=[column]).column(0))
                writer.write_batch(pa.record_batch([audio], schema=schema))
                rows += len(audio)
                size += audio.nbytes
        os.replace(partial, dest)
    finally:
        partial.unlink(missing_ok=True)
    return {"file": parquet, "skipped": False, "rows": rows, "bytes": size, "seconds": time.monotonic() - started}


def repack(pairs: list[tuple[str, str]], workers: int, force: bool = False, log=print) -> dict:
    jobs = [(str(p), str(b), bool(force)) for p, b in pairs]
    started = last = time.monotonic()
    done = skipped = rows = size = 0
    if not jobs:
        return {"files": 0, "skipped": 0, "rows": 0, "bytes": 0}
    with mp.get_context("fork").Pool(max(1, min(int(workers), len(jobs)))) as pool:
        for result in pool.imap_unordered(repack_file, jobs):
            done += 1
            skipped += int(result["skipped"])
            rows += result["rows"]
            size += result["bytes"]
            if time.monotonic() - last > 10 or done == len(jobs):
                last = time.monotonic()
                elapsed = last - started
                log(f"[repack] {done}/{len(jobs)} files ({skipped} up to date), {rows} rows, "
                    f"{size / 2**30:.1f} GiB, {size / 2**20 / max(elapsed, 1e-6):.0f} MiB/s, {int(elapsed)}s")
    return {"files": len(jobs), "skipped": skipped, "rows": rows, "bytes": size}


class _BlobFiles:
    """Memory-mapped blob files of this process, least recently used closed first."""

    def __init__(self, limit: int = _MAX_OPEN):
        self.limit = limit
        self.pid = -1
        self.open: OrderedDict[str, pa.ChunkedArray] = OrderedDict()

    def column(self, file: str) -> pa.ChunkedArray:
        if self.pid != os.getpid():
            # A forked DataLoader worker opens its own maps; the pages stay shared through the page cache.
            self.open.clear()
            self.pid = os.getpid()
        column = self.open.get(file)
        if column is None:
            column = pa.ipc.open_file(pa.memory_map(file)).read_all().column("audio")
            self.open[file] = column
            while len(self.open) > self.limit:
                self.open.popitem(last=False)
        else:
            self.open.move_to_end(file)
        return column


_FILES = _BlobFiles()


def read_blob(ref: str) -> bytes:
    parsed = parse_blob_ref(ref)
    if parsed is None:
        raise ValueError(f"not a blob ref: {ref}")
    file, row = parsed
    column = _FILES.column(file)
    if not 0 <= row < len(column):
        raise IndexError(f"{ref}: row out of range (file has {len(column)})")
    return column[row].as_py()
