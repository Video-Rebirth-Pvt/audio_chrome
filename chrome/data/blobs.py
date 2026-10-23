"""Audio embedded in Parquet (HF ``datasets`` style), repacked once into memory-mappable Arrow files.

HF shards keep every clip of a file in one ZSTD dictionary page, so a single row costs decoding the whole
~100 MB column. ``repack_file`` copies the encoded bytes untouched into an uncompressed Arrow IPC file next to
the row key; a training row then reads only its own ~100 KB through the page cache.

An index row points at a clip with a ref string ``<file>.arrow#<row>``; ``read_blob`` resolves it.

    <blobs>/<parent dir of the parquet>/<parquet stem>.arrow     audio: large_binary, segment_id (if the shard has it)

``migrate`` writes the Arrow file next to its parquet and, once the bytes and the sidecar rows are checked,
can delete the parquet; the sidecar then is the manifest (``chrome/data/manifest.py``).

Null audio is copied as null (WorldSpeech's metadata-only locales have no audio in any row); the index build
drops those rows as ``no_audio``, and ``migrate`` keeps the parquet of a shard without any audio.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import time
from collections import OrderedDict
from pathlib import Path

import pyarrow as pa

BLOB_SUFFIX = ".arrow"
BLOB_VERSION = "2"
# Row keys copied next to the audio so a sidecar can be checked against the blob row by row.
KEY_COLUMNS = ("segment_id", "id")
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


def key_column(names) -> str | None:
    return next((key for key in KEY_COLUMNS if key in names), None)


def _bytes_of(column: pa.ChunkedArray) -> pa.Array:
    array = column.combine_chunks()
    if pa.types.is_struct(array.type):
        # flatten() carries a null struct into its children; field() would not.
        array = array.flatten()[array.type.get_field_index("bytes")]
    return array.cast(pa.large_binary())


def source_stamp(path: str | Path) -> dict[str, str]:
    stat = Path(path).stat()
    return {"source": str(path), "size": str(stat.st_size), "mtime_ns": str(stat.st_mtime_ns), "version": BLOB_VERSION}


def _open_blob(blob: str | Path) -> pa.Table:
    return pa.ipc.open_file(pa.memory_map(str(blob))).read_all()


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
    held = _open_blob(blob).num_rows
    if held != rows:
        raise ValueError(f"{blob} holds {held} rows, {parquet} has {rows}")


def check_blob_rows(blob: str | Path, rows: int, first: int = 0, keys: dict[str, list] | None = None) -> None:
    """Sidecar manifest: the blob holds exactly ``rows`` clips and its row keys from ``first`` equal ``keys``."""
    if not Path(blob).is_file():
        raise FileNotFoundError(f"no {blob}; run: bash scripts/data/migrate_audio.sh")
    table = _open_blob(blob)
    if table.num_rows != rows:
        raise ValueError(f"{blob} holds {table.num_rows} rows, its manifest has {rows}")
    for key, values in (keys or {}).items():
        if key not in table.column_names:
            continue
        held = table[key].slice(first, len(values)).to_pylist()
        if held != values:
            row = next(i for i, (a, b) in enumerate(zip(held, values)) if a != b)
            raise ValueError(f"{blob} row {first + row}: {key} {held[row]!r} != manifest {values[row]!r}")


def null_audio_rows(blob: str | Path, first: int, count: int) -> list[int]:
    """Offsets (from ``first``) of the rows in [first, first + count) whose audio is null."""
    audio = _open_blob(blob)["audio"].slice(first, count)
    if audio.null_count == 0:
        return []
    return [i for i, valid in enumerate(audio.is_valid().to_pylist()) if not valid]


def _footer_null_audio(parquet: str | Path) -> int | None:
    """Null clips counted from the parquet footer statistics (no data read); None if they are missing."""
    import pyarrow.parquet as pq

    meta = pq.ParquetFile(parquet).metadata
    nulls = 0
    for g in range(meta.num_row_groups):
        group = meta.row_group(g)
        for i in range(group.num_columns):
            column = group.column(i)
            if column.path_in_schema not in ("audio", "audio.bytes"):
                continue
            stats = column.statistics
            if stats is None or not stats.has_null_count:
                return None
            nulls += stats.null_count
    return nulls


def check_sidecar(blob: str | Path, sidecar: str | Path) -> int:
    """Before a parquet is deleted: its sidecar exists, has the same rows and the same row keys as the blob."""
    import pyarrow.parquet as pq

    if not Path(sidecar).is_file():
        raise FileNotFoundError(f"no sidecar {sidecar}")
    handle = pq.ParquetFile(sidecar)
    key = key_column(_open_blob(blob).column_names)
    if key is None:
        raise ValueError(f"{blob} has no row key column {KEY_COLUMNS}; cannot check it against {sidecar}")
    if key not in handle.schema_arrow.names:
        raise ValueError(f"sidecar {sidecar} has no {key} column")
    values = handle.read(columns=[key]).column(0).to_pylist()
    check_blob_rows(blob, handle.metadata.num_rows, 0, {key: values})
    return len(values)


def _verify_written(path: Path, batches: list[pa.RecordBatch]) -> None:
    reader = pa.ipc.open_file(pa.memory_map(str(path)))
    if reader.num_record_batches != len(batches):
        raise ValueError(f"{path}: wrote {len(batches)} batches, reads back {reader.num_record_batches}")
    for i, batch in enumerate(batches):
        if not reader.get_batch(i).equals(batch):
            raise ValueError(f"{path}: batch {i} reads back different bytes")


def repack_file(job: tuple[str, str, bool]) -> dict:
    """One parquet → one Arrow file (audio bytes + row key), one record batch per row group, read back before use."""
    import pyarrow.parquet as pq

    parquet, blob, force = job
    started = time.monotonic()
    if not force and blob_is_current(parquet, blob):
        return {"file": parquet, "skipped": True, "rows": 0, "bytes": 0, "seconds": 0.0}
    handle = pq.ParquetFile(parquet)
    column = embedded_audio_column(handle.schema_arrow)
    if column is None:
        raise ValueError(f"{parquet} has no binary audio column")
    key = key_column(handle.schema_arrow.names)
    fields = [pa.field("audio", pa.large_binary())]
    if key:
        fields.append(handle.schema_arrow.field(key))
    schema = pa.schema(fields, metadata=source_stamp(parquet))
    dest = Path(blob)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + f".partial-{os.getpid()}")
    batches: list[pa.RecordBatch] = []
    rows = size = nulls = 0
    try:
        with pa.OSFile(str(partial), "wb") as sink, pa.ipc.new_file(sink, schema) as writer:
            for group in range(handle.num_row_groups):
                table = handle.read_row_group(group, columns=[column] + ([key] if key else []))
                arrays = [_bytes_of(table.column(column))]
                if key:
                    arrays.append(table.column(key).combine_chunks())
                batch = pa.record_batch(arrays, schema=schema)
                writer.write_batch(batch)
                batches.append(batch)
                rows += batch.num_rows
                size += arrays[0].nbytes
                nulls += arrays[0].null_count
        if rows != handle.metadata.num_rows:
            raise ValueError(f"{parquet}: copied {rows} rows of {handle.metadata.num_rows}")
        _verify_written(partial, batches)
        os.replace(partial, dest)
    finally:
        partial.unlink(missing_ok=True)
    return {"file": parquet, "skipped": False, "rows": rows, "bytes": size, "null_rows": nulls,
            "seconds": time.monotonic() - started}


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


def migrate_file(job: dict) -> dict:
    """Repack one parquet next to itself (unless current), check its sidecar, then optionally delete the parquet.

    A shard without any audio keeps its parquet and is reported as ``no_audio``.
    """
    import pyarrow.parquet as pq

    parquet, blob, sidecar = job["parquet"], job["blob"], job["sidecar"]
    result = {"file": parquet, "blob": blob, "repacked": False, "deleted": False, "no_audio": False,
              "rows": 0, "null_rows": 0, "bytes": 0}
    try:
        current = blob_is_current(parquet, blob)
        if job["dry_run"]:
            rows = pq.ParquetFile(parquet).metadata.num_rows
            nulls = _footer_null_audio(parquet)
            no_audio = rows > 0 and nulls == rows
            result.update(repacked=not current, deleted=bool(job["delete"]) and not no_audio, no_audio=no_audio,
                          rows=rows, null_rows=nulls or 0, bytes=Path(parquet).stat().st_size)
            return result
        if not current or job["force"]:
            done = repack_file((parquet, blob, True))
            result.update(repacked=True, bytes=done["bytes"])
        audio = _open_blob(blob)["audio"]
        result.update(rows=len(audio), null_rows=audio.null_count, no_audio=len(audio) > 0 and audio.null_count == len(audio))
        if sidecar:
            result["rows"] = check_sidecar(blob, sidecar)
        if job["delete"] and not result["no_audio"]:
            check_blob(parquet, blob, pq.ParquetFile(parquet).metadata.num_rows)
            freed = Path(parquet).stat().st_size
            os.remove(parquet)
            result.update(deleted=True, freed=freed)
    except Exception as error:  # one bad shard must not stop a 1.6 TB run; it is reported and kept
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def migrate(jobs: list[dict], workers: int, report: str | Path | None = None, log=print) -> dict:
    started = last = time.monotonic()
    totals = {"files": len(jobs), "repacked": 0, "deleted": 0, "failed": 0, "no_audio": 0, "rows": 0, "null_rows": 0,
              "bytes": 0, "freed": 0}
    # Locale (parent dir) → [shards without any audio, shards with some null clips, null clips].
    nulls: dict[str, list[int]] = {}
    failures: list[dict] = []
    dry_run = bool(jobs and jobs[0]["dry_run"])
    repacked, deleted = ("need a repack", "parquet to delete") if dry_run else ("repacked", "parquet deleted")
    sink = None
    if report and not dry_run:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        sink = open(report, "a", encoding="utf-8")
    try:
        with mp.get_context("fork").Pool(max(1, min(int(workers), len(jobs) or 1))) as pool:
            for done, result in enumerate(pool.imap_unordered(migrate_file, jobs), start=1):
                if "error" in result:
                    totals["failed"] += 1
                    failures.append(result)
                    log(f"[migrate] FAILED {result['file']}: {result['error']}")
                totals["repacked"] += int(result["repacked"])
                totals["deleted"] += int(result["deleted"])
                totals["no_audio"] += int(result["no_audio"])
                totals["rows"] += result["rows"]
                totals["null_rows"] += result["null_rows"]
                totals["bytes"] += result["bytes"]
                totals["freed"] += result.get("freed", 0)
                if result["null_rows"]:
                    entry = nulls.setdefault(Path(result["file"]).parent.name, [0, 0, 0])
                    entry[0 if result["no_audio"] else 1] += 1
                    entry[2] += result["null_rows"]
                if sink:
                    sink.write(json.dumps({**result, "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
                    sink.flush()
                if time.monotonic() - last > 10 or done == len(jobs):
                    last = time.monotonic()
                    elapsed = last - started
                    rate = "" if dry_run else f"{totals['bytes'] / 2**20 / max(elapsed, 1e-6):.0f} MiB/s, "
                    log(f"[migrate] {done}/{len(jobs)} files, {totals['repacked']} {repacked}, {totals['deleted']} "
                        f"{deleted}, {totals['no_audio']} without audio, {totals['failed']} failed, "
                        f"{totals['bytes'] / 2**30:.1f} GiB, {rate}{int(elapsed)}s")
    finally:
        if sink:
            sink.close()
    totals["failures"] = failures
    totals["nulls"] = dict(sorted(nulls.items()))
    return totals


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
