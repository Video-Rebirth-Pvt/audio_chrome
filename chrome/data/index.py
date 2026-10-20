"""Per-source training index: manifests scanned once in parallel, stored as a memory-mapped Arrow file.

Layout under ``data.index_root``::

    <source>/index.arrow   path, caption, duration, seconds, sample_rate, text_tokens, key, holdout
    <source>/meta.json     fingerprints, drop counts, hours, length and caption histograms

Every row is already a training clip: ``seconds`` is the whole-second length it is cut or padded to.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
from omegaconf import OmegaConf

from chrome.data.manifest import Columns, first_value, iter_rows, manifest_stamp, plan_chunks, row_kept

INDEX_VERSION = 1
INDEX_FILE = "index.arrow"
META_FILE = "meta.json"
HOLDOUT_SCALE = 1_000_000

SCHEMA = pa.schema(
    [
        ("path", pa.large_string()),
        ("caption", pa.large_string()),
        ("duration", pa.float32()),
        ("seconds", pa.uint8()),
        ("sample_rate", pa.int32()),
        ("text_tokens", pa.int32()),
        ("key", pa.uint64()),
    ]
)


@dataclass(frozen=True)
class LengthPolicy:
    """Whole-second clip lengths in [min_seconds, max_seconds].

    An overrun of less than ``slack`` past a whole second is cropped off. Otherwise ``fit: pad`` rounds up
    and zero-pads (at most one second; speech and events keep their endings) and ``fit: crop`` rounds down
    and takes a random window (music). Longer than ``max + slack``: ``long: drop`` or ``long: crop``.
    """

    min_seconds: int
    max_seconds: int
    slack: float = 0.1

    def seconds_for(self, duration: float, fit: str, long: str) -> int | str:
        if duration < self.min_seconds - 1e-3:
            return "too_short"
        if duration >= self.max_seconds + self.slack:
            return self.max_seconds if long == "crop" else "too_long"
        whole = math.floor(duration + 1e-6)
        seconds = whole if (duration - whole < self.slack or fit == "crop") else whole + 1
        return min(max(seconds, self.min_seconds), self.max_seconds)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    paths: tuple[str, ...]
    task: str = "misc"
    # None: loader mixes this source in proportion to its train rows (after max_records).
    weight: float | None = None
    columns: Columns = field(default_factory=Columns)
    keep: Mapping = field(default_factory=dict)
    fit: str = "pad"
    long: str = "drop"
    caption_prefix: str = ""
    max_records: int | None = None
    holdout_ppm: int = 500

    @classmethod
    def from_config(cls, source, defaults) -> "SourceSpec":
        merged = {**_plain(defaults), **_plain(source)}
        paths = merged.get("paths") or ([merged["path"]] if merged.get("path") else [])
        if not merged.get("name") or not paths:
            raise ValueError(f"every source needs a name and paths: {merged}")
        spec = cls(
            name=str(merged["name"]),
            paths=tuple(str(p) for p in paths),
            task=str(merged.get("task", "misc")),
            weight=_optional_float(merged.get("weight")),
            columns=Columns.from_config(merged.get("columns")),
            keep=merged.get("keep") or {},
            fit=str(merged.get("fit", "pad")),
            long=str(merged.get("long", "drop")),
            caption_prefix=str(merged.get("caption_prefix") or ""),
            max_records=None if merged.get("max_records") in (None, "") else int(merged["max_records"]),
            holdout_ppm=int(merged.get("holdout_ppm", 500)),
        )
        if spec.fit not in ("pad", "crop") or spec.long not in ("drop", "crop"):
            raise ValueError(f"{spec.name}: fit must be pad|crop and long drop|crop")
        if spec.weight is not None and spec.weight < 0:
            raise ValueError(f"{spec.name}: weight must be >= 0")
        if spec.max_records is not None and spec.max_records < 0:
            raise ValueError(f"{spec.name}: max_records must be >= 0")
        return spec

    def content(self) -> dict:
        """Fields that change the index contents (weight and task do not)."""
        data = asdict(self)
        data.pop("weight")
        data.pop("task")
        return data


def _optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def mixture_weight(spec: "SourceSpec", n_rows: int) -> float:
    """Sampling weight: an explicit ``weight``, otherwise the number of train rows kept."""
    return float(n_rows) if spec.weight is None else float(spec.weight)


def _plain(value) -> dict:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return dict(value)


def source_specs(cfg) -> list[SourceSpec]:
    data = cfg["data"]
    specs = [SourceSpec.from_config(s, data.get("source_defaults")) for s in data["sources"]]
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"source names must be unique: {names}")
    return specs


def length_policy(cfg) -> LengthPolicy:
    audio, data = cfg["audio"], cfg["data"]
    return LengthPolicy(int(audio["min_seconds"]), int(audio["max_seconds"]), float(data["slack_seconds"]))


def path_key(path: str) -> int:
    return int.from_bytes(hashlib.blake2b(path.encode("utf-8"), digest_size=8).digest(), "little")


def read_path_list(path: str | Path) -> set[str]:
    """Paths from a preview-style JSON list (``audio`` field) or a text file (first tab field per line)."""
    path = Path(path)
    if not path.is_file():
        return set()
    if path.suffix == ".json":
        items = json.loads(path.read_text(encoding="utf-8"))
        return {str(i["audio"] if isinstance(i, Mapping) else i) for i in items}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {line.split("\t", 1)[0].strip() for line in lines if line.strip()}


def _paths_from(lists) -> set[str]:
    found: set[str] = set()
    for item in lists or []:
        found |= read_path_list(item)
    return found


def config_fingerprint(spec: SourceSpec, cfg) -> str:
    data = cfg["data"]
    payload = {
        "version": INDEX_VERSION,
        "source": spec.content(),
        "policy": asdict(length_policy(cfg)),
        "max_text_tokens": int(data["max_text_tokens"]),
        "tokenizer": str(data["tokenizer"]),
        "exclude": sorted(_paths_from(data.get("exclude"))),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- build (worker side)

_WORKER: dict = {}


def _init_worker(tokenizer_dir: str, exclude: set[str], bad: set[str]) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from chrome.data.captions import load_caption_tokenizer

    _WORKER.update(tokenizer=load_caption_tokenizer(tokenizer_dir), exclude=exclude, bad=bad)


def probe_duration(path: str) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(path)
        if info.frames > 0 and info.samplerate > 0:
            return info.frames / info.samplerate
    except Exception:
        pass
    try:
        import torchaudio

        info = torchaudio.info(path)
        if info.num_frames > 0 and info.sample_rate > 0:
            return info.num_frames / info.sample_rate
    except Exception:
        pass
    return None


def _number(value, cast):
    try:
        return cast(float(value))
    except (TypeError, ValueError):
        return None


def _scan_chunk(job: dict) -> dict:
    from chrome.data.captions import caption_lengths

    spec: SourceSpec = job["spec"]
    policy: LengthPolicy = job["policy"]
    exclude, bad = _WORKER["exclude"], _WORKER["bad"]
    drops: Counter = Counter()
    scanned = 0
    cols = {name: [] for name in ("path", "caption", "duration", "seconds", "sample_rate")}
    for row in iter_rows(job["chunk"]):
        scanned += 1
        if row is None:
            drops["bad_json"] += 1
            continue
        if spec.keep and not row_kept(row, spec.keep):
            drops["keep_filter"] += 1
            continue
        path = first_value(row, spec.columns.audio)
        if not path:
            drops["no_audio"] += 1
            continue
        path = str(path)
        caption = str(first_value(row, spec.columns.caption) or "").strip()
        if not caption:
            drops["no_caption"] += 1
            continue
        if path in exclude:
            drops["excluded"] += 1
            continue
        if path in bad:
            drops["bad_audio"] += 1
            continue
        duration = _number(first_value(row, spec.columns.duration), float)
        if not duration or duration <= 0:
            duration = probe_duration(path)
            if duration is None:
                drops["probe_failed"] += 1
                continue
        seconds = policy.seconds_for(duration, spec.fit, spec.long)
        if isinstance(seconds, str):
            drops[seconds] += 1
            continue
        cols["path"].append(path)
        cols["caption"].append(spec.caption_prefix + caption)
        cols["duration"].append(duration)
        cols["seconds"].append(seconds)
        cols["sample_rate"].append(_number(first_value(row, spec.columns.sample_rate), int) or 0)

    tokens = np.asarray(caption_lengths(_WORKER["tokenizer"], cols["caption"]), dtype=np.int32)
    fits = tokens <= job["max_text_tokens"]
    drops["caption_too_long"] += int((~fits).sum())
    keep = np.flatnonzero(fits)
    table = pa.table(
        {
            "path": pa.array([cols["path"][i] for i in keep], pa.large_string()),
            "caption": pa.array([cols["caption"][i] for i in keep], pa.large_string()),
            "duration": pa.array(np.asarray(cols["duration"], np.float32)[keep]),
            "seconds": pa.array(np.asarray(cols["seconds"], np.uint8)[keep]),
            "sample_rate": pa.array(np.asarray(cols["sample_rate"], np.int32)[keep]),
            "text_tokens": pa.array(tokens[keep]),
            "key": pa.array(np.asarray([path_key(cols["path"][i]) for i in keep], np.uint64)),
        },
        schema=SCHEMA,
    )
    with pa.OSFile(job["dest"], "wb") as sink, pa.ipc.new_file(sink, SCHEMA) as writer:
        writer.write_table(table)
    return {"scanned": scanned, "kept": table.num_rows, "drops": dict(drops)}


# ---------------------------------------------------------------- build (parent side)


def _histogram(values: np.ndarray) -> dict[str, int]:
    keys, counts = np.unique(values, return_counts=True)
    return {str(int(k)): int(c) for k, c in zip(keys, counts)}


def _stats(table: pa.Table, train: np.ndarray) -> dict:
    seconds = table["seconds"].to_numpy()[train].astype(np.int64)
    duration = table["duration"].to_numpy()[train].astype(np.float64)
    tokens = table["text_tokens"].to_numpy()[train]
    rates = table["sample_rate"].to_numpy()[train]
    if seconds.size == 0:
        return {"clip_hours": 0.0, "source_hours": 0.0}
    return {
        "clip_hours": float(seconds.sum() / 3600),
        "source_hours": float(duration.sum() / 3600),
        "mean_seconds": float(seconds.mean()),
        "seconds_hist": _histogram(seconds),
        "padded_rows": int((duration < seconds).sum()),
        "cropped_rows": int((duration > seconds + 1e-3).sum()),
        "text_tokens": {
            "mean": float(tokens.mean()),
            "p50": int(np.percentile(tokens, 50)),
            "p95": int(np.percentile(tokens, 95)),
            "p99": int(np.percentile(tokens, 99)),
            "max": int(tokens.max()),
        },
        "sample_rate_hist": _histogram(rates),
    }


def build_source(spec: SourceSpec, cfg, workers: int, force: bool = False, log=print) -> dict:
    data = cfg["data"]
    out = Path(data["index_root"]) / spec.name
    meta_path = out / META_FILE
    config_fp = config_fingerprint(spec, cfg)
    stamps = manifest_stamp(spec.paths)
    if not force and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if meta.get("config_fingerprint") == config_fp and meta.get("manifests") == [list(s) for s in stamps]:
            log(f"[index] {spec.name}: up to date ({meta['train_rows']} train rows), skipped")
            return meta

    started = time.monotonic()
    chunks = plan_chunks(spec.paths, max(1, int(float(data["build"]["chunk_mb"]) * (1 << 20))))
    tmp = out / f".tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    policy = length_policy(cfg)
    jobs = [
        {
            "spec": spec,
            "policy": policy,
            "chunk": chunk,
            "max_text_tokens": int(data["max_text_tokens"]),
            "dest": str(tmp / f"{i:06d}.arrow"),
        }
        for i, chunk in enumerate(chunks)
    ]
    exclude = _paths_from(data.get("exclude"))
    bad = _paths_from([data["bad_list"], *(data.get("extra_bad_lists") or [])])
    drops: Counter = Counter()
    scanned = kept = 0
    log(f"[index] {spec.name}: {len(chunks)} chunks from {len(stamps)} manifest file(s), {workers} workers")
    last = time.monotonic()
    context = mp.get_context("fork")
    try:
        with context.Pool(
            min(workers, len(jobs)), initializer=_init_worker, initargs=(str(data["tokenizer"]), exclude, bad)
        ) as pool:
            for done, result in enumerate(pool.imap_unordered(_scan_chunk, jobs), start=1):
                scanned += result["scanned"]
                kept += result["kept"]
                drops.update(result["drops"])
                if time.monotonic() - last > 10 or done == len(jobs):
                    last = time.monotonic()
                    log(
                        f"[index] {spec.name}: {done}/{len(jobs)} chunks, scanned {scanned}, kept {kept}, "
                        f"{int(last - started)}s"
                    )

        tables = [pa.ipc.open_file(pa.memory_map(job["dest"])).read_all() for job in jobs]
        table = pa.concat_tables(tables)
        keys = table["key"].to_numpy()
        _, first = np.unique(keys, return_index=True)
        if first.size < table.num_rows:
            drops["duplicate"] += table.num_rows - first.size
            table = table.take(pa.array(np.sort(first)))
            keys = table["key"].to_numpy()
        holdout = (keys % HOLDOUT_SCALE) < spec.holdout_ppm
        train = ~holdout
        if spec.max_records is not None and train.sum() > spec.max_records:
            # Hash order is a fixed random order, so the cap is stable across rebuilds.
            rows = np.flatnonzero(train)
            chosen = rows[np.argpartition(keys[rows], spec.max_records)[: spec.max_records]]
            drops["max_records"] += int(rows.size - chosen.size)
            selected = np.zeros(table.num_rows, bool)
            selected[chosen] = True
            selected |= holdout
            table = table.filter(pa.array(selected))
            holdout = holdout[selected]
            train = ~holdout
        table = table.append_column("holdout", pa.array(holdout))

        out.mkdir(parents=True, exist_ok=True)
        partial = out / (INDEX_FILE + ".partial")
        with pa.OSFile(str(partial), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table, max_chunksize=1 << 20)
        os.replace(partial, out / INDEX_FILE)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    meta = {
        "version": INDEX_VERSION,
        "name": spec.name,
        "config_fingerprint": config_fp,
        "manifests": [list(s) for s in stamps],
        "source": spec.content(),
        "scanned": scanned,
        "rows": table.num_rows,
        "train_rows": int(train.sum()),
        "holdout_rows": int(holdout.sum()),
        "drops": dict(sorted(drops.items(), key=lambda kv: -kv[1])),
        "train": _stats(table, train),
        "build_seconds": round(time.monotonic() - started, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    log(
        f"[index] {spec.name}: {meta['train_rows']} train + {meta['holdout_rows']} holdout rows, "
        f"{meta['train']['clip_hours']:.0f} h, {meta['build_seconds']}s"
    )
    return meta


# ---------------------------------------------------------------- read


class SourceIndex:
    """One source's index. Arrays the sampler needs are loaded eagerly; strings are read per row from the map."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        meta_path = self.root / META_FILE
        if not meta_path.is_file():
            raise FileNotFoundError(f"no index at {self.root}; run scripts/data/build_index.sh")
        self.meta = json.loads(meta_path.read_text())
        self._table: pa.Table | None = None
        self._pid = -1
        table = self.table
        self.seconds = table["seconds"].to_numpy()
        self.text_tokens = table["text_tokens"].to_numpy()
        self.holdout = table["holdout"].to_numpy(zero_copy_only=False)

    @property
    def name(self) -> str:
        return self.meta["name"]

    @property
    def table(self) -> pa.Table:
        # Re-open after fork so every DataLoader worker owns its file handle; the pages stay shared.
        if self._table is None or self._pid != os.getpid():
            self._table = pa.ipc.open_file(pa.memory_map(str(self.root / INDEX_FILE))).read_all()
            self._pid = os.getpid()
        return self._table

    def __len__(self) -> int:
        return len(self.seconds)

    def row(self, index: int) -> tuple[str, str, float]:
        table = self.table
        return table["path"][index].as_py(), table["caption"][index].as_py(), float(table["duration"][index].as_py())

    def keys(self) -> np.ndarray:
        return self.table["key"].to_numpy()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_table"] = None
        return state


def check_index(spec: SourceSpec, index: SourceIndex, cfg, log=print) -> None:
    """Config changes that alter the rows must rebuild; a touched manifest only warns."""
    if index.meta.get("config_fingerprint") != config_fingerprint(spec, cfg):
        raise RuntimeError(
            f"index for source {spec.name!r} was built from a different config; "
            f"rerun: bash scripts/data/build_index.sh data.only=[{spec.name}]"
        )
    if index.meta.get("manifests") != [list(s) for s in manifest_stamp(spec.paths)]:
        log(f"[data] warning: manifests of {spec.name!r} changed since the index was built")
