"""Train / validation DataLoaders over the built indexes.

    loader, sampler = build_train_loader(cfg, rank, world_size, state=ckpt.get("data"))
    for batch in loader:
        latents = encode_waves(vae, batch["waves"], device)        # list of [2, 32, 40·s]
        ...
        ckpt["data"] = sampler.state_dict(batch["batch_id"])       # resume after this batch
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from chrome.data.batch import collate_t2a
from chrome.data.dataset import BadAudioList, T2ADataset
from chrome.data.index import SourceIndex, SourceSpec, check_index, mixture_weight, source_specs
from chrome.data.sampler import FixedBatchSampler, MixtureBatchSampler, SourceStream, pack_in_order, sample_seed
from chrome.spec import AudioSpec


def open_sources(cfg, log=print) -> tuple[list[SourceSpec], list[SourceIndex]]:
    specs = source_specs(cfg)
    root = Path(cfg["data"]["index_root"])
    indexes = []
    for spec in specs:
        index = SourceIndex(root / spec.name)
        check_index(spec, index, cfg, log)
        indexes.append(index)
    return specs, indexes


def rows_per_second(spec: AudioSpec) -> int:
    return spec.rows_for_frames(spec.frames_for_seconds(1))


def sample_cost(index: SourceIndex, spec: AudioSpec) -> np.ndarray:
    """Packed DiT rows per index row: both ears' latent frames plus the caption tokens."""
    return rows_per_second(spec) * index.seconds.astype(np.int32) + index.text_tokens.astype(np.int32)


def train_rows(index: SourceIndex, subset: int | None = None) -> np.ndarray:
    """Non-holdout rows; with ``subset``, only the first N by path hash (stable, no index rebuild)."""
    rows = np.flatnonzero(~index.holdout).astype(np.int32)
    if subset is not None and rows.size > subset:
        rows = np.sort(rows[np.argsort(index.keys()[rows], kind="stable")[:subset]])
    return rows


def train_streams(
    specs: list[SourceSpec], indexes: list[SourceIndex], spec: AudioSpec, subset: int | None = None
) -> list[SourceStream]:
    tag = "" if subset is None else f":subset{subset}"
    streams = []
    for s, index in zip(specs, indexes):
        rows = train_rows(index, subset)
        streams.append(
            SourceStream(
                name=s.name,
                rows=rows,
                cost=sample_cost(index, spec),
                weight=mixture_weight(s, len(rows)),
                fingerprint=f"{index.meta['config_fingerprint']}:{len(index)}{tag}",
            )
        )
    return streams


def _worker_init(_: int) -> None:
    torch.set_num_threads(1)


def _dataset(cfg, indexes, crop: str, cfg_drop: float) -> T2ADataset:
    data = cfg["data"]
    return T2ADataset(
        indexes,
        AudioSpec.from_config(cfg),
        tokenizer_dir=data["tokenizer"],
        bad_audio=BadAudioList(data["bad_list"], data.get("extra_bad_lists") or []),
        cfg_drop=cfg_drop,
        peak=data.get("peak"),
        crop=crop,
    )


def _loader(dataset, batch_sampler, workers: int, prefetch: int) -> DataLoader:
    extra = {"prefetch_factor": prefetch, "persistent_workers": True} if workers > 0 else {}
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=workers,
        collate_fn=collate_t2a,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_init,
        **extra,
    )


def build_train_loader(
    cfg, rank: int = 0, world_size: int = 1, state: dict | None = None, log=print
) -> tuple[DataLoader, MixtureBatchSampler]:
    loader_cfg = cfg["data"]["loader"]
    specs, indexes = open_sources(cfg, log)
    subset = loader_cfg.get("subset")
    sampler = MixtureBatchSampler(
        train_streams(specs, indexes, AudioSpec.from_config(cfg), None if subset is None else int(subset)),
        max_tokens=int(loader_cfg["max_tokens"]),
        max_samples=int(loader_cfg["max_samples"]),
        rank=rank,
        world_size=world_size,
        seed=int(loader_cfg["seed"]),
    )
    if state is not None:
        sampler.load_state_dict(state, log)
    dataset = _dataset(cfg, indexes, crop=str(loader_cfg["crop"]), cfg_drop=float(loader_cfg["cfg_drop"]))
    loader = _loader(dataset, sampler, int(loader_cfg["num_workers"]), int(loader_cfg["prefetch_factor"]))
    return loader, sampler


def allocate_quota(sizes: list[int], budget: int) -> list[int]:
    """Keep ``budget`` items across groups; every non-empty group keeps at least one when possible.

    ``budget <= 0`` means keep all. If ``budget`` is smaller than the number of non-empty groups it is
    raised so a later source is never dropped just because it was packed after the others.
    """
    sizes = [max(0, int(n)) for n in sizes]
    total = sum(sizes)
    if budget <= 0 or budget >= total:
        return sizes
    alive = [i for i, n in enumerate(sizes) if n > 0]
    if not alive:
        return sizes
    if budget < len(alive):
        budget = len(alive)
    take = [0] * len(sizes)
    leftover = budget
    for i in alive:
        take[i] = 1
        leftover -= 1
    remain = [sizes[i] - take[i] for i in range(len(sizes))]
    pool = sum(remain)
    if leftover <= 0 or pool <= 0:
        return take
    extra = [leftover * remain[i] / pool for i in range(len(sizes))]
    add = [int(x) for x in extra]
    leftover -= sum(add)
    order = sorted(alive, key=lambda i: (extra[i] - add[i], remain[i], -i), reverse=True)
    while leftover > 0:
        progressed = False
        for i in order:
            if leftover <= 0:
                break
            if add[i] < remain[i]:
                add[i] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break
    return [take[i] + add[i] for i in range(len(sizes))]


def even_split(sizes: list[int], budget: int) -> list[int]:
    """Equal shares of ``budget``; what a small group cannot use goes to the others (in order)."""
    take = [0] * len(sizes)
    left = max(0, int(budget))
    open_ = [i for i, n in enumerate(sizes) if n > 0]
    while left > 0 and open_:
        share = max(1, left // len(open_))
        for i in list(open_):
            add = min(share, sizes[i] - take[i], left)
            take[i] += add
            left -= add
            if take[i] == sizes[i]:
                open_.remove(i)
            if left == 0:
                break
    return take


def holdout_rows(specs: list[SourceSpec], indexes: list[SourceIndex], per_source: int) -> list[np.ndarray]:
    """Up to ``per_source`` holdout rows per val group, smallest path hash first.

    A source without ``group`` is its own group. In a shared group the budget is split evenly over its
    tasks; inside a task, the rows of all its sources are merged in hash order.
    """
    chosen = [np.empty(0, dtype=np.int32) for _ in specs]
    keys = [index.keys() for index in indexes]
    groups: dict[str, list[int]] = {}
    for i, spec in enumerate(specs):
        groups.setdefault(spec.val_group, []).append(i)
    for members in groups.values():
        tasks: dict[str, list[int]] = {}
        for i in members:
            tasks.setdefault(specs[i].task, []).append(i)
        pools = []
        for sources in tasks.values():
            pool = [(int(keys[i][r]), i, int(r)) for i in sources for r in np.flatnonzero(indexes[i].holdout)]
            pools.append(sorted(pool))
        for pool, n in zip(pools, even_split([len(p) for p in pools], per_source)):
            for i in {i for _, i, _ in pool[:n]}:
                rows = np.array([r for _, j, r in pool[:n] if j == i], dtype=np.int32)
                chosen[i] = rows[np.argsort(keys[i][rows], kind="stable")]
    return chosen


def _val_batch_limit(cfg) -> int:
    train = cfg.get("train")
    if train is None:
        return 0
    return int(train.get("val_batches") or 0)


def validation_batches(cfg, specs: list[SourceSpec], indexes: list[SourceIndex], log=print) -> list[list[tuple]]:
    """Holdout clips, capped per source then packed. Batches never mix sources.

    ``data.val.per_source`` is the clip cap per source, or per ``group`` of sources (hash-stable, see
    ``holdout_rows``). ``data.val.max_total_clips``
    (0 = off) is a global safety cap shared across sources. ``train.val_batches`` (0 = all packed
    batches) is a batch budget, also shared. Both budgets keep every source that still has holdout
    rows, so adding a dataset cannot wipe the ones packed after it.
    """
    data, audio = cfg["data"], AudioSpec.from_config(cfg)
    per_source = int(data["val"]["per_source"])
    max_total = int(data["val"].get("max_total_clips") or 0)
    seed = int(data["loader"]["seed"])
    for spec, index in zip(specs, indexes):
        if not index.holdout.any():
            log(f"[val] {spec.name}: no holdout rows, skipped")
    chosen = holdout_rows(specs, indexes, per_source)
    counts = [int(rows.size) for rows in chosen]
    clipped = allocate_quota(counts, max_total)
    if clipped != counts:
        log(f"[val] max_total_clips={max_total} scaled {counts} → {clipped} (every source with holdout kept)")
        chosen = [rows[:n] for rows, n in zip(chosen, clipped)]

    packed_by_source: list[list[list[tuple]]] = []
    for source, (spec, index, rows) in enumerate(zip(specs, indexes, chosen)):
        if rows.size == 0:
            packed_by_source.append([])
            continue
        cost = sample_cost(index, audio)
        items = [(source, int(r), sample_seed(seed, source, int(r))) for r in rows]
        packed_by_source.append(
            pack_in_order(
                items, [int(cost[r]) for r in rows], int(data["loader"]["max_tokens"]), int(data["loader"]["max_samples"])
            )
        )
    batch_counts = [len(groups) for groups in packed_by_source]
    kept = allocate_quota(batch_counts, _val_batch_limit(cfg))
    if kept != batch_counts:
        log(f"[val] val_batches={_val_batch_limit(cfg)} kept {kept} of {batch_counts} batches (source-balanced)")
        packed_by_source = [groups[:n] for groups, n in zip(packed_by_source, kept)]

    batches: list[list[tuple]] = []
    parts = []
    for spec, rows, groups in zip(specs, chosen, packed_by_source):
        clips = sum(len(batch) for batch in groups)
        if clips:
            parts.append(f"{spec.name}={clips}/{len(groups)}")
        for batch in groups:
            batches.append([(len(batches) + 1, *item[1:]) for item in batch])
    log(f"[val] {sum(len(b) for b in batches)} clips / {len(batches)} batches | " + " ".join(parts)
        + f" (per_source={per_source}, max_total_clips={max_total or 'off'}, val_batches={_val_batch_limit(cfg) or 'all'})")
    return batches


def build_val_loader(cfg, rank: int = 0, world_size: int = 1, log=print) -> DataLoader:
    """Fixed holdout clips: center windows, no caption dropout; the same set every time."""
    specs, indexes = open_sources(cfg, log)
    batches = validation_batches(cfg, specs, indexes, log=log)
    dataset = _dataset(cfg, indexes, crop="center", cfg_drop=0.0)
    val = cfg["data"]["val"]
    return _loader(dataset, FixedBatchSampler(batches, rank, world_size), int(val["num_workers"]), 2)
