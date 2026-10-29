"""Batch samplers. Training: an endless mixture of sources, packed to a token budget, resumable.

Every sample is drawn by first picking a source with probability ∝ weight, then taking the next row of
that source's shard. Weight defaults to the number of train rows kept (``max_records``), so the mix
follows the counts in the source yaml. A shard is this rank's slice ``[rank::world]`` of a per-source
permutation seeded by (seed, source name, source epoch); ranks therefore see disjoint rows within a
source epoch, and a small source simply wraps into its next epoch more often.

Batches are lists of ``(batch_id, source, row, sample_seed)``. A batch closes when the next sample would
push ``sum(cost)`` past ``max_tokens`` (cost = DiT audio rows + caption tokens); that sample opens the
next batch, so nothing is dropped. With ``silence_pad`` (chrome.data.silence) every item gets a fifth
field, the whole seconds of silence it takes from the batch's spare rows. The sampler keeps a snapshot
of its state after each recent batch; the trainer saves ``state_dict(batch_id)`` for the last batch it
consumed, and ``load_state_dict`` continues with exactly the following batch.
"""

from __future__ import annotations

import zlib
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Sampler

from chrome.data.silence import PAD_TAG, SilencePad

STATE_VERSION = 1
_MASK = (1 << 64) - 1


def _splitmix(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK
    return x ^ (x >> 31)


def sample_seed(*values: int) -> int:
    x = 0
    for value in values:
        x = _splitmix(x ^ (value & _MASK))
    return x >> 1


@dataclass
class SourceStream:
    """Train rows of one source and the packing cost of every row of its index."""

    name: str
    rows: np.ndarray
    cost: np.ndarray
    weight: float
    fingerprint: str
    # Clip seconds of every index row; only silence padding reads it.
    seconds: np.ndarray | None = None


def shard(stream: SourceStream, seed: int, epoch: int, rank: int, world_size: int) -> np.ndarray:
    order = np.random.default_rng([seed, zlib.crc32(stream.name.encode()), epoch]).permutation(len(stream.rows))
    if len(order) >= world_size:
        order = order[rank::world_size]
    else:
        order = np.roll(order, -rank)
    return stream.rows[order]


class MixtureBatchSampler(Sampler):
    def __init__(
        self,
        streams: list[SourceStream],
        max_tokens: int,
        max_samples: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        history: int = 4096,
        silence_pad: SilencePad | None = None,
    ):
        active = [s for s in streams if s.weight > 0 and len(s.rows) > 0]
        if not active:
            raise ValueError("no source has weight > 0 and train rows")
        largest = max(int(s.cost[s.rows].max()) for s in active)
        if largest > max_tokens:
            raise ValueError(f"max_tokens={max_tokens} is below the largest sample cost {largest}")
        self.streams = streams
        self.max_tokens = int(max_tokens)
        self.max_samples = int(max_samples)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.history = int(history)
        if silence_pad is not None and any(streams[i].seconds is None for i in silence_pad.eligible):
            raise ValueError("silence_pad needs the clip seconds of every padded stream")
        self.silence_pad = silence_pad
        weights = np.array([s.weight if len(s.rows) > 0 else 0.0 for s in streams], dtype=np.float64)
        self._cumulative = np.cumsum(weights / weights.sum())
        self._cumulative[-1] = 1.0
        self._rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed, self.rank, 0x5EED])))
        self._epoch = [0] * len(streams)
        self._cursor = [0] * len(streams)
        self._shards: list[np.ndarray | None] = [None] * len(streams)
        self._carry: tuple[int, int, int] | None = None
        self._batch_id = 0
        self._samples = 0
        self._snapshots: OrderedDict[int, dict] = OrderedDict()

    def _draw(self) -> tuple[int, int, int]:
        source = int(np.searchsorted(self._cumulative, self._rng.random(), side="right"))
        rows = self._shards[source]
        if rows is None or self._cursor[source] >= len(rows):
            if rows is not None:
                self._epoch[source] += 1
                self._cursor[source] = 0
            stream = self.streams[source]
            rows = shard(stream, self.seed, self._epoch[source], self.rank, self.world_size)
            self._shards[source] = rows
        row = int(rows[self._cursor[source]])
        self._cursor[source] += 1
        self._samples += 1
        return source, row, sample_seed(self.seed, self.rank, self._samples)

    def next_batch(self) -> list[tuple[int, ...]]:
        batch: list[tuple[int, int, int]] = []
        tokens = 0
        while True:
            item, self._carry = (self._carry or self._draw()), None
            cost = int(self.streams[item[0]].cost[item[1]])
            if batch and (tokens + cost > self.max_tokens or len(batch) >= self.max_samples):
                self._carry = item
                break
            batch.append(item)
            tokens += cost
        self._batch_id += 1
        self._snapshots[self._batch_id] = self._snapshot()
        while len(self._snapshots) > self.history:
            self._snapshots.popitem(last=False)
        if self.silence_pad is None:
            return [(self._batch_id, source, row, seed) for source, row, seed in batch]
        extras = self.silence_pad.extras(batch, [s.seconds for s in self.streams], self.max_tokens - tokens,
                                         sample_seed(self.seed, self.rank, self._batch_id, PAD_TAG))
        return [(self._batch_id, source, row, seed, extra) for (source, row, seed), extra in zip(batch, extras)]

    def __iter__(self) -> Iterator[list[tuple[int, int, int, int]]]:
        while True:
            yield self.next_batch()

    def _snapshot(self) -> dict:
        carry = None
        if self._carry is not None:
            source, row, seed = self._carry
            carry = [self.streams[source].name, row, seed]
        return {
            "batch_id": self._batch_id,
            "samples": self._samples,
            "rng": self._rng.bit_generator.state,
            "carry": carry,
            "sources": {
                s.name: {"epoch": self._epoch[i], "cursor": self._cursor[i], "fingerprint": s.fingerprint}
                for i, s in enumerate(self.streams)
            },
        }

    def state_dict(self, batch_id: int | None = None) -> dict:
        """State right after ``batch_id`` (default: the newest batch handed out, which may be prefetched)."""
        if not self._snapshots:
            raise RuntimeError("no batch has been drawn yet")
        batch_id = next(reversed(self._snapshots)) if batch_id is None else int(batch_id)
        if batch_id not in self._snapshots:
            raise KeyError(f"batch {batch_id} is older than the last {self.history} snapshots")
        return {
            "version": STATE_VERSION,
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            **self._snapshots[batch_id],
        }

    def load_state_dict(self, state: dict, log=print) -> None:
        if state.get("version") != STATE_VERSION:
            raise ValueError(f"sampler state version {state.get('version')} != {STATE_VERSION}")
        if (state["rank"], state["world_size"]) != (self.rank, self.world_size):
            raise ValueError(
                f"sampler state is for rank {state['rank']}/{state['world_size']}, "
                f"not {self.rank}/{self.world_size}; start the data stream fresh instead"
            )
        if state["seed"] != self.seed:
            log(f"[data] warning: sampler state seed {state['seed']} != configured {self.seed}")
        self._rng.bit_generator.state = state["rng"]
        self._batch_id = int(state["batch_id"])
        self._samples = int(state["samples"])
        saved = state["sources"]
        for i, stream in enumerate(self.streams):
            entry = saved.get(stream.name)
            if entry is None:
                self._epoch[i], self._cursor[i] = 0, 0
            elif entry["fingerprint"] == stream.fingerprint:
                self._epoch[i], self._cursor[i] = int(entry["epoch"]), int(entry["cursor"])
            else:
                log(f"[data] {stream.name}: index changed since the checkpoint; starting its next epoch")
                self._epoch[i], self._cursor[i] = int(entry["epoch"]) + 1, 0
        self._shards = [None] * len(self.streams)
        for i, stream in enumerate(self.streams):
            if self._cursor[i] > 0:
                self._shards[i] = shard(stream, self.seed, self._epoch[i], self.rank, self.world_size)
        self._carry = None
        if state.get("carry"):
            name, row, seed = state["carry"]
            names = [s.name for s in self.streams]
            if name in names:
                source = names.index(name)
                entry = saved.get(name)
                if entry is not None and entry["fingerprint"] == self.streams[source].fingerprint:
                    self._carry = (source, int(row), int(seed))
        self._snapshots.clear()


class FixedBatchSampler(Sampler):
    """A fixed list of batches (validation); this rank takes every ``world_size``-th one."""

    def __init__(self, batches: list[list[tuple[int, int, int, int]]], rank: int = 0, world_size: int = 1):
        self.batches = batches[rank::world_size]

    def __iter__(self):
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


def pack_in_order(
    items: list[tuple[int, int, int]], cost: list[int], max_tokens: int, max_samples: int
) -> list[list[tuple[int, int, int, int]]]:
    """Greedy packing that keeps the given order; batch ids start at 1."""
    batches: list[list[tuple[int, int, int]]] = []
    current: list[tuple[int, int, int]] = []
    tokens = 0
    for item, c in zip(items, cost):
        if current and (tokens + c > max_tokens or len(current) >= max_samples):
            batches.append(current)
            current, tokens = [], 0
        current.append(item)
        tokens += c
    if current:
        batches.append(current)
    return [[(i, *item) for item in batch] for i, batch in enumerate(batches, start=1)]
