"""Export music clips that share a filename but disagree on resp_new.

One JSON object per filename. ``copies`` holds every manifest row (path, resp, resp_new).
This matches names only; it does not compare waveforms.

    python scripts/data/export_music_caption_conflicts.py
    python scripts/data/export_music_caption_conflicts.py -o outputs/data/music_caption_conflicts.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/mnt/data04/480634/public/0_DATA/1_AUDIO/llp/data/cleaned_jsonl_0922")
MANIFESTS = [
    DATA / "cleaned_music1_train1.jsonl",
    DATA / "cleaned_music2_train.jsonl",
    DATA / "cleaned_music3_train.jsonl",
    DATA / "cleaned_music4_train.jsonl",
    DATA / "cleaned_music5_train.jsonl",
    DATA / "cleaned_music6_train.jsonl",
]


def basename(path: str) -> str:
    return Path(path).name


def count_names(manifests: list[Path]) -> Counter:
    counts: Counter = Counter()
    for manifest in manifests:
        with manifest.open() as handle:
            for line in handle:
                row = json.loads(line)
                counts[basename(row["path"])] += 1
    return counts


def collect(manifests: list[Path], wanted: set[str]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for manifest in manifests:
        with manifest.open() as handle:
            for line in handle:
                row = json.loads(line)
                name = basename(row["path"])
                if name not in wanted:
                    continue
                groups[name].append(
                    {
                        "manifest": manifest.name,
                        "path": row["path"],
                        "duration": row.get("duration"),
                        "sample_rate": row.get("sample_rate"),
                        "uid": row.get("uid"),
                        "resp": row.get("resp"),
                        "resp_new": row.get("resp_new"),
                    }
                )
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=ROOT / "outputs/data/music_caption_conflicts.jsonl",
    )
    args = parser.parse_args()
    missing = [str(p) for p in MANIFESTS if not p.is_file()]
    if missing:
        raise SystemExit("missing manifests:\n" + "\n".join(missing))

    counts = count_names(MANIFESTS)
    repeated = {name for name, n in counts.items() if n >= 2}
    groups = collect(MANIFESTS, repeated)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows_out = 0
    copies = 0
    hours = 0.0
    by_manifest: Counter = Counter()
    by_n_labels: Counter = Counter()
    with args.out.open("w") as handle:
        for name in sorted(groups):
            items = groups[name]
            labels = {item["resp_new"] for item in items}
            if len(labels) < 2:
                continue
            for item in items:
                by_manifest[item["manifest"]] += 1
                if item["duration"]:
                    hours += float(item["duration"]) / 3600
            by_n_labels[len(labels)] += 1
            copies += len(items)
            rows_out += 1
            record = {
                "name": name,
                "n_copies": len(items),
                "n_resp_new": len(labels),
                "copies": items,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[conflicts] filenames with different resp_new: {rows_out}")
    print(f"[conflicts] manifest rows involved: {copies}")
    print(f"[conflicts] sum of those rows' duration: {hours:.1f} h")
    print("[conflicts] rows by manifest:")
    for name, n in sorted(by_manifest.items()):
        print(f"  {name}: {n}")
    print("[conflicts] filenames by number of distinct resp_new:")
    for n, c in sorted(by_n_labels.items()):
        print(f"  {n}: {c}")
    print(f"[conflicts] → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
