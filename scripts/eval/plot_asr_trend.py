"""Plot ASR error vs step from asr_trend.tsv. Drops clips whose max error stays below --min-max."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def load_tsv(path: Path) -> tuple[list[int], list[tuple[str, str, list[float]]]]:
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    steps = [int(col.rsplit("_", 1)[-1]) for col in header[2:]]
    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        name, language = parts[0], parts[1]
        values = [float(x) for x in parts[2:]]
        rows.append((name, language, values))
    return steps, rows


# Same split as chrome/eval/asr_preview.py: these languages score characters, the rest score words.
CHAR_LANGUAGES = frozenset({"Chinese", "Cantonese", "Japanese", "Korean", "Thai"})


def short_name(name: str) -> str:
    return name.removeprefix("preview_")


def metric_name(language: str) -> str:
    return "CER" if language in CHAR_LANGUAGES else "WER"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=None)
    parser.add_argument(
        "--min-max",
        type=float,
        default=0.10,
        help="keep a clip only if its max error is at least this (MEAN always kept)",
    )
    args = parser.parse_args()
    tsv = args.tsv
    out = args.out or tsv.with_suffix(".png")

    steps, rows = load_tsv(tsv)
    mean = next((r for r in rows if r[0].startswith("MEAN")), None)
    clips = [r for r in rows if not r[0].startswith("MEAN") and max(r[2]) >= args.min_max]
    if not clips and mean is None:
        raise SystemExit("no series left after filtering")

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=160)
    cmap = plt.get_cmap("tab10")
    markevery = 1 if len(steps) <= 16 else max(1, round(len(steps) / 12))
    for i, (name, language, values) in enumerate(clips):
        label = f"{short_name(name)} · {metric_name(language)}"
        ax.plot(
            steps, values, marker="o", markersize=3.5, markevery=markevery, linewidth=1.6,
            color=cmap(i % 10), label=label,
        )
    if mean is not None:
        ax.plot(
            steps, mean[2], color="0.15", linewidth=2.2, linestyle="--", marker="s", markersize=3.5,
            markevery=markevery, label="MEAN(err, clipped at 1)", zorder=5,
        )

    ax.set_title("Preview error vs step  ·  WER (words) / CER (characters)")
    ax.set_xlabel("step")
    ax.set_ylabel("WER or CER")
    if len(steps) <= 16:
        ticks = steps
    else:
        ticks = [s for s in steps if s % 10000 == 0] or steps[:: max(1, len(steps) // 10)]
    ax.set_xticks(ticks)
    pad = max(200, (steps[-1] - steps[0]) * 0.01)
    ax.set_xlim(steps[0] - pad, steps[-1] + pad)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"[plot] {len(clips)} clips + MEAN → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
