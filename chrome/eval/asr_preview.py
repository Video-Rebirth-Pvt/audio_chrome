"""ASR intelligibility of [SPEECH] previews: Qwen3-ASR transcript vs the text inside the caption.

Runs in the qwen3-asr env (needs ``qwen_asr`` + ``zhconv``, ``matplotlib`` for the plot, no torch-side
chrome imports). For every finished ``samples/step_*`` it writes:

    samples/step_X/asr.jsonl          one row per [SPEECH] clip: ref / hyp / S D I / error rate / language
    samples/step_X/asr_summary.json   step mean error, micro CER / WER, LID accuracy, per language
    <run_dir>/asr_log.jsonl           event=asr row per step; chrome.train.tensorboard imports it (asr/*)
    <run_dir>/asr_trend.tsv           clip x step error rate
    <run_dir>/asr_trend.png           line chart of the tsv: mean per language, then every clip

Error unit follows the language: characters for zh / yue / ja / ko / th, words otherwise (CER is kept
for every clip as well). Auto language detection, so a clip in the wrong language shows up as lid_ok=false.
Cantonese folds spoken characters to written Chinese on both sides; err_strict keeps the unfolded score.

    bash scripts/eval/asr_preview.sh outputs/train/t2a_10b_s20_ws/20261021_210505
    bash scripts/eval/asr_preview.sh outputs/train/t2a_10b_s20_ws --watch
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import warnings
from pathlib import Path
from typing import Any

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import zhconv

DEFAULT_MODEL = "/mnt/data04/480634/728342682@qq.com/1_CODE/ASR/QWEN3ASR/Qwen3-ASR/Qwen3-ASR-1.7B"
ASR_LOG = "asr_log.jsonl"
ASR_TREND = "asr_trend.tsv"
ASR_PLOT = "asr_trend.png"
CLIP_ROWS = "asr.jsonl"
SUMMARY = "asr_summary.json"

SPEECH_RE = re.compile(r"^\s*\[SPEECH\]\s*:\s*<d>(.*?)(?:</d>)?\s*$", re.DOTALL)
TAGS_RE = re.compile(r"^((?:\s*\[[^\]]*\])+)\s*(.*)$", re.DOTALL)

# Qwen3-ASR language names; the first caption tag is expected to be one of them.
QWEN_LANGUAGES = frozenset({
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish", "Portuguese", "Indonesian",
    "Italian", "Korean", "Russian", "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch",
    "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek", "Romanian",
    "Hungarian", "Macedonian",
})
LANGUAGE_ALIASES = {"Mandarin": "Chinese", "Yue": "Cantonese"}
CHAR_LANGUAGES = frozenset({"Chinese", "Cantonese", "Japanese", "Korean", "Thai"})

ARABIC_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")
ARABIC_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي", "ؤ": "و", "ئ": "ي"})

# Cantonese captions are written Chinese, Qwen3-ASR writes spoken Cantonese (喺 / 嘅 / 系 / 啲 …). Both sides
# fold to the written form (keys are post-zhconv simplified; zhconv turns 嗰 into 𠮶), and sentence
# particles are dropped. Applied to ref and hyp alike, so a word shared by both never turns into an error.
CANTONESE_WRITTEN = {
    "乜嘢": "什么", "咩嘢": "什么", "点解": "为什么", "点样": "怎样", "而家": "现在", "边个": "谁", "边度": "哪里",
    "呢度": "这里", "呢个": "这个", "呢啲": "这些", "呢样": "这样", "同埋": "和", "系咪": "是不是",
    "嗰度": "那里", "嗰个": "那个", "嗰啲": "那些", "𠮶度": "那里", "𠮶个": "那个", "𠮶啲": "那些",
    "冇": "没有", "嘅": "的", "喺": "在", "系": "是", "佢": "他", "哋": "们", "咗": "了", "啲": "些",
    "嗰": "那", "𠮶": "那", "唔": "不", "睇": "看", "嚟": "来", "揾": "找", "畀": "给", "俾": "给",
    "嘢": "东西", "仲": "还", "啱": "对", "乜": "什么",
}
CANTONESE_WORD_RE = re.compile("|".join(sorted(map(re.escape, CANTONESE_WRITTEN), key=len, reverse=True)))
CANTONESE_PARTICLES = re.compile("[啊呀啦喇嘞啰囉咯喎㖞㗎啩噃吖啫嘛哦噢喔]")


# ------------------------------------------------------------ caption / text


def parse_speech_caption(caption: str) -> tuple[list[str], str] | None:
    """``[SPEECH]: <d>[Chinese][Taiwan][Male] 文本</d>`` → (["Chinese", "Taiwan", "Male"], "文本"); else None."""
    match = SPEECH_RE.match(caption.strip())
    if not match:
        return None
    body = match.group(1).strip()
    tagged = TAGS_RE.match(body)
    if not tagged:
        return [], body
    tags = re.findall(r"\[([^\]]*)\]", tagged.group(1))
    return [t.strip() for t in tags], tagged.group(2).strip()


def expected_language(tags: list[str]) -> str | None:
    if not tags:
        return None
    name = LANGUAGE_ALIASES.get(tags[0], tags[0])
    return name if name in QWEN_LANGUAGES else None


def normalize_text(text: str) -> str:
    """Language-independent, so ref and hyp go through the same folding (simplified Han, Arabic letter forms)."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = zhconv.convert(text, "zh-hans")
    text = ARABIC_DIACRITICS.sub("", text).translate(ARABIC_FOLD)
    text = text.replace("'", "").replace("’", "")
    out: list[str] = []
    for char in text:
        out.append(char if unicodedata.category(char)[0] in "LNM" else " ")
    return " ".join("".join(out).split())


def fold_cantonese(normalized: str) -> str:
    text = CANTONESE_WORD_RE.sub(lambda m: CANTONESE_WRITTEN[m.group(0)], normalized)
    return " ".join(CANTONESE_PARTICLES.sub("", text).split())


def tokens(normalized: str, unit: str) -> list[str]:
    return normalized.split() if unit == "word" else list(normalized.replace(" ", ""))


def edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) of a minimum-cost Levenshtein alignment."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        row, prev, r = d[i], d[i - 1], ref[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (r != hyp[j - 1]))
    sub = dele = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            sub += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return sub, dele, ins


def rate(errors: int, length: int) -> float:
    return errors / length if length else float(errors > 0)


def score_clip(name: str, tags: list[str], ref: str, asr_language: str, hyp: str) -> dict[str, Any]:
    """err scores ref_norm vs hyp_norm (Cantonese: folded to written form); err_strict skips the folding."""
    language = expected_language(tags)
    unit = "char" if language in CHAR_LANGUAGES or language is None else "word"
    ref_norm, hyp_norm = normalize_text(ref), normalize_text(hyp)
    strict_ref, strict_hyp = tokens(ref_norm, unit), tokens(hyp_norm, unit)
    err_strict = round(rate(sum(edit_ops(strict_ref, strict_hyp)), len(strict_ref)), 4)
    if language == "Cantonese":
        ref_norm, hyp_norm = fold_cantonese(ref_norm), fold_cantonese(hyp_norm)
    ref_chars, hyp_chars = tokens(ref_norm, "char"), tokens(hyp_norm, "char")
    char_edits = sum(edit_ops(ref_chars, hyp_chars))
    if unit == "char":
        ref_tok = ref_chars
        sub, dele, ins = edit_ops(ref_chars, hyp_chars)
    else:
        ref_tok = tokens(ref_norm, "word")
        sub, dele, ins = edit_ops(ref_tok, tokens(hyp_norm, "word"))
    detected = [x.strip() for x in str(asr_language or "").split(",") if x.strip()]
    err = round(rate(sub + dele + ins, len(ref_tok)), 4)
    return {
        "name": name,
        "tags": tags,
        "language": language,
        "asr_language": asr_language,
        "lid_ok": (language in detected) if language else None,
        "unit": unit,
        "ref": ref,
        "hyp": hyp,
        "ref_norm": ref_norm,
        "hyp_norm": hyp_norm,
        "ref_len": len(ref_tok),
        "sub": sub,
        "del": dele,
        "ins": ins,
        "err": err,
        "err_strict": err_strict,
        "cer": round(rate(char_edits, len(ref_chars)), 4),
        "wer": err if unit == "word" else None,
        "char_len": len(ref_chars),
        "char_edits": char_edits,
    }


def summarize(step: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """err: mean of per-clip error (each clipped at 1). cer / wer: micro (total edits / total ref length)."""
    if not rows:
        return {"step": step, "clips": 0}

    def mean_err(group: list[dict], key: str = "err") -> float:
        return round(sum(min(1.0, r.get(key, r["err"])) for r in group) / len(group), 4)

    def lid(group: list[dict]) -> float | None:
        known = [r["lid_ok"] for r in group if r["lid_ok"] is not None]
        return round(sum(known) / len(known), 4) if known else None

    char_edits = char_len = word_edits = word_len = 0
    for r in rows:
        char_edits += r["char_edits"]
        char_len += r["char_len"]
        if r["unit"] == "word":
            word_edits += r["sub"] + r["del"] + r["ins"]
            word_len += r["ref_len"]
    by_language: dict[str, dict[str, Any]] = {}
    for language in sorted({r["language"] or "unknown" for r in rows}):
        group = [r for r in rows if (r["language"] or "unknown") == language]
        by_language[language] = {
            "clips": len(group), "err": mean_err(group), "err_strict": mean_err(group, "err_strict"), "lid_acc": lid(group),
        }
    return {
        "step": step,
        "clips": len(rows),
        "err": mean_err(rows),
        "cer": round(rate(char_edits, char_len), 4),
        "wer": round(rate(word_edits, word_len), 4) if word_len else None,
        "lid_acc": lid(rows),
        "by_language": by_language,
    }


# ------------------------------------------------------------ run layout


def resolve_run(path: Path) -> Path:
    """A run dir (has samples/) or a family dir (newest stamped child with samples/)."""
    path = Path(path)
    if (path / "samples").is_dir():
        return path
    runs = sorted(child for child in path.iterdir() if (child / "samples").is_dir()) if path.is_dir() else []
    if not runs:
        raise FileNotFoundError(f"no samples/ under {path}")
    return runs[-1]


def step_of(folder: Path) -> int | None:
    try:
        return int(folder.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def step_dirs(run: Path) -> list[Path]:
    found = [d for d in (run / "samples").glob("step_*") if d.is_dir() and step_of(d) is not None]
    return sorted(found, key=step_of)


def settled(folder: Path, seconds: float) -> bool:
    """Ranks write their share then barrier; no new wav for ``seconds`` means the preview is complete."""
    wavs = list(folder.glob("*.wav"))
    if not wavs:
        return False
    newest = max(max(p.stat().st_mtime for p in wavs), folder.stat().st_mtime)
    return time.time() - newest >= seconds


def speech_items(folder: Path) -> list[tuple[str, Path, list[str], str]]:
    items = []
    for txt in sorted(folder.glob("*.txt")):
        wav = txt.with_suffix(".wav")
        parsed = parse_speech_caption(txt.read_text(encoding="utf-8"))
        if parsed is None or not wav.is_file() or not parsed[1]:
            continue
        items.append((txt.stem, wav, parsed[0], parsed[1]))
    return items


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rebuild_reports(run: Path) -> None:
    """asr_log.jsonl (TensorBoard) and asr_trend.tsv from every evaluated step, sorted by step."""
    records, trend, languages = [], {}, {}
    steps = []
    for folder in step_dirs(run):
        summary_path, rows_path = folder / SUMMARY, folder / CLIP_ROWS
        if not summary_path.is_file() or not rows_path.is_file():
            continue
        step = step_of(folder)
        summary = json.loads(summary_path.read_text())
        rows = read_jsonl(rows_path)
        steps.append(step)
        record: dict[str, Any] = {"event": "asr", "step": step}
        for key in ("err", "cer", "wer", "lid_acc", "clips"):
            if summary.get(key) is not None:
                record[key] = summary[key]
        record["err_lang"] = {k: v["err"] for k, v in summary.get("by_language", {}).items()}
        record["clip"] = {r["name"]: r["err"] for r in rows}
        records.append(record)
        for r in rows:
            trend.setdefault(r["name"], {})[step] = r["err"]
            languages[r["name"]] = r["language"] or "unknown"
    write_jsonl(run / ASR_LOG, records)
    lines = ["\t".join(["name", "language", *[f"step_{s:07d}" for s in steps]])]
    for name in sorted(trend):
        lines.append("\t".join([name, languages[name], *[f"{trend[name][s]:.3f}" if s in trend[name] else "" for s in steps]]))
    lines.append("\t".join(["MEAN(err)", "-", *[f"{r['err']:.3f}" if "err" in r else "" for r in records]]))
    (run / ASR_TREND).write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_trend(run)


def plot_trend(run: Path) -> Path | None:
    """asr_trend.png from asr_trend.tsv: mean err per language (top), every clip (bottom), against step."""
    path = run / ASR_TREND
    if not path.is_file():
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[asr] matplotlib not installed; skipping asr_trend.png", flush=True)
        return None
    header, *body = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    steps = [int(cell.split("_", 1)[1]) for cell in header[2:]]
    if not steps:
        return None
    clips, mean = [], None
    for cells in body:
        values = np.array([float(c) if c else np.nan for c in (cells[2:] + [""] * len(steps))[: len(steps)]])
        if cells[0].startswith("MEAN"):
            mean = values
        else:
            clips.append((cells[0], cells[1], values))
    groups: dict[str, list[np.ndarray]] = {}
    for _, language, values in clips:
        groups.setdefault(language, []).append(np.minimum(values, 1.0))

    def unit_of(language: str) -> str:
        return "CER" if language in CHAR_LANGUAGES or language == "unknown" else "WER"

    marker = {"marker": "o", "markersize": 3} if len(steps) <= 30 else {}
    title = f"{run.parent.name}/{run.name}" if re.fullmatch(r"\d{8}_\d{6}", run.name) else run.name
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for language in sorted(groups, key=lambda k: (-len(groups[k]), k)):
            stack = np.vstack(groups[language])
            top.plot(steps, np.nanmean(stack, axis=0), label=f"{language} ({len(stack)}) · {unit_of(language)}", lw=1.6, **marker)
    colors = plt.get_cmap("tab10").colors
    styles = ("-", "--", "-.", ":")
    for i, (name, language, values) in enumerate(clips):
        bottom.plot(
            steps, values, label=f"{name.removeprefix('preview_')} · {unit_of(language)}",
            color=colors[i % len(colors)], ls=styles[(i // len(colors)) % len(styles)], lw=1.4, **marker,
        )
    for ax in (top, bottom):
        if mean is not None:
            ax.plot(steps, mean, "k--", lw=2.4, label="MEAN(err, clipped at 1)", **({"marker": "s", "markersize": 4} if marker else {}))
        ax.set_ylim(bottom=0)
        ax.set_ylabel("err  (WER words / CER characters)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False, ncol=1 if len(ax.lines) <= 28 else 2)
    top.set_title(f"{title} · preview ASR error vs step — mean per language (each clip clipped at 1)")
    bottom.set_title("per clip (CER for zh / yue / ja / ko / th, WER otherwise)")
    bottom.set_xlabel("step")
    fig.tight_layout()
    out = run / ASR_PLOT
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------ ASR


class Transcriber:
    def __init__(self, model: str, device: str, batch: int, max_new_tokens: int):
        import torch
        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.from_pretrained(
            model, dtype=torch.bfloat16, device_map=device,
            max_inference_batch_size=batch, max_new_tokens=max_new_tokens,
        )

    def __call__(self, wavs: list[Path]) -> list[tuple[str, str]]:
        results = self.model.transcribe(audio=[str(p) for p in wavs], language=None)
        return [(str(r.language or ""), str(r.text or "")) for r in results]


def evaluate_step(folder: Path, transcribe: Transcriber) -> dict[str, Any]:
    step = step_of(folder)
    items = speech_items(folder)
    rows = []
    if items:
        outputs = transcribe([wav for _, wav, _, _ in items])
        rows = [
            score_clip(name, tags, ref, language, hyp)
            for (name, _, tags, ref), (language, hyp) in zip(items, outputs)
        ]
    return write_step(folder, rows)


def rescore_step(folder: Path) -> dict[str, Any]:
    """Recompute metrics from the stored ref / hyp (after a metric change); no ASR model needed."""
    rows = [score_clip(r["name"], r["tags"], r["ref"], r["asr_language"], r["hyp"]) for r in read_jsonl(folder / CLIP_ROWS)]
    return write_step(folder, rows)


def write_step(folder: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    write_jsonl(folder / CLIP_ROWS, rows)
    summary = summarize(step_of(folder), rows)
    (folder / SUMMARY).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    head = f"[asr] step {summary['step']}: clips={summary['clips']}"
    if not summary["clips"]:
        return head
    wer = "-" if summary["wer"] is None else f"{summary['wer']:.3f}"
    lid = "-" if summary["lid_acc"] is None else f"{summary['lid_acc']:.2f}"
    langs = " ".join(f"{k}={v['err']:.3f}" for k, v in summary["by_language"].items())
    return f"{head} err={summary['err']:.3f} cer={summary['cer']:.3f} wer={wer} lid={lid} | {langs}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="run dir (<family>/<stamp>) or family dir (newest stamp)")
    parser.add_argument("--steps", type=int, nargs="*", default=None, help="only these steps (default: all)")
    parser.add_argument("--force", action="store_true", help="re-score steps that already have asr_summary.json")
    parser.add_argument("--rescore", action="store_true", help="recompute metrics from existing asr.jsonl, no ASR")
    parser.add_argument("--watch", action="store_true", help="keep polling for new samples/step_* dirs")
    parser.add_argument("--interval", type=float, default=120.0, help="--watch poll period (s)")
    parser.add_argument("--settle", type=float, default=120.0, help="a step dir counts as complete after this many idle seconds")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args(argv)

    run = resolve_run(args.run)
    print(f"[asr] run {run}", flush=True)
    wanted = set(args.steps) if args.steps else None
    if args.rescore:
        done = [d for d in step_dirs(run) if (d / CLIP_ROWS).is_file() and (wanted is None or step_of(d) in wanted)]
        for folder in done:
            print(format_summary(rescore_step(folder)), flush=True)
        if done:
            rebuild_reports(run)
        print(f"[asr] rescored {len(done)} steps; wrote {ASR_LOG}, {ASR_TREND}, {ASR_PLOT} under {run}", flush=True)
        return 0
    forced: set[int] = set()
    transcribe: Transcriber | None = None
    first = True
    while True:
        pending = []
        for folder in step_dirs(run):
            step = step_of(folder)
            if wanted is not None and step not in wanted:
                continue
            redo = args.force and step not in forced
            if (folder / SUMMARY).is_file() and not redo:
                continue
            if not settled(folder, args.settle):
                continue
            pending.append(folder)
        for folder in pending:
            if transcribe is None:
                print(f"[asr] loading {args.model} on {args.device}", flush=True)
                transcribe = Transcriber(args.model, args.device, args.batch, args.max_new_tokens)
            started = time.perf_counter()
            summary = evaluate_step(folder, transcribe)
            forced.add(summary["step"])
            print(f"{format_summary(summary)} ({time.perf_counter() - started:.0f}s)", flush=True)
        if not pending and not args.watch:
            print("[asr] nothing new to score", flush=True)
        if pending or (first and any((d / SUMMARY).is_file() for d in step_dirs(run))):
            rebuild_reports(run)
            print(f"[asr] wrote {ASR_LOG}, {ASR_TREND}, {ASR_PLOT} under {run}", flush=True)
        first = False
        if not args.watch:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
