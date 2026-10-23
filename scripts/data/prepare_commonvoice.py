"""CommonVoice manifests with WorldSpeech-style dialect tags.

The raw manifests tag every Spanish clip ``[Spanish][Peninsular]`` (35% are Mexican), every English clip
``[English][English]``, and Hong Kong Cantonese ``[Cantonese][Hong Kong]`` where WorldSpeech and CV ``yue``
write ``[Cantonese][Cantonese]``. Here the dialect tag of es / en comes from the speaker's ``accents`` field
(the WorldSpeech name where one exists; anything unknown, mixed or empty gets the generic tag), zh-HK
becomes Cantonese, and every other locale is copied as it is. Only ``resp`` and ``dialect`` change;
``dialect_cv`` keeps the original tag.

    python scripts/data/prepare_commonvoice.py
    python scripts/data/prepare_commonvoice.py --src <raw jsonl dir> --out <output dir>

Output: <out>/commonvoice_<locale>_26.0.resp.jsonl (same names as the raw files) and <out>/report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import orjson

DATA = Path("/mnt/data04/480634/public/0_DATA/1_AUDIO/ASR")
PATTERN = "commonvoice_*_26.0.resp.jsonl"

# accents segment prefix -> dialect tag; first match wins, segments that match nothing are ignored.
SPANISH = [
    ("España: Islas Canarias", None),
    ("España", "Peninsular"),
    ("Islas Baleares", "Peninsular"),
    ("México", "Mexican"),
    ("Ciudad de México", "Mexican"),
    ("Rioplatense", "Argentinian"),
    ("Chileno", "Chilean"),
]
ENGLISH = [
    ("United States English", "American"),
    ("England English", "British"),
    ("Scottish English", "Scottish"),
    ("Irish English", "Irish"),
    ("Northern Irish", "Northern Irish"),
    ("Welsh English", "Welsh"),
    ("Canadian English", "Canadian"),
    ("Australian English", "Australian"),
    ("New Zealand English", "New Zealand"),
    ("India and South Asia", "Indian"),
    ("Southern African", "South African"),
    ("Filipino", "Filipino"),
    ("Hong Kong English", "Hong Kong"),
    ("Singaporean English", "Singaporean"),
    ("Malaysian English", "Malaysian"),
    ("West Indies and Bermuda", "Caribbean"),
]
ACCENT_RULES = {"es": (SPANISH, "Spanish"), "en": (ENGLISH, "English")}
FIXED_DIALECT = {"zh-HK": "Cantonese"}

TAGS_RE = re.compile(r"^(\[SPEECH\]: <d>\[[^\]]*\])\[[^\]]*\]")


def accent_dialect(accents: str, rules: list[tuple[str, str | None]], generic: str) -> str:
    """One dialect when every recognised segment agrees on it, else the generic tag."""
    found = set()
    for segment in (s.strip() for s in accents.split("|")):
        for prefix, dialect in rules:
            if segment.startswith(prefix):
                found.add(dialect)
                break
    found.discard(None)
    return found.pop() if len(found) == 1 else generic


def retag(row: dict, dialect: str) -> dict:
    resp, count = TAGS_RE.subn(lambda m: f"{m.group(1)}[{dialect}]", row["resp"], count=1)
    if count != 1:
        raise ValueError(f"unexpected resp {row['resp'][:80]!r} in {row['path']}")
    row["dialect_cv"] = row.get("dialect", "")
    row["dialect"] = dialect
    row["resp"] = resp
    return row


def rewrite(src: Path, dest: Path, locale: str) -> Counter:
    counts: Counter = Counter()
    rules = ACCENT_RULES.get(locale)
    partial = dest.with_suffix(dest.suffix + ".partial")
    with src.open("rb") as reader, partial.open("wb") as writer:
        for line in reader:
            if not line.strip():
                continue
            row = orjson.loads(line)
            if rules is not None:
                dialect = accent_dialect(row.get("accents") or "", *rules)
            else:
                dialect = FIXED_DIALECT[locale]
            counts[f"{row.get('dialect', '')} -> {dialect}"] += 1
            writer.write(orjson.dumps(retag(row, dialect)) + b"\n")
    os.replace(partial, dest)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=DATA / "jsonl")
    parser.add_argument("--out", type=Path, default=DATA / "jsonl_chrome")
    args = parser.parse_args()
    if args.src.resolve() == args.out.resolve():
        raise SystemExit("--out must differ from --src")
    files = sorted(args.src.glob(PATTERN))
    if not files:
        raise SystemExit(f"no {PATTERN} in {args.src}")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for src in files:
        locale = src.name.split("_")[1]
        dest = args.out / src.name
        if locale in ACCENT_RULES or locale in FIXED_DIALECT:
            counts = rewrite(src, dest, locale)
            report[locale] = dict(counts.most_common())
            print(f"[cv] {locale}: retagged {sum(counts.values())} rows  " + ", ".join(f"{k}: {v}" for k, v in counts.most_common()))
        else:
            shutil.copyfile(src, dest)
            report[locale] = "copied"
            print(f"[cv] {locale}: copied")
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"[cv] wrote {len(files)} manifests to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
