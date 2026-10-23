#!/usr/bin/env bash
# Music clips that share a filename across the six cleaned manifests but disagree on resp_new.
# One JSON object per filename; copies[] keeps every path, resp and resp_new. Names only, not waveforms.
#
#   bash scripts/data/export_music_caption_conflicts.sh
#   bash scripts/data/export_music_caption_conflicts.sh -o outputs/data/music_caption_conflicts.jsonl
#
# Output: outputs/data/music_caption_conflicts.jsonl
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

"${PYTHON}" scripts/data/export_music_caption_conflicts.py "$@"
