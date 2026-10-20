#!/usr/bin/env bash
# S1 data: per-source rows / hours / drop reasons / length histogram, and what max_records (or an
# explicit weight) turn into (clip, audio-hour and token shares; epochs per N steps). Reads meta.json
# only, plus path hashes for the cross-source duplicate check.
#
# Usage (from anywhere; extra args are config overrides):
#   bash scripts/data/inspect_data.sh
#   bash scripts/data/inspect_data.sh inspect.world_size=64 inspect.grad_accum=2 inspect.steps=300000
#   bash scripts/data/inspect_data.sh data.loader.max_tokens=32768 inspect.duplicates=false
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

"${PYTHON}" -m chrome.data.cli inspect "$@"
