#!/usr/bin/env bash
# S1 data: copy the audio bytes embedded in HF parquet shards into memory-mappable Arrow files, so a training
# row reads only its own clip. Bytes are copied as is (no re-encode). Only sources with `blobs` are touched;
# shards whose Arrow file is newer than the parquet are skipped. Run before build_index.sh.
#
# Usage (from anywhere; extra args are config overrides):
#   bash scripts/data/repack_audio.sh --config configs/data/t2a_worldspeech.yaml
#   bash scripts/data/repack_audio.sh --config configs/data/t2a_worldspeech.yaml data.only=[ws_yue_hk]
#   bash scripts/data/repack_audio.sh --config configs/data/t2a_worldspeech.yaml data.repack.workers=64
#   bash scripts/data/repack_audio.sh --config configs/data/t2a_worldspeech.yaml data.repack.force=true
#
# Output: ${paths.audio_arrow_root}/<dataset>/<locale>/<shard>.arrow
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

"${PYTHON}" -m chrome.data.cli repack "$@"
