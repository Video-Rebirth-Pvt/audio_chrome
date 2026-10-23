#!/usr/bin/env bash
# S1 data: copy the audio bytes embedded in HF parquet shards into memory-mappable Arrow files, so a training
# row reads only its own clip. Bytes are copied as is (no re-encode). Only sources whose manifests are such
# parquet and that set `blobs` are touched; shards whose Arrow file is current are skipped. Run before
# build_index.sh. WorldSpeech instead migrates its shards in place: scripts/data/migrate_audio.sh.
#
# Usage (from anywhere; extra args are config overrides):
#   bash scripts/data/repack_audio.sh --config configs/data/<config>.yaml
#   bash scripts/data/repack_audio.sh --config configs/data/<config>.yaml data.only=[<source>] data.repack.workers=64
#   bash scripts/data/repack_audio.sh --config configs/data/<config>.yaml data.repack.force=true
#
# Output: <blobs>/<parent dir of the shard>/<shard>.arrow
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

"${PYTHON}" -m chrome.data.cli repack "$@"
