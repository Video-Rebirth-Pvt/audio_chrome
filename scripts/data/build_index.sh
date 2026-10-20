#!/usr/bin/env bash
# S1 data: scan every source's manifests in parallel into outputs/data/index/<source>/{index.arrow,meta.json}.
# Sources whose config and manifest files are unchanged are skipped.
#
# Usage (from anywhere; extra args are config overrides):
#   bash scripts/data/build_index.sh
#   bash scripts/data/build_index.sh data.only=[music,freesound]
#   bash scripts/data/build_index.sh data.only=[zhizhou] data.build.force=true data.build.workers=96
#   bash scripts/data/build_index.sh --config configs/data/t2a_base.yaml data.build.chunk_mb=128
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

"${PYTHON}" -m chrome.data.cli build "$@"
