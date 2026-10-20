#!/usr/bin/env bash
# S1 data: self-check of index build, length policy, sampler (determinism, resume, rank shards, budget),
# dataset decoding and VAE batch grouping, on synthetic audio in a temp dir. CPU only, ~1 minute.
#
# Usage (from anywhere):
#   bash scripts/data/check_data_pipeline.sh
#   bash scripts/data/check_data_pipeline.sh --keep      # keep the temp dir for inspection
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

"${PYTHON}" tests/test_data_pipeline.py "$@"
