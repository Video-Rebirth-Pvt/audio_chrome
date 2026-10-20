#!/usr/bin/env bash
# S2: encode every caption a run needs (train subset, holdout clips, preview set, " ") into the text cache,
# then print hidden-state statistics of the encoder output (decides text_norm before condition_proj).
# Training never loads Qwen when all its captions are cached. One GPU (~49 GB for Qwen3-VL-32B/50).
#
# Usage (from anywhere):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/text/cache_text.sh configs/train/t2a_smoke.yaml
#   CUDA_VISIBLE_DEVICES=0 bash scripts/text/cache_text.sh configs/train/t2a_smoke.yaml text.num_layers=32
#
# Output: outputs/text_cache/<encoder fingerprint hash>/
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

CONFIG="${1:-configs/train/t2a_smoke.yaml}"
shift || true
"${PYTHON}" -m chrome.train.cli text-cache "${CONFIG}" "$@"
