#!/usr/bin/env bash
# S2/S3: turn a (sharded) checkpoint into one fp32 safetensors file. CPU only; inference also reads the sharded
# state/ directly, so this is for sharing weights or other tools. 10B: ~43 GB file, a few minutes.
#
# Usage (from anywhere):
#   bash scripts/train/export_t2a.sh outputs/train/t2a_smoke
#   bash scripts/train/export_t2a.sh outputs/train/t2a_10b --checkpoint 2000 --weights ema
#   bash scripts/train/export_t2a.sh outputs/train/t2a_10b --out /path/to/t2a_10b_ema.safetensors
# Family dir resolves to the newest timestamped experiment (or a legacy flat run).
#
# Output: <run_dir>/checkpoints/<step>/<weights>.safetensors (or --out)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

RUN_DIR="${1:?usage: export_t2a.sh <run_dir> [--checkpoint latest] [--weights ema|model] [--out file]}"
shift
CUDA_VISIBLE_DEVICES= "${PYTHON}" -m chrome.train.cli export "${RUN_DIR}" "$@"
