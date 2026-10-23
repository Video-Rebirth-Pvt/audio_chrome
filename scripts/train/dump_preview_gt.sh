#!/usr/bin/env bash
# Trace preview samples back to the original dataset clips and write them once.
# Does not run during training. Re-run is a no-op unless you pass --force.
#
# Usage (from anywhere):
#   bash scripts/train/dump_preview_gt.sh outputs/train/t2a_10b_s20_ws/20261021_210505
#   bash scripts/train/dump_preview_gt.sh outputs/train/t2a_10b_s20_ws
#   bash scripts/train/dump_preview_gt.sh configs/train/t2a_10b_s20_ws.yaml
#   bash scripts/train/dump_preview_gt.sh outputs/train/t2a_10b_s20_ws --out outputs/eval/preview_gt_s20_ws
#   bash scripts/train/dump_preview_gt.sh outputs/train/t2a_10b_s20_ws --force
#
# Default output: <run_dir>/samples/gt/{preview_<label>.wav, preview_<label>.txt, sources.tsv, manifest.json}
# Compare: samples/gt/preview_rain.wav  vs  samples/step_0009000/preview_rain.wav
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

SOURCE="${1:?usage: dump_preview_gt.sh <run_dir|config.yaml> [--out dir] [--force]}"
shift
CUDA_VISIBLE_DEVICES= "${PYTHON}" -m chrome.train.cli dump-gt "${SOURCE}" "$@"
