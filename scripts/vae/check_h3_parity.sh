#!/usr/bin/env bash
# S0: check that chrome.codec reproduces the released H3 audio VAE (decode, encode, export, real clips).
#
# Usage (from anywhere):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/vae/check_h3_parity.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/vae/check_h3_parity.sh parity.clip_seconds=4
#   CUDA_VISIBLE_DEVICES=0 bash scripts/vae/check_h3_parity.sh parity.atol=1e-3 parity.output_dir=outputs/vae/parity_debug
#   bash scripts/vae/check_h3_parity.sh parity.device=cpu parity.clip_labels='[solo piano]'
#
# Output: outputs/vae/parity/report.json and <label>_{src,rec}.wav; exit code 1 if any check fails.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

"${PYTHON}" tests/test_h3_parity.py --config configs/vae/h3_frozen.yaml "$@"
