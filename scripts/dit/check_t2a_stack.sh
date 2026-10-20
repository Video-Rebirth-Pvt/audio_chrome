#!/usr/bin/env bash
# S2: self-check of packing, MM-RoPE, DiT (incl. parity with DiffSynth MiniMaxH3DiT), flow, text cache,
# checkpoints and the inference pipeline. CPU, under a minute; also compares flash vs SDPA if a GPU is visible.
#
# Usage (from anywhere):
#   CUDA_VISIBLE_DEVICES= bash scripts/dit/check_t2a_stack.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/dit/check_t2a_stack.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

"${PYTHON}" tests/test_t2a_stack.py "$@"
