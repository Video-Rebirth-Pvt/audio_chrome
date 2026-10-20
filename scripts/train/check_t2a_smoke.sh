#!/usr/bin/env bash
# S2: end-to-end check on the 500-clip smoke subset through the real launchers: text cache + stats, DDP = FSDP
# from the same init, FSDP resume replays the uninterrupted run, HSDP, text_norm=stats, previews under FSDP,
# inference from sharded vs exported weights. 8 idle GPUs, ~10 minutes.
#
# Usage (from anywhere):
#   bash scripts/train/check_t2a_smoke.sh
#   bash scripts/train/check_t2a_smoke.sh --keep                       # keep outputs/check/t2a_e2e/
#   bash scripts/train/check_t2a_smoke.sh --gpus 0,1,2,3,4,5,6,7
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

"${PYTHON}" tests/test_t2a_e2e.py "$@"
