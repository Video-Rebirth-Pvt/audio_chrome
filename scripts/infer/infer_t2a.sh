#!/usr/bin/env bash
# S2: generate 32 kHz stereo wavs from a training run (EMA weights of the latest checkpoint by default).
# Extra args override the run's config (infer.* section of configs/train/t2a_base.yaml).
#
# Usage (from anywhere):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/infer/infer_t2a.sh outputs/train/t2a_smoke 'infer.prompts=["Heavy rain on a tin roof with distant thunder."]'
#   CUDA_VISIBLE_DEVICES=0 bash scripts/infer/infer_t2a.sh outputs/train/t2a_smoke infer.prompt_file=configs/eval/preview_set.json infer.steps=30
#   CUDA_VISIBLE_DEVICES=0 bash scripts/infer/infer_t2a.sh outputs/train/t2a_smoke infer.checkpoint=400 infer.weights=model infer.cfg_scale=1
#   CUDA_VISIBLE_DEVICES=0 bash scripts/infer/infer_t2a.sh outputs/train/t2a_10b 'infer.prompts=["Rain on a tin roof."]'
# Family dir resolves to the newest timestamped experiment (or a legacy flat run).
#
# Output: <run_dir>/infer/<name>.{wav,txt} (or infer.output_dir)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

RUN_DIR="${1:?usage: infer_t2a.sh <run_dir> [key=value ...]}"
shift
"${PYTHON}" -m chrome.train.cli infer "${RUN_DIR}" "$@"
