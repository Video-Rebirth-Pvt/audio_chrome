#!/usr/bin/env bash
# Single-GPU text → 32 kHz stereo wav. --gpu / --ckpt / --text plus the usual sampling knobs.
# Extra args still accept OmegaConf overrides (infer.steps=50). Family dirs resolve to the newest stamp.
#
# 15 s 10B, step 100000 EMA, GPU 0:
#   bash scripts/infer/infer_t2a.sh --gpu 0 \
#     --ckpt outputs/train/t2a_10b/checkpoints/step_0100000 \
#     --text "Rain on a tin roof with distant thunder." \
#     --seconds 10 --steps 50 --cfg 4.0 --seed 0 \
#     --out outputs/infer/t2a_10b_s15 --name rain
#
# Same run via the family dir (latest checkpoint):
#   bash scripts/infer/infer_t2a.sh --gpu 0 --ckpt outputs/train/t2a_10b \
#     --text "一段钢琴独奏，大厅混响。" --seconds 15
#
# 20 s run (newest stamp under t2a_10b_s20):
#   bash scripts/infer/infer_t2a.sh --gpu 0 --ckpt outputs/train/t2a_10b_s20 \
#     --text "Rain on a tin roof." --seconds 20 --steps 50 --cfg 4.0
#
# Several captions, raw weights, a .wav path:
#   bash scripts/infer/infer_t2a.sh --gpu 1 \
#     --ckpt outputs/train/t2a_10b/checkpoints/step_0100000 \
#     --weights model --text "Birds in a courtyard." --text "Heavy rain." \
#     --out outputs/infer/pair --seed 7
#   bash scripts/infer/infer_t2a.sh --gpu 0 --ckpt outputs/train/t2a_10b \
#     --text "Soft guitar." --out outputs/infer/guitar.wav
#
# Flags: --gpu --ckpt --text (repeatable) --prompt-file --seconds --steps --cfg/--cfg-scale
#        --seed --sampler --weights ema|model --shift --name --out
# Legacy:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/infer/infer_t2a.sh outputs/train/t2a_10b \
#     'infer.prompts=["Rain on a tin roof."]' infer.seconds=10 infer.checkpoint=100000
#
# Output: <run_dir>/infer/<name>.{wav,txt}  (or --out)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      export CUDA_VISIBLE_DEVICES="${2:?--gpu needs a device id}"
      shift 2
      ;;
    --gpu=*)
      export CUDA_VISIBLE_DEVICES="${1#*=}"
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "usage: infer_t2a.sh --gpu 0 --ckpt <run-or-step> --text \"...\" [--seconds N] [--steps N] [--cfg 4.0]" >&2
  exit 1
fi

"${PYTHON}" -m chrome.train.cli infer "${ARGS[@]}"
