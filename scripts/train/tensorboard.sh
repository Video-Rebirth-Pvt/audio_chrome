#!/usr/bin/env bash
# Serve TensorBoard for one run or a family. Always syncs log.jsonl + samples/ into <run_dir>/tb
# first (idempotent), so a job that started before TensorBoard support still plots.
#
# Usage (from anywhere):
#   bash scripts/train/tensorboard.sh outputs/train/t2a_10b
#   bash scripts/train/tensorboard.sh outputs/train
#   PORT=6007 HOST=127.0.0.1 bash scripts/train/tensorboard.sh outputs/train/t2a_10b
#   bash scripts/train/tensorboard.sh outputs/train/t2a_10b --sync-only
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

SYNC_ONLY=0
LOGDIR="outputs/train"
for arg in "$@"; do
  if [[ "${arg}" == "--sync-only" ]]; then
    SYNC_ONLY=1
  else
    LOGDIR="${arg}"
  fi
done

"${PYTHON}" -m chrome.train.tensorboard "${LOGDIR}"

if [[ "${SYNC_ONLY}" -eq 1 ]]; then
  exit 0
fi

TB="$(dirname "${PYTHON}")/tensorboard"
if [[ ! -x "${TB}" ]]; then
  echo "tensorboard CLI not found next to ${PYTHON}" >&2
  exit 1
fi

PORT="${PORT:-6006}"
HOST="${HOST:-0.0.0.0}"
echo "[tb] ${TB} --logdir ${LOGDIR} --host ${HOST} --port ${PORT}"
exec "${TB}" --logdir "${LOGDIR}" --host "${HOST}" --port "${PORT}"
