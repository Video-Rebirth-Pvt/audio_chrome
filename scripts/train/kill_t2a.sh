#!/usr/bin/env bash
# Kill leftover chrome.train.cli ranks/workers that survive Ctrl-C / a dead torchrun.
# Those orphans keep GPU memory and make the next torchrun fail in NCCL broadcast.
#
# Usage (from anywhere):
#   bash scripts/train/kill_t2a.sh
#   bash scripts/train/kill_t2a.sh configs/train/t2a_10b.yaml
#   bash scripts/train/kill_t2a.sh --force
#   bash scripts/train/kill_t2a.sh configs/train/t2a_10b.yaml --force
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

FORCE=0
CONFIG=""
for arg in "$@"; do
  if [[ "${arg}" == "--force" ]]; then
    FORCE=1
  else
    CONFIG="${arg}"
  fi
done

PATTERN="chrome.train.cli train"
if [[ -n "${CONFIG}" ]]; then
  PATTERN="${PATTERN} ${CONFIG}"
fi

list_pids() {
  pgrep -f "${PATTERN}" | while read -r pid; do
    [[ -z "${pid}" ]] && continue
    [[ "${pid}" == "$$" ]] && continue
    [[ "${pid}" == "${PPID}" ]] && continue
    cmd=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)
    [[ "${cmd}" == *kill_t2a.sh* ]] && continue
    [[ "${cmd}" != *chrome.train.cli* ]] && continue
    printf '%s\n' "${pid}"
  done
}

pids=$(list_pids | sort -u | tr '\n' ' ')
if [[ -z "${pids// }" ]]; then
  echo "no leftover chrome.train.cli processes for: ${PATTERN}"
else
  echo "killing leftover ranks: ${pids}"
  if [[ "${FORCE}" -eq 0 ]]; then
    kill -TERM ${pids} 2>/dev/null || true
    sleep 3
    leftover=$(list_pids | sort -u | tr '\n' ' ')
    if [[ -n "${leftover// }" ]]; then
      echo "still alive after SIGTERM, sending SIGKILL: ${leftover}"
      kill -KILL ${leftover} 2>/dev/null || true
      sleep 2
    fi
  else
    kill -KILL ${pids} 2>/dev/null || true
    sleep 2
  fi
fi

left=$(list_pids | sort -u | tr '\n' ' ')
if [[ -n "${left// }" ]]; then
  echo "WARNING: still alive: ${left}"
  exit 1
fi

echo "all matching chrome.train.cli processes are gone"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
