#!/usr/bin/env bash
# S2/S3: train the T2A DiT with torchrun on every visible GPU (or NPROC of them). Extra args are config overrides.
# Re-running the same command resumes from <output_dir>/checkpoints/latest (train.resume=auto).
# Single node by default; set NNODES / NODE_RANK / MASTER_ADDR (/ MASTER_PORT) on every node for multi-node.
#
# Usage (from anywhere):
#   bash scripts/train/train_t2a.sh configs/train/t2a_smoke.yaml                                    # 8 GPUs, DDP, 500 clips
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train/train_t2a.sh configs/train/t2a_smoke.yaml train.run_name=t2a_smoke_1gpu
#   bash scripts/train/train_t2a.sh configs/train/t2a_smoke.yaml train.parallel.kind=fsdp train.run_name=t2a_smoke_fsdp
#   bash scripts/train/train_t2a.sh configs/train/t2a_smoke.yaml train.max_steps=50 train.resume=null
#   bash scripts/train/train_t2a.sh configs/train/t2a_smoke.yaml dit.adaln=shared train.run_name=smoke_shared
#   bash scripts/train/train_t2a.sh configs/train/t2a_1b.yaml
#   bash scripts/train/train_t2a.sh configs/train/t2a_10b.yaml data.loader.subset=100 train.max_steps=20 train.run_name=t2a_10b_dry
#   NNODES=4 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/train/train_t2a.sh configs/train/t2a_10b.yaml train.parallel.shard_size=8
#   bash scripts/train/train_t2a.sh configs/train/t2a_10b.yaml train.tensorboard=false
#
# Stop leftover ranks (Ctrl-C / closing the terminal often leaves orphans on GPU):
#   bash scripts/train/kill_t2a.sh
#   bash scripts/train/kill_t2a.sh configs/train/t2a_10b.yaml
#   bash scripts/train/kill_t2a.sh --force
#
# After a mid-run c10d / RendezvousConnectionError (rank 0 hung writing the 173 GB snapshot):
#   bash scripts/train/kill_t2a.sh
#   bash scripts/train/train_t2a.sh configs/train/t2a_10b_s20.yaml
# resume=auto continues checkpoints/latest (last complete step). Incomplete step_* dirs are dropped.
#
# TensorBoard (another terminal; also imports an older run's log.jsonl / samples/):
#   bash scripts/train/tensorboard.sh outputs/train/t2a_10b
#   bash scripts/train/tensorboard.sh outputs/train
#   PORT=6007 bash scripts/train/tensorboard.sh outputs/train/t2a_10b
#   bash scripts/train/tensorboard.sh outputs/train/t2a_10b --sync-only
#
# Output: outputs/train/<run_name>/{config.yaml, log.jsonl, tb/, checkpoints/, samples/}
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

CONFIG="${1:-configs/train/t2a_smoke.yaml}"
shift || true
if [[ -z "${NPROC:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NPROC=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
  else
    NPROC=$(nvidia-smi -L 2>/dev/null | wc -l)
  fi
fi
NPROC=$(( NPROC > 0 ? NPROC : 1 ))
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Rank-0 hosts the c10d TCPStore. A 10B JuiceFS write can stall that process; give the NCCL
# watchdog long enough that a slow-but-alive writer is not treated as a dead rank.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"

NNODES="${NNODES:-1}"
if [[ "${NNODES}" -gt 1 ]]; then
  LAUNCH=(--nnodes "${NNODES}" --node_rank "${NODE_RANK:?set NODE_RANK on every node}"
          --master_addr "${MASTER_ADDR:?set MASTER_ADDR to node 0}" --master_port "${MASTER_PORT:-29500}")
else
  LAUNCH=(--standalone)
fi

"${PYTHON}" -m torch.distributed.run "${LAUNCH[@]}" --nproc_per_node "${NPROC}" \
  -m chrome.train.cli train "${CONFIG}" "$@"
