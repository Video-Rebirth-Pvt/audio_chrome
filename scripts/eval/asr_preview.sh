#!/usr/bin/env bash
# ASR intelligibility of [SPEECH] preview clips: Qwen3-ASR-1.7B transcript vs the caption text
# (CER for zh / yue / ja / ko / th, WER otherwise, plus language-ID accuracy). Runs in the qwen3-asr env,
# outside training; already scored steps are skipped. TensorBoard plots it under asr/*.
#
# Usage (from anywhere):
#   CUDA_VISIBLE_DEVICES=7 bash scripts/eval/asr_preview.sh outputs/train/t2a_10b_s20_ws/20261021_210505
#   CUDA_VISIBLE_DEVICES=7 bash scripts/eval/asr_preview.sh outputs/train/t2a_10b_s20_ws --steps 10000
#   CUDA_VISIBLE_DEVICES=7 bash scripts/eval/asr_preview.sh outputs/train/t2a_10b_s20_ws --force
#   bash scripts/eval/asr_preview.sh outputs/train/t2a_10b_s20_ws --rescore      # metrics only, reuse asr.jsonl, no GPU
#   CUDA_VISIBLE_DEVICES=7 nohup bash scripts/eval/asr_preview.sh outputs/train/t2a_10b_s20_ws --watch \
#       > outputs/train/t2a_10b_s20_ws/asr_watch.log 2>&1 &
#   ASR_MODEL=/path/to/Qwen3-ASR-0.6B bash scripts/eval/asr_preview.sh <run_dir> --batch 8
#
# Output: samples/step_X/{asr.jsonl, asr_summary.json}, <run_dir>/{asr_log.jsonl, asr_trend.tsv, asr_trend.png}
# The chart is redrawn on every run; with every step already scored the command only redraws (no model load).
# TensorBoard: bash scripts/train/tensorboard.sh <run_dir>   (a live trainer also picks up new asr rows)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

ASR_PYTHON="${ASR_PYTHON:-/data/jsfeng/miniconda3/envs/qwen3-asr/bin/python}"
ASR_MODEL="${ASR_MODEL:-/mnt/data04/480634/728342682@qq.com/1_CODE/ASR/QWEN3ASR/Qwen3-ASR/Qwen3-ASR-1.7B}"

RUN="${1:?usage: asr_preview.sh <run_dir|family_dir> [--steps N ...] [--force] [--watch] [--batch N]}"
shift
"${ASR_PYTHON}" -m chrome.eval.asr_preview "${RUN}" --model "${ASR_MODEL}" "$@"
