#!/usr/bin/env bash
# Line chart of asr_trend.tsv. Y axis is WER for word languages and CER for zh/yue/ja/ko/th.
# Clips whose max error stays below --min-max (default 0.10) are omitted; MEAN is always drawn.
#
#   bash scripts/eval/plot_asr_trend.sh outputs/train/t2a_10b_s20_ws/20261021_210505/asr_trend.tsv
#   bash scripts/eval/plot_asr_trend.sh outputs/train/t2a_10b_s20_ws/20261021_210505/asr_trend.tsv -o outputs/train/t2a_10b_s20_ws/20261021_210505/asr_trend.png
#   bash scripts/eval/plot_asr_trend.sh outputs/train/t2a_10b_s20_ws/20261021_210505/asr_trend.tsv --min-max 0.15
#
# Output: same path as the tsv with .png, unless -o is set.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

TSV="${1:?usage: plot_asr_trend.sh <asr_trend.tsv> [-o out.png] [--min-max 0.10]}"
shift
"${PYTHON}" scripts/eval/plot_asr_trend.py "${TSV}" "$@"
