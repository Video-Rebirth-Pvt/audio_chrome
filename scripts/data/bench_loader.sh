#!/usr/bin/env bash
# S1 data: run the training loader alone and report batches/s, clips/s, packing fill, source shares;
# optionally encode every batch with the frozen H3 VAE. Writes a few clips + captions for listening.
#
# Usage (from anywhere; extra args are config overrides):
#   bash scripts/data/bench_loader.sh
#   bash scripts/data/bench_loader.sh bench.batches=500 data.loader.num_workers=20
#   CUDA_VISIBLE_DEVICES=0 bash scripts/data/bench_loader.sh bench.vae=true bench.allow_tf32=true
#   bash scripts/data/bench_loader.sh bench.rank=3 bench.world_size=8 bench.output_dir=outputs/data/bench_r3
#   bash scripts/data/bench_loader.sh --config configs/train/t2a_10b_s20_ws_cv.yaml data.loader.silence_pad.enabled=true bench.dump=32 bench.output_dir=outputs/data/bench_pad
#
# Output: numbers on stdout; outputs/data/bench/<nn>_<source>_<s>s[_pad<N>].{wav,txt}
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

"${PYTHON}" -m chrome.data.cli bench "$@"
