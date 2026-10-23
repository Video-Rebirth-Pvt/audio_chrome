#!/usr/bin/env bash
# CommonVoice 26.0 manifests for training: es / en dialect tags from the speaker's accent, zh-HK tagged
# [Cantonese][Cantonese], the other locales copied. Rerun after the raw manifests change, then rebuild the
# cv_* indexes (their manifest stamps change, so build_index picks them up).
#
#   bash scripts/data/prepare_commonvoice.sh
#   bash scripts/data/prepare_commonvoice.sh --src /path/to/raw/jsonl --out /path/to/jsonl_chrome
#
# Output: ${paths.commonvoice_jsonl} = /mnt/data04/480634/public/0_DATA/1_AUDIO/ASR/jsonl_chrome/
#         {commonvoice_<locale>_26.0.resp.jsonl, report.json}
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

"${PYTHON}" scripts/data/prepare_commonvoice.py "$@"
