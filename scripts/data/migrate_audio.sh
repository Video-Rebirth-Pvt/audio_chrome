#!/usr/bin/env bash
# S1 data: replace the HF parquet shards of WorldSpeech by Arrow files of their audio bytes, in place:
#   data/<locale>/<shard>.parquet  →  data/<locale>/<shard>.arrow   (audio bytes as is + segment_id)
# Every Arrow file is read back and compared with the parquet bytes before it is renamed into place, and its
# segment_ids are checked against qa_sidecar/<locale>/<shard>.parquet, which keeps every other column.
# Only with migrate.delete=true is the parquet then deleted. Shards whose Arrow file is current are not
# rewritten, so an interrupted run just resumes. Each shard's outcome is appended to migrate.report.
#
# Usage (from anywhere; extra args are config overrides, see `migrate:` in configs/data/t2a_worldspeech.yaml):
#   bash scripts/data/migrate_audio.sh migrate.dry_run=true                          # what would happen
#   bash scripts/data/migrate_audio.sh migrate.locales=[yue_hk]                      # one locale, keep parquet
#   bash scripts/data/migrate_audio.sh migrate.locales=[yue_hk] migrate.delete=true  # then delete its parquet
#   bash scripts/data/migrate_audio.sh migrate.workers=32                            # all 128 locales
#   bash scripts/data/migrate_audio.sh migrate.delete=true                           # all, delete verified parquet
#
# Exit code 1 if any shard failed; failed shards keep their parquet and are listed at the end.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh

"${PYTHON}" -m chrome.data.cli migrate --config configs/data/t2a_worldspeech.yaml "$@"
