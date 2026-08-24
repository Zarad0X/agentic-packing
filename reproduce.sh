#!/usr/bin/env bash
set -euo pipefail

backend="${1:-quasistatic}"

python -m physcensis.cli generate \
  --config configs/paper.yaml \
  --program examples/dining_table.json \
  --output "output/scenes/dining_table_${backend}" \
  --backend "${backend}" \
  --stability-samples 0
