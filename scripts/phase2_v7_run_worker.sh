#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 REPO_ROOT WORKER_ID ENV_MANIFEST LEGACY_REPORT RUNTIME_ROOT PACKAGE_ROOT" >&2
  exit 2
fi

PHASE2_REPO_ROOT="$1"
PHASE2_WORKER_ID="$2"
PHASE2_ENV_MANIFEST="$3"
PHASE2_LEGACY_REPORT="$4"
PHASE2_RUNTIME_ROOT="$5"
PHASE2_PACKAGE_ROOT="$6"
export CUDA_VISIBLE_DEVICES=0
cd "$PHASE2_REPO_ROOT"

python scripts/run_phase2_v7_worker.py \
  --config configs/phase2_crossed_48cell_v7.json \
  --worker-id "$PHASE2_WORKER_ID" \
  --cuda-visible-device 0 \
  --environment-manifest "$PHASE2_ENV_MANIFEST" \
  --legacy-backend-report "$PHASE2_LEGACY_REPORT" \
  --control-root "$PHASE2_RUNTIME_ROOT/control" \
  --log-root "$PHASE2_RUNTIME_ROOT/logs" \
  --package-root "$PHASE2_PACKAGE_ROOT" \
  --monitor-seconds 300 \
  --hard-stop-temperature-c 80 \
  --max-attempts 3 \
  --cpu-threads 8 \
  --resume-interrupted \
  --operator-confirmation PHASE2_V7_32CELL_BLOCK_APPROVED
