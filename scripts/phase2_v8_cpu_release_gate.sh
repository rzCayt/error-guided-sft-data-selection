#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 REPO_ROOT DEPLOYMENT_MANIFEST RELEASE_ARCHIVE OUTPUT_DIR" >&2
  exit 2
fi

REPO_ROOT="$1"
DEPLOYMENT_MANIFEST="$2"
RELEASE_ARCHIVE="$3"
OUTPUT_DIR="$4"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "$REPO_ROOT"

python scripts/phase2_v8_cpu_release_gate.py \
  --deployment-manifest "$DEPLOYMENT_MANIFEST" \
  --canonical-runtime configs/CANONICAL_RUNTIME_FILES_v8_RELEASE.json \
  --release-archive "$RELEASE_ARCHIVE" \
  --output-dir "$OUTPUT_DIR"
