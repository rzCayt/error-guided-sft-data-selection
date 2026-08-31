#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 REPO_ROOT MODEL_SNAPSHOT OUTPUT_ROOT" >&2
  exit 2
fi

PHASE2_REPO_ROOT="$1"
PHASE2_MODEL_SNAPSHOT="$2"
PHASE2_OUTPUT_ROOT="$3"
cd "$PHASE2_REPO_ROOT"

python -m pytest \
  tests/test_phase2_crossed_v7.py \
  tests/test_phase2_v7_control.py \
  tests/test_phase2_v7_canary.py \
  tests/test_phase2_v7_environment.py \
  tests/test_phase2_v7_statistics.py \
  tests/test_identifiable_batch_backend.py -q

python scripts/build_phase2_v7_semantic_manifest.py \
  --parent-commit 54a232d60cba939f0ea1f212e5c8aae2a73bbd3c \
  --output "$PHASE2_OUTPUT_ROOT/semantic_code_manifest.json"

python scripts/prepare_phase2_v7_static_runtime.py \
  --model-snapshot "$PHASE2_MODEL_SNAPSHOT" \
  --semantic-code-manifest "$PHASE2_OUTPUT_ROOT/semantic_code_manifest.json" \
  --output-dir "$PHASE2_OUTPUT_ROOT/static_runtime"

python scripts/preflight_phase2_v7.py \
  --config configs/phase2_crossed_48cell_v7.json \
  --output-root "$PHASE2_OUTPUT_ROOT/preflight" \
  --min-free-gib 100

echo "PHASE2_V7_CPU_READY"
