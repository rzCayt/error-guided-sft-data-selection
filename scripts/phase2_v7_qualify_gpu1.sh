#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 REPO_ROOT MODEL_SNAPSHOT ADAPTER_DIR STATIC_ROOT ADAPTER_ANCHOR SESSION_ROOT" >&2
  exit 2
fi

PHASE2_REPO_ROOT="$1"
PHASE2_MODEL_SNAPSHOT="$2"
PHASE2_ADAPTER_DIR="$3"
PHASE2_STATIC_ROOT="$4"
PHASE2_ADAPTER_ANCHOR="$5"
PHASE2_SESSION_ROOT="$6"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
cd "$PHASE2_REPO_ROOT"
mkdir -p "$PHASE2_SESSION_ROOT"

nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
python scripts/collect_phase2_v7_environment.py \
  --worker-id gpu1 \
  --model-snapshot "$PHASE2_MODEL_SNAPSHOT" \
  --static-runtime "$PHASE2_STATIC_ROOT/static_runtime/static_runtime.json" \
  --semantic-code-manifest "$PHASE2_STATIC_ROOT/semantic_code_manifest.json" \
  --output "$PHASE2_SESSION_ROOT/gpu1_environment.json"

python scripts/run_phase2_v7_canary.py \
  --role base_model_16 \
  --environment-manifest "$PHASE2_SESSION_ROOT/gpu1_environment.json" \
  --model-snapshot "$PHASE2_MODEL_SNAPSHOT" \
  --output-dir "$PHASE2_SESSION_ROOT/gpu1_base_canary"

python scripts/run_phase2_v7_canary.py \
  --role archived_adapter_16 \
  --environment-manifest "$PHASE2_SESSION_ROOT/gpu1_environment.json" \
  --model-snapshot "$PHASE2_MODEL_SNAPSHOT" \
  --adapter-dir "$PHASE2_ADAPTER_DIR" \
  --adapter-token-anchor "$PHASE2_ADAPTER_ANCHOR" \
  --output-dir "$PHASE2_SESSION_ROOT/gpu1_adapter_canary"

echo "PHASE2_V7_GPU1_CANARY_READY"
