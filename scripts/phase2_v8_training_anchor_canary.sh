#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 10 ]]; then
  echo "usage: $0 REPO_ROOT WORKER_ID EXPECTED_GPU_UUID ENV_MANIFEST MODEL_SNAPSHOT MODEL_MANIFEST TOKENIZER_MANIFEST ANCHOR_ADAPTER OUTPUT_DIR TOKEN_ANCHOR_OR_NONE" >&2
  exit 2
fi

REPO_ROOT="$1"
WORKER_ID="$2"
EXPECTED_GPU_UUID="$3"
ENV_MANIFEST="$4"
MODEL_SNAPSHOT="$5"
MODEL_MANIFEST="$6"
TOKENIZER_MANIFEST="$7"
ANCHOR_ADAPTER="$8"
OUTPUT_DIR="$9"
TOKEN_ANCHOR="${10}"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=17
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "$REPO_ROOT"

COMMAND=(python scripts/run_phase2_v8_canary.py
  --role training_anchor128
  --worker-id "$WORKER_ID"
  --expected-gpu-uuid "$EXPECTED_GPU_UUID"
  --environment-manifest "$ENV_MANIFEST"
  --model-snapshot "$MODEL_SNAPSHOT"
  --model-files-manifest "$MODEL_MANIFEST"
  --tokenizer-files-manifest "$TOKENIZER_MANIFEST"
  --adapter-dir "$ANCHOR_ADAPTER"
  --output-dir "$OUTPUT_DIR")
if [[ "$TOKEN_ANCHOR" != "NONE" ]]; then
  COMMAND+=(--new-block-token-anchor "$TOKEN_ANCHOR")
fi
"${COMMAND[@]}"
