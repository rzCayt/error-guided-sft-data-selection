#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "usage: $0 REPO_ROOT MODEL_SNAPSHOT ARCHIVED_ADAPTER STATIC_ROOT SESSION_ROOT WORKER_ID EXPECTED_GPU_UUID SEMANTIC_MANIFEST DATASET_CACHE_REPORT" >&2
  exit 2
fi

REPO_ROOT="$1"
MODEL_SNAPSHOT="$2"
ARCHIVED_ADAPTER="$3"
STATIC_ROOT="$4"
SESSION_ROOT="$5"
WORKER_ID="$6"
EXPECTED_GPU_UUID="$7"
SEMANTIC_MANIFEST="$8"
DATASET_CACHE_REPORT="$9"
[[ "$WORKER_ID" == "gpu0" ]] || { echo "worker must be gpu0" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=17
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
cd "$REPO_ROOT"
mkdir -p "$SESSION_ROOT"

python scripts/collect_phase2_v8_environment.py \
  --worker-id gpu0 \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --static-runtime "$STATIC_ROOT/static_runtime.json" \
  --semantic-code-manifest "$SEMANTIC_MANIFEST" \
  --dataset-cache-report "$DATASET_CACHE_REPORT" \
  --output "$SESSION_ROOT/gpu0_environment.json"

python scripts/run_phase2_v8_canary.py \
  --role base16 \
  --worker-id gpu0 \
  --expected-gpu-uuid "$EXPECTED_GPU_UUID" \
  --environment-manifest "$SESSION_ROOT/gpu0_environment.json" \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --model-files-manifest "$STATIC_ROOT/model_files_manifest.json" \
  --tokenizer-files-manifest "$STATIC_ROOT/tokenizer_files_manifest.json" \
  --output-dir "$SESSION_ROOT/gpu0_base16"

python scripts/run_phase2_v8_canary.py \
  --role adapter128 \
  --worker-id gpu0 \
  --expected-gpu-uuid "$EXPECTED_GPU_UUID" \
  --environment-manifest "$SESSION_ROOT/gpu0_environment.json" \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --model-files-manifest "$STATIC_ROOT/model_files_manifest.json" \
  --tokenizer-files-manifest "$STATIC_ROOT/tokenizer_files_manifest.json" \
  --adapter-dir "$ARCHIVED_ADAPTER" \
  --output-dir "$SESSION_ROOT/gpu0_adapter128"

echo "PHASE2_V8_GPU0_INFERENCE_ANCHORS_READY"
