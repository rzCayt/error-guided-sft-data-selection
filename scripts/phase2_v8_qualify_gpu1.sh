#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 11 ]]; then
  echo "usage: $0 REPO_ROOT MODEL_SNAPSHOT ARCHIVED_ADAPTER STATIC_ROOT SESSION_ROOT WORKER_ID EXPECTED_GPU_UUID GPU0_BASE_ANCHOR GPU0_ADAPTER_ANCHOR SEMANTIC_MANIFEST DATASET_CACHE_REPORT" >&2
  exit 2
fi

REPO_ROOT="$1"
MODEL_SNAPSHOT="$2"
ARCHIVED_ADAPTER="$3"
STATIC_ROOT="$4"
SESSION_ROOT="$5"
WORKER_ID="$6"
EXPECTED_GPU_UUID="$7"
GPU0_BASE_ANCHOR="$8"
GPU0_ADAPTER_ANCHOR="$9"
SEMANTIC_MANIFEST="${10}"
DATASET_CACHE_REPORT="${11}"
[[ "$WORKER_ID" == "gpu1" ]] || { echo "worker must be gpu1" >&2; exit 2; }
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
  --worker-id gpu1 \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --static-runtime "$STATIC_ROOT/static_runtime.json" \
  --semantic-code-manifest "$SEMANTIC_MANIFEST" \
  --dataset-cache-report "$DATASET_CACHE_REPORT" \
  --output "$SESSION_ROOT/gpu1_environment.json"

python scripts/run_phase2_v8_canary.py \
  --role base16 \
  --worker-id gpu1 \
  --expected-gpu-uuid "$EXPECTED_GPU_UUID" \
  --environment-manifest "$SESSION_ROOT/gpu1_environment.json" \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --model-files-manifest "$STATIC_ROOT/model_files_manifest.json" \
  --tokenizer-files-manifest "$STATIC_ROOT/tokenizer_files_manifest.json" \
  --new-block-token-anchor "$GPU0_BASE_ANCHOR" \
  --output-dir "$SESSION_ROOT/gpu1_base16"

python scripts/run_phase2_v8_canary.py \
  --role adapter128 \
  --worker-id gpu1 \
  --expected-gpu-uuid "$EXPECTED_GPU_UUID" \
  --environment-manifest "$SESSION_ROOT/gpu1_environment.json" \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --model-files-manifest "$STATIC_ROOT/model_files_manifest.json" \
  --tokenizer-files-manifest "$STATIC_ROOT/tokenizer_files_manifest.json" \
  --adapter-dir "$ARCHIVED_ADAPTER" \
  --new-block-token-anchor "$GPU0_ADAPTER_ANCHOR" \
  --output-dir "$SESSION_ROOT/gpu1_adapter128"

echo "PHASE2_V8_GPU1_INFERENCE_ANCHORS_READY"
