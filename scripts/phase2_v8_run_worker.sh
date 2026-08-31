#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 15 ]]; then
  echo "usage: $0 REPO_ROOT WORKER_ID ENV_MANIFEST BACKEND_REPORT TRAINING_ANCHOR_REPORT MATERIALIZED_CONTRACTS MODEL_SNAPSHOT MODEL_MANIFEST TOKENIZER_MANIFEST CANONICAL_MANIFEST DEPLOYMENT_MANIFEST RELEASE_ARCHIVE RELEASE_GO RUNTIME_ROOT PACKAGE_ROOT" >&2
  exit 2
fi

REPO_ROOT="$1"
WORKER_ID="$2"
ENV_MANIFEST="$3"
BACKEND_REPORT="$4"
TRAINING_ANCHOR_REPORT="$5"
MATERIALIZED_CONTRACTS="$6"
MODEL_SNAPSHOT="$7"
MODEL_MANIFEST="$8"
TOKENIZER_MANIFEST="$9"
CANONICAL_MANIFEST="${10}"
DEPLOYMENT_MANIFEST="${11}"
RELEASE_ARCHIVE="${12}"
RELEASE_GO="${13}"
RUNTIME_ROOT="${14}"
PACKAGE_ROOT="${15}"
CPU_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=17
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
cd "$REPO_ROOT"

python scripts/run_phase2_v8_worker.py \
  --config configs/phase2_clean_common24_v8_canonical.json \
  --canonical-runtime-files "$CANONICAL_MANIFEST" \
  --deployment-manifest "$DEPLOYMENT_MANIFEST" \
  --release-archive "$RELEASE_ARCHIVE" \
  --release-authorization "$RELEASE_GO" \
  --worker-id "$WORKER_ID" \
  --environment-manifest "$ENV_MANIFEST" \
  --legacy-backend-report "$BACKEND_REPORT" \
  --training-anchor-report "$TRAINING_ANCHOR_REPORT" \
  --training-input-contract-root "$MATERIALIZED_CONTRACTS" \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --model-files-manifest "$MODEL_MANIFEST" \
  --tokenizer-files-manifest "$TOKENIZER_MANIFEST" \
  --control-root "$RUNTIME_ROOT/control" \
  --log-root "$RUNTIME_ROOT/logs" \
  --package-root "$PACKAGE_ROOT" \
  --monitor-seconds 300 \
  --hard-stop-temperature-c 80 \
  --max-attempts 3 \
  --cpu-threads "$CPU_THREADS" \
  --resume-interrupted \
  --operator-confirmation START_PHASE2_V8_COMMON24
