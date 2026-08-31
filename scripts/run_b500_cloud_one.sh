#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: bash scripts/run_b500_cloud_one.sh <random|rds_all|rds_error> <17|29|41>" >&2
  exit 2
fi

strategy="$1"
seed="$2"
case "$strategy" in
  random|rds_all|rds_error) ;;
  *) echo "invalid strategy: $strategy" >&2; exit 2 ;;
esac
case "$seed" in
  17|29|41) ;;
  *) echo "invalid seed: $seed" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

log_root="${B500_CLOUD_LOG_ROOT:-/root/autodl-tmp/b500-cloud-logs}"
mkdir -p "$log_root"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_path="$log_root/${timestamp}_${strategy}_seed${seed}.log"
matrix="configs/b500_formal_matrix_cloud_4090_v1.json"

echo "cloud_job strategy=$strategy seed=$seed log=$log_path"
python scripts/preflight_b500_formal_matrix.py --matrix-config "$matrix" | tee "$log_path"
python scripts/run_b500_formal_resumable.py \
  --matrix-config "$matrix" \
  --strategy "$strategy" \
  --seed "$seed" \
  --preflight-only | tee -a "$log_path"
python scripts/run_b500_formal_resumable.py \
  --matrix-config "$matrix" \
  --strategy "$strategy" \
  --seed "$seed" 2>&1 | tee -a "$log_path"

echo "cloud_job_complete strategy=$strategy seed=$seed log=$log_path"
