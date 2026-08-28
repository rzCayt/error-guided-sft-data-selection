#!/usr/bin/env bash
set -u
set -o pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/budget-v3-controller-qualification-3627bef}"
analysis_root="${ANALYSIS_ROOT:-/root/autodl-tmp/error-guided-sft-analysis-run-93da459}"
output_root="${SMOKE128_OUTPUT_ROOT:-/root/autodl-tmp/budget-v3-controller-qualification-smoke128-20260827-p1-v2}"
offline_compat_root="${OFFLINE_COMPAT_ROOT:-/root/autodl-tmp/offline-hf-compat-v3}"
model_snapshot="${EG_SFT_OFFLINE_MODEL_SNAPSHOT:-/root/autodl-tmp/hf-cache/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323}"
python_bin="${PYTHON_BIN:-python}"
runner="$project_root/scripts/run_identifiable_backend_qualification_v2.py"
config="$project_root/configs/identifiable_backend_qualification_v2.json"
random_root="${RANDOM_COMMON_ADAPTER_ROOT:-$analysis_root/.aris/compute/budget_equivalent_phase1_runs_v3/20260824T214450Z_budget_equivalent_rep1_random_common_mix_train17_e51c7e5a2b_s17}"
rds_root="${RDS_COMMON_ADAPTER_ROOT:-$analysis_root/.aris/compute/budget_equivalent_phase1_runs_v3/20260825T023716Z_budget_equivalent_rep1_rds_error_common_mix_train17_7a6317a9f5_s17}"
log_path="$output_root/controller.log"

for required_path in \
  "$project_root" \
  "$analysis_root" \
  "$offline_compat_root/sitecustomize.py" \
  "$model_snapshot" \
  "$runner" \
  "$config" \
  "$project_root/.aris/compute/budget_equivalent_v3_selections/information_gates.json" \
  "$random_root/audit/formal_cell_audit.json" \
  "$random_root/training_complete/adapter/adapter_model.safetensors" \
  "$rds_root/audit/formal_cell_audit.json" \
  "$rds_root/training_complete/adapter/adapter_model.safetensors"; do
  if [[ ! -e "$required_path" ]]; then
    echo "required qualification path is missing: $required_path" >&2
    exit 2
  fi
done

mkdir -p "$output_root"
exec 9>"$output_root/controller.lock"
if ! flock -n 9; then
  echo "another smoke128 controller holds the lock" >&2
  exit 3
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export EG_SFT_OFFLINE_MODEL_SNAPSHOT="$model_snapshot"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="$offline_compat_root:$project_root/src:$project_root/scripts"

common_args=(
  --config "$config"
  --output-root "$output_root"
  --adapter-root "random_common_rep1_seed17=$random_root"
  --adapter-root "rds_error_common_rep1_seed17=$rds_root"
)

log_hardware() {
  date --iso-8601=seconds
  nvidia-smi --query-gpu=name,uuid,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader
}

run_cli() {
  {
    echo "COMMAND_START $(date --iso-8601=seconds) $*"
    log_hardware
  } | tee -a "$log_path"
  "$python_bin" "$runner" "${common_args[@]}" "$@" 2>&1 | tee -a "$log_path"
  status=${PIPESTATUS[0]}
  echo "COMMAND_END $(date --iso-8601=seconds) status=$status $*" | tee -a "$log_path"
  return "$status"
}

run_and_replay() {
  model_id="$1"
  batch_size="$2"
  if ! run_cli --stage smoke128 --model-id "$model_id" --batch-size "$batch_size"; then
    return 1
  fi
  run_cli --stage smoke128 --model-id "$model_id" --batch-size "$batch_size"
}

cd "$project_root"

if ! run_cli --contract-only; then
  exit 10
fi

# The only artificial interruption: base, batch2, exactly 64 of 128 records.
if ! run_cli --stage smoke128 --model-id base --batch-size 2 --stop-after-records 64; then
  exit 11
fi
if ! run_and_replay base 2; then
  exit 12
fi

for batch_size in 1 4 8; do
  if ! run_and_replay base "$batch_size"; then
    if [[ "$batch_size" == "1" ]]; then
      exit 20
    fi
  fi
done

for model_id in random_common_rep1_seed17 rds_error_common_rep1_seed17; do
  for batch_size in 1 2 4 8; do
    if ! run_and_replay "$model_id" "$batch_size"; then
      if [[ "$batch_size" == "1" ]]; then
        exit 30
      fi
    fi
  done
done

if ! run_cli --select-smoke; then
  exit 40
fi
if ! run_cli --audit-smoke128; then
  exit 41
fi

sha256sum "$output_root/qualification_smoke128.json" | tee -a "$log_path"
echo "SMOKE128_CONTROLLER_COMPLETE $(date --iso-8601=seconds)" | tee -a "$log_path"
