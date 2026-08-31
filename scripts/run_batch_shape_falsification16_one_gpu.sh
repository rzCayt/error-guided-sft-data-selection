#!/usr/bin/env bash
set -u
set -o pipefail

project_root="${PROJECT_ROOT:-/root/autodl-tmp/budget-v3-controller-qualification-3627bef}"
source_root="${SOURCE_SMOKE_ROOT:-/root/autodl-tmp/budget-v3-controller-qualification-smoke128-20260827-p1-v2/runs}"
output_root="${FALSIFICATION16_OUTPUT_ROOT:-/root/autodl-tmp/batch-shape-falsification16-20260827-v1}"
offline_compat_root="${OFFLINE_COMPAT_ROOT:-/root/autodl-tmp/offline-hf-compat-v3}"
model_snapshot="${EG_SFT_OFFLINE_MODEL_SNAPSHOT:-/root/autodl-tmp/hf-cache/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323}"
python_bin="${PYTHON_BIN:-python}"
runner="$project_root/scripts/run_batch_shape_falsification16.py"
config="$project_root/configs/batch_shape_falsification16_v1.json"
log_path="$output_root/controller.log"

for required_path in \
  "$project_root" \
  "$source_root" \
  "$offline_compat_root/sitecustomize.py" \
  "$model_snapshot" \
  "$runner" \
  "$config"; do
  if [[ ! -e "$required_path" ]]; then
    echo "required falsification path is missing: $required_path" >&2
    exit 2
  fi
done

mkdir -p "$output_root"
exec 9>"$output_root/controller.lock"
if ! flock -n 9; then
  echo "another falsification16 controller holds the lock" >&2
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
  --source-smoke-root "$source_root"
  --output-root "$output_root"
)

run_cli() {
  {
    echo "COMMAND_START $(date --iso-8601=seconds) $*"
    nvidia-smi --query-gpu=name,uuid,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader
  } | tee -a "$log_path"
  "$python_bin" "$runner" "${common_args[@]}" "$@" 2>&1 | tee -a "$log_path"
  status=${PIPESTATUS[0]}
  echo "COMMAND_END $(date --iso-8601=seconds) status=$status $*" | tee -a "$log_path"
  return "$status"
}

read_decision() {
  phase="$1"
  "$python_bin" -c "import json; print(json.load(open(r'$output_root/audit/$phase.json', encoding='utf-8'))['decision'])"
}

stop_if_not_continue() {
  phase="$1"
  decision="$(read_decision "$phase")"
  echo "PHASE_DECISION $(date --iso-8601=seconds) phase=$phase decision=$decision" | tee -a "$log_path"
  if [[ "$decision" != "CONTINUE" ]]; then
    echo "FALSIFICATION16_STOPPED $(date --iso-8601=seconds) decision=$decision" | tee -a "$log_path"
    exit 0
  fi
}

check_gpu_budget() {
  if [[ -z "${gpu_started_epoch:-}" ]]; then
    return
  fi
  now_epoch="$(date +%s)"
  elapsed_seconds=$((now_epoch - gpu_started_epoch))
  if (( elapsed_seconds >= 1500 )); then
    echo "FALSIFICATION16_BUDGET_STOP elapsed_seconds=$elapsed_seconds" | tee -a "$log_path"
    exit 0
  fi
}

cd "$project_root"

run_cli --contract-only || exit 10
run_cli --prepare || exit 11

gpu_started_epoch="$(date +%s)"

run_cli --pass-id bf16_b1_natural_repeat || exit 20
run_cli --audit-phase baseline_repeat || exit 21
stop_if_not_continue baseline_repeat
check_gpu_budget

run_cli --pass-id bf16_b4_fixed_a || exit 30
run_cli --pass-id bf16_b4_fixed_b || exit 31
run_cli --audit-phase bf16_repeat || exit 32
stop_if_not_continue bf16_repeat
check_gpu_budget

run_cli --pass-id bf16_b1_fixed || exit 40
run_cli --audit-phase width_effect || exit 41
stop_if_not_continue width_effect
check_gpu_budget

run_cli --fp32-preflight || {
  echo "FALSIFICATION16_STOPPED $(date --iso-8601=seconds) decision=PERMANENT_BATCH1_FP32_PREFLIGHT_FAILED" | tee -a "$log_path"
  exit 0
}
run_cli --pass-id fp32_b1_fixed || {
  echo "FALSIFICATION16_STOPPED $(date --iso-8601=seconds) decision=PERMANENT_BATCH1_FP32_B1_FAILED" | tee -a "$log_path"
  exit 0
}
check_gpu_budget
run_cli --pass-id fp32_b4_fixed || {
  echo "FALSIFICATION16_STOPPED $(date --iso-8601=seconds) decision=PERMANENT_BATCH1_FP32_B4_FAILED" | tee -a "$log_path"
  exit 0
}
run_cli --audit-phase final_mechanism || exit 50

final_decision="$(read_decision final_mechanism)"
echo "FALSIFICATION16_COMPLETE $(date --iso-8601=seconds) decision=$final_decision" | tee -a "$log_path"
sha256sum "$output_root/audit/"*.json "$output_root/"*.json | tee -a "$log_path"
