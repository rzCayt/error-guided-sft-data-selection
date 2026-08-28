#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 REPO_ROOT MODEL_SNAPSHOT MATERIALIZED_CONTRACTS STATIC_ROOT HOST_PREFLIGHT_ROOT SEMANTIC_MANIFEST" >&2
  exit 2
fi

REPO_ROOT="$1"
MODEL_SNAPSHOT="$2"
MATERIALIZED_CONTRACTS="$3"
STATIC_ROOT="$4"
HOST_PREFLIGHT_ROOT="$5"
SEMANTIC_MANIFEST="$6"
cd "$REPO_ROOT"
export HF_DATASETS_OFFLINE=1

python scripts/audit_phase2_v8_materialized_contracts.py \
  --config configs/phase2_clean_common24_v8_canonical.json \
  --contract-root "$MATERIALIZED_CONTRACTS" \
  --output "$HOST_PREFLIGHT_ROOT/materialized_contract_audit.json"

python scripts/preflight_phase2_v8.py \
  --config configs/phase2_clean_common24_v8_canonical.json \
  --canonical-runtime-files configs/CANONICAL_RUNTIME_FILES_v8_RELEASE.json \
  --precision-simulation artifacts/phase2_v8_preflight/precision_simulation.json \
  --parent-evidence-index artifacts/phase2_v8_parent_evidence/PARENT_EVIDENCE_INDEX.json \
  --training-input-contract-root "$MATERIALIZED_CONTRACTS" \
  --output-root "$HOST_PREFLIGHT_ROOT/preflight" \
  --min-free-gib 80

python scripts/prepare_phase2_v8_static_runtime.py \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --semantic-code-manifest "$SEMANTIC_MANIFEST" \
  --output-dir "$STATIC_ROOT"

python scripts/qualify_phase2_v8_offline_datasets.py \
  --config configs/phase2_clean_common24_v8_canonical.json \
  --output "$HOST_PREFLIGHT_ROOT/dataset_cache_qualification.json"

echo "PHASE2_V8_HOST_CPU_READY"
