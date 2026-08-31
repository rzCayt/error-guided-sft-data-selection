from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v8_launch_scripts_fail_closed_and_keep_one_gpu_process() -> None:
    worker = (ROOT / "scripts/phase2_v8_run_worker.sh").read_text(encoding="utf-8")
    assert "START_PHASE2_V8_COMMON24" in worker
    assert "--release-authorization" in worker
    assert "--deployment-manifest" in worker
    assert "--resume-interrupted" in worker
    assert "CUDA_VISIBLE_DEVICES=0" in worker
    assert " &\n" not in worker
    assert "\n&" not in worker


def test_v8_host_preparation_runs_before_qualification() -> None:
    prepare = (ROOT / "scripts/phase2_v8_prepare_host.sh").read_text(encoding="utf-8")
    assert "audit_phase2_v8_materialized_contracts.py" in prepare
    assert "preflight_phase2_v8.py" in prepare
    assert "prepare_phase2_v8_static_runtime.py" in prepare
    assert "qualify_phase2_v8_offline_datasets.py" in prepare
    assert "HF_DATASETS_OFFLINE=1" in prepare
    assert "run_phase2_v8_canary.py" not in prepare


def test_v8_gpu_qualification_binds_offline_dataset_report() -> None:
    for name in ("phase2_v8_qualify_gpu0.sh", "phase2_v8_qualify_gpu1.sh"):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "DATASET_CACHE_REPORT" in source
        assert "--dataset-cache-report" in source


def test_v8_materialized_contract_root_is_canonical_everywhere() -> None:
    worker = (ROOT / "scripts/run_phase2_v8_worker.py").read_text(encoding="utf-8")
    anchor = (ROOT / "scripts/run_phase2_v8_training_anchor.py").read_text(
        encoding="utf-8"
    )
    preflight = (ROOT / "scripts/preflight_phase2_v8.py").read_text(encoding="utf-8")
    shell = (ROOT / "scripts/phase2_v8_training_anchor_worker.sh").read_text(
        encoding="utf-8"
    )
    for source in (worker, anchor, preflight):
        assert 'role="materialized_contracts"' in source
    assert "--canonical-runtime-files" in anchor
    assert "CANONICAL_MANIFEST" in shell


def test_v8_cpu_release_gate_forces_offline_mode() -> None:
    source = (ROOT / "scripts/phase2_v8_cpu_release_gate.sh").read_text(
        encoding="utf-8"
    )
    assert "HF_HUB_OFFLINE=1" in source
    assert "TRANSFORMERS_OFFLINE=1" in source
    assert "HF_DATASETS_OFFLINE=1" in source
