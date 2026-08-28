from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eg_sft.experiment.cell_evidence_package import package_cell_evidence
from eg_sft.experiment.phase2_v7_control import Phase2StateStore, worker_schedule


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase2_clean_common24_v8_canonical.json"


def test_two_controllers_cannot_lock_the_same_cell(tmp_path: Path) -> None:
    store = Phase2StateStore(root=tmp_path / "control", matrix_path=CONFIG)
    store.initialize()
    cell = worker_schedule(store.matrix, "gpu0")[0]

    def acquire(attempt: str) -> str:
        try:
            store.transition(
                cell_id=cell,
                target="LOCKED",
                worker_id="gpu0",
                attempt_id=attempt,
                reason="failure injection contention",
            )
            return "PASS"
        except ValueError:
            return "REJECTED"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(acquire, ("attempt-a", "attempt-b")))
    assert sorted(results) == ["PASS", "REJECTED"]
    assert store.read_state(cell)["state"] == "LOCKED"


def test_crash_after_lock_preserves_attempt_and_requires_new_attempt(tmp_path: Path) -> None:
    store = Phase2StateStore(root=tmp_path / "control", matrix_path=CONFIG)
    store.initialize()
    cell = worker_schedule(store.matrix, "gpu1")[0]
    store.transition(
        cell_id=cell,
        target="LOCKED",
        worker_id="gpu1",
        attempt_id="crashed",
        reason="injected crash after lock",
    )
    store.fail(
        cell_id=cell,
        worker_id="gpu1",
        attempt_id="crashed",
        reason="controller restart records old attempt",
    )
    store.transition(
        cell_id=cell,
        target="LOCKED",
        worker_id="gpu1",
        attempt_id="recovery",
        reason="new recovery attempt",
    )
    events = (store.cell_dir(cell) / "events.jsonl").read_text(encoding="utf-8")
    assert "crashed" in events and "recovery" in events
    assert store.read_state(cell)["attempt_id"] == "recovery"


def _write(path: Path, payload: str = "{}\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def test_evidence_package_uses_atomic_final_name_and_ignores_stale_temp(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    manifest = {
        "run_id": "run-1",
        "git_commit": "a" * 40,
        "config_hash": "b" * 64,
        "config": {
            "cell_id": "v8_rep1_random_common_mix_train17",
            "study": "clean_new_environment_common_block",
        },
    }
    _write(run / "manifest.json", json.dumps(manifest))
    required = (
        "resolved_recipe.json",
        "optimizer_step_tokens.jsonl",
        "runtime_events.jsonl",
        "cell_complete.json",
        "training_input_contract.json",
        "release_binding.json",
        "audit/formal_cell_audit.json",
        "audit/formal_cell_audit.sha256",
        "audit/ood_audit.json",
        "audit/ood_audit.sha256",
        "training_complete/training_metrics.json",
        "training_complete/token_audit.json",
        "training_complete/token_budget_audit.json",
        "training_complete/adapter/adapter_model.safetensors",
        "training_complete/adapter/adapter_config.json",
    )
    for relative in required:
        _write(run / relative)
    output = tmp_path / "cell_evidence.tar.gz"
    stale = output.parent / f".{output.name}.stale.tmp"
    stale.write_bytes(b"interrupted-attempt")
    package_cell_evidence(run_dir=run, output=output)
    assert output.is_file()
    assert output.with_suffix(output.suffix + ".sha256").is_file()
    assert stale.read_bytes() == b"interrupted-attempt"


def test_v8_forbids_parallel_ood_lanes_and_requires_sequential_formal_workers() -> None:
    worker_source = (ROOT / "scripts/run_phase2_v8_worker.py").read_text(
        encoding="utf-8"
    )
    cell_source = (ROOT / "scripts/run_budget_equivalent_cell.py").read_text(
        encoding="utf-8"
    )
    assert "_run_ood_lanes" not in worker_source
    assert "for dataset in (\"svamp\", \"asdiv_numeric\", \"multiarith\")" in worker_source
    assert "sequential_gpu_workers" in cell_source
    assert "v8 sequential formal worker failure" in cell_source
