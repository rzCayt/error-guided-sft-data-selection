import ast
import subprocess
import sys
from pathlib import Path

from eg_sft.experiment.cloud_v2_train_runtime import (
    expected_temperature_sample_count,
)


def test_uninterrupted_profiles_have_equal_temperature_sample_count() -> None:
    assert expected_temperature_sample_count(optimizer_steps_planned=4) == 5


def test_fixed_entry_is_independent_of_the_rejected_prototype() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_b500_cloud_v2_train_calibration_fixed.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
    assert "run_b500_cloud_v2_train_calibration import" not in source
    assert "compute_seconds_excluding_monitor_and_checkpoint_io" in source
    assert "temperature_sampling_rule" in source


def test_fixed_entry_help_is_cpu_safe() -> None:
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [
            sys.executable,
            "scripts/run_b500_cloud_v2_train_calibration_fixed.py",
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0
    assert "--profile" in process.stdout
    assert "--resume-run-dir" in process.stdout
