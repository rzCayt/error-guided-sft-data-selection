from eg_sft.experiment.budget_equivalent_blind import (
    blinded_registry,
    build_blind_map,
)


def _matrix() -> dict:
    methods = [
        "random_free_mix",
        "rds_error_free_mix",
        "random_common_mix",
        "rds_error_common_mix",
    ]
    jobs = []
    for replicate in range(1, 5):
        for method in methods:
            jobs.append(
                {
                    "cell_id": f"rep{replicate}_{method}_train17",
                    "replicate_index": replicate,
                    "method": method,
                    "selection_seed": replicate,
                    "train_seed": 17,
                    "selection_manifest": {"path": "x", "sha256": "a" * 64},
                }
            )
    return {
        "phase1_protocol_version": "budget-equivalent-phase1-matrix-v3",
        "methods": methods,
        "job_order": jobs,
        "output_root": ".aris/runs",
        "training": {
            "selection_budget": 500,
            "epochs": 2,
            "optimizer_steps": 64,
            "max_length": 512,
            "loss_normalization": "optimizer_step_response_token_sum_over_count",
            "single_training_process": True,
        },
        "evaluation": {"expected_record_count": 1319},
        "execution_policy": {
            "one_cell_per_invocation": True,
            "automatic_next_cell": False,
            "accuracy_blind_until_all_audits": True,
        },
    }


def test_blind_map_is_deterministic_and_public_manifest_hides_methods() -> None:
    private_1, public_1 = build_blind_map(
        matrix_config=_matrix(), matrix_sha256="b" * 64, secret_hex="01" * 32
    )
    private_2, public_2 = build_blind_map(
        matrix_config=_matrix(), matrix_sha256="b" * 64, secret_hex="01" * 32
    )
    assert private_1 == private_2
    assert public_1 == public_2
    assert set(private_1["method_to_alias"].values()) == {
        "method_A",
        "method_B",
        "method_C",
        "method_D",
    }
    public_text = str(public_1)
    assert "random_free_mix" not in public_text
    assert "rds_error_common_mix" not in public_text


def test_unblinding_requires_all_sixteen_audited_cells() -> None:
    private, _ = build_blind_map(
        matrix_config=_matrix(), matrix_sha256="b" * 64, secret_hex="02" * 32
    )
    jobs = [
        {
            "cell_id": row["cell_id"],
            "status": "AUDITED_PASS" if index < 15 else "PENDING",
        }
        for index, row in enumerate(private["cells"])
    ]
    blocked = blinded_registry(private_map=private, registry={"jobs": jobs})
    assert blocked["audited_pass_count"] == 15
    assert blocked["unblinding_permitted"] is False
    assert blocked["accuracy_withheld"] is True
    jobs[-1]["status"] = "AUDITED_PASS"
    ready = blinded_registry(private_map=private, registry={"jobs": jobs})
    assert ready["audited_pass_count"] == 16
    assert ready["unblinding_permitted"] is True
    assert ready["accuracy_withheld"] is False
