from eg_sft.training.token_budget import balanced_optimizer_step_plan


def test_phase1_step_partition_has_sixty_four_boundaries() -> None:
    plan = balanced_optimizer_step_plan(
        epoch_orders=[list(range(500)), list(reversed(range(500)))],
        optimizer_steps=64,
    )
    assert len(plan) == 64
    assert sum(map(len, plan)) == 1000
