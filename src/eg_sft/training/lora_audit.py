"""Mechanical checks for LoRA trainability and gradient isolation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


_DEFAULT_ADAPTER_MARKERS = ("lora_", "modules_to_save")


@dataclass(frozen=True)
class LoraAuditReport:
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int
    missing_trainable_gradients: tuple[str, ...] = ()
    frozen_parameters_with_gradients: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_lora_parameters(
    model: torch.nn.Module,
    *,
    adapter_markers: tuple[str, ...] = _DEFAULT_ADAPTER_MARKERS,
) -> LoraAuditReport:
    """Require every trainable parameter to belong to an adapter module."""

    trainable_names: list[str] = []
    frozen_names: list[str] = []
    unexpected_trainable: list[str] = []
    trainable_parameters = 0
    total_parameters = 0

    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total_parameters += count
        if parameter.requires_grad:
            trainable_names.append(name)
            trainable_parameters += count
            if not any(marker in name for marker in adapter_markers):
                unexpected_trainable.append(name)
        else:
            frozen_names.append(name)

    if not trainable_names:
        raise AssertionError("no trainable adapter parameters were found")
    if unexpected_trainable:
        raise AssertionError(
            "non-adapter parameters are trainable: " + ", ".join(unexpected_trainable)
        )

    return LoraAuditReport(
        trainable_names=tuple(trainable_names),
        frozen_names=tuple(frozen_names),
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
    )


def audit_lora_gradients(
    model: torch.nn.Module,
    *,
    adapter_markers: tuple[str, ...] = _DEFAULT_ADAPTER_MARKERS,
) -> LoraAuditReport:
    """Verify a completed backward pass touched adapters and no frozen weights."""

    parameter_report = audit_lora_parameters(
        model, adapter_markers=adapter_markers
    )
    missing_trainable_gradients: list[str] = []
    frozen_parameters_with_gradients: list[str] = []

    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is None:
            missing_trainable_gradients.append(name)
        if not parameter.requires_grad and parameter.grad is not None:
            frozen_parameters_with_gradients.append(name)

    if missing_trainable_gradients:
        raise AssertionError(
            "trainable parameters without gradients: "
            + ", ".join(missing_trainable_gradients)
        )
    if frozen_parameters_with_gradients:
        raise AssertionError(
            "frozen parameters received gradients: "
            + ", ".join(frozen_parameters_with_gradients)
        )

    return LoraAuditReport(
        trainable_names=parameter_report.trainable_names,
        frozen_names=parameter_report.frozen_names,
        trainable_parameters=parameter_report.trainable_parameters,
        total_parameters=parameter_report.total_parameters,
        missing_trainable_gradients=tuple(missing_trainable_gradients),
        frozen_parameters_with_gradients=tuple(frozen_parameters_with_gradients),
    )
