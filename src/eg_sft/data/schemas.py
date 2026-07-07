from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TaskFamily = Literal[
    "ratio_change",
    "multiplicative_relation",
    "weighted_aggregation",
    "temporal_numeric_constraint",
]

SplitName = Literal[
    "candidate_pool",
    "dev_diagnostic",
    "test_id",
    "test_ood_template",
    "test_ood_range",
]

Difficulty = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class Buckets:
    difficulty_bucket: str
    answer_magnitude_bucket: str
    reasoning_length_bucket: str


@dataclass(frozen=True)
class Example:
    id: str
    split: SplitName
    task_family: TaskFamily
    difficulty: Difficulty
    prompt: str
    answer: float
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)
    buckets: Buckets = field(
        default_factory=lambda: Buckets(
            difficulty_bucket="easy",
            answer_magnitude_bucket="small",
            reasoning_length_bucket="short",
        )
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["answer"] = round(float(self.answer), 6)
        return data


@dataclass(frozen=True)
class DiagnosticResult:
    id: str
    split: str
    task_family: str
    difficulty_bucket: str
    answer_magnitude_bucket: str
    reasoning_length_bucket: str
    answer: float
    prediction: str
    parsed_prediction: float | None
    parse_success: bool
    numeric_accuracy: bool
    output_length: int
    error_type: str
