from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


GOAL_TYPES = (
    "distance_threshold",
    "contact",
    "dihedral_window",
    "rmsd_threshold",
    "state_membership",
)

GOAL_OPERATORS = (
    "less_than",
    "greater_than",
    "inside",
)

VALUE_TARGETS = (
    "event",
    "flux",
    "event_and_flux",
)


@dataclass(frozen=True)
class GoalSpec:
    """
    Structured user target for goal-conditioned STRIDE scoring.

    Version 1 intentionally uses auditable YAML-style fields. The same
    object can drive delayed-label generation and provide numeric conditioning
    features to the model.
    """

    name: str
    type: str
    selections: tuple[str, ...]
    operator: str
    threshold: float
    horizon_iterations: int
    value_target: str = "event_and_flux"
    lower_bound: float | None = None
    upper_bound: float | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GoalSpec":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoalSpec":
        goal = data.get("goal", data)

        selections = goal.get("selections")
        if selections is None:
            selection_a = goal.get("selection_a")
            selection_b = goal.get("selection_b")
            selections = [s for s in (selection_a, selection_b) if s is not None]

        if selections is None:
            selections = []

        spec = cls(
            name=str(goal["name"]),
            type=str(goal["type"]),
            selections=tuple(str(s) for s in selections),
            operator=str(goal["operator"]),
            threshold=float(goal["threshold"]),
            horizon_iterations=int(goal["horizon_iterations"]),
            value_target=str(goal.get("value_target", "event_and_flux")),
            lower_bound=(
                float(goal["lower_bound"]) if goal.get("lower_bound") is not None else None
            ),
            upper_bound=(
                float(goal["upper_bound"]) if goal.get("upper_bound") is not None else None
            ),
        )
        spec.validate()
        return spec

    @property
    def feature_dim(self) -> int:
        return len(GOAL_TYPES) + len(GOAL_OPERATORS) + len(VALUE_TARGETS) + 4

    def validate(self) -> None:
        if self.type not in GOAL_TYPES:
            raise ValueError(f"Unknown goal type: {self.type}")
        if self.operator not in GOAL_OPERATORS:
            raise ValueError(f"Unknown goal operator: {self.operator}")
        if self.value_target not in VALUE_TARGETS:
            raise ValueError(f"Unknown value target: {self.value_target}")
        if self.horizon_iterations <= 0:
            raise ValueError("horizon_iterations must be positive.")
        if self.type == "dihedral_window":
            if self.operator != "inside":
                raise ValueError("dihedral_window goals require operator='inside'.")
            if self.lower_bound is None or self.upper_bound is None:
                raise ValueError(
                    "dihedral_window goals require lower_bound and upper_bound."
                )
            if len(self.selections) != 4:
                raise ValueError("dihedral_window goals require exactly four selections.")

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert the structured goal into a deterministic numeric vector.

        This is the Phase 1 goal encoder input. A learned embedding layer can
        consume this vector without making the public goal interface latent or
        hard to audit.
        """
        self.validate()

        values: list[float] = []
        values.extend(_one_hot(self.type, GOAL_TYPES))
        values.extend(_one_hot(self.operator, GOAL_OPERATORS))
        values.extend(_one_hot(self.value_target, VALUE_TARGETS))
        values.append(float(self.threshold))
        values.append(float(self.horizon_iterations))
        values.append(float(self.lower_bound) if self.lower_bound is not None else 0.0)
        values.append(float(self.upper_bound) if self.upper_bound is not None else 0.0)

        return np.asarray(values, dtype=np.float32)


def _one_hot(value: str, choices: tuple[str, ...]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]
