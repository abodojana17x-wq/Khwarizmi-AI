"""Transparent comparison of candidate actions."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping

from .models import ActionComparison, CandidateAction, Constraint


class ActionEvaluator:
    """Score actions with a small, inspectable multi-criteria function."""

    DEFAULT_WEIGHTS = {
        "utility": 0.30,
        "feasibility": 0.25,
        "constraint_satisfaction": 0.20,
        "evidence_support": 0.10,
        "estimated_cost": -0.075,
        "risk": -0.075,
    }

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            unknown = set(weights) - set(self.DEFAULT_WEIGHTS)
            if unknown:
                raise ValueError(f"Unknown action score weights: {sorted(unknown)}")
            self.weights.update(weights)

    def score(self, action: CandidateAction) -> float:
        value = sum(float(getattr(action, key)) * weight for key, weight in self.weights.items())
        return round(value, 6)

    def compare(
        self,
        actions: Iterable[CandidateAction],
        constraints: Iterable[Constraint] | None = None,
    ) -> ActionComparison:
        options = list(actions)
        if not options:
            return ActionComparison([], None, "No candidate actions were supplied.")

        constraints = list(constraints or [])
        hard_violations = any(constraint.satisfied is False and constraint.kind.value == "hard" for constraint in constraints)
        scored: list[CandidateAction] = []
        for option in options:
            candidate = replace(option)
            if hard_violations:
                candidate.constraint_satisfaction = 0.0
            candidate.score = self.score(candidate)
            scored.append(candidate)
        ranked = sorted(scored, key=lambda item: (item.score if item.score is not None else float("-inf"), item.name), reverse=True)
        selected = ranked[0]
        rationale = (
            f"Selected '{selected.name}' with score {selected.score:.3f}; "
            "score combines utility, feasibility, constraint fit, evidence, cost, and risk."
        )
        return ActionComparison(ranked, selected.name, rationale)
