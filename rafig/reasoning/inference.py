"""Small forward-chaining and causal inference utilities.

This is a symbolic rule engine.  Conclusions are produced only when every
premise of an explicit rule is known; there are no generated or canned
answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .models import CausalRelation, Evidence, Hypothesis, HypothesisStatus


def _fact(value: str) -> str:
    return " ".join(value.strip().lower().split())


@dataclass(frozen=True, slots=True)
class InferenceRule:
    """A Horn-style rule: all premises imply one conclusion."""

    premises: tuple[str, ...]
    conclusion: str
    name: str = "rule"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "premises", tuple(_fact(item) for item in self.premises))
        object.__setattr__(self, "conclusion", _fact(self.conclusion))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        if not self.premises:
            raise ValueError("An inference rule must have at least one premise")
        if not self.conclusion:
            raise ValueError("An inference rule must have a conclusion")


@dataclass(slots=True)
class InferenceStep:
    conclusion: str
    premises: tuple[str, ...]
    rule_name: str
    confidence: float


@dataclass(slots=True)
class InferenceReport:
    facts: set[str]
    derived_facts: set[str]
    steps: list[InferenceStep] = field(default_factory=list)
    consistent: bool = True


class InferenceEngine:
    """Perform deterministic forward chaining over explicit facts and rules."""

    def __init__(self, rules: Iterable[InferenceRule] | None = None) -> None:
        self.rules: list[InferenceRule] = list(rules or [])

    def add_rule(self, rule: InferenceRule) -> None:
        if rule not in self.rules:
            self.rules.append(rule)

    def infer(self, facts: Iterable[str | Evidence]) -> set[str]:
        """Return the deductive closure, including the supplied facts."""
        return self.infer_with_trace(facts).facts

    def infer_with_trace(self, facts: Iterable[str | Evidence]) -> InferenceReport:
        known: set[str] = set()
        confidence: dict[str, float] = {}
        for item in facts:
            if isinstance(item, Evidence):
                proposition = _fact(item.statement)
                known.add(proposition)
                confidence[proposition] = max(confidence.get(proposition, 0.0), item.confidence)
            else:
                proposition = _fact(item)
                if proposition:
                    known.add(proposition)
                    confidence[proposition] = 1.0

        initial = set(known)
        steps: list[InferenceStep] = []
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                if rule.conclusion in known or not set(rule.premises).issubset(known):
                    continue
                premise_confidence = min(confidence.get(item, 1.0) for item in rule.premises)
                conclusion_confidence = premise_confidence * rule.confidence
                known.add(rule.conclusion)
                confidence[rule.conclusion] = conclusion_confidence
                steps.append(
                    InferenceStep(
                        conclusion=rule.conclusion,
                        premises=rule.premises,
                        rule_name=rule.name,
                        confidence=conclusion_confidence,
                    )
                )
                changed = True

        return InferenceReport(
            facts=known,
            derived_facts=known - initial,
            steps=steps,
            consistent=self.is_consistent(known),
        )

    def entails(self, conclusion: str, facts: Iterable[str | Evidence]) -> bool:
        return _fact(conclusion) in self.infer(facts)

    @staticmethod
    def is_consistent(facts: Iterable[str]) -> bool:
        normalized = {_fact(item) for item in facts}
        for proposition in normalized:
            if proposition.startswith("not "):
                opposite = proposition[4:]
            else:
                opposite = f"not {proposition}"
            if opposite in normalized:
                return False
        return True

    @staticmethod
    def evidence_from_report(report: InferenceReport) -> list[Evidence]:
        return [
            Evidence(
                statement=step.conclusion,
                source=f"inference:{step.rule_name}",
                confidence=step.confidence,
                supports=[step.conclusion],
                metadata={"premises": list(step.premises)},
            )
            for step in report.steps
        ]

    @staticmethod
    def evaluate_hypothesis(hypothesis: Hypothesis, evidence: Iterable[Evidence]) -> Hypothesis:
        """Update a hypothesis using explicit support and contradiction links."""
        proposition = _fact(hypothesis.statement)
        support: list[Evidence] = []
        contradiction: list[Evidence] = []
        for item in evidence:
            supported = {_fact(value) for value in item.supports}
            contradicted = {_fact(value) for value in item.contradicts}
            if proposition == _fact(item.statement) or proposition in supported:
                support.append(item)
            if proposition in contradicted or _fact(item.statement) == f"not {proposition}":
                contradiction.append(item)

        hypothesis.supporting_evidence_ids = [item.id for item in support]
        hypothesis.contradicting_evidence_ids = [item.id for item in contradiction]
        support_weight = sum(item.confidence for item in support)
        contradiction_weight = sum(item.confidence for item in contradiction)
        total = support_weight + contradiction_weight
        if total:
            hypothesis.confidence = support_weight / total
        if support_weight > contradiction_weight:
            hypothesis.status = HypothesisStatus.SUPPORTED
        elif contradiction_weight > support_weight:
            hypothesis.status = HypothesisStatus.REJECTED
        elif total:
            hypothesis.status = HypothesisStatus.UNCERTAIN
        return hypothesis


class CausalReasoner:
    """Track directed cause/effect links and predict reachable effects."""

    def __init__(self, relations: Iterable[CausalRelation] | None = None) -> None:
        self.relations: list[CausalRelation] = list(relations or [])

    def add_relation(self, relation: CausalRelation) -> None:
        if relation not in self.relations:
            self.relations.append(relation)

    def predict_effects(
        self,
        causes: Iterable[str],
        available_conditions: Iterable[str] | None = None,
    ) -> set[str]:
        active = {_fact(item) for item in causes}
        conditions = {_fact(item) for item in (available_conditions or [])} | active
        effects: set[str] = set()
        changed = True
        while changed:
            changed = False
            for relation in self.relations:
                cause = _fact(relation.cause)
                effect = _fact(relation.effect)
                required = {_fact(item) for item in relation.conditions}
                if cause in active and required.issubset(conditions) and effect not in active:
                    active.add(effect)
                    effects.add(effect)
                    changed = True
        return effects

    def causes_of(self, effect: str) -> list[CausalRelation]:
        normalized = _fact(effect)
        return [relation for relation in self.relations if _fact(relation.effect) == normalized]

    def effects_of(self, cause: str) -> list[CausalRelation]:
        normalized = _fact(cause)
        return [relation for relation in self.relations if _fact(relation.cause) == normalized]

    def explain(self, effect: str) -> list[str]:
        return [
            f"{relation.cause} {relation.relation} {relation.effect}"
            for relation in self.causes_of(effect)
        ]
