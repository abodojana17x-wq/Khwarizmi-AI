"""Deterministic SCAMPER candidate generator."""
from __future__ import annotations
from dataclasses import dataclass

_HAZARD_TERMS = ("weapon", "explosive", "missile", "gun", "drone attack", "poison", "harmful optimization")
ACTIONS = {
    "Substitute": "Replace one costly or fragile element in {brief} with a simpler local alternative.",
    "Combine": "Combine {brief} with a complementary routine, checklist, or shared resource.",
    "Adapt": "Adapt a proven pattern from education, libraries, or repair culture to {brief}.",
    "Modify": "Modify the scale, timing, interface, or constraints of {brief} to reduce friction.",
    "Put to other use": "Reuse byproducts, data, or idle capacity from {brief} for a benign secondary benefit.",
    "Eliminate": "Remove the least useful step in {brief} and make the default path shorter.",
    "Reverse": "Reverse the usual order or ownership model of {brief} to expose a new workflow.",
}

@dataclass(frozen=True)
class ScamperCandidate:
    technique: str
    idea: str
    novelty: float
    usefulness: float
    rationale: str

@dataclass(frozen=True)
class ScamperReport:
    safety_verdict: str
    candidates: list[ScamperCandidate]
    message: str = ""

def generate_scamper(brief: str) -> ScamperReport:
    lowered = brief.lower()
    if any(t in lowered for t in _HAZARD_TERMS):
        return ScamperReport("blocked", [], "Redirect to safe, educational, non-hazardous ideation.")
    words = {w.strip(".,:;!?()[]").lower() for w in brief.split() if len(w.strip(".,:;!?()[]")) > 3}
    candidates = []
    for index, (technique, template) in enumerate(ACTIONS.items(), start=1):
        idea = template.format(brief=brief)
        unique_bonus = len(words.symmetric_difference(set(technique.lower().split()))) % 7 / 20
        novelty = round(min(1.0, 0.45 + index * 0.055 + unique_bonus), 2)
        usefulness = round(min(1.0, 0.82 - abs(4-index)*0.045 + min(len(words), 12)/100), 2)
        rationale = f"{technique} changes a distinct design lever; score uses deterministic brief vocabulary diversity and action practicality."
        candidates.append(ScamperCandidate(technique, idea, novelty, usefulness, rationale))
    candidates.sort(key=lambda c: (c.usefulness + c.novelty, c.usefulness), reverse=True)
    return ScamperReport("allowed", candidates)
