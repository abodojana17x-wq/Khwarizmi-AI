"""Plain-text parser for educational physics problem frames."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Optional

_HAZARD_TERMS = ("weapon", "explosive", "warhead", "missile", "gun", "bullet", "armor penetration", "blast radius", "detonate", "lethal")
_VALUE_UNIT_RE = re.compile(r"(?P<name>[A-Za-z][A-Za-z0-9_ ]{0,32}?)\s*(?:=|is|of)?\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*(?P<unit>m/s\^2|m/s|kg|m|s|N|J|W|Pa|Hz|K|C|V|A|mol|rad|deg|cm|km|g)\b", re.I)
_UNKNOWN_RE = re.compile(r"(?:find|calculate|determine|what is|solve for)\s+(?:the\s+)?(?P<unknown>[A-Za-z][A-Za-z0-9_ /-]{1,60})(?:\?|\.|,|$)", re.I)
_ASSUMPTION_MARKERS = ("assume", "neglect", "ignore", "frictionless", "constant", "uniform", "ideal")


@dataclass(frozen=True)
class Quantity:
    name: str
    value: Optional[float] = None
    unit: str = ""
    raw: str = ""


@dataclass(frozen=True)
class PhysicsProblemFrame:
    quantities: List[Quantity] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)
    givens: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    requested_output: str = ""
    safety_verdict: str = "allowed"
    safety_reason: str = ""


def parse_physics_problem(statement: str) -> PhysicsProblemFrame:
    """Parse a plain-text educational physics statement into a deterministic frame."""
    text = " ".join(statement.strip().split())
    lowered = text.lower()
    if any(term in lowered for term in _HAZARD_TERMS):
        return PhysicsProblemFrame(requested_output="safe educational redirection", safety_verdict="blocked", safety_reason="Hazardous engineering or weapons-related optimization request detected.")

    quantities: list[Quantity] = []
    for match in _VALUE_UNIT_RE.finditer(text):
        name = match.group("name").strip(" ,.;:")
        # Keep the closest noun-like phrase compact and readable.
        name = re.sub(r"^(a|an|the|with|and|from|at|moving at|mass of)\s+", "", name, flags=re.I).strip() or "quantity"
        raw = match.group(0)
        quantities.append(Quantity(name=name, value=float(match.group("value")), unit=match.group("unit"), raw=raw))

    givens = [q.raw for q in quantities]
    units = {q.name: q.unit for q in quantities}
    unknowns = [m.group("unknown").strip(" .?") for m in _UNKNOWN_RE.finditer(text)]
    if not unknowns and "?" in text:
        unknowns = [text.rsplit("?", 1)[0].split(".")[-1].strip()]
    assumptions = []
    for sentence in re.split(r"(?<=[.!?])\s+", statement.strip()):
        if any(marker in sentence.lower() for marker in _ASSUMPTION_MARKERS):
            assumptions.append(sentence.strip())
    requested = unknowns[0] if unknowns else "unspecified"
    return PhysicsProblemFrame(quantities, units, givens, unknowns, assumptions, requested)
