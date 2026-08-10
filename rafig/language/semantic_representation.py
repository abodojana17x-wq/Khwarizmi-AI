"""Lightweight semantic representation layer for RAFIQ."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import re


@dataclass(slots=True)
class SemanticRepresentation:
    """Normalized semantic meaning for a user request."""

    raw_text: str
    intent: str
    action: str
    object: str
    language: str
    goals: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    confidence: float = 0.0


class SemanticAnalyzer:
    """Convert language-understanding output into a reusable semantic form."""

    def __init__(self) -> None:
        self._repair_terms = {"repair", "fix", "اصلح", "تصلح", "tesala7", "salah"}
        self._create_terms = {"create", "أعمل", "اعمل", "build", "make"}
        self._python_terms = {"python", "بايثون", "py"}

    def analyze(self, text: str) -> SemanticRepresentation:
        normalized = self._normalize(text)
        lowered = normalized.lower()
        if any(term in lowered for term in self._repair_terms):
            intent = "code_repair"
            action = "repair"
            obj = "python_code" if any(term in lowered for term in self._python_terms) else "code"
            language = self._detect_language(normalized)
            confidence = 0.9
            goals = ["repair_existing_code"]
            constraints = ["preserve_intent"]
            entities = [obj]
            relationships = ["request->repair"]
        elif any(term in lowered for term in self._create_terms):
            intent = "code_generation"
            action = "create"
            obj = "python_code" if any(term in lowered for term in self._python_terms) else "code"
            language = self._detect_language(normalized)
            confidence = 0.85
            goals = ["generate_new_code"]
            constraints = ["produce_python"]
            entities = [obj]
            relationships = ["request->create"]
        else:
            intent = "unknown"
            action = "understand"
            obj = "text"
            language = self._detect_language(normalized)
            confidence = 0.5
            goals = ["understand_request"]
            constraints = []
            entities = []
            relationships = []

        return SemanticRepresentation(
            raw_text=text,
            intent=intent,
            action=action,
            object=obj,
            language=language,
            goals=goals,
            constraints=constraints,
            entities=entities,
            relationships=relationships,
            confidence=confidence,
        )

    def _normalize(self, text: str) -> str:
        return text.strip()

    def _detect_language(self, text: str) -> str:
        if re.search(r"[\u0600-\u06FF]", text):
            if re.search(r"[A-Za-z]", text):
                return "mixed"
            return "ar"
        if re.search(r"[A-Za-z]", text):
            return "en"
        return "unknown"
