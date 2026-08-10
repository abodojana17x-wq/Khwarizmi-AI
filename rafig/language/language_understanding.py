"""Lightweight language understanding for RAFIQ."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(slots=True)
class LanguageAnalysis:
    """Structured analysis of a user input."""

    raw_text: str
    normalized_text: str
    detected_language: str
    language_mix: Dict[str, int] = field(default_factory=dict)
    egyptian_indicators: List[str] = field(default_factory=list)
    is_code_like: bool = False
    text_content: str = ""
    sentence_boundaries: List[int] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    intent_candidates: List[str] = field(default_factory=list)


class LanguageAnalyzer:
    """Offline analyzer for Arabic, English, Franco, and code-like text."""

    def __init__(self) -> None:
        self._arabic_chars = set("أإآابةتثجحخسشصضطظعغفقكلمنهويءىؤئًٌٍَُِّْ")
        self._english_words = {
            "want": "request",
            "fix": "repair",
            "repair": "repair",
            "code": "code",
            "program": "code",
            "help": "help",
            "create": "create",
            "learn": "learn",
            "python": "python",
        }
        self._arabic_words = {
            "عايز": "request",
            "أريد": "request",
            "أعمل": "create",
            "أصلح": "repair",
            "اصلح": "repair",
            "اكمل": "continue",
            "تعلم": "learn",
            "بايثون": "python",
            "كود": "code",
            "برمجة": "code",
        }
        self._franco_words = {
            "3ayez": "request",
            "3ayezak": "request",
            "tesala7": "repair",
            "salah": "repair",
            "code": "code",
            "python": "python",
            "program": "code",
        }

    def analyze(self, text: str) -> LanguageAnalysis:
        normalized = self._normalize(text)
        language = self._detect_language(normalized)
        mix = self._detect_mix(normalized)
        egyptian = self._detect_egyptian_indicators(normalized)
        is_code_like = self._looks_like_code(normalized)
        entities = self._extract_entities(normalized, is_code_like)
        intent_candidates = self._infer_intents(normalized)
        boundaries = self._estimate_boundaries(normalized)
        return LanguageAnalysis(
            raw_text=text,
            normalized_text=normalized,
            detected_language=language,
            language_mix=mix,
            egyptian_indicators=egyptian,
            is_code_like=is_code_like,
            text_content=normalized,
            sentence_boundaries=boundaries,
            entities=entities,
            intent_candidates=intent_candidates,
        )

    def _normalize(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return text.strip()

    def _detect_language(self, text: str) -> str:
        arabic_words = self._count_arabic_words(text)
        english_count = sum(1 for token in re.findall(r"\b\w+\b", text.lower()) if token in self._english_words)
        franco_count = sum(1 for token in re.findall(r"\w+", text.lower()) if token in self._franco_words)
        if arabic_words > 0 and (english_count > 0 or franco_count > 0):
            return "mixed"
        if arabic_words > 0:
            return "ar"
        if franco_count > 0 and franco_count >= english_count:
            return "franco"
        if english_count > 0 or "python" in text.lower():
            return "en"
        return "unknown"

    def _detect_mix(self, text: str) -> Dict[str, int]:
        counts: Dict[str, int] = {"ar": 0, "en": 0, "franco": 0}
        counts["ar"] = self._count_arabic_words(text)
        if re.search(r"\b(?:want|fix|repair|python|program|help|create|learn)\b", text.lower()):
            counts["en"] += 1
        if re.search(r"\b(?:3ayez|3ayezak|tesala7|salah|code|python|program)\b", text.lower()):
            counts["franco"] += 1
        if "python" in text.lower():
            counts["en"] += 1
        return counts

    def _detect_egyptian_indicators(self, text: str) -> List[str]:
        indicators: List[str] = []
        egyptian_words = ["عايز", "عايزك", "أعمل", "أصلح", "كود", "إللي", "هعمل"]
        for word in egyptian_words:
            if word in text:
                indicators.append(word)
        return indicators

    def _looks_like_code(self, text: str) -> bool:
        code_markers = ["def ", "import ", "class ", "return ", "if ", "for ", "while ", ":", "(", ")"]
        return any(marker in text for marker in code_markers)

    def _extract_entities(self, text: str, is_code_like: bool) -> List[str]:
        entities: List[str] = []
        lowered = text.lower()
        if "python" in lowered:
            entities.append("python")
        if "code" in lowered or "kood" in lowered:
            entities.append("code")
        if is_code_like or re.search(r"\b(def|return|import|class)\b", lowered):
            entities.append("function")
            entities.append("python")
        return sorted(set(entities))

    def _infer_intents(self, text: str) -> List[str]:
        lowered = text.lower()
        intents: List[str] = []
        if any(token in lowered for token in ["fix", "repair", "اصلح", "أصلح", "تصلح", "tesala7", "salah", "تصلحلي"]):
            intents.append("repair")
        if any(token in lowered for token in ["create", "أعمل", "اعمل", "عمل", "program"]):
            intents.append("create")
        if any(token in lowered for token in ["learn", "تعلم", "at3alem"]):
            intents.append("learn")
        return intents

    def _estimate_boundaries(self, text: str) -> List[int]:
        sentence_endings = [match.end() for match in re.finditer(r"[.!?]", text)]
        if not sentence_endings:
            return [len(text)]
        return sentence_endings

    def _count_arabic_words(self, text: str) -> int:
        return len(re.findall(r"[\u0600-\u06FF]+", text))
