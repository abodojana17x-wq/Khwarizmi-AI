"""
Khwarizmi Offline Agent Input Filter and Language Detector.

Implements lightweight input sanitization and multi-lingual prompt frame formatting
as specified in Section 2 of the Khwarizmi AI Blueprint.
"""

import re
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class SanitizedInputFrame:
    """
    Structured Input Frame prepared for the Khwarizmi Neural Core.

    Attributes:
        raw_text: Cleaned natural language or source code text.
        detected_language: Detected language code ("en", "ar", "franco", "code", "mixed").
        has_code_payload: Boolean flag indicating presence of Python/code snippets.
        has_dag_payload: Boolean flag indicating structured project tasks or DAG descriptions.
        metadata: Additional sanitization flags.
    """
    raw_text: str
    detected_language: str
    has_code_payload: bool
    has_dag_payload: bool
    metadata: Dict[str, Any]


class InputSanitizer:
    """
    Lightweight pre-sanitizer and multi-lingual prompt classifier.
    """

    ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF]")
    CODE_KEYWORDS_PATTERN = re.compile(r"\b(def |class |import |from |return |if __name__ ==)\b")
    DAG_KEYWORDS_PATTERN = re.compile(r"\b(task|dependency|prerequisite|milestone|subtask|pipeline|dag)\b", re.IGNORECASE)

    @classmethod
    def sanitize(cls, raw_input: str) -> SanitizedInputFrame:
        """
        Sanitize raw string prompt and classify payload modalities.

        Args:
            raw_input: Raw string input from user or API.

        Returns:
            SanitizedInputFrame dataclass.
        """
        cleaned = raw_input.strip()

        has_arabic = bool(cls.ARABIC_CHAR_PATTERN.search(cleaned))
        has_english = bool(re.search(r"[a-zA-Z]", cleaned))
        has_code = bool(cls.CODE_KEYWORDS_PATTERN.search(cleaned)) or ("```" in cleaned)
        has_dag = bool(cls.DAG_KEYWORDS_PATTERN.search(cleaned))

        if has_code:
            lang = "code"
        elif has_arabic and has_english:
            lang = "mixed"
        elif has_arabic:
            lang = "ar"
        else:
            lang = "en"

        return SanitizedInputFrame(
            raw_text=cleaned,
            detected_language=lang,
            has_code_payload=has_code,
            has_dag_payload=has_dag,
            metadata={
                "char_count": len(cleaned),
                "word_count": len(cleaned.split()),
            },
        )
