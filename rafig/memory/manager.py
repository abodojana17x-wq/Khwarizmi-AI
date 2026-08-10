"""MemoryManager — single entry point for every memory sub-system.

Usage::

    manager = MemoryManager(db_path=Path("data/rafiq_memory.db"))
    manager.conversation.add_turn("user", "Hi, how are you?")
    manager.semantic.store("Python is my preferred language")
    results = manager.search("Python")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .base import Importance, MemoryEntry
from .conversation import ConversationMemory
from .diagnostics import MemoryDiagnostics, collect_diagnostics
from .episodic import EpisodicMemory
from .project import ProjectMemory
from .semantic import SemanticMemory
from .working import WorkingMemory


@dataclass(slots=True)
class SearchHit:
    """One row in the unified search result list."""

    entry: MemoryEntry
    store: str
    score: float


class MemoryManager:
    """Owns every memory sub-system and exposes a unified search API."""

    def __init__(
        self,
        db_path: Path | str = Path("data/rafiq_memory.db"),
        working_capacity: int = 64,
        conversation_max: int = 200,
        episodic_max: int = 5000,
        semantic_max: int = 5000,
        project_max: int = 2000,
    ) -> None:
        self.db_path = Path(db_path)
        self.working = WorkingMemory(capacity=working_capacity)
        self.conversation = ConversationMemory(self.db_path, max_turns=conversation_max)
        self.episodic = EpisodicMemory(self.db_path, max_entries=episodic_max)
        self.semantic = SemanticMemory(self.db_path, max_entries=semantic_max)
        self.project = ProjectMemory(self.db_path, max_entries=project_max)

    # ------------------------------------------------------------------
    # Unified search
    # ------------------------------------------------------------------
    def search(self, query: str, limit: int = 10) -> List[SearchHit]:
        """Run a unified relevance search across every persistent store."""
        stores: Dict[str, "Memory"] = {
            "conversation": self.conversation,
            "episodic": self.episodic,
            "semantic": self.semantic,
            "project": self.project,
            "working": self.working,
        }
        terms = query.split() if query else []
        hits: List[SearchHit] = []
        for name, store in stores.items():
            for entry in store.retrieve(query=query, limit=limit):
                hits.append(SearchHit(entry=entry, store=name, score=entry.relevance_score(terms)))
        hits.sort(key=lambda hit: (-hit.score, -hit.entry.created_at))
        return hits[:limit]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def clear_all(self) -> Dict[str, int]:
        return {
            "working": self.working.clear(),
            "conversation": self.conversation.clear(),
            "episodic": self.episodic.clear(),
            "semantic": self.semantic.clear(),
            "project": self.project.clear(),
        }

    def close(self) -> None:
        self.conversation.close()
        self.episodic.close()
        self.semantic.close()
        self.project.close()

    def diagnostics(self) -> MemoryDiagnostics:
        return collect_diagnostics(self)

    def __enter__(self) -> "MemoryManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
