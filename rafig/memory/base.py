"""Base types shared by every memory sub-system."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Importance(int, Enum):
    """How important a memory entry is.

    Higher numeric value == more important.  Used for relevance scoring
    and as a tiebreaker when the memory store needs to evict entries.
    """

    TRIVIAL = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(slots=True)
class MemoryEntry:
    """A single unit of stored memory.

    Every sub-system persists ``MemoryEntry`` objects, although the
    in-memory ``WorkingMemory`` keeps a lighter representation internally
    and only materialises a full entry on export.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    content: str = ""
    source: str = ""
    context: str = ""
    importance: Importance = Importance.NORMAL
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Mark the entry as accessed right now."""
        self.access_count += 1
        self.last_accessed = time.time()

    def relevance_score(self, query_terms: List[str]) -> float:
        """Return a 0..1 score describing how relevant the entry is.

        The score mixes keyword overlap, importance weight and recency.
        """
        if not query_terms:
            return 0.5

        text = f"{self.content} {self.context} {' '.join(self.tags)}".lower()
        hits = sum(1 for term in query_terms if term.lower() in text)
        overlap = hits / len(query_terms)

        importance_weight = self.importance.value / Importance.CRITICAL.value
        age_hours = max(0.0, (time.time() - self.created_at) / 3600.0)
        recency = 1.0 / (1.0 + age_hours / 24.0)

        return round(0.5 * overlap + 0.3 * importance_weight + 0.2 * recency, 4)


class Memory:
    """Abstract interface every memory sub-system implements."""

    def store(self, content: str, **kwargs: Any) -> MemoryEntry:
        raise NotImplementedError

    def retrieve(self, query: str = "", limit: int = 10) -> List[MemoryEntry]:
        raise NotImplementedError

    def update(self, entry_id: str, **changes: Any) -> Optional[MemoryEntry]:
        raise NotImplementedError

    def delete(self, entry_id: str) -> bool:
        raise NotImplementedError

    def clear(self) -> int:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError
