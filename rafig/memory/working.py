"""WorkingMemory — a fast in-process scratch pad.

WorkingMemory holds the transient state of whatever RAFIQ is doing
*right now*: the current task, intermediate results, flags, etc.
Nothing is persisted to disk; the store is wiped when the process
exits or when :pymeth:`clear` is called.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from .base import Importance, Memory, MemoryEntry


class WorkingMemory(Memory):
    """Bounded LRU scratch pad kept entirely in RAM."""

    def __init__(self, capacity: int = 64) -> None:
        if capacity < 1:
            raise ValueError("WorkingMemory capacity must be >= 1")
        self._capacity = capacity
        self._store: OrderedDict[str, MemoryEntry] = OrderedDict()

    @property
    def capacity(self) -> int:
        return self._capacity

    def store(self, content: str, **kwargs: Any) -> MemoryEntry:
        entry = MemoryEntry(
            content=content,
            source=kwargs.get("source", "working"),
            context=kwargs.get("context", ""),
            importance=kwargs.get("importance", Importance.NORMAL),
            tags=list(kwargs.get("tags", [])),
            metadata=dict(kwargs.get("metadata", {})),
        )
        self._store[entry.id] = entry
        self._store.move_to_end(entry.id)
        self._evict_if_needed()
        return entry

    def retrieve(self, query: str = "", limit: int = 10) -> List[MemoryEntry]:
        terms = query.split()
        scored = []
        for entry in self._store.values():
            entry.touch()
            scored.append((entry.relevance_score(terms), entry))
        scored.sort(key=lambda pair: (-pair[0], -pair[1].created_at))
        return [entry for _, entry in scored[:limit]]

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        entry = self._store.get(entry_id)
        if entry is not None:
            entry.touch()
            self._store.move_to_end(entry_id)
        return entry

    def update(self, entry_id: str, **changes: Any) -> Optional[MemoryEntry]:
        entry = self._store.get(entry_id)
        if entry is None:
            return None
        for key, value in changes.items():
            if hasattr(entry, key) and key not in {"id", "created_at"}:
                setattr(entry, key, value)
        entry.updated_at = time.time()
        entry.touch()
        return entry

    def delete(self, entry_id: str) -> bool:
        return self._store.pop(entry_id, None) is not None

    def clear(self) -> int:
        count = len(self._store)
        self._store.clear()
        return count

    def size(self) -> int:
        return len(self._store)

    def _evict_if_needed(self) -> None:
        while len(self._store) > self._capacity:
            self._store.popitem(last=False)
