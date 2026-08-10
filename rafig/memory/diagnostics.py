"""Diagnostics for the memory subsystem."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .base import Memory


@dataclass(slots=True)
class MemoryDiagnostics:
    """Snapshot describing the current memory footprint."""

    working_size: int
    working_capacity: int
    conversation_size: int
    conversation_max: int
    episodic_size: int
    episodic_max: int
    semantic_size: int
    semantic_max: int
    project_size: int
    project_max: int
    db_size_bytes: int
    db_path: str

    def to_dict(self) -> Dict[str, int | str]:
        return {
            "working_size": self.working_size,
            "working_capacity": self.working_capacity,
            "conversation_size": self.conversation_size,
            "conversation_max": self.conversation_max,
            "episodic_size": self.episodic_size,
            "episodic_max": self.episodic_max,
            "semantic_size": self.semantic_size,
            "semantic_max": self.semantic_max,
            "project_size": self.project_size,
            "project_max": self.project_max,
            "db_size_bytes": self.db_size_bytes,
            "db_path": self.db_path,
        }

    def __str__(self) -> str:
        parts = [
            "RAFIQ Memory Diagnostics",
            "========================",
            f"Working memory    : {self.working_size}/{self.working_capacity} entries",
            f"Conversation store: {self.conversation_size}/{self.conversation_max} turns",
            f"Episodic store    : {self.episodic_size}/{self.episodic_max} episodes",
            f"Semantic store    : {self.semantic_size}/{self.semantic_max} facts",
            f"Project store     : {self.project_size}/{self.project_max} items",
            f"Database size     : {self.db_size_bytes} bytes ({self.db_path})",
        ]
        return "\n".join(parts)


def collect_diagnostics(manager: "MemoryManager") -> MemoryDiagnostics:  # type: ignore[name-defined]
    from .manager import MemoryManager  # local import to avoid cycle

    db_path = Path(manager.db_path)
    size = db_path.stat().st_size if db_path.exists() else 0
    return MemoryDiagnostics(
        working_size=manager.working.size(),
        working_capacity=manager.working.capacity,
        conversation_size=manager.conversation.size(),
        conversation_max=manager.conversation.max_turns,
        episodic_size=manager.episodic.size(),
        episodic_max=manager.episodic.max_entries,
        semantic_size=manager.semantic.size(),
        semantic_max=manager.semantic.max_entries,
        project_size=manager.project.size(),
        project_max=manager.project.max_entries,
        db_size_bytes=size,
        db_path=str(db_path),
    )
