"""Memory system for RAFIQ — Phase 05.

Provides a lightweight, fully offline memory architecture with:
- WorkingMemory   : fast in-process scratch pad (current task context)
- ConversationMemory : rolling window of recent user/assistant turns
- EpisodicMemory  : timestamped experiences stored in SQLite
- SemanticMemory  : facts and knowledge stored in SQLite
- ProjectMemory   : project-specific context stored in SQLite
- MemoryManager   : single entry point that owns every sub-store
"""

from .base import Memory, MemoryEntry, Importance
from .working import WorkingMemory
from .conversation import ConversationMemory
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .project import ProjectMemory
from .manager import MemoryManager
from .diagnostics import MemoryDiagnostics

__all__ = [
    "Memory",
    "MemoryEntry",
    "Importance",
    "WorkingMemory",
    "ConversationMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "ProjectMemory",
    "MemoryManager",
    "MemoryDiagnostics",
]
