"""SemanticMemory — facts and knowledge.

Semantic memories store abstract facts that outlive any single
conversation: definitions, user preferences, project conventions,
learned rules.
"""

from __future__ import annotations

from pathlib import Path

from ._sqlite import SQLiteMemory


class SemanticMemory(SQLiteMemory):
    _table = "semantics"
    _default_source = "semantic"
