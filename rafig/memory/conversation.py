"""ConversationMemory — a rolling window of user/assistant turns.

Conversation memory is stored in a local SQLite database so that
multi-session continuity is possible without any external service.
The conversation keeps at most ``max_turns`` entries; older turns are
evicted starting with the least-important ones.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, List, Optional

from .base import Importance, Memory, MemoryEntry


class ConversationMemory(Memory):
    """SQLite-backed rolling conversation log."""

    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_SYSTEM = "system"

    def __init__(self, db_path: Path, max_turns: int = 200) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self._db_path = Path(db_path)
        self._max_turns = max_turns
        self._conn = self._connect()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversation (
                id TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                context TEXT NOT NULL DEFAULT '',
                importance INTEGER NOT NULL DEFAULT 2,
                tags TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_conversation_created
                ON conversation(created_at);
            CREATE INDEX IF NOT EXISTS idx_conversation_importance
                ON conversation(importance);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def max_turns(self) -> int:
        return self._max_turns

    def add_turn(self, role: str, content: str, **kwargs: Any) -> MemoryEntry:
        """Convenience wrapper around :pymeth:`store` for conversational turns."""
        kwargs["source"] = kwargs.get("source", f"conversation:{role}")
        kwargs["metadata"] = {**kwargs.get("metadata", {}), "role": role}
        entry = self.store(content, **kwargs)
        self._trim_if_needed()
        return entry

    def store(self, content: str, **kwargs: Any) -> MemoryEntry:
        entry = MemoryEntry(
            content=content,
            source=kwargs.get("source", "conversation"),
            context=kwargs.get("context", ""),
            importance=kwargs.get("importance", Importance.NORMAL),
            tags=list(kwargs.get("tags", [])),
            metadata=dict(kwargs.get("metadata", {})),
        )
        self._conn.execute(
            """
            INSERT INTO conversation
                (id, role, content, source, context, importance, tags, metadata,
                 created_at, updated_at, access_count, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.metadata.get("role", self.ROLE_USER),
                entry.content,
                entry.source,
                entry.context,
                entry.importance.value,
                json.dumps(entry.tags),
                json.dumps(entry.metadata),
                entry.created_at,
                entry.updated_at,
                entry.access_count,
                entry.last_accessed,
            ),
        )
        self._conn.commit()
        return entry

    def retrieve(self, query: str = "", limit: int = 10) -> List[MemoryEntry]:
        if not query:
            rows = self._conn.execute(
                "SELECT * FROM conversation ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            like = f"%{query}%"
            rows = self._conn.execute(
                """
                SELECT * FROM conversation
                WHERE content LIKE ? OR context LIKE ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (like, like, limit),
            ).fetchall()
        entries = [self._row_to_entry(row) for row in rows]
        for entry in entries:
            entry.touch()
        return entries

    def recent(self, limit: int = 10) -> List[MemoryEntry]:
        """Return the most recent turns in chronological order."""
        rows = self._conn.execute(
            "SELECT * FROM conversation ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_entry(row) for row in reversed(rows)]

    def update(self, entry_id: str, **changes: Any) -> Optional[MemoryEntry]:
        allowed = {"content", "source", "context", "importance", "tags", "metadata"}
        updates = {k: v for k, v in changes.items() if k in allowed}
        if not updates:
            return self._fetch(entry_id)
        updates["updated_at"] = time.time()
        serialised: dict[str, Any] = {}
        for key, value in updates.items():
            if isinstance(value, Importance):
                serialised[key] = value.value
            elif isinstance(value, (list, dict)):
                serialised[key] = json.dumps(value)
            else:
                serialised[key] = value
        set_clause = ", ".join(f"{key} = ?" for key in serialised)
        values = list(serialised.values()) + [entry_id]
        cursor = self._conn.execute(
            f"UPDATE conversation SET {set_clause} WHERE id = ?",
            values,
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            return None
        return self._fetch(entry_id)

    def delete(self, entry_id: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM conversation WHERE id = ?", (entry_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def clear(self) -> int:
        count = self.size()
        self._conn.execute("DELETE FROM conversation")
        self._conn.commit()
        return count

    def size(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM conversation").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _fetch(self, entry_id: str) -> Optional[MemoryEntry]:
        row = self._conn.execute(
            "SELECT * FROM conversation WHERE id = ?", (entry_id,)
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def _row_to_entry(self, row: sqlite3.Row | None) -> Optional[MemoryEntry]:
        if row is None:
            return None
        tags = json.loads(row["tags"]) if isinstance(row["tags"], str) else []
        metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else {}
        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            source=row["source"],
            context=row["context"],
            importance=Importance(row["importance"]),
            tags=tags,
            metadata=metadata,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            access_count=row["access_count"],
            last_accessed=row["last_accessed"],
        )

    def _trim_if_needed(self) -> None:
        total = self.size()
        if total <= self._max_turns:
            return
        excess = total - self._max_turns
        self._conn.execute(
            """
            DELETE FROM conversation WHERE id IN (
                SELECT id FROM conversation
                ORDER BY importance ASC, created_at ASC
                LIMIT ?
            )
            """,
            (excess,),
        )
        self._conn.commit()
