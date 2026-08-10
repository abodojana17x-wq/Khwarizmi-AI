"""Generic SQLite-backed persistent memory used by the episodic,
semantic and project sub-systems.

The three concrete stores differ only in table name and the default
``source`` value; all behaviour lives here so we avoid duplication.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional

from .base import Importance, Memory, MemoryEntry


class SQLiteMemory(Memory):
    """Base class for SQLite-backed memory stores."""

    _table: str = "memories"
    _default_source: str = "generic"

    def __init__(self, db_path: Path, max_entries: int = 5000) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._db_path = Path(db_path)
        self._max_entries = max_entries
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
        table = self._table
        self._conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT '',
                context TEXT NOT NULL DEFAULT '',
                importance INTEGER NOT NULL DEFAULT 2,
                tags TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{{}}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_{table}_importance
                ON {table}(importance);
            CREATE INDEX IF NOT EXISTS idx_{table}_created
                ON {table}(created_at);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    @property
    def max_entries(self) -> int:
        return self._max_entries

    def store(self, content: str, **kwargs: Any) -> MemoryEntry:
        entry = MemoryEntry(
            content=content,
            source=kwargs.get("source", self._default_source),
            context=kwargs.get("context", ""),
            importance=kwargs.get("importance", Importance.NORMAL),
            tags=list(kwargs.get("tags", [])),
            metadata=dict(kwargs.get("metadata", {})),
        )
        self._conn.execute(
            f"""
            INSERT INTO {self._table}
                (id, content, source, context, importance, tags, metadata,
                 created_at, updated_at, access_count, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
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
        self._trim_if_needed()
        return entry

    def retrieve(self, query: str = "", limit: int = 10) -> List[MemoryEntry]:
        terms = query.split() if query else []
        if not terms:
            rows = self._conn.execute(
                f"SELECT * FROM {self._table} ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            entries = [self._row_to_entry(row) for row in rows]
            for entry in entries:
                entry.touch()
            return entries

        # Build a scored result set in Python (cheap for bounded stores)
        candidates = self._fetch_all()
        scored = sorted(
            (
                (entry.relevance_score(terms), -entry.created_at, entry)
                for entry in candidates
            ),
            key=lambda triple: (triple[0], triple[1]),
            reverse=True,
        )
        results = [entry for _, _, entry in scored[:limit]]
        for entry in results:
            entry.touch()
            self._record_access(entry.id)
        return results

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
            f"UPDATE {self._table} SET {set_clause} WHERE id = ?",
            values,
        )
        self._conn.commit()
        if cursor.rowcount == 0:
            return None
        return self._fetch(entry_id)

    def delete(self, entry_id: str) -> bool:
        cursor = self._conn.execute(
            f"DELETE FROM {self._table} WHERE id = ?", (entry_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def clear(self) -> int:
        count = self.size()
        self._conn.execute(f"DELETE FROM {self._table}")
        self._conn.commit()
        return count

    def size(self) -> int:
        row = self._conn.execute(
            f"SELECT COUNT(*) FROM {self._table}"
        ).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _fetch(self, entry_id: str) -> Optional[MemoryEntry]:
        row = self._conn.execute(
            f"SELECT * FROM {self._table} WHERE id = ?", (entry_id,)
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def _fetch_all(self) -> List[MemoryEntry]:
        rows = self._conn.execute(f"SELECT * FROM {self._table}").fetchall()
        return [self._row_to_entry(row) for row in rows]

    def _record_access(self, entry_id: str) -> None:
        self._conn.execute(
            f"""
            UPDATE {self._table}
            SET access_count = access_count + 1, last_accessed = ?
            WHERE id = ?
            """,
            (time.time(), entry_id),
        )
        self._conn.commit()

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
        if total <= self._max_entries:
            return
        excess = total - self._max_entries
        self._conn.execute(
            f"""
            DELETE FROM {self._table} WHERE id IN (
                SELECT id FROM {self._table}
                ORDER BY importance ASC, last_accessed ASC
                LIMIT ?
            )
            """,
            (excess,),
        )
        self._conn.commit()
