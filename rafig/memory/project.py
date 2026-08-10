"""ProjectMemory — project-specific context.

Project memories describe the structure and conventions of the
project RAFIQ is currently working on: file layout, dependencies,
recent errors, known issues, etc.
"""

from __future__ import annotations

from pathlib import Path

from ._sqlite import SQLiteMemory


class ProjectMemory(SQLiteMemory):
    _table = "projects"
    _default_source = "project"
