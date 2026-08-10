"""Configuration helpers for RAFIQ."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


PROJECT_NAME = "RAFIQ"
PROJECT_VERSION = "0.1.0"


@dataclass(slots=True)
class MemorySettings:
    """Tunables for the RAFIQ memory subsystem (Phase 05)."""

    working_capacity: int = 64
    conversation_max: int = 200
    episodic_max: int = 5000
    semantic_max: int = 5000
    project_max: int = 2000
    db_filename: str = "rafiq_memory.db"


@dataclass(slots=True)
class Settings:
    """Minimal runtime settings for the foundation phase."""

    project_name: str = PROJECT_NAME
    version: str = PROJECT_VERSION
    offline_mode: bool = True
    project_root: Path | None = None
    log_level: str = "INFO"
    memory: MemorySettings = field(default_factory=MemorySettings)

    def __post_init__(self) -> None:
        if self.project_root is None:
            self.project_root = Path(__file__).resolve().parent.parent


def get_settings() -> Settings:
    """Build settings from environment variables when available."""
    offline_mode = os.getenv("RAFIQ_OFFLINE_MODE", "true").lower() == "true"
    log_level = os.getenv("RAFIQ_LOG_LEVEL", "INFO")
    project_root = os.getenv("RAFIQ_PROJECT_ROOT")
    return Settings(
        offline_mode=offline_mode,
        log_level=log_level,
        project_root=Path(project_root).resolve() if project_root else None,
    )


def get_project_paths(settings: Settings | None = None) -> Dict[str, Path]:
    """Return important project directories and ensure they exist."""
    settings = settings or get_settings()
    root = settings.project_root
    assert root is not None
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "root": root,
        "logs": root / "logs",
        "data": root / "data",
        "memory": root / "data" / "memory",
        "tests": root / "tests",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
