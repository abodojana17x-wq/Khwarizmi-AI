"""Path helpers for RAFIQ."""

from __future__ import annotations

from pathlib import Path

from .config import Settings, get_project_paths


def ensure_project_directories(settings: Settings | None = None) -> dict[str, Path]:
    """Create the standard RAFIQ directories if needed."""
    paths = get_project_paths(settings)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
