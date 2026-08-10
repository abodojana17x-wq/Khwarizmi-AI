"""Core RAFIQ engine skeleton."""

from __future__ import annotations

import logging
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PROJECT_VERSION, Settings, get_settings
from .paths import ensure_project_directories


@dataclass(slots=True)
class DiagnosticReport:
    """Minimal runtime diagnostics for the foundation phase."""

    python_version: str
    operating_system: str
    project_version: str
    available_memory_mb: int | None
    offline_mode: bool


class Rafiq:
    """Minimal RAFIQ engine for phase 01."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.logger = self._build_logger()
        self.paths = ensure_project_directories(self.settings)
        self.started = False
        self._diagnostics: DiagnosticReport | None = None

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger("rafig")
        logger.setLevel(getattr(logging, self.settings.log_level.upper(), logging.INFO))
        logger.handlers.clear()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
        return logger

    def start(self) -> None:
        """Initialize the project and mark startup complete."""
        self.logger.info("Starting RAFIQ foundation...")
        self._diagnostics = self._collect_diagnostics()
        self.started = True
        self.logger.info("RAFIQ started successfully. Foundation modules are ready.")

    def run(self) -> None:
        """Run the main loop for the foundation phase."""
        if not self.started:
            raise RuntimeError("Rafiq has not been started")
        self.logger.info("RAFIQ is running in offline mode without external services.")

    def shutdown(self) -> None:
        """Clean shutdown for the foundation phase."""
        if self.started:
            self.logger.info("Shutting down RAFIQ cleanly.")
        self.started = False

    def _collect_diagnostics(self) -> DiagnosticReport:
        return DiagnosticReport(
            python_version=sys.version.split()[0],
            operating_system=platform.platform(),
            project_version=self.settings.version,
            available_memory_mb=self._detect_memory(),
            offline_mode=self.settings.offline_mode,
        )

    def _detect_memory(self) -> int | None:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        value = line.split()[1]
                        return int(int(value) / 1024)
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            return None
        return None

    @property
    def diagnostics(self) -> DiagnosticReport | None:
        return self._diagnostics
