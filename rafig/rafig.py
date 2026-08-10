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
from .memory import MemoryManager
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
        self.memory: MemoryManager | None = None
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
        self.memory = self._init_memory()
        self.started = True
        self.logger.info("RAFIQ started successfully. Foundation modules are ready.")

    def run(self) -> None:
        """Run the main loop for the foundation phase."""
        if not self.started:
            raise RuntimeError("Rafiq has not been started")
        self.logger.info("RAFIQ is running in offline mode without external services.")
        if self.memory is not None:
            diag = self.memory.diagnostics()
            self.logger.info(
                "Memory system ready: %d conversation turns, %d episodic, %d semantic, %d project",
                diag.conversation_size,
                diag.episodic_size,
                diag.semantic_size,
                diag.project_size,
            )

    def shutdown(self) -> None:
        """Clean shutdown for the foundation phase."""
        if self.memory is not None:
            try:
                self.memory.close()
            except Exception as exc:  # pragma: no cover - defensive guard
                self.logger.warning("Memory close error: %s", exc)
            self.memory = None
        if self.started:
            self.logger.info("Shutting down RAFIQ cleanly.")
        self.started = False

    def _init_memory(self) -> MemoryManager:
        mem = self.settings.memory
        db_path = self.paths["memory"] / mem.db_filename
        return MemoryManager(
            db_path=db_path,
            working_capacity=mem.working_capacity,
            conversation_max=mem.conversation_max,
            episodic_max=mem.episodic_max,
            semantic_max=mem.semantic_max,
            project_max=mem.project_max,
        )

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
