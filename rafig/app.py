"""Application entry point for RAFIQ."""

from __future__ import annotations

from typing import Sequence

from .rafig import Rafiq


def main(argv: Sequence[str] | None = None) -> int:
    """Run the RAFIQ foundation application."""
    del argv
    engine = Rafiq()
    try:
        engine.start()
        engine.run()
        return 0
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"RAFIQ failed to start: {exc}")
        return 1
    finally:
        engine.shutdown()
