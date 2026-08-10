"""Parsing helpers for the Python Brain.

Parsing is done with the standard-library ``ast`` module only.  No code is
ever executed here; the goal is safe, offline structural analysis.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .model import SyntaxIssue


class PythonParseError(Exception):
    """Raised when source code cannot be parsed as Python."""

    def __init__(
        self,
        message: str,
        line: int = 0,
        column: int = 0,
        text: str | None = None,
        filename: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column
        self.text = text
        self.filename = filename

    def __str__(self) -> str:
        location = f"{self.filename or '<source>'}:{self.line}:{self.column}"
        return f"{location}: {self.message}"


class PythonParser:
    """Parse Python source into an AST without executing it."""

    @staticmethod
    def parse(source: str) -> tuple[ast.Module | None, SyntaxIssue | None]:
        """Return ``(tree, None)`` on success or ``(None, SyntaxIssue)`` on failure."""
        try:
            tree = ast.parse(source, mode="exec")
            return tree, None
        except SyntaxError as exc:
            return None, SyntaxIssue(
                line=exc.lineno or 0,
                column=exc.offset or 0,
                message=exc.msg or "Invalid syntax",
                text=(exc.text or "").rstrip("\n"),
                error_type=type(exc).__name__,
            )

    @staticmethod
    def parse_strict(source: str) -> ast.Module:
        """Parse and raise :class:`PythonParseError` on failure."""
        tree, error = PythonParser.parse(source)
        if tree is None or error is not None:
            raise PythonParseError(error.message, error.line, error.column, error.text)
        return tree

    @staticmethod
    def parse_file(path: str | Path) -> tuple[ast.Module | None, SyntaxIssue | None]:
        """Parse a Python file on disk (UTF-8)."""
        return PythonParser.parse(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def inspect(node: ast.AST) -> str:
        """Return a readable AST dump for inspection purposes."""
        return ast.dump(node, indent=2, include_attributes=False)


__all__ = ["PythonParseError", "PythonParser"]
