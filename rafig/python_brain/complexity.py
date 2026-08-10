"""Basic complexity analysis for the Python Brain.

Computes cyclomatic complexity (decision points + 1), maximum nesting
depth, and statement counts for modules and functions, using the AST only.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(slots=True)
class ComplexityInfo:
    """Complexity measurement for a module or function."""

    cyclomatic: int = 1
    nesting_depth: int = 0
    statement_count: int = 0
    lines_of_code: int = 0
    label: str = "simple"

    @classmethod
    def label_for(cls, cyclomatic: int) -> str:
        if cyclomatic <= 4:
            return "simple"
        if cyclomatic <= 8:
            return "moderate"
        if cyclomatic <= 15:
            return "complex"
        return "very complex"


_CONTROL_KINDS = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.TryStar,
    ast.Match,
)


class _DecisionCounter(ast.NodeVisitor):
    """Counts cyclomatic decision points in a subtree."""

    def __init__(self) -> None:
        self.decisions = 0

    def visit_If(self, node: ast.If) -> None:
        self.decisions += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.decisions += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.decisions += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.decisions += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.decisions += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.decisions += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.decisions += len(node.values) - 1
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self.decisions += len(node.cases)
        self.generic_visit(node)

    def _count_comprehension(self, node: ast.AST) -> None:
        generators = getattr(node, "generators", [])
        self.decisions += len(generators)
        for gen in generators:
            self.decisions += len(gen.ifs)
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._count_comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._count_comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._count_comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._count_comprehension(node)


def _measure_structure(stmts: List[ast.stmt], depth: int = 0) -> Tuple[int, int]:
    """Return ``(max_nesting_depth, statement_count)`` for a statement list.

    Function and class bodies restart nesting at depth 0 because they are
    independent units.
    """
    max_depth = depth
    count = 0
    for stmt in stmts:
        count += 1
        if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            nested, counted = _measure_structure(list(stmt.body), depth + 1)
            max_depth = max(max_depth, nested)
            count += counted
            if isinstance(stmt, ast.If) and len(stmt.orelse) == 1 and isinstance(
                stmt.orelse[0], ast.If
            ):
                # elif chains stay at the same nesting depth as the if
                nested, counted = _measure_structure(list(stmt.orelse), depth)
            else:
                nested, counted = _measure_structure(list(stmt.orelse), depth + 1)
            max_depth = max(max_depth, nested)
            count += counted
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            nested, counted = _measure_structure(list(stmt.body), depth + 1)
            max_depth = max(max_depth, nested)
            count += counted
        elif isinstance(stmt, (ast.Try, ast.TryStar)):
            children = list(stmt.body) + list(stmt.orelse) + list(stmt.finalbody)
            for handler in stmt.handlers:
                children.extend(handler.body)
            nested, counted = _measure_structure(children, depth + 1)
            max_depth = max(max_depth, nested)
            count += counted
        elif isinstance(stmt, ast.Match):
            children = []
            for case in stmt.cases:
                children.extend(case.body)
            nested, counted = _measure_structure(children, depth + 1)
            max_depth = max(max_depth, nested)
            count += counted
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _, counted = _measure_structure(stmt.body, 0)
            count += counted
    return max_depth, count


def measure_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Tuple[int, int, int]:
    """Return ``(cyclomatic, nesting_depth, statement_count)`` for a function."""
    counter = _DecisionCounter()
    counter.visit(node)
    nesting, statements = _measure_structure(list(node.body))
    return counter.decisions + 1, nesting, statements


def measure_module(tree: ast.Module) -> ComplexityInfo:
    """Measure the whole module."""
    counter = _DecisionCounter()
    counter.visit(tree)
    nesting, statements = _measure_structure(list(tree.body))
    cyclomatic = counter.decisions + 1
    return ComplexityInfo(
        cyclomatic=cyclomatic,
        nesting_depth=nesting,
        statement_count=statements,
        lines_of_code=_lines_of_code(tree),
        label=ComplexityInfo.label_for(cyclomatic),
    )


def _lines_of_code(tree: ast.Module) -> int:
    last = 0
    for node in ast.walk(tree):
        last = max(last, getattr(node, "end_lineno", 0) or 0)
    return last


__all__ = ["ComplexityInfo", "measure_function", "measure_module"]
