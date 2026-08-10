"""Structured data models for the RAFIQ Python Brain.

These dataclasses describe the structure of a Python program: functions,
classes, variables, imports, scopes, symbol tables, control flow, loops,
exception handling, calls, returns, and program structure.  They are
produced by :class:`~rafig.python_brain.analyzer.PythonAnalyzer` and are
independent of the exact source text so later phases (code generation,
repair, verification) can consume them.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional

MODULE_SCOPE = "<module>"
BUILTIN_SCOPE = "<builtins>"


def _default_builtins() -> frozenset[str]:
    import builtins

    return frozenset(dir(builtins))


# ---------------------------------------------------------------------------
# Syntax errors
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SyntaxIssue:
    """Details of a Python syntax error captured without executing code."""

    line: int
    column: int
    message: str
    text: str | None = None
    error_type: str = "SyntaxError"

    def __str__(self) -> str:
        return f"line {self.line}, column {self.column}: {self.message}"


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ParameterInfo:
    """A single function parameter."""

    name: str
    kind: str = "positional"  # positional | positional_only | keyword_only | varargs | varkw
    has_default: bool = False
    default: str | None = None
    annotation: str | None = None
    line: int = 0


@dataclass(slots=True)
class FunctionInfo:
    """Structured representation of a function definition."""

    name: str
    line: int
    end_line: int
    scope_name: str = ""
    parameters: List[ParameterInfo] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False
    is_method: bool = False
    class_name: str | None = None
    docstring: str | None = None
    return_annotation: str | None = None
    returns: List[str] = field(default_factory=list)  # inferred return types
    has_explicit_return: bool = False
    calls: List["CallInfo"] = field(default_factory=list)
    local_variables: List["VariableInfo"] = field(default_factory=list)
    raises: List[str] = field(default_factory=list)  # exception names raised
    complexity: int = 1
    nesting_depth: int = 0
    statement_count: int = 0

    @property
    def signature(self) -> str:
        parts: List[str] = []
        for param in self.parameters:
            text = param.name
            if param.annotation:
                text = f"{text}: {param.annotation}"
            if param.has_default and param.default is not None:
                text = f"{text} = {param.default}"
            parts.append(text)
        return f"{self.name}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ClassInfo:
    """Structured representation of a class definition."""

    name: str
    line: int
    end_line: int
    scope_name: str = ""
    bases: List[str] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)
    docstring: str | None = None
    is_dataclass: bool = False
    methods: List[FunctionInfo] = field(default_factory=list)
    class_variables: List["VariableInfo"] = field(default_factory=list)
    instance_variables: List["VariableInfo"] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VariableInfo:
    """A variable binding discovered during analysis."""

    name: str
    kind: str = "variable"  # variable | parameter | global | import | loop_var
    #                        | with_target | exception_var | comprehension
    #                        | class | instance
    scope: str = MODULE_SCOPE
    defined_line: int = 0
    assigned_lines: List[int] = field(default_factory=list)
    used_lines: List[int] = field(default_factory=list)
    inferred_type: str | None = None
    class_name: str | None = None

    @property
    def is_used(self) -> bool:
        return bool(self.used_lines)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ImportInfo:
    """An import statement."""

    module: str
    names: List[str] = field(default_factory=list)
    asname: str | None = None
    lineno: int = 0
    is_from: bool = False
    is_star: bool = False
    level: int = 0


# ---------------------------------------------------------------------------
# Calls and returns
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CallInfo:
    """A function/method call site."""

    name: str
    lineno: int = 0
    args_count: int = 0
    keyword_args: List[str] = field(default_factory=list)
    is_method_call: bool = False
    receiver: str | None = None
    scope: str = MODULE_SCOPE


@dataclass(slots=True)
class ReturnInfo:
    """A return statement."""

    lineno: int = 0
    value: str | None = None
    value_type: str | None = None
    scope: str = MODULE_SCOPE


# ---------------------------------------------------------------------------
# Scopes and symbol tables
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ScopeInfo:
    """A lexical scope (module, function, class, lambda, comprehension)."""

    name: str
    kind: str  # module | function | class | lambda | comprehension
    line: int = 1
    parent: str | None = None
    globals_declared: List[str] = field(default_factory=list)
    nonlocals_declared: List[str] = field(default_factory=list)


@dataclass(slots=True)
class SymbolEntry:
    """A symbol definition in a scope."""

    name: str
    kind: str  # function | class | variable | parameter | import
    #           | loop_var | with_target | exception_var | comprehension | builtin
    scope: str = MODULE_SCOPE
    line: int = 0
    type_name: str | None = None


@dataclass(slots=True)
class SymbolReference:
    """A use of a symbol in a scope."""

    name: str
    scope: str = MODULE_SCOPE
    line: int = 0
    column: int = 0
    context: str = "load"  # load | store


class SymbolTable:
    """A lightweight symbol table with Python-like scope-chain resolution."""

    def __init__(self, builtins: frozenset[str] | None = None) -> None:
        self.scopes: Dict[str, ScopeInfo] = {}
        self.entries: Dict[str, List[SymbolEntry]] = {}
        self.references: Dict[str, List[SymbolReference]] = {}
        self.builtins = builtins if builtins is not None else _default_builtins()

    def add_scope(self, scope: ScopeInfo) -> None:
        self.scopes[scope.name] = scope
        self.entries.setdefault(scope.name, [])
        self.references.setdefault(scope.name, [])

    def define(self, entry: SymbolEntry) -> None:
        self.entries.setdefault(entry.scope, []).append(entry)

    def reference(self, ref: SymbolReference) -> None:
        self.references.setdefault(ref.scope, []).append(ref)

    def find_in_scope(self, name: str, scope: str) -> SymbolEntry | None:
        for entry in self.entries.get(scope, []):
            if entry.name == name:
                return entry
        return None

    def scope_chain(self, scope: str) -> List[str]:
        """Return the scope chain from ``scope`` up to the module scope."""
        chain: List[str] = []
        current: str | None = scope
        while current is not None:
            chain.append(current)
            scope_info = self.scopes.get(current)
            current = scope_info.parent if scope_info else None
        return chain

    def resolve(self, name: str, scope: str) -> SymbolEntry | None:
        """Resolve ``name`` from ``scope`` following Python scoping rules.

        Handles ``global`` and ``nonlocal`` declarations, closure lookup,
        and finally the builtins namespace.
        """
        current: str | None = scope
        visited: set[str] = set()
        while current is not None and current not in visited:
            visited.add(current)
            scope_info = self.scopes.get(current)
            if scope_info is None:
                break
            if name in scope_info.globals_declared and current != MODULE_SCOPE:
                current = MODULE_SCOPE
                continue
            if name in scope_info.nonlocals_declared:
                current = self._nearest_enclosing_function(scope_info.parent)
                continue
            entry = self.find_in_scope(name, current)
            if entry is not None:
                return entry
            current = scope_info.parent
        if name in self.builtins:
            return SymbolEntry(name=name, kind="builtin", scope=BUILTIN_SCOPE, line=0)
        return None

    def _nearest_enclosing_function(self, scope: str | None) -> str | None:
        current = self.scopes.get(scope) if scope else None
        while current is not None:
            if current.kind == "function":
                return current.name
            current = self.scopes.get(current.parent or "") if current.parent else None
        return None

    def is_defined(self, name: str, scope: str) -> bool:
        return self.resolve(name, scope) is not None

    def names_defined(self, scope: str) -> List[str]:
        return [entry.name for entry in self.entries.get(scope, [])]


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ControlFlowInfo:
    """A control-flow statement (if/for/while/with/try/match/return/...)."""

    kind: str
    lineno: int = 0
    end_lineno: int = 0
    condition: str | None = None
    branches: int = 0
    scope: str = MODULE_SCOPE


@dataclass(slots=True)
class LoopInfo:
    """A for or while loop."""

    kind: str  # for | while
    lineno: int = 0
    end_lineno: int = 0
    iterable: str | None = None
    variables: List[str] = field(default_factory=list)
    scope: str = MODULE_SCOPE
    has_break: bool = False
    has_continue: bool = False
    has_else: bool = False
    body_size: int = 0


@dataclass(slots=True)
class ExceptionHandlerInfo:
    """A try/except/finally block."""

    lineno: int = 0
    end_lineno: int = 0
    scope: str = MODULE_SCOPE
    handlers: List[tuple[str | None, str | None]] = field(default_factory=list)
    raises_in_try: List[str] = field(default_factory=list)
    has_else: bool = False
    has_finally: bool = False


# ---------------------------------------------------------------------------
# Program structure
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StatementSummary:
    """A summary of one top-level statement."""

    kind: str
    line: int = 0
    name: str | None = None


@dataclass(slots=True)
class ProgramStructure:
    """High-level structural facts about the analyzed module."""

    module_docstring: str | None = None
    lines_of_code: int = 0
    statement_counts: Dict[str, int] = field(default_factory=dict)
    top_level_statements: List[StatementSummary] = field(default_factory=list)
    modules_imported: List[str] = field(default_factory=list)
    has_main_guard: bool = False
    entry_points: List[str] = field(default_factory=list)
    total_functions: int = 0
    total_classes: int = 0
    total_imports: int = 0


# ---------------------------------------------------------------------------
# Internal record used while walking the AST
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _VarRecord:
    """Internal accumulation record for a variable binding."""

    name: str
    kind: str
    scope: str
    defined_line: int
    assigned_lines: List[int] = field(default_factory=list)
    used_lines: List[int] = field(default_factory=list)
    class_name: str | None = None


__all__ = [
    "BUILTIN_SCOPE",
    "MODULE_SCOPE",
    "CallInfo",
    "ClassInfo",
    "ControlFlowInfo",
    "ExceptionHandlerInfo",
    "FunctionInfo",
    "ImportInfo",
    "LoopInfo",
    "ParameterInfo",
    "ProgramStructure",
    "ReturnInfo",
    "ScopeInfo",
    "StatementSummary",
    "SymbolEntry",
    "SymbolReference",
    "SymbolTable",
    "SyntaxIssue",
    "VariableInfo",
]
