"""Issue detection for the Python Brain.

Detectors implemented here:

* undefined names (with scope-aware resolution)
* use-before-assignment
* unreachable code
* suspicious constructs (mutable defaults, ``== None``, bare ``except``,
  shadowed builtins, infinite ``while True`` loops, duplicate dict keys,
  empty bodies, literal conditions, ...)
* unused variables / imports / parameters

Everything is computed from the AST; no regex-based analysis.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from .model import MODULE_SCOPE, SymbolTable, VariableInfo


class IssueSeverity:
    """Severity levels for analysis issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    ORDER = {ERROR: 0, WARNING: 1, INFO: 2}


@dataclass(slots=True)
class Issue:
    """A single problem found in the analyzed source."""

    kind: str
    severity: str
    line: int
    message: str
    column: int = 0
    name: str | None = None
    scope: str | None = None
    suggestion: str | None = None


# ---------------------------------------------------------------------------
# Undefined names & use-before-assignment
# ---------------------------------------------------------------------------


def detect_undefined_names(
    symbol_table: SymbolTable,
    usage_events: Dict[Tuple[str, str], List[Tuple[int, str]]],
    star_import: bool,
) -> List[Issue]:
    """Detect undefined names and use-before-assignment problems."""
    issues: List[Issue] = []

    if not star_import:
        for scope, refs in symbol_table.references.items():
            for ref in refs:
                if ref.context != "load":
                    continue
                if symbol_table.resolve(ref.name, ref.scope) is None:
                    issues.append(
                        Issue(
                            kind="undefined-name",
                            severity=IssueSeverity.ERROR,
                            line=ref.line,
                            column=ref.column,
                            message=f"Name '{ref.name}' is used but never defined.",
                            name=ref.name,
                            scope=ref.scope,
                            suggestion=f"Define '{ref.name}' or import it before use.",
                        )
                    )

    for (scope, name), events in usage_events.items():
        stores = [line for line, kind in events if kind == "store"]
        if not stores:
            continue
        scope_info = symbol_table.scopes.get(scope)
        if scope_info is None:
            continue
        if name in scope_info.globals_declared or name in scope_info.nonlocals_declared:
            continue
        first_store = min(stores)
        for line, kind in events:
            if kind == "load" and line < first_store:
                issues.append(
                    Issue(
                        kind="used-before-assignment",
                        severity=IssueSeverity.WARNING,
                        line=line,
                        message=(
                            f"Name '{name}' is used before it is assigned "
                            f"(first assignment on line {first_store})."
                        ),
                        name=name,
                        scope=scope,
                        suggestion=(
                            "Move the assignment before the first use, or remove the "
                            "local definition so the enclosing binding is used."
                        ),
                    )
                )
                break
    return issues


# ---------------------------------------------------------------------------
# Unreachable code
# ---------------------------------------------------------------------------


def detect_unreachable_code(tree: ast.Module) -> List[Issue]:
    """Find statements that can never execute (after return/raise/break/continue)."""
    issues: List[Issue] = []
    _scan_body(tree.body, in_loop=False, issues=issues)
    return issues


def _scan_body(
    body: List[ast.stmt],
    in_loop: bool,
    issues: List[Issue],
) -> None:
    reachable = True
    previous_kind = ""
    previous_line = 0
    for stmt in body:
        if not reachable:
            issues.append(
                Issue(
                    kind="unreachable-code",
                    severity=IssueSeverity.WARNING,
                    line=stmt.lineno,
                    message=(
                        f"Statement is unreachable: execution stops at "
                        f"'{previous_kind}' on line {previous_line}."
                    ),
                    suggestion="Remove the statement or restructure the control flow.",
                )
            )
            continue
        if isinstance(stmt, (ast.Return, ast.Raise)):
            reachable = False
            previous_kind, previous_line = stmt.__class__.__name__.lower(), stmt.lineno
        elif isinstance(stmt, (ast.Break, ast.Continue)) and in_loop:
            reachable = False
            previous_kind, previous_line = stmt.__class__.__name__.lower(), stmt.lineno
        _recurse_into(stmt, in_loop, issues)


def _recurse_into(stmt: ast.stmt, in_loop: bool, issues: List[Issue]) -> None:
    if isinstance(stmt, (ast.If,)):
        _scan_body(stmt.body, in_loop, issues)
        _scan_body(stmt.orelse, in_loop, issues)
    elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
        _scan_body(stmt.body, True, issues)
        _scan_body(stmt.orelse, in_loop, issues)
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        _scan_body(stmt.body, in_loop, issues)
    elif isinstance(stmt, (ast.Try, ast.TryStar)):
        _scan_body(stmt.body, in_loop, issues)
        for handler in stmt.handlers:
            _scan_body(handler.body, in_loop, issues)
        _scan_body(stmt.orelse, in_loop, issues)
        _scan_body(stmt.finalbody, in_loop, issues)
    elif isinstance(stmt, ast.Match):
        for case in stmt.cases:
            _scan_body(case.body, in_loop, issues)
    elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        _scan_body(stmt.body, False, issues)


# ---------------------------------------------------------------------------
# Suspicious constructs
# ---------------------------------------------------------------------------


def detect_suspicious_constructs(tree: ast.Module, builtins: frozenset[str]) -> List[Issue]:
    """Find common code smells and suspicious constructs."""
    detector = _SuspiciousDetector(builtins)
    detector.visit(tree)
    return detector.issues


class _SuspiciousDetector(ast.NodeVisitor):
    def __init__(self, builtins: frozenset[str]) -> None:
        self.builtins = builtins
        self.issues: List[Issue] = []
        self._scope_depth = 0

    # -- helpers ----------------------------------------------------------

    def _issue(self, kind: str, severity: str, node: ast.AST, message: str,
               suggestion: str | None = None) -> None:
        self.issues.append(
            Issue(
                kind=kind,
                severity=severity,
                line=node.lineno,
                column=getattr(node, "col_offset", 0),
                message=message,
                suggestion=suggestion,
            )
        )

    @staticmethod
    def _is_none(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and node.value is None

    @staticmethod
    def _is_literal_non_none(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and node.value is not None

    @staticmethod
    def _is_mutable_default(node: ast.AST) -> bool:
        if isinstance(node, (ast.List, ast.Dict, ast.Set)):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id in {"list", "dict", "set"}
        return False

    @staticmethod
    def _is_empty_body(body: List[ast.stmt]) -> bool:
        meaningful = [
            stmt
            for stmt in body
            if not (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            )
        ]
        return bool(meaningful) and all(isinstance(stmt, ast.Pass) for stmt in meaningful)

    @staticmethod
    def _body_has_exit(body: List[ast.stmt]) -> bool:
        """True if the body always eventually leaves the loop (break/return/raise)."""

        def walk(stmts: List[ast.stmt]) -> bool:
            for stmt in stmts:
                if isinstance(stmt, (ast.Break, ast.Return, ast.Raise)):
                    return True
                if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
                    # nested loops do not exit the outer loop via break/continue
                    if walk(stmt.body):
                        return True
                    continue
                for child in ast.iter_child_nodes(stmt):
                    if isinstance(child, ast.stmt) and walk([child]):
                        return True
            return False

        return walk(body)

    @staticmethod
    def _constant_bool(node: ast.AST) -> bool | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        return None

    # -- visitors -----------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_function(node)
        self._scope_depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self._scope_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_function(node)
        self._scope_depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self._scope_depth -= 1

    def _check_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        defaults = list(node.args.defaults) + [
            d for d in node.args.kw_defaults if d is not None
        ]
        for default in defaults:
            if self._is_mutable_default(default):
                self._issue(
                    "mutable-default-arg",
                    IssueSeverity.WARNING,
                    node,
                    f"Function '{node.name}' uses a mutable default argument, "
                    "which is shared across all calls.",
                    "Use None as the default and create the mutable inside the function.",
                )
        if node.name in self.builtins:
            self._issue(
                "shadowed-builtin",
                IssueSeverity.WARNING,
                node,
                f"Function '{node.name}' shadows a Python builtin of the same name.",
                "Rename the function to avoid surprising behaviour.",
            )
        if self._is_empty_body(node.body):
            self._issue(
                "empty-body",
                IssueSeverity.INFO,
                node,
                f"Function '{node.name}' has an empty body.",
            )
        # parameter shadowing (informational)
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            if arg.arg in self.builtins:
                self._issue(
                    "shadowed-builtin",
                    IssueSeverity.INFO,
                    arg,
                    f"Parameter '{arg.arg}' shadows a Python builtin.",
                    "Consider renaming the parameter.",
                )
        if node.args.vararg and node.args.vararg.arg in self.builtins:
            self._issue("shadowed-builtin", IssueSeverity.INFO, node.args.vararg,
                        f"Parameter '{node.args.vararg.arg}' shadows a Python builtin.")
        if node.args.kwarg and node.args.kwarg.arg in self.builtins:
            self._issue("shadowed-builtin", IssueSeverity.INFO, node.args.kwarg,
                        f"Parameter '{node.args.kwarg.arg}' shadows a Python builtin.")
        # also scan the signature for suspicious constructs
        for deco in node.decorator_list:
            self.visit(deco)
        if node.returns is not None:
            self.visit(node.returns)
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            if arg.annotation is not None:
                self.visit(arg.annotation)
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._scope_depth += 1
        self.visit(node.body)
        self._scope_depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name in self.builtins:
            self._issue(
                "shadowed-builtin",
                IssueSeverity.WARNING,
                node,
                f"Class '{node.name}' shadows a Python builtin of the same name.",
                "Rename the class to avoid surprising behaviour.",
            )
        if self._is_empty_body(node.body):
            self._issue(
                "empty-body",
                IssueSeverity.INFO,
                node,
                f"Class '{node.name}' has an empty body.",
            )
        self._scope_depth += 1
        for deco in node.decorator_list:
            self.visit(deco)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        for stmt in node.body:
            self.visit(stmt)
        self._scope_depth -= 1

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._scope_depth == 0:
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in self.builtins:
                    self._issue(
                        "shadowed-builtin",
                        IssueSeverity.WARNING,
                        node,
                        f"Variable '{target.id}' shadows a Python builtin at module level.",
                        "Rename the variable.",
                    )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self._scope_depth == 0 and isinstance(node.target, ast.Name) \
                and node.target.id in self.builtins:
            self._issue(
                "shadowed-builtin",
                IssueSeverity.WARNING,
                node,
                f"Variable '{node.target.id}' shadows a Python builtin at module level.",
                "Rename the variable.",
            )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            if name in self.builtins:
                self._issue(
                    "shadowed-builtin",
                    IssueSeverity.WARNING,
                    node,
                    f"Import '{name}' shadows a Python builtin.",
                    "Use an alias, e.g. 'import x as y'.",
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                continue
            name = alias.asname or alias.name
            if name in self.builtins:
                self._issue(
                    "shadowed-builtin",
                    IssueSeverity.WARNING,
                    node,
                    f"Import '{name}' shadows a Python builtin.",
                    "Use an alias, e.g. 'from x import y as z'.",
                )
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op, right in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Eq, ast.NotEq)):
                if self._is_none(node.left) or self._is_none(right):
                    self._issue(
                        "none-comparison",
                        IssueSeverity.WARNING,
                        node,
                        "Comparing with None using '==' or '!='.",
                        "Use 'is None' / 'is not None' instead.",
                    )
                elif (
                    isinstance(node.left, ast.Constant) and isinstance(node.left.value, bool)
                ) or (isinstance(right, ast.Constant) and isinstance(right.value, bool)):
                    self._issue(
                        "bool-comparison",
                        IssueSeverity.INFO,
                        node,
                        "Comparing a value against True/False with '=='.",
                        "Use 'if value:' or 'if value is True:' instead.",
                    )
            elif isinstance(op, (ast.Is, ast.IsNot)):
                if self._is_literal_non_none(node.left) or self._is_literal_non_none(right):
                    self._issue(
                        "identity-comparison",
                        IssueSeverity.WARNING,
                        node,
                        "Using 'is' to compare against a non-None literal.",
                        "Use '==' for value comparison or 'is None' for None.",
                    )
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self._issue(
                "bare-except",
                IssueSeverity.WARNING,
                node,
                "Bare 'except:' catches every exception, including KeyboardInterrupt.",
                "Catch specific exception types instead.",
            )
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            self._issue(
                "exception-swallowed",
                IssueSeverity.WARNING,
                node,
                "Exception handler swallows the exception silently (pass-only body).",
                "Log or re-raise the exception instead of ignoring it.",
            )
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        value = self._constant_bool(node.test)
        if value is True:
            if not self._body_has_exit(node.body):
                self._issue(
                    "infinite-loop",
                    IssueSeverity.WARNING,
                    node,
                    "'while True:' loop has no break, return, or raise; it may run forever.",
                    "Add an exit condition or a break statement.",
                )
        elif value is False:
            self._issue(
                "literal-condition",
                IssueSeverity.INFO,
                node,
                "'while False:' body never executes.",
                "Remove the loop or fix the condition.",
            )
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        value = self._constant_bool(node.test)
        if value is True:
            self._issue(
                "literal-condition",
                IssueSeverity.INFO,
                node,
                "Condition is always True.",
                "Remove the condition or use the body directly.",
            )
        elif value is False:
            self._issue(
                "literal-condition",
                IssueSeverity.INFO,
                node,
                "Condition is always False; its body never executes.",
            )
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        seen: Dict[object, int] = {}
        for key in node.keys:
            if key is None:
                continue
            if isinstance(key, ast.Constant) and isinstance(
                key.value, (str, int, float, bool)
            ):
                if key.value in seen:
                    self._issue(
                        "duplicate-dict-key",
                        IssueSeverity.WARNING,
                        node,
                        f"Duplicate literal key {key.value!r} in dict literal "
                        f"(first occurrence on line {seen[key.value]}).",
                        "Remove the duplicate key.",
                    )
                else:
                    seen[key.value] = key.lineno
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Unused symbols
# ---------------------------------------------------------------------------


def detect_unused_symbols(variables: Iterable[VariableInfo]) -> List[Issue]:
    """Flag unused variables, imports, loop variables, and parameters."""
    issues: List[Issue] = []
    for var in variables:
        if var.used_lines:
            continue
        if var.name.startswith("_"):
            continue
        if var.kind in {"class", "instance", "future_import"}:
            continue
        if var.kind == "parameter":
            issues.append(
                Issue(
                    kind="unused-parameter",
                    severity=IssueSeverity.INFO,
                    line=var.defined_line,
                    message=f"Parameter '{var.name}' is never used.",
                    name=var.name,
                    scope=var.scope,
                )
            )
        elif var.kind in {"variable", "import", "loop_var"}:
            issues.append(
                Issue(
                    kind="unused-variable",
                    severity=IssueSeverity.WARNING,
                    line=var.defined_line,
                    message=(
                        f"{'Import' if var.kind == 'import' else 'Variable'} "
                        f"'{var.name}' is defined but never used."
                    ),
                    name=var.name,
                    scope=var.scope,
                    suggestion=(
                        "Remove it, or use it, or rename it with a leading underscore "
                        "to mark it intentionally unused."
                    ),
                )
            )
        # class / instance / with_target / exception_var / comprehension
        # variables are intentionally excluded to avoid noisy reports.
        # Class variables (including dataclass fields) are often public API.
    return issues


__all__ = ["Issue", "IssueSeverity", "detect_suspicious_constructs",
           "detect_undefined_names", "detect_unreachable_code", "detect_unused_symbols"]
