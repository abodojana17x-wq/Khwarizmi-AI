"""Core analyzer for the RAFIQ Python Brain.

The analyzer parses Python source with the standard-library ``ast`` module
and produces a structured, language-independent model of the program:
functions, classes, variables, imports, scopes, a symbol table, control
flow, loops, exception handling, calls, returns, type information,
complexity, and a list of detected issues.

No code is executed and no external tools are used.
"""

from __future__ import annotations

import ast
import builtins
import platform
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .complexity import ComplexityInfo, measure_function, measure_module
from .explain import PythonExplainer
from .issues import (
    Issue,
    IssueSeverity,
    detect_suspicious_constructs,
    detect_undefined_names,
    detect_unreachable_code,
    detect_unused_symbols,
)
from .model import (
    MODULE_SCOPE,
    CallInfo,
    ClassInfo,
    ControlFlowInfo,
    ExceptionHandlerInfo,
    FunctionInfo,
    ImportInfo,
    LoopInfo,
    ParameterInfo,
    ProgramStructure,
    ReturnInfo,
    ScopeInfo,
    StatementSummary,
    SymbolEntry,
    SymbolReference,
    SymbolTable,
    SyntaxIssue,
    VariableInfo,
    _VarRecord,
)
from .parser import PythonParser
from .types import TypeInfo, TypeInference


# ---------------------------------------------------------------------------
# Analysis result
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AnalysisResult:
    """The full structured model produced by analyzing a Python program."""

    source: str
    parse_successful: bool
    syntax_error: SyntaxIssue | None = None
    tree: ast.Module | None = None
    structure: ProgramStructure | None = None
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    variables: List[VariableInfo] = field(default_factory=list)
    scopes: List[ScopeInfo] = field(default_factory=list)
    symbol_table: SymbolTable | None = None
    control_flow: List[ControlFlowInfo] = field(default_factory=list)
    loops: List[LoopInfo] = field(default_factory=list)
    exception_handlers: List[ExceptionHandlerInfo] = field(default_factory=list)
    calls: List[CallInfo] = field(default_factory=list)
    returns: List[ReturnInfo] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    complexity: ComplexityInfo | None = None

    # -- helpers ----------------------------------------------------------

    @property
    def is_valid(self) -> bool:
        """True when the code parses and has no error-severity issues."""
        return self.parse_successful and not self.error_issues

    @property
    def error_issues(self) -> List[Issue]:
        return [issue for issue in self.issues if issue.severity == IssueSeverity.ERROR]

    @property
    def warning_issues(self) -> List[Issue]:
        return [issue for issue in self.issues if issue.severity == IssueSeverity.WARNING]

    @property
    def info_issues(self) -> List[Issue]:
        return [issue for issue in self.issues if issue.severity == IssueSeverity.INFO]

    def issues_of_kind(self, kind: str) -> List[Issue]:
        return [issue for issue in self.issues if issue.kind == kind]

    def ast_summary(self) -> str:
        """Return a readable AST dump for inspection."""
        if self.tree is None:
            return ""
        return ast.dump(self.tree, indent=2, include_attributes=False)

    def diagnostics(self) -> Dict[str, object]:
        """Return a compact diagnostics report for this analysis."""
        counts: Dict[str, int] = {"error": 0, "warning": 0, "info": 0}
        for issue in self.issues:
            counts[issue.severity] = counts.get(issue.severity, 0) + 1
        return {
            "parse_successful": self.parse_successful,
            "python_version": platform.python_version(),
            "lines_of_code": self.structure.lines_of_code if self.structure else 0,
            "functions": len(self.functions),
            "classes": len(self.classes),
            "imports": len(self.imports),
            "variables": len(self.variables),
            "calls": len(self.calls),
            "control_flow_nodes": len(self.control_flow),
            "loops": len(self.loops),
            "exception_handlers": len(self.exception_handlers),
            "returns": len(self.returns),
            "cyclomatic_complexity": self.complexity.cyclomatic if self.complexity else 0,
            "undefined_names": [
                issue.name for issue in self.issues if issue.kind == "undefined-name"
            ],
            "issues": counts,
        }


# ---------------------------------------------------------------------------
# The AST walker
# ---------------------------------------------------------------------------


class _CodeWalker(ast.NodeVisitor):
    """Single-pass walker that extracts a structured model of a module."""

    def __init__(self, tree: ast.Module, source: str) -> None:
        self.tree = tree
        self.source = source
        self.builtin_names = frozenset(dir(builtins))

        # collected output
        self.imports: List[ImportInfo] = []
        self.functions: List[FunctionInfo] = []
        self.classes: List[ClassInfo] = []
        self.variables: List[VariableInfo] = []
        self.scopes: List[ScopeInfo] = []
        self.control_flow: List[ControlFlowInfo] = []
        self.loops: List[LoopInfo] = []
        self.exception_handlers: List[ExceptionHandlerInfo] = []
        self.calls: List[CallInfo] = []
        self.returns: List[ReturnInfo] = []
        self.structure = self._build_structure(tree)
        self.symbol_table = SymbolTable(self.builtin_names)
        self.usage_events: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
        self.inference = TypeInference()
        self.star_import = False
        self.function_nodes: Dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self._return_value_nodes: List[Optional[ast.AST]] = []
        self._instance_values: Dict[Tuple[str, str], List[Tuple[int, ast.AST, str]]] = {}

        # runtime state
        self._scope_stack: List[ScopeInfo] = []
        self._function_stack: List[FunctionInfo] = []
        self._class_stack: List[ClassInfo] = []
        self._loop_stack: List[LoopInfo] = []
        self._var_records: Dict[Tuple[str, str], _VarRecord] = {}

        module_scope = ScopeInfo(MODULE_SCOPE, "module", 1, parent=None)
        self._scope_stack.append(module_scope)
        self._register_scope(module_scope)

    # -- public -----------------------------------------------------------

    def run(self) -> None:
        self.visit(self.tree)

    def finalize(self) -> None:
        self._finalize_usage()
        self._finalize_variables()
        for cls in self.classes:
            cls.methods = [fn for fn in self.functions if fn.class_name == cls.name]
            cls.class_variables = [
                var
                for var in self.variables
                if var.kind == "class" and var.scope == cls.scope_name
            ]
            cls.instance_variables = [
                var
                for var in self.variables
                if var.kind == "instance" and var.class_name == cls.name
            ]
        for fn in self.functions:
            fn.local_variables = [
                var for var in self.variables if var.scope == fn.scope_name
            ]

    # -- scope helpers ------------------------------------------------------

    def _register_scope(self, scope: ScopeInfo) -> None:
        self.scopes.append(scope)
        self.symbol_table.add_scope(scope)
        self.inference.register_scope(scope.name, scope.parent, scope.kind)

    @property
    def _current_scope(self) -> ScopeInfo:
        return self._scope_stack[-1]

    def _effective_scope_for(self, name: str) -> str:
        """Map a name to the scope where its binding actually lives."""
        scope = self._current_scope
        if name in scope.globals_declared:
            return MODULE_SCOPE
        if name in scope.nonlocals_declared:
            for outer in reversed(self._scope_stack[:-1]):
                if outer.kind == "function":
                    return outer.name
        return scope.name

    def _define_symbol(self, name: str, kind: str, line: int) -> None:
        scope = self._effective_scope_for(name)
        self.symbol_table.define(
            SymbolEntry(name=name, kind=kind, scope=scope, line=line)
        )

    def _assign_variable(
        self,
        name: str,
        kind: str,
        line: int,
        class_name: str | None = None,
    ) -> None:
        scope = self._effective_scope_for(name)
        self._define_symbol(name, kind, line)
        self._usage_events_append(scope, name, line, "store")
        key = (scope, name)
        record = self._var_records.get(key)
        if record is None:
            record = _VarRecord(
                name=name, kind=kind, scope=scope, defined_line=line, class_name=class_name
            )
            self._var_records[key] = record
        record.assigned_lines.append(line)

    def _assign_instance_var(self, attr: str, line: int, class_name: str) -> None:
        if not self._class_stack:
            return
        cls = self._class_stack[-1]
        key = (cls.scope_name, attr)
        record = self._var_records.get(key)
        if record is None:
            record = _VarRecord(
                name=attr,
                kind="instance",
                scope=cls.scope_name,
                defined_line=line,
                class_name=class_name,
            )
            self._var_records[key] = record
        record.assigned_lines.append(line)

    def _record_instance_value(self, attr: str, line: int, value: ast.AST) -> None:
        if not self._class_stack:
            return
        cls = self._class_stack[-1]
        self._instance_values.setdefault((cls.scope_name, attr), []).append(
            (line, value, self._current_scope.name)
        )

    def _reference(self, name: str, line: int, column: int) -> None:
        self.symbol_table.reference(
            SymbolReference(name=name, scope=self._current_scope.name, line=line,
                            column=column, context="load")
        )
        self._usage_events_append(self._current_scope.name, name, line, "load")

    def _usage_events_append(self, scope: str, name: str, line: int, kind: str) -> None:
        self.usage_events.setdefault((scope, name), []).append((line, kind))

    def _assign_targets(self, target: ast.AST, line: int, kind: str = "variable") -> None:
        if isinstance(target, ast.Name):
            self._assign_variable(target.id, kind, line)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._assign_targets(elt, line, kind)
        elif isinstance(target, ast.Starred):
            self._assign_targets(target.value, line, kind)

    def _register_assignment_for_targets(
        self, target: ast.AST, value: ast.AST | None, line: int
    ) -> None:
        if isinstance(target, ast.Name):
            scope = self._effective_scope_for(target.id)
            self.inference.register_assignment(scope, target.id, line, value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._register_assignment_for_targets(elt, value, line)
        elif isinstance(target, ast.Starred):
            self._register_assignment_for_targets(target.value, value, line)

    def _is_instance_target(self, target: ast.AST) -> bool:
        return (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id in {"self", "cls"}
            and bool(self._function_stack)
            and self._function_stack[-1].is_method
        )

    def _add_parameter(self, name: str, line: int, annotation: str | None) -> None:
        scope = self._current_scope.name
        self._define_symbol(name, "parameter", line)
        self._usage_events_append(scope, name, line, "store")
        key = (scope, name)
        if key not in self._var_records:
            self._var_records[key] = _VarRecord(
                name=name, kind="parameter", scope=scope, defined_line=line
            )
        if annotation:
            self.inference.register_annotation(scope, name, annotation)
        else:
            self.inference.register_parameter(scope, name)

    def _finalize_usage(self) -> None:
        """Mark variables as used by resolving references through scope chains."""
        for scope, refs in self.symbol_table.references.items():
            for ref in refs:
                if ref.context != "load":
                    continue
                for chain_scope in self.symbol_table.scope_chain(ref.scope):
                    if self.symbol_table.find_in_scope(ref.name, chain_scope) is not None:
                        record = self._var_records.get((chain_scope, ref.name))
                        if record is not None:
                            record.used_lines.append(ref.line)
                        break

    def _finalize_variables(self) -> None:
        for record in self._var_records.values():
            self.variables.append(
                VariableInfo(
                    name=record.name,
                    kind=record.kind,
                    scope=record.scope,
                    defined_line=record.defined_line,
                    assigned_lines=sorted(set(record.assigned_lines)),
                    used_lines=sorted(set(record.used_lines)),
                    class_name=record.class_name,
                )
            )

    # -- structure ---------------------------------------------------------

    def _build_structure(self, tree: ast.Module) -> ProgramStructure:
        top: List[StatementSummary] = []
        statement_counts: Dict[str, int] = {}
        for stmt in tree.body:
            top.append(self._summarize_statement(stmt))
        for node in ast.walk(tree):
            if isinstance(node, ast.stmt):
                name = node.__class__.__name__
                statement_counts[name] = statement_counts.get(name, 0) + 1

        modules_imported: List[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in modules_imported:
                        modules_imported.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = (("." * node.level) + (node.module or "")) or "<relative>"
                if module not in modules_imported:
                    modules_imported.append(module)

        has_main_guard = any(self._is_main_guard(stmt) for stmt in tree.body)
        entry_points: List[str] = []
        for stmt in tree.body:
            if self._is_main_guard(stmt):
                for inner in stmt.body:
                    if isinstance(inner, ast.Expr) and isinstance(inner.value, ast.Call):
                        entry_points.append(self._call_name(inner.value.func))
                    elif isinstance(inner, ast.Assign) and isinstance(inner.value, ast.Call):
                        entry_points.append(self._call_name(inner.value.func))

        lines_of_code = len(
            [line for line in self.source.splitlines() if line.strip()]
        )
        return ProgramStructure(
            module_docstring=ast.get_docstring(tree),
            lines_of_code=lines_of_code,
            statement_counts=statement_counts,
            top_level_statements=top,
            modules_imported=modules_imported,
            has_main_guard=has_main_guard,
            entry_points=entry_points,
            total_functions=0,
            total_classes=0,
            total_imports=0,
        )

    def _summarize_statement(self, stmt: ast.stmt) -> StatementSummary:
        name: str | None = None
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = stmt.name
        elif isinstance(stmt, ast.Import):
            name = ", ".join(alias.asname or alias.name.split(".")[0] for alias in stmt.names)
        elif isinstance(stmt, ast.ImportFrom):
            name = ", ".join(alias.asname or alias.name for alias in stmt.names)
        elif isinstance(stmt, ast.Assign):
            name = ", ".join(self._target_name(t) for t in stmt.targets)
        elif isinstance(stmt, ast.AnnAssign):
            name = self._target_name(stmt.target)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            name = self._call_name(stmt.value.func)
        return StatementSummary(kind=stmt.__class__.__name__, line=stmt.lineno, name=name)

    @staticmethod
    def _target_name(target: ast.AST) -> str:
        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, ast.Attribute):
            return f"{_CodeWalker._target_name(target.value)}.{target.attr}"
        if isinstance(target, ast.Subscript):
            return f"{_CodeWalker._target_name(target.value)}[...]"
        if isinstance(target, (ast.Tuple, ast.List)):
            return ", ".join(_CodeWalker._target_name(elt) for elt in target.elts)
        return "?"

    @staticmethod
    def _call_name(func: ast.AST) -> str:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return f"{_CodeWalker._call_name(func.value)}.{func.attr}"
        return "?"

    @staticmethod
    def _is_main_guard(stmt: ast.stmt) -> bool:
        return (
            isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Compare)
            and len(stmt.test.ops) == 1
            and isinstance(stmt.test.ops[0], ast.Eq)
            and isinstance(stmt.test.left, ast.Name)
            and stmt.test.left.id == "__name__"
            and len(stmt.test.comparators) == 1
            and isinstance(stmt.test.comparators[0], ast.Constant)
            and stmt.test.comparators[0].value == "__main__"
        )

    # -- module --------------------------------------------------------------

    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            self.visit(stmt)

    # -- functions & classes --------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, is_async=True)

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool
    ) -> None:
        parent_scope = self._current_scope
        parent_class = self._class_stack[-1] if self._class_stack else None
        is_method = parent_class is not None
        scope_name = (
            node.name
            if parent_scope.name == MODULE_SCOPE
            else f"{parent_scope.name}.{node.name}"
        )

        fn = FunctionInfo(
            name=node.name,
            line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            scope_name=scope_name,
            decorators=[ast.unparse(d) for d in node.decorator_list],
            is_async=is_async,
            is_method=is_method,
            class_name=parent_class.name if parent_class else None,
            docstring=ast.get_docstring(node),
            return_annotation=ast.unparse(node.returns) if node.returns else None,
        )
        parameters, defaults_map = self._collect_parameters(node)
        fn.parameters = parameters

        self._define_symbol(node.name, "function", node.lineno)
        self.inference.register_assignment(
            parent_scope.name, node.name, node.lineno, node
        )
        self.inference.register_function(scope_name, node.name)
        self.function_nodes[scope_name] = node

        # Signature parts are evaluated in the enclosing scope.
        for deco in node.decorator_list:
            self.visit(deco)
        if node.returns is not None:
            self.visit(node.returns)
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            if arg.annotation is not None:
                self.visit(arg.annotation)

        # Push the function scope.  Methods skip the class scope for name
        # resolution (Python closures do not see class bodies).
        if is_method:
            parent_for_scope = parent_scope.parent
        else:
            parent_for_scope = parent_scope.name
        scope = ScopeInfo(scope_name, "function", node.lineno, parent=parent_for_scope)
        self._register_scope(scope)
        self._scope_stack.append(scope)
        self._function_stack.append(fn)
        self.inference.register_enclosing_class(
            scope_name, parent_class.scope_name if parent_class else None
        )

        for param in parameters:
            self._add_parameter(param.name, node.lineno, param.annotation)
            default_node = defaults_map.get(param.name)
            if default_node is not None and not param.annotation:
                self.inference.register_assignment(
                    scope_name, param.name, node.lineno, default_node
                )

        for stmt in node.body:
            self.visit(stmt)

        self._scope_stack.pop()
        self._function_stack.pop()
        self.functions.append(fn)

    @staticmethod
    def _collect_parameters(
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
    ) -> Tuple[List[ParameterInfo], Dict[str, ast.AST]]:
        parameters: List[ParameterInfo] = []
        defaults_map: Dict[str, ast.AST] = {}

        positional = node.args.posonlyargs + node.args.args
        n_defaults = len(node.args.defaults)
        offset = len(positional) - n_defaults
        for index, arg in enumerate(positional):
            has_default = index >= offset
            default_node = node.args.defaults[index - offset] if has_default else None
            kind = "positional_only" if index < len(node.args.posonlyargs) else "positional"
            parameters.append(
                ParameterInfo(
                    name=arg.arg,
                    kind=kind,
                    has_default=has_default,
                    default=ast.unparse(default_node) if default_node else None,
                    annotation=ast.unparse(arg.annotation) if arg.annotation else None,
                    line=node.lineno,
                )
            )
            if default_node is not None:
                defaults_map[arg.arg] = default_node

        if node.args.vararg is not None:
            parameters.append(
                ParameterInfo(
                    name=node.args.vararg.arg,
                    kind="varargs",
                    annotation=(
                        ast.unparse(node.args.vararg.annotation)
                        if node.args.vararg.annotation
                        else None
                    ),
                    line=node.lineno,
                )
            )
        for index, arg in enumerate(node.args.kwonlyargs):
            default_node = (
                node.args.kw_defaults[index] if index < len(node.args.kw_defaults) else None
            )
            parameters.append(
                ParameterInfo(
                    name=arg.arg,
                    kind="keyword_only",
                    has_default=default_node is not None,
                    default=ast.unparse(default_node) if default_node else None,
                    annotation=ast.unparse(arg.annotation) if arg.annotation else None,
                    line=node.lineno,
                )
            )
            if default_node is not None:
                defaults_map[arg.arg] = default_node
        if node.args.kwarg is not None:
            parameters.append(
                ParameterInfo(
                    name=node.args.kwarg.arg,
                    kind="varkw",
                    annotation=(
                        ast.unparse(node.args.kwarg.annotation)
                        if node.args.kwarg.annotation
                        else None
                    ),
                    line=node.lineno,
                )
            )
        return parameters, defaults_map

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        parent_scope = self._current_scope
        scope_name = (
            node.name
            if parent_scope.name == MODULE_SCOPE
            else f"{parent_scope.name}.{node.name}"
        )
        decorators = [ast.unparse(d) for d in node.decorator_list]
        cls = ClassInfo(
            name=node.name,
            line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            scope_name=scope_name,
            bases=[ast.unparse(base) for base in node.bases],
            decorators=decorators,
            docstring=ast.get_docstring(node),
            is_dataclass=any(
                deco == "dataclass" or deco.endswith(".dataclass") for deco in decorators
            ),
        )
        self._define_symbol(node.name, "class", node.lineno)
        self.inference.register_assignment(parent_scope.name, node.name, node.lineno, node)

        # Bases, keywords, and decorators are evaluated in the enclosing scope.
        for deco in node.decorator_list:
            self.visit(deco)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)

        scope = ScopeInfo(scope_name, "class", node.lineno, parent=parent_scope.name)
        self._register_scope(scope)
        self._scope_stack.append(scope)
        self._class_stack.append(cls)

        for stmt in node.body:
            self.visit(stmt)

        self._scope_stack.pop()
        self._class_stack.pop()
        self.classes.append(cls)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        parent_scope = self._current_scope
        scope_name = f"{parent_scope.name}.<lambda>@{node.lineno}"
        scope = ScopeInfo(scope_name, "lambda", node.lineno, parent=parent_scope.name)
        self._register_scope(scope)
        self._scope_stack.append(scope)
        self.inference.register_enclosing_class(
            scope_name, self._class_stack[-1].scope_name if self._class_stack else None
        )
        for default in node.args.defaults:
            self.visit(default)
        parameters, _ = self._collect_parameters(node)
        for param in parameters:
            self._add_parameter(param.name, node.lineno, param.annotation)
        self.visit(node.body)
        self._scope_stack.pop()

    # -- comprehensions --------------------------------------------------------

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node, node.elt, None, None)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node, node.elt, None, None)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node, node.key, node.value, None)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node, node.elt, None, None)

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        elt: ast.AST | None,
        key: ast.AST | None,
        value: ast.AST | None,
    ) -> None:
        parent_scope = self._current_scope
        scope_name = f"{parent_scope.name}.<comprehension>@{node.lineno}"
        scope = ScopeInfo(scope_name, "comprehension", node.lineno, parent=parent_scope.name)
        self._register_scope(scope)
        self._scope_stack.append(scope)
        self.inference.register_enclosing_class(
            scope_name, self._class_stack[-1].scope_name if self._class_stack else None
        )

        for gen in node.generators:
            self._assign_targets(gen.target, node.lineno, "comprehension")
            self._register_assignment_for_targets(gen.target, None, node.lineno)
        for gen in node.generators:
            self.visit(gen.iter)
            for cond in gen.ifs:
                self.visit(cond)
        if elt is not None:
            self.visit(elt)
        if key is not None:
            self.visit(key)
        if value is not None:
            self.visit(value)

        self._scope_stack.pop()

    # -- imports ----------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        scope = self._current_scope.name
        for alias in node.names:
            defined_name = alias.asname or alias.name.split(".")[0]
            self._define_symbol(defined_name, "import", node.lineno)
            self._assign_variable(defined_name, "import", node.lineno)
            self.inference.register_assignment(scope, defined_name, node.lineno, node)
            self.imports.append(
                ImportInfo(
                    module=alias.name,
                    names=[defined_name],
                    asname=alias.asname,
                    lineno=node.lineno,
                    is_from=False,
                    is_star=False,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        scope = self._current_scope.name
        for alias in node.names:
            if alias.name == "*":
                self.star_import = True
                self.imports.append(
                    ImportInfo(
                        module=node.module or "",
                        names=["*"],
                        lineno=node.lineno,
                        is_from=True,
                        is_star=True,
                        level=node.level,
                    )
                )
                continue
            defined_name = alias.asname or alias.name
            kind = "future_import" if node.module == "__future__" else "import"
            self._define_symbol(defined_name, kind, node.lineno)
            self._assign_variable(defined_name, kind, node.lineno)
            self.inference.register_assignment(scope, defined_name, node.lineno, node)
            self.imports.append(
                ImportInfo(
                    module=node.module or "",
                    names=[defined_name],
                    asname=alias.asname,
                    lineno=node.lineno,
                    is_from=True,
                    is_star=False,
                    level=node.level,
                )
            )

    # -- assignments -----------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            if self._is_instance_target(target):
                method = self._function_stack[-1]
                self._assign_instance_var(target.attr, node.lineno, method.class_name or "")
                self._record_instance_value(target.attr, node.lineno, node.value)
                self.visit(target.value)
                continue
            if isinstance(target, (ast.Attribute, ast.Subscript)):
                self.visit(target.value)
                if isinstance(target, ast.Subscript):
                    self.visit(target.slice)
                continue
            kind = "class" if self._current_scope.kind == "class" else "variable"
            self._assign_targets(target, node.lineno, kind)
            self._register_assignment_for_targets(target, node.value, node.lineno)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.annotation)
        target = node.target
        if isinstance(target, ast.Name):
            annotation = ast.unparse(node.annotation)
            scope = self._effective_scope_for(target.id)
            self.inference.register_annotation(scope, target.id, annotation)
            kind = "class" if self._current_scope.kind == "class" else "variable"
            self._assign_targets(target, node.lineno, kind)
            if node.value is not None:
                self.inference.register_assignment(scope, target.id, node.lineno, node.value)
        elif self._is_instance_target(target):
            method = self._function_stack[-1]
            self._assign_instance_var(target.attr, node.lineno, method.class_name or "")
            if node.value is not None:
                self._record_instance_value(target.attr, node.lineno, node.value)
            self.visit(target.value)
        elif isinstance(target, ast.Attribute):
            self.visit(target.value)
        elif isinstance(target, ast.Subscript):
            self.visit(target.value)
            self.visit(target.slice)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        target = node.target
        if isinstance(target, ast.Name):
            self._reference(target.id, node.lineno, target.col_offset)
            self._assign_targets(target, node.lineno, "variable")
            synthetic = ast.BinOp(
                left=ast.Name(id=target.id, ctx=ast.Load()),
                op=node.op,
                right=node.value,
            )
            synthetic.lineno = node.lineno
            scope = self._effective_scope_for(target.id)
            self.inference.register_assignment(scope, target.id, node.lineno, synthetic)
        elif self._is_instance_target(target):
            method = self._function_stack[-1]
            self._assign_instance_var(target.attr, node.lineno, method.class_name or "")
            synthetic = ast.BinOp(left=target, op=node.op, right=node.value)
            synthetic.lineno = node.lineno
            self._record_instance_value(target.attr, node.lineno, synthetic)
            self.visit(target.value)
        elif isinstance(target, ast.Attribute):
            self.visit(target.value)
        elif isinstance(target, ast.Subscript):
            self.visit(target.value)
            self.visit(target.slice)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._assign_targets(node.target, node.lineno, "variable")
            scope = self._effective_scope_for(node.target.id)
            self.inference.register_assignment(scope, node.target.id, node.lineno, node.value)
        else:
            self.visit(node.target)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._assign_targets(target, node.lineno, "variable")
            else:
                self.visit(target)

    # -- control flow ----------------------------------------------------------

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        self.control_flow.append(
            ControlFlowInfo(
                kind="if",
                lineno=node.lineno,
                end_lineno=node.end_lineno or node.lineno,
                condition=ast.unparse(node.test),
                branches=self._count_if_branches(node),
                scope=self._current_scope.name,
            )
        )
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    @staticmethod
    def _count_if_branches(node: ast.If) -> int:
        count = 1
        remaining = node.orelse
        while remaining:
            if len(remaining) == 1 and isinstance(remaining[0], ast.If):
                count += 1
                remaining = remaining[0].orelse
            else:
                count += 1
                break
        return count

    def visit_For(self, node: ast.For) -> None:
        self._visit_loop(node, async_=False)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_loop(node, async_=True)

    def _visit_loop(self, node: ast.For | ast.AsyncFor, async_: bool) -> None:
        self.visit(node.iter)
        kind = "for" if not async_ else "async_for"
        loop = LoopInfo(
            kind=kind,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            iterable=ast.unparse(node.iter),
            variables=[n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)],
            scope=self._current_scope.name,
            has_else=bool(node.orelse),
            body_size=len(node.body),
        )
        self._assign_targets(node.target, node.lineno, "loop_var")
        self._register_assignment_for_targets(node.target, node.iter, node.lineno)
        self._loop_stack.append(loop)
        for stmt in node.body:
            self.visit(stmt)
        self._loop_stack.pop()
        for stmt in node.orelse:
            self.visit(stmt)
        self.loops.append(loop)

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        loop = LoopInfo(
            kind="while",
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            iterable=ast.unparse(node.test),
            scope=self._current_scope.name,
            has_else=bool(node.orelse),
            body_size=len(node.body),
        )
        self._loop_stack.append(loop)
        for stmt in node.body:
            self.visit(stmt)
        self._loop_stack.pop()
        for stmt in node.orelse:
            self.visit(stmt)
        self.loops.append(loop)

    def visit_With(self, node: ast.With) -> None:
        self._visit_with(node, async_=False)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_with(node, async_=True)

    def _visit_with(self, node: ast.With | ast.AsyncWith, async_: bool) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._assign_targets(item.optional_vars, node.lineno, "with_target")
                self._register_assignment_for_targets(item.optional_vars, None, node.lineno)
        self.control_flow.append(
            ControlFlowInfo(
                kind="with" if not async_ else "async_with",
                lineno=node.lineno,
                end_lineno=node.end_lineno or node.lineno,
                scope=self._current_scope.name,
            )
        )
        for stmt in node.body:
            self.visit(stmt)

    def visit_Try(self, node: ast.Try) -> None:
        self._visit_try(node)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self._visit_try(node)

    def _visit_try(self, node: ast.Try | ast.TryStar) -> None:
        self.control_flow.append(
            ControlFlowInfo(
                kind="try",
                lineno=node.lineno,
                end_lineno=node.end_lineno or node.lineno,
                scope=self._current_scope.name,
            )
        )
        handlers_info: List[Tuple[str | None, str | None]] = []
        for handler in node.handlers:
            handlers_info.append(
                (
                    ast.unparse(handler.type) if handler.type else None,
                    handler.name,
                )
            )
        raises_in_try = self._collect_raise_names(list(node.body))
        for stmt in node.body:
            self.visit(stmt)
        for handler in node.handlers:
            self.visit(handler)
        for stmt in node.orelse:
            self.visit(stmt)
        for stmt in node.finalbody:
            self.visit(stmt)
        self.exception_handlers.append(
            ExceptionHandlerInfo(
                lineno=node.lineno,
                end_lineno=node.end_lineno or node.lineno,
                scope=self._current_scope.name,
                handlers=handlers_info,
                raises_in_try=raises_in_try,
                has_else=bool(node.orelse),
                has_finally=bool(node.finalbody),
            )
        )

    def _collect_raise_names(self, body: List[ast.stmt]) -> List[str]:
        found: List[str] = []

        def visit_stmts(stmts: List[ast.stmt]) -> None:
            for stmt in stmts:
                if isinstance(stmt, ast.Raise):
                    exc = stmt.exc
                    if isinstance(exc, ast.Name):
                        found.append(exc.id)
                    elif isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                        found.append(exc.func.id)
                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                elif isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While)):
                    visit_stmts(stmt.body)
                    visit_stmts(stmt.orelse)
                elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                    visit_stmts(stmt.body)
                elif isinstance(stmt, (ast.Try, ast.TryStar)):
                    visit_stmts(stmt.body)
                    visit_stmts(stmt.orelse)
                    visit_stmts(stmt.finalbody)
                    for handler in stmt.handlers:
                        visit_stmts(handler.body)
                elif isinstance(stmt, ast.Match):
                    for case in stmt.cases:
                        visit_stmts(case.body)

        visit_stmts(body)
        return found

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name:
            self._assign_variable(node.name, "exception_var", node.lineno)
            self.inference.register_parameter(self._current_scope.name, node.name)
            if node.type is not None:
                self.inference.register_fixed_type(
                    self._current_scope.name,
                    node.name,
                    TypeInfo(ast.unparse(node.type), 0.7, "exception"),
                )
        for stmt in node.body:
            self.visit(stmt)

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        self.control_flow.append(
            ControlFlowInfo(
                kind="match",
                lineno=node.lineno,
                end_lineno=node.end_lineno or node.lineno,
                branches=len(node.cases),
                scope=self._current_scope.name,
            )
        )
        for case in node.cases:
            self._define_match_bindings(case.pattern, node.lineno)
            if case.guard is not None:
                self.visit(case.guard)
            for stmt in case.body:
                self.visit(stmt)

    def _define_match_bindings(self, pattern: ast.AST, line: int) -> None:
        if isinstance(pattern, ast.MatchAs):
            if pattern.name:
                self._assign_variable(pattern.name, "variable", line)
            if pattern.pattern is not None:
                self._define_match_bindings(pattern.pattern, line)
        elif isinstance(pattern, ast.MatchStar):
            if pattern.name:
                self._assign_variable(pattern.name, "variable", line)
        elif isinstance(pattern, ast.MatchMapping):
            if pattern.rest:
                self._assign_variable(pattern.rest, "variable", line)
        elif isinstance(pattern, (ast.MatchSequence, ast.MatchOr)):
            for sub in pattern.patterns:
                self._define_match_bindings(sub, line)
        elif isinstance(pattern, ast.MatchClass):
            for sub in pattern.patterns:
                self._define_match_bindings(sub, line)
            for kwd in pattern.kwds:
                self._define_match_bindings(kwd.pattern, line)

    # -- statements -------------------------------------------------------------

    def visit_Return(self, node: ast.Return) -> None:
        fn = self._function_stack[-1] if self._function_stack else None
        if node.value is not None:
            self.visit(node.value)
        value_dump = ast.unparse(node.value) if node.value else None
        scope = fn.scope_name if fn else MODULE_SCOPE
        self.returns.append(
            ReturnInfo(lineno=node.lineno, value=value_dump, scope=scope)
        )
        self._return_value_nodes.append(node.value)
        self.control_flow.append(
            ControlFlowInfo(
                kind="return", lineno=node.lineno, condition=value_dump, scope=scope
            )
        )
        if fn is not None:
            fn.has_explicit_return = True
            self.inference.register_return(fn.scope_name, node.value)

    def visit_Raise(self, node: ast.Raise) -> None:
        if node.exc is not None:
            self.visit(node.exc)
        if node.cause is not None:
            self.visit(node.cause)
        scope = self._current_scope.name
        self.control_flow.append(
            ControlFlowInfo(
                kind="raise",
                lineno=node.lineno,
                condition=ast.unparse(node.exc) if node.exc else None,
                scope=scope,
            )
        )
        fn = self._function_stack[-1] if self._function_stack else None
        if fn is not None and isinstance(node.exc, (ast.Name, ast.Call)):
            if isinstance(node.exc, ast.Name):
                name = node.exc.id
            elif isinstance(node.exc.func, ast.Name):
                name = node.exc.func.id
            else:
                name = None
            if name and name not in fn.raises:
                fn.raises.append(name)

    def visit_Break(self, node: ast.Break) -> None:
        self.control_flow.append(
            ControlFlowInfo(kind="break", lineno=node.lineno, scope=self._current_scope.name)
        )
        if self._loop_stack:
            self._loop_stack[-1].has_break = True

    def visit_Continue(self, node: ast.Continue) -> None:
        self.control_flow.append(
            ControlFlowInfo(
                kind="continue", lineno=node.lineno, scope=self._current_scope.name
            )
        )
        if self._loop_stack:
            self._loop_stack[-1].has_continue = True

    def visit_Global(self, node: ast.Global) -> None:
        self._current_scope.globals_declared.extend(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self._current_scope.nonlocals_declared.extend(node.names)

    def visit_Call(self, node: ast.Call) -> None:
        call = CallInfo(
            name=self._call_name(node.func),
            lineno=node.lineno,
            args_count=len(node.args),
            keyword_args=[kw.arg for kw in node.keywords if kw.arg is not None],
            is_method_call=isinstance(node.func, ast.Attribute),
            receiver=(
                ast.unparse(node.func.value) if isinstance(node.func, ast.Attribute) else None
            ),
            scope=self._current_scope.name,
        )
        self.calls.append(call)
        fn = self._function_stack[-1] if self._function_stack else None
        if fn is not None:
            fn.calls.append(call)
        self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._reference(node.id, node.lineno, node.col_offset)
        elif isinstance(node.ctx, ast.Store):
            self._assign_targets(node, node.lineno, "variable")


# ---------------------------------------------------------------------------
# Public analyzer
# ---------------------------------------------------------------------------


class PythonAnalyzer:
    """Analyze Python source code and produce a structured model."""

    def __init__(self) -> None:
        self._builtin_names = frozenset(dir(builtins))

    def analyze(self, source: str) -> AnalysisResult:
        """Analyze Python source and return an :class:`AnalysisResult`."""
        tree, error = PythonParser.parse(source)
        if error is not None:
            return AnalysisResult(source=source, parse_successful=False, syntax_error=error)

        walker = _CodeWalker(tree, source)
        walker.run()
        walker.finalize()

        issues: List[Issue] = []
        issues.extend(
            detect_undefined_names(walker.symbol_table, walker.usage_events, walker.star_import)
        )
        issues.extend(detect_unused_symbols(walker.variables))
        issues.extend(detect_unreachable_code(tree))
        issues.extend(detect_suspicious_constructs(tree, self._builtin_names))
        issues.sort(key=lambda issue: (issue.line, issue.severity))

        # Complexity per function.
        for fn in walker.functions:
            node = walker.function_nodes.get(fn.scope_name)
            if node is not None:
                cyclomatic, nesting, statements = measure_function(node)
                fn.complexity = cyclomatic
                fn.nesting_depth = nesting
                fn.statement_count = statements

        # Type inference: first pass over plain variables.
        inference = walker.inference
        for var in walker.variables:
            var.inferred_type = inference.infer_variable(var.scope, var.name).type_name

        # Register instance/class attribute types so 'self.x' resolves.
        # A few passes let chained attributes (self.a = self.b) resolve;
        # known (non-unknown) assignments are preferred over unknown ones.
        for _ in range(3):
            inference.clear_cache()
            for (class_scope, attr), values in walker._instance_values.items():
                best: TypeInfo | None = None
                for _, value, assign_scope in values:
                    inferred = inference.infer_expression(value, assign_scope)
                    if inferred.type_name:
                        best = inferred
                if best is None:
                    _, last_value, assign_scope = values[-1]
                    best = inference.infer_expression(last_value, assign_scope)
                inference.register_instance_attribute(
                    class_scope,
                    attr,
                    TypeInfo(best.type_name, best.confidence, "instance_attribute"),
                )
        for var in walker.variables:
            if var.kind == "class" and var.inferred_type:
                inference.register_instance_attribute(
                    var.scope, var.name, TypeInfo(var.inferred_type, 0.7, "class_variable")
                )

        # Second pass now that 'self.x' and class methods are resolvable.
        inference.clear_cache()
        for var in walker.variables:
            if var.kind == "instance":
                var.inferred_type = inference.infer_instance_attribute(
                    var.scope, var.name
                ).type_name
            else:
                var.inferred_type = inference.infer_variable(var.scope, var.name).type_name
        for fn in walker.functions:
            fn.returns = inference.infer_function_returns(fn.scope_name)
        for return_info, value_node in zip(walker.returns, walker._return_value_nodes):
            if value_node is None:
                return_info.value_type = "NoneType"
            else:
                return_info.value_type = inference.infer_expression(
                    value_node, return_info.scope
                ).type_name

        structure = walker.structure
        structure.total_functions = len(walker.functions)
        structure.total_classes = len(walker.classes)
        structure.total_imports = len(walker.imports)

        return AnalysisResult(
            source=source,
            parse_successful=True,
            syntax_error=None,
            tree=tree,
            structure=structure,
            functions=walker.functions,
            classes=walker.classes,
            imports=walker.imports,
            variables=walker.variables,
            scopes=walker.scopes,
            symbol_table=walker.symbol_table,
            control_flow=walker.control_flow,
            loops=walker.loops,
            exception_handlers=walker.exception_handlers,
            calls=walker.calls,
            returns=walker.returns,
            issues=issues,
            complexity=measure_module(tree),
        )

    def parse(self, source: str) -> tuple[ast.Module | None, SyntaxIssue | None]:
        """Parse only; convenience wrapper around :class:`PythonParser`."""
        return PythonParser.parse(source)

    def inspect(self, source: str) -> str | None:
        """Return an AST dump of the source (AST inspection helper)."""
        tree, _ = PythonParser.parse(source)
        return PythonParser.inspect(tree) if tree is not None else None

    def explain(self, source: str) -> str:
        """Analyze and produce a structural explanation."""
        return PythonExplainer().explain(self.analyze(source))

    def diagnostics(self, source: str) -> Dict[str, object]:
        """Analyze and return a diagnostics report."""
        return self.analyze(source).diagnostics()


__all__ = ["AnalysisResult", "PythonAnalyzer"]
