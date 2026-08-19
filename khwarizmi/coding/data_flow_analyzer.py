"""
Khwarizmi Data Flow Analyzer — Static Analysis for Variable Definitions/Uses.

This module provides offline, deterministic data flow analysis for Python code
using only the standard library ast module. It tracks variable definitions and
uses across the AST to detect:
- Unused variables (defined but never read)
- Undefined references (used before definition)
- Dead assignments (overwritten before being read)

All analysis is purely static — no code execution occurs.
"""

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any


@dataclass
class VariableBinding:
    """Represents a single variable binding in a scope."""
    name: str
    scope: str  # e.g., "module", "function_name", "class_name.method"
    defined_line: int
    defined_col: int = 0
    used_lines: List[int] = field(default_factory=list)
    assigned_lines: List[int] = field(default_factory=list)
    is_parameter: bool = False
    is_import: bool = False
    is_loop_var: bool = False
    is_exception_var: bool = False
    is_comprehension_var: bool = False

    @property
    def is_used(self) -> bool:
        return len(self.used_lines) > 0

    @property
    def is_unused(self) -> bool:
        return not self.is_used


@dataclass
class DataFlowIssue:
    """Represents a data flow issue detected during analysis."""
    issue_type: str  # "unused_variable", "undefined_reference", "dead_assignment"
    name: str
    line: int
    col: int = 0
    scope: str = ""
    message: str = ""

    def __str__(self) -> str:
        return f"[{self.issue_type}] {self.name} at line {self.line}: {self.message}"


@dataclass
class DataFlowReport:
    """Structured report of data flow analysis results."""
    source_code: str
    parse_successful: bool = True
    syntax_error: Optional[str] = None
    
    # Variable bindings organized by scope
    bindings_by_scope: Dict[str, List[VariableBinding]] = field(default_factory=dict)
    
    # All bindings flattened
    all_bindings: List[VariableBinding] = field(default_factory=list)
    
    # Detected issues
    unused_variables: List[DataFlowIssue] = field(default_factory=list)
    undefined_references: List[DataFlowIssue] = field(default_factory=list)
    dead_assignments: List[DataFlowIssue] = field(default_factory=list)
    
    # Summary statistics
    total_variables: int = 0
    total_unused: int = 0
    total_undefined: int = 0
    total_dead: int = 0

    @property
    def is_valid(self) -> bool:
        """Return True if no data flow issues were detected."""
        return (
            self.parse_successful
            and len(self.unused_variables) == 0
            and len(self.undefined_references) == 0
            and len(self.dead_assignments) == 0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to dictionary."""
        return {
            "parse_successful": self.parse_successful,
            "syntax_error": self.syntax_error,
            "total_variables": self.total_variables,
            "total_unused": self.total_unused,
            "total_undefined": self.total_undefined,
            "total_dead": self.total_dead,
            "is_valid": self.is_valid,
            "unused_variables": [
                {"name": i.name, "line": i.line, "scope": i.scope, "message": i.message}
                for i in self.unused_variables
            ],
            "undefined_references": [
                {"name": i.name, "line": i.line, "scope": i.scope, "message": i.message}
                for i in self.undefined_references
            ],
            "dead_assignments": [
                {"name": i.name, "line": i.line, "scope": i.scope, "message": i.message}
                for i in self.dead_assignments
            ],
        }


class DataFlowAnalyzer(ast.NodeVisitor):
    """
    AST visitor that performs data flow analysis on Python code.

    Tracks variable definitions and uses to identify:
    - Unused variables
    - Undefined references  
    - Dead assignments (variable overwritten before use)

    Usage:
        analyzer = DataFlowAnalyzer()
        report = analyzer.analyze(source_code)
    """

    # Built-in names that should not be flagged as undefined
    BUILTINS = frozenset([
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray',
        'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex',
        'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec',
        'filter', 'float', 'format', 'frozenset', 'getattr', 'globals',
        'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance',
        'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max',
        'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow',
        'print', 'property', 'range', 'repr', 'reversed', 'round', 'set',
        'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super',
        'tuple', 'type', 'vars', 'zip', '__import__',
        'True', 'False', 'None', 'Ellipsis', 'NotImplemented',
        '__name__', '__doc__', '__package__', '__loader__', '__spec__',
        '__build_class__', '__debug__', 'copyright', 'credits', 'license',
    ])

    def __init__(self):
        self.bindings: Dict[str, VariableBinding] = {}
        self.issues: List[DataFlowIssue] = []
        self.current_scope: str = "module"
        self.scope_stack: List[str] = []
        # Track last assignment line per variable for dead assignment detection
        self.last_assignment: Dict[str, int] = {}
        self.last_assignment_was_read: Dict[str, bool] = {}
        # Track which variables are defined in current scope
        self.defined_in_scope: Dict[str, Set[str]] = {"module": set()}

    def analyze(self, source_code: str) -> DataFlowReport:
        """
        Analyze Python source code and return a DataFlowReport.

        Args:
            source_code: Python source code as string

        Returns:
            DataFlowReport with analysis results
        """
        self.bindings = {}
        self.issues = []
        self.current_scope = "module"
        self.scope_stack = ["module"]
        self.last_assignment = {}
        self.last_assignment_was_read = {}
        self.defined_in_scope = {"module": set()}

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return DataFlowReport(
                source_code=source_code,
                parse_successful=False,
                syntax_error=f"Line {e.lineno}: {e.msg}" if e.lineno else str(e),
            )

        self.visit(tree)

        # Post-analysis: check for unused variables and undefined references
        self._finalize_analysis()

        # Build report
        bindings_by_scope: Dict[str, List[VariableBinding]] = {}
        for binding in self.bindings.values():
            if binding.scope not in bindings_by_scope:
                bindings_by_scope[binding.scope] = []
            bindings_by_scope[binding.scope].append(binding)

        unused = [i for i in self.issues if i.issue_type == "unused_variable"]
        undefined = [i for i in self.issues if i.issue_type == "undefined_reference"]
        dead = [i for i in self.issues if i.issue_type == "dead_assignment"]

        return DataFlowReport(
            source_code=source_code,
            parse_successful=True,
            bindings_by_scope=bindings_by_scope,
            all_bindings=list(self.bindings.values()),
            unused_variables=unused,
            undefined_references=undefined,
            dead_assignments=dead,
            total_variables=len(self.bindings),
            total_unused=len(unused),
            total_undefined=len(undefined),
            total_dead=len(dead),
        )

    def _get_full_scope_name(self, base_name: str) -> str:
        """Get full scoped name for a function/method/class."""
        if len(self.scope_stack) <= 1:
            return base_name
        parent = self.scope_stack[-2] if len(self.scope_stack) > 1 else "module"
        if parent == "module":
            return base_name
        return f"{parent}.{base_name}"

    def _make_key(self, name: str) -> str:
        """Create unique key for a variable in current scope."""
        return f"{self.current_scope}:{name}"

    def _add_binding(
        self,
        name: str,
        line: int,
        col: int = 0,
        is_parameter: bool = False,
        is_import: bool = False,
        is_loop_var: bool = False,
        is_exception_var: bool = False,
        is_comprehension_var: bool = False,
    ) -> VariableBinding:
        """Add or update a variable binding."""
        key = self._make_key(name)
        
        # Check for dead assignment (previous binding not read)
        if key in self.bindings and not self.last_assignment_was_read.get(key, True):
            prev = self.bindings[key]
            self.issues.append(DataFlowIssue(
                issue_type="dead_assignment",
                name=name,
                line=prev.assigned_lines[-1] if prev.assigned_lines else line,
                col=0,
                scope=self.current_scope,
                message=f"Variable '{name}' overwritten at line {line} before being used",
            ))

        if key not in self.bindings:
            binding = VariableBinding(
                name=name,
                scope=self.current_scope,
                defined_line=line,
                defined_col=col,
                is_parameter=is_parameter,
                is_import=is_import,
                is_loop_var=is_loop_var,
                is_exception_var=is_exception_var,
                is_comprehension_var=is_comprehension_var,
            )
            self.bindings[key] = binding
            self.defined_in_scope.setdefault(self.current_scope, set()).add(name)
        else:
            binding = self.bindings[key]
        
        binding.assigned_lines.append(line)
        self.last_assignment[key] = line
        self.last_assignment_was_read[key] = False
        
        return binding

    def _record_use(self, name: str, line: int, col: int = 0) -> bool:
        """
        Record a variable use. Returns True if variable was found, False if undefined.
        """
        if name in self.BUILTINS:
            return True

        # Search in current scope and enclosing scopes
        for i in range(len(self.scope_stack) - 1, -1, -1):
            scope = self.scope_stack[i]
            key = f"{scope}:{name}"
            if key in self.bindings:
                binding = self.bindings[key]
                if line not in binding.used_lines:
                    binding.used_lines.append(line)
                # Mark previous assignment as read
                if key in self.last_assignment:
                    self.last_assignment_was_read[key] = True
                return True

        # Not found - could be undefined or forward reference
        # For now, don't flag as undefined (Python allows forward refs in some cases)
        return False

    def _finalize_analysis(self):
        """Post-process to find unused variables and undefined references."""
        for key, binding in self.bindings.items():
            # Skip certain types of bindings from unused check
            if binding.is_import or binding.is_parameter:
                continue
            # Module-level dunder assignments are typically not "unused"
            if binding.scope == "module" and binding.name.startswith("__"):
                continue
                
            if not binding.is_used:
                self.issues.append(DataFlowIssue(
                    issue_type="unused_variable",
                    name=binding.name,
                    line=binding.defined_line,
                    scope=binding.scope,
                    message=f"Variable '{binding.name}' is defined but never used",
                ))

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._visit_function(node)

    def _visit_function(self, node):
        """Visit a function definition."""
        func_name = self._get_full_scope_name(node.name)
        self.scope_stack.append(func_name)
        self.current_scope = func_name
        self.defined_in_scope.setdefault(func_name, set())

        # Add function itself as binding in parent scope
        parent_scope = self.scope_stack[-2] if len(self.scope_stack) > 1 else "module"
        parent_key = f"{parent_scope}:{node.name}"
        if parent_key not in self.bindings:
            self.bindings[parent_key] = VariableBinding(
                name=node.name,
                scope=parent_scope,
                defined_line=node.lineno,
            )

        # Visit decorators in parent scope
        for decorator in node.decorator_list:
            self.visit(decorator)

        # Register parameters
        args = node.args
        param_names = []
        
        # Positional-only args (Python 3.8+)
        for arg in args.posonlyargs:
            param_names.append(arg.arg)
            self._add_binding(arg.arg, arg.lineno, arg.col_offset, is_parameter=True)
        
        # Regular positional/keyword args
        for arg in args.args:
            param_names.append(arg.arg)
            self._add_binding(arg.arg, arg.lineno, arg.col_offset, is_parameter=True)
        
        # *args
        if args.vararg:
            param_names.append(args.vararg.arg)
            self._add_binding(args.vararg.arg, args.vararg.lineno, 
                            args.vararg.col_offset, is_parameter=True)
        
        # Keyword-only args
        for arg in args.kwonlyargs:
            param_names.append(arg.arg)
            self._add_binding(arg.arg, arg.lineno, arg.col_offset, is_parameter=True)
        
        # **kwargs
        if args.kwarg:
            param_names.append(args.kwarg.arg)
            self._add_binding(args.kwarg.arg, args.kwarg.lineno,
                            args.kwarg.col_offset, is_parameter=True)

        # Visit default values in parent scope (before entering function body)
        old_scope = self.current_scope
        self.current_scope = parent_scope
        for default in args.defaults + args.kw_defaults:
            if default is not None:
                self.visit(default)
        self.current_scope = old_scope

        # Visit function body
        for stmt in node.body:
            self.visit(stmt)

        self.scope_stack.pop()
        self.current_scope = self.scope_stack[-1] if self.scope_stack else "module"

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit a class definition."""
        class_name = self._get_full_scope_name(node.name)
        self.scope_stack.append(class_name)
        self.current_scope = class_name
        self.defined_in_scope.setdefault(class_name, set())

        # Add class as binding in parent scope
        parent_scope = self.scope_stack[-2] if len(self.scope_stack) > 1 else "module"
        parent_key = f"{parent_scope}:{node.name}"
        if parent_key not in self.bindings:
            self.bindings[parent_key] = VariableBinding(
                name=node.name,
                scope=parent_scope,
                defined_line=node.lineno,
            )

        # Visit decorators and bases in parent scope
        old_scope = self.current_scope
        self.current_scope = parent_scope
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self.current_scope = old_scope

        # Visit class body
        for stmt in node.body:
            self.visit(stmt)

        self.scope_stack.pop()
        self.current_scope = self.scope_stack[-1] if self.scope_stack else "module"

    def visit_Name(self, node: ast.Name):
        """Visit a name reference (load, store, del context)."""
        if isinstance(node.ctx, ast.Store):
            self._add_binding(node.id, node.lineno, node.col_offset)
        elif isinstance(node.ctx, ast.Load):
            self._record_use(node.id, node.lineno, node.col_offset)
        # Del context - variable being deleted
        elif isinstance(node.ctx, ast.Del):
            self._record_use(node.id, node.lineno, node.col_offset)

    def visit_Import(self, node: ast.Import):
        """Visit import statement."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            # For dotted imports like "import os.path", bind just "os"
            if "." in alias.name and not alias.asname:
                name = alias.name.split(".")[0]
            self._add_binding(name, node.lineno, node.col_offset, is_import=True)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Visit from ... import statement."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self._add_binding(name, node.lineno, node.col_offset, is_import=True)

    def visit_For(self, node: ast.For):
        """Visit for loop - handle loop variables."""
        # Visit iterable first (in current scope)
        self.visit(node.iter)
        
        # Register loop variables
        self._visit_target(node.target, node.lineno)
        
        # Visit body
        for stmt in node.body:
            self.visit(stmt)
        
        # Visit else clause
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_While(self, node: ast.While):
        """Visit while loop."""
        self.visit(node.test)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_With(self, node: ast.With):
        """Visit with statement."""
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self._visit_target(item.optional_vars, node.lineno)
        for stmt in node.body:
            self.visit(stmt)

    def visit_AsyncFor(self, node: ast.AsyncFor):
        """Visit async for loop."""
        self.visit(node.iter)
        self._visit_target(node.target, node.lineno)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_AsyncWith(self, node: ast.AsyncWith):
        """Visit async with statement."""
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self._visit_target(item.optional_vars, node.lineno)
        for stmt in node.body:
            self.visit(stmt)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        """Visit exception handler."""
        if node.type:
            self.visit(node.type)
        if node.name:
            self._add_binding(node.name, node.lineno, node.col_offset, 
                            is_exception_var=True)
        for stmt in node.body:
            self.visit(stmt)

    def visit_comprehension(self, node: ast.comprehension):
        """Visit comprehension clause."""
        # Visit iterable first
        self.visit(node.iter)
        
        # Register loop variable
        self._visit_target(node.target, node.lineno, is_comprehension=True)
        
        # Visit ifs
        for if_clause in node.ifs:
            self.visit(if_clause)

    def _visit_target(self, target: ast.AST, lineno: int, is_comprehension: bool = False):
        """Visit assignment target (handles tuples, lists, attributes)."""
        if isinstance(target, ast.Name):
            self._add_binding(
                target.id, lineno, target.col_offset,
                is_loop_var=not is_comprehension,
                is_comprehension_var=is_comprehension,
            )
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._visit_target(elt, lineno, is_comprehension)
        elif isinstance(target, ast.Starred):
            self._visit_target(target.value, lineno, is_comprehension)
        # Ignore attribute assignments (obj.attr = x)

    def visit_Call(self, node: ast.Call):
        """Visit function call."""
        # Visit the function being called
        self.visit(node.func)
        # Visit arguments
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)

    def visit_Attribute(self, node: ast.Attribute):
        """Visit attribute access."""
        # Only visit the value part, not the attribute name itself
        self.visit(node.value)

    def visit_Assign(self, node: ast.Assign):
        """Visit assignment statement."""
        # Visit value first
        self.visit(node.value)
        # Then visit targets
        for target in node.targets:
            self._visit_target(target, node.lineno)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Visit annotated assignment."""
        if node.value:
            self.visit(node.value)
        self.visit(node.annotation)
        if node.target:
            self._visit_target(node.target, node.lineno)

    def visit_AugAssign(self, node: ast.AugAssign):
        """Visit augmented assignment (+=, -=, etc.)."""
        self.visit(node.value)
        self._visit_target(node.target, node.lineno)
        # Augmented assign is both read and write
        if isinstance(node.target, ast.Name):
            self._record_use(node.target.id, node.lineno, node.col_offset)

    def visit_Global(self, node: ast.Global):
        """Visit global declaration."""
        # Global refers to module-level variable, mark as used
        for name in node.names:
            self._record_use(name, node.lineno, node.col_offset)

    def visit_Nonlocal(self, node: ast.Nonlocal):
        """Visit nonlocal declaration."""
        for name in node.names:
            self._record_use(name, node.lineno, node.col_offset)

    def visit_ListComp(self, node: ast.ListComp):
        """Visit list comprehension."""
        # Process generators first to set up loop variables
        for gen in node.generators:
            self.visit_comprehension(gen)
        # Then visit the element expression
        self.visit(node.elt)

    def visit_SetComp(self, node: ast.SetComp):
        """Visit set comprehension."""
        for gen in node.generators:
            self.visit_comprehension(gen)
        self.visit(node.elt)

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        """Visit generator expression."""
        for gen in node.generators:
            self.visit_comprehension(gen)
        self.visit(node.elt)

    def visit_DictComp(self, node: ast.DictComp):
        """Visit dict comprehension."""
        for gen in node.generators:
            self.visit_comprehension(gen)
        self.visit(node.key)
        self.visit(node.value)

    def generic_visit(self, node):
        """Default visitor for unhandled node types."""
        for child in ast.iter_child_nodes(node):
            self.visit(child)
