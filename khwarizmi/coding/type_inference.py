"""
Khwarizmi Type Inference — Lightweight Deterministic Local Type Inference.

This module provides offline, deterministic type inference for Python code
using only the standard library ast module. It infers types for:
- Literals (int, float, str, bool, list, dict, tuple, set, None)
- Variable assignments
- Binary operations
- Return statements
- Function calls (limited to known builtins)

All inference is purely static and local — no external services or execution.
"""

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Union


@dataclass
class InferredType:
    """Represents an inferred type for a variable or expression."""
    type_name: str  # "int", "float", "str", "bool", "list", "dict", "tuple", "set", "None", "Any", "Unknown"
    confidence: float = 1.0  # 0.0 to 1.0 confidence in the inference
    source_line: int = 0
    container_type: Optional[str] = None  # For containers: element type
    key_type: Optional[str] = None  # For dicts: key type
    value_type: Optional[str] = None  # For dicts: value type

    def __str__(self) -> str:
        if self.type_name == "list" and self.container_type:
            return f"list[{self.container_type}]"
        elif self.type_name == "dict" and self.key_type and self.value_type:
            return f"dict[{self.key_type}, {self.value_type}]"
        elif self.type_name == "tuple" and self.container_type:
            return f"tuple[{self.container_type}]"
        elif self.type_name == "set" and self.container_type:
            return f"set[{self.container_type}]"
        return self.type_name

    @classmethod
    def unknown(cls, line: int = 0) -> "InferredType":
        return cls(type_name="Unknown", confidence=0.0, source_line=line)

    @classmethod
    def any_type(cls, line: int = 0) -> "InferredType":
        return cls(type_name="Any", confidence=0.5, source_line=line)

    @classmethod
    def from_literal(cls, value: Any, line: int = 0) -> "InferredType":
        """Create InferredType from a literal value."""
        if value is None:
            return cls(type_name="None", source_line=line)
        elif isinstance(value, bool):
            return cls(type_name="bool", source_line=line)
        elif isinstance(value, int):
            return cls(type_name="int", source_line=line)
        elif isinstance(value, float):
            return cls(type_name="float", source_line=line)
        elif isinstance(value, str):
            return cls(type_name="str", source_line=line)
        elif isinstance(value, list):
            elem_type = cls.from_list_elements(value, line).type_name
            return cls(type_name="list", container_type=elem_type, source_line=line)
        elif isinstance(value, tuple):
            elem_type = cls.from_list_elements(list(value), line).type_name
            return cls(type_name="tuple", container_type=elem_type, source_line=line)
        elif isinstance(value, set):
            elem_type = cls.from_list_elements(list(value), line).type_name
            return cls(type_name="set", container_type=elem_type, source_line=line)
        elif isinstance(value, dict):
            if not value:
                return cls(type_name="dict", key_type="Any", value_type="Any", source_line=line)
            keys = [cls.from_literal(k, line) for k in value.keys()]
            values = [cls.from_literal(v, line) for v in value.values()]
            key_type = cls._common_type(keys).type_name
            value_type = cls._common_type(values).type_name
            return cls(type_name="dict", key_type=key_type, value_type=value_type, source_line=line)
        else:
            return cls.any_type(line)

    @staticmethod
    def from_list_elements(elements: list, line: int = 0) -> "InferredType":
        """Infer type from list elements."""
        if not elements:
            return InferredType(type_name="Any", confidence=0.0, source_line=line)
        elem_types = [InferredType.from_literal(e, line) for e in elements]
        return InferredType._common_type(elem_types)

    @staticmethod
    def _common_type(types: List["InferredType"]) -> "InferredType":
        """Find common type among a list of types."""
        if not types:
            return InferredType.unknown()
        
        # If all same type, return it
        type_names = set(t.type_name for t in types)
        if len(type_names) == 1:
            return types[0]
        
        # Handle numeric hierarchy
        if type_names <= {"int", "float"}:
            if "float" in type_names:
                return InferredType(type_name="float", confidence=0.9)
            return InferredType(type_name="int", confidence=0.9)
        
        # Mixed types -> Any
        return InferredType(type_name="Any", confidence=0.3)


@dataclass
class TypeBinding:
    """Represents a type binding for a variable."""
    name: str
    scope: str
    inferred_type: InferredType
    defined_line: int
    assigned_lines: List[int] = field(default_factory=list)
    type_sources: List[str] = field(default_factory=list)  # How type was inferred


@dataclass
class TypeInferenceReport:
    """Structured report of type inference results."""
    source_code: str
    parse_successful: bool = True
    syntax_error: Optional[str] = None
    
    # Type bindings by scope
    bindings_by_scope: Dict[str, List[TypeBinding]] = field(default_factory=dict)
    
    # All bindings flattened
    all_bindings: List[TypeBinding] = field(default_factory=list)
    
    # Function return type inferences
    function_returns: Dict[str, InferredType] = field(default_factory=dict)
    
    # Summary statistics
    total_variables: int = 0
    typed_variables: int = 0  # Variables with non-Unknown/Any type
    unknown_variables: int = 0

    @property
    def typing_coverage(self) -> float:
        """Return percentage of variables with known types."""
        if self.total_variables == 0:
            return 0.0
        return self.typed_variables / self.total_variables

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to dictionary."""
        return {
            "parse_successful": self.parse_successful,
            "syntax_error": self.syntax_error,
            "total_variables": self.total_variables,
            "typed_variables": self.typed_variables,
            "unknown_variables": self.unknown_variables,
            "typing_coverage": self.typing_coverage,
            "bindings": [
                {
                    "name": b.name,
                    "scope": b.scope,
                    "type": str(b.inferred_type),
                    "confidence": b.inferred_type.confidence,
                    "defined_line": b.defined_line,
                }
                for b in self.all_bindings
            ],
            "function_returns": {
                name: str(ret) for name, ret in self.function_returns.items()
            },
        }


class TypeInference(ast.NodeVisitor):
    """
    Perform lightweight deterministic type inference on Python AST.

    Infers types for:
    - Literals (int, float, str, bool, None, list, dict, tuple, set)
    - Variable assignments (propagates types from RHS to LHS)
    - Binary operations (e.g., int + int -> int, int + float -> float)
    - Return statements (infers function return type)
    - Some builtin function calls (len -> int, str() -> str, etc.)

    Usage:
        inferencer = TypeInference()
        report = inferencer.analyze(source_code)
    """

    # Builtin functions with known return types
    BUILTIN_RETURN_TYPES = {
        'len': 'int',
        'str': 'str',
        'int': 'int',
        'float': 'float',
        'bool': 'bool',
        'list': 'list',
        'tuple': 'tuple',
        'dict': 'dict',
        'set': 'set',
        'sum': 'int',  # Simplified
        'min': 'Any',
        'max': 'Any',
        'abs': 'int',  # Simplified
        'round': 'int',  # Simplified
        'range': 'range',
        'enumerate': 'enumerate',
        'zip': 'zip',
        'map': 'map',
        'filter': 'filter',
        'type': 'type',
        'isinstance': 'bool',
        'hasattr': 'bool',
        'getattr': 'Any',
        'repr': 'str',
        'ord': 'int',
        'chr': 'str',
        'hex': 'str',
        'oct': 'str',
        'bin': 'str',
        'id': 'int',
        'hash': 'int',
        'dir': 'list',
        'vars': 'dict',
        'globals': 'dict',
        'locals': 'dict',
    }

    # Binary operation type rules
    BINARY_OP_RULES = {
        ('int', 'int'): {
            '+': 'int', '-': 'int', '*': 'int', '//': 'int', '%': 'int',
            '**': 'int', '<<': 'int', '>>': 'int', '&': 'int', '|': 'int', '^': 'int',
        },
        ('int', 'float'): {
            '+': 'float', '-': 'float', '*': 'float', '/': 'float', '//': 'float',
            '%': 'float', '**': 'float',
        },
        ('float', 'int'): {
            '+': 'float', '-': 'float', '*': 'float', '/': 'float', '//': 'float',
            '%': 'float', '**': 'float',
        },
        ('float', 'float'): {
            '+': 'float', '-': 'float', '*': 'float', '/': 'float', '//': 'float',
            '%': 'float', '**': 'float',
        },
        ('str', 'str'): {'+': 'str', '*': 'str'},
        ('list', 'list'): {'+': 'list'},
        ('int', 'int', 'cmp'): {'==': 'bool', '!=': 'bool', '<': 'bool', '>': 'bool', '<=': 'bool', '>=': 'bool'},
    }

    def __init__(self):
        self._reset_state()

    def _reset_state(self):
        """Reset internal state for new analysis."""
        self.bindings: Dict[str, TypeBinding] = {}
        self.function_returns: Dict[str, InferredType] = {}
        self.current_scope = "module"
        self.scope_stack = ["module"]
        self.return_types: List[InferredType] = []
        self.current_function: Optional[str] = None

    def analyze(self, source_code: str) -> TypeInferenceReport:
        """
        Analyze Python source code and return TypeInferenceReport.

        Args:
            source_code: Python source code as string

        Returns:
            TypeInferenceReport with inferred types
        """
        self._reset_state()

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return TypeInferenceReport(
                source_code=source_code,
                parse_successful=False,
                syntax_error=f"Line {e.lineno}: {e.msg}" if e.lineno else str(e),
            )

        self.visit(tree)

        # Build report
        bindings_by_scope: Dict[str, List[TypeBinding]] = {}
        for binding in self.bindings.values():
            if binding.scope not in bindings_by_scope:
                bindings_by_scope[binding.scope] = []
            bindings_by_scope[binding.scope].append(binding)

        typed = sum(1 for b in self.bindings.values() 
                   if b.inferred_type.type_name not in ("Unknown", "Any"))
        unknown = len(self.bindings) - typed

        return TypeInferenceReport(
            source_code=source_code,
            parse_successful=True,
            bindings_by_scope=bindings_by_scope,
            all_bindings=list(self.bindings.values()),
            function_returns=self.function_returns,
            total_variables=len(self.bindings),
            typed_variables=typed,
            unknown_variables=unknown,
        )

    def _make_key(self, name: str) -> str:
        """Create unique key for a variable in current scope."""
        return f"{self.current_scope}:{name}"

    def _add_binding(
        self,
        name: str,
        inferred_type: InferredType,
        line: int,
        source: str = "assignment",
    ):
        """Add or update a type binding."""
        key = self._make_key(name)
        
        if key not in self.bindings:
            binding = TypeBinding(
                name=name,
                scope=self.current_scope,
                inferred_type=inferred_type,
                defined_line=line,
                assigned_lines=[line],
                type_sources=[source],
            )
            self.bindings[key] = binding
        else:
            binding = self.bindings[key]
            binding.assigned_lines.append(line)
            binding.type_sources.append(source)
            # Update type if new info is more specific
            if inferred_type.confidence > binding.inferred_type.confidence:
                binding.inferred_type = inferred_type

    def _get_binding(self, name: str) -> Optional[TypeBinding]:
        """Get type binding for a name, searching enclosing scopes."""
        for i in range(len(self.scope_stack) - 1, -1, -1):
            scope = self.scope_stack[i]
            key = f"{scope}:{name}"
            if key in self.bindings:
                return self.bindings[key]
        return None

    def _infer_expr_type(self, node: ast.AST) -> InferredType:
        """Infer type from an expression node."""
        if isinstance(node, ast.Constant):
            return InferredType.from_literal(node.value, node.lineno)
        
        elif isinstance(node, ast.List):
            if node.elts:
                elem_types = [self._infer_expr_type(e) for e in node.elts]
                common = InferredType._common_type(elem_types)
                return InferredType(type_name="list", container_type=common.type_name,
                                   confidence=common.confidence, source_line=node.lineno)
            return InferredType(type_name="list", container_type="Any", 
                               confidence=0.5, source_line=node.lineno)
        
        elif isinstance(node, ast.Tuple):
            if node.elts:
                elem_types = [self._infer_expr_type(e) for e in node.elts]
                common = InferredType._common_type(elem_types)
                return InferredType(type_name="tuple", container_type=common.type_name,
                                   confidence=common.confidence, source_line=node.lineno)
            return InferredType(type_name="tuple", container_type="Any",
                               confidence=0.5, source_line=node.lineno)
        
        elif isinstance(node, ast.Dict):
            if node.keys:
                key_types = [self._infer_expr_type(k) for k in node.keys if k]
                val_types = [self._infer_expr_type(v) for v in node.values]
                common_key = InferredType._common_type(key_types)
                common_val = InferredType._common_type(val_types)
                return InferredType(
                    type_name="dict",
                    key_type=common_key.type_name,
                    value_type=common_val.type_name,
                    confidence=min(common_key.confidence, common_val.confidence),
                    source_line=node.lineno,
                )
            return InferredType(type_name="dict", key_type="Any", value_type="Any",
                               confidence=0.5, source_line=node.lineno)
        
        elif isinstance(node, ast.Set):
            if node.elts:
                elem_types = [self._infer_expr_type(e) for e in node.elts]
                common = InferredType._common_type(elem_types)
                return InferredType(type_name="set", container_type=common.type_name,
                                   confidence=common.confidence, source_line=node.lineno)
            return InferredType(type_name="set", container_type="Any",
                               confidence=0.5, source_line=node.lineno)
        
        elif isinstance(node, ast.Name):
            binding = self._get_binding(node.id)
            if binding:
                return binding.inferred_type
            return InferredType.unknown(node.lineno)
        
        elif isinstance(node, ast.BinOp):
            left_type = self._infer_expr_type(node.left)
            right_type = self._infer_expr_type(node.right)
            return self._infer_binary_op_type(left_type, right_type, node.op)
        
        elif isinstance(node, ast.UnaryOp):
            operand_type = self._infer_expr_type(node.operand)
            return self._infer_unary_op_type(operand_type, node.op)
        
        elif isinstance(node, ast.Compare):
            # Comparisons return bool
            return InferredType(type_name="bool", confidence=0.9, source_line=node.lineno)
        
        elif isinstance(node, ast.BoolOp):
            # and/or return type of operands (usually bool in context)
            return InferredType(type_name="bool", confidence=0.8, source_line=node.lineno)
        
        elif isinstance(node, ast.IfExp):  # Ternary: x if cond else y
            true_type = self._infer_expr_type(node.body)
            false_type = self._infer_expr_type(node.orelse)
            return InferredType._common_type([true_type, false_type])
        
        elif isinstance(node, ast.Call):
            return self._infer_call_type(node)
        
        elif isinstance(node, ast.Subscript):
            # x[i] - try to infer from container type
            value_type = self._infer_expr_type(node.value)
            if value_type.type_name == "list" and value_type.container_type:
                return InferredType(type_name=value_type.container_type, 
                                   confidence=value_type.confidence * 0.8,
                                   source_line=node.lineno)
            elif value_type.type_name == "dict" and value_type.value_type:
                return InferredType(type_name=value_type.value_type,
                                   confidence=value_type.confidence * 0.8,
                                   source_line=node.lineno)
            return InferredType.any_type(node.lineno)
        
        elif isinstance(node, ast.Attribute):
            # obj.attr - can't infer without knowing obj's type
            return InferredType.any_type(node.lineno)
        
        elif isinstance(node, ast.Lambda):
            return InferredType(type_name="callable", confidence=0.7, source_line=node.lineno)
        
        elif isinstance(node, ast.Comprehension):
            return InferredType.any_type(node.lineno)
        
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            # Comprehensions create their own scope
            return InferredType.any_type(node.lineno)
        
        elif isinstance(node, ast.DictComp):
            return InferredType.any_type(node.lineno)
        
        return InferredType.unknown(node.lineno if hasattr(node, 'lineno') else 0)

    def _infer_binary_op_type(
        self, 
        left: InferredType, 
        right: InferredType, 
        op: ast.AST
    ) -> InferredType:
        """Infer result type of binary operation."""
        op_name = type(op).__name__
        
        # Map operator class to symbol
        op_map = {
            'Add': '+', 'Sub': '-', 'Mult': '*', 'MatMult': '@',
            'Div': '/', 'Mod': '%', 'Pow': '**', 'LShift': '<<',
            'RShift': '>>', 'BitOr': '|', 'BitXor': '^', 'BitAnd': '&',
            'FloorDiv': '//',
        }
        op_sym = op_map.get(op_name, op_name)
        
        # Check for comparison operators (always return bool)
        cmp_ops = {'Eq': '==', 'NotEq': '!=', 'Lt': '<', 'LtE': '<=', 
                   'Gt': '>', 'GtE': '>=', 'Is': 'is', 'IsNot': 'is not',
                   'In': 'in', 'NotIn': 'not in'}
        if op_name in cmp_ops:
            return InferredType(type_name="bool", confidence=0.95, source_line=left.source_line)
        
        # Look up type rules
        key = (left.type_name, right.type_name)
        if key in self.BINARY_OP_RULES and op_sym in self.BINARY_OP_RULES[key]:
            result_type = self.BINARY_OP_RULES[key][op_sym]
            confidence = min(left.confidence, right.confidence) * 0.9
            return InferredType(type_name=result_type, confidence=confidence,
                               source_line=left.source_line)
        
        # Reverse lookup
        key_rev = (right.type_name, left.type_name)
        if key_rev in self.BINARY_OP_RULES and op_sym in self.BINARY_OP_RULES[key_rev]:
            result_type = self.BINARY_OP_RULES[key_rev][op_sym]
            confidence = min(left.confidence, right.confidence) * 0.9
            return InferredType(type_name=result_type, confidence=confidence,
                               source_line=left.source_line)
        
        return InferredType.any_type(left.source_line)

    def _infer_unary_op_type(
        self, 
        operand: InferredType, 
        op: ast.AST
    ) -> InferredType:
        """Infer result type of unary operation."""
        op_name = type(op).__name__
        
        if op_name == 'UAdd' or op_name == 'USub':
            # +x, -x preserves numeric type
            if operand.type_name in ('int', 'float'):
                return operand
            return InferredType(type_name="int", confidence=0.5, source_line=operand.source_line)
        
        elif op_name == 'Invert':
            # ~x for bitwise invert
            if operand.type_name == 'int':
                return operand
            return InferredType(type_name="int", confidence=0.5, source_line=operand.source_line)
        
        elif op_name == 'Not':
            # not x returns bool
            return InferredType(type_name="bool", confidence=0.95, source_line=operand.source_line)
        
        return InferredType.any_type(operand.source_line)

    def _infer_call_type(self, node: ast.Call) -> InferredType:
        """Infer return type of a function call."""
        func = node.func
        
        # Direct function call
        if isinstance(func, ast.Name):
            func_name = func.id
            
            # Check builtins
            if func_name in self.BUILTIN_RETURN_TYPES:
                ret_type = self.BUILTIN_RETURN_TYPES[func_name]
                if ret_type in ('list', 'tuple', 'dict', 'set'):
                    return InferredType(type_name=ret_type, confidence=0.8,
                                       source_line=node.lineno)
                return InferredType(type_name=ret_type, confidence=0.85,
                                   source_line=node.lineno)
            
            # Check if it's a known function binding
            binding = self._get_binding(func_name)
            if binding and binding.inferred_type.type_name == "callable":
                return InferredType.any_type(node.lineno)
        
        # Method call
        elif isinstance(func, ast.Attribute):
            # Common method patterns
            method_name = func.attr
            
            # String methods
            if method_name in ('lower', 'upper', 'strip', 'split', 'join', 
                              'replace', 'format', 'capitalize'):
                return InferredType(type_name="str", confidence=0.8, source_line=node.lineno)
            elif method_name in ('isdigit', 'isalpha', 'startswith', 'endswith'):
                return InferredType(type_name="bool", confidence=0.8, source_line=node.lineno)
            
            # List methods
            elif method_name in ('append', 'extend', 'sort', 'reverse', 'clear'):
                return InferredType(type_name="None", confidence=0.9, source_line=node.lineno)
            elif method_name == 'pop':
                return InferredType.any_type(node.lineno)
            elif method_name == 'copy':
                return InferredType(type_name="list", confidence=0.7, source_line=node.lineno)
            
            # Dict methods
            elif method_name in ('keys', 'values'):
                return InferredType.any_type(node.lineno)
            elif method_name == 'items':
                return InferredType(type_name="list", container_type="tuple",
                                   confidence=0.7, source_line=node.lineno)
            elif method_name in ('get', 'pop', 'setdefault'):
                return InferredType.any_type(node.lineno)
        
        return InferredType.any_type(node.lineno)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition."""
        func_name = node.name
        if len(self.scope_stack) > 1:
            func_name = f"{self.scope_stack[-1]}.{node.name}"
        
        self.scope_stack.append(func_name)
        old_scope = self.current_scope
        self.current_scope = func_name
        self.return_types = []
        self.current_function = func_name
        
        # Visit parameters with annotations
        args = node.args
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            if arg.annotation:
                ann_type = self._infer_expr_type(arg.annotation)
                self._add_binding(arg.arg, ann_type, arg.lineno, "parameter_annotation")
            else:
                self._add_binding(arg.arg, InferredType.any_type(arg.lineno), 
                                 arg.lineno, "parameter")
        
        # Visit decorators
        for decorator in node.decorator_list:
            self.visit(decorator)
        
        # Visit body
        for stmt in node.body:
            self.visit(stmt)
        
        # Record function return type
        if self.return_types:
            common_return = InferredType._common_type(self.return_types)
            self.function_returns[func_name] = common_return
        
        self.scope_stack.pop()
        self.current_scope = old_scope
        self.current_function = None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definition."""
        self.visit_FunctionDef(node)

    def visit_Return(self, node: ast.Return):
        """Visit return statement."""
        if node.value:
            ret_type = self._infer_expr_type(node.value)
            self.return_types.append(ret_type)
        else:
            self.return_types.append(InferredType(type_name="None", confidence=1.0,
                                                  source_line=node.lineno))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        """Visit assignment statement."""
        value_type = self._infer_expr_type(node.value)
        
        for target in node.targets:
            self._visit_assignment_target(target, value_type, node.lineno)
        
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Visit annotated assignment."""
        if node.annotation:
            ann_type = self._infer_expr_type(node.annotation)
        else:
            ann_type = InferredType.any_type(node.lineno)
        
        if node.value:
            value_type = self._infer_expr_type(node.value)
            # Use annotation type if available, otherwise value type
            final_type = ann_type if ann_type.confidence >= value_type.confidence else value_type
        else:
            final_type = ann_type
        
        if node.target:
            self._visit_assignment_target(node.target, final_type, node.lineno)
        
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        """Visit augmented assignment."""
        if isinstance(node.target, ast.Name):
            current_type = self._infer_expr_type(node.target)
            value_type = self._infer_expr_type(node.value)
            result_type = self._infer_binary_op_type(current_type, value_type, node.op)
            self._add_binding(node.target.id, result_type, node.lineno, "augmented_assign")
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        """Visit for loop."""
        iter_type = self._infer_expr_type(node.iter)
        
        # Infer loop variable type from iterable
        if iter_type.type_name == "list" and iter_type.container_type:
            loop_var_type = InferredType(type_name=iter_type.container_type,
                                        confidence=iter_type.confidence * 0.9,
                                        source_line=node.lineno)
        elif iter_type.type_name == "str":
            loop_var_type = InferredType(type_name="str", confidence=0.9,
                                        source_line=node.lineno)
        elif iter_type.type_name == "range":
            loop_var_type = InferredType(type_name="int", confidence=0.9,
                                        source_line=node.lineno)
        else:
            loop_var_type = InferredType.any_type(node.lineno)
        
        self._visit_assignment_target(node.target, loop_var_type, node.lineno)
        self.generic_visit(node)

    def visit_With(self, node: ast.With):
        """Visit with statement."""
        for item in node.items:
            if item.optional_vars:
                # Context managers typically yield something
                ctx_type = self._infer_expr_type(item.context_expr)
                var_type = InferredType.any_type(node.lineno)
                self._visit_assignment_target(item.optional_vars, var_type, node.lineno)
        self.generic_visit(node)

    def _visit_assignment_target(
        self, 
        target: ast.AST, 
        value_type: InferredType, 
        line: int
    ):
        """Process assignment target (handles tuples, lists, attributes)."""
        if isinstance(target, ast.Name):
            self._add_binding(target.id, value_type, line, "assignment")
        elif isinstance(target, (ast.Tuple, ast.List)):
            # Unpacking - distribute type or use Any
            num_elts = len(target.elts)
            if value_type.type_name == "tuple" and value_type.container_type:
                # Tuple unpacking
                for elt in target.elts:
                    self._visit_assignment_target(elt, value_type, line)
            elif value_type.type_name == "list" and value_type.container_type:
                # List unpacking
                elem_type = InferredType(type_name=value_type.container_type,
                                        confidence=value_type.confidence * 0.8,
                                        source_line=line)
                for elt in target.elts:
                    self._visit_assignment_target(elt, elem_type, line)
            else:
                # Unknown container - assign Any to each element
                for elt in target.elts:
                    self._visit_assignment_target(elt, InferredType.any_type(line), line)
        elif isinstance(target, ast.Starred):
            self._visit_assignment_target(target.value, value_type, line)
        # Ignore attribute assignments

    def generic_visit(self, node):
        """Default visitor for unhandled node types."""
        for child in ast.iter_child_nodes(node):
            self.visit(child)
