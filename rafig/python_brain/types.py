"""Basic type inference for the Python Brain.

This is a lightweight, purely static type inference layer.  It infers types
from literals, annotations, builtin calls, arithmetic operations, method
calls on common builtin types, comprehensions, and aggregated function
return statements.  It never executes code and it is intentionally simple:
when in doubt it reports ``unknown`` instead of guessing wildly.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .model import MODULE_SCOPE


@dataclass(slots=True)
class TypeInfo:
    """An inferred type with a confidence score and its source."""

    type_name: str | None
    confidence: float = 0.0
    source: str = "unknown"


# Builtin callables -> return type name.
_BUILTIN_CALLS: Dict[str, str] = {
    "int": "int",
    "float": "float",
    "str": "str",
    "bool": "bool",
    "bytes": "bytes",
    "bytearray": "bytearray",
    "complex": "complex",
    "list": "list",
    "tuple": "tuple",
    "dict": "dict",
    "set": "set",
    "frozenset": "frozenset",
    "range": "range",
    "open": "file",
    "input": "str",
    "print": "NoneType",
    "len": "int",
    "type": "type",
    "sorted": "list",
    "reversed": "iterator",
    "iter": "iterator",
    "next": "unknown",
    "isinstance": "bool",
    "issubclass": "bool",
    "format": "str",
    "repr": "str",
    "chr": "str",
    "ord": "int",
    "hex": "str",
    "oct": "str",
    "bin": "str",
    "id": "int",
    "hash": "int",
    "any": "bool",
    "all": "bool",
    "sum": "int",
    "min": "unknown",
    "max": "unknown",
    "abs": "unknown",
    "round": "int",
    "divmod": "tuple",
    "pow": "int",
    "enumerate": "enumerate",
    "zip": "zip",
    "map": "iterator",
    "filter": "iterator",
    "getattr": "unknown",
    "setattr": "NoneType",
    "hasattr": "bool",
    "delattr": "NoneType",
    "vars": "dict",
    "dir": "list",
    "globals": "dict",
    "locals": "dict",
    "super": "super",
    "property": "property",
    "staticmethod": "staticmethod",
    "classmethod": "classmethod",
    "compile": "code",
    "eval": "unknown",
    "exec": "NoneType",
    "callable": "bool",
    "memoryview": "memoryview",
    "slice": "slice",
    "object": "object",
    "breakpoint": "NoneType",
}

# Builtins used as plain values (references without a call).
_BUILTIN_VALUES: Dict[str, str] = {
    name: "type"
    for name in (
        "str int float bool bytes bytearray complex list tuple dict set frozenset "
        "range object type memoryview slice property super staticmethod classmethod"
    ).split()
}
_BUILTIN_VALUES.update(
    {
        name: "function"
        for name in (
            "len print input open sorted reversed iter next isinstance issubclass "
            "format repr chr ord hex oct bin id hash any all sum min max abs round "
            "divmod pow enumerate zip map filter getattr setattr hasattr delattr "
            "vars dir globals locals compile eval exec callable breakpoint"
        ).split()
    }
)

# Method tables for common builtin types: method name -> return type name.
_STR_METHODS: Dict[str, str] = {
    "upper": "str", "lower": "str", "title": "str", "capitalize": "str",
    "strip": "str", "lstrip": "str", "rstrip": "str", "replace": "str",
    "format": "str", "join": "str", "encode": "bytes", "casefold": "str",
    "zfill": "str", "ljust": "str", "rjust": "str", "center": "str",
    "removeprefix": "str", "removesuffix": "str", "expandtabs": "str",
    "split": "list", "rsplit": "list", "splitlines": "list", "partition": "tuple",
    "rpartition": "tuple",
    "find": "int", "rfind": "int", "index": "int", "rindex": "int", "count": "int",
    "startswith": "bool", "endswith": "bool", "isalpha": "bool",
    "isdigit": "bool", "isalnum": "bool", "isspace": "bool", "isupper": "bool",
    "islower": "bool", "isnumeric": "bool", "isdecimal": "bool", "isascii": "bool",
    "istitle": "bool", "isprintable": "bool", "isidentifier": "bool",
}

_LIST_METHODS: Dict[str, str] = {
    "append": "NoneType", "extend": "NoneType", "insert": "NoneType",
    "remove": "NoneType", "pop": "unknown", "sort": "NoneType",
    "reverse": "NoneType", "clear": "NoneType", "copy": "list",
    "count": "int", "index": "int",
}

_DICT_METHODS: Dict[str, str] = {
    "get": "unknown", "keys": "list", "values": "list", "items": "list",
    "pop": "unknown", "popitem": "tuple", "update": "NoneType",
    "setdefault": "unknown", "clear": "NoneType", "copy": "dict",
    "fromkeys": "dict",
}

_SET_METHODS: Dict[str, str] = {
    "add": "NoneType", "discard": "NoneType", "remove": "NoneType",
    "union": "set", "intersection": "set", "difference": "set",
    "symmetric_difference": "set", "update": "NoneType",
    "clear": "NoneType", "copy": "set", "pop": "unknown",
    "issubset": "bool", "issuperset": "bool", "isdisjoint": "bool",
}

_TUPLE_METHODS: Dict[str, str] = {"count": "int", "index": "int"}
_INT_METHODS: Dict[str, str] = {
    "bit_length": "int", "to_bytes": "bytes", "from_bytes": "int",
    "real": "int", "imag": "int", "numerator": "int", "denominator": "int",
    "conjugate": "int",
}
_FLOAT_METHODS: Dict[str, str] = {
    "is_integer": "bool", "as_integer_ratio": "tuple", "hex": "str", "fromhex": "float",
}
_BYTES_METHODS: Dict[str, str] = {
    "decode": "str", "hex": "str", "upper": "bytes", "lower": "bytes",
    "strip": "bytes", "count": "int", "index": "int", "find": "int",
    "replace": "bytes", "split": "list", "join": "bytes",
}
_FILE_METHODS: Dict[str, str] = {
    "read": "str", "readline": "str", "readlines": "list",
    "write": "int", "writelines": "NoneType", "close": "NoneType",
    "flush": "NoneType", "seek": "int", "tell": "int",
    "readinto": "int", "truncate": "int", "fileno": "int",
}

_METHODS: Dict[str, Dict[str, str]] = {
    "str": _STR_METHODS,
    "list": _LIST_METHODS,
    "dict": _DICT_METHODS,
    "set": _SET_METHODS,
    "tuple": _TUPLE_METHODS,
    "int": _INT_METHODS,
    "float": _FLOAT_METHODS,
    "bytes": _BYTES_METHODS,
    "file": _FILE_METHODS,
}

_NUMERIC_ORDER = {"bool": 0, "int": 1, "float": 2, "complex": 3}


class TypeInference:
    """Static, offline type inference over a parsed module."""

    def __init__(self) -> None:
        self._parents: Dict[str, str | None] = {}
        self._scope_kinds: Dict[str, str] = {}
        self._enclosing_class: Dict[str, str | None] = {}
        self._annotations: Dict[Tuple[str, str], str] = {}
        self._fixed: Dict[Tuple[str, str], TypeInfo] = {}
        self._parameters: Set[Tuple[str, str]] = set()
        self._assignments: Dict[Tuple[str, str], List[Tuple[int, ast.AST | None]]] = {}
        self._function_scopes_by_name: Dict[str, List[str]] = {}
        self._function_returns: Dict[str, List[ast.AST | None]] = {}
        self._instance_attrs: Dict[Tuple[str, str], TypeInfo] = {}
        self._cache: Dict[Tuple[str, str], TypeInfo] = {}
        self._expr_cache: Dict[int, TypeInfo] = {}

    # -- registration API (used by the analyzer walker) --------------------

    def register_scope(self, scope: str, parent: str | None, kind: str | None = None) -> None:
        self._parents[scope] = parent
        if kind is not None:
            self._scope_kinds[scope] = kind

    def register_enclosing_class(self, scope: str, class_scope: str | None) -> None:
        """Record which class a scope belongs to (for ``self`` resolution)."""
        self._enclosing_class[scope] = class_scope

    def register_instance_attribute(self, class_scope: str, attr: str, info: TypeInfo) -> None:
        """Register the inferred type of an instance/class attribute."""
        self._instance_attrs[(class_scope, attr)] = info

    def clear_cache(self) -> None:
        """Drop cached inferences (used after late registrations)."""
        self._cache.clear()
        self._expr_cache.clear()

    def register_annotation(self, scope: str, name: str, annotation: str) -> None:
        self._annotations[(scope, name)] = annotation

    def register_fixed_type(self, scope: str, name: str, info: TypeInfo) -> None:
        self._fixed[(scope, name)] = info

    def register_parameter(self, scope: str, name: str) -> None:
        self._parameters.add((scope, name))

    def register_assignment(
        self, scope: str, name: str, line: int, value: ast.AST | None
    ) -> None:
        self._assignments.setdefault((scope, name), []).append((line, value))

    def register_function(self, scope: str, name: str) -> None:
        self._function_scopes_by_name.setdefault(name, []).append(scope)
        self._function_returns.setdefault(scope, [])

    def register_return(self, scope: str, value: ast.AST | None) -> None:
        self._function_returns.setdefault(scope, []).append(value)

    # -- inference API -------------------------------------------------------

    def infer_variable(self, scope: str, name: str) -> TypeInfo:
        """Infer the static type of a variable binding in a scope."""
        key = (scope, name)
        if key in self._cache:
            return self._cache[key]
        # Mark in-progress to break cycles (e.g. ``x = x + 1``).
        self._cache[key] = TypeInfo(None, 0.0, "in-progress")
        info = self._infer_variable_uncached(scope, name)
        self._cache[key] = info
        return info

    def _infer_variable_uncached(self, scope: str, name: str) -> TypeInfo:
        key = (scope, name)

        if key in self._annotations:
            info = TypeInfo(self._annotations[key], 0.9, "annotation")
            self._cache[key] = info
            return info

        assignments = self._assignments.get(key)
        if assignments:
            types: List[str] = []
            last: TypeInfo | None = None
            for _, value in assignments:
                inferred = (
                    self.infer_expression(value, scope)
                    if value is not None
                    else TypeInfo(None, 0.0, "unknown")
                )
                if inferred.type_name:
                    types.append(inferred.type_name)
                last = inferred
            if types and all(t == types[-1] for t in types):
                info = TypeInfo(types[-1], 0.75, "assignment")
            else:
                info = TypeInfo(last.type_name if last else None, 0.4, "assignment")
            self._cache[key] = info
            return info

        if key in self._fixed:
            info = self._fixed[key]
            self._cache[key] = info
            return info

        if key in self._parameters:
            info = TypeInfo(None, 0.3, "parameter")
            self._cache[key] = info
            return info

        parent = self._parents.get(scope)
        if parent:
            outer = self.infer_variable(parent, name)
            if outer.type_name:
                return outer
        return TypeInfo(None, 0.0, "unknown")

    def infer_expression(self, node: ast.AST | None, scope: str) -> TypeInfo:
        """Infer the type of an arbitrary expression node."""
        if node is None:
            return TypeInfo(None, 0.0, "unknown")
        key = id(node)
        if key in self._expr_cache:
            return self._expr_cache[key]
        # Mark in-progress to break cyclic inference (e.g. ``x = x + 1``).
        self._expr_cache[key] = TypeInfo(None, 0.0, "in-progress")
        info = self._infer_expr(node, scope)
        self._expr_cache[key] = info
        return info

    def infer_instance_attribute(self, class_scope: str, attr: str) -> TypeInfo:
        """Return the registered type of an instance/class attribute."""
        info = self._instance_attrs.get((class_scope, attr))
        if info is not None:
            return info
        return TypeInfo(None, 0.0, "unknown")

    def infer_function_returns(self, scope: str) -> List[str]:
        """Aggregate the inferred return types of a function scope."""
        values = self._function_returns.get(scope)
        if not values:
            return ["NoneType"]
        types: List[str] = []
        for value in values:
            inferred = (
                self.infer_expression(value, scope)
                if value is not None
                else TypeInfo("NoneType", 1.0, "implicit")
            )
            name = inferred.type_name or "unknown"
            if name not in types:
                types.append(name)
        return types[:6]

    # -- internals ------------------------------------------------------------

    def _infer_expr(self, node: ast.AST, scope: str) -> TypeInfo:
        if isinstance(node, ast.Constant):
            return TypeInfo(self._constant_type(node.value), 1.0, "literal")
        if isinstance(node, ast.Name):
            return self._infer_name(node, scope)
        if isinstance(node, ast.List):
            return TypeInfo("list", 0.9, "literal")
        if isinstance(node, ast.Tuple):
            return TypeInfo("tuple", 0.9, "literal")
        if isinstance(node, ast.Set):
            return TypeInfo("set", 0.9, "literal")
        if isinstance(node, ast.Dict):
            return TypeInfo("dict", 0.9, "literal")
        if isinstance(node, ast.ListComp):
            return TypeInfo("list", 0.8, "comprehension")
        if isinstance(node, ast.SetComp):
            return TypeInfo("set", 0.8, "comprehension")
        if isinstance(node, ast.DictComp):
            return TypeInfo("dict", 0.8, "comprehension")
        if isinstance(node, ast.GeneratorExp):
            return TypeInfo("generator", 0.8, "comprehension")
        if isinstance(node, ast.Lambda):
            return TypeInfo("function", 0.9, "definition")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return TypeInfo("function", 0.9, "definition")
        if isinstance(node, ast.ClassDef):
            return TypeInfo("type", 0.9, "definition")
        if isinstance(node, ast.Import):
            return TypeInfo("module", 0.7, "import")
        if isinstance(node, ast.ImportFrom):
            return TypeInfo("module", 0.5, "import")
        if isinstance(node, ast.JoinedStr):
            return TypeInfo("str", 0.95, "literal")
        if isinstance(node, ast.BinOp):
            return self._infer_binop(node, scope)
        if isinstance(node, ast.UnaryOp):
            return self._infer_unary(node, scope)
        if isinstance(node, ast.Compare):
            return TypeInfo("bool", 0.9, "operation")
        if isinstance(node, ast.BoolOp):
            left = self.infer_expression(node.values[0], scope)
            right = self.infer_expression(node.values[-1], scope)
            if left.type_name and left.type_name == right.type_name:
                return TypeInfo(left.type_name, 0.5, "operation")
            return TypeInfo(None, 0.2, "unknown")
        if isinstance(node, ast.IfExp):
            yes = self.infer_expression(node.body, scope)
            no = self.infer_expression(node.orelse, scope)
            if yes.type_name and yes.type_name == no.type_name:
                return TypeInfo(yes.type_name, 0.6, "operation")
            return TypeInfo(None, 0.2, "unknown")
        if isinstance(node, ast.Subscript):
            return self._infer_subscript(node, scope)
        if isinstance(node, ast.Call):
            return self._infer_call(node, scope)
        if isinstance(node, ast.Attribute):
            return self._infer_attribute(node, scope)
        if isinstance(node, ast.Starred):
            return self.infer_expression(node.value, scope)
        if isinstance(node, ast.Await):
            return TypeInfo(None, 0.1, "unknown")
        return TypeInfo(None, 0.0, "unknown")

    def _infer_name(self, node: ast.Name, scope: str) -> TypeInfo:
        info = self.infer_variable(scope, node.id)
        if info.type_name:
            return info
        if node.id in _BUILTIN_VALUES:
            return TypeInfo(_BUILTIN_VALUES[node.id], 0.9, "builtin")
        return TypeInfo(None, 0.0, "unknown")

    def _infer_binop(self, node: ast.BinOp, scope: str) -> TypeInfo:
        left = self.infer_expression(node.left, scope)
        right = self.infer_expression(node.right, scope)
        lt, rt = left.type_name, right.type_name
        op = node.op

        if isinstance(op, ast.Add):
            if lt and lt == rt and lt in ("str", "list", "bytes", "tuple"):
                return TypeInfo(lt, 0.85, "operation")
            if self._is_numeric(lt) and self._is_numeric(rt):
                return TypeInfo(self._numeric_result(lt, rt), 0.85, "operation")
        elif isinstance(op, (ast.Sub, ast.Mult, ast.Pow, ast.Mod, ast.Div, ast.FloorDiv)):
            if self._is_numeric(lt) and self._is_numeric(rt):
                if isinstance(op, ast.Div):
                    return TypeInfo("float", 0.85, "operation")
                if isinstance(op, ast.FloorDiv):
                    result = "int" if lt in ("int", "bool") and rt in ("int", "bool") else "float"
                    return TypeInfo(result, 0.85, "operation")
                return TypeInfo(self._numeric_result(lt, rt), 0.85, "operation")
            if isinstance(op, ast.Mod) and lt == "str":
                return TypeInfo("str", 0.8, "operation")
            if isinstance(op, ast.Mult) and lt in ("str", "list", "tuple") and rt in ("int", "bool"):
                return TypeInfo(lt, 0.8, "operation")
        elif isinstance(op, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)):
            if self._is_numeric(lt) and self._is_numeric(rt):
                return TypeInfo("int", 0.8, "operation")
        return TypeInfo(None, 0.2, "unknown")

    def _infer_unary(self, node: ast.UnaryOp, scope: str) -> TypeInfo:
        if isinstance(node.op, ast.Not):
            return TypeInfo("bool", 0.9, "operation")
        operand = self.infer_expression(node.operand, scope)
        if isinstance(node.op, ast.Invert):
            if self._is_numeric(operand.type_name):
                return TypeInfo("int", 0.8, "operation")
        if isinstance(node.op, (ast.USub, ast.UAdd)) and self._is_numeric(operand.type_name):
            return TypeInfo(operand.type_name, 0.8, "operation")
        return TypeInfo(None, 0.2, "unknown")

    def _infer_subscript(self, node: ast.Subscript, scope: str) -> TypeInfo:
        base = self.infer_expression(node.value, scope)
        if base.type_name == "str":
            return TypeInfo("str", 0.8, "operation")
        if base.type_name == "bytes":
            return TypeInfo("int", 0.8, "operation")
        if base.type_name in ("list", "tuple", "set", "dict", "range", "frozenset"):
            return TypeInfo(None, 0.3, "unknown")
        return TypeInfo(None, 0.2, "unknown")

    def _infer_call(self, node: ast.Call, scope: str) -> TypeInfo:
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
            if name in _BUILTIN_CALLS:
                return TypeInfo(_BUILTIN_CALLS[name], 0.9, "builtin_call")
            for function_scope in self._function_scopes_by_name.get(name, []):
                if self._is_closure_visible(function_scope, scope):
                    return self._returns_for(function_scope)
            for function_scope in self._function_scopes_by_name.get(name, []):
                if self._is_same_module(function_scope, scope):
                    return self._returns_for(function_scope)
            return TypeInfo(None, 0.2, "unknown")
        if isinstance(func, ast.Attribute):
            receiver = self.infer_expression(func.value, scope)
            if receiver.type_name and receiver.type_name in _METHODS:
                result = _METHODS[receiver.type_name].get(func.attr)
                if result is not None:
                    return TypeInfo(result, 0.85, "method_call")
            # method call on self/cls -> look up the class's own method
            if isinstance(func.value, ast.Name) and func.value.id in {"self", "cls"}:
                class_scope = self._enclosing_class.get(scope)
                if class_scope:
                    method_scope = f"{class_scope}.{func.attr}"
                    if method_scope in self._function_returns:
                        return self._returns_for(method_scope)
            return TypeInfo(None, 0.2, "unknown")
        return TypeInfo(None, 0.1, "unknown")

    def _infer_attribute(self, node: ast.Attribute, scope: str) -> TypeInfo:
        receiver = self.infer_expression(node.value, scope)
        if isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}:
            class_scope = self._enclosing_class.get(scope)
            if class_scope:
                info = self._instance_attrs.get((class_scope, node.attr))
                if info is not None:
                    return info
        if receiver.type_name in _METHODS:
            return TypeInfo("method", 0.6, "attribute")
        return TypeInfo(None, 0.1, "unknown")

    def _returns_for(self, function_scope: str) -> TypeInfo:
        values = self._function_returns.get(function_scope, [])
        if not values:
            return TypeInfo("NoneType", 0.8, "function_call")
        types: List[str] = []
        for value in values:
            inferred = (
                self.infer_expression(value, function_scope)
                if value is not None
                else TypeInfo("NoneType", 1.0, "implicit")
            )
            if inferred.type_name and inferred.type_name not in types:
                types.append(inferred.type_name)
        if not types:
            return TypeInfo(None, 0.2, "unknown")
        if len(types) == 1:
            return TypeInfo(types[0], 0.7, "function_call")
        return TypeInfo(", ".join(types[:3]), 0.5, "function_call")

    def _is_closure_visible(self, target_scope: str, from_scope: str) -> bool:
        """True when ``target_scope`` encloses ``from_scope`` (closure lookup)."""
        current: str | None = from_scope
        while current is not None:
            if current == target_scope:
                return True
            current = self._parents.get(current)
        return False

    def _is_same_module(self, target_scope: str, from_scope: str) -> bool:
        """True when both scopes live in the same module (module-level function)."""
        def reaches_module(scope: str | None) -> bool:
            current = scope
            while current is not None:
                if current == MODULE_SCOPE:
                    return True
                current = self._parents.get(current)
            return False

        return reaches_module(target_scope) and reaches_module(from_scope)

    @staticmethod
    def _constant_type(value: object) -> str:
        if value is None:
            return "NoneType"
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, complex):
            return "complex"
        if isinstance(value, str):
            return "str"
        if isinstance(value, bytes):
            return "bytes"
        if isinstance(value, bytearray):
            return "bytearray"
        if value is Ellipsis:
            return "ellipsis"
        return "unknown"

    @staticmethod
    def _is_numeric(type_name: str | None) -> bool:
        return type_name in _NUMERIC_ORDER

    @staticmethod
    def _numeric_result(left: str, right: str) -> str:
        return left if _NUMERIC_ORDER[left] >= _NUMERIC_ORDER[right] else right


__all__ = ["TypeInfo", "TypeInference"]
