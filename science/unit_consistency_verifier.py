"""Deterministic SI dimensional-analysis verifier."""
from __future__ import annotations

from dataclasses import dataclass
import ast
from typing import Dict

BASE = ("kg", "m", "s", "A", "K", "mol", "cd")
Dim = Dict[str, int]

UNIT_DIMS: dict[str, Dim] = {
    "": {}, "1": {}, "kg": {"kg": 1}, "m": {"m": 1}, "s": {"s": 1}, "A": {"A": 1}, "K": {"K": 1}, "mol": {"mol": 1}, "cd": {"cd": 1},
    "Hz": {"s": -1}, "N": {"kg": 1, "m": 1, "s": -2}, "J": {"kg": 1, "m": 2, "s": -2}, "W": {"kg": 1, "m": 2, "s": -3},
    "Pa": {"kg": 1, "m": -1, "s": -2}, "C": {"A": 1, "s": 1}, "V": {"kg": 1, "m": 2, "s": -3, "A": -1},
}
DEFAULT_SYMBOLS: dict[str, Dim] = {
    "F": UNIT_DIMS["N"], "m": UNIT_DIMS["kg"], "a": {"m": 1, "s": -2}, "v": {"m": 1, "s": -1}, "u": {"m": 1, "s": -1}, "t": {"s": 1},
    "d": {"m": 1}, "x": {"m": 1}, "s": {"m": 1}, "E": UNIT_DIMS["J"], "P": UNIT_DIMS["W"], "p": {"kg": 1, "m": 1, "s": -1},
    "rho": {"kg": 1, "m": -3}, "V": {"m": 3}, "f": {"s": -1}, "lambda": {"m": 1}, "c": {"m": 1, "s": -1}, "g": {"m": 1, "s": -2},
}

@dataclass(frozen=True)
class UnitVerdict:
    ok: bool
    lhs_dimensions: Dim
    rhs_dimensions: Dim
    details: str


def _clean(dim: Dim) -> Dim:
    return {k: v for k, v in sorted(dim.items()) if v}

def _add(a: Dim, b: Dim, scale: int = 1) -> Dim:
    out = dict(a)
    for k, v in b.items(): out[k] = out.get(k, 0) + scale * v
    return _clean(out)

def _mul_power(a: Dim, p: int) -> Dim:
    return _clean({k: v * p for k, v in a.items()})

def parse_unit(unit: str) -> Dim:
    expr = unit.replace("^", "**").replace(" ", "*") or "1"
    return _eval(ast.parse(expr, mode="eval").body, UNIT_DIMS)

def _eval(node: ast.AST, symbols: dict[str, Dim]) -> Dim:
    if isinstance(node, ast.Constant): return {}
    if isinstance(node, ast.Name):
        if node.id not in symbols: raise ValueError(f"Unknown symbol/unit '{node.id}'")
        return symbols[node.id]
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Mult): return _add(_eval(node.left, symbols), _eval(node.right, symbols))
        if isinstance(node.op, ast.Div): return _add(_eval(node.left, symbols), _eval(node.right, symbols), -1)
        if isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, int): raise ValueError("Powers must be integers")
            return _mul_power(_eval(node.left, symbols), node.right.value)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            left, right = _eval(node.left, symbols), _eval(node.right, symbols)
            if left != right: raise ValueError(f"Cannot add/subtract mismatched dimensions {left} and {right}")
            return left
    if isinstance(node, ast.UnaryOp): return _eval(node.operand, symbols)
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")

def verify_equation(equation: str, symbol_dimensions: dict[str, Dim] | None = None) -> UnitVerdict:
    if "=" not in equation: raise ValueError("Equation must contain '='")
    symbols = dict(DEFAULT_SYMBOLS)
    if symbol_dimensions: symbols.update(symbol_dimensions)
    lhs, rhs = [part.strip().replace("^", "**") for part in equation.split("=", 1)]
    lhs_dim = _eval(ast.parse(lhs, mode="eval").body, symbols)
    rhs_dim = _eval(ast.parse(rhs, mode="eval").body, symbols)
    ok = lhs_dim == rhs_dim
    return UnitVerdict(ok, lhs_dim, rhs_dim, "dimensions match" if ok else f"dimension mismatch: lhs {lhs_dim} vs rhs {rhs_dim}")
