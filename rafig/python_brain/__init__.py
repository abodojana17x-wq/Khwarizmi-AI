"""RAFIQ Python Brain — a from-scratch Python intelligence engine.

Phase 07 of the RAFIQ project.  This package analyzes Python source code
using only the standard-library ``ast`` module and produces structured,
language-independent models: functions, classes, variables, imports,
scopes, symbol tables, control flow, type information, complexity, and
detected issues.  No code is executed and no external tools are used.
"""

from .analyzer import AnalysisResult, PythonAnalyzer
from .complexity import ComplexityInfo
from .explain import PythonExplainer
from .issues import Issue, IssueSeverity
from .model import (
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
)
from .parser import PythonParseError, PythonParser
from .types import TypeInfo, TypeInference

__version__ = "0.7.0"

__all__ = [
    "AnalysisResult",
    "CallInfo",
    "ClassInfo",
    "ComplexityInfo",
    "ControlFlowInfo",
    "ExceptionHandlerInfo",
    "FunctionInfo",
    "ImportInfo",
    "Issue",
    "IssueSeverity",
    "LoopInfo",
    "ParameterInfo",
    "ProgramStructure",
    "PythonAnalyzer",
    "PythonExplainer",
    "PythonParseError",
    "PythonParser",
    "ReturnInfo",
    "ScopeInfo",
    "StatementSummary",
    "SymbolEntry",
    "SymbolReference",
    "SymbolTable",
    "SyntaxIssue",
    "TypeInference",
    "TypeInfo",
    "VariableInfo",
]
