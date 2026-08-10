"""Structural explanation of analyzed Python programs.

The explainer turns the structured model produced by the analyzer into
readable, human language describing what a program does *structurally*:
what it defines, what control flow it uses, what it imports, how complex
it is, and what issues were found.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Counter, List

if TYPE_CHECKING:  # pragma: no cover
    from .analyzer import AnalysisResult
    from .model import ClassInfo, FunctionInfo


class PythonExplainer:
    """Produce human-readable structural explanations."""

    def explain(self, result: "AnalysisResult") -> str:
        """Explain an analysis result in plain English."""
        if not result.parse_successful or result.syntax_error is not None:
            error = result.syntax_error
            if error is None:
                return "The source code could not be analyzed."
            text = (
                f"The source code could not be parsed as Python.\n"
                f"{error.error_type} at line {error.line}, column {error.column}: "
                f"{error.message}"
            )
            if error.text:
                text += f"\nOffending line: {error.text}"
            return text

        lines: List[str] = []
        structure = result.structure
        if structure is not None and structure.module_docstring:
            first_line = structure.module_docstring.strip().splitlines()[0]
            lines.append(f"Module purpose: {first_line}")

        lines.append(
            f"The program defines {len(result.functions)} function(s), "
            f"{len(result.classes)} class(es), and imports "
            f"{len(result.imports)} module(s)."
        )
        if structure is not None:
            imports = ", ".join(structure.modules_imported) or "none"
            lines.append(
                f"Imported modules: {imports}. Lines of code: "
                f"{structure.lines_of_code}."
            )

        for fn in result.functions:
            lines.append(self.explain_function(fn, result))
        for cls in result.classes:
            lines.append(self.explain_class(cls, result))

        if result.complexity is not None:
            lines.append(
                f"Overall cyclomatic complexity: {result.complexity.cyclomatic} "
                f"({result.complexity.label}); maximum nesting depth "
                f"{result.complexity.nesting_depth}."
            )

        if result.issues:
            lines.append("")
            lines.append(self.explain_issues(result))
        return "\n".join(lines)

    def explain_function(self, fn: "FunctionInfo", result: "AnalysisResult") -> str:
        """Explain a single function."""
        params = ", ".join(param.name for param in fn.parameters) or "no parameters"
        returns = ", ".join(fn.returns) if fn.returns else "unknown"
        description = (
            f"Function '{fn.name}' (lines {fn.line}-{fn.end_line}) accepts {params}; "
            f"returns {returns}; cyclomatic complexity {fn.complexity}."
        )
        if fn.is_async:
            description = description.replace("Function", "Async function")
        if fn.is_method:
            description += f" It is a method of class '{fn.class_name}'."
        if fn.docstring:
            description += f" Purpose: {fn.docstring.strip().splitlines()[0]}"
        flow_kinds = [
            cf.kind for cf in result.control_flow if cf.scope == fn.scope_name
        ]
        if flow_kinds:
            description += f" Control flow: {self._flow_summary(flow_kinds)}."
        if fn.raises:
            description += f" May raise: {', '.join(fn.raises)}."
        return description

    def explain_class(self, cls: "ClassInfo", result: "AnalysisResult") -> str:
        """Explain a single class."""
        bases = ", ".join(cls.bases) if cls.bases else "object"
        methods = ", ".join(method.name for method in cls.methods) or "none"
        description = (
            f"Class '{cls.name}' (lines {cls.line}-{cls.end_line}) inherits from "
            f"{bases}; methods: {methods}; class variables: "
            f"{len(cls.class_variables)}; instance variables: "
            f"{len(cls.instance_variables)}."
        )
        if cls.is_dataclass:
            description += " It is a dataclass."
        if cls.docstring:
            description += f" Purpose: {cls.docstring.strip().splitlines()[0]}"
        return description

    def explain_issues(self, result: "AnalysisResult") -> str:
        """Format the detected issues as text."""
        if not result.issues:
            return "No issues detected."
        lines = ["Analysis issues:"]
        for issue in result.issues:
            location = f"line {issue.line}"
            suggestion = f" Hint: {issue.suggestion}" if issue.suggestion else ""
            lines.append(f"  [{issue.severity}] {location}: {issue.message}{suggestion}")
        return "\n".join(lines)

    @staticmethod
    def _flow_summary(kinds: List[str]) -> str:
        counts = Counter(kinds)
        parts: List[str] = []
        if counts.get("if"):
            parts.append(f"{counts['if']} conditional(s)")
        for loop_kind in ("for", "async_for", "while"):
            if counts.get(loop_kind):
                parts.append(f"{counts[loop_kind]} {loop_kind} loop(s)")
        if counts.get("try"):
            parts.append(f"{counts['try']} try block(s)")
        if counts.get("with"):
            parts.append(f"{counts['with']} with block(s)")
        if counts.get("match"):
            parts.append(f"{counts['match']} match statement(s)")
        if counts.get("return"):
            parts.append(f"{counts['return']} return(s)")
        if counts.get("raise"):
            parts.append(f"{counts['raise']} raise(s)")
        return ", ".join(parts) if parts else "linear"


__all__ = ["PythonExplainer"]
