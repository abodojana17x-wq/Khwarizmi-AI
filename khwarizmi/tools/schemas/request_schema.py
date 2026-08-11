"""
Khwarizmi Tool Request Data Schemas.

Defines structured data contracts decoupling the neural core from deterministic
local verification and symbolic tools as specified in Section 2 and Section 3 of
the Khwarizmi AI Blueprint.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class ToolVerificationRequest:
    """
    Structured request contract for optional deterministic tools.

    Attributes:
        tool_name: Identifies target tool ("python_brain" or "project_planner").
        payload: String source code (for python_brain) or structured plan data (for project_planner).
        metadata: Optional context metadata (e.g., confidence score, pathway name).
    """
    tool_name: str
    payload: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolVerificationResult:
    """
    Structured response contract from optional deterministic tools.

    Attributes:
        success: Whether the symbolic verification succeeded without errors.
        tool_name: Name of tool that executed.
        diagnostics: Structured dictionary of detected issues, AST stats, or DAG warnings.
        execution_overhead_ms: Milliseconds spent executing the tool.
    """
    success: bool
    tool_name: str
    diagnostics: Dict[str, Any]
    execution_overhead_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
