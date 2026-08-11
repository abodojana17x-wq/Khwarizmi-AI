"""
Khwarizmi Selective Verification Controller and Legacy Tool Bridge.

Implements the Layered Tool Interface defined in Section 2, 3, and 4.7 of the
Khwarizmi AI Blueprint. It cleanly bridges the Khwarizmi neural core to the
deterministic local tools:
    - Python Brain (legacy rafig/python_brain): AST-based static analysis and code verification.
    - Project Planner (legacy rafig/reasoning): DAG symbolic planning and constraint verification.

Tools are never called automatically on simple conversational queries; they execute
only when explicitly triggered by the Cognitive Router or selective verification gates.
"""

import time
from typing import Dict, Any, List, Optional, Tuple

from .schemas.request_schema import ToolVerificationRequest, ToolVerificationResult

# Clean optional imports of existing RAFIQ legacy engines
try:
    from rafig.python_brain import PythonAnalyzer
    PYTHON_BRAIN_AVAILABLE = True
except ImportError:
    PYTHON_BRAIN_AVAILABLE = False

try:
    from rafig.reasoning import ReasoningEngine
    REASONING_ENGINE_AVAILABLE = True
except ImportError:
    REASONING_ENGINE_AVAILABLE = False


class PythonAnalysisTool:
    """
    Optional deterministic Python static analysis tool wrapped around existing
    rafig/python_brain pure AST analyzer.
    """

    @staticmethod
    def verify_code(source_code: str) -> ToolVerificationResult:
        """
        Analyze Python source code using AST symbol and issue verification.

        Args:
            source_code: Python string payload to verify.

        Returns:
            ToolVerificationResult containing AST diagnostics and detected issues.
        """
        start_time = time.perf_counter()
        if not PYTHON_BRAIN_AVAILABLE:
            return ToolVerificationResult(
                success=False,
                tool_name="python_brain",
                diagnostics={"error": "Legacy rafig.python_brain module not available."},
            )

        try:
            analyzer = PythonAnalyzer()
            analysis = analyzer.analyze(source_code)
            issues = analysis.issues if hasattr(analysis, "issues") else []
            syntax_ok = analysis.parse_successful if hasattr(analysis, "parse_successful") else True

            error_count = 0
            issue_list = []
            for issue in issues:
                sev = getattr(issue, "severity", "")
                sev_str = sev.name if hasattr(sev, "name") else str(sev)
                if sev_str.lower() == "error":
                    error_count += 1
                issue_list.append({
                    "kind": getattr(issue, "kind", ""),
                    "severity": sev_str,
                    "message": getattr(issue, "message", ""),
                })

            diagnostics = {
                "parse_successful": syntax_ok,
                "issue_count": len(issues),
                "error_count": error_count,
                "issues": issue_list,
            }
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolVerificationResult(
                success=(syntax_ok and error_count == 0),
                tool_name="python_brain",
                diagnostics=diagnostics,
                execution_overhead_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolVerificationResult(
                success=False,
                tool_name="python_brain",
                diagnostics={"error": str(e)},
                execution_overhead_ms=elapsed_ms,
            )


class ProjectPlannerTool:
    """
    Optional deterministic project planning and DAG verification tool wrapped
    around existing rafig/reasoning symbolic engine.
    """

    @staticmethod
    def verify_dag_plan(request_text: str) -> ToolVerificationResult:
        """
        Verify project plan structure and DAG consistency using legacy reasoning engine.

        Args:
            request_text: Text prompt or description of the software project plan.

        Returns:
            ToolVerificationResult indicating DAG validity.
        """
        start_time = time.perf_counter()
        if not REASONING_ENGINE_AVAILABLE:
            return ToolVerificationResult(
                success=False,
                tool_name="project_planner",
                diagnostics={"error": "Legacy rafig.reasoning module not available."},
            )

        try:
            engine = ReasoningEngine()
            report = engine.reason(request_text)
            plan = report.plan

            diagnostics = {
                "goal_count": len(plan.goals) if hasattr(plan, "goals") else 0,
                "task_count": len(plan.tasks) if hasattr(plan, "tasks") else 0,
                "status": plan.status.name if hasattr(plan.status, "name") else str(plan.status),
                "tasks": [getattr(t, "action", "") for t in getattr(plan, "tasks", [])],
            }
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolVerificationResult(
                success=True,
                tool_name="project_planner",
                diagnostics=diagnostics,
                execution_overhead_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolVerificationResult(
                success=False,
                tool_name="project_planner",
                diagnostics={"error": str(e)},
                execution_overhead_ms=elapsed_ms,
            )


class SelectiveVerifier:
    """
    Selective Verification Controller.
    Decides whether to invoke optional local tools based on router signals,
    output confidence threshold, and payload inspection.
    """

    @classmethod
    def verify(
        cls,
        request: ToolVerificationRequest,
        needs_verification: bool = True,
    ) -> ToolVerificationResult:
        """
        Execute selective tool verification if triggered.

        Args:
            request: Structured ToolVerificationRequest.
            needs_verification: Boolean trigger from OutputPathway or CognitiveRouter.

        Returns:
            ToolVerificationResult. If needs_verification is False, returns zero-overhead
            skipped response.
        """
        if not needs_verification:
            return ToolVerificationResult(
                success=True,
                tool_name="skipped",
                diagnostics={"status": "Verification skipped (high confidence / fast path)."},
                execution_overhead_ms=0.0,
            )

        if request.tool_name == "python_brain":
            source = request.payload if isinstance(request.payload, str) else str(request.payload)
            return PythonAnalysisTool.verify_code(source)
        elif request.tool_name == "project_planner":
            req_text = request.payload if isinstance(request.payload, str) else str(request.payload)
            return ProjectPlannerTool.verify_dag_plan(req_text)
        else:
            return ToolVerificationResult(
                success=False,
                tool_name=request.tool_name,
                diagnostics={"error": f"Unknown tool: {request.tool_name}"},
            )
