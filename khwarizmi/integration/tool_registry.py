"""
Khwarizmi Integration Tool Registry — Unified Tool Interface.

This module provides a centralized registry for all callable tools in the
Khwarizmi agentic system. Each tool exposes a consistent interface:
    - name: Human-readable tool identifier
    - args_schema: Dictionary describing expected arguments
    - invoke(): Execute the tool with given arguments
    - safety_check(): Validate inputs before execution

Tools registered:
    - ExecutionSandbox (khwarizmi.coding)
    - UnitConsistencyVerifier (science)
    - AestheticScorer (art)
    - ScamperEngine (creativity)
    - DataFlowAnalyzer (khwarizmi.coding)
    - ControlFlowGraph (khwarizmi.coding)
    - TypeInference (khwarizmi.coding)

All tools are designed for offline-first operation with <4GB RAM constraints.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Type
import sys
import os

# Add workspace to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ToolSchema:
    """Schema definition for a tool's arguments."""
    name: str
    arg_type: str  # "str", "int", "float", "dict", "list", "code"
    required: bool = True
    description: str = ""
    default: Any = None


@dataclass
class ToolResult:
    """Standardized result from tool invocation."""
    success: bool
    tool_name: str
    output: Any = None
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        return self.success and self.error_message == ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "output": self.output,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class BaseTool(ABC):
    """Abstract base class for all registered tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return human-readable tool name."""
        pass
    
    @property
    @abstractmethod
    def args_schema(self) -> Dict[str, ToolSchema]:
        """Return schema describing expected arguments."""
        pass
    
    @abstractmethod
    def invoke(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        pass
    
    @abstractmethod
    def safety_check(self, **kwargs) -> tuple[bool, str]:
        """
        Validate inputs before execution.
        Returns (is_safe, error_message).
        """
        pass


class ExecutionSandboxTool(BaseTool):
    """Wrapper for khwarizmi.coding.execution_sandbox.ExecutionSandbox."""
    
    @property
    def name(self) -> str:
        return "ExecutionSandbox"
    
    @property
    def args_schema(self) -> Dict[str, ToolSchema]:
        return {
            "code": ToolSchema("code", "code", True, "Python source code to execute"),
            "timeout": ToolSchema("timeout", "float", False, "Max execution time in seconds", 5.0),
            "max_memory_mb": ToolSchema("max_memory_mb", "int", False, "Memory limit in MB", 100),
        }
    
    def invoke(self, **kwargs) -> ToolResult:
        from khwarizmi.coding.execution_sandbox import ExecutionSandbox
        
        code = kwargs.get("code", "")
        timeout = kwargs.get("timeout", 5.0)
        max_memory_mb = kwargs.get("max_memory_mb", 100)
        
        try:
            sandbox = ExecutionSandbox(timeout=timeout, max_memory_mb=max_memory_mb)
            result = sandbox.execute(code)
            return ToolResult(
                success=result.is_valid,
                tool_name=self.name,
                output=result.to_dict(),
                metadata={"timed_out": result.timed_out, "memory_exceeded": result.memory_exceeded},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error_message=str(e),
            )
    
    def safety_check(self, **kwargs) -> tuple[bool, str]:
        code = kwargs.get("code", "")
        if not code or not isinstance(code, str):
            return False, "Code must be a non-empty string"
        if len(code) > 100000:
            return False, "Code exceeds maximum length (100KB)"
        # Check for dangerous patterns
        dangerous = ["__import__('os'", "subprocess", "socket", "eval(", "exec("]
        for pattern in dangerous:
            if pattern in code:
                return False, f"Dangerous pattern detected: {pattern}"
        return True, ""


class UnitConsistencyVerifierTool(BaseTool):
    """Wrapper for science.unit_consistency_verifier."""
    
    @property
    def name(self) -> str:
        return "UnitConsistencyVerifier"
    
    @property
    def args_schema(self) -> Dict[str, ToolSchema]:
        return {
            "equation": ToolSchema("equation", "str", True, "Physics equation to verify (e.g., 'F = m * a')"),
            "symbol_dimensions": ToolSchema("symbol_dimensions", "dict", False, "Custom symbol dimensions", {}),
        }
    
    def invoke(self, **kwargs) -> ToolResult:
        from science.unit_consistency_verifier import verify_equation
        
        equation = kwargs.get("equation", "")
        symbol_dimensions = kwargs.get("symbol_dimensions", {})
        
        try:
            verdict = verify_equation(equation, symbol_dimensions if symbol_dimensions else None)
            return ToolResult(
                success=verdict.ok,
                tool_name=self.name,
                output={
                    "lhs_dimensions": verdict.lhs_dimensions,
                    "rhs_dimensions": verdict.rhs_dimensions,
                    "details": verdict.details,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error_message=str(e),
            )
    
    def safety_check(self, **kwargs) -> tuple[bool, str]:
        equation = kwargs.get("equation", "")
        if not equation or not isinstance(equation, str):
            return False, "Equation must be a non-empty string"
        if "=" not in equation:
            return False, "Equation must contain '=' operator"
        if len(equation) > 1000:
            return False, "Equation exceeds maximum length"
        return True, ""


class AestheticScorerTool(BaseTool):
    """Wrapper for art.aesthetic_scorer."""
    
    @property
    def name(self) -> str:
        return "AestheticScorer"
    
    @property
    def args_schema(self) -> Dict[str, ToolSchema]:
        return {
            "description": ToolSchema("description", "dict", True, "Art brief with composition/color parameters"),
        }
    
    def invoke(self, **kwargs) -> ToolResult:
        from art.aesthetic_scorer import score_aesthetics
        
        description = kwargs.get("description", {})
        
        try:
            report = score_aesthetics(description)
            return ToolResult(
                success=True,
                tool_name=self.name,
                output={
                    "composition_score": report.composition_score,
                    "color_score": report.color_score,
                    "overall_score": report.overall_score,
                    "findings": report.findings,
                    "suggestions": report.suggestions,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error_message=str(e),
            )
    
    def safety_check(self, **kwargs) -> tuple[bool, str]:
        description = kwargs.get("description", {})
        if not isinstance(description, dict):
            return False, "Description must be a dictionary"
        return True, ""


class ScamperEngineTool(BaseTool):
    """Wrapper for creativity.scamper_engine."""
    
    @property
    def name(self) -> str:
        return "ScamperEngine"
    
    @property
    def args_schema(self) -> Dict[str, ToolSchema]:
        return {
            "brief": ToolSchema("brief", "str", True, "Creative brief or problem statement"),
        }
    
    def invoke(self, **kwargs) -> ToolResult:
        from creativity.scamper_engine import generate_scamper
        
        brief = kwargs.get("brief", "")
        
        try:
            report = generate_scamper(brief)
            candidates = [
                {
                    "technique": c.technique,
                    "idea": c.idea,
                    "novelty": c.novelty,
                    "usefulness": c.usefulness,
                    "rationale": c.rationale,
                }
                for c in report.candidates
            ]
            return ToolResult(
                success=report.safety_verdict == "allowed",
                tool_name=self.name,
                output={
                    "safety_verdict": report.safety_verdict,
                    "candidates": candidates,
                    "message": report.message,
                },
                metadata={"candidate_count": len(candidates)},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error_message=str(e),
            )
    
    def safety_check(self, **kwargs) -> tuple[bool, str]:
        brief = kwargs.get("brief", "")
        if not brief or not isinstance(brief, str):
            return False, "Brief must be a non-empty string"
        if len(brief) > 10000:
            return False, "Brief exceeds maximum length"
        return True, ""


class DataFlowAnalyzerTool(BaseTool):
    """Wrapper for khwarizmi.coding.data_flow_analyzer."""
    
    @property
    def name(self) -> str:
        return "DataFlowAnalyzer"
    
    @property
    def args_schema(self) -> Dict[str, ToolSchema]:
        return {
            "source_code": ToolSchema("source_code", "code", True, "Python source code to analyze"),
        }
    
    def invoke(self, **kwargs) -> ToolResult:
        from khwarizmi.coding.data_flow_analyzer import DataFlowAnalyzer
        
        source_code = kwargs.get("source_code", "")
        
        try:
            analyzer = DataFlowAnalyzer()
            report = analyzer.analyze(source_code)
            return ToolResult(
                success=report.is_valid,
                tool_name=self.name,
                output=report.to_dict(),
                metadata={
                    "total_variables": report.total_variables,
                    "total_unused": report.total_unused,
                    "total_undefined": report.total_undefined,
                    "total_dead": report.total_dead,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error_message=str(e),
            )
    
    def safety_check(self, **kwargs) -> tuple[bool, str]:
        source_code = kwargs.get("source_code", "")
        if not source_code or not isinstance(source_code, str):
            return False, "Source code must be a non-empty string"
        if len(source_code) > 100000:
            return False, "Source code exceeds maximum length"
        return True, ""


class ControlFlowGraphTool(BaseTool):
    """Wrapper for khwarizmi.coding.control_flow_graph."""
    
    @property
    def name(self) -> str:
        return "ControlFlowGraph"
    
    @property
    def args_schema(self) -> Dict[str, ToolSchema]:
        return {
            "source_code": ToolSchema("source_code", "code", True, "Python source code to analyze"),
        }
    
    def invoke(self, **kwargs) -> ToolResult:
        from khwarizmi.coding.control_flow_graph import ControlFlowGraph
        
        source_code = kwargs.get("source_code", "")
        
        try:
            cfg = ControlFlowGraph()
            report = cfg.build(source_code)
            return ToolResult(
                success=report.parse_successful,
                tool_name=self.name,
                output=report.to_dict(),
                metadata={
                    "cyclomatic_complexity": report.cyclomatic_complexity,
                    "num_nodes": report.num_nodes,
                    "num_edges": report.num_edges,
                    "num_decision_points": report.num_decision_points,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error_message=str(e),
            )
    
    def safety_check(self, **kwargs) -> tuple[bool, str]:
        source_code = kwargs.get("source_code", "")
        if not source_code or not isinstance(source_code, str):
            return False, "Source code must be a non-empty string"
        if len(source_code) > 100000:
            return False, "Source code exceeds maximum length"
        return True, ""


class TypeInferenceTool(BaseTool):
    """Wrapper for khwarizmi.coding.type_inference."""
    
    @property
    def name(self) -> str:
        return "TypeInference"
    
    @property
    def args_schema(self) -> Dict[str, ToolSchema]:
        return {
            "source_code": ToolSchema("source_code", "code", True, "Python source code to analyze"),
        }
    
    def invoke(self, **kwargs) -> ToolResult:
        from khwarizmi.coding.type_inference import TypeInferenceEngine
        
        source_code = kwargs.get("source_code", "")
        
        try:
            engine = TypeInferenceEngine()
            report = engine.infer(source_code)
            return ToolResult(
                success=report.parse_successful,
                tool_name=self.name,
                output=report.to_dict(),
                metadata={
                    "total_inferences": report.total_inferences,
                    "high_confidence_count": report.high_confidence_count,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error_message=str(e),
            )
    
    def safety_check(self, **kwargs) -> tuple[bool, str]:
        source_code = kwargs.get("source_code", "")
        if not source_code or not isinstance(source_code, str):
            return False, "Source code must be a non-empty string"
        if len(source_code) > 100000:
            return False, "Source code exceeds maximum length"
        return True, ""


class ToolRegistry:
    """
    Central registry for all callable tools in the Khwarizmi agentic system.
    
    Usage:
        registry = ToolRegistry()
        tool = registry.get("ExecutionSandbox")
        result = tool.invoke(code="print('hello')")
    """
    
    DEFAULT_TOOLS: List[Type[BaseTool]] = [
        ExecutionSandboxTool,
        UnitConsistencyVerifierTool,
        AestheticScorerTool,
        ScamperEngineTool,
        DataFlowAnalyzerTool,
        ControlFlowGraphTool,
        TypeInferenceTool,
    ]
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_defaults()
    
    def _register_defaults(self) -> None:
        """Register all default tools."""
        for tool_class in self.DEFAULT_TOOLS:
            tool = tool_class()
            self._tools[tool.name] = tool
    
    def register(self, tool: BaseTool) -> None:
        """Register a custom tool instance."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def invoke(self, tool_name: str, **kwargs) -> ToolResult:
        """Invoke a tool by name with arguments."""
        tool = self.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error_message=f"Tool '{tool_name}' not found in registry",
            )
        
        # Safety check first
        is_safe, error_msg = tool.safety_check(**kwargs)
        if not is_safe:
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error_message=f"Safety check failed: {error_msg}",
            )
        
        return tool.invoke(**kwargs)
    
    def get_schema(self, tool_name: str) -> Optional[Dict[str, ToolSchema]]:
        """Get the argument schema for a tool."""
        tool = self.get(tool_name)
        return tool.args_schema if tool else None


# Singleton instance
_registry_instance: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry singleton."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ToolRegistry()
    return _registry_instance


def invoke_tool(tool_name: str, **kwargs) -> ToolResult:
    """Convenience function to invoke a tool via the global registry."""
    return get_registry().invoke(tool_name, **kwargs)
