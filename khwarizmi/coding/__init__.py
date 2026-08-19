"""
Khwarizmi Coding Module — Phase 1 Static Analysis & Code Generation Tools.

This module provides offline, deterministic static analysis and code generation
capabilities for Python code:

- DataFlowAnalyzer: Track variable definitions/uses, detect unused variables,
  undefined references, and dead assignments.
- ControlFlowGraph: Build CFG from AST with basic blocks and edges, compute
  cyclomatic complexity, export adjacency dict and text visualization.
- TypeInference: Lightweight deterministic local type inference for literals,
  assignments, binary ops, and returns.
- ExecutionSandbox: Execute generated code locally with hard timeout, memory cap,
  no network, restricted builtins -> SandboxResult.
- MultiStageGenerator: Pipeline design -> interface -> implementation -> tests
  -> refactor using reasoning module between stages.

All tools are 100% offline, stdlib only, and designed to run on <4GB RAM
consumer CPU hardware.
"""

from .data_flow_analyzer import DataFlowAnalyzer, DataFlowReport
from .control_flow_graph import ControlFlowGraph, CFGNode, CFGEdge
from .type_inference import TypeInference, InferredType
from .execution_sandbox import ExecutionSandbox, SandboxResult
from .multi_stage_generator import MultiStageGenerator, StageResult

__all__ = [
    "DataFlowAnalyzer",
    "DataFlowReport",
    "ControlFlowGraph",
    "CFGNode",
    "CFGEdge",
    "TypeInference",
    "InferredType",
    "ExecutionSandbox",
    "SandboxResult",
    "MultiStageGenerator",
    "StageResult",
]
