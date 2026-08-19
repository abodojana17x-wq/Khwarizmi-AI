"""
Khwarizmi Integration Package — Agentic System Integration Layer.

This package provides the integration layer that turns core + tools into
ONE agentic system:
    - tool_registry: Central registry for all callable tools
    - cognitive_router: Domain task routing (CODE/SCIENCE/ART/CREATIVITY/GENERAL)
    - agentic_loop: Full agentic execution loop with think-pick-execute-verify

All components are designed for offline-first operation with <4GB RAM constraints.
"""

from .tool_registry import (
    ToolSchema,
    ToolResult,
    BaseTool,
    ExecutionSandboxTool,
    UnitConsistencyVerifierTool,
    AestheticScorerTool,
    ScamperEngineTool,
    DataFlowAnalyzerTool,
    ControlFlowGraphTool,
    TypeInferenceTool,
    ToolRegistry,
    get_registry,
    invoke_tool,
)

from .cognitive_router import (
    DomainResult,
    CognitiveRouter,
    get_router,
    route_task,
)

from .agentic_loop import (
    AgentStep,
    AgenticLoopResult,
    ProcessRewardModel,
    TestTimeComputeScaling,
    AgenticLoop,
    run_agentic_loop,
    create_loop,
)

__all__ = [
    # Tool Registry
    "ToolSchema",
    "ToolResult",
    "BaseTool",
    "ExecutionSandboxTool",
    "UnitConsistencyVerifierTool",
    "AestheticScorerTool",
    "ScamperEngineTool",
    "DataFlowAnalyzerTool",
    "ControlFlowGraphTool",
    "TypeInferenceTool",
    "ToolRegistry",
    "get_registry",
    "invoke_tool",
    # Cognitive Router
    "DomainResult",
    "CognitiveRouter",
    "get_router",
    "route_task",
    # Agentic Loop
    "AgentStep",
    "AgenticLoopResult",
    "ProcessRewardModel",
    "TestTimeComputeScaling",
    "AgenticLoop",
    "run_agentic_loop",
    "create_loop",
]
