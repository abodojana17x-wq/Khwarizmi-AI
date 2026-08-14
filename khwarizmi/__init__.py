"""
Khwarizmi AI: System Architecture Core Namespace.

Version: 0.4.0-phase4
Provides foundational implementations of:
    - Khwarizmi State Cell (KSC) & Residual Blocks
    - Short-Term Working State & Utility-Gated Long-Term Persistent Memory
    - Cognitive Router & Compute Pathway Dispatcher
    - Sparse Mixture-of-Experts (MoE) Specialists
    - Adaptive Recurrent Reasoning Cycles (ARRC)
    - Output Pathway & Selective Verification Trigger
    - Layered Agent Orchestrator & Legacy Deterministic Tool Bridge
"""

__version__ = "0.4.0-phase4"

from .config import (
    KhwarizmiConfig,
    get_tiny_test_config,
    get_prototype_config,
    get_prototype_50m_config,
    get_prototype_150m_config,
    get_small_config,
    get_edge_config,
)
from .core import (
    SinusoidalPositionalEncoding,
    KhwarizmiEmbeddings,
    KhwarizmiStateCell,
    FeedForwardNetwork,
    KSCResidualBlock,
    KhwarizmiKSCPrototype,
    KSCPrototypeOutput,
    build_ksc_prototype,
    KhwarizmiDualMemoryPrototype,
    KhwarizmiDualMemoryOutput,
    OutputPathway,
    KhwarizmiModel,
    KhwarizmiOutput,
)
from .memory import (
    ShortTermWorkingState,
    MemoryGatingController,
    UtilityGatingPolicy,
    LongTermPersistentMemory,
    DualMemory,
    DualMemoryOutput,
    RETAIN,
    WRITE,
    UPDATE,
    FORGET,
    DECISION_NAMES,
)
from .routing import (
    CognitiveRouter,
    PathwayDispatcher,
    PathwayExecutionFlags,
)
from .experts import (
    SparseMoELayer,
    ExpertLayer,
    MoERoutingDecision,
    create_standard_specialists,
    SPECIALIZATION_NAMES,
)
from .reasoning import (
    AdaptiveComputeBlock,
    LatentReasoner,
)
from .tools import (
    ToolVerificationRequest,
    ToolVerificationResult,
    PythonAnalysisTool,
    ProjectPlannerTool,
    SelectiveVerifier,
)
from .agent import (
    InputSanitizer,
    SanitizedInputFrame,
    KhwarizmiAgentLoop,
    AgentResponseFrame,
)

__all__ = [
    "__version__",
    "KhwarizmiConfig",
    "get_tiny_test_config",
    "get_prototype_config",
    "get_prototype_50m_config",
    "get_prototype_150m_config",
    "get_small_config",
    "get_edge_config",
    "SinusoidalPositionalEncoding",
    "KhwarizmiEmbeddings",
    "KhwarizmiStateCell",
    "FeedForwardNetwork",
    "KSCResidualBlock",
    "KhwarizmiKSCPrototype",
    "KSCPrototypeOutput",
    "build_ksc_prototype",
    "KhwarizmiDualMemoryPrototype",
    "KhwarizmiDualMemoryOutput",
    "OutputPathway",
    "KhwarizmiModel",
    "KhwarizmiOutput",
    "ShortTermWorkingState",
    "MemoryGatingController",
    "UtilityGatingPolicy",
    "LongTermPersistentMemory",
    "DualMemory",
    "DualMemoryOutput",
    "RETAIN",
    "WRITE",
    "UPDATE",
    "FORGET",
    "DECISION_NAMES",
    "CognitiveRouter",
    "PathwayDispatcher",
    "PathwayExecutionFlags",
    "SparseMoELayer",
    "ExpertLayer",
    "MoERoutingDecision",
    "create_standard_specialists",
    "SPECIALIZATION_NAMES",
    "AdaptiveComputeBlock",
    "LatentReasoner",
    "ToolVerificationRequest",
    "ToolVerificationResult",
    "PythonAnalysisTool",
    "ProjectPlannerTool",
    "SelectiveVerifier",
    "InputSanitizer",
    "SanitizedInputFrame",
    "KhwarizmiAgentLoop",
    "AgentResponseFrame",
]
