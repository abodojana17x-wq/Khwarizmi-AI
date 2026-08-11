"""
Khwarizmi AI: System Architecture Core Namespace.

Version: 0.1.0-phase1
Provides foundational implementations of:
    - Khwarizmi State Cell (KSC) & Residual Blocks
    - Short-Term Working State & Utility-Gated Long-Term Persistent Memory
    - Cognitive Router & Compute Pathway Dispatcher
    - Sparse Mixture-of-Experts (MoE) Specialists
    - Adaptive Recurrent Reasoning Cycles (ARRC)
    - Output Pathway & Selective Verification Trigger
    - Layered Agent Orchestrator & Legacy Deterministic Tool Bridge
"""

__version__ = "0.1.0-phase1"

from .config import (
    KhwarizmiConfig,
    get_tiny_test_config,
    get_prototype_config,
    get_small_config,
    get_edge_config,
)
from .core import (
    SinusoidalPositionalEncoding,
    KhwarizmiEmbeddings,
    KhwarizmiStateCell,
    FeedForwardNetwork,
    KSCResidualBlock,
    OutputPathway,
    KhwarizmiModel,
    KhwarizmiOutput,
)
from .memory import (
    ShortTermWorkingState,
    MemoryGatingController,
    LongTermPersistentMemory,
)
from .routing import (
    CognitiveRouter,
    PathwayDispatcher,
    PathwayExecutionFlags,
)
from .experts import (
    SparseMoELayer,
    ExpertLayer,
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
    "get_small_config",
    "get_edge_config",
    "SinusoidalPositionalEncoding",
    "KhwarizmiEmbeddings",
    "KhwarizmiStateCell",
    "FeedForwardNetwork",
    "KSCResidualBlock",
    "OutputPathway",
    "KhwarizmiModel",
    "KhwarizmiOutput",
    "ShortTermWorkingState",
    "MemoryGatingController",
    "LongTermPersistentMemory",
    "CognitiveRouter",
    "PathwayDispatcher",
    "PathwayExecutionFlags",
    "SparseMoELayer",
    "ExpertLayer",
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
