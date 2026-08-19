"""
Khwarizmi Adaptive Compute and Reasoning Package.

Phase 5: Adaptive Compute / ARRC (per-token ACT halting)
Phase 6: Neural Reasoning Core + Fast Neuro-Symbolic Substrate
"""

from .adaptive_compute import AdaptiveComputeBlock, PonderCostLoss
from .latent_reasoner import LatentReasoner
from .neural_reasoning_core import (
    LatentSynthesisBlock,
    ConsistencyHead,
    SelfCorrectionBlock,
    ReasoningLosses,
    NeuralReasoningCore,
    ReasoningOutput,
)
from .reasoning_substrate import (
    # Evidence
    Evidence,
    SourceMetadata,
    VerificationStatus,
    ContradictionStatus,
    EvidenceStore,
    # Hypothesis
    Hypothesis,
    # Analogy
    ProblemStructure,
    CausalFactor,
    AnalogyMatch,
    # Verification
    VerificationResult,
    VerificationEngine,
    # Contradiction
    ContradictionDetector,
    # Backtracking
    BacktrackState,
    BacktrackingEngine,
    # Routing
    ReasoningRouter,
    # Trace
    ReasoningTrace,
)

__all__ = [
    # Phase 5 - Adaptive Compute
    "AdaptiveComputeBlock",
    "PonderCostLoss",
    "LatentReasoner",
    # Phase 6 - Neural Reasoning Core
    "LatentSynthesisBlock",
    "ConsistencyHead",
    "SelfCorrectionBlock",
    "ReasoningLosses",
    "NeuralReasoningCore",
    "ReasoningOutput",
    # Phase 6 - Fast Neuro-Symbolic Substrate
    "Evidence",
    "SourceMetadata",
    "VerificationStatus",
    "ContradictionStatus",
    "EvidenceStore",
    "Hypothesis",
    "ProblemStructure",
    "CausalFactor",
    "AnalogyMatch",
    "VerificationResult",
    "VerificationEngine",
    "ContradictionDetector",
    "BacktrackState",
    "BacktrackingEngine",
    "ReasoningRouter",
    "ReasoningTrace",
]
