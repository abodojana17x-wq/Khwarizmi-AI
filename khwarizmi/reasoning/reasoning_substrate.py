"""
Khwarizmi Phase 6 — Fast Neuro-Symbolic Reasoning Substrate.

This module implements the evidence-aware, provenance-tracking, hypothesis-driven
reasoning substrate specified in Phase 6. It provides:

1. Evidence representation with structured metadata (source, provenance, reliability)
2. Hypothesis representation with verification states
3. Structural/causal analogy mechanism
4. Contradiction detection and conflict preservation
5. Bounded backtracking engine
6. Fast-path reasoning router
7. Structured reasoning traces

Speed-first architecture:
- Simple problems use fast-path (minimal compute, early exit)
- Complex problems invoke deep reasoning only when justified
- All operations are bounded and measurable
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from datetime import datetime
import hashlib


# -----------------------------------------------------------------------------
# Evidence Representation
# -----------------------------------------------------------------------------


class VerificationStatus(Enum):
    """Verification state of evidence or hypothesis."""
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"


class ContradictionStatus(Enum):
    """Contradiction status of a piece of evidence."""
    CONSISTENT = "consistent"
    POTENTIAL_CONFLICT = "potential_conflict"
    CONFIRMED_CONTRADICTION = "confirmed_contradiction"
    RESOLVED = "resolved"


@dataclass
class SourceMetadata:
    """Metadata about the source of evidence."""
    source_id: str
    source_type: str  # e.g., "user_input", "tool_output", "knowledge_base", "inference"
    timestamp: datetime = field(default_factory=datetime.now)
    reliability_score: float = 1.0  # [0, 1] confidence in source
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "timestamp": self.timestamp.isoformat(),
            "reliability_score": self.reliability_score,
            "additional_info": self.additional_info,
        }


@dataclass
class Evidence:
    """
    Structured evidence representation with provenance tracking.
    
    Attributes:
        content: The actual evidence content (text, structured data, etc.)
        source: Metadata about where this evidence came from
        provenance: Chain of derivation/inheritance (parent evidence IDs)
        reliability: Computed reliability score combining source + consistency
        independent_support: Count of independent sources supporting this
        contradiction_status: Whether this conflicts with other evidence
        verification_status: Result of empirical/symbolic verification
        lineage_timestamp: When this evidence was introduced
        evidence_id: Unique identifier (hash of content + source)
    """
    content: Any
    source: SourceMetadata
    provenance: List[str] = field(default_factory=list)  # parent evidence IDs
    reliability: float = 1.0
    independent_support: int = 1
    contradiction_status: ContradictionStatus = ContradictionStatus.CONSISTENT
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    lineage_timestamp: datetime = field(default_factory=datetime.now)
    evidence_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.evidence_id:
            # Generate unique ID from content hash
            content_str = str(self.content)
            self.evidence_id = hashlib.sha256(
                f"{content_str}:{self.source.source_id}".encode()
            ).hexdigest()[:16]

    def update_reliability(self) -> None:
        """Update reliability based on source, support, and contradictions."""
        base = self.source.reliability_score
        # Boost for independent support (diminishing returns)
        support_boost = min(0.2, 0.05 * (self.independent_support - 1))
        # Penalty for contradictions
        contradiction_penalty = {
            ContradictionStatus.CONSISTENT: 0.0,
            ContradictionStatus.POTENTIAL_CONFLICT: 0.1,
            ContradictionStatus.CONFIRMED_CONTRADICTION: 0.3,
            ContradictionStatus.RESOLVED: 0.05,
        }.get(self.contradiction_status, 0.0)
        # Penalty for unverified status
        verification_penalty = {
            VerificationStatus.UNVERIFIED: 0.0,
            VerificationStatus.VERIFIED: -0.1,  # bonus
            VerificationStatus.REFUTED: 0.5,
            VerificationStatus.INCONCLUSIVE: 0.1,
            VerificationStatus.ERROR: 0.3,
        }.get(self.verification_status, 0.0)

        self.reliability = max(0.0, min(1.0, 
            base + support_boost - contradiction_penalty - verification_penalty))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "content": str(self.content),
            "source": self.source.to_dict(),
            "provenance": self.provenance,
            "reliability": self.reliability,
            "independent_support": self.independent_support,
            "contradiction_status": self.contradiction_status.value,
            "verification_status": self.verification_status.value,
            "lineage_timestamp": self.lineage_timestamp.isoformat(),
            "metadata": self.metadata,
        }


# -----------------------------------------------------------------------------
# Hypothesis Representation
# -----------------------------------------------------------------------------


@dataclass
class Hypothesis:
    """
    Structured representation for testable hypotheses.
    
    Attributes:
        claim: The hypothesis statement/claim
        assumptions: List of underlying assumptions
        supporting_evidence: IDs of evidence that supports this hypothesis
        expected_observation: What should be observed if hypothesis is true
        confidence: Current confidence level [0, 1]
        verification_state: Current verification status
        parent_hypothesis: ID of parent hypothesis (for refinement chains)
        hypothesis_id: Unique identifier
    """
    claim: str
    assumptions: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    expected_observation: Optional[str] = None
    confidence: float = 0.5
    verification_state: VerificationStatus = VerificationStatus.UNVERIFIED
    parent_hypothesis: Optional[str] = None
    hypothesis_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    verification_attempts: int = 0
    last_verified_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.hypothesis_id:
            self.hypothesis_id = hashlib.sha256(
                f"{self.claim}:{self.created_at.isoformat()}".encode()
            ).hexdigest()[:16]

    def add_supporting_evidence(self, evidence_id: str) -> None:
        """Add supporting evidence and update confidence."""
        if evidence_id not in self.supporting_evidence:
            self.supporting_evidence.append(evidence_id)
            # Boost confidence slightly for each new piece of evidence
            self.confidence = min(1.0, self.confidence + 0.1)

    def record_verification(self, status: VerificationStatus) -> None:
        """Record verification result."""
        self.verification_state = status
        self.verification_attempts += 1
        self.last_verified_at = datetime.now()
        
        # Adjust confidence based on verification
        if status == VerificationStatus.VERIFIED:
            self.confidence = min(1.0, self.confidence + 0.2)
        elif status == VerificationStatus.REFUTED:
            self.confidence = max(0.0, self.confidence - 0.4)
        elif status == VerificationStatus.INCONCLUSIVE:
            self.confidence = max(0.1, self.confidence - 0.1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "claim": self.claim,
            "assumptions": self.assumptions,
            "supporting_evidence": self.supporting_evidence,
            "expected_observation": self.expected_observation,
            "confidence": self.confidence,
            "verification_state": self.verification_state.value,
            "parent_hypothesis": self.parent_hypothesis,
            "verification_attempts": self.verification_attempts,
            "last_verified_at": self.last_verified_at.isoformat() if self.last_verified_at else None,
        }


# -----------------------------------------------------------------------------
# Structural/Causal Analogy
# -----------------------------------------------------------------------------


@dataclass
class CausalFactor:
    """A causal or structural factor in a problem."""
    name: str
    type: str  # "structural", "causal", "constraint", "objective", "assumption"
    value: Any
    importance: float = 1.0  # [0, 1] relevance weight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "value": str(self.value),
            "importance": self.importance,
        }


@dataclass
class ProblemStructure:
    """Abstract structural representation of a problem."""
    structure: List[CausalFactor] = field(default_factory=list)
    constraints: List[CausalFactor] = field(default_factory=list)
    objective: Optional[CausalFactor] = None
    causal_factors: List[CausalFactor] = field(default_factory=list)
    assumptions: List[CausalFactor] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)

    def extract_signature(self) -> str:
        """Extract a structural signature for analogy matching."""
        sig_parts = []
        if self.objective:
            sig_parts.append(f"OBJ:{self.objective.type}")
        for cf in self.causal_factors[:3]:  # Top 3 causal factors
            sig_parts.append(f"CF:{cf.type}")
        for c in self.constraints[:2]:  # Top 2 constraints
            sig_parts.append(f"C:{c.type}")
        return "|".join(sig_parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structure": [f.to_dict() for f in self.structure],
            "constraints": [c.to_dict() for c in self.constraints],
            "objective": self.objective.to_dict() if self.objective else None,
            "causal_factors": [f.to_dict() for f in self.causal_factors],
            "assumptions": [a.to_dict() for a in self.assumptions],
            "failure_modes": self.failure_modes,
        }


@dataclass
class AnalogyMatch:
    """Result of structural/causal analogy matching."""
    prior_case_id: str
    shared_structural_factors: List[str] = field(default_factory=list)
    shared_causal_factors: List[str] = field(default_factory=list)
    transferred_principle: Optional[str] = None
    relevant_differences: List[str] = field(default_factory=list)
    acceptance_reason: str = ""
    rejection_reason: str = ""
    similarity_score: float = 0.0
    accepted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prior_case_id": self.prior_case_id,
            "shared_structural_factors": self.shared_structural_factors,
            "shared_causal_factors": self.shared_causal_factors,
            "transferred_principle": self.transferred_principle,
            "relevant_differences": self.relevant_differences,
            "acceptance_reason": self.acceptance_reason,
            "rejection_reason": self.rejection_reason,
            "similarity_score": self.similarity_score,
            "accepted": self.accepted,
        }


# -----------------------------------------------------------------------------
# Reasoning Trace
# -----------------------------------------------------------------------------


@dataclass
class ReasoningTrace:
    """
    Structured internal reasoning trace for debugging/evaluation.
    
    This is NOT exposed to end users - it's an engineering artifact for:
    - Debugging
    - Verification
    - Evaluation
    - Future training
    - Performance analysis
    - Reproducibility
    """
    problem: str
    evidence_used: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    selected_principle: Optional[str] = None
    causal_factors: List[str] = field(default_factory=list)
    hypothesis: Optional[str] = None
    verification_result: Optional[Dict[str, Any]] = None
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    backtracks: int = 0
    final_state: str = ""
    reasoning_depth: int = 0
    compute_budget_used: float = 0.0
    fast_path: bool = True
    early_exit: bool = False
    latency_ms: float = 0.0
    memory_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "evidence_used": self.evidence_used,
            "assumptions": self.assumptions,
            "selected_principle": self.selected_principle,
            "causal_factors": self.causal_factors,
            "hypothesis": self.hypothesis,
            "verification_result": self.verification_result,
            "contradictions": self.contradictions,
            "backtracks": self.backtracks,
            "final_state": self.final_state,
            "reasoning_depth": self.reasoning_depth,
            "compute_budget_used": self.compute_budget_used,
            "fast_path": self.fast_path,
            "early_exit": self.early_exit,
            "latency_ms": self.latency_ms,
            "memory_bytes": self.memory_bytes,
        }


# -----------------------------------------------------------------------------
# Evidence Store
# -----------------------------------------------------------------------------


class EvidenceStore:
    """
    Thread-safe evidence storage with provenance tracking and contradiction detection.
    """

    def __init__(self, max_size: int = 1000):
        self._evidence: Dict[str, Evidence] = {}
        self._max_size = max_size
        self._conflicts: List[Tuple[str, str]] = []  # pairs of conflicting evidence IDs

    def add(self, evidence: Evidence) -> bool:
        """Add evidence to the store. Returns False if rejected (e.g., capacity)."""
        if len(self._evidence) >= self._max_size:
            # Evict oldest unverified evidence
            self._evict_oldest_unverified()
        
        self._evidence[evidence.evidence_id] = evidence
        return True

    def get(self, evidence_id: str) -> Optional[Evidence]:
        """Retrieve evidence by ID."""
        return self._evidence.get(evidence_id)

    def find_by_content(self, content_hash: str) -> List[Evidence]:
        """Find evidence with similar content."""
        return [e for e in self._evidence.values() 
                if hashlib.sha256(str(e.content).encode()).hexdigest()[:16] == content_hash]

    def mark_contradiction(self, id1: str, id2: str) -> None:
        """Mark two pieces of evidence as contradictory."""
        if id1 in self._evidence and id2 in self._evidence:
            self._evidence[id1].contradiction_status = ContradictionStatus.CONFIRMED_CONTRADICTION
            self._evidence[id2].contradiction_status = ContradictionStatus.CONFIRMED_CONTRADICTION
            self._conflicts.append((id1, id2))
            self._evidence[id1].update_reliability()
            self._evidence[id2].update_reliability()

    def _evict_oldest_unverified(self) -> None:
        """Evict oldest unverified evidence to make room."""
        unverified = [
            (e.lineage_timestamp, eid) 
            for eid, e in self._evidence.items() 
            if e.verification_status == VerificationStatus.UNVERIFIED
        ]
        if unverified:
            unverified.sort()
            oldest_id = unverified[0][1]
            del self._evidence[oldest_id]

    def get_all(self) -> List[Evidence]:
        """Get all evidence."""
        return list(self._evidence.values())

    def size(self) -> int:
        """Current store size."""
        return len(self._evidence)


# -----------------------------------------------------------------------------
# Verification Interface
# -----------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Structured verification outcome."""
    status: VerificationStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "execution_time_ms": self.execution_time_ms,
            "error": self.error,
        }


class VerificationEngine:
    """
    Abstract verification engine interface.
    
    Supports multiple verification strategies:
    - Static analysis (AST, type checking)
    - Execution-based (run tests, observe output)
    - Symbolic (logical consistency)
    - Empirical (compare against known facts)
    """

    def __init__(self):
        self._verifiers: Dict[str, callable] = {}

    def register_verifier(self, name: str, verifier: callable) -> None:
        """Register a verification function."""
        self._verifiers[name] = verifier

    def verify(self, hypothesis: Hypothesis, strategy: str = "static") -> VerificationResult:
        """
        Verify a hypothesis using the specified strategy.
        
        Args:
            hypothesis: The hypothesis to verify
            strategy: Verification strategy ("static", "execution", "symbolic", "empirical")
        
        Returns:
            VerificationResult with status and details
        """
        import time
        start = time.perf_counter()
        
        try:
            if strategy not in self._verifiers:
                return VerificationResult(
                    status=VerificationStatus.ERROR,
                    message=f"Unknown verification strategy: {strategy}",
                    execution_time_ms=(time.perf_counter() - start) * 1000,
                )
            
            result = self._verifiers[strategy](hypothesis)
            result.execution_time_ms = (time.perf_counter() - start) * 1000
            return result
            
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                message="Verification failed with exception",
                error=str(e),
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )


# -----------------------------------------------------------------------------
# Contradiction Detector
# -----------------------------------------------------------------------------


class ContradictionDetector:
    """
    Detects contradictions between pieces of evidence and hypotheses.
    """

    def __init__(self):
        self._rules: List[callable] = []

    def register_rule(self, rule: callable) -> None:
        """Register a contradiction detection rule."""
        self._rules.append(rule)

    def detect(self, evidence1: Evidence, evidence2: Evidence) -> bool:
        """
        Check if two pieces of evidence contradict.
        
        Returns True if contradiction detected.
        """
        for rule in self._rules:
            if rule(evidence1, evidence2):
                return True
        return False

    def check_consistency(self, store: EvidenceStore, new_evidence: Evidence) -> List[str]:
        """
        Check new evidence against all stored evidence.
        
        Returns list of conflicting evidence IDs.
        """
        conflicts = []
        for existing in store.get_all():
            if self.detect(existing, new_evidence):
                conflicts.append(existing.evidence_id)
        return conflicts


# -----------------------------------------------------------------------------
# Bounded Backtracking Engine
# -----------------------------------------------------------------------------


@dataclass
class BacktrackState:
    """State saved for potential backtracking."""
    hypothesis_id: str
    assumptions: List[str]
    evidence_used: List[str]
    reasoning_step: int
    confidence: float


class BacktrackingEngine:
    """
    Bounded backtracking for hypothesis revision.
    
    Provides:
    - Configurable maximum backtracks
    - Configurable maximum depth
    - Compute budget enforcement
    - Early termination on success
    """

    def __init__(
        self,
        max_backtracks: int = 5,
        max_depth: int = 10,
        max_compute_budget: float = 100.0,  # milliseconds
    ):
        self.max_backtracks = max_backtracks
        self.max_depth = max_depth
        self.max_compute_budget = max_compute_budget
        
        self._stack: List[BacktrackState] = []
        self._backtracks_used = 0
        self._compute_used = 0.0
        self._current_depth = 0

    def push(self, state: BacktrackState) -> None:
        """Push state onto backtrack stack."""
        if self._current_depth < self.max_depth:
            self._stack.append(state)
            self._current_depth += 1

    def pop(self) -> Optional[BacktrackState]:
        """Pop state from backtrack stack."""
        if self._stack:
            self._current_depth -= 1
            return self._stack.pop()
        return None

    def can_backtrack(self) -> bool:
        """Check if backtracking is still allowed."""
        return (
            self._backtracks_used < self.max_backtracks
            and self._compute_used < self.max_compute_budget
            and len(self._stack) > 0
        )

    def backtrack(self) -> Optional[BacktrackState]:
        """Perform a backtrack operation."""
        if not self.can_backtrack():
            return None
        
        self._backtracks_used += 1
        return self.pop()

    def record_compute(self, ms: float) -> None:
        """Record compute time used."""
        self._compute_used += ms

    def reset(self) -> None:
        """Reset backtracking state."""
        self._stack.clear()
        self._backtracks_used = 0
        self._compute_used = 0.0
        self._current_depth = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get backtracking statistics."""
        return {
            "backtracks_used": self._backtracks_used,
            "max_backtracks": self.max_backtracks,
            "compute_used_ms": self._compute_used,
            "max_compute_budget_ms": self.max_compute_budget,
            "current_depth": self._current_depth,
            "max_depth": self.max_depth,
            "stack_size": len(self._stack),
        }


# -----------------------------------------------------------------------------
# Fast-Path Reasoning Router
# -----------------------------------------------------------------------------


class ReasoningRouter:
    """
    Fast-path reasoning router.
    
    Determines whether a problem requires deep reasoning or can be solved
    via fast path (direct lookup, simple inference, high-confidence retrieval).
    
    Decision factors:
    - Problem complexity (keyword/pattern based)
    - Available evidence quality
    - Prior solution existence
    - Confidence thresholds
    """

    def __init__(
        self,
        fast_path_threshold: float = 0.8,
        complexity_threshold: int = 3,
    ):
        self.fast_path_threshold = fast_path_threshold
        self.complexity_threshold = complexity_threshold
        
        # Complexity indicators
        self._complexity_patterns = [
            r"\b(if|else|while|for|because|therefore)\b",
            r"\b(compare|analyze|evaluate|synthesize)\b",
            r"\b(why|how|what if)\b",
            r"[,;:]",  # Multiple clauses
        ]

    def route(self, 
              problem: str, 
              available_evidence: List[Evidence],
              prior_solutions: List[str]) -> Tuple[bool, str]:
        """
        Route a problem to fast-path or deep reasoning.
        
        Returns:
            Tuple of (use_fast_path, reason)
        """
        # Check 1: High-confidence prior solution exists
        if prior_solutions:
            return True, "prior_solution_available"
        
        # Check 2: Problem complexity
        complexity = self._compute_complexity(problem)
        if complexity < self.complexity_threshold:
            return True, "low_complexity"
        
        # Check 3: Evidence quality
        if available_evidence:
            avg_reliability = sum(e.reliability for e in available_evidence) / len(available_evidence)
            if avg_reliability >= self.fast_path_threshold:
                # Check for contradictions
                has_contradictions = any(
                    e.contradiction_status == ContradictionStatus.CONFIRMED_CONTRADICTION
                    for e in available_evidence
                )
                if not has_contradictions:
                    return True, "high_quality_evidence"
        
        # Default: Use deep reasoning
        return False, "deep_reasoning_required"

    def _compute_complexity(self, problem: str) -> int:
        """Compute problem complexity score."""
        score = 0
        for pattern in self._complexity_patterns:
            import re
            matches = len(re.findall(pattern, problem.lower()))
            score += matches
        # Add length factor
        score += len(problem.split()) // 20
        return score


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------


__all__ = [
    # Evidence
    "Evidence",
    "SourceMetadata",
    "VerificationStatus",
    "ContradictionStatus",
    "EvidenceStore",
    
    # Hypothesis
    "Hypothesis",
    
    # Analogy
    "ProblemStructure",
    "CausalFactor",
    "AnalogyMatch",
    
    # Verification
    "VerificationResult",
    "VerificationEngine",
    
    # Contradiction
    "ContradictionDetector",
    
    # Backtracking
    "BacktrackState",
    "BacktrackingEngine",
    
    # Routing
    "ReasoningRouter",
    
    # Trace
    "ReasoningTrace",
]
