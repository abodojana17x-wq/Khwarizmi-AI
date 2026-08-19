"""
Khwarizmi AI Core Configuration & Dimensional Settings.

This module implements the configuration schema and validation for the Khwarizmi
neural architecture layer as specified in the Phase 0 Architecture Blueprint.
It enables scaling across hardware tiers (Prototype -> Small -> Edge) without
rewriting core architecture logic.
"""

from dataclasses import dataclass, field, asdict
import json
import math
from typing import Dict, Any, Optional


@dataclass
class KhwarizmiConfig:
    """
    Configuration parameters for the Khwarizmi AI Neural Core.
    
    Attributes:
        vocab_size: Number of tokens in vocabulary.
        d_model: Internal feature dimension (D).
        n_layers: Number of sequential residual blocks.
        n_heads: Number of KSC attention/recurrent heads (H).
        d_expansion: Memory expansion bank size per head (d_n).
        d_ff: Intermediate dimension for Feed-Forward / Expert networks.
        num_experts: Total number of experts in Sparse MoE layers (E).
        top_k_experts: Number of active experts per token in MoE layers (K).
        moe_frequency: Interval of layers between MoE blocks (e.g. 2 = every 2nd layer).
        enable_moe: Master switch for the Phase 4 Sparse MoE sublayers. When False,
            every residual block uses a dense FFN and no experts/router are built.
        moe_noise_enabled: Whether the MoE router adds learnable gating noise during
            training (noise is always disabled at inference time).
        expert_d_ff: Intermediate dimension of each MoE expert FFN. If None, experts
            use d_ff.
        max_seq_len: Maximum supported sequence length for windowing/embeddings.
        gamma_min: Minimum retention eigenvalue bound for KSC stability.
        gamma_max: Maximum retention eigenvalue bound for KSC stability.
        max_recurrent_cycles: Max ARRC recurrent reasoning cycles (K_max). Hard
            upper bound on adaptive computation — recurrence always terminates
            by this step.
        min_recurrent_cycles: Min ARRC recurrent reasoning cycles (K_min). No
            token may halt before this step. Must satisfy
            1 <= min_recurrent_cycles <= max_recurrent_cycles.
        enable_adaptive_compute: Master switch for the Phase 5 Adaptive Compute /
            ARRC halting engine. When False, the model performs a single fixed
            pass with no recurrent reasoning cycles, no halting gates are built,
            and ponder loss is exactly zero (pre-Phase-5-compatible path).
        halting_epsilon: Halting threshold slack epsilon. A token halts at the
            first cycle k where the accumulated halting probability satisfies
            sum_{j<=k} p_j >= 1 - epsilon. Must be in (0, 1).
        memory_dim: Key/Value feature dimension for Long-Term Persistent Memory.
        memory_slots: Maximum capacity of slots in Long-Term Persistent Memory table.
        short_term_capacity: Bounded capacity (token window) of Short-Term Working State.
        utility_threshold: Minimum utility score for a WRITE/promotion decision.
        read_threshold: Gating probability threshold for the READ operation.
        write_threshold: Gating probability threshold for the WRITE operation.
        update_threshold: Gating probability threshold for the UPDATE operation.
        forget_threshold: Gating probability threshold for the FORGET operation.
        update_similarity_threshold: Cosine-similarity threshold above which a
            candidate is merged into an existing slot (UPDATE) rather than inserted.
        utility_decay_lambda: Exponential time-decay constant for utility eviction.
        num_pathways: Number of discrete computational pathways in Cognitive Router.
        dropout: Regularization dropout probability.
        temperature: Router softmax sampling temperature.
        load_balance_alpha: Coefficient for MoE load balancing auxiliary loss.
        ponder_cost_beta: Coefficient for Adaptive Compute ponder cost loss.
        verification_threshold: Confidence score threshold below which verification triggers.
        enable_reasoning_core: Master switch for the Phase 6 Neural Reasoning Core
            (Latent Synthesis & Bounded Self-Correction). When False, no reasoning
            synthesis/correction/confidence submodules are built, the model performs
            a single fixed pass through the ARRC output with no iterative reasoning,
            and reasoning-specific auxiliary losses are exactly zero (pre-Phase-6 path).
        min_reasoning_steps: Minimum number of bounded reasoning iterations K_r^min.
            No reasoning termination may occur before this step. Must satisfy
            1 <= min_reasoning_steps <= max_reasoning_steps.
        max_reasoning_steps: Hard maximum number of reasoning iterations K_r^max.
            The reasoning loop always terminates by this step (forced convergence).
            Independent from the ARRC max_recurrent_cycles compute budget.
        reasoning_confidence_threshold: Confidence score above which the latent
            reasoning state is considered sufficiently refined and the loop halts.
            Must be in [0, 1].
        max_reasoning_corrections: Hard maximum number of self-correction refinements.
            Each correction is a bounded latent update conditioned by the confidence
            signal. Must satisfy 0 <= max_reasoning_corrections <= max_reasoning_steps.
        reasoning_confidence_beta: Coefficient for the reasoning consistency loss.
        reasoning_refinement_beta: Coefficient for the reasoning refinement loss.
        enable_full_neural_core: Master switch for the Phase 7 Full Khwarizmi Neural
            Core integration. This is *additive only*: the unified forward path
            (KSC -> Sparse MoE -> Dual Memory -> ARRC -> Neural Reasoning Core ->
            Output) is already executed by the model regardless of this flag.
            When True (default), the structured ``full_core`` diagnostics
            namespace is emitted on every forward pass, exposing the
            contribution/status of each subsystem (KSC, MoE, Memory, ARRC,
            Reasoning, Full Core) in one inspectable, numerical structure.
            When False, the output is identical to the pre-Phase-7 contract
            (no ``full_core`` diagnostics key), preserving backward compatibility.
        tier_name: Human-readable hardware/model tier label.
    """
    vocab_size: int = 1024
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_expansion: int = 16
    d_ff: int = 256
    num_experts: int = 4
    top_k_experts: int = 2
    moe_frequency: int = 2
    enable_moe: bool = True
    moe_noise_enabled: bool = True
    expert_d_ff: Optional[int] = None
    max_seq_len: int = 512
    gamma_min: float = 0.85
    gamma_max: float = 0.999
    max_recurrent_cycles: int = 3
    min_recurrent_cycles: int = 1
    enable_adaptive_compute: bool = True
    halting_epsilon: float = 0.01
    memory_dim: int = 64
    memory_slots: int = 32
    short_term_capacity: int = 512
    utility_threshold: float = 0.8
    read_threshold: float = 0.5
    write_threshold: float = 0.5
    update_threshold: float = 0.5
    forget_threshold: float = 0.7
    update_similarity_threshold: float = 0.88
    utility_decay_lambda: float = 0.01
    num_pathways: int = 5
    dropout: float = 0.1
    temperature: float = 1.0
    load_balance_alpha: float = 0.01
    ponder_cost_beta: float = 0.01
    enable_reasoning_core: bool = True
    min_reasoning_steps: int = 1
    max_reasoning_steps: int = 3
    reasoning_confidence_threshold: float = 0.85
    max_reasoning_corrections: int = 2
    reasoning_confidence_beta: float = 0.01
    reasoning_refinement_beta: float = 0.01
    verification_threshold: float = 0.75
    enable_full_neural_core: bool = True
    tier_name: str = "TinyTest"

    def __post_init__(self) -> None:
        """Validate configuration constraints upon instantiation."""
        self.validate()

    @property
    def d_k(self) -> int:
        """Return dimension per KSC head (d_k = d_model // n_heads)."""
        return self.d_model // self.n_heads

    def validate(self) -> None:
        """
        Validate tensor shapes, dimensions, and mathematical bounds.
        Raises ValueError if any configuration parameter violates architectural invariants.
        """
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be cleanly divisible by n_heads ({self.n_heads})"
            )
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.d_expansion <= 0:
            raise ValueError(f"d_expansion must be positive, got {self.d_expansion}")
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {self.num_experts}")
        if self.top_k_experts <= 0 or self.top_k_experts > self.num_experts:
            raise ValueError(
                f"top_k_experts ({self.top_k_experts}) must be in range [1, num_experts ({self.num_experts})]"
            )
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        if self.moe_frequency < 1:
            raise ValueError(
                f"moe_frequency must be >= 1, got {self.moe_frequency}"
            )
        if self.expert_d_ff is not None and self.expert_d_ff <= 0:
            raise ValueError(
                f"expert_d_ff must be positive when set, got {self.expert_d_ff}"
            )
        if not (math.isfinite(self.load_balance_alpha) and self.load_balance_alpha >= 0.0):
            raise ValueError(
                f"load_balance_alpha must be a finite non-negative value, got {self.load_balance_alpha}"
            )
        if not (0.0 < self.gamma_min < self.gamma_max < 1.0):
            raise ValueError(
                f"Eigenvalue bounds must satisfy 0 < gamma_min ({self.gamma_min}) < gamma_max ({self.gamma_max}) < 1.0"
            )
        if self.max_recurrent_cycles < 1:
            raise ValueError(
                f"max_recurrent_cycles must be >= 1, got {self.max_recurrent_cycles}"
            )
        if self.min_recurrent_cycles < 1:
            raise ValueError(
                f"min_recurrent_cycles must be >= 1, got {self.min_recurrent_cycles}"
            )
        if self.min_recurrent_cycles > self.max_recurrent_cycles:
            raise ValueError(
                f"min_recurrent_cycles ({self.min_recurrent_cycles}) must be <= "
                f"max_recurrent_cycles ({self.max_recurrent_cycles})"
            )
        if not (0.0 < self.halting_epsilon < 1.0):
            raise ValueError(
                f"halting_epsilon must be in (0, 1), got {self.halting_epsilon}"
            )
        if not (math.isfinite(self.ponder_cost_beta) and self.ponder_cost_beta >= 0.0):
            raise ValueError(
                f"ponder_cost_beta must be a finite non-negative value, got {self.ponder_cost_beta}"
            )
        if self.memory_slots < 1 or self.memory_dim <= 0:
            raise ValueError(
                f"Invalid long-term memory configuration: slots={self.memory_slots}, dim={self.memory_dim}"
            )
        if self.short_term_capacity < 1:
            raise ValueError(
                f"short_term_capacity must be >= 1, got {self.short_term_capacity}"
            )
        for name in (
            "utility_threshold",
            "read_threshold",
            "write_threshold",
            "update_threshold",
            "forget_threshold",
            "update_similarity_threshold",
        ):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.utility_decay_lambda < 0.0:
            raise ValueError(
                f"utility_decay_lambda must be >= 0, got {self.utility_decay_lambda}"
            )
        if self.num_pathways < 1:
            raise ValueError(f"num_pathways must be >= 1, got {self.num_pathways}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        # ---- Phase 6: Neural Reasoning Core bounds ----
        if self.max_reasoning_steps < 1:
            raise ValueError(
                f"max_reasoning_steps must be >= 1, got {self.max_reasoning_steps}"
            )
        if self.min_reasoning_steps < 1:
            raise ValueError(
                f"min_reasoning_steps must be >= 1, got {self.min_reasoning_steps}"
            )
        if self.min_reasoning_steps > self.max_reasoning_steps:
            raise ValueError(
                f"min_reasoning_steps ({self.min_reasoning_steps}) must be <= "
                f"max_reasoning_steps ({self.max_reasoning_steps})"
            )
        if not (0.0 <= self.reasoning_confidence_threshold <= 1.0):
            raise ValueError(
                f"reasoning_confidence_threshold must be in [0, 1], "
                f"got {self.reasoning_confidence_threshold}"
            )
        if self.max_reasoning_corrections < 0:
            raise ValueError(
                f"max_reasoning_corrections must be >= 0, "
                f"got {self.max_reasoning_corrections}"
            )
        if self.max_reasoning_corrections > self.max_reasoning_steps:
            raise ValueError(
                f"max_reasoning_corrections ({self.max_reasoning_corrections}) "
                f"must be <= max_reasoning_steps ({self.max_reasoning_steps})"
            )
        if not (
            math.isfinite(self.reasoning_confidence_beta)
            and self.reasoning_confidence_beta >= 0.0
        ):
            raise ValueError(
                f"reasoning_confidence_beta must be a finite non-negative value, "
                f"got {self.reasoning_confidence_beta}"
            )
        if not (
            math.isfinite(self.reasoning_refinement_beta)
            and self.reasoning_refinement_beta >= 0.0
        ):
            raise ValueError(
                f"reasoning_refinement_beta must be a finite non-negative value, "
                f"got {self.reasoning_refinement_beta}"
            )
        # ---- Phase 7: Full Neural Core integration flag ----
        # Additive-only master switch for the structured full-core diagnostics
        # namespace. No behavioral change to the forward path; validated as a
        # strict boolean so it cannot silently take a truthy non-bool value.
        if not isinstance(self.enable_full_neural_core, bool):
            raise ValueError(
                f"enable_full_neural_core must be a bool, got "
                f"{type(self.enable_full_neural_core).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KhwarizmiConfig":
        """Instantiate configuration from dictionary."""
        return cls(**data)

    def to_json_string(self) -> str:
        """Serialize configuration to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json_string(cls, json_str: str) -> "KhwarizmiConfig":
        """Instantiate configuration from JSON string."""
        return cls.from_dict(json.loads(json_str))
