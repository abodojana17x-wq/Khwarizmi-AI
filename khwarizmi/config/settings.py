"""
Khwarizmi AI Core Configuration & Dimensional Settings.

This module implements the configuration schema and validation for the Khwarizmi
neural architecture layer as specified in the Phase 0 Architecture Blueprint.
It enables scaling across hardware tiers (Prototype -> Small -> Edge) without
rewriting core architecture logic.
"""

from dataclasses import dataclass, field, asdict
import json
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
        max_seq_len: Maximum supported sequence length for windowing/embeddings.
        gamma_min: Minimum retention eigenvalue bound for KSC stability.
        gamma_max: Maximum retention eigenvalue bound for KSC stability.
        max_recurrent_cycles: Max ARRC recurrent reasoning cycles (K_max).
        memory_dim: Key/Value feature dimension for Long-Term Persistent Memory.
        memory_slots: Maximum capacity of slots in Long-Term Persistent Memory table.
        num_pathways: Number of discrete computational pathways in Cognitive Router.
        dropout: Regularization dropout probability.
        temperature: Router softmax sampling temperature.
        load_balance_alpha: Coefficient for MoE load balancing auxiliary loss.
        ponder_cost_beta: Coefficient for Adaptive Compute ponder cost loss.
        verification_threshold: Confidence score threshold below which verification triggers.
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
    max_seq_len: int = 512
    gamma_min: float = 0.85
    gamma_max: float = 0.999
    max_recurrent_cycles: int = 3
    memory_dim: int = 64
    memory_slots: int = 32
    num_pathways: int = 5
    dropout: float = 0.1
    temperature: float = 1.0
    load_balance_alpha: float = 0.01
    ponder_cost_beta: float = 0.01
    verification_threshold: float = 0.75
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
        if not (0.0 < self.gamma_min < self.gamma_max < 1.0):
            raise ValueError(
                f"Eigenvalue bounds must satisfy 0 < gamma_min ({self.gamma_min}) < gamma_max ({self.gamma_max}) < 1.0"
            )
        if self.max_recurrent_cycles < 1:
            raise ValueError(
                f"max_recurrent_cycles must be >= 1, got {self.max_recurrent_cycles}"
            )
        if self.memory_slots < 1 or self.memory_dim <= 0:
            raise ValueError(
                f"Invalid long-term memory configuration: slots={self.memory_slots}, dim={self.memory_dim}"
            )
        if self.num_pathways < 1:
            raise ValueError(f"num_pathways must be >= 1, got {self.num_pathways}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

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
