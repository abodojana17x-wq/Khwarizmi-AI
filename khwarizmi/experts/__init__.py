"""
Khwarizmi Sparse Mixture-of-Experts (MoE) Package — Phase 4.

Sparse Top-K Noisy-Gated MoE per Sections 4.4 / 5.4 of the Blueprint:
    - ``SparseMoELayer``: noisy Top-K router, sparse expert dispatch, and
      load-balancing auxiliary loss.
    - ``ExpertLayer``: independently parameterized Swish FFN expert.
    - ``MoERoutingDecision``: structured routing result (logits, probs,
      Top-K indices/weights, dispatch fractions, auxiliary loss).
    - ``create_standard_specialists``: named specialist expert factory.
"""

from .moe_layer import SparseMoELayer, ExpertLayer, MoERoutingDecision
from .specialists import create_standard_specialists, SPECIALIZATION_NAMES

__all__ = [
    "SparseMoELayer",
    "ExpertLayer",
    "MoERoutingDecision",
    "create_standard_specialists",
    "SPECIALIZATION_NAMES",
]
