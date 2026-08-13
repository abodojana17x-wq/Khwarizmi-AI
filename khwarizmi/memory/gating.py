"""
Khwarizmi Memory Gating Controller Module.

Implements the learned gating network for the Long-Term Persistent Memory store as
defined in Section 5.2 of the Khwarizmi AI Blueprint. It outputs continuous/discrete
probabilities for READ, WRITE, UPDATE, and FORGET operations conditioned on the
current latent state and task context.

In addition, this module provides the *deterministic* utility-gating decision
policy (:class:`UtilityGatingPolicy`) that maps gate probabilities, predicted
utility scores and associative similarity into one of four discrete actions:

* ``RETAIN`` — keep the information in short-term memory only.
* ``WRITE``  — promote the information into persistent memory.
* ``UPDATE`` — merge the information into an existing persistent slot.
* ``FORGET`` — discard / evict according to the memory policy.

The policy is fully deterministic under deterministic inputs, numerically stable,
bounded, and independent of the future MoE / Router / Adaptive-Compute components.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from ..config.settings import KhwarizmiConfig

# Discrete memory-decision codes (stable public interface).
RETAIN = 0
WRITE = 1
UPDATE = 2
FORGET = 3

DECISION_NAMES = {
    RETAIN: "retain",
    WRITE: "write",
    UPDATE: "update",
    FORGET: "forget",
}


class MemoryGatingController(nn.Module):
    """
    Learned Memory Gating Controller.
    Computes activation probabilities for four persistent memory operations:
        - READ: Retrieve historical decisions/facts via associative similarity.
        - WRITE: Insert new knowledge tuple when utility threshold is exceeded.
        - UPDATE: Refine existing knowledge nodes.
        - FORGET: Evict obsolete/low-utility entries.
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.lambda_mem = 0.01

        # Gating projections for each operation (maps 2 * d_model -> 1)
        self.w_read = nn.Linear(2 * self.d_model, 1)
        self.w_write = nn.Linear(2 * self.d_model, 1)
        self.w_update = nn.Linear(2 * self.d_model, 1)
        self.w_forget = nn.Linear(2 * self.d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize gating weights with slight negative bias for conservative writes."""
        nn.init.xavier_uniform_(self.w_read.weight)
        nn.init.xavier_uniform_(self.w_write.weight)
        nn.init.xavier_uniform_(self.w_update.weight)
        nn.init.xavier_uniform_(self.w_forget.weight)
        nn.init.constant_(self.w_write.bias, -1.0)
        nn.init.constant_(self.w_forget.bias, -1.0)

    def forward(
        self,
        h_t: torch.Tensor,
        s_task: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute READ, WRITE, UPDATE, and FORGET gating probabilities.

        Args:
            h_t: Current latent representation of shape (batch_size, d_model).
            s_task: Optional task context vector of shape (batch_size, d_model).
                    If None, defaults to zeroes.

        Returns:
            Dictionary containing 'read', 'write', 'update', 'forget' probability tensors
            of shape (batch_size,).
        """
        if s_task is None:
            s_task = torch.zeros_like(h_t)

        combined = torch.cat([h_t, s_task], dim=-1)

        g_read = torch.sigmoid(self.w_read(combined)).squeeze(-1)
        g_write = torch.sigmoid(self.w_write(combined)).squeeze(-1)
        g_update = torch.sigmoid(self.w_update(combined)).squeeze(-1)
        g_forget = torch.sigmoid(self.w_forget(combined)).squeeze(-1)

        return {
            "read": g_read,
            "write": g_write,
            "update": g_update,
            "forget": g_forget,
        }

    def compute_gating_regularization(
        self, gates: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute L2 activity regularization on gating probabilities to prevent
        gate saturation and ensure smooth gradient flow across all gates.

        Args:
            gates: Dictionary of gating probability tensors.

        Returns:
            Scalar regularization loss tensor.
        """
        loss = 0.0
        for gate_val in gates.values():
            loss = loss + torch.mean(gate_val ** 2)
        return self.lambda_mem * loss


class UtilityGatingPolicy:
    """
    Deterministic utility-gating decision policy for the Dual Memory lifecycle.

    Given the learned gate probabilities, a predicted utility score per candidate,
    and the top-1 associative similarity to the existing persistent table, the
    policy assigns each batch item exactly one discrete action:

    Priority order (first match wins):

    1. ``FORGET`` — ``g_forget >= forget_threshold`` (explicit eviction request).
    2. ``UPDATE`` — ``max_sim >= update_similarity_threshold`` **and**
       ``g_update >= update_threshold`` (refine an existing slot).
    3. ``WRITE`` — ``utility >= utility_threshold`` **and**
       ``g_write >= write_threshold`` (promote to persistent memory).
    4. ``RETAIN`` — otherwise (remain in short-term memory only).

    The policy is pure (holds no parameters), deterministic, bounded, and
    numerically stable; it has no dependency on the MoE / Router / Adaptive-Compute
    components.
    """

    def __init__(self, config: KhwarizmiConfig):
        self.utility_threshold = config.utility_threshold
        self.write_threshold = config.write_threshold
        self.update_threshold = config.update_threshold
        self.forget_threshold = config.forget_threshold
        self.update_similarity_threshold = config.update_similarity_threshold

    def decide(
        self,
        gates: Dict[str, torch.Tensor],
        utilities: torch.Tensor,
        max_similarity: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Assign a discrete memory action to each batch item.

        Args:
            gates: Gate-probability dict (``read``/``write``/``update``/``forget``),
                each of shape ``(batch_size,)``.
            utilities: Predicted utility scores of shape ``(batch_size,)`` in [0, 1].
            max_similarity: Top-1 cosine similarity to the persistent table of
                shape ``(batch_size,)``; ``-1`` denotes an empty table.

        Returns:
            Dictionary with:
                decision: LongTensor of shape (batch_size,) holding one of
                    ``RETAIN`` / ``WRITE`` / ``UPDATE`` / ``FORGET``.
                retain_mask, write_mask, update_mask, forget_mask:
                    Boolean tensors of shape (batch_size,) selecting each action.
        """
        g_forget = gates["forget"]
        g_write = gates["write"]
        g_update = gates["update"]
        batch_size = g_forget.size(0)

        forget_mask = g_forget >= self.forget_threshold
        update_mask = (max_similarity >= self.update_similarity_threshold) & (
            g_update >= self.update_threshold
        )
        write_mask = (utilities >= self.utility_threshold) & (
            g_write >= self.write_threshold
        )

        # Enforce priority: FORGET > UPDATE > WRITE > RETAIN.
        write_mask = write_mask & (~forget_mask) & (~update_mask)
        update_mask = update_mask & (~forget_mask)

        decision = torch.full(
            (batch_size,), RETAIN, dtype=torch.long, device=g_forget.device
        )
        decision = torch.where(write_mask, torch.full_like(decision, WRITE), decision)
        decision = torch.where(update_mask, torch.full_like(decision, UPDATE), decision)
        decision = torch.where(forget_mask, torch.full_like(decision, FORGET), decision)

        retain_mask = ~(forget_mask | update_mask | write_mask)

        return {
            "decision": decision,
            "retain_mask": retain_mask,
            "write_mask": write_mask,
            "update_mask": update_mask,
            "forget_mask": forget_mask,
        }

    @staticmethod
    def decision_name(code: torch.Tensor) -> str:
        """Map an integer decision code to its human-readable name."""
        return DECISION_NAMES[int(code.item())]
