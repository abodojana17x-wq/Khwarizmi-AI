"""
Khwarizmi Memory Gating Controller Module.

Implements the learned gating network for the Long-Term Persistent Memory store as
defined in Section 5.2 of the Khwarizmi AI Blueprint. It outputs continuous/discrete
probabilities for READ, WRITE, UPDATE, and FORGET operations conditioned on the
current latent state and task context.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from ..config.settings import KhwarizmiConfig


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
