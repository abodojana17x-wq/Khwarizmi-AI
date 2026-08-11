"""
Khwarizmi Cognitive Router Module.

Implements the learned gating policy network pi_theta(p|x, M_short) defined in
Section 4.3 and Section 5.3 of the Khwarizmi AI Blueprint.
Evaluates prompt and working state summaries to dynamically select a discrete
computational pathway, treating compute as a finite resource.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List

from ..config.settings import KhwarizmiConfig


class CognitiveRouter(nn.Module):
    """
    Learned Cognitive Router and Computational Dispatcher.

    Pathways:
        0: FAST_PATH -> Minimal compute, KSC base layers only.
        1: CODING_PATH -> Activates Coding MoE experts and Python Brain AST verification.
        2: REASONING_PATH -> Activates Adaptive Compute recurrent loops (K >= 2).
        3: PROJECT_PLAN_PATH -> Activates Long-Term Memory READ/WRITE and DAG symbolic planner.
        4: VERIFICATION_PATH -> Triggers formal AST and constraint consistency verification.

    Policy Formulation:
        pi_theta(p_c | x) = softmax(psi_c^T Enc(x, M_short) / T)

    Loss Regularization:
        L_router = lambda_cost * sum(pi * FLOPs_c) - lambda_ent * Entropy(pi)
    """

    PATHWAY_NAMES = [
        "FAST",
        "CODING",
        "REASONING",
        "PROJECT_PLAN",
        "VERIFICATION",
    ]

    # Normalized computational cost per pathway (FLOPs_c) in [0, 1]
    PATHWAY_COSTS = [0.1, 0.6, 0.7, 0.8, 0.9]

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_pathways = config.num_pathways
        self.temperature = config.temperature

        # Encoder projection layer for router policy
        self.policy_head = nn.Linear(self.d_model, self.num_pathways)

        # Register relative compute costs as non-persistent buffer
        costs = torch.tensor(self.PATHWAY_COSTS[: self.num_pathways], dtype=torch.float)
        self.register_buffer("pathway_costs", costs, persistent=False)

        self.lambda_cost = 0.05
        self.lambda_ent = 0.01

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize orthogonal policy weights."""
        nn.init.xavier_uniform_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

    def forward(
        self,
        summary_repr: torch.Tensor,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate summary representation to select computational pathway.

        Args:
            summary_repr: Combined input/working-state summary tensor of shape (batch_size, d_model).
            deterministic: If True, selects argmax pathway; otherwise samples from policy probabilities.

        Returns:
            Tuple of:
                routing_probs: Softmax policy distribution of shape (batch_size, num_pathways).
                selected_pathway: Selected pathway integer index tensor of shape (batch_size,).
                routing_loss: Scalar regularization tensor balancing FLOP cost and entropy.
        """
        if summary_repr.dim() != 2 or summary_repr.size(-1) != self.d_model:
            raise ValueError(
                f"summary_repr must have shape (batch_size, d_model), got {summary_repr.shape}"
            )

        logits = self.policy_head(summary_repr) / max(self.temperature, 1e-4)
        routing_probs = F.softmax(logits, dim=-1)

        if deterministic:
            selected_pathway = torch.argmax(routing_probs, dim=-1)
        else:
            selected_pathway = torch.multinomial(routing_probs, num_samples=1).squeeze(-1)

        routing_loss = self.compute_regularization_loss(routing_probs)

        return routing_probs, selected_pathway, routing_loss

    def compute_regularization_loss(
        self, routing_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute router regularization loss:
        Penalizes expensive computational pathways while encouraging policy entropy
        to prevent mode collapse.

        Args:
            routing_probs: Policy probabilities of shape (batch_size, num_pathways).

        Returns:
            Scalar regularization loss tensor.
        """
        # Expected compute cost per sequence in batch
        expected_cost = torch.sum(routing_probs * self.pathway_costs, dim=-1)
        mean_cost = torch.mean(expected_cost)

        # Policy entropy across batch
        log_probs = torch.log(routing_probs + 1e-9)
        entropy = -torch.sum(routing_probs * log_probs, dim=-1)
        mean_entropy = torch.mean(entropy)

        # L_router = lambda_cost * cost - lambda_ent * entropy
        loss = self.lambda_cost * mean_cost - self.lambda_ent * mean_entropy
        return loss

    @classmethod
    def get_pathway_name(cls, index: int) -> str:
        """Convert numerical pathway index to human-readable string name."""
        if 0 <= index < len(cls.PATHWAY_NAMES):
            return cls.PATHWAY_NAMES[index]
        return "UNKNOWN"
