"""
Khwarizmi Latent Reasoner Module.

Implements multi-step reasoning in latent state space rather than dumping verbose,
unverified ASCII chain-of-thought tokens, conserving inference latency and context window
as specified in Section 4.5 of the Khwarizmi AI Blueprint.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from ..config.settings import KhwarizmiConfig
from .adaptive_compute import AdaptiveComputeBlock


class LatentReasoner(nn.Module):
    """
    Latent Space Reasoning Engine.
    Executes Adaptive Recurrent Reasoning Cycles (ARRC) in latent state space
    when Cognitive Router selects REASONING_PATH (2) or VERIFICATION_PATH (4).
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.config = config
        self.adaptive_compute = AdaptiveComputeBlock(config)

    def reason(
        self,
        latent_repr: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        pathway_id: Optional[torch.Tensor] = None,
        force_cycles: Optional[int] = None,
        min_cycles: Optional[int] = None,
        max_cycles: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Execute latent reasoning over input representation if pathway requires adaptive compute.

        Args:
            latent_repr: Latent tensor of shape (batch_size, seq_len, d_model) or (batch_size, d_model).
            state: Optional KSC recurrent state.
            pathway_id: Optional pathway selection tensor of shape (batch_size,).
            force_cycles: Optional integer forcing exact recurrent cycles.
            min_cycles: Optional override for the minimum adaptive cycles K_min.
            max_cycles: Optional override for the maximum adaptive cycles K_max.

        Returns:
            Tuple of:
                reasoned_repr: Output representation after latent reasoning.
                updated_state: Updated KSC recurrent state.
                ponder_loss: Ponder cost loss tensor.
                diagnostics: Diagnostics dictionary containing cycle and halting metrics.
        """
        z_out, updated_state, ponder_loss, diagnostics = self.adaptive_compute(
            latent_repr,
            state=state,
            force_cycles=force_cycles,
            min_cycles=min_cycles,
            max_cycles=max_cycles,
        )

        # If pathway_id is provided, sequences on non-reasoning pathways (e.g. FAST=0)
        # can bypass adaptive ponder penalty
        if pathway_id is not None:
            is_reasoning = (pathway_id == 2) | (pathway_id == 4)
            if latent_repr.dim() == 2:
                mask = is_reasoning.unsqueeze(-1)
            else:
                mask = is_reasoning.view(-1, 1, 1)
            reasoned_repr = torch.where(mask, z_out, latent_repr)
            if not is_reasoning.any():
                ponder_loss = torch.tensor(0.0, device=latent_repr.device, dtype=latent_repr.dtype)
        else:
            reasoned_repr = z_out

        return reasoned_repr, updated_state, ponder_loss, diagnostics
