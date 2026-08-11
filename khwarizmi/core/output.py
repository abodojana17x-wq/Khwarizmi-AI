"""
Khwarizmi Output Pathway Module.

Implements the final LayerNorm, vocabulary linear projection head, output confidence
estimation C(y), and selective verification trigger logic as defined in Section 4.7
of the Khwarizmi AI Architecture Blueprint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..config.settings import KhwarizmiConfig


class OutputPathway(nn.Module):
    """
    Output Pathway for Khwarizmi AI Core.
    Transforms final sequence representations into vocabulary logits, calculates
    statistical confidence scores, and evaluates selective verification triggers.
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.config = config
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )
        self.verification_threshold = config.verification_threshold

    def forward(
        self,
        x: torch.Tensor,
        pathway_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute vocabulary logits, sequence confidence scores, and verification triggers.

        Args:
            x: Input representation of shape (batch_size, seq_len, d_model) or (batch_size, d_model).
            pathway_id: Optional integer tensor of shape (batch_size,) indicating selected pathway index
                        (0=FAST, 1=CODING, 2=REASONING, 3=PROJECT_PLAN, 4=VERIFICATION).

        Returns:
            Tuple of:
                logits: Vocabulary logit tensor of shape (batch_size, seq_len, vocab_size).
                confidence: Confidence scores C(y) of shape (batch_size,).
                needs_verification: Boolean tensor of shape (batch_size,) indicating whether
                                    optional local verification tools should be triggered.
        """
        normed_x = self.final_norm(x)
        logits = self.lm_head(normed_x)  # Shape: (B, L, V) or (B, V)

        confidence = self.estimate_confidence(logits)

        if pathway_id is None:
            batch_size = x.size(0)
            pathway_id = torch.zeros(
                batch_size, dtype=torch.long, device=x.device
            )

        needs_verification = self.check_verification_required(
            confidence, pathway_id
        )

        return logits, confidence, needs_verification

    def estimate_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Estimate statistical confidence score C(y) in [0, 1] from output logits.
        Calculates the mean of the maximum softmax probability across tokens.

        Args:
            logits: Vocabulary logits of shape (batch_size, seq_len, vocab_size) or (batch_size, vocab_size).

        Returns:
            Tensor of confidence scores C(y) of shape (batch_size,).
        """
        probs = F.softmax(logits, dim=-1)
        max_probs, _ = torch.max(probs, dim=-1)
        if max_probs.dim() > 1:
            confidence = torch.mean(max_probs, dim=1)
        else:
            confidence = max_probs
        return confidence

    def check_verification_required(
        self, confidence: torch.Tensor, pathway_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Determine whether selective verification tools should be activated.
        Triggers when:
        1. Selected pathway is CODING (1), PROJECT_PLAN (3), or VERIFICATION (4).
        2. And confidence score C(y) < verification_threshold (or verification pathway is explicitly selected).

        Args:
            confidence: Confidence score tensor of shape (batch_size,).
            pathway_id: Pathway index tensor of shape (batch_size,).

        Returns:
            Boolean tensor of shape (batch_size,) indicating verification activation.
        """
        is_coding = pathway_id == 1
        is_project_plan = pathway_id == 3
        is_verification = pathway_id == 4

        low_confidence = confidence < self.verification_threshold
        sensitive_pathway = is_coding | is_project_plan

        needs_verif = (sensitive_pathway & low_confidence) | is_verification
        return needs_verif
