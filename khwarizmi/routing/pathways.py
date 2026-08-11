"""
Khwarizmi Computational Pathway Dispatcher Module.

Maps discrete cognitive router pathway decisions to execution flags and control
signals for downstream neural core blocks and optional local tools.
"""

from dataclasses import dataclass
import torch
from typing import Dict, List, Any


@dataclass
class PathwayExecutionFlags:
    """
    Execution control flags for a batch of sequences.

    Attributes:
        use_moe: Boolean tensor of shape (batch_size,) indicating MoE block activation.
        use_adaptive_compute: Boolean tensor indicating Adaptive Recurrent Reasoning Cycles (ARRC).
        use_memory_read: Boolean tensor indicating Long-Term Memory READ eligibility.
        use_memory_write: Boolean tensor indicating Long-Term Memory WRITE eligibility.
        trigger_verification: Boolean tensor indicating selective verification eligibility.
    """
    use_moe: torch.Tensor
    use_adaptive_compute: torch.Tensor
    use_memory_read: torch.Tensor
    use_memory_write: torch.Tensor
    trigger_verification: torch.Tensor

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "use_moe": self.use_moe,
            "use_adaptive_compute": self.use_adaptive_compute,
            "use_memory_read": self.use_memory_read,
            "use_memory_write": self.use_memory_write,
            "trigger_verification": self.trigger_verification,
        }


class PathwayDispatcher:
    """
    Dispatches cognitive router pathway decisions into downstream neural execution flags.
    """

    @staticmethod
    def dispatch(
        selected_pathways: torch.Tensor,
    ) -> PathwayExecutionFlags:
        """
        Convert selected pathway indices (0=FAST, 1=CODING, 2=REASONING, 3=PROJECT_PLAN, 4=VERIFICATION)
        into boolean execution flags per sequence in the batch.

        Args:
            selected_pathways: Integer tensor of shape (batch_size,).

        Returns:
            PathwayExecutionFlags dataclass with boolean tensors of shape (batch_size,).
        """
        if selected_pathways.dim() != 1:
            raise ValueError(
                f"selected_pathways must be 1D tensor of shape (batch_size,), got {selected_pathways.shape}"
            )

        # Pathway definitions:
        # 0 = FAST: all off
        # 1 = CODING: MoE=True, MemoryRead=True, Verify=True
        # 2 = REASONING: MoE=True, Adaptive=True, MemoryRead=True, MemoryWrite=True
        # 3 = PROJECT_PLAN: MoE=True, MemoryRead=True, MemoryWrite=True, Verify=True
        # 4 = VERIFICATION: MoE=True, Adaptive=True, Verify=True
        use_moe = selected_pathways > 0
        use_adaptive_compute = (selected_pathways == 2) | (selected_pathways == 4)
        use_memory_read = (selected_pathways == 1) | (selected_pathways == 2) | (selected_pathways == 3)
        use_memory_write = (selected_pathways == 2) | (selected_pathways == 3)
        trigger_verification = (selected_pathways == 1) | (selected_pathways == 3) | (selected_pathways == 4)

        return PathwayExecutionFlags(
            use_moe=use_moe,
            use_adaptive_compute=use_adaptive_compute,
            use_memory_read=use_memory_read,
            use_memory_write=use_memory_write,
            trigger_verification=trigger_verification,
        )
