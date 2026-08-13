"""
Khwarizmi Residual Block Module.

Implements sequential residual blocks incorporating the Khwarizmi State Cell (KSC),
LayerNorm, and optional Sparse Mixture-of-Experts (MoE) or Feed-Forward sub-layers.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any

from ..config.settings import KhwarizmiConfig
from .ksc_cell import KhwarizmiStateCell


class FeedForwardNetwork(nn.Module):
    """Standard dense feed-forward sub-layer with Swish activation."""

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff)
        self.w2 = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)))


class KSCResidualBlock(nn.Module):
    """
    Residual block containing:
    - LayerNorm -> KhwarizmiStateCell -> Residual Addition
    - LayerNorm -> FFN / Sparse MoE -> Residual Addition
    """

    def __init__(self, config: KhwarizmiConfig, is_moe_layer: bool = False):
        super().__init__()
        self.config = config
        self.is_moe_layer = is_moe_layer

        self.norm1 = nn.LayerNorm(config.d_model)
        self.ksc = KhwarizmiStateCell(config)
        self.norm2 = nn.LayerNorm(config.d_model)

        if not is_moe_layer:
            self.ffn = FeedForwardNetwork(config)
        else:
            self.ffn = None  # Will be provided or wired via SparseMoELayer

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        moe_layer: Optional[nn.Module] = None,
        use_moe: bool = True,
        return_retention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Execute residual block forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            state: Optional KSC recurrent state of shape (batch_size, n_heads, d_k, d_n).
            moe_layer: Optional SparseMoELayer module if this block is configured for MoE.
            use_moe: Whether to route through MoE (True) or bypass MoE (e.g. FAST path).
            return_retention: Whether to return KSC retention history.

        Returns:
            Tuple of:
                x_out: Block output tensor of shape (batch_size, seq_len, d_model).
                new_state: Updated KSC recurrent state.
                aux_loss: Load balancing auxiliary loss if MoE was executed, else None.
                ret_history: Retention gates history if requested, else None.
        """
        if x.dim() not in (2, 3):
            raise ValueError(
                f"KSCResidualBlock expects a 2D or 3D input, got shape {x.shape}"
            )
        if x.size(-1) != self.config.d_model:
            raise ValueError(
                f"Input feature dimension must be d_model ({self.config.d_model}), "
                f"got {x.size(-1)}"
            )
        if state is not None and state.shape != (
            x.size(0),
            self.config.n_heads,
            self.config.d_k,
            self.config.d_expansion,
        ):
            raise ValueError(
                f"state shape mismatch: expected "
                f"{(x.size(0), self.config.n_heads, self.config.d_k, self.config.d_expansion)}, "
                f"got {tuple(state.shape)}"
            )

        # Step 1: KSC sequence sub-layer with pre-layer normalization
        normed_x = self.norm1(x)
        ksc_out, new_state, ret_history = self.ksc(
            normed_x, state=state, return_retention=return_retention
        )
        x = x + ksc_out

        # Step 2: Feed-Forward or MoE sub-layer with pre-layer normalization
        normed_x2 = self.norm2(x)
        aux_loss = None

        if self.is_moe_layer and moe_layer is not None and use_moe:
            moe_out, aux_loss = moe_layer(normed_x2)
            x = x + moe_out
        elif self.ffn is not None:
            ffn_out = self.ffn(normed_x2)
            x = x + ffn_out

        return x, new_state, aux_loss, ret_history
