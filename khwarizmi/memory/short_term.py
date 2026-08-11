"""
Khwarizmi Short-Term Working State Module.

Implements the Short-Term Working State (M_short) architecture defined in
Section 4.2 of the Khwarizmi AI Blueprint. It stores ephemeral reasoning context
inside the active KSC recurrent state matrix S_t alongside a rolling window buffer
of recent normalized token representations.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from ..config.settings import KhwarizmiConfig


class ShortTermWorkingState(nn.Module):
    """
    Short-Term Working State (M_short) for ephemeral reasoning context.

    Stores:
        1. Recurrent State Matrix S_t of shape (batch_size, n_heads, d_k, d_n).
        2. Rolling window buffer of recent token embeddings of shape (batch_size, window_len, d_model).

    Provides summary representations for cognitive routing and dynamic syntactic parsing.
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.d_n = config.d_expansion
        self.max_window_len = config.max_seq_len

        # Summary projection from short-term state and buffer to d_model
        self.state_summary_proj = nn.Linear(
            self.n_heads * self.d_k * self.d_n,
            self.d_model,
        )

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Initialize short-term working state containers for a new batch.

        Args:
            batch_size: Number of concurrent sequences.
            device: Torch device.
            dtype: Tensor dtype.

        Returns:
            Dictionary containing initial recurrent_state and window_buffer tensors.
        """
        recurrent_state = torch.zeros(
            batch_size,
            self.n_heads,
            self.d_k,
            self.d_n,
            device=device,
            dtype=dtype,
        )
        window_buffer = torch.zeros(
            batch_size,
            0,
            self.d_model,
            device=device,
            dtype=dtype,
        )
        return {
            "recurrent_state": recurrent_state,
            "window_buffer": window_buffer,
        }

    def update(
        self,
        current_state: Dict[str, torch.Tensor],
        new_recurrent_state: torch.Tensor,
        new_token_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Update working state with new KSC recurrent state and recent token features.
        Applies FIFO truncation if rolling window exceeds max_seq_len.

        Args:
            current_state: Dictionary containing existing short-term state.
            new_recurrent_state: Newly updated S_t tensor of shape (batch_size, n_heads, d_k, d_n).
            new_token_features: Step token feature tensor of shape (batch_size, step_len, d_model).

        Returns:
            Updated state dictionary with new recurrent state and rolling window buffer.
        """
        if new_token_features.dim() == 2:
            new_token_features = new_token_features.unsqueeze(1)

        old_buffer = current_state["window_buffer"]
        combined_buffer = torch.cat([old_buffer, new_token_features], dim=1)

        if combined_buffer.size(1) > self.max_window_len:
            combined_buffer = combined_buffer[:, -self.max_window_len :, :]

        return {
            "recurrent_state": new_recurrent_state,
            "window_buffer": combined_buffer,
        }

    def get_summary_vector(
        self, current_state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute a compact summary vector of shape (batch_size, d_model) from the short-term
        working state for Cognitive Router input conditioning.

        Args:
            current_state: Dictionary containing current short-term state.

        Returns:
            Summary vector tensor of shape (batch_size, d_model).
        """
        rec_state = current_state["recurrent_state"]
        batch_size = rec_state.size(0)
        flat_state = rec_state.reshape(batch_size, -1)
        state_repr = self.state_summary_proj(flat_state)

        win_buffer = current_state["window_buffer"]
        if win_buffer.size(1) > 0:
            win_mean = torch.mean(win_buffer, dim=1)
            summary = 0.5 * (state_repr + win_mean)
        else:
            summary = state_repr

        return summary
