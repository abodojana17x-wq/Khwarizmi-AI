"""
Khwarizmi Short-Term Working State Module.

Implements the Short-Term Working State (M_short) architecture defined in
Section 4.2 of the Khwarizmi AI Blueprint. It stores ephemeral reasoning context
inside the active KSC recurrent state matrix S_t alongside a *bounded* rolling
window buffer of recent normalized token representations.

The rolling window buffer is strictly capacity-bounded: it never grows beyond
``config.short_term_capacity`` tokens, guaranteeing O(1)-in-sequence-length
memory for the short-term store regardless of how long the input sequence is.
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
        2. A *bounded* rolling window buffer of recent token embeddings of shape
           (batch_size, min(stored, short_term_capacity), d_model).

    Exposes explicit, testable memory operations:

    * :meth:`write` — append new token features with strict FIFO capacity eviction.
    * :meth:`read`  — retrieve the (optionally most-recent) window contents.
    * :meth:`forget` — clear the rolling window (working-state reset).
    * :meth:`update` — backward-compatible alias of :meth:`write` used by the
      full :class:`khwarizmi.core.model.KhwarizmiModel` integration.

    Provides summary representations for cognitive routing and dynamic syntactic
    parsing via :meth:`get_summary_vector`.
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.d_n = config.d_expansion
        self._capacity = config.short_term_capacity

        # Summary projection from short-term state and buffer to d_model
        self.state_summary_proj = nn.Linear(
            self.n_heads * self.d_k * self.d_n,
            self.d_model,
        )

    # ------------------------------------------------------------------ sizing
    @property
    def capacity(self) -> int:
        """Return the bounded rolling-window capacity (in token features)."""
        return self._capacity

    # ------------------------------------------------------------------- state
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

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _as_window_features(
        new_token_features: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize input to shape (batch_size, step_len, d_model)."""
        if new_token_features.dim() == 2:
            new_token_features = new_token_features.unsqueeze(1)
        return new_token_features

    def num_stored(self, current_state: Dict[str, torch.Tensor]) -> int:
        """Return the number of token features currently in the rolling window."""
        return int(current_state["window_buffer"].size(1))

    # ------------------------------------------------------------- operations
    def write(
        self,
        current_state: Dict[str, torch.Tensor],
        new_token_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        WRITE: append new token features to the rolling window with strict
        FIFO capacity eviction.

        The window is hard-bounded to ``capacity``: once full, the oldest
        features are dropped. This guarantees no unbounded growth over
        arbitrarily long sequences.

        Args:
            current_state: Dictionary containing existing short-term state.
            new_token_features: Token features of shape (batch_size, step_len, d_model)
                or (batch_size, d_model).

        Returns:
            Updated state dictionary with the appended, capacity-bounded buffer.
        """
        new_token_features = self._as_window_features(new_token_features)
        if new_token_features.size(-1) != self.d_model:
            raise ValueError(
                f"new_token_features last dim must be d_model ({self.d_model}), "
                f"got {new_token_features.size(-1)}"
            )

        old_buffer = current_state["window_buffer"]
        if old_buffer.size(-1) != self.d_model:
            raise ValueError(
                f"window_buffer last dim must be d_model ({self.d_model}), "
                f"got {old_buffer.size(-1)}"
            )

        combined_buffer = torch.cat([old_buffer, new_token_features], dim=1)
        if combined_buffer.size(1) > self._capacity:
            combined_buffer = combined_buffer[:, -self._capacity :, :]

        return {
            "recurrent_state": current_state["recurrent_state"],
            "window_buffer": combined_buffer,
        }

    def read(
        self,
        current_state: Dict[str, torch.Tensor],
        n_recent: Optional[int] = None,
    ) -> torch.Tensor:
        """
        READ: retrieve the rolling window contents (optionally the most recent
        ``n_recent`` features).

        Args:
            current_state: Dictionary containing current short-term state.
            n_recent: Optional number of most-recent features to return. If
                ``None``, the full window is returned.

        Returns:
            Tensor of shape (batch_size, min(stored, n_recent), d_model).
        """
        buffer = current_state["window_buffer"]
        if n_recent is not None:
            if n_recent < 0:
                raise ValueError(f"n_recent must be >= 0, got {n_recent}")
            buffer = buffer[:, -n_recent:, :]
        return buffer

    def forget(
        self, current_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        FORGET: clear the rolling window buffer, resetting the ephemeral
        working context (the recurrent state is preserved).

        Args:
            current_state: Dictionary containing current short-term state.

        Returns:
            Updated state dictionary with an empty window buffer.
        """
        buffer = current_state["window_buffer"]
        cleared = torch.zeros_like(buffer)[:, :0, :]
        return {
            "recurrent_state": current_state["recurrent_state"],
            "window_buffer": cleared,
        }

    def update(
        self,
        current_state: Dict[str, torch.Tensor],
        new_recurrent_state: torch.Tensor,
        new_token_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Update working state with new KSC recurrent state and recent token features.

        This is the full-model integration entry point (used by
        :class:`khwarizmi.core.model.KhwarizmiModel`); it refreshes the recurrent
        state *and* appends token features via :meth:`write`.

        Args:
            current_state: Dictionary containing existing short-term state.
            new_recurrent_state: Newly updated S_t tensor of shape (batch_size, n_heads, d_k, d_n).
            new_token_features: Step token feature tensor of shape (batch_size, step_len, d_model).

        Returns:
            Updated state dictionary with new recurrent state and rolling window buffer.
        """
        updated = self.write(current_state, new_token_features)
        updated["recurrent_state"] = new_recurrent_state
        return updated

    def get_summary_vector(
        self, current_state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute a compact summary vector of shape (batch_size, d_model) from the
        short-term working state for Cognitive Router / memory-gating conditioning.

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
