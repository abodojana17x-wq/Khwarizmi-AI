"""
Khwarizmi State Cell (KSC) Module.

Implements the sub-quadratic recurrent sequence modeling building block defined in
Section 4.1 and Section 5.1 of the Khwarizmi AI Architecture Blueprint.
KSC uses an input-selective, eigenvalue-bounded recurrent state matrix S_t
with structured diagonal Hurwitz retention gating to guarantee numerical stability
over infinite sequence lengths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from ..config.settings import KhwarizmiConfig


class KhwarizmiStateCell(nn.Module):
    """
    Khwarizmi State Cell (KSC) Recurrent Operator.

    State Representation:
        Each KSC head maintains a latent recurrent state matrix S_t in R^{d_k x d_n},
        where d_k is the state head dimension and d_n is the expansion memory bank size.

    Selective State Update:
        At token step t, input x_t is projected into query q_t, key k_t, value v_t,
        and step size Delta_t.
        The discretized state retention gate A_bar_t is bounded to [gamma_min, gamma_max]
        to ensure eigenvalues remain strictly inside the unit circle.

    Complexity:
        O(1) decoding memory footprint per sequence.
        O(D * d_n) compute per token.
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.d_n = config.d_expansion
        self.gamma_min = config.gamma_min
        self.gamma_max = config.gamma_max

        # Input projection matrices: D -> H * d_k (for q, k, delta) and H * d_n (for v)
        self.W_q = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(self.d_model, self.n_heads * self.d_n, bias=False)
        self.W_delta = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=True)

        # Base continuous dynamics parameter a in R^{H x d_k}, initialized positive
        self.log_a = nn.Parameter(torch.zeros(self.n_heads, self.d_k))

        # Output gating and projection
        self.W_g = nn.Linear(self.d_model, self.n_heads * self.d_n, bias=True)
        self.layer_norm = nn.LayerNorm(self.n_heads * self.d_n)
        self.W_o = nn.Linear(self.n_heads * self.d_n, self.d_model, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize orthogonal parameters for stability."""
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.normal_(self.log_a, mean=0.0, std=0.1)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Initialize the recurrent state tensor S_0 to zeroes.

        Args:
            batch_size: Number of sequences in the batch.
            device: Target torch device.
            dtype: Target tensor dtype.

        Returns:
            Tensor S_0 of shape (batch_size, n_heads, d_k, d_n).
        """
        return torch.zeros(
            batch_size,
            self.n_heads,
            self.d_k,
            self.d_n,
            device=device,
            dtype=dtype,
        )

    def compute_retention_gate(
        self, delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the discretized state retention gate A_bar_t bounded within
        [gamma_min, gamma_max].

        Args:
            delta: Projected step size tensor of shape (batch_size, n_heads, d_k).

        Returns:
            Bounded retention gate tensor A_bar_t of shape (batch_size, n_heads, d_k).
        """
        # softplus(delta_t) >= 0 and a_param >= 0
        delta_pos = F.softplus(delta)
        a_param = F.softplus(self.log_a)
        # unscaled decay in (0, 1)
        raw_retention = torch.exp(-delta_pos * a_param)
        # Affine bounding to strictly enforce eigenvalue limits
        bounded_retention = (
            self.gamma_min + (self.gamma_max - self.gamma_min) * raw_retention
        )
        return bounded_retention

    def step_forward(
        self, x_t: torch.Tensor, state_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute a single KSC recurrent step t.

        Args:
            x_t: Input vector at step t, shape (batch_size, d_model).
            state_prev: Previous recurrent state S_{t-1}, shape (batch_size, n_heads, d_k, d_n).

        Returns:
            Tuple of:
                y_t: Output representation at step t, shape (batch_size, d_model).
                state_new: Updated recurrent state S_t, shape (batch_size, n_heads, d_k, d_n).
                retention_gate: Bounded diagonal retention gate A_bar_t, shape (batch_size, n_heads, d_k).
        """
        if x_t.dim() != 2 or x_t.size(-1) != self.d_model:
            raise ValueError(
                f"x_t must have shape (batch_size, d_model), got {x_t.shape}"
            )
        if state_prev.shape[1:] != (self.n_heads, self.d_k, self.d_n):
            raise ValueError(
                f"state_prev shape mismatch: expected (_, {self.n_heads}, {self.d_k}, {self.d_n}), "
                f"got {state_prev.shape}"
            )

        batch_size = x_t.size(0)

        # Project inputs and reshape to multi-head format
        q = self.W_q(x_t).view(batch_size, self.n_heads, self.d_k)
        k = self.W_k(x_t).view(batch_size, self.n_heads, self.d_k)
        v = self.W_v(x_t).view(batch_size, self.n_heads, self.d_n)
        delta = self.W_delta(x_t).view(batch_size, self.n_heads, self.d_k)

        # Compute bounded diagonal retention gate
        A_t = self.compute_retention_gate(delta)  # Shape: (B, H, d_k)

        # Broadcast for state matrix update: S_t = A_t * S_{t-1} + (1 - A_t) * (k_t (x) v_t^T)
        A_t_exp = A_t.unsqueeze(-1)  # Shape: (B, H, d_k, 1)
        k_exp = k.unsqueeze(-1)      # Shape: (B, H, d_k, 1)
        v_exp = v.unsqueeze(-2)      # Shape: (B, H, 1, d_n)

        # Outer product k_t (x) v_t^T
        kv_outer = k_exp * v_exp     # Shape: (B, H, d_k, d_n)

        state_new = A_t_exp * state_prev + (1.0 - A_t_exp) * kv_outer

        # Output projection: out_head = q_t^T S_t -> shape (B, H, d_n)
        head_out = torch.einsum("b h k, b h k n -> b h n", q, state_new)
        head_out_flat = head_out.reshape(batch_size, self.n_heads * self.d_n)

        # Swish-gated linear projection
        gate = F.silu(self.W_g(x_t))
        gated_out = self.layer_norm(head_out_flat * gate)
        y_t = self.W_o(gated_out)

        return y_t, state_new, A_t

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        return_retention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Execute forward pass over an input sequence or single token.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) or (batch_size, d_model).
            state: Optional initial recurrent state S_0. If None, initializes to zeroes.
            return_retention: If True, returns retention gates for stability inspection.

        Returns:
            Tuple of:
                output: Sequence output of shape (batch_size, seq_len, d_model).
                final_state: Final recurrent state S_L of shape (batch_size, n_heads, d_k, d_n).
                retention_history: If requested, tensor of retention gates across sequence steps.
        """
        is_3d = x.dim() == 3
        if not is_3d:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            else:
                raise ValueError(
                    f"Input x must be 2D or 3D tensor, got shape {x.shape}"
                )

        batch_size, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(
                f"Input dimension d_model mismatch: expected {self.d_model}, got {d_model}"
            )

        if state is None:
            curr_state = self.init_state(batch_size, device=x.device, dtype=x.dtype)
        else:
            if state.shape != (batch_size, self.n_heads, self.d_k, self.d_n):
                raise ValueError(
                    f"State shape mismatch: expected {(batch_size, self.n_heads, self.d_k, self.d_n)}, "
                    f"got {state.shape}"
                )
            curr_state = state

        outputs = []
        retention_history = [] if return_retention else None

        for t in range(seq_len):
            x_t = x[:, t, :]
            y_t, curr_state, A_t = self.step_forward(x_t, curr_state)
            outputs.append(y_t.unsqueeze(1))
            if return_retention:
                retention_history.append(A_t.unsqueeze(1))

        out_seq = torch.cat(outputs, dim=1)
        ret_out = (
            torch.cat(retention_history, dim=1) if return_retention else None
        )

        if not is_3d:
            out_seq = out_seq.squeeze(1)

        return out_seq, curr_state, ret_out
