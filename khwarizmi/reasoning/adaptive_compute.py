"""
Khwarizmi Adaptive Compute and Recurrent Halting (ARRC) Module.

Implements Adaptive Recurrent Reasoning Cycles (ARRC), learned halting gates, and
ponder cost regularization as defined in Section 4.5 and Section 5.5 of the
Khwarizmi AI Blueprint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from ..config.settings import KhwarizmiConfig
from ..core.ksc_cell import KhwarizmiStateCell


class AdaptiveComputeBlock(nn.Module):
    """
    Adaptive Recurrent Reasoning Cycles (ARRC) Block.

    Recurrent Depth:
        Executes a recurrent reasoning transformation iteratively k times (k in [1, K_max])
        on the same intermediate representation.

    Learned Halting:
        h_t^(k) = sigmoid(w_h^T z^(k) + b_h)
        Stopping step K = min(k' : sum_{j=1}^k' p_j >= 1 - epsilon)

    Effective Output & Ponder Cost:
        z_out = sum_{k=1}^{K-1} p_k z^(k) + R z^(K), where remainder R = 1 - sum_{j=1}^{K-1} p_j.
        L_ponder = beta_ponder * E[K + R]
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.max_cycles = config.max_recurrent_cycles
        self.beta_ponder = config.ponder_cost_beta
        self.epsilon = 0.01

        # Recurrent reasoning transformation cell in latent space
        self.reasoning_cell = KhwarizmiStateCell(config)
        self.norm = nn.LayerNorm(config.d_model)

        # Learned halting gate projection
        self.w_halting = nn.Linear(config.d_model, 1, bias=True)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize halting gate with negative bias to encourage at least 2 cycles on complex tasks."""
        nn.init.xavier_uniform_(self.w_halting.weight)
        nn.init.constant_(self.w_halting.bias, -1.5)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        max_cycles: Optional[int] = None,
        force_cycles: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Execute Adaptive Recurrent Reasoning Cycles (ARRC) forward pass.

        Args:
            x: Input representation of shape (batch_size, seq_len, d_model) or (batch_size, d_model).
            state: Optional KSC recurrent state of shape (batch_size, n_heads, d_k, d_n).
            max_cycles: Optional override for maximum cycles K_max.
            force_cycles: Optional integer to force an exact number of recurrent cycles (for deterministic tests).

        Returns:
            Tuple of:
                z_out: Effective weighted output tensor of same shape as x.
                final_state: Updated KSC recurrent state after stopping.
                ponder_loss: Ponder cost regularization loss scalar tensor.
                diagnostics: Dictionary containing mean_cycles, halting_history, and remainders.
        """
        is_3d = x.dim() == 3
        if not is_3d:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            else:
                raise ValueError(
                    f"Input x must be 2D or 3D tensor, got shape {x.shape}"
                )

        batch_size, seq_len, _ = x.shape
        k_max = force_cycles if force_cycles is not None else (
            max_cycles if max_cycles is not None else self.max_cycles
        )

        curr_x = x
        curr_state = state

        # Accumulators across batch
        z_out = torch.zeros_like(x)
        cum_prob = torch.zeros(batch_size, 1, 1, device=x.device, dtype=x.dtype)
        halted = torch.zeros(batch_size, 1, 1, device=x.device, dtype=torch.bool)
        remainders = torch.zeros(batch_size, 1, 1, device=x.device, dtype=x.dtype)
        cycles_taken = torch.zeros(batch_size, 1, 1, device=x.device, dtype=x.dtype)

        halting_history = []

        for k in range(1, k_max + 1):
            # Recurrent transformation
            normed_x = self.norm(curr_x)
            transformed_x, curr_state, _ = self.reasoning_cell(normed_x, state=curr_state)
            curr_x = curr_x + transformed_x

            # Compute halting probability per sequence (mean across tokens if 3D)
            p_step = torch.sigmoid(self.w_halting(curr_x))  # (B, L, 1)
            p_seq = torch.mean(p_step, dim=1, keepdim=True) # (B, 1, 1)

            halting_history.append(p_seq.squeeze(-1).squeeze(-1))

            if force_cycles is not None:
                # Forced cycles: equal weighting or final step remainder
                if k == k_max:
                    weight = 1.0 - cum_prob
                    z_out = z_out + weight * curr_x
                    remainders = weight
                    cycles_taken = cycles_taken + 1.0
                else:
                    weight = p_seq
                    z_out = z_out + weight * curr_x
                    cum_prob = cum_prob + weight
                    cycles_taken = cycles_taken + 1.0
            else:
                # Standard adaptive halting
                new_cum = cum_prob + p_seq

                # For sequences halting at this cycle
                will_halt = (new_cum >= 1.0 - self.epsilon) & (~halted)
                remainders = torch.where(will_halt, 1.0 - cum_prob, remainders)

                # Weight for this cycle
                weight = torch.where(
                    will_halt,
                    1.0 - cum_prob,
                    torch.where(halted, torch.zeros_like(p_seq), p_seq),
                )

                z_out = z_out + weight * curr_x
                cum_prob = torch.where(will_halt, torch.ones_like(cum_prob), new_cum)
                cycles_taken = cycles_taken + torch.where(~halted, torch.ones_like(cycles_taken), torch.zeros_like(cycles_taken))
                halted = halted | will_halt

                if halted.all() and k >= 2:
                    break

        # If any sequence reached k_max without halting, add remainder to final z_out
        not_halted = ~halted
        if not_halted.any():
            rem = 1.0 - cum_prob
            remainders = torch.where(not_halted, rem, remainders)
            z_out = z_out + torch.where(not_halted, rem, torch.zeros_like(rem)) * curr_x

        mean_cycles = torch.mean(cycles_taken)
        mean_rem = torch.mean(remainders)

        ponder_loss = self.beta_ponder * (mean_cycles + mean_rem)

        if not is_3d:
            z_out = z_out.squeeze(1)

        diagnostics = {
            "mean_cycles": mean_cycles.item(),
            "mean_remainder": mean_rem.item(),
            "cycles_taken": cycles_taken.squeeze(-1).squeeze(-1),
            "halting_history": halting_history,
        }

        return z_out, curr_state, ponder_loss, diagnostics
