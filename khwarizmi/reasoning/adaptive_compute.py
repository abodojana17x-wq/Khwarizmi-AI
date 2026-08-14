"""
Khwarizmi Adaptive Compute and Recurrent Halting (ARRC) Module — Phase 5.

Implements Adaptive Recurrent Reasoning Cycles (ARRC): per-token learned halting
gates with an ACT-style (Adaptive Computation Time, Graves 2016) remainder
formulation and ponder cost regularization, as defined in Section 4.5 and
Section 5.5 of the Khwarizmi AI Blueprint.

Mathematical specification (ARCHITECTURE.md §5.5):

    Halting probability at cycle k for latent state z^(k):
        p_k = sigmoid(w_h^T z^(k) + b_h)

    Stopping step:
        K = min { k' : sum_{j=1}^{k'} p_j >= 1 - epsilon }   (hard-capped at K_max)

    Effective output with remainder R = 1 - sum_{j=1}^{K-1} p_j:
        z_out = sum_{k=1}^{K-1} p_k z^(k) + R z^(K)

    Ponder cost:
        L_ponder = beta_ponder * E[ N + R ]

    where N is the number of executed cycles. R is differentiable w.r.t. the
    halting gate parameters, so minimizing L_ponder pressures the gate towards
    earlier halting; N accounts for the discrete cycle count.

Guarantees enforced by this module:
    * Termination: recurrence always halts by ``max_recurrent_cycles`` (K_max);
      an unfinished token is force-halted on the final cycle with its remainder.
    * Minimum compute: no token can halt before ``min_recurrent_cycles`` (K_min).
    * Valid probability accounting: the per-token accumulated halting
      probability never exceeds 1 and the per-cycle output weights always sum
      to exactly 1 for every token.
    * Determinism: the forward pass contains no sampling; identical inputs and
      parameters produce identical outputs, step counts, and halting scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from ..config.settings import KhwarizmiConfig
from ..core.ksc_cell import KhwarizmiStateCell


class PonderCostLoss(nn.Module):
    """
    ACT Ponder Cost Regularizer (Phase 5 roadmap deliverable).

    Computes  L_ponder = beta_ponder * mean( N + R )  where:
        N: per-token number of executed recurrent cycles (discrete, detached),
        R: per-token halting remainder (differentiable through the halting gate).

    The remainder term is the differentiable pressure: increasing early halting
    probabilities reduces R (and eventually N), so unnecessary computation is
    penalized through gradient descent rather than a hand-tuned heuristic.
    """

    def __init__(self, beta_ponder: float):
        super().__init__()
        if not (beta_ponder >= 0.0 and beta_ponder == beta_ponder):
            raise ValueError(
                f"beta_ponder must be a finite non-negative value, got {beta_ponder}"
            )
        self.beta_ponder = float(beta_ponder)

    def forward(
        self, n_updates: torch.Tensor, remainders: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            n_updates: Per-token executed cycle counts, shape (batch, seq_len).
            remainders: Per-token halting remainders R in [0, 1], shape (batch, seq_len).

        Returns:
            Scalar ponder cost loss tensor.
        """
        if n_updates.shape != remainders.shape:
            raise ValueError(
                f"n_updates shape {tuple(n_updates.shape)} must match "
                f"remainders shape {tuple(remainders.shape)}"
            )
        # N is a discrete step count — detach so no spurious gradient path;
        # R carries the differentiable halting-gate gradient.
        ponder = n_updates.detach() + remainders
        return self.beta_ponder * torch.mean(ponder)


class AdaptiveComputeBlock(nn.Module):
    """
    Adaptive Recurrent Reasoning Cycles (ARRC) Block with per-token ACT halting.

    Recurrent Depth:
        A shared KSC reasoning cell is applied iteratively k times
        (k in [K_min, K_max]) to the same latent representation. Tokens that
        have halted are frozen: their latent state stops updating and they
        contribute zero weight on subsequent cycles.

    Learned Halting:
        h^(k) = sigmoid(w_h^T z^(k) + b_h), accumulated per token. A token
        halts at the first cycle k >= K_min where the accumulated probability
        reaches 1 - epsilon, and is force-halted at K_max with the remainder
        weight R = 1 - sum_{j<K} p_j. Different tokens may therefore receive
        different amounts of computation within one batch/sequence.

    Effective Output & Ponder Cost:
        z_out = sum_{k<K} p_k z^(k) + R z^(K)  per token; the per-token output
        weights always sum to exactly 1. L_ponder = beta * E[N + R].
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.max_cycles = config.max_recurrent_cycles
        self.min_cycles = config.min_recurrent_cycles
        self.beta_ponder = config.ponder_cost_beta
        self.epsilon = config.halting_epsilon

        # Recurrent reasoning transformation cell in latent space
        self.reasoning_cell = KhwarizmiStateCell(config)
        self.norm = nn.LayerNorm(config.d_model)

        # Learned halting gate projection (per-token scalar logit)
        self.w_halting = nn.Linear(config.d_model, 1, bias=True)

        # Ponder cost loss module (roadmap Phase 5 deliverable)
        self.ponder_cost = PonderCostLoss(config.ponder_cost_beta)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize halting gate with negative bias to encourage at least 2 cycles on complex tasks."""
        nn.init.xavier_uniform_(self.w_halting.weight)
        nn.init.constant_(self.w_halting.bias, -1.5)

    def halting_probability(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the per-token halting probability p = sigmoid(w_h^T z + b_h).

        Args:
            z: Latent tensor of shape (batch, seq_len, d_model).

        Returns:
            Halting probabilities in (0, 1) of shape (batch, seq_len).
        """
        return torch.sigmoid(self.w_halting(z)).squeeze(-1)

    @staticmethod
    def _validate_cycle_bounds(
        min_cycles: int, max_cycles: int
    ) -> None:
        if min_cycles < 1:
            raise ValueError(f"min_cycles must be >= 1, got {min_cycles}")
        if max_cycles < min_cycles:
            raise ValueError(
                f"max_cycles ({max_cycles}) must be >= min_cycles ({min_cycles})"
            )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        max_cycles: Optional[int] = None,
        force_cycles: Optional[int] = None,
        min_cycles: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Execute Adaptive Recurrent Reasoning Cycles (ARRC) forward pass.

        Args:
            x: Input representation of shape (batch_size, seq_len, d_model) or (batch_size, d_model).
            state: Optional KSC recurrent state of shape (batch_size, n_heads, d_k, d_n).
            max_cycles: Optional override for maximum cycles K_max.
            force_cycles: Optional integer to force an exact number of recurrent
                cycles for every token (deterministic fixed-compute mode).
            min_cycles: Optional override for minimum cycles K_min.

        Returns:
            Tuple of:
                z_out: Effective ACT-weighted output tensor of same shape as x.
                final_state: Updated KSC recurrent state after stopping.
                ponder_loss: Ponder cost regularization loss scalar tensor.
                diagnostics: Dictionary containing mean_cycles, per-token
                    cycles_taken, remainders, accumulated halting probability,
                    halting_history, and the per-step halting distribution.
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

        if force_cycles is not None:
            if force_cycles < 1:
                raise ValueError(f"force_cycles must be >= 1, got {force_cycles}")
            k_max = force_cycles
            k_min = force_cycles
        else:
            k_max = max_cycles if max_cycles is not None else self.max_cycles
            k_min = min_cycles if min_cycles is not None else self.min_cycles
            # Config-level overrides must remain internally consistent, but an
            # explicit max_cycles override lower than the configured minimum
            # tightens the minimum too (hard cap always wins).
            if max_cycles is not None and min_cycles is None:
                k_min = min(k_min, k_max)
            self._validate_cycle_bounds(k_min, k_max)

        forced = force_cycles is not None

        curr_x = x
        curr_state = state

        device, dtype = x.device, x.dtype

        # Per-token ACT accounting — shape (B, L)
        z_out = torch.zeros_like(x)
        cum_prob = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        remainders = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        n_updates = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        still_running = torch.ones(
            batch_size, seq_len, device=device, dtype=torch.bool
        )

        halting_history = []
        halted_at_step = torch.zeros(
            batch_size, seq_len, device=device, dtype=torch.long
        )

        final_state = curr_state

        for k in range(1, k_max + 1):
            run_mask = still_running.to(dtype=dtype).unsqueeze(-1)  # (B, L, 1)

            # Recurrent transformation; halted tokens are frozen (no update).
            normed_x = self.norm(curr_x)
            transformed_x, new_state, _ = self.reasoning_cell(
                normed_x, state=curr_state
            )
            curr_x = curr_x + run_mask * transformed_x

            # Freeze the recurrent state of sequences whose tokens have ALL
            # halted so no stale state drift leaks between batch examples.
            if curr_state is not None:
                seq_running = still_running.any(dim=1).view(-1, 1, 1, 1)
                new_state = torch.where(seq_running, new_state, curr_state)
            curr_state = new_state
            final_state = new_state

            # Per-token halting probability p^(k) in (0, 1) — shape (B, L)
            p_step = self.halting_probability(curr_x)
            halting_history.append(p_step)

            is_last = k == k_max

            if forced:
                # Fixed-compute mode: no early halting; every token halts at
                # exactly k_max with the ACT remainder. Intermediate weights
                # are clamped so accumulated probability never exceeds 1.
                if is_last:
                    weight = (1.0 - cum_prob) * still_running.to(dtype=dtype)
                    remainders = torch.where(still_running, 1.0 - cum_prob, remainders)
                    halted_at_step = torch.where(
                        still_running,
                        torch.full_like(halted_at_step, k),
                        halted_at_step,
                    )
                    cum_prob = torch.where(
                        still_running, torch.ones_like(cum_prob), cum_prob
                    )
                    n_updates = n_updates + still_running.to(dtype=dtype)
                    z_out = z_out + weight.unsqueeze(-1) * curr_x
                    still_running = torch.zeros_like(still_running)
                else:
                    p_eff = torch.minimum(p_step, 1.0 - cum_prob)
                    weight = p_eff * still_running.to(dtype=dtype)
                    z_out = z_out + weight.unsqueeze(-1) * curr_x
                    cum_prob = cum_prob + weight
                    n_updates = n_updates + still_running.to(dtype=dtype)
                continue

            # ---- Adaptive halting path ----
            # Minimum-compute enforcement: halting mass is only allowed to
            # accumulate from cycle K_min onwards.
            if k < k_min:
                p_eff = torch.zeros_like(p_step)
            else:
                p_eff = p_step

            new_cum = cum_prob + p_eff * still_running.to(dtype=dtype)

            # Tokens crossing the threshold this cycle (or hitting the hard cap)
            threshold_halt = new_cum >= (1.0 - self.epsilon)
            can_halt = k >= k_min
            halt_now = still_running & ((threshold_halt & can_halt) | is_last)

            # ACT weights: remainder for halting tokens, p for running tokens.
            remainder_weight = 1.0 - cum_prob
            weight = torch.where(
                halt_now,
                remainder_weight,
                p_eff * still_running.to(dtype=dtype),
            )

            z_out = z_out + weight.unsqueeze(-1) * curr_x

            remainders = torch.where(halt_now, remainder_weight, remainders)
            halted_at_step = torch.where(
                halt_now, torch.full_like(halted_at_step, k), halted_at_step
            )
            cum_prob = torch.where(halt_now, torch.ones_like(cum_prob), new_cum)
            n_updates = n_updates + still_running.to(dtype=dtype)
            still_running = still_running & ~halt_now

            if not bool(still_running.any()):
                break

        mean_cycles = torch.mean(n_updates)
        mean_rem = torch.mean(remainders)

        ponder_loss = self.ponder_cost(n_updates, remainders)

        # Per-step halting distribution (counts of tokens halting at each cycle)
        step_histogram = [
            int(torch.sum(halted_at_step == step).item())
            for step in range(1, k_max + 1)
        ]

        if not is_3d:
            z_out = z_out.squeeze(1)

        diagnostics = {
            "mean_cycles": mean_cycles.item(),
            "mean_remainder": mean_rem.item(),
            "cycles_taken": n_updates,
            "remainders": remainders,
            "accumulated_halting_prob": cum_prob,
            "halted_at_step": halted_at_step,
            "step_histogram": step_histogram,
            "halting_history": halting_history,
            "min_cycles": k_min,
            "max_cycles": k_max,
        }

        return z_out, final_state, ponder_loss, diagnostics
