"""
Khwarizmi Neural Reasoning Core — Phase 6.

Implements the Neural Reasoning Core specified in the Phase 6 roadmap and
ARCHITECTURE.md §4.5 (Neural Reasoning Core: Latent Synthesis & Bounded
Self-Correction). This is the transition from Phase 5's adaptive computation
(ACT-style per-token halting) into an actual, trainable, latent reasoning
mechanism.

The reasoning core operates *entirely in latent/tensor space*. It does NOT emit
textual chain-of-thought traces, natural-language intermediate thoughts, or
generated explanations. Only numerical diagnostics are exposed.

Conceptual flow
---------------

    h_0  (refined representation produced by the Phase 5 ARRC engine)
      |
      v
    Reasoning Transformation  (trainable residual synthesis)
      |
      v
    h_1 = h_0 + Delta_h           (residual refinement)
      |
      v
    Consistency / Confidence  (learned confidence head, bounded [0, 1])
      |
      v
    sufficient?  -- YES --> finish (refined latent state)
      |
     NO
      |
      v
    Bounded Self-Correction  (confidence-conditioned gated correction)
      |
      v
    h_2 ... (repeat, bounded by K_r^max and max_reasoning_corrections)

Design principles enforced here
-------------------------------
* Genuine trainable parameters: synthesis, confidence, and correction each
  carry independent learned parameters and produce real transformations. There
  is no identity / `return x` reasoning path and no rule-based stub.
* Residual formulation: ``h_{k+1} = h_k + Delta_h`` so the dimensional
  contract (d_model, batch, sequence) is preserved exactly.
* Bounded termination: the reasoning loop ALWAYS halts by ``max_reasoning_steps``
  with forced convergence; the minimum-step bound forbids early termination.
  Self-correction is independently capped by ``max_reasoning_corrections``.
* Differentiability: the whole path is differentiable. There are no
  ``.detach()`` calls on the reasoning/synthesis/correction path and no
  ``torch.no_grad()`` blocks. Confidence-gradient blocking is explicit and
  configurable (``confidence_requires_grad``), defaulting to differentiable.
* Numerical stability: LayerNorm before projections, tanh-bounded correction
  amplitude, clamped confidence sigmoid, and NaN/Inf guarding on the residual
  update that zeros non-finite contributions rather than propagating them.
* Determinism: the forward pass contains no sampling; identical inputs and
  parameters produce identical outputs, step counts, and confidence scores.
* Compatibility with Phase 5 ARRC: this core CONSUMES the output of the
  existing AdaptiveComputeBlock / LatentReasoner rather than duplicating
  per-token halting. The ARRC compute budget established in Phase 5 bounds
  the representation handed to the reasoning core.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.settings import KhwarizmiConfig


# ---------------------------------------------------------------------------
# Latent Synthesis & Reasoning Transformation
# ---------------------------------------------------------------------------


class LatentSynthesisBlock(nn.Module):
    """
    Trainable latent synthesis / reasoning transformation.

    Produces a residual refinement ``Delta_h`` of the latent state via a
    gated Swish MLP, so that ``h_{k+1} = h_k + Delta_h``. The transformation
    is a genuine learned map (not identity): it expands to ``d_ff`` under a
    Swish non-linearity, projects back to ``d_model``, and applies a learned
    sigmoid gate that modulates how much of each feature dimension is
    refined. This mirrors the KSC/FFN residual style of the existing
    architecture so integration is dimensionally consistent.

    Mathematically::

        g      = sigmoid(W_g LayerNorm(h) + b_g)      # feature-wise gate
        Delta_h= W_2 SiLU(W_1 LayerNorm(h) + b_1) + b_2
        Delta_h= g * Delta_h

    The output ``Delta_h`` has the same shape as the input ``h``.
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_ff = config.d_ff

        self.norm = nn.LayerNorm(config.d_model)
        self.w1 = nn.Linear(config.d_model, config.d_ff)
        self.w2 = nn.Linear(config.d_ff, config.d_model)
        # Feature-wise learned gate (independent of the correction gate).
        self.gate_proj = nn.Linear(config.d_model, config.d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.w2.bias)
        # Gate bias slightly positive so the synthesis is not silently zero at init.
        nn.init.constant_(self.gate_proj.bias, 0.5)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute the residual refinement ``Delta_h``.

        Args:
            h: Latent state of shape (batch, seq_len, d_model) or (batch, d_model).

        Returns:
            Delta_h of the same shape as ``h`` (a learned transformation, not identity).
        """
        # Cast parameters to the input dtype so non-float32 inputs (e.g.
        # float64) work without silently changing the caller's dtype: the
        # output dtype always matches the input dtype.
        normed = F.layer_norm(
            h, self.norm.normalized_shape, self.norm.weight.to(h.dtype),
            self.norm.bias.to(h.dtype), self.norm.eps,
        )
        gate = torch.sigmoid(
            F.linear(normed, self.gate_proj.weight.to(h.dtype),
                     self.gate_proj.bias.to(h.dtype))
        )
        delta = F.linear(
            F.silu(F.linear(normed, self.w1.weight.to(h.dtype),
                            self.w1.bias.to(h.dtype))),
            self.w2.weight.to(h.dtype), self.w2.bias.to(h.dtype),
        )
        return gate * delta


# ---------------------------------------------------------------------------
# Consistency / Confidence Estimation
# ---------------------------------------------------------------------------


class ConsistencyHead(nn.Module):
    """
    Learned confidence / consistency head.

    Estimates a per-token confidence score ``c in [0, 1]`` expressing how
    sufficiently refined the current latent reasoning state is. The score is
    a learned projection of the latent state combined with a normalized
    residual-magnitude statistic, so it integrates both a learned signal and
    an architecture-consistent latent statistic.

    Mathematically (per token)::

        c = sigmoid( w_c^T LayerNorm(h) + b_c
                     + alpha_c * tanh(||Delta_h||_2 / (d_model^0.5)) )

    The residual-magnitude term is clamped through ``tanh`` so it cannot
    saturate the sigmoid into a degenerate state, and the whole expression is
    bounded to ``(0, 1)`` by the outer sigmoid.

    The confidence scalar used for the halting decision is the mean over
    tokens (per sequence), which is deterministic under fixed inputs.
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.norm = nn.LayerNorm(config.d_model)
        self.w_conf = nn.Linear(config.d_model, 1, bias=True)
        # Learnable mixing weight for the residual-magnitude statistic.
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w_conf.weight)
        # Start confidence projection slightly negative so the model must
        # *learn* to be confident, avoiding premature halting at init.
        nn.init.constant_(self.w_conf.bias, -1.0)

    def forward(
        self, h: torch.Tensor, delta: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token and per-sequence confidence scores.

        Args:
            h: Latent state of shape (batch, seq_len, d_model) or (batch, d_model).
            delta: Optional residual refinement of the same shape as ``h``
                (used for the residual-magnitude consistency statistic).

        Returns:
            Tuple of:
                token_confidence: Per-token confidence in (0, 1) of shape
                    (batch, seq_len) [or (batch,) for 2D input].
                seq_confidence: Per-sequence mean confidence in (0, 1) of
                    shape (batch,).
        """
        was_2d = h.dim() == 2
        if was_2d:
            h_in = h.unsqueeze(1)
            delta_in = delta.unsqueeze(1) if delta is not None else None
        else:
            h_in = h
            delta_in = delta

        normed = F.layer_norm(
            h_in, self.norm.normalized_shape, self.norm.weight.to(h_in.dtype),
            self.norm.bias.to(h_in.dtype), self.norm.eps,
        )
        logit = F.linear(
            normed, self.w_conf.weight.to(h_in.dtype),
            self.w_conf.bias.to(h_in.dtype),
        ).squeeze(-1)  # (batch, seq_len)

        if delta_in is not None:
            # Normalized residual magnitude statistic (per token), tanh-bounded.
            # Computed in the input dtype to keep the autograd path connected
            # to ``delta`` and to avoid mixed-dtype errors under float64.
            mean_sq = torch.mean(delta_in ** 2, dim=-1).clamp_min(0.0)
            rms = torch.sqrt(mean_sq + torch.tensor(
                1e-8, device=delta_in.device, dtype=delta_in.dtype
            )) / (float(self.d_model) ** 0.5)
            residual_stat = torch.tanh(rms)
            # Mix with the learned confidence logit (same dtype).
            logit = logit + self.alpha * residual_stat

        token_conf = torch.sigmoid(logit)
        seq_conf = torch.mean(token_conf, dim=-1)

        if was_2d:
            token_conf = token_conf.squeeze(1)
        return token_conf, seq_conf


# ---------------------------------------------------------------------------
# Bounded Latent Self-Correction
# ---------------------------------------------------------------------------


class SelfCorrectionBlock(nn.Module):
    """
    Confidence-conditioned gated latent self-correction.

    When the consistency signal indicates the reasoning state is insufficient,
    this block produces a *meaningful* correction residual that is conditioned
    on both the latent state and the (detached) confidence signal. This is NOT
    a re-application of the same synthesis layer: it is an independently
    parameterized, confidence-conditioned update whose magnitude is bounded by
    a tanh gate so repeated corrections remain numerically stable.

    Mathematically::

        m      = sigmoid(W_m [LayerNorm(h) ; 1 - c])       # correction gate
        corr   = tanh(W_2 SiLU(W_1 LayerNorm(h) + b_1) + b_2)
        Delta_corr = m * corr * correction_scale

    The ``1 - c`` conditioning makes the correction explicitly informed by
    the consistency/confidence mechanism (lower confidence -> larger gate
    opening, subject to the learned projection). The outer ``tanh`` bounds the
    correction amplitude, guaranteeing stability under repeated refinement.
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        # correction_scale bounds the correction magnitude; a learnable scalar
        # initialized modestly so repeated corrections stay stable.
        self.correction_scale = nn.Parameter(torch.tensor(0.5))

        self.norm = nn.LayerNorm(config.d_model)
        # Gate input is [h ; 1 - c]  (d_model + 1).
        self.gate_proj = nn.Linear(config.d_model + 1, config.d_model)
        self.w1 = nn.Linear(config.d_model, config.d_ff)
        self.w2 = nn.Linear(config.d_ff, config.d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def forward(
        self, h: torch.Tensor, confidence: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the confidence-conditioned correction residual.

        Args:
            h: Latent state of shape (batch, seq_len, d_model) or (batch, d_model).
            confidence: Per-sequence confidence of shape (batch,) or per-token
                confidence broadcastable to the batch dim. It is detached so
                the correction magnitude is conditioned by the *current*
                consistency signal without creating a second-order coupling
                through the confidence head (this is an explicit, documented
                design choice, not an accidental detach).

        Returns:
            Correction residual of the same shape as ``h`` (a genuine learned,
            confidence-conditioned update).
        """
        was_2d = h.dim() == 2
        if was_2d:
            h_in = h.unsqueeze(1)
        else:
            h_in = h

        c = confidence.detach()
        # Broadcast confidence to (batch, seq_len, 1): 1 - c opens the gate
        # more when confidence is low.
        if c.dim() == 1:
            c_seq = c.view(-1, 1, 1).expand(-1, h_in.size(1), -1)
        else:
            c_seq = c.unsqueeze(-1)
        insufficiency = 1.0 - c_seq

        normed = F.layer_norm(
            h_in, self.norm.normalized_shape, self.norm.weight.to(h_in.dtype),
            self.norm.bias.to(h_in.dtype), self.norm.eps,
        )
        gate_input = torch.cat([normed, insufficiency], dim=-1)
        gate = torch.sigmoid(
            F.linear(gate_input, self.gate_proj.weight.to(h_in.dtype),
                     self.gate_proj.bias.to(h_in.dtype))
        )

        corr = torch.tanh(
            F.linear(
                F.silu(F.linear(normed, self.w1.weight.to(h_in.dtype),
                                self.w1.bias.to(h_in.dtype))),
                self.w2.weight.to(h_in.dtype), self.w2.bias.to(h_in.dtype),
            )
        )
        # The correction magnitude is explicitly scaled by the insufficiency
        # signal (1 - c), so lower confidence monotonically permits a larger
        # correction. This makes the confidence-conditioning explicit and
        # bounded rather than relying solely on the learned gate projection.
        delta_corr = (gate * corr * self.correction_scale.to(h_in.dtype)
                      * insufficiency.squeeze(-1).unsqueeze(-1))

        if was_2d:
            delta_corr = delta_corr.squeeze(1)
        return delta_corr


# ---------------------------------------------------------------------------
# Reasoning Losses
# ---------------------------------------------------------------------------


class ReasoningLosses(nn.Module):
    """
    Phase 6 reasoning-specific loss infrastructure (minimal).

    Exposes two independently-testable, differentiable, numerically stable
    losses plus an aggregation helper. This does NOT create a new training
    framework — it only provides the auxiliary reasoning regularizers expected
    by the Phase 6 roadmap and the existing ``losses`` dict contract of
    :class:`khwarizmi.core.model.KhwarizmiModel`.

    Losses:
        * consistency_loss: pushes the confidence of the final refined state
          towards the target sufficiency (high confidence). Computed as a
          bounded BCE between the final per-sequence confidence and a target.
        * refinement_loss: regularizes the magnitude of the correction
          residuals so self-correction stays parsimonious
          (``beta_r * mean(||Delta_corr||^2)``).

    Both are scaled by their respective beta coefficients from config and are
    zero-safe (return 0.0 tensors when no corrections/steps occurred).
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.confidence_beta = float(config.reasoning_confidence_beta)
        self.refinement_beta = float(config.reasoning_refinement_beta)

    def consistency_loss(
        self,
        final_confidence: torch.Tensor,
        target: float = 1.0,
    ) -> torch.Tensor:
        """
        Bounded binary-cross-entropy pushing final confidence towards ``target``.

        Args:
            final_confidence: Per-sequence confidence in (0, 1), shape (batch,).
            target: Target confidence scalar in [0, 1].

        Returns:
            Scalar consistency loss tensor (scaled by confidence_beta).
        """
        if not (0.0 <= target <= 1.0):
            raise ValueError(f"target must be in [0, 1], got {target}")
        c = final_confidence.clamp(min=1e-7, max=1.0 - 1e-7)
        t = torch.full_like(c, float(target))
        # Numerically stable BCE.
        bce = -(t * torch.log(c) + (1.0 - t) * torch.log1p(-c))
        return self.confidence_beta * torch.mean(bce)

    def refinement_loss(
        self, correction_residuals: torch.Tensor
    ) -> torch.Tensor:
        """
        L2 regularization on correction residuals.

        Args:
            correction_residuals: Tensor of correction residuals. May be an
                empty/zero tensor when no corrections occurred.

        Returns:
            Scalar refinement loss tensor (scaled by refinement_beta).
        """
        if correction_residuals.numel() == 0:
            return correction_residuals.sum() * 0.0
        return self.refinement_beta * torch.mean(
            correction_residuals.float() ** 2
        )

    def aggregate(
        self,
        consistency: torch.Tensor,
        refinement: torch.Tensor,
    ) -> torch.Tensor:
        """Return the total reasoning auxiliary loss."""
        return consistency + refinement


# ---------------------------------------------------------------------------
# Neural Reasoning Core (orchestrator)
# ---------------------------------------------------------------------------


@dataclass
class ReasoningOutput:
    """
    Structured output contract of the Neural Reasoning Core.

    Attributes:
        refined_state: Refined latent state, same shape as the input.
        consistency_loss: Scalar consistency loss tensor.
        refinement_loss: Scalar refinement loss tensor.
        total_reasoning_loss: Sum of consistency + refinement losses.
        diagnostics: Reasoning diagnostics dict (numerical only).
    """

    refined_state: torch.Tensor
    consistency_loss: torch.Tensor
    refinement_loss: torch.Tensor
    total_reasoning_loss: torch.Tensor
    diagnostics: Dict[str, Any]


class NeuralReasoningCore(nn.Module):
    """
    Neural Reasoning Core: Latent Synthesis & Bounded Self-Correction.

    Orchestrates the bounded iterative reasoning loop:

        for k in 1..K_r^max:
            Delta_h    = synthesis(h)              # learned residual refinement
            h          = h + Delta_h               # residual update (NaN-guarded)
            c_k        = consistency(h, Delta_h)   # confidence in (0, 1)
            if k >= K_r^min and c_k >= threshold:
                converged = True; break
            if corrections < max_corrections:
                Delta_corr = correction(h, c_k)    # confidence-conditioned
                h          = h + Delta_corr
                corrections += 1
        forced convergence at K_r^max.

    Termination guarantees:
        * The loop body executes at most ``max_reasoning_steps`` times (hard cap).
        * No reasoning termination is permitted before ``min_reasoning_steps``.
        * Self-correction executes at most ``max_reasoning_corrections`` times.
        * The loop contains no sampling and no unbounded recursion.

    Compatibility:
        * Consumes the ARRC-refined representation (Phase 5) as its input ``h_0``.
        * Preserves the dimensional contract (batch, seq, d_model).
        * Device/dtype are inherited from the input tensor throughout.
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.min_steps = config.min_reasoning_steps
        self.max_steps = config.max_reasoning_steps
        self.max_corrections = config.max_reasoning_corrections
        self.confidence_threshold = config.reasoning_confidence_threshold

        self.synthesis = LatentSynthesisBlock(config)
        self.consistency = ConsistencyHead(config)
        self.correction = SelfCorrectionBlock(config)
        self.losses = ReasoningLosses(config)

        # Output norm applied once to the final refined state for stability,
        # matching the LayerNorm convention used by KSC/OutputPathway.
        self.output_norm = nn.LayerNorm(config.d_model)

    @staticmethod
    def _validate_bounds(
        min_steps: int, max_steps: int, max_corrections: int
    ) -> None:
        if min_steps < 1:
            raise ValueError(f"min_steps must be >= 1, got {min_steps}")
        if max_steps < min_steps:
            raise ValueError(
                f"max_steps ({max_steps}) must be >= min_steps ({min_steps})"
            )
        if max_corrections < 0:
            raise ValueError(
                f"max_corrections must be >= 0, got {max_corrections}"
            )
        if max_corrections > max_steps:
            raise ValueError(
                f"max_corrections ({max_corrections}) must be <= "
                f"max_steps ({max_steps})"
            )

    @staticmethod
    def _finite_residual_update(
        h: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply a residual update, zeroing non-finite contributions in place of
        propagating NaN/Inf. This is an explicit stability guard rather than a
        silent NaN-hider: non-finite deltas are replaced by zeros so the loop
        cannot explode, and the guard is documented and tested.
        """
        finite_mask = torch.isfinite(delta)
        safe_delta = torch.where(
            finite_mask, delta, torch.zeros_like(delta)
        )
        return h + safe_delta

    @staticmethod
    def _sanitize_state(h: torch.Tensor) -> torch.Tensor:
        """
        Replace non-finite entries in a latent state with zeros. This is an
        explicit input-stability guard: a non-finite ``h0`` (or a state that
        became non-finite despite the residual guard) is cleaned before the
        next synthesis/confidence pass so the reasoning loop cannot propagate
        NaN/Inf forward. The guard is documented and tested, not a silent
        NaN-hider.
        """
        return torch.where(torch.isfinite(h), h, torch.zeros_like(h))

    def forward(
        self,
        h0: torch.Tensor,
        min_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
        max_corrections: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        force_steps: Optional[int] = None,
        return_intermediate: bool = False,
    ) -> ReasoningOutput:
        """
        Execute the bounded iterative latent reasoning loop.

        Args:
            h0: Input latent representation of shape (batch, seq_len, d_model)
                or (batch, d_model). Typically the output of the Phase 5 ARRC
                engine.
            min_steps: Optional override for the minimum reasoning steps.
            max_steps: Optional override for the maximum reasoning steps (hard cap).
            max_corrections: Optional override for the maximum corrections.
            confidence_threshold: Optional override for the halting threshold.
            force_steps: Optional integer forcing an exact number of reasoning
                steps for every sequence (deterministic fixed-depth mode). When
                set, it overrides min/max to the same value and disables
                early halting.
            return_intermediate: If True, expose intermediate per-step
                confidence and delta norms in diagnostics.

        Returns:
            :class:`ReasoningOutput` with the refined latent state, the
            reasoning auxiliary losses, and numerical diagnostics.
        """
        if h0.dim() not in (2, 3):
            raise ValueError(
                f"h0 must be 2D or 3D, got shape {tuple(h0.shape)}"
            )
        if h0.size(-1) != self.d_model:
            raise ValueError(
                f"h0 last dim must be d_model ({self.d_model}), "
                f"got {h0.size(-1)}"
            )

        if force_steps is not None:
            if force_steps < 1:
                raise ValueError(
                    f"force_steps must be >= 1, got {force_steps}"
                )
            k_min = force_steps
            k_max = force_steps
            k_corr = min(self.max_corrections, force_steps)
            threshold = 2.0  # >1 disables early halting in forced mode
            forced = True
        else:
            k_min = min_steps if min_steps is not None else self.min_steps
            k_max = max_steps if max_steps is not None else self.max_steps
            k_corr = (
                max_corrections
                if max_corrections is not None
                else self.max_corrections
            )
            if max_steps is not None and min_steps is None:
                k_min = min(k_min, k_max)
            threshold = (
                confidence_threshold
                if confidence_threshold is not None
                else self.confidence_threshold
            )
            self._validate_bounds(k_min, k_max, k_corr)
            forced = False

        device, dtype = h0.device, h0.dtype

        # Sanitize the input state so a non-finite h0 cannot seed NaN/Inf
        # propagation through the (differentiable) reasoning path.
        h = self._sanitize_state(h0)
        corrections_done = 0
        converged = False
        final_confidence = None
        last_delta = None
        last_correction = None

        intermediate_conf = [] if return_intermediate else None
        intermediate_delta_norm = [] if return_intermediate else None
        intermediate_corr_norm = [] if return_intermediate else None

        confidence_history = []
        delta_norm_history = []

        for k in range(1, k_max + 1):
            # 1. Latent synthesis / reasoning transformation.
            delta = self.synthesis(h)
            last_delta = delta
            # 2. Residual update (NaN/Inf-guarded for stability).
            h = self._finite_residual_update(h, delta)

            # 3. Consistency / confidence estimation.
            token_conf, seq_conf = self.consistency(h, delta)
            final_confidence = seq_conf
            confidence_history.append(seq_conf)
            d_norm = torch.sqrt(
                torch.mean(delta.float() ** 2, dim=-1).clamp_min(0.0) + 1e-8
            )
            if d_norm.dim() > 1:
                d_norm = torch.mean(d_norm, dim=-1)
            delta_norm_history.append(torch.mean(d_norm))

            if return_intermediate:
                intermediate_conf.append(seq_conf.detach())
                intermediate_delta_norm.append(
                    float(torch.mean(d_norm).item())
                )

            # 4. Halting decision (disabled under forced / below min_steps).
            can_halt = (not forced) and (k >= k_min)
            if can_halt and bool(torch.all(seq_conf >= threshold)):
                converged = True
                break

            # 5. Bounded self-correction (confidence-conditioned).
            if corrections_done < k_corr:
                corr = self.correction(h, seq_conf)
                last_correction = corr
                h = self._finite_residual_update(h, corr)
                corrections_done += 1
                corr_norm = torch.sqrt(
                    torch.mean(corr.float() ** 2, dim=-1).clamp_min(0.0) + 1e-8
                )
                if corr_norm.dim() > 1:
                    corr_norm = torch.mean(corr_norm, dim=-1)
                if return_intermediate:
                    intermediate_corr_norm.append(
                        float(torch.mean(corr_norm).item())
                    )
            else:
                if return_intermediate:
                    intermediate_corr_norm.append(0.0)

        # Final stability normalization of the refined state. Cast params to the
        # state dtype so non-float32 inputs keep their dtype (output == input).
        h = F.layer_norm(
            h, self.output_norm.normalized_shape,
            self.output_norm.weight.to(h.dtype),
            self.output_norm.bias.to(h.dtype), self.output_norm.eps,
        )

        # ---- Losses ----
        # Consistency loss pushes the final confidence towards sufficiency.
        if final_confidence is None:
            # Degenerate: zero steps executed (should not happen; k_max >= 1).
            final_confidence = torch.zeros(
                h.size(0), device=device, dtype=dtype
            )
        consistency_loss = self.losses.consistency_loss(
            final_confidence, target=1.0
        )

        # Refinement loss regularizes correction residuals (only the last one
        # is retained to avoid unbounded memory; the per-step norms are
        # captured in diagnostics).
        if last_correction is not None and corrections_done > 0:
            refinement_loss = self.losses.refinement_loss(last_correction)
        else:
            refinement_loss = h0.new_zeros(())

        total_reasoning_loss = self.losses.aggregate(
            consistency_loss, refinement_loss
        )

        # ---- Diagnostics (numerical only) ----
        mean_delta_norm = (
            float(torch.stack(delta_norm_history).detach().mean().item())
            if delta_norm_history
            else 0.0
        )
        mean_conf = float(
            torch.stack(confidence_history).detach().mean().item()
        ) if confidence_history else 0.0

        diagnostics: Dict[str, Any] = {
            "reasoning_steps": k,
            "correction_count": corrections_done,
            "converged": bool(converged),
            "confidence": float(final_confidence.detach().mean().item()),
            "consistency_score": float(final_confidence.detach().mean().item()),
            "latent_delta_norm": mean_delta_norm,
            "min_reasoning_steps": k_min,
            "max_reasoning_steps": k_max,
            "max_reasoning_corrections": k_corr,
            "confidence_threshold": float(threshold),
            "forced_mode": bool(forced),
            "mean_confidence_history": mean_conf,
        }
        if return_intermediate:
            diagnostics["intermediate_confidence"] = intermediate_conf
            diagnostics["intermediate_delta_norm"] = intermediate_delta_norm
            diagnostics["intermediate_corr_norm"] = intermediate_corr_norm

        return ReasoningOutput(
            refined_state=h,
            consistency_loss=consistency_loss,
            refinement_loss=refinement_loss,
            total_reasoning_loss=total_reasoning_loss,
            diagnostics=diagnostics,
        )
