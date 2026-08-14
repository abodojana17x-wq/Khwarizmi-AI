"""
Khwarizmi Sparse Mixture-of-Experts (MoE) Layer Module — Phase 4.

Implements the Sparse Top-K Noisy-Gated Mixture-of-Experts sublayer specified in
Sections 4.4 and 5.4 of the Khwarizmi AI Blueprint (ARCHITECTURE.md):

    H(z_t)_i = z_t W_g,i + eps_i * softplus(z_t W_noise,i),   eps_i ~ N(0, 1)
    G(z_t)    = Softmax(TopK(H(z_t), K))
    MoEOut(z_t) = sum_{i in TopK} G(z_t)_i * E_i(z_t)

    L_balance = alpha_moe * E * sum_i f_i * P_i
        f_i = fraction of tokens routed to expert i (dispatch fraction)
        P_i = mean gating probability allocated to expert i

Design notes (Phase 4):
    * Each expert is an independently parameterized Swish FFN with compatible
      input/output dimensions (d_model -> expert_d_ff -> d_model).
    * The forward pass is genuinely sparse: for every token only the Top-K
      selected experts are evaluated. Experts that receive no tokens perform
      zero computation. No dense masking over all experts is used.
    * Router noise is applied only during training (``self.training``); at
      inference time routing is deterministic. Top-K ties are broken
      deterministically by torch.topk for a given input tensor.
    * Gradients flow to the selected experts and to the routing weights of the
      selected experts through ``G(z_t)``; the load-balancing auxiliary loss is
      differentiable and provides a gradient signal to the gating parameters of
      *all* experts (via the full softmax P_i), which is what prevents routing
      collapse during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..config.settings import KhwarizmiConfig


class ExpertLayer(nn.Module):
    """
    Standard specialist Feed-Forward Expert Subnetwork.

    Independently parameterized two-layer MLP:
        x -> Linear(d_model, d_ff) -> SiLU -> Linear(d_ff, d_model)
    All experts share the same input/output dimensionality (d_model) so their
    outputs can be combined by the router; the intermediate dimension is
    configurable via ``config.expert_d_ff`` (defaults to ``config.d_ff``).

    Args:
        config: KhwarizmiConfig instance.
        specialization_name: Human-readable specialization label.
    """

    def __init__(self, config: KhwarizmiConfig, specialization_name: str = "General"):
        super().__init__()
        self.config = config
        self.specialization_name = specialization_name
        self.d_model = config.d_model
        self.d_ff = (
            config.expert_d_ff if config.expert_d_ff is not None else config.d_ff
        )
        self.w1 = nn.Linear(self.d_model, self.d_ff)
        self.w2 = nn.Linear(self.d_ff, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute the specialist FFN transformation on routed tokens."""
        return self.w2(F.silu(self.w1(x)))

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_ff={self.d_ff}, "
            f"specialization={self.specialization_name!r}"
        )


@dataclass
class MoERoutingDecision:
    """
    Structured routing result of a single Sparse MoE forward decision.

    Attributes:
        noisy_logits: Router logits H(z) of shape (N, num_experts) — noisy in
            training (when enabled), clean otherwise.
        router_probs: Full softmax gating probabilities P_i of shape
            (N, num_experts); used by the load-balancing loss.
        topk_indices: Selected expert indices of shape (N, top_k), dtype long.
        topk_weights: Normalized routing weights G(z_t) of shape (N, top_k);
            softmax over the Top-K logits so weights are positive and sum to 1.
        expert_fractions: f_i dispatch fractions of shape (num_experts,) —
            fraction of tokens routed to each expert (sums to top_k).
        expert_probs: P_i mean gating probabilities of shape (num_experts,).
        aux_loss: Load-balancing auxiliary loss scalar
            (alpha_moe * E * sum_i f_i * P_i), differentiable.
    """

    noisy_logits: torch.Tensor
    router_probs: torch.Tensor
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor
    expert_fractions: torch.Tensor
    expert_probs: torch.Tensor
    aux_loss: torch.Tensor


class SparseMoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts (MoE) Layer with Noisy Top-K Gating.

    E independently parameterized experts; every token is dispatched to exactly
    the Top-K experts selected by the router and combined with normalized
    routing weights. Only selected experts are executed (sparse execution).

    Args:
        config: KhwarizmiConfig instance providing ``num_experts``,
            ``top_k_experts``, ``load_balance_alpha``, ``moe_noise_enabled``,
            ``expert_d_ff``.
        experts: Optional pre-built list of experts (length must equal
            ``config.num_experts``). Defaults to freshly initialized
            ``ExpertLayer`` instances with independent parameters.
    """

    def __init__(
        self,
        config: KhwarizmiConfig,
        experts: Optional[Sequence[nn.Module]] = None,
    ):
        super().__init__()
        self.config = config
        self.d_model = int(config.d_model)
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.top_k_experts)
        self.alpha_moe = float(config.load_balance_alpha)
        self.noise_enabled = bool(config.moe_noise_enabled)

        # Architectural validation (mirrors KhwarizmiConfig.validate()).
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {self.num_experts}")
        if not (1 <= self.top_k <= self.num_experts):
            raise ValueError(
                f"top_k ({self.top_k}) must be in [1, num_experts ({self.num_experts})]"
            )
        if self.alpha_moe < 0.0:
            raise ValueError(
                f"load_balance_alpha must be non-negative, got {self.alpha_moe}"
            )

        # Gating projections. Construction order (gates first, then experts)
        # is part of the initialization contract and preserves RNG determinism.
        self.w_gate = nn.Linear(self.d_model, self.num_experts, bias=False)
        self.w_noise = nn.Linear(self.d_model, self.num_experts, bias=False)

        if experts is not None:
            if len(experts) != self.num_experts:
                raise ValueError(
                    f"Expected {self.num_experts} experts, got {len(experts)}"
                )
            self.experts = nn.ModuleList(experts)
            self._validate_expert_dimensions()
        else:
            self.experts = nn.ModuleList(
                [
                    ExpertLayer(config, f"Expert_{i}")
                    for i in range(self.num_experts)
                ]
            )

        self._reset_parameters()

        # Diagnostics: expert indices actually executed by the most recent
        # forward pass (in ascending order). Used by tests/benchmarks to verify
        # that unselected experts are never evaluated.
        self.last_routed_experts: List[int] = []

    def _reset_parameters(self) -> None:
        """Initialize gating projections (orthogonal/xavier init)."""
        nn.init.xavier_uniform_(self.w_gate.weight)
        nn.init.xavier_uniform_(self.w_noise.weight)

    def _validate_expert_dimensions(self) -> None:
        """Ensure every supplied expert has compatible d_model input/output."""
        for i, expert in enumerate(self.experts):
            if not isinstance(expert, nn.Module):
                raise TypeError(
                    f"experts[{i}] must be an nn.Module, got {type(expert).__name__}"
                )
            w1 = getattr(expert, "w1", None)
            w2 = getattr(expert, "w2", None)
            if w1 is None or w2 is None:
                raise TypeError(
                    f"experts[{i}] ({type(expert).__name__}) must expose "
                    f"'w1'/'w2' Linear projections (ExpertLayer-compatible)"
                )
            if w1.in_features != self.d_model or w2.out_features != self.d_model:
                raise ValueError(
                    f"experts[{i}] input/output dimension mismatch: expected "
                    f"d_model={self.d_model}, got in={w1.in_features}, "
                    f"out={w2.out_features}"
                )

    # ------------------------------------------------------------- parameters
    def count_expert_parameters(self) -> int:
        """Total parameter count of all expert subnetworks."""
        return sum(
            p.numel() for expert in self.experts for p in expert.parameters()
        )

    def count_router_parameters(self) -> int:
        """Total parameter count of the gating projections (gate + noise)."""
        return sum(
            p.numel()
            for proj in (self.w_gate, self.w_noise)
            for p in proj.parameters()
        )

    def count_active_parameters(self) -> int:
        """Parameters active per token: router + exactly ``top_k`` experts."""
        per_expert = self.count_expert_parameters() / self.num_experts
        return self.count_router_parameters() + int(self.top_k * per_expert)

    # ------------------------------------------------------------ validation
    def _validate_and_flatten(self, x: torch.Tensor) -> torch.Tensor:
        """
        Validate the input shape and flatten leading batch/sequence dims.

        Accepts (batch, seq, d_model), (batch, d_model) or any (..., d_model)
        tensor with at least two dimensions; raises ValueError otherwise.
        """
        if not torch.is_tensor(x):
            raise TypeError(f"input must be a torch.Tensor, got {type(x).__name__}")
        if x.dim() < 2:
            raise ValueError(
                f"MoE input must have shape (..., d_model) with at least 2 dims, "
                f"got shape {tuple(x.shape)}"
            )
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"MoE input feature dimension must be d_model ({self.d_model}), "
                f"got {x.size(-1)}"
            )
        if x.numel() == 0:
            raise ValueError("MoE input must contain at least one token")
        return x.reshape(-1, self.d_model)

    # -------------------------------------------------------------- routing
    def compute_noisy_logits(
        self, x: torch.Tensor, use_noise: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Compute gating logits H(z) = z W_g + eps * softplus(z W_noise).

        Args:
            x: Input tensor of shape (..., d_model).
            use_noise: If True, adds learned-magnitude Gaussian noise; if False,
                uses clean logits; if None (default), follows
                ``config.moe_noise_enabled``. Noise is applied only while the
                layer is in training mode — inference is always deterministic.

        Returns:
            Logit tensor of shape (..., num_experts).
        """
        clean_logits = self.w_gate(x)
        apply_noise = self.noise_enabled if use_noise is None else bool(use_noise)
        if apply_noise and self.training:
            noise_std = F.softplus(self.w_noise(x))
            noise = torch.randn_like(clean_logits)
            logits = clean_logits + noise * noise_std
        else:
            logits = clean_logits
        return logits

    def compute_load_balance_loss(
        self,
        expert_fractions: torch.Tensor,
        expert_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Load-balancing auxiliary loss (Section 5.4 of the Blueprint):

            L_balance = alpha_moe * E * sum_i f_i * P_i

        Minimized when the dispatch fractions f_i match the gating
        probabilities P_i across the batch, which encourages balanced expert
        utilization and prevents routing collapse.

        Args:
            expert_fractions: Dispatch fractions f_i of shape (num_experts,).
            expert_probs: Mean gating probabilities P_i of shape (num_experts,).

        Returns:
            Differentiable scalar auxiliary loss.
        """
        if expert_fractions.shape != (self.num_experts,):
            raise ValueError(
                f"expert_fractions must have shape ({self.num_experts},), "
                f"got {tuple(expert_fractions.shape)}"
            )
        if expert_probs.shape != (self.num_experts,):
            raise ValueError(
                f"expert_probs must have shape ({self.num_experts},), "
                f"got {tuple(expert_probs.shape)}"
            )
        return (
            self.alpha_moe
            * self.num_experts
            * torch.sum(expert_fractions * expert_probs)
        )

    def route(
        self, x: torch.Tensor, use_noise: Optional[bool] = None
    ) -> MoERoutingDecision:
        """
        Execute the routing decision only (no expert computation).

        Computes noisy gating logits, the full-softmax router probabilities,
        the deterministic Top-K selection with normalized routing weights, the
        per-expert dispatch fractions and the load-balancing auxiliary loss.

        Args:
            x: Input tensor of shape (..., d_model).
            use_noise: Overrides ``config.moe_noise_enabled``; noise only
                applies in training mode.

        Returns:
            MoERoutingDecision with all routing tensors and the auxiliary loss.
        """
        flat_x = self._validate_and_flatten(x)
        num_tokens = flat_x.size(0)

        logits = self.compute_noisy_logits(flat_x, use_noise=use_noise)  # (N, E)
        router_probs = F.softmax(logits, dim=-1)  # (N, E) full P_i

        topk_vals, topk_indices = torch.topk(logits, k=self.top_k, dim=-1)  # (N, K)
        topk_weights = F.softmax(topk_vals, dim=-1)  # (N, K) normalized G(z)

        # Dispatch fractions f_i: fraction of tokens routed to expert i.
        one_hot = F.one_hot(
            topk_indices, num_classes=self.num_experts
        ).to(dtype=logits.dtype)  # (N, K, E)
        expert_fractions = one_hot.sum(dim=(0, 1)) / num_tokens  # (E,)
        expert_probs = router_probs.mean(dim=0)  # (E,) mean P_i

        aux_loss = self.compute_load_balance_loss(expert_fractions, expert_probs)

        return MoERoutingDecision(
            noisy_logits=logits,
            router_probs=router_probs,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            expert_fractions=expert_fractions,
            expert_probs=expert_probs,
            aux_loss=aux_loss,
        )

    # -------------------------------------------------------------- dispatch
    def _dispatch_experts(
        self,
        flat_x: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Sparse expert execution: evaluate ONLY experts that received tokens.

        Tokens are gathered per active expert, evaluated in a single batched
        call, weighted by their routing weight and accumulated into the output.
        Experts that receive no tokens are never called.

        Args:
            flat_x: Flattened tokens of shape (N, d_model).
            topk_indices: Selected expert indices of shape (N, top_k).
            topk_weights: Routing weights of shape (N, top_k).

        Returns:
            Tuple of (combined output (N, d_model), executed expert indices).
        """
        output_flat = torch.zeros_like(flat_x)
        executed: List[int] = []

        active_experts = torch.unique(topk_indices.reshape(-1)).tolist()
        for expert_idx in active_experts:
            expert = self.experts[expert_idx]

            routed = (topk_indices == expert_idx).any(dim=-1)  # (N,)
            token_ids = routed.nonzero(as_tuple=False).reshape(-1)
            if token_ids.numel() == 0:
                continue

            expert_inputs = flat_x.index_select(0, token_ids)  # (n_i, D)
            expert_outputs = expert(expert_inputs)  # (n_i, D)

            # Routing weight of this expert for each routed token.
            matches = (
                topk_indices.index_select(0, token_ids) == expert_idx
            )  # (n_i, K)
            weights = topk_weights.index_select(0, token_ids)  # (n_i, K)
            expert_weights = (weights * matches.to(weights.dtype)).sum(
                dim=-1, keepdim=True
            )  # (n_i, 1)

            output_flat.index_add_(
                0, token_ids, expert_outputs * expert_weights
            )
            executed.append(expert_idx)

        return output_flat, executed

    # -------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        use_noise: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute the Sparse MoE forward pass with Top-K expert routing.

        For every token: compute router logits (noisy in training), select the
        Top-K experts, execute only those experts, combine their outputs with
        normalized routing weights, and return the load-balancing auxiliary
        loss for the batch.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) or
                (batch_size, d_model).
            use_noise: Overrides ``config.moe_noise_enabled``; noise is applied
                only while training.

        Returns:
            Tuple of:
                output: MoE output tensor of the same shape as x.
                aux_loss: Load-balancing auxiliary loss scalar tensor.
        """
        decision = self.route(x, use_noise=use_noise)
        flat_x = self._validate_and_flatten(x)

        output_flat, executed = self._dispatch_experts(
            flat_x, decision.topk_indices, decision.topk_weights
        )
        self.last_routed_experts = executed

        return output_flat.reshape(x.shape), decision.aux_loss
