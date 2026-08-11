"""
Khwarizmi Sparse Mixture-of-Experts (MoE) Layer Module.

Implements Noisy Top-K sparse expert routing and load balancing auxiliary loss
as defined in Section 4.4 and Section 5.4 of the Khwarizmi AI Blueprint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from ..config.settings import KhwarizmiConfig


class ExpertLayer(nn.Module):
    """
    Standard specialist Feed-Forward Expert Subnetwork.
    Uses Swish (SiLU) activation function.
    """

    def __init__(self, config: KhwarizmiConfig, specialization_name: str = "General"):
        super().__init__()
        self.config = config
        self.specialization_name = specialization_name
        self.w1 = nn.Linear(config.d_model, config.d_ff)
        self.w2 = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute specialist FFN transformation."""
        return self.w2(F.silu(self.w1(x)))


class SparseMoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts (MoE) Layer with Noisy Top-K Gating.

    Architecture:
        E total experts with Top-K selective activation per token.

    Noisy Top-K Gating:
        H(z)_i = z * W_g,i + epsilon_i * softplus(z * W_noise,i)
        G(z) = Softmax(TopK(H(z), K))
        MoEOut(z) = sum_{i in TopK} G(z)_i * E_i(z)

    Load Balancing Auxiliary Loss:
        L_balance = alpha_moe * E * sum(f_i * P_i), where f_i is fraction of tokens
        routed to expert i and P_i is mean gating probability allocated to expert i.
    """

    def __init__(
        self,
        config: KhwarizmiConfig,
        experts: Optional[List[ExpertLayer]] = None,
    ):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.alpha_moe = config.load_balance_alpha

        # Gating projections
        self.w_gate = nn.Linear(self.d_model, self.num_experts, bias=False)
        self.w_noise = nn.Linear(self.d_model, self.num_experts, bias=False)

        if experts is not None:
            if len(experts) != self.num_experts:
                raise ValueError(
                    f"Expected {self.num_experts} experts, got {len(experts)}"
                )
            self.experts = nn.ModuleList(experts)
        else:
            self.experts = nn.ModuleList(
                [ExpertLayer(config, f"Expert_{i}") for i in range(self.num_experts)]
            )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize orthogonal gating parameters."""
        nn.init.xavier_uniform_(self.w_gate.weight)
        nn.init.xavier_uniform_(self.w_noise.weight)

    def compute_noisy_logits(
        self, x: torch.Tensor, use_noise: bool = True
    ) -> torch.Tensor:
        """
        Compute noisy gating logits H(z) for expert selection.

        Args:
            x: Input tensor of shape (..., d_model).
            use_noise: If True, adds normal noise during training/exploration.

        Returns:
            Logit tensor of shape (..., num_experts).
        """
        clean_logits = self.w_gate(x)
        if use_noise and self.training:
            noise_std = F.softplus(self.w_noise(x))
            noise = torch.randn_like(clean_logits)
            logits = clean_logits + noise * noise_std
        else:
            logits = clean_logits
        return logits

    def forward(
        self,
        x: torch.Tensor,
        use_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute Sparse MoE forward pass with Top-K expert routing.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) or (batch_size, d_model).
            use_noise: Whether to apply noise in gating logits.

        Returns:
            Tuple of:
                output: MoE output tensor of same shape as x.
                aux_loss: Load balancing auxiliary loss scalar tensor.
        """
        orig_shape = x.shape
        flat_x = x.reshape(-1, self.d_model)
        num_tokens = flat_x.size(0)

        logits = self.compute_noisy_logits(flat_x, use_noise=use_noise)  # (N, E)

        # Softmax over all experts to get P_i for balance loss
        full_probs = F.softmax(logits, dim=-1)  # (N, E)

        # Top-K selection
        topk_vals, topk_indices = torch.topk(logits, k=self.top_k, dim=-1)  # (N, K)
        topk_probs = F.softmax(topk_vals, dim=-1)  # (N, K)

        # Initialize output accumulator
        output_flat = torch.zeros_like(flat_x)

        # Fraction of tokens routed to each expert f_i
        # Count occurrences of each expert in topk_indices
        one_hot_indices = F.one_hot(topk_indices, num_classes=self.num_experts).to(
            dtype=torch.float
        )  # (N, K, E)
        routing_mask = torch.sum(one_hot_indices, dim=1)  # (N, E)
        f_i = torch.mean(routing_mask, dim=0)             # (E,)
        P_i = torch.mean(full_probs, dim=0)               # (E,)

        aux_loss = self.alpha_moe * self.num_experts * torch.sum(f_i * P_i)

        # Dispatch tokens to specialists
        for i, expert in enumerate(self.experts):
            # Find which tokens routed to expert i
            expert_mask = routing_mask[:, i] > 0
            if not expert_mask.any():
                continue

            # Compute expert output for all tokens (CPU-efficient vectorization for small batches)
            # or selectively for routed tokens
            idx_tokens = expert_mask.nonzero(as_tuple=True)[0]
            token_inputs = flat_x[idx_tokens]
            expert_outputs = expert(token_inputs)  # (N_i, D)

            # Find the topk weight for these tokens for expert i
            # topk_indices has shape (N, K)
            mask_matches = topk_indices[idx_tokens] == i  # (N_i, K)
            # Extract weights corresponding to expert i
            weights_i = torch.sum(
                topk_probs[idx_tokens] * mask_matches.to(dtype=topk_probs.dtype),
                dim=-1,
                keepdim=True,
            )  # (N_i, 1)

            output_flat[idx_tokens] = (
                output_flat[idx_tokens] + expert_outputs * weights_i
            )

        output = output_flat.reshape(orig_shape)
        return output, aux_loss
