"""
Phase 2 — Minimal Khwarizmi State Cell (KSC) Prototype.

This module implements a *clean, modular* language-model head built exclusively
from Phase 1 / Phase 2 components:

    1. Token + sinusoidal positional embeddings (:class:`KhwarizmiEmbeddings`).
    2. A stack of KSC residual blocks (:class:`KSCResidualBlock`).
    3. A final LayerNorm + vocabulary projection (LM) head.

It deliberately excludes Sparse-MoE, Dual Memory, the Cognitive Router and
Adaptive Compute — those belong to later phases (Phase 3+ per the roadmap).
The recurrent KSC state is ``O(1)`` in sequence length, which gives the
prototype a sub-quadratic inference memory footprint (the Phase 2 success
criterion of "no memory growth linear in sequence length during inference").

The prototype supports two execution modes:

* ``forward`` — batched, vectorized prefill over a full token sequence
  (used for training and evaluation).
* ``step`` — single-token autoregressive decode, maintaining a constant-size
  per-layer recurrent state. The decoded state size is independent of the
  sequence length, demonstrating ``O(1)`` per-token decoding memory.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..config.settings import KhwarizmiConfig
from ..config.tiers import get_prototype_50m_config, get_prototype_150m_config
from .embeddings import KhwarizmiEmbeddings
from .ksc_block import KSCResidualBlock


@dataclass
class KSCPrototypeOutput:
    """
    Structured output contract for :class:`KhwarizmiKSCPrototype`.

    Attributes:
        logits: Vocabulary logits of shape ``(batch_size, seq_len, vocab_size)``.
        states: Per-layer recurrent KSC states, each of shape
            ``(batch_size, n_heads, d_k, d_n)``. The list has ``n_layers``
            entries and is ``O(1)`` in sequence length.
        retention_history: Optional stacked retention gates of shape
            ``(n_layers, batch_size, seq_len, n_heads, d_k)`` when requested,
            otherwise ``None``.
    """

    logits: torch.Tensor
    states: List[torch.Tensor]
    retention_history: Optional[torch.Tensor] = None


class KhwarizmiKSCPrototype(nn.Module):
    """
    Minimal KSC-only language-model prototype (Phase 2).

    A straightforward, fully-differentiable stack of KSC residual blocks with a
    language-modeling head — the "minimal KSC prototype" called for by the
    Phase 2 roadmap. It is trainable end-to-end on next-token prediction and
    exposes a constant-memory autoregressive ``step`` interface.
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        if config is None:
            raise ValueError("config must be a KhwarizmiConfig instance, got None")
        self.config = config

        self.embeddings = KhwarizmiEmbeddings(config)

        # Stack of pure KSC residual blocks (no MoE sub-layer for Phase 2).
        self.layers = nn.ModuleList(
            [KSCResidualBlock(config, is_moe_layer=False) for _ in range(config.n_layers)]
        )

        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize the LM head for stable early training."""
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ states
    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[torch.Tensor]:
        """
        Initialize the per-layer recurrent KSC states for a fresh batch.

        Returns:
            A list of ``n_layers`` zero tensors, each of shape
            ``(batch_size, n_heads, d_k, d_n)``.
        """
        return [
            layer.ksc.init_state(batch_size, device=device, dtype=dtype)
            for layer in self.layers
        ]

    # -------------------------------------------------------------- embeddings
    def _embed_step(self, token_id: torch.Tensor, position: int) -> torch.Tensor:
        """
        Embed a single token at an absolute sequence ``position``.

        Mirrors :meth:`KhwarizmiEmbeddings.forward` exactly for a one-token
        sequence (token embedding + sinusoidal PE at ``position`` + LayerNorm;
        dropout is identity in ``eval`` mode). This lets ``step`` reproduce the
        vectorized ``forward`` output token-by-token.
        """
        x = self.embeddings.token_embedding(token_id).unsqueeze(1)  # (B, 1, D)
        pe = self.embeddings.positional_encoding.pe[:, position : position + 1, :].to(
            device=x.device, dtype=x.dtype
        )
        x = x + pe  # (B, 1, D)
        x = self.embeddings.layer_norm(x)
        return x

    # ------------------------------------------------------------ main forward
    def forward(
        self,
        input_ids: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None,
        return_retention: bool = False,
        memory_conditioning: Optional[torch.Tensor] = None,
    ) -> KSCPrototypeOutput:
        """
        Batched, vectorized prefill over a full token sequence.

        Args:
            input_ids: LongTensor of shape ``(batch_size, seq_len)``.
            state: Optional list of per-layer recurrent states. If ``None``,
                each layer is initialized to a zero state.
            return_retention: If ``True``, also returns the stacked KSC
                retention gates for numerical-stability inspection.
            memory_conditioning: Optional Dual Memory recall vector of shape
                ``(batch_size, d_model)`` added residually to the embedded input
                (Phase 3 integration point). ``None`` (default) preserves the
                exact Phase 2 behavior.

        Returns:
            :class:`KSCPrototypeOutput` with ``logits`` and updated ``states``.
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2-dimensional (batch_size, seq_len), got {input_ids.shape}"
            )

        batch_size, seq_len = input_ids.shape
        if state is None:
            state = self.init_state(
                batch_size, device=input_ids.device, dtype=input_ids.dtype
            )
        else:
            if len(state) != len(self.layers):
                raise ValueError(
                    f"state must contain one tensor per layer ({len(self.layers)}), "
                    f"got {len(state)}"
                )
            for i, s in enumerate(state):
                expected = (batch_size, self.config.n_heads, self.config.d_k, self.config.d_expansion)
                if s.shape != expected:
                    raise ValueError(
                        f"state[{i}] shape mismatch: expected {expected}, got {s.shape}"
                    )

        x = self.embeddings(input_ids)  # (B, L, D)

        if memory_conditioning is not None:
            if memory_conditioning.shape != (batch_size, self.config.d_model):
                raise ValueError(
                    f"memory_conditioning must have shape "
                    f"(batch_size, d_model)={(batch_size, self.config.d_model)}, "
                    f"got {tuple(memory_conditioning.shape)}"
                )
            x = x + memory_conditioning.unsqueeze(1)

        new_states: List[torch.Tensor] = []
        retention_history: Optional[List[torch.Tensor]] = [] if return_retention else None

        for i, layer in enumerate(self.layers):
            x, s, _, ret = layer(x, state=state[i], return_retention=return_retention)
            new_states.append(s)
            if return_retention and ret is not None:
                retention_history.append(ret)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, L, V)

        ret_out = (
            torch.stack(retention_history, dim=0) if return_retention else None
        )

        return KSCPrototypeOutput(logits=logits, states=new_states, retention_history=ret_out)

    # -------------------------------------------------------- autoregressive
    def step(
        self,
        token_id: torch.Tensor,
        state: List[torch.Tensor],
        position: int = 0,
        memory_conditioning: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Decode a single token autoregressively.

        Args:
            token_id: LongTensor of shape ``(batch_size,)`` with the next token.
            state: List of per-layer recurrent states (from ``init_state`` or a
                previous ``step`` call).
            position: Absolute sequence position of ``token_id`` (used to pick
                the correct sinusoidal positional encoding).
            memory_conditioning: Optional Dual Memory recall vector of shape
                ``(batch_size, d_model)`` added residually to the embedded input
                (Phase 3 integration point). ``None`` (default) preserves the
                exact Phase 2 behavior.

        Returns:
            Tuple of:
                logits: Next-token logits of shape ``(batch_size, vocab_size)``.
                new_state: Updated per-layer recurrent states.

        Note:
            The returned recurrent state is ``O(1)`` in sequence length — its
            size depends only on ``(batch_size, n_heads, d_k, d_n)`` and is
            identical whether ``position`` is 1 or 16384.
        """
        if token_id.dim() != 1:
            raise ValueError(
                f"token_id must be 1-dimensional (batch_size,), got {token_id.shape}"
            )
        if len(state) != len(self.layers):
            raise ValueError(
                f"state must contain one tensor per layer ({len(self.layers)}), got {len(state)}"
            )

        x = self._embed_step(token_id, position)  # (B, 1, D)

        if memory_conditioning is not None:
            if memory_conditioning.shape != (token_id.size(0), self.config.d_model):
                raise ValueError(
                    f"memory_conditioning must have shape "
                    f"(batch_size, d_model)={(token_id.size(0), self.config.d_model)}, "
                    f"got {tuple(memory_conditioning.shape)}"
                )
            x = x + memory_conditioning.unsqueeze(1)

        new_states: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            x, s, _, _ = layer(x, state=state[i])
            new_states.append(s)

        x = self.final_norm(x)
        logits = self.lm_head(x).squeeze(1)  # (B, V)
        return logits, new_states

    # ------------------------------------------------------------- analytics
    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def memory_footprint_bytes(self) -> int:
        """Return the CPU RAM footprint of the model parameters in bytes."""
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def recurrent_state_bytes(self, batch_size: int = 1) -> int:
        """
        Return the bytes consumed by the *recurrent* decode state for one step.

        This is ``O(1)`` in sequence length by construction: it depends only on
        ``batch_size * n_layers * n_heads * d_k * d_n``. Comparing this value at
        ``seq_len=1`` vs ``seq_len=16384`` is the Phase 2 memory regression test.
        """
        total = 0
        for s in self.init_state(batch_size):
            total += s.numel() * s.element_size()
        return total


# --------------------------------------------------------------- factories
def build_ksc_prototype(config_name: str = "50m") -> "KhwarizmiKSCPrototype":
    """
    Build a Phase 2 KSC prototype model from a named tier configuration.

    Args:
        config_name: One of ``"50m"`` or ``"150m"``.

    Returns:
        An initialized :class:`KhwarizmiKSCPrototype`.
    """
    if config_name == "50m":
        config = get_prototype_50m_config()
    elif config_name == "150m":
        config = get_prototype_150m_config()
    else:
        raise ValueError(
            f"Unknown prototype config name {config_name!r}; expected '50m' or '150m'"
        )
    return KhwarizmiKSCPrototype(config)
