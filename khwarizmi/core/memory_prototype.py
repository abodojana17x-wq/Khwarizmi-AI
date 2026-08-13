"""
Phase 3 — Dual Memory + KSC Prototype Integration.

This module composes the Phase 2 :class:`KhwarizmiKSCPrototype` with the Phase 3
:class:`khwarizmi.memory.DualMemory` system via *composition* (no Phase 1 / Phase 2
logic is rewritten). It demonstrates the smallest clean integration point called
for by the roadmap:

* The Dual Memory system performs an associative ``READ`` using the short-term
  summary as query.
* The recalled memory vector conditions the KSC sequence pass (a residual
  addition to the embedded input — the optional ``memory_conditioning`` argument
  added to :class:`KhwarizmiKSCPrototype`).
* After the KSC pass, a compressed candidate (mean input embedding) is passed
  through the utility-gated ``WRITE`` / ``UPDATE`` / ``FORGET`` lifecycle, and the
  short-term recurrent state is refreshed.

The resulting memory is strictly bounded: the short-term window is capped at
``config.short_term_capacity`` and the persistent table at ``config.memory_slots``.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..config.settings import KhwarizmiConfig
from ..memory.dual_memory import DualMemory
from .prototype import KhwarizmiKSCPrototype


@dataclass
class KhwarizmiDualMemoryOutput:
    """
    Structured output contract for :class:`KhwarizmiDualMemoryPrototype`.

    Attributes:
        logits: Vocabulary logits of shape ``(batch_size, seq_len, vocab_size)``.
        ksc_state: Per-layer KSC recurrent states (list of ``n_layers`` tensors).
        memory_state: Combined updated short-term + long-term memory state dict.
        decision: Discrete utility-gating decision per batch item (LongTensor).
        diagnostics: Operational memory metrics.
    """

    logits: torch.Tensor
    ksc_state: List[torch.Tensor]
    memory_state: Dict[str, Any]
    decision: torch.Tensor
    diagnostics: Dict[str, Any]


class KhwarizmiDualMemoryPrototype(nn.Module):
    """
    Phase 3 prototype: KSC sequence modeling + utility-gated Dual Memory.

    Composes (does not rewrite):
        * ``KhwarizmiKSCPrototype`` — Phase 2 KSC-only language model.
        * ``DualMemory`` — Phase 3 short-term + persistent memory.
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        if config is None:
            raise ValueError("config must be a KhwarizmiConfig instance, got None")
        self.config = config
        self.ksc = KhwarizmiKSCPrototype(config)
        self.memory = DualMemory(config)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Initialize fresh KSC recurrent states and a combined memory state."""
        ksc_state = self.ksc.init_state(batch_size, device=device, dtype=dtype)
        memory_state = self.memory.init_state(batch_size, device=device, dtype=dtype)
        return ksc_state, memory_state

    def forward(
        self,
        input_ids: torch.Tensor,
        ksc_state: Optional[List[torch.Tensor]] = None,
        memory_state: Optional[Dict[str, Any]] = None,
        step_counter: int = 0,
    ) -> KhwarizmiDualMemoryOutput:
        """
        Run one memory-augmented KSC forward pass over ``input_ids``.

        Args:
            input_ids: LongTensor of shape ``(batch_size, seq_len)``.
            ksc_state: Optional per-layer KSC recurrent states.
            memory_state: Optional combined memory state.
            step_counter: Monotonic step counter for timestamped eviction.

        Returns:
            :class:`KhwarizmiDualMemoryOutput`.
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2-dimensional (batch_size, seq_len), got {input_ids.shape}"
            )

        batch_size, _ = input_ids.shape
        dtype = torch.float32
        if ksc_state is None:
            ksc_state = self.ksc.init_state(
                batch_size, device=input_ids.device, dtype=dtype
            )
        if memory_state is None:
            memory_state = self.memory.init_state(
                batch_size, device=input_ids.device, dtype=dtype
            )

        # 1. Recall prior context (READ) using the short-term summary as query.
        query = self.memory.short_term.get_summary_vector(
            memory_state["short_term"]
        )
        read_vector = self.memory.read(
            query, memory_state, current_step=step_counter
        )

        # 2. KSC sequence pass conditioned on the recalled memory vector.
        ksc_out = self.ksc(
            input_ids, state=ksc_state, memory_conditioning=read_vector
        )
        ksc_state = ksc_out.states

        # 3. Compressed candidate representation to remember.
        embedded = self.ksc.embeddings(input_ids)
        candidate = torch.mean(embedded, dim=1)  # (B, D)

        # 4. Full utility-gated memory lifecycle on the candidate.
        mem_out = self.memory(
            candidate, memory_state, step_counter=step_counter
        )

        # 5. Refresh the short-term recurrent state with the final KSC state.
        mem_out.state["short_term"]["recurrent_state"] = ksc_state[-1]

        return KhwarizmiDualMemoryOutput(
            logits=ksc_out.logits,
            ksc_state=ksc_state,
            memory_state=mem_out.state,
            decision=mem_out.decision,
            diagnostics=mem_out.diagnostics,
        )

    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
