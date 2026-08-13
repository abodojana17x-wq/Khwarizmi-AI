"""
Khwarizmi Dual Memory Facade Module (Phase 3).

Composes the three Phase 3 building blocks — :class:`ShortTermWorkingState`,
:class:`MemoryGatingController`, and :class:`LongTermPersistentMemory` — together
with the deterministic :class:`UtilityGatingPolicy` into a single, clean unit that
executes the full memory lifecycle:

    1. Compute READ / WRITE / UPDATE / FORGET gate probabilities.
    2. Predict a candidate utility score and associative similarity.
    3. Resolve a discrete utility-gating decision (RETAIN / WRITE / UPDATE / FORGET).
    4. Perform associative READ retrieval.
    5. Apply the selected persistent-memory operation (write / update / forget).
    6. Append the current representation to the bounded short-term window.

This facade is the integration point used by
:class:`khwarizmi.core.memory_prototype.KhwarizmiDualMemoryPrototype`; it keeps the
three lower-level modules independent and individually testable.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from ..config.settings import KhwarizmiConfig
from .short_term import ShortTermWorkingState
from .gating import MemoryGatingController, UtilityGatingPolicy
from .long_term import LongTermPersistentMemory


@dataclass
class DualMemoryOutput:
    """
    Structured output contract for :class:`DualMemory`.

    Attributes:
        read_vector: Associative retrieval output of shape ``(batch_size, d_model)``.
        state: Combined updated state dictionary holding ``short_term`` and
            ``long_term`` sub-states.
        decision: LongTensor of shape ``(batch_size,)`` with the discrete
            utility-gating decision per item (RETAIN / WRITE / UPDATE / FORGET).
        utilities: Predicted candidate utility scores of shape ``(batch_size,)``.
        max_similarity: Top-1 cosine similarity to the persistent table, shape
            ``(batch_size,)`` (``-1`` for an empty table).
        diagnostics: Operational metrics (gate values, action counts).
    """

    read_vector: torch.Tensor
    state: Dict[str, Any]
    decision: torch.Tensor
    utilities: torch.Tensor
    max_similarity: torch.Tensor
    diagnostics: Dict[str, Any]


class DualMemory(nn.Module):
    """
    Utility-gated Dual Memory system: Short-Term Working State + Persistent Memory.

    Bounded by construction:
        * Short-term window is capped at ``config.short_term_capacity``.
        * Persistent table is capped at ``config.memory_slots``.
    No Python list/dict grows with sequence length or operation count.
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.config = config
        self.short_term = ShortTermWorkingState(config)
        self.gating = MemoryGatingController(config)
        self.long_term = LongTermPersistentMemory(config)
        self.policy = UtilityGatingPolicy(config)

    # ------------------------------------------------------------------- state
    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Initialize a fresh combined short-term + long-term memory state."""
        return {
            "short_term": self.short_term.init_state(batch_size, device=device, dtype=dtype),
            "long_term": self.long_term.init_memory_table(
                batch_size, device=device, dtype=dtype
            ),
        }

    # --------------------------------------------------------- individual ops
    def read(
        self,
        query_repr: torch.Tensor,
        state: Dict[str, Any],
        current_step: int = 0,
    ) -> torch.Tensor:
        """READ: perform associative retrieval conditioned on the read gate."""
        gates = self.gating(query_repr)
        out, _ = self.long_term.read(
            query_repr,
            state["long_term"],
            g_read=gates["read"],
            current_step=current_step,
        )
        return out

    # The WRITE / UPDATE / FORGET operations are exposed with explicit,
    # testable gates on :class:`LongTermPersistentMemory` (``self.long_term``)
    # and are executed automatically — driven by the utility-gating decision —
    # inside :meth:`forward`. See the individual operation methods there.

    # ------------------------------------------------------------ lifecycle
    def forward(
        self,
        h_t: torch.Tensor,
        state: Optional[Dict[str, Any]] = None,
        s_task: Optional[torch.Tensor] = None,
        step_counter: int = 0,
    ) -> DualMemoryOutput:
        """
        Execute the full utility-gated memory lifecycle for a batch of latent
        representations ``h_t`` of shape ``(batch_size, d_model)``.

        Args:
            h_t: Current latent representation (the candidate to store).
            state: Optional combined memory state; initialized when ``None``.
            s_task: Optional task-context vector of shape ``(batch_size, d_model)``.
            step_counter: Monotonic step counter for timestamped eviction.

        Returns:
            :class:`DualMemoryOutput`.
        """
        batch_size = h_t.size(0)
        if state is None:
            state = self.init_state(
                batch_size, device=h_t.device, dtype=h_t.dtype
            )

        # 1. Gate probabilities
        gates = self.gating(h_t, s_task=s_task)

        # 2. Candidate utility + associative similarity
        utilities = torch.sigmoid(
            self.long_term.util_proj(h_t)
        ).squeeze(-1)  # (B,)
        similarities = self.long_term.cosine_similarity(
            h_t, state["long_term"]
        )  # (B, M)
        max_similarity = torch.max(similarities, dim=-1).values  # (B,)

        # 3. Deterministic utility-gating decision
        decision_out = self.policy.decide(gates, utilities, max_similarity)
        decision = decision_out["decision"]

        # 4. Associative READ
        read_vector, _ = self.long_term.read(
            h_t,
            state["long_term"],
            g_read=gates["read"],
            current_step=step_counter,
        )

        # 5. Apply selected persistent-memory operation
        if decision_out["write_mask"].any():
            self.long_term.write(
                h_t,
                state["long_term"],
                g_write=decision_out["write_mask"].to(dtype=h_t.dtype),
                current_step=step_counter,
                threshold=0.5,
            )
        if decision_out["update_mask"].any():
            self.long_term.update(
                h_t,
                state["long_term"],
                g_update=decision_out["update_mask"].to(dtype=h_t.dtype),
                current_step=step_counter,
                threshold=0.5,
            )
        if decision_out["forget_mask"].any():
            self.long_term.forget(
                state["long_term"],
                g_forget=decision_out["forget_mask"].to(dtype=h_t.dtype),
                threshold=0.5,
            )

        # 6. Append current representation to the bounded short-term window
        state["short_term"] = self.short_term.write(
            state["short_term"], h_t.unsqueeze(1)
        )

        diagnostics = {
            "num_long_term_slots": int(
                torch.sum(state["long_term"]["valid_mask"]).item()
            ),
            "num_short_term_items": int(
                state["short_term"]["window_buffer"].size(1)
            ),
            "gates": {k: v.detach().clone() for k, v in gates.items()},
        }

        return DualMemoryOutput(
            read_vector=read_vector,
            state=state,
            decision=decision,
            utilities=utilities.detach(),
            max_similarity=max_similarity,
            diagnostics=diagnostics,
        )
