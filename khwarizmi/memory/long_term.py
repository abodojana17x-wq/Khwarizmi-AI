"""
Khwarizmi Long-Term Persistent Memory Module.

Implements the non-parametric local key-value project store with learned READ,
WRITE, UPDATE, and FORGET operations and time-decayed utility eviction as defined
in Section 4.2 and Section 5.2 of the Khwarizmi AI Blueprint.

The store is a *fixed-capacity* table of slots
``M = {(k_i, v_i, u_i, t_i)}_{i=1}^{M}`` where ``k_i`` is the key, ``v_i`` the value,
``u_i`` the utility score and ``t_i`` the last-access step. All state is held in
pre-allocated tensors of shape ``(batch_size, max_slots, memory_dim)`` (plus
``(batch_size, max_slots)`` scalar/bool tables), so memory usage is strictly
bounded regardless of how many operations are performed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from ..config.settings import KhwarizmiConfig

_REQUIRED_TABLE_KEYS = ("keys", "values", "utilities", "access_times", "valid_mask")


class LongTermPersistentMemory(nn.Module):
    """
    Utility-Gated Non-Parametric Key-Value Long-Term Memory Store.

    Table Representation:
        Fixed-capacity table of slots M = {(k_i, v_i, u_i, t_i)} for i = 1...M,
        where k_i is key, v_i is value, u_i is utility score, and t_i is access step.

    Associative Retrieval (READ):
        Computes scaled dot-product attention over valid memory slots and scales
        output by read gate g_read.

    Selective Write & Utility Eviction (WRITE):
        Inserts candidate tuples when g_write > tau_write. If memory is full,
        evicts slot j* with minimum time-decayed utility u_j * exp(-lambda * delta_t).

    Refinement (UPDATE):
        Merges a candidate into the most-similar existing slot when the update
        gate is active and the top-1 cosine similarity exceeds a threshold,
        preventing duplicate entries while preserving consistency.

    Eviction (FORGET):
        Removes the lowest-utility valid slot when the forget gate is active, or
        a specific slot index when explicitly requested.
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.config = config
        self.memory_dim = config.memory_dim
        self.max_slots = config.memory_slots
        self.d_model = config.d_model
        self.decay_lambda = config.utility_decay_lambda
        self.update_similarity_threshold = config.update_similarity_threshold

        # Projection from latent representation to memory key/value/utility space
        self.key_proj = nn.Linear(self.d_model, self.memory_dim)
        self.val_proj = nn.Linear(self.d_model, self.memory_dim)
        self.util_proj = nn.Linear(self.d_model, 1)
        self.out_proj = nn.Linear(self.memory_dim, self.d_model)

    # ------------------------------------------------------------------- state
    def init_memory_table(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Initialize an empty memory table for each sequence in the batch.

        Args:
            batch_size: Number of concurrent sequences.
            device: Torch device.
            dtype: Tensor dtype.

        Returns:
            Dictionary containing keys, values, utilities, access_times, and valid_mask.
        """
        keys = torch.zeros(
            batch_size,
            self.max_slots,
            self.memory_dim,
            device=device,
            dtype=dtype,
        )
        values = torch.zeros(
            batch_size,
            self.max_slots,
            self.memory_dim,
            device=device,
            dtype=dtype,
        )
        utilities = torch.zeros(
            batch_size,
            self.max_slots,
            device=device,
            dtype=dtype,
        )
        access_times = torch.zeros(
            batch_size,
            self.max_slots,
            device=device,
            dtype=torch.long,
        )
        valid_mask = torch.zeros(
            batch_size,
            self.max_slots,
            device=device,
            dtype=torch.bool,
        )
        return {
            "keys": keys,
            "values": values,
            "utilities": utilities,
            "access_times": access_times,
            "valid_mask": valid_mask,
        }

    # -------------------------------------------------------------- validation
    def _validate_table(
        self, memory_table: Dict[str, torch.Tensor], batch_size: int
    ) -> None:
        """Validate that ``memory_table`` is well-formed for the given batch."""
        for key in _REQUIRED_TABLE_KEYS:
            if key not in memory_table:
                raise ValueError(f"memory_table is missing required key {key!r}")

        keys = memory_table["keys"]
        expected_shape = (batch_size, self.max_slots, self.memory_dim)
        if keys.shape != expected_shape:
            raise ValueError(
                f"memory_table['keys'] shape mismatch: expected {expected_shape}, "
                f"got {tuple(keys.shape)}"
            )
        if memory_table["values"].shape != expected_shape:
            raise ValueError(
                f"memory_table['values'] shape mismatch: expected {expected_shape}, "
                f"got {tuple(memory_table['values'].shape)}"
            )
        expected_scalar_shape = (batch_size, self.max_slots)
        for key in ("utilities", "access_times", "valid_mask"):
            if memory_table[key].shape != expected_scalar_shape:
                raise ValueError(
                    f"memory_table[{key!r}] shape mismatch: expected "
                    f"{expected_scalar_shape}, got {tuple(memory_table[key].shape)}"
                )

    # -------------------------------------------------------------- utilities
    def compute_projection_regularization(
        self, query_repr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute associative autoencoder regularization loss on memory projections
        to ensure stable gradient flow and preserve candidate representations.

        Args:
            query_repr: Latent representation tensor of shape (batch_size, d_model).

        Returns:
            Scalar regularization loss tensor.
        """
        q_mem = self.key_proj(query_repr)
        v_mem = self.val_proj(query_repr)
        u_mem = self.util_proj(query_repr)
        reconstructed = self.out_proj(v_mem)
        mse = F.mse_loss(reconstructed, query_repr)
        norm_reg = torch.mean(q_mem ** 2) + torch.mean(u_mem ** 2)
        return 0.01 * (mse + 0.01 * norm_reg)

    def num_stored(self, memory_table: Dict[str, torch.Tensor]) -> int:
        """Return the number of valid (occupied) slots across the batch."""
        self._validate_table(memory_table, memory_table["valid_mask"].size(0))
        return int(torch.sum(memory_table["valid_mask"]).item())

    def is_full(self, memory_table: Dict[str, torch.Tensor]) -> bool:
        """Return True if every slot in the batch is occupied."""
        return self.num_stored(memory_table) == (
            memory_table["valid_mask"].size(0) * self.max_slots
        )

    def cosine_similarity(
        self,
        query_repr: torch.Tensor,
        memory_table: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute L2-normalized cosine similarity between a candidate key and every
        stored key, masking invalid slots with ``-1``.

        Args:
            query_repr: Candidate latent representation of shape (batch_size, d_model).
            memory_table: Dictionary of persistent memory table tensors.

        Returns:
            Cosine similarity tensor of shape (batch_size, max_slots); invalid
            slots hold ``-1``.
        """
        batch_size = query_repr.size(0)
        self._validate_table(memory_table, batch_size)

        with torch.no_grad():
            q = self.key_proj(query_repr)  # (B, D_m)
            keys = memory_table["keys"]    # (B, M, D_m)
            valid = memory_table["valid_mask"]

            q_norm = F.normalize(q, p=2, dim=-1)          # (B, D_m)
            k_norm = F.normalize(keys, p=2, dim=-1)       # (B, M, D_m)
            cos = torch.bmm(k_norm, q_norm.unsqueeze(-1)).squeeze(-1)  # (B, M)
            # Guard against zero-norm slots (all-zero keys) producing 0/0.
            cos = torch.where(torch.isfinite(cos), cos, torch.zeros_like(cos))
            return torch.where(valid, cos, torch.full_like(cos, -1.0))

    def _time_decayed_utilities(
        self, memory_table: Dict[str, torch.Tensor], current_step: int
    ) -> torch.Tensor:
        """Compute u_j * exp(-lambda * (t - tau_j)) per slot (invalid slots -> +inf)."""
        utilities = memory_table["utilities"]
        access_times = memory_table["access_times"]
        valid = memory_table["valid_mask"]
        age = (current_step - access_times).to(dtype=utilities.dtype).clamp_min(0.0)
        decayed = utilities * torch.exp(-self.decay_lambda * age)
        inf = torch.full_like(decayed, float("inf"))
        return torch.where(valid, decayed, inf)

    # ---------------------------------------------------------------- READ
    def read(
        self,
        query_repr: torch.Tensor,
        memory_table: Dict[str, torch.Tensor],
        g_read: torch.Tensor,
        current_step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Associative READ retrieval from persistent memory table.

        Args:
            query_repr: Query latent representation of shape (batch_size, d_model).
            memory_table: Dictionary of persistent memory table tensors.
            g_read: Read gating probability tensor of shape (batch_size,).
            current_step: Current sequence/step integer counter for timestamp updates.

        Returns:
            Tuple of:
                retrieved_vector: Read output of shape (batch_size, d_model).
                attention_weights: Attention distribution over slots of shape (batch_size, max_slots).
        """
        batch_size = query_repr.size(0)
        self._validate_table(memory_table, batch_size)

        q_mem = self.key_proj(query_repr).unsqueeze(1)  # (B, 1, D_m)
        keys = memory_table["keys"]                     # (B, M, D_m)
        values = memory_table["values"]                 # (B, M, D_m)
        valid_mask = memory_table["valid_mask"]         # (B, M)

        # Compute scaled dot-product scores
        scores = torch.bmm(q_mem, keys.transpose(1, 2)).squeeze(1)  # (B, M)
        scores = scores / math.sqrt(self.memory_dim)

        # Mask invalid slots with -1e9
        mask_val = torch.tensor(-1e9, device=query_repr.device, dtype=query_repr.dtype)
        scores_masked = torch.where(valid_mask, scores, mask_val)

        # Check if any slot is valid in each sequence
        has_valid = valid_mask.any(dim=-1, keepdim=True)  # (B, 1)

        attn_weights = F.softmax(scores_masked, dim=-1)
        # Zero out attention for sequences where no slot is valid
        attn_weights = torch.where(has_valid, attn_weights, torch.zeros_like(attn_weights))

        # Retrieve weighted sum of values
        retrieved_val = torch.bmm(attn_weights.unsqueeze(1), values).squeeze(1)  # (B, D_m)
        out_repr = self.out_proj(retrieved_val) * g_read.unsqueeze(-1)
        # An empty table retrieves nothing: zero the output (suppress the
        # out_proj bias) so empty-memory reads are exact zeros.
        out_repr = out_repr * has_valid.to(dtype=out_repr.dtype)

        # Update access timestamp for top retrieved slot if read gate is active
        with torch.no_grad():
            for b in range(batch_size):
                if has_valid[b, 0] and g_read[b].item() > 0.5:
                    top_idx = torch.argmax(attn_weights[b]).item()
                    memory_table["access_times"][b, top_idx] = current_step

        return out_repr, attn_weights

    # ---------------------------------------------------------------- WRITE
    def write(
        self,
        candidate_repr: torch.Tensor,
        memory_table: Dict[str, torch.Tensor],
        g_write: torch.Tensor,
        current_step: int = 0,
        threshold: float = 0.5,
        similarity_threshold: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Selective WRITE insertion and time-decayed utility eviction.

        If ``similarity_threshold`` is provided, a candidate whose key is
        within that cosine-similarity of an existing slot is *merged* into the
        existing slot (via the UPDATE rule) instead of occupying a new slot,
        preventing near-duplicate entries.

        Args:
            candidate_repr: Candidate latent representation of shape (batch_size, d_model).
            memory_table: Dictionary of persistent memory table tensors.
            g_write: Write gating probability tensor of shape (batch_size,).
            current_step: Current step counter.
            threshold: Minimum write gate probability to trigger insertion.
            similarity_threshold: Optional cosine-similarity threshold triggering
                merge-into-existing behavior (disabled when ``None``).

        Returns:
            Updated persistent memory table dictionary.
        """
        batch_size = candidate_repr.size(0)
        self._validate_table(memory_table, batch_size)

        with torch.no_grad():
            cand_keys = self.key_proj(candidate_repr)
            cand_vals = self.val_proj(candidate_repr)
            cand_utils = torch.sigmoid(self.util_proj(candidate_repr)).squeeze(-1)

            keys = memory_table["keys"]
            values = memory_table["values"]
            utilities = memory_table["utilities"]
            access_times = memory_table["access_times"]
            valid_mask = memory_table["valid_mask"]

            # Precompute batch-wise cosine similarity for near-duplicate detection.
            sims = None
            if similarity_threshold is not None:
                sims = self.cosine_similarity(candidate_repr, memory_table)

            for b in range(batch_size):
                if g_write[b].item() < threshold:
                    continue

                # Near-duplicate detection: merge into the most similar slot.
                if similarity_threshold is not None and valid_mask[b].any():
                    best_slot = torch.argmax(sims[b]).item()
                    if sims[b, best_slot].item() >= similarity_threshold:
                        values[b, best_slot] = (
                            0.5 * values[b, best_slot] + 0.5 * cand_vals[b]
                        )
                        utilities[b, best_slot] = torch.max(
                            utilities[b, best_slot], cand_utils[b]
                        )
                        access_times[b, best_slot] = current_step
                        continue

                # Find empty slot if available
                empty_indices = (valid_mask[b] == False).nonzero(as_tuple=True)[0]
                if len(empty_indices) > 0:
                    target_idx = empty_indices[0].item()
                else:
                    # Table full -> Evict slot with minimum time-decayed utility
                    decayed = self._time_decayed_utilities(memory_table, current_step)
                    target_idx = torch.argmin(decayed[b]).item()

                keys[b, target_idx] = cand_keys[b]
                values[b, target_idx] = cand_vals[b]
                utilities[b, target_idx] = cand_utils[b]
                access_times[b, target_idx] = current_step
                valid_mask[b, target_idx] = True

        return memory_table

    # ---------------------------------------------------------------- UPDATE
    def update(
        self,
        candidate_repr: torch.Tensor,
        memory_table: Dict[str, torch.Tensor],
        g_update: torch.Tensor,
        current_step: int = 0,
        threshold: float = 0.5,
        similarity_threshold: Optional[float] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        UPDATE: merge a candidate into the most-similar existing slot while
        preserving memory consistency (no new slots, no duplication).

        A merge occurs for batch item ``b`` only when both conditions hold:
            1. ``g_update[b] >= threshold`` (the update gate is active);
            2. the top-1 cosine similarity ``max_i(q^T k_i) >= similarity_threshold``.

        On merge, the stored value is blended with the candidate (equal-weight
        EMA), the utility is raised to ``max(old, new)``, and the access
        timestamp is refreshed to ``current_step``. The slot count never changes.

        Args:
            candidate_repr: Candidate latent representation of shape (batch_size, d_model).
            memory_table: Dictionary of persistent memory table tensors.
            g_update: Update gating probability tensor of shape (batch_size,).
            current_step: Current step counter.
            threshold: Minimum update gate probability to trigger a merge.
            similarity_threshold: Cosine-similarity threshold (defaults to
                ``config.update_similarity_threshold``).

        Returns:
            Tuple of:
                memory_table: Updated persistent memory table dictionary.
                update_mask: Boolean tensor of shape (batch_size,) marking which
                    items were actually updated.
        """
        batch_size = candidate_repr.size(0)
        self._validate_table(memory_table, batch_size)
        if similarity_threshold is None:
            similarity_threshold = self.update_similarity_threshold

        values = memory_table["values"]
        utilities = memory_table["utilities"]
        access_times = memory_table["access_times"]
        valid_mask = memory_table["valid_mask"]

        update_mask = torch.zeros(batch_size, dtype=torch.bool, device=candidate_repr.device)

        with torch.no_grad():
            cand_vals = self.val_proj(candidate_repr)
            cand_utils = torch.sigmoid(self.util_proj(candidate_repr)).squeeze(-1)
            sims = self.cosine_similarity(candidate_repr, memory_table)  # (B, M)

            for b in range(batch_size):
                if g_update[b].item() < threshold:
                    continue
                if not valid_mask[b].any():
                    continue
                best_slot = torch.argmax(sims[b]).item()
                if sims[b, best_slot].item() >= similarity_threshold:
                    # Equal-weight EMA blend preserves slot count and consistency.
                    values[b, best_slot] = 0.5 * values[b, best_slot] + 0.5 * cand_vals[b]
                    utilities[b, best_slot] = torch.max(
                        utilities[b, best_slot], cand_utils[b]
                    )
                    access_times[b, best_slot] = current_step
                    update_mask[b] = True

        return memory_table, update_mask

    # ---------------------------------------------------------------- FORGET
    def forget(
        self,
        memory_table: Dict[str, torch.Tensor],
        g_forget: torch.Tensor,
        threshold: float = 0.7,
        slot_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        FORGET operation.

        Two eviction modes:
            * Gate-driven (default): when ``g_forget[b] >= threshold``, evict the
              lowest-utility valid slot for that batch item.
            * Explicit: when ``slot_index`` is provided (a ``(batch_size,)``
              LongTensor of slot ids), evict exactly those slots, regardless of
              the gate. Out-of-range or already-invalid indices raise ``ValueError``.

        Args:
            memory_table: Dictionary of persistent memory table tensors.
            g_forget: Forget gating probability tensor of shape (batch_size,).
            threshold: Activation threshold for the gate-driven forget operation.
            slot_index: Optional explicit slot id per batch item.

        Returns:
            Updated persistent memory table dictionary.
        """
        batch_size = g_forget.size(0)
        self._validate_table(memory_table, batch_size)
        valid_mask = memory_table["valid_mask"]
        utilities = memory_table["utilities"]

        with torch.no_grad():
            if slot_index is not None:
                if slot_index.shape != (batch_size,):
                    raise ValueError(
                        f"slot_index must have shape ({batch_size},), "
                        f"got {tuple(slot_index.shape)}"
                    )
                for b in range(batch_size):
                    idx = int(slot_index[b].item())
                    if idx < 0 or idx >= self.max_slots:
                        raise ValueError(
                            f"memory id {idx} out of range [0, {self.max_slots})"
                        )
                    if not valid_mask[b, idx].item():
                        raise ValueError(f"memory id {idx} is already invalid/empty")
                    valid_mask[b, idx] = False
                    utilities[b, idx] = 0.0
                return memory_table

            for b in range(batch_size):
                if g_forget[b].item() >= threshold and valid_mask[b].any():
                    # Mask invalid utilities with high number
                    mask_val = torch.tensor(1e9, device=utilities.device, dtype=utilities.dtype)
                    valid_utils = torch.where(valid_mask[b], utilities[b], mask_val)
                    evict_idx = torch.argmin(valid_utils).item()
                    valid_mask[b, evict_idx] = False
                    utilities[b, evict_idx] = 0.0

        return memory_table
