"""
Khwarizmi Long-Term Persistent Memory Module.

Implements the non-parametric local key-value project store with learned READ, WRITE,
UPDATE, and FORGET operations and time-decayed utility eviction as defined in
Section 4.2 and Section 5.2 of the Khwarizmi AI Blueprint.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from ..config.settings import KhwarizmiConfig


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
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.config = config
        self.memory_dim = config.memory_dim
        self.max_slots = config.memory_slots
        self.d_model = config.d_model

        # Projection from latent representation to memory key/value/utility space
        self.key_proj = nn.Linear(self.d_model, self.memory_dim)
        self.val_proj = nn.Linear(self.d_model, self.memory_dim)
        self.util_proj = nn.Linear(self.d_model, 1)
        self.out_proj = nn.Linear(self.memory_dim, self.d_model)

        self.decay_lambda = 0.01

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
        q_mem = self.key_proj(query_repr).unsqueeze(1)  # (B, 1, D_m)
        keys = memory_table["keys"]                     # (B, M, D_m)
        values = memory_table["values"]                 # (B, M, D_m)
        valid_mask = memory_table["valid_mask"]         # (B, M)

        # Compute cosine similarity / scaled dot-product scores
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

        # Update access timestamp for top retrieved slot if read gate is active
        with torch.no_grad():
            for b in range(batch_size):
                if has_valid[b, 0] and g_read[b].item() > 0.5:
                    top_idx = torch.argmax(attn_weights[b]).item()
                    memory_table["access_times"][b, top_idx] = current_step

        return out_repr, attn_weights

    def write(
        self,
        candidate_repr: torch.Tensor,
        memory_table: Dict[str, torch.Tensor],
        g_write: torch.Tensor,
        current_step: int = 0,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Selective WRITE insertion and time-decayed utility eviction.

        Args:
            candidate_repr: Candidate latent representation of shape (batch_size, d_model).
            memory_table: Dictionary of persistent memory table tensors.
            g_write: Write gating probability tensor of shape (batch_size,).
            current_step: Current step counter.
            threshold: Minimum write gate probability to trigger insertion.

        Returns:
            Updated persistent memory table dictionary.
        """
        batch_size = candidate_repr.size(0)
        with torch.no_grad():
            cand_keys = self.key_proj(candidate_repr)
            cand_vals = self.val_proj(candidate_repr)
            cand_utils = torch.sigmoid(self.util_proj(candidate_repr)).squeeze(-1)

            keys = memory_table["keys"]
            values = memory_table["values"]
            utilities = memory_table["utilities"]
            access_times = memory_table["access_times"]
            valid_mask = memory_table["valid_mask"]

            for b in range(batch_size):
                if g_write[b].item() < threshold:
                    continue

                # Find empty slot if available
                empty_indices = (valid_mask[b] == False).nonzero(as_tuple=True)[0]
                if len(empty_indices) > 0:
                    target_idx = empty_indices[0].item()
                else:
                    # Table full -> Evict slot with minimum time-decayed utility
                    age = current_step - access_times[b].to(dtype=torch.float)
                    decayed_utility = utilities[b] * torch.exp(-self.decay_lambda * age)
                    target_idx = torch.argmin(decayed_utility).item()

                keys[b, target_idx] = cand_keys[b]
                values[b, target_idx] = cand_vals[b]
                utilities[b, target_idx] = cand_utils[b]
                access_times[b, target_idx] = current_step
                valid_mask[b, target_idx] = True

        return memory_table

    def forget(
        self,
        memory_table: Dict[str, torch.Tensor],
        g_forget: torch.Tensor,
        threshold: float = 0.7,
    ) -> Dict[str, torch.Tensor]:
        """
        FORGET operation: Evicts lowest utility memory slot when forget gate exceeds threshold.

        Args:
            memory_table: Dictionary of persistent memory table tensors.
            g_forget: Forget gating probability tensor of shape (batch_size,).
            threshold: Activation threshold for forget operation.

        Returns:
            Updated persistent memory table dictionary.
        """
        batch_size = g_forget.size(0)
        valid_mask = memory_table["valid_mask"]
        utilities = memory_table["utilities"]

        with torch.no_grad():
            for b in range(batch_size):
                if g_forget[b].item() >= threshold and valid_mask[b].any():
                    # Mask invalid utilities with high number
                    mask_val = torch.tensor(1e9, device=utilities.device, dtype=utilities.dtype)
                    valid_utils = torch.where(valid_mask[b], utilities[b], mask_val)
                    evict_idx = torch.argmin(valid_utils).item()
                    valid_mask[b, evict_idx] = False
                    utilities[b, evict_idx] = 0.0

        return memory_table
