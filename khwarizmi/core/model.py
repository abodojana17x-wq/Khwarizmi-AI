"""
Khwarizmi AI Core Model Module.

Implements the foundational neural architecture layer integrating:
    1. Subword & Positional Embeddings
    2. Khwarizmi State Cells (KSC) in Residual Blocks
    3. Sparse Mixture-of-Experts (MoE) Sublayers
    4. Short-Term Working State (M_short)
    5. Utility-Gated Long-Term Persistent Memory (M_long)
    6. Cognitive Router & Pathway Dispatcher
    7. Adaptive Recurrent Reasoning Cycles (ARRC)
    8. Output Pathway with Confidence Estimation & Selective Verification Trigger

All components operate modularly and differentiably on CPU (and future GPUs).
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List

from ..config.settings import KhwarizmiConfig
from .embeddings import KhwarizmiEmbeddings
from .ksc_block import KSCResidualBlock
from .output import OutputPathway
from ..memory.short_term import ShortTermWorkingState
from ..memory.gating import MemoryGatingController
from ..memory.long_term import LongTermPersistentMemory
from ..routing.router import CognitiveRouter
from ..routing.pathways import PathwayDispatcher
from ..experts.moe_layer import SparseMoELayer
from ..experts.specialists import create_standard_specialists
from ..reasoning.latent_reasoner import LatentReasoner


@dataclass
class KhwarizmiOutput:
    """
    Structured Output Data Contract from KhwarizmiModel forward pass.

    Attributes:
        logits: Vocabulary logits of shape (batch_size, seq_len, vocab_size).
        confidence: Output confidence scores C(y) of shape (batch_size,).
        needs_verification: Boolean tensor of shape (batch_size,) indicating tool trigger.
        selected_pathways: Integer tensor of shape (batch_size,) with router selections.
        routing_probs: Router policy probabilities of shape (batch_size, num_pathways).
        short_term_state: Updated ephemeral working state dictionary.
        long_term_table: Updated persistent memory table dictionary.
        losses: Dictionary of auxiliary regularizers (routing, MoE balance, ARRC ponder, memory gating).
        diagnostics: Operational execution metrics.
    """
    logits: torch.Tensor
    confidence: torch.Tensor
    needs_verification: torch.Tensor
    selected_pathways: torch.Tensor
    routing_probs: torch.Tensor
    short_term_state: Dict[str, torch.Tensor]
    long_term_table: Dict[str, torch.Tensor]
    losses: Dict[str, torch.Tensor]
    diagnostics: Dict[str, Any]


class KhwarizmiModel(nn.Module):
    """
    Khwarizmi AI Neural Architecture Foundation Model.
    Provides a modular, CPU-testable, GPU-ready neural core implementing all
    mathematical operators specified in the Phase 0 Architecture Blueprint.
    """

    def __init__(self, config: KhwarizmiConfig):
        super().__init__()
        self.config = config

        # 1. Input Representation
        self.embeddings = KhwarizmiEmbeddings(config)

        # 2. Dual Memory Architecture
        self.short_term_state_handler = ShortTermWorkingState(config)
        self.memory_gating = MemoryGatingController(config)
        self.long_term_memory = LongTermPersistentMemory(config)

        # 3. Cognitive Router
        self.cognitive_router = CognitiveRouter(config)

        # 4. Sparse Mixture-of-Experts Specialists (Phase 4).
        # Configurable via config.enable_moe: when disabled, every residual
        # block carries a dense FFN and no experts/router are built, preserving
        # the pre-Phase-4 dense behavior.
        if config.enable_moe:
            specialists = create_standard_specialists(config)
            self.shared_moe_layer = SparseMoELayer(config, experts=specialists)
        else:
            self.shared_moe_layer = None

        # 5. KSC Residual Blocks Sequence
        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            is_moe = config.enable_moe and ((i + 1) % config.moe_frequency == 0)
            block = KSCResidualBlock(config, is_moe_layer=is_moe)
            self.layers.append(block)

        # 6. Adaptive Recurrent Reasoning Engine
        self.reasoner = LatentReasoner(config)

        # 7. Output Pathway
        self.output_pathway = OutputPathway(config)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Initialize Short-Term Working State and Long-Term Persistent Memory tables
        for a new batch.

        Args:
            batch_size: Number of concurrent sequences.
            device: Torch device.
            dtype: Tensor dtype.

        Returns:
            Tuple of (short_term_state, long_term_table) dictionaries.
        """
        st_state = self.short_term_state_handler.init_state(
            batch_size, device=device, dtype=dtype
        )
        lt_table = self.long_term_memory.init_memory_table(
            batch_size, device=device, dtype=dtype
        )
        return st_state, lt_table

    def forward(
        self,
        input_ids: torch.Tensor,
        short_term_state: Optional[Dict[str, torch.Tensor]] = None,
        long_term_table: Optional[Dict[str, torch.Tensor]] = None,
        deterministic_router: bool = True,
        force_cycles: Optional[int] = None,
        step_counter: int = 1,
    ) -> KhwarizmiOutput:
        """
        Execute full end-to-end forward pass through the Khwarizmi Neural Core.

        Args:
            input_ids: Token indices tensor of shape (batch_size, seq_len).
            short_term_state: Optional active working state dictionary.
            long_term_table: Optional persistent memory table dictionary.
            deterministic_router: Whether Cognitive Router selects argmax pathway.
            force_cycles: Optional override for exact Adaptive Compute cycles.
            step_counter: Monotonic sequence step counter for timestamped eviction.

        Returns:
            KhwarizmiOutput structured dataclass.
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2-dimensional (batch_size, seq_len), got {input_ids.shape}"
            )

        batch_size, seq_len = input_ids.shape

        if short_term_state is None or long_term_table is None:
            init_st, init_lt = self.init_state(
                batch_size, device=input_ids.device, dtype=torch.float32
            )
            short_term_state = short_term_state or init_st
            long_term_table = long_term_table or init_lt

        # 1. Embeddings
        x = self.embeddings(input_ids)  # Shape: (B, L, D)

        # 2. Extract Working State Summary Vector
        summary_repr = self.short_term_state_handler.get_summary_vector(
            short_term_state
        )

        # 3. Cognitive Router Evaluation
        routing_probs, selected_pathways, routing_loss = self.cognitive_router(
            summary_repr, deterministic=deterministic_router
        )
        pathway_flags = PathwayDispatcher.dispatch(selected_pathways)

        # 4. Long-Term Memory Gating & Associative READ
        mem_gates = self.memory_gating(summary_repr)
        memory_gate_loss = self.memory_gating.compute_gating_regularization(
            mem_gates
        )
        memory_proj_loss = self.long_term_memory.compute_projection_regularization(
            summary_repr
        )

        mem_read_out, attn_weights = self.long_term_memory.read(
            summary_repr,
            long_term_table,
            g_read=mem_gates["read"],
            current_step=step_counter,
        )

        # Inject memory recall into sequence embeddings if read flag is active
        read_mask = pathway_flags.use_memory_read.to(dtype=x.dtype).view(-1, 1, 1)
        x = x + read_mask * mem_read_out.unsqueeze(1)

        # 5. KSC Sequence Layers (with optional MoE sublayers)
        curr_state = short_term_state["recurrent_state"]
        total_moe_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for layer_idx, layer in enumerate(self.layers):
            use_moe_in_layer = (
                layer.is_moe_layer and pathway_flags.use_moe.any().item()
            )
            x, curr_state, moe_loss, _ = layer(
                x,
                state=curr_state,
                moe_layer=self.shared_moe_layer if layer.is_moe_layer else None,
                use_moe=use_moe_in_layer,
            )
            if moe_loss is not None:
                total_moe_loss = total_moe_loss + moe_loss

        # 6. Adaptive Recurrent Reasoning Cycles (ARRC)
        reasoned_x, final_state, ponder_loss, reasoner_diag = self.reasoner.reason(
            x,
            state=curr_state,
            pathway_id=selected_pathways,
            force_cycles=force_cycles,
        )

        # 7. Update Short-Term Working State
        updated_short_term = self.short_term_state_handler.update(
            current_state=short_term_state,
            new_recurrent_state=final_state,
            new_token_features=reasoned_x,
        )

        # 8. Selective WRITE & FORGET on Long-Term Persistent Memory
        write_mask_active = pathway_flags.use_memory_write.any().item()
        if write_mask_active:
            mean_seq_repr = torch.mean(reasoned_x, dim=1)
            updated_long_term = self.long_term_memory.write(
                candidate_repr=mean_seq_repr,
                memory_table=long_term_table,
                g_write=mem_gates["write"],
                current_step=step_counter,
            )
        else:
            updated_long_term = long_term_table

        updated_long_term = self.long_term_memory.forget(
            memory_table=updated_long_term,
            g_forget=mem_gates["forget"],
        )

        # 9. Output Pathway
        logits, confidence, needs_verification = self.output_pathway(
            reasoned_x, pathway_id=selected_pathways
        )

        # 10. Compile Regularization Losses
        total_aux_loss = (
            routing_loss
            + total_moe_loss
            + ponder_loss
            + memory_gate_loss
            + memory_proj_loss
        )
        losses = {
            "routing_loss": routing_loss,
            "moe_aux_loss": total_moe_loss,
            "ponder_loss": ponder_loss,
            "memory_gate_loss": memory_gate_loss,
            "memory_proj_loss": memory_proj_loss,
            "total_aux_loss": total_aux_loss,
        }

        diagnostics = {
            "selected_pathway_names": [
                CognitiveRouter.get_pathway_name(idx.item())
                for idx in selected_pathways
            ],
            "mean_confidence": torch.mean(confidence).item(),
            "verification_trigger_count": int(torch.sum(needs_verification).item()),
            "reasoner_diagnostics": reasoner_diag,
            "memory_valid_slots_count": int(
                torch.sum(updated_long_term["valid_mask"]).item()
            ),
        }

        return KhwarizmiOutput(
            logits=logits,
            confidence=confidence,
            needs_verification=needs_verification,
            selected_pathways=selected_pathways,
            routing_probs=routing_probs,
            short_term_state=updated_short_term,
            long_term_table=updated_long_term,
            losses=losses,
            diagnostics=diagnostics,
        )

    def count_parameters(self) -> int:
        """Return total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_memory_footprint_mb(self) -> float:
        """Estimate CPU RAM footprint of model parameters in megabytes."""
        total_bytes = sum(
            p.numel() * p.element_size() for p in self.parameters()
        )
        return total_bytes / (1024.0 * 1024.0)
