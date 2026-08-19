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
from ..reasoning.neural_reasoning_core import NeuralReasoningCore


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

        # 6. Adaptive Recurrent Reasoning Engine (Phase 5).
        # Configurable via config.enable_adaptive_compute: when disabled, no
        # halting gates or reasoning cell are built, the model performs a single
        # fixed pass, and the ponder loss is exactly zero — preserving the
        # fixed-compute pre-Phase-5 execution path.
        if config.enable_adaptive_compute:
            self.reasoner = LatentReasoner(config)
        else:
            self.reasoner = None

        # 7. Neural Reasoning Core (Phase 6): Latent Synthesis & Bounded
        # Self-Correction. Configurable via config.enable_reasoning_core: when
        # disabled, no synthesis/correction/confidence submodules are built,
        # the ARRC-refined representation passes through unchanged, and the
        # reasoning auxiliary losses are exactly zero (pre-Phase-6 path).
        if config.enable_reasoning_core:
            self.reasoning_core = NeuralReasoningCore(config)
        else:
            self.reasoning_core = None

        # 8. Output Pathway
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

        # Phase 7 full-core boundary captures (detached scalars only — these are
        # pure diagnostics and must never carry gradient through the loss path).
        _p7_read_active = bool(pathway_flags.use_memory_read.any().item())
        _p7_mem_read_norm = float(
            torch.mean(torch.norm(mem_read_out.detach(), dim=-1)).item()
        )
        _p7_pre_ksc_norm = float(
            torch.mean(torch.norm(x.detach(), dim=-1)).item()
        )

        # 5. KSC Sequence Layers (with optional MoE sublayers)
        curr_state = short_term_state["recurrent_state"]
        _p7_pre_ksc_state_norm = float(torch.norm(curr_state.detach()).item())
        total_moe_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        _p7_moe_layer_executed = 0
        _p7_ksc_layers = len(self.layers)

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
            if use_moe_in_layer:
                _p7_moe_layer_executed += 1

        _p7_post_ksc_norm = float(
            torch.mean(torch.norm(x.detach(), dim=-1)).item()
        )
        _p7_post_ksc_state_norm = float(
            torch.norm(curr_state.detach()).item()
        )

        # 6. Adaptive Recurrent Reasoning Cycles (ARRC)
        if self.reasoner is not None:
            reasoned_x, final_state, ponder_loss, reasoner_diag = self.reasoner.reason(
                x,
                state=curr_state,
                pathway_id=selected_pathways,
                force_cycles=force_cycles,
            )
        else:
            # Adaptive Compute disabled: single fixed pass, zero ponder cost.
            reasoned_x = x
            final_state = curr_state
            ponder_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            reasoner_diag = {
                "adaptive_compute_enabled": False,
                "mean_cycles": 0.0,
                "mean_remainder": 0.0,
            }

        # 7. Neural Reasoning Core (Phase 6): bounded latent refinement &
        # self-correction operating on the ARRC-refined representation. This is
        # a genuine trainable latent mechanism (no textual chain-of-thought);
        # it consumes the Phase 5 compute budget output and preserves dims.
        _p7_post_arrc_norm = float(
            torch.mean(torch.norm(reasoned_x.detach(), dim=-1)).item()
        )
        if self.reasoning_core is not None:
            reasoning_out = self.reasoning_core(reasoned_x)
            _p7_pre_reasoning_norm = float(
                torch.mean(torch.norm(reasoned_x.detach(), dim=-1)).item()
            )
            reasoned_x = reasoning_out.refined_state
            reasoning_loss = reasoning_out.total_reasoning_loss
            reasoning_diag = reasoning_out.diagnostics
        else:
            _p7_pre_reasoning_norm = _p7_post_arrc_norm
            reasoning_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            reasoning_diag = {
                "reasoning_core_enabled": False,
                "reasoning_steps": 0,
                "correction_count": 0,
                "converged": False,
                "confidence": 0.0,
                "consistency_score": 0.0,
                "latent_delta_norm": 0.0,
            }
        _p7_post_reasoning_norm = float(
            torch.mean(torch.norm(reasoned_x.detach(), dim=-1)).item()
        )

        # 8. Update Short-Term Working State
        updated_short_term = self.short_term_state_handler.update(
            current_state=short_term_state,
            new_recurrent_state=final_state,
            new_token_features=reasoned_x,
        )

        # 9. Selective WRITE & FORGET on Long-Term Persistent Memory
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
        _p7_memory_stored_before = int(
            torch.sum(long_term_table["valid_mask"]).item()
        )
        _p7_memory_stored_after = int(
            torch.sum(updated_long_term["valid_mask"]).item()
        )

        # 10. Output Pathway
        logits, confidence, needs_verification = self.output_pathway(
            reasoned_x, pathway_id=selected_pathways
        )

        # 11. Compile Regularization Losses
        total_aux_loss = (
            routing_loss
            + total_moe_loss
            + ponder_loss
            + reasoning_loss
            + memory_gate_loss
            + memory_proj_loss
        )
        losses = {
            "routing_loss": routing_loss,
            "moe_aux_loss": total_moe_loss,
            "ponder_loss": ponder_loss,
            "reasoning_loss": reasoning_loss,
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
            "reasoning_core_diagnostics": reasoning_diag,
            "memory_valid_slots_count": int(
                torch.sum(updated_long_term["valid_mask"]).item()
            ),
        }

        # Phase 7: structured full-core diagnostics namespace (additive only).
        # Exposes the contribution/status of each subsystem in one inspectable,
        # numerical structure. All values are detached scalars/ints/bools so
        # they can never carry gradient or mutate the training path. Emitted
        # only when config.enable_full_neural_core is True; when False the
        # diagnostics dict is identical to the pre-Phase-7 contract.
        if self.config.enable_full_neural_core:
            diagnostics["full_core"] = self._build_full_core_diagnostics(
                batch_size=batch_size,
                seq_len=seq_len,
                selected_pathways=selected_pathways,
                pathway_flags=pathway_flags,
                read_active=_p7_read_active,
                mem_read_norm=_p7_mem_read_norm,
                pre_ksc_norm=_p7_pre_ksc_norm,
                pre_ksc_state_norm=_p7_pre_ksc_state_norm,
                post_ksc_norm=_p7_post_ksc_norm,
                post_ksc_state_norm=_p7_post_ksc_state_norm,
                ksc_layers=_p7_ksc_layers,
                moe_layers_executed=_p7_moe_layer_executed,
                post_arrc_norm=_p7_post_arrc_norm,
                pre_reasoning_norm=_p7_pre_reasoning_norm,
                post_reasoning_norm=_p7_post_reasoning_norm,
                reasoner_diag=reasoner_diag,
                reasoning_diag=reasoning_diag,
                memory_stored_before=_p7_memory_stored_before,
                memory_stored_after=_p7_memory_stored_after,
                moe_loss=total_moe_loss,
                ponder_loss=ponder_loss,
                reasoning_loss=reasoning_loss,
                logits=logits,
                confidence=confidence,
            )

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

    def _build_full_core_diagnostics(
        self,
        batch_size: int,
        seq_len: int,
        selected_pathways: torch.Tensor,
        pathway_flags: Any,
        read_active: bool,
        mem_read_norm: float,
        pre_ksc_norm: float,
        pre_ksc_state_norm: float,
        post_ksc_norm: float,
        post_ksc_state_norm: float,
        ksc_layers: int,
        moe_layers_executed: int,
        post_arrc_norm: float,
        pre_reasoning_norm: float,
        post_reasoning_norm: float,
        reasoner_diag: Dict[str, Any],
        reasoning_diag: Dict[str, Any],
        memory_stored_before: int,
        memory_stored_after: int,
        moe_loss: torch.Tensor,
        ponder_loss: torch.Tensor,
        reasoning_loss: torch.Tensor,
        logits: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Assemble the Phase 7 structured full-core diagnostics namespace.

        This is a pure read-out of the unified forward path: it exposes, in one
        inspectable structure, the contribution/status of every subsystem
        (KSC, MoE, Memory, ARRC, Neural Reasoning, Full Core). All values are
        detached scalars/ints/bools — diagnostics never carry gradient, never
        store textual chain-of-thought, and never mutate caller tensors.

        Args:
            batch_size, seq_len: Input shape (for boundary tensor contracts).
            selected_pathways: Router pathway selections per batch element.
            pathway_flags: PathwayExecutionFlags from the dispatcher.
            read_active: Whether the memory READ path was eligible this step.
            mem_read_norm: Mean L2 norm of the memory read vector (detached).
            pre_ksc_norm, post_ksc_norm: Embedding norm before/after KSC stack.
            pre_ksc_state_norm, post_ksc_state_norm: KSC recurrent state norm.
            ksc_layers: Number of KSC residual blocks in the stack.
            moe_layers_executed: How many MoE sublayers actually ran.
            post_arrc_norm: Norm of the ARRC-refined representation.
            pre_reasoning_norm, post_reasoning_norm: Norm around reasoning core.
            reasoner_diag: ARRC diagnostics dict.
            reasoning_diag: Neural reasoning core diagnostics dict.
            memory_stored_before/after: Valid memory slot counts around WRITE.
            moe_loss, ponder_loss, reasoning_loss: Subsystem auxiliary losses.
            logits, confidence: Final output tensors (read-only, detached).

        Returns:
            Structured diagnostics dictionary (numerical only).
        """
        # KSC contribution/status.
        ksc_info = {
            "enabled": True,
            "n_layers": int(ksc_layers),
            "pre_norm": float(pre_ksc_norm),
            "post_norm": float(post_ksc_norm),
            "state_delta_norm": float(abs(post_ksc_state_norm - pre_ksc_state_norm)),
            "post_state_norm": float(post_ksc_state_norm),
            # Recurrent state shape is fixed by config; expose numerically.
            "recurrent_state_shape": [
                batch_size,
                self.config.n_heads,
                self.config.d_k,
                self.config.d_expansion,
            ],
        }

        # Sparse MoE contribution/status (router remains authoritative).
        moe_enabled = self.shared_moe_layer is not None
        last_routed = (
            list(self.shared_moe_layer.last_routed_experts)
            if moe_enabled and hasattr(self.shared_moe_layer, "last_routed_experts")
            else []
        )
        moe_info = {
            "enabled": moe_enabled,
            "num_experts": int(self.config.num_experts) if moe_enabled else 0,
            "top_k_experts": int(self.config.top_k_experts) if moe_enabled else 0,
            "moe_layers_executed": int(moe_layers_executed),
            "experts_executed_last": [int(e) for e in last_routed],
            "aux_loss": float(moe_loss.detach().item()) if moe_enabled else 0.0,
        }

        # Dual Memory contribution/status. The gating regularization scalar is
        # already computed on the live forward path (``memory_gate_loss``); it is
        # surfaced in the losses dict, so it is not re-computed here.
        memory_info = {
            "read_active": bool(read_active),
            "read_vector_norm": float(mem_read_norm),
            "write_active": bool(pathway_flags.use_memory_write.any().item()),
            "stored_slots_before": int(memory_stored_before),
            "stored_slots_after": int(memory_stored_after),
            "max_slots": int(self.config.memory_slots),
        }

        # ARRC contribution/status (preserves existing diagnostics).
        arrc_info = {
            "enabled": self.reasoner is not None,
            "mean_cycles": float(reasoner_diag.get("mean_cycles", 0.0)),
            "mean_remainder": float(reasoner_diag.get("mean_remainder", 0.0)),
            "post_norm": float(post_arrc_norm),
            "ponder_loss": float(ponder_loss.detach().item()),
        }

        # Neural Reasoning contribution/status (latent, no textual trace).
        reasoning_info = {
            "enabled": self.reasoning_core is not None,
            "reasoning_steps": int(reasoning_diag.get("reasoning_steps", 0)),
            "correction_count": int(reasoning_diag.get("correction_count", 0)),
            "converged": bool(reasoning_diag.get("converged", False)),
            "confidence": float(reasoning_diag.get("confidence", 0.0)),
            "latent_delta_norm": float(reasoning_diag.get("latent_delta_norm", 0.0)),
            "pre_norm": float(pre_reasoning_norm),
            "post_norm": float(post_reasoning_norm),
            "reasoning_loss": float(reasoning_loss.detach().item()),
        }

        # Full-core execution summary.
        full_core_info = {
            "batch_size": int(batch_size),
            "seq_len": int(seq_len),
            "d_model": int(self.config.d_model),
            "selected_pathways": [int(p.item()) for p in selected_pathways],
            "output_logits_shape": [int(s) for s in logits.shape],
            "mean_output_confidence": float(
                torch.mean(confidence.detach()).item()
            ),
            "total_aux_loss": float(
                (moe_loss + ponder_loss + reasoning_loss).detach().item()
            ),
            "components_integrated": {
                "ksc": True,
                "moe": moe_enabled,
                "memory": True,
                "arrc": self.reasoner is not None,
                "reasoning": self.reasoning_core is not None,
            },
        }

        return {
            "ksc": ksc_info,
            "moe": moe_info,
            "memory": memory_info,
            "arrc": arrc_info,
            "reasoning": reasoning_info,
            "full_core": full_core_info,
        }

    def count_parameters(self) -> int:
        """Return total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_memory_footprint_mb(self) -> float:
        """Estimate CPU RAM footprint of model parameters in megabytes."""
        total_bytes = sum(
            p.numel() * p.element_size() for p in self.parameters()
        )
        return total_bytes / (1024.0 * 1024.0)
