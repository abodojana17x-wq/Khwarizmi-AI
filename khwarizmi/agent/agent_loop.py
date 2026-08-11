"""
Khwarizmi Layered Offline Agent Loop Orchestrator.

Implements the end-to-end execution loop linking:
    User Input
    -> Input Sanitizer
    -> Short-Term Working State & Long-Term Memory (READ)
    -> Cognitive Router (Selects Pathway)
    -> Khwarizmi Core Sequence Modeling (KSC Blocks + Sparse MoE)
    -> Adaptive Computation / Latent Reasoning Loop
    -> Output Pathway (Logits + Confidence + Needs Verification)
    -> Optional Local Tools (Python Brain / Project Planner if Verification triggered)
    -> Final Output Response
"""

from dataclasses import dataclass
import torch
from typing import Dict, Any, Optional, Tuple, List

from ..config.settings import KhwarizmiConfig
from ..core.model import KhwarizmiModel, KhwarizmiOutput
from ..tools.verifier import SelectiveVerifier, ToolVerificationRequest, ToolVerificationResult
from .input_filter import InputSanitizer, SanitizedInputFrame


@dataclass
class AgentResponseFrame:
    """
    Final end-to-end response frame produced by the Khwarizmi offline agent.

    Attributes:
        input_frame: Sanitized input metadata.
        neural_output: KhwarizmiOutput structured dataclass from the core model.
        tool_verification: Optional ToolVerificationResult if selective verification was triggered.
        selected_pathway: Human-readable name of the computational pathway selected by router.
        confidence_score: Output statistical confidence C(y).
        diagnostics: Complete end-to-end execution diagnostics.
    """
    input_frame: SanitizedInputFrame
    neural_output: KhwarizmiOutput
    tool_verification: Optional[ToolVerificationResult]
    selected_pathway: str
    confidence_score: float
    diagnostics: Dict[str, Any]


class KhwarizmiAgentLoop:
    """
    Layered Offline Assistant Orchestrator.
    Decouples raw neural sequence modeling from deterministic verification tools.
    """

    def __init__(self, config: KhwarizmiConfig, model: Optional[KhwarizmiModel] = None):
        self.config = config
        self.model = model or KhwarizmiModel(config)
        self.step_counter = 0

    def encode_prompt_to_ids(self, text: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Minimal offline byte-fallback character/subword encoder for Phase 1 CPU testing.
        Maps text characters into vocabulary indices clamped to vocab_size.

        Args:
            text: Input text prompt string.
            device: Target torch device.

        Returns:
            2D token id Tensor of shape (1, seq_len).
        """
        token_ids = [min(ord(c), self.config.vocab_size - 1) for c in text[: self.config.max_seq_len]]
        if not token_ids:
            token_ids = [0]
        return torch.tensor([token_ids], dtype=torch.long, device=device)

    def process_request(
        self,
        user_input: str,
        short_term_state: Optional[Dict[str, torch.Tensor]] = None,
        long_term_table: Optional[Dict[str, torch.Tensor]] = None,
        deterministic_router: bool = True,
        force_cycles: Optional[int] = None,
    ) -> AgentResponseFrame:
        """
        Execute full end-to-end user request processing loop.

        Args:
            user_input: Raw natural language or code string.
            short_term_state: Optional active ephemeral working state dictionary.
            long_term_table: Optional persistent memory table dictionary.
            deterministic_router: Whether Cognitive Router selects argmax pathway.
            force_cycles: Optional override for exact Adaptive Compute cycles.

        Returns:
            AgentResponseFrame containing neural outputs, optional tool verifications, and diagnostics.
        """
        self.step_counter += 1

        # Step 1: Layer 1 - Offline Agent Input Sanitization & Classification
        sanitized = InputSanitizer.sanitize(user_input)
        input_ids = self.encode_prompt_to_ids(sanitized.raw_text)

        # Step 2: Layers 2-4 - Cognitive Router & Khwarizmi Neural Core Sequence Pass
        neural_out = self.model(
            input_ids=input_ids,
            short_term_state=short_term_state,
            long_term_table=long_term_table,
            deterministic_router=deterministic_router,
            force_cycles=force_cycles,
            step_counter=self.step_counter,
        )

        pathway_idx = neural_out.selected_pathways[0].item()
        pathway_name = neural_out.diagnostics["selected_pathway_names"][0]
        confidence_score = neural_out.confidence[0].item()
        needs_verif = neural_out.needs_verification[0].item()

        # Step 3: Layer 5 - Optional Local Deterministic Tools (never called automatically on every request)
        tool_result = None
        if needs_verif or pathway_name in ("CODING", "PROJECT_PLAN", "VERIFICATION"):
            if sanitized.has_code_payload or pathway_name == "CODING":
                req = ToolVerificationRequest(
                    tool_name="python_brain",
                    payload=sanitized.raw_text,
                    metadata={"pathway": pathway_name, "confidence": confidence_score},
                )
                tool_result = SelectiveVerifier.verify(req, needs_verification=True)
            elif sanitized.has_dag_payload or pathway_name == "PROJECT_PLAN":
                req = ToolVerificationRequest(
                    tool_name="project_planner",
                    payload=sanitized.raw_text,
                    metadata={"pathway": pathway_name, "confidence": confidence_score},
                )
                tool_result = SelectiveVerifier.verify(req, needs_verification=True)
            else:
                req = ToolVerificationRequest(
                    tool_name="skipped",
                    payload="",
                )
                tool_result = SelectiveVerifier.verify(req, needs_verification=False)

        diagnostics = {
            "step_counter": self.step_counter,
            "detected_language": sanitized.detected_language,
            "has_code_payload": sanitized.has_code_payload,
            "has_dag_payload": sanitized.has_dag_payload,
            "selected_pathway": pathway_name,
            "confidence_score": confidence_score,
            "needs_verification": needs_verif,
            "tool_executed": tool_result.tool_name if tool_result else "none",
            "tool_success": tool_result.success if tool_result else True,
            "model_diagnostics": neural_out.diagnostics,
        }

        return AgentResponseFrame(
            input_frame=sanitized,
            neural_output=neural_out,
            tool_verification=tool_result,
            selected_pathway=pathway_name,
            confidence_score=confidence_score,
            diagnostics=diagnostics,
        )
