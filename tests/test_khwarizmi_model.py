"""
Comprehensive CPU Unit Tests for Khwarizmi AI End-to-End Core Model.

Tests:
    - Configuration validation, bounds checking, and JSON serialization.
    - Invalid configuration handling (raises ValueError for invalid shapes/bounds).
    - Tiny test configuration parameter count (<500K) and CPU memory usage (<3 MB).
    - Complete end-to-end forward pass across all internal modules.
    - Differentiable neural path: prove gradients flow through all components
      (Embeddings, KSC layers, MoE specialists, Dual Memory, Router, and Output Pathway).
"""

import unittest
import torch
import torch.nn as nn

from khwarizmi.config import (
    KhwarizmiConfig,
    get_tiny_test_config,
    get_prototype_config,
    get_small_config,
    get_edge_config,
)
from khwarizmi.core import KhwarizmiModel, KhwarizmiOutput


class TestKhwarizmiModel(unittest.TestCase):
    def setUp(self) -> None:
        # Deterministic initialization: the Cognitive Router selects its pathway
        # from randomly-initialized parameters, which gates whether the Sparse-MoE
        # sub-layer is executed. Seeding avoids an order-dependent flake where the
        # router would pick the FAST pathway (MoE bypassed) and MoE parameters
        # legitimately receive no gradient.
        torch.manual_seed(0)
        self.config = get_tiny_test_config()
        self.model = KhwarizmiModel(self.config)

    def test_khwarizmi_config_validation_and_json_serialization(self) -> None:
        json_str = self.config.to_json_string()
        loaded_cfg = KhwarizmiConfig.from_json_string(json_str)
        self.assertEqual(self.config.d_model, loaded_cfg.d_model)
        self.assertEqual(self.config.n_heads, loaded_cfg.n_heads)
        self.assertEqual(self.config.tier_name, loaded_cfg.tier_name)

    def test_khwarizmi_config_invalid_handling(self) -> None:
        # Indivisible head dim
        with self.assertRaises(ValueError):
            KhwarizmiConfig(d_model=64, n_heads=5)

        # Invalid top_k_experts
        with self.assertRaises(ValueError):
            KhwarizmiConfig(num_experts=4, top_k_experts=5)

        # Invalid eigenvalue bounds
        with self.assertRaises(ValueError):
            KhwarizmiConfig(gamma_min=1.0, gamma_max=0.5)

    def test_khwarizmi_model_parameter_count_and_memory_footprint(self) -> None:
        param_count = self.model.count_parameters()
        mem_mb = self.model.get_memory_footprint_mb()

        # Phase 1 constraint: keep tiny test config lightweight and CPU-runnable (< 500k params)
        self.assertLess(param_count, 500000)
        self.assertLess(mem_mb, 3.0)

        # Verify predefined tier configs scale cleanly without architecture rewriting
        proto_cfg = get_prototype_config()
        small_cfg = get_small_config()
        edge_cfg = get_edge_config()
        self.assertEqual(proto_cfg.tier_name, "Prototype")
        self.assertEqual(small_cfg.tier_name, "Small")
        self.assertEqual(edge_cfg.tier_name, "Edge")

    def test_khwarizmi_model_end_to_end_forward_pass(self) -> None:
        batch_size = 2
        seq_len = 12
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        out = self.model(input_ids)

        self.assertIsInstance(out, KhwarizmiOutput)
        self.assertEqual(out.logits.shape, (batch_size, seq_len, self.config.vocab_size))
        self.assertEqual(out.confidence.shape, (batch_size,))
        self.assertEqual(out.needs_verification.shape, (batch_size,))
        self.assertEqual(out.selected_pathways.shape, (batch_size,))
        self.assertEqual(out.routing_probs.shape, (batch_size, self.config.num_pathways))

        self.assertIn("routing_loss", out.losses)
        self.assertIn("moe_aux_loss", out.losses)
        self.assertIn("ponder_loss", out.losses)
        self.assertIn("total_aux_loss", out.losses)

        self.assertIn("selected_pathway_names", out.diagnostics)
        self.assertEqual(len(out.diagnostics["selected_pathway_names"]), batch_size)

    def test_khwarizmi_model_differentiable_gradient_flow(self) -> None:
        """
        Critical Phase 1 Quality Gate:
        Prove that gradients flow through the complete differentiable neural path
        (Embeddings -> KSC -> MoE -> ARRC -> Dual Memory -> Router -> Output).
        """
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        out = self.model(input_ids)

        # Main cross-entropy task loss + auxiliary regularization
        loss_fct = nn.CrossEntropyLoss()
        task_loss = loss_fct(out.logits.view(-1, self.config.vocab_size), targets.view(-1))
        total_loss = task_loss + out.losses["total_aux_loss"]

        total_loss.backward()

        tensors_with_grad = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad,
                    msg=f"Parameter '{name}' did not receive gradients during backpropagation",
                )
                self.assertFalse(
                    torch.isnan(param.grad).any().item(),
                    msg=f"Parameter '{name}' grad contains NaN",
                )
                self.assertFalse(
                    torch.isinf(param.grad).any().item(),
                    msg=f"Parameter '{name}' grad contains Inf",
                )
                tensors_with_grad += 1

        # Assert that dozens of parameter tensors across all sub-modules received valid gradients
        self.assertGreater(tensors_with_grad, 50)


if __name__ == "__main__":
    unittest.main()
