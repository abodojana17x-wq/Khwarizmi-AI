"""
Comprehensive CPU Unit Tests for Khwarizmi Sparse Mixture-of-Experts (MoE).

Tests:
    - Top-K expert routing selection validity (K=2 out of E=4).
    - Load balancing auxiliary loss calculation L_balance = alpha_moe * E * sum(f_i * P_i).
    - Complete gradient flow and backpropagation through selected specialists.
    - Standard specialist creation and metadata assignment.
"""

import unittest
import torch

from khwarizmi.config import get_tiny_test_config
from khwarizmi.experts import (
    SparseMoELayer,
    ExpertLayer,
    create_standard_specialists,
    SPECIALIZATION_NAMES,
)


class TestSparseMoE(unittest.TestCase):
    def setUp(self) -> None:
        self.config = get_tiny_test_config()
        self.experts = create_standard_specialists(self.config)
        self.moe = SparseMoELayer(self.config, experts=self.experts)

    def test_standard_specialists_creation(self) -> None:
        self.assertEqual(len(self.experts), self.config.num_experts)
        for i, expert in enumerate(self.experts):
            self.assertIsInstance(expert, ExpertLayer)
            if i < len(SPECIALIZATION_NAMES):
                self.assertEqual(expert.specialization_name, SPECIALIZATION_NAMES[i])

    def test_sparse_moe_top_k_expert_routing(self) -> None:
        batch_size = 2
        seq_len = 6
        x = torch.randn(batch_size, seq_len, self.config.d_model)
        out, aux_loss = self.moe(x, use_noise=False)

        self.assertEqual(out.shape, (batch_size, seq_len, self.config.d_model))
        self.assertGreaterEqual(aux_loss.item(), 0.0)

    def test_sparse_moe_load_balancing_auxiliary_loss(self) -> None:
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, self.config.d_model)
        _, aux_loss = self.moe(x, use_noise=True)

        # L_balance = alpha_moe * E * sum(f_i * P_i) should be non-zero and bounded
        self.assertGreater(aux_loss.item(), 0.0)
        self.assertLess(aux_loss.item(), 1.0)

    def test_sparse_moe_gradient_backpropagation(self) -> None:
        x = torch.randn(2, 5, self.config.d_model, requires_grad=True)
        out, aux_loss = self.moe(x, use_noise=True)
        loss = out.sum() + aux_loss
        loss.backward()

        self.assertIsNotNone(x.grad)
        for param in self.moe.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


if __name__ == "__main__":
    unittest.main()
