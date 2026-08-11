"""
Comprehensive CPU Unit Tests for Khwarizmi Adaptive Compute & Latent Reasoner.

Tests:
    - Recurrent cycles halting probability accumulation and stopping step K.
    - Forced cycles deterministic behavior for reproducibility.
    - Ponder cost regularization loss calculation and gradient backpropagation.
    - LatentReasoner pathway masking (bypasses adaptive compute for FAST pathway).
"""

import unittest
import torch

from khwarizmi.config import get_tiny_test_config
from khwarizmi.reasoning import AdaptiveComputeBlock, LatentReasoner


class TestAdaptiveCompute(unittest.TestCase):
    def setUp(self) -> None:
        self.config = get_tiny_test_config()
        self.block = AdaptiveComputeBlock(self.config)
        self.reasoner = LatentReasoner(self.config)

    def test_adaptive_compute_recurrent_cycles_and_halting(self) -> None:
        batch_size = 2
        seq_len = 5
        x = torch.randn(batch_size, seq_len, self.config.d_model)
        z_out, final_state, ponder_loss, diag = self.block(x)

        self.assertEqual(z_out.shape, (batch_size, seq_len, self.config.d_model))
        self.assertGreaterEqual(diag["mean_cycles"], 1.0)
        self.assertLessEqual(diag["mean_cycles"], float(self.config.max_recurrent_cycles))
        self.assertGreaterEqual(ponder_loss.item(), 0.0)

    def test_adaptive_compute_forced_cycles_deterministic_behavior(self) -> None:
        batch_size = 3
        seq_len = 4
        x = torch.randn(batch_size, seq_len, self.config.d_model)

        z1, _, ponder1, diag1 = self.block(x, force_cycles=2)
        z2, _, ponder2, diag2 = self.block(x, force_cycles=2)

        self.assertEqual(diag1["mean_cycles"], 2.0)
        self.assertTrue(torch.allclose(z1, z2, atol=1e-5))
        self.assertAlmostEqual(ponder1.item(), ponder2.item(), places=5)

    def test_adaptive_compute_ponder_loss_and_gradients(self) -> None:
        x = torch.randn(2, 4, self.config.d_model, requires_grad=True)
        z_out, _, ponder_loss, _ = self.block(x)
        loss = z_out.sum() + ponder_loss
        loss.backward()

        self.assertIsNotNone(x.grad)
        for param in self.block.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_latent_reasoner_pathway_masking(self) -> None:
        batch_size = 3
        seq_len = 5
        x = torch.randn(batch_size, seq_len, self.config.d_model)

        # Sequences 0 and 1 on FAST (0), Sequence 2 on REASONING (2)
        pathways = torch.tensor([0, 0, 2], dtype=torch.long)
        out, state, ponder, diag = self.reasoner.reason(x, pathway_id=pathways)

        self.assertEqual(out.shape, (batch_size, seq_len, self.config.d_model))
        # Sequences on FAST pathway should remain unmodified from input x
        self.assertTrue(torch.allclose(out[0], x[0], atol=1e-5))
        self.assertTrue(torch.allclose(out[1], x[1], atol=1e-5))
        # Sequence on REASONING pathway should be transformed
        self.assertFalse(torch.allclose(out[2], x[2], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
