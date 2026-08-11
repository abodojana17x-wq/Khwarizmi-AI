"""
Comprehensive CPU Unit Tests for Khwarizmi Cognitive Router.

Tests:
    - Policy distribution probabilities sum to 1.0 across discrete pathways.
    - Deterministic argmax selection versus stochastic multinomial sampling.
    - PathwayDispatcher execution flags for FAST, CODING, REASONING, PROJECT_PLAN, VERIFICATION.
    - Multi-objective regularized router loss and gradient backpropagation.
"""

import unittest
import torch

from khwarizmi.config import get_tiny_test_config
from khwarizmi.routing import CognitiveRouter, PathwayDispatcher


class TestCognitiveRouter(unittest.TestCase):
    def setUp(self) -> None:
        self.config = get_tiny_test_config()
        self.router = CognitiveRouter(self.config)

    def test_cognitive_router_probabilities_sum_to_one(self) -> None:
        batch_size = 5
        x = torch.randn(batch_size, self.config.d_model)
        probs, _, _ = self.router(x)

        self.assertEqual(probs.shape, (batch_size, self.config.num_pathways))
        sums = torch.sum(probs, dim=-1)
        for s in sums:
            self.assertAlmostEqual(s.item(), 1.0, places=5)

    def test_cognitive_router_deterministic_vs_stochastic_selection(self) -> None:
        batch_size = 4
        x = torch.randn(batch_size, self.config.d_model)

        _, sel_det, _ = self.router(x, deterministic=True)
        _, sel_stoch, _ = self.router(x, deterministic=False)

        self.assertEqual(sel_det.shape, (batch_size,))
        self.assertEqual(sel_stoch.shape, (batch_size,))
        self.assertTrue((sel_det >= 0).all() and (sel_det < self.config.num_pathways).all())
        self.assertTrue((sel_stoch >= 0).all() and (sel_stoch < self.config.num_pathways).all())

    def test_pathway_dispatcher_flags(self) -> None:
        # Test explicit indices [0, 1, 2, 3, 4]
        selections = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        flags = PathwayDispatcher.dispatch(selections)

        # FAST (0) -> All False
        self.assertFalse(flags.use_moe[0].item())
        self.assertFalse(flags.use_adaptive_compute[0].item())
        self.assertFalse(flags.use_memory_read[0].item())
        self.assertFalse(flags.use_memory_write[0].item())
        self.assertFalse(flags.trigger_verification[0].item())

        # CODING (1) -> MoE=True, MemoryRead=True, Verify=True
        self.assertTrue(flags.use_moe[1].item())
        self.assertTrue(flags.use_memory_read[1].item())
        self.assertTrue(flags.trigger_verification[1].item())

        # REASONING (2) -> MoE=True, Adaptive=True, MemoryRead=True, MemoryWrite=True
        self.assertTrue(flags.use_moe[2].item())
        self.assertTrue(flags.use_adaptive_compute[2].item())
        self.assertTrue(flags.use_memory_write[2].item())

        # PROJECT_PLAN (3) -> MoE=True, MemoryRead=True, MemoryWrite=True, Verify=True
        self.assertTrue(flags.use_moe[3].item())
        self.assertTrue(flags.use_memory_write[3].item())
        self.assertTrue(flags.trigger_verification[3].item())

        # VERIFICATION (4) -> MoE=True, Adaptive=True, Verify=True
        self.assertTrue(flags.use_moe[4].item())
        self.assertTrue(flags.use_adaptive_compute[4].item())
        self.assertTrue(flags.trigger_verification[4].item())

    def test_router_regularization_loss_gradients(self) -> None:
        x = torch.randn(3, self.config.d_model, requires_grad=True)
        _, _, loss = self.router(x)
        loss.backward()

        self.assertIsNotNone(x.grad)
        for param in self.router.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


if __name__ == "__main__":
    unittest.main()
