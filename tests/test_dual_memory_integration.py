"""
Phase 3 — Dual Memory ↔ KSC Prototype Integration & Regression Tests.

Verifies that:

    * ``KhwarizmiDualMemoryPrototype`` composes the Phase 2 KSC prototype with the
      Phase 3 Dual Memory system and produces correct shapes.
    * Memory recall conditions the KSC forward pass (recalled state changes logits).
    * Memory remains bounded over long sequences.
    * The Phase 2 ``KhwarizmiKSCPrototype`` is *not* regressed: adding the optional
      ``memory_conditioning`` argument (None) preserves exact Phase 2 outputs, and
      existing public interfaces remain intact.
"""

import unittest

import torch

from khwarizmi.config import get_tiny_test_config
from khwarizmi.core import (
    KhwarizmiKSCPrototype,
    KhwarizmiDualMemoryPrototype,
    KhwarizmiDualMemoryOutput,
)


def _tiny_config():
    cfg = get_tiny_test_config()
    cfg.n_layers = 2
    cfg.d_model = 32
    cfg.n_heads = 4
    cfg.d_expansion = 8
    cfg.d_ff = 64
    cfg.vocab_size = 128
    cfg.max_seq_len = 64
    cfg.short_term_capacity = 32
    cfg.memory_slots = 8
    cfg.dropout = 0.0
    return cfg


class TestDualMemoryPrototype(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.config = _tiny_config()
        self.model = KhwarizmiDualMemoryPrototype(self.config)
        self.model.eval()

    def test_forward_shapes(self) -> None:
        ids = torch.randint(0, self.config.vocab_size, (2, 7))
        out = self.model(ids)
        self.assertIsInstance(out, KhwarizmiDualMemoryOutput)
        self.assertEqual(
            out.logits.shape, (2, 7, self.config.vocab_size)
        )
        self.assertEqual(len(out.ksc_state), self.config.n_layers)
        self.assertIn("short_term", out.memory_state)
        self.assertIn("long_term", out.memory_state)
        self.assertEqual(out.decision.shape, (2,))

    def test_memory_state_persists_across_calls(self) -> None:
        ids = torch.randint(0, self.config.vocab_size, (1, 5))
        out1 = self.model(ids, step_counter=0)
        slots1 = int(
            torch.sum(out1.memory_state["long_term"]["valid_mask"]).item()
        )
        out2 = self.model(
            ids, ksc_state=out1.ksc_state,
            memory_state=out1.memory_state, step_counter=1
        )
        self.assertIsNotNone(out2.memory_state)
        # Carried state is reused (not re-initialized) — table is still bounded.
        self.assertLessEqual(
            int(torch.sum(out2.memory_state["long_term"]["valid_mask"]).item()),
            self.config.memory_slots,
        )
        self.assertLessEqual(
            out2.memory_state["short_term"]["window_buffer"].size(1),
            self.config.short_term_capacity,
        )

    def test_recalled_memory_conditions_ksc(self) -> None:
        """Injecting a non-zero memory recall vector must change the logits."""
        ids = torch.randint(0, self.config.vocab_size, (1, 5))
        base = self.model.ksc(ids).logits
        conditioning = torch.randn(1, self.config.d_model)
        conditioned = self.model.ksc(
            ids, memory_conditioning=conditioning
        ).logits
        self.assertFalse(
            torch.allclose(base, conditioned, atol=1e-6),
            msg="memory_conditioning did not affect the KSC output",
        )

    def test_long_sequence_memory_bounded(self) -> None:
        """Repeated forwards must keep both memory stores within configured limits."""
        ksc_state, memory_state = self.model.init_state(1)
        for step in range(100):
            ids = torch.randint(0, self.config.vocab_size, (1, 4))
            out = self.model(
                ids,
                ksc_state=ksc_state,
                memory_state=memory_state,
                step_counter=step,
            )
            ksc_state = out.ksc_state
            memory_state = out.memory_state
            self.assertLessEqual(
                int(torch.sum(memory_state["long_term"]["valid_mask"]).item()),
                self.config.memory_slots,
            )
            self.assertLessEqual(
                memory_state["short_term"]["window_buffer"].size(1),
                self.config.short_term_capacity,
            )

    def test_gradient_flows_through_memory(self) -> None:
        self.model.train()
        ids = torch.randint(0, self.config.vocab_size, (2, 6))
        targets = torch.randint(0, self.config.vocab_size, (2, 6))
        out = self.model(ids)
        loss = torch.nn.functional.cross_entropy(
            out.logits.view(-1, self.config.vocab_size), targets.view(-1)
        )
        loss.backward()
        # The memory read path must receive gradients.
        self.assertIsNotNone(
            self.model.memory.long_term.out_proj.weight.grad
        )


class TestPhase2Regression(unittest.TestCase):
    """Guarantee Phase 1/Phase 2 behavior is unchanged by the Phase 3 integration."""

    def setUp(self) -> None:
        torch.manual_seed(3)
        self.config = _tiny_config()
        self.model = KhwarizmiKSCPrototype(self.config)
        self.model.eval()

    def test_memory_conditioning_none_preserves_phase2_output(self) -> None:
        ids = torch.randint(0, self.config.vocab_size, (2, 9))
        baseline = self.model(ids).logits
        with_none = self.model(ids, memory_conditioning=None).logits
        self.assertTrue(torch.equal(baseline, with_none))

    def test_forward_without_conditioning_matches_step(self) -> None:
        ids = torch.randint(0, self.config.vocab_size, (1, 6))
        logits_full = self.model(ids).logits
        state = self.model.init_state(1)
        stepped = []
        for pos in range(ids.size(1)):
            lg, state = self.model.step(ids[:, pos], state, position=pos)
            stepped.append(lg)
        logits_step = torch.stack(stepped, dim=1)
        self.assertTrue(torch.allclose(logits_full, logits_step, atol=1e-5, rtol=1e-4))

    def test_invalid_conditioning_shape_raises(self) -> None:
        ids = torch.randint(0, self.config.vocab_size, (2, 4))
        with self.assertRaises(ValueError):
            self.model(ids, memory_conditioning=torch.randn(2, 7))


if __name__ == "__main__":
    unittest.main()
