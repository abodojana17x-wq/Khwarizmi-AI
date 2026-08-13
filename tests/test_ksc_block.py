"""
Phase 2 Unit Tests for the KSC Residual Block.

Tests for :class:`khwarizmi.core.ksc_block.KSCResidualBlock` and
:class:`khwarizmi.core.ksc_block.FeedForwardNetwork`.

Covered:
    - Initialization, tensor shapes and config validation.
    - Forward/backward pass shape correctness (sequence and single-token).
    - Complete gradient flow backpropagation (no NaN/Inf gradients).
    - Sequence invariance / recurrence consistency (vectorized forward must
      equal token-by-token recurrence through the KSC sub-layer).
    - Retention-gate eigenvalue bounds propagated from the Phase 1 KSC cell.
    - Numerical stability over long sequences (no NaN/Inf in output or state).
    - Edge cases (seq_len == 1, very long sequence).
    - Invalid input handling (non 2D/3D tensor, d_model mismatch, bad state).
    - Integration with the Phase 1 :class:`KhwarizmiStateCell` (aux_loss is
      ``None`` for a pure (non-MoE) Phase 2 block).
    - Regression: deterministic output for identical inputs in ``eval`` mode.
"""

import unittest

import torch

from khwarizmi.config import get_tiny_test_config, KhwarizmiConfig
from khwarizmi.core.ksc_block import KSCResidualBlock, FeedForwardNetwork


class TestKSCResidualBlock(unittest.TestCase):
    def setUp(self) -> None:
        self.config = get_tiny_test_config()
        self.block = KSCResidualBlock(self.config, is_moe_layer=False)
        torch.manual_seed(0)

    # ---------------------------------------------------------- shapes
    def test_block_initialization_and_shapes(self) -> None:
        batch_size, seq_len = 2, 9
        x = torch.randn(batch_size, seq_len, self.config.d_model)
        # Fresh zero state for the single KSC sub-layer inside the block.
        state = self.block.ksc.init_state(batch_size)

        out, new_state, aux_loss, ret = self.block(x, state=state)

        self.assertEqual(out.shape, (batch_size, seq_len, self.config.d_model))
        self.assertEqual(
            new_state.shape,
            (batch_size, self.config.n_heads, self.config.d_k, self.config.d_expansion),
        )
        self.assertIsNone(aux_loss)
        self.assertIsNone(ret)

    def test_block_single_token_forward_shape(self) -> None:
        batch_size = 3
        x = torch.randn(batch_size, self.config.d_model)
        state = self.block.ksc.init_state(batch_size)

        out, new_state, _, _ = self.block(x, state=state)

        self.assertEqual(out.shape, (batch_size, self.config.d_model))
        self.assertEqual(
            new_state.shape,
            (batch_size, self.config.n_heads, self.config.d_k, self.config.d_expansion),
        )

    # ----------------------------------------------------- gradient flow
    def test_block_gradient_flow(self) -> None:
        batch_size, seq_len = 2, 7
        x = torch.randn(batch_size, seq_len, self.config.d_model, requires_grad=True)
        state = self.block.ksc.init_state(batch_size)

        out, _, _, _ = self.block(x, state=state)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        grad_receivers = 0
        for name, param in self.block.named_parameters():
            self.assertIsNotNone(param.grad, msg=f"param {name} has no grad")
            self.assertFalse(
                torch.isnan(param.grad).any().item(), msg=f"param {name} grad has NaN"
            )
            self.assertFalse(
                torch.isinf(param.grad).any().item(), msg=f"param {name} grad has Inf"
            )
            grad_receivers += 1
        # The block has many parameter tensors (KSC + FFN + norms).
        self.assertGreater(grad_receivers, 10)

    # ----------------------------------------- sequence invariance
    def test_block_recurrence_consistency(self) -> None:
        """
        The block's vectorized forward must equal processing the same tokens
        one at a time through the recurrent KSC sub-layer (state carried
        between steps). This is the defining "sequence invariance" property of
        a causal recurrent model.
        """
        batch_size, seq_len = 2, 12
        x = torch.randn(batch_size, seq_len, self.config.d_model)

        # Vectorized forward.
        state = self.block.ksc.init_state(batch_size)
        out_full, _, _, _ = self.block(x, state=state)

        # Token-by-token recurrence.
        state = self.block.ksc.init_state(batch_size)
        per_token = []
        for t in range(seq_len):
            out_t, state, _, _ = self.block(x[:, t : t + 1, :], state=state)
            per_token.append(out_t)
        out_step = torch.cat(per_token, dim=1)

        self.assertEqual(out_full.shape, out_step.shape)
        self.assertTrue(
            torch.allclose(out_full, out_step, atol=1e-5, rtol=1e-4),
            msg="Vectorized block forward diverges from token-by-token recurrence",
        )

    # ------------------------------------------- retention bounds
    def test_block_retention_bounds(self) -> None:
        batch_size, seq_len = 4, 14
        x = torch.randn(batch_size, seq_len, self.config.d_model) * 5.0
        state = self.block.ksc.init_state(batch_size)

        _, _, _, ret = self.block(x, state=state, return_retention=True)
        self.assertIsNotNone(ret)
        # ret shape: (B, L, H, d_k)
        self.assertEqual(
            ret.shape,
            (batch_size, seq_len, self.config.n_heads, self.config.d_k),
        )
        self.assertGreaterEqual(ret.min().item(), self.config.gamma_min - 1e-6)
        self.assertLessEqual(ret.max().item(), self.config.gamma_max + 1e-6)

    # ------------------------------------------ numerical stability
    def test_block_long_sequence_numerical_stability(self) -> None:
        batch_size = 1
        state = self.block.ksc.init_state(batch_size)
        x_fixed = torch.randn(batch_size, self.config.d_model)

        with torch.no_grad():
            for _ in range(2000):
                _, state, _, _ = self.block(x_fixed, state=state)

        self.assertFalse(torch.isnan(state).any().item())
        self.assertFalse(torch.isinf(state).any().item())
        self.assertLess(torch.max(torch.abs(state)).item(), 1e4)

    # ------------------------------------------------------- edge cases
    def test_block_seq_len_one(self) -> None:
        batch_size = 2
        x = torch.randn(batch_size, 1, self.config.d_model)
        state = self.block.ksc.init_state(batch_size)
        out, new_state, _, _ = self.block(x, state=state)
        self.assertEqual(out.shape, (batch_size, 1, self.config.d_model))
        self.assertFalse(torch.isnan(new_state).any().item())

    def test_block_invalid_input_dimension_raises(self) -> None:
        bad = torch.randn(2, 3, 4, self.config.d_model)  # 4D
        state = self.block.ksc.init_state(2)
        with self.assertRaises(ValueError):
            self.block(bad, state=state)

    def test_block_d_model_mismatch_raises(self) -> None:
        bad = torch.randn(2, 5, self.config.d_model + 1)
        state = self.block.ksc.init_state(2)
        with self.assertRaises(ValueError):
            self.block(bad, state=state)

    def test_block_invalid_state_shape_raises(self) -> None:
        x = torch.randn(2, 5, self.config.d_model)
        bad_state = torch.zeros(2, self.config.n_heads, self.config.d_k, self.config.d_expansion + 1)
        with self.assertRaises(ValueError):
            self.block(x, state=bad_state)

    # ----------------------------------- Phase 1 integration / regression
    def test_block_runs_with_phase1_ksc_cell(self) -> None:
        """The block is built on the Phase 1 KhwarizmiStateCell (no MoE)."""
        from khwarizmi.core.ksc_cell import KhwarizmiStateCell

        self.assertIsInstance(self.block.ksc, KhwarizmiStateCell)
        # Non-MoE Phase 2 block never produces an auxiliary MoE loss.
        x = torch.randn(2, 6, self.config.d_model)
        state = self.block.ksc.init_state(2)
        _, _, aux_loss, _ = self.block(x, state=state)
        self.assertIsNone(aux_loss)

    def test_block_deterministic_in_eval_mode(self) -> None:
        self.block.eval()
        x = torch.randn(2, 8, self.config.d_model)
        state = self.block.ksc.init_state(2)
        out1, _, _, _ = self.block(x, state=state)
        state2 = self.block.ksc.init_state(2)
        out2, _, _, _ = self.block(x, state=state2)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))


class TestFeedForwardNetwork(unittest.TestCase):
    def setUp(self) -> None:
        self.config = get_tiny_test_config()
        self.ffn = FeedForwardNetwork(self.config)

    def test_ffn_shapes_and_gradient(self) -> None:
        x = torch.randn(2, 6, self.config.d_model, requires_grad=True)
        out = self.ffn(x)
        self.assertEqual(out.shape, x.shape)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        for p in self.ffn.parameters():
            self.assertIsNotNone(p.grad)

    def test_ffn_is_used_by_non_moe_block(self) -> None:
        block = KSCResidualBlock(self.config, is_moe_layer=False)
        self.assertIsNotNone(block.ffn)
        self.assertIsInstance(block.ffn, FeedForwardNetwork)


if __name__ == "__main__":
    unittest.main()
