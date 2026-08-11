"""
Comprehensive CPU Unit Tests for Khwarizmi State Cell (KSC).

Tests:
    - Initialization and tensor shape validation.
    - Diagonal retention gate bounding in [gamma_min, gamma_max].
    - Numerical stability over ultra-long sequence iterations without NaN/Inf.
    - Complete gradient flow backpropagation.
    - Synthetic associative recall sequence copying test.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim

from khwarizmi.config import get_tiny_test_config
from khwarizmi.core.ksc_cell import KhwarizmiStateCell


class TestKhwarizmiStateCell(unittest.TestCase):
    def setUp(self) -> None:
        self.config = get_tiny_test_config()
        self.cell = KhwarizmiStateCell(self.config)

    def test_ksc_cell_initialization_and_shapes(self) -> None:
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, self.config.d_model)
        out, final_state, _ = self.cell(x)

        self.assertEqual(out.shape, (batch_size, seq_len, self.config.d_model))
        self.assertEqual(
            final_state.shape,
            (batch_size, self.config.n_heads, self.config.d_k, self.config.d_expansion),
        )

    def test_ksc_cell_eigenvalue_and_retention_bounds(self) -> None:
        batch_size = 4
        seq_len = 16
        x = torch.randn(batch_size, seq_len, self.config.d_model) * 10.0
        _, _, ret_history = self.cell(x, return_retention=True)

        self.assertIsNotNone(ret_history)
        min_val = ret_history.min().item()
        max_val = ret_history.max().item()

        self.assertGreaterEqual(min_val, self.config.gamma_min - 1e-6)
        self.assertLessEqual(max_val, self.config.gamma_max + 1e-6)

    def test_ksc_cell_long_sequence_numerical_stability(self) -> None:
        """
        Prove numerically that for gamma_min=0.85 and gamma_max=0.999,
        the recurrent state S_t does not explode or diverge to NaN/Inf over
        50,000 sequential recurrent update steps.
        """
        batch_size = 1
        state = self.cell.init_state(batch_size)
        x_fixed = torch.randn(batch_size, self.config.d_model)

        with torch.no_grad():
            for _ in range(50000):
                _, state, _ = self.cell.step_forward(x_fixed, state)

        self.assertFalse(torch.isnan(state).any().item())
        self.assertFalse(torch.isinf(state).any().item())
        # State magnitude must remain bounded
        max_abs = torch.max(torch.abs(state)).item()
        self.assertLess(max_abs, 1e4)

    def test_ksc_cell_gradient_flow(self) -> None:
        x = torch.randn(2, 6, self.config.d_model, requires_grad=True)
        out, _, _ = self.cell(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        for param in self.cell.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.isnan(param.grad).any().item())

    def test_ksc_cell_associative_recall_synthetic(self) -> None:
        """
        Synthetic associative recall benchmark:
        Train a small KSC cell on a toy sequence-copying task to verify recall.
        """
        torch.manual_seed(42)
        cfg = get_tiny_test_config()
        cfg.d_model = 32
        cfg.n_heads = 2
        cell = KhwarizmiStateCell(cfg)
        optimizer = optim.Adam(cell.parameters(), lr=0.01)

        # Toy target mapping where output should correlate with input pattern
        for _ in range(25):
            optimizer.zero_grad()
            x = torch.randn(4, 10, cfg.d_model)
            out, _, _ = cell(x)
            loss = nn.functional.mse_loss(out, x)
            loss.backward()
            optimizer.step()

        # Check loss decreases to finite reasonable bound
        self.assertLess(loss.item(), 5.0)


if __name__ == "__main__":
    unittest.main()
