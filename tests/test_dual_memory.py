"""
Comprehensive CPU Unit Tests for Khwarizmi Dual Memory Architecture.

Tests:
    - Short-Term Working State initialization, rolling window buffer, and summary vector.
    - Long-Term Persistent Memory associative retrieval via cosine similarity.
    - Selective WRITE insertion and time-decayed utility eviction when full.
    - FORGET eviction operation.
    - Memory Gating Controller probability outputs and gradient flow.
"""

import unittest
import torch
import torch.nn as nn

from khwarizmi.config import get_tiny_test_config
from khwarizmi.memory import (
    ShortTermWorkingState,
    MemoryGatingController,
    LongTermPersistentMemory,
)


class TestDualMemoryArchitecture(unittest.TestCase):
    def setUp(self) -> None:
        self.config = get_tiny_test_config()
        self.st = ShortTermWorkingState(self.config)
        self.gating = MemoryGatingController(self.config)
        self.lt = LongTermPersistentMemory(self.config)

    def test_short_term_working_state_window_and_summary(self) -> None:
        batch_size = 2
        st_state = self.st.init_state(batch_size)

        self.assertEqual(
            st_state["recurrent_state"].shape,
            (batch_size, self.config.n_heads, self.config.d_k, self.config.d_expansion),
        )
        self.assertEqual(
            st_state["window_buffer"].shape,
            (batch_size, 0, self.config.d_model),
        )

        # Append tokens exceeding window size to test FIFO eviction
        x_seq = torch.randn(batch_size, self.config.max_seq_len + 10, self.config.d_model)
        new_rec_state = torch.ones_like(st_state["recurrent_state"])
        st_state = self.st.update(st_state, new_rec_state, x_seq)

        self.assertEqual(
            st_state["window_buffer"].shape,
            (batch_size, self.config.max_seq_len, self.config.d_model),
        )

        summary = self.st.get_summary_vector(st_state)
        self.assertEqual(summary.shape, (batch_size, self.config.d_model))

    def test_long_term_memory_associative_read(self) -> None:
        batch_size = 2
        mem_table = self.lt.init_memory_table(batch_size)

        # Write a known vector into slot 0
        cand = torch.ones(batch_size, self.config.d_model)
        mem_table = self.lt.write(
            cand,
            mem_table,
            g_write=torch.ones(batch_size),
            current_step=1,
            threshold=0.1,
        )

        self.assertTrue(mem_table["valid_mask"][:, 0].all().item())

        # Perform read with query aligned to cand
        read_out, attn = self.lt.read(
            cand,
            mem_table,
            g_read=torch.ones(batch_size),
            current_step=2,
        )

        self.assertEqual(read_out.shape, (batch_size, self.config.d_model))
        self.assertEqual(attn.shape, (batch_size, self.config.memory_slots))
        self.assertGreater(attn[:, 0].min().item(), 0.9)

    def test_long_term_memory_write_and_time_decay_eviction(self) -> None:
        """Test write insertion and verify time-decayed utility eviction when table is full."""
        batch_size = 1
        mem_table = self.lt.init_memory_table(batch_size)

        # Fill table to capacity
        for idx in range(self.config.memory_slots):
            cand = torch.randn(batch_size, self.config.d_model)
            mem_table = self.lt.write(
                cand,
                mem_table,
                g_write=torch.ones(batch_size),
                current_step=idx + 1,
                threshold=0.1,
            )

        self.assertTrue(mem_table["valid_mask"].all().item())

        # Manually set slot 0 to have very old access time and low utility
        mem_table["access_times"][0, 0] = 0
        mem_table["utilities"][0, 0] = 0.001

        # Write one more item when table is full
        new_cand = torch.randn(batch_size, self.config.d_model)
        mem_table = self.lt.write(
            new_cand,
            mem_table,
            g_write=torch.ones(batch_size),
            current_step=1000,
            threshold=0.1,
        )

        # Slot 0 should have been evicted and replaced with timestamp 1000
        self.assertEqual(mem_table["access_times"][0, 0].item(), 1000)

    def test_long_term_memory_forget(self) -> None:
        batch_size = 2
        mem_table = self.lt.init_memory_table(batch_size)
        cand = torch.randn(batch_size, self.config.d_model)
        mem_table = self.lt.write(
            cand, mem_table, g_write=torch.ones(batch_size), current_step=1, threshold=0.1
        )
        self.assertTrue(mem_table["valid_mask"].any().item())

        mem_table = self.lt.forget(
            mem_table, g_forget=torch.ones(batch_size), threshold=0.5
        )
        # Lowest utility valid item should be evicted
        self.assertFalse(mem_table["valid_mask"].all().item())

    def test_memory_gating_controller_shapes_and_gradients(self) -> None:
        h = torch.randn(3, self.config.d_model, requires_grad=True)
        gates = self.gating(h)

        for gate_name in ("read", "write", "update", "forget"):
            self.assertIn(gate_name, gates)
            self.assertEqual(gates[gate_name].shape, (3,))
            self.assertTrue((gates[gate_name] >= 0.0).all().item())
            self.assertTrue((gates[gate_name] <= 1.0).all().item())

        loss = sum(gates[k].sum() for k in gates)
        loss.backward()

        self.assertIsNotNone(h.grad)
        for param in self.gating.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)


if __name__ == "__main__":
    unittest.main()
