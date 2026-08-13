"""
Phase 3 — Comprehensive Dual Memory Unit Tests.

Covers the full Phase 3 contract:

    * Short-Term Working State: initialization, bounded capacity, deterministic
      READ/WRITE/FORGET, FIFO eviction, invalid input.
    * Long-Term Persistent Memory: READ/WRITE/UPDATE/FORGET operations, utility
      eviction, near-duplicate detection, capacity limits, invalid memory ids
      and invalid states.
    * Utility gating: deterministic decision policy (RETAIN/WRITE/UPDATE/FORGET)
      and priority ordering.
    * DualMemory facade: combined lifecycle, boundedness over long sequences.
    * Determinism: identical inputs produce identical state.
"""

import unittest

import torch

from khwarizmi.config import get_tiny_test_config, KhwarizmiConfig
from khwarizmi.memory import (
    ShortTermWorkingState,
    LongTermPersistentMemory,
    MemoryGatingController,
    UtilityGatingPolicy,
    DualMemory,
    RETAIN,
    WRITE,
    UPDATE,
    FORGET,
)


class TestShortTermMemory(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.config = get_tiny_test_config()
        self.st = ShortTermWorkingState(self.config)

    def test_init_state_is_empty(self) -> None:
        state = self.st.init_state(3)
        self.assertEqual(
            state["recurrent_state"].shape,
            (3, self.config.n_heads, self.config.d_k, self.config.d_expansion),
        )
        self.assertEqual(state["window_buffer"].shape, (3, 0, self.config.d_model))
        self.assertEqual(self.st.num_stored(state), 0)

    def test_capacity_is_explicit_and_positive(self) -> None:
        self.assertEqual(self.st.capacity, self.config.short_term_capacity)
        self.assertGreaterEqual(self.st.capacity, 1)

    def test_write_appends_in_order_and_respects_capacity(self) -> None:
        state = self.st.init_state(1)
        cap = self.st.capacity
        # Write more than capacity and confirm strict FIFO bounding.
        feats = torch.randn(1, cap + 50, self.config.d_model)
        state = self.st.write(state, feats)
        self.assertEqual(state["window_buffer"].shape, (1, cap, self.config.d_model))
        # The buffer must hold the *most recent* `cap` features.
        self.assertTrue(
            torch.allclose(state["window_buffer"][0], feats[0, -cap:, :])
        )

    def test_repeated_writes_never_exceed_capacity(self) -> None:
        state = self.st.init_state(2)
        for _ in range(20):
            state = self.st.write(
                state, torch.randn(2, 7, self.config.d_model)
            )
        self.assertLessEqual(
            state["window_buffer"].size(1), self.st.capacity
        )

    def test_read_returns_window_and_recent(self) -> None:
        state = self.st.init_state(2)
        feats = torch.randn(2, 5, self.config.d_model)
        state = self.st.write(state, feats)
        full = self.st.read(state)
        self.assertEqual(full.shape, (2, 5, self.config.d_model))
        recent = self.st.read(state, n_recent=2)
        self.assertEqual(recent.shape, (2, 2, self.config.d_model))
        self.assertTrue(torch.allclose(recent[0], feats[0, -2:, :]))

    def test_forget_clears_window_preserves_recurrent_state(self) -> None:
        state = self.st.init_state(1)
        state["recurrent_state"] = torch.ones_like(state["recurrent_state"])
        state = self.st.write(state, torch.randn(1, 3, self.config.d_model))
        self.assertEqual(self.st.num_stored(state), 3)
        state = self.st.forget(state)
        self.assertEqual(self.st.num_stored(state), 0)
        self.assertEqual(state["window_buffer"].shape, (1, 0, self.config.d_model))
        self.assertTrue(
            torch.all(state["recurrent_state"] == 1.0).item()
        )

    def test_write_invalid_feature_dim_raises(self) -> None:
        state = self.st.init_state(1)
        with self.assertRaises(ValueError):
            self.st.write(state, torch.randn(1, 2, self.config.d_model + 1))

    def test_read_negative_n_recent_raises(self) -> None:
        state = self.st.init_state(1)
        with self.assertRaises(ValueError):
            self.st.read(state, n_recent=-1)


class TestLongTermMemoryOperations(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1)
        self.config = get_tiny_test_config()
        self.lt = LongTermPersistentMemory(self.config)

    def test_init_empty_table(self) -> None:
        table = self.lt.init_memory_table(2)
        self.assertEqual(table["keys"].shape, (2, self.config.memory_slots, self.config.memory_dim))
        self.assertEqual(self.lt.num_stored(table), 0)
        self.assertFalse(self.lt.is_full(table))

    def test_read_empty_table_is_safe(self) -> None:
        table = self.lt.init_memory_table(2)
        q = torch.randn(2, self.config.d_model)
        out, attn = self.lt.read(q, table, g_read=torch.ones(2), current_step=0)
        self.assertTrue(torch.all(out == 0.0).item())
        self.assertTrue(torch.all(attn == 0.0).item())
        self.assertFalse(torch.isnan(out).any().item())

    def test_write_inserts_and_read_retrieves(self) -> None:
        table = self.lt.init_memory_table(1)
        cand = torch.ones(1, self.config.d_model)
        table = self.lt.write(
            cand, table, g_write=torch.ones(1), current_step=1, threshold=0.1
        )
        self.assertEqual(self.lt.num_stored(table), 1)
        out, attn = self.lt.read(
            cand, table, g_read=torch.ones(1), current_step=2
        )
        self.assertEqual(out.shape, (1, self.config.d_model))

    def test_write_gated_below_threshold_no_insert(self) -> None:
        table = self.lt.init_memory_table(1)
        table = self.lt.write(
            torch.randn(1, self.config.d_model),
            table,
            g_write=torch.zeros(1),
            current_step=1,
            threshold=0.5,
        )
        self.assertEqual(self.lt.num_stored(table), 0)

    def test_full_table_evicts_lowest_decayed_utility(self) -> None:
        table = self.lt.init_memory_table(1)
        for idx in range(self.config.memory_slots):
            table = self.lt.write(
                torch.randn(1, self.config.d_model),
                table,
                g_write=torch.ones(1),
                current_step=idx + 1,
                threshold=0.1,
            )
        self.assertTrue(self.lt.is_full(table))

        # Make slot 0 stale + low utility.
        table["access_times"][0, 0] = 0
        table["utilities"][0, 0] = 0.001

        table = self.lt.write(
            torch.randn(1, self.config.d_model),
            table,
            g_write=torch.ones(1),
            current_step=1000,
            threshold=0.1,
        )
        self.assertEqual(table["access_times"][0, 0].item(), 1000)
        self.assertTrue(self.lt.is_full(table))

    def test_repeated_writes_are_bounded(self) -> None:
        table = self.lt.init_memory_table(3)
        for step in range(200):
            table = self.lt.write(
                torch.randn(3, self.config.d_model),
                table,
                g_write=torch.ones(3),
                current_step=step,
                threshold=0.1,
            )
        self.assertLessEqual(
            self.lt.num_stored(table), 3 * self.config.memory_slots
        )
        self.assertTrue(self.lt.is_full(table))

    def test_update_merges_similar_slot(self) -> None:
        table = self.lt.init_memory_table(1)
        cand = torch.ones(1, self.config.d_model)
        table = self.lt.write(
            cand, table, g_write=torch.ones(1), current_step=1, threshold=0.1
        )
        slots_before = self.lt.num_stored(table)
        # A near-identical candidate should merge (same key direction).
        table, mask = self.lt.update(
            cand,
            table,
            g_update=torch.ones(1),
            current_step=5,
            threshold=0.1,
            similarity_threshold=0.8,
        )
        self.assertTrue(bool(mask[0].item()))
        self.assertEqual(self.lt.num_stored(table), slots_before)  # no new slot
        self.assertEqual(table["access_times"][0, 0].item(), 5)   # timestamp refreshed

    def test_update_skipped_when_dissimilar(self) -> None:
        table = self.lt.init_memory_table(1)
        base = torch.ones(1, self.config.d_model)
        table = self.lt.write(
            base, table, g_write=torch.ones(1), current_step=1, threshold=0.1
        )
        # An orthogonal candidate should NOT merge.
        other = torch.randn(1, self.config.d_model) * 10 + 1.0
        table, mask = self.lt.update(
            other,
            table,
            g_update=torch.ones(1),
            current_step=5,
            threshold=0.1,
            similarity_threshold=0.95,
        )
        self.assertFalse(bool(mask[0].item()))
        self.assertEqual(table["access_times"][0, 0].item(), 1)

    def test_update_skipped_below_gate_threshold(self) -> None:
        table = self.lt.init_memory_table(1)
        cand = torch.ones(1, self.config.d_model)
        table = self.lt.write(
            cand, table, g_write=torch.ones(1), current_step=1, threshold=0.1
        )
        table, mask = self.lt.update(
            cand,
            table,
            g_update=torch.zeros(1),
            current_step=5,
            threshold=0.5,
            similarity_threshold=0.8,
        )
        self.assertFalse(bool(mask[0].item()))

    def test_update_on_empty_table_noop(self) -> None:
        table = self.lt.init_memory_table(1)
        table, mask = self.lt.update(
            torch.ones(1, self.config.d_model),
            table,
            g_update=torch.ones(1),
            current_step=1,
            threshold=0.1,
            similarity_threshold=0.8,
        )
        self.assertFalse(bool(mask[0].item()))
        self.assertEqual(self.lt.num_stored(table), 0)

    def test_write_near_duplicate_merges_instead_of_inserting(self) -> None:
        table = self.lt.init_memory_table(1)
        cand = torch.ones(1, self.config.d_model)
        table = self.lt.write(
            cand, table, g_write=torch.ones(1), current_step=1, threshold=0.1
        )
        # Writing a near-identical candidate must not consume a second slot.
        table = self.lt.write(
            cand + 1e-3,
            table,
            g_write=torch.ones(1),
            current_step=2,
            threshold=0.1,
            similarity_threshold=0.9,
        )
        self.assertEqual(self.lt.num_stored(table), 1)

    def test_forget_gate_evicts_lowest_utility(self) -> None:
        table = self.lt.init_memory_table(2)
        for i in range(3):
            table = self.lt.write(
                torch.randn(2, self.config.d_model),
                table,
                g_write=torch.ones(2),
                current_step=i,
                threshold=0.1,
            )
        before = self.lt.num_stored(table)
        table = self.lt.forget(
            table, g_forget=torch.ones(2), threshold=0.5
        )
        self.assertEqual(self.lt.num_stored(table), before - 2)

    def test_forget_explicit_index(self) -> None:
        table = self.lt.init_memory_table(1)
        table = self.lt.write(
            torch.randn(1, self.config.d_model),
            table,
            g_write=torch.ones(1),
            current_step=1,
            threshold=0.1,
        )
        table = self.lt.forget(
            table,
            g_forget=torch.ones(1),
            slot_index=torch.tensor([0]),
        )
        self.assertEqual(self.lt.num_stored(table), 0)

    def test_forget_invalid_index_raises(self) -> None:
        table = self.lt.init_memory_table(1)
        with self.assertRaises(ValueError):
            self.lt.forget(
                table,
                g_forget=torch.ones(1),
                slot_index=torch.tensor([self.config.memory_slots + 5]),
            )

    def test_forget_empty_table_is_safe(self) -> None:
        table = self.lt.init_memory_table(2)
        table = self.lt.forget(table, g_forget=torch.ones(2), threshold=0.5)
        self.assertEqual(self.lt.num_stored(table), 0)

    def test_invalid_memory_table_raises(self) -> None:
        table = self.lt.init_memory_table(1)
        del table["utilities"]
        with self.assertRaises(ValueError):
            self.lt.read(
                torch.randn(1, self.config.d_model),
                table,
                g_read=torch.ones(1),
            )

    def test_invalid_table_shape_raises(self) -> None:
        table = self.lt.init_memory_table(1)
        table["keys"] = table["keys"][:, : self.config.memory_slots - 1, :]
        with self.assertRaises(ValueError):
            self.lt.write(
                torch.randn(1, self.config.d_model),
                table,
                g_write=torch.ones(1),
                threshold=0.1,
            )

    def test_cosine_similarity_masks_invalid_slots(self) -> None:
        table = self.lt.init_memory_table(1)
        table = self.lt.write(
            torch.ones(1, self.config.d_model),
            table,
            g_write=torch.ones(1),
            current_step=1,
            threshold=0.1,
        )
        sims = self.lt.cosine_similarity(torch.ones(1, self.config.d_model), table)
        self.assertGreater(sims[0, 0].item(), 0.99)
        self.assertEqual(sims[0, 1].item(), -1.0)  # empty slot masked


class TestUtilityGating(unittest.TestCase):
    def setUp(self) -> None:
        self.config = get_tiny_test_config()
        self.gating = MemoryGatingController(self.config)
        self.policy = UtilityGatingPolicy(self.config)

    def _gates(self, read=0.0, write=0.0, update=0.0, forget=0.0):
        b = 1
        return {
            "read": torch.full((b,), read),
            "write": torch.full((b,), write),
            "update": torch.full((b,), update),
            "forget": torch.full((b,), forget),
        }

    def test_gating_outputs_four_probabilities(self) -> None:
        torch.manual_seed(0)
        h = torch.randn(3, self.config.d_model)
        gates = self.gating(h)
        for name in ("read", "write", "update", "forget"):
            self.assertEqual(gates[name].shape, (3,))
            self.assertTrue((gates[name] >= 0.0).all().item())
            self.assertTrue((gates[name] <= 1.0).all().item())

    def test_policy_decides_write(self) -> None:
        gates = self._gates(write=1.0)
        utilities = torch.tensor([0.9])
        max_sim = torch.tensor([-1.0])  # empty table
        out = self.policy.decide(gates, utilities, max_sim)
        self.assertEqual(int(out["decision"][0].item()), WRITE)
        self.assertTrue(bool(out["write_mask"][0].item()))

    def test_policy_decides_update(self) -> None:
        gates = self._gates(update=1.0)
        utilities = torch.tensor([0.1])
        max_sim = torch.tensor([0.95])
        out = self.policy.decide(gates, utilities, max_sim)
        self.assertEqual(int(out["decision"][0].item()), UPDATE)

    def test_policy_decides_forget(self) -> None:
        gates = self._gates(forget=1.0, write=1.0)
        utilities = torch.tensor([0.99])
        max_sim = torch.tensor([0.99])
        out = self.policy.decide(gates, utilities, max_sim)
        self.assertEqual(int(out["decision"][0].item()), FORGET)

    def test_policy_decides_retain(self) -> None:
        gates = self._gates()
        utilities = torch.tensor([0.4])
        max_sim = torch.tensor([-1.0])
        out = self.policy.decide(gates, utilities, max_sim)
        self.assertEqual(int(out["decision"][0].item()), RETAIN)

    def test_policy_priority_forget_over_update_over_write(self) -> None:
        # All gates hot: FORGET must win.
        gates = self._gates(write=1.0, update=1.0, forget=1.0)
        utilities = torch.tensor([0.99])
        max_sim = torch.tensor([0.99])
        out = self.policy.decide(gates, utilities, max_sim)
        self.assertEqual(int(out["decision"][0].item()), FORGET)

        # update beats write when both eligible.
        gates = self._gates(write=1.0, update=1.0, forget=0.0)
        out = self.policy.decide(gates, utilities, max_sim)
        self.assertEqual(int(out["decision"][0].item()), UPDATE)

    def test_policy_is_deterministic(self) -> None:
        gates = self._gates(write=1.0, update=0.9)
        utilities = torch.tensor([0.9])
        max_sim = torch.tensor([0.9])
        d1 = self.policy.decide(gates, utilities, max_sim)["decision"]
        d2 = self.policy.decide(gates, utilities, max_sim)["decision"]
        self.assertTrue(torch.equal(d1, d2))

    def test_policy_thresholds_configurable(self) -> None:
        strict = KhwarizmiConfig(
            **{
                **self.config.to_dict(),
                "utility_threshold": 0.99,
            }
        )
        policy = UtilityGatingPolicy(strict)
        gates = self._gates(write=1.0)
        out = policy.decide(gates, torch.tensor([0.9]), torch.tensor([-1.0]))
        self.assertEqual(int(out["decision"][0].item()), RETAIN)


class TestDualMemoryFacade(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2)
        self.config = get_tiny_test_config()
        self.dm = DualMemory(self.config)

    def test_init_combined_state(self) -> None:
        state = self.dm.init_state(2)
        self.assertIn("short_term", state)
        self.assertIn("long_term", state)
        self.assertEqual(
            state["long_term"]["keys"].shape,
            (2, self.config.memory_slots, self.config.memory_dim),
        )

    def test_forward_shapes(self) -> None:
        state = self.dm.init_state(2)
        h = torch.randn(2, self.config.d_model)
        out = self.dm(h, state, step_counter=1)
        self.assertEqual(out.read_vector.shape, (2, self.config.d_model))
        self.assertEqual(out.decision.shape, (2,))
        self.assertEqual(out.utilities.shape, (2,))
        self.assertEqual(out.max_similarity.shape, (2,))
        for code in out.decision:
            self.assertIn(int(code.item()), (RETAIN, WRITE, UPDATE, FORGET))

    def test_forward_promotes_to_persistent_when_gated(self) -> None:
        """A high-utility candidate with a hot write gate must be WRITTEN."""
        torch.manual_seed(0)
        dm = DualMemory(self.config)
        with torch.no_grad():
            dm.gating.w_write.bias.fill_(5.0)    # write gate -> ~1
            dm.gating.w_update.bias.fill_(-10.0)  # update gate -> ~0
            dm.gating.w_forget.bias.fill_(-10.0)  # forget gate -> ~0
            dm.long_term.util_proj.weight.data.fill_(0.0)
            dm.long_term.util_proj.bias.data.fill_(5.0)  # utility -> ~1
        state = dm.init_state(1)
        h = torch.randn(1, self.config.d_model)
        out = dm(h, state, step_counter=0)
        self.assertEqual(int(out.decision[0].item()), WRITE)
        self.assertGreaterEqual(
            dm.long_term.num_stored(out.state["long_term"]), 1
        )

    def test_forward_bounded_over_long_sequence(self) -> None:
        state = self.dm.init_state(1)
        for step in range(500):
            h = torch.randn(1, self.config.d_model)
            out = self.dm(h, state, step_counter=step)
            state = out.state
            self.assertLessEqual(
                state["short_term"]["window_buffer"].size(1),
                self.config.short_term_capacity,
            )
            self.assertLessEqual(
                self.dm.long_term.num_stored(state["long_term"]),
                self.config.memory_slots,
            )

    def test_explicit_operations_roundtrip_via_long_term(self) -> None:
        state = self.dm.init_state(1)
        h = torch.randn(1, self.config.d_model)
        # WRITE
        state["long_term"] = self.dm.long_term.write(
            h, state["long_term"], g_write=torch.ones(1),
            current_step=1, threshold=0.1,
        )
        self.assertGreaterEqual(
            self.dm.long_term.num_stored(state["long_term"]), 1
        )
        # READ
        out = self.dm.read(h, state, current_step=2)
        self.assertEqual(out.shape, (1, self.config.d_model))
        # UPDATE
        state["long_term"], mask = self.dm.long_term.update(
            h, state["long_term"], g_update=torch.ones(1),
            current_step=3, threshold=0.1,
        )
        # FORGET (explicit index)
        state["long_term"] = self.dm.long_term.forget(
            state["long_term"], g_forget=torch.ones(1),
            slot_index=torch.tensor([0]),
        )
        self.assertEqual(self.dm.long_term.num_stored(state["long_term"]), 0)

    def test_deterministic_given_seed(self) -> None:
        def run():
            torch.manual_seed(7)
            dm = DualMemory(get_tiny_test_config())
            state = dm.init_state(1)
            for step in range(10):
                h = torch.randn(1, dm.config.d_model)
                out = dm(h, state, step_counter=step)
                state = out.state
            return state["long_term"]["keys"].clone(), out.decision.clone()

        k1, d1 = run()
        k2, d2 = run()
        self.assertTrue(torch.equal(k1, k2))
        self.assertTrue(torch.equal(d1, d2))


if __name__ == "__main__":
    unittest.main()
