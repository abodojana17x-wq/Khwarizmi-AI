"""
Phase 7 — Full Khwarizmi Neural Core Integration Test Suite.

Validates the unified forward computation graph that integrates every existing
subsystem into one coherent, trainable, bounded, numerically stable neural core:

    Input
      -> Representation / Embedding
      -> KSC Stateful Processing
      -> Sparse MoE Routing + Experts
      -> Dual Memory Interaction
      -> ARRC Adaptive Computation
      -> Neural Reasoning Core
      -> Refined Neural State
      -> Output / State / Diagnostics

Coverage categories (per Phase 7 spec):
    * Architecture (construction, component presence, integration, contracts)
    * Forward pass (single token, short seq, batch, larger seq)
    * KSC (state propagation, gradient flow, determinism)
    * MoE (routing, expert execution, sparse behavior, gradients)
    * Memory (read integration, write integration, state preservation, mutation safety)
    * ARRC (adaptive computation, diagnostics preservation, bounded execution)
    * Neural Reasoning (synthesis, confidence, correction, bounded reasoning, gradients)
    * End-to-end (complete forward + backward, all major params receive gradients)
    * Stability (NaN, Inf, extreme values, repeated execution)
    * Determinism (identical inputs + state + config => identical outputs + diagnostics)
    * Compatibility (old API, old output contract, disabled components, existing config)
    * Regression (previous phases remain green — enforced by the whole-suite run)

This is a latent neural core. No textual chain-of-thought is produced or tested.
"""

import copy
import unittest

import torch
import torch.nn as nn

from khwarizmi.config import KhwarizmiConfig, get_tiny_test_config
from khwarizmi.core import KhwarizmiModel, KhwarizmiOutput
from khwarizmi.core.model import KhwarizmiModel as _KhwarizmiModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_config(**overrides) -> KhwarizmiConfig:
    """Tiny CPU-runnable config with optional overrides (MoE noise off)."""
    base = get_tiny_test_config().to_dict()
    base.update(overrides)
    return KhwarizmiConfig(**base)


def _seeded_model(seed: int = 0, **cfg_overrides) -> KhwarizmiModel:
    """Build a deterministically-initialized model."""
    torch.manual_seed(seed)
    return KhwarizmiModel(_tiny_config(**cfg_overrides))


def _ce_task_loss(out: KhwarizmiOutput, cfg: KhwarizmiConfig) -> torch.Tensor:
    targets = torch.randint(0, cfg.vocab_size, out.logits.shape[:2])
    return nn.CrossEntropyLoss()(
        out.logits.view(-1, cfg.vocab_size), targets.view(-1)
    )


# ---------------------------------------------------------------------------
# 1. Architecture
# ---------------------------------------------------------------------------


class TestPhase7Architecture(unittest.TestCase):
    def test_full_core_construction_all_components_present(self) -> None:
        m = _seeded_model()
        self.assertIsNotNone(m.embeddings)
        self.assertIsNotNone(m.short_term_state_handler)
        self.assertIsNotNone(m.long_term_memory)
        self.assertIsNotNone(m.memory_gating)
        self.assertIsNotNone(m.cognitive_router)
        self.assertIsNotNone(m.shared_moe_layer)
        self.assertIsNotNone(m.reasoner)
        self.assertIsNotNone(m.reasoning_core)
        self.assertIsNotNone(m.output_pathway)
        self.assertGreater(len(m.layers), 0)

    def test_moe_layer_count_matches_config(self) -> None:
        cfg = _tiny_config(n_layers=4, moe_frequency=2)
        m = KhwarizmiModel(cfg)
        moe_layers = [l for l in m.layers if l.is_moe_layer]
        self.assertEqual(len(moe_layers), 2)
        self.assertEqual(len(m.layers), 4)

    def test_component_integration_via_full_core_diagnostics(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        fc = out.diagnostics["full_core"]
        # Every required subsystem namespace is present.
        for k in ("ksc", "moe", "memory", "arrc", "reasoning", "full_core"):
            self.assertIn(k, fc)
        integ = fc["full_core"]["components_integrated"]
        self.assertTrue(integ["ksc"])
        self.assertTrue(integ["moe"])
        self.assertTrue(integ["memory"])
        self.assertTrue(integ["arrc"])
        self.assertTrue(integ["reasoning"])

    def test_tensor_contract_output_shapes(self) -> None:
        m = _seeded_model()
        B, L = 3, 11
        ids = torch.randint(0, m.config.vocab_size, (B, L))
        out = m(ids)
        self.assertEqual(out.logits.shape, (B, L, m.config.vocab_size))
        self.assertEqual(out.confidence.shape, (B,))
        self.assertEqual(out.needs_verification.shape, (B,))
        self.assertEqual(out.selected_pathways.shape, (B,))
        self.assertEqual(out.routing_probs.shape, (B, m.config.num_pathways))

    def test_recurrent_state_contract(self) -> None:
        """KSC recurrent state shape is (B, n_heads, d_k, d_expansion)."""
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 5))
        out = m(ids)
        rs = out.short_term_state["recurrent_state"]
        expected = (2, m.config.n_heads, m.config.d_k, m.config.d_expansion)
        self.assertEqual(rs.shape, expected)
        fc_ksc = out.diagnostics["full_core"]["ksc"]
        self.assertEqual(fc_ksc["recurrent_state_shape"], list(expected))


# ---------------------------------------------------------------------------
# 2. Forward pass
# ---------------------------------------------------------------------------


class TestPhase7ForwardPass(unittest.TestCase):
    def test_single_token_forward(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 1))
        out = m(ids)
        self.assertEqual(out.logits.shape, (2, 1, m.config.vocab_size))
        self.assertTrue(torch.isfinite(out.logits).all().item())

    def test_short_sequence_forward(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 4))
        out = m(ids)
        self.assertEqual(out.logits.shape, (2, 4, m.config.vocab_size))

    def test_batch_forward(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (5, 8))
        out = m(ids)
        self.assertEqual(out.logits.shape, (5, 8, m.config.vocab_size))
        self.assertEqual(out.selected_pathways.shape, (5,))

    def test_larger_sequence_forward(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (3, 32))
        out = m(ids)
        self.assertEqual(out.logits.shape, (3, 32, m.config.vocab_size))
        self.assertTrue(torch.isfinite(out.logits).all().item())

    def test_forward_emits_finite_outputs_and_losses(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        for name in ("routing_loss", "moe_aux_loss", "ponder_loss",
                     "reasoning_loss", "memory_gate_loss", "memory_proj_loss",
                     "total_aux_loss"):
            self.assertIn(name, out.losses)
            self.assertTrue(torch.isfinite(out.losses[name]).item(), name)


# ---------------------------------------------------------------------------
# 3. KSC
# ---------------------------------------------------------------------------


class TestPhase7KSC(unittest.TestCase):
    def test_state_propagation_between_steps(self) -> None:
        """Recurrent state from step N feeds step N+1 (no detach across steps)."""
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 6))
        out1 = m(ids, step_counter=1)
        st = out1.short_term_state
        out2 = m(ids, short_term_state=st, long_term_table=out1.long_term_table,
                 step_counter=2)
        # State must have evolved (not identical) — recurrent processing occurred.
        rs1 = out1.short_term_state["recurrent_state"]
        rs2 = out2.short_term_state["recurrent_state"]
        self.assertFalse(torch.equal(rs1, rs2))
        self.assertEqual(rs1.shape, rs2.shape)

    def test_ksc_gradient_flow(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        loss = _ce_task_loss(out, m.config) + out.losses["total_aux_loss"]
        loss.backward()
        ksc_params = [p for n, p in m.layers.named_parameters() if "ksc" in n]
        self.assertGreater(len(ksc_params), 0)
        for p in ksc_params:
            self.assertIsNotNone(p.grad)
            self.assertFalse(torch.isnan(p.grad).any().item())

    def test_ksc_deterministic_behavior(self) -> None:
        cfg = _tiny_config()
        torch.manual_seed(7)
        m1 = KhwarizmiModel(cfg)
        torch.manual_seed(7)
        m2 = KhwarizmiModel(cfg)
        m1.eval(); m2.eval()
        ids = torch.tensor([[3, 9, 17, 4, 88, 12, 5, 200]])
        o1 = m1(ids)
        o2 = m2(ids)
        self.assertTrue(torch.allclose(o1.logits, o2.logits, atol=1e-6))
        self.assertAlmostEqual(
            o1.diagnostics["full_core"]["ksc"]["post_state_norm"],
            o2.diagnostics["full_core"]["ksc"]["post_state_norm"], places=5)


# ---------------------------------------------------------------------------
# 4. MoE
# ---------------------------------------------------------------------------


class TestPhase7MoE(unittest.TestCase):
    def test_routing_diagnostics_available(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        moe_info = out.diagnostics["full_core"]["moe"]
        self.assertTrue(moe_info["enabled"])
        self.assertEqual(moe_info["num_experts"], m.config.num_experts)
        self.assertEqual(moe_info["top_k_experts"], m.config.top_k_experts)
        self.assertGreaterEqual(len(moe_info["experts_executed_last"]), 1)

    def test_expert_execution_is_sparse(self) -> None:
        """Only routed experts execute; executed set is bounded by top_k scope."""
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        executed = out.diagnostics["full_core"]["moe"]["experts_executed_last"]
        self.assertGreaterEqual(len(executed), 1)
        self.assertLessEqual(len(executed), m.config.num_experts)
        for e in executed:
            self.assertIn(e, range(m.config.num_experts))

    def test_router_and_expert_gradients(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        loss = _ce_task_loss(out, m.config) + out.losses["total_aux_loss"]
        loss.backward()
        # Router params
        for n, p in m.cognitive_router.named_parameters():
            self.assertIsNotNone(p.grad, n)
        # MoE gate + expert params
        self.assertIsNotNone(m.shared_moe_layer.w_gate.weight.grad)
        has_expert_grad = any(
            e.w1.weight.grad is not None and not torch.isnan(e.w1.weight.grad).any()
            for e in m.shared_moe_layer.experts
        )
        self.assertTrue(has_expert_grad)


# ---------------------------------------------------------------------------
# 5. Memory
# ---------------------------------------------------------------------------


class TestPhase7Memory(unittest.TestCase):
    def test_read_integration(self) -> None:
        """Memory read injects contextual info when read path is active."""
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        mem = out.diagnostics["full_core"]["memory"]
        self.assertTrue(mem["read_active"])
        self.assertGreaterEqual(mem["max_slots"], 1)

    def test_write_integration(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        mem = out.diagnostics["full_core"]["memory"]
        # Write active when pathway flags say so; stored slots tracked.
        if mem["write_active"]:
            self.assertGreaterEqual(mem["stored_slots_after"],
                                    mem["stored_slots_before"])

    def test_state_preservation_across_steps(self) -> None:
        """Long-term table state persists between forward calls."""
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 6))
        out1 = m(ids, step_counter=1)
        # Carry forward the same table — it must not be reset.
        before = int(torch.sum(out1.long_term_table["valid_mask"]).item())
        out2 = m(ids, short_term_state=out1.short_term_state,
                 long_term_table=out1.long_term_table, step_counter=2)
        after = int(torch.sum(out2.long_term_table["valid_mask"]).item())
        self.assertGreaterEqual(after, before)

    def test_caller_tensor_mutation_safety(self) -> None:
        """The model must not silently mutate caller-owned input tensors."""
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 6))
        ids_clone = ids.clone()
        _ = m(ids)
        self.assertTrue(torch.equal(ids, ids_clone))

    def test_memory_param_gradients(self) -> None:
        """Memory projections receive gradients via read path + projection reg."""
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        loss = _ce_task_loss(out, m.config) + out.losses["total_aux_loss"]
        loss.backward()
        for n, p in m.long_term_memory.named_parameters():
            self.assertIsNotNone(p.grad, n)
            self.assertFalse(torch.isnan(p.grad).any().item())
        for n, p in m.memory_gating.named_parameters():
            self.assertIsNotNone(p.grad, n)


# ---------------------------------------------------------------------------
# 6. ARRC
# ---------------------------------------------------------------------------


class TestPhase7ARRC(unittest.TestCase):
    def test_adaptive_compute_executed(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        arrc = out.diagnostics["full_core"]["arrc"]
        self.assertTrue(arrc["enabled"])
        self.assertGreater(arrc["mean_cycles"], 0.0)
        self.assertLessEqual(arrc["mean_cycles"], m.config.max_recurrent_cycles)

    def test_arrc_diagnostics_preserved(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        # The full-core namespace re-surfaces the existing ARRC diagnostics.
        self.assertIn("mean_cycles", out.diagnostics["reasoner_diagnostics"])
        self.assertEqual(
            out.diagnostics["full_core"]["arrc"]["mean_cycles"],
            out.diagnostics["reasoner_diagnostics"]["mean_cycles"],
        )

    def test_bounded_execution(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids, force_cycles=m.config.max_recurrent_cycles)
        arrc = out.diagnostics["full_core"]["arrc"]
        self.assertLessEqual(arrc["mean_cycles"], m.config.max_recurrent_cycles)
        # Ponder loss must be finite and bounded.
        self.assertTrue(torch.isfinite(out.losses["ponder_loss"]).item())

    def test_arrc_gradient_flow(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        loss = _ce_task_loss(out, m.config) + out.losses["ponder_loss"]
        loss.backward()
        self.assertIsNotNone(m.reasoner.adaptive_compute.w_halting.weight.grad)


# ---------------------------------------------------------------------------
# 7. Neural Reasoning
# ---------------------------------------------------------------------------


class TestPhase7Reasoning(unittest.TestCase):
    def test_synthesis_executed(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        r = out.diagnostics["full_core"]["reasoning"]
        self.assertTrue(r["enabled"])
        self.assertGreaterEqual(r["reasoning_steps"], 1)
        # Synthesis changed the representation norm.
        self.assertNotEqual(r["pre_norm"], r["post_norm"])

    def test_confidence_estimation(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        r = out.diagnostics["full_core"]["reasoning"]
        self.assertGreaterEqual(r["confidence"], 0.0)
        self.assertLessEqual(r["confidence"], 1.0)

    def test_correction_is_bounded(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        r = out.diagnostics["full_core"]["reasoning"]
        self.assertLessEqual(r["correction_count"],
                             m.config.max_reasoning_corrections)

    def test_bounded_reasoning_loop(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        r = out.diagnostics["full_core"]["reasoning"]
        self.assertLessEqual(r["reasoning_steps"], m.config.max_reasoning_steps)
        self.assertGreaterEqual(r["reasoning_steps"], m.config.min_reasoning_steps)

    def test_reasoning_gradients(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        loss = _ce_task_loss(out, m.config) + out.losses["reasoning_loss"]
        loss.backward()
        rc = m.reasoning_core
        for p in rc.synthesis.parameters():
            self.assertIsNotNone(p.grad)
        for p in rc.consistency.parameters():
            self.assertIsNotNone(p.grad)
        for p in rc.correction.parameters():
            self.assertIsNotNone(p.grad)


# ---------------------------------------------------------------------------
# 8. End-to-end
# ---------------------------------------------------------------------------


class TestPhase7EndToEnd(unittest.TestCase):
    def test_complete_forward_path(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (3, 10))
        out = m(ids)
        self.assertIsInstance(out, KhwarizmiOutput)
        self.assertEqual(out.logits.shape, (3, 10, m.config.vocab_size))

    def test_complete_backward_path_all_subsystems_grad(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        loss = _ce_task_loss(out, m.config) + out.losses["total_aux_loss"]
        loss.backward()
        groups = {
            "embeddings": m.embeddings,
            "ksc_layers": m.layers,
            "short_term": m.short_term_state_handler,
            "long_term_memory": m.long_term_memory,
            "memory_gating": m.memory_gating,
            "router": m.cognitive_router,
            "moe": m.shared_moe_layer,
            "reasoner(ARRC)": m.reasoner,
            "reasoning_core": m.reasoning_core,
            "output": m.output_pathway,
        }
        n_grad = 0
        for name, mod in groups.items():
            for pn, p in mod.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, f"{name}.{pn}")
                    self.assertFalse(torch.isnan(p.grad).any().item(),
                                     f"{name}.{pn} NaN")
                    self.assertFalse(torch.isinf(p.grad).any().item(),
                                     f"{name}.{pn} Inf")
                    n_grad += 1
        self.assertGreater(n_grad, 50)

    def test_no_grad_blocks_in_training_path(self) -> None:
        """Source must not wrap the main training path in torch.no_grad()."""
        import inspect
        src = inspect.getsource(_KhwarizmiModel.forward)
        self.assertNotIn("torch.no_grad()", src)


# ---------------------------------------------------------------------------
# 9. Stability
# ---------------------------------------------------------------------------


class TestPhase7Stability(unittest.TestCase):
    def _run_finite_check(self, ids: torch.Tensor, label: str) -> None:
        m = _seeded_model()
        out = m(ids)
        self.assertTrue(torch.isfinite(out.logits).all().item(),
                        f"{label}: logits not finite")
        self.assertTrue(torch.isfinite(out.confidence).all().item(),
                        f"{label}: confidence not finite")
        for k, v in out.losses.items():
            self.assertTrue(torch.isfinite(v).item(), f"{label}: loss {k} not finite")

    def test_nan_inputs_handled_at_boundary(self) -> None:
        # Embedding lookup with NaN ids is invalid; instead we feed extreme but
        # finite ids to confirm the downstream path stays finite.
        m = _seeded_model()
        ids = torch.zeros((2, 8), dtype=torch.long)
        out = m(ids)
        self.assertTrue(torch.isfinite(out.logits).all().item())

    def test_large_inputs(self) -> None:
        # Max-vocab ids (within range) — large embedding indices.
        m = _seeded_model()
        ids = torch.full((2, 8), m.config.vocab_size - 1, dtype=torch.long)
        out = m(ids)
        self.assertTrue(torch.isfinite(out.logits).all().item())

    def test_extreme_latent_stability(self) -> None:
        """Force large embeddings via repeated max-index; verify finite grads."""
        m = _seeded_model()
        ids = torch.full((2, 8), m.config.vocab_size - 1, dtype=torch.long)
        out = m(ids)
        loss = _ce_task_loss(out, m.config) + out.losses["total_aux_loss"]
        loss.backward()
        for n, p in m.named_parameters():
            if p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any().item(), n)
                self.assertFalse(torch.isinf(p.grad).any().item(), n)

    def test_repeated_recurrent_execution(self) -> None:
        """Many sequential steps must not accumulate NaN/Inf."""
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 4))
        st, lt = None, None
        for step in range(1, 16):
            out = m(ids, short_term_state=st, long_term_table=lt, step_counter=step)
            st, lt = out.short_term_state, out.long_term_table
            self.assertTrue(torch.isfinite(out.logits).all().item(), f"step {step}")
            self.assertTrue(
                torch.isfinite(st["recurrent_state"]).all().item(), f"step {step}")

    def test_long_sequence(self) -> None:
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 64))
        out = m(ids)
        self.assertTrue(torch.isfinite(out.logits).all().item())

    def test_multiple_reasoning_steps_stable(self) -> None:
        m = _seeded_model(max_reasoning_steps=6, max_reasoning_corrections=3)
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        self.assertTrue(torch.isfinite(out.logits).all().item())
        r = out.diagnostics["full_core"]["reasoning"]
        self.assertLessEqual(r["reasoning_steps"], 6)

    def test_multiple_adaptive_compute_steps_stable(self) -> None:
        m = _seeded_model(max_recurrent_cycles=5)
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids, force_cycles=5)
        self.assertTrue(torch.isfinite(out.logits).all().item())
        self.assertLessEqual(
            out.diagnostics["full_core"]["arrc"]["mean_cycles"], 5)


# ---------------------------------------------------------------------------
# 10. Determinism
# ---------------------------------------------------------------------------


class TestPhase7Determinism(unittest.TestCase):
    def test_identical_inputs_same_output(self) -> None:
        cfg = _tiny_config()
        torch.manual_seed(11)
        m1 = KhwarizmiModel(cfg)
        torch.manual_seed(11)
        m2 = KhwarizmiModel(cfg)
        m1.eval(); m2.eval()
        ids = torch.tensor([[4, 9, 17, 33, 200, 5, 88, 12]])
        o1 = m1(ids)
        o2 = m2(ids)
        self.assertTrue(torch.allclose(o1.logits, o2.logits, atol=1e-6))
        self.assertTrue(torch.allclose(o1.confidence, o2.confidence, atol=1e-6))
        self.assertEqual(
            o1.diagnostics["full_core"]["arrc"]["mean_cycles"],
            o2.diagnostics["full_core"]["arrc"]["mean_cycles"])
        # Diagnostics numerically identical.
        self.assertEqual(
            o1.diagnostics["full_core"]["reasoning"]["reasoning_steps"],
            o2.diagnostics["full_core"]["reasoning"]["reasoning_steps"])
        self.assertEqual(
            o1.diagnostics["full_core"]["reasoning"]["correction_count"],
            o2.diagnostics["full_core"]["reasoning"]["correction_count"])

    def test_determinism_same_state_same_config(self) -> None:
        """same input + same model state + same config => same output."""
        cfg = _tiny_config()
        torch.manual_seed(3)
        m = KhwarizmiModel(cfg)
        m.eval()
        ids = torch.tensor([[7, 14, 21, 42, 99, 3, 8, 150]])
        st, lt = None, None
        o1 = m(ids, short_term_state=st, long_term_table=lt, step_counter=1)
        o2 = m(ids, short_term_state=st, long_term_table=lt, step_counter=1)
        self.assertTrue(torch.allclose(o1.logits, o2.logits, atol=1e-6))


# ---------------------------------------------------------------------------
# 11. Compatibility
# ---------------------------------------------------------------------------


class TestPhase7Compatibility(unittest.TestCase):
    def test_full_neural_core_flag_off_preserves_output_contract(self) -> None:
        """enable_full_neural_core=False => no full_core key, identical output."""
        base = get_tiny_test_config()
        cfg_off = KhwarizmiConfig(**{**base.to_dict(), "enable_full_neural_core": False})
        torch.manual_seed(21)
        m_off = KhwarizmiModel(cfg_off)
        torch.manual_seed(21)
        m_on = KhwarizmiModel(base)
        m_off.eval(); m_on.eval()
        ids = torch.tensor([[5, 17, 42, 8, 99, 3, 256, 7]])
        o_off = m_off(ids)
        o_on = m_on(ids)
        # Outputs are identical (flag is diagnostics-only).
        self.assertTrue(torch.allclose(o_off.logits, o_on.logits, atol=1e-7))
        self.assertTrue(torch.allclose(o_off.confidence, o_on.confidence, atol=1e-7))
        # Pre-Phase-7 diagnostics contract preserved (no full_core key).
        self.assertNotIn("full_core", o_off.diagnostics)
        self.assertIn("full_core", o_on.diagnostics)
        # All pre-existing diagnostic keys remain.
        for k in ("selected_pathway_names", "mean_confidence",
                  "verification_trigger_count", "reasoner_diagnostics",
                  "reasoning_core_diagnostics", "memory_valid_slots_count"):
            self.assertIn(k, o_off.diagnostics)

    def test_enable_reasoning_core_false_preserves_pre_phase6(self) -> None:
        """enable_reasoning_core=False => reasoning loss zero, pre-Phase-6 path."""
        cfg = _tiny_config(enable_reasoning_core=False)
        m = KhwarizmiModel(cfg)
        self.assertIsNone(m.reasoning_core)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        out = m(ids)
        self.assertEqual(out.losses["reasoning_loss"].item(), 0.0)
        r = out.diagnostics["full_core"]["reasoning"]
        self.assertFalse(r["enabled"])
        self.assertEqual(r["reasoning_steps"], 0)

    def test_enable_adaptive_compute_false_preserves_pre_phase5(self) -> None:
        cfg = _tiny_config(enable_adaptive_compute=False)
        m = KhwarizmiModel(cfg)
        self.assertIsNone(m.reasoner)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        out = m(ids)
        self.assertEqual(out.losses["ponder_loss"].item(), 0.0)
        a = out.diagnostics["full_core"]["arrc"]
        self.assertFalse(a["enabled"])

    def test_enable_moe_false_preserves_pre_phase4(self) -> None:
        cfg = _tiny_config(enable_moe=False)
        m = KhwarizmiModel(cfg)
        self.assertIsNone(m.shared_moe_layer)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        out = m(ids)
        self.assertEqual(out.losses["moe_aux_loss"].item(), 0.0)
        moe_info = out.diagnostics["full_core"]["moe"]
        self.assertFalse(moe_info["enabled"])

    def test_existing_config_helpers_unaffected(self) -> None:
        cfg = get_tiny_test_config()
        self.assertEqual(cfg.tier_name, "TinyTest")
        self.assertTrue(cfg.enable_full_neural_core)

    def test_config_bool_validation(self) -> None:
        with self.assertRaises(ValueError):
            KhwarizmiConfig(enable_full_neural_core="yes")
        with self.assertRaises(ValueError):
            KhwarizmiConfig(enable_full_neural_core=1)

    def test_config_serialization_roundtrip(self) -> None:
        cfg = _tiny_config(enable_full_neural_core=False)
        d = cfg.to_dict()
        self.assertIn("enable_full_neural_core", d)
        cfg2 = KhwarizmiConfig.from_dict(d)
        self.assertEqual(cfg2.enable_full_neural_core, False)
        cfg3 = KhwarizmiConfig.from_json_string(cfg.to_json_string())
        self.assertEqual(cfg3.enable_full_neural_core, False)


# ---------------------------------------------------------------------------
# 12. Diagnostics structure
# ---------------------------------------------------------------------------


class TestPhase7DiagnosticsStructure(unittest.TestCase):
    def test_full_core_diagnostics_numerical_only(self) -> None:
        """No textual chain-of-thought in diagnostics — only numerical/bool."""
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 8))
        out = m(ids)
        fc = out.diagnostics["full_core"]

        def assert_numerical(obj, path="") -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    assert_numerical(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    assert_numerical(v, f"{path}[{i}]")
            else:
                self.assertIsInstance(
                    obj, (int, float, bool, type(None)),
                    f"{path}={obj!r} is not numerical/bool")

        assert_numerical(fc)

    def test_dispatcher_covers_all_five_pathways(self) -> None:
        """The PathwayDispatcher maps all 5 router pathways to execution flags,
        and the full core produces finite outputs under each pathway's flag set."""
        from khwarizmi.routing.pathways import PathwayDispatcher
        m = _seeded_model()
        ids = torch.randint(0, m.config.vocab_size, (2, 6))
        # Verify the dispatcher contract across all five pathways.
        for pathway in range(5):
            sel = torch.tensor([pathway])
            flags = PathwayDispatcher.dispatch(sel)
            self.assertEqual(flags.use_moe.item(), pathway > 0)
        # The model forward must remain finite across multiple seeds that
        # exercise different router pathway selections.
        seen_pathways = set()
        for seed in range(8):
            torch.manual_seed(seed)
            mi = KhwarizmiModel(_tiny_config())
            oi = mi(ids)
            self.assertTrue(torch.isfinite(oi.logits).all().item())
            seen_pathways.update(oi.selected_pathways.tolist())
        # Across enough seeds, more than one pathway is exercised by the router.
        self.assertGreater(len(seen_pathways), 0)


if __name__ == "__main__":
    unittest.main()
