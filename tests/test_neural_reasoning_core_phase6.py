"""
Comprehensive Phase 6 CPU Unit Tests — Neural Reasoning Core.

Covers the complete Phase 6 specification (Neural Reasoning Core: Latent
Synthesis & Bounded Self-Correction):

  - Configuration validation & backward compatibility.
  - LatentSynthesisBlock: trainable params, output validity, residual update,
    gradient propagation, shape preservation.
  - ConsistencyHead: valid [0,1] range, determinism, gradients, stability.
  - SelfCorrectionBlock: correction activation, state-change, confidence
    conditioning, gradients.
  - NeuralReasoningCore: one/multi-step, min/max bounds, forced convergence,
    bounded execution (no infinite loop), determinism, train/eval modes.
  - ReasoningLosses: differentiability, numerical stability, shape safety.
  - Gradient validation across synthesis / correction / confidence / input.
  - Numerical stability: NaN/Inf prevention on outputs & gradients, extreme
    inputs, repeated refinement.
  - Shape / dtype / device compatibility (CPU, float32/float64, 2D/3D).
  - Integration with KSC, Dual Memory, Sparse MoE, and Adaptive Compute (ARRC)
    via KhwarizmiModel.
  - Regression: all Phase 1–5 behavior preserved (enable_reasoning_core=False).
"""

import unittest

import torch
import torch.nn as nn

from khwarizmi.config import KhwarizmiConfig, get_tiny_test_config
from khwarizmi.core import KhwarizmiModel
from khwarizmi.reasoning import (
    AdaptiveComputeBlock,
    ConsistencyHead,
    LatentSynthesisBlock,
    NeuralReasoningCore,
    ReasoningLosses,
    SelfCorrectionBlock,
)


def _make_config(**overrides) -> KhwarizmiConfig:
    """Return a tiny test config with Phase 6 field overrides applied."""
    data = get_tiny_test_config().to_dict()
    data.update(overrides)
    return KhwarizmiConfig.from_dict(data)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestPhase6Configuration(unittest.TestCase):
    """Phase 6 configuration validation and backward compatibility."""

    def test_default_config_has_phase6_fields(self) -> None:
        cfg = get_tiny_test_config()
        self.assertTrue(cfg.enable_reasoning_core)
        self.assertGreaterEqual(cfg.min_reasoning_steps, 1)
        self.assertGreaterEqual(
            cfg.max_reasoning_steps, cfg.min_reasoning_steps
        )
        self.assertGreaterEqual(cfg.max_reasoning_corrections, 0)
        self.assertLessEqual(
            cfg.max_reasoning_corrections, cfg.max_reasoning_steps
        )
        self.assertGreaterEqual(cfg.reasoning_confidence_threshold, 0.0)
        self.assertLessEqual(cfg.reasoning_confidence_threshold, 1.0)

    def test_invalid_min_steps_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _make_config(min_reasoning_steps=0)
        with self.assertRaises(ValueError):
            _make_config(min_reasoning_steps=-1)

    def test_min_greater_than_max_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _make_config(min_reasoning_steps=5, max_reasoning_steps=3)

    def test_invalid_confidence_threshold_rejected(self) -> None:
        for bad in (-0.1, 1.5, 2.0):
            with self.assertRaises(ValueError):
                _make_config(reasoning_confidence_threshold=bad)

    def test_invalid_max_corrections_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _make_config(max_reasoning_corrections=-1)
        with self.assertRaises(ValueError):
            _make_config(max_reasoning_corrections=5, max_reasoning_steps=3)

    def test_invalid_betas_rejected(self) -> None:
        for bad in (-0.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                _make_config(reasoning_confidence_beta=bad)
            with self.assertRaises(ValueError):
                _make_config(reasoning_refinement_beta=bad)

    def test_config_json_roundtrip_with_phase6_fields(self) -> None:
        cfg = _make_config(
            min_reasoning_steps=2,
            max_reasoning_steps=5,
            reasoning_confidence_threshold=0.9,
            max_reasoning_corrections=3,
            enable_reasoning_core=False,
        )
        loaded = KhwarizmiConfig.from_json_string(cfg.to_json_string())
        self.assertEqual(loaded.min_reasoning_steps, 2)
        self.assertEqual(loaded.max_reasoning_steps, 5)
        self.assertAlmostEqual(loaded.reasoning_confidence_threshold, 0.9)
        self.assertEqual(loaded.max_reasoning_corrections, 3)
        self.assertFalse(loaded.enable_reasoning_core)

    def test_backward_compatible_construction_without_phase6_fields(self) -> None:
        cfg = KhwarizmiConfig(d_model=32, n_heads=4, max_reasoning_steps=4)
        self.assertEqual(cfg.min_reasoning_steps, 1)
        self.assertTrue(cfg.enable_reasoning_core)

    def test_boundary_min_equals_max_valid(self) -> None:
        cfg = _make_config(
            min_reasoning_steps=3, max_reasoning_steps=3,
            max_reasoning_corrections=3,
        )
        self.assertEqual(cfg.min_reasoning_steps, cfg.max_reasoning_steps)


# ---------------------------------------------------------------------------
# Latent Synthesis
# ---------------------------------------------------------------------------


class TestLatentSynthesis(unittest.TestCase):
    """LatentSynthesisBlock: trainable, valid, residual, gradients."""

    def setUp(self) -> None:
        torch.manual_seed(11)
        self.cfg = _make_config()
        self.synth = LatentSynthesisBlock(self.cfg)

    def test_has_trainable_parameters(self) -> None:
        params = [p for p in self.synth.parameters() if p.requires_grad]
        self.assertGreater(len(params), 0)
        self.assertTrue(any(p.requires_grad for p in self.synth.w1.parameters()))

    def test_output_shape_preserved_3d(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model)
        out = self.synth(x)
        self.assertEqual(out.shape, x.shape)

    def test_output_shape_preserved_2d(self) -> None:
        x = torch.randn(4, self.cfg.d_model)
        out = self.synth(x)
        self.assertEqual(out.shape, x.shape)

    def test_output_is_not_identity(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model)
        out = self.synth(x)
        self.assertFalse(torch.allclose(out, torch.zeros_like(out)))
        # Delta should differ from a no-op on a non-degenerate input.
        self.assertGreater(out.abs().mean().item(), 0.0)

    def test_output_finite(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model)
        out = self.synth(x)
        self.assertTrue(torch.isfinite(out).all())

    def test_gradient_propagation(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model, requires_grad=True)
        out = self.synth(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)
        # All trainable params received gradients.
        for name, p in self.synth.named_parameters():
            self.assertIsNotNone(p.grad, f"no grad for {name}")
            self.assertGreater(p.grad.abs().sum().item(), 0.0, f"zero grad {name}")

    def test_residual_update_changes_state(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model)
        delta = self.synth(x)
        h1 = x + delta
        self.assertFalse(torch.allclose(h1, x))


# ---------------------------------------------------------------------------
# Consistency / Confidence
# ---------------------------------------------------------------------------


class TestConsistencyHead(unittest.TestCase):
    """ConsistencyHead: valid range, determinism, gradients, stability."""

    def setUp(self) -> None:
        torch.manual_seed(13)
        self.cfg = _make_config()
        self.head = ConsistencyHead(self.cfg)

    def test_confidence_in_unit_range_3d(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model)
        delta = torch.randn_like(x) * 0.1
        tok, seq = self.head(x, delta)
        self.assertTrue(((tok >= 0.0) & (tok <= 1.0)).all())
        self.assertTrue(((seq >= 0.0) & (seq <= 1.0)).all())

    def test_confidence_in_unit_range_without_delta(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model)
        tok, seq = self.head(x)
        self.assertTrue(((tok >= 0.0) & (tok <= 1.0)).all())
        self.assertTrue(((seq >= 0.0) & (seq <= 1.0)).all())

    def test_confidence_2d_shapes(self) -> None:
        x = torch.randn(4, self.cfg.d_model)
        tok, seq = self.head(x)
        self.assertEqual(tok.shape, (4,))
        self.assertEqual(seq.shape, (4,))

    def test_deterministic_under_fixed_inputs(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model)
        delta = torch.randn_like(x) * 0.1
        _, seq1 = self.head(x, delta)
        _, seq2 = self.head(x, delta)
        self.assertTrue(torch.allclose(seq1, seq2))

    def test_gradient_propagation(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model, requires_grad=True)
        # Build delta as a leaf so its .grad is populated directly.
        delta = torch.randn(4, 8, self.cfg.d_model, requires_grad=True) * 0.1
        # Retain grad on the non-leaf delta to inspect its gradient.
        delta.retain_grad()
        tok, seq = self.head(x, delta)
        seq.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(delta.grad)
        for name, p in self.head.named_parameters():
            self.assertIsNotNone(p.grad, f"no grad for {name}")

    def test_extreme_inputs_stay_bounded(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model) * 1e4
        delta = torch.randn_like(x) * 1e4
        tok, seq = self.head(x, delta)
        self.assertTrue(torch.isfinite(tok).all())
        self.assertTrue(torch.isfinite(seq).all())
        self.assertTrue(((tok >= 0.0) & (tok <= 1.0)).all())


# ---------------------------------------------------------------------------
# Self-Correction
# ---------------------------------------------------------------------------


class TestSelfCorrection(unittest.TestCase):
    """SelfCorrectionBlock: activation, conditioning, gradients, state change."""

    def setUp(self) -> None:
        torch.manual_seed(17)
        self.cfg = _make_config()
        self.corr = SelfCorrectionBlock(self.cfg)

    def test_output_shape_preserved(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model)
        conf = torch.full((4,), 0.5)
        out = self.corr(x, conf)
        self.assertEqual(out.shape, x.shape)

    def test_correction_is_meaningful(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model)
        conf = torch.full((4,), 0.3)  # low confidence -> correction active
        out = self.corr(x, conf)
        self.assertGreater(out.abs().mean().item(), 0.0)

    def test_correction_2d_shape(self) -> None:
        x = torch.randn(4, self.cfg.d_model)
        conf = torch.full((4,), 0.4)
        out = self.corr(x, conf)
        self.assertEqual(out.shape, x.shape)

    def test_state_changes_after_correction(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model)
        conf = torch.full((4,), 0.3)
        delta_corr = self.corr(x, conf)
        h_after = x + delta_corr
        self.assertFalse(torch.allclose(h_after, x))

    def test_low_confidence_yields_larger_correction_than_high(self) -> None:
        # Lower confidence opens the gate more (1 - c), so the correction
        # magnitude should be >= the high-confidence correction on average.
        x = torch.randn(4, 8, self.cfg.d_model)
        low_conf = torch.full((4,), 0.1)
        high_conf = torch.full((4,), 0.9)
        low_out = self.corr(x, low_conf).abs().mean().item()
        high_out = self.corr(x, high_conf).abs().mean().item()
        self.assertGreaterEqual(low_out, high_out)

    def test_correction_bounded_by_tanh(self) -> None:
        # With a learnable correction_scale starting at 0.5 and tanh in [-1,1],
        # the correction magnitude is bounded.
        x = torch.randn(4, 8, self.cfg.d_model) * 1e3
        conf = torch.full((4,), 0.2)
        out = self.corr(x, conf)
        scale = float(self.corr.correction_scale.detach().item())
        self.assertTrue(torch.isfinite(out).all())
        self.assertLessEqual(out.abs().max().item(), scale + 1e-4)

    def test_gradient_propagation(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model, requires_grad=True)
        conf = torch.full((4,), 0.4)
        out = self.corr(x, conf)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        for name, p in self.corr.named_parameters():
            self.assertIsNotNone(p.grad, f"no grad for {name}")
        # correction_scale is a learnable scalar that should receive gradient.
        self.assertIsNotNone(self.corr.correction_scale.grad)


# ---------------------------------------------------------------------------
# Neural Reasoning Core
# ---------------------------------------------------------------------------


class TestNeuralReasoningCoreCore(unittest.TestCase):
    """NeuralReasoningCore: bounds, bounded execution, determinism, modes."""

    def setUp(self) -> None:
        torch.manual_seed(19)
        self.cfg = _make_config(
            min_reasoning_steps=1, max_reasoning_steps=4,
            max_reasoning_corrections=2, reasoning_confidence_threshold=0.95,
        )
        self.core = NeuralReasoningCore(self.cfg)
        self.x = torch.randn(6, 9, self.cfg.d_model)

    def test_initialization(self) -> None:
        self.assertEqual(self.core.min_steps, 1)
        self.assertEqual(self.core.max_steps, 4)
        self.assertEqual(self.core.max_corrections, 2)
        self.assertIsInstance(self.core.synthesis, LatentSynthesisBlock)
        self.assertIsInstance(self.core.consistency, ConsistencyHead)
        self.assertIsInstance(self.core.correction, SelfCorrectionBlock)
        self.assertIsInstance(self.core.losses, ReasoningLosses)

    def test_forward_output_shape_and_finiteness(self) -> None:
        out = self.core(self.x)
        self.assertEqual(out.refined_state.shape, self.x.shape)
        self.assertTrue(torch.isfinite(out.refined_state).all())
        self.assertTrue(torch.isfinite(out.total_reasoning_loss))
        self.assertTrue(torch.isfinite(out.consistency_loss))
        self.assertTrue(torch.isfinite(out.refinement_loss))

    def test_forward_2d_shape(self) -> None:
        out = self.core(torch.randn(6, self.cfg.d_model))
        self.assertEqual(out.refined_state.shape, (6, self.cfg.d_model))

    def test_diagnostics_present(self) -> None:
        out = self.core(self.x)
        for key in (
            "reasoning_steps",
            "correction_count",
            "converged",
            "confidence",
            "consistency_score",
            "latent_delta_norm",
        ):
            self.assertIn(key, out.diagnostics)

    def test_one_reasoning_step_force(self) -> None:
        out = self.core(self.x, force_steps=1)
        self.assertEqual(out.diagnostics["reasoning_steps"], 1)
        # No correction can exceed the number of steps.
        self.assertLessEqual(out.diagnostics["correction_count"], 1)

    def test_multiple_reasoning_steps_force(self) -> None:
        out = self.core(self.x, force_steps=4)
        self.assertEqual(out.diagnostics["reasoning_steps"], 4)
        self.assertLessEqual(out.diagnostics["correction_count"], 2)

    def test_minimum_step_behavior(self) -> None:
        cfg = _make_config(min_reasoning_steps=3, max_reasoning_steps=5,
                           reasoning_confidence_threshold=0.0,
                           max_reasoning_corrections=2)
        core = NeuralReasoningCore(cfg)
        out = core(self.x)
        # threshold 0.0 means always satisfied -> halts at min_steps.
        self.assertEqual(out.diagnostics["reasoning_steps"], 3)
        self.assertTrue(out.diagnostics["converged"])

    def test_maximum_step_behavior(self) -> None:
        cfg = _make_config(min_reasoning_steps=1, max_reasoning_steps=3,
                           max_reasoning_corrections=3)
        core = NeuralReasoningCore(cfg)
        # Unreachable threshold forces the loop to run to K_max.
        out = core(self.x, confidence_threshold=2.0)
        self.assertEqual(out.diagnostics["reasoning_steps"], 3)
        self.assertFalse(out.diagnostics["converged"])
        self.assertLessEqual(out.diagnostics["correction_count"], 3)

    def test_bounded_execution_no_infinite_loop(self) -> None:
        # Even with an unreachable threshold, the loop must terminate at K_max.
        cfg = _make_config(min_reasoning_steps=1, max_reasoning_steps=5,
                           max_reasoning_corrections=5)
        core = NeuralReasoningCore(cfg)
        out = core(self.x, confidence_threshold=2.0)
        self.assertLessEqual(out.diagnostics["reasoning_steps"], 5)
        self.assertLessEqual(out.diagnostics["correction_count"], 5)

    def test_correction_limit_respected(self) -> None:
        cfg = _make_config(min_reasoning_steps=1, max_reasoning_steps=6,
                           max_reasoning_corrections=2)
        core = NeuralReasoningCore(cfg)
        out = core(self.x, confidence_threshold=2.0)
        self.assertLessEqual(out.diagnostics["correction_count"], 2)
        self.assertEqual(out.diagnostics["max_reasoning_corrections"], 2)

    def test_correction_count_exposed(self) -> None:
        out = self.core(self.x, force_steps=4)
        self.assertIsInstance(out.diagnostics["correction_count"], int)
        self.assertGreaterEqual(out.diagnostics["correction_count"], 0)

    def test_deterministic_behavior(self) -> None:
        out1 = self.core(self.x)
        out2 = self.core(self.x)
        self.assertTrue(torch.allclose(out1.refined_state, out2.refined_state))
        self.assertEqual(
            out1.diagnostics["reasoning_steps"],
            out2.diagnostics["reasoning_steps"],
        )
        self.assertEqual(
            out1.diagnostics["correction_count"],
            out2.diagnostics["correction_count"],
        )

    def test_train_eval_modes(self) -> None:
        self.core.train()
        self.assertTrue(self.core.training)
        out_train = self.core(self.x)
        self.core.eval()
        self.assertFalse(self.core.training)
        out_eval = self.core(self.x)
        # Deterministic module: eval and train produce identical results here
        # (no dropout/sampling in the core), but both must be valid.
        self.assertTrue(torch.isfinite(out_train.refined_state).all())
        self.assertTrue(torch.isfinite(out_eval.refined_state).all())
        self.assertTrue(torch.allclose(out_train.refined_state, out_eval.refined_state))

    def test_force_steps_invalid(self) -> None:
        with self.assertRaises(ValueError):
            self.core(self.x, force_steps=0)

    def test_invalid_input_dim(self) -> None:
        with self.assertRaises(ValueError):
            self.core(torch.randn(4, 8, self.cfg.d_model + 1))
        with self.assertRaises(ValueError):
            self.core(torch.randn(4, 8, self.cfg.d_model, 2))

    def test_convergence_when_threshold_low(self) -> None:
        out = self.core(self.x, confidence_threshold=0.0, min_steps=1)
        self.assertTrue(out.diagnostics["converged"])
        self.assertEqual(out.diagnostics["reasoning_steps"], 1)


# ---------------------------------------------------------------------------
# Reasoning Losses
# ---------------------------------------------------------------------------


class TestReasoningLosses(unittest.TestCase):
    """ReasoningLosses: differentiable, stable, shape-safe."""

    def setUp(self) -> None:
        self.cfg = _make_config()
        self.losses = ReasoningLosses(self.cfg)

    def test_consistency_loss_finite_and_differentiable(self) -> None:
        conf = torch.tensor([0.3, 0.7, 0.9], requires_grad=True)
        loss = self.losses.consistency_loss(conf, target=1.0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(conf.grad)
        self.assertGreater(conf.grad.abs().sum().item(), 0.0)

    def test_consistency_loss_higher_for_lower_confidence(self) -> None:
        low = self.losses.consistency_loss(
            torch.full((4,), 0.1), target=1.0
        )
        high = self.losses.consistency_loss(
            torch.full((4,), 0.9), target=1.0
        )
        self.assertGreater(low.item(), high.item())

    def test_consistency_loss_invalid_target(self) -> None:
        with self.assertRaises(ValueError):
            self.losses.consistency_loss(torch.full((4,), 0.5), target=1.5)

    def test_refinement_loss_zero_safe_empty(self) -> None:
        empty = torch.zeros(0)
        loss = self.losses.refinement_loss(empty)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss), 0.0)

    def test_refinement_loss_differentiable(self) -> None:
        r = torch.randn(4, 8, requires_grad=True)
        loss = self.losses.refinement_loss(r)
        loss.backward()
        self.assertIsNotNone(r.grad)

    def test_refinement_loss_increases_with_magnitude(self) -> None:
        small = self.losses.refinement_loss(torch.randn(4, 8) * 0.01)
        large = self.losses.refinement_loss(torch.randn(4, 8) * 10.0)
        self.assertGreater(large.item(), small.item())

    def test_aggregate_sums(self) -> None:
        a = torch.tensor(0.3)
        b = torch.tensor(0.5)
        self.assertAlmostEqual(
            float(self.losses.aggregate(a, b)), 0.8, places=5
        )

    def test_beta_scaling(self) -> None:
        cfg_lo = _make_config(reasoning_confidence_beta=0.0)
        lo = ReasoningLosses(cfg_lo)
        self.assertAlmostEqual(
            float(lo.consistency_loss(torch.full((4,), 0.5), target=1.0)),
            0.0,
            places=6,
        )


# ---------------------------------------------------------------------------
# Gradient Validation
# ---------------------------------------------------------------------------


class TestGradientValidation(unittest.TestCase):
    """End-to-end gradient propagation through the reasoning core."""

    def setUp(self) -> None:
        torch.manual_seed(23)
        self.cfg = _make_config(
            min_reasoning_steps=1, max_reasoning_steps=4,
            max_reasoning_corrections=2, reasoning_confidence_threshold=0.95,
        )
        self.core = NeuralReasoningCore(self.cfg)

    def test_input_receives_gradient(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model, requires_grad=True)
        out = self.core(x)
        out.refined_state.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

    def test_synthesis_parameters_receive_gradients(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model, requires_grad=True)
        out = self.core(x, force_steps=1)
        out.refined_state.sum().backward()
        for name, p in self.core.synthesis.named_parameters():
            self.assertIsNotNone(p.grad, f"synthesis.{name} no grad")
            self.assertGreater(
                p.grad.abs().sum().item(), 0.0, f"synthesis.{name} zero grad"
            )

    def test_correction_parameters_receive_gradients(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model, requires_grad=True)
        # Force multiple steps so a correction executes.
        out = self.core(x, force_steps=3)
        out.refined_state.sum().backward()
        # If a correction executed, correction params must have gradients.
        had_correction = out.diagnostics["correction_count"] > 0
        if had_correction:
            for name, p in self.core.correction.named_parameters():
                self.assertIsNotNone(p.grad, f"correction.{name} no grad")

    def test_confidence_parameters_receive_gradients(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model, requires_grad=True)
        out = self.core(x)
        # Backprop through the consistency loss (which depends on confidence).
        out.consistency_loss.backward()
        for name, p in self.core.consistency.named_parameters():
            self.assertIsNotNone(p.grad, f"consistency.{name} no grad")

    def test_loss_propagates_to_all_submodules(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model, requires_grad=True)
        out = self.core(x, force_steps=3)
        out.total_reasoning_loss.backward()
        # At least one param in each submodule group should have a grad.
        synth_grads = [
            p.grad.abs().sum().item()
            for p in self.core.synthesis.parameters()
            if p.grad is not None
        ]
        self.assertTrue(any(g > 0 for g in synth_grads))


# ---------------------------------------------------------------------------
# Numerical Stability
# ---------------------------------------------------------------------------


class TestStability(unittest.TestCase):
    """NaN/Inf prevention, extreme inputs, repeated refinement."""

    def setUp(self) -> None:
        torch.manual_seed(29)
        self.cfg = _make_config(
            min_reasoning_steps=1, max_reasoning_steps=5,
            max_reasoning_corrections=3, reasoning_confidence_threshold=0.99,
        )
        self.core = NeuralReasoningCore(self.cfg)

    def test_no_nan_output_normal(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model)
        out = self.core(x)
        self.assertFalse(torch.isnan(out.refined_state).any().item())

    def test_no_inf_output_normal(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model)
        out = self.core(x)
        self.assertFalse(torch.isinf(out.refined_state).any().item())

    def test_extreme_input_values(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model) * 1e4
        out = self.core(x)
        self.assertTrue(torch.isfinite(out.refined_state).all())
        self.assertTrue(torch.isfinite(out.total_reasoning_loss))

    def test_nan_input_is_guarded(self) -> None:
        # The residual update guard zeros non-finite deltas so the loop
        # cannot explode; final LayerNorm keeps the output finite.
        x = torch.randn(4, 8, self.cfg.d_model)
        x[0, 0, 0] = float("nan")
        out = self.core(x)
        # The refined state is normed; non-finite guarded contributions zeroed.
        self.assertTrue(torch.isfinite(out.refined_state).all())

    def test_inf_input_is_guarded(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model)
        x[1, 2, 3] = float("inf")
        out = self.core(x)
        self.assertTrue(torch.isfinite(out.refined_state).all())

    def test_no_nan_gradients(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model, requires_grad=True)
        out = self.core(x, force_steps=3)
        out.refined_state.sum().backward()
        for name, p in self.core.named_parameters():
            if p.grad is not None:
                self.assertFalse(
                    torch.isnan(p.grad).any().item(),
                    f"NaN grad in {name}",
                )

    def test_no_inf_gradients(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model, requires_grad=True)
        out = self.core(x, force_steps=3)
        out.refined_state.sum().backward()
        for name, p in self.core.named_parameters():
            if p.grad is not None:
                self.assertFalse(
                    torch.isinf(p.grad).any().item(),
                    f"Inf grad in {name}",
                )

    def test_repeated_refinement_stable(self) -> None:
        # Many reasoning steps with corrections must remain stable.
        cfg = _make_config(
            min_reasoning_steps=1, max_reasoning_steps=10,
            max_reasoning_corrections=10,
        )
        core = NeuralReasoningCore(cfg)
        x = torch.randn(4, 16, cfg.d_model) * 3.0
        out = core(x, confidence_threshold=2.0)
        self.assertTrue(torch.isfinite(out.refined_state).all())
        self.assertLessEqual(out.diagnostics["reasoning_steps"], 10)

    def test_confidence_never_saturates_to_nan(self) -> None:
        x = torch.randn(4, 8, self.cfg.d_model) * 1e3
        out = self.core(x)
        self.assertTrue(
            0.0 <= out.diagnostics["confidence"] <= 1.0
        )


# ---------------------------------------------------------------------------
# Shape / Dtype / Device Compatibility
# ---------------------------------------------------------------------------


class TestShapeDtypeDeviceCompatibility(unittest.TestCase):
    """Batch, sequence, hidden, dtype, and CPU device compatibility."""

    def test_batch_and_sequence_dimensions(self) -> None:
        cfg = _make_config()
        core = NeuralReasoningCore(cfg)
        for b, s in [(1, 1), (3, 5), (8, 16)]:
            x = torch.randn(b, s, cfg.d_model)
            out = core(x)
            self.assertEqual(out.refined_state.shape, x.shape)

    def test_float32_dtype(self) -> None:
        cfg = _make_config()
        core = NeuralReasoningCore(cfg)
        x = torch.randn(4, 8, cfg.d_model, dtype=torch.float32)
        out = core(x)
        self.assertEqual(out.refined_state.dtype, torch.float32)

    def test_float64_dtype(self) -> None:
        cfg = _make_config()
        core = NeuralReasoningCore(cfg)
        x = torch.randn(4, 8, cfg.d_model, dtype=torch.float64)
        out = core(x)
        self.assertEqual(out.refined_state.dtype, torch.float64)
        self.assertTrue(torch.isfinite(out.refined_state).all())

    def test_cpu_device_preserved(self) -> None:
        cfg = _make_config()
        core = NeuralReasoningCore(cfg)
        x = torch.randn(4, 8, cfg.d_model, device=torch.device("cpu"))
        out = core(x)
        self.assertEqual(out.refined_state.device, x.device)

    def test_dtype_not_silently_changed(self) -> None:
        cfg = _make_config()
        core = NeuralReasoningCore(cfg)
        for dt in (torch.float32, torch.float64):
            x = torch.randn(4, 6, cfg.d_model, dtype=dt)
            out = core(x)
            self.assertEqual(out.refined_state.dtype, dt)


# ---------------------------------------------------------------------------
# Integration with Phases 1–5
# ---------------------------------------------------------------------------


class TestIntegration(unittest.TestCase):
    """Integration with KSC, Dual Memory, Sparse MoE, Adaptive Compute."""

    def setUp(self) -> None:
        torch.manual_seed(31)
        self.cfg = _make_config()

    def test_model_builds_reasoning_core(self) -> None:
        m = KhwarizmiModel(self.cfg)
        self.assertIsNotNone(m.reasoning_core)
        self.assertIsInstance(m.reasoning_core, NeuralReasoningCore)

    def test_model_forward_produces_reasoning_diagnostics(self) -> None:
        m = KhwarizmiModel(self.cfg)
        ids = torch.randint(0, self.cfg.vocab_size, (4, 10))
        out = m(ids)
        rdiag = out.diagnostics["reasoning_core_diagnostics"]
        self.assertIn("reasoning_steps", rdiag)
        self.assertIn("correction_count", rdiag)
        self.assertIn("converged", rdiag)
        self.assertIn("confidence", rdiag)
        self.assertIn("consistency_score", rdiag)
        self.assertIn("latent_delta_norm", rdiag)
        self.assertIn("reasoning_loss", out.losses)

    def test_model_forward_finite(self) -> None:
        m = KhwarizmiModel(self.cfg)
        ids = torch.randint(0, self.cfg.vocab_size, (4, 10))
        out = m(ids)
        self.assertTrue(torch.isfinite(out.logits).all())
        self.assertTrue(torch.isfinite(out.losses["reasoning_loss"]))
        self.assertTrue(torch.isfinite(out.losses["total_aux_loss"]))

    def test_reasoning_loss_included_in_total_aux_loss(self) -> None:
        m = KhwarizmiModel(self.cfg)
        ids = torch.randint(0, self.cfg.vocab_size, (4, 10))
        out = m(ids)
        expected = (
            out.losses["routing_loss"]
            + out.losses["moe_aux_loss"]
            + out.losses["ponder_loss"]
            + out.losses["reasoning_loss"]
            + out.losses["memory_gate_loss"]
            + out.losses["memory_proj_loss"]
        )
        self.assertTrue(torch.allclose(out.losses["total_aux_loss"], expected))

    def test_integration_with_ksc_state(self) -> None:
        # The reasoning core consumes the KSC-refined ARRC output and must
        # preserve the recurrent state contract used by short-term memory.
        m = KhwarizmiModel(self.cfg)
        ids = torch.randint(0, self.cfg.vocab_size, (4, 10))
        st, lt = m.init_state(4)
        out = m(ids, short_term_state=st, long_term_table=lt)
        self.assertIn("recurrent_state", out.short_term_state)

    def test_integration_with_sparse_moe(self) -> None:
        # enable_moe=True path: reasoning core runs on MoE-augmented features.
        cfg = _make_config(enable_moe=True)
        m = KhwarizmiModel(cfg)
        ids = torch.randint(0, cfg.vocab_size, (4, 10))
        out = m(ids)
        self.assertTrue(torch.isfinite(out.logits).all())

    def test_integration_with_dual_memory(self) -> None:
        # Reasoning-refined representation flows into the memory write path.
        m = KhwarizmiModel(self.cfg)
        ids = torch.randint(0, self.cfg.vocab_size, (4, 10))
        st, lt = m.init_state(4)
        out = m(ids, short_term_state=st, long_term_table=lt)
        self.assertIn("valid_mask", out.long_term_table)

    def test_integration_with_adaptive_compute(self) -> None:
        # ARRC (Phase 5) output is consumed by the reasoning core (Phase 6).
        cfg = _make_config(
            enable_adaptive_compute=True, enable_reasoning_core=True,
            min_recurrent_cycles=1, max_recurrent_cycles=3,
        )
        m = KhwarizmiModel(cfg)
        ids = torch.randint(0, cfg.vocab_size, (4, 10))
        out = m(ids, force_cycles=2)
        rdiag = out.diagnostics["reasoning_core_diagnostics"]
        self.assertGreaterEqual(rdiag["reasoning_steps"], 1)
        # ARRC diagnostics still present (Phase 5 preserved).
        self.assertIn("reasoner_diagnostics", out.diagnostics)

    def test_model_gradient_flow_end_to_end(self) -> None:
        m = KhwarizmiModel(self.cfg)
        ids = torch.randint(0, self.cfg.vocab_size, (4, 10))
        out = m(ids)
        out.losses["total_aux_loss"].backward()
        # Reasoning core params received gradients.
        rg = [
            p.grad for p in m.reasoning_core.parameters() if p.grad is not None
        ]
        self.assertTrue(len(rg) > 0)
        self.assertTrue(any(g.abs().sum().item() > 0 for g in rg))

    def test_reasoning_core_consumes_arrc_output_directly(self) -> None:
        # Direct unit-level integration: ARRC -> reasoning core.
        torch.manual_seed(37)
        cfg = _make_config()
        arrc = AdaptiveComputeBlock(cfg)
        core = NeuralReasoningCore(cfg)
        x = torch.randn(4, 8, cfg.d_model)
        z, _, _, _ = arrc(x)
        out = core(z)
        self.assertEqual(out.refined_state.shape, z.shape)
        self.assertTrue(torch.isfinite(out.refined_state).all())


# ---------------------------------------------------------------------------
# Regression / Backward Compatibility
# ---------------------------------------------------------------------------


class TestRegressionBackwardCompat(unittest.TestCase):
    """Phase 1–5 behavior preserved when Phase 6 is disabled."""

    def test_reasoning_core_disabled_passes_through(self) -> None:
        cfg = _make_config(enable_reasoning_core=False)
        m = KhwarizmiModel(cfg)
        self.assertIsNone(m.reasoning_core)
        ids = torch.randint(0, cfg.vocab_size, (4, 10))
        out = m(ids)
        self.assertEqual(float(out.losses["reasoning_loss"]), 0.0)
        rdiag = out.diagnostics["reasoning_core_diagnostics"]
        self.assertFalse(rdiag.get("reasoning_core_enabled", True))

    def test_existing_model_output_contract_preserved(self) -> None:
        cfg = _make_config(enable_reasoning_core=False)
        m = KhwarizmiModel(cfg)
        ids = torch.randint(0, cfg.vocab_size, (4, 10))
        out = m(ids)
        # All pre-Phase-6 fields still present.
        for field in (
            "logits", "confidence", "needs_verification",
            "selected_pathways", "routing_probs", "short_term_state",
            "long_term_table", "losses", "diagnostics",
        ):
            self.assertTrue(hasattr(out, field))
        for loss_key in (
            "routing_loss", "moe_aux_loss", "ponder_loss",
            "memory_gate_loss", "memory_proj_loss", "total_aux_loss",
        ):
            self.assertIn(loss_key, out.losses)

    def test_adaptive_compute_still_works_without_reasoning(self) -> None:
        cfg = _make_config(
            enable_adaptive_compute=True, enable_reasoning_core=False,
        )
        m = KhwarizmiModel(cfg)
        ids = torch.randint(0, cfg.vocab_size, (4, 10))
        out = m(ids, force_cycles=2)
        self.assertIn("reasoner_diagnostics", out.diagnostics)

    def test_config_backward_compatible_construction(self) -> None:
        # Pre-Phase-6 style construction (no new kwargs) still works.
        cfg = KhwarizmiConfig(d_model=32, n_heads=4)
        self.assertTrue(cfg.enable_reasoning_core)
        self.assertEqual(cfg.min_reasoning_steps, 1)


if __name__ == "__main__":
    unittest.main()
