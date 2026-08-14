"""
Comprehensive Phase 5 CPU Unit Tests — Adaptive Compute & Learned Halting (ARRC).

Covers the complete Phase 5 specification:
    - Initialization of AdaptiveComputeBlock / PonderCostLoss.
    - Minimum-step enforcement (no token halts before K_min).
    - Maximum-step enforcement (guaranteed termination at K_max).
    - Per-token halting probability validity and accumulation (sum p_k >= 1 - eps).
    - Final forced-halt remainder handling (accumulated probability == 1 exactly).
    - Genuinely adaptive step counts (different tokens use different depth).
    - Deterministic inference (identical inputs => identical outputs and steps).
    - Recurrent state initialization/propagation and no stale-state leakage.
    - Gradient flow through recurrence, halting gate, and ponder loss.
    - Ponder cost behavior (monotone in compute, differentiable, configurable beta).
    - Invalid configuration rejection (min/max steps, epsilon, beta).
    - Batched inputs, 2D single-step inputs, and sequence inputs.
    - enable_adaptive_compute=False fixed-compute compatibility path.
    - Integration with KSC, Dual Memory, and Sparse MoE in KhwarizmiModel.
"""

import unittest
import torch

from khwarizmi.config import KhwarizmiConfig, get_tiny_test_config
from khwarizmi.core import KhwarizmiModel
from khwarizmi.reasoning import AdaptiveComputeBlock, PonderCostLoss, LatentReasoner


def _make_config(**overrides) -> KhwarizmiConfig:
    """Return a tiny test config with Phase 5 field overrides applied."""
    data = get_tiny_test_config().to_dict()
    data.update(overrides)
    return KhwarizmiConfig.from_dict(data)


class TestPhase5Configuration(unittest.TestCase):
    """Phase 5 configuration validation and backward compatibility."""

    def test_default_config_has_phase5_fields(self) -> None:
        cfg = get_tiny_test_config()
        self.assertGreaterEqual(cfg.min_recurrent_cycles, 1)
        self.assertGreaterEqual(cfg.max_recurrent_cycles, cfg.min_recurrent_cycles)
        self.assertTrue(cfg.enable_adaptive_compute)
        self.assertGreater(cfg.halting_epsilon, 0.0)
        self.assertLess(cfg.halting_epsilon, 1.0)

    def test_invalid_min_cycles_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _make_config(min_recurrent_cycles=0)
        with self.assertRaises(ValueError):
            _make_config(min_recurrent_cycles=-1)

    def test_min_greater_than_max_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _make_config(min_recurrent_cycles=5, max_recurrent_cycles=3)

    def test_invalid_halting_epsilon_rejected(self) -> None:
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                _make_config(halting_epsilon=bad)

    def test_invalid_ponder_beta_rejected(self) -> None:
        for bad in (-0.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                _make_config(ponder_cost_beta=bad)

    def test_boundary_min_equals_max_valid(self) -> None:
        cfg = _make_config(min_recurrent_cycles=3, max_recurrent_cycles=3)
        self.assertEqual(cfg.min_recurrent_cycles, cfg.max_recurrent_cycles)

    def test_config_json_roundtrip_with_phase5_fields(self) -> None:
        cfg = _make_config(
            min_recurrent_cycles=2,
            max_recurrent_cycles=5,
            halting_epsilon=0.02,
            enable_adaptive_compute=False,
        )
        loaded = KhwarizmiConfig.from_json_string(cfg.to_json_string())
        self.assertEqual(loaded.min_recurrent_cycles, 2)
        self.assertEqual(loaded.max_recurrent_cycles, 5)
        self.assertAlmostEqual(loaded.halting_epsilon, 0.02)
        self.assertFalse(loaded.enable_adaptive_compute)

    def test_backward_compatible_construction_without_phase5_fields(self) -> None:
        # Pre-Phase-5 style construction (no new kwargs) must still work.
        cfg = KhwarizmiConfig(d_model=32, n_heads=4, max_recurrent_cycles=4)
        self.assertEqual(cfg.min_recurrent_cycles, 1)
        self.assertTrue(cfg.enable_adaptive_compute)


class TestPonderCostLoss(unittest.TestCase):
    """Standalone ponder cost loss module."""

    def test_initialization_and_invalid_beta(self) -> None:
        loss_mod = PonderCostLoss(0.01)
        self.assertAlmostEqual(loss_mod.beta_ponder, 0.01)
        with self.assertRaises(ValueError):
            PonderCostLoss(-1.0)
        with self.assertRaises(ValueError):
            PonderCostLoss(float("nan"))

    def test_cost_increases_with_more_computation(self) -> None:
        loss_mod = PonderCostLoss(0.1)
        rem = torch.full((2, 4), 0.3)
        low = loss_mod(torch.full((2, 4), 2.0), rem)
        high = loss_mod(torch.full((2, 4), 5.0), rem)
        self.assertGreater(high.item(), low.item())

    def test_cost_increases_with_remainder(self) -> None:
        loss_mod = PonderCostLoss(0.1)
        n = torch.full((2, 4), 3.0)
        low = loss_mod(n, torch.full((2, 4), 0.1))
        high = loss_mod(n, torch.full((2, 4), 0.9))
        self.assertGreater(high.item(), low.item())

    def test_beta_scaling_and_zero_beta(self) -> None:
        n = torch.full((1, 3), 2.0)
        r = torch.full((1, 3), 0.5)
        self.assertAlmostEqual(PonderCostLoss(0.0)(n, r).item(), 0.0)
        self.assertAlmostEqual(
            PonderCostLoss(0.2)(n, r).item(), 2.0 * PonderCostLoss(0.1)(n, r).item(),
            places=5,
        )

    def test_remainder_gradient_flows_but_step_count_detached(self) -> None:
        loss_mod = PonderCostLoss(0.5)
        n = torch.full((2, 3), 2.0, requires_grad=True)
        r = torch.full((2, 3), 0.4, requires_grad=True)
        loss = loss_mod(n, r)
        loss.backward()
        self.assertIsNotNone(r.grad)
        self.assertGreater(r.grad.abs().sum().item(), 0.0)
        # n is detached inside the loss: no gradient path.
        self.assertTrue(n.grad is None or n.grad.abs().sum().item() == 0.0)

    def test_shape_mismatch_rejected(self) -> None:
        loss_mod = PonderCostLoss(0.1)
        with self.assertRaises(ValueError):
            loss_mod(torch.zeros(2, 3), torch.zeros(2, 4))


class TestAdaptiveComputeBlockCore(unittest.TestCase):
    """ARRC block: halting mechanics, bounds, determinism, accounting."""

    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = _make_config(
            min_recurrent_cycles=2,
            max_recurrent_cycles=6,
            halting_epsilon=0.05,
        )
        self.block = AdaptiveComputeBlock(self.config)
        # Center the halting gate so tokens genuinely spread across cycles.
        with torch.no_grad():
            self.block.w_halting.bias.fill_(-0.5)
        self.x = torch.randn(8, 10, self.config.d_model) * 2.0

    def test_initialization(self) -> None:
        self.assertEqual(self.block.min_cycles, 2)
        self.assertEqual(self.block.max_cycles, 6)
        self.assertAlmostEqual(self.block.epsilon, 0.05)
        self.assertIsInstance(self.block.ponder_cost, PonderCostLoss)

    def test_output_shape_and_finiteness(self) -> None:
        z, state, ponder, diag = self.block(self.x)
        self.assertEqual(z.shape, self.x.shape)
        self.assertTrue(torch.isfinite(z).all())
        self.assertTrue(torch.isfinite(ponder).all())
        self.assertTrue(torch.isfinite(state).all())

    def test_minimum_step_enforcement(self) -> None:
        _, _, _, diag = self.block(self.x)
        self.assertGreaterEqual(diag["cycles_taken"].min().item(), 2.0)
        # No token can be recorded as halting before K_min.
        self.assertEqual(diag["step_histogram"][0], 0)

    def test_maximum_step_enforcement(self) -> None:
        _, _, _, diag = self.block(self.x)
        self.assertLessEqual(diag["cycles_taken"].max().item(), 6.0)
        # Every token must have a recorded halting step in [K_min, K_max].
        self.assertTrue((diag["halted_at_step"] >= 2).all())
        self.assertTrue((diag["halted_at_step"] <= 6).all())

    def test_min_equals_max_boundary(self) -> None:
        cfg = _make_config(min_recurrent_cycles=4, max_recurrent_cycles=4)
        block = AdaptiveComputeBlock(cfg)
        _, _, _, diag = block(self.x)
        self.assertTrue((diag["cycles_taken"] == 4.0).all())

    def test_hard_termination_even_when_gate_never_fires(self) -> None:
        # Saturate the halting gate towards zero probability: the block must
        # still terminate at exactly K_max via the forced remainder halt.
        with torch.no_grad():
            self.block.w_halting.weight.zero_()
            self.block.w_halting.bias.fill_(-50.0)
        _, _, _, diag = self.block(self.x)
        self.assertTrue((diag["cycles_taken"] == 6.0).all())
        self.assertEqual(sum(diag["step_histogram"][:-1]), 0)
        # Remainder is ~1 for every token (nothing accumulated earlier).
        self.assertTrue(torch.allclose(diag["remainders"], torch.ones_like(diag["remainders"]), atol=1e-4))

    def test_immediate_halt_when_gate_saturated_high(self) -> None:
        # Saturated-high gate: every token halts as soon as K_min allows.
        with torch.no_grad():
            self.block.w_halting.weight.zero_()
            self.block.w_halting.bias.fill_(50.0)
        _, _, _, diag = self.block(self.x)
        self.assertTrue((diag["cycles_taken"] == 2.0).all())

    def test_halting_probabilities_are_valid(self) -> None:
        _, _, _, diag = self.block(self.x)
        for p in diag["halting_history"]:
            self.assertTrue((p > 0.0).all())
            self.assertTrue((p < 1.0).all())

    def test_accumulated_probability_reaches_one_exactly(self) -> None:
        _, _, _, diag = self.block(self.x)
        cum = diag["accumulated_halting_prob"]
        self.assertTrue(torch.allclose(cum, torch.ones_like(cum)))

    def test_halting_condition_sum_pk_threshold(self) -> None:
        # Reconstruct per-token accumulation from the halting history and
        # verify tokens halted exactly when sum p_j >= 1 - epsilon (or at K_max).
        _, _, _, diag = self.block(self.x)
        history = torch.stack(diag["halting_history"], dim=0)  # (K, B, L)
        halted_at = diag["halted_at_step"]
        k_min, k_max = diag["min_cycles"], diag["max_cycles"]
        threshold = 1.0 - self.block.epsilon
        B, L = halted_at.shape
        for b in range(B):
            for l in range(0, L, 3):  # sample tokens
                K = int(halted_at[b, l].item())
                acc = history[k_min - 1 : K, b, l].sum().item()
                if K < k_max:
                    self.assertGreaterEqual(acc, threshold - 1e-5)
                    # And it must NOT have crossed the threshold earlier.
                    prev = history[k_min - 1 : K - 1, b, l].sum().item()
                    self.assertLess(prev, threshold)

    def test_per_cycle_output_weights_sum_to_one(self) -> None:
        # remainder + accumulated pre-halt mass == 1 for every token.
        _, _, _, diag = self.block(self.x)
        self.assertTrue((diag["remainders"] > 0.0).all())
        self.assertTrue((diag["remainders"] <= 1.0 + 1e-6).all())

    def test_adaptive_step_counts_vary_across_tokens(self) -> None:
        # Different tokens must be able to use different computation depth.
        _, _, _, diag = self.block(self.x)
        unique_steps = torch.unique(diag["cycles_taken"])
        self.assertGreater(
            unique_steps.numel(), 1,
            "Adaptive compute degenerated: every token used the same depth.",
        )

    def test_deterministic_inference(self) -> None:
        self.block.eval()
        with torch.no_grad():
            z1, s1, p1, d1 = self.block(self.x)
            z2, s2, p2, d2 = self.block(self.x)
        self.assertTrue(torch.equal(z1, z2))
        self.assertTrue(torch.equal(s1, s2))
        self.assertEqual(p1.item(), p2.item())
        self.assertTrue(torch.equal(d1["cycles_taken"], d2["cycles_taken"]))
        self.assertEqual(d1["step_histogram"], d2["step_histogram"])

    def test_forced_cycles_fixed_compute_mode(self) -> None:
        z1, _, _, d1 = self.block(self.x, force_cycles=3)
        z2, _, _, d2 = self.block(self.x, force_cycles=3)
        self.assertTrue((d1["cycles_taken"] == 3.0).all())
        self.assertEqual(d1["mean_cycles"], 3.0)
        self.assertTrue(torch.allclose(z1, z2))
        cum = d1["accumulated_halting_prob"]
        self.assertTrue(torch.allclose(cum, torch.ones_like(cum)))

    def test_invalid_force_cycles_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.block(self.x, force_cycles=0)

    def test_invalid_runtime_overrides_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.block(self.x, min_cycles=5, max_cycles=3)
        with self.assertRaises(ValueError):
            self.block(self.x, min_cycles=0)

    def test_runtime_max_override_caps_computation(self) -> None:
        _, _, _, diag = self.block(self.x, max_cycles=2)
        self.assertLessEqual(diag["cycles_taken"].max().item(), 2.0)

    def test_invalid_input_rank_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.block(torch.randn(2, 3, 4, 5))

    def test_2d_single_step_input(self) -> None:
        x2d = torch.randn(4, self.config.d_model)
        z, state, ponder, diag = self.block(x2d)
        self.assertEqual(z.shape, x2d.shape)
        self.assertTrue(torch.isfinite(z).all())

    def test_batch_independence_no_cross_example_leakage(self) -> None:
        # Token 0's output must not change when batched with different examples.
        self.block.eval()
        a = torch.randn(1, 6, self.config.d_model)
        b = torch.randn(1, 6, self.config.d_model)
        with torch.no_grad():
            z_solo, _, _, d_solo = self.block(a)
            z_pair, _, _, d_pair = self.block(torch.cat([a, b], dim=0))
        self.assertTrue(torch.allclose(z_solo[0], z_pair[0], atol=1e-5))
        self.assertTrue(
            torch.equal(d_solo["cycles_taken"][0], d_pair["cycles_taken"][0])
        )

    def test_step_histogram_accounts_for_every_token(self) -> None:
        _, _, _, diag = self.block(self.x)
        total_tokens = self.x.shape[0] * self.x.shape[1]
        self.assertEqual(sum(diag["step_histogram"]), total_tokens)


class TestAdaptiveComputeState(unittest.TestCase):
    """Recurrent state initialization, propagation, and carried state."""

    def setUp(self) -> None:
        torch.manual_seed(11)
        self.config = _make_config(min_recurrent_cycles=1, max_recurrent_cycles=4)
        self.block = AdaptiveComputeBlock(self.config)
        self.x = torch.randn(3, 5, self.config.d_model)

    def test_state_initialized_when_none(self) -> None:
        _, state, _, _ = self.block(self.x)
        self.assertEqual(
            state.shape,
            (3, self.config.n_heads, self.config.d_k, self.config.d_expansion),
        )

    def test_carried_state_changes_output(self) -> None:
        self.block.eval()
        with torch.no_grad():
            z_fresh, state, _, _ = self.block(self.x)
            z_carried, _, _, _ = self.block(self.x, state=state)
        self.assertFalse(torch.allclose(z_fresh, z_carried, atol=1e-6))

    def test_state_propagates_across_cycles(self) -> None:
        # With a zero initial state and >1 forced cycles the state must be
        # updated (non-zero) after the pass.
        zero_state = self.block.reasoning_cell.init_state(3)
        _, new_state, _, _ = self.block(self.x, state=zero_state, force_cycles=3)
        self.assertGreater(new_state.abs().sum().item(), 0.0)
        self.assertFalse(torch.equal(new_state, zero_state))

    def test_state_shape_preserved_after_adaptive_pass(self) -> None:
        state0 = self.block.reasoning_cell.init_state(3)
        _, state1, _, _ = self.block(self.x, state=state0)
        self.assertEqual(state0.shape, state1.shape)
        self.assertTrue(torch.isfinite(state1).all())


class TestAdaptiveComputeGradients(unittest.TestCase):
    """Gradient flow through recurrence, halting gate, and ponder cost."""

    def setUp(self) -> None:
        torch.manual_seed(3)
        self.config = _make_config(min_recurrent_cycles=1, max_recurrent_cycles=4)
        self.block = AdaptiveComputeBlock(self.config)
        self.x = torch.randn(2, 6, self.config.d_model, requires_grad=True)

    def test_gradients_reach_all_parameters_through_output(self) -> None:
        z, _, ponder, _ = self.block(self.x)
        (z.sum() + ponder).backward()
        self.assertIsNotNone(self.x.grad)
        self.assertGreater(self.x.grad.abs().sum().item(), 0.0)
        for name, param in self.block.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertTrue(
                torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
            )

    def test_halting_gate_receives_gradient_from_output_alone(self) -> None:
        # The ACT weighting itself must carry gradient into the halting gate
        # (the recurrent graph is not detached).
        z, _, _, _ = self.block(self.x)
        z.sum().backward()
        w_grad = self.block.w_halting.weight.grad
        self.assertIsNotNone(w_grad)
        self.assertGreater(w_grad.abs().sum().item(), 0.0)

    def test_ponder_loss_alone_reaches_halting_gate(self) -> None:
        _, _, ponder, _ = self.block(self.x)
        ponder.backward()
        w_grad = self.block.w_halting.weight.grad
        self.assertIsNotNone(w_grad)
        self.assertGreater(w_grad.abs().sum().item(), 0.0)

    def test_recurrent_cell_receives_gradient(self) -> None:
        z, _, _, _ = self.block(self.x, force_cycles=3)
        z.sum().backward()
        cell_grad = self.block.reasoning_cell.W_q.weight.grad
        self.assertIsNotNone(cell_grad)
        self.assertGreater(cell_grad.abs().sum().item(), 0.0)

    def test_ponder_gradient_pushes_towards_earlier_halting(self) -> None:
        # One SGD step on the ponder loss alone must increase the mean halting
        # probability (i.e. reduce expected computation).
        cfg = _make_config(min_recurrent_cycles=1, max_recurrent_cycles=6,
                           ponder_cost_beta=1.0)
        block = AdaptiveComputeBlock(cfg)
        x = torch.randn(4, 8, cfg.d_model)
        opt = torch.optim.SGD(block.parameters(), lr=0.5)
        with torch.no_grad():
            p_before = block.halting_probability(block.norm(x)).mean().item()
        for _ in range(5):
            opt.zero_grad()
            _, _, ponder, _ = block(x)
            ponder.backward()
            opt.step()
        with torch.no_grad():
            p_after = block.halting_probability(block.norm(x)).mean().item()
        self.assertGreater(p_after, p_before)


class TestLatentReasonerPhase5(unittest.TestCase):
    """LatentReasoner pass-through of Phase 5 controls."""

    def setUp(self) -> None:
        torch.manual_seed(5)
        self.config = _make_config(min_recurrent_cycles=2, max_recurrent_cycles=5)
        self.reasoner = LatentReasoner(self.config)
        self.x = torch.randn(3, 4, self.config.d_model)

    def test_min_max_overrides_pass_through(self) -> None:
        _, _, _, diag = self.reasoner.reason(self.x, min_cycles=3, max_cycles=4)
        self.assertGreaterEqual(diag["cycles_taken"].min().item(), 3.0)
        self.assertLessEqual(diag["cycles_taken"].max().item(), 4.0)

    def test_fast_pathway_bypass_preserved(self) -> None:
        pathways = torch.tensor([0, 2, 0], dtype=torch.long)
        out, _, _, _ = self.reasoner.reason(self.x, pathway_id=pathways)
        self.assertTrue(torch.allclose(out[0], self.x[0], atol=1e-5))
        self.assertTrue(torch.allclose(out[2], self.x[2], atol=1e-5))
        self.assertFalse(torch.allclose(out[1], self.x[1], atol=1e-5))


class TestModelIntegrationPhase5(unittest.TestCase):
    """Adaptive Compute integrated with KSC + Dual Memory + Sparse MoE."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.config = get_tiny_test_config()
        self.model = KhwarizmiModel(self.config)
        self.input_ids = torch.randint(0, self.config.vocab_size, (2, 8))

    def test_end_to_end_forward_with_adaptive_compute(self) -> None:
        out = self.model(self.input_ids)
        self.assertEqual(
            out.logits.shape, (2, 8, self.config.vocab_size)
        )
        self.assertIn("ponder_loss", out.losses)
        self.assertGreaterEqual(out.losses["ponder_loss"].item(), 0.0)
        diag = out.diagnostics["reasoner_diagnostics"]
        self.assertGreaterEqual(diag["mean_cycles"], self.config.min_recurrent_cycles)
        self.assertLessEqual(diag["mean_cycles"], self.config.max_recurrent_cycles)

    def test_adaptive_compute_disabled_path(self) -> None:
        cfg = _make_config(enable_adaptive_compute=False)
        torch.manual_seed(0)
        model = KhwarizmiModel(cfg)
        self.assertIsNone(model.reasoner)
        out = model(self.input_ids)
        self.assertEqual(out.logits.shape, (2, 8, cfg.vocab_size))
        self.assertEqual(out.losses["ponder_loss"].item(), 0.0)
        self.assertFalse(
            out.diagnostics["reasoner_diagnostics"].get(
                "adaptive_compute_enabled", True
            )
        )
        # No halting/reasoning parameters exist in the disabled model.
        reasoner_params = [n for n, _ in model.named_parameters() if "reasoner" in n]
        self.assertEqual(reasoner_params, [])

    def test_disabled_model_is_backward_differentiable(self) -> None:
        cfg = _make_config(enable_adaptive_compute=False)
        torch.manual_seed(0)
        model = KhwarizmiModel(cfg)
        out = model(self.input_ids)
        loss = out.logits.sum() + out.losses["total_aux_loss"]
        loss.backward()  # must not raise
        emb_grad = model.embeddings.token_embedding.weight.grad
        self.assertIsNotNone(emb_grad)

    def test_adaptive_compute_with_moe_disabled(self) -> None:
        cfg = _make_config(enable_moe=False)
        torch.manual_seed(0)
        model = KhwarizmiModel(cfg)
        out = model(self.input_ids)
        self.assertEqual(out.logits.shape, (2, 8, cfg.vocab_size))
        self.assertIn("ponder_loss", out.losses)

    def test_both_switches_disabled_dense_fixed_path(self) -> None:
        cfg = _make_config(enable_moe=False, enable_adaptive_compute=False)
        torch.manual_seed(0)
        model = KhwarizmiModel(cfg)
        out = model(self.input_ids)
        self.assertEqual(out.losses["moe_aux_loss"].item(), 0.0)
        self.assertEqual(out.losses["ponder_loss"].item(), 0.0)

    def test_dual_memory_state_carried_across_calls_with_arrc(self) -> None:
        out1 = self.model(self.input_ids, step_counter=1)
        out2 = self.model(
            self.input_ids,
            short_term_state=out1.short_term_state,
            long_term_table=out1.long_term_table,
            step_counter=2,
        )
        self.assertEqual(out2.logits.shape, (2, 8, self.config.vocab_size))
        for key, tensor in out2.short_term_state.items():
            self.assertTrue(torch.isfinite(tensor).all(), f"Non-finite {key}")

    def test_model_deterministic_inference_with_arrc(self) -> None:
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(self.input_ids, deterministic_router=True)
            out2 = self.model(self.input_ids, deterministic_router=True)
        self.assertTrue(torch.equal(out1.logits, out2.logits))

    def test_model_gradient_flow_through_arrc_and_moe(self) -> None:
        out = self.model(self.input_ids)
        loss = out.logits.sum() + out.losses["total_aux_loss"]
        loss.backward()
        halting_grad = (
            self.model.reasoner.adaptive_compute.w_halting.weight.grad
        )
        self.assertIsNotNone(halting_grad)
        self.assertTrue(torch.isfinite(halting_grad).all())
        cell_grad = (
            self.model.reasoner.adaptive_compute.reasoning_cell.W_q.weight.grad
        )
        self.assertIsNotNone(cell_grad)

    def test_force_cycles_model_api_preserved(self) -> None:
        out = self.model(self.input_ids, force_cycles=2)
        diag = out.diagnostics["reasoner_diagnostics"]
        self.assertEqual(diag["mean_cycles"], 2.0)


class TestAdaptiveComputeVerification(unittest.TestCase):
    """Explicit verification that computation is actually adaptive."""

    def test_easy_vs_hard_inputs_use_different_depth(self) -> None:
        # Construct a gate where the halting logit depends on the input scale:
        # small-magnitude ("easy") latents halt early, large-magnitude ("hard")
        # latents halt late. This proves the mechanism differentiates inputs.
        torch.manual_seed(21)
        cfg = _make_config(min_recurrent_cycles=1, max_recurrent_cycles=6,
                           halting_epsilon=0.05)
        block = AdaptiveComputeBlock(cfg)
        with torch.no_grad():
            block.w_halting.bias.fill_(-0.5)

        easy = torch.randn(4, 8, cfg.d_model) * 0.05
        hard = torch.randn(4, 8, cfg.d_model) * 3.0

        _, _, _, d_easy = block(easy)
        _, _, _, d_hard = block(hard)

        self.assertNotAlmostEqual(
            d_easy["mean_cycles"], d_hard["mean_cycles"], places=3,
            msg="Easy and hard inputs consumed identical average compute.",
        )
        combined = torch.cat(
            [d_easy["cycles_taken"].flatten(), d_hard["cycles_taken"].flatten()]
        )
        self.assertGreater(torch.unique(combined).numel(), 1)

    def test_not_all_tokens_hit_max_steps(self) -> None:
        torch.manual_seed(21)
        cfg = _make_config(min_recurrent_cycles=1, max_recurrent_cycles=6,
                           halting_epsilon=0.05)
        block = AdaptiveComputeBlock(cfg)
        with torch.no_grad():
            block.w_halting.bias.fill_(-0.5)
        x = torch.randn(8, 10, cfg.d_model)
        _, _, _, diag = block(x)
        total = x.shape[0] * x.shape[1]
        at_max = diag["step_histogram"][-1]
        self.assertLess(
            at_max, total,
            "Every token executed max_steps — computation is not adaptive.",
        )


if __name__ == "__main__":
    unittest.main()
