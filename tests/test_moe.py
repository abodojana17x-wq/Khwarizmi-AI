"""
Phase 4 Unit Tests — Sparse Mixture-of-Experts (MoE).

Comprehensive CPU tests for the Phase 4 Sparse Top-K Noisy-Gated MoE layer
(`khwarizmi/experts/moe_layer.py`) and its integration with the Khwarizmi
neural core, per ROADMAP Phase 4:

    - Expert initialization (independent parameters, configurable dims).
    - Router output: clean/noisy logits, noise behavior, determinism.
    - Top-K selection validity and normalized routing weights.
    - Genuinely sparse expert execution (unselected experts never evaluated).
    - Output equivalence with a manual Top-K reference computation.
    - Gradient flow: selected experts, unselected experts, router weights,
      routing weights, noise projection, differentiable auxiliary loss.
    - Load-balancing loss: closed-form values, balance vs. collapse,
      alpha scaling, bounds, dispatch fractions.
    - Batch and sequence inputs (2D and 3D).
    - Invalid configurations and invalid input shapes.
    - Integration with KhwarizmiModel (MoE enabled/disabled), compatibility
      with the Phase 2 KSC prototype and Phase 3 Dual Memory, and
      MoE-disabled regression behavior.

The test module snapshots/restores the global torch RNG around every test so
it cannot perturb the seed-sensitive expectations of other test modules.
"""

import unittest

import torch

from khwarizmi.config import KhwarizmiConfig, get_tiny_test_config
from khwarizmi.core import KhwarizmiModel, KhwarizmiKSCPrototype
from khwarizmi.experts import (
    SparseMoELayer,
    ExpertLayer,
    MoERoutingDecision,
    create_standard_specialists,
    SPECIALIZATION_NAMES,
)


class RngNeutralTestCase(unittest.TestCase):
    """Restore the global torch RNG after each test (RNG-neutral suite)."""

    def setUp(self) -> None:
        self._rng_state = torch.random.get_rng_state()

    def tearDown(self) -> None:
        torch.random.set_rng_state(self._rng_state)


# --------------------------------------------------------------------------
# Experts
# --------------------------------------------------------------------------
class TestExpertLayer(RngNeutralTestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0)
        self.config = get_tiny_test_config()

    def test_expert_initialization_dimensions(self) -> None:
        expert = ExpertLayer(self.config, specialization_name="Python_Coding")
        self.assertEqual(expert.w1.in_features, self.config.d_model)
        self.assertEqual(expert.w1.out_features, self.config.d_ff)
        self.assertEqual(expert.w2.in_features, self.config.d_ff)
        self.assertEqual(expert.w2.out_features, self.config.d_model)
        self.assertEqual(expert.specialization_name, "Python_Coding")

    def test_expert_custom_intermediate_dimension(self) -> None:
        cfg = KhwarizmiConfig(d_model=32, d_ff=64, expert_d_ff=96)
        expert = ExpertLayer(cfg)
        self.assertEqual(expert.w1.out_features, 96)
        self.assertEqual(expert.w2.in_features, 96)
        self.assertEqual(expert.d_ff, 96)

    def test_experts_have_independent_parameters(self) -> None:
        a = ExpertLayer(self.config)
        b = ExpertLayer(self.config)
        # Separate storage (no parameter sharing between experts).
        self.assertIsNot(a.w1.weight, b.w1.weight)
        self.assertNotEqual(a.w1.weight.data_ptr(), b.w1.weight.data_ptr())
        self.assertIsNot(a.w2.weight, b.w2.weight)
        # Mutating one expert never affects another.
        snapshot = b.w1.weight.detach().clone()
        with torch.no_grad():
            a.w1.weight.add_(10.0)
        self.assertTrue(torch.equal(b.w1.weight, snapshot))

    def test_expert_forward_matches_manual_swish_ffn(self) -> None:
        expert = ExpertLayer(self.config)
        x = torch.randn(7, self.config.d_model)
        expected = expert.w2(torch.nn.functional.silu(expert.w1(x)))
        self.assertTrue(torch.allclose(expert(x), expected, atol=1e-6))

    def test_expert_gradient_flow(self) -> None:
        expert = ExpertLayer(self.config)
        x = torch.randn(5, self.config.d_model, requires_grad=True)
        out = expert(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        for name, param in expert.named_parameters():
            self.assertIsNotNone(param.grad, msg=f"{name} has no grad")
            self.assertTrue(param.grad.abs().sum().item() > 0)

    def test_create_standard_specialists(self) -> None:
        experts = create_standard_specialists(self.config)
        self.assertEqual(len(experts), self.config.num_experts)
        for i, expert in enumerate(experts):
            self.assertIsInstance(expert, ExpertLayer)
            expected = (
                SPECIALIZATION_NAMES[i]
                if i < len(SPECIALIZATION_NAMES)
                else f"General_Specialist_{i}"
            )
            self.assertEqual(expert.specialization_name, expected)

    def test_specialists_beyond_named_list_get_general_labels(self) -> None:
        cfg = KhwarizmiConfig(num_experts=10)
        experts = create_standard_specialists(cfg)
        self.assertEqual(len(experts), 10)
        self.assertEqual(experts[7].specialization_name, "General_Fact_Recall")
        self.assertEqual(experts[8].specialization_name, "General_Specialist_8")
        self.assertEqual(experts[9].specialization_name, "General_Specialist_9")


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
class TestSparseMoERouter(RngNeutralTestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0)
        self.config = get_tiny_test_config()
        self.moe = SparseMoELayer(self.config)
        self.moe.eval()  # deterministic routing by default in tests

    def test_router_logits_shape(self) -> None:
        x = torch.randn(3, 5, self.config.d_model)
        decision = self.moe.route(x)
        self.assertEqual(
            decision.noisy_logits.shape, (15, self.config.num_experts)
        )

    def test_router_probs_sum_to_one(self) -> None:
        x = torch.randn(16, self.config.d_model)
        decision = self.moe.route(x)
        sums = decision.router_probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-6))

    def test_clean_logits_are_linear_in_input(self) -> None:
        x = torch.randn(8, self.config.d_model)
        expected = x @ self.moe.w_gate.weight.t()
        logits = self.moe.compute_noisy_logits(x, use_noise=False)
        self.assertTrue(torch.allclose(logits, expected, atol=1e-6))

    def test_topk_selection_is_optimal_and_distinct(self) -> None:
        x = torch.randn(32, self.config.d_model)
        decision = self.moe.route(x)
        idx = decision.topk_indices  # (N, K)
        self.assertEqual(idx.shape, (32, self.config.top_k_experts))
        self.assertTrue(((idx >= 0) & (idx < self.config.num_experts)).all())
        # Distinct experts per token.
        for row in idx:
            self.assertEqual(len(set(row.tolist())), self.config.top_k_experts)
        # Every selected logit >= every unselected logit (true Top-K).
        logits = decision.noisy_logits
        for n in range(logits.size(0)):
            selected = logits[n, idx[n]]
            unselected_mask = torch.ones(
                self.config.num_experts, dtype=torch.bool
            )
            unselected_mask[idx[n]] = False
            unselected = logits[n, unselected_mask]
            if unselected.numel() > 0:
                self.assertGreaterEqual(
                    selected.min().item(), unselected.max().item() - 1e-6
                )

    def test_topk_weights_are_normalized_softmax(self) -> None:
        x = torch.randn(16, self.config.d_model)
        decision = self.moe.route(x)
        weights = decision.topk_weights
        self.assertTrue((weights > 0).all())
        sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-6))
        # Manual check: softmax over the selected logits.
        sel_logits = torch.gather(
            decision.noisy_logits, 1, decision.topk_indices
        )
        expected = torch.softmax(sel_logits, dim=-1)
        self.assertTrue(torch.allclose(weights, expected, atol=1e-6))

    def test_noise_is_disabled_at_inference(self) -> None:
        self.moe.eval()
        x = torch.randn(8, self.config.d_model)
        clean = self.moe.compute_noisy_logits(x, use_noise=False)
        default = self.moe.compute_noisy_logits(x)  # config enables noise
        forced = self.moe.compute_noisy_logits(x, use_noise=True)
        self.assertTrue(torch.equal(default, clean))
        self.assertTrue(torch.equal(forced, clean))

    def test_noise_is_applied_during_training(self) -> None:
        self.moe.train()
        x = torch.randn(64, self.config.d_model)
        clean = self.moe.compute_noisy_logits(x, use_noise=False)
        noisy = self.moe.compute_noisy_logits(x, use_noise=True)
        self.assertFalse(torch.equal(clean, noisy))
        # Noise magnitude equals softplus(w_noise(x)); standardized noise
        # should have unit variance.
        noise_std = torch.nn.functional.softplus(self.moe.w_noise(x))
        expected_std = torch.nn.functional.softplus(
            x @ self.moe.w_noise.weight.t()
        )
        self.assertTrue(torch.allclose(noise_std, expected_std, atol=1e-6))
        standardized = (noisy - clean) / noise_std
        var = standardized.var().item()
        self.assertGreater(var, 0.8)
        self.assertLess(var, 1.2)

    def test_noise_flag_overrides_config(self) -> None:
        cfg = KhwarizmiConfig(
            d_model=16, num_experts=4, top_k_experts=2, moe_noise_enabled=False
        )
        moe = SparseMoELayer(cfg).train()
        x = torch.randn(8, 16)
        clean = moe.compute_noisy_logits(x, use_noise=False)
        default = moe.compute_noisy_logits(x)  # follows config: no noise
        forced = moe.compute_noisy_logits(x, use_noise=True)
        self.assertTrue(torch.equal(default, clean))
        self.assertFalse(torch.equal(forced, clean))

    def test_deterministic_inference(self) -> None:
        self.moe.eval()
        x = torch.randn(4, 7, self.config.d_model)
        out1, loss1 = self.moe(x)
        torch.manual_seed(12345)  # seeding must not change inference
        out2, loss2 = self.moe(x)
        self.assertTrue(torch.equal(out1, out2))
        self.assertTrue(torch.equal(loss1, loss2))

    def test_tie_handling_is_deterministic(self) -> None:
        self.moe.eval()
        # Identical logits for all experts -> Top-K ties.
        with torch.no_grad():
            self.moe.w_gate.weight.zero_()
        x = torch.ones(6, self.config.d_model)
        d1 = self.moe.route(x)
        d2 = self.moe.route(x)
        self.assertTrue(torch.equal(d1.topk_indices, d2.topk_indices))
        # Ties still respect K experts per token.
        self.assertEqual(
            d1.topk_indices.shape, (6, self.config.top_k_experts)
        )

    def test_route_accepts_batch_and_sequence_inputs(self) -> None:
        x2 = torch.randn(9, self.config.d_model)
        x3 = torch.randn(3, 4, self.config.d_model)
        d2 = self.moe.route(x2)
        d3 = self.moe.route(x3)
        self.assertEqual(d2.noisy_logits.shape, (9, self.config.num_experts))
        self.assertEqual(d3.noisy_logits.shape, (12, self.config.num_experts))
        self.assertEqual(d2.aux_loss.shape, torch.Size([]))
        self.assertEqual(d3.aux_loss.shape, torch.Size([]))


# --------------------------------------------------------------------------
# Sparse execution
# --------------------------------------------------------------------------
class TestSparseMoEForward(RngNeutralTestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0)
        self.config = get_tiny_test_config()
        self.moe = SparseMoELayer(self.config)
        self.moe.eval()

    def _call_counts(self):
        """Register forward hooks counting calls + tokens per expert."""
        counts = {i: 0 for i in range(self.config.num_experts)}
        tokens = {i: 0 for i in range(self.config.num_experts)}

        def make_hook(idx):
            def hook(module, args, out):
                counts[idx] += 1
                tokens[idx] += args[0].shape[0]
            return hook

        handles = [
            expert.register_forward_hook(make_hook(i))
            for i, expert in enumerate(self.moe.experts)
        ]
        return counts, tokens, handles

    def test_forward_shapes_batch_and_sequence(self) -> None:
        x3 = torch.randn(2, 6, self.config.d_model)
        out3, loss3 = self.moe(x3)
        self.assertEqual(out3.shape, x3.shape)
        self.assertEqual(loss3.shape, torch.Size([]))
        x2 = torch.randn(5, self.config.d_model)
        out2, loss2 = self.moe(x2)
        self.assertEqual(out2.shape, x2.shape)

    def test_batch_and_sequence_outputs_are_consistent(self) -> None:
        # Routing is per-token: (B, L, D) must equal concatenated (N, D).
        x3 = torch.randn(3, 4, self.config.d_model)
        out3, loss3 = self.moe(x3)
        out2, loss2 = self.moe(x3.reshape(-1, self.config.d_model))
        self.assertTrue(torch.allclose(out3, out2.reshape(3, 4, -1), atol=1e-6))
        self.assertTrue(torch.allclose(loss3, loss2, atol=1e-6))

    def test_only_topk_experts_are_executed(self) -> None:
        counts, tokens, _ = self._call_counts()
        x = torch.randn(64, self.config.d_model)
        decision = self.moe.route(x)
        out, _ = self.moe(x)

        expected = {
            int(i) for i in torch.unique(decision.topk_indices).tolist()
        }
        called = {i for i, c in counts.items() if c > 0}
        # Exactly the routed experts were called — never any other expert.
        self.assertEqual(called, expected)
        # Each executed expert is called exactly once (batched gather).
        for i in called:
            self.assertEqual(counts[i], 1)
        # The total token throughput equals N * K and nothing more.
        self.assertEqual(sum(tokens.values()), 64 * self.config.top_k_experts)
        # Diagnostic records the executed experts.
        self.assertEqual(set(self.moe.last_routed_experts), expected)

    def test_unselected_experts_never_executed(self) -> None:
        # Force routing to experts {0, 1} for every token: gate row 0 dominates,
        # row 1 second, so Top-2 = {0, 1} always (positive inputs).
        with torch.no_grad():
            self.moe.w_gate.weight.fill_(0.0)
            self.moe.w_gate.weight[0, :] = 2.0
            self.moe.w_gate.weight[1, :] = 1.0
        counts, tokens, _ = self._call_counts()
        x = torch.rand(32, self.config.d_model)
        out, _ = self.moe(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(counts[0], 1)
        self.assertEqual(counts[1], 1)
        self.assertEqual(counts[2], 0)
        self.assertEqual(counts[3], 0)
        self.assertEqual(tokens[0] + tokens[1], 32 * 2)
        self.assertEqual(self.moe.last_routed_experts, [0, 1])

    def test_output_matches_manual_topk_reference(self) -> None:
        x = torch.randn(13, self.config.d_model)
        decision = self.moe.route(x)
        out, _ = self.moe(x)

        reference = torch.zeros_like(x)
        idx = decision.topk_indices
        weights = decision.topk_weights
        for n in range(x.size(0)):
            for k in range(self.config.top_k_experts):
                e = int(idx[n, k])
                reference[n] += (
                    weights[n, k] * self.moe.experts[e](x[n : n + 1]).squeeze(0)
                )
        self.assertTrue(torch.allclose(out, reference, atol=1e-5))

    def test_topk_equals_num_experts_uses_all_experts(self) -> None:
        cfg = KhwarizmiConfig(d_model=16, d_ff=32, num_experts=3, top_k_experts=3)
        moe = SparseMoELayer(cfg).eval()
        x = torch.randn(8, 16)
        decision = moe.route(x)
        out, _ = moe(x)
        self.assertEqual(set(moe.last_routed_experts), {0, 1, 2})
        # With K == E the sparse layer equals the router-weighted dense
        # combination of all experts.
        dense = torch.zeros_like(x)
        for i, expert in enumerate(moe.experts):
            dense += decision.router_probs[:, i : i + 1] * expert(x)
        self.assertTrue(torch.allclose(out, dense, atol=1e-5))

    def test_invalid_input_shapes(self) -> None:
        with self.assertRaises(ValueError):
            self.moe(torch.randn(self.config.d_model))  # 1-D
        with self.assertRaises(ValueError):
            self.moe(torch.randn(2, self.config.d_model + 1))  # wrong dim
        with self.assertRaises(ValueError):
            self.moe(torch.randn(0, self.config.d_model))  # empty
        with self.assertRaises(TypeError):
            self.moe("not a tensor")


# --------------------------------------------------------------------------
# Gradient flow
# --------------------------------------------------------------------------
class TestSparseMoEGradients(RngNeutralTestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0)
        self.config = get_tiny_test_config()
        self.moe = SparseMoELayer(self.config)
        self.moe.eval()

    def _force_routing_to_first_two_experts(self) -> None:
        # Gate rows (one row per expert): expert 0 dominates, expert 1 second,
        # so Top-2 = {0, 1} for every token with positive inputs.
        with torch.no_grad():
            self.moe.w_gate.weight.fill_(0.0)
            self.moe.w_gate.weight[0, :] = 2.0
            self.moe.w_gate.weight[1, :] = 1.0

    def test_selected_experts_receive_gradients(self) -> None:
        self._force_routing_to_first_two_experts()
        x = torch.rand(16, self.config.d_model, requires_grad=True)
        out, aux = self.moe(x, use_noise=False)
        loss = out.sum() + aux
        loss.backward()
        for i in (0, 1):  # selected experts
            self.assertIsNotNone(self.moe.experts[i].w1.weight.grad)
            self.assertGreater(
                self.moe.experts[i].w1.weight.grad.abs().sum().item(), 0.0
            )
        self.assertIsNotNone(x.grad)

    def test_unselected_experts_receive_no_gradient(self) -> None:
        self._force_routing_to_first_two_experts()
        x = torch.rand(16, self.config.d_model)
        out, aux = self.moe(x, use_noise=False)
        loss = out.sum() + aux
        loss.backward()
        for i in (2, 3):  # never routed
            self.assertIsNone(self.moe.experts[i].w1.weight.grad)
            self.assertIsNone(self.moe.experts[i].w2.weight.grad)

    def test_router_receives_gradients_for_all_experts(self) -> None:
        self._force_routing_to_first_two_experts()
        x = torch.rand(16, self.config.d_model)
        out, aux = self.moe(x, use_noise=False)
        loss = out.sum() + aux
        loss.backward()
        gate_grad = self.moe.w_gate.weight.grad
        self.assertIsNotNone(gate_grad)
        # Every expert column receives a non-zero gradient: the selected
        # columns through the routing weights, the unselected columns through
        # the differentiable load-balancing loss (anti-collapse signal).
        for j in range(self.config.num_experts):
            self.assertGreater(
                gate_grad[:, j].abs().sum().item(), 0.0,
                msg=f"gate column {j} received no gradient",
            )

    def test_gradient_flows_through_routing_weights(self) -> None:
        x = torch.randn(8, self.config.d_model)
        decision = self.moe.route(x)
        decision.topk_weights.sum().backward()
        gate_grad = self.moe.w_gate.weight.grad
        self.assertIsNotNone(gate_grad)
        # Only the selected experts' gate columns receive weight-gradient.
        selected = set(torch.unique(decision.topk_indices).tolist())
        for j in range(self.config.num_experts):
            col = gate_grad[:, j].abs().sum().item()
            if j in selected:
                self.assertGreater(col, 0.0)
            else:
                self.assertEqual(col, 0.0)

    def test_noise_projection_receives_gradients_in_training(self) -> None:
        self.moe.train()
        x = torch.randn(32, self.config.d_model)
        logits = self.moe.compute_noisy_logits(x, use_noise=True)
        logits.sum().backward()
        self.assertIsNotNone(self.moe.w_noise.weight.grad)
        self.assertGreater(
            self.moe.w_noise.weight.grad.abs().sum().item(), 0.0
        )

    def test_aux_loss_gradient_matches_analytic_formula(self) -> None:
        # L = alpha*E*sum_i f_i*P_i, with f_i constant (discrete dispatch):
        # dL/dlogits_nj = (alpha*E/N) * [f_j*P_nj - P_nj*sum_i f_i*P_ni]
        torch.manual_seed(3)
        x = torch.randn(10, self.config.d_model)
        decision = self.moe.route(x)
        aux = decision.aux_loss
        logits = decision.noisy_logits  # same computation graph as aux

        (dlogits,) = torch.autograd.grad(aux, logits)
        P = decision.router_probs
        f = decision.expert_fractions
        n = x.size(0)
        weighted = (f * P).sum(dim=-1, keepdim=True)  # (N, 1)
        expected = (self.moe.alpha_moe * self.moe.num_experts / n) * (
            f.unsqueeze(0) * P - P * weighted
        )
        self.assertTrue(torch.allclose(dlogits, expected, atol=1e-6))


# --------------------------------------------------------------------------
# Load-balancing auxiliary loss
# --------------------------------------------------------------------------
class TestLoadBalancingLoss(RngNeutralTestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0)
        self.config = get_tiny_test_config()
        self.moe = SparseMoELayer(self.config)
        self.moe.eval()

    def test_loss_closed_form_values(self) -> None:
        # alpha * E * sum(f * P): with f = [1, 1, 0, 0], P = [0.25]*4
        f = torch.tensor([1.0, 1.0, 0.0, 0.0])
        P = torch.full((4,), 0.25)
        loss = self.moe.compute_load_balance_loss(f, P)
        self.assertAlmostEqual(
            loss.item(), 0.01 * 4 * (0.25 + 0.25), places=6
        )

    def test_loss_requires_grad(self) -> None:
        f = torch.tensor([1.0, 0.0, 1.0, 0.0])
        P = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
        loss = self.moe.compute_load_balance_loss(f, P)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNotNone(P.grad)
        self.assertTrue(torch.allclose(P.grad, torch.full((4,), 0.01 * 4) * f))

    def test_loss_shape_validation(self) -> None:
        with self.assertRaises(ValueError):
            self.moe.compute_load_balance_loss(torch.ones(3), torch.ones(4))

    def test_alpha_zero_disables_loss(self) -> None:
        cfg = KhwarizmiConfig(
            d_model=16, d_ff=32, num_experts=4, top_k_experts=2,
            load_balance_alpha=0.0,
        )
        moe = SparseMoELayer(cfg).eval()
        _, aux = moe(torch.randn(8, 16))
        self.assertEqual(aux.item(), 0.0)

    def test_balanced_routing_has_closed_form_loss(self) -> None:
        # Uniform gates -> P_i = 1/E for all tokens; Top-K ties are broken
        # deterministically, so exactly K experts receive f_i = 1 each:
        # L = alpha * E * (K * 1/E) = alpha * K.
        with torch.no_grad():
            self.moe.w_gate.weight.zero_()
        x = torch.ones(32, self.config.d_model)
        decision = self.moe.route(x)
        self.assertAlmostEqual(
            decision.aux_loss.item(),
            self.moe.alpha_moe * self.moe.top_k,
            places=6,
        )
        self.assertAlmostEqual(
            decision.expert_fractions.sum().item(), float(self.moe.top_k), places=6
        )

    def test_collapsed_routing_increases_loss(self) -> None:
        # Collapsed: one expert captures nearly all probability mass.
        with torch.no_grad():
            self.moe.w_gate.weight.fill_(0.0)
            self.moe.w_gate.weight[0, :] = 2.0
            self.moe.w_gate.weight[1, :] = 1.0
        x = torch.rand(64, self.config.d_model)
        collapsed = self.moe.route(x).aux_loss.item()
        # Balanced reference: alpha * K (uniform routing).
        balanced = self.moe.alpha_moe * self.moe.top_k
        self.assertGreater(collapsed, balanced * 1.5)
        # Collapse detection: the dominant expert absorbs ~all tokens.
        self.assertGreater(self.moe.route(x).expert_fractions[0].item(), 0.95)

    def test_aux_loss_within_bounds(self) -> None:
        # 0 <= L <= alpha * E since sum_i f_i * P_i <= sum_i P_i = 1.
        x = torch.randn(100, self.config.d_model)
        decision = self.moe.route(x)
        aux = decision.aux_loss.item()
        self.assertGreaterEqual(aux, 0.0)
        self.assertLessEqual(aux, self.moe.alpha_moe * self.moe.num_experts + 1e-6)

    def test_random_routing_uses_all_experts(self) -> None:
        torch.manual_seed(7)
        x = torch.randn(512, self.config.d_model)
        out, _ = self.moe(x)
        self.assertEqual(out.shape, x.shape)
        f = self.moe.route(x).expert_fractions
        # No routing collapse under random initialization: every expert
        # receives a non-trivial share (utilization check).
        for j in range(self.config.num_experts):
            self.assertGreater(f[j].item(), 0.05)
            self.assertLess(f[j].item(), 0.8)
        self.assertEqual(
            set(self.moe.last_routed_experts), set(range(self.config.num_experts))
        )


# --------------------------------------------------------------------------
# Configuration validation
# --------------------------------------------------------------------------
class TestSparseMoEConfigValidation(RngNeutralTestCase):
    def setUp(self) -> None:
        super().setUp()

    def _base(self, **overrides):
        kwargs = dict(
            vocab_size=64, d_model=16, n_heads=2, d_expansion=4, d_ff=32,
            num_experts=4, top_k_experts=2, moe_frequency=2,
            max_seq_len=64, memory_dim=16, memory_slots=8,
            short_term_capacity=32, num_pathways=5, n_layers=2,
        )
        kwargs.update(overrides)
        return KhwarizmiConfig(**kwargs)

    def test_valid_moe_config_accepted(self) -> None:
        cfg = self._base(enable_moe=False)
        self.assertFalse(cfg.enable_moe)
        cfg = self._base(expert_d_ff=64, moe_noise_enabled=False)
        self.assertEqual(cfg.expert_d_ff, 64)
        self.assertFalse(cfg.moe_noise_enabled)

    def test_invalid_expert_count_rejected(self) -> None:
        for bad in (0, -1, -5):
            with self.assertRaises(ValueError):
                self._base(num_experts=bad)

    def test_invalid_topk_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._base(num_experts=4, top_k_experts=0)
        with self.assertRaises(ValueError):
            self._base(num_experts=4, top_k_experts=5)  # Top-K > experts
        with self.assertRaises(ValueError):
            self._base(num_experts=2, top_k_experts=3)

    def test_invalid_routing_dimensions_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._base(d_model=0)
        with self.assertRaises(ValueError):
            self._base(d_model=-8)
        with self.assertRaises(ValueError):
            self._base(d_ff=0)  # expert intermediate dim
        with self.assertRaises(ValueError):
            self._base(expert_d_ff=0)
        with self.assertRaises(ValueError):
            self._base(expert_d_ff=-16)

    def test_invalid_load_balance_alpha_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._base(load_balance_alpha=-0.01)
        with self.assertRaises(ValueError):
            self._base(load_balance_alpha=float("nan"))
        with self.assertRaises(ValueError):
            self._base(load_balance_alpha=float("inf"))

    def test_invalid_moe_frequency_rejected(self) -> None:
        for bad in (0, -2):
            with self.assertRaises(ValueError):
                self._base(moe_frequency=bad)

    def test_moe_fields_survive_json_round_trip(self) -> None:
        cfg = self._base(expert_d_ff=48, enable_moe=False)
        loaded = KhwarizmiConfig.from_json_string(cfg.to_json_string())
        self.assertEqual(loaded.expert_d_ff, 48)
        self.assertFalse(loaded.enable_moe)
        self.assertEqual(loaded.moe_noise_enabled, cfg.moe_noise_enabled)
        self.assertEqual(loaded.load_balance_alpha, cfg.load_balance_alpha)

    def test_layer_rejects_wrong_expert_count(self) -> None:
        cfg = self._base()
        experts = create_standard_specialists(cfg)[:2]
        with self.assertRaises(ValueError):
            SparseMoELayer(cfg, experts=experts)

    def test_layer_rejects_incompatible_experts(self) -> None:
        cfg = self._base()
        bad = ExpertLayer(self._base(d_model=32))  # wrong d_model
        with self.assertRaises(ValueError):
            SparseMoELayer(cfg, experts=[bad] * cfg.num_experts)
        with self.assertRaises(TypeError):
            SparseMoELayer(cfg, experts=["nope"] * cfg.num_experts)

    def test_layer_rejects_invalid_config(self) -> None:
        with self.assertRaises(ValueError):
            SparseMoELayer(self._base(num_experts=0))


# --------------------------------------------------------------------------
# Model integration / compatibility / regression
# --------------------------------------------------------------------------
class TestSparseMoEIntegration(RngNeutralTestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(0)

    def _dummy_input(self, cfg, batch_size=2, seq_len=8):
        return torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    def test_model_builds_moe_by_default(self) -> None:
        cfg = get_tiny_test_config()
        model = KhwarizmiModel(cfg)
        self.assertIsNotNone(model.shared_moe_layer)
        moe_blocks = [i for i, l in enumerate(model.layers) if l.is_moe_layer]
        self.assertEqual(len(moe_blocks), cfg.n_layers // cfg.moe_frequency)

    def test_moe_frequency_places_moe_every_nth_block(self) -> None:
        cfg = KhwarizmiConfig(
            vocab_size=64, d_model=16, n_heads=2, n_layers=4,
            moe_frequency=2, max_seq_len=32, memory_slots=4,
        )
        model = KhwarizmiModel(cfg)
        flags = [layer.is_moe_layer for layer in model.layers]
        self.assertEqual(flags, [False, True, False, True])

    def test_moe_disabled_preserves_dense_behavior(self) -> None:
        cfg = get_tiny_test_config()
        cfg.enable_moe = False
        model = KhwarizmiModel(cfg)
        # No experts/router built; every block is a dense FFN block.
        self.assertIsNone(model.shared_moe_layer)
        self.assertTrue(all(not layer.is_moe_layer for layer in model.layers))
        self.assertTrue(all(layer.ffn is not None for layer in model.layers))

        out = model(self._dummy_input(cfg))
        self.assertEqual(
            out.logits.shape, (2, 8, cfg.vocab_size)
        )
        # No MoE auxiliary loss is produced in the dense configuration.
        self.assertEqual(out.losses["moe_aux_loss"].item(), 0.0)

        # Full backward pass remains healthy without any MoE component.
        out.losses["total_aux_loss"].backward()
        self.assertGreater(model.count_parameters(), 0)

    def test_moe_enabled_model_routes_and_reports_aux_loss(self) -> None:
        cfg = get_tiny_test_config()
        model = KhwarizmiModel(cfg)
        model.eval()
        out = model(self._dummy_input(cfg))
        self.assertGreaterEqual(out.losses["moe_aux_loss"].item(), 0.0)
        # If the auxiliary loss was accumulated, the shared MoE layer must
        # have recorded the experts it actually executed.
        if out.losses["moe_aux_loss"].item() > 0.0:
            self.assertGreater(len(model.shared_moe_layer.last_routed_experts), 0)

    def test_moe_disabled_model_is_deterministic_in_eval(self) -> None:
        cfg = get_tiny_test_config()
        cfg.enable_moe = False
        model = KhwarizmiModel(cfg).eval()
        x = self._dummy_input(cfg)
        out1 = model(x).logits
        out2 = model(x).logits
        self.assertTrue(torch.equal(out1, out2))

    def test_phase2_ksc_prototype_remains_moe_free(self) -> None:
        # Phase 2 boundary: the KSC prototype has no MoE components.
        from khwarizmi.config import get_prototype_50m_config
        cfg = get_prototype_50m_config()
        prototype = KhwarizmiKSCPrototype(cfg)
        self.assertFalse(hasattr(prototype, "shared_moe_layer"))
        x = self._dummy_input(cfg, batch_size=1, seq_len=4)
        out = prototype(x)
        self.assertEqual(out.logits.shape, (1, 4, cfg.vocab_size))

    def test_phase3_dual_memory_compat_with_moe(self) -> None:
        # Full Phase 1-4 stack: KSC + Dual Memory + Router + Sparse MoE.
        cfg = get_tiny_test_config()
        model = KhwarizmiModel(cfg)
        st_state, lt_table = model.init_state(2)
        out = model(self._dummy_input(cfg), st_state, lt_table)
        self.assertIn("moe_aux_loss", out.losses)
        self.assertGreaterEqual(out.losses["moe_aux_loss"].item(), 0.0)
        # Dual Memory state structures are produced and carried through.
        self.assertIn("recurrent_state", out.short_term_state)
        self.assertIn("valid_mask", out.long_term_table)
        self.assertIn("memory_valid_slots_count", out.diagnostics)
        # Writing to the Phase 3 long-term store still works alongside MoE.
        written = model.long_term_memory.write(
            candidate_repr=torch.randn(2, cfg.d_model),
            memory_table=model.long_term_memory.init_memory_table(2),
            g_write=torch.ones(2, 1),
            current_step=1,
        )
        self.assertGreater(int(torch.sum(written["valid_mask"]).item()), 0)

    def test_full_model_gradient_flow_with_moe(self) -> None:
        cfg = get_tiny_test_config()
        model = KhwarizmiModel(cfg)
        out = model(self._dummy_input(cfg))
        (out.logits.sum() + out.losses["total_aux_loss"]).backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad, msg=f"{name} received no gradient"
                )
        # MoE expert parameters must receive gradients when MoE executes.
        moe_grads = sum(
            p.grad.abs().sum().item()
            for p in model.shared_moe_layer.experts.parameters()
        )
        self.assertGreater(moe_grads, 0.0)


if __name__ == "__main__":
    unittest.main()
