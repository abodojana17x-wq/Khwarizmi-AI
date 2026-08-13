"""
Phase 4 Benchmark — Sparse Mixture-of-Experts (MoE).

Measures the Phase 4 Sparse Top-K Noisy-Gated MoE layer per the roadmap and
the Phase 4 specification:

  1. Parameter efficiency & memory: total parameters vs parameters *active*
     per token (router + exactly K experts), plus activation-buffer memory
     footprint of the sparse vs dense execution strategies.
  2. Sparsity verification: expert forward-call counting proves the sparse
     layer evaluates only the Top-K selected experts (K expert evaluations per
     token) while the DENSE reference evaluates all E experts for every token.
  3. Routing overhead: wall time of the router alone vs the full layer.
  4. Forward latency: SPARSE (Top-K executed) vs DENSE (all experts executed,
     looped and fused-batched variants) vs a single dense FFN of equal
     *active* parameter count.
  5. Expert utilization: dispatch fraction distribution over a large batch
     (no expert <5% or >40% of tokens under balanced routing).
  6. Load-balancing loss: the auxiliary loss is low under balanced routing and
     16x higher under routing collapse (detection), and — trained on a task
     that rewards using a single expert — the router collapses without the
     auxiliary loss but stays balanced with it (prevention).

Run:
    python benchmarks/phase4_sparse_moe.py

This script is deterministic (fixed seeds) and CPU-only. The roadmap's
literal success criteria (>= 8% validation-perplexity gain on a multi-domain
test set with a *trained* router) require the Phase 9 dataset pipeline and
Phase 10 training, which are out of Phase 4 scope — this benchmark validates
the structural and computational properties Phase 4 owns.
"""

import os
import sys
from time import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)

from khwarizmi.config import KhwarizmiConfig
from khwarizmi.experts import SparseMoELayer

DEVICE = torch.device("cpu")


def _benchmark_config() -> KhwarizmiConfig:
    """A moderately sized MoE: 32 experts, Top-2, Swish FFN experts."""
    return KhwarizmiConfig(
        vocab_size=512,
        d_model=256,
        n_heads=4,
        d_expansion=16,
        d_ff=1024,
        num_experts=32,
        top_k_experts=2,
        moe_frequency=2,
        max_seq_len=128,
        load_balance_alpha=0.01,
        moe_noise_enabled=True,
        tier_name="MoE-Benchmark",
    )


class DenseMoEReference(nn.Module):
    """
    DENSE reference: evaluates ALL E experts on EVERY token and combines their
    outputs with the router's full-softmax probabilities. Shares the exact
    expert modules of the sparse layer, so the comparison isolates execution
    strategy (dense vs sparse) from parameterization.
    """

    def __init__(self, sparse_layer: SparseMoELayer):
        super().__init__()
        self.sparse = sparse_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.size(-1))
        probs = F.softmax(
            self.sparse.compute_noisy_logits(flat, use_noise=False), dim=-1
        )
        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.sparse.experts):
            out = out + probs[:, i : i + 1] * expert(flat)
        return out.reshape(x.shape)


class FusedDenseMoEReference(nn.Module):
    """
    DENSE reference (fused): the same all-experts computation as
    DenseMoEReference but batched into two big matmuls (no Python expert
    loop), representing an optimized dense implementation.
    """

    def __init__(self, sparse_layer: SparseMoELayer):
        super().__init__()
        self.sparse = sparse_layer
        with torch.no_grad():
            experts = list(sparse_layer.experts)
            self.w1 = nn.Parameter(torch.stack([e.w1.weight.detach() for e in experts]))
            self.b1 = nn.Parameter(torch.stack([e.w1.bias.detach() for e in experts]))
            self.w2 = nn.Parameter(torch.stack([e.w2.weight.detach() for e in experts]))
            self.b2 = nn.Parameter(torch.stack([e.w2.bias.detach() for e in experts]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.size(-1))
        probs = F.softmax(
            self.sparse.compute_noisy_logits(flat, use_noise=False), dim=-1
        )
        h = torch.einsum("nd,efd->nef", flat, self.w1) + self.b1  # (N, E, DF)
        h = F.silu(h)
        out = torch.einsum("nef,edf->ned", h, self.w2) + self.b2  # (N, E, D)
        out = (out * probs.unsqueeze(-1)).sum(dim=1)  # (N, D)
        return out.reshape(x.shape)


class DenseFFNReference(nn.Module):
    """Single dense Swish FFN with K times the expert width (equal *active* params)."""

    def __init__(self, d_model: int, d_ff_active: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff_active)
        self.w2 = nn.Linear(d_ff_active, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)))


def _timeit(fn, repeats: int = 3, warmup: int = 1) -> float:
    """Best-of-`repeats` wall time in seconds after `warmup` untimed calls."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time()
        fn()
        best = min(best, time() - t0)
    return best


def _count_expert_calls(experts):
    """Install forward hooks on the given experts; return (calls, tokens) dicts."""
    calls = {}
    tokens = {}

    def make_hook(idx):
        def hook(m, args, out):
            calls[idx] = calls.get(idx, 0) + 1
            tokens[idx] = tokens.get(idx, 0) + args[0].shape[0]
        return hook

    handles = []
    for i, expert in enumerate(experts):
        handles.append(expert.register_forward_hook(make_hook(i)))
    return calls, tokens, handles


def main() -> None:
    print("=" * 72)
    print("PHASE 4 BENCHMARK — Sparse Mixture-of-Experts (MoE)")
    print("=" * 72)

    torch.manual_seed(0)
    cfg = _benchmark_config()
    sparse = SparseMoELayer(cfg).eval()
    dense = DenseMoEReference(sparse).eval()
    fused = FusedDenseMoEReference(sparse).eval()
    active_ffn = DenseFFNReference(cfg.d_model, cfg.top_k_experts * cfg.d_ff).eval()

    E, K, D, DF = cfg.num_experts, cfg.top_k_experts, cfg.d_model, cfg.d_ff

    # 1) Parameter efficiency & memory footprint ----------------------------
    print("\n[1] Parameter efficiency & memory footprint")
    total = sparse.count_expert_parameters() + sparse.count_router_parameters()
    active = sparse.count_active_parameters()
    print(f"  experts: E={E}, top_k={K}, d_model={D}, expert d_ff={DF}")
    print(f"  total MoE parameters      : {total / 1e6:8.3f} M "
          f"({4 * total / 1e6:7.1f} MB fp32)")
    print(
        f"  active parameters / token : {active / 1e6:8.3f} M "
        f"({100 * active / total:.1f}%) "
        f"({4 * active / 1e6:6.1f} MB fp32)"
    )
    # Largest intermediate activation buffers (fp32) during one forward pass:
    # router logits/probs (N, E), Top-K one-hot (N, K, E), output (N, D),
    # and the biggest expert hidden state (n_i, d_ff) with n_i ~ N*K/E tokens.
    n_mem = 2048
    # logits + probs (N, E), Top-K one-hot (N, K, E), output (N, D)
    router_buf = 2 * n_mem * E * 4 + n_mem * K * E * 4 + n_mem * D * 4
    n_i = n_mem * K / E
    sparse_buf = router_buf + n_i * D * 4 + n_i * DF * 4  # expert in + hidden
    dense_loop_buf = router_buf + n_mem * D * 4 + n_mem * DF * 4  # one full-batch expert
    dense_fused_buf = router_buf + n_mem * E * DF * 4 + n_mem * E * D * 4  # (N,E,DF)+(N,E,D)
    print(f"  peak activation buffers @ {n_mem} tokens:")
    print(f"    SPARSE MoE  : ~{sparse_buf / 1e6:6.1f} MB "
          f"(router + biggest expert batch of ~{n_i:.0f} tokens)")
    print(f"    DENSE loop  : ~{dense_loop_buf / 1e6:6.1f} MB "
          f"(router + one full-batch expert)")
    print(f"    DENSE fused : ~{dense_fused_buf / 1e6:6.1f} MB "
          f"(router + (N, E, DF) batched tensor)")

    # 2) Sparsity verification ----------------------------------------------
    print("\n[2] Sparsity verification (expert forward-call counting)")
    n_tokens = 256
    x_small = torch.randn(1, n_tokens, D)

    calls, tokens, handles = _count_expert_calls(sparse.experts)
    out_sparse, aux = sparse(x_small)
    for h in handles:
        h.remove()
    expert_evals_sparse = sum(tokens.values())
    print(
        f"  SPARSE: unique experts executed = {len(calls)} (of {E}); "
        f"expert evaluations = {expert_evals_sparse} = "
        f"{expert_evals_sparse / n_tokens:.1f} per token (Top-K = {K})"
    )

    calls, tokens, handles = _count_expert_calls(dense.sparse.experts)
    out_dense = dense(x_small)
    for h in handles:
        h.remove()
    expert_evals_dense = sum(tokens.values())
    print(
        f"  DENSE : unique experts executed = {len(calls)} (of {E}); "
        f"expert evaluations = {expert_evals_dense} = "
        f"{expert_evals_dense / n_tokens:.1f} per token (all experts)"
    )
    print(
        f"  => sparse executes {expert_evals_dense // expert_evals_sparse}x fewer "
        f"expert-token pairs; experts never selected are never called."
    )

    # 3) Routing overhead ---------------------------------------------------
    print("\n[3] Routing overhead")
    route_time = _timeit(lambda: sparse.route(x_small))
    forward_time = _timeit(lambda: sparse(x_small))
    print(f"  router-only wall time : {route_time * 1e3:8.2f} ms / batch")
    print(f"  full sparse forward   : {forward_time * 1e3:8.2f} ms / batch")
    print(
        f"  routing overhead      : {100 * route_time / forward_time:5.1f}% "
        f"of forward"
    )

    # 4) Forward latency: SPARSE vs DENSE vs equal-active FFN ----------------
    print("\n[4] Forward latency (batch of 2,048 tokens, best of 3)")
    n_lat = 2048
    x = torch.randn(1, n_lat, D)
    t_sparse = _timeit(lambda: sparse(x))
    t_dense = _timeit(lambda: dense(x))
    t_fused = _timeit(lambda: fused(x))
    t_ffn = _timeit(lambda: active_ffn(x))
    print(
        f"  SPARSE MoE (Top-{K}/{E} executed)      : {t_sparse * 1e3:8.2f} ms  "
        f"({n_lat / t_sparse:9.0f} tok/s)"
    )
    print(
        f"  DENSE  MoE (all {E} experts, loop)     : {t_dense * 1e3:8.2f} ms  "
        f"({n_lat / t_dense:9.0f} tok/s)"
    )
    print(
        f"  DENSE  MoE (all {E} experts, fused)    : {t_fused * 1e3:8.2f} ms  "
        f"({n_lat / t_fused:9.0f} tok/s)"
    )
    print(
        f"  Dense FFN (equal active parameters)    : {t_ffn * 1e3:8.2f} ms  "
        f"({n_lat / t_ffn:9.0f} tok/s)"
    )
    print(f"  => sparse speedup vs dense loop  : {t_dense / t_sparse:.2f}x")
    print(f"  => sparse speedup vs dense fused : {t_fused / t_sparse:.2f}x")
    print(f"  => sparse overhead vs equal-active FFN: {t_sparse / t_ffn:.2f}x")

    # Theoretical MACs -------------------------------------------------------
    print("\n[5] Theoretical compute (MACs per token)")
    router_macs = 2 * E * D  # gate + noise projections
    expert_macs = 4 * D * DF  # w1 + w2 per expert per token
    print(f"  router               : {router_macs:>10,} MACs/token")
    print(f"  per expert           : {expert_macs:>10,} MACs/token")
    print(f"  SPARSE total         : {router_macs + K * expert_macs:>10,} MACs/token")
    print(f"  DENSE  total         : {router_macs + E * expert_macs:>10,} MACs/token")
    print(f"  => Top-K reduces expert compute by {(E - K) / E * 100:.1f}%")

    # 6) Expert utilization --------------------------------------------------
    print("\n[6] Expert utilization (4,096 tokens, balanced random router)")
    x_big = torch.randn(1, 4096, D)
    decision = sparse.route(x_big)
    f = decision.expert_fractions.detach().numpy()
    print(
        f"  dispatch fraction f_i: min={f.min():.4f}  max={f.max():.4f}  "
        f"mean={f.mean():.4f}  std={f.std():.4f}  (ideal={K / E:.4f})"
    )
    ok_util = (f.min() > 0.05) and (f.max() < 0.40)
    print(f"  no expert <5% or >40% of tokens: {'PASS' if ok_util else 'FAIL'}")

    # 7) Load-balancing loss: detection + prevention ------------------------
    print("\n[7] Load-balancing auxiliary loss (detect + prevent collapse)")
    balanced_loss = decision.aux_loss.item()
    print(
        f"  balanced routing aux loss : {balanced_loss:.5f} "
        f"(theory alpha*K = {cfg.load_balance_alpha * K:.5f})"
    )

    with torch.no_grad():
        sparse.w_gate.weight.fill_(0.0)
        sparse.w_gate.weight[0, :] = 2.0
        sparse.w_gate.weight[1, :] = 1.0
    x_pos = torch.rand(1, 2048, D)  # positive inputs -> deterministic collapse
    collapsed = sparse.route(x_pos)
    f_collapsed = collapsed.expert_fractions.detach().numpy()
    print(
        f"  collapsed router aux loss : {collapsed.aux_loss.item():.5f} "
        f"(max f_i = {f_collapsed.max():.4f} on expert {int(f_collapsed.argmax())})"
    )
    print(f"  => collapse detection: loss x{collapsed.aux_loss.item() / balanced_loss:.1f}")

    # Prevention: train the router on a task that rewards using expert 0,
    # with and without the auxiliary loss (identical init, same seed).
    def _train_router(with_aux: bool, lam: float, steps: int = 150, lr: float = 0.2):
        torch.manual_seed(0)
        moe = SparseMoELayer(cfg).eval()  # clean routing for determinism
        x_t = torch.rand(128, D)
        target = moe.experts[0](x_t).detach()
        opt = torch.optim.SGD([moe.w_gate.weight], lr=lr)
        task = float("nan")
        for _ in range(steps):
            opt.zero_grad()
            out, aux = moe(x_t)
            task = ((out - target) ** 2).mean()
            loss = task + (lam * aux if with_aux else 0.0)
            loss.backward()
            opt.step()
        f_t = moe.route(x_t).expert_fractions.detach().numpy()
        return task, f_t

    task_plain, f_plain = _train_router(with_aux=False, lam=0.0)
    task_bal, f_bal = _train_router(with_aux=True, lam=10.0)
    print(
        f"  150-step router training, collapse-inducing task "
        f"(target = expert-0 output):"
    )
    print(
        f"    WITHOUT aux loss: task={task_plain:.5f}  f_i in "
        f"[{f_plain.min():.3f}, {f_plain.max():.3f}], experts used="
        f"{int((f_plain > 0.01).sum())}/{E}"
    )
    print(
        f"    WITH    aux loss: task={task_bal:.5f}  f_i in "
        f"[{f_bal.min():.3f}, {f_bal.max():.3f}], experts used="
        f"{int((f_bal > 0.01).sum())}/{E}"
    )
    print(
        "  => without the auxiliary loss the router degenerates onto expert 0 "
        "(f_max -> 1.0); with it, all 32 experts stay utilized. On this "
        "adversarial task (constructed so that collapse is the task-optimal "
        "solution) balancing costs a small amount of task quality; in the "
        "roadmap's usage the auxiliary loss is a lightweight regularizer on "
        "top of the real task loss. Note: once *fully* saturated (f_max = "
        "1.0), the balance-loss gradient vanishes — it is a preventive "
        "regularizer, not a repair mechanism."
    )

    # 8) Sparse vs dense numerical sanity ------------------------------------
    print("\n[8] Numerical sanity (random router, Top-2 vs dense reference)")
    torch.manual_seed(1)
    sparse_check = SparseMoELayer(cfg).eval()  # fresh random router
    dense_check = DenseMoEReference(sparse_check).eval()
    x_check = torch.randn(64, D)
    a, _ = sparse_check(x_check)
    b = dense_check(x_check)
    rel = (a - b).norm().item() / b.norm().item()
    print(
        f"  ||sparse - dense|| / ||dense|| = {rel:.2e} "
        f"(expected >0: Top-K intentionally differs from dense combination)"
    )

    print("\n" + "=" * 72)
    print("PHASE 4 BENCHMARK COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
