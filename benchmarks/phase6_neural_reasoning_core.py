"""
Phase 6 Benchmark — Neural Reasoning Core (Latent Synthesis & Bounded Self-Correction).

Measures the Phase 6 Neural Reasoning Core per the roadmap and ARCHITECTURE.md
§4.5 (Neural Reasoning Core):

  1. Baseline vs reasoning: latency of the reasoning core disabled (baseline),
     one reasoning step, multiple reasoning steps, and self-correction enabled.
  2. Bounded termination: with an unreachable confidence threshold the loop
     always terminates at exactly K_r^max and correction_count <= max_corrections.
  3. Adaptive reasoning depth: with a reachable threshold, the loop halts early
     (converged=True) when confidence is sufficient, consuming fewer steps.
  4. Correction frequency: average correction count across configurations.
  5. Memory: parameter overhead of the reasoning core.
  6. Determinism: identical inputs produce identical refined states & diagnostics.

Run:
    python benchmarks/phase6_neural_reasoning_core.py

This script is deterministic (fixed seeds) and CPU-only. It validates the
structural, boundedness, and efficiency properties Phase 6 owns; the roadmap's
literal success criteria over a *trained* language model require the Phase 9
dataset pipeline and Phase 10 training — out of Phase 6 scope.
"""

import os
import sys
import tracemalloc
from time import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.set_num_threads(4)

from khwarizmi.config import KhwarizmiConfig
from khwarizmi.reasoning import NeuralReasoningCore

DEVICE = torch.device("cpu")


def _benchmark_config() -> KhwarizmiConfig:
    return KhwarizmiConfig(
        vocab_size=512,
        d_model=128,
        n_heads=4,
        d_expansion=16,
        d_ff=256,
        max_seq_len=128,
        min_recurrent_cycles=1,
        max_recurrent_cycles=3,
        halting_epsilon=0.05,
        ponder_cost_beta=0.01,
        min_reasoning_steps=1,
        max_reasoning_steps=6,
        reasoning_confidence_threshold=0.85,
        max_reasoning_corrections=3,
        reasoning_confidence_beta=0.01,
        reasoning_refinement_beta=0.01,
        enable_moe=True,
        dropout=0.0,
        tier_name="ReasoningCore-Benchmark",
    )


def _timeit(fn, repeats: int = 5, warmup: int = 2) -> float:
    """Best-of-`repeats` wall time in seconds after `warmup` untimed calls."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time()
        fn()
        best = min(best, time() - t0)
    return best


def _param_count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _peak_alloc_mb(fn) -> float:
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024.0 * 1024.0)


def main() -> None:
    print("=" * 72)
    print("PHASE 6 BENCHMARK — NEURAL REASONING CORE")
    print("  Latent Synthesis & Bounded Self-Correction")
    print("=" * 72)

    cfg = _benchmark_config()
    torch.manual_seed(0)
    core = NeuralReasoningCore(cfg).eval()

    B, L = 16, 32
    x = torch.randn(B, L, cfg.d_model, device=DEVICE)

    print(f"\nConfig: d_model={cfg.d_model}, "
          f"K_r^min={cfg.min_reasoning_steps}, "
          f"K_r^max={cfg.max_reasoning_steps}, "
          f"max_corrections={cfg.max_reasoning_corrections}, "
          f"threshold={cfg.reasoning_confidence_threshold}")
    print(f"Input: batch={B}, seq={L}, tokens={B * L}")
    print(f"Reasoning core parameters: {_param_count(core):,}")

    # 1) Baseline vs reasoning latency -------------------------------------
    print("\n[1] Latency: baseline (no reasoning) vs reasoning steps")

    def baseline() -> None:
        # Identity-style baseline: a single LayerNorm pass (no reasoning loop).
        torch.nn.functional.layer_norm(
            x, (cfg.d_model,), torch.zeros(cfg.d_model),
            torch.ones(cfg.d_model), 1e-5,
        )

    def one_step() -> None:
        with torch.no_grad():
            core(x, force_steps=1)

    def multi_step() -> None:
        with torch.no_grad():
            core(x, force_steps=cfg.max_reasoning_steps)

    t_base = _timeit(baseline)
    t_one = _timeit(one_step)
    t_multi = _timeit(multi_step)
    print(f"  baseline (single norm):        {t_base * 1e3:>8.3f} ms")
    print(f"  1 reasoning step:              {t_one * 1e3:>8.3f} ms  "
          f"(overhead {(t_one - t_base) * 1e3:.3f} ms)")
    print(f"  {cfg.max_reasoning_steps} reasoning steps (forced):    "
          f"{t_multi * 1e3:>8.3f} ms  "
          f"(overhead {(t_multi - t_base) * 1e3:.3f} ms)")
    # Bounded overhead: multi-step should be a modest multiple of one-step.
    assert t_multi >= t_one, "multi-step must not be faster than one-step"

    # 2) Self-correction overhead ------------------------------------------
    print("\n[2] Self-correction overhead (corrections enabled vs disabled)")

    def no_correction() -> None:
        with torch.no_grad():
            core(x, force_steps=cfg.max_reasoning_steps,
                 max_corrections=0)

    def with_correction() -> None:
        with torch.no_grad():
            core(x, force_steps=cfg.max_reasoning_steps,
                 max_corrections=cfg.max_reasoning_corrections)

    t_no_corr = _timeit(no_correction)
    t_corr = _timeit(with_correction)
    with torch.no_grad():
        out_corr = core(x, force_steps=cfg.max_reasoning_steps,
                        max_corrections=cfg.max_reasoning_corrections)
    print(f"  corrections disabled (K_c=0):  {t_no_corr * 1e3:>8.3f} ms")
    print(f"  corrections enabled  (K_c={cfg.max_reasoning_corrections}):   "
          f"{t_corr * 1e3:>8.3f} ms  "
          f"(overhead {(t_corr - t_no_corr) * 1e3:.3f} ms)")
    print(f"  correction count (enabled):    "
          f"{out_corr.diagnostics['correction_count']}")

    # 3) Bounded termination -----------------------------------------------
    print("\n[3] Bounded termination (unreachable threshold)")
    with torch.no_grad():
        out = core(x, confidence_threshold=2.0)
    print(f"  reasoning_steps = {out.diagnostics['reasoning_steps']} "
          f"(hard cap K_r^max = {cfg.max_reasoning_steps})")
    print(f"  correction_count = {out.diagnostics['correction_count']} "
          f"(<= max_corrections = {cfg.max_reasoning_corrections})")
    print(f"  converged = {out.diagnostics['converged']}")
    assert out.diagnostics["reasoning_steps"] == cfg.max_reasoning_steps, \
        "loop must run to K_r^max under unreachable threshold"
    assert out.diagnostics["correction_count"] <= cfg.max_reasoning_corrections

    # 4) Adaptive reasoning depth (early halt) -----------------------------
    print("\n[4] Adaptive reasoning depth (reachable threshold)")
    with torch.no_grad():
        out_low = core(x, confidence_threshold=0.0, min_steps=1)
    print(f"  threshold=0.0 (always sufficient): "
          f"steps={out_low.diagnostics['reasoning_steps']}, "
          f"converged={out_low.diagnostics['converged']}, "
          f"confidence={out_low.diagnostics['confidence']:.4f}")
    assert out_low.diagnostics["converged"], "must converge at threshold 0.0"
    assert out_low.diagnostics["reasoning_steps"] == cfg.min_reasoning_steps

    # 5) Average reasoning iterations & correction frequency ---------------
    print("\n[5] Average reasoning iterations & correction frequency")
    torch.manual_seed(1)
    xs = [torch.randn(B, L, cfg.d_model) for _ in range(5)]
    steps, corrs, confs = [], [], []
    with torch.no_grad():
        for xi in xs:
            o = core(xi)
            steps.append(o.diagnostics["reasoning_steps"])
            corrs.append(o.diagnostics["correction_count"])
            confs.append(o.diagnostics["confidence"])
    print(f"  avg reasoning steps:   {sum(steps) / len(steps):.3f}")
    print(f"  avg correction count:   {sum(corrs) / len(corrs):.3f}")
    print(f"  avg final confidence:   {sum(confs) / len(confs):.4f}")
    for s in steps:
        assert s <= cfg.max_reasoning_steps, "unbounded step count"

    # 6) Determinism -------------------------------------------------------
    print("\n[6] Determinism (identical inputs => identical outputs)")
    with torch.no_grad():
        o1 = core(x, force_steps=4)
        o2 = core(x, force_steps=4)
    same = torch.allclose(o1.refined_state, o2.refined_state)
    same_diag = (o1.diagnostics["reasoning_steps"]
                 == o2.diagnostics["reasoning_steps"])
    print(f"  identical refined state: {same}")
    print(f"  identical diagnostics:   {same_diag}")
    assert same and same_diag, "reasoning must be deterministic"

    # 7) Memory footprint --------------------------------------------------
    print("\n[7] Memory footprint (peak traced allocation, single forward)")
    alloc = _peak_alloc_mb(lambda: core(x, force_steps=cfg.max_reasoning_steps))
    print(f"  peak traced allocation: {alloc:.4f} MB")

    # 8) Stability under stress -------------------------------------------
    print("\n[8] Stability under stress (extreme & repeated refinement)")
    with torch.no_grad():
        out_stress = core(torch.randn(B, L, cfg.d_model) * 1e3,
                          force_steps=cfg.max_reasoning_steps)
    finite = bool(torch.isfinite(out_stress.refined_state).all().item())
    print(f"  extreme-input forward finite: {finite}")
    assert finite, "extreme inputs must not produce NaN/Inf"

    print("\n" + "=" * 72)
    print("PHASE 6 BENCHMARK COMPLETE — reasoning bounded, deterministic, stable")
    print("=" * 72)


if __name__ == "__main__":
    main()
