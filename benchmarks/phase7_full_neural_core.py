"""
Phase 7 Benchmark — Full Khwarizmi Neural Core Integration.

Measures the unified forward computation graph that integrates every existing
subsystem (KSC -> Sparse MoE -> Dual Memory -> ARRC -> Neural Reasoning Core ->
Output) per the Phase 7 roadmap ("Full Khwarizmi Neural Core Integration").

Benchmark surface (per Phase 7 spec):
  1. Baseline model path (pre-Phase-6: reasoning core disabled).
  2. Full neural core enabled (all subsystems integrated).
  3. KSC contribution (KSC-only configuration).
  4. MoE contribution (MoE-on vs MoE-off).
  5. ARRC contribution (adaptive compute on vs off).
  6. Reasoning contribution (reasoning core on vs off).
  7. Complete integrated path (full core, eval, latency + memory).

Reports:
  * latency (best-of-N, ms)
  * relative overhead vs the baseline / disabled variant
  * parameter count
  * peak traced allocation (MB) where practical

The benchmark is deterministic (fixed seeds), CPU-only, and reproducible. It
validates the structural and efficiency properties Phase 7 owns; the roadmap's
literal success criteria over a *trained* model require Phase 8+ (trainer) and
Phase 10 (training) — out of Phase 7 scope.

Run:
    python benchmarks/phase7_full_neural_core.py
"""

import os
import sys
import tracemalloc
from time import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.set_num_threads(4)

from khwarizmi.config import KhwarizmiConfig
from khwarizmi.core import KhwarizmiModel

DEVICE = torch.device("cpu")


def _benchmark_config(**overrides) -> KhwarizmiConfig:
    base = dict(
        vocab_size=512,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_expansion=16,
        d_ff=256,
        num_experts=4,
        top_k_experts=2,
        moe_frequency=2,
        max_seq_len=128,
        min_recurrent_cycles=1,
        max_recurrent_cycles=3,
        halting_epsilon=0.05,
        ponder_cost_beta=0.01,
        min_reasoning_steps=1,
        max_reasoning_steps=4,
        reasoning_confidence_threshold=0.85,
        max_reasoning_corrections=2,
        reasoning_confidence_beta=0.01,
        reasoning_refinement_beta=0.01,
        memory_dim=128,
        memory_slots=32,
        short_term_capacity=128,
        enable_moe=True,
        enable_adaptive_compute=True,
        enable_reasoning_core=True,
        enable_full_neural_core=True,
        dropout=0.0,
        tier_name="Phase7-FullCore-Benchmark",
    )
    base.update(overrides)
    return KhwarizmiConfig(**base)


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


def _make_model(cfg: KhwarizmiConfig, seed: int = 0) -> KhwarizmiModel:
    torch.manual_seed(seed)
    m = KhwarizmiModel(cfg).eval()
    return m


def main() -> None:
    print("=" * 76)
    print("PHASE 7 BENCHMARK — FULL KHWARIZMI NEURAL CORE INTEGRATION")
    print("  KSC -> Sparse MoE -> Dual Memory -> ARRC -> Neural Reasoning -> Output")
    print("=" * 76)

    B, L = 8, 32
    cfg_full = _benchmark_config()
    print(f"\nConfig: d_model={cfg_full.d_model}, n_layers={cfg_full.n_layers}, "
          f"experts={cfg_full.num_experts}x top-{cfg_full.top_k_experts}, "
          f"ARRC K_max={cfg_full.max_recurrent_cycles}, "
          f"reasoning K_r^max={cfg_full.max_reasoning_steps}")
    print(f"Input: batch={B}, seq={L}, tokens={B * L}")

    torch.manual_seed(0)
    ids = torch.randint(0, cfg_full.vocab_size, (B, L), device=DEVICE)

    # ------------------------------------------------------------------ #
    # 1) Baseline vs full neural core                                    #
    # ------------------------------------------------------------------ #
    print("\n[1] Latency: baseline (pre-Phase-6, reasoning off) vs full core")
    cfg_base = _benchmark_config(enable_reasoning_core=False,
                                 enable_full_neural_core=False)
    m_base = _make_model(cfg_base)
    m_full = _make_model(cfg_full)

    def run_base() -> None:
        with torch.no_grad():
            m_base(ids)

    def run_full() -> None:
        with torch.no_grad():
            m_full(ids)

    t_base = _timeit(run_base)
    t_full = _timeit(run_full)
    overhead = (t_full - t_base) * 1e3
    print(f"  baseline (reasoning off):   {t_base * 1e3:>8.3f} ms")
    print(f"  full neural core:           {t_full * 1e3:>8.3f} ms  "
          f"(overhead {overhead:.3f} ms, "
          f"{(t_full / t_base):.2f}x)")
    assert t_full >= t_base, "full core must not be faster than baseline"

    # ------------------------------------------------------------------ #
    # 2) KSC contribution (KSC-only vs full)                             #
    # ------------------------------------------------------------------ #
    print("\n[2] KSC contribution (KSC-only config vs full core)")
    cfg_ksc = _benchmark_config(enable_moe=False, enable_adaptive_compute=False,
                               enable_reasoning_core=False,
                               enable_full_neural_core=False)
    m_ksc = _make_model(cfg_ksc)

    def run_ksc() -> None:
        with torch.no_grad():
            m_ksc(ids)

    t_ksc = _timeit(run_ksc)
    print(f"  KSC-only:                   {t_ksc * 1e3:>8.3f} ms")
    print(f"  full core:                  {t_full * 1e3:>8.3f} ms  "
          f"(KSC is {100 * t_ksc / t_full:.1f}% of full-core latency)")
    assert t_ksc <= t_full, "KSC-only must not exceed full core"

    # ------------------------------------------------------------------ #
    # 3) MoE contribution (MoE on vs MoE off)                            #
    # ------------------------------------------------------------------ #
    print("\n[3] Sparse MoE contribution (MoE on vs off, other subsystems on)")
    cfg_no_moe = _benchmark_config(enable_moe=False)
    m_no_moe = _make_model(cfg_no_moe)

    def run_no_moe() -> None:
        with torch.no_grad():
            m_no_moe(ids)

    t_no_moe = _timeit(run_no_moe)
    print(f"  MoE off:                    {t_no_moe * 1e3:>8.3f} ms")
    print(f"  MoE on:                     {t_full * 1e3:>8.3f} ms  "
          f"(MoE overhead {(t_full - t_no_moe) * 1e3:.3f} ms)")
    assert t_full >= t_no_moe, "MoE-on must not be faster than MoE-off"

    # ------------------------------------------------------------------ #
    # 4) ARRC contribution (adaptive compute on vs off)                  #
    # ------------------------------------------------------------------ #
    print("\n[4] ARRC contribution (adaptive compute on vs off)")
    cfg_no_arrc = _benchmark_config(enable_adaptive_compute=False)
    m_no_arrc = _make_model(cfg_no_arrc)

    def run_no_arrc() -> None:
        with torch.no_grad():
            m_no_arrc(ids)

    t_no_arrc = _timeit(run_no_arrc)
    print(f"  ARRC off:                   {t_no_arrc * 1e3:>8.3f} ms")
    print(f"  ARRC on:                    {t_full * 1e3:>8.3f} ms  "
          f"(ARRC overhead {(t_full - t_no_arrc) * 1e3:.3f} ms)")
    assert t_full >= t_no_arrc, "ARRC-on must not be faster than ARRC-off"

    # ------------------------------------------------------------------ #
    # 5) Reasoning contribution (reasoning core on vs off)               #
    # ------------------------------------------------------------------ #
    print("\n[5] Neural Reasoning contribution (reasoning core on vs off)")
    cfg_no_reason = _benchmark_config(enable_reasoning_core=False)
    m_no_reason = _make_model(cfg_no_reason)

    def run_no_reason() -> None:
        with torch.no_grad():
            m_no_reason(ids)

    t_no_reason = _timeit(run_no_reason)
    print(f"  reasoning off:              {t_no_reason * 1e3:>8.3f} ms")
    print(f"  reasoning on:               {t_full * 1e3:>8.3f} ms  "
          f"(reasoning overhead {(t_full - t_no_reason) * 1e3:.3f} ms)")
    assert t_full >= t_no_reason, "reasoning-on must not be faster than off"

    # ------------------------------------------------------------------ #
    # 6) Complete integrated path (determinism + full-core diagnostics)  #
    # ------------------------------------------------------------------ #
    print("\n[6] Complete integrated path — determinism & diagnostics")
    with torch.no_grad():
        o1 = m_full(ids)
        o2 = m_full(ids)
    same_logits = torch.allclose(o1.logits, o2.logits)
    fc = o1.diagnostics["full_core"]
    print(f"  deterministic logits:       {same_logits}")
    print(f"  components integrated:       {fc['full_core']['components_integrated']}")
    print(f"  KSC layers / post norm:      {fc['ksc']['n_layers']} / "
          f"{fc['ksc']['post_norm']:.4f}")
    print(f"  MoE experts executed:       {fc['moe']['experts_executed_last']}")
    print(f"  memory read/write active:    {fc['memory']['read_active']} / "
          f"{fc['memory']['write_active']}")
    print(f"  ARRC mean cycles:            {fc['arrc']['mean_cycles']:.4f}")
    print(f"  reasoning steps / corrected: {fc['reasoning']['reasoning_steps']} / "
          f"{fc['reasoning']['correction_count']}")
    print(f"  total aux loss:              {fc['full_core']['total_aux_loss']:.6f}")
    assert same_logits, "full core must be deterministic in eval mode"

    # ------------------------------------------------------------------ #
    # 7) Parameter counts                                               #
    # ------------------------------------------------------------------ #
    print("\n[7] Parameter counts")
    print(f"  full neural core:           {_param_count(m_full):>10,}")
    print(f"  baseline (reasoning off):    {_param_count(m_base):>10,}")
    print(f"  KSC-only:                    {_param_count(m_ksc):>10,}")
    print(f"  MoE-off:                      {_param_count(m_no_moe):>10,}")
    print(f"  ARRC-off:                     {_param_count(m_no_arrc):>10,}")
    print(f"  reasoning-off:                {_param_count(m_no_reason):>10,}")
    print(f"  full-core memory footprint:  "
          f"{m_full.get_memory_footprint_mb():.4f} MB")

    # ------------------------------------------------------------------ #
    # 8) Memory footprint (peak traced allocation)                       #
    # ------------------------------------------------------------------ #
    print("\n[8] Memory footprint (peak traced allocation, single forward)")
    alloc_full = _peak_alloc_mb(lambda: m_full(ids))
    alloc_base = _peak_alloc_mb(lambda: m_base(ids))
    print(f"  full core peak alloc:        {alloc_full:.4f} MB")
    print(f"  baseline peak alloc:         {alloc_base:.4f} MB")
    print(f"  allocation overhead:         "
          f"{(alloc_full - alloc_base):.4f} MB")

    # ------------------------------------------------------------------ #
    # 9) Stability under the full integrated path                        #
    # ------------------------------------------------------------------ #
    print("\n[9] Stability under stress (extreme inputs through the full core)")
    with torch.no_grad():
        extreme = torch.full((B, L), cfg_full.vocab_size - 1, dtype=torch.long)
        o_ext = m_full(extreme)
    finite = bool(torch.isfinite(o_ext.logits).all().item())
    print(f"  extreme-input forward finite: {finite}")
    assert finite, "extreme inputs must not produce NaN/Inf through full core"

    # ------------------------------------------------------------------ #
    # 10) Repeated recurrent execution (no memory leak / NaN growth)     #
    # ------------------------------------------------------------------ #
    print("\n[10] Repeated recurrent execution (15 sequential steps)")
    st, lt = None, None
    last_finite = True
    for step in range(1, 16):
        with torch.no_grad():
            o = m_full(ids, short_term_state=st, long_term_table=lt,
                       step_counter=step)
        st, lt = o.short_term_state, o.long_term_table
        if not bool(torch.isfinite(o.logits).all().item()):
            last_finite = False
            break
    print(f"  15-step forward finite:     {last_finite}")
    assert last_finite, "repeated execution must not accumulate NaN/Inf"

    print("\n" + "=" * 76)
    print("PHASE 7 BENCHMARK COMPLETE — full core integrated, bounded, "
          "deterministic, stable")
    print("=" * 76)


if __name__ == "__main__":
    main()
