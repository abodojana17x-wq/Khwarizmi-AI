# Khwarizmi AI — Experiments & Ablation Plan

> Ablation is **mandatory**. No component ships unless it earns its cost on measured metrics.

## 1. Mandated component ablation (run in Phase 11, and per-component in Phases 2–6)

| ID | Configuration | What it isolates |
|---|---|---|
| A0 | **Baseline** (dense Transformer, same param budget) | Reference quality/cost |
| A1 | Baseline + **KSC** (replace attention with KSC layers) | Recurrent state value |
| A2 | Baseline + **Memory** (LTM controller) | Long-term memory value |
| A3 | Baseline + **Experts** (MoE FFN) | Sparse MoE value |
| A4 | Baseline + **Adaptive Compute** (early-exit + loops) | Adaptive compute value |
| A5 | **Full Khwarizmi** (KSC + Memory + Experts? + Adaptive + Router + Reasoning) | Whole-system |

For each configuration, record: **quality, latency, RAM, VRAM, training stability,
parameter count, active parameters**.

**Decision rule:** if a component's delta-quality < threshold AND its cost (latency/RAM/
complexity) > 0, **remove or redesign it.** MoE removal is explicitly allowed (Phase 4).

## 2. KSC internal ablations (Phase 2–3)
- **KSC−surprise:** fix `σₜ = 1` (no Titans gate). Does surprise gating help?
- **KSC−erase:** tie erase to write (scalar gate, like Gated DeltaNet). Does decoupled erase help?
- **KSC−decay:** fixed decay `λ`. Does data-dependent decay help?
- **KSC−delta:** plain additive memory `S += k⊗v`. Does delta rule help?
- **KSC+SSM-vector:** add vector `hₜ`. Net gain vs cost?

## 3. Adaptive compute ablations (Phase 5)
- Fixed compute (all tokens full depth, no loops) vs Adaptive.
- Early-exit only vs loops only vs both.
- Threshold sweep: accuracy vs tokens/latency Pareto.

## 4. Memory ablations (Phase 3)
- No memory vs STM-only vs STM+LTM.
- Surprise-gated write vs always-write vs random-write.
- k-NN read vs no read.

## 5. Experiment tracking (minimum)
Every run logs: config hash, seed, dataset version, steps, loss curves, eval metrics,
hardware, wall-clock, peak RAM. Stored locally (offline). Reproducible from config.

## 6. Statistical discipline
- ≥3 seeds per comparison; report mean ± std.
- Same tokenizer, same eval set, same budget across ablations.
- Pre-register the decision thresholds (e.g., "MoE kept only if +≥2% on coding at equal
  active FLOPs") before reading results.
