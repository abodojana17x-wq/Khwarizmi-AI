# Khwarizmi AI — 16-Phase Roadmap

> Critical development rule: **never advance just because code runs.** A phase is complete
> only when its success criteria are met. If a benchmark fails: STOP → analyze → modify →
> retest → only then continue.

Each phase lists: **Objective · Deliverables · Files/Modules · Dependencies · Tests ·
Benchmarks · Success · Failure · Exit · NOT-yet.**

## Phase 0 — Repository Audit + Architecture Reset ✅ (this document set)
- **Objective:** Audit repo; design clean offline architecture; produce blueprint.
- **Deliverables:** ARCHITECTURE, RESEARCH, AUDIT, MEMORY, ROADMAP, EXPERIMENTS, BENCHMARKS,
  TRAINING, DATA, REPOSITORY_STRUCTURE, RISKS, DEPLOYMENT, CONTRIBUTING, BLUEPRINT.
- **Success:** Every existing component has a decision; KSC math drafted; roadmap defined.
- **Exit:** Begin Phase 1.

## Phase 1 — Mathematical Specification
- **Objective:** Lock KSC + router + memory + adaptive-compute math into a reference impl.
- **Deliverables:** `khwarizmi/neural/ksc.py` (un-trained, reference + chunk-parallel),
  `router.py`, `memory_controller.py`, loss definitions, numerical unit tests.
- **Dependencies:** PyTorch (dev), numpy.
- **Tests:** recurrence matches a slow reference; shapes/param counts correct; dwell/dtype
  checks; loss gradients flow.
- **Benchmarks:** (none yet) — establish baseline comparison harness (Transformer, Mamba-2,
  RWKV-6, Gated DeltaNet) on a tiny synthetic recall task.
- **Success:** Math is unambiguous; reference impl numerically correct; baseline harness runs.
- **Failure:** Ambiguous equations; reference mismatch.
- **Exit:** Equations + reference pass review.
- **NOT-yet:** No real training, no datasets, no large model.

## Phase 2 — Minimal KSC Prototype
- **Objective:** Train a 50M prototype; validate KSC vs baselines on small tasks.
- **Deliverables:** training loop, 50M KSC, baseline models, logging.
- **Benchmarks:** synthetic recall (MQAR-style), tiny LM perplexity, ablations.
- **Success:** KSC ≥ baselines on recall at equal params; trains stably on Colab.
- **Failure:** KSC worse and not fixable by tuning; instability.
- **Exit:** KSC validated or redesigned.

## Phase 3 — Memory Prototype
- **Objective:** Implement short-term + long-term memory; validate selective memory.
- **Benchmarks:** retrieval accuracy, retention-under-distractors, long-project consistency.
- **Success:** Memory improves long-context tasks vs no-memory; forgetting works.
- **Failure:** Memory hurts or adds latency without benefit → simplify.

## Phase 4 — Sparse Expert Prototype
- **Objective:** Add MoE FFN; measure quality/active-params/latency.
- **Success:** >X% gain at fixed active compute, stable training.
- **Failure:** <X% gain or unstable → **REMOVE MoE** (dense KSC).

## Phase 5 — Adaptive Compute
- **Objective:** Early-exit heads + recurrent reasoning loops + confidence halt.
- **Benchmarks:** Fixed vs Adaptive (tokens, latency, accuracy).
- **Success:** Equal/better accuracy at lower avg tokens.
- **Failure:** Hurts hard tasks or adds overhead → tune or drop loops.

## Phase 6 — Neural Reasoning
- **Objective:** Reasoning controller (decompose→plan→self-check→revise→synthesize).
- **Benchmarks:** reasoning/planning evals; measure vs "just longer CoT".
- **Success:** Structured reasoning beats longer-CoT baseline on hard tasks.
- **Failure:** No improvement → keep simpler recurrent loop.

## Phase 7 — Full Khwarizmi Neural Core
- **Objective:** Integrate KSC + memory + (maybe) experts + adaptive + reasoning + router.
- **Success:** Full core passes all unit + integration tests; clean interfaces.
- **Exit:** Core frozen for training experiments.

## Phase 8 — Training Infrastructure
- **Objective:** Reproducible, low-cost training (Colab-friendly), checkpointing, logging,
  distributed-optional.
- **Success:** Can train 50M→700M reproducibly; resume/ablate from config.

## Phase 9 — Dataset Pipeline
- **Objective:** Quality>quantity pipeline (EN/AR/Egyptian/code/math/reasoning/planning/
  tools/verify), dedup, contamination/leakage controls.
- **Success:** Clean, documented, versioned datasets; leakage report.

## Phase 10 — Small Model Training
- **Objective:** Train 300M–700M with the full recipe; distillation where useful.
- **Success:** Meets Phase 11 benchmarks at target size; reasoning emerges.

## Phase 11 — Evaluation + Ablation
- **Objective:** Run full benchmark + mandated ablations:
  Baseline · +KSC · +Memory · +Experts · +Adaptive · Full.
- **Success:** Each component's contribution quantified (quality/latency/RAM/params).
- **Failure:** A component fails → remove/redesign per its phase rules.

## Phase 12 — Optimization
- **Objective:** KV-free inference speed, kernel fusion, quantization (INT8/INT4), CPU tuning.
- **Success:** Edge-tier latency/RAM targets met; quality within tolerance of FP.

## Phase 13 — Offline Agent + Local Tools
- **Objective:** Wire the agent layer: router → core → tools (Project Planner, Python
  Analysis, Symbolic Verify, Safe Exec), all offline.
- **Success:** Tools invoked only when router flags; end-to-end offline demo.

## Phase 14 — Project Intelligence
- **Objective:** Long-horizon project memory, planning, replanning, failure recovery.
- **Benchmarks:** long-horizon project consistency, replanning, failure recovery.
- **Success:** Coherent multi-session project management.

## Phase 15 — Edge Deployment
- **Objective:** Package for CPU/edge; custom C runtime or ONNX; GGUF *if compatible*;
  quantization; Android/edge investigation.
- **Success:** Runs offline on target edge device within RAM/latency budget.

## Phase 16 — Scaling + Stable Release
- **Objective:** Scale to 5B–10B **only if benchmarks justify**; stabilize; document; release.
- **Success:** Measured improvement at scale; stable offline release artifact.
- **Rule:** Scale only on evidence. If 700M meets targets, do not force 10B.

---

## Phase 1 Checklist (immediate, detailed)

- [ ] **KSC equations finalized** in `khwarizmi/neural/ksc.py` docstring (decay, read, erase,
      delta, surprise write, local conv, gated MLP), with tensor shapes and param counts.
- [ ] **Reference (slow, looping) implementation** of the KSC recurrence (numerical ground truth).
- [ ] **Chunk-parallel implementation** (training scan) that matches the reference within tolerance.
- [ ] **Cognitive Router** spec + tiny reference impl (route decision, flags).
- [ ] **Memory Controller** interface + surprise-gated write stub.
- [ ] **Adaptive compute** spec: early-exit head shape, halt probability, budget loop.
- [ ] **Loss definitions:** LM cross-entropy + MoE aux load-balance + (optional) memory/
      router policy losses.
- [ ] **Baseline models** implemented or wrapped for comparison: Transformer, Mamba-2 (SSD),
      RWKV-6, Gated DeltaNet.
- [ ] **Numerical unit tests:** reference vs chunk match; shape checks; dtype/stability;
      gradient flow; zero-state determinism.
- [ ] **Config schema** for tiers (50M/300–700M/1–3B/5–10B) with dims, heads, state size.
- [ ] **Documentation:** this math locked in `ARCHITECTURE.md` §4; review sign-off.
- [ ] **Do NOT:** train on real data, build datasets, implement tools, or scale.

## Immediate Next Action

> **Start Phase 1.** Implement `khwarizmi/neural/ksc.py` as an **un-trained reference +
> chunk-parallel** module with numerical unit tests proving the recurrence is correct, and
> stand up the **baseline comparison harness** (Transformer / Mamba-2 / RWKV-6 / Gated
> DeltaNet) on a tiny synthetic associative-recall task. No training, no datasets, no scaling.
> Report the math + tests for review before any parameter is updated by gradient descent.
