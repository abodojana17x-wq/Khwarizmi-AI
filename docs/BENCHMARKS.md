# Khwarizmi AI — Evaluation & Benchmarks

> Create benchmarks **before scaling**. Measure intelligence, project intelligence, efficiency,
> memory, and adaptive compute.

## 1. Intelligence
- **Reasoning:** GSM8K-style (EN), synthetic multi-step logic, AR-translated reasoning.
- **Mathematics:** grade-school + competition-lite (MATH-lite), symbolic.
- **Coding:** HumanEval / MBPP (EN), plus Python-correctness via the Python Brain verifier.
- **Language understanding:** EN + AR + Egyptian + Franco comprehension (small, curated).
- **Instruction following:** IFEval-style (adapted, offline).

## 2. Project Intelligence (primary specialization)
- **Planning:** decompose a stated goal into correct task graph.
- **Task decomposition:** coverage + correctness of subtasks.
- **Dependency reasoning:** valid topological order; detect cycles/conflicts.
- **Long-horizon consistency:** across sessions, respect earlier constraints/decisions.
- **Project memory:** recall decisions/failures from Long-Term Memory.
- **Failure recovery:** given a failed task, produce a valid replan (use existing planner).
- **Replanning:** adapt when a constraint changes mid-project.

## 3. Efficiency (offline targets)
- **tokens/sec** (CPU + optional GPU offload).
- **first-token latency** (time-to-first-byte of response).
- **RAM** (RSS) and **VRAM** at rest and under load.
- **model size** on disk (quantized).
- **CPU usage / energy** where measurable (RAPL / powermetrics).

## 4. Memory quality
- retrieval accuracy @k; retention under distractors; forgetting rate of stale/contradicted
  items; long-project consistency; write precision (see `MEMORY.md`).

## 5. Adaptive Compute
- Compare **Fixed Compute** vs **Adaptive Compute** on:
  - average tokens/request, p50/p95 latency, accuracy by difficulty bucket.
  - Does adaptive save on easy requests without hurting hard ones?

## 6. Harness
- All benchmarks run **locally/offline**; results stored in `experiments/`.
- Deterministic seeds; versioned datasets; leaderboard table per phase.
- The **Project Planner** (kept tool) is used to generate ground-truth task graphs for the
  project-intelligence evals.

## 7. Reporting
Each phase ends with a benchmark table: metric | baseline | +component | full. Color-coded
pass/fail vs the phase's pre-registered success criteria.
