# Khwarizmi AI — Phase 0 Blueprint (Architecture Reset)

> **This is the Phase 0 deliverable.** It is a complete, technically justified blueprint for
> a new, clean, offline-first, reasoning-focused AI architecture. **Nothing is implemented
> yet beyond planning docs.** After this, the project is ready for **Phase 1: Mathematical
> Specification**.
>
> Companion docs live in `docs/`. This file consolidates the 16 required Phase-0 outputs.

---

## 1. Executive Architecture Summary

Khwarizmi is a layered, fully-offline, reasoning-centric assistant. Its neural core is built
around a novel **Khwarizmi State Cell (KSC)** — a single recurrence that fuses (a) a
**delta-rule matrix associative memory** (precise recall), (b) **channel-wise decay +
decoupled erase/write gates** (controllable forgetting), and (c) a **surprise/importance
gate** (memorize what violates expectation). On top sit a cheap **Cognitive Router**, optional
**Sparse Experts**, **Adaptive Compute** (early-exit + recurrent reasoning loops), a **Dual
Memory** system, and a **Neural Reasoning** controller. An **Offline Agent** layer may invoke
**local deterministic tools** (the existing symbolic Project Planner and Python Brain) *only*
when the router decides they help.

Objective: **maximum intelligence and reasoning per unit of compute and memory**, specialized
in complex reasoning, coding, and long-horizon project intelligence — on CPU / consumer GPU /
edge hardware, with zero network dependency.

---

## 2. Full Architecture Specification

See **`docs/ARCHITECTURE.md`** for the complete spec: layered diagram, KSC math (§4), router,
MoE, adaptive compute, dual memory, neural reasoning, interfaces, and the "what KSC is NOT"
statement. Key points:

- **KSC recurrence** (per head): decay `S̃=diag(λ)S`, read `r=S̃ᵀq`, erase `Ŝ=S̃−diag(ε)(S̃ᵀk)kᵀ`,
  delta `δ=v−Ŝᵀq`, surprise-gated write `S=Ŝ+(βσ)k⊗δ`, output `ŷ=g·MLP(x̃)+(1−g)·(Sᵀq)`.
- **Constant per-token cost** O(d_k·d_v), **no KV cache**, context-length-independent latency.
- **Offline-first, modular, measurable, experiment-driven.**

---

## 3. Existing Repository Audit

See **`docs/AUDIT.md`**. Critical finding: the current `rafig/` (RAFIQ) repo is a
**deterministic, rule-based, symbolic engine with zero neural components** — offline by
design (stdlib only), 141 tests passing. It is preserved **as the tool/agent layer**, not as
the neural core.

Decision highlights:
- **REPLACE:** char-level tokenizer (unsuitable for a neural LLM).
- **REPLACE:** rule-based intent/semantic mapping (neural takes over understanding).
- **EXTERNAL TOOL (KEEP):** the symbolic **Project Planner** (`rafig/reasoning/*`) and the
  **Python Brain** (`rafig/python_brain/*`) → offline planning/verification tools.
- **KEEP:** config, paths, diagnostics, offline principle, tests.
- **REFACTOR:** `rafig.py`/app/main → agent/runtime bootstrap.

---

## 4. Old Component Decisions

| Component | Decision | Future location |
|---|---|---|
| Rafiq core | REFACTOR | `khwarizmi/runtime`,`agent` |
| config / paths | KEEP | `khwarizmi/config.py`,`common/paths.py` |
| app / main | REFACTOR | `khwarizmi/cli.py` |
| tokenizer (char) | REPLACE | `khwarizmi/neural/tokenizer` |
| language_understanding | MOVE→router feature | `khwarizmi/agent/langid.py` |
| semantic_representation | REPLACE | neural post-processor (optional) |
| reasoning/* | EXTERNAL TOOL | `khwarizmi/tools/project_planner`,`symbolic_verify` |
| python_brain/* | KEEP+TOOL | `khwarizmi/tools/python_analysis` |
| tests | KEEP+EXTEND | `tests/neural`,`tests/tools` |
| requirements | REPLACE(split) | `requirements*.txt` |

Full reasoning in `docs/AUDIT.md` §2.

---

## 5. Research Comparison

See **`docs/RESEARCH.md`**. Techniques reviewed with **[FACT]/[RF]/[HYP]/[DD]** labels:
Mamba/Mamba-2, RWKV, xLSTM, DeltaNet/Gated DeltaNet/Gated DeltaNet-2, Titans, Liquid NN,
Sparse MoE, Adaptive Compute/Early-exit, Quantization/GGUF/llama.cpp, and frontier
(GPT/Claude/Kimi) principles only.

**Adoption:** adopt selective decay, delta-rule matrix memory, decoupled erase/write, surprise
gating, local conv, early-exit/learned halting, recurrent reasoning, quantization. **Reject**
Liquid ODE dynamics for the core; **reject** unbounded attention in the core; **conditional**
on MoE (ablation-gated); **no** cloud inference. We study these and **do not copy them** — KSC
is a distinct synthesis.

---

## 6. Mathematical Design Proposal

See **`docs/ARCHITECTURE.md` §4** for the full Khwarizmi State Cell formulation, notation,
complexity analysis, and the baseline-comparison table proving KSC is distinct from Mamba-2 /
RWKV-6 / Gated DeltaNet. Dual Memory math/controller in `docs/MEMORY.md`.

---

## 7. Training Strategy

See **`docs/TRAINING.md`**: architecture validation → pretraining (Colab-friendly, bf16,
chunk-parallel) → instruction → reasoning (offline distillation) → coding (exec-verified) →
project-mgmt → memory → tool-use → verification → distillation → quantization → local deploy.
Cost discipline: config-driven, resumable, stop-and-fix on failure.

---

## 8. Dataset Strategy

See **`docs/DATA.md`**: quality>quantity; EN/AR/Egyptian/Franco/code/math/reasoning/
planning/tool-use/verify; pipeline = ingest→clean→filter→dedup→contamination check→tokenize→
version→audit; multilingual balance; local versioned storage.

---

## 9. Evaluation Strategy

See **`docs/BENCHMARKS.md`**: Intelligence (reasoning/math/coding/lang/IF); Project
Intelligence (planning/decomposition/dependency/long-horizon/memory/recovery/replanning);
Efficiency (tok/s, first-token latency, RAM/VRAM, size, CPU/energy); Memory quality; Adaptive
Compute (Fixed vs Adaptive). Benchmarks built **before scaling**; run offline.

---

## 10. Resource Targets

See **`docs/DEPLOYMENT.md` §1**. Prototype 50–150M · Small 300–700M · Edge 1–3B · Advanced
5–10B+ (only if justified). Optimize for CPU/consumer GPU/edge; quantization INT8/INT4;
GGUF/llama.cpp *only if compatible without harm*; otherwise a custom C/CPU runtime.

---

## 11. 16-Phase Roadmap

See **`docs/ROADMAP.md`** for all phases 0–16 with objective / deliverables / tests /
benchmarks / success / failure / exit / NOT-yet. Phases:

0 Audit+Reset ✅ · 1 Math Spec · 2 KSC Prototype · 3 Memory · 4 Sparse Experts ·
5 Adaptive Compute · 6 Neural Reasoning · 7 Full Core · 8 Training Infra · 9 Data Pipeline ·
10 Small Training · 11 Eval+Ablation · 12 Optimization · 13 Offline Agent+Tools ·
14 Project Intelligence · 15 Edge Deployment · 16 Scaling+Release.

---

## 12. Repository Structure

See **`docs/REPOSITORY_STRUCTURE.md`**. Clean split: `khwarizmi/neural` (clean core, no tool
imports), `khwarizmi/agent`, `khwarizmi/tools` (planner/python_analysis/symbolic_verify),
`khwarizmi/training`, `khwarizmi/data`, `khwarizmi/eval`, `khwarizmi/runtime`,
`khwarizmi/deploy`, `experiments/`, `tests/{neural,tools,agent}`, `docs/`. Old `rafig/`
archived during migration, not deleted.

---

## 13. Risks

See **`docs/RISKS.md`** §1. Top risks: KSC underperforms baselines (R1), recurrent forgetting
(R2), MoE instability (R3), adaptive-compute hurting hard tasks (R4), quant damaging quality
(R5), data scarcity for Arabic reasoning (R6), scope creep (R7). Each has likelihood/impact/
mitigation.

---

## 14. Failure Modes

See **`docs/RISKS.md`** §2. Includes silent quality regression, memory overflow/forgetting,
router thrashing, verification skipped, training divergence, offline leak (CI-enforced
network ban), benchmark overfitting. Guardrails in §3.

---

## 15. Phase 1 Checklist

(Detailed in `docs/ROADMAP.md`.) Lock KSC equations + reference impl + chunk-parallel impl;
numerical unit tests (reference vs chunk match, shapes, dtype/stability, gradients); router +
memory-controller + adaptive-compute specs; loss definitions (LM + MoE aux + policy); baseline
harness (Transformer/Mamba-2/RWKV-6/Gated DeltaNet) on tiny synthetic recall; config schema for
tiers. **Do NOT train on real data, build datasets, implement tools, or scale.**

---

## 16. Immediate Next Action

> **Begin Phase 1.** Implement `khwarizmi/neural/ksc.py` as an **un-trained reference +
> chunk-parallel** module with numerical unit tests proving the recurrence is correct, and
> stand up the **baseline comparison harness** (Transformer / Mamba-2 / RWKV-6 / Gated
> DeltaNet) on a tiny synthetic associative-recall task. No training, no datasets, no scaling.
> Report the math + tests for review before any parameter is updated by gradient descent.

---

### Document index
`docs/ARCHITECTURE.md` · `RESEARCH.md` · `AUDIT.md` · `MEMORY.md` · `ROADMAP.md` ·
`EXPERIMENTS.md` · `BENCHMARKS.md` · `TRAINING.md` · `DATA.md` · `REPOSITORY_STRUCTURE.md` ·
`RISKS.md` · `DEPLOYMENT.md` · `CONTRIBUTING.md`
