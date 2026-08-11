# Khwarizmi AI: Highly Intelligent, Fully Offline Reasoning & Project Assistant
**Project Name:** Khwarizmi AI (formerly RAFIQ foundation)  
**Version:** 2.0.0 (Phase 0 Architecture Reset & Complete Redesign)  
**Date:** 2026-08-11  
**Status:** Implementation-Ready Architecture Blueprint (Ready for Phase 1)  

---

## 1. Executive Summary

**Khwarizmi AI** is an advanced, reasoning-focused artificial intelligence project designed from first principles to operate **100% offline, privately, and locally** without APIs, cloud inference, Wi-Fi, or internet access. 

The fundamental research and engineering objective of Khwarizmi AI is:

> **Maximum intelligence and reasoning capability per unit of compute and memory.**

The system specializes in:
* **Complex Multi-Step Reasoning & Software Engineering**
* **Large Project Management & Technical Leadership** (DAG planning, task decomposition, dependency reasoning, milestone tracking, and long-horizon failure recovery)
* **Local/Private Execution on Modest Hardware** (Consumer CPUs, <4 GB RAM, consumer GPUs, and edge/Android devices)

---

## 2. Master Documentation & Technical Blueprints

Following the **Phase 0 Architecture Reset (2026-08-11)**, the entire project has been audited, mathematically redesigned, and documented. Explore the implementation-ready specifications below:

| Document | Description |
| :--- | :--- |
| **[ARCHITECTURE.md](./ARCHITECTURE.md)** | **System Architecture Blueprint:** Complete specification of the Khwarizmi State Cell (KSC), Dual Memory, Cognitive Router, Sparse MoE, Adaptive Compute, and Layered Tool Architecture. |
| **[ROADMAP.md](./ROADMAP.md)** | **16-Phase Execution Roadmap:** Detailed phase-by-phase deliverables, dependencies, success/failure gates, Phase 1 Checklist, and immediate next actions. |
| **[RESEARCH.md](./RESEARCH.md)** | **Architectural Research Comparison:** Exhaustive analysis of Mamba, RWKV, xLSTM, DeltaNet, Titans, LNNs, MoE, Adaptive Compute, and frontier principles (GPT/Claude/Kimi), distinguishing *Fact, Finding, Hypothesis,* and *Design Decision*. |
| **[EXPERIMENTS.md](./EXPERIMENTS.md)** | **Experimental & Ablation Protocol:** Mandatory ablation testing ladder (Tier 0–Tier 5), statistical reporting rules, and component pruning gates. |
| **[BENCHMARKS.md](./BENCHMARKS.md)** | **Evaluation Strategy & Threshold Gates:** 5-pillar evaluation suite (Intelligence, Project Intelligence, Efficiency, Memory, Adaptive Compute) across 4 model tiers. |
| **[TRAINING.md](./TRAINING.md)** | **12-Stage Training & Dataset Strategy:** Low-resource training (QLoRA, micro-batching, Colab limits), multi-lingual dataset ingestion, MinHash deduplication, and benchmark de-contamination. |
| **[MEMORY.md](./MEMORY.md)** | **Dual Memory Architecture:** Mathematical specification of Short-Term Working State ($S_t$) and Utility-Gated Persistent KV / Symbolic DAG Store with learned `READ`, `WRITE`, `UPDATE`, and `FORGET` gates. |
| **[DEPLOYMENT.md](./DEPLOYMENT.md)** | **Offline Hardware & Edge Blueprint:** Zero-cloud guarantees, Prototype (150M), Small (700M), Edge (2B), and Advanced (7B) tiers, SIMD CPU inference, and GGUF/`llama.cpp` compatibility. |
| **[CONTRIBUTING.md](./CONTRIBUTING.md)** | **Contributor Guidelines:** Architecture-first development rules, phase-gate enforcement, and local workflow instructions. |

---

## 3. Layered Architecture & Legacy Component Audit

A critical principle of Khwarizmi AI is: **Architecture quality > feature count** and **Quality > previous work**. We do not preserve existing code merely because it exists, nor do we force every query through rigid symbolic parsers.

Instead, Khwarizmi AI introduces a clean **Layered Separation of Concerns**:

```
                        +--------------------------------------------------+
                        |                    USER INPUT                    |
                        +--------------------------------------------------+
                                                 |
                                                 v
                        +--------------------------------------------------+
                        |            KHWARIZMI OFFLINE AGENT               |
                        +--------------------------------------------------+
                                                 |
                                                 v
                        +--------------------------------------------------+
                        |                 COGNITIVE ROUTER                 |
                        |     Learned Compute & Path Policy π_θ(p|x)        |
                        +--------------------------------------------------+
                                                 |
            +--------------------+---------------+--------------------+--------------------+
            |                    |                                    |                    |
            v                    v                                    v                    v
  +-------------------+  +-------------------+              +-------------------+  +-------------------+
  |     FAST PATH     |  |    CODING PATH    |              |  REASONING PATH   |  | PROJECT PLAN PATH |
  | (Single KSC Pass, |  | (Coding Experts + |              | (Adaptive Halting |  | (Long-Term Memory |
  |  Minimal Compute) |  |   Python Brain)   |              |  Recurrent Loops) |  |  + DAG Symbolic)  |
  +-------------------+  +-------------------+              +-------------------+  +-------------------+
                                                 |
                                                 v
                        +--------------------------------------------------+
                        |              KHWARIZMI NEURAL CORE               |
                        |    (KSC Blocks + Dual Memory + Sparse Experts)   |
                        +--------------------------------------------------+
                                                 |
                                                 +<========================+
                                                 | (Selective Tool Call)   |
                                                 v                         |
                        +--------------------------------------------------+
                        |               OPTIONAL LOCAL TOOLS               |
                        |  ├── Project Planner (`rafig/reasoning` DAGs)    |
                        |  ├── Python Brain (`rafig/python_brain` AST)     |
                        |  └── Symbolic Verification & Consistency         |
                        +--------------------------------------------------+
                                                 |
                                                 v
                        +--------------------------------------------------+
                        |                  FINAL RESPONSE                  |
                        +--------------------------------------------------+
```

### Existing Codebase Audit Summary (`rafig/`)
* **`rafig/reasoning/` (Phase 06):** Preserved as an **Optional Deterministic Tool (`Project Planner`)**. It provides symbolic DAG task decomposition, dependency checking, and causal inference, but is invoked only when the Cognitive Router selects project planning or plan verification.
* **`rafig/python_brain/` (Phase 07):** Preserved as an **Optional Deterministic Tool (`Python Brain`)**. A standard-library AST static analyzer that inspects Python functions, classes, scopes, types, complexity, and issues without executing code. Invoked selectively during code generation and debugging.
* **`rafig/language/tokenizer.py`:** To be **REPLACED** in Phase 09 with an offline byte-fallback BPE/Unigram tokenizer optimized for Arabic, Egyptian Arabic, English, and Code.
* **`rafig/language/language_understanding.py`:** To be **REPLACED** by the Neural Core and lightweight CLI pre-filters.

---

## 4. Current Phase Status

| Phase | Title | Status | Primary Deliverable |
| :---: | :--- | :---: | :--- |
| **00** | **Repository Audit + Architecture Reset** | ✅ **COMPLETE** | Full Architecture Blueprint & Master Documentation |
| **01** | **Mathematical Specification & Verification** | ⏳ **IMMEDIATE NEXT** | Formal KSC eigenvalue stability proofs & NumPy reference |
| **02** | **Minimal KSC Prototype (50M–150M)** | ⏳ Planned | KSC residual blocks & language modeling baseline |
| **03** | **Dual Memory Architecture Prototype** | ⏳ Planned | Short-Term state & Utility-Gated Persistent KV store |
| **04** | **Sparse Mixture-of-Experts (MoE) Prototype**| ⏳ Planned | Top-2/8 noisy gated experts + CPU RAM ablation gate |
| **05** | **Adaptive Compute & Learned Halting** | ⏳ Planned | Adaptive Recurrent Reasoning Cycles (ARRC) & ponder loss |
| **06** | **Neural Reasoning Core** | ⏳ Planned | Latent state synthesis & self-checking verification |
| **07–16** | **Unified Core to Stable Edge Release** | ⏳ Planned | Training, evaluation, GGUF export, agent tools, edge release |

---

## 5. Running the Tests & Offline Verification

All legacy symbolic tool suites and foundation tests are 100% operational and verified. To run the automated regression test suite:

```bash
python -m unittest discover -v
```

All 141 existing tests pass cleanly, confirming that the foundation, tokenization, semantic representation, reasoning DAG planner, and Python Brain AST analyzer are ready for layered integration.

---

## 6. How to Proceed to Phase 1

1. Read **[ROADMAP.md](./ROADMAP.md)** for the actionable **Phase 1 Checklist**.
2. Review the mathematical formulation in **[ARCHITECTURE.md](./ARCHITECTURE.md#5-mathematical-design-proposal)**.
3. Begin Phase 1 implementation by drafting reference mathematical operators in `khwarizmi/core/ksc_cell.py` and verifying eigenvalue stability across $100{,}000$ sequential steps.
