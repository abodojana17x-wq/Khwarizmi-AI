# Khwarizmi AI: Offline Intelligence Research & Verifiable Project Assistance

**Project:** Khwarizmi AI (formerly RAFIQ foundation)

**Planning review:** 2026-09-05 — roadmap v6.0; not a runtime version bump.

**Current status:** Research prototypes and local tooling. Useful trained assistant capability, release-grade isolation and GPT-6 superiority have **not** been established by the inspected evidence.

## Start here: the updated strategy

- **[الخطة العربية](./الخطة%20.md):** الفكرة المحسنة، خطة 90 يوم، الموارد، وما الذي يمكن إثباته فعلًا.
- **[ROADMAP.md — active plan](./ROADMAP.md):** Two-track strategy, first ten tickets, milestone dependencies, owners, acceptance/stop gates and resource budgets.
- **[RESEARCH.md](./RESEARCH.md):** Official GPT-6 Astra sources reviewed on 2026-09-05, their limitations, repository observations and controlled research hypotheses.
- **[BENCHMARKS.md — active protocol](./BENCHMARKS.md):** Planned held-out 200-case challenge, fair tool/budget comparisons, contamination controls and statistical claim rules.

**Mission:** maximize useful intelligence per unit of compute and memory, with private offline operation. The general-architecture research ambition remains; the first proposed proof of value is a Python project assistant that returns a minimal patch, current project decisions and evidence from checks actually performed.

**No inflated claim:** There is no head-to-head GPT-6 result here. A narrow task win, an efficiency win and offline deployment suitability are different claims. No 90-day promise of general frontier intelligence is made.

**Preserved constraint:** Original KSC research remains from scratch. An optional pretrained, open-weight local backend is only a proposal for a separate practical-system track and requires owner approval and license review. No model download, paid training/API call or private-data upload is authorized by these documents.

**Safety boundary:** `khwarizmi/coding/execution_sandbox.py` is a restricted same-process executor, not a reviewed OS isolation boundary. Do not use it to execute untrusted model-generated code. Safe execution is a blocking implementation milestone, not a feature delivered by this planning update.

**Evidence boundary:** Existing phase descriptions below are historical implementation notes. “Complete” does not establish integration, training, semantic capability or current passing tests. Archived reality-check artifacts include unsupported measurements and inconsistent summaries. PyTorch is absent from the planning-review environment; neural tests were not rerun. Follow the active roadmap/protocol when older documents conflict.

---

## Historical architecture and implementation notes

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

Following the **Phase 0 Architecture Reset** and **Phase 1 Foundational Architecture Implementation (2026-08-11)**, the entire project has been audited, mathematically redesigned, and documented. Explore the implementation-ready specifications below:

| Document | Description |
| :--- | :--- |
| **[ARCHITECTURE.md](./ARCHITECTURE.md)** | **System Architecture Blueprint:** Complete specification of the Khwarizmi State Cell (KSC), Dual Memory, Cognitive Router, Sparse MoE, Adaptive Compute, and Layered Tool Architecture. |
| **[ROADMAP.md](./ROADMAP.md)** | **Active 90-Day Plan:** Research/product tracks, milestone gates, resource budgets and prioritized tickets, followed by the historical architecture backlog. |
| **[RESEARCH.md](./RESEARCH.md)** | **Evidence-Led Research:** Official GPT-6 sources, repository reality audit, corrected engineering claims and minimal ablation hypotheses. |
| **[EXPERIMENTS.md](./EXPERIMENTS.md)** | **Experimental & Ablation Protocol:** Mandatory ablation testing ladder (Tier 0–Tier 5), statistical reporting rules, and component pruning gates. |
| **[BENCHMARKS.md](./BENCHMARKS.md)** | **Active Evaluation Protocol:** Proposed held-out task suite, controlled baselines, statistical decision rules and clearly separated historical measurements. |
| **[TRAINING.md](./TRAINING.md)** | **12-Stage Training & Dataset Strategy:** Low-resource training (QLoRA, micro-batching, Colab limits), multi-lingual dataset ingestion, MinHash deduplication, and benchmark de-contamination. |
| **[MEMORY.md](./MEMORY.md)** | **Dual Memory Architecture:** Mathematical specification of Short-Term Working State ($S_t$) and Utility-Gated Persistent KV / Symbolic DAG Store with learned `READ`, `WRITE`, `UPDATE`, and `FORGET` gates. |
| **[PHYSICS_ART_CREATIVITY_PLAN.md](./PHYSICS_ART_CREATIVITY_PLAN.md)** | **Physics, Art & Creativity Blueprint:** Domain expansion plan for simulation-aware science, structured aesthetic reasoning, and divergent innovation workflows. |
| **[DEPLOYMENT.md](./DEPLOYMENT.md)** | **Offline Hardware & Edge Blueprint:** Zero-cloud guarantees, Prototype (150M), Small (700M), Edge (2B), and Advanced (7B) tiers, SIMD CPU inference, and GGUF/`llama.cpp` compatibility. |
| **[CONTRIBUTING.md](./CONTRIBUTING.md)** | **Contributor Guidelines:** Architecture-first development rules, phase-gate enforcement, and local workflow instructions. |

---

## 3. Layered Architecture & Legacy Component Audit

A critical principle of Khwarizmi AI is: **Architecture quality > feature count** and **Quality > previous work**. We do not preserve existing code merely because it exists, nor do we force every query through rigid symbolic parsers.

Instead, Khwarizmi AI enforces a strict **Layered Separation of Concerns**:

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
* **`rafig/reasoning/` (Phase 06):** Preserved and bridged as an **Optional Deterministic Tool (`ProjectPlannerTool`)**. It provides symbolic DAG task decomposition, dependency checking, and causal inference, but is invoked only when the Cognitive Router selects project planning or plan verification.
* **`rafig/python_brain/` (Phase 07):** Preserved and bridged as an **Optional Deterministic Tool (`PythonAnalysisTool`)**. A standard-library AST static analyzer that inspects Python functions, classes, scopes, types, complexity, and issues without executing code. Invoked selectively during code generation and debugging.
* **`rafig/language/tokenizer.py`:** To be upgraded in Phase 09 with an offline byte-fallback BPE/Unigram tokenizer optimized for Arabic, Egyptian Arabic, English, and Code.
* **`rafig/language/language_understanding.py`:** Replaced by neural representations; supplemented by lightweight pre-filters in `khwarizmi/agent/input_filter.py`.

---

## 4. Current Phase Status

| Phase | Title | Status | Primary Deliverable |
| :---: | :--- | :---: | :--- |
| **00** | **Repository Audit + Architecture Reset** | ✅ **COMPLETE** | Full Architecture Blueprint & Master Documentation |
| **01** | **Foundational Neural Architecture Layer** | ✅ **COMPLETE** | Fully Differentiable CPU Core, KSC, Dual Memory, Router, MoE & Tools Bridge (`v0.1.0-phase1`) |
| **02** | **Minimal KSC Prototype (50M–150M)** | ✅ **COMPLETE** | KSC residual blocks, Prototype 50M/150M configs & language-modeling prototype (`v0.2.0-phase2`) |
| **03** | **Dual Memory Architecture Prototype** | ✅ **COMPLETE** | Short-Term Working State + Utility-Gated Persistent KV store with READ/WRITE/UPDATE/FORGET |
| **04** | **Sparse Mixture-of-Experts (MoE) Prototype**| ✅ **COMPLETE** | Sparse Top-K Noisy-Gated MoE + load-balancing loss + CPU benchmark (`v0.4.0-phase4`) |
| **05** | **Adaptive Compute & Learned Halting** | ✅ **COMPLETE** | Per-token ACT-style ARRC halting + ponder cost loss + adaptivity benchmark (`v0.5.0-phase5`) |
| **06** | **Neural Reasoning Core** | ⏳ Planned | Latent state synthesis & self-checking verification |
| **07–16** | **Unified Core to Stable Edge Release** | ⏳ Planned | Pretraining, evaluation, GGUF export, agent tools, edge release |

---

## 5. Phase 1: Foundational Neural Architecture Layer Overview

### 5.1 Implementation Summary
Phase 1 implements the complete foundational neural architecture layer (`v0.1.0-phase1`) defined by the approved Phase 0 blueprint. The objective of this phase is architectural correctness, modularity, testability, and full differentiability on CPU (and future GPUs) without attempting large-scale model training yet.

### 5.2 Component Responsibilities & Module Structure
The `khwarizmi/` Python package is structured into clean, modular components with strict interfaces:

* **`khwarizmi.config` (`settings.py`, `tiers.py`):** Defines `KhwarizmiConfig` with automated tensor shape/bound validation and JSON serialization. Implements predefined scaling tiers (`get_tiny_test_config()`, `get_prototype_config()`, `get_small_config()`, `get_edge_config()`) so the architecture scales from prototype to edge without rewriting core logic.
* **`khwarizmi.core` (`ksc_cell.py`, `ksc_block.py`, `embeddings.py`, `output.py`, `model.py`):**
  * **Khwarizmi State Cell (KSC):** Sub-quadratic recurrent state cell $S_t \in \mathbb{R}^{B \times H \times d_k \times d_n}$ with discretized diagonal Hurwitz retention gating $\bar{A}_t$ strictly bounded in $[\gamma_{\min}, \gamma_{\max}]$ ($[0.85, 0.999]$), guaranteeing zero numerical overflow/divergence over ultra-long sequences.
  * **Residual KSC Blocks:** Sequential residual blocks chaining LayerNorm, KSC, and optional FFN / Sparse MoE sub-layers.
  * **Output Pathway:** Transforms latent representations to vocabulary logits, estimates statistical confidence score $C(y)$, and triggers selective verification when confidence falls below threshold.
  * **KhwarizmiModel:** End-to-end differentiable neural model integrating all sub-modules.
* **`khwarizmi.memory` (`short_term.py`, `gating.py`, `long_term.py`):**
  * **Short-Term Working State ($\mathcal{M}_{\text{short}}$):** Maintains active KSC recurrent state $S_t$ and a rolling window buffer ($L_{\text{window}}$) of recent token features.
  * **Long-Term Persistent Memory ($\mathcal{M}_{\text{long}}$):** Non-parametric key-value project store $\mathcal{M} = \{(k_i, v_i, u_i, t_i)\}$ operated by learned **READ**, **WRITE**, **UPDATE**, and **FORGET** gates (`MemoryGatingController`). Employs time-decayed utility eviction when capacity is reached.
* **`khwarizmi.routing` (`router.py`, `pathways.py`):** Implements `CognitiveRouter` policy $\pi_\theta(p|x, \mathcal{M}_{\text{short}})$ over 5 discrete computational pathways (`FAST`, `CODING`, `REASONING`, `PROJECT_PLAN`, `VERIFICATION`), with regularized multi-objective FLOP budget and entropy loss. `PathwayDispatcher` maps router decisions to downstream execution flags.
* **`khwarizmi.experts` (`moe_layer.py`, `specialists.py`):** Implements `SparseMoELayer` with Noisy Top-$K$ gating ($K=2$ out of $E=4/8$) and load-balancing auxiliary loss $\mathcal{L}_{\text{balance}} = \alpha_{\text{moe}} E \sum f_i P_i$. Defines standard candidate specialist identities (`Coding`, `Reasoning`, `ProjectPlanning`, etc.).
* **`khwarizmi.reasoning` (`adaptive_compute.py`, `latent_reasoner.py`):** Implements Adaptive Recurrent Reasoning Cycles (ARRC) with learned halting gates $p_k = \sigma(w_h^T z^{(k)} + b_h)$ and ponder cost regularization $\mathcal{L}_{\text{ponder}} = \beta_{\text{ponder}} \mathbb{E}[K + R]$. `LatentReasoner` executes reasoning in latent space without verbose ASCII CoT token dumping.
* **`khwarizmi.tools` (`verifier.py`, `schemas/`):** Layered bridge wrapping legacy `rafig.python_brain` (AST static analysis) and `rafig.reasoning` (DAG project planning) into clean optional tools (`PythonAnalysisTool`, `ProjectPlannerTool`). Tools are **never called automatically on simple queries**, executing only when triggered by the router or selective verification threshold.
* **`khwarizmi.agent` (`input_filter.py`, `agent_loop.py`):** Multi-lingual input sanitizer and orchestrator loop (`KhwarizmiAgentLoop`) managing the complete end-to-end flow.

### 5.3 Complete Tensor Flow
```
User Prompt (String)
  ↓ [InputSanitizer]
Token IDs Tensor (batch_size, seq_len)
  ↓ [KhwarizmiEmbeddings]
Token Features x_0 (batch_size, seq_len, d_model)
  ↓ [ShortTermWorkingState]  ──→ Summary Vector (batch_size, d_model)
  ↓                                ↓ [CognitiveRouter]
  ↓                                Pathway Flags (use_moe, use_adaptive, use_mem_read, use_mem_write)
  ↓                                ↓ [MemoryGatingController]
  ↓                              g_read, g_write, g_update, g_forget
  ↓                                ↓ [LongTermPersistentMemory.read(g_read)]
  ↓                              Retrieved Memory Vector (batch_size, d_model)
  ↓ (Residual Addition of Memory Recall)
  ↓ [KSCResidualBlock 1...N] (with SparseMoELayer Top-K specialists every N layers)
  ↓ [LatentReasoner (AdaptiveComputeBlock ARRC Halting Loops)]
  ↓ (Update Short-Term State S_t & Long-Term Memory write/evict)
  ↓ [OutputPathway]
Logits (batch_size, seq_len, vocab_size) + Confidence Score C(y) + Verification Trigger
  ↓ [SelectiveVerifier (Optional rafig/python_brain AST or rafig/reasoning DAG tool)]
AgentResponseFrame (Neural Output + Tool Diagnostics)
```

### 5.4 Configuration System & CPU Resource Footprint
To verify architecture behavior within tight CPU memory constraints (~3.8 GiB RAM, no GPU), Phase 1 uses the parameterized `TinyTest` configuration (`get_tiny_test_config()`):

| Parameter | TinyTest Value | Purpose / Notes |
| :--- | :--- | :--- |
| **Vocabulary Size (`vocab_size`)** | `512` | Compact subword/character test vocabulary |
| **Model Dimension (`d_model`)** | `64` | Lightweight internal feature dimension |
| **Layers (`n_layers`)** | `2` | 2 residual blocks for CPU unit testing |
| **Attention Heads (`n_heads`)** | `4` | `d_k = 16` per head |
| **Memory Expansion (`d_expansion`)** | `16` | KSC memory bank size $d_n = 16$ |
| **Expert FFN Dim (`d_ff`)** | `128` | Intermediate specialist width |
| **Total Experts (`num_experts`)** | `4` | $E=4$ specialists |
| **Active Experts (`top_k_experts`)** | `2` | Top-$K=2$ selective routing |
| **Eigenvalue Bounds (`gamma_min`/`gamma_max`)** | `0.85` / `0.999` | Hurwitz diagonal retention bounding |
| **ARRC Cycles (`max_recurrent_cycles`)** | `3` | Up to 3 adaptive reasoning cycles |
| **Persistent Memory (`memory_slots`)** | `16` | Key-value table slots ($D_m = 64$) |
| **Total Trainable Parameters** | **`303,563`** | **< 305K parameters for rapid CPU tests** |
| **CPU RAM Parameter Footprint** | **`1.16 MB`** | **< 2 MB memory footprint** |

### 5.5 Running the Test Suite
The complete regression test suite runs on consumer CPUs without requiring external data, GPUs, or internet access:

```bash
# Run all existing RAFIQ legacy tests AND new Phase 1 Khwarizmi tests
python -m unittest discover -v
```

**Test Suite Coverage Summary:**
1. **141 Legacy RAFIQ Tests (`tests/test_*.py`):** 100% passing. Proves zero breakage to symbolic AST and DAG engines.
2. **32 New Khwarizmi Phase 1 Tests (`tests/test_ksc_cell.py`, `test_dual_memory.py`, `test_cognitive_router.py`, `test_sparse_moe.py`, `test_adaptive_compute.py`, `test_khwarizmi_model.py`, `test_agent_tools_bridge.py`):** 100% passing.
   * Proves numerical stability over **50,000 sequential recurrent steps** without NaN/Inf (`test_ksc_cell_long_sequence_numerical_stability`).
   * Proves associative cosine retrieval, time-decayed utility eviction, and forget operations in Dual Memory.
   * Proves policy probability sums and pathway dispatching in the Cognitive Router.
   * Proves Top-2 expert routing and load balancing auxiliary loss in Sparse MoE.
   * Proves adaptive recurrent cycle accumulation and ponder loss in ARRC.
   * Proves that **gradients flow through the complete differentiable neural path** across all 8 core modules (`test_khwarizmi_model_differentiable_gradient_flow`).

### 5.6 Known Limitations & Intentionally Deferred Components
Following the Phase 0 blueprint instructions, components whose mathematical specifications or hardware optimizations are deferred to later phases remain cleanly abstracted:
* **Pretraining / Weights:** The Phase 1 model is an architectural reference implementation initialized with orthogonal/Xavier weights; it is **not** trained on natural language or coding corpora yet (deferred to Phase 2–10).
* **Tokenizer:** Input encoding in Phase 1 uses a byte-fallback tokenization abstraction for CPU unit testing; formal multi-lingual BPE/Unigram tokenizer training and ingestion is deferred to Phase 9.
* **Low-Level SIMD / Quantization:** Custom C++/CUDA kernel bindings and 4-bit/5-bit/8-bit GGUF quantization are deferred to Phase 12 and Phase 15.

### 5.7 What Phase 2 Will Build
**Phase 2: Minimal Khwarizmi State Cell (KSC) Prototype (50M–150M)** builds upon this verified architecture to:
1. Scale `KhwarizmiConfig` to the `Prototype` tier (50M–150M parameters).
2. Train the KSC sequence modeling core on initial language modeling and sequence benchmarks.
3. Quantitatively compare perplexity, sub-quadratic sequence scaling, and context memory footprint against an equal-sized dense Transformer baseline.

> **Phase 2 Status (2026-08-12): ✅ COMPLETE.** See Section 5.8 for the implemented deliverables and `BENCHMARKS.md` §6 for measured results.

### 5.8 Phase 2: Minimal KSC Prototype — Implementation Summary
Phase 2 delivers a **clean, modular KSC-only language-modeling prototype** built exclusively from Phase 1 components. It deliberately excludes Sparse MoE, Dual Memory, the Cognitive Router, and Adaptive Compute (all deferred to Phase 3+), keeping the prototype faithful to the roadmap's "No sparse experts / no external memory" boundary.

**Deliverables implemented:**
* **`khwarizmi/core/ksc_block.py`** — `KSCResidualBlock` (LayerNorm → KSC → residual; LayerNorm → FFN → residual) and `FeedForwardNetwork`. Hardened with explicit input/state validation that raises `ValueError` on malformed tensors (consistent with the Phase 1 `KhwarizmiStateCell`).
* **`khwarizmi/core/embeddings.py`** — `KhwarizmiEmbeddings` (token + sinusoidal positional encoding, LayerNorm, dropout) carried over from Phase 1 and reused directly.
* **`khwarizmi/core/prototype.py` (new)** — `KhwarizmiKSCPrototype`: a trainable LM head stacking `KhwarizmiEmbeddings` + `N × KSCResidualBlock` + final LayerNorm + LM head. Provides both a batched `forward` (prefill) and a single-token `step` (autoregressive decode with an `O(1)`-in-sequence-length recurrent state). Factories `build_ksc_prototype("50m" | "150m")` build the tier models.
* **`khwarizmi/config/tiers.py`** — Two new Prototype Tier configurations: `get_prototype_50m_config()` (~51.4M params) and `get_prototype_150m_config()` (~150.4M params), both with `max_seq_len = 16384` for the 4K/16K context benchmarks. `get_prototype_config()` is retained unchanged for backward compatibility (used by `tests/test_khwarizmi_model.py`).
* **Tests:** `tests/test_ksc_block.py` (15 tests) and `tests/test_ksc_prototype.py` (13 tests) — forward/backward shape, gradient-flow, recurrence-consistency (vectorized == token-by-token), retention-gate bounds, O(1) decoding memory, causality, state-reuse regression, and invalid-input handling.
* **Benchmark:** `benchmarks/phase2_ksc_prototype.py` — footprint, sub-quadratic-memory, latency, and an equal-sized Transformer LM comparison.

**Key interfaces (stable, backward-compatible):**
* `KhwarizmiKSCPrototype(input_ids[, state][, return_retention]) -> KSCPrototypeOutput(logits, states, retention_history)` — returns a dataclass, not a tuple (does not break the Phase 1 `KhwarizmiModel`/`KhwarizmiOutput` contract).
* `KhwarizmiKSCPrototype.step(token_id, state, position) -> (logits, new_state)` — autoregressive decode.
* All existing `khwarizmi` public symbols and the 173 Phase 1 tests remain unchanged.

**How to run:**
```bash
# Phase 2 unit tests
python -m unittest tests.test_ksc_block tests.test_ksc_prototype -v

# Full regression suite (Phase 1 + Phase 2)
python -m unittest discover -s tests -p "test_*.py"

# Phase 2 benchmark (CPU, deterministic)
python benchmarks/phase2_ksc_prototype.py
```

**Known limitations (documented, not implemented — future phases):**
* **WikiText-103 perplexity comparison:** the roadmap's literal success criterion requires the Phase 9 dataset pipeline + Phase 10 training, which are out of Phase 2 scope. The benchmark substitutes a deterministic causal synthetic LM task as a faithful offline proxy; the `KhwarizmiKSCPrototype` learns it and converges toward the equal-size Transformer baseline with training (see `BENCHMARKS.md` §6).
* **Pretrained weights:** the prototype ships with orthogonal/Xavier initialization; it is not yet trained on real multilingual/coding corpora.
* **Inference speed:** `forward` uses an unoptimized Python recurrent scan; sub-quadratic *memory* is proven, while SIMD/quantized *latency* optimization is deferred to Phase 12/15.

### 5.9 Phase 3: Dual Memory Architecture Prototype — Implementation Summary

Phase 3 delivers a **bounded, utility-gated Dual Memory system** composed (not rewritten) on top of the Phase 1/Phase 2 components. It implements the two-tier memory architecture specified in `MEMORY.md` and integrates it with the KSC prototype via a clean compositional interface.

**Deliverables implemented:**
* **`khwarizmi/memory/short_term.py`** — `ShortTermWorkingState`: the bounded short-term working state (KSC recurrent state + rolling token window). The window is hard-capped at `config.short_term_capacity`; it exposes deterministic `read` / `write` / `forget` operations plus `get_summary_vector` (used by the router/gating) and the backward-compatible `update` integration entry point.
* **`khwarizmi/memory/long_term.py`** — `LongTermPersistentMemory`: the fixed-capacity (`config.memory_slots`) non-parametric key-value store with real `READ` (associative retrieval), `WRITE` (selective insertion + time-decayed utility eviction), `UPDATE` (similarity-gated merge into an existing slot, no duplication), and `FORGET` (gate-driven lowest-utility eviction or explicit slot-id eviction). Near-duplicate detection is supported in `WRITE`.
* **`khwarizmi/memory/gating.py`** — `MemoryGatingController` (learned READ/WRITE/UPDATE/FORGET probabilities) plus the new **`UtilityGatingPolicy`**: a deterministic, parameter-free decision policy resolving each candidate to `RETAIN` / `WRITE` / `UPDATE` / `FORGET` (priority: FORGET > UPDATE > WRITE > RETAIN).
* **`khwarizmi/memory/dual_memory.py` (new)** — `DualMemory`: a facade composing the three modules + policy into one bounded memory lifecycle (`init_state`, `read`, `forward`).
* **`khwarizmi/core/memory_prototype.py` (new)** — `KhwarizmiDualMemoryPrototype`: composes `KhwarizmiKSCPrototype` with `DualMemory`; recalled memory conditions the KSC pass via the new optional `memory_conditioning` argument (a minimal, backward-compatible addition to the Phase 2 prototype), and the post-KSC candidate flows through the utility-gated write/update/forget lifecycle.
* **`khwarizmi/config/settings.py`** — new Phase 3 configuration: `short_term_capacity`, `utility_threshold`, `read_threshold`, `write_threshold`, `update_threshold`, `forget_threshold`, `update_similarity_threshold`, `utility_decay_lambda` (all validated; defaults preserve existing tier configs).
* **Tests:** `tests/test_dual_memory_phase3.py` (33 tests) and `tests/test_dual_memory_integration.py` (14 tests) — initialization, READ/WRITE/UPDATE/FORGET, utility gating, capacity limits, eviction, near-duplicates, determinism, invalid inputs, KSC-prototype integration, and Phase 1/Phase 2 regression.
* **Benchmark:** `benchmarks/phase3_dual_memory.py` — bounded footprint, 10,000-cycle long-sequence stability, per-operation throughput, and bottom-10% utility-eviction quality.

**Boundedness guarantee:** both stores are fixed-size tensors — the short-term window is capped at `short_term_capacity` and the persistent table at `memory_slots` — so memory usage is `O(1)` in sequence length and operation count (verified by tests and the benchmark).

**Known limitations (documented, not implemented — future phases):**
* **Learned gating policy:** the READ/WRITE/UPDATE/FORGET *gate network* ships with conservative (negatively-biased) initialization and is not yet trained; the decision *policy* is deterministic but the gate probabilities only become meaningful after Phase 8–10 training. NIAH ≥95% and selective-write precision are therefore deferred to the trained phases (see `MEMORY.md` §6).
* **DAG project store:** the symbolic DAG integration with `rafig/reasoning` remains a Phase 13 tool concern; Phase 3 implements the associative KV tier only.

### 5.10 Phase 4: Sparse Mixture-of-Experts (MoE) Prototype — Implementation Summary

Phase 4 delivers the **Sparse Top-K Noisy-Gated Mixture-of-Experts** sublayer specified in `ARCHITECTURE.md` §4.4/§5.4, integrated into the Khwarizmi neural core without rewriting Phase 1–3 systems.

**Deliverables implemented:**
* **`khwarizmi/experts/moe_layer.py`** — `SparseMoELayer` (noisy Top-K router + genuinely sparse expert dispatch + load-balancing auxiliary loss), `ExpertLayer` (independently parameterized Swish FFN, configurable `expert_d_ff`), and `MoERoutingDecision` (structured routing result: logits, full-softmax probs, Top-K indices/weights, dispatch fractions f_i, mean gating probs P_i, auxiliary loss). `forward` evaluates **only the Top-K selected experts** (tokens are gathered per active expert; unselected experts are never called). Noise is applied only during training; inference routing is deterministic.
* **`khwarizmi/experts/specialists.py`** — `create_standard_specialists`: the 8 named specialist experts from the blueprint (`Multilingual_Arabic` … `General_Fact_Recall`), with independent parameters (names are metadata; real specialization emerges through training).
* **`khwarizmi/core/model.py`** — integration point: the shared MoE sublayer is wired into every `moe_frequency`-th KSC residual block of `KhwarizmiModel` (as before), now gated by `config.enable_moe`. With `enable_moe=False` every block carries a dense FFN and no experts/router are built — preserving the pre-Phase-4 dense behavior exactly.
* **`khwarizmi/config/settings.py`** — new Phase 4 configuration (all validated): `enable_moe`, `moe_noise_enabled`, `expert_d_ff`, plus stricter validation for `num_experts` ≥ 1, `1 ≤ top_k_experts ≤ num_experts`, `moe_frequency ≥ 1`, `d_ff > 0`, and finite non-negative `load_balance_alpha`.
* **Tests:** `tests/test_moe.py` (57 tests) — expert init/independence, router logits/noise/determinism/ties, Top-K validity, normalized routing weights, sparse-execution call counting, manual Top-K output equivalence, expert/router/routing-weight/noise gradient flow, analytic auxiliary-loss gradient, load-balancing closed forms and collapse behavior, batch/sequence inputs, invalid configs and shapes, `KhwarizmiModel` integration, MoE-disabled regression, and Phase 2 (KSC prototype) / Phase 3 (Dual Memory) compatibility. The pre-existing `tests/test_sparse_moe.py` remains intact.
* **Benchmark:** `benchmarks/phase4_sparse_moe.py` — parameter efficiency, expert-execution counting (SPARSE 2.0 vs DENSE 32.0 expert evaluations/token), routing overhead (~1%), SPARSE vs DENSE vs fused-DENSE vs equal-active-FFN latency, theoretical MACs, expert utilization, and load-balancing collapse detection + prevention experiments (see `BENCHMARKS.md` §7).

**Sparsity guarantee:** the sparse layer executes exactly the routed experts — verified by forward-hook call counting (unit tests) and by the benchmark, which measures 16× fewer expert evaluations than the dense reference at E=32/K=2 (93.8% expert-MAC reduction; 3.5–4.3× measured forward-latency speedup on CPU).

**Known limitations (documented — future phases):**
* **Trained-router quality gains:** the roadmap's "≥8% validation-perplexity improvement over an equal-active dense baseline" requires the Phase 9 dataset pipeline + Phase 10 training; Phase 4 ships the trainable mechanism and verifies gradient flow, sparsity, and balance-loss behavior offline.
* **Saturated collapse is not repairable by the balance loss alone:** at f_max = 1.0 the auxiliary-loss gradient vanishes (it is a *preventive* regularizer); a fully collapsed trained router would need re-initialization or an additional entropy/z-loss, deferred to Phase 8+ training tooling.
* **CPU dispatch overhead:** per-expert token gather/scatter and small per-expert batches make the sparse layer ~6× slower than a single equal-active dense FFN at the benchmark scale (still ~3–4× faster than evaluating all 32 experts); expert-fused kernels are a Phase 12 optimization concern.

### 5.11 Phase 5: Adaptive Compute & Learned Halting (ARRC) — Implementation Summary

Phase 5 delivers the **Adaptive Recurrent Reasoning Cycles (ARRC)** engine specified in `ARCHITECTURE.md` §4.5/§5.5: per-token ACT-style learned halting with exact remainder accounting and a differentiable ponder cost, integrated into the Khwarizmi neural core without rewriting Phase 1–4 systems.

**Deliverables implemented:**
* **`khwarizmi/reasoning/adaptive_compute.py`** — `AdaptiveComputeBlock` (per-token ACT halting engine: shared KSC reasoning cell applied iteratively; halting gate p_k = σ(w_h·z⁽ᵏ⁾ + b_h) accumulated per token; halt at the first cycle k ≥ K_min with Σp_j ≥ 1 − ε; forced remainder halt at K_max; ACT-weighted output z_out = Σ_{k<K} p_k z⁽ᵏ⁾ + R z⁽ᴷ⁾ with per-token weights summing to exactly 1; halted tokens frozen so different tokens receive different compute depth) and **`PonderCostLoss`** (the roadmap's ponder cost module: L_ponder = β·E[N + R], step count N detached, remainder R differentiable through the halting gate).
* **`khwarizmi/reasoning/latent_reasoner.py`** — `LatentReasoner.reason` now passes through `min_cycles`/`max_cycles` runtime overrides while preserving the Cognitive Router FAST-pathway bypass.
* **`khwarizmi/core/model.py`** — integration point: the reasoner is gated by `config.enable_adaptive_compute`. With `enable_adaptive_compute=False`, no halting gates or reasoning cell are built, the model performs a single fixed pass, and `losses["ponder_loss"]` is exactly 0 — preserving the pre-Phase-5 fixed-compute path.
* **`khwarizmi/config/settings.py`** — new Phase 5 configuration (all validated): `enable_adaptive_compute`, `min_recurrent_cycles` (≥1), `halting_epsilon` (∈ (0,1)), plus `min_recurrent_cycles ≤ max_recurrent_cycles` and finite non-negative `ponder_cost_beta` validation. Defaults are backward compatible.
* **Tests:** `tests/test_adaptive_compute_phase5.py` (57 tests) — configuration validation and boundary values, PonderCostLoss behavior (monotone in compute/remainder, β-scaled, N detached / R differentiable), min-step and max-step enforcement (including saturated-gate extremes), halting-probability validity and Σp_k ≥ 1 − ε verification, exact accumulated-probability capping, adaptive per-token step-count variation, deterministic inference, forced fixed-compute mode, state initialization/propagation/carrying, batch independence (no cross-example leakage), gradient flow (output→all parameters, output→halting gate, ponder→halting gate, ponder-only training provably increases halting probability), 2D/3D inputs, disabled-path compatibility, and full `KhwarizmiModel` integration with KSC + Dual Memory + Sparse MoE (including both switches off). The pre-existing `tests/test_adaptive_compute.py` remains intact.
* **Benchmark:** `benchmarks/phase5_adaptive_compute.py` — halting-step distribution (min/avg/max steps, % halting per step), trained easy-vs-hard compute differentiation, termination-guarantee check, ADAPTIVE vs FIXED COMPUTE latency in mixed- and uniform-halting regimes, and parameter/allocation memory (see `BENCHMARKS.md` §8).

**Adaptivity guarantee:** the benchmark and unit tests verify that tokens actually halt at different depths (e.g. untrained gate: 3.3%/41.2%/15.0%/7.8%/4.3%/28.3% across steps 1–6, K_avg ≈ 3.54 with K_max = 6; after ponder training easy inputs average 1.21 cycles vs 1.47 for hard inputs) — computation is measurably input-dependent, never a fixed-depth loop in disguise.

**Known limitations (documented — future phases):**
* **Wall-clock vs FLOP savings:** the batch-level early exit skips remaining cycles only once *every* token in the batch has halted; with mixed halting the wall time is ≈ parity with fixed compute (per-token FLOP demand K_avg/K_max ≈ 0.42), while the uniform-halting regime shows the real ≈6× cycle-skipping speedup. Per-token kernels are a Phase 12 runtime concern.
* **Trained easy/hard gates:** the roadmap's literal K_avg ≤ 1.2 (easy) / ≥ 2.5 (hard) and ≥15% hard-task accuracy targets are defined over a trained language model and require the Phase 9 dataset + Phase 10 training; Phase 5 ships and verifies the trainable mechanism.

---

## 6. Verification & Quality Gates

Every implementation phase in Khwarizmi AI must pass rigorous verification gates before code is merged:

```
+-----------------------------------------------------------------------------------+
|                        PHASE 1 IMPLEMENTATION QUALITY GATE                         |
+-----------------------------------------------------------------------------------+
|  [x] Complete existing test suite passing (141/141 legacy tests OK)               |
|  [x] All new Phase 1 CPU tests passing (32/32 tests OK; 173 total tests passing)   |
|  [x] Zero existing functionality broken in rafig/ symbolic engines                 |
|  [x] Complete repository runs offline on modest CPU within ~3.8 GiB RAM            |
|  [x] Zero large datasets, pretrained models, or binaries accidentally added        |
|  [x] Architecture parameterized (KhwarizmiConfig) to scale without core rewriting  |
|  [x] Complete differentiable gradient path proven across all neural modules        |
+-----------------------------------------------------------------------------------+
|                        PHASE 2 IMPLEMENTATION QUALITY GATE                         |
+-----------------------------------------------------------------------------------+
|  [x] Phase 1 suite intact (173 tests) + 28 new Phase 2 tests (201 total passing)   |
|  [x] KSC residual block + Prototype 50M/150M configs + KSC prototype LM delivered   |
|  [x] Sub-quadratic inference memory proven (O(1) decode state; ~1024x < Transformer |
|      KV-cache at 16K context)                                                       |
|  [x] Forward/backward + recurrence-consistency + causality tests passing          |
|  [x] No Phase 1 public interface broken (KhwarizmiModel/KhwarizmiOutput unchanged) |
|  [x] No sparse experts / external memory introduced (Phase 3+ boundary respected)   |
+-----------------------------------------------------------------------------------+
|                        PHASE 3 IMPLEMENTATION QUALITY GATE                         |
+-----------------------------------------------------------------------------------+
|  [x] Phase 1 + Phase 2 suite intact (201 tests) + 47 new Phase 3 tests (248 total) |
|  [x] Short-Term Working State (bounded) + Persistent KV store with READ/WRITE/     |
|      UPDATE/FORGET + deterministic UtilityGatingPolicy delivered                   |
|  [x] Memory strictly bounded (short-term window + fixed-capacity table) —          |
|      no unbounded Python list/dict growth (verified over 10,000-cycle benchmark)   |
|  [x] KSC prototype integration via composition; Phase 1/2 interfaces unchanged      |
|  [x] No Sparse MoE / Adaptive Compute / Router redesign / dataset work (Phase 4+)   |
+-----------------------------------------------------------------------------------+
|                        PHASE 4 IMPLEMENTATION QUALITY GATE                         |
+-----------------------------------------------------------------------------------+
|  [x] Phase 1-3 suite intact (249 tests) + 57 new Phase 4 tests (306 total passing) |
|  [x] Noisy Top-K gating (softplus-noise, training-only) + normalized routing weights|
|  [x] Genuinely sparse execution: only Top-K experts evaluated (hook-count verified) |
|  [x] Load-balancing auxiliary loss differentiable, closed-form verified             |
|  [x] Gradient flow proven: selected experts, router, routing weights, noise, loss   |
|  [x] enable_moe=False preserves dense pre-Phase-4 behavior (regression tested)     |
|  [x] Phase 2 KSC prototype and Phase 3 Dual Memory untouched and compatible         |
|  [x] Benchmark: 16x fewer expert evals vs dense; 93.8% expert-MAC reduction         |
|  [x] No Adaptive Compute / Cognitive Router redesign / dataset work (Phase 5+)      |
+-----------------------------------------------------------------------------------+
```
