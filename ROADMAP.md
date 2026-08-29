# KHwarizmi Master Roadmap — Architecture Research & General Intelligence Strategy

**Document Status:** Authoritative Master Plan  
**Version:** 5.0 (Architecture Research & General Intelligence Upgrade)  
**Date:** 2026-08-28  
**Supersedes:** ROADMAP.md v4.0

---

## Executive Summary

KHwarizmi AI is being developed as a **general-purpose cognitive architecture** capable of achieving extremely high reasoning, planning, coding, mathematical, linguistic, scientific, and multimodal capabilities while operating with dramatically lower resource requirements than conventional frontier systems.

The ultimate objective is NOT merely:
* An Arabic assistant
* An offline chatbot
* A coding assistant
* A small LLM
* A Transformer replacement

Those may be applications or capabilities, but they are NOT the ultimate research objective.

The goal is to discover whether a fundamentally more efficient cognitive architecture can outperform much larger systems in meaningful capability domains through:
* General intelligence capability
* Reasoning quality
* Reliability
* Learning efficiency
* Inference efficiency
* Memory efficiency
* Adaptive computation
* Planning
* Verification
* Continual improvement
* Scalability across Nano/Mobile/Pro/Ultra tiers

**This outcome is NOT guaranteed.** The roadmap is designed to **maximize the probability of discovering such an architecture through rigorous experimentation**.

### Four-Tier Resource Strategy

| Version | Target Footprint | Primary Target |
|---------|------------------|----------------|
| **Nano** | ≤ 500 MB | Weakest smartphones, old devices, extreme offline constraints |
| **Mobile** | ≤ 900 MB | Capable smartphones, weak laptops, low-resource educational/work environments |
| **Pro** | ≤ 1.5 GB (hard target) | Frontier-level reasoning on consumer hardware |
| **Ultra** | ≤ 2 GB | Maximum capability within accessible hardware limits |

**Core Design Philosophy:** Intelligence must not depend solely on model size. KHwarizmi achieves capability through efficient reasoning, selective computation, specialized subsystems, verification, compression, modularity, and intelligent resource allocation.

---

## Guiding Research Principle

> **External research produces hypotheses, not architectural decisions.**

No external architecture gets promoted into the production architecture without experimental evidence.

**Evaluation Process:**
1. Research → Hypothesis
2. Minimal experiment
3. Ablation study
4. Benchmark comparison
5. Compare with KHwarizmi baseline
6. Decision: KEEP / MODIFY / REPLACE / COMBINE / INVENT SOMETHING NEW

> **Do not copy the frontier. Discover a better computational path to intelligence.**

And if existing research does not provide that path:

> **KHwarizmi becomes the research program that attempts to invent it.**

---

## Critical Rule: No Premature Rebuild

> **Never replace a working KHwarizmi component solely because a newer paper proposes an alternative.**

A component can only be replaced when:
1. A measurable weakness is identified
2. A candidate alternative is implemented in isolation
3. The candidate is benchmarked
4. Ablation demonstrates meaningful improvement
5. Resource cost is evaluated
6. Regression tests pass
7. Integration is justified

---

## Part I: Repository Audit & Current State Assessment

### I.1 What Currently Exists

The repository has been thoroughly audited. The following components are **implemented and tested**:

#### Completed Implementations (Phases 0–6 + 7A)

| Component | Location | Status | Tests |
|-----------|----------|--------|-------|
| **Khwarizmi State Cell (KSC)** | `khwarizmi/core/ksc_cell.py`, `ksc_block.py` | ✅ Complete | `test_ksc_cell.py`, `test_ksc_block.py` |
| **Dual Memory Architecture** | `khwarizmi/memory/dual_memory.py`, `gating.py`, `short_term.py`, `long_term.py` | ✅ Complete | `test_dual_memory.py` (57 tests) |
| **Sparse Mixture-of-Experts** | `khwarizmi/experts/moe_layer.py`, `specialists.py` | ✅ Complete | `test_moe.py`, `test_sparse_moe.py` (57 tests) |
| **Adaptive Compute (ARRC)** | `khwarizmi/reasoning/adaptive_compute.py` | ✅ Complete | `test_adaptive_compute_phase5.py` (57 tests) |
| **Neural Reasoning Core** | `khwarizmi/reasoning/neural_reasoning_core.py`, `latent_reasoner.py` | ✅ Complete | `test_neural_reasoning_core_phase6.py` |
| **Cognitive Router** | `khwarizmi/routing/router.py`, `pathways.py` | ✅ Complete | `test_router.py`, `test_cognitive_router.py` |
| **Python Brain (AST Analyzer)** | `rafig/python_brain/` (bridged) | ✅ Complete | Multiple parser/type tests |
| **Project Planner (DAG Engine)** | `rafig/reasoning/` (bridged) | ✅ Complete | `test_data_flow.py`, `test_cfg.py` |
| **Agent Loop** | `khwarizmi/agent/agent_loop.py` | ✅ Complete | `test_agentic_loop.py` |
| **Execution Sandbox** | `khwarizmi/coding/execution_sandbox.py` | ✅ Complete | `test_sandbox.py` |
| **Data Pipeline (CoT + Dedup)** | `khwarizmi/data/qwen_cot_pipeline.py`, `minhash_dedup.py` | ✅ Complete | — |
| **Tier Configurations** | `khwarizmi/config/tiers.py` | ✅ Complete | Prototype 50M/150M, Small, Edge |

#### Partially Implemented

| Component | Location | Gap |
|-----------|----------|-----|
| **Islamic Alignment Engine** | `khwarizmi/routing/sharia_router.py`, `training/islamic_loss.py` | Phase 7B planned but not fully integrated |
| **Physics/Art/Creativity Domains** | `science/`, `art/` | Unit verifier exists; full domain integration pending |
| **Multilingual Tokenizer** | — | Phase 9 not yet implemented; current tokenizer is minimal |
| **Quantization Runtime** | — | Phase 12 not yet implemented |
| **Full Training Infrastructure** | `khwarizmi/training/` | Basic losses exist; full QLoRA trainer pending |

#### Not Yet Implemented (Future Phases)

- Full model training runs (Phase 10)
- Comprehensive evaluation suite execution (Phase 11)
- Quantized GGUF export and CPU runtime (Phase 12)
- Edge deployment packaging (Phase 15)
- Adversarial self-testing framework (new requirement)

### I.2 Test Evidence

Current test status (as of audit):
- **Total tests:** 604
- **Passing:** 578 (95.7%)
- **Failing:** 17
- **Errors:** 9

Most failures are in router confidence thresholds and edge-case sandbox tests—not fundamental architecture failures. The core components (KSC, MoE, Adaptive Compute, Dual Memory) pass their unit tests.

### I.3 Protected Baseline

The following components form the **protected baseline** — they are working, tested, and must NOT be replaced without experimental justification:

| Component | Status | Evidence |
|-----------|--------|----------|
| **KSC State Cell** | ✅ Complete | `test_ksc_cell.py`, `test_ksc_block.py` pass; eigenvalue-bounded recurrence implemented |
| **Dual Memory** | ✅ Complete | `test_dual_memory.py` (57 tests); READ/WRITE/UPDATE/FORGET with utility gating |
| **Sparse MoE** | ✅ Complete | `test_moe.py`, `test_sparse_moe.py` (57 tests); Top-2/8 routing with load balancing |
| **Adaptive Compute (ARRC)** | ✅ Complete | `test_adaptive_compute_phase5.py` (57 tests); ACT-style halting with ponder cost |
| **Neural Reasoning Core** | ✅ Complete | `test_neural_reasoning_core_phase6.py`; latent synthesis + consistency checking |
| **Cognitive Router** | ✅ Complete | `test_router.py`, `test_cognitive_router.py`; 5 pathways (FAST/CODING/REASONING/PROJECT_PLAN/VERIFICATION) |
| **Python Brain (AST)** | ✅ Complete | Multiple parser/type/inference tests; deterministic code verification |
| **Project Planner (DAG)** | ✅ Complete | `test_data_flow.py`, `test_cfg.py`; symbolic dependency reasoning |
| **Agent Loop** | ✅ Complete | `test_agentic_loop.py`; input sanitization + tool orchestration |
| **Execution Sandbox** | ✅ Complete | `test_sandbox.py`; safe code execution |

**Critical Principle:** These components are NOT to be replaced simply because external research proposes alternatives. They form the experimental baseline against which all new architectures must be compared.

### I.4 Current Test Evidence

Current test status (as of audit):
- **Total tests:** 604
- **Passing:** 578 (95.7%)
- **Failing:** 17
- **Errors:** 9

Most failures are in router confidence thresholds and edge-case sandbox tests—not fundamental architecture failures. The core components (KSC, MoE, Adaptive Compute, Dual Memory) pass their unit tests.

### I.5 Capability Gaps Identified

The following capabilities require further development or experimental validation:

| Gap Category | Current State | Required Investigation |
|--------------|---------------|----------------------|
| **Multi-strategy reasoning** | Single-path ARRC | Branching, candidate generation, reconciliation |
| **Failure-driven learning** | Basic failure detection | Failure classification, signature storage, strategy modification |
| **Contradiction detection** | Consistency heads only | Explicit conflict detection, backtracking mechanisms |
| **Skill graph** | Not implemented | Structured capability taxonomy with benchmarks |
| **World/causal representation** | Implicit in language | Structured entity/relation/state representations |
| **Memory hierarchy** | Dual (short/long-term) | Episodic, semantic, skill, strategy, failure memory levels |
| **Architecture search** | Human-designed | Automated component search (staged approach) |
| **Novel architecture discovery** | KSC-only variants | New computational mechanisms if gaps persist |

### I.6 Obsolete Assumptions Requiring Revision

1. **Single-model scaling assumption:** Old roadmap assumes one model scaled from 150M → 10B+. New vision requires four distinct profiles (Nano/Mobile/Pro/Ultra) with shared core but different capacity allocations.

2. **Phase ordering:** Old roadmap treats phases as strictly sequential. New architecture requires parallel development of shared core + version-specific optimizations.

3. **Size targets mismatch:** Old roadmap mentions "Edge Tier (1B–3B)" and "Advanced Tier (5B–10B+)" which exceed the new Ultra hard limit of 2 GB packaged footprint.

4. **Islamic alignment positioning:** Phase 7B was added but not integrated into the four-version strategy.

### I.7 What Must Be Refactored

| Component | Reason | Action |
|-----------|--------|--------|
| `khwarizmi/config/tiers.py` | Does not include Nano/Mobile/Pro/Ultra profiles | Add four new tier configurations |
| Router pathways | Only 5 pathways; needs meta-reasoning, contradiction, failure paths | Extend to support new reasoning primitives |
| Memory gating | No explicit failure memory or contradiction tracking | Add failure classification and conflict detection |
| Compression strategy | Not yet implemented | Add selective/heterogeneous compression framework |
| Benchmark suite | Lacks prompt-independence, cross-lingual, adversarial tests | Expand benchmark matrix |

### I.8 What Must Be Removed

- References to "Advanced Tier (5B–10B+)" — exceeds 2 GB Ultra limit
- Any implication that smaller versions sacrifice reasoning architecture
- Assumptions that cloud/API fallback is acceptable for core intelligence
- Hardcoded superiority claims without benchmark evidence

---

## Part II: The Khwarizmi Core Architecture

### II.1 Shared Core Components

All four versions (Nano, Mobile, Pro, Ultra) share the same fundamental reasoning architecture:

```
                    ┌─────────────────────┐
                    │   KHARIZMI CORE     │
                    │  (Shared by All)    │
                    └──────────┬──────────┘
                               │
        ┌──────────────┬───────┴───────┬──────────────┐
        │              │               │              │
   ┌────▼────┐   ┌─────▼─────┐   ┌────▼────┐   ┌─────▼────┐
   │  NANO   │   │  MOBILE   │   │   PRO   │   │   ULTRA  │
   │ ≤500 MB │   │ ≤900 MB   │   │≤1.5 GB  │   │  ≤2 GB   │
   └─────────┘   └───────────┘   └─────────┘   └──────────┘
```

#### Core Capabilities (All Versions)

1. **Problem Understanding**
   - Semantic representation extraction
   - Intent recovery from imperfect prompts
   - Constraint and context identification

2. **Reasoning Orchestration**
   - Pathway selection via cognitive router
   - Adaptive compute allocation
   - Multi-path exploration (when needed)

3. **Verification**
   - Consistency checking
   - Contradiction detection
   - Computational verification (math/code)

4. **Memory Systems**
   - Short-term working state
   - Long-term persistent memory with utility gating
   - Failure memory (new)

5. **Meta-Reasoning**
   - Strategy selection
   - Difficulty estimation
   - Compute budget allocation
   - Stopping criteria

6. **Multilingual Semantics**
   - Language-independent semantic representation
   - Broad global language support

7. **Multimodal Interfaces**
   - Image understanding
   - Audio analysis
   - Document processing

8. **Tool Orchestration**
   - Selective activation of deterministic tools
   - Python AST verification
   - DAG planning engine

### II.2 Version-Specific Scaling Parameters

Differences between versions come from capacity allocations, not reasoning philosophy:

| Parameter | Nano | Mobile | Pro | Ultra |
|-----------|------|--------|-----|-------|
| **Model Capacity** | Minimal | Moderate | Large | Maximum |
| **Memory Slots** | 128 | 256 | 512 | 1024+ |
| **Active Experts** | 2 of 4 | 2 of 6 | 2 of 8 | 3 of 8 |
| **Search Breadth** | 1 path | 1–2 paths | 2–3 paths | 3–5 paths |
| **Search Depth** | Shallow | Medium | Deep | Very Deep |
| **Reasoning Budget** | Low | Medium | High | Maximum |
| **Working Memory** | 64 KB | 128 KB | 256 KB | 512 KB+ |
| **Context Capacity** | 8K | 16K | 32K | 64K+ |
| **Verification Budget** | Minimal | Moderate | Substantial | Extensive |
| **Precision Allocation** | INT4/INT5 | INT5/INT8 | FP16/INT8 | FP16 |
| **Modality Capacity** | Basic | Good | Strong | State-of-the-art |

---

## Part IIA: Architecture Research Track

### IIA.1 Purpose and Scope

This research track runs **in parallel** with the protected baseline development. It is NOT a replacement for working components but an exploration space for discovering more efficient computational mechanisms.

**Critical Distinction:**
- **Production Baseline:** Working, tested components (KSC, Dual Memory, MoE, ARRC, etc.)
- **Research Lab:** Experimental variants and novel architectures under evaluation

### IIA.2 Research Areas

#### Neural Architecture Exploration

| Area | Research Questions | Evaluation Criteria |
|------|-------------------|---------------------|
| **Recurrent alternatives** | Can xLSTM-style matrix memory improve over KSC? | Associative recall, sequence modeling, stability |
| **State Space Models** | Do Mamba-3 selective scans offer advantages? | Long-context retention, compute efficiency |
| **Gated DeltaNet** | Is delta-rule learning beneficial? | Fast associative binding, in-context learning |
| **Looped computation** | Does weight-tied depth expansion help? | Parameter efficiency, reasoning quality |
| **Hierarchical reasoning** | Can HRM-style latent hierarchies improve planning? | Multi-step task success, abstraction quality |
| **Latent reasoning** | Is implicit reasoning superior to explicit CoT? | Accuracy, token efficiency, verification |

#### Memory Architecture Investigation

Research whether memory should be organized as:

1. **Neural/State Memory** — Recurrent hidden state (currently KSC state matrix)
2. **Working Memory** — Short-term bounded buffer (currently short-term module)
3. **Episodic Memory** — Time-stamped experience storage
4. **Semantic Memory** — Fact/concept knowledge base
5. **Skill Memory** — Learned procedure representations
6. **Strategy Memory** — Successful reasoning patterns
7. **Failure Memory** — Error signatures and corrections

**Open Question:** Should these be separate modules, unified representations, or hierarchical structures?

**Decision Method:** Ablation experiments comparing:
- Single unified memory vs. separated functional memories
- Neural-only vs. symbolic-augmented vs. hybrid representations
- Fixed capacity vs. dynamic allocation

#### Reasoning Strategy Research

Investigate multi-strategy reasoning beyond single-path ARRC:

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Single-path depth** | Current ARRC approach | Easy-medium tasks, high confidence |
| **Branching exploration** | Generate multiple candidate paths | Ambiguous inputs, low confidence |
| **Candidate comparison** | Evaluate competing hypotheses | Conflicting evidence, multiple interpretations |
| **Disagreement detection** | Identify contradictions between paths | Verification failures |
| **Reconciliation** | Merge compatible partial solutions | Complementary reasoning chains |
| **Bounded search** | Explore tree within budget | Hard problems requiring systematic search |
| **Strategy switching** | Change approach mid-reasoning | Stalled progress, detected dead ends |

**Research Question:** Is intelligently allocating reasoning across multiple candidate paths more effective than simply increasing depth on one path?

#### Hybrid Systems Investigation

Explore neuro-symbolic combinations:

- Neural generation + symbolic verification (current Python Brain model)
- Neural pattern recognition + graph-based causal reasoning
- Neural embedding + formal logic constraints
- Neural intuition + algorithmic execution

**Key Principle:** Symbolic tools are invoked selectively via cognitive router, not on every token.

#### Training Strategy Research

Investigate modern training approaches:

| Approach | Purpose | Status |
|----------|---------|--------|
| **Knowledge distillation** | Transfer from strong teachers | Planned Phase 11 |
| **Reasoning distillation** | Extract latent reasoning traces | Research needed |
| **Synthetic curriculum** | Generate targeted training problems | Partially implemented (CoT pipeline) |
| **Verifier-guided training** | Use verification signals as rewards | Research needed |
| **Self-play** | Model generates and solves its own problems | Research needed |
| **Failure-driven training** | Target weak capabilities identified by benchmarks | Research needed |
| **Skill-specific training** | Train specialized capability modules | Research needed |

**Critical Rule:** Teacher outputs are NOT ground truth. They are candidate knowledge that must be independently verified where possible.

### IIA.3 Novel Architecture Track

This track is activated when experiments demonstrate that existing architectures cannot adequately close an identified capability gap.

**Process:**
1. Observed limitation → Document specific failure mode
2. Formulate hypothesis → Mathematical/computational specification
3. Minimal prototype → Isolated implementation
4. Controlled experiment → Compare against baseline
5. Ablation study → Identify active components
6. Benchmark → Measure on relevant tasks
7. Comparison → Quantify improvement over baseline
8. Iteration → Refine or discard

**If existing architectures are insufficient:** Design a genuinely new computational mechanism with coherent principles, not a "Frankenstein architecture" combining unrelated components.

### IIA.4 Architecture Search Strategy

Automated architecture search will be pursued in stages:

| Stage | Approach | Prerequisites |
|-------|----------|---------------|
| **Stage 1** | Human-designed candidate architectures | Baseline stable, benchmark suite complete |
| **Stage 2** | Automated hyperparameter/component search | Stage 1 complete, search infrastructure ready |
| **Stage 3** | Learned strategy selection | Stage 2 complete, sufficient training data |
| **Stage 4** | Architecture-level optimization | Stage 3 complete, clear optimization objectives |
| **Stage 5** | Controlled self-modification (only if justified) | All prior stages complete, safety guarantees established |

**Critical Constraint:** Every stage requires evidence before advancing. No automatic progression.

### IIA.5 Epistemological Classification

All claims in this roadmap are tagged with epistemological status:

| Label | Meaning | Example |
|-------|---------|---------|
| **[FACT]** | Proven mathematical reality or established empirical consensus | "KSC has O(1) decoding memory" |
| **[RESEARCH FINDING]** | Empirical result under specific published conditions | "Mamba-3 achieves X on Y benchmark" |
| **[HYPOTHESIS]** | Plausible but unproven for KHwarizmi | "Hierarchical reasoning may improve planning" |
| **[DESIGN DECISION]** | Architectural choice based on trade-off analysis | "Use Top-2/8 MoE with ablation gate" |
| **[OPEN RESEARCH]** | No satisfactory solution currently exists | "Optimal memory hierarchy structure" |

**Avoid Unsupported Claims:**
- ❌ "HRM proves small models can beat frontier AI"
- ✅ "HRM provides evidence that recurrent hierarchical computation can produce strong results on specific reasoning tasks under small parameter budgets [RESEARCH FINDING]"

---

## Part III: Core Reasoning Philosophy

### III.1 Islamic/Medieval Scientific Reasoning Traditions

Khwarizmi draws formalizable reasoning principles from classical Islamic scholarship:

| Principle | Computational Mechanism |
|-----------|------------------------|
| Precise problem formulation | Semantic representation extraction |
| Decomposition | Task breakdown via router pathways |
| Observation | Multimodal input processing |
| Hypothesis formation | Multi-path reasoning generation |
| Deduction | Symbolic verification and constraint propagation |
| Induction | Pattern recognition in neural core |
| Analogy | Cross-domain transfer via shared representations |
| Proof | Computational verification (math/code) |
| Verification | Consistency heads and self-correction blocks |
| Measurement | Quantitative benchmarking |
| Consistency checking | Contradiction detection module |
| Correction after failure | Failure intelligence and backtracking |
| Systematic investigation | Structured search and planning |
| Simple to complex | Adaptive compute allocation |
| Reject unsupported conclusions | Uncertainty calibration and abstention |

These are **computational mechanisms**, not slogans. Each principle maps to specific architectural components.

### III.2 Modern Reasoning Methods Integration

Khwarizmi integrates state-of-the-art reasoning approaches:

| Method | Implementation |
|--------|----------------|
| Inference-time compute | Adaptive Recurrent Reasoning Cycles (ARRC) |
| Search | Multi-path exploration with pruning |
| Tree reasoning | DAG-based project planning |
| Planning | Symbolic dependency graphs |
| Self-correction | Latent consistency heads |
| Backtracking | Failure-aware state restoration |
| Tool augmentation | Python AST + DAG planners |
| Symbolic reasoning | Deterministic verification tools |
| Neural reasoning | KSC + MoE + latent reasoner |
| Memory systems | Dual architecture with utility gating |
| Mixture-of-experts | Sparse routing with load balancing |
| Selective activation | Cognitive router pathways |
| Structured representations | Semantic frames + DAGs |
| Multimodal reasoning | Unified embedding space |
| Programmatic verification | Execution sandbox + AST checks |
| Efficient inference | Quantization + selective activation |

---

## Part IV: Adaptive Reasoning Architecture

### IV.1 Adaptive Compute Flow

```
Input
  ↓
Problem Understanding (Semantic Extraction)
  ↓
Difficulty Estimation (Meta-Reasoning)
  ↓
Strategy Selection (Router Pathway)
  ↓
Adaptive Compute Allocation (Budget Assignment)
  ↓
Reasoning Execution (KSC + MoE + Memory)
  ↓
Verification (Consistency + Contradiction Check)
  ↓
If Verification Fails:
  ├→ Search Alternative Paths
  ├→ Replan Strategy
  ├→ Backtrack to Decision Point
  └→ Increase Compute Budget
  ↓
Final Result (with Confidence Score)
```

### IV.2 Explicit Stages

| Stage | Component | Function |
|-------|-----------|----------|
| **Difficulty Estimation** | Meta-Reasoning Head | Predicts problem complexity class (Easy/Medium/Hard) |
| **Budget Allocation** | Adaptive Compute Block | Assigns recurrent cycles K ∈ [1, K_max] |
| **Stopping Criteria** | Halting Gate | σ(w_h^T z^(k) + b_h) ≥ 1 - ε |
| **Dynamic Expansion** | Ponder Cost Loss | Allows budget increase if confidence low |
| **Compute-Aware Routing** | Cognitive Router | Routes to appropriate pathway (FAST/CODING/REASONING/etc.) |
| **Confidence Signals** | Consistency Head | Outputs uncertainty estimate |
| **Cost-Aware Reasoning** | Load-Balancing Loss | Prevents expert collapse, ensures efficiency |

---

## Part V: Meta-Reasoning Layer

### V.1 Meta-Reasoning Decisions

The meta-reasoning layer decides **how to solve** the problem, not just the solution:

| Decision Type | Options | Trigger Conditions |
|---------------|---------|-------------------|
| **Answer Directly** | Single-pass FAST pathway | Easy queries, high confidence |
| **Decompose** | PROJECT_PLAN pathway | Multi-step tasks, dependencies detected |
| **Search** | Multi-path activation | Hard problems, low initial confidence |
| **Generate Hypotheses** | REASONING pathway with K ≥ 2 | Ambiguous inputs, multiple interpretations |
| **Use Symbolic Methods** | Tool invocation (Python/DAG) | Code generation, mathematical verification |
| **Invoke Tools** | CODING/VERIFICATION pathways | Code tasks, consistency checks required |
| **Verify** | Consistency head activation | All non-trivial outputs |
| **Critique** | Self-correction block | Low confidence, high-stakes domains |
| **Backtrack** | State restoration | Contradiction detected, verification failed |
| **Replan** | DAG revision | Task failure, constraint change |
| **Allocate More Compute** | Increase K | Hard difficulty, low halting probability |
| **Stop** | Halting gate saturation | Confidence threshold met or budget exhausted |

### V.2 Implementation

Meta-reasoning is implemented via:
- Extended cognitive router with additional pathways
- Difficulty estimation head on top of KSC state
- Halting gate with learnable threshold
- Confidence calibration layer

---

## Part VI: Failure Intelligence

### VI.1 Failure Analysis Pipeline

```
Attempt
  ↓
Failure Detection (Verification Failed / Contradiction / Timeout)
  ↓
Failure Classification
  ├→ Logical Error
  ├→ Factual Hallucination
  ├→ Computational Mistake
  ├→ Tool Misuse
  ├→ Memory Retrieval Failure
  ├→ Contradiction with Prior Knowledge
  └→ Resource Exhaustion
  ↓
Failure Memory Storage (with metadata)
  ↓
Strategy Update (avoid similar path)
  ↓
New Attempt (modified approach)
  ↓
Verification
```

### VI.2 Failure Classification Schema

| Category | Signature | Response |
|----------|-----------|----------|
| **Logical Error** | Inconsistent deductions | Backtrack + replan reasoning chain |
| **Factual Hallucination** | Contradicts verified knowledge | Query long-term memory + verify |
| **Computational Mistake** | Arithmetic/symbolic error | Invoke computational tool |
| **Tool Misuse** | Incorrect API/tool call | Review tool schema + retry |
| **Memory Failure** | Cannot retrieve relevant fact | Expand memory search + update utility |
| **Contradiction** | Conflicts with prior statement | Detect + backtrack + resolve |
| **Resource Exhaustion** | Exceeded compute budget | Simplify strategy or abort gracefully |

### VI.3 Implementation Requirements

- **Failure Memory:** Dedicated storage in long-term memory with high utility score
- **Error Attribution:** Identify which component caused failure (router, KSC, MoE, tool, memory)
- **Strategy Adaptation:** Update router weights to avoid failed pathways for similar inputs
- **Repeated-Error Avoidance:** Flag recurring failures and force alternative strategies

---

## Part VII: Contradiction Detection and Backtracking

### VII.1 Contradiction Types

| Type | Detection Method |
|------|------------------|
| **Internal Contradiction** | Consistency head compares statements within same response |
| **External Contradiction** | Output contradicts verified knowledge in long-term memory |
| **Temporal Contradiction** | Current statement conflicts with earlier turn in conversation |
| **Cross-Modal Contradiction** | Text output contradicts image/audio analysis |
| **Logical Contradiction** | A ∧ ¬A detected in reasoning chain |

### VII.2 Backtracking Mechanism

```
Detect Contradiction
  ↓
Identify Affected Branch (which reasoning path?)
  ↓
Locate Decision Point (where did divergence occur?)
  ↓
Restore State (rollback to pre-contradiction state)
  ↓
Change Assumption/Strategy (modify pathway or parameters)
  ↓
Recompute (alternative path)
  ↓
Verify (ensure no new contradictions)
```

### VII.3 Implementation

- **Contradiction Head:** Neural layer trained to detect conflicting statements
- **State Checkpointing:** Save intermediate states at decision points
- **Dependency Tracking:** Maintain graph of which conclusions depend on which assumptions
- **Resolution Strategies:** Predefined responses for each contradiction type

---

## Part VIII: Multi-Path Reasoning

### VIII.1 Adaptive Multi-Path Exploration

```
                 Problem
              /     |     \
           Path A  Path B  Path C
             |       |       |
           Verify  Verify   Verify
              \       |       /
                 Compare
                    ↓
                Best Path (or merge if complementary)
```

### VIII.2 When to Use Multi-Path

| Problem Difficulty | Paths Explored | Rationale |
|--------------------|----------------|-----------|
| **Easy** | 1 (single pass) | No benefit from exploration; waste of compute |
| **Medium** | 1–2 | Optional second path if confidence low |
| **Hard** | 2–3 | Standard multi-path exploration |
| **Very Hard** | 3–5+ | Extensive exploration with pruning |

### VIII.3 Path Diversity Strategies

- **Different Initial Assumptions:** Vary starting premises
- **Different Reasoning Orders:** Change decomposition sequence
- **Different Tool Usage:** Some paths use tools, others don't
- **Different Abstraction Levels:** Concrete vs. abstract reasoning

---

## Part IX: Mathematical Reasoning Architecture

### IX.1 Hybrid Math Processing

**Principle:** Do NOT rely on language prediction alone for mathematics.

```
Mathematical Problem (Natural Language)
  ↓
Understanding (Neural Core extracts equation/constraints)
  ↓
Formulation (Convert to symbolic representation)
  ↓
Computation/Verification (Symbolic/Numerical Engine)
  ↓
Result Validation (Check against constraints)
  ↓
Natural Language Explanation (if requested)
```

### IX.2 Mathematical Components

| Component | Function | Implementation |
|-----------|----------|----------------|
| **Problem Understanding** | Extract mathematical intent | Neural core with math-specialized experts |
| **Symbolic Representation** | Convert to formal notation | Symbolic parser module |
| **Equation Solving** | Algebraic manipulation | SymPy-like symbolic engine (offline) |
| **Numerical Computation** | Arithmetic, calculus | NumPy-compatible numerical backend |
| **Proof Verification** | Check logical validity | Formal proof checker (limited scope) |
| **Exact Arithmetic** | Avoid floating-point errors | Rational number representation |
| **Computational Tools** | Heavy computation | Sandboxed Python execution |

### IX.3 Mathematical Benchmark Ladder

| Level | Target Capability | Benchmark |
|-------|------------------|-----------|
| **Level 1** | Arithmetic, basic algebra | GSM8K ≥ 55% (Nano) |
| **Level 2** | Intermediate algebra, geometry | GSM8K ≥ 75% (Mobile) |
| **Level 3** | Advanced algebra, calculus basics | GSM8K ≥ 85% (Pro) |
| **Level 4** | Competition mathematics | AIME qualification (Ultra) |
| **Level 5** | Research-level proofs | Formal verification assistance (future) |

---

## Part X: Multimodal Intelligence

### X.1 Supported Modalities

| Modality | Nano | Mobile | Pro | Ultra |
|----------|------|--------|-----|-------|
| **Text** | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| **Images** | ✅ Basic | ✅ Good | ✅ Strong | ✅ Advanced |
| **Audio** | ✅ Basic | ✅ Good | ✅ Strong | ✅ Advanced |
| **Documents** | ✅ PDF/text | ✅ +OCR | ✅ +Diagrams | ✅ +Complex layouts |
| **Structured Files** | ✅ JSON/CSV | ✅ +XML/YAML | ✅ +Binary formats | ✅ All common formats |

### X.2 Multimodal Architecture

- **Unified Embedding Space:** All modalities projected to shared semantic representation
- **Modality-Specific Encoders:** Lightweight encoders for each modality
- **Cross-Modal Attention:** Enable reasoning across modalities
- **Selective Activation:** Only activate relevant encoders per task

### X.3 Nano Optimization

Nano must excel at:
- Reading images (OCR, object detection, scene understanding)
- Understanding audio (speech-to-text, speaker intent, emotion)
- Document processing (extract key information, summarize)
- File transformation (convert formats, reorganize content)

---

## Part XI: Global Multilingual Intelligence

### XI.1 Target Languages

Khwarizmi supports **global multilingual capability**, not limited to specific languages:

**Priority Languages:**
1. Arabic (Modern Standard + Dialects)
2. English
3. Chinese (Simplified + Traditional)
4. Spanish
5. French
6. German
7. Russian
8. Japanese
9. Korean
10. Portuguese
11. Hindi/Urdu
12. Turkish
13. Indonesian/Malay
14. Vietnamese
15. Thai

**Additional:** Support for 100+ languages via language-independent semantic reasoning.

### XI.2 Language-Independent Architecture

```
Any Human Language (input)
  ↓
Language Detection + Normalization
  ↓
Semantic Representation (language-neutral)
  ↓
Reasoning (operates on semantics, not surface form)
  ↓
Verification (language-independent checks)
  ↓
Target Language (output, matching user's language)
```

### XI.3 Prompt-Skill Independence

**Goal:** Users should NOT need to learn prompt engineering.

| Input Type | System Response Quality |
|------------|------------------------|
| Expert wording | Optimal |
| Beginner wording | Optimal |
| Slang/colloquial | Optimal |
| Imperfect grammar | Optimal |
| Short/minimal | Optimal |
| Long/verbose | Optimal |
| Mixed languages | Optimal |
| Translated text | Optimal |

**Implementation:**
- Robust semantic extraction from noisy inputs
- Intent recovery even from poorly formulated requests
- Implicit constraint inference
- Context-aware disambiguation

---

## Part XII: Cross-Lingual Consistency

### XII.1 Benchmark Category

Same task expressed in multiple languages and writing qualities:

| Dimension | Measurement |
|-----------|-------------|
| **Task Understanding** | Accuracy across languages |
| **Correctness** | Factual consistency |
| **Reasoning Depth** | Comparable chain length/quality |
| **Mathematical Result** | Identical answers |
| **Code Correctness** | Same functionality |
| **Document Analysis** | Equivalent insights |
| **Instruction Following** | Compliance rate |
| **Factuality** | Hallucination rate |
| **Output Quality** | Human preference scores |

### XII.2 Target

**Semantic and capability equivalence** across languages, not necessarily identical token outputs.

---

## Part XIII: Knowledge Architecture

### XIII.1 Hybrid Knowledge Storage

A 500 MB Nano cannot memorize the world's knowledge as dense weights.

**Solution:** Hybrid architecture combining:

| Knowledge Type | Storage Method | Activation |
|----------------|----------------|------------|
| **Core Reasoning** | Model weights | Always active |
| **Procedural Knowledge** | Model weights + tools | On-demand |
| **Factual Knowledge** | Compressed indexes | Relevance-gated |
| **Semantic Concepts** | Shared multilingual embeddings | Always available |
| **Domain Specialization** | Modular experts | Selective activation |
| **Historical Details** | Long-term memory | Utility-gated retrieval |
| **Rare Knowledge** | External compressed index | On-request loading |

### XIII.2 Knowledge-on-Demand

```
Compressed Knowledge Store
  ↓
Request Arrives
  ↓
Relevance Detection (semantic match)
  ↓
Activate Relevant Knowledge (decompress/load)
  ↓
Use in Reasoning
  ↓
Release/Compress (free memory)
```

**Benefits:**
- Higher information density
- Lower active memory footprint
- Compatible with constrained devices

### XIII.3 Separation of Concerns

| Layer | Function |
|-------|----------|
| **Knowledge Representation** | Language-neutral semantic structures |
| **Language Realization** | Surface form generation in target language |

This avoids redundant storage of same knowledge for each language.

---

## Part XIV: Selective and Heterogeneous Compression

### XIV.1 Capability-Aware Compression

Do NOT compress all components equally.

| Component | Compression Tolerance | Rationale |
|-----------|----------------------|-----------|
| **Reasoning Core (KSC)** | Low (INT8 minimum) | Highly sensitive to quantization |
| **Verification Modules** | None (FP16 required) | Must preserve exact computation |
| **General Language** | Moderate (INT5–INT8) | Tolerates some precision loss |
| **Low-Frequency Knowledge** | High (INT4) | Rarely accessed, can reload |
| **Unused Experts** | N/A (inactive) | Remain compressed until activated |
| **Memory Embeddings** | Moderate (INT6–INT8) | Some degradation acceptable |
| **Router/Gating** | Low (INT8 minimum) | Critical for correct routing |

### XIV.2 Empirical Validation Framework

Measure impact of each compression level:

| Metric | Measurement Method |
|--------|-------------------|
| **Memory Saved** | Compare FP16 vs quantized size |
| **Speed Gained** | Tokens/sec improvement |
| **Accuracy Lost** | Benchmark degradation % |
| **Reasoning Degradation** | Logic/math accuracy drop |
| **Modality Degradation** | Image/audio understanding drop |
| **Language Degradation** | Per-language performance drop |
| **Coding Degradation** | HumanEval pass@1 drop |

**Decision Rule:** If accuracy loss > 2% for >5% speed gain, reconsider compression level.

---

## Part XV: Capability-Aware Memory Allocation

### XV.1 Task-Driven Resource Priority

| Task Type | Prioritized Resources |
|-----------|----------------------|
| **Mathematical** | Math experts, symbolic tools, verification modules |
| **Image-Heavy** | Vision encoder, cross-modal attention |
| **Audio** | Audio encoder, speech processing |
| **Historical Factual** | Long-term memory, knowledge indexes |
| **Coding** | Code experts, Python AST tool, sandbox |
| **Planning** | DAG engine, working memory expansion |
| **Creative** | Divergent thinking modules, aesthetic scorer |

### XV.2 Static vs. Dynamic Allocation

| Approach | Advantages | Disadvantages |
|----------|------------|---------------|
| **Static** | Predictable, simple | Inflexible, may waste resources |
| **Dynamic** | Adapts to task | Requires runtime overhead |

**Recommended:** Hybrid approach—static base allocation + dynamic adjustment based on router pathway.

---

## Part XVI: Efficiency Metrics

### XVI.1 Reasoning > Parameters

Create efficiency metrics:

```
Capability Score / MB        = Intelligence density
Capability Score / RAM       = Runtime efficiency
Capability Score / FLOP      = Compute efficiency
Capability Score / Latency   = Speed-efficiency tradeoff
Capability Score / Energy    = Power efficiency
```

### XVI.2 Speed as First-Class Requirement

Optimize:

| Aspect | Target |
|--------|--------|
| **Startup Time** | < 2 seconds (Nano), < 5 seconds (Ultra) |
| **Inference Latency** | < 50ms per token (Nano), < 100ms (Ultra) |
| **Memory Bandwidth** | Minimize via selective activation |
| **Active Parameters** | Top-K experts only |
| **Routing Efficiency** | < 1ms router overhead |
| **Cache Behavior** | Optimize for CPU cache lines |
| **Model Loading** | Incremental/on-demand loading |
| **Module Activation** | Lazy loading of specialists |
| **Verification Cost** | Proportional to task criticality |
| **Search Cost** | Adaptive depth based on difficulty |

---

## Part XVII: Offline-First and Accessibility

### XVII.1 Design Priorities

| Priority | Implementation |
|----------|----------------|
| **Offline Operation** | No cloud dependencies for core intelligence |
| **Low-Resource Hardware** | Targets devices with <4GB RAM |
| **Privacy** | All processing local; no data exfiltration |
| **Low Energy Use** | Efficient inference, sleep modes |
| **Small Downloads** | ≤500 MB (Nano), ≤2 GB (Ultra) |
| **Fast Startup** | Optimized loading sequences |
| **Broad Language Support** | 100+ languages |
| **Everyday Usefulness** | Practical tasks prioritized |

### XVII.2 Optional Online Mechanisms

Cloud features (if any) must be:
- **Optional**, not required
- **Enhancements**, not core dependencies
- **Privacy-preserving**, with local fallback

---

## Part XVIII: Benchmark Philosophy

### XVIII.1 Avoid Comparative Claims

Do NOT define success as "Better than Model X."

Instead, use **absolute capability thresholds**.

### XVIII.2 Benchmark Matrix

| Category | Subcategories |
|----------|---------------|
| **Reasoning** | Logic, deduction, induction, analogy, planning, decomposition, multi-step |
| **Mathematics** | Arithmetic, algebra, geometry, symbolic, numerical, proof-oriented, difficult |
| **Coding** | Generation, debugging, repository understanding, testing, repair, tool use |
| **Language** | Multilingual understanding, translation, instruction following, low-resource, informal, mixed |
| **Multimodal** | Image understanding, OCR, diagrams, charts, audio, documents |
| **Reliability** | Hallucination, contradiction, uncertainty, calibration, verification success |
| **Prompt Independence** | Expert, weak, sloppy, colloquial, minimal, multilingual prompts |
| **Efficiency** | Size, RAM, latency, energy, active compute |

---

## Part XIX: Adversarial Self-Testing

### XIX.1 Systematic Weakness Identification

Actively test against attempts to break the system:

| Attack Type | Examples |
|-------------|----------|
| **Adversarial Prompts** | Jailbreak attempts, role-play bypasses |
| **Multilingual Adversarial** | Attacks in low-resource languages |
| **Mathematical Traps** | Problems designed to trigger common errors |
| **Logical Contradictions** | Self-contradictory premises |
| **Misleading Documents** | Fake citations, incorrect data |
| **Ambiguous Questions** | Multiple valid interpretations |
| **Coding Edge Cases** | Unusual syntax, recursion limits |
| **Modality Corruption** | Noisy images, distorted audio |
| **Resource Starvation** | Extremely long contexts, memory pressure |
| **Long-Horizon Tasks** | 100+ step projects with changing constraints |
| **Distribution Shifts** | Out-of-domain queries |

### XIX.2 Goal

**No known systematic weakness should remain untested.**

Not perfection, but comprehensive testing coverage.

---

## Part XX: Four-Version Engineering Strategy

### XX.1 Component Sharing

| Component | Nano | Mobile | Pro | Ultra |
|-----------|------|--------|-----|-------|
| **KSC Architecture** | ✅ Shared | ✅ Shared | ✅ Shared | ✅ Shared |
| **Dual Memory** | ✅ (128 slots) | ✅ (256 slots) | ✅ (512 slots) | ✅ (1024 slots) |
| **MoE Experts** | 4 total, 2 active | 6 total, 2 active | 8 total, 2 active | 8 total, 3 active |
| **Router Pathways** | 5 basic | 6 + meta | 7 + meta + failure | All pathways |
| **Compression** | Aggressive (INT4–INT6) | Moderate (INT5–INT8) | Light (INT8–FP16) | Minimal (FP16) |
| **Tools** | Python AST only | + DAG planner | + All tools | + Advanced tools |
| **Context** | 8K | 16K | 32K | 64K+ |
| **Multimodal** | Basic encoders | Standard encoders | Strong encoders | State-of-art encoders |

### XX.2 Build Strategy

```
Shared Core Development
         ↓
    Nano Profile (most constrained)
         ↓
    Mobile Profile (extend capacity)
         ↓
    Pro Profile (add advanced features)
         ↓
    Ultra Profile (maximum capability)
```

**Rationale:** Building for most constrained first ensures efficiency carries through all versions.

### XX.3 Never-Removed Capabilities

The following must exist in ALL versions:

1. Adaptive compute (even if K_max differs)
2. Verification (even if simplified)
3. Contradiction detection
4. Multilingual support
5. Basic multimodal understanding
6. Tool orchestration (at least Python AST)
7. Memory systems (scaled appropriately)

### XX.4 Compression-Tolerant Components

| Component | Can Tolerate Aggressive Compression |
|-----------|-------------------------------------|
| Low-frequency factual knowledge | ✅ Yes |
| Rarely-used expert pathways | ✅ Yes |
| Historical memory entries | ✅ Yes |
| Auxiliary language models | ✅ Yes |
| Core reasoning weights | ❌ No (preserve precision) |
| Verification modules | ❌ No (exact computation required) |
| Router/gating networks | ❌ No (critical for correctness) |

---

## Part XXI: Roadmap Phases (Restructured)

### Phase Structure

Each phase now includes:
- Objective
- Problem Statement
- Architecture Changes
- Algorithms
- Interfaces
- Tests
- Benchmarks
- Acceptance Criteria
- Risks
- Dependencies
- Resource Impact
- Version Impact

---

## Phase 1 — Khwarizmi Core Foundation

### Objective
Establish the shared reasoning core used by all four versions.

### Problem
Previous roadmap treated versions as separate scaling targets. Need unified core architecture.

### Architecture
- KSC state cell (eigenvalue-bounded recurrence)
- Dual memory (short-term + long-term with utility gating)
- Cognitive router with extended pathways
- Sparse MoE with load balancing
- Adaptive compute with learned halting
- Neural reasoning core (latent synthesis + consistency)

### Algorithms
- Selective state update with stability bounds
- Utility-gated memory operations (READ/WRITE/UPDATE/FORGET)
- Top-K noisy gating for MoE
- ACT-style halting with ponder cost
- Latent consistency checking

### Interfaces
- `KhwarizmiConfig` with Nano/Mobile/Pro/Ultra presets
- `CognitiveRouter` with pathway flags
- `DualMemory` with operation interface
- `AdaptiveComputeBlock` with halting control

### Tests
- Core forward/backward pass tests
- Memory operation tests
- Router pathway tests
- MoE load balancing tests
- Halting termination tests

### Benchmarks
- Associative recall (synthetic)
- Sequence modeling (WikiText proxy)
- Compute differentiation (easy vs. hard tasks)

### Acceptance Criteria
- All core tests passing
- No numerical instability over 100K steps
- Compute differentiation: K_avg ≤ 1.2 (easy), ≥ 2.5 (hard)

### Risks
- Numerical instability in recurrent state
- Expert collapse in MoE
- Non-terminating adaptive loops

### Dependencies
- None (foundational)

### Resource Impact
- Base architecture for all versions
- Nano: ~300M params (compressed to ≤500 MB)
- Mobile: ~500M params (compressed to ≤900 MB)
- Pro: ~800M params (compressed to ≤1.5 GB)
- Ultra: ~1.2B params (compressed to ≤2 GB)

### Version Impact
- **All four versions**

---

## Phase 2 — Meta-Reasoning and Difficulty Estimation

### Objective
Add meta-reasoning layer for strategy selection and compute allocation.

### Problem
Current router selects pathways but doesn't estimate difficulty or allocate budgets dynamically.

### Architecture
- Difficulty estimation head on KSC state
- Strategy selector module
- Compute budget allocator
- Stopping criteria manager

### Algorithms
- Supervised difficulty classification (Easy/Medium/Hard)
- Reinforcement learning for strategy selection
- Dynamic budget adjustment based on confidence

### Interfaces
- `MetaReasoningOutput` with difficulty, strategy, budget
- `DifficultyEstimator` class
- `StrategySelector` with decision logging

### Tests
- Difficulty prediction accuracy tests
- Strategy appropriateness tests
- Budget allocation efficiency tests

### Benchmarks
- Difficulty prediction vs. actual solve time correlation
- Strategy success rate by problem type
- Compute efficiency (accuracy per FLOP)

### Acceptance Criteria
- Difficulty prediction accuracy ≥ 80%
- Meta-reasoning adds < 5ms overhead
- Improved accuracy on hard problems with increased budget

### Risks
- Meta-reasoning itself becomes computationally expensive
- Difficulty estimation inaccurate without training data

### Dependencies
- Phase 1 (Core Foundation)

### Resource Impact
- +5M parameters (negligible)
- +2–5ms latency per query

### Version Impact
- **All four versions** (with different budget ranges)

---

## Phase 3 — Failure Intelligence and Backtracking

### Objective
Implement failure detection, classification, memory, and strategy adaptation.

### Problem
Failed attempts currently disappear without learning. Need systematic failure analysis.

### Architecture
- Failure detector module
- Failure classifier (7 categories)
- Failure memory store
- Strategy updater

### Algorithms
- Pattern matching for failure signatures
- Clustering for failure classification
- Reinforcement learning for strategy updates

### Interfaces
- `FailureReport` with category, attribution, severity
- `FailureMemory` with retrieval API
- `StrategyUpdater` with modification hooks

### Tests
- Failure detection sensitivity tests
- Classification accuracy tests
- Strategy adaptation effectiveness tests

### Benchmarks
- Repeated error reduction rate
- Recovery success rate after failure
- Time-to-recovery metric

### Acceptance Criteria
- Detect ≥ 90% of failures
- Classify correctly ≥ 80% of detected failures
- Reduce repeated errors by ≥ 50%

### Risks
- False positive failure detection
- Overfitting to specific failure patterns

### Dependencies
- Phase 1 (Core), Phase 2 (Meta-Reasoning)

### Resource Impact
- +10M parameters
- +10–20ms for failure analysis when triggered

### Version Impact
- **All four versions** (Nano has simpler classifier)

---

## Phase 4 — Contradiction Detection Module

### Objective
Add first-class contradiction detection and resolution.

### Problem
System can generate contradictory statements without awareness.

### Architecture
- Contradiction detection head
- Dependency graph tracker
- State checkpointing system
- Resolution strategy selector

### Algorithms
- Neural contradiction detection (trained on conflicting pairs)
- Graph-based dependency tracking
- State rollback and recomputation

### Interfaces
- `ContradictionDetector` with confidence scoring
- `DependencyGraph` with add/query/remove
- `StateCheckpoint` with save/restore

### Tests
- Contradiction detection precision/recall
- Dependency tracking accuracy
- Successful resolution rate

### Benchmarks
- Contradiction rate in generated text
- Resolution success rate
- User-reported inconsistency rate

### Acceptance Criteria
- Detect ≥ 85% of contradictions
- Resolve ≥ 70% of detected contradictions
- Reduce user-reported inconsistencies by ≥ 60%

### Risks
- Over-detection (flagging non-contradictions)
- Resolution creates new contradictions

### Dependencies
- Phase 1 (Core), Phase 3 (Failure Intelligence)

### Resource Impact
- +8M parameters
- +5–10ms for contradiction checking

### Version Impact
- **All four versions**

---

## Phase 5 — Multi-Path Reasoning Engine

### Objective
Enable adaptive exploration of multiple reasoning paths.

### Problem
Single-path reasoning fails on complex problems requiring exploration.

### Architecture
- Path generator module
- Path executor (parallel or sequential)
- Path comparator and merger
- Adaptive path counter (based on difficulty)

### Algorithms
- Diverse path generation (different assumptions/orders)
- Beam-search style pruning
- Weighted voting or merging

### Interfaces
- `MultiPathConfig` with max_paths, diversity_params
- `PathResult` with solution, confidence, compute_cost
- `PathMerger` with combination strategies

### Tests
- Path diversity measurement
- Comparison accuracy tests
- Adaptive count appropriateness

### Benchmarks
- Multi-path vs. single-path accuracy on hard problems
- Compute efficiency (accuracy gain per additional path)
- Optimal path count by difficulty

### Acceptance Criteria
- Multi-path improves hard problem accuracy by ≥ 15%
- Easy problems use single path ≥ 90% of time
- Compute overhead proportional to difficulty

### Risks
- Exploding compute on trivial problems
- Paths not sufficiently diverse

### Dependencies
- Phase 2 (Meta-Reasoning), Phase 4 (Contradiction)

### Resource Impact
- Variable compute (1× to 5× base)
- +15M parameters for path management

### Version Impact
- **All four versions** (different max_paths: Nano=2, Mobile=3, Pro=4, Ultra=5+)

---

## Phase 6 — Mathematical Reasoning Hybrid Architecture

### Objective
Implement hybrid neural + symbolic mathematical reasoning.

### Problem
Pure language prediction fails on precise mathematical computation.

### Architecture
- Math problem understanding module
- Symbolic representation converter
- Computational backend (SymPy-like + NumPy-like)
- Verification layer

### Algorithms
- Natural language to symbolic parsing
- Equation solving algorithms
- Numerical computation with exact arithmetic
- Proof checking (limited scope)

### Interfaces
- `MathProblemFrame` with parsed components
- `SymbolicEngine` with solve/verify/simplify
- `NumericalEngine` with compute/check_precision

### Tests
- Parsing accuracy on word problems
- Solver correctness on benchmark suites
- Verification reliability tests

### Benchmarks
- GSM8K accuracy
- MATH dataset performance
- Computational error rate

### Acceptance Criteria
- GSM8K ≥ 55% (Nano), ≥ 75% (Mobile), ≥ 85% (Pro), ≥ 90% (Ultra)
- Zero computational errors on verified results
- Clear separation: understanding (neural) vs. computing (symbolic)

### Risks
- Parser fails on ambiguous formulations
- Symbolic engine too slow for interactive use

### Dependencies
- Phase 1 (Core), Phase 4 (Verification)

### Resource Impact
- +20M parameters
- +50–200ms for symbolic computation (task-dependent)

### Version Impact
- **All four versions** (Nano has limited symbolic engine)

---

## Phase 7 — Multimodal Intelligence Integration

### Objective
Add comprehensive multimodal understanding (image, audio, documents).

### Problem
Current architecture is text-centric; multimodality treated as add-on.

### Architecture
- Unified multimodal embedding space
- Modality-specific encoders (vision, audio, document)
- Cross-modal attention mechanism
- Modality router (selective activation)

### Algorithms
- Contrastive learning for alignment
- Modality fusion strategies
- Selective encoder activation

### Interfaces
- `MultimodalInput` with type detection
- `ModalityEncoder` registry
- `CrossModalAttention` module

### Tests
- Modality detection accuracy
- Cross-modal retrieval tests
- Selective activation efficiency

### Benchmarks
- Image understanding (VQA-style)
- Audio transcription accuracy
- Document extraction quality
- OCR accuracy

### Acceptance Criteria
- Image QA accuracy ≥ 70% (Nano), ≥ 85% (Ultra)
- Audio transcription WER ≤ 15% (clean), ≤ 30% (noisy)
- Document extraction F1 ≥ 0.8

### Risks
- Encoders too large for Nano
- Cross-modal attention computationally expensive

### Dependencies
- Phase 1 (Core)

### Resource Impact
- Nano: +50M (basic encoders)
- Mobile: +100M (standard encoders)
- Pro: +200M (strong encoders)
- Ultra: +400M (advanced encoders)

### Version Impact
- **All four versions** (different encoder capacities)

---

## Phase 8 — Global Multilingual Tokenizer and Semantics

### Objective
Build tokenizer and semantic layer supporting 100+ languages.

### Problem
Current tokenizer minimal; lacks broad multilingual optimization.

### Architecture
- Byte-fallback BPE/Unigram tokenizer
- Language-independent semantic representation
- Cross-lingual transfer mechanisms
- Prompt-skill independence layer

### Algorithms
- Subword tokenization with byte fallback
- Semantic parsing to language-neutral forms
- Intent recovery from noisy inputs

### Interfaces
- `MultilingualTokenizer` with 100+ language support
- `SemanticParser` with language-agnostic output
- `IntentRecovery` module

### Tests
- Roundtrip encoding/decoding per language
- Token fertility comparison (vs. GPT-4/Llama-3)
- Intent recovery from degraded inputs

### Benchmarks
- Token fertility: Arabic < 1.4 tokens/word
- Cross-lingual task consistency
- Prompt-skill independence scores

### Acceptance Criteria
- Support 100+ languages
- Arabic token fertility ≤ 1.4
- Comparable capability across languages (within 5%)
- Prompt-skill independence: expert ≈ beginner ≈ sloppy prompts

### Risks
- Tokenizer too large for Nano
- Low-resource languages underperform

### Dependencies
- Phase 1 (Core)

### Resource Impact
- +10M parameters (tokenizer vocabulary)
- Negligible latency impact

### Version Impact
- **All four versions** (shared tokenizer)

---

## Phase 9 — Knowledge-on-Demand Architecture

### Objective
Implement compressed knowledge with relevance-gated activation.

### Problem
Cannot store all knowledge in active memory; need selective loading.

### Architecture
- Compressed knowledge index
- Relevance detector
- On-demand loader
- Release/compress mechanism

### Algorithms
- Semantic similarity for relevance
- Incremental decompression
- LRU-style caching for loaded knowledge

### Interfaces
- `CompressedKnowledgeStore` with load/unload
- `RelevanceDetector` with threshold tuning
- `KnowledgeCache` with eviction policy

### Tests
- Relevance detection accuracy
- Load/unload latency
- Cache hit rate

### Benchmarks
- Information density (facts per MB)
- Knowledge retrieval latency
- Active memory reduction

### Acceptance Criteria
- 3× information density vs. naive storage
- Knowledge load latency < 50ms
- Active memory reduced by ≥ 40%

### Risks
- Relevance detection misses critical knowledge
- Loading latency impacts responsiveness

### Dependencies
- Phase 1 (Core), Phase 8 (Multilingual Semantics)

### Resource Impact
- Reduces active memory footprint by 40–60%
- +5M parameters for relevance detection

### Version Impact
- **All four versions** (critical for Nano)

---

## Phase 10 — Selective and Heterogeneous Compression

### Objective
Implement capability-aware compression framework.

### Problem
Uniform compression degrades critical components unnecessarily.

### Architecture
- Component-wise compression selector
- Mixed-precision inference engine
- Compression impact profiler

### Algorithms
- Sensitivity analysis per component
- Progressive quantization with validation
- Heterogeneous precision assignment

### Interfaces
- `CompressionProfile` per component
- `MixedPrecisionEngine` with type casting
- `CompressionProfiler` with metrics

### Tests
- Per-component accuracy retention
- Mixed-precision correctness
- Profiler accuracy

### Benchmarks
- Memory saved per compression level
- Speed gained vs. accuracy lost
- Pareto frontier of compression choices

### Acceptance Criteria
- Nano: ≤ 500 MB with < 3% accuracy loss
- Mobile: ≤ 900 MB with < 2% accuracy loss
- Pro: ≤ 1.5 GB with < 1% accuracy loss
- Ultra: ≤ 2 GB with < 0.5% accuracy loss

### Risks
- Over-compression of sensitive components
- Mixed-precision bugs

### Dependencies
- Phase 1 (Core), Phase 6 (Math), Phase 7 (Multimodal)

### Resource Impact
- Major reduction in storage and RAM
- Minor latency for type conversion

### Version Impact
- **All four versions**

---

## Phase 11 — Training Infrastructure and Data Pipeline

### Objective
Build low-resource training infrastructure with Islamic alignment.

### Problem
Need to train all four versions efficiently with quality data.

### Architecture
- QLoRA-based trainer
- Micro-batching with gradient checkpointing
- Islamic data filtering pipeline
- CoT data generation (via Qwen distillation)
- MinHash deduplication

### Algorithms
- QLoRA parameter-efficient fine-tuning
- Gradient accumulation for small batches
- Islamic content filtering (halal/haram classification)
- MinHash LSH for deduplication

### Interfaces
- `LowResourceTrainer` with memory optimization
- `IslamicFilter` with sharia compliance checks
- `CoTGenerator` for reasoning data
- `DeduplicationPipeline`

### Tests
- Training stability tests
- Islamic filter accuracy
- Deduplication ratio measurement

### Benchmarks
- Training throughput (tokens/sec/GPU)
- Islamic compliance (≥ 99% on haram queries)
- Deduplication ratio (≥ 30%)

### Acceptance Criteria
- Train 700M model on single 15GB GPU
- Islamic filter: 0% haram outputs on test set
- Training data deduplicated by ≥ 30%

### Risks
- Training instability with QLoRA
- Islamic filter false positives/negatives

### Dependencies
- Phase 1 (Core), Phase 8 (Multilingual)

### Resource Impact
- Enables efficient training of all versions

### Version Impact
- **All four versions**

---

## Phase 12 — Comprehensive Evaluation Suite

### Objective
Execute full benchmark matrix across all dimensions.

### Problem
Need empirical evidence of capability across all areas.

### Architecture
- Automated benchmark runner
- Statistical significance testing
- Ablation study framework
- Adversarial testing suite

### Algorithms
- Bootstrap resampling for confidence intervals
- Paired t-tests for ablation comparisons
- Adversarial example generation

### Interfaces
- `BenchmarkSuite` with standardized runners
- `AblationStudy` with component toggles
- `AdversarialTester` with attack generators

### Tests
- All benchmark categories from Part XVIII
- Cross-lingual consistency tests
- Prompt-skill independence tests
- Adversarial robustness tests

### Benchmarks
- Full matrix: reasoning, math, coding, language, multimodal, reliability, efficiency

### Acceptance Criteria
- All benchmarks executed for all four versions
- Statistical significance reported (p < 0.01)
- No systematic weaknesses untested

### Risks
- Benchmarks not representative of real use
- Adversarial tests miss novel attack vectors

### Dependencies
- Phases 1–11 (all implementations)

### Resource Impact
- Compute-intensive but one-time per version

### Version Impact
- **All four versions** evaluated

---

## Phase 13 — Quantization and Offline Runtime

### Objective
Implement production quantization and CPU-optimized runtime.

### Problem
Need fast, offline inference on low-resource devices.

### Architecture
- INT4/INT5/INT8 quantization pipeline
- GGUF export compatibility
- SIMD-optimized CPU inference engine
- Incremental model loading

### Algorithms
- Quantization-aware training (optional)
- Post-training quantization with calibration
- SIMD vectorization for matrix ops

### Interfaces
- `QuantizationPipeline` with format selection
- `GGUFExporter` for compatibility
- `CPUInferenceEngine` with SIMD optimizations

### Tests
- Quantization error bounds
- Perplexity degradation (< 2% at INT4)
- Generation speed on CPU

### Benchmarks
- Tokens/sec on consumer CPU (AVX2/NEON)
- Peak RAM during inference
- First-token latency

### Acceptance Criteria
- Nano: ≥ 45 tok/s (CPU), < 400 MB RAM
- Mobile: ≥ 30 tok/s (CPU), < 900 MB RAM
- Pro: ≥ 18 tok/s (CPU), < 1.5 GB RAM
- Ultra: ≥ 10 tok/s (CPU), < 2 GB RAM

### Risks
- Quantization degrades reasoning quality
- CPU engine slower than expected

### Dependencies
- Phase 10 (Compression), Phase 11 (Training)

### Resource Impact
- Major speed and memory improvements

### Version Impact
- **All four versions**

---

## Phase 14 — Edge Deployment and Packaging

### Objective
Package all four versions for target devices.

### Problem
Need distributable packages optimized for each tier.

### Architecture
- Nano: Android APK + lightweight runtime
- Mobile: Cross-platform mobile package
- Pro: Desktop application (Windows/Mac/Linux)
- Ultra: High-end desktop/server package

### Algorithms
- Platform-specific optimizations
- Model bundling and compression
- Update mechanism (offline-compatible)

### Interfaces
- Platform SDKs (Android, iOS, Windows, Mac, Linux)
- Update manager
- Configuration UI

### Tests
- Installation on target devices
- Performance benchmarks per platform
- Battery/thermal stress tests

### Benchmarks
- Startup time per platform
- Battery drain per 1000 tokens
- Thermal throttling behavior

### Acceptance Criteria
- Nano: Works on 5-year-old Android with 2GB RAM
- Mobile: Works on mid-range smartphone
- Pro: Works on consumer laptop
- Ultra: Works on high-end workstation

### Risks
- Platform-specific bugs
- Distribution size exceeds targets

### Dependencies
- Phase 13 (Runtime)

### Resource Impact
- Packaging overhead minimal

### Version Impact
- **All four versions** (different platforms)

---

## Phase 15 — Adversarial Self-Testing and Hardening

### Objective
Systematically identify and fix weaknesses.

### Problem
Need proactive weakness discovery before users find them.

### Architecture
- Adversarial prompt generator
- Multilingual attack suite
- Edge case explorer
- Weakness tracker and fix verifier

### Algorithms
- Genetic algorithm for prompt evolution
- Fuzzing for edge cases
- Regression testing for fixes

### Interfaces
- `AdversarialGenerator` with strategy selection
- `WeaknessTracker` with severity scoring
- `FixVerifier` with regression prevention

### Tests
- All attack types from Part XIX
- Fix effectiveness verification
- Regression prevention

### Benchmarks
- Attack success rate (should decrease over time)
- Time-to-fix for discovered weaknesses
- Residual weakness count

### Acceptance Criteria
- ≥ 10,000 adversarial tests executed
- Attack success rate < 5%
- All critical weaknesses addressed

### Risks
- Adversarial generation misses novel attacks
- Fixes introduce new bugs

### Dependencies
- Phase 12 (Evaluation)

### Resource Impact
- Ongoing compute for testing

### Version Impact
- **All four versions**

---

## Phase 16 — Stable Release and Documentation

### Objective
Finalize stable release with comprehensive documentation.

### Problem
Need production-ready packages with clear usage guidance.

### Architecture
- Versioned releases (v1.0.0)
- User documentation (all languages)
- Developer API documentation
- Benchmark reports

### Algorithms
- Documentation generation
- Release automation
- Continuous integration

### Interfaces
- Public APIs (stable)
- CLI tools
- GUI applications

### Tests
- Full regression suite
- Documentation accuracy checks
- User acceptance testing

### Benchmarks
- Final benchmark scorecard for all versions
- User satisfaction surveys
- Real-world performance metrics

### Acceptance Criteria
- All four versions released and documented
- Benchmarks publicly reported
- User feedback incorporated

### Risks
- Last-minute bugs
- Documentation incomplete

### Dependencies
- Phases 1–15 (all complete)

### Resource Impact
- Final packaging and publishing

### Version Impact
- **All four versions**

---

## Part XXII: Migration from Old Roadmap

### Retained Phases (with modifications)

| Old Phase | New Status | Changes |
|-----------|------------|---------|
| Phase 0 (Audit) | ✅ Complete | Superseded by this document |
| Phase 1 (Math Spec) | ✅ Complete | Integrated into Phase 1 |
| Phase 2 (KSC Prototype) | ✅ Complete | Validated, retained |
| Phase 3 (Dual Memory) | ✅ Complete | Validated, retained |
| Phase 4 (MoE) | ✅ Complete | Validated, retained |
| Phase 5 (Adaptive Compute) | ✅ Complete | Validated, retained |
| Phase 6 (Neural Reasoning) | ✅ Complete | Validated, retained |
| Phase 7A (Excellence Pipeline) | ✅ Partial | Integrated into Phase 11 |
| Phase 7B (Islamic Alignment) | ⚠️ Pending | Integrated into Phase 11 |
| Phase 7C (Domains) | ⚠️ Partial | Integrated into Phases 6–7 |

### Modified Phases

| Old Phase | New Phase | Modification |
|-----------|-----------|--------------|
| Phase 7 (Integration) | Phase 1 | Expanded to include meta-reasoning, failure intelligence |
| Phase 8 (Training Infra) | Phase 11 | Added Islamic alignment, multilingual support |
| Phase 9 (Tokenizer) | Phase 8 | Expanded to 100+ languages, semantic layer |
| Phase 10 (Small Training) | Phase 11 | Now covers all four versions |
| Phase 11 (Evaluation) | Phase 12 | Expanded benchmark matrix |
| Phase 12 (Quantization) | Phase 13 | Added heterogeneous compression |
| Phase 13 (Tools) | Phase 6 | Integrated into math reasoning |
| Phase 14 (Project Intelligence) | Phase 5 | Integrated into multi-path reasoning |
| Phase 15 (Edge Deployment) | Phase 14 | Aligned with four versions |
| Phase 16 (Stable Release) | Phase 16 | Updated for four-version release |

### Removed Phases

| Old Phase | Reason for Removal |
|-----------|-------------------|
| References to 5B–10B+ models | Exceeds 2 GB Ultra limit |
| Cloud-dependent features | Violates offline-first principle |
| Separate "Advanced Tier" | Replaced by Ultra profile |

### New Phases

| New Phase | Purpose |
|-----------|---------|
| Phase 2 (Meta-Reasoning) | Strategy selection, difficulty estimation |
| Phase 3 (Failure Intelligence) | Failure analysis, classification, adaptation |
| Phase 4 (Contradiction Detection) | Conflict detection and resolution |
| Phase 5 (Multi-Path Reasoning) | Adaptive exploration |
| Phase 9 (Knowledge-on-Demand) | Compressed knowledge with selective activation |
| Phase 10 (Heterogeneous Compression) | Capability-aware quantization |
| Phase 15 (Adversarial Testing) | Systematic weakness identification |

---

## Part XXIII: Research Decision Matrix

This matrix evaluates every major research direction against Khwarizmi's objectives. Each hypothesis receives an explicit decision based on evidence, feasibility, and strategic alignment.

| Hypothesis | Problem Solved | Evidence | Expected Impact | Compute Cost | Memory Cost | Feasibility | Novelty | Risk | Priority | Decision |
|------------|----------------|----------|-----------------|--------------|-------------|-------------|---------|------|----------|----------|
| **H1: KSC recurrent architecture matches Transformer reasoning at lower resource cost** | O(L²) attention memory prevents edge deployment | KSC prototype passes stability tests; O(1) memory mathematically proven; Mamba/xLSTM show competitive results | High - enables 100K+ context on <2GB RAM | Medium - requires custom kernels for speed | Low - constant state memory | High - already implemented | Medium - must prove perplexity parity | Medium - training stability concerns | **HIGH** | **KEEP & VALIDATE** - Core architecture; proceed to full benchmark comparison |
| **H2: Adaptive compute (ARRC) improves efficiency without sacrificing capability** | Fixed compute wastes resources on easy tasks | ACT research (Graves 2016); Phase 5 tests show compute differentiation | High - 40-60% compute savings on easy tasks | Low - adds halting gate | Low - minimal state overhead | High - already implemented | Medium - halting stability | Low | **HIGH** | **KEEP & EXTEND** - Proven mechanism; extend to multi-path branching |
| **H3: Multi-strategy branching outperforms single-path depth scaling** | Single-path ARRC may miss better reasoning strategies | Tree-of-thought research; classical AI search theory | Medium-High - improved hard task success | High - multiple parallel paths | Medium - checkpoint storage | Medium - requires meta-reasoning | High - untested | Medium - compute unpredictability | **MEDIUM** | **EXPERIMENT** - Implement as Experiment 001 |
| **H4: Failure-driven learning accelerates capability acquisition** | Uniform training doesn't target weaknesses | Error-driven learning theory; curriculum learning research | High - faster convergence on weak capabilities | Medium - failure classification | Medium - failure memory store | Medium - requires robust failure detector | High - novel approach | Medium - overfitting risk | **HIGH** | **EXPERIMENT** - Implement failure intelligence module |
| **H5: Hierarchical memory improves long-horizon reasoning** | Dual memory may be insufficient for multi-week projects | Cognitive science models; Titans research | Medium - better long-context retention | Medium - additional memory levels | High - hierarchical structure | Medium - design complexity | High - unproven | Medium - added complexity | **DEFER** - Wait for dual memory limitation evidence |
| **H6: Neuro-symbolic hybrids outperform pure approaches** | Neural alone unreliable for math/code; symbolic alone inflexible | Python Brain + neural core integration works; DAG planner success | High - reliable verification without latency penalty | Low - selective tool activation | Low - external tools | High - tools already work | Medium - integration novelty | Low | **HIGH** | **KEEP & EXPAND** - Add more deterministic tools |
| **H7: Small models with efficient architecture match larger models on specific domains** | Scaling laws suggest bigger always better | HRM research; efficient architecture papers | Very High - validates entire project mission | High - requires training runs | Medium - careful resource allocation | Medium - depends on training infrastructure | Very High - if true, paradigm-shifting | High - may fail | **HIGH** | **ULTIMATE OBJECTIVE** - Test after Ultra tier trained |
| **H8: xLSTM-style matrix memory improves over KSC** | KSC scalar state may limit associative recall | xLSTM paper shows mLSTM advantages | Medium - better sequence modeling | Medium - matrix state operations | High - O(d²) state | Medium - would require KSC redesign | Low - copies existing work | High - destabilizes baseline | **LOW** | **REJECT** - KSC already has matrix state; no clear advantage |
| **H9: Gated DeltaNet offers faster associative binding** | KSC update rule may be suboptimal | DeltaNet paper shows fast in-context learning | Medium - improved few-shot learning | Low - similar to KSC | Low - constant state | Medium - implementation effort | Low - incremental variation | Low | **DEFER** - No evidence KSC is inadequate |
| **H10: Titans-style learned neural memory improves ultra-long context** | Dual memory may lose information over 100K+ tokens | Titans paper shows 1M+ context retention | Medium - better very-long-context tasks | High - dual-loop training | Medium - additional memory module | Low - complex integration | Medium - combines attention + memory | High - training instability | **DEFER** - Wait for dual memory limitation evidence |
| **H11: Skill-based MoE experts outperform domain-based** | Current MoE experts may not capture reusable skills | Skill learning research; modular AI theory | Medium - better transfer learning | Low - same MoE infrastructure | Low - same parameter count | Medium - requires skill taxonomy | High - novel approach | Medium - expert collapse risk | **MEDIUM** | **EXPERIMENT** - Test after baseline MoE validated |
| **H12: Priming techniques improve small model reasoning** | Small models may need inference-time guidance | Priming research shows prompt-based improvements | Low-Medium - marginal gains | Low - prompt engineering | Low - no architectural change | High - can test immediately | Low - well-documented | Low | **LOW PRIORITY** - Test as part of benchmark protocol |
| **H13: SSM/Transformer hybrid outperforms pure KSC** | Pure recurrent may lack global attention benefits | Mamba-3 hybrid shows strong results | Medium - best of both worlds | High - dual architecture | High - both state types | Low - major redesign | Low - copies existing work | High - complexity | **REJECT** - Violates sub-quadratic memory principle |
| **H14: Automated architecture search discovers better components** | Human-designed may miss optimal structures | NAS research shows automated discovery possible | Very High - could find novel mechanisms | Very High - requires search infrastructure | High - many candidates | Low - not ready | Very High - black-box risk | Very High - premature | **DEFER** - Stage 1 first (human-designed candidates) |
| **H15: Heterogeneous compression preserves critical precision** | Uniform quantization degrades sensitive components | Precision sensitivity analysis in literature | Medium - better quality at same footprint | Low - post-training | Low - mixed precision storage | High - can implement in Phase 10 | Medium - novel approach | Low | **HIGH** | **KEEP** - Add to Phase 10 quantization plan |

### Decision Summary

| Decision Category | Count | Hypotheses |
|-------------------|-------|------------|
| **KEEP & VALIDATE** | 3 | H1 (KSC), H2 (ARRC), H6 (Neuro-symbolic) |
| **KEEP & EXTEND** | 1 | H2 (Adaptive Compute → Multi-path) |
| **EXPERIMENT** | 3 | H3 (Multi-strategy), H4 (Failure learning), H11 (Skill MoE) |
| **DEFER** | 4 | H5 (Hierarchical memory), H9 (DeltaNet), H10 (Titans), H14 (Auto search) |
| **REJECT** | 2 | H8 (xLSTM replacement), H13 (SSM/Transformer hybrid) |
| **LOW PRIORITY** | 1 | H12 (Priming) |
| **ULTIMATE OBJECTIVE** | 1 | H7 (Small model superiority) |

---

## Part XXIV: Benchmark Strategy

### Benchmark Philosophy

Khwarizmi does not rely on a single benchmark. Evaluation must be multi-dimensional, measuring:

1. **Capability** - Can it solve the task?
2. **Efficiency** - At what resource cost?
3. **Reliability** - How consistent is it?
4. **Calibration** - Does it know when it's uncertain?

### Benchmark Matrix

| Category | Benchmark | Metric | Target (Ultra) | Local Reproducible Variant |
|----------|-----------|--------|----------------|---------------------------|
| **Reasoning** | ARC-Challenge | Accuracy | ≥75% | ARC-Easy subset (local) |
| **Reasoning** | Big-Bench Hard | Subset accuracy | ≥60% | Selected reasoning tasks (local) |
| **Mathematics** | GSM8K | Exact match | ≥75% | GSM8K-100 (local curated) |
| **Mathematics** | Symbolic arithmetic | Accuracy | ≥95% | Generated arithmetic (local) |
| **Coding** | HumanEval | pass@1 | ≥64% | HumanEval-Offline (existing) |
| **Coding** | MBPP | pass@1 | ≥70% | MBPP-curated (local) |
| **Planning** | DAG-Synth-1000 | Cycle-free plans | ≥96% | DAG-100 (local) |
| **Planning** | Long-horizon projects | Consistency | ≥90% | LH-Project-20 (local) |
| **Memory** | NIAH-32K | Retrieval accuracy | ≥98% | NIAH-8K (local) |
| **Memory** | Utility eviction | Precision | ≥94% | Synthetic update test (local) |
| **Language (EN)** | BoolQ | Accuracy | ≥82% | BoolQ-subset (local) |
| **Language (AR)** | ALu-Offline | Accuracy | ≥78% | Existing dataset |
| **Language (EGY)** | Franco-Arabic intent | Accuracy | ≥85% | Curated dialogue set (local) |
| **Reliability** | Contradiction rate | Conflicts per 1000 tokens | ≤5 | Self-consistency check (local) |
| **Reliability** | Verification success | Verified / total | ≥90% | Tool-based verification (local) |
| **Efficiency** | Tokens/sec (CPU) | Generation speed | ≥10 tok/s | Built-in profiler |
| **Efficiency** | RAM footprint | Peak memory | ≤2 GB | Built-in profiler |
| **Efficiency** | FLOPs/token | Compute efficiency | Track only | Estimated via layer calls |

### Local Reproducible Benchmarks

All benchmarks must have a "local" variant that:
- Requires ≤4 GB RAM
- Runs in ≤30 minutes on consumer CPU
- Uses ≤10K test samples
- Is deterministic (fixed seed)
- Requires no internet access

These enable rapid iteration during development without cloud dependencies.

---

## Part XXV: Experiment 001 — Multi-Path Branching Reasoning

### Motivation

Current ARRC implementation uses **single-path depth scaling**: it applies recurrent cycles K ∈ [1, K_max] to one reasoning trajectory. However, some problems require exploring **multiple candidate strategies** rather than simply going deeper on one path.

**Research Question:** Is intelligently allocating reasoning across multiple candidate paths more effective than simply increasing depth on one path?

### Hypothesis

**H3:** For ambiguous inputs, low-confidence initial states, or problems with multiple valid approaches, generating and comparing 2-3 candidate reasoning paths will achieve higher success rates than single-path ARRC with equivalent compute budget.

### Experimental Design

#### Baseline (A): Single-Path ARRC
- Standard ARRC with adaptive K ∈ [1, 4]
- One reasoning trajectory per problem
- Halting based on cumulative confidence

#### Candidate (B): Multi-Path Branching
- Generate 3 candidate initial strategies via router
- Execute each with K ∈ [1, 2] cycles (shallow depth)
- Compare candidate outputs via consistency head
- Select highest-confidence candidate OR reconcile if compatible

#### Control (C): Fixed-Depth Single-Path
- Fixed K = 3 cycles (matches average compute of B)
- One reasoning trajectory
- No branching

### Dataset

**Multi-Strategy Reasoning Benchmark (MSR-100):**
- 100 problems requiring strategy selection
- Categories:
  - Ambiguous natural language (20)
  - Multiple valid solution approaches (30)
  - Problems where first guess often wrong (30)
  - Tasks benefiting from comparison (20)

Curated locally, no internet required.

### Metrics

| Metric | Measurement | Success Threshold |
|--------|-------------|-------------------|
| **Accuracy** | % correct solutions | B > A by ≥8%, B > C by ≥5% |
| **Compute Efficiency** | FLOPs per correct answer | B ≤ C (no worse efficiency) |
| **Calibration** | Confidence-accuracy correlation | B has better calibration than A |
| **Recovery Rate** | % cases where branching recovers from bad initial path | ≥40% of initially-wrong cases |

### Implementation Requirements

1. **Branching Controller** - Extends cognitive router to generate multiple pathway selections
2. **State Checkpointing** - Save KSC state at branch point for restoration
3. **Candidate Comparator** - Compares outputs for agreement/conflict
4. **Reconciliation Module** - Merges compatible partial solutions (if needed)
5. **Selection Gate** - Chooses final output based on confidence + consistency

### Resource Budget

- **Implementation Time:** 2-3 weeks
- **Training:** Fine-tune existing Prototype (150M) - no pretraining from scratch
- **Evaluation:** CPU-only, ≤4 hours for full MSR-100
- **Memory Overhead:** ≤50 MB for checkpointing

### Success Criteria (KEEP)

Multi-Path Branching is **adopted** if:
1. Accuracy improvement ≥8% over Baseline A on MSR-100
2. Accuracy improvement ≥5% over Control C on MSR-100
3. Compute efficiency no worse than Control C (FLOPs per correct answer)
4. Calibration (confidence-accuracy correlation) improves
5. No regression on standard benchmarks (ARC-Easy, GSM8K-100)

### Modification Criteria (MODIFY)

Multi-Path Branching is **modified** if:
1. Accuracy improvement 3-8% (promising but not conclusive)
2. Compute efficiency slightly worse (≤20%)
3. Works only on specific problem categories

→ Action: Optimize branching strategy, reduce candidate count, or apply selectively

### Failure Criteria (REJECT)

Multi-Path Branching is **rejected** if:
1. Accuracy improvement <3% over Baseline A
2. Compute efficiency >30% worse than Control C
3. Introduces instability or training difficulties
4. No improvement on calibration

→ Action: Abandon multi-path approach; focus on single-path depth optimization

### Timeline

| Week | Milestone |
|------|-----------|
| 1 | Implement branching controller and state checkpointing |
| 2 | Implement candidate comparator and selection gate |
| 3 | Create MSR-100 benchmark dataset |
| 4 | Run experiments (A/B/C comparison) |
| 5 | Analyze results, write experiment report |
| 6 | Decision: KEEP / MODIFY / REJECT |

---

## Part XXVI: Experiment Sequencing Rules

Experiments must follow strict sequencing to maximize information gain while minimizing resource waste:

### Rule 1: Baseline First
Never run an experiment without a stable, well-characterized baseline. Current baseline (Phases 0-6) must pass all unit tests before Experiment 001 begins.

### Rule 2: One Variable at a Time
Each experiment isolates ONE architectural change. Do not combine multi-path branching with failure intelligence in the same experiment.

### Rule 3: Predefined Success Criteria
Success/failure criteria must be defined BEFORE running the experiment. Post-hoc criterion adjustment is prohibited.

### Rule 4: Compute Budget Enforcement
Each experiment has a maximum compute budget. If preliminary results clearly exceed budget without conclusive findings, pause and reassess.

### Rule 5: Ablation Required
Any successful experiment must undergo ablation to identify which components are actually responsible for improvement.

### Rule 6: Regression Testing
Experimental modifications must not degrade performance on existing benchmarks. Run full benchmark suite before and after.

### Rule 7: Documentation Mandatory
Every experiment produces a written report including:
- Hypothesis
- Experimental design
- Raw results (tables, plots)
- Statistical analysis
- Decision with justification
- Lessons learned

---

## Part XXVII: Architecture Promotion Rules

An experimental component graduates to production architecture only after satisfying ALL criteria:

### Promotion Criteria

| Criterion | Requirement | Evidence Needed |
|-----------|-------------|-----------------|
| **Benchmark Improvement** | Statistically significant (p<0.01) improvement on relevant benchmarks | Comparison tables, t-test results |
| **Efficiency** | No unacceptable resource overhead | RAM, FLOPs, latency measurements |
| **Stability** | No training or inference instabilities introduced | Training logs, failure rate analysis |
| **Ablation** | Component's individual contribution isolated and verified | Ablation study results |
| **Regression Tests** | No degradation on existing benchmarks | Full benchmark suite comparison |
| **Code Quality** | Production-ready implementation with tests | Unit tests, integration tests pass |
| **Documentation** | Complete documentation of usage and limitations | Architecture docs updated |

### Promotion Process

1. **Experiment Complete** → Written report submitted
2. **Peer Review** → At least one other developer reviews results
3. **Integration Plan** → Detailed plan for merging into production
4. **Staging Branch** → Merge to staging branch first
5. **Full Benchmark Run** → Complete benchmark suite on staging
6. **Production Merge** → Merge to main branch after all checks pass
7. **Version Bump** → Increment minor version number

---

## Part XXVIII: Failure/Abandonment Rules

Not all experiments will succeed. Clear abandonment criteria prevent wasting resources on dead ends.

### Abandonment Criteria

An experiment is abandoned if ANY of the following occur:

| Condition | Trigger | Action |
|-----------|---------|--------|
| **Clear Failure** | Results fail to meet minimum success threshold by wide margin (>50% shortfall) | Stop immediately, document lessons |
| **Resource Exhaustion** | Experiment exceeds budget by >100% without conclusive results | Pause, reassess feasibility |
| **Instability** | Introduces NaN/inf, training divergence, or inference crashes | Debug for 1 week; if unresolved, abandon |
| **Better Alternative Found** | Another approach shows superior results in parallel experiment | Pivot to better approach |
| **Premature** | Prerequisites not yet met (e.g., need better training infrastructure first) | Defer to future phase |
| **Fundamental Flaw Discovered** | Theoretical or empirical flaw invalidates hypothesis | Document flaw, abandon cleanly |

### Clean Abandonment Process

1. **Stop Computation** - Halt all training/evaluation immediately
2. **Document Results** - Write "negative results" report explaining what was tried and why it failed
3. **Archive Code** - Move experimental code to `experiments/archive/` directory
4. **Update Roadmap** - Mark hypothesis as REJECTED with justification
5. **Lessons Learned** - Identify what was learned that informs future directions

### Negative Results Value

Failed experiments are valuable if they:
- Rule out a promising-looking direction
- Reveal unexpected interactions or constraints
- Provide data that refines future hypotheses
- Prevent others from making the same mistake

**All negative results must be documented with the same rigor as positive results.**

---

## Part XXIX: Long-Term Architecture Discovery Program

### Vision

The ultimate goal is not merely to assemble known components but to discover computational mechanisms that deliver unprecedented intelligence-per-compute ratios.

### Staged Approach

| Stage | Method | Prerequisites | Timeline |
|-------|--------|---------------|----------|
| **Stage 1: Human-Designed Candidates** | Researchers propose architectures based on principles | Stable baseline, benchmark suite | Phase 1-3 |
| **Stage 2: Automated Hyperparameter Search** | Grid/random/Bayesian search over component parameters | Stage 1 complete, search infrastructure | Phase 4-6 |
| **Stage 3: Component Selection** | Automated selection among known component types | Stage 2 complete, component library | Phase 7-9 |
| **Stage 4: Architecture Optimization** | Gradient-based or evolutionary architecture search | Stage 3 complete, differentiable search space | Phase 10-12 |
| **Stage 5: Controlled Self-Modification** | System proposes and tests its own modifications | All prior stages, safety guarantees | Phase 13+ |

### Safety Constraints

Every stage enforces:
1. **Human Oversight** - Humans approve progression to next stage
2. **Interpretability** - Discovered architectures must be analyzable
3. **Safety Bounds** - Hard constraints on resource usage, behavior
4. **Reversibility** - Any change can be rolled back
5. **Verification** - Discovered architectures must pass verification suite

### Novel Mechanism Discovery

If existing architectures prove insufficient, Khwarizmi will pursue genuinely novel mechanisms:

**Design Principles:**
- Computational efficiency (FLOPs, memory, latency)
- Stability (bounded gradients, eigenvalues, activations)
- Interpretability (analyzable internal representations)
- Modularity (clean interfaces between components)
- Verifiability (deterministic checking where possible)

**Process:**
1. Identify specific failure mode of existing architectures
2. Formulate mathematical/computational specification for new mechanism
3. Implement minimal prototype
4. Test in isolation with controlled experiments
5. Iterate or discard based on evidence

---

## Part XXX: Nano/Mobile/Pro/Ultra Progression

### Tier Definition

The four tiers represent capacity allocations over a shared cognitive architecture, not fundamentally different systems.

| Parameter | Nano | Mobile | Pro | Ultra |
|-----------|------|--------|-----|-------|
| **Target Footprint** | ≤500 MB | ≤900 MB | ≤1.5 GB | ≤2 GB |
| **Model Parameters** | ~50M | ~150M | ~500M | ~1.5B |
| **Active Parameters** | ~25M | ~75M | ~250M | ~750M |
| **Memory Slots** | 128 | 256 | 512 | 1024 |
| **Context Length** | 8K | 16K | 32K | 64K |
| **MoE Experts** | 4 total, 2 active | 6 total, 2 active | 8 total, 2 active | 8 total, 3 active |
| **Max Recurrent Cycles** | 3 | 4 | 6 | 8 |
| **Search Paths** | 1 (fixed) | 1-2 | 2-3 | 3-5 |
| **Precision** | INT4/INT5 | INT5/INT8 | FP16/INT8 | FP16 |
| **Compression** | Aggressive | Moderate | Light | Minimal |

### Progression Strategy

Development follows a "Nano-first" approach:

1. **Validate on Nano** - If it doesn't work at smallest scale, it won't work at larger scales
2. **Scale to Mobile** - Verify improvements hold with moderate capacity
3. **Optimize for Pro** - Tune for sweet spot of capability vs. resource
4. **Maximize at Ultra** - Push to absolute limits within 2 GB constraint

### Shared Core Guarantee

All tiers guarantee:
- Same reasoning philosophy
- Same cognitive pathways
- Same memory operations
- Same verification mechanisms
- Same tool interfaces

Differences are ONLY in:
- Capacity (parameters, memory slots, context)
- Depth (layers, recurrent cycles)
- Breadth (experts, search paths)
- Precision (quantization level)

---

## Part XXXI: Universal Utility Goals

### Target Users

Khwarizmi ultimately serves diverse users across languages and domains:

| User Category | Primary Needs | Khwarizmi Capabilities Required |
|---------------|---------------|--------------------------------|
| **Students** | Homework help, concept explanation, practice problems | Reasoning, mathematics, language understanding |
| **Teachers** | Lesson planning, exercise generation, grading assistance | Planning, language generation, verification |
| **Programmers** | Code generation, debugging, refactoring, documentation | Coding, AST verification, planning |
| **Researchers** | Literature review, hypothesis generation, experiment design | Reasoning, memory, planning, multilingual |
| **Scientists** | Data analysis, equation solving, unit verification | Mathematics, physics tools, verification |
| **Doctors** | Medical information retrieval, guideline lookup | Knowledge retrieval, verification, caution |
| **Employees** | Document drafting, email composition, scheduling | Language, planning, tool use |
| **Parents** | Educational activities, explanations, safety advice | Reasoning, language, verification |
| **General Users** | Questions, recommendations, conversation | All capabilities |
| **Arabic Speakers** | All above in Arabic (MSA, dialects) | Multilingual, cross-lingual consistency |
| **English Speakers** | All above in English | Language understanding |
| **Franco-Arabic Users** | Mixed-language interaction | Dialect handling, code-switching |

### Development Priority

Capabilities are developed in order of universal utility:

1. **Foundation** - Basic reasoning, language understanding (all users)
2. **Verification** - Reliable, trustworthy outputs (all users)
3. **Efficiency** - Runs on user devices (accessibility)
4. **Specialization** - Domain-specific capabilities (programmers, scientists, etc.)
5. **Optimization** - Performance tuning for specific use cases

### Application Strategy

**Do NOT build separate specialized systems prematurely.**

First establish the underlying cognitive architecture with universal capabilities. Applications come afterward as configurations or fine-tunings of the base system.

---

## Part XXXII: Risk Management

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **KSC fails perplexity parity** | Medium | High | Hybrid fallback; continue optimization |
| **MoE too slow on CPU** | Medium | High | Prune MoE; use dense layers instead |
| **Adaptive compute unstable** | Low | Medium | Fixed-depth fallback; improve training |
| **Memory gating fails to learn** | Medium | Medium | Simplify gates; manual heuristics |
| **Multi-path branching too expensive** | Medium | Low | Reduce candidate count; selective activation |
| **Training infrastructure inadequate** | High | High | Partner with institutions; cloud credits |
| **Quantization degrades quality** | Medium | Medium | Heterogeneous compression; selective precision |

### Strategic Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Frontier models improve faster** | High | Medium | Focus on efficiency niche; different metrics |
| **Funding insufficient** | Medium | High | Open-source community; phased development |
| **Key developers leave** | Medium | Medium | Documentation; knowledge sharing |
| **Hardware requirements increase** | Low | Medium | Aggressive quantization; algorithm improvements |
| **Regulatory restrictions** | Low | Medium | Compliance-first design; transparency |

### Contingency Plans

**If KSC underperforms:**
1. Optimize KSC variants (different gating, state structures)
2. Test alternative recurrent architectures (xLSTM, RWKV principles)
3. Consider lightweight attention hybrids (local attention only)
4. Last resort: accept modest parameter increase for equivalent capability

**If MoE proves impractical:**
1. Reduce expert count
2. Use Top-1 instead of Top-2
3. Replace with conditional computation (dynamic depth)
4. Fall back to dense layers with careful pruning

**If multi-path reasoning fails:**
1. Improve single-path ARRC (more cycles, better halting)
2. Add verification iterations instead of branching
3. Use external tools for difficult problems
4. Accept capability limitation; focus on efficiency wins

---

## Part XXXIII: Immediate Next Action

### Current Status
- Phases 0-6: ✅ Complete and tested
- Router confidence thresholds: ⚠️ Need tuning (3 failing tests)
- Agent loop edge case: ⚠️ Need fix (1 failing test)
- Sandbox import blocking: ⚠️ Need review (5 error tests)
- DAG comprehension handling: ⚠️ Need fix (1 error test)

### Priority Order (Next 4 Weeks)

**Week 1: Stabilize Baseline**
1. Fix router confidence threshold tests
2. Fix agent loop max iteration test
3. Review sandbox import blocking logic
4. Fix DAG comprehension analyzer bug
5. Ensure all Phase 0-6 tests pass

**Week 2: Prepare for Experiment 001**
1. Create MSR-100 benchmark dataset
2. Implement state checkpointing for KSC
3. Design branching controller interface
4. Set up experiment tracking infrastructure

**Week 3-4: Begin Experiment 001 Implementation**
1. Implement branching controller
2. Implement candidate comparator
3. Implement selection gate
4. Run preliminary tests on small scale

### Success Milestones

| Date | Milestone |
|------|-----------|
| Week 1 end | All Phase 0-6 tests passing (604/604) |
| Week 2 end | Experiment 001 infrastructure ready |
| Week 3 end | Branching implementation complete |
| Week 4 end | Preliminary results available |

---

## Appendix A: Glossary

[Existing glossary retained - see previous version]

---

## Appendix B: Configuration Examples

[Existing examples retained - see previous version]

Add these to `khwarizmi/config/tiers.py`:

```python
def get_nano_config() -> KhwarizmiConfig:
    """Nano tier: ≤500 MB footprint, minimal smartphones."""
    return KhwarizmiConfig(
        vocab_size=32768,
        d_model=384,
        n_layers=12,
        n_heads=6,
        num_experts=4,
        top_k_experts=2,
        moe_frequency=3,
        memory_dim=384,
        memory_slots=128,
        short_term_capacity=64,
        max_recurrent_cycles=3,
        max_seq_len=8192,
        gamma_min=0.85,
        gamma_max=0.995,
        default_compression="int5",
        num_pathways=5,
        tier_name="Nano",
        target_footprint_mb=500,
    )


def get_mobile_config() -> KhwarizmiConfig:
    """Mobile tier: ≤900 MB footprint, capable smartphones."""
    return KhwarizmiConfig(
        vocab_size=32768,
        d_model=512,
        n_layers=18,
        n_heads=8,
        num_experts=6,
        top_k_experts=2,
        moe_frequency=3,
        memory_dim=512,
        memory_slots=256,
        short_term_capacity=128,
        max_recurrent_cycles=4,
        max_seq_len=16384,
        gamma_min=0.85,
        gamma_max=0.997,
        default_compression="int5",
        num_pathways=5,
        tier_name="Mobile",
        target_footprint_mb=900,
    )


def get_pro_config() -> KhwarizmiConfig:
    """Pro tier: ≤1.5 GB footprint, frontier reasoning on consumer hardware."""
    return KhwarizmiConfig(
        vocab_size=65536,
        d_model=1024,
        n_layers=30,
        n_heads=16,
        num_experts=8,
        top_k_experts=2,
        moe_frequency=2,
        memory_dim=1024,
        memory_slots=512,
        short_term_capacity=256,
        max_recurrent_cycles=6,
        max_seq_len=32768,
        gamma_min=0.88,
        gamma_max=0.998,
        default_compression="fp16",
        num_pathways=7,
        tier_name="Pro",
        target_footprint_mb=1500,
    )


def get_ultra_config() -> KhwarizmiConfig:
    """Ultra tier: ≤2 GB footprint, maximum capability within accessible limits."""
    return KhwarizmiConfig(
        vocab_size=65536,
        d_model=2048,
        n_layers=48,
        n_heads=32,
        num_experts=8,
        top_k_experts=3,
        moe_frequency=2,
        memory_dim=2048,
        memory_slots=1024,
        short_term_capacity=512,
        max_recurrent_cycles=8,
        max_seq_len=65536,
        gamma_min=0.90,
        gamma_max=0.999,
        default_compression="fp16",
        num_pathways=10,
        tier_name="Ultra",
        target_footprint_mb=2000,
    )
```

---

## Appendix C: Architecture Decision Log

[Existing log retained - see previous version]

---

## Appendix D: Research Hypothesis Registry

[Expanded with decisions from Research Decision Matrix - see Part XXIII]

---

## Appendix E: Architecture Kill List

[Existing list retained - see previous version]

Additional entries:

| Approach | Reason for Exclusion | Conditions for Reconsideration |
|----------|---------------------|-------------------------------|
| **Replacing KSC with xLSTM** | KSC already has matrix state; no evidence xLSTM superior | If KSC fails associative recall benchmarks where xLSTM succeeds |
| **SSM/Transformer hybrid** | Violates sub-quadratic memory principle | Only if pure recurrent fundamentally inadequate |
| **Blind copying of Priming/Titans/DeltaNet** | External research = hypothesis, not decision | If ablation shows clear advantage on Khwarizmi benchmarks |
| **Multi-path without ablation** | Must prove value over single-path depth | Only if single-path demonstrably limited |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-08-11 | Initial 17-phase roadmap |
| 2.0 | 2026-08-11 | Architecture reset with Islamic alignment |
| 3.0 | 2026-08-19 | Added Physics/Art/Creativity domains |
| 4.0 | 2026-08-27 | Complete restructuring for Nano/Mobile/Pro/Ultra |
| 5.0 | 2026-08-28 | Architecture Research & General Intelligence Strategy upgrade |
| **6.0** | **2026-08-29** | **Final Architecture Audit & Roadmap Reconciliation: Added Research Decision Matrix, Experiment 001 specification, Benchmark Strategy, Experiment Sequencing Rules, Architecture Promotion/Abandonment Rules, Long-Term Discovery Program, Universal Utility Goals, Risk Management, Immediate Next Actions** |

---

**End of Khwarizmi Master Roadmap v6.0**

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **ARRC** | Adaptive Recurrent Reasoning Cycles |
| **ACT** | Adaptive Computation Time |
| **KSC** | Khwarizmi State Cell |
| **MoE** | Mixture of Experts |
| **NIAH** | Needle-In-A-Haystack (long-context retrieval test) |
| **QLoRA** | Quantized Low-Rank Adaptation |
| **GGUF** | GPT-Generated Unified Format (quantized model format) |
| **SIMD** | Single Instruction Multiple Data (CPU vectorization) |

---

## Appendix B: Configuration Examples

### Nano Configuration (Example)

```python
def get_nano_config() -> KhwarizmiConfig:
    return KhwarizmiConfig(
        # Model capacity
        vocab_size=32768,
        d_model=384,
        n_layers=12,
        n_heads=6,
        
        # MoE (4 experts, 2 active)
        num_experts=4,
        top_k_experts=2,
        moe_frequency=3,
        
        # Memory (minimal)
        memory_dim=384,
        memory_slots=128,
        short_term_capacity=64,
        
        # Adaptive compute (limited depth)
        max_recurrent_cycles=3,
        
        # Context (constrained)
        max_seq_len=8192,
        
        # Stability
        gamma_min=0.85,
        gamma_max=0.995,
        
        # Compression (aggressive)
        default_compression="int5",
        
        # Pathways (basic set)
        num_pathways=5,
        
        tier_name="Nano",
        target_footprint_mb=500,
    )
```

### Ultra Configuration (Example)

```python
def get_ultra_config() -> KhwarizmiConfig:
    return KhwarizmiConfig(
        # Model capacity (maximum)
        vocab_size=65536,
        d_model=2048,
        n_layers=48,
        n_heads=32,
        
        # MoE (8 experts, 3 active)
        num_experts=8,
        top_k_experts=3,
        moe_frequency=2,
        
        # Memory (extensive)
        memory_dim=2048,
        memory_slots=1024,
        short_term_capacity=512,
        
        # Adaptive compute (deep reasoning)
        max_recurrent_cycles=8,
        
        # Context (very long)
        max_seq_len=65536,
        
        # Stability (tight bounds)
        gamma_min=0.90,
        gamma_max=0.999,
        
        # Compression (minimal)
        default_compression="fp16",
        
        # Pathways (full set)
        num_pathways=10,
        
        tier_name="Ultra",
        target_footprint_mb=2000,
    )
```

---

## Appendix C: Architecture Decision Log

| Decision | Date | Evidence | Confidence | Benchmark | Alternatives Considered | Reason for Choosing | Conditions for Reconsideration |
|----------|------|----------|------------|-----------|------------------------|---------------------|-------------------------------|
| Four-version strategy (Nano/Mobile/Pro/Ultra) | 2026-08-27 | Resource constraints analysis | High | Footprint targets | Single scalable model | Accessibility across device classes | If unified scaling proves more efficient |
| 2 GB Ultra hard limit | 2026-08-27 | Consumer hardware survey | High | RAM availability | No upper limit | Maintain offline accessibility | If edge hardware capabilities increase significantly |
| Shared core architecture | 2026-08-27 | Design principle | High | Consistency tests | Version-specific cores | Ensure consistent reasoning philosophy | If tier-specific optimizations require divergence |
| KSC as primary sequence model | 2026-08-11 | O(1) memory proof, stability bounds | Medium-High | Associative recall benchmarks | Mamba, xLSTM, RWKV, Transformer | Sub-quadratic memory, eigenvalue stability | If SSMs or recurrent alternatives show >20% improvement on reasoning tasks |
| Dual Memory (short/long-term) | 2026-08-11 | Cognitive science analogy, utility gating tests | Medium | Memory operation tests | Single memory, hierarchical memory | Clear separation of concerns, bounded growth | If unified memory shows better efficiency |
| Top-2/8 Sparse MoE | 2026-08-11 | Load balancing loss implementation | Medium | Expert utilization metrics | Dense layers, Top-1/4, no MoE | Parameter efficiency with domain specialization | If MoE overhead exceeds benefit on edge devices |
| ARRC adaptive compute | 2026-08-11 | ACT-style halting with ponder cost | Medium | Compute differentiation benchmarks | Fixed depth, always-maximum compute | Dynamic allocation based on task difficulty | If alternative adaptive methods show better efficiency |
| 5 cognitive pathways | 2026-08-11 | Router implementation | Medium | Pathway selection accuracy | More/fewer pathways, no router | Covers major task categories | If new pathway types emerge from failure analysis |
| Python Brain as external tool | 2026-08-11 | Layered architecture decision | High | Latency measurements | Integrated verification, no tools | Selective activation reduces average latency | If integrated verification proves faster |
| DAG Project Planner as tool | 2026-08-11 | Symbolic/neural separation | High | Planning benchmarks | Neural-only planning | Deterministic dependency reasoning | If neural planning matches symbolic reliability |
| Prompt-skill independence | 2026-08-27 | Usability principle | High | User testing | Skill-requiring prompts | Remove barrier to entry | If skill indicators improve routing |
| Cross-lingual consistency | 2026-08-27 | Multilingual design goal | Medium | Cross-lingual benchmarks | Translation-based approach | True multilingual intelligence | If translation proves sufficient |
| Heterogeneous compression | 2026-08-27 | Precision sensitivity analysis | Medium | Quantization ablation | Uniform quantization | Preserve critical component precision | If uniform quantization achieves same quality |
| Knowledge-on-demand | 2026-08-27 | Information density requirement | Medium | Knowledge retrieval tests | Always-active knowledge | Reduce active parameters | If always-active proves more efficient |
| Adversarial self-testing | 2026-08-27 | Robustness requirement | High | Attack success rate | No adversarial testing | Proactive weakness identification | — |
| No cloud/API fallback | 2026-08-11 | Offline-first tenet | High | Connectivity tests | Hybrid online/offline | Privacy, accessibility, reliability | If offline requirements relax |
| Experimental evaluation required | 2026-08-28 | Research principle | High | Ablation protocol | Paper-based decisions | Evidence-driven architecture | — |

---

## Appendix D: Research Hypothesis Registry

### H1: Recurrent architectures can match or exceed Transformer reasoning quality at lower resource cost

| Field | Value |
|-------|-------|
| **Hypothesis** | KSC-style recurrent architectures with selective state updates can achieve comparable or superior reasoning quality to Transformers while using O(1) decoding memory |
| **Supporting Evidence** | KSC prototype tests pass; O(1) memory is mathematically proven; Mamba/xLSTM research shows competitive results |
| **Counter-Evidence** | Some benchmarks favor attention for associative recall; recurrent models historically harder to train |
| **Experiment** | Compare KSC vs. Transformer vs. Mamba on identical reasoning benchmarks with matched parameter counts |
| **Metric** | Accuracy per MB, accuracy per FLOP, training convergence time |
| **Status** | Under investigation |

### H2: Adaptive compute improves efficiency without sacrificing capability

| Field | Value |
|-------|-------|
| **Hypothesis** | ARRC-style adaptive computation allocates resources efficiently, using less compute on easy tasks and more on hard tasks, improving overall efficiency |
| **Supporting Evidence** | ACT research (Graves 2016); Phase 5 tests show compute differentiation |
| **Counter-Evidence** | Halting stability can be challenging; may add training complexity |
| **Experiment** | Compare fixed-depth vs. ARRC on mixed-difficulty benchmark suite |
| **Metric** | Average compute per token, accuracy on easy/hard subsets, training stability |
| **Status** | Partially validated (tests pass); needs full benchmark evaluation |

### H3: Multi-strategy reasoning outperforms single-path depth scaling

| Field | Value |
|-------|-------|
| **Hypothesis** | Intelligently allocating reasoning across multiple candidate paths is more effective than simply increasing depth on one path |
| **Supporting Evidence** | Tree-of-thought research; branching search in classical AI |
| **Counter-Evidence** | May increase compute unpredictably; requires sophisticated meta-reasoning |
| **Experiment** | Compare single-path ARRC vs. branching multi-candidate system on hard reasoning tasks |
| **Metric** | Success rate, compute efficiency, calibration quality |
| **Status** | Open research (not yet implemented) |

### H4: Failure-driven learning accelerates capability acquisition

| Field | Value |
|-------|-------|
| **Hypothesis** | Systematically analyzing failures, storing failure signatures, and modifying strategies based on failure types accelerates learning compared to uniform training |
| **Supporting Evidence** | Curriculum learning research; error-driven learning theory |
| **Counter-Evidence** | Requires robust failure classification; may overfit to known failure modes |
| **Experiment** | Compare uniform training vs. failure-targeted retraining on specific capability gaps |
| **Metric** | Learning rate, final accuracy, sample efficiency |
| **Status** | Open research (not yet implemented) |

### H5: Hierarchical memory organization improves long-horizon reasoning

| Field | Value |
|-------|-------|
| **Hypothesis** | Organizing memory into functional levels (episodic, semantic, skill, strategy, failure) improves long-horizon task performance compared to flat dual memory |
| **Supporting Evidence** | Cognitive science memory models; Titans research |
| **Counter-Evidence** | Adds architectural complexity; may require more training data |
| **Experiment** | Compare dual memory vs. hierarchical memory on multi-week project tasks |
| **Metric** | Long-context retrieval accuracy, task completion rate, memory efficiency |
| **Status** | Open research (not yet implemented) |

### H6: Neuro-symbolic hybrids outperform pure neural or pure symbolic approaches

| Field | Value |
|-------|-------|
| **Hypothesis** | Combining neural pattern recognition with symbolic verification and planning produces more reliable reasoning than either approach alone |
| **Supporting Evidence** | Python Brain + neural core integration; DAG planner success |
| **Counter-Evidence** | Integration complexity; potential latency from tool calls |
| **Experiment** | Compare neural-only, symbolic-only, and hybrid on code/math/planning tasks |
| **Metric** | Accuracy, verification success rate, latency |
| **Status** | Partially validated (tools work); needs systematic comparison |

### H7: Small models with efficient architecture can match larger models on specific capability domains

| Field | Value |
|-------|-------|
| **Hypothesis** | A well-designed ≤2GB model can match or exceed 10B+ parameter models on specific reasoning, coding, and planning tasks through architectural efficiency |
| **Supporting Evidence** | HRM research; efficient architecture papers; KHwarizmi design principles |
| **Counter-Evidence** | Scaling laws suggest advantages for larger models; limited empirical evidence yet |
| **Experiment** | Benchmark KHwarizmi Ultra against frontier models on targeted capability tests |
| **Metric** | Capability/resource ratios, domain-specific accuracy |
| **Status** | Ultimate research objective (not yet testable until training complete) |

---

## Appendix E: Architecture Kill List

The following approaches are explicitly **excluded** from consideration unless new evidence changes the decision:

| Approach | Reason for Exclusion | Conditions for Reconsideration |
|----------|---------------------|-------------------------------|
| **Replacing KSC merely because Transformer alternatives are popular** | Popularity ≠ suitability; KSC has proven O(1) memory advantage | If KSC consistently fails benchmarks where alternatives succeed by >30% |
| **Blindly copying Mamba/HRM/Titans/Hope architectures** | External research = hypothesis, not decision; must validate for KHwarizmi use case | If ablation studies show clear advantage on KHwarizmi-specific benchmarks |
| **Adding components without ablation evidence** | Violates evidence-driven design principle | Only if component is provably necessary for safety or correctness |
| **Rebuilding entire system before identifying measurable limitation** | Wastes working baseline; violates "no premature rebuild" rule | Only if fundamental flaw discovered that cannot be fixed incrementally |
| **Optimizing benchmark scores while destroying resource efficiency** | Violates core efficiency mission | Never acceptable for production tiers |
| **Cloud/API dependency for core intelligence** | Violates offline-first tenet | Only if offline requirements formally relaxed by project stakeholders |
| **Dense Transformer as primary sequence model** | O(L²) memory violates edge deployment requirements | Only if sub-quadratic alternatives prove fundamentally inadequate |
| **Always-maximum compute for all tokens** | Violates adaptive compute efficiency principle | Only if adaptive mechanisms prove unreliable |
| **Monolithic memory without utility gating** | Violates bounded memory design; risks unbounded growth | Only if gated memory proves too complex to train |
| **Skill-requiring prompts** | Violates prompt-skill independence goal | Only if skill indicators demonstrably improve routing without user burden |
| **Uniform quantization across all components** | Ignores differential precision sensitivity | Only if heterogeneous compression shows no advantage |
| **Architecture decisions based solely on paper claims** | Violates experimental evaluation requirement | Never acceptable; all claims must be empirically validated |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-08-11 | Initial 17-phase roadmap |
| 2.0 | 2026-08-11 | Architecture reset with Islamic alignment |
| 3.0 | 2026-08-19 | Added Physics/Art/Creativity domains |
| 4.0 | 2026-08-27 | Complete restructuring for Nano/Mobile/Pro/Ultra |
| 5.0 | 2026-08-28 | Architecture Research & General Intelligence Strategy upgrade |

---

**End of KHwarizmi Master Roadmap v5.0**
