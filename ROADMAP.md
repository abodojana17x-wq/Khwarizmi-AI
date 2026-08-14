# Khwarizmi AI: 16-Phase Master Engineering & Research Roadmap
**Document Version:** 2.0 (Phase 0 Architecture Reset)  
**Date:** 2026-08-11  
**Status:** Implementation-Ready Blueprint  

---

## Roadmap Principles & Strict Development Rule

> **CRITICAL DEVELOPMENT RULE:**  
> Never move to the next major phase simply because the previous code runs. A phase is complete **only** when its success criteria are met. If a benchmark fails: **STOP. Analyze. Modify. Retest.** Only then continue.  
> Do **NOT** implement features belonging to a future phase prematurely.

---

## Complete 16-Phase Roadmap Overview

```
+---------------------------------------------------------------------------------------------------------+
|                                    KHWARIZMI AI 16-PHASE MASTER PLAN                                    |
+---------------------------------------------------------------------------------------------------------+
| [PHASE 00] Repository Audit + Architecture Reset                  (COMPLETE - 2026-08-11)               |
| [PHASE 01] Mathematical Specification & Hardware Verification     (IMMEDIATE NEXT PHASE)                |
| [PHASE 02] Minimal Khwarizmi State Cell (KSC) Prototype           (50M-150M Prototype Tier)             |
| [PHASE 03] Dual Memory Architecture Prototype                     (Short-Term + Persistent KV Store)    |
| [PHASE 04] Sparse Mixture-of-Experts (MoE) Prototype              (Top-K Gating & Balance Validation)   |
| [PHASE 05] Adaptive Compute & Learned Halting                     (ARRC Recurrent Reasoning Cycles)     |
| [PHASE 06] Neural Reasoning Core                                  (Latent Synthesis & Self-Correction)  |
| [PHASE 07] Full Khwarizmi Neural Core Integration                 (Unifying KSC + MoE + Memory + Router)|
| [PHASE 08] Low-Resource Training Infrastructure                   (Micro-batch, Colab, Checkpointing)   |
| [PHASE 09] Dataset Pipeline & Offline Multi-Lingual Tokenizer     (BPE Byte-Fallback, Clean Pipeline)   |
| [PHASE 10] Small Model Training & Optimization                    (300M-700M Small Tier Training)       |
| [PHASE 11] Comprehensive Evaluation & Mandatory Ablation Testing  (Statistical Component Verification)  |
| [PHASE 12] Offline Inference Optimization & Quantization          (4/5/8-bit GGUF, SIMD CPU Engine)     |
| [PHASE 13] Offline Assistant & Layered Local Tool Integration     (Agent Router + Python/DAG Tools)     |
| [PHASE 14] Project Intelligence Specialization                    (Long-Horizon Planning & Verification)|
| [PHASE 15] Edge Deployment & Hardware Verification                (1B-3B Edge Tier, Android/Low-RAM)    |
| [PHASE 16] Scaling Analysis & Stable Release                      (Production Certification)            |
+---------------------------------------------------------------------------------------------------------+
```

---

## Detailed Phase Specifications

### Phase 0: Repository Audit + Architecture Reset
* **Objective:** Audit the existing codebase, isolate legacy symbolic components into an optional external tool layer, establish the formal architecture blueprint, and create implementation-ready documentation.
* **Exact Deliverables:** `ARCHITECTURE.md`, `ROADMAP.md`, `RESEARCH.md`, `EXPERIMENTS.md`, `BENCHMARKS.md`, `TRAINING.md`, `MEMORY.md`, `DEPLOYMENT.md`, `CONTRIBUTING.md`, updated `README.md`.
* **Files/Modules:** Top-level `.md` files; preservation of existing `rafig/` Python files without breaking tests.
* **Dependencies:** Python standard library (`unittest`).
* **Tests:** `python -m unittest discover -v` (141 existing tests passing).
* **Benchmarks:** Zero-regression check on existing unit tests.
* **Success Criteria:** 100% of documentation files written, mathematically justified, and verified; 0 broken existing tests.
* **Failure Criteria:** Any missing required section from Phase 0 deliverables; test regression.
* **Exit Criteria:** Approved architecture specification by engineering team.
* **What Must NOT Be Implemented Yet:** No neural network training, no new PyTorch models, no deletion of legacy python files.

---

### Phase 1: Mathematical Specification & Hardware Verification
* **Objective:** Formally specify and mathematically prove stability bounds, eigenvalue constraints, and algorithmic complexity for the Khwarizmi State Cell (KSC), Dual Memory gating, and Cognitive Router.
* **Exact Deliverables:** Mathematical verification scripts simulating state eigenvalues over 100,000 steps; reference NumPy/CPU implementations of KSC recurrence.
* **Files/Modules:** `khwarizmi/core/ksc_cell.py`, `tests/test_ksc_cell.py`.
* **Dependencies:** NumPy, SciPy (for numerical verification in testing only).
* **Tests:** Numerical stability tests ensuring no NaN/Inf over $10^5$ sequence iterations; gradient flow check.
* **Benchmarks:** Synthetic associative recall memory benchmark (copying and needle retrieval on toy sequences).
* **Success Criteria:** 0% NaN/overflows across $10^5$ random FP32/FP16 sequence steps; associative recall $>98\%$ on 4096-token synthetic sequences.
* **Failure Criteria:** Numerical divergence or exploding gradients in KSC recurrence.
* **Exit Criteria:** Verified NumPy/PyTorch reference operators passing all stability tests.
* **What Must NOT Be Implemented Yet:** No MoE layers, no Multi-Head Attention, no Large Language Model training.

---

### Phase 2: Minimal Khwarizmi State Cell (KSC) Prototype (50M–150M)
* **Objective:** Implement a clean, modular KSC residual block and train a small 50M–150M parameter prototype model on language modeling and sequence tasks.
* **Exact Deliverables:** `khwarizmi/core/ksc_block.py`, `khwarizmi/core/embeddings.py`, Prototype Tier config (`50M`, `150M`).
* **Files/Modules:** `khwarizmi/core/*`, `tests/test_ksc_block.py`.
* **Dependencies:** PyTorch (CPU & GPU).
* **Tests:** Forward/backward pass shape and gradient tests; sequence invariance tests.
* **Benchmarks:** WikiText-103 perplexity comparison against equal-sized Transformer baseline; First-token latency and memory footprint at 4K and 16K token context.
* **Success Criteria:** KSC matches or outperforms baseline Transformer perplexity on WikiText-103 while using $\le 50\%$ RAM at 16K context length.
* **Failure Criteria:** Worse perplexity ($>5\%$ higher) than standard GRU/Transformer baseline or memory growth linear in sequence length during inference.
* **Exit Criteria:** Completed ablation report proving sub-quadratic memory and competitive language modeling capability.
* **What Must NOT Be Implemented Yet:** No sparse experts, no external memory databases, no symbolic tools.

---

### Phase 3: Dual Memory Architecture Prototype
* **Objective:** Build and test the Short-Term Working State buffer and Utility-Gated Long-Term Persistent Memory store with learned `READ`, `WRITE`, `UPDATE`, and `FORGET` operations.
* **Exact Deliverables:** `khwarizmi/memory/short_term.py`, `khwarizmi/memory/long_term.py`, `khwarizmi/memory/gating.py`.
* **Files/Modules:** `khwarizmi/memory/*`, `tests/test_dual_memory.py`.
* **Dependencies:** PyTorch, standard library serialization.
* **Tests:** Memory write/eviction unit tests; utility score decay tests; key-value associative recall tests.
* **Benchmarks:** Long-Context Needle-In-A-Haystack (NIAH) up to 32,000 tokens; Dynamic Knowledge Update Benchmark.
* **Success Criteria:** NIAH retrieval accuracy $\ge 95\%$ across 32K context; successful eviction of bottom-10% lowest utility items without losing high-utility facts.
* **Failure Criteria:** Memory table saturation leading to retrieval degradation ($<70\%$ accuracy) or catastrophic forgetting of high-utility keys.
* **Exit Criteria:** Experimental validation report confirming Dual Memory improves long-horizon recall without increasing KSC state size.
* **What Must NOT Be Implemented Yet:** No MoE routing, no project planner tool integration.

---

### Phase 4: Sparse Mixture-of-Experts (MoE) Prototype
* **Status:** Implemented — `khwarizmi/experts/moe_layer.py` + `khwarizmi/experts/specialists.py`, 57 tests in `tests/test_moe.py`, benchmark in `benchmarks/phase4_sparse_moe.py`; see `ARCHITECTURE.md` §4.4 and `BENCHMARKS.md` §7.
* **Objective:** Implement Sparse Top-$K$ Noisy Gated MoE layers with auxiliary load-balancing loss and measure parameter efficiency and latency.
* **Exact Deliverables:** `khwarizmi/experts/moe_layer.py`, `khwarizmi/experts/specialists.py`.
* **Files/Modules:** `khwarizmi/experts/*`, `tests/test_moe.py`.
* **Dependencies:** PyTorch.
* **Tests:** Router load balancing tests (ensuring no expert receives $<5\%$ or $>40\%$ of tokens); gradient flow through Top-$K$ gating.
* **Benchmarks:** Per-token inference latency (CPU and GPU); active parameter vs quality trade-off curve on multi-domain test set.
* **Success Criteria:** MoE achieves $\ge 8\%$ validation perplexity improvement over dense baseline of equal *active* parameters, with auxiliary balance loss $\le 0.05$.
* **Failure Criteria:** Expert collapse (1-2 experts dominating $>60\%$ of tokens) or CPU inference latency overhead exceeding 20%.
* **Exit Criteria:** Mandatory Ablation Decision: If MoE fails latency/quality thresholds on CPU, it is removed; otherwise, approved for Small Tier integration.
* **What Must NOT Be Implemented Yet:** No adaptive halting, no full agent CLI.

---

### Phase 5: Adaptive Compute & Learned Halting
* **Status:** Implemented — per-token ACT-style ARRC halting in `khwarizmi/reasoning/adaptive_compute.py` (`AdaptiveComputeBlock` + `PonderCostLoss`), 57 tests in `tests/test_adaptive_compute_phase5.py`, benchmark in `benchmarks/phase5_adaptive_compute.py`; see `ARCHITECTURE.md` §4.5 and `BENCHMARKS.md` §8.
* **Objective:** Implement Adaptive Recurrent Reasoning Cycles (ARRC) allowing dynamic depth/iteration halting based on token complexity.
* **Exact Deliverables:** `khwarizmi/reasoning/adaptive_compute.py`, ponder cost loss module.
* **Files/Modules:** `khwarizmi/reasoning/adaptive_compute.py`, `tests/test_adaptive_compute.py`.
* **Dependencies:** PyTorch.
* **Tests:** Halting condition verification ($\sum p_k \ge 1 - \epsilon$); ponder loss penalty gradient check.
* **Benchmarks:** Easy vs Hard Task Compute Comparison Benchmark (measuring average cycles $K_{\text{avg}}$ on arithmetic vs basic factual recall).
* **Success Criteria:** $K_{\text{avg}} \le 1.2$ on easy tasks and $K_{\text{avg}} \ge 2.5$ on hard tasks, achieving $\ge 15\%$ accuracy improvement on hard math/logic problems over fixed single-pass compute.
* **Failure Criteria:** Infinite loops ($K = K_{\max}$ on $>20\%$ of samples) or zero compute differentiation between easy and hard queries.
* **Exit Criteria:** Verified adaptive compute efficiency without regression on fast-path queries.
* **What Must NOT Be Implemented Yet:** No Python AST repair tool integration.

---

### Phase 6: Neural Reasoning Core
* **Objective:** Implement latent space reasoning refinement, self-checking, and error detection without exposing raw verbose ASCII chain-of-thought.
* **Exact Deliverables:** `khwarizmi/reasoning/latent_reasoner.py`, structured reasoning synthesis head.
* **Files/Modules:** `khwarizmi/reasoning/*`, `tests/test_reasoning_core.py`.
* **Dependencies:** PyTorch.
* **Tests:** Step-wise state consistency tests; self-checking error detection accuracy tests.
* **Benchmarks:** GSM8K-Offline and Symbolic Logic Reasoning Suite.
* **Success Criteria:** GSM8K-Offline exact match $\ge 65\%$ on Small Tier (300M-700M) without emitting external scratchpad tokens.
* **Failure Criteria:** Reasoning accuracy below standard chain-of-thought baseline or state divergence during latent iterations.
* **Exit Criteria:** Benchmarked superiority of latent reasoning over unverified CoT.
* **What Must NOT Be Implemented Yet:** No end-user chat UI, no external agent loops.

---

### Phase 7: Full Khwarizmi Neural Core Integration
* **Objective:** Combine KSC blocks, Dual Memory, Cognitive Router, Sparse MoE (if retained), and Adaptive Compute into a unified, clean neural architecture.
* **Exact Deliverables:** `khwarizmi/core/model.py` (unified architecture), `khwarizmi/routing/router.py`.
* **Files/Modules:** `khwarizmi/core/*`, `khwarizmi/routing/*`, `tests/test_unified_core.py`.
* **Dependencies:** PyTorch.
* **Tests:** End-to-end forward/backward integration tests across all 5 cognitive router pathways (`FAST`, `CODING`, `REASONING`, `PROJECT_PLAN`, `VERIFICATION`).
* **Benchmarks:** Holistic System Latency Benchmark across all paths; memory footprint on CPU with 4GB RAM budget.
* **Success Criteria:** 100% correct routing of canonical test queries; end-to-end memory footprint $<1.8\text{ GB}$ for Small Tier (500M FP16).
* **Failure Criteria:** Memory leaks across successive inference calls or router misclassification rate $>10\%$.
* **Exit Criteria:** Complete unified architecture passing all integration tests.
* **What Must NOT Be Implemented Yet:** No production pretraining runs, no UI deployment.

---

### Phase 8: Low-Resource Training Infrastructure
* **Objective:** Build a robust, memory-efficient trainer supporting micro-batching, activation checkpointing, QLoRA, and single-GPU/Google Colab execution.
* **Exact Deliverables:** `khwarizmi/training/trainer.py`, `khwarizmi/training/losses.py`.
* **Files/Modules:** `khwarizmi/training/*`, `tests/test_trainer.py`.
* **Dependencies:** PyTorch, Accelerate/minimal training utils.
* **Tests:** Checkpoint save/resume tests; OOM recovery tests; multi-loss gradient scaling tests.
* **Benchmarks:** Training throughput (tokens/sec/GPU) and maximum trainable context length on a single 15GB VRAM GPU (Colab T4/L4).
* **Success Criteria:** Ability to train a 700M Khwarizmi model at 4096 context length on a single 15GB GPU without OOM.
* **Failure Criteria:** Out-of-memory errors on 15GB GPU or loss instability/NaNs during mixed-precision training.
* **Exit Criteria:** Verified trainer ready for full-scale data ingestion.
* **What Must NOT Be Implemented Yet:** No actual large-scale dataset training runs.

---

### Phase 9: Dataset Pipeline & Offline Multi-Lingual Tokenizer
* **Objective:** Create an offline-first byte-fallback BPE/Unigram tokenizer optimized for Arabic, Egyptian Arabic, English, and Python, alongside a deduplicated dataset processing pipeline.
* **Exact Deliverables:** `khwarizmi/data/tokenizer/`, `khwarizmi/data/deduplication.py`, trained vocabulary (`vocab_khwarizmi_64k.json`).
* **Files/Modules:** `khwarizmi/data/*`, `tests/test_tokenizer_pipeline.py`.
* **Dependencies:** Standard library, minimal tokenizer library (SentencePiece or pure-Python BPE implementation).
* **Tests:** Roundtrip encoding/decoding tests across Arabic, Egyptian Arabic, English, and Python code; MinHash/LSH near-deduplication tests; N-gram benchmark de-contamination tests.
* **Benchmarks:** Token fertility rate (tokens per word/line of code) compared to GPT-4/Llama-3 tokenizers; processing speed (MB/sec).
* **Success Criteria:** Arabic token fertility $<1.4$ tokens/word (vs $>2.2$ in legacy tokenizers); 100% removal of test-set contaminated 13-grams.
* **Failure Criteria:** Syntax destruction on Python AST code during tokenization or dialectal Arabic token fragmentation.
* **Exit Criteria:** Released 64K vocabulary and validated, de-contaminated pretraining dataset.
* **What Must NOT Be Implemented Yet:** No full model training.

---

### Phase 10: Small Model Training & Optimization (300M–700M)
* **Objective:** Train the Khwarizmi Small Tier (300M–700M parameters) using the validated architecture and clean multi-lingual/coding dataset.
* **Exact Deliverables:** Trained weights (`khwarizmi_small_700m.pt`), training log reports, loss curves.
* **Files/Modules:** `khwarizmi/config/tiers.py`, training run scripts.
* **Dependencies:** PyTorch, Phase 8 Trainer, Phase 9 Dataset.
* **Tests:** Checkpoint integrity and deterministic reproducibility tests.
* **Benchmarks:** Pretraining validation perplexity across English, Arabic, Egyptian Arabic, Python code, and Reasoning corpora.
* **Success Criteria:** Converged training loss without gradient spikes; validation perplexity outperforming equal-sized dense transformer baselines.
* **Failure Criteria:** Loss divergence or stagnation.
* **Exit Criteria:** Approved Small Tier weights ready for formal evaluation and ablation.
* **What Must NOT Be Implemented Yet:** No quantization, no edge deployment.

---

### Phase 11: Comprehensive Evaluation & Mandatory Ablation Testing
* **Objective:** Execute the complete evaluation suite across Intelligence, Project Intelligence, Efficiency, Memory, and Adaptive Compute, and conduct formal ablation testing.
* **Exact Deliverables:** Complete Evaluation Report (`eval_results_phase11.json`), formal Ablation Scorecard.
* **Files/Modules:** `khwarizmi/evaluation/*`, `tests/test_evaluation_suite.py`.
* **Dependencies:** PyTorch, standard evaluation harness.
* **Tests:** Statistical significance tests ($p < 0.01$) across ablation variants.
* **Benchmarks:** Intelligence Suite, Project Intelligence Suite, Memory NIAH, Adaptive Compute Suite.
* **Success Criteria:** Small Tier model passes all baseline thresholds; every retained component (KSC, Memory, Router, MoE, Adaptive Compute) demonstrates statistically significant measurable gain.
* **Failure Criteria:** Any component failing to provide measurable gain relative to its compute/memory cost.
* **Exit Criteria:** Execution of "Prune or Retain" decisions; final architecture lock for production optimization.
* **What Must NOT Be Implemented Yet:** No mobile deployment.

---

### Phase 12: Offline Inference Optimization & Quantization
* **Objective:** Implement 4-bit, 5-bit, and 8-bit integer quantization and export models to GGUF and custom CPU-optimized local runtimes.
* **Exact Deliverables:** `khwarizmi/runtime/engine.py`, `khwarizmi/runtime/quantization.py`, quantized GGUF artifacts (`khwarizmi_small_4bit.gguf`).
* **Files/Modules:** `khwarizmi/runtime/*`, `tests/test_runtime.py`.
* **Dependencies:** NumPy, standard C/C++ runtime bindings / pure Python memory-mapped engine.
* **Tests:** Quantization error bounds checks; perplexity degradation tests (<2% increase at 4-bit).
* **Benchmarks:** Tokens/sec on CPU (AVX2/NEON); First-Token Latency (TTFT); peak RAM consumption during inference.
* **Success Criteria:** Generation speed $\ge 25\text{ tokens/sec}$ on consumer CPU (x86_64 AVX2 / ARM64); peak RAM $<1.2\text{ GB}$ for 4-bit Small model.
* **Failure Criteria:** Generation speed $<15\text{ tokens/sec}$ on CPU or perplexity degradation $>5\%$ from FP16 baseline.
* **Exit Criteria:** Stable, hyper-fast local runtime engine verified offline.
* **What Must NOT Be Implemented Yet:** No end-user project agent orchestration.

---

### Phase 13: Offline Assistant & Layered Local Tool Integration
* **Objective:** Connect the Khwarizmi Offline Agent Layer and Cognitive Router to the optional deterministic tools (`Python Brain` AST analyzer and `Project Planner` DAG engine).
* **Exact Deliverables:** `khwarizmi/agent/agent_loop.py`, `khwarizmi/tools/python_brain/*`, `khwarizmi/tools/project_planner/*`.
* **Files/Modules:** `khwarizmi/agent/*`, `khwarizmi/tools/*`, `tests/test_agent_tools.py`.
* **Dependencies:** Phase 12 Runtime, Python standard library `ast`.
* **Tests:** Tool-invocation accuracy tests; zero-latency overhead test on fast-path non-tool requests.
* **Benchmarks:** End-to-End Tool Verification Benchmark (measuring accuracy of detecting and repairing subtle Python bugs using Python Brain AST feedback).
* **Success Criteria:** 100% correct AST issue detection when Python Brain tool is invoked; 0 milliseconds tool-loading overhead on simple conversational queries.
* **Failure Criteria:** Uncontrolled tool-calling loops or calling AST verification on non-coding requests.
* **Exit Criteria:** Layered assistant successfully executing code verification and DAG planning offline.
* **What Must NOT Be Implemented Yet:** No edge mobile packaging.

---

### Phase 14: Project Intelligence Specialization
* **Objective:** Specializing and fine-tuning the system for large project planning, dependency DAG reasoning, milestone tracking, and failure recovery over extended horizons.
* **Exact Deliverables:** Long-horizon project management fine-tuned weights, project state tracking schemas.
* **Files/Modules:** `khwarizmi/tools/project_planner/`, long-horizon evaluation scripts.
* **Dependencies:** Phase 13 Layered Agent.
* **Tests:** DAG cyclic dependency detection tests; replanning consistency tests after simulated task failures.
* **Benchmarks:** Long-Horizon Project Management Suite (100-step software project simulation with changing constraints).
* **Success Criteria:** 100% DAG dependency compliance across multi-step project plans; $\ge 90\%$ successful replanning after simulated task failure without violating hard constraints.
* **Failure Criteria:** Hallucinating completed prerequisite tasks or forgetting project constraints after 10 turns.
* **Exit Criteria:** Certification of Khwarizmi AI as an expert offline Technical Project Planner.
* **What Must NOT Be Implemented Yet:** No scaling to Advanced Tier (>5B) unless benchmarks justify it.

---

### Phase 15: Edge Deployment & Hardware Verification (1B–3B Tier)
* **Objective:** Scale up to the Edge Tier (1B–3B parameters) if benchmark evidence justifies scaling, and verify deployment on low-RAM edge environments (Android/ARM/consumer edge PCs).
* **Exact Deliverables:** `khwarizmi_edge_2b_4bit.gguf`, edge device verification report.
* **Files/Modules:** `khwarizmi/runtime/*`, edge testing suite.
* **Dependencies:** Phase 12 Runtime.
* **Tests:** Edge device thermal, battery, and memory-constraint stress tests.
* **Benchmarks:** Edge Latency & Energy Benchmark (tokens/watt and battery drain per 1,000 tokens generated).
* **Success Criteria:** Sustained $\ge 15\text{ tokens/sec}$ on ARM64 edge processor with total RAM footprint $<2.5\text{ GB}$.
* **Failure Criteria:** Thermal throttling or OS out-of-memory process termination on 4GB edge devices.
* **Exit Criteria:** Stable Edge Tier release verified across target hardware.
* **What Must NOT Be Implemented Yet:** No unnecessary scaling to 10B+ models.

---

### Phase 16: Scaling Analysis & Stable Release
* **Objective:** Finalize the stable release of Khwarizmi AI, publish comprehensive research benchmarks, and provide empirical scaling laws to determine if Advanced Tier (5B–10B+) is warranted.
* **Exact Deliverables:** Stable Release `v1.0.0` (Offline Package), empirical scaling law report, final documentation package.
* **Files/Modules:** Complete repository.
* **Dependencies:** All previous phases completed and verified.
* **Tests:** Full regression test suite across all 16 phases.
* **Benchmarks:** Complete Khwarizmi AI Master Evaluation Scorecard.
* **Success Criteria:** All 16 phases passing 100% of defined success criteria and ablation gates.
* **Failure Criteria:** Any unresolved P0/P1 architectural or offline deployment bug.
* **Exit Criteria:** Public stable offline release of Khwarizmi AI.

---

## Phase 1 Checklist (Actionable & Rigorous)

Before writing any code for **Phase 1: Mathematical Specification & Hardware Verification**, the engineering team must execute and check off the following verification list:

- [ ] **1. Mathematics Specification Check**
  - [ ] Write LaTeX/PDF document mathematically defining the continuous-time dynamics and discretized retention gates $\bar{A}_t$ of the Khwarizmi State Cell.
  - [ ] Prove analytically that for $\gamma_{\min} = 0.85$ and $\gamma_{\max} = 1 - \epsilon$, the recurrent state $S_t$ has eigenvalues strictly bounded inside the unit circle, preventing overflow over infinite sequence lengths.
  - [ ] Document exact FLOPS and memory complexity per token for decoding ($O(1)$ state) and prefill ($O(L \log L)$ associative scan).
- [ ] **2. Hardware & Precision Verification Plan**
  - [ ] Establish reference NumPy FP32 and FP16 simulation scripts for 100,000 sequential recurrent update steps.
  - [ ] Verify that FP16 accumulation does not suffer from underflow when state retention $\bar{A}_t \to 1$.
  - [ ] Ensure reference operators do not use any cloud or internet-dependent libraries.
- [ ] **3. Tool & Legacy Isolation Verification**
  - [ ] Confirm that legacy `rafig/reasoning` and `rafig/python_brain` test suites (`python -m unittest discover -v`) remain 100% passing and decoupled from Phase 1 sequence operators.
- [ ] **4. Benchmark Baseline Preparation**
  - [ ] Define the exact synthetic associative recall and needle-in-a-haystack verification sequence generators for testing Phase 1 KSC operators.

---

## Immediate Next Action

1. **Lock Phase 0 Documentation:** Ensure all 9 master documentation files (`ARCHITECTURE.md`, `ROADMAP.md`, `RESEARCH.md`, `EXPERIMENTS.md`, `BENCHMARKS.md`, `TRAINING.md`, `MEMORY.md`, `DEPLOYMENT.md`, `CONTRIBUTING.md`) are committed and available in the repository root.
2. **Open Phase 1 Implementation Branch / Track:** Begin Phase 1 by drafting `khwarizmi/core/ksc_cell.py` as a purely mathematical NumPy/PyTorch reference operator and implementing `tests/test_ksc_cell.py` to validate eigenvalue stability across 100,000 token steps.
