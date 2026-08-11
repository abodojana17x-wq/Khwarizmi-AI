# Khwarizmi AI: Comprehensive Benchmarking & Evaluation Suite
**Document Version:** 2.0 (Phase 0 Architecture Reset)  
**Date:** 2026-08-11  
**Status:** Implementation-Ready Evaluation Blueprint  

---

## 1. Evaluation Philosophy & Pre-Scaling Rule

In alignment with the core project principle—**Measured improvement > theoretical claims**—Khwarizmi AI enforces an absolute Pre-Scaling Rule:

> **PRE-SCALING RULE:**  
> Complete benchmark suites must be created, verified, and integrated into the CI/CD test harness **before** training any model larger than the Prototype Tier (150M).  
> Scaling parameter count to the Edge Tier (1B–3B) or Advanced Tier (5B–10B+) is strictly prohibited unless lower-tier models achieve target threshold gates on these benchmarks.

---

## 2. Five-Pillar Evaluation Framework

Khwarizmi AI evaluates candidate architectures across five orthogonal dimensions:
1. **Intelligence:** Mathematical reasoning, offline Python coding, multilingual language understanding (Modern Standard Arabic - MSA, Egyptian Arabic dialect, English), and instruction following.
2. **Project Intelligence:** Long-horizon software project planning, DAG dependency reasoning, task decomposition, milestone tracking, and failure replanning.
3. **Efficiency:** Offline token generation speed (tokens/sec), Time-To-First-Token (TTFT), peak RAM/VRAM footprint, model disk size, CPU utilization, and energy consumption.
4. **Memory:** Long-range associative recall, Needle-In-A-Haystack (NIAH), utility-gated retention, forgetting accuracy, and memory table stability.
5. **Adaptive Compute:** Direct empirical comparison of **Fixed Compute vs. Adaptive Compute (ARRC)** across Easy, Medium, and Hard problem distributions.

---

## 3. Tiered Threshold Gates (Prototype, Small, Edge, Advanced)

The table below specifies the hard quantitative success gates for each model tier. A tier is certified only when all required thresholds are met.

| Pillar | Benchmark / Metric | Prototype Tier (50M–150M) | Small Tier (300M–700M) | Edge Tier (1B–3B) | Advanced Tier (5B–10B+) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Intelligence** | English General Reasoning (Arc-Easy / BoolQ equiv.) | $\ge 55.0\%$ | $\ge 72.0\%$ | $\ge 82.0\%$ | $\ge 88.0\%$ |
| **Intelligence** | Arabic & MSA Understanding (ALu-Offline Suite) | $\ge 50.0\%$ | $\ge 68.0\%$ | $\ge 78.0\%$ | $\ge 85.0\%$ |
| **Intelligence** | Egyptian Arabic Dialect & Franco-Arabic Intent | $\ge 60.0\%$ | $\ge 75.0\%$ | $\ge 85.0\%$ | $\ge 92.0\%$ |
| **Intelligence** | Offline Python Coding (HumanEval-Offline / MBPP) | $\ge 15.0\%$ | $\ge 42.0\%$ | $\ge 64.0\%$ | $\ge 78.0\%$ |
| **Intelligence** | Mathematical Reasoning (GSM8K-Offline Exact Match) | $\ge 18.0\%$ | $\ge 55.0\%$ | $\ge 75.0\%$ | $\ge 86.0\%$ |
| **Project Intel**| DAG Dependency & Cycle-Free Plan Generation | $\ge 70.0\%$ | $\ge 88.0\%$ | $\ge 96.0\%$ | $\ge 99.0\%$ |
| **Project Intel**| Long-Horizon Consistency (100-Step Software Project) | $\ge 50.0\%$ | $\ge 75.0\%$ | $\ge 90.0\%$ | $\ge 95.0\%$ |
| **Project Intel**| Failure Recovery & Replanning Accuracy | $\ge 60.0\%$ | $\ge 80.0\%$ | $\ge 92.0\%$ | $\ge 97.0\%$ |
| **Efficiency**   | CPU Inference Speed (x86_64 AVX2 / ARM NEON, 4-bit) | $\ge 45\text{ tok/s}$ | $\ge 30\text{ tok/s}$ | $\ge 18\text{ tok/s}$ | $\ge 10\text{ tok/s}$ |
| **Efficiency**   | Time-To-First-Token (TTFT) @ 1,024 prompt tokens | $\le 120\text{ ms}$ | $\le 250\text{ ms}$ | $\le 500\text{ ms}$ | $\le 900\text{ ms}$ |
| **Efficiency**   | Peak Inference RAM Footprint (GGUF 4-bit / FP16 state)| $\le 400\text{ MB}$ | $\le 1.2\text{ GB}$ | $\le 2.4\text{ GB}$ | $\le 6.5\text{ GB}$ |
| **Memory**       | 16K–32K Needle-In-A-Haystack (NIAH) Retrieval Acc | $\ge 85.0\%$ | $\ge 95.0\%$ | $\ge 98.0\%$ | $\ge 99.5\%$ |
| **Memory**       | Utility-Gated Eviction Precision (Obsolete Fact Removal)| $\ge 75.0\%$ | $\ge 88.0\%$ | $\ge 94.0\%$ | $\ge 98.0\%$ |
| **Adaptive**     | Hard Task Accuracy vs Fixed Single-Pass Compute | $+8.0\%$ gain | $+15.0\%$ gain | $+18.0\%$ gain | $+20.0\%$ gain |
| **Adaptive**     | Easy Task Compute Savings (FLOP reduction vs Max) | $\ge 30.0\%$ | $\ge 45.0\%$ | $\ge 55.0\%$ | $\ge 60.0\%$ |

---

## 4. Detailed Benchmark Specifications

### 4.1 Intelligence Suite (Multilingual & Python Coding)
* **Khwarizmi-ALu-Offline (Arabic Language Understanding):**
  * Evaluates formal MSA grammar, semantic entailment, and summarization across 5,000 offline curated Arabic test pairs.
  * Evaluates Egyptian Arabic colloquial dialogue, idiom interpretation, and mixed Franco-Arabic ("ezayyakh ya bashaa, 3ayz a-debug el code da") intent classification.
* **Khwarizmi-HumanEval-Offline:**
  * 164 Python standard-library coding tasks with verified unit tests.
  * Measures pass@1 accuracy both *before* and *after* invoking the deterministic `Python Brain` AST verification tool.
  * **Success Target:** $\ge 64.0\%$ pass@1 on Edge Tier (1B–3B) with AST tool-assisted verification.

### 4.2 Project Intelligence Suite (Specialization Focus)
* **DAG Dependency & Plan Synthesis Benchmark (DAG-Synth-1000):**
  * Evaluates the model's ability to decompose natural language project descriptions into formal structured `Goal`, `Task`, and `Subtask` objects with Directed Acyclic Graph (DAG) dependency arrays.
  * Automatically verified by the `rafig/reasoning` symbolic engine to ensure 0 circular dependencies and 100% prerequisite ordering compliance.
* **Long-Horizon Project Consistency Suite (LH-Project-100):**
  * Simulates a 100-turn software engineering project spanning initial architecture, coding, bug discovery, feature refinement, and constraint modification.
  * Tests whether the model recalls decisions made in Turn 3 during Turn 95 without contradicting earlier constraints.
* **Failure Recovery & Replanning Benchmark (Replan-Bench):**
  * Introduces simulated task failures (e.g., "Task 4 failed: API dependency deprecated; must use standard library SQLite instead").
  * Measures accuracy of revising downstream dependent subtasks without resetting unaffected completed tasks.

### 4.3 Efficiency Suite (CPU/GPU Zero-Cloud Operation)
* **Offline Throughput & TTFT Runner:**
  * Automatically runs on standardized local CPU hardware (4-core AVX2 consumer CPU, 4 GB available RAM) and edge ARM devices.
  * Measures generation tokens per second, prompt prefill rate (tokens/sec), and initial TTFT across 512, 2,048, 8,192, and 32,768 token prompts.
* **Memory Footprint Profile:**
  * Uses `/proc/self/statm` and PyTorch/NumPy memory profilers to record peak resident set size (RSS) during 32K long-context decoding.

### 4.4 Memory Suite (Dual Memory & Eviction)
* **NIAH-32K (Needle-In-A-Haystack):**
  * Inserts 3 random UUID "project secret keys" at varying depths (5%, 50%, 95%) across 32,000 tokens of technical prose.
  * Measures exact-match retrieval accuracy.
* **Dynamic Knowledge Update & Forgetting Benchmark:**
  * Tests the Utility-Gated Persistent Memory store by feeding contradictory updates (e.g., "Update: Database port is now 5433, forget port 5432").
  * Verifies that `FORGET` gates purge old entries and prevent stale recall.

### 4.5 Adaptive Compute Suite (Fixed vs. Adaptive Comparison)
* **Compute Differentiation Suite:**
  * Compares Khwarizmi AI operating in **Fixed Compute Mode** ($K=1$ recurrent cycle, all MoE active) against **Adaptive Compute Mode** (Cognitive Router + ARRC Halting).
  * Measures total FLOPs, wall-clock latency, and accuracy across three buckets:
    1. **Easy Bucket (500 queries):** Greetings, simple Arabic translations, basic Python syntax questions.
    2. **Medium Bucket (500 queries):** Standard function implementation, moderate debugging, single-step math.
    3. **Hard Bucket (500 queries):** Multi-step GSM8K math, complex recursive DAG planning, multi-module AST refactoring.
  * **Success Condition:** Adaptive Compute must reduce FLOPs by $\ge 45\%$ on Easy queries while improving accuracy by $\ge 15\%$ on Hard queries relative to Fixed Compute.

---

## 5. Automated CI/CD Benchmark Harness Integration

All benchmarks specified above are implemented as offline, deterministic Python test harnesses inside `khwarizmi/evaluation/`.
* Running `python -m unittest discover -v tests/test_evaluation_suite.py` executes unit-level verification of all benchmark scoring metrics.
* Full tier certification runs are triggered offline via:
  ```bash
  python -m khwarizmi.evaluation.run_benchmarks --tier=small --offline-mode=True --output=reports/eval_small_tier.json
  ```
