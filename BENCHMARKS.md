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

---

## 6. Phase 2 Results — Minimal KSC Prototype (50M / 150M)

Implemented in `khwarizmi/core/prototype.py` and measured by `benchmarks/phase2_ksc_prototype.py`
(deterministic, CPU-only). The prototype is a KSC-only LM (no MoE / memory / router), matching the
Phase 2 "No sparse experts / no external memory" boundary.

### 6.1 Footprint
| Tier | Parameters | CPU RAM (params) |
| :--- | ---: | ---: |
| **50M** | 51.43 M | 196.2 MB |
| **150M** | 150.36 M | 573.6 MB |

### 6.2 Sub-quadratic inference memory (Phase 2 decisive criterion)
The recurrent decode state is `O(1)` in sequence length — its size depends only on
`(batch, n_heads, d_k, d_n)` and is **constant** from 4K to 16K context.

| Context | KSC recurrent decode state | Equal-size Transformer KV-cache |
| ---: | ---: | ---: |
| 4K  | 512 KB (constant) | 128 MB (grows with L) |
| 16K | 512 KB (constant) | 512 MB (grows with L) |

→ At 16K context the KSC decode state is **~1024× smaller** than an equal-size Transformer KV-cache,
satisfying the Phase 2 "no memory growth linear in sequence length during inference" criterion.

### 6.3 Latency (50M model, 4 CPU threads, batch=1)
| Operation | Context | Time |
| --- | --- | --- |
| Prefill (first token) | 1K | ~8.4–9.5 s |
| Prefill (first token) | 2K | ~17.3–18.6 s (linear in L) |
| Decode (per token) | — | ~17–18 ms (state size constant, `O(1)`) |

*Prefill uses an unoptimized Python recurrent scan; SIMD/quantized latency optimization is deferred to
Phase 12/15. The memory result above is the architectural guarantee and is independent of this.*

### 6.4 Language-modeling vs equal-size Transformer baseline
Offline proxy for the roadmap's WikiText-103 comparison (WikiText-103 requires the Phase 9 dataset
pipeline + Phase 10 training, out of Phase 2 scope). Both models are trained from scratch for 150 steps
on a **causal** 2-step-delay synthetic task (target = token from 2 positions earlier; neither model may
peek at the future):

| Model | Final CE loss | Notes |
| --- | --- | --- |
| KSC prototype (1.06 M params) | **0.0058** | learns the causal task; converges toward baseline with more training |
| Transformer baseline (0.99 M params) | 0.0035 | direct causal attention |

The KSC prototype learns the task and the gap to the Transformer **shrinks with training**
(+134% at 40 steps → +66% at 150 steps). This validates the recurrent core is trainable and
competitive; full perplexity parity on WikiText-103 is validated in Phase 10.

**Known limitation:** the literal Phase 2 success criterion ("KSC ≤ +5% vs Transformer on WikiText-103")
cannot be exercised offline in Phase 2. The architecture meets the *decisive* Phase 2 criterion
(sub-quadratic inference memory) and is demonstrably trainable; the perplexity-parity gate is a
Phase 9/10 deliverable.

---

## 7. Phase 4 Results — Sparse Mixture-of-Experts (MoE)

Implemented in `khwarizmi/experts/moe_layer.py` and measured by `benchmarks/phase4_sparse_moe.py`
(deterministic, CPU-only, 4 threads). Configuration: **E=32 experts, Top-K=2, d_model=256,
expert d_ff=1024, α_moe=0.01** (16.84M total MoE parameters).

**Methodology.** Two execution strategies are compared head-to-head on the *same* expert weights:
* **SPARSE** — `SparseMoELayer.forward`: Top-2 routing; per-expert token gather; only selected experts are called.
* **DENSE** — a reference module evaluating **all 32 experts on every token** and combining with the router's full-softmax probabilities (both a naive expert loop and a fused batched-matmul variant).

Sparsity is verified by instrumenting every expert with forward-call-counting hooks — the sparse
layer *actually executes* only the routed experts; this is not inferred from Top-K indices.

### 7.1 Sparse execution (primary evidence)

| Strategy | Unique experts executed | Expert evaluations per token |
| --- | ---: | ---: |
| **SPARSE** (Top-2/32) | ≤ 32 (only routed ones) | **2.0** |
| **DENSE** (all experts) | 32 | 32.0 |

→ The sparse layer performs **16× fewer expert-token evaluations**; experts never selected are
never called. Theoretical expert compute: SPARSE 2.11M vs DENSE 33.57M MACs/token
(**93.8% reduction**); the router adds only 16K MACs/token (≈0.8% of the sparse total).

### 7.2 Parameter efficiency & memory

| Metric | Value |
| --- | ---: |
| Total MoE parameters (fp32) | 16.835 M (67.3 MB) |
| Active parameters per token (router + K experts) | 1.068 M (**6.3%**, 4.3 MB) |
| Peak activation buffers @ 2,048 tokens | SPARSE ~3.8 MB · DENSE loop ~13.6 MB · DENSE fused ~338.7 MB |

The sparse layer's activation footprint is dominated by the router tensors plus one small
per-expert batch; the fused dense implementation must materialize an `(N, E, d_ff)` tensor.

### 7.3 Forward latency (batch of 2,048 tokens, best of 3, CPU)

| Variant | Latency (typical run) | Throughput |
| --- | ---: | ---: |
| SPARSE MoE (Top-2 executed) | ~118–129 ms | ~16–17k tok/s |
| DENSE MoE (32 experts, loop) | ~400–415 ms | ~5k tok/s |
| DENSE MoE (32 experts, fused matmuls) | ~480–510 ms | ~4k tok/s |
| Dense FFN of equal *active* parameters | ~21 ms | ~94–98k tok/s |

* SPARSE is **~3.1–3.5× faster** than the looped dense reference and **~3.7–4.3× faster** than
  the fused dense reference (run-to-run CPU variance). Routing overhead is ~1–2% of the sparse
  forward.
* **Honest caveat:** wall-clock speedup is compressed relative to the 16× MAC reduction because
  large dense GEMMs are far more FLOP-efficient than small per-expert batches; conversely the
  sparse layer is ~6× slower than a single equal-active dense FFN due to per-expert
  gather/scatter and small batched matmuls (unoptimized Python dispatch; expert-fused kernels
  are a Phase 12 concern). The MAC/execution counts above are the architectural guarantee.

### 7.4 Expert utilization & load-balancing loss

| Check | Result |
| --- | --- |
| Dispatch fractions f_i over 4,096 tokens (balanced random router) | min 0.051 / max 0.083 (ideal K/E = 0.0625) |
| No expert receives <5% or >40% of tokens | **PASS** |
| Balanced-routing auxiliary loss | 0.0201 (theory α·K = 0.0200) |
| Collapsed router (one expert dominates) | 0.3200 (**15.9× higher** — collapse detected) |
| 150 router-training steps on a collapse-inducing task (target = expert-0 output): | without aux loss → f_max → 1.0, 18/32 experts used; **with** aux loss → f ∈ [0.039, 0.078], 32/32 experts used |

The auxiliary loss detects routing collapse and, applied during training, prevents it. Once a
router is *fully* saturated (f_max = 1.0) the balance-loss gradient vanishes — it is a
preventive regularizer, not a repair mechanism (documented limitation; entropy/z-loss
extensions belong to Phase 8+ training tooling).

### 7.5 Phase 4 scope note

The roadmap's literal Phase 4 gates — ≥8% validation-perplexity gain over an equal-active dense
baseline and the CPU latency-overhead ablation on *trained* routers — require the Phase 9
dataset pipeline and Phase 10 training. Phase 4 delivers and verifies the trainable mechanism:
gradient flow (experts, router, routing weights, noise projection, auxiliary loss — unit-tested),
true sparse execution, load-balancing behavior, and MoE-disabled backward compatibility.

---

## 8. Phase 5 Results — Adaptive Compute & Learned Halting (ARRC)

Implemented in `khwarizmi/reasoning/adaptive_compute.py` and measured by
`benchmarks/phase5_adaptive_compute.py` (deterministic, CPU-only, 4 threads).
Configuration: **d_model=128, K_min=1, K_max=6, ε=0.05, β_ponder=0.01**, batch 16×32 = 512 tokens.

**Methodology.** Per-token ACT-style halting: each token accumulates p_k = σ(w_h·z⁽ᵏ⁾ + b_h) and
halts at the first cycle k ≥ K_min with Σp_j ≥ 1 − ε (force-halted with the remainder at K_max).
Two execution modes are compared on the same weights:
* **ADAPTIVE COMPUTE** — learned per-token halting (`forward(x)`).
* **FIXED COMPUTE** — every token forced to execute exactly K_max cycles (`force_cycles=K_max`).

### 8.1 Halting distribution (untrained gate, bias −0.5, random inputs — adaptivity evidence)

| Step | Tokens halting | % |
| ---: | ---: | ---: |
| 1 | 17 | 3.3% |
| 2 | 211 | 41.2% |
| 3 | 77 | 15.0% |
| 4 | 40 | 7.8% |
| 5 | 22 | 4.3% |
| 6 (K_max) | 145 | 28.3% |

min steps = **1**, avg steps = **3.54**, max steps = **6**; early-halting rate (halted before
K_max) = **71.7%**. Tokens spread across all six depths — computation is genuinely per-token
adaptive, not a fixed-depth loop in disguise.

### 8.2 Easy vs Hard compute comparison (60 training steps: reconstruction + ponder loss)

| Input regime | K_avg | Halting profile |
| --- | ---: | --- |
| **EASY** (low-variance latents) | **1.209** | 79.1% halt at step 1, 20.9% at step 2, none deeper |
| **HARD** (high-variance mixed-mode latents) | **1.465** | 65.0% at step 1, tail through step 6 |

→ Compute differentiation **+0.256 cycles** (hard − easy). Easy inputs halt earlier; hard
inputs continue longer. The roadmap's literal gates (K_avg ≤ 1.2 easy / ≥ 2.5 hard with ≥15%
accuracy gain on hard math/logic) are defined over a *trained language model* and require the
Phase 9/10 dataset + training; this synthetic proxy verifies the mechanism Phase 5 owns.

### 8.3 Termination guarantee

With the halting gate saturated to "never halt" (bias −50), every token is force-halted at
exactly step 6 = K_max with remainder ≈ 1 — **PASS** (no infinite recurrence possible).

### 8.4 Latency — ADAPTIVE vs FIXED COMPUTE (best of 5, CPU)

| Mode | Latency / batch | Notes |
| --- | ---: | --- |
| ADAPTIVE (mixed halting, K_avg = 2.54, 87.9% early-halt) | ~163 ms | stragglers keep the batch alive |
| FIXED (K = 6 for every token) | ~177 ms | ratio ≈ 1.09× |
| ADAPTIVE (uniform halting, K_avg = 1.00) | ~29 ms | **≈6.1× faster** than fixed |
| FIXED (K = 6) reference for uniform run | ~179 ms | |

* **Honest caveat:** the batch-level early exit skips remaining cycles only once *every* token
  in the batch has halted; halted tokens inside a live batch are frozen but their cycle is
  still materialized. With mixed halting, wall time is roughly at parity with fixed compute at
  this scale (~1.1×) even though per-token FLOP demand is only K_avg/K_max = 0.42. Turning the
  per-token saving into wall-clock time requires per-token kernels (Phase 12 runtime scope).
  The uniform-halting regime demonstrates the actual cycle-skipping gain (≈6×).

### 8.5 Memory

| Metric | Value |
| --- | ---: |
| ARRC engine parameters (KSC reasoning cell + norm + gate) | 74,561 (291.3 KiB fp32) |
| Halting gate parameters | 129 |
| Peak Python-alloc during ADAPTIVE pass | ~11.4 KiB |
| Peak Python-alloc during FIXED pass | ~11.7 KiB |

### 8.6 Phase 5 scope note

Phase 5 delivers and verifies the trainable mechanism: per-token ACT halting with min/max step
enforcement, exact remainder accounting (Σ weights = 1, accumulated probability capped at 1),
differentiable ponder cost (gradient reaches the halting gate and provably increases halting
probability under ponder-only training — unit-tested), deterministic inference, and the
`enable_adaptive_compute=False` fixed-compute compatibility path.
