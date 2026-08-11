# Khwarizmi AI: 12-Stage Low-Resource Training & Dataset Strategy
**Document Version:** 2.0 (Phase 0 Architecture Reset)  
**Date:** 2026-08-11  
**Status:** Implementation-Ready Training & Dataset Blueprint  

---

## 1. Low-Resource Training Philosophy

Unlike cloud-scale models trained on tens of thousands of H100 GPUs, Khwarizmi AI optimizes for **free, low-cost, and consumer-grade hardware environments** (e.g., single consumer GPUs with 12 GB–16 GB VRAM, Google Colab free/pro tiers, and local multi-core workstations). 

To maximize intelligence per unit compute, Khwarizmi AI enforces:
* **Quality > Quantity:** 10 billion tokens of ultra-high-quality, clean, deduplicated, and synthetic-verified data outperforms 1 trillion tokens of web-scraped noise.
* **Curricular Sequence Progression:** Training begins on short sequences (512 tokens) and scales up to 16,384 tokens only after baseline convergence.
* **Parameter-Efficient Scaling:** Extensive use of micro-batching, gradient checkpointing, QLoRA/LoRA parameter-efficient fine-tuning, and activation CPU offloading.

---

## 2. The 12-Stage Training Strategy

```
+---------------------------------------------------------------------------------------------------------+
|                                  12-STAGE TRAINING & OPTIMIZATION PIPELINE                              |
+---------------------------------------------------------------------------------------------------------+
| [STAGE 01] Architecture Validation       : Numerical stability, synthetic copying, KSC eigenvalue check |
| [STAGE 02] Phase-Wise Pretraining        : Multilingual (MSA/Egyptian/English), Python, Math foundation |
| [STAGE 03] Instruction Tuning            : Multi-turn conversations, bilingual formatting, system rules |
| [STAGE 04] Reasoning Training            : Structured decomposition, latent ARRC state supervision      |
| [STAGE 05] Coding Training               : Python standard library, AST repair pairs, code debugging    |
| [STAGE 06] Project-Management Training   : Long-horizon DAG generation, milestone tracking, replanning  |
| [STAGE 07] Memory Training               : Explicit READ/WRITE/FORGET supervision, NIAH retrieval       |
| [STAGE 08] Tool-Use Training             : Deterministic Python Brain & Project Planner tool queries    |
| [STAGE 09] Verification Training         : Self-checking, AST error-feedback loops, confidence gating   |
| [STAGE 10] Knowledge Distillation        : Teacher-student distillation of high-value reasoning traces  |
| [STAGE 11] Post-Training Quantization    : Activation-calibrated 4-bit, 5-bit, and 8-bit GGUF export    |
| [STAGE 12] Local Deployment Verification : Memory-mapped CPU SIMD/NEON runtime testing on 4GB hardware|
+---------------------------------------------------------------------------------------------------------+
```

### Stage 1: Architecture Validation (Toy Sequences & Stability)
* **Objective:** Verify that KSC residual blocks, Dual Memory gates, and Cognitive Router policies train without numerical instability.
* **Procedure:** Train on 100 million tokens of synthetic associative recall, prefix-copying, and parity tasks.
* **Pass Criterion:** Zero NaNs across $10^5$ FP16 gradient steps; $>98\%$ associative recall accuracy.

### Stage 2: Pretraining (High-Quality Foundation)
* **Objective:** Establish core language understanding in Modern Standard Arabic (MSA), Egyptian Arabic, English, Python code, and formal mathematics.
* **Curriculum Allocation:**
  * **40% English Prose & Technical Knowledge:** High-quality books, Wikipedia, computer science documentation.
  * **25% Arabic & Dialectal Arabic:** Cleaned MSA literature, Arabic technical articles, colloquial Egyptian Arabic dialogues.
  * **20% Python Code:** Clean standard-library repositories, ast-valid scripts, algorithmic libraries.
  * **15% Mathematics & Logic:** LaTeX proofs, symbolic math, deductive logic problems.
* **Hardware Optimization:** 4K context length, gradient accumulation over micro-batches of size 2, Flash/Associative-Scan KSC kernels.

### Stage 3: Instruction Tuning (Bilingual & Multi-Turn)
* **Objective:** Align the foundation model to respond accurately to user prompts in English, Arabic, Egyptian Arabic, and Franco-Arabic.
* **Dataset:** 500,000 multi-turn instruction pairs with balanced dialectal distribution.
* **Loss Function:** Standard Cross-Entropy over assistant response tokens, masking user prompt tokens.

### Stage 4: Reasoning Training (Latent State ARRC Supervision)
* **Objective:** Train the Adaptive Recurrent Reasoning Cycles (ARRC) and halting gates to solve complex math and logic tasks in latent state space.
* **Supervision Mechanism:** Multi-step reasoning datasets are mapped to intermediate latent state supervision targets. The ponder cost penalty $\mathcal{L}_{\text{ponder}} = \beta \mathbb{E}[K + R]$ is co-optimized to penalize unnecessary recurrent cycles on simple queries.

### Stage 5: Coding Training (Python AST & Debugging Pairs)
* **Objective:** Specialize the model in Python standard-library coding, AST inspection, and issue repair.
* **Data Format:** Curated pairs of `(Broken_Code + Syntax_Error_Diagnostic) -> (Repaired_Code + Structural_Explanation)`.

### Stage 6: Project-Management Training (DAGs & Long-Horizon Planning)
* **Objective:** Specialize Khwarizmi AI as an offline Technical Project Planner.
* **Data Format:** Multi-week software development simulations requiring structured JSON/dataclass DAG outputs (`Goal`, `Task`, `Subtask`, `dependencies`).
* **Supervision:** Enforcing zero-cycle dependency constraints verified by the `rafig/reasoning` engine.

### Stage 7: Memory Training (Explicit READ/WRITE/FORGET Control)
* **Objective:** Train the Cognitive Router and Dual Memory Gating Controller to operate the persistent key-value store.
* **Procedure:** Using long-context needle-in-a-haystack and evolving fact dialogues, optimize the memory gating loss so the model learns when to write high-utility facts ($U > 0.8$) and when to forget obsolete entries.

### Stage 8: Tool-Use Training (Deterministic Tool Orchestration)
* **Objective:** Train the model to emit explicit tool-call tokens for `Python Brain` and `Project Planner` without hallucinating tool arguments.
* **Procedure:** Tool-use fine-tuning on 100,000 structured tool-calling dialogues.

### Stage 9: Verification Training (Self-Correction Loops)
* **Objective:** Train the system to inspect code AST summaries and symbolic DAG validation reports, revising output before presenting it to the user.
* **Procedure:** Reinforcement Learning / Direct Preference Optimization (DPO) favoring self-corrected, syntax-error-free solutions over unverified initial outputs.

### Stage 10: Knowledge Distillation (Teacher-Student Compression)
* **Objective:** Compress high-level reasoning and project planning capabilities from frontier models (via open-license synthetic traces) and larger Khwarizmi models into the **Small Tier (300M–700M)** and **Edge Tier (1B–3B)**.
* **Loss Function:** Combined task Cross-Entropy and KL-Divergence over student-teacher output distributions:
  $$\mathcal{L}_{\text{distill}} = (1 - \alpha) \mathcal{L}_{\text{CE}} + \alpha T^2 D_{\text{KL}}\left( \sigma\left(\frac{z_{\text{student}}}{T}\right) \parallel \sigma\left(\frac{z_{\text{teacher}}}{T}\right) \right)$$

### Stage 11: Post-Training Quantization & Calibration
* **Objective:** Convert FP16/FP32 trained weights into 4-bit, 5-bit, and 8-bit integer formats (`GGUF` format) without perplexity degradation.
* **Procedure:** Perform activation calibration on a balanced 2,048-sample calibration set (English, Arabic, Code, Math) to protect sensitive outlier weights in KSC recurrent projections.

### Stage 12: Local Deployment Verification
* **Objective:** Verify memory-mapped CPU SIMD/AVX2/ARM NEON runtime performance on target hardware (<4 GB RAM).

---

## 3. Comprehensive Dataset Strategy & Pipeline

```
+---------------------------------------------------------------------------------------------------------+
|                                     OFFLINE DATASET INGESTION PIPELINE                                  |
+---------------------------------------------------------------------------------------------------------+
|  [Raw Multi-Domain Sources]                                                                             |
|    ├── English Prose & Tech Docs         (40%)                                                          |
|    ├── Arabic MSA & Egyptian Dialect     (25%)                                                          |
|    ├── Python Code & Standard Library    (20%)                                                          |
|    └── Formal Mathematics & Planning     (15%)                                                          |
|                   |                                                                                     |
|                   v                                                                                     |
|  [Stage A: Toxicity & Harmful Content Screening]      --> Removes harmful, private, or toxic data       |
|                   |                                                                                     |
|                   v                                                                                     |
|  [Stage B: MinHash/LSH Deduplication]                 --> Strips near-duplicates within/across corpora  |
|                   |                                                                                     |
|                   v                                                                                     |
|  [Stage C: Benchmark De-Contamination]                --> 13-gram overlap filter against all test sets  |
|                   |                                                                                     |
|                   v                                                                                     |
|  [Stage D: Synthetic Quality Validation]              --> AST parser check on code; logical check on DAGs|
|                   |                                                                                     |
|                   v                                                                                     |
|  [Stage E: Byte-Fallback BPE Tokenization]            --> Outputs clean, high-fertility token shards    |
+---------------------------------------------------------------------------------------------------------+
```

### 3.1 Strict Quality & De-Contamination Controls

1. **MinHash / LSH Near-Deduplication:**
   * All ingested documents are sharded into 5-shingle sets and hashed via 128-permutation MinHash. Any document exhibiting Jaccard similarity $> 0.75$ with an existing document is discarded to prevent repetition and overfitting.
2. **Benchmark De-Contamination (13-Gram Overlap Filter):**
   * Prior to tokenization, an automated 13-gram index is constructed from all evaluation benchmarks (`HumanEval-Offline`, `GSM8K-Offline`, `DAG-Synth-1000`, `ALu-Offline`).
   * **Absolute Policy:** Any training document sharing even a single 13-gram with a validation or test set is automatically purged.
3. **Synthetic Data Verification:**
   * To eliminate "low-quality synthetic data," all synthetic Python code must parse cleanly through `rafig/python_brain` AST analyzer without syntax errors. All synthetic project plans must pass circular-dependency verification in `rafig/reasoning`.
4. **Dialectal Arabic Quality Control:**
   * Egyptian Arabic datasets are filtered for natural syntactic structure and verified against colloquial Arabic fluency idioms, preventing literal or robotic machine-translation artifacts.
