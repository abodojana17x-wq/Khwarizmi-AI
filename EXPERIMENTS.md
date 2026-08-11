# Khwarizmi AI: Experimental & Ablation Testing Protocol
**Document Version:** 2.0 (Phase 0 Architecture Reset)  
**Date:** 2026-08-11  
**Status:** Implementation-Ready Experimental Blueprint  

---

## 1. Core Experimental Philosophy & Mandatory Ablation Rule

In accordance with the fundamental research objective—**Maximum intelligence and reasoning capability per unit of compute and memory**—every architectural component in Khwarizmi AI must be empirically justified through controlled experimentation.

> **MANDATORY ABLATION RULE:**  
> A component survives **only** if it provides measurable, statistically significant value ($p < 0.01$) that outweighs its memory, latency, and training complexity costs.  
> **If an experiment shows that a component (e.g., Sparse MoE or Adaptive Halting) does not provide sufficient benefit: REMOVE OR REDESIGN IT.**  
> Theoretical claims never override empirical benchmark measurements.

---

## 2. Mandatory Ablation Testing Schedule

To isolate the contribution of each architectural innovation, Khwarizmi AI enforces a 6-tier progressive ablation protocol across all target model scales (50M Prototype, 700M Small, 2B Edge).

```
+---------------------------------------------------------------------------------------------------------+
|                                    PROGRESSIVE ABLATION TEST LADDER                                     |
+---------------------------------------------------------------------------------------------------------+
|  [Tier 0] BASELINE             : Standard Dense Transformer / GRU baseline with identical parameters.    |
|  [Tier 1] BASELINE + KSC       : Replaces attention with Khwarizmi State Cell (KSC) blocks.             |
|  [Tier 2] BASELINE + MEMORY    : Adds Utility-Gated Dual Memory (Short-Term + Long-Term KV store).       |
|  [Tier 3] BASELINE + MOE       : Adds Sparse Top-2/8 Mixture-of-Experts routing layers.                 |
|  [Tier 4] BASELINE + ADAPTIVE  : Adds Adaptive Recurrent Reasoning Cycles (ARRC) and halting gates.     |
|  [Tier 5] FULL ARCHITECTURE    : Unifies KSC + Memory + Router + MoE (if retained) + Adaptive Compute.   |
+---------------------------------------------------------------------------------------------------------+
```

### 2.1 Ablation Scorecard & Pruning Gates

For every ablation experiment, the evaluation harness records six hard metrics. Any component failing the defined Pruning Gate is removed from production tiers.

| Component Under Test | Primary Quality Metric | Max Allowed Memory Overhead | Max Allowed Latency Overhead | Mandatory Pruning Gate (Removal Condition) |
| :--- | :--- | :--- | :--- | :--- |
| **Khwarizmi State Cell (KSC)** | Perplexity on WikiText-103 / MSA Corpus | $\le 10\%$ of baseline RAM at 16K context | $\le 20\%$ of linear baseline | Remove if KSC perplexity $>5\%$ higher than Transformer baseline while using $>50\%$ RAM at 16K tokens. |
| **Dual Memory System** | Needle-In-A-Haystack (NIAH) 32K Accuracy | $\le 150\text{ MB}$ persistent table RAM | $\le 5\text{ ms}$ per retrieval query | Remove if NIAH accuracy $<80\%$ or if memory table indexing increases generation latency by $>15\%$. |
| **Cognitive Router** | Multi-path routing accuracy & compute saving | $\le 10\text{ MB}$ parameter RAM | $\le 1\text{ ms}$ classification overhead | Remove if router misroutes $>10\%$ of coding/reasoning queries or saves $<20\%$ net compute on easy tasks. |
| **Sparse Experts (MoE)** | Coding & Math multi-domain accuracy | $\le 1.5\text{ GB}$ total expert RAM on 3B Edge | $\le 15\%$ CPU latency overhead | **HIGH SCRUTINY:** Remove if MoE does not improve multi-domain accuracy by $\ge 8\%$ over equal *active* parameter dense baseline, or if RAM footprint causes OS paging on 4GB edge hardware. |
| **Adaptive Compute (ARRC)** | Hard reasoning task accuracy (GSM8K/Logic) | $\le 5\text{ MB}$ state buffer RAM | Dynamic ($\le 1.2\times$ on Easy, $\le 3.0\times$ on Hard) | Remove if Adaptive Compute fails to beat fixed single-pass compute by $\ge 10\%$ on hard reasoning tasks. |

---

## 3. Detailed Experimental Designs

### 3.1 Experiment EX-01: KSC Numerical Stability & Sub-Quadratic Scaling
* **Objective:** Prove that KSC achieves $O(1)$ decoding memory and linear prefill compute without numerical overflow over ultra-long sequences ($100{,}000$ tokens).
* **Experimental Setup:**
  * Generate synthetic random sequences of length $L \in \{512, 2048, 8192, 32768, 100000\}$.
  * Run forward and backward passes using (a) Baseline Multi-Head Attention, (b) Mamba-2 reference, and (c) Khwarizmi State Cell (KSC) in FP32 and FP16.
* **Measured Variables:** Peak RAM consumption (MB), forward pass time (ms), backward pass time (ms), maximum absolute recurrent state eigenvalue $\max_i |\gamma_{t,i}|$.
* **Hypothesis Under Test:** KSC RAM consumption remains constant ($<75\text{ MB}$) across all $L$, whereas Attention RAM scales as $O(L^2)$ and OOMs at $L > 32768$ on 8GB hardware.
* **Pass/Fail Criteria:** Pass if KSC exhibits 0 overflows/NaNs across $10^5$ steps and $\le 50\%$ RAM of baseline at 16K context.

### 3.2 Experiment EX-02: Dual Memory Utility Gating & Forgetting
* **Objective:** Validate that the learned `READ`/`WRITE`/`UPDATE`/`FORGET` gates retain high-utility project decisions while successfully evicting noise.
* **Experimental Setup:**
  * Construct a 32,000-token synthetic project dialogue containing 50 critical "Architectural Decision Records" (ADRs) interspersed with 10,000 conversational filler tokens.
  * Populate Long-Term Memory table ($\text{capacity} = 128\text{ slots}$).
* **Measured Variables:** Precision@5 and Recall@5 for ADR retrieval; memory table saturation rate; utility score correlation with true ADR relevance.
* **Hypothesis Under Test:** Utility-Gated Dual Memory retains $>95\%$ of critical ADRs while evicting 100% of conversational filler when capacity is reached.
* **Pass/Fail Criteria:** Pass if NIAH recall $\ge 95\%$ and eviction precision $\ge 90\%$.

### 3.3 Experiment EX-03: Cognitive Router Compute Economy
* **Objective:** Measure net computational savings (FLOPs/token) and latency reduction achieved by the Cognitive Router on mixed workloads.
* **Experimental Setup:**
  * Prepare a benchmark suite of 1,000 queries distributed as: 50% Simple Conversational (`FAST`), 25% Python Code (`CODING`), 15% Multi-step Math (`REASONING`), and 10% Large Project Planning (`PROJECT_PLAN`).
  * Execute suite under (a) Fixed Maximum Compute (All MoE + 3 Reasoning Cycles + Tool check) vs (b) Cognitive Router dispatch.
* **Measured Variables:** Average FLOPs per query, total execution wall-clock time (seconds), output quality score.
* **Hypothesis Under Test:** Cognitive Router reduces total suite FLOPs by $\ge 40\%$ and wall-clock latency by $\ge 35\%$ without degrading coding or planning accuracy by more than $1\%$.
* **Pass/Fail Criteria:** Pass if FLOP reduction $\ge 35\%$ and quality regression $\le 1\%$.

### 3.4 Experiment EX-04: MoE RAM Bandwidth vs Capability on Edge CPU
* **Objective:** Rigorously determine whether Sparse MoE (Top-2/8) is viable on consumer CPU hardware with 4 GB to 8 GB RAM.
* **Experimental Setup:**
  * Deploy Khwarizmi Small (700M active / 2.1B total MoE parameters) on an x86_64 AVX2 consumer CPU machine constrained to 4 GB available RAM.
  * Measure token generation throughput (tokens/sec) and L3 cache miss rate.
* **Measured Variables:** Tokens/sec, RAM footprint (GB), OS swap paging frequency, benchmark accuracy on HumanEval-Offline.
* **Hypothesis Under Test:** MoE achieves higher HumanEval accuracy than a 700M dense baseline, but if memory bandwidth bottlenecks cause tokens/sec to drop below $15\text{ tok/sec}$, MoE must be pruned.
* **Pass/Fail Criteria:** Pass if tokens/sec $\ge 15$ AND HumanEval accuracy improves by $\ge 8\%$. Otherwise, execute Pruning Gate.

### 3.5 Experiment EX-05: Adaptive Recurrent Reasoning Cycles (ARRC) Halting
* **Objective:** Validate that learned halting gates successfully terminate recurrence early on easy tasks and extend cycles on difficult math/logic tasks.
* **Experimental Setup:**
  * Test model on 500 Easy Arithmetic queries (e.g., $12 + 15$) and 500 Hard Word Math queries (GSM8K equivalents).
  * Record number of recurrent cycles $K$ executed per query before cumulative halting $\sum p_k \ge 0.95$.
* **Measured Variables:** Average cycles $K_{\text{easy}}$ and $K_{\text{hard}}$, accuracy vs fixed $K=1$, ponder loss penalty convergence.
* **Hypothesis Under Test:** $K_{\text{easy}} \le 1.1$ and $K_{\text{hard}} \ge 2.6$, achieving $\ge 15\%$ higher accuracy on hard queries than single-pass inference.
* **Pass/Fail Criteria:** Pass if $K_{\text{hard}} / K_{\text{easy}} \ge 2.0$ and hard task accuracy improves by $\ge 15\%$.

---

## 4. Reproducibility & Statistical Reporting Guidelines

All ablation and experimental reports submitted to the Khwarizmi repository must adhere to the following strict reporting standards:
1. **Hardware Specification:** State CPU model, available RAM, GPU model (if used), operating system, and precision format (FP32, FP16, INT8, INT4).
2. **Statistical Significance:** All comparative quality gains must report mean, standard deviation across 3 random seeds, and two-tailed Student's t-test p-value ($p < 0.01$ required for adoption).
3. **Offline Verification:** All experiments must be executed with `--offline-mode=True`, ensuring zero network requests.
