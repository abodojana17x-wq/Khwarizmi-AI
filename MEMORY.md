# Khwarizmi AI: Dual Memory System Architecture Specification
**Document Version:** 2.0 (Phase 0 Architecture Reset)  
**Date:** 2026-08-11  
**Status:** Implementation-Ready Technical Blueprint  

---

## 1. Architectural Need: Why Dual Memory?

Traditional sequence models rely either on unconstrained context stuffing (e.g., passing 1,000,000 tokens of dialogue history into attention blocks) or on simple RAG retrieval pipelines that cannot learn when to update or forget obsolete facts.

In **Khwarizmi AI**, memory is treated as an explicit, learned, two-tier cognitive system:
* **Short-Term Working State ($\mathcal{M}_{\text{short}}$):** Handles immediate syntactic parsing, local reasoning steps, and active dialogue turns.
* **Long-Term Persistent Memory ($\mathcal{M}_{\text{long}}$):** Stores critical project decisions, architectural constraints, debugging discoveries, and DAG task milestones across multi-month software engineering projects.

```
+---------------------------------------------------------------------------------------------------------+
|                                    KHWARIZMI DUAL MEMORY ARCHITECTURE                                   |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|   +-------------------------------------------------------------------------------------------------+   |
|   |                              SHORT-TERM WORKING STATE (M_short)                                 |   |
|   |  - Recurrent KSC Matrix State : S_t in R^(d_k x d_n)  (O(1) continuous state)                  |   |
|   |  - Rolling Context Buffer     : Normalized window of recent L_window = 512 tokens               |   |
|   |  - Active Reasoning Buffer    : Latent intermediate vectors z_t^(k) during ARRC cycles          |   |
|   +-------------------------------------------------------------------------------------------------+   |
|                                                    |                                                    |
|                                 (Learned Memory Gating Controller)                                      |
|                             [ READ | WRITE (U > 0.8) | UPDATE | FORGET ]                                |
|                                                    |                                                    |
|                                                    v                                                    |
|   +-------------------------------------------------------------------------------------------------+   |
|   |                             LONG-TERM PERSISTENT MEMORY (M_long)                                |   |
|   |                                                                                                 |   |
|   |   +---------------------------------------+       +-----------------------------------------+   |
|   |   |        ASSOCIATIVE KV TABLE           |       |        SYMBOLIC DAG PROJECT STORE       |   |
|   |   |   Slots : { (k_i, v_i, u_i) }         |       |   Structured Nodes from rafig/reasoning |   |
|   |   |   Capacity : 128 - 512 slots          |       |   (Goals, Tasks, Subtasks, Dependencies)|   |
|   |   +---------------------------------------+       +-----------------------------------------+   |
|   +-------------------------------------------------------------------------------------------------+   |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Short-Term Working State ($\mathcal{M}_{\text{short}}$)

The working state provides instantaneous context without requiring external disk or database lookups:
1. **Recurrent KSC State Matrix ($S_t$):**
   Each Khwarizmi State Cell head maintains an internal $d_k \times d_n$ FP32/FP16 matrix that accumulates syntactic and semantic information over the active sequence via the stabilized recurrence:
   $$S_t = \text{diag}(\bar{A}_t) S_{t-1} + (1 - \bar{A}_t) \odot \left( k_t \otimes v_t^T \right)$$
2. **Rolling Token Buffer:**
   A fixed sliding window of the last $L_{\text{window}} = 512$ tokens is retained in RAM to provide exact syntactic reference for code generation and token-level editing.
3. **Latent Reasoning Buffer:**
   During Adaptive Recurrent Reasoning Cycles (ARRC), intermediate latent vectors $z_t^{(1)}, \dots, z_t^{(K)}$ are stored in temporary scratchpad registers until the halting condition $\sum p_k \ge 1 - \epsilon$ is satisfied.

---

## 3. Long-Term Persistent Memory ($\mathcal{M}_{\text{long}}$)

For tasks spanning days, weeks, or months (e.g., maintaining a large Python repository), Khwarizmi AI uses a **non-parametric, persistent key-value store** coupled with a **symbolic project graph store**.

### 3.1 Associative Key-Value Table
The KV table consists of a fixed capacity $M \in \{128, 256, 512\}$ of memory tuples:
$$\mathcal{M}_{\text{KV}} = \left\{ (k_i, v_i, u_i, \tau_i) \right\}_{i=1}^M$$
* $k_i \in \mathbb{R}^{D_m}$: L2-normalized key embedding vector representing the semantic question or context of the stored fact.
* $v_i \in \mathbb{R}^{D_m}$: Value embedding vector encoding the structured fact or architectural constraint.
* $u_i \in [0, 1]$: Dynamic scalar **Utility Score** measuring the importance and historical usefulness of the memory item.
* $\tau_i \in \mathbb{R}^+$: Last-access timestamp used for exponential utility decay.

### 3.2 Symbolic DAG Project Store (Integrated with Legacy Tool)
When the Cognitive Router selects `PROJECT PLAN PATH`, long-term memory also interacts with the structured DAG project store managed by `rafig/reasoning`:
* **Node Schema:** Explicit `Goal`, `Task`, and `Subtask` dataclasses containing descriptions, hard constraints, assumption lists, and prerequisite dependency IDs.
* **Graph Coherence:** When a task is marked `completed` or `failed`, the persistent memory graph updates the dependency frontier, allowing Khwarizmi AI to resume a complex software project after offline restarts without re-reading the entire project history.

---

## 4. Learned Memory Control Policy (Gating Equations)

A learned, lightweight Multi-Layer Perceptron (the **Memory Gating Controller**) evaluates the current latent state $h_t$ and task context $s_{\text{task}}$ to emit four gating probabilities:
$$[g_{\text{read}}, g_{\text{write}}, g_{\text{update}}, g_{\text{forget}}] = \sigma\left( W_{\text{mem}} [h_t; s_{\text{task}}] + b_{\text{mem}} \right)$$

### 4.1 READ Operation (Associative Retrieval)
If $g_{\text{read}} > \theta_{\text{read}}$, the system generates a query vector $q_t = W_q h_t$ and retrieves the top-1 or top-k matching memory slots using temperature-scaled cosine similarity:
$$\alpha_i = \frac{\exp\left( \frac{q_t^T k_i}{\tau_{\text{ret}}} \right)}{\sum_{j=1}^M \exp\left( \frac{q_t^T k_j}{\tau_{\text{ret}}} \right)}, \quad v_{\text{retrieved}} = \sum_{i=1}^M \alpha_i v_i$$
The retrieved vector $v_{\text{retrieved}}$ is injected into the subsequent KSC residual block.

### 4.2 WRITE Operation & Utility Gating
To prevent memory pollution from casual banter ("hello", "thanks"), a write occurs **only if both conditions are met:**
1. Write gate active: $g_{\text{write}} > \theta_{\text{write}}$.
2. Predicted utility score: $U(h_t) > U_{\text{threshold}} = 0.80$.

When triggered, a new tuple $(k_{\text{new}}, v_{\text{new}}, U(h_t), t)$ is created.

### 4.3 UPDATE Operation (Contradiction Resolution)
If $g_{\text{update}} > \theta_{\text{update}}$ and the retrieved top-1 similarity $\max_i(q_t^T k_i) > 0.88$, the system merges the new information into slot $i^*$, preventing duplicate entries and updating the timestamp $\tau_{i^*} \leftarrow t$.

### 4.4 FORGET Operation & Time-Decay Eviction
If the memory table is full ($|\mathcal{M}_{\text{KV}}| = M$) when a new write occurs, or if explicit forgetting is triggered ($g_{\text{forget}} > \theta_{\text{forget}}$), the system purges slot $j^*$ with the lowest effective utility:
$$j^* = \arg\min_{j \in \{1, \dots, M\}} \left( u_j \cdot e^{-\lambda (t - \tau_j)} \right)$$
where $\lambda = 10^{-4}$ is the historical decay constant.

---

## 5. Experimental Verification Protocol for Dual Memory

To certify that Dual Memory provides measurable improvement without inflating RAM:
1. **Long-Context Needle-In-A-Haystack (NIAH-32K):**
   * Must achieve $\ge 95\%$ exact-match retrieval of project decision keys embedded across 32,000 tokens.
2. **Selective Write Quality Test:**
   * Must achieve $\ge 90\%$ precision in writing Architectural Decision Records (ADRs) while rejecting 100% of casual filler sentences.
3. **Forgetting Verification Test:**
   * Must achieve 100% successful purging of obsolete port numbers or deprecated API constraints when an explicit contradicting instruction is provided.
