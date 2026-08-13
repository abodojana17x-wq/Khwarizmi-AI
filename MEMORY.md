# Khwarizmi AI: Dual Memory System Architecture Specification
**Document Version:** 2.1 (Phase 3 Implementation Complete)  
**Date:** 2026-08-13  
**Status:** Phase 3 Implemented — §6 documents the delivered API and known limitations  

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

---

## 6. Phase 3 Implementation Status (Documentation of Delivered Functionality)

This section documents **only the functionality that actually exists** as of Phase 3
(`v0.3.0`). It is the authoritative reference for the implemented API.

### 6.1 Implemented Modules & Interfaces

| Module | Class | Role |
| :--- | :--- | :--- |
| `khwarizmi/memory/short_term.py` | `ShortTermWorkingState` | Bounded short-term working state: KSC recurrent state + rolling token window capped at `short_term_capacity`. Operations `read` / `write` / `forget`, plus `get_summary_vector` and backward-compatible `update`. |
| `khwarizmi/memory/long_term.py` | `LongTermPersistentMemory` | Fixed-capacity (`memory_slots`) associative key-value store with real `read` / `write` / `update` / `forget` operations. |
| `khwarizmi/memory/gating.py` | `MemoryGatingController` | Learned READ/WRITE/UPDATE/FORGET probability heads. |
| `khwarizmi/memory/gating.py` | `UtilityGatingPolicy` | Deterministic, parameter-free decision policy → `RETAIN` / `WRITE` / `UPDATE` / `FORGET`. |
| `khwarizmi/memory/dual_memory.py` | `DualMemory` | Facade composing the above into one bounded lifecycle (`init_state`, `read`, `forward`). |
| `khwarizmi/core/memory_prototype.py` | `KhwarizmiDualMemoryPrototype` | Compositional integration of `KhwarizmiKSCPrototype` + `DualMemory`. |

### 6.2 Operations (explicit, testable, real state transitions)

* **READ** — `LongTermPersistentMemory.read(query, table, g_read, step)` and
  `DualMemory.read(...)`: scaled dot-product attention over valid slots, gated by
  `g_read`; an empty table returns exact zeros (no `out_proj` bias leak).
* **WRITE** — `LongTermPersistentMemory.write(candidate, table, g_write, step, threshold[, similarity_threshold])`:
  inserts a candidate when the write gate exceeds `threshold`; evicts the slot with
  the minimum time-decayed utility when full; with `similarity_threshold` set,
  near-duplicate candidates are merged instead of inserted.
* **UPDATE** — `LongTermPersistentMemory.update(candidate, table, g_update, step, threshold[, similarity_threshold])`:
  merges a candidate into the most-similar existing slot (equal-weight EMA value
  blend, `utility = max(old, new)`, timestamp refresh) when the gate is active and
  top-1 cosine similarity ≥ threshold. Never adds slots (no duplication).
* **FORGET** — `LongTermPersistentMemory.forget(table, g_forget, threshold[, slot_index])`:
  gate-driven eviction of the lowest-utility valid slot, or explicit eviction of a
  given slot id (out-of-range / already-empty ids raise `ValueError`).

### 6.3 Utility Gating (deterministic decision policy)

`UtilityGatingPolicy.decide(gates, utilities, max_similarity)` resolves each batch
item to exactly one action, with strict priority:

1. `FORGET` — `g_forget ≥ forget_threshold`.
2. `UPDATE` — `max_similarity ≥ update_similarity_threshold` **and** `g_update ≥ update_threshold`.
3. `WRITE` — `utility ≥ utility_threshold` **and** `g_write ≥ write_threshold`.
4. `RETAIN` — otherwise (information stays in short-term memory only).

### 6.4 Configuration (`KhwarizmiConfig`)

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `short_term_capacity` | `512` | Bounded rolling-window size (short-term). |
| `memory_slots` | `32` | Persistent-table slot capacity. |
| `memory_dim` | `64` | Key/value embedding dimension. |
| `utility_threshold` | `0.8` | Minimum utility for a WRITE/promotion decision. |
| `read_threshold` / `write_threshold` / `update_threshold` / `forget_threshold` | `0.5` / `0.5` / `0.5` / `0.7` | Per-operation gate activation thresholds. |
| `update_similarity_threshold` | `0.88` | Cosine-similarity threshold for UPDATE/merge. |
| `utility_decay_lambda` | `0.01` | Exponential time-decay constant for eviction. |

All fields are validated at construction; the `TinyTest` tier pins
`short_term_capacity = 128`.

### 6.5 Capacity / Boundedness Guarantees

* Both stores are **pre-allocated tensors**: short-term `(batch, capacity, d_model)`,
  persistent `(batch, memory_slots, memory_dim)` (+ scalar/bool tables). No Python
  list/dict grows with sequence length or operation count.
* Tests cover empty memory, full memory, repeated writes/updates/forgets,
  duplicate/near-duplicate entries, invalid memory ids, invalid states, capacity
  limits, and long-running sequences; `benchmarks/phase3_dual_memory.py` exercises
  10,000 lifecycle cycles and confirms boundedness.

### 6.6 Known Limitations (documented, deferred to later phases)

* **Learned gate policy**: the gate *network* ships with conservative
  (negatively-biased) initialization and is **not yet trained**; the decision
  *policy* is deterministic but gate probabilities only become meaningful after
  Phase 8–10 gradient training. Consequently the roadmap's NIAH-32K (≥95%) and
  selective-write (≥90%) success criteria are not yet measurable — they require
  the Phase 9/10 dataset + training pipeline.
* **Symbolic DAG project store**: the associative KV tier is implemented; the
  symbolic DAG integration with `rafig/reasoning` remains a Phase 13 tool-layer
  concern.
* **Latent ARRC reasoning buffer**: intermediate ARRC latent vectors are a Phase 5
  concern and are not part of the Phase 3 short-term store.
