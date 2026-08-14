# Khwarizmi AI: System Architecture Specification
**Document Version:** 2.0 (Phase 0 Architecture Reset & Complete Redesign)  
**Date:** 2026-08-11  
**Status:** Implementation-Ready Blueprint  

---

## 1. Executive Architecture Summary

**Khwarizmi AI** is a lightweight, highly intelligent, reasoning-focused and project-intelligence specialized artificial intelligence system designed from the ground up for **100% offline, private, low-resource operation**. Unlike traditional large language models that rely on massive dense transformers, quadratic-cost attention mechanisms, multi-gigabyte KV caches, and cloud inference APIs, Khwarizmi AI introduces a distinct, clean, and modular architecture engineered around **maximum intelligence and reasoning capability per unit of compute and memory**.

### Key Design Tenets
1. **Zero Cloud / Zero API Dependency:** All inference, memory retrieval, symbolic verification, and task planning execute locally on consumer CPUs, modest RAM (<4 GB), consumer GPUs, or edge devices.
2. **Selective Computation (Cognitive Router):** Compute is treated as a finite resource. A learned cognitive router dynamically evaluates request complexity and dispatches tokens to specialized computational pathways (`FAST`, `CODING`, `REASONING`, `PROJECT_PLAN`, `VERIFICATION`). Simple requests consume minimal compute; complex software engineering and long-horizon project tasks unlock iterative recurrent reasoning and deterministic tool verification.
3. **Sub-Quadratic Sequence Modeling (Khwarizmi State Cell - KSC):** Replacing default multi-head attention with a selective recurrent state cell that achieves linear-time decoding $O(1)$ per token memory footprint and efficient associative scan prefill $O(L)$, while maintaining strong associative recall and state stability over long sequences.
4. **Dual Memory Architecture:** Separates ephemeral reasoning context (**Short-Term Working State**) from structured, utility-gated persistent project and factual knowledge (**Long-Term Persistent Memory**) equipped with learned `READ`, `WRITE`, `UPDATE`, and `FORGET` gates.
5. **Layered Agentic Tool Separation:** Neural inference is cleanly decoupled from deterministic symbolic execution. Existing Python AST analyzers and DAG project planners are isolated in an external tool layer, activated only when the cognitive router deems verification or symbolic graph manipulation necessary.

```
                        +--------------------------------------------------+
                        |                    USER INPUT                    |
                        +--------------------------------------------------+
                                                 |
                                                 v
                        +--------------------------------------------------+
                        |            KHWARIZMI OFFLINE AGENT               |
                        |      (Input Sanitization, Language Detect)       |
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
            |                    |                                    |                    |
            +--------------------+---------------+--------------------+--------------------+
                                                 |
                                                 v
                        +--------------------------------------------------+
                        |              KHWARIZMI NEURAL CORE               |
                        |      (KSC Layers + Sparse MoE + Dual Memory)     |
                        +--------------------------------------------------+
                                                 |
                                                 +<========================+
                                                 | (Selective Tool Call)   |
                                                 v                         |
                        +--------------------------------------------------+
                        |               OPTIONAL LOCAL TOOLS               |
                        |  ├── Project Planner (DAG Symbolic Engine)       |
                        |  ├── Python Analysis (AST Python Brain)          |
                        |  ├── Symbolic Verification & Consistency         |
                        |  └── Deterministic File & Environment Tools      |
                        +--------------------------------------------------+
                                                 |
                                                 v
                        +--------------------------------------------------+
                        |                  FINAL RESPONSE                  |
                        +--------------------------------------------------+
```

---

## 2. Layered Architecture: Decoupling Neural Core from Deterministic Tools

A critical flaw in hybrid symbolic-neural systems is forcing every token or user request through slow, rigid symbolic parsers or rule engines. Khwarizmi enforces a strict **Layered Separation of Concerns**:

1. **User Layer:** Accepts raw natural language (English, Arabic, Egyptian Arabic, Franco-Arabic) or source code.
2. **Khwarizmi Offline Agent Layer:** Performs lightweight input sanitization, detects language mix, and formats prompts into structured internal frames.
3. **Cognitive Router Layer:** Evaluates the request frame using a lightweight gating network ($<5\text{M}$ parameters) to assign a discrete execution policy.
4. **Khwarizmi Neural Core Layer:** Executes sequence modeling via KSC blocks, Sparse Mixture-of-Experts (if activated), and Dual Memory reads/writes.
5. **Optional Local Tools Layer:** Deterministic, rule-based engines (including the legacy `rafig/reasoning` DAG planner and `rafig/python_brain` AST analyzer). These tools are **never called automatically on every request**. They are invoked only via explicit neural tool-calling tokens or router-triggered verification steps.

---

## 3. Existing Repository Audit & Old Component Decisions

An exhaustive audit of the existing `Khwarizmi-AI` (`rafig/`) repository was conducted on 2026-08-11. Under the critical architectural principle that **quality > previous work** and **architecture quality > feature count**, each existing component has been evaluated against the new Khwarizmi architecture.

| Component | Current Purpose | Decision | Justification | Future Location |
| :--- | :--- | :--- | :--- | :--- |
| `rafig/language/tokenizer.py` | Minimal character-aware JSON tokenizer with small vocabulary. | **REPLACE** | Lacks multi-lingual subword compression (BPE/Unigram) needed for efficient Arabic/English/Code tokenization and GGUF compatibility. | `khwarizmi/data/tokenizer/` (custom offline byte-fallback BPE). |
| `rafig/language/language_understanding.py` | Rule-based regex heuristics for detecting language and basic intents. | **REPLACE** | Regex heuristics cannot generalize to multi-turn semantic reasoning; neural core and router naturally absorb language detection. | Replaced by neural core; lightweight pre-sanitizer in `khwarizmi/agent/input_filter.py`. |
| `rafig/language/semantic_representation.py` | Structured dataclass representation of intents, actions, goals, constraints. | **REFACTOR** | Highly valuable as an internal structured schema for project tools and router data contracts, but must be decoupled from mandatory regex pipelines. | `khwarizmi/tools/schemas/request_schema.py`. |
| `rafig/reasoning/` (`decomposition`, `planner`, `evaluation`, `inference`, `engine`, `models`) | Phase 06 symbolic DAG project planner and causal inference engine. | **MOVE TO EXTERNAL TOOL LAYER** | Outstanding deterministic project planner and constraint verifier, but must not execute on every neural inference step. Survives as the **Project Planner Tool** callable by the router. | `khwarizmi/tools/project_planner/`. |
| `rafig/python_brain/` (`analyzer`, `model`, `complexity`, `explain`, `issues`, `parser`, `types`) | Phase 07 pure-AST Python static analysis and issue detection engine. | **MOVE TO EXTERNAL TOOL LAYER** | Exceptional offline, zero-dependency Python verifier. Serves as the **Python Analysis Tool** for coding tasks and code verification without adding latency to non-code queries. | `khwarizmi/tools/python_brain/`. |
| `rafig/config.py`, `rafig/app.py`, `rafig/rafig.py`, `rafig/paths.py` | Existing CLI entry point, directory setup, and configuration loader. | **REFACTOR** | Adapt from calling symbolic engines directly to orchestrating the layered offline agent, cognitive router, and neural core runtime. | `khwarizmi/agent/`, `khwarizmi/config/`, `khwarizmi/runtime/`. |

---

## 4. Full Architecture Specification

### 4.1 Khwarizmi State Cell (KSC)
The **Khwarizmi State Cell (KSC)** is the primary sequence-modeling building block. Unlike Mamba or linear attention which use fixed scalar decay or unstructured state recurrence, KSC introduces an **input-selective, eigenvalue-bounded recurrent state matrix** with structured gating.

* **State Representation:** Each KSC head maintains a latent recurrent state matrix $S_t \in \mathbb{R}^{d_k \times d_n}$, where $d_k$ is the state head dimension and $d_n$ is the expansion memory bank size.
* **Selective State Update:** At token step $t$, input vector $x_t \in \mathbb{R}^D$ is projected into query, key, value, and gating parameters.
* **Stability:** To prevent exploding or vanishing states over sequences of $100{,}000+$ tokens, KSC applies a hard eigenvalue bounding diagonal matrix $\Gamma_t = \text{diag}(\gamma_{t,1}, \dots, \gamma_{t,d_k})$ where $\gamma_{t,i} \in (\gamma_{\min}, 1 - \epsilon)$.
* **Complexity:** Decoding requires $O(1)$ memory per sequence and $O(D \cdot d_n)$ compute per token. Prefill/training is parallelized via associative prefix scans in $O(L \log L)$ time.

### 4.2 Dual Memory System (Short-Term Working + Long-Term Persistent)
Khwarizmi AI implements an explicit two-tier cognitive memory architecture:

1. **Short-Term Working State ($\mathcal{M}_{\text{short}}$):**
   * Stored directly inside the active KSC recurrent state $S_t$ plus a small rolling window buffer of recent normalized tokens ($L_{\text{window}} = 512$).
   * Handles conversational turns, active syntactic parsing, and immediate step-by-step reasoning variables.
2. **Long-Term Persistent Memory ($\mathcal{M}_{\text{long}}$):**
   * A non-parametric, local key-value project store combined with structured DAG knowledge nodes.
   * Operated by a learned **Memory Gating Controller** that outputs discrete/continuous probabilities for four operations:
     * `READ`: Retrieve relevant historical decisions, constraints, or previous bug fixes using associative cosine similarity.
     * `WRITE`: Insert a high-value fact, architectural constraint, or project decision with a learned utility score $U > U_{\text{threshold}}$.
     * `UPDATE`: Merge or refine existing knowledge nodes when project specifications evolve.
     * `FORGET`: Evict low-utility, obsolete, or contradictory entries when memory budgets are reached.

### 4.3 Cognitive Router
The **Cognitive Router** is a lightweight gating classifier ($\pi_\theta(p | x_t, \mathcal{M}_{\text{short}})$) positioned before neural sequence blocks. It evaluates the current prompt and working state to select a computational pathway $p \in \mathcal{P}$:
* **FAST PATH:** Simple informational or conversational queries → Single pass through base KSC layers; MoE, memory retrieval, and tools bypassed.
* **CODING PATH:** Code generation, debugging, or refactoring → Activates coding-specialized MoE experts and primes the AST `Python Brain` verification tool.
* **REASONING PATH:** Multi-step math, logic puzzles, or complex deduction → Activates **Adaptive Compute** recurrent loops ($K \ge 2$ cycles) before final decoding.
* **PROJECT PLAN PATH:** Software architecture, multi-week planning, dependency analysis → Invokes `READ`/`WRITE` on Long-Term Memory and permits calling the deterministic `Project Planner` DAG tool.
* **VERIFICATION PATH:** Self-correction and consistency checking → Triggers formal constraint validation and code AST inspection before emitting output.

### 4.4 Sparse Mixture-of-Experts (MoE)
To scale knowledge and domain capability without inflating offline inference RAM or compute latency, Khwarizmi uses **Sparse Mixture-of-Experts (MoE)** layers interspersed every $N=4$ KSC blocks:
* **Architecture:** $E=8$ total experts per MoE layer, with Top-$K=2$ selective activation ($25\%$ active parameter footprint).
* **Expert Specialization:** Initial candidate specializations include: *(1) Multilingual Language/Arabic, (2) Egyptian Arabic/Dialect, (3) Python/Coding, (4) Software Engineering/Architecture, (5) Mathematical/Symbolic Reasoning, (6) Project Planning/DAGs, (7) Tool Use/Verification, (8) General Fact Recall.*
* **Validation Mandate:** The router learnability and expert distinctiveness are experimentally measured. If MoE routing overhead or RAM fragmentation on edge devices outweighs the quality gain, **MoE will be pruned or replaced by a denser, smaller layer during ablation.**

#### Phase 4 Implementation Status (Sparse MoE)

Implemented in `khwarizmi/experts/moe_layer.py` and `khwarizmi/experts/specialists.py`, tested in
`tests/test_moe.py` (57 tests), benchmarked by `benchmarks/phase4_sparse_moe.py`:

* **Experts:** `ExpertLayer` — independently parameterized Swish FFNs `d_model → d_ff → d_model`
  (per-expert width configurable via `config.expert_d_ff`). All experts share input/output
  dimensions so the router can combine them. `create_standard_specialists` instantiates the 8
  named specializations (names are metadata only).
* **Router:** two parameterized projections `W_gate` and `W_noise` (both `d_model → E`). Noisy
  gating `H(z) = z W_g + ε·softplus(z W_noise)` is applied **only during training**; inference
  uses clean logits and is fully deterministic (Top-K ties resolved deterministically by
  `torch.topk` for a given input). `MoERoutingDecision` exposes the full routing state.
* **Top-K selection & weights:** `G(z) = Softmax(TopK(H(z), K))` — routing weights are positive,
  normalized (sum to 1), and differentiable through the selected logits. `1 ≤ K ≤ E` enforced.
* **Sparse execution:** `forward` gathers tokens per *selected* expert and calls only experts
  that received tokens — unselected experts perform zero computation (no dense masking).
  Executed expert ids are recorded in `SparseMoELayer.last_routed_experts`.
* **Load-balancing loss:** `L_balance = α_moe · E · Σ_i f_i·P_i` (Section 5.4), returned by
  `forward`/`route` and independently testable via `compute_load_balance_loss`; differentiable
  w.r.t. the router (gradient reaches gating rows of *all* experts, including unselected ones).
  It detects collapse (16× higher at full collapse) and — as a training regularizer — prevents
  it (benchmark §7); it cannot repair an already fully saturated router (gradient vanishes).
* **Integration point:** `KhwarizmiModel` wires one shared `SparseMoELayer` into every
  `moe_frequency`-th KSC residual block (cognitive-router pathway flags may bypass it at
  runtime). `config.enable_moe=False` disables MoE entirely: all blocks become dense FFN
  blocks, no experts/router are built, and the model behaves exactly as the pre-Phase-4 dense
  architecture.
* **Configuration:** `num_experts`, `top_k_experts`, `moe_frequency`, `enable_moe`,
  `moe_noise_enabled`, `expert_d_ff`, `load_balance_alpha` (all validated: e.g. `num_experts ≥ 1`,
  `1 ≤ top_k ≤ num_experts`, `α ≥ 0` finite).
* **Known limitations:** the roadmap's perplexity-vs-dense-baseline and CPU-latency-overhead
  gates require Phase 9/10 data + training; CPU per-expert gather/scatter makes the sparse
  layer slower than a single equal-active dense FFN at small scale (see `BENCHMARKS.md` §7).

### 4.5 Adaptive Compute & Recurrent Reasoning
Instead of spending identical compute on simple and difficult tokens, Khwarizmi introduces **Adaptive Recurrent Reasoning Cycles (ARRC)**:
* **Recurrent Depth:** Certain residual blocks can be executed iteratively $k$ times ($k \in [1, K_{\max}]$) on the same token or reasoning intermediate representation.
* **Learned Halting:** A halting gate $h_t^{(k)} = \sigma(w_h^T z_t^{(k)} + b_h)$ monitors the internal confidence and convergence of the recurrent state. Once cumulative halting probability $\sum_{j=1}^k h_t^{(j)} \ge 1 - \epsilon$, recurrence halts.
* **Internal Latent Reasoning:** Reasoning occurs in latent state space rather than dumping verbose, unverified ASCII chain-of-thought tokens, saving inference latency and token context.

**Phase 5 implementation status (implemented & verified):**
* **Module:** `khwarizmi/reasoning/adaptive_compute.py` — `AdaptiveComputeBlock` (per-token ACT-style ARRC halting engine) and `PonderCostLoss` (standalone ponder cost module); `khwarizmi/reasoning/latent_reasoner.py` wraps the block behind the Cognitive Router pathway mask.
* **Halting granularity:** *per token*. Each token accumulates its own halting probability $p_k = \sigma(w_h^T z^{(k)} + b_h)$ and halts independently at the first cycle $k \ge K_{\min}$ where $\sum_{j \le k} p_j \ge 1 - \epsilon$. Halted tokens are frozen (latent stops updating, zero output weight on later cycles) while the rest of the batch continues, so different tokens genuinely receive different compute depth.
* **ACT remainder formulation:** the output is the exact §5.5 mixture $z_{\text{out}} = \sum_{k<K} p_k z^{(k)} + R\,z^{(K)}$ with remainder $R = 1 - \sum_{j<K} p_j$; per-token output weights always sum to exactly 1 and the accumulated halting probability is capped at 1 (including the forced final halt).
* **Termination guarantees:** no token halts before `min_recurrent_cycles` ($K_{\min}$, halting mass is masked to zero on earlier cycles) and every token is force-halted with its remainder at `max_recurrent_cycles` ($K_{\max}$) — an infinite recurrent loop is structurally impossible. `force_cycles` provides a deterministic fixed-compute mode (all tokens execute exactly $K$ cycles).
* **Ponder cost:** $\mathcal{L}_{\text{ponder}} = \beta_{\text{ponder}} \cdot \mathbb{E}[N + R]$ where the discrete cycle count $N$ is detached and the remainder $R$ carries the differentiable halting-gate gradient; exposed through `KhwarizmiOutput.losses["ponder_loss"]` and summed into `total_aux_loss`.
* **Recurrent core:** the reasoning transformation reuses the `KhwarizmiStateCell` (KSC) recurrence with LayerNorm pre-activation and residual accumulation; the KSC state propagates across cycles and freezes once every token of a sequence has halted, so no stale state drifts between examples.
* **Configuration:** `enable_adaptive_compute` (master switch — when `False` no halting gates/reasoning cell are built, the model runs a single fixed pass, and ponder loss is exactly 0, matching the pre-Phase-5 path), `min_recurrent_cycles` ($\ge 1$), `max_recurrent_cycles` ($\ge$ min), `halting_epsilon` ($\in (0,1)$), `ponder_cost_beta` (finite, $\ge 0$). All validated at config construction.
* **Determinism:** the forward pass contains no sampling — identical inputs, parameters, and state produce identical outputs, step counts, and halting distributions.
* **Known limitations:** the batch-level early exit skips remaining cycles only after *every* token in the batch has halted; per-token FLOP savings ($K_{\text{avg}}/K_{\max}$) become wall-clock savings only with per-token kernels (Phase 12 runtime scope). The roadmap's trained easy/hard $K_{\text{avg}}$ and accuracy gates require Phase 9/10 data + training (see `BENCHMARKS.md` §8).

### 4.6 Project Intelligence (Specialization Core)
Khwarizmi AI is uniquely specialized for **Large Project Management & Software Technical Leadership**:
* **Long-Horizon Coherence:** Tracks software projects over extended lifecycles by binding natural language discussions to formal DAG task structures.
* **Structured Dependency Reasoning:** Integrates with the deterministic `Project Planner` tool to validate that Subtask $B$ cannot begin until Dependency $A$ is marked complete.
* **Failure Recovery & Replanning:** When an action or code test fails, Khwarizmi queries Long-Term Memory for previous failure reports, invokes causal revision rules, and updates the task dependency graph without restarting from scratch.

### 4.7 Selective Verification
Verification is expensive and must be applied selectively:
* **Trigger Conditions:** Verification is activated only when:
  1. The cognitive router selects `CODING PATH` or `PROJECT PLAN PATH`.
  2. The output confidence score $C(y) < \theta_{\text{verif}}$.
  3. An explicit structural assertion or code generation block is produced.
* **Verification Mechanisms:**
  * **AST Syntax & Type Checking:** Invoking `rafig/python_brain` to verify syntactically valid Python, detect undefined variables, check scope chains, and calculate cyclomatic complexity.
  * **Symbolic Constraint Checking:** Invoking `rafig/reasoning` to check for dependency cycles or assumption violations in project plans.
  * **Self-Consistency Scoring:** Evaluating dual-state forward-backward consistency on mathematical answers.

---

## 5. Mathematical Design Proposal

### 5.1 Khwarizmi State Cell (KSC) Formulation
Let input at sequence step $t$ be $x_t \in \mathbb{R}^D$. For each head $m \in \{1, \dots, H\}$ with head dimension $d_k = D / H$ and memory expansion $d_n$:

1. **Input Projections:**
   $$q_t = W_q x_t, \quad k_t = W_k x_t, \quad v_t = W_v x_t, \quad \Delta_t = \text{softplus}(W_\delta x_t)$$
   where $q_t, k_t \in \mathbb{R}^{d_k}$, $v_t \in \mathbb{R}^{d_n}$, and scalar/vector step size $\Delta_t \in \mathbb{R}^{d_k}$.

2. **State Transition Matrix & Bounding:**
   Let base continuous dynamics matrix be $A \in \mathbb{R}^{d_k \times d_k}$ initialized to structured Hurwitz orthogonal form. The discretized state retention gate is:
   $$\bar{A}_t = \exp\left( - \text{softplus}(\Delta_t \odot a) \right) \in (0, 1)^{d_k}$$
   where $a \in \mathbb{R}^{d_k}$ is a learned positive decay parameter vector, ensuring $\bar{A}_{t,i} \in (\gamma_{\min}, 1 - \epsilon)$ with $\gamma_{\min} = 0.85$.

3. **Recurrent State Update:**
   The recurrent state $S_t \in \mathbb{R}^{d_k \times d_n}$ evolves as:
   $$S_t = \text{diag}(\bar{A}_t) S_{t-1} + (1 - \bar{A}_t) \odot \left( k_t \otimes v_t^T \right)$$
   where $\otimes$ denotes the outer product.

4. **Output Projection:**
   $$y_t = W_o \left( \text{LayerNorm}\left( q_t^T S_t \odot \sigma(W_g x_t) \right) \right)$$
   where $\sigma(W_g x_t)$ is a Swish-gated output linear projection.

### 5.2 Dual Memory Gating & Retention Dynamics
Let persistent memory store be a fixed-capacity table of slots $\mathcal{M} = \{ (k_i, v_i, u_i) \}_{i=1}^M$, where $k_i \in \mathbb{R}^{D_m}$ is key, $v_i \in \mathbb{R}^{D_m}$ is value, and $u_i \in \mathbb{R}$ is scalar utility.

1. **Memory Gating Controller:**
   Given current latent state $h_t$ and task context $s_{\text{task}}$:
   $$g_{\text{write}}(t) = \sigma(w_w^T [h_t; s_{\text{task}}]), \quad g_{\text{read}}(t) = \sigma(w_r^T [h_t; s_{\text{task}}])$$

2. **Associative Retrieval (READ):**
   If $g_{\text{read}}(t) > \tau_{\text{read}}$, compute attention weights over slots:
   $$\alpha_i(t) = \frac{\exp\left( \frac{\langle q_{\text{mem}}(h_t), k_i \rangle}{\sqrt{D_m}} \right)}{\sum_{j=1}^M \exp\left( \frac{\langle q_{\text{mem}}(h_t), k_j \rangle}{\sqrt{D_m}} \right)}$$
   $$\text{MemoryOut}_t = \sum_{i=1}^M \alpha_i(t) v_i$$

3. **Selective Write & Utility Eviction (WRITE/FORGET):**
   When $g_{\text{write}}(t) > \tau_{\text{write}}$, new candidate tuple $(k_{\text{new}}, v_{\text{new}}, u_{\text{new}})$ is generated.
   If table is full ($|\mathcal{M}| = M$), evict slot $j^*$ such that:
   $$j^* = \arg\min_{j \in \{1, \dots, M\}} \left( u_j \cdot e^{-\lambda(t - t_{\text{last\_access}, j})} \right)$$
   $$k_{j^*} \leftarrow k_{\text{new}}, \quad v_{j^*} \leftarrow v_{\text{new}}, \quad u_{j^*} \leftarrow u_{\text{new}}$$

### 5.3 Cognitive Router Policy & Optimization
Let computational pathways be $\mathcal{P} = \{p_1, p_2, \dots, p_C\}$.
1. **Policy Network:**
   $$\pi_\theta(p_c | x_t, \mathcal{M}_{\text{short}}) = \frac{\exp(\psi_c^T \text{Enc}(x_t, \mathcal{M}_{\text{short}}) / T)}{\sum_{j=1}^C \exp(\psi_j^T \text{Enc}(x_t, \mathcal{M}_{\text{short}}) / T)}$$

2. **Regularized Multi-Objective Loss:**
   To penalize unnecessary computational expense:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{cost}} \sum_{c=1}^C \pi_\theta(p_c | x) \cdot \text{FLOPs}(p_c) + \lambda_{\text{ent}} \mathcal{H}(\pi_\theta(\cdot | x))$$
   where $\text{FLOPs}(p_c)$ is the normalized computational cost of path $p_c$ and $\mathcal{H}$ prevents premature mode collapse.

### 5.4 Sparse Mixture-of-Experts (MoE) Gating & Balance Loss
For input token representation $z_t \in \mathbb{R}^D$ entering an MoE layer with $E$ experts $\{E_1, \dots, E_E\}$:
1. **Noisy Top-K Gating:**
   $$H(z_t)_i = z_t W_{g,i} + \epsilon_i \cdot \text{softplus}(z_t W_{\text{noise}, i}), \quad \epsilon_i \sim \mathcal{N}(0, 1)$$
   $$G(z_t) = \text{Softmax}\left( \text{TopK}(H(z_t), K) \right)$$
   $$\text{MoEOut}(z_t) = \sum_{i \in \text{TopK}} G(z_t)_i E_i(z_t)$$

2. **Load Balancing Auxiliary Loss:**
   Let $f_i$ be the fraction of tokens routed to expert $i$ in a batch, and $P_i$ be the mean gating probability allocated to expert $i$:
   $$\mathcal{L}_{\text{balance}} = \alpha_{\text{moe}} \cdot E \sum_{i=1}^E f_i \cdot P_i$$

### 5.5 Adaptive Compute & Recurrent Halting (ARRC)
For a recurrent reasoning block with latent state $z^{(k)}$ at cycle $k \in \{1, \dots, K_{\max}\}$:
1. **Halting Probability:**
   $$p_k = \sigma(w_h^T z^{(k)} + b_h), \quad \text{stopping step } K = \min \left\{ k' : \sum_{j=1}^{k'} p_j \ge 1 - \epsilon \right\}$$
2. **Effective Output & Ponder Cost:**
   Let remainder be $R = 1 - \sum_{j=1}^{K-1} p_j$. The output state is:
   $$z_{\text{out}} = \sum_{k=1}^{K-1} p_k z^{(k)} + R z^{(K)}$$
   $$\mathcal{L}_{\text{ponder}} = \beta_{\text{ponder}} \cdot \mathbb{E}\left[ K + R \right]$$

---

## 6. Proposed Clean Repository Structure

To support modular research, offline deployment, and strict separation between the neural core and deterministic tools, the repository will transition to the following clean structure:

```
Khwarizmi-AI/
├── ARCHITECTURE.md                 # Master Architecture Specification (This Document)
├── ROADMAP.md                      # Complete 16-Phase Execution Roadmap
├── RESEARCH.md                     # Deep Architecture & Literature Research Comparison
├── EXPERIMENTS.md                  # Comprehensive Experimental & Ablation Protocol
├── BENCHMARKS.md                   # Multi-Tier Evaluation Strategy & Threshold Gates
├── TRAINING.md                     # 12-Stage Low-Resource Training Strategy
├── MEMORY.md                       # Dual Memory System & Retention Specifications
├── DEPLOYMENT.md                   # Fully Offline Hardware & Edge Deployment Blueprint
├── CONTRIBUTING.md                 # Architecture-First Contribution Guidelines
├── README.md                       # Project Portal & Phase Tracking Index
├── requirements.txt                # Standard-library & minimal offline ML dependencies
├── khwarizmi/                      # Main Package Namespace
│   ├── __init__.py
│   ├── config/                     # Configuration Profiles & Hardware Tiers
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   └── tiers.py                # Prototype, Small, Edge, Advanced Tier Specs
│   ├── core/                       # Neural Sequence Modeling Core
│   │   ├── __init__.py
│   │   ├── ksc_cell.py             # Khwarizmi State Cell (KSC) Operators
│   │   ├── ksc_block.py            # Residual KSC Blocks & Normalization
│   │   └── embeddings.py           # Subword & Positional Embeddings
│   ├── memory/                     # Dual Memory Architecture
│   │   ├── __init__.py
│   │   ├── short_term.py           # Ephemeral State & Window Buffer
│   │   ├── long_term.py            # Utility-Gated Non-Parametric Key-Value Store
│   │   └── gating.py               # READ/WRITE/UPDATE/FORGET Control Nets
│   ├── routing/                    # Cognitive Router & Compute Dispatch
│   │   ├── __init__.py
│   │   ├── router.py               # Policy Network π_θ(p|x)
│   │   └── pathways.py             # Fast, Coding, Reasoning, Plan, Verif Handlers
│   ├── experts/                    # Sparse Mixture-of-Experts (MoE)
│   │   ├── __init__.py
│   │   ├── moe_layer.py            # Top-K Noisy Gating Layer
│   │   └── specialists.py          # Coding, Planning, Language Expert Subnets
│   ├── reasoning/                  # Adaptive Compute & Recurrent Reasoning
│   │   ├── __init__.py
│   │   ├── adaptive_compute.py     # ARRC Halting Gates & Ponder Loss
│   │   └── latent_reasoner.py      # Latent State Synthesis & Revision
│   ├── tools/                      # Optional Local Deterministic Tools
│   │   ├── __init__.py
│   │   ├── schemas/                # Request & Action Data Schemas
│   │   ├── project_planner/        # Legacy rafig/reasoning DAG Symbolic Engine
│   │   ├── python_brain/           # Legacy rafig/python_brain AST Static Analyzer
│   │   └── verifier.py             # Symbolic & Code Verification Controller
│   ├── agent/                      # Layered Offline Assistant Layer
│   │   ├── __init__.py
│   │   ├── input_filter.py         # Multi-lingual & Code Sanitization
│   │   └── agent_loop.py           # Orchestrator linking User -> Router -> Core -> Tools
│   ├── data/                       # Offline Datasets & Tokenizer
│   │   ├── __init__.py
│   │   ├── tokenizer/              # Custom Byte-Fallback BPE Tokenizer
│   │   └── deduplication.py        # MinHash/LSH Contamination & Duplicate Filter
│   ├── training/                   # Low-Resource Training Infrastructure
│   │   ├── __init__.py
│   │   ├── trainer.py              # Single-GPU & Colab Micro-Batch Trainer
│   │   └── losses.py               # Multi-objective Task, Balance, Ponder Losses
│   ├── evaluation/                 # Automated Benchmarking & Ablation Suite
│   │   ├── __init__.py
│   │   ├── benchmarks.py           # Intelligence, Project, Efficiency Runners
│   │   └── ablation.py             # Systematic Component Pruning & Statistical Tests
│   └── runtime/                    # Offline Inference & Edge Deployment Engine
│       ├── __init__.py
│       ├── engine.py               # Local CPU/GPU Memory-Mapped Runtime
│       └── quantization.py         # GGUF 4/5/8-bit Exporters & Activation Calibrators
└── tests/                          # Automated Verification Test Suite
    ├── test_foundation.py
    ├── test_ksc_cell.py
    ├── test_dual_memory.py
    ├── test_cognitive_router.py
    ├── test_python_brain_tool.py
    └── test_project_planner_tool.py
```

---

## 7. Risks & Failure Modes Analysis

### 7.1 Technical, Architectural, and Resource Risk Matrix

| Risk ID | Risk Category | Risk Description | Severity | Likelihood | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **R1** | **Architectural** | **KSC Recurrent State Instability:** Numerical divergence or exploding/vanishing states over ultra-long contexts ($>50{,}000$ tokens). | **High** | Medium | Enforce strict diagonal Hurwitz eigenvalue bounding ($\gamma_{\min} = 0.85$), orthogonal initialization, and FP32 state accumulation during SIMD inference. |
| **R2** | **Architectural** | **Memory Pollution & Catastrophic Forgetting:** Long-Term Memory gets flooded with trivial facts, causing high associative noise and eviction of critical project constraints. | **High** | Medium | Require dual-gate thresholding ($g_{\text{write}} > \tau_w$ and Utility $U > U_{\text{min}}$). Enforce time-decay utility pruning and explicit contradiction checking. |
| **R3** | **Architectural** | **Router Mode Collapse:** The Cognitive Router falls into a degenerate policy, routing all requests to either FAST (degrading reasoning) or REASONING (wasting latency). | **Medium** | High | Apply entropy regularization $\lambda_{\text{ent}} \mathcal{H}(\pi_\theta)$ and compute-budget cost penalties during router training. |
| **R4** | **Resource** | **MoE RAM Fragmentation on Edge Devices:** Sparse MoE layers exceed memory bandwidth or cause VRAM/RAM paging delays on 4GB consumer devices. | **High** | Medium | Conduct mandatory phase-gate ablation. If MoE latency/RAM overhead exceeds 15% without >10% quality gain, prune MoE in the Small and Edge tiers. |
| **R5** | **Execution** | **Benchmark Leakage in Training Pipeline:** Training data inadvertently containsHumanEval or project planning test sets, producing false capability signals. | **Critical** | Low | Integrate automated 13-gram MinHash/LSH overlap filtering against all validation/test benchmarks prior to pretraining. |

### 7.2 Failure Modes, Detection Signals, and Recovery Mechanisms

```
+---------------------------------------------------------------------------------------------------------+
|                                    FAILURE MODE & RECOVERY PROTOCOL                                     |
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [ FAILURE MODE 1: KSC Numerical Overflow / NaN State ]                                                 |
|    ├── Detection Signal : Maximum absolute state element |S_{t,ij}| > 1e4 or NaN token logits.          |
|    └── Recovery Action  : Dynamic state clipping to [-1e3, 1e3]; reset offending head state;            |
|                           log numerical warning to runtime diagnostics.                                 |
|                                                                                                         |
|  [ FAILURE MODE 2: Cognitive Router Infinite Loop / Trapped Recurrence ]                                |
|    ├── Detection Signal : Adaptive Compute cycle count k reaches K_max without cumulative halting.      |
|    └── Recovery Action  : Force early exit at k = K_max; emit best intermediate latent synthesis        |
|                           z_out; apply ponder cost penalty during offline replay training.              |
|                                                                                                         |
|  [ FAILURE MODE 3: Python Brain Verification Trap ]                                                     |
|    ├── Detection Signal : Generated code fails syntax/type check 3 times consecutively.                 |
|    └── Recovery Action  : Abort local repair loop; invoke Project Planner to simplify subtask           |
|                           specification; present diagnostics clearly to user.                           |
|                                                                                                         |
|  [ FAILURE MODE 4: Persistent Memory Key Collision / False Recall ]                                     |
|    ├── Detection Signal : Cosine similarity < 0.65 on retrieved top-1 memory slot during explicit read. |
|    └── Recovery Action  : Treat read as cache-miss; rely on current KSC working state; suppress         |
|                           unverified memory citations in output.                                        |
+---------------------------------------------------------------------------------------------------------+
```

---

## 8. Summary of Phase 0 Architecture Reset

With the completion of this Architecture Blueprint, Khwarizmi AI transitions from a legacy symbolic script into a rigorous, research-driven, offline-first artificial intelligence program. 

* The existing Python Brain and Reasoning components are preserved and elevated into clean, optional deterministic tools.
* The sequence modeling core is formally defined around the **Khwarizmi State Cell (KSC)**, **Dual Memory**, **Cognitive Router**, and **Adaptive Compute**.
* The project is now immediately ready to proceed to **Phase 1: Mathematical Specification & Hardware Verification**.
