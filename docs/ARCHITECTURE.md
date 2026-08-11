# Khwarizmi AI — Architecture Specification

> **Status:** Phase 0 (Architecture Reset). This document is the authoritative
> specification for the *next* architecture. It replaces the prior "RAFIQ"
> design intent (deterministic symbolic-only) with a clean, offline-first,
> reasoning-focused neural architecture.
> **Nothing here is implemented yet.** This is the blueprint.

- **Project name:** Khwarizmi AI (neural core codename: **KSC** — *Khwarizmi State Cell*)
- **Hard requirement:** 100% offline inference. No API, cloud, Wi-Fi, or remote inference.
- **Primary objective:** Maximum intelligence and reasoning capability **per unit of compute and memory**.
- **Primary specialization:** Complex reasoning, software engineering, and **long-horizon project intelligence**, all on modest CPU / consumer GPU / edge hardware.

---

## 1. Executive Architecture Summary

Khwarizmi is a **layered, offline, reasoning-centric assistant**. The neural core is a
recurrent sequence model built around a novel **Khwarizmi State Cell (KSC)** that fuses
three complementary memory mechanisms into one mathematically coherent recurrence:

1. **Delta-rule associative memory** (precise recall of key→value associations, à la DeltaNet/Gated DeltaNet).
2. **Channel-wise decay + decoupled erase/write gates** (surprise-modulated, controllable forgetting, à la Mamba-2 / RWKV / Gated DeltaNet-2).
3. **A selective surprise/importance gate** (memorize what violates expectation, à la Titans neural memory).

On top of KSC we layer: a **Cognitive Router** (cheap learned routing that decides
whether a request needs the fast path, the coding path, or the full
reasoning+planning+verification path), **Sparse Experts** (only if ablations justify
them), **Adaptive Compute** (per-token / per-request early-exit and recurrent reasoning
loops), a **Dual Memory** system (short-term recurrent state + long-term selective
external memory), and a **Neural Reasoning** controller. All of this is wrapped by an
**Offline Agent** layer that may invoke **local deterministic tools** (the existing
symbolic Project Planner and Python Brain) *only when the router decides they are useful*.

The architecture is **experiment-driven**. Every component has a defined baseline
ablation. Components that do not earn their cost are removed or redesigned. We do **not**
preserve the old symbolic code because it once existed; we keep it only where it provides
measured value as an *offline tool*.

---

## 2. Design Principles (non-negotiable)

1. **Offline-first.** Inference dependencies are local: weights, tokenizer, runtime, memory store, agent logic, local tools. Designed offline from day one, not bolted on.
2. **Intelligence per compute > raw capability.** A 700M model that reasons well beats a 7B model that wastes tokens.
3. **Quality > previous work.** Components survive only on measured value.
4. **Modular & measurable.** Every module has a clean interface, a metric, and an ablation.
5. **No hidden CoT in the deployed assistant.** Internal reasoning state is never surfaced as the final answer.
6. **Router-gated tools.** Deterministic tools add latency; they run only when expected benefit justifies cost.
7. **Scale only on evidence.** Tiers (50M→10B+) are targets, not commitments.

---

## 3. Layered System Architecture

```
USER
  │  (prompt in EN / AR / Egyptian / Franco / code)
  ▼
KHWARIZMI OFFLINE AGENT            (orchestration, CLI, session, safety)
  │
  ▼
COGNITIVE ROUTER                   (cheap gating: path + flags)
  │  decides: fast | coding | reasoning | planning
  │  flags:   use_memory? use_reasoning_loops? use_verification? use_tools?
  ▼
KHWARIZMI NEURAL CORE (KSC stacks)
  │   - KSC recurrent layers
  │   - optional Sparse Experts (MoE FFN)
  │   - Adaptive Compute (early-exit / recurrent reasoning)
  │   - Dual Memory controller (READ/WRITE/UPDATE/FORGET)
  │   - Neural Reasoning controller
  ▼  (optional, router-gated)
OPTIONAL LOCAL TOOLS
  ├── Project Planner   (symbolic plan / task / dependency engine)
  ├── Symbolic Verification (constraints, causal checks)
  ├── Python Analysis   (AST-based code analysis, complexity, issues)
  ├── Safe Exec Sandbox (deterministic test runs)
  └── File / IO tools
  ▼
NEURAL CORE (synthesis pass)
  ▼
FINAL RESPONSE                (clean; no internal CoT exposed)
```

The neural core is **clean**: the symbolic/project/tooling system is *not* part of every
neural inference step. It is an agent-layer resource invoked by the router.

---

## 4. The Khwarizmi State Cell (KSC) — Mathematical Specification

> Label legend used throughout: **[FACT]** established result; **[RF]** research finding
> from cited literature; **[HYP]** hypothesis (untested); **[DD]** Khwarizmi design decision.

### 4.1 Motivation and lineage (what we borrow, what we reject)

- **[FACT]** Softmax attention has O(T²) compute and an unbounded KV cache that grows with sequence length (Vaswani et al., 2017).
- **[RF]** Linear/recurrent models (Mamba-2 SSD, RWKV, Gated DeltaNet, xLSTM) replace the KV cache with a fixed-size recurrent state, giving O(T·d) compute and **constant per-step memory** (Dao & Gu 2024; Peng et al. 2023; Yang et al. ICLR 2025; Beck et al. NeurIPS 2024).
- **[RF]** The *delta rule* (Sₜ = Sₜ₋₁ + βₜ kₜ ⊗ (vₜ − Sₜ₋₁ᵀ kₜ)) gives **superior associative recall** over plain additive memory because it performs error-correcting writes (DeltaNet; Gated DeltaNet).
- **[RF]** Decoupling **erase** and **write** gates (channel-wise) reduces interference when editing compressed memory (Gated DeltaNet-2, 2026).
- **[RF]** A *surprise/importance* gate that memorizes "events that violate expectations" improves long-context retention (Titans, Behrouz et al. 2025).
- **[DD]** Khwarizmi does **not** copy Mamba/RWKV/DeltaNet. We define **KSC** as a single recurrence that (a) keeps a **matrix associative memory** for precise recall, (b) applies **channel-wise decay + decoupled erase** for controllable forgetting, and (c) modulates the write by a **surprise gate**. This is a *distinct* formulation we will validate against those baselines.

### 4.2 Notation

| Symbol | Meaning | Shape |
|---|---|---|
| `xₜ` | input embedding at step t | `d_model` |
| `kₜ, qₜ, vₜ` | key / query / value | `d_k`, `d_k`, `d_v` |
| `Sₜ` | **associative memory matrix** (the core KSC state) | `d_k × d_v` |
| `λₜ` | channel-wise decay gate | `d_k` ∈ (0,1] |
| `εₜ` | channel-wise **erase** gate | `d_k` ∈ [0,1] |
| `βₜ` | scalar **write-rate** gate | scalar ∈ (0,1] |
| `σₜ` | **surprise/importance** gate | scalar ∈ [0,1] |
| `cₜ` | local convolution state (short-range mixing) | `d_model × w` |
| `ŷₜ` | cell output | `d_model` |

We use **multi-head** KSC: the state, keys, and values are split across `H` heads
(`d_k = d_model/H`, `d_v = d_model/H` or `d_v = d_model` depending on head grouping — decided in Phase 1).

### 4.3 KSC recurrent update (the heart of Khwarizmi)

For each head `h`, with input projections `W_q, W_k, W_v, W_λ, W_ε, W_β, W_σ`:

**Input projections**
```
kₜ = W_k xₜ                      qₜ = W_q xₜ                      vₜ = W_v xₜ
λₜ = sigmoid(W_λ xₜ)             εₜ = sigmoid(W_ε xₜ)
βₜ = sigmoid(w_βᵀ xₜ)            σₜ = sigmoid(w_σᵀ xₜ)
```

**1. Decay (selective temporal forgetting)** — Mamba-2 / RWKV style
```
S̃ₜ = diag(λₜ) ⊗ Sₜ₋₁
```

**2. Read**
```
rₜ = S̃ₜᵀ qₜ                      # d_v
```

**3. Erase (decoupled, channel-wise)** — Gated DeltaNet-2 style. Removes the portion of
memory associated with `kₜ`, gated per channel by `εₜ`:
```
Ŝₜ = S̃ₜ − diag(εₜ) · (S̃ₜᵀ kₜ) · kₜᵀ
```

**4. Delta error (error-correcting write target)**
```
δₜ = vₜ − Ŝₜᵀ qₜ
```

**5. Surprise-modulated write** — Titans-style gating
```
γₜ = βₜ · σₜ                     # effective write rate
Sₜ = Ŝₜ + γₜ · kₜ ⊗ δₜ           # d_k × d_v  (outer product)
```

**6. Output read (post-write retrieval)**
```
mₜ = Sₜᵀ qₜ                      # retrieved content, d_v
```

**7. Local context (short-range mixing)** — CausalConv1D, à la Gated DeltaNet / Qwen
```
x̃ₜ = Conv1D(cₜ; xₜ)             # captures local n-gram order
```

**8. Gated MLP output**
```
gₜ = sigmoid(W_g x̃ₜ)
ĥₜ = MLP(x̃ₜ)                     # the "channel mix" / FFN
ŷₜ = gₜ ⊙ ĥₜ + (1 − gₜ) ⊙ mₜ    # blend memory read with MLP (gated)
```

Optionally a **second SSM-style vector state** `hₜ = diag(λₜ)⊗hₜ₋₁ + B xₜ` can be added for
smooth compression; this is an ablation candidate, not a default.

### 4.4 Complexity & efficiency (why this is CPU-friendly)

- **Per-token per-head compute:** dominated by two `d_k × d_v` matrix–vector products
  (read and write). Cost = **O(d_k · d_v)** ≈ O(d_model²/H). Independent of sequence length `T`.
- **Recurrent state size:** `S` is `d_k × d_v` per head → **constant memory**, no KV cache.
- **Inference latency:** constant per token regardless of context length (unlike attention).
- **[HYP]** The fused delta+erase+surprise formulation yields better long-context recall
  than Mamba-2 SSD at equal state size, and better CPU throughput than attention. **To be
  proven in Phase 2–3 ablations.**
- **[DD]** Training uses chunkwise parallel scan (like SSD / Gated DeltaNet), not pure
  sequential, so we keep GPU/Colab training feasible while preserving the recurrent
  inference form.

### 4.5 Relationship to baselines (clearly distinct)

| Property | Mamba-2 SSD | RWKV-6 | Gated DeltaNet | **KSC (Khwarizmi)** |
|---|---|---|---|---|
| State type | vector `h` (diagonal A) | vector `h` (channel decay) | matrix `S` (delta) | **matrix `S` (delta)** |
| Forgetting | structured decay | channel decay `wₜ` | scalar decay `gₜ` | **channel decay `λₜ` + erase `εₜ`** |
| Write rule | linear `B x` | additive `k⊗v` | delta `β(k⊗δ)` | **surprise-gated delta `βσ(k⊗δ)`** |
| Surprise gating | no | no | no | **yes (`σₜ`)** |
| Decoupled erase/write | no | no | no (tied scalar) | **yes (εₜ vs βₜ)** |
| Local conv | yes (d_conv) | yes | yes (CausalConv1D) | yes |

KSC is **not** any of these; it is a deliberate synthesis we will benchmark head-to-head.

---

## 5. Cognitive Router

**Purpose:** decide, cheaply, how much machinery a request deserves.

- **[DD]** Router is a *tiny* network (1–2 linear layers over a compact state: the last
  hidden vector + a request-type embedding + language ID). It must cost ≪ 1% of a full forward pass.
- **Outputs:**
  - `path ∈ {FAST, CODING, REASONING, PLANNING}`
  - boolean flags: `use_memory`, `use_reasoning_loops`, `use_verification`, `use_tools`
- **Phase plan:** start as a **rule/heuristic router** (cheap, debuggable), then train a
  learned router in Phase 5/7 using policy-gradient / behavior-cloning from oracle routing
  derived from downstream task success.
- **[HYP]** A learned router improves end-to-end efficiency (tokens, latency) at equal
  quality vs. always-full-compute. Validate with the Fixed-vs-Adaptive ablation.

---

## 6. Sparse Experts (MoE) — conditional, ablation-gated

- Replace selected dense FFN layers with `E` experts; route each token to top-`k`
  (k=1 or 2) experts via a linear router + softmax.
- **Auxiliary load-balancing loss** (Switch Transformer / GLA style) to prevent expert collapse.
- **Active parameters** = `k/E × total`. For a 700M model with 8 experts, k=2 → ~175M active.
- **[HYP]** MoE improves quality at fixed active compute for coding/reasoning/planning
  specialization. **[DD]** If Phase 4 ablations show <X% gain or unstable training, **remove MoE** and keep dense KSC.

---

## 7. Adaptive Compute

- **Per-depth early-exit:** attach lightweight exit heads at intermediate layers; exit when
  confidence ≥ threshold (Early-Exit LLMs, 2026).
- **Recurrent reasoning loops:** for `REASONING`/`PLANNING` paths, allow the core to run
  additional recurrence passes over an internal thought state, gated by a **learned halt**
  probability `p_halt = sigmoid(linear(h))` (learned halting, à la ACT).
- **Confidence-based termination** for reasoning chains (CaR, EMNLP-Industry 2025; DEER, 2025).
- **[DD]** Internal reasoning state is **never** the final output. A separate synthesis pass
  produces the user-facing answer.
- **[HYP]** Adaptive compute yields equal-or-better accuracy at lower average tokens/latency. Proven via Fixed-vs-Adaptive benchmark.

---

## 8. Dual Memory

See `MEMORY.md` for the full design. Summary:

- **Short-Term State** = the KSC recurrent state `Sₜ` + `hₜ` + a small working buffer
  (recent plan/thought). This *is* immediate context and current reasoning.
- **Long-Term Memory** = an external, local store (offline: SQLite / memory-mapped /
  flat files) of facts, decisions, goals, constraints, prior failures, project state. A
  small **memory controller** (neural) decides READ / WRITE / UPDATE / FORGET per item,
  modulated by a surprise gate (Titans-inspired). Selective — we do **not** store everything.
- **Quality metric:** retrieval accuracy, retention over distractor-heavy streams, and
  long-project consistency (see `BENCHMARKS.md`).

---

## 9. Neural Reasoning & Project Intelligence

- **Neural Reasoning controller:** decomposition → plan draft → intermediate state →
  self-check → revision → synthesis. Implemented as recurrent reasoning passes + a
  controller, **not** merely longer CoT.
- **Project Intelligence (primary specialization):** the neural core *delegates* to the
  existing symbolic **Project Planner** (kept as a local tool) for dependency-aware
  task/goal/milestone management, failure recovery, and replanning — but only when the
  router requests the `PLANNING` path. The neural core supplies intent; the tool supplies
  structured, inspectable plans. This keeps project logic coherent over long horizons
  without burdening every neural step.
- **Verification:** selective. Uses deterministic local tools (Python Brain AST analysis,
  safe exec sandbox, constraint checks, symbolic consistency). Activated only when
  expected benefit justifies cost.

---

## 10. Interfaces (contracts, not implementations)

```python
# Pseudocode contracts — Phase 1 reference only.
class KSCLayer:
    state: KSCState                      # S (matrix) + optional h (vector)
    def step(self, x: Tensor, state) -> (Tensor, KSCState): ...
    def chunk(self, X: Tensor) -> Tensor: ...   # parallel training scan

class CognitiveRouter:
    def route(self, h: Tensor, req_type: Tensor, lang: int) -> RouteDecision: ...

class MemoryController:
    def read(self, query: Tensor) -> Tensor: ...
    def write(self, item: MemoryItem, surprise: float) -> None: ...
    def forget(self, key: Tensor) -> None: ...

class ReasoningController:
    def run(self, prompt_state, budget) -> ThoughtState: ...
    def synthesize(self, thought_state) -> Response: ...   # clean output

class LocalTool(Protocol):
    def name(self) -> str: ...
    def can_help(self, request) -> float: ...     # expected benefit estimate
    def run(self, request) -> ToolResult: ...
```

---

## 11. What KSC is NOT

- Not Mamba (no LTI/selective-scan scalar A alone; we use matrix delta memory).
- Not RWKV (no single-vector funnel; we keep associative matrix memory + decoupled gates).
- Not GPT/Claude/Kimi (no unbounded attention; offline-first, small-footprint).
- Not a copy of any cited work. It is a *distinct* synthesis we will validate empirically.

---

## 12. Phase 0 exit criteria (for this document set)

- [x] Architecture specified mathematically.
- [x] Research comparison complete with FACT/RF/HYP/DD labels.
- [x] Existing repo audited; each component has a decision.
- [x] 16-phase roadmap defined with per-phase success/failure criteria.
- [x] Repository structure proposed.
- [ ] Phase 1 (Mathematical Specification) begins next — see `ROADMAP.md` and the Phase 1 checklist.

> **Next action:** Begin Phase 1 — formalize KSC math into a reference (un-trained)
> implementation + numerical unit tests, and lock the baseline comparison set
> (Transformer, Mamba-2, RWKV-6, Gated DeltaNet). Do **not** train yet.
