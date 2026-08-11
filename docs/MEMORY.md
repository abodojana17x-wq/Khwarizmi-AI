# Khwarizmi AI — Dual Memory Design

> Part of the architecture spec. Defines short-term and long-term memory for the neural core.

## 1. Principles

- **Memory is selective.** We do **not** store everything. A write happens only when
  surprise/importance exceeds a threshold (Titans-inspired).
- **Two tiers:** (a) **Short-Term State** lives inside the model (recurrent KSC state +
  working buffer); (b) **Long-Term Memory** is an external, local, offline store.
- **READ / WRITE / UPDATE / FORGET** are learned decisions made by a small memory controller.
- **Offline:** the long-term store is a local file/SQLite/memory-mapped structure. No network.
- **Measured quality:** retrieval accuracy, retention under distractors, and long-project
  consistency are tracked as first-class metrics (see `BENCHMARKS.md`).

## 2. Short-Term State (in-model)

- **Recurrent state `Sₜ`** (KSC matrix memory) + optional `hₜ` (SSM vector). This *is* the
  immediate context, current reasoning, and recent conversation — it is inherently
  token-by-token and constant-memory.
- **Working buffer:** a bounded (e.g., last `W` steps) cache of the current plan skeleton,
  active task id, and latest thought-state reference. Used by the reasoning controller.
- **Forgetting:** implicit, via the KSC decay/erase gates. No external action needed.

## 3. Long-Term Memory (external, offline)

### 3.1 What it stores
- **Facts:** durable knowledge, definitions, project decisions.
- **Goals / constraints:** active objectives and hard/soft constraints.
- **Decisions & rationale:** why something was chosen (audit trail).
- **Previous failures:** what broke, the diagnosis, the recovery (failure memory).
- **Project state:** milestones, dependencies, progress, open risks.
- **Conclusions:** validated results the system should remember.

### 3.2 Controller interface
```python
class MemoryController:
    def embed(self, item: str | dict) -> Tensor: ...
    def read(self, query: Tensor, k: int = 8) -> list[MemoryItem]: ...
    def write(self, item: MemoryItem, surprise: float) -> None: ...   # gated by surprise
    def update(self, item_id: str, patch: dict) -> None: ...
    def forget(self, item_id: str) -> None: ...
    def consolidate(self) -> None: ...   # periodic dedupe / summarize
```

### 3.3 Write policy (surprise-gated)
- Compute **surprise** `σ` of an incoming item relative to current state (e.g., prediction
  error or a learned importance score).
- Write only if `σ > θ_write`. Otherwise discard (selective memory).
- On write, optionally **UPDATE** an existing similar item instead of adding a duplicate
  (deduplication by embedding similarity).

### 3.4 Read policy
- The router/controller issues a read when `use_memory` is set (coding/planning/reasoning
  paths), or when the core's retrieval confidence is low.
- Top-`k` nearest items by embedding similarity; fused into the core via a cross-attention or
  gating step (cheap, local-window attention over `k` memories — O(k·d), not O(T·d)).

### 3.5 Forget policy
- **Time/usage decay:** items unread for long or never used decay in importance.
- **Contradiction:** new high-confidence info that contradicts an old item triggers UPDATE or
  FORGET of the stale item.
- **Capacity:** when over budget, evict lowest-importance items (importance = f(surprise,
  access frequency, recency, confirmation count)).

## 4. Storage backend (offline)
- **Prototype / dev:** SQLite + pickle/JSON for items; `numpy`/`sqlitevec` for embeddings.
- **Edge:** memory-mapped files; optional lightweight vector index (flat or IVF) kept local.
- **No external vector DB, no cloud.** Everything ships with the model.

## 5. How it connects to Project Intelligence
- The external **Project Planner** tool persists its plans/tasks to Long-Term Memory.
- On a new session, the agent loads relevant project state from memory → coherence over
  long horizons (days/weeks of development) without re-deriving everything.

## 6. Metrics (definition of "memory quality")
- **Retrieval accuracy @k** on held-out facts.
- **Retention** after `N` distractor items (does the right fact survive?).
- **Forgetting rate** of stale/contradicted items.
- **Long-project consistency:** do later decisions respect earlier constraints?
- **Write precision:** fraction of writes that are later read-and-useful.

These are exercised in `BENCHMARKS.md` and the memory ablations (Baseline vs Baseline+Memory).
