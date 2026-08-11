# Khwarizmi AI — Existing Repository Audit & Old Component Decisions

> **Audit basis:** full source read of the `rafig/` package (RAFIQ), `main.py`, `tests/`,
> `requirements.txt`, `README.md`. **141 unit tests pass.** Critical finding: the current
> repo is a **deterministic, rule-based, symbolic engine with ZERO neural components, weights,
> or training**. It is offline by design (stdlib only). That is a strength to preserve *as a
> tool layer*, not as the neural core.

## 1. Component inventory

| # | Component | LOC | Purpose (current) |
|---|---|---|---|
| 1 | `rafig/rafig.py` (Rafiq) | 90 | Foundation skeleton: settings, logger, diagnostics, start/run/shutdown |
| 2 | `rafig/config.py` | 57 | Settings dataclass + env-based config + path bootstrap |
| 3 | `rafig/paths.py` | 16 | Create standard project dirs |
| 4 | `rafig/app.py` + `main.py` | 23 | CLI entry point (starts Rafiq) |
| 5 | `rafig/language/tokenizer.py` | 118 | **Char-level** tokenizer (vocab of ASCII + Arabic blocks) |
| 6 | `rafig/language/language_understanding.py` | 157 | Rule-based lang ID (AR/EN/Franco/code) + intent keywords |
| 7 | `rafig/language/semantic_representation.py` | 91 | Rule-based intent→action/object/constraints mapping |
| 8 | `rafig/reasoning/engine.py` | 517 | Orchestrator: reason(), plan, record_result, revise_plan |
| 9 | `rafig/reasoning/models.py` | 411 | Dataclasses: Goal/Task/Subtask/Plan/Constraint/Evidence/... |
| 10 | `rafig/reasoning/decomposition.py` | 259 | Action-term extraction, goal/constraint/assumption mining |
| 11 | `rafig/reasoning/planner.py` | 295 | Builds dependency-aware tasks/subtasks + candidate actions |
| 12 | `rafig/reasoning/evaluation.py` | 59 | Weighted multi-criteria action scoring |
| 13 | `rafig/reasoning/inference.py` | 213 | Forward-chaining Horn rules + causal reasoner |
| 14 | `rafig/python_brain/*` (7 files) | ~3350 | AST-based Python analysis: parse, model, types, complexity, issues, explain |
| 15 | `tests/*` (12 files) | ~1450 | Unit tests for all of the above |
| 16 | `requirements.txt` | 4 | Empty (stdlib only) |

## 2. Audit & decisions

Legend: **KEEP** · **REFACTOR** · **REPLACE** · **EXTERNAL TOOL** · **REMOVE**

### #1 `rafig/rafig.py` (Rafiq core) — **REFACTOR → AGENT/RUNTIME BOOTSTRAP**
- **Why:** It is a placeholder skeleton. The diagnostics/logger/offline-mode pattern is
  useful, but the name "neural core" it implies is false — there is no network.
- **Future location:** `khwarizmi/runtime/` (engine bootstrap) or `khwarizmi/agent/`.
- **Action:** Keep diagnostics + logging + offline flag; rewrite as the offline agent/runtime
  entrypoint, not a fake "core".

### #2 `rafig/config.py` — **KEEP (REFACTOR)**
- **Why:** Clean settings pattern; offline-first already default. Extend with model tier,
  device, quantization, memory-store path, router policy.
- **Future location:** `khwarizmi/config.py`.

### #3 `rafig/paths.py` — **KEEP**
- **Why:** Directory management is needed for local memory store, logs, models, tools.
- **Future location:** `khwarizmi/common/paths.py`.

### #4 `rafig/app.py` + `main.py` — **REFACTOR**
- **Why:** Entry point pattern is fine; must become the **offline agent CLI** that loads the
  neural core + router + tools and serves requests.
- **Future location:** `khwarizmi/cli.py` / `main.py`.

### #5 `rafig/language/tokenizer.py` — **REPLACE**
- **Why (critical):** A **character-level** tokenizer is fundamentally unsuitable for a neural
  LLM. It wastes enormous compute (one token per char), has a tiny effective vocabulary, and
  cannot represent subwords/code idioms efficiently. Char-level was appropriate for the old
  symbolic system, not for KSC.
- **Future location:** `khwarizmi/neural/tokenizer/` — a trained **BPE/Unigram** tokenizer
  (SentencePiece / HuggingFace `tokenizers`) covering EN + AR + Egyptian + code, with
  special tokens for memory/tool/route control.
- **Action:** Archive the old tokenizer tests; write new tokenizer tests for the trained vocab.

### #6 `rafig/language/language_understanding.py` — **MOVE → ROUTER FEATURE (REPLACE heavy logic)**
- **Why:** Rule-based keyword intent/sentiment is superseded by the neural core for
  understanding. **However**, the multilingual awareness (Arabic / Egyptian / Franco / English /
  code-like detection) is valuable as a *cheap routing feature* and as *data/prompt guidance*.
- **Future location:** Language-ID becomes a small feature fed to the Cognitive Router; the
  heavy semantic mapping is replaced by neural outputs.
- **Action:** Extract the language-detection heuristics into a lightweight utility; delete the
  rule-based intent mapping.

### #7 `rafig/language/semantic_representation.py` — **REPLACE**
- **Why:** Hard-coded `repair/create → code_repair/code_generation` mapping is brittle and
  narrow. The *structured schema idea* (intent/action/object/constraints) is good and should
  be **produced by the neural core**, not by regex.
- **Future location:** Optional thin neural post-processor; not on the critical path.

### #8 `rafig/reasoning/*` (engine, models, decomposition, planner, evaluation, inference) — **EXTERNAL TOOL (Project Planner + Symbolic Verification)**
- **Why (KEY INSIGHT):** This is the **strongest existing asset** and maps *exactly* onto the
  "Project Planner" and "Symbolic Verification" tools in the layered architecture. It produces
  inspectable, dependency-aware plans, tasks, subtasks, constraints, causal relations, and
  failure-recovery revisions. It is deterministic, offline, and stdlib-only.
- **But:** It must **NOT** run on every neural inference step. It is invoked by the agent
  layer only when the router selects the `PLANNING` path (or verification).
- **Future location:** `khwarizmi/tools/project_planner/` and `khwarizmi/tools/symbolic_verify/`.
- **Action:** Keep all code + tests; **REFACTOR** to (a) consume neural outputs (intent,
  goals) instead of regenerating them via regex, (b) be callable as a tool with a
  `can_help()`/`run()` interface, (c) persist plans to the long-term memory store.

### #9 `rafig/python_brain/*` — **KEEP + EXTERNAL TOOL (Symbolic Code Verification)**
- **Why:** This is a serious, correct, offline AST analysis engine (parse, scope resolution,
  type inference, complexity, issue detection, explanation). It is a **perfect deterministic
  verifier** for the coding path — exactly what "selective verification" needs.
- **Future location:** `khwarizmi/tools/python_analysis/`.
- **Action:** Keep + extend: add safe sandboxed execution (deterministic test runs), more
  linters/analyzers, and a `can_help()`/`run()` tool interface. This is a core offline
  verification asset — do not delete.

### #10 `tests/*` — **KEEP + EXTEND**
- **Why:** 141 passing tests protect the tool layer (planner, python_brain). They become
  regression tests for the external tools.
- **Action:** Keep; add neural-core unit tests (math correctness, recurrence vs reference,
  router, memory controller) in Phases 1–3.

### #11 `requirements.txt` — **REPLACE**
- **Why:** Neural training needs PyTorch (or JAX/MLX), tokenizers, numpy, datasets. Keep
  stdlib-only for the *tool* layer; add a separate `requirements-neural.txt` / pinned env.
- **Future location:** `requirements.txt` (runtime/tools, stdlib+optional) +
  `requirements-dev.txt` (training stack).

## 3. Decision summary table

| Component | Decision | Future location |
|---|---|---|
| Rafiq core (`rafig.py`) | REFACTOR | `khwarizmi/runtime` / `agent` |
| config | KEEP | `khwarizmi/config.py` |
| paths | KEEP | `khwarizmi/common/paths.py` |
| app/main | REFACTOR | `khwarizmi/cli.py` |
| tokenizer | REPLACE | `khwarizmi/neural/tokenizer` |
| language_understanding | MOVE→router feature | `khwarizmi/agent/langid.py` |
| semantic_representation | REPLACE | neural post-processor (optional) |
| reasoning/* | EXTERNAL TOOL | `khwarizmi/tools/project_planner`, `symbolic_verify` |
| python_brain/* | KEEP + EXTERNAL TOOL | `khwarizmi/tools/python_analysis` |
| tests | KEEP + EXTEND | `tests/` (tools) + `tests/neural/` |
| requirements | REPLACE (split) | `requirements*.txt` |

## 4. What we explicitly do NOT preserve as-is
- The **char-level tokenizer** (replaced).
- The **rule-based intent/semantic mapping** as the understanding mechanism (replaced by neural).
- The assumption that the `rafig` package *is* the AI (it is now the **tool/agent layer**).

## 5. What we explicitly KEEP (because it provides measured value)
- The **symbolic Project Planner** (decomposition/planner/inference) → offline planning tool.
- The **Python Brain** → offline code verification tool.
- **Deterministic, stdlib-only, offline** property of the whole package → preserved as a
  principle for the tool layer.

> **Rule honored:** Quality > previous work. Components survive on value, not on existence.
> The two largest, most valuable modules (planner, python_brain) are kept — as *tools*, not
> as the neural core.
