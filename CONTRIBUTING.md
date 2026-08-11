# Contributing to Khwarizmi AI
**Document Version:** 2.0 (Phase 0 Architecture Reset)  
**Date:** 2026-08-11  

---

## 1. Welcome to Khwarizmi AI

Thank you for your interest in contributing to **Khwarizmi AI**. This project is dedicated to a fundamental research and engineering mission:

> **Maximum intelligence and reasoning capability per unit of compute and memory, operating 100% offline.**

We are building an AI assistant specialized in **software engineering, coding, complex reasoning, and long-horizon project management** that runs locally on consumer CPUs, modest RAM (<4 GB), and consumer GPUs without cloud APIs or internet dependency.

---

## 2. Core Architectural Principles (Must Read)

All contributors, researchers, and engineers must strictly adhere to the following principles when submitting pull requests or proposals:

1. **Architecture Quality > Feature Count:** Do not add bells and whistles. Every added component must integrate cleanly into the layered architecture.
2. **Measured Improvement > Theoretical Claims:** Never claim a technique (e.g., MoE, attention alternative, or memory gate) is superior without empirical benchmark verification.
3. **Do NOT Preserve Existing Code Simply Because It Exists:** If an old module makes the system slower or more complex without measurable benefit, refactor or remove it.
4. **Do NOT Create Complicated Hybrid Architectures:** Avoid combining multiple techniques just because they sound powerful. Keep interfaces modular and mathematically clean.
5. **Offline-First Guarantee:** Zero cloud, zero external APIs, zero Wi-Fi dependencies. All inference and tool calls must run standalone.

---

## 3. Strict Phase-Gate Enforcement

Khwarizmi AI follows a **16-Phase Master Roadmap** (`ROADMAP.md`).

```
+---------------------------------------------------------------------------------------------------------+
|                                        THE PHASE-GATE MANDATE                                           |
+---------------------------------------------------------------------------------------------------------+
|  1. NEVER move to the next major phase simply because the previous code "runs".                         |
|  2. A phase is complete ONLY when 100% of its defined quantitative success criteria are met.            |
|  3. IF A BENCHMARK FAILS: STOP. ANALYZE. MODIFY. RETEST. Only then continue.                             |
|  4. NEVER implement features belonging to a future phase prematurely (e.g., no MoE in Phase 1).          |
+---------------------------------------------------------------------------------------------------------+
```

---

## 4. Mandatory Ablation Protocol

Before introducing any new neural component, layer, or routing mechanism:
1. You must conduct an ablation study against the baseline as defined in `EXPERIMENTS.md`.
2. Report statistical significance ($p < 0.01$) across 3 random seeds.
3. If the component fails to exceed its quality threshold while staying within its memory/latency budget, it will be rejected or pruned.

---

## 5. Development Workflow & Local Testing

### 5.1 Setting Up Your Environment
Only the Python standard library and minimal ML dependencies (`numpy`, `torch`) are required for core development.
```bash
# Clone repository and ensure clean working tree
git status

# Run existing regression test suite (141 tests must pass)
python -m unittest discover -v
```

### 5.2 Layered Architecture Rules
When working on deterministic tools (such as `rafig/python_brain` or `rafig/reasoning`):
* **Do NOT** import PyTorch or neural core modules into deterministic tools.
* Deterministic tools must remain standalone standard-library packages callable by the **Cognitive Router** (`khwarizmi/routing/`) or **Offline Assistant Layer** (`khwarizmi/agent/`).

### 5.3 Commit & Pull Request Guidelines
* Branching: This session and all associated work are tracked on branch `arena/019ff0df-khwarizmi-ai`.
* Commit messages must clearly state the Phase number and component modified (e.g., `[Phase 01] Add mathematical KSC eigenvalue stability test`).
* Pull requests must include:
  1. A summary of the architectural impact.
  2. Test logs confirming `python -m unittest discover -v` passes 100%.
  3. Ablation metrics if modifying a neural core component.

---

## 6. Questions & Architecture Proposals

For architectural discussions or proposed changes to the mathematical specification, please reference `ARCHITECTURE.md` and `RESEARCH.md` and open a formal Issue tag `[RFC - Architecture Proposal]`.
