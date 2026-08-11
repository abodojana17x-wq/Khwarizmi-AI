# Contributing to Khwarizmi AI

## Philosophy
- **Offline-first, resource-efficient, reasoning-focused.**
- **Quality > previous work. Measured improvement > theoretical claims.**
- Every component must justify its cost via ablation.

## How to contribute
1. Read `docs/BLUEPRINT.md` and the relevant `docs/*.md` for the active phase.
2. Work on the **current phase only**. Do not jump ahead (Phase 0 rule).
3. Add **numerical/unit tests** for any neural or tool change.
4. Log experiments in `experiments/` with config hash + seed + metrics.
5. Keep the inference path **network-free** (CI-enforced).

## Phase discipline
- A phase is done only when its success criteria pass.
- If a benchmark fails: stop, analyze, modify, retest — then continue.
- Old `rafig/` components are migrated to `khwarizmi/tools/` or `khwarizmi/agent/`; do not
  rewrite them prematurely. See `docs/AUDIT.md`.

## Code style
- Python 3.11+, type hints, `slots=True` dataclasses where appropriate.
- Tool layer stays stdlib-friendly; neural layer may use PyTorch.
- Document math in module docstrings (FACT/RF/HYP/DD labels where claims are made).

## Pull requests
- Target `arena/019ff0d3-khwarizmi-ai`.
- Describe the phase, the change, the tests, and the benchmark delta.
- Include ablation results when changing the neural core.
