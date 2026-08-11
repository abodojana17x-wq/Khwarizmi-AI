# Khwarizmi AI (formerly RAFIQ)

> **Architecture reset in progress.** This repository is being re-architected from a
> deterministic symbolic engine (RAFIQ) into a clean, offline-first, reasoning-focused neural
> system. **Start with [`BLUEPRINT.md`](BLUEPRINT.md)** — the Phase 0 blueprint — and the
> planning docs in [`docs/`](docs/).

RAFIQ was a completely offline AI assistant built from scratch in Python.
The long-term goal is a lightweight assistant specialized in Python that
understands Arabic, English, Egyptian Arabic, and basic Franco-Arabic —
with language understanding, memory, reasoning, code generation, and code
repair, all without any external AI services or pretrained models.

## Project status — implemented phases

| Phase | Module | Status |
|-------|--------|--------|
| 01 | Project Foundation (`rafig/`, `config.py`, `app.py`, `Rafiq`) | ✅ |
| 02 | Tokenizer (`rafig/language/tokenizer.py`) | ✅ |
| 03 | Language Understanding (`rafig/language/language_understanding.py`) | ✅ |
| 04 | Semantic Representation (`rafig/language/semantic_representation.py`) | ✅ |
| 05 | Memory System | ⏳ not yet |
| 06 | Reasoning Engine (`rafig/reasoning/`) | ✅ |
| 07 | Python Brain (`rafig/python_brain/`) | ✅ |
| 08–15 | Remaining phases | ⏳ not yet |

## Phase 06 — Reasoning Engine

The phase 06 engine creates a symbolic plan rather than executing a request or
generating code. Its output contains explicit `Goal`, `Task`, `Subtask`,
`Plan`, `Evidence`, `Hypothesis`, `Assumption`, `Constraint`, and `Result`
objects.

```python
from rafig.reasoning import ReasoningEngine

engine = ReasoningEngine()
report = engine.reason(
    "Create a Python program that reads files, groups them by extension, "
    "and moves them into folders."
)

for task in report.plan.tasks:
    print(task.description, task.dependencies)
```

The reasoning package is split into:

- `decomposition.py` — extracts actions, goals, assumptions, and constraints.
- `planner.py` — creates dependency-aware tasks and procedural subtasks.
- `evaluation.py` — compares possible actions with transparent weighted scores.
- `inference.py` — forward-chaining logical rules and causal relationships.
- `engine.py` — tracks results, checks completion, and revises failed plans.
- `models.py` — serializable state structures shared with future components.

Language and semantic analyzers are accepted through small analyzer interfaces.
The engine also accepts an optional memory adapter and retains a Python Brain
integration point, without implementing either future phase.

## Phase 07 — Python Brain

A serious Python analysis engine built from scratch using only the
standard-library `ast` module (no regex-based analysis, no execution of
analyzed code, no external tools).

### Features

- **Parsing & AST inspection** — safe parsing with structured syntax errors
- **Functions** — parameters (kinds, defaults, annotations), decorators,
  async functions, methods, docstrings, inferred return types
- **Classes** — bases, decorators, dataclass detection, methods, class
  variables, instance variables (`self.x`)
- **Variables** — locals, globals, parameters, loop variables, imports,
  with-targets, exception variables, comprehension variables
- **Imports** — plain, from-imports, aliases, star imports
- **Scopes & symbol tables** — Python-like scope-chain resolution with
  `global`/`nonlocal` handling and correct class-scope semantics
- **Control flow** — if/elif/else, for/while, with, try/except/finally,
  match, return, raise, break, continue
- **Basic type inference** — literals, annotations, builtin calls, arithmetic
  operations, methods on builtin types, comprehensions, aggregated function
  returns, instance/class attributes, exception variables
- **Issue detection** — undefined names, use-before-assignment, unreachable
  code, suspicious constructs (mutable defaults, `== None`, bare `except`,
  shadowed builtins, infinite `while True`, duplicate dict keys, empty
  bodies, ...), unused variables/imports/parameters
- **Complexity analysis** — cyclomatic complexity, nesting depth, statement
  counts, per-function and per-module
- **Code structure analysis** — module docstring, top-level statements,
  statement counts, imports, `__main__` guard and entry points
- **Structural explanation** — plain-English explanation of what a program
  does structurally, and a readable list of detected issues

### Quick usage

```python
from rafig.python_brain import PythonAnalyzer

result = PythonAnalyzer().analyze("""
def greet(name):
    print("Hello", name)
    return name
""")

print(result.diagnostics())
print(result.functions[0].returns)      # inferred return types
for issue in result.issues:             # detected problems
    print(issue.severity, issue.line, issue.message)

print(PythonAnalyzer().explain("def f(): return 1"))   # structural explanation
```

## Running the tests

```bash
python -m unittest discover -v
```

Only the Python standard library is required.

## Rules

- Python only, completely offline, no APIs, no cloud, no pretrained models.
- Lightweight standard-library components suitable for low-RAM computers.
- Each phase is implemented, tested, and verified before moving to the next.
