# RAFIQ (Khwarizmi-AI)

RAFIQ is a completely offline AI assistant built from scratch in Python.
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
| 06 | Reasoning Engine | ⏳ not yet |
| 07 | Python Brain (`rafig/python_brain/`) | ✅ |
| 08–15 | Remaining phases | ⏳ not yet |

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

### Running the tests

```bash
python3 -m unittest discover -s tests -v
```

## Rules

- Python only, completely offline, no APIs, no cloud, no pretrained models.
- Lightweight standard-library components suitable for low-RAM computers.
- Each phase is implemented, tested, and verified before moving to the next.
