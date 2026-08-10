# Khwarizmi-AI / RAFIQ

RAFIQ is an offline Python project that builds its language and reasoning
components from scratch. It does not use hosted APIs or pretrained models.

## Phase 06: Reasoning Engine

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

## Run tests

```bash
python -m unittest discover -v
```

Only the Python standard library is required.
